#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities for datasets, losses, evaluation, and KM plotting."""

import os
import random
import pickle
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class STADSurvivalDataset(Dataset):
    """TCGA survival dataset backed by PASG and MGAS hypergraphs."""

    def __init__(
        self,
        pasg_hyperedge_dir,
        mgas_hyperedge_dir,
        feature_dir,
        clinical_file,
        sample_limit=None,
        use_pasg: bool = True,
        use_mgas: bool = True,
    ):
        self.pasg_hyperedge_dir = pasg_hyperedge_dir
        self.mgas_hyperedge_dir = mgas_hyperedge_dir
        self.feature_dir = feature_dir
        self.use_pasg = bool(use_pasg)
        self.use_mgas = bool(use_mgas)

        clinical_df = pd.read_csv(clinical_file, sep="\t")

        all_samples = []
        if os.path.isdir(feature_dir):
            for ft_file in sorted(os.listdir(feature_dir)):
                if not ft_file.endswith("_fts.npy"):
                    continue

                sample_name = ft_file[:-8]
                pasg_dir = os.path.join(pasg_hyperedge_dir, sample_name + "_fts")
                mgas_dir = os.path.join(mgas_hyperedge_dir, sample_name)

                ok_pasg = (not self.use_pasg) or os.path.isdir(pasg_dir)
                ok_mgas = (not self.use_mgas) or os.path.isdir(mgas_dir)
                if ok_pasg and ok_mgas:
                    all_samples.append(sample_name)

        print(f"Found {len(all_samples)} samples with all required hypergraph files in {feature_dir}")

        if sample_limit is not None:
            all_samples = all_samples[:sample_limit]
            print(f"[Debug mode] Using the first {sample_limit} samples only")
        else:
            print(f"[Full mode] Using all {len(all_samples)} samples")

        self.valid_samples = []
        self.survival_times = []
        self.censors = []
        self.stages = []

        print("Matching samples to clinical records...")
        for sample_name in all_samples:
            parts = sample_name.split("_")
            if len(parts) < 2:
                continue

            tcga_id = "-".join(parts[1].split("-")[:3])
            match = clinical_df[clinical_df["cases.submitter_id"] == tcga_id]
            if match.shape[0] == 0:
                continue

            vital_status = match["demographic.vital_status"].iloc[0]
            stage_str = match["diagnoses.ajcc_pathologic_stage"].iloc[0]

            def parse_survival_days(days_value):
                if pd.isna(days_value):
                    return None
                days_str = str(days_value).strip()
                if days_str in {"'--", "--", ""} or days_str.lower() == "na":
                    return None
                try:
                    days = float(days_str)
                except (ValueError, TypeError):
                    return None
                if days <= 0 or not np.isfinite(days):
                    return None
                return int(days)

            if vital_status == "Alive":
                survival_days = parse_survival_days(match["diagnoses.days_to_last_follow_up"].iloc[0])
                if survival_days is None:
                    continue
                censor = 0
            else:
                survival_days = parse_survival_days(match["demographic.days_to_death"].iloc[0])
                if survival_days is None:
                    continue
                censor = 1

            if stage_str == "'--" or pd.isna(stage_str):
                continue

            if "IV" in stage_str or "X" in stage_str:
                stage = 4
            elif "III" in stage_str:
                stage = 3
            elif "II" in stage_str:
                stage = 2
            elif "I" in stage_str:
                stage = 1
            else:
                stage = 0

            self.valid_samples.append(sample_name)
            self.survival_times.append(survival_days)
            self.censors.append(censor)
            self.stages.append(stage)

        print(f"Matched {len(self.valid_samples)} valid samples")
        print(f"  Events (censor=1): {sum(self.censors)}")
        print(f"  Censored (censor=0): {len(self.censors) - sum(self.censors)}")

        T = np.array(self.survival_times, dtype=np.float32)
        T_max, T_min = T.max(), T.min()

        if np.any(T <= 0):
            invalid_count = int(np.sum(T <= 0))
            raise ValueError(f"Found {invalid_count} invalid survival times (<= 0).")

        if T_max == T_min:
            self.true_risks = np.zeros(len(T), dtype=np.float32)
        else:
            self.true_risks = (T_min * (T_max - T)) / (T * (T_max - T_min))
            if np.any(np.isnan(self.true_risks)) or np.any(np.isinf(self.true_risks)):
                nan_count = int(np.sum(np.isnan(self.true_risks)))
                inf_count = int(np.sum(np.isinf(self.true_risks)))
                raise ValueError(
                    f"true_risks contains invalid values: {nan_count} NaNs and {inf_count} Infs."
                )

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        sample_name = self.valid_samples[idx]
        pasg_sample_dir = os.path.join(self.pasg_hyperedge_dir, sample_name + "_fts")
        mgas_sample_dir = os.path.join(self.mgas_hyperedge_dir, sample_name)

        feat_file = os.path.join(self.feature_dir, sample_name + "_fts.npy")
        features = np.load(feat_file)
        if len(features.shape) == 1:
            features = features.reshape(-1, 512)
        features = torch.from_numpy(features).float()

        if self.use_pasg:
            pasg_hyperedge_file = os.path.join(pasg_sample_dir, "hyperedges.pkl")
            with open(pasg_hyperedge_file, "rb") as f:
                pasg_hyperedges = pickle.load(f)
        else:
            pasg_hyperedges = []

        if self.use_mgas:
            mgas_hyperedge_file = os.path.join(mgas_sample_dir, "hyperedges.pkl")
            with open(mgas_hyperedge_file, "rb") as f:
                mgas_hyperedges = pickle.load(f)
        else:
            mgas_hyperedges = []

        return {
            "features": features,
            "pasg_hyperedges": pasg_hyperedges,
            "mgas_hyperedges": mgas_hyperedges,
            "survival_time": self.survival_times[idx],
            "censor": self.censors[idx],
            "true_risk": self.true_risks[idx],
            "stage": self.stages[idx],
            "sample_name": sample_name,
        }


def collate_fn_survival(batch):
    """Return a list because each sample is a separate graph."""
    return batch


class MSELoss(nn.Module):
    """Mean squared error loss."""

    def forward(self, pred_risks, true_risks):
        return F.mse_loss(pred_risks, true_risks)


class NLLLoss(nn.Module):
    """Negative Cox partial log-likelihood."""

    def forward(self, pred_risks, survival_times, censors):
        sorted_indices = torch.argsort(survival_times, descending=True)
        pred_risks = pred_risks[sorted_indices]
        censors = censors[sorted_indices]

        if torch.sum(censors) == 0:
            return torch.tensor(0.0, device=pred_risks.device, requires_grad=True)

        hazard_ratio = torch.exp(pred_risks)
        log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0) + 1e-7)
        uncensored_likelihood = pred_risks - log_risk
        censored_likelihood = uncensored_likelihood * censors
        return -torch.sum(censored_likelihood) / (torch.sum(censors) + 1e-7)


class CoxLoss(nn.Module):
    """Vectorized Cox partial likelihood loss."""

    def forward(self, pred_risks, survival_times, censors):
        device = pred_risks.device

        if torch.sum(censors) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        sorted_indices = torch.argsort(survival_times, descending=True)
        pred_risks_sorted = pred_risks[sorted_indices]
        censors_sorted = censors[sorted_indices]
        survival_times_sorted = survival_times[sorted_indices]

        exp_risks = torch.exp(pred_risks_sorted)
        time_matrix = survival_times_sorted.unsqueeze(1) >= survival_times_sorted.unsqueeze(0)
        cumulative_risks = torch.matmul(time_matrix.float(), exp_risks.unsqueeze(1)).squeeze(1)
        log_cumulative_risks = torch.log(cumulative_risks + 1e-10)

        event_mask = censors_sorted == 1
        if torch.sum(event_mask) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        contributions = pred_risks_sorted - log_cumulative_risks
        event_contributions = contributions[event_mask]
        return -torch.sum(event_contributions) / torch.sum(event_mask).float()


class BCRLoss(nn.Module):
    """Bayesian Concordance Readjust Loss."""

    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.mse_loss = MSELoss()

    def forward(self, pred_risks, true_risks, wsi_features, W, survival_times, censors):
        B = pred_risks.size(0)
        mse = self.mse_loss(pred_risks, true_risks)
        concordance_loss_list = []

        for i in range(B):
            for j in range(B):
                if i == j:
                    continue
                if survival_times[i] > survival_times[j] and (censors[i] == 1 or censors[j] == 1):
                    feature_diff = wsi_features[i] - wsi_features[j]
                    score = torch.matmul(W, feature_diff.unsqueeze(1)).squeeze()
                    prob = torch.sigmoid(score)
                    concordance_loss_list.append(-torch.log(prob + 1e-7))

        if concordance_loss_list:
            concordance_loss = torch.stack(concordance_loss_list).mean()
        else:
            concordance_loss = torch.tensor(0.0, device=pred_risks.device)

        total_loss = concordance_loss + self.alpha * mse
        return total_loss, concordance_loss, mse


def compute_cindex(survival_times, risk_scores, censors):
    """Compute the C-index when valid event information is available."""
    times = np.array(survival_times)
    risks = np.array(risk_scores)
    cens = np.array(censors)

    if np.sum(cens) == 0:
        print("Warning: no event samples are available, so the C-index cannot be computed.")
        return None

    if len(np.unique(times)) < 2:
        print("Warning: all survival times are identical, so the C-index cannot be computed.")
        return None

    try:
        return concordance_index(times, -risks, cens)
    except Exception as e:
        print(f"Warning: failed to compute the C-index: {e}")
        return None


def ensure_dir(path: str):
    """Create a directory if it does not already exist."""
    if path:
        os.makedirs(path, exist_ok=True)


def save_predictions_csv(sample_names, times, events, risks, out_csv: str, fold: Optional[int] = None):
    """Save per-sample predictions to CSV."""
    ensure_dir(os.path.dirname(out_csv))
    df = pd.DataFrame({
        "sample_name": list(sample_names),
        "time": list(times),
        "event": list(events),
        "risk": list(risks),
    })
    if fold is not None:
        df["fold"] = fold
    df.to_csv(out_csv, index=False)
    return df


def km_plot_from_arrays(times, events, risks, out_png: str, title: str | None = None,
                        split: str = "median", q: float = 0.5,
                        ci: float = 0.8, ci_alpha: float = 0.25, show_ci: bool = True,
                        truncate_tail: bool = False, min_at_risk_ratio: float = 0.10, min_at_risk_abs: int = 5,
                        optimize_cutoff: bool = False, q_min: float = 0.2, q_max: float = 0.8, q_step: float = 0.01,
                        min_group_ratio: float = 0.2, min_group_abs: int = 10, random_state: int | None = None,
                        xlim: tuple[float, float] | None = None, ylim: tuple[float, float] | None = (0.0, 1.0),
                        x_major_step: float | None = None, y_major_step: float | None = 0.2,
                        y_ticks: list[float] | tuple[float, ...] | None = None,
                        legend_fontsize: int = 12, pvalue_fontsize: int = 14, tick_labelsize: int = 12,
                        legend_outside: bool = False):
    """Draw a KM curve from arrays and save it as PNG."""
    times = np.asarray(times)
    events = np.asarray(events)
    risks = np.asarray(risks)

    def _logrank_p(threshold: float):
        high_mask = risks >= threshold
        low_mask = ~high_mask
        n_high = int(high_mask.sum())
        n_low = int(low_mask.sum())
        min_n = max(int(np.ceil(min_group_ratio * len(risks))), int(min_group_abs))
        if n_high < min_n or n_low < min_n:
            return None, n_high, n_low
        try:
            lr = logrank_test(times[high_mask], times[low_mask],
                              event_observed_A=events[high_mask], event_observed_B=events[low_mask])
            return float(lr.p_value), n_high, n_low
        except Exception:
            return None, n_high, n_low

    if optimize_cutoff:
        qs = np.arange(float(q_min), float(q_max) + 1e-12, float(q_step))
        best = {"p_value": None, "q": None, "threshold": None, "n_high": None, "n_low": None}

        if random_state is not None:
            rng = np.random.default_rng(int(random_state))
            rng.shuffle(qs)

        for qq in qs:
            threshold_candidate = float(np.quantile(risks, float(qq)))
            p_value, n_high, n_low = _logrank_p(threshold_candidate)
            if p_value is None:
                continue
            if best["p_value"] is None or p_value < best["p_value"]:
                best.update({
                    "p_value": p_value,
                    "q": float(qq),
                    "threshold": threshold_candidate,
                    "n_high": int(n_high),
                    "n_low": int(n_low),
                })

        if best["threshold"] is None:
            print("Warning: no valid optimized cutoff satisfied the group-size constraints. Falling back to the median split.")
            threshold = float(np.median(risks))
        else:
            threshold = float(best["threshold"])
            split = "optimal"
            q = float(best["q"]) if best["q"] is not None else q
    else:
        threshold = float(np.quantile(risks, q)) if split == "quantile" else float(np.median(risks))

    high_mask = risks >= threshold
    low_mask = risks < threshold
    n_high = int(high_mask.sum())
    n_low = int(low_mask.sum())

    if n_high == 0 or n_low == 0:
        print("Warning: one risk group is empty, so the KM curve cannot be drawn.")
        return {"p_value": None, "n_high": n_high, "n_low": n_low, "threshold": float(threshold), "t_trunc": None}

    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()
    dh, eh = times[high_mask], events[high_mask]
    dl, el = times[low_mask], events[low_mask]

    alpha = max(1.0 - float(ci), 0.0)
    kmf_high.fit(durations=dh, event_observed=eh, label="High risk", alpha=alpha)
    kmf_low.fit(durations=dl, event_observed=el, label="Low risk", alpha=alpha)

    try:
        lr = logrank_test(dh, dl, event_observed_A=eh, event_observed_B=el)
        p_value = float(lr.p_value)
    except Exception as e:
        print(f"Warning: log-rank test failed: {e}")
        p_value = None

    t_trunc = None
    if truncate_tail:
        threshold_high = max(int(np.ceil(min_at_risk_ratio * n_high)), int(min_at_risk_abs))
        threshold_low = max(int(np.ceil(min_at_risk_ratio * n_low)), int(min_at_risk_abs))
        try:
            et_h = kmf_high.event_table
            et_l = kmf_low.event_table
            t_h = et_h.index[et_h["at_risk"] >= threshold_high]
            t_l = et_l.index[et_l["at_risk"] >= threshold_low]
            if len(t_h) > 0 and len(t_l) > 0:
                t_trunc = float(min(t_h.max(), t_l.max()))
        except Exception:
            t_trunc = None

    ensure_dir(os.path.dirname(out_png))
    plt.figure(figsize=(6, 4.5), dpi=150)
    ax = plt.subplot(111)
    kmf_high.plot(ci_show=bool(show_ci), ax=ax, color="#d62728", ci_alpha=ci_alpha, linewidth=4)
    kmf_low.plot(ci_show=bool(show_ci), ax=ax, color="#1f77b4", ci_alpha=ci_alpha, linewidth=4)

    if legend_outside:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=2, frameon=False, fontsize=legend_fontsize)
    else:
        legend = ax.legend(loc="upper right", fontsize=legend_fontsize, frameon=True)
        if legend is not None:
            legend.get_frame().set_alpha(0.85)

    ax.set_xlabel("")
    ax.set_ylabel("")
    if title is not None and str(title).strip() != "":
        ax.set_title(str(title))

    if p_value is not None:
        ax.text(0.02, 0.02, f"p = {p_value:.2e}", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=pvalue_fontsize)

    if xlim is not None:
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
    elif t_trunc is not None and np.isfinite(t_trunc) and t_trunc > 0:
        ax.set_xlim(0, t_trunc)

    if ylim is not None:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))

    if x_major_step is not None and x_major_step > 0:
        xmin, xmax = ax.get_xlim()
        xticks = np.arange(0.0 if xmin < 0 else xmin, xmax + 1e-12, float(x_major_step))
        if len(xticks) > 1:
            ax.set_xticks(xticks)

    if y_ticks is not None and len(y_ticks) > 0:
        ax.set_yticks([float(v) for v in y_ticks])
    elif y_major_step is not None and y_major_step > 0:
        ymin, ymax = ax.get_ylim()
        yticks = np.arange(ymin, ymax + 1e-12, float(y_major_step))
        if len(yticks) > 1:
            ax.set_yticks(yticks)

    ax.tick_params(axis="both", which="major", labelsize=tick_labelsize)
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.85, linestyle="-", linewidth=1.0)
    ax.grid(True, which="minor", alpha=0.55, linestyle="--", linewidth=0.8)

    if legend_outside:
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    else:
        plt.tight_layout()

    plt.savefig(out_png)
    plt.close()

    return {
        "p_value": p_value,
        "n_high": n_high,
        "n_low": n_low,
        "threshold": float(threshold),
        "t_trunc": t_trunc,
    }


def km_plot_from_csv(pred_csv: str, out_png: str, title: str | None = None,
                     split: str = "median", q: float = 0.5):
    """Load predictions from CSV and draw a KM curve."""
    df = pd.read_csv(pred_csv)
    times = df["time"].values
    events = df["event"].values
    risks = df["risk"].values
    return km_plot_from_arrays(times, events, risks, out_png, title=title, split=split, q=q)
