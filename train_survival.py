#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Training script for HyperMIL survival prediction."""

import os
import argparse
import warnings
import json
from datetime import datetime
import glob

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from utils import (
    set_seed,
    STADSurvivalDataset,
    collate_fn_survival,
    MSELoss,
    NLLLoss,
    CoxLoss,
    BCRLoss,
    compute_cindex,
    save_predictions_csv,
    km_plot_from_arrays,
)
from models import DualHypergraphModel

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def train_epoch(model, dataloader, optimizer, loss_fn, loss_type="mse", device=device, batch_size=8):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    batch_count = 0

    batch_risks = []
    batch_times = []
    batch_censors = []
    batch_true_risks = []
    batch_wsi_features = []

    for batch in tqdm(dataloader, desc="Training"):
        for sample in batch:
            features = sample["features"].to(device)
            pasg_hyperedges = sample["pasg_hyperedges"]
            mgas_hyperedges = sample["mgas_hyperedges"]

            risk_score, wsi_feature = model(features, pasg_hyperedges, mgas_hyperedges)

            if sample["survival_time"] <= 0:
                continue
            if np.isnan(sample["true_risk"]) or np.isinf(sample["true_risk"]):
                continue

            batch_risks.append(risk_score)
            batch_times.append(sample["survival_time"])
            batch_censors.append(sample["censor"])
            batch_true_risks.append(sample["true_risk"])
            batch_wsi_features.append(wsi_feature.squeeze(0))

            if len(batch_risks) >= batch_size:
                loss = _compute_loss(
                    model,
                    loss_fn,
                    loss_type,
                    batch_risks,
                    batch_times,
                    batch_censors,
                    batch_true_risks,
                    batch_wsi_features,
                    device,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1
                batch_risks, batch_times, batch_censors = [], [], []
                batch_true_risks, batch_wsi_features = [], []

    if batch_risks:
        loss = _compute_loss(
            model,
            loss_fn,
            loss_type,
            batch_risks,
            batch_times,
            batch_censors,
            batch_true_risks,
            batch_wsi_features,
            device,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batch_count += 1

    return total_loss / max(batch_count, 1)


def _compute_loss(model, loss_fn, loss_type, batch_risks, batch_times, batch_censors,
                  batch_true_risks, batch_wsi_features, device):
    pred_risks = torch.stack(batch_risks)

    if loss_type == "mse":
        true_risks = torch.tensor(batch_true_risks, dtype=torch.float32, device=device)
        return loss_fn(pred_risks, true_risks)

    survival_times = torch.tensor(batch_times, dtype=torch.float32, device=device)
    censors = torch.tensor(batch_censors, dtype=torch.float32, device=device)

    if loss_type == "nll":
        return loss_fn(pred_risks, survival_times, censors)

    if loss_type == "cox":
        return loss_fn(pred_risks, survival_times, censors)

    true_risks = torch.tensor(batch_true_risks, dtype=torch.float32, device=device)
    wsi_features = torch.stack(batch_wsi_features)
    W = model.bcr_weight
    loss, _, _ = loss_fn(pred_risks, true_risks, wsi_features, W, survival_times, censors)
    return loss


def evaluate(model, dataloader, device=device):
    """Evaluate the model with the C-index."""
    model.eval()
    all_risks, all_times, all_censors = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            for sample in batch:
                features = sample["features"].to(device)
                pasg_hyperedges = sample["pasg_hyperedges"]
                mgas_hyperedges = sample["mgas_hyperedges"]

                risk_score, _ = model(features, pasg_hyperedges, mgas_hyperedges)
                all_risks.append(risk_score.cpu().item())
                all_times.append(sample["survival_time"])
                all_censors.append(sample["censor"])

    cindex = compute_cindex(all_times, all_risks, all_censors)
    return cindex if cindex is not None else 0.0


def collect_predictions(model, dataloader, device=device):
    """Collect per-sample predictions."""
    model.eval()
    names, times, events, risks = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            for sample in batch:
                features = sample["features"].to(device)
                pasg_hyperedges = sample["pasg_hyperedges"]
                mgas_hyperedges = sample["mgas_hyperedges"]
                risk_score, _ = model(features, pasg_hyperedges, mgas_hyperedges)
                names.append(sample.get("sample_name", ""))
                times.append(sample["survival_time"])
                events.append(sample["censor"])
                risks.append(float(risk_score.cpu().item()))

    return names, times, events, risks


def main():
    parser = argparse.ArgumentParser(description="Train HyperMIL for TCGA survival prediction.")
    parser.add_argument("--dataset", type=str, default="STAD",
                        choices=["STAD", "THCA", "CHOL", "LIHC"],
                        help="Dataset name.")
    parser.add_argument("--loss_type", type=str, default="mse",
                        choices=["mse", "nll", "cox", "bcr"],
                        help="Loss function.")
    parser.add_argument("--eval_only", action="store_true",
                        help="Evaluation mode. Load saved checkpoints and reproduce validation results.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Logical batch size.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs.")
    parser.add_argument("--base_dir", type=str, default=PROJECT_ROOT,
                        help="Project root directory.")
    args = parser.parse_args()

    set_seed(args.seed)

    dataset_name = args.dataset.upper()
    base_dir = os.path.abspath(args.base_dir)
    pasg_hyperedge_dir = os.path.join(base_dir, f"PASG_Hyperedge/{dataset_name}")
    mgas_hyperedge_dir = os.path.join(base_dir, f"MGAS_Hyperedge/{dataset_name}")
    feature_dir = os.path.join(base_dir, f"WSI_data/TCGA-{dataset_name}/patch_ft")
    clinical_file = os.path.join(base_dir, f"WSI_data/TCGA-{dataset_name}/clinical.tsv")

    output_dir = os.path.join(base_dir, "best_models", dataset_name)
    vis_dir = os.path.join(base_dir, "vis_res", "KM", dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    loss_type = args.loss_type

    print("=" * 80)
    print(f"TCGA-{dataset_name} survival prediction with dual hypergraphs (PASG + MGAS)")
    print("=" * 80)
    print(f"Dataset: {dataset_name}")
    print(f"Loss: {loss_type.upper()}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {epochs}")
    print(f"Seed: {args.seed}")
    print(f"Project root: {base_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Mode: {'evaluation' if args.eval_only else 'training'}")
    print("=" * 80)

    dataset = STADSurvivalDataset(
        pasg_hyperedge_dir,
        mgas_hyperedge_dir,
        feature_dir,
        clinical_file,
        sample_limit=None,
    )

    if len(dataset) == 0:
        print("Error: no valid samples were found.")
        return

    num_events = sum(dataset.censors[i] for i in range(len(dataset)))
    if num_events == 0:
        print("Error: the dataset does not contain any event samples.")
        return
    print(f"Detected {num_events} event samples out of {len(dataset)} total samples.")

    if loss_type == "mse":
        loss_fn = MSELoss()
    elif loss_type == "nll":
        loss_fn = NLLLoss()
    elif loss_type == "cox":
        loss_fn = CoxLoss()
    else:
        loss_fn = BCRLoss(alpha=0.5)

    censor_labels = [dataset.censors[i] for i in range(len(dataset))]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    fold_splits = list(skf.split(range(len(dataset)), censor_labels))

    fold_results = []
    fold_best_epochs = []

    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        print(f"\n{'=' * 80}")
        print(f"Fold {fold + 1}/5")
        print(f"{'=' * 80}")

        train_events = sum(dataset.censors[i] for i in train_idx)
        val_events = sum(dataset.censors[i] for i in val_idx)
        print(f"Train split: {len(train_idx)} samples, {train_events} events")
        print(f"Validation split: {len(val_idx)} samples, {val_events} events")

        if val_events == 0:
            print(f"Warning: fold {fold + 1} has no validation events and will be skipped.")
            continue

        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=1, shuffle=True, collate_fn=collate_fn_survival)
        val_loader = DataLoader(val_subset, batch_size=1, shuffle=False, collate_fn=collate_fn_survival)

        model = DualHypergraphModel(
            in_channels=512,
            hid_channels=256,
            num_heads=4,
            dropout=0.3,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        lr_str = str(lr).replace(".", "_")
        model_path = os.path.join(output_dir, f"best_model_{loss_type}_fold{fold + 1}_bs{batch_size}_lr{lr_str}.pth")

        if args.eval_only:
            if not os.path.exists(model_path):
                print(f"Warning: checkpoint not found, skipping fold: {model_path}")
                continue

            print(f"Loading checkpoint: {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            val_cindex = evaluate(model, val_loader, device)
            print(f"Validation C-index: {val_cindex:.4f}")

            tag = f"{loss_type}_fold{fold + 1}_bs{batch_size}_lr{lr_str}"
            names, times, events, risks = collect_predictions(model, val_loader, device)
            csv_path = os.path.join(output_dir, f"val_preds_{tag}_eval.csv")
            save_predictions_csv(names, times, events, risks, csv_path, fold=fold + 1)
            print(f"Saved per-fold validation CSV for OOF aggregation: {csv_path}")

            fold_results.append(val_cindex)
            fold_best_epochs.append(0)
            continue

        best_cindex = 0.0
        best_epoch = 0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            train_loss = train_epoch(model, train_loader, optimizer, loss_fn, loss_type, device, batch_size)
            val_cindex = evaluate(model, val_loader, device)

            print(f"  Train loss: {train_loss:.4f}")
            print(f"  Validation C-index: {val_cindex:.4f}")

            if val_cindex > best_cindex:
                best_cindex = val_cindex
                best_epoch = epoch + 1
                torch.save(model.state_dict(), model_path)
                print(f"  Saved best checkpoint (C-index: {best_cindex:.4f})")

            model.train()

        fold_results.append(best_cindex)
        fold_best_epochs.append(best_epoch)

        tag = f"{loss_type}_fold{fold + 1}_bs{batch_size}_lr{lr_str}"
        print(f"Loading best checkpoint for prediction export: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        names, times, events, risks = collect_predictions(model, val_loader, device)
        csv_path = os.path.join(output_dir, f"val_preds_{tag}.csv")
        save_predictions_csv(names, times, events, risks, csv_path, fold=fold + 1)
        print(f"Saved per-fold validation CSV for OOF aggregation: {csv_path}")

        print(f"\n{'=' * 80}")
        print(f"Finished fold {fold + 1}")
        print(f"{'=' * 80}")
        print(f"Best validation C-index: {best_cindex:.4f}")
        print(f"Best epoch: {best_epoch}")
        print(f"{'=' * 80}")

    results = {
        "dataset": dataset_name,
        "loss_type": loss_type,
        "batch_size": batch_size,
        "learning_rate": lr,
        "epochs": epochs,
        "seed": args.seed,
        "num_samples": len(dataset),
        "num_events": num_events,
        "fold_results": fold_results,
        "fold_best_epochs": fold_best_epochs,
        "mean_cindex": float(np.mean(fold_results)) if fold_results else 0.0,
        "std_cindex": float(np.std(fold_results)) if fold_results else 0.0,
        "test_cindex_summary": f"{np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}" if fold_results else "N/A",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    lr_str = str(lr).replace(".", "_")
    results_file = os.path.join(output_dir, f"results_{loss_type}_bs{batch_size}_lr{lr_str}.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 80}")
    print("5-fold cross-validation summary")
    print(f"{'=' * 80}")
    for i, cindex in enumerate(fold_results):
        print(f"Fold {i + 1} validation C-index: {cindex:.4f}")
    if fold_results:
        mean_cindex = np.mean(fold_results)
        std_cindex = np.std(fold_results)
        print(f"{'=' * 80}")
        print(f"All folds validation C-index = {mean_cindex:.4f} ± {std_cindex:.4f}")
        print(f"{'=' * 80}")

    try:
        suffix = "_eval" if args.eval_only else ""
        pattern = os.path.join(output_dir, f"val_preds_{loss_type}_fold*_bs{batch_size}_lr{lr_str}{suffix}.csv")
        csv_list = sorted(glob.glob(pattern))
        if csv_list:
            df_all = pd.concat([pd.read_csv(p) for p in csv_list], ignore_index=True)
            all_csv = os.path.join(output_dir, f"all_val_preds_{loss_type}_bs{batch_size}_lr{lr_str}{suffix}.csv")
            df_all.to_csv(all_csv, index=False)
            print(f"Saved aggregated OOF validation CSV: {all_csv} (n={len(df_all)})")

            times = df_all["time"].values
            events = df_all["event"].values
            risks = df_all["risk"].values
            km_png = os.path.join(vis_dir, f"km_{loss_type}_allfolds_bs{batch_size}_lr{lr_str}{suffix}.png")
            info = km_plot_from_arrays(times, events, risks, km_png, title=f"{dataset_name} {loss_type} OOF", ci=0.8)
            print(
                f"Saved overall KM curve: {km_png}  "
                f"p={info.get('p_value')}  "
                f"high_n={info.get('n_high')}  "
                f"low_n={info.get('n_low')}  "
                f"threshold={info.get('threshold')}"
            )
        else:
            print(f"Note: no per-fold CSV files matched: {pattern}")
    except Exception as e:
        print(f"Warning: failed to aggregate OOF predictions or draw the KM curve: {e}")

    print(f"Results saved to: {results_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
