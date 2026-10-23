"""GMM prototype discovery and sparse cosine-based semantic incidence."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from hypermil.models.hypermil import SparseIncidence
from .gmm import RegularizedGMM, Float64RegularizedGMM


def configure_determinism():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def new_model(settings, device, use_float64=False):
    cls = Float64RegularizedGMM if use_float64 else RegularizedGMM
    return cls(K=int(settings['max_components']), covariance=settings['covariance'],
               concentration=float(settings['concentration']),
               min_component_mass=float(settings['min_component_mass']),
               max_iter=int(settings['em_max_iter']), tol=float(settings['em_tolerance']),
               seed=int(settings['seed']), device=device)


@dataclass
class PASGResult:
    incidence: SparseIncidence
    prototypes: torch.Tensor
    original_ids: torch.Tensor
    pi: torch.Tensor
    effective_support: torch.Tensor
    hard_support: torch.Tensor
    metadata: dict


def build_pasg(features, slide_id, settings, device='cpu'):
    configure_determinism()
    started = time.perf_counter()
    model = new_model(settings, device)
    use_float64 = False
    try:
        model.fit_predict_proba(features)
    except FloatingPointError:
        use_float64 = True
        model = new_model(settings, device, use_float64=True)
        model.fit_predict_proba(features)
    dtype = torch.float64 if use_float64 else torch.float32
    x = torch.as_tensor(np.asarray(features), dtype=dtype, device=device)
    gamma = torch.softmax(model._log_prob(x), dim=1)
    if not torch.isfinite(gamma).all():
        raise FloatingPointError('GMM responsibilities are non-finite')
    support = gamma.sum(0)
    hard = torch.bincount(gamma.argmax(1), minlength=model.K)
    retained = torch.nonzero(support >= float(settings['support_threshold'])).flatten()
    top_m = int(settings['top_m'])
    if retained.numel() < top_m:
        raise RuntimeError('Too few supported prototypes for the configured top_m')
    cosine = F.normalize(x, p=2, dim=1, eps=1e-12) @ F.normalize(
        model.means[retained], p=2, dim=1, eps=1e-12).T
    scores, positions = torch.topk(cosine, k=top_m, dim=1)
    weights = torch.softmax(scores / float(settings['temperature']), dim=1)
    positive = weights > 0
    used = torch.unique(positions[positive], sorted=True)
    mapping = torch.full((len(retained),), -1, dtype=torch.long, device=device)
    mapping[used] = torch.arange(len(used), device=device)
    rows = torch.arange(len(x), device=device)[:, None].expand_as(positions)[positive]
    cols = mapping[positions[positive]]
    incidence = SparseIncidence(rows.cpu(), cols.cpu(), weights[positive].float().cpu(), (len(x), len(used)))
    original_ids = retained[used]
    meta = dict(slide_id=slide_id, n_patches=len(x), Kmax=model.K, Keffective=len(used),
                tau=float(settings['temperature']), top_m=top_m,
                support_threshold=float(settings['support_threshold']),
                prior_type='posterior_mean_style_pseudocount_smoothing',
                alpha_k=float(settings['concentration']),
                GMM_converged=bool(model.converged), GMM_iterations=int(model.iterations_run),
                GMM_dtype='float64' if use_float64 else 'float32',
                em_max_iter=int(settings['em_max_iter']), em_tolerance=float(settings['em_tolerance']),
                total_seconds=time.perf_counter()-started, build_version='pasg-1')
    return PASGResult(incidence, model.means[original_ids].detach().cpu(), original_ids.cpu(),
                      model.weights[original_ids].detach().cpu(), support[original_ids].detach().cpu(),
                      hard[original_ids].detach().cpu(), meta)


def save_pasg(result, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = result.metadata
    np.savez_compressed(path, slide_id=np.asarray(meta['slide_id']), n_patches=np.int64(meta['n_patches']),
        Kmax=np.int64(meta['Kmax']), Keffective=np.int64(meta['Keffective']),
        prototype_vectors=result.prototypes.numpy(), prototype_original_ids=result.original_ids.numpy(),
        prototype_final_ids=np.arange(meta['Keffective'], dtype=np.int64), pi_k=result.pi.numpy(),
        effective_support_k=result.effective_support.numpy(), hard_support_k=result.hard_support.numpy(),
        GMM_converged=np.bool_(meta['GMM_converged']), GMM_iterations=np.int64(meta['GMM_iterations']),
        GMM_dtype=np.asarray(meta['GMM_dtype']), incidence_row=result.incidence.rows.numpy(),
        incidence_col=result.incidence.cols.numpy(), incidence_weight=result.incidence.values.numpy(),
        incidence_shape=np.asarray(result.incidence.shape, dtype=np.int64), tau=np.float64(meta['tau']),
        top_m=np.int64(meta['top_m']), support_threshold=np.float64(meta['support_threshold']),
        prior_type=np.asarray(meta['prior_type']), alpha_k=np.float64(meta['alpha_k']),
        build_version=np.asarray(meta['build_version']), metadata_json=np.asarray(json.dumps(meta, sort_keys=True)))
