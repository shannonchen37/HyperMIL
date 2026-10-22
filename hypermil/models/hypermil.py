from __future__ import annotations
import math, types, json, csv, os, random
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence, List, Optional, Tuple
import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
@dataclass
class SparseIncidence:
    rows: torch.Tensor
    cols: torch.Tensor
    values: torch.Tensor
    shape: tuple[int, int]

    def __post_init__(self):
        self.rows = torch.as_tensor(self.rows, dtype=torch.long)
        self.cols = torch.as_tensor(self.cols, dtype=torch.long)
        self.values = torch.as_tensor(self.values, dtype=torch.float32)
        if not self.rows.numel() == self.cols.numel() == self.values.numel():
            raise ValueError('rows, cols and values must have equal length')
        if self.values.numel() and (not torch.isfinite(self.values).all()):
            raise ValueError('Incidence contains non-finite values')

    @classmethod
    def from_hyperedges(cls, hyperedges, n_nodes: int, values: float=1.0):
        rows, cols, vals = ([], [], [])
        for edge_id, edge in enumerate(hyperedges):
            for node in sorted(set((int(x) for x in edge))):
                rows.append(node)
                cols.append(edge_id)
                vals.append(values)
        return cls(rows, cols, vals, (n_nodes, len(hyperedges)))

    @classmethod
    def from_dense(cls, dense: torch.Tensor, atol: float=0.0):
        dense = torch.as_tensor(dense)
        idx = torch.nonzero(dense.abs() > atol, as_tuple=False)
        if idx.numel() == 0:
            return cls([], [], [], tuple(dense.shape))
        return cls(idx[:, 0], idx[:, 1], dense[idx[:, 0], idx[:, 1]], tuple(dense.shape))

    def dense(self, device=None, dtype=torch.float32) -> torch.Tensor:
        out = torch.zeros(self.shape, device=device, dtype=dtype)
        if self.values.numel():
            out[self.rows.to(device), self.cols.to(device)] = self.values.to(device=device, dtype=dtype)
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, rows=self.rows.numpy(), cols=self.cols.numpy(), values=self.values.numpy(), shape=np.asarray(self.shape, dtype=np.int64))

    @classmethod
    def load(cls, path: str | Path):
        data = np.load(path)
        return cls(data['rows'], data['cols'], data['values'], tuple((int(x) for x in data['shape'])))

    def statistics(self) -> dict:
        counts = torch.zeros(self.shape[0], dtype=torch.long)
        if self.rows.numel():
            counts.scatter_add_(0, self.rows, torch.ones_like(self.rows))
        covered = counts > 0
        return {'coverage': float(covered.float().mean().item()) if self.shape[0] else 1.0, 'mean_memberships_per_node': float(counts.float().mean().item()) if self.shape[0] else 0.0, 'max_memberships_per_node': int(counts.max().item()) if counts.numel() else 0}
class SparseWeightedHGNN(nn.Module):
    """Standard weighted HGNN propagation using COO reductions only."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool=True, eps: float=1e-08):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.eps = eps

    def propagate(self, x: torch.Tensor, incidence: SparseIncidence, edge_weights: torch.Tensor | None=None) -> torch.Tensor:
        rows = incidence.rows.to(x.device)
        cols = incidence.cols.to(x.device)
        values = incidence.values.to(device=x.device, dtype=x.dtype)
        n_nodes, n_edges = incidence.shape
        if n_nodes != x.shape[0]:
            raise ValueError(f'node mismatch: incidence={n_nodes}, features={x.shape[0]}')
        weights = torch.ones(n_edges, device=x.device, dtype=x.dtype) if edge_weights is None else edge_weights.to(device=x.device, dtype=x.dtype)
        edge_degree = torch.zeros(n_edges, device=x.device, dtype=x.dtype)
        edge_degree.index_add_(0, cols, values)
        vertex_degree = torch.zeros(n_nodes, device=x.device, dtype=x.dtype)
        vertex_degree.index_add_(0, rows, values * weights[cols])
        inv_vertex = vertex_degree.clamp_min(self.eps).rsqrt()
        edge_features = torch.zeros(n_edges, x.shape[1], device=x.device, dtype=x.dtype)
        edge_features.index_add_(0, cols, values[:, None] * inv_vertex[rows, None] * x[rows])
        edge_features = edge_features * (weights / edge_degree.clamp_min(self.eps))[:, None]
        propagated = torch.zeros_like(x)
        propagated.index_add_(0, rows, values[:, None] * edge_features[cols])
        return inv_vertex[:, None] * propagated

    def forward(self, x: torch.Tensor, incidence: SparseIncidence, edge_weights: torch.Tensor | None=None) -> torch.Tensor:
        return self.linear(self.propagate(x, incidence, edge_weights))

class HGNNBranch(nn.Module):

    def __init__(self, input_dim: int=512, hidden_dim: int=256, depth: int=3, dropout: float=0.0, residual: bool=False, layer_norm: bool=False):
        super().__init__()
        if depth < 1:
            raise ValueError('HGNN depth must be positive')
        dims = [input_dim] + [hidden_dim] * depth
        self.layers = nn.ModuleList((SparseWeightedHGNN(dims[i], dims[i + 1]) for i in range(depth)))
        self.activation = nn.ReLU()
        self.hidden_dropout = nn.Dropout(dropout)
        self.residual = residual
        self.residuals = nn.ModuleList((nn.Identity() if dims[i] == dims[i + 1] else nn.Linear(dims[i], dims[i + 1]) for i in range(depth))) if residual else None
        self.norms = nn.ModuleList((nn.LayerNorm(hidden_dim) for _ in range(depth))) if layer_norm else None

    def forward(self, x: torch.Tensor, incidence: SparseIncidence) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            previous = x
            x = self.hidden_dropout(self.activation(layer(x, incidence)))
            if self.residual:
                x = self.norms[index](self.residuals[index](previous) + x)
        return x

class VectorGatedFusion(nn.Module):

    def __init__(self, dim: int=256):
        super().__init__()
        self.gate = nn.Linear(2 * dim, dim)

    def forward(self, spatial: torch.Tensor, semantic: torch.Tensor):
        if spatial.shape != semantic.shape:
            raise ValueError(f'branch shape mismatch: {spatial.shape} != {semantic.shape}')
        gate = torch.sigmoid(self.gate(torch.cat((spatial, semantic), dim=-1)))
        fused = gate * spatial + (1.0 - gate) * semantic
        return (fused, gate)

class GatedMILAttention(nn.Module):

    def __init__(self, dim: int=256):
        super().__init__()
        self.v = nn.Linear(dim, dim)
        self.u = nn.Linear(dim, dim)
        self.w = nn.Linear(dim, 1)

    def forward(self, patches: torch.Tensor):
        logits = self.w(torch.tanh(self.v(patches)) * torch.sigmoid(self.u(patches))).squeeze(-1)
        attention = torch.softmax(logits, dim=0)
        slide = torch.sum(attention[:, None] * patches, dim=0)
        return (slide, attention)

@dataclass
class SlideOutput:
    embedding: torch.Tensor
    spatial: torch.Tensor
    semantic: torch.Tensor
    fused: torch.Tensor
    gate: torch.Tensor
    attention: torch.Tensor | None

class HyperMILV2Gated(nn.Module):
    TRUE_PARALLEL_BRANCHES = True
    SHARED_BRANCH_WEIGHTS = False
    CASCADE_ARCHITECTURE = False
    GATE_TYPE = 'vector'
    GATE_ACTIVATION = 'sigmoid'

    def __init__(self, input_dim: int=512, hidden_dim: int=256, depth: int=3, dropout: float=0.0, residual: bool=False, hgnn_layer_norm: bool=False, branch_layer_norm: bool=False, pre_attention_layer_norm: bool=False):
        super().__init__()
        self.mgas_branch = HGNNBranch(input_dim, hidden_dim, depth, dropout, residual, hgnn_layer_norm)
        self.pasg_branch = HGNNBranch(input_dim, hidden_dim, depth, dropout, residual, hgnn_layer_norm)
        self.mgas_norm = nn.LayerNorm(hidden_dim) if branch_layer_norm else nn.Identity()
        self.pasg_norm = nn.LayerNorm(hidden_dim) if branch_layer_norm else nn.Identity()
        self.fusion = VectorGatedFusion(hidden_dim)
        self.pre_attention_norm = nn.LayerNorm(hidden_dim) if pre_attention_layer_norm else nn.Identity()
        self.attention = GatedMILAttention(hidden_dim)
        self.risk_head = nn.Linear(hidden_dim, 1)

    def encode_slide(self, features: torch.Tensor, mgas: SparseIncidence, pasg: SparseIncidence) -> SlideOutput:
        spatial = self.mgas_norm(self.mgas_branch(features, mgas))
        semantic = self.pasg_norm(self.pasg_branch(features, pasg))
        fused, gate = self.fusion(spatial, semantic)
        attention_input = self.pre_attention_norm(fused)
        embedding, attention = self.attention(attention_input)
        return SlideOutput(embedding, spatial, semantic, attention_input, gate, attention)

    def patient_risk(self, slides: list[tuple[torch.Tensor, SparseIncidence, SparseIncidence]], observer=None) -> torch.Tensor:
        outputs = [self.encode_slide(*slide) for slide in slides]
        if observer is not None:
            for output in outputs:
                observer(output)
        patient_embedding = torch.stack([x.embedding for x in outputs]).mean(dim=0)
        return self.risk_head(patient_embedding).reshape(())
@dataclass
class PreparedIncidence:
    rows: torch.Tensor
    cols: torch.Tensor
    values: torch.Tensor
    inv_vertex: torch.Tensor
    edge_scale: torch.Tensor
    shape: tuple[int, int]

def fast_propagate(layer, x: torch.Tensor, incidence: PreparedIncidence, edge_weights=None):
    if edge_weights is not None:
        raise ValueError('frozen official engine only supports unit MGAS / stored PASG incidence weights')
    rows, cols, values = (incidence.rows, incidence.cols, incidence.values)
    n_nodes, n_edges = incidence.shape
    if n_nodes != x.shape[0]:
        raise ValueError(f'node mismatch: incidence={n_nodes}, features={x.shape[0]}')
    edge_features = torch.zeros(n_edges, x.shape[1], device=x.device, dtype=x.dtype)
    edge_features.index_add_(0, cols, values[:, None] * incidence.inv_vertex[rows, None] * x[rows])
    edge_features = edge_features * incidence.edge_scale[:, None]
    propagated = torch.zeros_like(x)
    propagated.index_add_(0, rows, values[:, None] * edge_features[cols])
    return incidence.inv_vertex[:, None] * propagated

def make_gated(device):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = HyperMILV2Gated(input_dim=512, hidden_dim=256, depth=2, dropout=.2,
        residual=False, hgnn_layer_norm=False, branch_layer_norm=False,
        pre_attention_layer_norm=False).to(device)
    for branch in (model.mgas_branch, model.pasg_branch):
        for layer in branch.layers:
            layer.propagate = types.MethodType(fast_propagate, layer)
    return model
def prepare(raw, device):
    rows, cols = raw.rows.to(device), raw.cols.to(device)
    values = raw.values.to(device=device, dtype=torch.float32)
    n, m = raw.shape
    de = torch.zeros(m, device=device).index_add_(0, cols, values)
    dv = torch.zeros(n, device=device).index_add_(0, rows, values * torch.ones(m, device=device)[cols])
    return PreparedIncidence(rows, cols, values, dv.clamp_min(1e-8).rsqrt(),
        torch.ones(m, device=device)/de.clamp_min(1e-8), raw.shape)
def late_fusion_encode_slide(self, features, mgas, pasg):
    structural = self.mgas_norm(self.mgas_branch(features, mgas))
    semantic = self.pasg_norm(self.pasg_branch(features, pasg))
    z_structural, attention_structural = self.attention_s(self.pre_attention_norm(structural))
    z_semantic, attention_semantic = self.attention_f(self.pre_attention_norm(semantic))
    mixing = torch.sigmoid(self.wsi_fusion(torch.cat((z_structural, z_semantic), dim=-1)))
    embedding = mixing * z_structural + (1.0 - mixing) * z_semantic
    return types.SimpleNamespace(embedding=embedding, spatial=structural, semantic=semantic, fused=embedding.unsqueeze(0), gate=mixing, attention=attention_structural, structural_attention=attention_structural, semantic_attention=attention_semantic)

def make_hypermil(device):
    model = make_gated(device)
    model.attention_s = model.attention
    del model.attention
    del model.fusion
    model.attention_f = type(model.attention_s)(256).to(device)
    model.wsi_fusion = nn.Linear(512, 1).to(device)
    model.encode_slide = types.MethodType(late_fusion_encode_slide, model)
    count = sum((parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    if count != 658692:
        raise RuntimeError(f'MODEL_MISMATCH: expected 658692 parameters, got {count}')
    if not isinstance(model.risk_head, nn.Linear) or model.risk_head.in_features != 256 or model.risk_head.out_features != 1:
        raise RuntimeError('MODEL_MISMATCH: risk head is not Linear(256,1)')
    if hasattr(model, 'fusion') or hasattr(model, 'attention'):
        raise RuntimeError('MODEL_MISMATCH: P0 patch fusion/single attention remains active')
    return model
