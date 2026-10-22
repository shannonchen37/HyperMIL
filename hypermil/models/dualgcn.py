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
from torch_geometric.nn import GCNConv
class GCNBlock(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, dropout: float=0.3):
        super().__init__()
        self.conv = GCNConv(in_dim, out_dim, add_self_loops=True, normalize=True)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv(x, edge_index)
        x = self.norm(x)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)
        return x

class GatedAttentionReadout(nn.Module):

    def __init__(self, in_dim: int, att_dim: int=128):
        super().__init__()
        self.U = nn.Linear(in_dim, att_dim, bias=True)
        self.V = nn.Linear(in_dim, att_dim, bias=True)
        self.w = nn.Linear(att_dim, 1, bias=False)

    def forward(self, G: torch.Tensor) -> torch.Tensor:
        """G: [N, D] 节点表示 -> 返回 z: [D]
        a_i = w^T( tanh(VG) ⊙ sigmoid(UG) ), α = softmax(a)
        z = Σ α_i g_i
        """
        A = torch.tanh(self.V(G)) * torch.sigmoid(self.U(G))
        a = self.w(A).squeeze(-1)
        alpha = torch.softmax(a, dim=0).unsqueeze(1)
        z = (alpha * G).sum(dim=0)
        return z

class AttentionJKNet(nn.Module):

    def __init__(self, in_dim: int=512, hidden_dim: int=256, num_layers: int=3, att_dim: int=128, dropout: float=0.3, jk_mode: str='sum'):
        super().__init__()
        assert num_layers >= 1
        self.num_layers = num_layers
        self.jk_mode = jk_mode.lower()
        layers: List[nn.Module] = []
        dim_in = in_dim
        for _ in range(num_layers):
            layers.append(GCNBlock(dim_in, hidden_dim, dropout))
            dim_in = hidden_dim
        self.layers = nn.ModuleList(layers)
        if self.jk_mode == 'concat':
            emb_dim = hidden_dim * num_layers
        else:
            emb_dim = hidden_dim
        self.emb_dim = emb_dim
        self.readout = GatedAttentionReadout(emb_dim, att_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h_list: List[torch.Tensor] = []
        for layer in self.layers:
            x = layer(x, edge_index)
            h_list.append(x)
        if self.jk_mode == 'concat':
            G = torch.cat(h_list, dim=1)
        else:
            G = torch.stack(h_list, dim=0).sum(dim=0)
        z = self.readout(G)
        return z

class DualGCNMIL(nn.Module):

    def __init__(self, in_dim: int=512, hidden_dim: int=256, num_layers: int=3, att_dim: int=128, dropout: float=0.3, jk_mode: str='sum', graph_root: str='KNN', dataset_name: str='THCA', k_coord: Optional[int]=8, k_feat: Optional[int]=6):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.att_dim = att_dim
        self.dropout = dropout
        self.jk_mode = jk_mode
        self.graph_root = graph_root
        self.dataset_name = dataset_name.upper()
        self.k_coord = k_coord
        self.k_feat = k_feat
        self.branch_coord = AttentionJKNet(in_dim, hidden_dim, num_layers, att_dim, dropout, jk_mode)
        self.branch_feat = AttentionJKNet(in_dim, hidden_dim, num_layers, att_dim, dropout, jk_mode)
        emb_dim = self.branch_coord.emb_dim
        self.risk_head = nn.Linear(2 * emb_dim, 1)
        self.bcr_weight = nn.Parameter(torch.randn(1, 2 * emb_dim))

    def _find_graph_file(self, subdir: str, basename: str, k_opt: Optional[int]) -> Optional[str]:
        """在 KNN/{cancer}/{feature|spatial} 下匹配图文件。
        - spatial: {basename}_knn_spatial_k{k}.npz
        - feature: {basename}_knn_feature_k{k}.npz
        严格匹配请求的 k；禁止回退到其他 k 的缓存。
        """
        cancer_dir = f'TCGA-{self.dataset_name}'
        base_dir = os.path.join(self.graph_root, cancer_dir, subdir)
        if k_opt is not None:
            if subdir == 'spatial':
                pattern = os.path.join(base_dir, f'{basename}_knn_spatial_k{k_opt}.npz')
            else:
                pattern = os.path.join(base_dir, f'{basename}_knn_feature_k{k_opt}.npz')
            files = glob.glob(pattern)
            if files:
                return files[0]
        return None

    def _load_edge_index(self, basename: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        coord_path = self._find_graph_file('spatial', basename, self.k_coord)
        feat_path = self._find_graph_file('feature', basename, self.k_feat)
        if coord_path is None or feat_path is None:
            raise FileNotFoundError(f'找不到图文件: coord={coord_path}, feat={feat_path}, basename={basename}')
        coord_npz = np.load(coord_path, allow_pickle=False)
        feat_npz = np.load(feat_path, allow_pickle=False)
        if int(coord_npz['k']) != int(self.k_coord):
            raise ValueError(f'spatial cache k mismatch: {coord_path}')
        if int(feat_npz['k']) != int(self.k_feat):
            raise ValueError(f'feature cache k mismatch: {feat_path}')
        ei_c = torch.as_tensor(coord_npz['edge_index'], dtype=torch.long, device=device)
        ei_f = torch.as_tensor(feat_npz['edge_index'], dtype=torch.long, device=device)
        return (ei_c, ei_f)

    def forward(self, x: torch.Tensor, sample_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: [N, in_dim] 节点特征；sample_name: 过滤列表中的 basename（不带后缀）"""
        device = x.device
        ei_c, ei_f = self._load_edge_index(sample_name, device)
        z_c = self.branch_coord(x, ei_c)
        z_f = self.branch_feat(x, ei_f)
        h_final = torch.cat([z_f, z_c], dim=0)
        risk = self.risk_head(h_final.unsqueeze(0)).squeeze(0).squeeze(-1)
        return (risk, h_final)
