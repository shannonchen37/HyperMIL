#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model definitions for HyperMIL."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperedgeConv(nn.Module):
    """Dense-incidence hypergraph convolution."""

    def __init__(self, in_channels, out_channels, use_bn=True, dropout=0.3):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.dropout = nn.Dropout(dropout)

    def _build_H_matrix(self, hyperedges, num_nodes, device):
        """Build the incidence matrix H with shape (N, E)."""
        num_hyperedges = len(hyperedges)
        H = torch.zeros(num_nodes, num_hyperedges, device=device)
        node_degree = torch.zeros(num_nodes, device=device)

        for e_idx, hyperedge in enumerate(hyperedges):
            if len(hyperedge) == 0:
                continue

            if isinstance(hyperedge, (list, tuple)):
                node_indices = torch.tensor(hyperedge, device=device, dtype=torch.long)
            else:
                node_indices = torch.tensor(list(hyperedge), device=device, dtype=torch.long)

            weight = 1.0 / len(hyperedge)
            H[node_indices, e_idx] = weight
            node_degree[node_indices] += 1.0

        return H, node_degree

    def forward(self, X, hyperedges):
        """Aggregate node features through hyperedges and project them."""
        num_nodes = X.size(0)
        device = X.device
        H, node_degree = self._build_H_matrix(hyperedges, num_nodes, device)

        X_hyperedge = torch.mm(H.t(), X)
        X_agg = torch.mm(H, X_hyperedge)
        X_agg = X_agg / node_degree.clamp(min=1.0).unsqueeze(1)

        X_out = self.fc(X_agg)
        if self.bn is not None:
            X_out = self.bn(X_out)
        return self.dropout(X_out)


class CrossAttentionAggregator(nn.Module):
    """Aggregate node features into a slide-level representation."""

    def __init__(self, feat_dim, num_heads=4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, feat_dim))
        self.mha = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.ln = nn.LayerNorm(feat_dim)

    def forward(self, node_features, return_attn=False):
        kv = node_features.unsqueeze(0)
        query = self.query.expand(kv.size(0), -1, -1)
        attn_output, attn_weights = self.mha(query, kv, kv, need_weights=True, average_attn_weights=True)
        wsi_feature = self.ln(attn_output + query).squeeze(1)

        if return_attn:
            return wsi_feature, attn_weights.squeeze(0).squeeze(0)
        return wsi_feature


class DualHypergraphModel(nn.Module):
    """HyperMIL with PASG hyperedges, MGAS hyperedges, and cross-attention pooling."""

    def __init__(self, in_channels=512, hid_channels=256, num_heads=4, dropout=0.3):
        super().__init__()
        self.hid_channels = hid_channels
        self.in_channels = in_channels

        self.pasg_conv = HyperedgeConv(in_channels, hid_channels, use_bn=True, dropout=dropout)
        self.mgas_conv_with_pasg = HyperedgeConv(hid_channels, hid_channels, use_bn=True, dropout=dropout)
        self.mgas_conv_without_pasg = HyperedgeConv(in_channels, hid_channels, use_bn=True, dropout=dropout)
        self.identity_proj = nn.Linear(in_channels, hid_channels)
        self.aggregator = CrossAttentionAggregator(hid_channels, num_heads)
        self.regressor = nn.Sequential(
            nn.Linear(hid_channels, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self.bcr_weight = nn.Parameter(torch.randn(1, hid_channels))

    def forward(self, X, pasg_hyperedges, mgas_hyperedges, return_attn=False, return_node_feats: bool = False):
        use_pasg = len(pasg_hyperedges) > 0
        use_mgas = len(mgas_hyperedges) > 0

        if use_pasg:
            X = F.relu(self.pasg_conv(X, pasg_hyperedges))

        if use_mgas:
            X = (
                self.mgas_conv_with_pasg(X, mgas_hyperedges)
                if use_pasg else
                self.mgas_conv_without_pasg(X, mgas_hyperedges)
            )
            X = F.relu(X)
        elif not use_pasg:
            X = F.relu(self.identity_proj(X))

        if return_attn:
            wsi_feature, attn_weights = self.aggregator(X, return_attn=True)
        else:
            wsi_feature = self.aggregator(X, return_attn=False)

        risk_score = self.regressor(wsi_feature).squeeze()

        if return_attn and return_node_feats:
            return risk_score, wsi_feature, attn_weights, X
        if return_attn:
            return risk_score, wsi_feature, attn_weights
        if return_node_feats:
            return risk_score, wsi_feature, X
        return risk_score, wsi_feature
