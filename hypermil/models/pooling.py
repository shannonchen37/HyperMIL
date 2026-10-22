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
class RiskHead(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x).reshape(())

class MeanPoolCox(nn.Module):
    embedding_dim = 512

    def __init__(self):
        super().__init__()
        self.risk_head = RiskHead(self.embedding_dim)

    def encode_slide(self, features, **_):
        return features.mean(0)

class GatedAttention(nn.Module):

    def __init__(self, dim: int, attention_dim: int):
        super().__init__()
        self.v = nn.Linear(dim, attention_dim)
        self.u = nn.Linear(dim, attention_dim)
        self.w = nn.Linear(attention_dim, 1)

    def forward(self, x):
        logits = self.w(torch.tanh(self.v(x)) * torch.sigmoid(self.u(x))).reshape(-1)
        weights = torch.softmax(logits, dim=0)
        return ((weights[:, None] * x).sum(0), weights)

class ABMILCox(nn.Module):
    embedding_dim = 256

    def __init__(self):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(512, 256), nn.ReLU())
        self.attention = GatedAttention(256, 128)
        self.risk_head = RiskHead(self.embedding_dim)

    def encode_slide(self, features, return_attention=False, **_):
        embedding, weights = self.attention(self.projection(features))
        return (embedding, weights) if return_attention else embedding

class PatchGCNCox(nn.Module):
    """Official Patch-GCN core with only declared 512-D/dropout/head adaptations."""
    embedding_dim = 512

    def __init__(self):
        super().__init__()
        from torch_geometric.nn import GENConv, DeepGCNLayer
        self.fc = nn.Sequential(nn.Linear(512, 128), nn.ReLU())
        self.layers = nn.ModuleList()
        for i in range(1, 4):
            conv = GENConv(128, 128, aggr='softmax', t=1.0, learn_t=True, num_layers=2, norm='layer')
            self.layers.append(DeepGCNLayer(conv, nn.LayerNorm(128), nn.ReLU(inplace=True), block='res', dropout=0.0, ckpt_grad=i % 3))
        self.path_phi = nn.Sequential(nn.Linear(512, 512), nn.ReLU())
        self.attention = GatedAttention(512, 512)
        self.path_rho = nn.Sequential(nn.Linear(512, 512), nn.ReLU())
        self.risk_head = RiskHead(self.embedding_dim)

    def encode_slide(self, features, edge_index=None, **_):
        if edge_index is None:
            raise ValueError('Patch-GCN requires official spatial edge_index')
        x = self.fc(features)
        streams = [x]
        x = self.layers[0].conv(x, edge_index)
        streams.append(x)
        for layer in self.layers[1:]:
            x = layer(x, edge_index)
            streams.append(x)
        h = self.path_phi(torch.cat(streams, dim=1))
        h, _ = self.attention(h)
        return self.path_rho(h)
class Head(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x).flatten()
