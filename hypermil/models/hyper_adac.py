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
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial import distance, cKDTree
from torch_geometric.nn import HypergraphConv
def canonical_labels(labels):
    groups = sorted((tuple(np.flatnonzero(labels == k)) for k in np.unique(labels)), key=lambda x: x[0])
    out = np.empty(len(labels), np.int64)
    for k, g in enumerate(groups):
        out[list(g)] = k
    return out

def indexed_adapter(x, c, ld=0.003, lf=0.001, th=0.8):
    keep = np.ptp(x, axis=0) > 0
    n = len(x)
    cond = np.ones(n * (n - 1) // 2, np.float64)
    tree = cKDTree(c)
    for i, js in enumerate(tree.query_ball_point(c, r=2000.0)):
        js = np.asarray([j for j in js if j > i], np.int64)
        if len(js):
            fd = np.linalg.norm(x[js][:, keep] - x[i, keep], axis=1)
            sd = np.linalg.norm(c[js] - c[i], axis=1)
            at = n * i - i * (i + 1) // 2 + js - i - 1
            cond[at] = 1 - np.exp(-lf * fd) * np.exp(-ld * sd)
    lab = fcluster(linkage(cond, method='average'), t=th, criterion='distance')
    return (canonical_labels(lab), cond)

def reduce_graph(x, c, labels):
    rx = np.stack([x[labels == k].mean(0) for k in np.unique(labels)]).astype(np.float32)
    rc = np.stack([c[labels == k].mean(0) for k in np.unique(labels)]).astype(np.float32)
    sim = np.exp(-0.5 * distance.cdist(rx, rx))
    inc = (sim >= 0.8).astype(np.float32)
    edge = np.vstack(np.nonzero(inc))[::-1].copy()
    return (rx, rc, inc, torch.from_numpy(edge).long())

class GatedAttention(nn.Module):

    def __init__(self):
        super().__init__()
        self.a = nn.Sequential(nn.Linear(128, 256), nn.Tanh())
        self.b = nn.Sequential(nn.Linear(128, 256), nn.Sigmoid())
        self.c = nn.Linear(256, 1)

    def forward(self, z):
        logits = self.c(self.a(z) * self.b(z)).flatten()
        w = torch.softmax(logits, 0)
        return ((w[:, None] * z).sum(0), w)

class HyperAdaC2K(nn.Module):

    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU())
        self.conv = HypergraphConv(256, 256, use_attention=False)
        self.linears = nn.ModuleList([nn.Linear(256, 128), nn.Linear(256, 128)])
        self.attn = GatedAttention()
        self.head = nn.Sequential(nn.Linear(128, 256), nn.BatchNorm1d(256), nn.SELU(), nn.Dropout(0.16), nn.Linear(256, 1), nn.BatchNorm1d(1), nn.SELU())

    def encode_slide(self, x, edge):
        x = self.first(x)
        z = self.linears[0](x)
        a, _ = self.attn(z)
        a = F.dropout(a, p=0.4, training=self.training)
        x = self.conv(x, edge)
        z = self.linears[1](x)
        b, _ = self.attn(z)
        b = F.dropout(b, p=0.4, training=self.training)
        return a + b

    def patient_embedding(self, graphs):
        return torch.stack([self.encode_slide(x, e) for x, e in graphs]).mean(0)

    def patient_risks(self, embeddings):
        return torch.sigmoid(self.head(embeddings)).flatten()
