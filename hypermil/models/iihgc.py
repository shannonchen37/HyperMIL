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
CFG={'intra_hiddens':[128,128,128],'dropout':.3}
def pairdist(x):
    return (x.square().sum(-1, keepdim=True) - 2 * x @ x.transpose(-1, -2) + x.square().sum(-1, keepdim=True).transpose(-1, -2)).clamp_min(0)

def full_H(x, k_nearest=None, k_threshold=None):
    d = pairdist(x)
    mx = d.max(1, keepdim=True).values.clamp_min(torch.finfo(d.dtype).eps)
    nd = d / mx
    parts = []
    if k_nearest is not None:
        t = torch.topk(d, k_nearest, dim=1, largest=False).values[:, -1:, :]
        parts.append((d <= t).to(x.dtype))
    if k_threshold is not None:
        parts.append((nd < k_threshold).to(x.dtype))
    H = torch.cat(parts, -1)
    return H

class IntraHyConv(nn.Module):

    def __init__(self, a, b, drop=0.3, ratio=0.05):
        super().__init__()
        self.theta = nn.Parameter(torch.empty(a, b))
        self.bias = nn.Parameter(torch.zeros(b))
        nn.init.xavier_uniform_(self.theta)
        self.drop = drop
        self.ratio = ratio

    def forward(self, x, H, train=False):
        y = einsum('nvc,co->nvo', x, self.theta)
        Dv = torch.diag_embed(1 / H.sum(-1))
        HDv = einsum('nkv,nve->nke', Dv, H)
        De = torch.diag_embed(1 / H.sum(1))
        HDe = einsum('nve,nek->nvk', H, De)
        y = einsum('nvc,nve->nec', y, HDe)
        if train:
            n, v, c = y.shape
            dn = int(v * self.ratio)
            if dn:
                threshold = torch.topk(y, dn, dim=1).values[:, -1:, :]
                mask = (y <= threshold) | (torch.rand_like(y) <= 1 - self.drop)
                y = y * mask
        return torch.relu(einsum('nec,nve->nvc', y, HDv) + self.bias[None, None])

class IntraModel(nn.Module):

    def __init__(self):
        super().__init__()
        hs = CFG['intra_hiddens']
        self.ft = nn.ModuleList([IntraHyConv(512, hs[0]), IntraHyConv(hs[0], hs[1]), IntraHyConv(hs[1], hs[2])])
        self.co = nn.ModuleList([IntraHyConv(512, hs[0]), IntraHyConv(hs[0], hs[1]), IntraHyConv(hs[1], hs[2])])
        self.dropout = nn.Dropout(CFG['dropout'])
        self.risk = nn.Linear(256, 1)

    def encode_slide(self, x, coor, train=False):
        Hf = full_H(x, 5, 0.1)
        Hc = full_H(coor, 5, 0.1)
        a = b = x
        for i, (lf, lc) in enumerate(zip(self.ft, self.co)):
            a = lf(a, Hf, train)
            b = lc(b, Hc, train)
            if i < 2:
                a = self.dropout(a)
                b = self.dropout(b)
        return torch.cat((a.max(1).values, b.max(1).values), -1)

    def patient(self, slides, train=False):
        z = torch.stack([self.encode_slide(x, c, train).squeeze(0) for x, c in slides]).mean(0)
        return (self.risk(self.dropout(z)).reshape(()), z)

class InterHyConv(nn.Module):

    def __init__(self, a, b):
        super().__init__()
        self.theta = nn.Parameter(torch.empty(a, b))
        self.bias = nn.Parameter(torch.zeros(b))
        nn.init.xavier_uniform_(self.theta)

    def forward(self, x, H):
        y = x @ self.theta + self.bias
        Dv = torch.diag(1 / H.sum(1))
        De = torch.diag(1 / H.sum(0))
        return Dv @ H @ (De @ H.T @ y)

class InterModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.path = nn.ModuleList([InterHyConv(256, 128), InterHyConv(128, 128), InterHyConv(128, 128)])
        self.label = nn.ModuleList([InterHyConv(1, 16), InterHyConv(16, 16), InterHyConv(16, 16)])
        self.label_fc = nn.Linear(16, 1)
        self.mlp = nn.ModuleList([nn.Linear(128, 128), nn.Linear(128, 128), nn.Linear(128, 1)])
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, risk, train_x=None, train_risk=None):
        n = x.shape[0]
        allx = torch.cat((x, train_x), 0) if train_x is not None else x
        allr = torch.cat((risk, train_risk), 0) if train_risk is not None else risk
        H = full_H(allx, k_threshold=0.01)
        l = allr
        self.last_graph_shape = tuple(H.shape)
        for m in self.label:
            l = self.dropout(m(l, H))
        l = self.label_fc(l)[:n]
        z = allx
        for m in self.path:
            z = self.dropout(m(z, H))
        z = z[:n] * l
        z = self.mlp[1](self.mlp[0](z))
        emb = z
        s2 = self.mlp[2](self.dropout(z))
        return (s2, s2 + risk, emb)
