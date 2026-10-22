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
from nystrom_attention import NystromAttention

class TransLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(512)
        self.attn = NystromAttention(dim=512, dim_head=64, heads=8, num_landmarks=256, pinv_iterations=6, residual=True, dropout=0.1)

    def forward(self, x):
        return x + self.attn(self.norm(x))

class PPEG(nn.Module):

    def __init__(self):
        super().__init__()
        self.p7 = nn.Conv2d(512, 512, 7, 1, 3, groups=512)
        self.p5 = nn.Conv2d(512, 512, 5, 1, 2, groups=512)
        self.p3 = nn.Conv2d(512, 512, 3, 1, 1, groups=512)

    def forward(self, x, h, w):
        cls, feat = (x[:, :1], x[:, 1:])
        z = feat.transpose(1, 2).reshape(x.shape[0], 512, h, w)
        z = self.p7(z) + z + self.p5(z) + self.p3(z)
        return torch.cat((cls, z.flatten(2).transpose(1, 2)), 1)

class TransMILSurvival(nn.Module):

    def __init__(self):
        super().__init__()
        self.input_projection = nn.Sequential(nn.Linear(512, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.layer1 = TransLayer()
        self.pos = PPEG()
        self.layer2 = TransLayer()
        self.norm = nn.LayerNorm(512)
        self.risk_head = nn.Linear(512, 1)

    def ckpt(self, fn, x):
        return checkpoint(fn, x, use_reentrant=False, preserve_rng_state=True) if self.training and torch.is_grad_enabled() else fn(x)

    def encode_slide(self, x):
        x = self.ckpt(self.input_projection, x).unsqueeze(0)
        n = x.shape[1]
        h = int(np.ceil(np.sqrt(n)))
        add = h * h - n
        if add:
            x = torch.cat((x, x[:, :add]), 1)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), 1)
        x = self.ckpt(self.layer1, x)
        x = self.ckpt(lambda y: self.pos(y, h, h), x)
        x = self.ckpt(self.layer2, x)
        return self.norm(x)[:, 0].squeeze(0)
