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
CONFIG={'droprate':0.,'droprate_2':0.,'numGroup':4}
class DimReduction(nn.Module):
    """Pinned official Model/network.py::DimReduction, numLayer_Res=0."""

    def __init__(self, n_channels=512, m_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu1(self.fc1(x))

class AttentionGated(nn.Module):
    """Pinned official Model/Attention.py::Attention_Gated."""

    def __init__(self, L=512, D=128, K=1):
        super().__init__()
        self.V = nn.Sequential(nn.Linear(L, D), nn.Tanh())
        self.U = nn.Sequential(nn.Linear(L, D), nn.Sigmoid())
        self.W = nn.Linear(D, K)

    def forward(self, x, is_norm=True):
        a = self.W(self.V(x) * self.U(x)).transpose(1, 0)
        return F.softmax(a, dim=1) if is_norm else a

class ScalarRiskHead(nn.Module):
    """Minimal survival replacement for official Classifier_1fc."""

    def __init__(self, L=512, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.fc = nn.Linear(L, 1)

    def forward(self, x):
        return self.fc(self.dropout(x))

class AttentionWithScalarRiskHead(nn.Module):
    """Official Attention_with_Classifier with only its terminal head made scalar."""

    def __init__(self):
        super().__init__()
        self.attention = AttentionGated()
        self.risk = ScalarRiskHead(dropout=CONFIG['droprate_2'])

    def representation(self, x):
        return self.attention(x) @ x

    def forward(self, x):
        return self.risk(self.representation(x))

@dataclass
class DTFDOutput:
    risk: torch.Tensor
    representation: torch.Tensor
    group_representations: tuple[torch.Tensor, ...]

class DTFDSurvivalAdapter(nn.Module):

    def __init__(self):
        super().__init__()
        self.dimReduction = DimReduction()
        self.attention = AttentionGated()
        self.tier1_risk = ScalarRiskHead(dropout=CONFIG['droprate'])
        self.UClassifier = AttentionWithScalarRiskHead()

    def optimizer0_parameters(self):
        return list(self.dimReduction.parameters()) + list(self.attention.parameters()) + list(self.tier1_risk.parameters())

    def optimizer1_parameters(self):
        return list(self.UClassifier.parameters())

    @staticmethod
    def random_partition(n, device):
        perm = torch.randperm(n, device=device)
        return tuple((x for x in torch.tensor_split(perm, CONFIG['numGroup']) if x.numel()))

    def encode_slide(self, x, partition=None):
        if partition is None:
            partition = self.random_partition(x.shape[0], x.device)
        mid = self.dimReduction(x)
        groups = []
        for idx in partition:
            sub = mid.index_select(0, idx)
            groups.append((self.attention(sub) @ sub).squeeze(0))
        group_tensor = torch.stack(groups)
        return (group_tensor.mean(0), group_tensor)

    def tier1_patient(self, bags, partitions=None):
        if partitions is None:
            partitions = [None] * len(bags)
        encoded = [self.encode_slide(x, p) for x, p in zip(bags, partitions)]
        rep = torch.stack([x[0] for x in encoded]).mean(0)
        return DTFDOutput(self.tier1_risk(rep).reshape(()), rep, tuple((x[1] for x in encoded)))

    def distilled_patient(self, bags, partitions=None):
        if partitions is None:
            partitions = [None] * len(bags)
        return tuple((self.encode_slide(x, p)[1] for x, p in zip(bags, partitions)))

    def tier2_from_distilled(self, distilled):
        slide_rep = torch.stack([self.UClassifier.representation(x) for x in distilled]).mean(0).squeeze(0)
        return DTFDOutput(self.UClassifier.risk(slide_rep).reshape(()), slide_rep, tuple(distilled))

    def tier2_patient(self, bags, partitions=None, detach_tier1=True):
        if detach_tier1:
            with torch.no_grad():
                distilled = self.distilled_patient(bags, partitions)
            distilled = tuple((x.detach() for x in distilled))
        else:
            distilled = self.distilled_patient(bags, partitions)
        return self.tier2_from_distilled(distilled)
