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
from hypermil.data import Patient
CONFIG = {'n_token': 5, 'n_masked_patch': 10, 'mask_drop': 0.6}

class DimReduction(nn.Module):
    """Official ACMIL DimReduction with numLayer_Res=0."""

    def __init__(self, n_channels=512, m_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu1(self.fc1(x))

class AttentionGated(nn.Module):
    """Official Attention_Gated."""

    def __init__(self, L=256, D=128, K=5):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(L, D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(L, D), nn.Sigmoid())
        self.attention_weights = nn.Linear(D, K)

    def forward(self, x):
        return self.attention_weights(self.attention_V(x) * self.attention_U(x)).transpose(1, 0)

def attention_diversity(attn_logits):
    attn = torch.softmax(attn_logits, dim=-1)
    pairs = CONFIG['n_token'] * (CONFIG['n_token'] - 1) / 2
    value = attn.new_zeros(())
    for i in range(CONFIG['n_token']):
        for j in range(i + 1, CONFIG['n_token']):
            value = value + F.cosine_similarity(attn[i], attn[j], dim=-1) / pairs
    return value

@dataclass
class PatientOutput:
    main_risk: torch.Tensor
    token_risks: torch.Tensor
    main_embedding: torch.Tensor
    token_embeddings: torch.Tensor
    attention_logits: tuple[torch.Tensor, ...]
    diversity: torch.Tensor

class ACMILSurvivalAdapter(nn.Module):
    """Official ACMIL-GA core; only pre-classification embeddings are exposed."""

    def __init__(self):
        super().__init__()
        self.dimreduction = DimReduction()
        self.attention = AttentionGated()
        self.token_risk_heads = nn.ModuleList([nn.Linear(256, 1) for _ in range(5)])
        self.main_risk_head = nn.Linear(256, 1)

    def forward_slide(self, patches):
        x = self.dimreduction(patches)
        A = self.attention(x)
        if CONFIG['n_masked_patch'] > 0 and self.training:
            k, n = A.shape
            n_masked = min(CONFIG['n_masked_patch'], n)
            _, indices = torch.topk(A, n_masked, dim=-1)
            rand_selected = torch.argsort(torch.rand(*indices.shape, device=A.device), dim=-1)[:, :int(n_masked * CONFIG['mask_drop'])]
            masked_indices = indices[torch.arange(indices.shape[0], device=A.device).unsqueeze(-1), rand_selected]
            random_mask = torch.ones(k, n, device=A.device)
            random_mask.scatter_(-1, masked_indices, 0)
            A = A.masked_fill(random_mask == 0, -1000000000.0)
        weights = F.softmax(A, dim=1)
        token_embeddings = torch.mm(weights, x)
        main_embedding = torch.mm(weights.mean(0, keepdim=True), x).squeeze(0)
        return (main_embedding, token_embeddings, A)

    def forward_patient_bags(self, bags):
        slide_outputs = [self.forward_slide(x) for x in bags]
        main_embedding = torch.stack([x[0] for x in slide_outputs]).mean(0)
        token_embeddings = torch.stack([x[1] for x in slide_outputs]).mean(0)
        main_risk = self.main_risk_head(main_embedding).reshape(())
        token_risks = torch.stack([head(token_embeddings[i]).reshape(()) for i, head in enumerate(self.token_risk_heads)])
        logits = tuple((x[2] for x in slide_outputs))
        diversity = torch.stack([attention_diversity(x) for x in logits]).mean()
        return PatientOutput(main_risk, token_risks, main_embedding, token_embeddings, logits, diversity)
