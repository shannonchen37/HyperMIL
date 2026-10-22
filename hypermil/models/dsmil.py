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

class DSMILSurvival(nn.Module):
    """Pinned official DSMIL core with the explicitly frozen two-stream survival adapter."""

    def __init__(self):
        super().__init__()
        self.instance_head = nn.Linear(512, 1)
        self.q = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh())
        self.bag_head = nn.Conv1d(1, 1, kernel_size=512)

    def encode_slide(self, x):
        instance_scores = self.instance_head(x).flatten()
        critical = torch.argmax(instance_scores)
        q_all = self.q(x)
        q_max = self.q(x[critical:critical + 1])
        attention = torch.softmax(q_all @ q_max.T / math.sqrt(128.0), dim=0)
        bag = (attention.T @ x).reshape(512)
        return (x[critical], bag)
