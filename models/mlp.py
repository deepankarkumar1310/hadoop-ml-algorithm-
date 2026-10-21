"""M5 comparison model: standard MLP (13 -> 64 -> 32 -> 16 -> 1) in PyTorch.

Same training loop as A-CPN (class-weighted BCE, AdamW, early stopping), but no development / climate
split, no attention, no KAN and no residual block - it shows what the A-CPN structure adds."""
import time

import numpy as np
import torch
from torch import nn

from models.acpn import train_torch


class MLP(nn.Module):
    def __init__(self, d_in=13, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(16, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


class MLPClassifier:
    def __init__(self, standardize=True, seed=42):
        self.standardize, self.seed = standardize, seed

    def _prep(self, X):
        X = np.asarray(X, dtype=np.float32)
        return (X - self.mu_) / self.sd_ if self.standardize else X

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        X = np.asarray(X, dtype=np.float32)
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-8
        self.net_ = MLP(X.shape[1])
        t0 = time.perf_counter()
        train_torch(self.net_, self._prep(X), np.asarray(y), lambda n, x: n(x), seed=self.seed)
        self.train_time_ = time.perf_counter() - t0
        return self

    def n_params(self):
        return sum(p.numel() for p in self.net_.parameters() if p.requires_grad)

    def predict_proba(self, X):
        self.net_.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.net_(torch.tensor(self._prep(X), dtype=torch.float32))).numpy()
        return np.column_stack([1 - p, p])
