"""Shared building blocks of B-ACPN: feature attention, residual block and the PyTorch training loop
(the training loop is also used by the standard MLP baseline)."""

import numpy as np
import torch
from torch import nn


class FeatureAttention(nn.Module):
    """Per-sample softmax weights over the input features, rescaled so the mean weight is 1."""

    def __init__(self, d, hidden=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, d))

    def forward(self, x):
        a = torch.softmax(self.net(x), dim=1)
        return x * a * x.shape[1], a


class ResidualBlock(nn.Module):
    def __init__(self, d, dropout=0.3):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x):
        return x + self.body(x)


def train_torch(net, Xtr, ytr, forward, lr=3e-3, wd=1e-3, batch=64, max_epochs=300, patience=30,
                val_frac=0.15, seed=42):
    """Shared training loop: class-weighted BCE, AdamW, early stopping on a stratified validation split.

    Returns the validation indices (used later for conformal calibration)."""
    rng = np.random.default_rng(seed)
    idx_val = np.concatenate([rng.choice(np.where(ytr == c)[0], max(1, int(round(val_frac * (ytr == c).sum()))),
                                         replace=False) for c in (0, 1)])
    idx_tr = np.setdiff1d(np.arange(len(ytr)), idx_val)
    pos = ytr[idx_tr].mean()
    w = torch.tensor([0.5 / (1 - pos), 0.5 / pos], dtype=torch.float32)
    X = torch.tensor(Xtr, dtype=torch.float32)
    Y = torch.tensor(ytr, dtype=torch.float32)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    best, best_state, wait, history = np.inf, None, 0, []
    g = torch.Generator().manual_seed(seed)
    for epoch in range(max_epochs):
        net.train()
        perm = idx_tr[torch.randperm(len(idx_tr), generator=g).numpy()]
        tl = 0.0
        for s in range(0, len(perm), batch):
            bi = perm[s:s + batch]
            if len(bi) < 2:
                continue
            logit = forward(net, X[bi])
            loss = nn.functional.binary_cross_entropy_with_logits(logit, Y[bi], weight=w[Y[bi].long()])
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * len(bi)
        net.eval()
        with torch.no_grad():
            lv = nn.functional.binary_cross_entropy_with_logits(
                forward(net, X[idx_val]), Y[idx_val], weight=w[Y[idx_val].long()]).item()
        history.append((tl / len(idx_tr), lv))
        if lv < best - 1e-4:
            best, wait = lv, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    return idx_val, history


