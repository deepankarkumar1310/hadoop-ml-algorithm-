"""A-CPN: Attentive Climate Push Network (proposed model).

    logit(High vulnerability) = development baseline  +  climate push
                                 Linear(5 dev features)   attention -> KAN -> residual MLP (8 hazard features)

  * Development baseline : one linear layer on the 5 development features (readable coefficients).
  * Feature attention    : per-district softmax weights over the 8 hazard features (which factor matters).
  * KAN layer            : Kolmogorov-Arnold layer, a learnable cubic B-spline on every input-output edge.
  * Residual block       : Linear -> BatchNorm -> ReLU -> Dropout, with a skip connection.
  * Climate push         : the climate network's output, added to the baseline (no bias term, so the
                           intercept stays in the baseline). Push > 0 = climate adds risk beyond development.
  * Uncertainty          : MC Dropout (dropout kept on, many passes) and split conformal prediction sets.

Every component can be switched off for the ablation study.
"""
import time

import numpy as np
import torch
from torch import nn

DEV_DIM, HAZ_DIM = 5, 8


class FeatureAttention(nn.Module):
    """Per-sample softmax weights over the input features, rescaled so the mean weight is 1."""

    def __init__(self, d, hidden=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, d))

    def forward(self, x):
        a = torch.softmax(self.net(x), dim=1)
        return x * a * x.shape[1], a


class KANLayer(nn.Module):
    """Kolmogorov-Arnold layer (Liu et al., 2024): y_j = sum_i [w_b * silu(x_i) + spline_ij(x_i)].

    Splines are cubic B-splines on a uniform grid over [-3, 3] (inputs are standardised)."""

    def __init__(self, d_in, d_out, grid=5, order=3, lo=-3.0, hi=3.0):
        super().__init__()
        h = (hi - lo) / grid
        knots = torch.arange(-order, grid + order + 1, dtype=torch.float32) * h + lo
        self.register_buffer("knots", knots)
        self.order = order
        self.base = nn.Linear(d_in, d_out)
        self.spline = nn.Parameter(torch.randn(d_out, d_in, grid + order) * 0.1)

    def bases(self, x):
        x = x.clamp(self.knots[self.order].item(), self.knots[-self.order - 1].item() - 1e-6).unsqueeze(-1)
        k = self.knots
        b = ((x >= k[:-1]) & (x < k[1:])).float()
        for p in range(1, self.order + 1):
            b = ((x - k[:-(p + 1)]) / (k[p:-1] - k[:-(p + 1)]) * b[..., :-1]
                 + (k[p + 1:] - x) / (k[p + 1:] - k[1:-p]) * b[..., 1:])
        return b  # (n, d_in, grid + order)

    def forward(self, x):
        return self.base(nn.functional.silu(x)) + torch.einsum("nik,oik->no", self.bases(x), self.spline)


class ResidualBlock(nn.Module):
    def __init__(self, d, dropout=0.3):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x):
        return x + self.body(x)


class ACPN(nn.Module):
    def __init__(self, hidden=16, dropout=0.3, use_attention=True, use_kan=True, use_residual=True,
                 use_dev=True, use_climate=True, linear_climate=False):
        super().__init__()
        self.use_dev, self.use_climate = use_dev, use_climate
        self.use_attention = use_attention and not linear_climate
        self.dev = nn.Linear(DEV_DIM, 1)
        self.bias = nn.Parameter(torch.zeros(1))  # intercept when the development part is switched off
        if linear_climate:
            self.climate = nn.Linear(HAZ_DIM, 1, bias=False)
        else:
            self.att = FeatureAttention(HAZ_DIM) if self.use_attention else None
            first = KANLayer(HAZ_DIM, hidden) if use_kan else nn.Sequential(nn.Linear(HAZ_DIM, hidden), nn.ReLU())
            layers = [first, nn.Dropout(dropout)]
            if use_residual:
                layers.append(ResidualBlock(hidden, dropout))
            layers.append(nn.Linear(hidden, 1, bias=False))
            self.climate = nn.Sequential(*layers)

    def forward(self, xd, xh):
        b = self.dev(xd) if self.use_dev else self.bias.expand(xd.shape[0], 1)
        a = torch.full_like(xh, 1.0 / HAZ_DIM)
        if self.use_climate:
            if getattr(self, "att", None) is not None:
                xh, a = self.att(xh)
            c = self.climate(xh)
        else:
            c = torch.zeros_like(b)
        return (b + c).squeeze(1), b.squeeze(1), c.squeeze(1), a


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


class ACPNClassifier:
    """scikit-learn style wrapper: columns 0-4 development, 5-12 hazard."""

    def __init__(self, standardize=True, seed=42, mc_passes=50, alpha=0.10, **arch):
        self.standardize, self.seed, self.mc_passes, self.alpha, self.arch = standardize, seed, mc_passes, alpha, arch

    def _prep(self, X):
        X = np.asarray(X, dtype=np.float32)
        return (X - self.mu_) / self.sd_ if self.standardize else X

    @staticmethod
    def _forward(net, X):
        return net(X[:, :DEV_DIM], X[:, DEV_DIM:])[0]

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        X = np.asarray(X, dtype=np.float32)
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-8
        Xs, y = self._prep(X), np.asarray(y)
        self.net_ = ACPN(**self.arch)
        t0 = time.perf_counter()
        idx_val, self.history_ = train_torch(self.net_, Xs, y, self._forward, seed=self.seed)
        self.train_time_ = time.perf_counter() - t0
        # centre the push on the training districts so its constant part moves into the baseline
        with torch.no_grad():
            Xt = torch.tensor(Xs, dtype=torch.float32)
            self.push_offset_ = float(self.net_(Xt[:, :DEV_DIM], Xt[:, DEV_DIM:])[2].mean())
        # split conformal: nonconformity = 1 - p(true class) on the held-out validation split
        p = self.predict_proba(X[idx_val])[:, 1]
        s = np.where(y[idx_val] == 1, 1 - p, p)
        n = len(s)
        self.qhat_ = np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n), method="higher")
        return self

    def n_params(self):
        return sum(p.numel() for p in self.net_.parameters() if p.requires_grad)

    def predict_proba(self, X):
        Xt = torch.tensor(self._prep(X), dtype=torch.float32)
        self.net_.eval()
        with torch.no_grad():
            p = torch.sigmoid(self._forward(self.net_, Xt)).numpy()
        return np.column_stack([1 - p, p])

    def conformal_sets(self, X):
        """Prediction sets at 1 - alpha coverage: columns (Low in set, High in set)."""
        p = self.predict_proba(X)[:, 1]
        return np.column_stack([p <= self.qhat_, 1 - p <= self.qhat_])

    def explain(self, X):
        """Baseline, climate push, attention weights; MC Dropout mean and SD of the push."""
        Xt = torch.tensor(self._prep(X), dtype=torch.float32)
        self.net_.eval()
        with torch.no_grad():
            _, b, c, a = self.net_(Xt[:, :DEV_DIM], Xt[:, DEV_DIM:])
            # MC Dropout: dropout layers active, BatchNorm kept in eval mode
            for m in self.net_.modules():
                if isinstance(m, nn.Dropout):
                    m.train()
            pushes = np.stack([self.net_(Xt[:, :DEV_DIM], Xt[:, DEV_DIM:])[2].numpy() for _ in range(self.mc_passes)])
            self.net_.eval()
        o = self.push_offset_
        return {"baseline": b.numpy() + o, "push": c.numpy() - o, "attention": a.numpy(),
                "push_mc_mean": pushes.mean(0) - o, "push_mc_sd": pushes.std(0)}
