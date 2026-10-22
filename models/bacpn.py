"""B-ACPN: Boosted Attentive Climate Push Network (proposed model).

Two stages, trained per fold:

  1. Tree stage    : a random forest on all 13 features gives a first risk score (log-odds).
                     For the training districts this score is computed out-of-fold (5 inner folds),
                     so the deep stage never sees a score the forest made on its own training rows.
  2. Deep stage    : a neural network starts from that score and learns what the forest misses:
                        logit = tree score + development term + climate push
                        climate push = (attention -> MLP -> residual block on the 8 hazard features)
                                       x development gate (how strongly climate acts at this
                                       development level)
  3. Ensemble      : 5 deep stages with different seeds, averaged.

Uncertainty: MC Dropout spread of the climate push; split-conformal prediction sets.
Every part can be switched off for the ablation study.
"""
import time

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from torch import nn

from models.acpn import FeatureAttention, ResidualBlock, train_torch

DEV_DIM, HAZ_DIM = 5, 8


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class DeepStage(nn.Module):
    def __init__(self, hidden=16, dropout=0.3, use_attention=True, use_gate=True, use_climate=True, use_dev=True):
        super().__init__()
        self.use_attention, self.use_gate, self.use_climate, self.use_dev = use_attention, use_gate, use_climate, use_dev
        self.dev = nn.Linear(DEV_DIM, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        self.att = FeatureAttention(HAZ_DIM)
        self.climate = nn.Sequential(nn.Linear(HAZ_DIM, hidden), nn.ReLU(), nn.Dropout(dropout),
                                     ResidualBlock(hidden, dropout), nn.Linear(hidden, 1, bias=False))
        self.gate = nn.Sequential(nn.Linear(DEV_DIM, 4), nn.Tanh(), nn.Linear(4, 1))

    def forward(self, x):
        """x = [5 development, 8 hazard, 1 tree score]."""
        xd, xh, off = x[:, :DEV_DIM], x[:, DEV_DIM:DEV_DIM + HAZ_DIM], x[:, -1:]
        b = (self.dev(xd) if self.use_dev else self.bias.expand(xd.shape[0], 1)) + off
        a = torch.full_like(xh, 1.0 / HAZ_DIM)
        if self.use_climate:
            h = xh
            if self.use_attention:
                h, a = self.att(xh)
            c = self.climate(h)
            if self.use_gate:
                c = c * 2 * torch.sigmoid(self.gate(xd))
        else:
            c = torch.zeros_like(b)
        return (b + c).squeeze(1), b.squeeze(1), c.squeeze(1), a


class BACPNClassifier:
    """scikit-learn style wrapper: columns 0-4 development, 5-12 hazard."""

    def __init__(self, use_trees=True, seeds=(42, 1, 2, 3, 4), n_trees=500, standardize=True, mc_passes=50,
                 alpha=0.10, **arch):
        self.use_trees, self.seeds, self.n_trees, self.standardize = use_trees, tuple(seeds), n_trees, standardize
        self.mc_passes, self.alpha, self.arch = mc_passes, alpha, arch

    def _forest(self):
        return RandomForestClassifier(self.n_trees, min_samples_leaf=2, random_state=42, n_jobs=2)

    def _design(self, X, score):
        Z = (X - self.mu_) / self.sd_ if self.standardize else X
        return torch.tensor(np.column_stack([Z, score]).astype(np.float32))

    def _tree_score(self, X):
        return _logit(self.forest_.predict_proba(X)[:, 1]) if self.use_trees else np.zeros(len(X))

    @staticmethod
    def _fwd(net, x):
        return net(x)[0]

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y)
        t0 = time.perf_counter()
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-8
        oof = np.zeros(len(y))
        if self.use_trees:
            for a, b in StratifiedKFold(5, shuffle=True, random_state=11).split(X, y):
                oof[b] = _logit(self._forest().fit(X[a], y[a]).predict_proba(X[b])[:, 1])
            self.forest_ = self._forest().fit(X, y)
        D = self._design(X, oof).numpy()
        self.nets_, self.history_, vals = [], None, []
        for s in self.seeds:
            torch.manual_seed(s)
            net = DeepStage(**self.arch)
            idx_val, hist = train_torch(net, D, y, self._fwd, seed=s)
            self.nets_.append(net)
            self.history_ = self.history_ or hist
            vals.append(idx_val)
        self.train_time_ = time.perf_counter() - t0
        # centre the push on the training districts (its constant moves into the baseline)
        with torch.no_grad():
            Dt = torch.tensor(D)
            self.push_offset_ = float(np.mean([n(Dt)[2].mean().item() for n in self.nets_]))
        # split conformal on the first network's held-out validation districts (out-of-fold tree scores)
        iv = vals[0]
        with torch.no_grad():
            p = torch.sigmoid(torch.stack([n(Dt[iv])[0] for n in self.nets_]).mean(0)).numpy()
        sc = np.where(y[iv] == 1, 1 - p, p)
        n = len(sc)
        self.qhat_ = np.quantile(sc, min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n), method="higher")
        return self

    def n_params(self):
        return sum(p.numel() for p in self.nets_[0].parameters() if p.requires_grad)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        D = self._design(X, self._tree_score(X))
        with torch.no_grad():
            p = torch.sigmoid(torch.stack([n.eval()(D)[0] for n in self.nets_])).mean(0).numpy()
        return np.column_stack([1 - p, p])

    def conformal_sets(self, X):
        p = self.predict_proba(X)[:, 1]
        return np.column_stack([p <= self.qhat_, 1 - p <= self.qhat_])

    def explain(self, X):
        """Baseline (tree score + development), climate push, attention; MC Dropout SD of the push."""
        X = np.asarray(X, dtype=np.float64)
        D = self._design(X, self._tree_score(X))
        bs, cs, atts, mc = [], [], [], []
        with torch.no_grad():
            for net in self.nets_:
                net.eval()
                _, b, c, a = net(D)
                bs.append(b.numpy()); cs.append(c.numpy()); atts.append(a.numpy())
                for m in net.modules():
                    if isinstance(m, nn.Dropout):
                        m.train()
                mc += [net(D)[2].numpy() for _ in range(max(1, self.mc_passes // len(self.nets_)))]
                net.eval()
        o = self.push_offset_
        return {"baseline": np.mean(bs, 0) + o, "push": np.mean(cs, 0) - o, "attention": np.mean(atts, 0),
                "push_mc_mean": np.mean(mc, 0) - o, "push_mc_sd": np.std(mc, 0)}
