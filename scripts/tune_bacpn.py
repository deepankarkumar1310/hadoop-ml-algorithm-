"""Hyperparameter tuning of B-ACPN's deep stage.

Grid: hidden units {8, 16, 32} x dropout {0.1, 0.3, 0.5} x learning rate {0.001, 0.003} = 18 settings.
The tree stage (500-tree random forest, out-of-fold score) does not depend on these settings, so it is fitted
once per fold and reused; every setting is then trained exactly like B-ACPN (5 networks, same seeds, same
early stopping) on the same 25 folds (5 x 5-fold stratified CV).

Selection uses the mean CV ROC-AUC only. The chosen setting is then checked on leave-one-state-out (LOSO),
which plays no part in the selection.

Writes results/tuning_grid.csv (all 18 settings) and results/tuning_summary.json.
Usage: python scripts/tune_bacpn.py [--workers 4]
"""
import argparse
import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.bacpn import BACPNClassifier, DeepStage, _logit  # noqa: E402
from models.layers import train_torch  # noqa: E402
from scripts.common import RES, get_folds, index_of, load_table  # noqa: E402

GRID = [dict(hidden=h, dropout=d, lr=lr) for h, d, lr in itertools.product((8, 16, 32), (0.1, 0.3, 0.5),
                                                                           (0.001, 0.003))]
DEFAULT = dict(hidden=16, dropout=0.3, lr=0.003)
SEEDS = (42, 1, 2, 3, 4)


def _fold(job):
    """Fit the tree stage once for this fold, then every deep-stage setting; return test probabilities."""
    torch.set_num_threads(1)
    X, y, tr, te, settings = job
    base = BACPNClassifier()
    Xtr, ytr = X[tr], y[tr]
    base.mu_, base.sd_ = Xtr.mean(0), Xtr.std(0) + 1e-8
    oof = np.zeros(len(ytr))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=11).split(Xtr, ytr):
        oof[b] = _logit(base._forest().fit(Xtr[a], ytr[a]).predict_proba(Xtr[b])[:, 1])
    base.forest_ = base._forest().fit(Xtr, ytr)
    Dtr = base._design(Xtr, oof).numpy()
    Dte = base._design(X[te], base._tree_score(X[te]))
    out = []
    for s in settings:
        nets = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            net = DeepStage(hidden=s["hidden"], dropout=s["dropout"])
            train_torch(net, Dtr, ytr, BACPNClassifier._fwd, lr=s["lr"], seed=seed)
            nets.append(net.eval())
        with torch.no_grad():
            p = torch.sigmoid(torch.stack([n(Dte)[0] for n in nets])).mean(0).numpy()
        out.append(p)
    return out


def evaluate(df, X, y, folds, settings, workers, pooled=False):
    idx = index_of(df)
    jobs = [(X, y, idx(f["train"]), idx(f["test"]), settings) for f in folds]
    with Pool(workers) as pool:
        res = pool.map(_fold, jobs)
    rows = []
    for k, s in enumerate(settings):
        if pooled:  # LOSO: one AUC over all held-out states, as in evaluate.py
            yt = np.concatenate([y[idx(f["test"])] for f in folds]); p = np.concatenate([r[k] for r in res])
            rows.append({**s, "roc_auc": roc_auc_score(yt, p), "accuracy": accuracy_score(yt, p >= 0.5),
                         "f1": f1_score(yt, p >= 0.5)})
            continue
        auc, acc, f1 = [], [], []
        for f, r in zip(folds, res):
            yt, p = y[idx(f["test"])], r[k]
            auc.append(roc_auc_score(yt, p)); acc.append(accuracy_score(yt, p >= 0.5)); f1.append(f1_score(yt, p >= 0.5))
        rows.append({**s, "roc_auc": np.mean(auc), "roc_auc_sd": np.std(auc), "accuracy": np.mean(acc),
                     "f1": np.mean(f1)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    df, X, y = load_table()
    folds = get_folds(df, y)
    grid = evaluate(df, X, y, folds["cv"], GRID, a.workers)
    grid["default"] = [all(r[k] == v for k, v in DEFAULT.items()) for _, r in grid.iterrows()]
    grid = grid.sort_values("roc_auc", ascending=False).reset_index(drop=True)
    grid.to_csv(RES / "tuning_grid.csv", index=False)
    best = {k: grid.loc[0, k] for k in DEFAULT}
    best = {"hidden": int(best["hidden"]), "dropout": float(best["dropout"]), "lr": float(best["lr"])}
    check = [DEFAULT] if best == DEFAULT else [DEFAULT, best]
    loso = evaluate(df, X, y, folds["loso"], check, a.workers, pooled=True)
    d = grid[grid.default].iloc[0]
    summary = {"settings_tried": len(GRID), "folds": len(folds["cv"]), "default": DEFAULT, "best": best,
               "best_is_default": best == DEFAULT,
               "cv_auc_default": round(float(d.roc_auc), 4), "cv_auc_best": round(float(grid.loc[0, "roc_auc"]), 4),
               "cv_auc_worst": round(float(grid.roc_auc.min()), 4),
               "loso_auc_default": round(float(loso.roc_auc.iloc[0]), 4),
               "loso_auc_best": round(float(loso.roc_auc.iloc[-1]), 4)}
    (RES / "tuning_summary.json").write_text(json.dumps(summary, indent=2))
    print(grid.to_string())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
