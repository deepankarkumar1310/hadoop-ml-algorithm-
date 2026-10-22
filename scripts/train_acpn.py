"""Train and evaluate the proposed model B-ACPN (and its ablation variants) on the shared folds.

Writes out-of-fold predictions to results/predictions/, per-district explanations
(baseline, climate push, attention, MC Dropout SD) to results/district_explanations.csv,
conformal coverage to results/conformal.csv, the model card to results/table4_model_card.json,
and a final model trained on all 469 districts to models/saved/bacpn_full.joblib.

Usage: python scripts/train_acpn.py [--ablation] [--raw]
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib  # noqa: E402
from models.bacpn import BACPNClassifier  # noqa: E402
from scripts.common import HAZ, PRED, RES, ROOT, get_folds, index_of, load_table  # noqa: E402

NAME = "B-ACPN (proposed)"
VARIANTS = {
    "Full B-ACPN": {},
    "Deep stage only (no tree stage)": {"use_trees": False},
    "Tree stage only (no deep stage)": {"trees_only": True},
    "Without climate push": {"use_climate": False},
    "Without development gate": {"use_gate": False},
    "Without feature attention": {"use_attention": False},
    "Single network (no ensemble)": {"seeds": (42,)},
}


class TreesOnly:
    """Ablation: the tree stage alone (same forest as inside B-ACPN)."""

    def fit(self, X, y):
        t0 = time.perf_counter()
        self.m = BACPNClassifier()._forest().fit(X, y)
        self.train_time_ = time.perf_counter() - t0
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)


def make(arch, standardize=True):
    if arch.get("trees_only"):
        return TreesOnly()
    return BACPNClassifier(standardize=standardize, **arch)


def run(df, X, y, folds, arch, standardize=True, explain=False, kinds=("cv", "loso")):
    idx = index_of(df)
    rows, expl, conf = [], [], []
    for kind in kinds:
        for f in folds[kind]:
            tr, te = idx(f["train"]), idx(f["test"])
            m = make(arch, standardize).fit(X[tr], y[tr])
            t0 = time.perf_counter()
            p = m.predict_proba(X[te])[:, 1]
            pt = (time.perf_counter() - t0) / len(te)
            rows.append(pd.DataFrame({"censuscode": df.censuscode.iloc[te].values, "label": y[te], "score": p,
                                      "prob": p, "prediction": (p >= 0.5).astype(int), "fold": f["fold"],
                                      "kind": kind, "train_time": m.train_time_, "predict_time": pt}))
            if explain and kind == "cv":
                e = m.explain(X[te])
                d = pd.DataFrame({"censuscode": df.censuscode.iloc[te].values, "fold": f["fold"],
                                  "baseline": e["baseline"], "push": e["push"], "push_mc_sd": e["push_mc_sd"]})
                for j, h in enumerate(HAZ):
                    d[f"att_{h}"] = e["attention"][:, j]
                expl.append(d)
                s = m.conformal_sets(X[te])
                cover = np.where(y[te] == 1, s[:, 1], s[:, 0])
                conf.append({"fold": f["fold"], "coverage": cover.mean(), "avg_set_size": s.sum(1).mean(),
                             "singleton_share": (s.sum(1) == 1).mean()})
    return pd.concat(rows), (pd.concat(expl) if expl else None), pd.DataFrame(conf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--raw", action="store_true", help="unscaled features (preprocessing before/after)")
    a = ap.parse_args()
    df, X, y = load_table()
    folds = get_folds(df, y)
    PRED.mkdir(parents=True, exist_ok=True)

    if a.raw:
        pred, _, _ = run(df, X, y, folds, {}, standardize=False, kinds=("cv",))
        pred.to_csv(PRED / f"{NAME}__raw.csv", index=False)
        print("raw B-ACPN done")
        return
    if a.ablation:
        for name, arch in VARIANTS.items():
            if name == "Full B-ACPN":
                continue
            pred, _, _ = run(df, X, y, folds, arch, kinds=("cv",))
            pred.to_csv(PRED / f"ablation__{name}.csv", index=False)
            print("ablation:", name)
        return

    pred, expl, conf = run(df, X, y, folds, {}, explain=True)
    pred.to_csv(PRED / f"{NAME}__scaled.csv", index=False)
    expl.to_csv(RES / "district_explanations_folds.csv", index=False)
    conf.to_csv(RES / "conformal.csv", index=False)

    # final model on all districts: saved for the model card and for predictions
    t0 = time.perf_counter()
    full = BACPNClassifier().fit(X, y)
    secs = time.perf_counter() - t0
    out = ROOT / "models" / "saved"
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(full, out / "bacpn_full.joblib", compress=3)
    cv_times = pred[pred.kind == "cv"].groupby("fold").train_time.first()
    card = {"parameters": full.n_params(), "ensemble": len(full.seeds), "trees": full.n_trees, "train_time": f"{cv_times.mean():.1f} s per fold",
            "train_time_full_s": round(secs, 2), "epochs_full": len(full.history_),
            "predict_ms_per_district": round(pred.predict_time.mean() * 1000, 4),
            "hardware": f"{platform.processor() or platform.machine()} CPU, torch {torch.__version__}",
            "conformal_target": 0.90, "conformal_coverage_mean": round(conf.coverage.mean(), 3),
            "conformal_avg_set_size": round(conf.avg_set_size.mean(), 3)}
    (RES / "table4_model_card.json").write_text(json.dumps(card, indent=2))
    hist = pd.DataFrame(full.history_, columns=["train_loss", "val_loss"])
    hist.to_csv(RES / "acpn_training_curve.csv", index_label="epoch")
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
