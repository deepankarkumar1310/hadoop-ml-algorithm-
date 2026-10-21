"""M5: standard MLP (PyTorch) on the shared folds, scaled and raw (preprocessing before/after).

Usage: python scripts/train_mlp.py
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.mlp import MLPClassifier  # noqa: E402
from scripts.common import PRED, RES, get_folds, index_of, load_table  # noqa: E402


def main():
    df, X, y = load_table()
    folds = get_folds(df, y)
    idx = index_of(df)
    PRED.mkdir(parents=True, exist_ok=True)
    for setting, kinds in (("scaled", ("cv", "loso")), ("raw", ("cv",))):
        rows = []
        for kind in kinds:
            for f in folds[kind]:
                tr, te = idx(f["train"]), idx(f["test"])
                m = MLPClassifier(standardize=setting == "scaled").fit(X[tr], y[tr])
                t0 = time.perf_counter()
                p = m.predict_proba(X[te])[:, 1]
                pt = (time.perf_counter() - t0) / len(te)
                rows.append(pd.DataFrame({"censuscode": df.censuscode.iloc[te].values, "label": y[te], "score": p,
                                          "prob": p, "prediction": (p >= 0.5).astype(int), "fold": f["fold"],
                                          "kind": kind, "train_time": m.train_time_, "predict_time": pt}))
        pd.concat(rows).to_csv(PRED / f"Standard MLP__{setting}.csv", index=False)
        print("MLP", setting, "done")
    (RES / "mlp_params.txt").write_text(str(m.n_params()))


if __name__ == "__main__":
    main()
