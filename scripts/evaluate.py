"""Turn the out-of-fold predictions into every comparison table (Tables 5 and 6, full metrics,
confusion matrices, confidence intervals, significance, preprocessing before/after, climate push).

Usage: python scripts/evaluate.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             cohen_kappa_score, confusion_matrix, f1_score, log_loss, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.common import HAZ, PRED, RES, ROOT, load_table  # noqa: E402
from scripts.train_acpn import VARIANTS  # noqa: E402

MODELS = ["B-ACPN (proposed)", "Logistic Regression", "Random Forest", "Gradient-Boosted Trees", "Linear SVM", "Standard MLP"]
TOOL = {"B-ACPN (proposed)": "Novel hybrid DL · PyTorch", "Standard MLP": "PyTorch"}
N, K = 469, 5  # districts, folds per repeat (for the corrected resampled t-test)


def ece(y, p, bins=10):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def fold_metrics(d):
    y, s, yhat = d.label.to_numpy().astype(int), d.score.to_numpy(), d.prediction.to_numpy().astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    m = {"roc_auc": roc_auc_score(y, s), "pr_auc": average_precision_score(y, s), "accuracy": accuracy_score(y, yhat),
         "bal_acc": balanced_accuracy_score(y, yhat), "precision": precision_score(y, yhat, zero_division=0),
         "recall": recall_score(y, yhat), "specificity": tn / (tn + fp), "f1": f1_score(y, yhat),
         "macro_f1": f1_score(y, yhat, average="macro"), "mcc": matthews_corrcoef(y, yhat),
         "kappa": cohen_kappa_score(y, yhat), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
         "train_time": d.train_time.iloc[0], "predict_time": d.predict_time.iloc[0]}
    if d.prob.notna().all():
        p = d.prob.to_numpy().clip(1e-7, 1 - 1e-7)
        m.update(log_loss=log_loss(y, p, labels=[0, 1]), brier=brier_score_loss(y, p), ece=ece(y, p))
    return m


def per_fold(path):
    d = pd.read_csv(path)
    cv = d[d.kind == "cv"]
    return pd.DataFrame([{"fold": f, **fold_metrics(g)} for f, g in cv.groupby("fold")]), d


def loso_auc(d):
    lo = d[d.kind == "loso"]
    return roc_auc_score(lo.label, lo.score) if len(lo) else np.nan


def corrected_ci_and_p(a, b=None):
    """Nadeau-Bengio corrected resampled t: CI of mean(a), or paired test of a - b."""
    x = np.asarray(a) - (0 if b is None else np.asarray(b))
    J = len(x)
    var = np.var(x, ddof=1) * (1 / J + (1 / K) / (1 - 1 / K))
    se = np.sqrt(var)
    tcrit = stats.t.ppf(0.975, J - 1)
    if b is None:
        return x.mean() - tcrit * se, x.mean() + tcrit * se
    t = x.mean() / se if se > 0 else 0.0
    return 2 * stats.t.sf(abs(t), J - 1)


def main():
    folds_m, preds = {}, {}
    for m in MODELS:
        folds_m[m], preds[m] = per_fold(PRED / f"{m}__scaled.csv")

    # Table 5 + full metrics + confusion + significance
    t5, full, conf, sig = [], [], [], []
    ref = folds_m["B-ACPN (proposed)"].sort_values("fold").roc_auc.to_numpy()
    for m in MODELS:
        f = folds_m[m].sort_values("fold")
        mean, sd = f.mean(numeric_only=True), f.std(numeric_only=True)
        t5.append({"model": m, "tool": TOOL.get(m, "Spark MLlib"),
                   **{c: mean[c] for c in ("roc_auc", "bal_acc", "accuracy", "f1")},
                   **{f"{c}_sd": sd[c] for c in ("roc_auc", "bal_acc", "accuracy", "f1")},
                   "loso_auc": loso_auc(preds[m]), "train_time": mean.train_time})
        row = {"model": m}
        for c in ("precision", "recall", "specificity", "mcc", "kappa", "pr_auc", "log_loss", "brier", "ece",
                  "macro_f1", "predict_time"):
            row[c] = mean.get(c, np.nan)
            row[f"{c}_sd"] = sd.get(c, np.nan)
        full.append(row)
        conf.append({"model": m, **{c: int(f[c].sum()) for c in ("tp", "tn", "fp", "fn")}})
        lo, hi = corrected_ci_and_p(f.roc_auc)
        p = np.nan if m.startswith("B-ACPN") else corrected_ci_and_p(ref, f.roc_auc.to_numpy())
        sig.append({"model": m, "auc_mean": f.roc_auc.mean(), "auc_ci": f"{lo:.3f} – {hi:.3f}",
                    "ci_low": lo, "ci_high": hi, "p_value": p})
    pd.DataFrame(t5).to_csv(RES / "table5_comparison.csv", index=False)
    pd.DataFrame(full).to_csv(RES / "classifier_metrics_full.csv", index=False)
    pd.DataFrame(conf).to_csv(RES / "confusion_matrices.csv", index=False)
    sg = pd.DataFrame(sig)
    sg["p_value_fmt"] = sg.p_value.map(lambda v: "—" if pd.isna(v) else ("< 0.001" if v < 0.001 else f"{v:.3f}"))
    sg.to_csv(RES / "significance.csv", index=False)
    pd.concat([f.assign(model=m) for m, f in folds_m.items()]).to_csv(RES / "fold_metrics.csv", index=False)

    # LOSO per state
    df, _, _ = load_table()
    st = df.set_index("censuscode").state_name
    ls = []
    for m in MODELS:
        lo = preds[m][preds[m].kind == "loso"].copy()
        lo["state"] = lo.censuscode.map(st)
        for s, g in lo.groupby("state"):
            ls.append({"model": m, "state": s, "n": len(g),
                       "auc": roc_auc_score(g.label, g.score) if g.label.nunique() == 2 else np.nan,
                       "accuracy": accuracy_score(g.label, g.prediction)})
    pd.DataFrame(ls).to_csv(RES / "loso_by_state.csv", index=False)

    # Ablation (Table 6a)
    full_auc = folds_m["B-ACPN (proposed)"].roc_auc
    ab = [{"model": "Full B-ACPN", "roc_auc": full_auc.mean(), "roc_auc_sd": full_auc.std(),
           "f1": folds_m["B-ACPN (proposed)"].f1.mean(), "delta": 0.0}]
    for v in VARIANTS:
        if v == "Full B-ACPN":
            continue
        f, _ = per_fold(PRED / f"ablation__{v}.csv")
        ab.append({"model": v, "roc_auc": f.roc_auc.mean(), "roc_auc_sd": f.roc_auc.std(), "f1": f.f1.mean(),
                   "delta": f.roc_auc.mean() - full_auc.mean(),
                   "p_value": corrected_ci_and_p(full_auc.to_numpy(), f.sort_values("fold").roc_auc.to_numpy())})
    pd.DataFrame(ab).to_csv(RES / "table6_ablation.csv", index=False)

    # Preprocessing before/after: raw vs preprocessed features, same folds
    ba = []
    for m in MODELS:
        raw, _ = per_fold(PRED / f"{m}__raw.csv")
        aft = folds_m[m]
        ba.append({"model": m, "auc_before": raw.roc_auc.mean(), "auc_after": aft.roc_auc.mean(),
                   "f1_before": raw.f1.mean(), "f1_after": aft.f1.mean(),
                   "acc_before": raw.accuracy.mean(), "acc_after": aft.accuracy.mean()})
    ba = pd.DataFrame(ba)
    ba["auc_gain"] = ba.auc_after - ba.auc_before
    ba["f1_gain"] = ba.f1_after - ba.f1_before
    ba.to_csv(RES / "preprocessing_before_after.csv", index=False)

    # Climate push analysis (out-of-fold, averaged over the 5 repeats)
    ex = pd.read_csv(RES / "district_explanations_folds.csv")
    att_cols = [f"att_{h}" for h in HAZ]
    e = ex.groupby("censuscode")[["baseline", "push", "push_mc_sd", *att_cols]].mean()
    e["top_factor"] = e[att_cols].idxmax(axis=1).str.replace("att_", "")
    e["climate_share"] = e.push.abs() / (e.push.abs() + e.baseline.abs())
    reg_p = ROOT / "data" / "processed" / "climate_regimes.csv"
    info = df.set_index("censuscode")[["district_name", "state_name", "quadrant", "vul_group"]]
    e = info.join(e)
    if reg_p.exists():
        e = e.join(pd.read_csv(reg_p).set_index("censuscode")[["regime"]])
    e.reset_index().to_csv(RES / "district_explanations.csv", index=False)
    lab = lambda r: f"{r.district_name} ({r.state_name})"  # noqa: E731
    by_q = {q: {"n": int(len(g)), "push": float(g.push.mean()), "push_sd": float(g.push.std()),
                "share_positive": float((g.push > 0).mean()),
                "factor": g.top_factor.mode().iat[0].replace("_", " ")} for q, g in e.groupby("quadrant")}
    by_r = {}
    if "regime" in e:
        by_r = {int(r): {"n": int(len(g)), "push": float(g.push.mean()),
                         "factor": g.top_factor.mode().iat[0].replace("_", " ")} for r, g in e.groupby("regime")}
    top_c = e.sort_values("push", ascending=False).head(5)
    top_d = e[e.push.abs() < e.baseline.abs()].sort_values("climate_share").head(5)
    a_vs = {"anomaly_A_mean_push": by_q["High dev / High vul"]["push"],
            "expected_high_dev_low_vul_mean_push": by_q["High dev / Low vul"]["push"],
            "anomaly_B_mean_push": by_q["Low dev / Low vul"]["push"],
            "expected_low_dev_high_vul_mean_push": by_q["Low dev / High vul"]["push"]}
    tA = stats.ttest_ind(e[e.quadrant == "High dev / High vul"].push, e[e.quadrant == "High dev / Low vul"].push)
    tB = stats.ttest_ind(e[e.quadrant == "Low dev / Low vul"].push, e[e.quadrant == "Low dev / High vul"].push)
    push = {"by_quadrant": by_q, "by_regime": by_r, **a_vs,
            "anomaly_A_vs_expected_p": float(tA.pvalue), "anomaly_B_vs_expected_p": float(tB.pvalue),
            "top_climate": [f"{lab(r)} {r.push:+.2f}" for r in top_c.itertuples()],
            "top_development": [f"{lab(r)} {r.climate_share:.0%}" for r in top_d.itertuples()],
            "mean_push_mc_sd": float(e.push_mc_sd.mean()),
            "attention_mean": {h: float(e[f"att_{h}"].mean()) for h in HAZ}}
    (RES / "table6_climate_push.json").write_text(json.dumps(push, indent=2))
    print(pd.DataFrame(t5).round(3).to_string())
    print(pd.DataFrame(ab).round(3).to_string())
    print(ba.round(3).to_string())
    print(json.dumps({k: v for k, v in push.items() if k not in ("attention_mean",)}, indent=1, default=str)[:2500])


if __name__ == "__main__":
    main()
