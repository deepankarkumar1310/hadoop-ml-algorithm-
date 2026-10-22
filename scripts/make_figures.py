"""All figures, made only after every results table exists (Review 2 order rule).

Usage: python scripts/make_figures.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import viz_style as V  # noqa: E402
from scripts.common import FEATURES, HAZ, PRED, RES, ROOT, load_table, run_label  # noqa: E402
from scripts.evaluate import MODELS  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PatchCollection  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve  # noqa: E402

FIG = ROOT / "figures"
COL = dict(zip(MODELS, [V.ORANGE, "#104281", "#3987e5", V.AQUA, V.MUTED, "#a32323"]))
SHORT = {"B-ACPN (proposed)": "B-ACPN", "Logistic Regression": "LR", "Random Forest": "RF",
         "Gradient-Boosted Trees": "GBT", "Linear SVM": "Lin. SVM", "Standard MLP": "MLP"}
NICE = lambda s: s.replace("_", " ")  # noqa: E731


def preds(m, setting="scaled"):
    d = pd.read_csv(PRED / f"{m}__{setting}.csv")
    return d[d.kind == "cv"]


def preprocessing(df, X):
    """Sehjodh: retention, data quality, feature scaling, correlation, feature signal."""
    pm = json.loads((RES / "preprocessing_metrics.json").read_text())
    ar = json.loads((RES / "analysis_results.json").read_text())
    # P1 retention funnel
    fig, ax = plt.subplots(figsize=(7, 3.6))
    st = [("Census 2011 districts", 640), ("Matched to NDAP", pm["matching"]["matched"]),
          ("With vulnerability index", ar["preprocessing"]["with_index"]),
          ("Complete: all 13 features", pm["final_table"]["rows"])]
    ax.barh([s for s, _ in st][::-1], [n for _, n in st][::-1], color=[V.BLUE, "#6da7ec", "#9ec5f4", "#cde2fb"][::-1])
    for i, (_, n) in enumerate(st[::-1]):
        ax.text(n + 8, i, f"{n}  ({n / 640:.0%})", va="center", fontsize=10)
    ax.set_xlim(0, 760); ax.grid(axis="y", visible=False)
    ax.set_title("District retention through preprocessing (no imputation)")
    V.save(fig, FIG / "prep_retention.png")

    # P2 data quality before vs after
    planned = ar["preprocessing"]["missing_pct_planned_ndap"]
    labels = ["Missing cells (% of\ntable: NDAP vs model)", "Missing in planned dev.\nindicators (% avg)",
              "Unmatched NDAP\ndistricts (post-2011)", "Wrong fuzzy\nmatches"]
    before = [pm["ndap"]["overall_missing_pct"], round(sum(planned.values()) / len(planned), 1),
              pm["matching"]["match_types"].get("unmatched", 97), 2]
    # after: the model table has no missing cells, unmatched districts are excluded (never guessed), bad matches blocked
    after = [0, pm["final_table"]["feature_missing_cells"], 0, 0]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    x = np.arange(len(labels)); w = 0.38
    b1 = ax.bar(x - w / 2, before, w, color=V.RED, label="Before (raw sources)")
    b2 = ax.bar(x + w / 2, after, w, color=V.AQUA, label="After preprocessing")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1.5, f"{r.get_height():g}", ha="center", fontsize=9)
    ax.set_xticks(x, labels); ax.legend(); ax.grid(axis="x", visible=False)
    ax.set_title("Data-quality problems: before vs after preprocessing")
    V.save(fig, FIG / "prep_quality_before_after.png")

    # P3 feature scale: raw vs standardised
    z = (X - X.mean(0)) / X.std(0)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, data, t in ((axs[0], X, "Before: raw features (very different scales)"),
                        (axs[1], z, "After: standardised z-scores")):
        ax.boxplot(data, orientation="horizontal", tick_labels=[NICE(f) for f in FEATURES], showfliers=False,
                   patch_artist=True, boxprops=dict(facecolor="#cde2fb", color=V.BLUE), medianprops=dict(color=V.ORANGE))
        ax.set_title(t, fontsize=10)
    axs[0].set_xscale("symlog", linthresh=1)
    V.save(fig, FIG / "prep_feature_scaling.png")

    # P5 correlation heatmap, P6 feature signal
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    c = np.corrcoef(X.T)
    im = ax.imshow(c, cmap=V.DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(13), [NICE(f) for f in FEATURES], rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(13), [NICE(f) for f in FEATURES], fontsize=8); ax.grid(False)
    fig.colorbar(im, shrink=0.8); ax.set_title("Feature correlation (Pearson)")
    V.save(fig, FIG / "prep_correlation.png")
    sig = pm["feature_signal_auc"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    k = list(sig)[::-1]
    ax.barh([NICE(i) for i in k], [sig[i] for i in k], color=[V.ORANGE if i in HAZ else V.BLUE for i in k])
    ax.axvline(0.5, color=V.MUTED, ls=":"); ax.set_xlim(0.45, 0.72); ax.grid(axis="y", visible=False)
    ax.set_title("Feature signal: single-feature ROC-AUC (blue dev, orange hazard)")
    V.save(fig, FIG / "prep_feature_signal.png")


def preprocessing_gain():
    """Tanmay: model ROC-AUC on raw vs preprocessed features."""
    ba = pd.read_csv(RES / "preprocessing_before_after.csv")
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    x = np.arange(len(ba))
    ax.bar(x - 0.19, ba.auc_before, 0.38, color=V.RED, label="Before: raw features")
    ax.bar(x + 0.19, ba.auc_after, 0.38, color=V.AQUA, label="After: preprocessed")
    for i, r in ba.iterrows():
        ax.text(i + 0.19, r.auc_after + 0.01, f"{round(r.auc_gain, 3) + 0.0:+.3f}", ha="center", fontsize=9, color=V.INK)
    ax.set_xticks(x, [SHORT[m] for m in ba.model]); ax.set_ylim(0.4, 0.95); ax.axhline(0.5, color=V.MUTED, lw=0.8, ls=":")
    ax.set_ylabel("ROC-AUC (25 folds)"); ax.legend(loc="upper right"); ax.grid(axis="x", visible=False)
    ax.set_title("Model ROC-AUC before vs after preprocessing (label = gain)")
    V.save(fig, FIG / "prep_model_gain.png")


def classifier(df):
    """Tanmay: ROC/PR, confusion, comparison, stability, calibration, LOSO. Devanshu: training curve, ablation."""
    t5 = pd.read_csv(RES / "table5_comparison.csv")
    fm = pd.read_csv(RES / "fold_metrics.csv")
    # C1 ROC + C2 PR
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
    for m in MODELS:
        d = preds(m)
        fpr, tpr, _ = roc_curve(d.label, d.score)
        pr, rc, _ = precision_recall_curve(d.label, d.score)
        lw = 3 if m.startswith("B-ACPN") else 1.6
        axs[0].plot(fpr, tpr, color=COL[m], lw=lw, label=f"{SHORT[m]} (AUC {roc_auc_score(d.label, d.score):.3f})")
        axs[1].plot(rc, pr, color=COL[m], lw=lw, label=SHORT[m])
    axs[0].plot([0, 1], [0, 1], color=V.MUTED, ls=":", lw=1)
    axs[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves (pooled 25 folds)")
    axs[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves")
    axs[0].legend(loc="lower right", fontsize=8); axs[1].legend(loc="lower left", fontsize=8)
    V.save(fig, FIG / "roc_curves.png")

    # C3 confusion matrices
    cm = pd.read_csv(RES / "confusion_matrices.csv").set_index("model")
    fig, axs = plt.subplots(2, 3, figsize=(11.5, 6.8), gridspec_kw=dict(wspace=0.45, hspace=0.35))
    for ax, m in zip(axs.flat, MODELS):
        r = cm.loc[m]; mat = np.array([[r.tn, r.fp], [r.fn, r.tp]])
        ax.imshow(mat, cmap=V.SEQUENTIAL); ax.grid(False)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{mat[i, j]}", ha="center", va="center", fontsize=12,
                        color="white" if mat[i, j] > mat.max() * 0.6 else V.INK)
        ax.set_xticks([0, 1], ["Pred Low", "Pred High"]); ax.set_yticks([0, 1], ["True Low", "True High"])
        ax.set_title(SHORT[m], color=V.ORANGE if m.startswith("B-ACPN") else V.INK)
    fig.suptitle("Confusion matrices (summed over 25 test folds)", x=0.02, ha="left", fontweight="bold")
    V.save(fig, FIG / "confusion_matrices.png")
    r = cm.loc["B-ACPN (proposed)"]; mat = np.array([[r.tn, r.fp], [r.fn, r.tp]])
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(mat, cmap=V.SEQUENTIAL); ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{mat[i, j]}\n({mat[i, j] / mat.sum():.0%})", ha="center", va="center", fontsize=13,
                    color="white" if mat[i, j] > mat.max() * 0.6 else V.INK)
    ax.set_xticks([0, 1], ["Predicted Low", "Predicted High"]); ax.set_yticks([0, 1], ["True Low", "True High"])
    ax.set_title("B-ACPN confusion matrix (25 folds)")
    V.save(fig, FIG / "confusion_acpn.png")

    # C4 comparison bars with error bars + C5 box plots
    fig, ax = plt.subplots(figsize=(10, 4.2))
    mets = [("roc_auc", "ROC-AUC"), ("bal_acc", "Balanced acc."), ("accuracy", "Accuracy"), ("f1", "F1")]
    x = np.arange(len(mets)); w = 0.13
    for i, m in enumerate(MODELS):
        r = t5[t5.model == m].iloc[0]
        ax.bar(x + (i - 2.5) * w, [r[c] for c, _ in mets], w, yerr=[r[f"{c}_sd"] for c, _ in mets],
               color=COL[m], label=SHORT[m], capsize=2, error_kw=dict(lw=0.8, ecolor=V.INK_2))
    ax.set_xticks(x, [l for _, l in mets]); ax.set_ylim(0.5, 0.95); ax.grid(axis="x", visible=False)
    ax.legend(ncol=6, loc="upper center", fontsize=8); ax.set_title("Model comparison: mean ± SD over 25 folds")
    V.save(fig, FIG / "comparison_bars.png")
    fig, ax = plt.subplots(figsize=(8, 4))
    data = [fm[fm.model == m].roc_auc for m in MODELS]
    bp = ax.boxplot(data, tick_labels=[SHORT[m] for m in MODELS], patch_artist=True, medianprops=dict(color=V.INK))
    for p, m in zip(bp["boxes"], MODELS):
        p.set_facecolor(COL[m]); p.set_alpha(0.75)
    ax.set_ylabel("ROC-AUC per fold"); ax.set_title("Fold-to-fold stability (25 folds)"); ax.grid(axis="x", visible=False)
    V.save(fig, FIG / "fold_boxplots.png")

    # C6 calibration
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for m in MODELS:
        d = preds(m)
        if d.prob.isna().any():
            continue
        fp, mp = calibration_curve(d.label, d.prob, n_bins=8, strategy="quantile")
        ax.plot(mp, fp, marker="o", ms=3, color=COL[m], lw=3 if m.startswith("B-ACPN") else 1.4, label=SHORT[m])
    ax.plot([0, 1], [0, 1], color=V.MUTED, ls=":")
    ax.set(xlabel="Predicted probability", ylabel="Observed share High", title="Calibration (Linear SVM has no probabilities)")
    ax.legend(fontsize=8)
    V.save(fig, FIG / "calibration.png")

    # C7 LOSO by state
    ls = pd.read_csv(RES / "loso_by_state.csv")
    piv = ls.pivot(index="state", columns="model", values="accuracy")
    order = piv["B-ACPN (proposed)"].sort_values().index
    fig, ax = plt.subplots(figsize=(8, 6))
    best = t5[~t5.model.str.startswith("B-ACPN")].sort_values("loso_auc").model.iloc[-1]
    y = np.arange(len(order))
    ax.barh(y + 0.2, piv.loc[order, "B-ACPN (proposed)"], 0.4, color=V.ORANGE, label="B-ACPN")
    ax.barh(y - 0.2, piv.loc[order, best], 0.4, color=COL[best], label=SHORT[best])
    n = ls[ls.model == best].set_index("state").n
    ax.set_yticks(y, [f"{s} (n={n[s]})" for s in order], fontsize=8)
    ax.set_xlabel("Accuracy on the held-out state"); ax.legend(); ax.grid(axis="y", visible=False)
    ax.set_title("Leave-one-state-out: accuracy per unseen state")
    V.save(fig, FIG / "loso_by_state.png")

    # C8 training curve, C9 ablation
    h = pd.read_csv(RES / "acpn_training_curve.csv")
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(h.epoch, h.train_loss, color=V.BLUE, label="Training loss")
    ax.plot(h.epoch, h.val_loss, color=V.ORANGE, label="Validation loss")
    best_ep = h.val_loss.idxmin()
    ax.axvline(best_ep, color=V.MUTED, ls=":"); ax.text(best_ep + 1, h.val_loss.max(), f"best epoch {best_ep}", fontsize=8)
    ax.set(xlabel="Epoch", ylabel="Weighted BCE", title="B-ACPN training curve (first network, early stopping)"); ax.legend()
    V.save(fig, FIG / "acpn_training_curve.png")
    ab = pd.read_csv(RES / "table6_ablation.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(ab.model[::-1], ab.roc_auc[::-1], xerr=ab.roc_auc_sd[::-1], capsize=3,
            color=[V.ORANGE if m == "Full B-ACPN" else "#9ec5f4" for m in ab.model[::-1]])
    for i, (v, s) in enumerate(zip(ab.roc_auc[::-1], ab.roc_auc_sd[::-1])):
        ax.text(v + s + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0.5, 0.95); ax.set_xlabel("ROC-AUC (25 folds)"); ax.grid(axis="y", visible=False)
    ax.set_title("Ablation: removing one B-ACPN component at a time")
    V.save(fig, FIG / "ablation.png")


def explanation(df):
    """Devanshu: climate push by group, attention, India map."""
    e = pd.read_csv(RES / "district_explanations.csv")
    # climate push by quadrant
    order = ["High dev / Low vul", "Low dev / High vul", "High dev / High vul", "Low dev / Low vul"]
    names = ["Expected\n(high dev, low vul)", "Expected\n(low dev, high vul)", "Anomaly A\n(developed, vulnerable)",
             "Anomaly B\n(less dev, resilient)"]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    rng = np.random.default_rng(0)
    for i, q in enumerate(order):
        v = e[e.quadrant == q].push
        s = V.QUADRANT_STYLE[q]
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=10, alpha=0.45, color=s["color"], marker=s["marker"])
        ax.plot([i - 0.3, i + 0.3], [v.mean()] * 2, color=V.INK, lw=2.5)
        ax.text(i + 0.33, v.mean(), f"{v.mean():+.2f}", va="center", fontsize=9, fontweight="bold")
    ax.axhline(0, color=V.MUTED, lw=1)
    ax.set_xticks(range(4), names, fontsize=9); ax.set_ylabel("Climate push (log-odds)")
    ax.set_title("B-ACPN climate push by group (bar = mean)"); ax.grid(axis="x", visible=False)
    V.save(fig, FIG / "climate_push_by_quadrant.png")

    # attention importance
    att = e[[f"att_{h}" for h in HAZ]].mean().sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.barh([NICE(a.replace("att_", "")) for a in att.index], att.values, color=V.ORANGE)
    ax.axvline(1 / 8, color=V.MUTED, ls=":"); ax.grid(axis="y", visible=False)
    ax.set_title("B-ACPN feature attention (mean weight; dotted = equal)")
    V.save(fig, FIG / "attention_importance.png")

    # India map of the climate push
    import shapefile
    sf = shapefile.Reader(str(ROOT / "data" / "raw" / "boundaries" / "2011_Dist.shp"))
    fields = [f[0] for f in sf.fields[1:]]
    key = next(f for f in fields if f.lower() in ("censuscode", "censuscod", "dist_cen", "c_code11"))
    push = e.set_index("censuscode").push
    lim = float(np.nanpercentile(np.abs(push), 95))
    fig, ax = plt.subplots(figsize=(7, 7.6))
    polys, vals, grey = [], [], []
    for rec, shp in zip(sf.records(), sf.shapes()):
        code = int(rec[fields.index(key)])
        pts = np.array(shp.points)
        parts = list(shp.parts) + [len(pts)]
        for a, b in zip(parts[:-1], parts[1:]):
            p = Polygon(pts[a:b], closed=True)
            (polys.append(p), vals.append(push[code])) if code in push.index else grey.append(p)
    ax.add_collection(PatchCollection(grey, facecolor="#ebebe8", edgecolor="white", lw=0.2))
    pc = PatchCollection(polys, cmap=V.DIVERGING, edgecolor="white", lw=0.2)
    pc.set_array(np.array(vals)); pc.set_clim(-lim, lim)
    ax.add_collection(pc); ax.autoscale_view(); ax.set_aspect("equal"); ax.axis("off")
    cb = fig.colorbar(pc, shrink=0.6); cb.set_label("Climate push (red = climate adds risk)")
    ax.set_title("B-ACPN climate push per district (grey = not in the 469)")
    V.save(fig, FIG / "climate_push_map.png")


def platform():
    """Deepankar: HDFS storage by zone (Table 2)."""
    hd = json.loads((RES / "hdfs_usage_bytes.json").read_text())
    fig, ax = plt.subplots(figsize=(7, 3.4))
    k = sorted(hd, key=hd.get)
    ax.barh(k, [hd[i] / 1e6 for i in k], color=V.BLUE)
    for i, z in enumerate(k):
        ax.text(hd[z] / 1e6 + 8, i, f"{hd[z] / 1e6:,.1f} MB", va="center", fontsize=9)
    ax.set_xlim(0, 900); ax.grid(axis="y", visible=False); ax.set_title(f"HDFS storage by zone ({run_label()})")
    V.save(fig, FIG / "hdfs_storage.png")


def processing():
    """Deepankar: MapReduce scaling and Spark k-means silhouette."""
    t = pd.read_csv(RES / "scalability_timings.csv").groupby("years").agg(rt=("runtime_s", "mean"), rec=("map_input_records", "first"))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(t.rec / 1e6, t.rt, marker="o", color=V.ORANGE)
    for yrs, r in t.iterrows():
        ax.text(r.rec / 1e6 + 0.15, r.rt, f"{yrs} yr: {r.rt:.0f} s", fontsize=9, va="center")
    ax.set(xlabel="District-day records (millions)", ylabel="Runtime (s)",
           title=f"MapReduce scaling, single node ({run_label()})")
    V.save(fig, FIG / "mapreduce_scalability.png")
    km = json.loads((RES / "kmeans_regimes.json").read_text())
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ks = sorted(km["silhouette_by_k"], key=int)
    ax.plot([int(k) for k in ks], [km["silhouette_by_k"][k] for k in ks], marker="o", color=V.BLUE)
    ax.scatter([km["k"]], [km["silhouette"]], s=90, color=V.ORANGE, zorder=3)
    ax.set(xlabel="k", ylabel="Silhouette", title=f"Spark k-means: chosen k = {km['k']}")
    V.save(fig, FIG / "kmeans_silhouette.png")


def main():
    V.apply()
    FIG.mkdir(exist_ok=True)
    df, X, _ = load_table()
    preprocessing(df, X)       # Sehjodh Grewal
    platform()                 # Deepankar
    processing()               # Deepankar
    preprocessing_gain()       # Tanmay Singh
    classifier(df)             # Tanmay Singh (comparison) + Devanshu Kumar (training curve, ablation)
    explanation(df)            # Devanshu Kumar


if __name__ == "__main__":
    main()
