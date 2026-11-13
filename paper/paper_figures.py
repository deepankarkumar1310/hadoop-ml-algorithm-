"""Figures for the IEEE paper, drawn at their printed size so every label is readable at 100 % zoom.

Each plot is drawn directly at the paper's column width (3.5 in) with 7.5-9 pt text, so the text prints at that
size instead of being shrunk from a slide-sized figure. Two-panel figures (ROC and precision-recall) are two small
panels placed side by side with one shared legend. The data and colours are the same as in figures/.

The dashboard snapshot is taken from dashboard/index.html in a 900 px window and printed across the page width.

Writes paper/figs/*.png.   Usage: python paper/paper_figures.py
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.make_figures as MF  # noqa: E402
from scripts.common import HAZ, RES, ROOT  # noqa: E402

plt, V = MF.plt, MF.V
OUT = Path(__file__).resolve().parent / "figs"
W = 3.5  # IEEE column width, inches
NAMES = {"annual_rain_mm": "Annual rain", "rain_variability": "Rain variability", "rx5day_mm": "Max 5-day rain",
         "heavy_rain_days": "Heavy-rain days", "max_dry_spell": "Longest dry spell", "mean_tmax": "Mean Tmax",
         "hot_days": "Hot days", "tropical_nights": "Tropical nights"}


def style():
    V.apply()
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                         "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "lines.linewidth": 1.2,
                         "axes.titlelocation": "left"})


def save(fig, name):
    fig.savefig(OUT / name, dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("figure ->", OUT / name)


def roc_pr():
    from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.0), gridspec_kw=dict(wspace=0.38))
    handles = []
    for m in MF.MODELS:
        d = MF.preds(m)
        fpr, tpr, _ = roc_curve(d.label, d.score)
        pr, rc, _ = precision_recall_curve(d.label, d.score)
        lw = 2.0 if m.startswith("B-ACPN") else 1.0
        h, = ax[0].plot(fpr, tpr, color=MF.COL[m], lw=lw, label=f"{MF.SHORT[m]} (AUC {roc_auc_score(d.label, d.score):.3f})")
        ax[1].plot(rc, pr, color=MF.COL[m], lw=lw)
        handles.append(h)
    ax[0].plot([0, 1], [0, 1], color=V.MUTED, ls=":", lw=0.8)
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall curves")
    for a in ax:
        a.set_xticks([0, 0.5, 1]); a.set_yticks([0, 0.5, 1])
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.5), frameon=False,
               columnspacing=1.2, handlelength=1.8)
    save(fig, "roc_curves.png")


def push_groups():
    e = pd.read_csv(RES / "district_explanations.csv")
    order = ["High dev / Low vul", "Low dev / High vul", "High dev / High vul", "Low dev / Low vul"]
    names = ["Expected\nhigh dev,\nlow vul", "Expected\nlow dev,\nhigh vul", "Anomaly A\ndeveloped,\nvulnerable",
             "Anomaly B\nless dev.,\nresilient"]
    fig, ax = plt.subplots(figsize=(W, 2.5), layout="constrained")
    rng = np.random.default_rng(0)
    for i, q in enumerate(order):
        v = e[e.quadrant == q].push
        s = V.QUADRANT_STYLE[q]
        ax.scatter(i + rng.uniform(-0.16, 0.16, len(v)), v, s=5, alpha=0.5, color=s["color"], marker=s["marker"],
                   linewidths=0)
        ax.plot([i - 0.28, i + 0.28], [v.mean()] * 2, color=V.INK, lw=1.8)
        ax.text(i + 0.3, v.mean(), f"{v.mean():+.2f}", va="center", fontsize=8, fontweight="bold")
    ax.axhline(0, color=V.MUTED, lw=0.8)
    ax.set_xticks(range(4), names)
    ax.set_xlim(-0.5, 3.75)
    ax.set_ylabel("Climate push (log-odds)")
    ax.set_title("B-ACPN climate push by group (bar = mean)")
    ax.grid(axis="x", visible=False)
    save(fig, "climate_push_by_quadrant.png")


def bars():
    t5 = pd.read_csv(RES / "table5_comparison.csv")
    fig, ax = plt.subplots(figsize=(W, 2.4), layout="constrained")
    mets = [("roc_auc", "ROC-AUC"), ("bal_acc", "Bal. acc."), ("accuracy", "Accuracy"), ("f1", "F1")]
    x = np.arange(len(mets)); w = 0.13
    for i, m in enumerate(MF.MODELS):
        r = t5[t5.model == m].iloc[0]
        ax.bar(x + (i - 2.5) * w, [r[c] for c, _ in mets], w, yerr=[r[f"{c}_sd"] for c, _ in mets],
               color=MF.COL[m], label=MF.SHORT[m], capsize=1.2, error_kw=dict(lw=0.6, ecolor=V.INK_2))
    ax.set_xticks(x, [l for _, l in mets]); ax.set_ylim(0.5, 0.95); ax.grid(axis="x", visible=False)
    ax.set_title("Model comparison: mean ± SD over 25 folds")
    ax.legend(ncol=3, loc="upper right", frameon=False, columnspacing=0.8, handlelength=1.2)
    save(fig, "comparison_bars.png")


def folds():
    fm = pd.read_csv(RES / "fold_metrics.csv")
    fig, ax = plt.subplots(figsize=(W, 2.1), layout="constrained")
    data = [fm[fm.model == m].roc_auc for m in MF.MODELS]
    bp = ax.boxplot(data, tick_labels=[MF.SHORT[m] for m in MF.MODELS], patch_artist=True,
                    medianprops=dict(color=V.INK), flierprops=dict(markersize=3))
    for p, m in zip(bp["boxes"], MF.MODELS):
        p.set_facecolor(MF.COL[m]); p.set_alpha(0.75)
    ax.set_ylabel("ROC-AUC per fold"); ax.set_title("Fold-to-fold stability (25 folds)")
    ax.grid(axis="x", visible=False)
    save(fig, "fold_boxplots.png")


def ablation():
    ab = pd.read_csv(RES / "table6_ablation.csv")
    fig, ax = plt.subplots(figsize=(W, 2.8), layout="constrained")
    labels = [m.replace(" (", "\n(") for m in ab.model[::-1]]
    ax.barh(labels, ab.roc_auc[::-1], xerr=ab.roc_auc_sd[::-1], capsize=2, error_kw=dict(lw=0.7),
            color=[V.ORANGE if m == "Full B-ACPN" else "#9ec5f4" for m in ab.model[::-1]])
    for i, (v, s) in enumerate(zip(ab.roc_auc[::-1], ab.roc_auc_sd[::-1])):
        ax.text(v + s + 0.005, i, f"{v:.3f}", va="center", fontsize=7.5)
    ax.set_xlim(0.5, 0.95); ax.set_xlabel("ROC-AUC (25 folds)"); ax.grid(axis="y", visible=False)
    ax.set_title("Ablation of B-ACPN")
    save(fig, "ablation.png")


def attention():
    e = pd.read_csv(RES / "district_explanations.csv")
    att = e[[f"att_{h}" for h in HAZ]].mean().sort_values()
    fig, ax = plt.subplots(figsize=(W, 2.2), layout="constrained")
    ax.barh([NAMES[a.replace("att_", "")] for a in att.index], att.values, color=V.ORANGE)
    for i, v in enumerate(att.values):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=7.5)
    ax.axvline(1 / 8, color=V.MUTED, ls=":"); ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 0.165); ax.set_xlabel("Mean attention weight")
    ax.set_title("B-ACPN attention (dotted = equal)")
    save(fig, "attention_importance.png")


def push_map():
    import shapefile
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon
    e = pd.read_csv(RES / "district_explanations.csv")
    sf = shapefile.Reader(str(ROOT / "data" / "raw" / "boundaries" / "2011_Dist.shp"))
    fields = [f[0] for f in sf.fields[1:]]
    key = next(f for f in fields if f.lower() in ("censuscode", "censuscod", "dist_cen", "c_code11"))
    push = e.set_index("censuscode").push
    lim = float(np.nanpercentile(np.abs(push), 95))
    fig, ax = plt.subplots(figsize=(W, 3.3))
    polys, vals, grey = [], [], []
    for rec, shp in zip(sf.records(), sf.shapes()):
        code = int(rec[fields.index(key)])
        pts = np.array(shp.points)
        parts = list(shp.parts) + [len(pts)]
        for a, b in zip(parts[:-1], parts[1:]):
            p = Polygon(pts[a:b], closed=True)
            (polys.append(p), vals.append(push[code])) if code in push.index else grey.append(p)
    ax.add_collection(PatchCollection(grey, facecolor="#ebebe8", edgecolor="white", lw=0.15))
    pc = PatchCollection(polys, cmap=V.DIVERGING, edgecolor="white", lw=0.15)
    pc.set_array(np.array(vals)); pc.set_clim(-lim, lim)
    ax.add_collection(pc); ax.autoscale_view(); ax.set_aspect("equal"); ax.axis("off")
    cb = fig.colorbar(pc, shrink=0.75, pad=0.02)
    cb.set_label("Climate push (red = adds risk)", fontsize=8); cb.ax.tick_params(labelsize=7.5)
    ax.set_title("Climate push per district (grey = not in the 469)")
    save(fig, "climate_push_map.png")


def confusion_column():
    """The six confusion matrices in 3 rows x 2 columns; True/Pred labels only on the outer edges."""
    cm = pd.read_csv(RES / "confusion_matrices.csv").set_index("model")
    fig, axs = plt.subplots(3, 2, figsize=(W, 5.0), gridspec_kw=dict(wspace=0.06, hspace=0.32))
    for k, (ax, m) in enumerate(zip(axs.flat, MF.MODELS)):
        r = cm.loc[m]; mat = np.array([[r.tn, r.fp], [r.fn, r.tp]])
        ax.imshow(mat, cmap=V.SEQUENTIAL); ax.grid(False)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{mat[i, j]}", ha="center", va="center", fontsize=9,
                        color="white" if mat[i, j] > mat.max() * 0.6 else V.INK)
        ax.set_xticks([0, 1], ["Pred\nLow", "Pred\nHigh"] if k >= 4 else ["", ""], fontsize=7.5)
        ax.set_yticks([0, 1], ["True\nLow", "True\nHigh"] if k % 2 == 0 else ["", ""], fontsize=7.5)
        ax.tick_params(length=0)
        ax.set_title(MF.SHORT[m], fontsize=8.5, fontweight="bold", pad=3,
                     color=V.ORANGE if m.startswith("B-ACPN") else V.INK)
    save(fig, "confusion_matrices_column.png")


def dashboard():
    """Same page and layout as figures/dashboard.png, in a 900 px window; printed across the page width."""
    url = (ROOT / "dashboard" / "index.html").as_uri() + "?d=Kishanganj"
    js = f"""
const {{ chromium }} = require('playwright');
(async () => {{
  const b = await chromium.launch();
  const pg = await b.newPage({{ viewport: {{ width: 900, height: 700 }}, deviceScaleFactor: 3 }});
  await pg.goto({json.dumps(url)});
  await pg.waitForTimeout(300);
  const box = await pg.locator('.grid').boundingBox();
  await pg.screenshot({{ path: {json.dumps(str(OUT / "dashboard_paper.png"))},
                        clip: {{ x: 0, y: 0, width: 900, height: box.y + box.height + 12 }} }});
  await b.close();
}})();
"""
    subprocess.run(["node", "-e", js], check=True)
    print("figure ->", OUT / "dashboard_paper.png")


def main():
    OUT.mkdir(exist_ok=True)
    for old in ("roc_paper.png", "push_paper.png", "confusion_matrices.png", "calibration.png"):
        (OUT / old).unlink(missing_ok=True)
    style()
    roc_pr(); push_groups(); bars(); folds(); ablation(); attention(); push_map(); confusion_column()
    dashboard()


if __name__ == "__main__":
    main()
