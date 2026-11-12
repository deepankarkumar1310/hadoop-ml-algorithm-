"""Figures sized for the IEEE paper: drawn at the column width (3.5 in) so the text prints at 7-9 pt.

The slide and report figures in figures/ are drawn for a full page and shrink to unreadable text in a
paper column, so the three image figures of the paper are redrawn here from the same results:
  figs/roc_paper.png        ROC and precision-recall curves (Fig. 3)
  figs/push_paper.png       climate push by group (Fig. 4)
  figs/dashboard_paper.png  dashboard snapshot, district view (Fig. 5), taken from dashboard/index.html
Usage: python paper/paper_figures.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import viz_style as V  # noqa: E402
from scripts.common import RES, ROOT  # noqa: E402
from scripts.make_figures import COL, SHORT, preds  # noqa: E402
from scripts.evaluate import MODELS  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve  # noqa: E402

OUT = Path(__file__).resolve().parent / "figs"
W = 3.5  # IEEE column width in inches


def style():
    V.apply()
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 7.5, "lines.linewidth": 1.4})


def save(fig, name):
    fig.savefig(OUT / name, dpi=400, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("figure ->", OUT / name)


def roc():
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.15))
    handles = []
    for m in MODELS:
        d = preds(m)
        fpr, tpr, _ = roc_curve(d.label, d.score)
        pr, rc, _ = precision_recall_curve(d.label, d.score)
        lw = 2.2 if m.startswith("B-ACPN") else 1.1
        h, = ax[0].plot(fpr, tpr, color=COL[m], lw=lw, label=f"{SHORT[m]} ({roc_auc_score(d.label, d.score):.3f})")
        ax[1].plot(rc, pr, color=COL[m], lw=lw)
        handles.append(h)
    ax[0].plot([0, 1], [0, 1], color=V.MUTED, ls=":", lw=0.8)
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall")
    for a in ax:
        a.set_xticks([0, 0.5, 1]); a.set_yticks([0, 0.5, 1]); a.set_xlim(-0.02, 1.02); a.set_ylim(-0.02, 1.04)
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.36), title="Model (pooled AUC)",
               title_fontsize=7.5, columnspacing=1.0, handlelength=1.6)
    fig.subplots_adjust(wspace=0.45)
    save(fig, "roc_paper.png")


def push():
    e = pd.read_csv(RES / "district_explanations.csv")
    order = ["High dev / Low vul", "Low dev / High vul", "High dev / High vul", "Low dev / Low vul"]
    names = ["Expected\nhigh dev,\nlow vul", "Expected\nlow dev,\nhigh vul", "Anomaly A\ndeveloped,\nvulnerable",
             "Anomaly B\nless dev.,\nresilient"]
    fig, ax = plt.subplots(figsize=(W, 2.6))
    rng = np.random.default_rng(0)
    for i, q in enumerate(order):
        v = e[e.quadrant == q].push
        s = V.QUADRANT_STYLE[q]
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=5, alpha=0.5, color=s["color"], marker=s["marker"],
                   linewidths=0)
        ax.plot([i - 0.3, i + 0.3], [v.mean()] * 2, color=V.INK, lw=1.8)
        ax.text(i, 1.3, f"{v.mean():+.2f}", ha="center", va="center", fontsize=8, fontweight="bold")
    ax.axhline(0, color=V.MUTED, lw=0.8)
    ax.set_xticks(range(4), names, fontsize=7.5)
    ax.set_ylabel("Climate push (log-odds)")
    ax.set_xlim(-0.55, 3.55); ax.set_ylim(top=1.55)
    ax.grid(axis="x", visible=False)
    save(fig, "push_paper.png")


def dashboard():
    """Screenshot of the district view with Node Playwright (the browser bundled in the environment)."""
    import json
    import subprocess
    url = (ROOT / "dashboard" / "index.html").as_uri() + "?d=Kishanganj"
    css = (".grid{grid-template-columns:1fr !important} .kpis{gap:6px !important} .kpi{padding:8px 8px !important}"
           " main{padding:10px 12px !important} header{padding:10px 12px !important}"
           " header span{display:none} .grid>.card:nth-child(2){display:none} #q{display:none}"
           " #sel{margin-bottom:8px !important} .grid>.card h2{margin-bottom:6px !important}")
    js = f"""
const {{ chromium }} = require('playwright');
(async () => {{
  const b = await chromium.launch();
  const pg = await b.newPage({{ viewport: {{ width: 540, height: 900 }}, deviceScaleFactor: 4 }});
  await pg.goto({json.dumps(url)});
  await pg.addStyleTag({{ content: {json.dumps(css)} }});
  await pg.waitForTimeout(300);
  const box = await pg.locator('.grid > .card').first().boundingBox();
  await pg.screenshot({{ path: {json.dumps(str(OUT / "dashboard_paper.png"))},
                        clip: {{ x: 0, y: 0, width: 540, height: box.y + box.height + 10 }} }});
  await b.close();
}})();
"""
    subprocess.run(["node", "-e", js], check=True)
    print("figure ->", OUT / "dashboard_paper.png")


def main():
    OUT.mkdir(exist_ok=True)
    style()
    roc()
    push()
    dashboard()


if __name__ == "__main__":
    main()
