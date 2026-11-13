"""Paper copies of the project figures: the same plots, layout and labels as figures/, with larger text.

The figures in figures/ are drawn 6-11.5 in wide for slides and the report; in a 3.5 in paper column their text
shrinks to 2-4 pt. Here the same plotting code (scripts/make_figures.py) is run again and every figure is saved at
a smaller physical width. Fonts keep their point size, so relative to the plot the text is 1.5-2.5x larger, while
the layout, colours and labels stay exactly as in the original figure.

The dashboard snapshot is taken again from dashboard/index.html in a narrower window (same two-panel layout), which
also makes its text larger relative to the picture.

Writes paper/figs/*.png.   Usage: python paper/paper_figures.py
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.make_figures as MF  # noqa: E402
from scripts.common import ROOT, load_table  # noqa: E402

OUT = Path(__file__).resolve().parent / "figs"
# physical width (in) each figure is saved at; the paper prints column figures 3.5 in and page-wide ones 7.2 in
WIDTH = {"roc_curves.png": 6.0, "climate_push_by_quadrant.png": 7.2, "confusion_matrices.png": 8.0,
         "comparison_bars.png": 5.6, "fold_boxplots.png": 4.8, "ablation.png": 4.8, "attention_importance.png": 4.4,
         "climate_push_map.png": 4.4, "calibration.png": 4.2}
_orig_save = MF.V.save


def save(fig, path):
    name = Path(path).name
    if name not in WIDTH:
        MF.plt.close(fig)
        return
    w, h = fig.get_size_inches()
    s = WIDTH[name] / w
    fig.set_size_inches(w * s, h * s)
    fig.savefig(OUT / name, dpi=400, bbox_inches="tight", pad_inches=0.03)
    MF.plt.close(fig)
    print("figure ->", OUT / name)


def confusion_column():
    """Confusion matrices for one paper column: the same six matrices and colours, in 3 rows x 2 columns, with the
    True/Pred labels only on the outer edges so neighbouring panels do not overlap."""
    import numpy as np
    import pandas as pd
    plt, V = MF.plt, MF.V
    cm = pd.read_csv(MF.RES / "confusion_matrices.csv").set_index("model")
    fig, axs = plt.subplots(3, 2, figsize=(3.5, 5.0), gridspec_kw=dict(wspace=0.06, hspace=0.32))
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
    fig.savefig(OUT / "confusion_matrices_column.png", dpi=400, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("figure ->", OUT / "confusion_matrices_column.png")


def dashboard():
    """Same page and layout as figures/dashboard.png, in a narrower window so the text is larger."""
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
    for old in ("roc_paper.png", "push_paper.png"):
        (OUT / old).unlink(missing_ok=True)
    MF.V.save = save
    MF.V.apply()
    df, X, _ = load_table()
    MF.preprocessing(df, X)
    MF.classifier(df)
    MF.explanation(df)
    MF.V.save = _orig_save
    confusion_column()
    dashboard()


if __name__ == "__main__":
    main()
