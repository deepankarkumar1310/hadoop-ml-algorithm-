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
         "comparison_bars.png": 8.4, "fold_boxplots.png": 4.8, "ablation.png": 4.8, "attention_importance.png": 4.4,
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
    dashboard()


if __name__ == "__main__":
    main()
