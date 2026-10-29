"""Build the Review 2 slide deck on the SRM template.

Every number comes from results/*.json|csv. Numbers that do not exist yet
(B-ACPN, Spark MLlib baselines, k-means regimes, ablation, climate push) are
written as PENDING and filled in automatically once results/ contains them.

Usage: python scripts/build_deck.py
"""
import json
import numbers
from pathlib import Path

import sys

import pandas as pd
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.common import run_label  # noqa: E402
RES = ROOT / "results"
TEMPLATE = ROOT / "docs" / "templates" / "Review_2_PPT.pptx"
OUT = ROOT / "docs" / "Review2_Deck.pptx"

NAVY = RGBColor(0x1B, 0x3A, 0x6B)
TEAL = RGBColor(0x0E, 0x6E, 0x6E)
ORANGE = RGBColor(0xC2, 0x5B, 0x12)
INK = RGBColor(0x1F, 0x1F, 0x1F)
GREY = RGBColor(0x59, 0x59, 0x59)
LIGHT = RGBColor(0xEE, 0xF2, 0xF8)
TINT_T = RGBColor(0xE3, 0xF1, 0xF0)
TINT_O = RGBColor(0xFC, 0xEB, 0xDD)
HILITE = RGBColor(0xFF, 0xF4, 0xCC)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
PEND = RGBColor(0xC0, 0x39, 0x2B)
TITLE_FONT, BODY_FONT = "Times New Roman", "Calibri"
PENDING = "PENDING"
REF = run_label()


def load(name):
    p = RES / name
    if not p.exists():
        return None
    return json.loads(p.read_text()) if p.suffix == ".json" else pd.read_csv(p)


# ----------------------------------------------------------------- helpers
def style_run(run, size, bold=False, color=INK, font=BODY_FONT, italic=False):
    run.font.size, run.font.bold, run.font.italic = Pt(size), bold, italic
    run.font.name = font
    run.font.color.rgb = color


def write(tf, text, size, bold=False, color=INK, align=PP_ALIGN.LEFT, font=BODY_FONT):
    """Write text into a text frame; PENDING is always shown in red italics."""
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = str(text)
    pend = str(text).startswith(PENDING)
    style_run(r, size, bold, PEND if pend else color, font, italic=pend)


def textbox(slide, x, y, w, h, text, size=16, bold=False, color=INK, align=PP_ALIGN.LEFT,
            anchor=MSO_ANCHOR.TOP, font=BODY_FONT):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap, tf.vertical_anchor = True, anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    write(tf, text, size, bold, color, align, font)
    return tb


def bullets(slide, x, y, w, h, items, size=16, space=6, color=INK):
    """items: str or (bold_label, rest)."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.05)
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(space)
        r = p.add_run()
        r.text = "▪  "
        style_run(r, size, color=TEAL)
        if isinstance(it, tuple):
            r = p.add_run(); r.text = it[0]; style_run(r, size, True, color)
            r = p.add_run(); r.text = it[1]; style_run(r, size, color=color)
        else:
            r = p.add_run(); r.text = it; style_run(r, size, color=color)
    return tb


def box(slide, x, y, w, h, text, fill=LIGHT, line=NAVY, size=13, bold=True, color=INK,
        shape=MSO_SHAPE.ROUNDED_RECTANGLE, sub=None, sub_size=11):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = fill
    s.line.color.rgb = line; s.line.width = Pt(1.25)
    s.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = 0.12
    tf = s.text_frame
    tf.word_wrap, tf.vertical_anchor = True, MSO_ANCHOR.MIDDLE
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, m, Inches(0.04))
    write(tf, text, size, bold, color, PP_ALIGN.CENTER)
    if sub:
        p = tf.add_paragraph(); p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = sub; style_run(r, sub_size, color=GREY)
    return s


def arrow(slide, x1, y1, x2, y2, color=NAVY):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color; c.line.width = Pt(1.75)
    ln = c.line._get_or_add_ln()
    tail = ln.makeelement("{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd",
                          {"type": "triangle", "w": "med", "len": "med"})
    ln.append(tail)
    return c


def table(slide, x, y, w, rows, widths, size=12, head_size=None, first_bold=True, row_h=0.32,
          highlight=(), head_fill=NAVY, align_num=True):
    head_size = head_size or size
    nr, nc = len(rows), len(rows[0])
    gs = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
    t = gs.table
    tot = sum(widths)
    for j, cw in enumerate(widths):
        t.columns[j].width = Inches(w * cw / tot)
    for i in range(nr):
        t.rows[i].height = Inches(row_h)
        for j in range(nc):
            c = t.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.06)
            c.margin_top = c.margin_bottom = Inches(0.03)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            txt = str(rows[i][j])
            tf = c.text_frame; tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if (j == 0 or not align_num) else PP_ALIGN.CENTER
            r = p.add_run(); r.text = txt
            c.fill.solid()
            if i == 0:
                c.fill.fore_color.rgb = head_fill
                style_run(r, head_size, True, WHITE)
            else:
                c.fill.fore_color.rgb = HILITE if i in highlight else (LIGHT if i % 2 == 0 else WHITE)
                pend = txt.startswith(PENDING)
                style_run(r, size, (j == 0 and first_bold) or i in highlight,
                          PEND if pend else INK, italic=pend)
    return gs


def panel(slide, x, y, w, h, title, body):
    """Framed placeholder for a figure that does not exist yet."""
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = RGBColor(0xF7, 0xF7, 0xF7)
    s.line.color.rgb = RGBColor(0xA6, 0xA6, 0xA6); s.line.width = Pt(1); s.line.dash_style = 4
    s.shadow.inherit = False
    tf = s.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    write(tf, title, 15, True, NAVY, PP_ALIGN.CENTER)
    p = tf.add_paragraph(); p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = body; style_run(r, 12, color=PEND, italic=True)


def fig_or_panel(slide, x, y, w, h, path, title, body):
    if path.exists():
        from PIL import Image
        iw, ih = Image.open(path).size
        s = min(w / iw, h / ih)
        slide.shapes.add_picture(str(path), Inches(x + (w - iw * s) / 2), Inches(y + (h - ih * s) / 2),
                                 Inches(iw * s), Inches(ih * s))
    else:
        panel(slide, x, y, w, h, title, body)


# ----------------------------------------------------------------- deck setup
prs = Presentation(str(TEMPLATE))
logo_pic = [s for s in prs.slides[0].shapes if s.shape_type == 13][0]
LOGO_BLOB = logo_pic.image.blob
LOGO_POS = (logo_pic.left, logo_pic.top, logo_pic.width, logo_pic.height)
TITLE_ONLY = [l for l in prs.slide_layouts if l.name == "Title Only"][0]

# drop the template's sample "Contents" slide; slide 1 becomes our title slide
sld_ids = prs.slides._sldIdLst
for sid in list(sld_ids)[1:]:
    prs.part.drop_rel(sid.rId)
    sld_ids.remove(sid)

import io  # noqa: E402


def new_slide(title, notes, size=28):
    s = prs.slides.add_slide(TITLE_ONLY)
    t = s.shapes.title
    t.left, t.top, t.width, t.height = Inches(0.4), Inches(0.3), Inches(6.95), Inches(0.9)
    tf = t.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.05)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT
    r = p.add_run(); r.text = title
    style_run(r, size, True, NAVY, TITLE_FONT)
    s.shapes.add_picture(io.BytesIO(LOGO_BLOB), *LOGO_POS)
    s.notes_slide.notes_text_frame.text = notes
    return s


def owner(slide, name):
    textbox(slide, 0.4, 7.05, 9.2, 0.3, f"Presenter: {name}", 10, color=GREY, align=PP_ALIGN.RIGHT)


# ----------------------------------------------------------------- data
pm = load("preprocessing_metrics.json")
ar = load("analysis_results.json")
hdfs = load("hdfs_usage_bytes.json")
timing = load("scalability_timings.csv")
hive = load("hive_check.json")
env = load("environment.json")
cmp5 = load("table5_comparison.csv")
full = load("classifier_metrics_full.csv")
conf = load("confusion_matrices.csv")
sig = load("significance.csv")
abl = load("table6_ablation.csv")
push = load("table6_climate_push.json")
kmeans = load("kmeans_regimes.json")
card = load("table4_model_card.json")

MB = lambda b: f"{b / 1e6:,.1f} MB"
tot_bytes = sum(hdfs.values())
mr = timing.groupby("years").runtime_s.mean().round(0).astype(int).to_dict()
mr30 = timing[timing.years == 30].iloc[0]


def g(df, model, col, fmt="{:.3f}"):
    """Look up a metric for a model, or PENDING."""
    if df is None or col not in df or model not in set(df.model):
        return PENDING
    v = df.loc[df.model == model, col].iloc[0]
    if isinstance(v, numbers.Number):
        return "n/a" if pd.isna(v) else fmt.format(v)
    return "n/a" if pd.isna(v) else str(v)


SHORTN = {"Logistic Regression": "LR", "Random Forest": "RF", "Gradient-Boosted Trees": "GBT",
          "Linear SVM": "Linear SVM", "Standard MLP": "MLP", "B-ACPN (proposed)": "B-ACPN"}
MODELS = [("B-ACPN (proposed)", "Novel hybrid DL · PyTorch"), ("Logistic Regression", "Spark MLlib"),
          ("Random Forest", "Spark MLlib"), ("Gradient-Boosted Trees", "Spark MLlib"),
          ("Linear SVM", "Spark MLlib"), ("Standard MLP", "PyTorch")]

# ================================================================= 1 TITLE
s1 = prs.slides[0]
for sh in list(s1.shapes):
    if sh.shape_type == 17:  # template's faculty/team box: replaced below
        sh._element.getparent().remove(sh._element)
t = s1.shapes.title
t.left, t.top, t.width, t.height = Inches(0.5), Inches(1.75), Inches(9.0), Inches(2.2)
tf = t.text_frame; tf.clear(); tf.word_wrap = True
p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
r = p.add_run(); r.text = "Beyond Development: Explaining Climate Resilience Across Indian Districts"
style_run(r, 32, True, NAVY, TITLE_FONT)
p = tf.add_paragraph(); p.alignment = PP_ALIGN.CENTER; p.space_before = Pt(10)
r = p.add_run(); r.text = "Review 2"; style_run(r, 26, True, INK, TITLE_FONT)
tb = s1.shapes.add_textbox(Inches(5.0), Inches(4.75), Inches(4.6), Inches(2.3))
tf = tb.text_frame; tf.word_wrap = True
lines = [("Faculty name - Dr Bhanu Priya", True), ("Team Members info-", True),
         ("Sehjodh Grewal (RA2411031010081)", False), ("Devanshu Kumar (RA2411031010102)", False),
         ("Deepankar (RA2411031010103)", False), ("Tanmay Singh (RA2411031010118)", False)]
for i, (txt, b) in enumerate(lines):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = txt; style_run(r, 16, b, INK, TITLE_FONT)
s1.notes_slide.notes_text_frame.text = (
    "Good morning. Our project asks why some developed Indian districts remain highly climate-vulnerable "
    "while some less-developed districts are resilient. We built a Big Data pipeline over 30 years of IMD "
    "climate data and a novel hybrid deep-learning model, B-ACPN, to explain each district.")

# ================================================================= 2 CONTENTS
s = new_slide("Contents", "The deck runs in three parts, each with its own tables and figures: preprocessing "
              "first, then the Big Data tools, then the proposed model and the comparison with existing models, "
              "followed by the key findings, team plan, references, photo and the Colab link.")
items = ["Dataset", "Preprocessing: steps, results and error metrics", "Effect of preprocessing on the models",
         "Big Data tools: architecture, HDFS, MapReduce, Hive, Spark", "Proposed Model (B-ACPN)",
         "Ablation and climate-push results", "Comparison with the existing classifier models",
         "Key findings", "Plan of Action / Member Tasks", "References",
         "Geotagged photo taken during Review 2 presentation", "Google Colab Implementation link"]
for i, it in enumerate(items):
    col, row = divmod(i, 6)
    x, y = 0.5 + col * 4.6, 1.55 + row * 0.88
    c = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(0.5), Inches(0.5))
    c.fill.solid(); c.fill.fore_color.rgb = NAVY if i < 8 else TEAL; c.line.fill.background()
    c.shadow.inherit = False
    ctf = c.text_frame
    ctf.word_wrap, ctf.vertical_anchor = False, MSO_ANCHOR.MIDDLE
    ctf.margin_left = ctf.margin_right = ctf.margin_top = ctf.margin_bottom = 0
    write(ctf, str(i + 1), 14, True, WHITE, PP_ALIGN.CENTER)
    textbox(s, x + 0.6, y - 0.05, 3.85, 0.65, it, 16, anchor=MSO_ANCHOR.MIDDLE)
owner(s, "Sehjodh Grewal")

# ================================================================= 3 PROBLEM
q = ar["quadrants"]
s = new_slide("Problem Statement & Objectives",
              "Development alone should predict resilience, but it does not always. Splitting districts at the "
              "median of development and of vulnerability gives four groups; 181 of 469 districts break the "
              "expected pattern. Our objectives are the pipeline, the novel model, a fair comparison, and a "
              "per-district explanation.")
box(s, 0.4, 1.35, 9.2, 0.95, "Why do some developed Indian districts remain highly climate-vulnerable, "
    "while some less-developed districts are resilient?", fill=TINT_T, line=TEAL, size=17, bold=True)
textbox(s, 0.4, 2.5, 4.6, 0.4, "Objectives", 18, True, NAVY)
bullets(s, 0.4, 2.95, 4.65, 3.9, [
    ("Big Data pipeline ", "over 30 years (1991–2020) of IMD daily climate data: HDFS, MapReduce, Hive, Spark"),
    ("Novel deep-learning model ", "B-ACPN (Boosted Attentive Climate Push Network)"),
    ("Compare ", "with 5 existing ML models on identical folds"),
    ("Explain ", "each district as climate-driven or development-driven")], size=15)
textbox(s, 5.3, 2.5, 4.3, 0.4, "469 districts, median splits", 18, True, NAVY)
gx, gy, cw, ch = 6.55, 3.3, 1.5, 1.25
textbox(s, gx, gy - 0.42, cw, 0.4, "Low vulnerability", 12, True, GREY, PP_ALIGN.CENTER)
textbox(s, gx + cw + 0.05, gy - 0.42, cw, 0.4, "High vulnerability", 12, True, GREY, PP_ALIGN.CENTER)
textbox(s, 5.3, gy, 1.2, ch, "High development", 12, True, GREY, PP_ALIGN.RIGHT, MSO_ANCHOR.MIDDLE)
textbox(s, 5.3, gy + ch + 0.05, 1.2, ch, "Low development", 12, True, GREY, PP_ALIGN.RIGHT, MSO_ANCHOR.MIDDLE)
cells = [(0, 0, q["High dev / Low vul"], "Expected", LIGHT, NAVY), (0, 1, q["High dev / High vul"], "Anomaly A", TINT_O, ORANGE),
         (1, 0, q["Low dev / Low vul"], "Anomaly B", TINT_T, TEAL), (1, 1, q["Low dev / High vul"], "Expected", LIGHT, NAVY)]
for r_, c_, n, lab, fill, ln in cells:
    box(s, gx + c_ * (cw + 0.05), gy + r_ * (ch + 0.05), cw, ch, str(n), fill, ln, 26, True, ln, sub=lab, sub_size=13)
textbox(s, 5.3, 5.95, 4.3, 0.8, f"{ar['anomalies']} districts ({ar['anomaly_share'] * 100:.1f}%) break the "
        "expected pattern", 14, True, ORANGE, PP_ALIGN.CENTER)
owner(s, "Sehjodh Grewal")

# ================================================================= 4 DATASET
s = new_slide("Dataset", "We combine five public sources. NDAP gives the outcome and two development "
              "variables, Census 2011 gives three more, and IMD gives 30 years of daily gridded rainfall "
              "and temperature, which is what makes this a Big Data problem. The vulnerability index is "
              "only ever the target, never an input.")
bullets(s, 0.4, 1.3, 9.2, 1.75, [
    ("5 development variables: ", "literacy, female literacy, agri-worker share (Census 2011); "
     "MGNREGA days, yield variability (NDAP 2021)"),
    ("8 hazard variables: ", "annual rain, rain variability, max 5-day rain, heavy-rain days, longest dry "
     "spell, mean Tmax, hot days, tropical nights"),
    ("Scale: ", "~212 million gridded climate values (30 years of daily IMD grids)")], size=15, space=3)
table(s, 0.4, 3.2, 9.2, [
    ["Dataset", "Source", "Year", "Format", "Size", "Role"],
    ["Climate Vulnerability Index + indicators", "NITI Aayog NDAP 7160", "2021", "CSV", "699 × 79", "Outcome + 2 dev vars"],
    ["Primary Census Abstract", "Census of India", "2011", "CSV", "640 districts", "3 dev vars"],
    ["Daily gridded rainfall 0.25°", "IMD Pune", "1991–2020", "binary .grd", "135 × 129 grid", "Hazard"],
    ["Daily gridded Tmax / Tmin 1°", "IMD Pune", "1991–2020", "binary .grd", "31 × 31 grid", "Hazard"],
    ["District boundaries", "DataMeet", "2011", "Shapefile", "641 polygons", "Grid → district"]],
    [2.6, 1.7, 0.9, 1.0, 1.15, 1.45], size=12, row_h=0.6)
owner(s, "Sehjodh Grewal")

# ================================================================= 5 PROPOSED MODEL
s = new_slide("Proposed Model: B-ACPN", "B-ACPN works in two stages. A random forest first gives a tree risk "
              "score. A deep network then starts from that score and learns what the trees miss: a development "
              "term and a climate push gated by development level. Five networks are averaged, attention shows "
              "which climate factor drives the push, and MC Dropout with conformal prediction says how certain we are.")
bullets(s, 0.4, 1.3, 9.2, 2.2, [
    ("Tree stage: ", "random forest (500 trees) on all 13 features → risk score, out-of-fold for training rows"),
    ("Deep stage: ", "tree score + linear development term + climate push (attention → MLP → residual block)"),
    ("Development gate: ", "climate push × 2σ(gate(development)): how strongly climate acts at that level"),
    ("Output: ", "ensemble of 5 networks; push > 0 means climate adds risk beyond development"),
    ("Uncertainty: ", "MC Dropout + conformal prediction; index never used as input (no circularity)")],
    size=15, space=3)
box(s, 0.4, 4.0, 1.75, 0.8, "All 13 features", sub="dev + climate", size=13)
box(s, 0.4, 5.55, 1.75, 0.8, "Climate (IMD)", sub="8 features", fill=TINT_T, line=TEAL, size=13)
box(s, 2.55, 4.0, 2.2, 0.8, "Random forest", sub="tree stage, 500 trees", size=13)
box(s, 2.55, 5.55, 2.2, 0.8, "Attention → MLP → Residual × dev gate", fill=TINT_T, line=TEAL, size=12)
box(s, 5.15, 4.0, 1.3, 0.8, "Tree + dev", sub="b", size=13)
box(s, 5.15, 5.55, 1.3, 0.8, "Climate push", sub="c (+/−)", fill=TINT_T, line=TEAL, size=13)
box(s, 6.85, 4.75, 0.8, 0.8, "+", fill=WHITE, shape=MSO_SHAPE.OVAL, size=24)
box(s, 8.0, 4.0, 1.6, 0.8, "σ(b + c) × 5", sub="ensemble mean", fill=HILITE, line=ORANGE, size=14)
box(s, 8.0, 5.55, 1.6, 0.8, "Uncertainty", sub="MC Dropout + conformal", fill=TINT_O, line=ORANGE, size=12, sub_size=10)
for y in (4.4, 5.95):
    arrow(s, 2.15, y, 2.55, y); arrow(s, 4.75, y, 5.15, y)
arrow(s, 6.45, 4.4, 6.9, 4.95); arrow(s, 6.45, 5.95, 6.9, 5.35)
arrow(s, 7.65, 5.15, 8.0, 4.5); arrow(s, 8.8, 4.8, 8.8, 5.55)
box(s, 0.4, 6.55, 9.2, 0.45, "Novelty: a tree stage boosted by a development-gated attentive climate network (with uncertainty) "
    "that separates each district's climate push from its tree/development baseline", WHITE, TEAL, 12, False, TEAL)
owner(s, "Devanshu Kumar")

# ================================================================= 6 ARCHITECTURE
s = new_slide("Overall System Architecture Diagram",
              "Data flows top-left to bottom-right. Raw sources land in the HDFS raw zone; Python converts the "
              "IMD grids to district-day rows in staging; MapReduce on YARN aggregates 7 million daily rows to "
              "yearly indices; Hive builds the 30-year baseline and joins development data; Spark finds climate "
              "regimes and trains the MLlib baselines; B-ACPN (random forest + PyTorch) is trained and compared.")
row1 = [("Sources", "NDAP · Census · IMD · DataMeet", LIGHT, NAVY), ("HDFS raw zone", "/project/raw", TINT_T, TEAL),
        ("Grid → district", "Python (NumPy, shapely)", LIGHT, NAVY), ("HDFS staging", "7.0 M district-day rows", TINT_T, TEAL)]
row2 = [("MapReduce on YARN", "→ 19,200 district-years", TINT_O, ORANGE), ("Hive", "30-yr baseline + join", TINT_O, ORANGE),
        ("Spark", "k-means + MLlib", TINT_O, ORANGE)]
row3 = [("B-ACPN", "RF + PyTorch", HILITE, ORANGE), ("Comparison tables", "Tables 1–6", LIGHT, NAVY),
        ("Report / slides", "results/ · figures/", LIGHT, NAVY)]
bw, bh = 1.95, 1.0
for i, (a, b_, f, l) in enumerate(row1):
    box(s, 0.4 + i * 2.4, 1.45, bw, bh, a, f, l, 14, sub=b_, sub_size=11)
    if i < 3:
        arrow(s, 0.4 + i * 2.4 + bw, 1.95, 0.4 + (i + 1) * 2.4, 1.95)
arrow(s, 8.6, 2.45, 8.6, 2.85)
for i, (a, b_, f, l) in enumerate(row2):
    xx = 7.6 - i * 2.95
    box(s, xx, 2.85, 2.1, bh, a, f, l, 14, sub=b_, sub_size=11)
    if i < 2:
        arrow(s, xx, 3.35, xx - 0.85, 3.35)
arrow(s, 2.3, 3.85, 2.3, 4.25)
for i, (a, b_, f, l) in enumerate(row3):
    box(s, 0.4 + i * 3.1, 4.25, 2.5, bh, a, f, l, 14, sub=b_, sub_size=11)
    if i < 2:
        arrow(s, 0.4 + i * 3.1 + 2.5, 4.75, 0.4 + (i + 1) * 3.1, 4.75)
textbox(s, 0.4, 5.55, 9.2, 0.4, "Legend:", 12, True, GREY)
for i, (lab, f, l) in enumerate([("Storage (HDFS)", TINT_T, TEAL), ("Big Data processing", TINT_O, ORANGE),
                                 ("Python / outputs", LIGHT, NAVY), ("Novel model", HILITE, ORANGE)]):
    box(s, 1.2 + i * 2.1, 5.55, 1.9, 0.4, lab, f, l, 11, False)
box(s, 0.4, 6.2, 9.2, 0.6, f"Single-node pseudo-distributed {env['hadoop']}, HDFS replication 1, "
    f"{env['cpu_count']} vCPU, {env['mem_mb'] / 1024:.1f} GB RAM ({REF}); no multi-machine speed-up is claimed",
    WHITE, GREY, 12, False, GREY)
owner(s, "Deepankar")

# ================================================================= 7 HDFS
s = new_slide("Big Data Platform & HDFS Storage",
              f"We installed Java 8, Hadoop with HDFS and YARN, Hive and Spark inside Colab. HDFS has three zones "
              f"so the raw data is never modified. In total {tot_bytes / 1e9:.2f} GB is stored, mostly the 30 years "
              f"of rainfall grids. These sizes were measured with hdfs dfs -du in the {REF}.")
table(s, 0.4, 1.3, 9.2, [
    ["HDFS zone", "Contents", "Size"],
    ["/project/raw/imd/rain", "30 yearly rainfall files (0.25° grid)", MB(hdfs["raw/imd/rain"])],
    ["/project/raw/imd/tmax, tmin", "60 temperature files (1° grid)", MB(hdfs["raw/imd/tmax"] + hdfs["raw/imd/tmin"])],
    ["/project/raw/sources", "NDAP, Census, boundaries", MB(hdfs["raw/census+ndap+boundaries"])],
    ["/project/staging/district_day", "7,013,120 district-day rows", MB(hdfs["staging/district_day"])],
    ["/project/processed", "MapReduce, Hive, Spark outputs", MB(hdfs["processed"])],
    ["Total", f"measured with hdfs dfs -du ({REF})", f"{tot_bytes / 1e9:.2f} GB"]],
    [2.7, 4.3, 1.4], size=13, row_h=0.4, highlight=(6,))
table(s, 0.4, 4.35, 9.2, [
    ["Tool", "What it does / purpose", "Why chosen"],
    ["HDFS", "Central store for raw, staging and processed zones", "128 MB blocks for parallel jobs; read directly by MapReduce, Hive, Spark"],
    ["YARN", "Allocates CPU / memory and runs MapReduce tasks", "Hadoop's standard scheduler for MapReduce"]],
    [0.9, 3.6, 4.7], size=12, row_h=0.45)
bullets(s, 0.4, 5.85, 9.2, 1.1, [
    f"{env['hadoop']} + YARN, Hive 3.1.3, Spark (pyspark), Java 8 (Hive 3.1.3 fails on Java 11)",
    "Daemons started directly (Colab has no SSH); replication 1, block size 128 MB"], size=15, space=2)
owner(s, "Deepankar")

# ================================================================= 8 PROCESSING
cc = load("preprocessing_colab_checks.json") or {}
_m = cc.get("mapreduce_vs_pandas")
mr_check = (f"RMSE ≤ {_m['rmse_max']:.2g}, MAE ≤ {_m['mae_max']:.2g}" if _m else PENDING)
km_txt = PENDING if kmeans is None else f"k = {kmeans['k']}, silhouette {kmeans['silhouette']:.3f}"
s = new_slide("Big Data Processing: MapReduce, Hive, Spark",
              "MapReduce does the heaviest step, 7 million daily rows to 19,200 district-years, because the work "
              "splits by district and year. Hive does the averaging and the join in SQL, and its output matched "
              "pandas exactly. Spark k-means finds the climate regimes because it works in memory. The last row "
              "checks the MapReduce output against an independent pandas recomputation.", size=24)
table(s, 0.4, 1.3, 9.2, [
    ["Job", "Input", "Output", "Result", "Why this tool"],
    ["MapReduce (Hadoop Streaming)", f"{int(mr30.map_input_records):,} district-day rows",
     f"{int(mr30.reduce_output_records):,} district-year rows, 7 indices", f"{mr[30]} s (30 years)",
     "Work splits by (district, year)"],
    ["MapReduce scalability", "1 / 10 / 30 years", "640 / 6,400 / 19,200 rows",
     f"{mr[1]} s / {mr[10]} s / {mr[30]} s", "Measures throughput, 2 repeats"],
    ["Hive baseline + join", "19,200 rows + dev table", f"{hive['hive_rows']}-district table, 8 hazard vars",
     f"identical to pandas (max diff {hive['max_abs_diff']})", "Averages + joins are SQL"],
    ["Spark k-means", "8 hazard vars", "climate regimes (k by silhouette)", km_txt, "In-memory iterative algorithm"],
    ["MapReduce correctness", "same district-days", "7 indices recomputed in pandas", mr_check, "Independent check of the job"]],
    [1.75, 1.75, 2.0, 1.75, 1.95], size=12, row_h=0.72)
textbox(s, 0.4, 5.85, 9.2, 0.9, f"MapReduce and Hive values: {REF} (2 vCPU, 12.7 GB RAM). Single node: "
        "throughput rises with input size, but no multi-machine speed-up is claimed.", 13, color=GREY)
owner(s, "Deepankar")

# ================================================================= 9 PREPROCESSING STEPS
s = new_slide("Preprocessing: Big Data Analysis Steps",
              "Seven preprocessing steps take the raw sources to a model-ready table. Matching is done inside "
              "states and two wrong fuzzy matches are blocked. We never impute: only 5 development variables "
              "are complete enough. Standardisation uses training-fold statistics only, so there is no leakage.")
mt = pm["matching"]["match_types"]
table(s, 0.4, 1.3, 9.2, [
    ["Step", "Operation", "Tool", "Result"],
    ["1 Census rates", "literacy, female literacy, agri share", "pandas", f"{pm['census']['districts']} districts; totals validated"],
    ["2 District matching", "NDAP ↔ Census by state + name", "Python", f"{pm['matching']['matched']}/{pm['matching']['ndap_districts']} matched; 2 wrong fuzzy matches blocked"],
    ["3 Variable selection", "NDAP sparsity analysis", "pandas", f"{pm['ndap']['cols_over_half_empty']}/{pm['ndap']['cols']} columns >50% empty → 5 dev vars, no imputation"],
    ["4 Grid weights", "cell overlap × cos(lat), normalised", "shapely, imdlib", f"{pm['grid_rain']['pairs']:,} rain + {pm['grid_temp']['pairs']:,} temp pairs, {pm['grid_rain']['districts']}/640"],
    ["5 Grid → district-day", "weighted mean over non-NaN cells", "NumPy sparse", f"{pm['district_day']['rows']:,} rows"],
    ["6 Orientation check", "Chennai rain must peak Oct–Dec", "Python", "passed"],
    ["7 Standardisation", "z-scores on training folds only", "scikit-learn", "no leakage"]],
    [1.9, 2.75, 1.35, 3.2], size=12, row_h=0.66)
owner(s, "Sehjodh Grewal")

# ================================================================= 10 PREPROCESSING RESULTS
ft = pm["final_table"]
s = new_slide("Results of Preprocessing",
              "Districts are kept only when every variable is real, which takes us from 640 Census districts to "
              "469 complete ones in 21 states. Census totals match the official figures. On the right, each "
              "feature's single-variable ROC-AUC shows how much signal it carries before any model is trained, and "
              "the rainfall signal-to-noise ratio shows the 30-year average is about four times larger than its "
              "year-to-year variation.")
textbox(s, 0.4, 1.25, 4.9, 0.4, "Table 1: retention and validation", 15, True, NAVY)
table(s, 0.4, 1.7, 4.9, [
    ["Stage / check", "Value"],
    ["Census 2011 districts", "640"], ["Matched to NDAP", f"{pm['matching']['matched']}"],
    ["With vulnerability index", f"{ar['preprocessing']['with_index']}"],
    ["Complete (all 13 features)", f"{ft['rows']} ({ft['states']} states)"],
    ["Census population", f"{pm['census']['population']:,}"],
    ["National literacy", f"{pm['census']['literacy_pct']:.2f}%"],
    ["Class balance High / Low", f"{ft['class_high']} / {ft['class_low']}"],
    ["NDAP columns >50% empty", f"{pm['ndap']['cols_over_half_empty']} of {pm['ndap']['cols']}"],
    ["Sample skew", "UP 70, MP 49, Bihar 38, Rajasthan 33"],
    ["Rainfall signal-to-noise (median)", f"{pm['rain_snr_median']:.2f} (mean ÷ year-to-year SD)"]],
    [2.6, 2.3], size=12, row_h=0.43, highlight=(4,))
textbox(s, 5.55, 1.25, 4.05, 0.4, "Feature signal (single-feature ROC-AUC)", 15, True, NAVY)
sig_rows = [["Feature", "AUC"]] + [[k.replace("_", " "), f"{v:.3f}"] for k, v in pm["feature_signal_auc"].items()]
table(s, 5.55, 1.7, 4.05, sig_rows, [2.9, 1.0], size=11, row_h=0.36, first_bold=False)
owner(s, "Sehjodh Grewal")

# ================================================================= 11 PREPROCESSING ERROR METRICS
c_, gr, gt, dd = pm["census"], pm["grid_rain"], pm["grid_temp"], pm["district_day"]
_a = cc.get("area_vs_nearest_cell")
nc_row = (["Area-weighted vs nearest-cell", f"naive method, {_a['year']}", f"{_a['districts']} districts",
           f"RMSE {_a['rmse_mm']:.0f} mm ({_a['mean_abs_pct']:.1f}%) avoided"] if _a else
          ["Area-weighted vs nearest-cell", "naive method", PENDING, "RMSE: PENDING"])
s = new_slide("Preprocessing Error Metrics: Before vs After",
              "Each preprocessing step is checked against a reference. Census totals reproduce the official "
              "figures, grid weights sum to one within rounding, missing climate cells stay missing instead of "
              "becoming zero, and the final table has no missing, duplicate or out-of-range values. The last check "
              "needs the raw IMD grids and is filled by the Colab run.", size=24)
table(s, 0.4, 1.25, 9.2, [
    ["Check", "Before / reference", "After / measured", "Error"],
    ["Census population", "official 1,210,854,977", f"{c_['population']:,}", f"{c_['population_abs_error']} ({c_['population_pct_error']:.3f}%)"],
    ["National literacy", "official 72.98%", f"{c_['literacy_pct']:.4f}%", f"{c_['literacy_abs_error_pp']} pp"],
    ["NDAP missing cells", f"{pm['ndap']['overall_missing_pct']:.1f}% of all cells", f"{ft['feature_missing_cells']} missing in 13 features", "no imputation"],
    ["District matching", "699 names; 97 post-2011; 2 wrong fuzzy", f"{pm['matching']['matched']} matched, {pm['matching']['duplicate_censuscodes_after']} duplicate codes", "2 errors blocked"],
    ["Grid weights Σw = 1", f"{gr['pairs']:,} rain / {gt['pairs']:,} temp pairs", f"max |Σw−1| {gr['max_abs_sum_error']:.0e} / {gt['max_abs_sum_error']:.0e}", f"RMSE {gr['rmse_sum_error']:.1e}"],
    ["District-day missing cells", f"rain {dd['rain_missing_pct']:.2f}%, Tmax {dd['tmax_missing_pct']:.2f}%", "kept as NaN, weights renormalised", "0 zero-filled"],
    ["Final table quality", "—", f"{ft['duplicate_censuscodes']} duplicates, {ft['out_of_range_values']} out of range", f"{ft['outliers_abs_z_gt3']} outlier cells flagged, kept"],
    nc_row],
    [2.15, 2.55, 2.5, 2.0], size=12, row_h=0.52)
textbox(s, 0.4, 6.5, 9.2, 0.45, f"Computed {pm['computed'].split(' ')[0]} from project data; district-day values from the {REF}. "
        "The nearest-cell check needs the raw IMD grids (Colab).", 11, color=GREY)
owner(s, "Sehjodh Grewal")

# ================================================================= 12 PREPROCESSING PERFORMANCE METRICS
def _prep_vals(d, worst=False):
    """8 error metrics of one preprocessing check; worst=True takes the worst of the 7 MapReduce indices."""
    if not d:
        return {}
    if worst:
        per = list(d["per_index"].values())
        out = {k: max(v[k] for v in per) for k in ("rmse", "mae") if all(k in v for v in per)}
        for k in ("mse", "mape_pct", "nrmse", "medae"):
            if all(k in v for v in per):
                out[k] = max(v[k] for v in per if pd.notna(v[k])) if any(pd.notna(v[k]) for v in per) else float("nan")
        if all("r2" in v for v in per):
            out["r2"] = min(v["r2"] for v in per if pd.notna(v["r2"]))
        if all("bias" in v for v in per):
            out["bias"] = max((v["bias"] for v in per), key=abs)
        if "mse" not in out and "rmse" in out:
            out["mse"] = out["rmse"] ** 2
        return out
    m = dict(d.get("metrics", {}))
    m.setdefault("rmse", d.get("rmse_mm")); m.setdefault("mae", d.get("mae_mm"))
    m.setdefault("mape_pct", d.get("mean_abs_pct"))
    if m.get("rmse") is not None:
        m.setdefault("mse", m["rmse"] ** 2)
    return {k: v for k, v in m.items() if v is not None}


def _pf(v, kind):
    if v is None:
        return PENDING
    if pd.isna(v):
        return "n/a"
    if kind == "pct":
        return f"{v:.2f}%"
    if kind == "r2":
        return f"{v:.6f}" if 0.9999 < v < 1 else f"{v:.4f}"
    if abs(v) != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
        return f"{v:.1e}"
    return f"{v:,.3f}" if abs(v) < 10 else f"{v:,.1f}"


pa = _prep_vals(cc.get("area_vs_nearest_cell"))
pmr = _prep_vals(cc.get("mapreduce_vs_pandas"), worst=True)
PREP_METRICS = [("RMSE", "Root Mean Square Error", "Square root of the average squared error", "rmse", "num"),
                ("MSE", "Mean Squared Error", "Average squared difference, actual vs computed", "mse", "num"),
                ("MAE", "Mean Absolute Error", "Average absolute difference", "mae", "num"),
                ("MAPE", "Mean Absolute Percentage Error", "Error as a percentage of the actual value", "mape_pct", "pct"),
                ("R²", "Coefficient of Determination", "How much of the variation is reproduced (1 = all)", "r2", "r2"),
                ("NRMSE", "Normalised RMSE", "RMSE divided by the range of the actual values", "nrmse", "num"),
                ("MedAE", "Median Absolute Error", "Median error; not driven by a few extreme values", "medae", "num"),
                ("Bias / ME", "Mean Error", "Systematic over- (+) or under- (−) estimation", "bias", "num")]
rows = [["Metric", "Full form", "Purpose", "Nearest cell vs our area-weighted rain", "MapReduce vs independent pandas"]]
for short, full_, purpose, k, kind in PREP_METRICS:
    rows.append([short, full_, purpose, _pf(pa.get(k), kind), _pf(pmr.get(k), kind)])
s = new_slide("Preprocessing Performance Metrics",
              "Eight error metrics for the two preprocessing steps that have an exact reference. Left: annual "
              "rainfall per district from the naive nearest-cell method against our area-weighted method, i.e. the "
              "error our method avoids. Right: the seven MapReduce climate indices against an independent pandas "
              "recomputation (worst of the seven), i.e. whether MapReduce is correct.", size=26)
table(s, 0.4, 1.25, 9.2, rows, [1.0, 2.0, 2.9, 1.65, 1.65], size=11, row_h=0.52)
_n_a, _n_m = (cc.get("area_vs_nearest_cell") or {}), (cc.get("mapreduce_vs_pandas") or {})
textbox(s, 0.4, 6.05, 9.2, 0.9,
        f"Left: {_n_a.get('districts', PENDING)} districts, year {_n_a.get('year', PENDING)}, in mm; large values are "
        f"the error the naive method would make. Right: {_n_m.get('rows_compared', PENDING):,} district-years × 7 indices; "
        "values near 0 (and R² near 1) mean MapReduce matches. MAPE skips zero actual values; NRMSE uses the range. "
        f"Computed in Colab section 10b (2026-10-28) from the IMD and MapReduce data of the {REF}." +
        (" PENDING = not yet computed (needs the raw IMD grids in Colab)." if PENDING in sum(rows, []) else ""),
        10, color=GREY)
owner(s, "Sehjodh Grewal")

# ================================================================= 13 MODEL CARD
pc = PENDING if card is None else f"{card['parameters']:,}"
tt = PENDING if card is None else card["train_time"]
cov = PENDING if card is None else f"{card['conformal_coverage_mean']:.1%} coverage (target 90%), avg set size {card['conformal_avg_set_size']:.2f}"
s = new_slide("Classification: B-ACPN Model Card",
              "Table 4 lists the full B-ACPN specification. The tree stage gives a strong first score; the deep "
              "stage is small, sized for 469 districts, and learns the development and climate corrections. Every "
              "model, including B-ACPN, uses the same 25 folds plus leave-one-state-out.")
table(s, 0.4, 1.25, 9.2, [
    ["Component", "Specification"],
    ["Tree stage", "Random forest, 500 trees, min leaf 2, all 13 features; out-of-fold (5 inner folds) score for training rows"],
    ["Deep stage: baseline", "tree log-odds + Linear(5 → 1) on the 5 development features"],
    ["Deep stage: climate push", "attention (8) → Linear(8→16), ReLU, Dropout 0.3 → residual block → Linear(16→1)"],
    ["Development gate", "push × 2σ(Linear(5→4), tanh, Linear(4→1)); ensemble of 5 seeds, probabilities averaged"],
    ["Uncertainty", f"MC Dropout (50 passes) + split conformal: {cov}"],
    ["Loss / optimiser", "Class-weighted BCE; AdamW lr 3e-3, weight decay 1e-3, batch 64, early stopping (patience 30)"],
    ["Parameters / train time", f"{pc} trainable per network (+ 500 trees) / {tt} (CPU)"],
    ["Validation", "Repeated stratified 5-fold × 5 (25 folds) + leave-one-state-out"],
    ["Target", "High vs Low vulnerability (median of 2021 index); index never an input"]],
    [2.3, 6.9], size=12, row_h=0.52)
owner(s, "Devanshu Kumar")

# ================================================================= 14 COMPARISON
rows = [["Model", "Type / Tool", "ROC-AUC", "Bal. acc.", "Accuracy", "F1", "LOSO AUC"]]
for m, typ in MODELS:
    rows.append([m, typ] + [g(cmp5, m, c) for c in ("roc_auc", "bal_acc", "accuracy", "f1", "loso_auc")])
if cmp5 is not None:
    srt = cmp5.sort_values("roc_auc", ascending=False).reset_index(drop=True)
    best = srt.iloc[0]
    acp = cmp5[cmp5.model == "B-ACPN (proposed)"].iloc[0]
    rank = int(srt.index[srt.model == "B-ACPN (proposed)"][0]) + 1
    gap = best.roc_auc - acp.roc_auc
    second = srt[srt.model != acp.model].iloc[0]
    lead = acp.roc_auc - second.roc_auc
    if rank == 1:
        verdict = (f"B-ACPN has the highest ROC-AUC ({acp.roc_auc:.3f}); next is {SHORTN[second.model]} "
                   f"({second.roc_auc:.3f}), lead {lead:+.3f} " +
                   ("> 1 SD, so B-ACPN is best" if lead > acp.roc_auc_sd else
                    f"< 1 SD ({acp.roc_auc_sd:.3f}), so the top models are comparable"))
    else:
        verdict = (f"{best.model} is highest (ROC-AUC {best.roc_auc:.3f}); B-ACPN ranks {rank} of 6 "
                   f"({acp.roc_auc:.3f}, gap {gap:.3f} {'>' if gap > acp.roc_auc_sd else '<'} 1 SD = {acp.roc_auc_sd:.3f})")
    beats = cmp5[(cmp5.roc_auc < acp.roc_auc) & (cmp5.model != acp.model)].model.map(lambda x: SHORTN.get(x, x)).tolist()
    cmp_items = [("Result: ", verdict),
                 ("B-ACPN is higher than: ", ", ".join(beats) if beats else "none of the baselines"),
                 ("Significant (p < 0.05): ", (", ".join(SHORTN[m] for m in sig[sig.p_value < 0.05].model) or "none")
                  + "; not significant: " + (", ".join(SHORTN[m] for m in sig[sig.p_value >= 0.05].model) or "none")),
                 ("B-ACPN's added value: ", "a per-district climate push, attention and uncertainty, which the baselines do not give")]
else:
    cmp_items = [("Result: ", PENDING)]
s = new_slide("Comparison with Existing Classifier Models",
              "Table 5 compares B-ACPN with five existing models on exactly the same 25 folds and the same "
              "leave-one-state-out split. Values are means over folds. We call a model best only if its lead is "
              "larger than one standard deviation, and we report a baseline that beats B-ACPN as it is.", size=24)
table(s, 0.4, 1.3, 9.2, rows, [2.25, 1.7, 1.05, 1.05, 1.05, 1.0, 1.1], size=12, row_h=0.55, highlight=(1,))
bullets(s, 0.4, 5.3, 9.2, 1.4, cmp_items, size=13, space=2)
textbox(s, 0.4, 6.7, 9.2, 0.35, f"B-ACPN and MLP: local CPU run 2026-10-22. Spark MLlib models: same data and folds, Spark 3.5.3 on 2026-10-21.", 10, color=GREY)
owner(s, "Tanmay Singh")

# ================================================================= EFFECT OF PREPROCESSING ON THE MODELS (Devanshu)
ba = load("preprocessing_before_after.csv")
PREP_COLS = [("acc", "Accuracy"), ("f1", "F1 score"), ("rmse", "RMSE"), ("auc", "ROC-AUC")]
rows = [["Model"] + [f"{n} {w}" for _, n in PREP_COLS for w in ("before", "after")]]
for m, _ in MODELS:
    rows.append([SHORTN[m] if ba is not None else m] +
                [g(ba, m, f"{c}_{w}") for c, _ in PREP_COLS for w in ("before", "after")])
if ba is not None:
    acp = ba[ba.model == "B-ACPN (proposed)"].iloc[0]
    neural = ba[ba.model.isin(["B-ACPN (proposed)", "Standard MLP"])]
    worse = [f"{SHORTN[r.model]} accuracy {r.acc_before:.3f} → {r.acc_after:.3f}"
             for r in neural.itertuples() if r.acc_after < r.acc_before - 0.0005]
    ba_note = (f"B-ACPN accuracy {acp.acc_before:.3f} → {acp.acc_after:.3f}, F1 {acp.f1_before:.3f} → {acp.f1_after:.3f}, "
               f"RMSE {acp.rmse_before:.3f} → {acp.rmse_after:.3f} (lower is better), ROC-AUC {acp.auc_before:.3f} → "
               f"{acp.auc_after:.3f}." + (f" Not every metric improves: {'; '.join(worse)}." if worse else "") +
               " RF and GBT ignore feature scale, and Spark's LR and Linear SVM standardise internally, so they do not change.")
else:
    ba_note = PENDING
s = new_slide("Effect of Preprocessing on the Models",
              "To measure what preprocessing buys, every model is trained twice on the same districts and folds: once on "
              "the raw features exactly as they come, and once on the preprocessed features. We compare accuracy, F1, "
              "RMSE and ROC-AUC before and after. The neural models gain the most. Random Forest and GBT do not depend on scale, and Spark's "
              "Logistic Regression and Linear SVM standardise features internally, so those four do not change.", size=24)
table(s, 0.4, 1.3, 9.2, rows, [1.2] + [1.0] * 8, size=11, row_h=0.55, highlight=(1,))
bullets(s, 0.4, 5.25, 9.2, 1.6, [
    ("Before: ", "raw features (rain in mm, days, %, °C on very different scales); After: preprocessed, same 25 folds"),
    ("RMSE: ", "error of the predicted probability against the true class (Linear SVM gives no probabilities: n/a)"),
    ("Result: ", ba_note)], size=12, space=2)
owner(s, "Tanmay Singh")

# ================================================================= 15 DETAILED METRICS
cols = [("precision", "Prec."), ("recall", "Recall"), ("specificity", "Spec."), ("mcc", "MCC"),
        ("kappa", "Kappa"), ("pr_auc", "PR-AUC"), ("log_loss", "Log loss"), ("brier", "Brier"), ("ece", "ECE")]
rows = [["Model"] + [c[1] for c in cols]]
for m, _ in MODELS:
    rows.append([m] + [g(full, m, c[0]) for c in cols])
s = new_slide("Detailed Classifier Metrics",
              "Beyond accuracy we report precision, recall and specificity, two chance-corrected scores (MCC and "
              "kappa), PR-AUC, and three probability-quality scores: log loss, Brier score and expected "
              "calibration error, where lower is better. Linear SVM gives no probabilities, so those cells are n/a.")
table(s, 0.4, 1.3, 9.2, rows, [2.2] + [0.78] * 9, size=11, row_h=0.6, highlight=(1,))
bullets(s, 0.4, 5.75, 9.2, 1.2, [
    ("Higher is better: ", "precision, recall, specificity, MCC, Cohen's kappa, PR-AUC"),
    ("Lower is better: ", "log loss, Brier, ECE (Linear SVM: n/a, no probabilities)")], size=15, space=3)
owner(s, "Tanmay Singh")

# ================================================================= 16 CONFUSION / CI / SIGNIFICANCE
rows = [["Model", "TP", "TN", "FP", "FN", "ROC-AUC 95% CI", "p vs B-ACPN"]]
for m, _ in MODELS:
    rows.append([m] + [g(conf, m, c, "{:.0f}") for c in ("tp", "tn", "fp", "fn")] +
                [g(sig, m, "auc_ci"), "—" if m.startswith("B-ACPN") else g(sig, m, "p_value_fmt")])
s = new_slide("Confusion Matrix, CIs and Significance",
              "Confusion counts are summed over all 25 test folds. The 95% confidence interval shows how certain each "
              "ROC-AUC is, and the corrected paired test tells us whether a difference from B-ACPN is real or "
              "could be noise from the folds; p below 0.05 means a real difference.", size=26)
table(s, 0.4, 1.3, 9.2, rows, [2.3, 0.8, 0.8, 0.8, 0.8, 1.9, 1.3], size=12, row_h=0.6, highlight=(1,))
bullets(s, 0.4, 5.65, 9.2, 1.3, [
    ("TP/TN/FP/FN: ", "summed over the 25 test folds (High = positive class)"),
    ("Significance: ", "corrected resampled t-test (Nadeau–Bengio) on paired fold ROC-AUC; p < 0.05 = real difference")],
    size=15, space=3)
owner(s, "Tanmay Singh")

# ================================================================= CLASSIFICATION PERFORMANCE METRICS
CLS_METRICS = [("Accuracy", "Classification Accuracy", "Share of districts classified correctly", "accuracy", cmp5, True),
               ("Precision", "Positive Predictive Value", "Of districts predicted High, share truly High", "precision", full, True),
               ("Recall", "Sensitivity / True Positive Rate", "Of truly High districts, share found", "recall", full, True),
               ("F1-score", "Harmonic mean of precision and recall", "Balances precision and recall", "f1", cmp5, True),
               ("Specificity", "True Negative Rate", "Of truly Low districts, share found", "specificity", full, True),
               ("ROC-AUC", "Area Under the ROC Curve", "Ability to separate High from Low", "roc_auc", cmp5, True),
               ("Log loss", "Cross-Entropy Loss", "Quality of predicted probabilities (lower is better)", "log_loss", full, False),
               ("MCC", "Matthews Correlation Coefficient", "Uses all four confusion-matrix cells", "mcc", full, True),
               ("Cohen's kappa", "Cohen's Kappa Coefficient", "Agreement with true labels, corrected for chance", "kappa", full, True),
               ("Balanced acc.", "Average class recall", "Accuracy that is fair to both classes", "bal_acc", cmp5, True)]
rows = [["Metric", "Full form", "Purpose", "B-ACPN", "Best other model"]]
for short, full_, purpose, k, src, hi in CLS_METRICS:
    if src is None:
        rows.append([short, full_, purpose, PENDING, PENDING]); continue
    sv = src.set_index("model")[k].dropna()
    others = sv.drop("B-ACPN (proposed)", errors="ignore")
    bo = others.idxmax() if hi else others.idxmin()
    rows.append([short, full_, purpose, f"{sv['B-ACPN (proposed)']:.3f}", f"{SHORTN[bo]} {others[bo]:.3f}"])
s = new_slide("Classification Performance Metrics",
              "All ten classification metrics, with what each one measures, for the proposed model and the best of "
              "the five existing models on that metric. Values are means over the same 25 folds; the full per-model "
              "values are in the comparison and detailed-metrics tables.", size=26)
table(s, 0.4, 1.25, 9.2, rows, [1.35, 2.35, 3.0, 1.0, 1.5], size=11, row_h=0.47, highlight=())
if cmp5 is not None and full is not None:
    _bm = [r[0] for r in rows[1:] if r[3] != PENDING and
           ((float(r[3]) >= float(r[4].split()[-1])) == next(c[5] for c in CLS_METRICS if c[0] == r[0]))]
    _cls_note = (f"B-ACPN is best on {len(_bm)} of 10 ({', '.join(_bm)}); not best on "
                 f"{', '.join(r[0] for r in rows[1:] if r[0] not in _bm) or 'none'}.")
else:
    _cls_note = PENDING
textbox(s, 0.4, 6.5, 9.2, 0.45, _cls_note + " Lower is better only for log loss.", 11, color=INK)
owner(s, "Tanmay Singh")

# ================================================================= ALL CLASSIFICATION METRICS BEFORE vs AFTER PREPROCESSING
BA_METRICS = [("Accuracy", "acc", True), ("Precision", "precision", True), ("Recall", "recall", True),
              ("F1-score", "f1", True), ("Specificity", "specificity", True), ("ROC-AUC", "auc", True),
              ("Log loss", "log_loss", False), ("MCC", "mcc", True), ("Cohen's kappa", "kappa", True),
              ("Balanced acc.", "bal_acc", True), ("RMSE", "rmse", False)]
rows = [["Metric"] + [SHORTN[m] for m, _ in MODELS]]
_better = {m: [0, 0, 0] for m, _ in MODELS}  # improved, unchanged, worse
for name, k, hi in BA_METRICS:
    r = [name + ("" if hi else " ↓")]
    for m, _ in MODELS:
        if ba is None:
            r.append(PENDING); continue
        b0, b1 = ba.set_index("model").loc[m, [f"{k}_before", f"{k}_after"]]
        if pd.isna(b0) or pd.isna(b1):
            r.append("n/a"); continue
        r.append(f"{b0:.3f} → {b1:.3f}")
        d = (round(b1, 3) - round(b0, 3)) * (1 if hi else -1)
        d = 0 if abs(d) < 5e-4 else d
        _better[m][0 if d > 0 else (1 if d == 0 else 2)] += 1
    rows.append(r)
s = new_slide("Classification Metrics: Before vs After",
              "Every classification metric for every model, trained on the raw features (before) and on the "
              "preprocessed features (after), on the same 25 folds. Arrows read before → after; for log loss and "
              "RMSE lower is better. B-ACPN improves or stays equal on all of them. The standard MLP gains recall, "
              "ROC-AUC and F1 but loses precision and specificity. Tree models and Spark's internally standardised "
              "models do not change.", size=22)
table(s, 0.4, 1.25, 9.2, rows, [1.45] + [1.29] * 6, size=10, row_h=0.42, highlight=())
if ba is not None:
    _a = _better["B-ACPN (proposed)"]; _m = _better["Standard MLP"]
    _note = (f"B-ACPN: {_a[0]} of {sum(_a)} metrics improve, {_a[1]} unchanged, {_a[2]} worse. "
             f"MLP: {_m[0]} improve, {_m[1]} unchanged, {_m[2]} worse. ↓ = lower is better. "
             "Linear SVM gives no probabilities (n/a).")
else:
    _note = PENDING
textbox(s, 0.4, 6.4, 9.2, 0.5, _note, 11, color=INK)
owner(s, "Tanmay Singh")

# ================================================================= 17 ABLATION
VAR = ["Full B-ACPN", "Deep stage only (no tree stage)", "Tree stage only (no deep stage)", "Without climate push",
       "Without development gate", "Without feature attention", "Single network (no ensemble)"]
rows = [["Variant", "ROC-AUC", "Change vs full", "p vs full"]]
for v in VAR:
    rows.append([v, g(abl, v, "roc_auc"), "—" if v == "Full B-ACPN" else g(abl, v, "delta", "{:+.3f}"),
                 "—" if v == "Full B-ACPN" else g(abl, v, "p_value")])
if abl is not None:
    fa = abl[abl.model == "Full B-ACPN"].iloc[0]
    hurt = abl[(abl.delta < -fa.roc_auc_sd)].model.tolist()
    nohelp = abl[(abl.model != "Full B-ACPN") & (abl.delta >= -fa.roc_auc_sd)].model.tolist()
    ab_items = [("Removal hurts by more than 1 SD: ", ", ".join(hurt) if hurt else "no single component"),
                ("Within 1 SD (no clear accuracy gain): ", ", ".join(nohelp) if nohelp else "none")]
else:
    ab_items = [("Result: ", PENDING)]
s = new_slide("Ablation Study", "Each row removes one part of B-ACPN and re-runs it on the same folds. If "
              "accuracy drops by more than one standard deviation, that part helps prediction; if not, the part is "
              "kept only for what it explains, and we say so.")
table(s, 0.4, 1.3, 9.2, rows, [4.6, 1.5, 1.6, 1.5], size=13, row_h=0.52, highlight=(1,))
bullets(s, 0.4, 5.6, 9.2, 1.4, ab_items, size=14, space=3)
owner(s, "Devanshu Kumar")

# ================================================================= 18 CLIMATE PUSH ANALYSIS
pq = (push or {}).get("by_quadrant", {})
rows = [["Group", "Districts", "Mean climate push", "Push > 0", "Top climate factor"]]
for k, lab in [("High dev / Low vul", "High dev / Low vul (expected)"), ("Low dev / High vul", "Low dev / High vul (expected)"),
               ("High dev / High vul", "Anomaly A: developed yet vulnerable"), ("Low dev / Low vul", "Anomaly B: less developed yet resilient")]:
    d = pq.get(k, {})
    rows.append([lab, str(q[k]), f"{d['push']:+.2f}" if d else PENDING,
                 f"{d['share_positive']:.0%}" if d else PENDING, d.get("factor", PENDING)])
for r_, d in sorted(((push or {}).get("by_regime") or {}).items()):
    rows.append([f"Spark k-means climate regime {r_}", str(d["n"]), f"{d['push']:+.2f}", "—", d["factor"]])
s = new_slide("Climate-Push Analysis: What It Explains",
              "The climate push answers our research question. A positive push means climate makes a district more "
              "vulnerable than its development predicts. We compare Anomaly A with the expected developed districts, "
              "and Anomaly B with the expected less-developed districts.", size=24)
table(s, 0.4, 1.25, 9.2, rows, [3.6, 1.0, 1.6, 1.0, 2.0], size=12, row_h=0.45, highlight=(3, 4))
y0 = 1.25 + 0.45 * len(rows) + 0.15
if push:
    textbox(s, 0.4, y0, 9.2, 0.4, f"Anomaly A vs expected developed: p = {push['anomaly_A_vs_expected_p']:.3g};  "
            f"Anomaly B vs expected less-developed: p = {push['anomaly_B_vs_expected_p']:.3g};  "
            f"mean MC Dropout SD of push = {push['mean_push_mc_sd']:.2f}", 12, color=GREY)
textbox(s, 0.4, y0 + 0.45, 4.45, 0.35, "Top 5 climate-driven districts (push)", 14, True, NAVY)
textbox(s, 5.15, y0 + 0.45, 4.45, 0.35, "Top 5 development-driven (climate share)", 14, True, NAVY)
top_c = (push or {}).get("top_climate", []) or [PENDING]
top_d = (push or {}).get("top_development", []) or [PENDING]
bullets(s, 0.4, y0 + 0.85, 4.6, 1.6, top_c, size=12, space=0)
bullets(s, 5.15, y0 + 0.85, 4.6, 1.6, top_d, size=12, space=0)
owner(s, "Devanshu Kumar")

# ================================================================= FIGURES (only from here on)
FIG = ROOT / "figures"


def fig_slide(title, notes, left, right, cap_l, cap_r, who, size=28, stacked=False):
    s = new_slide(title, notes, size)
    if right is None:
        fig_or_panel(s, 0.4, 1.35, 9.2, 5.0, FIG / left, cap_l, "PENDING")
        textbox(s, 0.4, 6.45, 9.2, 0.5, cap_l, 12, color=GREY, align=PP_ALIGN.CENTER)
    elif stacked:
        fig_or_panel(s, 0.4, 1.3, 9.2, 2.65, FIG / left, cap_l, "PENDING")
        fig_or_panel(s, 0.4, 4.05, 9.2, 2.65, FIG / right, cap_r, "PENDING")
        textbox(s, 0.4, 6.72, 9.2, 0.32, f"Top: {cap_l}.  Bottom: {cap_r}.", 12, color=GREY, align=PP_ALIGN.CENTER)
    else:
        fig_or_panel(s, 0.4, 1.35, 4.5, 5.0, FIG / left, cap_l, "PENDING")
        fig_or_panel(s, 5.1, 1.35, 4.5, 5.0, FIG / right, cap_r, "PENDING")
        textbox(s, 0.4, 6.45, 4.5, 0.5, cap_l, 12, color=GREY, align=PP_ALIGN.CENTER)
        textbox(s, 5.1, 6.45, 4.5, 0.5, cap_r, 12, color=GREY, align=PP_ALIGN.CENTER)
    owner(s, who)


fig_slide("Preprocessing: Before vs After (Figures)",
          "Top: the data-quality problems in the raw sources and after preprocessing; missing values, unmatched and "
          "wrongly matched districts all go to zero without imputation. Bottom: the raw features live on very "
          "different scales, and after standardisation they are comparable.",
          "prep_quality_before_after.png", "prep_feature_scaling.png",
          "Data-quality problems before vs after", "Feature scales before (raw) vs after (standardised)",
          "Sehjodh Grewal", size=26, stacked=True)
fig_slide("Preprocessing Results (Figures)",
          "Top: how many districts survive each preprocessing stage when nothing is imputed. Bottom: how much "
          "signal each preprocessed feature carries on its own, before any model is trained.",
          "prep_retention.png", "prep_feature_signal.png",
          "District retention 640 → 469", "Single-feature ROC-AUC (blue development, orange hazard)",
          "Sehjodh Grewal", size=26, stacked=True)
fig_slide("HDFS Storage (Figure)",
          "HDFS storage by zone, measured with hdfs dfs -du. The 30 years of rainfall grids dominate; the "
          "staging zone holds the 7 million district-day rows that MapReduce reads.",
          "hdfs_storage.png", None, f"HDFS storage by zone ({REF})", None, "Deepankar")
fig_slide("MapReduce & Spark Results (Figures)",
          "Left: MapReduce runtime against input size for 1, 10 and 30 years on a single node. Right: the "
          "silhouette score used to choose the number of Spark k-means climate regimes.",
          "mapreduce_scalability.png", "kmeans_silhouette.png",
          f"MapReduce scaling, single node ({REF})", "Spark k-means: silhouette by k", "Deepankar")
fig_slide("B-ACPN Training & Attention (Figures)",
          "Left: B-ACPN deep-stage training and validation loss; training stops at the best validation epoch. Right: the mean "
          "attention weight of each climate feature, showing which hazards drive the climate push.",
          "bacpn_training_curve.png", "attention_importance.png",
          "B-ACPN deep stage: training curve (early stopping)", "Feature attention (mean weight)", "Devanshu Kumar")
fig_slide("Ablation (Figure)",
          "Each bar is B-ACPN with one component removed, on the same 25 folds. A drop larger than one standard "
          "deviation means the part helps accuracy; parts within one SD are kept for the explanation they give.",
          "ablation.png", None, "Ablation: one B-ACPN component removed at a time", None, "Devanshu Kumar")
fig_slide("Climate-Push Visualisation",
          "Left: the climate push of every district by group, with the group mean. Right: an India map of the "
          "climate push, so the panel can see where climate drives vulnerability beyond development.",
          "climate_push_by_quadrant.png", "climate_push_map.png",
          "Climate push by group (bar = mean)", "Climate push per district (469)", "Devanshu Kumar")
fig_slide("Results of Classification (Figures)",
          "Top: ROC and precision-recall curves for B-ACPN and the five baselines, pooled over the 25 folds. Bottom: "
          "mean and standard deviation of four metrics for every model.",
          "roc_curves.png", "comparison_bars.png",
          "ROC and precision-recall curves (25 folds)", "Comparison: mean ± SD over 25 folds",
          "Tanmay Singh", size=26, stacked=True)
fig_slide("Confusion Matrices & Stability (Figures)",
          "Top: confusion matrices of all six models summed over the 25 test folds. Bottom: the spread of ROC-AUC "
          "across the folds, which shows how stable each model is.",
          "confusion_matrices.png", "fold_boxplots.png",
          "Confusion matrices, all models", "ROC-AUC per fold, all models", "Tanmay Singh", size=26, stacked=True)
fig_slide("Effect of Preprocessing (Figure)",
          "Accuracy, F1, ROC-AUC and RMSE of every model on raw versus preprocessed features, with the change written "
          "above each pair. The neural models change; tree models and Spark's internally standardised models do not.",
          "prep_model_gain.png", None, "Accuracy, F1, ROC-AUC and RMSE before vs after preprocessing", None,
          "Tanmay Singh")

# ================================================================= 20 KEY FINDINGS
rg = ar["regression"]
if cmp5 is not None and push:
    _a = cmp5[cmp5.model == "B-ACPN (proposed)"].iloc[0]
    _b = cmp5.sort_values("roc_auc").iloc[-1]
    bacpn_finding = (f"ROC-AUC {_a.roc_auc:.3f} vs best {SHORTN[_b.model]} {_b.roc_auc:.3f}; climate push "
                    f"Anomaly A {push['anomaly_A_mean_push']:+.2f} vs expected {push['expected_high_dev_low_vul_mean_push']:+.2f}")
else:
    bacpn_finding = PENDING
s = new_slide("Key Findings", "Development explains only 13% of the variation in vulnerability; adding climate "
              "raises that to 33%, and state alone explains 48%. Anomaly A districts have about twice as many "
              "tropical nights, and Anomaly B districts are drier. Limitations are listed honestly.")
bullets(s, 0.4, 1.3, 5.0, 5.6, [
    ("Development is not enough: ", f"R² {rg['m1_dev_only']['r2']:.3f}; Pearson {ar['association']['development_score']['pearson']:.3f}"),
    ("Climate adds signal: ", f"R² rises to {rg['m2_dev_climate']['r2']:.3f} with the 8 hazard variables"),
    ("Anomalies are common: ", f"{ar['anomalies']}/469 districts ({ar['anomaly_share'] * 100:.1f}%)"),
    ("Anomaly A: ", "≈2× tropical nights (91 vs 45 per year)"),
    ("Anomaly B: ", "drier (863 vs 1078 mm annual rain)"),
    ("Limitations: ", "2011 Census vs 2021 index; 1° temperature grid; single-node Hadoop; 469 districts")],
    size=15, space=10)
cd = CategoryChartData()
cd.categories = ["Dev only", "Dev + climate", "State only"]
cd.add_series("R²", (rg["m1_dev_only"]["r2"], rg["m2_dev_climate"]["r2"], rg["state_only"]["r2"]))
ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(5.6), Inches(1.4), Inches(4.0), Inches(4.6), cd).chart
ch.has_legend = False
ch.has_title = True
ch.chart_title.text_frame.text = "Variance in vulnerability explained (R²)"
style_run(ch.chart_title.text_frame.paragraphs[0].runs[0], 13, True, NAVY)
pl = ch.plots[0]; pl.gap_width = 60; pl.has_data_labels = True
pl.data_labels.number_format = "0.000"; pl.data_labels.number_format_is_linked = False
pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END; pl.data_labels.font.size = Pt(12)
pl.series[0].format.fill.solid(); pl.series[0].format.fill.fore_color.rgb = TEAL
ch.value_axis.has_major_gridlines = False; ch.value_axis.maximum_scale = 0.6
ch.value_axis.minimum_scale = 0; ch.value_axis.major_unit = 0.1
ch.value_axis.tick_labels.number_format = "0.0"; ch.value_axis.tick_labels.number_format_is_linked = False
ch.value_axis.tick_labels.font.size = Pt(11); ch.category_axis.tick_labels.font.size = Pt(11)
textbox(s, 5.6, 6.1, 4.0, 0.5, f"Source: {REF}", 11, color=GREY, align=PP_ALIGN.CENTER)
owner(s, "Tanmay Singh")

# ================================================================= 21 PLAN OF ACTION
team = json.loads((ROOT / "docs" / "team.json").read_text())
s = new_slide("Plan of Action / Member Tasks",
              "Each member owns one stage with no overlap: Sehjodh the dataset and preprocessing, Deepankar the Big "
              "Data pipeline, Devanshu the novel deep-learning model, and Tanmay the comparison with existing models "
              "and the visualisation. Each member presents only the slides of their own stage.")
rows = [["Member", "Role", "Key Responsibilities", "Status"]]
for mbr in team["members"]:
    rows.append([mbr["name"], mbr["role"], mbr["responsibilities"], mbr["completion"]])
table(s, 0.4, 1.25, 9.2, rows, [1.6, 1.8, 4.4, 1.6], size=11, row_h=0.9)
textbox(s, 0.4, 6.45, 9.2, 0.55, "Next (Review 3): full 1901–2024 rainfall in HDFS and re-timing; B-ACPN tuning; "
        "comparison with CEEW / DST district rankings; final report.", 12, color=INK)
owner(s, "Tanmay Singh")

# ================================================================= 22 REFERENCES
refs = [
    "A. Mohanty and S. Wadhawan, Mapping India's Climate Vulnerability, CEEW, 2021.",
    "NITI Aayog, District-Level Climate Vulnerability Indicators, NDAP dataset 7160.",
    "Census of India 2011, Primary Census Abstract.",
    "D. S. Pai et al., MAUSAM 65(1), 2014 (IMD gridded rainfall).",
    "A. K. Srivastava et al., Atmospheric Science Letters 10, 2009 (IMD gridded temperature).",
    "H.-T. Cheng et al., Wide & Deep Learning for Recommender Systems, DLRS 2016.",
    "S. Ö. Arık and T. Pfister, TabNet: Attentive Interpretable Tabular Learning, AAAI 2021.",
    "L. Breiman, Random Forests, Machine Learning 45, 2001.",
    "D. H. Wolpert, Stacked Generalization, Neural Networks 5, 1992.",
    "K. He et al., Deep Residual Learning for Image Recognition, CVPR 2016.",
    "Y. Gal and Z. Ghahramani, Dropout as a Bayesian Approximation, ICML 2016.",
    "A. N. Angelopoulos and S. Bates, A Gentle Introduction to Conformal Prediction, arXiv:2107.07511, 2021.",
    "K. Shvachko et al., The Hadoop Distributed File System, MSST 2010.",
    "J. Dean and S. Ghemawat, MapReduce, Communications of the ACM, 2008.",
    "A. Thusoo et al., Hive, VLDB 2009.",
    "M. Zaharia et al., Apache Spark: A Unified Engine for Big Data Processing, CACM 59(11), 2016.",
    "X. Meng et al., MLlib: Machine Learning in Apache Spark, JMLR 17, 2016.",
    "DataMeet, Community Created Maps of India."]
s = new_slide("References", "These are the data sources, the papers behind each B-ACPN component, and the "
              "Big Data tools we used.")
tb = s.shapes.add_textbox(Inches(0.4), Inches(1.3), Inches(9.2), Inches(5.7))
tf = tb.text_frame; tf.word_wrap = True
for i, rtxt in enumerate(refs):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.space_after = Pt(4)
    r = p.add_run(); r.text = f"[{i + 1}]  "; style_run(r, 13, True, NAVY)
    r = p.add_run(); r.text = rtxt; style_run(r, 13)
owner(s, "Tanmay Singh")

# ================================================================= 23 PHOTO
s = new_slide("Geotagged Photo: Review 2", "The team adds the geotagged photo taken during the Review 2 "
              "presentation here.")
ph = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.6), Inches(1.4), Inches(8.8), Inches(5.4))
ph.fill.background(); ph.line.color.rgb = GREY; ph.line.width = Pt(1.5); ph.shadow.inherit = False
ph.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
write(ph.text_frame, "Insert geotagged photo", 20, False, GREY, PP_ALIGN.CENTER)
owner(s, "Tanmay Singh")

# ================================================================= 24 COLAB
s = new_slide("Google Colab Implementation Link",
              "The Colab notebook runs the whole project in the Review 2 order, with every table printed before "
              "any figure. The model sections can also run on their own from the committed analysis table.")
textbox(s, 0.4, 1.3, 9.2, 0.35, "Colab (with outputs):", 14, True, NAVY)
textbox(s, 0.4, 1.62, 9.2, 0.55, "https://colab.research.google.com/github/deepankarkumar1310/hadoop-ml-algorithm-/"
        "blob/main/notebooks/Review2_Colab.ipynb", 12, color=TEAL)
textbox(s, 0.4, 2.2, 9.2, 0.35, "GitHub:  https://github.com/deepankarkumar1310/hadoop-ml-algorithm-", 13, True, NAVY)
table(s, 0.4, 2.75, 9.2, [
    ["Notebook section", "What it shows", "Owner"],
    ["0–2 Setup, data, preprocessing", "Install, IMD download to Drive, Table 1 + error metrics", "Sehjodh"],
    ["3–7 Hadoop, HDFS, MapReduce, Hive, Spark", "Cluster, HDFS load (Table 2), YARN job + timings, Hive, k-means (Table 3)", "Deepankar"],
    ["8 B-ACPN (proposed model)", "Architecture, model card (Table 4), ablation, climate push (Table 6)", "Devanshu"],
    ["9–10 Comparison", "5 existing models, all metrics, preprocessing before/after (Table 5)", "Tanmay"],
    ["11 Visualisations", "All figures, only after every table", "Each owner"],
    ["12 Slides", "Builds this deck from results/", "Tanmay"]],
    [2.7, 5.0, 1.5], size=12, row_h=0.55)
owner(s, "Tanmay Singh")

# ----------------------------------------------------------------- save
cp = prs.core_properties
cp.author = cp.last_modified_by = "Deepankar"
cp.title = "Beyond Development: Explaining Climate Resilience Across Indian Districts - Review 2"
cp.subject, cp.keywords, cp.comments = "Big Data Review 2", "", ""
OUT.parent.mkdir(parents=True, exist_ok=True)
# Final order: intro → preprocessing → Big Data tools → ML models → visualisations → closing.
# Slides are built in the original order above and moved here by title.
ORDER = ["Beyond Development", "Contents", "Problem Statement", "Dataset",
         # 1. preprocessing: steps, results, error metrics, figures, effect on the models
         "Preprocessing: Big Data Analysis Steps", "Results of Preprocessing", "Preprocessing Error Metrics",
         "Preprocessing Performance Metrics", "Preprocessing: Before vs After (Figures)",
         "Preprocessing Results (Figures)", "Effect of Preprocessing on the Models",
         "Classification Metrics: Before vs After", "Effect of Preprocessing (Figure)",
         # 2. Big Data tools
         "Overall System Architecture", "Big Data Platform", "Big Data Processing", "HDFS Storage (Figure)",
         "MapReduce & Spark Results",
         # 3. ML models: proposed model, then comparison with existing models
         "Proposed Model", "Classification: B-ACPN Model Card", "Ablation Study", "Climate-Push Analysis",
         "B-ACPN Training & Attention", "Ablation (Figure)", "Climate-Push Visualisation",
         "Comparison with Existing", "Detailed Classifier Metrics", "Confusion Matrix, CIs",
         "Classification Performance Metrics", "Results of Classification (Figures)", "Confusion Matrices & Stability",
         # closing
         "Key Findings", "Plan of Action", "References", "Geotagged Photo", "Google Colab Implementation"]


def _title(sl):
    return next((sh.text_frame.text for sh in sl.shapes if sh.has_text_frame and sh.text_frame.text.strip()), "")


_ids = prs.slides._sldIdLst
_els = list(_ids)
_titles = [_title(sl) for sl in prs.slides]
_new = []
for key in ORDER:
    hit = [i for i, t in enumerate(_titles) if t.startswith(key) and i not in _new]
    assert len(hit) == 1, (key, hit)
    _new.append(hit[0])
assert sorted(_new) == list(range(len(_els))), "every slide must appear exactly once"
for el in _els:
    _ids.remove(el)
for i in _new:
    _ids.append(_els[i])
prs.save(str(OUT))
print(f"saved {OUT.relative_to(ROOT)} with {len(prs.slides)} slides")
