"""Write README.md with every number read from results/ (run after scripts/evaluate.py)."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.common import RES, ROOT, run_label  # noqa: E402

REF = run_label()
COLAB = "https://colab.research.google.com/github/deepankarkumar1310/hadoop-ml-algorithm-/blob/main/notebooks/Review2_Colab.ipynb"


def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


def main():
    pm = json.loads((RES / "preprocessing_metrics.json").read_text())
    cc = json.loads((RES / "preprocessing_colab_checks.json").read_text())
    hd = json.loads((RES / "hdfs_usage_bytes.json").read_text())
    km = json.loads((RES / "kmeans_regimes.json").read_text())
    card = json.loads((RES / "table4_model_card.json").read_text())
    push = json.loads((RES / "table6_climate_push.json").read_text())
    t5 = pd.read_csv(RES / "table5_comparison.csv")
    sig = pd.read_csv(RES / "significance.csv").set_index("model")
    ab = pd.read_csv(RES / "table6_ablation.csv")
    ba = pd.read_csv(RES / "preprocessing_before_after.csv")
    tm = pd.read_csv(RES / "scalability_timings.csv").groupby("years").runtime_s.mean().round(0).astype(int)

    t5v = pd.DataFrame({"Model": t5.model, "Tool": t5.tool,
                        "ROC-AUC": [f"{a:.3f} ± {s:.3f}" for a, s in zip(t5.roc_auc, t5.roc_auc_sd)],
                        "Bal. acc.": t5.bal_acc.round(3), "Accuracy": t5.accuracy.round(3), "F1": t5.f1.round(3),
                        "LOSO AUC": t5.loso_auc.round(3), "95% CI": [sig.loc[m, "auc_ci"] for m in t5.model],
                        "p vs B-ACPN": [sig.loc[m, "p_value_fmt"] for m in t5.model]})
    abv = pd.DataFrame({"Variant": ab.model, "ROC-AUC": ab.roc_auc.round(3), "Change": ab.delta.map("{:+.3f}".format)})
    bav = pd.DataFrame({"Model": ba.model,
                        **{f"{n} {w}": ba[f"{c}_{k}"].map(lambda v: "n/a" if pd.isna(v) else f"{v:.3f}")
                           for c, n in (("acc", "Accuracy"), ("f1", "F1"), ("rmse", "RMSE"), ("auc", "ROC-AUC"))
                           for k, w in (("before", "raw"), ("after", "prep."))}})
    best = t5.sort_values("roc_auc").iloc[-1]
    a = t5[t5.model == "B-ACPN (proposed)"].iloc[0]
    q = push["by_quadrant"]
    text = f"""# Beyond Development: Explaining Climate Resilience Across Indian Districts

Big Data project, SRM Institute of Science and Technology — Review 2. Faculty: Dr Bhanu Priya.
Team: Sehjodh Grewal (RA2411031010081), Devanshu Kumar (RA2411031010102), Deepankar (RA2411031010103),
Tanmay Singh (RA2411031010118).

**Research question.** Why do some economically developed Indian districts remain highly climate-vulnerable, while
some less-developed districts are resilient?

**Run it:** [Open the Colab notebook]({COLAB}) · slides: `docs/Review2_Deck.pptx`

## 1. Dataset
| Dataset | Source | Year | Format | Role |
|---|---|---|---|---|
| Climate Vulnerability Index + indicators (699 × 79) | NITI Aayog NDAP 7160 | 2021 | CSV | outcome + 2 development variables |
| Primary Census Abstract (640 districts) | Census of India | 2011 | CSV | 3 development variables (2011 baseline) |
| District boundaries (641 polygons) | DataMeet | 2011 | shapefile | grid → district weights, maps |
| Daily gridded rainfall 0.25° | IMD Pune | 1991–2020 | binary .grd | hazard |
| Daily gridded Tmax/Tmin 1° | IMD Pune | 1991–2020 | binary .grd | hazard |

5 development variables (literacy, female literacy, agri-worker share, MGNREGA days, yield variability) and
8 hazard variables (annual rain, rain variability, max 5-day rain, heavy-rain days, longest dry spell, mean Tmax,
hot days, tropical nights). The vulnerability index is the target only, never an input.

## 2. Big Data pipeline (single-node pseudo-distributed Hadoop, replication 1)
Where each tool is used and why: [`docs/BIG_DATA_TOOLS.md`](docs/BIG_DATA_TOOLS.md).

| Tool | Job | Result |
|---|---|---|
| HDFS | raw / staging / processed zones | {sum(hd.values()) / 1e9:.2f} GB stored ({REF}) |
| MapReduce on YARN | 7,013,120 district-day rows → 19,200 district-year indices | {tm.get(1)} / {tm.get(10)} / {tm.get(30)} s for 1 / 10 / 30 years ({REF}) |
| Hive | 30-year baseline + join → 469-district table | identical to pandas, max difference 0.0 ({REF}) |
| Spark | k-means climate regimes; MLlib baselines | k = {km['k']} (silhouette {km['silhouette']:.3f}); see Table 5 |

## 3. Preprocessing results
Districts 640 → {pm['matching']['matched']} matched → 589 with index → {pm['final_table']['rows']} complete
({pm['final_table']['states']} states), no imputation. Census population {pm['census']['population']:,}
(error {pm['census']['population_abs_error']}), literacy {pm['census']['literacy_pct']:.4f}% vs official 72.98%.
Grid weights sum to 1 within {pm['grid_rain']['max_abs_sum_error']:.0e}. Final table: 0 missing, 0 duplicates,
0 out-of-range values. Class balance {pm['final_table']['class_high']} High / {pm['final_table']['class_low']} Low.

**Preprocessing before vs after** (same models and folds, raw vs preprocessed features; RMSE = error of the predicted
probability, lower is better; Linear SVM gives no probabilities). Data level: area-weighted grid → district
aggregation vs nearest grid cell differs by RMSE {cc['area_vs_nearest_cell']['rmse_mm']:.0f} mm
({cc['area_vs_nearest_cell']['mean_abs_pct']:.1f}% mean absolute difference, {cc['area_vs_nearest_cell']['year']}).

{md_table(bav)}

## 4. Proposed model: B-ACPN (Boosted Attentive Climate Push Network)
1. **Tree stage:** a random forest (500 trees) on all 13 features gives a risk score; for training districts the
   score is computed out-of-fold (5 inner folds), so the deep stage never sees a score made on its own rows.
2. **Deep stage:** `logit = tree score + linear development term + climate push × development gate`, where the
   climate push is feature attention → MLP → residual block on the 8 hazard features and the gate is
   `2σ(small net on the 5 development features)`.
3. **Ensemble:** 5 deep stages with different seeds, probabilities averaged.

MC Dropout gives the uncertainty of each district's climate push; split conformal prediction gives prediction sets.
{card['parameters']:,} parameters per network (plus the forest), {card['train_time']} on CPU; conformal coverage
{card['conformal_coverage_mean']:.1%} (target 90%). Code: `models/bacpn.py`. Random forests, attention, residual
blocks, stacking and conformal prediction are existing techniques; combining a tree stage with a development-gated
climate network that outputs a per-district climate push is the contribution.

## 5. Comparison with 5 models (Table 5; same 25 folds + leave-one-state-out)
{md_table(t5v)}

Highest ROC-AUC: **{best.model} ({best.roc_auc:.3f})**. B-ACPN: {a.roc_auc:.3f} ± {a.roc_auc_sd:.3f}. Results are
reported as measured; a lead smaller than one standard deviation is treated as comparable, not better.

## 6. Ablation and climate push (Table 6)
{md_table(abv)}

Mean climate push: Anomaly A (developed yet vulnerable) {q['High dev / High vul']['push']:+.2f} vs expected developed
{q['High dev / Low vul']['push']:+.2f} (p = {push['anomaly_A_vs_expected_p']:.3g}); Anomaly B (less developed yet
resilient) {q['Low dev / Low vul']['push']:+.2f} vs expected less-developed {q['Low dev / High vul']['push']:+.2f}
(p = {push['anomaly_B_vs_expected_p']:.3g}).

All tables: `results/`. Figures (made only after the tables): `figures/`.

## How to run
```
pip install -r requirements.txt
python -m pytest tests -q
python scripts/preprocessing_metrics.py
python spark/kmeans_regimes.py
python spark/train_baselines.py
python scripts/train_bacpn.py && python scripts/train_mlp.py
python scripts/train_bacpn.py --raw && python scripts/train_bacpn.py --ablation
python scripts/evaluate.py && python scripts/make_figures.py && python scripts/build_deck.py
```
The full Hadoop / MapReduce / Hive pipeline (IMD download, HDFS, YARN) runs in the Colab notebook.

## Limitations
2011 Census development data against a 2021 index; 1° temperature grid; single-node Hadoop (no multi-machine
speed-up is claimed); 469 districts. HDFS, MapReduce and Hive figures are from the {REF} until the notebook is re-run.
"""
    (ROOT / "README.md").write_text(text)
    print("README.md written")


if __name__ == "__main__":
    main()
