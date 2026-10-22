# Beyond Development: Explaining Climate Resilience Across Indian Districts

Big Data project, SRM Institute of Science and Technology — Review 2. Faculty: Dr Bhanu Priya.
Team: Sehjodh Grewal (RA2411031010081), Devanshu Kumar (RA2411031010102), Deepankar (RA2411031010103),
Tanmay Singh (RA2411031010118).

**Research question.** Why do some economically developed Indian districts remain highly climate-vulnerable, while
some less-developed districts are resilient?

**Run it:** [Open the Colab notebook](https://colab.research.google.com/github/deepankarkumar1310/hadoop-ml-algorithm-/blob/main/notebooks/Review2_Colab.ipynb) · slides: `docs/Review2_Deck.pptx`

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
| HDFS | raw / staging / processed zones | 1.08 GB stored (Colab run 2026-10-22) |
| MapReduce on YARN | 7,013,120 district-day rows → 19,200 district-year indices | 55 / 140 / 333 s for 1 / 10 / 30 years (Colab run 2026-10-22) |
| Hive | 30-year baseline + join → 469-district table | identical to pandas, max difference 0.0 (Colab run 2026-10-22) |
| Spark | k-means climate regimes; MLlib baselines | k = 2 (silhouette 0.651); see Table 5 |

## 3. Preprocessing results
Districts 640 → 602 matched → 589 with index → 469 complete
(21 states), no imputation. Census population 1,210,854,977
(error 0), literacy 72.9819% vs official 72.98%.
Grid weights sum to 1 within 7e-08. Final table: 0 missing, 0 duplicates,
0 out-of-range values. Class balance 237 High / 232 Low.

**Preprocessing before vs after** (same models and folds, raw vs preprocessed features):

| Model | AUC raw | AUC preprocessed | Gain |
|---|---|---|---|
| A-CPN (proposed) | 0.737 | 0.76 | +0.023 |
| Logistic Regression | 0.747 | 0.747 | +0.000 |
| Random Forest | 0.805 | 0.804 | -0.000 |
| Gradient-Boosted Trees | 0.811 | 0.811 | -0.000 |
| Linear SVM | 0.755 | 0.755 | +0.000 |
| Standard MLP | 0.759 | 0.784 | +0.025 |

## 4. Proposed model: A-CPN (Attentive Climate Push Network)
`logit = linear development baseline (5 features) + climate push (feature attention → KAN layer → residual MLP, 8 features)`,
with MC Dropout uncertainty and split conformal prediction sets. 1,775 parameters,
2.0 s per fold on CPU; conformal coverage 91.2% (target 90%).
Code: `models/acpn.py`. Feature attention, KAN layers, residual blocks and conformal prediction are existing techniques;
their combination into a development-anchored network that outputs a per-district climate push is the contribution.

## 5. Comparison with 5 models (Table 5; same 25 folds + leave-one-state-out)
| Model | Tool | ROC-AUC | Bal. acc. | Accuracy | F1 | LOSO AUC | 95% CI | p vs A-CPN |
|---|---|---|---|---|---|---|---|---|
| A-CPN (proposed) | Novel DL · PyTorch | 0.760 ± 0.045 | 0.701 | 0.701 | 0.693 | 0.694 | 0.711 – 0.810 | — |
| Logistic Regression | Spark MLlib | 0.747 ± 0.042 | 0.691 | 0.692 | 0.701 | 0.661 | 0.700 – 0.793 | 0.602 |
| Random Forest | Spark MLlib | 0.804 ± 0.036 | 0.715 | 0.715 | 0.708 | 0.706 | 0.764 – 0.845 | 0.047 |
| Gradient-Boosted Trees | Spark MLlib | 0.811 ± 0.034 | 0.726 | 0.725 | 0.724 | 0.698 | 0.773 – 0.849 | 0.071 |
| Linear SVM | Spark MLlib | 0.755 ± 0.039 | 0.696 | 0.696 | 0.708 | 0.691 | 0.711 – 0.798 | 0.835 |
| Standard MLP | PyTorch | 0.784 ± 0.037 | 0.699 | 0.699 | 0.692 | 0.71 | 0.743 – 0.825 | 0.311 |

Best ROC-AUC: **Gradient-Boosted Trees (0.811)**. A-CPN: 0.760. Results are reported as measured;
A-CPN's contribution is the per-district explanation below.

## 6. Ablation and climate push (Table 6)
| Variant | ROC-AUC | Change |
|---|---|---|
| Full A-CPN | 0.76 | +0.000 |
| Without feature attention | 0.774 | +0.013 |
| Without KAN layer (plain MLP) | 0.784 | +0.023 |
| Without residual block | 0.756 | -0.005 |
| Climate only (no development baseline) | 0.757 | -0.004 |
| Development only (no climate push) | 0.67 | -0.090 |
| Linear climate part (no deep layers) | 0.733 | -0.028 |

Mean climate push: Anomaly A (developed yet vulnerable) +0.32 vs expected developed
-0.53 (p = 6.31e-07); Anomaly B (less developed yet
resilient) -0.99 vs expected less-developed +0.81
(p = 2.46e-18).

All tables: `results/`. Figures (made only after the tables): `figures/`.

## How to run
```
pip install -r requirements.txt
python -m pytest tests -q
python scripts/preprocessing_metrics.py
python spark/kmeans_regimes.py
python spark/train_baselines.py
python scripts/train_acpn.py && python scripts/train_mlp.py
python scripts/train_acpn.py --raw && python scripts/train_acpn.py --ablation
python scripts/evaluate.py && python scripts/make_figures.py && python scripts/build_deck.py
```
The full Hadoop / MapReduce / Hive pipeline (IMD download, HDFS, YARN) runs in the Colab notebook.

## Limitations
2011 Census development data against a 2021 index; 1° temperature grid; single-node Hadoop (no multi-machine
speed-up is claimed); 469 districts. HDFS, MapReduce and Hive figures are from the Colab run 2026-10-22 until the notebook is re-run.
