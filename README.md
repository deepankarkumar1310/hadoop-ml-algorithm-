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

**Preprocessing before vs after** (same models and folds, raw vs preprocessed features; RMSE = error of the predicted
probability, lower is better; Linear SVM gives no probabilities). Data level: area-weighted grid → district
aggregation vs nearest grid cell differs by RMSE 287 mm
(11.1% mean absolute difference, 2020).

| Model | Accuracy raw | Accuracy prep. | F1 raw | F1 prep. | RMSE raw | RMSE prep. | ROC-AUC raw | ROC-AUC prep. |
|---|---|---|---|---|---|---|---|---|
| B-ACPN (proposed) | 0.735 | 0.738 | 0.726 | 0.728 | 0.422 | 0.417 | 0.806 | 0.820 |
| Logistic Regression | 0.692 | 0.692 | 0.701 | 0.701 | 0.454 | 0.454 | 0.747 | 0.747 |
| Random Forest | 0.715 | 0.715 | 0.708 | 0.708 | 0.427 | 0.427 | 0.805 | 0.804 |
| Gradient-Boosted Trees | 0.726 | 0.725 | 0.724 | 0.724 | 0.432 | 0.433 | 0.811 | 0.811 |
| Linear SVM | 0.696 | 0.696 | 0.708 | 0.708 | n/a | n/a | 0.755 | 0.755 |
| Standard MLP | 0.711 | 0.699 | 0.678 | 0.692 | 0.445 | 0.440 | 0.759 | 0.784 |

## 4. Proposed model: B-ACPN (Boosted Attentive Climate Push Network)
1. **Tree stage:** a random forest (500 trees) on all 13 features gives a risk score; for training districts the
   score is computed out-of-fold (5 inner folds), so the deep stage never sees a score made on its own rows.
2. **Deep stage:** `logit = tree score + linear development term + climate push × development gate`, where the
   climate push is feature attention → MLP → residual block on the 8 hazard features and the gate is
   `2σ(small net on the 5 development features)`.
3. **Ensemble:** 5 deep stages with different seeds, probabilities averaged.

MC Dropout gives the uncertainty of each district's climate push; split conformal prediction gives prediction sets.
780 parameters per network (plus the forest), 11.8 s per fold on CPU; conformal coverage
92.9% (target 90%). Code: `models/bacpn.py`. Random forests, attention, residual
blocks, stacking and conformal prediction are existing techniques; combining a tree stage with a development-gated
climate network that outputs a per-district climate push is the contribution.

## 5. Comparison with 5 models (Table 5; same 25 folds + leave-one-state-out)
| Model | Tool | ROC-AUC | Bal. acc. | Accuracy | F1 | LOSO AUC | 95% CI | p vs B-ACPN |
|---|---|---|---|---|---|---|---|---|
| B-ACPN (proposed) | Novel hybrid DL · PyTorch | 0.820 ± 0.035 | 0.739 | 0.738 | 0.728 | 0.717 | 0.780 – 0.859 | — |
| Logistic Regression | Spark MLlib | 0.747 ± 0.042 | 0.691 | 0.692 | 0.701 | 0.661 | 0.700 – 0.793 | 0.004 |
| Random Forest | Spark MLlib | 0.804 ± 0.036 | 0.715 | 0.715 | 0.708 | 0.706 | 0.764 – 0.845 | 0.042 |
| Gradient-Boosted Trees | Spark MLlib | 0.811 ± 0.034 | 0.726 | 0.725 | 0.724 | 0.698 | 0.773 – 0.849 | 0.550 |
| Linear SVM | Spark MLlib | 0.755 ± 0.039 | 0.696 | 0.696 | 0.708 | 0.691 | 0.711 – 0.798 | 0.006 |
| Standard MLP | PyTorch | 0.784 ± 0.037 | 0.699 | 0.699 | 0.692 | 0.71 | 0.743 – 0.825 | 0.019 |

Highest ROC-AUC: **B-ACPN (proposed) (0.820)**. B-ACPN: 0.820 ± 0.035. Results are
reported as measured; a lead smaller than one standard deviation is treated as comparable, not better.

## 6. Ablation and climate push (Table 6)
| Variant | ROC-AUC | Change |
|---|---|---|
| Full B-ACPN | 0.82 | +0.000 |
| Deep stage only (no tree stage) | 0.8 | -0.019 |
| Tree stage only (no deep stage) | 0.818 | -0.002 |
| Without climate push | 0.816 | -0.004 |
| Without development gate | 0.821 | +0.001 |
| Without feature attention | 0.817 | -0.003 |
| Single network (no ensemble) | 0.801 | -0.018 |

Mean climate push: Anomaly A (developed yet vulnerable) +0.09 vs expected developed
-0.10 (p = 1.56e-05); Anomaly B (less developed yet
resilient) -0.17 vs expected less-developed +0.13
(p = 4.45e-07).

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
speed-up is claimed); 469 districts. HDFS, MapReduce and Hive figures are from the Colab run 2026-10-22 until the notebook is re-run.
