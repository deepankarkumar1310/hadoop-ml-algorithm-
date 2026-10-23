"""Generate notebooks/Review2_Colab.ipynb (run from the repository root).

Review 2 order: dataset -> Big Data pipeline -> preprocessing results -> B-ACPN -> comparison ->
tables; every figure comes after every table. Sections 8-10 also work on their own (fast path)
from the committed data/processed/analysis_table.csv.
"""
import nbformat as nbf

REPO = "https://github.com/deepankarkumar1310/hadoop-ml-algorithm-.git"
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip()))


md("""
# Beyond Development: Explaining Climate Resilience Across Indian Districts — Review 2

**Research question.** Why do some economically developed Indian districts remain highly climate-vulnerable,
while some less-developed districts are resilient?

| Section | Content |
|---|---|
| 0 | Setup: Drive, repository, packages, tests |
| 1 | Data download: IMD 1991–2020 to Google Drive |
| 2 | Preprocessing + Table 1 + preprocessing error metrics |
| 3 | Big Data platform: Hadoop (HDFS + YARN), Hive, Spark |
| 4 | HDFS load, listing, Table 2 |
| 5 | MapReduce on YARN: local test, job, timings, correctness check |
| 6 | Hive: 30-year baseline + join, check against pandas |
| 7 | Spark k-means climate regimes + Table 3 |
| 8 | B-ACPN (proposed model): architecture, parameters, Table 4 |
| 9 | Comparison models: Spark MLlib (LR, RF, GBT, Linear SVM) + MLP; B-ACPN ablation |
| 10 | Table 5 (comparison, metrics; accuracy, F1, RMSE, ROC-AUC before/after preprocessing) and Table 6 (ablation, climate push) |
| 11 | Visualisations (only after every table) |
| 12 | Build the slides; appendix to show code |

**Fast path:** to show only the models, run section 0 and then sections 8–12; they use the committed
`data/processed/analysis_table.csv`. Single-node pseudo-distributed Hadoop (replication 1): no multi-machine
speed-up is claimed.
""")

md("## 0. Setup")
code(f"""
import os, subprocess, sys
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    PROJECT = '/content/drive/MyDrive/climate_project'      # Colab wipes its disk: persistent data lives on Drive
    REPO_DIR, IMD_DIR = f'{{PROJECT}}/repo', f'{{PROJECT}}/imd'
    os.makedirs(PROJECT, exist_ok=True)
    if not os.path.isdir(REPO_DIR):
        subprocess.run(['git', 'clone', '{REPO}', REPO_DIR], check=True)
    os.chdir(REPO_DIR)
    # always use the latest main branch (files the notebook regenerates are replaced; Drive data is untouched)
    subprocess.run(['git', 'fetch', '-q', 'origin', 'main'], check=True)
    subprocess.run(['git', 'checkout', '-q', '-f', '-B', 'main', 'origin/main'], check=True)
    print(subprocess.run(['git', 'log', '-1', '--format=repo at %h %s'], capture_output=True, text=True).stdout.strip())
else:
    os.chdir(os.path.dirname(os.getcwd()) if os.path.basename(os.getcwd()) == 'notebooks' else os.getcwd())
    IMD_DIR = 'data/imd'
START, END = 1991, 2020
os.environ.update(IMD_DIR=IMD_DIR, START=str(START), END=str(END))
print(os.getcwd())
""")
code("""
!python3 -m pip -q install -r requirements.txt
!python3 -c "import torch, pyspark, imdlib; print('torch', torch.__version__, '| pyspark', pyspark.__version__)"
!python3 -m pytest tests -q
""")

md("""
## 1. Dataset and IMD download

| Dataset | Source | Year | Format | Role |
|---|---|---|---|---|
| Climate Vulnerability Index + indicators (699 × 79) | NITI Aayog NDAP 7160 | 2021 | CSV | outcome + 2 development variables |
| Primary Census Abstract (640 districts) | Census of India | 2011 | CSV | 3 development variables (2011 baseline) |
| District boundaries (641 polygons) | DataMeet | 2011 | shapefile | grid → district weights |
| Daily gridded rainfall 0.25° | IMD Pune | 1991–2020 | binary .grd | hazard |
| Daily gridded Tmax / Tmin 1° | IMD Pune | 1991–2020 | binary .grd | hazard |

The IMD files (~848 MB) are downloaded to Google Drive and are never committed to GitHub.
""")
code("""
!python3 scripts/download_imd.py --imd-dir "$IMD_DIR" --start $START --end $END
""")

md("""
## 2. Preprocessing and preprocessing results

P1 Census rates · P2 NDAP↔Census matching · P3 development score and quadrants · P4 grid→district area weights ·
P5 district-day table (missing stays missing) · P6 Chennai orientation gate · P7 standardisation on training folds only.
""")
code("""
!python3 scripts/census_vars.py | tail -8
!python3 scripts/match_districts.py | tail -8
!python3 scripts/build_integrated_table.py | tail -5
!python3 scripts/build_district_day.py --imd-dir "$IMD_DIR" --out-dir data/staging/district_day --start $START --end $END
# error check: our area-weighted aggregation vs the naive nearest-cell method
!python3 scripts/colab_checks.py --check aggregation --imd-dir "$IMD_DIR" --year $END
""")
code("""
# TABLE 1 + preprocessing error metrics (each step checked against a reference)
import json, pandas as pd
!python3 scripts/preprocessing_metrics.py > /dev/null
pm = json.load(open('results/preprocessing_metrics.json'))
ar = json.load(open('results/analysis_results.json'))
t1 = pd.DataFrame([
    ('Census 2011 districts', 640), ('Matched to NDAP', pm['matching']['matched']),
    ('With vulnerability index', ar['preprocessing']['with_index']),
    ('Complete (all 13 features)', pm['final_table']['rows']),
    ('Census population (official 1,210,854,977)', f"{pm['census']['population']:,}"),
    ('National literacy (official 72.98%)', f"{pm['census']['literacy_pct']:.4f}%"),
    ('Class balance High / Low', f"{pm['final_table']['class_high']} / {pm['final_table']['class_low']}"),
    ('NDAP columns >50% empty', f"{pm['ndap']['cols_over_half_empty']} of {pm['ndap']['cols']}"),
], columns=['Stage / check', 'Value'])
display(t1)
display(pd.DataFrame(pm['feature_signal_auc'].items(), columns=['feature', 'single-feature ROC-AUC']))
pm
""")

md("""
## 3. Big Data platform

| Tool | Where | What it does | Why chosen |
|---|---|---|---|
| **HDFS** | `hadoop/setup_hadoop.sh`, `hadoop/load_hdfs.sh` | Stores raw, staging and processed zones | 128 MB blocks for parallel jobs; read directly by MapReduce, Hive, Spark |
| **YARN** | `hadoop/setup_hadoop.sh` | Allocates CPU/memory and runs MapReduce tasks | Hadoop's standard scheduler |
| **Hive** | `hive/setup_hive.sh`, `hive/project.hql` | SQL over HDFS: 30-year averages + join | Aggregation and joins are SQL tasks |
| **Spark** | `spark/setup_spark.sh`, `spark/*.py` | k-means climate regimes; MLlib baselines | In-memory, fast for iterative algorithms |

Java 8 for everything (Hive 3.1.3 fails on Java 11). Colab has no SSH, so every daemon is started directly.
""")
code("""
def load_env(script):
    out = subprocess.run(['bash', '-c', f'source {script} > /tmp/setup.log 2>&1 && env'], capture_output=True, text=True)
    print(open('/tmp/setup.log').read()[-1500:])
    for line in out.stdout.splitlines():
        k, _, v = line.partition('=')
        if k in ('JAVA_HOME', 'HADOOP_HOME', 'HADOOP_CONF_DIR', 'PATH', 'HIVE_HOME', 'HADOOP_STREAMING_JAR',
                 'HDFS_NAMENODE_USER', 'HDFS_DATANODE_USER', 'YARN_RESOURCEMANAGER_USER', 'YARN_NODEMANAGER_USER',
                 'PYSPARK_PYTHON', 'SPARK_LOCAL_IP'):
            os.environ[k] = v

load_env('hadoop/setup_hadoop.sh')
!hadoop version | head -1
!jps
!hdfs dfsadmin -report | head -12
""")

md("## 4. HDFS load and Table 2")
code("""
!bash hadoop/load_hdfs.sh "$IMD_DIR" data/staging/district_day
!hdfs dfs -put -f data/processed/analysis_table.csv /project/processed/
!hdfs dfs -ls /project /project/raw /project/raw/imd
!hdfs dfs -du -s -h /project/raw /project/staging /project/processed /project
""")

md("""
## 5. MapReduce on YARN

**Map:** each district-day row → key (censuscode, year). **Reduce:** sort the days by date and compute 7 yearly
indices (rain total, heavy-rain days, max 5-day rain, longest dry spell, mean Tmax, hot days, tropical nights).
Incomplete years are left blank (never 0); malformed rows go to a Hadoop counter.
""")
code("""
# local test of the same mapper/reducer
!head -2000 data/staging/district_day/district_day_${END}.csv | python3 mapreduce/mapper.py | sort | python3 mapreduce/reducer.py 2>/dev/null | head -3
!bash hadoop/run_mapreduce.sh /project/staging/district_day /project/processed/district_year 2 2>&1 | grep -E "Map input records|Reduce output records|bad_rows|completed successfully|ERROR"
!hdfs dfs -getmerge /project/processed/district_year data/processed/district_year_indices.csv
!wc -l data/processed/district_year_indices.csv
""")
code("""
# scalability: 1 / 10 / 30 years, 2 repeats each (single node)
!python3 scripts/time_mapreduce.py --years 1 10 30 --repeats 2
!cat results/scalability_timings.csv
# correctness check: the 7 MapReduce indices recomputed independently in pandas
!python3 scripts/colab_checks.py --check mapreduce
""")

md("## 6. Hive: 30-year baseline + join, checked against pandas")
code("""
load_env('hive/setup_hive.sh')
!cd /opt/hive-metastore && hive --hivevar start=$START --hivevar end=$END -f $OLDPWD/hive/project.hql 2>/dev/null
!hdfs dfs -getmerge /project/hive/warehouse/project.db/integrated_district data/processed/integrated_district_hive.csv
!python3 scripts/build_climate_baseline.py --start $START --end $END
!python3 scripts/join_climate.py --check-hive data/processed/integrated_district_hive.csv
!cat results/hive_check.json
""")

md("## 7. Spark k-means climate regimes + Table 3")
code("""
load_env('spark/setup_spark.sh')
!python3 spark/kmeans_regimes.py --input hdfs://localhost:9000/project/processed/analysis_table.csv --output hdfs://localhost:9000/project/processed/climate_regimes 2>&1 | grep -E "k=|chosen"
import pandas as pd, json
km = json.load(open('results/kmeans_regimes.json'))
t = pd.read_csv('results/scalability_timings.csv')
display(pd.DataFrame([
    ('MapReduce', f"{int(t.map_input_records.max()):,} district-day rows", f"{int(t.reduce_output_records.max()):,} district-years",
     f"{t[t.years == t.years.max()].runtime_s.mean():.0f} s"),
    ('Hive baseline + join', 'district-years + development table', '469-district table', 'checked against pandas'),
    ('Spark k-means', '8 hazard variables', f"{km['k']} climate regimes", f"silhouette {km['silhouette']:.3f}"),
], columns=['Job', 'Input', 'Output', 'Result']))
""")

md("""
## 8. Proposed model: B-ACPN (Boosted Attentive Climate Push Network)

1. **Tree stage:** a random forest (500 trees) on all 13 features gives a first risk score (out-of-fold for the
   training districts, so the deep stage never sees a score the forest made on its own training rows).
2. **Deep stage:** `logit = tree score + development term (linear, 5 features) + climate push × development gate`,
   where the climate push is attention → MLP → residual block on the 8 hazard features.
3. **Ensemble:** 5 deep stages with different seeds, probabilities averaged.

MC Dropout gives the uncertainty of each district's push; split conformal prediction gives 90% prediction sets.
The vulnerability index is only the target, never an input.
""")
code("""
import sys; sys.path.insert(0, '.')
from models.bacpn import DeepStage
net = DeepStage()
print('Tree stage: RandomForestClassifier(500 trees, min_samples_leaf=2), all 13 features')
print('Deep stage (one of 5 in the ensemble):'); print(net)
print('trainable parameters per deep stage:', sum(p.numel() for p in net.parameters() if p.requires_grad))
# True = retrain every model here (about 2-3 hours on 2 vCPU: B-ACPN, its ablation and the Spark baselines);
# False = show the committed results in results/ (produced by the same scripts)
RERUN_MODELS = False
if RERUN_MODELS:
    !python3 scripts/train_bacpn.py
print(open('results/table4_model_card.json').read())
""")

md("## 9. Comparison models: Spark MLlib (LR, RF, GBT, Linear SVM) and a standard MLP")
code("""
if RERUN_MODELS:
    !python3 spark/train_baselines.py 2>&1 | grep -E "folds done|districts read"
    !python3 scripts/train_mlp.py
    !python3 scripts/train_bacpn.py --raw        # effect of preprocessing on the models (raw vs preprocessed features)
    !python3 scripts/train_bacpn.py --ablation
""")

md("## 10. Tables 5 and 6")
code("""
!python3 scripts/evaluate.py > /dev/null
t5 = pd.read_csv('results/table5_comparison.csv')
print('TABLE 5 - comparison (mean over 25 folds; LOSO = leave-one-state-out)')
display(t5[['model', 'tool', 'roc_auc', 'roc_auc_sd', 'bal_acc', 'accuracy', 'f1', 'loso_auc', 'train_time']].round(3))
print('Detailed metrics'); display(pd.read_csv('results/classifier_metrics_full.csv').round(3))
print('Confusion matrices (summed over 25 folds)'); display(pd.read_csv('results/confusion_matrices.csv'))
print('95% CI and corrected paired t-test vs B-ACPN'); display(pd.read_csv('results/significance.csv').round(4))
print('TABLE 6a - ablation'); display(pd.read_csv('results/table6_ablation.csv').round(3))
ba = pd.read_csv('results/preprocessing_before_after.csv')
print('Preprocessing before vs after (raw vs preprocessed features): accuracy, F1, RMSE (lower is better), ROC-AUC')
display(ba[['model', 'acc_before', 'acc_after', 'f1_before', 'f1_after', 'rmse_before', 'rmse_after', 'auc_before', 'auc_after']].round(3))
print('TABLE 6b - climate push'); print(open('results/table6_climate_push.json').read()[:3000])
""")

md("## 11. Visualisations (only after every table)")
code("""
!python3 scripts/make_figures.py > /dev/null
from IPython.display import Image, display
def show(*names):
    for f in names:
        print(f); display(Image(f'figures/{f}.png'))
""")
md("### 11a. Preprocessing figures")
code("show('prep_quality_before_after', 'prep_feature_scaling', 'prep_retention', 'prep_feature_signal', 'prep_correlation')")
md("### 11b. HDFS storage")
code("show('hdfs_storage')")
md("### 11c. MapReduce and Spark k-means")
code("show('mapreduce_scalability', 'kmeans_silhouette')")
md("### 11d. Models")
code("show('roc_curves', 'comparison_bars', 'fold_boxplots', 'confusion_matrices', 'calibration', 'loso_by_state', "
     "'bacpn_training_curve', 'prep_model_gain', 'ablation', 'attention_importance', 'climate_push_by_quadrant', "
     "'climate_push_map')")

md("## 12. Slides and appendix")
code("""
!python3 scripts/build_deck.py
""")
code("""
# appendix: show any source file during the review
!cat mapreduce/mapper.py mapreduce/reducer.py
!cat hive/project.hql
!cat models/bacpn.py
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python"}, "colab": {"provenance": []}})
nbf.write(nb, "notebooks/Review2_Colab.ipynb")
print("written -> notebooks/Review2_Colab.ipynb")
