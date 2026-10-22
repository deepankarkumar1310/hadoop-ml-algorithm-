# Big Data tools: where they are used, what they do, and why

All four tools run on **single-node, pseudo-distributed Hadoop 3.3.6** (HDFS replication 1) inside Google Colab,
with Java 8 for every component. The same scripts run unchanged on a real cluster, but this project makes
**no multi-machine speed-up claim**.

```
IMD / NDAP / Census / boundaries
      │
      ▼
   HDFS ───────── stores everything (raw → staging → processed)
      │
      ▼
MapReduce/YARN ── 7,013,120 district-day rows → 19,200 district-year rows
      │
      ▼
   Hive ───────── 30-year averages + join → 469-district model-ready table
      │
      ▼
  Spark ───────── k-means climate regimes + MLlib baseline models
      │
      ▼
 B-ACPN (RF + PyTorch) ─ proposed model, compared with the Spark baselines
```

## 1. HDFS (Hadoop Distributed File System)
| | |
|---|---|
| **Where** | `hadoop/setup_hadoop.sh`, `hadoop/load_hdfs.sh`; notebook sections 3–4 |
| **What it does** | Stores all data in three zones: `/project/raw` (IMD grids, NDAP, Census, boundaries; never modified), `/project/staging` (district-day CSVs), `/project/processed` (MapReduce, Hive, Spark outputs) |
| **Purpose** | One shared store that MapReduce, Hive and Spark all read and write |
| **Why chosen** | 30 years of daily grids (~212 million values, 1.08 GB in HDFS) are split into 128 MB blocks that jobs process in parallel; every other Hadoop tool reads HDFS directly |
| **Evidence** | `hdfs dfs -du` per zone → `results/hdfs_usage_bytes.json` (Table 2) |

## 2. MapReduce on YARN (Hadoop Streaming, Python)
| | |
|---|---|
| **Where** | `mapreduce/mapper.py`, `mapreduce/reducer.py`, `hadoop/run_mapreduce.sh`, `scripts/time_mapreduce.py`; section 5 |
| **What it does** | Map: each district-day row → key (censuscode, year). Reduce: sort the days by date, compute 7 yearly indices (annual rain, heavy-rain days, max 5-day rain, longest dry spell, mean Tmax, hot days, tropical nights). Incomplete years stay blank, never 0; bad rows go to a counter |
| **Purpose** | The heaviest computation: 7,013,120 daily rows → 19,200 district-year rows |
| **Why chosen** | The job splits naturally by (district, year); YARN schedules the map and reduce tasks; order-dependent indices (dry spells, 5-day rain) are computed after sorting in the reducer |
| **Evidence** | Records in/out, runtimes for 1/10/30 years → `results/scalability_timings.csv` (Table 3) |

## 3. Hive (SQL on HDFS)
| | |
|---|---|
| **Where** | `hive/setup_hive.sh`, `hive/project.hql`; section 6 |
| **What it does** | External tables over the HDFS files; 30-year averages and STDDEV_SAMP of annual rain → 8 hazard variables per district; join with the development table on `censuscode` |
| **Purpose** | Produces the 469-district model-ready table |
| **Why chosen** | Averages, grouping and joins are SQL tasks; Hive runs them on HDFS without moving the data out |
| **Evidence** | Hive vs pandas: 469 = 469 rows, max difference 0.0 → `results/hive_check.json` |

## 4. Apache Spark 3.5 (+ MLlib)
| | |
|---|---|
| **Where** | `spark/setup_spark.sh`, `spark/kmeans_regimes.py`, `spark/train_baselines.py`; sections 7 and 9 |
| **What it does** | (a) k-means on the 8 standardised hazard variables → climate regimes (k by silhouette, k = 2..8); (b) distributed training of LogisticRegression, RandomForest (200 trees), GBT (100 iterations, depth 3) and LinearSVC on the same folds as B-ACPN |
| **Purpose** | Find natural climate zones (to check that the B-ACPN climate push follows real climate patterns) and train the comparison models |
| **Why chosen** | In-memory execution suits iterative algorithms (k-means, model fitting); MLlib provides the standard models; Spark reads HDFS directly |
| **Evidence** | `results/kmeans_regimes.json`, `results/table5_comparison.csv` |

## 5. YARN (resource manager)
Allocates CPU and memory to the MapReduce tasks and runs them (`hadoop/setup_hadoop.sh`).

## What is not distributed, and why
- **Grid → district-day** (`scripts/build_district_day.py`) runs in Python/NumPy because it needs polygon geometry;
  its output is loaded into HDFS for MapReduce.
- **B-ACPN** is trained with scikit-learn (random-forest stage) and PyTorch (deep stage): with 469 districts, distributed training brings no benefit.
