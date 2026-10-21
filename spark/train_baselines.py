"""BD5: Spark MLlib baselines M1-M4 on the identical folds used for A-CPN.

  M1 LogisticRegression   M2 RandomForestClassifier (200 trees)
  M3 GBTClassifier (100 iterations, depth 3)   M4 LinearSVC (AUC from rawPrediction)

Each fold: VectorAssembler -> StandardScaler fitted on the training rows only (skipped for the
'raw' setting used in the preprocessing before/after experiment) -> model -> predictions.

Usage:
  python spark/train_baselines.py                                   # local file, scaled + raw
  python spark/train_baselines.py --input hdfs://localhost:9000/project/processed/analysis_table.csv
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.common import DATA, FEATURES, PRED, get_folds, load_table  # noqa: E402

from pyspark.ml import Pipeline  # noqa: E402
from pyspark.ml.classification import GBTClassifier, LinearSVC, LogisticRegression, RandomForestClassifier  # noqa: E402
from pyspark.ml.feature import StandardScaler, VectorAssembler  # noqa: E402
from pyspark.ml.functions import vector_to_array  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

MODELS = {
    "Logistic Regression": lambda: LogisticRegression(maxIter=200, regParam=0.01),
    "Random Forest": lambda: RandomForestClassifier(numTrees=200, seed=42),
    "Gradient-Boosted Trees": lambda: GBTClassifier(maxIter=100, maxDepth=3, seed=42),
    "Linear SVM": lambda: LinearSVC(maxIter=200, regParam=0.01),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DATA))
    ap.add_argument("--settings", default="scaled,raw")
    ap.add_argument("--master", default="local[*]")
    a = ap.parse_args()

    df_pd, _, y = load_table()
    folds = get_folds(df_pd, y)
    spark = (SparkSession.builder.master(a.master).appName("project-mllib-baselines")
             .config("spark.ui.enabled", "false").config("spark.sql.shuffle.partitions", "4").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    sdf = (spark.read.csv(a.input, header=True, inferSchema=True)
           .withColumn("label", (F.col("vul_group") == "High").cast("double"))
           .select("censuscode", "label", *FEATURES).dropna().cache())
    print(f"Spark {spark.version}: {sdf.count()} districts read from {a.input}")
    PRED.mkdir(parents=True, exist_ok=True)

    for setting in a.settings.split(","):
        jobs = [("cv", f) for f in folds["cv"]] + ([("loso", f) for f in folds["loso"]] if setting == "scaled" else [])
        for name, make in MODELS.items():
            rows = []
            for kind, f in jobs:
                tr = sdf.filter(F.col("censuscode").isin(f["train"]))
                te = sdf.filter(F.col("censuscode").isin(f["test"]))
                stages = [VectorAssembler(inputCols=FEATURES, outputCol="raw_features")]
                if setting == "scaled":
                    stages.append(StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True))
                    feat = "features"
                else:
                    feat = "raw_features"
                model = make().setFeaturesCol(feat).setLabelCol("label")
                t0 = time.perf_counter()
                fitted = Pipeline(stages=stages + [model]).fit(tr)
                t1 = time.perf_counter()
                out = fitted.transform(te).withColumn("raw", vector_to_array("rawPrediction"))
                if name != "Linear SVM":
                    out = out.withColumn("prob", vector_to_array("probability")[1])
                else:
                    out = out.withColumn("prob", F.lit(None).cast("double"))
                res = out.select("censuscode", "label", F.col("raw")[1].alias("score"), "prob", "prediction").toPandas()
                t2 = time.perf_counter()
                res["fold"], res["kind"] = f["fold"], kind
                res["train_time"], res["predict_time"] = t1 - t0, (t2 - t1) / max(len(res), 1)
                rows.append(res)
            pd.concat(rows).to_csv(PRED / f"{name}__{setting}.csv", index=False)
            print(f"  {setting:6s} {name}: {len(jobs)} folds done")
    spark.stop()


if __name__ == "__main__":
    main()
