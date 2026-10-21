"""BD4: Spark distributed k-means on the 8 standardised hazard variables -> climate regimes.

k is chosen by the silhouette score over k = 2..8. Writes one regime label per district.

Usage:
  python spark/kmeans_regimes.py
  python spark/kmeans_regimes.py --input hdfs://localhost:9000/project/processed/analysis_table.csv \
                                 --output hdfs://localhost:9000/project/processed/climate_regimes
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.common import DATA, HAZ, RES, ROOT  # noqa: E402

from pyspark.ml.clustering import KMeans  # noqa: E402
from pyspark.ml.evaluation import ClusteringEvaluator  # noqa: E402
from pyspark.ml.feature import StandardScaler, VectorAssembler  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DATA))
    ap.add_argument("--output", default=None, help="optional HDFS/directory output for the regime table")
    ap.add_argument("--master", default="local[*]")
    a = ap.parse_args()
    spark = (SparkSession.builder.master(a.master).appName("project-kmeans-regimes")
             .config("spark.ui.enabled", "false").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    df = spark.read.csv(a.input, header=True, inferSchema=True).select("censuscode", "district_name", *HAZ).dropna()
    vec = VectorAssembler(inputCols=HAZ, outputCol="raw").transform(df)
    data = StandardScaler(inputCol="raw", outputCol="features", withMean=True, withStd=True).fit(vec).transform(vec).cache()
    ev = ClusteringEvaluator(featuresCol="features", metricName="silhouette")
    scores = {}
    for k in range(2, 9):
        scores[k] = ev.evaluate(KMeans(k=k, seed=42, featuresCol="features").fit(data).transform(data))
        print(f"k={k} silhouette={scores[k]:.3f}")
    best = max(scores, key=scores.get)
    out = KMeans(k=best, seed=42, featuresCol="features").fit(data).transform(data).withColumnRenamed("prediction", "regime")
    sizes = {int(r["regime"]): r["count"] for r in out.groupBy("regime").count().collect()}
    prof = out.groupBy("regime").agg(*[F.round(F.avg(h), 1).alias(h) for h in HAZ]).orderBy("regime").toPandas()
    pdf = out.select("censuscode", "district_name", "regime").toPandas()
    pdf.to_csv(ROOT / "data" / "processed" / "climate_regimes.csv", index=False)
    if a.output:
        out.select("censuscode", "regime").write.mode("overwrite").csv(a.output, header=True)
    res = {"k": best, "silhouette": scores[best], "silhouette_by_k": scores, "sizes": sizes,
           "profiles": prof.to_dict(orient="records"), "spark_version": spark.version}
    (RES / "kmeans_regimes.json").write_text(json.dumps(res, indent=2))
    print(f"chosen k={best} (silhouette {scores[best]:.3f}); sizes {sizes}")
    spark.stop()


if __name__ == "__main__":
    main()
