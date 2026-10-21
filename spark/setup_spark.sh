#!/usr/bin/env bash
# Apache Spark 3.5 (pyspark) on top of hadoop/setup_hadoop.sh, reading hdfs://localhost:9000.
#
#   source spark/setup_spark.sh
#
# Spark runs in local mode on the same single node as HDFS/YARN; it reads and writes HDFS
# through fs.defaultFS. Java 8 (installed by setup_hadoop.sh) is used for Hadoop, Hive and Spark.
set -euo pipefail

SPARK_VERSION="${SPARK_VERSION:-3.5.3}"
export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-8-openjdk-amd64}"

if ! python3 -c "import pyspark, sys; sys.exit(0 if pyspark.__version__.startswith('3.5') else 1)" 2>/dev/null; then
  python3 -m pip install -q "pyspark==${SPARK_VERSION}"
fi
export PYSPARK_PYTHON=python3
export SPARK_LOCAL_IP=127.0.0.1

python3 - <<'PY'
from pyspark.sql import SparkSession
s = SparkSession.builder.master("local[*]").appName("health-check").config("spark.ui.enabled", "false").getOrCreate()
s.sparkContext.setLogLevel("ERROR")
print(f"Spark {s.version} OK, default parallelism {s.sparkContext.defaultParallelism}")
s.stop()
PY
