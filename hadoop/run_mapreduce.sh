#!/usr/bin/env bash
# Stage 9: Hadoop Streaming job, district-day -> district-year climate indices.
#
#   bash hadoop/run_mapreduce.sh <hdfs_input[,input2,...]> <hdfs_output> [n_reducers]
#
# The input may be a directory, a glob or a comma-separated list of files.
# Output lines are plain CSV (the key/value separator is set to a comma):
#   censuscode,year,n_days,n_rain,n_tmax,n_tmin,annual_rain_mm,heavy_rain_days,
#   rx5day_mm,max_dry_spell,mean_tmax,hot_days,tropical_nights
set -euo pipefail
IN="${1:?usage: run_mapreduce.sh <input> <output> [reducers]}"
OUT="${2:?usage: run_mapreduce.sh <input> <output> [reducers]}"
REDUCERS="${3:-2}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${HADOOP_STREAMING_JAR:=$(ls "$HADOOP_HOME"/share/hadoop/tools/lib/hadoop-streaming-*.jar)}"

hdfs dfs -rm -r -f -skipTrash "$OUT" > /dev/null 2>&1 || true

hadoop jar "$HADOOP_STREAMING_JAR" \
  -D mapreduce.job.name="project-district-year" \
  -D mapreduce.job.reduces="$REDUCERS" \
  -D mapreduce.output.textoutputformat.separator=, \
  -files "$HERE/mapreduce/mapper.py,$HERE/mapreduce/reducer.py" \
  -mapper "python3 mapper.py" \
  -reducer "python3 reducer.py" \
  -input "$IN" \
  -output "$OUT"
