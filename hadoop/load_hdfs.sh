#!/usr/bin/env bash
# Stage 8: load raw sources (unmodified) and the district-day staging table into HDFS.
#
#   bash hadoop/load_hdfs.sh <imd_dir> [staging_dir]
#
# HDFS zones
#   /project/raw/{ndap,census,boundaries}      source files exactly as obtained
#   /project/raw/imd/{rain,tmax,tmin}          IMD binary .grd files, one per year
#   /project/staging/district_day              derived district-day CSV (MapReduce input)
#   /project/processed/...                     MapReduce and Hive outputs
# Prints the measured size of every zone at the end (hdfs dfs -du).
set -euo pipefail
IMD_DIR="${1:?usage: load_hdfs.sh <imd_dir> [staging_dir]}"
STAGING="${2:-data/staging/district_day}"
ROOT=/project

hdfs dfs -mkdir -p $ROOT/raw/ndap $ROOT/raw/census $ROOT/raw/boundaries \
  $ROOT/raw/imd/rain $ROOT/raw/imd/tmax $ROOT/raw/imd/tmin \
  $ROOT/staging/district_day $ROOT/processed

hdfs dfs -put -f data/raw/ndap_climate_vulnerability_2021.csv $ROOT/raw/ndap/
hdfs dfs -put -f data/raw/census/pca-full.csv $ROOT/raw/census/
hdfs dfs -put -f data/raw/boundaries/2011_Dist.* $ROOT/raw/boundaries/
for v in rain tmax tmin; do
  if compgen -G "$IMD_DIR/$v/*.grd" > /dev/null; then
    hdfs dfs -put -f "$IMD_DIR"/$v/*.grd $ROOT/raw/imd/$v/
  fi
done
hdfs dfs -put -f "$STAGING"/district_day_*.csv $ROOT/staging/district_day/

# socio-economic tables produced by scripts/ 1-4, one directory each for Hive
for t in census_2011:census2011_district_variables ndap_crosswalk:ndap_census_district_crosswalk \
         socioeconomic:integrated_district_table; do
  hdfs dfs -mkdir -p $ROOT/processed/${t%%:*}
  hdfs dfs -put -f data/processed/${t#*:}.csv $ROOT/processed/${t%%:*}/
done

echo "HDFS usage by zone (measured):"
hdfs dfs -du -h $ROOT/raw $ROOT/raw/imd $ROOT/staging
hdfs dfs -du -s -h $ROOT
