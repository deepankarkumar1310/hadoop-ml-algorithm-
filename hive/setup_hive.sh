#!/usr/bin/env bash
# Hive 3.1.3 with an embedded Derby metastore, on top of hadoop/setup_hadoop.sh.
#
#   source hadoop/setup_hadoop.sh && source hive/setup_hive.sh
#
# Time-box this step to about an hour in Colab. If it fails, the pandas path
# (scripts/build_climate_baseline.py + scripts/join_climate.py) produces the same
# tables, and the report should then describe Hive as planned work.
set -euo pipefail
HIVE_VERSION="${HIVE_VERSION:-3.1.3}"
HIVE_PREFIX_DIR="${HIVE_PREFIX_DIR:-${HADOOP_PREFIX_DIR:-/opt}}"
export HIVE_HOME="${HIVE_PREFIX_DIR}/apache-hive-${HIVE_VERSION}-bin"
METASTORE_DIR="${METASTORE_DIR:-/opt/hive-metastore}"

if [ ! -x "$HIVE_HOME/bin/hive" ]; then
  tgz="apache-hive-${HIVE_VERSION}-bin.tar.gz"
  for m in https://dlcdn.apache.org/hive https://archive.apache.org/dist/hive; do
    if curl -fsSL -o "/tmp/$tgz" "$m/hive-${HIVE_VERSION}/$tgz"; then break; fi
  done
  tar -xzf "/tmp/$tgz" -C "$HIVE_PREFIX_DIR"
  rm -f "/tmp/$tgz"
fi
export PATH="$HIVE_HOME/bin:$PATH"

# Hive 3.1.3 ships Guava 19, Hadoop 3.3 needs Guava 27: use Hadoop's copy.
if ls "$HIVE_HOME"/lib/guava-19*.jar > /dev/null 2>&1; then
  rm -f "$HIVE_HOME"/lib/guava-19*.jar
  cp "$HADOOP_HOME"/share/hadoop/common/lib/guava-27*.jar "$HIVE_HOME/lib/"
fi

cat > "$HIVE_HOME/conf/hive-site.xml" <<EOF
<configuration>
  <property><name>javax.jdo.option.ConnectionURL</name>
    <value>jdbc:derby:;databaseName=${METASTORE_DIR}/metastore_db;create=true</value></property>
  <property><name>hive.metastore.warehouse.dir</name><value>/project/hive/warehouse</value></property>
  <property><name>hive.metastore.schema.verification</name><value>false</value></property>
  <property><name>hive.exec.mode.local.auto</name><value>false</value></property>
  <property><name>hive.execution.engine</name><value>mr</value></property>
  <property><name>hive.cli.print.header</name><value>true</value></property>
</configuration>
EOF

hdfs dfs -mkdir -p /tmp /project/hive/warehouse
hdfs dfs -chmod g+w /tmp /project/hive/warehouse
if [ ! -d "${METASTORE_DIR}/metastore_db" ]; then
  mkdir -p "$METASTORE_DIR"
  (cd "$METASTORE_DIR" && schematool -dbType derby -initSchema > /dev/null 2>&1)
fi
echo "Hive ${HIVE_VERSION} ready (embedded Derby metastore at ${METASTORE_DIR})"
set +euo pipefail
