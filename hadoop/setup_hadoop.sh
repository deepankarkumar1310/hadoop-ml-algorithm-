#!/usr/bin/env bash
# Single-node pseudo-distributed Hadoop 3.3.6 (HDFS + YARN), replication 1.
#
# Works in Google Colab (runs as root, no SSH server): the daemons are started
# one by one with `hdfs --daemon` / `yarn --daemon` instead of start-dfs.sh,
# which needs passwordless SSH.
#
#   source hadoop/setup_hadoop.sh          # installs if needed, starts daemons, exports env
#
# Re-running is safe: an existing install is reused and the NameNode is
# formatted only once (its metadata lives under $HADOOP_DATA).
set -euo pipefail

HADOOP_VERSION="${HADOOP_VERSION:-3.3.6}"
HADOOP_PREFIX_DIR="${HADOOP_PREFIX_DIR:-/opt}"
export HADOOP_HOME="${HADOOP_PREFIX_DIR}/hadoop-${HADOOP_VERSION}"
HADOOP_DATA="${HADOOP_DATA:-/opt/hadoop-data}"
MIRRORS=("https://dlcdn.apache.org/hadoop/common" "https://archive.apache.org/dist/hadoop/common")

# ---- Java 8 (Hadoop 3.3 runs on 8 or 11; Hive 3.1 runs only on 8) ---------
if [ ! -d /usr/lib/jvm/java-8-openjdk-amd64 ]; then
  apt-get update -qq && apt-get install -y -qq openjdk-8-jdk-headless > /dev/null
fi
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64

# ---- Hadoop binaries ------------------------------------------------------
if [ ! -x "$HADOOP_HOME/bin/hadoop" ]; then
  tgz="hadoop-${HADOOP_VERSION}.tar.gz"
  for m in "${MIRRORS[@]}"; do
    if curl -fsSL -o "/tmp/$tgz" "$m/hadoop-${HADOOP_VERSION}/$tgz"; then break; fi
  done
  tar -xzf "/tmp/$tgz" -C "$HADOOP_PREFIX_DIR"
  rm -f "/tmp/$tgz"
fi

export HADOOP_CONF_DIR="$HADOOP_HOME/etc/hadoop"
export PATH="$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$JAVA_HOME/bin:$PATH"
export HDFS_NAMENODE_USER=root HDFS_DATANODE_USER=root HDFS_SECONDARYNAMENODE_USER=root
export YARN_RESOURCEMANAGER_USER=root YARN_NODEMANAGER_USER=root
export HADOOP_STREAMING_JAR="$HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-${HADOOP_VERSION}.jar"

# ---- configuration --------------------------------------------------------
# YARN gets most of the machine's memory; a container is 1 GB, so the node runs
# about (memory / 1 GB) map or reduce tasks at once, capped by vcores.
MEM_MB=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1024 ))
YARN_MB=$(( MEM_MB * 3 / 4 ))
VCORES=$(nproc)

sed -i '/^export JAVA_HOME=/d' "$HADOOP_CONF_DIR/hadoop-env.sh"
echo "export JAVA_HOME=${JAVA_HOME}" >> "$HADOOP_CONF_DIR/hadoop-env.sh"

cat > "$HADOOP_CONF_DIR/core-site.xml" <<EOF
<configuration>
  <property><name>fs.defaultFS</name><value>hdfs://localhost:9000</value></property>
  <property><name>hadoop.tmp.dir</name><value>${HADOOP_DATA}/tmp</value></property>
</configuration>
EOF

cat > "$HADOOP_CONF_DIR/hdfs-site.xml" <<EOF
<configuration>
  <property><name>dfs.replication</name><value>1</value></property>
  <property><name>dfs.blocksize</name><value>134217728</value></property>
  <property><name>dfs.namenode.name.dir</name><value>file://${HADOOP_DATA}/namenode</value></property>
  <property><name>dfs.datanode.data.dir</name><value>file://${HADOOP_DATA}/datanode</value></property>
  <property><name>dfs.permissions.enabled</name><value>false</value></property>
</configuration>
EOF

cat > "$HADOOP_CONF_DIR/mapred-site.xml" <<EOF
<configuration>
  <property><name>mapreduce.framework.name</name><value>yarn</value></property>
  <property><name>mapreduce.application.classpath</name>
    <value>\$HADOOP_MAPRED_HOME/share/hadoop/mapreduce/*:\$HADOOP_MAPRED_HOME/share/hadoop/mapreduce/lib/*</value></property>
  <property><name>yarn.app.mapreduce.am.env</name><value>HADOOP_MAPRED_HOME=${HADOOP_HOME}</value></property>
  <property><name>mapreduce.map.env</name><value>HADOOP_MAPRED_HOME=${HADOOP_HOME}</value></property>
  <property><name>mapreduce.reduce.env</name><value>HADOOP_MAPRED_HOME=${HADOOP_HOME}</value></property>
  <property><name>yarn.app.mapreduce.am.resource.mb</name><value>1024</value></property>
  <property><name>mapreduce.map.memory.mb</name><value>1024</value></property>
  <property><name>mapreduce.reduce.memory.mb</name><value>1024</value></property>
</configuration>
EOF

cat > "$HADOOP_CONF_DIR/yarn-site.xml" <<EOF
<configuration>
  <property><name>yarn.nodemanager.aux-services</name><value>mapreduce_shuffle</value></property>
  <property><name>yarn.nodemanager.env-whitelist</name>
    <value>JAVA_HOME,HADOOP_COMMON_HOME,HADOOP_HDFS_HOME,HADOOP_CONF_DIR,CLASSPATH_PREPEND_DISTCACHE,HADOOP_YARN_HOME,HADOOP_HOME,PATH,LANG,TZ,HADOOP_MAPRED_HOME</value></property>
  <property><name>yarn.nodemanager.resource.memory-mb</name><value>${YARN_MB}</value></property>
  <property><name>yarn.nodemanager.resource.cpu-vcores</name><value>${VCORES}</value></property>
  <property><name>yarn.scheduler.maximum-allocation-mb</name><value>${YARN_MB}</value></property>
  <property><name>yarn.nodemanager.vmem-check-enabled</name><value>false</value></property>
  <property><name>yarn.nodemanager.disk-health-checker.max-disk-utilization-per-disk-percentage</name><value>99</value></property>
</configuration>
EOF

# ---- format once, then start the four daemons ------------------------------
if [ ! -d "${HADOOP_DATA}/namenode/current" ]; then
  mkdir -p "$HADOOP_DATA"
  hdfs namenode -format -nonInteractive -force > /dev/null 2>&1
fi

start() {  # start <hdfs|yarn> <daemon> unless already running
  if ! jps 2>/dev/null | awk '{print $2}' | grep -qx "$3"; then
    "$1" --daemon start "$2"
  fi
}
start hdfs namenode NameNode
start hdfs datanode DataNode
start yarn resourcemanager ResourceManager
start yarn nodemanager NodeManager

# wait for HDFS to leave safe mode and for the NodeManager to register
hdfs dfsadmin -safemode wait > /dev/null
for _ in $(seq 1 30); do
  yarn node -list 2>/dev/null | grep -q RUNNING && break
  sleep 2
done

echo "Hadoop ${HADOOP_VERSION} running (single node, replication 1, ${VCORES} vcores, YARN ${YARN_MB} MB)"
jps | grep -v Jps
set +euo pipefail
