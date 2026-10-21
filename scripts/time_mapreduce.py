"""Stage 13: measured scalability of the MapReduce job on this machine.

Runs the real Hadoop Streaming job on growing slices of the district-day table
already in HDFS (1, 10, 30 years and the full period by default), each slice
REPEATS times, and records what was measured:

    wall-clock runtime, HDFS input bytes (hdfs dfs -du), map input records and
    reduce output records (from the job's own counters), throughput

Single-node pseudo-distributed Hadoop: these numbers describe the job on one
machine. No multi-worker speedup is computed or implied.

Output: results/scalability_timings.csv (one row per run), results/environment.json
and figures fig08_10_scalability.png.

    python3 scripts/time_mapreduce.py --years 1 10 30 all --repeats 2
"""
import argparse
import json
import os
import platform
import re
import subprocess
import time

import pandas as pd

import viz_style as vs
from viz_style import plt

HDFS_DIR = "/project/staging/district_day"
OUT_ROOT = "/project/processed/timing"
COUNTER = re.compile(r"^\s*(Map input records|Reduce output records|Bytes Read)=(\d+)", re.M)


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def hdfs_years():
    r = sh(f"hdfs dfs -ls {HDFS_DIR}")
    return sorted(int(m) for m in re.findall(r"district_day_(\d{4})\.csv", r.stdout))


def hdfs_bytes(paths):
    r = sh("hdfs dfs -du -s " + " ".join(paths))
    return sum(int(line.split()[0]) for line in r.stdout.strip().splitlines())


def environment():
    mem = int(open("/proc/meminfo").read().split()[1]) // 1024
    hv = sh("hadoop version").stdout.splitlines()
    return {"hadoop": hv[0] if hv else "", "python": platform.python_version(),
            "cpu_count": os.cpu_count(), "mem_mb": mem, "platform": platform.platform(),
            "mode": "single-node pseudo-distributed, HDFS replication 1"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", default=["1", "10", "30", "all"])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--reducers", type=int, default=2)
    a = ap.parse_args()

    years = hdfs_years()
    assert years, f"no district_day_*.csv in {HDFS_DIR}"
    print(f"HDFS has {len(years)} years: {years[0]}-{years[-1]}")
    env = environment()
    env["run_date"] = time.strftime("%Y-%m-%d")
    os.makedirs("results", exist_ok=True)
    json.dump(env, open("results/environment.json", "w"), indent=2)

    rows = []
    for spec in a.years:
        n = len(years) if spec == "all" else int(spec)
        if n > len(years):
            print(f"  skip {spec}: only {len(years)} years in HDFS")
            continue
        sel = years[-n:]                         # most recent n years
        paths = [f"{HDFS_DIR}/district_day_{y}.csv" for y in sel]
        size = hdfs_bytes(paths)
        for rep in range(1, a.repeats + 1):
            out = f"{OUT_ROOT}/y{n}_r{rep}"
            t0 = time.time()
            r = sh(f"bash hadoop/run_mapreduce.sh {','.join(paths)} {out} {a.reducers}")
            secs = time.time() - t0
            if r.returncode != 0:
                print(r.stderr[-3000:])
                raise SystemExit(f"job failed for {n} years")
            c = dict(COUNTER.findall(r.stderr))
            rec = int(c.get("Map input records", 0))
            rows.append({"years": n, "first_year": sel[0], "last_year": sel[-1], "repeat": rep,
                         "hdfs_input_bytes": size, "map_input_records": rec,
                         "reduce_output_records": int(c.get("Reduce output records", 0)),
                         "runtime_s": round(secs, 2), "records_per_s": round(rec / secs),
                         "reducers": a.reducers})
            print(f"  {n:3d} years  run {rep}: {rec:,} records, {size / 1e6:,.1f} MB, {secs:.1f} s", flush=True)
            sh(f"hdfs dfs -rm -r -f -skipTrash {out}")

    df = pd.DataFrame(rows)
    df.to_csv("results/scalability_timings.csv", index=False)
    print("results -> results/scalability_timings.csv")
    plot(df)


def plot(df):
    vs.apply()
    g = df.groupby("years").agg(mb=("hdfs_input_bytes", lambda b: b.iloc[0] / 1e6),
                                rec=("map_input_records", "first"),
                                t=("runtime_s", "mean"), tmin=("runtime_s", "min"), tmax=("runtime_s", "max"),
                                thr=("records_per_s", "mean")).reset_index()
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    ax[0].errorbar(g.rec / 1e6, g.t, yerr=[g.t - g.tmin, g.tmax - g.t], color=vs.BLUE, marker="o",
                   markersize=8, capsize=4)
    for _, r in g.iterrows():
        ax[0].annotate(f"{int(r.years)} yr", (r.rec / 1e6, r.t), xytext=(6, -12), textcoords="offset points",
                       fontsize=8.5, color=vs.INK_2)
    ax[0].set_title("Fig 8: MapReduce runtime vs input (measured)")
    ax[0].set_xlabel("Input records (millions of district-days)")
    ax[0].set_ylabel("Wall-clock runtime (s)")
    ax[0].set_ylim(bottom=0)
    ax[1].plot(g.rec / 1e6, g.thr / 1e3, color=vs.BLUE, marker="o", markersize=8)
    ax[1].set_title("Fig 9: Throughput vs input (measured)")
    ax[1].set_xlabel("Input records (millions)")
    ax[1].set_ylabel("Thousand records / s")
    ax[1].set_ylim(bottom=0)
    zones = hdfs_zone_sizes()
    if zones:
        names, mb = list(zones), [v / 1e6 for v in zones.values()]
        ax[2].barh(names, mb, color=vs.SEQ[5], height=0.6)
        for i, v in enumerate(mb):
            ax[2].text(v, i, f" {v:,.0f} MB", va="center", fontsize=8.5, color=vs.INK_2)
        ax[2].set_xlim(0, max(mb) * 1.25)
    ax[2].set_title("Fig 10: HDFS storage by zone (hdfs dfs -du)")
    ax[2].set_xlabel("MB")
    ax[2].grid(axis="y", visible=False)
    fig.tight_layout()
    vs.save(fig, "figures/fig08_10_scalability.png")


def hdfs_zone_sizes():
    zones = {"raw/imd/rain": "/project/raw/imd/rain", "raw/imd/tmax": "/project/raw/imd/tmax",
             "raw/imd/tmin": "/project/raw/imd/tmin", "raw/census+ndap+boundaries":
             "/project/raw/census /project/raw/ndap /project/raw/boundaries",
             "staging/district_day": "/project/staging", "processed": "/project/processed"}
    out = {}
    for k, p in zones.items():
        r = sh(f"hdfs dfs -du -s {p}")
        if r.returncode == 0 and r.stdout.strip():
            out[k] = sum(int(line.split()[0]) for line in r.stdout.strip().splitlines())
    json.dump(out, open("results/hdfs_usage_bytes.json", "w"), indent=2)
    return out


if __name__ == "__main__":
    main()
