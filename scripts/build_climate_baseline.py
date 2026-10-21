"""Stage 10: MapReduce district-year output -> one climate profile per district.

Averages every index over one fixed baseline window (default 1991-2020) that is
identical for every district, and takes the standard deviation of annual
rainfall across those years as rain_variability. The result is the 8-variable
climate exposure layer:

    annual_rain_mm, rain_variability, rx5day_mm, heavy_rain_days,
    max_dry_spell, mean_tmax, hot_days, tropical_nights

Years whose index is empty (incomplete daily record, or temperature before
1951) are left out of that index's average; n_years_rain and n_years_temp
record how many years each average rests on. Nothing is imputed.

The same aggregation is written in HiveQL in hive/project.hql; this script
is the pandas path used when Hive is not available.

Usage
    hdfs dfs -getmerge /project/processed/district_year data/processed/district_year_indices.csv
    python3 scripts/build_climate_baseline.py --start 1991 --end 2020
"""
import argparse

import pandas as pd

COLS = ["censuscode", "year", "n_days", "n_rain", "n_tmax", "n_tmin",
        "annual_rain_mm", "heavy_rain_days", "rx5day_mm", "max_dry_spell",
        "mean_tmax", "hot_days", "tropical_nights"]
RAIN = ["annual_rain_mm", "heavy_rain_days", "rx5day_mm", "max_dry_spell"]
TEMP = ["mean_tmax", "hot_days", "tropical_nights"]
CLIMATE_VARS = ["annual_rain_mm", "rain_variability", "rx5day_mm", "heavy_rain_days",
                "max_dry_spell", "mean_tmax", "hot_days", "tropical_nights"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/district_year_indices.csv")
    ap.add_argument("--output", default="data/processed/climate_baseline.csv")
    ap.add_argument("--start", type=int, default=1991)
    ap.add_argument("--end", type=int, default=2020)
    a = ap.parse_args()

    dy = pd.read_csv(a.input, header=None, names=COLS)
    print(f"district-year rows: {len(dy):,}, years {dy.year.min()}-{dy.year.max()}, "
          f"districts {dy.censuscode.nunique()}")
    dup = dy.duplicated(["censuscode", "year"]).sum()
    assert dup == 0, f"{dup} duplicate district-years in MapReduce output"

    w = dy[(dy.year >= a.start) & (dy.year <= a.end)]
    g = w.groupby("censuscode")
    out = g[RAIN + TEMP].mean()
    out["rain_variability"] = g.annual_rain_mm.std(ddof=1)
    out["n_years_rain"] = g.annual_rain_mm.count()
    out["n_years_temp"] = g.mean_tmax.count()
    out = out.reset_index()[["censuscode"] + CLIMATE_VARS + ["n_years_rain", "n_years_temp"]]
    out[CLIMATE_VARS] = out[CLIMATE_VARS].round(3)
    out.to_csv(a.output, index=False)

    n_years = a.end - a.start + 1
    print(f"baseline window {a.start}-{a.end} ({n_years} years)")
    print(f"districts: {len(out)}; with all {n_years} rain years: {(out.n_years_rain == n_years).sum()}; "
          f"with all temperature years: {(out.n_years_temp == n_years).sum()}")
    print(out[CLIMATE_VARS].describe().T.round(2).to_string())
    print(f"written -> {a.output}")


if __name__ == "__main__":
    main()
