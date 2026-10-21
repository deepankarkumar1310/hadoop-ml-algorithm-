"""Stage 11 (pandas path): join the climate baseline to the socio-economic table.

Mirrors the final join in hive/project.hql. Use this when Hive is not
available, or to cross-check Hive's output (--check-hive <file>).

    python3 scripts/join_climate.py
    python3 scripts/join_climate.py --check-hive data/processed/integrated_district_hive.csv
"""
import argparse
import json

import numpy as np
import pandas as pd

from build_climate_baseline import CLIMATE_VARS

SOCIO = "data/processed/integrated_district_table.csv"
BASE = "data/processed/climate_baseline.csv"
OUT = "data/processed/analysis_table.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-hive")
    a = ap.parse_args()

    s = pd.read_csv(SOCIO)
    b = pd.read_csv(BASE)
    df = s.merge(b, on="censuscode", how="inner")
    lost = sorted(set(s.censuscode) - set(b.censuscode))
    print(f"socio-economic districts {len(s)}, with climate profile {len(df)}, lost {len(lost)} {lost[:10]}")
    df.to_csv(OUT, index=False)
    print(f"written -> {OUT}")

    if a.check_hive:
        h = pd.read_csv(a.check_hive, header=None, na_values=["\\N"],   # Hive writes NULL as \N
                        names=list(s.columns) + CLIMATE_VARS + ["n_years_rain", "n_years_temp"])
        h = h.sort_values("censuscode").reset_index(drop=True)
        p = df.sort_values("censuscode").reset_index(drop=True)
        assert len(h) == len(p), f"row count differs: hive {len(h)}, pandas {len(p)}"
        diff = np.abs(h[CLIMATE_VARS].to_numpy() - p[CLIMATE_VARS].to_numpy())
        same_nulls = (h[CLIMATE_VARS].isna().to_numpy() == p[CLIMATE_VARS].isna().to_numpy()).all()
        print(f"Hive vs pandas: {len(h)} rows, max abs difference {np.nanmax(diff):.4f}, "
              f"NULLs in the same places: {same_nulls}")
        assert np.nanmax(diff) < 0.01 and same_nulls
        json.dump({"hive_rows": len(h), "pandas_rows": len(p), "max_abs_diff": float(np.nanmax(diff)),
                   "nulls_match": bool(same_nulls)}, open("results/hive_check.json", "w"), indent=2)


if __name__ == "__main__":
    main()
