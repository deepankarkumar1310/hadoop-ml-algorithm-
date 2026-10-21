"""Preprocessing error metrics: each step checked against a reference (before vs after).

Usage: python scripts/preprocessing_metrics.py
"""
import json, sys
from datetime import date

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
O = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/preprocessing_metrics.json"
m = {}
# Census validation
pca = pd.read_csv(f"{O}/data/raw/census/pca-full.csv", low_memory=False)
t = pca[pca.TRU == "Total"]
pop = int(t.TOT_P.sum()); lit = t.P_LIT.sum() / (t.TOT_P.sum() - t.P_06.sum()) * 100
m["census"] = {"districts": int(len(t)), "population": pop, "population_official": 1210854977,
               "population_abs_error": abs(pop - 1210854977), "population_pct_error": abs(pop - 1210854977) / 1210854977 * 100,
               "literacy_pct": round(lit, 4), "literacy_official": 72.98, "literacy_abs_error_pp": round(abs(lit - 72.98), 4)}
# NDAP sparsity
nd = pd.read_csv(f"{O}/data/raw/ndap_climate_vulnerability_2021.csv")
miss = nd.isna().mean()
m["ndap"] = {"rows": nd.shape[0], "cols": nd.shape[1], "overall_missing_pct": round(miss.mean() * 100, 2),
             "cols_over_half_empty": int((miss > 0.5).sum())}
# Matching
cw = pd.read_csv(f"{O}/data/processed/ndap_census_district_crosswalk.csv")
mt = cw.match_type.fillna("unmatched").value_counts().to_dict()
matched = cw[cw.censuscode.notna()]
m["matching"] = {"ndap_districts": len(cw), "match_types": {k: int(v) for k, v in mt.items()},
                 "matched": int(len(matched)), "duplicate_censuscodes_after": int(matched.censuscode.duplicated().sum())}
# Grid weights
for g in ("rain", "temp"):
    w = pd.read_csv(f"{O}/data/processed/grid_district_weights_{g}.csv")
    s = w.groupby("censuscode").weight.sum()
    m[f"grid_{g}"] = {"pairs": len(w), "districts": int(s.size), "max_abs_sum_error": float((s - 1).abs().max()),
                      "rmse_sum_error": float(np.sqrt(((s - 1) ** 2).mean()))}
# District-day missingness (reference Colab run manifest)
man = pd.read_csv(f"{O}/results/district_day_manifest.csv")
r = man.rows.sum()
m["district_day"] = {"years": int(len(man)), "rows": int(r),
                     "rain_missing_pct": float(man.rain_missing.sum() / r * 100),
                     "tmax_missing_pct": float(man.tmax_missing.sum() / r * 100),
                     "tmin_missing_pct": float(man.tmin_missing.sum() / r * 100),
                     "source": "Colab run 2026-10-11 manifest"}
# Final analysis table
a = pd.read_csv(f"{O}/data/processed/analysis_table.csv")
DEV = ["literacy_rate", "female_literacy_rate", "agri_worker_share", "mgnrega_days", "yield_variability"]
HAZ = ["annual_rain_mm", "rain_variability", "rx5day_mm", "heavy_rain_days", "max_dry_spell", "mean_tmax", "hot_days", "tropical_nights"]
F = DEV + HAZ
rng = {"literacy_rate": (0, 100), "female_literacy_rate": (0, 100), "agri_worker_share": (0, 100)}
oor = sum(int(((a[c] < lo) | (a[c] > hi)).sum()) for c, (lo, hi) in rng.items())
oor += sum(int((a[c] < 0).sum()) for c in HAZ + ["mgnrega_days", "yield_variability"])
z = (a[F] - a[F].mean()) / a[F].std()
y = (a.vul_group == "High").astype(int)
m["final_table"] = {"rows": len(a), "states": int(a.state_name.nunique()), "feature_missing_cells": int(a[F].isna().sum().sum()),
                    "duplicate_censuscodes": int(a.censuscode.duplicated().sum()), "out_of_range_values": oor,
                    "outliers_abs_z_gt3": int((z.abs() > 3).sum().sum()), "class_high": int(y.sum()), "class_low": int((1 - y).sum()),
                    }
auc = {c: round(max(roc_auc_score(y, a[c]), 1 - roc_auc_score(y, a[c])), 3) for c in F}
m["feature_signal_auc"] = dict(sorted(auc.items(), key=lambda kv: -kv[1]))
snr = a.annual_rain_mm / a.rain_variability
m["rain_snr_median"] = round(float(snr.median()), 2)
m["computed"] = date.today().isoformat() + " from project data"
json.dump(m, open(OUT, "w"), indent=2); print(json.dumps(m, indent=1))
