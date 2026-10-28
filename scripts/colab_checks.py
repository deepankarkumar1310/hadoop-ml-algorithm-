"""Two preprocessing error checks that need the raw IMD grids (run in Colab after MapReduce).

1. MapReduce vs pandas (Deepankar, notebook section 5): the 7 yearly indices recomputed with an independent vectorised pandas
   implementation from the same district-day staging files; RMSE, MSE, MAE, MAPE, R², NRMSE, MedAE, bias and max error
   per index.
2. Area-weighted vs nearest-cell (Sehjodh, notebook section 2): annual rainfall per district from our area-weighted aggregation
   compared with the naive method (take the single grid cell with the largest overlap).
   This is the size of the error the naive method would have introduced (same 8 metrics).

Writes results/preprocessing_colab_checks.json (read by the slide builder).

Usage: python3 scripts/colab_checks.py --check aggregation --imd-dir <imd> --year 2020
       python3 scripts/colab_checks.py --check mapreduce
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_district_day import GRID, read_grd  # noqa: E402

IDX = ["annual_rain_mm", "heavy_rain_days", "rx5day_mm", "max_dry_spell", "mean_tmax", "hot_days", "tropical_nights"]
MR_COLS = ["censuscode", "year", "n_days", "n_rain", "n_tmax", "n_tmin"] + IDX


def longest_run(mask):
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def pandas_indices(staging):
    out = []
    for f in sorted(glob.glob(os.path.join(staging, "district_day_*.csv"))):
        d = pd.read_csv(f, header=None, names=["censuscode", "date", "rain", "tmax", "tmin"]).sort_values(["censuscode", "date"])
        d["year"] = d.date.str[:4].astype(int)
        for (c, y), g in d.groupby(["censuscode", "year"]):
            r, tx, tn = g.rain.to_numpy(), g.tmax.to_numpy(), g.tmin.to_numpy()
            row = {"censuscode": c, "year": y}
            if not np.isnan(r).any():
                row.update(annual_rain_mm=r.sum(), heavy_rain_days=(r >= 64.5).sum(),
                           rx5day_mm=np.convolve(r, np.ones(5), "valid").max(), max_dry_spell=longest_run(r < 1.0))
            if not np.isnan(tx).any():
                row.update(mean_tmax=tx.mean(), hot_days=(tx >= 35.0).sum())
            if not np.isnan(tn).any():
                row.update(tropical_nights=(tn >= 25.0).sum())
            out.append(row)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", default="data/staging/district_day")
    ap.add_argument("--mr", default="data/processed/district_year_indices.csv")
    ap.add_argument("--imd-dir", default="data/imd")
    ap.add_argument("--check", choices=["mapreduce", "aggregation", "all"], default="all")
    ap.add_argument("--year", type=int, default=2020)
    a = ap.parse_args()
    out = Path("results/preprocessing_colab_checks.json")
    res = json.loads(out.read_text()) if out.exists() else {}

    if a.check in ("mapreduce", "all"):
        mapreduce_check(a, res)
    if a.check in ("aggregation", "all"):
        aggregation_check(a, res)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def reg_metrics(ref, pred):
    """The 8 preprocessing error metrics of `pred` against the reference `ref` (both 1-D, no NaN).

    MAPE skips reference values of 0; NRMSE = RMSE / (max - min of the reference); bias = mean(pred - ref)."""
    ref, pred = np.asarray(ref, dtype=float), np.asarray(pred, dtype=float)
    d = pred - ref
    mse = float((d ** 2).mean())
    nz = ref != 0
    span = float(ref.max() - ref.min())
    sst = float(((ref - ref.mean()) ** 2).sum())
    return {"n": int(len(d)), "rmse": float(np.sqrt(mse)), "mse": mse, "mae": float(np.abs(d).mean()),
            "mape_pct": float((np.abs(d[nz]) / np.abs(ref[nz])).mean() * 100) if nz.any() else float("nan"),
            "r2": float(1 - (d ** 2).sum() / sst) if sst > 0 else float("nan"),
            "nrmse": float(np.sqrt(mse) / span) if span > 0 else float("nan"),
            "medae": float(np.median(np.abs(d))), "bias": float(d.mean())}


def mapreduce_check(a, res):
    mr = pd.read_csv(a.mr, header=None, names=MR_COLS)
    pdx = pandas_indices(a.staging)
    m = mr.merge(pdx, on=["censuscode", "year"], suffixes=("_mr", "_pd"))
    per = {}
    for c in IDX:
        x, y = m[f"{c}_mr"], m[f"{c}_pd"]
        ok = x.notna() & y.notna()
        diff = (x[ok] - y[ok]).to_numpy()
        per[c] = {**reg_metrics(y[ok], x[ok]), "max_abs": float(np.abs(diff).max()),
                  "blank_mismatch": int((x.isna() != y.isna()).sum())}
    res["mapreduce_vs_pandas"] = {"rows_compared": len(m), "per_index": per,
                                  "rmse_max": max(v["rmse"] for v in per.values()),
                                  "mae_max": max(v["mae"] for v in per.values())}


def aggregation_check(a, res):
    w = pd.read_csv("data/processed/grid_district_weights_rain.csv")
    nlat, nlon = GRID["rain"]
    grid = read_grd(os.path.join(a.imd_dir, "rain", f"{a.year}.grd"), "rain", a.year, nlat, nlon)
    total = np.nansum(grid, axis=0)  # annual rain per cell
    w["cell"] = w.ilat * nlon + w.ilon
    w["cell_rain"] = total[w.cell]
    w["valid"] = ~np.isnan(grid).all(axis=0)[w.cell]
    ww = w[w.valid]
    area = ww.groupby("censuscode").apply(lambda g: (g.weight * g.cell_rain).sum() / g.weight.sum())
    near = ww.loc[ww.groupby("censuscode").weight.idxmax()].set_index("censuscode").cell_rain
    diff = (near - area).dropna()
    res["area_vs_nearest_cell"] = {"year": a.year, "districts": int(len(diff)),
                                   "rmse_mm": float(np.sqrt((diff ** 2).mean())), "mae_mm": float(diff.abs().mean()),
                                   "mean_abs_pct": float((diff.abs() / area[diff.index]).mean() * 100),
                                   # reference = our area-weighted value, prediction = naive nearest cell
                                   "metrics": reg_metrics(area[diff.index], near[diff.index])}


if __name__ == "__main__":
    main()
