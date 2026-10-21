"""Stage 7: IMD grids + precomputed area weights -> district-day CSV (MapReduce input).

For every day and district
    value(d, day) = sum_c w(c, d) * grid[day, c]  /  sum_c w(c, d) over non-missing c
i.e. the area weights of data/processed/grid_district_weights_{rain,temp}.csv,
renormalised over the cells that have data that day. IMD missing values
(rain -999, temperature 99.9) are NaN, never zero. A district with no valid
cell on a day gets an empty field.

File layout read: <imd_dir>/{rain,tmax,tmin}/<year>.grd, little-endian float32
stored as (days, nlat, nlon), latitude ascending from the south-west corner
(the same layout imdlib reads). The weight tables index cells by (ilat, ilon)
in this same orientation. The orientation is verified after the build with the
Chennai north-east-monsoon check, and the script fails if it does not pass.

Output: one file per year, no header
    <out_dir>/district_day_<year>.csv     censuscode,date,rain_mm,tmax_c,tmin_c
plus results/district_day_manifest.csv (rows, bytes, missing counts per year).

Usage
    python3 scripts/build_district_day.py --imd-dir /content/drive/MyDrive/climate_project/imd \
        --out-dir data/staging/district_day --start 1991 --end 2020
"""
import argparse
import calendar
import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse

WEIGHTS = {"rain": "data/processed/grid_district_weights_rain.csv",
           "temp": "data/processed/grid_district_weights_temp.csv"}
GRID = {"rain": (129, 135), "temp": (31, 31)}            # (nlat, nlon)
MISSING = {"rain": -999.0, "tmax": 99.9, "tmin": 99.9}

CHENNAI, JAISALMER = 603, 114


class Reducer:
    """Sparse cell -> district averaging operator for one grid."""

    def __init__(self, weights_csv, nlat, nlon, codes):
        w = pd.read_csv(weights_csv)
        self.nlon = nlon
        flat = (w.ilat * nlon + w.ilon).to_numpy()
        self.cells, cell_pos = np.unique(flat, return_inverse=True)
        col = pd.Index(codes).get_indexer(w.censuscode)
        assert (col >= 0).all(), "weight table has a district not in the code list"
        self.W = sparse.csr_matrix((w.weight.to_numpy(), (cell_pos, col)),
                                   shape=(len(self.cells), len(codes)))

    def apply(self, grid_days):
        """grid_days: (days, nlat*nlon) float array with NaN for missing."""
        vals = grid_days[:, self.cells]
        valid = ~np.isnan(vals)
        num = np.where(valid, vals, 0.0) @ self.W
        den = valid.astype(float) @ self.W
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
        return np.asarray(out)


def read_grd(path, var, year, nlat, nlon):
    days = 366 if calendar.isleap(year) else 365
    a = np.fromfile(path, dtype="<f4")
    if a.size != days * nlat * nlon:
        raise ValueError(f"{path}: {a.size} values, expected {days * nlat * nlon}")
    a = a.reshape(days, nlat * nlon).astype(np.float64)
    a[np.isclose(a, MISSING[var], atol=1e-3)] = np.nan
    return a


def build_year(year, imd_dir, codes, red):
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    cols = {}
    for var, grid in (("rain", "rain"), ("tmax", "temp"), ("tmin", "temp")):
        p = os.path.join(imd_dir, var, f"{year}.grd")
        if os.path.exists(p):
            cols[var] = red[grid].apply(read_grd(p, var, year, *GRID[grid]))
        else:
            cols[var] = None
    if cols["rain"] is None:
        return None                                  # rainfall defines the record
    n_d, n_c = len(days), len(codes)
    df = pd.DataFrame({
        "censuscode": np.tile(codes, n_d),
        "date": np.repeat(days.strftime("%Y-%m-%d"), n_c),
        "rain_mm": cols["rain"].ravel(),
        "tmax_c": cols["tmax"].ravel() if cols["tmax"] is not None else np.nan,
        "tmin_c": cols["tmin"].ravel() if cols["tmin"] is not None else np.nan,
    })
    return df


def check_orientation(out_dir):
    """Chennai's rain must peak in Oct-Dec (north-east monsoon); Jaisalmer's in
    Jun-Sep, and Jaisalmer must be hotter in May than in January. A flipped
    latitude axis puts Chennai's cells in the western Himalaya and fails this."""
    files = sorted(f for f in os.listdir(out_dir) if f.endswith(".csv"))
    parts = []
    for f in files:
        d = pd.read_csv(os.path.join(out_dir, f), header=None,
                        names=["censuscode", "date", "rain_mm", "tmax_c", "tmin_c"])
        parts.append(d[d.censuscode.isin([CHENNAI, JAISALMER])])
    d = pd.concat(parts)
    d["month"] = d.date.str[5:7].astype(int)
    years = d.date.str[:4].nunique()
    m = d.groupby(["censuscode", "month"]).agg(rain=("rain_mm", "sum"), tmax=("tmax_c", "mean"))
    m["rain"] /= years

    def share(code, months):
        r = m.loc[code, "rain"]
        return r[r.index.isin(months)].sum() / r.sum()

    ond, jjas = [10, 11, 12], [6, 7, 8, 9]
    print("\norientation check (mean monthly rain, mm):")
    print(m.rain.unstack(0).round(1).rename(columns={CHENNAI: "Chennai", JAISALMER: "Jaisalmer"}).to_string())
    ok = True
    c_ond, c_jjas = share(CHENNAI, ond), share(CHENNAI, jjas)
    j_jjas = share(JAISALMER, jjas)
    print(f"  Chennai   Oct-Dec share {c_ond:.2f}, Jun-Sep share {c_jjas:.2f}  (need Oct-Dec > Jun-Sep)")
    print(f"  Jaisalmer Jun-Sep share {j_jjas:.2f}  (need > 0.6)")
    ok &= c_ond > c_jjas and j_jjas > 0.6
    if m.tmax.notna().any():
        jan, may = m.loc[(JAISALMER, 1), "tmax"], m.loc[(JAISALMER, 5), "tmax"]
        print(f"  Jaisalmer mean Tmax Jan {jan:.1f} C, May {may:.1f} C  (need May > Jan + 8)")
        ok &= may > jan + 8
    print("  ORIENTATION OK" if ok else "  ORIENTATION FAILED - check the ilat axis before using any output")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imd-dir", required=True)
    ap.add_argument("--out-dir", default="data/staging/district_day")
    ap.add_argument("--start", type=int, default=1991)
    ap.add_argument("--end", type=int, default=2020)
    ap.add_argument("--manifest", default="results/district_day_manifest.csv")
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    if not a.check_only:
        codes = np.array(sorted(pd.read_csv(WEIGHTS["rain"]).censuscode.unique()))
        red = {g: Reducer(WEIGHTS[g], *GRID[g], codes) for g in WEIGHTS}
        manifest = []
        for year in range(a.start, a.end + 1):
            df = build_year(year, a.imd_dir, codes, red)
            if df is None:
                print(f"  {year}: no rainfall file, skipped")
                continue
            out = os.path.join(a.out_dir, f"district_day_{year}.csv")
            df.to_csv(out, index=False, header=False, float_format="%.2f", na_rep="")
            manifest.append({"year": year, "rows": len(df), "bytes": os.path.getsize(out),
                             "rain_missing": int(df.rain_mm.isna().sum()),
                             "tmax_missing": int(df.tmax_c.isna().sum()),
                             "tmin_missing": int(df.tmin_c.isna().sum())})
            print(f"  {year}: {len(df):,} rows, {manifest[-1]['bytes'] / 1e6:.1f} MB, "
                  f"missing rain {manifest[-1]['rain_missing']}, tmax {manifest[-1]['tmax_missing']}",
                  flush=True)
        mf = pd.DataFrame(manifest)
        os.makedirs(os.path.dirname(a.manifest) or ".", exist_ok=True)
        mf.to_csv(a.manifest, index=False)
        print(f"\ntotal: {mf.rows.sum():,} rows, {mf.bytes.sum() / 1e6:,.1f} MB, {len(mf)} years")

    if not check_orientation(a.out_dir):
        sys.exit(1)


if __name__ == "__main__":
    main()
