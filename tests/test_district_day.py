"""Plumbing tests for scripts/build_district_day.py on synthetic grids.

The synthetic .grd files are test fixtures, not IMD data, and never enter a
reported result. They check that (a) a cell's (ilat, ilon) in the weight table
addresses the right place in the binary layout, (b) missing cells are
renormalised away rather than read as zero, and (c) the orientation check
fails when the latitude axis is flipped.
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import build_district_day as bdd  # noqa: E402

YEAR = 2001                                   # 365 days
RAIN_GRID = (129, 135, 66.5, 6.5, 0.25)       # nlat, nlon, lon0, lat0, step
TEMP_GRID = (31, 31, 67.5, 7.5, 1.0)


def lat_lon(grid):
    nlat, nlon, lon0, lat0, step = grid
    lat = lat0 + step * np.arange(nlat)
    lon = lon0 + step * np.arange(nlon)
    return np.meshgrid(lat, lon, indexing="ij")


def write(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr.astype("<f4").tofile(path)


def synthetic_year(imd_dir, flip=False):
    """Rain: south of 16N peaks Oct-Dec, north of it peaks Jun-Sep.
    Tmax: 30 C in January rising to 42 C in May. Tmin: 20 C."""
    lat, _ = lat_lon(RAIN_GRID)
    if flip:
        lat = lat[::-1]
    doy = np.arange(365)
    month = pd.date_range(f"{YEAR}-01-01", periods=365).month.to_numpy()
    south = lat < 16
    rain = np.zeros((365,) + lat.shape)
    rain[np.isin(month, [10, 11, 12])] += np.where(south, 8.0, 0.2)
    rain[np.isin(month, [6, 7, 8, 9])] += np.where(south, 2.0, 6.0)
    write(f"{imd_dir}/rain/{YEAR}.grd", rain)
    tlat, _ = lat_lon(TEMP_GRID)
    tmax = (30 + 12 * np.clip((month - 1) / 4, 0, 1))[:, None, None] * np.ones_like(tlat)
    write(f"{imd_dir}/tmax/{YEAR}.grd", tmax)
    write(f"{imd_dir}/tmin/{YEAR}.grd", np.full((365,) + tlat.shape, 20.0))
    return doy


def run(imd_dir, out_dir):
    return subprocess.run(
        [sys.executable, "scripts/build_district_day.py", "--imd-dir", imd_dir,
         "--out-dir", out_dir, "--start", str(YEAR), "--end", str(YEAR),
         "--manifest", os.path.join(os.path.dirname(out_dir), "manifest.csv")],
        capture_output=True, text=True)


def test_cell_addressing_matches_district_location():
    """Fill each cell with its own latitude: a district's value must be the
    weighted mean latitude of its cells, which sits inside the district."""
    codes = np.array(sorted(pd.read_csv(bdd.WEIGHTS["rain"]).censuscode.unique()))
    red = bdd.Reducer(bdd.WEIGHTS["rain"], 129, 135, codes)
    lat, lon = lat_lon(RAIN_GRID)
    out_lat = red.apply(lat.reshape(1, -1))[0]
    out_lon = red.apply(lon.reshape(1, -1))[0]
    by = dict(zip(codes, zip(out_lat, out_lon)))
    chennai, leh = by[603], by[3]
    assert 12.9 < chennai[0] < 13.3 and 80.1 < chennai[1] < 80.4
    assert leh[0] > 32 and leh[1] > 75


def test_missing_cells_are_renormalised_not_zero():
    codes = np.array([603])
    w = pd.read_csv(bdd.WEIGHTS["rain"])
    red = bdd.Reducer(bdd.WEIGHTS["rain"], 129, 135, np.array(sorted(w.censuscode.unique())))
    grid = np.full((1, 129 * 135), 10.0)
    # Chennai uses cells (ilat 26, ilon 55) and (ilat 27, ilon 55); blank the big one
    grid[0, 26 * 135 + 55] = np.nan
    col = list(sorted(w.censuscode.unique())).index(603)
    assert red.apply(grid)[0, col] == 10.0           # not 10 * 0.045
    grid[0, 27 * 135 + 55] = np.nan
    assert np.isnan(red.apply(grid)[0, col])          # no valid cell -> missing


def test_end_to_end_and_orientation_check(tmp_path):
    imd, out = str(tmp_path / "imd"), str(tmp_path / "staging" / "district_day")
    synthetic_year(imd)
    r = run(imd, out)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ORIENTATION OK" in r.stdout
    d = pd.read_csv(f"{out}/district_day_{YEAR}.csv", header=None)
    assert len(d) == 640 * 365
    assert d[0].nunique() == 640


def test_orientation_check_catches_flipped_latitude(tmp_path):
    imd, out = str(tmp_path / "imd"), str(tmp_path / "staging" / "district_day")
    synthetic_year(imd, flip=True)
    r = run(imd, out)
    assert r.returncode == 1
    assert "ORIENTATION FAILED" in r.stdout
