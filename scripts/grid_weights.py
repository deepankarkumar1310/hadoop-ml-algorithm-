"""Precompute IMD grid-cell -> district area weights for the 2011 census districts.

Why weights and not a simple cell->district lookup: at 1 degree (temperature)
most districts are smaller than one grid cell, so nearest-centre assignment is
badly biased. Instead every cell that overlaps a district contributes to that
district in proportion to the share of the cell inside it.

Weight of cell c for district d
    w = (fraction of cell c covered by d) * cos(latitude of cell centre)
normalised so the weights of each district sum to 1. A district average for a
day is then sum(w * value) over the district's cells.

The fraction is computed in the spherical Lambert cylindrical equal-area
projection (x = lon, y = sin lat), where grid cells are rectangles and areas
are true areas. This reuses imdlib's own helpers, so the weights agree with
imdlib's region() rather than being a second, slightly different convention.

Inputs
    2011_Dist.shp               DataMeet 2011 district boundaries (641 polygons)
    imdlib GRIDS['r025']        rainfall grid, 0.25 deg, 135 x 129
    imdlib GRIDS['t100']        temperature grid, 1.0 deg, 31 x 31

Outputs
    grid_district_weights_rain.csv
    grid_district_weights_temp.csv
        censuscode, district_name, state_name, ilon, ilat, lon, lat, weight
"""
import numpy as np
import pandas as pd
import shapefile
import shapely
from shapely.geometry import shape

from imdlib.util import GRIDS
from imdlib.regions import _cell_fractions, _equal_area

SHP = "data/raw/boundaries/2011_Dist.shp"
OUT = {"rain": "data/processed/grid_district_weights_rain.csv",
       "temp": "data/processed/grid_district_weights_temp.csv"}
GRID_FOR = {"rain": GRIDS["r025"], "temp": GRIDS["t100"]}

# Cells covering less of a district than this are numerical slivers, not real overlap.
SLIVER = 1e-6


def load_districts():
    r = shapefile.Reader(SHP)
    fields = [f[0] for f in r.fields[1:]]
    out = []
    for sr in r.shapeRecords():
        rec = dict(zip(fields, sr.record))
        code = int(rec["censuscode"])
        if code == 0:          # the "Data Not Available" placeholder polygon in J&K
            continue
        geom = shape(sr.shape.__geo_interface__)
        if not geom.is_valid:
            geom = geom.buffer(0)
        out.append((code, rec["DISTRICT"], rec["ST_NM"], geom))
    return out


def weights_for(grid, districts):
    lon_edges, y_edges = grid.edges()
    lons, lats = grid.lon, grid.lat
    rows = []
    empty = []
    for code, name, state, geom in districts:
        ilon, ilat, frac = _cell_fractions(_equal_area(geom), lon_edges, y_edges)
        keep = frac > SLIVER
        ilon, ilat, frac = ilon[keep], ilat[keep], frac[keep]
        if len(frac) == 0:
            empty.append((code, name, state))
            continue
        w = frac * np.cos(np.deg2rad(lats[ilat]))
        w = w / w.sum()
        rows.append(pd.DataFrame({
            "censuscode": code, "district_name": name, "state_name": state,
            "ilon": ilon, "ilat": ilat,
            "lon": lons[ilon].round(4), "lat": lats[ilat].round(4),
            "weight": w.round(8),
        }))
    return pd.concat(rows, ignore_index=True), empty


def main():
    districts = load_districts()
    print(f"districts loaded: {len(districts)}")
    for label, grid in GRID_FOR.items():
        df, empty = weights_for(grid, districts)
        df.to_csv(OUT[label], index=False)
        n_dist = df.censuscode.nunique()
        per = df.groupby("censuscode").size()
        sums = df.groupby("censuscode").weight.sum()
        print(f"\n--- {label}  grid {grid.nlon}x{grid.nlat} @ {grid.step} deg ---")
        print(f"  districts with weights : {n_dist}/{len(districts)}")
        print(f"  districts with no cell : {len(empty)} {empty if empty else ''}")
        print(f"  rows (cell-district pairs): {len(df):,}")
        print(f"  cells used             : {df.groupby(['ilon','ilat']).ngroups:,} of {grid.nlon*grid.nlat:,}")
        print(f"  cells per district     : min {per.min()}, median {int(per.median())}, max {per.max()}")
        print(f"  weight sums (should be 1.0): min {sums.min():.6f}, max {sums.max():.6f}")
        shared = df.groupby(["ilon", "ilat"]).censuscode.nunique()
        print(f"  cells shared by >1 district: {(shared > 1).sum():,} "
              f"(max {shared.max()} districts in one cell)")
        print(f"  written -> {OUT[label]}")


if __name__ == "__main__":
    main()
