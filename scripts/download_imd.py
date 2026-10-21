"""Stage 6: download IMD gridded rainfall, Tmax and Tmin, one file per year.

Run in Google Colab (the IMD server is not reachable from every network).
Files are written to <imd_dir>/{rain,tmax,tmin}/<year>.grd. Put <imd_dir> on
Google Drive, because Colab wipes its local disk when the session ends.

The IMD server is intermittently down. Each year is an independent file, so a
failed year is retried on its own, and a year already on disk with the right
byte size is skipped. Every file is checked against its expected size
(days x nlat x nlon x 4 bytes); a short or corrupt file is deleted and retried.

Usage
    python3 scripts/download_imd.py --imd-dir /content/drive/MyDrive/climate_project/imd \
        --start 1991 --end 2020
    # optional larger HDFS volume: rainfall back to 1901 (analysis still uses 1991-2020)
    python3 scripts/download_imd.py --imd-dir ... --start 1901 --end 2024 --vars rain
"""
import argparse
import calendar
import os
import sys
import time

import imdlib as imd
from imdlib.util import ARCHIVE_GRIDS

FIRST_YEAR = {"rain": 1901, "tmax": 1951, "tmin": 1951}


def expected_bytes(var, year):
    nlat, nlon = ARCHIVE_GRIDS[var]
    days = 366 if calendar.isleap(year) else 365
    return days * nlat * nlon * 4


def path_for(imd_dir, var, year):
    return os.path.join(imd_dir, var, f"{year}.grd")


def normalise_name(imd_dir, var, year):
    """imdlib saves rainfall as <year>.grd but Tmax/Tmin as <year>.GRD. Rename to
    the lower-case name that the rest of the pipeline reads."""
    upper = os.path.join(imd_dir, var, f"{year}.GRD")
    lower = path_for(imd_dir, var, year)
    if os.path.exists(upper) and not os.path.exists(lower):
        os.rename(upper, lower)


def is_complete(imd_dir, var, year):
    normalise_name(imd_dir, var, year)
    p = path_for(imd_dir, var, year)
    return os.path.exists(p) and os.path.getsize(p) == expected_bytes(var, year)


def fetch(imd_dir, var, year, retries):
    for attempt in range(1, retries + 1):
        try:
            imd.get_data(var, year, year, fn_format="yearwise", file_dir=imd_dir)
        except Exception as e:                       # server errors, timeouts
            print(f"    {var} {year}: attempt {attempt} failed ({e})", flush=True)
        if is_complete(imd_dir, var, year):
            return True
        for p in (path_for(imd_dir, var, year), os.path.join(imd_dir, var, f"{year}.GRD")):
            if os.path.exists(p):
                os.remove(p)                         # never keep a partial file
        time.sleep(min(60, 5 * 2 ** (attempt - 1)))
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imd-dir", required=True)
    ap.add_argument("--start", type=int, default=1991)
    ap.add_argument("--end", type=int, default=2020)
    ap.add_argument("--vars", nargs="+", default=["rain", "tmax", "tmin"])
    ap.add_argument("--retries", type=int, default=5)
    a = ap.parse_args()

    failed = []
    for var in a.vars:
        os.makedirs(os.path.join(a.imd_dir, var), exist_ok=True)
        for year in range(max(a.start, FIRST_YEAR[var]), a.end + 1):
            if is_complete(a.imd_dir, var, year):
                continue
            print(f"  downloading {var} {year}", flush=True)
            if not fetch(a.imd_dir, var, year, a.retries):
                failed.append((var, year))

    print("\nsummary")
    for var in a.vars:
        years = range(max(a.start, FIRST_YEAR[var]), a.end + 1)
        ok = [y for y in years if is_complete(a.imd_dir, var, y)]
        size = sum(os.path.getsize(path_for(a.imd_dir, var, y)) for y in ok)
        print(f"  {var:5s}: {len(ok)}/{len(years)} years, {size / 1e6:,.1f} MB")
    if failed:
        print(f"\nFAILED (re-run the same command to retry only these): {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
