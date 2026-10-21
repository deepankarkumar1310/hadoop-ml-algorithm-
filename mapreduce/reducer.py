#!/usr/bin/env python3
"""Hadoop Streaming reducer: one district-year of daily values -> climate indices.

Hadoop sorts by key but not by date within a key. A district-year has at most
366 rows, so the reducer buffers them, sorts by date and then computes:

    annual_rain_mm    sum of daily rain
    heavy_rain_days   days with rain >= 64.5 mm (IMD "heavy rain")
    rx5day_mm         maximum 5-day running rainfall sum
    max_dry_spell     longest run of consecutive days with rain < 1 mm
    mean_tmax         mean daily maximum temperature
    hot_days          days with Tmax >= 35 C
    tropical_nights   days with Tmin >= 25 C

Missing values: a rain index is written only if every day of the year has a
rain value, and a temperature index only if every day has that temperature
value. Otherwise the field is left empty. Nothing is filled or interpolated.
Rainfall starts in 1901 and temperature in 1951, so earlier years carry rain
indices and empty temperature fields.

Running windows (rx5day, dry spells) are computed within the calendar year and
do not cross into the next year.

Output (key TAB value; with the job option
mapreduce.output.textoutputformat.separator=, each line is plain CSV)
    censuscode,year <TAB> n_days,n_rain,n_tmax,n_tmin,annual_rain_mm,
        heavy_rain_days,rx5day_mm,max_dry_spell,mean_tmax,hot_days,tropical_nights
"""
import sys

HEAVY_RAIN_MM = 64.5
DRY_DAY_MM = 1.0
HOT_DAY_C = 35.0
TROPICAL_NIGHT_C = 25.0

FIELDS = ["n_days", "n_rain", "n_tmax", "n_tmin", "annual_rain_mm",
          "heavy_rain_days", "rx5day_mm", "max_dry_spell", "mean_tmax",
          "hot_days", "tropical_nights"]


def num(s):
    return float(s) if s != "" else None


def fmt(x, nd=2):
    if x is None:
        return ""
    return str(x) if isinstance(x, int) else f"{x:.{nd}f}"


def indices(rows):
    """rows: list of (date, rain, tmax, tmin) with None for missing."""
    rows.sort(key=lambda r: r[0])
    # a date that appears twice is an upstream error; keep the first, count it
    dedup, seen = [], set()
    for r in rows:
        if r[0] in seen:
            sys.stderr.write("reporter:counter:Project,duplicate_days,1\n")
            continue
        seen.add(r[0])
        dedup.append(r)
    rows = dedup

    n = len(rows)
    rain = [r[1] for r in rows]
    tmax = [r[2] for r in rows]
    tmin = [r[3] for r in rows]
    n_rain = sum(v is not None for v in rain)
    n_tmax = sum(v is not None for v in tmax)
    n_tmin = sum(v is not None for v in tmin)

    out = dict(n_days=n, n_rain=n_rain, n_tmax=n_tmax, n_tmin=n_tmin,
               annual_rain_mm=None, heavy_rain_days=None, rx5day_mm=None,
               max_dry_spell=None, mean_tmax=None, hot_days=None,
               tropical_nights=None)

    if n_rain == n and n > 0:
        out["annual_rain_mm"] = sum(rain)
        out["heavy_rain_days"] = sum(v >= HEAVY_RAIN_MM for v in rain)
        if n >= 5:
            window = sum(rain[:5])
            best = window
            for i in range(5, n):
                window += rain[i] - rain[i - 5]
                best = max(best, window)
            out["rx5day_mm"] = best
        run = longest = 0
        for v in rain:
            run = run + 1 if v < DRY_DAY_MM else 0
            longest = max(longest, run)
        out["max_dry_spell"] = longest

    if n_tmax == n and n > 0:
        out["mean_tmax"] = sum(tmax) / n
        out["hot_days"] = sum(v >= HOT_DAY_C for v in tmax)
    if n_tmin == n and n > 0:
        out["tropical_nights"] = sum(v >= TROPICAL_NIGHT_C for v in tmin)
    return out


def emit(key, rows):
    ix = indices(rows)
    print(key + "\t" + ",".join(fmt(ix[f]) for f in FIELDS))


def main():
    current, rows = None, []
    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line:
            continue
        key, value = line.split("\t", 1)
        if key != current:
            if current is not None:
                emit(current, rows)
            current, rows = key, []
        date, rain, tmax, tmin = value.split(",")
        rows.append((date, num(rain), num(tmax), num(tmin)))
    if current is not None:
        emit(current, rows)


if __name__ == "__main__":
    main()
