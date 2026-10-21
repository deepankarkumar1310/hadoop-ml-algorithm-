#!/usr/bin/env python3
"""Hadoop Streaming mapper: district-day records -> (district, year) groups.

Input  (one line per district per day, CSV, header optional)
    censuscode,date,rain_mm,tmax_c,tmin_c
    e.g. 603,1991-10-14,12.37,31.20,24.05
    A missing value is an empty field, never zero.

Output (key TAB value)
    censuscode,year <TAB> date,rain_mm,tmax_c,tmin_c

Malformed lines are counted in the Hadoop counter Project:bad_rows and
dropped; they are never repaired or guessed.
"""
import sys


def counter(name, n=1):
    sys.stderr.write(f"reporter:counter:Project,{name},{n}\n")


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line or line.startswith("censuscode"):
            continue
        parts = line.split(",")
        if len(parts) != 5:
            counter("bad_rows")
            continue
        code, date, rain, tmax, tmin = parts
        if not code.isdigit() or len(date) != 10 or date[4] != "-":
            counter("bad_rows")
            continue
        print(f"{code},{date[:4]}\t{date},{rain},{tmax},{tmin}")


if __name__ == "__main__":
    main()
