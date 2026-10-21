"""Unit tests for the Hadoop Streaming mapper and reducer.

The fixture is a hand-built district-year whose indices can be worked out by
hand. It is test data only and never enters any reported result.

Run from the repo root:  python3 -m pytest tests/ -q
"""
import datetime as dt
import subprocess
import sys

PY = sys.executable
MAPPER = "mapreduce/mapper.py"
REDUCER = "mapreduce/reducer.py"


def run_pipeline(lines):
    """cat | mapper | sort | reducer, exactly as Hadoop Streaming would chain them."""
    mapped = subprocess.run([PY, MAPPER], input="\n".join(lines) + "\n",
                            capture_output=True, text=True, check=True).stdout
    shuffled = "".join(sorted(mapped.splitlines(keepends=True)))
    out = subprocess.run([PY, REDUCER], input=shuffled,
                         capture_output=True, text=True, check=True).stdout
    rows = {}
    for line in out.splitlines():
        key, value = line.split("\t")
        rows[key] = value.split(",")
    return rows


def year_days(year):
    d = dt.date(year, 1, 1)
    while d.year == year:
        yield d
        d += dt.timedelta(days=1)


def fixture_2001():
    """District 999, year 2001 (365 days).

    Rain : 2 mm every day, except days 100-104 = 70, 80, 10, 0, 90 mm and
           days 200-239 = 0 mm (a 40-day dry run, plus day 103 = 1-day run).
    Tmax : 30 C, except 10 days at 36 C.
    Tmin : 20 C, except 7 days at 26 C.
    """
    lines = []
    for i, d in enumerate(year_days(2001)):
        rain = 2.0
        if 100 <= i <= 104:
            rain = [70.0, 80.0, 10.0, 0.0, 90.0][i - 100]
        if 200 <= i <= 239:
            rain = 0.0
        tmax = 36.0 if i < 10 else 30.0
        tmin = 26.0 if 50 <= i < 57 else 20.0
        lines.append(f"999,{d.isoformat()},{rain},{tmax},{tmin}")
    return lines


def test_indices_by_hand():
    lines = fixture_2001()
    # Hadoop does not sort values within a key: shuffle the input order
    lines = lines[::2] + lines[1::2]
    rows = run_pipeline(["censuscode,date,rain_mm,tmax_c,tmin_c"] + lines)
    assert list(rows) == ["999,2001"]
    (n_days, n_rain, n_tmax, n_tmin, annual, heavy, rx5, dry,
     mean_tmax, hot, trop) = rows["999,2001"]

    normal_days = 365 - 5 - 40
    assert int(n_days) == 365 and int(n_rain) == 365
    assert int(n_tmax) == 365 and int(n_tmin) == 365
    assert float(annual) == normal_days * 2.0 + 70 + 80 + 10 + 0 + 90
    assert int(heavy) == 3                       # 70, 80, 90 >= 64.5
    assert float(rx5) == 70 + 80 + 10 + 0 + 90   # days 100-104
    assert int(dry) == 40
    assert abs(float(mean_tmax) - (10 * 36 + 355 * 30) / 365) < 0.006
    assert int(hot) == 10
    assert int(trop) == 7


def test_missing_values_are_not_filled():
    lines = fixture_2001()
    # blank one rain value and one tmin value
    lines[30] = lines[30].replace(",2.0,", ",,")
    code, date, rain, tmax, tmin = lines[60].split(",")
    lines[60] = ",".join([code, date, rain, tmax, ""])
    vals = run_pipeline(lines)["999,2001"]
    n_days, n_rain, n_tmax, n_tmin = map(int, vals[:4])
    assert (n_days, n_rain, n_tmax, n_tmin) == (365, 364, 365, 364)
    # every rain index and tropical nights must be empty, not zero
    assert vals[4:8] == ["", "", "", ""]
    assert vals[10] == ""
    # Tmax is complete, so its indices are still computed
    assert vals[8] != "" and vals[9] == "10"


def test_pre_1951_years_have_no_temperature():
    lines = [f"5,{d.isoformat()},1.5,," for d in year_days(1920)]
    vals = run_pipeline(lines)["5,1920"]
    assert vals[4] != "" and vals[8:] == ["", "", ""]


def test_bad_rows_dropped_and_districts_kept_apart():
    lines = fixture_2001() + ["garbage", "1,2001-01-01,1,2", "x,2001-01-01,1,2,3"]
    lines += [f"7,{d.isoformat()},0.0,40.0,30.0" for d in year_days(2004)]
    rows = run_pipeline(lines)
    assert set(rows) == {"999,2001", "7,2004"}
    v = rows["7,2004"]
    assert v[0] == "366"            # leap year
    assert v[7] == "366"            # whole year dry
    assert v[9] == "366" and v[10] == "366"
