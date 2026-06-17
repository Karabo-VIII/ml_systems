"""src/strat/period_store.py -- period-keyed storage convention for portfolio runs.

WHY (user /orc 2026-06-12): *"create an annual subfolder for each of the years within
TRAIN/VAL/OOS/UNSEEN -- TRAIN/2020/JAN..., VAL/2023/JAN... for each month/year. This is where we store
the raw runs, charts, and analysis for that particular period."* The point (user methodology): do
things slowly, store every period's artifacts in one place, so structural gaps are found + fixed early
and improvements are measurable period-by-period.

LAYOUT:  runs/periods/<SPLIT>/<YEAR>/<MM_MON>/{raw,charts,analysis}/
  e.g.   runs/periods/TRAIN/2020/01_JAN/charts/ma_killers.png
SPLIT is derived from the start date via the project's canonical TRAIN/VAL/OOS/UNSEEN boundaries
(matches portfolio_replay.WIN). raw/*.json (large, regenerable) is gitignored; charts + analysis track.

API:
  split_of("2020-01-07")            -> "TRAIN"
  period_dir("2020-01-07")          -> Path(.../TRAIN/2020/01_JAN), creating raw/charts/analysis
  sub("2020-01-07", "charts")       -> Path(.../TRAIN/2020/01_JAN/charts)
No emoji (cp1252).
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PERIODS = ROOT / "runs" / "periods"
# canonical split boundaries (inclusive-lo, exclusive-hi); None = open
BOUNDS = [("TRAIN", None, "2024-05-15"), ("VAL", "2024-05-15", "2025-03-15"),
          ("OOS", "2025-03-15", "2025-12-31"), ("UNSEEN", "2025-12-31", "2026-06-01")]
MON = {1: "01_JAN", 2: "02_FEB", 3: "03_MAR", 4: "04_APR", 5: "05_MAY", 6: "06_JUN",
       7: "07_JUL", 8: "08_AUG", 9: "09_SEP", 10: "10_OCT", 11: "11_NOV", 12: "12_DEC"}


def split_of(date) -> str:
    d = dt.date.fromisoformat(str(date)[:10])
    for name, lo, hi in BOUNDS:
        lo_ok = lo is None or d >= dt.date.fromisoformat(lo)
        hi_ok = hi is None or d < dt.date.fromisoformat(hi)
        if lo_ok and hi_ok:
            return name
    return "OUT_OF_RANGE"


def period_dir(start_date, create=True) -> Path:
    d = dt.date.fromisoformat(str(start_date)[:10])
    p = PERIODS / split_of(start_date) / str(d.year) / MON[d.month]
    if create:
        for sub_ in ("raw", "charts", "analysis"):
            (p / sub_).mkdir(parents=True, exist_ok=True)
    return p


def sub(start_date, which="charts", create=True) -> Path:
    p = period_dir(start_date, create) / which
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "2020-01-07"
    print(f"{d} -> split={split_of(d)}  dir={period_dir(d, create=False)}")
