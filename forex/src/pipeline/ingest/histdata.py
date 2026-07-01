"""HistData.com M1 ingester — the simplest free bulk archive (secondary source).

HistData provides free 1-minute OHLC bars (bid quotes only; VOLUME IS ALWAYS 0
for FX) back to 2000, as per-pair/per-month ZIPs. Two gotchas handled here:
  1. Timestamps are EST WITHOUT DST (a FIXED UTC-5 offset), not America/New_York
     wall-clock -> converted to UTC by subtracting 5h uniformly.
  2. Volume column is always 0 for FX -> we drop it and rely on bar tick_count
     from the Dukascopy path instead.

Downloading is automated by the community `histdata` PyPI lib (philipperemy)
when installed; otherwise this module still parses an already-downloaded CSV.
The correct, always-available part is the parser + UTC normalization.

Output: data/raw/<PAIR>/m1/<pair>_m1_<YYYY-MM>.parquet, columns
[timestamp(ms, UTC), open, high, low, close] (bid-based).

    python -m pipeline.ingest.histdata --pairs EURUSD --start 2024-01 --end 2024-03
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

_PIPELINE_PARENT = str(Path(__file__).resolve().parents[2])
if _PIPELINE_PARENT not in sys.path:
    sys.path.insert(0, _PIPELINE_PARENT)

from pipeline import layout
from pipeline.parquet_io import atomic_write_parquet

__contract__ = {
    "kind": "pipeline_stage",
    "stage": "fetch_histdata",
    "inputs": {"args": ["--pairs", "--start (YYYY-MM)", "--end (YYYY-MM)"]},
    "outputs": {"files": "data/raw/<PAIR>/m1/<pair>_m1_<YYYY-MM>.parquet",
                "columns": ["timestamp", "open", "high", "low", "close"]},
    "invariants": {"est_no_dst_to_utc": True, "volume_always_zero_dropped": True,
                   "atomic_write": True},
    "rationale": "Secondary free M1 archive; cross-check / backfill for Dukascopy.",
}

# HistData M1 timestamps are EST without DST == fixed UTC-5.
_EST_NO_DST_OFFSET = timedelta(hours=5)


def parse_m1_csv(path: Path, pair: str) -> pl.DataFrame:
    """Parse a HistData ASCII M1 CSV (`YYYYMMDD HHMMSS;O;H;L;C;V`) to UTC bars."""
    df = pl.read_csv(path, separator=";", has_header=False,
                     new_columns=["dt", "open", "high", "low", "close", "volume"])
    # dt like "20240102 170000" in EST-no-DST -> parse then + 5h to UTC.
    out = df.with_columns(
        pl.col("dt").str.strptime(pl.Datetime, "%Y%m%d %H%M%S", strict=False)
    ).drop_nulls("dt").with_columns(
        ((pl.col("dt").dt.timestamp("ms")) + int(_EST_NO_DST_OFFSET.total_seconds() * 1000))
        .alias("timestamp")
    ).select(["timestamp",
              pl.col("open").cast(pl.Float64), pl.col("high").cast(pl.Float64),
              pl.col("low").cast(pl.Float64), pl.col("close").cast(pl.Float64)])
    return out.sort("timestamp").unique(subset=["timestamp"], keep="first")


def ingest_csv(path: Path, pair: str, year: int, month: int) -> dict:
    df = parse_m1_csv(path, pair)
    out = layout.m1_month_path(pair, year, month)
    atomic_write_parquet(df, out, required_cols={"timestamp", "open", "close"})
    return {"status": "ok", "pair": pair, "month": f"{year:04d}-{month:02d}",
            "bars": len(df), "path": str(out)}


def download_month(pair: str, year: int, month: int, tmp_dir: Path) -> Path | None:
    """Download one month's M1 ZIP via the `histdata` lib if available. Returns
    the extracted CSV path, or None with instructions if the lib is missing."""
    try:
        from histdata import download_hist_data
        from histdata.api import Platform, TimeFrame
    except ImportError:
        print("[histdata] `pip install histdata` to enable auto-download; "
              "otherwise download the M1 ASCII ZIP manually from histdata.com "
              "and pass the CSV to ingest_csv().", flush=True)
        return None
    n = layout.normalize_pair(pair)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    # Verified signature (philipperemy/histdata): download_hist_data(year, month,
    # pair, time_frame, platform, output_directory, verbose) -> zip filename (str).
    # Platform.GENERIC_ASCII='ASCII', TimeFrame.ONE_MINUTE='M1'. Only M1/tick exist
    # (resample higher cadences ourselves). month=None would pull the whole year.
    try:
        zip_ret = download_hist_data(year=str(year), month=str(month),
                                     pair=n.lower, time_frame=TimeFrame.ONE_MINUTE,
                                     platform=Platform.GENERIC_ASCII,
                                     output_directory=str(tmp_dir), verbose=False)
    except Exception as e:
        print(f"[histdata] download {pair} {year}-{month:02d} FAIL: "
              f"{type(e).__name__}: {e}", flush=True)
        return None
    # Returned value is a zip filename; resolve relative to tmp_dir if not absolute.
    zip_path = Path(zip_ret)
    if not zip_path.is_absolute():
        zip_path = tmp_dir / zip_path.name
    import zipfile
    with zipfile.ZipFile(zip_path) as z:
        csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not csvs:
            return None
        z.extract(csvs[0], tmp_dir)
        return tmp_dir / csvs[0]


def _months(start: str, end: str):
    y, m = (int(x) for x in start.split("-"))
    ey, em = (int(x) for x in end.split("-"))
    while (y, m) <= (ey, em):
        yield y, m
        m += 1
        if m > 12:
            m, y = 1, y + 1


def main() -> None:
    from pipeline import cli
    ap = argparse.ArgumentParser(description="HistData M1 fetcher (secondary)")
    # start/end here are YYYY-MM months, not full dates.
    cli.add_standard_args(ap, default_workers=1, date_window=False)
    ap.add_argument("--start", default="2024-01", help="Start month YYYY-MM.")
    ap.add_argument("--end", default="2024-01", help="End month YYYY-MM inclusive.")
    args = ap.parse_args()
    pairs = cli.resolve_pairs(args, stage_name="histdata")
    tmp = layout.DATA / "_histdata_tmp"
    for p in pairs:
        for y, m in _months(args.start, args.end):
            csv = download_month(p, y, m, tmp)
            if csv is None:
                continue
            res = ingest_csv(Path(csv), p, y, m)
            print(f"[histdata] {res}", flush=True)


if __name__ == "__main__":
    main()
