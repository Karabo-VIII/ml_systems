"""Alternative information-driven bars (silver) — tick-count + range bars.

Time bars sample on a fixed clock; these sample on information/activity, so more
bars form when the market is active (news, London/NY open) and fewer in the
Asian lull. The FX analog of crypto's DIB / range bars.

  tick_count bars: a new bar every N ticks (fixed information content per bar).
  range bars:      a new bar every time mid moves R pips (fixed price content).

Both carry MID OHLC + bid/ask close + tick_count. Built from raw Dukascopy ticks.

    python src/pipeline/build_alt_bars.py --universe u7 --tick-n 500 --range-pips 10
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

_PIPELINE_PARENT = str(Path(__file__).resolve().parents[1])
if _PIPELINE_PARENT not in sys.path:
    sys.path.insert(0, _PIPELINE_PARENT)

from pipeline import cli, layout
from pipeline.dispatch import run_per_task
from pipeline.parquet_io import atomic_write_parquet
from pipeline.universe_loader import UniverseLoader

__contract__ = {
    "kind": "pipeline_stage",
    "stage": "build_alt_bars",
    "inputs": {"upstream": "data/raw/<PAIR>/ticks/*.parquet",
               "args": ["--pairs", "--universe", "--tick-n", "--range-pips"]},
    "outputs": {"files": "data/processed/bars/{tick_count,range}/<pair>_<type>_<YYYYMMDD>.parquet",
                "columns": ["timestamp", "open", "high", "low", "close",
                            "bid_close", "ask_close", "tick_count"]},
    "invariants": {"information_driven_sampling": True, "mid_ohlc": True, "atomic_write": True},
    "rationale": "Alt bar fabric; activity-clock sampling for microstructure studies.",
}

_ALT_COLS = ["timestamp", "open", "high", "low", "close", "bid_close", "ask_close", "tick_count"]


def tick_count_bars(df: pl.DataFrame, n: int) -> pl.DataFrame:
    """Fixed-N-tick bars (vectorized via row-index // n grouping)."""
    b = (df.with_row_index("i")
         .with_columns((pl.col("i") // n).alias("bar_id"))
         .group_by("bar_id").agg([
             pl.col("timestamp").first().alias("timestamp"),
             pl.col("mid").first().alias("open"), pl.col("mid").max().alias("high"),
             pl.col("mid").min().alias("low"), pl.col("mid").last().alias("close"),
             pl.col("bid").last().alias("bid_close"), pl.col("ask").last().alias("ask_close"),
             pl.len().alias("tick_count"),
         ]).sort("timestamp"))
    return b.select(_ALT_COLS)


def range_bars(df: pl.DataFrame, range_pips: float, pip: float) -> pl.DataFrame:
    """New bar whenever the intrabar mid range reaches range_pips (numpy loop)."""
    thr = range_pips * pip
    ts = df["timestamp"].to_numpy()
    mid = df["mid"].to_numpy()
    bid = df["bid"].to_numpy()
    ask = df["ask"].to_numpy()
    if len(mid) == 0:
        return pl.DataFrame(schema={c: (pl.Int64 if c in ("timestamp", "tick_count") else pl.Float64)
                                    for c in _ALT_COLS})
    rows = []
    o = h = lo = mid[0]
    start_ts = ts[0]
    n = 1
    for i in range(1, len(mid)):
        m = mid[i]
        h = m if m > h else h
        lo = m if m < lo else lo
        n += 1
        if h - lo >= thr:
            rows.append((int(start_ts), float(o), float(h), float(lo), float(m),
                         float(bid[i]), float(ask[i]), int(n)))
            o = h = lo = m
            start_ts = ts[i]
            n = 1
    if n > 1:                       # trailing partial bar
        rows.append((int(start_ts), float(o), float(h), float(lo), float(mid[-1]),
                     float(bid[-1]), float(ask[-1]), int(n)))
    return pl.DataFrame(rows, schema=_ALT_COLS, orient="row")


def build_pair(pair: str, n_ticks: int, range_pips: float, start: date | None,
               end: date | None) -> dict:
    days = layout.list_tick_days(pair)
    if start:
        days = [d for d in days if d >= start]
    if end:
        days = [d for d in days if d < end]
    if not days:
        return {"status": "no_ticks", "pair": pair}
    paths = [str(layout.tick_day_path(pair, d)) for d in days]
    df = (pl.scan_parquet(paths)
          .with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"))
          .sort("timestamp").collect())
    if df.is_empty():
        return {"status": "no_ticks", "pair": pair}
    U = UniverseLoader.load()
    pip = U.pip(pair)
    latest = max(days)
    tcb = tick_count_bars(df, n_ticks)
    atomic_write_parquet(tcb, layout.bars_path(pair, "tick_count", latest_date=latest),
                         required_cols={"timestamp", "open", "close"})
    rb = range_bars(df, range_pips, pip)
    atomic_write_parquet(rb, layout.bars_path(pair, "range", latest_date=latest),
                         required_cols={"timestamp", "open", "close"})
    return {"status": "ok", "pair": pair, "n_tick_bars": len(tcb), "n_range_bars": len(rb)}


def _worker(pair, n_ticks, range_pips, start, end):
    return build_pair(pair, n_ticks, range_pips, start, end)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build information-driven FX bars")
    cli.add_standard_args(ap, default_workers=4, default_start="2020-01-01")
    ap.add_argument("--tick-n", type=int, default=500, help="Ticks per tick-count bar.")
    ap.add_argument("--range-pips", type=float, default=10.0, help="Pips per range bar.")
    args = ap.parse_args()
    pairs = cli.resolve_pairs(args, stage_name="build_alt_bars")
    start = date.fromisoformat(args.start) if args.start else None
    end = date.fromisoformat(args.end) if args.end else None
    tasks = [(p, args.tick_n, args.range_pips, start, end) for p in pairs]
    if args.dry_run:
        for t in tasks:
            print(f"  would build alt bars {t[0]} tick_n={t[1]} range_pips={t[2]}")
        return
    run_per_task(tasks, _worker, workers=args.workers,
                 mode="thread" if args.workers > 1 else "serial",
                 stage_name="build_alt_bars",
                 progress_summary_keys=["n_tick_bars", "n_range_bars"])


if __name__ == "__main__":
    main()
