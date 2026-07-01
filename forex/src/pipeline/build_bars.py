"""Build time bars from raw Dukascopy ticks (silver layer).

Spot FX has no consolidated volume, so TIME bars are the primary bar type
(resampled from ticks). Each bar carries MID OHLC plus the FX-first-class
fields crypto bars don't need: bid/ask close and spread stats (for spread-aware
execution) and a tick_count volume proxy.

    data/raw/<PAIR>/ticks/*.parquet  ->  data/processed/bars/<cadence>/<pair>_<cadence>_<YYYYMMDD>.parquet

Bar schema per cadence:
    timestamp    bar-open time (ms, UTC)
    open/high/low/close   MID price OHLC
    bid_close, ask_close  last bid/ask in the bar (execution / cost modelling)
    spread_mean, spread_max   intrabar spread (price units)
    tick_count   number of ticks in the bar (volume proxy)
    volume_est   Dukascopy bid+ask volume estimate (millions base ccy; proxy)

Cadences (time-only): 1m 5m 15m 30m 1h 4h 1d. No spurious weekend bars are
created — windows exist only where ticks exist (24/5 calendar respected).
NOTE: 1d bars use UTC-midnight boundaries; a 17:00-New-York daily boundary is a
documented future refinement (see CLAUDE.md).

    python src/pipeline/build_bars.py --universe u7 --cadences 1m 5m 1h 1d
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import polars as pl

_PIPELINE_PARENT = str(Path(__file__).resolve().parents[1])
if _PIPELINE_PARENT not in sys.path:
    sys.path.insert(0, _PIPELINE_PARENT)

from pipeline import cli, layout
from pipeline.dispatch import run_per_task
from pipeline.parquet_io import atomic_write_parquet

__contract__ = {
    "kind": "pipeline_stage",
    "stage": "build_bars",
    "inputs": {"upstream": "data/raw/<PAIR>/ticks/*.parquet",
               "args": ["--pairs", "--universe", "--cadences", "--start", "--end", "--force"]},
    "outputs": {"files": "data/processed/bars/<cadence>/<pair>_<cadence>_<YYYYMMDD>.parquet",
                "columns": ["timestamp", "open", "high", "low", "close", "bid_close",
                            "ask_close", "spread_mean", "spread_max", "tick_count", "volume_est"]},
    "invariants": {"mid_ohlc": True, "bid_ask_spread_first_class": True,
                   "no_weekend_bars": True, "atomic_write": True},
    "rationale": "Silver bar fabric; features + strategies read these.",
}

_BAR_COLS = ["timestamp", "open", "high", "low", "close", "bid_close", "ask_close",
             "spread_mean", "spread_max", "tick_count", "volume_est"]


def _resample(df: pl.DataFrame, cadence: str) -> pl.DataFrame:
    """Resample a tick DataFrame (with a 'dt' Datetime col) into OHLC bars."""
    b = (df.group_by_dynamic("dt", every=cadence, closed="left", label="left")
         .agg([
             pl.col("mid").first().alias("open"),
             pl.col("mid").max().alias("high"),
             pl.col("mid").min().alias("low"),
             pl.col("mid").last().alias("close"),
             pl.col("bid").last().alias("bid_close"),
             pl.col("ask").last().alias("ask_close"),
             pl.col("spread").mean().alias("spread_mean"),
             pl.col("spread").max().alias("spread_max"),
             pl.len().alias("tick_count"),
             (pl.col("bid_volume") + pl.col("ask_volume")).sum().alias("volume_est"),
         ]))
    return (b.with_columns(pl.col("dt").dt.timestamp("ms").alias("timestamp"))
            .sort("timestamp").select(_BAR_COLS))


def build_pair(pair: str, cadences: list[str], start: date | None,
               end: date | None, force: bool) -> dict:
    days = layout.list_tick_days(pair)
    if start:
        days = [d for d in days if d >= start]
    if end:
        days = [d for d in days if d < end]
    if not days:
        return {"status": "no_ticks", "pair": pair}
    paths = [str(layout.tick_day_path(pair, d)) for d in days]
    df = (pl.scan_parquet(paths)
          .with_columns([
              ((pl.col("bid") + pl.col("ask")) / 2).alias("mid"),
              (pl.col("ask") - pl.col("bid")).alias("spread"),
              pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("dt"),
          ])
          .sort("dt")
          .collect())
    if df.is_empty():
        return {"status": "no_ticks", "pair": pair}
    latest = max(days)
    out = {"status": "ok", "pair": pair, "tick_days": len(days)}
    for cad in cadences:
        bars = _resample(df, cad)
        dest = layout.bars_path(pair, cad, latest_date=latest)
        atomic_write_parquet(bars, dest, required_cols={"timestamp", "open", "close"})
        out[f"n_{cad}"] = len(bars)
    return out


def _worker(pair, cadences, start, end, force):
    return build_pair(pair, cadences, start, end, force)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build FX time bars from ticks")
    cli.add_standard_args(ap, default_workers=4, default_start="2020-01-01")
    ap.add_argument("--cadences", nargs="+", default=list(layout.CADENCES),
                    help=f"Cadences to build (default all: {' '.join(layout.CADENCES)}).")
    args = ap.parse_args()

    for c in args.cadences:
        if c not in layout.CADENCES:
            raise SystemExit(f"unknown cadence {c!r}; valid: {layout.CADENCES}")

    pairs = cli.resolve_pairs(args, stage_name="build_bars")
    start = date.fromisoformat(args.start) if args.start else None
    end = date.fromisoformat(args.end) if args.end else None
    tasks = [(p, args.cadences, start, end, args.force) for p in pairs]
    if args.dry_run:
        for t in tasks:
            print(f"  would build {t[0]} cadences={t[1]}")
        return
    run_per_task(tasks, _worker, workers=args.workers,
                 mode="thread" if args.workers > 1 else "serial",
                 stage_name="build_bars",
                 progress_summary_keys=[f"n_{c}" for c in args.cadences[:3]] + ["tick_days"])


if __name__ == "__main__":
    main()
