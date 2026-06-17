"""Tick / Volume / Dollar Runs Bars (Lopez de Prado, AFML chapter 2).

A "run" = consecutive same-direction trades. Runs bars close when the
*imbalanced directional flow* exceeds the expected run length, suggesting
informed-trading is present.

Formal definition (de Prado):
    Cumulative signed ticks / vol / $: T_t = sum(b_i) where b_i ∈ {-1, +1}
    Expected per-bar imbalance under no informed trading ≈ 0
    Bar closes when |T_t| > Expectation_of_max_run(T)

Simplified practical implementation: close bar when the longer of
{consecutive buy ticks, consecutive sell ticks} exceeds a threshold, OR
when cumulative signed count exceeds threshold. We use the LATTER (simpler
and stable).

Scope: top-10 assets, 2024-2026 window.
Output: data/frontier/runs_bars/<asset>_tick_runs.parquet
        data/frontier/runs_bars/<asset>_vol_runs.parquet
        data/frontier/runs_bars/<asset>_dollar_runs.parquet (same as DIB; skip)
"""
from __future__ import annotations

__contract__ = {
    "kind": "bar_producer",
    "inputs": ["data/raw/<SYM>/aggTrades/*.parquet"],
    "outputs": [
        "data/processed/bars/runs_tick/<SYM>_tick_runs_<tag>.parquet",
        "data/processed/bars/runs_volume/<SYM>_vol_runs_<tag>.parquet",
    ],
    "invariants": [
        "ts_ms_13digit",
        "bar_id_globally_unique_per_asset",
        "imbalance_reset_semantics",
        "atomic_write_via_parquet_io",
        "canonical_dir_names: runs_tick / runs_volume",
    ],
}

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

# Framework primitives.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parquet_io import atomic_write_parquet, delta_state, append_parquet
from dispatch import run_per_task
from cli import add_standard_args, resolve_assets

# Shared u87 threshold fallback (added 2026-05-12 for alt-bar full-coverage rebuild)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _thresholds import get_threshold as _get_yaml_threshold
from _aggtrades_utils import prepare_aggtrades, imbalance_bar_ids


# 2026-05-22 oracle pipeline-progress closure: lazy phase_log helper with
# dual-import fallback (works whether src/ or src/pipeline/ is on sys.path).
def _pl(phase, message, **kw):
    try:
        from progress import phase_log
    except ImportError:
        from pipeline.progress import phase_log
    phase_log("runs", phase, message, **kw)


def _date_from_aggtrades_path(fp):
    parts = fp.stem.split("-")
    if len(parts) < 5:
        return None
    try:
        return pd.Timestamp("-".join(parts[-3:])).date()
    except Exception:
        return None

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"
# Canonical layout v3: data/processed/bars/runs_<mode>/<sym>_<mode>_runs.parquet
# Mode is 'tick' or 'volume'; the runner picks subfolder per write below.
OUT_DIR = ROOT / "data" / "processed" / "bars"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Per-asset thresholds for TICK runs (count of signed ticks)
TICK_RUNS_THRESHOLDS = {
    "BTCUSDT": 3000, "ETHUSDT": 2500, "SOLUSDT": 2000, "BNBUSDT": 1500,
    "XRPUSDT": 1500, "DOGEUSDT": 1500, "ADAUSDT": 1200, "AVAXUSDT": 1200,
    "LINKUSDT": 1000, "LTCUSDT": 1000,
}

# Per-asset thresholds for VOLUME runs (signed volume in base units)
VOL_RUNS_THRESHOLDS = {
    "BTCUSDT": 200, "ETHUSDT": 5000, "SOLUSDT": 50000, "BNBUSDT": 3000,
    "XRPUSDT": 5_000_000, "DOGEUSDT": 50_000_000, "ADAUSDT": 10_000_000,
    "AVAXUSDT": 100_000, "LINKUSDT": 500_000, "LTCUSDT": 50_000,
}


def build_runs_bars_day(fp: Path, mode: str, threshold: float) -> pl.DataFrame:
    """Build runs bars for one day.
    mode: 'tick' = signed-tick imbalance; 'vol' = signed-volume imbalance.
    """
    try:
        df = pl.read_parquet(fp)
    except Exception:
        return pl.DataFrame()
    if df.is_empty():
        return pl.DataFrame()
    # Normalize ts scale + sort (Binance us-scale + unsort issues, 2025+)
    df = prepare_aggtrades(df, ts_col="timestamp")

    df = df.with_columns([
        pl.when(pl.col("is_buyer_maker")).then(-1.0).otherwise(1.0).alias("sign"),
        (pl.col("price") * pl.col("qty")).alias("value_usd"),
    ])
    if mode == "tick":
        df = df.with_columns(pl.col("sign").alias("run_signal"))
    elif mode == "vol":
        df = df.with_columns((pl.col("sign") * pl.col("qty")).alias("run_signal"))
    else:
        raise ValueError(f"unknown mode: {mode}")

    # AFML runs/imbalance bar WITH RESET: close a bar (reset accumulator) when
    # the run signal accumulated since the last bar crosses +/- threshold. The
    # previous floor(cumulative_sum/threshold) never reset (grouped by absolute
    # cumulative level), which is not a runs bar for an oscillating signal.
    bar_ids = imbalance_bar_ids(df["run_signal"].to_numpy(), threshold)
    df = df.with_columns(pl.Series("bar_id", bar_ids))

    bars = df.group_by("bar_id").agg([
        pl.col("timestamp").first().alias("bar_start_ts"),
        pl.col("timestamp").last().alias("bar_end_ts"),
        pl.col("price").first().alias("open"),
        pl.col("price").max().alias("high"),
        pl.col("price").min().alias("low"),
        pl.col("price").last().alias("close"),
        pl.col("qty").sum().alias("volume"),
        pl.col("run_signal").sum().alias("signed_run"),
        pl.when(pl.col("is_buyer_maker")).then(pl.col("value_usd")).otherwise(0).sum().alias("sell_usd"),
        pl.when(~pl.col("is_buyer_maker")).then(pl.col("value_usd")).otherwise(0).sum().alias("buy_usd"),
        pl.len().alias("tick_count"),
    ]).sort("bar_id")
    return bars


def build_asset_from_fps(symbol: str, mode: str, threshold: float,
                          fps_filt: list) -> pl.DataFrame:
    """Build runs bars over an explicit file list (delta-mode friendly)."""
    print(f"[{symbol}/{mode}] threshold={threshold}, {len(fps_filt)} days...",
          flush=True)
    all_bars = []
    t0 = time.time()
    for i, fp in enumerate(fps_filt):
        bars = build_runs_bars_day(Path(fp), mode, threshold)
        if not bars.is_empty():
            # Tag each bar with its source date so delta append works.
            d = _date_from_aggtrades_path(Path(fp))
            if d is not None:
                bars = bars.with_columns(pl.lit(d).alias("date"))
            all_bars.append(bars)
        if (i + 1) % 100 == 0:
            print(f"  [{symbol}/{mode}] {i+1}/{len(fps_filt)} days in "
                  f"{time.time()-t0:.0f}s", flush=True)
    if not all_bars:
        return pl.DataFrame()
    return pl.concat(all_bars)


def _build_one_task(symbol: str, mode: str, threshold: float,
                     fps_to_process: list, out_path: str,
                     write_mode: str) -> dict:
    """ProcessPool worker: build one (asset, mode) combination.
    write_mode='rebuild' -> atomic full write; 'append' -> delta append by date.
    """
    bars = build_asset_from_fps(symbol, mode, threshold, fps_to_process)
    if bars.is_empty():
        # Append mode + empty = legitimate (low-vol days never crossed
        # the runs-imbalance threshold). Existing rows unchanged.
        is_real_failure = (write_mode == "rebuild")
        return {"status": "empty" if is_real_failure else "ok",
                "symbol": symbol, "mode": mode, "n_bars": 0,
                "write_mode": write_mode,
                "note": "no bars produced"}
    # Merge with existing (append) BEFORE reindexing so bar_id is globally unique
    # across the whole file (CLAUDE.md invariant). Consumers join on timestamps.
    if write_mode == "append" and Path(out_path).exists():
        existing = pl.read_parquet(out_path)
        new_dates = set(bars["date"].to_list())
        keep = existing.filter(~pl.col("date").is_in(list(new_dates)))
        final = pl.concat([keep, bars], how="vertical_relaxed")
    else:
        final = bars
    if "bar_id" in final.columns:
        final = final.drop("bar_id")
    final = final.sort(["date", "bar_start_ts"]).with_row_index("bar_id")
    atomic_write_parquet(final, out_path)
    n_bars = len(final)
    return {"status": "ok", "symbol": symbol, "mode": mode,
            "n_bars": n_bars, "n_new_days": len(fps_to_process),
            "write_mode": write_mode}


def main():
    ap = argparse.ArgumentParser()
    add_standard_args(ap, default_workers=1)
    ap.add_argument("--modes", nargs="+", default=["tick", "vol"],
                    help="Bar modes to build: tick, vol, or both.")
    ap.add_argument("--burn-from-first-gap", action="store_true",
                    help="On mid-stream gap, rebuild from gap forward.")
    # Phase 7 bidirectional pattern
    ap.add_argument("-r", "--reverse", action="store_true",
                    help="Reverse asset iteration (Z->A) for meet-in-middle "
                         "2x speedup. Run two terminals: one without -r, one with.")
    args = ap.parse_args()

    symbols = resolve_assets(args, default=list(TICK_RUNS_THRESHOLDS.keys()),
                              stage_name="runs")
    from pipeline.bidirectional import iter_assets
    symbols = list(iter_assets(symbols, reverse=args.reverse))
    if args.reverse:
        print(f"[runs] REVERSE mode: iterating {len(symbols)} assets Z->A",
              flush=True)

    tasks: list[tuple] = []
    n_skipped = n_appends = n_rebuilds = 0
    for symbol in symbols:
        raw_dir = RAW / symbol / "aggTrades"
        raw_fps = []
        if raw_dir.exists():
            raw_fps = [Path(fp) for fp in glob.glob(
                            str(raw_dir / f"{symbol}-aggTrades-*.parquet"))
                        if args.start <= "-".join(Path(fp).stem.split("-")[-3:]) < args.end]
        for mode in args.modes:
            if mode == "tick":
                th = TICK_RUNS_THRESHOLDS.get(symbol) or _get_yaml_threshold(symbol, "tick_runs")
            elif mode == "vol":
                th = VOL_RUNS_THRESHOLDS.get(symbol) or _get_yaml_threshold(symbol, "vol_runs")
            else:
                _pl("SKIP", f"unknown mode {mode!r}; skipping")
                continue
            # Canonical dir names: tick -> runs_tick, vol -> runs_volume
            # (layout.bars_latest expects "runs_volume"; "runs_vol" was a legacy
            # short-name only tolerated via fallback).
            dir_name = "runs_volume" if mode == "vol" else f"runs_{mode}"
            sub = OUT_DIR / dir_name
            year_tag = args.start[:4]
            out = sub / f"{symbol}_{mode}_runs_{year_tag}.parquet"
            delta = delta_state(out, raw_fps, force=args.force,
                                 date_from_filename=_date_from_aggtrades_path,
                                 burn_from_first_gap=args.burn_from_first_gap,
                                 required_cols={"date", "open", "close", "signed_run"},
                                 max_null_rate={"close": 0.01, "signed_run": 0.01})
            if delta["mode"] == "fresh":
                _pl("SKIP", f"{symbol}/{mode} SKIP (fresh: {out.name})")
                n_skipped += 1
                continue
            if delta["mode"] == "append":
                n_appends += 1
            else:
                n_rebuilds += 1
            tasks.append((symbol, mode, th,
                           [str(p) for p in delta["new_inputs"]],
                           str(out), delta["mode"]))

    if args.dry_run:
        print(f"[runs] dry-run: {n_appends} appends + {n_rebuilds} rebuilds + "
              f"{n_skipped} fresh-skips")
        return

    if not tasks:
        _pl("SKIP", f"nothing to build ({n_skipped} skipped)")
        return

    run_per_task(tasks, _build_one_task,
                  workers=args.workers, mode="process",
                  stage_name="runs",
                  progress_summary_keys=["mode", "write_mode", "n_new_days", "n_bars"])


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
