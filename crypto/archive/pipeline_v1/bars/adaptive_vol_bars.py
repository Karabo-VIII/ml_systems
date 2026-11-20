"""Adaptive Volatility Bars — bar threshold scales with EWMA realized vol.

Standard dollar bars: fixed $X threshold regardless of market regime.
In high-vol periods, bars close faster and contain less time but equal $.
In low-vol periods, bars close slower.

Adaptive vol bars: dynamic threshold T_t = T_base × (σ_t / σ_ref)^α
Effect: in high-vol periods, LARGER $ threshold (slower bar closes) — compensates
for noisy price action; in low-vol periods, smaller threshold (faster closes) —
surfaces meaningful micro-moves.

Simplified: use EWMA absolute-return volatility (computed minute-by-minute from
aggTrades). Each day, set threshold = T_base × max(σ_day/σ_30d, 0.3).

Output: data/processed/bars/adaptive_vol/<asset>_adaptive_vol_<tag>.parquet

NOTE (perf, known limitation): append/windowed-delta mode currently re-reads the
full date range each run (delta["new_inputs"] is not threaded into build_asset).
Output is correct; only redundant compute. Orphan compute today (no chimera
consumer), so left as-is pending a consumer.
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

__contract__ = {
    "kind": "bar_producer",
    "inputs": ["data/raw/<SYM>/aggTrades/*.parquet"],
    "outputs": ["data/processed/bars/adaptive_vol/<SYM>_adaptive_vol_<tag>.parquet"],
    "invariants": [
        "ts_ms_13digit",
        "date_column_present",
        "bar_id_globally_unique_per_asset",
        "atomic_write_via_parquet_io",
    ],
}

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
from _aggtrades_utils import prepare_aggtrades


# 2026-05-22 oracle pipeline-progress closure: lazy phase_log helper with
# dual-import fallback (works whether src/ or src/pipeline/ is on sys.path).
def _pl(phase, message, **kw):
    try:
        from progress import phase_log
    except ImportError:
        from pipeline.progress import phase_log
    phase_log("adaptive_vol", phase, message, **kw)


def _date_from_aggtrades_path(fp):
    parts = fp.stem.split("-")
    if len(parts) < 5:
        return None
    try:
        return pd.Timestamp("-".join(parts[-3:])).date()
    except Exception:
        return None


# Adaptive-vol uses a 30-day rolling sigma_30d for the threshold scaling;
# delta-mode must therefore include 30 days of overlap context so the
# threshold for newly-appended days uses the correct vol baseline.
ADAPTIVE_VOL_WINDOW_DAYS = 30

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"
# Canonical layout v3: data/processed/bars/adaptive_vol/<sym>_adaptive_vol.parquet
OUT_DIR = ROOT / "data" / "processed" / "bars" / "adaptive_vol"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Base thresholds (same as dollar bars calibration). Adaptive threshold will
# scale these by relative volatility.
BASE_THRESHOLDS_USD = {
    "BTCUSDT": 10_000_000, "ETHUSDT": 5_000_000, "SOLUSDT": 1_000_000,
    "BNBUSDT": 1_000_000, "XRPUSDT": 500_000, "DOGEUSDT": 500_000,
    "ADAUSDT": 300_000, "AVAXUSDT": 300_000, "LINKUSDT": 300_000, "LTCUSDT": 300_000,
}


def _daily_log_range_vol(df: pl.DataFrame) -> float:
    """Quick per-day realized vol proxy: log(high/low)."""
    price = df["price"]
    if price.len() < 2:
        return 0.0
    hi = price.max()
    lo = price.min()
    if lo <= 0:
        return 0.0
    return float(np.log(hi / lo))


def build_day_with_adaptive(fp: Path, base_threshold: float,
                            vol_ref: float, vol_today: float) -> pl.DataFrame:
    """Threshold for today scales: T = base × clip(vol_today / vol_ref, 0.3, 3.0)."""
    try:
        df = pl.read_parquet(fp)
    except Exception:
        return pl.DataFrame()
    # Normalize ts scale + sort (Binance us-scale + unsort issues, 2025+)
    if not df.is_empty():
        df = prepare_aggtrades(df, ts_col="timestamp")
    if df.is_empty():
        return pl.DataFrame()

    scale = np.clip(vol_today / max(vol_ref, 1e-6), 0.3, 3.0)
    threshold = base_threshold * scale

    df = df.with_columns([
        (pl.col("price") * pl.col("qty")).alias("value_usd"),
    ])
    df = df.with_columns(pl.col("value_usd").cum_sum().alias("cum_value"))
    df = df.with_columns(
        (pl.col("cum_value") / threshold).floor().alias("bar_id")
    )
    bars = df.group_by("bar_id").agg([
        pl.col("timestamp").first().alias("bar_start_ts"),
        pl.col("timestamp").last().alias("bar_end_ts"),
        pl.col("price").first().alias("open"),
        pl.col("price").max().alias("high"),
        pl.col("price").min().alias("low"),
        pl.col("price").last().alias("close"),
        pl.col("qty").sum().alias("volume"),
        pl.col("value_usd").sum().alias("dollar_volume"),
        pl.when(pl.col("is_buyer_maker")).then(pl.col("value_usd")).otherwise(0).sum().alias("sell_usd"),
        pl.when(~pl.col("is_buyer_maker")).then(pl.col("value_usd")).otherwise(0).sum().alias("buy_usd"),
        pl.len().alias("tick_count"),
    ]).sort("bar_id").with_columns(pl.lit(threshold).alias("threshold_used"))
    return bars


def build_asset(symbol: str, start: str, end: str) -> pl.DataFrame:
    base = BASE_THRESHOLDS_USD.get(symbol) or _get_yaml_threshold(symbol, "adaptive_vol_base")
    fps = sorted(glob.glob(str(RAW / symbol / "aggTrades" / f"{symbol}-aggTrades-*.parquet")))
    fps_filt = [fp for fp in fps if start <= "-".join(Path(fp).stem.split("-")[-3:]) < end]

    _pl("BUILD", f"{symbol}: base=${base:,}, {len(fps_filt)} days...")

    # Two-pass: first compute daily vol, then use rolling-30d mean as vol_ref
    t0 = time.time()
    daily_vols = []
    for fp in fps_filt:
        try:
            df = pl.read_parquet(Path(fp), columns=["price"])
            daily_vols.append(_daily_log_range_vol(df))
        except Exception:
            daily_vols.append(0.0)
    _pl("OK", f"{symbol}: vol pass done in {time.time()-t0:.0f}s")

    vol_series = pd.Series(daily_vols)
    # Rolling 30d mean (shift-1 for no leakage)
    vol_ref = vol_series.shift(1).rolling(30, min_periods=10).mean().bfill()

    all_bars = []
    for i, fp in enumerate(fps_filt):
        bars = build_day_with_adaptive(Path(fp), base, vol_ref.iloc[i], daily_vols[i])
        if not bars.is_empty():
            # Tag each day's bars with its source date. (Previously NO date column
            # was ever added, so delta-append + the required_cols={"date",...}
            # freshness check forced a full rebuild every run.)
            d = _date_from_aggtrades_path(Path(fp))
            if d is not None:
                ds = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
                bars = bars.with_columns(pl.lit(ds).alias("date"))
            all_bars.append(bars)
        if (i + 1) % 200 == 0:
            _pl("BUILD", f"{symbol}: {i+1}/{len(fps_filt)} in {time.time()-t0:.0f}s")

    if not all_bars:
        return pl.DataFrame()
    return pl.concat(all_bars)


def _build_one_asset(symbol: str, start: str, end: str, out_path: str,
                      write_mode: str) -> dict:
    """ProcessPool worker. write_mode='rebuild' (full) | 'append' (windowed delta)."""
    bars = build_asset(symbol, start, end)
    if bars.is_empty():
        # Append mode + empty = legitimate. Existing rows unchanged.
        is_real_failure = (write_mode == "rebuild")
        return {"status": "empty" if is_real_failure else "ok",
                "symbol": symbol, "n_bars": 0,
                "write_mode": write_mode,
                "note": "no bars produced"}
    # date is now tagged per-day inside build_asset. Merge with existing (append)
    # BEFORE reindexing so bar_id is globally unique across the whole file
    # (CLAUDE.md invariant; was per-day floor() -> duplicate ids across days).
    if write_mode == "append" and Path(out_path).exists() and "date" in bars.columns:
        existing = pl.read_parquet(out_path)
        new_dates = set(bars["date"].to_list())
        keep = existing.filter(~pl.col("date").is_in(list(new_dates)))
        final = pl.concat([keep, bars], how="vertical_relaxed")
    else:
        final = bars
    if "bar_id" in final.columns:
        final = final.drop("bar_id")
    sort_cols = ["date", "bar_start_ts"] if "date" in final.columns else ["bar_start_ts"]
    final = final.sort(sort_cols).with_row_index("bar_id")
    atomic_write_parquet(final, out_path)
    n_bars = len(final)
    return {"status": "ok", "symbol": symbol, "n_bars": n_bars,
            "write_mode": write_mode}


def main():
    ap = argparse.ArgumentParser()
    add_standard_args(ap, default_workers=1)
    ap.add_argument("--burn-from-first-gap", action="store_true",
                    help="On mid-stream gap, rebuild from gap forward.")
    # Phase 7 bidirectional pattern
    ap.add_argument("-r", "--reverse", action="store_true",
                    help="Reverse asset iteration (Z->A) for meet-in-middle "
                         "2x speedup. Run two terminals: one without -r, one with.")
    args = ap.parse_args()

    # Default for adaptive_vol: full u10.
    symbols = resolve_assets(
        args,
        default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
                  "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"],
        stage_name="adaptive_vol")
    from pipeline.bidirectional import iter_assets
    symbols = list(iter_assets(symbols, reverse=args.reverse))
    if args.reverse:
        print(f"[adaptive_vol] REVERSE mode: iterating {len(symbols)} "
              f"assets Z->A", flush=True)

    tasks: list[tuple] = []
    n_skipped = n_appends = n_rebuilds = 0
    for symbol in symbols:
        out = OUT_DIR / f"{symbol}_adaptive_vol.parquet"
        raw_dir = RAW / symbol / "aggTrades"
        raw_fps = []
        if raw_dir.exists():
            raw_fps = [Path(fp) for fp in glob.glob(str(raw_dir / f"{symbol}-aggTrades-*.parquet"))
                        if args.start <= "-".join(Path(fp).stem.split("-")[-3:]) < args.end]
        # Adaptive-vol uses 30-day rolling sigma; window_days=30 ensures the
        # last 30 existing days get re-evaluated alongside new days so the
        # threshold for new bars uses the correct vol baseline.
        delta = delta_state(out, raw_fps, force=args.force,
                             date_from_filename=_date_from_aggtrades_path,
                             burn_from_first_gap=args.burn_from_first_gap,
                             window_days=ADAPTIVE_VOL_WINDOW_DAYS,
                             required_cols={"date", "open", "close", "dollar_volume"},
                             max_null_rate={"close": 0.01, "dollar_volume": 0.01})
        if delta["mode"] == "fresh":
            _pl("SKIP", f"{symbol} SKIP (fresh: {out.name})")
            n_skipped += 1
            continue
        if delta["mode"] == "append":
            n_appends += 1
        else:
            n_rebuilds += 1
        # Note: build_asset reads its own date range from start/end, ignoring
        # delta["new_inputs"]. For windowed delta on adaptive_vol we currently
        # rebuild the full window (build_asset is a single big function); the
        # write step still appends-by-date, which gives correctness via the
        # idempotency of append_parquet. Future: refactor build_asset to take
        # an explicit fps list so we can also save COMPUTE on delta runs.
        tasks.append((symbol, args.start, args.end, str(out), delta["mode"]))

    if args.dry_run:
        print(f"[adaptive_vol] dry-run: {n_appends} appends + {n_rebuilds} rebuilds + "
              f"{n_skipped} fresh-skips")
        return

    if not tasks:
        _pl("SKIP", f"nothing to build ({n_skipped} skipped)")
        return

    run_per_task(tasks, _build_one_asset,
                  workers=args.workers, mode="process",
                  stage_name="adaptive_vol",
                  progress_summary_keys=["write_mode", "n_bars"])


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
