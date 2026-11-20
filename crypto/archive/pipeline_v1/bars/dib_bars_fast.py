"""Dollar Imbalance Bars (DIB) -- polars + numba.

Strategy: per trade, accumulate signed dollar volume since the last bar; close
a bar (and reset the accumulator) when |accumulated| crosses the per-asset
threshold (AFML imbalance-bar definition). bar_id is reassigned globally-unique
per asset after concatenating days.

Scope: any universe/date-range via the canonical CLI (--assets/--universe/
--start/--end). Default (no flags) builds BTC + ETH.

Downstream: consumed by features/lob_proxy_panel.py (bar-aligned LOB proxy
features). Other bar types (runs/range/adaptive_vol) currently have no chimera
consumer.
"""
from __future__ import annotations

import argparse
import glob
import sys
import warnings
from pathlib import Path

__contract__ = {
    "kind": "bar_producer",
    "inputs": ["data/raw/<SYM>/aggTrades/*.parquet"],
    "outputs": ["data/processed/bars/dib/<SYM>_dib_<tag>.parquet"],
    "invariants": [
        "ts_ms_13digit",
        "bar_id_globally_unique_per_asset",
        "imbalance_reset_semantics (close+reset on |accum| >= threshold)",
        "atomic_write_via_parquet_io",
    ],
}

import numpy as np
import pandas as pd
import polars as pl

# Framework primitives (post-2026-05-01 unification).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parquet_io import (atomic_write_parquet, delta_state, append_parquet)
from dispatch import run_per_task
from cli import add_standard_args, resolve_assets

# Shared u87 threshold fallback (added 2026-05-12 for alt-bar full-coverage rebuild)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _thresholds import get_threshold as _get_yaml_threshold
from _aggtrades_utils import prepare_aggtrades, imbalance_bar_ids

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"
# Canonical layout v3: data/processed/bars/<bartype>/<sym>_<bartype>_<DATE>.parquet
OUT_DIR = ROOT / "data" / "processed" / "bars" / "dib"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIB_THRESHOLDS_USD = {
    "BTCUSDT": 2_000_000,
    "ETHUSDT": 1_000_000,
    "SOLUSDT": 500_000,
    "BNBUSDT": 500_000,
    "XRPUSDT": 300_000,
    "DOGEUSDT": 200_000,
    "ADAUSDT": 200_000,
    "AVAXUSDT": 200_000,
    "LINKUSDT": 150_000,
    "LTCUSDT": 200_000,
}


# 2026-05-22 oracle pipeline-progress closure: lazy phase_log helper with
# dual-import fallback (works whether src/ or src/pipeline/ is on sys.path).
def _pl(phase, message, **kw):
    try:
        from progress import phase_log
    except ImportError:
        from pipeline.progress import phase_log
    phase_log("dib", phase, message, **kw)


def build_dib_polars(fp: Path, threshold: float) -> pl.DataFrame:
    """Vectorized: load aggTrades, compute cumsum, find bar boundaries."""
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
    df = df.with_columns(
        (pl.col("sign") * pl.col("value_usd")).alias("signed_usd")
    )
    # AFML imbalance-bar boundaries WITH RESET: a bar closes when the signed
    # dollar volume accumulated SINCE THE LAST BAR crosses +/- threshold, then
    # the accumulator resets. (The previous floor(cumulative_sum/threshold) never
    # reset -- it grouped by absolute cumulative level, which is not a DIB and
    # produces wrong bars for mean-reverting flow.)
    bar_ids = imbalance_bar_ids(df["signed_usd"].to_numpy(), threshold)
    df = df.with_columns(pl.Series("bar_id", bar_ids))

    # Aggregate each bar_id group
    bars = df.group_by("bar_id").agg([
        pl.col("timestamp").first().alias("bar_start_ts"),
        pl.col("timestamp").last().alias("bar_end_ts"),
        pl.col("price").first().alias("open"),
        pl.col("price").max().alias("high"),
        pl.col("price").min().alias("low"),
        pl.col("price").last().alias("close"),
        pl.col("qty").sum().alias("volume"),
        pl.col("signed_usd").sum().alias("signed_usd"),
        pl.when(pl.col("is_buyer_maker")).then(pl.col("value_usd")).otherwise(0).sum().alias("sell_usd"),
        pl.when(~pl.col("is_buyer_maker")).then(pl.col("value_usd")).otherwise(0).sum().alias("buy_usd"),
        pl.len().alias("tick_count"),
    ]).sort("bar_id")

    return bars


def _date_from_aggtrades_path(fp: Path) -> "pd.Timestamp.date | None":
    """BTCUSDT-aggTrades-2024-01-01.parquet -> date(2024, 1, 1)."""
    parts = fp.stem.split("-")
    if len(parts) < 5:
        return None
    try:
        return pd.Timestamp("-".join(parts[-3:])).date()
    except Exception:
        return None


def _build_one_asset(symbol: str, threshold: float, fps_to_process: list,
                      out_path: str, mode: str) -> dict:
    """ProcessPool worker: build DIB bars for the given subset of inputs.

    `mode` selects the write path:
        'rebuild' -> full overwrite (atomic_write_parquet)
        'append'  -> read existing, drop overlapping dates, concat, atomic write
    """
    import time
    t0 = time.time()
    all_bars = []
    for fp in fps_to_process:
        bars = build_dib_polars(Path(fp), threshold)
        if not bars.is_empty():
            # Use the tested path-date parser (raw string split on "aggTrades-"
            # is fragile to path/naming changes and was a known bug source).
            d = _date_from_aggtrades_path(Path(fp))
            if d is None:
                continue
            bars = bars.with_columns(pl.lit(d.strftime("%Y-%m-%d")).alias("date"))
            all_bars.append(bars)
    if not all_bars:
        # Append mode + empty = legitimate (low-vol days never crossed
        # the dollar-imbalance threshold). Existing rows unchanged.
        is_real_failure = (mode == "rebuild")
        return {"status": "empty" if is_real_failure else "ok",
                "symbol": symbol, "n_bars": 0,
                "mode": mode,
                "elapsed_s": round(time.time() - t0, 1),
                "note": "no bars produced"}
    new_df = pl.concat(all_bars)
    # Merge with existing (append mode) BEFORE reindexing so bar_id is globally
    # unique across the WHOLE file, not just the new batch (CLAUDE.md invariant:
    # bar_id unique per asset). Consumers join on bar_start_ts/bar_end_ts, not
    # bar_id, so the relabel is safe.
    if mode == "append" and Path(out_path).exists():
        existing = pl.read_parquet(out_path)
        new_dates = set(new_df["date"].to_list())
        keep = existing.filter(~pl.col("date").is_in(list(new_dates)))
        final = pl.concat([keep, new_df], how="vertical_relaxed")
    else:
        final = new_df
    final = (final.sort(["date", "bar_start_ts"])
                  .drop("bar_id")
                  .with_row_index("bar_id"))
    atomic_write_parquet(final, out_path)
    n_bars = len(final)

    return {"status": "ok", "symbol": symbol, "n_bars": n_bars,
            "n_new_days": len(all_bars), "mode": mode,
            "elapsed_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    add_standard_args(ap, default_workers=1)
    ap.add_argument("--burn-from-first-gap", action="store_true",
                    help="On detecting a mid-stream gap (date present in inputs "
                         "but missing from existing output), rebuild every input "
                         "from that date forward instead of just filling the gap. "
                         "Default: pure set-difference fill.")
    # Phase 7 bidirectional pattern (pipeline.bidirectional framework)
    ap.add_argument("-r", "--reverse", action="store_true",
                    help="Reverse asset iteration order (Z->A). For "
                         "meet-in-middle 2x speedup: run two terminals, "
                         "one without -r and one with -r. Per-asset writes "
                         "are independent so no race.")
    args = ap.parse_args()

    # Default for DIB if neither --universe nor --assets given: BTC + ETH only.
    symbols = resolve_assets(args, default=["BTCUSDT", "ETHUSDT"],
                              stage_name="dib")

    # Phase 7 bidirectional: reverse iteration when -r is set so two terminals
    # meet in the middle.
    from pipeline.bidirectional import iter_assets
    symbols = list(iter_assets(symbols, reverse=args.reverse))
    if args.reverse:
        print(f"[dib] REVERSE mode: iterating {len(symbols)} assets Z->A "
              f"(meet-in-middle pattern)", flush=True)

    tasks: list[tuple] = []
    n_skipped = 0
    n_appends = n_rebuilds = 0
    for symbol in symbols:
        threshold = DIB_THRESHOLDS_USD.get(symbol) or _get_yaml_threshold(symbol, "dib")
        fps = sorted(glob.glob(str(RAW / symbol / "aggTrades" / f"{symbol}-aggTrades-*.parquet")))
        fps_filt = [fp for fp in fps
                    if args.start <= "-".join(Path(fp).stem.split("-")[-3:]) < args.end]
        year_tag = args.start[:4]
        out = OUT_DIR / f"{symbol}_dib_{year_tag}.parquet"
        if not fps_filt:
            print(f"[dib] {symbol} no aggTrades in [{args.start}, {args.end}); skip",
                  flush=True)
            continue

        delta = delta_state(
            out, [Path(p) for p in fps_filt],
            force=args.force,
            date_from_filename=_date_from_aggtrades_path,
            burn_from_first_gap=args.burn_from_first_gap,
            # Corruption guards: existing parquet must have these cols
            # populated; if not, fall through to full rebuild.
            required_cols={"date", "open", "close", "signed_usd"},
            max_null_rate={"close": 0.01, "signed_usd": 0.01},
        )
        if delta["mode"] == "fresh":
            _pl("SKIP", f"{symbol} SKIP (fresh: {out.name})")
            n_skipped += 1
            continue
        if delta["mode"] == "append":
            n_appends += 1
        else:
            n_rebuilds += 1
        n_new = len(delta["new_inputs"])
        print(f"[dib] {symbol} {delta['mode']}: threshold=${threshold:,}, "
              f"{n_new} files | {delta['reason'][:90]}", flush=True)
        tasks.append((symbol, threshold,
                       [str(p) for p in delta["new_inputs"]],
                       str(out), delta["mode"]))

    if args.dry_run:
        print(f"[dib] dry-run: {n_appends} appends + {n_rebuilds} rebuilds + "
              f"{n_skipped} fresh-skips")
        return

    if not tasks:
        _pl("SKIP", f"nothing to build ({n_skipped} skipped)")
        return

    run_per_task(tasks, _build_one_asset,
                  workers=args.workers, mode="process",
                  stage_name="dib",
                  progress_summary_keys=["mode", "n_new_days", "n_bars", "elapsed_s"])


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
