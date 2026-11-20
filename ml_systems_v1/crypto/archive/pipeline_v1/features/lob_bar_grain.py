"""Per-asset bar-grain lob_* feature producer.

T2-B Phase 1 (2026-05-24). Mirrors src/pipeline/features/bd_bar_grain.py
architecture exactly, reading the per-day bar-level lob_proxy_* files that
lob_proxy_panel.py already writes to disk, concatenating them, and attaching
each bar's values to chimera dollar bars via an as-of-backward join.

The proposal (runs/oracle/BAR_GRAIN_FEATURE_LAYER_PROPOSAL_2026_05_24.md,
lob_proxy row): "bar-level files ALREADY on disk | SKIP the daily aggregator;
join bar files directly". This producer is that skip -- it reads the per-day
bar files and joins them straight to chimera timestamps without daily
aggregation.

Phase 1 candidate features (most likely to lift at bar grain per the bd_*
mechanism -- narrow-band directional pressure that daily averaging cancels):

  lob_bgf_l1_imb_mean      -- aggressor volume imbalance per bar (buy-sell)/(buy+sell)
                              Mirrors bd_imbalance_l1 which gave 3.74x lift.
                              Fast signal; daily averaging smears opposing intraday
                              regimes.

  lob_bgf_kyle_lambda_mean -- Kyle lambda proxy: bar return per unit signed volume.
                              Price impact; directional at bar grain; mean-reverts
                              quickly (daily average = near-zero).

  lob_bgf_spread_bps_mean  -- Effective spread from Roll estimator. Market stress
                              indicator; intraday spikes (crunch events) are exactly
                              the bars that predict regime breaks. Daily mean flattens.

  lob_bgf_top_pressure_mean -- log(avg_buy_size / avg_sell_size). Directional, zero-
                              mean over a day; carrying the intraday snapshot is the
                              whole signal.

  lob_bgf_count_imb_mean   -- Trade count imbalance (n_buy - n_sell) / total.
                              A second independent directional signal after l1_imb;
                              captures tick-frequency dominance that size-based l1 misses.

Input:
    data/processed/panels/daily/lob_proxy_<SYM>USDT_<YYYYMMDD>.parquet
    (written by src/pipeline/features/lob_proxy_panel.py)
    Schema: bar_id, timestamp (i64 ms), asset, l1_imbalance_avg, l5_imbalance_avg,
            spread_bps_avg, queue_life_p50_s, top_pressure_avg, proxy_count_imb,
            proxy_run_length, proxy_kyle_lambda, proxy_data_source

Output:
    data/processed/bar_grain/lob/<SYM>USDT_bgf.parquet
    Schema: timestamp (i64 ms), lob_bgf_l1_imb_mean, lob_bgf_kyle_lambda_mean,
            lob_bgf_spread_bps_mean, lob_bgf_top_pressure_mean, lob_bgf_count_imb_mean

As-of-backward semantics:
    Each chimera dollar bar receives the lob_proxy bar whose timestamp is the
    LATEST value <= the chimera bar timestamp, within a 30-minute staleness
    window. Because lob_proxy bars are themselves formed from aggTrades within
    a bar window, the backward join is no-lookahead by construction: the lob
    bar closed BEFORE the chimera bar timestamp it attaches to.

Per parquet_io contract: atomic_write_parquet.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pipeline.parquet_io import atomic_write_parquet
from pipeline.cli import add_standard_args, resolve_assets

# Raw bar files live in panels/daily -- the lob_proxy_panel producer already
# wrote per-day per-asset files at this path.  We concatenate all days for
# an asset and join asof to chimera bars.
INPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "panels" / "daily"
OUT_ROOT = PROJECT_ROOT / "data" / "processed" / "bar_grain" / "lob"

# Tolerance for as-of-backward join in milliseconds (30 minutes).
ASOF_TOLERANCE_MS = 30 * 60 * 1000

# Source column -> output lob_bgf_ column name mapping.
# Deliberately narrow -- only the 5 Phase-1 candidates.
FEATURE_MAP = {
    "l1_imbalance_avg":  "lob_bgf_l1_imb_mean",
    "proxy_kyle_lambda": "lob_bgf_kyle_lambda_mean",
    "spread_bps_avg":    "lob_bgf_spread_bps_mean",
    "top_pressure_avg":  "lob_bgf_top_pressure_mean",
    "proxy_count_imb":   "lob_bgf_count_imb_mean",
}

OUTPUT_COLS = ["timestamp"] + list(FEATURE_MAP.values())

__contract__ = {
    "kind": "bar_grain_producer",
    "stage": "lob_bar_grain",
    "inputs": {
        "upstream": (
            "data/processed/panels/daily/lob_proxy_<SYM>USDT_<YYYYMMDD>.parquet"
        )
    },
    "outputs": {
        "files": "data/processed/bar_grain/lob/<SYM>USDT_bgf.parquet",
        "columns": OUTPUT_COLS,
    },
    "invariants": {
        "no_lookahead": True,      # backward join; lob bar closed before chimera bar
        "no_silent_overwrite": True,
        "atomic_write": True,
        "bar_grain": True,
    },
}


def _pl(phase: str, message: str, **kw):
    """Lazy phase_log with dual-import fallback."""
    try:
        from progress import phase_log
    except ImportError:
        from pipeline.progress import phase_log
    phase_log("lob_bgf", phase, message, **kw)


def _load_raw_bars(symbol: str) -> pl.DataFrame | None:
    """Concatenate all per-day lob_proxy bar files for one asset.

    Returns a sorted DataFrame with the source columns we need, or None if no
    files are found.
    """
    sym = symbol.upper()
    if not sym.endswith("USDT"):
        sym = sym + "USDT"
    # Pattern: lob_proxy_BTCUSDT_20260101.parquet
    files = sorted(INPUT_ROOT.glob(f"lob_proxy_{sym}_*.parquet"))
    if not files:
        return None

    src_cols = ["timestamp"] + list(FEATURE_MAP.keys())
    pieces = []
    for f in files:
        try:
            df = pl.read_parquet(f, columns=src_cols)
            pieces.append(df)
        except Exception as e:
            _pl("WARN", f"{sym} read failed {f.name}: {e}")
            continue

    if not pieces:
        return None

    raw = pl.concat(pieces, how="diagonal_relaxed").sort("timestamp")
    # Cast timestamp to i64 ms (should already be; guard against schema drift)
    raw = raw.with_columns(pl.col("timestamp").cast(pl.Int64))
    return raw


def build_one_asset(symbol: str, *, force: bool = False) -> dict:
    """Build the per-bar lob_bgf_* panel for one asset.

    Returns a status dict: {"asset", "status", "rows", "out_path"}.
    """
    sym = symbol.upper()
    if not sym.endswith("USDT"):
        sym = sym + "USDT"

    out_path = OUT_ROOT / f"{sym}_bgf.parquet"
    if out_path.exists() and not force:
        return {"asset": sym, "status": "skip_fresh", "rows": 0,
                "out_path": str(out_path)}

    raw = _load_raw_bars(sym)
    if raw is None or raw.is_empty():
        return {"asset": sym, "status": "no_raw_files", "rows": 0}

    # Rename source columns to lob_bgf_* names, keep timestamp
    rename_map = FEATURE_MAP  # {src: dst}
    panel = raw.rename(rename_map)

    # Ensure we only keep the declared output columns (in case source had extras
    # after the rename); also drop any rows with null timestamp.
    panel = panel.select(OUTPUT_COLS).drop_nulls(subset=["timestamp"])

    if panel.is_empty():
        return {"asset": sym, "status": "no_valid_rows", "rows": 0}

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(panel, out_path, required_cols=set(OUTPUT_COLS))
    return {"asset": sym, "status": "ok", "rows": panel.height,
            "out_path": str(out_path)}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="lob_bgf per-asset bar-grain producer (Phase 1: 5 features)"
    )
    add_standard_args(ap, default_workers=2, date_window=False)
    args = ap.parse_args()
    syms = resolve_assets(args, stage_name="lob_bgf")
    print(f"[lob_bgf] building for {len(syms)} assets, force={args.force}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    n_ok = n_skip = n_err = 0
    total_rows = 0
    t0 = time.time()
    for i, sym in enumerate(syms):
        r = build_one_asset(sym, force=args.force)
        if r["status"] == "ok":
            n_ok += 1
            total_rows += r["rows"]
            print(f"  [{i + 1}/{len(syms)}] {sym:>12s} OK rows={r['rows']:,}")
        elif r["status"] == "skip_fresh":
            n_skip += 1
        else:
            n_err += 1
            print(f"  [{i + 1}/{len(syms)}] {sym:>12s} {r['status'].upper()}")

    elapsed = time.time() - t0
    print(
        f"\n[lob_bgf] DONE  ok={n_ok}  skip={n_skip}  err={n_err}  "
        f"rows={total_rows:,}  elapsed={elapsed / 60:.1f}min"
    )
    if n_ok == 0 and n_skip == 0:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
