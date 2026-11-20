"""Alt-bar chimera builder -- feature-enrich ANY chart type (L2 mineability).

Today the 184-feature chimera exists only on DOLLAR bars (+ 1d/4h time-resamples).
The other chart types (dib / runs_tick / runs_volume / range / adaptive_vol) are
built as bare OHLC and carry NONE of the daily/bar-grain features, so the strat
and WM layers cannot mine indicators on them under feature conditioning. This
builder closes that gap by producing a feature-enriched chimera for any bar type.

DESIGN (reuses the AUDITED dollar-chimera machinery so look-ahead safety is
inherited -- see CLAUDE.md + DATA_LAYER_CANONICAL_REFERENCE):
  1. load alt bars (BarFabric)           OHLC + buy/sell_usd + tick_count + bar_id
  2. physics.calculate_v50_features()    34 base features + causal forward targets
  3. attach_frontier(silver)             80+ daily features, joined by date with
                                         the +1-DAY LAG (no same-day pub race)
  4. attach_bargrain()                   bar-grain LOB/BD, backward as-of join
  -> data/processed/chimera/<bartype>/<sym>usdt_v51_chimera_<bartype>_<DATE>.parquet

LEAKAGE INVARIANTS (inherited; do NOT bypass):
  - targets: forward returns then tail-drop; NEVER fill_null on targets.
  - frontier join: +1d lag (attach_frontier line ~170) -- a daily value realized
    end-of-day D is only visible to a bar on D+1.
  - bar-grain join: backward as-of (only snapshots BEFORE the bar timestamp).
  - rolling normalization: trailing windows in calculate_v50_features (causal).

SCOPE: per (asset, bar_type). xd_* cross-asset base features are NOT computed here
(they require a cross-asset pass over aligned alt bars -- deferred; the daily
frontier already carries the bulk of cross-asset signal e.g. funding spread).
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "pipeline"))

import sota_shared_logic_v50 as physics          # noqa: E402
import layout as _layout                          # noqa: E402
from bar_fabric import BarFabric                   # noqa: E402
from frontier_consolidator import consolidate_one_asset  # noqa: E402
from feature_registry import FeatureRegistry       # noqa: E402
from parquet_io import atomic_write_parquet         # noqa: E402
import make_dataset as _md                          # noqa: E402 (reuse attach_frontier/attach_bargrain)

__contract__ = {
    "kind": "gold_builder",
    "inputs": ["data/processed/bars/<bartype>/<SYM>_*.parquet", "frontier silver", "bar-grain panels"],
    "outputs": ["data/processed/chimera/<bartype>/<sym>usdt_v51_chimera_<bartype>_<DATE>.parquet"],
    "invariants": [
        "reuses audited calculate_v50_features + attach_frontier(+1d lag) + attach_bargrain",
        "targets causal (forward + tail-drop, never fill_null)",
        "no same-day publication race (frontier +1d lag)",
    ],
}

ALT_BAR_TYPES = ("dib", "runs_tick", "runs_volume", "range", "adaptive_vol")


def _load_all_bars(sym_u: str, bar_type: str) -> pl.DataFrame | None:
    """Concatenate ALL partition files for (asset, bar_type), not just the latest.

    Alt bars are year-partitioned (<SYM>_<tag>_<year>.parquet); BarFabric.load
    resolves only one file, which would truncate history to a single year. Mining
    needs the full series, so glob + concat every partition for this asset.
    """
    import glob as _glob
    bdir = _layout.bars_dir(bar_type) if hasattr(_layout, "bars_dir") else \
        (PROJECT_ROOT / "data" / "processed" / "bars" / bar_type)
    fps = sorted(_glob.glob(str(Path(bdir) / f"{sym_u}_*.parquet")))
    if not fps:
        return None
    frames = []
    for fp in fps:
        try:
            frames.append(pl.read_parquet(fp))
        except Exception as e:
            print(f"[chimera_bars] WARN unreadable {Path(fp).name}: {e}", flush=True)
    if not frames:
        return None
    out = pl.concat(frames, how="vertical_relaxed")
    ts = "bar_end_ts" if "bar_end_ts" in out.columns else (
        "timestamp" if "timestamp" in out.columns else "bar_start_ts")
    # de-dup any overlap across partitions, global sort
    return out.unique(subset=[ts], keep="last").sort(ts)


def _prep_bars(bars: pl.DataFrame) -> pl.DataFrame:
    """Map alt-bar columns to what calculate_v50_features expects.

    Alt bars carry bar_start_ts/bar_end_ts + buy_usd/sell_usd; the feature engine
    wants `timestamp` + (optional) buy_vol/sell_vol. Use bar_end_ts as the bar's
    decision timestamp (the bar is only known once complete).
    """
    cols = bars.columns
    ts_src = "bar_end_ts" if "bar_end_ts" in cols else ("timestamp" if "timestamp" in cols else "bar_start_ts")
    bars = bars.with_columns(pl.col(ts_src).alias("timestamp"))
    # date (for the frontier +1d-lag join). timestamp is epoch ms.
    bars = bars.with_columns(
        pl.from_epoch(pl.col("timestamp"), time_unit="ms").dt.date().alias("date"))
    # flow inputs: alt bars expose buy_usd/sell_usd; the ratio-based flow features
    # are scale-free so USD works as a proxy for base-volume buy/sell split.
    ren = {}
    if "buy_usd" in cols and "buy_vol" not in cols:
        ren["buy_usd"] = "buy_vol"
    if "sell_usd" in cols and "sell_vol" not in cols:
        ren["sell_usd"] = "sell_vol"
    if ren:
        bars = bars.rename(ren)
    return bars.sort("timestamp")


def build(sym: str, bar_type: str, write: bool = True) -> pl.DataFrame:
    sym_u = sym.upper() if sym.upper().endswith("USDT") else sym.upper() + "USDT"
    if bar_type not in ALT_BAR_TYPES:
        raise ValueError(f"bar_type must be one of {ALT_BAR_TYPES}, got {bar_type!r}")

    print(f"[chimera_bars] {sym_u}/{bar_type}: loading alt bars...", flush=True)
    bars = _load_all_bars(sym_u, bar_type)
    if bars is None or bars.is_empty():
        print(f"[chimera_bars] {sym_u}/{bar_type}: no alt bars on disk "
              f"(build them first via bars/{bar_type}...)", flush=True)
        raise SystemExit(2)

    bars = _prep_bars(bars)

    # 1. base features + causal targets (audited engine)
    print(f"[chimera_bars] {sym_u}/{bar_type}: {bars.height} bars -> base features...", flush=True)
    chim = physics.calculate_v50_features(bars)

    # 2. frontier silver (build/read per asset) + attach with +1d lag (REUSE)
    reg = FeatureRegistry.load()
    silver = consolidate_one_asset(
        sym_u, reg, forward_fill_max_days=reg.chimera.forward_fill_max_days, out_path=None)
    if silver is not None and not silver.is_empty():
        chim = _md.attach_frontier(chim, silver)
        print(f"[chimera_bars] {sym_u}/{bar_type}: +frontier ({silver.width} silver cols)", flush=True)
    else:
        print(f"[chimera_bars] {sym_u}/{bar_type}: WARN no silver; frontier features absent", flush=True)

    # 3. bar-grain LOB/BD (backward as-of join) (REUSE)
    try:
        chim = _md.attach_bargrain(chim, sym_u)
    except Exception as e:
        print(f"[chimera_bars] {sym_u}/{bar_type}: bargrain skipped ({type(e).__name__}: {e})", flush=True)

    if write:
        out_dir = PROJECT_ROOT / "data" / "processed" / "chimera" / bar_type
        out_dir.mkdir(parents=True, exist_ok=True)
        date_tag = datetime.now(timezone.utc).strftime("%Y%m%d")
        out_path = out_dir / f"{sym_u.lower()}_v51_chimera_{bar_type}_{date_tag}.parquet"
        atomic_write_parquet(chim, out_path, required_cols={"timestamp", "date", "close", "target_return_1"})
        print(f"[chimera_bars] {sym_u}/{bar_type}: wrote {out_path.name} "
              f"({chim.height} rows x {chim.width} cols)", flush=True)
    return chim


def main():
    ap = argparse.ArgumentParser(description="Build feature-enriched chimera on alt bar types")
    ap.add_argument("--assets", nargs="+", required=True, help="e.g. PEPE BTC ETH")
    ap.add_argument("--bar-types", nargs="+", default=["dib"], choices=list(ALT_BAR_TYPES))
    ap.add_argument("--dry-run", action="store_true", help="Build in-memory, do not write")
    args = ap.parse_args()
    n_ok = n_fail = 0
    for sym in args.assets:
        for bt in args.bar_types:
            try:
                build(sym, bt, write=not args.dry_run)
                n_ok += 1
            except SystemExit:
                n_fail += 1
            except Exception as e:
                print(f"[chimera_bars] {sym}/{bt} FAILED: {type(e).__name__}: {e}", flush=True)
                n_fail += 1
    print(f"[chimera_bars] done: {n_ok} ok / {n_fail} fail", flush=True)
    if n_ok == 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
