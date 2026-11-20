"""Per-asset bar-grain bd_* feature: bd_bgf_imbalance_l1.

T2-A SURGICAL Phase 2 (2026-05-24). Of the 3 bd_* features tested in Phase 1
(runs/oracle/bd_bar_grain_phase1_btc_multiperm_RESULT.txt), only
`bd_imbalance_l1` showed clean lift at bar-grain (3.74x vs daily-broadcast).
The deep-book features (`bd_imbalance_l5`, `bd_total_depth_l5`) REGRESSED,
because daily averaging cancels noise that single-snapshot bar-grain
exposes. So this producer is intentionally NARROW: one bar-grain feature.

Mechanism (why l1 wins at bar-grain):
    The ratio depth(-1%)/depth(+1%) at narrow bands captures fast directional
    book pressure that varies on minute timescales. Daily averaging over
    ~2880 snapshots smears the signal across opposing intraday regimes.
    Per-bar resolution preserves the directional signal in its native horizon.

Input:
    data/raw_external/binance_vision/depth_profile/<SYM>USDT/<DATE>.parquet
    Schema: ts_ms, symbol, percentage, depth, notional (~2880 snaps/day @ 30s)

Output:
    data/processed/bar_grain/bd/<SYM>_bgf.parquet
    Schema: timestamp (i64 ms), bd_bgf_imbalance_l1 (f64)

Per parquet_io contract: atomic_write_parquet.
Per bar attach: chimera consumer uses join_asof(strategy='backward',
                tolerance=30min) to pair each dollar bar with the LATEST
                snapshot before the bar's timestamp.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pipeline.parquet_io import atomic_write_parquet
from pipeline.cli import add_standard_args, resolve_assets

INPUT_ROOT = PROJECT_ROOT / "data" / "raw_external" / "binance_vision" / "depth_profile"
OUT_ROOT = PROJECT_ROOT / "data" / "processed" / "bar_grain" / "bd"

__contract__ = {
    "kind": "bar_grain_producer",
    "stage": "bd_bar_grain",
    "inputs": {"upstream": "data/raw_external/binance_vision/depth_profile/<SYM>USDT/<DATE>.parquet"},
    "outputs": {
        "files": "data/processed/bar_grain/bd/<SYM>_bgf.parquet",
        "columns": ["timestamp", "bd_bgf_imbalance_l1"],
    },
    "invariants": {
        "no_lookahead": True,
        "no_silent_overwrite": True,
        "atomic_write": True,
        "bar_grain": True,    # per-snapshot output, not daily-aggregated
    },
}


def _pl(phase, message, **kw):
    try:
        from progress import phase_log
    except ImportError:
        from pipeline.progress import phase_log
    phase_log("bd_bgf", phase, message, **kw)


def _snapshot_imbalance_l1(day: pl.DataFrame) -> Optional[pl.DataFrame]:
    """Pivot one day's snapshots; compute depth(-1%)/depth(+1%) per snapshot.

    Returns (ts_ms, bd_bgf_imbalance_l1) per snapshot, or None if bands missing.
    """
    if day.is_empty():
        return None
    wide = (day
            .group_by(["ts_ms", "percentage"])
            .agg(pl.col("depth").first())
            .pivot(index="ts_ms", on="percentage", values="depth"))
    cols = wide.columns
    def _c(pct):
        for c in cols:
            if c != "ts_ms":
                try:
                    if float(c) == pct:
                        return c
                except (ValueError, TypeError):
                    continue
        return None
    c_neg1, c_pos1 = _c(-1.0), _c(1.0)
    if c_neg1 is None or c_pos1 is None:
        return None
    return wide.select([
        pl.col("ts_ms").cast(pl.Int64).alias("timestamp"),
        (pl.col(c_neg1) / (pl.col(c_pos1) + 1e-9))
            .clip(0.1, 10.0)
            .alias("bd_bgf_imbalance_l1"),
    ])


def build_one_asset(symbol: str, *, force: bool = False) -> dict:
    """Build the per-snapshot bd_bgf_imbalance_l1 panel for one asset."""
    sym = symbol.upper()
    sym_dir = INPUT_ROOT / sym
    if not sym_dir.exists():
        return {"asset": sym, "status": "no_raw_dir", "rows": 0}
    files = sorted(sym_dir.glob("*.parquet"))
    if not files:
        return {"asset": sym, "status": "no_raw_files", "rows": 0}

    out_path = OUT_ROOT / f"{sym}_bgf.parquet"
    if out_path.exists() and not force:
        return {"asset": sym, "status": "skip_fresh", "rows": 0,
                "out_path": str(out_path)}

    pieces = []
    for f in files:
        try:
            day = pl.read_parquet(f)
        except Exception as e:
            _pl("WARN", f"{sym} read failed {f.name}: {e}")
            continue
        feats = _snapshot_imbalance_l1(day)
        if feats is None or feats.is_empty():
            continue
        pieces.append(feats)
    if not pieces:
        return {"asset": sym, "status": "no_valid_snapshots", "rows": 0}

    panel = pl.concat(pieces, how="diagonal_relaxed").sort("timestamp")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(panel, out_path,
                          required_cols=["timestamp", "bd_bgf_imbalance_l1"])
    return {"asset": sym, "status": "ok", "rows": panel.height,
            "out_path": str(out_path)}


def main():
    ap = argparse.ArgumentParser(description="bd_bgf_imbalance_l1 per-asset bar-grain producer")
    add_standard_args(ap, default_workers=2, date_window=False)
    args = ap.parse_args()
    syms = resolve_assets(args, stage_name="bd_bgf")
    print(f"[bd_bgf] building for {len(syms)} assets, force={args.force}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    n_ok = n_skip = n_err = 0
    total_rows = 0
    t0 = time.time()
    for i, sym in enumerate(syms):
        r = build_one_asset(sym, force=args.force)
        if r["status"] == "ok":
            n_ok += 1
            total_rows += r["rows"]
            print(f"  [{i+1}/{len(syms)}] {sym:>10s} OK rows={r['rows']:,}")
        elif r["status"] == "skip_fresh":
            n_skip += 1
        else:
            n_err += 1
            print(f"  [{i+1}/{len(syms)}] {sym:>10s} {r['status'].upper()}")
    elapsed = time.time() - t0
    print(f"\n[bd_bgf] DONE  ok={n_ok}  skip={n_skip}  err={n_err}  "
          f"rows={total_rows:,}  elapsed={elapsed/60:.1f}min")
    if n_ok == 0 and n_skip == 0:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
