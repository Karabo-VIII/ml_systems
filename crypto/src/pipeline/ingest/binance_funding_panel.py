"""Wide funding-rate panel from raw Binance futures funding files -- canonical producer.

Reads data/raw/<ASSET>USDT/funding/*.parquet (already fetched by fetch_all.py; 3
obs/day at 8h interval) and aggregates to a WIDE daily panel (date x asset), one
column per asset named `<asset>_fund` (daily mean funding rate as a fraction).

This is DISTINCT from features/funding_panel_daily.py: that one emits per-asset
long-format `fund_*` features; THIS one emits the wide cross-asset `fp_*` panel
the chimera consumes via the `funding_panel` registry source (wide_per_asset,
pattern ^([a-z0-9_]+)_fund$).

Output (registry source `funding_panel`):
    data/raw_external/binance_futures_panels/funding_panel_daily.parquet

Provenance: restored 2026-05-29 from src/_archive/frontier/ingest/binance_funding_panel.py
(orphaned when src/frontier was archived; chimera `fp_*` features went ~5wk stale).
No API calls -- reads local raw funding (run fetch_all.py first for fresh funding).
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd
import polars as pl

# Path bootstrap: refresh.py runs this as a direct script, so src/ is not on
# sys.path. Mirror the canonical producer pattern (etf_flows.py) before importing
# the pipeline framework. Fixes ModuleNotFoundError on the 2026-05-30 refresh run.
import sys
_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src" / "pipeline"))
sys.path.insert(0, str(_ROOT / "src"))
from pipeline.parquet_io import atomic_write_parquet

__contract__ = {
    "kind": "panel_builder",
    "inputs": ["data/raw/<ASSET>USDT/funding/*.parquet (local; from fetch_all.py)"],
    "outputs": ["data/raw_external/binance_futures_panels/funding_panel_daily.parquet"],
    "invariants": [
        "wide schema: date + one <asset>_fund column per asset",
        "daily mean of 8h funding settlements",
        "atomic_write_via_parquet_io",
    ],
}

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW = PROJECT_ROOT / "data" / "raw"
OUT_PATH = (PROJECT_ROOT / "data" / "raw_external" / "binance_futures_panels"
            / "funding_panel_daily.parquet")


def load_asset_funding(asset_dir: Path) -> pd.DataFrame | None:
    pattern = str(asset_dir / "funding" / "*.parquet")
    if not glob.glob(pattern):
        return None
    try:
        df = pl.scan_parquet(pattern).collect().to_pandas()
    except Exception:
        return None
    if "funding_rate" not in df.columns or len(df) == 0:
        return None
    df["date"] = pd.to_datetime(
        df["timestamp"].apply(lambda _t: _t // 1000 if _t >= 1e15 else _t), unit="ms").dt.normalize()
    daily = df.groupby("date")["funding_rate"].mean().reset_index()
    asset = asset_dir.name.lower().replace("usdt", "")
    return daily.rename(columns={"funding_rate": f"{asset}_fund"})


def build_panel() -> pd.DataFrame:
    asset_dirs = sorted([d for d in RAW.iterdir() if d.is_dir() and (d / "funding").is_dir()])
    panel = None
    ok = 0
    for d in asset_dirs:
        df = load_asset_funding(d)
        if df is None or len(df) < 90:
            continue
        ok += 1
        panel = df if panel is None else panel.merge(df, on="date", how="outer")
    if panel is None:
        return pd.DataFrame()
    panel = panel.sort_values("date").reset_index(drop=True)
    print(f"[funding_panel] {len(panel)} days x {ok} assets", flush=True)
    return panel


def main():
    ap = argparse.ArgumentParser(description="Build wide funding panel -> raw_external/binance_futures_panels/")
    ap.add_argument("--force", action="store_true", help="No-op (always full rebuild from local raw)")
    ap.add_argument("--workers", type=int, default=1, help="No-op; accepted for refresh.py")
    ap.add_argument("--dry-run", action="store_true", help="Build + report, do not write")
    args = ap.parse_args()
    panel = build_panel()
    if panel.empty:
        print("[funding_panel] ERROR: no raw funding found under data/raw/*/funding/ "
              "(run fetch_all.py first); output unchanged", flush=True)
        raise SystemExit(2)
    if args.dry_run:
        print(f"[funding_panel] DRY-RUN: would write {len(panel)} rows x {len(panel.columns)} cols "
              f"to {OUT_PATH}", flush=True)
        return
    atomic_write_parquet(panel, OUT_PATH, required_cols={"date"})
    print(f"[funding_panel] saved: {OUT_PATH} "
          f"({panel['date'].min().date()} -> {panel['date'].max().date()})", flush=True)


if __name__ == "__main__":
    main()
