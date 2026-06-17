"""Daily funding-rate panel from raw Binance futures funding files.

Reads data/raw/<ASSET>/funding/*.parquet (3 obs/day, 8h interval) and aggregates
to daily mean funding rate per asset. Produces a wide panel (date x asset).

Output:
    data/frontier/funding/funding_panel_daily.parquet
        columns: date + one column per asset (e.g., btc_fund, eth_fund, ...)
        values: daily mean funding rate (as fraction, e.g., 0.00005 = 0.005%/8h)

Usage:
    python src/frontier/ingest/binance_funding_panel.py

Notes:
    - Data is ALREADY on disk (no API calls). Funding fetched via data-pipeline
      monthly cron.
    - Universe: all assets in data/raw/ that have a funding/ subdir with data.
    - Funding is a perp-market signal. We use it for SPOT trades (our constraint)
      via the funding-divergence overlay.
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed" / "panels" / "daily"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "funding_panel_daily.parquet"


def load_asset_funding(asset_dir: Path) -> pd.DataFrame | None:
    """Load all funding parquet files for an asset via polars glob (vectorized)."""
    pattern = str(asset_dir / "funding" / "*.parquet")
    fps = glob.glob(pattern)
    if not fps:
        return None
    try:
        # polars lazy scan across all per-day files at once — ~100x faster than per-file loop
        df = pl.scan_parquet(pattern).collect().to_pandas()
    except Exception:
        return None
    if "funding_rate" not in df.columns or len(df) == 0:
        return None
    df["date"] = pd.to_datetime(df["timestamp"].apply(lambda _t: _t // 1000 if _t >= 1e15 else _t), unit="ms").dt.normalize()
    daily = df.groupby("date")["funding_rate"].mean().reset_index()
    asset = asset_dir.name.lower().replace("usdt", "")
    daily = daily.rename(columns={"funding_rate": f"{asset}_fund"})
    return daily


def build_panel() -> pd.DataFrame:
    asset_dirs = sorted([d for d in RAW.iterdir() if d.is_dir() and (d / "funding").is_dir()])
    panel = None
    ok = 0
    for d in asset_dirs:
        df = load_asset_funding(d)
        if df is None or len(df) < 90:
            continue
        ok += 1
        col = [c for c in df.columns if c != "date"][0]
        latest = df[col].iloc[-1]
        print(f"  {d.name}: {len(df)} days, latest {latest:+.6f}/8h ({latest*3*365*100:+.1f}% annualized)")
        panel = df if panel is None else panel.merge(df, on="date", how="outer")
    panel = panel.sort_values("date").reset_index(drop=True)
    print(f"\n[funding] panel assembled: {len(panel)} days × {ok} assets")
    return panel


def main():
    panel = build_panel()
    panel.to_parquet(OUT_PATH, index=False)
    print(f"[funding] saved: {OUT_PATH}")
    print(f"[funding] date range: {panel['date'].min().date()} -> {panel['date'].max().date()}")


if __name__ == "__main__":
    main()
