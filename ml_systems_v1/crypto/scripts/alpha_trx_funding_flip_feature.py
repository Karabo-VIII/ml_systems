"""Alpha turn-020: Q10 debt clearance -- TRX funding-flip as xsec feature.

Produces a per-asset daily feature panel `funding_flip_z` for xsec
feature ingestion. Output goes to data/frontier/funding/funding_flip_feature_daily.parquet
with schema: (date, asset, funding_rate, fund_z30, flip_neg_today,
days_since_flip_neg, fund_z30_signed).

On next xsec retrain, include this parquet as an additional feature source
via src/frontier/ingest/ pattern. Until retrain, pure shelf feature.

Scope: full 45 U50 assets with funding data (5 missing funding: PEPE, SHIB,
1000SATS, BOME, BONK).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.strategy.universe import UNIVERSE_50_LIQUID

OUT = ROOT / "data" / "frontier" / "funding" / "funding_flip_feature_daily.parquet"


def compute_feature(df: pd.DataFrame, asset: str, fund_col: str) -> pd.DataFrame:
    """Produce per-date feature row for one asset."""
    d = df[["date", fund_col]].copy().dropna(subset=[fund_col]).sort_values("date").reset_index(drop=True)
    if len(d) < 30:
        return pd.DataFrame()
    d = d.rename(columns={fund_col: "fund"})
    d["asset"] = asset
    d["fund_mean30"] = d["fund"].rolling(30, min_periods=10).mean()
    d["fund_std30"] = d["fund"].rolling(30, min_periods=10).std()
    d["fund_z30"] = (d["fund"] - d["fund_mean30"]) / d["fund_std30"]
    d["fund_prev"] = d["fund"].shift(1)
    d["flip_neg_today"] = ((d["fund"] < 0) & (d["fund_prev"] >= 0)).astype(int)
    # Days since last flip-to-negative (forward-fill)
    d["_flip_date"] = d.loc[d["flip_neg_today"] == 1, "date"]
    d["_flip_date"] = d["_flip_date"].ffill()
    d["days_since_flip_neg"] = (d["date"] - d["_flip_date"]).dt.days.fillna(999).clip(upper=999)
    # Signed z-score (symmetric feature)
    d["fund_z30_signed"] = d["fund_z30"]
    return d[["date", "asset", "fund", "fund_z30", "flip_neg_today",
              "days_since_flip_neg", "fund_z30_signed"]]


def main() -> None:
    fund = pd.read_parquet(ROOT / "data" / "frontier" / "funding" / "funding_panel_daily.parquet")
    fund["date"] = pd.to_datetime(fund["date"]).dt.normalize()
    panel_rows = []
    processed, missing = [], []
    for asset in UNIVERSE_50_LIQUID:
        col = f"{asset.lower()}_fund"
        if col not in fund.columns:
            missing.append(asset)
            continue
        feat = compute_feature(fund, asset, col)
        if feat.empty:
            missing.append(f"{asset}(thin)")
            continue
        panel_rows.append(feat)
        processed.append(asset)

    if not panel_rows:
        print("[ERR] no data produced")
        return
    wide = pd.concat(panel_rows, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(OUT)
    print(f"[OK] wrote {OUT}")
    print(f"shape: {wide.shape}")
    print(f"processed: {len(processed)} assets -> {processed}")
    print(f"missing:   {len(missing)} -> {missing}")
    # Smoke: show TRX (the E3 survivor) stats
    trx = wide[wide["asset"] == "TRX"]
    if len(trx) > 0:
        n_flips = int(trx["flip_neg_today"].sum())
        print(f"\nTRX sanity: {len(trx)} daily rows, {n_flips} flip-to-negative events")
        print(trx.tail(5)[["date", "fund", "fund_z30", "flip_neg_today"]].to_string(index=False))


if __name__ == "__main__":
    main()
