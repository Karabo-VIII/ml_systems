"""Cross-sectional funding divergence features.

Hypothesis:
    When perp funding goes extreme NEGATIVE for an asset (shorts over-crowded),
    the short side is paying longs. Any upside catalyst squeezes shorts and
    accelerates a rally. SPOT traders (who pay no funding) can capture this
    squeeze by buying SPOT during the extreme-negative-funding window.

    Conversely, extreme POSITIVE funding = longs over-crowded = reversion DOWN
    is more likely (avoid or do not enter).

Features produced (per asset per day):
    raw_funding   (annualized %, 8h rate × 3 × 365)
    fund_zscore_30d  (z-score vs own rolling 30d baseline)
    fund_xsec_zscore (same-day cross-sectional z-score vs peer universe)
    fund_rank_pct  (0=most negative today, 1=most positive today)
    fund_shorts_crowded  (1 if fund_xsec_zscore < -1.5 OR fund_rank_pct < 0.15)
    fund_longs_crowded   (1 if fund_xsec_zscore > +1.5 OR fund_rank_pct > 0.85)
    fund_extreme_neg  (1 if daily funding < -0.001 per 8h = < -109% annualized)
    fund_extreme_pos  (1 if daily funding > +0.001 per 8h = > +109% annualized)

Output (long format for strategy):
    data/frontier/funding/funding_features_long.parquet
        columns: date, asset, raw_funding, fund_zscore_30d, fund_xsec_zscore,
                 fund_rank_pct, fund_shorts_crowded, fund_longs_crowded,
                 fund_extreme_neg, fund_extreme_pos
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PANEL = ROOT / "data" / "processed" / "panels" / "daily" / "funding_panel_daily.parquet"
OUT = ROOT / "data" / "processed" / "panels" / "daily" / "funding_features_long.parquet"


def build_features() -> pd.DataFrame:
    wide = pd.read_parquet(PANEL)
    wide["date"] = pd.to_datetime(wide["date"])
    asset_cols = [c for c in wide.columns if c.endswith("_fund")]
    print(f"[funding_feat] panel: {wide.shape}, assets: {len(asset_cols)}")

    # Melt to long format
    long = wide.melt(id_vars="date", value_vars=asset_cols, var_name="asset_col",
                     value_name="raw_funding").dropna(subset=["raw_funding"])
    long["asset"] = long["asset_col"].str.replace("_fund", "").str.upper()
    long = long.drop(columns="asset_col").sort_values(["asset", "date"]).reset_index(drop=True)

    # Per-asset rolling z-score vs own 30d baseline
    long["_roll_mean"] = long.groupby("asset")["raw_funding"].transform(
        lambda s: s.shift(1).rolling(30, min_periods=10).mean()
    )
    long["_roll_std"] = long.groupby("asset")["raw_funding"].transform(
        lambda s: s.shift(1).rolling(30, min_periods=10).std()
    )
    long["fund_zscore_30d"] = (long["raw_funding"] - long["_roll_mean"]) / long["_roll_std"].replace(0, np.nan)
    long = long.drop(columns=["_roll_mean", "_roll_std"])

    # Cross-sectional z-score + rank per day
    def _xsec(grp):
        x = grp["raw_funding"].values
        m, s = np.nanmean(x), np.nanstd(x)
        grp["fund_xsec_zscore"] = (x - m) / (s if s > 0 else 1.0)
        ranks = grp["raw_funding"].rank(pct=True)
        grp["fund_rank_pct"] = ranks
        return grp

    long = long.groupby("date", group_keys=False).apply(_xsec)

    # Boolean signals
    long["fund_shorts_crowded"] = (
        (long["fund_xsec_zscore"] < -1.5) | (long["fund_rank_pct"] < 0.15)
    ).astype(int)
    long["fund_longs_crowded"] = (
        (long["fund_xsec_zscore"] > 1.5) | (long["fund_rank_pct"] > 0.85)
    ).astype(int)
    long["fund_extreme_neg"] = (long["raw_funding"] < -0.001).astype(int)
    long["fund_extreme_pos"] = (long["raw_funding"] > +0.001).astype(int)
    long["fund_deep_neg"] = (long["raw_funding"] < -0.0005).astype(int)   # -55% annualized
    long["fund_deep_pos"] = (long["raw_funding"] > +0.0005).astype(int)

    # Annualized funding
    long["raw_funding_ann_pct"] = long["raw_funding"] * 3 * 365 * 100

    return long


def main():
    df = build_features()
    df.to_parquet(OUT, index=False)
    print(f"[funding_feat] saved: {OUT}")
    print(f"  rows: {len(df)}, assets: {df['asset'].nunique()}, dates: {df['date'].nunique()}")

    # Signal incidence
    for s in ["fund_shorts_crowded", "fund_longs_crowded", "fund_extreme_neg",
              "fund_extreme_pos", "fund_deep_neg", "fund_deep_pos"]:
        n = int(df[s].sum())
        print(f"  {s}: {n} asset-days ({100*n/len(df):.2f}%)")

    # Recent (2025+) signal examples
    recent = df[df["date"] >= "2025-01-01"].copy()
    print(f"\n[recent] shorts_crowded events by asset (2025+):")
    sc = recent[recent["fund_shorts_crowded"] == 1].groupby("asset").size().nlargest(15)
    print(sc.to_string())


if __name__ == "__main__":
    main()
