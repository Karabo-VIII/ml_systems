"""Stablecoin mint/flow features — computed from DeFiLlama daily supply.

Produces:
    data/frontier/defillama/stable_flow_features.parquet
        date (datetime), total_usd, usdt_usd, ...  (raw)
        total_delta_1d_pct, total_delta_7d_pct  (day-over-day % change)
        usdt_delta_1d_usd (USD-absolute net mint)
        usdt_zscore_30d  (z-score of daily delta vs 30d baseline)
        stable_zscore_30d (on aggregate delta)
        stable_shock  (1 if total_zscore_30d > +2)
        stable_crash  (1 if total_zscore_30d < -2)

Design notes:
    - Absolute USD delta (not % delta) is what the Griffin & Shams literature uses
      for predictive regressions. A $5B USDT mint day is rare (historically ~2%
      of days) and tends to precede BTC rallies 48-72h later.
    - Z-score over rolling 30 days normalizes the recent regime. In a fast-growing
      stablecoin era (2020-2021), $2B/day was routine; in 2022-2023 the baseline
      was $200M/day. Z-score against recent baseline catches regime changes.
    - We compute on both aggregate (lower noise, market-wide signal) and USDT
      (higher specificity for BTC/alt rally signal per literature).

Run:
    python src/frontier/features/stable_flow_signals.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "data" / "processed" / "panels" / "daily" / "stable_flows_daily.parquet"
OUT = ROOT / "data" / "processed" / "panels" / "daily" / "stable_flow_features.parquet"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True).copy()

    for col in ["total_usd", "usdt_usd", "usdc_usd", "usde_usd", "dai_usd"]:
        if col in df.columns:
            base = col.replace("_usd", "")
            df[f"{base}_delta_1d_usd"] = df[col].diff()
            df[f"{base}_delta_1d_pct"] = df[col].pct_change()
            df[f"{base}_delta_7d_pct"] = df[col].pct_change(7)
            df[f"{base}_delta_30d_pct"] = df[col].pct_change(30)
            delta = df[f"{base}_delta_1d_usd"]
            # z-score vs rolling 30d baseline (shifted to avoid leakage)
            rolling_mean = delta.shift(1).rolling(30, min_periods=10).mean()
            rolling_std = delta.shift(1).rolling(30, min_periods=10).std()
            df[f"{base}_zscore_30d"] = (delta - rolling_mean) / rolling_std.replace(0, np.nan)
            # Log-transformed absolute delta (for regression features)
            df[f"{base}_delta_1d_logusd"] = np.sign(delta) * np.log1p(np.abs(delta))

    # Composite signals (use aggregate z-score as primary; USDT as secondary)
    if "total_zscore_30d" in df.columns:
        df["stable_shock"] = (df["total_zscore_30d"] > 2.0).astype(int)
        df["stable_crash"] = (df["total_zscore_30d"] < -2.0).astype(int)
        df["stable_shock_strong"] = (df["total_zscore_30d"] > 3.0).astype(int)
        df["stable_pos_regime"] = (df["total_delta_7d_pct"] > 0.005).astype(int)  # +0.5%/7d growth

    if "usdt_zscore_30d" in df.columns:
        df["usdt_shock"] = (df["usdt_zscore_30d"] > 2.0).astype(int)
        df["usdt_shock_strong"] = (df["usdt_zscore_30d"] > 3.0).astype(int)

    # Compound event: both aggregate + USDT both fire (highest conviction)
    if "stable_shock" in df.columns and "usdt_shock" in df.columns:
        df["compound_shock"] = ((df["stable_shock"] == 1) & (df["usdt_shock"] == 1)).astype(int)

    return df


def main():
    df = pd.read_parquet(SRC)
    feats = build_features(df)
    feats.to_parquet(OUT, index=False)
    print(f"[features] saved: {OUT} ({len(feats)} rows, {len(feats.columns)} cols)")

    # Empirical distribution of signals
    for signal in ["stable_shock", "stable_shock_strong", "usdt_shock", "compound_shock"]:
        if signal in feats.columns:
            n = int(feats[signal].sum())
            pct = 100 * n / len(feats)
            print(f"  {signal}: {n} days ({pct:.2f}%)")

    # Show high-signal dates in 2024-2026
    recent = feats[(feats["date"] >= "2024-01-01") & (feats["date"] < "2026-04-22")]
    shocks = recent[recent["stable_shock"] == 1]
    print(f"\n[stable_shock events 2024-2026]: {len(shocks)}")
    print(shocks[["date", "total_zscore_30d", "total_delta_1d_usd"]].tail(20).to_string(index=False))


if __name__ == "__main__":
    main()
