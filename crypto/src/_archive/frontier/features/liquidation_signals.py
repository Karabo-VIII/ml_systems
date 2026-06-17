"""Liquidation approximation features (from aggTrades-derived daily aggregates).

Features per asset per day:
    raw: liq_long_usd, liq_short_usd, liq_long_count, liq_short_count,
         liq_delta_usd, liq_total_usd

    rolling z-scores (30d, shift-1, no-leak):
        liq_long_z30, liq_short_z30, liq_delta_z30, liq_total_z30

    cross-sectional per day:
        liq_long_xsec_z, liq_short_xsec_z, liq_delta_xsec_z

    event flags:
        liq_long_spike      z_long > +2       (potential capitulation)
        liq_short_spike     z_short > +2      (potential top/squeeze)
        liq_capitulation    z_long > +3 AND ret_1d < -3%  (double-confirmed)
        liq_short_panic     z_short > +3 AND ret_1d > +3% (double-confirmed squeeze)
        liq_delta_bull      delta_z > +2     (net short-liq > long-liq, bullish)
        liq_delta_bear      delta_z < -2     (net long-liq > short-liq, bearish)

Join with price panel for ret_1d to build the capitulation flags.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
LIQ_PANEL = ROOT / "data" / "processed" / "panels" / "daily" / "liq_daily_approx.parquet"
DATA = ROOT / "data" / "processed"
OUT = ROOT / "data" / "processed" / "panels" / "daily" / "liq_features_long.parquet"


def build_price_rets() -> pd.DataFrame:
    fps = sorted(glob.glob(str(DATA / "*_chimera.parquet")))
    rows = []
    for fp in fps:
        try:
            df = pl.read_parquet(fp, columns=["timestamp", "close"]).to_pandas()
        except Exception:
            continue
        if len(df) < 500:
            continue
        df["date"] = pd.to_datetime(df["timestamp"].apply(lambda _t: _t // 1000 if _t >= 1e15 else _t), unit="ms").dt.normalize()
        d = df.groupby("date").agg({"close": "last"}).reset_index()
        d["asset"] = Path(fp).stem.replace("usdt_v50_chimera", "").upper()
        d["ret_1d"] = d["close"].pct_change()
        rows.append(d)
    return pd.concat(rows, ignore_index=True)[["date", "asset", "ret_1d"]]


def build() -> pd.DataFrame:
    liq = pd.read_parquet(LIQ_PANEL)
    liq["date"] = pd.to_datetime(liq["date"])
    liq = liq.sort_values(["asset", "date"]).reset_index(drop=True)

    # Per-asset z-scores
    for col in ["liq_long_usd", "liq_short_usd", "liq_delta_usd", "liq_total_usd"]:
        g = liq.groupby("asset")[col]
        rm = g.transform(lambda s: s.shift(1).rolling(30, min_periods=10).mean())
        rs = g.transform(lambda s: s.shift(1).rolling(30, min_periods=10).std())
        z_col = col.replace("_usd", "_z30")
        liq[z_col] = (liq[col] - rm) / rs.replace(0, np.nan)

    # Cross-sectional z per day
    for col in ["liq_long_usd", "liq_short_usd", "liq_delta_usd"]:
        liq[f"{col.replace('_usd', '_xsec_z')}"] = liq.groupby("date")[col].transform(
            lambda s: (s - s.mean()) / (s.std() if s.std() > 0 else 1.0)
        )

    # Merge with price returns
    rets = build_price_rets()
    liq = liq.merge(rets, on=["date", "asset"], how="left")

    # Event flags
    liq["liq_long_spike"] = (liq["liq_long_z30"] > 2).astype(int)
    liq["liq_short_spike"] = (liq["liq_short_z30"] > 2).astype(int)
    liq["liq_capitulation"] = ((liq["liq_long_z30"] > 3) & (liq["ret_1d"] < -0.03)).astype(int)
    liq["liq_short_panic"] = ((liq["liq_short_z30"] > 3) & (liq["ret_1d"] > 0.03)).astype(int)
    liq["liq_delta_bull"] = (liq["liq_delta_z30"] > 2).astype(int)
    liq["liq_delta_bear"] = (liq["liq_delta_z30"] < -2).astype(int)

    return liq


def main():
    df = build()
    df.to_parquet(OUT, index=False)
    print(f"[liq_feat] saved: {OUT} ({len(df)} rows, {len(df.columns)} cols)")
    print(f"  assets: {df['asset'].nunique()}, dates: {df['date'].nunique()}")
    print(f"  date range: {df['date'].min().date()} -> {df['date'].max().date()}")

    print("\n[signal incidence]:")
    for s in ["liq_long_spike", "liq_short_spike", "liq_capitulation",
              "liq_short_panic", "liq_delta_bull", "liq_delta_bear"]:
        if s in df.columns:
            n = int(df[s].fillna(0).sum())
            frac = 100 * n / (df[s].notna().sum() or 1)
            print(f"  {s}: {n} asset-days ({frac:.2f}%)")

    # Sample capitulation events
    cap = df[df["liq_capitulation"] == 1].tail(10)
    print(f"\n[recent capitulations]:")
    print(cap[["date", "asset", "liq_long_usd", "liq_long_z30", "ret_1d"]].to_string(index=False))


if __name__ == "__main__":
    main()
