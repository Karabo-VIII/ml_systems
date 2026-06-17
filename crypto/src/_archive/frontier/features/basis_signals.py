"""Spot-perp basis features.

basis_pct = (perp_close - spot_close) / spot_close × 100
Positive = contango (perp premium, bullish positioning)
Negative = backwardation (perp discount, stress/panic)

Features per asset per day:
    spot_close, perp_close, basis_pct
    basis_z30             per-asset z-score vs 30d rolling baseline
    basis_xsec_z          cross-sectional z per day
    basis_delta_1d        d/dt change
    basis_bull_shock      basis_z > +2 (extreme contango, potential reversion down)
    basis_bear_shock      basis_z < -2 (extreme backwardation, panic → buy spot)
    basis_panic           raw basis < -0.5% (absolute stress threshold)
    basis_frenzy          raw basis > +1.0% (absolute overextension)
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
SPOT_IN = ROOT / "data" / "processed" / "panels" / "daily" / "spot_klines_daily.parquet"
PERP_DATA = ROOT / "data" / "processed"
OUT = ROOT / "data" / "processed" / "panels" / "daily" / "basis_features_long.parquet"


def load_perp_daily() -> pd.DataFrame:
    fps = sorted(glob.glob(str(PERP_DATA / "*_chimera.parquet")))
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
        d = d.rename(columns={"close": "perp_close"})
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def build() -> pd.DataFrame:
    spot = pd.read_parquet(SPOT_IN)
    spot["date"] = pd.to_datetime(spot["date"])
    perp = load_perp_daily()
    perp["date"] = pd.to_datetime(perp["date"])

    df = spot.merge(perp, on=["date", "asset"], how="inner")
    df["basis_pct"] = (df["perp_close"] - df["spot_close"]) / df["spot_close"] * 100
    df = df.sort_values(["asset", "date"]).reset_index(drop=True)

    # Per-asset rolling z + delta
    g = df.groupby("asset")["basis_pct"]
    rm = g.transform(lambda s: s.shift(1).rolling(30, min_periods=10).mean())
    rs = g.transform(lambda s: s.shift(1).rolling(30, min_periods=10).std())
    df["basis_z30"] = (df["basis_pct"] - rm) / rs.replace(0, np.nan)
    df["basis_delta_1d"] = g.diff()
    df["basis_delta_3d"] = g.diff(3)

    # Cross-sectional z per day
    df["basis_xsec_z"] = df.groupby("date")["basis_pct"].transform(
        lambda s: (s - s.mean()) / (s.std() if s.std() > 0 else 1.0)
    )
    df["basis_xsec_rank"] = df.groupby("date")["basis_pct"].rank(pct=True)

    # Event flags
    df["basis_bull_shock"] = (df["basis_z30"] > 2.0).astype(int)
    df["basis_bear_shock"] = (df["basis_z30"] < -2.0).astype(int)
    df["basis_panic"] = (df["basis_pct"] < -0.5).astype(int)   # strong backwardation
    df["basis_frenzy"] = (df["basis_pct"] > 1.0).astype(int)    # strong contango
    df["basis_flip_bull"] = ((df["basis_delta_1d"] > 0.3) & (df["basis_pct"] > 0)).astype(int)
    df["basis_flip_bear"] = ((df["basis_delta_1d"] < -0.3) & (df["basis_pct"] < 0)).astype(int)

    return df


def main():
    df = build()
    df.to_parquet(OUT, index=False)
    print(f"[basis_feat] saved: {OUT} ({len(df)} rows, {df['asset'].nunique()} assets)")
    print(f"  date range: {df['date'].min().date()} -> {df['date'].max().date()}")

    # Sample basis distribution per asset
    print("\n[basis_pct distribution per asset]:")
    for a in df["asset"].unique():
        s = df[df["asset"] == a]["basis_pct"]
        print(f"  {a}: mean {s.mean():+.3f}% / median {s.median():+.3f}% / "
              f"p10 {s.quantile(0.1):+.3f}% / p90 {s.quantile(0.9):+.3f}%")

    print("\n[signal incidence]:")
    for s in ["basis_bull_shock", "basis_bear_shock", "basis_panic", "basis_frenzy",
              "basis_flip_bull", "basis_flip_bear"]:
        if s in df.columns:
            n = int(df[s].fillna(0).sum())
            print(f"  {s}: {n} asset-days ({100*n/len(df):.2f}%)")


if __name__ == "__main__":
    main()
