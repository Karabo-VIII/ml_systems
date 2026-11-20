"""ETF flow features — built from Farside BTC+ETH daily net flow data.

Signals produced (per day, one row per day for all BTC assets, similar for ETH):
    etf_total_usdm                daily total net flow (USD millions)
    etf_ibit_usdm                 BlackRock IBIT only (largest issuer, cleanest signal)
    etf_total_z30                 z-score vs rolling 30d baseline (shift-1)
    etf_ibit_z30                  IBIT-specific z-score
    etf_total_7d                  rolling 7d sum (regime indicator)
    etf_total_7d_z                z-score of the 7d-sum
    etf_inflow_shock              1 if etf_total_z30 > +2
    etf_outflow_shock             1 if etf_total_z30 < -2
    etf_mega_inflow               1 if total > +500M (absolute threshold, rare)
    etf_mega_outflow              1 if total < -500M
    etf_consistent_inflow_7d      1 if 7d-sum > +1000M (persistent bull)
    etf_consistent_outflow_7d     1 if 7d-sum < -1000M

Data structure:
    data/frontier/etf/btc_etf_flows.parquet  (date, IBIT, FBTC, ..., Total)
    data/frontier/etf/eth_etf_flows.parquet  (similar for ETH)

Output:
    data/frontier/etf/etf_flow_features.parquet (wide, date + BTC/ETH features)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BTC_IN = ROOT / "data" / "processed" / "panels" / "daily" / "btc_etf_flows.parquet"
ETH_IN = ROOT / "data" / "processed" / "panels" / "daily" / "eth_etf_flows.parquet"
OUT = ROOT / "data" / "processed" / "panels" / "daily" / "etf_flow_features.parquet"


def _build_one(df: pd.DataFrame, prefix: str, ibit_col: str) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True).copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    total_col = "Total" if "Total" in df.columns else [c for c in df.columns if "Total" in c][0]
    df[f"{prefix}_total_usdm"] = df[total_col]
    df[f"{prefix}_ibit_usdm"] = df[ibit_col] if ibit_col in df.columns else np.nan

    # Rolling z-scores
    for src_col, z_col in [
        (f"{prefix}_total_usdm", f"{prefix}_total_z30"),
        (f"{prefix}_ibit_usdm", f"{prefix}_ibit_z30"),
    ]:
        rm = df[src_col].shift(1).rolling(30, min_periods=10).mean()
        rs = df[src_col].shift(1).rolling(30, min_periods=10).std()
        df[z_col] = (df[src_col] - rm) / rs.replace(0, np.nan)

    # Rolling sums
    df[f"{prefix}_total_7d"] = df[f"{prefix}_total_usdm"].rolling(7, min_periods=3).sum()
    df[f"{prefix}_total_14d"] = df[f"{prefix}_total_usdm"].rolling(14, min_periods=5).sum()
    rm7 = df[f"{prefix}_total_7d"].shift(1).rolling(30, min_periods=10).mean()
    rs7 = df[f"{prefix}_total_7d"].shift(1).rolling(30, min_periods=10).std()
    df[f"{prefix}_total_7d_z"] = (df[f"{prefix}_total_7d"] - rm7) / rs7.replace(0, np.nan)

    # Event signals
    df[f"{prefix}_inflow_shock"] = (df[f"{prefix}_total_z30"] > 2.0).astype(int)
    df[f"{prefix}_outflow_shock"] = (df[f"{prefix}_total_z30"] < -2.0).astype(int)
    df[f"{prefix}_mega_inflow"] = (df[f"{prefix}_total_usdm"] > 500).astype(int)
    df[f"{prefix}_mega_outflow"] = (df[f"{prefix}_total_usdm"] < -500).astype(int)
    df[f"{prefix}_consistent_inflow_7d"] = (df[f"{prefix}_total_7d"] > 1000).astype(int)
    df[f"{prefix}_consistent_outflow_7d"] = (df[f"{prefix}_total_7d"] < -1000).astype(int)

    keep = ["date"] + [c for c in df.columns if c.startswith(prefix + "_")]
    return df[keep]


def build() -> pd.DataFrame:
    btc = pd.read_parquet(BTC_IN)
    eth = pd.read_parquet(ETH_IN)

    # Find the ETH IBIT equivalent (ETHA is BlackRock's ETH ETF)
    eth_blackrock_col = None
    for c in eth.columns:
        if "Blackrock" in c and "ETHA" in c:
            eth_blackrock_col = c
            break

    btc_feats = _build_one(btc, "btc_etf", "IBIT")
    eth_feats = _build_one(eth, "eth_etf", eth_blackrock_col or "ETHA")

    merged = btc_feats.merge(eth_feats, on="date", how="outer").sort_values("date").reset_index(drop=True)
    merged["date"] = pd.to_datetime(merged["date"])

    # Combined signals
    if "btc_etf_total_z30" in merged.columns and "eth_etf_total_z30" in merged.columns:
        merged["any_inflow_shock"] = ((merged["btc_etf_inflow_shock"].fillna(0) == 1) |
                                       (merged["eth_etf_inflow_shock"].fillna(0) == 1)).astype(int)
        merged["both_inflow_shock"] = ((merged["btc_etf_inflow_shock"].fillna(0) == 1) &
                                        (merged["eth_etf_inflow_shock"].fillna(0) == 1)).astype(int)

    return merged


def main():
    df = build()
    df.to_parquet(OUT, index=False)
    print(f"[etf_feat] saved: {OUT}")
    print(f"  rows: {len(df)}, cols: {len(df.columns)}")
    print(f"  date range: {df['date'].min().date()} -> {df['date'].max().date()}")

    print("\n[signal incidence]:")
    for s in ["btc_etf_inflow_shock", "btc_etf_outflow_shock", "btc_etf_mega_inflow",
              "btc_etf_mega_outflow", "btc_etf_consistent_inflow_7d",
              "eth_etf_inflow_shock", "eth_etf_outflow_shock",
              "any_inflow_shock", "both_inflow_shock"]:
        if s in df.columns:
            n = int(df[s].fillna(0).sum())
            frac = n / (df[s].notna().sum() or 1)
            print(f"  {s}: {n} days ({100*frac:.2f}%)")

    # Sanity print recent shocks
    shocks = df[df["btc_etf_inflow_shock"] == 1].tail(15)
    print(f"\n[recent BTC ETF inflow shocks]:")
    print(shocks[["date", "btc_etf_total_usdm", "btc_etf_total_z30"]].to_string(index=False))


if __name__ == "__main__":
    main()
