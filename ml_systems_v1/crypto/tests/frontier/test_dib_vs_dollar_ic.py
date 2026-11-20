"""DIB vs Dollar Bars — signal quality comparison at DAILY level.

For each bar type, compute:
    daily_flow_imbalance = sum(signed_usd) / sum(|signed_usd|)  per day
    daily_buy_sell_ratio = sum(buy_usd) / sum(sell_usd)  per day
    daily_total_signed   = sum(signed_usd) per day
    daily_tick_count     = count of bars per day

Then IC vs forward returns (1d, 3d, 5d) for each bar-type-derived feature.
If DIB features have materially higher |IC| or t-stat than dollar-bar features,
DIB is worth the pipeline build.

Universe: BTC + ETH (where DIB is built). 2025 window.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
DIB_DIR = ROOT / "data" / "frontier" / "dib"
CHIMERA = ROOT / "data" / "processed"


def extract_daily_from_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Aggregate bar-level to daily per-asset."""
    bars = bars.copy()
    if "ts_datetime" not in bars.columns:
        # Use bar_end_ts (microseconds for 2025 Binance data? let's check)
        # Binance aggTrades ts is microseconds post-2024 August
        ts = bars["bar_end_ts"].iloc[0]
        unit = "us" if ts > 1e15 else "ms"
        bars["ts_datetime"] = pd.to_datetime(bars["bar_end_ts"], unit=unit)
    bars["date"] = bars["ts_datetime"].dt.normalize()
    daily = bars.groupby("date").agg({
        "signed_usd": "sum",
        "buy_usd": "sum",
        "sell_usd": "sum",
        "volume": "sum",
        "tick_count": "sum",
        "close": "last",
    }).reset_index()
    daily["flow_imbalance"] = daily["signed_usd"] / (daily["buy_usd"] + daily["sell_usd"]).replace(0, np.nan)
    daily["buy_sell_ratio"] = daily["buy_usd"] / daily["sell_usd"].replace(0, np.nan)
    daily["log_signed"] = np.sign(daily["signed_usd"]) * np.log1p(np.abs(daily["signed_usd"]))
    daily["ret_1d"] = daily["close"].pct_change().shift(-1)
    daily["ret_3d"] = daily["close"].pct_change(3).shift(-3)
    daily["ret_5d"] = daily["close"].pct_change(5).shift(-5)
    return daily


def load_dollar_bar_daily(asset: str) -> pd.DataFrame:
    """Load chimera dollar bars, compute daily flow imbalance."""
    fp = CHIMERA / f"{asset.lower()}usdt_v50_chimera.parquet"
    df = pl.read_parquet(fp, columns=[
        "timestamp", "close", "volume_usd", "buy_vol", "sell_vol", "tick_count"
    ]).to_pandas()
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.normalize()
    df = df[df["date"] >= "2025-01-01"].copy()
    df = df[df["date"] < "2026-01-01"]
    # Already approximately in USD at bar level via volume_usd
    df["buy_usd"] = df["buy_vol"] * df["close"]  # rough approximation
    df["sell_usd"] = df["sell_vol"] * df["close"]
    df["signed_usd"] = df["buy_usd"] - df["sell_usd"]

    daily = df.groupby("date").agg({
        "signed_usd": "sum",
        "buy_usd": "sum",
        "sell_usd": "sum",
        "tick_count": "sum",
        "close": "last",
    }).reset_index()
    daily["flow_imbalance"] = daily["signed_usd"] / (daily["buy_usd"] + daily["sell_usd"]).replace(0, np.nan)
    daily["buy_sell_ratio"] = daily["buy_usd"] / daily["sell_usd"].replace(0, np.nan)
    daily["log_signed"] = np.sign(daily["signed_usd"]) * np.log1p(np.abs(daily["signed_usd"]))
    daily["ret_1d"] = daily["close"].pct_change().shift(-1)
    daily["ret_3d"] = daily["close"].pct_change(3).shift(-3)
    daily["ret_5d"] = daily["close"].pct_change(5).shift(-5)
    return daily


def ic_test(df: pd.DataFrame, feat_cols: list[str], ret_cols: list[str], label: str):
    print(f"\n[{label}] n={len(df)} days")
    for feat in feat_cols:
        for rc in ret_cols:
            r = df[[feat, rc]].dropna()
            if len(r) < 10:
                continue
            # Simple pearson + spearman
            pear = r[feat].corr(r[rc], method="pearson")
            spear = r[feat].corr(r[rc], method="spearman")
            # t-stat (pearson, n-2 df)
            n = len(r)
            t = pear * np.sqrt(max(n - 2, 1)) / np.sqrt(max(1 - pear ** 2, 1e-6)) if abs(pear) < 1 else 0
            print(f"  {feat:<25} vs {rc}: pear {pear:+.4f} spear {spear:+.4f} t={t:+.2f}")


def main():
    for asset in ["BTC", "ETH"]:
        print(f"\n{'=' * 60}\n=== {asset} ===")
        # DIB
        dib_fp = DIB_DIR / f"{asset}USDT_dib_2025.parquet"
        dib_df = pl.read_parquet(dib_fp).to_pandas()
        dib_daily = extract_daily_from_bars(dib_df)

        # Dollar bars
        dollar_daily = load_dollar_bar_daily(asset)

        feats = ["flow_imbalance", "buy_sell_ratio", "log_signed"]
        rets = ["ret_1d", "ret_3d", "ret_5d"]

        ic_test(dib_daily, feats, rets, f"{asset} DIB daily")
        ic_test(dollar_daily, feats, rets, f"{asset} DOLLAR daily")


if __name__ == "__main__":
    main()
