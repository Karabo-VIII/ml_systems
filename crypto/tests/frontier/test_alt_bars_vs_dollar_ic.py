"""IC comparison: Tick Runs / Volume Runs / Adaptive Vol bars vs dollar bars.

For each bar type, compute daily flow_imbalance + buy_sell_ratio on top-10,
then test IC vs forward 1d/3d returns.

Decision: if any new bar type gives IC > 0.10 at |t| > 3, ships as a valid
feature source for future ranker retrain. Otherwise concede.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
BAR_DIRS = {
    "tick_runs": ROOT / "data" / "frontier" / "runs_bars",
    "vol_runs": ROOT / "data" / "frontier" / "runs_bars",
    "adaptive_vol": ROOT / "data" / "frontier" / "adaptive_bars",
    "dib": ROOT / "data" / "frontier" / "dib",
}

ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
          "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]


def extract_daily(fp: Path, asset: str) -> pd.DataFrame:
    df = pl.read_parquet(fp).to_pandas()
    ts_col = "bar_end_ts" if "bar_end_ts" in df.columns else "timestamp"
    ts_arr = df[ts_col].values
    ts_norm = np.where(ts_arr > 1e15, ts_arr / 1000, ts_arr).astype(np.int64)
    df["date"] = pd.to_datetime(ts_norm, unit="ms").floor("D")
    # Some bar types use different column names. Pick whatever's available.
    buy_col = "buy_usd" if "buy_usd" in df.columns else None
    sell_col = "sell_usd" if "sell_usd" in df.columns else None
    signed_col = "signed_usd" if "signed_usd" in df.columns else ("signed_run" if "signed_run" in df.columns else None)
    if not (buy_col and sell_col):
        return pd.DataFrame()
    daily = df.groupby("date").agg({
        "close": "last",
        buy_col: "sum",
        sell_col: "sum",
    }).reset_index()
    if signed_col:
        daily["signed_proxy"] = df.groupby("date")[signed_col].sum().values
    else:
        daily["signed_proxy"] = daily[buy_col] - daily[sell_col]
    daily["flow_imbalance"] = daily["signed_proxy"] / (daily[buy_col] + daily[sell_col]).replace(0, np.nan)
    daily["buy_sell_ratio"] = daily[buy_col] / daily[sell_col].replace(0, np.nan)
    daily["ret_1d"] = daily["close"].pct_change().shift(-1)  # fwd 1d
    daily["ret_3d"] = daily["close"].pct_change(3).shift(-3)
    daily["asset"] = asset.replace("USDT", "")
    return daily


def ic_test(df: pd.DataFrame, label: str):
    # Pool across assets (cross-sectional-in-time)
    df = df.dropna(subset=["flow_imbalance", "ret_1d"])
    if len(df) < 50:
        return None
    results = {}
    for feat in ["flow_imbalance", "buy_sell_ratio"]:
        for ret in ["ret_1d", "ret_3d"]:
            r = df[[feat, ret]].dropna()
            if len(r) < 50:
                continue
            pear = r[feat].corr(r[ret], method="pearson")
            n = len(r)
            t = pear * np.sqrt(n - 2) / np.sqrt(max(1 - pear ** 2, 1e-6)) if abs(pear) < 1 else 0
            results[f"{feat}_vs_{ret}"] = (pear, t, n)
            print(f"  {label:<30} {feat:<17} vs {ret}: IC {pear:+.4f} t={t:+.2f} n={n}")
    return results


def main():
    print("\n=== Bar-type IC comparison (aligned 2025-01-01 to 2026-04-16, pooled top-10) ===")
    for bar_type, bar_dir in BAR_DIRS.items():
        all_frames = []
        for asset in ASSETS:
            if bar_type == "tick_runs":
                fp = bar_dir / f"{asset}_tick_runs.parquet"
            elif bar_type == "vol_runs":
                fp = bar_dir / f"{asset}_vol_runs.parquet"
            elif bar_type == "adaptive_vol":
                fp = bar_dir / f"{asset}_adaptive_vol.parquet"
            elif bar_type == "dib":
                fp = bar_dir / f"{asset}_dib_2025.parquet"
            if not fp.exists():
                continue
            d = extract_daily(fp, asset)
            if not d.empty:
                d = d[(d["date"] >= "2025-01-01") & (d["date"] < "2026-04-16")]
                all_frames.append(d)
        if not all_frames:
            print(f"  [{bar_type}] no data")
            continue
        pool = pd.concat(all_frames, ignore_index=True)
        ic_test(pool, bar_type)


if __name__ == "__main__":
    main()
