"""Pre-build the mover-continuation signal panel used by the production sleeve.

Reads chimera/1d parquets, computes per-(asset, date):
  - ret_1d
  - btc_30d (joined from BTC)
  - rvol (vol / 10d-MA)
  - mover_continuation_fire (=1 when ret_1d >= +15% AND btc_30d >= -5% AND bucket in {DEGEN,VOLATILE})
  - mover_continuation_strong_fire (=1 when ret_1d >= +25% additional filter)

Output: data/processed/panels/daily/mover_continuation_panel.parquet
"""
from __future__ import annotations
from pathlib import Path
import sys

import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
CHIMERA_DIR = ROOT / "data" / "processed" / "chimera" / "1d"
OUT_DIR = ROOT / "data" / "processed" / "panels" / "daily"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "mover_continuation_panel.parquet"

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from strategy.tier_router_v1 import DNA_BUCKET  # type: ignore

TRIGGER_THRESH_NORMAL = 0.15
TRIGGER_THRESH_STRONG = 0.25
REGIME_MIN_BTC_30D    = -0.05
ELIGIBLE_BUCKETS      = ("DEGEN", "VOLATILE")


def bucket_of(sym: str) -> str:
    return DNA_BUCKET.get(sym, "VOLATILE")


def main():
    files = sorted(CHIMERA_DIR.glob("*_v51_chimera_1d_*.parquet"))
    print(f"Reading {len(files)} chimera 1d parquets...")
    frames = []
    for f in files:
        sym = f.name.split("_")[0].upper().replace("USDT", "")
        df = pl.read_parquet(f, columns=["timestamp","close","volume"]).to_pandas()
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
        df["asset"] = sym
        df = df.sort_values("date").reset_index(drop=True)
        df["ret_1d"] = df["close"].pct_change()
        df["vol_10d_ma"] = df["volume"].rolling(10, min_periods=3).mean()
        df["rvol"] = df["volume"] / df["vol_10d_ma"]
        df["bucket"] = bucket_of(sym)
        frames.append(df[["asset","date","close","volume","ret_1d","rvol","bucket"]])

    panel = pd.concat(frames, ignore_index=True)
    # BTC 30d return
    btc = panel[panel["asset"] == "BTC"][["date","close"]].sort_values("date").copy()
    btc["btc_30d"] = btc["close"].pct_change(30)
    panel = panel.merge(btc[["date","btc_30d"]], on="date", how="left")

    # Fire flags
    panel["mover_continuation_fire"] = (
        (panel["ret_1d"] >= TRIGGER_THRESH_NORMAL) &
        (panel["btc_30d"] >= REGIME_MIN_BTC_30D) &
        (panel["bucket"].isin(ELIGIBLE_BUCKETS))
    ).astype("int8")
    panel["mover_continuation_strong_fire"] = (
        (panel["ret_1d"] >= TRIGGER_THRESH_STRONG) &
        (panel["btc_30d"] >= REGIME_MIN_BTC_30D) &
        (panel["bucket"].isin(ELIGIBLE_BUCKETS))
    ).astype("int8")

    panel.to_parquet(OUT_PATH)
    print(f"Wrote {OUT_PATH}: {len(panel)} rows, "
          f"{panel['mover_continuation_fire'].sum()} normal fires, "
          f"{panel['mover_continuation_strong_fire'].sum()} strong fires "
          f"(over all dates including pre-2024 and UNSEEN -- consumer must filter)")
    print(f"Per-bucket fire counts (normal trigger, all dates):")
    print(panel.groupby('bucket')['mover_continuation_fire'].sum())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
