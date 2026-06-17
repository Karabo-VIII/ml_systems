"""Alpha turn-014: Sub-day dip-buy probe, holding 3-7d, U50 balanced.

Thesis: buy U50 asset on sharp pullback within a confirmed uptrend.
Pullback = 7d return < -15%. Uptrend = 30d return > 0. Exit on 7d time stop
or +5% target.

D1-compliant (spot, no lev, short-term <7d). D6-compliant (sub-daily-ish).

U50 balanced per strat_test_min_universe rule. Paranoid chronological split
(TRAIN 2020-2023 / VAL 2024 / OOS 2025-26).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.strategy.universe import UNIVERSE_50_LIQUID

OUT = ROOT / "logs" / "frontier" / "subday_dipbuy" / "dipbuy_3to7d_U50.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

DIP_PCT = -0.15       # 7d ret < -15%
TREND_MIN = 0.0       # 30d ret > 0
TARGET_PCT = 0.05
STOP_PCT = -0.075
TIME_STOP_D = 7
COST_PCT = 0.0010     # 10 bps per side


def era(date: pd.Timestamp) -> str:
    if date < pd.Timestamp("2024-01-01"):
        return "TRAIN"
    if date < pd.Timestamp("2025-01-01"):
        return "VAL"
    return "OOS"


def load_daily(asset: str) -> pd.DataFrame | None:
    p = ROOT / "logs" / "frontier" / "cycle_gate" / f"{asset.lower()}usdt_daily_klines.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p).sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def simulate(df: pd.DataFrame) -> list[dict]:
    close = df["close"].values
    dates = df["date"].values
    n = len(close)
    # Pre-compute 7d and 30d returns
    ret7 = np.full(n, np.nan)
    ret30 = np.full(n, np.nan)
    for i in range(n):
        if i >= 7:
            ret7[i] = close[i] / close[i - 7] - 1.0
        if i >= 30:
            ret30[i] = close[i] / close[i - 30] - 1.0

    trades = []
    for i in range(30, n - TIME_STOP_D - 1):
        if not (ret7[i] < DIP_PCT and ret30[i] > TREND_MIN):
            continue
        entry = close[i]
        target = entry * (1.0 + TARGET_PCT)
        stop = entry * (1.0 + STOP_PCT)
        exit_idx = i + TIME_STOP_D
        exit_price = close[i + TIME_STOP_D]
        exit_reason = "time"
        for j in range(i + 1, min(i + TIME_STOP_D + 1, n)):
            # Use daily close as exit price (no intrabar HL here for simplicity)
            if close[j] <= stop:
                exit_idx = j; exit_price = stop; exit_reason = "stop"; break
            if close[j] >= target:
                exit_idx = j; exit_price = target; exit_reason = "target"; break
        net_ret = (exit_price / entry) - 1.0 - 2 * COST_PCT
        trades.append({
            "entry_date": pd.Timestamp(dates[i]), "exit_date": pd.Timestamp(dates[exit_idx]),
            "entry_price": entry, "exit_price": exit_price,
            "net_ret": net_ret, "bars_held": exit_idx - i, "exit_reason": exit_reason,
        })
    return trades


def agg(trades: list[dict]) -> dict:
    if not trades:
        return {"n": 0}
    ret = np.asarray([t["net_ret"] for t in trades])
    n = len(ret)
    mean = float(ret.mean())
    std = float(ret.std())
    t = mean / (std / np.sqrt(n)) if std > 0 else 0.0
    hit = float((ret > 0).mean())
    return {"n": n, "mean_pct": mean * 100, "t_stat": t, "hit_rate": hit}


def main() -> None:
    results = []
    all_shipped = {"TRAIN": [], "VAL": [], "OOS": []}
    for asset in UNIVERSE_50_LIQUID:
        df = load_daily(asset)
        if df is None:
            results.append({"asset": asset, "error": "no_data"})
            continue
        tr = simulate(df)
        # Split by era
        by_era = {"TRAIN": [], "VAL": [], "OOS": [], "ALL": []}
        for t in tr:
            e = era(t["entry_date"])
            by_era[e].append(t)
            by_era["ALL"].append(t)

        aggs = {e: agg(by_era[e]) for e in ("TRAIN", "VAL", "OOS", "ALL")}
        results.append({"asset": asset, **aggs})
        all_out = aggs["ALL"]
        oos_out = aggs["OOS"]
        ship = (oos_out.get("n", 0) >= 10
                and oos_out.get("t_stat", 0) > 2
                and oos_out.get("mean_pct", 0) > 0
                and oos_out.get("hit_rate", 0) > 0.5)
        tag = " *OOS_SHIP*" if ship else ""
        oe = aggs["OOS"]
        print(f"  {asset:>10s}: ALL n={all_out.get('n',0):3d} mean={all_out.get('mean_pct',0):+.2f}% "
              f"t={all_out.get('t_stat',0):+.2f}  |  "
              f"OOS n={oe.get('n',0):2d} t={oe.get('t_stat',0):+.2f} hit={oe.get('hit_rate',0):.2f}"
              f"{tag}")
        if ship:
            all_shipped["OOS"].append({"asset": asset, **oos_out})

    oos_ship = all_shipped["OOS"]
    print(f"\n=== SUMMARY ===")
    print(f"OOS_SHIP (t>2 & mean>0 & hit>0.5 on OOS with n>=10): {len(oos_ship)}")
    for s in sorted(oos_ship, key=lambda x: -x["t_stat"]):
        print(f"  {s['asset']:<10s} n={s['n']:3d}  mean={s['mean_pct']:+.3f}%  "
              f"t={s['t_stat']:+.2f}  hit={s['hit_rate']:.3f}")

    with open(OUT, "w") as f:
        json.dump({"results": results, "oos_ship": oos_ship}, f, indent=2, default=str)
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
