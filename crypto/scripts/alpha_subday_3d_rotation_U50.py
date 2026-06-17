"""Alpha turn-014: Simple 3-day momentum rotation, hold 3d, U50 balanced.

Thesis: at each day, rank U50 by trailing 3-day return, buy top-K
(equal weight), hold 3 days, rebalance. Different from xsec K=10+10 (which
uses ML ranker + daily rebalance + long/short) -- this is naive top-K
momentum with 3d cadence.

Test on full U50 with paranoid TRAIN/VAL/OOS split.
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

OUT = ROOT / "logs" / "frontier" / "subday_3d_mom" / "mom3d_top5_U50.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

LOOKBACK = 3   # days
HOLD = 3
TOP_K = 5
COST_PCT = 0.0010  # per side


def era(date: pd.Timestamp) -> str:
    if date < pd.Timestamp("2024-01-01"):
        return "TRAIN"
    if date < pd.Timestamp("2025-01-01"):
        return "VAL"
    return "OOS"


def load_panel() -> pd.DataFrame:
    rows = []
    for asset in UNIVERSE_50_LIQUID:
        p = ROOT / "logs" / "frontier" / "cycle_gate" / f"{asset.lower()}usdt_daily_klines.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p)[["date", "close"]].copy()
        d["asset"] = asset
        rows.append(d)
    if not rows:
        raise SystemExit("no data")
    df = pd.concat(rows, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def main() -> None:
    df = load_panel()
    wide = df.pivot(index="date", columns="asset", values="close").sort_index()
    rets = wide.pct_change()
    # 3-day momentum = close_t / close_{t-3} - 1
    mom = wide / wide.shift(LOOKBACK) - 1.0

    # At each rebalance date (every HOLD days), rank by mom, buy top-K
    dates = wide.index
    equity_flat = 10000.0
    eq_series = []
    era_tally = {"TRAIN": [], "VAL": [], "OOS": []}

    for i in range(LOOKBACK + 1, len(dates) - HOLD):
        if i % HOLD != 0:
            continue  # rebalance only every HOLD days
        day = dates[i]
        m = mom.loc[day].dropna()
        if len(m) < TOP_K * 2:
            continue
        top = m.sort_values(ascending=False).head(TOP_K).index.tolist()
        # Forward return over HOLD days (equal-weight basket)
        fwd_end = dates[i + HOLD]
        entry_prices = wide.loc[day, top].values
        exit_prices = wide.loc[fwd_end, top].values
        if np.any(np.isnan(entry_prices)) or np.any(np.isnan(exit_prices)):
            # skip rebalance if any of the top-K lacks a price
            continue
        asset_rets = (exit_prices / entry_prices) - 1.0
        basket_ret = float(np.mean(asset_rets))
        # Turnover: full basket changes every HOLD, so 2 sides of costs per rebal
        net = basket_ret - 2 * COST_PCT
        equity_flat *= (1.0 + net)
        eq_series.append((day, equity_flat, net))
        e = era(day)
        era_tally[e].append(net)

    print(f"Rebalances: {len(eq_series)}")
    print(f"Final equity (start $10000): ${equity_flat:,.2f}")
    print()
    print(f"{'ERA':<8} {'n':>4} {'mean':>8} {'std':>8} {'t':>6} {'hit':>6} {'annualized':>12}")
    results = {}
    for e in ("TRAIN", "VAL", "OOS"):
        arr = np.asarray(era_tally[e])
        if len(arr) < 3:
            print(f"{e:<8} n={len(arr)} thin")
            results[e] = {"n": int(len(arr))}
            continue
        n = len(arr); mean = arr.mean(); std = arr.std()
        t = mean / (std / np.sqrt(n)) if std > 0 else 0.0
        hit = float((arr > 0).mean())
        ann_cagr = (1.0 + mean) ** (365.0 / HOLD) - 1.0
        print(f"{e:<8} n={n:>3d} mean={mean*100:+.3f}% std={std*100:.3f}% t={t:+.2f} hit={hit:.3f} cagr={ann_cagr*100:+.1f}%")
        results[e] = {"n": int(n), "mean_pct": float(mean * 100),
                      "std_pct": float(std * 100), "t_stat": float(t),
                      "hit_rate": hit, "ann_cagr_pct": float(ann_cagr * 100)}

    # Full window
    all_rets = np.concatenate([np.asarray(era_tally[e]) for e in ("TRAIN", "VAL", "OOS")])
    if len(all_rets) > 3:
        mean = all_rets.mean(); std = all_rets.std()
        t = mean / (std / np.sqrt(len(all_rets))) if std > 0 else 0.0
        hit = float((all_rets > 0).mean())
        ann_cagr = (1.0 + mean) ** (365.0 / HOLD) - 1.0
        print(f"\n{'ALL':<8} n={len(all_rets):>3d} mean={mean*100:+.3f}% std={std*100:.3f}% t={t:+.2f} hit={hit:.3f} cagr={ann_cagr*100:+.1f}%")

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
