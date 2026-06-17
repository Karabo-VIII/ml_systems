"""Alpha turn-011: Scalp flow-imbalance burst probe, U50 balanced.

Third scalp trigger to test before conceding scalp-on-bars as a paradigm.

Trigger: norm_flow_imbalance > 2 sigma (feature already z-scored in dataset)
         AND directional (close > prev close)
Entry: close of trigger bar
Target: +1% | Stop: -0.5% | Time stop: 30 bars (~2h at 4min bars)
Cost: 20 bps round trip

Full U50 coverage per new constitution rule.
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

OUT = ROOT / "logs" / "frontier" / "scalp_flowimb" / "scalp_flowimb_U50.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

TARGET_PCT = 0.01
STOP_PCT = -0.005
TIME_STOP_BARS = 30
COST_PCT = 0.0010
FLOW_THRESHOLD = 2.0
START = "2024-01-01"


def run_one(asset: str) -> dict:
    path = ROOT / "data" / "processed" / f"{asset.lower()}usdt_v50_chimera.parquet"
    if not path.exists():
        return {"asset": asset, "error": "no_data"}
    try:
        df = pd.read_parquet(path, columns=["timestamp", "close", "high", "low", "norm_flow_imbalance"])
    except Exception as e:
        return {"asset": asset, "error": f"read_err:{e}"}
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df[df["ts"] >= pd.Timestamp(START)].reset_index(drop=True)
    if len(df) < 500:
        return {"asset": asset, "error": f"thin:{len(df)}"}

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    fi = df["norm_flow_imbalance"].values
    prev_close = np.concatenate([[close[0]], close[:-1]])
    trigger = (fi > FLOW_THRESHOLD) & (close > prev_close)

    rets = []
    n = len(df)
    for i in np.where(trigger)[0]:
        if i + TIME_STOP_BARS >= n:
            continue
        entry = close[i]
        target = entry * (1.0 + TARGET_PCT)
        stop = entry * (1.0 + STOP_PCT)
        exit_price = close[i + TIME_STOP_BARS]
        for j in range(i + 1, i + TIME_STOP_BARS + 1):
            if low[j] <= stop:
                exit_price = stop; break
            if high[j] >= target:
                exit_price = target; break
        rets.append((exit_price / entry) - 1.0 - 2 * COST_PCT)

    if not rets:
        return {"asset": asset, "error": "no_trades"}
    arr = np.asarray(rets)
    n_t = len(arr)
    mean = float(arr.mean())
    std = float(arr.std())
    t = mean / (std / np.sqrt(n_t)) if std > 0 else 0.0
    hit = float((arr > 0).mean())
    return {
        "asset": asset,
        "n_trades": int(n_t),
        "mean_net_pct": mean * 100,
        "t_stat": float(t),
        "hit_rate": hit,
    }


def main() -> None:
    results = []
    ok = 0
    for asset in UNIVERSE_50_LIQUID:
        r = run_one(asset)
        results.append(r)
        if "error" in r:
            print(f"  {asset:>10s}: ERROR={r['error']}")
            continue
        ok += 1
        star = ""
        if r["t_stat"] > 2 and r["mean_net_pct"] > 0 and r["hit_rate"] > 0.5:
            star = "  *SHIP*"
        print(f"  {asset:>10s}: n={r['n_trades']:5d}  mean={r['mean_net_pct']:+.3f}%  "
              f"t={r['t_stat']:+.2f}  hit={r['hit_rate']:.3f}{star}")

    candidates = [r for r in results if "error" not in r
                  and r["t_stat"] > 2 and r["mean_net_pct"] > 0 and r["hit_rate"] > 0.5]
    print(f"\n=== SUMMARY: {ok}/{len(UNIVERSE_50_LIQUID)} U50 assets tested ===")
    print(f"Scalp candidates (t>2 & mean>0 & hit>0.5): {len(candidates)}")
    for c in sorted(candidates, key=lambda x: -x["t_stat"]):
        print(f"  {c['asset']:<10s} n={c['n_trades']:5d}  mean={c['mean_net_pct']:+.3f}%  "
              f"t={c['t_stat']:+.2f}  hit={c['hit_rate']:.3f}")

    with open(OUT, "w") as f:
        json.dump({"results": results, "candidates": candidates}, f, indent=2)
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
