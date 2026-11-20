"""Alpha turn-010.5 follow-up#2: Scalp vol-spike over FULL U50 universe.

User rightly called out: earlier probes used partial U10 only. Re-running
across all 50 U50 assets where processed dollar-bar data exists. Mid-cap
alts have more intrinsic vol -> more scalp headroom.
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

OUT_DIR = ROOT / "logs" / "frontier" / "scalp_volspike"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VOL_WINDOW = 20
VOL_MULT = 3.0
TARGET_PCT = 0.01
STOP_PCT = -0.005
TIME_STOP_BARS = 30
COST_PCT = 0.0010


def run_one(symbol: str, start: str = "2024-01-01") -> dict:
    path = ROOT / "data" / "processed" / f"{symbol.lower()}_v50_chimera.parquet"
    if not path.exists():
        return {"symbol": symbol, "error": "no_data"}
    try:
        df = pd.read_parquet(path, columns=["timestamp", "close", "high", "low", "volume"])
    except Exception as e:
        return {"symbol": symbol, "error": f"read_err:{e}"}
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df[df["ts"] >= pd.Timestamp(start)].reset_index(drop=True)
    if len(df) < 500:
        return {"symbol": symbol, "error": f"thin_data:{len(df)}"}

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    vol = df["volume"].values

    vol_ma = pd.Series(vol).rolling(VOL_WINDOW, min_periods=VOL_WINDOW).mean().values
    prev_close = np.concatenate([[close[0]], close[:-1]])
    trigger = (vol > VOL_MULT * vol_ma) & (close > prev_close)

    trades = []
    n = len(df)
    for i in np.where(trigger)[0]:
        if i + TIME_STOP_BARS >= n:
            continue
        entry = close[i]
        target = entry * (1.0 + TARGET_PCT)
        stop = entry * (1.0 + STOP_PCT)
        exit_idx = i + TIME_STOP_BARS
        exit_price = close[i + TIME_STOP_BARS]
        for j in range(i + 1, i + TIME_STOP_BARS + 1):
            if low[j] <= stop:
                exit_idx = j; exit_price = stop; break
            if high[j] >= target:
                exit_idx = j; exit_price = target; break
        net_ret = (exit_price / entry) - 1.0 - 2 * COST_PCT
        trades.append(net_ret)

    if not trades:
        return {"symbol": symbol, "error": "no_trades"}

    arr = np.asarray(trades)
    n_t = len(arr)
    mean = arr.mean()
    std = arr.std()
    t = mean / (std / np.sqrt(n_t)) if std > 0 else 0.0
    hit = float((arr > 0).mean())
    return {
        "symbol": symbol,
        "n_trades": int(n_t),
        "mean_net_pct": float(mean * 100),
        "t_stat": float(t),
        "hit_rate": hit,
        "n_bars": int(n),
    }


def main() -> None:
    results = []
    shipped = []
    sym_map = {s: f"{s}USDT" for s in UNIVERSE_50_LIQUID}
    for asset, symbol in sym_map.items():
        r = run_one(symbol)
        results.append(r)
        if "error" in r:
            print(f"  {asset:>10s}: ERROR={r['error']}")
            continue
        sig = ""
        if r["t_stat"] > 2 and r["mean_net_pct"] > 0 and r["hit_rate"] > 0.5:
            sig = "  *** SCALP CANDIDATE ***"
            shipped.append(r)
        print(f"  {asset:>10s}: n={r['n_trades']:5d}  mean={r['mean_net_pct']:+.3f}%  "
              f"t={r['t_stat']:+.2f}  hit={r['hit_rate']:.3f}{sig}")

    print()
    print(f"=== SUMMARY: {len(results)} U50 assets tested ===")
    clean = [r for r in results if "error" not in r]
    losing = [r for r in clean if r["t_stat"] < 0]
    winning = [r for r in clean if r["t_stat"] > 2 and r["mean_net_pct"] > 0]
    print(f"  tested: {len(clean)}/{len(UNIVERSE_50_LIQUID)}")
    print(f"  net-losing (t<0): {len(losing)}")
    print(f"  SCALP CANDIDATES (t>2 & mean>0 & hit>0.5): {len(winning)}")
    if winning:
        print("  >> CANDIDATES:")
        for r in sorted(winning, key=lambda x: -x["t_stat"]):
            print(f"     {r['symbol']:<12} n={r['n_trades']:4d}  mean={r['mean_net_pct']:+.3f}%  "
                  f"t={r['t_stat']:+.2f}  hit={r['hit_rate']:.3f}")
    with open(OUT_DIR / "scalp_volspike_U50_result.json", "w") as f:
        json.dump({"results": results, "shipped": shipped}, f, indent=2)
    print(f"\n[SAVE] {OUT_DIR / 'scalp_volspike_U50_result.json'}")


if __name__ == "__main__":
    main()
