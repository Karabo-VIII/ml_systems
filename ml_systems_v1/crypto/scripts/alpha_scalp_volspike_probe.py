"""Alpha turn-010.5 follow-up: Scalp volume-spike probe.

First MVP (bollinger-squeeze breakout) failed hard. Trying a different trigger:
volume-spike burst. Hypothesis: high-volume bars carry more info than the
continuous bar stream; the volume spike IS the signal.

Setup:
  - Trigger: volume on current bar > 3x rolling 20-bar avg volume AND
             current close > prev close (directional momentum confirmation)
  - Entry: close of trigger bar (long only, per D1 spot)
  - Exit: +1% target / -0.5% stop / 30-bar time stop (~2h avg)

Also test on altcoins, since BTC is low-vol compared to alts -- scalp headroom
may exist there.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "logs" / "frontier" / "scalp_volspike"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VOL_WINDOW = 20
VOL_MULT = 3.0
TARGET_PCT = 0.01
STOP_PCT = -0.005
TIME_STOP_BARS = 30
COST_PCT = 0.0010


def run_probe(symbol: str, start: str = "2024-01-01") -> dict:
    try:
        df = pd.read_parquet(
            ROOT / "data" / "processed" / f"{symbol.lower()}_v50_chimera.parquet",
            columns=["timestamp", "close", "high", "low", "volume"],
        )
    except Exception as e:
        return {"error": str(e), "symbol": symbol}
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df[df["ts"] >= pd.Timestamp(start)].reset_index(drop=True)
    if len(df) < 200:
        return {"error": "insufficient data", "symbol": symbol}

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    vol = df["volume"].values

    vol_ma = pd.Series(vol).rolling(VOL_WINDOW, min_periods=VOL_WINDOW).mean().values
    vol_spike = vol > (VOL_MULT * vol_ma)
    prev_close = np.concatenate([[close[0]], close[:-1]])
    directional = close > prev_close

    trigger = vol_spike & directional
    n_triggers = int(trigger.sum())
    print(f"[{symbol}] {n_triggers} vol-spike triggers / {len(df)} bars ({100*n_triggers/len(df):.3f}%)")

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
        exit_reason = "time"
        for j in range(i + 1, i + TIME_STOP_BARS + 1):
            if low[j] <= stop:
                exit_idx = j; exit_price = stop; exit_reason = "stop"
                break
            if high[j] >= target:
                exit_idx = j; exit_price = target; exit_reason = "target"
                break
        raw_ret = (exit_price / entry) - 1.0
        net_ret = raw_ret - 2 * COST_PCT
        trades.append({
            "raw_ret": raw_ret, "net_ret": net_ret,
            "bars_held": exit_idx - i, "exit_reason": exit_reason,
        })

    td = pd.DataFrame(trades)
    if td.empty:
        return {"error": "no trades", "symbol": symbol}

    mean_net = td["net_ret"].mean()
    std_net = td["net_ret"].std()
    t_stat = mean_net / (std_net / np.sqrt(len(td))) if std_net > 0 else 0.0
    hit_rate = float((td["net_ret"] > 0).mean())

    # Shuffle control
    rng = np.random.default_rng(42)
    rand_entries = rng.integers(VOL_WINDOW, n - TIME_STOP_BARS, size=len(td))
    shuffled = []
    for i in rand_entries:
        shuffled.append(close[i + TIME_STOP_BARS] / close[i] - 1.0 - 2 * COST_PCT)
    sh_mean = float(np.mean(shuffled))
    sh_std = float(np.std(shuffled))
    null_z = (mean_net - sh_mean) / (sh_std / np.sqrt(len(td))) if sh_std > 0 else 0.0

    return {
        "symbol": symbol,
        "n_trades": int(len(td)),
        "mean_net_pct": float(mean_net * 100),
        "t_stat": float(t_stat),
        "hit_rate": hit_rate,
        "mean_hold_bars": float(td["bars_held"].mean()),
        "exit_reasons": td["exit_reason"].value_counts().to_dict(),
        "shuffled_mean_pct": sh_mean * 100,
        "null_z_vs_shuffle": null_z,
    }


def main() -> None:
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "AVAXUSDT"]
    results = {}
    for s in symbols:
        r = run_probe(s)
        results[s] = r
        if "error" in r:
            print(f"  {s}: SKIPPED -- {r['error']}")
            continue
        sig_mark = " (signal!)" if r["null_z_vs_shuffle"] > 2 and r["t_stat"] > 2 else ""
        print(f"  {s}: n={r['n_trades']:5d}  mean={r['mean_net_pct']:+.3f}%  "
              f"t={r['t_stat']:+.2f}  hit={r['hit_rate']:.3f}  "
              f"nullz={r['null_z_vs_shuffle']:+.2f}{sig_mark}")
    with open(OUT_DIR / "scalp_volspike_panel_result.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVE] {OUT_DIR / 'scalp_volspike_panel_result.json'}")


if __name__ == "__main__":
    main()
