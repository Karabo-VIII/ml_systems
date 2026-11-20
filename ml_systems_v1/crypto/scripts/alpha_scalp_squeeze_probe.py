"""Alpha turn-010.5 (inline user D4 follow-up): Scalp Squeeze-Breakout probe.

User clarified scalping definition: enter on accumulation/vol-expansion in
dollar bars, target 1-4% move, exit within a couple of hours.

Test: does a Bollinger-squeeze-into-breakout pattern on BTC dollar bars
produce scalp-size moves with directional predictability?

Setup:
  - Rolling Bollinger-band width over last 20 bars
  - SQUEEZE = current width is in bottom-quartile of trailing 200-bar widths
  - BREAKOUT = first bar after squeeze where close > upper BB (long) OR below lower BB (short, skipped under D1 spot-only)
  - ENTRY at close of breakout bar
  - EXIT: whichever of +2% target / -1% stop / 60-bar time-stop fires

Dollar-bar cadence on BTCUSDT ~= 4 min (early 2020) to ~15 min (low-vol
periods) -- 60 bars = 4-15 hours hold, aligns with "couple of hours" ask.

Replay scope: 2024-01-01 to 2026-04-22 on BTCUSDT (~600K bars). Report
hit rate, mean net, t-stat, aggregate CAGR if deployed at 10% of capital.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "logs" / "frontier" / "scalp_squeeze"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BB_WINDOW = 20
WIDTH_LOOKBACK = 200
TARGET_PCT = 0.02     # +2% target
STOP_PCT = -0.01      # -1% stop
TIME_STOP_BARS = 60   # 60 bars (~4h avg)
COST_PCT = 0.0010     # 10 bps per side (maker-optimistic); 20 bps round trip


def run_probe(symbol: str = "BTCUSDT", start: str = "2024-01-01") -> dict:
    df = pd.read_parquet(
        ROOT / "data" / "processed" / f"{symbol.lower()}_v50_chimera.parquet",
        columns=["timestamp", "close", "high", "low"],
    )
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df[df["ts"] >= pd.Timestamp(start)].reset_index(drop=True)
    print(f"[DATA] {symbol}: {len(df)} bars from {df['ts'].iloc[0]} -> {df['ts'].iloc[-1]}")

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    # Bollinger width
    ma = pd.Series(close).rolling(BB_WINDOW, min_periods=BB_WINDOW).mean().values
    sd = pd.Series(close).rolling(BB_WINDOW, min_periods=BB_WINDOW).std().values
    upper = ma + 2 * sd
    lower = ma - 2 * sd
    width = (upper - lower) / np.maximum(ma, 1e-9)  # relative width

    # Squeeze: current width in bottom-quartile of trailing 200
    width_q25 = pd.Series(width).rolling(WIDTH_LOOKBACK, min_periods=WIDTH_LOOKBACK).quantile(0.25).values
    is_squeeze = width < width_q25

    # Breakout: close above upper on first bar after squeeze
    prev_was_squeeze = np.concatenate([[False], is_squeeze[:-1]])
    close_above_upper = close > upper
    long_trigger = prev_was_squeeze & close_above_upper

    print(f"[SETUP] bars with long triggers: {long_trigger.sum()} / {len(df)} "
          f"({100 * long_trigger.sum() / len(df):.2f}%)")

    # Simulate forward on each trigger
    trades = []
    n = len(df)
    for i in np.where(long_trigger)[0]:
        if i + TIME_STOP_BARS >= n:
            continue
        entry = close[i]
        target = entry * (1.0 + TARGET_PCT)
        stop = entry * (1.0 + STOP_PCT)
        exit_idx = i + TIME_STOP_BARS
        exit_reason = "time"
        exit_price = close[i + TIME_STOP_BARS]
        # Walk forward bar-by-bar, check stop first (conservative)
        for j in range(i + 1, i + TIME_STOP_BARS + 1):
            if low[j] <= stop:
                exit_idx = j; exit_price = stop; exit_reason = "stop"
                break
            if high[j] >= target:
                exit_idx = j; exit_price = target; exit_reason = "target"
                break
        raw_ret = (exit_price / entry) - 1.0
        net_ret = raw_ret - 2 * COST_PCT  # round-trip
        trades.append({
            "entry_idx": int(i),
            "entry_ts": str(df["ts"].iloc[i]),
            "entry_price": float(entry),
            "exit_idx": int(exit_idx),
            "exit_ts": str(df["ts"].iloc[exit_idx]),
            "exit_price": float(exit_price),
            "bars_held": int(exit_idx - i),
            "raw_ret": float(raw_ret),
            "net_ret": float(net_ret),
            "exit_reason": exit_reason,
        })

    td = pd.DataFrame(trades)
    if td.empty:
        return {"error": "no trades"}

    # Stats
    n_trades = len(td)
    mean_net = td["net_ret"].mean()
    std_net = td["net_ret"].std()
    t_stat = mean_net / (std_net / np.sqrt(n_trades)) if std_net > 0 else 0.0
    hit_rate = float((td["net_ret"] > 0).mean())
    mean_hold = float(td["bars_held"].mean())

    # Exit reason breakdown
    reason_counts = td["exit_reason"].value_counts().to_dict()

    # If deployed at 10% of capital per trade (sequential, not concurrent):
    # compounded aggregate return
    eq_mult = (1.0 + 0.10 * td["net_ret"]).prod()

    # Shuffle control (randomize entry timestamps within universe)
    rng = np.random.default_rng(42)
    shuffled_rets = []
    for _ in range(20):
        rand_entries = rng.integers(WIDTH_LOOKBACK + BB_WINDOW, n - TIME_STOP_BARS, size=n_trades)
        for i in rand_entries:
            entry = close[i]
            exit_price = close[i + TIME_STOP_BARS]
            shuffled_rets.append((exit_price / entry) - 1.0 - 2 * COST_PCT)
    shuffled_mean = float(np.mean(shuffled_rets))
    shuffled_std = float(np.std(shuffled_rets))
    n_sh = len(shuffled_rets)
    null_z = (mean_net - shuffled_mean) / (shuffled_std / np.sqrt(n_sh)) if shuffled_std > 0 else 0.0

    out = {
        "symbol": symbol,
        "start": start,
        "n_trades": int(n_trades),
        "mean_net_pct": float(mean_net * 100),
        "std_net_pct": float(std_net * 100),
        "t_stat": float(t_stat),
        "hit_rate": hit_rate,
        "mean_hold_bars": mean_hold,
        "exit_reasons": reason_counts,
        "eq_mult_at_10pct_sizing": float(eq_mult),
        "shuffled_mean_pct": shuffled_mean * 100,
        "null_z_vs_shuffle": null_z,
    }
    with open(OUT_DIR / f"scalp_squeeze_{symbol}_result.json", "w") as f:
        json.dump(out, f, indent=2)
    td.to_csv(OUT_DIR / f"scalp_squeeze_{symbol}_trades.csv", index=False)
    return out


def main() -> None:
    r = run_probe("BTCUSDT", "2024-01-01")
    print()
    print(f"=== SCALP SQUEEZE PROBE -- BTCUSDT, 2024-2026 ===")
    for k, v in r.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
