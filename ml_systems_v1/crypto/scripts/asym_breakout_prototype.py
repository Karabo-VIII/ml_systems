"""Asymmetric strategy prototype: N-day-high breakout with trailing stop.

Family A from docs/ASYMMETRIC_STRATEGIES_FRONTIER_2026_04_24.md.

Thesis: price breaking a recent high shows momentum with disproportionate
upside. Tight initial stop + trailing stop caps downside at -2% while
letting winners run to +10-30%.

Design:
    Entry: asset breaks N_breakout-day high (try N in {10, 20, 30, 60})
    Initial stop: entry_price * (1 - init_stop_pct)
    Trailing stop: peak_since_entry * (1 - trail_stop_pct)
    Exit: initial stop hit OR trailing stop hit OR max_hold days

Reports:
    n_trades, hit_rate, mean_win, median_win, max_win,
    mean_loss, median_loss, max_loss,
    ASYMMETRY_RATIO = mean_win / |mean_loss|
    Kelly-log growth per trade
    Sharpe (for comparison)
    Max drawdown
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import glob
import numpy as np
import pandas as pd
import polars as pl

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

sys.path.insert(0, str(ROOT / "src" / "strategy"))
from universe import UNIVERSE_50_LIQUID
UNIVERSE = set(UNIVERSE_50_LIQUID)

MAKER_RT = 0.08
TEST_START = "2025-01-01"
TEST_END = "2026-04-22"


def build_daily_panel():
    all_fps = sorted(glob.glob(str(DATA / "*_chimera.parquet")))
    rows = []
    for fp in all_fps:
        asset = Path(fp).stem.replace("usdt_v50_chimera", "").upper()
        if asset not in UNIVERSE:
            continue
        try:
            df = pl.read_parquet(fp, columns=["timestamp", "close", "high", "low", "open"]).to_pandas()
        except Exception:
            continue
        if len(df) < 1000:
            continue
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
        d = df.groupby("date").agg({
            "close": "last", "high": "max", "low": "min", "open": "first",
        }).reset_index()
        d["asset"] = asset
        rows.append(d)
    panel = pd.concat(rows, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    return panel


def simulate_breakout(panel, N_breakout=20, init_stop=0.02, trail_stop=0.05,
                      max_hold=30, capital_per_trade=10000.0):
    """Run breakout backtest on the full panel, return per-trade stats."""
    trades = []
    for asset, adf in panel.groupby("asset"):
        adf = adf.sort_values("date").reset_index(drop=True)
        # Rolling N-day high (exclusive of today)
        adf["breakout_high"] = adf["high"].shift(1).rolling(N_breakout, min_periods=N_breakout).max()

        adf = adf[adf["date"] >= TEST_START].reset_index(drop=True)
        adf = adf[adf["date"] <= TEST_END].reset_index(drop=True)
        if len(adf) < max_hold + 5:
            continue

        i = 0
        while i < len(adf):
            row = adf.iloc[i]
            if pd.isna(row["breakout_high"]) or row["close"] <= row["breakout_high"]:
                i += 1
                continue
            # Entry at close of breakout day
            entry_price = row["close"]
            entry_date = row["date"]
            entry_idx = i
            peak = entry_price
            stop = entry_price * (1 - init_stop)
            exit_price = None
            exit_date = None
            exit_reason = None
            for j in range(i + 1, min(i + 1 + max_hold, len(adf))):
                r2 = adf.iloc[j]
                # Check if stop is hit during the day (low <= stop)
                if r2["low"] <= stop:
                    exit_price = stop  # assume exit at stop
                    exit_date = r2["date"]
                    exit_reason = "stop"
                    break
                # Update peak and trailing stop
                if r2["high"] > peak:
                    peak = r2["high"]
                    new_trail = peak * (1 - trail_stop)
                    if new_trail > stop:
                        stop = new_trail
            else:
                # Max-hold exit
                r2 = adf.iloc[min(entry_idx + max_hold, len(adf) - 1)]
                exit_price = r2["close"]
                exit_date = r2["date"]
                exit_reason = "max_hold"

            net_ret_pct = (exit_price / entry_price - 1) * 100 - MAKER_RT
            trades.append({
                "asset": asset,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "hold_days": (pd.Timestamp(exit_date) - pd.Timestamp(entry_date)).days,
                "net_ret_pct": net_ret_pct,
                "exit_reason": exit_reason,
                "entry_price": entry_price,
                "peak_price": peak,
                "exit_price": exit_price,
            })
            # Skip ahead past the exit to avoid overlapping trades per asset
            # (could be relaxed for higher turnover)
            i = max(i + 1, adf.index.get_loc(adf[adf["date"] == exit_date].index[0]) + 1
                    if len(adf[adf["date"] == exit_date]) else i + 1)

    return pd.DataFrame(trades)


def stats(trades, capital=10000.0):
    if len(trades) == 0:
        return {"n_trades": 0}
    r = trades["net_ret_pct"].values
    wins = r[r > 0]
    losses = r[r <= 0]
    hit_rate = len(wins) / len(r) if len(r) else 0
    asym = (wins.mean() / abs(losses.mean())) if len(wins) and len(losses) and losses.mean() != 0 else float("inf")
    # Kelly-log growth per trade: p*log(1+W) + (1-p)*log(1+L)
    # Convert pct returns to fractions
    mean_win_frac = wins.mean() / 100 if len(wins) else 0
    mean_loss_frac = losses.mean() / 100 if len(losses) else 0
    kelly_g = hit_rate * np.log1p(mean_win_frac) + (1 - hit_rate) * np.log1p(mean_loss_frac)
    # Per-trade EV (raw fraction)
    ev = r.mean() / 100
    # Annualized (assuming ~hold_days between trades on average)
    avg_hold = trades["hold_days"].mean()
    # Equity curve (one trade at a time, full capital per trade)
    eq = capital * np.cumprod(1 + r / 100)
    total_ret = (eq[-1] / capital - 1) * 100
    daily_eq = pd.Series(eq).values
    dd = ((daily_eq - np.maximum.accumulate(daily_eq)) / np.maximum.accumulate(daily_eq)).min() * 100

    return {
        "n_trades": len(r),
        "hit_rate": hit_rate,
        "mean_trade_pct": r.mean(),
        "median_trade_pct": float(np.median(r)),
        "mean_win_pct": wins.mean() if len(wins) else 0,
        "median_win_pct": float(np.median(wins)) if len(wins) else 0,
        "max_win_pct": wins.max() if len(wins) else 0,
        "mean_loss_pct": losses.mean() if len(losses) else 0,
        "median_loss_pct": float(np.median(losses)) if len(losses) else 0,
        "max_loss_pct": losses.min() if len(losses) else 0,
        "asymmetry_ratio": asym,
        "kelly_log_g_per_trade": kelly_g,
        "avg_hold_days": avg_hold,
        "total_ret_one_trade_at_a_time_pct": total_ret,
        "sequential_max_dd_pct": dd,
    }


def main():
    print("[breakout] building daily panel...")
    t0 = time.time()
    panel = build_daily_panel()
    print(f"[breakout] panel {panel.shape} in {time.time()-t0:.1f}s")

    # Sweep N_breakout, init_stop, trail_stop, max_hold
    configs = [
        # (N, init_stop, trail_stop, max_hold, label)
        (10, 0.02, 0.05, 15, "N10_s2_t5_h15"),
        (20, 0.02, 0.05, 30, "N20_s2_t5_h30"),
        (20, 0.02, 0.08, 30, "N20_s2_t8_h30"),
        (20, 0.03, 0.05, 30, "N20_s3_t5_h30"),
        (30, 0.02, 0.05, 45, "N30_s2_t5_h45"),
        (60, 0.03, 0.08, 60, "N60_s3_t8_h60"),
        (20, 0.015, 0.04, 20, "N20_s15_t4_h20"),  # tight variant
        (20, 0.05, 0.10, 60, "N20_s5_t10_h60"),  # loose variant
    ]

    all_stats = []
    print("\n" + "=" * 100)
    print("ASYMMETRIC BREAKOUT SWEEP on UNIVERSE_50_LIQUID  (2025-01-01 -> 2026-04-22)")
    print("=" * 100)
    print(f"{'config':<20} {'n_tr':>5} {'hit%':>5} {'avg_w':>7} {'avg_l':>7} "
          f"{'ASYM':>5} {'kelly_g':>8} {'tot_ret%':>9}")
    print("-" * 100)

    for (N, istop, tstop, mh, lbl) in configs:
        t0 = time.time()
        trades = simulate_breakout(panel, N, istop, tstop, mh)
        s = stats(trades)
        s["config"] = lbl
        s["params"] = {"N": N, "init_stop": istop, "trail_stop": tstop, "max_hold": mh}
        all_stats.append(s)
        dt = time.time() - t0
        print(f"{lbl:<20} {s['n_trades']:>5} {s['hit_rate']*100:>4.0f} "
              f"{s['mean_win_pct']:>+6.2f} {s['mean_loss_pct']:>+6.2f} "
              f"{s['asymmetry_ratio']:>4.2f} {s['kelly_log_g_per_trade']:>+7.4f} "
              f"{s['total_ret_one_trade_at_a_time_pct']:>+8.1f}  ({dt:.0f}s)")

    # Save
    from datetime import datetime, timezone
    out_dir = ROOT / "logs" / "deployment" / str(datetime.now(timezone.utc).date())
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "asym_breakout_prototype.json"
    with open(out, "w") as f:
        json.dump({
            "run_utc": datetime.now(timezone.utc).isoformat(),
            "universe": "UNIVERSE_50_LIQUID",
            "cost_model": f"maker RT {MAKER_RT}%",
            "results": all_stats,
        }, f, indent=2, default=str)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
