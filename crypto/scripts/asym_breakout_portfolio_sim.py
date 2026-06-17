"""Realistic portfolio-level simulation of asymmetric breakout strategy.

Upgrades asym_breakout_prototype.py (per-trade sequential) to proper
portfolio accounting:
    - Multiple concurrent positions capped (e.g. max 10)
    - % of capital per trade (e.g. 5-10%)
    - Daily MTM equity curve
    - Per-asset duplicate-position prevention
    - Cost applied on entry AND exit
    - Sharpe, DD, Kelly-log, asymmetry all measured on daily returns

Baseline config from prototype winners:
    Family A-best-return: N=20 breakout + 3% init stop + 5% trail + 30d max hold

Produces daily_snapshot.csv so it can drop into portfolio_aggregator +
deploy_all_top_strats.py blend.
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
SEEDS_DIR = ROOT / "logs" / "paper_trader_v2" / "seeds"

sys.path.insert(0, str(ROOT / "src" / "strategy"))
from universe import UNIVERSE_50_LIQUID
UNIVERSE = set(UNIVERSE_50_LIQUID)

MAKER_RT = 0.08  # % per round trip
CAPITAL = 10000.0
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
        d = df.groupby("date").agg({"close": "last", "high": "max", "low": "min", "open": "first"}).reset_index()
        d["asset"] = asset
        rows.append(d)
    panel = pd.concat(rows, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["asset", "date"]).reset_index(drop=True)
    return panel


def run_portfolio_sim(panel, N_breakout=20, init_stop=0.03, trail_stop=0.05,
                      max_hold=30, max_concurrent=10, pct_per_trade=0.10,
                      skip_asset_reentry_days=5):
    """Daily-MTM portfolio sim with capital accounting.

    Returns dict with daily equity series + trade log + summary.
    """
    # Pre-compute breakout_high per asset
    panel = panel.copy()
    panel["breakout_high"] = panel.groupby("asset")["high"].transform(
        lambda s: s.shift(1).rolling(N_breakout, min_periods=N_breakout).max())

    # Get all dates in test window
    all_dates = sorted(panel[(panel["date"] >= TEST_START) &
                              (panel["date"] <= TEST_END)]["date"].unique())

    # Build per-date lookup: asset -> (close, high, low, breakout_high)
    lookups = {}
    for d, grp in panel.groupby("date"):
        lookups[d] = {row["asset"]: (row["close"], row["high"], row["low"],
                                      row["breakout_high"])
                       for _, row in grp.iterrows()}

    # State
    cash = CAPITAL
    positions = {}  # asset -> dict(entry_price, entry_date, entry_idx_in_log, size, stop, peak)
    last_exit = {}  # asset -> date (for reentry cooldown)
    trade_log = []
    daily_equity = []

    for d in all_dates:
        if d not in lookups:
            continue
        lkup = lookups[d]

        # 1) MTM existing positions + check exits
        closed_today = []
        for asset, pos in list(positions.items()):
            if asset not in lkup:
                continue
            close, high, low, _ = lkup[asset]
            # Update peak + trailing stop
            if high > pos["peak"]:
                pos["peak"] = high
                new_trail = pos["peak"] * (1 - trail_stop)
                if new_trail > pos["stop"]:
                    pos["stop"] = new_trail
            # Check exit: stop hit during day
            exit_price = None
            exit_reason = None
            if low <= pos["stop"]:
                exit_price = pos["stop"]
                exit_reason = "stop"
            elif (d - pos["entry_date"]).days >= max_hold:
                exit_price = close
                exit_reason = "max_hold"

            if exit_price is not None:
                size = pos["size"]
                pnl = size * (exit_price / pos["entry_price"] - 1)
                # Cost: exit side
                pnl -= size * (MAKER_RT / 200.0)
                cash += size + pnl
                net_pct = (exit_price / pos["entry_price"] - 1) * 100 - MAKER_RT
                trade_log.append({
                    "asset": asset,
                    "entry_date": pos["entry_date"],
                    "exit_date": d,
                    "hold_days": (d - pos["entry_date"]).days,
                    "entry_price": pos["entry_price"],
                    "exit_price": exit_price,
                    "peak_price": pos["peak"],
                    "size": size,
                    "pnl": pnl,
                    "net_ret_pct": net_pct,
                    "exit_reason": exit_reason,
                })
                closed_today.append(asset)
                last_exit[asset] = d

        for asset in closed_today:
            del positions[asset]

        # 2) New entries: scan universe for breakouts
        if len(positions) < max_concurrent:
            candidates = []
            for asset, (close, high, low, bhigh) in lkup.items():
                if asset in positions:
                    continue
                # Cooldown
                if asset in last_exit and (d - last_exit[asset]).days < skip_asset_reentry_days:
                    continue
                if pd.isna(bhigh) or close <= bhigh:
                    continue
                # Compute strength of breakout for ranking
                strength = (close - bhigh) / bhigh
                candidates.append((strength, asset, close))
            # Sort by strength desc; take top N to fill concurrent slots
            candidates.sort(reverse=True)
            slots_available = max_concurrent - len(positions)
            for strength, asset, close in candidates[:slots_available]:
                # Position sizing
                size = cash * pct_per_trade
                # Can we afford it?
                if size > cash:
                    break
                # Entry cost
                entry_cost = size * (MAKER_RT / 200.0)
                cash -= size  # lock capital
                cash -= entry_cost  # entry cost
                positions[asset] = {
                    "entry_date": d,
                    "entry_price": close,
                    "peak": close,
                    "stop": close * (1 - init_stop),
                    "size": size,
                }

        # 3) Mark-to-market equity
        position_value = 0.0
        for asset, pos in positions.items():
            if asset in lkup:
                mtm_price = lkup[asset][0]
                position_value += pos["size"] * (mtm_price / pos["entry_price"])
        equity = cash + position_value
        daily_equity.append({"date": d, "equity": equity, "cash": cash,
                              "n_positions": len(positions),
                              "position_value": position_value})

    eq_df = pd.DataFrame(daily_equity)
    trades_df = pd.DataFrame(trade_log)
    return eq_df, trades_df


def summarize(eq_df, trades_df, label=""):
    eq = eq_df["equity"].values
    if len(eq) < 2:
        return {"label": label, "status": "insufficient"}
    total = (eq[-1] / CAPITAL - 1) * 100
    daily_ret = np.diff(eq) / eq[:-1]
    days_span = (eq_df["date"].iloc[-1] - eq_df["date"].iloc[0]).days or 1
    cagr = ((eq[-1] / CAPITAL) ** (365 / days_span) - 1) * 100
    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(365) if daily_ret.std() > 0 else 0
    cum_max = np.maximum.accumulate(eq)
    dd = ((eq - cum_max) / cum_max).min() * 100

    # Trade asymmetry stats
    t_stats = {}
    if len(trades_df) > 0:
        r = trades_df["net_ret_pct"].values
        wins = r[r > 0]; losses = r[r <= 0]
        hit = len(wins) / len(r)
        asym = (wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else float("inf")
        kelly_g = (hit * np.log1p(wins.mean()/100 if len(wins) else 0)
                   + (1 - hit) * np.log1p(losses.mean()/100 if len(losses) else 0))
        t_stats = {
            "n_trades": len(r),
            "hit_rate": hit,
            "mean_trade_pct": r.mean(),
            "mean_win_pct": wins.mean() if len(wins) else 0,
            "mean_loss_pct": losses.mean() if len(losses) else 0,
            "asymmetry_ratio": asym,
            "kelly_log_g_per_trade": kelly_g,
            "avg_hold_days": trades_df["hold_days"].mean(),
            "exit_stop_pct": (trades_df["exit_reason"] == "stop").sum() / len(r) * 100,
        }

    return {
        "label": label,
        "days": len(eq),
        "cagr_pct": cagr,
        "sharpe": sharpe,
        "max_dd_pct": dd,
        "total_ret_pct": total,
        "end_nav": float(eq[-1]),
        **t_stats,
    }


def save_daily_snapshot(eq_df, trades_df, seed_name):
    """Write paper_trader_v2-compatible daily_snapshot.csv for aggregator."""
    out_dir = SEEDS_DIR / seed_name
    out_dir.mkdir(parents=True, exist_ok=True)
    snap = eq_df.rename(columns={"equity": "total_equity"}).copy()
    snap["bar_idx"] = np.arange(len(snap))
    snap["bar_ts"] = [int(pd.Timestamp(d).timestamp() * 1000) for d in snap["date"]]
    snap["swing_equity"] = snap["total_equity"]
    snap["short_equity"] = 0.0
    snap["total_ret_pct"] = (snap["total_equity"] / CAPITAL - 1) * 100
    snap["swing_ret_pct"] = snap["total_ret_pct"]
    snap["short_ret_pct"] = 0.0
    snap["swing_open_positions"] = snap["n_positions"].astype(int)
    cols = ["date", "bar_idx", "bar_ts", "total_equity", "swing_equity", "short_equity",
            "total_ret_pct", "swing_ret_pct", "short_ret_pct", "swing_open_positions"]
    snap[cols].to_csv(out_dir / "daily_snapshot.csv", index=False)
    trades_df.to_csv(out_dir / "trade_log.csv", index=False)
    print(f"  saved: {out_dir}/daily_snapshot.csv + trade_log.csv")


def main():
    print("[breakout-portfolio] building daily panel...")
    t0 = time.time()
    panel = build_daily_panel()
    print(f"[breakout-portfolio] panel {panel.shape} in {time.time()-t0:.1f}s")

    # Configs: vary max_concurrent + pct_per_trade
    configs = [
        # (N, istop, tstop, max_hold, max_concurrent, pct_per_trade, label)
        (20, 0.03, 0.05, 30, 10, 0.10, "brk_N20_max10_pct10"),
        (20, 0.03, 0.05, 30, 10, 0.05, "brk_N20_max10_pct5"),
        (20, 0.03, 0.05, 30, 5, 0.20, "brk_N20_max5_pct20"),
        (20, 0.05, 0.10, 60, 10, 0.10, "brk_N20_loose_max10_pct10"),
        (10, 0.02, 0.05, 15, 10, 0.10, "brk_N10_tight_max10_pct10"),
        (20, 0.03, 0.05, 30, 20, 0.05, "brk_N20_max20_pct5"),
    ]

    print("\n" + "=" * 100)
    print("PORTFOLIO-LEVEL BREAKOUT SIM")
    print("=" * 100)
    print(f"{'config':<35} {'days':>4} {'CAGR%':>7} {'Sh':>5} {'DD%':>6} "
          f"{'n_tr':>4} {'hit%':>4} {'asym':>5} {'kelly':>6}")
    print("-" * 100)
    all_results = []
    for (N, istop, tstop, mh, mc, pct, lbl) in configs:
        t0 = time.time()
        eq_df, trades_df = run_portfolio_sim(panel, N, istop, tstop, mh, mc, pct)
        s = summarize(eq_df, trades_df, label=lbl)
        s["config"] = {"N": N, "init_stop": istop, "trail_stop": tstop,
                       "max_hold": mh, "max_concurrent": mc, "pct_per_trade": pct}
        all_results.append(s)
        print(f"{lbl:<35} {s['days']:>4} {s['cagr_pct']:>+6.2f} "
              f"{s['sharpe']:>+4.2f} {s['max_dd_pct']:>+5.2f} "
              f"{s.get('n_trades', 0):>4} {s.get('hit_rate', 0)*100:>3.0f} "
              f"{s.get('asymmetry_ratio', 0):>4.2f} "
              f"{s.get('kelly_log_g_per_trade', 0):>+5.4f}  ({time.time()-t0:.0f}s)")
        # Save the 10-max + 10% variant as the canonical "pt_asym_breakout" seed
        if lbl == "brk_N20_max10_pct10":
            save_daily_snapshot(eq_df, trades_df, "pt_asym_breakout")

    # Save all results
    from datetime import datetime, timezone
    out_dir = ROOT / "logs" / "deployment" / str(datetime.now(timezone.utc).date())
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "asym_breakout_portfolio.json"
    with open(out, "w") as f:
        json.dump({
            "run_utc": datetime.now(timezone.utc).isoformat(),
            "universe": "UNIVERSE_50_LIQUID",
            "cost_model": f"maker RT {MAKER_RT}%",
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
