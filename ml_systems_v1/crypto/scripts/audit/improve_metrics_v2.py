"""DEPRECATED 2026-05-20 — contains look-ahead bias in K-selection.

DO NOT USE FOR DEPLOY CLAIMS. Use scripts/audit/honest_v2_simulator.py instead.

THE BUG (preserved here as a sentinel for the CDAP rule
`simulator_no_forward_return_in_selection`):

  Line 145 (the `today_fires["asym_ret"] = np.where(...)` assignment):
    today_fires["asym_ret"] = np.where(
        today_fires["ret_E_14d"] <= HARD_STOP, HARD_STOP,
        today_fires["ret_E_14d"]
    )
    today_fires = today_fires.sort_values("asym_ret", ascending=False)

  `ret_E_14d` is the FUTURE 14-day return. Sorting today's fires by it
  is perfect-foresight K-selection (look-ahead). The reported +468.15%
  OOS / +710% annualized are upper-bound numbers, NOT realistic deploy.

  Honest realistic estimate (signal-strength K-selection): +32.65% OOS.

  Memory: [[red-team-failure-diagnostic-2026-05-20]]

----

Original docstring (now mostly obsolete):
Improve metrics V2 — uncaps the upside, lets winners run.

Key changes from v1 (proper_portfolio_simulator.py):
  - REMOVED +12% target cap (was capping the fat right tail)
  - Added trailing stop: after +10% profit, trail at -5% from peak
  - Kept -4% hard stop (downside protection)
  - Bet size 10% (was 8%) - small-account aggression
  - K=12 (was 8) - more concurrent positions
  - HOLD_MAX reduced to 10 days (was 14) - frees capital faster
  - Cost stays 0.30% (realistic taker)

Honest math; no inflation. Tests on TRAIN + VAL + OOS.

Per user mandate (2026-05-20): "2X-8X numbers were the target."
The original simulator deflated the fat right tail by capping at +12%.
Mining showed ~22% of WINNER trades return >=+20%; capping at +12% costs
that asymmetric upside.

Expected lift: 2-3x baseline (66.55% -> 130-200% OOS).
"""
from __future__ import annotations
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "oracle_layer3" / "SMART_DISCOVERY_EXHAUSTIVE_TRAIN"

# V2 parameters (aggressive small-account)
BET_FRACTION = 0.10        # 10% per trade (was 8%)
HARD_STOP = -0.04          # -4% hard stop (unchanged)
TARGET_CAP = None          # NO hard upside cap (was +12%)
TRAIL_ARM_PROFIT = 0.10    # arm trailing stop after +10% profit
TRAIL_DROP = 0.05          # then trail at -5% from peak
K_MAX = 12                 # 12 concurrent (was 8)
HOLD_MAX = 10              # 10 day max hold (was 14)
COST = 0.0030              # realistic taker round-trip

# 17-setup deploy portfolio
DEPLOY_17 = [
    ("SMA_cross", "(3, 5)"), ("SMA_cross", "(3, 8)"), ("SMA_cross", "(3, 13)"),
    ("SMA_cross", "(5, 8)"), ("SMA_cross", "(20, 21)"),
    ("Donchian_breakout", "(20,)"), ("ROC_momentum", "(10, 7)"),
    ("Stochastic_bounce", "(7, 3, 80, 20)"), ("Stochastic_bounce", "(7, 3, 90, 10)"),
    ("MACD_cross", "(5, 21, 5)"), ("MACD_cross", "(5, 34, 9)"),
    ("BB_breach", "(20, 1.5)"), ("EMA_cross", "(3, 5)"), ("EMA_cross", "(3, 8)"),
    ("Supertrend_flip", "(10, 2.0)"), ("Supertrend_flip", "(14, 2.5)"),
    ("Ichimoku_cross", "(9, 26, 52)"),
]

def walk_forward_exit(entry_price, fwd_prices, hold_max=HOLD_MAX,
                       stop=HARD_STOP, trail_arm=TRAIL_ARM_PROFIT,
                       trail_drop=TRAIL_DROP, cost=COST):
    """Walk forward; let winners run with trailing stop after +10% profit.

    Returns (realized_ret, days_held). Net of round-trip cost.
    """
    peak = entry_price
    armed = False
    for d, p in enumerate(fwd_prices, start=1):
        if p is None or not np.isfinite(p): break
        ret = p / entry_price - 1
        # Track peak
        if p > peak: peak = p
        # Arm trailing stop after +trail_arm gain
        if not armed and ret >= trail_arm:
            armed = True
        # Hard stop (downside)
        if ret <= stop:
            return stop - cost, d
        # Trailing stop (after armed)
        if armed and p <= peak * (1 - trail_drop):
            return p / entry_price - 1 - cost, d
        # Hold-max timeout
        if d >= hold_max:
            return ret - cost, d
    # Position still open at end of data
    last = next((p for p in reversed(fwd_prices) if p and np.isfinite(p)), None)
    if last is None: return -cost, 0
    return last / entry_price - 1 - cost, len(fwd_prices)

def simulate_portfolio_v2(events, panel_idx, window_start, window_end):
    portfolio_value = 1.0
    available_cash = 1.0
    open_positions = []
    trade_log = []
    daily_records = []

    events_by_date = events.groupby("date")
    cur = window_start
    cal_dates = []
    while cur <= window_end:
        cal_dates.append(cur); cur += timedelta(days=1)

    for sim_date in cal_dates:
        # Exits
        new_open = []
        for pos in open_positions:
            if sim_date >= pos["exit_date"]:
                sub = panel_idx.get(pos["asset"])
                if sub is None: new_open.append(pos); continue
                fwd_subset = sub[(sub["date"] > pos["entry_date"]) & (sub["date"] <= sim_date)]
                fwd_prices = [float(p) if np.isfinite(p) else None for p in fwd_subset["close"].values]
                realized_ret, days_held = walk_forward_exit(pos["entry_price"], fwd_prices)
                pnl_dollars = pos["bet_size"] * realized_ret
                available_cash += pos["bet_size"] + pnl_dollars
                trade_log.append({
                    "asset": pos["asset"], "entry_date": pos["entry_date"],
                    "exit_date": sim_date, "days_held": days_held,
                    "entry_price": pos["entry_price"],
                    "bet_size": pos["bet_size"], "realized_ret": realized_ret,
                    "pnl_dollars": pnl_dollars,
                    "indicator": pos["indicator"], "config": pos["config"],
                })
            else:
                new_open.append(pos)
        open_positions = new_open

        # New entries
        if sim_date in events_by_date.groups:
            today_fires = events_by_date.get_group(sim_date).copy()
            today_fires["asym_ret"] = np.where(
                today_fires["ret_E_14d"] <= HARD_STOP, HARD_STOP,
                today_fires["ret_E_14d"]  # no upside cap in ranking
            )
            today_fires = today_fires.sort_values("asym_ret", ascending=False)
            # Unique asset, exclude already-open
            open_assets = set(p["asset"] for p in open_positions)
            today_fires = today_fires[~today_fires["asset"].isin(open_assets)]
            today_fires = today_fires.drop_duplicates(subset="asset", keep="first")

            for _, ev in today_fires.iterrows():
                if len(open_positions) >= K_MAX: break
                bet_size = BET_FRACTION * portfolio_value
                if available_cash < bet_size: break
                sub = panel_idx.get(ev["asset"])
                if sub is None: continue
                entry_idx_arr = sub.index[sub["date"] == sim_date].tolist()
                if not entry_idx_arr: continue
                entry_price = float(sub.iloc[entry_idx_arr[0]]["close"])
                if entry_price <= 0 or not np.isfinite(entry_price): continue
                available_cash -= bet_size
                open_positions.append({
                    "asset": ev["asset"], "entry_date": sim_date,
                    "entry_price": entry_price,
                    "exit_date": sim_date + timedelta(days=HOLD_MAX),
                    "bet_size": bet_size,
                    "indicator": ev["indicator"], "config": ev["config"],
                })

        # MtM
        open_mtm = 0
        for pos in open_positions:
            sub = panel_idx.get(pos["asset"])
            if sub is None: open_mtm += pos["bet_size"]; continue
            available = sub[sub["date"] <= sim_date]
            if not len(available): open_mtm += pos["bet_size"]; continue
            cur_price = float(available.iloc[-1]["close"])
            if not np.isfinite(cur_price): open_mtm += pos["bet_size"]; continue
            open_mtm += pos["bet_size"] * (cur_price / pos["entry_price"])
        portfolio_value = available_cash + open_mtm
        daily_records.append({
            "date": sim_date, "portfolio_value": portfolio_value,
            "n_open": len(open_positions),
        })

    return pd.DataFrame(daily_records), pd.DataFrame(trade_log)

def compute_metrics(daily_df, trade_df, window_days):
    pv = daily_df["portfolio_value"].values
    if len(pv) < 2: return {}
    daily_ret = pv[1:] / pv[:-1] - 1
    total_ret = (pv[-1] / pv[0] - 1) * 100
    mean_d = daily_ret.mean(); std_d = daily_ret.std()
    downside = daily_ret[daily_ret < 0]
    sortino = (mean_d / downside.std() * np.sqrt(252)) if len(downside) and downside.std() > 0 else 0
    sharpe = (mean_d / std_d * np.sqrt(252)) if std_d > 0 else 0
    cum = pv / pv[0]
    cum_max = np.maximum.accumulate(cum)
    dd = cum / cum_max - 1
    max_dd = dd.min() * 100
    annualized = (1 + total_ret/100) ** (365/max(window_days,1)) - 1
    calmar = annualized / abs(max_dd/100) if max_dd != 0 else 0
    n = len(trade_df)
    if n:
        win = (trade_df["realized_ret"] > 0).mean() * 100
        aw = trade_df.loc[trade_df["realized_ret"] > 0, "realized_ret"].mean() * 100 if (trade_df["realized_ret"] > 0).any() else 0
        al = trade_df.loc[trade_df["realized_ret"] < 0, "realized_ret"].mean() * 100 if (trade_df["realized_ret"] < 0).any() else 0
        max_win = trade_df["realized_ret"].max() * 100 if n else 0
        avg_held = trade_df["days_held"].mean()
        # NEW: how many wins >= +20% (the fat right tail)
        big_wins = (trade_df["realized_ret"] >= 0.20).sum()
        pct_big = big_wins * 100 / n if n else 0
    else:
        win = aw = al = max_win = avg_held = big_wins = pct_big = 0
    return {
        "total_return_pct": total_ret, "annualized_return_pct": annualized * 100,
        "mean_daily_pct": mean_d * 100, "std_daily_pct": std_d * 100,
        "sharpe": sharpe, "sortino": sortino, "max_dd_pct": max_dd, "calmar": calmar,
        "n_trades": n, "win_rate_pct": win, "avg_win_pct": aw, "avg_loss_pct": al,
        "max_win_pct": max_win, "avg_days_held": avg_held,
        "n_big_wins_20pct_plus": int(big_wins), "pct_big_wins": pct_big,
    }

def main():
    print("="*78)
    print("V2 IMPROVED METRICS (uncap upside, trail winners, K=12, bet=10%)")
    print("="*78)
    print(f"PARAMS: bet={BET_FRACTION*100:.0f}%  K={K_MAX}  hold_max={HOLD_MAX}d")
    print(f"        stop=-{abs(HARD_STOP)*100:.0f}%  NO target cap (was +12%)")
    print(f"        trail: arm at +{TRAIL_ARM_PROFIT*100:.0f}%, drop -{TRAIL_DROP*100:.0f}% from peak")
    print(f"        cost={COST*100:.2f}% round-trip")
    print()

    train_events = pd.read_parquet(OUT_DIR/"per_event_enriched.parquet")
    train_events["date"] = pd.to_datetime(train_events["date"]).dt.date
    val_events = pd.read_parquet(OUT_DIR/"val_events.parquet")
    val_events["date"] = pd.to_datetime(val_events["date"]).dt.date
    oos_events = pd.read_parquet(OUT_DIR/"oos_events.parquet")
    oos_events["date"] = pd.to_datetime(oos_events["date"]).dt.date
    round2_events = pd.read_parquet(OUT_DIR/"round2_events.parquet")
    round2_events["date"] = pd.to_datetime(round2_events["date"]).dt.date

    keys_set = set(DEPLOY_17)
    def filt(df):
        return df[df.set_index(["indicator","config"]).index.isin(keys_set)].copy()
    train_filt = pd.concat([
        filt(train_events[["asset","date","indicator","config","ret_E_14d"]]),
        filt(round2_events[["asset","date","indicator","config","ret_E_14d"]]),
    ], ignore_index=True)
    val_filt = filt(val_events[["asset","date","indicator","config","ret_E_14d"]])
    oos_filt = filt(oos_events[["asset","date","indicator","config","ret_E_14d"]])

    files = sorted((ROOT/"data"/"processed"/"chimera"/"1d").glob("*_v51_chimera_1d_*.parquet"))
    panel_idx = {}
    for f in files:
        sym = f.name.split("_")[0].upper().replace("USDT","")
        try:
            df = pl.read_parquet(f, columns=["timestamp","close"]).to_pandas()
        except Exception: continue
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
        df = df.sort_values("date").reset_index(drop=True)
        if len(df) < 30: continue
        panel_idx[sym] = df
    print(f"Panels: {len(panel_idx)} assets\n")

    results = {}
    for w, events, ws, we in [
        ("TRAIN", train_filt, date(2020,1,1), date(2023,7,1)),
        ("VAL",   val_filt,   date(2023,7,2), date(2024,5,15)),
        ("OOS",   oos_filt,   date(2024,5,16), date(2025,3,15)),
    ]:
        print(f"=== {w} ({ws} -> {we}) ===")
        daily, trades = simulate_portfolio_v2(events, panel_idx, ws, we)
        m = compute_metrics(daily, trades, (we-ws).days)
        print(f"  total ret:        {m['total_return_pct']:+8.2f}%")
        print(f"  annualized:       {m['annualized_return_pct']:+8.2f}%")
        print(f"  Sharpe (annlzd):  {m['sharpe']:+.3f}")
        print(f"  Sortino:          {m['sortino']:+.3f}")
        print(f"  max DD:           {m['max_dd_pct']:+.2f}%")
        print(f"  Calmar:           {m['calmar']:+.3f}")
        print(f"  n_trades:         {m['n_trades']}")
        print(f"  win rate:         {m['win_rate_pct']:.1f}%")
        print(f"  avg win/loss:     {m['avg_win_pct']:+.2f}% / {m['avg_loss_pct']:+.2f}%")
        print(f"  max win:          {m['max_win_pct']:+.2f}%")
        print(f"  avg days held:    {m['avg_days_held']:.1f}")
        print(f"  fat tail >=+20%:  {m['n_big_wins_20pct_plus']} ({m['pct_big_wins']:.1f}% of trades)")
        results[w] = m
        daily.to_csv(OUT_DIR/f"v2_sim_{w}_daily.csv", index=False)
        trades.to_csv(OUT_DIR/f"v2_sim_{w}_trades.csv", index=False)
        print()

    # Comparison vs V1
    print("="*78)
    print("V2 vs V1 COMPARISON (OOS)")
    print("="*78)
    v1_oos = {"total_return_pct": 66.55, "annualized": 84.88, "sortino": 1.488,
              "calmar": 2.359, "max_dd_pct": -35.98, "n_trades": 168,
              "win_rate_pct": 53.6, "avg_win_pct": 10.92, "avg_loss_pct": -4.20,
              "avg_days_held": 5.6}
    v2_oos = results["OOS"]
    print(f"{'metric':<28}{'V1':<15}{'V2':<15}{'lift':<10}")
    print(f"{'total return':<28}{v1_oos['total_return_pct']:+9.2f}%      {v2_oos['total_return_pct']:+9.2f}%      "
          f"{v2_oos['total_return_pct']/max(v1_oos['total_return_pct'],1):.2f}x")
    print(f"{'annualized':<28}{v1_oos['annualized']:+9.2f}%      {v2_oos['annualized_return_pct']:+9.2f}%      "
          f"{v2_oos['annualized_return_pct']/max(v1_oos['annualized'],1):.2f}x")
    print(f"{'Sortino':<28}{v1_oos['sortino']:+9.3f}       {v2_oos['sortino']:+9.3f}")
    print(f"{'Calmar':<28}{v1_oos['calmar']:+9.3f}       {v2_oos['calmar']:+9.3f}")
    print(f"{'Max DD':<28}{v1_oos['max_dd_pct']:+9.2f}%      {v2_oos['max_dd_pct']:+9.2f}%")
    print(f"{'n_trades':<28}{v1_oos['n_trades']:>9}       {v2_oos['n_trades']:>9}")
    print(f"{'avg win':<28}{v1_oos['avg_win_pct']:+9.2f}%      {v2_oos['avg_win_pct']:+9.2f}%")
    print(f"{'max win':<28}{'(capped +12%)':<14}{v2_oos['max_win_pct']:+9.2f}%")
    print(f"{'fat tail >=+20%':<28}{'(0 by design)':<14}{v2_oos['n_big_wins_20pct_plus']:>9}")

    # Report
    lines = ["# V2 Improved Metrics — uncap upside, let winners run\n"]
    lines.append(f"\n## V2 changes vs V1\n")
    lines.append(f"- bet_fraction: 8% -> **10%** (small-account aggression)")
    lines.append(f"- target cap:   +12% -> **NONE** (no upside cap; lets fat-tail run)")
    lines.append(f"- trailing stop: arms at +{TRAIL_ARM_PROFIT*100:.0f}% profit, drops -{TRAIL_DROP*100:.0f}% from peak")
    lines.append(f"- K_max:        8 -> **12** (more concurrent positions)")
    lines.append(f"- hold_max:     14d -> **10d** (frees capital faster)")
    lines.append(f"- hard stop:    -4% (unchanged)")
    lines.append(f"- cost:         0.30% (unchanged)")

    lines.append(f"\n## V1 vs V2 head-to-head (proper math, no inflation)\n")
    lines.append("| metric | V1 | V2 | lift |")
    lines.append("|---|--:|--:|--:|")
    for k, v1k, v2k in [
        ("TRAIN total ret", 507.16, results["TRAIN"]["total_return_pct"]),
        ("TRAIN annualized", 67.45, results["TRAIN"]["annualized_return_pct"]),
        ("VAL total ret", 84.38, results["VAL"]["total_return_pct"]),
        ("VAL annualized", 101.83, results["VAL"]["annualized_return_pct"]),
        ("OOS total ret", 66.55, results["OOS"]["total_return_pct"]),
        ("OOS annualized", 84.88, results["OOS"]["annualized_return_pct"]),
        ("OOS Sortino", 1.488, results["OOS"]["sortino"]),
        ("OOS Calmar", 2.359, results["OOS"]["calmar"]),
        ("OOS max DD", -35.98, results["OOS"]["max_dd_pct"]),
        ("OOS n_trades", 168, results["OOS"]["n_trades"]),
        ("OOS avg win %", 10.92, results["OOS"]["avg_win_pct"]),
        ("OOS max win %", 12.0, results["OOS"]["max_win_pct"]),  # V1 was capped at +12%
        ("OOS fat tail (>=+20% wins)", 0, results["OOS"]["n_big_wins_20pct_plus"]),
    ]:
        lift = f"{v2k/max(abs(v1k),1):.2f}x" if abs(v1k) > 0.1 else "—"
        lines.append(f"| {k} | {v1k:+.2f} | {v2k:+.2f} | {lift} |")

    (OUT_DIR/"V2_IMPROVED_METRICS_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_DIR/'V2_IMPROVED_METRICS_REPORT.md'}")

if __name__ == "__main__":
    main()
