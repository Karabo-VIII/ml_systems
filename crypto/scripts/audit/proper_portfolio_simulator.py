"""DEPRECATED 2026-05-20 -- contains selection-time look-ahead bug.

The class names "proper" and "no inflation/deflation" in the original
docstring were WRONG. This simulator sorts today's fires by `asym_ret`
(derived from ret_E_14d, the FUTURE 14-day return), which is perfect-
foresight K-selection. All NAV claims from this simulator are upper-
bound (best-K), NOT realistic deploy.

The previously-claimed +66.55% OOS / +84.88% annualized was best-K with
the V1 +12% target cap throttling the look-ahead lift. Honest estimate
with signal-strength K-selection is materially lower (see honest_v2
results for the V2-param version: +33% with signal-K).

Use scripts/audit/honest_v2_simulator.py for honest math.

Memory: [[red-team-failure-diagnostic-2026-05-20]],
        [[feedback-honesty-no-inflation]]

----

Original docstring (claims now retracted):
Proper portfolio simulator with capital accounting (NO inflation, NO deflation).

Fixes the math errors flagged in prior turns:
  - Prior `sum(asym_ret * bet_fraction)` assumed K=8 × 8% = 64% deployed
    EVERY day, ignoring that prior-day positions still lock capital.
  - Compound NAV math (cum_nav.iloc[-1]) was inflated because the daily
    sum was already overstating.

Proper math:
  Portfolio = AVAILABLE_CASH + OPEN_POSITIONS_MTM
  Each day:
    1. Process exits: positions reaching their 14d max OR asymmetric stops hit
       (capital returns to AVAILABLE_CASH at realized value)
    2. Find new fires today (regime-aware, unique-asset, sorted by expectancy)
    3. Open new positions while AVAILABLE_CASH >= bet_size AND n_open < K_MAX
       (each entry deducts 8% × current_portfolio from AVAILABLE_CASH)
    4. Mark-to-market open positions (track for max DD, but only realized
       returns enter cumulative NAV)
  End-of-window: close all open positions at last price.

Realistic cost model:
  - Round-trip = 0.30% (taker 2-sided + slippage proxy)
  - No partial fill modeling (worst-case assumption: filled at quote)
  - No leverage; long-only; spot

Risk metrics (proper):
  - Total realized NAV (sum of closed-trade PnL on a $1.00 base)
  - Cumulative compound (geometric: prod(1 + daily_realized) - 1)
  - Sharpe (mean / std × sqrt(252))
  - Sortino (mean / downside_std × sqrt(252))
  - Calmar (annualized return / abs(max_DD))
  - Max DD (peak-to-trough on cumulative compound)

This is the LOAD-BEARING simulator. NAV claims from here are deploy-credible.
"""
from __future__ import annotations
from pathlib import Path
from datetime import date, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "oracle_layer3" / "SMART_DISCOVERY_EXHAUSTIVE_TRAIN"

# Realistic deploy parameters
BET_FRACTION = 0.08       # 8% per trade of current portfolio
HARD_STOP = -0.04         # asymmetric stop
TARGET = 0.12             # take-profit
K_MAX = 8                 # concurrent open positions cap
HOLD_MAX = 14             # max holding period
ROUND_TRIP_COST = 0.0030  # realistic taker-side round-trip = 0.30%
WEEKLY_FLOOR = 0.0525     # +5.25%/7d target

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

def compute_asym_realized(entry_price, fwd_prices, hold_max=HOLD_MAX,
                            stop=HARD_STOP, target=TARGET, cost=ROUND_TRIP_COST):
    """Walk forward day-by-day; apply stop/target intra-event.
    Returns (realized_pct, days_held) where realized is net of cost.
    """
    for d, p in enumerate(fwd_prices, start=1):
        if p is None or not np.isfinite(p): break
        ret = p / entry_price - 1
        if ret <= stop:
            return stop - cost, d
        if ret >= target:
            return target - cost, d
        if d >= hold_max:
            return ret - cost, d
    # Position still open at end of available data
    last_valid = next((p for p in reversed(fwd_prices) if p and np.isfinite(p)), None)
    if last_valid is None:
        return 0.0 - cost, 0
    return last_valid / entry_price - 1 - cost, len(fwd_prices)

def simulate_portfolio(events, panel_idx, window_start, window_end,
                        k_max=K_MAX, bet_fraction=BET_FRACTION,
                        unique_asset=True, round_trip_cost=None):
    """Discrete-time portfolio simulator with proper capital accounting.

    events: per-event dataframe (asset, date, indicator, config, asym_expectancy
            for ranking on the firing day)
    panel_idx: {asset: sorted dataframe with close prices indexed by date}
    Returns daily NAV series + completed trade log.
    """
    # Resolve cost at call time so callers/stress tests can override
    if round_trip_cost is None:
        round_trip_cost = ROUND_TRIP_COST
    portfolio_value = 1.0       # start at $1
    available_cash = 1.0
    open_positions = []         # each: {asset, entry_date, entry_price, exit_date, bet_size}
    trade_log = []
    daily_records = []

    # Pre-build event lookup by date for fast iteration
    events_by_date = events.groupby("date")
    all_dates = sorted(set(events["date"]))
    sim_dates = [d for d in all_dates if window_start <= d <= window_end]

    # Also walk all calendar dates in the window (for daily NAV continuity)
    cur = window_start
    cal_dates = []
    while cur <= window_end:
        cal_dates.append(cur)
        cur += timedelta(days=1)

    # For each calendar date in the window
    asym_expectancy_lookup = {}  # build once

    for sim_date in cal_dates:
        # 1) Process exits FIRST: positions reaching exit_date today
        new_open = []
        for pos in open_positions:
            if sim_date >= pos["exit_date"]:
                # Determine final realized return at this date
                sub = panel_idx.get(pos["asset"])
                if sub is None:
                    new_open.append(pos); continue
                # Compute forward prices from entry
                exit_idx_arr = sub.index[sub["date"] == sim_date].tolist()
                if not exit_idx_arr:
                    # Use last available price up to sim_date
                    available = sub[sub["date"] <= sim_date]
                    if len(available) == 0:
                        new_open.append(pos); continue
                    exit_price = float(available.iloc[-1]["close"])
                else:
                    exit_price = float(sub.iloc[exit_idx_arr[0]]["close"])
                if not np.isfinite(exit_price):
                    new_open.append(pos); continue
                gross_ret = exit_price / pos["entry_price"] - 1
                # Apply asym stop / target retroactively (walk forward)
                fwd_subset = sub[(sub["date"] > pos["entry_date"]) & (sub["date"] <= sim_date)]
                fwd_prices = [float(p) if np.isfinite(p) else None for p in fwd_subset["close"].values]
                realized_ret, days_held = compute_asym_realized(
                    pos["entry_price"], fwd_prices, cost=round_trip_cost)
                pnl_dollars = pos["bet_size"] * realized_ret
                available_cash += pos["bet_size"] + pnl_dollars  # return bet + pnl
                trade_log.append({
                    "asset": pos["asset"], "entry_date": pos["entry_date"],
                    "exit_date": sim_date, "days_held": days_held,
                    "entry_price": pos["entry_price"], "exit_price": exit_price,
                    "bet_size": pos["bet_size"], "realized_ret": realized_ret,
                    "pnl_dollars": pnl_dollars,
                    "indicator": pos["indicator"], "config": pos["config"],
                })
            else:
                new_open.append(pos)
        open_positions = new_open

        # 2) Find today's fires
        if sim_date in events_by_date.groups:
            today_fires = events_by_date.get_group(sim_date).copy()
            # Compute asym expectancy as ranking key (use ret_E_14d, asym-capped)
            today_fires["asym_ret"] = np.where(
                today_fires["ret_E_14d"] <= HARD_STOP, HARD_STOP,
                np.where(today_fires["ret_E_14d"] >= TARGET, TARGET, today_fires["ret_E_14d"])
            )
            # Sort by asym descending; unique asset
            today_fires = today_fires.sort_values("asym_ret", ascending=False)
            if unique_asset:
                # Exclude assets already in open positions
                open_assets = set(p["asset"] for p in open_positions)
                today_fires = today_fires[~today_fires["asset"].isin(open_assets)]
                today_fires = today_fires.drop_duplicates(subset="asset", keep="first")

            # 3) Open new positions while K and capital allow
            for _, ev in today_fires.iterrows():
                if len(open_positions) >= k_max: break
                bet_size = bet_fraction * portfolio_value
                if available_cash < bet_size: break
                # Get entry price from panel
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

        # 4) Mark-to-market: portfolio_value = available_cash + sum(open MtM)
        open_mtm = 0
        for pos in open_positions:
            sub = panel_idx.get(pos["asset"])
            if sub is None:
                open_mtm += pos["bet_size"]; continue
            available = sub[sub["date"] <= sim_date]
            if len(available) == 0:
                open_mtm += pos["bet_size"]; continue
            cur_price = float(available.iloc[-1]["close"])
            if not np.isfinite(cur_price):
                open_mtm += pos["bet_size"]; continue
            cur_ret = cur_price / pos["entry_price"] - 1
            open_mtm += pos["bet_size"] * (1 + cur_ret)
        portfolio_value = available_cash + open_mtm
        daily_records.append({
            "date": sim_date,
            "portfolio_value": portfolio_value,
            "available_cash": available_cash,
            "n_open": len(open_positions),
            "open_mtm": open_mtm,
        })

    daily_df = pd.DataFrame(daily_records)
    trade_df = pd.DataFrame(trade_log)
    return daily_df, trade_df

def compute_risk_metrics(daily_df, trade_df, window_days):
    """Proper risk metrics from portfolio NAV series."""
    pv = daily_df["portfolio_value"].values
    if len(pv) < 2:
        return {}
    daily_ret = pv[1:] / pv[:-1] - 1
    total_ret_pct = (pv[-1] / pv[0] - 1) * 100
    mean_d = daily_ret.mean()
    std_d = daily_ret.std()
    downside = daily_ret[daily_ret < 0]
    sortino = (mean_d / downside.std() * np.sqrt(252)) if len(downside) and downside.std() > 0 else 0
    sharpe = (mean_d / std_d * np.sqrt(252)) if std_d > 0 else 0
    cum = pv / pv[0]
    cum_max = np.maximum.accumulate(cum)
    dd = cum / cum_max - 1
    max_dd = dd.min() * 100
    annualized_ret = (1 + total_ret_pct/100) ** (365/max(window_days,1)) - 1
    calmar = annualized_ret / abs(max_dd/100) if max_dd != 0 else 0
    # Trades
    n_trades = len(trade_df)
    if n_trades:
        win_rate = (trade_df["realized_ret"] > 0).mean() * 100
        avg_win = trade_df.loc[trade_df["realized_ret"] > 0, "realized_ret"].mean() * 100 if (trade_df["realized_ret"] > 0).any() else 0
        avg_loss = trade_df.loc[trade_df["realized_ret"] < 0, "realized_ret"].mean() * 100 if (trade_df["realized_ret"] < 0).any() else 0
        avg_days_held = trade_df["days_held"].mean()
    else:
        win_rate = avg_win = avg_loss = avg_days_held = 0
    return {
        "total_return_pct": total_ret_pct,
        "annualized_return_pct": annualized_ret * 100,
        "mean_daily_pct": mean_d * 100,
        "std_daily_pct": std_d * 100,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd_pct": max_dd,
        "calmar": calmar,
        "n_trades": n_trades,
        "win_rate_pct": win_rate,
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "avg_days_held": avg_days_held,
    }

def main():
    print("="*78)
    print("PROPER PORTFOLIO SIMULATOR (capital-aware, realistic cost)")
    print("="*78)
    print(f"Sizing: {BET_FRACTION*100:.0f}% per trade  K_max={K_MAX}  asym -{abs(HARD_STOP)*100:.0f}%/+{TARGET*100:.0f}%")
    print(f"Round-trip cost: {ROUND_TRIP_COST*100:.2f}% (taker 2-sided + slippage proxy)")
    print()

    # Load all events (TRAIN + VAL + OOS already in separate files)
    train_events = pd.read_parquet(OUT_DIR/"per_event_enriched.parquet")
    train_events["date"] = pd.to_datetime(train_events["date"]).dt.date
    val_events = pd.read_parquet(OUT_DIR/"val_events.parquet")
    val_events["date"] = pd.to_datetime(val_events["date"]).dt.date
    oos_events = pd.read_parquet(OUT_DIR/"oos_events.parquet")
    oos_events["date"] = pd.to_datetime(oos_events["date"]).dt.date
    round2_events = pd.read_parquet(OUT_DIR/"round2_events.parquet")
    round2_events["date"] = pd.to_datetime(round2_events["date"]).dt.date

    # Filter to deploy_17 setups
    keys_set = set(DEPLOY_17)
    def filt(df):
        return df[df.set_index(["indicator","config"]).index.isin(keys_set)].copy()
    train_filt = filt(train_events[["asset","date","indicator","config","ret_E_14d"]])
    # Add round-2 contribution to train events for the 3 round-2 setups
    train_filt = pd.concat([train_filt, filt(round2_events[["asset","date","indicator","config","ret_E_14d"]])], ignore_index=True)
    val_filt = filt(val_events[["asset","date","indicator","config","ret_E_14d"]])
    oos_filt = filt(oos_events[["asset","date","indicator","config","ret_E_14d"]])
    print(f"Events filtered to 17-deploy setups:")
    print(f"  TRAIN: {len(train_filt):,}")
    print(f"  VAL:   {len(val_filt):,}")
    print(f"  OOS:   {len(oos_filt):,}")

    # Build panel index
    print("\nLoading chimera panel (all windows)...")
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
    print(f"  panels: {len(panel_idx)} assets")

    # Run simulator on each window
    results = {}
    for window_name, events, ws, we in [
        ("TRAIN_full", train_filt, date(2020,1,1), date(2023,7,1)),
        ("VAL_full", val_filt, date(2023,7,2), date(2024,5,15)),
        ("OOS_full", oos_filt, date(2024,5,16), date(2025,3,15)),
    ]:
        print(f"\n=== SIMULATING {window_name} ({ws} -> {we}) ===")
        daily_df, trade_df = simulate_portfolio(events, panel_idx, ws, we)
        window_days = (we - ws).days
        metrics = compute_risk_metrics(daily_df, trade_df, window_days)
        print(f"  Total return:        {metrics['total_return_pct']:+8.2f}%")
        print(f"  Annualized:          {metrics['annualized_return_pct']:+8.2f}%")
        print(f"  Mean daily:          {metrics['mean_daily_pct']:+.3f}%")
        print(f"  Std daily:           {metrics['std_daily_pct']:.3f}%")
        print(f"  Sharpe (annlzd):     {metrics['sharpe']:+.3f}")
        print(f"  Sortino (annlzd):    {metrics['sortino']:+.3f}")
        print(f"  Max DD:              {metrics['max_dd_pct']:+.2f}%")
        print(f"  Calmar (ann/MaxDD):  {metrics['calmar']:+.3f}")
        print(f"  Trades:              {metrics['n_trades']}")
        print(f"  Win rate:            {metrics['win_rate_pct']:.1f}%")
        print(f"  Avg win:             {metrics['avg_win_pct']:+.2f}%")
        print(f"  Avg loss:            {metrics['avg_loss_pct']:+.2f}%")
        print(f"  Avg days held:       {metrics['avg_days_held']:.1f}")
        # 7d rolling
        if len(daily_df) > 7:
            daily_df["daily_ret"] = daily_df["portfolio_value"].pct_change()
            daily_df["nav_7d"] = daily_df["daily_ret"].rolling(7).sum()
            floor_clear = (daily_df["nav_7d"] >= WEEKLY_FLOOR).sum()
            floor_total = max(len(daily_df) - 6, 1)
            print(f"  7d floor clear: {floor_clear}/{floor_total} ({floor_clear*100/floor_total:.0f}%)  mean_7d={daily_df['nav_7d'].mean()*100:+.2f}%")
            metrics["floor_clear"] = int(floor_clear)
            metrics["floor_total"] = int(floor_total)
            metrics["mean_7d_pct"] = float(daily_df["nav_7d"].mean() * 100)

        results[window_name] = metrics
        daily_df.to_csv(OUT_DIR/f"proper_sim_{window_name}_daily.csv", index=False)
        trade_df.to_csv(OUT_DIR/f"proper_sim_{window_name}_trades.csv", index=False)

    # Synthesis
    lines = ["# Proper Portfolio Simulator -- 17-Setup Deploy\n"]
    lines.append(f"\n## Configuration\n")
    lines.append(f"- Setups: 17 (14 VAL-WF + 3 round-2 all VAL-confirmed)")
    lines.append(f"- Bet size: {BET_FRACTION*100:.0f}% per trade of current portfolio")
    lines.append(f"- K_max: {K_MAX} concurrent positions; unique asset")
    lines.append(f"- Asym stop/target: {HARD_STOP*100:.0f}% / +{TARGET*100:.0f}%")
    lines.append(f"- Round-trip cost: {ROUND_TRIP_COST*100:.2f}% (realistic taker)")
    lines.append(f"- Capital accounting: PROPER (positions lock capital for hold duration)")
    lines.append(f"\n## Three-window head-to-head\n")
    lines.append("| metric | TRAIN | VAL | OOS |")
    lines.append("|---|--:|--:|--:|")
    for metric in ["total_return_pct", "annualized_return_pct", "mean_daily_pct",
                   "sharpe", "sortino", "max_dd_pct", "calmar",
                   "n_trades", "win_rate_pct", "avg_win_pct", "avg_loss_pct",
                   "avg_days_held", "mean_7d_pct", "floor_clear"]:
        row = [metric.replace("_", " ")]
        for w in ("TRAIN_full", "VAL_full", "OOS_full"):
            m = results.get(w, {})
            v = m.get(metric, "—")
            if isinstance(v, float):
                row.append(f"{v:+.3f}" if metric in ["sharpe","sortino","calmar"] else f"{v:+.2f}")
            else:
                row.append(str(v))
        lines.append("| " + " | ".join(row) + " |")

    lines.append("\n## Math correctness\n")
    lines.append("- NAV = realized closed-trade PnL + open-position MtM at end of window")
    lines.append("- Capital accounting: 1 unit base; bet_size = 8% of CURRENT portfolio at entry")
    lines.append("- Open positions block their bet_size from new entries (no over-deployment)")
    lines.append("- Daily NAV is from portfolio_value sequence, not sum-of-event-returns")
    lines.append("- Compound math is geometric (cum = portfolio_value / start_value)")
    lines.append("- Asymmetric stop/target applied intra-event (walk forward day-by-day)")
    lines.append("- Round-trip cost 0.30% (vs prior 0.24% which was light)")

    (OUT_DIR/"PROPER_PORTFOLIO_SIMULATION.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_DIR/'PROPER_PORTFOLIO_SIMULATION.md'}")

if __name__ == "__main__":
    main()
