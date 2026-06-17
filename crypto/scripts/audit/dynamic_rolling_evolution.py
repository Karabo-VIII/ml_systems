"""DEPRECATED 2026-05-20 -- contains selection-time look-ahead bug.

This simulator sorts by `asym_ret` (derived from future ret_E_14d) for
K-selection in BOTH the DYNAMIC and STATIC modes. The "STATIC beats
DYNAMIC by 13pp" finding from prior turn was based on best-K math for
both -- the relative comparison may hold but absolute numbers are
upper-bound, not realistic deploy.

Use scripts/audit/honest_v2_simulator.py for honest math.

Memory: [[red-team-failure-diagnostic-2026-05-20]]

----

Original docstring:
Dynamic rolling-window setup evolution simulator.

Per user mandate (2026-05-20):
  "Is there a way to dynamically evolve the fixed rules strats?
   Top K configs and then each within a rolling window are replayed,
   new winner config class selected for next phase. 1-month rolling?"

Design:
  - At t=0, use TRAIN to bootstrap active set of K=15 setups.
  - Every 30 days, re-rank candidates by trailing-30-day performance.
  - Promote new winners (asym_NAV last 30d) into active set; demote losers.
  - Track turnover: how many setups change per cycle?
  - Compare DYNAMIC vs STATIC deploy NAV.

Window: trailing-30-day evaluation; 30-day step (monthly).
Candidate pool: the full 219 VAL-stable library (not just the 14 deploy).
Active set size: K_ACTIVE = 15 (slightly larger than 14 deploy).

Tests run on VAL+OOS combined for sufficient rolling history.

Outputs:
  runs/oracle_layer3/SMART_DISCOVERY_EXHAUSTIVE_TRAIN/dynamic_rolling_active_sets.csv
  runs/oracle_layer3/SMART_DISCOVERY_EXHAUSTIVE_TRAIN/dynamic_vs_static_nav.csv
  runs/oracle_layer3/SMART_DISCOVERY_EXHAUSTIVE_TRAIN/DYNAMIC_ROLLING_REPORT.md
"""
from __future__ import annotations
from pathlib import Path
from datetime import date, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "oracle_layer3" / "SMART_DISCOVERY_EXHAUSTIVE_TRAIN"

VAL_START = date(2023, 7, 2)
VAL_END = date(2024, 5, 15)
OOS_START = date(2024, 5, 16)
OOS_END = date(2025, 3, 15)

COST = 0.0024
BET_FRACTION = 0.08
HARD_STOP = -0.04
TARGET = 0.12
K_MAX = 8
K_ACTIVE = 15           # active set size
WINDOW_DAYS = 30        # trailing evaluation window (1 month)
STEP_DAYS = 30          # cycle step (monthly)
WEEKLY_FLOOR = 0.0525

def asymmetric_returns(rets):
    out = np.copy(rets)
    out = np.where(out <= HARD_STOP, HARD_STOP, out)
    out = np.where(out >= TARGET, TARGET, out)
    return out

def main():
    print("="*78)
    print("DYNAMIC ROLLING-WINDOW EVOLUTION (1-month re-rank)")
    print("="*78)

    # Load VAL + OOS events (combined for rolling history)
    val_events = pd.read_parquet(OUT_DIR/"val_events.parquet")
    val_events["date"] = pd.to_datetime(val_events["date"]).dt.date
    print(f"VAL events: {len(val_events):,}")

    oos_events = pd.read_parquet(OUT_DIR/"oos_events.parquet")
    oos_events["date"] = pd.to_datetime(oos_events["date"]).dt.date
    print(f"OOS events (14-setup portfolio only): {len(oos_events):,}")
    # NOTE: OOS events are filtered to the 14-setup portfolio for the OOS replay run.
    # For dynamic rolling, we need the FULL VAL-stable candidate pool on OOS too.
    # Skip OOS-only setups not in VAL for the dynamic test; use VAL as the sandbox.

    # Use VAL events as the dynamic sandbox -- has all 225 setups firing across 10 months.
    all_events = val_events.copy()
    all_events = all_events[(all_events["date"] >= VAL_START) & (all_events["date"] <= VAL_END)]
    print(f"\nDynamic sandbox events: {len(all_events):,} over {all_events['date'].nunique()} days")

    all_events["asym_ret"] = asymmetric_returns(all_events["ret_E_14d"].fillna(0).values)

    # Candidate pool: ALL setups in val_events (the VAL-stable 225)
    candidates = list(set(zip(all_events["indicator"], all_events["config"])))
    print(f"Candidate pool size: {len(candidates)} setups")

    # The 14-setup STATIC portfolio (for comparison)
    STATIC_PORTFOLIO = [
        ("SMA_cross", "(3, 5)"), ("SMA_cross", "(3, 8)"), ("SMA_cross", "(3, 13)"),
        ("SMA_cross", "(5, 8)"), ("SMA_cross", "(20, 21)"),
        ("Donchian_breakout", "(20,)"), ("ROC_momentum", "(10, 7)"),
        ("Stochastic_bounce", "(7, 3, 80, 20)"), ("Stochastic_bounce", "(7, 3, 90, 10)"),
        ("MACD_cross", "(5, 21, 5)"), ("MACD_cross", "(5, 34, 9)"),
        ("BB_breach", "(20, 1.5)"), ("EMA_cross", "(3, 5)"), ("EMA_cross", "(3, 8)"),
    ]
    static_set = set(STATIC_PORTFOLIO)

    # Define rolling cycles
    cycles = []
    cur = VAL_START + timedelta(days=WINDOW_DAYS)
    while cur < VAL_END:
        cycles.append({
            "cycle_start": cur,
            "cycle_end": min(cur + timedelta(days=STEP_DAYS - 1), VAL_END),
            "eval_window_start": cur - timedelta(days=WINDOW_DAYS),
            "eval_window_end": cur - timedelta(days=1),
        })
        cur += timedelta(days=STEP_DAYS)
    print(f"\nCycles: {len(cycles)} (1-month each)")

    # Per-cycle: evaluate candidates on trailing window, select top K_ACTIVE
    active_history = []
    dynamic_daily = []
    static_daily = []
    prior_active = None
    turnover_counts = []

    for ci, cyc in enumerate(cycles):
        ws, we = cyc["eval_window_start"], cyc["eval_window_end"]
        cs, ce = cyc["cycle_start"], cyc["cycle_end"]

        # Score each candidate on trailing window
        eval_events = all_events[(all_events["date"] >= ws) & (all_events["date"] <= we)]
        scores = []
        for ind, cfg in candidates:
            sub = eval_events[(eval_events["indicator"] == ind) & (eval_events["config"] == cfg)]
            if len(sub) < 5:  # too few events to score
                continue
            asym = sub["asym_ret"].values
            score = asym.sum() * BET_FRACTION  # 30d total asym NAV at 8% sizing
            scores.append({"indicator": ind, "config": cfg, "n_30d": len(sub),
                           "asym_nav_30d": score, "asym_mean": asym.mean()})
        scores_df = pd.DataFrame(scores).sort_values("asym_nav_30d", ascending=False)
        active = list(zip(scores_df.head(K_ACTIVE)["indicator"], scores_df.head(K_ACTIVE)["config"]))
        active_set = set(active)

        # Turnover
        if prior_active is not None:
            turnover = len(active_set ^ prior_active) / 2  # symmetric diff / 2 = # swapped
            turnover_counts.append(turnover)
        prior_active = active_set
        active_history.append({
            "cycle": ci + 1,
            "cycle_start": cs, "cycle_end": ce,
            "eval_window": f"{ws} to {we}",
            "n_active": len(active),
            "turnover_from_prior": turnover_counts[-1] if turnover_counts else 0,
            "active_setups": ",".join(f"{i}|{c}" for i, c in active),
        })

        # Run cycle simulation with active set vs static
        cycle_events = all_events[(all_events["date"] >= cs) & (all_events["date"] <= ce)]
        for mode_name, port_set, sink in [("DYNAMIC", active_set, dynamic_daily),
                                            ("STATIC", static_set, static_daily)]:
            port_events = cycle_events[
                cycle_events.set_index(["indicator", "config"]).index.isin(port_set)
            ]
            for d, day_grp in port_events.groupby("date"):
                uniq = day_grp.sort_values("asym_ret", ascending=False).drop_duplicates(subset="asset", keep="first")
                picked = uniq.head(K_MAX)
                nav = picked["asym_ret"].sum() * BET_FRACTION
                sink.append({"cycle": ci + 1, "date": d, "nav_pct": nav,
                             "n_picked": len(picked), "n_fires": len(day_grp)})

    dyn_df = pd.DataFrame(dynamic_daily)
    sta_df = pd.DataFrame(static_daily)
    if not len(dyn_df) or not len(sta_df):
        print("ERROR: insufficient daily NAV data")
        return

    # Compare
    for name, df in (("DYNAMIC", dyn_df), ("STATIC", sta_df)):
        df = df.sort_values("date").reset_index(drop=True)
        df["nav_7d"] = df["nav_pct"].rolling(7).sum()
        df["cum_nav"] = (1 + df["nav_pct"]).cumprod()

    dyn_df = dyn_df.sort_values("date").reset_index(drop=True)
    sta_df = sta_df.sort_values("date").reset_index(drop=True)
    dyn_df["nav_7d"] = dyn_df["nav_pct"].rolling(7).sum()
    sta_df["nav_7d"] = sta_df["nav_pct"].rolling(7).sum()
    dyn_df["cum_nav"] = (1 + dyn_df["nav_pct"]).cumprod()
    sta_df["cum_nav"] = (1 + sta_df["nav_pct"]).cumprod()

    def floor_clear(df):
        return (df["nav_7d"] >= WEEKLY_FLOOR).sum(), max(len(df) - 6, 1)

    print("\n=== DYNAMIC vs STATIC HEAD-TO-HEAD (VAL sandbox) ===")
    for name, df in (("DYNAMIC", dyn_df), ("STATIC", sta_df)):
        total_nav = df["nav_pct"].sum() * 100
        mean_d = df["nav_pct"].mean() * 100
        positive = (df["nav_pct"] > 0).mean() * 100
        cum_final = (df["cum_nav"].iloc[-1] - 1) * 100
        cum_max = df["cum_nav"].cummax()
        max_dd = ((df["cum_nav"] / cum_max - 1) * 100).min()
        mean_7d = df["nav_7d"].mean() * 100
        fc, ft = floor_clear(df)
        print(f"  {name:<8} total_NAV={total_nav:+8.2f}% mean_d={mean_d:+.3f}%  +days={positive:.1f}%  "
              f"cum_compound={cum_final:+9.2f}%  max_DD={max_dd:+.2f}%  "
              f"mean_7d={mean_7d:+.2f}%  floor_clear={fc}/{ft} ({fc*100/ft:.0f}%)")

    print(f"\nMean turnover per cycle: {np.mean(turnover_counts):.1f} setups (of {K_ACTIVE} active)")
    print(f"Max turnover: {np.max(turnover_counts):.0f}; Min: {np.min(turnover_counts):.0f}")

    ah_df = pd.DataFrame(active_history)
    ah_df.to_csv(OUT_DIR/"dynamic_rolling_active_sets.csv", index=False)
    # Long-format daily NAV for plotting
    combined = pd.concat([
        dyn_df.assign(mode="DYNAMIC"),
        sta_df.assign(mode="STATIC"),
    ], ignore_index=True)
    combined.to_csv(OUT_DIR/"dynamic_vs_static_nav.csv", index=False)

    # Report
    lines = ["# Dynamic Rolling-Window Evolution\n"]
    lines.append(f"\n## Design\n")
    lines.append(f"- Trailing eval window: **{WINDOW_DAYS} days**")
    lines.append(f"- Cycle step: **{STEP_DAYS} days** (monthly re-rank)")
    lines.append(f"- Active set size: **K_ACTIVE = {K_ACTIVE}**")
    lines.append(f"- Candidate pool: **{len(candidates)}** VAL-stable setups")
    lines.append(f"- Sandbox: VAL window {VAL_START} -> {VAL_END} ({len(cycles)} cycles)")
    lines.append(f"- Selection metric: trailing-window asym_NAV (deterministic best-K)")

    lines.append(f"\n## Head-to-head DYNAMIC vs STATIC (best-K, K=8, asym -4%/+12%)\n")
    lines.append("| mode | total NAV | mean daily | +days | cum compound | max DD | mean 7d | floor clear |")
    lines.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    for name, df in (("DYNAMIC", dyn_df), ("STATIC", sta_df)):
        total_nav = df["nav_pct"].sum() * 100
        mean_d = df["nav_pct"].mean() * 100
        positive = (df["nav_pct"] > 0).mean() * 100
        cum_final = (df["cum_nav"].iloc[-1] - 1) * 100
        cum_max = df["cum_nav"].cummax()
        max_dd = ((df["cum_nav"] / cum_max - 1) * 100).min()
        mean_7d = df["nav_7d"].mean() * 100
        fc, ft = floor_clear(df)
        lines.append(f"| {name} | {total_nav:+.2f}% | {mean_d:+.3f}% | {positive:.1f}% | {cum_final:+.2f}% | {max_dd:+.2f}% | {mean_7d:+.2f}% | {fc}/{ft} ({fc*100/ft:.0f}%) |")

    lines.append(f"\n## Turnover\n")
    lines.append(f"Mean: {np.mean(turnover_counts):.1f} setups swapped per cycle (of {K_ACTIVE}).")
    lines.append(f"Range: {np.min(turnover_counts):.0f} - {np.max(turnover_counts):.0f}.")
    lines.append("Low turnover = portfolio is stable; high turnover = regime adapting.")

    lines.append(f"\n## Per-cycle active set (compact)\n")
    lines.append("| cycle | window | turnover | n_active |")
    lines.append("|--:|---|--:|--:|")
    for _, r in ah_df.iterrows():
        lines.append(f"| {int(r['cycle'])} | {r['cycle_start']} - {r['cycle_end']} | {int(r['turnover_from_prior'])} | {int(r['n_active'])} |")

    lines.append(f"\n## Interpretation\n")
    dyn_fc, dyn_ft = floor_clear(dyn_df)
    sta_fc, sta_ft = floor_clear(sta_df)
    if dyn_fc * 100 / dyn_ft > sta_fc * 100 / sta_ft:
        lines.append(f"DYNAMIC outperforms STATIC by {(dyn_fc/dyn_ft - sta_fc/sta_ft)*100:.1f}pp on floor-clear rate.")
        lines.append("Monthly re-ranking captures regime shifts the static portfolio misses.")
    elif dyn_fc * 100 / dyn_ft < sta_fc * 100 / sta_ft:
        lines.append(f"STATIC outperforms DYNAMIC by {(sta_fc/sta_ft - dyn_fc/dyn_ft)*100:.1f}pp on floor-clear rate.")
        lines.append("Re-ranking on 30d trailing window induces noise. STATIC robust by construction.")
    else:
        lines.append("DYNAMIC and STATIC perform similarly; turnover not adding alpha.")

    (OUT_DIR/"DYNAMIC_ROLLING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_DIR/'DYNAMIC_ROLLING_REPORT.md'}")

if __name__ == "__main__":
    main()
