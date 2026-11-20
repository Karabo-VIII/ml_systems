"""Stress-test the 17-setup deploy on OOS across:
  - K_MAX (4, 6, 8, 10, 12) -- concurrent position cap
  - ROUND_TRIP_COST (0.20%, 0.30%, 0.50%, 1.00%) -- realistic to pessimistic
  - BET_FRACTION (4%, 6%, 8%, 10%) -- conservative to aggressive sizing

Each grid cell runs the proper portfolio simulator on OOS. Reports
Sharpe, Sortino, Calmar, max_DD, total_return for each combination.

Purpose: confirm the deploy candidate's edge survives across reasonable
parameter ranges (robustness check). If edge collapses under small
parameter changes, the candidate is overfit.

Outputs:
  runs/oracle_layer3/SMART_DISCOVERY_EXHAUSTIVE_TRAIN/stress_test_grid.csv
  runs/oracle_layer3/SMART_DISCOVERY_EXHAUSTIVE_TRAIN/STRESS_TEST_REPORT.md
"""
from __future__ import annotations
from pathlib import Path
from datetime import date, timedelta
from itertools import product

import numpy as np
import pandas as pd
import polars as pl

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"/"audit"))
from proper_portfolio_simulator import (
    DEPLOY_17, simulate_portfolio, compute_risk_metrics,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "oracle_layer3" / "SMART_DISCOVERY_EXHAUSTIVE_TRAIN"

OOS_START = date(2024, 5, 16)
OOS_END = date(2025, 3, 15)

# Stress dimensions
K_GRID = [4, 6, 8, 10, 12]
COST_GRID = [0.0020, 0.0030, 0.0050, 0.0100]  # 0.20% to 1.00%
BET_GRID = [0.04, 0.06, 0.08, 0.10]  # 4% to 10% per trade

def main():
    print("="*78)
    print("DEPLOY STRESS TEST -- 17-setup portfolio, OOS window")
    print("="*78)
    print(f"OOS: {OOS_START} -> {OOS_END}")
    print(f"Grid: K in {K_GRID}, cost in {[f'{c*100:.2f}%' for c in COST_GRID]}, bet in {[f'{b*100:.0f}%' for b in BET_GRID]}")
    print(f"Total cells: {len(K_GRID) * len(COST_GRID) * len(BET_GRID)}")
    print()

    # Load OOS events (already filtered to 17-deploy in oos_events.parquet from prior turn)
    oos_events = pd.read_parquet(OUT_DIR/"oos_events.parquet")
    oos_events["date"] = pd.to_datetime(oos_events["date"]).dt.date
    keys_set = set(DEPLOY_17)
    oos_filt = oos_events[oos_events.set_index(["indicator","config"]).index.isin(keys_set)][
        ["asset","date","indicator","config","ret_E_14d"]
    ].copy()
    print(f"OOS events: {len(oos_filt):,}")

    # Load panel
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
    print(f"Panels: {len(panel_idx)} assets")

    # Monkey-patch the cost into proper simulator's module-level constants
    import proper_portfolio_simulator as pps

    rows = []
    n = len(K_GRID) * len(COST_GRID) * len(BET_GRID)
    i = 0
    for K, cost, bet in product(K_GRID, COST_GRID, BET_GRID):
        i += 1
        # Pass cost EXPLICITLY (default-arg-binding fix from prior bug)
        daily_df, trade_df = simulate_portfolio(oos_filt, panel_idx, OOS_START, OOS_END,
                                                  k_max=K, bet_fraction=bet,
                                                  round_trip_cost=cost)
        m = compute_risk_metrics(daily_df, trade_df, (OOS_END - OOS_START).days)
        m["K"] = K; m["cost_pct"] = cost*100; m["bet_pct"] = bet*100
        rows.append(m)
        print(f"  [{i:2d}/{n}]  K={K:2d}  cost={cost*100:.2f}%  bet={bet*100:.0f}%  "
              f"ret={m['total_return_pct']:+8.2f}%  Sortino={m['sortino']:+.3f}  "
              f"Calmar={m['calmar']:+.3f}  Max_DD={m['max_dd_pct']:+.2f}%")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR/"stress_test_grid.csv", index=False)

    # Summary stats
    print("\n=== STRESS TEST SUMMARY ===")
    print(f"Total return RANGE: {df['total_return_pct'].min():+.2f}% to {df['total_return_pct'].max():+.2f}%")
    print(f"Sortino RANGE: {df['sortino'].min():+.3f} to {df['sortino'].max():+.3f}")
    print(f"Calmar RANGE: {df['calmar'].min():+.3f} to {df['calmar'].max():+.3f}")
    print(f"Max DD RANGE: {df['max_dd_pct'].min():.2f}% to {df['max_dd_pct'].max():.2f}%")
    print(f"% cells with POSITIVE total return: {(df['total_return_pct']>0).mean()*100:.0f}%")
    print(f"% cells with Sortino > 1.0: {(df['sortino']>1.0).mean()*100:.0f}%")
    print(f"% cells with Calmar > 1.0: {(df['calmar']>1.0).mean()*100:.0f}%")

    # Effect of each dimension
    print("\n=== PARAMETER EFFECTS ===")
    print("\nBy K (averaged over cost x bet):")
    print(df.groupby("K")[["total_return_pct","sortino","calmar","max_dd_pct"]].mean().round(3))
    print("\nBy cost_pct:")
    print(df.groupby("cost_pct")[["total_return_pct","sortino","calmar","max_dd_pct"]].mean().round(3))
    print("\nBy bet_pct:")
    print(df.groupby("bet_pct")[["total_return_pct","sortino","calmar","max_dd_pct"]].mean().round(3))

    # Best / worst cells
    print("\n=== BEST CELLS (by Calmar) ===")
    best = df.nlargest(5, "calmar")[["K","cost_pct","bet_pct","total_return_pct","sortino","calmar","max_dd_pct"]]
    print(best.to_string(index=False))

    print("\n=== WORST CELLS (by Calmar) ===")
    worst = df.nsmallest(5, "calmar")[["K","cost_pct","bet_pct","total_return_pct","sortino","calmar","max_dd_pct"]]
    print(worst.to_string(index=False))

    # Report
    lines = ["# Deploy Stress Test -- 17-Setup OOS Sensitivity\n"]
    lines.append(f"\n## Grid\n")
    lines.append(f"- K in {K_GRID}")
    lines.append(f"- cost in {[f'{c*100:.2f}%' for c in COST_GRID]}")
    lines.append(f"- bet in {[f'{b*100:.0f}%' for b in BET_GRID]}")
    lines.append(f"- Total cells: {len(rows)}")

    lines.append(f"\n## Robustness summary\n")
    lines.append(f"- Total return range: {df['total_return_pct'].min():+.2f}% to {df['total_return_pct'].max():+.2f}%")
    lines.append(f"- Sortino range: {df['sortino'].min():+.3f} to {df['sortino'].max():+.3f}")
    lines.append(f"- Calmar range: {df['calmar'].min():+.3f} to {df['calmar'].max():+.3f}")
    lines.append(f"- % cells positive total return: **{(df['total_return_pct']>0).mean()*100:.0f}%**")
    lines.append(f"- % cells Sortino > 1.0: **{(df['sortino']>1.0).mean()*100:.0f}%**")
    lines.append(f"- % cells Calmar > 1.0: **{(df['calmar']>1.0).mean()*100:.0f}%**")

    lines.append(f"\n## Parameter effect averages\n")
    lines.append("### By K (concurrent positions)\n")
    lines.append("| K | total return | Sortino | Calmar | max DD |")
    lines.append("|--:|--:|--:|--:|--:|")
    for k_val, row in df.groupby("K")[["total_return_pct","sortino","calmar","max_dd_pct"]].mean().iterrows():
        lines.append(f"| {k_val} | {row['total_return_pct']:+.2f}% | {row['sortino']:+.3f} | {row['calmar']:+.3f} | {row['max_dd_pct']:+.2f}% |")

    lines.append("\n### By cost (round-trip)\n")
    lines.append("| cost % | total return | Sortino | Calmar | max DD |")
    lines.append("|--:|--:|--:|--:|--:|")
    for c_val, row in df.groupby("cost_pct")[["total_return_pct","sortino","calmar","max_dd_pct"]].mean().iterrows():
        lines.append(f"| {c_val:.2f}% | {row['total_return_pct']:+.2f}% | {row['sortino']:+.3f} | {row['calmar']:+.3f} | {row['max_dd_pct']:+.2f}% |")

    lines.append("\n### By bet size\n")
    lines.append("| bet % | total return | Sortino | Calmar | max DD |")
    lines.append("|--:|--:|--:|--:|--:|")
    for b_val, row in df.groupby("bet_pct")[["total_return_pct","sortino","calmar","max_dd_pct"]].mean().iterrows():
        lines.append(f"| {b_val:.0f}% | {row['total_return_pct']:+.2f}% | {row['sortino']:+.3f} | {row['calmar']:+.3f} | {row['max_dd_pct']:+.2f}% |")

    lines.append(f"\n## Best 5 cells (by Calmar)\n")
    lines.append("| K | cost | bet | total ret | Sortino | Calmar | max DD |")
    lines.append("|--:|--:|--:|--:|--:|--:|--:|")
    for _, r in best.iterrows():
        lines.append(f"| {int(r['K'])} | {r['cost_pct']:.2f}% | {r['bet_pct']:.0f}% | {r['total_return_pct']:+.2f}% | {r['sortino']:+.3f} | {r['calmar']:+.3f} | {r['max_dd_pct']:+.2f}% |")

    lines.append(f"\n## Worst 5 cells (by Calmar)\n")
    lines.append("| K | cost | bet | total ret | Sortino | Calmar | max DD |")
    lines.append("|--:|--:|--:|--:|--:|--:|--:|")
    for _, r in worst.iterrows():
        lines.append(f"| {int(r['K'])} | {r['cost_pct']:.2f}% | {r['bet_pct']:.0f}% | {r['total_return_pct']:+.2f}% | {r['sortino']:+.3f} | {r['calmar']:+.3f} | {r['max_dd_pct']:+.2f}% |")

    lines.append(f"\n## Deploy verdict\n")
    if (df['total_return_pct']>0).mean() >= 0.80 and (df['sortino']>1.0).mean() >= 0.50:
        lines.append("**ROBUST**: 80%+ cells positive, 50%+ Sortino > 1.0.")
    elif (df['total_return_pct']>0).mean() >= 0.60:
        lines.append("**MODERATELY ROBUST**: 60%+ cells positive but Sortino edge sensitive to params.")
    else:
        lines.append("**FRAGILE**: < 60% cells positive; edge is parameter-sensitive (overfit risk).")

    (OUT_DIR/"STRESS_TEST_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_DIR/'STRESS_TEST_REPORT.md'}")

if __name__ == "__main__":
    main()
