"""multi_day_capture_analysis.py - Test multi-day holding hypothesis.

Per user 2026-05-18: reject 1-day cutoff. Test whether 1-3 day holds enable
better capture of accumulating moves. Run numbers across:
  - Oracle availability at 1d/3d/5d horizons
  - Per-event capture scenarios at each horizon
  - What-if extended holds on existing trade ledger
"""
from __future__ import annotations
import os
from pathlib import Path
from datetime import date

import numpy as np
import polars as pl

os.environ["PYTHONIOENCODING"] = "utf-8"

ROOT = Path(__file__).resolve().parents[2]


def main():
    # === A. Oracle availability at multi-day horizons ===
    print("=" * 80)
    print("A. ORACLE K=5 LO IDEAL AT 1d / 3d / 5d HORIZONS")
    print("=" * 80)
    oc = pl.read_parquet(str(ROOT / "data/processed/outcome_catalog.parquet"))
    print(f"catalog rows={len(oc)}  range={oc['date'].min()} -> {oc['date'].max()}")
    print()

    for col, label in [
        ("ideal_k5_1d_ret", "1-day K=5 LO"),
        ("ideal_k5_3d_ret", "3-day K=5 LO"),
        ("ideal_k5_5d_ret", "5-day K=5 LO"),
    ]:
        s = oc[col].drop_nulls()
        pct_p2 = (s >= 0.02).sum() / len(s) * 100
        pct_p5 = (s >= 0.05).sum() / len(s) * 100
        pct_p10 = (s >= 0.10).sum() / len(s) * 100
        print(f"  {label}:")
        print(f"    mean={s.mean()*100:+6.2f}%  median={s.median()*100:+6.2f}%")
        print(f"    p20={s.quantile(0.20)*100:+6.2f}%  p80={s.quantile(0.80)*100:+6.2f}%  max={s.max()*100:+6.2f}%")
        print(f"    pct >=2%: {pct_p2:.0f}%   pct >=5%: {pct_p5:.0f}%   pct >=10%: {pct_p10:.0f}%")
        print()

    # === B. 8Q oracle means and wealth scenarios ===
    print("=" * 80)
    print("B. WEALTH SCENARIOS — per-event capture at each horizon")
    print("=" * 80)
    oc_8q = oc.filter(
        (pl.col("date") >= date(2024, 1, 1)) & (pl.col("date") <= date(2025, 12, 31))
    )
    o1d = float(oc_8q["ideal_k5_1d_ret"].mean())
    o3d = float(oc_8q["ideal_k5_3d_ret"].mean())
    o5d = float(oc_8q["ideal_k5_5d_ret"].mean())
    print(f"8Q window oracle means (NET of 24bps RT):")
    print(f"  1d: {o1d*100:+.2f}%")
    print(f"  3d: {o3d*100:+.2f}%")
    print(f"  5d: {o5d*100:+.2f}%")
    print()

    print(f"{'cap':>5s} {'horiz':>6s} {'per-event':>11s} {'events/yr':>10s} {'yr compound':>13s} {'2yr wealth':>13s} {'daily-eq':>10s}")
    print("-" * 80)
    for cap_pct in [0.25, 0.50, 0.75, 1.00]:
        for label, days, oa in [("1d", 1, o1d), ("3d", 3, o3d), ("5d", 5, o5d)]:
            per_ev = oa * cap_pct
            ev_yr = 365 / days
            gf = (1 + per_ev) ** ev_yr
            yr_pct = (gf - 1) * 100
            w2 = 10000 * gf ** 2
            daily_eq = ((1 + per_ev) ** (1 / days) - 1) * 100
            # Truncate huge numbers
            yr_str = f"{yr_pct:,.0f}%"
            if abs(yr_pct) > 1e9:
                yr_str = f"{yr_pct:.2e}%"
            w2_str = f"${w2:,.0f}"
            if abs(w2) > 1e12:
                w2_str = f"${w2:.2e}"
            print(f"  {cap_pct*100:>3.0f}% {label:>6s} {per_ev*100:>+10.2f}% {ev_yr:>9.1f} {yr_str:>13s} {w2_str:>13s} {daily_eq:>+8.3f}%/d")
        print()

    # === C. Current strategy actual holding-period distribution ===
    print("=" * 80)
    print("C. CURRENT STRATEGY: actual hold-day distribution")
    print("=" * 80)
    trades = pl.read_parquet(str(ROOT / "runs/audit/capture_ratio_scoreboard_v2_pertrade.parquet"))
    print(f"Total trades: {len(trades)}")
    print(f"{'hold_days':>10s} {'n':>6s} {'pct':>6s} {'mean_gross':>11s} {'mean_net':>10s} {'win_rate':>9s}")
    hd = trades.group_by("hold_days").agg([
        pl.len().alias("n"),
        pl.col("net_pnl_pct").mean().alias("mean_net"),
        pl.col("gross_ret_pct").mean().alias("mean_gross"),
        (pl.col("net_pnl_pct") > 0).cast(pl.Float64).mean().alias("win_rate"),
    ]).sort("hold_days")
    total = len(trades)
    for r in hd.iter_rows(named=True):
        pct = r["n"] / total * 100
        print(f"  {r['hold_days']:>8d} {r['n']:>6d} {pct:>5.1f}% {r['mean_gross']:>+10.3f}% {r['mean_net']:>+9.4f}% {r['win_rate']*100:>8.1f}%")

    # === D. What-if extended holds on ALL trades ===
    print()
    print("=" * 80)
    print("D. WHAT-IF: hold every trade to entry + N days (chimera-1d close)")
    print("=" * 80)
    CHIMERA_1D = ROOT / "data/processed/chimera/1d"

    asset_map = {}
    for f in os.listdir(CHIMERA_1D):
        if not f.endswith(".parquet"):
            continue
        sym = f.split("usdt")[0].upper()
        path = CHIMERA_1D / f
        if sym not in asset_map or str(path) > str(asset_map[sym]):
            asset_map[sym] = path
    frames = []
    for sym, path in sorted(asset_map.items()):
        df = pl.read_parquet(path, columns=["date", "close"]).rename({"close": sym})
        frames.append(df)
    wide = frames[0]
    for df in frames[1:]:
        wide = wide.join(df, on="date", how="full", coalesce=True)
    wide = wide.sort("date")

    asset_cols = [c for c in wide.columns if c != "date"]
    rows = []
    for asset in asset_cols:
        a_df = wide.select(["date", asset]).rename({asset: "close"}).drop_nulls().sort("date")
        a_df = a_df.with_columns([
            pl.col("date").alias("entry_date"),
            pl.col("close").alias("c0"),
            pl.col("close").shift(-1).alias("c1"),
            pl.col("close").shift(-2).alias("c2"),
            pl.col("close").shift(-3).alias("c3"),
            pl.col("close").shift(-5).alias("c5"),
            pl.col("close").shift(-7).alias("c7"),
            pl.lit(asset).alias("asset"),
        ])
        rows.append(a_df.select(["asset", "entry_date", "c0", "c1", "c2", "c3", "c5", "c7"]))
    future = pl.concat(rows)

    all_tr = trades.select([
        "quarter", "asset", "entry_date", "size_pct",
        "net_pnl_pct", "gross_ret_pct", "exit_reason",
    ])
    all_tr = all_tr.join(future, on=["asset", "entry_date"], how="left")
    all_tr = all_tr.with_columns([
        pl.when(pl.col("c0") > 0).then((pl.col("c1") / pl.col("c0") - 1) * 100).alias("wi_1d"),
        pl.when(pl.col("c0") > 0).then((pl.col("c2") / pl.col("c0") - 1) * 100).alias("wi_2d"),
        pl.when(pl.col("c0") > 0).then((pl.col("c3") / pl.col("c0") - 1) * 100).alias("wi_3d"),
        pl.when(pl.col("c0") > 0).then((pl.col("c5") / pl.col("c0") - 1) * 100).alias("wi_5d"),
        pl.when(pl.col("c0") > 0).then((pl.col("c7") / pl.col("c0") - 1) * 100).alias("wi_7d"),
    ])

    print(f"{'horizon':>10s} {'mean':>10s} {'median':>10s} {'pct_pos':>9s} {'P75':>10s} {'P90':>10s} {'P10':>10s}")
    for h_label, col in [("entry+1d", "wi_1d"), ("entry+2d", "wi_2d"), ("entry+3d", "wi_3d"), ("entry+5d", "wi_5d"), ("entry+7d", "wi_7d")]:
        s = all_tr[col].drop_nulls()
        pct_pos = (s > 0).sum() / len(s) * 100
        print(f"  {h_label:>8s} {s.mean():>+8.3f}% {s.median():>+8.3f}% {pct_pos:>7.1f}% {s.quantile(0.75):>+8.3f}% {s.quantile(0.90):>+8.3f}% {s.quantile(0.10):>+8.3f}%")

    # NAV impact at each horizon (size-weighted)
    print()
    print("NAV impact (size-weighted, sum across all trades 8Q):")
    print(f"{'horizon':>10s} {'sum_nav_pct':>14s} {'mean_per_trade_nav':>20s} {'net (-0.12% RT)':>16s}")
    for h_label, col in [("entry+1d", "wi_1d"), ("entry+2d", "wi_2d"), ("entry+3d", "wi_3d"), ("entry+5d", "wi_5d"), ("entry+7d", "wi_7d")]:
        all_tr2 = all_tr.with_columns((pl.col(col) * pl.col("size_pct")).alias("nav_pct"))
        sum_nav = float(all_tr2["nav_pct"].drop_nulls().sum())
        mean_per = float(all_tr2["nav_pct"].drop_nulls().mean())
        # Net of RT cost: each trade pays ~0.12% RT cost on size_pct (approx)
        # cost = 0.0012 * size_pct (already as fraction); to make % NAV, multiply by 100
        # Approximate: total cost = n_trades * mean_cost = 1808 * 0.12 * mean_size... too rough
        # Just subtract aggregate cost: 0.12% RT per trade * mean size 0.04 = 0.0048% NAV per trade * 1808 = 8.68% total
        total_cost_nav = 1808 * 0.0012 * float(all_tr["size_pct"].mean()) * 100
        net_nav = sum_nav - total_cost_nav
        print(f"  {h_label:>8s} {sum_nav:>+12.2f}% {mean_per:>+19.4f}% {net_nav:>+15.2f}%")

    # === E. Compare to current 8Q realized +20.87% ===
    print()
    print(f"Current 8Q realized (1-day max hold actual): +20.87%")
    print()
    print(f"What's the *upper bound* of holding all current entries to +3d net of cost?")
    h = "wi_3d"
    all_tr2 = all_tr.with_columns((pl.col(h) * pl.col("size_pct")).alias("nav_pct"))
    sum_nav_3d = float(all_tr2["nav_pct"].drop_nulls().sum())
    cost_3d = 1808 * 0.0012 * float(all_tr["size_pct"].mean()) * 100
    print(f"  Sum size-weighted NAV at entry+3d: {sum_nav_3d:+.2f}%")
    print(f"  Less aggregate cost (assume same RT): -{cost_3d:.2f}%")
    print(f"  Net upper bound: {sum_nav_3d - cost_3d:+.2f}%")
    print(f"  vs current +20.87%")

    # === F. Capture-rate scenarios at 3-day horizon (the user's framing) ===
    print()
    print("=" * 80)
    print("F. USER'S FRAMING: capture rate of 50-75% on 3d moves")
    print("=" * 80)
    print(f'Oracle 3d mean: {o3d*100:+.2f}%')
    print(f"Per-event ROI at capture rates:")
    for cr in [0.10, 0.25, 0.50, 0.75]:
        per_ev = o3d * cr
        # Events/year if we fire on each oracle-positive day
        oc_3d_pos_days = (oc_8q['ideal_k5_3d_ret'] > 0.02).sum()
        events_per_year = oc_3d_pos_days / 2 * (365/((oc_8q['date'].max() - oc_8q['date'].min()).days))
        # Assume overlapping holds: 5 positions, 3d hold = ~1.67 entries/day at full deployment
        # = 365 entries / 3-day windows = ~120 windows/yr
        # If 5 positions per window: 5 * 120 = 600 entries/yr at full deployment
        # But position cap means we capture portfolio-level: per_ev * portfolio_share
        # If 5 positions * 4% size = 20% NAV deployed, per-event captures 20% * per_ev
        portfolio_size_pct = 0.20  # 5 positions x 4% cap
        weekly_ret = per_ev * portfolio_size_pct  # per 3d period, NAV-weighted
        weekly_compound_yr = (1 + weekly_ret) ** (365/3) - 1
        w2 = 10000 * (1 + weekly_compound_yr)**2 if weekly_compound_yr < 100 else float('inf')
        daily_eq = ((1 + weekly_ret)**(1/3) - 1) * 100
        print(f'  capture={cr*100:>3.0f}% per-event={per_ev*100:+6.2f}% portfolio_pe={weekly_ret*100:+5.2f}% yr_compound={weekly_compound_yr*100:>+10,.0f}% daily-eq={daily_eq:+.3f}%/d')


if __name__ == "__main__":
    main()
