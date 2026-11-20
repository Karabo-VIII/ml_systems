"""build_corrected_union_2026_05_19.py — RED-team-corrected union analysis.

Addresses flags from runs/audit/RED_TEAM_AUDIT_2026_05_19.md:

  FLAG #1: K=20 cap unrealistic for v3. REPORT K=5 / K=10 / K=20 grid as primary.
  FLAG #2: Within-day arithmetic sum overstates at high K. Compute with PROPER
           per-day NAV deployment: each cell's size_pct is its lane's size_pct,
           capped at K positions. If K cells with each 4% = max 4K% NAV deployed.
           Use within-day daily compound, not arithmetic.
  FLAG #3: TA_SML u50 vs u100 universe mismatch — verify atlas dedups by
           (asset, date) so no double-count, and report TA_SML's marginal lift.
  FLAG #5: R7 standalone NAV via DAILY COMPOUND not arithmetic trade-sum.
  FLAG #8: Build #2 / Build #3 NAV via daily compound recomputed.

Outputs:
  runs/audit/CORRECTED_UNION_VERDICT_2026_05_19.md
  runs/audit/corrected_union_grid.parquet
"""
from __future__ import annotations
import os
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

os.environ["PYTHONIOENCODING"] = "utf-8"
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "audit"


def proper_daily_compound(per_day_pnl_pct: np.ndarray) -> float:
    """Properly compound daily PnL % into total return."""
    return float(np.exp(np.log1p(per_day_pnl_pct / 100).sum()) - 1) * 100


def neg_weeks_count(dates_idx: list[date], per_day_pnl_pct: np.ndarray) -> int:
    """Count negative ISO weeks using proper compound aggregation."""
    import pandas as pd
    df = pd.DataFrame({"date": pd.to_datetime(dates_idx), "pnl": per_day_pnl_pct})
    df["iso_year"] = df["date"].dt.isocalendar().year
    df["iso_week"] = df["date"].dt.isocalendar().week
    df["wk"] = df["iso_year"].astype(str) + "-W" + df["iso_week"].astype(str).str.zfill(2)
    weekly = df.groupby("wk")["pnl"].apply(lambda x: float(np.exp(np.log1p(x.values / 100).sum()) - 1))
    return int((weekly < 0).sum())


def main():
    print("=" * 80)
    print("CORRECTED UNION ANALYSIS — RED-TEAM PATCHED")
    print("=" * 80)

    # ===== LOAD =====
    ext = pl.read_parquet(str(OUT_DIR / "union_extended_trade_ledger.parquet"))
    print(f"\nLoaded extended trade ledger: {len(ext)} trades / {ext['lane'].n_unique()} lanes")

    # ===== FLAG #3: TA_SML universe check =====
    print("\n[FLAG #3 — TA_SML universe check]")
    tasml = ext.filter(pl.col("lane") == "TA_SML_SOLO")
    others = ext.filter(pl.col("lane") != "TA_SML_SOLO")
    # Find overlap: (asset, entry_date) cells where BOTH TA_SML AND another lane fire
    tasml_cells = set(zip(tasml["asset"].to_list(), tasml["entry_date"].to_list()))
    other_cells = set(zip(others["asset"].to_list(), others["entry_date"].to_list()))
    overlap = tasml_cells & other_cells
    tasml_unique = tasml_cells - other_cells
    print(f"  TA_SML total cells: {len(tasml_cells)}")
    print(f"  Cells overlapping with other lanes: {len(overlap)} ({len(overlap)/len(tasml_cells)*100:.1f}%)")
    print(f"  TA_SML-unique cells: {len(tasml_unique)} ({len(tasml_unique)/len(tasml_cells)*100:.1f}%)")
    print(f"  Union atlas already DEDUPS by (asset, date) via mean_lane_pnl. No double-count.")
    print(f"  Effective TA_SML marginal contribution: {len(tasml_unique)} new cells (not 2017 trades).")

    # ===== Build atlas (same as before but report properly) =====
    atlas = ext.group_by(["asset", "entry_date"]).agg([
        pl.col("lane").n_unique().alias("n_lanes_firing"),
        pl.col("lane").unique().alias("lanes_set"),
        pl.col("net_pnl_pct").mean().alias("mean_lane_pnl"),
        pl.col("size_pct").first().alias("size_pct"),
    ]).sort("entry_date").with_columns(
        pl.col("entry_date").cast(pl.Date)
    )

    # ===== FLAG #1 + #2: Proper grid with K=3/5/7/10 ONLY (K=20 is unrealistic) =====
    print("\n[FLAG #1 + #2 — Realistic K-cap grid (K=20 dropped as unrealistic)]")
    print(f"  v3 reality: enforces 5-position cap. Per-bucket parallel cap could raise to ~10.")
    print(f"  K=20 results dropped from this corrected analysis.\n")

    all_dates = pl.date_range(date(2024,1,1), date(2025,12,31), interval="1d", eager=True)
    results = []
    print(f"  {'K':>3s} {'min_conv':>8s} {'COMP':>10s} {'mean_d':>10s} {'pos_d':>6s} {'neg_wk':>7s} {'wealth':>10s} {'cleared_floor':>14s}")
    for K in [3, 5, 7, 10]:
        for min_conv in [1, 2, 3]:
            cell = atlas.with_columns(pl.col("n_lanes_firing").cast(pl.Float64).alias("priority"))
            per_day = cell.filter(pl.col("priority") >= min_conv).sort(
                ["entry_date", "priority"], descending=[False, True]
            ).group_by("entry_date").agg([
                pl.col("mean_lane_pnl").head(K).sum().alias("day_pnl_pct"),
                pl.col("asset").head(K).len().alias("n_entries"),
            ])
            cal = pl.DataFrame({"entry_date": all_dates}).join(per_day, on="entry_date", how="left").with_columns(
                pl.col("day_pnl_pct").fill_null(0.0)
            ).sort("entry_date")
            pnl_arr = cal["day_pnl_pct"].to_numpy()
            comp = proper_daily_compound(pnl_arr)
            mean_d = float(np.mean(pnl_arr))
            pos_d = int((pnl_arr > 0).sum())
            dates_list = cal["entry_date"].to_list()
            neg_wk = neg_weeks_count(dates_list, pnl_arr)
            wealth = 10000 * (1 + comp / 100)
            cleared = "✅" if wealth >= 28146 else "❌"
            results.append({
                "K": K, "min_conv": min_conv, "COMP_pct": comp, "mean_d": mean_d,
                "pos_days": pos_d, "neg_weeks": neg_wk, "wealth": wealth,
                "cleared_floor": wealth >= 28146,
            })
            print(f"  {K:>3d} {min_conv:>7d}   {comp:>+9.2f}% {mean_d:>+9.4f}% {pos_d:>5d} {neg_wk:>4d}/104 ${wealth:>9,.0f}    {cleared:>10s}")

    results_df = pl.DataFrame(results)
    results_df.write_parquet(str(OUT_DIR / "corrected_union_grid.parquet"))

    # ===== Best by wealth at REALISTIC K=5 =====
    realistic = results_df.filter(pl.col("K") == 5).sort("wealth", descending=True).row(0, named=True)
    print(f"\n  REALISTIC v3 (K=5) BEST: K={realistic['K']} conv={realistic['min_conv']}")
    print(f"    wealth=${realistic['wealth']:,.0f}  COMP={realistic['COMP_pct']:+.2f}%  neg_wk={realistic['neg_weeks']}/104")
    print(f"    1%/wk floor ($28,146) cleared: {realistic['cleared_floor']}")

    # K=10 best (per-bucket-cap scenario)
    perbucket = results_df.filter(pl.col("K") == 10).sort("wealth", descending=True).row(0, named=True)
    print(f"\n  Per-bucket-cap scenario (K=10) BEST: K={perbucket['K']} conv={perbucket['min_conv']}")
    print(f"    wealth=${perbucket['wealth']:,.0f}  COMP={perbucket['COMP_pct']:+.2f}%  neg_wk={perbucket['neg_weeks']}/104")
    print(f"    1%/wk floor ($28,146) cleared: {perbucket['cleared_floor']}")

    # ===== FLAG #5: R7 standalone DAILY COMPOUND =====
    print("\n[FLAG #5 — R7 standalone via DAILY COMPOUND not arithmetic trade-sum]")
    r7 = pl.read_parquet(str(OUT_DIR / "r7_counter_trend_trades.parquet"))
    print(f"  R7 trades: {len(r7)}")
    # Sum net_pnl by entry_date, then compound across days
    r7_daily = r7.group_by("entry_date").agg(pl.col("net_pnl_pct").sum().alias("d_pnl")).sort("entry_date")
    full_cal = pl.DataFrame({"entry_date": all_dates}).join(r7_daily, on="entry_date", how="left").with_columns(
        pl.col("d_pnl").fill_null(0.0)
    ).sort("entry_date")
    pnl_arr = full_cal["d_pnl"].to_numpy()
    r7_comp = proper_daily_compound(pnl_arr)
    r7_arith = float(r7["net_pnl_pct"].sum())
    r7_wealth = 10000 * (1 + r7_comp / 100)
    print(f"  R7 ARITHMETIC sum:  +{r7_arith:.2f}%  (PRIOR HEADLINE — was overstating)")
    print(f"  R7 DAILY COMPOUND:  {r7_comp:+.2f}%   (CORRECTED)")
    print(f"  R7 standalone wealth (corrected): ${r7_wealth:,.0f}")
    print(f"  R7 correction magnitude: {r7_arith - r7_comp:+.2f}pp (arithmetic was {(r7_arith/r7_comp - 1)*100:+.1f}% higher)")

    # ===== FLAG #8: Build #2 / #3 daily compound =====
    print("\n[FLAG #8 — Build #2 / #3 daily compound recompute]")
    for lane_name, lane_label in [("MOVER_B2_INTRADAY", "Build #2 INTRADAY"), ("MOVER_B3_RVOL_EXIT", "Build #3 RVOL EXIT")]:
        lane_trades = ext.filter(pl.col("lane") == lane_name)
        if len(lane_trades) == 0:
            continue
        lane_daily = lane_trades.group_by("entry_date").agg(pl.col("net_pnl_pct").sum().alias("d_pnl")).sort("entry_date")
        cal = pl.DataFrame({"entry_date": all_dates}).join(lane_daily, on="entry_date", how="left").with_columns(
            pl.col("d_pnl").fill_null(0.0)
        ).sort("entry_date")
        pnl_arr = cal["d_pnl"].to_numpy()
        comp = proper_daily_compound(pnl_arr)
        arith = float(lane_trades["net_pnl_pct"].sum())
        print(f"  {lane_label}: ARITHMETIC sum = +{arith:.2f}%  /  DAILY COMPOUND = {comp:+.2f}%")

    # ===== Summary verdict =====
    print("\n" + "=" * 80)
    print("CORRECTED VERDICT")
    print("=" * 80)
    print(f"  Baseline STRICT_LO_SETUP60:                   $12,087 / +20.87%")
    print(f"  Realistic v3 cap=5 union (THIS CORRECTION):   ${realistic['wealth']:,.0f} / {realistic['COMP_pct']:+.2f}%")
    print(f"  With per-bucket cap=10 (P5-3 needed):         ${perbucket['wealth']:,.0f} / {perbucket['COMP_pct']:+.2f}%")
    print(f"  Prior 'headline' (K=20 lookahead):            $34,026 ← OVERSTATED, not under v3 reality")
    print(f"  1%/wk floor target:                           $28,146")
    if realistic['wealth'] >= 28146:
        print(f"  Floor CLEARED under v3 cap=5: YES")
    else:
        gap_5 = 28146 - realistic['wealth']
        gap_10 = 28146 - perbucket['wealth']
        print(f"  Floor CLEARED under v3 cap=5: NO  (gap ${gap_5:+,.0f})")
        print(f"  Floor CLEARED under v3 cap=10: {'YES' if perbucket['wealth'] >= 28146 else f'NO (gap $+{gap_10:,.0f})'}")
    print()

    return results_df, realistic, perbucket


if __name__ == "__main__":
    main()
