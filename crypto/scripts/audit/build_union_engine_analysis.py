"""build_union_engine_analysis.py — P4-1c through P4-6: union engine analysis.

Combines existing trade ledgers from STRICT_LO_SETUP60, R-MDH, R-MDH+BC, and the
6 mover-lane builds. Computes:
  - Per-day union: which lanes have intents on each (asset, date)?
  - Conjunction structure: when N lanes co-fire, what's the per-event EV?
  - Composed simulation: cap-aware sizing, cash-bias on no-fire days, K-position cap grid.
  - 8Q WF NAV: union vs each individual lane.

Outputs:
  runs/audit/union_trade_ledger.parquet — all trades across lanes (normalized)
  runs/audit/union_daily_intents.parquet — per-day intent inventory
  runs/audit/union_composed_daily.parquet — composed engine per-day NAV
  runs/audit/P4_END_TO_END_UNION_VERDICT_2026_05_19.md — final verdict
"""
from __future__ import annotations
import os
import math
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

os.environ["PYTHONIOENCODING"] = "utf-8"
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "audit"

# Mover-lane builds used SIZE_PCT=0.04 and COST_RT=0.0024 (24bps RT)
# Build #1/#3 ledgers have 'gross_ret' or 'gross_ret_pct' and need
# net_pnl_pct conversion.
COST_RT_MOVER = 0.0024
SIZE_MOVER = 0.04


def normalize_strict(path: Path, lane: str) -> pl.DataFrame:
    df = pl.read_parquet(str(path))
    return df.select([
        pl.lit(lane).alias("lane"),
        "asset", "entry_date", "exit_date",
        pl.col("net_pnl_pct").cast(pl.Float64),
        pl.col("gross_ret_pct").cast(pl.Float64),
        pl.col("size_pct").cast(pl.Float64),
        "exit_reason", "quarter",
    ])


def normalize_mover_b1_b3(path: Path, lane: str) -> pl.DataFrame:
    """Mover Build #1 and #3 ledger schema."""
    df = pl.read_parquet(str(path))
    # entry_date_t1 is the actual entry date; gross_ret is decimal
    # Compute net_pnl_pct = (gross_ret - COST_RT) * SIZE * 100  (as % NAV)
    gross_col = "gross_ret" if "gross_ret" in df.columns else "gross_ret_pct"
    # If gross_ret_pct, assume it's already in pct; otherwise decimal
    gross_factor = 100 if gross_col == "gross_ret" else 1
    return df.with_columns([
        pl.lit(lane).alias("lane"),
        (pl.col(gross_col) * gross_factor).alias("gross_ret_pct"),
        (((pl.col(gross_col) * gross_factor / 100) - COST_RT_MOVER) * SIZE_MOVER * 100).alias("net_pnl_pct"),
        pl.lit(SIZE_MOVER).alias("size_pct"),
    ]).select([
        "lane",
        "asset",
        pl.col("entry_date_t1").alias("entry_date"),
        "exit_date",
        "net_pnl_pct",
        "gross_ret_pct",
        "size_pct",
        pl.col("exit_reason").cast(pl.Utf8) if "exit_reason" in df.columns else pl.lit("unknown").alias("exit_reason"),
        pl.lit("").alias("quarter"),
    ])


def normalize_mover_b2(path: Path, lane: str) -> pl.DataFrame:
    """Build #2 intraday ledger has a different schema."""
    df = pl.read_parquet(str(path))
    # entry_dt is a datetime; need to extract date
    if df.schema.get("entry_dt") in (pl.Datetime, pl.Datetime("us"), pl.Datetime("ns"), pl.Datetime("ms")):
        df = df.with_columns(pl.col("entry_dt").dt.date().alias("entry_date"))
    else:
        df = df.with_columns(pl.col("entry_dt").str.to_date().alias("entry_date"))
    if "exit_dt" in df.columns:
        if df.schema.get("exit_dt") in (pl.Datetime, pl.Datetime("us")):
            df = df.with_columns(pl.col("exit_dt").dt.date().alias("exit_date"))
        else:
            df = df.with_columns(pl.col("exit_dt").str.to_date().alias("exit_date"))
    else:
        # If only trigger_date is present, exit_date = trigger_date + 5d
        df = df.with_columns(pl.col("entry_date").dt.offset_by("5d").alias("exit_date"))
    # net_ret in decimal; convert to % NAV
    return df.select([
        pl.lit(lane).alias("lane"),
        "asset",
        "entry_date",
        "exit_date",
        (pl.col("net_ret") * SIZE_MOVER * 100).alias("net_pnl_pct"),
        (pl.col("gross_ret") * 100).alias("gross_ret_pct"),
        pl.lit(SIZE_MOVER).alias("size_pct"),
        pl.lit("intraday").alias("exit_reason"),
        pl.lit("").alias("quarter"),
    ])


def main():
    print("[union] loading all trade ledgers...")
    union_frames = []

    union_frames.append(normalize_strict(OUT_DIR / "capture_ratio_scoreboard_v2_pertrade.parquet", "STRICT_BASELINE"))
    union_frames.append(normalize_strict(OUT_DIR / "rmdh_only_scoreboard_pertrade.parquet", "R_MDH"))
    union_frames.append(normalize_strict(OUT_DIR / "rmdh_scoreboard_pertrade.parquet", "R_MDH_BC"))

    mover_files = [
        ("MOVER_B1_15_5d_GATED", "mover_lane_trades_BUILD1_+15%_5d_GATED_BUCKETED.parquet"),
        ("MOVER_B1_15_5d_BULL", "mover_lane_trades_BUILD1_+15%_5d_BULLONLY_BUCKETED.parquet"),
        ("MOVER_B1_25_5d_GATED", "mover_lane_trades_BUILD1_+25%_5d_GATED_BUCKETED.parquet"),
        ("MOVER_B1+3_25_5d_BULL_RVOL", "mover_lane_trades_BUILD1+3_+25%_5d_BULLONLY_RVOL.parquet"),
        ("MOVER_B3_RVOL_EXIT", "mover_lane_trades_BUILD3_+15%_5d_GATED_RVOL.parquet"),
    ]
    for lane, fname in mover_files:
        union_frames.append(normalize_mover_b1_b3(OUT_DIR / fname, lane))

    union_frames.append(normalize_mover_b2(OUT_DIR / "mover_lane_trades_BUILD2_INTRADAY.parquet", "MOVER_B2_INTRADAY"))

    union = pl.concat(union_frames, how="diagonal_relaxed")
    union = union.with_columns([
        pl.col("entry_date").cast(pl.Date),
        pl.col("exit_date").cast(pl.Date),
    ])

    # Filter to 8Q WF window (24Q1-25Q4)
    union = union.filter((pl.col("entry_date") >= date(2024, 1, 1)) & (pl.col("entry_date") <= date(2025, 12, 31)))
    union = union.sort(["entry_date", "lane", "asset"])
    union.write_parquet(str(OUT_DIR / "union_trade_ledger.parquet"))

    print(f"[union] total trades across 9 lanes: {len(union)}")
    print(f"[union] date range: {union['entry_date'].min()} -> {union['entry_date'].max()}")
    print(f"\nPer-lane summary:")
    by_lane = union.group_by("lane").agg([
        pl.len().alias("n"),
        pl.col("net_pnl_pct").mean().alias("mean_net"),
        pl.col("net_pnl_pct").sum().alias("sum_net"),
        (pl.col("net_pnl_pct") > 0).cast(pl.Float64).mean().alias("wr"),
    ]).sort("sum_net", descending=True)
    print(f"  {'lane':<32} {'n':>5} {'mean_net':>10} {'sum_net':>10} {'wr':>6}")
    for r in by_lane.iter_rows(named=True):
        print(f"  {r['lane']:<32} {r['n']:>5d} {r['mean_net']:>+9.4f}% {r['sum_net']:>+9.2f}% {r['wr']*100:>5.1f}%")

    # === COVERAGE ATLAS ===
    print("\n[union] computing coverage atlas...")
    # For each (asset, entry_date), how many lanes fire?
    atlas = union.group_by(["asset", "entry_date"]).agg([
        pl.col("lane").n_unique().alias("n_lanes_firing"),
        pl.col("lane").unique().alias("lanes"),
        pl.col("net_pnl_pct").mean().alias("mean_lane_pnl"),
        pl.col("net_pnl_pct").sum().alias("sum_lane_pnl"),
        pl.col("net_pnl_pct").max().alias("best_lane_pnl"),
        pl.col("size_pct").first().alias("size_pct"),
    ]).sort("entry_date")

    print(f"  Unique (asset, date) cells with ≥1 lane: {len(atlas)}")

    # Coverage by # lanes
    print(f"\n  Conjunction structure (how many lanes co-fire per cell):")
    by_conj = atlas.group_by("n_lanes_firing").agg([
        pl.len().alias("n_cells"),
        pl.col("mean_lane_pnl").mean().alias("mean_pnl"),
        (pl.col("mean_lane_pnl") > 0).cast(pl.Float64).mean().alias("frac_pos"),
    ]).sort("n_lanes_firing")
    for r in by_conj.iter_rows(named=True):
        print(f"  {r['n_lanes_firing']} lanes: {r['n_cells']:>5d} cells  mean_pnl={r['mean_pnl']:+.4f}%  frac_pos={r['frac_pos']*100:.1f}%")

    # Daily union: per date, how many unique assets fire?
    daily = atlas.group_by("entry_date").agg([
        pl.col("asset").n_unique().alias("n_assets_firing"),
        pl.col("n_lanes_firing").sum().alias("total_lane_fires"),
        pl.col("mean_lane_pnl").mean().alias("avg_lane_pnl"),
    ]).sort("entry_date")
    active_days = len(daily)
    total_days_8q = 731
    print(f"\n  Active days (≥1 lane firing): {active_days}/{total_days_8q} ({active_days/total_days_8q*100:.1f}%)")
    print(f"  Mean assets firing per active day: {daily['n_assets_firing'].mean():.1f}")
    print(f"  Median assets firing per active day: {daily['n_assets_firing'].median():.0f}")
    print(f"  Days with ≥5 assets firing: {(daily['n_assets_firing'] >= 5).sum()}")
    print(f"  Days with ≥10 assets firing: {(daily['n_assets_firing'] >= 10).sum()}")
    print(f"  Days with ≥20 assets firing: {(daily['n_assets_firing'] >= 20).sum()}")

    atlas.write_parquet(str(OUT_DIR / "union_daily_intents.parquet"))

    # === COMPOSED ENGINE SIMULATION ===
    print("\n[union] running composed engine simulation (cap-aware, cash-bias)...")

    # Per (asset, date), pick the BEST lane (highest mean_lane_pnl)
    # Then per date, pick top-K assets by best_lane_pnl
    # Apply cap, cash on no-fire days

    # Step 1: per-cell conviction = n_lanes_firing (NO lookahead — conviction is the count
    # of independent specialists agreeing, NOT the realized outcome)
    # When we pick "top-K", we pick by conviction. Realized PnL is what we MEASURE.
    cell_best = atlas.with_columns([
        pl.col("n_lanes_firing").cast(pl.Float64).alias("priority"),
    ])

    def simulate_engine(intents: pl.DataFrame, K_cap: int, min_conviction: float = 0.0):
        """Simulate composed engine: per day, pick top-K cells by CONVICTION (n_lanes),
        then realize the mean_lane_pnl (NOT best — that would be lookahead).
        min_conviction = minimum n_lanes_firing to count a cell."""
        per_day = intents.filter(pl.col("priority") >= min_conviction).sort(
            ["entry_date", "priority"], descending=[False, True]
        ).group_by("entry_date").agg([
            pl.col("priority").head(K_cap).alias("top_priorities"),
            # USE MEAN_LANE_PNL — average across lanes that fired (no peek at best)
            pl.col("mean_lane_pnl").head(K_cap).sum().alias("day_pnl_pct"),
            pl.col("asset").head(K_cap).len().alias("n_entries"),
        ])
        # Build full calendar (with 0 on no-fire days)
        all_dates = pl.date_range(date(2024,1,1), date(2025,12,31), interval="1d", eager=True)
        cal = pl.DataFrame({"entry_date": all_dates}).join(per_day, on="entry_date", how="left").with_columns([
            pl.col("day_pnl_pct").fill_null(0.0),
            pl.col("n_entries").fill_null(0),
        ]).sort("entry_date")
        return cal

    # Grid: K_cap × min_n_lanes (conviction)
    print(f"\n  Composed engine grid (K_cap × min n_lanes_firing for conviction):")
    print(f"  {'K_cap':>6} {'min_n_lanes':>12} {'COMP':>9} {'mean_d':>10} {'pos_d':>8} {'neg_wk':>8} {'wealth':>10}")
    results = []
    for K in [5, 10, 20]:
        for min_conv in [1, 2, 3, 4]:
            cal = simulate_engine(cell_best, K, min_conv)
            pnl_arr = cal["day_pnl_pct"].to_numpy()
            comp = float((np.exp(np.log1p(pnl_arr / 100).sum()) - 1) * 100)
            mean_d = float(cal["day_pnl_pct"].mean())
            pos_d = int((cal["day_pnl_pct"] > 0).sum())
            # Weekly aggregate
            cal_wk = cal.with_columns([
                pl.col("entry_date").dt.year().alias("y"),
                pl.col("entry_date").dt.week().alias("w"),
            ])
            cal_wk = cal_wk.with_columns(
                (pl.col("y").cast(pl.Utf8) + "-W" + pl.col("w").cast(pl.Utf8).str.zfill(2)).alias("iso_week")
            )
            # Compute weekly returns via numpy (polars expr.exp() unavailable in this context)
            cal_wk_pd = cal_wk.select(["iso_week", "day_pnl_pct"]).to_pandas()
            wk_groups = cal_wk_pd.groupby("iso_week")["day_pnl_pct"].apply(
                lambda x: float(np.exp(np.log1p(x.values / 100).sum()) - 1)
            )
            weekly = pl.DataFrame({"iso_week": wk_groups.index.tolist(), "wk_ret": wk_groups.values.tolist()})
            n_neg_wk = int((weekly["wk_ret"] < 0).sum())
            wealth = 10000 * (1 + comp / 100)
            results.append({
                "K_cap": K,
                "min_conv": min_conv,
                "COMP_pct": comp,
                "mean_d": mean_d,
                "pos_d": pos_d,
                "neg_wk": n_neg_wk,
                "wealth": wealth,
                "n_total_days_fired": int((cal["n_entries"] > 0).sum()),
            })
            print(f"  {K:>6d} {min_conv:>9.3f}  {comp:>+8.2f}% {mean_d:>+9.4f}% {pos_d:>7d} {n_neg_wk:>5d}/104 ${wealth:>9,.0f}")

    results_df = pl.DataFrame(results)
    results_df.write_parquet(str(OUT_DIR / "union_composed_daily.parquet"))

    # Find best by wealth
    best = results_df.sort("wealth", descending=True).head(1).row(0, named=True)
    print(f"\n  BEST: K_cap={best['K_cap']} min_conv={best['min_conv']:+.2f}")
    print(f"        COMP={best['COMP_pct']:+.2f}% wealth=${best['wealth']:,.0f} neg_weeks={best['neg_wk']}/104 days_fired={best['n_total_days_fired']}")

    # Compare to single-strategy baselines
    print(f"\n  Comparison vs single-strategy baselines:")
    print(f"    STRICT_LO_SETUP60 (baseline):    $12,087 / 8Q COMP +20.87%")
    print(f"    R-MDH only:                       $13,459 / 8Q COMP +34.59%")
    print(f"    R-MDH + bear-cash:                $12,830 / 8Q COMP +28.30%")
    print(f"    MOVER_B2 INTRADAY (best single):  $15,908 / 8Q COMP +59.08%")
    print(f"    *** UNION ENGINE BEST:            ${best['wealth']:,.0f} / 8Q COMP {best['COMP_pct']:+.2f}% ***")
    print(f"    Delta vs best single (B2 +59.08%): {best['COMP_pct'] - 59.08:+.2f}pp")
    if best["wealth"] >= 28000:
        print("    Wealth target 1%/wk floor ($28k): CLEARED")
    else:
        gap = 28000 - best["wealth"]
        print(f"    Wealth target 1%/wk floor ($28k): Gap ${gap:+,.0f}")

    return atlas, results_df, best


if __name__ == "__main__":
    main()
