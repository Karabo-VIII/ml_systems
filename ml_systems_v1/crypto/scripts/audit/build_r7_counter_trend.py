"""build_r7_counter_trend.py — P5-1: R7 counter-trend (capitulation bounce) specialist.

Anti-correlated to mover-continuation: fires when ret_1d <= -threshold (capitulation),
expects multi-day bounce. Tests the hypothesis that adding a counter-trend specialist
covers the bear-quarter neg-weeks that momentum lanes can't.

Uses Binance kline panel already pulled (oracle_panel_binance_2026_05_18.parquet).
Same 24bps RT cost, 4% NAV per entry, max 5 cap.

Triggers tested:
  ret_1d <= -10%
  ret_1d <= -15%
  ret_1d <= -20%

Forward horizons: 1d, 3d, 5d (from t+1 close entry).

Outputs:
  runs/audit/r7_counter_trend_trades.parquet — trade ledger for the picked config
  runs/audit/P5_1_R7_COUNTER_TREND_VERDICT_2026_05_19.md — verdict + integration
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

COST_RT = 0.0024
SIZE = 0.04


def main():
    print("[r7] loading Binance panel...")
    panel = pl.read_parquet(str(OUT_DIR / "oracle_panel_binance_2026_05_18.parquet"))
    print(f"  panel rows={len(panel)}  cols={len(panel.columns)}  sample cols={panel.columns[:8]}")

    # Determine schema — find ret_1d
    needed = ["asset", "date", "close", "ret_1d"]
    has_ret_1d = "ret_1d" in panel.columns
    if not has_ret_1d and "close" in panel.columns:
        panel = panel.sort(["asset", "date"]).with_columns(
            pl.col("close").pct_change(1).over("asset").alias("ret_1d")
        )

    # Filter to 8Q
    if "date" in panel.columns:
        panel = panel.with_columns(pl.col("date").cast(pl.Date, strict=False))
        panel = panel.filter((pl.col("date") >= date(2024,1,1)) & (pl.col("date") <= date(2025,12,31)))
    print(f"  8Q panel rows: {len(panel)}")

    # Compute forward returns from t+1 close
    panel = panel.sort(["asset", "date"]).with_columns([
        pl.col("close").shift(-1).over("asset").alias("close_t1"),
        pl.col("close").shift(-2).over("asset").alias("close_t2"),
        pl.col("close").shift(-4).over("asset").alias("close_t4"),
        pl.col("close").shift(-6).over("asset").alias("close_t6"),
    ]).with_columns([
        ((pl.col("close_t2") / pl.col("close_t1")) - 1).alias("fwd_1d"),
        ((pl.col("close_t4") / pl.col("close_t1")) - 1).alias("fwd_3d"),
        ((pl.col("close_t6") / pl.col("close_t1")) - 1).alias("fwd_5d"),
    ])

    # === DISTRIBUTION TABLES for capitulation triggers ===
    print("\n[r7] distribution of fwd returns after capitulation triggers:")
    for thresh in [-0.10, -0.15, -0.20]:
        events = panel.filter(pl.col("ret_1d") <= thresh).drop_nulls(subset=["fwd_1d","fwd_3d","fwd_5d"])
        n = len(events)
        if n == 0:
            print(f"  ret_1d <= {thresh:+.2f}: 0 events")
            continue
        print(f"  ret_1d <= {thresh:+.2f}: {n} events")
        for hcol, hname in [("fwd_1d","1d"), ("fwd_3d","3d"), ("fwd_5d","5d")]:
            s = events[hcol].drop_nulls()
            mean = float(s.mean())
            median = float(s.median())
            p10 = float(s.quantile(0.10))
            p90 = float(s.quantile(0.90))
            pos = float((s > 0).cast(pl.Float64).mean())
            print(f"    fwd_{hname}: mean={mean*100:+6.3f}% med={median*100:+6.3f}% p10={p10*100:+6.2f}% p90={p90*100:+6.2f}% pos={pos*100:.1f}%")

    # === SIMULATE R7: ret_1d <= -10%, 3d hold ===
    print("\n[r7] simulating R7 strategy: ret_1d <= -10% trigger, t+1 close entry, 3d hold...")
    for thresh, label in [(-0.10, "ret_1d<=-10%"), (-0.15, "ret_1d<=-15%")]:
        for hold_col, hold_name in [("fwd_3d", "3d"), ("fwd_5d", "5d")]:
            events = panel.filter(pl.col("ret_1d") <= thresh).drop_nulls(subset=[hold_col])
            n = len(events)
            if n == 0:
                continue
            gross = events[hold_col].sum()
            net = float(gross) - COST_RT * n
            nav_8q = SIZE * net * 100
            mean_net = (float(events[hold_col].mean()) - COST_RT) * SIZE * 100
            print(f"  {label} × hold {hold_name}: n={n} gross={float(gross)*100:+.2f}% net_sum={net*100:+.2f}% mean_net_per_trade={mean_net:+.4f}% NAV_8Q@4%={nav_8q:+.2f}%")

    # === BUILD R7 TRADE LEDGER (chosen: -10% × 3d) ===
    print("\n[r7] building R7 trade ledger (ret_1d <= -10%, 3d hold)...")
    chosen_thresh = -0.10
    chosen_hold_col = "fwd_3d"
    r7_events = panel.filter(pl.col("ret_1d") <= chosen_thresh).drop_nulls(subset=[chosen_hold_col])
    r7_trades = r7_events.with_columns([
        pl.lit("R7_COUNTER_TREND").alias("lane"),
        pl.col("date").alias("trigger_date"),
        # entry_date = trigger_date + 1 (we enter at t+1 close)
        pl.col("date").dt.offset_by("1d").alias("entry_date"),
        pl.col("date").dt.offset_by("4d").alias("exit_date"),
        (pl.col(chosen_hold_col) * 100).alias("gross_ret_pct"),
        ((pl.col(chosen_hold_col) - COST_RT) * SIZE * 100).alias("net_pnl_pct"),
        pl.lit(SIZE).alias("size_pct"),
        pl.lit("counter_trend_hold").alias("exit_reason"),
        pl.lit("").alias("sleeve"),
        pl.lit("").alias("strategy_id"),
        pl.lit("").alias("quarter"),
        pl.lit("long").alias("side"),
        pl.lit(COST_RT*100).alias("cost_pct"),
    ]).select(["lane","asset","entry_date","exit_date","size_pct","gross_ret_pct","cost_pct","net_pnl_pct","exit_reason","sleeve","strategy_id","quarter","side"])
    r7_trades = r7_trades.filter(
        (pl.col("entry_date") >= date(2024,1,1)) & (pl.col("entry_date") <= date(2025,12,31))
    )
    r7_trades.write_parquet(str(OUT_DIR / "r7_counter_trend_trades.parquet"))
    print(f"  wrote {len(r7_trades)} R7 trades")

    # Per-quarter R7 performance
    print("\n  R7 per-quarter NAV contribution:")
    r7_q = r7_trades.with_columns([
        pl.col("entry_date").dt.year().alias("y"),
        pl.col("entry_date").dt.month().alias("m"),
    ]).with_columns(
        pl.when(pl.col("m") <= 3).then(pl.lit("Q1"))
          .when(pl.col("m") <= 6).then(pl.lit("Q2"))
          .when(pl.col("m") <= 9).then(pl.lit("Q3"))
          .otherwise(pl.lit("Q4")).alias("q")
    ).with_columns(
        (pl.col("y").cast(pl.Utf8) + pl.col("q")).alias("quarter")
    )
    qg = r7_q.group_by("quarter").agg([
        pl.len().alias("n"),
        pl.col("net_pnl_pct").sum().alias("sum_net"),
        pl.col("net_pnl_pct").mean().alias("mean_net"),
        (pl.col("net_pnl_pct") > 0).cast(pl.Float64).mean().alias("wr"),
    ]).sort("quarter")
    for r in qg.iter_rows(named=True):
        print(f"    {r['quarter']}: n={r['n']:>4d}  sum_net={r['sum_net']:+.2f}%  mean={r['mean_net']:+.4f}%  wr={r['wr']*100:.1f}%")

    # === MERGE INTO EXTENDED UNION ===
    print("\n[r7] merging R7 into extended union...")
    ext = pl.read_parquet(str(OUT_DIR / "union_extended_trade_ledger.parquet"))
    # Add R7 trades with same schema
    r7_normalized = r7_trades.select(ext.columns)
    combined = pl.concat([ext, r7_normalized])
    print(f"  combined trades: {len(combined)} / {combined['lane'].n_unique()} lanes")

    # Recompute coverage atlas + composed engine
    atlas = combined.group_by(["asset", "entry_date"]).agg([
        pl.col("lane").n_unique().alias("n_lanes_firing"),
        pl.col("net_pnl_pct").mean().alias("mean_lane_pnl"),
        pl.col("size_pct").first().alias("size_pct"),
    ]).sort("entry_date")
    print(f"  Coverage cells: {len(atlas)}")
    daily = atlas.group_by("entry_date").agg(pl.col("asset").n_unique().alias("n_assets")).sort("entry_date")
    print(f"  Active days: {len(daily)}/731 ({len(daily)/731*100:.1f}%)")

    # Simulate
    print("\n  R7-augmented composed engine:")
    print(f"  {'K_cap':>6} {'min_n_lanes':>12} {'COMP':>9} {'mean_d':>10} {'neg_wk':>8} {'wealth':>10}")
    results = []
    for K in [5, 10, 20]:
        for min_conv in [1, 2, 3]:
            cell = atlas.with_columns(pl.col("n_lanes_firing").cast(pl.Float64).alias("priority"))
            per_day = cell.filter(pl.col("priority") >= min_conv).sort(
                ["entry_date", "priority"], descending=[False, True]
            ).group_by("entry_date").agg([
                pl.col("mean_lane_pnl").head(K).sum().alias("day_pnl_pct"),
                pl.col("asset").head(K).len().alias("n_entries"),
            ])
            all_dates = pl.date_range(date(2024,1,1), date(2025,12,31), interval="1d", eager=True)
            cal = pl.DataFrame({"entry_date": all_dates}).join(per_day, on="entry_date", how="left").with_columns([
                pl.col("day_pnl_pct").fill_null(0.0),
            ]).sort("entry_date")
            pnl_arr = cal["day_pnl_pct"].to_numpy()
            comp = float((np.exp(np.log1p(pnl_arr / 100).sum()) - 1) * 100)
            mean_d = float(np.mean(pnl_arr))
            cal_wk = cal.with_columns([
                pl.col("entry_date").dt.year().alias("y"),
                pl.col("entry_date").dt.week().alias("w"),
            ]).with_columns(
                (pl.col("y").cast(pl.Utf8) + "-W" + pl.col("w").cast(pl.Utf8).str.zfill(2)).alias("iso_week")
            )
            cal_wk_pd = cal_wk.select(["iso_week", "day_pnl_pct"]).to_pandas()
            wk = cal_wk_pd.groupby("iso_week")["day_pnl_pct"].apply(lambda x: float(np.exp(np.log1p(x.values/100).sum()) - 1))
            neg_wk = int((wk < 0).sum())
            wealth = 10000 * (1 + comp / 100)
            results.append({"K":K,"min_conv":min_conv,"COMP":comp,"mean_d":mean_d,"neg_wk":neg_wk,"wealth":wealth})
            print(f"  {K:>6d} {min_conv:>9d}    {comp:>+8.2f}% {mean_d:>+9.4f}% {neg_wk:>5d}/104 ${wealth:>9,.0f}")

    best = max(results, key=lambda r: r["wealth"])
    best_min_negwk = min(results, key=lambda r: r["neg_wk"])
    print(f"\n  BEST wealth: K={best['K']} conv={best['min_conv']}  COMP={best['COMP']:+.2f}%  wealth=${best['wealth']:,.0f}  neg_wk={best['neg_wk']}/104")
    print(f"  BEST neg-wk: K={best_min_negwk['K']} conv={best_min_negwk['min_conv']}  COMP={best_min_negwk['COMP']:+.2f}%  wealth=${best_min_negwk['wealth']:,.0f}  neg_wk={best_min_negwk['neg_wk']}/104")

    print(f"\n  Comparison:")
    print(f"    Extended 17-lane union (no R7):  $34,026 / +240.26% / 55 neg wk")
    print(f"    R7-augmented union (best):       ${best['wealth']:>9,.0f} / {best['COMP']:>+7.2f}% / {best['neg_wk']} neg wk")
    delta = best['wealth'] - 34026
    print(f"    Wealth delta from R7: ${delta:+,.0f}")
    print(f"    Neg-wk delta from R7: {best['neg_wk'] - 55:+d}")


if __name__ == "__main__":
    main()
