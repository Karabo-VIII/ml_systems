"""build_r7_v2_regime_gated.py — R7-v2: regime-gated counter-trend specialist.

R7-v1 had a 25Q1 disaster (-42%) because deep-bear capitulations keep going down
("catching falling knives"). R7-v2 adds a regime gate: only fire when BTC 30d is
NOT in extreme bear (btc_30d > -0.10) AND the asset is in DEGEN/VOLATILE bucket
(where bounces are statistically strongest).

Also tests trigger thresholds (-10%, -15%, -20%) and hold horizons (3d, 5d).

Outputs:
  runs/audit/r7_v2_counter_trend_trades.parquet (best config)
  runs/audit/P5_2_R7_V2_REGIME_GATED_VERDICT_2026_05_19.md
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
    panel = pl.read_parquet(str(OUT_DIR / "oracle_panel_binance_2026_05_18.parquet"))
    print(f"[r7-v2] panel rows={len(panel)}")
    print(f"  schema sample: {panel.columns[:10]}")

    # Add ret_1d + forward returns
    panel = panel.sort(["asset", "date"]).with_columns(
        pl.col("close").pct_change(1).over("asset").alias("ret_1d")
    ).with_columns([
        pl.col("close").shift(-1).over("asset").alias("close_t1"),
        pl.col("close").shift(-2).over("asset").alias("close_t2"),
        pl.col("close").shift(-4).over("asset").alias("close_t4"),
        pl.col("close").shift(-6).over("asset").alias("close_t6"),
    ]).with_columns([
        ((pl.col("close_t2") / pl.col("close_t1")) - 1).alias("fwd_1d"),
        ((pl.col("close_t4") / pl.col("close_t1")) - 1).alias("fwd_3d"),
        ((pl.col("close_t6") / pl.col("close_t1")) - 1).alias("fwd_5d"),
    ])
    panel = panel.with_columns(pl.col("date").cast(pl.Date, strict=False))

    # BTC regime: extract BTC 30d return per date
    btc = panel.filter(pl.col("asset") == "BTC").sort("date").with_columns(
        (pl.col("close") / pl.col("close").shift(30) - 1).alias("btc_30d")
    ).select(["date", "btc_30d"])
    print(f"  BTC regime: {len(btc)} rows; range {btc['btc_30d'].min():.3f} -> {btc['btc_30d'].max():.3f}")

    # Join btc_30d back
    panel = panel.join(btc, on="date", how="left")

    # Bucket mapping
    DEGEN = ["BONK", "PEPE", "SHIB", "WIF", "WLD"]
    STEADY = ["BCH", "BNB", "ETC", "SOL", "TRX", "XRP"]
    BLUE = ["BTC", "ETH"]

    def bucket_for(a):
        if a in BLUE: return "BLUE"
        if a in STEADY: return "STEADY"
        if a in DEGEN: return "DEGEN"
        return "VOLATILE"

    panel = panel.with_columns(
        pl.col("asset").map_elements(bucket_for, return_dtype=pl.Utf8).alias("bucket")
    )
    panel = panel.filter((pl.col("date") >= date(2024,1,1)) & (pl.col("date") <= date(2025,12,31)))
    print(f"  8Q panel: {len(panel)} rows")

    # === GRID: trigger x hold x regime-filter x bucket-filter ===
    print("\n[r7-v2] Counter-trend grid (8Q NAV @ 4% size):")
    print(f"  {'trigger':>10} {'hold':>5} {'regime_gate':>14} {'bucket_filter':>14} {'n':>5} {'mean_net':>10} {'NAV_8Q':>9}")
    results = []
    for trig in [-0.10, -0.15, -0.20]:
        for hold_col, hold in [("fwd_3d", 3), ("fwd_5d", 5)]:
            # 4 gate configs:
            for regime_gate_label, regime_filter in [
                ("none", pl.lit(True)),
                ("btc>-0.10", pl.col("btc_30d") > -0.10),
                ("btc>-0.05", pl.col("btc_30d") > -0.05),
                ("btc>0", pl.col("btc_30d") > 0),
            ]:
                for bucket_filter_label, bucket_filter in [
                    ("ALL", pl.lit(True)),
                    ("DEGEN+VOL", pl.col("bucket").is_in(["DEGEN", "VOLATILE"])),
                ]:
                    events = panel.filter(
                        (pl.col("ret_1d") <= trig) & regime_filter & bucket_filter
                    ).drop_nulls(subset=[hold_col])
                    n = len(events)
                    if n < 20:
                        continue
                    mean_fwd = float(events[hold_col].mean())
                    mean_net = (mean_fwd - COST_RT) * SIZE * 100
                    nav_8q = (events[hold_col].sum() - COST_RT * n) * SIZE * 100
                    results.append({
                        "trigger": trig, "hold": hold, "regime_gate": regime_gate_label,
                        "bucket_filter": bucket_filter_label, "n": n,
                        "mean_net_pct": mean_net, "nav_8q_pct": float(nav_8q),
                    })
                    print(f"  {trig:>+9.2f} {hold:>4d}d {regime_gate_label:>14} {bucket_filter_label:>14} {n:>5d} {mean_net:>+9.4f}% {float(nav_8q):>+8.2f}%")

    results_df = pl.DataFrame(results)
    # Best by NAV that also has n >= 100
    eligible = results_df.filter(pl.col("n") >= 100).sort("nav_8q_pct", descending=True)
    if len(eligible) > 0:
        best = eligible.head(1).row(0, named=True)
        print(f"\n  BEST (n>=100): {best}")
    else:
        best = results_df.sort("nav_8q_pct", descending=True).head(1).row(0, named=True)
        print(f"\n  BEST (any n): {best}")

    # === Build best R7-v2 trade ledger ===
    print(f"\n[r7-v2] building best-config trade ledger...")
    best_trig = best["trigger"]
    best_hold_col = "fwd_3d" if best["hold"] == 3 else "fwd_5d"
    best_regime = best["regime_gate"]
    best_bucket = best["bucket_filter"]

    regime_expr = {
        "none": pl.lit(True),
        "btc>-0.10": pl.col("btc_30d") > -0.10,
        "btc>-0.05": pl.col("btc_30d") > -0.05,
        "btc>0": pl.col("btc_30d") > 0,
    }[best_regime]
    bucket_expr = pl.col("bucket").is_in(["DEGEN", "VOLATILE"]) if best_bucket == "DEGEN+VOL" else pl.lit(True)

    r7v2 = panel.filter((pl.col("ret_1d") <= best_trig) & regime_expr & bucket_expr).drop_nulls(subset=[best_hold_col])
    r7v2_trades = r7v2.with_columns([
        pl.lit("R7_V2_REGIME_GATED").alias("lane"),
        pl.col("date").dt.offset_by("1d").alias("entry_date"),
        pl.col("date").dt.offset_by(f"{best['hold']+1}d").alias("exit_date"),
        (pl.col(best_hold_col) * 100).alias("gross_ret_pct"),
        ((pl.col(best_hold_col) - COST_RT) * SIZE * 100).alias("net_pnl_pct"),
        pl.lit(SIZE).alias("size_pct"),
        pl.lit("counter_trend_v2").alias("exit_reason"),
        pl.lit("").alias("sleeve"),
        pl.lit("").alias("strategy_id"),
        pl.lit("").alias("quarter"),
        pl.lit("long").alias("side"),
        pl.lit(COST_RT*100).alias("cost_pct"),
    ]).select(["lane","asset","entry_date","exit_date","size_pct","gross_ret_pct","cost_pct","net_pnl_pct","exit_reason","sleeve","strategy_id","quarter","side"])

    r7v2_trades.write_parquet(str(OUT_DIR / "r7_v2_counter_trend_trades.parquet"))
    print(f"  wrote {len(r7v2_trades)} R7-v2 trades  net_sum={r7v2_trades['net_pnl_pct'].sum():.2f}%")

    # Per-quarter
    r7v2_q = r7v2_trades.with_columns([
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
    print(f"\n  Per-quarter:")
    for r in r7v2_q.group_by("quarter").agg([
        pl.len().alias("n"),
        pl.col("net_pnl_pct").sum().alias("sum_net"),
        (pl.col("net_pnl_pct") > 0).cast(pl.Float64).mean().alias("wr"),
    ]).sort("quarter").iter_rows(named=True):
        print(f"    {r['quarter']}: n={r['n']:>4d}  sum_net={r['sum_net']:>+7.2f}%  wr={r['wr']*100:.1f}%")

    # === MERGE R7-v2 INTO EXTENDED UNION + REMEASURE ===
    print(f"\n[r7-v2] merging into extended union + re-measuring...")
    ext = pl.read_parquet(str(OUT_DIR / "union_extended_trade_ledger.parquet"))
    r7v2_norm = r7v2_trades.select(ext.columns)
    combined = pl.concat([ext, r7v2_norm])

    atlas = combined.group_by(["asset", "entry_date"]).agg([
        pl.col("lane").n_unique().alias("n_lanes_firing"),
        pl.col("net_pnl_pct").mean().alias("mean_lane_pnl"),
        pl.col("size_pct").first().alias("size_pct"),
    ]).sort("entry_date")
    daily = atlas.group_by("entry_date").agg(pl.col("asset").n_unique().alias("n_assets")).sort("entry_date")
    print(f"  Coverage cells: {len(atlas)}; active days: {len(daily)}/731 ({len(daily)/731*100:.1f}%)")

    # Composed simulation
    print(f"\n  R7-v2 augmented composed engine:")
    print(f"  {'K_cap':>6} {'min_n_lanes':>12} {'COMP':>9} {'mean_d':>10} {'neg_wk':>8} {'wealth':>10}")
    best_overall = None
    for K in [10, 20]:
        for min_conv in [1, 2]:
            cell = atlas.with_columns(pl.col("n_lanes_firing").cast(pl.Float64).alias("priority"))
            per_day = cell.filter(pl.col("priority") >= min_conv).sort(
                ["entry_date", "priority"], descending=[False, True]
            ).group_by("entry_date").agg([
                pl.col("mean_lane_pnl").head(K).sum().alias("day_pnl_pct"),
            ])
            all_dates = pl.date_range(date(2024,1,1), date(2025,12,31), interval="1d", eager=True)
            cal = pl.DataFrame({"entry_date": all_dates}).join(per_day, on="entry_date", how="left").with_columns(
                pl.col("day_pnl_pct").fill_null(0.0)
            ).sort("entry_date")
            pnl_arr = cal["day_pnl_pct"].to_numpy()
            comp = float((np.exp(np.log1p(pnl_arr / 100).sum()) - 1) * 100)
            cal_wk = cal.with_columns([
                pl.col("entry_date").dt.year().alias("y"),
                pl.col("entry_date").dt.week().alias("w"),
            ]).with_columns(
                (pl.col("y").cast(pl.Utf8) + "-W" + pl.col("w").cast(pl.Utf8).str.zfill(2)).alias("iso_week")
            )
            cal_wk_pd = cal_wk.select(["iso_week", "day_pnl_pct"]).to_pandas()
            wk = cal_wk_pd.groupby("iso_week")["day_pnl_pct"].apply(lambda x: float(np.exp(np.log1p(x.values/100).sum()) - 1))
            neg_wk = int((wk < 0).sum())
            wealth = 10000 * (1 + comp/100)
            mean_d = float(np.mean(pnl_arr))
            entry = {"K":K,"conv":min_conv,"COMP":comp,"mean_d":mean_d,"neg_wk":neg_wk,"wealth":wealth}
            if best_overall is None or wealth > best_overall["wealth"]:
                best_overall = entry
            print(f"  {K:>6d} {min_conv:>9d}    {comp:>+8.2f}% {mean_d:>+9.4f}% {neg_wk:>5d}/104 ${wealth:>9,.0f}")

    print(f"\n  R7-v2 best: K={best_overall['K']} conv={best_overall['conv']}")
    print(f"  COMP={best_overall['COMP']:+.2f}%  wealth=${best_overall['wealth']:,.0f}  neg_wk={best_overall['neg_wk']}/104")
    print(f"\n  vs extended 17-lane (no R7):   $34,026 / +240.26% / 55 neg wk")
    print(f"  vs R7-v1 augmented (naive):    $33,369 / +233.69% / 56 neg wk")
    print(f"  vs R7-v2 augmented (this):     ${best_overall['wealth']:,.0f} / {best_overall['COMP']:+.2f}% / {best_overall['neg_wk']} neg wk")


if __name__ == "__main__":
    main()
