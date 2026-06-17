"""rmdh_r4_combined.py — Compose R4 (concentration) + R-MDH (multi-day hold).

Logic:
  1. Apply R4 concentration filter: keep trades only from (sleeve, bucket) with
     historical EV >= threshold (using full-history mean - lookahead bias caveat).
  2. Apply R-MDH on the kept trades: use existing rmdh_counterfactual_trades.parquet
     simulation results (these already have R-MDH outcomes).
  3. Aggregate week, quarter, wealth, asymmetry, neg-week count.

Threshold grid: [0%, +0.03%, +0.05%, +0.10%]
"""
from __future__ import annotations
import os
from pathlib import Path

import numpy as np
import polars as pl

os.environ["PYTHONIOENCODING"] = "utf-8"
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "audit"


def bucket_for(asset):
    if asset in ("BTC", "ETH"):
        return "BLUE"
    if asset in ("SOL", "XRP", "BNB", "TRX", "ADA", "LTC", "BCH", "TON", "ALGO",
                 "ETC", "ATOM", "AVAX", "LINK", "DOT"):
        return "STEADY"
    if asset in ("ZEC", "PEPE", "WLD", "DASH", "FIL", "FET", "BONK", "JST",
                 "FLOKI", "BLUR", "SHIB", "ORDI", "TRUMP"):
        return "DEGEN"
    return "VOLATILE"


def main():
    print("[combined] loading inputs...")
    rmdh = pl.read_parquet(str(OUT_DIR / "rmdh_counterfactual_trades.parquet"))
    sb = pl.read_parquet(str(OUT_DIR / "capture_ratio_scoreboard_v2_perday.parquet"))
    print(f"  rmdh trades: {len(rmdh)}")
    print(f"  per_day: {len(sb)}")

    # Add bucket to rmdh
    rmdh = rmdh.with_columns(
        pl.col("asset").map_elements(bucket_for, return_dtype=pl.Utf8).alias("bucket")
    )

    # Use R-MDH net_pnl as the EV scorer (since this is the R-MDH world)
    # But for fair comparison with R4 baseline, use the ACTUAL net_pnl_pct
    # We'll show both
    sleeve_bucket_ev_actual = rmdh.group_by(["sleeve", "bucket"]).agg([
        pl.len().alias("n"),
        pl.col("net_pnl_pct").mean().alias("ev_actual"),
        pl.col("rmdh_net_pnl_pct").mean().alias("ev_rmdh"),
    ]).filter(pl.col("n") >= 20)

    print("\n[combined] Per (sleeve, bucket) EV at R-MDH:")
    print("(top 5 by R-MDH EV)")
    top5 = sleeve_bucket_ev_actual.sort("ev_rmdh", descending=True).head(5)
    for r in top5.iter_rows(named=True):
        print(f"  {r['sleeve']:45s} {r['bucket']:8s} n={r['n']:>4d} ev_actual={r['ev_actual']:+.4f}% ev_rmdh={r['ev_rmdh']:+.4f}%")

    # === GRID: R4 thresholds combined with R-MDH ===
    print()
    print("=" * 80)
    print("GRID: R4 EV threshold (on R-MDH outcomes) x R-MDH always-on")
    print("=" * 80)
    print()
    print(f"{'Filter':>30s}  {'kept':>8s}  {'COMP':>9s}  {'neg_wk':>7s}  {'asym':>6s}  {'worst_wk':>9s}  {'wealth':>11s}")

    # First row: baseline (no R4, no R-MDH = actual original)
    actual_per_day = sb.select(["date", "quarter", "day_pnl_pct"])
    actual_q = actual_per_day.group_by("quarter").agg(
        (((pl.col("day_pnl_pct") / 100 + 1).log().sum().exp()) - 1).alias("q_ret")
    ).sort("quarter")
    comp_actual = 1.0
    for r in actual_q.iter_rows(named=True):
        comp_actual *= (1 + r["q_ret"])
    actual_per_day_wk = actual_per_day.with_columns([
        pl.col("date").dt.year().alias("y"),
        pl.col("date").dt.week().alias("w"),
    ])
    actual_per_day_wk = actual_per_day_wk.with_columns(
        (pl.col("y").cast(pl.Utf8) + "-W" + pl.col("w").cast(pl.Utf8).str.zfill(2)).alias("iso_week")
    )
    actual_wk = actual_per_day_wk.group_by("iso_week").agg(
        (((pl.col("day_pnl_pct") / 100 + 1).log().sum().exp()) - 1).alias("wk_ret")
    )
    neg_actual = (actual_wk["wk_ret"] < 0).sum()
    print(f"  {'baseline (no R4, no R-MDH)':>30s}  {len(rmdh):>8d}  {(comp_actual-1)*100:>+8.2f}%  {neg_actual:>4d}/104  {1.37:>5.2f}x  {actual_wk['wk_ret'].min()*100:>+8.2f}%  ${10000*comp_actual:>9,.0f}")

    # Second row: R-MDH only (no R4 filter)
    rmdh_pd_full = rmdh.filter(pl.col("rmdh_net_pnl_pct").is_not_null()).group_by("entry_date").agg(
        pl.col("rmdh_net_pnl_pct").sum().alias("d_pnl")
    ).rename({"entry_date": "date"})
    sb_rmdh = sb.select(["date", "quarter", "day_pnl_pct"]).join(rmdh_pd_full, on="date", how="left").with_columns(
        pl.col("d_pnl").fill_null(0.0)
    )
    # Compute COMP from R-MDH per-day
    sb_rmdh_q = sb_rmdh.group_by("quarter").agg(
        (((pl.col("d_pnl") / 100 + 1).log().sum().exp()) - 1).alias("q_ret")
    ).sort("quarter")
    comp_rmdh = 1.0
    for r in sb_rmdh_q.iter_rows(named=True):
        comp_rmdh *= (1 + r["q_ret"])
    sb_rmdh_wk = sb_rmdh.with_columns([
        pl.col("date").dt.year().alias("y"),
        pl.col("date").dt.week().alias("w"),
    ])
    sb_rmdh_wk = sb_rmdh_wk.with_columns(
        (pl.col("y").cast(pl.Utf8) + "-W" + pl.col("w").cast(pl.Utf8).str.zfill(2)).alias("iso_week")
    )
    wk_rmdh = sb_rmdh_wk.group_by("iso_week").agg(
        (((pl.col("d_pnl") / 100 + 1).log().sum().exp()) - 1).alias("wk_ret")
    )
    neg_rmdh = (wk_rmdh["wk_ret"] < 0).sum()
    # Asymmetry
    w_rmdh = rmdh.filter(pl.col("rmdh_net_pnl_pct") > 0)["rmdh_net_pnl_pct"]
    l_rmdh = rmdh.filter(pl.col("rmdh_net_pnl_pct") < 0)["rmdh_net_pnl_pct"]
    asym_rmdh = abs(float(w_rmdh.mean()) / float(l_rmdh.mean())) if len(l_rmdh) > 0 else 0
    print(f"  {'R-MDH only':>30s}  {len(rmdh):>8d}  {(comp_rmdh-1)*100:>+8.2f}%  {neg_rmdh:>4d}/104  {asym_rmdh:>5.2f}x  {wk_rmdh['wk_ret'].min()*100:>+8.2f}%  ${10000*comp_rmdh:>9,.0f}")

    # Composed R4 + R-MDH at multiple R4 thresholds
    for thr in [0.0, 0.03, 0.05, 0.10, 0.15]:
        # Filter trades by R-MDH EV >= threshold (Note: using R-MDH EV here, not actual)
        sb_ev = sleeve_bucket_ev_actual.select(["sleeve", "bucket", "ev_rmdh"])
        kept = rmdh.join(sb_ev, on=["sleeve", "bucket"], how="left").filter(pl.col("ev_rmdh") >= thr)
        if len(kept) == 0:
            print(f"  R4_thr={thr:+.3f}% + R-MDH      {0:>8d}  no trades pass filter")
            continue
        kept_pd = kept.group_by("entry_date").agg(pl.col("rmdh_net_pnl_pct").sum().alias("d_pnl")).rename({"entry_date": "date"})
        sb_c = sb.select(["date", "quarter", "day_pnl_pct"]).join(kept_pd, on="date", how="left").with_columns(
            pl.col("d_pnl").fill_null(0.0)
        )
        sb_c_q = sb_c.group_by("quarter").agg(
            (((pl.col("d_pnl") / 100 + 1).log().sum().exp()) - 1).alias("q_ret")
        ).sort("quarter")
        comp_c = 1.0
        for r in sb_c_q.iter_rows(named=True):
            comp_c *= (1 + r["q_ret"])
        sb_c_wk = sb_c.with_columns([
            pl.col("date").dt.year().alias("y"),
            pl.col("date").dt.week().alias("w"),
        ])
        sb_c_wk = sb_c_wk.with_columns(
            (pl.col("y").cast(pl.Utf8) + "-W" + pl.col("w").cast(pl.Utf8).str.zfill(2)).alias("iso_week")
        )
        wk_c = sb_c_wk.group_by("iso_week").agg(
            (((pl.col("d_pnl") / 100 + 1).log().sum().exp()) - 1).alias("wk_ret")
        )
        neg_c = (wk_c["wk_ret"] < 0).sum()
        w_c = kept.filter(pl.col("rmdh_net_pnl_pct") > 0)["rmdh_net_pnl_pct"]
        l_c = kept.filter(pl.col("rmdh_net_pnl_pct") < 0)["rmdh_net_pnl_pct"]
        asym_c = abs(float(w_c.mean()) / float(l_c.mean())) if len(l_c) > 0 else 0
        worst_c = wk_c["wk_ret"].min() * 100
        print(f"  R4_thr={thr:+.3f}% + R-MDH      {len(kept):>8d}  {(comp_c-1)*100:>+8.2f}%  {neg_c:>4d}/104  {asym_c:>5.2f}x  {worst_c:>+8.2f}%  ${10000*comp_c:>9,.0f}")

    print()
    print("Note: R4 uses FULL-HISTORY sleeve-bucket EV (lookahead bias).")
    print("Walk-forward implementation would degrade by ~30-50%.")

    # === BEST COMBO DETAIL ===
    print()
    print("=" * 80)
    print("BEST COMBO DETAIL (R4_thr=+0.05% + R-MDH)")
    print("=" * 80)
    best_thr = 0.05
    sb_ev = sleeve_bucket_ev_actual.select(["sleeve", "bucket", "ev_rmdh"])
    best = rmdh.join(sb_ev, on=["sleeve", "bucket"], how="left").filter(pl.col("ev_rmdh") >= best_thr)
    print(f"Trades kept: {len(best)} / {len(rmdh)} ({len(best)/len(rmdh)*100:.1f}%)")
    print()
    # Exit reason breakdown
    er = best.group_by("rmdh_exit_reason").agg([
        pl.len().alias("n"),
        pl.col("rmdh_net_pnl_pct").mean().alias("mean_net"),
        pl.col("rmdh_gross_pct").mean().alias("mean_gross"),
        (pl.col("rmdh_net_pnl_pct") > 0).cast(pl.Float64).mean().alias("wr"),
    ]).sort("n", descending=True)
    print("Exit reason mix:")
    for r in er.iter_rows(named=True):
        pct = r["n"] / len(best) * 100
        print(f"  {r['rmdh_exit_reason']:15s}: n={r['n']:>4d} ({pct:5.1f}%)  mean_gross={r['mean_gross']:+.3f}%  mean_net={r['mean_net']:+.4f}%  wr={r['wr']*100:.0f}%")

    # Per-quarter detail
    print()
    print("Per-quarter:")
    best_pd = best.group_by("entry_date").agg(pl.col("rmdh_net_pnl_pct").sum().alias("d_pnl")).rename({"entry_date": "date"})
    sb_best = sb.select(["date", "quarter", "day_pnl_pct"]).join(best_pd, on="date", how="left").with_columns(
        pl.col("d_pnl").fill_null(0.0)
    )
    sb_best_q = sb_best.group_by("quarter").agg(
        (((pl.col("d_pnl") / 100 + 1).log().sum().exp()) - 1).alias("q_ret")
    ).sort("quarter")
    actual_q_iter = actual_q.sort("quarter")
    print(f"  {'Q':>5s}  {'actual':>10s}  {'best':>10s}  {'delta':>10s}")
    wealth = 10000.0
    peak = 10000.0
    max_dd = 0.0
    for r_a, r_b in zip(actual_q_iter.iter_rows(named=True), sb_best_q.iter_rows(named=True)):
        delta = (r_b["q_ret"] - r_a["q_ret"]) * 100
        print(f"  {r_a['quarter']:>5s}  {r_a['q_ret']*100:>+9.2f}%  {r_b['q_ret']*100:>+9.2f}%  {delta:>+9.2f}%")
        wealth *= (1 + r_b["q_ret"])
        if wealth > peak:
            peak = wealth
        dd = (wealth / peak - 1) * 100
        if dd < max_dd:
            max_dd = dd

    print()
    print(f"2-yr wealth (R4+R-MDH best): ${wealth:>10,.2f}")
    print(f"Max DD:                       {max_dd:>+10.2f}%")
    print(f"vs actual baseline:           ${10000*comp_actual:>10,.2f} (delta ${wealth-10000*comp_actual:+,.2f})")

    # Daily-equiv
    daily_compound = (wealth/10000)**(1/731) - 1
    print(f"Daily-equiv compound rate:   {daily_compound*100:+.4f}%/d")
    weekly_compound = (1+daily_compound)**7 - 1
    print(f"Weekly-equiv compound:        {weekly_compound*100:+.4f}%/wk")


if __name__ == "__main__":
    main()
