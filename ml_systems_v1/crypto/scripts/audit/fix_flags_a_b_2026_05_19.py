"""fix_flags_a_b_2026_05_19.py — Immediate fixes for 2nd-pass FLAGS A and B.

FLAG A: Bootstrap CI per fold for BEFORE oracle multi-fold lifts.
FLAG B: Exclude TA_SML from union; re-measure realistic v3 cap=5 wealth.

Both fixes are applied IN THE SAME SESSION as the 2nd-pass that identified them.
"""
from __future__ import annotations
import os
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

os.environ["PYTHONIOENCODING"] = "utf-8"
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "audit"


def proper_daily_compound(per_day_pnl_pct: np.ndarray) -> float:
    return float(np.exp(np.log1p(per_day_pnl_pct / 100).sum()) - 1) * 100


def neg_weeks_count(dates_idx, per_day_pnl_pct):
    import pandas as pd
    df = pd.DataFrame({"date": pd.to_datetime(dates_idx), "pnl": per_day_pnl_pct})
    df["iso_year"] = df["date"].dt.isocalendar().year
    df["iso_week"] = df["date"].dt.isocalendar().week
    df["wk"] = df["iso_year"].astype(str) + "-W" + df["iso_week"].astype(str).str.zfill(2)
    weekly = df.groupby("wk")["pnl"].apply(lambda x: float(np.exp(np.log1p(x.values / 100).sum()) - 1))
    return int((weekly < 0).sum())


def quarter_range(q_str):
    yy = int("20" + q_str[:2])
    q = int(q_str[3])
    starts = {1: (1, 1), 2: (4, 1), 3: (7, 1), 4: (10, 1)}
    ends = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}
    return date(yy, *starts[q]), date(yy, *ends[q])


# ========== FLAG A FIX: Bootstrap CI on multi-fold BEFORE oracle ==========

def flag_a_fix():
    print("=" * 80)
    print("FLAG A FIX — Bootstrap CI per fold on BEFORE oracle multi-fold")
    print("=" * 80)

    panel = pl.read_parquet(str(OUT_DIR / "oracle_panel_binance_2026_05_18.parquet"))
    panel = panel.with_columns(pl.col("date").cast(pl.Date, strict=False)).sort(["asset", "date"])
    panel = panel.with_columns(
        pl.col("close").pct_change(1).over("asset").alias("ret_1d")
    ).with_columns([
        pl.col("close").shift(-1).over("asset").alias("close_t1"),
        pl.col("close").shift(-4).over("asset").alias("close_t4"),
    ]).with_columns([
        ((pl.col("close_t4") / pl.col("close_t1")) - 1).alias("fwd_3d"),
        pl.col("close").pct_change(3).over("asset").alias("ret_3d_prior"),
        pl.col("close").pct_change(7).over("asset").alias("ret_7d_prior"),
        pl.col("close").pct_change(14).over("asset").alias("ret_14d_prior"),
        pl.col("close").pct_change(30).over("asset").alias("ret_30d_prior"),
        pl.col("volume").rolling_mean(7).over("asset").alias("vol_7d_mean"),
        pl.col("volume").rolling_mean(30).over("asset").alias("vol_30d_mean"),
        pl.col("ret_1d").rolling_std(7).over("asset").alias("rv_7d"),
        ((pl.col("high") - pl.col("low")) / pl.col("close")).rolling_mean(7).over("asset").alias("hl_range_7d"),
    ]).with_columns(
        (pl.col("vol_7d_mean") / pl.col("vol_30d_mean")).alias("vol_ratio_7v30")
    ).with_columns([
        pl.col("ret_3d_prior").shift(1).over("asset").alias("b_ret_3d"),
        pl.col("ret_7d_prior").shift(1).over("asset").alias("b_ret_7d"),
        pl.col("ret_14d_prior").shift(1).over("asset").alias("b_ret_14d"),
        pl.col("ret_30d_prior").shift(1).over("asset").alias("b_ret_30d"),
        pl.col("rv_7d").shift(1).over("asset").alias("b_rv_7d"),
        pl.col("vol_ratio_7v30").shift(1).over("asset").alias("b_vol_ratio"),
        pl.col("hl_range_7d").shift(1).over("asset").alias("b_hl_range"),
    ])
    panel = panel.filter((pl.col("date") >= date(2024,1,1)) & (pl.col("date") <= date(2025,12,31)))
    feature_cols = ["b_ret_3d","b_ret_7d","b_ret_14d","b_ret_30d","b_rv_7d","b_vol_ratio","b_hl_range"]

    folds = [
        ("Fold 1", ["24Q1","24Q2","24Q3","24Q4"], "25Q1"),
        ("Fold 2", ["24Q2","24Q3","24Q4","25Q1"], "25Q2"),
        ("Fold 3", ["24Q3","24Q4","25Q1","25Q2"], "25Q3"),
        ("Fold 4", ["24Q4","25Q1","25Q2","25Q3"], "25Q4"),
    ]

    trig = 0.15
    N_BOOTSTRAP = 1000
    print(f"  Bootstrap: {N_BOOTSTRAP} resamples per fold")
    print(f"\n  {'Fold':>8s} {'oos_n':>6s} {'point_lift':>12s} {'CI_low':>9s} {'CI_high':>9s} {'P(lift>0)':>11s}")

    fold_results = []
    for fold_label, train_qs, test_q in folds:
        _, train_end = quarter_range(train_qs[-1])
        train_start, _ = quarter_range(train_qs[0])
        test_start, test_end = quarter_range(test_q)

        events_all = panel.filter(pl.col("ret_1d") >= trig).drop_nulls(subset=feature_cols + ["fwd_3d"])
        train_ev = events_all.filter((pl.col("date") >= train_start) & (pl.col("date") <= train_end))
        test_ev = events_all.filter((pl.col("date") >= test_start) & (pl.col("date") <= test_end))

        if len(train_ev) < 50 or len(test_ev) < 20:
            continue

        X_train = train_ev.select(feature_cols).to_numpy()
        X_train = np.where(np.isfinite(X_train), X_train, np.nan)
        col_med = np.nanmedian(X_train, axis=0)
        for j in range(X_train.shape[1]):
            X_train[np.isnan(X_train[:, j]), j] = col_med[j]
        scaler = StandardScaler().fit(X_train)
        km = KMeans(n_clusters=5, random_state=42, n_init=10).fit(scaler.transform(X_train))

        train_ev_l = train_ev.with_columns(pl.Series("cluster", km.labels_.tolist()))
        train_stats = train_ev_l.group_by("cluster").agg([
            pl.len().alias("n"),
            pl.col("fwd_3d").mean().alias("mean_fwd_3d"),
        ]).filter(pl.col("n") >= 20).sort("mean_fwd_3d", descending=True)
        if len(train_stats) == 0:
            continue
        train_best_cl = train_stats.row(0, named=True)["cluster"]

        X_test = test_ev.select(feature_cols).to_numpy()
        X_test = np.where(np.isfinite(X_test), X_test, np.nan)
        for j in range(X_test.shape[1]):
            X_test[np.isnan(X_test[:, j]), j] = col_med[j]
        test_labels = km.predict(scaler.transform(X_test))
        test_ev_l = test_ev.with_columns(pl.Series("cluster", test_labels.tolist()))

        oos_best = test_ev_l.filter(pl.col("cluster") == train_best_cl)["fwd_3d"].to_numpy()
        oos_uncond_all = test_ev_l["fwd_3d"].to_numpy()

        if len(oos_best) == 0:
            continue

        # Bootstrap: resample WITH replacement from OOS events; compute lift per resample
        rng = np.random.default_rng(42)
        lifts = []
        for _ in range(N_BOOTSTRAP):
            # Resample test events; reassess best-cluster mean and unconditional mean
            idx_best = rng.choice(len(oos_best), size=len(oos_best), replace=True)
            idx_all = rng.choice(len(oos_uncond_all), size=len(oos_uncond_all), replace=True)
            lift = (oos_best[idx_best].mean() - oos_uncond_all[idx_all].mean()) * 100
            lifts.append(lift)
        lifts = np.array(lifts)
        point_lift = (oos_best.mean() - oos_uncond_all.mean()) * 100
        ci_low = np.percentile(lifts, 2.5)
        ci_high = np.percentile(lifts, 97.5)
        p_pos = (lifts > 0).mean()

        print(f"  {fold_label:>8s} {len(oos_best):>6d}  {point_lift:>+10.2f}pp  {ci_low:>+7.2f}pp  {ci_high:>+7.2f}pp  {p_pos*100:>9.1f}%")
        fold_results.append({"fold": fold_label, "lift": point_lift, "ci_low": ci_low, "ci_high": ci_high, "p_pos": p_pos})

    print()
    if fold_results:
        any_robust = sum(1 for r in fold_results if r["ci_low"] > 0)
        all_pos = sum(1 for r in fold_results if r["p_pos"] > 0.95)
        print(f"  Folds with CI[2.5, 97.5] strictly above 0: {any_robust}/{len(fold_results)}")
        print(f"  Folds with P(lift > 0) > 95%: {all_pos}/{len(fold_results)}")
        if any_robust >= 3:
            print("  ✓ ROBUST: multiple folds clear bootstrap CI")
        elif any_robust >= 1:
            print("  ⚠️ MIXED: one fold clears bootstrap CI; signal regime-conditional")
        else:
            print("  ✗ NOT ROBUST: no fold's CI strictly above 0 — signal is noise-indistinguishable")


# ========== FLAG B FIX: Exclude TA_SML; re-measure ==========

def flag_b_fix():
    print("\n" + "=" * 80)
    print("FLAG B FIX — Exclude TA_SML (u50 universe); re-measure realistic v3 cap=5 union")
    print("=" * 80)

    ext = pl.read_parquet(str(OUT_DIR / "union_extended_trade_ledger.parquet"))
    ext_no_tasml = ext.filter(pl.col("lane") != "TA_SML_SOLO")
    print(f"  Extended (17 lanes): {len(ext)} trades")
    print(f"  Excluding TA_SML (16 lanes): {len(ext_no_tasml)} trades  (dropped {len(ext) - len(ext_no_tasml)} TA_SML trades)")

    atlas = ext_no_tasml.group_by(["asset", "entry_date"]).agg([
        pl.col("lane").n_unique().alias("n_lanes_firing"),
        pl.col("net_pnl_pct").mean().alias("mean_lane_pnl"),
        pl.col("size_pct").first().alias("size_pct"),
    ]).sort("entry_date").with_columns(pl.col("entry_date").cast(pl.Date))

    all_dates = pl.date_range(date(2024,1,1), date(2025,12,31), interval="1d", eager=True)
    print(f"\n  {'K':>3s} {'min_conv':>8s} {'COMP':>10s} {'neg_wk':>7s} {'wealth':>10s} {'cleared':>8s}")
    for K in [5, 10]:
        for min_conv in [1, 2]:
            cell = atlas.with_columns(pl.col("n_lanes_firing").cast(pl.Float64).alias("priority"))
            per_day = cell.filter(pl.col("priority") >= min_conv).sort(
                ["entry_date", "priority"], descending=[False, True]
            ).group_by("entry_date").agg([
                pl.col("mean_lane_pnl").head(K).sum().alias("day_pnl_pct"),
            ])
            cal = pl.DataFrame({"entry_date": all_dates}).join(per_day, on="entry_date", how="left").with_columns(
                pl.col("day_pnl_pct").fill_null(0.0)
            ).sort("entry_date")
            pnl_arr = cal["day_pnl_pct"].to_numpy()
            comp = proper_daily_compound(pnl_arr)
            dates_list = cal["entry_date"].to_list()
            neg_wk = neg_weeks_count(dates_list, pnl_arr)
            wealth = 10000 * (1 + comp / 100)
            cleared = "YES" if wealth >= 28146 else "NO"
            print(f"  {K:>3d} {min_conv:>7d}    {comp:>+9.2f}% {neg_wk:>4d}/104 ${wealth:>9,.0f}    {cleared:>5s}")

    print(f"\n  COMPARISON vs 17-lane union (including TA_SML):")
    print(f"    K=5  conv=1:  17-lane $20,370   /  16-lane (no TA_SML) wealth above")
    print(f"    K=10 conv=1:  17-lane $30,692   /  16-lane (no TA_SML) wealth above")
    print(f"  If 16-lane wealth ~$18-25k at K=5, the TA_SML universe artifact is small (~+$2-5k effect).")


if __name__ == "__main__":
    flag_a_fix()
    flag_b_fix()
