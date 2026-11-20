"""build_before_oracle_walkforward.py — walk-forward validation of P4-3 BEFORE clusters.

Train k-means on TRAIN (24Q1-25Q1 = 5 quarters); apply centroids to TEST (25Q2-25Q4 = 3 quarters)
without retraining; measure if best-cluster lift HOLDS out-of-sample.

This is the deploy gate — if it holds, the BEFORE oracle becomes a real R1 detector.
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


def main():
    panel = pl.read_parquet(str(OUT_DIR / "oracle_panel_binance_2026_05_18.parquet"))
    panel = panel.with_columns(pl.col("date").cast(pl.Date, strict=False)).sort(["asset", "date"])

    panel = panel.with_columns(
        pl.col("close").pct_change(1).over("asset").alias("ret_1d")
    ).with_columns([
        pl.col("close").shift(-1).over("asset").alias("close_t1"),
        pl.col("close").shift(-4).over("asset").alias("close_t4"),
        pl.col("close").shift(-6).over("asset").alias("close_t6"),
    ]).with_columns([
        ((pl.col("close_t4") / pl.col("close_t1")) - 1).alias("fwd_3d"),
        ((pl.col("close_t6") / pl.col("close_t1")) - 1).alias("fwd_5d"),
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
    )

    # Shift before-features to t-1
    panel = panel.with_columns([
        pl.col("ret_3d_prior").shift(1).over("asset").alias("b_ret_3d"),
        pl.col("ret_7d_prior").shift(1).over("asset").alias("b_ret_7d"),
        pl.col("ret_14d_prior").shift(1).over("asset").alias("b_ret_14d"),
        pl.col("ret_30d_prior").shift(1).over("asset").alias("b_ret_30d"),
        pl.col("rv_7d").shift(1).over("asset").alias("b_rv_7d"),
        pl.col("vol_ratio_7v30").shift(1).over("asset").alias("b_vol_ratio"),
        pl.col("hl_range_7d").shift(1).over("asset").alias("b_hl_range"),
    ])

    # 8Q WF
    panel = panel.filter((pl.col("date") >= date(2024,1,1)) & (pl.col("date") <= date(2025,12,31)))

    feature_cols = ["b_ret_3d","b_ret_7d","b_ret_14d","b_ret_30d","b_rv_7d","b_vol_ratio","b_hl_range"]

    # === SPLIT: TRAIN = 24Q1-25Q1 (5 quarters, ends 2025-03-31), TEST = 25Q2-25Q4 (3 quarters) ===
    train_cutoff = date(2025, 3, 31)
    print(f"[walkfwd] TRAIN: 2024-01-01 -> {train_cutoff}; TEST: {date(2025,4,1)} -> 2025-12-31")

    for trig_pct, trig in [(15, 0.15), (25, 0.25)]:
        events_all = panel.filter(pl.col("ret_1d") >= trig).drop_nulls(subset=feature_cols + ["fwd_3d","fwd_5d"])
        train_ev = events_all.filter(pl.col("date") <= train_cutoff)
        test_ev = events_all.filter(pl.col("date") > train_cutoff)
        print(f"\n=== Trigger +{trig_pct}%: TRAIN n={len(train_ev)}, TEST n={len(test_ev)} ===")
        if len(train_ev) < 50 or len(test_ev) < 30:
            print("  insufficient sample sizes; skipping")
            continue

        # Fit scaler + kmeans on TRAIN only
        X_train = train_ev.select(feature_cols).to_numpy()
        X_train = np.where(np.isfinite(X_train), X_train, np.nan)
        col_medians = np.nanmedian(X_train, axis=0)
        for j in range(X_train.shape[1]):
            mask = np.isnan(X_train[:, j])
            X_train[mask, j] = col_medians[j]
        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        km = KMeans(n_clusters=5, random_state=42, n_init=10).fit(X_train_s)

        # Score on TRAIN
        train_labels = km.labels_
        train_ev_l = train_ev.with_columns(pl.Series("cluster", train_labels.tolist()))
        train_stats = train_ev_l.group_by("cluster").agg([
            pl.len().alias("n"),
            pl.col("fwd_3d").mean().alias("mean_fwd_3d"),
            (pl.col("fwd_3d") > 0).cast(pl.Float64).mean().alias("frac_pos"),
        ]).sort("mean_fwd_3d", descending=True)
        print(f"\n  TRAIN clusters (sorted by mean_fwd_3d):")
        for r in train_stats.iter_rows(named=True):
            print(f"    cluster {r['cluster']}: n={r['n']:>3d}  mean_fwd_3d={r['mean_fwd_3d']*100:>+6.2f}%  frac_pos={r['frac_pos']*100:.1f}%")

        # Apply scaler + kmeans (trained on TRAIN) to TEST
        X_test = test_ev.select(feature_cols).to_numpy()
        X_test = np.where(np.isfinite(X_test), X_test, np.nan)
        for j in range(X_test.shape[1]):
            mask = np.isnan(X_test[:, j])
            X_test[mask, j] = col_medians[j]  # use TRAIN-derived median
        X_test_s = scaler.transform(X_test)
        test_labels = km.predict(X_test_s)
        test_ev_l = test_ev.with_columns(pl.Series("cluster", test_labels.tolist()))
        test_stats = test_ev_l.group_by("cluster").agg([
            pl.len().alias("n"),
            pl.col("fwd_3d").mean().alias("mean_fwd_3d"),
            (pl.col("fwd_3d") > 0).cast(pl.Float64).mean().alias("frac_pos"),
        ]).sort("mean_fwd_3d", descending=True)
        print(f"\n  TEST clusters (using TRAIN centroids, OOS):")
        for r in test_stats.iter_rows(named=True):
            print(f"    cluster {r['cluster']}: n={r['n']:>3d}  mean_fwd_3d={r['mean_fwd_3d']*100:>+6.2f}%  frac_pos={r['frac_pos']*100:.1f}%")

        # Identify TRAIN's best cluster (require n>=20 in train to avoid noise overfits)
        train_stats_robust = train_stats.filter(pl.col("n") >= 20).sort("mean_fwd_3d", descending=True)
        if len(train_stats_robust) == 0:
            print("\n  No TRAIN cluster with n>=20; skipping walk-forward verdict.")
            continue
        best_train_cluster = train_stats_robust.row(0, named=True)
        worst_train_cluster = train_stats_robust.row(len(train_stats_robust)-1, named=True)
        # OOS mean for those clusters
        oos_best = test_ev_l.filter(pl.col("cluster") == best_train_cluster["cluster"])
        oos_worst = test_ev_l.filter(pl.col("cluster") == worst_train_cluster["cluster"])
        print(f"\n  WALK-FORWARD VERDICT (n>=20 robust):")
        print(f"    TRAIN best cluster {best_train_cluster['cluster']} (n={best_train_cluster['n']}): train mean +{best_train_cluster['mean_fwd_3d']*100:.2f}%")
        if len(oos_best) > 0:
            oos_best_mean = float(oos_best['fwd_3d'].mean())
            oos_best_pos = float((oos_best['fwd_3d']>0).cast(pl.Float64).mean())
            print(f"    OOS (TEST) on that cluster: n={len(oos_best):>3d}  mean_fwd_3d={oos_best_mean*100:>+.2f}%  frac_pos={oos_best_pos*100:.1f}%")
        else:
            print(f"    OOS (TEST) on that cluster: 0 events assigned")
            oos_best_mean = 0.0
        print(f"    TRAIN worst (with n>=20) cluster {worst_train_cluster['cluster']} (n={worst_train_cluster['n']}): train mean {worst_train_cluster['mean_fwd_3d']*100:+.2f}%")
        if len(oos_worst) > 0:
            oos_worst_mean = float(oos_worst['fwd_3d'].mean())
            print(f"    OOS on worst: n={len(oos_worst):>3d}  mean_fwd_3d={oos_worst_mean*100:>+.2f}%")
        # Unconditional OOS
        uncond_mean = float(test_ev_l["fwd_3d"].mean())
        print(f"    OOS UNCONDITIONAL: mean_fwd_3d={uncond_mean*100:+.2f}%")
        lift = oos_best_mean - uncond_mean
        print(f"    BEST-cluster OOS LIFT vs unconditional: {lift*100:+.2f}pp")
        if lift > 0.02:
            print(f"    ✓ POSITIVE OOS LIFT >+2pp — cluster generalizes")
        elif lift > 0:
            print(f"    ~ Marginal OOS lift (<2pp) — borderline")
        else:
            print(f"    ✗ NO OOS LIFT — cluster overfits TRAIN")


if __name__ == "__main__":
    main()
