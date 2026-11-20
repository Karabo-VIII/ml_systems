"""build_before_oracle_multifold_cv.py — FLAG #4 fix: multi-fold CV on BEFORE oracle.

Prior walk-forward was single-split (5Q TRAIN, 3Q TEST). RED-team flagged this
as insufficient robustness check. This script does 4-fold rolling-window CV:

  Fold 1: TRAIN 24Q1-24Q4 (4 quarters), TEST 25Q1
  Fold 2: TRAIN 24Q2-25Q1, TEST 25Q2
  Fold 3: TRAIN 24Q3-25Q2, TEST 25Q3
  Fold 4: TRAIN 24Q4-25Q3, TEST 25Q4

Tests whether the cluster best-vs-unconditional lift HOLDS across all 4 folds.

Verdict criteria:
  ✓ Cluster signal generalizes: best-cluster OOS lift > +1pp in ≥3/4 folds AND
    mean OOS lift across folds > +2pp
  ⚠️ Mixed: best-cluster OOS lift > +1pp in 2/4 folds
  ✗ No generalization: best-cluster OOS lift < +1pp in ≥3/4 folds
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


def quarter_range(q_str: str):
    """Return (start_date, end_date) for a quarter label like '24Q1'."""
    yy = int("20" + q_str[:2])
    q = int(q_str[3])
    starts = {1: (1, 1), 2: (4, 1), 3: (7, 1), 4: (10, 1)}
    ends = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}
    return date(yy, *starts[q]), date(yy, *ends[q])


def main():
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

    # 4-fold CV
    folds = [
        ("Fold 1", ["24Q1","24Q2","24Q3","24Q4"], "25Q1"),
        ("Fold 2", ["24Q2","24Q3","24Q4","25Q1"], "25Q2"),
        ("Fold 3", ["24Q3","24Q4","25Q1","25Q2"], "25Q3"),
        ("Fold 4", ["24Q4","25Q1","25Q2","25Q3"], "25Q4"),
    ]

    print("="*80)
    print("BEFORE-ORACLE MULTI-FOLD CV (4 rolling folds, +15% trigger only)")
    print("="*80)

    trig = 0.15
    fold_results = []

    for fold_label, train_qs, test_q in folds:
        train_start, _ = quarter_range(train_qs[0])
        _, train_end = quarter_range(train_qs[-1])
        test_start, test_end = quarter_range(test_q)

        events_all = panel.filter(pl.col("ret_1d") >= trig).drop_nulls(subset=feature_cols + ["fwd_3d"])
        train_ev = events_all.filter((pl.col("date") >= train_start) & (pl.col("date") <= train_end))
        test_ev = events_all.filter((pl.col("date") >= test_start) & (pl.col("date") <= test_end))

        if len(train_ev) < 50 or len(test_ev) < 20:
            print(f"\n{fold_label}: insufficient sample (TRAIN={len(train_ev)}, TEST={len(test_ev)}) — skipping")
            continue

        X_train = train_ev.select(feature_cols).to_numpy()
        X_train = np.where(np.isfinite(X_train), X_train, np.nan)
        col_med = np.nanmedian(X_train, axis=0)
        for j in range(X_train.shape[1]):
            X_train[np.isnan(X_train[:, j]), j] = col_med[j]
        scaler = StandardScaler().fit(X_train)
        km = KMeans(n_clusters=5, random_state=42, n_init=10).fit(scaler.transform(X_train))

        # Train cluster stats
        train_labels = km.labels_
        train_ev_l = train_ev.with_columns(pl.Series("cluster", train_labels.tolist()))
        train_stats = train_ev_l.group_by("cluster").agg([
            pl.len().alias("n"),
            pl.col("fwd_3d").mean().alias("mean_fwd_3d"),
        ]).filter(pl.col("n") >= 20).sort("mean_fwd_3d", descending=True)
        if len(train_stats) == 0:
            print(f"\n{fold_label}: no robust train clusters — skip")
            continue
        train_best = train_stats.row(0, named=True)

        # OOS test
        X_test = test_ev.select(feature_cols).to_numpy()
        X_test = np.where(np.isfinite(X_test), X_test, np.nan)
        for j in range(X_test.shape[1]):
            X_test[np.isnan(X_test[:, j]), j] = col_med[j]
        test_labels = km.predict(scaler.transform(X_test))
        test_ev_l = test_ev.with_columns(pl.Series("cluster", test_labels.tolist()))
        oos_best = test_ev_l.filter(pl.col("cluster") == train_best["cluster"])
        oos_uncond_mean = float(test_ev_l["fwd_3d"].mean())
        oos_best_mean = float(oos_best["fwd_3d"].mean()) if len(oos_best) > 0 else float("nan")
        lift_pp = (oos_best_mean - oos_uncond_mean) * 100 if not np.isnan(oos_best_mean) else float("nan")

        print(f"\n{fold_label}: TRAIN={'+'.join(train_qs)} TEST={test_q}")
        print(f"  TRAIN best cluster {train_best['cluster']} (n={train_best['n']}): mean_fwd_3d={train_best['mean_fwd_3d']*100:+.2f}%")
        print(f"  OOS n={len(oos_best)}  mean={oos_best_mean*100:+.2f}%  uncond={oos_uncond_mean*100:+.2f}%  LIFT={lift_pp:+.2f}pp")
        fold_results.append({"fold": fold_label, "train_n": int(train_best['n']), "oos_n": len(oos_best), "oos_lift_pp": lift_pp, "oos_best": oos_best_mean, "oos_uncond": oos_uncond_mean})

    # Verdict
    print("\n" + "="*80)
    print("MULTI-FOLD CV VERDICT")
    print("="*80)
    lifts = [r["oos_lift_pp"] for r in fold_results if not np.isnan(r["oos_lift_pp"])]
    print(f"  Folds completed: {len(fold_results)}")
    print(f"  Per-fold OOS lifts: {[f'{l:+.2f}pp' for l in lifts]}")
    if lifts:
        pos_folds = sum(1 for l in lifts if l > 1.0)
        mean_lift = np.mean(lifts)
        median_lift = np.median(lifts)
        print(f"  Folds with OOS lift > +1pp: {pos_folds}/{len(lifts)}")
        print(f"  Mean OOS lift: {mean_lift:+.2f}pp")
        print(f"  Median OOS lift: {median_lift:+.2f}pp")
        print()
        if pos_folds >= 3 and mean_lift > 2:
            print(f"  ✓ ROBUST: signal generalizes across folds")
        elif pos_folds >= 2:
            print(f"  ⚠️ MIXED: signal generalizes in some folds; needs richer features or per-regime")
        else:
            print(f"  ✗ NOT ROBUST: signal does not generalize multi-fold")


if __name__ == "__main__":
    main()
