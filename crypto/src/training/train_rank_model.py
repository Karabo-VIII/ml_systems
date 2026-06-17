"""
Phase C Cross-Sectional Rank Model Training
=============================================

Trains a LightGBM LambdaRank model on daily panel data to predict the
CROSS-SECTIONAL RANK of forward returns.

This is different from per-asset return prediction: instead of "predict BTC's
return", we ask "among today's 10-24 assets, which ones will outperform and
which will underperform?" This is the objective that DIRECTLY drives cycling
alpha — we don't need to know absolute returns, we need to know the ORDER.

LambdaRank (Burges 2010) maximizes NDCG directly. Cryptos per day = one query.
The model learns feature patterns that predict relative (not absolute) performance.

Data construction:
    For each day t from first-valid-day to last:
        for each asset a in universe:
            sample = (features[a, t], forward_return_rank[a, t+1])
        group_size = n_active_assets_that_day

Features per asset per day: aggregate daily values of the 34 chimera features.
We compute: mean, last value over the trailing day's bars per feature.

Target: dense rank of next-day's return, normalized to [0, 1].

Training:
    - Split 70/15/15 time-wise train/val/test
    - LightGBM LambdaRank with NDCG@10 eval
    - Early stopping on val NDCG@10
    - Save to models/rank_v1/ranker.txt

Run:
    python src/training/train_rank_model.py
    python src/training/train_rank_model.py --universe 24
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "chimera_legacy"
MODEL_DIR = PROJECT_ROOT / "models" / "rank_v1"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT / "src" / "strategy"))
from universe import UNIVERSE_10, UNIVERSE_24


# Feature columns from chimera (the 34 base features)
FEATURE_NAMES = [
    "norm_deviation", "norm_fd_close", "norm_vpin", "norm_flow_imbalance",
    "norm_vol_cluster", "norm_funding", "norm_tick_count", "norm_log_volume",
    "norm_hl_spread", "hurst_regime", "norm_oi_change", "norm_return_1",
    "norm_spread_bps", "norm_ma_distance", "norm_whale", "norm_efficiency",
    "norm_return_4", "norm_return_16", "norm_return_kurtosis",
    "norm_bar_duration", "norm_funding_momentum",
    "norm_hawkes_intensity", "norm_hawkes_buy_intensity",
    "norm_hawkes_sell_intensity", "norm_hawkes_imbalance",
    "norm_momentum_accel", "norm_vol_price_corr", "norm_vol_ratio",
    "norm_flow_persistence", "norm_oi_price_divergence",
    "norm_yz_volatility", "norm_cs_spread", "norm_perm_entropy",
    "norm_kyle_lambda",
]


def load_asset_daily_panel(asset: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load per-day (features[34], forward_return, day_id) for an asset.
    Vectorized via polars groupby — 100x faster than per-day Python loops.
    """
    path = DATA_DIR / f"{asset.lower()}_v50_chimera.parquet"
    if not path.exists():
        return None, None, None

    head_cols = pl.read_parquet(path, n_rows=1).columns
    available_feats = [c for c in FEATURE_NAMES if c in head_cols]
    cols = ["timestamp", "close"] + available_feats
    df = pl.read_parquet(path, columns=cols)
    if df.height < 500:
        return None, None, None

    # Compute day_id column
    df = df.with_columns((pl.col("timestamp") // 86400000).cast(pl.Int64).alias("day_id"))
    # Group by day: last close, mean of each feature
    agg_exprs = [pl.col("close").last().alias("close_eod")]
    for f in available_feats:
        agg_exprs.append(pl.col(f).mean().alias(f))
    daily = df.group_by("day_id").agg(agg_exprs).sort("day_id")

    day_id_arr = daily["day_id"].to_numpy()
    daily_close = daily["close_eod"].to_numpy()
    n_days = len(daily_close)
    if n_days < 100:
        return None, None, None

    daily_feats = np.zeros((n_days, len(FEATURE_NAMES)))
    for j, f in enumerate(FEATURE_NAMES):
        if f in available_feats:
            arr = daily[f].to_numpy()
            arr = np.where(np.isfinite(arr), arr, 0.0)
            daily_feats[:, j] = arr

    fwd_ret = np.full(n_days, np.nan)
    fwd_ret[:-1] = np.diff(daily_close) / daily_close[:-1]
    return daily_feats, fwd_ret, day_id_arr


def build_panel_dataset(universe: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the daily panel. Returns (X, y, group_sizes, day_indices).

    X: [n_samples, F] features
    y: [n_samples] rank labels (0-K integers for LambdaRank)
    group_sizes: [n_days] count of samples per day
    day_indices: [n_samples] day idx for each sample (for time split)
    """
    per_asset: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for a in universe:
        feats, fwd, days = load_asset_daily_panel(a)
        if feats is None:
            continue
        per_asset[a] = (feats, fwd, days)
    print(f"Loaded panel for {len(per_asset)} assets")

    # Union day axis
    all_days = sorted(set().union(*[set(d) for _, _, d in per_asset.values()]))
    day_to_idx = {d: i for i, d in enumerate(all_days)}
    # For each asset, map days to matrix indices
    X_list, y_list, group_list, day_list = [], [], [], []
    for day in all_days:
        # Collect samples for this day across assets
        day_samples = []
        for a, (feats, fwd, days) in per_asset.items():
            if day in days:
                i = int(np.where(days == day)[0][0])
                if i >= len(fwd) - 1 or not np.isfinite(fwd[i]):
                    continue
                f = feats[i]
                if not np.all(np.isfinite(f)):
                    continue
                day_samples.append((a, f, fwd[i]))
        if len(day_samples) < 3:
            continue
        # Rank within-day by forward return, integer labels (higher = outperform)
        fwds = np.array([s[2] for s in day_samples])
        ranks = fwds.argsort().argsort()  # dense rank 0..K-1
        for (a, f, _), rank in zip(day_samples, ranks):
            X_list.append(f)
            y_list.append(int(rank))
            day_list.append(day_to_idx[day])
        group_list.append(len(day_samples))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    groups = np.array(group_list, dtype=np.int32)
    day_ids = np.array(day_list, dtype=np.int64)
    return X, y, groups, day_ids


def time_split(X, y, groups, day_ids, train_frac=0.70, val_frac=0.15):
    """Time-based split preserving group boundaries."""
    unique_days = np.unique(day_ids)
    n = len(unique_days)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train_days = set(unique_days[:train_end])
    val_days = set(unique_days[train_end:val_end])
    test_days = set(unique_days[val_end:])

    def select(days_set):
        mask = np.isin(day_ids, list(days_set))
        return X[mask], y[mask], day_ids[mask]

    def group_sizes_for_days(days_set):
        # Recompute groups aligned to filtered samples
        sel_days = [d for d in unique_days if d in days_set]
        out = []
        for d in sel_days:
            out.append(int((day_ids == d).sum()))
        return np.array(out, dtype=np.int32)

    X_tr, y_tr, d_tr = select(train_days)
    X_v, y_v, d_v = select(val_days)
    X_te, y_te, d_te = select(test_days)
    g_tr = group_sizes_for_days(train_days)
    g_v = group_sizes_for_days(val_days)
    g_te = group_sizes_for_days(test_days)
    return (X_tr, y_tr, g_tr, d_tr), (X_v, y_v, g_v, d_v), (X_te, y_te, g_te, d_te)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", type=int, default=10, choices=[10, 24])
    parser.add_argument("--n-trees", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    args = parser.parse_args()

    import lightgbm as lgb

    universe = UNIVERSE_24 if args.universe == 24 else UNIVERSE_10
    print(f"Building panel for {len(universe)} assets...")
    t0 = time.time()
    X, y, groups, day_ids = build_panel_dataset(universe)
    print(f"Panel: {X.shape[0]} samples, {X.shape[1]} features, "
          f"{len(groups)} days, {time.time() - t0:.1f}s")

    (X_tr, y_tr, g_tr, _), (X_v, y_v, g_v, _), (X_te, y_te, g_te, _) = \
        time_split(X, y, groups, day_ids, train_frac=0.70, val_frac=0.15)
    print(f"Splits: train {X_tr.shape}, val {X_v.shape}, test {X_te.shape}")
    print(f"Groups: train {len(g_tr)}, val {len(g_v)}, test {len(g_te)}")

    # LambdaRank training
    train_data = lgb.Dataset(X_tr, label=y_tr, group=g_tr,
                              feature_name=FEATURE_NAMES)
    val_data = lgb.Dataset(X_v, label=y_v, group=g_v, reference=train_data,
                            feature_name=FEATURE_NAMES)
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3, 5, 10],
        "learning_rate": args.lr,
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "max_position": 10,
    }
    print(f"\nTraining LambdaRank ({args.n_trees} trees max)...")
    t1 = time.time()
    model = lgb.train(
        params, train_data, num_boost_round=args.n_trees,
        valid_sets=[train_data, val_data], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(stopping_rounds=30),
                   lgb.log_evaluation(period=50)],
    )
    print(f"Training done in {time.time() - t1:.1f}s, best iter {model.best_iteration}")

    # Evaluate on test
    print("\nEvaluating on test set...")
    preds_te = model.predict(X_te)
    # Compute NDCG@1 manually: fraction of days where top predicted = top actual
    top1_hit = 0
    n_eval = 0
    idx = 0
    for g in g_te:
        if g < 2:
            idx += g
            continue
        day_preds = preds_te[idx:idx + g]
        day_ys = y_te[idx:idx + g]
        top_pred = int(np.argmax(day_preds))
        top_actual = int(np.argmax(day_ys))
        if top_pred == top_actual:
            top1_hit += 1
        n_eval += 1
        idx += g
    print(f"TEST Top-1 hit rate: {top1_hit}/{n_eval} = {100*top1_hit/max(n_eval,1):.1f}%")
    print(f"Random baseline on avg group {np.mean(g_te):.1f}: "
          f"{100/np.mean(g_te):.1f}%")

    # Feature importance
    imp = model.feature_importance(importance_type="gain")
    imp_sorted = sorted(zip(FEATURE_NAMES, imp), key=lambda x: -x[1])
    print("\nTop 15 features by gain:")
    for f, i in imp_sorted[:15]:
        print(f"  {f:<30} {i:>10.0f}")

    # Save
    out = MODEL_DIR / f"ranker_u{args.universe}.txt"
    model.save_model(str(out), num_iteration=model.best_iteration)
    # Save feature names alongside
    (MODEL_DIR / f"features_u{args.universe}.txt").write_text("\n".join(FEATURE_NAMES))
    print(f"\nModel saved: {out}")


if __name__ == "__main__":
    main()
