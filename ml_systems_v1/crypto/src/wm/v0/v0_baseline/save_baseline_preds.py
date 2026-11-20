"""
Save Baseline Predictions -- K-Fold OOF for Pipeline Integration

Generates out-of-fold GBT predictions for each asset and saves them
as new columns in chimera parquet files. These predictions become
input features for V1.7 (baseline-seeded temporal model).

K-Fold OOF approach:
  - Split each asset's training data into K folds with purge gaps
  - Train GBT on K-1 folds, predict on held-out fold
  - Concatenate OOF predictions -> full coverage, no leakage
  - Val set (last 10%) predicted by model trained on all training data

New columns added to chimera files:
  - baseline_gbt_h1:  GBT prediction for target_return_1
  - baseline_gbt_h4:  GBT prediction for target_return_4
  - baseline_gbt_h16: GBT prediction for target_return_16
  - baseline_gbt_h64: GBT prediction for target_return_64

Usage:
    python save_baseline_preds.py                  # 22 features, all horizons
    python save_baseline_preds.py --features 13    # 13 features
    python save_baseline_preds.py --horizons 1     # h=1 only (fastest)
    python save_baseline_preds.py --dry-run        # Show shapes, don't write
"""
import numpy as np
import polars as pl
import sys
import argparse
import time
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from settings import (
    DATA_DIR, FEATURE_LIST, FEATURE_LIST_13, FEATURE_LIST_18,
    FEATURE_LIST_30, FEATURE_LIST_34, FEATURE_LIST_37, FEATURE_LIST_41,
    ASSET_TO_IDX, REWARD_HORIZONS, PURGE_GAP_BARS,
    TRAIN_RATIO, VAL_RATIO,
    get_feature_config, list_supported_features,
)
from pipeline.data_integrity import selective_drop_nulls


# ---- Configuration ----
N_FOLDS = 5
GBT_MAX_TRAIN = 100_000  # Subsample per fold for speed
GBT_PARAMS = dict(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_leaf=50,
    random_state=42,
)


def train_gbt_fold(X_train, y_train, X_pred):
    """Train GBT on one fold, return predictions on held-out data."""
    scaler = StandardScaler()

    # Subsample training data if too large
    if len(X_train) > GBT_MAX_TRAIN:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_train), GBT_MAX_TRAIN, replace=False)
        X_sub, y_sub = X_train[idx], y_train[idx]
    else:
        X_sub, y_sub = X_train, y_train

    X_ts = scaler.fit_transform(X_sub)
    X_ps = scaler.transform(X_pred)

    model = GradientBoostingRegressor(**GBT_PARAMS)
    model.fit(X_ts, y_sub)
    return model.predict(X_ps)


def generate_oof_predictions(feats, targets_h, n_folds=N_FOLDS, purge_bars=PURGE_GAP_BARS):
    """
    Generate K-fold out-of-fold predictions with purge gaps.

    Args:
        feats: [N, F] numpy features array
        targets_h: [N] numpy target array for one horizon
        n_folds: number of folds
        purge_bars: gap between train and val folds (prevents leakage)

    Returns:
        oof_preds: [N] array with OOF predictions for every bar
    """
    n = len(feats)
    oof_preds = np.full(n, np.nan, dtype=np.float32)

    # Use TRAIN_RATIO for OOF folds, next VAL_RATIO predicted by full-train model
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    fold_size = train_end // n_folds

    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = min((fold + 1) * fold_size, train_end)

        # Training indices: everything NOT in this fold (with purge gap)
        train_mask = np.ones(train_end, dtype=bool)
        purge_start = max(0, val_start - purge_bars)
        purge_end = min(train_end, val_end + purge_bars)
        train_mask[purge_start:purge_end] = False

        train_idx = np.where(train_mask)[0]
        if len(train_idx) < 1000:
            # Fold too small after purging, skip
            continue

        X_train = feats[train_idx]
        y_train = targets_h[train_idx]
        X_val = feats[val_start:val_end]

        preds = train_gbt_fold(X_train, y_train, X_val)
        oof_preds[val_start:val_end] = preds

    # Predict val split (validation set) using model trained on all training data
    if train_end < val_end:
        X_all_train = feats[:train_end]
        y_all_train = targets_h[:train_end]
        X_holdout = feats[train_end:val_end]
        preds_holdout = train_gbt_fold(X_all_train, y_all_train, X_holdout)
        oof_preds[train_end:val_end] = preds_holdout

    # Fill any remaining NaN gaps (from purge boundaries) with 0.0
    nan_mask = np.isnan(oof_preds)
    n_nan = nan_mask.sum()
    if n_nan > 0:
        oof_preds[nan_mask] = 0.0

    return oof_preds


def save_baseline_predictions(n_features=41, horizons=None, dry_run=False):
    """
    Generate OOF GBT predictions and save to chimera files.

    Args:
        n_features: any from feature_sets.SUPPORTED_COUNTS
        horizons: list of horizons to predict (default: all [1, 4, 16, 64])
        dry_run: if True, compute but don't write to disk
    """
    feature_list, _input_dim, _bd = get_feature_config(n_features)
    if horizons is None:
        horizons = REWARD_HORIZONS

    print("=" * 70)
    print(f"  BASELINE PREDICTION GENERATOR (K={N_FOLDS} Fold OOF)")
    print(f"  Features:  {n_features}")
    print(f"  Horizons:  {horizons}")
    print(f"  Purge gap: {PURGE_GAP_BARS} bars")
    print(f"  GBT:       {GBT_PARAMS['n_estimators']} trees, "
          f"depth={GBT_PARAMS['max_depth']}, lr={GBT_PARAMS['learning_rate']}")
    print(f"  Dry run:   {dry_run}")
    print("=" * 70)

    files = sorted(DATA_DIR.glob("*_v51_chimera*.parquet"))
    if not files:
        print("  [ERROR] No chimera files found.")
        return

    total_t0 = time.time()

    for f in files:
        asset_name = f.stem.split("_")[0].upper()
        if asset_name not in ASSET_TO_IDX:
            continue

        print(f"\n  {asset_name}:")
        df = pl.read_parquet(f)
        n_rows_original = len(df)

        # Extract features and targets (using raw return targets)
        df_clean = selective_drop_nulls(df, feature_list, horizons, asset_name)

        # Get feature columns as numpy
        feat_cols = [c for c in feature_list if c in df_clean.columns]
        if len(feat_cols) != len(feature_list):
            missing = set(feature_list) - set(feat_cols)
            print(f"    [WARN] Missing features: {missing}, skipping")
            continue

        feats = df_clean.select(feat_cols).to_numpy().astype(np.float32)

        # Use raw return targets (voladj deprecated)
        target_prefix = "target_return"

        print(f"    Bars: {len(feats):,}, Features: {feats.shape[1]}, "
              f"Target: {target_prefix}")

        # Generate OOF predictions for each horizon
        new_cols = {}
        for h in horizons:
            target_col = f"{target_prefix}_{h}"
            if target_col not in df_clean.columns:
                print(f"    [WARN] Missing {target_col}, skipping h={h}")
                continue

            targets_h = df_clean[target_col].to_numpy().astype(np.float32)

            t0 = time.time()
            oof_preds = generate_oof_predictions(feats, targets_h)
            elapsed = time.time() - t0

            # Quality check: IC of OOF predictions
            mask = np.isfinite(oof_preds) & np.isfinite(targets_h)
            if mask.sum() > 100:
                ic = float(np.corrcoef(oof_preds[mask], targets_h[mask])[0, 1])
            else:
                ic = 0.0

            col_name = f"baseline_gbt_h{h}"
            new_cols[col_name] = oof_preds
            print(f"    h={h}: OOF IC={ic:+.4f}, "
                  f"mean={np.mean(oof_preds):.6f}, "
                  f"std={np.std(oof_preds):.6f} ({elapsed:.1f}s)")

        if not new_cols:
            print(f"    [WARN] No predictions generated, skipping")
            continue

        if dry_run:
            print(f"    [DRY RUN] Would add {len(new_cols)} columns to {f.name}")
            continue

        # Add predictions to the CLEAN dataframe, then join back to original
        # Strategy: add columns to df_clean, then write a new chimera with
        # the baseline columns. Since df_clean may have fewer rows than df
        # (null drops), we need to align by index.
        #
        # Simpler approach: add columns to the ORIGINAL df using row alignment.
        # df_clean is a subset of df (rows with no nulls in model columns).
        # We need to map predictions back to original row positions.

        # Get the indices of clean rows in the original df
        # Since selective_drop_nulls doesn't preserve index, we need to
        # recompute which rows survived the null drop.
        drop_subset = list(feature_list) + [f"{target_prefix}_{h}" for h in horizons]
        drop_subset = [c for c in drop_subset if c in df.columns]

        # Create a mask of non-null rows in the original df
        null_expr = [pl.col(c).is_not_null() for c in drop_subset]
        combined_mask = null_expr[0]
        for expr in null_expr[1:]:
            combined_mask = combined_mask & expr
        valid_mask = df.select(combined_mask.alias("valid"))["valid"].to_numpy()

        # Create full-length columns (NaN for null-dropped rows)
        for col_name, preds in new_cols.items():
            full_col = np.full(n_rows_original, np.nan, dtype=np.float32)
            full_col[valid_mask] = preds

            # Drop existing column if present (re-run safety)
            if col_name in df.columns:
                df = df.drop(col_name)
            df = df.with_columns(pl.Series(name=col_name, values=full_col))

        # Write back to chimera file (atomic: write temp, then rename)
        tmp_path = f.with_suffix(".parquet.tmp")
        df.write_parquet(tmp_path)

        # Verify
        verify_df = pl.read_parquet(tmp_path, n_rows=1)
        for col_name in new_cols:
            if col_name not in verify_df.columns:
                print(f"    [ERROR] Column {col_name} not in written file!")
                tmp_path.unlink()
                break
        else:
            # All good - atomic rename
            if f.exists():
                f.unlink()
            tmp_path.rename(f)
            print(f"    [OK] Saved {len(new_cols)} baseline columns to {f.name} "
                  f"({len(df.columns)} total columns)")

    total_elapsed = time.time() - total_t0
    print(f"\n  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    if not dry_run:
        print("  Chimera files updated with baseline predictions.")
        print("  V1.7 can now use baseline_gbt_h* as input features.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save GBT Baseline Predictions to Chimera Files")
    parser.add_argument("--features", type=int,
                        choices=list_supported_features(),
                        default=41,
                        help="Feature count from src/feature_sets.py registry "
                             "(13/18/21/25/29/30/34/37/41 = v50; 46-121 = v51 frontier).")
    parser.add_argument("--horizons", type=int, nargs="+", default=None,
                        help="Horizons to predict (default: 1 4 16 64)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute predictions but don't write to chimera files")
    args = parser.parse_args()

    save_baseline_predictions(
        n_features=args.features,
        horizons=args.horizons,
        dry_run=args.dry_run,
    )
