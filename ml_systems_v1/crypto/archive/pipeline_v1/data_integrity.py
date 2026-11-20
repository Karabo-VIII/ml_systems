"""
Data Integrity Utilities -- Shared validation for pipeline and training.

Provides:
  - validate_parquet_schema(): Quick column presence check (no full load)
  - validate_loaded_data():    Full validation before training (std, nulls, etc.)
  - atomic_write_parquet():    Write-to-temp + verify + rename pattern
  - selective_drop_nulls():    Drop nulls only on model-used columns
  - extract_features_targets(): Validated feature/target extraction to numpy

Used by: anti_fragile.py, make_dataset.py, make_dataset_legacy.py, validate_world.py (all versions),
         linear_baseline.py, inspect_*.py scripts.
"""
import polars as pl
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

# ============================================================================
# Expected schema constants
# ============================================================================

EXPECTED_BASE_FEATURES = [
    # Legacy (0-12)
    "norm_deviation", "norm_fd_close", "norm_vpin", "norm_flow_imbalance",
    "norm_vol_cluster", "norm_funding", "norm_tick_count", "norm_log_volume",
    "norm_hl_spread", "hurst_regime", "norm_oi_change", "norm_return_1",
    "norm_spread_bps",
    # Extended (13-17)
    "norm_ma_distance",
    "norm_whale", "norm_efficiency", "norm_return_4", "norm_return_16",
    # Tier 1 (18-20)
    "norm_return_kurtosis", "norm_bar_duration", "norm_funding_momentum",
    # Hawkes (21-24)
    "norm_hawkes_intensity", "norm_hawkes_buy_intensity",
    "norm_hawkes_sell_intensity", "norm_hawkes_imbalance",
    # IC-boost Tier 2 (25-29)
    "norm_momentum_accel", "norm_vol_price_corr", "norm_vol_ratio",
    "norm_flow_persistence", "norm_oi_price_divergence",
    # SOTA Tier 3 (30-33)
    "norm_yz_volatility", "norm_cs_spread", "norm_perm_entropy",
    "norm_kyle_lambda",
]

EXPECTED_XD_FEATURES = [
    "xd_btc_return", "xd_btc_volatility", "xd_funding_spread",
    "xd_cross_return_mean", "xd_cross_vol_mean",
    "xd_ma_distance", "xd_momentum_rank",
]

EXPECTED_TARGETS = [
    "target_return_1", "target_return_4", "target_return_16",
    "target_return_64", "target_return_50", "target_vol_20",
]

# Vol-adjusted targets (DEPRECATED: creates vol shortcut, do NOT use for training)
EXPECTED_VOLADJ_TARGETS = [
    "target_voladj_1", "target_voladj_4", "target_voladj_16",
    "target_voladj_64",
]


def _resolve_target_prefix(df, reward_horizons, target_prefix=None):
    """Auto-detect target column prefix from DataFrame columns.

    Args:
        df: polars DataFrame with target columns
        reward_horizons: list of horizon ints (e.g. [1, 4, 16, 64])
        target_prefix: None=auto-detect (prefer raw), or explicit prefix.
                       Use "target_voladj" to opt in to vol-adjusted targets.

    Returns:
        effective_prefix: "target_return" (default) or "target_voladj"
    """
    if target_prefix is not None:
        return target_prefix
    # Default to raw returns. Voladj targets create a vol shortcut
    # where the model predicts vol (denominator) rather than returns.
    return "target_return"

# Thresholds
MIN_FEATURE_STD = 0.01
MIN_TARGET_STD = 1e-6
MAX_NULL_DROP_PCT = 10.0  # warn if >10% rows dropped
MAX_TAIL_ZEROS = 10       # in last 100 rows of target_return_50 (fill_null creates 50+, real zeros <10)


# ============================================================================
# Schema validation (lightweight, reads 0 rows)
# ============================================================================

def validate_parquet_schema(
    filepath: Path,
    require_xd: bool = True,
) -> Tuple[bool, List[str]]:
    """
    Quick schema check without loading data.

    Args:
        filepath: Path to parquet file.
        require_xd: If True, also check for 5 cross-asset features.

    Returns:
        (is_valid, list_of_issues)
    """
    issues = []
    try:
        df_cols = pl.read_parquet(filepath, n_rows=0).columns
    except Exception as e:
        return False, [f"Cannot read parquet: {e}"]

    missing_base = [f for f in EXPECTED_BASE_FEATURES if f not in df_cols]
    if missing_base:
        issues.append(f"Missing base features: {missing_base}")

    if require_xd:
        missing_xd = [f for f in EXPECTED_XD_FEATURES if f not in df_cols]
        if missing_xd:
            issues.append(f"Missing XD features: {missing_xd}")

    # Check targets (accept either raw or voladj)
    has_voladj = all(t in df_cols for t in EXPECTED_VOLADJ_TARGETS)
    missing_targets = [t for t in EXPECTED_TARGETS if t not in df_cols]
    if missing_targets and not has_voladj:
        issues.append(f"Missing targets: {missing_targets}")

    return len(issues) == 0, issues


# ============================================================================
# Full data validation (for training / evaluation)
# ============================================================================

def validate_loaded_data(
    df: pl.DataFrame,
    feature_list: list,
    reward_horizons: list,
    asset_name: str,
    strict: bool = True,
    target_prefix: str = None,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Validate that a loaded DataFrame has all expected columns with non-degenerate data.

    Args:
        df: polars DataFrame (after drop_nulls).
        feature_list: Expected feature column names.
        reward_horizons: Expected target horizons (e.g., [1, 4, 16, 64]).
        asset_name: For error messages.
        strict: If True, raise ValueError on missing columns or degenerate targets.

    Returns:
        (missing_features, missing_targets, warnings)

    Raises:
        ValueError: In strict mode, if features/targets are missing or degenerate.
    """
    missing_features = [f for f in feature_list if f not in df.columns]
    effective_prefix = _resolve_target_prefix(df, reward_horizons, target_prefix)
    target_cols = [f"{effective_prefix}_{h}" for h in reward_horizons]
    missing_targets = [t for t in target_cols if t not in df.columns]
    warnings = []

    # -- Missing columns (CRITICAL) --
    if missing_features:
        msg = f"{asset_name}: Missing features: {missing_features}"
        if strict:
            raise ValueError(f"[CRITICAL] {msg}")
        warnings.append(f"[CRITICAL] {msg}")

    if missing_targets:
        msg = f"{asset_name}: Missing targets: {missing_targets}"
        if strict:
            raise ValueError(f"[CRITICAL] {msg}")
        warnings.append(f"[CRITICAL] {msg}")

    # -- Degenerate features (std too low = dead signal) --
    for feat_name in feature_list:
        if feat_name in df.columns:
            std_val = df[feat_name].std()
            if std_val is not None and std_val < MIN_FEATURE_STD:
                warnings.append(
                    f"{asset_name}: Feature {feat_name} has std={std_val:.6f} "
                    f"(below {MIN_FEATURE_STD}, degenerate)"
                )

    # -- Degenerate targets (std too low = no signal) --
    degenerate_targets = []
    for t in target_cols:
        if t in df.columns:
            std_val = df[t].std()
            if std_val is not None and std_val < MIN_TARGET_STD:
                degenerate_targets.append((t, float(std_val)))

    if degenerate_targets:
        for tgt, std_val in degenerate_targets:
            msg = f"{asset_name}: Target {tgt} has std={std_val:.9f} (near-zero, degenerate)"
            warnings.append(f"[CRITICAL] {msg}")
        if strict:
            raise ValueError(
                f"[CRITICAL] {asset_name}: Degenerate targets: {degenerate_targets}"
            )

    # -- Tail corruption check --
    if "target_return_50" in df.columns and len(df) >= 100:
        tail = df["target_return_50"].tail(100).to_numpy()
        n_zeros = int(np.sum(np.abs(tail) < 1e-9))
        if n_zeros >= MAX_TAIL_ZEROS:
            warnings.append(
                f"{asset_name}: Tail corruption: {n_zeros} zeros in last 100 rows "
                f"of target_return_50 (threshold: {MAX_TAIL_ZEROS})"
            )

    return missing_features, missing_targets, warnings


# ============================================================================
# Atomic parquet write
# ============================================================================

def detect_ts_unit(ts_value) -> str:
    """Autodetect Binance aggTrades timestamp unit (G-AUDIT-022).

    Binance switched from milliseconds (13 digits) to microseconds (16 digits)
    in 2024-2025. Per-row autodetect via magnitude:
      ts < 1e15  -> "ms"
      ts >= 1e15 -> "us"
    Use this in any builder that reads raw aggTrades to handle the
    pre/post-switch boundary correctly.

    >>> detect_ts_unit(1577836800594)        # 2020 ms
    'ms'
    >>> detect_ts_unit(1777254909665360)     # 2026 us
    'us'
    """
    return "us" if int(ts_value) >= 1e15 else "ms"


def atomic_write_parquet(
    df: pl.DataFrame,
    out_path: Path,
    min_cols: int = 30,
    required_cols: set | None = None,
) -> None:
    """
    Write parquet via temp file + verify + atomic rename.

    On crash mid-write, the original file is preserved. Use `required_cols`
    for the strong contract — count-only validation lets schema regressions
    slip through (a 56-col file missing 7 expected features still has 56 cols).

    Args:
        df: DataFrame to write.
        out_path: Final output path.
        min_cols: Minimum expected columns (weak fallback check).
        required_cols: Set/iterable of column names that MUST be present
                       (strong contract). Preferred over min_cols.

    Raises:
        RuntimeError: If write or verification fails.
    """
    tmp_path = out_path.with_suffix(".parquet.tmp")
    try:
        df.write_parquet(tmp_path)
        # Verify the temp file is readable
        verify_cols = set(pl.read_parquet_schema(tmp_path).keys())
        if required_cols is not None:
            missing = set(required_cols) - verify_cols
            if missing:
                raise ValueError(f"missing required cols: {sorted(missing)}")
        elif len(verify_cols) < min_cols:
            raise ValueError(
                f"Written file has {len(verify_cols)} cols, expected >= {min_cols}"
            )
        # Atomic rename (Windows: must remove target first)
        if out_path.exists():
            out_path.unlink()
        tmp_path.rename(out_path)
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"Atomic write failed for {out_path.name}: {e}") from e


# ============================================================================
# Shared data loading helpers (for validate_world.py, linear_baseline.py)
# ============================================================================

def selective_drop_nulls(
    df: pl.DataFrame,
    feature_list: list,
    reward_horizons: list,
    asset_name: str = "unknown",
    target_prefix: str = None,
) -> pl.DataFrame:
    """
    Drop rows where model-used columns are null (not ALL columns).

    Only drops rows where features or targets have nulls. OHLCV, metadata,
    and other columns can have nulls without losing the row.

    Args:
        df: Raw polars DataFrame from parquet.
        feature_list: Feature column names the model uses.
        reward_horizons: Target horizons (e.g., [1, 4, 16, 64]).
        asset_name: For warning messages.

    Returns:
        Cleaned DataFrame with null rows removed.
    """
    effective_prefix = _resolve_target_prefix(df, reward_horizons, target_prefix)
    drop_subset = list(feature_list) + [f"{effective_prefix}_{h}" for h in reward_horizons]
    # Also check auxiliary targets if present (target_return_50, target_vol_20)
    aux_targets = ["target_return_50", "target_vol_20"]
    drop_subset += [c for c in aux_targets if c in df.columns]
    drop_subset = [c for c in drop_subset if c in df.columns]

    rows_before = len(df)
    df = df.drop_nulls(subset=drop_subset)
    rows_after = len(df)

    pct_dropped = (1 - rows_after / max(rows_before, 1)) * 100
    if pct_dropped > MAX_NULL_DROP_PCT:
        print(f"  [WARN] {asset_name}: dropped {pct_dropped:.1f}% rows as null "
              f"({rows_before:,} -> {rows_after:,})")

    return df


def extract_features_targets(
    df: pl.DataFrame,
    feature_list: list,
    reward_horizons: list,
    asset_name: str = "unknown",
    target_prefix: str = None,
) -> Tuple[np.ndarray, dict]:
    """
    Extract features and targets from a DataFrame with full validation.

    Validates all columns exist and checks for degenerate data.
    Never zero-pads — raises ValueError if columns are missing.

    Args:
        df: polars DataFrame (already null-dropped).
        feature_list: Feature column names to extract.
        reward_horizons: Target horizons (e.g., [1, 4, 16, 64]).
        asset_name: For error messages.

    Returns:
        (features [N, C], targets {horizon: [N]})

    Raises:
        ValueError: If features or targets are missing or degenerate.
    """
    # Validate features exist
    missing_feat = [fn for fn in feature_list if fn not in df.columns]
    if missing_feat:
        raise ValueError(
            f"[CRITICAL] {asset_name}: Missing features: {missing_feat}. "
            f"Available: {df.columns}"
        )

    # Validate targets exist (default: raw returns; voladj opt-in only)
    effective_prefix = _resolve_target_prefix(df, reward_horizons, target_prefix)
    target_cols = [f"{effective_prefix}_{h}" for h in reward_horizons]
    missing_tgt = [t for t in target_cols if t not in df.columns]
    if missing_tgt:
        raise ValueError(
            f"[CRITICAL] {asset_name}: Missing targets: {missing_tgt}. "
            f"Available: {df.columns}"
        )

    # Quality checks
    for feat_name in feature_list:
        std_val = df[feat_name].std()
        if std_val is not None and std_val < MIN_FEATURE_STD:
            print(f"  [WARN] {asset_name}: {feat_name} std={std_val:.6f} (degenerate)")

    for t in target_cols:
        std_val = df[t].std()
        if std_val is not None and std_val < MIN_TARGET_STD:
            raise ValueError(
                f"[CRITICAL] {asset_name}: {t} std={std_val:.9f} (degenerate)"
            )

    # Extract arrays (no zero-padding)
    feat_arrays = [df[fn].to_numpy().astype(np.float32) for fn in feature_list]
    feats = np.column_stack(feat_arrays)

    targets = {}
    for h in reward_horizons:
        col = f"{effective_prefix}_{h}"
        targets[h] = df[col].to_numpy().astype(np.float32)

    return feats, targets
