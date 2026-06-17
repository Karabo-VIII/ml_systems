"""
Live Feature Calculator
========================

Computes the 30 base features on a rolling buffer of dollar bars.
Reuses the pipeline's sota_shared_logic_v50 functions.

For live mode, we compute features on the full buffer (~2000 bars)
each time a new bar completes. This happens roughly every 5 minutes
for BTC ($2M bars), so computational cost is acceptable.

Cross-asset (XD) features are NOT available in live mode because they
require synchronized data across all assets. The 30 base features are
sufficient for all price-action strategies. WM inference with XD
features would require all assets' bars to be time-aligned.
"""
import sys
import logging
from pathlib import Path
from typing import Optional, Dict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PIPELINE_DIR = PROJECT_ROOT / "src" / "pipeline"

logger = logging.getLogger("prod.features")

# We'll import the pipeline functions lazily to avoid import-time issues
_shared_logic = None


def _get_shared_logic():
    """Lazy import of sota_shared_logic_v50."""
    global _shared_logic
    if _shared_logic is None:
        pipeline_path = str(PIPELINE_DIR)
        if pipeline_path not in sys.path:
            sys.path.insert(0, pipeline_path)
        import sota_shared_logic_v50
        _shared_logic = sota_shared_logic_v50
    return _shared_logic


def compute_features_from_buffer(bar_data: Dict[str, np.ndarray]
                                  ) -> Optional[np.ndarray]:
    """Compute 30 base features from a bar buffer.

    Args:
        bar_data: dict with keys: close, open, high, low, volume,
                  volume_usd, buy_vol, sell_vol, tick_count, timestamp, n

    Returns:
        features: np.ndarray [n, 30] or None if insufficient data
    """
    try:
        import polars as pl
    except ImportError:
        logger.error("polars not installed")
        return None

    n = bar_data["n"]
    if n < 200:
        logger.warning("Only %d bars -- need at least 200 for features", n)
        return None

    logic = _get_shared_logic()

    # Build a polars DataFrame matching the pipeline's expected schema
    df = pl.DataFrame({
        "bar_id": list(range(n)),
        "timestamp": bar_data["timestamp"].tolist(),
        "open": bar_data["open"].tolist(),
        "high": bar_data["high"].tolist(),
        "low": bar_data["low"].tolist(),
        "close": bar_data["close"].tolist(),
        "volume": bar_data["volume"].tolist(),
        "volume_usd": bar_data["volume_usd"].tolist(),
        "buy_vol": bar_data["buy_vol"].tolist(),
        "sell_vol": bar_data["sell_vol"].tolist(),
        "tick_count": bar_data["tick_count"].tolist(),
    })

    # Compute features using the shared pipeline logic
    try:
        df = logic.calculate_legacy_features(df)
    except Exception as e:
        logger.error("Feature computation failed: %s", e)
        return None

    # Extract the 30 base feature columns
    feature_cols = [
        "norm_deviation", "norm_fd_close", "norm_vpin",
        "norm_flow_imbalance", "norm_vol_cluster", "norm_funding",
        "norm_tick_count", "norm_log_volume", "norm_hl_spread",
        "hurst_regime", "norm_oi_change", "norm_return_1",
        "norm_spread_bps",
        # Extended (13-17)
        "norm_ma_distance", "norm_whale", "norm_efficiency",
        "norm_return_4", "norm_return_16",
        # Tier 1 (18-20)
        "norm_return_kurtosis", "norm_bar_duration", "norm_funding_momentum",
        # Hawkes (21-24)
        "norm_hawkes_intensity", "norm_hawkes_buy_intensity",
        "norm_hawkes_sell_intensity", "norm_hawkes_imbalance",
        # IC-boost (25-29)
        "norm_momentum_accel", "norm_vol_price_corr", "norm_vol_ratio",
        "norm_flow_persistence", "norm_oi_price_divergence",
    ]

    available_cols = [c for c in feature_cols if c in df.columns]
    if len(available_cols) < 5:
        logger.error("Only %d feature columns computed (need >= 5)",
                     len(available_cols))
        return None

    # In live mode, features requiring external data streams (funding rate,
    # open interest) are filled with zeros. The model treats zero as "neutral"
    # since features are normalized. Available features (OHLCV-derived) are
    # computed correctly; missing ones default to 0.
    missing = [c for c in feature_cols if c not in df.columns]
    if missing and not getattr(compute_features_from_buffer, "_missing_logged", False):
        logger.info("Live mode: %d/%d features available, %d filled with zeros",
                    len(available_cols), len(feature_cols), len(missing))
        compute_features_from_buffer._missing_logged = True

    # Fill missing columns with zeros
    for col in feature_cols:
        if col not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias(col))

    features = df.select(feature_cols).to_numpy().astype(np.float64)

    # Replace NaN/Inf with 0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    logger.debug("Computed %d features for %d bars (%d available)",
                len(feature_cols), n, len(available_cols))

    return features


def compute_regime_label(close: np.ndarray, window: int = 200) -> np.ndarray:
    """Compute SMA-200 regime label (matching pipeline).

    0 = bear (price < SMA * 0.95)
    1 = neutral
    2 = bull (price > SMA * 1.05)
    """
    n = len(close)
    regime = np.ones(n, dtype=np.float64)  # Default neutral

    if n < window:
        return regime

    # Rolling SMA
    sma = np.convolve(close, np.ones(window) / window, mode="full")[:n]
    # First `window-1` values are unreliable
    sma[:window - 1] = close[:window - 1]

    for i in range(window, n):
        ratio = close[i] / sma[i] if sma[i] > 0 else 1.0
        if ratio > 1.05:
            regime[i] = 2.0
        elif ratio < 0.95:
            regime[i] = 0.0
        else:
            regime[i] = 1.0

    return regime
