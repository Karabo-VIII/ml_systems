"""
Asset Mapper -- Dynamic Nearest-Neighbor Embedding Assignment
===============================================================

Maps any asset to its closest trained WM asset based on rolling
return correlation over a recent window (default: 500 bars).

The WM models use 10 asset embeddings (BTC, ETH, SOL, BNB, XRP,
DOGE, ADA, AVAX, LINK, LTC). When trading a new asset (e.g., PEPE),
we need to assign it one of these 10 embeddings for WM inference.

Method:
  1. Compute recent returns for the new asset
  2. Compute recent returns for all 10 trained assets (from chimera data)
  3. Pearson correlation between new asset and each trained asset
  4. Assign the embedding of the highest-correlated trained asset

This is recomputed periodically (daily) so the mapping adapts to
changing market structure -- an asset that correlates with SOL today
might correlate with DOGE tomorrow during a meme rally.

The mapping also determines which walk-forward-validated strategies
to use: if PEPE maps to DOGE, it gets DOGE's strategy portfolio
(WM_Threshold, WM_Momentum, FlowMom).
"""
import logging
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger("prod.asset_mapper")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# The 10 trained assets with their embedding indices
TRAINED_ASSETS = {
    "btcusdt": 0, "ethusdt": 1, "solusdt": 2, "bnbusdt": 3,
    "xrpusdt": 4, "dogeusdt": 5, "adausdt": 6, "avaxusdt": 7,
    "linkusdt": 8, "ltcusdt": 9,
}

# Walk-forward validated strategies per trained asset
# (from walk_forward_20260331_034626.json, top 3 per asset)
WF_STRATEGIES = {
    "btcusdt": [
        {"strategy": "WM_Threshold", "params": {"rebal_days": 1.0, "threshold": 0.001}},
        {"strategy": "VPIN_Trigger", "params": {"vpin_threshold": 1.0, "rebal_days": 2.0}},
        {"strategy": "WM_Momentum",  "params": {"rebal_days": 5.0}},
    ],
    "ethusdt": [
        {"strategy": "WM_Threshold", "params": {"rebal_days": 1.0, "threshold": 0.001}},
        {"strategy": "VolBreak",     "params": {"rebal_days": 2.0}},
        {"strategy": "Donchian",     "params": {"period_days": 5.0}},
    ],
    "solusdt": [
        {"strategy": "WM_Momentum",  "params": {"rebal_days": 2.0}},
        {"strategy": "Bollinger",    "params": {"period_days": 2.0}},
        {"strategy": "FlowMom",      "params": {"rebal_days": 2.0}},
    ],
    "bnbusdt": [
        {"strategy": "WM_Threshold", "params": {"rebal_days": 1.0, "threshold": 0.001}},
        {"strategy": "Donchian",     "params": {"period_days": 3.0}},
        {"strategy": "WM_DonchFilter", "params": {"period_days": 2.0}},
    ],
    "xrpusdt": [
        {"strategy": "WM_VPIN",      "params": {"vpin_threshold": 2.0, "rebal_days": 2.0}},
        {"strategy": "WM_Momentum",  "params": {"rebal_days": 5.0}},
        {"strategy": "VPIN_Trigger", "params": {"vpin_threshold": 1.0, "rebal_days": 2.0}},
    ],
    "dogeusdt": [
        {"strategy": "WM_Threshold", "params": {"rebal_days": 1.0, "threshold": 0.001}},
        {"strategy": "WM_Momentum",  "params": {"rebal_days": 0.25}},
        {"strategy": "FlowMom",      "params": {"rebal_days": 2.0}},
    ],
    "adausdt": [
        {"strategy": "FlowMom",      "params": {"rebal_days": 2.0}},
        {"strategy": "WM_VPIN",      "params": {"vpin_threshold": 1.5, "rebal_days": 2.0}},
        {"strategy": "WM_Momentum",  "params": {"rebal_days": 5.0}},
    ],
    "avaxusdt": [
        {"strategy": "WM_VPIN",      "params": {"vpin_threshold": 1.0, "rebal_days": 2.0}},
        {"strategy": "WM_DonchFilter", "params": {"period_days": 1.0}},
        {"strategy": "VolBreak",     "params": {"rebal_days": 2.0}},
    ],
    "linkusdt": [
        {"strategy": "WM_Threshold", "params": {"rebal_days": 1.0, "threshold": 0.001}},
        {"strategy": "WM_Momentum",  "params": {"rebal_days": 2.0}},
        {"strategy": "WM_DonchFilter", "params": {"period_days": 1.0}},
    ],
    "ltcusdt": [
        {"strategy": "WM_Threshold", "params": {"rebal_days": 1.0, "threshold": 0.001}},
        {"strategy": "WM_Momentum",  "params": {"rebal_days": 5.0}},
        {"strategy": "VolBreak",     "params": {"rebal_days": 2.0}},
    ],
}


class AssetMapper:
    """Maps any asset to its nearest trained WM asset via return correlation.

    The mapping provides:
      1. asset_id (int) -- for WM embedding lookup
      2. strategy portfolio -- walk-forward-validated strategies from the mapped asset
      3. correlation strength -- confidence measure (low corr = less reliable mapping)
    """

    def __init__(self, correlation_window: int = 500,
                 min_correlation: float = 0.3):
        """
        Args:
            correlation_window: Number of recent bars for correlation calc
            min_correlation: Below this, WM strategies are disabled (price-action only)
        """
        self.window = correlation_window
        self.min_corr = min_correlation
        self.mappings: Dict[str, Dict] = {}  # {asset: {mapped_to, asset_id, corr, strategies}}
        self._trained_returns: Dict[str, np.ndarray] = {}  # Cache

    def load_trained_returns(self):
        """Load recent returns for all 10 trained assets from chimera data."""
        try:
            import polars as pl
        except ImportError:
            logger.error("polars not installed")
            return

        data_dir = PROJECT_ROOT / "data" / "processed"
        for asset in TRAINED_ASSETS:
            clean = asset.upper()
            path = data_dir / f"{clean}_v50_chimera.parquet"
            if not path.exists():
                logger.warning("No chimera for %s", asset)
                continue
            try:
                df = pl.read_parquet(path, columns=["close"])
                close = df["close"].to_numpy()
                if len(close) > self.window:
                    # Use last `window` bars
                    close = close[-self.window:]
                returns = np.diff(close) / close[:-1]
                self._trained_returns[asset] = returns
            except Exception as e:
                logger.error("Failed to load %s: %s", asset, e)

        logger.info("Loaded returns for %d trained assets", len(self._trained_returns))

    def map_asset(self, asset: str,
                   recent_close: np.ndarray) -> Dict:
        """Map a new asset to its nearest trained asset.

        Args:
            asset: e.g., 'pepeusdt'
            recent_close: array of recent close prices (>= window bars)

        Returns:
            {
                "mapped_to": str (trained asset name),
                "asset_id": int (embedding index),
                "correlation": float (Pearson r),
                "strategies": list (walk-forward validated),
                "wm_enabled": bool (corr >= min threshold),
            }
        """
        asset = asset.lower()

        # If it's already a trained asset, direct mapping
        if asset in TRAINED_ASSETS:
            self.mappings[asset] = {
                "mapped_to": asset,
                "asset_id": TRAINED_ASSETS[asset],
                "correlation": 1.0,
                "strategies": WF_STRATEGIES.get(asset, []),
                "wm_enabled": True,
            }
            return self.mappings[asset]

        # Compute returns for the new asset
        if len(recent_close) < 100:
            # Not enough data -- use BTC as default
            result = {
                "mapped_to": "btcusdt",
                "asset_id": 0,
                "correlation": 0.0,
                "strategies": WF_STRATEGIES["btcusdt"],
                "wm_enabled": False,
            }
            self.mappings[asset] = result
            return result

        n = min(len(recent_close), self.window)
        new_returns = np.diff(recent_close[-n:]) / recent_close[-n:-1]

        # Correlate with each trained asset
        if not self._trained_returns:
            self.load_trained_returns()

        best_corr = -1.0
        best_asset = "btcusdt"

        for trained_asset, trained_ret in self._trained_returns.items():
            # Align lengths
            common_len = min(len(new_returns), len(trained_ret))
            if common_len < 50:
                continue

            r1 = new_returns[-common_len:]
            r2 = trained_ret[-common_len:]

            # Pearson correlation
            corr = float(np.corrcoef(r1, r2)[0, 1])
            if np.isnan(corr):
                corr = 0.0

            if corr > best_corr:
                best_corr = corr
                best_asset = trained_asset

        wm_enabled = best_corr >= self.min_corr

        result = {
            "mapped_to": best_asset,
            "asset_id": TRAINED_ASSETS[best_asset],
            "correlation": round(best_corr, 4),
            "strategies": WF_STRATEGIES.get(best_asset, []),
            "wm_enabled": wm_enabled,
        }

        self.mappings[asset] = result
        logger.info("%s -> %s (corr=%.3f, wm=%s, strats=%d)",
                   asset.upper(), best_asset.upper(), best_corr,
                   wm_enabled, len(result["strategies"]))

        return result

    def get_all_correlations(self, asset: str,
                              recent_close: np.ndarray) -> Dict[str, float]:
        """Get correlations with ALL trained assets (for diagnostics)."""
        if len(recent_close) < 100:
            return {}

        if not self._trained_returns:
            self.load_trained_returns()

        n = min(len(recent_close), self.window)
        new_returns = np.diff(recent_close[-n:]) / recent_close[-n:-1]

        correlations = {}
        for trained_asset, trained_ret in self._trained_returns.items():
            common_len = min(len(new_returns), len(trained_ret))
            if common_len < 50:
                continue
            r1 = new_returns[-common_len:]
            r2 = trained_ret[-common_len:]
            corr = float(np.corrcoef(r1, r2)[0, 1])
            correlations[trained_asset] = round(corr if not np.isnan(corr) else 0.0, 4)

        return dict(sorted(correlations.items(), key=lambda x: -x[1]))

    def get_mapping(self, asset: str) -> Optional[Dict]:
        """Get cached mapping for an asset."""
        return self.mappings.get(asset.lower())

    def save_mappings(self, path: Path = None):
        """Save current mappings to JSON."""
        if path is None:
            from prod.config import STATE_DIR
            path = STATE_DIR / "asset_mappings.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "window": self.window,
                "min_correlation": self.min_corr,
                "mappings": self.mappings,
            }, f, indent=2)
        logger.info("Saved %d asset mappings to %s", len(self.mappings), path)

    def load_mappings(self, path: Path = None):
        """Load cached mappings from JSON."""
        if path is None:
            from prod.config import STATE_DIR
            path = STATE_DIR / "asset_mappings.json"
        if not path.exists():
            return
        try:
            with open(path) as f:
                data = json.load(f)
            self.mappings = data.get("mappings", {})
            logger.info("Loaded %d cached mappings", len(self.mappings))
        except Exception as e:
            logger.error("Failed to load mappings: %s", e)
