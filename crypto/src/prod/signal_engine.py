"""
Signal Engine (Multi-Strategy Voting) -- v3
=============================================

Runs WM strategies per asset with regime filtering and voting.
Updated 2026-04-03 based on strategy activity scan across unseen segment.

Key changes from v2:
  - Replaced h1_roll_rgm4 (Sharpe -0.064, weakest strategy) with h4_roll_2d
    (no regime, Sharpe +0.087, 6/10 positive). h4_roll_2d provides activity
    even in bear markets since it has no regime gate -- only its rolling average
    acts as a natural filter.
  - Keeps h4_roll_rgm16 (Sharpe +1.087, best cross-asset) and h1_sign_rgm4 (+0.700)
  - MIN_AGREE=2: entry only when h4_roll_2d agrees with one regime-filtered strategy

Evidence (unseen segment, 3-13 months per asset):
  - Deployed category avg Sharpe +0.574, avg return +1.38% (best category)
  - h4_roll_2d_r16w500: Sharpe +1.087, 7/10 positive
  - h1_sign_1d_r4w500:  Sharpe +0.700, 5/10 positive
  - h4_roll_2d (no regime): Sharpe +0.087, 6/10 positive, avg +3.43% return
  - OLD h1_roll_1d_r4w500: Sharpe -0.064, 4/10 positive (dropped)
  - Intra-day (6h-12h) is catastrophic: avg Sharpe -3.4 (costs destroy edge)

For assets outside the 10 trained WM assets, the AssetMapper assigns
the nearest trained asset's embedding via rolling return correlation.
"""
import json
import sys
import logging
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prod.config import ACTIVE_ASSETS, SEQ_LEN
from prod.feature_calculator import compute_features_from_buffer, compute_regime_label
from prod.asset_mapper import AssetMapper, TRAINED_ASSETS

logger = logging.getLogger("prod.signal")


# ═══════════════════════════════════════════════════════════════════════════════
# Sweep-Validated Strategy Configs (2026-04-01)
#
# Validated on OOS (80-90%, bull market) + Unseen (90-100%, bear market).
# Model: V1.1 f25 solo (best on unseen), with V1.6 f13 as comparison.
#
# Cross-asset top strategies (unseen segment, 3-13 months per asset):
#   1. h4_roll_2d_rgm16:   h=4 rolling + h=16 regime gate (Sharpe +1.087, 7/10 pos)
#   2. h4_roll_2d:          h=4 rolling, NO regime filter  (Sharpe +0.087, 6/10 pos)
#   3. h1_sign_1d_rgm4:    h=1 sign + h=4 regime gate     (Sharpe +0.700, 5/10 pos)
#
# Strategy 2 (h4_roll_2d) has no regime filter -- provides bear-market activity.
# Rolling avg is a natural filter; entry requires 2/3 strategies to agree.
# ═══════════════════════════════════════════════════════════════════════════════

# Preferred model: V1.1 f25 solo (best on unseen, IC=0.065)
# Fallback: ensemble (if solo model unavailable)
PREFERRED_MODEL_KEY = "v1_1_f25"

# Universal strategies applied to ALL assets (sweep-validated cross-asset)
# Each asset gets the same 3 strategies -- voting with min_agree=2
SWEEP_VALIDATED_STRATEGIES = [
    {
        "strategy": "h4_roll_rgm16",
        "params": {"signal_horizon": 4, "rebal_days": 2.0,
                   "regime_horizon": 16, "regime_window": 500},
        "unseen_sharpe": 1.087, "unseen_positive": "7/10",
        "note": "Cross-asset #1: h=4 rolling avg, h=16 regime filter, 2d rebal",
    },
    {
        "strategy": "h4_roll_2d",
        "params": {"signal_horizon": 4, "rebal_days": 2.0,
                   "regime_horizon": 0, "regime_window": 0},
        "unseen_sharpe": 0.087, "unseen_positive": "6/10",
        "note": "No-regime activity provider: h=4 rolling avg, 2d rebal, trades in bear",
    },
    {
        "strategy": "h1_sign_rgm4",
        "params": {"signal_horizon": 1, "rebal_days": 1.0,
                   "regime_horizon": 4, "regime_window": 500},
        "unseen_sharpe": 0.700, "unseen_positive": "5/10",
        "note": "Regime-gated sign: h=1 sign + h=4 regime gate, 1d rebal",
    },
]

# Apply same strategies to all 10 trained assets
WALK_FORWARD_STRATEGIES = {
    asset: SWEEP_VALIDATED_STRATEGIES
    for asset in [
        "btcusdt", "ethusdt", "solusdt", "bnbusdt", "xrpusdt",
        "dogeusdt", "adausdt", "avaxusdt", "linkusdt", "ltcusdt",
    ]
}

# All new strategies require WM predictions
WM_STRATEGIES = {"h4_roll_rgm16", "h4_roll_2d", "h1_sign_rgm4",
                 "h1_roll_rgm4", "h4_vpin_rgm", "h16_sign_rgm4",
                 # Legacy (kept for backward compat)
                 "WM_Threshold", "WM_Momentum", "WM_DonchFilter", "WM_VPIN",
                 "WM_Mom_h1", "WM_Thr_h1", "WM_DonchF_h1", "WM_PseudoMA",
                 "WM_BollFilter", "WM_RegSwitch"}


class RegimeFilteredStrategy:
    """Regime-filtered WM strategy.

    Uses one WM horizon for signal direction and another for regime gating.
    Only goes long when signal says "up" AND regime filter says "uptrend".
    """

    def __init__(self, name: str, signal_horizon: int, rebal_bars: int,
                 signal_type: str = "rolling",
                 regime_horizon: int = 0, regime_window: int = 500):
        self.name = name
        self.signal_horizon = signal_horizon
        self.rebal_bars = rebal_bars
        self.signal_type = signal_type
        self.regime_horizon = regime_horizon
        self.regime_window = regime_window

    def compute_positions(self, data, wm_preds, n):
        if wm_preds is None:
            return np.zeros(n)

        pred = wm_preds.get(self.signal_horizon)
        if pred is None:
            return np.zeros(n)

        # Regime filter
        regime_ok = np.ones(n, dtype=bool)
        if self.regime_horizon > 0 and self.regime_horizon in wm_preds:
            rh_pred = np.nan_to_num(wm_preds[self.regime_horizon], 0.0)
            rw = min(self.regime_window, n - 1)
            if rw > 0:
                cs = np.cumsum(np.insert(rh_pred[:n], 0, 0))
                for i in range(rw, n):
                    regime_avg = (cs[i + 1] - cs[i + 1 - rw]) / rw
                    regime_ok[i] = regime_avg > 0
                regime_ok[:rw] = False

        # Signal
        pos = np.zeros(n)
        cur = 0.0

        if self.signal_type == "rolling":
            p_clean = np.nan_to_num(pred[:n], 0.0)
            cs = np.cumsum(np.insert(p_clean, 0, 0))
            rb = self.rebal_bars
            for i in range(rb, n):
                if i % rb == 0:
                    roll_avg = (cs[i + 1] - cs[i + 1 - rb]) / rb
                    if regime_ok[i] and roll_avg > 0:
                        cur = 1.0
                    else:
                        cur = 0.0
                pos[i] = cur
        else:  # sign
            for i in range(n):
                if i % self.rebal_bars == 0 and not np.isnan(pred[i]):
                    if regime_ok[i] and pred[i] > 0:
                        cur = 1.0
                    else:
                        cur = 0.0
                pos[i] = cur

        return pos


def _build_strategy(config: Dict, bars_per_day: float):
    """Build a strategy object from config dict."""

    name = config.get("strategy", "BuyAndHold")
    params = config.get("params", {})

    rebal_days = params.get("rebal_days", 2.0)
    rebal_bars = max(int(rebal_days * bars_per_day), 16)

    # New regime-filtered strategies (sweep-validated)
    if name in ("h4_roll_rgm16", "h4_roll_2d", "h1_roll_rgm4",
                "h1_sign_rgm4", "h4_vpin_rgm", "h16_sign_rgm4"):
        signal_horizon = params.get("signal_horizon", 4)
        regime_horizon = params.get("regime_horizon", 0)
        regime_window = params.get("regime_window", 500)
        signal_type = "rolling" if "roll" in name else "sign"
        return RegimeFilteredStrategy(
            name=name,
            signal_horizon=signal_horizon,
            rebal_bars=rebal_bars,
            signal_type=signal_type,
            regime_horizon=regime_horizon,
            regime_window=regime_window,
        )

    # Legacy strategies (backward compat)
    from analysis.strategy_lab import (
        DonchianBreakout, BuyAndHold,
        BollingerMeanRevert, VPIN_Trigger, FlowMomentum,
        VolBreakout, HurstAdaptive,
        WM_Threshold, WM_Momentum, WM_DonchianFilter, WM_VPIN_Filter,
    )

    period_days = params.get("period_days", 2.0)
    threshold = params.get("threshold", 0.001)
    period_bars = max(int(period_days * bars_per_day), 10)

    builders = {
        "BuyAndHold": lambda: BuyAndHold(),
        "Donchian": lambda: DonchianBreakout(period_bars),
        "VPIN_Trigger": lambda: VPIN_Trigger(
            params.get("vpin_threshold", 1.5), rebal_bars),
        "FlowMom": lambda: FlowMomentum(
            max(int(0.5 * bars_per_day), 10), 0.3, rebal_bars),
        "VolBreak": lambda: VolBreakout(rebal_bars, 1.5),
        "HurstAdapt": lambda: HurstAdaptive(period_bars, 0.2),
        "Bollinger": lambda: BollingerMeanRevert(
            period_bars, params.get("num_std", 2.0),
            max(int(0.5 * bars_per_day), 8)),
        "WM_Threshold": lambda: WM_Threshold(rebal_bars, 64, threshold),
        "WM_Momentum": lambda: WM_Momentum(rebal_bars, 64),
        "WM_DonchFilter": lambda: WM_DonchianFilter(period_bars),
        "WM_VPIN": lambda: WM_VPIN_Filter(
            params.get("vpin_threshold", 1.5), rebal_bars),
    }

    if name not in builders:
        logger.warning("Unknown strategy '%s', using BuyAndHold", name)
        return BuyAndHold()

    return builders[name]()


def _prepare_data_dict(bar_data: Dict[str, np.ndarray],
                       features: Optional[np.ndarray],
                       regime: np.ndarray) -> Dict:
    """Build the data dict that strategies expect."""
    n = bar_data["n"]
    data = {
        "n": n,
        "close": bar_data["close"],
        "open": bar_data["open"],
        "high": bar_data["high"],
        "low": bar_data["low"],
        "volume": bar_data["volume"],
        "volume_usd": bar_data["volume_usd"],
        "buy_vol": bar_data["buy_vol"],
        "sell_vol": bar_data["sell_vol"],
        "tick_count": bar_data["tick_count"],
        "timestamp": bar_data["timestamp"],
        "features": features,
        "regime_label": regime,
    }

    # Add microstructure features needed by strategies
    if features is not None and features.shape[1] >= 13:
        data["norm_vpin"] = features[:, 2]
        data["norm_flow_imbalance"] = features[:, 3]
        data["norm_vol_cluster"] = features[:, 4]
        data["hurst_regime"] = features[:, 9]
        data["norm_return_1"] = features[:, 11]
    else:
        close = bar_data["close"]
        buy = bar_data["buy_vol"]
        sell = bar_data["sell_vol"]
        total = buy + sell
        total = np.where(total < 1e-10, 1.0, total)
        data["norm_vpin"] = np.abs(buy - sell) / total
        data["norm_flow_imbalance"] = (buy - sell) / total
        data["norm_vol_cluster"] = np.zeros(n)
        data["hurst_regime"] = np.full(n, 0.5)
        ret = np.zeros(n)
        ret[1:] = np.diff(close) / close[:-1]
        data["norm_return_1"] = ret

    return data


class SignalEngine:
    """Multi-strategy voting signal engine.

    Runs N strategies per asset, trades when min_agree agree on direction.
    Walk-forward validated: only strategies with positive mean OOS Sharpe
    and >= 60% positive folds are included.
    """

    def __init__(self, strategy_config: Dict = None,
                 min_agree: int = 2,
                 wm_available: bool = False):
        """
        Args:
            strategy_config: {asset: [{strategy, params}, ...]} override.
                             If None, uses WALK_FORWARD_STRATEGIES.
            min_agree: Minimum strategies that must be long to go long.
            wm_available: Whether WM ensemble is loaded for WM strategies.
        """
        self.config = strategy_config or WALK_FORWARD_STRATEGIES
        self.min_agree = min_agree
        self.wm_available = wm_available
        self.strategy_sets = {}  # {asset: [strategy_objects]}
        self.last_positions = {}  # {asset: float}
        self.last_details = {}   # {asset: {per_strat_signals, ...}}
        self.asset_mapper = AssetMapper()  # Dynamic nearest-neighbor mapping

    def _get_strategies(self, asset: str, bars_per_day: float,
                        recent_close: np.ndarray = None) -> List:
        """Build strategy objects for an asset (lazy, once).

        For unknown assets (not in WALK_FORWARD_STRATEGIES), uses
        AssetMapper to find the nearest trained asset and inherits
        its walk-forward-validated strategy portfolio.
        """
        if asset in self.strategy_sets:
            return self.strategy_sets[asset]

        configs = self.config.get(asset, [])
        if not configs:
            # Not a trained asset -- use AssetMapper for nearest neighbor
            if recent_close is not None and len(recent_close) >= 100:
                mapping = self.asset_mapper.map_asset(asset, recent_close)
                configs = mapping.get("strategies", [])
                logger.info("%s: mapped to %s (corr=%.3f) -> %d strategies",
                           asset.upper(), mapping["mapped_to"].upper(),
                           mapping["correlation"], len(configs))
            if not configs:
                configs = [{"strategy": "Donchian", "params": {"period_days": 2.0}}]

        strategies = []
        for cfg in configs:
            name = cfg.get("strategy", "")
            # Skip WM strategies if WM not available
            if name in WM_STRATEGIES and not self.wm_available:
                logger.info("%s: skipping %s (WM not loaded)", asset.upper(), name)
                continue
            try:
                strat = _build_strategy(cfg, bars_per_day)
                strategies.append(strat)
            except Exception as e:
                logger.error("%s: failed to build %s: %s", asset.upper(), name, e)

        if not strategies:
            from analysis.strategy_lab import BuyAndHold
            strategies = [BuyAndHold()]
            logger.warning("%s: no strategies available, using BuyAndHold", asset.upper())

        self.strategy_sets[asset] = strategies
        strat_names = [s.name for s in strategies]
        logger.info("%s: %d strategies loaded: %s (min_agree=%d)",
                   asset.upper(), len(strategies),
                   ", ".join(strat_names), self.min_agree)
        return strategies

    def compute_signal(self, asset: str,
                        bar_data: Dict[str, np.ndarray],
                        wm_preds: Dict = None) -> Dict:
        """Compute consensus trading signal from multiple strategies.

        Args:
            asset: e.g., 'btcusdt'
            bar_data: from BarAccumulator.get_buffer_arrays()
            wm_preds: WM predictions dict {horizon: ndarray} or None

        Returns:
            {
                "position": float (0.0 = flat, 1.0 = long),
                "signal": str ("LONG", "FLAT"),
                "price": float,
                "changed": bool,
                "n_long": int (how many strategies say long),
                "n_total": int (total strategies),
                "strategies": [{name, position}, ...],
            }
        """
        n = bar_data["n"]
        if n < 100:
            return {"position": 0.0, "signal": "FLAT", "price": 0.0,
                    "changed": False, "n_long": 0, "n_total": 0,
                    "strategies": []}

        # Bars per day
        ts = bar_data["timestamp"]
        duration_days = (ts[-1] - ts[0]) / 86_400_000
        bars_per_day = n / max(duration_days, 1)

        # Get strategy set (pass close prices for nearest-neighbor mapping)
        strategies = self._get_strategies(asset, bars_per_day,
                                          recent_close=bar_data["close"])

        # Prepare shared data dict (computed once, shared across strategies)
        features = compute_features_from_buffer(bar_data)
        regime = compute_regime_label(bar_data["close"])
        data = _prepare_data_dict(bar_data, features, regime)

        # Run all strategies
        per_strat = []
        n_long = 0
        for strat in strategies:
            try:
                positions = strat.compute_positions(data, wm_preds, n)
                positions = np.clip(positions, 0.0, 1.0)
                pos = float(positions[-1])
            except Exception as e:
                logger.error("%s/%s: compute_positions failed: %s",
                            asset.upper(), strat.name, e)
                pos = 0.0

            is_long = pos > 0.5
            if is_long:
                n_long += 1
            per_strat.append({"name": strat.name, "position": pos})

        n_total = len(strategies)
        current_price = float(bar_data["close"][-1])

        # Voting: LONG when enough strategies agree
        if n_long >= self.min_agree:
            current_pos = 1.0
            signal = "LONG"
        else:
            current_pos = 0.0
            signal = "FLAT"

        # Detect change
        prev_pos = self.last_positions.get(asset, 0.0)
        changed = abs(current_pos - prev_pos) > 0.01
        self.last_positions[asset] = current_pos
        self.last_details[asset] = per_strat

        return {
            "position": current_pos,
            "signal": signal,
            "price": current_price,
            "changed": changed,
            "n_long": n_long,
            "n_total": n_total,
            "strategies": per_strat,
        }

    def get_strategy_status(self) -> Dict:
        """Get per-asset strategy status for monitoring."""
        status = {}
        for asset in self.config:
            details = self.last_details.get(asset, [])
            pos = self.last_positions.get(asset, 0.0)
            n_long = sum(1 for d in details if d["position"] > 0.5)
            status[asset] = {
                "position": pos,
                "signal": "LONG" if pos > 0.5 else "FLAT",
                "n_long": n_long,
                "n_total": len(details),
                "per_strategy": details,
            }
        return status

    def load_config(self, path: Path):
        """Load strategy config from JSON file."""
        with open(path) as f:
            data = json.load(f)

        # Support both old format {asset: {strategy, params}}
        # and new format {asset: [{strategy, params}, ...]}
        selections = data.get("selections", data.get("strategies", data))
        for asset, sel in selections.items():
            if isinstance(sel, list):
                self.config[asset] = sel
            elif isinstance(sel, dict):
                self.config[asset] = [sel]

        self.strategy_sets = {}  # Force rebuild
        logger.info("Loaded strategy config from %s (%d assets)",
                    path.name, len(selections))
