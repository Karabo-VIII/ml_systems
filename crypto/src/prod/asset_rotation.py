"""
Asset Rotation Engine -- Weekly cross-asset ranking and allocation.
===================================================================

Ranks all tradeable assets by composite score and allocates capital
to the top N with highest conviction. This is the primary lever for
portfolio returns -- being in the RIGHT assets matters more than
entry timing within an asset.

Composite Score = w1*momentum + w2*vpin_activity + w3*wm_signal + w4*regime

The engine runs weekly (or on-demand) and outputs:
  - Ranked list of assets with scores
  - Allocation weights for top N (conviction-sized)
  - Regime classification for strategy selection

Usage:
    from prod.asset_rotation import AssetRotationEngine
    engine = AssetRotationEngine(top_n=7)
    allocations = engine.rank_and_allocate(asset_data_dict)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class AssetScore:
    """Score components for a single asset."""
    symbol: str
    momentum_score: float = 0.0      # Price momentum (multi-timeframe)
    vpin_score: float = 0.0          # Microstructure activity
    wm_score: float = 0.0            # World model signal strength
    regime_score: float = 0.0        # Regime favorability
    composite: float = 0.0           # Weighted combination
    allocation: float = 0.0          # Capital allocation (0-1)
    regime: str = "neutral"          # bull/bear/neutral


@dataclass
class RotationConfig:
    """Configuration for asset rotation."""
    top_n: int = 7                   # Number of assets to hold
    min_score: float = 0.0           # Minimum composite score to enter

    # Score weights (sum to 1.0)
    w_momentum: float = 0.40         # Momentum is the strongest cross-asset predictor
    w_vpin: float = 0.20             # Microstructure activity (dollar-bar edge)
    w_wm: float = 0.20              # World model signal
    w_regime: float = 0.20           # Regime favorability

    # Momentum parameters
    mom_short_bars: int = 288        # ~1 day of dollar bars
    mom_medium_bars: int = 1440      # ~5 days
    mom_long_bars: int = 4320        # ~15 days

    # VPIN activity window
    vpin_window: int = 288           # ~1 day lookback for VPIN activity

    # Regime classification
    regime_sma_window: int = 1440    # ~5 day SMA for regime

    # Position sizing
    max_single_asset: float = 0.30   # Max 30% in one asset
    min_allocation: float = 0.05     # Min 5% if included


class AssetRotationEngine:
    """Ranks assets and produces allocation weights."""

    def __init__(self, config: Optional[RotationConfig] = None):
        self.config = config or RotationConfig()

    def compute_momentum(self, close: np.ndarray) -> float:
        """Multi-timeframe momentum score in [-1, 1].

        Combines short (1d), medium (5d), and long (15d) momentum.
        Faster timeframes weighted higher for responsiveness.
        """
        n = len(close)
        if n < 20:
            return 0.0

        scores = []
        weights = [0.5, 0.3, 0.2]  # Short > medium > long

        for period, w in zip([self.config.mom_short_bars,
                              self.config.mom_medium_bars,
                              self.config.mom_long_bars], weights):
            if n >= period + 1:
                ret = (close[-1] / close[-min(period, n - 1)] - 1)
                # Normalize to [-1, 1] using tanh (handles any magnitude)
                scores.append(w * np.tanh(ret * 5))  # Scale factor for crypto vol
            else:
                scores.append(0.0)

        return np.sum(scores)

    def compute_vpin_activity(self, vpin: np.ndarray,
                               flow: np.ndarray) -> float:
        """VPIN microstructure activity score in [0, 1].

        High VPIN + positive flow = informed buying (bullish).
        High VPIN + negative flow = informed selling (bearish for long-only).
        """
        n = len(vpin)
        w = min(self.config.vpin_window, n)
        if w < 10:
            return 0.0

        recent_vpin = vpin[-w:]
        recent_flow = flow[-w:]

        # Fraction of recent bars with elevated VPIN
        valid = ~np.isnan(recent_vpin) & ~np.isnan(recent_flow)
        if np.sum(valid) < 10:
            return 0.0

        high_vpin_mask = np.abs(recent_vpin[valid]) > 1.0
        if np.sum(high_vpin_mask) == 0:
            return 0.0

        # Average flow direction during high-VPIN periods
        high_vpin_flow = recent_flow[valid][high_vpin_mask]
        avg_flow = np.mean(high_vpin_flow)

        # Score: positive only if informed flow is bullish (long-only system)
        activity_pct = np.sum(high_vpin_mask) / np.sum(valid)
        if avg_flow > 0:
            return float(np.clip(activity_pct * np.tanh(avg_flow), 0, 1))
        else:
            return 0.0  # Informed selling = avoid for long-only

    def compute_wm_score(self, wm_preds: Optional[Dict],
                          n_bars: int) -> float:
        """World model signal strength in [-1, 1].

        Uses h=1 and h=4 predictions (generalizing horizons).
        """
        if wm_preds is None:
            return 0.0

        scores = []
        for h, weight in [(1, 0.5), (4, 0.5)]:
            pred = wm_preds.get(h)
            if pred is None or len(pred) < 50:
                continue
            # Average of last ~1 day of predictions
            lookback = min(288, len(pred))
            recent = pred[-lookback:]
            valid = recent[~np.isnan(recent)]
            if len(valid) < 10:
                continue
            avg_pred = np.mean(valid)
            scores.append(weight * np.tanh(avg_pred * 100))  # Scale for small preds

        return float(np.sum(scores)) if scores else 0.0

    def classify_regime(self, close: np.ndarray) -> Tuple[str, float]:
        """Regime classification with favorability score.

        Returns (regime_name, score) where score in [-1, 1]:
          +1 = strong bull (above SMA, rising)
           0 = neutral
          -1 = strong bear (below SMA, falling)
        """
        n = len(close)
        w = min(self.config.regime_sma_window, n - 1)
        if w < 50:
            return "neutral", 0.0

        sma = np.mean(close[-w:])
        current = close[-1]

        # Distance from SMA (normalized)
        distance = (current - sma) / (sma + 1e-9)

        # SMA slope (is it rising or falling?)
        if n > w + 100:
            sma_prev = np.mean(close[-(w + 100):-100])
            slope = (sma - sma_prev) / (sma_prev + 1e-9)
        else:
            slope = 0.0

        score = np.tanh(distance * 10 + slope * 50)

        if score > 0.3:
            regime = "bull"
        elif score < -0.3:
            regime = "bear"
        else:
            regime = "neutral"

        return regime, float(score)

    def rank_and_allocate(
        self,
        asset_data: Dict[str, Dict],
        wm_preds: Optional[Dict[str, Dict]] = None,
    ) -> List[AssetScore]:
        """Rank all assets and produce allocations.

        Args:
            asset_data: {symbol: {"close": np.array, "vpin": np.array, "flow": np.array}}
            wm_preds: {symbol: {horizon: np.array}} -- optional WM predictions

        Returns:
            List of AssetScore sorted by composite (descending).
            Top N have non-zero allocation weights.
        """
        cfg = self.config
        scores = []

        for symbol, data in asset_data.items():
            close = data.get("close")
            if close is None or len(close) < 100:
                continue

            s = AssetScore(symbol=symbol)

            # Momentum
            s.momentum_score = self.compute_momentum(close)

            # VPIN activity
            vpin = data.get("vpin")
            flow = data.get("flow")
            if vpin is not None and flow is not None:
                s.vpin_score = self.compute_vpin_activity(vpin, flow)

            # WM signal
            if wm_preds and symbol in wm_preds:
                s.wm_score = self.compute_wm_score(
                    wm_preds[symbol], len(close))

            # Regime
            s.regime, s.regime_score = self.classify_regime(close)

            # Composite score
            s.composite = (
                cfg.w_momentum * s.momentum_score
                + cfg.w_vpin * s.vpin_score
                + cfg.w_wm * s.wm_score
                + cfg.w_regime * max(s.regime_score, 0)  # Only reward bull regime
            )

            scores.append(s)

        # Sort by composite (descending)
        scores.sort(key=lambda x: x.composite, reverse=True)

        # Allocate to top N with positive score
        eligible = [s for s in scores if s.composite > cfg.min_score]
        top = eligible[:cfg.top_n]

        if top:
            # Conviction-weighted allocation (proportional to score)
            total_score = sum(max(s.composite, 0.01) for s in top)
            for s in top:
                raw_weight = max(s.composite, 0.01) / total_score
                # Clip to min/max allocation
                s.allocation = float(np.clip(
                    raw_weight,
                    cfg.min_allocation,
                    cfg.max_single_asset
                ))

            # Renormalize to sum to 1.0
            total_alloc = sum(s.allocation for s in top)
            if total_alloc > 0:
                for s in top:
                    s.allocation /= total_alloc

        return scores

    def format_report(self, scores: List[AssetScore]) -> str:
        """Format a readable allocation report."""
        lines = []
        lines.append("=" * 70)
        lines.append("  ASSET ROTATION -- Top %d Allocation" % self.config.top_n)
        lines.append("=" * 70)
        lines.append("  %-10s %6s %6s %6s %6s %8s %6s %7s" % (
            "Asset", "Mom", "VPIN", "WM", "Regime", "Score", "Alloc", "Regime"))
        lines.append("  " + "-" * 65)

        for s in scores:
            if s.allocation > 0:
                lines.append(
                    "  %-10s %+5.2f %+5.2f %+5.2f %+5.2f %+7.3f %5.1f%% %7s" % (
                        s.symbol, s.momentum_score, s.vpin_score,
                        s.wm_score, s.regime_score, s.composite,
                        s.allocation * 100, s.regime))

        lines.append("  " + "-" * 65)

        # Show next 5 (bench)
        bench = [s for s in scores if s.allocation == 0][:5]
        if bench:
            lines.append("  Bench:")
            for s in bench:
                lines.append(
                    "  %-10s %+5.2f %+5.2f %+5.2f %+5.2f %+7.3f   --   %7s" % (
                        s.symbol, s.momentum_score, s.vpin_score,
                        s.wm_score, s.regime_score, s.composite, s.regime))

        lines.append("=" * 70)
        return "\n".join(lines)
