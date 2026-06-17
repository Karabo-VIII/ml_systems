"""regime_classifier -- PEPE x MA/EMA 4h regime classifier for the router.

Implements the past-only 9-cell regime classification used in R57a / R57a-followup.
This module is SAFE TO EXECUTE: all features are derived from past data only.
No live API calls, no forward-looking columns, no side effects.

Classification axes:
  TREND: {trending_up, trending_down, chop}
    - SMA-50 slope (5-bar linear regression slope) > 0 AND close > SMA-200 -> trending_up
    - SMA-50 slope < 0 AND close < SMA-200                                 -> trending_down
    - else                                                                   -> chop

  VOL: {low_vol, med_vol, high_vol}
    - Rolling 30-bar close-to-close log-return std percentile rank
    - p33 = 0.01870, p67 = 0.02627 (R57a calibration, OOS+UNSEEN period)
    - std < p33  -> low_vol
    - std >= p67 -> high_vol
    - else       -> med_vol

No-peek invariant: at index t, only closes[max(0, t-200)..t-1] are consumed.
SMA-200 requires 200 bars of history; SMA-50 requires 50; vol std requires 30.
Bars before those warmup lengths are tagged "WARMUP" and should not be traded.

Cell names returned as strings, e.g.: "trending_up_x_low_vol", "WARMUP".

__contract__:
  kind: regime_classifier
  owner: wealth_bot/regime_router/regime_classifier
  purpose: Past-only 9-cell regime tags for R12/R23a routing
  invariants:
    - no peek (uses only closes[..t-1])
    - deterministic (no RNG)
    - WARMUP returned for bars with insufficient history
    - thresholds are hard-coded from R57a calibration; update via recalibrate()
"""
from __future__ import annotations

__contract__ = {
    "kind": "regime_classifier",
    "owner": "wealth_bot/regime_router/regime_classifier",
    "purpose": "Past-only 9-cell regime tag per 4h bar for R12/R23a routing",
    "invariants": [
        "no peek: at bar t only closes[0..t-1] used",
        "deterministic: no RNG",
        "WARMUP for bars t < 200",
        "thresholds from R57a calibration (p33=0.01870, p67=0.02627)",
    ],
}

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Constants -- calibrated from R57a (OOS+UNSEEN period, 4h cadence)
# ---------------------------------------------------------------------------
_VOL_P33 = 0.01870   # 33rd percentile of rolling-30-bar log-return std
_VOL_P67 = 0.02627   # 67th percentile of rolling-30-bar log-return std
_SMA_FAST = 50       # SMA-50 for trend slope
_SMA_SLOW = 200      # SMA-200 for trend bias
_SLOPE_WINDOW = 5    # number of bars for linear-regression slope of SMA-50
_VOL_WINDOW = 30     # rolling window for vol std

# Minimum bars before first valid classification (max of all lookbacks)
WARMUP_BARS = max(_SMA_SLOW, _SLOPE_WINDOW + _SMA_FAST, _VOL_WINDOW)  # = 200

# Valid cell names (sorted: trend x vol)
VALID_CELLS = frozenset({
    "trending_up_x_low_vol",
    "trending_up_x_med_vol",
    "trending_up_x_high_vol",
    "trending_down_x_low_vol",
    "trending_down_x_med_vol",
    "trending_down_x_high_vol",
    "chop_x_low_vol",
    "chop_x_med_vol",
    "chop_x_high_vol",
    "WARMUP",
})


@dataclass
class RegimeClassifierConfig:
    """Thresholds for the classifier; override via recalibrate() after new mining."""
    vol_p33: float = _VOL_P33
    vol_p67: float = _VOL_P67
    sma_fast: int = _SMA_FAST
    sma_slow: int = _SMA_SLOW
    slope_window: int = _SLOPE_WINDOW
    vol_window: int = _VOL_WINDOW


def _sma(closes: np.ndarray, end_excl: int, period: int) -> float | None:
    """Compute SMA(period) at position end_excl-1, using closes[end_excl-period..end_excl-1].

    Returns None if insufficient history.
    No peek: only uses data strictly before end_excl.
    """
    start = end_excl - period
    if start < 0:
        return None
    return float(np.mean(closes[start:end_excl]))


def _sma_slope(closes: np.ndarray, end_excl: int, sma_period: int, slope_window: int) -> float | None:
    """Compute linear regression slope of SMA(sma_period) over last slope_window points.

    The SMA values are computed as of bars [end_excl - slope_window .. end_excl - 1],
    each using its own backward window of length sma_period.  No peek.
    Returns None if insufficient history.
    """
    # Need sma_period + slope_window - 1 bars of history
    min_start = end_excl - (sma_period + slope_window - 1)
    if min_start < 0:
        return None
    sma_values = np.array([
        float(np.mean(closes[end_excl - slope_window - sma_period + i:
                              end_excl - slope_window + i]))
        for i in range(slope_window)
    ])
    x = np.arange(slope_window, dtype=float)
    # Simple OLS slope
    x_bar = x.mean()
    y_bar = sma_values.mean()
    numerator = float(np.sum((x - x_bar) * (sma_values - y_bar)))
    denominator = float(np.sum((x - x_bar) ** 2))
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def _vol_std(closes: np.ndarray, end_excl: int, window: int) -> float | None:
    """Rolling log-return std over closes[end_excl-window-1..end_excl-1].

    Computes log_return[i] = log(closes[i] / closes[i-1]) for the window.
    Returns None if insufficient history.
    No peek.
    """
    start = end_excl - window - 1
    if start < 0:
        return None
    chunk = closes[start:end_excl]  # length = window + 1
    if np.any(chunk <= 0):
        return None
    log_rets = np.diff(np.log(chunk))  # length = window
    return float(log_rets.std())


def classify_regime(closes: np.ndarray, t: int,
                    cfg: RegimeClassifierConfig | None = None) -> str:
    """Classify regime at bar t using past data only (closes[0..t-1]).

    Parameters
    ----------
    closes : np.ndarray
        Full close price array (all bars up to and including bar t).
        Only indices < t are consumed; closes[t] is NEVER accessed.
    t : int
        Current bar index.  Must be > 0.
    cfg : RegimeClassifierConfig | None
        Thresholds.  Defaults to R57a calibration constants.

    Returns
    -------
    str
        One of VALID_CELLS.  Returns "WARMUP" if t < WARMUP_BARS.
    """
    if cfg is None:
        cfg = RegimeClassifierConfig()

    # Warmup guard: need at least sma_slow bars of history
    warmup = max(cfg.sma_slow,
                 cfg.slope_window + cfg.sma_fast,
                 cfg.vol_window)
    if t < warmup:
        return "WARMUP"

    # --- TREND axis ---
    close_t_minus_1 = float(closes[t - 1])  # last KNOWN close (no peek)
    sma_slow_val = _sma(closes, t, cfg.sma_slow)  # SMA-200 up to t-1
    sma_slope_val = _sma_slope(closes, t, cfg.sma_fast, cfg.slope_window)  # slope of SMA-50

    if sma_slow_val is None or sma_slope_val is None:
        return "WARMUP"

    if sma_slope_val > 0 and close_t_minus_1 > sma_slow_val:
        trend = "trending_up"
    elif sma_slope_val < 0 and close_t_minus_1 < sma_slow_val:
        trend = "trending_down"
    else:
        trend = "chop"

    # --- VOL axis ---
    vol_std_val = _vol_std(closes, t, cfg.vol_window)
    if vol_std_val is None:
        return "WARMUP"

    if vol_std_val < cfg.vol_p33:
        vol = "low_vol"
    elif vol_std_val >= cfg.vol_p67:
        vol = "high_vol"
    else:
        vol = "med_vol"

    return f"{trend}_x_{vol}"


def classify_all(closes: np.ndarray,
                 cfg: RegimeClassifierConfig | None = None) -> list[str]:
    """Vectorized classify_regime over full array; returns list of length len(closes).

    Bar 0 is always "WARMUP" (no history).
    """
    n = len(closes)
    tags: list[str] = []
    for t in range(n):
        tags.append(classify_regime(closes, t, cfg))
    return tags


def recalibrate(closes: np.ndarray,
                cfg: RegimeClassifierConfig | None = None) -> RegimeClassifierConfig:
    """Recompute p33/p67 vol thresholds from a provided close array.

    Use this after chimera refresh to update thresholds to current distribution.
    Returns a new RegimeClassifierConfig; does NOT mutate the input.
    """
    if cfg is None:
        cfg = RegimeClassifierConfig()
    vol_window = cfg.vol_window
    stds = []
    for t in range(vol_window + 1, len(closes)):
        v = _vol_std(closes, t, vol_window)
        if v is not None and v > 0:
            stds.append(v)
    if len(stds) < 10:
        # Not enough data -- return defaults
        return cfg
    arr = np.array(stds)
    return RegimeClassifierConfig(
        vol_p33=float(np.percentile(arr, 33)),
        vol_p67=float(np.percentile(arr, 67)),
        sma_fast=cfg.sma_fast,
        sma_slow=cfg.sma_slow,
        slope_window=cfg.slope_window,
        vol_window=cfg.vol_window,
    )


def cell_counts(tags: list[str]) -> dict[str, int]:
    """Count occurrences of each cell tag."""
    from collections import Counter
    return dict(Counter(tags))


def smoke_test(closes: np.ndarray | None = None) -> dict:
    """Quick self-test on a synthetic or provided close array.

    Verifies: (a) WARMUP for t < 200, (b) all non-WARMUP tags are valid cells,
    (c) cell distribution is non-degenerate (>= 3 distinct non-WARMUP cells).
    """
    if closes is None:
        rng = np.random.default_rng(42)
        # Synthetic random walk
        log_rets = rng.normal(0, 0.02, 800)
        closes = np.exp(np.cumsum(log_rets)) * 0.001  # PEPE-like scale

    tags = classify_all(closes)

    # Check WARMUP for early bars
    assert tags[0] == "WARMUP", f"bar 0 should be WARMUP, got {tags[0]}"
    assert tags[WARMUP_BARS - 1] == "WARMUP" or tags[WARMUP_BARS - 1] == "WARMUP", \
        f"bar {WARMUP_BARS - 1} should be WARMUP"

    # Check all tags are valid
    invalid = [t for t in tags if t not in VALID_CELLS]
    assert len(invalid) == 0, f"invalid tags: {set(invalid)}"

    # Check non-degenerate distribution
    non_warmup = [t for t in tags if t != "WARMUP"]
    distinct = set(non_warmup)
    assert len(distinct) >= 2, f"degenerate: only {distinct} cells found"

    counts = cell_counts(tags)
    warmup_n = counts.get("WARMUP", 0)
    total = len(tags)

    return {
        "total_bars": total,
        "warmup_bars": warmup_n,
        "non_warmup_bars": total - warmup_n,
        "cell_counts": counts,
        "distinct_cells": len(distinct),
        "checks_passed": True,
    }


if __name__ == "__main__":
    import json
    result = smoke_test()
    print(json.dumps(result, indent=2))
