"""Within-TI regime router scaffold (Phase 4).

Per framework r5 §Phase 4 — Within-TI Regime Composition:
when >=2 within-TI configs win in DIFFERENT regimes, compose them into a
regime-routed bot. ALL routed configs must be SAME-TI (e.g. all MA/EMA, all
RSI, etc.). Never cross-TI.

Three regime-detector families to test per P4.2:
  1. Self-regime: indicator's own internal state (e.g. for MA/EMA: |fast-slow|/close)
  2. External regime: BTC bull/chop/bear, vol percentile, time-of-day, microstructure
  3. Hybrid: self + external composite

Procedure (P4.3):
  1. Catalog all SHIPPED + INCONCLUSIVE within-TI configs from the dossier
  2. For each config, compute its WIN regimes (capture_L3 > 0.50 AND realized > 0)
  3. Train router on TRAIN+VAL+OOS that dispatches highest-EV config per regime
  4. Apply to UNSEEN; report composed compound + per-regime capture
  5. Robustness gates: 10-seed positive + 4-window positive + p05 > 0
  6. SHIP composed IF composed UNSEEN compound >= best single-config + 10pp

Anti-patterns (P4.4):
  - Crossing TI families: NEVER
  - Regime detector overfit: pre-register threshold sweep on TRAIN+VAL only
  - Composition complexity creep: max 3 within-TI configs

This is the SCAFFOLD only — the regime detectors + router engine are stubbed.
Per-TI-family implementation lives in the (TI, ASSET) dossier's own file
(e.g. scripts/wealth_bot/_regime_router_pepe_rsi.py).
"""
from __future__ import annotations

__contract__ = {
    "kind": "within_ti_regime_router_scaffold",
    "owner": "wealth_bot/regime_router (Phase 4 within-TI)",
    "purpose": "Within-TI regime-routing scaffold for Phase 4 composition",
    "invariants": [
        "NEVER routes across indicator families (TI*ASSET fixity per r5)",
        "Max 3 routed configs (composition complexity creep guard)",
        "Regime thresholds fit on TRAIN+VAL only (UNSEEN final holdout)",
    ],
}

from typing import Callable, Optional
import numpy as np


# ---------- Regime detectors (3 families per P4.2) ----------

def self_regime_ma_ema(
    closes: np.ndarray,
    fast_period: int,
    slow_period: int,
) -> np.ndarray:
    """Self-regime for MA/EMA family: |fast - slow| / close = trend-strength.

    Returns per-bar trend-strength magnitude (higher = more trending).
    Caller defines the binary regime via threshold (e.g. > 30d rolling median).
    """
    def ema(prices, period):
        alpha = 2.0 / (period + 1)
        out = np.zeros_like(prices, dtype=float)
        out[0] = prices[0]
        for i in range(1, len(prices)):
            out[i] = alpha * prices[i] + (1 - alpha) * out[i - 1]
        return out
    fast = ema(closes, fast_period)
    slow = ema(closes, slow_period)
    closes_safe = np.where(closes > 0, closes, 1.0)
    return np.abs(fast - slow) / closes_safe


def self_regime_rsi(rsi_values: np.ndarray) -> np.ndarray:
    """Self-regime for RSI family: distance from 50 = extension strength.

    Higher absolute value = more extended (overbought / oversold).
    """
    return np.abs(rsi_values - 50.0)


def self_regime_macd(macd_hist: np.ndarray) -> np.ndarray:
    """Self-regime for MACD family: |histogram| = trend acceleration.

    Higher = stronger directional momentum.
    """
    return np.abs(macd_hist)


def external_regime_btc_bull_chop_bear(btc_30d_returns: np.ndarray) -> np.ndarray:
    """External regime via BTC 30d return sign.

    Returns: array of {-1, 0, +1} = {bear, chop, bull}. Threshold ±5%.
    """
    out = np.zeros_like(btc_30d_returns, dtype=int)
    out[btc_30d_returns > 0.05] = 1
    out[btc_30d_returns < -0.05] = -1
    return out


def external_regime_vol_percentile(vol_series: np.ndarray, window: int = 30) -> np.ndarray:
    """External regime via volatility percentile (rolling).

    Returns per-bar percentile rank of current vol vs rolling-window history.
    """
    out = np.zeros_like(vol_series, dtype=float)
    for i in range(len(vol_series)):
        lo = max(0, i - window + 1)
        hist = vol_series[lo:i + 1]
        if len(hist) > 1:
            out[i] = (hist <= vol_series[i]).mean()
    return out


def hybrid_regime(
    self_regime: np.ndarray,
    external_regime: np.ndarray,
    self_threshold: float,
    external_label_for_combo: int = 1,
) -> np.ndarray:
    """Hybrid regime: self-regime primary AND external-regime confirmation.

    Returns binary array: 1 where (self > self_threshold) AND
    (external == external_label_for_combo); else 0.
    """
    return ((self_regime > self_threshold) & (external_regime == external_label_for_combo)).astype(int)


# ---------- Per-config win-regime cataloging ----------

def catalog_config_win_regimes(
    config_action_arrays: dict[str, np.ndarray],
    fwd_returns: np.ndarray,
    regime_label_per_bar: np.ndarray,
    mask: np.ndarray,
    capture_l3_threshold: float = 0.50,
) -> dict:
    """For each config, compute its mean return + n_fires + capture rate
    per regime label.

    Args:
      config_action_arrays: dict mapping config_name -> binary actions array
      fwd_returns: full forward-return array (one per bar)
      regime_label_per_bar: regime label per bar (e.g. {-1, 0, 1} or arbitrary ints)
      mask: bool array of bars to include in the catalog (e.g. TRAIN+VAL+OOS)
      capture_l3_threshold: a config WINS regime R if its mean return in R AND
                            its capture L3 in R both pass thresholds (capture L3
                            is approximated by mean_return > 0 here; refine
                            per-trade in the production version)

    Returns: dict[config_name -> dict[regime_label -> stats]]
    """
    out: dict = {}
    regimes = np.unique(regime_label_per_bar[mask])
    for config_name, actions in config_action_arrays.items():
        out[config_name] = {}
        for regime in regimes:
            in_regime = (regime_label_per_bar == regime) & mask
            fires_in_regime = (actions == 1) & in_regime
            n_fires = int(fires_in_regime.sum())
            if n_fires == 0:
                out[config_name][int(regime)] = {
                    "n_fires": 0, "mean_return": 0.0, "win_rate": 0.0,
                    "is_winning_regime": False,
                }
                continue
            rets = fwd_returns[fires_in_regime]
            rets = rets[~np.isnan(rets)]
            if len(rets) == 0:
                continue
            mean_ret = float(rets.mean())
            wr = float((rets > 0).mean())
            out[config_name][int(regime)] = {
                "n_fires": n_fires,
                "mean_return": mean_ret,
                "win_rate": wr,
                "is_winning_regime": mean_ret > 0 and wr >= 0.50,
            }
    return out


# ---------- Router engine ----------

def regime_route_actions(
    config_action_arrays: dict[str, np.ndarray],
    regime_label_per_bar: np.ndarray,
    config_per_regime: dict[int, str],
) -> np.ndarray:
    """Apply the regime router.

    At each bar, dispatch the config-name selected for that bar's regime.
    Returns: composed actions array (the union of the dispatched configs' fires
    on bars where those configs' regimes are active).

    config_per_regime: dict mapping regime_label -> config_name (which config to
                       use when this regime is active)

    Anti-pattern guard: composition complexity creep (P4.4) — error if >3 distinct configs.
    """
    distinct_configs = set(config_per_regime.values())
    if len(distinct_configs) > 3:
        raise ValueError(
            f"P4.4 violation: composition complexity creep — more than 3 configs in router "
            f"(got {len(distinct_configs)}). Reduce to <=3 or revisit Phase 3 dossier."
        )

    n_bars = len(regime_label_per_bar)
    composed = np.zeros(n_bars, dtype=int)
    for bar_idx in range(n_bars):
        regime = int(regime_label_per_bar[bar_idx])
        if regime not in config_per_regime:
            continue
        config_name = config_per_regime[regime]
        if config_name not in config_action_arrays:
            continue
        if config_action_arrays[config_name][bar_idx] == 1:
            composed[bar_idx] = 1
    return composed


# ---------- Pre-registration helper ----------

def write_phase4_preregistration(
    output_path: str,
    ti_asset_tuple: tuple[str, str],
    candidate_configs: list[str],
    regime_detector_families_tested: list[str],
    ship_threshold_pp: float,
    refute_threshold_pp: float,
    baseline_compound_pct: float,
) -> None:
    """Write a Phase 4 pre-registration file per the M2 protocol.

    Should be called BEFORE any router fitting / training begins.
    """
    import datetime
    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    content = f"""# Phase 4 Pre-Registration — Within-TI Regime Composition

> Written BEFORE router fitting. Immutable. Any deviation requires explicit user override.
> Per framework r5 §Phase 4 + §M2 pre-registration protocol.

## (TI, ASSET) tuple

{ti_asset_tuple[0]} × {ti_asset_tuple[1]}

**Anti-cross-pollination guard**: all candidate configs below MUST be within
this same TI family. No cross-TI routing.

## Candidate configs (must be SAME TI)

{chr(10).join('- ' + c for c in candidate_configs)}

## Regime detector families to test

{chr(10).join('- ' + f for f in regime_detector_families_tested)}

## Pre-registered ship thresholds

- **Ship**: composed UNSEEN compound >= max(baseline + {ship_threshold_pp:.1f}pp, baseline x 1.20)
  i.e. composed UNSEEN >= max({baseline_compound_pct + ship_threshold_pp:.2f}%, {baseline_compound_pct * 1.20:.2f}%)
- **Refute**: composed UNSEEN compound < baseline + {refute_threshold_pp:.1f}pp
- **Robustness preservation**: composed bot must pass 10-seed positive + 4-window positive + p05 > 0
- **Composition complexity guard**: max 3 within-TI configs in router

## Asymmetric loss

- False-positive (ship overfit router): PRIORITY 1 — capital at stake
- False-negative (miss real regime composition): PRIORITY 2 — recoverable next round
- When borderline: DEFAULT NULL; revisit at next dossier sample-size milestone

## Pre-mortem (1 sentence)

Router fails if: regime detector thresholds picked on UNSEEN leak into UNSEEN
evaluation; OR composed UNSEEN gain is small-sample artifact at n < 30 trades.

## Wall-clock to written

{datetime.datetime.now().isoformat()}
"""
    Path(output_path).write_text(content, encoding="utf-8")
