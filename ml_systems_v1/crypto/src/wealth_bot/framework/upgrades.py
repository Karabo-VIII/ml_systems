"""upgrades -- architectural improvements on top of the baseline signal_picker.

U1: 10-seed inference ensemble (mean of per-seed predictions)
U2: Bayesian-grid threshold calibration on VAL
U3: Regime-conditional sub-models (BTC regime tags)
U4: Synthetic-data augmentation (TRAIN-only, with gate)

__contract__:
  - U1: inputs N independent picker outputs, returns one ensemble PickerOutput
  - U2: inputs preds + VAL mask + fwd_ret, returns optimal threshold
  - U3: inputs regime tags + chimera, returns per-regime PickerOutputs
  - U4: inputs TRAIN data, returns augmented (X, y) for picker training; preserves
        marginal tail-event rate within +/- 2pp
"""
from __future__ import annotations

__contract__ = {
    "kind": "ml_upgrades",
    "owner": "wealth_bot/framework/upgrades",
    "purpose": "Stacking U1-U4 on baseline picker",
    "invariants": [
        "U1: ensemble averages predictions, NEVER mixes seeds in training",
        "U2: threshold chosen on VAL only, applied to OOS+UNSEEN",
        "U3: regime tags computed from BTC return with NO peek (lagged)",
        "U4: synthetic samples only in TRAIN, never VAL/OOS/UNSEEN",
    ],
}

import numpy as np
import pandas as pd

from .config import ModelConfig
from .signal_picker import PickerOutput, train_picker, evaluate_actions


# ============================================================================
# U1: 10-seed inference ensemble
# ============================================================================

def train_ensemble(
    df_lag: pd.DataFrame,
    signals: np.ndarray,
    fwd_ret: np.ndarray,
    chimera_features: list[str],
    fwd_bars: int,
    model_cfg: ModelConfig,
    n_seeds: int = 10,
    threshold: float = 0.0,
    verbose: bool = False,
) -> tuple[PickerOutput, list[PickerOutput]]:
    """Train n_seeds pickers, return ensemble of their predictions.

    Returns:
      ensemble_output: PickerOutput with ensemble preds + ensemble-derived actions/chosen
      per_seed_outputs: list of N PickerOutputs (per seed) for diagnostic / variance analysis
    """
    per_seed: list[PickerOutput] = []
    for s in range(n_seeds):
        if verbose:
            print(f"  ensemble seed {s+1}/{n_seeds}")
        out = train_picker(df_lag, signals, fwd_ret, chimera_features, fwd_bars,
                            model_cfg, seed=s, threshold=threshold)
        per_seed.append(out)

    # Stack preds and average (NaN-aware)
    stacked = np.stack([o.preds for o in per_seed], axis=0)  # (n_seeds, n, K)
    ensemble_preds = np.nanmean(stacked, axis=0)             # (n, K)

    # Recompute actions/chosen from ensemble preds (non-overlapping, threshold-aware)
    n, K = signals.shape
    actions = np.zeros(n, dtype=int)
    chosen = np.full(n, -1, dtype=int)
    i = 0
    while i < n:
        elig = np.where(signals[i] == 1)[0]
        if len(elig) == 0:
            i += 1
            continue
        elig_preds = ensemble_preds[i, elig]
        valid = ~np.isnan(elig_preds)
        if not valid.any():
            i += 1
            continue
        k_star_local = int(np.argmax(np.where(valid, elig_preds, -np.inf)))
        if elig_preds[k_star_local] <= threshold:
            i += 1
            continue
        actions[i] = 1
        chosen[i] = elig[k_star_local]
        i += fwd_bars

    ensemble_out = PickerOutput(
        preds=ensemble_preds,
        actions=actions,
        chosen=chosen,
        n_refits=per_seed[0].n_refits,
    )
    return ensemble_out, per_seed


# ============================================================================
# U2: Threshold calibration
# ============================================================================

def calibrate_threshold(
    preds: np.ndarray,
    signals: np.ndarray,
    fwd_ret: np.ndarray,
    masks: dict[str, np.ndarray],
    fwd_bars: int,
    threshold_grid: list[float],
    metric: str = "compound_pct",
    seg: str = "VAL",
) -> tuple[float, dict[float, float]]:
    """Pick threshold that maximizes `metric` on `seg`.

    Returns (best_threshold, threshold -> metric_value mapping).
    """
    n, K = signals.shape
    scores: dict[float, float] = {}
    for thr in threshold_grid:
        actions = np.zeros(n, dtype=int)
        i = 0
        while i < n:
            elig = np.where(signals[i] == 1)[0]
            if len(elig) == 0:
                i += 1
                continue
            elig_preds = preds[i, elig]
            valid = ~np.isnan(elig_preds)
            if not valid.any():
                i += 1
                continue
            k_star_local = int(np.argmax(np.where(valid, elig_preds, -np.inf)))
            if elig_preds[k_star_local] <= thr:
                i += 1
                continue
            actions[i] = 1
            i += fwd_bars
        results = evaluate_actions(actions, fwd_ret, {seg: masks[seg]}, fwd_bars)
        scores[thr] = results[seg][metric]
    best_thr = max(scores, key=scores.get)
    return best_thr, scores


def apply_threshold(
    preds: np.ndarray,
    signals: np.ndarray,
    fwd_bars: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a fixed threshold to produce (actions, chosen) from preds + signals."""
    n, K = signals.shape
    actions = np.zeros(n, dtype=int)
    chosen = np.full(n, -1, dtype=int)
    i = 0
    while i < n:
        elig = np.where(signals[i] == 1)[0]
        if len(elig) == 0:
            i += 1
            continue
        elig_preds = preds[i, elig]
        valid = ~np.isnan(elig_preds)
        if not valid.any():
            i += 1
            continue
        k_star_local = int(np.argmax(np.where(valid, elig_preds, -np.inf)))
        if elig_preds[k_star_local] <= threshold:
            i += 1
            continue
        actions[i] = 1
        chosen[i] = elig[k_star_local]
        i += fwd_bars
    return actions, chosen


# ============================================================================
# U3: Regime tagging (BTC-derived)
# ============================================================================

def compute_btc_regime(
    btc_close: np.ndarray,
    lag_bars: int = 6,
    bull_30d_ret_thr: float = 0.10,
    bear_30d_ret_thr: float = -0.05,
) -> np.ndarray:
    """Assign each bar a regime tag based on lagged BTC 30d return.

    Returns int array: 0=BEAR, 1=CALM, 2=BULL.
    """
    n = len(btc_close)
    out = np.full(n, 1, dtype=int)  # default CALM
    # 30d on 4h = 180 bars
    bars_30d = 180
    for i in range(bars_30d + lag_bars, n):
        ref_close = btc_close[i - lag_bars - bars_30d]
        cur_close = btc_close[i - lag_bars]
        if ref_close <= 0:
            continue
        ret_30d = cur_close / ref_close - 1
        if ret_30d > bull_30d_ret_thr:
            out[i] = 2  # BULL
        elif ret_30d < bear_30d_ret_thr:
            out[i] = 0  # BEAR
        else:
            out[i] = 1  # CALM
    return out


def train_per_regime(
    df_lag: pd.DataFrame,
    signals: np.ndarray,
    fwd_ret: np.ndarray,
    chimera_features: list[str],
    regime_tags: np.ndarray,
    fwd_bars: int,
    model_cfg: ModelConfig,
    seed: int,
    threshold: float = 0.0,
    use_global_fallback: bool = True,
) -> PickerOutput:
    """Train one picker per regime; assemble bar-by-bar predictions per regime tag.

    If a bar's regime has too few training samples, fall back to global model.
    """
    n, K = signals.shape
    regimes = np.unique(regime_tags)
    preds_per_regime: dict[int, np.ndarray] = {}

    for r in regimes:
        regime_mask = (regime_tags == r)
        if regime_mask.sum() < model_cfg.wf_train_window:
            preds_per_regime[r] = np.full((n, K), np.nan)
            continue
        signals_r = signals.copy()
        signals_r[~regime_mask] = 0
        out = train_picker(df_lag, signals_r, fwd_ret, chimera_features,
                            fwd_bars, model_cfg, seed=seed, threshold=threshold)
        preds_per_regime[r] = out.preds

    # Global fallback
    global_preds: np.ndarray | None = None
    if use_global_fallback:
        global_out = train_picker(df_lag, signals, fwd_ret, chimera_features,
                                    fwd_bars, model_cfg, seed=seed, threshold=threshold)
        global_preds = global_out.preds

    # Assemble final preds: for each bar, use the regime's preds if not NaN else global
    final_preds = np.full((n, K), np.nan)
    for i in range(n):
        r = regime_tags[i]
        rp = preds_per_regime.get(r)
        if rp is not None and not np.all(np.isnan(rp[i])):
            final_preds[i] = rp[i]
        elif global_preds is not None:
            final_preds[i] = global_preds[i]

    actions, chosen = apply_threshold(final_preds, signals, fwd_bars, threshold)
    return PickerOutput(preds=final_preds, actions=actions, chosen=chosen, n_refits=0)


# ============================================================================
# U4: Synthetic-data augmentation (TRAIN-only)
# ============================================================================

def gaussian_jitter_augment(
    X: np.ndarray,
    y: np.ndarray,
    n_synth: int,
    sigma_X: float = 0.02,
    sigma_y: float = 0.005,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simplest baseline synthetic data: gaussian-jittered copies of real samples.

    Preserves marginal distributions tightly (parametric copies).
    Use as null control for any neural-generative approach.

    Args:
      X: (n, d) features
      y: (n,) targets
      n_synth: # synthetic samples to generate
      sigma_X: stddev of feature noise (multiplicative on feature std)
      sigma_y: stddev of target noise
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n, d = X.shape
    if n_synth <= 0:
        return X, y
    feat_std = X.std(axis=0, keepdims=True)
    feat_std = np.where(feat_std > 0, feat_std, 1.0)

    # Sample with replacement from real
    idx = rng.integers(0, n, size=n_synth)
    Xs = X[idx] + rng.normal(0, sigma_X, size=(n_synth, d)) * feat_std
    ys = y[idx] + rng.normal(0, sigma_y, size=n_synth)

    return np.vstack([X, Xs]), np.concatenate([y, ys])


def stationary_block_bootstrap(
    X: np.ndarray,
    y: np.ndarray,
    n_synth: int,
    avg_block_size: int = 20,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Politis-Romano 1994 stationary block bootstrap.

    Preserves autocorrelation structure better than Gaussian jitter.
    Block size geometric with mean avg_block_size.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n, d = X.shape
    if n_synth <= 0:
        return X, y
    p = 1.0 / avg_block_size

    Xs = np.empty((n_synth, d))
    ys = np.empty(n_synth)
    i = 0
    while i < n_synth:
        start = int(rng.integers(0, n))
        length = max(1, int(rng.geometric(p)))
        length = min(length, n_synth - i)
        for j in range(length):
            src = (start + j) % n
            Xs[i + j] = X[src]
            ys[i + j] = y[src]
        i += length

    return np.vstack([X, Xs]), np.concatenate([y, ys])


def tail_preservation_check(
    y_real: np.ndarray,
    y_synth: np.ndarray,
    threshold: float = 0.10,
    tolerance_pp: float = 2.0,
) -> tuple[bool, float, float]:
    """Verify synthetic y preserves marginal tail-event rate within tolerance_pp.

    Returns (passes, real_rate, synth_rate).
    """
    real_rate = float((y_real >= threshold).mean() * 100)
    synth_rate = float((y_synth >= threshold).mean() * 100)
    delta = abs(real_rate - synth_rate)
    return delta <= tolerance_pp, real_rate, synth_rate
