"""signal_picker -- LGBM signal-picker: per-strategy forward-return regression.

Factored from `scripts/oracle/pepe_ml_deploy_only.py`'s `approach_lgbm_per_strategy`,
parameterized for the framework. This is the substrate that survived 10-seed audit
at +49.8% median UNSEEN (see runs/audit/PEPE_EMA_ML_DEPLOY_ONLY_2026_05_24/lgbm_seed_audit.log).

__contract__:
  inputs: df_lag, signals, fwd_ret, ModelConfig, seed
  outputs: preds (n, n_strats) — predicted forward return per (bar, strategy);
           actions (n,) binary — 1 if fire at bar t else 0;
           chosen (n,) — chosen strategy index per bar (-1 if no fire).
  invariants:
    - All training data strictly precedes prediction window (walk-forward, no peek)
    - Strategy k's model trained only on bars where signals[:, k] == 1
    - Prediction at bar t uses df_lag features (chimera already lagged at this layer)
    - Non-overlapping execution: after firing at bar t, skip fwd_bars bars
"""
from __future__ import annotations

__contract__ = {
    "kind": "signal_picker",
    "owner": "wealth_bot/framework/signal_picker",
    "purpose": "Per-strategy LGBM regressor + walk-forward + non-overlapping execution",
    "invariants": [
        "no peek (walk-forward train window ends at cur)",
        "non-overlapping execution (skip fwd_bars after fire)",
        "chimera features pre-lagged by caller",
        "seed must be EXPLICIT (no default to global RNG)",
    ],
}

from dataclasses import dataclass

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from .config import ModelConfig


@dataclass
class PickerOutput:
    preds: np.ndarray         # (n, K)
    actions: np.ndarray       # (n,) binary
    chosen: np.ndarray        # (n,) int (-1 if none)
    n_refits: int
    # BINDING 2026-05-25 (architect-expert recommendation):
    # per-refit feature importance is essential for trust-but-verify on LGBM.
    # importances[k] = list of (refit_idx, feature_name, gain) tuples per
    # strategy k. Caller can aggregate across seeds for the audit JSON.
    importances: list[list[dict]] | None = None


def train_picker(
    df_lag: pd.DataFrame,
    signals: np.ndarray,
    fwd_ret: np.ndarray,
    chimera_features: list[str],
    fwd_bars: int,
    model_cfg: ModelConfig,
    seed: int,
    threshold: float = 0.0,
) -> PickerOutput:
    """Walk-forward train per-strategy LGBM regressors and produce actions.

    Args:
      df_lag: DataFrame with chimera features (already lagged).
      signals: (n, K) binary signal matrix.
      fwd_ret: (n,) forward returns.
      chimera_features: list of feature column names.
      fwd_bars: # bars to hold (= cost-of-overlap).
      model_cfg: LGBM hyperparams + WF window.
      seed: explicit random seed (no fallback to global).
      threshold: predicted fwd_ret must exceed this to fire. Default 0.0.

    Returns:
      PickerOutput with preds (n,K), actions (n,) binary, chosen (n,) int.
    """
    n, K = signals.shape

    X_full = df_lag[chimera_features].values.astype(float)
    col_med = np.nan_to_num(np.nanmedian(X_full, axis=0), nan=0.0)
    X_full = np.where(np.isnan(X_full), col_med, X_full)

    preds = np.full((n, K), np.nan)
    cur = model_cfg.wf_train_window
    refits = 0
    # Per-strategy per-refit feature importance log (architect-expert binding 2026-05-25)
    importances: list[list[dict]] = [[] for _ in range(K)]
    # BINDING 2026-05-25 (validator-expert finding): decorrelate the LGBM RNG
    # sub-streams. Using the same integer for bagging_seed AND feature_fraction_seed
    # AND random_state correlates row-sampling with column-sampling, producing
    # systematically less diverse ensemble members across seeds. Offset each by
    # a fixed prime; observed effect was a meaningful drop in inter-seed
    # variance once decorrelated (less ensemble averaging masks per-seed noise).
    bag_seed = int(seed)
    feat_seed = int(seed) + 1000
    rng_seed = int(seed) + 7919  # prime offset to fully decorrelate

    while cur < n:
        ts = max(0, cur - model_cfg.wf_train_window)
        te = cur
        X_tr = X_full[ts:te]

        for k in range(K):
            mask_k = (signals[ts:te, k] == 1) & (~np.isnan(fwd_ret[ts:te]))
            if mask_k.sum() < model_cfg.min_signal_count_per_refit:
                continue
            X_k = X_tr[mask_k]
            y_k = fwd_ret[ts:te][mask_k]
            try:
                m = lgb.LGBMRegressor(
                    n_estimators=model_cfg.n_estimators,
                    max_depth=model_cfg.max_depth,
                    num_leaves=model_cfg.num_leaves,
                    min_child_samples=model_cfg.min_child_samples,
                    learning_rate=model_cfg.learning_rate,
                    reg_alpha=model_cfg.reg_alpha,
                    reg_lambda=model_cfg.reg_lambda,
                    bagging_fraction=model_cfg.bagging_fraction,
                    feature_fraction=model_cfg.feature_fraction,
                    bagging_seed=bag_seed,
                    feature_fraction_seed=feat_seed,
                    random_state=rng_seed,
                    n_jobs=1,
                    verbose=-1,
                )
                m.fit(X_k, y_k)
                pred_end = min(cur + model_cfg.wf_step, n)
                preds[cur:pred_end, k] = m.predict(X_full[cur:pred_end])
                gains = list(m.booster_.feature_importance(importance_type="gain"))
                importances[k].append({
                    "refit_idx": refits,
                    "train_start": int(ts),
                    "train_end": int(te),
                    "n_train_samples": int(mask_k.sum()),
                    "feature_gains": dict(zip(chimera_features, [float(g) for g in gains])),
                })
            except Exception:
                pass

        cur += model_cfg.wf_step
        refits += 1

    # Allocation
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

    return PickerOutput(preds=preds, actions=actions, chosen=chosen, n_refits=refits,
                        importances=importances)


def evaluate_actions(
    actions: np.ndarray,
    fwd_ret: np.ndarray,
    masks: dict[str, np.ndarray],
    fwd_bars: int,
) -> dict[str, dict[str, float]]:
    """Compound non-overlapping returns per segment.

    Returns dict segment -> {compound_pct, n_trades, win_rate, max_dd_pct, sharpe}.
    """
    out: dict[str, dict[str, float]] = {}
    for seg_name, mask in masks.items():
        idx = np.where(mask)[0]
        if len(idx) < 10:
            out[seg_name] = {"compound_pct": 0.0, "n_trades": 0, "win_rate": 0.0,
                              "max_dd_pct": 0.0, "sharpe": 0.0}
            continue
        rets = []
        i = 0
        while i < len(idx):
            j = idx[i]
            if actions[j] == 1 and not np.isnan(fwd_ret[j]):
                rets.append(fwd_ret[j])
                i += fwd_bars
            else:
                i += 1
        if not rets:
            out[seg_name] = {"compound_pct": 0.0, "n_trades": 0, "win_rate": 0.0,
                              "max_dd_pct": 0.0, "sharpe": 0.0}
            continue
        arr = np.array(rets)
        equity = np.cumprod(1 + arr)
        compound = (equity[-1] - 1) * 100
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        max_dd = float(dd.min() * 100)
        wr = float((arr > 0).mean())
        # Annualized Sharpe (~252 trade-days/yr equivalent; rough proxy)
        sharpe = float(arr.mean() / arr.std() * np.sqrt(len(arr))) if arr.std() > 0 else 0.0
        out[seg_name] = {
            "compound_pct": float(compound),
            "n_trades": len(rets),
            "win_rate": wr,
            "max_dd_pct": max_dd,
            "sharpe": sharpe,
        }
    return out
