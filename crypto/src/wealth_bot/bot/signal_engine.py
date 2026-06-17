"""signal_engine -- per-bar decision from ensemble preds + threshold.

Wraps the trained ensemble preds matrix and produces a per-bar SignalDecision
that the runner consumes. This module is the "inference head" of the bot;
training happened upstream in framework.walk_forward / framework.upgrades.

__contract__:
  inputs:
    - ensemble_preds: (n_bars, n_strats) ndarray of predicted forward returns
    - threshold: float -- predicted fwd_ret must exceed this to fire
    - signals: (n_bars, n_strats) binary static-eligibility matrix
  outputs:
    - SignalDecision per (bar, signals_row, ensemble_preds_row)
  invariants:
    - fire only if (a) at least one static strategy is eligible AND
                   (b) the best-eligible predicted fwd_ret > threshold
    - chosen_strategy_idx is the eligible strategy maximizing predicted fwd_ret
    - confidence in [0, 1]; degenerates to 0 if no fire
"""
from __future__ import annotations

__contract__ = {
    "kind": "signal_engine",
    "owner": "wealth_bot/bot/signal_engine",
    "purpose": "Per-bar inference head: ensemble preds -> SignalDecision",
    "invariants": [
        "fire requires static eligibility AND predicted fwd_ret > threshold",
        "chosen idx is the argmax-eligible (NaN-safe)",
        "confidence bounded [0, 1]",
        "no peek -- only the bar's own preds row is consulted",
    ],
}

from dataclasses import dataclass

import numpy as np


@dataclass
class SignalDecision:
    """Per-bar decision packet consumed by the runner."""
    fire: bool
    chosen_strategy_idx: int        # -1 if no fire
    predicted_fwd_ret: float        # NaN if no fire
    confidence: float               # 0..1


class SignalEngine:
    """Inference head: ensemble preds -> SignalDecision per bar.

    Holds a frozen ensemble predictions matrix (already trained via walk_forward).
    Stateless per-call; calling threads can share one instance safely.
    """

    def __init__(
        self,
        ensemble_preds: np.ndarray,
        signals: np.ndarray,
        threshold: float = 0.0,
        strategy_names: list[str] | None = None,
    ) -> None:
        if ensemble_preds.shape != signals.shape:
            raise ValueError(
                f"ensemble_preds shape {ensemble_preds.shape} != signals shape {signals.shape}"
            )
        self.ensemble_preds = ensemble_preds
        self.signals = signals
        self.threshold = float(threshold)
        self.strategy_names = strategy_names or [f"strat_{i}" for i in range(signals.shape[1])]
        self.n_bars, self.n_strats = signals.shape

    def predict_bar(
        self,
        bar_idx: int,
        signals_row: np.ndarray | None = None,
        ensemble_preds_row: np.ndarray | None = None,
    ) -> SignalDecision:
        """Decide for one bar.

        Optional signals_row / ensemble_preds_row let the caller override (e.g.,
        in a live setting where rows arrive from a stream). When omitted we
        index into the frozen matrices.
        """
        s_row = signals_row if signals_row is not None else self.signals[bar_idx]
        p_row = ensemble_preds_row if ensemble_preds_row is not None else self.ensemble_preds[bar_idx]

        elig = np.where(s_row == 1)[0]
        if len(elig) == 0:
            return SignalDecision(fire=False, chosen_strategy_idx=-1,
                                  predicted_fwd_ret=float("nan"), confidence=0.0)

        elig_preds = p_row[elig]
        valid = ~np.isnan(elig_preds)
        if not valid.any():
            return SignalDecision(fire=False, chosen_strategy_idx=-1,
                                  predicted_fwd_ret=float("nan"), confidence=0.0)

        # Argmax over valid eligible preds (mask NaNs out)
        masked = np.where(valid, elig_preds, -np.inf)
        k_star_local = int(np.argmax(masked))
        best_pred = float(elig_preds[k_star_local])

        if best_pred <= self.threshold:
            return SignalDecision(fire=False, chosen_strategy_idx=-1,
                                  predicted_fwd_ret=best_pred, confidence=0.0)

        # Confidence = (best - threshold) / max(|preds across STRATEGIES at this bar|)
        # Use the full row (including non-eligible) so the denominator is
        # stable across bars; fall back to 1 if all NaN or zero.
        row_abs = np.nanmax(np.abs(p_row))
        if not np.isfinite(row_abs) or row_abs <= 0:
            conf = 1.0
        else:
            conf = (best_pred - self.threshold) / row_abs
            conf = float(np.clip(conf, 0.0, 1.0))

        return SignalDecision(
            fire=True,
            chosen_strategy_idx=int(elig[k_star_local]),
            predicted_fwd_ret=best_pred,
            confidence=conf,
        )
