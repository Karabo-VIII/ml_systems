"""Selection-null estimator (PURE).

A strategy's trigger selects a subset of candidate windows; each window has a
forward ("move") return over its horizon. Do trigger-picked windows beat windows
chosen at random, matched on the same strata (regime x horizon) and the same
per-stratum counts?

This module is PURE: it does NOT generate, simulate, load, or mutate data. It
takes forward_returns, trigger_mask, and strata as INPUTS and returns observed
effect, null distribution, and p-value. Data generation lives in the caller.

method="exact": stratified exact permutation test. Within each stratum the
trigger picks k_s of n_s windows; under the null those picks are exchangeable.
We enumerate subset sums per stratum (C(n_s,k_s) each) and convolve across
strata for the FULL null distribution of the picked-return sum -> an exact
rational p-value with NO random draws (closed-form ground truth).

method="bootstrap": same null via Monte-Carlo with a fixed seed (reproducible).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from itertools import combinations
from typing import Optional, Sequence

import numpy as np

__all__ = ["SelectionNullResult", "make_strata", "selection_null_test"]

_EPS = 1e-9


@dataclass(frozen=True)
class SelectionNullResult:
    observed_effect: float      # mean forward return of trigger-picked windows
    null_mean: float            # mean of the null distribution of that statistic
    effect_vs_null: float       # observed_effect - null_mean
    p_value: float              # stratified permutation p-value
    method: str
    alternative: str
    n_picked: int
    n_total: int
    n_strata: int
    null_size: int              # exact: total combos; bootstrap: n_boot
    exact: bool                 # True if p-value has no random draws

    def as_dict(self) -> dict:
        return asdict(self)


def make_strata(regime=None, horizon=None, n=None):
    """Composite stratum label array from regime and/or horizon dimensions."""
    if regime is None and horizon is None:
        if n is None:
            raise ValueError("provide regime, horizon, or n")
        return np.zeros(int(n), dtype=object)
    length = None
    if regime is not None:
        regime = np.asarray(regime, dtype=object)
        length = len(regime)
    if horizon is not None:
        horizon = np.asarray(horizon, dtype=object)
        length = len(horizon)
    if regime is not None and horizon is not None and len(regime) != len(horizon):
        raise ValueError("regime and horizon must have equal length")
    out = np.empty(length, dtype=object)
    for i in range(length):
        r = regime[i] if regime is not None else "_"
        h = horizon[i] if horizon is not None else "_"
        out[i] = (r, h)
    return out
