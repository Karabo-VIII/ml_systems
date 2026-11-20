"""Production RSI / MACD / Bollinger Indicator implementations.

These are real ``Indicator`` subclasses (fitting the ABC in ``oracle.engine``)
built on pandas_ta.  They replace the ``_NotImplementedIndicator`` stubs in
``INDICATOR_REGISTRY``.

NUMPY 2.x / PANDAS_TA COMPATIBILITY SHIM
-----------------------------------------
pandas_ta 0.3.14b0 uses ``from numpy import NaN as npNaN`` inside
``pandas_ta/overlap/squeeze_pro.py``. ``numpy 2.x`` removed the ``NaN``
alias (only ``nan`` survives). The venv-patched file fix is NOT ROBUST --
it disappears on venv recreate.  Instead, we re-add the alias IN OUR CODE
before importing pandas_ta so that the import succeeds even on a fresh venv
with numpy 2.x.  This shim is idempotent: if pandas_ta was already imported
(e.g. in a REPL) the alias was already set; the ``getattr`` fallback is
harmless.

No code outside this module needs to know about the shim.
"""
from __future__ import annotations

import warnings

# -------------------------------------------------------------------
# ROBUST numpy.NaN shim -- must run BEFORE any pandas_ta import.
# pandas_ta 0.3.14b0 does "from numpy import NaN as npNaN" internally;
# numpy 2.x removed NaN. Re-add the alias so the import does not crash
# even on a freshly recreated venv.  getattr ensures we never shadow a
# version of numpy that still has the attribute (numpy < 2.0).
# -------------------------------------------------------------------
import numpy as _np
if not hasattr(_np, "NaN"):
    _np.NaN = _np.nan  # type: ignore[attr-defined]

# Suppress pkg_resources deprecation warnings emitted by pandas_ta on import.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import pandas_ta as ta  # noqa: E402

import numpy as np
import pandas as pd

# Import the ABC only -- never import INDICATOR_REGISTRY from engine here
# (that would create a circular import: engine -> indicators_ta -> engine).
# engine.py imports THIS module at the BOTTOM (after INDICATOR_REGISTRY is
# fully defined) and patches the registry.  By the time this module is
# imported, `Indicator` is already defined in engine.py's namespace.
from oracle.engine import Indicator  # noqa: E402


__contract__ = {
    "kind": "oracle_indicators_ta",
    "inputs": ["pandas_ta>=0.3.14", "chimera closes array (float64)", "config dict"],
    "outputs": {"golden_idx": "list[int]", "death_idx": "list[int]"},
    "invariants": [
        "signal() is CAUSAL: output[t] depends only on closes[:t+1]",
        "all indices are past-only, in-bounds [0, len(closes)-1], monotone ascending",
        "numpy.NaN shim applied before pandas_ta import (robust across venv recreates)",
        "pkg_resources DeprecationWarning suppressed at import time",
        "no emoji in print/log (cp1252 safe)",
    ],
}


# ===========================================================================
# 1.  RSIIndicator
#     config_grid: RSI length x (oversold, overbought) threshold pairs
#     Entry (golden): RSI crosses UP through oversold threshold
#                     (was <= oversold, now > oversold)
#     Exit  (death):  RSI crosses DOWN through overbought threshold
#                     (was >= overbought, now < overbought)
#     NaN warmup: RSI length + 1 bars before first valid value
# ===========================================================================
class RSIIndicator(Indicator):
    """RSI threshold-crossing entry/exit indicator.

    Golden cross: RSI crosses UP through the oversold level (entry).
    Death  cross: RSI crosses DOWN through the overbought level (exit).
    config keys:  period (int), lo (float oversold), hi (float overbought).
    """

    name = "rsi"

    def config_grid(self) -> list[dict]:
        configs = []
        for period in (7, 14, 21):
            for lo, hi in ((25, 75), (30, 70), (35, 65)):
                configs.append({"period": period, "lo": lo, "hi": hi})
        return configs

    def signal(self, dates: list, closes: np.ndarray, cfg: dict) -> dict:
        period = int(cfg["period"])
        lo = float(cfg["lo"])
        hi = float(cfg["hi"])

        s = pd.Series(closes)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rsi = ta.rsi(s, length=period)  # same-length Series, NaN at warmup

        golden_idx: list[int] = []
        death_idx: list[int] = []

        warmup = period + 1  # first bar where a crossover can be detected
        n = len(closes)

        for t in range(warmup, n):
            if np.isnan(rsi.iloc[t]) or np.isnan(rsi.iloc[t - 1]):
                continue
            prev = rsi.iloc[t - 1]
            curr = rsi.iloc[t]
            # golden: RSI crosses UP through oversold (prev <= lo, curr > lo)
            if prev <= lo and curr > lo:
                golden_idx.append(t)
            # death: RSI crosses DOWN through overbought (prev >= hi, curr < hi)
            if prev >= hi and curr < hi:
                death_idx.append(t)

        return {"golden_idx": golden_idx, "death_idx": death_idx}


# ===========================================================================
# 2.  MACDIndicator
#     config_grid: (fast, slow, signal_period) triples
#     Entry (golden): MACD line crosses ABOVE signal line
#     Exit  (death):  MACD line crosses BELOW signal line
#     NaN warmup: slow + signal_period - 1 bars
# ===========================================================================
class MACDIndicator(Indicator):
    """MACD line vs signal-line crossover entry/exit indicator.

    Golden cross: MACD line crosses above the signal line (entry).
    Death  cross: MACD line crosses below the signal line (exit).
    config keys:  fast (int), slow (int), signal (int).
    Column-name format from pandas_ta: MACD_{fast}_{slow}_{signal} /
    MACDs_{fast}_{slow}_{signal}.
    """

    name = "macd"

    def config_grid(self) -> list[dict]:
        return [
            {"fast": 12, "slow": 26, "signal": 9},
            {"fast": 8,  "slow": 21, "signal": 5},
            {"fast": 5,  "slow": 13, "signal": 3},
        ]

    def signal(self, dates: list, closes: np.ndarray, cfg: dict) -> dict:
        fast = int(cfg["fast"])
        slow = int(cfg["slow"])
        sig_len = int(cfg["signal"])

        s = pd.Series(closes)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            macd_df = ta.macd(s, fast=fast, slow=slow, signal=sig_len)

        macd_col   = f"MACD_{fast}_{slow}_{sig_len}"
        signal_col = f"MACDs_{fast}_{slow}_{sig_len}"

        if macd_col not in macd_df.columns or signal_col not in macd_df.columns:
            return {"golden_idx": [], "death_idx": []}

        macd_line   = macd_df[macd_col]
        signal_line = macd_df[signal_col]
        diff = macd_line - signal_line  # > 0 means MACD above signal

        warmup = slow + sig_len - 1
        n = len(closes)

        golden_idx: list[int] = []
        death_idx: list[int] = []

        for t in range(warmup, n):
            if np.isnan(diff.iloc[t]) or np.isnan(diff.iloc[t - 1]):
                continue
            prev = diff.iloc[t - 1]
            curr = diff.iloc[t]
            if prev <= 0 and curr > 0:
                golden_idx.append(t)
            if prev >= 0 and curr < 0:
                death_idx.append(t)

        return {"golden_idx": golden_idx, "death_idx": death_idx}


# ===========================================================================
# 3.  BollingerIndicator
#     Signal type: MEAN-REVERT LONG
#       Entry (golden): close crosses UP through lower band
#                       (prev_close <= prev_lower, curr_close > curr_lower)
#       Exit  (death):  close crosses DOWN through middle band (SMA)
#                       (prev_close >= prev_middle, curr_close < curr_middle)
#     NaN warmup: length bars (SMA needs `length` bars)
# ===========================================================================
class BollingerIndicator(Indicator):
    """Bollinger Bands mean-revert long entry/exit indicator.

    Entry (golden): close crosses UP through the lower band (oversold bounce).
    Exit  (death):  close crosses DOWN through the middle band (SMA).
    config keys: length (int), std (float).
    Column-name format: BBL_{length}_{std} / BBM_{length}_{std}.
    Handles pandas_ta format variants robustly (prefix fallback).
    """

    name = "bollinger"

    def config_grid(self) -> list[dict]:
        return [
            {"length": 20, "std": 2.0},
            {"length": 20, "std": 2.5},
            {"length": 14, "std": 2.0},
            {"length": 30, "std": 2.0},
        ]

    def signal(self, dates: list, closes: np.ndarray, cfg: dict) -> dict:
        length = int(cfg["length"])
        std = float(cfg["std"])

        s = pd.Series(closes)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bb = ta.bbands(s, length=length, std=std)

        lower_col  = f"BBL_{length}_{std}"
        middle_col = f"BBM_{length}_{std}"

        if lower_col not in bb.columns or middle_col not in bb.columns:
            # pandas_ta sometimes formats std without trailing zero (e.g. "2.0" vs "2")
            lower_candidates  = [c for c in bb.columns if c.startswith("BBL_")]
            middle_candidates = [c for c in bb.columns if c.startswith("BBM_")]
            if not lower_candidates or not middle_candidates:
                return {"golden_idx": [], "death_idx": []}
            lower_col  = lower_candidates[0]
            middle_col = middle_candidates[0]

        lower  = bb[lower_col]
        middle = bb[middle_col]

        warmup = length  # SMA needs `length` bars; first valid at bar length-1
        n = len(closes)

        golden_idx: list[int] = []
        death_idx: list[int] = []

        for t in range(warmup, n):
            if np.isnan(lower.iloc[t])  or np.isnan(lower.iloc[t - 1]):
                continue
            if np.isnan(middle.iloc[t]) or np.isnan(middle.iloc[t - 1]):
                continue
            prev_c = closes[t - 1]
            curr_c = closes[t]
            prev_l = lower.iloc[t - 1]
            curr_l = lower.iloc[t]
            prev_m = middle.iloc[t - 1]
            curr_m = middle.iloc[t]

            # Entry: close crosses UP through lower band (mean-revert long)
            if prev_c <= prev_l and curr_c > curr_l:
                golden_idx.append(t)

            # Exit: close crosses DOWN through middle band (reversion complete)
            if prev_c >= prev_m and curr_c < curr_m:
                death_idx.append(t)

        return {"golden_idx": golden_idx, "death_idx": death_idx}
