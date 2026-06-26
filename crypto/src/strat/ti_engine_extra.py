"""SOTA TI Engine — VOLUME family + MISCELLANEOUS indicators.

Extends the proven SOTA architecture (ti_engine_sota.py) to the TWO genuine gaps
the first 8-family campaign never covered:

  VOLUME family (uses REAL chimera volume/buy_vol/sell_vol — NO price proxies):
    - OBV       On-Balance Volume vs its own EMA (accumulation)
    - VWAP      rolling VWAP, long when close > VWAP
    - CMF       Chaikin Money Flow > threshold (buying pressure)
    - MFI       Money Flow Index (volume-weighted RSI), hysteresis
    - FLOW      buy/sell volume imbalance (chimera buy_vol/sell_vol) — crypto-native

  MISC family (the non-redundant trend/oscillator gaps):
    - DONCHIAN  N-bar breakout (turtle) — different mechanism (range break, not MA)
    - CCI       Commodity Channel Index > threshold
    - WILLR     Williams %R out of oversold
    - PSAR      Parabolic SAR trailing-stop (close > SAR)
    - AROON     Aroon-Up > Aroon-Down + strength
    - VORTEX    VI+ > VI-
    - KELTNER   EMA ± mult*ATR channel (close > mid)

EVERY engine plugs into the SAME rigor as the validated families:
  1. CONFIG ENSEMBLE — multiple params, majority vote = selectivity
  2. REGIME GATE — V1 BTC SMA100 (Layer 1, verified)
  3. MAKER COST — 0.0006 RT on every turnover event
  4. DECOUPLED lookback (signal) != slice/hold (trade horizon)
  5. PER-SLICE 7d non-overlapping evaluation

All indicator functions are CAUSAL (no look-ahead) and vectorized where the
recurrence allows (PSAR / Aroon keep a single light loop — inherently sequential).

Honest framing carried over from the prior campaign (FORENSIC_AUDIT_RESULT.md):
DEV-window compounds are inflated ~9.5x by the EW-rebalancing premium + regime
gate. The HONEST arbiter is OOS linear/wk edge vs always-long + random-null +
DEV/OOS consistency. This module produces holds; ti_extra_runner does the gauntlet.
"""
from __future__ import annotations

import warnings; warnings.filterwarnings("ignore")
import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CRYPTO = ROOT.parent

from strat.ma_type_upgrade import _ema


# =============================================================================
# VOLUME FAMILY — indicator held functions (causal, real volume)
# =============================================================================

def obv_held(close: np.ndarray, volume: np.ndarray, ema_period: int) -> np.ndarray:
    """On-Balance Volume vs its own EMA. Long when OBV > EMA(OBV) = accumulation.

    OBV[i] = OBV[i-1] + sign(close[i]-close[i-1]) * volume[i]. Causal cumulative.
    """
    c = np.asarray(close, dtype=float)
    v = np.asarray(volume, dtype=float)
    n = len(c)
    if np.all(~np.isfinite(v)):
        return np.zeros(n, dtype=np.int8)
    v = np.nan_to_num(v, nan=0.0)
    sign = np.sign(np.diff(c, prepend=c[0]))
    obv = np.cumsum(sign * v)
    obv_ema = _ema(obv, ema_period)
    held = (obv > obv_ema).astype(np.int8)
    held[:ema_period] = 0
    return held


def vwap_held(high: np.ndarray, low: np.ndarray, close: np.ndarray,
              volume: np.ndarray, period: int) -> np.ndarray:
    """Rolling VWAP. Long when close > rolling VWAP (price above value)."""
    c = np.asarray(close, dtype=float)
    v = np.asarray(volume, dtype=float)
    n = len(c)
    if np.all(~np.isfinite(v)):
        return np.zeros(n, dtype=np.int8)
    v = np.nan_to_num(v, nan=0.0)
    tp = (np.asarray(high, float) + np.asarray(low, float) + c) / 3.0
    pv = pd.Series(tp * v).rolling(period, min_periods=period).sum().to_numpy()
    vv = pd.Series(v).rolling(period, min_periods=period).sum().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = np.where(vv > 0, pv / vv, np.nan)
    held = (c > vwap).astype(np.int8)
    held[~np.isfinite(vwap)] = 0
    return held


def cmf_held(high: np.ndarray, low: np.ndarray, close: np.ndarray,
             volume: np.ndarray, period: int, threshold: float) -> np.ndarray:
    """Chaikin Money Flow. Long when CMF > threshold (net buying pressure)."""
    h = np.asarray(high, float); l = np.asarray(low, float); c = np.asarray(close, float)
    v = np.asarray(volume, float)
    n = len(c)
    if np.all(~np.isfinite(v)):
        return np.zeros(n, dtype=np.int8)
    v = np.nan_to_num(v, nan=0.0)
    rng = h - l
    with np.errstate(divide="ignore", invalid="ignore"):
        mfm = np.where(rng > 1e-12, ((c - l) - (h - c)) / rng, 0.0)
    mfv = mfm * v
    sum_mfv = pd.Series(mfv).rolling(period, min_periods=period).sum().to_numpy()
    sum_v = pd.Series(v).rolling(period, min_periods=period).sum().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        cmf = np.where(sum_v > 0, sum_mfv / sum_v, np.nan)
    held = (cmf > threshold).astype(np.int8)
    held[~np.isfinite(cmf)] = 0
    return held


def mfi_held(high: np.ndarray, low: np.ndarray, close: np.ndarray,
             volume: np.ndarray, period: int, long_thr: float, exit_thr: float) -> np.ndarray:
    """Money Flow Index (volume-weighted RSI). Hysteresis enter>long / exit<exit."""
    h = np.asarray(high, float); l = np.asarray(low, float); c = np.asarray(close, float)
    v = np.asarray(volume, float)
    n = len(c)
    if np.all(~np.isfinite(v)):
        return np.zeros(n, dtype=np.int8)
    v = np.nan_to_num(v, nan=0.0)
    tp = (h + l + c) / 3.0
    rmf = tp * v
    dtp = np.diff(tp, prepend=tp[0])
    pos = np.where(dtp > 0, rmf, 0.0)
    neg = np.where(dtp < 0, rmf, 0.0)
    pos_s = pd.Series(pos).rolling(period, min_periods=period).sum().to_numpy()
    neg_s = pd.Series(neg).rolling(period, min_periods=period).sum().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        mfr = np.where(neg_s > 1e-12, pos_s / neg_s, 100.0)
    mfi = 100.0 - (100.0 / (1.0 + mfr))
    held = np.zeros(n, dtype=np.int8)
    state = 0
    for i in range(period, n):
        if not np.isfinite(mfi[i]):
            held[i] = state
            continue
        if state == 0 and mfi[i] > long_thr:
            state = 1
        elif state == 1 and mfi[i] < exit_thr:
            state = 0
        held[i] = state
    return held


def flow_held(buy_vol: np.ndarray, sell_vol: np.ndarray, period: int,
              threshold: float) -> np.ndarray:
    """Buy/sell volume-imbalance (crypto-native, real chimera buy_vol/sell_vol).
    Long when rolling buy-fraction - 0.5 > threshold (aggressive buying dominates)."""
    bv = np.asarray(buy_vol, float); sv = np.asarray(sell_vol, float)
    n = len(bv)
    if np.all(~np.isfinite(bv)) or np.all(~np.isfinite(sv)):
        return np.zeros(n, dtype=np.int8)
    bv = np.nan_to_num(bv, nan=0.0); sv = np.nan_to_num(sv, nan=0.0)
    bsum = pd.Series(bv).rolling(period, min_periods=period).sum().to_numpy()
    ssum = pd.Series(sv).rolling(period, min_periods=period).sum().to_numpy()
    tot = bsum + ssum
    with np.errstate(divide="ignore", invalid="ignore"):
        buy_frac = np.where(tot > 0, bsum / tot, np.nan)
    held = ((buy_frac - 0.5) > threshold).astype(np.int8)
    held[~np.isfinite(buy_frac)] = 0
    return held


# =============================================================================
# MISC FAMILY — indicator held functions (causal)
# =============================================================================

def donchian_held(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                  period: int) -> np.ndarray:
    """Donchian breakout (turtle). Long when close breaks PRIOR N-bar high; exit on
    break of PRIOR N-bar low. Bands use shift(1) so no look-ahead."""
    h = pd.Series(high, dtype=float); l = pd.Series(low, dtype=float)
    c = np.asarray(close, float)
    upper = h.rolling(period, min_periods=period).max().shift(1).to_numpy()
    lower = l.rolling(period, min_periods=period).min().shift(1).to_numpy()
    n = len(c)
    held = np.zeros(n, dtype=np.int8)
    state = 0
    for i in range(period, n):
        if not (np.isfinite(upper[i]) and np.isfinite(lower[i])):
            held[i] = state
            continue
        if state == 0 and c[i] > upper[i]:
            state = 1
        elif state == 1 and c[i] < lower[i]:
            state = 0
        held[i] = state
    return held


def cci_held(high: np.ndarray, low: np.ndarray, close: np.ndarray,
             period: int, threshold: float) -> np.ndarray:
    """Commodity Channel Index. Long when CCI > threshold."""
    h = np.asarray(high, float); l = np.asarray(low, float); c = np.asarray(close, float)
    tp = (h + l + c) / 3.0
    s = pd.Series(tp)
    sma = s.rolling(period, min_periods=period).mean()
    md = s.rolling(period, min_periods=period).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cci = (tp - sma.to_numpy()) / (0.015 * md.to_numpy())
    held = (cci > threshold).astype(np.int8)
    held[~np.isfinite(cci)] = 0
    return held


def willr_held(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               period: int, entry_level: float, exit_level: float) -> np.ndarray:
    """Williams %R (range -100..0). Hysteresis: enter when %R rises above entry_level
    (momentum), exit when %R falls below exit_level (oversold collapse)."""
    h = pd.Series(high, dtype=float); l = pd.Series(low, dtype=float)
    c = np.asarray(close, float)
    hh = h.rolling(period, min_periods=period).max().to_numpy()
    ll = l.rolling(period, min_periods=period).min().to_numpy()
    rng = hh - ll
    with np.errstate(divide="ignore", invalid="ignore"):
        wr = np.where(rng > 1e-12, (hh - c) / rng * -100.0, np.nan)
    n = len(c)
    held = np.zeros(n, dtype=np.int8)
    state = 0
    for i in range(period, n):
        if not np.isfinite(wr[i]):
            held[i] = state
            continue
        if state == 0 and wr[i] > entry_level:
            state = 1
        elif state == 1 and wr[i] < exit_level:
            state = 0
        held[i] = state
    return held


def psar_held(high: np.ndarray, low: np.ndarray, close: np.ndarray,
              af_step: float, af_max: float) -> np.ndarray:
    """Parabolic SAR trailing stop. Long when price above SAR (uptrend). Sequential."""
    h = np.asarray(high, float); l = np.asarray(low, float); c = np.asarray(close, float)
    n = len(c)
    held = np.zeros(n, dtype=np.int8)
    if n < 3:
        return held
    # init
    up = True
    af = af_step
    ep = h[0]
    sar = l[0]
    for i in range(1, n):
        prev_sar = sar
        sar = prev_sar + af * (ep - prev_sar)
        if up:
            sar = min(sar, l[i-1], l[i-2] if i >= 2 else l[i-1])
            if h[i] > ep:
                ep = h[i]; af = min(af + af_step, af_max)
            if l[i] < sar:  # flip to down
                up = False; sar = ep; ep = l[i]; af = af_step
        else:
            sar = max(sar, h[i-1], h[i-2] if i >= 2 else h[i-1])
            if l[i] < ep:
                ep = l[i]; af = min(af + af_step, af_max)
            if h[i] > sar:  # flip to up
                up = True; sar = ep; ep = h[i]; af = af_step
        held[i] = 1 if up else 0
    return held


def aroon_held(high: np.ndarray, low: np.ndarray, period: int,
               strength: float) -> np.ndarray:
    """Aroon. Long when AroonUp > AroonDown AND AroonUp > strength (0..100)."""
    h = pd.Series(high, dtype=float)
    l = pd.Series(low, dtype=float)
    # periods since highest high / lowest low over the trailing window (incl current)
    def _since_max(x):
        return (len(x) - 1 - np.argmax(x))
    def _since_min(x):
        return (len(x) - 1 - np.argmin(x))
    win = period + 1
    since_hh = h.rolling(win, min_periods=win).apply(_since_max, raw=True).to_numpy()
    since_ll = l.rolling(win, min_periods=win).apply(_since_min, raw=True).to_numpy()
    aroon_up = 100.0 * (period - since_hh) / period
    aroon_dn = 100.0 * (period - since_ll) / period
    held = ((aroon_up > aroon_dn) & (aroon_up > strength)).astype(np.int8)
    held[~np.isfinite(since_hh)] = 0
    return held


def vortex_held(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                period: int) -> np.ndarray:
    """Vortex Indicator. Long when VI+ > VI- (uptrend dominates)."""
    h = np.asarray(high, float); l = np.asarray(low, float); c = np.asarray(close, float)
    prev_c = np.roll(c, 1); prev_c[0] = c[0]
    prev_l = np.roll(l, 1); prev_l[0] = l[0]
    prev_h = np.roll(h, 1); prev_h[0] = h[0]
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    vm_plus = np.abs(h - prev_l)
    vm_minus = np.abs(l - prev_h)
    tr_s = pd.Series(tr).rolling(period, min_periods=period).sum().to_numpy()
    vmp_s = pd.Series(vm_plus).rolling(period, min_periods=period).sum().to_numpy()
    vmm_s = pd.Series(vm_minus).rolling(period, min_periods=period).sum().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        vi_plus = np.where(tr_s > 1e-12, vmp_s / tr_s, np.nan)
        vi_minus = np.where(tr_s > 1e-12, vmm_s / tr_s, np.nan)
    held = (vi_plus > vi_minus).astype(np.int8)
    held[~np.isfinite(vi_plus)] = 0
    return held


def keltner_held(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                 period: int, mult: float, breakout: bool = False) -> np.ndarray:
    """Keltner Channel: EMA(close) +/- mult*ATR. Long when close > mid (trend) or
    close > upper (breakout=True)."""
    h = np.asarray(high, float); l = np.asarray(low, float); c = np.asarray(close, float)
    n = len(c)
    prev_c = np.roll(c, 1); prev_c[0] = c[0]
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    atr = pd.Series(tr).rolling(period, min_periods=period).mean().to_numpy()
    mid = _ema(c, period)
    upper = mid + mult * atr
    ref = upper if breakout else mid
    held = (c > ref).astype(np.int8)
    held[~np.isfinite(atr)] = 0
    held[:period] = 0
    return held


# =============================================================================
# CONFIG GRIDS (ensemble = selectivity via majority vote)
# =============================================================================

OBV_CONFIGS = [(10,), (20,), (30,), (50,), (14,), (40,)]
VWAP_CONFIGS = [(20,), (30,), (50,), (100,), (14,), (40,)]
CMF_CONFIGS = [(20, 0.0), (20, 0.05), (20, 0.1), (30, 0.0), (30, 0.05), (14, 0.0), (14, 0.1)]
MFI_CONFIGS = [(7, 55, 40), (7, 60, 45), (14, 55, 40), (14, 60, 45),
               (14, 65, 50), (21, 55, 40), (21, 60, 45), (10, 50, 35)]
FLOW_CONFIGS = [(7, 0.0), (7, 0.02), (14, 0.0), (14, 0.02), (14, 0.05),
                (21, 0.0), (21, 0.02), (30, 0.0)]

DONCHIAN_CONFIGS = [(10,), (20,), (30,), (40,), (55,), (15,)]
CCI_CONFIGS = [(14, 0), (14, 100), (20, 0), (20, 100), (20, -100),
               (30, 0), (30, 100), (10, 0)]
WILLR_CONFIGS = [(14, -50, -80), (14, -40, -80), (14, -50, -90),
                 (21, -50, -80), (21, -40, -80), (10, -50, -80), (28, -50, -85)]
PSAR_CONFIGS = [(0.02, 0.2), (0.02, 0.1), (0.01, 0.2), (0.03, 0.2), (0.015, 0.15)]
AROON_CONFIGS = [(14, 50), (14, 70), (25, 50), (25, 70), (20, 60), (10, 50), (25, 80)]
VORTEX_CONFIGS = [(14,), (21,), (30,), (10,), (40,)]
KELTNER_CONFIGS = [(20, 1.5, False), (20, 2.0, False), (20, 1.5, True),
                   (20, 2.0, True), (14, 1.5, False), (30, 2.0, False), (40, 2.0, True)]


# Which families need which inputs
VOLUME_FAMILIES = {"OBV", "VWAP", "CMF", "MFI", "FLOW"}
MISC_FAMILIES = {"DONCHIAN", "CCI", "WILLR", "PSAR", "AROON", "VORTEX", "KELTNER"}
ALL_CONFIGS = {
    "OBV": OBV_CONFIGS, "VWAP": VWAP_CONFIGS, "CMF": CMF_CONFIGS,
    "MFI": MFI_CONFIGS, "FLOW": FLOW_CONFIGS,
    "DONCHIAN": DONCHIAN_CONFIGS, "CCI": CCI_CONFIGS, "WILLR": WILLR_CONFIGS,
    "PSAR": PSAR_CONFIGS, "AROON": AROON_CONFIGS, "VORTEX": VORTEX_CONFIGS,
    "KELTNER": KELTNER_CONFIGS,
}


def single_held(family: str, cfg, o, h, l, c, volume=None, buy_vol=None, sell_vol=None):
    """Dispatch one config to its held function."""
    if family == "OBV":
        return obv_held(c, volume, *cfg)
    if family == "VWAP":
        return vwap_held(h, l, c, volume, *cfg)
    if family == "CMF":
        return cmf_held(h, l, c, volume, *cfg)
    if family == "MFI":
        return mfi_held(h, l, c, volume, *cfg)
    if family == "FLOW":
        return flow_held(buy_vol, sell_vol, *cfg)
    if family == "DONCHIAN":
        return donchian_held(h, l, c, *cfg)
    if family == "CCI":
        return cci_held(h, l, c, *cfg)
    if family == "WILLR":
        return willr_held(h, l, c, *cfg)
    if family == "PSAR":
        return psar_held(h, l, c, *cfg)
    if family == "AROON":
        return aroon_held(h, l, *cfg)
    if family == "VORTEX":
        return vortex_held(h, l, c, *cfg)
    if family == "KELTNER":
        return keltner_held(h, l, c, *cfg)
    raise ValueError(f"unknown family {family}")


def ensemble_held(family: str, configs, o, h, l, c, vote_threshold=0.5,
                  volume=None, buy_vol=None, sell_vol=None) -> np.ndarray:
    """Majority-vote ensemble across configs -> int8 held."""
    n = len(c)
    votes = np.zeros(n, dtype=float)
    n_valid = 0
    for cfg in configs:
        try:
            held = single_held(family, cfg, o, h, l, c, volume, buy_vol, sell_vol)
            votes += held.astype(float)
            n_valid += 1
        except Exception:
            continue
    if n_valid == 0:
        return np.zeros(n, dtype=np.int8)
    return ((votes / n_valid) > vote_threshold).astype(np.int8)
