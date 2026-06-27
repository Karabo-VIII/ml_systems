"""SOTA TI Engine Framework — the proper way to build non-MA indicator engines.

Mirrors the MA family's proven architecture:
1. CONFIG ENSEMBLE: multiple parameter sets, majority-vote for selectivity
2. REGIME GATE: V1 BTC SMA100 (Layer 1, verified)
3. MAKER COST: 0.0006 RT on every turnover event
4. DECOUPLED: lookback (signal horizon) ≠ hold (trade horizon)
5. PER-SLICE: 7d non-overlapping evaluation

The critical lesson from the MA family: a SINGLE config at any TI will be
either too permissive (95% deployed = always-long) or too restrictive (too few
signals = noisy). The ENSEMBLE solves this by requiring AGREEMENT across multiple
parameterizations — only when most configs agree does the engine enter.

This is why the MA family gets 35% deployment with 59% win: the 100+ band
configs rarely ALL agree, but when they do, it's a strong signal.
"""
from __future__ import annotations

import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CRYPTO = ROOT.parent

import strat.fleet_lab as fl
from strat.ma_wealth_driver import _load_asset, _u100_symbols, MAKER_RT
from strat.ma_type_upgrade import _ema

BARS_PER_DAY = fl.BARS_PER_DAY
OUTDIR = CRYPTO / "runs" / "engines" / "V3_ti_families"


# =============================================================================
# INDICATOR COMPUTATION (causal, vectorized)
# =============================================================================

def compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI via Wilder's smoothing. Causal."""
    n = len(close)
    delta = np.diff(close, prepend=close[0])
    gain = np.maximum(delta, 0.0)
    loss = np.maximum(-delta, 0.0)
    
    alpha = 1.0 / period
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    if n > period:
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        for i in range(period + 1, n):
            avg_gain[i] = avg_gain[i-1] * (1 - alpha) + gain[i] * alpha
            avg_loss[i] = avg_loss[i-1] * (1 - alpha) + loss[i] * alpha
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = np.where(avg_loss > 1e-10, avg_gain / avg_loss, 100.0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = 50.0
    return rsi


def compute_macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line, signal line, histogram. Causal."""
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger(close: np.ndarray, period: int = 20, std_mult: float = 2.0):
    """Bollinger Bands. Causal (vectorized rolling)."""
    n = len(close)
    # Use pandas for efficient rolling computation
    s = pd.Series(close)
    middle = s.rolling(period, min_periods=period).mean().values
    std = s.rolling(period, min_periods=period).std(ddof=0).values
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    return middle, upper, lower


def compute_stochastic(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                       k_period: int = 14, d_period: int = 3) -> tuple:
    """Stochastic %K and %D. Causal (vectorized)."""
    n = len(close)
    k = np.full(n, 50.0)
    
    # Vectorized rolling high/low
    h_series = pd.Series(high)
    l_series = pd.Series(low)
    rolling_high = h_series.rolling(k_period, min_periods=k_period).max().values
    rolling_low = l_series.rolling(k_period, min_periods=k_period).min().values
    
    denom = rolling_high - rolling_low
    valid = denom > 1e-10
    k[valid] = ((close[valid] - rolling_low[valid]) / denom[valid]) * 100.0
    
    # %D = SMA of %K
    d = pd.Series(k).rolling(d_period, min_periods=d_period).mean().values
    d = np.nan_to_num(d, nan=50.0)
    
    return k, d


# =============================================================================
# ENSEMBLE CONFIG GRIDS (multiple params → majority vote = selectivity)
# =============================================================================

RSI_CONFIGS = [
    # (period, long_threshold, exit_threshold)
    # The signal: RSI > long_threshold = bullish. Exit when RSI < exit_threshold.
    (7, 55, 40), (7, 60, 45), (7, 65, 50),
    (10, 55, 40), (10, 60, 45), (10, 65, 50),
    (14, 55, 40), (14, 60, 45), (14, 65, 50),
    (21, 55, 40), (21, 60, 45), (21, 65, 50),
    (7, 50, 35), (14, 50, 35), (21, 50, 35),
    (7, 70, 55), (14, 70, 55), (21, 70, 55),
]

MACD_CONFIGS = [
    # (fast, slow, signal)
    (8, 21, 5), (8, 21, 9), (8, 26, 5),
    (12, 26, 9), (12, 26, 5), (12, 21, 9),
    (5, 13, 3), (5, 13, 8), (5, 21, 9),
    (12, 36, 9), (8, 17, 9), (10, 22, 7),
    (6, 19, 5), (9, 26, 7), (15, 30, 9),
]

BOLLINGER_CONFIGS = [
    # (period, std_mult, entry_type)
    # entry_type: "breakout" = close > upper, "squeeze" = bandwidth < threshold
    (15, 1.5, "breakout"), (15, 2.0, "breakout"), (15, 2.5, "breakout"),
    (20, 1.5, "breakout"), (20, 2.0, "breakout"), (20, 2.5, "breakout"),
    (25, 1.5, "breakout"), (25, 2.0, "breakout"), (25, 2.5, "breakout"),
    (30, 2.0, "breakout"), (10, 1.5, "breakout"), (10, 2.0, "breakout"),
]

STOCHASTIC_CONFIGS = [
    # (k_period, d_period, overbought, oversold)
    (5, 3, 80, 20), (5, 3, 70, 30), (5, 3, 75, 25),
    (9, 3, 80, 20), (9, 3, 70, 30), (9, 3, 75, 25),
    (14, 3, 80, 20), (14, 3, 70, 30), (14, 3, 75, 25),
    (14, 5, 80, 20), (21, 3, 80, 20), (21, 5, 70, 30),
]

# Family 5: ADX (trend strength) — long when ADX > threshold (strong trend)
# Uses the chimera's norm_efficiency as a proxy for ADX/trend-strength
ADX_CONFIGS = [
    # (lookback_period, adx_threshold) — proxy via price efficiency ratio
    (7, 0.3), (7, 0.4), (7, 0.5),
    (14, 0.3), (14, 0.4), (14, 0.5),
    (21, 0.3), (21, 0.4), (21, 0.5),
    (10, 0.35), (10, 0.45), (10, 0.55),
]

# Family 6: Supertrend (ATR-based trend direction)
SUPERTREND_CONFIGS = [
    # (atr_period, multiplier) — long when price > supertrend line
    (7, 1.5), (7, 2.0), (7, 2.5), (7, 3.0),
    (10, 1.5), (10, 2.0), (10, 2.5), (10, 3.0),
    (14, 2.0), (14, 2.5), (14, 3.0),
    (20, 2.0), (20, 3.0),
]

# Family 7: Funding-rate extreme (crypto-native, uses chimera norm_funding)
FUNDING_CONFIGS = [
    # (lookback, z_threshold_long, z_threshold_exit)
    # Long when funding z < -threshold (contrarian: negative funding = short crowded)
    (7, -1.0, 0.0), (7, -1.5, 0.0), (7, -2.0, 0.5),
    (14, -1.0, 0.0), (14, -1.5, 0.0), (14, -2.0, 0.5),
    (21, -1.0, 0.0), (21, -1.5, 0.0), (21, -2.0, 0.5),
    (7, -0.5, 0.5), (14, -0.5, 0.5), (21, -0.5, 0.5),
]

# Family 8: OI-Price Divergence (crypto-native)
OI_DIVERGENCE_CONFIGS = [
    # (lookback, divergence_threshold)
    # Long when OI rising but price flat/down (accumulation)
    (7, 0.3), (7, 0.5), (7, 0.7),
    (14, 0.3), (14, 0.5), (14, 0.7),
    (21, 0.3), (21, 0.5), (21, 0.7),
    (10, 0.4), (10, 0.6), (10, 0.8),
]


# =============================================================================
# PER-CONFIG HELD STATE (one binary signal per config)
# =============================================================================

def rsi_single_held(close: np.ndarray, period: int, long_thr: float, exit_thr: float) -> np.ndarray:
    """One RSI config → held state (1 when RSI > long_thr, 0 when < exit_thr, hysteresis)."""
    rsi = compute_rsi(close, period)
    n = len(close)
    held = np.zeros(n, dtype=np.int8)
    state = 0
    for i in range(period, n):
        if state == 0 and rsi[i] > long_thr:
            state = 1
        elif state == 1 and rsi[i] < exit_thr:
            state = 0
        held[i] = state
    return held


def macd_single_held(close: np.ndarray, fast: int, slow: int, signal: int) -> np.ndarray:
    """One MACD config → held state (1 when MACD > signal line)."""
    macd_line, signal_line, _ = compute_macd(close, fast, slow, signal)
    n = len(close)
    held = np.zeros(n, dtype=np.int8)
    warmup = slow + signal
    for i in range(warmup, n):
        if macd_line[i] > signal_line[i]:
            held[i] = 1
    return held


def bollinger_single_held(close: np.ndarray, period: int, std_mult: float, 
                          entry_type: str) -> np.ndarray:
    """One Bollinger config → held state."""
    middle, upper, lower = compute_bollinger(close, period, std_mult)
    n = len(close)
    held = np.zeros(n, dtype=np.int8)
    state = 0
    for i in range(period, n):
        if np.isnan(upper[i]):
            continue
        if entry_type == "breakout":
            # Long when price above middle band (trending up)
            # Exit when price below middle band
            if state == 0 and close[i] > middle[i]:
                state = 1
            elif state == 1 and close[i] < middle[i]:
                state = 0
        held[i] = state
    return held


def stochastic_single_held(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                           k_period: int, d_period: int, 
                           overbought: float, oversold: float) -> np.ndarray:
    """One Stochastic config → held state (bullish cross: %K > %D above oversold)."""
    k, d = compute_stochastic(close, high, low, k_period, d_period)
    n = len(close)
    held = np.zeros(n, dtype=np.int8)
    state = 0
    warmup = k_period + d_period
    for i in range(warmup, n):
        if state == 0 and k[i] > d[i] and k[i] > oversold:
            state = 1
        elif state == 1 and (k[i] < d[i] or k[i] > overbought):
            state = 0
        held[i] = state
    return held


def adx_single_held(close: np.ndarray, lookback: int, threshold: float) -> np.ndarray:
    """ADX proxy via efficiency ratio: |net move| / sum(|bar moves|). 
    Long when efficiency > threshold (price trending efficiently)."""
    n = len(close)
    held = np.zeros(n, dtype=np.int8)
    
    for i in range(lookback, n):
        window = close[i-lookback:i+1]
        net_move = abs(window[-1] - window[0])
        sum_moves = np.sum(np.abs(np.diff(window)))
        if sum_moves > 1e-10:
            efficiency = net_move / sum_moves
        else:
            efficiency = 0
        # Also check direction: only long if trending UP
        if efficiency > threshold and window[-1] > window[0]:
            held[i] = 1
    return held


def supertrend_single_held(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                           atr_period: int, multiplier: float) -> np.ndarray:
    """Supertrend: ATR-based trend direction. Long when price > supertrend line."""
    n = len(close)
    held = np.zeros(n, dtype=np.int8)
    
    # Compute ATR
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).rolling(atr_period, min_periods=atr_period).mean().values
    
    # Supertrend
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend = np.zeros(n)
    direction = np.ones(n, dtype=int)  # 1 = up, -1 = down
    
    for i in range(atr_period, n):
        mid = (high[i] + low[i]) / 2
        upper_band[i] = mid + multiplier * atr[i]
        lower_band[i] = mid - multiplier * atr[i]
        
        if i > atr_period:
            # Band continuity
            if lower_band[i] < lower_band[i-1] and close[i-1] > lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if upper_band[i] > upper_band[i-1] and close[i-1] < upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
            
            # Direction
            if close[i] > upper_band[i-1]:
                direction[i] = 1
            elif close[i] < lower_band[i-1]:
                direction[i] = -1
            else:
                direction[i] = direction[i-1]
        
        held[i] = 1 if direction[i] == 1 else 0
    
    return held


def funding_single_held(close: np.ndarray, lookback: int, 
                        z_long: float, z_exit: float) -> np.ndarray:
    """Funding-rate proxy: use price momentum as funding proxy.
    In reality this would use norm_funding from chimera.
    Here we proxy with: negative momentum = "shorts are crowded" = contrarian long."""
    n = len(close)
    held = np.zeros(n, dtype=np.int8)
    
    # Proxy: rolling z-score of returns (negative = oversold = contrarian long)
    rets = np.zeros(n)
    rets[1:] = np.diff(close) / close[:-1]
    
    state = 0
    for i in range(lookback, n):
        window = rets[i-lookback:i]
        if np.std(window) > 1e-10:
            z = (rets[i] - np.mean(window)) / np.std(window)
        else:
            z = 0
        
        if state == 0 and z < z_long:  # negative z = contrarian long signal
            state = 1
        elif state == 1 and z > z_exit:  # z recovered = exit
            state = 0
        held[i] = state
    return held


def oi_divergence_single_held(close: np.ndarray, lookback: int, 
                              threshold: float) -> np.ndarray:
    """OI-Price divergence proxy: long when price is range-bound but trending up slightly.
    In reality this would use norm_oi_price_divergence from chimera.
    Proxy: price efficiency low (range-bound) + slight upward bias."""
    n = len(close)
    held = np.zeros(n, dtype=np.int8)
    
    for i in range(lookback, n):
        window = close[i-lookback:i+1]
        net_move = window[-1] - window[0]
        sum_moves = np.sum(np.abs(np.diff(window)))
        
        if sum_moves > 1e-10:
            efficiency = abs(net_move) / sum_moves
            # Low efficiency = range-bound, slight positive = accumulation
            if efficiency < threshold and net_move > 0:
                held[i] = 1
    
    return held


# =============================================================================
# ENSEMBLE PROVIDER (majority vote across configs = selectivity)
# =============================================================================

def make_ensemble_provider(family: str, configs: list, vote_threshold: float = 0.5):
    """Build an ensemble provider: majority of configs must agree to be long.
    
    Returns: provider(sym, o, h, l, c) -> int8 held array
    Same interface as the MA make_hold_provider.
    """
    _vote_thr = vote_threshold  # capture in closure
    
    def provider(sym, o, h, l, c):
        close = np.asarray(c, dtype=float)
        high = np.asarray(h, dtype=float)
        low = np.asarray(l, dtype=float)
        n = len(close)
        votes = np.zeros(n, dtype=float)
        n_valid = 0
        
        for cfg in configs:
            try:
                if family == "RSI":
                    period, long_thr, exit_thr = cfg
                    held = rsi_single_held(close, period, long_thr, exit_thr)
                elif family == "MACD":
                    fast, slow, signal = cfg
                    held = macd_single_held(close, fast, slow, signal)
                elif family == "BOLLINGER":
                    period, std_mult, entry_type = cfg
                    held = bollinger_single_held(close, period, std_mult, entry_type)
                elif family == "STOCHASTIC":
                    k_period, d_period, overbought, oversold = cfg
                    held = stochastic_single_held(close, high, low, 
                                                  k_period, d_period, overbought, oversold)
                elif family == "ADX":
                    lookback, thr = cfg
                    held = adx_single_held(close, lookback, thr)
                elif family == "SUPERTREND":
                    atr_period, multiplier = cfg
                    held = supertrend_single_held(close, high, low, atr_period, multiplier)
                elif family == "FUNDING":
                    lookback, z_long, z_exit = cfg
                    held = funding_single_held(close, lookback, z_long, z_exit)
                elif family == "OI_DIVERGENCE":
                    lookback, thr = cfg
                    held = oi_divergence_single_held(close, lookback, thr)
                else:
                    continue
                votes += held.astype(float)
                n_valid += 1
            except Exception:
                continue
        
        if n_valid == 0:
            return np.zeros(n, dtype=np.int8)
        
        # Majority vote: held when > threshold fraction of configs agree
        frac = votes / n_valid
        return (frac > _vote_thr).astype(np.int8)
    
    return provider


def make_stochastic_ensemble_provider(configs: list, threshold: float = 0.5):
    """Stochastic needs OHLC, not just close."""
    def provider(sym, o, h, l, c):
        close = np.asarray(c, dtype=float)
        high = np.asarray(h, dtype=float)
        low = np.asarray(l, dtype=float)
        n = len(close)
        votes = np.zeros(n, dtype=float)
        n_valid = 0
        
        for cfg in configs:
            try:
                k_period, d_period, overbought, oversold = cfg
                held = stochastic_single_held(close, high, low, k_period, d_period, overbought, oversold)
                votes += held.astype(float)
                n_valid += 1
            except Exception:
                continue
        
        if n_valid == 0:
            return np.zeros(n, dtype=np.int8)
        frac = votes / n_valid
        return (frac > threshold).astype(np.int8)
    
    return provider


# =============================================================================
# ENGINE CELL RUNNER (with regime gate + cost)
# =============================================================================

def run_engine_cell(family: str, tf: str, provider, slice_days: int = 7,
                    use_regime_gate: bool = True) -> dict:
    """Run one TI engine cell with full SOTA accounting."""
    t0 = time.time()
    bpd = BARS_PER_DAY[tf]
    slice_bars = slice_days * bpd
    
    wall_ms = int(pd.Timestamp(fl.DEV_END).value // 10**6)
    symbols = _u100_symbols()
    
    # Regime gate (V1 BTC SMA100)
    btc_regime = None
    if use_regime_gate:
        btc_rec = _load_asset("BTCUSDT", "1d", wall_ms)
        if btc_rec:
            btc_dates, _, _, _, btc_close = btc_rec
            sma100 = pd.Series(btc_close, index=btc_dates).rolling(100).mean()
            regime_s = pd.Series(0, index=btc_dates, dtype=int)
            state = 0
            for i in range(100, len(btc_dates)):
                s = sma100.iloc[i]
                c = btc_close[i]
                if np.isnan(s):
                    continue
                if state == 0 and c > s * 1.02:
                    state = 1
                elif state == 1 and c < s * 0.98:
                    state = 0
                regime_s.iloc[i] = state
            # DE-LEAK (V1.1 align_htf_on bug class): regime[D] is the day-D 1d CLOSE,
            # unknown until ~23:59. A bar on day D must read the last CLOSED 1d bar =
            # regime[D-1]. Shift forward one 1d bar before the per-bar prior[-1] broadcast.
            # See runs/engines/AUDIT_ORACLE_UPGRADE/leakcheck.json.
            btc_regime = regime_s.shift(1).fillna(0).astype(int)
    
    # Load assets and compute holds
    close_dict = {}
    held_dict = {}
    
    for sym in symbols:
        rec = _load_asset(sym, tf, wall_ms)
        if rec is None:
            continue
        dates, o, h, l, c = rec
        held = provider(sym, o, h, l, c)
        close_dict[sym] = pd.Series(c, index=dates)
        held_dict[sym] = pd.Series(held, index=dates)
    
    n_assets = len(close_dict)
    if n_assets < 20:
        return {"status": "INSUFFICIENT_ASSETS", "n_assets": n_assets}
    
    # Build returns with COST and REGIME GATE
    all_dates = sorted(set().union(*[s.index.tolist() for s in close_dict.values()]))
    
    engine_rets = []
    al_rets = []
    
    for i in range(1, len(all_dates)):
        dp = all_dates[i-1]
        dc = all_dates[i]
        
        # Regime check
        if btc_regime is not None:
            d_1d = pd.Timestamp(dp).normalize()
            prior = btc_regime.index[btc_regime.index <= d_1d]
            regime_val = int(btc_regime[prior[-1]]) if len(prior) > 0 else 0
            if regime_val == 0:
                engine_rets.append(0.0)  # CASH in bear
                # Still compute AL for comparison
                all_r = []
                for sym in close_dict:
                    if dp in close_dict[sym].index and dc in close_dict[sym].index:
                        cp = close_dict[sym][dp]
                        cc = close_dict[sym][dc]
                        if cp > 0:
                            all_r.append((cc / cp) - 1.0)
                al_rets.append(float(np.mean(all_r)) if all_r else 0.0)
                continue
        
        eng_r = []
        all_r = []
        n_transitions = 0
        
        for sym in close_dict:
            if dp not in close_dict[sym].index or dc not in close_dict[sym].index:
                continue
            cp = close_dict[sym][dp]
            cc = close_dict[sym][dc]
            if cp <= 0:
                continue
            
            ret = (cc / cp) - 1.0
            all_r.append(ret)
            
            if dp in held_dict[sym].index and held_dict[sym][dp] == 1:
                eng_r.append(ret)
                # Check for transitions (cost)
                if i > 1:
                    dp_prev = all_dates[i-2]
                    if dp_prev in held_dict[sym].index:
                        if held_dict[sym][dp_prev] != held_dict[sym][dp]:
                            n_transitions += 1
        
        if eng_r:
            # EW return minus cost on transitions
            ew_ret = float(np.mean(eng_r))
            cost = n_transitions * (MAKER_RT / 2.0) / max(1, len(eng_r))
            engine_rets.append(ew_ret - cost)
        else:
            engine_rets.append(0.0)
        
        al_rets.append(float(np.mean(all_r)) if all_r else 0.0)
    
    eng = np.array(engine_rets)
    al = np.array(al_rets)
    
    # Per-slice metrics
    n_bars = len(eng)
    eng_slices = []
    al_slices = []
    i = 0
    while i + slice_bars <= n_bars:
        eng_slices.append(float(np.prod(1 + eng[i:i+slice_bars]) - 1) * 100)
        al_slices.append(float(np.prod(1 + al[i:i+slice_bars]) - 1) * 100)
        i += slice_bars
    
    n_slices = len(eng_slices)
    pct_prof = sum(1 for s in eng_slices if s > 0) / n_slices if n_slices > 0 else 0
    median_s = float(np.median(eng_slices)) if eng_slices else 0
    compound = float(np.prod(1 + eng) - 1) * 100
    al_compound = float(np.prod(1 + al) - 1) * 100
    
    vs_al = [e - a for e, a in zip(eng_slices, al_slices)]
    beats_al = sum(1 for v in vs_al if v > 0) / len(vs_al) if vs_al else 0
    
    cum = np.cumprod(1 + eng)
    maxdd = float(np.min((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum))) * 100 if len(cum) > 1 else 0
    deploy = float(np.mean(eng != 0))
    
    elapsed = time.time() - t0
    
    return {
        "status": "OK",
        "family": family, "tf": tf,
        "n_assets": n_assets, "n_slices": n_slices,
        "pct_profitable": round(pct_prof, 3),
        "median_slice_pct": round(median_s, 2),
        "compound_pct": round(compound, 1),
        "al_compound_pct": round(al_compound, 1),
        "vs_al_pp": round(compound - al_compound, 1),
        "beats_al_pct": round(beats_al, 3),
        "maxdd_pct": round(maxdd, 1),
        "deployment_rate": round(deploy, 3),
        "elapsed_s": round(elapsed, 1),
    }


# =============================================================================
# SWEEP (per-family, across TFs)
# =============================================================================

def run_family_sweep(family: str, configs: list, threshold: float = 0.5):
    """Run a full TI family sweep: ensemble at all TFs with regime gate."""
    print(f"\n{'='*80}")
    print(f"  {family} ENGINE FAMILY — SOTA BUILD (ensemble + regime gate + cost)")
    print(f"{'='*80}")
    print(f"  Configs: {len(configs)} | Threshold: >{threshold:.0%} must agree")
    
    if family == "STOCHASTIC":
        provider = make_stochastic_ensemble_provider(configs, threshold)
    else:
        provider = make_ensemble_provider(family, configs, vote_threshold=threshold)
    
    tfs = ["1d", "4h", "2h", "1h", "30m"]
    results = []
    
    print(f"\n  {'TF':<6} {'%Prof':>6} {'Median':>8} {'Compound':>10} {'vsAL':>8} {'MaxDD':>7} {'Deploy':>7}")
    print(f"  {'-'*55}")
    
    for tf in tfs:
        r = run_engine_cell(family, tf, provider)
        results.append(r)
        if r["status"] == "OK":
            print(f"  {tf:<6} {r['pct_profitable']:>5.0%} {r['median_slice_pct']:>+7.2f}% "
                  f"{r['compound_pct']:>9.1f}% {r['vs_al_pp']:>+7.1f} {r['maxdd_pct']:>6.1f}% "
                  f"{r['deployment_rate']:>6.1%}")
        else:
            print(f"  {tf:<6} {r.get('status', 'ERROR')}")
    
    # Save
    outdir = OUTDIR / family
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(outdir / f"{family.lower()}_sota_sweep_{ts}.json", "w") as f:
        json.dump({"family": family, "n_configs": len(configs), "threshold": threshold,
                   "regime_gate": True, "cost_rt": MAKER_RT,
                   "results": results, "timestamp": ts}, f, indent=2)
    
    return results


def main():
    """Run all TI families."""
    all_results = {}
    
    all_results["RSI"] = run_family_sweep("RSI", RSI_CONFIGS, threshold=0.5)
    all_results["MACD"] = run_family_sweep("MACD", MACD_CONFIGS, threshold=0.5)
    all_results["BOLLINGER"] = run_family_sweep("BOLLINGER", BOLLINGER_CONFIGS, threshold=0.5)
    all_results["STOCHASTIC"] = run_family_sweep("STOCHASTIC", STOCHASTIC_CONFIGS, threshold=0.5)
    all_results["ADX"] = run_family_sweep("ADX", ADX_CONFIGS, threshold=0.5)
    all_results["SUPERTREND"] = run_family_sweep("SUPERTREND", SUPERTREND_CONFIGS, threshold=0.5)
    all_results["FUNDING"] = run_family_sweep("FUNDING", FUNDING_CONFIGS, threshold=0.5)
    all_results["OI_DIVERGENCE"] = run_family_sweep("OI_DIVERGENCE", OI_DIVERGENCE_CONFIGS, threshold=0.5)
    
    # Summary
    print(f"\n\n{'='*80}")
    print("CROSS-FAMILY COMPARISON (4h, the proven sweet spot)")
    print(f"{'='*80}")
    print(f"  {'Family':<15} {'%Prof':>6} {'Median':>8} {'vsAL':>10} {'Deploy':>7} {'MaxDD':>7}")
    print(f"  {'-'*55}")
    
    for fam, results in all_results.items():
        r4h = next((r for r in results if r.get("status") == "OK" and r.get("tf") == "4h"), None)
        if r4h:
            print(f"  {fam:<15} {r4h['pct_profitable']:>5.0%} {r4h['median_slice_pct']:>+7.2f}% "
                  f"{r4h['vs_al_pp']:>+9.1f} {r4h['deployment_rate']:>6.1%} {r4h['maxdd_pct']:>6.1f}%")
    
    print(f"\n  Reference: MA family at 4h = 59% prof, +4.03% median, 35% deploy, -38.8% MaxDD (OOS)")


if __name__ == "__main__":
    main()
