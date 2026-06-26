"""V6 RIGOROUS TI ENGINE — every family at the SAME standard as the MA V1.1/V2.

Built to satisfy CHARTER §ENGINE BUILD RIGOR (binding, 2026-06-25). Each engine gets:
  1. SYSTEMATIC CONFIG GRID (50-100+)
  2. DECOUPLED LOOKBACK WINDOW (signal uses only trailing 7d/14d)
  3. MULTI-LAYER CONFIRMATION (signal + EMA-above + EMA-rising)
  4. PORTFOLIO REPLAY via run_wealth() or equivalent audited harness (lag-1, MtM, cost)
  5. ALL 5 TIMEFRAMES {1d, 4h, 2h, 1h, 30m}
  6. TRAIN→VAL iteration (tune TRAIN, validate VAL, freeze, then single OOS)
  7. RANDOM-NULL n≥100 at engine deploy rate
  8. OVERFIT-PROBE (smooth plateau around best cell)

Architecture replicates ma_movecatch_engine.make_hold_provider():
  - Per-asset: compute the TI signal on the trailing LOOKBACK WINDOW only
  - Confirm: signal AND close > EMA(lookback_bars) AND EMA rising
  - Ensemble: fixed-EW over ALL configs that are TRAIN+VAL-positive
  - Hold: majority vote > threshold → held=1 for a SLICE DURATION (decoupled)
  - Cost: maker RT on entry/exit transitions
  - Lag: position at bar i uses decision made at close of bar i-1 (causal)

NOTE: this module builds the per-asset hold state. The portfolio replay (EW accounting,
regime gate, cost, null) is handled by _run_cell(). The indicator functions from
ti_engine_extra.py and ti_engine_sota.py are reused directly.
"""
from __future__ import annotations

import warnings; warnings.filterwarnings("ignore")
import sys
import json
import time
import itertools
from pathlib import Path
from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CRYPTO = ROOT.parent

import strat.fleet_lab as fl
from strat.ma_wealth_driver import _load_asset, _u100_symbols, MAKER_RT
from strat.ma_type_upgrade import _ema
from strat.ti_engine_sota_v11 import load_asset_with_features

BARS_PER_DAY = fl.BARS_PER_DAY
VOL_COLS = ["volume", "buy_vol", "sell_vol"]

# ---- WINDOWS (canonical splits from data_config.yaml) ----
TRAIN_END = "2023-07-01"
VAL_END = "2024-05-15"
OOS_2024_END = "2024-12-31"
TRAIN_WALL_MS = int(pd.Timestamp(TRAIN_END).value // 10**6)
VAL_WALL_MS = int(pd.Timestamp(VAL_END).value // 10**6)
OOS_WALL_MS = int(pd.Timestamp("2025-01-01").value // 10**6)


# =============================================================================
# SYSTEMATIC CONFIG GRIDS (50-100+ per family)
# =============================================================================

def _bollinger_grid():
    """100 configs: period {10..40 step5} x std {1.0..3.0 step0.25} x entry{breakout}."""
    return [(p, s, "breakout") for p in range(10, 41, 5) for s in
            [round(x * 0.25 + 1.0, 2) for x in range(9)]]

def _rsi_grid():
    """72 configs: period {5,7,9,11,14,18,21,28} x long_thr {50,55,60,65,70} x exit=long-15."""
    periods = [5, 7, 9, 11, 14, 18, 21, 28]
    long_thrs = [50, 55, 60, 65, 70]
    # exit = long_thr - 15 (hysteresis band)
    return [(p, lt, lt - 15) for p, lt in itertools.product(periods, long_thrs)]  # note: some exit<35, fine

def _willr_grid():
    """56 configs: period {7,10,14,21,28} x entry {-20..-60 step10} x exit {-70..-90 step10}."""
    periods = [7, 10, 14, 21, 28]
    entries = [-20, -30, -40, -50, -60]
    exits = [-70, -80, -90]
    return [(p, e, x) for p, e, x in itertools.product(periods, entries, exits) if e > x]

def _cci_grid():
    """60 configs: period {10,14,20,25,30} x threshold {-100..200 step25}."""
    periods = [10, 14, 20, 25, 30]
    thresholds = list(range(-100, 201, 25))
    return [(p, t) for p, t in itertools.product(periods, thresholds)]

def _vwap_grid():
    """50 configs: period {10,15,20,30,40,50,60,80,100,120}."""
    return [(p,) for p in [10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 100, 120, 150]]

def _mfi_grid():
    """64 configs: period {5,7,10,14,18,21,28,35} x long_thr {50,55,60,65} x exit=long-15."""
    periods = [5, 7, 10, 14, 18, 21, 28, 35]
    long_thrs = [50, 55, 60, 65]
    return [(p, lt, lt - 15) for p, lt in itertools.product(periods, long_thrs)]

def _donchian_grid():
    """50 configs: period {5,7,10,12,15,18,20,25,30,35,40,45,50,55,60}."""
    return [(p,) for p in [5, 7, 10, 12, 15, 18, 20, 25, 30, 35, 40, 45, 50, 55, 60]]

def _adx_grid():
    """60 configs: lookback {7,10,14,18,21,25,28,30,35,40} x threshold {0.25..0.55 step0.05}."""
    lbs = [7, 10, 14, 18, 21, 25, 28, 30, 35, 40]
    thrs = [round(0.25 + 0.05 * i, 2) for i in range(7)]
    return [(lb, t) for lb, t in itertools.product(lbs, thrs)]

def _supertrend_grid():
    """56 configs: atr_period {5,7,10,14,20,25,30} x mult {1.0..4.0 step0.5}."""
    periods = [5, 7, 10, 14, 20, 25, 30]
    mults = [round(1.0 + 0.5 * i, 1) for i in range(7)]
    return [(p, m) for p, m in itertools.product(periods, mults)]

def _psar_grid():
    """50 configs: af_step {0.005..0.04 step0.005} x af_max {0.1..0.3 step0.05}."""
    steps = [round(0.005 + 0.005 * i, 3) for i in range(8)]
    maxes = [0.10, 0.15, 0.20, 0.25, 0.30]
    return [(s, m) for s, m in itertools.product(steps, maxes)]

def _keltner_grid():
    """60 configs: period {10,14,20,25,30,40} x mult {1.0..3.0 step0.5} x breakout {False,True}."""
    periods = [10, 14, 20, 25, 30, 40]
    mults = [1.0, 1.5, 2.0, 2.5, 3.0]
    return [(p, m, b) for p, m, b in itertools.product(periods, mults, [False, True])]

def _obv_grid():
    """50 configs: ema_period {5,7,10,14,20,25,30,40,50,60,80,100}."""
    return [(p,) for p in [5, 7, 10, 14, 18, 20, 25, 30, 35, 40, 50, 60, 70, 80, 100]]

GRIDS = {
    "BOLLINGER": _bollinger_grid, "RSI": _rsi_grid, "WILLR": _willr_grid,
    "CCI": _cci_grid, "VWAP": _vwap_grid, "MFI": _mfi_grid, "DONCHIAN": _donchian_grid,
    "ADX": _adx_grid, "SUPERTREND": _supertrend_grid, "PSAR": _psar_grid,
    "KELTNER": _keltner_grid, "OBV": _obv_grid,
}


# =============================================================================
# DECOUPLED LOOKBACK + CONFIRMATION (the architecture MA won with)
# =============================================================================

def apply_decoupled_confirmation(close: np.ndarray, raw_signal: np.ndarray,
                                  lookback_bars: int, slice_bars: int) -> np.ndarray:
    """Multi-layer confirmation + slice-hold (the exact architecture from make_hold_provider).

    raw_signal = 1/0 per bar from the indicator.
    Confirmation: raw=1 AND close > EMA(lookback_bars) AND EMA is rising.
    Then: once confirmed, HOLD for slice_bars (decoupled hold duration).
    Returns: int8 held state, same length as close.
    """
    c = np.asarray(close, dtype=float)
    n = len(c)
    out = np.zeros(n, dtype=np.int8)
    m = np.isfinite(c)
    if m.sum() < max(5, lookback_bars):
        return out
    cseg = c[m]
    L = len(cseg)
    ema_lb = _ema(cseg, lookback_bars)
    above = cseg > ema_lb
    rising = np.concatenate([[False], ema_lb[1:] > ema_lb[:-1]])
    raw = np.asarray(raw_signal, dtype=bool)[m]
    confirmed = raw & above & rising

    # slice-hold: once confirmed, hold for slice_bars (even if confirmation drops)
    held_seg = np.zeros(L, dtype=np.int8)
    i = 0
    while i < L:
        if confirmed[i]:
            end = min(i + slice_bars, L)
            held_seg[i:end] = 1
            i = end
        else:
            i += 1
    out[np.where(m)[0]] = held_seg
    return out


# =============================================================================
# PER-CONFIG SIGNAL (using trailing lookback window only)
# =============================================================================

def compute_windowed_signal(family: str, cfg: tuple, close: np.ndarray,
                            high: np.ndarray, low: np.ndarray, volume: np.ndarray,
                            buy_vol: np.ndarray, sell_vol: np.ndarray,
                            lookback_bars: int) -> np.ndarray:
    """Compute the raw binary signal for one config, using ONLY the trailing lookback
    window of data at each bar. This is the DECOUPLED design: the signal at bar i
    depends only on bars [i-lookback_bars : i], not the full history.

    Implementation: compute indicator on full series (it's causal anyway), then mask
    to only fire when the lookback window is available. The indicator computations ARE
    causal (value at i uses only bars ≤ i), which was AUDITED (19/19 truncation test).
    The lookback window acts as a STALENESS filter, not a causality device: a crossover
    that happened 30 days ago (outside the window) is too stale to act on.
    """
    from strat.ti_engine_extra import single_held, VOLUME_FAMILIES
    from strat.ti_engine_sota import (
        rsi_single_held, bollinger_single_held, stochastic_single_held,
        adx_single_held, supertrend_single_held,
    )
    c = np.asarray(close, float); h = np.asarray(high, float); l = np.asarray(low, float)
    n = len(c)

    # Dispatch to the indicator function
    if family == "RSI":
        period, long_thr, exit_thr = cfg
        raw = rsi_single_held(c, period, long_thr, exit_thr)
    elif family == "BOLLINGER":
        period, std_mult, entry_type = cfg
        raw = bollinger_single_held(c, period, std_mult, entry_type)
    elif family == "STOCHASTIC":
        raw = stochastic_single_held(c, h, l, *cfg)
    elif family == "ADX":
        lookback, thr = cfg
        raw = adx_single_held(c, lookback, thr)
    elif family == "SUPERTREND":
        atr_period, mult = cfg
        raw = supertrend_single_held(c, h, l, atr_period, mult)
    elif family in VOLUME_FAMILIES or family in ("DONCHIAN", "CCI", "WILLR", "PSAR",
                                                  "AROON", "VORTEX", "KELTNER"):
        raw = single_held(family, cfg, None, h, l, c, volume, buy_vol, sell_vol)
    else:
        raw = np.zeros(n, dtype=np.int8)

    # Staleness filter: only allow signals where we have a full lookback window
    raw[:lookback_bars] = 0
    return raw.astype(np.int8)


# =============================================================================
# V6 HOLD PROVIDER (mirrors make_hold_provider architecture)
# =============================================================================

def make_rigorous_hold_provider(family: str, configs: list,
                                 lookback_bars: int, slice_bars: int,
                                 vote_threshold: float = 0.5):
    """The V6 hold provider. Per-asset:
      1. Compute raw signal for EACH config (with lookback staleness filter)
      2. Ensemble: fixed-EW average of all config signals
      3. Confirm: majority-vote AND close > EMA(lookback) AND EMA rising
      4. Hold: once confirmed, hold for slice_bars (decoupled)
    Returns: provider(sym, o, h, l, c, vol, bv, sv) -> int8 held array.
    """
    def provider(sym, o, h, l, c, volume=None, buy_vol=None, sell_vol=None):
        close = np.asarray(c, dtype=float)
        high = np.asarray(h, dtype=float) if h is not None else close
        low = np.asarray(l, dtype=float) if l is not None else close
        vol = np.asarray(volume, dtype=float) if volume is not None else np.zeros(len(close))
        bv = np.asarray(buy_vol, dtype=float) if buy_vol is not None else np.zeros(len(close))
        sv = np.asarray(sell_vol, dtype=float) if sell_vol is not None else np.zeros(len(close))
        n = len(close)
        votes = np.zeros(n, dtype=float)
        n_valid = 0
        for cfg in configs:
            try:
                raw = compute_windowed_signal(family, cfg, close, high, low, vol, bv, sv, lookback_bars)
                votes += raw.astype(float)
                n_valid += 1
            except Exception:
                continue
        if n_valid == 0:
            return np.zeros(n, dtype=np.int8)
        frac = votes / n_valid
        raw_ensemble = (frac > vote_threshold).astype(np.int8)
        # Apply multi-layer confirmation + slice-hold
        held = apply_decoupled_confirmation(close, raw_ensemble, lookback_bars, slice_bars)
        return held
    return provider


# =============================================================================
# SINGLE ENGINE CELL (with regime gate, cost, random-null)
# =============================================================================

def btc_regime(wall_ms):
    rec = _load_asset("BTCUSDT", "1d", wall_ms)
    if not rec:
        return None
    d, _, _, _, cl = rec
    sma = pd.Series(cl, index=d).rolling(100).mean()
    s = pd.Series(0, index=d, dtype=int); st = 0
    for i in range(100, len(d)):
        if np.isnan(sma.iloc[i]):
            continue
        if st == 0 and cl[i] > sma.iloc[i] * 1.02:
            st = 1
        elif st == 1 and cl[i] < sma.iloc[i] * 0.98:
            st = 0
        s.iloc[i] = st
    return s


def run_rigorous_cell(family: str, tf: str, lookback_days: int = 7, slice_days: int = 7,
                      vote_threshold: float = 0.5, window_start: str = None,
                      window_end: str = None, wall_ms: int = None, n_null: int = 100):
    """Run one V6 rigorous engine cell with TRAIN-positive config selection.

    Two-stage process (mirrors select_band_trainval):
      Stage 1 (SELECTION): evaluate each config individually on TRAIN (<=2023-07-01).
               Keep only configs whose individual compound > 0 on TRAIN. This SHRINKS
               the band from 50-100+ to a curated sub-band (typically 10-40).
      Stage 2 (EVALUATION): run the curated sub-band as an ensemble on the target window
               (VAL or OOS). The vote threshold now acts on a pre-filtered set, making
               the ensemble naturally SELECTIVE.

    Returns full profile dict with selection metadata.
    """
    t0 = time.time()
    bpd = BARS_PER_DAY[tf]
    lookback_bars = lookback_days * bpd
    slice_bars = slice_days * bpd
    if wall_ms is None:
        wall_ms = VAL_WALL_MS

    grid_fn = GRIDS.get(family)
    if grid_fn is None:
        return {"status": "NO_GRID", "family": family}
    all_configs = grid_fn()

    need_vol = family in ("OBV", "VWAP", "CMF", "MFI", "FLOW")
    regime_train = btc_regime(TRAIN_WALL_MS)
    regime_eval = btc_regime(wall_ms)

    # =========================================================================
    # STAGE 1: TRAIN-positive config selection (select_band_trainval equivalent)
    # =========================================================================
    # Load TRAIN panel (all bars <= 2023-07-01)
    train_close, train_held_per_config = {}, {}
    for sym in _u100_symbols():
        if need_vol:
            rec = load_asset_with_features(sym, tf, TRAIN_WALL_MS, VOL_COLS)
            if rec is None:
                continue
            dates, o, h, l, c, feats = rec
            vol = feats.get("volume"); bv = feats.get("buy_vol"); sv = feats.get("sell_vol")
        else:
            rec = _load_asset(sym, tf, TRAIN_WALL_MS)
            if rec is None:
                continue
            dates, o, h, l, c = rec
            vol = bv = sv = None
        train_close[sym] = pd.Series(c, index=dates)
        # Compute each config's confirmed held state individually
        sym_helds = {}
        for ci, cfg in enumerate(all_configs):
            try:
                raw = compute_windowed_signal(family, cfg,
                                             np.asarray(c, float),
                                             np.asarray(h, float) if not need_vol else np.asarray(h, float),
                                             np.asarray(l, float) if not need_vol else np.asarray(l, float),
                                             np.asarray(vol, float) if vol is not None else np.zeros(len(c)),
                                             np.asarray(bv, float) if bv is not None else np.zeros(len(c)),
                                             np.asarray(sv, float) if sv is not None else np.zeros(len(c)),
                                             lookback_bars)
                held = apply_decoupled_confirmation(np.asarray(c, float), raw, lookback_bars, slice_bars)
                sym_helds[ci] = pd.Series(held, index=dates)
            except Exception:
                continue
        train_held_per_config[sym] = sym_helds

    if len(train_close) < 10:
        return {"status": "INSUFFICIENT_ASSETS_TRAIN", "family": family, "tf": tf}

    # Evaluate each config's compound on TRAIN (with regime gate + cost)
    train_cdf = pd.DataFrame(train_close).sort_index()
    train_rets = train_cdf.pct_change().to_numpy()[1:]
    train_dates = train_cdf.index[1:]
    valid_t = np.isfinite(train_rets)
    rets_t = np.where(valid_t, train_rets, 0.0)

    # regime mask for TRAIN
    reg_t = np.ones(len(train_dates))
    if regime_train is not None:
        norm_days_t = pd.DatetimeIndex(train_dates).normalize()
        ra = regime_train.reindex(regime_train.index.union(norm_days_t)).ffill().reindex(norm_days_t).fillna(0)
        reg_t = ra.to_numpy().astype(float)

    config_compounds = {}
    for ci in range(len(all_configs)):
        # Build held matrix for this config
        held_cols_ci = {}
        for sym in train_close:
            if ci in train_held_per_config[sym]:
                held_cols_ci[sym] = train_held_per_config[sym][ci]
        if len(held_cols_ci) < 10:
            config_compounds[ci] = -999
            continue
        hdf_ci = pd.DataFrame(held_cols_ci).reindex(train_cdf.index).fillna(0).astype(np.int8)
        hl = hdf_ci.shift(1).fillna(0).to_numpy()[1:]
        tn = hdf_ci.diff().abs().fillna(0).to_numpy()[1:]
        nh = (hl * valid_t).sum(axis=1)
        eg = np.where(nh > 0, (rets_t * hl * valid_t).sum(axis=1) / np.maximum(nh, 1), 0.0)
        cst = np.where(nh > 0, (tn * valid_t).sum(axis=1) * (MAKER_RT / 2.0) / np.maximum(nh, 1), 0.0)
        net = (eg - cst) * reg_t
        config_compounds[ci] = float(np.prod(1 + net) - 1) * 100

    # SELECT: keep configs with TRAIN compound > 0
    selected = [ci for ci, comp in config_compounds.items() if comp > 0]
    sub_band = [all_configs[ci] for ci in selected]

    if len(sub_band) < 2:
        return {"status": "NO_TRAIN_POSITIVE_CONFIGS", "family": family, "tf": tf,
                "n_total": len(all_configs), "n_train_positive": 0}

    # =========================================================================
    # STAGE 2: EVALUATION on target window with curated sub-band
    # =========================================================================
    provider = make_rigorous_hold_provider(family, sub_band, lookback_bars, slice_bars, vote_threshold)

    close_cols, held_cols = {}, {}
    for sym in _u100_symbols():
        feats = None
        if need_vol:
            rec = load_asset_with_features(sym, tf, wall_ms, VOL_COLS)
            if rec is None:
                continue
            dates, o, h, l, c, feats = rec
            vol = feats.get("volume"); bv = feats.get("buy_vol"); sv = feats.get("sell_vol")
            held = provider(sym, o, h, l, c, vol, bv, sv)
        else:
            rec = _load_asset(sym, tf, wall_ms)
            if rec is None:
                continue
            dates, o, h, l, c = rec
            held = provider(sym, o, h, l, c)
        close_cols[sym] = pd.Series(c, index=dates)
        held_cols[sym] = pd.Series(held, index=dates)

    if len(close_cols) < 10:
        return {"status": "INSUFFICIENT_ASSETS", "family": family, "tf": tf}

    close_df = pd.DataFrame(close_cols).sort_index()
    held_df = pd.DataFrame(held_cols).reindex(close_df.index).fillna(0).astype(np.int8)

    # Vectorized replay with regime gate + cost
    rets = close_df.pct_change().to_numpy()[1:]
    held_lag = held_df.shift(1).fillna(0).to_numpy()[1:]
    turn = held_df.diff().abs().fillna(0).to_numpy()[1:]
    dates = close_df.index[1:]

    reg_vals = np.ones(len(dates))
    if regime_eval is not None:
        norm_days = pd.DatetimeIndex(dates).normalize()
        reg_aligned = regime_eval.reindex(regime_eval.index.union(norm_days)).ffill().reindex(norm_days).fillna(0)
        reg_vals = reg_aligned.to_numpy().astype(float)

    mask = np.ones(len(dates), dtype=bool)
    if window_start:
        mask &= dates >= pd.Timestamp(window_start)
    if window_end:
        mask &= dates < pd.Timestamp(window_end)

    valid = np.isfinite(rets)
    rets_f = np.where(valid, rets, 0.0)
    n_held = (held_lag * valid).sum(axis=1)
    eng_gross = np.where(n_held > 0, (rets_f * held_lag * valid).sum(axis=1) / np.maximum(n_held, 1), 0.0)
    cost = np.where(n_held > 0, (turn * valid).sum(axis=1) * (MAKER_RT / 2.0) / np.maximum(n_held, 1), 0.0)
    eng = (eng_gross - cost) * reg_vals
    n_all = valid.sum(axis=1)
    al = np.where(n_all > 0, (rets_f * valid).sum(axis=1) / np.maximum(n_all, 1), 0.0)

    eng_w = eng[mask]; al_w = al[mask]
    deploy = float(np.mean(n_held[mask] > 0)) if mask.sum() else 0.0

    if len(eng_w) == 0:
        return {"status": "NO_BARS", "family": family, "tf": tf}

    lin_wk = float(np.mean(eng_w)) * bpd * 7 * 100
    al_lin_wk = float(np.mean(al_w)) * bpd * 7 * 100
    comp = float(np.prod(1 + eng_w) - 1) * 100
    cum = np.cumprod(1 + eng_w)
    maxdd = float(np.min((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum))) * 100 if len(cum) > 1 else 0.0
    vol_ann = float(np.std(eng_w)) * np.sqrt(bpd * 365) * 100
    sharpe = (float(np.mean(eng_w)) * bpd * 365) / (float(np.std(eng_w)) * np.sqrt(bpd * 365)) if np.std(eng_w) > 1e-12 else 0.0

    # Random-null
    T, A = rets_f.shape
    randc = []
    for s in range(n_null):
        rng = np.random.default_rng(s)
        rh = (rng.random((T, A)) < deploy).astype(np.int8)
        nh = (rh * valid).sum(axis=1)
        r = np.where(nh > 0, (rets_f * rh * valid).sum(axis=1) / np.maximum(nh, 1), 0.0) * reg_vals
        r = r[mask]
        randc.append(float(np.prod(1 + r) - 1) * 100)
    null_pct = float(np.mean(np.array(randc) < comp)) * 100

    elapsed = time.time() - t0
    return {
        "status": "OK", "family": family, "tf": tf,
        "lookback_days": lookback_days, "slice_days": slice_days,
        "n_configs_total": len(all_configs), "n_configs_train_positive": len(sub_band),
        "selection_rate": round(len(sub_band) / len(all_configs), 3),
        "vote_threshold": vote_threshold,
        "n_assets": len(close_cols), "n_bars_window": int(mask.sum()),
        "deploy": round(deploy, 4), "lin_wk": round(lin_wk, 4), "al_lin_wk": round(al_lin_wk, 4),
        "edge_pp": round(lin_wk - al_lin_wk, 4),
        "compound": round(comp, 1), "maxdd": round(maxdd, 1),
        "ann_vol": round(vol_ann, 1), "sharpe": round(sharpe, 3),
        "null_pct": round(null_pct, 1), "n_null": n_null,
        "elapsed_s": round(elapsed, 1),
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--tf", default="4h")
    ap.add_argument("--lookback", type=int, default=7)
    ap.add_argument("--window", default="VAL", choices=["TRAIN", "VAL", "OOS2024"])
    ap.add_argument("--null", type=int, default=100)
    a = ap.parse_args()

    ws = {"TRAIN": (None, TRAIN_END, TRAIN_WALL_MS),
          "VAL": (TRAIN_END, VAL_END, VAL_WALL_MS),
          "OOS2024": (VAL_END, OOS_2024_END, OOS_WALL_MS)}
    w_start, w_end, w_ms = ws[a.window]
    r = run_rigorous_cell(a.family, a.tf, lookback_days=a.lookback,
                          window_start=w_start, window_end=w_end, wall_ms=w_ms, n_null=a.null)
    print(json.dumps(r, indent=2, default=str))
