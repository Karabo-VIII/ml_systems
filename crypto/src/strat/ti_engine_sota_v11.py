"""SOTA TI Engine V1.1 — vectorized + real-chimera-feature families (6-8).

This is the V1.1 iteration on `ti_engine_sota.py` addressing the two flagged issues:

1. VECTORIZATION (item 2): Supertrend / Funding / OI signal functions in the V1
   build used Python `for` loops over every bar. The Supertrend loop is inherently
   sequential (band continuity depends on the prior bar) so it stays a loop, but the
   ADX/efficiency, Funding-z, and OI-divergence proxies are replaced with vectorized
   pandas-rolling implementations here.

2. REAL CHIMERA FEATURES (item 3): the V1 Funding and OI engines used PRICE PROXIES
   (rolling z-score of returns, price efficiency). V1.1 wires in the ACTUAL chimera
   columns `norm_funding`, `norm_funding_momentum`, `norm_oi_change`,
   `norm_oi_price_divergence`. These are pre-normalized (rolling-z) and causal.

Honest framing (per FORENSIC_AUDIT_RESULT.md + OOS_VALIDATION_RESULT.md): the
DEV-window compounds are inflated by the EW-rebalancing premium (~9.5x) and the
regime gate. The OOS test (ti_sota_oos_test) is the honest arbiter. This module
makes families 6-8 fast and feature-correct so they can be put through the SAME
OOS gauntlet rather than judged on inflated DEV numbers.
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

import strat.fleet_lab as fl
from strat.ma_wealth_driver import _norm_sym
from pipeline.chimera_loader import ChimeraLoader

_2H_MS = 2 * 3600 * 1000
_FREQ = {"1d": "D", "4h": "4h", "2h": "2h", "1h": "1h", "30m": "30min", "15m": "15min"}


# =============================================================================
# CHIMERA FEATURE LOADER (real norm_funding / norm_oi features, causal, walled)
# =============================================================================

def load_asset_with_features(sym: str, tf: str, wall_ms: int,
                             feature_cols: list[str], floor_ms: int | None = None):
    """Load OHLC + named chimera features for one asset, DEV/eval-walled and causal.

    Mirrors `_load_asset` exactly (same 2h synthesis, same wall/floor) but ALSO
    returns the requested pre-normalized chimera feature arrays. For 2h, chimera
    norm features have NO native baseline (per fleet_lab) so they are returned as
    NaN — callers must treat NaN as "feature unavailable" (vote abstains).

    Returns (dates, o, h, l, c, {col: np.array}) or None.
    """
    load_cad = "1h" if tf == "2h" else tf
    want = ["open", "high", "low", "close"] + feature_cols
    try:
        df = ChimeraLoader().load(_norm_sym(sym), cadence=load_cad, features=want)
    except Exception:
        try:
            df = ChimeraLoader().load(_norm_sym(sym), cadence=load_cad,
                                      features=["open", "high", "low", "close"])
        except Exception:
            return None

    have_feats = [col for col in feature_cols if col in df.columns]

    if tf == "2h":
        ts1h = df["timestamp"].to_numpy()
        b = (ts1h // _2H_MS).astype("int64")
        agg = {"open": ("open", "first"), "high": ("high", "max"),
               "low": ("low", "min"), "close": ("close", "last")}
        pdf_data = {"open": df["open"].to_numpy().astype(float),
                    "high": df["high"].to_numpy().astype(float),
                    "low": df["low"].to_numpy().astype(float),
                    "close": df["close"].to_numpy().astype(float), "__b": b}
        pdf = pd.DataFrame(pdf_data)
        g = pdf.groupby("__b", sort=True).agg(**agg)
        ts = (g.index.to_numpy() * _2H_MS).astype("int64")
        o = g["open"].to_numpy(); h = g["high"].to_numpy()
        l = g["low"].to_numpy(); c = g["close"].to_numpy()
        # chimera norm features have no native-2h baseline -> NaN (unavailable)
        feats = {col: np.full(len(ts), np.nan) for col in feature_cols}
    else:
        ts = df["timestamp"].to_numpy()
        o = df["open"].to_numpy().astype(float); h = df["high"].to_numpy().astype(float)
        l = df["low"].to_numpy().astype(float); c = df["close"].to_numpy().astype(float)
        feats = {}
        for col in feature_cols:
            if col in have_feats:
                feats[col] = df[col].to_numpy().astype(float)
            else:
                feats[col] = np.full(len(ts), np.nan)

    keep = ts < wall_ms
    if floor_ms is not None:
        keep = keep & (ts >= floor_ms)
    if keep.sum() < 31:
        return None
    dates = pd.to_datetime(ts[keep], unit="ms").floor(_FREQ.get(tf, "D"))
    feats_kept = {col: arr[keep] for col, arr in feats.items()}
    return dates, o[keep], h[keep], l[keep], c[keep], feats_kept


# =============================================================================
# VECTORIZED SIGNAL FUNCTIONS (families 5-8)
# =============================================================================

def adx_held_vec(close: np.ndarray, lookback: int, threshold: float) -> np.ndarray:
    """Vectorized efficiency-ratio (ADX proxy). Long when trend efficiency > thr AND up.

    efficiency[i] = |close[i]-close[i-lookback]| / sum(|diff| over the window)
    Replaces the V1 per-bar Python loop with pandas rolling.
    """
    n = len(close)
    s = pd.Series(close)
    net_move = (s - s.shift(lookback)).abs()
    abs_diff = s.diff().abs()
    sum_moves = abs_diff.rolling(lookback, min_periods=lookback).sum()
    eff = (net_move / sum_moves.replace(0, np.nan)).to_numpy()
    up = (s > s.shift(lookback)).to_numpy()
    held = ((eff > threshold) & up).astype(np.int8)
    held[:lookback] = 0
    return np.nan_to_num(held).astype(np.int8)


def supertrend_held_vec(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                        atr_period: int, multiplier: float) -> np.ndarray:
    """Supertrend. ATR via vectorized true-range + rolling mean; the band-continuity
    direction loop stays sequential (it is inherently recursive), but only ONE light
    loop over bars with all heavy work vectorized."""
    n = len(close)
    h = pd.Series(high); l = pd.Series(low); c = pd.Series(close)
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.rolling(atr_period, min_periods=atr_period).mean().to_numpy()
    mid = ((high + low) / 2.0)
    upper = mid + multiplier * atr
    lower = mid - multiplier * atr

    held = np.zeros(n, dtype=np.int8)
    direction = 1
    fl_lower = lower.copy(); fl_upper = upper.copy()
    for i in range(atr_period + 1, n):
        if np.isnan(atr[i]):
            continue
        if lower[i] < fl_lower[i-1] and close[i-1] > fl_lower[i-1]:
            fl_lower[i] = fl_lower[i-1]
        if upper[i] > fl_upper[i-1] and close[i-1] < fl_upper[i-1]:
            fl_upper[i] = fl_upper[i-1]
        if close[i] > fl_upper[i-1]:
            direction = 1
        elif close[i] < fl_lower[i-1]:
            direction = -1
        held[i] = 1 if direction == 1 else 0
    return held


def funding_held_vec(norm_funding: np.ndarray, lookback: int,
                     z_long: float, z_exit: float) -> np.ndarray:
    """REAL funding signal. norm_funding is already a rolling-z normalized chimera
    feature, so we threshold it directly (contrarian: very negative funding = shorts
    crowded = bullish). Hysteresis enter/exit, vectorized state via the standard
    'last decisive signal forward-fill' trick.

    Returns zeros where norm_funding is NaN (feature unavailable, e.g. synth-2h).
    """
    nf = np.asarray(norm_funding, dtype=float)
    n = len(nf)
    if np.all(np.isnan(nf)):
        return np.zeros(n, dtype=np.int8)
    enter = nf < z_long          # contrarian long when funding deeply negative
    exit_ = nf > z_exit
    # encode: +1 at enter, -1 at exit, 0 otherwise; forward-fill the last decisive state
    sig = np.where(enter, 1, np.where(exit_, -1, 0))
    state = pd.Series(np.where(sig == 0, np.nan, sig)).ffill().fillna(0).to_numpy()
    held = (state > 0).astype(np.int8)
    held[np.isnan(nf)] = 0
    return held


def funding_momentum_held_vec(norm_funding_mom: np.ndarray, z_long: float, z_exit: float) -> np.ndarray:
    """REAL funding-momentum signal: long when funding momentum turns up from extreme."""
    fm = np.asarray(norm_funding_mom, dtype=float)
    n = len(fm)
    if np.all(np.isnan(fm)):
        return np.zeros(n, dtype=np.int8)
    enter = fm > z_long
    exit_ = fm < z_exit
    sig = np.where(enter, 1, np.where(exit_, -1, 0))
    state = pd.Series(np.where(sig == 0, np.nan, sig)).ffill().fillna(0).to_numpy()
    held = (state > 0).astype(np.int8)
    held[np.isnan(fm)] = 0
    return held


def oi_divergence_held_vec(norm_oi_div: np.ndarray, threshold: float) -> np.ndarray:
    """REAL OI-price-divergence signal. norm_oi_price_divergence is the pre-normalized
    chimera feature: positive = OI rising faster than price (accumulation -> bullish).
    Long when divergence > threshold. Vectorized."""
    od = np.asarray(norm_oi_div, dtype=float)
    n = len(od)
    if np.all(np.isnan(od)):
        return np.zeros(n, dtype=np.int8)
    held = (od > threshold).astype(np.int8)
    held[np.isnan(od)] = 0
    return held


def oi_change_held_vec(norm_oi_change: np.ndarray, threshold: float) -> np.ndarray:
    """REAL OI-change signal: long when open interest is rising (norm_oi_change > thr)."""
    oc = np.asarray(norm_oi_change, dtype=float)
    n = len(oc)
    if np.all(np.isnan(oc)):
        return np.zeros(n, dtype=np.int8)
    held = (oc > threshold).astype(np.int8)
    held[np.isnan(oc)] = 0
    return held


# =============================================================================
# V1.1 CONFIG GRIDS (in normalized-z units, since features are pre-normalized)
# =============================================================================

# Funding (real norm_funding): contrarian long on deeply negative z
FUNDING_REAL_CONFIGS = [
    (-0.5, 0.0), (-1.0, 0.0), (-1.5, 0.0),
    (-1.0, 0.5), (-1.5, 0.5), (-2.0, 0.5),
    (-0.5, 0.5), (-1.0, 1.0), (-1.5, 1.0),
]

# Funding momentum (real norm_funding_momentum): trend-follow funding
FUNDING_MOM_CONFIGS = [
    (0.5, -0.5), (1.0, -0.5), (0.5, 0.0), (1.0, 0.0), (1.5, 0.0),
]

# OI divergence (real norm_oi_price_divergence): long on accumulation
OI_DIV_REAL_CONFIGS = [
    (0.0,), (0.25,), (0.5,), (0.75,), (1.0,), (-0.25,),
]

# OI change (real norm_oi_change): long on rising OI
OI_CHANGE_CONFIGS = [
    (0.0,), (0.25,), (0.5,), (0.75,), (-0.25,),
]


# =============================================================================
# FEATURE-AWARE ENSEMBLE PROVIDERS (families 6-8, real chimera)
# =============================================================================

def make_funding_provider(configs=FUNDING_REAL_CONFIGS, vote_threshold: float = 0.5):
    """Ensemble over real norm_funding configs. Returns provider(feats) -> held."""
    def provider(norm_funding: np.ndarray) -> np.ndarray:
        n = len(norm_funding)
        if np.all(np.isnan(norm_funding)):
            return np.zeros(n, dtype=np.int8)
        votes = np.zeros(n, dtype=float)
        for z_long, z_exit in configs:
            votes += funding_held_vec(norm_funding, 7, z_long, z_exit).astype(float)
        frac = votes / len(configs)
        return (frac > vote_threshold).astype(np.int8)
    return provider


def make_oi_provider(div_configs=OI_DIV_REAL_CONFIGS, vote_threshold: float = 0.5):
    """Ensemble over real norm_oi_price_divergence configs."""
    def provider(norm_oi_div: np.ndarray) -> np.ndarray:
        n = len(norm_oi_div)
        if np.all(np.isnan(norm_oi_div)):
            return np.zeros(n, dtype=np.int8)
        votes = np.zeros(n, dtype=float)
        for (thr,) in div_configs:
            votes += oi_divergence_held_vec(norm_oi_div, thr).astype(float)
        frac = votes / len(div_configs)
        return (frac > vote_threshold).astype(np.int8)
    return provider


def make_supertrend_provider(configs, vote_threshold: float = 0.5):
    """Ensemble over Supertrend configs (vectorized ATR)."""
    def provider(o, h, l, c) -> np.ndarray:
        close = np.asarray(c, dtype=float); high = np.asarray(h, dtype=float); low = np.asarray(l, dtype=float)
        n = len(close)
        votes = np.zeros(n, dtype=float)
        for atr_p, mult in configs:
            votes += supertrend_held_vec(close, high, low, atr_p, mult).astype(float)
        frac = votes / len(configs)
        return (frac > vote_threshold).astype(np.int8)
    return provider


if __name__ == "__main__":
    # quick smoke test: load BTC 4h with real funding/OI features, run vectorized signals
    wall_ms = int(pd.Timestamp(fl.DEV_END).value // 10**6)
    feat_cols = ["norm_funding", "norm_funding_momentum", "norm_oi_change", "norm_oi_price_divergence"]
    rec = load_asset_with_features("BTCUSDT", "4h", wall_ms, feat_cols)
    if rec is None:
        print("FAILED to load BTC with features")
        sys.exit(1)
    dates, o, h, l, c, feats = rec
    print(f"Loaded BTC 4h: {len(c)} bars, {dates.min()} -> {dates.max()}")
    for col in feat_cols:
        arr = feats[col]
        n_valid = int(np.sum(np.isfinite(arr)))
        print(f"  {col}: {n_valid}/{len(arr)} finite, "
              f"range [{np.nanmin(arr):.2f}, {np.nanmax(arr):.2f}]" if n_valid else f"  {col}: ALL NaN")

    fp = make_funding_provider()
    oip = make_oi_provider()
    f_held = fp(feats["norm_funding"])
    oi_held = oip(feats["norm_oi_price_divergence"])
    print(f"\n  Funding ensemble deploy rate: {np.mean(f_held):.1%}")
    print(f"  OI-divergence ensemble deploy rate: {np.mean(oi_held):.1%}")

    stp = make_supertrend_provider([(7, 2.0), (10, 2.5), (14, 3.0)])
    st_held = stp(o, h, l, c)
    print(f"  Supertrend ensemble deploy rate: {np.mean(st_held):.1%}")
    print("\n  Vectorized signals OK.")
