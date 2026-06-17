"""data_loader -- chimera load + features + labels + signals, parameterized.

__contract__:
  inputs: BotConfig
  outputs: DataFrame with [date, close, chimera_features_lagged, regime tags];
           signals matrix (n, n_strats); forward returns vector;
           segment masks for TRAIN/VAL/OOS/UNSEEN.
  invariants:
    - chimera features lagged by config.chimera_lag_bars (no peek)
    - signals lagged by 1 bar (yesterday's MA-cross fires today's entry)
    - whale filter lagged by config.chimera_lag_bars
    - fwd_return uses close.shift(-fwd_bars) which is correct for "we know the future
      AT TRAINING TIME but the signal at bar t cannot see beyond t"
"""
from __future__ import annotations

__contract__ = {
    "kind": "data_loader",
    "owner": "wealth_bot/framework/data_loader",
    "purpose": "Load chimera, build features + signals + labels, parameterized by config",
    "outputs": {
        "df": "DataFrame[date, close, ...chimera_lagged]",
        "signals": "ndarray(n, n_strats) of {0,1}",
        "fwd_ret": "ndarray(n,) forward returns net of round-trip cost",
        "masks": "dict[segment_name -> bool mask]",
    },
    "invariants": [
        "chimera lagged by chimera_lag_bars",
        "signals lagged by 1 bar",
        "no peek into future for any input feature",
    ],
}

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from .config import BotConfig, StrategySpec

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


def _sma(arr: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(arr).rolling(n, min_periods=n).mean().values


def _ema(arr: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(arr).ewm(span=n, adjust=False, min_periods=n).mean().values


def _dema(arr: np.ndarray, n: int) -> np.ndarray:
    """Double EMA: 2*EMA(n) - EMA(EMA(n))."""
    e1 = pd.Series(arr).ewm(span=n, adjust=False, min_periods=n).mean()
    e2 = e1.ewm(span=n, adjust=False, min_periods=n).mean()
    return (2 * e1 - e2).values


def _wma(arr: np.ndarray, n: int) -> np.ndarray:
    """Linear-weighted MA."""
    s = pd.Series(arr)
    weights = np.arange(1, n + 1)
    return s.rolling(n, min_periods=n).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True).values


def _hma(arr: np.ndarray, n: int) -> np.ndarray:
    """Hull MA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))."""
    if n < 4:
        return _wma(arr, n)
    half = max(2, n // 2)
    sq = max(2, int(np.sqrt(n)))
    w1 = _wma(arr, half)
    w2 = _wma(arr, n)
    raw = 2 * w1 - w2
    raw_no_nan = np.where(np.isnan(raw), 0.0, raw)
    h = _wma(raw_no_nan, sq)
    nan_mask = np.isnan(raw) | (np.arange(len(arr)) < n)
    h[nan_mask] = np.nan
    return h


def load_chimera(asset: str, cadence: str, extra_features: list[str]) -> pd.DataFrame:
    """Load latest chimera parquet for asset+cadence. Returns DF with date + close + features."""
    asset_lc = asset.lower()
    fp = sorted((ROOT / "data" / "processed" / "chimera" / cadence).glob(
        f"{asset_lc}_v51_chimera_{cadence}_*.parquet"))[-1]
    schema = pl.read_parquet_schema(fp)
    base_cols = ["timestamp", "close", "high", "low", "volume_usd"]
    cols = [c for c in base_cols + extra_features if c in schema]
    df = pl.read_parquet(fp, columns=cols).to_pandas()
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df.sort_values("date").reset_index(drop=True)


_BARS_PER_DAY = {"1h": 24, "4h": 6, "1d": 1}


def _bars_per_day(cadence: str) -> int:
    if cadence not in _BARS_PER_DAY:
        raise ValueError(
            f"unsupported cadence {cadence!r} for whale filter window scaling; "
            f"expected one of {list(_BARS_PER_DAY)}"
        )
    return _BARS_PER_DAY[cadence]


# Simple lag+gt/lt filter recipes used by the per-cadence sweep.
# Each entry: (column_name, comparator, threshold).
# Comparators: ">" or "<". Apply after shifting `lag_bars`.
_SIMPLE_FILTERS: dict[str, tuple[str, str, float]] = {
    "btc_tape>0":      ("te_btc_imb",        ">", 0.0),
    "short_liq_z>0":   ("liq_short_z30",     ">", 0.0),
    "long_liq_z<0":    ("liq_long_z30",      "<", 0.0),
    "basis_z<0":       ("bs_basis_z30",      "<", 0.0),
    "btc_ret<0":       ("xd_btc_return",     "<", 0.0),
    "tape_imb>0":      ("hbr_eta_imbalance", ">", 0.0),
    "hbr_eta_buy>0":   ("hbr_eta_buy",       ">", 0.0),
}


# Rolling-median filters (window scaled by cadence so 30d_median = 30 calendar days)
# Each entry: (column_name, window_days, comparator vs rolling-median).
# Added 2026-05-24 holistic re-mine: bd_imb>med is the GOLDEN orthogonal alpha at
# corr 0.168 to existing 31, all-4-positive on PEPE 4h.
_ROLLING_MEDIAN_FILTERS: dict[str, tuple[str, int, str]] = {
    "bd_imb>med":   ("bd_imbalance_l1", 30, ">"),   # book imbalance vs 30d median
    "fund_low":     ("fund_rate_mean",  30, "<"),   # funding lower than 30d median
    "lob_kyle_low": ("lob_bgf_kyle_lambda_mean", 30, "<"),  # low Kyle lambda
}


def _rolling_median_filter(df: pd.DataFrame, col: str, window_days: int, comp: str,
                             lag_bars: int, cadence: str = "4h") -> np.ndarray:
    """Filter signal = (col compared-to-its-rolling-window-median), lag-aware, cadence-aware."""
    if col not in df.columns:
        raise ValueError(f"rolling-median filter needs column {col!r} but missing from chimera")
    v = df[col].astype(float).values
    bpd = _bars_per_day(cadence)
    win = window_days * bpd
    mp = max(5, min(5 * bpd, win))
    med = pd.Series(v).rolling(win, min_periods=mp).median().values
    if comp == ">":
        f = (v > med).astype(int)
    elif comp == "<":
        f = (v < med).astype(int)
    else:
        raise ValueError(f"unknown comparator {comp!r}")
    return pd.Series(f).shift(lag_bars).fillna(0).astype(int).values


def _simple_lag_filter(df: pd.DataFrame, col: str, comp: str, thr: float,
                        lag_bars: int) -> np.ndarray:
    if col not in df.columns:
        raise ValueError(f"filter needs column '{col}' but missing from chimera")
    v = df[col].fillna(0).astype(float).values
    v_lag = pd.Series(v).shift(lag_bars).values
    if comp == ">":
        f = (v_lag > thr).astype(int)
    elif comp == "<":
        f = (v_lag < thr).astype(int)
    else:
        raise ValueError(f"unknown comparator {comp}")
    return np.where(np.isnan(v_lag), 0, f).astype(int)


def _whale_filter(df: pd.DataFrame, kind: str, lag_bars: int, cadence: str = "4h") -> np.ndarray:
    """Binary array of filter pass/fail (lag-aware, cadence-aware).

    Supported filter kinds:
      none, no_filter                       -> always 1
      whale_net>0                           -> whale_net_usd > 0
      whale_net>30d_median, whale_net>60d_median -> rolling-median (calendar days)
      btc_tape>0, short_liq_z>0, long_liq_z<0,
      basis_z<0, btc_ret<0, tape_imb>0      -> simple lagged gt/lt on chimera col
      whale&btc_tape, whale&short_liq,
      whale30d&btc_weak                     -> AND combo

    Rolling-median window is scaled by cadence so "30d_median" means
    30 calendar days regardless of bar granularity (1h=720 bars,
    4h=180 bars, 1d=30 bars). Min-periods sized to roughly 5 days of
    history (cadence-scaled), capped to the full window when needed
    on the 1d cadence where 30 bars is itself small.
    """
    if kind in ("none", "no_filter"):
        return np.ones(len(df), dtype=int)
    bpd = _bars_per_day(cadence)

    # Whale rolling-median family
    if kind in ("whale_net>0", "whale_net>30d_median", "whale_net>60d_median"):
        if "wh_whale_net_usd" not in df.columns:
            raise ValueError(f"whale filter '{kind}' requested but wh_whale_net_usd missing")
        w = df["wh_whale_net_usd"].astype(float).values
        if kind == "whale_net>0":
            f = (w > 0).astype(int)
        elif kind == "whale_net>30d_median":
            win = 30 * bpd
            mp = max(5, min(5 * bpd, win))
            med = pd.Series(w).rolling(win, min_periods=mp).median().values
            f = (w > med).astype(int)
        else:  # whale_net>60d_median
            win = 60 * bpd
            mp = max(10, min(10 * bpd, win))
            med = pd.Series(w).rolling(win, min_periods=mp).median().values
            f = (w > med).astype(int)
        return pd.Series(f).shift(lag_bars).fillna(0).astype(int).values

    # Simple lagged single-feature filters
    if kind in _SIMPLE_FILTERS:
        col, comp, thr = _SIMPLE_FILTERS[kind]
        return _simple_lag_filter(df, col, comp, thr, lag_bars)

    # Rolling-median filters (vs trailing window)
    if kind in _ROLLING_MEDIAN_FILTERS:
        col, window_days, comp = _ROLLING_MEDIAN_FILTERS[kind]
        return _rolling_median_filter(df, col, window_days, comp, lag_bars, cadence)

    # AND combos
    if kind == "whale&btc_tape":
        a = _whale_filter(df, "whale_net>0", lag_bars, cadence)
        b = _whale_filter(df, "btc_tape>0", lag_bars, cadence)
        return (a & b).astype(int)
    if kind == "whale&short_liq":
        a = _whale_filter(df, "whale_net>0", lag_bars, cadence)
        b = _whale_filter(df, "short_liq_z>0", lag_bars, cadence)
        return (a & b).astype(int)
    if kind == "whale30d&btc_weak":
        a = _whale_filter(df, "whale_net>30d_median", lag_bars, cadence)
        b = _whale_filter(df, "btc_ret<0", lag_bars, cadence)
        return (a & b).astype(int)
    if kind == "whale&hbr_eta_buy":
        a = _whale_filter(df, "whale_net>0", lag_bars, cadence)
        b = _whale_filter(df, "hbr_eta_buy>0", lag_bars, cadence)
        return (a & b).astype(int)

    # OR composites (R23c family — 2026-05-25 wiring)
    if kind == "whale_OR_pz_neg":
        # whale_net>0 OR premium_z90<0 (lagged)
        a = _whale_filter(df, "whale_net>0", lag_bars, cadence)
        if "premium_z90" not in df.columns:
            raise ValueError("filter 'whale_OR_pz_neg' needs column 'premium_z90' but missing from chimera")
        v = df["premium_z90"].fillna(0).astype(float).values
        v_lag = pd.Series(v).shift(lag_bars).values
        b = np.where(np.isnan(v_lag), 0, (v_lag < 0).astype(int))
        return (a | b).astype(int)

    raise ValueError(f"unknown filter kind '{kind}'")


_MA_FUNCS = {
    "SMA": _sma,
    "EMA": _ema,
    "DEMA": _dema,
    "WMA": _wma,
    "HMA": _hma,
}


def _ma_signal(closes: np.ndarray, spec: StrategySpec) -> np.ndarray:
    """Binary LONG signal per bar, lag-aware (uses yesterday's MA values).

    Supported ma_type forms (must match scripts/wealth_bot/scan_action_space_expansion.py):
      - "{KIND}_cross"  : KIND in {SMA, EMA, DEMA, WMA} -- needs fast<slow
      - "{KIND}_state"  : KIND in {SMA, EMA, HMA}        -- needs fast (slow ignored)
      - "{KIND}_dist"   : KIND in {SMA, EMA, WMA}        -- needs slow (MA period) +
                          spec.threshold_pct (fire when close/MA - 1 > threshold_pct/100)
    """
    ma_type = spec.ma_type
    if ma_type.endswith("_cross"):
        kind = ma_type.split("_")[0]
        if kind not in _MA_FUNCS:
            raise ValueError(f"unknown MA kind '{kind}' in spec.ma_type '{ma_type}'")
        fast_ma = _MA_FUNCS[kind](closes, spec.fast)
        slow_ma = _MA_FUNCS[kind](closes, spec.slow)
        sig = (fast_ma > slow_ma).astype(int)
    elif ma_type.endswith("_state"):
        kind = ma_type.split("_")[0]
        if kind not in _MA_FUNCS:
            raise ValueError(f"unknown MA kind '{kind}' in spec.ma_type '{ma_type}'")
        fast_ma = _MA_FUNCS[kind](closes, spec.fast)
        sig = (closes > fast_ma).astype(int)
    elif ma_type.endswith("_dist"):
        kind = ma_type.split("_")[0]
        if kind not in _MA_FUNCS:
            raise ValueError(f"unknown MA kind '{kind}' in spec.ma_type '{ma_type}'")
        ma = _MA_FUNCS[kind](closes, spec.slow)
        thr = float(spec.threshold_pct) / 100.0
        with np.errstate(divide="ignore", invalid="ignore"):
            dist = (closes / ma - 1.0)
        sig = (dist > thr).astype(int)
    else:
        raise ValueError(f"unknown ma_type '{spec.ma_type}'")
    return pd.Series(sig).shift(1).fillna(0).astype(int).values


def build_signals(df: pd.DataFrame, cfg: BotConfig) -> np.ndarray:
    """Build (n, n_strats) binary signal matrix per spec list."""
    closes = df["close"].values
    n = len(closes)
    K = len(cfg.strategies)
    signals = np.zeros((n, K), dtype=int)
    for k, spec in enumerate(cfg.strategies):
        ma_sig = _ma_signal(closes, spec)
        whale_sig = _whale_filter(df, spec.filter_kind, cfg.chimera_lag_bars, cfg.cadence)
        signals[:, k] = ma_sig & whale_sig
    return signals


def build_forward_returns(
    closes: np.ndarray,
    fwd_bars: int,
    cost_per_side: float,
    fill_mode: str = "next_bar_close",
) -> np.ndarray:
    """Forward fwd_bars-bar net-of-cost return for each bar.

    fill_mode:
      - "next_bar_close" (DEFAULT, leak-free): signal observed at close[i],
        entry at close[i+1], exit at close[i+1+fwd_bars]. Real-trading-realistic.
      - "same_bar_close" (LEAKY, legacy): entry at close[i] same bar as signal.
        Carries look-ahead premium; ONLY use for reproducing pre-2026-05-25 numbers.
    """
    n = len(closes)
    fr = np.full(n, np.nan)
    cost_rt = 2 * cost_per_side
    if fill_mode == "same_bar_close":
        for i in range(n - fwd_bars):
            fr[i] = (closes[i + fwd_bars] / closes[i] - 1) - cost_rt
    elif fill_mode == "next_bar_close":
        for i in range(n - fwd_bars - 1):
            fr[i] = (closes[i + fwd_bars + 1] / closes[i + 1] - 1) - cost_rt
    else:
        raise ValueError(f"Unknown fill_mode={fill_mode!r}")
    return fr


def lag_chimera(df: pd.DataFrame, feature_cols: list[str], lag_bars: int) -> pd.DataFrame:
    """Return a copy of df with chimera feature_cols shifted by lag_bars."""
    out = df.copy()
    for c in feature_cols:
        if c in out.columns:
            out[c] = out[c].shift(lag_bars)
    return out


# Default purge gap (bars). Per CLAUDE.md invariant: "Purge gap: 400 bars
# between segments to prevent normalization leakage." This is enforced at the
# walk-forward training window level by the model layer. Adding it at the
# segment-mask layer is OPT-IN (default 0) to avoid silently invalidating
# existing audit JSONs; new bots should pass purge_bars=400 via BotConfig.
# 400 bars = ~67 days at 4h cadence which comfortably exceeds the longest
# MA/EMA lookback (200-bar SMA) in scope.
PURGE_BARS_DEFAULT = 0
PURGE_BARS_RECOMMENDED = 400


def segment_masks(df: pd.DataFrame, cfg: BotConfig,
                    purge_bars: int | None = None) -> dict[str, np.ndarray]:
    """Bool masks for TRAIN/VAL/OOS/UNSEEN with optional purge gap at each segment start.

    purge_bars resolution order:
      1. explicit `purge_bars=` kwarg if given
      2. `cfg.purge_bars` attribute if present
      3. PURGE_BARS_DEFAULT (=0) for backward compatibility with existing audits

    When >0, that many bars at the START of each non-TRAIN segment are masked
    False to prevent indicator-window contamination from straddling boundaries.
    PURGE_BARS_RECOMMENDED=400 covers a 200-bar SMA + 200-bar buffer.

    New bots should opt in via BotConfig.purge_bars=400 (or pass kwarg) for
    full leakage protection.
    """
    if purge_bars is None:
        purge_bars = int(getattr(cfg, "purge_bars", PURGE_BARS_DEFAULT))
    w = cfg.windows
    dates = df["date"]
    raw = {
        "TRAIN": ((dates >= w.train_start) & (dates < w.train_end)).values,
        "VAL":   ((dates >= w.val_start)   & (dates < w.val_end)).values,
        "OOS":   ((dates >= w.oos_start)   & (dates < w.oos_end)).values,
        "UNSEEN":((dates >= w.unseen_start) & (dates < w.unseen_end)).values,
    }
    if purge_bars <= 0:
        return raw
    # Mask first `purge_bars` of each non-TRAIN segment as False.
    purged = {}
    for seg, m in raw.items():
        if seg == "TRAIN":
            purged[seg] = m.copy()
            continue
        idx = np.flatnonzero(m)
        m2 = m.copy()
        if len(idx) > 0:
            cut = idx[:min(purge_bars, len(idx))]
            m2[cut] = False
        purged[seg] = m2
    return purged


def prepare(cfg: BotConfig) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """End-to-end: load + lag + build signals + fwd returns + masks.

    Returns:
      df:        raw DataFrame with date + close + chimera (unlagged)
      df_lag:    DataFrame with chimera features lagged (for model input)
      signals:   (n, n_strats) binary
      fwd_ret:   (n,) net-of-cost forward returns
      masks:     dict segment -> bool mask
    """
    df = load_chimera(cfg.asset + "USDT" if not cfg.asset.endswith("USDT") else cfg.asset,
                      cfg.cadence, cfg.chimera_features)
    df_lag = lag_chimera(df, cfg.chimera_features, cfg.chimera_lag_bars)
    signals = build_signals(df, cfg)
    fwd_ret = build_forward_returns(df["close"].values, cfg.fwd_bars,
                                     cfg.risk.cost_per_side_pct / 100.0)
    masks = segment_masks(df, cfg)
    return df, df_lag, signals, fwd_ret, masks
