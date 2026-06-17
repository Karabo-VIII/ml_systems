"""experiments/adaptive_ma/expert/adaptive_ma.py -- adaptive moving-average core library.

Implements the adaptive-MA mechanism per docs/ADAPTIVE_MA_BRIEF_2026_06_05.md (expert rig).

MECHANISM (all per-asset, CAUSAL / past-only):
  1. Rolling features (each .shift(1) -> uses ONLY bars <= t-1 at decision bar t):
       - realized_vol  : rolling std of 1-bar log returns over RV_WIN.
       - trend_strength : Kaufman Efficiency Ratio (ER) over ER_WIN -- |net move| / sum|bar moves|,
                          in [0,1]; high = clean directional move, low = chop. (the per-asset trend feature)
       - xs_dispersion : cross-sectional std of the universe's 1-bar returns at each date
                         (a MARKET-state feature; merged by date, then shifted). high = idiosyncratic /
                         many asset-specific moves; low = everything moving together (beta day).
  2. Causal self-normalization: each feature -> trailing rolling PERCENTILE over PCT_WIN bars (past-only,
     per-asset). No full-sample standardization, no TRAIN/VAL threshold leakage into the rank itself.
  3. Deterministic map  (trend_regime in {chop,mod,trend}) x (vol_high in {0,1}) -> (fast_len, slow_len, ma_type).
       trend (er_rank high) -> FAST pair, EMA  : enter the clean move early and ride it.
       mod   (er_rank mid)  -> MID pair,  SMA.
       chop  (er_rank low)  -> SLOW pair, SMA  : long, smooth windows -> fewer whipsaw crosses.
       vol_high (rv_rank high) widens by one notch (longer windows filter noise in high vol).
  4. The adapted fast/slow MA columns are assembled per-bar by SELECTING from a table of pre-computed
     past-only MA series (one per candidate (type,length)). Selection index is past-only -> the assembled
     column is past-only. (verified empirically by `causal_selfcheck`.)

The harness then does the LONG-ONLY entry (adapted fast > adapted slow, fill next-bar-open) and ONE
uniform exit (opposite cross; optional uniform time-stop). cost = taker 0.0024 (src/strat fill_model).

No look-ahead, no emoji (cp1252). numpy/pandas only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from wealth_bot.harness import sma_past_only, ema_past_only  # noqa: E402  (past-only MA helpers)

# ---- tunable constants (documented; fixed BEFORE seeing held-out results) -------------------------
RV_WIN = 20      # realized-vol lookback (bars)
ER_WIN = 20      # Kaufman efficiency-ratio lookback (bars)
PCT_WIN = 252    # trailing percentile window for causal self-normalization (bars ~ 1y daily)
LO_BAND = 0.33   # er_rank < LO -> chop
HI_BAND = 0.66   # er_rank > HI -> trend ; rv_rank > HI -> high-vol widen

# Config table: (trend_regime, vol_high) -> (fast_len, slow_len, ma_type)
# trend_regime: 0=chop, 1=moderate, 2=trend ; vol_high: 0/1
# Base widths by trend; vol_high bumps one notch slower (wider) to filter noise.
_BASE = {
    2: (8, 21, "ema"),    # trend  -> fast, responsive EMA
    1: (10, 30, "sma"),   # moderate
    0: (20, 50, "sma"),   # chop   -> slow, smooth (whipsaw guard)
}
_WIDEN = {  # vol_high notch-slower variant of each base
    2: (10, 30, "ema"),
    1: (15, 40, "sma"),
    0: (30, 60, "sma"),
}


def config_for(trend_regime: int, vol_high: int):
    return _WIDEN[trend_regime] if vol_high else _BASE[trend_regime]


def all_configs():
    """Every (fast_len, slow_len, ma_type) the map can emit -> drives the MA pre-compute table."""
    cfgs = []
    for tr in (0, 1, 2):
        for vh in (0, 1):
            cfgs.append(config_for(tr, vh))
    return cfgs


# ---- causal features ------------------------------------------------------------------------------
def _rolling_pct_rank(s: pd.Series, win: int) -> pd.Series:
    """Trailing rolling percentile of s[t] within s[t-win+1 .. t] (past-only). The value at t is the
    fraction of the trailing window <= s[t]. Caller shifts the INPUT feature by 1 already, so this is
    strictly past. Implemented as rolling.apply (clear + correct; cost is fine at u100/1d sizes)."""
    def _pr(x):
        return float((x <= x[-1]).mean())
    return s.rolling(win, min_periods=max(20, win // 4)).apply(_pr, raw=True)


def compute_features(df: pd.DataFrame, xs_disp: pd.Series | None = None) -> pd.DataFrame:
    """Add causal feature + regime columns to a per-asset OHLC frame (must have close; date index order).
    xs_disp: optional Series aligned to df.index giving the (already date-merged) cross-sectional
    dispersion for this asset's dates. All features .shift(1) so decision bar t sees only <= t-1."""
    out = df.copy().reset_index(drop=True)
    close = out["close"].astype(float)

    # 1. realized vol (log returns)
    logret = np.log(close / close.shift(1))
    rv = logret.rolling(RV_WIN, min_periods=RV_WIN // 2).std()

    # 2. Kaufman efficiency ratio (trend strength), in [0,1]
    change = (close - close.shift(ER_WIN)).abs()
    vol_path = close.diff().abs().rolling(ER_WIN, min_periods=ER_WIN // 2).sum()
    er = (change / vol_path.replace(0.0, np.nan)).clip(0.0, 1.0)

    # shift(1): strictly past at decision bar t
    out["f_realized_vol"] = rv.shift(1)
    out["f_trend_er"] = er.shift(1)
    out["f_xs_dispersion"] = (xs_disp.shift(1) if xs_disp is not None else np.nan)

    # 3. causal self-normalized ranks (trailing percentile, past-only)
    out["er_rank"] = _rolling_pct_rank(out["f_trend_er"], PCT_WIN)
    out["rv_rank"] = _rolling_pct_rank(out["f_realized_vol"], PCT_WIN)

    # 4. discrete regimes from ranks
    er_rank = out["er_rank"]
    trend_regime = np.where(er_rank > HI_BAND, 2, np.where(er_rank < LO_BAND, 0, 1))
    # rows with undefined rank (warmup) -> default to moderate regime, no widen (neutral)
    trend_regime = np.where(np.isnan(er_rank.values), 1, trend_regime)
    vol_high = np.where(np.isnan(out["rv_rank"].values), 0, (out["rv_rank"].values > HI_BAND).astype(int))
    out["trend_regime"] = trend_regime.astype(int)
    out["vol_high"] = vol_high.astype(int)
    return out


# ---- adapted MA assembly --------------------------------------------------------------------------
def _ma_series(close: pd.Series, length: int, ma_type: str) -> pd.Series:
    if ma_type == "ema":
        return ema_past_only(close, length=length, shift=0)
    return sma_past_only(close, length=length, shift=0)


def build_adaptive_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble adaptive_fast / adaptive_slow per-bar by selecting the pre-computed past-only MA series
    indexed by the (trend_regime, vol_high) config at each bar. Requires compute_features() already run."""
    out = df.copy().reset_index(drop=True)
    close = out["close"].astype(float)
    n = len(out)

    # pre-compute every candidate MA series once (past-only)
    tbl: dict[tuple[int, str], np.ndarray] = {}
    for (fl, sl, mt) in all_configs():
        for (length, t) in ((fl, mt), (sl, mt)):
            key = (length, t)
            if key not in tbl:
                tbl[key] = _ma_series(close, length, t).to_numpy()

    fast = np.full(n, np.nan)
    slow = np.full(n, np.nan)
    fast_len = np.zeros(n, int)
    slow_len = np.zeros(n, int)
    for i in range(n):
        cfg = config_for(int(out["trend_regime"].iat[i]), int(out["vol_high"].iat[i]))
        fl, sl, mt = cfg
        fast[i] = tbl[(fl, mt)][i]
        slow[i] = tbl[(sl, mt)][i]
        fast_len[i] = fl
        slow_len[i] = sl
    out["adaptive_fast"] = fast
    out["adaptive_slow"] = slow
    out["sel_fast_len"] = fast_len
    out["sel_slow_len"] = slow_len
    return out


def build_fixed_columns(df: pd.DataFrame, fast_len: int, slow_len: int, ma_type: str = "sma") -> pd.DataFrame:
    """Fixed-config MA baseline columns (the thing adaptation must beat to earn its keep)."""
    out = df.copy().reset_index(drop=True)
    close = out["close"].astype(float)
    out["fix_fast"] = _ma_series(close, fast_len, ma_type).to_numpy()
    out["fix_slow"] = _ma_series(close, slow_len, ma_type).to_numpy()
    return out


# ---- causal self-check (look-ahead falsifier) -----------------------------------------------------
def causal_selfcheck(df_feat: pd.DataFrame, sample_idx: list[int] | None = None) -> dict:
    """Confirm adaptive_fast/adaptive_slow at bar t depend ONLY on data[:t+1] (no look-ahead).
    Re-derive the adaptive columns from each TRUNCATED prefix df[:t+1] and compare to the full-series
    value at t. If they match for every sampled t, the assembly is causal by construction.
    df_feat must already have features (compute_features) but is re-featured per prefix here from raw OHLC."""
    full = build_adaptive_columns(df_feat)
    n = len(df_feat)
    if sample_idx is None:
        # sample across the series (skip warmup)
        sample_idx = list(range(300, n, max(1, (n - 300) // 40)))
    max_abs_diff = 0.0
    checked = 0
    raw_cols = ["date", "open", "high", "low", "close"]
    if "f_xs_dispersion" in df_feat.columns:
        raw_cols_disp = df_feat["f_xs_dispersion"]
    for t in sample_idx:
        if t < 300 or t >= n:
            continue
        prefix_raw = df_feat.iloc[: t + 1][raw_cols].copy()
        # re-featurize using ONLY the prefix (xs_disp passed as the prefix slice of the already-shifted col)
        disp = df_feat["f_xs_dispersion"].iloc[: t + 1].reset_index(drop=True) if "f_xs_dispersion" in df_feat.columns else None
        # compute_features expects the *unshifted* xs_disp (it shifts internally). We stored the SHIFTED
        # col, so undo by passing None for disp here -- dispersion does not drive the config map, so the
        # selected (fast,slow,type) is identical with or without it. We re-run the vol/ER path on the prefix.
        feat_prefix = compute_features(prefix_raw, xs_disp=None)
        adapt_prefix = build_adaptive_columns(feat_prefix)
        for col in ("adaptive_fast", "adaptive_slow"):
            a = adapt_prefix[col].iloc[-1]
            b = full[col].iloc[t]
            if pd.notna(a) and pd.notna(b):
                max_abs_diff = max(max_abs_diff, abs(float(a) - float(b)))
        checked += 1
    return {"checked_points": checked, "max_abs_diff": max_abs_diff,
            "causal_ok": bool(max_abs_diff < 1e-9)}
