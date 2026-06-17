"""src/strat/deep2020_ti_pipeline.py -- GENERIC per-config TI x TF leaderboard, BASE vs IRONED (2020).

User /orc 2026-06-14 (6h autonomous): the other instance owns the 8 base MA types; THIS pipeline runs the
REMAINING trend indicators (MACD, ...) and then OTHER families (mean-reversion oscillators, breakout) through
the SAME end-to-end we did for MAs: every config x TF, BASE vs IRONED, wealth-ranked top-10, robust/non-robust
split, weakness teardown. Per-config per-TI per-TF NUMBERS (NOT correlation). 2020 band. STRICT long-only spot.

Reuses the CORRECTED machinery: fixed-EW alignment (unlisted=cash, cadence-invariant), VAL-only vol target,
base FULL stack (signal -> trail10 -> min_hold12 -> lag1 -> maker), VAL Jul-Sep / OOS Oct-Dec. Each indicator
provides base_fn + iron_fn (signal-level {0,1} held) + a config grid; the IRON is the signal-level family iron
(deep-research-informed) PLUS a vol-target overlay. WEALTH (OOS compound) is the ranking metric, not Sharpe.

RWYB: python -m strat.deep2020_ti_pipeline --indicator MACD --cadences 1d,4h. No emoji (cp1252).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.portfolio_replay as PR
from strat.portfolio_replay import apply_trail_stop, MAKER_RT
from strat.structural_fixes import min_hold
from strat.ma_type_upgrade import _ema, _sma
from strat.ma_2020_breakdown import _panel
from strat.deep2020_osc import _rsi, _stoch, _bbpct, _cci, _mr_held

WIN = ("2020-07-01", "2021-01-01"); SPLIT = "2020-10-01"
WARMUP = 400
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]
VOLWIN = {"1d": 14, "4h": 84, "2h": 168, "1h": 336, "30m": 672, "15m": 1344}
DRIFT_TOL = 10.0


# ============================ data (OHLC, fixed-EW, VAL-only vt) ============================
def load_ohlc(cad):
    s_ms = pd.Timestamp(WIN[0]).value // 10**6; e_ms = pd.Timestamp(WIN[1]).value // 10**6
    split_ms = pd.Timestamp(SPLIT).value // 10**6
    vw = VOLWIN[cad]; assets = []; rv_meds = []
    for sym in SYMS:
        try:
            o, h, l, c, ms = _panel(sym, cad)
        except Exception:
            continue
        e = int(np.searchsorted(ms, e_ms)); s0 = max(0, int(np.searchsorted(ms, s_ms)) - WARMUP)
        o2, h2, l2, c2, ms2 = o[s0:e], h[s0:e], l[s0:e], c[s0:e], ms[s0:e]
        if len(c2) < 40:
            continue
        win = ms2 >= s_ms
        if win.sum() < 30:
            continue
        ret = np.zeros(len(c2)); ret[1:] = c2[1:] / c2[:-1] - 1.0
        rv = pd.Series(ret).rolling(vw, min_periods=max(3, vw // 3)).std().shift(1).to_numpy()
        idx = pd.to_datetime(ms2[win], unit="ms")
        assets.append({"o": o2, "h": h2, "l": l2, "c": c2, "ret": ret, "win": win, "idx": idx, "rv": rv})
        vm = (ms2 >= s_ms) & (ms2 < split_ms)
        rv_meds.append(np.nanmedian(rv[vm]) if vm.sum() > 5 else np.nan)
    vt = float(np.nanmedian(rv_meds)) if rv_meds and not np.all(np.isnan(rv_meds)) else None
    return assets, vt


def load_ohlcv(cad):
    """Like load_ohlc but ALSO loads volume + buy_vol + sell_vol (order-flow) from the chimera parquet,
    aligned to the same window. Native cadences only (1d/4h/1h/30m/15m). For the VOLUME family."""
    import glob
    import polars as pl
    s_ms = pd.Timestamp(WIN[0]).value // 10**6; e_ms = pd.Timestamp(WIN[1]).value // 10**6
    split_ms = pd.Timestamp(SPLIT).value // 10**6
    vw = VOLWIN[cad]; assets = []; rv_meds = []
    for sym in SYMS:
        fs = sorted(glob.glob(f"data/processed/chimera/{cad}/{sym.lower()}*.parquet"))
        if not fs:
            continue
        df = pl.read_parquet(fs[-1], columns=["timestamp", "open", "high", "low", "close",
                                              "volume", "buy_vol", "sell_vol"]).sort("timestamp")
        ms = df["timestamp"].to_numpy()
        e = int(np.searchsorted(ms, e_ms)); s0 = max(0, int(np.searchsorted(ms, s_ms)) - WARMUP)
        sl = slice(s0, e)
        c2 = df["close"].to_numpy()[sl]; ms2 = ms[sl]
        if len(c2) < 40:
            continue
        win = ms2 >= s_ms
        if win.sum() < 30:
            continue
        ret = np.zeros(len(c2)); ret[1:] = c2[1:] / c2[:-1] - 1.0
        rv = pd.Series(ret).rolling(vw, min_periods=max(3, vw // 3)).std().shift(1).to_numpy()
        assets.append({"o": df["open"].to_numpy()[sl], "h": df["high"].to_numpy()[sl], "l": df["low"].to_numpy()[sl],
                       "c": c2, "vol": df["volume"].to_numpy()[sl], "buy_vol": df["buy_vol"].to_numpy()[sl],
                       "sell_vol": df["sell_vol"].to_numpy()[sl], "ret": ret, "win": win,
                       "idx": pd.to_datetime(ms2[win], unit="ms"), "rv": rv})
        vm = (ms2 >= s_ms) & (ms2 < split_ms)
        rv_meds.append(np.nanmedian(rv[vm]) if vm.sum() > 5 else np.nan)
    vt = float(np.nanmedian(rv_meds)) if rv_meds and not np.all(np.isnan(rv_meds)) else None
    return assets, vt


# ============================ the common stack -> book -> metrics ============================
def _book(assets, held_fn, params, vt, minhold=12):
    """held_fn(A) -> signal-level held {0,1} for one asset. Apply stack (trail10 + min_hold) + optional vol-target.
    minhold is per-family (trend=12 / mean-reversion=6: MR reversions are quick, a long min-hold over-holds)."""
    nets, tin, ntr = [], [], []
    for A in assets:
        c2, ret, win, idx, rv = A["c"], A["ret"], A["win"], A["idx"], A["rv"]
        held0 = held_fn(A, params)
        held = min_hold(apply_trail_stop(held0.copy().astype(np.int8), c2, 0.10)[0].astype(np.int8), minhold).astype(np.float64)
        # n_trades (round-trips in window)
        d = np.diff(np.concatenate([[0], held[win].astype(int), [0]]))
        ntr.append(int(min((d == 1).sum(), (d == -1).sum())))
        pos = np.zeros(len(c2)); pos[1:] = held[:-1]
        if vt is not None:
            pos = pos * np.clip(vt / (np.nan_to_num(rv, nan=vt) + 1e-12), 0.0, 1.0)
        flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
        nets.append(pd.Series((pos * ret - flips * (MAKER_RT / 2.0))[win], index=idx))
        s = pd.Series(pos[win], index=idx); s = s[s.index >= pd.Timestamp(SPLIT)]
        if len(s):
            tin.append(float(s.mean()))
    if not nets:
        return None
    b = pd.concat(nets, axis=1).fillna(0.0).mean(axis=1)
    d = b.resample("1D").apply(lambda x: float(np.prod(1 + x) - 1)).dropna()
    return d, (round(float(np.mean(tin)), 2) if tin else None), (round(float(np.mean(ntr)), 1) if ntr else None)


def _metrics(d, tin, ntr):
    val = d[d.index < pd.Timestamp(SPLIT)].to_numpy(); oos = d[d.index >= pd.Timestamp(SPLIT)].to_numpy()
    if len(oos) < 5:
        return None
    eq = np.cumprod(1 + oos); pk = np.maximum.accumulate(eq); dd = float(((eq - pk) / pk).min() * 100)
    vn = float((np.prod(1 + val) - 1) * 100) if len(val) else 0.0
    on = float((eq[-1] - 1) * 100)
    return {"val_net": round(vn, 1), "net": round(on, 1), "worst": round(min(vn, on), 1),
            "sharpe": round(float(np.mean(oos) / (np.std(oos) + 1e-12) * np.sqrt(365)), 2),
            "maxdd": round(dd, 1), "drift": round(on - vn, 1), "robust": bool(abs(on - vn) <= DRIFT_TOL),
            "time_in": tin, "n_trades": ntr}


def _buyhold(assets, vt=None):
    nets = []
    for A in assets:
        ret, win, idx, rv = A["ret"], A["win"], A["idx"], A["rv"]
        pos = np.ones(len(ret))
        if vt is not None:
            pos = np.clip(vt / (np.nan_to_num(rv, nan=vt) + 1e-12), 0.0, 1.0)
        nets.append(pd.Series((pos * ret)[win], index=idx))
    b = pd.concat(nets, axis=1).fillna(0.0).mean(axis=1)
    d = b.resample("1D").apply(lambda x: float(np.prod(1 + x) - 1)).dropna()
    oos = d[d.index >= pd.Timestamp(SPLIT)].to_numpy(); eq = np.cumprod(1 + oos); pk = np.maximum.accumulate(eq)
    return {"net": round(float((eq[-1] - 1) * 100), 1),
            "sharpe": round(float(np.mean(oos) / (np.std(oos) + 1e-12) * np.sqrt(365)), 2),
            "maxdd": round(float(((eq - pk) / pk).min() * 100), 1)}


# ============================ INDICATORS (base_fn + iron_fn + grid) ============================
def _macd_lines(c, fast, slow, sig):
    macd = _ema(c, fast) - _ema(c, slow)
    return macd, _ema(macd, sig)


def macd_base(A, p):
    """LONG when MACD line > signal line (the classic crossover)."""
    fast, slow, sig = p
    macd, sigl = _macd_lines(A["c"], fast, slow, sig)
    return np.nan_to_num(macd > sigl).astype(np.int8)


def macd_iron(A, p):
    """IRON: signal-cross AND zero-line filter (macd>0 = fast EMA>slow EMA -> only long in up-trend).
    The classic MACD chop-killer (deep-research-confirmed: zero-line/trend filter removes ranging whipsaw)."""
    fast, slow, sig = p
    macd, sigl = _macd_lines(A["c"], fast, slow, sig)
    return np.nan_to_num((macd > sigl) & (macd > 0)).astype(np.int8)


def macd_grid():
    g = []
    for fast in (5, 8, 12, 19, 26):
        for slow in (26, 35, 52, 75, 100):
            if slow < fast * 1.6:
                continue
            for sig in (9, 14, 21):
                g.append((fast, slow, sig))
    return g


# ---- mean-reversion oscillators (RSI / Stochastic / Bollinger %b / CCI) ----
# base: buy oversold (val<lo), exit reverted (val>hi). iron: ONLY buy the dip in an UPTREND (close>SMA(100))
# -- the standard fix for "oversold stays oversold in a downtrend" (the MR killer) -- + vol-target overlay.
_OSC = {"RSI": _rsi, "BBPCT": _bbpct}                       # close-only
_OSC_HL = {"STOCH": _stoch, "CCI": _cci}                    # need high/low


def _osc_val(kind, A, n):
    if kind in _OSC:
        return _OSC[kind](A["c"], n)
    return _OSC_HL[kind](A["c"], A["h"], A["l"], n)


def _mk_osc(kind):
    def base(A, p):
        n, lo, hi = p
        return _mr_held(_osc_val(kind, A, n), lo, hi)

    def iron(A, p):
        n, lo, hi = p
        held = _mr_held(_osc_val(kind, A, n), lo, hi)
        up = (A["c"] > _sma(A["c"], 100)).astype(np.int8)   # buy-the-dip-in-uptrend filter
        return (held & up).astype(np.int8)
    return base, iron


def _osc_grid(kind):
    if kind == "RSI":
        return [(n, lo, hi) for n in (7, 14, 21) for lo in (25, 30, 35) for hi in (55, 60, 65)]
    if kind == "STOCH":
        return [(n, lo, hi) for n in (14, 21) for lo in (15, 20, 25) for hi in (55, 70, 80)]
    if kind == "BBPCT":
        return [(n, lo, hi) for n in (14, 20) for lo in (0.0, 0.1, 0.2) for hi in (0.5, 0.7, 0.9)]
    if kind == "CCI":
        return [(n, lo, hi) for n in (14, 20) for lo in (-150, -100, -80) for hi in (0, 80, 100)]
    return []


# ---- breakout (Donchian channel): enter on N-bar high breakout, exit on M-bar low ----
def _atr(h, l, c, n):
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).rolling(n, min_periods=1).mean().to_numpy()


def _donchian(A, p, atr_margin=0.0):
    """LONG on close > prior N-bar high (+ optional atr_margin*ATR confirmation); exit on close < prior M-bar low."""
    entry_n, exit_m = p[0], p[1]
    h, l, c = A["h"], A["l"], A["c"]
    hh = pd.Series(h).rolling(entry_n, min_periods=1).max().shift(1).to_numpy()
    ll = pd.Series(l).rolling(exit_m, min_periods=1).min().shift(1).to_numpy()
    thr = hh + (atr_margin * _atr(h, l, c, entry_n) if atr_margin else 0.0)
    held = np.zeros(len(c), np.int8); cur = 0
    for i in range(len(c)):
        if np.isnan(thr[i]) or np.isnan(ll[i]):
            continue
        if cur == 0 and c[i] > thr[i]:
            cur = 1
        elif cur == 1 and c[i] < ll[i]:
            cur = 0
        held[i] = cur
    return held


def brk_base(A, p):
    return _donchian(A, p, 0.0)


def brk_iron(A, p):
    """IRON: require the breakout to clear the channel by 0.5*ATR (kills marginal FALSE breakouts) -- the
    classic Donchian/turtle volatility-confirmation -- + vol-target overlay (pipeline)."""
    return _donchian(A, p, 0.5)


def brk_grid():
    return [(n, m) for n in (10, 20, 30, 55) for m in (5, 10, 20) if m < n]


# ---- Supertrend (ATR-band trend) -- the crypto-popular trend indicator, distinct from MA/MACD ----
def _supertrend_dir(A, p):
    """Standard recursive Supertrend. Returns held{0,1} = 1 in up-trend. p = (atr_period, multiplier)."""
    n, mult = p
    h, l, c = A["h"], A["l"], A["c"]
    atr = _atr(h, l, c, n); hl2 = (h + l) / 2.0
    upper = hl2 + mult * atr; lower = hl2 - mult * atr
    fu = upper.copy(); fl = lower.copy()
    for i in range(1, len(c)):
        fu[i] = upper[i] if (upper[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lower[i] if (lower[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
    held = np.zeros(len(c), np.int8); d = 1
    for i in range(1, len(c)):
        if c[i] > fu[i - 1]:
            d = 1
        elif c[i] < fl[i - 1]:
            d = -1
        held[i] = 1 if d == 1 else 0
    return held


def st_base(A, p):
    return _supertrend_dir(A, p)


def st_iron(A, p):
    """IRON: Supertrend up AND a slower-trend confirm (close>SMA100) -- gate out fast flip-flops -- + vol-target."""
    held = _supertrend_dir(A, p)
    up = (A["c"] > _sma(A["c"], 100)).astype(np.int8)
    return (held & up).astype(np.int8)


def st_grid():
    return [(n, m) for n in (7, 10, 14, 21) for m in (1.5, 2.0, 3.0, 4.0)]


# ---- ROC momentum (time-series momentum: long when N-bar rate-of-change > threshold) ----
def _roc_held(A, p):
    n, thr = p
    c = A["c"]
    roc = np.full(len(c), np.nan); roc[n:] = (c[n:] / c[:-n] - 1.0) * 100
    return np.nan_to_num(roc > thr).astype(np.int8)


def roc_base(A, p):
    return _roc_held(A, p)


def roc_iron(A, p):
    """IRON: positive momentum AND uptrend confirm (close>SMA100) -- avoid a positive ROC inside a downtrend -- + vol-target."""
    held = _roc_held(A, p)
    up = (A["c"] > _sma(A["c"], 100)).astype(np.int8)
    return (held & up).astype(np.int8)


def roc_grid():
    return [(n, thr) for n in (10, 20, 50, 100) for thr in (0.0, 2.0, 5.0, 10.0)]


# ---- Parabolic SAR (trailing-stop trend) ----
def _psar(A, p):
    af0, afmax = p
    h, l, c = A["h"], A["l"], A["c"]; n = len(c)
    held = np.zeros(n, np.int8)
    up = True; af = af0; ep = h[0]; sar = l[0]
    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if up:
            sar = min(sar, l[i - 1], l[max(0, i - 2)])
            if h[i] > ep:
                ep = h[i]; af = min(af + af0, afmax)
            if l[i] < sar:
                up = False; sar = ep; ep = l[i]; af = af0
        else:
            sar = max(sar, h[i - 1], h[max(0, i - 2)])
            if l[i] < ep:
                ep = l[i]; af = min(af + af0, afmax)
            if h[i] > sar:
                up = True; sar = ep; ep = h[i]; af = af0
        held[i] = 1 if up else 0
    return held


def psar_base(A, p):
    return _psar(A, p)


def psar_iron(A, p):
    held = _psar(A, p); up = (A["c"] > _sma(A["c"], 100)).astype(np.int8)
    return (held & up).astype(np.int8)


def psar_grid():
    return [(a, m) for a in (0.01, 0.02, 0.04) for m in (0.1, 0.2, 0.3)]


# ---- Williams %R (mean-reversion, inverted stochastic) ----
def _willr(A, p):
    n, lo, hi = p
    h, l, c = A["h"], A["l"], A["c"]
    hh = pd.Series(h).rolling(n, min_periods=1).max().to_numpy(); ll = pd.Series(l).rolling(n, min_periods=1).min().to_numpy()
    wr = -100 * (hh - c) / (hh - ll + 1e-12)
    return _mr_held(wr, lo, hi)


def willr_base(A, p):
    return _willr(A, p)


def willr_iron(A, p):
    held = _willr(A, p); up = (A["c"] > _sma(A["c"], 100)).astype(np.int8)
    return (held & up).astype(np.int8)


def willr_grid():
    return [(n, lo, hi) for n in (14, 21) for lo in (-90, -80) for hi in (-50, -30, -20)]


# ---- Keltner channel breakout (volatility breakout) ----
def _keltner(A, p, atr_margin=0.0):
    n, mult = p
    h, l, c = A["h"], A["l"], A["c"]
    mid = _ema(c, n); atr = _atr(h, l, c, n); upper = mid + mult * atr
    thr = upper + atr_margin * atr
    held = np.zeros(len(c), np.int8); cur = 0
    for i in range(len(c)):
        if cur == 0 and c[i] > thr[i]:
            cur = 1
        elif cur == 1 and c[i] < mid[i]:
            cur = 0
        held[i] = cur
    return held


def kelt_base(A, p):
    return _keltner(A, p, 0.0)


def kelt_iron(A, p):
    held = _keltner(A, p, 0.0); up = (A["c"] > _sma(A["c"], 100)).astype(np.int8)
    return (held & up).astype(np.int8)


def kelt_grid():
    return [(n, m) for n in (20, 30) for m in (1.5, 2.0, 2.5)]


# ---- Vortex (trend), TSI (momentum), ADX/DI (trend, regime-gated iron) ----
def _vortex(A, p):
    n = p[0]; h, l, c = A["h"], A["l"], A["c"]
    pl_ = np.concatenate([[l[0]], l[:-1]]); ph = np.concatenate([[h[0]], h[:-1]])
    vmp = np.abs(h - pl_); vmm = np.abs(l - ph); tr = _atr(h, l, c, 1)
    sp = pd.Series(vmp).rolling(n, min_periods=1).sum().to_numpy()
    sm = pd.Series(vmm).rolling(n, min_periods=1).sum().to_numpy()
    st_ = pd.Series(tr).rolling(n, min_periods=1).sum().to_numpy() + 1e-12
    return np.nan_to_num((sp / st_) > (sm / st_)).astype(np.int8)


def vortex_base(A, p):
    return _vortex(A, p)


def vortex_iron(A, p):
    held = _vortex(A, p); up = (A["c"] > _sma(A["c"], 100)).astype(np.int8)
    return (held & up).astype(np.int8)


def _tsi_val(A, p):
    r, s = p[0], p[1]; c = A["c"]; m = np.diff(c, prepend=c[0])
    e2 = _ema(_ema(m, r), s); a2 = _ema(_ema(np.abs(m), r), s)
    return 100 * e2 / (a2 + 1e-12)


def tsi_base(A, p):
    return np.nan_to_num(_tsi_val(A, p) > 0).astype(np.int8)


def tsi_iron(A, p):
    held = np.nan_to_num(_tsi_val(A, p) > 0).astype(np.int8); up = (A["c"] > _sma(A["c"], 100)).astype(np.int8)
    return (held & up).astype(np.int8)


def _adx_di(A, n):
    h, l, c = A["h"], A["l"], A["c"]
    up = h - np.concatenate([[h[0]], h[:-1]]); dn = np.concatenate([[l[0]], l[:-1]]) - l
    pdm = np.where((up > dn) & (up > 0), up, 0.0); mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr = pd.Series(_atr(h, l, c, 1)).rolling(n, min_periods=1).mean().to_numpy() + 1e-12
    pdi = 100 * pd.Series(pdm).rolling(n, min_periods=1).mean().to_numpy() / atr
    mdi = 100 * pd.Series(mdm).rolling(n, min_periods=1).mean().to_numpy() / atr
    dx = 100 * np.abs(pdi - mdi) / (pdi + mdi + 1e-12)
    adx = pd.Series(dx).rolling(n, min_periods=1).mean().to_numpy()
    return pdi, mdi, adx


def adx_base(A, p):
    pdi, mdi, adx = _adx_di(A, p[0])
    return np.nan_to_num(pdi > mdi).astype(np.int8)


def adx_iron(A, p):
    """IRON = the REGIME GATE: long +DI>-DI ONLY when ADX>threshold (trending regime) -- the evidence-backed
    iron (deep-research: trend signals work in trending/greed regimes, break in chop/fear) -- + vol-target."""
    n, thr = p; pdi, mdi, adx = _adx_di(A, n)
    return np.nan_to_num((pdi > mdi) & (adx > thr)).astype(np.int8)


# ---- VOLUME / ORDER-FLOW family (OBV trend, MFI mean-rev, buy/sell imbalance) ----
def _obv(A, p):
    n = p[0]; c = A["c"]; vol = A["vol"]
    obv = np.cumsum(np.sign(np.diff(c, prepend=c[0])) * np.nan_to_num(vol))
    return np.nan_to_num(obv > _ema(obv, n)).astype(np.int8)


def obv_base(A, p):
    return _obv(A, p)


def obv_iron(A, p):
    held = _obv(A, p); up = (A["c"] > _sma(A["c"], 100)).astype(np.int8)
    return (held & up).astype(np.int8)


def _mfi_val(A, n):
    h, l, c, vol = A["h"], A["l"], A["c"], np.nan_to_num(A["vol"])
    tp = (h + l + c) / 3.0; mf = tp * vol; dtp = np.diff(tp, prepend=tp[0])
    pos = pd.Series(np.where(dtp > 0, mf, 0.0)).rolling(n, min_periods=1).sum().to_numpy()
    neg = pd.Series(np.where(dtp < 0, mf, 0.0)).rolling(n, min_periods=1).sum().to_numpy()
    return 100 - 100 / (1 + pos / (neg + 1e-12))


def mfi_base(A, p):
    n, lo, hi = p
    return _mr_held(_mfi_val(A, n), lo, hi)


def mfi_iron(A, p):
    n, lo, hi = p
    held = _mr_held(_mfi_val(A, n), lo, hi); up = (A["c"] > _sma(A["c"], 100)).astype(np.int8)
    return (held & up).astype(np.int8)


def _volimb(A, p):
    """Order-flow: smoothed taker buy-share = buy_vol/(buy_vol+sell_vol); long when buying pressure > thr."""
    n, thr = p
    bv = np.nan_to_num(A["buy_vol"]); sv = np.nan_to_num(A["sell_vol"])
    imb = bv / (bv + sv + 1e-12)
    imb_s = pd.Series(imb).rolling(n, min_periods=1).mean().to_numpy()
    return (imb_s > thr).astype(np.int8)


def volimb_base(A, p):
    return _volimb(A, p)


def volimb_iron(A, p):
    held = _volimb(A, p); up = (A["c"] > _sma(A["c"], 100)).astype(np.int8)
    return (held & up).astype(np.int8)


def _cmf(A, p):
    n = p[0]; h, l, c, vol = A["h"], A["l"], A["c"], np.nan_to_num(A["vol"])
    mfm = ((c - l) - (h - c)) / ((h - l) + 1e-12); mfv = mfm * vol
    sm = pd.Series(mfv).rolling(n, min_periods=1).sum().to_numpy()
    sv = pd.Series(vol).rolling(n, min_periods=1).sum().to_numpy() + 1e-12
    return sm / sv


def cmf_base(A, p):
    return (np.nan_to_num(_cmf(A, p)) > 0).astype(np.int8)


def cmf_iron(A, p):
    held = (np.nan_to_num(_cmf(A, p)) > 0).astype(np.int8); up = (A["c"] > _sma(A["c"], 100)).astype(np.int8)
    return (held & up).astype(np.int8)


INDICATORS = {
    "MACD": {"family": "trend", "base": macd_base, "iron": macd_iron, "grid": macd_grid, "minhold": 12,
             "name": lambda p: f"MACD({p[0]},{p[1]},{p[2]})"},
    "VORTEX": {"family": "trend", "base": vortex_base, "iron": vortex_iron, "minhold": 12,
               "grid": lambda: [(n,) for n in (7, 14, 21, 28)], "name": lambda p: f"VORTEX({p[0]})"},
    "ADX": {"family": "trend", "base": adx_base, "iron": adx_iron, "minhold": 12,
            "grid": lambda: [(n, thr) for n in (14, 21) for thr in (20, 25, 30)], "name": lambda p: f"ADX({p[0]},{p[1]})"},
    "TSI": {"family": "momentum", "base": tsi_base, "iron": tsi_iron, "minhold": 12,
            "grid": lambda: [(r, s) for r, s in ((25, 13), (13, 7), (40, 20), (20, 10))], "name": lambda p: f"TSI({p[0]},{p[1]})"},
    "CMF": {"family": "volume", "base": cmf_base, "iron": cmf_iron, "loader": "ohlcv", "minhold": 12,
            "grid": lambda: [(n,) for n in (14, 20, 50)], "name": lambda p: f"CMF({p[0]})"},
    "OBV": {"family": "volume", "base": obv_base, "iron": obv_iron, "loader": "ohlcv", "minhold": 12,
            "grid": lambda: [(n,) for n in (10, 20, 50, 100)], "name": lambda p: f"OBV({p[0]})"},
    "MFI": {"family": "volume", "base": mfi_base, "iron": mfi_iron, "loader": "ohlcv", "minhold": 6,
            "grid": lambda: [(n, lo, hi) for n in (14, 21) for lo in (20, 30) for hi in (55, 70, 80)],
            "name": lambda p: f"MFI({p[0]},lo{p[1]},hi{p[2]})"},
    "VOLIMB": {"family": "volume", "base": volimb_base, "iron": volimb_iron, "loader": "ohlcv", "minhold": 12,
               "grid": lambda: [(n, thr) for n in (3, 7, 14, 28) for thr in (0.50, 0.52, 0.55)],
               "name": lambda p: f"VOLIMB({p[0]},thr{p[1]})"},
    "SUPERTREND": {"family": "trend", "base": st_base, "iron": st_iron, "grid": st_grid, "minhold": 12,
                   "name": lambda p: f"ST({p[0]},{p[1]})"},
    "PSAR": {"family": "trend", "base": psar_base, "iron": psar_iron, "grid": psar_grid, "minhold": 12,
             "name": lambda p: f"PSAR({p[0]},{p[1]})"},
    "ROC": {"family": "momentum", "base": roc_base, "iron": roc_iron, "grid": roc_grid, "minhold": 12,
            "name": lambda p: f"ROC({p[0]},thr{p[1]})"},
    "DONCHIAN": {"family": "breakout", "base": brk_base, "iron": brk_iron, "grid": brk_grid, "minhold": 12,
                 "name": lambda p: f"DONCH({p[0]},{p[1]})"},
    "KELTNER": {"family": "breakout", "base": kelt_base, "iron": kelt_iron, "grid": kelt_grid, "minhold": 12,
                "name": lambda p: f"KELT({p[0]},{p[1]})"},
    "WILLR": {"family": "mean-reversion", "base": willr_base, "iron": willr_iron, "grid": willr_grid, "minhold": 6,
              "name": lambda p: f"WILLR({p[0]},lo{p[1]},hi{p[2]})"},
}
for _k in ("RSI", "STOCH", "BBPCT", "CCI"):
    _b, _i = _mk_osc(_k)
    INDICATORS[_k] = {"family": "mean-reversion", "base": _b, "iron": _i,
                      "grid": (lambda kk=_k: (lambda: _osc_grid(kk)))(), "minhold": 6,
                      "name": (lambda kk=_k: (lambda p: f"{kk}({p[0]},lo{p[1]},hi{p[2]})"))()}


# ============================ run one indicator over a cadence ============================
def run_indicator(ind_key, cads):
    ind = INDICATORS[ind_key]
    loader = load_ohlcv if ind.get("loader") == "ohlcv" else load_ohlc
    export = {}
    for cad in cads:
        assets, vt = loader(cad)
        if not assets:
            print(f"## {cad}: no assets"); continue
        bh = _buyhold(assets); vbh = _buyhold(assets, vt)
        mh = ind.get("minhold", 12)
        rows = []
        for p in ind["grid"]():
            rb = _book(assets, ind["base"], p, None, mh)
            ri = _book(assets, ind["iron"], p, vt, mh)
            if rb is None or ri is None:
                continue
            mb = _metrics(*rb); mi = _metrics(*ri)
            if mb is None or mi is None:
                continue
            rows.append({"cfg": ind["name"](p), "base": mb, "iron": mi})
        if not rows:
            continue
        export[cad] = {"buyhold": bh, "voltgt_bh": vbh, "rows": rows}
        # ---- print: top-10 by WEALTH (ironed OOS net), base vs ironed side by side ----
        print(f"\n########## {ind_key} @ {cad}   BUYHOLD net {bh['net']}% Sh {bh['sharpe']} DD {bh['maxdd']}% | "
              f"VOLTGT_BH net {vbh['net']}% Sh {vbh['sharpe']} ##########")
        rows.sort(key=lambda r: -r["iron"]["net"])
        print(f"   TOP-10 by WEALTH (ironed OOS net). cfg | BASE net/Sh/DD/drift | IRONED net/Sh/DD/drift/robust")
        for r in rows[:10]:
            b, i = r["base"], r["iron"]
            print(f"   {r['cfg']:20} | base {b['net']:>6}/{b['sharpe']:>4}/{b['maxdd']:>6}/{b['drift']:>5} "
                  f"| IRON {i['net']:>6}/{i['sharpe']:>4}/{i['maxdd']:>6}/{i['drift']:>5}/{'R' if i['robust'] else '-'}")
        # ---- robust/non-robust split (ironed) ----
        rob = [r for r in rows if r["iron"]["robust"]]
        nonrob = [r for r in rows if not r["iron"]["robust"]]
        # base-vs-iron aggregate improvement
        import statistics as st
        d_net = st.median([r["iron"]["net"] - r["base"]["net"] for r in rows])
        d_dd = st.median([r["iron"]["maxdd"] - r["base"]["maxdd"] for r in rows])
        d_dr = st.median([abs(r["iron"]["drift"]) - abs(r["base"]["drift"]) for r in rows])
        print(f"   SPLIT: {len(rob)} robust / {len(nonrob)} non-robust (ironed). "
              f"IRON vs BASE (median): dNet {d_net:+.1f}pp, dMaxDD {d_dd:+.1f}pp, d|drift| {d_dr:+.1f}pp")
    op = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
    op.mkdir(parents=True, exist_ok=True)
    jt = "_".join(cads)
    out = op / f"ti_{ind_key}_{jt}.json"
    json.dump(export, open(out, "w"), indent=1, default=str)
    print(f"\n[json] {out}")
    return 0


def main() -> int:
    ind = "MACD"
    if "--indicator" in sys.argv:
        ind = sys.argv[sys.argv.index("--indicator") + 1]
    cads = ["1d", "4h"]
    if "--cadences" in sys.argv:
        cads = sys.argv[sys.argv.index("--cadences") + 1].split(",")
    if ind not in INDICATORS:
        print(f"unknown indicator {ind}; have {list(INDICATORS)}"); return 1
    print(f"TI PIPELINE: {ind} ({INDICATORS[ind]['family']}) | {len(INDICATORS[ind]['grid']())} configs x {len(cads)} TF "
          f"| BASE vs IRONED | WEALTH-ranked | 2020 fixed-EW long-only\n")
    return run_indicator(ind, cads)


if __name__ == "__main__":
    sys.exit(main())
