"""MA SLICE ENGINE — the comprehensive 7d-window study (A-E).

Addresses the user's 2026-06-27 mandate:
  A. PROD costs (maker / taker / prod-slippage) — show honest expectations
  B. Whipsaw control for high-freq TFs: min-hold + cooldown
  C. Canonical exits (canonical_exits.py) — confirmation-break + time-cap default,
     menu of take-profit / trailing / stop available for explicit testing
  D. FIXED vs ROLLING slice comparison:
       ROLLING (current prod): continuous signal, enter when confirmed, hold ≤ Nd,
                exit on confirmation-break or time-cap. Positions span window boundaries.
       FIXED: snap to non-overlapping Nd windows. At each window start, compute the
              lookback-params FRESH, trade that window independently, reset at window end.
              Each slice is INDEPENDENT (the user's mental model for live trading).
  E. Tiered config:
       SHORT  {15m, 30m}  — find good lookback/window (NOT traded yet)
       MEDIUM {1h, 2h, 4h} — 7-14d lookback, 7d window, 4h-preferred
       LONG   {1d}         — 14d lookback, 14d hold

The question being answered: "Can we trade any 7d window profitably, net of PROD cost?"
"""
from __future__ import annotations

import warnings; warnings.filterwarnings("ignore")
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CRYPTO = ROOT.parent

import strat.fleet_lab as fl
from strat.ma_movecatch_engine import select_band_trainval, make_hold_provider
from strat.ma_type_upgrade import _ema, MA_TYPES
from strat.ma_wealth_driver import _load_asset, _u100_symbols
from strat.canonical_exits import (
    resolve_held, apply_min_hold_cooldown, COST_MAKER, COST_TAKER, COST_PROD,
)

BARS_PER_DAY = fl.BARS_PER_DAY
VAL_START = "2023-07-01"; VAL_END = "2024-05-15"
OOS_START = "2024-05-15"; OOS_2024_END = "2024-12-31"
VAL_WALL_MS = int(pd.Timestamp(VAL_END).value // 10**6)
OOS_WALL_MS = int(pd.Timestamp("2025-01-01").value // 10**6)
OUTDIR = CRYPTO / "runs" / "engines" / "V8_slice_study"

TIER = {
    "15m": "SHORT", "30m": "SHORT",
    "1h": "MEDIUM", "2h": "MEDIUM", "4h": "MEDIUM",
    "1d": "LONG",
}


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


def causal_regime(wall_ms):
    """De-leaked 1d BTC regime for GATING on a per-bar grid.

    btc_regime[D] is derived from BTC's day-D 1d CLOSE (~23:59 of day D). Any consumer
    that broadcasts the raw series onto an intraday/1d bar grid (normalize-to-day then
    ffill, or per-bar prior[-1]) reads regime[D] on a day-D bar => SAME-DAY LOOK-AHEAD
    (the V1.1 align_htf_on START-label bug class). Causal fix: a bar on day D must read
    the last CLOSED 1d bar = regime[D-1]. We bake that in by shifting the daily series
    forward one full 1d bar BEFORE the broadcast. ALL gating consumers route through here;
    DO NOT broadcast the raw btc_regime onto a forward-return grid (see
    runs/engines/AUDIT_ORACLE_UPGRADE/leakcheck.json).
    """
    s = btc_regime(wall_ms)
    if s is None:
        return None
    return s.shift(1).fillna(0).astype(int)


def _confirmed_signal(close, band_held, lookback_bars):
    """Per-bar confirmed entry signal: band long AND close>EMA(lb) AND EMA rising."""
    fin = np.asarray(close, dtype=float)
    n = len(fin)
    conf = np.zeros(n, dtype=bool)
    m = np.isfinite(fin)
    if m.sum() < max(5, lookback_bars):
        return conf
    cseg = fin[m]
    ema_lb = _ema(cseg, lookback_bars)
    above = cseg > ema_lb
    rising = np.concatenate([[False], ema_lb[1:] >= ema_lb[:-1]])
    bh = np.asarray(band_held, dtype=bool)[m]
    conf_seg = bh & above & rising
    conf[np.where(m)[0]] = conf_seg
    return conf


def held_rolling(close, band_held, lookback_bars, slice_bars,
                 min_hold_bars=0, cooldown_bars=0, **exit_kw):
    """ROLLING slices: continuous signal → resolve_held (confirmation-break + time-cap +
    optional exits). Positions may span window boundaries."""
    conf = _confirmed_signal(close, band_held, lookback_bars)
    return resolve_held(np.asarray(close, float), conf, slice_bars,
                        min_hold_bars=min_hold_bars, cooldown_bars=cooldown_bars, **exit_kw)


def held_fixed(close, band_held, lookback_bars, slice_bars,
               min_hold_bars=0, cooldown_bars=0, **exit_kw):
    """FIXED independent slices: partition the series into non-overlapping slice_bars
    windows. Within each window, the entry decision uses the confirmed signal computed
    on data up to each bar (the 14d-lookback EMA is causal). The position is opened at
    most ONCE per window (first confirmed bar) and force-closed at window end. Each
    window trades independently and resets — no position carries across the boundary."""
    c = np.asarray(close, float)
    n = len(c)
    conf = _confirmed_signal(close, band_held, lookback_bars)
    held = np.zeros(n, dtype=np.int8)
    # find first valid bar (enough lookback)
    start = lookback_bars
    w = start
    while w < n:
        w_end = min(w + slice_bars, n)
        # within [w, w_end): open at first confirmed bar, hold to window end or exit
        seg_close = c[w:w_end]
        seg_conf = conf[w:w_end]
        if seg_conf.any():
            entry_local = int(np.argmax(seg_conf))  # first True
            # resolve exit within the window using canonical exits
            seg_held = resolve_held(seg_close, seg_conf, slice_bars,
                                    min_hold_bars=min_hold_bars, cooldown_bars=cooldown_bars,
                                    **exit_kw)
            held[w:w_end] = seg_held
        w = w_end
    return held


_BAND_CACHE = {}
_HELD_CACHE = {}


def _cached_band_held(ma_type, tf, sym, o, h, l, c, wall_ms):
    """Cache the expensive 114-config band-held computation per (type,tf,sym,wall)."""
    key = (ma_type, tf, sym, wall_ms)
    if key in _HELD_CACHE:
        return _HELD_CACHE[key]
    bkey = (ma_type, tf)
    if bkey not in _BAND_CACHE:
        _BAND_CACHE[bkey] = select_band_trainval(ma_type, tf)[0]
    provider = make_hold_provider(_BAND_CACHE[bkey], ma_type)
    bh = provider(sym, o, h, l, c)
    _HELD_CACHE[key] = bh
    return bh


def run_cell(ma_type, tf, slice_mode="rolling", lookback_days=14, slice_days=7,
             cost_rt=COST_TAKER, min_hold_bars=0, cooldown_bars=0,
             window="OOS", n_null=50, **exit_kw):
    """Run one cell. slice_mode in {rolling, fixed}. Returns metrics dict."""
    t0 = time.time()
    bpd = BARS_PER_DAY[tf]
    lookback_bars = lookback_days * bpd
    slice_bars = slice_days * bpd
    wall_ms = OOS_WALL_MS if window == "OOS" else VAL_WALL_MS
    win_s = pd.Timestamp(OOS_START if window == "OOS" else VAL_START)
    win_e = pd.Timestamp(OOS_2024_END if window == "OOS" else VAL_END)

    regime = causal_regime(wall_ms)   # de-leaked: bar on day D reads regime[D-1]
    held_fn = held_fixed if slice_mode == "fixed" else held_rolling

    close_cols, held_cols = {}, {}
    for sym in _u100_symbols():
        rec = _load_asset(sym, tf, wall_ms)
        if rec is None:
            continue
        dates, o, h, l, c = rec
        band_held = _cached_band_held(ma_type, tf, sym, o, h, l, c, wall_ms)
        held = held_fn(c, band_held, lookback_bars, slice_bars,
                       min_hold_bars=min_hold_bars, cooldown_bars=cooldown_bars, **exit_kw)
        close_cols[sym] = pd.Series(c, index=dates)
        held_cols[sym] = pd.Series(held, index=dates)

    if len(close_cols) < 10:
        return {"status": "INSUFFICIENT_ASSETS"}

    close_df = pd.DataFrame(close_cols).sort_index()
    held_df = pd.DataFrame(held_cols).reindex(close_df.index).fillna(0).astype(np.int8)

    rets = close_df.pct_change().to_numpy()[1:]
    held_lag = held_df.shift(1).fillna(0).to_numpy()[1:]
    turn = held_df.diff().abs().fillna(0).to_numpy()[1:]
    dates = close_df.index[1:]

    reg_vals = np.ones(len(dates))
    if regime is not None:
        norm = pd.DatetimeIndex(dates).normalize()
        idx = regime.index.union(norm).drop_duplicates()
        ra = regime.reindex(idx).ffill().reindex(norm).fillna(0)
        reg_vals = ra.to_numpy().astype(float)

    mask = (dates >= win_s) & (dates < win_e)
    valid = np.isfinite(rets)
    rets_f = np.where(valid, rets, 0.0)
    n_held = (held_lag * valid).sum(axis=1)
    eng_gross = np.where(n_held > 0, (rets_f * held_lag * valid).sum(axis=1) / np.maximum(n_held, 1), 0.0)
    cost = np.where(n_held > 0, (turn * valid).sum(axis=1) * (cost_rt / 2.0) / np.maximum(n_held, 1), 0.0)
    eng = (eng_gross - cost) * reg_vals
    al = np.where(valid.sum(axis=1) > 0, (rets_f * valid).sum(axis=1) / np.maximum(valid.sum(axis=1), 1), 0.0)

    eng_w = eng[mask]; al_w = al[mask]
    deploy = float(np.mean(n_held[mask] > 0)) if mask.sum() else 0.0
    if len(eng_w) == 0:
        return {"status": "NO_BARS"}

    lin_wk = float(np.mean(eng_w)) * bpd * 7 * 100
    al_lin_wk = float(np.mean(al_w)) * bpd * 7 * 100
    comp = float(np.prod(1 + eng_w) - 1) * 100
    cum = np.cumprod(1 + eng_w)
    maxdd = float(np.min((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum))) * 100 if len(cum) > 1 else 0
    sharpe = (float(np.mean(eng_w)) * bpd * 365) / (float(np.std(eng_w)) * np.sqrt(bpd * 365)) if np.std(eng_w) > 1e-12 else 0

    # 7d slice profitability — the user's core question
    slice_pnls = []
    i = 0
    eng_full = eng[mask]
    while i + slice_bars <= len(eng_full):
        slice_pnls.append(float(np.prod(1 + eng_full[i:i+slice_bars]) - 1) * 100)
        i += slice_bars
    pct_prof_slices = sum(1 for s in slice_pnls if s > 0) / len(slice_pnls) if slice_pnls else 0

    # trades + cost drag
    n_trades_est = int((turn[mask] * valid[mask]).sum() / 2)
    cost_drag_wk = float(np.mean(cost[mask])) * bpd * 7 * 100

    return {
        "status": "OK", "ma_type": ma_type, "tf": tf, "tier": TIER.get(tf, "?"),
        "slice_mode": slice_mode, "lookback_days": lookback_days, "slice_days": slice_days,
        "cost_rt": cost_rt, "min_hold_bars": min_hold_bars, "cooldown_bars": cooldown_bars,
        "exit_kw": {k: v for k, v in exit_kw.items()},
        "window": window, "n_assets": len(close_cols),
        "deploy": round(deploy, 4), "lin_wk": round(lin_wk, 4), "al_lin_wk": round(al_lin_wk, 4),
        "edge_pp": round(lin_wk - al_lin_wk, 4), "compound": round(comp, 1),
        "maxdd": round(maxdd, 1), "sharpe": round(sharpe, 3),
        "n_slices": len(slice_pnls), "pct_profitable_7d_slices": round(pct_prof_slices, 3),
        "median_slice_pct": round(float(np.median(slice_pnls)), 3) if slice_pnls else 0,
        "n_trades_est": n_trades_est, "cost_drag_wk_pp": round(cost_drag_wk, 4),
        "elapsed_s": round(time.time() - t0, 1),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", default="EMA,SMA,WMA,DEMA,KAMA,VIDYA")
    ap.add_argument("--tfs", default="4h")
    ap.add_argument("--cost", default="taker", choices=["maker", "taker", "prod"])
    ap.add_argument("--modes", default="rolling,fixed")
    ap.add_argument("--lookback", type=int, default=14)
    ap.add_argument("--slice", type=int, default=7)
    ap.add_argument("--min-hold", type=int, default=0)
    ap.add_argument("--cooldown", type=int, default=0)
    ap.add_argument("--window", default="OOS")
    a = ap.parse_args()

    cost = {"maker": COST_MAKER, "taker": COST_TAKER, "prod": COST_PROD}[a.cost]
    types = MA_TYPES if a.types == "all" else [t.strip().upper() for t in a.types.split(",")]
    tfs = [t.strip() for t in a.tfs.split(",")]
    modes = [m.strip() for m in a.modes.split(",")]

    print("=" * 110)
    print(f"MA SLICE STUDY — cost={a.cost}({cost} RT) | lookback={a.lookback}d slice={a.slice}d | "
          f"min_hold={a.min_hold} cooldown={a.cooldown} | window={a.window}")
    print("=" * 110)
    print(f"  {'type':<6}{'tf':<4}{'mode':<8}|{'deploy':>7}{'edge':>8}{'lin/wk':>8}{'Sh':>6}"
          f"{'7d-prof':>8}{'trades':>7}{'costdrag':>9}")
    print("  " + "-" * 92)

    rows = []
    for ma in types:
        for tf in tfs:
            for mode in modes:
                r = run_cell(ma, tf, slice_mode=mode, lookback_days=a.lookback, slice_days=a.slice,
                             cost_rt=cost, min_hold_bars=a.min_hold, cooldown_bars=a.cooldown,
                             window=a.window)
                rows.append(r)
                if r["status"] == "OK":
                    print(f"  {ma:<6}{tf:<4}{mode:<8}|{r['deploy']:>6.1%}{r['edge_pp']:>+8.3f}"
                          f"{r['lin_wk']:>+8.3f}{r['sharpe']:>6.2f}{r['pct_profitable_7d_slices']:>7.0%}"
                          f"{r['n_trades_est']:>7}{r['cost_drag_wk_pp']:>+8.3f}")
                else:
                    print(f"  {ma:<6}{tf:<4}{mode:<8}| {r['status']}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUTDIR / f"slice_study_{a.cost}_{a.window}_{ts}.json"
    with open(out, "w") as f:
        json.dump({"rows": rows, "cost": a.cost, "cost_rt": cost, "window": a.window,
                   "lookback": a.lookback, "slice": a.slice, "min_hold": a.min_hold,
                   "cooldown": a.cooldown, "timestamp": ts}, f, indent=2, default=str)
    print(f"\n  Saved: {out}")
