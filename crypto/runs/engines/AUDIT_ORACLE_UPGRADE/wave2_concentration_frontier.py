"""WAVE 2 (upgrade) -- the DE-LEAKED concentration frontier + the risk-adjusted operating point.

Oracle-gap finding: the single biggest RECOVERABLE headroom (V2 engine -> realizable ceiling)
is the per-name CAP (concentration), not entry/exit timing. The deployable headline used cap=0.15.
This sweeps the cap finely on the DE-LEAKED book (regime.shift(1)) and reports, per cap:
  - lin_wk (capture), Sharpe, maxDD, 7d %profitable  -> the risk/return frontier
  - cap_bind_frac: fraction of (bar,name) live weights actually clipped by the cap -> where the cap stops mattering
  - avg_gross_exposure, n_names_held

The make-or-break question this answers: is the cap a genuine DRAWDOWN dial (then keep it tight),
or just a RETURN tax that buys no drawdown protection (then loosen it -- the regime gate is the real
DD control)? OOS-2024, PROD cost, 2025 SEALED, long-only spot lev=1.

Reuses the AUDITED de-leaked path (audit_deployable_run.build_book) verbatim -- only the cap sweep
is finer + a cap-binding diagnostic is added.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(ROOT))
CRYPTO = Path(__file__).resolve().parents[3]

import strat.fleet_lab as fl
from strat.ma_wealth_driver import _load_asset, _u100_symbols
from strat.ma_slice_engine import held_rolling, _cached_band_held, btc_regime, OOS_WALL_MS, OOS_START, OOS_2024_END
from strat.canonical_exits import COST_PROD

BARS_PER_DAY = fl.BARS_PER_DAY
TIER_CFG = {"1d": {"lookback": 14, "slice": 14}, "4h": {"lookback": 14, "slice": 7}}
TYPES = ["SMA", "EMA", "WMA", "VIDYA", "KAMA"]


def build_book(tf, cap=0.15, deleak=True):
    """ma_slice_roi.build_book_net, DE-LEAKED by default (regime.shift(1)). Returns the net
    stream + breadth + the cap-binding fraction (how often the cap actually clips a live weight)."""
    cfg = TIER_CFG[tf]; bpd = BARS_PER_DAY[tf]
    lb, sl = cfg["lookback"]*bpd, cfg["slice"]*bpd
    reg = btc_regime(OOS_WALL_MS)
    if deleak:
        reg = reg.shift(1).fillna(0).astype(int)   # day-D reads the last CLOSED 1d regime bar
    closes = {}
    for sym in _u100_symbols():
        rec = _load_asset(sym, tf, OOS_WALL_MS)
        if rec is None:
            continue
        d, o, h, l, c = rec
        closes[sym] = (d, o, h, l, np.asarray(c, float))
    syms = list(closes.keys())
    close_df = pd.DataFrame({s: pd.Series(closes[s][4], index=closes[s][0]) for s in syms}).sort_index()
    assert close_df.index.max() < pd.Timestamp("2025-01-01"), "2025 LEAK"
    idx = close_df.index; T = len(idx); A = len(syms)
    conv = np.zeros((T, A))
    for ma in TYPES:
        hc = {}
        for s in syms:
            d, o, h, l, c = closes[s]
            bh = _cached_band_held(ma, tf, s, o, h, l, c, OOS_WALL_MS)
            hc[s] = pd.Series(held_rolling(c, bh, lb, sl), index=d)
        hd = pd.DataFrame(hc).reindex(idx).fillna(0).reindex(columns=close_df.columns).fillna(0).to_numpy()
        conv += hd
    ret = close_df.pct_change().fillna(0.0).to_numpy()
    vol = close_df.pct_change().rolling(max(20, bpd*5), min_periods=10).std().to_numpy()
    rg = np.ones(T); ridx = reg.index
    for k, dt in enumerate(idx):
        d1 = pd.Timestamp(dt).normalize(); prior = ridx[ridx <= d1]
        rg[k] = int(reg[prior[-1]]) if len(prior) else 0
    mult = (conv / len(TYPES)) * rg[:, None]
    iv = np.where((mult > 0) & np.isfinite(vol) & (vol > 0), 1.0/np.where(vol > 0, vol, np.nan), 0.0)
    iv = np.nan_to_num(iv) * mult
    s = iv.sum(axis=1, keepdims=True)
    w_uncapped = np.where(s > 0, iv/np.maximum(s, 1e-12), 0.0)
    w = np.minimum(w_uncapped, cap)
    # cap-binding diagnostic: of all LIVE (uncapped weight>0) name-bars, what frac got clipped?
    live = w_uncapped > 1e-9
    clipped = live & (w_uncapped > cap + 1e-9)
    cap_bind_frac = float(clipped.sum() / max(live.sum(), 1))
    Wl = np.vstack([np.zeros((1, A)), w[:-1]])
    gr = np.nansum(Wl*ret, axis=1)
    turn = np.nansum(np.abs(w - np.vstack([np.zeros((1, A)), w[:-1]])), axis=1)
    net = gr - turn*(COST_PROD/2)
    n_names = (w > 0).sum(axis=1)
    gross_exp = Wl.sum(axis=1)
    mask = (idx >= pd.Timestamp(OOS_START)) & (idx < pd.Timestamp(OOS_2024_END))
    return net[mask], idx[mask], bpd, n_names[mask], gross_exp[mask], A, cap_bind_frac


def metrics(net, bpd, n_names, gross_exp, n_total, cap_bind_frac):
    lin_wk = float(np.mean(net)) * bpd * 7 * 100
    comp = float(np.prod(1+net) - 1) * 100
    cum = np.cumprod(1+net)
    maxdd = float(np.min((cum - np.maximum.accumulate(cum))/np.maximum.accumulate(cum)))*100 if len(cum)>1 else 0
    sharpe = (float(np.mean(net))*bpd*365)/(float(np.std(net))*np.sqrt(bpd*365)) if np.std(net)>1e-12 else 0
    sb = int(7*bpd); sl=[]; i=0
    while i+sb <= len(net):
        sl.append(float(np.prod(1+net[i:i+sb])-1)*100); i+=sb
    pct7 = sum(1 for x in sl if x>0)/len(sl) if sl else 0
    active = n_names[n_names > 0]
    # return-per-unit-drawdown = the de-risk-mandate efficiency metric
    rpdd = round(lin_wk / abs(maxdd), 4) if abs(maxdd) > 1e-9 else 0
    return {
        "lin_wk": round(lin_wk, 3), "compound": round(comp, 1), "sharpe": round(sharpe, 3),
        "maxdd": round(maxdd, 1), "return_per_drawdown": rpdd, "cap_bind_frac": round(cap_bind_frac, 3),
        "n_names_held_median": int(np.median(active)) if len(active) else 0,
        "avg_gross_exposure": round(float(np.mean(gross_exp)), 3),
        "pct_profitable_7d_slices": round(pct7, 3), "n_bars": int(len(net)),
    }


def main():
    caps = [0.10, 0.15, 0.20, 0.25, 0.33, 0.40, 0.50, 0.75, 1.0]
    out = {"check": "WAVE2_concentration_frontier_DELEAKED",
           "window": "OOS-2024 (2024-05-15..2024-12-31); 2025 SEALED", "cost": f"PROD {COST_PROD} RT",
           "engine": "ma_slice_roi.build_book_net cross-type conv inv-vol, DE-LEAKED (regime.shift(1))",
           "cells": {}, "ts": datetime.now().strftime("%Y%m%d_%H%M%S")}
    for tf in ["1d", "4h"]:
        print(f"\n{'='*92}\n{tf}  DE-LEAKED concentration frontier (OOS-2024, prod cost)\n{'='*92}")
        print(f"  {'cap':>6}{'lin_wk':>9}{'Sharpe':>8}{'maxDD':>8}{'ret/DD':>8}{'7dprof':>8}"
              f"{'capbind':>9}{'n_med':>7}{'gross':>7}")
        print("  " + "-"*84)
        for cap in caps:
            net, dates, bpd, nn, ge, A, cbf = build_book(tf, cap=cap, deleak=True)
            m = metrics(net, bpd, nn, ge, A, cbf)
            lbl = "EW" if cap >= 1.0 else f"{int(cap*100)}%"
            out["cells"][f"{tf}|cap{lbl}"] = m
            print(f"  {lbl:>6}{m['lin_wk']:>+9.3f}{m['sharpe']:>+8.2f}{m['maxdd']:>+8.1f}"
                  f"{m['return_per_drawdown']:>+8.3f}{m['pct_profitable_7d_slices']*100:>7.0f}%"
                  f"{m['cap_bind_frac']*100:>8.0f}%{m['n_names_held_median']:>7}{m['avg_gross_exposure']:>7.2f}")
    with open(HERE / "wave2_concentration_frontier.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved {HERE/'wave2_concentration_frontier.json'}")


if __name__ == "__main__":
    main()
