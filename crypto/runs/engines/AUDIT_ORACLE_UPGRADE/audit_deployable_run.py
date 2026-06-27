"""ADVERSARIAL re-verification of the DEPLOYABLE capped wealth claim.

Claim (CURRENT.md L86-90): inv-vol + 15% cap book, OOS-2024, prod cost -> SMA|1d +1.67/wk,
EMA|1d +1.68, SMA|4h +1.60; cap is the dial 15=1.4 / 33=1.6 / 50=2.1 / EW=3.4.

This RE-RUNS the EXACT deployable engine (ma_slice_roi.build_book_net = cross-type conv book,
inv-vol weights clipped at `cap`, leaked btc_regime) and ALSO a DE-LEAKED variant (regime
shifted +1 full 1d bar so day-D reads the last CLOSED daily regime). For each (cap, leak-state):
  - lin_wk (mean per-bar net * bpd * 7 * 100)  -- the headline metric
  - compound, sharpe, maxdd, 7d %profitable, n_names (breadth, the avg # held names), n_bars
No rebalancing-DEV inflation: this is a single per-bar NET STREAM (not the alwayslong compound).
2025 SEALED (asserted). Long-only spot lev=1.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

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


def build_book(tf, cap=0.15, deleak=False):
    """Reproduce ma_slice_roi.build_book_net. deleak=True shifts the daily regime +1 bar
    (day-D reads regime[D-1], the last CLOSED 1d bar) -- the LeakCheck fix."""
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
    w = np.where(s > 0, iv/np.maximum(s, 1e-12), 0.0)
    w = np.minimum(w, cap)
    Wl = np.vstack([np.zeros((1, A)), w[:-1]])
    gr = np.nansum(Wl*ret, axis=1)
    turn = np.nansum(np.abs(w - np.vstack([np.zeros((1, A)), w[:-1]])), axis=1)
    net = gr - turn*(COST_PROD/2)
    n_names = (w > 0).sum(axis=1)         # held breadth per bar
    gross_exp = Wl.sum(axis=1)            # deployed capital fraction (1.0 = fully invested)
    mask = (idx >= pd.Timestamp(OOS_START)) & (idx < pd.Timestamp(OOS_2024_END))
    return net[mask], idx[mask], bpd, n_names[mask], gross_exp[mask], A


def metrics(net, bpd, n_names, gross_exp, n_total):
    lin_wk = float(np.mean(net)) * bpd * 7 * 100
    comp = float(np.prod(1+net) - 1) * 100
    cum = np.cumprod(1+net)
    maxdd = float(np.min((cum - np.maximum.accumulate(cum))/np.maximum.accumulate(cum)))*100 if len(cum)>1 else 0
    sharpe = (float(np.mean(net))*bpd*365)/(float(np.std(net))*np.sqrt(bpd*365)) if np.std(net)>1e-12 else 0
    # 7d slice profitability
    sb = int(7*bpd); sl=[]; i=0
    while i+sb <= len(net):
        sl.append(float(np.prod(1+net[i:i+sb])-1)*100); i+=sb
    pct7 = sum(1 for x in sl if x>0)/len(sl) if sl else 0
    active = n_names[n_names > 0]
    return {
        "lin_wk": round(lin_wk, 3), "compound": round(comp, 1), "sharpe": round(sharpe, 3),
        "maxdd": round(maxdd, 1), "n_bars": int(len(net)), "n_total_admitted": int(n_total),
        "n_names_held_median": int(np.median(active)) if len(active) else 0,
        "n_names_held_max": int(n_names.max()),
        "deploy_frac": round(float(np.mean(gross_exp > 1e-9)), 3),
        "avg_gross_exposure": round(float(np.mean(gross_exp)), 3),
        "pct_profitable_7d_slices": round(pct7, 3), "n_7d_slices": len(sl),
    }


def main():
    out = {"check": "DEPLOYABLE_capped_wealth_reverify", "window": "OOS-2024 (2024-05-15..2024-12-31)",
           "cost": f"PROD {COST_PROD} RT", "engine": "ma_slice_roi.build_book_net (cross-type conv, inv-vol, clipped)",
           "note": "single per-bar NET STREAM (NOT the alwayslong rebalanced-DEV compound; no 9.5x inflation here)",
           "cells": {}, "ts": datetime.now().strftime("%Y%m%d_%H%M%S")}
    for tf in ["1d", "4h"]:
        for cap, lbl in [(0.15, "cap15"), (0.33, "cap33"), (0.50, "cap50"), (1.0, "EW_nocap")]:
            for deleak in [False, True]:
                net, dates, bpd, nn, ge, A = build_book(tf, cap=cap, deleak=deleak)
                m = metrics(net, bpd, nn, ge, A)
                key = f"{tf}|{lbl}|{'deleaked' if deleak else 'leaked'}"
                out["cells"][key] = m
                print(f"{key:<28} lin_wk={m['lin_wk']:+7.3f}  Sh={m['sharpe']:+.2f}  "
                      f"DD={m['maxdd']:+6.1f}  7dprof={m['pct_profitable_7d_slices']:.0%}  "
                      f"n_names(med/max)={m['n_names_held_median']}/{m['n_names_held_max']}  "
                      f"n_total={m['n_total_admitted']}  deploy={m['deploy_frac']:.0%}")
    od = CRYPTO / "runs" / "engines" / "AUDIT_ORACLE_UPGRADE"
    with open(od / "audit_deployable.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved {od/'audit_deployable.json'}")


if __name__ == "__main__":
    main()
