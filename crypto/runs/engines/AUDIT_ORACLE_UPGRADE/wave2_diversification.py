"""WAVE 2 PART B -- does a DE-CORRELATED second mechanism lift the frontier beyond MA-band@cap33?

Within the MA-band mechanism, cap33 is ~the robust ceiling (Part A). To go higher robustly you must
ADD a mechanism that is positive-EV AND de-correlated. The one genuine non-MA SELECTIVE survivor is
ADX|1d (PROVISIONAL: OOS +0.9-1.2pp, VAL-fit, audit says DO NOT SIZE until 2025 confirms). This test
quantifies the diversification POTENTIAL (gated), it is NOT a deploy decision:

  1. Build BOTH sleeves' per-bar NET streams on the SAME basis: de-leaked (regime.shift(1)), PROD cost,
     EW long-only, u100, 1d -- so the blend is apples-to-apples.
       MA = cross-type conv band @cap33 (the Part-A robust point).
       ADX = ensemble over lb{24,26,28} x thr{0.42,0.45} (audit-recommended; drop thr0.48 + lb<22).
  2. Correlation of the two net streams (low corr = real diversification headroom).
  3. Blend w_adx in {0, 0.25, 0.5}; report lin_wk / Sharpe / maxDD on BOTH VAL and OOS.
     A genuine lever IMPROVES risk-adjusted return (Sharpe / return-per-DD) on BOTH windows, not one.

2025 SEALED. long-only spot lev=1.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(ROOT))

import strat.fleet_lab as fl
from strat.ma_wealth_driver import _load_asset, _u100_symbols
from strat.ma_slice_engine import (held_rolling, _cached_band_held, btc_regime,
    OOS_WALL_MS, OOS_START, OOS_2024_END, VAL_WALL_MS, VAL_START, VAL_END)
from strat.ti_engine_sota import adx_single_held
from strat.canonical_exits import COST_PROD

BARS_PER_DAY = fl.BARS_PER_DAY
TYPES = ["SMA", "EMA", "WMA", "VIDYA", "KAMA"]
WIN = {"OOS": (OOS_WALL_MS, OOS_START, OOS_2024_END), "VAL": (VAL_WALL_MS, VAL_START, VAL_END)}
TF = "1d"; LB_SL = (14, 14)   # 1d tier config


def _regime_on_grid(reg, idx):
    """De-leaked broadcast: bar on day D reads regime[D-1] (last CLOSED 1d bar)."""
    reg = reg.shift(1).fillna(0).astype(int); ridx = reg.index
    rg = np.ones(len(idx))
    for k, dt in enumerate(idx):
        d1 = pd.Timestamp(dt).normalize(); prior = ridx[ridx <= d1]
        rg[k] = int(reg[prior[-1]]) if len(prior) else 0
    return rg


def _ew_net_from_held(close_df, held_mat, rg, cost_rt=COST_PROD):
    """EW long-only net per-bar stream from a held matrix (T x A), causal lag-1, regime-gated."""
    ret = close_df.pct_change().to_numpy()[1:]
    held_lag = np.vstack([np.zeros((1, held_mat.shape[1])), held_mat[:-1]])[1:]
    turn = np.abs(np.diff(held_mat, axis=0, prepend=held_mat[:1]))[1:]
    rgw = rg[1:]
    valid = np.isfinite(ret); ret_f = np.where(valid, ret, 0.0)
    n_held = (held_lag * valid).sum(axis=1)
    gross = np.where(n_held > 0, (ret_f*held_lag*valid).sum(axis=1)/np.maximum(n_held, 1), 0.0)
    cost = np.where(n_held > 0, (turn*valid).sum(axis=1)*(cost_rt/2)/np.maximum(n_held, 1), 0.0)
    net = (gross - cost) * rgw
    return net, close_df.index[1:]


def ma_band_net(window, cap=0.33):
    """Cross-type conv band, inv-vol weights @cap, de-leaked, PROD -> date-indexed net Series."""
    wall, ws, we = WIN[window]; bpd = BARS_PER_DAY[TF]
    lb, sl = LB_SL[0]*bpd, LB_SL[1]*bpd
    reg = btc_regime(wall)
    closes = {}
    for sym in _u100_symbols():
        rec = _load_asset(sym, TF, wall)
        if rec is None: continue
        d, o, h, l, c = rec
        closes[sym] = (d, o, h, l, np.asarray(c, float))
    syms = list(closes.keys())
    close_df = pd.DataFrame({s: pd.Series(closes[s][4], index=closes[s][0]) for s in syms}).sort_index()
    idx = close_df.index; T = len(idx); A = len(syms)
    conv = np.zeros((T, A))
    for ma in TYPES:
        hc = {}
        for s in syms:
            d, o, h, l, c = closes[s]
            bh = _cached_band_held(ma, TF, s, o, h, l, c, wall)
            hc[s] = pd.Series(held_rolling(c, bh, lb, sl), index=d)
        conv += pd.DataFrame(hc).reindex(idx).fillna(0).reindex(columns=close_df.columns).fillna(0).to_numpy()
    ret = close_df.pct_change().fillna(0.0).to_numpy()
    vol = close_df.pct_change().rolling(max(20, bpd*5), min_periods=10).std().to_numpy()
    rg = _regime_on_grid(reg, idx)
    mult = (conv/len(TYPES)) * rg[:, None]
    iv = np.where((mult > 0) & np.isfinite(vol) & (vol > 0), 1.0/np.where(vol > 0, vol, np.nan), 0.0)
    iv = np.nan_to_num(iv) * mult
    s = iv.sum(axis=1, keepdims=True)
    w = np.minimum(np.where(s > 0, iv/np.maximum(s, 1e-12), 0.0), cap)
    Wl = np.vstack([np.zeros((1, A)), w[:-1]])
    gr = np.nansum(Wl*ret, axis=1)
    turn = np.nansum(np.abs(w - np.vstack([np.zeros((1, A)), w[:-1]])), axis=1)
    net = gr - turn*(COST_PROD/2)
    mask = (idx >= pd.Timestamp(ws)) & (idx < pd.Timestamp(we))
    return pd.Series(net[mask], index=idx[mask])


def adx_net(window):
    """ADX|1d ensemble (lb{24,26,28} x thr{0.42,0.45}) EW, de-leaked, PROD -> date-indexed net Series."""
    wall, ws, we = WIN[window]; bpd = BARS_PER_DAY[TF]
    cc = {}
    held_stack = None
    for sym in _u100_symbols():
        rec = _load_asset(sym, TF, wall)
        if rec is None: continue
        dates, o, h, l, c = rec
        cc[sym] = pd.Series(np.asarray(c, float), index=dates)
    cdf = pd.DataFrame(cc).sort_index(); idx = cdf.index; A = cdf.shape[1]
    # ensemble: a name is HELD if ANY (lb,thr) cell holds it (majority-vote also possible; use mean>=0.5)
    ens = np.zeros((len(idx), A))
    n_cells = 0
    for lb in (24, 26, 28):
        for thr in (0.42, 0.45):
            hc = {}
            for j, sym in enumerate(cdf.columns):
                c = cdf[sym].to_numpy()
                hc[sym] = pd.Series(adx_single_held(c, lb, thr), index=idx)
            ens += pd.DataFrame(hc).reindex(idx).fillna(0).reindex(columns=cdf.columns).fillna(0).to_numpy()
            n_cells += 1
    held = (ens / n_cells >= 0.5).astype(float)   # majority of cells hold -> hold
    reg = btc_regime(wall); rg = _regime_on_grid(reg, idx)
    net, dates = _ew_net_from_held(cdf, held, rg)
    mask = (dates >= pd.Timestamp(ws)) & (dates < pd.Timestamp(we))
    return pd.Series(net[mask], index=dates[mask])


def met(net, bpd=1):
    n = net.to_numpy()
    lin = float(np.mean(n))*BARS_PER_DAY[TF]*7*100
    cum = np.cumprod(1+n)
    dd = float(np.min((cum-np.maximum.accumulate(cum))/np.maximum.accumulate(cum)))*100 if len(cum)>1 else 0
    sh = (float(np.mean(n))*BARS_PER_DAY[TF]*365)/(float(np.std(n))*np.sqrt(BARS_PER_DAY[TF]*365)) if np.std(n)>1e-12 else 0
    sb = 7
    sl = [float(np.prod(1+n[i:i+sb])-1)*100 for i in range(0, len(n)-sb+1, sb)]
    pct7 = sum(1 for x in sl if x>0)/len(sl) if sl else 0
    return {"lin_wk": round(lin,3), "sharpe": round(sh,3), "maxdd": round(dd,1),
            "ret_per_dd": round(lin/abs(dd),4) if abs(dd)>1e-9 else 0, "pct7d": round(pct7,3)}


def main():
    out = {"check": "WAVE2B_diversification_MAband_plus_ADX", "ts": datetime.now().strftime("%Y%m%d_%H%M%S"),
           "note": "ADX is PROVISIONAL/VAL-fit -> potential only, gated on 2025", "windows": {}}
    for window in ["VAL", "OOS"]:
        ma = ma_band_net(window, cap=0.33)
        adx = adx_net(window)
        common = ma.index.intersection(adx.index)
        ma, adx = ma.reindex(common), adx.reindex(common)
        corr = float(np.corrcoef(ma.to_numpy(), adx.to_numpy())[0, 1])
        print(f"\n{'='*78}\n{window}: MA-band@cap33 vs ADX|1d ensemble  (de-leaked, PROD, n_bars={len(common)})")
        print(f"  corr(MA, ADX) = {corr:+.3f}   {'(low -> diversification headroom)' if abs(corr)<0.6 else '(high -> redundant)'}")
        print(f"{'='*78}\n  {'blend':<14}{'lin_wk':>9}{'Sharpe':>8}{'maxDD':>8}{'ret/DD':>8}{'7dprof':>8}")
        print("  " + "-"*63)
        rows = {"corr": round(corr, 3)}
        for w_adx in [0.0, 0.25, 0.5]:
            blend = (1-w_adx)*ma + w_adx*adx
            m = met(blend)
            lbl = "MA only" if w_adx == 0 else f"MA{int((1-w_adx)*100)}/ADX{int(w_adx*100)}"
            rows[lbl] = m
            print(f"  {lbl:<14}{m['lin_wk']:>+9.3f}{m['sharpe']:>+8.2f}{m['maxdd']:>+8.1f}"
                  f"{m['ret_per_dd']:>+8.3f}{m['pct7d']*100:>7.0f}%")
        # standalone ADX for reference
        ma_s, adx_s = met(ma), met(adx)
        print(f"  {'(ADX alone)':<14}{adx_s['lin_wk']:>+9.3f}{adx_s['sharpe']:>+8.2f}{adx_s['maxdd']:>+8.1f}"
              f"{adx_s['ret_per_dd']:>+8.3f}{adx_s['pct7d']*100:>7.0f}%")
        rows["ADX_alone"] = adx_s
        out["windows"][window] = rows
    with open(HERE / "wave2_diversification.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved {HERE/'wave2_diversification.json'}")


if __name__ == "__main__":
    main()
