"""LOOKBACK ABLATION — does the lookback-window confirmation add anything?

The lookback feeds the confirmation layer: enter only when band-signal long AND
close > EMA(lookback) AND EMA(lookback) rising. This ablates that layer to measure its
contribution, and sweeps the lookback LENGTH.

Arms (all else identical: frozen band, regime gate, prod cost, 15% cap, slice-hold):
  A) NONE        : band signal alone, NO confirmation (lookback removed)
  B) ABOVE-only  : band AND close>EMA(lb), no 'rising'
  C) RISING-only : band AND EMA(lb) rising, no 'above'
  D) FULL @ lb   : band AND above AND rising, for lb in {3,7,14,21,28} days

If FULL ~ NONE -> lookback adds nothing. If FULL > NONE robustly -> it earns its place,
and the sweep shows the best length. OOS-2024, 2025 sealed, strictly causal.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CRYPTO = ROOT.parent

import strat.fleet_lab as fl
from strat.ma_wealth_driver import _load_asset, _u100_symbols
from strat.ma_movecatch_engine import select_band_trainval, make_hold_provider
from strat.ma_type_upgrade import _ema
from strat.canonical_exits import COST_PROD, resolve_held
from strat.ma_slice_engine import causal_regime, OOS_WALL_MS, OOS_START, OOS_2024_END

BARS_PER_DAY = fl.BARS_PER_DAY
OUTDIR = CRYPTO / "runs" / "engines" / "V15_lookback_ablation"
TIER_CFG = {"1d": {"slice": 14}, "4h": {"slice": 7}}


def confirmed(close, band_held, lookback_bars, mode):
    c = np.asarray(close, float); n = len(c)
    bh = np.asarray(band_held, dtype=bool)
    m = np.isfinite(c)
    out = np.zeros(n, dtype=bool)
    if mode == "none":
        out[m] = bh[m]
        return out
    if m.sum() < max(5, lookback_bars):
        return out
    cseg = c[m]; ema = _ema(cseg, lookback_bars)
    above = cseg > ema
    rising = np.concatenate([[False], ema[1:] >= ema[:-1]])
    bhs = bh[m]
    if mode == "above":
        cs = bhs & above
    elif mode == "rising":
        cs = bhs & rising
    elif mode == "full":
        cs = bhs & above & rising
    else:
        cs = bhs
    out[np.where(m)[0]] = cs
    return out


def eval_arm(ma_type, tf, mode, lb_days, closes, prov_cache, reg, cap=0.15):
    bpd = BARS_PER_DAY[tf]; sl = TIER_CFG[tf]["slice"] * bpd
    lb = int(lb_days * bpd)
    hc, cc = {}, {}
    for sym, (d, o, h, l, c) in closes.items():
        bh = prov_cache[sym]
        conf = confirmed(c, bh, lb, mode)
        hc[sym] = pd.Series(resolve_held(c, conf, sl), index=d)
        cc[sym] = pd.Series(c, index=d)
    close = pd.DataFrame(cc).sort_index()
    heldp = pd.DataFrame(hc).reindex(close.index).fillna(0).astype(np.int8)
    ret = close.pct_change().fillna(0.0).to_numpy()
    H = heldp.to_numpy().astype(float)
    vol = close.pct_change().rolling(max(20, bpd*5), min_periods=10).std().to_numpy()
    dates = close.index; T, A = H.shape
    rg = np.ones(T); ridx = reg.index
    for k, dt in enumerate(dates):
        d1 = pd.Timestamp(dt).normalize(); prior = ridx[ridx <= d1]
        rg[k] = int(reg[prior[-1]]) if len(prior) else 0
    Hg = H * rg[:, None]
    iv = np.where((Hg > 0) & np.isfinite(vol) & (vol > 0), 1.0/np.where(vol > 0, vol, np.nan), 0.0)
    iv = np.nan_to_num(iv); s = iv.sum(axis=1, keepdims=True)
    w = np.where(s > 0, iv/np.maximum(s, 1e-12), 0.0); w = np.minimum(w, cap)
    Wl = np.vstack([np.zeros((1, A)), w[:-1]])
    gr = np.nansum(Wl*ret, axis=1)
    turn = np.nansum(np.abs(w - np.vstack([np.zeros((1, A)), w[:-1]])), axis=1)
    net = gr - turn*(COST_PROD/2)
    mask = (dates >= pd.Timestamp(OOS_START)) & (dates < pd.Timestamp(OOS_2024_END))
    nw = net[mask]
    lin = float(np.mean(nw))*bpd*7*100
    sh = (float(np.mean(nw))*bpd*365)/(float(np.std(nw))*np.sqrt(bpd*365)) if np.std(nw) > 1e-12 else 0
    cum = np.cumprod(1+nw); dd = float(np.min((cum-np.maximum.accumulate(cum))/np.maximum.accumulate(cum)))*100 if len(cum)>1 else 0
    # 7d-slice profitability (the mandate metric)
    sb = 7*bpd; sl_pnl = []
    i = 0
    while i+sb <= len(nw):
        sl_pnl.append(float(np.prod(1+nw[i:i+sb])-1)); i += sb
    pct7 = float(np.mean(np.array(sl_pnl) > 0)) if sl_pnl else 0
    dep = float(np.mean(H[mask].sum(axis=1) > 0))
    return {"lin_wk": round(lin,3), "sharpe": round(sh,2), "maxdd": round(dd,1),
            "pct7_prof": round(pct7,3), "deploy": round(dep,3)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", default="SMA,EMA,VIDYA")
    ap.add_argument("--tf", default="1d")
    a = ap.parse_args()
    types = [t.strip().upper() for t in a.types.split(",")]
    tf = a.tf
    reg = causal_regime(OOS_WALL_MS)   # de-leaked: bar on day D reads regime[D-1]
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    allout = {}
    print(f"LOOKBACK ABLATION — {tf}, OOS-2024, prod, 15% cap (lin/wk | Sh | 7d%prof | deploy)")
    print("=" * 100)
    for ma in types:
        band = select_band_trainval(ma, tf)[0]
        prov = make_hold_provider(band, ma)
        closes = {}
        prov_cache = {}
        for sym in _u100_symbols():
            rec = _load_asset(sym, tf, OOS_WALL_MS)
            if rec is None:
                continue
            d, o, h, l, c = rec; c = np.asarray(c, float)
            closes[sym] = (d, o, h, l, c)
            prov_cache[sym] = prov(sym, o, h, l, c)
        print(f"\n  {ma}|{tf}")
        res = {}
        res["NONE (no confirm)"] = eval_arm(ma, tf, "none", 0, closes, prov_cache, reg)
        res["above-only (lb7)"] = eval_arm(ma, tf, "above", 7, closes, prov_cache, reg)
        res["rising-only (lb7)"] = eval_arm(ma, tf, "rising", 7, closes, prov_cache, reg)
        for lbd in [3, 7, 14, 21, 28]:
            res[f"full lb{lbd}d"] = eval_arm(ma, tf, "full", lbd, closes, prov_cache, reg)
        allout[ma] = res
        for label, m in res.items():
            print(f"    {label:<20} {m['lin_wk']:>+6.2f}%  Sh {m['sharpe']:>5.2f}  "
                  f"7d%prof {m['pct7_prof']:>4.0%}  dep {m['deploy']:>4.0%}")
    with open(OUTDIR / f"lookback_ablation_{tf}_{ts}.json", "w") as f:
        json.dump({"tf": tf, "by_type": allout, "ts": ts}, f, indent=2, default=str)
    print(f"\n  Saved lookback_ablation_{tf}_{ts}.json")


if __name__ == "__main__":
    main()
