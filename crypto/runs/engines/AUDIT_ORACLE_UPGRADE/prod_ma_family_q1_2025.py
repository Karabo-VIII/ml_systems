"""PROD-GRADE backtest of THE MA FAMILY OF ENGINES (the deployable) -- primary window Q1-2025.

CORRECTION (user /orc 2026-06-28): the deployable is NOT "trade BTC" -- that collapses the breadth
surface (u100 -> BTC), off-spec. The strategy spec = the MA FAMILY OF ENGINES on the FULL u100 alt
universe, canonical 14d lookback / 7d trading window, ROLLING slices, regime-gated, long-only spot.
This harness backtests THAT, prod-grade. BTC appears only as a context benchmark, never the product.

PROD-GRADE: next-open fills (open-to-open, lag-1), de-leaked regime, point-in-time listing universe,
prod cost (taker+slip 0.0028 RT), turnover costed. (close-to-close also reported as the reference.)
DATA CAVEAT: volume feed unreliable -> no ADV filter; u100 top-list IS the liquidity screen.

ENGINES (the MA family): per-type bands {SMA,EMA,WMA,VIDYA,KAMA} + the cross-type CONVERGENCE book
(the V12/V14 deployable landing). WEIGHTS: inv-vol clipped at cap (canonical 15% + robust 33%).
GATE: cross-sectional BREADTH (validated) AND single-BTC-SMA100 (canonical) -- both shown, de-leaked.
WINDOWS: Q1-2025 (PRIMARY) + VAL + OOS-2024. BENCHMARK: BH alt-basket (just hold u100). long-only lev=1.
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
from strat.ma_slice_engine import held_rolling, _cached_band_held
from strat.ma_slice_engine import OOS_START, OOS_2024_END, VAL_START, VAL_END
from strat.canonical_exits import COST_PROD

BARS_PER_DAY = fl.BARS_PER_DAY
TYPES = ["SMA", "EMA", "WMA", "VIDYA", "KAMA"]
LB_D, SL_D = 14, 7            # CANONICAL SPEC: 14d lookback, 7d trading window
WIN = {
    "VAL":  (int(pd.Timestamp(VAL_END).value//10**6), VAL_START, VAL_END),
    "OOS24":(int(pd.Timestamp("2025-01-01").value//10**6), OOS_START, OOS_2024_END),
    "Q1_25":(int(pd.Timestamp("2025-04-01").value//10**6), "2025-01-01", "2025-04-01"),
}


def hyst(sig, up, dn):
    n = len(sig); s = np.zeros(n, dtype=int); st = 0
    up = np.full(n, up, float) if np.isscalar(up) else np.asarray(up, float)
    dn = np.full(n, dn, float) if np.isscalar(dn) else np.asarray(dn, float)
    for i in range(n):
        if not (np.isfinite(sig[i]) and np.isfinite(up[i]) and np.isfinite(dn[i])): s[i] = st; continue
        if st == 0 and sig[i] > up[i]: st = 1
        elif st == 1 and sig[i] < dn[i]: st = 0
        s[i] = st
    return s


def build(window, tf):
    wall, ws, we = WIN[window]; bpd = BARS_PER_DAY[tf]; lb, sl = LB_D*bpd, SL_D*bpd; sw = 100*bpd
    closes, opens = {}, {}
    for sym in _u100_symbols():
        rec = _load_asset(sym, tf, wall)
        if rec is None: continue
        d, o, h, l, c = rec
        closes[sym] = pd.Series(np.asarray(c, float), index=d); opens[sym] = pd.Series(np.asarray(o, float), index=d)
    cdf = pd.DataFrame(closes).sort_index(); odf = pd.DataFrame(opens).reindex(cdf.index).reindex(columns=cdf.columns)
    idx = cdf.index; T = len(idx); A = len(cdf.columns)
    typ_held = {}; conv = np.zeros((T, A))
    for ma in TYPES:
        hc = {}
        for s in cdf.columns:
            rec = _load_asset(s, tf, wall)
            if rec is None: continue
            d, o, h, l, c = rec
            bh = _cached_band_held(ma, tf, s, o, h, l, c, wall)
            hc[s] = pd.Series(held_rolling(c, bh, lb, sl), index=d)
        H = pd.DataFrame(hc).reindex(idx).fillna(0).reindex(columns=cdf.columns).fillna(0).to_numpy()
        typ_held[ma] = H; conv += H
    C = cdf.to_numpy(); O = odf.to_numpy()
    vol = cdf.pct_change().rolling(max(20, bpd*5), min_periods=10).std().to_numpy()
    roo = np.full((T, A), 0.0); roo[:-1] = O[1:]/np.where(O[:-1] > 0, O[:-1], np.nan) - 1.0; roo = np.nan_to_num(roo)
    rcc = cdf.pct_change().fillna(0.0).to_numpy()
    sall = cdf.rolling(sw, min_periods=sw//2).mean().to_numpy(); vc = np.isfinite(sall).sum(axis=1)
    brd = np.where(vc > 0, np.nansum((C > sall)*np.isfinite(sall), axis=1)/np.maximum(vc,1), np.nan)
    g_brd = hyst(brd, 0.5, 0.4)
    btc = "BTCUSDT" if "BTCUSDT" in cdf.columns else cdf.columns[0]
    bc = cdf[btc]; bsma = bc.rolling(sw, min_periods=sw//2).mean().to_numpy()
    g_btc = hyst(bc.to_numpy(), bsma*1.02, bsma*0.98); j_btc = list(cdf.columns).index(btc)
    mask = (idx >= pd.Timestamp(ws)) & (idx < pd.Timestamp(we))
    return {"cdf":cdf,"conv":conv,"typ":typ_held,"vol":vol,"roo":roo,"rcc":rcc,"g_brd":g_brd,"g_btc":g_btc,
            "j_btc":j_btc,"mask":mask,"bpd":bpd,"A":A}


def net(W, ret, cost=COST_PROD):
    Wl = np.vstack([np.zeros((1, W.shape[1])), W[:-1]])
    gr = np.nansum(Wl*ret, axis=1)
    turn = np.nansum(np.abs(W - np.vstack([np.zeros((1, W.shape[1])), W[:-1]])), axis=1)
    return gr - turn*(cost/2)


def invvol_W(held_mult, vol, cap):
    iv = np.where((held_mult > 0) & np.isfinite(vol) & (vol > 0), 1.0/np.where(vol > 0, vol, np.nan), 0.0)
    iv = np.nan_to_num(iv)*held_mult; s = iv.sum(axis=1, keepdims=True)
    return np.minimum(np.where(s > 0, iv/np.maximum(s, 1e-12), 0.0), cap)


def book_W(f, kind, gate, cap):
    g = (gate[:, None] > 0)
    if kind == "conv":
        return invvol_W((f["conv"]/len(TYPES))*g, f["vol"], cap)
    return invvol_W(f["typ"][kind].astype(float)*g, f["vol"], cap)   # per-type


def bh_alt_W(f):
    live = np.isfinite(f["cdf"].to_numpy()); cnt = live.sum(axis=1, keepdims=True)
    return np.where(live & (cnt > 0), 1.0/np.maximum(cnt, 1), 0.0)


def bh_btc_W(f):
    W = np.zeros((len(f["cdf"]), f["A"])); W[:, f["j_btc"]] = 1.0; return W


def metrics(n, f):
    n = n[f["mask"]]
    if len(n) == 0: return None
    comp = float(np.prod(1+n)-1)*100; lin = float(np.mean(n))*f["bpd"]*7*100
    cum = np.cumprod(1+n)
    dd = float(np.min((cum-np.maximum.accumulate(cum))/np.maximum.accumulate(cum)))*100 if len(cum)>1 else 0
    sh = (float(np.mean(n))*f["bpd"]*365)/(float(np.std(n))*np.sqrt(f["bpd"]*365)) if np.std(n)>1e-12 else 0
    sb = 7*f["bpd"]; sl=[float(np.prod(1+n[i:i+sb])-1)*100 for i in range(0,len(n)-sb+1,sb)]
    pct7 = sum(1 for x in sl if x>0)/len(sl) if sl else 0
    return {"compound":round(comp,1),"lin_wk":round(lin,3),"sharpe":round(sh,2),"maxdd":round(dd,1),"pct7d":round(pct7,3)}


def main():
    out = {"check":"PROD_GRADE_MA_FAMILY","spec":"u100 full, 14d lookback / 7d window, ROLLING, de-leaked, next-open fills, prod cost",
           "ts":datetime.now().strftime("%Y%m%d_%H%M%S"),"windows":{}}
    for window in ["VAL","OOS24","Q1_25"]:
        rows = {}
        for tf in ["1d","4h"]:
            f = build(window, tf); ret = f["roo"]
            if tf == "1d":
                rows["BH alt-basket (hold u100)"] = metrics(net(bh_alt_W(f), ret), f)
                rows["BH BTC (context only)"] = metrics(net(bh_btc_W(f), ret), f)
            # the deployable: cross-type CONV book, breadth gate, cap15 + cap33
            for cap,ct in [(0.15,"15"),(0.33,"33")]:
                rows[f"CONV {tf} breadth cap{ct}"] = metrics(net(book_W(f,"conv",f["g_brd"],cap), ret), f)
            rows[f"CONV {tf} btc-gate cap33"] = metrics(net(book_W(f,"conv",f["g_btc"],0.33), ret), f)
            # the MA FAMILY: per-type (breadth gate, cap33)
            for ma in TYPES:
                rows[f"{ma} {tf} breadth cap33"] = metrics(net(book_W(f,ma,f["g_brd"],0.33), ret), f)
        out["windows"][window] = rows
        wl = WIN[window]
        print(f"\n{'='*94}\n{window} ({wl[1]}..{wl[2]})  MA FAMILY prod-grade (u100, 14d/7d, de-leaked, next-open)  [by WEALTH]")
        print(f"{'='*94}")
        print(f"  {'engine':<30}{'compound':>10}{'lin_wk':>8}{'Sharpe':>7}{'maxDD':>7}{'7dprof':>7}")
        print("  "+"-"*74)
        for name,m in sorted(rows.items(), key=lambda kv:-kv[1]["compound"]):
            print(f"  {name:<30}{m['compound']:>+9.1f}%{m['lin_wk']:>+8.2f}{m['sharpe']:>+7.2f}{m['maxdd']:>+7.0f}{m['pct7d']*100:>6.0f}%")
    with open(HERE/"prod_ma_family_q1_2025.json","w") as fo:
        json.dump(out, fo, indent=2, default=str)
    print(f"\nSaved {HERE/'prod_ma_family_q1_2025.json'}")


if __name__ == "__main__":
    main()
