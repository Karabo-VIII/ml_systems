"""WAVE 4 -- the 2025-Q1 UNSEAL: the final out-of-sample gate for the MA-family candidates.

User authorized unsealing Q1 2025 (Jan-Mar; data ends 2025-03-31 = all the 2025 OOS that exists).
Run EVERY MA-family candidate FROZEN (params fixed from TRAIN/VAL; NOTHING tuned on 2025), de-leaked,
PROD cost, long-only spot lev=1, on the window 2025-01-01..2025-04-01. Rank by COMPOUND (wealth).

Q1-2025 was a DOWN/CHOPPY quarter (BTC peaked ~mid-Jan, fell hard Feb-Mar) -> a genuine stress test:
the gate must preserve capital. This is a SINGLE short window (~13 weeks) = a directional confirm
(does V3/momentum/breadth survive, not collapse?), now the 3rd window after VAL + OOS-2024.

Candidates: buy-hold benchmarks; per-MA-type EW; the cap dial (btc gate + conv + EW, cap15/33/50/EW);
breadth+EW (lever B); btc+momentum (lever A); V3 (breadth + momentum top-K inv-vol, K3/5/8/10).
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
from strat.canonical_exits import COST_PROD

BARS_PER_DAY = fl.BARS_PER_DAY
TYPES = ["SMA", "EMA", "WMA", "VIDYA", "KAMA"]
LB_D, SL_D = 14, 14
Q1_WALL = int(pd.Timestamp("2025-04-01").value // 10**6)
Q1_START, Q1_END = "2025-01-01", "2025-04-01"


def hyst(sig, up, dn):
    """up/dn may be scalars or per-bar arrays (e.g. BTC vs its own moving SMA*1.02/0.98)."""
    n = len(sig); s = np.zeros(n, dtype=int); st = 0
    up = np.full(n, up, float) if np.isscalar(up) else np.asarray(up, float)
    dn = np.full(n, dn, float) if np.isscalar(dn) else np.asarray(dn, float)
    for i in range(n):
        if not (np.isfinite(sig[i]) and np.isfinite(up[i]) and np.isfinite(dn[i])): s[i] = st; continue
        if st == 0 and sig[i] > up[i]: st = 1
        elif st == 1 and sig[i] < dn[i]: st = 0
        s[i] = st
    return s


def build(tf):
    bpd = BARS_PER_DAY[tf]; lb, sl = LB_D*bpd, SL_D*bpd; sw = 100*bpd
    closes = {}
    for sym in _u100_symbols():
        rec = _load_asset(sym, tf, Q1_WALL)
        if rec is None: continue
        d, o, h, l, c = rec
        closes[sym] = (d, o, h, l, np.asarray(c, float))
    syms = list(closes.keys())
    cdf = pd.DataFrame({s: pd.Series(closes[s][4], index=closes[s][0]) for s in syms}).sort_index()
    idx = cdf.index; T = len(idx); A = len(syms)
    # per-type band-held (for per-type books) + cross-type conv
    typ_held = {}
    conv = np.zeros((T, A))
    for ma in TYPES:
        hc = {}
        for s in syms:
            d, o, h, l, c = closes[s]
            bh = _cached_band_held(ma, tf, s, o, h, l, c, Q1_WALL)
            hc[s] = pd.Series(held_rolling(c, bh, lb, sl), index=d)
        H = pd.DataFrame(hc).reindex(idx).fillna(0).reindex(columns=cdf.columns).fillna(0).to_numpy()
        typ_held[ma] = H; conv += H
    ret = cdf.pct_change().fillna(0.0).to_numpy()
    vol = cdf.pct_change().rolling(max(20, bpd*5), min_periods=10).std().to_numpy()
    mom = cdf.pct_change(lb).to_numpy()
    C = cdf.to_numpy()
    btc = "BTCUSDT" if "BTCUSDT" in cdf.columns else cdf.columns[0]
    bc = cdf[btc]; bsma = bc.rolling(sw, min_periods=sw//2).mean().to_numpy()
    g_btc = hyst(bc.to_numpy(), bsma*1.02, bsma*0.98)
    sall = cdf.rolling(sw, min_periods=sw//2).mean().to_numpy(); vc = np.isfinite(sall).sum(axis=1)
    brd = np.where(vc > 0, np.nansum((C > sall)*np.isfinite(sall), axis=1)/np.maximum(vc,1), np.nan)
    g_brd = hyst(brd, 0.5, 0.4)
    mask = (idx >= pd.Timestamp(Q1_START)) & (idx < pd.Timestamp(Q1_END))
    n_q1 = int(np.sum([1 for s in syms if ((pd.DatetimeIndex(closes[s][0])>=pd.Timestamp(Q1_START))
                                            & (pd.DatetimeIndex(closes[s][0])<pd.Timestamp(Q1_END))).sum()>=60]))
    return {"cdf":cdf,"ret":ret,"vol":vol,"mom":mom,"conv":conv,"typ":typ_held,"g_btc":g_btc,"g_brd":g_brd,
            "mask":mask,"bpd":bpd,"A":A,"idx":idx,"n_q1_names":n_q1}


def net(W, ret, cost=COST_PROD):
    Wl = np.vstack([np.zeros((1, W.shape[1])), W[:-1]])
    gr = np.nansum(Wl*ret, axis=1)
    turn = np.nansum(np.abs(W - np.vstack([np.zeros((1, W.shape[1])), W[:-1]])), axis=1)
    return gr - turn*(cost/2)


def conv_ew_W(f, gate, cap, conv=None):
    conv = f["conv"] if conv is None else conv; vol = f["vol"]
    mult = (conv/len(TYPES)) * (gate[:, None] > 0)
    iv = np.where((mult > 0) & np.isfinite(vol) & (vol > 0), 1.0/np.where(vol > 0, vol, np.nan), 0.0)
    iv = np.nan_to_num(iv)*mult; s = iv.sum(axis=1, keepdims=True)
    return np.minimum(np.where(s > 0, iv/np.maximum(s, 1e-12), 0.0), cap)


def type_ew_W(f, gate, H):
    vol = f["vol"]; mult = H * (gate[:, None] > 0)
    iv = np.where((mult > 0) & np.isfinite(vol) & (vol > 0), 1.0/np.where(vol > 0, vol, np.nan), 0.0)
    iv = np.nan_to_num(iv)*mult; s = iv.sum(axis=1, keepdims=True)
    return np.minimum(np.where(s > 0, iv/np.maximum(s, 1e-12), 0.0), 0.33)


def mom_W(f, gate, K):
    conv, vol, mom = f["conv"], f["vol"], f["mom"]; elig = (conv > 0) & (gate[:, None] > 0)
    T, A = conv.shape; W = np.zeros((T, A)); sc = np.where(elig & np.isfinite(mom), mom, -np.inf)
    for t in range(T):
        e = np.where(np.isfinite(sc[t]))[0]
        if len(e) == 0: continue
        k = min(K, len(e)); top = e[np.argsort(-sc[t][e])[:k]]
        v = vol[t][top]; ivw = np.where(np.isfinite(v) & (v > 0), 1.0/v, 0.0)
        W[t, top] = ivw/ivw.sum() if ivw.sum() > 0 else 1.0/k
    return W


def bh_W(f, which):
    A = f["A"]; T = len(f["idx"])
    if which == "u100":
        live = np.isfinite(f["cdf"].to_numpy())
        cnt = live.sum(axis=1, keepdims=True)
        return np.where(live & (cnt > 0), 1.0/np.maximum(cnt, 1), 0.0)
    else:  # BTC only
        W = np.zeros((T, A)); j = list(f["cdf"].columns).index("BTCUSDT" if "BTCUSDT" in f["cdf"].columns else f["cdf"].columns[0])
        W[:, j] = 1.0; return W


def metrics(W, f):
    n = net(W, f["ret"])[f["mask"]]
    if len(n) == 0: return None
    comp = float(np.prod(1+n)-1)*100; lin = float(np.mean(n))*f["bpd"]*7*100
    cum = np.cumprod(1+n)
    dd = float(np.min((cum-np.maximum.accumulate(cum))/np.maximum.accumulate(cum)))*100 if len(cum)>1 else 0
    sh = (float(np.mean(n))*f["bpd"]*365)/(float(np.std(n))*np.sqrt(f["bpd"]*365)) if np.std(n)>1e-12 else 0
    sb = 7*f["bpd"]; sl=[float(np.prod(1+n[i:i+sb])-1)*100 for i in range(0,len(n)-sb+1,sb)]
    pct7 = sum(1 for x in sl if x>0)/len(sl) if sl else 0
    Wl = np.vstack([np.zeros((1, W.shape[1])), W[:-1]])[f["mask"]]
    dep = float(np.mean(Wl.sum(axis=1) > 1e-9))
    return {"compound":round(comp,1),"lin_wk":round(lin,3),"sharpe":round(sh,2),"maxdd":round(dd,1),
            "pct7d":round(pct7,3),"deploy":round(dep,3)}


def main():
    out = {"check":"WAVE4_2025Q1_unseal","window":f"{Q1_START}..{Q1_END} (Q1-2025, all 2025 OOS data)",
           "cost":f"PROD {COST_PROD}","frozen":"all params from TRAIN/VAL; nothing tuned on 2025","tf":{}}
    for tf in ["1d","4h"]:
        f = build(tf)
        cands = {}
        cands["BH EW-u100"] = bh_W(f,"u100")
        cands["BH BTC"] = bh_W(f,"btc")
        for ma in TYPES:
            cands[f"type {ma} (btc,EW33)"] = type_ew_W(f, f["g_btc"], f["typ"][ma])
        for cap,lbl in [(0.15,"cap15"),(0.33,"cap33"),(0.50,"cap50"),(1.0,"EW")]:
            cands[f"conv btc+{lbl}"] = conv_ew_W(f, f["g_btc"], cap)
        cands["conv breadth+EW33"] = conv_ew_W(f, f["g_brd"], 0.33)
        cands["conv btc+mom8"] = mom_W(f, f["g_btc"], 8)
        for K in [3,5,8,10]:
            cands[f"V3 breadth+mom{K}"] = mom_W(f, f["g_brd"], K)
        res = {name: metrics(W, f) for name, W in cands.items()}
        out["tf"][tf] = {"n_q1_names": f["n_q1_names"], "results": res}
        print(f"\n{'='*92}\n{tf}  2025-Q1 UNSEAL (frozen, de-leaked, PROD)  n_names_w_Q1={f['n_q1_names']}  [ranked by WEALTH]")
        print(f"{'='*92}")
        print(f"  {'candidate':<24}{'compound':>9}{'lin_wk':>8}{'Sharpe':>7}{'maxDD':>7}{'7dprof':>7}{'deploy':>7}")
        print("  "+"-"*84)
        for name,m in sorted(res.items(), key=lambda kv:-kv[1]["compound"]):
            print(f"  {name:<24}{m['compound']:>+8.1f}%{m['lin_wk']:>+8.2f}{m['sharpe']:>+7.2f}{m['maxdd']:>+7.0f}{m['pct7d']*100:>6.0f}%{m['deploy']*100:>6.0f}%")
    with open(HERE/"wave4_2025q1_unseal.json","w") as fo:
        json.dump(out, fo, indent=2, default=str)
    print(f"\nSaved {HERE/'wave4_2025q1_unseal.json'}")


if __name__ == "__main__":
    main()
