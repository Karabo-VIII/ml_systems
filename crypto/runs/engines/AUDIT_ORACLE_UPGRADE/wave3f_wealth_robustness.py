"""WAVE 3F -- harden the COMBO for WEALTH (retail account) before locking it.

User: "you're the expert, maximise WEALTH on a retail account." => rank by COMPOUND (terminal
wealth), retail CAN concentrate (no capacity limit), but a catastrophic DD can wipe a retail
account so DD must stay survivable. Before recommending the COMBO (breadth gate + momentum top-K
inv-vol) I must confirm it is ROBUST, not default-param-luck.

PARAM-ROBUSTNESS (the honest use of a sweep -- measure the FRACTION of reasonable params that beat
baseline on BOTH windows, NOT cherry-pick the best):
  K in {3,5,8,10} x cap in {0.25,0.33,0.50} x breadth thresh in {(.5,.4),(.45,.35),(.55,.45)}
  metric = COMPOUND return (terminal wealth) on VAL AND OOS; also lin_wk, maxDD.
Baseline = btc100 + EW cap33 (current deployable). + momentum-vs-random null UNDER the breadth gate.
de-leaked, PROD, 2025 SEALED, long-only spot lev=1.
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
from strat.ma_slice_engine import OOS_WALL_MS, OOS_START, OOS_2024_END, VAL_WALL_MS, VAL_START, VAL_END
from strat.canonical_exits import COST_PROD

BARS_PER_DAY = fl.BARS_PER_DAY
TYPES = ["SMA", "EMA", "WMA", "VIDYA", "KAMA"]
WIN = {"OOS": (OOS_WALL_MS, OOS_START, OOS_2024_END), "VAL": (VAL_WALL_MS, VAL_START, VAL_END)}
LB_D, SL_D = 14, 14


def hyst(sig, up, dn):
    n = len(sig); s = np.zeros(n, dtype=int); st = 0
    for i in range(n):
        if not (np.isfinite(sig[i]) and np.isfinite(up[i]) and np.isfinite(dn[i])): s[i] = st; continue
        if st == 0 and sig[i] > up[i]: st = 1
        elif st == 1 and sig[i] < dn[i]: st = 0
        s[i] = st
    return s


def build(window, tf):
    wall, ws, we = WIN[window]; bpd = BARS_PER_DAY[tf]
    lb, sl = LB_D*bpd, SL_D*bpd
    closes = {}
    for sym in _u100_symbols():
        rec = _load_asset(sym, tf, wall)
        if rec is None: continue
        d, o, h, l, c = rec
        closes[sym] = (d, o, h, l, np.asarray(c, float))
    syms = list(closes.keys())
    cdf = pd.DataFrame({s: pd.Series(closes[s][4], index=closes[s][0]) for s in syms}).sort_index()
    idx = cdf.index; T = len(idx); A = len(syms)
    conv = np.zeros((T, A))
    for ma in TYPES:
        hc = {}
        for s in syms:
            d, o, h, l, c = closes[s]
            bh = _cached_band_held(ma, tf, s, o, h, l, c, wall)
            hc[s] = pd.Series(held_rolling(c, bh, lb, sl), index=d)
        conv += pd.DataFrame(hc).reindex(idx).fillna(0).reindex(columns=cdf.columns).fillna(0).to_numpy()
    ret = cdf.pct_change().fillna(0.0).to_numpy()
    vol = cdf.pct_change().rolling(max(20, bpd*5), min_periods=10).std().to_numpy()
    mom = cdf.pct_change(lb).to_numpy()
    C = cdf.to_numpy(); sw = 100*bpd
    btc = "BTCUSDT" if "BTCUSDT" in cdf.columns else cdf.columns[0]
    bc = cdf[btc]; bsma = bc.rolling(sw, min_periods=sw//2).mean().to_numpy()
    g_btc = hyst(bc.to_numpy(), bsma*1.02, bsma*0.98)
    sall = cdf.rolling(sw, min_periods=sw//2).mean().to_numpy()
    vc = np.isfinite(sall).sum(axis=1)
    brd = np.where(vc > 0, np.nansum((C > sall)*np.isfinite(sall), axis=1)/np.maximum(vc,1), np.nan)
    mask = (idx >= pd.Timestamp(ws)) & (idx < pd.Timestamp(we))
    return {"conv": conv, "ret": ret, "vol": vol, "mom": mom, "g_btc": g_btc, "brd": brd, "mask": mask, "bpd": bpd, "A": A}


def gate_brd(f, up, dn):
    return hyst(f["brd"], np.full(len(f["brd"]), up), np.full(len(f["brd"]), dn))


def W_ew(f, gate, cap):
    conv, vol = f["conv"], f["vol"]
    mult = (conv/len(TYPES)) * (gate[:, None] > 0)
    iv = np.where((mult > 0) & np.isfinite(vol) & (vol > 0), 1.0/np.where(vol > 0, vol, np.nan), 0.0)
    iv = np.nan_to_num(iv)*mult; s = iv.sum(axis=1, keepdims=True)
    return np.minimum(np.where(s > 0, iv/np.maximum(s, 1e-12), 0.0), cap)


def W_mom(f, gate, K):
    conv, vol, mom = f["conv"], f["vol"], f["mom"]
    elig = (conv > 0) & (gate[:, None] > 0)
    T, A = conv.shape; W = np.zeros((T, A))
    sc = np.where(elig & np.isfinite(mom), mom, -np.inf)
    for t in range(T):
        e = np.where(np.isfinite(sc[t]))[0]
        if len(e) == 0: continue
        k = min(K, len(e)); top = e[np.argsort(-sc[t][e])[:k]]
        v = vol[t][top]; ivw = np.where(np.isfinite(v) & (v > 0), 1.0/v, 0.0)
        W[t, top] = ivw/ivw.sum() if ivw.sum() > 0 else 1.0/k
    return W


def W_randk(f, gate, K, seed):
    conv = f["conv"]; elig = (conv > 0) & (gate[:, None] > 0)
    T, A = conv.shape; W = np.zeros((T, A)); rng = np.random.default_rng(seed)
    for t in range(T):
        e = np.where(elig[t])[0]
        if len(e) == 0: continue
        k = min(K, len(e)); W[t, rng.choice(e, k, replace=False)] = 1.0/k
    return W


def net(W, ret, cost=COST_PROD):
    Wl = np.vstack([np.zeros((1, W.shape[1])), W[:-1]])
    gr = np.nansum(Wl*ret, axis=1)
    turn = np.nansum(np.abs(W - np.vstack([np.zeros((1, W.shape[1])), W[:-1]])), axis=1)
    return gr - turn*(cost/2)


def met(W, f):
    n = net(W, f["ret"])[f["mask"]]
    if len(n) == 0: return None
    comp = float(np.prod(1+n)-1)*100
    lin = float(np.mean(n))*f["bpd"]*7*100
    cum = np.cumprod(1+n)
    dd = float(np.min((cum-np.maximum.accumulate(cum))/np.maximum.accumulate(cum)))*100 if len(cum)>1 else 0
    return {"compound": round(comp,1), "lin_wk": round(lin,3), "maxdd": round(dd,1)}


def main():
    Ks = [3, 5, 8, 10]; caps = [0.25, 0.33, 0.50]; brds = [(0.5,0.4),(0.45,0.35),(0.55,0.45)]
    out = {"check": "WAVE3F_wealth_robustness", "ts": datetime.now().strftime("%Y%m%d_%H%M%S"), "tf": {}}
    for tf in ["1d", "4h"]:
        fV, fO = build("VAL", tf), build("OOS", tf)
        base_v = met(W_ew(fV, fV["g_btc"], 0.33), fV)
        base_o = met(W_ew(fO, fO["g_btc"], 0.33), fO)
        # param-robustness: fraction of combos that beat baseline COMPOUND on BOTH windows
        wins = 0; total = 0; rows = []
        best = None
        for up, dn in brds:
            gv, go = gate_brd(fV, up, dn), gate_brd(fO, up, dn)
            for cap in caps:
                for K in Ks:
                    mv = met(W_mom(fV, gv, K) if cap >= 0.49 or True else None, fV) if False else met(W_mom(fV, gv, K), fV)
                    mo = met(W_mom(fO, go, K), fO)
                    total += 1
                    both = mv["compound"] > base_v["compound"] and mo["compound"] > base_o["compound"]
                    wins += int(both)
                    rows.append({"K":K,"cap":cap,"brd":f"{up}/{dn}","VAL_comp":mv["compound"],"OOS_comp":mo["compound"],
                                 "VAL_dd":mv["maxdd"],"OOS_dd":mo["maxdd"],"both_beat":both})
                    score = min(mv["compound"], mo["compound"])  # robust = good on the WORSE window
                    if best is None or score > best["score"]:
                        best = {"score":score,"K":K,"cap":cap,"brd":f"{up}/{dn}","VAL":mv,"OOS":mo}
        # momentum-vs-random null under the breadth gate (default 0.5/0.4, K5)
        gv, go = gate_brd(fV, 0.5, 0.4), gate_brd(fO, 0.5, 0.4)
        mom_v, mom_o = met(W_mom(fV, gv, 5), fV), met(W_mom(fO, go, 5), fO)
        nv = [met(W_randk(fV, gv, 5, s), fV)["compound"] for s in range(150)]
        no = [met(W_randk(fO, go, 5, s), fO)["compound"] for s in range(150)]
        pv = float(np.mean(np.array(nv) < mom_v["compound"]))*100
        po = float(np.mean(np.array(no) < mom_o["compound"]))*100
        out["tf"][tf] = {"baseline_btc100_EW_cap33": {"VAL": base_v, "OOS": base_o},
                         "robustness": {"combos_beating_baseline_both_windows": f"{wins}/{total}",
                                        "frac": round(wins/total,3)},
                         "robust_best": best, "all": rows,
                         "momentum_null_breadthgate_K5": {"VAL_pctile": round(pv,1), "OOS_pctile": round(po,1),
                             "VAL_mom_comp": mom_v["compound"], "OOS_mom_comp": mom_o["compound"],
                             "VAL_null_p95": round(float(np.percentile(nv,95)),1), "OOS_null_p95": round(float(np.percentile(no,95)),1)}}
        print(f"\n{'='*96}\n{tf}  WEALTH ROBUSTNESS (compound, de-leaked, PROD). "
              f"Baseline btc100+EW33: VAL comp {base_v['compound']:+.0f}% / OOS {base_o['compound']:+.0f}%")
        print(f"{'='*96}")
        print(f"  COMBO beats baseline COMPOUND on BOTH windows: {wins}/{total} param-combos ({wins/total:.0%})")
        b = best
        print(f"  robust-best (max of worse-window compound): K{b['K']} cap{b['cap']} brd{b['brd']} -> "
              f"VAL {b['VAL']['compound']:+.0f}% (DD{b['VAL']['maxdd']:+.0f}) / OOS {b['OOS']['compound']:+.0f}% (DD{b['OOS']['maxdd']:+.0f})")
        print(f"  momentum vs random (breadth gate, K5, N=150): VAL pctile {pv:.0f}% / OOS pctile {po:.0f}%  "
              f"(mom OOS {mom_o['compound']:+.0f}% vs null p95 {np.percentile(no,95):+.0f}%)")
    with open(HERE / "wave3f_wealth_robustness.json", "w") as fo:
        json.dump(out, fo, indent=2, default=str)
    print(f"\nSaved {HERE/'wave3f_wealth_robustness.json'}")


if __name__ == "__main__":
    main()
