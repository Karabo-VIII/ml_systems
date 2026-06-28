"""FORWARD MOVER DIAGNOSIS -- "you can't have missed the movers all year and be net-negative."

User critique (2026-06-28, valid): the engine trades 7d windows on the full u100, yet ended NEGATIVE
over 2025-Q2..2026-05 despite a huge 2025 bull (basket Q3-25 +32.6%, BTC $85k->$124k). A -20% Q2-25
in a +2% market is the engine MISSING the movers, not "de-risk." Investigate, don't defend.

HYPOTHESES under test:
  H1 inv-vol weighting is ANTI-MOVER -- it overweights LOW-vol names; the movers are HIGH-vol.
  H2 momentum (mover-selection) was shelved on ONE quarter (Q1-2025 = a momentum CRASH, worst case);
     never tested over the FULL forward window with the 2025 bull in it.
  H3 the engine captures little of the available per-7d-window opportunity.

TEST: on the breadth-gated, band-confirmed eligible set, forward window 2025-Q2..2026-05, compare
SELECTION/WEIGHTING schemes (1d, 14d/7d, de-leaked, prod cost, next-open fills):
  EW | inv-vol cap33 (current) | mom top8 inv-vol | mom top8 EW | VOL-tilt top8 (chase high-vol movers)
  | mom top8 VOL-weighted (double down on movers)
plus the ORACLE: perfect top-8 by realized fwd-7d return per slice (the available-opportunity ceiling).
Quarterly breakdown so we see capture in the bull (Q3-25) vs the crashes (Q4-25/Q1-26).
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
import paper_trade_ma_family as P
from strat.canonical_exits import COST_PROD

BPD = P.BARS_PER_DAY[P.TF]; CAP = 0.33; K = 8; LB = P.LB_D*BPD


def net_from_W(W, ret, cost=COST_PROD):
    Wl = np.vstack([np.zeros((1, W.shape[1])), W[:-1]])
    gr = np.nansum(Wl*ret, axis=1)
    turn = np.nansum(np.abs(W - np.vstack([np.zeros((1, W.shape[1])), W[:-1]])), axis=1)
    return gr - turn*(cost/2)


def met(net, mask, bpd=BPD):
    n = net[mask]
    if len(n) == 0: return None
    eq = np.cumprod(1+n); comp = float(eq[-1]-1)*100
    dd = float(np.min((eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq)))*100 if len(eq)>1 else 0
    sh = (float(np.mean(n))*bpd*365)/(float(np.std(n))*np.sqrt(bpd*365)) if np.std(n)>1e-12 else 0
    return {"compound":round(comp,1),"sharpe":round(sh,2),"maxdd":round(dd,1)}


def quarterly(net, dates):
    s = pd.Series(net, index=pd.DatetimeIndex(dates))
    return {str(q): round(float(np.prod(1+g.to_numpy())-1)*100,1) for q,g in s.groupby(s.index.to_period("Q"))}


def main():
    f = P.build()
    elig = (f["conv"] > 0) & (f["g_brd"][:, None] > 0)
    vol = f["vol"]; ret = f["roo"]; mask = f["mask"]; idx = f["idx"]; cdf = f["cdf"]; A = f["A"]; T = len(idx)
    mom = cdf.pct_change(LB).to_numpy()                 # trailing 14d momentum (causal)
    fwd7 = np.full((T, A), np.nan)                      # realized FWD 7d return per name (oracle only)
    C = cdf.to_numpy()
    sb = 7*BPD
    for t in range(T-sb):
        fwd7[t] = C[t+sb]/np.where(C[t]>0, C[t], np.nan) - 1.0

    def W_topk(score, weight, k=K):
        W = np.zeros((T, A)); s = np.where(elig & np.isfinite(score), score, -np.inf)
        for t in range(T):
            e = np.where(np.isfinite(s[t]))[0]
            if len(e) == 0: continue
            kk = min(k, len(e)); top = e[np.argsort(-s[t][e])[:kk]]
            if weight == "ew": w = np.ones(kk)
            elif weight == "invvol":
                v = vol[t][top]; w = np.where(np.isfinite(v)&(v>0), 1.0/v, 0.0)
            elif weight == "vol":
                v = vol[t][top]; w = np.where(np.isfinite(v)&(v>0), v, 0.0)
            else: w = np.ones(kk)
            W[t, top] = w/w.sum() if w.sum()>0 else 1.0/kk
        return W

    def W_all(weight):
        W = np.zeros((T, A))
        for t in range(T):
            e = np.where(elig[t])[0]
            if len(e)==0: continue
            if weight=="ew": w=np.ones(len(e))
            elif weight=="invvol":
                v=vol[t][e]; w=np.where(np.isfinite(v)&(v>0),1.0/v,0.0)
            W[t,e]=np.minimum(w/w.sum() if w.sum()>0 else 1.0/len(e), CAP)
            if W[t].sum()>0: W[t]/=W[t].sum()
        return W

    schemes = {
        "EW all (eligible)":        W_all("ew"),
        "inv-vol cap33 (CURRENT)":  W_all("invvol"),
        "mom top8 inv-vol":         W_topk(mom, "invvol"),
        "mom top8 EW":              W_topk(mom, "ew"),
        "VOL-tilt top8 (movers)":   W_topk(vol, "ew"),         # pick the highest-vol eligible names
        "mom top8 VOL-weighted":    W_topk(mom, "vol"),
        "ORACLE perfect top8 (fwd)":W_topk(fwd7, "ew"),        # hindsight ceiling
    }
    out = {"check":"FORWARD_MOVER_DIAGNOSIS","window":f["mask"].sum(),"ts":datetime.now().strftime("%Y%m%d_%H%M%S"),"schemes":{}}
    dates_w = idx[mask]
    # opportunity premise: % of 7d slices with >=1 eligible name up >=10%
    print(f"{'='*100}\nFORWARD MOVER DIAGNOSIS  2025-Q2..2026-05 ({int(mask.sum())} bars)  -- selection/weighting on the SAME eligible set")
    print(f"{'='*100}")
    print(f"  {'scheme':<28}{'compound':>10}{'maxDD':>7}{'Sharpe':>7}  quarterly")
    print("  "+"-"*92)
    for nm, W in schemes.items():
        net = net_from_W(W, ret); m = met(net, mask); q = quarterly(net[mask], dates_w)
        out["schemes"][nm] = {**m, "quarterly": q}
        print(f"  {nm:<28}{m['compound']:>+9.1f}%{m['maxdd']:>+7.0f}{m['sharpe']:>+7.2f}  {q}")
    # benchmarks
    for w in ["alt","btc"]:
        nb, eb = P.bench_net(f, w); m = met(nb, np.ones(len(nb),bool))
        out["schemes"][f"BH_{w}"] = m
        print(f"  {'BH '+w:<28}{m['compound']:>+9.1f}%{m['maxdd']:>+7.0f}{m['sharpe']:>+7.2f}")
    with open(HERE/"forward_mover_diagnosis.json","w") as fo:
        json.dump(out, fo, indent=2, default=str)
    print(f"\nSaved {HERE/'forward_mover_diagnosis.json'}")


if __name__ == "__main__":
    main()
