"""STRAT 2-7%/wk -- Cycle 2: EXIT/HOLD ASYMMETRY on the momentum book (flip the negative deployed-week median).

Cycle 1 found score doesn't fix the bear; the deployed-week median is negative because the hold captures
full down-moves of reversers. Test a per-name exit OVERLAY on the momentum top-K book:
  STOP-LOSS (cut a held name when since-entry ret <= -sl)  -> truncate down-weeks, raise median
  TAKE-PROFIT (exit when since-entry ret >= +tp)           -> cap into the [2,7] band, raise hit-in-band
  TRAILING  (exit when ret falls trail from peak)          -> let winners run, cut reversals
DEAD-LIST CAVEAT (D23/RED-TEAM/exit-axis-null): stops HURT COMPOUND on the MA entry. But that was the COMPOUND
objective; here the objective is WEEKLY-DISTRIBUTION RELIABILITY (median/hit/in-band). Re-test for THIS objective,
report BOTH weekly-dist AND compound honestly. mom14 top-5, breadth-gated, 1d, de-leaked, prod, next-open fills, x-window.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import strat_2to7_movers as M   # reuse build(), constants
from strat.canonical_exits import COST_PROD

BPD = M.BPD; K = M.K; SL = M.SL


def book_with_exits(f, score, sl=None, tp=None, trail=None):
    """Momentum top-K, but a held name is FORCED OUT when since-entry close-ret hits -sl/+tp or trail-from-peak.
    Exited capital -> cash until the name drops out of top-K and re-enters. close-based check, open-to-open fill (lag-1)."""
    elig, vol, roo, T, A = f["elig"], f["vol"], f["roo"], f["T"], f["A"]
    # we need per-name CLOSE returns to track since-entry P&L; reconstruct close from roo? no -- use cumulative open-to-open as proxy
    # use the score panel's close via cumulative product of roo is wrong; instead track since-entry via roo cumprod per name.
    s = np.where(elig & np.isfinite(score), score, -np.inf)
    W = np.zeros((T, A))
    entry_cum = np.full(A, np.nan)      # cumulative open-index level at entry (use a synthetic price = cumprod roo)
    peak_cum = np.full(A, np.nan)
    price = np.cumprod(1 + np.nan_to_num(roo), axis=0)   # synthetic per-name price path (open-to-open)
    stopped = np.zeros(A, bool)
    for t in range(T):
        row = s[t]; e = np.where(np.isfinite(row))[0]
        desired = set()
        if len(e):
            kk = min(K, len(e)); desired = set(int(j) for j in e[np.argsort(-row[e])[:kk]])
        active = []
        for j in list(desired):
            if np.isnan(entry_cum[j]):                     # new entry
                entry_cum[j] = price[t, j]; peak_cum[j] = price[t, j]; stopped[j] = False
            peak_cum[j] = max(peak_cum[j], price[t, j])
            ret = price[t, j]/entry_cum[j] - 1.0
            dd = price[t, j]/peak_cum[j] - 1.0
            hit = ((sl is not None and ret <= -sl) or (tp is not None and ret >= tp) or
                   (trail is not None and dd <= -trail))
            if hit: stopped[j] = True
            if not stopped[j]: active.append(j)
        # reset entry for names no longer desired (so re-entry later is a fresh trade)
        for j in range(A):
            if j not in desired: entry_cum[j] = np.nan; peak_cum[j] = np.nan; stopped[j] = False
        if active:
            v = vol[t][active]; w = np.where(np.isfinite(v) & (v > 0), 1.0/v, 0.0)
            ww = w/w.sum() if w.sum() > 0 else np.ones(len(active))/len(active)
            for j, wj in zip(active, ww): W[t, j] = wj
    Wl = np.vstack([np.zeros((1, A)), W[:-1]]); gr = np.nansum(Wl*roo, axis=1)
    turn = np.nansum(np.abs(W-Wl), axis=1)
    book_with_exits._lastW = W                      # expose target weights for the live book
    return gr - turn*(COST_PROD/2), Wl


def wdist(net, Wl, mask):
    nm = net[mask]; gross = Wl.sum(axis=1)[mask]; wk = []; i = 0
    while i+SL <= len(nm):
        if np.mean(gross[i:i+SL] > 1e-9) > 0.5: wk.append(float(np.prod(1+nm[i:i+SL])-1)*100)
        i += SL
    if not wk: return None
    wk = np.array(wk); eq = np.cumprod(1+nm)
    return {"dep":len(wk),"median":round(float(np.median(wk)),2),"mean":round(float(np.mean(wk)),2),
            "hit2":round(float(np.mean(wk>=2)),3),"in27":round(float(np.mean((wk>=2)&(wk<=7))),3),
            "neg":round(float(np.mean(wk<0)),3),"comp":round(float(eq[-1]-1)*100,1)}


def main():
    F = {w: M.build(w) for w in ["VAL","OOS24","FWD"]}
    configs = [("baseline",None,None,None),("SL6",0.06,None,None),("SL10",0.10,None,None),
               ("TP7",None,0.07,None),("TP5",None,0.05,None),("SL6+TP7",0.06,0.07,None),
               ("SL10+TP10",0.10,0.10,None),("trail10",None,None,0.10),("SL8+trail12",0.08,None,0.12)]
    out = {"check":"STRAT_2to7_cycle2_exits","score":"mom14","K":K,"ts":datetime.now().strftime("%Y%m%d_%H%M%S"),"windows":{}}
    for w in ["VAL","OOS24","FWD"]:
        f = F[w]; sc = f["scores"]["mom14"]; out["windows"][w] = {}
        print(f"\n{'='*96}\n{w}  EXIT-ASYMMETRY on mom14 top-{K} (deployed-week dist + compound)")
        print(f"{'='*96}")
        print(f"  {'config':<13}{'dep':>5}{'median':>8}{'mean':>7}{'hit>=2':>8}{'in[2,7]':>9}{'neg':>6}{'compound':>10}")
        print("  "+"-"*76)
        for nm, sl, tp, tr in configs:
            net, Wl = book_with_exits(f, sc, sl, tp, tr); d = wdist(net, Wl, f["mask"])
            if d is None: continue
            out["windows"][w][nm] = d
            print(f"  {nm:<13}{d['dep']:>5}{d['median']:>+8.2f}{d['mean']:>+7.2f}{d['hit2']*100:>7.0f}%{d['in27']*100:>8.0f}%{d['neg']*100:>5.0f}%{d['comp']:>+9.0f}%")
    with open(HERE/"strat_2to7_exits.json","w") as fo: json.dump(out, fo, indent=2, default=str)
    print(f"\nSaved {HERE/'strat_2to7_exits.json'}")


if __name__ == "__main__":
    main()
