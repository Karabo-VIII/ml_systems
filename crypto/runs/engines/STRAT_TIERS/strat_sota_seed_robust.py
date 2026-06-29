"""SOLVE-FOR-SOTA next-value: GBM-SUBSAMPLE STABILITY of V5 -- the walk-forward GBM (subsample=0.7) is stochastic.

Caught during freeze: walkforward_P had NO random_state -> V5 compound swung +168/+215/+262% across runs. Fixed.
RED-TEAM CORRECTION (wbw77ivx2): this is NOT independent-draw seed-robustness -- all 10 "seeds" share the SAME data,
features, label, walk-forward split, and momentum book; they differ ONLY in the GBM's 70%-subsample RNG. So this
measures GBM-SUBSAMPLE STABILITY, not out-of-sample robustness. The honest reads here are the FRAGILITY signals:
OOS-2025 bear only 7/10 positive (FAILS the 10/10 bar), and a 133-294% magnitude swing (2.2x) on subsample alone --
which corroborates the DSR/block-boot lumpiness finding. Do NOT cite "10/10" as robustness.
1d, u100, momentum top-8, WF-timing gate, de-leaked, next-open, prod. lev=1 long-only.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd

HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE.parent/"STRAT_2to7"))
ROOT=Path(__file__).resolve().parents[3]/"src"; sys.path.insert(0,str(ROOT))
import strat_2to7_movers as M, strat_2to7_walkforward as W
from strat.canonical_exits import COST_PROD
BPD=M.BPD; SB=7*BPD; CUT=pd.Timestamp("2024-05-15"); OOS25=pd.Timestamp("2025-01-01"); SEEDS=range(10)


def main():
    f=M.build("FWD"); idx=f["idx"]; conv=f["conv"]; vol=f["vol"]; roo=f["roo"]; brd=f["brd"]; T=f["T"]; A=f["A"]; mom=f["scores"]["mom14"]
    g=M.hyst(brd,0.5,0.4); X,lab,ewfwd,_,_=W.features_labels(f)
    def book_net(Pwf):
        gate=(conv>0)&(g[:,None]>0)&(Pwf[:,None]>0.55); Wt=np.zeros((T,A)); s=np.where(gate&np.isfinite(mom),mom,-np.inf)
        for t in range(T):
            e=np.where(np.isfinite(s[t]))[0]
            if len(e)==0: continue
            k=min(8,len(e)); top=e[np.argsort(-s[t][e])[:k]]; v=vol[t][top]; w=np.where(np.isfinite(v)&(v>0),1.0/v,0.0); Wt[t,top]=w/w.sum() if w.sum()>0 else 1.0/k
        Wl=np.vstack([np.zeros((1,A)),Wt[:-1]]); return np.nansum(Wl*roo,axis=1)-np.nansum(np.abs(Wt-Wl),axis=1)*(COST_PROD/2), Wl
    def comp(n,wm): return float(np.prod(1+n[wm])-1)*100
    def dd(n,wm):
        eq=np.cumprod(1+n[wm]); return float(np.min((eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq))*100) if len(eq)>1 else 0
    rows={"OOS-2024":[],"OOS-2025":[],"OOS-ALL":[]}; live=[]
    for sd in SEEDS:
        Pwf=np.nan_to_num(W.walkforward_P(X,lab,ewfwd,idx,T,seed=sd),nan=0.0); n,Wl=book_net(Pwf)
        rows["OOS-2024"].append(comp(n,(idx>=CUT)&(idx<OOS25))); rows["OOS-2025"].append(comp(n,(idx>=OOS25))); rows["OOS-ALL"].append(comp(n,(idx>=CUT)))
        live.append(round(float(Pwf[T-1]),3))
    out={"check":"STRAT_SOTA_gbm_subsample_stability","caveat":"NOT independent draws -- same data/split, only GBM subsample RNG varies; measures stability, not robustness",
         "seeds":list(SEEDS),"ts":datetime.now().strftime("%Y%m%d_%H%M%S"),"windows":{},
         "live_P_by_seed":live,"live_deploy_votes":int(sum(p>0.55 for p in live))}
    print("="*92); print("V5 GBM-SUBSAMPLE STABILITY (seeds 0-9; NOT independent draws) -- fragility band, not robustness"); print("="*92)
    print(f"  {'window':<12}{'min%':>8}{'median%':>9}{'max%':>8}{'mean%':>8}{'pos/10':>8}   spread")
    print("  "+"-"*78)
    for wn,vals in rows.items():
        v=np.array(vals); out["windows"][wn]={"min":round(float(v.min()),1),"median":round(float(np.median(v)),1),"max":round(float(v.max()),1),"mean":round(float(v.mean()),1),"pos":int((v>0).sum())}
        print(f"  {wn:<12}{v.min():>+7.0f}%{np.median(v):>+8.0f}%{v.max():>+7.0f}%{v.mean():>+7.0f}%{(v>0).sum():>6}/10   [{v.min():+.0f}..{v.max():+.0f}]")
    print(f"\n  LIVE P(good-week) across seeds: {live}")
    print(f"  live deploy votes (P>0.55): {out['live_deploy_votes']}/10  -> {'DEPLOY' if out['live_deploy_votes']>=6 else 'CASH'} (majority)")
    with open(HERE/"strat_sota_seed_robust.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\nSaved {HERE/'strat_sota_seed_robust.json'}")


if __name__=="__main__":
    main()
