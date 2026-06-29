"""SOLVE FOR SOTA (within long-only spot, internal price data) -- v6: stack the orthogonal axes the 5-lens diagnosis surfaced.

Diagnosis: direction unforecastable (rank-IC~0); the recoverable edge is TIMING (when) + CONCENTRATION (which, momentum)
+ NEW levers: continuous/vol-targeted SIZING (variance IS forecastable; binary gate zeroes 62% of weeks) + residual-
momentum selection (raise effective breadth beyond ~7 BTC-clones). Stack them on the validated WF deploy-timing base.

BASE (V5) = breadth & WF-GBM-P(good-week)>0.55 binary gate + momentum top-8 inv-vol.
LEVERS (test marginal + stacked, OOS-2024 + OOS-2025):
  C continuous exposure: gross = clip((Pwf-0.45)/0.20, 0, 1) (proportional to timing-confidence, not binary on/off)
  V vol-target: scale gross by min(1, target_ann/ trailing-book-vol) [lev=1 -> scale DOWN only, de-risk high vol]
  R residual-momentum: rank by idio = mom14 - beta*BTC_mom14 (idiosyncratic, raises effective breadth)
Report per window: compound, wk_mean, wk_median, %wk>0, in[2,7], maxDD, deploy/avg-gross. vs ORACLE +12-14%/wk.
1d, u100, 14d/7d, de-leaked, next-open fills, prod cost. long-only spot lev=1.
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
BPD=M.BPD; SB=7*BPD; CUT=pd.Timestamp("2024-05-15"); OOS25=pd.Timestamp("2025-01-01")


def main():
    f=M.build("FWD"); idx=f["idx"]; conv=f["conv"]; vol=f["vol"]; roo=f["roo"]; brd=f["brd"]; cdf=f["cdf"]; T=f["T"]; A=f["A"]; mom=f["scores"]["mom14"]
    g=M.hyst(brd,0.5,0.4)
    X,lab,ewfwd,_,_=W.features_labels(f); Pwf=np.nan_to_num(W.walkforward_P(X,lab,ewfwd,idx,T),nan=0.0)
    jb=list(cdf.columns).index("BTCUSDT") if "BTCUSDT" in cdf.columns else 0
    idio=mom-mom[:,jb:jb+1]   # residual momentum vs BTC
    # base per-bar book (momentum top-8 inv-vol), gated; returns weight matrix given a (score, gross_vector)
    def book(score, gross):
        Wt=np.zeros((T,A)); s=np.where((conv>0)&(g[:,None]>0)&np.isfinite(score),score,-np.inf)
        for t in range(T):
            if gross[t]<=1e-9: continue
            e=np.where(np.isfinite(s[t]))[0]
            if len(e)==0: continue
            k=min(8,len(e)); top=e[np.argsort(-s[t][e])[:k]]; v=vol[t][top]; w=np.where(np.isfinite(v)&(v>0),1.0/v,0.0)
            w=w/w.sum() if w.sum()>0 else np.ones(k)/k; Wt[t,top]=w*gross[t]
        return Wt
    def net(Wt):
        Wl=np.vstack([np.zeros((1,A)),Wt[:-1]]); gr=np.nansum(Wl*roo,axis=1); turn=np.nansum(np.abs(Wt-Wl),axis=1)
        return gr-turn*(COST_PROD/2), Wl
    def voltarget(Wt, target=0.60):
        n0,_=net(Wt); rv=pd.Series(n0).rolling(20,min_periods=10).std().shift(1).to_numpy()*np.sqrt(365)
        sc=np.where(np.isfinite(rv)&(rv>1e-9),np.minimum(1.0,target/rv),1.0); return Wt*sc[:,None]
    # gross profiles
    gross_binary=(Pwf>0.55).astype(float)
    gross_cont=np.clip((Pwf-0.45)/0.20,0,1)
    configs={
        "V5 base (bin gate, mom)": book(mom,gross_binary),
        "+continuous expo":        book(mom,gross_cont),
        "+vol-target":             voltarget(book(mom,gross_binary)),
        "+residual-mom":           book(idio,gross_binary),
        "+cont +voltgt":           voltarget(book(mom,gross_cont)),
        "+cont +voltgt +residual": voltarget(book(idio,gross_cont)),
    }
    def stats(Wt,wm):
        nt,Wl=net(Wt); n=nt[wm]; eq=np.cumprod(1+n)
        dd=float(np.min((eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq))*100) if len(eq)>1 else 0
        sb=SB; wk=[float(np.prod(1+n[i:i+sb])-1)*100 for i in range(0,len(n)-sb+1,sb)]; wk=np.array(wk) if wk else np.array([0.0])
        gavg=float(np.mean(Wl.sum(axis=1)[wm]))
        return {"comp":round(float(eq[-1]-1)*100,1),"wk_mean":round(float(np.mean(wk)),2),"wk_med":round(float(np.median(wk)),2),
                "pos":round(float(np.mean(wk>0)),2),"in27":round(float(np.mean((wk>=2)&(wk<=7))),2),"maxdd":round(dd,1),"avg_gross":round(gavg,2)}
    out={"check":"STRAT_SOTA_v6","ts":datetime.now().strftime("%Y%m%d_%H%M%S"),"windows":{}}
    for wn,wm in [("OOS-2024",(idx>=CUT)&(idx<OOS25)),("OOS-2025",(idx>=OOS25))]:
        out["windows"][wn]={}
        print(f"\n{'='*100}\n{wn}  SOTA-v6 lever stack (WF-timing + momentum + sizing/selection), 1d, de-leaked, prod")
        print(f"{'='*100}")
        print(f"  {'config':<26}{'comp%':>8}{'wk_mean':>9}{'wk_med':>8}{'%wk>0':>7}{'in[2,7]':>9}{'maxDD':>7}{'avg_gross':>10}")
        print("  "+"-"*86)
        for nm,Wt in configs.items():
            s=stats(Wt,wm); out["windows"][wn][nm]=s
            print(f"  {nm:<26}{s['comp']:>+7.0f}%{s['wk_mean']:>+8.2f}%{s['wk_med']:>+7.2f}%{s['pos']*100:>6.0f}%{s['in27']*100:>8.0f}%{s['maxdd']:>+7.0f}{s['avg_gross']:>10.2f}")
    with open(HERE/"strat_sota_v6.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\nSaved {HERE/'strat_sota_v6.json'}")


if __name__=="__main__":
    main()
