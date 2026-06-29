"""STRAT TIERS -- the coverage hypothesis: does a long-only MEAN-REVERSION engine DE-CORRELATE from momentum?

The ranker found trend engines are correlated (deploy same weeks) -> no coverage. The only de-correlation path:
a long-only MR engine (buy OVERSOLD dips) that deploys in the CHOP/DOWN weeks momentum sits out. Test: MR held =
RSI(14)<35 oversold-buy, exit>55, EW, u100, 1d, NOT breadth-gated (so it can deploy when momentum is cash). Measure:
weekly net, CORR with momentum8 weekly net, % of momentum-CASH weeks where MR deploys (coverage), and combined
(momentum-when-deployed-else-MR). If MR is positive AND fills momentum's cash weeks -> genuine coverage tier; if MR
is negative (falling knives) -> coverage refuted, V5-alone is final.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd

HERE=Path(__file__).resolve().parent; ROOT=Path(__file__).resolve().parents[3]/"src"; sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(HERE.parent/"STRAT_2to7"))
import strat_2to7_movers as M
import strat.ti_engine_sota as T
from strat.canonical_exits import COST_PROD
BPD=M.BPD; SB=7*BPD; CUT=pd.Timestamp("2024-05-15")


def net_from_W(W,roo):
    Wl=np.vstack([np.zeros((1,W.shape[1])),W[:-1]]); gr=np.nansum(Wl*roo,axis=1)
    turn=np.nansum(np.abs(W-np.vstack([np.zeros((1,W.shape[1])),W[:-1]])),axis=1); return gr-turn*(COST_PROD/2)


def main():
    f=M.build("FWD"); idx=f["idx"]; conv=f["conv"]; vol=f["vol"]; roo=f["roo"]; brd=f["brd"]; cdf=f["cdf"]; T_=f["T"]; A=f["A"]; mom=f["scores"]["mom14"]
    Cn=cdf.to_numpy(); g=M.hyst(brd,0.5,0.4)
    # momentum8 (breadth-gated) weights
    Wm=np.zeros((T_,A)); s=np.where((conv>0)&(g[:,None]>0)&np.isfinite(mom),mom,-np.inf)
    for t in range(T_):
        e=np.where(np.isfinite(s[t]))[0]
        if len(e)==0: continue
        k=min(8,len(e)); top=e[np.argsort(-s[t][e])[:k]]; v=vol[t][top]; w=np.where(np.isfinite(v)&(v>0),1.0/v,0.0); Wm[t,top]=w/w.sum() if w.sum()>0 else 1.0/k
    # MR engine: RSI<35 oversold-buy, exit>55, EW, NOT breadth-gated
    Hd=np.zeros((T_,A))
    for j in range(A): Hd[:,j]=T.rsi_single_held(Cn[:,j],14,35,55)  # held when RSI in oversold-recovery band
    # invert: rsi_single_held holds when RSI>long_thr; for MR we want OVERSOLD so use a low long_thr trick ->
    # define MR held = close < rolling mean - 1 std (oversold) instead, cleaner:
    ret=cdf.pct_change(); mu=cdf.rolling(20*BPD,min_periods=10).mean().to_numpy(); sd=cdf.rolling(20*BPD,min_periods=10).std().to_numpy()
    z=(Cn-mu)/np.where(sd>0,sd,np.nan)
    mr_held=(np.nan_to_num(z,nan=0.0) < -1.0).astype(float)   # oversold: >1 std below 20d mean
    sM=mr_held.sum(axis=1,keepdims=True); Wmr=np.where(sM>0,mr_held/np.maximum(sM,1),0.0)
    nm=net_from_W(Wm,roo); nmr=net_from_W(Wmr,roo)
    wk=list(range(0,T_-SB+1,SB)); wkd=[idx[i] for i in wk]
    wm=np.array([float(np.prod(1+nm[i:i+SB])-1) for i in wk]); wmr=np.array([float(np.prod(1+nmr[i:i+SB])-1) for i in wk])
    gm=(np.vstack([np.zeros((1,A)),Wm[:-1]]).sum(axis=1)>0); gmr=(np.vstack([np.zeros((1,A)),Wmr[:-1]]).sum(axis=1)>0)
    dep_m=np.array([np.mean(gm[i:i+SB])>0.5 for i in wk]); dep_mr=np.array([np.mean(gmr[i:i+SB])>0.5 for i in wk])
    oos=np.array([d>=CUT for d in wkd])
    corr=float(np.corrcoef(wm[oos],wmr[oos])[0,1])
    # coverage: of momentum-CASH oos weeks, how many does MR deploy?
    mcash=oos&(~dep_m); fill=float(np.mean(dep_mr[mcash])) if mcash.sum() else 0
    def st(x):
        x=x[oos]; eq=np.cumprod(1+x); return round(float(eq[-1]-1)*100,1),round(float(np.mean(x))*100,2),round(float(np.mean(x>0)),3)
    mc,mm_,mp=st(wm); mrc,mrm,mrp=st(wmr)
    # combined: momentum when it deploys, else MR
    comb=np.where(dep_m,wm,np.where(dep_mr,wmr,0.0)); cc,cm,cp=st(comb)
    cov_comb=float(np.mean((dep_m|dep_mr)[oos]))
    out={"check":"STRAT_TIERS_mr","corr_mom_mr":round(corr,3),"mr_fills_mom_cash_frac":round(fill,3),
         "momentum":{"comp":mc,"wk_mean":mm_,"wk_pos":mp,"deploy":round(float(np.mean(dep_m[oos])),3)},
         "mean_rev":{"comp":mrc,"wk_mean":mrm,"wk_pos":mrp,"deploy":round(float(np.mean(dep_mr[oos])),3)},
         "combined":{"comp":cc,"wk_mean":cm,"wk_pos":cp,"coverage":round(cov_comb,3)},"ts":datetime.now().strftime("%Y%m%d_%H%M%S")}
    print("="*82); print("MEAN-REVERSION coverage test (does MR de-correlate + fill momentum's cash weeks?)"); print("="*82)
    print(f"  corr(momentum_wk, MR_wk) OOS = {corr:+.3f}   (low/neg = de-correlated)")
    print(f"  momentum:  comp {mc:+.0f}%  wk_mean {mm_:+.2f}%  wk>0 {mp:.0%}  deploy {out['momentum']['deploy']:.0%}")
    print(f"  mean_rev:  comp {mrc:+.0f}%  wk_mean {mrm:+.2f}%  wk>0 {mrp:.0%}  deploy {out['mean_rev']['deploy']:.0%}")
    print(f"  MR fills {fill:.0%} of momentum's CASH weeks")
    print(f"  COMBINED (mom-else-MR): comp {cc:+.0f}%  wk_mean {cm:+.2f}%  wk>0 {cp:.0%}  COVERAGE {cov_comb:.0%} of weeks in-market")
    verdict = "MR is a COVERAGE tier (positive + fills cash weeks)" if (mrc>0 and fill>0.3) else "MR coverage REFUTED (negative or doesn't fill)"
    print(f"  VERDICT: {verdict}"); out["verdict"]=verdict
    with open(HERE/"strat_tiers_mr.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\nSaved {HERE/'strat_tiers_mr.json'}")


if __name__=="__main__":
    main()
