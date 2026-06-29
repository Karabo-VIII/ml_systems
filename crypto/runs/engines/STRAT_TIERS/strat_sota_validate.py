"""SOLVE FOR SOTA -- rigorous validation of the V5 (WF deploy-timing + momentum) -- real edge or overfit?

The 5-lens diagnosis: V5 is the realizable SOTA-within-constraints, BUT the deploy-timing gate is provisional and
has NOT passed DSR/CSCV-PBO -- "would not ship until it survives." So validate the V5 OOS net stream:
  1. Block-bootstrap p05 of weekly return (is the 5th-pctile > 0? robust, not a few-week fluke)
  2. Shuffle-null re-confirm (timing P shuffled in time -> compound collapses?)
  3. Deflated-Sharpe (Bailey/Lopez de Prado): deflate the OOS Sharpe by the N~24 feature/model/threshold trials run
  4. Per-window: OOS-2024 + OOS-2025 + combined, Sharpe + p05 + DSR
If p05>0 AND DSR clears -> the timing edge is real (shippable SOTA-within-constraints); else provisional.
1d, u100, momentum top-8, WF-timing gate, de-leaked, next-open, prod. lev=1 long-only.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd
from scipy import stats as ss

HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE.parent/"STRAT_2to7"))
ROOT=Path(__file__).resolve().parents[3]/"src"; sys.path.insert(0,str(ROOT))
import strat_2to7_movers as M, strat_2to7_walkforward as W
from strat.canonical_exits import COST_PROD
BPD=M.BPD; SB=7*BPD; CUT=pd.Timestamp("2024-05-15"); OOS25=pd.Timestamp("2025-01-01"); N_TRIALS=24


def main():
    f=M.build("FWD"); idx=f["idx"]; conv=f["conv"]; vol=f["vol"]; roo=f["roo"]; brd=f["brd"]; T=f["T"]; A=f["A"]; mom=f["scores"]["mom14"]
    g=M.hyst(brd,0.5,0.4); X,lab,ewfwd,_,_=W.features_labels(f); Pwf=np.nan_to_num(W.walkforward_P(X,lab,ewfwd,idx,T),nan=0.0)
    gate=(g[:,None]>0)&(Pwf[:,None]>0.55)
    Wt=np.zeros((T,A)); s=np.where((conv>0)&gate&np.isfinite(mom),mom,-np.inf)
    for t in range(T):
        e=np.where(np.isfinite(s[t]))[0]
        if len(e)==0: continue
        k=min(8,len(e)); top=e[np.argsort(-s[t][e])[:k]]; v=vol[t][top]; w=np.where(np.isfinite(v)&(v>0),1.0/v,0.0); Wt[t,top]=w/w.sum() if w.sum()>0 else 1.0/k
    Wl=np.vstack([np.zeros((1,A)),Wt[:-1]]); net=np.nansum(Wl*roo,axis=1)-np.nansum(np.abs(Wt-Wl),axis=1)*(COST_PROD/2)
    def weekly(n):
        return np.array([float(np.prod(1+n[i:i+SB])-1) for i in range(0,len(n)-SB+1,SB)])
    def dsr(wk,n_trials):
        # Bailey-LdP deflated Sharpe: prob the true SR>0 given n_trials of selection. wk = weekly returns.
        if len(wk)<8 or np.std(wk)<1e-9: return None
        sr=np.mean(wk)/np.std(wk); n=len(wk)
        g3=ss.skew(wk); g4=ss.kurtosis(wk,fisher=False)
        # expected max Sharpe under n_trials nulls (variance of SR estimates ~1/n)
        emc=0.5772; z=ss.norm.ppf; sr0=np.sqrt(1.0/n)*((1-emc)*z(1-1.0/n_trials)+emc*z(1-1.0/(n_trials*np.e)))
        denom=np.sqrt(max(1e-9,1 - g3*sr + (g4-1)/4.0*sr**2))
        dsr_stat=ss.norm.cdf(((sr-sr0)*np.sqrt(n-1))/denom)
        return {"sr_wk":round(float(sr),3),"sr_ann":round(float(sr*np.sqrt(52)),2),"sr0_haircut":round(float(sr0),3),"DSR":round(float(dsr_stat),3)}
    def bootp05(wk,nb=2000):
        if len(wk)<8: return None
        rng=np.random.default_rng(0); bl=4; idxs=np.arange(len(wk))
        comps=[]
        for _ in range(nb):
            # block bootstrap (block=4 weeks)
            pick=[]
            while len(pick)<len(wk):
                st=int(rng.integers(0,len(wk))); pick+=list(idxs[st:st+bl])
            pick=np.array(pick[:len(wk)]); comps.append(float(np.prod(1+wk[pick])-1)*100)
        return {"p05_compound":round(float(np.percentile(comps,5)),1),"median_compound":round(float(np.percentile(comps,50)),1),"p95":round(float(np.percentile(comps,95)),1)}
    def shufnull(wm,nb=200):
        rng=np.random.default_rng(1); base=float(np.prod(1+net[wm])-1)*100; nulls=[]
        for _ in range(nb):
            Psh=rng.permutation(Pwf); gt=(g[:,None]>0)&(Psh[:,None]>0.55)
            Ws=np.zeros((T,A)); ss_=np.where((conv>0)&gt&np.isfinite(mom),mom,-np.inf)
            for t in range(T):
                e=np.where(np.isfinite(ss_[t]))[0]
                if len(e)==0: continue
                k=min(8,len(e)); top=e[np.argsort(-ss_[t][e])[:k]]; v=vol[t][top]; w=np.where(np.isfinite(v)&(v>0),1.0/v,0.0); Ws[t,top]=w/w.sum() if w.sum()>0 else 1.0/k
            Wls=np.vstack([np.zeros((1,A)),Ws[:-1]]); ns=np.nansum(Wls*roo,axis=1)-np.nansum(np.abs(Ws-Wls),axis=1)*(COST_PROD/2)
            nulls.append(float(np.prod(1+ns[wm])-1)*100)
        return {"base_compound":round(base,1),"null_p50":round(float(np.percentile(nulls,50)),1),"null_p95":round(float(np.percentile(nulls,95)),1),"pctile":round(float(np.mean(np.array(nulls)<base))*100,1)}
    out={"check":"STRAT_SOTA_validate_v5","n_trials_for_dsr":N_TRIALS,"ts":datetime.now().strftime("%Y%m%d_%H%M%S"),"windows":{}}
    print("="*88); print("VALIDATE V5 (WF deploy-timing + momentum top-8) -- DSR + block-boot p05 + shuffle-null"); print("="*88)
    for wn,wm in [("OOS-2024",(idx>=CUT)&(idx<OOS25)),("OOS-2025",(idx>=OOS25)),("OOS-ALL",(idx>=CUT))]:
        wk=weekly(net[wm]); comp=float(np.prod(1+net[wm])-1)*100
        D=dsr(wk,N_TRIALS); B=bootp05(wk);
        out["windows"][wn]={"compound":round(comp,1),"n_weeks":len(wk),"dsr":D,"bootstrap":B}
        print(f"\n  {wn}: compound {comp:+.0f}% over {len(wk)} weeks")
        if D: print(f"     Sharpe(ann) {D['sr_ann']}  DSR(>0 prob, deflated by {N_TRIALS} trials) = {D['DSR']}  {'PASS' if D['DSR']>0.95 else 'FAIL(<0.95)'}")
        if B: print(f"     block-boot compound p05 {B['p05_compound']:+.0f}% / p50 {B['median_compound']:+.0f}% / p95 {B['p95']:+.0f}%  {'p05>0 ROBUST' if B['p05_compound']>0 else 'p05<=0 NOT robust'}")
    print("\n  shuffle-null (OOS-ALL, 200x, timing P shuffled in time):")
    S=shufnull((idx>=CUT)); out["shuffle_null_OOS_ALL"]=S
    print(f"     base {S['base_compound']:+.0f}% vs null p50 {S['null_p50']:+.0f}% / p95 {S['null_p95']:+.0f}%  -> base at {S['pctile']:.0f}th pctile  {'BEATS null' if S['pctile']>95 else 'within null'}")
    with open(HERE/"strat_sota_validate.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\nSaved {HERE/'strat_sota_validate.json'}")


if __name__=="__main__":
    main()
