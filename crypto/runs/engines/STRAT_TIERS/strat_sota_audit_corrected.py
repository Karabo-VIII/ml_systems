"""SOLVE-FOR-SOTA -- CORRECTED honest stats (supersedes the over-kind claims the red-team refuted).

Adversarial red-team (workflow wbw77ivx2) refuted my significance + robustness framing. This locks in the HONEST
numbers, deterministic at seed=0, with reproducibility lineage (sklearn version + git SHA):
  1. AUTOCORRELATION-adjusted significance: iid t-test inflates (ACF1-3 positive). Report AR1 n_eff + Newey-West(L=3) p.
  2. LUMPINESS: top-k-week-removal table (what % of compound the top 1/2/3/5 weeks carry) + skew/kurt.
  3. COMPONENT DECOMPOSITION: breadth-ALONE vs Pwf-ALONE vs full V5 -> which piece is the real edge?
  4. DSR at honest trial counts (24 / 100 / 200) -- the 24-count under-counts the real search (hundreds).
  5. shuffle-null at 1000 (was under-resolved at 200) + exact empirical-p bound.
1d, u100, momentum top-8, seed=0, de-leaked, next-open, prod. lev=1 long-only.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json, subprocess
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd
import sklearn
from scipy import stats as ss

HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE.parent/"STRAT_2to7"))
ROOT=Path(__file__).resolve().parents[3]/"src"; sys.path.insert(0,str(ROOT))
import strat_2to7_movers as M, strat_2to7_walkforward as W
from strat.canonical_exits import COST_PROD
BPD=M.BPD; SB=7*BPD; CUT=pd.Timestamp("2024-05-15"); OOS25=pd.Timestamp("2025-01-01")


def book_net(conv,g,Pwf,mom,vol,roo,T,A,use_breadth,use_pwf):
    gate=(conv>0)
    if use_breadth: gate=gate&(g[:,None]>0)
    if use_pwf: gate=gate&(Pwf[:,None]>0.55)
    Wt=np.zeros((T,A)); s=np.where(gate&np.isfinite(mom),mom,-np.inf)
    for t in range(T):
        e=np.where(np.isfinite(s[t]))[0]
        if len(e)==0: continue
        k=min(8,len(e)); top=e[np.argsort(-s[t][e])[:k]]; v=vol[t][top]; w=np.where(np.isfinite(v)&(v>0),1.0/v,0.0); Wt[t,top]=w/w.sum() if w.sum()>0 else 1.0/k
    Wl=np.vstack([np.zeros((1,A)),Wt[:-1]]); return np.nansum(Wl*roo,axis=1)-np.nansum(np.abs(Wt-Wl),axis=1)*(COST_PROD/2)


def newey_west_t(wk, L=3):
    """t-stat for mean(wk)>0 with Newey-West HAC variance (L lags). Returns (t, one-sided p, n_eff_AR1)."""
    n=len(wk); mu=np.mean(wk); x=wk-mu; g0=np.dot(x,x)/n
    var=g0
    for l in range(1,L+1):
        gl=np.dot(x[l:],x[:-l])/n; var+=2*(1-l/(L+1))*gl
    se=np.sqrt(max(var,1e-12)/n); t=mu/se; p=float(1-ss.t.cdf(t,n-1))
    r1=np.dot(x[1:],x[:-1])/np.dot(x,x); n_eff=n*(1-r1)/(1+r1) if (1+r1)>0 else n
    return float(t),p,float(n_eff),float(r1)


def main():
    f=M.build("FWD"); idx=f["idx"]; conv=f["conv"]; vol=f["vol"]; roo=f["roo"]; brd=f["brd"]; T=f["T"]; A=f["A"]; mom=f["scores"]["mom14"]
    g=M.hyst(brd,0.5,0.4); X,lab,ewfwd,_,_=W.features_labels(f); Pwf=np.nan_to_num(W.walkforward_P(X,lab,ewfwd,idx,T,seed=0),nan=0.0)
    net=book_net(conv,g,Pwf,mom,vol,roo,T,A,True,True)
    try: sha=subprocess.run(["git","rev-parse","--short","HEAD"],cwd=str(ROOT.parent),capture_output=True,text=True).stdout.strip()
    except Exception: sha="unknown"
    out={"check":"STRAT_SOTA_audit_corrected","lineage":{"sklearn":sklearn.__version__,"git_sha":sha,"seed":0},"ts":datetime.now().strftime("%Y%m%d_%H%M%S"),"windows":{}}
    print("="*94); print("CORRECTED HONEST STATS (post red-team) -- V5, seed=0, prod, long-only"); print(f"  lineage: sklearn {sklearn.__version__}  git {sha}"); print("="*94)
    for wn,wm in [("OOS-2024",(idx>=CUT)&(idx<OOS25)),("OOS-2025",(idx>=OOS25)),("OOS-ALL",(idx>=CUT))]:
        n=net[wm]; wk=np.array([float(np.prod(1+n[i:i+SB])-1) for i in range(0,len(n)-SB+1,SB)])
        comp=float(np.prod(1+n)-1)*100
        # lumpiness: contribution of top-k weeks to compound (log-additive)
        order=np.argsort(-wk); logtot=np.sum(np.log1p(wk))
        def topk_share(k):
            keep=np.ones(len(wk),bool); keep[order[:k]]=False
            return round((1-(np.exp(np.sum(np.log1p(wk[keep])))-1)/(np.exp(logtot)-1))*100,1) if (np.exp(logtot)-1)!=0 else None
        # significance (NW + iid for contrast)
        t_nw,p_nw,neff,r1=newey_west_t(wk,3); t_iid=np.mean(wk)/(np.std(wk,ddof=1)/np.sqrt(len(wk))); p_iid=float(1-ss.t.cdf(t_iid,len(wk)-1))
        out["windows"][wn]={"compound":round(comp,1),"n_weeks":len(wk),"skew":round(float(ss.skew(wk)),2),"kurt":round(float(ss.kurtosis(wk,fisher=False)),1),
            "top1_share":topk_share(1),"top2_share":topk_share(2),"top3_share":topk_share(3),"top5_share":topk_share(5),
            "p_iid":round(p_iid,4),"p_newey_west_L3":round(p_nw,4),"n_eff_AR1":round(neff,0),"acf1":round(r1,3)}
        print(f"\n  {wn}: compound {comp:+.0f}%  ({len(wk)} weeks, skew {ss.skew(wk):+.1f}, kurt {ss.kurtosis(wk,fisher=False):.0f})")
        print(f"    LUMPINESS: top-1 wk = {topk_share(1)}% of compound | top-2 = {topk_share(2)}% | top-3 = {topk_share(3)}% | top-5 = {topk_share(5)}%")
        print(f"    SIGNIFICANCE: iid p={p_iid:.3f}  vs  Newey-West(L3) p={p_nw:.3f}  (ACF1={r1:+.2f}, n_eff {neff:.0f}/{len(wk)})  {'NOT sig' if p_nw>0.05 else 'sig'}")
    # component decomposition (OOS-ALL)
    wm=(idx>=CUT)
    nb=book_net(conv,g,Pwf,mom,vol,roo,T,A,True,False); npf=book_net(conv,g,Pwf,mom,vol,roo,T,A,False,True); nf=net
    comp_=lambda x:round(float(np.prod(1+x[wm])-1)*100,1)
    bear=(idx>=OOS25)
    out["component_decomp_OOS_ALL"]={"breadth_alone":comp_(nb),"pwf_alone":comp_(npf),"full_V5":comp_(nf),
        "breadth_alone_bear":round(float(np.prod(1+nb[bear])-1)*100,1),"pwf_alone_bear":round(float(np.prod(1+npf[bear])-1)*100,1)}
    print(f"\n  COMPONENT DECOMPOSITION (OOS-ALL): breadth-ALONE {comp_(nb):+.0f}% | Pwf-ALONE {comp_(npf):+.0f}% | full V5 {comp_(nf):+.0f}%")
    print(f"    -> the REAL edge is the {'BREADTH gate' if comp_(nb)>comp_(npf) else 'Pwf timing'} (de-risking); the timing overlay adds the lumpy mega-week interaction")
    print(f"    bear: breadth-alone {out['component_decomp_OOS_ALL']['breadth_alone_bear']:+.0f}% | Pwf-alone {out['component_decomp_OOS_ALL']['pwf_alone_bear']:+.0f}%")
    # DSR at honest trial counts (OOS-ALL)
    wk=np.array([float(np.prod(1+nf[wm][i:i+SB])-1) for i in range(0,int(wm.sum())-SB+1,SB)]); sr=np.mean(wk)/np.std(wk)
    g3=ss.skew(wk); g4=ss.kurtosis(wk,fisher=False); emc=0.5772; z=ss.norm.ppf
    def dsr(nt):
        sr0=np.sqrt(1.0/len(wk))*((1-emc)*z(1-1.0/nt)+emc*z(1-1.0/(nt*np.e))); denom=np.sqrt(max(1e-9,1-g3*sr+(g4-1)/4.0*sr**2)); return round(float(ss.norm.cdf(((sr-sr0)*np.sqrt(len(wk)-1))/denom)),3)
    out["DSR_by_trialcount"]={"n24":dsr(24),"n100":dsr(100),"n200":dsr(200)}
    print(f"\n  DSR by honest trial count: n=24 -> {dsr(24)} | n=100 -> {dsr(100)} | n=200 -> {dsr(200)}   (real search ~hundreds -> DSR well below the 0.95 bar)")
    # shuffle-null at 1000 (OOS-ALL)
    rng=np.random.default_rng(1); base=comp_(nf); nulls=[]
    for _ in range(1000):
        Psh=rng.permutation(Pwf); ns=book_net(conv,g,Psh,mom,vol,roo,T,A,True,True); nulls.append(float(np.prod(1+ns[wm])-1)*100)
    nulls=np.array(nulls); beat=float(np.mean(nulls<base)); out["shuffle_null_1000"]={"base":round(base,1),"p95":round(float(np.percentile(nulls,95)),1),"frac_below_base":round(beat,3),"empirical_p":round(1-beat,4)}
    print(f"\n  shuffle-null (1000x): base {base:+.0f}% vs null p95 {np.percentile(nulls,95):+.0f}%  -> empirical p = {1-beat:.4f}  ({'beats null' if beat>0.95 else 'within null'})")
    with open(HERE/"strat_sota_audit_corrected.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\nSaved {HERE/'strat_sota_audit_corrected.json'}")


if __name__=="__main__":
    main()
