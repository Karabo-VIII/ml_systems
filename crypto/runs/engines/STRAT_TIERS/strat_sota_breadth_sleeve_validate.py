"""AUDIT: validate the BREADTH-GATED MOMENTUM SLEEVE (the claimed 'real deployable') with the SAME rigor as V5.

The red-team component-decomp said breadth-alone +57% is the robust edge, Pwf-timing-alone -7%, full +170%. I then
CLAIMED in the runbook that the breadth-gated momentum sleeve is 'the honest deployable' -- but NEVER ran the robustness
battery on it (Pattern R: a mechanism claim un-verified against data). This closes that gap: run the breadth-ALONE
sleeve (conv>0 & breadth-hysteresis, momentum top-8 inv-vol, NO Pwf timing gate) through Newey-West HAC p + block-boot
p05 + DSR + top-k-week lumpiness + deploy rate + vs BTC buy-hold + maxDD, bull/bear/all. Verdict: is the sleeve a ROBUST
deployable (de-risking product) or is it ALSO fragile -> everything paper-trade-only?
1d, u100, momentum top-8, seed=0, de-leaked, next-open, prod. lev=1 long-only spot.
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
    Wl=np.vstack([np.zeros((1,A)),Wt[:-1]]); net=np.nansum(Wl*roo,axis=1)-np.nansum(np.abs(Wt-Wl),axis=1)*(COST_PROD/2)
    return net, Wl


def nw_p(wk,L=3):
    n=len(wk); mu=np.mean(wk); x=wk-mu; var=np.dot(x,x)/n
    for l in range(1,L+1): var+=2*(1-l/(L+1))*(np.dot(x[l:],x[:-l])/n)
    se=np.sqrt(max(var,1e-12)/n); t=mu/se; return float(t),float(1-ss.t.cdf(t,n-1))


def bootp05(wk,nb=2000):
    if len(wk)<8: return None
    rng=np.random.default_rng(0); bl=4; comps=[]
    for _ in range(nb):
        pick=[]
        while len(pick)<len(wk):
            st=int(rng.integers(0,len(wk))); pick+=list(range(st,min(st+bl,len(wk))))
        pick=np.array(pick[:len(wk)]); comps.append(float(np.prod(1+wk[pick]/100)-1)*100)
    return round(float(np.percentile(comps,5)),1)


def topk_share(wk,k):
    order=np.argsort(-wk); logtot=np.sum(np.log1p(wk/100)); tot=np.exp(logtot)-1
    if abs(tot)<1e-9: return None
    keep=np.ones(len(wk),bool); keep[order[:k]]=False
    return round((1-(np.exp(np.sum(np.log1p(wk[keep]/100)))-1)/tot)*100,1)


def dsr(wk,nt):
    if len(wk)<8 or np.std(wk)<1e-9: return None
    sr=np.mean(wk)/np.std(wk); n=len(wk); g3=ss.skew(wk); g4=ss.kurtosis(wk,fisher=False); emc=0.5772; z=ss.norm.ppf
    sr0=np.sqrt(1.0/n)*((1-emc)*z(1-1.0/nt)+emc*z(1-1.0/(nt*np.e))); denom=np.sqrt(max(1e-9,1-g3*sr+(g4-1)/4.0*sr**2))
    return round(float(ss.norm.cdf(((sr-sr0)*np.sqrt(n-1))/denom)),3)


def main():
    f=M.build("FWD"); idx=f["idx"]; conv=f["conv"]; vol=f["vol"]; roo=f["roo"]; brd=f["brd"]; cdf=f["cdf"]; T=f["T"]; A=f["A"]; mom=f["scores"]["mom14"]
    g=M.hyst(brd,0.5,0.4); X,lab,ewfwd,_,_=W.features_labels(f); Pwf=np.nan_to_num(W.walkforward_P(X,lab,ewfwd,idx,T,seed=0),nan=0.0)
    sleeve,Wls=book_net(conv,g,Pwf,mom,vol,roo,T,A,True,False)   # breadth-ALONE (no Pwf)
    full,Wlf=book_net(conv,g,Pwf,mom,vol,roo,T,A,True,True)       # full V5
    btc=cdf["BTCUSDT"].to_numpy() if "BTCUSDT" in cdf.columns else None
    out={"check":"BREADTH_SLEEVE_validate","ts":datetime.now().strftime("%Y%m%d_%H%M%S"),"windows":{}}
    print("="*98); print("AUDIT: BREADTH-GATED MOMENTUM SLEEVE (breadth-alone, NO Pwf) -- full robustness battery vs V5 & BTC-BH"); print("="*98)
    for wn,wm in [("OOS-2024",(idx>=CUT)&(idx<OOS25)),("OOS-2025",(idx>=OOS25)),("OOS-ALL",(idx>=CUT))]:
        ns=sleeve[wm]; nf=full[wm]
        wks=np.array([float(np.prod(1+ns[i:i+SB])-1)*100 for i in range(0,len(ns)-SB+1,SB)])
        comp_s=float(np.prod(1+ns)-1)*100; comp_f=float(np.prod(1+nf)-1)*100
        eqs=np.cumprod(1+ns); dd_s=float(np.min((eqs-np.maximum.accumulate(eqs))/np.maximum.accumulate(eqs))*100) if len(eqs)>1 else 0
        dep=float(np.mean(Wls.sum(axis=1)[wm]>1e-9))
        t_nw,p_nw=nw_p(wks,3)
        bh=None; bh_dd=None
        if btc is not None:
            b=btc[wm]; b=b[np.isfinite(b)]
            if len(b)>1:
                bh=float(b[-1]/b[0]-1)*100; beq=b/b[0]; bh_dd=float(np.min((beq-np.maximum.accumulate(beq))/np.maximum.accumulate(beq))*100)
        s={"comp_sleeve":round(comp_s,1),"comp_fullV5":round(comp_f,1),"btc_bh":round(bh,1) if bh else None,
           "maxdd_sleeve":round(dd_s,1),"btc_bh_maxdd":round(bh_dd,1) if bh_dd else None,"deploy":round(dep,2),
           "n_wk":len(wks),"wk_mean":round(float(np.mean(wks)),2),"pct_pos":round(float(np.mean(wks>0)),2),
           "nw_p":round(p_nw,4),"p05":bootp05(wks),"dsr_n24":dsr(wks,24),"dsr_n100":dsr(wks,100),
           "top1":topk_share(wks,1),"top2":topk_share(wks,2),"top3":topk_share(wks,3)}
        out["windows"][wn]=s
        print(f"\n  {wn}:  SLEEVE comp {comp_s:+.0f}% (maxDD {dd_s:+.0f}%, deploy {dep*100:.0f}%)  |  full-V5 {comp_f:+.0f}%  |  BTC-BH {('%+.0f%%'%bh) if bh else 'n/a'} (DD {('%+.0f%%'%bh_dd) if bh_dd else 'n/a'})")
        print(f"     robustness: Newey-West p={p_nw:.3f} {'(sig)' if p_nw<0.05 else '(NOT sig)'} | block-boot p05={s['p05']}% {'(>0 ROBUST)' if (s['p05'] or 0)>0 else '(<=0 fragile)'} | DSR n24={s['dsr_n24']} n100={s['dsr_n100']}")
        print(f"     lumpiness: top-1 wk {s['top1']}% | top-2 {s['top2']}% | top-3 {s['top3']}%  ({len(wks)} wks, {s['pct_pos']*100:.0f}% pos, mean {s['wk_mean']:+.2f}%)")
    a=out["windows"]["OOS-ALL"]
    verdict=("ROBUST de-risking deployable" if (a["nw_p"]<0.05 and (a["p05"] or -1)>0) else
             "de-risking sleeve: positive but NOT statistically robust (lumpy) -> paper-track too" )
    out["verdict"]=verdict
    print(f"\n  SLEEVE VERDICT (OOS-ALL): {verdict}")
    print(f"     -> de-risking value = maxDD {a['maxdd_sleeve']:+.0f}% vs BTC-BH {a['btc_bh_maxdd']}% ; return {a['comp_sleeve']:+.0f}% vs BTC-BH {a['btc_bh']}%")
    with open(HERE/"strat_sota_breadth_sleeve_validate.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\nSaved {HERE/'strat_sota_breadth_sleeve_validate.json'}")


if __name__=="__main__":
    main()
