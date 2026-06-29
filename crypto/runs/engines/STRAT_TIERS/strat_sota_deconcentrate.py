"""AUDIT: de-concentration as the lumpiness dial -- is there a ROBUST breadth-gated de-risking deployable?

Both full V5 and the breadth-momentum-top8 sleeve are lumpy (top-2 weeks ~77-121% of compound) BECAUSE of concentration
(top-8 inv-vol). Test K (and uncapped breadth-EW) to see if de-concentration spreads the return -> lower lumpiness +
p05>0, at the cost of lower compound (the de-risking trade). Breadth-gated (conv>0 & breadth-hyst), momentum top-K
inv-vol, NO Pwf. Report per K: compound, maxDD, top-2-wk share, Newey-West p, block-boot p05, vs BTC-BH. OOS-ALL + bear.
The honest question: does ANY de-concentrated breadth sleeve give a statistically-robust positive return, or is only the
DRAWDOWN-reduction robust (return-alpha lumpy at every K)? 1d, u100, seed=0, prod, long-only spot.
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


def book_net(conv,g,mom,vol,roo,T,A,K,cap=1.0):
    # pure inv-vol normalized to gross=1.0; K is the concentration dial (no under-investment confound).
    gate=(conv>0)&(g[:,None]>0); Wt=np.zeros((T,A)); s=np.where(gate&np.isfinite(mom),mom,-np.inf)
    for t in range(T):
        e=np.where(np.isfinite(s[t]))[0]
        if len(e)==0: continue
        k=min(K,len(e)); top=e[np.argsort(-s[t][e])[:k]]; v=vol[t][top]; w=np.where(np.isfinite(v)&(v>0),1.0/v,0.0)
        w=w/w.sum() if w.sum()>0 else np.ones(k)/k   # ALWAYS normalize to gross 1.0 (deployed weeks fully invested)
        Wt[t,top]=w
    Wl=np.vstack([np.zeros((1,A)),Wt[:-1]]); return np.nansum(Wl*roo,axis=1)-np.nansum(np.abs(Wt-Wl),axis=1)*(COST_PROD/2), Wl


def nw_p(wk,L=3):
    n=len(wk); mu=np.mean(wk); x=wk-mu; var=np.dot(x,x)/n
    for l in range(1,L+1): var+=2*(1-l/(L+1))*(np.dot(x[l:],x[:-l])/n)
    se=np.sqrt(max(var,1e-12)/n); return float(1-ss.t.cdf(mu/se,n-1))


def bootp05(wk,nb=2000):
    if len(wk)<8: return None
    rng=np.random.default_rng(0); bl=4; comps=[]
    for _ in range(nb):
        pick=[]
        while len(pick)<len(wk):
            st=int(rng.integers(0,len(wk))); pick+=list(range(st,min(st+bl,len(wk))))
        pick=np.array(pick[:len(wk)]); comps.append(float(np.prod(1+wk[pick]/100)-1)*100)
    return round(float(np.percentile(comps,5)),1)


def top2(wk):
    order=np.argsort(-wk); tot=np.exp(np.sum(np.log1p(wk/100)))-1
    if abs(tot)<1e-9: return None
    keep=np.ones(len(wk),bool); keep[order[:2]]=False
    return round((1-(np.exp(np.sum(np.log1p(wk[keep]/100)))-1)/tot)*100,1)


def main():
    f=M.build("FWD"); idx=f["idx"]; conv=f["conv"]; vol=f["vol"]; roo=f["roo"]; brd=f["brd"]; cdf=f["cdf"]; T=f["T"]; A=f["A"]; mom=f["scores"]["mom14"]
    g=M.hyst(brd,0.5,0.4); btc=cdf["BTCUSDT"].to_numpy() if "BTCUSDT" in cdf.columns else None
    configs=[("top8",8,1.0),("top16",16,1.0),("top30",30,1.0),("allEligible",999,1.0)]
    out={"check":"BREADTH_SLEEVE_deconcentrate","ts":datetime.now().strftime("%Y%m%d_%H%M%S"),"windows":{}}
    print("="*100); print("AUDIT: de-concentration dial on the breadth-gated sleeve -- does spreading reduce lumpiness to robust?"); print("="*100)
    for wn,wm in [("OOS-ALL",(idx>=CUT)),("OOS-2025 bear",(idx>=OOS25))]:
        out["windows"][wn]={}
        bh=None;bhdd=None
        if btc is not None:
            b=btc[wm]; b=b[np.isfinite(b)]
            if len(b)>1: bh=float(b[-1]/b[0]-1)*100; beq=b/b[0]; bhdd=float(np.min((beq-np.maximum.accumulate(beq))/np.maximum.accumulate(beq))*100)
        print(f"\n  {wn}  (BTC-BH {('%+.0f%%'%bh) if bh is not None else 'n/a'}, DD {('%+.0f%%'%bhdd) if bhdd is not None else 'n/a'})")
        print(f"    {'config':<18}{'comp%':>8}{'maxDD':>7}{'deploy':>8}{'top2wk%':>9}{'NW_p':>7}{'p05':>8}{'%wk>0':>7}")
        print("    "+"-"*72)
        for nm,K,cap in configs:
            net,Wl=book_net(conv,g,mom,vol,roo,T,A,K,cap); n=net[wm]
            wk=np.array([float(np.prod(1+n[i:i+SB])-1)*100 for i in range(0,len(n)-SB+1,SB)])
            comp=float(np.prod(1+n)-1)*100; eq=np.cumprod(1+n); dd=float(np.min((eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq))*100) if len(eq)>1 else 0
            dep=float(np.mean(Wl.sum(axis=1)[wm]>1e-9)); p=nw_p(wk); p05=bootp05(wk); t2=top2(wk); pos=float(np.mean(wk>0))
            out["windows"][wn][nm]={"comp":round(comp,1),"maxdd":round(dd,1),"deploy":round(dep,2),"top2_wk":t2,"nw_p":round(p,4),"p05":p05,"pct_pos":round(pos,2)}
            print(f"    {nm:<18}{comp:>+7.0f}%{dd:>+7.0f}{dep*100:>7.0f}%{(t2 if t2 else 0):>8.0f}%{p:>7.3f}{(p05 if p05 is not None else 0):>+7.0f}%{pos*100:>6.0f}%")
    print("\n  READ: if top2-wk% stays high (>60) and p05<0 / NW_p>0.05 at EVERY K -> return-alpha is lumpy at all concentrations;")
    print("        only the DRAWDOWN-reduction vs BTC-BH is the robust deployable value (de-risking, not return-alpha).")
    with open(HERE/"strat_sota_deconcentrate.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\nSaved {HERE/'strat_sota_deconcentrate.json'}")


if __name__=="__main__":
    main()
