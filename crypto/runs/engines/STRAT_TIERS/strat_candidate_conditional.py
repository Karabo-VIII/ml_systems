"""ORC autonomous: Candidate C -- CONDITIONAL gate (best-of-both?) + sub-period robustness vs A (BIN) and B (BIN_x_ER).

Hypothesis (pre-registered): BIN_x_ER cuts the bull (+13 vs +56) because it requires ER ALWAYS. If we require trend-
efficiency ONLY when breadth is MARGINAL (not when breadth is strong), we keep the bull (strong-breadth weeks deploy
regardless of ER) AND keep the bear protection (marginal/early-bear weeks need a clean trend). 3-state gate:
  breadth HIGH (>0.60) -> DEPLOY (full)             [capture bull]
  breadth MARGINAL (0.40-0.60) -> deploy iff ER>0.35 [avoid chop/early-bear]
  breadth LOW (<0.40) -> CASH                        [avoid bear]
H0: C does NOT dominate (no gate gets both high return AND low DD -- it's a frontier, pick a point).
Plus sub-period robustness: split OOS into 4 sub-windows; a deployable should be consistent, not 1-window-driven.
Single SMA engine, 14d/7d, invvol cap33, next-open, prod. Caveat: another gate on the same data (multiple comparisons).
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd
from scipy import stats as ss

HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE)); ROOT=Path(__file__).resolve().parents[3]/"src"; sys.path.insert(0,str(ROOT))
from strat.ma_slice_engine import held_rolling, _cached_band_held
from strat.ma_wealth_driver import _load_asset, _u100_symbols
from strat.canonical_exits import COST_PROD
import strat.fleet_lab as fl
BPD=fl.BARS_PER_DAY["1d"]; LB=14*BPD; SL=7*BPD; SW=100*BPD; CAP=0.33; ANN=np.sqrt(365)
WALL=int(pd.Timestamp("2026-07-01").value//10**6); CUT=pd.Timestamp("2024-05-15"); OOS25=pd.Timestamp("2025-01-01")


def hyst(s,u,d):
    n=len(s);o=np.zeros(n,int);st=0
    for i in range(n):
        if not np.isfinite(s[i]): o[i]=st; continue
        if st==0 and s[i]>u: st=1
        elif st==1 and s[i]<d: st=0
        o[i]=st
    return o


def bootp05(wk,nb=1500):
    if len(wk)<8: return None
    rng=np.random.default_rng(0); bl=4; comps=[]
    for _ in range(nb):
        pick=[]
        while len(pick)<len(wk):
            st=int(rng.integers(0,len(wk))); pick+=list(range(st,min(st+bl,len(wk))))
        pick=np.array(pick[:len(wk)]); comps.append(float(np.prod(1+wk[pick]/100)-1)*100)
    return round(float(np.percentile(comps,5)),1)


def main():
    closes={};opens={};H={}
    for s in _u100_symbols():
        r=_load_asset(s,"1d",WALL)
        if r is None: continue
        d,o,h,l,c=r; closes[s]=pd.Series(np.asarray(c,float),index=d); opens[s]=pd.Series(np.asarray(o,float),index=d)
        H[s]=pd.Series(held_rolling(c,_cached_band_held("SMA","1d",s,o,h,l,c,WALL),LB,SL),index=d)
    cdf=pd.DataFrame(closes).sort_index(); idx=cdf.index; cols=list(cdf.columns); T=len(idx); A=len(cols)
    odf=pd.DataFrame(opens).reindex(idx).reindex(columns=cols); C=cdf.to_numpy(); O=odf.to_numpy()
    Hm=pd.DataFrame(H).reindex(idx).fillna(0).reindex(columns=cols).fillna(0).to_numpy()
    vol=cdf.pct_change(fill_method=None).rolling(max(20,5*BPD),min_periods=10).std().to_numpy()
    roo=np.full((T,A),0.0); roo[:-1]=O[1:]/np.where(O[:-1]>0,O[:-1],np.nan)-1.0; roo=np.nan_to_num(roo)
    sall=cdf.rolling(SW,min_periods=SW//2).mean().to_numpy(); vc=np.isfinite(sall).sum(axis=1)
    brd=np.where(vc>0,np.nansum((C>sall)*np.isfinite(sall),axis=1)/np.maximum(vc,1),np.nan); brd=np.nan_to_num(brd)
    gbin=hyst(brd,0.5,0.4).astype(float)
    bc=cdf["BTCUSDT"].to_numpy(); mv=np.abs(np.diff(bc,prepend=bc[0])); er=np.abs(bc-pd.Series(bc).shift(14).to_numpy())/np.maximum(pd.Series(mv).rolling(14,min_periods=7).sum().to_numpy(),1e-9)
    erok=(er>0.35).astype(float)
    # Candidate C: 3-state conditional gate
    gcond=np.where(brd>0.60,1.0,np.where(brd<0.40,0.0,erok))
    def b1(x): return np.broadcast_to(x[:,None],(T,A)).astype(float)
    gates={"A_BIN":b1(gbin),"B_BIN_x_ER":b1(gbin*erok),"C_CONDITIONAL":b1(gcond)}
    def book(gt):
        mult=Hm*gt; iv=np.where((mult>0)&np.isfinite(vol)&(vol>0),1.0/np.where(vol>0,vol,np.nan),0.0)
        iv=np.nan_to_num(iv)*mult; sden=iv.sum(axis=1,keepdims=True); W=np.minimum(np.where(sden>0,iv/np.maximum(sden,1e-12),0.0),CAP)
        Wl=np.vstack([np.zeros((1,A)),W[:-1]]); return np.nansum(Wl*roo,axis=1)-np.nansum(np.abs(W-Wl),axis=1)*(COST_PROD/2), W
    def cm(net,m):
        n=net[m]; eq=np.cumprod(1+n); comp=float(eq[-1]-1)*100; dd=float(np.min((eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq))*100) if len(eq)>1 else 0
        cal=float(comp/abs(dd)) if abs(dd)>1e-9 else 0
        starts=list(range(0,len(n)-SL+1,SL)); wk=np.array([float(np.prod(1+n[i:i+SL])-1)*100 for i in starts])
        return round(comp,1),round(dd,1),round(cal,2),(bootp05(wk) if len(wk)>=8 else None)
    nets={nm:book(gt) for nm,gt in gates.items()}
    # sub-periods (4): 2024H2, 2025H1, 2025H2, 2026H1
    SUBS={"2024H2":((idx>=CUT)&(idx<pd.Timestamp("2025-01-01"))),"2025H1":((idx>=pd.Timestamp("2025-01-01"))&(idx<pd.Timestamp("2025-07-01"))),
          "2025H2":((idx>=pd.Timestamp("2025-07-01"))&(idx<pd.Timestamp("2026-01-01"))),"2026H1":((idx>=pd.Timestamp("2026-01-01")))}
    out={"check":"candidate_conditional","ts":datetime.now().strftime("%Y%m%d_%H%M%S"),"main":{},"subperiods":{}}
    print("="*96); print("CANDIDATE C (conditional gate) vs A (BIN) / B (BIN_x_ER) -- main windows + sub-period robustness"); print("="*96)
    for w,m in {"OOS-2024":(idx>=CUT)&(idx<OOS25),"OOS-2025":(idx>=OOS25),"OOS-ALL":(idx>=CUT)}.items():
        print(f"\n  {w}: {'':<16}{'comp%':>8}{'maxDD':>8}{'Calmar':>8}{'p05':>8}")
        for nm in gates:
            c,dd,cal,p05=cm(nets[nm][0],m); out["main"].setdefault(nm,{})[w]={"comp":c,"maxdd":dd,"calmar":cal,"p05":p05}
            print(f"    {nm:<18}{c:>+7.0f}%{dd:>+8.0f}{cal:>8.2f}{(p05 if p05 is not None else 0):>+7.0f}%")
    print(f"\n  SUB-PERIOD comp% (consistency check -- positive across sub-windows = robust, not 1-window-driven):")
    print(f"    {'gate':<18}"+"".join(f"{s:>9}" for s in SUBS))
    for nm in gates:
        row=""; cells={}
        for s,m in SUBS.items():
            c,_,_,_=cm(nets[nm][0],m); cells[s]=c; row+=f"{c:>+8.0f}%"
        out["subperiods"][nm]=cells; print(f"    {nm:<18}{row}")
    # verdict
    a=out["main"]; print(f"\n  READ: C dominates only if it gets A's return AND B's DD. Else it's a frontier point.")
    for nm in gates:
        al=a[nm]["OOS-ALL"]; print(f"    {nm}: OOS-ALL {al['comp']:+.0f}% / maxDD {al['maxdd']:+.0f}% / Calmar {al['calmar']} ; bear(OOS-2025) {a[nm]['OOS-2025']['comp']:+.0f}%")
    with open(HERE/"strat_candidate_conditional.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\nSaved {HERE/'strat_candidate_conditional.json'}")


if __name__=="__main__":
    main()
