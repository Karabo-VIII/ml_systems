"""QUANT: base momentum top-K deployed + try to CLOSE the oracle gap mechanically. Capture opportunity if it's real.

Oracle-gap (prior): ORACLE_top8 +18.3%/slice ceiling; MOM_top8 +0.97 (best causal); MA_strat +0.31. The +17.3 selection
gap is un-capturable IF price can't rank names (AUC 0.548). USER: run momentum top-K, then close gaps if opportunity exists.

PRE-REGISTERED. H0: no selection variant robustly beats base MOM8 OOS (selection capped). ALT: some variant captures more
AND survives sub-period persistence (4 half-years) + DSR-deflation (multiple-comparisons corrected). Asymmetric loss:
false-ship an overfit selection edge = worst. Bar = SUB-PERIOD CONSISTENCY + DSR, not raw OOS compound.
Variants (per-7d-slice rebalance, top-K, inv-vol, next-open, prod cost, u100 1d):
  MOM8         : momentum(14d) top-8, no gate (base)
  MOM8_BRD     : momentum top-8, breadth-gated (V5 selection -- does gating fix the reversal crash?)
  MOM8_ER      : momentum top-8, breadth AND trend-efficiency
  MOM3 / MOM16 : concentration dial
  MOM30_8      : 30d momentum (slower)
  COMPOSITE8   : z(mom14)+z(accel)+z(breakout) top-8 (multi-signal -- beat single momentum?)
  MOM8_QUAL    : momentum top-8 among names above own 50d-SMA (quality filter -- avoid pump/dumps)
Grade each: compound, maxDD, %slices+, NW p, block-boot p05, 4 sub-periods, DSR(n_variants). vs ORACLE ceiling.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd
from scipy import stats as ss

HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE)); ROOT=Path(__file__).resolve().parents[3]/"src"; sys.path.insert(0,str(ROOT))
from strat.ma_wealth_driver import _load_asset, _u100_symbols
from strat.canonical_exits import COST_PROD
import strat.fleet_lab as fl
BPD=fl.BARS_PER_DAY["1d"]; SL=7*BPD; SW=100*BPD; CAP=0.33
WALL=int(pd.Timestamp("2026-07-01").value//10**6); CUT=pd.Timestamp("2024-05-15"); OOS25=pd.Timestamp("2025-01-01")


def hyst(s,u,d):
    n=len(s);o=np.zeros(n,int);st=0
    for i in range(n):
        if not np.isfinite(s[i]): o[i]=st; continue
        if st==0 and s[i]>u: st=1
        elif st==1 and s[i]<d: st=0
        o[i]=st
    return o


def nw_p(wk,L=3):
    n=len(wk)
    if n<4 or np.std(wk)<1e-12: return 1.0
    mu=np.mean(wk); x=wk-mu; var=np.dot(x,x)/n
    for l in range(1,L+1): var+=2*(1-l/(L+1))*(np.dot(x[l:],x[:-l])/n)
    return float(1-ss.t.cdf(mu/np.sqrt(max(var,1e-12)/n),n-1))


def p05b(wk,nb=1500):
    if len(wk)<8: return None
    rng=np.random.default_rng(0); cs=[]
    for _ in range(nb):
        pk=[]
        while len(pk)<len(wk):
            s0=int(rng.integers(0,len(wk))); pk+=list(range(s0,min(s0+4,len(wk))))
        pk=np.array(pk[:len(wk)]); cs.append(float(np.prod(1+wk[pk]/100)-1)*100)
    return round(float(np.percentile(cs,5)),1)


def dsr(wk,nt):
    if len(wk)<8 or np.std(wk)<1e-9: return None
    sr=np.mean(wk)/np.std(wk); n=len(wk); g3=ss.skew(wk); g4=ss.kurtosis(wk,fisher=False); emc=0.5772; z=ss.norm.ppf
    sr0=np.sqrt(1.0/n)*((1-emc)*z(1-1.0/nt)+emc*z(1-1.0/(nt*np.e))); den=np.sqrt(max(1e-9,1-g3*sr+(g4-1)/4.0*sr**2))
    return round(float(ss.norm.cdf(((sr-sr0)*np.sqrt(n-1))/den)),3)


def main():
    closes={};opens={}
    for s in _u100_symbols():
        r=_load_asset(s,"1d",WALL)
        if r is None: continue
        d,o,h,l,c=r; closes[s]=pd.Series(np.asarray(c,float),index=d); opens[s]=pd.Series(np.asarray(o,float),index=d)
    cdf=pd.DataFrame(closes).sort_index(); idx=cdf.index; cols=list(cdf.columns); T=len(idx); A=len(cols)
    odf=pd.DataFrame(opens).reindex(idx).reindex(columns=cols); C=cdf.to_numpy(); O=odf.to_numpy()
    vol=cdf.pct_change(fill_method=None).rolling(max(20,5*BPD),min_periods=10).std().to_numpy()
    roo=np.full((T,A),0.0); roo[:-1]=O[1:]/np.where(O[:-1]>0,O[:-1],np.nan)-1.0; roo=np.nan_to_num(roo)
    sall=cdf.rolling(SW,min_periods=SW//2).mean().to_numpy(); vc=np.isfinite(sall).sum(axis=1)
    brd=np.where(vc>0,np.nansum((C>sall)*np.isfinite(sall),axis=1)/np.maximum(vc,1),np.nan); gbin=hyst(brd,0.5,0.4)
    bc=cdf["BTCUSDT"].to_numpy(); mv=np.abs(np.diff(bc,prepend=bc[0])); er=(np.abs(bc-pd.Series(bc).shift(14).to_numpy())/np.maximum(pd.Series(mv).rolling(14,min_periods=7).sum().to_numpy(),1e-9)>0.35).astype(int)
    mom14=cdf.pct_change(14*BPD).to_numpy(); mom30=cdf.pct_change(30*BPD).to_numpy()
    accel=mom14-np.vstack([np.full((14*BPD,A),np.nan),mom14[:-14*BPD]]) if T>14*BPD else np.full_like(mom14,np.nan)
    rollmax=cdf.rolling(30*BPD,min_periods=10).max().to_numpy(); brk=C/np.where(rollmax>0,rollmax,np.nan)
    sma50=cdf.rolling(50,min_periods=20).mean().to_numpy()
    def zc(M):
        mu=np.nanmean(M,axis=1,keepdims=True); sd=np.nanstd(M,axis=1,keepdims=True); return np.where(sd>1e-9,(M-mu)/np.where(sd>1e-9,sd,1),0.0)
    comp_score=zc(mom14)+zc(np.nan_to_num(accel))+zc(brk)
    starts=[i for i in range(0,T-SL+1,SL)]
    def build(score, K, gatevec=None, qual=None):
        W=np.zeros((T,A))
        for i in starts:
            if gatevec is not None and gatevec[i]<=0: continue
            s=score[i].copy()
            if qual is not None: s=np.where(qual[i], s, -np.inf)
            e=np.where(np.isfinite(s)&(s>-1e17)&np.isfinite(C[i]))[0]
            if len(e)==0: continue
            k=min(K,len(e)); top=e[np.argsort(-s[e])[:k]]
            v=vol[i][top]; w=np.where(np.isfinite(v)&(v>0),1.0/v,0.0); w=w/w.sum() if w.sum()>0 else np.ones(k)/k
            W[i:i+SL,top]=np.minimum(w,CAP)
        return W
    def net_of(W):
        Wl=np.vstack([np.zeros((1,A)),W[:-1]]); return np.nansum(Wl*roo,axis=1)-np.nansum(np.abs(W-Wl),axis=1)*(COST_PROD/2)
    qual50=(C>sma50)&np.isfinite(sma50)
    VAR={"MOM8":build(mom14,8),"MOM8_BRD":build(mom14,8,gatevec=gbin),"MOM8_ER":build(mom14,8,gatevec=gbin*er),
         "MOM3":build(mom14,3),"MOM16":build(mom14,16),"MOM30_8":build(mom30,8),
         "COMPOSITE8":build(comp_score,8),"MOM8_QUAL":build(mom14,8,qual=qual50)}
    NV=len(VAR)
    def grade(W):
        net=net_of(W); m=(idx>=CUT); n=net[m]; eq=np.cumprod(1+n)
        dd=float(np.min((eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq))*100) if len(eq)>1 else 0
        st=[i for i in range(0,len(n)-SL+1,SL)]; wk=np.array([float(np.prod(1+n[i:i+SL])-1)*100 for i in st])
        subs={}
        for nm,(a,b) in {"24H2":(CUT,pd.Timestamp("2025-01-01")),"25H1":(pd.Timestamp("2025-01-01"),pd.Timestamp("2025-07-01")),
                         "25H2":(pd.Timestamp("2025-07-01"),pd.Timestamp("2026-01-01")),"26H1":(pd.Timestamp("2026-01-01"),pd.Timestamp("2026-08-01"))}.items():
            mm=(idx>=a)&(idx<b); nn=net[mm]; subs[nm]=round(float(np.prod(1+nn)-1)*100,1) if len(nn) else None
        return {"comp":round(float(np.prod(1+n)-1)*100,1),"maxdd":round(dd,1),"pct_pos":round(float(np.mean(wk>0)),2),
                "nw_p":round(nw_p(wk),3),"p05":p05b(wk),"dsr":dsr(wk,NV),"subs":subs}
    out={"check":"momentum_capture","n_variants":NV,"ts":datetime.now().strftime("%Y%m%d_%H%M%S"),"grade":{}}
    print("="*112); print(f"MOMENTUM SELECTION -- base + gap-closers, graded OOS 2024-05..2026-05 ({NV} variants, DSR-corrected)"); print("="*112)
    print(f"\n  {'variant':<12}{'comp%':>7}{'maxDD':>7}{'%sl+':>6}{'NW_p':>6}{'p05':>7}{'DSR':>6}  | sub-periods 24H2/25H1/25H2/26H1  (pos/4)")
    for nm,W in VAR.items():
        g=grade(W); out["grade"][nm]=g; sv=g["subs"]; npos=sum(1 for v in sv.values() if v and v>0)
        print(f"  {nm:<12}{g['comp']:>+6.0f}%{g['maxdd']:>+7.0f}{g['pct_pos']*100:>5.0f}%{g['nw_p']:>6.2f}{(g['p05'] or 0):>+6.0f}%{(g['dsr'] or 0):>6.2f}  | "
              f"{sv['24H2']:>+5.0f}/{sv['25H1']:>+5.0f}/{sv['25H2']:>+5.0f}/{sv['26H1']:>+5.0f}  ({npos}/4)")
    print(f"\n  ROBUSTNESS BAR: positive in >=3/4 sub-periods AND DSR>0.5 (deflated for {NV} variants) AND p05>-20%.")
    print(f"  H0 test: does ANY variant beat base MOM8 on sub-period consistency + DSR? (raw compound alone = overfit trap)")
    with open(HERE/"strat_momentum_capture.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\nSaved {HERE/'strat_momentum_capture.json'}")


if __name__=="__main__":
    main()
