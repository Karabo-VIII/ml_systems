"""SOLVE-FOR-SOTA next-value: QUALITY-GATED PULLBACK complement (long-only spot) -- attack the p05 lumpiness.

V5 (WF-timing + momentum) is real but LUMPY (block-boot p05<=0; rides a few trend weeks). The honest fix for
lumpiness = fill some of the CASH weeks with a POSITIVE, DE-CORRELATED long-only entry -> more weeks contribute ->
higher hit-rate + p05. My earlier MR test was a STRAWMAN (RSI<35 oversold-buy, no trend/breadth gate = falling
knives -85%). This is the QUALITY-GATED version, still 100% long-only spot:

  PULLBACK ENTRY (buy a DIP within a confirmed UPTREND, healthy breadth) -- all required:
    (i)   conv>0          name is in a confirmed MA-band UPTREND (the trend system already likes it)
    (ii)  mom14>0         name's 14d trend is positive (not a downtrend)
    (iii) g>0             breadth healthy (NOT the bear -> not catching falling knives)
    (iv)  z10 < ZIN       pulled back below its 10d mean (the dip)            [buy strength on a dip]
  EXIT: z10 > ZOUT (bounce) OR time-cap 7d.  Rank top-K by deepest dip (-z10) OR by momentum. inv-vol, cap.
  GATED BY BREADTH ONLY (not the WF-timing Pwf) -> it CAN deploy in weeks V5 sits out (Pwf<0.55 but breadth OK).

Measure (OOS-2024 + OOS-2025): pullback's own dist; corr(weekly) with V5; % of V5-CASH weeks pullback fills;
COMBINED book (V5 when V5 deploys, else pullback) -> does it raise %wk>0 and block-boot p05 WITHOUT killing compound?
If yes -> genuine de-lumping complement. If no -> V5-alone is final (selectivity confirmed w/ the quality-gated test).
1d, u100, de-leaked, next-open, prod cost. lev=1 long-only spot.
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


def stateful_pullback_held(z, elig, zin, zout, tcap):
    """held[t,j]=1 from the bar AFTER a dip-entry (z<zin & elig) until bounce (z>zout) or time-cap."""
    T,A=z.shape; H=np.zeros((T,A));
    for j in range(A):
        held=0; age=0
        for t in range(T):
            if held:
                age+=1
                if (np.isfinite(z[t,j]) and z[t,j]>zout) or age>=tcap or not elig[t,j]: held=0; age=0
            else:
                if elig[t,j] and np.isfinite(z[t,j]) and z[t,j]<zin: held=1; age=0
            H[t,j]=held
    return H


def main():
    f=M.build("FWD"); idx=f["idx"]; conv=f["conv"]; vol=f["vol"]; roo=f["roo"]; brd=f["brd"]; cdf=f["cdf"]; T=f["T"]; A=f["A"]; mom=f["scores"]["mom14"]
    g=M.hyst(brd,0.5,0.4); C=cdf.to_numpy()
    X,lab,ewfwd,_,_=W.features_labels(f); Pwf=np.nan_to_num(W.walkforward_P(X,lab,ewfwd,idx,T),nan=0.0)
    # z10 = (close - mean10)/std10 (causal)
    m10=cdf.rolling(10*BPD,min_periods=5).mean().to_numpy(); s10=cdf.rolling(10*BPD,min_periods=5).std().to_numpy()
    z10=(C-m10)/np.where(s10>1e-12,s10,np.nan)
    # ---- V5 book (WF-timing + momentum top-8) ----
    def topk_book(score, elig, k=8):
        Wt=np.zeros((T,A)); s=np.where(elig&np.isfinite(score),score,-np.inf)
        for t in range(T):
            e=np.where(np.isfinite(s[t]))[0]
            if len(e)==0: continue
            kk=min(k,len(e)); top=e[np.argsort(-s[t][e])[:kk]]; v=vol[t][top]; w=np.where(np.isfinite(v)&(v>0),1.0/v,0.0); Wt[t,top]=w/w.sum() if w.sum()>0 else 1.0/kk
        return Wt
    def net(Wt):
        Wl=np.vstack([np.zeros((1,A)),Wt[:-1]]); return np.nansum(Wl*roo,axis=1)-np.nansum(np.abs(Wt-Wl),axis=1)*(COST_PROD/2), Wl
    Wv=topk_book(mom,(conv>0)&(g[:,None]>0)&(Pwf[:,None]>0.55),8); nv,Wlv=net(Wv)
    # ---- pullback variants ----
    pelig=(conv>0)&(g[:,None]>0)&(mom>0)   # uptrend + breadth + positive trend
    variants={}
    for (zin,zout) in [(-0.5,0.3),(-1.0,0.0),(-0.75,0.25)]:
        H=stateful_pullback_held(z10,pelig,zin,zout,7*BPD)
        for rankby,rname in [(-z10,"deepdip"),(mom,"mom")]:
            sc=np.where(H>0,rankby,-np.inf)
            Wp=topk_book(sc,H>0,8); npb,Wlp=net(Wp)
            variants[f"z{zin}/{zout}|{rname}"]=(npb,Wlp)
    def weekly(n,mask): return np.array([float(np.prod(1+n[mask][i:i+SB])-1)*100 for i in range(0,int(mask.sum())-SB+1,SB)])
    def depweek(Wl,mask):
        gr=Wl.sum(axis=1)[mask]; return np.array([bool(np.mean(gr[i:i+SB]>1e-9)>0.5) for i in range(0,int(mask.sum())-SB+1,SB)])
    def bootp05(wk,nb=2000):
        if len(wk)<8: return None
        rng=np.random.default_rng(0); bl=4; comps=[]
        for _ in range(nb):
            pick=[]
            while len(pick)<len(wk):
                st=int(rng.integers(0,len(wk))); pick+=list(range(st,min(st+bl,len(wk))))
            pick=np.array(pick[:len(wk)]); comps.append(float(np.prod(1+wk[pick]/100)-1)*100)
        return round(float(np.percentile(comps,5)),1)
    out={"check":"STRAT_SOTA_pullback","ts":datetime.now().strftime("%Y%m%d_%H%M%S"),"windows":{}}
    for wn,wm in [("OOS-2024",(idx>=CUT)&(idx<OOS25)),("OOS-2025",(idx>=OOS25))]:
        out["windows"][wn]={}
        wkv=weekly(nv,wm); depv=depweek(Wlv,wm)
        print(f"\n{'='*104}\n{wn}  quality-gated PULLBACK complement vs V5 (timing+momentum), 1d, de-leaked, prod, long-only")
        print(f"{'='*104}")
        cv=float(np.prod(1+nv[wm])-1)*100
        print(f"  V5 base: comp {cv:+.0f}%  wk_mean {np.mean(wkv):+.2f}%  %wk>0 {np.mean(wkv>0)*100:.0f}%  p05 {bootp05(wkv)}  deploy {np.mean(depv)*100:.0f}%")
        print(f"  {'pullback variant':<22}{'comp%':>7}{'wk_mean':>9}{'%wk>0':>7}{'corr_V5':>9}{'fillV5cash':>11}{'COMBO_comp':>11}{'COMBO_%>0':>10}{'COMBO_p05':>10}")
        print("  "+"-"*97)
        best=None
        for nm,(npb,Wlp) in variants.items():
            wkp=weekly(npb,wm); depp=depweek(Wlp,wm); cp=float(np.prod(1+npb[wm])-1)*100
            n=min(len(wkv),len(wkp)); corr=float(np.corrcoef(wkv[:n],wkp[:n])[0,1]) if n>2 and np.std(wkp[:n])>0 else 0.0
            vcash=~depv[:n]; fill=float(np.mean(depp[:n][vcash])) if vcash.sum() else 0.0
            # combined: V5 when V5 deploys, else pullback
            combo=np.where(depv[:n],wkv[:n],np.where(depp[:n],wkp[:n],0.0))
            cc=float(np.prod(1+combo/100)-1)*100; cpos=float(np.mean(combo>0)); cp05=bootp05(combo)
            out["windows"][wn][nm]={"comp":round(cp,1),"wk_mean":round(float(np.mean(wkp)),2),"corr_V5":round(corr,2),"fill_v5cash":round(fill,2),"combo_comp":round(cc,1),"combo_pos":round(cpos,2),"combo_p05":cp05}
            print(f"  {nm:<22}{cp:>+6.0f}%{np.mean(wkp):>+8.2f}%{np.mean(wkp>0)*100:>6.0f}%{corr:>+9.2f}{fill*100:>10.0f}%{cc:>+10.0f}%{cpos*100:>9.0f}%{str(cp05):>10}")
            if best is None or cc>best[1]: best=(nm,cc)
        out["windows"][wn]["_v5_base"]={"comp":round(cv,1),"wk_pos":round(float(np.mean(wkv>0)),2),"p05":bootp05(wkv)}
    with open(HERE/"strat_sota_pullback.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\nSaved {HERE/'strat_sota_pullback.json'}")


if __name__=="__main__":
    main()
