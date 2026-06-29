"""PROD-GRADE weekly market REPLAY of the MA engines -- OOS-2024 + OOS-2025+ (per week).

User: "MA engine results for OOS 2024 and OOS 2025 window (per week); PROD-grade market replay."
Engines: cross-type CONV book (deployable) + per-type {SMA,EMA,WMA,VIDYA,KAMA}, config = breadth-gated (de-leaked),
inv-vol weights @cap33, 14d lookback / 7d window, ROLLING, 1d. PROD-grade: NEXT-OPEN fills (open-to-open, lag-1),
prod cost 0.0028 RT, point-in-time u100, turnover costed, long-only spot lev=1. Reports per-week (7d) return + deploy
state + summary (mean/wk, median, %wk>0, in[2,7], compound, maxDD, deploy-frac) for each window.
WINDOWS: OOS-2024 = 2024-05-15..2024-12-31 ; OOS-2025 = 2025-01-01..2026-05-28 (all unseen 2025+).
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd

HERE=Path(__file__).resolve().parent; ROOT=Path(__file__).resolve().parents[3]/"src"; sys.path.insert(0,str(ROOT))
import strat.fleet_lab as fl
from strat.ma_wealth_driver import _load_asset, _u100_symbols
from strat.ma_slice_engine import held_rolling, _cached_band_held
from strat.canonical_exits import COST_PROD
BPD=fl.BARS_PER_DAY["1d"]; TYPES=["SMA","EMA","WMA","VIDYA","KAMA"]; LB=14*BPD; SLh=7*BPD; SW=100*BPD; CAP=0.33
WIN={"OOS-2024":(int(pd.Timestamp("2025-01-01").value//10**6),"2024-05-15","2024-12-31"),
     "OOS-2025":(int(pd.Timestamp("2026-07-01").value//10**6),"2025-01-01","2026-07-01")}


def hyst(s,u,d):
    n=len(s);o=np.zeros(n,int);st=0
    for i in range(n):
        if not np.isfinite(s[i]): o[i]=st; continue
        if st==0 and s[i]>u: st=1
        elif st==1 and s[i]<d: st=0
        o[i]=st
    return o


def build(window):
    wall,ws,we=WIN[window]; closes={};opens={}
    for s in _u100_symbols():
        r=_load_asset(s,"1d",wall)
        if r is None: continue
        d,o,h,l,c=r; closes[s]=pd.Series(np.asarray(c,float),index=d); opens[s]=pd.Series(np.asarray(o,float),index=d)
    cdf=pd.DataFrame(closes).sort_index(); idx=cdf.index; cols=list(cdf.columns); T=len(idx); A=len(cols)
    odf=pd.DataFrame(opens).reindex(idx).reindex(columns=cols)
    typ={}; conv=np.zeros((T,A))
    for ma in TYPES:
        hc={}
        for s in cols:
            r=_load_asset(s,"1d",wall)
            if r is None: continue
            d,o,h,l,c=r; hc[s]=pd.Series(held_rolling(c,_cached_band_held(ma,"1d",s,o,h,l,c,wall),LB,SLh),index=d)
        H=pd.DataFrame(hc).reindex(idx).fillna(0).reindex(columns=cols).fillna(0).to_numpy(); typ[ma]=H; conv+=H
    C=cdf.to_numpy();O=odf.to_numpy()
    # fill_method=None (NOT the default 'pad') -> a padded/missing bar yields NaN, not a spurious 0-return that would
    # deflate the inv-vol estimate. Consistent with ma_wealth_driver.py / ma_movecatch_engine (audit wlirg1ui9).
    vol=cdf.pct_change(fill_method=None).rolling(max(20,5*BPD),min_periods=10).std().to_numpy()
    roo=np.full((T,A),0.0); roo[:-1]=O[1:]/np.where(O[:-1]>0,O[:-1],np.nan)-1.0; roo=np.nan_to_num(roo)
    sall=cdf.rolling(SW,min_periods=SW//2).mean().to_numpy(); vc=np.isfinite(sall).sum(axis=1)
    brd=np.where(vc>0,np.nansum((C>sall)*np.isfinite(sall),axis=1)/np.maximum(vc,1),np.nan); g=hyst(brd,0.5,0.4)
    mask=(idx>=pd.Timestamp(ws))&(idx<pd.Timestamp(we))
    return {"typ":typ,"conv":conv,"vol":vol,"roo":roo,"g":g,"mask":mask,"idx":idx,"T":T,"A":A}


def W_invvol(held_mult, vol, g, cap):
    mult=held_mult*(g[:,None]>0)
    iv=np.where((mult>0)&np.isfinite(vol)&(vol>0),1.0/np.where(vol>0,vol,np.nan),0.0); iv=np.nan_to_num(iv)*mult
    s=iv.sum(axis=1,keepdims=True); return np.minimum(np.where(s>0,iv/np.maximum(s,1e-12),0.0),cap)


def net_of(W,roo):
    Wl=np.vstack([np.zeros((1,W.shape[1])),W[:-1]]); gr=np.nansum(Wl*roo,axis=1)
    turn=np.nansum(np.abs(W-Wl),axis=1); return gr-turn*(COST_PROD/2), Wl


def weekly(net,Wl,mask):
    n=net[mask]; g=Wl.sum(axis=1)[mask]; wk=[];dep=[];i=0
    while i+SLh<=len(n):
        wk.append(float(np.prod(1+n[i:i+SLh])-1)*100); dep.append(bool(np.mean(g[i:i+SLh]>1e-9)>0.5)); i+=SLh
    return np.array(wk),np.array(dep)


def summ(wk,dep,net,mask):
    n=net[mask]; eq=np.cumprod(1+n); dd=float(np.min((eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq)))*100 if len(eq)>1 else 0
    return {"n_wk":len(wk),"mean_wk":round(float(np.mean(wk)),2),"median_wk":round(float(np.median(wk)),2),
            "pct_pos":round(float(np.mean(wk>0)),3),"in_2to7":round(float(np.mean((wk>=2)&(wk<=7))),3),
            "deploy_frac":round(float(np.mean(dep)),3),"compound":round(float(eq[-1]-1)*100,1),"maxdd":round(dd,1)}


def main():
    out={"check":"MA_weekly_replay_PROD","config":"breadth-gated de-leaked, inv-vol cap33, 14d/7d, 1d, next-open fills, prod 0.0028RT","ts":datetime.now().strftime("%Y%m%d_%H%M%S"),"windows":{}}
    for window in ["OOS-2024","OOS-2025"]:
        f=build(window); out["windows"][window]={"summary":{},"conv_per_week":[]}
        # summary table all engines
        print(f"\n{'#'*92}\n# {window}  ({WIN[window][1]}..{WIN[window][2]})  PROD-grade MA replay (breadth-gated, invvol cap33, next-open)\n{'#'*92}")
        print(f"  {'engine':<8}{'n_wk':>5}{'mean/wk':>9}{'median':>8}{'%wk>0':>7}{'in[2,7]':>9}{'deploy':>8}{'compound':>10}{'maxDD':>7}")
        print("  "+"-"*78)
        eng={"CONV":f["conv"]/len(TYPES)}; eng.update({ma:f["typ"][ma].astype(float) for ma in TYPES})
        nets={}
        for nm,hm in eng.items():
            W=W_invvol(hm,f["vol"],f["g"],CAP); net,Wl=net_of(W,f["roo"]); wk,dep=weekly(net,Wl,f["mask"]); s=summ(wk,dep,net,f["mask"])
            out["windows"][window]["summary"][nm]=s; nets[nm]=(net,Wl,wk,dep)
            print(f"  {nm:<8}{s['n_wk']:>5}{s['mean_wk']:>+8.2f}%{s['median_wk']:>+7.2f}%{s['pct_pos']*100:>6.0f}%{s['in_2to7']*100:>8.0f}%{s['deploy_frac']*100:>7.0f}%{s['compound']:>+9.0f}%{s['maxdd']:>+7.0f}")
        # per-week table for CONV (the deployable) + per-week dates
        net,Wl,wk,dep=nets["CONV"]; mi=np.where(f["mask"])[0]; dts=f["idx"][f["mask"]]
        wkdates=[str(pd.Timestamp(dts[i]).date()) for i in range(0,len(dts)-SLh+1,SLh)]
        print(f"\n  -- CONV book PER-WEEK (week_start : return% : state) --")
        line=""
        for i,(d,r,dp) in enumerate(zip(wkdates,wk,dep)):
            line+=f"  {d} {r:>+6.1f}% {'DEP' if dp else 'csh'}"
            out["windows"][window]["conv_per_week"].append({"week":d,"ret_pct":round(float(r),2),"deployed":bool(dp)})
            if (i+1)%3==0: print(line); line=""
        if line: print(line)
    with open(HERE/"ma_weekly_replay.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\nSaved {HERE/'ma_weekly_replay.json'}")


if __name__=="__main__":
    main()
