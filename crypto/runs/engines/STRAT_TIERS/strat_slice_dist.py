"""Honest distribution behind the '3-4% per slice' MEAN: for the lead deployable (HMA + mom-top5 + K3 multi-gate),
report MEAN vs MEDIAN vs %positive for all-slices AND deployed-slices, vs the market slice (how much is beta), on
TRAIN+VAL (OOS held out). A mean pulled by a few fat-tail weeks is NOT a bankable weekly rate.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[3]/"src"; sys.path.insert(0,str(ROOT))
from strat.ma_wealth_driver import _u100_symbols, _norm_sym
from strat.ma_slice_engine import held_rolling, _cached_band_held, VAL_START, VAL_END
from pipeline.chimera_loader import ChimeraLoader
from strat.canonical_exits import COST_PROD
import strat.fleet_lab as fl
BPD=fl.BARS_PER_DAY["1d"]; LB=14*BPD; SL=7*BPD; CAP=0.33
WALL=int(pd.Timestamp(VAL_END).value//10**6); TI="HMA"
TR=("2020-01-01",VAL_START); VL=(VAL_START,VAL_END)


def hyst(s,u,d):
    o=np.zeros(len(s),int);st=0
    for i in range(len(s)): st=1 if s[i]>u else (0 if s[i]<d else st); o[i]=st
    return o


def main():
    data={}
    for s in _u100_symbols():
        try: df=ChimeraLoader().load(_norm_sym(s),cadence="1d",features=["open","high","low","close"])
        except Exception: continue
        ts=df["timestamp"].to_numpy(); sel=np.where(ts<WALL)[0]
        if len(sel)<200: continue
        d=pd.to_datetime(ts[sel],unit="ms").floor("D"); keep=~pd.Index(d).duplicated(keep="last"); sel=sel[keep]; d=d[keep]
        data[s]={"d":d,"o":df["open"].to_numpy().astype(float)[sel],"h":df["high"].to_numpy().astype(float)[sel],
                 "l":df["low"].to_numpy().astype(float)[sel],"c":df["close"].to_numpy().astype(float)[sel]}
    syms=list(data)
    cdf=pd.DataFrame({s:pd.Series(data[s]["c"],index=data[s]["d"]) for s in syms}).sort_index()
    odf=pd.DataFrame({s:pd.Series(data[s]["o"],index=data[s]["d"]) for s in syms}).reindex(cdf.index)
    idx=cdf.index; cols=list(cdf.columns); T=len(idx); A=len(cols); C=cdf.to_numpy(); O=odf.to_numpy()
    R=np.full((T,A),np.nan); R[1:]=C[1:]/np.where(C[:-1]>0,C[:-1],np.nan)-1.0
    roo=np.full((T,A),0.0); roo[:-1]=O[1:]/np.where(O[:-1]>0,O[:-1],np.nan)-1.0; roo=np.nan_to_num(roo)
    vol=cdf.pct_change(fill_method=None).rolling(20,min_periods=10).std().to_numpy(); valid=np.isfinite(C)
    sma100=cdf.rolling(100,min_periods=50).mean().to_numpy(); vc=np.isfinite(sma100).sum(axis=1)
    brd=np.where(vc>0,np.nansum((C>sma100)*np.isfinite(sma100),axis=1)/np.maximum(vc,1),0.0)
    bc=cdf["BTCUSDT"].to_numpy() if "BTCUSDT" in cols else C[:,0]; mv=np.abs(np.diff(bc,prepend=bc[0]))
    ber=np.abs(bc-pd.Series(bc).shift(14).to_numpy())/np.maximum(pd.Series(mv).rolling(14,min_periods=7).sum().to_numpy(),1e-9)
    bsma=pd.Series(bc).rolling(50,min_periods=50).mean().to_numpy(); disp=np.nanstd(R,axis=1); dmed=np.nanmedian(disp[np.isfinite(disp)])
    score=hyst(brd,0.5,0.4)+(brd>pd.Series(brd).shift(7).to_numpy()).astype(int)+(ber>0.35).astype(int)+(bc>bsma).astype(int)+(disp>dmed).astype(int)
    mom14=cdf.pct_change(14).to_numpy(); starts=[i for i in range(0,T-SL+1,SL)]
    ew=np.where(valid,1.0,0.0); ew=ew/np.maximum(ew.sum(axis=1,keepdims=True),1)
    H=np.zeros((T,A))
    for j,s in enumerate(syms):
        try:
            band=_cached_band_held(TI,"1d",s,data[s]["o"],data[s]["h"],data[s]["l"],data[s]["c"],WALL); hd=held_rolling(data[s]["c"],band,LB,SL)
            H[:,j]=pd.Series(hd,index=data[s]["d"]).reindex(idx).fillna(0).to_numpy()
        except Exception: pass
    W=np.zeros((T,A))
    for i in starts:
        if score[i]<3: continue
        elig=np.where((H[i]>0)&np.isfinite(C[i]))[0]
        if len(elig)==0: continue
        sm=mom14[i][elig]; pick=elig[np.argsort(-np.where(np.isfinite(sm),sm,-1e9))[:min(5,len(elig))]]
        v=vol[i][pick]; w=np.where(np.isfinite(v)&(v>0),1.0/v,0.0); w=w/w.sum() if w.sum()>0 else np.ones(len(pick))/max(len(pick),1)
        W[i:i+SL,pick]=np.minimum(w,CAP)
    def net_of(Wm):
        Wl=np.vstack([np.zeros((1,A)),Wm[:-1]]); Wll=np.vstack([np.zeros((1,A)),Wl[:-1]])
        return np.nansum(np.where(valid,Wl*roo,0.0),axis=1)-np.nansum(np.abs(Wl-Wll),axis=1)*(COST_PROD/2), Wl
    net,Wl=net_of(W); mkt,_=net_of(ew)
    def rep(w0,w1,label):
        rows=[i for i in starts if idx[i]>=pd.Timestamp(w0) and idx[i]<pd.Timestamp(w1)]
        sn=np.array([float(np.prod(1+net[i:i+SL])-1)*100 for i in rows]); mn=np.array([float(np.prod(1+mkt[i:i+SL])-1)*100 for i in rows])
        expo=np.array([float(np.mean(Wl[i:i+SL].sum(axis=1))) for i in rows]); dep=expo>0.05
        print(f"\n  {label}  ({len(sn)} slices, {int(dep.sum())} deployed / {int((~dep).sum())} cash)")
        print(f"    ALL slices     : mean {np.mean(sn):+.2f}%  median {np.median(sn):+.2f}%  %pos {np.mean(sn>0)*100:.0f}%")
        if dep.any():
            ds=sn[dep]; print(f"    DEPLOYED slices: mean {np.mean(ds):+.2f}%  median {np.median(ds):+.2f}%  %pos {np.mean(ds>0)*100:.0f}%  top3 {np.sort(ds)[-3:][::-1].round(1)}")
        print(f"    MARKET (EW-u100): mean {np.mean(mn):+.2f}%  median {np.median(mn):+.2f}%   (=how much of the mean is just beta)")
    print("="*96); print(f"SLICE-RETURN DISTRIBUTION -- lead bot HMA+mom-top5+K3-gate (the '3-4% per slice' number, honestly)"); print("="*96)
    rep(*TR,"TRAIN (2020-01..2023-07)"); rep(*VL,"VAL (2023-07..2024-05)")
    print(f"\n  READ: the 3-4% is the MEAN (fat-tail-pulled). The MEDIAN deployed week is the bankable typical; the MARKET")
    print(f"        mean shows how much is beta. Deployed %pos ~55-70% = many deployed weeks are 0/negative.")


if __name__=="__main__":
    main()
