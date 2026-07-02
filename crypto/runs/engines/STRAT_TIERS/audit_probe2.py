"""Deeper probes: (a) 2022 delta stability (is +0.52 real or noise), (b) does TI eligible set
even differ from ungated when breadth is up (overlap of picks), (c) drop-2021 AND drop-VAL joint,
(d) TI vs ungated on TRAIN-ONLY (2020..2023H1) -- if TI==ungated on train, TI is decorative."""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE=Path(__file__).resolve().parent; ROOT=Path(__file__).resolve().parents[3]/"src"; sys.path.insert(0,str(ROOT))
from strat.ma_wealth_driver import _u100_symbols, _norm_sym
from strat.ma_slice_engine import held_rolling, _cached_band_held, VAL_START, VAL_END
from pipeline.chimera_loader import ChimeraLoader
from strat.canonical_exits import COST_PROD
import strat.fleet_lab as fl
BPD=fl.BARS_PER_DAY["1d"]; LB=14*BPD; SL=7*BPD; CAP=0.33
WALL=int(pd.Timestamp(VAL_END).value//10**6)
def hyst(s,u,d):
    o=np.zeros(len(s),int);st=0
    for i in range(len(s)): st=1 if s[i]>u else (0 if s[i]<d else st); o[i]=st
    return o
def load():
    data={}
    for s in _u100_symbols():
        try: df=ChimeraLoader().load(_norm_sym(s),cadence="1d",features=["open","high","low","close"])
        except Exception: continue
        ts=df["timestamp"].to_numpy(); sel=np.where(ts<WALL)[0]
        if len(sel)<200: continue
        d=pd.to_datetime(ts[sel],unit="ms").floor("D"); keep=~pd.Index(d).duplicated(keep="last"); sel=sel[keep]; d=d[keep]
        data[s]={"d":d,"o":df["open"].to_numpy().astype(float)[sel],"h":df["high"].to_numpy().astype(float)[sel],
                 "l":df["low"].to_numpy().astype(float)[sel],"c":df["close"].to_numpy().astype(float)[sel]}
    return data
data=load(); syms=list(data)
cdf=pd.DataFrame({s:pd.Series(data[s]["c"],index=data[s]["d"]) for s in syms}).sort_index()
odf=pd.DataFrame({s:pd.Series(data[s]["o"],index=data[s]["d"]) for s in syms}).reindex(cdf.index)
idx=cdf.index; T=len(idx); A=len(cdf.columns); C=cdf.to_numpy(); O=odf.to_numpy()
roo=np.full((T,A),0.0); roo[:-1]=O[1:]/np.where(O[:-1]>0,O[:-1],np.nan)-1.0; roo=np.nan_to_num(roo)
vol=cdf.pct_change(fill_method=None).rolling(20,min_periods=10).std().to_numpy(); valid=np.isfinite(C)
sma100=cdf.rolling(100,min_periods=50).mean().to_numpy(); vc=np.isfinite(sma100).sum(axis=1)
brd=np.where(vc>0,np.nansum((C>sma100)*np.isfinite(sma100),axis=1)/np.maximum(vc,1),0.0); gbin=hyst(brd,0.5,0.4)
mom14=cdf.pct_change(14).to_numpy(); starts=[i for i in range(0,T-SL+1,SL)]
ew=np.where(valid,1.0,0.0); ew=ew/np.maximum(ew.sum(axis=1,keepdims=True),1)
def net_of(W):
    Wl=np.vstack([np.zeros((1,A)),W[:-1]]); Wll=np.vstack([np.zeros((1,A)),Wl[:-1]])
    return np.nansum(np.where(valid,Wl*roo,0.0),axis=1)-np.nansum(np.abs(Wl-Wll),axis=1)*(COST_PROD/2)
mkt=net_of(ew)
H=np.zeros((T,A))
for j,s in enumerate(syms):
    try:
        band=_cached_band_held("HMA","1d",s,data[s]["o"],data[s]["h"],data[s]["l"],data[s]["c"],WALL); hd=held_rolling(data[s]["c"],band,LB,SL)
        H[:,j]=pd.Series(hd,index=data[s]["d"]).reindex(idx).fillna(0).to_numpy()
    except Exception: pass
def picks(i,ungated):
    if gbin[i]<=0: return None
    elig=np.where(np.isfinite(C[i]))[0] if ungated else np.where((H[i]>0)&np.isfinite(C[i]))[0]
    if len(elig)==0: return None
    s=mom14[i][elig]; return set(elig[np.argsort(-np.where(np.isfinite(s),s,-1e9))[:min(5,len(elig))]].tolist())
# pick overlap TI vs ungated on deployed weeks
ov=[]; n_elig=[]; identical=0; deployed=0
for i in starts:
    pt=picks(i,False); pu=picks(i,True)
    if pt is None and pu is None: continue
    if pt is None or pu is None: continue
    deployed+=1
    inter=len(pt&pu); ov.append(inter/5.0)
    if pt==pu: identical+=1
    elig=np.where((H[i]>0)&np.isfinite(C[i]))[0]; n_elig.append(len(elig))
print(f"HMA deployed weeks (both gate on): {deployed}")
print(f"  mean TI/ungated top5 overlap = {np.mean(ov)*100:.0f}%  | identical picks in {identical}/{deployed} weeks ({identical/deployed*100:.0f}%)")
print(f"  median #TI-eligible names on deployed weeks = {int(np.median(n_elig))} (of {A})  -> if >>5, TI barely constrains the top5")
def build(ungated):
    W=np.zeros((T,A))
    for i in starts:
        if gbin[i]<=0: continue
        elig=np.where(np.isfinite(C[i]))[0] if ungated else np.where((H[i]>0)&np.isfinite(C[i]))[0]
        if len(elig)==0: continue
        s=mom14[i][elig]; pick=elig[np.argsort(-np.where(np.isfinite(s),s,-1e9))[:min(5,len(elig))]]
        v=vol[i][pick]; w=np.where(np.isfinite(v)&(v>0),1.0/v,0.0); w=w/w.sum() if w.sum()>0 else np.ones(len(pick))/max(len(pick),1)
        W[i:i+SL,pick]=np.minimum(w,CAP)
    return W
def dm(net,a,b):
    rows=[i for i in starts if idx[i]>=pd.Timestamp(a) and idx[i]<pd.Timestamp(b)]
    sn=np.array([float(np.prod(1+net[i:i+SL])-1) for i in rows]); mn=np.array([float(np.prod(1+mkt[i:i+SL])-1) for i in rows])
    return sn,mn
nti=net_of(build(False)); nun=net_of(build(True))
for label,a,b in [("TRAIN-only 2020..2023H1","2020-01-01",VAL_START),("VAL",VAL_START,VAL_END),("FULL","2020-01-01",VAL_END)]:
    s1,m1=dm(nti,a,b); s2,m2=dm(nun,a,b)
    print(f"  {label:<26}: TI delta {(np.mean(s1)-np.mean(m1))*100:+.2f}%   UNGATED delta {(np.mean(s2)-np.mean(m2))*100:+.2f}%   (TI-UNG={( (np.mean(s1)-np.mean(m1))-(np.mean(s2)-np.mean(m2)))*100:+.2f}pp)")
# bootstrap the FULL delta to see if 0.92 is distinguishable from 0
s1,m1=dm(nti,"2020-01-01",VAL_END); d=s1-m1
rng=np.random.default_rng(0); bs=np.array([np.mean(rng.choice(d,len(d),replace=True)) for _ in range(5000)])*100
print(f"  FULL delta bootstrap: mean {np.mean(bs):+.2f}%  p05 {np.percentile(bs,5):+.2f}%  p95 {np.percentile(bs,95):+.2f}%  P(delta<=0)={np.mean(bs<=0):.2f}")
