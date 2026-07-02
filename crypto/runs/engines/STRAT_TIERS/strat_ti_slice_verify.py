"""GOAL verification: is the redeveloped TI bot's DELTA-success (beat avg 7d slice) DISTRIBUTED or regime-concentrated?
Bot = TI-eligible (canonical held_rolling 14d/7d) breadth-gated, hold momentum top-5, inv-vol cap33. TRAIN+VAL only
(WALL=VAL_END, OOS never loaded). Checks: (1) per-YEAR slice-delta (2020/2021/2022/2023H1/VAL) -- distributed or one
year? (2) top-K {3,5,8,all} sensitivity. (3) drop-top-3-slices jackknife on delta. (4) fraction of positive-delta
slices (how often the bot beats the market that week -- the honest per-slice win-vs-market). Lead TIs + SMA baseline.
"""
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
TIS=["HMA","TEMA","SMA"]
YEARS={"2020":("2020-01-01","2021-01-01"),"2021":("2021-01-01","2022-01-01"),"2022":("2022-01-01","2023-01-01"),
       "2023H1":("2023-01-01",VAL_START),"VAL":(VAL_START,VAL_END)}


def hyst(s,u,d):
    o=np.zeros(len(s),int);st=0
    for i in range(len(s)):
        st=1 if s[i]>u else (0 if s[i]<d else st); o[i]=st
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


def main():
    data=load(); syms=list(data)
    cdf=pd.DataFrame({s:pd.Series(data[s]["c"],index=data[s]["d"]) for s in syms}).sort_index()
    odf=pd.DataFrame({s:pd.Series(data[s]["o"],index=data[s]["d"]) for s in syms}).reindex(cdf.index)
    idx=cdf.index; cols=list(cdf.columns); T=len(idx); A=len(cols); C=cdf.to_numpy(); O=odf.to_numpy()
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
    def build(H,sel):
        W=np.zeros((T,A))
        for i in starts:
            if gbin[i]<=0: continue
            elig=np.where((H[i]>0)&np.isfinite(C[i]))[0]
            if len(elig)==0: continue
            if sel=="all": pick=elig
            else:
                k=int(sel); s=mom14[i][elig]; pick=elig[np.argsort(-np.where(np.isfinite(s),s,-1e9))[:min(k,len(elig))]]
            v=vol[i][pick]; w=np.where(np.isfinite(v)&(v>0),1.0/v,0.0); w=w/w.sum() if w.sum()>0 else np.ones(len(pick))/max(len(pick),1)
            W[i:i+SL,pick]=np.minimum(w,CAP)
        return W
    def sl(net,w0,w1):
        rows=[i for i in starts if idx[i]>=pd.Timestamp(w0) and idx[i]<pd.Timestamp(w1)]
        sn=np.array([float(np.prod(1+net[i:i+SL])-1) for i in rows]); mn=np.array([float(np.prod(1+mkt[i:i+SL])-1) for i in rows])
        return sn,mn
    out={"check":"ti_slice_verify","wall":VAL_END,"tis":{}}
    print("="*118); print("TI SLICE-SUCCESS VERIFY -- is delta DISTRIBUTED or regime-concentrated? (TRAIN+VAL, OOS held out)"); print("="*118)
    for ti in TIS:
        H=np.zeros((T,A))
        for j,s in enumerate(syms):
            try:
                band=_cached_band_held(ti,"1d",s,data[s]["o"],data[s]["h"],data[s]["l"],data[s]["c"],WALL); hd=held_rolling(data[s]["c"],band,LB,SL)
                H[:,j]=pd.Series(hd,index=data[s]["d"]).reindex(idx).fillna(0).to_numpy()
            except Exception: pass
        net=net_of(build(H,"5"))
        # per-year delta
        yr={}
        for y,(a,b) in YEARS.items():
            sn,mn=sl(net,a,b)
            yr[y]={"delta":round(float(np.mean(sn)-np.mean(mn))*100,2) if len(sn) else None,"n":len(sn),
                   "beat_rate":round(float(np.mean(sn>mn)),2) if len(sn) else None}
        # full TRAIN+VAL delta + drop-top-3-slice jackknife on (sn-mn)
        sn,mn=sl(net,"2020-01-01",VAL_END); d=sn-mn; order=np.argsort(-d)
        full_delta=round(float(np.mean(d))*100,2); jk=round(float(np.mean(np.delete(d,order[:3])))*100,2)
        beat=round(float(np.mean(sn>mn)),2)
        # top-K sensitivity (full TRAIN+VAL delta)
        tk={}
        for sel in ["3","5","8","all"]:
            s2,m2=sl(net_of(build(H,sel)),"2020-01-01",VAL_END); tk[sel]=round(float(np.mean(s2-m2))*100,2)
        out["tis"][ti]={"per_year":yr,"full_delta":full_delta,"jackknife_drop3":jk,"beat_rate":beat,"topk_delta":tk}
        print(f"\n  {ti}:  full TRAIN+VAL slice-delta {full_delta:+.2f}% | drop-top3 {jk:+.2f}% | beats-market-slice {beat*100:.0f}% of weeks")
        print(f"     per-year delta: "+"  ".join(f"{y}{yr[y]['delta']:+.2f}%(n{yr[y]['n']})" for y in YEARS))
        print(f"     top-K delta:    "+"  ".join(f"top{sel}{tk[sel]:+.2f}%" for sel in ['3','5','8','all']))
        yrs_pos=sum(1 for y in yr if yr[y]['delta'] is not None and yr[y]['delta']>0)
        print(f"     -> delta-positive in {yrs_pos}/{len(yr)} years; drop-top3 {'STAYS +' if jk>0 else 'goes NEG'} = {'DISTRIBUTED' if jk>0 and yrs_pos>=3 else 'CONCENTRATED'}")
    with open(HERE/"strat_ti_slice_verify.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\n  Saved {HERE/'strat_ti_slice_verify.json'}")


if __name__=="__main__":
    main()
