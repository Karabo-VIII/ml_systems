"""GOAL redevelopment: per TI, find a config (WHEN-gate x SELECTION) that meets the SLICE-SUCCESS criterion --
delta-profitable vs the average 7d slice AND low losing-deployed-week rate. Tune on TRAIN, validate on VAL.
OOS NEVER loaded (WALL=VAL_END). The TI (MA cross, canonical held_rolling 14d/7d) defines the ELIGIBLE set; we
improve WHEN it deploys (gate) and WHICH eligibles it holds (selection) -- the two levers the audit says are missing
(gate lets in ~20% losing weeks; whole-basket holding makes delta negative). Cost prod 0.0028 RT, inv-vol cap33.

Configs/TI: WHEN {breadth-hyst, breadth AND BTC-ER>0.35} x SELECT {invvol-all-eligible, mom14 top-8, mom14 top-5}.
Pick the config MAXIMIZING TRAIN delta (subject to deploy_rate>=0.15), report TRAIN+VAL delta + loss_slice + depWin.
SUCCESS = chosen config delta>0 on BOTH TRAIN and VAL (beats avg slice) with loss_slice materially below baseline 20%.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE=Path(__file__).resolve().parent; ROOT=Path(__file__).resolve().parents[3]/"src"; sys.path.insert(0,str(ROOT))
from strat.ma_wealth_driver import _u100_symbols, _norm_sym
from strat.ma_slice_engine import held_rolling, _cached_band_held, VAL_START, VAL_END
from strat.ma_type_upgrade import MA_TYPES
from pipeline.chimera_loader import ChimeraLoader
from strat.canonical_exits import COST_PROD
import strat.fleet_lab as fl
BPD=fl.BARS_PER_DAY["1d"]; LB=14*BPD; SL=7*BPD; CAP=0.33
WALL=int(pd.Timestamp(VAL_END).value//10**6)
TR0=pd.Timestamp("2020-01-01"); TR1=pd.Timestamp(VAL_START); V0=pd.Timestamp(VAL_START); V1=pd.Timestamp(VAL_END)


def hyst(s,u,d):
    o=np.zeros(len(s),int);st=0
    for i in range(len(s)):
        if s[i]>u: st=1
        elif s[i]<d: st=0
        o[i]=st
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
    bc=cdf["BTCUSDT"].to_numpy() if "BTCUSDT" in cols else C[:,0]; mv=np.abs(np.diff(bc,prepend=bc[0]))
    ber=np.abs(bc-pd.Series(bc).shift(14).to_numpy())/np.maximum(pd.Series(mv).rolling(14,min_periods=7).sum().to_numpy(),1e-9)
    erg=(ber>0.35).astype(int); mom14=cdf.pct_change(14).to_numpy(); starts=[i for i in range(0,T-SL+1,SL)]
    ew=np.where(valid,1.0,0.0); ew=ew/np.maximum(ew.sum(axis=1,keepdims=True),1)
    def net_of(W):
        Wl=np.vstack([np.zeros((1,A)),W[:-1]]); Wll=np.vstack([np.zeros((1,A)),Wl[:-1]])
        return np.nansum(np.where(valid,Wl*roo,0.0),axis=1)-np.nansum(np.abs(Wl-Wll),axis=1)*(COST_PROD/2), Wl
    mkt_net,_=net_of(ew)
    WHENS={"brd":gbin,"brd_ER":gbin*erg}
    SELECTS=["all","top8","top5"]
    def build(Held,whenv,sel):
        gate=whenv; W=np.zeros((T,A))
        for i in starts:
            if gate[i]<=0: continue
            elig=np.where((Held[i]>0)&np.isfinite(C[i]))[0]
            if len(elig)==0: continue
            if sel=="all": pick=elig
            else:
                k=int(sel[3:]); s=mom14[i][elig]; order=np.argsort(-np.where(np.isfinite(s),s,-1e9)); pick=elig[order[:min(k,len(elig))]]
            v=vol[i][pick]; w=np.where(np.isfinite(v)&(v>0),1.0/v,0.0); w=w/w.sum() if w.sum()>0 else np.ones(len(pick))/max(len(pick),1)
            W[i:i+SL,pick]=np.minimum(w,CAP)
        return W
    def metrics(net,Wl,w0,w1):
        rows=[i for i in starts if idx[i]>=w0 and idx[i]<w1]
        sn=np.array([float(np.prod(1+net[i:i+SL])-1) for i in rows]); expo=np.array([float(np.mean(Wl[i:i+SL].sum(axis=1))) for i in rows])
        mn=np.array([float(np.prod(1+mkt_net[i:i+SL])-1) for i in rows]); dep=expo>0.05
        loss=float(np.mean(dep&(sn<-1e-4))) if len(sn) else 1.0; dwin=float(np.mean(sn[dep]>0)) if dep.any() else 0.0
        return {"delta":round(float(np.mean(sn)-np.mean(mn))*100,2),"loss_slice":round(loss,3),"dep_win":round(dwin,2),
                "deploy":round(float(np.mean(dep)),2),"mean_sl":round(float(np.mean(sn))*100,2),"comp":round(float(np.prod(1+sn)-1)*100,0),"n":len(sn)}
    out={"check":"ti_redevelop","wall":VAL_END,"tis":{}}
    print("="*122); print(f"TI REDEVELOP -- best WHENxSELECT per TI (pick max TRAIN delta, deploy>=0.15), validate VAL | OOS HELD OUT"); print("="*122)
    print(f"\n  {'TI':<6}{'chosen':<16}| {'TRAIN d%':>9}{'tr_loss':>8}{'tr_win':>7}{'tr_dep':>7} | {'VAL d%':>8}{'v_loss':>7}{'v_win':>7}{'v_dep':>7}{'v_comp':>8}")
    for ti in MA_TYPES:
        H=np.zeros((T,A))
        for j,s in enumerate(syms):
            try:
                band=_cached_band_held(ti,"1d",s,data[s]["o"],data[s]["h"],data[s]["l"],data[s]["c"],WALL); hd=held_rolling(data[s]["c"],band,LB,SL)
                H[:,j]=pd.Series(hd,index=data[s]["d"]).reindex(idx).fillna(0).to_numpy()
            except Exception: pass
        best=None
        for wn,wv in WHENS.items():
            for sel in SELECTS:
                W=build(H,wv,sel); net,Wl=net_of(W); mt=metrics(net,Wl,TR0,TR1)
                if mt["deploy"]<0.15: continue
                if best is None or mt["delta"]>best[2]["delta"]: best=(f"{wn}|{sel}",W,mt,net,Wl)
        if best is None: print(f"  {ti:<6}(no config with deploy>=0.15)"); continue
        name,W,mt,net,Wl=best; mv=metrics(net,Wl,V0,V1)
        out["tis"][ti]={"config":name,"TRAIN":mt,"VAL":mv}
        flag="OK" if mt["delta"]>0 and mv["delta"]>0 else ("VAL-neg" if mt["delta"]>0 else "no")
        print(f"  {ti:<6}{name:<16}| {mt['delta']:>+8.2f}%{mt['loss_slice']*100:>7.0f}%{mt['dep_win']*100:>6.0f}%{mt['deploy']*100:>6.0f}% | {mv['delta']:>+7.2f}%{mv['loss_slice']*100:>6.0f}%{mv['dep_win']*100:>6.0f}%{mv['deploy']*100:>6.0f}%{mv['comp']:>+7.0f}%  {flag}")
    with open(HERE/"strat_ti_redevelop.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    npos=sum(1 for ti in out["tis"] if out["tis"][ti]["TRAIN"]["delta"]>0 and out["tis"][ti]["VAL"]["delta"]>0)
    print(f"\n  {npos}/{len(out['tis'])} TIs delta-positive on BOTH TRAIN and VAL (= beat the avg 7d slice, out-of-tuning-sample on VAL).")
    print(f"  Saved {HERE/'strat_ti_redevelop.json'}")


if __name__=="__main__":
    main()
