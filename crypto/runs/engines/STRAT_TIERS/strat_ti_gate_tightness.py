"""GOAL clause-1: since the delta-edge (clause-2) is refuted, maximize CASH-OR-PROFITABLE rate ('profitable on any 7d
window') via gate tightness -- a MECHANICAL, robust property (choosing selectivity), NOT a fitted edge. HMA + momentum
top-5, 1d, 14d/7d. TRAIN+VAL only (WALL=VAL_END, OOS never loaded). Sweep gates loose->tight; report per window:
deploy_rate, cash_or_profit_rate (1 - losing-deployed-week rate), losing_week_rate, mean slice, delta. Find the gate
that pushes cash-or-profitable high at a still-tradeable deploy rate (>=~15%). This is de-risked beta characterized honestly.
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
WALL=int(pd.Timestamp(VAL_END).value//10**6); TI="HMA"
TR0=pd.Timestamp("2020-01-01"); TR1=pd.Timestamp(VAL_START); V0=pd.Timestamp(VAL_START); V1=pd.Timestamp(VAL_END)


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
    brd=np.where(vc>0,np.nansum((C>sma100)*np.isfinite(sma100),axis=1)/np.maximum(vc,1),0.0)
    bc=cdf["BTCUSDT"].to_numpy() if "BTCUSDT" in cols else C[:,0]; mv=np.abs(np.diff(bc,prepend=bc[0]))
    ber=np.abs(bc-pd.Series(bc).shift(14).to_numpy())/np.maximum(pd.Series(mv).rolling(14,min_periods=7).sum().to_numpy(),1e-9)
    bsma=pd.Series(bc).rolling(50,min_periods=50).mean().to_numpy(); brd_rise=brd>pd.Series(brd).shift(7).to_numpy()
    mom14=cdf.pct_change(14).to_numpy(); starts=[i for i in range(0,T-SL+1,SL)]
    ew=np.where(valid,1.0,0.0); ew=ew/np.maximum(ew.sum(axis=1,keepdims=True),1)
    def net_of(W):
        Wl=np.vstack([np.zeros((1,A)),W[:-1]]); Wll=np.vstack([np.zeros((1,A)),Wl[:-1]])
        return np.nansum(np.where(valid,Wl*roo,0.0),axis=1)-np.nansum(np.abs(Wl-Wll),axis=1)*(COST_PROD/2), Wl
    mkt,_=net_of(ew)
    # TI held
    H=np.zeros((T,A))
    for j,s in enumerate(syms):
        try:
            band=_cached_band_held(TI,"1d",s,data[s]["o"],data[s]["h"],data[s]["l"],data[s]["c"],WALL); hd=held_rolling(data[s]["c"],band,LB,SL)
            H[:,j]=pd.Series(hd,index=data[s]["d"]).reindex(idx).fillna(0).to_numpy()
        except Exception: pass
    gb=hyst(brd,0.5,0.4)
    GATES={"brd(0.5/0.4)":gb,"brd&ER35":gb*(ber>0.35),"brd&BTC>SMA50":gb*(bc>bsma).astype(int),
           "brd&ER35&rise":gb*(ber>0.35)*brd_rise.astype(int),"brd&ER45&BTCup":gb*(ber>0.45)*(bc>bsma).astype(int)}
    def build(gate):
        W=np.zeros((T,A))
        for i in starts:
            if gate[i]<=0: continue
            elig=np.where((H[i]>0)&np.isfinite(C[i]))[0]
            if len(elig)==0: continue
            s=mom14[i][elig]; pick=elig[np.argsort(-np.where(np.isfinite(s),s,-1e9))[:min(5,len(elig))]]
            v=vol[i][pick]; w=np.where(np.isfinite(v)&(v>0),1.0/v,0.0); w=w/w.sum() if w.sum()>0 else np.ones(len(pick))/max(len(pick),1)
            W[i:i+SL,pick]=np.minimum(w,CAP)
        return W
    def m(net,Wl,w0,w1):
        rows=[i for i in starts if idx[i]>=w0 and idx[i]<w1]
        sn=np.array([float(np.prod(1+net[i:i+SL])-1) for i in rows]); expo=np.array([float(np.mean(Wl[i:i+SL].sum(axis=1))) for i in rows])
        mn=np.array([float(np.prod(1+mkt[i:i+SL])-1) for i in rows]); dep=expo>0.05
        return {"deploy":round(float(np.mean(dep)),2),"cash_or_profit":round(1-float(np.mean(dep&(sn<-1e-4))),3),
                "losing_wk":round(float(np.mean(dep&(sn<-1e-4))),3),"mean_sl":round(float(np.mean(sn))*100,2),
                "delta":round(float(np.mean(sn)-np.mean(mn))*100,2),"n":len(sn)}
    out={"check":"ti_gate_tightness","TI":TI,"gates":{}}
    print("="*110); print(f"CLAUSE-1 (cash-or-profitable) vs GATE TIGHTNESS -- {TI}+momentum-top5, TRAIN+VAL (OOS held out)"); print("="*110)
    print(f"\n  {'gate':<18}{'win':<7}{'deploy%':>8}{'cash/prof%':>11}{'losing_wk%':>11}{'mean_sl%':>9}{'delta%':>8}")
    for gn,gv in GATES.items():
        W=build(gv); net,Wl=net_of(W); mt=m(net,Wl,TR0,TR1); mv=m(net,Wl,V0,V1)
        out["gates"][gn]={"TRAIN":mt,"VAL":mv}
        for wn,mm in [("TRAIN",mt),("VAL",mv)]:
            print(f"  {gn:<18}{wn:<7}{mm['deploy']*100:>7.0f}%{mm['cash_or_profit']*100:>10.0f}%{mm['losing_wk']*100:>10.0f}%{mm['mean_sl']:>+8.2f}%{mm['delta']:>+7.2f}%")
    with open(HERE/"strat_ti_gate_tightness.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\n  READ: tighter gate -> lower deploy%, higher cash/prof% (fewer losing weeks). The honest clause-1 bot maximizes")
    print(f"        cash/prof% at a still-tradeable deploy% (>=~15%). This is de-risked-beta SELECTIVITY, robust by construction.")
    print(f"  Saved {HERE/'strat_ti_gate_tightness.json'}")


if __name__=="__main__":
    main()
