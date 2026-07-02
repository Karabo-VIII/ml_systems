"""GOAL (2026-07-02): audit the TI (MA-family) bots against the SLICE-SUCCESS criterion, TRAIN+VAL ONLY (OOS never
loaded -- WALL=VAL_END). Success = every 7d slice is CASH-OR-PROFITABLE (never a losing DEPLOYED week; cash when it
can't win) OR delta-profitable vs the average 7d slice. TF=1d canonical (only net-positive TF). Structure 14d/7d.

Per MA TI (canonical held_rolling over the train/val band, inv-vol cap33, breadth-gated), on TRAIN (pre-2023-07) and
VAL (2023-07..2024-05-15): loss_slice_rate (deployed & net<0 -- the violators, want ~0), cash_or_profit_rate,
deployed_win_rate, mean 7d slice, avg market slice (EW-u100), DELTA. This is the AUDIT (where we stand); redevelopment
of the gate to MEET the criterion follows. Cost prod 0.0028 RT. OOS (2024-05-15+) is HELD OUT -- not touched here.
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
BPD=fl.BARS_PER_DAY["1d"]; LB=14*BPD; SL=7*BPD
WALL=int(pd.Timestamp(VAL_END).value//10**6)  # HARD WALL: never load OOS
TR0=pd.Timestamp("2020-01-01"); TR1=pd.Timestamp(VAL_START); V0=pd.Timestamp(VAL_START); V1=pd.Timestamp(VAL_END)
CAP=0.33


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


def hyst(s,u,d):
    o=np.zeros(len(s),int);st=0
    for i in range(len(s)):
        if s[i]>u: st=1
        elif s[i]<d: st=0
        o[i]=st
    return o


def main():
    data=load(); syms=list(data)
    cdf=pd.DataFrame({s:pd.Series(data[s]["c"],index=data[s]["d"]) for s in syms}).sort_index()
    odf=pd.DataFrame({s:pd.Series(data[s]["o"],index=data[s]["d"]) for s in syms}).reindex(cdf.index)
    idx=cdf.index; cols=list(cdf.columns); T=len(idx); A=len(cols); C=cdf.to_numpy(); O=odf.to_numpy()
    roo=np.full((T,A),0.0); roo[:-1]=O[1:]/np.where(O[:-1]>0,O[:-1],np.nan)-1.0; roo=np.nan_to_num(roo)
    vol=cdf.pct_change(fill_method=None).rolling(20,min_periods=10).std().to_numpy(); valid=np.isfinite(C)
    sma100=cdf.rolling(100,min_periods=50).mean().to_numpy(); vc=np.isfinite(sma100).sum(axis=1)
    brd=np.where(vc>0,np.nansum((C>sma100)*np.isfinite(sma100),axis=1)/np.maximum(vc,1),0.0); gbin=hyst(brd,0.5,0.4)
    starts=[i for i in range(0,T-SL+1,SL)]
    ew=np.where(valid,1.0,0.0); ew=ew/np.maximum(ew.sum(axis=1,keepdims=True),1)
    def net_of(W):
        Wl=np.vstack([np.zeros((1,A)),W[:-1]]); Wll=np.vstack([np.zeros((1,A)),Wl[:-1]])
        return np.nansum(np.where(valid,Wl*roo,0.0),axis=1)-np.nansum(np.abs(Wl-Wll),axis=1)*(COST_PROD/2), Wl
    mkt_net,_=net_of(ew)
    def ivbook(H):
        w=np.where((H>0)&np.isfinite(vol)&(vol>0),1.0/np.where(vol>0,vol,np.nan),0.0); w=np.nan_to_num(w)*H
        s=w.sum(axis=1,keepdims=True); return np.minimum(np.where(s>0,w/np.maximum(s,1e-12),0.0),CAP)
    def slice_metrics(net,Wl,win0,win1):
        rows=[i for i in starts if idx[i]>=win0 and idx[i]<win1]
        sn=np.array([float(np.prod(1+net[i:i+SL])-1) for i in rows])            # bot 7d slice net
        expo=np.array([float(np.mean(Wl[i:i+SL].sum(axis=1))) for i in rows])    # avg gross in slice
        mn=np.array([float(np.prod(1+mkt_net[i:i+SL])-1) for i in rows])         # market EW slice
        dep=expo>0.05
        loss_slice=float(np.mean(dep&(sn<-1e-4))) if len(sn) else None            # deployed & net<0
        cash_or_profit=1.0-loss_slice if loss_slice is not None else None
        dep_win=float(np.mean(sn[dep]>0)) if dep.any() else None
        return {"n":len(sn),"deploy_rate":round(float(np.mean(dep)),2),"loss_slice_rate":round(loss_slice,3) if loss_slice is not None else None,
                "cash_or_profit_rate":round(cash_or_profit,3) if cash_or_profit is not None else None,
                "deployed_win":round(dep_win,2) if dep_win is not None else None,
                "mean_slice_pct":round(float(np.mean(sn))*100,2),"avg_mkt_slice_pct":round(float(np.mean(mn))*100,2),
                "delta_pct":round(float(np.mean(sn)-np.mean(mn))*100,2),"compound_pct":round(float(np.prod(1+sn)-1)*100,0)}
    out={"check":"ti_slice_audit","wall":VAL_END,"note":"TRAIN+VAL only, OOS never loaded","ts":"NA","tis":{}}
    print("="*120); print(f"TI SLICE-SUCCESS AUDIT (breadth-gated canonical) -- TRAIN(pre-{VAL_START}) + VAL({VAL_START}..{VAL_END}) | OOS HELD OUT"); print("="*120)
    print(f"\n  success = loss_slice_rate ~0 (never a losing deployed week) OR delta_pct>0 (beat avg slice). {A} assets.")
    print(f"\n  {'TI':<7}{'win':<6}{'n':>4}{'deploy%':>8}{'LOSS_sl%':>9}{'cash/prof%':>11}{'depWin%':>8}{'mean_sl%':>9}{'avgMkt%':>8}{'delta%':>8}{'comp%':>7}")
    for ti in MA_TYPES:
        H=np.zeros((T,A))
        for j,s in enumerate(syms):
            try:
                band=_cached_band_held(ti,"1d",s,data[s]["o"],data[s]["h"],data[s]["l"],data[s]["c"],WALL); hd=held_rolling(data[s]["c"],band,LB,SL)
                H[:,j]=pd.Series(hd,index=data[s]["d"]).reindex(idx).fillna(0).to_numpy()
            except Exception: pass
        W=ivbook(H*gbin[:,None]); net,Wl=net_of(W)
        mt=slice_metrics(net,Wl,TR0,TR1); mv=slice_metrics(net,Wl,V0,V1)
        out["tis"][ti]={"TRAIN":mt,"VAL":mv}
        for wn,m in [("TRAIN",mt),("VAL",mv)]:
            print(f"  {ti:<7}{wn:<6}{m['n']:>4}{m['deploy_rate']*100:>7.0f}%{(m['loss_slice_rate'] or 0)*100:>8.0f}%{(m['cash_or_profit_rate'] or 0)*100:>10.0f}%{(m['deployed_win'] or 0)*100:>7.0f}%{m['mean_slice_pct']:>+8.2f}%{m['avg_mkt_slice_pct']:>+7.2f}%{m['delta_pct']:>+7.2f}%{m['compound_pct']:>+6.0f}%")
    with open(HERE/"strat_ti_slice_audit.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\n  READ: LOSS_sl% = deployed weeks that lost (want ~0). delta% = mean bot slice - mean market slice (want >0).")
    print(f"        A good bot: low LOSS_sl% (gate keeps it out of losing weeks) AND/OR positive delta. Redevelop the GATE next.")
    print(f"  Saved {HERE/'strat_ti_slice_audit.json'}")


if __name__=="__main__":
    main()
