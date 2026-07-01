"""QUANT: ALL engine candidates -- full weekly breakdown over the OOS window, side by side (honest, u100 not BTC).

Every candidate built across the arc, one weekly table (7d, next-open fills, prod cost 0.0028 RT, u100 cross-sectional,
cap33). Regime-gate variants on the single SMA engine + the old cross-type ENSEMBLE (CONV) + PARTIAL + benchmarks:
  CONV   : cross-type ensemble (SMA/EMA/WMA/VIDYA/KAMA convergence), breadth-gated  [old canonical config]
  SMA_BIN: SMA, binary breadth gate                                                  [A wealth-max]
  SMA_ER : SMA, breadth AND market trend-efficiency (ER>0.35)                        [B capital-preservation]
  SMA_CNT: SMA, CONTINUOUS breadth exposure (gross prop. to breadth)                 [refuted]
  SMA_SLF: SMA, per-asset SELF regime (own 100d-MA)                                  [bull-greedy/bear-exposed]
  SMA_CND: SMA, 3-state conditional gate (full>0.6 / ER-if-marginal / cash<0.4)      [refuted, bull-overfit]
  PARTIAL: always 15-100% invested scaled by breadth (trend else EW-mkt)             [consistency]
  BH_alt : buy-hold EW-u100 (the traded universe benchmark)
  BH_btc : buy-hold BTC (context benchmark)
Full OOS 2024-05-15..data-end. Report per-WEEK % for all + summary (compound/maxDD/%pos). NOT BTC-only (u100 basket).
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd

HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE)); ROOT=Path(__file__).resolve().parents[3]/"src"; sys.path.insert(0,str(ROOT))
from strat.ma_slice_engine import held_rolling, _cached_band_held
from strat.ma_wealth_driver import _load_asset, _u100_symbols
from strat.canonical_exits import COST_PROD
import strat.fleet_lab as fl
BPD=fl.BARS_PER_DAY["1d"]; LB=14*BPD; SL=7*BPD; SW=100*BPD; CAP=0.33; START=10000.0
TYPES=["SMA","EMA","WMA","VIDYA","KAMA"]; WALL=int(pd.Timestamp("2026-07-01").value//10**6); CUT=pd.Timestamp("2024-05-15")


def hyst(s,u,d):
    n=len(s);o=np.zeros(n,int);st=0
    for i in range(n):
        if not np.isfinite(s[i]): o[i]=st; continue
        if st==0 and s[i]>u: st=1
        elif st==1 and s[i]<d: st=0
        o[i]=st
    return o


def main():
    closes={};opens={}; typ_held={t:{} for t in TYPES}
    for s in _u100_symbols():
        r=_load_asset(s,"1d",WALL)
        if r is None: continue
        d,o,h,l,c=r; closes[s]=pd.Series(np.asarray(c,float),index=d); opens[s]=pd.Series(np.asarray(o,float),index=d)
        for t in TYPES: typ_held[t][s]=pd.Series(held_rolling(c,_cached_band_held(t,"1d",s,o,h,l,c,WALL),LB,SL),index=d)
    cdf=pd.DataFrame(closes).sort_index(); idx=cdf.index; cols=list(cdf.columns); T=len(idx); A=len(cols)
    odf=pd.DataFrame(opens).reindex(idx).reindex(columns=cols); C=cdf.to_numpy(); O=odf.to_numpy()
    conv=np.zeros((T,A))
    for t in TYPES:
        Ht=pd.DataFrame(typ_held[t]).reindex(idx).fillna(0).reindex(columns=cols).fillna(0).to_numpy(); typ_held[t]=Ht; conv+=Ht
    Hsma=typ_held["SMA"]
    vol=cdf.pct_change(fill_method=None).rolling(max(20,5*BPD),min_periods=10).std().to_numpy()
    roo=np.full((T,A),0.0); roo[:-1]=O[1:]/np.where(O[:-1]>0,O[:-1],np.nan)-1.0; roo=np.nan_to_num(roo)
    sall=cdf.rolling(SW,min_periods=SW//2).mean().to_numpy(); vc=np.isfinite(sall).sum(axis=1)
    brd=np.where(vc>0,np.nansum((C>sall)*np.isfinite(sall),axis=1)/np.maximum(vc,1),np.nan); brd=np.nan_to_num(brd)
    gbin=hyst(brd,0.5,0.4).astype(float); gcont=np.clip((brd-0.40)/0.20,0,1)
    bc=cdf["BTCUSDT"].to_numpy(); mv=np.abs(np.diff(bc,prepend=bc[0])); er=np.abs(bc-pd.Series(bc).shift(14).to_numpy())/np.maximum(pd.Series(mv).rolling(14,min_periods=7).sum().to_numpy(),1e-9)
    erok=(er>0.35).astype(float); gcond=np.where(brd>0.60,1.0,np.where(brd<0.40,0.0,erok))
    own=cdf.rolling(SW,min_periods=SW//2).mean().to_numpy(); gself=((C>own)&np.isfinite(own)).astype(float)
    def b1(x): return np.broadcast_to(x[:,None],(T,A)).astype(float)
    def iv(held,gate2d):
        mult=held*gate2d; w=np.where((mult>0)&np.isfinite(vol)&(vol>0),1.0/np.where(vol>0,vol,np.nan),0.0)
        w=np.nan_to_num(w)*mult; s=w.sum(axis=1,keepdims=True); return np.minimum(np.where(s>0,w/np.maximum(s,1e-12),0.0),CAP)
    def net_of(W):
        Wl=np.vstack([np.zeros((1,A)),W[:-1]]); return np.nansum(Wl*roo,axis=1)-np.nansum(np.abs(W-Wl),axis=1)*(COST_PROD/2)
    # PARTIAL
    e=np.clip((brd-0.20)/0.60,0.15,1.0); ew=np.where(np.isfinite(C),1.0,0.0); ew=ew/np.maximum(ew.sum(axis=1,keepdims=True),1)
    Wtr=iv(Hsma,np.ones((T,A))); trend_any=Wtr.sum(axis=1,keepdims=True)>1e-9; Wpart=e[:,None]*np.where(trend_any,Wtr,ew)
    CAND={"CONV":iv(conv/len(TYPES),b1(gbin)),"SMA_BIN":iv(Hsma,b1(gbin)),"SMA_ER":iv(Hsma,b1(gbin*erok)),
          "SMA_CNT":iv(Hsma,b1(gcont)),"SMA_SLF":iv(Hsma,gself),"SMA_CND":iv(Hsma,b1(gcond)),"PARTIAL":Wpart}
    nets={k:net_of(W) for k,W in CAND.items()}
    # benchmarks
    btcr=np.zeros(T); bo=odf["BTCUSDT"].to_numpy(); btcr[:-1]=bo[1:]/np.where(bo[:-1]>0,bo[:-1],np.nan)-1.0; nets["BH_btc"]=np.nan_to_num(btcr)
    Wew=np.where(np.isfinite(C),1.0,0.0); Wew=Wew/np.maximum(Wew.sum(axis=1,keepdims=True),1); nets["BH_alt"]=net_of(Wew)
    m=(idx>=CUT); dts=idx[m]; starts=list(range(0,int(m.sum())-SL+1,SL)); wkd=[str(pd.Timestamp(dts[i]).date()) for i in starts]
    order=["CONV","SMA_BIN","SMA_ER","SMA_CNT","SMA_SLF","SMA_CND","PARTIAL","BH_alt","BH_btc"]
    wk={k:[float(np.prod(1+nets[k][m][i:i+SL])-1)*100 for i in starts] for k in order}
    # summary
    summ={}
    for k in order:
        n=nets[k][m]; eq=np.cumprod(1+n); comp=float(eq[-1]-1)*100; dd=float(np.min((eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq))*100)
        w=np.array(wk[k]); summ[k]={"comp":round(comp,1),"maxdd":round(dd,1),"pct_pos":round(float(np.mean(w>0)),2),"final":round(START*(1+comp/100),0)}
    print("="*118)
    print(f"ALL CANDIDATES -- WEEKLY %, full OOS {wkd[0]}..{wkd[-1]} ({len(wkd)} wks) | u100 basket, next-open, prod cost, $10k")
    print("="*118)
    print(f"\n  SUMMARY:  {'':<10}"+"".join(f"{k:>9}" for k in order))
    print(f"    {'final$':<10}"+"".join(f"{summ[k]['final']:>9.0f}" for k in order))
    print(f"    {'compound%':<10}"+"".join(f"{summ[k]['comp']:>+8.0f}%" for k in order))
    print(f"    {'maxDD%':<10}"+"".join(f"{summ[k]['maxdd']:>+8.0f}%" for k in order))
    print(f"    {'%wk pos':<10}"+"".join(f"{summ[k]['pct_pos']*100:>8.0f}%" for k in order))
    print(f"\n  WEEKLY %:  {'week':<11}"+"".join(f"{k[:8]:>9}" for k in order))
    for i,dt in enumerate(wkd):
        print(f"    {dt:<11}"+"".join(f"{wk[k][i]:>+9.1f}" for k in order))
    out={"check":"all_candidates_weekly","window":f"{wkd[0]}..{wkd[-1]}","order":order,"summary":summ,
         "weekly":{"dates":wkd,**{k:[round(x,2) for x in wk[k]] for k in order}},
         "note":"u100 cross-sectional basket (NOT BTC); BH_btc/BH_alt are benchmarks; SMA_CNT/SMA_CND refuted; gates designed on this data (de-risked beta, not robust alpha)","ts":datetime.now().strftime("%Y%m%d_%H%M%S")}
    with open(HERE/"strat_all_candidates_weekly.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\nSaved {HERE/'strat_all_candidates_weekly.json'}")


if __name__=="__main__":
    main()
