"""LOOP: can a SMARTER multi-conditioner deploy gate break the cash-or-profitable<->delta TENSION? The single-threshold
gate forces the trade (tight=high cash/prof but misses up-weeks=delta-neg; loose=delta+ but 20% losing wks). Test a
COUNT-based gate (deploy when >=K of 5 causal market conditioners favorable -- robust, not a fitted weighted score):
  1 breadth>0.5(hyst)  2 breadth rising(7d)  3 BTC-ER>0.35  4 BTC>SMA50  5 cross-sec dispersion>median.
HMA + momentum-top-5, 1d, 14d/7d, TRAIN+VAL only (WALL=VAL_END, OOS held out). For each K: deploy%, cash/prof%,
losing_wk%, delta% on TRAIN & VAL. If some K gives HIGH cash/prof AND delta>0 on BOTH (robust to deep-jackknife) =
tension broken = working bot. Else = ceiling confirmed. Deep rigor (drop-top5, no-2021, bootstrap) on the best K.
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
    R=np.full((T,A),np.nan); R[1:]=C[1:]/np.where(C[:-1]>0,C[:-1],np.nan)-1.0
    roo=np.full((T,A),0.0); roo[:-1]=O[1:]/np.where(O[:-1]>0,O[:-1],np.nan)-1.0; roo=np.nan_to_num(roo)
    vol=cdf.pct_change(fill_method=None).rolling(20,min_periods=10).std().to_numpy(); valid=np.isfinite(C)
    sma100=cdf.rolling(100,min_periods=50).mean().to_numpy(); vc=np.isfinite(sma100).sum(axis=1)
    brd=np.where(vc>0,np.nansum((C>sma100)*np.isfinite(sma100),axis=1)/np.maximum(vc,1),0.0)
    bc=cdf["BTCUSDT"].to_numpy() if "BTCUSDT" in cols else C[:,0]; mv=np.abs(np.diff(bc,prepend=bc[0]))
    ber=np.abs(bc-pd.Series(bc).shift(14).to_numpy())/np.maximum(pd.Series(mv).rolling(14,min_periods=7).sum().to_numpy(),1e-9)
    bsma=pd.Series(bc).rolling(50,min_periods=50).mean().to_numpy()
    disp=np.nanstd(R,axis=1); disp_med=np.nanmedian(disp[np.isfinite(disp)])
    c1=hyst(brd,0.5,0.4); c2=(brd>pd.Series(brd).shift(7).to_numpy()).astype(int); c3=(ber>0.35).astype(int)
    c4=(bc>bsma).astype(int); c5=(disp>disp_med).astype(int)
    score=c1+c2+c3+c4+c5  # 0..5 causal conditioners
    mom14=cdf.pct_change(14).to_numpy(); starts=[i for i in range(0,T-SL+1,SL)]
    ew=np.where(valid,1.0,0.0); ew=ew/np.maximum(ew.sum(axis=1,keepdims=True),1)
    def net_of(W):
        Wl=np.vstack([np.zeros((1,A)),W[:-1]]); Wll=np.vstack([np.zeros((1,A)),Wl[:-1]])
        return np.nansum(np.where(valid,Wl*roo,0.0),axis=1)-np.nansum(np.abs(Wl-Wll),axis=1)*(COST_PROD/2), Wl
    mkt,_=net_of(ew)
    H=np.zeros((T,A))
    for j,s in enumerate(syms):
        try:
            band=_cached_band_held(TI,"1d",s,data[s]["o"],data[s]["h"],data[s]["l"],data[s]["c"],WALL); hd=held_rolling(data[s]["c"],band,LB,SL)
            H[:,j]=pd.Series(hd,index=data[s]["d"]).reindex(idx).fillna(0).to_numpy()
        except Exception: pass
    def build(K):
        W=np.zeros((T,A))
        for i in starts:
            if score[i]<K: continue
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
        return {"deploy":round(float(np.mean(dep)),2),"cash_prof":round(1-float(np.mean(dep&(sn<-1e-4))),3),
                "delta":round(float(np.mean(sn)-np.mean(mn))*100,2),"mean_sl":round(float(np.mean(sn))*100,2),"n":len(sn)},sn,mn,rows
    out={"check":"smart_gate","TI":TI,"gates":{}}
    print("="*104); print(f"SMART MULTI-CONDITIONER GATE (>=K of 5) -- {TI}+mom-top5, TRAIN+VAL (OOS held out)"); print("="*104)
    print(f"\n  {'K':<3}{'win':<7}{'deploy%':>8}{'cash/prof%':>11}{'delta%':>8}{'mean_sl%':>9}")
    best=None
    for K in [1,2,3,4,5]:
        W=build(K); net,Wl=net_of(W); mt,_,_,_=m(net,Wl,TR0,TR1); mv,snv,mnv,_=m(net,Wl,V0,V1)
        out["gates"][f"K{K}"]={"TRAIN":mt,"VAL":mv}
        for wn,mm in [("TRAIN",mt),("VAL",mv)]:
            print(f"  {K:<3}{wn:<7}{mm['deploy']*100:>7.0f}%{mm['cash_prof']*100:>10.0f}%{mm['delta']:>+7.2f}%{mm['mean_sl']:>+8.2f}%")
        # candidate for "tension broken": cash/prof high AND delta>0 on BOTH
        if mt["delta"]>0 and mv["delta"]>0 and mt["cash_prof"]>=0.88 and mv["cash_prof"]>=0.88 and mt["deploy"]>=0.12:
            if best is None or (mt["cash_prof"]+mv["cash_prof"])>best[1]: best=(K,mt["cash_prof"]+mv["cash_prof"])
    # deep rigor on the full-sample delta for the best-cash/prof-with-positive-delta K (or K=3 default)
    Kd=best[0] if best else 3; W=build(Kd); net,Wl=net_of(W)
    rows=[i for i in starts if idx[i]>=TR0 and idx[i]<V1]; sn=np.array([float(np.prod(1+net[i:i+SL])-1) for i in rows])
    mn=np.array([float(np.prod(1+mkt[i:i+SL])-1) for i in rows]); d=sn-mn; order=np.argsort(-d)
    y21=np.array([TR1<=idx[i]<pd.Timestamp("2022-01-01") for i in rows]) if False else np.array([pd.Timestamp("2021-01-01")<=idx[i]<pd.Timestamp("2022-01-01") for i in rows])
    full=float(np.mean(d))*100; dt5=float(np.mean(np.delete(d,order[:5])))*100; no21=float(np.mean(d[~y21]))*100 if (~y21).any() else None
    rng=np.random.default_rng(0); bs=[float(np.mean(d[rng.integers(0,len(d),len(d))])) for _ in range(5000)]; p_le0=float(np.mean(np.array(bs)<=0))
    out["deep_rigor_K"]={"K":Kd,"full_delta":round(full,2),"drop_top5":round(dt5,2),"no_2021":round(no21,2) if no21 is not None else None,"boot_P_delta_le0":round(p_le0,3)}
    print(f"\n  DEEP RIGOR on K={Kd}: full delta {full:+.2f}% | drop-top5 {dt5:+.2f}% | no-2021 {(no21 if no21 is not None else 0):+.2f}% | bootstrap P(delta<=0) {p_le0:.2f}")
    verdict = "TENSION BROKEN" if (best and dt5>0 and p_le0<0.10 and (no21 is None or no21>0)) else "CEILING CONFIRMED (no robust high-cash/prof + delta>0 gate)"
    out["verdict"]=verdict
    print(f"  VERDICT: {verdict}")
    with open(HERE/"strat_smart_gate.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"  Saved {HERE/'strat_smart_gate.json'}")


if __name__=="__main__":
    main()
