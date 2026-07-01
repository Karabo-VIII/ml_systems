"""QUANT: the 4 final candidates on VAL vs OOS -- the honest out-of-sample-for-the-DESIGN test.

The gate params (breadth 0.5/0.4, ER>0.35, cap33, K8) were chosen looking at OOS this session -> OOS is IN-SAMPLE for
the gate DESIGN. VAL (2023-07-01..2024-05-15) is the window NEVER tuned on -> the genuine OOS test for the candidate
design. (VAL is in-sample only for the MA band config selection, which is a denoised ensemble, not a tuned param.)
H0: candidates degrade on VAL (design overfit to OOS). Report compound/maxDD/Calmar/%wk+/NW p/p05 side by side.
Per-window point-in-time wall (VAL wall=2024-05-15, OOS wall=2026-07-01). u100, 1d, 14d/7d, next-open, prod cost, $10k.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd
from scipy import stats as ss

HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE)); ROOT=Path(__file__).resolve().parents[3]/"src"; sys.path.insert(0,str(ROOT))
from strat.ma_slice_engine import held_rolling, _cached_band_held, VAL_START, VAL_END, OOS_START
from strat.ma_wealth_driver import _load_asset, _u100_symbols
from strat.canonical_exits import COST_PROD
import strat.fleet_lab as fl
BPD=fl.BARS_PER_DAY["1d"]; LB=14*BPD; SL=7*BPD; SW=100*BPD; CAP=0.33; START=10000.0; TYPES=["SMA","EMA","WMA","VIDYA","KAMA"]
WINDOWS={"VAL":(int(pd.Timestamp(VAL_END).value//10**6),VAL_START,VAL_END),
         "OOS":(int(pd.Timestamp("2026-07-01").value//10**6),OOS_START,"2026-07-01")}


def hyst(s,u,d):
    n=len(s);o=np.zeros(n,int);st=0
    for i in range(n):
        if not np.isfinite(s[i]): o[i]=st; continue
        if st==0 and s[i]>u: st=1
        elif st==1 and s[i]<d: st=0
        o[i]=st
    return o


def nw_p(wk,L=3):
    n=len(wk)
    if n<4 or np.std(wk)<1e-12: return 1.0
    mu=np.mean(wk); x=wk-mu; var=np.dot(x,x)/n
    for l in range(1,L+1): var+=2*(1-l/(L+1))*(np.dot(x[l:],x[:-l])/n)
    return float(1-ss.t.cdf(mu/np.sqrt(max(var,1e-12)/n),n-1))


def p05b(wk,nb=1500):
    if len(wk)<8: return None
    rng=np.random.default_rng(0); cs=[]
    for _ in range(nb):
        pk=[]
        while len(pk)<len(wk):
            s0=int(rng.integers(0,len(wk))); pk+=list(range(s0,min(s0+4,len(wk))))
        pk=np.array(pk[:len(wk)]); cs.append(float(np.prod(1+wk[pk]/100)-1)*100)
    return round(float(np.percentile(cs,5)),1)


def build_candidates(wall, ws, we):
    closes={};opens={}; th={t:{} for t in TYPES}
    for s in _u100_symbols():
        r=_load_asset(s,"1d",wall)
        if r is None: continue
        d,o,h,l,c=r; closes[s]=pd.Series(np.asarray(c,float),index=d); opens[s]=pd.Series(np.asarray(o,float),index=d)
        for t in TYPES: th[t][s]=pd.Series(held_rolling(c,_cached_band_held(t,"1d",s,o,h,l,c,wall),LB,SL),index=d)
    cdf=pd.DataFrame(closes).sort_index(); idx=cdf.index; cols=list(cdf.columns); T=len(idx); A=len(cols)
    odf=pd.DataFrame(opens).reindex(idx).reindex(columns=cols); C=cdf.to_numpy(); O=odf.to_numpy()
    conv=np.zeros((T,A))
    for t in TYPES: th[t]=pd.DataFrame(th[t]).reindex(idx).fillna(0).reindex(columns=cols).fillna(0).to_numpy(); conv+=th[t]
    Hsma=th["SMA"]
    vol=cdf.pct_change(fill_method=None).rolling(max(20,5*BPD),min_periods=10).std().to_numpy()
    roo=np.full((T,A),0.0); roo[:-1]=O[1:]/np.where(O[:-1]>0,O[:-1],np.nan)-1.0; roo=np.nan_to_num(roo)
    sall=cdf.rolling(SW,min_periods=SW//2).mean().to_numpy(); vc=np.isfinite(sall).sum(axis=1)
    brd=np.where(vc>0,np.nansum((C>sall)*np.isfinite(sall),axis=1)/np.maximum(vc,1),np.nan); brd=np.nan_to_num(brd); gbin=hyst(brd,0.5,0.4)
    bc=cdf["BTCUSDT"].to_numpy() if "BTCUSDT" in cols else C[:,0]; mv=np.abs(np.diff(bc,prepend=bc[0]))
    er=np.abs(bc-pd.Series(bc).shift(14).to_numpy())/np.maximum(pd.Series(mv).rolling(14,min_periods=7).sum().to_numpy(),1e-9)
    mom14=cdf.pct_change(14*BPD).to_numpy(); starts=[i for i in range(0,T-SL+1,SL)]
    def b1(x): return np.broadcast_to(x[:,None],(T,A)).astype(float)
    def iv(held,g2):
        mult=held*g2; w=np.where((mult>0)&np.isfinite(vol)&(vol>0),1.0/np.where(vol>0,vol,np.nan),0.0)
        w=np.nan_to_num(w)*mult; s=w.sum(axis=1,keepdims=True); return np.minimum(np.where(s>0,w/np.maximum(s,1e-12),0.0),CAP)
    def mom8er():
        gate=gbin*(er>0.35).astype(int); W=np.zeros((T,A))
        for i in starts:
            if gate[i]<=0: continue
            s=mom14[i]; e=np.where(np.isfinite(s)&np.isfinite(C[i]))[0]
            if len(e)==0: continue
            k=min(8,len(e)); top=e[np.argsort(-s[e])[:k]]; v=vol[i][top]; w=np.where(np.isfinite(v)&(v>0),1.0/v,0.0)
            w=w/w.sum() if w.sum()>0 else np.ones(k)/k; W[i:i+SL,top]=np.minimum(w,CAP)
        return W
    e=np.clip((brd-0.20)/0.60,0.15,1.0); ew=np.where(np.isfinite(C),1.0,0.0); ew=ew/np.maximum(ew.sum(axis=1,keepdims=True),1)
    Wtr=iv(Hsma,np.ones((T,A))); ta=Wtr.sum(axis=1,keepdims=True)>1e-9
    CAND={"CONV":iv(conv/len(TYPES),b1(gbin)),"SMA_ER":iv(Hsma,b1(gbin*(er>0.35).astype(int))),"PARTIAL":e[:,None]*np.where(ta,Wtr,ew),"MOM8_ER":mom8er()}
    btcr=np.zeros(T); bo=odf["BTCUSDT"].to_numpy() if "BTCUSDT" in cols else O[:,0]; btcr[:-1]=bo[1:]/np.where(bo[:-1]>0,bo[:-1],np.nan)-1.0
    def net_of(W):
        Wl=np.vstack([np.zeros((1,A)),W[:-1]]); return np.nansum(Wl*roo,axis=1)-np.nansum(np.abs(W-Wl),axis=1)*(COST_PROD/2)
    m=(idx>=pd.Timestamp(ws))&(idx<pd.Timestamp(we))
    def grade(net):
        n=net[m]; eq=np.cumprod(1+n); comp=float(eq[-1]-1)*100
        dd=float(np.min((eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq))*100) if len(eq)>1 else 0
        st=[i for i in range(0,len(n)-SL+1,SL)]; wk=np.array([float(np.prod(1+n[i:i+SL])-1)*100 for i in st])
        return {"final":round(START*(1+comp/100),0),"comp":round(comp,1),"maxdd":round(dd,1),"calmar":round(comp/abs(dd),2) if abs(dd)>1e-9 else 0,
                "n_wk":len(wk),"pct_pos":round(float(np.mean(wk>0)),2),"nw_p":round(nw_p(wk),3),"p05":p05b(wk)}
    res={nm:grade(net_of(W)) for nm,W in CAND.items()}
    res["BH_btc"]=grade(np.nan_to_num(btcr)); res["BH_alt"]=grade(net_of(ew))
    return res


def main():
    out={"check":"val_vs_oos","note":"VAL is OOS-for-the-gate-DESIGN (tuned on OOS); in-sample only for the MA band ensemble","ts":datetime.now().strftime("%Y%m%d_%H%M%S"),"windows":{}}
    for w,(wall,ws,we) in WINDOWS.items():
        out["windows"][w]={"span":f"{ws}..{we}","results":build_candidates(wall,ws,we)}
    print("="*104); print("CANDIDATES: VAL (2023-07..2024-05, untuned) vs OOS (2024-05..2026-05, gate-design in-sample)"); print("="*104)
    order=["CONV","SMA_ER","PARTIAL","MOM8_ER","BH_btc","BH_alt"]
    for w in ["VAL","OOS"]:
        R=out["windows"][w]["results"]; print(f"\n  {w}  ({out['windows'][w]['span']})   [{R['CONV']['n_wk']} weeks]")
        print(f"    {'candidate':<10}{'final$':>9}{'comp%':>8}{'maxDD':>8}{'Calmar':>8}{'%wk+':>7}{'NW_p':>7}{'p05':>7}")
        for nm in order:
            g=R[nm]; print(f"    {nm:<10}{g['final']:>9.0f}{g['comp']:>+7.0f}%{g['maxdd']:>+8.0f}{g['calmar']:>8.2f}{g['pct_pos']*100:>6.0f}%{g['nw_p']:>7.2f}{(g['p05'] or 0):>+6.0f}%")
    print(f"\n  VAL->OOS HOLD? (compound, Calmar):")
    for nm in ["CONV","SMA_ER","PARTIAL","MOM8_ER"]:
        v=out["windows"]["VAL"]["results"][nm]; o=out["windows"]["OOS"]["results"][nm]
        hold="HOLDS" if (v["comp"]>0 and o["comp"]>0) else ("VAL-only" if v["comp"]>0 and o["comp"]<=0 else ("OOS-only (VAL fails!)" if v["comp"]<=0 else "both weak"))
        print(f"    {nm:<10} VAL {v['comp']:+.0f}%/Cal {v['calmar']:>5.2f}  ->  OOS {o['comp']:+.0f}%/Cal {o['calmar']:>5.2f}   {hold}")
    with open(HERE/"strat_val_oos.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\nSaved {HERE/'strat_val_oos.json'}")


if __name__=="__main__":
    main()
