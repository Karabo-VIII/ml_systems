"""NATIVE SOUND-FAMILY FLEET (user intent: a curated fleet of DISTINCT, mechanistically-sound bots -- MR removed).
Each sound TI (trend/momentum/breakout) trades its OWN native signal (whole-basket of names where it's long), K3
multi-conditioner gated, inv-vol cap33 -- NO momentum-top-5 overlay (that homogenized them). Per TI: slice stats
(deploy%, cash-or-profit%, mean, deployed-median, delta). THEN the key question: cross-correlation of the weekly P&L
streams -> are they a REAL fleet (distinct, n_eff>1) or still ONE trend engine? Curate the best + least-correlated.
TRAIN+VAL only (WALL=VAL_END, OOS held out). This is the honest 'do you have a fleet' test.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE=Path(__file__).resolve().parent; ROOT=Path(__file__).resolve().parents[3]/"src"; sys.path.insert(0,str(ROOT))
from strat.ma_wealth_driver import _u100_symbols, _norm_sym
from strat.ma_slice_engine import VAL_START, VAL_END
from strat.deep2020_ti_pipeline import INDICATORS
from pipeline.chimera_loader import ChimeraLoader
from strat.canonical_exits import COST_PROD
SL=7; CAP=0.33
WALL=int(pd.Timestamp(VAL_END).value//10**6)
SOUND={k for k in INDICATORS if INDICATORS[k]["family"] in ("trend","momentum","breakout")}


def hyst(s,u,d):
    o=np.zeros(len(s),int);st=0
    for i in range(len(s)): st=1 if s[i]>u else (0 if s[i]<d else st); o[i]=st
    return o


def load():
    data={}; nvol=0
    for s in _u100_symbols():
        try: df=ChimeraLoader().load(_norm_sym(s),cadence="1d",features=["open","high","low","close","volume"]); v=df["volume"].to_numpy().astype(float); nvol+=1
        except Exception:
            try: df=ChimeraLoader().load(_norm_sym(s),cadence="1d",features=["open","high","low","close"]); v=None
            except Exception: continue
        ts=df["timestamp"].to_numpy(); sel=np.where(ts<WALL)[0]
        if len(sel)<200: continue
        d=pd.to_datetime(ts[sel],unit="ms").floor("D"); keep=~pd.Index(d).duplicated(keep="last"); sel=sel[keep]; d=d[keep]
        data[s]={"d":d,"o":df["open"].to_numpy().astype(float)[sel],"h":df["high"].to_numpy().astype(float)[sel],
                 "l":df["low"].to_numpy().astype(float)[sel],"c":df["close"].to_numpy().astype(float)[sel],
                 "vol":(v[sel] if v is not None else np.full(len(sel),np.nan))}
    return data,nvol


def main():
    data,nvol=load(); syms=list(data)
    cdf=pd.DataFrame({s:pd.Series(data[s]["c"],index=data[s]["d"]) for s in syms}).sort_index()
    odf=pd.DataFrame({s:pd.Series(data[s]["o"],index=data[s]["d"]) for s in syms}).reindex(cdf.index)
    idx=cdf.index; cols=list(cdf.columns); T=len(idx); A=len(cols); C=cdf.to_numpy(); O=odf.to_numpy(); aidx={s:j for j,s in enumerate(cols)}
    R=np.full((T,A),np.nan); R[1:]=C[1:]/np.where(C[:-1]>0,C[:-1],np.nan)-1.0
    roo=np.full((T,A),0.0); roo[:-1]=O[1:]/np.where(O[:-1]>0,O[:-1],np.nan)-1.0; roo=np.nan_to_num(roo)
    vol=cdf.pct_change(fill_method=None).rolling(20,min_periods=10).std().to_numpy(); valid=np.isfinite(C)
    sma100=cdf.rolling(100,min_periods=50).mean().to_numpy(); vc=np.isfinite(sma100).sum(axis=1)
    brd=np.where(vc>0,np.nansum((C>sma100)*np.isfinite(sma100),axis=1)/np.maximum(vc,1),0.0)
    bc=cdf["BTCUSDT"].to_numpy() if "BTCUSDT" in cols else C[:,0]; mv=np.abs(np.diff(bc,prepend=bc[0]))
    ber=np.abs(bc-pd.Series(bc).shift(14).to_numpy())/np.maximum(pd.Series(mv).rolling(14,min_periods=7).sum().to_numpy(),1e-9)
    bsma=pd.Series(bc).rolling(50,min_periods=50).mean().to_numpy(); disp=np.nanstd(R,axis=1); dmed=np.nanmedian(disp[np.isfinite(disp)])
    score=hyst(brd,0.5,0.4)+(brd>pd.Series(brd).shift(7).to_numpy()).astype(int)+(ber>0.35).astype(int)+(bc>bsma).astype(int)+(disp>dmed).astype(int)
    starts=[i for i in range(0,T-SL+1,SL)]; ew=np.where(valid,1.0,0.0); ew=ew/np.maximum(ew.sum(axis=1,keepdims=True),1)
    def net_of(W):
        Wl=np.vstack([np.zeros((1,A)),W[:-1]]); Wll=np.vstack([np.zeros((1,A)),Wl[:-1]])
        return np.nansum(np.where(valid,Wl*roo,0.0),axis=1)-np.nansum(np.abs(Wl-Wll),axis=1)*(COST_PROD/2), Wl
    mkt,_=net_of(ew)
    def type_held(k):
        spec=INDICATORS[k]; grid=spec["grid"]() if callable(spec["grid"]) else spec["grid"]; acc=np.zeros((T,A)); cnt=0
        for p in grid:
            Hc=np.zeros((T,A)); ok=False
            for s in syms:
                try: h=spec["base"](data[s],p); ok=True
                except Exception: continue
                Hc[:,aidx[s]]=pd.Series(h,index=data[s]["d"]).reindex(idx).fillna(0).to_numpy()
            if ok: acc+=Hc; cnt+=1
        return (acc/max(cnt,1))>=0.5
    def native_book(Held):  # whole-basket of held names, gated, inv-vol -- NO momentum selection
        gated=Held*(score>=3)[:,None]
        w=np.where((gated>0)&np.isfinite(vol)&(vol>0),1.0/np.where(vol>0,vol,np.nan),0.0); w=np.nan_to_num(w)*gated
        s=w.sum(axis=1,keepdims=True); return np.minimum(np.where(s>0,w/np.maximum(s,1e-12),0.0),CAP)
    def wk(net,rows): return np.array([float(np.prod(1+net[i:i+SL])-1)*100 for i in rows])
    rows=[i for i in starts if idx[i]<pd.Timestamp(VAL_END)]; mwk=wk(mkt,rows)
    res={}; pnl={}
    for k in sorted(SOUND,key=lambda x:INDICATORS[x]["family"]):
        if INDICATORS[k].get("loader")=="ohlcv" and nvol<A*0.5: continue
        try: net,Wl=net_of(native_book(type_held(k)))
        except Exception: continue
        sn=wk(net,rows); expo=np.array([float(np.mean(Wl[i:i+SL].sum(axis=1))) for i in rows]); dep=expo>0.05
        res[k]={"family":INDICATORS[k]["family"],"deploy":round(float(np.mean(dep)),2),"cash_prof":round(1-float(np.mean(dep&(sn<-1e-4))),2),
                "mean_sl":round(float(np.mean(sn)),2),"dep_med":round(float(np.median(sn[dep])),2) if dep.any() else 0.0,
                "delta":round(float(np.mean(sn)-np.mean(mwk)),2)}
        pnl[k]=sn
    # cross-correlation / n_eff of the native fleet weekly P&L
    ks=list(pnl); P=np.array([pnl[k] for k in ks]); sd=P.std(axis=1); ks=[k for k,s in zip(ks,sd) if s>1e-9]; P=P[sd>1e-9]
    n=len(P); cm=np.corrcoef(P); rbar=float((cm.sum()-n)/(n*(n-1))); neff=n/(1+(n-1)*max(rbar,0))
    z=(P-P.mean(axis=1,keepdims=True))/P.std(axis=1,keepdims=True); ev=np.linalg.eigvalsh(np.corrcoef(z))[::-1]; pc1=float(ev[0]/n)
    print("="*104); print(f"NATIVE SOUND-FAMILY FLEET ({n} bots, each on its OWN signal, K3-gated) -- TRAIN+VAL (OOS held out)"); print("="*104)
    print(f"\n  {'TI':<11}{'family':<12}{'deploy%':>8}{'cash/prof%':>11}{'mean_sl%':>9}{'DEPmed%':>8}{'delta%':>8}")
    curf=None
    for k in sorted(res,key=lambda x:(res[x]["family"],-res[x]["dep_med"])):
        r=res[k]
        if r["family"]!=curf: curf=r["family"]; print("  "+"-"*92)
        print(f"  {k:<11}{r['family']:<12}{r['deploy']*100:>7.0f}%{r['cash_prof']*100:>10.0f}%{r['mean_sl']:>+8.2f}%{r['dep_med']:>+7.2f}%{r['delta']:>+7.2f}%")
    print(f"\n  FLEET DISTINCTNESS (the key test): mean pairwise P&L corr {rbar:+.2f} -> n_eff = {neff:.1f} of {n}  |  PC1 = {pc1*100:.0f}% of variance")
    print(f"  {'-> REAL fleet (distinct bots)' if neff>=3 else '-> NOT a fleet: ~one trend engine (native signals still co-move)'}")
    out={"check":"native_fleet","n_bots":n,"rbar":round(rbar,3),"n_eff":round(neff,2),"pc1":round(pc1,3),"bots":res}
    with open(HERE/"strat_native_fleet.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"  Saved {HERE/'strat_native_fleet.json'}")


if __name__=="__main__":
    main()
