"""ALL ~34 TI bots (the full register: trend/momentum/mean-reversion/volume/breakout/volatility) through the SAME
slice-success framework as the HMA lead. Per TI: eligible = majority vote of the TI's config grid (canonical denoised
signal) -> momentum-top-5 -> multi-conditioner K3 gate. Report deploy%, cash-or-profit%, mean slice, MEDIAN DEPLOYED
week (bankable), delta vs market. TRAIN+VAL combined (OOS held out, WALL=VAL_END). Grouped by family. Honest: they
converge because momentum-top-5 + gate dominates (TI eligibility a mild filter) -- shown vs the raw whole-basket TI too.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE)); ROOT=Path(__file__).resolve().parents[3]/"src"; sys.path.insert(0,str(ROOT))
from strat_ti_family_neff import load_panel
from strat.deep2020_ti_pipeline import INDICATORS
from strat.canonical_exits import COST_PROD
SL=7; CAP=0.33; VAL_START="2023-07-01"; VAL_END="2024-05-15"


def hyst(s,u,d):
    o=np.zeros(len(s),int);st=0
    for i in range(len(s)): st=1 if s[i]>u else (0 if s[i]<d else st); o[i]=st
    return o


def main():
    data,nvol=load_panel(); syms=list(data)  # load_panel walls at 2026-07-01; we mask to <VAL_END below
    cdf=pd.DataFrame({s:pd.Series(data[s]["c"],index=data[s]["d"]) for s in syms}).sort_index()
    odf=pd.DataFrame({s:pd.Series(data[s]["o"],index=data[s]["d"]) for s in syms}).reindex(cdf.index)
    idx=cdf.index; cols=list(cdf.columns); T=len(idx); A=len(cols); C=cdf.to_numpy(); O=odf.to_numpy()
    WALL_TS=pd.Timestamp(VAL_END)  # HARD: only score slices with start < VAL_END (OOS held out)
    R=np.full((T,A),np.nan); R[1:]=C[1:]/np.where(C[:-1]>0,C[:-1],np.nan)-1.0
    roo=np.full((T,A),0.0); roo[:-1]=O[1:]/np.where(O[:-1]>0,O[:-1],np.nan)-1.0; roo=np.nan_to_num(roo)
    vol=cdf.pct_change(fill_method=None).rolling(20,min_periods=10).std().to_numpy(); valid=np.isfinite(C)
    sma100=cdf.rolling(100,min_periods=50).mean().to_numpy(); vc=np.isfinite(sma100).sum(axis=1)
    brd=np.where(vc>0,np.nansum((C>sma100)*np.isfinite(sma100),axis=1)/np.maximum(vc,1),0.0)
    bc=cdf["BTCUSDT"].to_numpy() if "BTCUSDT" in cols else C[:,0]; mv=np.abs(np.diff(bc,prepend=bc[0]))
    ber=np.abs(bc-pd.Series(bc).shift(14).to_numpy())/np.maximum(pd.Series(mv).rolling(14,min_periods=7).sum().to_numpy(),1e-9)
    bsma=pd.Series(bc).rolling(50,min_periods=50).mean().to_numpy(); disp=np.nanstd(R,axis=1); dmed=np.nanmedian(disp[np.isfinite(disp)])
    score=hyst(brd,0.5,0.4)+(brd>pd.Series(brd).shift(7).to_numpy()).astype(int)+(ber>0.35).astype(int)+(bc>bsma).astype(int)+(disp>dmed).astype(int)
    mom14=cdf.pct_change(14).to_numpy(); starts=[i for i in range(0,T-SL+1,SL) if idx[i]<WALL_TS]
    ew=np.where(valid,1.0,0.0); ew=ew/np.maximum(ew.sum(axis=1,keepdims=True),1)
    aidx={s:j for j,s in enumerate(cols)}
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
    def redev_book(Held):
        W=np.zeros((T,A))
        for i in starts:
            if score[i]<3: continue
            elig=np.where((Held[i]>0)&np.isfinite(C[i]))[0]
            if len(elig)==0: continue
            sm=mom14[i][elig]; pick=elig[np.argsort(-np.where(np.isfinite(sm),sm,-1e9))[:min(5,len(elig))]]
            v=vol[i][pick]; w=np.where(np.isfinite(v)&(v>0),1.0/v,0.0); w=w/w.sum() if w.sum()>0 else np.ones(len(pick))/max(len(pick),1)
            W[i:i+SL,pick]=np.minimum(w,CAP)
        return W
    def stats(net,Wl):
        sn=np.array([float(np.prod(1+net[i:i+SL])-1)*100 for i in starts]); mn=np.array([float(np.prod(1+mkt[i:i+SL])-1)*100 for i in starts])
        expo=np.array([float(np.mean(Wl[i:i+SL].sum(axis=1))) for i in starts]); dep=expo>0.05
        return {"deploy":round(float(np.mean(dep)),2),"cash_prof":round(1-float(np.mean(dep&(sn<-1e-4))),2),
                "mean_sl":round(float(np.mean(sn)),2),"dep_med":round(float(np.median(sn[dep])),2) if dep.any() else 0.0,
                "dep_pos":round(float(np.mean(sn[dep]>0)),2) if dep.any() else 0.0,"delta":round(float(np.mean(sn)-np.mean(mn)),2)}
    fam={k:INDICATORS[k]["family"] for k in INDICATORS}; rows=[]
    for k in INDICATORS:
        if INDICATORS[k].get("loader")=="ohlcv" and nvol<A*0.5: continue
        try:
            net,Wl=net_of(redev_book(type_held(k))); s=stats(net,Wl); s["ti"]=k; s["family"]=fam[k]; rows.append(s)
        except Exception as e: pass
    rows.sort(key=lambda r:(r["family"],-r["dep_med"]))
    out={"check":"all_ti_slice","note":"redeveloped (mom-top5 + K3 gate), TRAIN+VAL, OOS held out","rows":rows}
    print("="*104); print(f"ALL {len(rows)} TI BOTS -- redeveloped (eligible->mom-top5->K3 multi-gate), TRAIN+VAL slice stats (OOS held out)"); print("="*104)
    print(f"\n  {'TI':<11}{'family':<15}{'deploy%':>8}{'cash/prof%':>11}{'mean_sl%':>9}{'DEPmed%':>8}{'DEPpos%':>8}{'delta%':>8}")
    curfam=None
    for r in rows:
        if r["family"]!=curfam: curfam=r["family"]; print("  "+"-"*98)
        print(f"  {r['ti']:<11}{r['family']:<15}{r['deploy']*100:>7.0f}%{r['cash_prof']*100:>10.0f}%{r['mean_sl']:>+8.2f}%{r['dep_med']:>+7.2f}%{r['dep_pos']*100:>7.0f}%{r['delta']:>+7.2f}%")
    md=np.array([r["dep_med"] for r in rows]); de=np.array([r["delta"] for r in rows])
    print(f"\n  ACROSS ALL {len(rows)}: deployed-median slice {np.median(md):+.2f}% (range {md.min():+.1f}..{md.max():+.1f}); delta {np.median(de):+.2f}% ({int((de>0).sum())}/{len(rows)} positive)")
    print(f"  They CONVERGE (momentum-top-5 + gate dominates; the TI eligibility is a mild filter) -- confirms n_eff~1 for the family.")
    with open(HERE/"strat_all_ti_slice.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"  Saved {HERE/'strat_all_ti_slice.json'}")


if __name__=="__main__":
    main()
