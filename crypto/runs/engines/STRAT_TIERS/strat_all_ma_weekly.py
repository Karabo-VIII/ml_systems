"""QUANT: all 8 MA TI types as INDEPENDENT standalone engines, side-by-side, VAL-2024 + OOS-2024 (old engine structure).

User: "report on all the TI MA types for the same window side by side; treat each as independently deployed; we are still
in the old trading engine structure." So: 8 types (SMA/EMA/WMA/HMA/DEMA/TEMA/KAMA/VIDYA), each its OWN full-band engine
(breadth-gated inv-vol cap33, 14d-lookback/7d-slice rolling, next-open, prod cost 0.0028 RT), point-in-time walls.
  VAL-2024 = 2023-07-01..2024-05-15 (in-sample) ; OOS-2024 = 2024-05-15..2024-12-31 (held-out).
Outputs: (A) side-by-side summary table per window; (B) the ALIGNED week-by-week return matrix (rows=weeks, cols=types);
(C) cross-type weekly-return correlation + effective-N (how independent are the 8 standalone engines?).
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd
from scipy import stats as ss

HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE)); ROOT=Path(__file__).resolve().parents[3]/"src"; sys.path.insert(0,str(ROOT))
from strat.ma_slice_engine import held_rolling, _cached_band_held, VAL_START, VAL_END, OOS_START, OOS_2024_END
from strat.ma_wealth_driver import _load_asset, _u100_symbols
from strat.canonical_exits import COST_PROD
import strat.fleet_lab as fl
BPD=fl.BARS_PER_DAY["1d"]; LB=14*BPD; SL=7*BPD; SW=100*BPD; CAP=0.33
TYPES=["SMA","EMA","WMA","HMA","DEMA","TEMA","KAMA","VIDYA"]
WIN={"VAL-2024":(int(pd.Timestamp(VAL_END).value//10**6),VAL_START,VAL_END),
     "OOS-2024":(int(pd.Timestamp("2025-01-01").value//10**6),OOS_START,OOS_2024_END)}


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


def bootp05(wk,nb=2000):
    if len(wk)<8: return None
    rng=np.random.default_rng(0); bl=4; comps=[]
    for _ in range(nb):
        pick=[]
        while len(pick)<len(wk):
            st=int(rng.integers(0,len(wk))); pick+=list(range(st,min(st+bl,len(wk))))
        pick=np.array(pick[:len(wk)]); comps.append(float(np.prod(1+wk[pick]/100)-1)*100)
    return round(float(np.percentile(comps,5)),1)


def main():
    out={"check":"ALL_MA_weekly_VAL_OOS_2024","types":TYPES,"engine":"per-type full band, breadth-gated invvol cap33, 14d/7d rolling, next-open, prod cost","ts":datetime.now().strftime("%Y%m%d_%H%M%S"),"windows":{}}
    for window in ["VAL-2024","OOS-2024"]:
        wall,ws,we=WIN[window]; closes={};opens={}; Hraw={t:{} for t in TYPES}
        syms=_u100_symbols()
        for s in syms:
            r=_load_asset(s,"1d",wall)
            if r is None: continue
            d,o,h,l,c=r; closes[s]=pd.Series(np.asarray(c,float),index=d); opens[s]=pd.Series(np.asarray(o,float),index=d)
            for t in TYPES: Hraw[t][s]=pd.Series(held_rolling(c,_cached_band_held(t,"1d",s,o,h,l,c,wall),LB,SL),index=d)
        cdf=pd.DataFrame(closes).sort_index(); idx=cdf.index; cols=list(cdf.columns); T=len(idx); A=len(cols)
        odf=pd.DataFrame(opens).reindex(idx).reindex(columns=cols); C=cdf.to_numpy(); O=odf.to_numpy()
        vol=cdf.pct_change(fill_method=None).rolling(max(20,5*BPD),min_periods=10).std().to_numpy()
        roo=np.full((T,A),0.0); roo[:-1]=O[1:]/np.where(O[:-1]>0,O[:-1],np.nan)-1.0; roo=np.nan_to_num(roo)
        sall=cdf.rolling(SW,min_periods=SW//2).mean().to_numpy(); vc=np.isfinite(sall).sum(axis=1)
        brd=np.where(vc>0,np.nansum((C>sall)*np.isfinite(sall),axis=1)/np.maximum(vc,1),np.nan); g=hyst(brd,0.5,0.4)
        mask=(idx>=pd.Timestamp(ws))&(idx<pd.Timestamp(we)); dts=idx[mask]
        starts=list(range(0,int(mask.sum())-SL+1,SL)); wkdates=[str(pd.Timestamp(dts[i]).date()) for i in starts]
        wkmat={}; summ={}
        for t in TYPES:
            Hm=pd.DataFrame(Hraw[t]).reindex(idx).fillna(0).reindex(columns=cols).fillna(0).to_numpy()
            mult=Hm*(g[:,None]>0); iv=np.where((mult>0)&np.isfinite(vol)&(vol>0),1.0/np.where(vol>0,vol,np.nan),0.0)
            iv=np.nan_to_num(iv)*mult; sden=iv.sum(axis=1,keepdims=True); W=np.minimum(np.where(sden>0,iv/np.maximum(sden,1e-12),0.0),CAP)
            Wl=np.vstack([np.zeros((1,A)),W[:-1]]); net=np.nansum(Wl*roo,axis=1)-np.nansum(np.abs(W-Wl),axis=1)*(COST_PROD/2)
            n=net[mask]; gross=Wl.sum(axis=1)[mask]
            wk=np.array([float(np.prod(1+n[i:i+SL])-1)*100 for i in starts]); wkmat[t]=wk
            dep=np.array([np.mean(gross[i:i+SL]>1e-9)>0.5 for i in starts]); eq=np.cumprod(1+n); dd=float(np.min((eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq))*100) if len(eq)>1 else 0
            summ[t]={"comp":round(float(np.prod(1+n)-1)*100,1),"maxdd":round(dd,1),"deploy":round(float(np.mean(dep)),2),
                     "mean_wk":round(float(np.mean(wk)),2),"median_wk":round(float(np.median(wk)),2),"pct_pos":round(float(np.mean(wk>0)),2),
                     "in27":round(float(np.mean((wk>=2)&(wk<=7))),2),"gt7":round(float(np.mean(wk>7)),2),"neg":round(float(np.mean(wk<0)),2),
                     "best":round(float(wk.max()),1),"worst":round(float(wk.min()),1),"nw_p":round(nw_p(wk),3),"p05":bootp05(wk)}
        # cross-type correlation + n_eff on weekly returns
        M=np.array([wkmat[t] for t in TYPES]); Mc=M-M.mean(axis=1,keepdims=True); sd=Mc.std(axis=1)
        Cm=np.corrcoef(Mc); iu=np.triu_indices_from(Cm,1); mc=float(np.nanmean(Cm[iu]))
        lam=np.linalg.eigvalsh(np.nan_to_num(Cm)); lam=lam[lam>1e-9]; neff=float((lam.sum()**2)/np.sum(lam**2))
        out["windows"][window]={"summary":summ,"weekly_matrix":{"dates":wkdates,**{t:[round(float(x),2) for x in wkmat[t]] for t in TYPES}},
                                "cross_type_mean_corr":round(mc,3),"effective_N_types":round(neff,2)}
        # ---- print ----
        tag="IN-SAMPLE" if window=="VAL-2024" else "HELD-OUT"
        print(f"\n{'#'*112}\n# {window}  ({ws}..{we})  --  {tag}   |  8 MA types, each standalone full-band engine, old structure\n{'#'*112}")
        print(f"  {'type':<7}{'comp%':>7}{'maxDD':>7}{'deploy':>7}{'mean/wk':>8}{'med/wk':>7}{'%pos':>6}{'in[2,7]':>8}{'>7%':>6}{'neg':>6}{'best':>7}{'worst':>7}{'NW_p':>6}{'p05':>6}")
        print("  "+"-"*100)
        for t in TYPES:
            s=summ[t]
            print(f"  {t:<7}{s['comp']:>+6.0f}%{s['maxdd']:>+7.0f}{s['deploy']*100:>6.0f}%{s['mean_wk']:>+7.2f}%{s['median_wk']:>+6.1f}%{s['pct_pos']*100:>5.0f}%{s['in27']*100:>7.0f}%{s['gt7']*100:>5.0f}%{s['neg']*100:>5.0f}%{s['best']:>+6.0f}%{s['worst']:>+6.0f}%{s['nw_p']:>6.2f}{(s['p05'] if s['p05'] is not None else 0):>+5.0f}%")
        print(f"\n  CROSS-TYPE: mean pairwise weekly-return corr = {mc:.2f}  ->  effective independent engines = {neff:.1f} of 8")
        # aligned weekly matrix
        print(f"\n  ALIGNED WEEKLY RETURNS (%, rows=week, cols=type):")
        print("    "+f"{'week':<12}"+"".join(f"{t[:5]:>8}" for t in TYPES))
        for i,dt in enumerate(wkdates):
            row="".join(f"{wkmat[t][i]:>+8.1f}" for t in TYPES)
            print(f"    {dt:<12}{row}")
    with open(HERE/"strat_all_ma_weekly.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\nSaved {HERE/'strat_all_ma_weekly.json'}")


if __name__=="__main__":
    main()
