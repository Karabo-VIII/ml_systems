"""QUANT: standalone, DECOUPLED, minimal per-TI-type engines (SMA / EMA / WMA / VIDYA / KAMA) vs the 2-7%/wk target.

User mandate (canonical): one engine PER MA type, NOT shared/CONV (no cross-type influence). Efficient (minimal configs:
the 443-cross CONV book = only n_eff~4 independent bets, per-type ~2.6, so ~6 horizon-spaced crosses capture a type).
Constraint: 14d lookback (past info -> infer regime/vol) + 7d trading window (rolling slice); long-only spot u100 1d;
goal = WIN the 7d window, canonical target 2-7%/wk.

PRE-REGISTERED. H0 (the honest null, from V14 + this session): per-7d-window profitability is structurally low (~34%),
deployed-week win ~50%, no statistically-significant alpha -> the engine CANNOT win 2-7% EVERY week; it wins SOME weeks
in-band, cash/loses others. ALT: a per-TI engine clears in-[2,7] on a majority of deployed weeks with NW p<0.05 / p05>0.
Decision: report, per type, FULL band vs MINIMAL (6 crosses): the 7d-window distribution (% in [2,7], median, %>0, mean),
compound, maxDD, deployed-week win-rate, NW(L3) p, block-boot p05. Asymmetric loss: do NOT sell "wins every week".
MINIMAL crosses chosen by HORIZON SPACING (NOT performance -- config performance is non-persistent, Spearman~0).
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd
from scipy import stats as ss

HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
ROOT=Path(__file__).resolve().parents[3]/"src"; sys.path.insert(0,str(ROOT))
import ma_weekly_replay as R
from strat.ma_type_upgrade import _MA, held_cross
from strat.ma_slice_engine import held_rolling, _cached_band_held
from strat.ma_movecatch_engine import select_band_trainval
from strat.ma_wealth_driver import _load_asset, _u100_symbols
from strat.canonical_exits import COST_PROD
BPD=R.BPD; LB=14*BPD; SL=7*BPD; SW=100*BPD; CAP=0.33; WALL=R.WIN["OOS-2025"][0]
CUT=pd.Timestamp("2024-05-15"); OOS25=pd.Timestamp("2025-01-01")
TYPES=["SMA","EMA","WMA","VIDYA","KAMA"]
# horizon-spanning minimal crosses (fast->slow), spacing-chosen NOT performance-chosen:
MINIMAL=[(3,10),(5,20),(8,40),(13,60),(21,100),(34,150)]


def hyst(s,u,d):
    n=len(s);o=np.zeros(n,int);st=0
    for i in range(n):
        if not np.isfinite(s[i]): o[i]=st; continue
        if st==0 and s[i]>u: st=1
        elif st==1 and s[i]<d: st=0
        o[i]=st
    return o


def minimal_band_held(c, ma_type):
    on=np.zeros(len(c))
    for a,b in MINIMAL: on=on+held_cross(np.asarray(c,float),[a,b],ma_type)
    return (on/len(MINIMAL) > 0.5).astype(np.int8)


def W_invvol(H, vol, g, cap):
    mult=H*(g[:,None]>0); iv=np.where((mult>0)&np.isfinite(vol)&(vol>0),1.0/np.where(vol>0,vol,np.nan),0.0)
    iv=np.nan_to_num(iv)*mult; s=iv.sum(axis=1,keepdims=True); return np.minimum(np.where(s>0,iv/np.maximum(s,1e-12),0.0),cap)


def net_of(W,roo):
    Wl=np.vstack([np.zeros((1,W.shape[1])),W[:-1]]); return np.nansum(Wl*roo,axis=1)-np.nansum(np.abs(W-Wl),axis=1)*(COST_PROD/2), Wl


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


def windist(net, Wl, wm):
    n=net[wm]; gw=Wl.sum(axis=1)[wm]; starts=list(range(0,len(n)-SL+1,SL))
    wk=np.array([float(np.prod(1+n[i:i+SL])-1)*100 for i in starts])
    dep=np.array([np.mean(gw[i:i+SL]>1e-9)>0.5 for i in starts])
    eq=np.cumprod(1+n); dd=float(np.min((eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq))*100) if len(eq)>1 else 0
    depwk=wk[dep]
    return {"comp":round(float(np.prod(1+n)-1)*100,1),"maxdd":round(dd,1),"n_wk":len(wk),"deploy":round(float(np.mean(dep)),2),
            "in27_all":round(float(np.mean((wk>=2)&(wk<=7))),2),"in27_dep":round(float(np.mean((depwk>=2)&(depwk<=7))) if len(depwk) else 0,2),
            "median_dep":round(float(np.median(depwk)),2) if len(depwk) else 0,"win_dep":round(float(np.mean(depwk>0)) if len(depwk) else 0,2),
            "mean_all":round(float(np.mean(wk)),2),"nw_p":round(nw_p(wk),3),"p05":bootp05(wk)}


def main():
    # ---- single point-in-time load (wall=latest); per-name FULL + MINIMAL held ----
    closes={}; opens={}; fullH={}; minH={}
    syms=_u100_symbols()
    for s in syms:
        r=_load_asset(s,"1d",WALL)
        if r is None: continue
        d,o,h,l,c=r; closes[s]=pd.Series(np.asarray(c,float),index=d); opens[s]=pd.Series(np.asarray(o,float),index=d)
    cdf=pd.DataFrame(closes).sort_index(); idx=cdf.index; cols=list(cdf.columns); T=len(idx); A=len(cols)
    odf=pd.DataFrame(opens).reindex(idx).reindex(columns=cols); C=cdf.to_numpy(); O=odf.to_numpy()
    vol=cdf.pct_change(fill_method=None).rolling(max(20,5*BPD),min_periods=10).std().to_numpy()
    roo=np.full((T,A),0.0); roo[:-1]=O[1:]/np.where(O[:-1]>0,O[:-1],np.nan)-1.0; roo=np.nan_to_num(roo)
    sall=cdf.rolling(SW,min_periods=SW//2).mean().to_numpy(); vc=np.isfinite(sall).sum(axis=1)
    brd=np.where(vc>0,np.nansum((C>sall)*np.isfinite(sall),axis=1)/np.maximum(vc,1),np.nan); g=hyst(brd,0.5,0.4)
    for t in TYPES:
        F={}; M={}
        for s in cols:
            r=_load_asset(s,"1d",WALL)
            if r is None: continue
            d,o,h,l,c=r
            F[s]=pd.Series(held_rolling(c,_cached_band_held(t,"1d",s,o,h,l,c,WALL),LB,SL),index=d)
            M[s]=pd.Series(held_rolling(c,minimal_band_held(c,t),LB,SL),index=d)
        fullH[t]=pd.DataFrame(F).reindex(idx).fillna(0).reindex(columns=cols).fillna(0).to_numpy()
        minH[t]=pd.DataFrame(M).reindex(idx).fillna(0).reindex(columns=cols).fillna(0).to_numpy()
    out={"check":"STRAT_per_ti_engines","minimal_crosses":MINIMAL,"target":"2-7%/wk (canonical)","ts":datetime.now().strftime("%Y%m%d_%H%M%S"),"windows":{}}
    print("="*116); print("STANDALONE PER-TI ENGINES (decoupled, 14d/7d, breadth-gated inv-vol cap33) vs 2-7%/wk target -- FULL band vs MINIMAL 6-cross"); print("="*116)
    for wn,wm in [("OOS-2024",(idx>=CUT)&(idx<OOS25)),("OOS-2025",(idx>=OOS25))]:
        out["windows"][wn]={}
        print(f"\n  {wn}")
        print(f"    {'engine':<14}{'comp%':>7}{'maxDD':>7}{'deploy':>7}{'in[2,7]all':>11}{'in[2,7]dep':>11}{'medianDep':>10}{'winDep':>8}{'NW_p':>7}{'p05':>7}")
        print("    "+"-"*94)
        for t in TYPES:
            for tag,H in [("FULL",fullH[t]),("MIN6",minH[t])]:
                W=W_invvol(H,vol,g,CAP); net,Wl=net_of(W,roo); s=windist(net,Wl,wm)
                out["windows"][wn][f"{t}_{tag}"]=s
                print(f"    {t+'|'+tag:<14}{s['comp']:>+6.0f}%{s['maxdd']:>+7.0f}{s['deploy']*100:>6.0f}%{s['in27_all']*100:>10.0f}%{s['in27_dep']*100:>10.0f}%{s['median_dep']:>+9.2f}%{s['win_dep']*100:>7.0f}%{s['nw_p']:>7.3f}{(s['p05'] if s['p05'] is not None else 0):>+6.0f}%")
    # honest scorecard vs 2-7% target (OOS-ALL deployed weeks, FULL engines)
    print(f"\n  TARGET CHECK (2-7%/wk): 'in[2,7]dep' = fraction of DEPLOYED 7d windows landing IN the 2-7% band.")
    print(f"  honest read: an engine 'wins' (in-band) on this fraction of the weeks it actually deploys; cash/out the rest.")
    with open(HERE/"strat_per_ti_engines.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\n  Saved {HERE/'strat_per_ti_engines.json'}")


if __name__=="__main__":
    main()
