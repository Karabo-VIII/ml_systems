"""QUANT: the ONE engine + regime gate -- SHARED (market) vs SELF (per-asset) gate, + the MOVER-MISS decomposition.

User: if all engines are one and regime is the one gate, show how it works (7d slice / 14d lookback). Compare gates.
Is the regime SHARED (one market switch, closes ALL assets at once) or SELF (per-asset)? If shared, do we MISS daily
movers when the gate closes for everyone?

ARCHITECTURE (confirmed in code): signal = per-asset MA cross (SELF); current gate = SHARED breadth (one scalar/bar,
W_invvol g[:,None] broadcast -> all assets cash at once).

PRE-REGISTERED. Compare on the single SMA engine (per-asset cross, 14d-lookback/7d-slice, inv-vol cap33, next-open, prod):
  NONE   : cross only, NO market gate (max mover-capture; trade any asset whose cross is long).
  SHARED : cross + market-breadth gate (OLD/current; closes all at once).
  SELF   : cross + per-asset own-regime (asset above its OWN 100d-MA) -- a mover can trade even if market is weak.
  BOTH   : shared breadth AND self.
DECISIVE MOVER-MISS TEST: split the NONE engine's weeks by whether the SHARED gate was OPEN or CLOSED that week.
  NONE's return IN shared-CLOSED weeks = the 'movers' the shared gate forgoes. If >0 -> shared gate COSTS us (misses
  movers). If <0 -> those movers are falling knives; the shared gate SAVES us. H0: shared-closed weeks are <=0 (knives).
OOS-2024 + OOS-2025, u100 1d, point-in-time.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd
from scipy import stats as ss

HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE)); ROOT=Path(__file__).resolve().parents[3]/"src"; sys.path.insert(0,str(ROOT))
from strat.ma_slice_engine import held_rolling, _cached_band_held
from strat.ma_wealth_driver import _load_asset, _u100_symbols
from strat.canonical_exits import COST_PROD
import strat.fleet_lab as fl
BPD=fl.BARS_PER_DAY["1d"]; LB=14*BPD; SL=7*BPD; SW=100*BPD; CAP=0.33
CUT=pd.Timestamp("2024-05-15"); OOS25=pd.Timestamp("2025-01-01")
WIN={"OOS-2024":(int(pd.Timestamp("2025-01-01").value//10**6),"2024-05-15","2024-12-31"),
     "OOS-2025":(int(pd.Timestamp("2026-07-01").value//10**6),"2025-01-01","2026-07-01")}


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
    out={"check":"regime_gate_compare_SMA","ts":datetime.now().strftime("%Y%m%d_%H%M%S"),"windows":{}}
    print("="*104); print("ONE ENGINE (SMA) + REGIME GATE: SHARED vs SELF vs NONE  +  the MOVER-MISS decomposition (7d/14d)"); print("="*104)
    for window in ["OOS-2024","OOS-2025"]:
        wall,ws,we=WIN[window]; closes={};opens={};H={}
        for s in _u100_symbols():
            r=_load_asset(s,"1d",wall)
            if r is None: continue
            d,o,h,l,c=r; closes[s]=pd.Series(np.asarray(c,float),index=d); opens[s]=pd.Series(np.asarray(o,float),index=d)
            H[s]=pd.Series(held_rolling(c,_cached_band_held("SMA","1d",s,o,h,l,c,wall),LB,SL),index=d)
        cdf=pd.DataFrame(closes).sort_index(); idx=cdf.index; cols=list(cdf.columns); T=len(idx); A=len(cols)
        odf=pd.DataFrame(opens).reindex(idx).reindex(columns=cols); C=cdf.to_numpy(); O=odf.to_numpy()
        Hm=pd.DataFrame(H).reindex(idx).fillna(0).reindex(columns=cols).fillna(0).to_numpy()
        vol=cdf.pct_change(fill_method=None).rolling(max(20,5*BPD),min_periods=10).std().to_numpy()
        roo=np.full((T,A),0.0); roo[:-1]=O[1:]/np.where(O[:-1]>0,O[:-1],np.nan)-1.0; roo=np.nan_to_num(roo)
        # SHARED gate (market breadth, one scalar/bar)
        sall=cdf.rolling(SW,min_periods=SW//2).mean().to_numpy(); vc=np.isfinite(sall).sum(axis=1)
        brd=np.where(vc>0,np.nansum((C>sall)*np.isfinite(sall),axis=1)/np.maximum(vc,1),np.nan); gshared=hyst(brd,0.5,0.4)
        # SELF gate (per-asset: asset above its OWN 100d-MA, per-asset hysteresis via level)
        own_sma=cdf.rolling(SW,min_periods=SW//2).mean().to_numpy(); gself=((C>own_sma)&np.isfinite(own_sma)).astype(float)
        def book(gate2d):
            mult=Hm*gate2d; iv=np.where((mult>0)&np.isfinite(vol)&(vol>0),1.0/np.where(vol>0,vol,np.nan),0.0)
            iv=np.nan_to_num(iv)*mult; sden=iv.sum(axis=1,keepdims=True); W=np.minimum(np.where(sden>0,iv/np.maximum(sden,1e-12),0.0),CAP)
            Wl=np.vstack([np.zeros((1,A)),W[:-1]]); net=np.nansum(Wl*roo,axis=1)-np.nansum(np.abs(W-Wl),axis=1)*(COST_PROD/2)
            return net, Wl
        gates={"NONE (no gate)":np.ones((T,A)),"SHARED breadth (OLD)":np.broadcast_to(gshared[:,None],(T,A)).astype(float),
               "SELF per-asset":gself,"BOTH (shared&self)":np.broadcast_to(gshared[:,None],(T,A)).astype(float)*gself}
        mask=(idx>=pd.Timestamp(ws))&(idx<pd.Timestamp(we)); starts=list(range(0,int(mask.sum())-SL+1,SL))
        def summ(net,Wl):
            n=net[mask]; gw=Wl.sum(axis=1)[mask]; wk=np.array([float(np.prod(1+n[i:i+SL])-1)*100 for i in starts])
            eq=np.cumprod(1+n); dd=float(np.min((eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq))*100) if len(eq)>1 else 0
            dep=np.array([np.mean(gw[i:i+SL]>1e-9)>0.5 for i in starts])
            return {"comp":round(float(np.prod(1+n)-1)*100,1),"maxdd":round(dd,1),"deploy":round(float(np.mean(dep)),2),
                    "wk_mean":round(float(np.mean(wk)),2),"pct_pos":round(float(np.mean(wk>0)),2),"nw_p":round(nw_p(wk),3),"p05":bootp05(wk)},wk,dep
        res={}; wks={}
        for nm,gt in gates.items():
            net,Wl=book(gt); s,wk,dep=summ(net,Wl); res[nm]=s; wks[nm]=(wk,dep,net,Wl)
        # shared gate weekly open/closed
        gsh_wk=np.array([np.mean(gshared[np.where(mask)[0][i:i+SL]]>0)>0.5 for i in starts])
        # MOVER-MISS: NONE engine return in weeks where SHARED gate CLOSED
        nonewk=wks["NONE (no gate)"][0]; closedw=~gsh_wk; openw=gsh_wk
        miss_closed=nonewk[closedw]; cap_open=nonewk[openw]
        out["windows"][window]={"gates":res,"mover_miss":{
            "n_shared_closed_wks":int(closedw.sum()),"n_shared_open_wks":int(openw.sum()),
            "NONE_ret_in_shared_CLOSED_wks_mean":round(float(np.mean(miss_closed)),2) if len(miss_closed) else None,
            "NONE_ret_in_shared_CLOSED_wks_median":round(float(np.median(miss_closed)),2) if len(miss_closed) else None,
            "NONE_compound_in_CLOSED_wks":round(float(np.prod(1+miss_closed/100)-1)*100,1) if len(miss_closed) else None,
            "NONE_ret_in_shared_OPEN_wks_mean":round(float(np.mean(cap_open)),2) if len(cap_open) else None}}
        print(f"\n  {window}  ({ws}..{we})  -- single SMA engine, 14d/7d")
        print(f"    {'gate':<22}{'comp%':>8}{'maxDD':>8}{'deploy':>8}{'wk_mean':>9}{'%pos':>6}{'NW_p':>7}{'p05':>7}")
        for nm,s in res.items():
            print(f"    {nm:<22}{s['comp']:>+7.0f}%{s['maxdd']:>+8.0f}{s['deploy']*100:>7.0f}%{s['wk_mean']:>+8.2f}%{s['pct_pos']*100:>5.0f}%{s['nw_p']:>7.3f}{(s['p05'] if s['p05'] is not None else 0):>+6.0f}%")
        mm=out["windows"][window]["mover_miss"]
        print(f"\n    MOVER-MISS (does the SHARED gate forgo profitable movers?):")
        print(f"      shared gate CLOSED in {mm['n_shared_closed_wks']} wks / OPEN in {mm['n_shared_open_wks']} wks")
        print(f"      NONE engine return IN shared-CLOSED weeks: mean {mm['NONE_ret_in_shared_CLOSED_wks_mean']}% / median {mm['NONE_ret_in_shared_CLOSED_wks_median']}% / compound {mm['NONE_compound_in_CLOSED_wks']}%")
        print(f"      -> {'movers are KNIVES (shared gate SAVES us)' if (mm['NONE_ret_in_shared_CLOSED_wks_mean'] or 0)<=0 else 'movers PROFITABLE (shared gate COSTS us)'}")
        print(f"      (for contrast, NONE in shared-OPEN weeks: mean {mm['NONE_ret_in_shared_OPEN_wks_mean']}%)")
    with open(HERE/"strat_regime_gate_compare.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\nSaved {HERE/'strat_regime_gate_compare.json'}")


if __name__=="__main__":
    main()
