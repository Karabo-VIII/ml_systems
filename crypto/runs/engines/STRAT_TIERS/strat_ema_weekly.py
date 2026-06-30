"""QUANT: the canonical EMA engine (full EMA|1d band, as configured) -- week-by-week on VAL-2024 + OOS-2024, independently.

User: "take the EMA engine (however it was configured before), run it through the 2024 VAL and 2024 OOS windows and report
the WEEKLY results, independently." The point: see the ACTUAL per-week returns, not aggregates -- '0% in [2,7]' can mean
the winning weeks are >7% (right tail), NOT losing.

Engine = canonical EMA|1d FULL band (105 crosses, _cached_band_held), breadth-gated inv-vol cap33, 14d-lookback/7d-slice
ROLLING, next-open fills, prod cost 0.0028 RT. Point-in-time walls per window (no leakage).
  VAL-2024  = 2023-07-01 .. 2024-05-15  (IN-SAMPLE: the band was selected on TRAIN+VAL -> expect it to look good)
  OOS-2024  = 2024-05-15 .. 2024-12-31  (HELD-OUT: the honest read)
Reports per window: every 7d week (start date, return%, DEPLOY/cash, running equity $10k) + summary.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd

HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE)); ROOT=Path(__file__).resolve().parents[3]/"src"; sys.path.insert(0,str(ROOT))
from strat.ma_slice_engine import held_rolling, _cached_band_held, VAL_START, VAL_END, OOS_START, OOS_2024_END
from strat.ma_wealth_driver import _load_asset, _u100_symbols
from strat.canonical_exits import COST_PROD
import strat.fleet_lab as fl
BPD=fl.BARS_PER_DAY["1d"]; LB=14*BPD; SL=7*BPD; SW=100*BPD; CAP=0.33; START=10000.0
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


def run(window):
    wall,ws,we=WIN[window]; closes={};opens={};H={}
    for s in _u100_symbols():
        r=_load_asset(s,"1d",wall)
        if r is None: continue
        d,o,h,l,c=r; closes[s]=pd.Series(np.asarray(c,float),index=d); opens[s]=pd.Series(np.asarray(o,float),index=d)
        H[s]=pd.Series(held_rolling(c,_cached_band_held("EMA","1d",s,o,h,l,c,wall),LB,SL),index=d)
    cdf=pd.DataFrame(closes).sort_index(); idx=cdf.index; cols=list(cdf.columns); T=len(idx); A=len(cols)
    odf=pd.DataFrame(opens).reindex(idx).reindex(columns=cols); C=cdf.to_numpy(); O=odf.to_numpy()
    Hm=pd.DataFrame(H).reindex(idx).fillna(0).reindex(columns=cols).fillna(0).to_numpy()
    vol=cdf.pct_change(fill_method=None).rolling(max(20,5*BPD),min_periods=10).std().to_numpy()
    roo=np.full((T,A),0.0); roo[:-1]=O[1:]/np.where(O[:-1]>0,O[:-1],np.nan)-1.0; roo=np.nan_to_num(roo)
    sall=cdf.rolling(SW,min_periods=SW//2).mean().to_numpy(); vc=np.isfinite(sall).sum(axis=1)
    brd=np.where(vc>0,np.nansum((C>sall)*np.isfinite(sall),axis=1)/np.maximum(vc,1),np.nan); g=hyst(brd,0.5,0.4)
    mult=Hm*(g[:,None]>0); iv=np.where((mult>0)&np.isfinite(vol)&(vol>0),1.0/np.where(vol>0,vol,np.nan),0.0)
    iv=np.nan_to_num(iv)*mult; sden=iv.sum(axis=1,keepdims=True); W=np.minimum(np.where(sden>0,iv/np.maximum(sden,1e-12),0.0),CAP)
    Wl=np.vstack([np.zeros((1,A)),W[:-1]]); net=np.nansum(Wl*roo,axis=1)-np.nansum(np.abs(W-Wl),axis=1)*(COST_PROD/2)
    mask=(idx>=pd.Timestamp(ws))&(idx<pd.Timestamp(we)); n=net[mask]; gross=Wl.sum(axis=1)[mask]; dts=idx[mask]
    weeks=[]
    for i in range(0,len(n)-SL+1,SL):
        ret=float(np.prod(1+n[i:i+SL])-1)*100; dep=bool(np.mean(gross[i:i+SL]>1e-9)>0.5)
        weeks.append({"week_start":str(pd.Timestamp(dts[i]).date()),"ret_pct":round(ret,2),"deployed":dep})
    return weeks


def summ(weeks):
    r=np.array([w["ret_pct"] for w in weeks]); dep=np.array([w["deployed"] for w in weeks]); eq=START*np.cumprod(1+r/100)
    eqf=START*np.cumprod(1+r/100); dd=float(np.min((eqf-np.maximum.accumulate(eqf))/np.maximum.accumulate(eqf))*100)
    rd=r[dep]
    return {"n_weeks":len(r),"deploy_frac":round(float(np.mean(dep)),2),"final_$":round(float(eq[-1]),0),
            "compound_pct":round(float(eq[-1]/START-1)*100,1),"mean_wk":round(float(np.mean(r)),2),"median_wk":round(float(np.median(r)),2),
            "median_wk_deployed":round(float(np.median(rd)),2) if len(rd) else 0,"pct_pos":round(float(np.mean(r>0)),2),
            "pct_in_2_7":round(float(np.mean((r>=2)&(r<=7))),2),"pct_gt7":round(float(np.mean(r>7)),2),"pct_neg":round(float(np.mean(r<0)),2),
            "best_wk":round(float(r.max()),2),"worst_wk":round(float(r.min()),2),"maxdd":round(dd,1)}


def main():
    out={"check":"EMA_weekly_VAL_OOS_2024","engine":"EMA|1d full band, breadth-gated invvol cap33, 14d/7d rolling, next-open, prod cost","ts":datetime.now().strftime("%Y%m%d_%H%M%S"),"windows":{}}
    for window in ["VAL-2024","OOS-2024"]:
        weeks=run(window); s=summ(weeks); out["windows"][window]={"summary":s,"weeks":weeks}
        tag="IN-SAMPLE (band selected on TRAIN+VAL)" if window=="VAL-2024" else "HELD-OUT (honest read)"
        print(f"\n{'='*78}\n{window}  ({WIN[window][1]}..{WIN[window][2]})  --  {tag}\n{'='*78}")
        print(f"  EMA|1d full band, breadth-gated invvol cap33, 14d/7d rolling, $10k start\n")
        print(f"  {'week_start':<12}{'ret%':>8}  state    equity$")
        eq=START
        for w in weeks:
            eq*= (1+w["ret_pct"]/100)
            bar=""
            if w["ret_pct"]>=7: bar="  <== >7% (right tail)"
            elif 2<=w["ret_pct"]<=7: bar="  <== in [2,7]"
            elif w["ret_pct"]<0: bar="  (loss)"
            print(f"  {w['week_start']:<12}{w['ret_pct']:>+7.2f}%  {'DEPLOY' if w['deployed'] else ' cash ':<7}{eq:>9,.0f}{bar}")
        print(f"\n  SUMMARY {window}: {s['n_weeks']} weeks, deploy {s['deploy_frac']*100:.0f}%  ->  ${s['final_$']:,.0f}  ({s['compound_pct']:+.0f}%), maxDD {s['maxdd']}%")
        print(f"    mean/wk {s['mean_wk']:+.2f}%  median/wk {s['median_wk']:+.2f}% (deployed {s['median_wk_deployed']:+.2f}%)  | %pos {s['pct_pos']*100:.0f}%  in[2,7] {s['pct_in_2_7']*100:.0f}%  >7% {s['pct_gt7']*100:.0f}%  neg {s['pct_neg']*100:.0f}%")
        print(f"    best wk {s['best_wk']:+.1f}%  worst wk {s['worst_wk']:+.1f}%")
    with open(HERE/"strat_ema_weekly.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\nSaved {HERE/'strat_ema_weekly.json'}")


if __name__=="__main__":
    main()
