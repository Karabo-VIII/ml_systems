"""QUANT: the ORACLE GAP per 7d slice -- 'in u100, something always rips; why can't we trade any slice?'

Decompose the per-slice opportunity into CAPTURABLE vs NOT. Per 7d slice (full OOS, 14d lookback informs selection):
  ORACLE_best1 : the single best u100 name's realized 7d return (proof: is there ALWAYS a mover?)
  ORACLE_top8  : perfect hindsight top-8 by realized 7d return, inv-vol (the AVAILABLE ceiling)
  MOM_top8     : CAUSAL top-8 by past-14d momentum, their realized 7d (best causal SELECTION)
  MA_strat     : SMA breadth-gated inv-vol (what we deploy -- cash when gate shut)
  SELF_trend   : per-asset trend, NO market gate (trade any name trending -- 'trade any slice')
  MARKET_ew    : equal-weight u100 (the beta)
  RANDOM_top8  : 8 random names (null)
GAP DECOMPOSITION: ORACLE_top8 - MOM_top8 = SELECTION ceiling (un-capturable if price can't rank names);
                   MOM_top8 - MA_strat = PARTICIPATION/method gap (we sit out; momentum trades).
Report per-source: mean/slice %, %slices positive, compound. + the honest verdict on 'trade any slice'.
1d u100 PIT, next-open realized (roo), prod cost on the traded books. non-overlapping 7d slices.
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
BPD=fl.BARS_PER_DAY["1d"]; LB=14*BPD; SL=7*BPD; SW=100*BPD; CAP=0.33; K=8
WALL=int(pd.Timestamp("2026-07-01").value//10**6); CUT=pd.Timestamp("2024-05-15")


def hyst(s,u,d):
    n=len(s);o=np.zeros(n,int);st=0
    for i in range(n):
        if not np.isfinite(s[i]): o[i]=st; continue
        if st==0 and s[i]>u: st=1
        elif st==1 and s[i]<d: st=0
        o[i]=st
    return o


def main():
    closes={};opens={};Hs={}
    for s in _u100_symbols():
        r=_load_asset(s,"1d",WALL)
        if r is None: continue
        d,o,h,l,c=r; cc=np.asarray(c,float); closes[s]=pd.Series(cc,index=d); opens[s]=pd.Series(np.asarray(o,float),index=d)
        Hs[s]=pd.Series(held_rolling(c,_cached_band_held("SMA","1d",s,o,h,l,c,WALL),LB,SL),index=d)
    cdf=pd.DataFrame(closes).sort_index(); idx=cdf.index; cols=list(cdf.columns); T=len(idx); A=len(cols)
    odf=pd.DataFrame(opens).reindex(idx).reindex(columns=cols); C=cdf.to_numpy(); O=odf.to_numpy()
    Hm=pd.DataFrame(Hs).reindex(idx).fillna(0).reindex(columns=cols).fillna(0).to_numpy()
    vol=cdf.pct_change(fill_method=None).rolling(max(20,5*BPD),min_periods=10).std().to_numpy()
    roo=np.full((T,A),0.0); roo[:-1]=O[1:]/np.where(O[:-1]>0,O[:-1],np.nan)-1.0; roo=np.nan_to_num(roo)
    sall=cdf.rolling(SW,min_periods=SW//2).mean().to_numpy(); vc=np.isfinite(sall).sum(axis=1)
    brd=np.where(vc>0,np.nansum((C>sall)*np.isfinite(sall),axis=1)/np.maximum(vc,1),np.nan); gbin=hyst(brd,0.5,0.4)
    mom=cdf.pct_change(14*BPD).to_numpy(); own=cdf.rolling(SW,min_periods=SW//2).mean().to_numpy()
    m=(idx>=CUT); mi=np.where(m)[0]; starts=[i for i in range(mi[0],mi[-1]-SL+1,SL)]
    rng=np.random.default_rng(0)
    def slice_ret_names(i, sel):  # inv-vol EW over selected names, realized 7d (compounded roo)
        if len(sel)==0: return 0.0
        v=vol[i][sel]; w=np.where(np.isfinite(v)&(v>0),1.0/v,0.0); w=w/w.sum() if w.sum()>0 else np.ones(len(sel))/len(sel)
        r7=np.prod(1+roo[i:i+SL][:,sel],axis=0)-1.0
        return float(np.nansum(w*r7))*100
    rows={"ORACLE_best1":[],"ORACLE_top8":[],"MOM_top8":[],"MA_strat":[],"SELF_trend":[],"MARKET_ew":[],"RANDOM_top8":[]}
    for i in starts:
        valid=np.where(np.isfinite(C[i])&np.isfinite(roo[i:i+SL]).all(axis=0))[0]
        if len(valid)==0:
            for k in rows: rows[k].append(0.0); continue
        r7=np.prod(1+roo[i:i+SL][:,valid],axis=0)-1.0                     # realized 7d per valid name
        # ORACLE
        ob=valid[np.argsort(-r7)]; rows["ORACLE_best1"].append(float(r7.max())*100)
        rows["ORACLE_top8"].append(slice_ret_names(i, ob[:K]))
        # MOM causal
        mv=mom[i][valid]; msel=valid[np.argsort(-np.where(np.isfinite(mv),mv,-1e9))[:K]]
        rows["MOM_top8"].append(slice_ret_names(i, msel))
        # MA strat (breadth-gated held, inv-vol) -- only the held names when gate open
        held=valid[(Hm[i][valid]>0)] if gbin[i]>0 else np.array([],int)
        rows["MA_strat"].append(slice_ret_names(i, held[:200]))
        # SELF: names above own 100d-MA (per-asset trend, no market gate)
        selfn=valid[(C[i][valid]>own[i][valid])&np.isfinite(own[i][valid])]
        rows["SELF_trend"].append(slice_ret_names(i, selfn[:200]))
        # market EW + random
        rows["MARKET_ew"].append(float(np.mean(r7))*100)
        rows["RANDOM_top8"].append(slice_ret_names(i, rng.choice(valid,size=min(K,len(valid)),replace=False)))
    def agg(x):
        x=np.array(x); comp=float(np.prod(1+x/100)-1)*100
        return {"mean_slice":round(float(np.mean(x)),2),"pct_pos":round(float(np.mean(x>0)),2),"compound":round(comp,1),"median":round(float(np.median(x)),2)}
    out={"check":"oracle_gap","n_slices":len(starts),"ts":datetime.now().strftime("%Y%m%d_%H%M%S"),"sources":{k:agg(v) for k,v in rows.items()}}
    print("="*96); print(f"ORACLE GAP per 7d slice ({len(starts)} slices, OOS 2024-05..2026-05, u100, next-open realized)"); print("="*96)
    print(f"\n  {'source':<14}{'mean/slice':>12}{'median':>9}{'%slices+':>10}{'compound':>11}   role")
    roles={"ORACLE_best1":"single best name (proof: always a mover)","ORACLE_top8":"PERFECT top-8 selection = the CEILING",
           "MOM_top8":"causal momentum top-8 = best CAUSAL selection","MA_strat":"what we deploy (cash when gate shut)",
           "SELF_trend":"per-asset trend, no gate = 'trade any slice'","MARKET_ew":"equal-weight u100 (beta)","RANDOM_top8":"random 8 (null)"}
    for k in ["ORACLE_best1","ORACLE_top8","MOM_top8","SELF_trend","MA_strat","MARKET_ew","RANDOM_top8"]:
        a=out["sources"][k]; print(f"  {k:<14}{a['mean_slice']:>+11.2f}%{a['median']:>+8.2f}%{a['pct_pos']*100:>9.0f}%{a['compound']:>+10.0f}%   {roles[k]}")
    o8=out["sources"]["ORACLE_top8"]["mean_slice"]; m8=out["sources"]["MOM_top8"]["mean_slice"]; ma=out["sources"]["MA_strat"]["mean_slice"]
    out["gap"]={"selection_gap_oracle_minus_mom":round(o8-m8,2),"participation_gap_mom_minus_ma":round(m8-ma,2)}
    print(f"\n  GAP DECOMPOSITION (mean/slice):")
    print(f"    SELECTION ceiling (ORACLE_top8 {o8:+.2f} - MOM_top8 {m8:+.2f}) = {o8-m8:+.2f}%/slice  -> UN-capturable if price can't rank names (prior: within-bar AUC 0.548)")
    print(f"    PARTICIPATION gap (MOM_top8 {m8:+.2f} - MA_strat {ma:+.2f}) = {m8-ma:+.2f}%/slice  -> capturable by trading more / momentum-selecting (SELF/PARTIAL do it, at DD cost)")
    print(f"\n  READ: 'trade any slice' is TRUE for the ORACLE (best1 +{out['sources']['ORACLE_best1']['mean_slice']:.0f}%/slice, {out['sources']['ORACLE_best1']['pct_pos']*100:.0f}% positive).")
    print(f"        The question is which gap is real: causal MOMENTUM top-8 gets {m8:+.2f}%/slice ({out['sources']['MOM_top8']['pct_pos']*100:.0f}% pos) vs our MA {ma:+.2f}%.")
    with open(HERE/"strat_oracle_gap.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\nSaved {HERE/'strat_oracle_gap.json'}")


if __name__=="__main__":
    main()
