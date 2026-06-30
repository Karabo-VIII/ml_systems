"""ORC autonomous: verify the 2 deployable candidates -- ER-threshold robustness + live weekly books.

(1) Is BIN_x_ER's bear-protection (-2% OOS-2025, -21% maxDD) a LUCKY ER threshold (0.35) or robust across thresholds?
    Sweep ER thr in {0.25,0.30,0.35,0.40,0.45}; if OOS-2025 stays ~flat + maxDD stays low across thr -> robust.
(2) LIVE weekly book (current decision) for BIN_breadth + BIN_x_ER: gate state now + held names + weights.
Single SMA engine, 14d/7d, invvol cap33, next-open, prod. point-in-time wall=latest.
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
BPD=fl.BARS_PER_DAY["1d"]; LB=14*BPD; SL=7*BPD; SW=100*BPD; CAP=0.33
WALL=int(pd.Timestamp("2026-07-01").value//10**6); CUT=pd.Timestamp("2024-05-15"); OOS25=pd.Timestamp("2025-01-01")


def hyst(s,u,d):
    n=len(s);o=np.zeros(n,int);st=0
    for i in range(n):
        if not np.isfinite(s[i]): o[i]=st; continue
        if st==0 and s[i]>u: st=1
        elif st==1 and s[i]<d: st=0
        o[i]=st
    return o


def main():
    closes={};opens={};H={}
    for s in _u100_symbols():
        r=_load_asset(s,"1d",WALL)
        if r is None: continue
        d,o,h,l,c=r; closes[s]=pd.Series(np.asarray(c,float),index=d); opens[s]=pd.Series(np.asarray(o,float),index=d)
        H[s]=pd.Series(held_rolling(c,_cached_band_held("SMA","1d",s,o,h,l,c,WALL),LB,SL),index=d)
    cdf=pd.DataFrame(closes).sort_index(); idx=cdf.index; cols=list(cdf.columns); T=len(idx); A=len(cols)
    odf=pd.DataFrame(opens).reindex(idx).reindex(columns=cols); C=cdf.to_numpy(); O=odf.to_numpy()
    Hm=pd.DataFrame(H).reindex(idx).fillna(0).reindex(columns=cols).fillna(0).to_numpy()
    vol=cdf.pct_change(fill_method=None).rolling(max(20,5*BPD),min_periods=10).std().to_numpy()
    roo=np.full((T,A),0.0); roo[:-1]=O[1:]/np.where(O[:-1]>0,O[:-1],np.nan)-1.0; roo=np.nan_to_num(roo)
    sall=cdf.rolling(SW,min_periods=SW//2).mean().to_numpy(); vc=np.isfinite(sall).sum(axis=1)
    brd=np.where(vc>0,np.nansum((C>sall)*np.isfinite(sall),axis=1)/np.maximum(vc,1),np.nan); gbin=hyst(brd,0.5,0.4).astype(float)
    bc=cdf["BTCUSDT"].to_numpy(); mv=np.abs(np.diff(bc,prepend=bc[0])); er=np.abs(bc-pd.Series(bc).shift(14).to_numpy())/np.maximum(pd.Series(mv).rolling(14,min_periods=7).sum().to_numpy(),1e-9)
    def b1(x): return np.broadcast_to(x[:,None],(T,A)).astype(float)
    def book(gt):
        mult=Hm*gt; iv=np.where((mult>0)&np.isfinite(vol)&(vol>0),1.0/np.where(vol>0,vol,np.nan),0.0)
        iv=np.nan_to_num(iv)*mult; sden=iv.sum(axis=1,keepdims=True); W=np.minimum(np.where(sden>0,iv/np.maximum(sden,1e-12),0.0),CAP)
        Wl=np.vstack([np.zeros((1,A)),W[:-1]]); return np.nansum(Wl*roo,axis=1)-np.nansum(np.abs(W-Wl),axis=1)*(COST_PROD/2), W
    def wm(s): return {"OOS-2024":(idx>=CUT)&(idx<OOS25),"OOS-2025":(idx>=OOS25),"OOS-ALL":(idx>=CUT)}[s]
    def cm(net,m):
        n=net[m]; eq=np.cumprod(1+n); comp=float(eq[-1]-1)*100; dd=float(np.min((eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq))*100)
        return round(comp,1),round(dd,1)
    out={"check":"candidates_verify","ts":datetime.now().strftime("%Y%m%d_%H%M%S")}
    print("="*92); print("(1) ER-THRESHOLD ROBUSTNESS of BIN_x_ER (bear-protection a lucky threshold?)"); print("="*92)
    print(f"  {'ER_thr':>7}{'OOS24_comp':>12}{'OOS25_comp':>12}{'OOSALL_comp':>13}{'OOSALL_maxDD':>14}{'deploy':>8}")
    ersweep={}
    for thr in [0.25,0.30,0.35,0.40,0.45]:
        net,W=book(b1(gbin*(er>thr).astype(float)))
        c24,_=cm(net,wm("OOS-2024")); c25,_=cm(net,wm("OOS-2025")); call,ddall=cm(net,wm("OOS-ALL"))
        dep=round(float(np.mean(W.sum(axis=1)[wm('OOS-ALL')]>1e-9)),2); ersweep[thr]={"oos24":c24,"oos25":c25,"oosall":call,"maxdd":ddall,"deploy":dep}
        print(f"  {thr:>7.2f}{c24:>+11.0f}%{c25:>+11.0f}%{call:>+12.0f}%{ddall:>+13.0f}%{dep*100:>7.0f}%")
    out["er_threshold_sweep"]=ersweep
    bearvals=[ersweep[t]["oos25"] for t in ersweep]; ddvals=[ersweep[t]["maxdd"] for t in ersweep]
    print(f"  -> OOS-2025 across thr: [{min(bearvals):+.0f}, {max(bearvals):+.0f}]%  maxDD [{min(ddvals):+.0f},{max(ddvals):+.0f}]%  {'ROBUST (bear ~flat all thr)' if max(bearvals)<5 and min(bearvals)>-12 else 'threshold-sensitive'}")
    # ---- (2) LIVE books ----
    print(f"\n{'='*92}\n(2) LIVE WEEKLY BOOK (as of {pd.Timestamp(idx[-1]).date()})\n{'='*92}")
    last=T-1; brd_now=float(brd[last]); er_now=float(er[last]); gbin_now=int(gbin[last])
    out["live"]={"as_of":str(pd.Timestamp(idx[-1]).date()),"breadth":round(brd_now,3),"breadth_gate_open":bool(gbin_now),"market_ER":round(er_now,3),"ER_gate_open(thr0.35)":bool(er_now>0.35)}
    print(f"  breadth={brd_now:.2f} (gate {'OPEN' if gbin_now else 'CLOSED'})  market_ER={er_now:.2f} (ER-filter {'OPEN' if er_now>0.35 else 'CLOSED'})")
    for nm,gt in [("BIN_breadth",b1(gbin)),("BIN_x_ER",b1(gbin*(er>0.35).astype(float)))]:
        _,W=book(gt); livew=W[last]; live=sorted(((cols[j],round(float(livew[j]),3)) for j in np.where(livew>1e-6)[0]),key=lambda x:-x[1])
        out["live"][nm]={"decision":"DEPLOY" if live else "CASH","n_names":len(live),"book":live[:25]}
        print(f"\n  {nm}: {'DEPLOY '+str(len(live))+' names' if live else 'CASH'}")
        if live:
            for nmn,wt in live[:12]: print(f"      {nmn:<14}{wt:6.1%}")
            if len(live)>12: print(f"      ... +{len(live)-12} more")
    with open(HERE/"strat_candidates_verify.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\nSaved {HERE/'strat_candidates_verify.json'}")


if __name__=="__main__":
    main()
