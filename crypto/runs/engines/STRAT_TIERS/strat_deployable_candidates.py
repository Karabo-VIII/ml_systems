"""ORC autonomous: DEPLOYABLE CANDIDATES -- single MA engine x the full regime-gate design space, rigorously graded.

Builds 'all that remains' from the quant arc: one SMA engine (all 8 types == 1; SMA simplest/cheapest), full band,
14d-lookback/7d-slice rolling, inv-vol cap33, next-open, prod cost -- under each gate design:
  NONE           : no gate (max participation)
  BIN_breadth    : binary market breadth gate (OLD/current)
  CONT_breadth   : CONTINUOUS exposure proportional to breadth (the un-built lever)
  BIN_x_ER       : breadth AND market trend-efficiency filter (deploy only clean-trend weeks)
  CONT_x_ER      : continuous breadth x efficiency
  SELF           : per-asset own-regime gate
Graded per candidate on OOS-2024 (bull) / OOS-2025 (bear) / OOS-ALL: compound, maxDD, deploy, wk_mean, %pos,
NW(L3) p, block-boot p05, deployed-win, Calmar, + vs BTC buy-hold (compound, maxDD, Calmar). DEPLOYABLE bar:
positive OOS-ALL, maxDD<35%, beats BTC-BH Calmar, OOS-2025 loss minimized. Honest: long-only can't be +in a bear;
'profitable' = positive full-cycle + beats BH risk-adjusted. 1d u100 point-in-time (wall=latest; survivorship noted).
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
BPD=fl.BARS_PER_DAY["1d"]; LB=14*BPD; SL=7*BPD; SW=100*BPD; CAP=0.33; ANN=np.sqrt(365)
WALL=int(pd.Timestamp("2026-07-01").value//10**6); CUT=pd.Timestamp("2024-05-15"); OOS25=pd.Timestamp("2025-01-01")


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
    brd=np.where(vc>0,np.nansum((C>sall)*np.isfinite(sall),axis=1)/np.maximum(vc,1),np.nan)
    gbin=hyst(brd,0.5,0.4).astype(float)
    gcont=np.clip((np.nan_to_num(brd)-0.40)/0.20,0,1)          # continuous: 0 @brd<=0.40 -> 1 @brd>=0.60
    bc=cdf["BTCUSDT"].to_numpy() if "BTCUSDT" in cols else C[:,0]
    mv=np.abs(np.diff(bc,prepend=bc[0])); er=np.abs(bc-pd.Series(bc).shift(14).to_numpy())/np.maximum(pd.Series(mv).rolling(14,min_periods=7).sum().to_numpy(),1e-9)
    erbin=(er>0.35).astype(float); ersc=np.clip((np.nan_to_num(er)-0.20)/0.30,0,1)
    own=cdf.rolling(SW,min_periods=SW//2).mean().to_numpy(); gself=((C>own)&np.isfinite(own)).astype(float)
    def b1(x): return np.broadcast_to(x[:,None],(T,A)).astype(float)
    gates={"NONE":np.ones((T,A)),"BIN_breadth":b1(gbin),"CONT_breadth":b1(gcont),
           "BIN_x_ER":b1(gbin*erbin),"CONT_x_ER":b1(gcont*ersc),"SELF":gself,"CONT_x_SELF":b1(gcont)*gself}
    def book(gt):
        mult=Hm*gt; iv=np.where((mult>0)&np.isfinite(vol)&(vol>0),1.0/np.where(vol>0,vol,np.nan),0.0)
        iv=np.nan_to_num(iv)*mult; sden=iv.sum(axis=1,keepdims=True); W=np.minimum(np.where(sden>0,iv/np.maximum(sden,1e-12),0.0),CAP)
        Wl=np.vstack([np.zeros((1,A)),W[:-1]]); return np.nansum(Wl*roo,axis=1)-np.nansum(np.abs(W-Wl),axis=1)*(COST_PROD/2), Wl
    # BTC-BH per window
    btcr=np.zeros(T); bo=odf["BTCUSDT"].to_numpy() if "BTCUSDT" in cols else O[:,0]
    btcr[:-1]=bo[1:]/np.where(bo[:-1]>0,bo[:-1],np.nan)-1.0; btcr=np.nan_to_num(btcr)
    def stats(net,Wl,wm):
        n=net[wm]; gw=Wl.sum(axis=1)[wm]; eq=np.cumprod(1+n)
        dd=float(np.min((eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq))*100) if len(eq)>1 else 0
        starts=list(range(0,len(n)-SL+1,SL)); wk=np.array([float(np.prod(1+n[i:i+SL])-1)*100 for i in starts])
        dep=np.array([np.mean(gw[i:i+SL]>1e-9)>0.5 for i in starts]); comp=float(np.prod(1+n)-1)*100
        sh=float(np.mean(n)/np.std(n)*ANN) if np.std(n)>1e-12 else 0; cal=float(comp/abs(dd)) if abs(dd)>1e-9 else 0
        dw=wk[dep]
        return {"comp":round(comp,1),"maxdd":round(dd,1),"deploy":round(float(np.mean(Wl.sum(axis=1)[wm]>1e-9)),2),
                "wk_mean":round(float(np.mean(wk)),2),"pct_pos":round(float(np.mean(wk>0)),2),"sharpe":round(sh,2),"calmar":round(cal,2),
                "dep_win":round(float(np.mean(dw>0)) if len(dw) else 0,2),"nw_p":round(nw_p(wk),3),"p05":bootp05(wk)}
    def btcstat(wm):
        b=btcr[wm]; eq=np.cumprod(1+b); comp=float(eq[-1]-1)*100
        dd=float(np.min((eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq))*100); cal=float(comp/abs(dd)) if abs(dd)>1e-9 else 0
        return {"comp":round(comp,1),"maxdd":round(dd,1),"calmar":round(cal,2)}
    WINS={"OOS-2024":(idx>=CUT)&(idx<OOS25),"OOS-2025":(idx>=OOS25),"OOS-ALL":(idx>=CUT)}
    out={"check":"deployable_candidates","engine":"SMA full band, 14d/7d, invvol cap33, next-open, prod","ts":datetime.now().strftime("%Y%m%d_%H%M%S"),"btc_bh":{w:btcstat(m) for w,m in WINS.items()},"candidates":{}}
    nets={nm:book(gt) for nm,gt in gates.items()}
    print("="*128); print("DEPLOYABLE CANDIDATES -- single SMA engine x gate design (graded OOS-2024/2025/ALL vs BTC-BH)"); print("="*128)
    for w,m in WINS.items():
        b=out["btc_bh"][w]; print(f"\n  {w}   [BTC-BH: comp {b['comp']:+.0f}% maxDD {b['maxdd']:+.0f}% Calmar {b['calmar']}]")
        print(f"    {'gate':<14}{'comp%':>7}{'maxDD':>7}{'deploy':>7}{'wk_mean':>8}{'%pos':>6}{'Sharpe':>7}{'Calmar':>7}{'depWin':>7}{'NW_p':>6}{'p05':>7}")
        for nm in gates:
            s=stats(*nets[nm],m); out["candidates"].setdefault(nm,{})[w]=s
            print(f"    {nm:<14}{s['comp']:>+6.0f}%{s['maxdd']:>+7.0f}{s['deploy']*100:>6.0f}%{s['wk_mean']:>+7.2f}%{s['pct_pos']*100:>5.0f}%{s['sharpe']:>7.2f}{s['calmar']:>7.2f}{s['dep_win']*100:>6.0f}%{s['nw_p']:>6.2f}{(s['p05'] if s['p05'] is not None else 0):>+6.0f}%")
    # rank: deployable = OOS-ALL comp>0 & maxDD>-35 & Calmar>BTC-ALL Calmar
    bcal=out["btc_bh"]["OOS-ALL"]["calmar"]
    print(f"\n  DEPLOYABLE SCREEN (OOS-ALL comp>0, maxDD>-35%, Calmar>BTC {bcal}):")
    ranked=[]
    for nm in gates:
        a=out["candidates"][nm]["OOS-ALL"]; ok=(a["comp"]>0 and a["maxdd"]>-35 and a["calmar"]>bcal)
        ranked.append((nm,a["comp"],a["maxdd"],a["calmar"],a["p05"],ok,out["candidates"][nm]["OOS-2025"]["comp"]))
    ranked.sort(key=lambda x:-x[3])
    for nm,c,dd,cal,p05,ok,c25 in ranked:
        print(f"    {'PASS' if ok else '    '} {nm:<14} OOS-ALL {c:+.0f}% maxDD {dd:+.0f}% Calmar {cal} p05 {p05}% | OOS-2025 {c25:+.0f}%")
    out["btc_all_calmar"]=bcal
    with open(HERE/"strat_deployable_candidates.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\nSaved {HERE/'strat_deployable_candidates.json'}")


if __name__=="__main__":
    main()
