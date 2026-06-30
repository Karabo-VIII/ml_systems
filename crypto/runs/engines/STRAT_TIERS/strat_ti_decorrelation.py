"""ORC autonomous: is there a genuinely DE-CORRELATED second engine? -- RSI/Bollinger/ADX vs the MA trend factor.

The MA family is ONE factor (n_eff~1). A real second DEPLOYABLE engine needs a non-MA TI whose WEEKLY P&L de-correlates
from MA (corr < ~0.6) AND is positive standalone -> genuine diversification (the path to a better combined book).
Same structure for all: per-asset TI held -> held_rolling(14d/7d) -> breadth-gated inv-vol cap33 -> weekly net.
Representative configs (1d): SMA(band), RSI(14, long>55, exit<45), Bollinger(20,2.0,breakout), ADX(14, thr0.25).
Report: per-TI standalone comp/maxDD/Calmar (OOS-2024/2025/ALL) + weekly-P&L correlation matrix vs SMA. 1d u100 PIT.
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
from strat.ti_engine_sota import rsi_single_held, bollinger_single_held, adx_single_held
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
    closes={};opens={}; held={"SMA":{},"RSI":{},"BOLL":{},"ADX":{}}
    for s in _u100_symbols():
        r=_load_asset(s,"1d",WALL)
        if r is None: continue
        d,o,h,l,c=r; cc=np.asarray(c,float); closes[s]=pd.Series(cc,index=d); opens[s]=pd.Series(np.asarray(o,float),index=d)
        held["SMA"][s]=pd.Series(held_rolling(c,_cached_band_held("SMA","1d",s,o,h,l,c,WALL),LB,SL),index=d)
        held["RSI"][s]=pd.Series(held_rolling(cc,rsi_single_held(cc,14,55.0,45.0),LB,SL),index=d)
        held["BOLL"][s]=pd.Series(held_rolling(cc,bollinger_single_held(cc,20,2.0,"breakout"),LB,SL),index=d)
        held["ADX"][s]=pd.Series(held_rolling(cc,adx_single_held(cc,14,0.25),LB,SL),index=d)
    cdf=pd.DataFrame(closes).sort_index(); idx=cdf.index; cols=list(cdf.columns); T=len(idx); A=len(cols)
    odf=pd.DataFrame(opens).reindex(idx).reindex(columns=cols); C=cdf.to_numpy(); O=odf.to_numpy()
    vol=cdf.pct_change(fill_method=None).rolling(max(20,5*BPD),min_periods=10).std().to_numpy()
    roo=np.full((T,A),0.0); roo[:-1]=O[1:]/np.where(O[:-1]>0,O[:-1],np.nan)-1.0; roo=np.nan_to_num(roo)
    sall=cdf.rolling(SW,min_periods=SW//2).mean().to_numpy(); vc=np.isfinite(sall).sum(axis=1)
    brd=np.where(vc>0,np.nansum((C>sall)*np.isfinite(sall),axis=1)/np.maximum(vc,1),np.nan); g=hyst(brd,0.5,0.4).astype(float)
    def book(Hd):
        Hm=pd.DataFrame(Hd).reindex(idx).fillna(0).reindex(columns=cols).fillna(0).to_numpy()
        mult=Hm*(g[:,None]>0); iv=np.where((mult>0)&np.isfinite(vol)&(vol>0),1.0/np.where(vol>0,vol,np.nan),0.0)
        iv=np.nan_to_num(iv)*mult; sden=iv.sum(axis=1,keepdims=True); W=np.minimum(np.where(sden>0,iv/np.maximum(sden,1e-12),0.0),CAP)
        Wl=np.vstack([np.zeros((1,A)),W[:-1]]); return np.nansum(Wl*roo,axis=1)-np.nansum(np.abs(W-Wl),axis=1)*(COST_PROD/2)
    nets={k:book(held[k]) for k in held}
    def stat(net,m):
        n=net[m]; eq=np.cumprod(1+n); comp=float(eq[-1]-1)*100; dd=float(np.min((eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq))*100) if len(eq)>1 else 0
        return round(comp,1),round(dd,1),round(comp/abs(dd),2) if abs(dd)>1e-9 else 0
    WINS={"OOS-2024":(idx>=CUT)&(idx<OOS25),"OOS-2025":(idx>=OOS25),"OOS-ALL":(idx>=CUT)}
    out={"check":"ti_decorrelation","ts":datetime.now().strftime("%Y%m%d_%H%M%S"),"standalone":{},"corr_vs_SMA":{}}
    print("="*92); print("SECOND-ENGINE SEARCH: non-MA TI standalone + weekly-P&L correlation vs the MA factor (SMA)"); print("="*92)
    for w,m in WINS.items():
        print(f"\n  {w}:  {'TI':<7}{'comp%':>8}{'maxDD':>8}{'Calmar':>8}")
        for k in held:
            c,dd,cal=stat(nets[k],m); out["standalone"].setdefault(k,{})[w]={"comp":c,"maxdd":dd,"calmar":cal}
            print(f"    {'':7}{k:<7}{c:>+7.0f}%{dd:>+8.0f}{cal:>8.2f}")
    # weekly P&L correlation (OOS-ALL) vs SMA
    m=WINS["OOS-ALL"]; starts=list(range(0,int(m.sum())-SL+1,SL))
    wks={k:np.array([float(np.prod(1+nets[k][m][i:i+SL])-1) for i in starts]) for k in held}
    print(f"\n  WEEKLY-P&L CORRELATION vs SMA (OOS-ALL) -- the diversification test (<0.6 = genuine 2nd engine):")
    for k in held:
        if k=="SMA": continue
        cr=float(np.corrcoef(wks["SMA"],wks[k])[0,1]) if np.std(wks[k])>0 else 1.0
        out["corr_vs_SMA"][k]=round(cr,2)
        sa=out["standalone"][k]["OOS-ALL"]
        verdict="GENUINE DIVERSIFIER" if (cr<0.6 and sa["comp"]>0) else ("redundant (corr high)" if cr>=0.6 else "decorrelated but unprofitable")
        print(f"    SMA vs {k:<6} corr {cr:+.2f}  | {k} standalone OOS-ALL {sa['comp']:+.0f}% Calmar {sa['calmar']}  -> {verdict}")
    with open(HERE/"strat_ti_decorrelation.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\nSaved {HERE/'strat_ti_decorrelation.json'}")


if __name__=="__main__":
    main()
