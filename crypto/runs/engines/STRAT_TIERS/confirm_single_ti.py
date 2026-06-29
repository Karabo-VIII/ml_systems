"""STRAT TIERS -- CONFIRM the single-TI engines' per-week numbers (RWYB, user precondition before wiring).

User: "before wiring them, confirm their numbers and the average they produce per week." Run each SINGLE-TI engine
at 1d, EW long-only on u100, breadth-gated (de-leaked), PROD cost (honest; the V5 DNA used optimistic maker 0.0006),
on the FULL OOS (2024-05-15..2026-05). Report per-week BLENDED avg (all weeks incl cash) + DEPLOYED avg (weeks with
positions) + deploy-fraction (selectivity) + compound. Tests the user's hypothesis: static single-TI ~ 1-3%/wk non-selective.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd

HERE=Path(__file__).resolve().parent; ROOT=Path(__file__).resolve().parents[3]/"src"; sys.path.insert(0,str(ROOT))
import strat.fleet_lab as fl
from strat.ma_wealth_driver import _load_asset, _u100_symbols
import strat.ti_engine_sota as T
from strat.canonical_exits import COST_PROD
BPD=fl.BARS_PER_DAY["1d"]; SW=100*BPD; WALL=int(pd.Timestamp("2026-07-01").value//10**6); CUT=pd.Timestamp("2024-05-15")


def hyst(s,u,d):
    n=len(s); o=np.zeros(n,int); st=0
    for i in range(n):
        if not np.isfinite(s[i]): o[i]=st; continue
        if st==0 and s[i]>u: st=1
        elif st==1 and s[i]<d: st=0
        o[i]=st
    return o


def main():
    O={};H={};L={};C={}
    for s in _u100_symbols():
        r=_load_asset(s,"1d",WALL)
        if r is None: continue
        d,o,h,l,c=r; C[s]=pd.Series(np.asarray(c,float),index=d);H[s]=pd.Series(np.asarray(h,float),index=d);L[s]=pd.Series(np.asarray(l,float),index=d);O[s]=pd.Series(np.asarray(o,float),index=d)
    cdf=pd.DataFrame(C).sort_index(); idx=cdf.index; cols=list(cdf.columns); T_=len(idx); A=len(cols)
    hdf=pd.DataFrame(H).reindex(idx).reindex(columns=cols); ldf=pd.DataFrame(L).reindex(idx).reindex(columns=cols); odf=pd.DataFrame(O).reindex(idx).reindex(columns=cols)
    Cn=cdf.to_numpy();Hn=hdf.to_numpy();Ln=ldf.to_numpy();On=odf.to_numpy()
    roo=np.full((T_,A),0.0); roo[:-1]=On[1:]/np.where(On[:-1]>0,On[:-1],np.nan)-1.0; roo=np.nan_to_num(roo)
    sall=cdf.rolling(SW,min_periods=SW//2).mean().to_numpy(); vc=np.isfinite(sall).sum(axis=1)
    brd=np.where(vc>0,np.nansum((Cn>sall)*np.isfinite(sall),axis=1)/np.maximum(vc,1),np.nan); gate=hyst(brd,0.5,0.4)
    # per-TI held matrix builders (1 config each, canonical)
    def held_mat(fn):
        M=np.zeros((T_,A))
        for j,s in enumerate(cols):
            try: M[:,j]=fn(Cn[:,j],Hn[:,j],Ln[:,j])
            except TypeError: M[:,j]=fn(Cn[:,j])
        return M
    engines={
        "RSI(7,55,40)": lambda c: T.rsi_single_held(c,7,55,40),
        "BOLLINGER(15,1.5,bo)": lambda c: T.bollinger_single_held(c,15,1.5,"breakout"),
        "ADX(7,0.3)": lambda c: T.adx_single_held(c,7,0.3),
        "MACD(12,26,9)": lambda c: T.macd_single_held(c,12,26,9),
        "SUPERTREND(7,1.5)": lambda c,h,l: T.supertrend_single_held(c,h,l,7,1.5),
        "STOCH(14)": lambda c,h,l: T.stochastic_single_held(c,h,l,14,3,80,20) if True else None,
    }
    mask=(idx>=CUT)
    out={"check":"CONFIRM_single_TI","cost":COST_PROD,"window":"OOS 2024-05-15..2026-05","ts":datetime.now().strftime("%Y%m%d_%H%M%S"),"engines":{}}
    print("="*96)
    print("CONFIRM single-TI engines @1d, EW long-only u100, breadth-gated, PROD cost, OOS 2024-05..2026-05")
    print("="*96)
    print(f"  {'engine':<22}{'blend_wk':>9}{'deploy_wk':>10}{'deploy_frac':>12}{'compound':>10}{'maxDD':>7}")
    print("  "+"-"*78)
    for nm,fn in engines.items():
        try:
            Hd=held_mat(fn)
        except Exception as e:
            print(f"  {nm:<22} ERR {str(e)[:40]}"); continue
        Hd=Hd*(gate[:,None]>0)
        Hl=np.vstack([np.zeros((1,A)),Hd[:-1]])
        nheld=Hl.sum(axis=1); gross=np.where(nheld>0,(roo*Hl).sum(axis=1)/np.maximum(nheld,1),0.0)
        turn=np.abs(Hd-np.vstack([np.zeros((1,A)),Hd[:-1]])).sum(axis=1); cost=np.where(nheld>0,turn*(COST_PROD/2)/np.maximum(nheld,1),0.0)
        net=gross-cost; nm_=net[mask]
        sb=7*BPD; wk=[]; dep=[]; i=0; gd=(Hl.sum(axis=1)>0)[mask]
        while i+sb<=len(nm_):
            wk.append(float(np.prod(1+nm_[i:i+sb])-1)*100); dep.append(np.mean(gd[i:i+sb]>0)>0.5); i+=sb
        wk=np.array(wk); dep=np.array(dep)
        eq=np.cumprod(1+nm_); ddv=float(np.min((eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq)))*100
        blend=float(np.mean(wk)); depwk=float(np.mean(wk[dep])) if dep.any() else 0; depf=float(np.mean(gd))
        out["engines"][nm]={"blend_wk":round(blend,2),"deploy_wk":round(depwk,2),"deploy_frac":round(depf,3),"compound":round(float(eq[-1]-1)*100,1),"maxdd":round(ddv,1)}
        print(f"  {nm:<22}{blend:>+8.2f}%{depwk:>+9.2f}%{depf*100:>11.0f}%{float(eq[-1]-1)*100:>+9.0f}%{ddv:>+7.0f}")
    with open(HERE/"confirm_single_ti.json","w") as f: json.dump(out,f,indent=2,default=str)
    print(f"\nSaved {HERE/'confirm_single_ti.json'}")


if __name__=="__main__":
    main()
