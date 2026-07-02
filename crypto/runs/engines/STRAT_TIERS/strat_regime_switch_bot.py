"""WORKING-BOT BUILD 2: regime-correct SWITCH -- ride trends in BULL, buy capitulation in BEAR, CASH in CHOP.

Per-regime attribution (strat_range_bot) revealed: TREND +24pp in BULL / -24 CHOP / -7 BEAR; RANGE(MR) -21 BULL /
-38 CHOP / +36 BEAR. So the MR bot is an OVERSOLD-BOUNCE bot (wins bear), not a chop bot; and BOTH lose chop. The
canonical prior (NOT a curve-fit): trend-follow in trends, buy capitulation in crashes, sit out chop.
  SWITCH_RC:  BTC BULL(>SMA50 & ER>0.35) -> TREND book | BEAR(<SMA50 & ER>0.35) -> RANGE/bounce book | CHOP -> CASH.
Regime is causal (BTC close up to t, decision earns from t+1). Graded on OOS (2024-05->end) AND the UNTUNED VAL
(2023-07..2024-05) so it is not fit to the attribution I just looked at. Success = net-positive with bounded DD in
BOTH windows + positive in >1 regime. u100 1d, next-open, prod cost, inv-vol cap33.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd

HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE)); ROOT=Path(__file__).resolve().parents[3]/"src"; sys.path.insert(0,str(ROOT))
from strat_ti_family_neff import load_panel
from strat_range_bot import range_held, trend_held, er
from strat.canonical_exits import COST_PROD
SL=7; CAP=0.33
WINDOWS={"VAL":("2023-07-01","2024-05-15"),"OOS":("2024-05-15","2026-08-01")}


def main():
    data,_=load_panel(); syms=list(data)
    cdf=pd.DataFrame({s:pd.Series(data[s]["c"],index=data[s]["d"]) for s in syms}).sort_index()
    odf=pd.DataFrame({s:pd.Series(data[s]["o"],index=data[s]["d"]) for s in syms}).reindex(cdf.index)
    idx=cdf.index; cols=list(cdf.columns); T=len(idx); A=len(cols); C=cdf.to_numpy(); O=odf.to_numpy()
    roo=np.full((T,A),0.0); roo[:-1]=O[1:]/np.where(O[:-1]>0,O[:-1],np.nan)-1.0; roo=np.nan_to_num(roo)
    vol=cdf.pct_change(fill_method=None).rolling(20,min_periods=10).std().to_numpy(); valid=np.isfinite(C)
    HR=np.zeros((T,A)); HT=np.zeros((T,A))
    for j,s in enumerate(syms):
        c=data[s]["c"]; HR[:,j]=pd.Series(range_held(c),index=data[s]["d"]).reindex(idx).fillna(0).to_numpy()
        HT[:,j]=pd.Series(trend_held(c),index=data[s]["d"]).reindex(idx).fillna(0).to_numpy()
    # causal BTC regime
    bc=cdf["BTCUSDT"].to_numpy() if "BTCUSDT" in cols else C[:,0]
    bsma=pd.Series(bc).rolling(50,min_periods=50).mean().to_numpy(); ber=er(bc)
    reg=np.where(ber<0.35,"CHOP",np.where(bc>bsma,"BULL","BEAR"))
    isbull=(reg=="BULL")[:,None]; isbear=(reg=="BEAR")[:,None]
    Hsw=HT*isbull+HR*isbear  # trend in bull, bounce in bear, cash in chop
    def book(H):
        w=np.where((H>0)&np.isfinite(vol)&(vol>0),1.0/np.where(vol>0,vol,np.nan),0.0); w=np.nan_to_num(w)*H
        s=w.sum(axis=1,keepdims=True); W=np.minimum(np.where(s>0,w/np.maximum(s,1e-12),0.0),CAP)
        Wl=np.vstack([np.zeros((1,A)),W[:-1]]); Wll=np.vstack([np.zeros((1,A)),Wl[:-1]])
        return np.nansum(np.where(valid,Wl*roo,0.0),axis=1)-np.nansum(np.abs(Wl-Wll),axis=1)*(COST_PROD/2),W
    nsw,Wsw=book(Hsw); nt,Wt=book(HT)
    # benchmarks
    ew=np.where(valid,1.0,0.0); ew=ew/np.maximum(ew.sum(axis=1,keepdims=True),1)
    bh_alt=np.nansum(np.vstack([np.zeros((1,A)),ew[:-1]])*roo,axis=1)
    btc_r=np.zeros(T); bo=odf["BTCUSDT"].to_numpy() if "BTCUSDT" in cols else O[:,0]; btc_r[:-1]=bo[1:]/np.where(bo[:-1]>0,bo[:-1],np.nan)-1.0; btc_r=np.nan_to_num(btc_r)
    def grade(net,W,w0,w1):
        m=(idx>=pd.Timestamp(w0))&(idx<pd.Timestamp(w1)); n=net[m]
        if len(n)<8: return None
        eq=np.cumprod(1+n); comp=float(eq[-1]-1)*100; dd=float(np.min((eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq))*100)
        wk=np.array([float(np.prod(1+n[i:i+SL])-1)*100 for i in range(0,len(n)-SL+1,SL)])
        pf=(-wk[wk>0].sum()/wk[wk<0].sum()) if (wk<0).any() and wk[wk<0].sum()!=0 else float("inf")
        expo=float(np.mean(W[m].sum(axis=1)))*100 if W is not None else 100.0
        return {"comp":round(comp,1),"maxdd":round(dd,1),"calmar":round(comp/abs(dd),2) if abs(dd)>1e-9 else 0,
                "pct_pos":round(float(np.mean(wk>0)),2),"pf":round(float(pf),2),"expo":round(expo,1),"n_wk":len(wk)}
    out={"check":"regime_switch_bot","ts":datetime.now().strftime("%Y%m%d_%H%M%S"),"windows":{}}
    print("="*104); print("WORKING-BOT BUILD 2: SWITCH_RC (trend@BULL / bounce@BEAR / cash@CHOP) -- OOS + untuned VAL"); print("="*104)
    for wn,(w0,w1) in WINDOWS.items():
        gsw=grade(nsw,Wsw,w0,w1); gt=grade(nt,Wt,w0,w1); gba=grade(bh_alt,ew,w0,w1); gbt=grade(btc_r,None,w0,w1)
        out["windows"][wn]={"SWITCH_RC":gsw,"TREND_only":gt,"BH_alt":gba,"BH_btc":gbt}
        print(f"\n  {wn} ({w0}..{w1})   [{gsw['n_wk']} weeks]")
        print(f"    {'bot':<12}{'comp%':>8}{'maxDD':>8}{'Calmar':>8}{'%wk+':>7}{'PF':>7}{'expo%':>7}")
        for nm,g in [("SWITCH_RC",gsw),("TREND_only",gt),("BH_alt",gba),("BH_btc",gbt)]:
            if g: print(f"    {nm:<12}{g['comp']:>+7.0f}%{g['maxdd']:>+8.0f}{g['calmar']:>8.2f}{g['pct_pos']*100:>6.0f}%{g['pf']:>7.2f}{g['expo']:>6.0f}%")
    sv=out["windows"]["VAL"]["SWITCH_RC"]; so=out["windows"]["OOS"]["SWITCH_RC"]
    print(f"\n  SWITCH_RC holds both windows? VAL {sv['comp']:+.0f}%/Cal{sv['calmar']:.2f}  OOS {so['comp']:+.0f}%/Cal{so['calmar']:.2f}  "
          f"-> {'YES (net-positive both)' if sv['comp']>0 and so['comp']>0 else 'NO'}")
    with open(HERE/"strat_regime_switch_bot.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"  Saved {HERE/'strat_regime_switch_bot.json'}")


if __name__=="__main__":
    main()
