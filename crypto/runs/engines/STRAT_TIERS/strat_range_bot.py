"""WORKING-BOT BUILD 1: the RANGE/MEAN-REVERSION bot (harvests CHOP) + regime-switch with the TREND bot.

The failure diagnosis: I only ever had ONE edge (trend), graded it in ALL regimes, and it loses in chop. Real
all-weather bots run the RIGHT edge per regime. Trend bot profits in trends; a RANGE bot profits in chop -- but
NAKED mean-reversion loses -85% (buys falling knives in downtrends). The fix that makes MR a real bot = a REGIME
GATE: only buy-the-dip when the asset is RANGING (low efficiency ratio), and a hard STOP for when the range breaks.

Three bots, u100 1d, next-open, prod cost 0.0028 RT, inv-vol sizing, graded by THEIR OWN criteria + complementarity:
  RANGE : buy z<-1 (oversold in 20d band) WHEN ranging (ER14<0.30); exit z>+0.5 (revert) | stop -15% | 20d cap.
  TREND : long when sma20>sma50 AND uptrend-efficient (ER14>0.35) -- the edge we know works in trends.
  SWITCH: per asset per bar, run whichever regime is active (they gate to near-disjoint regimes) -> all-weather.
Success criterion = NOT 'beats buy-hold DSR>0.95'. It's: positive expectancy in-condition, PF>1, bounded DD, and
COMPLEMENTARITY (range positive in weeks trend is flat/negative) so the SWITCH is smoother + positive in more weeks.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd

HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE)); ROOT=Path(__file__).resolve().parents[3]/"src"; sys.path.insert(0,str(ROOT))
from strat_ti_family_neff import load_panel
from strat.canonical_exits import COST_PROD
SL=7; CAP=0.33; OOS=pd.Timestamp("2024-05-15")


def er(c,n=14):
    num=np.abs(c-pd.Series(c).shift(n).to_numpy()); den=pd.Series(np.abs(np.diff(c,prepend=c[0]))).rolling(n,min_periods=n//2).sum().to_numpy()
    return np.where(den>1e-9,num/np.maximum(den,1e-9),0.0)


def range_held(c):
    """buy oversold in a RANGING market; exit on revert / stop / time-cap. state machine, causal."""
    n=len(c); sma=pd.Series(c).rolling(20,min_periods=20).mean().to_numpy(); sd=pd.Series(c).rolling(20,min_periods=20).std().to_numpy()
    z=np.where(sd>1e-9,(c-sma)/np.maximum(sd,1e-9),0.0); e=er(c); held=np.zeros(n,np.int8); pos=0; entry=0.0; bars=0
    for i in range(n):
        if not (np.isfinite(z[i]) and np.isfinite(sma[i])): held[i]=pos; continue
        if pos==0:
            if z[i]<-1.0 and e[i]<0.30: pos=1; entry=c[i]; bars=0
        else:
            bars+=1
            if z[i]>0.5 or c[i]<entry*0.85 or bars>=20: pos=0
        held[i]=pos
    return held


def trend_held(c):
    sma20=pd.Series(c).rolling(20,min_periods=20).mean().to_numpy(); sma50=pd.Series(c).rolling(50,min_periods=50).mean().to_numpy()
    e=er(c); return ((sma20>sma50)&(e>0.35)&np.isfinite(sma50)).astype(np.int8)


def main():
    data,_=load_panel(); syms=list(data)
    cdf=pd.DataFrame({s:pd.Series(data[s]["c"],index=data[s]["d"]) for s in syms}).sort_index()
    odf=pd.DataFrame({s:pd.Series(data[s]["o"],index=data[s]["d"]) for s in syms}).reindex(cdf.index)
    idx=cdf.index; cols=list(cdf.columns); T=len(idx); A=len(cols); C=cdf.to_numpy(); O=odf.to_numpy()
    roo=np.full((T,A),0.0); roo[:-1]=O[1:]/np.where(O[:-1]>0,O[:-1],np.nan)-1.0; roo=np.nan_to_num(roo)
    vol=cdf.pct_change(fill_method=None).rolling(20,min_periods=10).std().to_numpy(); valid=np.isfinite(C)
    om=np.asarray(idx>=OOS); st=[i for i in range(0,T-SL+1,SL)]
    HR=np.zeros((T,A)); HT=np.zeros((T,A))
    for j,s in enumerate(syms):
        c=data[s]["c"]; HR[:,j]=pd.Series(range_held(c),index=data[s]["d"]).reindex(idx).fillna(0).to_numpy()
        HT[:,j]=pd.Series(trend_held(c),index=data[s]["d"]).reindex(idx).fillna(0).to_numpy()
    HSW=((HR+HT)>0).astype(float)  # switch = whichever regime active (near-disjoint)
    def book(H):
        w=np.where((H>0)&np.isfinite(vol)&(vol>0),1.0/np.where(vol>0,vol,np.nan),0.0); w=np.nan_to_num(w)*H
        s=w.sum(axis=1,keepdims=True); W=np.minimum(np.where(s>0,w/np.maximum(s,1e-12),0.0),CAP)
        Wl=np.vstack([np.zeros((1,A)),W[:-1]]); Wll=np.vstack([np.zeros((1,A)),Wl[:-1]])
        return np.nansum(np.where(valid,Wl*roo,0.0),axis=1)-np.nansum(np.abs(Wl-Wll),axis=1)*(COST_PROD/2), W
    def grade(net,W):
        n=net[om]; eq=np.cumprod(1+n); comp=float(eq[-1]-1)*100
        dd=float(np.min((eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq))*100) if len(eq)>1 else 0
        wk=np.array([float(np.prod(1+n[i:i+SL])-1)*100 for i in range(0,len(n)-SL+1,SL)])
        pf=(-wk[wk>0].sum()/wk[wk<0].sum()) if (wk<0).any() and wk[wk<0].sum()!=0 else float("inf")
        expo=float(np.mean(W[om].sum(axis=1)))*100
        return {"comp":round(comp,1),"maxdd":round(dd,1),"pct_pos":round(float(np.mean(wk>0)),2),"pf":round(float(pf),2),
                "expo":round(expo,1),"n_wk":len(wk)},wk
    nr,Wr=book(HR); nt,Wt=book(HT); nsw,Wsw=book(HSW)
    gr,wr=grade(nr,Wr); gt,wt=grade(nt,Wt); gsw,wsw=grade(nsw,Wsw)
    # PER-REGIME attribution (the decisive test): does each edge exist in ITS condition?
    bc=cdf["BTCUSDT"].to_numpy() if "BTCUSDT" in cols else C[:,0]
    bsma=pd.Series(bc).rolling(50,min_periods=50).mean().to_numpy(); ber=er(bc)
    reg=np.where(ber<0.35,"CHOP",np.where(bc>bsma,"BULL","BEAR"))  # BTC regime per bar
    regime_attr={}
    for rn in ["BULL","CHOP","BEAR"]:
        rm=(reg==rn)&om
        if rm.sum()<5: continue
        regime_attr[rn]={"n_bars":int(rm.sum()),
            "RANGE_pnl_pp":round(float(np.sum(nr[rm]))*100,1),"TREND_pnl_pp":round(float(np.sum(nt[rm]))*100,1)}
    # complementarity: corr of weekly P&L (range vs trend); and range's contribution in trend-flat weeks
    corr=float(np.corrcoef(wr,wt)[0,1]) if len(wr)==len(wt) and np.std(wr)>1e-9 and np.std(wt)>1e-9 else None
    trend_flat=np.abs(wt)<0.5; range_pos_when_flat=float(np.mean(wr[trend_flat]>0)) if trend_flat.any() else None
    range_contrib_flat=float(np.sum(wr[trend_flat])) if trend_flat.any() else None
    out={"check":"range_bot","oos":str(OOS.date()),"ts":datetime.now().strftime("%Y%m%d_%H%M%S"),
         "RANGE":gr,"TREND":gt,"SWITCH":gsw,"complementarity":{"weekly_corr":round(corr,3) if corr is not None else None,
         "range_pct_pos_when_trend_flat":round(range_pos_when_flat,2) if range_pos_when_flat is not None else None,
         "range_sum_pnl_when_trend_flat_pp":round(range_contrib_flat,1) if range_contrib_flat is not None else None}}
    print("="*96); print("WORKING-BOT BUILD 1: RANGE (chop) + TREND + SWITCH  (u100 1d, OOS 2024-05->end, prod cost)"); print("="*96)
    print(f"\n  {'bot':<8}{'comp%':>8}{'maxDD':>8}{'%wk+':>7}{'PF':>7}{'expo%':>7}{'weeks':>7}")
    for nm,g in [("RANGE",gr),("TREND",gt),("SWITCH",gsw)]:
        print(f"  {nm:<8}{g['comp']:>+7.0f}%{g['maxdd']:>+8.0f}{g['pct_pos']*100:>6.0f}%{g['pf']:>7.2f}{g['expo']:>6.0f}%{g['n_wk']:>7}")
    print(f"\n  COMPLEMENTARITY (the whole point):")
    print(f"    weekly P&L corr(RANGE, TREND) = {corr:+.2f}   (low/neg = they fill different weeks)")
    print(f"    RANGE positive in {range_pos_when_flat*100:.0f}% of weeks where TREND is flat  (fills the chop the trend bot sits out)")
    print(f"    RANGE total P&L contributed in trend-flat weeks = {range_contrib_flat:+.0f}pp")
    print(f"\n  SWITCH vs TREND-alone: %wk+ {gsw['pct_pos']*100:.0f}% vs {gt['pct_pos']*100:.0f}%, comp {gsw['comp']:+.0f}% vs {gt['comp']:+.0f}%, DD {gsw['maxdd']:+.0f} vs {gt['maxdd']:+.0f}")
    out["regime_attr"]=regime_attr
    print(f"\n  PER-REGIME P&L (the decisive test -- does each edge exist in ITS condition?):")
    print(f"    {'BTC regime':<8}{'bars':>6}{'RANGE pp':>10}{'TREND pp':>10}")
    for rn,ra in regime_attr.items():
        print(f"    {rn:<8}{ra['n_bars']:>6}{ra['RANGE_pnl_pp']:>+9.0f} {ra['TREND_pnl_pp']:>+9.0f}")
    print(f"    (RANGE should be >0 in CHOP if the chop edge is real; TREND >0 in BULL. Both<0 in BEAR = long-only floor.)")
    with open(HERE/"strat_range_bot.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\n  Saved {HERE/'strat_range_bot.json'}")


if __name__=="__main__":
    main()
