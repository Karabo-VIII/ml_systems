"""WORKING-BOT BUILD 3: REBALANCING / VOLATILITY-HARVEST bot (Shannon's Demon) -- the CHOP fill.

Research-flagged as the SAFE chop fill for a long-only spot basket: monetizes oscillation + low correlation via the
REBALANCING PREMIUM (mechanically sell winners / buy losers back to target) -- no forecast, no regime filter, bounded
worst-case (floor = EW buy-hold beta). Enemy = fees -> threshold rebalancing controls turnover.

Isolate the premium cleanly on the SURVIVOR basket (names present whole window, so no listing noise -- survivorship
caveat noted): premium = rebalanced_CAGR - buyhold_same_basket_CAGR. Variants: weekly-calendar, threshold(+/-25% rel).
Net of prod cost 0.0028 RT. Per-regime (premium should be biggest in CHOP). + breadth-gated deployable version.
u100 1d, close-to-close (portfolio-construction bot, always-invested base).
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd

HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE)); ROOT=Path(__file__).resolve().parents[3]/"src"; sys.path.insert(0,str(ROOT))
from strat_ti_family_neff import load_panel
from strat_range_bot import er
from strat.canonical_exits import COST_PROD
OOS=pd.Timestamp("2024-05-15"); SL=7


def sim_rebal(R, mode, band=0.25, cost=COST_PROD):
    """EW rebalance sim on a fixed basket. R=(T,A) simple returns. mode: 'hold'|'weekly'|'thresh'. returns equity, turnover_sum."""
    T,A=R.shape; w=np.ones(A)/A; eq=np.zeros(T); eq[0]=1.0; tot_turn=0.0; tgt=np.ones(A)/A
    for t in range(1,T):
        w=w*(1+R[t]); pv=w.sum(); w=w/pv; eq[t]=eq[t-1]*pv  # drift
        reb=False
        if mode=="weekly" and t%SL==0: reb=True
        elif mode=="thresh" and (np.any(w>tgt*(1+band)) or np.any(w<tgt*(1-band))): reb=True
        if reb:
            turn=np.abs(tgt-w).sum(); tot_turn+=turn; eq[t]*=(1-turn*(cost/2)); w=tgt.copy()
    return eq,tot_turn


def cagr(eq,nyr): return (eq[-1]**(1/max(nyr,1e-9))-1)*100
def maxdd(eq): return float(np.min((eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq))*100)


def main():
    data,_=load_panel(); syms=list(data)
    cdf=pd.DataFrame({s:pd.Series(data[s]["c"],index=data[s]["d"]) for s in syms}).sort_index()
    # OOS survivor basket: names with full data over OOS
    oidx=cdf.index[cdf.index>=OOS]
    sub=cdf.loc[oidx]; keep=[s for s in syms if sub[s].notna().mean()>0.98]
    sub=sub[keep].dropna(how="any"); idx=sub.index; A=len(keep); T=len(idx)
    R=sub.pct_change(fill_method=None).fillna(0).to_numpy(); nyr=T/365.0
    bc=cdf["BTCUSDT"].reindex(idx).to_numpy() if "BTCUSDT" in cdf else sub.iloc[:,0].to_numpy()
    bsma=pd.Series(bc).rolling(50,min_periods=50).mean().to_numpy(); ber=er(bc)
    reg=np.where(ber<0.35,"CHOP",np.where(bc>bsma,"BULL","BEAR"))
    print("="*96); print(f"WORKING-BOT BUILD 3: REBALANCING / VOL-HARVEST -- survivor basket {A} names, OOS {T} bars ({nyr:.1f}y)"); print("="*96)
    res={}
    for mode in ["hold","weekly","thresh"]:
        eq,tt=sim_rebal(R,mode); res[mode]={"cagr":round(cagr(eq,nyr),1),"final":round(float(eq[-1]),3),
            "maxdd":round(maxdd(eq),1),"turnover":round(float(tt),1)}
    prem_w=res["weekly"]["cagr"]-res["hold"]["cagr"]; prem_t=res["thresh"]["cagr"]-res["hold"]["cagr"]
    print(f"\n  {'variant':<10}{'CAGR%':>8}{'final x':>9}{'maxDD':>8}{'turnover(RT)':>14}")
    for m in ["hold","weekly","thresh"]:
        r=res[m]; print(f"  {m:<10}{r['cagr']:>+7.1f}%{r['final']:>9.2f}{r['maxdd']:>+8.1f}{r['turnover']:>14.1f}")
    print(f"\n  REBALANCING PREMIUM (net of prod cost): weekly {prem_w:+.1f}pp/yr  |  threshold {prem_t:+.1f}pp/yr  vs buy-hold")
    # per-regime premium: run daily-rebal minus hold within each regime segment (compound within regime)
    print(f"\n  Per-regime premium (weekly-rebal vs hold, compounded within regime):")
    seg={}
    for rn in ["BULL","CHOP","BEAR"]:
        m=(reg==rn);
        if m.sum()<10: continue
        Rr=R[m]
        eqh,_=sim_rebal(Rr,"hold"); eqw,_=sim_rebal(Rr,"weekly")
        ch=(eqh[-1]-1)*100; cw=(eqw[-1]-1)*100
        seg[rn]={"n":int(m.sum()),"hold_pp":round(ch,1),"rebal_pp":round(cw,1),"premium_pp":round(cw-ch,1)}
        print(f"    {rn:<6} {int(m.sum()):>4} bars  hold {ch:>+6.1f}%  rebal {cw:>+6.1f}%  premium {cw-ch:>+5.1f}pp")
    out={"check":"rebalance_bot","n_names":A,"n_years":round(nyr,2),"variants":res,
         "premium_weekly_pp_yr":round(prem_w,1),"premium_thresh_pp_yr":round(prem_t,1),"regime":seg,
         "ts":datetime.now().strftime("%Y%m%d_%H%M%S")}
    with open(HERE/"strat_rebalance_bot.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\n  READ: premium>0 net of cost = rebalancing HARVESTS chop (adds return over buy-hold same basket). It's a")
    print(f"        RELATIVE edge (both are long-beta); pair with the breadth->cash gate for the crash regime. Survivorship: fixed basket.")
    print(f"  Saved {HERE/'strat_rebalance_bot.json'}")


if __name__=="__main__":
    main()
