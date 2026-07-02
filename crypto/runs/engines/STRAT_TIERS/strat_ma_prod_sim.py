"""PROD SIMULATION -- each MA-family bot broken down PER ASSET (user 2026-07-02).

For every MA type (SMA/EMA/WMA/HMA/DEMA/TEMA/KAMA/VIDYA) run the bot STANDALONE on EACH asset with production-realistic
execution and attribute the result per asset: this is the "if the system were live, how did each bot do on each coin" view.
  signal : the DEPLOYED MA held (held_rolling over the canonical train/val-selected band -> confirmation-break + 7d time-cap)
  exec   : next-OPEN fills (pos earns open[t]->open[t+1]), CAUSAL lag-1, prod cost 0.0028 RT charged on each fill
  window : OOS = 2024-05-15 -> end (the live/forward-relevant period); FULL history also reported
  unit   : single-asset, in-market-or-cash (sizing is a portfolio choice, out of scope here -- this is per-asset attribution)
Per (bot,asset): OOS compound, #trades, win%, avg trade, time-in-market, maxDD, FULL compound. Aggregate = EW portfolio of
all assets (the naked cross-sectional MA bot). Full 8xN matrix -> CSV + JSON; console shows aggregate + top/bottom assets.
Survivorship panel (today's u100) -> per-asset headline overstated; the RELATIVE map (which coins the bot works on) is the point.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json, csv
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd

HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE)); ROOT=Path(__file__).resolve().parents[3]/"src"; sys.path.insert(0,str(ROOT))
from strat_ti_family_neff import load_panel
from strat.ma_slice_engine import held_rolling, _cached_band_held, VAL_END, OOS_START
from strat.ma_type_upgrade import MA_TYPES
from strat.canonical_exits import COST_PROD
import strat.fleet_lab as fl
BPD=fl.BARS_PER_DAY["1d"]; LB=14*BPD; SL=7*BPD
WALL=int(pd.Timestamp("2026-07-01").value//10**6); OOS_TS=pd.Timestamp(OOS_START); COST=COST_PROD


def sim_asset(o,c,held):
    """single-asset prod sim: pos=held lag1, earn open[t]->open[t+1], prod cost on fills. returns net stream + per-trade rets."""
    n=len(c); roo=np.zeros(n); roo[:-1]=o[1:]/np.where(o[:-1]>0,o[:-1],np.nan)-1.0; roo=np.nan_to_num(roo)
    pos=np.zeros(n); pos[1:]=held[:-1]  # causal: signal at close t-1 -> in at open t
    turn=np.abs(np.diff(pos,prepend=0.0))
    net=pos*roo-turn*(COST/2)
    # per-trade returns (segment held runs on pos)
    trades=[]; i=1
    while i<n:
        if pos[i]>0 and pos[i-1]==0:
            j=i
            while j<n and pos[j]>0: j+=1
            seg=np.prod(1+roo[i:j])-1-COST  # one round trip cost
            trades.append(float(seg)); i=j
        else: i+=1
    return net,roo,np.array(trades)


def grade(net,mask):
    n=net[mask]
    if len(n)<5: return None
    eq=np.cumprod(1+n); comp=float(eq[-1]-1)*100
    dd=float(np.min((eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq))*100)
    return {"comp":round(comp,1),"maxdd":round(dd,1)}


def main():
    data,_=load_panel(); syms=list(data)
    types=[t for t in MA_TYPES]  # all 8 MA-family types
    out={"check":"ma_prod_sim","window_oos":f"{OOS_START}..end","cost_rt":COST,"ts":datetime.now().strftime("%Y%m%d_%H%M%S"),"bots":{}}
    csv_rows=[["bot","asset","oos_comp","oos_maxdd","trades","win_pct","avg_trade_pct","tim_pct","full_comp"]]
    print("="*118); print(f"MA-FAMILY PROD SIMULATION -- per bot x per asset  ({len(syms)} assets, OOS {OOS_START}->end, next-open, prod cost {COST})"); print("="*118)
    for t in types:
        per=[]; oos_nets=[]
        for s in syms:
            d=data[s]["d"]; o=data[s]["o"]; h=data[s]["h"]; l=data[s]["l"]; c=data[s]["c"]
            try:
                band=_cached_band_held(t,"1d",s,o,h,l,c,WALL); held=held_rolling(c,band,LB,SL)
            except Exception: continue
            net,roo,tr=sim_asset(o,c,held)
            oos_m=np.asarray(pd.DatetimeIndex(d)>=OOS_TS)
            go=grade(net,oos_m); gf=grade(net,np.ones(len(net),bool))
            if go is None: continue
            pos=np.zeros(len(c)); pos[1:]=held[:-1]; tim=float(np.mean(pos[oos_m]))*100
            # per-trade within OOS: recompute trades restricted to oos window
            tr_o=[]; i=1; n=len(c)
            while i<n:
                if pos[i]>0 and pos[i-1]==0:
                    j=i
                    while j<n and pos[j]>0: j+=1
                    if oos_m[i]: tr_o.append(float(np.prod(1+roo[i:j])-1-COST))
                    i=j
                else: i+=1
            tr_o=np.array(tr_o); win=float(np.mean(tr_o>0))*100 if len(tr_o) else 0.0
            avg=float(np.mean(tr_o))*100 if len(tr_o) else 0.0
            rec={"asset":s,"oos_comp":go["comp"],"oos_maxdd":go["maxdd"],"trades":len(tr_o),"win_pct":round(win,0),
                 "avg_trade_pct":round(avg,2),"tim_pct":round(tim,0),"full_comp":gf["comp"] if gf else None}
            per.append(rec); oos_nets.append((s,net,oos_m))
            csv_rows.append([t,s,go["comp"],go["maxdd"],len(tr_o),round(win,0),round(avg,2),round(tim,0),gf["comp"] if gf else ""])
        if not per: continue
        # EW portfolio of all assets (the naked cross-sectional MA bot), OOS
        alld=sorted(set().union(*[set(pd.Series(0,index=data[s]["d"]).index) for s,_,_ in oos_nets]))
        idx=pd.DatetimeIndex(alld); PN=pd.DataFrame({s:pd.Series(net,index=data[s]["d"]) for s,net,_ in oos_nets}).reindex(idx)
        ew=PN.mean(axis=1,skipna=True).fillna(0).to_numpy(); om=np.asarray(idx>=OOS_TS); gew=grade(ew,om)
        oc=np.array([r["oos_comp"] for r in per]); pos_frac=float(np.mean(oc>0))*100
        per.sort(key=lambda r:-r["oos_comp"])
        out["bots"][t]={"n_assets":len(per),"ew_portfolio_oos":gew,"pct_assets_profitable_oos":round(pos_frac,0),
                        "median_asset_oos_comp":round(float(np.median(oc)),1),"per_asset":per}
        print(f"\n  === {t} bot ===  EW-portfolio OOS {gew['comp']:+.0f}% (DD {gew['maxdd']:+.0f})  |  {pos_frac:.0f}% of {len(per)} assets profitable  |  median asset {np.median(oc):+.0f}%")
        print(f"    {'asset':<12}{'OOScomp':>9}{'DD':>6}{'trades':>7}{'win%':>6}{'avgTr%':>8}{'TiM%':>6}{'FULLcomp':>10}")
        show=per[:12]+[None]+per[-6:] if len(per)>18 else per
        for r in show:
            if r is None: print(f"    {'... (' + str(len(per)-18) + ' more in CSV) ...':<40}"); continue
            print(f"    {r['asset']:<12}{r['oos_comp']:>+8.0f}%{r['oos_maxdd']:>+6.0f}{r['trades']:>7}{r['win_pct']:>5.0f}%{r['avg_trade_pct']:>+7.1f}%{r['tim_pct']:>5.0f}%{(r['full_comp'] if r['full_comp'] is not None else 0):>+9.0f}%")
    with open(HERE/"strat_ma_prod_sim.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    with open(HERE/"ma_prod_sim_per_asset.csv","w",newline="") as fo: csv.writer(fo).writerows(csv_rows)
    print(f"\n  Full 8-bot x per-asset matrix -> ma_prod_sim_per_asset.csv  +  strat_ma_prod_sim.json")
    print(f"  READ: single-asset in/cash attribution, prod exec. Per-asset headline is survivorship-inflated (today's u100);")
    print(f"        the deployable signal is the RELATIVE map -- which coins each MA bot works on + win-rate/trade-count shape.")


if __name__=="__main__":
    main()
