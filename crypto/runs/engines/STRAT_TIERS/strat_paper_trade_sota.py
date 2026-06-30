"""QUANT: wire the SOTA MA-family builds to the canonical PAPER TRADER, run the FULL OOS window, report WEEKLY (honest).

Reuses the project paper trader (AUDIT_ORACLE_UPGRADE/paper_trade_ma_family.py: next-open fills, prod cost 0.0028 RT,
PIT u100, $10k, journal). SOTA SELECTION (principled, NOT best-OOS-config -> no overfit):
  CONV    : canonical cross-type breadth-gated book (the historical deployable)
  SMA     : per-type breadth-gated (all 8 types are n_eff~1; SMA = simplest representative)
  B_ER    : SMA + breadth AND market trend-efficiency (ER>0.35) -- session capital-preservation candidate
  PARTIAL : always 15-100% invested scaled by breadth (trend-names else EW-mkt) -- session consistency candidate
Window = FULL OOS 2024-05-15 .. data-end. Report: per-WEEK (7d) return + equity for each + BH-BTC; summary; honest caveats.
NOTE (binding): these gates were designed on this OOS data (multiple comparisons) -> the weekly replay is the honest
OOS read, but NW p>0.05 / p05<0 (prior battery) => de-risked beta, NOT statistically-robust alpha. Survivorship: PIT-ish
wall but today's u100 -> returns if anything overstated.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd

HERE=Path(__file__).resolve().parent; AOU=HERE.parent/"AUDIT_ORACLE_UPGRADE"; sys.path.insert(0,str(AOU))
ROOT=Path(__file__).resolve().parents[3]/"src"; sys.path.insert(0,str(ROOT))
import paper_trade_ma_family as P   # the canonical paper trader
P.FWD_START="2024-05-15"            # FULL OOS (override the Q2-2025 default)
BPD=P.BARS_PER_DAY["1d"]; SL=7*BPD; START=P.START_CAP; CAP=P.CAP


def main():
    f=P.build(); cdf=f["cdf"]; idx=f["idx"]; cols=f["cols"]; T=len(idx); A=f["A"]; vol=f["vol"]; roo=f["roo"]; gb=f["g_brd"]
    C=cdf.to_numpy(); sw=100*BPD
    sall=cdf.rolling(sw,min_periods=sw//2).mean().to_numpy(); vc=np.isfinite(sall).sum(axis=1)
    brd=np.where(vc>0,np.nansum((C>sall)*np.isfinite(sall),axis=1)/np.maximum(vc,1),np.nan); brd=np.nan_to_num(brd)
    bc=cdf["BTCUSDT"].to_numpy(); mv=np.abs(np.diff(bc,prepend=bc[0]))
    er=np.abs(bc-pd.Series(bc).shift(14).to_numpy())/np.maximum(pd.Series(mv).rolling(14,min_periods=7).sum().to_numpy(),1e-9)
    erok=(er>0.35).astype(float)
    def w_invvol(held,gate2d):
        mult=held*gate2d; iv=np.where((mult>0)&np.isfinite(vol)&(vol>0),1.0/np.where(vol>0,vol,np.nan),0.0)
        iv=np.nan_to_num(iv)*mult; s=iv.sum(axis=1,keepdims=True); return np.minimum(np.where(s>0,iv/np.maximum(s,1e-12),0.0),CAP)
    gbB=np.broadcast_to(gb[:,None],(T,A)).astype(float)
    sma_held=f["typ"]["SMA"].astype(float)
    # candidates
    W_conv=P.weights(f,"conv"); W_sma=P.weights(f,"SMA")
    W_ber=w_invvol(sma_held, gbB*np.broadcast_to(erok[:,None],(T,A)))
    # PARTIAL: always-in scaled exposure
    e=np.clip((brd-0.20)/0.60,0.15,1.0); ew=np.where(np.isfinite(C),1.0,0.0); ew=ew/np.maximum(ew.sum(axis=1,keepdims=True),1)
    Wtr_n=w_invvol(sma_held,np.ones((T,A))); trend_any=Wtr_n.sum(axis=1,keepdims=True)>1e-9
    W_part=e[:,None]*np.where(trend_any,Wtr_n,ew)
    CANDS={"CONV":W_conv,"SMA":W_sma,"B_ER":W_ber,"PARTIAL":W_part}
    # paper-trade each via the canonical runner
    res={};
    for nm,W in CANDS.items():
        net,dates,eq,journal=P.paper_trade(f,W); res[nm]={"net":net,"dates":dates,"eq":eq,"m":P.metrics(net,eq,BPD)}
    # BH-BTC
    nb,eb=P.bench_net(f,"btc"); res["BH_BTC"]={"net":nb,"dates":idx[f["mask"]],"eq":START*np.cumprod(1+nb),"m":P.metrics(nb,eb,BPD)}
    dates=res["CONV"]["dates"]; starts=list(range(0,len(dates)-SL+1,SL))
    def wkrets(net): return [float(np.prod(1+net[i:i+SL])-1)*100 for i in starts]
    wkdates=[str(pd.Timestamp(dates[i]).date()) for i in starts]
    print("="*108)
    print(f"PAPER-TRADE SOTA MA builds -- FULL OOS {P.FWD_START}..{str(pd.Timestamp(dates[-1]).date())} | $10k, next-open, prod cost | WEEKLY")
    print("="*108)
    print(f"\n  SUMMARY:  {'candidate':<10}{'final$':>9}{'compound':>10}{'lin/wk':>8}{'Sharpe':>7}{'maxDD':>7}{'7dprof%':>8}")
    for nm in ["CONV","SMA","B_ER","PARTIAL","BH_BTC"]:
        m=res[nm]["m"]; print(f"    {'':8}{nm:<10}{m['final_equity']:>9.0f}{m['compound_pct']:>+9.1f}%{m['lin_wk']:>+8.2f}{m['sharpe']:>+7.2f}{m['maxdd_pct']:>+7.0f}{m['pct7d']*100:>7.0f}%")
    # weekly table
    wk={nm:wkrets(res[nm]["net"]) for nm in ["CONV","SMA","B_ER","PARTIAL","BH_BTC"]}
    print(f"\n  WEEKLY RETURNS (%, 7d):  {'week':<12}{'CONV':>8}{'SMA':>8}{'B_ER':>8}{'PARTIAL':>9}{'BTC':>8}")
    for i,dt in enumerate(wkdates):
        print(f"    {'':4}{dt:<12}{wk['CONV'][i]:>+7.1f}{wk['SMA'][i]:>+8.1f}{wk['B_ER'][i]:>+8.1f}{wk['PARTIAL'][i]:>+8.1f}{wk['BH_BTC'][i]:>+8.1f}")
    out={"check":"paper_trade_sota","window":f"{P.FWD_START}..{str(pd.Timestamp(dates[-1]).date())}","exec":"canonical paper trader: next-open, prod 0.0028RT, $10k, PIT u100",
         "summary":{nm:res[nm]["m"] for nm in res},"weekly":{"dates":wkdates,**{nm:[round(x,2) for x in wk[nm]] for nm in wk}},
         "caveats":["gates designed on this OOS data (multiple comparisons)","de-risked beta not robust alpha (NW p>0.05, p05<0 prior battery)","survivorship: today's u100 -> returns overstated"],
         "ts":datetime.now().strftime("%Y%m%d_%H%M%S")}
    with open(HERE/"strat_paper_trade_sota.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\nSaved {HERE/'strat_paper_trade_sota.json'}")


if __name__=="__main__":
    main()
