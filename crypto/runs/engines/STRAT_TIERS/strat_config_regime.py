"""QUANT: within an MA type, does config SPEED specialize by REGIME? (does 9/21 win chop, 21/81 win bull, persistently?)

User hypothesis: even within SMA, no single config for all circumstances -- fast crosses (e.g. 9/21) for one regime,
slow (21/81) for another. Test it with rigor, decoupling the SIGNAL (cross) from gate/sizing.

PRE-REGISTERED.
  H0: config-speed performance is REGIME-INVARIANT (no fast/slow x regime interaction beyond noise).
  ALT: fast vs slow crosses specialize by regime AND the specialization PERSISTS out-of-sample.
  DECISIVE: the fast-minus-slow weekly-return spread must hold the SAME SIGN in-sample (<=2024-05-15) AND OOS (>).
  In-sample-only specialization = overfit, NOT a regime engine. Bar = PERSISTENCE, not in-sample significance.
Method: per type, each config = ungated GROSS cross book (EW over held names, close-to-close, lag-1) -> isolates the
signal from the breadth gate + cost (a separate layer; turnover reported since fast crosses pay more). Bucket by speed
(slow-MA period): FAST<=25 / MED 26-60 / SLOW>60. Regime per week (causal): BTC vs SMA50 (bull/bear) x efficiency-ratio
ER14 (trend/chop). Report bucket x regime mean weekly %, FAST-SLOW spread, IS vs OOS, + turnover. 1d, u100, point-in-time.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json, re
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd

HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE)); ROOT=Path(__file__).resolve().parents[3]/"src"; sys.path.insert(0,str(ROOT))
from strat.ma_type_upgrade import _MA
from strat.ma_movecatch_engine import select_band_trainval
from strat.ma_wealth_driver import _load_asset, _u100_symbols
import strat.fleet_lab as fl
BPD=fl.BARS_PER_DAY["1d"]; SL=7*BPD; WALL=int(pd.Timestamp("2026-07-01").value//10**6); CUT=pd.Timestamp("2024-05-15")
TYPES=["SMA","EMA","WMA","HMA","DEMA","TEMA","KAMA","VIDYA"]
def nums(n): return [int(x) for x in re.findall(r"\d+",n)]


def main():
    closes={}
    for s in _u100_symbols():
        r=_load_asset(s,"1d",WALL)
        if r is None: continue
        d,o,h,l,c=r; closes[s]=pd.Series(np.asarray(c,float),index=d)
    cdf=pd.DataFrame(closes).sort_index(); idx=cdf.index; cols=list(cdf.columns); T=len(idx); A=len(cols); C=cdf.to_numpy()
    ret=np.zeros((T,A)); ret[1:]=C[1:]/np.where(C[:-1]>0,C[:-1],np.nan)-1.0; ret=np.nan_to_num(ret)
    # ---- causal regime from BTC ----
    bc=cdf["BTCUSDT"].to_numpy() if "BTCUSDT" in cols else C[:,0]
    sma50=pd.Series(bc).rolling(50,min_periods=20).mean().to_numpy(); bull=bc>sma50
    mv=np.abs(np.diff(bc,prepend=bc[0])); er=np.abs(bc-pd.Series(bc).shift(14).to_numpy())/np.maximum(pd.Series(mv).rolling(14,min_periods=7).sum().to_numpy(),1e-9)
    trend=er>0.35   # efficiency-ratio: high=clean trend, low=chop
    reg=np.where(~bull,"BEAR",np.where(trend,"BULL_TREND","BULL_CHOP"))
    starts=list(range(0,T-SL+1,SL)); wkreg=[reg[i] for i in starts]; wkis=[idx[i]<CUT for i in starts]
    def buck(maxp): return "FAST" if maxp<=25 else ("MED" if maxp<=60 else "SLOW")
    out={"check":"config_regime_specialization","regime_def":"BTC vs SMA50 (bull/bear) x ER14>0.35 (trend/chop); week-start causal","ts":datetime.now().strftime("%Y%m%d_%H%M%S"),"types":{}}
    REGS=["BULL_TREND","BULL_CHOP","BEAR"]
    print("="*112); print("CONFIG-SPEED x REGIME specialization (ungated GROSS cross books) -- does fast/slow specialize + PERSIST?"); print("="*112)
    nT=sum(1 for r in wkreg if r=="BULL_TREND"); nC=sum(1 for r in wkreg if r=="BULL_CHOP"); nB=sum(1 for r in wkreg if r=="BEAR")
    print(f"  regime weeks: BULL_TREND {nT}  BULL_CHOP {nC}  BEAR {nB}  (total {len(wkreg)})\n")
    for t in TYPES:
        band,_=select_band_trainval(t,"1d"); f=_MA[t]
        periods=sorted({p for cfg in band for p in nums(cfg)}); cache={}
        for j in range(A):
            cj=C[:,j]
            for p in periods: cache[(j,p)]=f(cj,p)
        # accumulate per-bucket net + turnover
        bk_net={b:np.zeros(T) for b in ["FAST","MED","SLOW"]}; bk_turn={b:0.0 for b in ["FAST","MED","SLOW"]}; bk_cnt={b:0 for b in ["FAST","MED","SLOW"]}
        for cfg in band:
            ps=nums(cfg); b=buck(max(ps)); H=np.zeros((T,A))
            for j in range(A):
                mas=[cache[(j,p)] for p in ps]
                hh=(mas[0]>mas[1]) if len(ps)==2 else ((mas[0]>mas[1])&(mas[1]>mas[2]))
                H[:,j]=np.nan_to_num(hh).astype(float)
            sden=H.sum(axis=1,keepdims=True); W=np.where(sden>0,H/np.maximum(sden,1),0.0); Wl=np.vstack([np.zeros((1,A)),W[:-1]])
            bk_net[b]+=np.nansum(Wl*ret,axis=1); bk_turn[b]+=float(np.nanmean(np.abs(W-Wl).sum(axis=1))); bk_cnt[b]+=1
        rows={}
        for b in ["FAST","MED","SLOW"]:
            if bk_cnt[b]==0: continue
            net=bk_net[b]/bk_cnt[b]; wk=np.array([float(np.prod(1+net[i:i+SL])-1)*100 for i in starts])
            cell={}
            for r in REGS:
                for span,isf in [("IS",True),("OOS",False)]:
                    m=[wk[k] for k in range(len(wk)) if wkreg[k]==r and wkis[k]==isf]
                    cell[f"{r}_{span}"]=round(float(np.mean(m)),2) if m else None
            cell["turnover_per_bar"]=round(bk_turn[b]/bk_cnt[b],3); cell["n_configs"]=bk_cnt[b]
            rows[b]=cell
        out["types"][t]={"buckets":rows}
        # FAST - SLOW spread per regime, IS vs OOS + persistence
        print(f"  ### {t}  (FAST<=25 / MED / SLOW>60 slow-period; mean weekly GROSS % by regime)")
        print(f"    {'bucket':<7}{'#cfg':>5}{'turn/bar':>9}  | {'TREND_IS':>9}{'TREND_OOS':>10} | {'CHOP_IS':>9}{'CHOP_OOS':>10} | {'BEAR_IS':>9}{'BEAR_OOS':>10}")
        for b in ["FAST","MED","SLOW"]:
            if b not in rows: continue
            c=rows[b]
            def fmt(x): return f"{x:+.2f}" if x is not None else "   .  "
            print(f"    {b:<7}{c['n_configs']:>5}{c['turnover_per_bar']:>9.3f}  | {fmt(c['BULL_TREND_IS']):>9}{fmt(c['BULL_TREND_OOS']):>10} | {fmt(c['BULL_CHOP_IS']):>9}{fmt(c['BULL_CHOP_OOS']):>10} | {fmt(c['BEAR_IS']):>9}{fmt(c['BEAR_OOS']):>10}")
        if "FAST" in rows and "SLOW" in rows:
            spreads={}
            for r in REGS:
                for span in ["IS","OOS"]:
                    a=rows["FAST"].get(f"{r}_{span}"); s=rows["SLOW"].get(f"{r}_{span}")
                    spreads[f"{r}_{span}"]=round(a-s,2) if (a is not None and s is not None) else None
            out["types"][t]["fast_minus_slow"]=spreads
            persist=[]
            for r in REGS:
                si=spreads.get(f"{r}_IS"); so=spreads.get(f"{r}_OOS")
                if si is not None and so is not None: persist.append((r,si,so,np.sign(si)==np.sign(so)))
            ptxt=" ".join(f"{r}:IS{si:+.1f}/OOS{so:+.1f}{'OK' if ok else 'FLIP'}" for r,si,so,ok in persist)
            print(f"    FAST-SLOW spread persistence: {ptxt}")
        print()
    with open(HERE/"strat_config_regime.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"  VERDICT RULE: a REAL regime engine needs the FAST-SLOW spread to keep its SIGN from IS->OOS in a regime.")
    print(f"  If signs FLIP (IS->OOS) -> config-speed specialization is OVERFIT noise, not a regime engine.")
    print(f"\nSaved {HERE/'strat_config_regime.json'}")


if __name__=="__main__":
    main()
