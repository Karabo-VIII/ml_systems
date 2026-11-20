"""src/strat/oracle_exit_knob.py -- is the EXIT (hold-length) knob factor-predictable + adaptable? (companion to
oracle_decompose_deep's entry-config decomposition). FINDING (2026-06-10, RWYB BTC/ETH/SOL/DOGE 1d/4h): the best
hold-length is EVEN LESS factor-predictable than the MA period (R2 0.01-0.13, ~0 at 4h); adapting the hold by factors
LOSES (1d edge -1.2..-4.2pp; 4h +-0.1pp). Best-hold ~11-12 bars, leave FIXED. Confirms: the exit is not an adaptive
lever either. No emoji (cp1252). Run: python src/strat/oracle_exit_knob.py
"""
import sys, warnings; sys.path.insert(0,'src'); warnings.filterwarnings("ignore")
import numpy as np
if not hasattr(np,"NaN"): np.NaN=np.nan
import pandas as pd
from strat.oracle_decompose_deep import load_ohlc, rich_factors, _zigzag_pivots, _sma

HOLDS=[2,4,6,10,16,24]   # the EXIT knob: bars to hold after entry
def sim_hold(close, entry_sig, H, a, b):
    """entries on rising edge of entry_sig within [a,b]; exit after H bars; sum realized ret (gross)."""
    tot=1.0; i=a
    while i<=b:
        if entry_sig[i] and (i==0 or not entry_sig[i-1]):
            j=min(i+H, len(close)-1)
            tot*= close[j]/close[i]; i=j+1
        else: i+=1
    return tot-1.0

def run(asset,cadence,bars=28,step=14,lo=0.02,train_frac=0.6):
    df=load_ohlc(asset+"USDT",cadence)
    if df is None or len(df)<200: return None
    close=df["close"].to_numpy(float);high=df["high"].to_numpy(float);low=df["low"].to_numpy(float)
    sma8=_sma(close,8); entry=(close>sma8).astype(float)   # in-trend; rising edge = entry
    recs=[]
    for w0 in range(60,len(close)-bars-max(HOLDS),step):
        a,b=w0,w0+bars-1
        piv=_zigzag_pivots(close[a:b+1],thr=lo)
        if not any(piv[k+1][1]>piv[k][1] and (piv[k+1][1]/piv[k][1]-1)>=lo for k in range(len(piv)-1)): continue
        rh={H:sim_hold(close,entry,H,a,b) for H in HOLDS}
        best=max(rh,key=rh.get)
        recs.append({"fac":rich_factors(close[a:b+1],high[a:b+1],low[a:b+1]),"rh":rh,"best":best,"logH":np.log(best)})
    if len(recs)<40: return None
    # factor -> best-hold correlation (descriptive)
    X=pd.DataFrame([r["fac"] for r in recs]); y=np.array([r["logH"] for r in recs])
    corr={k:round(float(X[k].corr(pd.Series(y))),2) for k in ["er","vol","autocorr","hurst","range","vov"]}
    # adaptability: predict best-hold_{w+1} from factor_w (linear), vs best-fixed-hold
    ntr=int(len(recs)*train_frac); use=["er","vol","vov","autocorr","hurst","range"]
    Xf=pd.DataFrame([r["fac"] for r in recs[:-1]])[use]; mu=Xf.iloc[:ntr].mean(); sd=Xf.iloc[:ntr].std()+1e-9
    Z=((Xf-mu)/sd).fillna(0).to_numpy(); yy=np.array([recs[i+1]["logH"] for i in range(len(recs)-1)])
    A=np.column_stack([np.ones(ntr),Z[:ntr]]); coef,*_=np.linalg.lstsq(A,yy[:ntr],rcond=None)
    pred=A@coef; r2=1-np.sum((yy[:ntr]-pred)**2)/(np.sum((yy[:ntr]-yy[:ntr].mean())**2)+1e-9)
    # fixed = best mean-hold on train
    meanH={H:np.mean([r["rh"][H] for r in recs[:ntr]]) for H in HOLDS}; fixedH=max(meanH,key=meanH.get)
    ad,fx,orc=[],[],[]
    for i in range(ntr,len(recs)-1):
        z=((pd.Series(recs[i]["fac"])[use]-mu)/sd).fillna(0).to_numpy()
        ph=min(HOLDS,key=lambda h:abs(np.log(h)-float(coef[0]+z@coef[1:])))
        nxt=recs[i+1]["rh"]; ad.append(nxt[ph]); fx.append(nxt[fixedH]); orc.append(max(nxt.values()))
    ad,fx,orc=map(np.array,[ad,fx,orc])
    print(f"  {asset} {cadence}: best-hold mean={np.mean([r['best'] for r in recs]):.1f} R2(factor->logH)={r2:.2f} corr={corr} | ADAPT: adaptive={np.mean(ad)*100:+.2f}% fixed(H={fixedH})={np.mean(fx)*100:+.2f}% oracle={np.mean(orc)*100:+.2f}% edge={np.mean(ad-fx)*100:+.3f}pp beats={np.mean(ad>fx):.0%}")
    return True

print("=== EXIT-KNOB decomposition: best HOLD-length per window, is it factor-predictable + adaptable? ===")
for cad in ["1d","4h"]:
    for a in ["BTC","ETH","SOL","DOGE"]: run(a,cad)
