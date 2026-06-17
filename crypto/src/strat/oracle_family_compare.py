"""src/strat/oracle_family_compare.py -- does the oracle "multitude" span INDICATOR FAMILIES, or is MA enough?
Compares per-window best-capture of the 2-10% moves across MA / RSI / MACD / Bollinger / Donchian (long-only).
FINDING (2026-06-10, RWYB BTC/ETH/SOL 1d/4h): MA DOMINATES (median capture +7-11% 1d vs ~0/neg for the rest; MA wins
72-79% of windows). Non-MA families enter late (Donchian) or mean-revert (Bollinger) -> capture LESS. The MA focus is
correct; no non-MA family is a hidden lever for capturing 2-10% up-moves. No emoji (cp1252).
Run: python src/strat/oracle_family_compare.py
"""
import sys, warnings; sys.path.insert(0,'src'); warnings.filterwarnings("ignore")
import numpy as np
if not hasattr(np,"NaN"): np.NaN=np.nan
import pandas as pd, pandas_ta as ta
from strat.oracle_decompose_deep import load_ohlc, _zigzag_pivots

def roi(pos, ret, a, b):
    pl=np.roll(pos,1); pl[0]=0.0
    return float(np.prod(1.0+pl[a:b+1]*ret[a:b+1])-1.0)

def fam_positions(close, high, low):
    c=pd.Series(close); pos={}
    # MA family (price>MA)
    for n in [5,8,13,21,34]: pos[f"ma{n}"]=(close> c.rolling(n,min_periods=2).mean().to_numpy()).astype(float)
    for n in [5,8,13]: pos[f"hma{n}"]=(close> ta.hma(c,length=n).to_numpy()).astype(float)
    # RSI (long when RSI>50, i.e. momentum regime)
    for n in [7,14]:
        r=ta.rsi(c,length=n)
        if r is not None: pos[f"rsi{n}"]=(r.to_numpy()>50).astype(float)
    # MACD (long when macd>signal)
    m=ta.macd(c)
    if m is not None and m.shape[1]>=3:
        pos["macd"]=(m.iloc[:,0].to_numpy()>m.iloc[:,2].to_numpy()).astype(float)
    # Bollinger (long when close>mid)
    bb=ta.bbands(c,length=20)
    if bb is not None: pos["bb"]=(close> bb.iloc[:,1].to_numpy()).astype(float)
    # Donchian breakout (long when close= rolling max)
    for n in [10,20]:
        hh=pd.Series(high).rolling(n,min_periods=2).max().to_numpy()
        pos[f"donch{n}"]=(close>=hh*0.999).astype(float)
    return pos

FAM={"MA":["ma5","ma8","ma13","ma21","ma34","hma5","hma8","hma13"],"RSI":["rsi7","rsi14"],"MACD":["macd"],"BB":["bb"],"DONCH":["donch10","donch20"]}

def test(asset,cadence,bars=28,step=14,lo=0.02):
    df=load_ohlc(asset+"USDT",cadence)
    if df is None or len(df)<200: return None
    close=df["close"].to_numpy(float);high=df["high"].to_numpy(float);low=df["low"].to_numpy(float)
    ret=np.zeros(len(close));ret[1:]=close[1:]/close[:-1]-1
    P=fam_positions(close,high,low)
    fam_best={f:[] for f in FAM}; ma_best=[]; overall_winner={}
    for w0 in range(60,len(close)-bars,step):
        a,b=w0,w0+bars-1
        piv=_zigzag_pivots(close[a:b+1],thr=lo)
        if not any(piv[k+1][1]>piv[k][1] and (piv[k+1][1]/piv[k][1]-1)>=lo for k in range(len(piv)-1)): continue
        rois={k:roi(P[k],ret,a,b) for k in P if k in [x for v in FAM.values() for x in v]}
        for f,keys in FAM.items():
            kk=[k for k in keys if k in rois]
            if kk: fam_best[f].append(max(rois[k] for k in kk))
        win=max(rois,key=rois.get); wf=[f for f,ks in FAM.items() if win in ks][0]
        overall_winner[wf]=overall_winner.get(wf,0)+1
    n=sum(overall_winner.values())
    print(f"  {asset} {cadence}: median best-capture by family: "+"  ".join(f"{f}={np.median(v)*100:+.2f}%" for f,v in fam_best.items() if v))
    print(f"      family WINS the window: "+"  ".join(f"{f} {100*overall_winner.get(f,0)//max(n,1)}%" for f in FAM))
    return True

print("=== NON-MA vs MA family oracle (best per-window capture of 2-10% moves), held-in descriptive ===")
for cad in ["1d","4h"]:
    for a in ["BTC","ETH","SOL"]: test(a,cad)
