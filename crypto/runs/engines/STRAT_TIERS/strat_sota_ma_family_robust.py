"""AUDIT (user q): are the MA FAMILY trend-riders robust? -- the actual product (NOT the momentum/timing magic-pill).

The prior audit hammered the momentum-top8 + learned-timing engine (a MOVER-SELECTION 'magic pill') and found no robust
return-alpha. But the canonical product is the MA trend-following FAMILY (SMA/EMA/WMA/VIDYA/KAMA, breadth-gated, inv-vol
cap33, 14d/7d ROLLING, next-open, prod cost) -- a DE-RISKED-BETA trend rider, not a selection-alpha engine. It has never
been run through the same battery. This runs it: per MA engine (CONV book + each type), per window (OOS-2024 bull-recovery
+ OOS-2025 bear/chop): Newey-West(L3) HAC p, block-boot p05, top-k-WEEK lumpiness, DSR, deploy rate, compound + maxDD vs
BTC buy-hold. The HONEST question for a trend-rider is NOT 'is there alpha' -- it is: (a) does it participate robustly,
(b) does it reduce drawdown vs BH (the de-risking value), (c) is its return lumpy or spread across many weeks?
Reuses the canonical engine (ma_weekly_replay.build). 1d, u100, point-in-time wall, prod 0.0028 RT, long-only spot lev=1.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd
from scipy import stats as ss

HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
ROOT=Path(__file__).resolve().parents[3]/"src"; sys.path.insert(0,str(ROOT))
import ma_weekly_replay as R
from strat.ma_wealth_driver import _load_asset
BPD=R.BPD; SB=7*BPD; CAP=R.CAP; TYPES=R.TYPES


def nw_p(wk,L=3):
    n=len(wk); mu=np.mean(wk); x=wk-mu; var=np.dot(x,x)/n
    for l in range(1,L+1): var+=2*(1-l/(L+1))*(np.dot(x[l:],x[:-l])/n)
    se=np.sqrt(max(var,1e-12)/n); t=mu/se; return float(t),float(1-ss.t.cdf(t,n-1))


def bootp05(wk,nb=2000):
    if len(wk)<8: return None
    rng=np.random.default_rng(0); bl=4; comps=[]
    for _ in range(nb):
        pick=[]
        while len(pick)<len(wk):
            st=int(rng.integers(0,len(wk))); pick+=list(range(st,min(st+bl,len(wk))))
        pick=np.array(pick[:len(wk)]); comps.append(float(np.prod(1+wk[pick]/100)-1)*100)
    return round(float(np.percentile(comps,5)),1)


def topk_share(wk,k):
    order=np.argsort(-wk); tot=np.exp(np.sum(np.log1p(wk/100)))-1
    if tot<0.10: return None
    keep=np.ones(len(wk),bool); keep[order[:k]]=False
    return round((1-(np.exp(np.sum(np.log1p(wk[keep]/100)))-1)/tot)*100,1)


def dsr(wk,nt):
    if len(wk)<8 or np.std(wk)<1e-9: return None
    sr=np.mean(wk)/np.std(wk); n=len(wk); g3=ss.skew(wk); g4=ss.kurtosis(wk,fisher=False); emc=0.5772; z=ss.norm.ppf
    sr0=np.sqrt(1.0/n)*((1-emc)*z(1-1.0/nt)+emc*z(1-1.0/(nt*np.e))); denom=np.sqrt(max(1e-9,1-g3*sr+(g4-1)/4.0*sr**2))
    return round(float(ss.norm.cdf(((sr-sr0)*np.sqrt(n-1))/denom)),3)


def maxdd(stream):
    eq=np.cumprod(1+stream); return float(np.min((eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq))*100) if len(eq)>1 else 0.0


def main():
    out={"check":"MA_FAMILY_robustness_battery","config":"breadth-gated de-leaked, inv-vol cap33, 14d/7d ROLLING, 1d, next-open, prod 0.0028RT","ts":datetime.now().strftime("%Y%m%d_%H%M%S"),"windows":{}}
    print("="*104); print("AUDIT: MA FAMILY (trend-rider) robustness battery -- the actual product, NOT the momentum/timing magic-pill"); print("="*104)
    for window in ["OOS-2024","OOS-2025"]:
        f=R.build(window); wm=f["mask"]; idx=f["idx"]; wall=R.WIN[window][0]
        # BTC buy-hold over the window
        rec=_load_asset("BTCUSDT","1d",wall); bh=None; bhdd=None
        if rec is not None:
            d,o,h,l,c=rec; bs=pd.Series(np.asarray(c,float),index=d).reindex(idx).to_numpy()[wm]; bs=bs[np.isfinite(bs)]
            if len(bs)>1: bh=float(bs[-1]/bs[0]-1)*100; beq=bs/bs[0]; bhdd=float(np.min((beq-np.maximum.accumulate(beq))/np.maximum.accumulate(beq))*100)
        out["windows"][window]={"btc_bh":round(bh,1) if bh is not None else None,"btc_bh_maxdd":round(bhdd,1) if bhdd is not None else None,"engines":{}}
        eng={"CONV":f["conv"]/len(TYPES)}; eng.update({ma:f["typ"][ma].astype(float) for ma in TYPES})
        print(f"\n  {window}  ({R.WIN[window][1]}..{R.WIN[window][2]})   BTC-BH {('%+.0f%%'%bh) if bh is not None else 'n/a'} (maxDD {('%+.0f%%'%bhdd) if bhdd is not None else 'n/a'})")
        print(f"    {'engine':<7}{'comp%':>8}{'maxDD':>7}{'deploy':>8}{'NW_p':>7}{'p05':>8}{'top2wk':>8}{'%wk>0':>7}{'DSRn24':>8}  vs BTC-BH")
        print("    "+"-"*86)
        for nm,hm in eng.items():
            W=R.W_invvol(hm,f["vol"],f["g"],CAP); net,Wl=R.net_of(W,f["roo"]); n=net[wm]
            wk=np.array([float(np.prod(1+n[i:i+SB])-1)*100 for i in range(0,len(n)-SB+1,SB)])
            comp=float(np.prod(1+n)-1)*100; dd=maxdd(n); dep=float(np.mean(Wl.sum(axis=1)[wm]>1e-9))
            t_nw,p_nw=nw_p(wk); p05=bootp05(wk); t2=topk_share(wk,2); pos=float(np.mean(wk>0)); d24=dsr(wk,24)
            beat=("ret+DD" if (bh is not None and comp>bh and dd>bhdd) else ("DD-only" if (bhdd is not None and dd>bhdd) else "neither"))
            out["windows"][window]["engines"][nm]={"comp":round(comp,1),"maxdd":round(dd,1),"deploy":round(dep,2),"nw_p":round(p_nw,4),"p05":p05,"top2_wk":t2,"pct_pos":round(pos,2),"dsr_n24":d24,"vs_btc_bh":beat}
            print(f"    {nm:<7}{comp:>+7.0f}%{dd:>+7.0f}{dep*100:>7.0f}%{p_nw:>7.3f}{(p05 if p05 is not None else 0):>+7.0f}%{(t2 if t2 is not None else 0):>7.0f}%{pos*100:>6.0f}%{(d24 if d24 is not None else 0):>8.3f}  {beat}")
    # verdict heuristic on CONV (the deployable book)
    c24=out["windows"]["OOS-2024"]["engines"]["CONV"]; c25=out["windows"]["OOS-2025"]["engines"]["CONV"]
    print(f"\n  CONV book read: OOS-2024 comp {c24['comp']:+.0f}% (NW p {c24['nw_p']}, p05 {c24['p05']}%, top2wk {c24['top2_wk']}%, {c24['pct_pos']*100:.0f}% wk>0)")
    print(f"                  OOS-2025 comp {c25['comp']:+.0f}% (NW p {c25['nw_p']}, p05 {c25['p05']}%, {c25['pct_pos']*100:.0f}% wk>0)")
    print(f"    de-risking: OOS-2024 maxDD {c24['maxdd']:+.0f}% vs BTC {out['windows']['OOS-2024']['btc_bh_maxdd']}% | OOS-2025 {c25['maxdd']:+.0f}% vs BTC {out['windows']['OOS-2025']['btc_bh_maxdd']}%")
    with open(HERE/"strat_sota_ma_family_robust.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\nSaved {HERE/'strat_sota_ma_family_robust.json'}")


if __name__=="__main__":
    main()
