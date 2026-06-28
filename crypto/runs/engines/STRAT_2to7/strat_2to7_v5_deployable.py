"""STRAT 2-7%/wk -- V5 DEPLOYABLE: learned deploy-timing gate (WF) + momentum top-8 + breadth. The campaign's win.

Validated (cy12-14): walk-forward + shuffle-null + feature-robust (core2 survives) + nonlinear (reconciles D85).
The engine deploys ~13% of weeks (cash 87%) into the genuinely-favorable weeks where deployed-week median ~+5%/wk
(in the 2-7% band), ~3x wealth vs the rule gate at lower drawdown. This emits: full-OOS track + the LIVE weekly
candidate book (names+weights or CASH). PROVISIONAL: feature-choice was data-informed -> validate live/forward.
gate = breadth & GBM-P(good-week|market-state)>0.55, quarterly-retrained. mom14 top-8 inv-vol, u100, 1d, 14d/7d, prod.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import strat_2to7_movers as M, strat_2to7_exits as E, strat_2to7_walkforward as W

BPD=M.BPD; CUT=pd.Timestamp("2024-05-15"); START=10000.0


def main():
    f=M.build("FWD"); idx=f["idx"]; brd=f["brd"]; T=f["T"]; cols=list(f["cdf"].columns)
    X,lab,ewfwd,_,_=W.features_labels(f)
    Pwf=W.walkforward_P(X,lab,ewfwd,idx,T)
    gate=(M.hyst(brd,0.5,0.4)&(np.nan_to_num(Pwf,nan=0.0)>0.55)).astype(int)
    f2=dict(f); f2["elig"]=(f["conv"]>0)&(gate[:,None]>0); E.K=8
    net,Wl=E.book_with_exits(f2,f2["scores"]["mom14"],sl=None,tp=None,trail=None)
    W_=E.book_with_exits._lastW; E.K=5
    oos=(idx>=CUT); n=net[oos]; eq=START*np.cumprod(1+n)
    dd=float(np.min((eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq)))*100
    g=Wl.sum(axis=1)[oos]; wk=[]; i=0
    while i+7*BPD<=len(n):
        if np.mean(g[i:i+7*BPD]>1e-9)>0.5: wk.append(float(np.prod(1+n[i:i+7*BPD])-1)*100)
        i+=7*BPD
    wk=np.array(wk)
    last=T-1; livew=W_[last]; live=sorted(((cols[j],round(float(livew[j]),3)) for j in np.where(livew>1e-6)[0]),key=lambda x:-x[1])
    out={"check":"STRAT_2to7_V5_DEPLOYABLE","spec":"breadth & WF-GBM-P(good-week)>0.55 gate + mom14 top-8 invvol, u100 1d 14d/7d prod",
         "oos_final":round(float(eq[-1]),0),"oos_comp":round(float(eq[-1]/START-1)*100,1),"oos_maxdd":round(dd,1),
         "deployed_weeks":int(len(wk)),"deployed_wk_median":round(float(np.median(wk)),2) if len(wk) else None,
         "deployed_in27":round(float(np.mean((wk>=2)&(wk<=7))),3) if len(wk) else None,
         "live_as_of":str(pd.Timestamp(idx[last]).date()),"live_gate":"DEPLOY" if live else "CASH",
         "live_P":round(float(np.nan_to_num(Pwf[last])),3),"live_candidates":live,"ts":datetime.now().strftime("%Y%m%d_%H%M%S")}
    print("="*82); print("STRAT 2-7%/wk V5 DEPLOYABLE -- learned deploy-timing + momentum top-8 (full OOS, $10k)"); print("="*82)
    print(f"  OOS 2024-05-15..{out['live_as_of']}: ${eq[-1]:,.0f}  compound {out['oos_comp']:+.0f}%  maxDD {dd:+.0f}%")
    print(f"  deployed weeks={len(wk)} ({'%.0f'%(np.mean(g>1e-9)*100)}%)  median {out['deployed_wk_median']:+.2f}%/wk  in[2,7] {out['deployed_in27']*100 if out['deployed_in27'] else 0:.0f}%")
    print(f"\n  *** LIVE BOOK as of {out['live_as_of']}: {out['live_gate']} (P(good-week)={out['live_P']}) ***")
    if live:
        for nm,wt in live: print(f"      {nm:<14}{wt:6.1%}")
    else:
        print("      [CASH] -- learned timing says NOT a favorable week (hold cash)")
    with open(HERE/"strat_2to7_v5_deployable.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\nSaved {HERE/'strat_2to7_v5_deployable.json'}")


if __name__=="__main__":
    main()
