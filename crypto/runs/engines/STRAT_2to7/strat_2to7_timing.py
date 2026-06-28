"""STRAT 2-7%/wk -- Cycle 10: LEARNED DEPLOY-TIMING gate (the temporal signal from cy8/9) vs the breadth gate.

Cy9: the predictable signal is TEMPORAL (when-to-deploy), not which-name. Test a learned gate: P(good week |
market-state at t), deploy when P>thr, vs the rule-based breadth gate. Label[t]=1 if EW-eligible forward-7d ret>0.
Features (market-state, causal): btc_mom14, breadth, dispersion, btc_trend, mkt_drawdown, breadth_rising. Train
<=2024-05-15, test OOS. Measure deployed-week median + compound + deploy-fraction on OOS24 + FWD vs breadth gate.
DEAD-LIST PRIOR (D85, config-selector REFUTED): learned regime-timing fails OOS -- test directly, report honestly.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import strat_2to7_movers as M, strat_2to7_exits as E
from sklearn.ensemble import GradientBoostingClassifier

BPD = M.BPD; K = M.K; SLc = 7*BPD; SW = M.SW; CUT = pd.Timestamp("2024-05-15")


def main():
    f = M.build("FWD")
    idx=f["idx"]; elig0=f["conv"]>0; brd=f["brd"]; T=f["T"]; A=f["A"]; cdf=f["cdf"]; roo=f["roo"]; conv=f["conv"]
    C=cdf.to_numpy(); ret=cdf.pct_change().fillna(0.0).to_numpy()
    mom14=f["scores"]["mom14"]
    jb=list(cdf.columns).index("BTCUSDT") if "BTCUSDT" in cdf.columns else 0
    btc_mom=mom14[:,jb]
    bc=cdf.iloc[:,jb]; bsma=bc.rolling(SW,min_periods=SW//2).mean().to_numpy()
    btc_trend=(bc.to_numpy()>bsma).astype(float)
    disp=np.nanstd(ret,axis=1)
    idxlvl=pd.Series(np.nanmean(ret,axis=1)).add(1).cumprod().to_numpy()
    roll=pd.Series(idxlvl).rolling(60*BPD,min_periods=20).max().to_numpy(); mkt_dd=idxlvl/np.where(roll>0,roll,np.nan)-1.0
    brd_rise=np.zeros(T)
    for t in range(7*BPD,T): brd_rise[t]=1.0 if (np.isfinite(brd[t]) and np.isfinite(brd[t-7*BPD]) and brd[t]>brd[t-7*BPD]) else 0.0
    Xall=np.column_stack([np.nan_to_num(btc_mom),np.nan_to_num(brd),np.nan_to_num(disp),btc_trend,np.nan_to_num(mkt_dd),brd_rise])
    # label: EW-eligible forward-7d ret > 0
    ewfwd=np.full(T,np.nan)
    for t in range(T-SLc):
        e=np.where(elig0[t])[0]
        if len(e)==0: continue
        ewfwd[t]=np.mean(np.prod(1+roo[t:t+SLc],axis=0)[e]-1.0)
    lab=(ewfwd>0).astype(int)
    tr=[t for t in range(T-SLc) if idx[t]<CUT and np.isfinite(ewfwd[t])]
    mdl=GradientBoostingClassifier(n_estimators=120,max_depth=3,learning_rate=0.05,subsample=0.7).fit(Xall[tr],lab[tr])
    Pgood=mdl.predict_proba(Xall)[:,1]
    out={"check":"STRAT_2to7_cycle10_timing","ts":datetime.now().strftime("%Y%m%d_%H%M%S"),"windows":{}}
    g_brd=M.hyst(brd,0.5,0.4)
    print("="*94)
    print("LEARNED DEPLOY-TIMING gate (P(good week)|market-state) vs breadth gate -- mom14 top-5 + TP5")
    print("="*94)
    for wnm, wm in [("OOS24",(idx>=CUT)&(idx<pd.Timestamp("2025-01-01"))),("FWD",(idx>=pd.Timestamp("2025-04-01"))&(idx<pd.Timestamp("2026-07-01")))]:
        out["windows"][wnm]={}
        print(f"\n  {wnm}:  {'gate':<20}{'dep_frac':>9}{'median':>8}{'mean':>7}{'hit>=2':>8}{'in[2,7]':>9}{'comp':>8}")
        gates=[("breadth (rule)",g_brd)]
        for thr in [0.5,0.55,0.6]:
            gates.append((f"learnedP>{thr}",(Pgood>thr).astype(int)))
            gates.append((f"breadth&P>{thr}",(g_brd&(Pgood>thr)).astype(int)))
        for nm,g in gates:
            f2=dict(f); f2["elig"]=(conv>0)&(g[:,None]>0)
            net,Wl=E.book_with_exits(f2,f2["scores"]["mom14"],sl=None,tp=0.05,trail=None)
            d=E.wdist(net,Wl,wm); df=round(float(np.mean(Wl.sum(axis=1)[wm]>1e-9)),3)
            if d is None:
                print(f"       {nm:<20}{df*100:>8.0f}%  (no dep)"); continue
            out["windows"][wnm][nm]={**d,"deploy_frac":df}
            print(f"       {nm:<20}{df*100:>8.0f}%{d['median']:>+8.2f}{d['mean']:>+7.2f}{d['hit2']*100:>7.0f}%{d['in27']*100:>8.0f}%{d['comp']:>+7.0f}%")
    with open(HERE/"strat_2to7_timing.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\nSaved {HERE/'strat_2to7_timing.json'}")


if __name__=="__main__":
    main()
