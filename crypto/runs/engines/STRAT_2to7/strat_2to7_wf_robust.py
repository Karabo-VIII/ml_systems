"""STRAT 2-7%/wk -- Cycle 14: WF-timing ROBUSTNESS (address the feature-design-overfit caveat).

The walk-forward timing win's deepest caveat: I chose the market-state features while iterating on these windows.
If the signal survives with FEWER features + a SIMPLER model + across thresholds, the overfit concern shrinks
(simpler = less room to overfit; survival across specs = not a single lucky config). Test feature-subsets x
{GBM, logistic} x P-threshold{.5,.55,.6}, full-OOS compound + beats-shuffle-null. Rule-gate baseline = +57%.
HONEST: even surviving, feature CHOICE was data-informed -> fresh data is the final arbiter.
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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BPD=M.BPD; SLc=7*BPD; CUT=pd.Timestamp("2024-05-15")
# feature order from W.features_labels: [btc_mom, breadth, disp, btc_trend, mkt_dd, brd_rise]
SUBSETS={"full6":[0,1,2,3,4,5],"core3":[1,0,4],"core2":[1,0],"breadth+dd":[1,4]}


def wf_P(X,lab,ewfwd,idx,T,cols,model):
    P=np.full(T,np.nan)
    for q in pd.period_range("2024Q3","2026Q3",freq="Q"):
        q0=q.start_time; q1=q.end_time
        test=[t for t in range(T) if q0<=idx[t]<=q1]
        if not test: continue
        tr=[t for t in range(T-SLc) if idx[t]+pd.Timedelta(days=7)<q0 and np.isfinite(ewfwd[t])]
        if len(tr)<200: continue
        Xtr=X[np.ix_(tr,cols)];
        if model=="gbm":
            m=GradientBoostingClassifier(n_estimators=120,max_depth=3,learning_rate=0.05,subsample=0.7,random_state=0).fit(Xtr,lab[tr])
            for t in test: P[t]=m.predict_proba(X[t:t+1,cols])[:,1][0]
        else:
            sc=StandardScaler().fit(Xtr); m=LogisticRegression(max_iter=500).fit(sc.transform(Xtr),lab[tr])
            for t in test: P[t]=m.predict_proba(sc.transform(X[t:t+1,cols]))[:,1][0]
    return P


def book_comp(f, gate, mask):
    f2=dict(f); f2["elig"]=(f["conv"]>0)&(gate[:,None]>0); E.K=8
    net,Wl=E.book_with_exits(f2,f2["scores"]["mom14"],sl=None,tp=None,trail=None); E.K=5
    n=net[mask]; return float(np.prod(1+n)-1)*100, net, Wl


def main():
    f=M.build("FWD"); idx=f["idx"]; brd=f["brd"]; T=f["T"]
    X,lab,ewfwd,_,_=W.features_labels(f)
    g_brd=M.hyst(brd,0.5,0.4); oos=(idx>=CUT)
    rule_comp,_,_=book_comp(f,g_brd,oos)
    rng=np.random.default_rng(2)
    out={"check":"STRAT_2to7_cycle14_wf_robust","rule_baseline_comp":round(rule_comp,1),"ts":datetime.now().strftime("%Y%m%d_%H%M%S"),"variants":{}}
    print("="*92); print(f"WF-TIMING ROBUSTNESS -- full-OOS compound vs rule baseline ({rule_comp:+.0f}%) + beats-shuffle-null"); print("="*92)
    print(f"  {'feats':<11}{'model':<7}{'thr':>5}{'comp%':>8}{'null_p90':>9}{'beats_null':>11}")
    wins=0; tot=0
    for sub,cols in SUBSETS.items():
        for model in ["gbm","logit"]:
            P=wf_P(X,lab,ewfwd,idx,T,cols,model)
            for thr in [0.5,0.55,0.6]:
                g=(g_brd&(np.nan_to_num(P,nan=0.0)>thr)).astype(int)
                comp,_,_=book_comp(f,g,oos)
                # shuffle null (10x)
                nulls=[]
                for s in range(10):
                    gn=(g_brd&(np.nan_to_num(rng.permutation(P),nan=0.0)>thr)).astype(int)
                    nulls.append(book_comp(f,gn,oos)[0])
                p90=float(np.percentile(nulls,90)); beats=comp>p90 and comp>rule_comp
                wins+=int(beats); tot+=1
                out["variants"][f"{sub}_{model}_{thr}"]={"comp":round(comp,1),"null_p90":round(p90,1),"beats":beats}
                print(f"  {sub:<11}{model:<7}{thr:>5}{comp:>+7.0f}%{p90:>+8.0f}%{'YES' if beats else 'no':>11}")
    print(f"\n  -> {wins}/{tot} variants beat BOTH the rule baseline AND their shuffle-null")
    out["robust_wins"]=f"{wins}/{tot}"
    with open(HERE/"strat_2to7_wf_robust.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"Saved {HERE/'strat_2to7_wf_robust.json'}")


if __name__=="__main__":
    main()
