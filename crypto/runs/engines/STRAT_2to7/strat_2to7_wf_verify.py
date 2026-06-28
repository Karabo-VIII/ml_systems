"""STRAT 2-7%/wk -- Cycle 13: VERIFY the walk-forward timing result (+181%) -- per-quarter + shuffle-null.

cy12 walk-forward (quarterly retrain) HELD: WF-timing+mom8 +181% (vs rule +57%), deploy 13%, deployed median +5.1
(consist config). Contradicts dead-list D85 -> demands maximal rigor. Checks:
  1. PER-QUARTER: is the +181% spread across quarters or one lucky stretch? (deploy + return per quarter)
  2. SHUFFLE-NULL (20x): shuffle the walk-forward P in time -> does +181% collapse to ~rule-gate level?
If spread across quarters AND beats shuffle-null -> robust (pending fresh-data); if one-quarter or within-null -> artifact.
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
import strat_2to7_walkforward as W

BPD=M.BPD; CUT=pd.Timestamp("2024-05-15")


def book_net(f, gate, K):
    f2=dict(f); f2["elig"]=(f["conv"]>0)&(gate[:,None]>0)
    E.K=K; net,Wl=E.book_with_exits(f2,f2["scores"]["mom14"],sl=None,tp=None,trail=None); E.K=5
    return net,Wl


def main():
    f=M.build("FWD"); idx=f["idx"]; brd=f["brd"]
    X,lab,ewfwd,_,T=W.features_labels(f)
    Pwf=W.walkforward_P(X,lab,ewfwd,idx,T)
    g_brd=M.hyst(brd,0.5,0.4); g_wf=(g_brd&(np.nan_to_num(Pwf,nan=0.0)>0.55)).astype(int)
    oos=(idx>=CUT)
    net,Wl=book_net(f,g_wf,8); netr,Wlr=book_net(f,g_brd,8)
    out={"check":"STRAT_2to7_cycle13_wf_verify","ts":datetime.now().strftime("%Y%m%d_%H%M%S")}
    # 1. per-quarter
    s=pd.Series(net,index=pd.DatetimeIndex(idx)); sr=pd.Series(netr,index=pd.DatetimeIndex(idx))
    gser=pd.Series(Wl.sum(axis=1),index=pd.DatetimeIndex(idx))
    print("="*84); print("WALK-FORWARD VERIFY -- per-quarter (WF-timing+mom8 vs rule+mom8)"); print("="*84)
    print(f"  {'quarter':<9}{'WF_comp%':>9}{'WF_depFrac':>11}{'rule_comp%':>11}")
    pq={}
    for q,gq in s[oos].groupby(s[oos].index.to_period("Q")):
        wfc=float(np.prod(1+gq.to_numpy())-1)*100
        rc=float(np.prod(1+sr[sr.index.to_period("Q")==q].to_numpy())-1)*100
        df=float(np.mean(gser[gser.index.to_period("Q")==q].to_numpy()>1e-9))
        pq[str(q)]={"wf_comp":round(wfc,1),"wf_deploy":round(df,2),"rule_comp":round(rc,1)}
        print(f"  {str(q):<9}{wfc:>+8.1f}%{df*100:>10.0f}%{rc:>+10.1f}%")
    out["per_quarter"]=pq
    pos_q=sum(1 for v in pq.values() if v["wf_comp"]>0); dep_q=sum(1 for v in pq.values() if v["wf_deploy"]>0.02)
    print(f"  -> WF positive quarters: {pos_q}/{len(pq)}; quarters with deploy: {dep_q}/{len(pq)}")
    # 2. shuffle-null
    rng=np.random.default_rng(1); comps=[]
    for sd in range(20):
        Psh=rng.permutation(Pwf); gn=(g_brd&(np.nan_to_num(Psh,nan=0.0)>0.55)).astype(int)
        nn,_=book_net(f,gn,8); comps.append(float(np.prod(1+nn[oos])-1)*100)
    wf_comp=float(np.prod(1+net[oos])-1)*100; rule_comp=float(np.prod(1+netr[oos])-1)*100
    print(f"\n  WF-timing comp {wf_comp:+.0f}%  vs  rule {rule_comp:+.0f}%  vs  shuffle-null mean {np.mean(comps):+.0f}% p90 {np.percentile(comps,90):+.0f}%")
    print(f"  -> WF {'BEATS shuffle-null p90' if wf_comp>np.percentile(comps,90) else 'WITHIN null (timing is artifact)'}")
    out["wf_comp"]=round(wf_comp,1); out["rule_comp"]=round(rule_comp,1); out["null_p90"]=round(float(np.percentile(comps,90)),1)
    out["positive_quarters"]=f"{pos_q}/{len(pq)}"
    with open(HERE/"strat_2to7_wf_verify.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\nSaved {HERE/'strat_2to7_wf_verify.json'}")


if __name__=="__main__":
    main()
