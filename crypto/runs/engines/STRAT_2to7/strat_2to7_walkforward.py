"""STRAT 2-7%/wk -- Cycle 12: WALK-FORWARD learned deploy-timing (the lead's best, most-honest shot + the deployable).

cy10/11: learned timing gate is real-but-fragile on a one-shot train/test. The proper test = WALK-FORWARD: retrain
the timing model each quarter on ALL data up to that quarter (never peeking), stitch the out-of-sample P, gate the
momentum book by breadth & P>0.55, run the FULL continuous OOS (2024-05-15..2026-05) $ track + deployed-week dist.
Compare to the breadth-rule gate. If walk-forward holds -> real; if it falls apart -> ceiling firm. Emits the live book.
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

BPD=M.BPD; SLc=7*BPD; SW=M.SW; CUT=pd.Timestamp("2024-05-15"); START_CAP=10000.0


def features_labels(f):
    idx=f["idx"]; elig0=f["conv"]>0; brd=f["brd"]; T=f["T"]; cdf=f["cdf"]; roo=f["roo"]
    ret=cdf.pct_change().fillna(0.0).to_numpy(); mom14=f["scores"]["mom14"]
    jb=list(cdf.columns).index("BTCUSDT") if "BTCUSDT" in cdf.columns else 0
    bc=cdf.iloc[:,jb]; bsma=bc.rolling(SW,min_periods=SW//2).mean().to_numpy()
    btc_trend=(bc.to_numpy()>bsma).astype(float); disp=np.nanstd(ret,axis=1)
    idxlvl=pd.Series(np.nanmean(ret,axis=1)).add(1).cumprod().to_numpy()
    roll=pd.Series(idxlvl).rolling(60*BPD,min_periods=20).max().to_numpy(); mkt_dd=idxlvl/np.where(roll>0,roll,np.nan)-1.0
    brd_rise=np.zeros(T)
    for t in range(7*BPD,T): brd_rise[t]=float(np.isfinite(brd[t]) and np.isfinite(brd[t-7*BPD]) and brd[t]>brd[t-7*BPD])
    X=np.column_stack([np.nan_to_num(mom14[:,jb]),np.nan_to_num(brd),np.nan_to_num(disp),btc_trend,np.nan_to_num(mkt_dd),brd_rise])
    ewfwd=np.full(T,np.nan)
    for t in range(T-SLc):
        e=np.where(elig0[t])[0]
        if len(e): ewfwd[t]=np.mean(np.prod(1+roo[t:t+SLc],axis=0)[e]-1.0)
    return X,(ewfwd>0).astype(int),ewfwd,idx,T


def walkforward_P(X,lab,ewfwd,idx,T,seed=0):
    """Quarterly retrain on all data up to the quarter start (label must be fully realized: t+7d < quarter start).
    seed: random_state for the GBM (subsample=0.7 is stochastic -> MUST be seeded for reproducible/frozen results)."""
    P=np.full(T,np.nan)
    qs=pd.period_range(start="2024Q3", end="2026Q3", freq="Q")
    for q in qs:
        q0=q.start_time; q1=q.end_time
        test=[t for t in range(T) if q0<=idx[t]<=q1]
        if not test: continue
        # train: labels realized strictly before q0 (t + 7d < q0).
        # PURGE: the t+7d<q0 rule IS the walk-forward purge -- the train label window [t,t+7d] closes before the test
        # quarter opens, so no train-outcome overlaps any test-outcome. (Distinct from the WM-pipeline PURGE_GAP_BARS=400,
        # which is for normalization-split leakage; do NOT shrink this boundary in a future edit.)
        tr=[t for t in range(T-SLc) if idx[t]+pd.Timedelta(days=7) < q0 and np.isfinite(ewfwd[t])]
        if len(tr)<200: continue
        mdl=GradientBoostingClassifier(n_estimators=120,max_depth=3,learning_rate=0.05,subsample=0.7,random_state=seed).fit(X[tr],lab[tr])
        for t in test: P[t]=mdl.predict_proba(X[t:t+1])[:,1][0]
    return P


def run_book(f, gate, K, tp, mask):
    f2=dict(f); f2["elig"]=(f["conv"]>0)&(gate[:,None]>0)
    import strat_2to7_exits as EE; EE.K=K
    net,Wl=EE.book_with_exits(f2,f2["scores"]["mom14"],sl=None,tp=tp,trail=None)
    EE.K=5
    return net,Wl


def stats(net, Wl, mask):
    nm=net[mask]
    if len(nm)==0: return None
    eq=START_CAP*np.cumprod(1+nm)
    dd=float(np.min((eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq)))*100
    g=Wl.sum(axis=1)[mask]; wk=[]; i=0
    while i+SLc<=len(nm):
        if np.mean(g[i:i+SLc]>1e-9)>0.5: wk.append(float(np.prod(1+nm[i:i+SLc])-1)*100)
        i+=SLc
    wk=np.array(wk) if wk else np.array([0.0])
    return {"final":round(float(eq[-1]),0),"comp":round(float(eq[-1]/START_CAP-1)*100,1),"maxdd":round(dd,1),
            "dep_wks":int(len(wk)),"wk_median":round(float(np.median(wk)),2),"wk_mean":round(float(np.mean(wk)),2),
            "in27":round(float(np.mean((wk>=2)&(wk<=7))),3),"deploy_frac":round(float(np.mean(g>1e-9)),3)}


def main():
    f=M.build("FWD"); idx=f["idx"]; brd=f["brd"]
    X,lab,ewfwd,idxi,T=features_labels(f)
    Pwf=walkforward_P(X,lab,ewfwd,idx,T)
    g_brd=M.hyst(brd,0.5,0.4)
    g_wf=(g_brd & (np.nan_to_num(Pwf,nan=0.0)>0.55)).astype(int)
    oos=(idx>=CUT)
    out={"check":"STRAT_2to7_cycle12_walkforward","ts":datetime.now().strftime("%Y%m%d_%H%M%S"),"configs":{}}
    print("="*94)
    print("WALK-FORWARD learned-timing (quarterly retrain) -- full OOS 2024-05-15..2026-05, $10k")
    print("="*94)
    print(f"  {'config':<34}{'final$':>10}{'comp':>8}{'maxDD':>7}{'depwk':>6}{'wkMed':>7}{'in27':>7}{'depFrac':>8}")
    print("  "+"-"*88)
    for nm, gate, Kk, tp in [
        ("breadth-rule + mom8 (wealth base)", g_brd, 8, None),
        ("WF-timing + mom8 (wealth)", g_wf, 8, None),
        ("breadth-rule + mom5 + TP5 (consist)", g_brd, 5, 0.05),
        ("WF-timing + mom5 + TP5 (consist)", g_wf, 5, 0.05)]:
        net,Wl=run_book(f,gate,Kk,tp,oos); s=stats(net,Wl,oos)
        out["configs"][nm]=s
        print(f"  {nm:<34}{s['final']:>10.0f}{s['comp']:>+7.0f}%{s['maxdd']:>+7.0f}{s['dep_wks']:>6}{s['wk_median']:>+7.2f}{s['in27']*100:>6.0f}%{s['deploy_frac']*100:>7.0f}%")
    # live gate state
    last=T-1; live_on = g_wf[last]>0
    out["live_as_of"]=str(pd.Timestamp(idx[last]).date()); out["live_gate"]="DEPLOY" if live_on else "CASH"; out["live_Pwf"]=round(float(np.nan_to_num(Pwf[last])),3)
    print(f"\n  LIVE gate as of {out['live_as_of']}: {out['live_gate']}  (walk-forward P(good week)={out['live_Pwf']})")
    with open(HERE/"strat_2to7_walkforward.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"Saved {HERE/'strat_2to7_walkforward.json'}")


if __name__=="__main__":
    main()
