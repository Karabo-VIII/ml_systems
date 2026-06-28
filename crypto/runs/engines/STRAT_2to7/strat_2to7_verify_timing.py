"""STRAT 2-7%/wk -- Cycle 11: ADVERSARIALLY VERIFY the learned deploy-timing gate (cy10) -- real or campaign-overfit?

cy10: breadth&P(good-week)>0.55 gave FWD deployed median +6.83/comp +109 (vs breadth-rule -0.36/-12). Contradicts
dead-list D85 (learned regime-timing refuted) + thresholds were chosen on the same OOS windows -> overfit risk HIGH.
Adversarial checks:
  1. TEMPORAL-SHUFFLE NULL: shuffle Pgood in time (break the timing signal) -> if the gain vanishes, signal is real.
  2. LUMPINESS: is the FWD +109% comp driven by 1-2 weeks (fragile) or broad?
  3. vs BEST RULE GATE (breadth&btc&notDD, cy4) -- does the LEARNED gate beat the best non-learned gate?
  4. FWD sub-period halves -- does it help both halves or one lucky stretch?
HONEST CAVEAT (stated regardless): OOS24/FWD are campaign-stale (optimized against over 10 cycles); only a PRISTINE
forward test settles it. mom14 top-5 + TP5, breadth base.
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

BPD=M.BPD; K=M.K; SLc=7*BPD; SW=M.SW; CUT=pd.Timestamp("2024-05-15")


def fit_pgood(f):
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
    lab=(ewfwd>0).astype(int); tr=[t for t in range(T-SLc) if idx[t]<CUT and np.isfinite(ewfwd[t])]
    mdl=GradientBoostingClassifier(n_estimators=120,max_depth=3,learning_rate=0.05,subsample=0.7).fit(X[tr],lab[tr])
    return mdl.predict_proba(X)[:,1], btc_trend, np.nan_to_num(mkt_dd)


def run(f, gate, wm):
    f2=dict(f); f2["elig"]=(f["conv"]>0)&(gate[:,None]>0)
    net,Wl=E.book_with_exits(f2,f2["scores"]["mom14"],sl=None,tp=0.05,trail=None)
    return net, Wl, E.wdist(net,Wl,wm)


def main():
    f=M.build("FWD"); idx=f["idx"]; brd=f["brd"]
    Pgood, btc_trend, mkt_dd = fit_pgood(f)
    g_brd=M.hyst(brd,0.5,0.4)
    fwd=(idx>=pd.Timestamp("2025-04-01"))&(idx<pd.Timestamp("2026-07-01"))
    out={"check":"STRAT_2to7_cycle11_verify_timing","ts":datetime.now().strftime("%Y%m%d_%H%M%S")}
    print("="*90); print("VERIFY learned deploy-timing gate (FWD focus) -- real or campaign-overfit?"); print("="*90)
    real=(g_brd&(Pgood>0.55)).astype(int)
    # 1. temporal-shuffle null (20 shuffles)
    rng=np.random.default_rng(0); nulls=[]
    for s in range(20):
        Psh=rng.permutation(Pgood); gnull=(g_brd&(Psh>0.55)).astype(int)
        d=run(f,gnull,fwd)[2]
        if d: nulls.append(d["median"])
    dr=run(f,real,fwd)[2]
    print(f"  1. REAL breadth&P>0.55 FWD median = {dr['median']:+.2f}  comp {dr['comp']:+.0f}%  in[2,7] {dr['in27']:.0%}")
    print(f"     SHUFFLE-NULL median: mean {np.mean(nulls):+.2f}, p90 {np.percentile(nulls,90):+.2f}  -> real {'BEATS null' if dr['median']>np.percentile(nulls,90) else 'within null (OVERFIT)'}")
    out["real_fwd_median"]=dr["median"]; out["null_median_p90"]=round(float(np.percentile(nulls,90)),2)
    # 2. lumpiness of FWD comp
    net,Wl,_=run(f,real,fwd); nm=net[fwd]; nm=nm[Wl.sum(axis=1)[fwd]>1e-9]
    wk=[]; i=0; g=Wl.sum(axis=1)[fwd]; nf=net[fwd]
    j=0
    while j+SLc<=len(nf):
        if np.mean(g[j:j+SLc]>1e-9)>0.5: wk.append(float(np.prod(1+nf[j:j+SLc])-1)*100)
        j+=SLc
    wk=np.array(sorted(wk,reverse=True))
    tot=float(np.sum(wk[wk>0])); top2=float(np.sum(wk[:2]))/tot if tot>0 else 0
    print(f"  2. LUMPINESS: {len(wk)} deployed wks; top-2 weeks = {top2:.0%} of positive sum; weeks: {[round(x,1) for x in wk[:5]]}...")
    out["fwd_top2_share"]=round(top2,2)
    # 3. vs best RULE gate (breadth & btc_trend & not-deep-DD)
    g_rule=(g_brd & btc_trend.astype(int) & (mkt_dd>-0.12).astype(int)).astype(int)
    dr_rule=run(f,g_rule,fwd)[2]
    print(f"  3. vs BEST RULE gate (breadth&btc&notDD): FWD median {dr_rule['median']:+.2f} comp {dr_rule['comp']:+.0f}%  | learned {dr['median']:+.2f}/{dr['comp']:+.0f}%")
    out["rule_gate_fwd_median"]=dr_rule["median"]
    # 4. FWD halves
    fi=np.where(fwd)[0]; mid=fi[len(fi)//2]
    h1=fwd.copy(); h1[mid:]=False; h2=fwd.copy(); h2[:mid]=False
    d1=run(f,real,h1)[2]; d2=run(f,real,h2)[2]
    print(f"  4. FWD halves: H1 median {d1['median'] if d1 else 'na':}  H2 median {d2['median'] if d2 else 'na':}  -> {'both+' if d1 and d2 and d1['median']>0 and d2['median']>0 else 'one-sided'}")
    out["fwd_half_medians"]=[d1['median'] if d1 else None, d2['median'] if d2 else None]
    with open(HERE/"strat_2to7_verify_timing.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\nSaved {HERE/'strat_2to7_verify_timing.json'}")


if __name__=="__main__":
    main()
