"""SOLVE-FOR-SOTA next-value: FREEZE V5 as a pre-registered spec + honest committed-config significance + live book.

The robustness finding (strat_sota_validate): DSR=0.353 FAILS (n=24) -- and the red-team (wbw77ivx2) showed that even
this is GENEROUS: the real search was hundreds of configs (DSR 0.13 at n=100, 0.07 at n=200). Going FORWARD with ONE
frozen config (no more searching), the only valid test is live/unseen weeks. This harness: (1) FREEZES the exact V5 spec
+ a config hash (pre-registration); (2) reports the committed-config significance HONESTLY -- Newey-West(L3) HAC p
(NOT the iid t, which inflates: the weekly series is autocorrelated ACF1-3 +0.13/+0.19/+0.35 -> iid p=0.046 but
HAC p~=0.087, i.e. NOT significant even before the search penalty) + the search-deflated DSR; (3) emits the LIVE
current-week decision; (4) writes the PRE-REGISTERED live-forward success criterion. No new search, no tuning. long-only spot.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json, hashlib, subprocess
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd
import sklearn
from scipy import stats as ss

HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE.parent/"STRAT_2to7"))
ROOT=Path(__file__).resolve().parents[3]/"src"; sys.path.insert(0,str(ROOT))
import strat_2to7_movers as M, strat_2to7_walkforward as W
from strat.canonical_exits import COST_PROD
BPD=M.BPD; SB=7*BPD; CUT=pd.Timestamp("2024-05-15")


def newey_west_p(wk, L=3):
    """one-sided p for mean(wk)>0 with Newey-West HAC variance (L lags) -- corrects for autocorrelation."""
    n=len(wk); mu=np.mean(wk); x=wk-mu; var=np.dot(x,x)/n
    for l in range(1,L+1): var+=2*(1-l/(L+1))*(np.dot(x[l:],x[:-l])/n)
    se=np.sqrt(max(var,1e-12)/n); t=mu/se; return float(t), float(1-ss.t.cdf(t,n-1))

FROZEN={"engine":"V5 (learned deploy-timing + momentum)","universe":"u100","cadence":"1d","lookback_d":14,"slice_d":7,
        "gate":"breadth hysteresis(0.5/0.4) AND walk-forward GBM P(good-week|market-state) > 0.55, quarterly retrained",
        "selection":"momentum14 top-8, inverse-vol weighted","exits":"confirmation-break + 7d time-cap, no stops/TP",
        "fills":"next-open (open-to-open, lag-1)","cost":"prod 0.0028 RT","leverage":1,"side":"long-only spot",
        "gbm":"GradientBoosting n_est=120 depth=3 lr=0.05 subsample=0.7 random_state=0 (SEEDED for reproducibility)",
        "seed":0,"seed_robustness":"10/10 seeds positive OOS-2024 & OOS-ALL; 7/10 in OOS-2025 bear; magnitude band +-30-40%",
        "features":"core market-state: BTC-momentum, breadth, cross-sectional dispersion, trend, drawdown (nonlinear GBM)"}


def main():
    f=M.build("FWD"); idx=f["idx"]; conv=f["conv"]; vol=f["vol"]; roo=f["roo"]; brd=f["brd"]; T=f["T"]; A=f["A"]; mom=f["scores"]["mom14"]; cols=list(f["cdf"].columns)
    g=M.hyst(brd,0.5,0.4); X,lab,ewfwd,_,_=W.features_labels(f); Pwf=np.nan_to_num(W.walkforward_P(X,lab,ewfwd,idx,T),nan=0.0)
    gate=(conv>0)&(g[:,None]>0)&(Pwf[:,None]>0.55)
    Wt=np.zeros((T,A)); s=np.where(gate&np.isfinite(mom),mom,-np.inf)
    for t in range(T):
        e=np.where(np.isfinite(s[t]))[0]
        if len(e)==0: continue
        k=min(8,len(e)); top=e[np.argsort(-s[t][e])[:k]]; v=vol[t][top]; w=np.where(np.isfinite(v)&(v>0),1.0/v,0.0); Wt[t,top]=w/w.sum() if w.sum()>0 else 1.0/k
    Wl=np.vstack([np.zeros((1,A)),Wt[:-1]]); net=np.nansum(Wl*roo,axis=1)-np.nansum(np.abs(Wt-Wl),axis=1)*(COST_PROD/2)
    oos=(idx>=CUT); n=net[oos]; wk=np.array([float(np.prod(1+n[i:i+SB])-1) for i in range(0,len(n)-SB+1,SB)])
    sr=float(np.mean(wk)/np.std(wk)); nW=len(wk)
    # committed-config significance: iid t INFLATES (autocorrelated weekly returns) -> use Newey-West(L3) HAC. + DSR.
    t_iid=float(np.mean(wk)/(np.std(wk,ddof=1)/np.sqrt(nW))); p_iid=float(1-ss.t.cdf(t_iid,nW-1))
    t_nw,p_nw=newey_west_p(wk,3)
    def dsr(n_trials):
        g3=ss.skew(wk); g4=ss.kurtosis(wk,fisher=False); emc=0.5772; z=ss.norm.ppf
        sr0=np.sqrt(1.0/nW)*((1-emc)*z(1-1.0/n_trials)+emc*z(1-1.0/(n_trials*np.e)))
        denom=np.sqrt(max(1e-9,1-g3*sr+(g4-1)/4.0*sr**2)); return float(ss.norm.cdf(((sr-sr0)*np.sqrt(nW-1))/denom))
    cfg_hash=hashlib.sha256(json.dumps(FROZEN,sort_keys=True).encode()).hexdigest()[:16]
    try: sha=subprocess.run(["git","rev-parse","--short","HEAD"],cwd=str(ROOT.parent),capture_output=True,text=True).stdout.strip()
    except Exception: sha="unknown"
    # live decision
    last=T-1; livew=Wt[last]; live=sorted(((cols[j],round(float(livew[j]),3)) for j in np.where(livew>1e-6)[0]),key=lambda x:-x[1])
    dep=Wl.sum(axis=1)[oos]; deprate=float(np.mean(dep>1e-9))
    out={"check":"STRAT_SOTA_freeze","frozen_spec":FROZEN,"config_hash":cfg_hash,"lineage":{"sklearn":sklearn.__version__,"git_sha":sha,"seed":0},
         "in_sample_oos":{"window":f"2024-05-15..{pd.Timestamp(idx[last]).date()}","compound_pct":round(float(np.prod(1+n)-1)*100,1),
            "n_weeks":nW,"sharpe_ann":round(sr*np.sqrt(52),2),"deploy_rate":round(deprate,3)},
         "significance":{"committed_iid":{"t":round(t_iid,2),"p_one_sided":round(p_iid,4),"note":"INFLATED -- ignores autocorrelation, do not use"},
            "committed_newey_west_L3":{"t":round(t_nw,2),"p_one_sided":round(p_nw,4),"interpretation":"HAC-corrected; if you COMMIT one config & never search again -- borderline-to-NOT-significant"},
            "search_deflated_DSR":{"n24":round(dsr(24),3),"interpretation":"n=24 is a LOWER BOUND; real search ~hundreds -> DSR 0.13@n100, 0.07@n200 (see strat_sota_audit_corrected.json)"},
            "lumpiness":"top-2 deployed weeks ~77% of OOS-ALL compound; component: breadth-alone +57% / Pwf-timing-alone -7% / full +170% -- the BREADTH gate is the robust edge"},
         "live":{"as_of":str(pd.Timestamp(idx[last]).date()),"decision":"DEPLOY" if live else "CASH","P_good_week":round(float(Pwf[last]),3),"book":live},
         "live_forward_protocol_PREREGISTERED":{
            "track":"emit weekly book each Mon 00:00 UTC from config_hash above; log realized 7d next-open return + deploy/cash",
            "min_sample":">= 20 DEPLOYED weeks before any verdict",
            "success_criteria_ALL":["deployed-week median > 0","compound beats 100% cash","compound beats BTC buy-hold measured ONLY over the same deployed weeks","deploy rate in [10%,35%]"],
            "kill_criteria_ANY":["deployed-week median <= 0 over 20+ deployed weeks","compound < cash","max drawdown > 35%"],
            "no_touch":"do NOT re-tune features/threshold/K during live tracking -- that re-introduces the search penalty"},
         "ts":datetime.now().strftime("%Y%m%d_%H%M%S")}
    print("="*88); print("FREEZE V5 -- pre-registered spec + honest committed-config significance + live book"); print("="*88)
    print(f"  config_hash: {cfg_hash}   (frozen; any change = new hash = new pre-registration)")
    print(f"  in-sample OOS {out['in_sample_oos']['window']}: compound {out['in_sample_oos']['compound_pct']:+.0f}%  "
          f"Sharpe(ann) {out['in_sample_oos']['sharpe_ann']}  deploy {deprate*100:.0f}%  ({nW} weeks)")
    print(f"\n  SIGNIFICANCE (honest, autocorrelation-corrected):")
    print(f"    iid t-test:        t={t_iid:.2f}  p={p_iid:.4f}   (INFLATED -- ignores ACF, do not use)")
    print(f"    Newey-West(L3):    t={t_nw:.2f}  p={p_nw:.4f}   <- HAC-corrected committed sig: {'NOT significant' if p_nw>0.05 else 'significant'} even before search penalty")
    print(f"    search-deflated DSR(n=24)={dsr(24):.3f} (LOWER BOUND; real search ~hundreds -> 0.13@n100, 0.07@n200)")
    print(f"    lumpiness: top-2 wks ~77% of compound | component: breadth-alone +57% / Pwf-alone -7% / full +170%")
    print(f"\n  *** LIVE as of {out['live']['as_of']}: {out['live']['decision']} (P(good-week)={out['live']['P_good_week']}) ***")
    if live:
        for nm,wt in live: print(f"      {nm:<14}{wt:6.1%}")
    else: print("      [CASH] -- learned timing says NOT a favorable week")
    print(f"\n  PRE-REGISTERED live-forward gate: >=20 deployed weeks, median>0 & beats cash & beats BTC-BH(on deployed wks) & deploy in [10,35]%. No re-tuning.")
    with open(HERE/"strat_sota_freeze.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\nSaved {HERE/'strat_sota_freeze.json'}")


if __name__=="__main__":
    main()
