"""QUANT REFEREE: is the MA family's de-risking TIMING SKILL or just LOWER EXPOSURE? (exposure-matched benchmark)

The audit compared the MA sleeve to 100% BTC buy-hold and found "lower drawdown". But a book in the market only ~38% of
the time MECHANICALLY has lower DD than 100% BTC -- that is lower EXPOSURE, not timing skill. The correct risk-adjusted
benchmark for a de-risked-beta sleeve is EXPOSURE-MATCHED: the same average exposure, held STATICALLY.

PRE-REGISTERED TEST:
  NULL  H0: the MA family's risk-adjusted performance is NOT better than constant-exposure holding of the same underlying
            (the "de-risking" = just lower exposure; replicable by a static e-BTC/cash mix, no MA needed).
  ALT   H1: the MA family beats the exposure-matched static benchmark on Sharpe AND/OR maxDD (genuine de-risking TIMING).
  Key facts used: scaling a return stream by constant e does NOT change Sharpe (so e-BTC Sharpe == BTC Sharpe) and scales
  maxDD by e. So: (a) Sharpe(sleeve) vs Sharpe(BTC/EW) IS the exposure-matched risk-adjusted test; (b) maxDD(sleeve) vs
  e*maxDD(underlying) isolates whether timing cuts DD MORE than proportional-to-exposure.
  CHURN-IMMUNE: the static benchmark does not reselect per bar (avoids the broken same-exposure-shuffle null).
  Decision: report Sharpe / maxDD / Calmar(comp/|maxDD|) for sleeve vs {BTC-BH, e-BTC, EW-u100-BH, e-EW} + block-boot CI
  on the Sharpe DIFFERENCE (sleeve - e-EW). VERDICT REAL(timing) / ARTIFACT(just-exposure) / AMBIGUOUS.
1d, u100, breadth-gated inv-vol cap33 14d/7d rolling, next-open roo, prod cost. seed-free (deterministic). long-only spot.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd

HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
ROOT=Path(__file__).resolve().parents[3]/"src"; sys.path.insert(0,str(ROOT))
import ma_weekly_replay as R
from strat.ma_wealth_driver import _load_asset
BPD=R.BPD; CAP=R.CAP; TYPES=R.TYPES; ANN=np.sqrt(365)


def stats(stream):
    stream=np.asarray(stream,float); stream=stream[np.isfinite(stream)]
    if len(stream)<2: return {"comp":0.0,"maxdd":0.0,"sharpe":0.0,"calmar":0.0}
    eq=np.cumprod(1+stream); comp=float(eq[-1]-1)*100
    dd=float(np.min((eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq))*100)
    sh=float(np.mean(stream)/np.std(stream)*ANN) if np.std(stream)>1e-12 else 0.0
    cal=float(comp/abs(dd)) if abs(dd)>1e-9 else 0.0
    return {"comp":round(comp,1),"maxdd":round(dd,1),"sharpe":round(sh,2),"calmar":round(cal,2)}


def boot_sharpe_diff(a,b,nb=2000):
    """block-bootstrap CI of Sharpe(a)-Sharpe(b) on paired daily streams (same index)."""
    a=np.asarray(a,float); b=np.asarray(b,float); m=np.isfinite(a)&np.isfinite(b); a=a[m]; b=b[m]; n=len(a)
    if n<20: return None
    rng=np.random.default_rng(0); bl=10; diffs=[]
    def sh(x): return np.mean(x)/np.std(x)*ANN if np.std(x)>1e-12 else 0.0
    for _ in range(nb):
        idx=[]
        while len(idx)<n:
            st=int(rng.integers(0,n)); idx+=list(range(st,min(st+bl,n)))
        idx=np.array(idx[:n]); diffs.append(sh(a[idx])-sh(b[idx]))
    diffs=np.array(diffs); return {"sharpe_diff_mean":round(float(np.mean(diffs)),2),"p05":round(float(np.percentile(diffs,5)),2),"p95":round(float(np.percentile(diffs,95)),2),"frac_gt0":round(float(np.mean(diffs>0)),3)}


def main():
    out={"check":"QUANT_exposure_matched","pre_registered":"H0: de-risking is just lower exposure (no timing skill); H1: sleeve beats exposure-matched static on Sharpe/maxDD","ts":datetime.now().strftime("%Y%m%d_%H%M%S"),"windows":{}}
    print("="*104); print("QUANT REFEREE: MA family -- TIMING SKILL or just LOWER EXPOSURE? (exposure-matched static benchmark)"); print("="*104)
    for window in ["OOS-2024","OOS-2025"]:
        f=R.build(window); wm=f["mask"]; idx=f["idx"]; roo=f["roo"]; wall=R.WIN[window][0]
        W=R.W_invvol(f["conv"]/len(TYPES),f["vol"],f["g"],CAP); net,Wl=R.net_of(W,f["roo"])
        gross=Wl.sum(axis=1); e=float(np.mean(gross[wm]))           # sleeve avg exposure
        ewret=np.nanmean(roo,axis=1)                                # EW-u100 next-open return per bar
        rec=_load_asset("BTCUSDT","1d",wall); btcret=np.zeros(len(idx))
        if rec is not None:
            d,o,h,l,c=rec; bo=pd.Series(np.asarray(o,float),index=d).reindex(idx).to_numpy()
            btcret[:-1]=bo[1:]/np.where(bo[:-1]>0,bo[:-1],np.nan)-1.0; btcret=np.nan_to_num(btcret)
        sl=net[wm]; bt=btcret[wm]; ew=ewret[wm]
        S_sleeve=stats(sl); S_btc=stats(bt); S_ebtc=stats(e*bt); S_ew=stats(ew); S_eew=stats(e*ew)
        diff=boot_sharpe_diff(net[wm], e*ewret[wm])
        out["windows"][window]={"sleeve_avg_exposure":round(e,3),"sleeve":S_sleeve,"BTC_BH":S_btc,"e_BTC_static":S_ebtc,
            "EWu100_BH":S_ew,"e_EW_static":S_eew,"sharpe_diff_sleeve_vs_eEW":diff,
            "maxdd_vs_proportional":{"sleeve":S_sleeve["maxdd"],"e*BTC_maxdd":round(e*S_btc["maxdd"],1),"e*EW_maxdd":round(e*S_ew["maxdd"],1)}}
        print(f"\n  {window}   sleeve avg exposure e = {e:.2f}")
        print(f"    {'book':<18}{'comp%':>8}{'maxDD':>8}{'Sharpe':>8}{'Calmar':>8}")
        print("    "+"-"*50)
        for nm,S in [("MA sleeve (gated)",S_sleeve),("BTC buy-hold 100%",S_btc),(f"e-BTC static ({e:.2f})",S_ebtc),("EW-u100 buy-hold",S_ew),(f"e-EW static ({e:.2f})",S_eew)]:
            print(f"    {nm:<18}{S['comp']:>+7.0f}%{S['maxdd']:>+8.0f}{S['sharpe']:>8.2f}{S['calmar']:>8.2f}")
        print(f"    RISK-ADJ: Sharpe(sleeve) {S_sleeve['sharpe']} vs BTC {S_btc['sharpe']} vs EW-u100 {S_ew['sharpe']}  (e-scaling does NOT change Sharpe)")
        print(f"    DD-TIMING: sleeve maxDD {S_sleeve['maxdd']}% vs proportional e*BTC {e*S_btc['maxdd']:.0f}% / e*EW {e*S_ew['maxdd']:.0f}%  (more-negative-than-proportional => timing does NOT help DD)")
        if diff: print(f"    Sharpe DIFF (sleeve - e-EW): mean {diff['sharpe_diff_mean']}  [p05 {diff['p05']}, p95 {diff['p95']}]  frac>0 {diff['frac_gt0']}  {'sleeve better' if diff['p05']>0 else 'NOT distinguishable (CI spans 0)'}")
    print("\n  VERDICT RULE: if Sharpe(sleeve) ~<= Sharpe(EW-u100) AND sleeve maxDD ~>= e*underlying maxDD AND Sharpe-diff CI spans 0")
    print("               => the de-risking is JUST LOWER EXPOSURE (no timing skill); replicable by a static e-mix, no MA needed.")
    with open(HERE/"strat_quant_exposure_matched.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\nSaved {HERE/'strat_quant_exposure_matched.json'}")


if __name__=="__main__":
    main()
