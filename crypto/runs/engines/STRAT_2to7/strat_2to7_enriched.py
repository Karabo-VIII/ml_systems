"""STRAT 2-7%/wk -- Cycle 8: ENRICHED meta-labeler -- is there ANY extractable mover-signal? (definitive ceiling test)

Cycle 5 (5 price factors) was ~random. Before declaring the prediction ceiling, try ORTHOGONAL feature classes:
  price: mom7/14/30, accel
  OHLC-shape: position-in-range (c-l)/(h-l), range (h-l)/c, gap (o/prevc-1)
  market-state (broadcast): BTC mom14, breadth, xs-dispersion
  lead-lag: idiosyncratic momentum = mom14 - beta*BTC_mom14 (residual after market)
Label: top-K forward-7d mover among eligible. Train <=2024-05-15, test OOS. Measure OOS AUC + lift over base-rate
(the DIRECT test of predictable signal) + the deployed-week book. If AUC ~= 0.5 -> ceiling = market, definitively.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(ROOT))
import strat.fleet_lab as fl
from strat.ma_wealth_driver import _load_asset, _u100_symbols
from strat.ma_slice_engine import held_rolling, _cached_band_held
from strat.canonical_exits import COST_PROD
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

BPD = fl.BARS_PER_DAY["1d"]; TYPES=["SMA","EMA","WMA","VIDYA","KAMA"]; LB=14*BPD; SLc=7*BPD; SW=100*BPD; K=5
WALL=int(pd.Timestamp("2026-07-01").value//10**6); CUT=pd.Timestamp("2024-05-15")


def hyst(sig,up,dn):
    n=len(sig); s=np.zeros(n,int); st=0
    for i in range(n):
        if not np.isfinite(sig[i]): s[i]=st; continue
        if st==0 and sig[i]>up: st=1
        elif st==1 and sig[i]<dn: st=0
        s[i]=st
    return s


def main():
    O={},; cl={}; hi={}; lo={}; op={}
    cl={}
    for sym in _u100_symbols():
        rec=_load_asset(sym,"1d",WALL)
        if rec is None: continue
        d,o,h,l,c=rec
        cl[sym]=pd.Series(np.asarray(c,float),index=d); hi[sym]=pd.Series(np.asarray(h,float),index=d)
        lo[sym]=pd.Series(np.asarray(l,float),index=d); op[sym]=pd.Series(np.asarray(o,float),index=d)
    cdf=pd.DataFrame(cl).sort_index(); idx=cdf.index; T=len(idx); A=len(cdf.columns)
    hdf=pd.DataFrame(hi).reindex(idx).reindex(columns=cdf.columns); ldf=pd.DataFrame(lo).reindex(idx).reindex(columns=cdf.columns); odf=pd.DataFrame(op).reindex(idx).reindex(columns=cdf.columns)
    C=cdf.to_numpy(); H=hdf.to_numpy(); L=ldf.to_numpy(); Op=odf.to_numpy()
    conv=np.zeros((T,A))
    for ma in TYPES:
        hc={}
        for s in cdf.columns:
            rec=_load_asset(s,"1d",WALL)
            if rec is None: continue
            d,o,h,l,c=rec; hc[s]=pd.Series(held_rolling(c,_cached_band_held(ma,"1d",s,o,h,l,c,WALL),LB,SLc),index=d)
        conv+=pd.DataFrame(hc).reindex(idx).fillna(0).reindex(columns=cdf.columns).fillna(0).to_numpy()
    sall=cdf.rolling(SW,min_periods=SW//2).mean().to_numpy(); vc=np.isfinite(sall).sum(axis=1)
    brd=np.where(vc>0,np.nansum((C>sall)*np.isfinite(sall),axis=1)/np.maximum(vc,1),np.nan)
    g=hyst(brd,0.5,0.4); elig=(conv>0)&(g[:,None]>0)
    roo=np.full((T,A),0.0); roo[:-1]=Op[1:]/np.where(Op[:-1]>0,Op[:-1],np.nan)-1.0; roo=np.nan_to_num(roo)
    ret=cdf.pct_change().fillna(0.0).to_numpy()
    # features
    mom7=cdf.pct_change(7*BPD).to_numpy(); mom14=cdf.pct_change(14*BPD).to_numpy(); mom30=cdf.pct_change(30*BPD).to_numpy()
    accel=mom14-np.vstack([np.full((14*BPD,A),np.nan),mom14[:-14*BPD]])
    posrange=(C-L)/np.where((H-L)>0,(H-L),np.nan); rng=(H-L)/np.where(C>0,C,np.nan)
    gap=Op/np.where(np.vstack([np.full((1,A),np.nan),C[:-1]])>0,np.vstack([np.full((1,A),np.nan),C[:-1]]),np.nan)-1.0
    btc="BTCUSDT" if "BTCUSDT" in cdf.columns else cdf.columns[0]; jb=list(cdf.columns).index(btc)
    btc_mom=np.tile(mom14[:,jb:jb+1],(1,A))
    breadthb=np.tile(brd[:,None],(1,A)); disp=np.tile(np.nanstd(ret,axis=1,keepdims=True),(1,A))
    idio=mom14-btc_mom   # residual momentum vs BTC (lead-lag)
    feats={"mom7":mom7,"mom14":mom14,"mom30":mom30,"accel":accel,"posrange":posrange,"rng":rng,"gap":gap,
           "btc_mom":btc_mom,"breadth":breadthb,"disp":disp,"idio":idio}
    Fm=np.stack([np.nan_to_num(feats[k],nan=0.0) for k in feats],axis=2)
    fwd=np.full((T,A),np.nan)
    for t in range(T-SLc): fwd[t]=np.prod(1+roo[t:t+SLc],axis=0)-1.0
    lab=np.zeros((T,A),np.int8)
    for t in range(T-SLc):
        e=np.where(elig[t]&np.isfinite(fwd[t]))[0]
        if len(e)==0: continue
        kk=min(K,len(e)); lab[t,e[np.argsort(-fwd[t][e])[:kk]]]=1
    tr=[t for t in range(T-SLc) if idx[t]<CUT]; te=[t for t in range(T-SLc) if idx[t]>=CUT]
    Xtr=np.vstack([Fm[t][np.where(elig[t])[0]] for t in tr if elig[t].any()])
    ytr=np.concatenate([lab[t][np.where(elig[t])[0]] for t in tr if elig[t].any()])
    Xte=np.vstack([Fm[t][np.where(elig[t])[0]] for t in te if elig[t].any()])
    yte=np.concatenate([lab[t][np.where(elig[t])[0]] for t in te if elig[t].any()])
    mdl=GradientBoostingClassifier(n_estimators=150,max_depth=3,learning_rate=0.05,subsample=0.7).fit(Xtr,ytr)
    pte=mdl.predict_proba(Xte)[:,1]
    auc=roc_auc_score(yte,pte); base=float(yte.mean())
    # lift: among top-decile predicted, what's the actual top-mover rate vs base
    thr=np.percentile(pte,90); topmask=pte>=thr; lift=float(yte[topmask].mean())/base if base>0 else 0
    fi=dict(zip(list(feats.keys()),[round(float(x),3) for x in mdl.feature_importances_]))
    out={"check":"STRAT_2to7_cycle8_enriched","oos_auc":round(float(auc),4),"base_rate":round(base,3),
         "top_decile_lift":round(lift,2),"feat_importance":fi,"ts":datetime.now().strftime("%Y%m%d_%H%M%S")}
    print("="*84)
    print("ENRICHED meta-labeler -- DIRECT prediction-skill test (11 features incl OHLC-shape, market, lead-lag)")
    print("="*84)
    print(f"  OOS AUC = {auc:.4f}   (0.50 = no skill)   base-rate(top-mover) = {base:.3f}")
    print(f"  top-decile-predicted actual-mover rate = {yte[topmask].mean():.3f}  -> LIFT {lift:.2f}x over base")
    print(f"  feature importance: "+", ".join(f"{k}:{v:.2f}" for k,v in sorted(fi.items(),key=lambda x:-x[1])[:6]))
    verdict = "SIGNAL (AUC>0.55)" if auc>0.55 else ("WEAK (0.52-0.55)" if auc>0.52 else "NO SKILL (~random) -> ceiling=market")
    print(f"  VERDICT: {verdict}")
    out["verdict"]=verdict
    with open(HERE/"strat_2to7_enriched.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\nSaved {HERE/'strat_2to7_enriched.json'}")


if __name__=="__main__":
    main()
