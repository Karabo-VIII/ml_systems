"""STRAT 2-7%/wk -- Cycle 9: VERIFY the enriched predictor (AUC 0.667) -- real cross-sectional skill or temporal artifact?

Adversarial verification (RWYB, no false victory):
  1. WITHIN-BAR AUC: per test bar, AUC of P vs label among that bar's eligible names; mean across bars. Isolates
     CROSS-SECTIONAL (which-name) skill, removing the temporal/base-rate confound that inflates pooled AUC.
  2. FEATURE-SHUFFLE NULL: shuffle per-name features within each bar -> within-bar AUC should collapse to ~0.5.
  3. BOOK TRANSLATION: rank eligible by P, top-K inv-vol, breadth-gated, + TP5; deployed-week dist + compound vs
     momentum baseline, cross-window (OOS24 + FWD). AUC means nothing if it doesn't lift the book net of cost.
Train <=2024-05-15, test OOS. If within-bar AUC>0.55 + book beats momentum cross-window -> REAL mover-predictor.
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
ROOT = Path(__file__).resolve().parents[3] / "src"; sys.path.insert(0, str(ROOT))
from strat.ma_wealth_driver import _load_asset, _u100_symbols
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

BPD = M.BPD; K = M.K; SLc = 7*BPD; CUT = pd.Timestamp("2024-05-15"); WALL = int(pd.Timestamp("2026-07-01").value//10**6)


def main():
    f = M.build("FWD")
    idx=f["idx"]; elig=f["elig"]; roo=f["roo"]; conv=f["conv"]; brd=f["brd"]; T=f["T"]; A=f["A"]; cdf=f["cdf"]; vol=f["vol"]
    C=cdf.to_numpy(); ret=cdf.pct_change().fillna(0.0).to_numpy()
    # OHLC for shape feats
    hi={}; lo={}; op={}
    for sym in cdf.columns:
        rec=_load_asset(sym,"1d",WALL)
        if rec is None: continue
        d,o,h,l,c=rec; hi[sym]=pd.Series(np.asarray(h,float),index=d); lo[sym]=pd.Series(np.asarray(l,float),index=d); op[sym]=pd.Series(np.asarray(o,float),index=d)
    H=pd.DataFrame(hi).reindex(idx).reindex(columns=cdf.columns).to_numpy(); L=pd.DataFrame(lo).reindex(idx).reindex(columns=cdf.columns).to_numpy(); Op=pd.DataFrame(op).reindex(idx).reindex(columns=cdf.columns).to_numpy()
    mom14=f["scores"]["mom14"]; mom30=cdf.pct_change(30*BPD).to_numpy(); mom7=cdf.pct_change(7*BPD).to_numpy()
    accel=np.nan_to_num(f["scores"]["accel"],nan=0.0)
    posrange=(C-L)/np.where((H-L)>0,(H-L),np.nan); rng=(H-L)/np.where(C>0,C,np.nan)
    gap=Op/np.where(np.vstack([np.full((1,A),np.nan),C[:-1]])>0,np.vstack([np.full((1,A),np.nan),C[:-1]]),np.nan)-1.0
    jb=list(cdf.columns).index("BTCUSDT") if "BTCUSDT" in cdf.columns else 0
    btc_mom=np.tile(mom14[:,jb:jb+1],(1,A)); breadthb=np.tile(brd[:,None],(1,A)); disp=np.tile(np.nanstd(ret,axis=1,keepdims=True),(1,A))
    idio=mom14-btc_mom
    PER_NAME=["mom7","mom14","mom30","accel","posrange","rng","gap","idio"]   # cross-sectional (vary within bar)
    MARKET=["btc_mom","breadth","disp"]                                        # constant within bar (temporal)
    feats={"mom7":mom7,"mom14":mom14,"mom30":mom30,"accel":accel,"posrange":posrange,"rng":rng,"gap":gap,
           "idio":idio,"btc_mom":btc_mom,"breadth":breadthb,"disp":disp}
    order=list(feats.keys()); Fm=np.stack([np.nan_to_num(feats[k],nan=0.0) for k in order],axis=2)
    fwd=np.full((T,A),np.nan)
    for t in range(T-SLc): fwd[t]=np.prod(1+roo[t:t+SLc],axis=0)-1.0
    lab=np.zeros((T,A),np.int8)
    for t in range(T-SLc):
        e=np.where(elig[t]&np.isfinite(fwd[t]))[0]
        if len(e)==0: continue
        kk=min(K,len(e)); lab[t,e[np.argsort(-fwd[t][e])[:kk]]]=1
    tr=[t for t in range(T-SLc) if idx[t]<CUT]; te=[t for t in range(T) if idx[t]>=CUT and t<T-SLc]
    Xtr=np.vstack([Fm[t][np.where(elig[t])[0]] for t in tr if elig[t].any()]); ytr=np.concatenate([lab[t][np.where(elig[t])[0]] for t in tr if elig[t].any()])
    mdl=GradientBoostingClassifier(n_estimators=150,max_depth=3,learning_rate=0.05,subsample=0.7,random_state=0).fit(Xtr,ytr)
    # P per bar
    P=np.full((T,A),-np.inf)
    for t in range(T):
        e=np.where(elig[t])[0]
        if len(e): P[t][e]=mdl.predict_proba(Fm[t][e])[:,1]
    # 1. within-bar AUC (cross-sectional)
    wb=[]; rng_=np.random.default_rng(0); wb_null=[]
    for t in te:
        e=np.where(elig[t])[0]
        if len(e)<6 or lab[t][e].sum()==0 or lab[t][e].sum()==len(e): continue
        wb.append(roc_auc_score(lab[t][e], P[t][e]))
        # shuffle null: shuffle P within bar
        wb_null.append(roc_auc_score(lab[t][e], rng_.permutation(P[t][e])))
    # also per-name-only model (drop market feats) within-bar AUC
    pn_idx=[order.index(k) for k in PER_NAME]
    Xtr_pn=Xtr[:,pn_idx]; mdl_pn=GradientBoostingClassifier(n_estimators=150,max_depth=3,learning_rate=0.05,subsample=0.7,random_state=0).fit(Xtr_pn,ytr)
    Ppn=np.full((T,A),-np.inf)
    for t in range(T):
        e=np.where(elig[t])[0]
        if len(e): Ppn[t][e]=mdl_pn.predict_proba(Fm[t][:,pn_idx][e])[:,1]
    wb_pn=[]
    for t in te:
        e=np.where(elig[t])[0]
        if len(e)<6 or lab[t][e].sum()==0 or lab[t][e].sum()==len(e): continue
        wb_pn.append(roc_auc_score(lab[t][e], Ppn[t][e]))
    out={"check":"STRAT_2to7_cycle9_verify","within_bar_auc":round(float(np.mean(wb)),4),
         "within_bar_auc_null":round(float(np.mean(wb_null)),4),"within_bar_auc_pernameonly":round(float(np.mean(wb_pn)),4),
         "n_bars":len(wb),"ts":datetime.now().strftime("%Y%m%d_%H%M%S"),"book":{}}
    print("="*86)
    print("VERIFY enriched predictor -- WITHIN-BAR (cross-sectional) AUC + book translation")
    print("="*86)
    print(f"  within-bar AUC (full feats) = {np.mean(wb):.4f}   shuffle-null = {np.mean(wb_null):.4f}   (0.50=no which-name skill)")
    print(f"  within-bar AUC (PER-NAME feats only, drop market) = {np.mean(wb_pn):.4f}")
    # 3. book translation: enriched-P top-K + TP5 vs momentum, cross-window
    print(f"\n  BOOK (top-{K} by P, +TP5, breadth-gated) deployed-week dist vs momentum:")
    print(f"  {'window':<8}{'selector':<14}{'median':>8}{'mean':>7}{'hit>=2':>8}{'in[2,7]':>9}{'comp':>8}")
    for wnm, wm in [("OOS24",(idx>=CUT)&(idx<pd.Timestamp("2025-01-01"))),("FWD",(idx>=pd.Timestamp("2025-04-01"))&(idx<pd.Timestamp("2026-07-01")))]:
        for sn, sc in [("enriched-P",P),("mom14",f["scores"]["mom14"])]:
            net,Wl=E.book_with_exits(f, sc, sl=None, tp=0.05, trail=None); d=E.wdist(net,Wl,wm)
            if d is None: continue
            out["book"][f"{wnm}_{sn}"]=d
            print(f"  {wnm:<8}{sn:<14}{d['median']:>+8.2f}{d['mean']:>+7.2f}{d['hit2']*100:>7.0f}%{d['in27']*100:>8.0f}%{d['comp']:>+7.0f}%")
    with open(HERE/"strat_2to7_verify.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\nSaved {HERE/'strat_2to7_verify.json'}")


if __name__=="__main__":
    main()
