"""STRAT 2-7%/wk -- Cycle 1: which MOVER-SCORE raises the deployed-week hit-rate?

The lever is mover-prediction (which names run THIS 7d). Compare causal price-only scores as the
SELECTION signal on the breadth-gated, band-confirmed eligible set, ranked by deployed-week DISTRIBUTION
(the user's 2-7%/wk = % of deployed weeks landing in [2,7], positive median), cross-window VAL/OOS24/fwd.

scores (per name, causal, close-only): mom14 (baseline) | mom_multi (z(7)+z(14)+z(30)) | breakout
(close/rollmax30) | accel (mom14 - mom14 lag14) | volexp (std14/std60) | composite (z mom14+accel+breakout)
| relrank (xs rank mom14). Top-K inv-vol, breadth-gated, 1d, de-leaked, prod, next-open fills.
METRIC per (score,window): deployed-week (7d) hit-rate %>=2, median, mean, %in[2,7], %>7, %neg, compound.
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
from strat.ma_slice_engine import OOS_START, OOS_2024_END, VAL_START, VAL_END
from strat.canonical_exits import COST_PROD

BPD = fl.BARS_PER_DAY["1d"]; TYPES = ["SMA","EMA","WMA","VIDYA","KAMA"]; LB=14*BPD; SL=7*BPD; SW=100*BPD; K=5
WIN = {"VAL": (int(pd.Timestamp(VAL_END).value//10**6), VAL_START, VAL_END),
       "OOS24": (int(pd.Timestamp("2025-01-01").value//10**6), OOS_START, OOS_2024_END),
       "FWD": (int(pd.Timestamp("2026-07-01").value//10**6), "2025-04-01", "2026-07-01")}


def hyst(sig, up, dn):
    n=len(sig); s=np.zeros(n,int); st=0
    for i in range(n):
        if not np.isfinite(sig[i]): s[i]=st; continue
        if st==0 and sig[i]>up: st=1
        elif st==1 and sig[i]<dn: st=0
        s[i]=st
    return s


def zscore_rows(M):
    mu = np.nanmean(M, axis=1, keepdims=True); sd = np.nanstd(M, axis=1, keepdims=True)
    return np.where(sd>1e-12, (M-mu)/np.where(sd>1e-12,sd,1), 0.0)


def build(window):
    wall, ws, we = WIN[window]
    closes, opens = {}, {}
    for sym in _u100_symbols():
        rec=_load_asset(sym,"1d",wall)
        if rec is None: continue
        d,o,h,l,c=rec; closes[sym]=pd.Series(np.asarray(c,float),index=d); opens[sym]=pd.Series(np.asarray(o,float),index=d)
    cdf=pd.DataFrame(closes).sort_index(); odf=pd.DataFrame(opens).reindex(cdf.index).reindex(columns=cdf.columns)
    idx=cdf.index; T=len(idx); A=len(cdf.columns)
    conv=np.zeros((T,A))
    for ma in TYPES:
        hc={}
        for s in cdf.columns:
            rec=_load_asset(s,"1d",wall)
            if rec is None: continue
            d,o,h,l,c=rec; hc[s]=pd.Series(held_rolling(c,_cached_band_held(ma,"1d",s,o,h,l,c,wall),LB,SL),index=d)
        conv+=pd.DataFrame(hc).reindex(idx).fillna(0).reindex(columns=cdf.columns).fillna(0).to_numpy()
    C=cdf.to_numpy(); O=odf.to_numpy()
    vol=cdf.pct_change().rolling(max(20,bpd5(5)),min_periods=10).std().to_numpy()
    roo=np.full((T,A),0.0); roo[:-1]=O[1:]/np.where(O[:-1]>0,O[:-1],np.nan)-1.0; roo=np.nan_to_num(roo)
    sall=cdf.rolling(SW,min_periods=SW//2).mean().to_numpy(); vc=np.isfinite(sall).sum(axis=1)
    brd=np.where(vc>0, np.nansum((C>sall)*np.isfinite(sall),axis=1)/np.maximum(vc,1), np.nan)
    g=hyst(brd,0.5,0.4)
    elig=(conv>0)&(g[:,None]>0)
    # ---- causal scores ----
    mom7=cdf.pct_change(7*BPD).to_numpy(); mom14=cdf.pct_change(14*BPD).to_numpy(); mom30=cdf.pct_change(30*BPD).to_numpy()
    accel=mom14-np.vstack([np.full((14*BPD,A),np.nan),mom14[:-14*BPD]]) if T>14*BPD else np.full_like(mom14,np.nan)
    rollmax=cdf.rolling(30*BPD,min_periods=10).max().to_numpy(); brk=C/np.where(rollmax>0,rollmax,np.nan)
    rstd_s=cdf.pct_change().rolling(14*BPD,min_periods=7).std().to_numpy(); rstd_l=cdf.pct_change().rolling(60*BPD,min_periods=20).std().to_numpy()
    volexp=rstd_s/np.where(rstd_l>0,rstd_l,np.nan)
    mom_multi=zscore_rows(mom7)+zscore_rows(mom14)+zscore_rows(mom30)
    composite=zscore_rows(mom14)+zscore_rows(np.nan_to_num(accel))+zscore_rows(brk)
    # cross-sectional rank of mom14
    relrank=np.full((T,A),np.nan)
    for t in range(T):
        e=np.where(np.isfinite(mom14[t]))[0]
        if len(e): r=np.argsort(np.argsort(mom14[t][e])); relrank[t][e]=r/max(len(e)-1,1)
    scores={"mom14":mom14,"mom_multi":mom_multi,"breakout":brk,"accel":np.nan_to_num(accel,nan=-1e9),
            "volexp":volexp,"composite":composite,"relrank":relrank}
    mask=(idx>=pd.Timestamp(ws))&(idx<pd.Timestamp(we))
    return {"elig":elig,"vol":vol,"roo":roo,"scores":scores,"mask":mask,"T":T,"A":A,
            "conv":conv,"brd":brd,"idx":idx,"cdf":cdf}


def bpd5(d): return BPD*d


def topk_net(score, f):
    elig,vol,roo,T,A=f["elig"],f["vol"],f["roo"],f["T"],f["A"]
    W=np.zeros((T,A)); s=np.where(elig & np.isfinite(score), score, -np.inf)
    for t in range(T):
        e=np.where(np.isfinite(s[t]))[0]
        if len(e)==0: continue
        kk=min(K,len(e)); top=e[np.argsort(-s[t][e])[:kk]]
        v=vol[t][top]; w=np.where(np.isfinite(v)&(v>0),1.0/v,0.0); W[t,top]=w/w.sum() if w.sum()>0 else 1.0/kk
    Wl=np.vstack([np.zeros((1,A)),W[:-1]]); gr=np.nansum(Wl*roo,axis=1)
    turn=np.nansum(np.abs(W-Wl),axis=1); return gr-turn*(COST_PROD/2), Wl


def weekly_dist(net, Wl, mask):
    nm=net[mask]; gross=Wl.sum(axis=1)[mask]
    wk=[]; i=0
    while i+SL<=len(nm):
        if np.mean(gross[i:i+SL]>1e-9)>0.5:           # deployed week
            wk.append(float(np.prod(1+nm[i:i+SL])-1)*100)
        i+=SL
    if not wk: return None
    wk=np.array(wk); eq=np.cumprod(1+nm); comp=float(eq[-1]-1)*100
    return {"dep_wks":len(wk),"median":round(float(np.median(wk)),2),"mean":round(float(np.mean(wk)),2),
            "hit_ge2":round(float(np.mean(wk>=2)),3),"in_2to7":round(float(np.mean((wk>=2)&(wk<=7))),3),
            "gt7":round(float(np.mean(wk>7)),3),"neg":round(float(np.mean(wk<0)),3),"compound":round(comp,1)}


def main():
    out={"check":"STRAT_2to7_cycle1_scores","K":K,"ts":datetime.now().strftime("%Y%m%d_%H%M%S"),"windows":{}}
    F={w:build(w) for w in ["VAL","OOS24","FWD"]}
    scores=list(F["VAL"]["scores"].keys())
    for w in ["VAL","OOS24","FWD"]:
        f=F[w]; out["windows"][w]={}
        print(f"\n{'='*100}\n{w}  deployed-week distribution by mover-score (K={K}, breadth-gated, 1d, de-leaked, prod)")
        print(f"{'='*100}")
        print(f"  {'score':<11}{'dep_wks':>8}{'median':>8}{'mean':>7}{'hit>=2':>8}{'in[2,7]':>9}{'>7%':>6}{'neg':>6}{'compound':>10}")
        print("  "+"-"*84)
        for sc in scores:
            net,Wl=topk_net(f["scores"][sc],f); d=weekly_dist(net,Wl,f["mask"])
            if d is None: continue
            out["windows"][w][sc]=d
            print(f"  {sc:<11}{d['dep_wks']:>8}{d['median']:>+8.2f}{d['mean']:>+7.2f}{d['hit_ge2']*100:>7.0f}%{d['in_2to7']*100:>8.0f}%{d['gt7']*100:>5.0f}%{d['neg']*100:>5.0f}%{d['compound']:>+9.0f}%")
    with open(HERE/"strat_2to7_movers.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\nSaved {HERE/'strat_2to7_movers.json'}")


if __name__=="__main__":
    main()
