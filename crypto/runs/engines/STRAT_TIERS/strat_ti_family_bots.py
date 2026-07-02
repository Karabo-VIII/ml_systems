"""FAMILIES OF TI BOTS + solve HOW/WHEN/WHERE each should operate (user ask 2026-07-02).

ONE bot per TI type (the denoised type signal = majority vote of that type's config grid; within-type configs are ~1 bet
so the vote IS the type). For each type, SOLVE the operating envelope -- not by imposing one shared gate, but by searching
a small PRE-REGISTERED policy set and picking the best on VAL, reporting OOS (honest: envelope chosen on VAL, tested OOS):
  WHERE (which fired assets to actually hold):  ALL fired   |   only assets in their OWN clean trend (per-asset ER>0.35)
  WHEN  (market timing):                        ALWAYS      |   only when u100 breadth up (>0.5)
  HOW   (sizing):                               inv-vol, cap 33% (fixed; EW reported as sensitivity note)
=> 4 policies/type. The SELF-trend WHERE is the per-asset answer the shared gate misses ("we still have daily movers").
Grade standalone WEALTH (compound, maxDD) on VAL (2023-07..2024-05) + OOS (2024-05..end). Report per family: naked
(ALWAYS/ALL) vs VAL-selected-best, and WHICH envelope won. u100 1d, next-open fills, prod cost 0.0028 RT, $10k, long-only.
STRUCTURE + wealth study; survivorship panel (size off DD, not headline). NO shared apparatus imposed -- solved per family.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd

HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE)); ROOT=Path(__file__).resolve().parents[3]/"src"; sys.path.insert(0,str(ROOT))
from strat_ti_family_neff import load_panel
from strat.deep2020_ti_pipeline import INDICATORS
from strat.canonical_exits import COST_PROD
SL=7; CAP=0.33; START=10000.0
VAL=("2023-07-01","2024-05-15"); OOS=("2024-05-15","2026-08-01")
ERTHR=0.35; BRDTHR=0.5


def sma_held(A_,p):
    c=A_["c"]; a,b=p; sa=pd.Series(c).rolling(a).mean().to_numpy(); sb=pd.Series(c).rolling(b).mean().to_numpy()
    return np.nan_to_num(sa>sb).astype(np.int8)


def main():
    data,nvol=load_panel(); syms=list(data)
    cdf=pd.DataFrame({s:pd.Series(data[s]["c"],index=data[s]["d"]) for s in syms}).sort_index()
    odf=pd.DataFrame({s:pd.Series(data[s]["o"],index=data[s]["d"]) for s in syms}).reindex(cdf.index)
    idx=cdf.index; cols=list(cdf.columns); T=len(idx); A=len(cols); C=cdf.to_numpy(); O=odf.to_numpy()
    aidx={s:j for j,s in enumerate(cols)}
    roo=np.full((T,A),0.0); roo[:-1]=O[1:]/np.where(O[:-1]>0,O[:-1],np.nan)-1.0; roo=np.nan_to_num(roo)
    vol=cdf.pct_change(fill_method=None).rolling(20,min_periods=10).std().to_numpy()
    # ---- conditioners (causal) ----
    # per-asset efficiency ratio (own clean-trend)
    def er_mat(M):
        num=np.abs(M-pd.DataFrame(M).shift(14).to_numpy())
        den=pd.DataFrame(np.abs(np.diff(M,axis=0,prepend=M[:1]))).rolling(14,min_periods=7).sum().to_numpy()
        return np.where(den>1e-9,num/np.maximum(den,1e-9),0.0)
    er_self=er_mat(C)  # T x A
    sma100=cdf.rolling(100,min_periods=50).mean().to_numpy(); vc=np.isfinite(sma100).sum(axis=1)
    brd=np.where(vc>0,np.nansum((C>sma100)*np.isfinite(sma100),axis=1)/np.maximum(vc,1),0.0)  # market breadth
    mkt_on=(brd>BRDTHR).astype(float)  # T
    valid=np.isfinite(C)
    starts=list(range(0,T-SL+1,SL))
    # ---- per-type denoised held (majority vote of the grid) ----
    def type_held(k):
        spec=INDICATORS[k]; grid=spec["grid"]() if callable(spec["grid"]) else spec["grid"]
        acc=np.zeros((T,A)); cnt=0
        for p in grid:
            H=np.zeros((T,A)); ok=False
            for s in syms:
                try: h=spec["base"](data[s],p); ok=True
                except Exception: continue
                H[:,aidx[s]]=pd.Series(h,index=data[s]["d"]).reindex(idx).fillna(0).to_numpy()
            if ok: acc+=H; cnt+=1
        return (acc/max(cnt,1))>=0.5  # majority vote
    TYPES={k:INDICATORS[k]["family"] for k in INDICATORS}
    # SMA anchor
    saH=np.zeros((T,A))
    for cx in [(2,4),(4,8),(8,16),(16,32),(10,20),(20,50)]:
        Hc=np.zeros((T,A))
        for s in syms: Hc[:,aidx[s]]=pd.Series(sma_held(data[s],cx),index=data[s]["d"]).reindex(idx).fillna(0).to_numpy()
        saH+=Hc
    held_by={"SMA":(saH/6.0)>=0.5}; famof={"SMA":"trend(MA)"}
    for k in INDICATORS: held_by[k]=type_held(k); famof[k]=TYPES[k]
    # ---- book + grade ----
    def book(held_bool, where, when):
        H=held_bool.astype(float)
        if where=="self": H=H*(er_self>ERTHR)
        if when=="mkt": H=H*mkt_on[:,None]
        w=np.where((H>0)&np.isfinite(vol)&(vol>0),1.0/np.where(vol>0,vol,np.nan),0.0); w=np.nan_to_num(w)*H
        s=w.sum(axis=1,keepdims=True); W=np.minimum(np.where(s>0,w/np.maximum(s,1e-12),0.0),CAP)
        Wl=np.vstack([np.zeros((1,A)),W[:-1]]); Wll=np.vstack([np.zeros((1,A)),Wl[:-1]])
        net=np.nansum(np.where(valid,Wl*roo,0.0),axis=1)-np.nansum(np.abs(Wl-Wll),axis=1)*(COST_PROD/2)
        expo=np.mean(W.sum(axis=1)>1e-9)
        return net,expo
    def grade(net,win):
        m=(idx>=pd.Timestamp(win[0]))&(idx<pd.Timestamp(win[1])); n=net[m]
        if len(n)<8: return None
        eq=np.cumprod(1+n); comp=float(eq[-1]-1)*100
        dd=float(np.min((eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq))*100)
        st=[i for i in range(0,len(n)-SL+1,SL)]; wk=np.array([float(np.prod(1+n[i:i+SL])-1)*100 for i in st])
        return {"comp":round(comp,1),"maxdd":round(dd,1),"calmar":round(comp/abs(dd),2) if abs(dd)>1e-9 else 0,
                "pct_pos":round(float(np.mean(wk>0)),2),"n_wk":len(wk)}
    POLICIES=[("all","always"),("self","always"),("all","mkt"),("self","mkt")]
    def pname(w,t): return {"all":"ALL","self":"SELF"}[w]+"/"+{"always":"ALWAYS","mkt":"MKT"}[t]
    out={"check":"ti_family_bots","note":"one bot per type; operating envelope solved on VAL, tested OOS","ERTHR":ERTHR,
         "BRDTHR":BRDTHR,"ts":datetime.now().strftime("%Y%m%d_%H%M%S"),"families":{}}
    rows=[]
    for k in held_by:
        pol={}
        for w,t in POLICIES:
            net,expo=book(held_by[k],w,t); gv=grade(net,VAL); go=grade(net,OOS)
            if gv and go: pol[pname(w,t)]={"val":gv,"oos":go,"expo":round(float(expo),2)}
        if not pol: continue
        naked=pol.get("ALL/ALWAYS")
        best=max(pol,key=lambda pn: pol[pn]["val"]["comp"])  # SELECT on VAL only
        out["families"][k]={"family":famof[k],"naked_oos":naked["oos"] if naked else None,
                            "val_best_policy":best,"val_best":pol[best]["val"],"oos_of_val_best":pol[best]["oos"],
                            "all_policies":pol}
        rows.append((k,famof[k],naked,best,pol[best]))
    # ---- print sorted by OOS wealth of the VAL-selected policy ----
    rows.sort(key=lambda r:-(r[4]["oos"]["comp"]))
    print("="*118); print(f"FAMILIES OF TI BOTS -- one bot/type, operating envelope SOLVED on VAL, tested OOS  ({A} assets, {len(rows)} bots)"); print("="*118)
    print(f"\n  {'bot':<10}{'family':<14}| {'NAKED (ALL/ALWAYS)':<22}| {'VAL-picked policy':<16}| {'that policy OOS':<26}")
    print(f"  {'':<10}{'':<14}| {'OOScomp  DD  Cal':<22}| {'':<16}| {'VALcomp -> OOScomp  DD':<26}")
    print("  "+"-"*114)
    for k,fam,naked,best,pb in rows:
        nk=f"{naked['oos']['comp']:>+6.0f}% {naked['oos']['maxdd']:>+4.0f} {naked['oos']['calmar']:>4.1f}" if naked else "   n/a"
        print(f"  {k:<10}{fam:<14}| {nk:<22}| {best:<16}| {pb['val']['comp']:>+6.0f}% -> {pb['oos']['comp']:>+6.0f}% {pb['oos']['maxdd']:>+5.0f} (Cal {pb['oos']['calmar']:.1f})")
    # summary: how often SELF/MKT beat NAKED on OOS
    imp=[(r[4]["oos"]["comp"]-(r[2]["oos"]["comp"] if r[2] else 0)) for r in rows if r[2]]
    beat=sum(1 for r in rows if r[2] and r[4]["oos"]["comp"]>r[2]["oos"]["comp"])
    ddbeat=sum(1 for r in rows if r[2] and r[4]["oos"]["maxdd"]>r[2]["oos"]["maxdd"])
    print(f"\n  Operating envelope (VAL-selected) vs NAKED, on OOS:  beats on wealth {beat}/{len(imp)}  |  reduces DD {ddbeat}/{len(imp)}  |  median wealth delta {np.median(imp):+.0f}pp")
    npos_naked=sum(1 for r in rows if r[2] and r[2]['oos']['comp']>0); npos_solved=sum(1 for r in rows if r[4]['oos']['comp']>0)
    print(f"  Families POSITIVE on OOS:  naked {npos_naked}/{len(rows)}   ->   solved-envelope {npos_solved}/{len(rows)}")
    with open(HERE/"strat_ti_family_bots.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\n  CAVEAT: envelope is chosen on VAL then read on OOS (honest), but 4 policies x {len(rows)} bots = multiple comparisons;")
    print(f"  a per-family OOS win needs the block-boot/DSR pass before it is a real edge. This is the STANDALONE-WEALTH map, not a ship.")
    print(f"\nSaved {HERE/'strat_ti_family_bots.json'}")


if __name__=="__main__":
    main()
