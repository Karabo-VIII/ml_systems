"""QUANT: n_eff of the OTHER TI families -- does the MA collapse (many configs/types = ~1 engine) hold across the
static ensemble bot's full TI register (deep2020_ti_pipeline.INDICATORS, ~30 types x config grids)?

PRE-REGISTERED. H0 (the MA precedent): within-type config grids are redundant (n_eff << N) AND cross-type weekly
gross P&L corr ~1 -> the "30 engines" are ~1-4 effective bets on the same u100-uptrend factor (matching MA: 8 types
n_eff=1.0, 106 configs n_eff=2.6, 443 crosses n_eff=4.0/PC1=45%). ALT: some family genuinely de-correlates
(|corr| < ~0.5 vs the SMA anchor) = a real 2nd engine (MR oscillators buy dips; volatility types gate regimes).
Method: u100 1d full panel (wall 2026-07-01), BASE variant per type (most diversity-favorable; iron adds trend
filters -> would only INCREASE collapse), per-config held -> fixed-EW gross book (UNGATED deliberately: the shared
breadth gate zeroes all books the same weeks and would manufacture correlation; the question is whether the SIGNALS
are distinct), lag-1 causal, 7d-compound weekly streams. Within-type n_eff = N/(1+(N-1)*rbar). Cross-type corr/PCA
on type-ensemble (band-EW) streams + per family bucket. STRUCTURE study, NOT a wealth claim (gross, ungated,
survivorship panel) -- levels are reported only to characterize the streams.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd

HERE=Path(__file__).resolve().parent; ROOT=Path(__file__).resolve().parents[3]/"src"; sys.path.insert(0,str(ROOT))
from strat.deep2020_ti_pipeline import INDICATORS
from strat.ma_wealth_driver import _u100_symbols, _norm_sym
from pipeline.chimera_loader import ChimeraLoader
WALL=int(pd.Timestamp("2026-07-01").value//10**6); SL=7
SMA_CROSSES=[(2,4),(4,8),(8,16),(16,32),(10,20),(20,50)]  # the MA anchor, same harness


def load_panel():
    data={}; nvol=0
    for s in _u100_symbols():
        try:
            df=ChimeraLoader().load(_norm_sym(s),cadence="1d",features=["open","high","low","close","volume"])
            vol=df["volume"].to_numpy().astype(float); nvol+=1
        except Exception:
            try:
                df=ChimeraLoader().load(_norm_sym(s),cadence="1d",features=["open","high","low","close"]); vol=None
            except Exception: continue
        ts=df["timestamp"].to_numpy()
        sel=np.where(ts<WALL)[0]
        if len(sel)<300: continue
        d=pd.to_datetime(ts[sel],unit="ms").floor("D")  # mirrors _load_asset: align all assets on the day grid
        keep=~pd.Index(d).duplicated(keep="last"); sel=sel[keep]; d=d[keep]
        data[s]={"d":d,"o":df["open"].to_numpy().astype(float)[sel],"h":df["high"].to_numpy().astype(float)[sel],
                 "l":df["low"].to_numpy().astype(float)[sel],"c":df["close"].to_numpy().astype(float)[sel],
                 "vol":(vol[sel] if vol is not None else np.full(len(sel),np.nan))}
    return data,nvol


def main():
    data,nvol=load_panel(); syms=list(data)
    cdf=pd.DataFrame({s:pd.Series(data[s]["c"],index=data[s]["d"]) for s in syms}).sort_index()
    idx=cdf.index; T=len(idx); A=len(syms); C=cdf.to_numpy()
    R=np.full((T,A),np.nan); R[1:]=C[1:]/np.where(C[:-1]>0,C[:-1],np.nan)-1.0  # close-to-close (structure study)
    valid=np.isfinite(R); nvalid=np.maximum(valid.sum(axis=1),1)
    print("="*110); print(f"TI-FAMILY n_eff STUDY -- {len(INDICATORS)} types + SMA anchor | u100 1d ({A} assets, {T} bars, vol on {nvol}) | UNGATED GROSS")
    print("="*110)
    aidx={s:j for j,s in enumerate(syms)}
    def book_of(held_by_sym):
        """fixed-EW gross book from per-sym held vectors (lag-1 causal)."""
        H=np.zeros((T,A))
        for s,h in held_by_sym.items():
            hh=pd.Series(h,index=data[s]["d"]).reindex(idx).fillna(0).to_numpy(); H[:,aidx[s]]=hh
        Hl=np.vstack([np.zeros((1,A)),H[:-1]])
        return np.nansum(np.where(valid,Hl*R,0.0),axis=1)/nvalid
    def weekly(x):
        st=range(0,T-SL+1,SL); return np.array([float(np.prod(1+x[i:i+SL])-1) for i in st])
    # ---- build all books ----
    types={}; skipped=[]
    for k,spec in INDICATORS.items():
        if spec.get("loader")=="ohlcv" and nvol<A*0.5: skipped.append(k); continue
        grid=spec["grid"]() if callable(spec["grid"]) else spec["grid"]
        cfgs={}
        for p in grid:
            hb={}
            for s in syms:
                Ad=data[s]
                try: hb[s]=spec["base"](Ad,p)
                except Exception: pass
            if hb: cfgs[str(p)]=book_of(hb)
        if cfgs: types[k]={"family":spec["family"],"cfg_books":cfgs}
    # SMA anchor through the SAME harness
    sma_cfgs={}
    for a,b in SMA_CROSSES:
        hb={}
        for s in syms:
            c=data[s]["c"]; sa=pd.Series(c).rolling(a).mean().to_numpy(); sb=pd.Series(c).rolling(b).mean().to_numpy()
            hb[s]=np.nan_to_num(sa>sb).astype(np.int8)
        sma_cfgs[f"({a},{b})"]=book_of(hb)
    types["SMA_X"]={"family":"trend(MA)","cfg_books":sma_cfgs}
    # ---- within-type n_eff ----
    out={"check":"ti_family_neff","note":"ungated gross, base variant, fixed-EW u100 1d","skipped_no_vol":skipped,
         "ts":datetime.now().strftime("%Y%m%d_%H%M%S"),"types":{},"cross":{}}
    print(f"\n  WITHIN-TYPE (config-grid redundancy)  [MA precedent: WMA 106 cfgs -> n_eff 2.6]")
    print(f"  {'type':<10}{'family':<16}{'Ncfg':>5}{'expo%':>7}{'rbar':>7}{'n_eff':>7}   verdict")
    ens={}
    for k in sorted(types,key=lambda x:types[x]["family"]):
        cb=types[k]["cfg_books"]; W=np.array([weekly(v) for v in cb.values()])
        sd=W.std(axis=1); keep=sd>1e-10; W=W[keep]; n=len(W)
        expo=float(np.mean([np.mean(np.abs(v)>1e-12) for v in cb.values()]))*100
        if n<2:
            rbar,neff=None,(1.0 if n==1 else 0.0)
        else:
            cm=np.corrcoef(W); rbar=float((cm.sum()-n)/(n*(n-1))); neff=n/(1+(n-1)*max(rbar,0))
        e=np.mean(np.array([v for v in cb.values()]),axis=0); ens[k]=e
        out["types"][k]={"family":types[k]["family"],"n_cfg":len(cb),"n_used":n,"expo_pct":round(expo,1),
                         "rbar":None if rbar is None else round(rbar,3),"n_eff":round(float(neff),2)}
        v=("COLLAPSED" if rbar is not None and rbar>0.7 else "partial" if rbar is not None and rbar>0.4 else "diverse") if rbar is not None else "n/a"
        print(f"  {k:<10}{types[k]['family']:<16}{len(cb):>5}{expo:>6.0f}%{(rbar if rbar is not None else float('nan')):>7.2f}{neff:>7.1f}   {v}")
    # ---- cross-type ----
    ks=list(ens); EW=np.array([weekly(ens[k]) for k in ks]); sd=EW.std(axis=1)
    ks=[k for k,s_ in zip(ks,sd) if s_>1e-10]; EW=EW[sd>1e-10]; n=len(ks)
    cm=np.corrcoef(EW); rbar=float((cm.sum()-n)/(n*(n-1))); neff=n/(1+(n-1)*max(rbar,0))
    z=(EW-EW.mean(axis=1,keepdims=True))/EW.std(axis=1,keepdims=True)
    ev=np.linalg.eigvalsh(np.corrcoef(z)); ev=ev[::-1]; pc1=float(ev[0]/n); pc2=float(ev[1]/n)
    ia=ks.index("SMA_X"); corr_ma={k:round(float(cm[ks.index(k),ia]),2) for k in ks}
    out["cross"]={"n_types":n,"rbar":round(rbar,3),"n_eff":round(float(neff),2),"pc1":round(pc1,3),"pc2":round(pc2,3),
                  "corr_vs_SMA":corr_ma}
    print(f"\n  CROSS-TYPE (type-ensemble streams, {n} types)  [MA precedent: 8 MA types -> n_eff 1.0]")
    print(f"    rbar {rbar:+.2f}  ->  n_eff = {neff:.1f} of {n}   PC1 {pc1*100:.0f}%  PC2 {pc2*100:.0f}%")
    print(f"\n  corr(type, SMA anchor) by family bucket:")
    fams={}
    for k in ks:
        if k=="SMA_X": continue
        fams.setdefault(types[k]["family"],[]).append((k,corr_ma[k]))
    for fam in sorted(fams):
        row=sorted(fams[fam],key=lambda x:-abs(x[1]))
        print(f"    {fam:<16} "+"  ".join(f"{k}{c:+.2f}" for k,c in row))
        # bucket-internal rbar
        kk=[k for k,_ in fams[fam]]
        if len(kk)>=2:
            sub=np.array([EW[ks.index(k)] for k in kk]); scm=np.corrcoef(sub); m=len(kk)
            srb=float((scm.sum()-m)/(m*(m-1)))
            out["cross"][f"bucket_{fam}"]={"n":m,"rbar":round(srb,3),"n_eff":round(m/(1+(m-1)*max(srb,0)),2)}
            print(f"    {'':<16}bucket rbar {srb:+.2f} -> n_eff {m/(1+(m-1)*max(srb,0)):.1f} of {m}")
    if skipped: print(f"\n  NOTE: skipped (no volume data): {skipped}")
    with open(HERE/"strat_ti_family_neff.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\nSaved {HERE/'strat_ti_family_neff.json'}")


if __name__=="__main__":
    main()
