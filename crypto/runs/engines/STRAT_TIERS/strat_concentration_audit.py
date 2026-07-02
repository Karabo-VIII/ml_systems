"""CONCENTRATION + VALID-NULL AUDIT (post-refutation). The expert-auditor REFUTED the VIDYA@1d "skill" claim:
(1) the rotation-null is degenerate on a net-negative window (69% of draws negative -> any drawdown-dodging trend
follower clears it); (2) the whole result is ZEC (77% of gross; drop ZEC -> +53% becomes -51%). RWYB: confirm that
empirically AND apply the SAME lens to MOM8_ER (my claimed working bot) + the other candidates -- is any of them
robust to (a) a per-name weight CAP, (b) dropping ZEC, (c) dropping top weeks, and (d) a VALID timing null
(same-count RANDOM name selection each deploy-week, which controls for exposure+regime, unlike the rotation null)?

A robust bot survives all four. u100 1d, OOS 2024-05-15->end, prod cost. Reuses the exact emitter candidate books.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE)); ROOT=Path(__file__).resolve().parents[3]/"src"; sys.path.insert(0,str(ROOT))
import strat_prod_deploy as P
from strat.ma_slice_engine import held_rolling, _cached_band_held
from strat.ma_type_upgrade import _vidya
from strat.canonical_exits import COST_PROD
import strat.fleet_lab as fl
BPD=fl.BARS_PER_DAY["1d"]; LB=14*BPD; SL=7*BPD; OOS=pd.Timestamp("2024-05-15")
rng=np.random.default_rng(0)


def main():
    f=P.build_all(); CAND=dict(f["CAND"]); idx=f["idx"]; cols=f["cols"]; T=f["T"]; A=f["A"]; roo=f["roo"]
    om=np.asarray(idx>=OOS); zc=cols.index("ZECUSDT") if "ZECUSDT" in cols else -1
    valid=np.isfinite(roo)|(roo!=0)  # bar has data
    def net_of(W):
        Wl=np.vstack([np.zeros((1,A)),W[:-1]]); Wll=np.vstack([np.zeros((1,A)),Wl[:-1]])
        return np.nansum(Wl*roo,axis=1)-np.nansum(np.abs(Wl-Wll),axis=1)*(COST_PROD/2)
    def comp(net): n=net[om]; return round(float(np.prod(1+n)-1)*100,0)
    def cap_renorm(W,c):
        Wc=np.minimum(W,c); rs=W.sum(axis=1,keepdims=True); rc=Wc.sum(axis=1,keepdims=True)
        return np.where(rc>1e-9,Wc*(rs/np.maximum(rc,1e-9)),Wc)  # preserve gross
    # add VIDYA@1d band book (EW) on the same panel via re-load (uses P internals)
    from strat.ma_wealth_driver import _load_asset, _u100_symbols
    WALL=P.WALL; Hv=np.zeros((T,A))
    for j,s in enumerate(cols):
        r=_load_asset(s,"1d",WALL)
        if r is None: continue
        d,o,h,l,c=r
        try:
            band=_cached_band_held("VIDYA","1d",s,o,h,l,c,WALL); hd=held_rolling(c,band,LB,SL)
            Hv[:,j]=pd.Series(hd,index=d).reindex(idx).fillna(0).to_numpy()
        except Exception: pass
    CAND["VIDYA_1d"]=Hv/np.maximum(Hv.sum(axis=1,keepdims=True),1e-9)
    starts=[i for i in range(0,T-SL+1,SL)]
    out={"check":"concentration_audit","oos":str(OOS.date()),"cands":{}}
    print("="*118); print("CONCENTRATION + VALID-NULL AUDIT -- does ANY candidate survive cap / drop-ZEC / drop-weeks / random-null?"); print("="*118)
    print(f"\n  {'cand':<11}{'base':>7}{'cap.33':>8}{'cap.10':>8}{'cap.05':>8}{'noZEC':>8}{'no-top3':>9}{'-top2wk':>9}{'randnull%':>10}")
    print("  "+"-"*112)
    for nm,W in CAND.items():
        base=comp(net_of(W))
        c33=comp(net_of(cap_renorm(W,0.33))); c10=comp(net_of(cap_renorm(W,0.10))); c05=comp(net_of(cap_renorm(W,0.05)))
        Wnz=W.copy();
        if zc>=0: Wnz[:,zc]=0
        nozec=comp(net_of(Wnz))
        contrib=np.nansum(np.where(om[:,None],np.vstack([np.zeros((1,A)),W[:-1]])*roo,0.0),axis=0); top3=np.argsort(-contrib)[:3]
        Wnt=W.copy(); Wnt[:,top3]=0; notop3=comp(net_of(Wnt))
        net=net_of(W); n=net[om]; st=[i for i in range(0,len(n)-SL+1,SL)]; wk=np.array([float(np.prod(1+n[i:i+SL])-1) for i in st])
        order=np.argsort(-wk); keep=np.ones(len(wk),bool); keep[order[:2]]=False; ntop2wk=round(float(np.prod(1+wk[keep])-1)*100,0)
        # VALID null: same-count random name selection each deploy start (preserves exposure timing + count), 200 draws
        nn=[]
        rowcount=(W>0).sum(axis=1)  # names held per bar
        for _ in range(200):
            Wr=np.zeros((T,A))
            for i in starts:
                k=int(rowcount[i])
                if k<=0: continue
                elig=np.where(np.isfinite(roo[i]))[0]
                if len(elig)<k: elig=np.arange(A)
                pick=rng.choice(elig,size=min(k,len(elig)),replace=False); Wr[i:i+SL,pick]=1.0/max(k,1)
            nn.append(comp(net_of(Wr)))
        randpct=round(float((np.array(nn)<base).mean())*100,0)  # bot beats X% of same-count random-selection books
        out["cands"][nm]={"base":base,"cap33":c33,"cap10":c10,"cap05":c05,"noZEC":nozec,"no_top3":notop3,
                          "drop_top2wk":ntop2wk,"random_null_pct":randpct}
        print(f"  {nm:<11}{base:>+6.0f}%{c33:>+7.0f}%{c10:>+7.0f}%{c05:>+7.0f}%{nozec:>+7.0f}%{notop3:>+8.0f}%{ntop2wk:>+8.0f}%{randpct:>9.0f}%")
    print("\n  ROBUST bot = base survives cap.05 + noZEC + drop-top2wk (stays clearly positive) AND beats the RANDOM-SELECTION")
    print("  null (>95% = the specific names/timing matter vs same-count random). Anything that dies on noZEC = a ZEC bet.")
    with open(HERE/"strat_concentration_audit.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\n  Saved {HERE/'strat_concentration_audit.json'}")


if __name__=="__main__":
    main()
