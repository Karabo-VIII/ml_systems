"""WORKING-BOT BUILD 4: BREAKOUT (Turtle S1 + Donchian) -- the last untested archetype, run STRAIGHT through the
concentration battery (the lens that killed VIDYA and vindicated MOM8_ER). Decisive: is breakout a 2nd ROBUST edge,
or another concentration artifact? + complementarity vs MOM8_ER (does it fill different weeks?).

Turtle S1 long: enter close > prior 20-bar high; exit close < prior 10-bar low (causal shifts). ATR(20)-vol-target
sizing, cap33. Also breadth-gated variant. u100 1d, next-open, prod cost. Battery: base / cap.05 / noZEC / no-top3 /
drop-top2wk / same-count random-null / weekly-corr vs MOM8_ER.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE)); ROOT=Path(__file__).resolve().parents[3]/"src"; sys.path.insert(0,str(ROOT))
from strat_ti_family_neff import load_panel
import strat_prod_deploy as P
from strat.canonical_exits import COST_PROD
SL=7; CAP=0.33; OOS=pd.Timestamp("2024-05-15"); rng=np.random.default_rng(0)


def turtle_held(h,l,c,en=20,ex=10):
    n=len(c); hh=pd.Series(h).rolling(en,min_periods=en).max().shift(1).to_numpy(); ll=pd.Series(l).rolling(ex,min_periods=ex).min().shift(1).to_numpy()
    held=np.zeros(n,np.int8); pos=0
    for i in range(n):
        if not (np.isfinite(hh[i]) and np.isfinite(ll[i])): held[i]=pos; continue
        if pos==0 and c[i]>hh[i]: pos=1
        elif pos==1 and c[i]<ll[i]: pos=0
        held[i]=pos
    return held


def main():
    data,_=load_panel(); syms=list(data)
    cdf=pd.DataFrame({s:pd.Series(data[s]["c"],index=data[s]["d"]) for s in syms}).sort_index()
    odf=pd.DataFrame({s:pd.Series(data[s]["o"],index=data[s]["d"]) for s in syms}).reindex(cdf.index)
    idx=cdf.index; cols=list(cdf.columns); T=len(idx); A=len(cols); C=cdf.to_numpy(); O=odf.to_numpy()
    roo=np.full((T,A),0.0); roo[:-1]=O[1:]/np.where(O[:-1]>0,O[:-1],np.nan)-1.0; roo=np.nan_to_num(roo)
    vol=cdf.pct_change(fill_method=None).rolling(20,min_periods=10).std().to_numpy()
    sma100=cdf.rolling(100,min_periods=50).mean().to_numpy(); vc=np.isfinite(sma100).sum(axis=1)
    brd=np.where(vc>0,np.nansum((C>sma100)*np.isfinite(sma100),axis=1)/np.maximum(vc,1),0.0)
    def hyst(s,u,d):
        o=np.zeros(len(s),int);st=0
        for i in range(len(s)):
            if s[i]>u: st=1
            elif s[i]<d: st=0
            o[i]=st
        return o
    gbin=hyst(brd,0.5,0.4)
    om=np.asarray(idx>=OOS); zc=cols.index("ZECUSDT") if "ZECUSDT" in cols else -1; starts=[i for i in range(0,T-SL+1,SL)]
    HT=np.zeros((T,A))
    for j,s in enumerate(syms):
        HT[:,j]=pd.Series(turtle_held(data[s]["h"],data[s]["l"],data[s]["c"]),index=data[s]["d"]).reindex(idx).fillna(0).to_numpy()
    def ivbook(H):
        w=np.where((H>0)&np.isfinite(vol)&(vol>0),1.0/np.where(vol>0,vol,np.nan),0.0); w=np.nan_to_num(w)*H
        s=w.sum(axis=1,keepdims=True); return np.minimum(np.where(s>0,w/np.maximum(s,1e-12),0.0),CAP)
    def net_of(W):
        Wl=np.vstack([np.zeros((1,A)),W[:-1]]); Wll=np.vstack([np.zeros((1,A)),Wl[:-1]])
        return np.nansum(Wl*roo,axis=1)-np.nansum(np.abs(Wl-Wll),axis=1)*(COST_PROD/2)
    def comp(net): n=net[om]; return round(float(np.prod(1+n)-1)*100,0)
    def cap_renorm(W,c):
        Wc=np.minimum(W,c); rs=W.sum(axis=1,keepdims=True); rc=Wc.sum(axis=1,keepdims=True)
        return np.where(rc>1e-9,Wc*(rs/np.maximum(rc,1e-9)),Wc)
    def battery(W,label):
        base=comp(net_of(W)); c05=comp(net_of(cap_renorm(W,0.05)))
        Wnz=W.copy();
        if zc>=0: Wnz[:,zc]=0
        nozec=comp(net_of(Wnz))
        contrib=np.nansum(np.where(om[:,None],np.vstack([np.zeros((1,A)),W[:-1]])*roo,0.0),axis=0); top3=np.argsort(-contrib)[:3]
        Wnt=W.copy(); Wnt[:,top3]=0; notop3=comp(net_of(Wnt))
        net=net_of(W); n=net[om]; wk=np.array([float(np.prod(1+n[i:i+SL])-1) for i in range(0,len(n)-SL+1,SL)])
        order=np.argsort(-wk); keep=np.ones(len(wk),bool); keep[order[:2]]=False; ntop2=round(float(np.prod(1+wk[keep])-1)*100,0)
        rowc=(W>0).sum(axis=1); nn=[]
        for _ in range(150):
            Wr=np.zeros((T,A))
            for i in starts:
                k=int(rowc[i]);
                if k<=0: continue
                elig=np.where(np.isfinite(roo[i]))[0]; pick=rng.choice(elig,size=min(k,len(elig)),replace=False); Wr[i:i+SL,pick]=1.0/max(k,1)
            nn.append(comp(net_of(Wr)))
        rp=round(float((np.array(nn)<base).mean())*100,0)
        return {"label":label,"base":base,"cap05":c05,"noZEC":nozec,"no_top3":notop3,"drop_top2wk":ntop2,"randnull_pct":rp},wk
    WT=ivbook(HT); WTg=ivbook(HT*gbin[:,None])
    rT,wkT=battery(WT,"TURTLE"); rTg,_=battery(WTg,"TURTLE_gated")
    # complementarity vs MOM8_ER
    f=P.build_all(); Wm=f["CAND"]["MOM8_ER"]
    # align MOM8 net to same idx (build_all idx may differ) -> recompute MOM8 weekly on ITS own then corr on overlap
    nm=net_of(WT);
    # MOM8 net via its own panel roo
    Wl=np.vstack([np.zeros((1,f["A"])),Wm[:-1]]); Wll=np.vstack([np.zeros((1,f["A"])),Wl[:-1]])
    mnet=np.nansum(Wl*f["roo"],axis=1)-np.nansum(np.abs(Wl-Wll),axis=1)*(COST_PROD/2)
    mom_s=pd.Series(mnet,index=f["idx"]); tur_s=pd.Series(net_of(WT),index=idx)
    j=mom_s.index.intersection(tur_s.index); mo=j>=OOS
    ms=mom_s.reindex(j).to_numpy(); ts=tur_s.reindex(j).to_numpy(); mm=np.asarray(j>=OOS)
    def wkarr(x): return np.array([float(np.prod(1+x[mm][i:i+SL])-1) for i in range(0,int(mm.sum())-SL+1,SL)])
    wm=wkarr(ms); wt=wkarr(ts); corr=float(np.corrcoef(wm,wt)[0,1]) if len(wm)==len(wt) and np.std(wm)>1e-9 and np.std(wt)>1e-9 else None
    out={"check":"breakout_bot","turtle":rT,"turtle_gated":rTg,"weekly_corr_vs_mom8er":round(corr,3) if corr is not None else None,"ts":pd.Timestamp.now().isoformat() if False else "NA"}
    print("="*104); print("BREAKOUT (Turtle) through the CONCENTRATION BATTERY + complementarity vs MOM8_ER"); print("="*104)
    print(f"\n  {'bot':<14}{'base':>7}{'cap.05':>8}{'noZEC':>8}{'no-top3':>9}{'-top2wk':>9}{'randnull%':>10}")
    for r in [rT,rTg]:
        print(f"  {r['label']:<14}{r['base']:>+6.0f}%{r['cap05']:>+7.0f}%{r['noZEC']:>+7.0f}%{r['no_top3']:>+8.0f}%{r['drop_top2wk']:>+8.0f}%{r['randnull_pct']:>9.0f}%")
    print(f"\n  weekly P&L corr(TURTLE, MOM8_ER) OOS = {corr if corr is None else round(corr,2)}  (low = complementary; high = same trend factor)")
    print(f"  ROBUST like MOM8_ER = survives noZEC + no-top3 + drop-top2wk. Else = another concentration-bound trend bot.")
    with open(HERE/"strat_breakout_bot.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\n  Saved {HERE/'strat_breakout_bot.json'}")


if __name__=="__main__":
    main()
