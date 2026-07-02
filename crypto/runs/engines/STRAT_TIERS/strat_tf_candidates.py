"""QUANT: MA candidates across 30m / 1h / 2h / 4h (+1d ref) -- IS FINER TF SKILL, OR JUST COST ON THE SAME BETA?

User intuition: "4h seems like riding the market, no skill." Testable. Pre-registered:
  H0: finer TF adds NO net wealth edge over 4h/1d (cost eats the extra trading) AND no timing skill beyond exposure.
  ALT: some finer TF captures net timing skill surviving prod cost AND beating the same-exposure/same-persistence null.
  SKILL TEST: circular-ROTATE the bot's held vector by random offsets (preserves exposure + run-lengths EXACTLY,
    destroys only alignment to real moves). real gross above the 95th pct of rotations = timing skill; else = pure
    exposure ("riding the market"). Rotation, NOT the broken random-K shuffle (project dead-list).
  COST: report gross vs NET (prod 0.0028 RT) + trades/asset/yr -- finer TF multiplies turnover; this is the crux.
Candidates per TF: SMA(20,50), SMA(50,100), EMA(20,50) bar-count crosses (fixed BAR periods -> finer TF trades faster).
u100, native 30m/1h/4h/1d + synth 2h from 1h, next-open fills, EW-of-held cross-sectional book, OOS 2024-05-15->end.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd

HERE=Path(__file__).resolve().parent; ROOT=Path(__file__).resolve().parents[3]/"src"; sys.path.insert(0,str(ROOT))
from strat.ma_wealth_driver import _u100_symbols, _norm_sym
from pipeline.chimera_loader import ChimeraLoader
from strat.canonical_exits import COST_PROD
WALL=int(pd.Timestamp("2026-07-01").value//10**6); OOS=pd.Timestamp("2024-05-15")
BPY={"30m":48*365,"1h":24*365,"2h":12*365,"4h":6*365,"1d":365}
CANDS=[("SMA",20,50),("SMA",50,100),("EMA",20,50)]
_2H=2*3600*1000


FREQ={"30m":"30min","1h":"h","2h":"2h","4h":"4h","1d":"D"}
def load_tf(tf):
    """u100 open/close matrices at cadence tf (2h synth from 1h). walled < WALL. FLOORED to the tf grid so all
    assets align on common bar boundaries (raw-ms indexing left the matrix sparse -> rolling never filled)."""
    load_cad="1h" if tf=="2h" else tf; opens={}; closes={}
    for s in _u100_symbols():
        try: df=ChimeraLoader().load(_norm_sym(s),cadence=load_cad,features=["open","high","low","close"])
        except Exception: continue
        ts=df["timestamp"].to_numpy(); o=df["open"].to_numpy().astype(float); c=df["close"].to_numpy().astype(float)
        if tf=="2h":
            b=(ts//_2H).astype("int64"); g=pd.DataFrame({"o":o,"c":c,"b":b}).groupby("b",sort=True).agg(o=("o","first"),c=("c","last"))
            ts=(g.index.to_numpy()*_2H).astype("int64"); o=g["o"].to_numpy(); c=g["c"].to_numpy()
        sel=np.where(ts<WALL)[0]
        if len(sel)<500: continue
        di=pd.to_datetime(ts[sel],unit="ms").floor(FREQ[tf]); keep=~pd.Index(di).duplicated(keep="last")
        di=di[keep]; sel=sel[keep]
        opens[s]=pd.Series(o[sel],index=di); closes[s]=pd.Series(c[sel],index=di)
    cdf=pd.DataFrame(closes).sort_index(); odf=pd.DataFrame(opens).reindex(cdf.index)
    return cdf,odf


def ma(a,n,kind):
    if kind=="EMA": return a.ewm(span=n,adjust=False,min_periods=n).mean()
    return a.rolling(n,min_periods=n).mean()


def main():
    out={"check":"tf_candidates","cost_rt":COST_PROD,"oos":str(OOS.date()),"ts":datetime.now().strftime("%Y%m%d_%H%M%S"),"tfs":{}}
    rng=np.random.default_rng(0)
    print("="*120); print(f"MA CANDIDATES BY TIMEFRAME -- skill (rotation null) vs cost, OOS 2024-05-15->end, prod cost {COST_PROD}"); print("="*120)
    print(f"\n  {'TF':<4}{'candidate':<12}{'GROSSc':>8}{'NETc':>8}{'maxDD':>7}{'expo%':>7}{'tr/yr':>7}{'costDrag':>9}{'skill%ile':>10}{'vsBH':>8}")
    print("  "+"-"*112)
    for tf in ["30m","1h","2h","4h","1d"]:
        cdf,odf=load_tf(tf); idx=cdf.index; cols=list(cdf.columns); T=len(idx); A=len(cols)
        if T<500: print(f"  {tf}: insufficient"); continue
        C=cdf.to_numpy(); O=odf.to_numpy()
        roo=np.full((T,A),0.0); roo[:-1]=O[1:]/np.where(O[:-1]>0,O[:-1],np.nan)-1.0; roo=np.nan_to_num(roo)
        valid=np.isfinite(C); om=np.asarray(idx>=OOS); nyr=float(om.sum())/BPY[tf]
        # buy-hold EW (all assets, always) OOS
        Wbh=np.where(valid,1.0,0.0); Wbh=Wbh/np.maximum(Wbh.sum(axis=1,keepdims=True),1)
        bh_net=np.nansum(np.vstack([np.zeros((1,A)),Wbh[:-1]])*roo,axis=1)
        bh_comp=float(np.prod(1+bh_net[om])-1)*100
        out["tfs"][tf]={"n_bars":T,"n_assets":A,"n_years_oos":round(nyr,2),"bh_oos":round(bh_comp,1),"cands":{}}
        for kind,a,b in CANDS:
            fa=ma(cdf,a,kind).to_numpy(); fb=ma(cdf,b,kind).to_numpy()
            held=(np.nan_to_num(fa>fb)&valid).astype(float)
            W=held/np.maximum(held.sum(axis=1,keepdims=True),1e-9)  # EW among held, gross<=1
            Wl=np.vstack([np.zeros((1,A)),W[:-1]])
            gross=np.nansum(Wl*roo,axis=1)
            turn=np.nansum(np.abs(W-np.vstack([np.zeros((1,A)),W[:-1]])),axis=1)
            net=gross-turn*(COST_PROD/2)
            gc=float(np.prod(1+gross[om])-1)*100; nc=float(np.prod(1+net[om])-1)*100
            eq=np.cumprod(1+net[om]); dd=float(np.min((eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq))*100) if om.sum()>1 else 0
            expo=float(np.mean(W[om].sum(axis=1)))*100
            # trades/asset/yr: per-asset entries in held
            hd=held[om]; ent=np.sum((hd[1:]>0)&(hd[:-1]==0),axis=0); tr_yr=float(np.mean(ent))/max(nyr,1e-9)
            # SKILL: circular-rotate W along time (preserve exposure + runs), gross percentile (OOS)
            nn=60 if tf=="30m" else 100; null=[]
            for _ in range(nn):
                tau=int(rng.integers(T//10,T-T//10)); Wr=np.roll(W,tau,axis=0); Wrl=np.vstack([np.zeros((1,A)),Wr[:-1]])
                gr=np.nansum(Wrl*roo,axis=1); null.append(float(np.prod(1+gr[om])-1)*100)
            null=np.array(null); pct=float((null<gc).mean())*100
            out["tfs"][tf]["cands"][f"{kind}{a}/{b}"]={"gross":round(gc,1),"net":round(nc,1),"maxdd":round(dd,1),
                "expo":round(expo,1),"tr_per_yr":round(tr_yr,1),"cost_drag":round(gc-nc,1),"skill_pctile":round(pct,0),"vs_bh":round(nc-bh_comp,1)}
            print(f"  {tf:<4}{kind+str(a)+'/'+str(b):<12}{gc:>+7.0f}%{nc:>+7.0f}%{dd:>+7.0f}{expo:>6.0f}%{tr_yr:>7.1f}{gc-nc:>+8.0f}%{pct:>9.0f}%{nc-bh_comp:>+7.0f}%")
        print(f"  {tf:<4}{'BUY-HOLD':<12}{bh_comp:>+7.0f}%{bh_comp:>+7.0f}%{'':>7}{'100%':>7}{'':>7}{'':>9}{'':>10}{'0':>8}")
        print("  "+"-"*112)
    print(f"\n  READ: skill%ile = where the bot's REAL gross falls in the rotation null (exposure+runs matched). <95 = no timing")
    print(f"        skill (it's exposure/'riding the market'). costDrag = gross-net (prod cost). vsBH = net - buy-hold.")
    with open(HERE/"strat_tf_candidates.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\nSaved {HERE/'strat_tf_candidates.json'}")


if __name__=="__main__":
    main()
