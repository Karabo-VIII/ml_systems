"""PREV-WORK (loop): canonical 7d-slice / 14d-lookback MA bot across TFs -- ADAPTIVE (VIDYA) vs SIMPLE (SMA).

The history (project-ma-per-tf-ironed) says my last-turn 'finer TF = cost death' used a NAIVE fixed-cross config;
the FINE-TF FLIP is that an ADAPTIVE MA (VIDYA) + the 14d/7d slice structure (held_rolling: 14d-confirm entry +
7d time-cap exit) COLLAPSES turnover so net clears cost. Test it on CURRENT full data, reconcile vs V1.1 D81 (which
said fine-TF is redundant/drawdown-insurance-only). Canonical structure held_rolling(c, band, LB=14*BPD, SL=7*BPD).
Cadence-INVARIANT benchmark: fixed-EW fillna(0)=cash (skipna inflates fine TF -- BH must be ~TF-invariant, a check).
Skill = circular-rotation null (exposure+runs matched); cost = gross vs net (prod 0.0028 RT); turnover = RT/asset/yr.
u100, next-open, OOS 2024-05-15->end.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd

HERE=Path(__file__).resolve().parent; ROOT=Path(__file__).resolve().parents[3]/"src"; sys.path.insert(0,str(ROOT))
from strat.ma_wealth_driver import _u100_symbols, _norm_sym
from strat.ma_slice_engine import held_rolling, _cached_band_held
from pipeline.chimera_loader import ChimeraLoader
from strat.canonical_exits import COST_PROD
import strat.fleet_lab as fl
WALL=int(pd.Timestamp("2026-07-01").value//10**6); OOS=pd.Timestamp("2024-05-15")
TFS=["1d","4h","2h","1h","30m"]; TYPES=["VIDYA","SMA"]
FREQ={"30m":"30min","1h":"h","2h":"2h","4h":"4h","1d":"D"}; _2H=2*3600*1000
BPY={"30m":48*365,"1h":24*365,"2h":12*365,"4h":6*365,"1d":365}


def load_tf_ohlc(tf):
    load_cad="1h" if tf=="2h" else tf; data={}
    for s in _u100_symbols():
        try: df=ChimeraLoader().load(_norm_sym(s),cadence=load_cad,features=["open","high","low","close"])
        except Exception: continue
        ts=df["timestamp"].to_numpy(); o=df["open"].to_numpy().astype(float); h=df["high"].to_numpy().astype(float)
        l=df["low"].to_numpy().astype(float); c=df["close"].to_numpy().astype(float)
        if tf=="2h":
            b=(ts//_2H).astype("int64"); g=pd.DataFrame({"o":o,"h":h,"l":l,"c":c,"b":b}).groupby("b",sort=True).agg(
                o=("o","first"),h=("h","max"),l=("l","min"),c=("c","last")); ts=(g.index.to_numpy()*_2H).astype("int64")
            o=g["o"].to_numpy();h=g["h"].to_numpy();l=g["l"].to_numpy();c=g["c"].to_numpy()
        sel=np.where(ts<WALL)[0]
        if len(sel)<600: continue
        di=pd.to_datetime(ts[sel],unit="ms").floor(FREQ[tf]); keep=~pd.Index(di).duplicated(keep="last"); di=di[keep]; sel=sel[keep]
        data[s]={"d":di,"o":o[sel],"h":h[sel],"l":l[sel],"c":c[sel]}
    return data


def main():
    rng=np.random.default_rng(0)
    out={"check":"tf_canonical","cost":COST_PROD,"ts":datetime.now().strftime("%Y%m%d_%H%M%S"),"tfs":{}}
    print("="*118); print("CANONICAL 7d-slice/14d-lookback MA bot by TF -- ADAPTIVE(VIDYA) vs SIMPLE(SMA), cadence-invariant, skill+cost"); print("="*118)
    print(f"\n  {'TF':<4}{'type':<7}{'GROSSc':>8}{'NETc':>8}{'maxDD':>7}{'expo%':>7}{'RT/yr':>7}{'costDrag':>9}{'skill%':>8}{'vsBH':>8}")
    print("  "+"-"*106)
    for tf in TFS:
        data=load_tf_ohlc(tf); syms=list(data)
        if len(syms)<10: print(f"  {tf}: insufficient ({len(syms)})"); continue
        BPD=fl.BARS_PER_DAY[tf]; LB=14*BPD; SL=7*BPD
        grid=pd.DatetimeIndex(sorted(set().union(*[set(data[s]["d"]) for s in syms]))); T=len(grid); A=len(syms)
        # returns (open-to-open) on the grid, fixed-EW cadence-invariant
        O=pd.DataFrame({s:pd.Series(data[s]["o"],index=data[s]["d"]) for s in syms}).reindex(grid).to_numpy()
        roo=np.full((T,A),0.0); roo[:-1]=O[1:]/np.where(O[:-1]>0,O[:-1],np.nan)-1.0; roo=np.nan_to_num(roo)
        valid=np.isfinite(O); om=np.asarray(grid>=OOS); nyr=float(om.sum())/BPY[tf]
        Wbh=np.where(valid,1.0,0.0); Wbh=Wbh/np.maximum(Wbh.sum(axis=1,keepdims=True),1)
        bh=float(np.prod(1+np.nansum(np.vstack([np.zeros((1,A)),Wbh[:-1]])*roo,axis=1)[om])-1)*100
        out["tfs"][tf]={"n_assets":A,"n_years":round(nyr,2),"bh_oos":round(bh,1),"types":{}}
        for ty in TYPES:
            H=np.zeros((T,A))
            for j,s in enumerate(syms):
                try:
                    band=_cached_band_held(ty,tf,s,data[s]["o"],data[s]["h"],data[s]["l"],data[s]["c"],WALL)
                    hd=held_rolling(data[s]["c"],band,LB,SL)
                except Exception: continue
                H[:,j]=pd.Series(hd,index=data[s]["d"]).reindex(grid).fillna(0).to_numpy()
            W=H/np.maximum(H.sum(axis=1,keepdims=True),1e-9)
            Wl=np.vstack([np.zeros((1,A)),W[:-1]]); gross=np.nansum(Wl*roo,axis=1)
            turn=np.nansum(np.abs(W-np.vstack([np.zeros((1,A)),W[:-1]])),axis=1); net=gross-turn*(COST_PROD/2)
            gc=float(np.prod(1+gross[om])-1)*100; nc=float(np.prod(1+net[om])-1)*100
            eq=np.cumprod(1+net[om]); dd=float(np.min((eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq))*100) if om.sum()>1 else 0
            expo=float(np.mean(W[om].sum(axis=1)))*100
            ent=np.sum((H[om][1:]>0)&(H[om][:-1]==0),axis=0); rt_yr=float(np.mean(ent))/max(nyr,1e-9)
            nn=40 if tf in("30m","1h") else 80; null=[]
            for _ in range(nn):
                tau=int(rng.integers(T//10,T-T//10)); Wr=np.roll(W,tau,axis=0); grr=np.nansum(np.vstack([np.zeros((1,A)),Wr[:-1]])*roo,axis=1)
                null.append(float(np.prod(1+grr[om])-1)*100)
            pct=float((np.array(null)<gc).mean())*100
            out["tfs"][tf]["types"][ty]={"gross":round(gc,1),"net":round(nc,1),"maxdd":round(dd,1),"expo":round(expo,1),
                "rt_yr":round(rt_yr,1),"cost_drag":round(gc-nc,1),"skill_pct":round(pct,0),"vs_bh":round(nc-bh,1)}
            print(f"  {tf:<4}{ty:<7}{gc:>+7.0f}%{nc:>+7.0f}%{dd:>+7.0f}{expo:>6.0f}%{rt_yr:>7.1f}{gc-nc:>+8.0f}%{pct:>7.0f}%{nc-bh:>+7.0f}%")
        print(f"  {tf:<4}{'BH':<7}{bh:>+7.0f}%{bh:>+7.0f}%{'':>7}{'100%':>7}{'':>7}{'':>9}{'':>8}{'0':>8}   <- BH must be ~TF-invariant")
        print("  "+"-"*106)
    with open(HERE/"strat_tf_canonical.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    bhs=[out["tfs"][tf]["bh_oos"] for tf in out["tfs"]]
    print(f"\n  BH across TF: {bhs}  (spread {max(bhs)-min(bhs):.0f}pp -- large => coverage/skipna artifact; small => clean)")
    print(f"  Saved {HERE/'strat_tf_canonical.json'}")


if __name__=="__main__":
    main()
