"""LOOP verify: the K=3 multi-conditioner gate delta looked robust (drop-top5 +, no-2021 +, bootstrap P=0.04).
Stress it the way the auditor killed the last claim: (1) drop-top-{5,10,15} slices; (2) per-year delta; (3) VACUITY
-- compare TI-eligible+mom-top5 vs UNGATED-mom-top5 (no TI) under the SAME K3 gate (does the TI add?), and vs the
single-breadth-gate (does the MULTI-gate add over breadth-only?); (4) bootstrap for K2 and K3. HMA, 1d, 14d/7d,
TRAIN+VAL only (WALL=VAL_END). Honest: is the improvement real, and is it the GATE or momentum?
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE=Path(__file__).resolve().parent; ROOT=Path(__file__).resolve().parents[3]/"src"; sys.path.insert(0,str(ROOT))
from strat.ma_wealth_driver import _u100_symbols, _norm_sym
from strat.ma_slice_engine import held_rolling, _cached_band_held, VAL_START, VAL_END
from pipeline.chimera_loader import ChimeraLoader
from strat.canonical_exits import COST_PROD
import strat.fleet_lab as fl
BPD=fl.BARS_PER_DAY["1d"]; LB=14*BPD; SL=7*BPD; CAP=0.33
WALL=int(pd.Timestamp(VAL_END).value//10**6); TI="HMA"; V1=pd.Timestamp(VAL_END)
YEARS={"2020":("2020-01-01","2021-01-01"),"2021":("2021-01-01","2022-01-01"),"2022":("2022-01-01","2023-01-01"),
       "2023H1":("2023-01-01",VAL_START),"VAL":(VAL_START,VAL_END)}


def hyst(s,u,d):
    o=np.zeros(len(s),int);st=0
    for i in range(len(s)): st=1 if s[i]>u else (0 if s[i]<d else st); o[i]=st
    return o


def load():
    data={}
    for s in _u100_symbols():
        try: df=ChimeraLoader().load(_norm_sym(s),cadence="1d",features=["open","high","low","close"])
        except Exception: continue
        ts=df["timestamp"].to_numpy(); sel=np.where(ts<WALL)[0]
        if len(sel)<200: continue
        d=pd.to_datetime(ts[sel],unit="ms").floor("D"); keep=~pd.Index(d).duplicated(keep="last"); sel=sel[keep]; d=d[keep]
        data[s]={"d":d,"o":df["open"].to_numpy().astype(float)[sel],"h":df["high"].to_numpy().astype(float)[sel],
                 "l":df["low"].to_numpy().astype(float)[sel],"c":df["close"].to_numpy().astype(float)[sel]}
    return data


def main():
    data=load(); syms=list(data)
    cdf=pd.DataFrame({s:pd.Series(data[s]["c"],index=data[s]["d"]) for s in syms}).sort_index()
    odf=pd.DataFrame({s:pd.Series(data[s]["o"],index=data[s]["d"]) for s in syms}).reindex(cdf.index)
    idx=cdf.index; cols=list(cdf.columns); T=len(idx); A=len(cols); C=cdf.to_numpy(); O=odf.to_numpy()
    R=np.full((T,A),np.nan); R[1:]=C[1:]/np.where(C[:-1]>0,C[:-1],np.nan)-1.0
    roo=np.full((T,A),0.0); roo[:-1]=O[1:]/np.where(O[:-1]>0,O[:-1],np.nan)-1.0; roo=np.nan_to_num(roo)
    vol=cdf.pct_change(fill_method=None).rolling(20,min_periods=10).std().to_numpy(); valid=np.isfinite(C)
    sma100=cdf.rolling(100,min_periods=50).mean().to_numpy(); vc=np.isfinite(sma100).sum(axis=1)
    brd=np.where(vc>0,np.nansum((C>sma100)*np.isfinite(sma100),axis=1)/np.maximum(vc,1),0.0)
    bc=cdf["BTCUSDT"].to_numpy() if "BTCUSDT" in cols else C[:,0]; mv=np.abs(np.diff(bc,prepend=bc[0]))
    ber=np.abs(bc-pd.Series(bc).shift(14).to_numpy())/np.maximum(pd.Series(mv).rolling(14,min_periods=7).sum().to_numpy(),1e-9)
    bsma=pd.Series(bc).rolling(50,min_periods=50).mean().to_numpy(); disp=np.nanstd(R,axis=1); dmed=np.nanmedian(disp[np.isfinite(disp)])
    c1=hyst(brd,0.5,0.4); c2=(brd>pd.Series(brd).shift(7).to_numpy()).astype(int); c3=(ber>0.35).astype(int); c4=(bc>bsma).astype(int); c5=(disp>dmed).astype(int)
    score=c1+c2+c3+c4+c5; gb=c1
    mom14=cdf.pct_change(14).to_numpy(); starts=[i for i in range(0,T-SL+1,SL)]
    ew=np.where(valid,1.0,0.0); ew=ew/np.maximum(ew.sum(axis=1,keepdims=True),1)
    def net_of(W):
        Wl=np.vstack([np.zeros((1,A)),W[:-1]]); Wll=np.vstack([np.zeros((1,A)),Wl[:-1]])
        return np.nansum(np.where(valid,Wl*roo,0.0),axis=1)-np.nansum(np.abs(Wl-Wll),axis=1)*(COST_PROD/2)
    mkt=net_of(ew)
    H=np.zeros((T,A))
    for j,s in enumerate(syms):
        try:
            band=_cached_band_held(TI,"1d",s,data[s]["o"],data[s]["h"],data[s]["l"],data[s]["c"],WALL); hd=held_rolling(data[s]["c"],band,LB,SL)
            H[:,j]=pd.Series(hd,index=data[s]["d"]).reindex(idx).fillna(0).to_numpy()
        except Exception: pass
    def build(gatev,use_ti):
        W=np.zeros((T,A))
        for i in starts:
            if gatev[i]<=0: continue
            base=np.where(np.isfinite(C[i]))[0]
            elig=base[H[i][base]>0] if use_ti else base
            if len(elig)==0: continue
            s=mom14[i][elig]; pick=elig[np.argsort(-np.where(np.isfinite(s),s,-1e9))[:min(5,len(elig))]]
            v=vol[i][pick]; w=np.where(np.isfinite(v)&(v>0),1.0/v,0.0); w=w/w.sum() if w.sum()>0 else np.ones(len(pick))/max(len(pick),1)
            W[i:i+SL,pick]=np.minimum(w,CAP)
        return W
    K3=(score>=3).astype(int); K2=(score>=2).astype(int)
    def dseries(gatev,use_ti):
        net=net_of(build(gatev,use_ti)); rows=[i for i in starts if idx[i]<V1]
        sn=np.array([float(np.prod(1+net[i:i+SL])-1) for i in rows]); mn=np.array([float(np.prod(1+mkt[i:i+SL])-1) for i in rows])
        return np.array(rows),sn-mn
    def stats(rows,d):
        order=np.argsort(-d); rng=np.random.default_rng(0); bs=[float(np.mean(d[rng.integers(0,len(d),len(d))])) for _ in range(5000)]
        yr={}
        for y,(a,b) in YEARS.items():
            m=np.array([pd.Timestamp(a)<=idx[i]<pd.Timestamp(b) for i in rows]); yr[y]=round(float(np.mean(d[m]))*100,2) if m.any() else None
        return {"full":round(float(np.mean(d))*100,2),"dtop5":round(float(np.mean(np.delete(d,order[:5])))*100,2),
                "dtop10":round(float(np.mean(np.delete(d,order[:10])))*100,2),"dtop15":round(float(np.mean(np.delete(d,order[:15])))*100,2),
                "P_le0":round(float(np.mean(np.array(bs)<=0)),3),"per_year":yr,"n":len(d)}
    CONFIGS={"K3_TI(bot)":(K3,True),"K3_noTI(mom only)":(K3,False),"K2_TI":(K2,True),"brd_TI(prior)":(gb,True)}
    out={"check":"smart_gate_verify","TI":TI,"configs":{}}
    print("="*114); print(f"SMART-GATE VERIFY -- is K3 delta robust + is it the GATE or momentum? {TI}, TRAIN+VAL (OOS held out)"); print("="*114)
    print(f"\n  {'config':<18}{'full d%':>8}{'dtop5':>7}{'dtop10':>8}{'dtop15':>8}{'P(d<=0)':>9}{'  per-year (20/21/22/23H1/VAL)'}")
    for nm,(g,ti) in CONFIGS.items():
        rows,d=dseries(g,ti); st=stats(rows,d); out["configs"][nm]=st
        yv=st["per_year"]; ys="  ".join(f"{(yv[y] if yv[y] is not None else 0):+.1f}" for y in YEARS)
        print(f"  {nm:<18}{st['full']:>+7.2f}%{st['dtop5']:>+6.2f}{st['dtop10']:>+7.2f}{st['dtop15']:>+7.2f}{st['P_le0']:>9.2f}   {ys}")
    # is K3-TI's edge over brd-TI real? and does TI add over K3-noTI?
    a=out["configs"]["K3_TI(bot)"]; b=out["configs"]["K3_noTI(mom only)"]; c=out["configs"]["brd_TI(prior)"]
    print(f"\n  GATE contribution (K3_TI vs brd_TI): full {a['full']:+.2f} vs {c['full']:+.2f}; dtop10 {a['dtop10']:+.2f} vs {c['dtop10']:+.2f}; P {a['P_le0']} vs {c['P_le0']}")
    print(f"  TI contribution (K3_TI vs K3_noTI): full {a['full']:+.2f} vs {b['full']:+.2f}; dtop10 {a['dtop10']:+.2f} vs {b['dtop10']:+.2f}")
    robust = a['dtop10']>0 and a['P_le0']<0.10 and sum(1 for y in a['per_year'] if a['per_year'][y] and a['per_year'][y]>0)>=3
    print(f"\n  K3_TI robust (dtop10>0 AND P<0.10 AND >=3/5 years+)? {'YES -- genuinely better gate' if robust else 'NO -- still fragile'}")
    out["k3_ti_robust"]=bool(robust)
    with open(HERE/"strat_smart_gate_verify.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"  Saved {HERE/'strat_smart_gate_verify.json'}")


if __name__=="__main__":
    main()
