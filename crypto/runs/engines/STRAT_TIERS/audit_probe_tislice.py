"""ADVERSARIAL AUDIT PROBE for the redeveloped TI bot slice-delta claim.
Reuses strat_ti_slice_verify.py's exact load()/net_of()/build() mechanics.
Attacks:
  1. DISTRIBUTED robustness: drop-top-{3,5,10} slices; drop 2021; per-year beat rate.
  2. VACUITY: TI-eligible mom-top5 vs UNGATED mom-top5 (no TI eligibility, same breadth gate).
     If identical/near-identical -> the "TI" adds nothing; it's momentum-in-a-bull.
  3. LEAK: WALL bars-max check; perturbation of OOS-region bars (there are none < WALL, so
     confirm no OOS bars present) + fill-lag sanity (shift correctness).
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
WALL=int(pd.Timestamp(VAL_END).value//10**6)
TIS=["HMA","TEMA","SMA"]
YEARS={"2020":("2020-01-01","2021-01-01"),"2021":("2021-01-01","2022-01-01"),"2022":("2022-01-01","2023-01-01"),
       "2023H1":("2023-01-01",VAL_START),"VAL":(VAL_START,VAL_END)}

def hyst(s,u,d):
    o=np.zeros(len(s),int);st=0
    for i in range(len(s)):
        st=1 if s[i]>u else (0 if s[i]<d else st); o[i]=st
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
                 "l":df["low"].to_numpy().astype(float)[sel],"c":df["close"].to_numpy().astype(float)[sel],
                 "ts_all":ts}  # keep full ts for leak audit
    return data

def main():
    data=load(); syms=list(data)
    # ---- LEAK AUDIT: confirm no bar >= WALL survived into any asset panel ----
    max_ts_used=0; any_oos=0; total_bars=0
    for s in syms:
        # reconstruct used ts by matching d back to ms
        used_ms=(data[s]["d"].view("int64")//10**6)
        total_bars+=len(used_ms)
        if len(used_ms):
            max_ts_used=max(max_ts_used,int(used_ms.max()))
            any_oos+=int((used_ms>=WALL).sum())
    wall_dt=pd.to_datetime(WALL,unit="ms")
    print("="*100)
    print("LEAK AUDIT")
    print(f"  WALL(VAL_END)={VAL_END} = {WALL} ms | max ts used across all assets = {pd.to_datetime(max_ts_used,unit='ms')}")
    print(f"  bars at/after WALL that leaked into panels: {any_oos} (of {total_bars} total)  -> {'CLEAN' if any_oos==0 else 'LEAK!!'}")

    cdf=pd.DataFrame({s:pd.Series(data[s]["c"],index=data[s]["d"]) for s in syms}).sort_index()
    odf=pd.DataFrame({s:pd.Series(data[s]["o"],index=data[s]["d"]) for s in syms}).reindex(cdf.index)
    idx=cdf.index; cols=list(cdf.columns); T=len(idx); A=len(cols); C=cdf.to_numpy(); O=odf.to_numpy()
    roo=np.full((T,A),0.0); roo[:-1]=O[1:]/np.where(O[:-1]>0,O[:-1],np.nan)-1.0; roo=np.nan_to_num(roo)
    vol=cdf.pct_change(fill_method=None).rolling(20,min_periods=10).std().to_numpy(); valid=np.isfinite(C)
    sma100=cdf.rolling(100,min_periods=50).mean().to_numpy(); vc=np.isfinite(sma100).sum(axis=1)
    brd=np.where(vc>0,np.nansum((C>sma100)*np.isfinite(sma100),axis=1)/np.maximum(vc,1),0.0); gbin=hyst(brd,0.5,0.4)
    mom14=cdf.pct_change(14).to_numpy(); starts=[i for i in range(0,T-SL+1,SL)]
    ew=np.where(valid,1.0,0.0); ew=ew/np.maximum(ew.sum(axis=1,keepdims=True),1)
    def net_of(W):
        Wl=np.vstack([np.zeros((1,A)),W[:-1]]); Wll=np.vstack([np.zeros((1,A)),Wl[:-1]])
        return np.nansum(np.where(valid,Wl*roo,0.0),axis=1)-np.nansum(np.abs(Wl-Wll),axis=1)*(COST_PROD/2)
    mkt=net_of(ew)

    # ---- generic builder: allow eligible-set override ----
    def build(H,sel,ungated=False):
        W=np.zeros((T,A))
        for i in starts:
            if gbin[i]<=0: continue
            if ungated:
                elig=np.where(np.isfinite(C[i]))[0]           # ALL valid names (no TI)
            else:
                elig=np.where((H[i]>0)&np.isfinite(C[i]))[0]   # TI-eligible only
            if len(elig)==0: continue
            if sel=="all": pick=elig
            else:
                k=int(sel); s=mom14[i][elig]; pick=elig[np.argsort(-np.where(np.isfinite(s),s,-1e9))[:min(k,len(elig))]]
            v=vol[i][pick]; w=np.where(np.isfinite(v)&(v>0),1.0/v,0.0); w=w/w.sum() if w.sum()>0 else np.ones(len(pick))/max(len(pick),1)
            W[i:i+SL,pick]=np.minimum(w,CAP)
        return W

    def slice_dm(net,w0,w1):
        rows=[i for i in starts if idx[i]>=pd.Timestamp(w0) and idx[i]<pd.Timestamp(w1)]
        sn=np.array([float(np.prod(1+net[i:i+SL])-1) for i in rows]); mn=np.array([float(np.prod(1+mkt[i:i+SL])-1) for i in rows])
        return sn,mn,rows

    def robust(net):
        sn,mn,rows=slice_dm(net,"2020-01-01",VAL_END); d=sn-mn; order=np.argsort(-d)
        full=float(np.mean(d))*100
        jk3=float(np.mean(np.delete(d,order[:3])))*100
        jk5=float(np.mean(np.delete(d,order[:5])))*100
        jk10=float(np.mean(np.delete(d,order[:10])))*100
        # drop 2021
        r21=[i for i in rows if pd.Timestamp("2021-01-01")<=idx[i]<pd.Timestamp("2022-01-01")]
        keep=[k for k,i in enumerate(rows) if i not in set(r21)]
        d_no21=d[keep]; no21=float(np.mean(d_no21))*100
        beat=float(np.mean(sn>mn))
        return dict(full=round(full,2),jk3=round(jk3,2),jk5=round(jk5,2),jk10=round(jk10,2),
                    no2021=round(no21,2),beat=round(beat,2),n=len(d),n21=len(r21))

    print("\n"+"="*100); print("ATTACK 1: DISTRIBUTED robustness (drop-top-K slices; drop 2021)")
    print("="*100)
    out={}
    Hmap={}
    for ti in TIS:
        H=np.zeros((T,A))
        for j,s in enumerate(syms):
            try:
                band=_cached_band_held(ti,"1d",s,data[s]["o"],data[s]["h"],data[s]["l"],data[s]["c"],WALL); hd=held_rolling(data[s]["c"],band,LB,SL)
                H[:,j]=pd.Series(hd,index=data[s]["d"]).reindex(idx).fillna(0).to_numpy()
            except Exception: pass
        Hmap[ti]=H
        net=net_of(build(H,"5"))
        rb=robust(net)
        out.setdefault(ti,{})["robust"]=rb
        print(f"  {ti}: full{rb['full']:+.2f}%  dropTop3{rb['jk3']:+.2f}%  dropTop5{rb['jk5']:+.2f}%  dropTop10{rb['jk10']:+.2f}%  |  NO-2021{rb['no2021']:+.2f}%  (n={rb['n']}, n2021={rb['n21']})  beat={rb['beat']*100:.0f}%")

    print("\n"+"="*100); print("ATTACK 2: VACUITY -- TI-eligible mom-top5 vs UNGATED mom-top5 (same breadth gate, no TI)")
    print("="*100)
    ung=net_of(build(None,"5",ungated=True))
    for ti in TIS:
        net=net_of(build(Hmap[ti],"5"))
        # per-year delta both
        line=[]
        for y,(a,b) in YEARS.items():
            sn,mn,_=slice_dm(net,a,b); su,mu,_=slice_dm(ung,a,b)
            dti=(np.mean(sn)-np.mean(mn))*100 if len(sn) else 0
            dun=(np.mean(su)-np.mean(mu))*100 if len(su) else 0
            line.append(f"{y}:TI{dti:+.2f}/UNG{dun:+.2f}")
        # full
        sn,mn,rows=slice_dm(net,"2020-01-01",VAL_END); su,mu,_=slice_dm(ung,"2020-01-01",VAL_END)
        dti=(np.mean(sn-mn))*100; dun=(np.mean(su-mu))*100
        # correlation of the two slice-return streams + how often picks differ
        corr=float(np.corrcoef(sn,su)[0,1]) if len(sn)>2 else float("nan")
        print(f"  {ti}: FULL TI{dti:+.2f}%  vs UNGATED{dun:+.2f}%  | slice-return corr(TI,UNG)={corr:.3f}")
        print(f"       peryr "+" ".join(line))
    # also VAL only headline
    su,mu,_=slice_dm(ung,VAL_START,VAL_END); print(f"\n  UNGATED mom-top5 VAL delta = {(np.mean(su)-np.mean(mu))*100:+.2f}%  (compare TI HMA VAL +2.28%)")

    print("\n"+"="*100); print("ATTACK 3: fill-lag / mechanics sanity")
    print("="*100)
    # Confirm Wl*roo means weight from slice-start earns NEXT open->open (no same-bar leak).
    # Build a trivial W that is long asset0 only at one start bar, check which roo row is earned.
    i0=starts[10]
    Wtest=np.zeros((T,A)); Wtest[i0,0]=1.0
    Wl=np.vstack([np.zeros((1,A)),Wtest[:-1]])
    earn_rows=np.where(Wl[:,0]>0)[0]
    print(f"  weight set at bar i0={i0} (date {idx[i0].date()}) -> earns roo at bar(s) {earn_rows} (date {idx[earn_rows[0]].date()})")
    print(f"  roo[k]=O[k+1]/O[k]-1 so bar {earn_rows[0]} earns open[{earn_rows[0]+1}]->... : selection at i0 uses close[i0], fill at open[i0+1]. lag OK if earn>i0.")
    print(f"  -> earn bar {earn_rows[0]} {'>' if earn_rows[0]>i0 else '<='} i0 {'(1-bar lag, CLEAN)' if earn_rows[0]==i0+1 else '(CHECK)'}")

    out["ungated_val"]=round(float((np.mean(su)-np.mean(mu))*100),2)
    with open(HERE/"audit_probe_tislice.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\n  Saved {HERE/'audit_probe_tislice.json'}")

if __name__=="__main__":
    main()
