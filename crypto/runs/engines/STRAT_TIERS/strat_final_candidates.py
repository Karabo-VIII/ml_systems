"""QUANT: FINAL candidate slate -- unified grade + full-search DSR + MOM8_ER robustness + live books.

The finalists from the whole arc, graded on ONE bar (OOS 2024-05..2026-05, u100, next-open, prod cost, $10k):
  CONV     : MA cross-type ensemble, breadth-gated, inv-vol cap33, 14d/7d rolling   [de-risked beta, wealth]
  SMA_ER   : SMA + breadth AND trend-efficiency (ER>0.35)                            [capital-preservation]
  PARTIAL  : always 15-100% invested scaled by breadth (trend else EW-mkt)          [consistency]
  MOM8_ER  : momentum(14d) top-8 + breadth AND ER, per-7d rebalance                  [highest return, PROMISING]
BINDING multiple-comparisons: ~40+ configs tried across the session -> DSR deflated at n=40 (honest), not the raw best.
MOM8_ER robustness sweep: ER thr {0.25..0.45} x K {5,8,12} -- is +46% a robust profile or a lucky (thr,K)?
Report: compound/maxDD/Calmar/%wk+/NW p/p05/DSR40 + 4 sub-periods + live book. Tiers: PROVEN de-risked-beta vs PROMISING.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd
from scipy import stats as ss

HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE)); ROOT=Path(__file__).resolve().parents[3]/"src"; sys.path.insert(0,str(ROOT))
from strat.ma_slice_engine import held_rolling, _cached_band_held
from strat.ma_wealth_driver import _load_asset, _u100_symbols
from strat.canonical_exits import COST_PROD
import strat.fleet_lab as fl
BPD=fl.BARS_PER_DAY["1d"]; LB=14*BPD; SL=7*BPD; SW=100*BPD; CAP=0.33; START=10000.0; NSEARCH=40
TYPES=["SMA","EMA","WMA","VIDYA","KAMA"]; WALL=int(pd.Timestamp("2026-07-01").value//10**6)
CUT=pd.Timestamp("2024-05-15"); OOS25=pd.Timestamp("2025-01-01")


def hyst(s,u,d):
    n=len(s);o=np.zeros(n,int);st=0
    for i in range(n):
        if not np.isfinite(s[i]): o[i]=st; continue
        if st==0 and s[i]>u: st=1
        elif st==1 and s[i]<d: st=0
        o[i]=st
    return o


def nw_p(wk,L=3):
    n=len(wk)
    if n<4 or np.std(wk)<1e-12: return 1.0
    mu=np.mean(wk); x=wk-mu; var=np.dot(x,x)/n
    for l in range(1,L+1): var+=2*(1-l/(L+1))*(np.dot(x[l:],x[:-l])/n)
    return float(1-ss.t.cdf(mu/np.sqrt(max(var,1e-12)/n),n-1))


def p05b(wk,nb=1500):
    if len(wk)<8: return None
    rng=np.random.default_rng(0); cs=[]
    for _ in range(nb):
        pk=[]
        while len(pk)<len(wk):
            s0=int(rng.integers(0,len(wk))); pk+=list(range(s0,min(s0+4,len(wk))))
        pk=np.array(pk[:len(wk)]); cs.append(float(np.prod(1+wk[pk]/100)-1)*100)
    return round(float(np.percentile(cs,5)),1)


def dsr(wk,nt):
    if len(wk)<8 or np.std(wk)<1e-9: return None
    sr=np.mean(wk)/np.std(wk); n=len(wk); g3=ss.skew(wk); g4=ss.kurtosis(wk,fisher=False); emc=0.5772; z=ss.norm.ppf
    sr0=np.sqrt(1.0/n)*((1-emc)*z(1-1.0/nt)+emc*z(1-1.0/(nt*np.e))); den=np.sqrt(max(1e-9,1-g3*sr+(g4-1)/4.0*sr**2))
    return round(float(ss.norm.cdf(((sr-sr0)*np.sqrt(n-1))/den)),3)


def main():
    closes={};opens={}; th={t:{} for t in TYPES}
    for s in _u100_symbols():
        r=_load_asset(s,"1d",WALL)
        if r is None: continue
        d,o,h,l,c=r; closes[s]=pd.Series(np.asarray(c,float),index=d); opens[s]=pd.Series(np.asarray(o,float),index=d)
        for t in TYPES: th[t][s]=pd.Series(held_rolling(c,_cached_band_held(t,"1d",s,o,h,l,c,WALL),LB,SL),index=d)
    cdf=pd.DataFrame(closes).sort_index(); idx=cdf.index; cols=list(cdf.columns); T=len(idx); A=len(cols)
    odf=pd.DataFrame(opens).reindex(idx).reindex(columns=cols); C=cdf.to_numpy(); O=odf.to_numpy()
    conv=np.zeros((T,A))
    for t in TYPES: th[t]=pd.DataFrame(th[t]).reindex(idx).fillna(0).reindex(columns=cols).fillna(0).to_numpy(); conv+=th[t]
    Hsma=th["SMA"]
    vol=cdf.pct_change(fill_method=None).rolling(max(20,5*BPD),min_periods=10).std().to_numpy()
    roo=np.full((T,A),0.0); roo[:-1]=O[1:]/np.where(O[:-1]>0,O[:-1],np.nan)-1.0; roo=np.nan_to_num(roo)
    sall=cdf.rolling(SW,min_periods=SW//2).mean().to_numpy(); vc=np.isfinite(sall).sum(axis=1)
    brd=np.where(vc>0,np.nansum((C>sall)*np.isfinite(sall),axis=1)/np.maximum(vc,1),np.nan); brd=np.nan_to_num(brd); gbin=hyst(brd,0.5,0.4)
    bc=cdf["BTCUSDT"].to_numpy(); mv=np.abs(np.diff(bc,prepend=bc[0])); er=np.abs(bc-pd.Series(bc).shift(14).to_numpy())/np.maximum(pd.Series(mv).rolling(14,min_periods=7).sum().to_numpy(),1e-9)
    mom14=cdf.pct_change(14*BPD).to_numpy(); starts=[i for i in range(0,T-SL+1,SL)]
    def b1(x): return np.broadcast_to(x[:,None],(T,A)).astype(float)
    def iv(held,gate2d):
        mult=held*gate2d; w=np.where((mult>0)&np.isfinite(vol)&(vol>0),1.0/np.where(vol>0,vol,np.nan),0.0)
        w=np.nan_to_num(w)*mult; s=w.sum(axis=1,keepdims=True); return np.minimum(np.where(s>0,w/np.maximum(s,1e-12),0.0),CAP)
    def mom_book(K,erthr):
        gate=gbin*(er>erthr).astype(int); W=np.zeros((T,A))
        for i in starts:
            if gate[i]<=0: continue
            s=mom14[i]; e=np.where(np.isfinite(s)&np.isfinite(C[i]))[0]
            if len(e)==0: continue
            k=min(K,len(e)); top=e[np.argsort(-s[e])[:k]]; v=vol[i][top]; w=np.where(np.isfinite(v)&(v>0),1.0/v,0.0)
            w=w/w.sum() if w.sum()>0 else np.ones(k)/k; W[i:i+SL,top]=np.minimum(w,CAP)
        return W
    def net_of(W):
        Wl=np.vstack([np.zeros((1,A)),W[:-1]]); return np.nansum(Wl*roo,axis=1)-np.nansum(np.abs(W-Wl),axis=1)*(COST_PROD/2), W
    e=np.clip((brd-0.20)/0.60,0.15,1.0); ew=np.where(np.isfinite(C),1.0,0.0); ew=ew/np.maximum(ew.sum(axis=1,keepdims=True),1)
    Wtr=iv(Hsma,np.ones((T,A))); trend_any=Wtr.sum(axis=1,keepdims=True)>1e-9
    CAND={"CONV":iv(conv/len(TYPES),b1(gbin)),"SMA_ER":iv(Hsma,b1(gbin*(er>0.35).astype(int))),
          "PARTIAL":e[:,None]*np.where(trend_any,Wtr,ew),"MOM8_ER":mom_book(8,0.35)}
    Wew=ew.copy(); btcr=np.zeros(T); bo=odf["BTCUSDT"].to_numpy(); btcr[:-1]=bo[1:]/np.where(bo[:-1]>0,bo[:-1],np.nan)-1.0
    m=(idx>=CUT)
    def grade(W,is_bench=None):
        net=is_bench if is_bench is not None else net_of(W)[0]; n=net[m]; eq=np.cumprod(1+n)
        dd=float(np.min((eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq))*100) if len(eq)>1 else 0
        st=[i for i in range(0,len(n)-SL+1,SL)]; wk=np.array([float(np.prod(1+n[i:i+SL])-1)*100 for i in st]); comp=float(np.prod(1+n)-1)*100
        subs={}
        for nm2,(a,b) in {"24H2":(CUT,pd.Timestamp("2025-01-01")),"25H1":(pd.Timestamp("2025-01-01"),pd.Timestamp("2025-07-01")),"25H2":(pd.Timestamp("2025-07-01"),pd.Timestamp("2026-01-01")),"26H1":(pd.Timestamp("2026-01-01"),pd.Timestamp("2026-08-01"))}.items():
            mm=(idx>=a)&(idx<b); nn=net[mm]; subs[nm2]=round(float(np.prod(1+nn)-1)*100,1) if len(nn) else None
        return {"final":round(START*(1+comp/100),0),"comp":round(comp,1),"maxdd":round(dd,1),"calmar":round(comp/abs(dd),2) if abs(dd)>1e-9 else 0,
                "pct_pos":round(float(np.mean(wk>0)),2),"nw_p":round(nw_p(wk),3),"p05":p05b(wk),"dsr40":dsr(wk,NSEARCH),"subs":subs,
                "sub_pos":sum(1 for v in subs.values() if v and v>0)}
    out={"check":"final_candidates","n_search_for_dsr":NSEARCH,"ts":datetime.now().strftime("%Y%m%d_%H%M%S"),"grade":{}}
    print("="*118); print(f"FINAL CANDIDATE SLATE -- unified grade, OOS 2024-05..2026-05, DSR deflated for ~{NSEARCH} configs tried"); print("="*118)
    print(f"\n  {'candidate':<10}{'final$':>9}{'comp%':>7}{'maxDD':>7}{'Calmar':>7}{'%wk+':>6}{'NW_p':>6}{'p05':>7}{'DSR40':>7}  sub-periods 24H2/25H1/25H2/26H1 (pos)")
    for nm,W in CAND.items():
        g=grade(W); out["grade"][nm]=g; sv=g["subs"]
        print(f"  {nm:<10}{g['final']:>9.0f}{g['comp']:>+6.0f}%{g['maxdd']:>+7.0f}{g['calmar']:>7.2f}{g['pct_pos']*100:>5.0f}%{g['nw_p']:>6.2f}{(g['p05'] or 0):>+6.0f}%{(g['dsr40'] or 0):>7.2f}  {sv['24H2']:>+5.0f}/{sv['25H1']:>+5.0f}/{sv['25H2']:>+5.0f}/{sv['26H1']:>+5.0f} ({g['sub_pos']}/4)")
    for nm,nb in [("BH_alt",net_of(Wew)[0]),("BH_btc",np.nan_to_num(btcr))]:
        g=grade(None,is_bench=nb); out["grade"][nm]=g
        print(f"  {nm:<10}{g['final']:>9.0f}{g['comp']:>+6.0f}%{g['maxdd']:>+7.0f}{g['calmar']:>7.2f}{g['pct_pos']*100:>5.0f}%{'':>6}{(g['p05'] or 0):>+6.0f}%{'':>7}  {g['subs']['24H2']:>+5.0f}/{g['subs']['25H1']:>+5.0f}/{g['subs']['25H2']:>+5.0f}/{g['subs']['26H1']:>+5.0f} ({g['sub_pos']}/4)")
    # MOM8_ER robustness sweep
    print(f"\n  MOM8_ER ROBUSTNESS (ER thr x K) -- is +46% a robust profile or a lucky config?")
    print(f"    {'':6}"+"".join(f"K={k:<6}" for k in [5,8,12]))
    sweep={}
    for thr in [0.25,0.30,0.35,0.40,0.45]:
        row=f"    thr{thr:<3}"
        for K in [5,8,12]:
            g=grade(mom_book(K,thr)); sweep[f"{thr}_{K}"]={"comp":g["comp"],"maxdd":g["maxdd"],"sub_pos":g["sub_pos"]}
            row+=f"{g['comp']:>+5.0f}%/{g['sub_pos']}"+" "*2
        print(row)
    out["mom8er_sweep"]=sweep
    print(f"    (cell = compound% / sub-periods-positive-of-4)")
    # live books
    last=T-1; out["live"]={"as_of":str(pd.Timestamp(idx[last]).date()),"breadth":round(float(brd[last]),2),"er":round(float(er[last]),2)}
    for nm,W in CAND.items():
        lw=W[last]; live=sorted(((cols[j],round(float(lw[j]),3)) for j in np.where(lw>1e-6)[0]),key=lambda x:-x[1])
        out["live"][nm]={"decision":"DEPLOY" if live else "CASH","n":len(live),"top":live[:6]}
    print(f"\n  LIVE (as of {out['live']['as_of']}, breadth {out['live']['breadth']}, ER {out['live']['er']}):")
    for nm in CAND: print(f"    {nm:<10}{out['live'][nm]['decision']}"+(f" ({out['live'][nm]['n']} names)" if out['live'][nm]['decision']=='DEPLOY' else ""))
    with open(HERE/"strat_final_candidates.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\nSaved {HERE/'strat_final_candidates.json'}")


if __name__=="__main__":
    main()
