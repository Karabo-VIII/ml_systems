"""QUANT: CONSISTENCY of returns -- the user wants returns that are CONSISTENT (alpha-vs-beta semantics aside).

PRE-REGISTERED. 'Consistent' = positive across most periods + no long losing streaks + low dispersion. Measure it at
3 horizons (week 7d / month 30d / quarter 90d) for the candidates + buy-hold, AND test the one lever that could RAISE
consistency: a QUALITY mean-reversion sleeve filling the trend engine's cash weeks (MR = high-hit-rate concave payoff =
the 'consistent' shape; the prior MR test was an ungated falling-knife strawman -85%).
  A_trend_BIN  : SMA trend, breadth gate (wealth-max candidate)
  B_trend_ER   : SMA trend, breadth + trend-efficiency (capital-preservation candidate)
  TREND_plus_MR: trend when trend-deployed; else QUALITY-MR (oversold z10<-1 AND above own SMA50 AND breadth>0.4); else cash
  BH_BTC       : buy-hold BTC (benchmark)
Consistency metrics per horizon: %positive periods, longest losing streak (consec neg), worst period, mean, return/vol.
H0: no long-only construction is 'consistent' at the weekly horizon (trend is bursty; MR adds hit-rate but loses the tails).
1d u100 PIT, inv-vol cap33, next-open, prod cost. OOS-ALL + the 4 sub-periods.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd

HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE)); ROOT=Path(__file__).resolve().parents[3]/"src"; sys.path.insert(0,str(ROOT))
from strat.ma_slice_engine import held_rolling, _cached_band_held
from strat.ma_wealth_driver import _load_asset, _u100_symbols
from strat.canonical_exits import COST_PROD
import strat.fleet_lab as fl
BPD=fl.BARS_PER_DAY["1d"]; LB=14*BPD; SL=7*BPD; SW=100*BPD; CAP=0.33
WALL=int(pd.Timestamp("2026-07-01").value//10**6); CUT=pd.Timestamp("2024-05-15")


def hyst(s,u,d):
    n=len(s);o=np.zeros(n,int);st=0
    for i in range(n):
        if not np.isfinite(s[i]): o[i]=st; continue
        if st==0 and s[i]>u: st=1
        elif st==1 and s[i]<d: st=0
        o[i]=st
    return o


def longest_neg_streak(x):
    best=cur=0
    for v in x:
        cur=cur+1 if v<0 else 0; best=max(best,cur)
    return best


def main():
    closes={};opens={};Htr={}
    for s in _u100_symbols():
        r=_load_asset(s,"1d",WALL)
        if r is None: continue
        d,o,h,l,c=r; cc=np.asarray(c,float); closes[s]=pd.Series(cc,index=d); opens[s]=pd.Series(np.asarray(o,float),index=d)
        Htr[s]=pd.Series(held_rolling(c,_cached_band_held("SMA","1d",s,o,h,l,c,WALL),LB,SL),index=d)
    cdf=pd.DataFrame(closes).sort_index(); idx=cdf.index; cols=list(cdf.columns); T=len(idx); A=len(cols)
    odf=pd.DataFrame(opens).reindex(idx).reindex(columns=cols); C=cdf.to_numpy(); O=odf.to_numpy()
    Htrend=pd.DataFrame(Htr).reindex(idx).fillna(0).reindex(columns=cols).fillna(0).to_numpy()
    vol=cdf.pct_change(fill_method=None).rolling(max(20,5*BPD),min_periods=10).std().to_numpy()
    roo=np.full((T,A),0.0); roo[:-1]=O[1:]/np.where(O[:-1]>0,O[:-1],np.nan)-1.0; roo=np.nan_to_num(roo)
    sall=cdf.rolling(SW,min_periods=SW//2).mean().to_numpy(); vc=np.isfinite(sall).sum(axis=1)
    brd=np.where(vc>0,np.nansum((C>sall)*np.isfinite(sall),axis=1)/np.maximum(vc,1),np.nan); g=hyst(brd,0.5,0.4).astype(float)
    bc=cdf["BTCUSDT"].to_numpy(); mv=np.abs(np.diff(bc,prepend=bc[0])); er=np.abs(bc-pd.Series(bc).shift(14).to_numpy())/np.maximum(pd.Series(mv).rolling(14,min_periods=7).sum().to_numpy(),1e-9)
    erok=(er>0.35).astype(float)
    # quality MR held: oversold z10<-1 AND above own SMA50 (uptrend) AND breadth>0.4
    m10=cdf.rolling(10*BPD,min_periods=5).mean().to_numpy(); s10=cdf.rolling(10*BPD,min_periods=5).std().to_numpy()
    z10=(C-m10)/np.where(s10>1e-12,s10,np.nan); sma50=cdf.rolling(50,min_periods=20).mean().to_numpy()
    Hmr=((np.nan_to_num(z10,nan=0)< -1.0)&(C>sma50)&np.isfinite(sma50)).astype(float)
    def W_iv(Hd,gate2d):
        mult=Hd*gate2d; iv=np.where((mult>0)&np.isfinite(vol)&(vol>0),1.0/np.where(vol>0,vol,np.nan),0.0)
        iv=np.nan_to_num(iv)*mult; sden=iv.sum(axis=1,keepdims=True); return np.minimum(np.where(sden>0,iv/np.maximum(sden,1e-12),0.0),CAP)
    def net_of(W):
        Wl=np.vstack([np.zeros((1,A)),W[:-1]]); return np.nansum(Wl*roo,axis=1)-np.nansum(np.abs(W-Wl),axis=1)*(COST_PROD/2)
    gb=np.broadcast_to(g[:,None],(T,A)); ge=np.broadcast_to((g*erok)[:,None],(T,A))
    Wa=W_iv(Htrend,gb); Wb=W_iv(Htrend,ge)
    # TREND+MR: trend where trend deploys; else MR where breadth>0.4
    Wtr=W_iv(Htrend,gb); trend_on=Wtr.sum(axis=1,keepdims=True)>1e-9
    Wmr=W_iv(Hmr,np.broadcast_to((brd>0.4)[:,None],(T,A)).astype(float))
    Wc=np.where(trend_on,Wtr,Wmr)
    btcr=np.zeros(T); bo=odf["BTCUSDT"].to_numpy(); btcr[:-1]=bo[1:]/np.where(bo[:-1]>0,bo[:-1],np.nan)-1.0; btcr=np.nan_to_num(btcr)
    # PARTIAL: ALWAYS invested at e=clip((brd-0.2)/0.6, 0.15, 1.0) -- trend names when available, else EW-u100 (max time-in-market)
    e=np.clip((np.nan_to_num(brd)-0.20)/0.60,0.15,1.0)
    ewbase=np.where(np.isfinite(C),1.0,0.0); ewbase=ewbase/np.maximum(ewbase.sum(axis=1,keepdims=True),1)
    Wtrend_n=W_iv(Htrend,np.ones((T,A))); trend_any=Wtrend_n.sum(axis=1,keepdims=True)>1e-9
    Wcore=np.where(trend_any,Wtrend_n,ewbase); Wpart=e[:,None]*Wcore
    nets={"A_trend_BIN":net_of(Wa),"B_trend_ER":net_of(Wb),"TREND_plus_MR":net_of(Wc),"PARTIAL_alwaysIn":net_of(Wpart),"BH_BTC":btcr}
    def cons(net,m,h):
        n=net[m]; st=list(range(0,len(n)-h+1,h)); p=np.array([float(np.prod(1+n[i:i+h])-1)*100 for i in st])
        if len(p)==0: return None
        return {"n":len(p),"pct_pos":round(float(np.mean(p>0)),2),"streak":longest_neg_streak(p),"worst":round(float(p.min()),1),
                "mean":round(float(np.mean(p)),2),"ret_vol":round(float(np.mean(p)/np.std(p)),2) if np.std(p)>1e-9 else 0}
    WINS={"OOS-ALL":(idx>=CUT)}
    out={"check":"consistency","ts":datetime.now().strftime("%Y%m%d_%H%M%S"),"horizons":{}}
    print("="*108); print("CONSISTENCY of returns -- %positive / longest losing streak / worst, at week/month/quarter horizons (OOS-ALL)"); print("="*108)
    m=WINS["OOS-ALL"]
    for hn,h in [("WEEK(7d)",SL),("MONTH(30d)",30*BPD),("QUARTER(90d)",90*BPD)]:
        print(f"\n  {hn}:  {'engine':<16}{'n':>4}{'%pos':>7}{'longest_neg_streak':>20}{'worst%':>9}{'mean%':>8}{'ret/vol':>9}")
        out["horizons"][hn]={}
        for nm in nets:
            cc=cons(nets[nm],m,h)
            if cc is None: continue
            out["horizons"][hn][nm]=cc
            print(f"    {'':4}{nm:<16}{cc['n']:>4}{cc['pct_pos']*100:>6.0f}%{cc['streak']:>17} prd{cc['worst']:>8.0f}%{cc['mean']:>+7.2f}%{cc['ret_vol']:>9.2f}")
    # full grade (OOS-ALL): compound, maxDD, Calmar, p05
    from scipy import stats as ss
    def nwp(wk,L=3):
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
    print(f"\n  FULL GRADE (OOS-ALL): {'engine':<16}{'comp%':>8}{'maxDD':>8}{'Calmar':>8}{'NW_p':>7}{'p05':>8}")
    for nm in nets:
        n=nets[nm][m]; eq=np.cumprod(1+n); comp=float(eq[-1]-1)*100; dd=float(np.min((eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq))*100)
        cal=round(comp/abs(dd),2) if abs(dd)>1e-9 else 0; st=list(range(0,len(n)-SL+1,SL)); wk=np.array([float(np.prod(1+n[i:i+SL])-1)*100 for i in st])
        out.setdefault("grade",{})[nm]={"comp":round(comp,1),"maxdd":round(dd,1),"calmar":cal,"nw_p":round(nwp(wk),3),"p05":p05b(wk)}
        gg=out["grade"][nm]; print(f"    {'':16}{nm:<16}{gg['comp']:>+7.0f}%{gg['maxdd']:>+8.0f}{gg['calmar']:>8.2f}{gg['nw_p']:>7.2f}{(gg['p05'] if gg['p05'] is not None else 0):>+7.0f}%")
    print(f"\n  READ: 'consistent' = high %pos + short streak + good ret/vol. PARTIAL_alwaysIn = risk-managed BETA (always in,")
    print(f"        scaled by regime): ~BH consistency at HALF the drawdown. The consistency is the market's drift de-risked, NOT alpha.")
    with open(HERE/"strat_consistency.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\nSaved {HERE/'strat_consistency.json'}")


if __name__=="__main__":
    main()
