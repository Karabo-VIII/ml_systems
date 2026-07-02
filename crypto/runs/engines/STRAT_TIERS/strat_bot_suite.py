"""WORKING-BOT SUITE: consolidate the net-positive edges into deployable candidates + rigor (block-boot p05).

Two edges survived: MOMENTUM top-8 (breadth+ER gated) = the offense (+46% OOS proven); OVERSOLD-BOUNCE = the bear
sleeve (+36pp in bear attribution). Chop = cash (both chop-fills lose). Research says the momentum ranker IS the
right bull allocator. Candidates:
  MOM8_ER      : momentum14 top-8, breadth AND ER>0.35 gate, inv-vol cap33 (baseline offense).
  MOM8+BOUNCE  : MOM8_ER + oversold-bounce sleeve added in BTC-BEAR regime (fill the bear MOM8 sits out).
  SWITCH_RC    : trend@BULL / bounce@BEAR / cash@CHOP (the plain-trend switch).
Grade VAL(untuned) + OOS: compound/maxDD/Calmar/PF/%wk+/block-boot p05/per-regime. Deployable = net-positive both
windows + best risk-adjusted. u100 1d, next-open, prod cost, inv-vol cap33. Honest: survivorship panel, p05 is the bar.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd

HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE)); ROOT=Path(__file__).resolve().parents[3]/"src"; sys.path.insert(0,str(ROOT))
from strat_ti_family_neff import load_panel
from strat_range_bot import range_held, trend_held, er
from strat.canonical_exits import COST_PROD
SL=7; CAP=0.33
WINDOWS={"VAL":("2023-07-01","2024-05-15"),"OOS":("2024-05-15","2026-08-01")}


def hyst(s,u,d):
    n=len(s);o=np.zeros(n,int);st=0
    for i in range(n):
        if not np.isfinite(s[i]): o[i]=st; continue
        if st==0 and s[i]>u: st=1
        elif st==1 and s[i]<d: st=0
        o[i]=st
    return o


def p05b(wk,nb=1500):
    if len(wk)<8: return None
    rng=np.random.default_rng(0); cs=[]
    for _ in range(nb):
        pk=[]
        while len(pk)<len(wk):
            s0=int(rng.integers(0,len(wk))); pk+=list(range(s0,min(s0+4,len(wk))))
        pk=np.array(pk[:len(wk)]); cs.append(float(np.prod(1+wk[pk]/100)-1)*100)
    return round(float(np.percentile(cs,5)),1)


def main():
    data,_=load_panel(); syms=list(data)
    cdf=pd.DataFrame({s:pd.Series(data[s]["c"],index=data[s]["d"]) for s in syms}).sort_index()
    odf=pd.DataFrame({s:pd.Series(data[s]["o"],index=data[s]["d"]) for s in syms}).reindex(cdf.index)
    idx=cdf.index; cols=list(cdf.columns); T=len(idx); A=len(cols); C=cdf.to_numpy(); O=odf.to_numpy()
    roo=np.full((T,A),0.0); roo[:-1]=O[1:]/np.where(O[:-1]>0,O[:-1],np.nan)-1.0; roo=np.nan_to_num(roo)
    vol=cdf.pct_change(fill_method=None).rolling(20,min_periods=10).std().to_numpy(); valid=np.isfinite(C)
    sma100=cdf.rolling(100,min_periods=50).mean().to_numpy(); vc=np.isfinite(sma100).sum(axis=1)
    brd=np.where(vc>0,np.nansum((C>sma100)*np.isfinite(sma100),axis=1)/np.maximum(vc,1),0.0); gbin=hyst(brd,0.5,0.4)
    bc=cdf["BTCUSDT"].to_numpy() if "BTCUSDT" in cols else C[:,0]; ber=er(bc)
    erb=(ber>0.35).astype(int); bsma=pd.Series(bc).rolling(50,min_periods=50).mean().to_numpy()
    reg=np.where(ber<0.35,"CHOP",np.where(bc>bsma,"BULL","BEAR"))
    mom14=cdf.pct_change(14).to_numpy(); starts=[i for i in range(0,T-SL+1,SL)]
    # MOM8_ER book
    def mom8():
        gate=gbin*erb; W=np.zeros((T,A))
        for i in starts:
            if gate[i]<=0: continue
            s=mom14[i]; e=np.where(np.isfinite(s)&np.isfinite(C[i]))[0]
            if len(e)==0: continue
            k=min(8,len(e)); top=e[np.argsort(-s[e])[:k]]; v=vol[i][top]; w=np.where(np.isfinite(v)&(v>0),1.0/v,0.0)
            w=w/w.sum() if w.sum()>0 else np.ones(k)/k; W[i:i+SL,top]=np.minimum(w,CAP)
        return W
    # bounce + trend held matrices
    HR=np.zeros((T,A)); HT=np.zeros((T,A))
    for j,s in enumerate(syms):
        c=data[s]["c"]; HR[:,j]=pd.Series(range_held(c),index=data[s]["d"]).reindex(idx).fillna(0).to_numpy()
        HT[:,j]=pd.Series(trend_held(c),index=data[s]["d"]).reindex(idx).fillna(0).to_numpy()
    def ivbook(H):
        w=np.where((H>0)&np.isfinite(vol)&(vol>0),1.0/np.where(vol>0,vol,np.nan),0.0); w=np.nan_to_num(w)*H
        s=w.sum(axis=1,keepdims=True); return np.minimum(np.where(s>0,w/np.maximum(s,1e-12),0.0),CAP)
    Wmom=mom8(); Wbounce=ivbook(HR); Wtrend=ivbook(HT); isbear=(reg=="BEAR")[:,None]; isbull=(reg=="BULL")[:,None]
    CANDS={"MOM8_ER":Wmom,"MOM8+BOUNCE":np.clip(Wmom+Wbounce*isbear,0,CAP),
           "SWITCH_RC":np.clip(Wtrend*isbull+Wbounce*isbear,0,CAP)}
    def net_of(W):
        Wl=np.vstack([np.zeros((1,A)),W[:-1]]); Wll=np.vstack([np.zeros((1,A)),Wl[:-1]])
        return np.nansum(np.where(valid,Wl*roo,0.0),axis=1)-np.nansum(np.abs(Wl-Wll),axis=1)*(COST_PROD/2)
    def grade(net,W,w0,w1):
        m=(idx>=pd.Timestamp(w0))&(idx<pd.Timestamp(w1)); n=net[m]
        if len(n)<8: return None
        eq=np.cumprod(1+n); comp=float(eq[-1]-1)*100; dd=float(np.min((eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq))*100)
        wk=np.array([float(np.prod(1+n[i:i+SL])-1)*100 for i in range(0,len(n)-SL+1,SL)])
        pf=(-wk[wk>0].sum()/wk[wk<0].sum()) if (wk<0).any() and wk[wk<0].sum()!=0 else 99.9
        return {"comp":round(comp,1),"maxdd":round(dd,1),"calmar":round(comp/abs(dd),2) if abs(dd)>1e-9 else 0,
                "pct_pos":round(float(np.mean(wk>0)),2),"pf":round(float(pf),2),"p05":p05b(wk),
                "expo":round(float(np.mean(W[m].sum(axis=1)))*100,1),"n_wk":len(wk)}
    nets={nm:net_of(W) for nm,W in CANDS.items()}
    out={"check":"bot_suite","ts":datetime.now().strftime("%Y%m%d_%H%M%S"),"windows":{}}
    print("="*112); print("WORKING-BOT SUITE -- deployable candidates, VAL(untuned) + OOS, block-boot p05"); print("="*112)
    for wn,(w0,w1) in WINDOWS.items():
        print(f"\n  {wn} ({w0}..{w1})")
        print(f"    {'bot':<13}{'comp%':>8}{'maxDD':>8}{'Calmar':>8}{'%wk+':>7}{'PF':>7}{'p05':>7}{'expo%':>7}")
        out["windows"][wn]={}
        for nm in CANDS:
            g=grade(nets[nm],CANDS[nm],w0,w1); out["windows"][wn][nm]=g
            if g: print(f"    {nm:<13}{g['comp']:>+7.0f}%{g['maxdd']:>+8.0f}{g['calmar']:>8.2f}{g['pct_pos']*100:>6.0f}%{g['pf']:>7.2f}{(g['p05'] or 0):>+6.0f}%{g['expo']:>6.0f}%")
    print(f"\n  HOLDS BOTH (net-positive VAL & OOS):")
    for nm in CANDS:
        v=out["windows"]["VAL"][nm]; o=out["windows"]["OOS"][nm]
        if v and o: print(f"    {nm:<13} VAL {v['comp']:+.0f}%/p05{(v['p05'] or 0):+.0f} -> OOS {o['comp']:+.0f}%/p05{(o['p05'] or 0):+.0f}  {'HOLDS' if v['comp']>0 and o['comp']>0 else 'no'}")
    with open(HERE/"strat_bot_suite.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\n  Saved {HERE/'strat_bot_suite.json'}")


if __name__=="__main__":
    main()
