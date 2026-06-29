"""STRAT TIERS -- the always-scanning RANKER: run every engine weekly, rank by recent edge, pick the active one.

User: "Even if we sit in cash, I want us in the market every week looking; rank which strategy is active." Build a
weekly ranker over the CONFIRMED engines (momentum top-8 = hyper-selective; Bollinger; MACD; MA cross-type conv), all
breadth-gated, u100, 1d, open-to-open, PROD cost. Each week: compute ALL engines (always scanning), the ACTIVE engine =
the one with the best trailing-4-week standalone net (recent-edge ranker); the book = the active engine's book that week.
Compare the ranker to each standalone engine + an equal-blend, full OOS. Emit the live active engine + its book.
HONEST: per the confirmation, only momentum (strong) + Bollinger (moderate) are solid; the ranker tests whether
recent-edge rotation across engines beats the best single engine (a real test, may be null).
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd

HERE=Path(__file__).resolve().parent; ROOT=Path(__file__).resolve().parents[3]/"src"; sys.path.insert(0,str(ROOT))
sys.path.insert(0, str(HERE.parent/"STRAT_2to7"))
import strat_2to7_movers as M
import strat.ti_engine_sota as T
from strat.ma_wealth_driver import _load_asset, _u100_symbols
from strat.canonical_exits import COST_PROD
BPD=M.BPD; SB=7*BPD; CUT=pd.Timestamp("2024-05-15"); CAP=0.33


def ohlc():
    H={};L={}
    for s in _u100_symbols():
        r=_load_asset(s,"1d",int(pd.Timestamp("2026-07-01").value//10**6))
        if r is None: continue
        d,o,h,l,c=r; H[s]=pd.Series(np.asarray(h,float),index=d); L[s]=pd.Series(np.asarray(l,float),index=d)
    return H,L


def net_from_W(W,roo):
    Wl=np.vstack([np.zeros((1,W.shape[1])),W[:-1]]); gr=np.nansum(Wl*roo,axis=1)
    turn=np.nansum(np.abs(W-np.vstack([np.zeros((1,W.shape[1])),W[:-1]])),axis=1); return gr-turn*(COST_PROD/2), Wl


def main():
    f=M.build("FWD"); idx=f["idx"]; conv=f["conv"]; vol=f["vol"]; roo=f["roo"]; brd=f["brd"]; cdf=f["cdf"]; cols=list(cdf.columns); T_=f["T"]; A=f["A"]; mom=f["scores"]["mom14"]
    g=M.hyst(brd,0.5,0.4); elig=(g[:,None]>0)
    Hh,Ll=ohlc(); Hn=pd.DataFrame(Hh).reindex(idx).reindex(columns=cols).to_numpy(); Ln=pd.DataFrame(Ll).reindex(idx).reindex(columns=cols).to_numpy(); Cn=cdf.to_numpy()
    # ENGINE weight matrices (breadth-gated)
    def W_momentum():
        W=np.zeros((T_,A)); s=np.where((conv>0)&elig&np.isfinite(mom),mom,-np.inf)
        for t in range(T_):
            e=np.where(np.isfinite(s[t]))[0]
            if len(e)==0: continue
            k=min(8,len(e)); top=e[np.argsort(-s[t][e])[:k]]; v=vol[t][top]; w=np.where(np.isfinite(v)&(v>0),1.0/v,0.0); W[t,top]=w/w.sum() if w.sum()>0 else 1.0/k
        return W
    def W_ti(fn,needs_hl=False):
        Hd=np.zeros((T_,A))
        for j in range(A):
            Hd[:,j]=fn(Cn[:,j],Hn[:,j],Ln[:,j]) if needs_hl else fn(Cn[:,j])
        Hd=Hd*(g[:,None]>0); s=Hd.sum(axis=1,keepdims=True); return np.where(s>0,Hd/np.maximum(s,1),0.0)
    def W_conv():
        mult=(conv/5.0)*(g[:,None]>0); iv=np.where((mult>0)&np.isfinite(vol)&(vol>0),1.0/np.where(vol>0,vol,np.nan),0.0); iv=np.nan_to_num(iv)*mult; s=iv.sum(axis=1,keepdims=True)
        return np.minimum(np.where(s>0,iv/np.maximum(s,1e-12),0.0),CAP)
    engines={"momentum8":W_momentum(),"bollinger":W_ti(lambda c:T.bollinger_single_held(c,15,1.5,"breakout")),
             "macd":W_ti(lambda c:T.macd_single_held(c,12,26,9)),"ma_conv":W_conv()}
    Wmats=engines; nets={k:net_from_W(W,roo)[0] for k,W in Wmats.items()}
    # weekly (7d non-overlapping) net per engine
    wk_idx=list(range(0,T_-SB+1,SB)); wk_dates=[idx[i] for i in wk_idx]
    wknet={k:np.array([float(np.prod(1+nets[k][i:i+SB])-1) for i in wk_idx]) for k in nets}
    names=list(engines.keys()); NW=len(wk_idx)
    # RANKER: each week, active = best trailing-4wk standalone net; deploy that engine's book this week
    LOOK=4; rank_net=np.zeros(NW); active=[None]*NW
    for w in range(NW):
        if w<LOOK: active[w]="momentum8"      # warmup default = the strongest confirmed engine
        else:
            tr={k:float(np.mean(wknet[k][w-LOOK:w])) for k in names}
            active[w]=max(tr,key=tr.get)
        rank_net[w]=wknet[active[w]][w]
    # OOS mask on weeks
    oosw=np.array([d>=CUT for d in wk_dates])
    def stats(wn,mask):
        x=wn[mask]; eq=np.cumprod(1+x)
        return {"comp":round(float(eq[-1]-1)*100,1),"wk_mean":round(float(np.mean(x))*100,2),"wk_median":round(float(np.median(x))*100,2),
                "in27":round(float(np.mean((x*100>=2)&(x*100<=7))),3),"pos":round(float(np.mean(x>0)),3),
                "maxdd":round(float(np.min((eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq))*100),1)}
    out={"check":"STRAT_TIERS_ranker","ts":datetime.now().strftime("%Y%m%d_%H%M%S"),"oos":{}}
    print("="*92); print("TIERED RANKER (always-scan, recent-edge active-engine pick) -- OOS 2024-05..2026-05 weekly"); print("="*92)
    print(f"  {'strategy':<16}{'comp%':>8}{'wk_mean':>9}{'wk_med':>8}{'in[2,7]':>9}{'wk>0':>7}{'maxDD':>7}")
    for k in names:
        s=stats(wknet[k],oosw); out["oos"][k]=s
        print(f"  {k:<16}{s['comp']:>+7.0f}%{s['wk_mean']:>+8.2f}%{s['wk_median']:>+7.2f}%{s['in27']*100:>8.0f}%{s['pos']*100:>6.0f}%{s['maxdd']:>+7.0f}")
    eqw=np.mean([wknet[k] for k in names],axis=0); se=stats(eqw,oosw); out["oos"]["equal_blend"]=se
    print(f"  {'equal_blend':<16}{se['comp']:>+7.0f}%{se['wk_mean']:>+8.2f}%{se['wk_median']:>+7.2f}%{se['in27']*100:>8.0f}%{se['pos']*100:>6.0f}%{se['maxdd']:>+7.0f}")
    sr=stats(rank_net,oosw); out["oos"]["RANKER"]=sr
    print(f"  {'RANKER':<16}{sr['comp']:>+7.0f}%{sr['wk_mean']:>+8.2f}%{sr['wk_median']:>+7.2f}%{sr['in27']*100:>8.0f}%{sr['pos']*100:>6.0f}%{sr['maxdd']:>+7.0f}")
    # active-engine usage (OOS) + live
    use={}
    for w in range(NW):
        if oosw[w]: use[active[w]]=use.get(active[w],0)+1
    tot=sum(use.values()); out["active_usage_oos"]={k:round(v/tot,2) for k,v in use.items()}
    print(f"\n  ACTIVE-engine usage (OOS weeks): "+", ".join(f"{k} {v/tot:.0%}" for k,v in sorted(use.items(),key=lambda x:-x[1])))
    live_active=active[-1]; lastbarW=Wmats[live_active][T_-1]; live=sorted(((cols[j],round(float(lastbarW[j]),3)) for j in np.where(lastbarW>1e-6)[0]),key=lambda x:-x[1])
    out["live_active_engine"]=live_active; out["live_as_of"]=str(pd.Timestamp(idx[-1]).date()); out["live_book"]=live
    print(f"\n  *** LIVE as of {out['live_as_of']}: ACTIVE engine = {live_active} ({'DEPLOY '+str(len(live))+' names' if live else 'CASH'}) ***")
    for n,wt in live[:8]: print(f"      {n:<14}{wt:6.1%}")
    if not live: print("      [CASH this week]")
    with open(HERE/"strat_tiers_ranker.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\nSaved {HERE/'strat_tiers_ranker.json'}")


if __name__=="__main__":
    main()
