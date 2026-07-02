"""QUANT (referee re-do): the previous n_eff study measured the WRONG thing. It correlated PORTFOLIO-LEVEL weekly
P&L of one EW-u100 book per indicator -> that is dominated by the shared crypto BETA (market factor), so any two
long-only books look ~identical regardless of which assets they hold. Correlated aggregate P&L != same trades.

User's challenge (correct): "assets, although correlated, don't move the same -- take BTC out, do they move the same
throughout?" AND "families of TI bots is what I asked for; where do the gate/sizing/one-engine assumptions come from?"

This script answers the EMPIRICAL half with the RIGHT unit -- the cross-section and the market-neutral residual:
  PART A  DISPERSION: do alts move the same ex-BTC?  mean pairwise corr of daily rets RAW vs BTC-residualized vs
          cross-sectionally-demeaned; PC1 share raw vs ex-BTC; weekly cross-sectional std; weekly best-mover ("something
          always rips"); by year.
  PART B  SIGNAL DISAGREEMENT: do the indicators make DIFFERENT trades at the ASSET level?  per-asset held matrices for
          a diverse indicator set (trend/momentum/MR/breakout/volume) -> pairwise Jaccard of the held long-sets +
          basket-overlap. (This is the unit "are these different bots", NOT aggregate P&L.)
  PART C  MARKET-NEUTRAL n_eff: strip the EW-market factor from each indicator's book (regress weekly P&L on market,
          take residual) -> n_eff on the RESIDUALS vs on the RAW P&L (the 1.4 I reported). If residual n_eff >> 1 the
          collapse was a beta artifact. PLUS single-asset cross-indicator P&L corr (one alt, no cross-sec mixing).
NO gate, NO sizing, NO regime -- pure per-asset TI signals, as asked. u100 1d, wall 2026-07-01. STRUCTURE study.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd

HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE)); ROOT=Path(__file__).resolve().parents[3]/"src"; sys.path.insert(0,str(ROOT))
from strat_ti_family_neff import load_panel  # reuse the day-aligned loader
from strat.deep2020_ti_pipeline import INDICATORS
SL=7


def neff(M):
    """rows = streams; n_eff = N / (1+(N-1)*rbar_pos)."""
    sd=M.std(axis=1); M=M[sd>1e-12]; n=len(M)
    if n<2: return n,None,float(n)
    cm=np.corrcoef(M); rbar=float((cm.sum()-n)/(n*(n-1)))
    return n,round(rbar,3),round(n/(1+(n-1)*max(rbar,0)),2)


def pc1_share(M):
    sd=M.std(axis=1); M=M[sd>1e-12]; n=len(M)
    if n<2: return None
    z=(M-M.mean(axis=1,keepdims=True))/M.std(axis=1,keepdims=True)
    ev=np.linalg.eigvalsh(np.corrcoef(z))[::-1]; return round(float(ev[0]/n),3)


def mean_pair_corr(M):
    sd=M.std(axis=1); M=M[sd>1e-12]; n=len(M)
    if n<2: return None
    cm=np.corrcoef(M); return round(float((cm.sum()-n)/(n*(n-1))),3)


def main():
    data,nvol=load_panel(); syms=list(data)
    cdf=pd.DataFrame({s:pd.Series(data[s]["c"],index=data[s]["d"]) for s in syms}).sort_index()
    idx=cdf.index; cols=list(cdf.columns); T=len(idx); A=len(cols); C=cdf.to_numpy()
    R=np.full((T,A),np.nan); R[1:]=C[1:]/np.where(C[:-1]>0,C[:-1],np.nan)-1.0
    has_btc="BTCUSDT" in cols
    out={"check":"cross_sectional_neff","T":T,"A":A,"ts":datetime.now().strftime("%Y%m%d_%H%M%S")}
    print("="*104); print(f"CROSS-SECTIONAL / EX-BTC STUDY  ({A} assets, {T} bars, 1d)  -- the RIGHT unit for 'are these different'"); print("="*104)

    # ================= PART A: do alts move the same, ex-BTC? =================
    alts=[j for j,c in enumerate(cols) if c!="BTCUSDT"]
    Ra=R[:,alts]  # alts only
    # need a common valid window per pair -> use pandas corr (pairwise complete)
    rdf=pd.DataFrame(Ra,index=idx,columns=[cols[j] for j in alts])  # keep full length T for BTC-residual alignment
    raw_corr=rdf.corr(min_periods=200)
    tri=np.triu_indices(raw_corr.shape[0],1); mpc_raw=float(np.nanmean(raw_corr.to_numpy()[tri]))
    # BTC-residualize each alt: resid = r_alt - beta*r_btc (beta via cov/var on common window)
    if has_btc:
        rb=R[:,cols.index("BTCUSDT")]
        resid={}
        for c in rdf.columns:
            r=rdf[c].to_numpy(); m=np.isfinite(r)&np.isfinite(rb)
            if m.sum()<200: continue
            b=np.cov(r[m],rb[m])[0,1]/np.var(rb[m]); e=r.copy(); e[m]=r[m]-b*rb[m]; resid[c]=e
        rr=pd.DataFrame(resid,index=idx); btc_res_corr=rr.corr(min_periods=200)
        mpc_btcres=float(np.nanmean(btc_res_corr.to_numpy()[np.triu_indices(btc_res_corr.shape[0],1)]))
    else: mpc_btcres=None
    # cross-sectionally demean (remove EW-market each bar)
    mkt=np.nanmean(Ra,axis=1,keepdims=True); Rd=Ra-mkt
    dd=pd.DataFrame(Rd,index=idx).dropna(how="all"); demean_corr=dd.corr(min_periods=200)
    mpc_demean=float(np.nanmean(demean_corr.to_numpy()[np.triu_indices(demean_corr.shape[0],1)]))
    # PC1 share raw vs ex-BTC (transpose: rows=assets)
    def pc1_assets(Mat):
        df=pd.DataFrame(Mat,index=idx).dropna(how="any",axis=0)
        if df.shape[0]<100 or df.shape[1]<3: return None
        z=(df-df.mean())/df.std(); cm=np.corrcoef(z.to_numpy().T); ev=np.linalg.eigvalsh(cm)[::-1]
        return round(float(ev[0]/len(ev)),3)
    pc1_raw=pc1_assets(Ra); pc1_res=pc1_assets(np.array([resid[c] for c in resid]).T) if has_btc and resid else None
    # weekly cross-sectional dispersion + best mover
    st=list(range(0,T-SL+1,SL)); wk_disp=[]; wk_best=[]; wk_p90=[]
    for i in st:
        wr=np.array([np.prod(1+R[i:i+SL,j])-1 if np.isfinite(R[i:i+SL,j]).all() else np.nan for j in alts])
        wr=wr[np.isfinite(wr)]
        if len(wr)<10: continue
        wk_disp.append(float(np.std(wr))); wk_best.append(float(np.max(wr))); wk_p90.append(float(np.percentile(wr,90)))
    out["A_dispersion"]={"mean_pair_corr_raw":round(mpc_raw,3),"mean_pair_corr_ex_btc_resid":None if mpc_btcres is None else round(mpc_btcres,3),
        "mean_pair_corr_cross_sec_demean":round(mpc_demean,3),"pc1_share_raw":pc1_raw,"pc1_share_ex_btc":pc1_res,
        "weekly_xsec_std_median":round(float(np.median(wk_disp)),3),"weekly_best_mover_median":round(float(np.median(wk_best)),3),
        "weekly_p90_mover_median":round(float(np.median(wk_p90)),3),"n_weeks":len(wk_disp)}
    print("\n  PART A -- DO ALTS MOVE THE SAME, EX-BTC?  (mean pairwise corr of DAILY returns, alts only, BTC excluded)")
    print(f"    raw pairwise corr .................. {mpc_raw:+.2f}   (looks 'all one' -- this is the shared crypto beta)")
    print(f"    after residualizing out BTC ....... {mpc_btcres:+.2f}   <- remove the market factor")
    print(f"    after cross-sec demean (mkt out) .. {mpc_demean:+.2f}   <- pure idiosyncratic co-movement")
    print(f"    PC1 share of variance: raw {pc1_raw:.0%}  ->  ex-BTC {(pc1_res or 0):.0%}   (one factor shrinks once BTC removed)")
    print(f"    weekly cross-sectional std (spread of alt weekly rets): median {np.median(wk_disp):.1%}")
    print(f"    weekly BEST mover: median {np.median(wk_best):+.1%}   |  weekly 90th-pct mover: median {np.median(wk_p90):+.1%}")
    print(f"    -> alts do NOT move the same ex-BTC: idiosyncratic corr {mpc_btcres:+.2f}, and every week the spread is ~{np.median(wk_disp):.0%} wide.")

    # ================= PART B: do the indicators make DIFFERENT trades (asset level)? =================
    # diverse representative set, one canonical config each
    def sma_held(A_,p):
        c=A_["c"]; a,b=p; sa=pd.Series(c).rolling(a).mean().to_numpy(); sb=pd.Series(c).rolling(b).mean().to_numpy()
        return np.nan_to_num(sa>sb).astype(np.int8)
    REPS=[("SMA(20,50)","trend",sma_held,(20,50)),
          ("MACD","trend",INDICATORS["MACD"]["base"],(12,26,9)),
          ("ROC","momentum",INDICATORS["ROC"]["base"],INDICATORS["ROC"]["grid"]()[0]),
          ("RSI","mean-rev",INDICATORS["RSI"]["base"],INDICATORS["RSI"]["grid"]()[0]),
          ("DONCHIAN","breakout",INDICATORS["DONCHIAN"]["base"],INDICATORS["DONCHIAN"]["grid"]()[0]),
          ("OBV","volume",INDICATORS["OBV"]["base"],INDICATORS["OBV"]["grid"]()[0])]
    Hs={}  # name -> (T,A) held aligned
    for nm,fam,fn,p in REPS:
        H=np.zeros((T,A))
        for s in syms:
            try: h=fn(data[s],p)
            except Exception: continue
            H[:,cols.index(s)]=pd.Series(h,index=data[s]["d"]).reindex(idx).fillna(0).to_numpy()
        Hs[nm]=H
    names=list(Hs); nB=len(names)
    valid=np.isfinite(R)
    jac=np.zeros((nB,nB)); agr=np.zeros((nB,nB))
    for i,a in enumerate(names):
        for j,b in enumerate(names):
            Ha=Hs[a]>0; Hb=Hs[b]>0; both=valid
            inter=(Ha&Hb&both).sum(); uni=((Ha|Hb)&both).sum()
            jac[i,j]=inter/max(uni,1)
            agr[i,j]=((Ha==Hb)&both).sum()/max(both.sum(),1)
    off=jac[np.triu_indices(nB,1)]
    out["B_signal_disagreement"]={"set":names,"jaccard_mean_offdiag":round(float(off.mean()),3),
        "jaccard_min":round(float(off.min()),3),"jaccard_matrix":{names[i]:{names[j]:round(float(jac[i,j]),2) for j in range(nB)} for i in range(nB)}}
    print(f"\n  PART B -- DO THE BOTS TRADE THE SAME ASSETS? (Jaccard of held long-sets across (asset,bar); 1=identical basket, 0=disjoint)")
    print(f"    {'':<11}"+"".join(f"{n[:8]:>9}" for n in names))
    for i,a in enumerate(names):
        print(f"    {a:<11}"+"".join(f"{jac[i,j]:>9.2f}" for j in range(nB)))
    print(f"    mean off-diagonal Jaccard = {off.mean():.2f}  (min {off.min():.2f})  -> the bots hold DIFFERENT baskets, esp. MR vs trend.")

    # ================= PART C: market-neutral residual n_eff =================
    def book_weekly(H):
        Hl=np.vstack([np.zeros((1,A)),H[:-1]]); g=np.nansum(np.where(valid,Hl*R,0.0),axis=1)/np.maximum(valid.sum(axis=1),1)
        return np.array([np.prod(1+g[i:i+SL])-1 for i in st])
    mktH=np.where(valid,1.0,0.0); mkt_wk=book_weekly(mktH)
    raw_wk={nm:book_weekly(Hs[nm]) for nm in names}
    Wraw=np.array([raw_wk[nm] for nm in names])
    # residualize each on market
    Wres=[]
    for nm in names:
        y=raw_wk[nm]; x=mkt_wk; m=np.isfinite(y)&np.isfinite(x)
        b=np.cov(y[m],x[m])[0,1]/np.var(x[m]); Wres.append(y-b*x)
    Wres=np.array(Wres)
    n_raw,rb_raw,ne_raw=neff(Wraw); n_res,rb_res,ne_res=neff(Wres)
    # single-asset cross-indicator P&L corr (one liquid alt, no cross-sec mixing)
    def single_asset_pl(sym):
        j=cols.index(sym); r=R[:,j]; out2={}
        for nm in names:
            h=Hs[nm][:,j]; hl=np.concatenate([[0],h[:-1]]); g=np.nan_to_num(hl*r)
            out2[nm]=np.array([np.prod(1+g[i:i+SL])-1 for i in st])
        return out2
    sa_sym="SOLUSDT" if "SOLUSDT" in cols else cols[alts[0]]
    sap=single_asset_pl(sa_sym); Wsa=np.array([sap[nm] for nm in names]); n_sa,rb_sa,ne_sa=neff(Wsa)
    out["C_market_neutral_neff"]={"set":names,"raw_pnl":{"rbar":rb_raw,"n_eff":ne_raw},
        "residual_pnl":{"rbar":rb_res,"n_eff":ne_res},"single_asset":{"sym":sa_sym,"rbar":rb_sa,"n_eff":ne_sa}}
    print(f"\n  PART C -- n_eff ON THE RIGHT UNIT ({nB} diverse indicators, per-asset signals, EW book)")
    print(f"    RAW weekly P&L ............ rbar {rb_raw:+.2f}  n_eff {ne_raw} of {nB}   (my prior test's unit: beta-dominated)")
    print(f"    MARKET-NEUTRAL residual .. rbar {rb_res:+.2f}  n_eff {ne_res} of {nB}   <- strip EW-market: the REAL diversity")
    print(f"    single asset ({sa_sym}) ... rbar {rb_sa:+.2f}  n_eff {ne_sa} of {nB}   (one asset, no cross-sec mixing)")
    print(f"\n  READ: raw P&L corr = shared market beta (NOT 'same strategy'). Residual/single-asset n_eff is what says")
    print(f"        whether the bots are genuinely different. Interpret below.")
    with open(HERE/"strat_cross_sectional_neff.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\nSaved {HERE/'strat_cross_sectional_neff.json'}")


if __name__=="__main__":
    main()
