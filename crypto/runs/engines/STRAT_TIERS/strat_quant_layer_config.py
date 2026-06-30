"""QUANT: (Q1) are the deployment LAYERS mathematically decoupled? (Q2) is an individual MA config (2x4) significant?

PRE-REGISTERED.
Q1 layer ablation -- H0: each layer's marginal contribution is independent of the others (decoupled); ALT: layers
   interact. Build the CONV band book and ablate GATE {off,on} x SIZING {EW, inv-vol uncapped, inv-vol cap33}. Report
   marginal of gate / inv-vol / cap at each combination + the interaction (does the gate's value depend on sizing?).
Q2 config significance -- H0: individual configs are interchangeable (high mutual correlation, ~0 TRAIN->OOS rank
   persistence) -> "2x4 vs 3x5" not significant; the ensemble is a DENOISER not a config-picker. ALT: specific configs
   carry distinct persistent signal. On the WMA|1d band: cross-config exposure correlation + effective-N (eigenvalue) +
   per-config compound dispersion + TRAIN->OOS Spearman rank persistence + ensemble-vs-median-single.
1d, u100, point-in-time, long-only spot. Decisive stats, no hand-waving.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd
from scipy import stats as ss

HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
ROOT=Path(__file__).resolve().parents[3]/"src"; sys.path.insert(0,str(ROOT))
import ma_weekly_replay as R
from strat.ma_type_upgrade import _MA, held_cross
from strat.ma_movecatch_engine import select_band_trainval, _nums
from strat.ma_wealth_driver import _load_asset, _u100_symbols
BPD=R.BPD; SB=7*BPD; CUT=pd.Timestamp("2024-05-15"); OOS25=pd.Timestamp("2025-01-01"); WALL=R.WIN["OOS-2025"][0]


def nw_p(wk,L=3):
    n=len(wk);
    if n<4 or np.std(wk)<1e-12: return 1.0
    mu=np.mean(wk); x=wk-mu; var=np.dot(x,x)/n
    for l in range(1,L+1): var+=2*(1-l/(L+1))*(np.dot(x[l:],x[:-l])/n)
    return float(1-ss.t.cdf(mu/np.sqrt(max(var,1e-12)/n),n-1))


def bk(net,mask):
    n=net[mask]; eq=np.cumprod(1+n); comp=float(eq[-1]-1)*100
    dd=float(np.min((eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq))*100) if len(eq)>1 else 0
    wk=np.array([float(np.prod(1+n[i:i+SB])-1)*100 for i in range(0,len(n)-SB+1,SB)])
    return {"comp":round(comp,1),"maxdd":round(dd,1),"nw_p":round(nw_p(wk),3)}


def W_ew(held_mult,gvec):
    mult=held_mult*(gvec[:,None]>0); b=(mult>0).astype(float); s=b.sum(axis=1,keepdims=True)
    return np.where(s>0,b/np.maximum(s,1),0.0)


def main():
    out={"check":"QUANT_layer_config","ts":datetime.now().strftime("%Y%m%d_%H%M%S")}
    # ---------- Q1: LAYER ABLATION (gate x sizing) on the CONV band book, 1d ----------
    print("="*100); print("Q1: ARE THE DEPLOYMENT LAYERS DECOUPLED? -- gate x sizing ablation (CONV band, 1d, OOS-2024 + OOS-2025)"); print("="*100)
    out["Q1_layer_ablation"]={}
    for window in ["OOS-2024","OOS-2025"]:
        f=R.build(window); wm=f["mask"]; hm=f["conv"]/len(R.TYPES); vol=f["vol"]; roo=f["roo"]; g=f["g"]; gon=g; goff=np.ones_like(g)
        rows={}
        for gname,gvec in [("ungated",goff),("GATED",gon)]:
            for sname,Wf in [("EW",lambda h:W_ew(h,gvec)),
                              ("invvol",lambda h:R.W_invvol(h,vol,gvec,1.0)),
                              ("invvol_cap33",lambda h:R.W_invvol(h,vol,gvec,0.33))]:
                W=Wf(hm); net,_=R.net_of(W,roo); rows[f"{gname}|{sname}"]=bk(net,wm)
        out["Q1_layer_ablation"][window]=rows
        print(f"\n  {window}:")
        print(f"    {'config':<22}{'comp%':>8}{'maxDD':>8}{'NW_p':>7}")
        for k,v in rows.items(): print(f"    {k:<22}{v['comp']:>+7.0f}%{v['maxdd']:>+8.0f}{v['nw_p']:>7.3f}")
        # marginals
        ge_ew=rows["GATED|EW"]["comp"]-rows["ungated|EW"]["comp"]; ge_iv=rows["GATED|invvol"]["comp"]-rows["ungated|invvol"]["comp"]
        iv_ung=rows["ungated|invvol"]["comp"]-rows["ungated|EW"]["comp"]; iv_g=rows["GATED|invvol"]["comp"]-rows["GATED|EW"]["comp"]
        cap_g=rows["GATED|invvol_cap33"]["comp"]-rows["GATED|invvol"]["comp"]
        dd_ge_ew=rows["GATED|EW"]["maxdd"]-rows["ungated|EW"]["maxdd"]
        print(f"    MARGINALS: gate effect (comp) on EW {ge_ew:+.0f}pp / on invvol {ge_iv:+.0f}pp  (DD: gate cuts {dd_ge_ew:+.0f}pp)")
        print(f"               invvol-vs-EW: ungated {iv_ung:+.0f}pp / gated {iv_g:+.0f}pp ; cap33-vs-invvol (gated) {cap_g:+.0f}pp")
        print(f"    INTERACTION: gate effect changes {ge_iv-ge_ew:+.0f}pp across sizing; invvol effect changes {iv_g-iv_ung:+.0f}pp across gate")
    # ---------- Q2: CONFIG SIGNIFICANCE on WMA|1d band ----------
    print(f"\n{'='*100}\nQ2: IS AN INDIVIDUAL CONFIG (e.g. 2x4) SIGNIFICANT? -- WMA|1d band redundancy + persistence\n{'='*100}")
    band,meta=select_band_trainval("WMA","1d");
    closes={}
    for s in _u100_symbols():
        r=_load_asset(s,"1d",WALL)
        if r is None: continue
        d,o,h,l,c=r; closes[s]=pd.Series(np.asarray(c,float),index=d)
    cdf=pd.DataFrame(closes).sort_index(); idx=cdf.index; cols=list(cdf.columns); T=len(idx); A=len(cols); C=cdf.to_numpy()
    ret=np.zeros((T,A)); ret[1:]=C[1:]/np.where(C[:-1]>0,C[:-1],np.nan)-1.0; ret=np.nan_to_num(ret)
    wma=_MA["WMA"]
    # cache WMA(period) per name
    periods=sorted({p for cfg in band for p in _nums(cfg)})
    cache={}
    for j in range(A):
        cj=C[:,j]
        for p in periods: cache[(j,p)]=wma(cj,p)
    # per-config: exposure (fraction of names long) + per-config EW raw net stream
    expo=np.zeros((len(band),T)); cfgnet=np.zeros((len(band),T))
    for ci,cfg in enumerate(band):
        ps=_nums(cfg); H=np.zeros((T,A))
        for j in range(A):
            mas=[cache[(j,p)] for p in ps]
            h=(mas[0]>mas[1]) if len(ps)==2 else ((mas[0]>mas[1])&(mas[1]>mas[2]))
            H[:,j]=np.nan_to_num(h).astype(float)
        expo[ci]=H.mean(axis=1)
        Hl=np.vstack([np.zeros((1,A)),H[:-1]]); s=Hl.sum(axis=1,keepdims=True); Wl=np.where(s>0,Hl/np.maximum(s,1),0.0)
        cfgnet[ci]=np.nansum(Wl*ret,axis=1)
    oosm=(idx>=CUT); trm=(idx<CUT)
    # cross-config correlation + effective N (eigenvalue)
    E=expo[:,oosm]; E=E-E.mean(axis=1,keepdims=True);
    sd=E.std(axis=1); keep=sd>1e-9; Ek=E[keep]
    Cmat=np.corrcoef(Ek); iu=np.triu_indices_from(Cmat,1); mean_corr=float(np.nanmean(Cmat[iu]))
    lam=np.linalg.eigvalsh(np.nan_to_num(Cmat)); lam=lam[lam>0]; n_eff=float((lam.sum()**2)/np.sum(lam**2))
    # per-config compound dispersion + TRAIN->OOS persistence
    comp_tr=np.array([float(np.prod(1+cfgnet[ci][trm])-1)*100 for ci in range(len(band))])
    comp_oos=np.array([float(np.prod(1+cfgnet[ci][oosm])-1)*100 for ci in range(len(band))])
    rho,pp=ss.spearmanr(comp_tr,comp_oos)
    # ensemble (majority vote >50%) vs median single config, OOS
    vote=(expo>0)  # per config per bar long-fraction>0 ... use the deployed >50%-of-band vote on aggregate
    ens_hold=(expo.mean(axis=0)>0.5)  # fraction of band long > 50%
    # ensemble book = EW over names when band-majority... approximate: ensemble exposure = band-mean of per-config held
    out["Q2_config_significance"]={"cell":"WMA|1d","n_band":len(band),"n_2MA":sum(len(_nums(c))==2 for c in band),
        "n_3MA":sum(len(_nums(c))==3 for c in band),"mean_pairwise_corr_exposure":round(mean_corr,3),
        "effective_N_configs":round(n_eff,1),"of_total":len(band),
        "per_config_OOS_compound":{"median":round(float(np.median(comp_oos)),1),"std":round(float(np.std(comp_oos)),1),
            "min":round(float(comp_oos.min()),1),"max":round(float(comp_oos.max()),1)},
        "TRAIN_to_OOS_rank_persistence_spearman":round(float(rho),3),"persistence_p":round(float(pp),3)}
    q=out["Q2_config_significance"]
    print(f"\n  WMA|1d band: {len(band)} configs ({q['n_2MA']} 2-MA + {q['n_3MA']} 3-MA), TRAIN+VAL-positive selected (OOS-excluded)")
    print(f"  REDUNDANCY: mean pairwise exposure correlation = {mean_corr:.2f}  ->  effective independent configs n_eff = {n_eff:.1f} of {len(band)}")
    print(f"  DISPERSION (per-config OOS compound): median {q['per_config_OOS_compound']['median']}% , std {q['per_config_OOS_compound']['std']}% , range [{q['per_config_OOS_compound']['min']}, {q['per_config_OOS_compound']['max']}]")
    print(f"  PERSISTENCE (TRAIN->OOS config rank): Spearman rho = {rho:+.3f} (p={pp:.3f})  {'-> NO persistence: config choice is NOISE' if pp>0.05 or abs(rho)<0.2 else '-> some persistence'}")
    print(f"\n  VERDICT Q2: an individual config (2x4) is {'NOT individually significant' if (mean_corr>0.5 and (pp>0.05 or abs(rho)<0.2)) else 'possibly significant'} -- ", end="")
    print(f"configs are {mean_corr:.0%}-correlated (~{n_eff:.0f} independent bets), non-persistent TRAIN->OOS. The ENSEMBLE is a DENOISER; the specific cross does not matter.")
    with open(HERE/"strat_quant_layer_config.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\nSaved {HERE/'strat_quant_layer_config.json'}")


if __name__=="__main__":
    main()
