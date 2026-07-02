"""VERIFY the VIDYA@1d skill finding (98%ile, net +53%). Band is 2020-selected (disjoint from 2024-26 test) so no
config-survivorship into OOS. Remaining checks: (A) concentration/jackknife -- is +53%/skill from a few weeks/names?
(B) mechanistic -- is the skill the ADAPTIVE MA or the band ensemble? raw single VIDYA cross vs band-selected, same
skill test. (C) does it hold on VAL 2023-24 too? u100 1d, canonical held_rolling(14d/7d), next-open, prod cost.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE)); ROOT=Path(__file__).resolve().parents[3]/"src"; sys.path.insert(0,str(ROOT))
from strat_ti_family_neff import load_panel
from strat.ma_slice_engine import held_rolling, _cached_band_held
from strat.ma_type_upgrade import _vidya
from strat.canonical_exits import COST_PROD
import strat.fleet_lab as fl
BPD=fl.BARS_PER_DAY["1d"]; LB=14*BPD; SL=7*BPD; WALL=int(pd.Timestamp("2026-07-01").value//10**6)
OOS=pd.Timestamp("2024-05-15"); VAL0=pd.Timestamp("2023-07-01"); VAL1=pd.Timestamp("2024-05-15")


def main():
    data,_=load_panel(); syms=list(data)
    cdf=pd.DataFrame({s:pd.Series(data[s]["c"],index=data[s]["d"]) for s in syms}).sort_index()
    odf=pd.DataFrame({s:pd.Series(data[s]["o"],index=data[s]["d"]) for s in syms}).reindex(cdf.index)
    idx=cdf.index; cols=list(cdf.columns); T=len(idx); A=len(cols); O=odf.to_numpy()
    roo=np.full((T,A),0.0); roo[:-1]=O[1:]/np.where(O[:-1]>0,O[:-1],np.nan)-1.0; roo=np.nan_to_num(roo)
    valid=np.isfinite(cdf.to_numpy()); rng=np.random.default_rng(0)
    def held_band(s):
        c=data[s]["c"]; band=_cached_band_held("VIDYA","1d",s,data[s]["o"],data[s]["h"],data[s]["l"],c,WALL)
        return held_rolling(c,band,LB,SL)
    def held_raw(s,fast=14,slow=40):
        c=np.asarray(data[s]["c"],float); vf=_vidya(c,fast); vs=_vidya(c,slow)
        sig=np.nan_to_num(vf>vs).astype(np.int8); return held_rolling(c,sig,LB,SL)
    def book(hfn):
        H=np.zeros((T,A))
        for j,s in enumerate(syms):
            try: H[:,j]=pd.Series(hfn(s),index=data[s]["d"]).reindex(idx).fillna(0).to_numpy()
            except Exception: pass
        W=H/np.maximum(H.sum(axis=1,keepdims=True),1e-9); return W
    def netstream(W):
        Wl=np.vstack([np.zeros((1,A)),W[:-1]]); Wll=np.vstack([np.zeros((1,A)),Wl[:-1]])
        gross=np.nansum(np.where(valid,Wl*roo,0.0),axis=1); net=gross-np.nansum(np.abs(Wl-Wll),axis=1)*(COST_PROD/2)
        return gross,net,Wl
    def comp(x,m): return float(np.prod(1+x[m])-1)*100
    def skill(W,gross,m):
        real=comp(gross,m); null=[]
        for _ in range(200):
            tau=int(rng.integers(T//10,T-T//10)); Wr=np.roll(W,tau,axis=0)
            grr=np.nansum(np.where(valid,np.vstack([np.zeros((1,A)),Wr[:-1]])*roo,0.0),axis=1); null.append(comp(grr,m))
        return round(float((np.array(null)<real).mean())*100,0),round(real,1)
    om=np.asarray(idx>=OOS); vm=(idx>=VAL0)&(idx<VAL1)
    print("="*96); print("VERIFY VIDYA@1d skill (band 2020-selected, disjoint from 2024-26 test)"); print("="*96)
    out={}
    for nm,hfn in [("VIDYA_band",held_band),("VIDYA_raw(14/40)",held_raw)]:
        W=book(hfn); gross,net,Wl=netstream(W)
        sk_o,gr_o=skill(W,gross,om); nc_o=comp(net,om); sk_v,gr_v=skill(W,gross,vm); nc_v=comp(net,vm)
        # jackknife: drop top-k weeks (OOS)
        n=net[om]; st=[i for i in range(0,len(n)-SL+1,SL)]; wk=np.array([float(np.prod(1+n[i:i+SL])-1) for i in st])
        order=np.argsort(-wk); jk=[]
        for k in [0,1,2,3]:
            keep=np.ones(len(wk),bool); keep[order[:k]]=False; jk.append(round(float(np.prod(1+wk[keep])-1)*100,0))
        # per-asset OOS contribution (net attributed): approx via per-asset Wl*roo summed
        contrib=np.nansum(np.where(valid&om[:,None],Wl*roo,0.0),axis=0); top=np.argsort(-contrib)[:5]
        out[nm]={"OOS_net":round(nc_o,1),"OOS_gross":gr_o,"OOS_skill%":sk_o,"VAL_net":round(nc_v,1),"VAL_skill%":sk_v,
                 "jackknife_drop_top_0_1_2_3_wk":jk,"top5_assets":[(cols[j],round(float(contrib[j])*100,0)) for j in top]}
        print(f"\n  {nm}")
        print(f"    OOS net {nc_o:+.0f}% gross {gr_o:+.0f}% skill {sk_o:.0f}%ile | VAL net {nc_v:+.0f}% skill {sk_v:.0f}%ile")
        print(f"    jackknife drop top {{0,1,2,3}} wks (OOS compound%): {jk}")
        print(f"    top-5 asset contributors (net pp): {out[nm]['top5_assets']}")
    with open(HERE/"strat_vidya_verify.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    print(f"\n  READ: raw-VIDYA skill>=95 too => skill is the ADAPTIVE MA (band just denoises). skill holds on VAL =>")
    print(f"        cross-period (2020-selected band times 2023-24 AND 2024-26). jackknife robust (not 1-2 weeks) => real.")
    print(f"  Saved {HERE/'strat_vidya_verify.json'}")


if __name__=="__main__":
    main()
