import sys, numpy as np, pandas as pd
sys.path.insert(0,'crypto/runs/engines/STRAT_TIERS'); sys.path.insert(0,'crypto/src')
import strat_prod_deploy as P
f=P.build_all(); W=f['CAND']['MOM8_ER']; roo=f['roo']; idx=f['idx']; cols=f['cols']; T=f['T']; A=f['A']
closes={}
for s in P._u100_symbols():
    r=P._load_asset(s,"1d",P.WALL)
    if r is None: continue
    d,o,h,l,c=r; closes[s]=pd.Series(np.asarray(c,float),index=d)
cdf=pd.DataFrame(closes).sort_index().reindex(idx).reindex(columns=cols)
Cpanel=cdf.to_numpy()
om=idx>=pd.Timestamp('2024-05-15'); oos_i=int(np.argmax(om))
first_valid=np.array([ (np.where(np.isfinite(Cpanel[:,j]))[0][0] if np.isfinite(Cpanel[:,j]).any() else T) for j in range(A)])
def net_of_W(Wm):
    Wl=np.vstack([np.zeros((1,A)),Wm[:-1]]); Wll=np.vstack([np.zeros((1,A)),Wl[:-1]])
    return np.nansum(Wl*roo,axis=1)-np.nansum(np.abs(Wl-Wll),axis=1)*(P.COST_PROD/2)
comp_base=np.prod(1+net_of_W(W)[om])-1
print("BASE:",round(comp_base*100,2),"%")
for buf,label in [(0,"cutoff=OOS start (drop any listing >=2024-05-15)"),(200,"buffer 200"),(400,"buffer 400 (PURGE_GAP)"),(2000,"buffer 2000 (drop anything not present at series start)")]:
    cutoff=oos_i-buf
    late=first_valid>cutoff
    Wp=W.copy(); Wp[:,late]=0.0
    comp=np.prod(1+net_of_W(Wp)[om])-1
    print(f"  {label:60s} nlate={int(late.sum()):2d}  pre-listed OOS={comp*100:7.2f}%  delta={ (comp-comp_base)*100:6.2f}pp")
