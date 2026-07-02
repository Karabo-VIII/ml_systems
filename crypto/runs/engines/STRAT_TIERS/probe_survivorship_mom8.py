import sys, numpy as np, pandas as pd
sys.path.insert(0,'crypto/runs/engines/STRAT_TIERS'); sys.path.insert(0,'crypto/src')
import strat_prod_deploy as P

f=P.build_all(); W=f['CAND']['MOM8_ER']; roo=f['roo']; idx=f['idx']; cols=f['cols']; T=f['T']; A=f['A']

# --- recover the close panel to get per-name first-valid bar ---
# rebuild closes exactly as build_all does
closes={}
for s in P._u100_symbols():
    r=P._load_asset(s,"1d",P.WALL)
    if r is None: continue
    d,o,h,l,c=r; closes[s]=pd.Series(np.asarray(c,float),index=d)
cdf=pd.DataFrame(closes).sort_index()
cdf=cdf.reindex(idx).reindex(columns=cols)   # align exactly to book's idx/cols
Cpanel=cdf.to_numpy()

om=idx>=pd.Timestamp('2024-05-15')
oos_start_i=int(np.argmax(om))  # first True
print("OOS start idx position:",oos_start_i,"date:",idx[oos_start_i])
print("T,A:",T,A)

# per-name first valid (non-nan) bar
first_valid=np.full(A,T,dtype=int)
for j in range(A):
    fv=np.where(np.isfinite(Cpanel[:,j]))[0]
    if len(fv)>0: first_valid[j]=fv[0]

cutoff=oos_start_i-200
late=np.array([first_valid[j]>cutoff for j in range(A)])
print("cutoff idx (oos_start-200):",cutoff, "date:", idx[max(cutoff,0)])
print("num late-listers (first_valid > cutoff):",int(late.sum()),"of",A)
late_names=[cols[j] for j in range(A) if late[j]]
print("late-lister names:",late_names)

def net_of_W(Wm):
    Wl=np.vstack([np.zeros((1,A)),Wm[:-1]]); Wll=np.vstack([np.zeros((1,A)),Wl[:-1]])
    return np.nansum(Wl*roo,axis=1)-np.nansum(np.abs(Wl-Wll),axis=1)*(P.COST_PROD/2)

# base
net_base=net_of_W(W)
comp_base=np.prod(1+net_base[om])-1
print("\nBASE OOS compound:", round(comp_base*100,2),"%")

# pre-listed-only: zero weight columns for late-listers
Wp=W.copy(); Wp[:,late]=0.0
net_pre=net_of_W(Wp)
comp_pre=np.prod(1+net_pre[om])-1
print("PRE-LISTED-ONLY OOS compound:", round(comp_pre*100,2),"%")

# --- top-10 contributor names (OOS) via per-name pnl attribution ---
Wl=np.vstack([np.zeros((1,A)),W[:-1]])
per_name_pnl=(Wl*roo)  # gross per-name per-bar
contrib=np.nansum(per_name_pnl[om],axis=0)  # sum over OOS bars, per name (additive gross approx)
order=np.argsort(-contrib)
print("\nTop-10 OOS contributors (gross sum attribution):")
n_late_top10=0
for rk,j in enumerate(order[:10]):
    tag="LATE-LISTER" if late[j] else ""
    if late[j]: n_late_top10+=1
    print(f"  {rk+1:2d}. {cols[j]:14s} contrib={contrib[j]*100:7.2f}%  first_valid={idx[first_valid[j]].date() if first_valid[j]<T else 'never'} {tag}")
print("late-listers in top-10:",n_late_top10)

# also check exposure lost
print("\nfrac of OOS bars where MOM8_ER held ANY late-lister (weight>0):")
held_late=(Wl[:,late]>0).any(axis=1)
print("  ", round(held_late[om].mean()*100,2),"% of OOS bars")
