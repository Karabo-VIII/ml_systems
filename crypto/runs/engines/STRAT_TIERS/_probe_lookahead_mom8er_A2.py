"""Sharper look-ahead probe: find perturbations that DO move MOM8_ER's P&L, then
verify first-changed P&L bar is strictly > t0. Also test SAME-BAR: perturb close[t0]
where t0 is a slice-start; a leak would change net at t0 itself."""
import warnings; warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, 'crypto/runs/engines/STRAT_TIERS')
sys.path.insert(0, 'crypto/src')
import numpy as np, pandas as pd
import strat_prod_deploy as P

f = P.build_all()
CAND = f['CAND']; idx = f['idx']; cols = f['cols']; T = f['T']; A = f['A']
net_of = f['net_of']; roo = f['roo']; gbin=f['gbin']; er=f['er']; vol=f['vol']
Wm = CAND['MOM8_ER']; base_net = net_of(Wm)
BPD = P.BPD; SL = P.SL
starts = [i for i in range(0, T-SL+1, SL)]

# For each slice-start, which names are selected (nonzero weight in that slice)?
# Perturbing a selected name's close AT the slice-start i must NOT change net at t<=i (decision uses close[i], trades open[i+1]).
def first_changed(t0, j0, fac):
    fp = P.build_all(C_override=(t0, j0, fac))
    pert = fp['net_of'](fp['CAND']['MOM8_ER'])
    diff = np.abs(base_net - pert)
    ch = np.where(diff > 1e-12)[0]
    return (int(ch[0]) if len(ch) else -1), len(ch)

# find slice-starts where gate is on (a deploy slice) so perturbations can matter
gate = (gbin*(er>0.35).astype(int))
deploy_starts = [i for i in starts if i>30 and i<T-3 and gate[i]>0]
print(f"deploy slice-starts (gate on): {len(deploy_starts)}  (of {len(starts)} starts)")

# strategy: at each deploy start i, perturb ONE selected name and ONE non-selected-but-close name.
# The MOST dangerous test = perturb close[i] itself (slice-start). If selection or gate at i uses close[i]
# to pick, that's allowed (as-of i); the ONLY leak is if net at bar <= i changes.
tested=0; moved=0; leaks=[]; same_bar_moves=[]
np.random.seed(1)
# sample deploy starts to keep runtime bounded, but include the first, mid, last
sample = deploy_starts[::max(1,len(deploy_starts)//25)]
print(f"sampling {len(sample)} deploy starts\n")
print(f"{'t0':>5} {'date':>12} {'name':>10} {'sel?':>5} {'fac':>4} {'first_chg':>9} {'dfirst':>12} {'gap':>4} {'#ch':>5} {'V':>10}")
for i in sample:
    selnames = np.where(Wm[i] > 0)[0]
    # candidate names to perturb: selected ones + a few near-miss by momentum
    cands = list(selnames[:3])
    # add a couple of high-momentum non-selected names
    for j in range(min(A,20)):
        if j not in cands and np.isfinite(f['brd'][i]): cands.append(j); break
    for j0 in cands[:4]:
        for fac in (1.5, 0.5):
            first, nch = first_changed(i, j0, fac)
            tested += 1
            issel = j0 in selnames
            if first != -1: moved += 1
            leak = (first != -1) and (first <= i)
            samebar = (first == i)
            if leak: leaks.append((i, cols[j0], fac, first))
            if samebar: same_bar_moves.append((i, cols[j0], fac))
            if first != -1:  # only print the ones that moved, to see WHERE
                v = "LEAK" if leak else "ok"
                print(f"{i:>5} {str(pd.Timestamp(idx[i]).date()):>12} {cols[j0]:>10} {str(bool(issel)):>5} {fac:>4} "
                      f"{first:>9} {str(pd.Timestamp(idx[first]).date()):>12} {first-i:>4} {nch:>5} {v:>10}")

print(f"\ntested={tested} moved={moved}")
print(f"SAME-BAR moves (first_chg == t0): {len(same_bar_moves)}")
for s in same_bar_moves: print("   SAME-BAR:", s)
print(f"LEAKS (first_chg <= t0): {len(leaks)}")
for l in leaks: print("   LEAK:", l)

# --- explicit min-gap check: across all perturbations that moved, what's the MIN (first_chg - t0)? ---
