"""LENS=LOOK-AHEAD probe for MOM8_ER. Perturb one name's CLOSE at interior bar t0;
confirm MOM8_ER's net P&L stream changes ONLY at t>t0 (strict causality)."""
import warnings; warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, 'crypto/runs/engines/STRAT_TIERS')
sys.path.insert(0, 'crypto/src')
import numpy as np, pandas as pd
import strat_prod_deploy as P

f = P.build_all()
CAND = f['CAND']; idx = f['idx']; cols = f['cols']; T = f['T']; A = f['A']
net_of = f['net_of']; roo = f['roo']

Wm = CAND['MOM8_ER']
base_net = net_of(Wm)

# reproduce the exact caller's net formula independently to confirm net_of matches
Wl = np.vstack([np.zeros((1, A)), Wm[:-1]]); Wll = np.vstack([np.zeros((1, A)), Wl[:-1]])
from strat.canonical_exits import COST_PROD
man_net = np.nansum(Wl * roo, 1) - np.nansum(np.abs(Wl - Wll), 1) * COST_PROD / 2
print("COST_PROD =", COST_PROD)
print("net_of matches manual reconstruction:", np.allclose(base_net, man_net, atol=0, rtol=0))

om = idx >= pd.Timestamp('2024-05-15')
comp = float(np.prod(1 + base_net[om]) - 1)
print(f"OOS compound (from net_of) = {comp*100:+.1f}%   n_oos_bars={int(om.sum())}")

# ---- PERTURBATION: flip a name's close at several interior bars, check first changed P&L bar ----
# pick names that actually get selected somewhere so the perturbation can matter
sel_any = np.where((Wm > 0).any(axis=0))[0]
print(f"\n#names ever selected by MOM8_ER: {len(sel_any)}")

def first_changed(t0, j0, fac):
    fp = P.build_all(C_override=(t0, j0, fac))
    pert = fp['net_of'](fp['CAND']['MOM8_ER'])
    diff = np.abs(base_net - pert)
    changed = np.where(diff > 1e-12)[0]
    first = int(changed[0]) if len(changed) else -1
    return first, len(changed), diff

print("\nPERTURB close[t0,name]*fac -> first bar whose net P&L changes (must be > t0):")
print(f"{'t0':>5} {'date_t0':>12} {'name':>10} {'fac':>5} {'first_chg':>10} {'date_first':>12} {'delta':>5} {'#changed':>9} {'VERDICT':>16}")
leaks = []
# test a spread of interior bars, and specifically bars near slice-starts and mid-slice
BPD = P.BPD; SL = P.SL
test_bars = [T-40, T-60, T-100, T-200, T-13, T-8]
# also test names that are selected; use a few of them
names_to_test = list(sel_any[:4]) if len(sel_any) >= 4 else list(sel_any)
if 'ETHUSDT' in cols and cols.index('ETHUSDT') not in names_to_test:
    names_to_test.append(cols.index('ETHUSDT'))

for t0 in test_bars:
    if t0 < 30 or t0 >= T-2: continue
    for j0 in names_to_test:
        for fac in (1.5, 0.5, 3.0):
            first, nch, diff = first_changed(t0, j0, fac)
            ok = (first == -1) or (first > t0)
            v = "PASS" if ok else "LEAK<=t0!"
            if not ok: leaks.append((t0, cols[j0], fac, first))
            print(f"{t0:>5} {str(pd.Timestamp(idx[t0]).date()):>12} {cols[j0]:>10} {fac:>5} {first:>10} "
                  f"{(str(pd.Timestamp(idx[first]).date()) if first>=0 else 'none'):>12} {first-t0 if first>=0 else 0:>5} {nch:>9} {v:>16}")

print(f"\n=== LEAKS FOUND: {len(leaks)} ===")
for l in leaks: print("  LEAK:", l)
