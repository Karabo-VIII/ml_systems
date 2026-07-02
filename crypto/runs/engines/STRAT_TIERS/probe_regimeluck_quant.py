# REGIME-LUCK / SUBPERIOD-CONCENTRATION probe for MOM8_ER (quant referee)
# Re-derives the OOS net stream from the REAL book, decomposes compound by calendar quarter,
# counts deploy weeks, and measures single-quarter / top-3-week concentration.
import sys, numpy as np, pandas as pd
sys.path.insert(0, "crypto/runs/engines/STRAT_TIERS")
sys.path.insert(0, "crypto/src")
import strat_prod_deploy as P

f = P.build_all()
W = f["CAND"]["MOM8_ER"]; roo = f["roo"]; idx = f["idx"]; cols = f["cols"]; T = f["T"]; A = f["A"]
COST = 0.0028

# Net stream EXACTLY as prompt specifies (== book net_of)
Wl  = np.vstack([np.zeros((1, A)), W[:-1]])
Wll = np.vstack([np.zeros((1, A)), Wl[:-1]])
net = np.nansum(Wl * roo, axis=1) - np.nansum(np.abs(Wl - Wll), axis=1) * COST / 2.0
# sanity vs book net_of
net_book = f["net_of"](W)
assert np.allclose(net, net_book, atol=0, rtol=0), "net stream != book net_of"

om = idx >= pd.Timestamp("2024-05-15")
comp_oos = np.prod(1 + net[om]) - 1.0
print(f"OOS compound (full)                 : {comp_oos*100:.2f}%   (bars={int(om.sum())})")
print(f"[sanity] net==book net_of           : True")

# ---- per-calendar-quarter compound ----
s_net = pd.Series(net, index=idx)
s_oos = s_net[om]
q = s_oos.index.to_period("Q")
print("\n-- per-OOS-calendar-quarter net compound --")
qrows = []
for period in sorted(q.unique()):
    m = (q == period)
    c = np.prod(1 + s_oos[m].values) - 1.0
    nb = int(m.sum())
    qrows.append((str(period), c, nb))
    print(f"  {str(period):8s}  compound={c*100:8.2f}%   bars={nb}")

n_q = len(qrows)
n_pos_q = sum(1 for _, c, _ in qrows if c > 0)
print(f"\n  quarters total={n_q}  positive={n_pos_q}  ({n_pos_q}/{n_q})")

# fraction of TOTAL compound from single best quarter.
# Use log-return additive decomposition so 'fraction of compound' is well-defined & sums to 1.
log_tot = np.log1p(net[om]).sum()
print(f"\n  log(1+comp_oos) total = {log_tot:.4f}  (exp-1 = {np.expm1(log_tot)*100:.2f}%)")
q_logs = []
for period in sorted(q.unique()):
    m = (q == period)
    lg = np.log1p(s_oos[m].values).sum()
    q_logs.append((str(period), lg))
best_q_name, best_q_log = max(q_logs, key=lambda x: x[1])
frac_best_q = best_q_log / log_tot if log_tot != 0 else float("nan")
print(f"  best quarter (by log-contrib) = {best_q_name}  log={best_q_log:.4f}  "
      f"fraction-of-total-log-compound = {frac_best_q*100:.1f}%")

# ---- deploy weeks (gate-on slices) ----
# A deploy-week = a slice-start i where the book actually took a position (row sum > 0).
SL = P.SL
starts = [i for i in range(0, T - SL + 1, SL)]
oos_starts = [i for i in starts if idx[i] >= pd.Timestamp("2024-05-15")]
deployed = [i for i in oos_starts if W[i].sum() > 1e-12]
print(f"\n-- deploy weeks (OOS) --")
print(f"  OOS slice-starts        = {len(oos_starts)}")
print(f"  deployed (gate-on)      = {len(deployed)}")
print(f"  deploy rate             = {len(deployed)/max(len(oos_starts),1)*100:.1f}%")

# per-deploy-week net compound (the 7d held slice return, net of cost, aligned to the held bars)
# The slice decided at start i is HELD over bars i+1..i+SL (Wl earns roo). Sum net over those bars.
wk_rets = []
for i in deployed:
    seg = net[i+1 : i+1+SL]              # bars where W[i] is the lagged holding
    r = np.prod(1 + seg) - 1.0
    wk_rets.append((idx[i], r))
wk_arr = np.array([r for _, r in wk_rets])
nz = np.sum(np.abs(wk_arr) > 1e-9)
print(f"  distinct non-zero deploy-week returns = {int(nz)}")
if len(wk_arr):
    print(f"  deployed-week median return           = {np.median(wk_arr)*100:.2f}%")
    print(f"  deployed-week mean return             = {np.mean(wk_arr)*100:.2f}%")
    print(f"  deployed-week win rate (>0)           = {np.mean(wk_arr>0)*100:.1f}%")

# ---- top-3 deploy-week concentration ----
# fraction of total log-compound from the 3 best deploy-weeks.
wk_logs = np.array([np.log1p(net[i+1:i+1+SL]).sum() for i in deployed])
order = np.argsort(-wk_logs)
top3_log = wk_logs[order[:3]].sum()
frac_top3 = top3_log / log_tot if log_tot != 0 else float("nan")
print(f"\n  top-3 deploy-weeks log-contrib = {top3_log:.4f}  "
      f"fraction-of-total-log-compound = {frac_top3*100:.1f}%")
print(f"  top-3 deploy-week dates+rets:")
for j in order[:3]:
    dt, r = wk_rets[j]
    print(f"     {str(dt.date())}  ret={r*100:7.2f}%  log={wk_logs[j]:.4f}")

# ---- VERDICT logic ----
majority_pos = n_pos_q > n_q / 2.0
not_one_burst = frac_best_q <= 0.60
print("\n" + "="*70)
print(f"majority quarters positive : {majority_pos}  ({n_pos_q}/{n_q})")
print(f"best-quarter <= 60% compound: {not_one_burst}  (best={frac_best_q*100:.1f}%)")
verdict = "SURVIVES" if (majority_pos and not_one_burst) else "REFUTED"
print(f"VERDICT (regime-luck lens) : {verdict}")
print("="*70)
