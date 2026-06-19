"""ADVERSARIAL REFEREE for the ORC fleet search. Independent re-derivation on fleet_lab (DEV-walled).
Does NOT trust the lane scripts -- re-derives from the harness directly with paired multi-seed stats.
No emoji. No git. DEV only.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(ROOT))
from strat.fleet_lab import load_wide, invoke, fleet_invoke, slice_dates, DEV_END, COST
from strat.fleet_search import load_4h_wide, merge_lab_4h

HOLD = 7; K = 5


def ew_on(lab, ds, hold=HOLD):
    C = lab["C"]; out = []
    for d in ds:
        if d + hold >= len(C.index):
            out.append(np.nan); continue
        vs = [C[s].iloc[d+hold]/C[s].iloc[d]-1 for s in C.columns
              if pd.notna(C[s].iloc[d]) and pd.notna(C[s].iloc[d+hold])]
        out.append(float(np.mean(vs)) if vs else np.nan)
    return np.array(out)


def agent_on(lab, feats, ds, signs=None, hold=HOLD, K=K):
    return np.array([invoke(lab, feats, d, hold, K, signs) for d in ds], dtype=float)


def fleet_on(lab, fleet, ds, hold=HOLD):
    return np.array([fleet_invoke(lab, fleet, d, hold) for d in ds], dtype=float)


def stats(rois, ew=None):
    r = rois[~np.isnan(rois)]
    d = {"n": len(r), "mean": 100*np.mean(r), "profit": 100*np.mean(r > 0),
         "worst": 100*np.min(r) if len(r) else np.nan}
    if ew is not None:
        m = ~np.isnan(rois) & ~np.isnan(ew)
        d["beatEW"] = 100*np.mean(rois[m] > ew[m])
        d["excess"] = 100*np.mean(rois[m] - ew[m])
    return d


def paired_t(a, b):
    m = ~np.isnan(a) & ~np.isnan(b)
    diff = a[m] - b[m]
    if len(diff) < 3: return np.nan, np.nan
    md = diff.mean(); se = diff.std(ddof=1)/np.sqrt(len(diff))
    t = md/se if se > 0 else np.nan
    # two-sided p via normal approx
    from math import erf, sqrt
    p = 2*(1 - 0.5*(1+erf(abs(t)/sqrt(2))))
    return md, p


print("="*70)
print("REFEREE 1: WALL CHECK")
lab = load_wide(n=50)
C = lab["C"]
print(f"  max date = {C.index.max().date()}  (must be < {DEV_END})")
assert C.index.max() < pd.Timestamp(DEV_END)
print(f"  assets={len(lab['syms'])}  dates={len(C.index)}  COST={COST}")

print("\n" + "="*70)
print("REFEREE 2: MULTI-SEED RE-DERIVATION of headline agents/fleets")
SEEDS = list(range(10))
NS = 200

# the deployable fleet (fleet_ensemble lane)
FLEET = [
    {"feats": ["mom14"],          "K": 3, "signs": None},
    {"feats": ["mom30", "dev"],   "K": 5, "signs": None},
    {"feats": ["mom30", "vpin"],  "K": 5, "signs": None},
    {"feats": ["rsi14", "mom14"], "K": 5, "signs": [-1.0, 1.0]},
]
# evolve champion
EVOLVE = {"feats": ["brk14", "mom30", "volexp"], "K": 3, "signs": [1.0, 1.0, -1.0]}

cands = {
    "EW": None,
    "mom14_K5": {"feats": ["mom14"], "K": 5},
    "mom14_K3": {"feats": ["mom14"], "K": 3},
    "mom30_dev_K5": {"feats": ["mom30", "dev"], "K": 5},
    "evolve_champ": EVOLVE,
}

rows = {k: [] for k in list(cands) + ["FLEET", "fleet_excess_vs_ew", "fleet_excess_vs_mom14"]}
for sd in SEEDS:
    ds = slice_dates(lab, NS, hold=HOLD, seed=sd)
    ew = ew_on(lab, ds)
    mom14 = agent_on(lab, ["mom14"], ds, K=5)
    for k, cfg in cands.items():
        if k == "EW":
            rows[k].append(np.nanmean(ew)*100); continue
        r = agent_on(lab, cfg["feats"], ds, cfg.get("signs"), K=cfg.get("K", 5))
        rows[k].append(np.nanmean(r)*100)
    fr = fleet_on(lab, FLEET, ds)
    rows["FLEET"].append(np.nanmean(fr)*100)
    m = ~np.isnan(fr) & ~np.isnan(ew)
    rows["fleet_excess_vs_ew"].append(np.nanmean(fr[m]-ew[m])*100)
    m2 = ~np.isnan(fr) & ~np.isnan(mom14)
    rows["fleet_excess_vs_mom14"].append(np.nanmean(fr[m2]-mom14[m2])*100)

print(f"  {'config':24}{'mean%':>9}{'std%':>8}{'min%':>8}{'max%':>8}")
for k, v in rows.items():
    v = np.array(v)
    print(f"  {k:24}{v.mean():>9.2f}{v.std():>8.2f}{v.min():>8.2f}{v.max():>8.2f}")

print("\n" + "="*70)
print("REFEREE 3: 4h_funding -- re-derive the ONLY 'genuinely new' claim")
print("  loading 4h features (causal merge w/ shift(1))...")
data4h = load_4h_wide(n=50)
lab = merge_lab_4h(lab, data4h)
assert "4h_funding" in lab["F"], "4h_funding not merged"
# verify causality: corrupt all funding rows AFTER each decision -> must not change invoke
fund = lab["F"]["4h_funding"].copy()

fund_excess = []; mom14_means = []; combo_means = []; shuf_means = []
for sd in SEEDS:
    ds = slice_dates(lab, NS, hold=HOLD, seed=sd)
    m14 = agent_on(lab, ["mom14"], ds, K=5)
    combo = agent_on(lab, ["mom14", "4h_funding"], ds, K=5)
    mom14_means.append(np.nanmean(m14)*100)
    combo_means.append(np.nanmean(combo)*100)
    # shuffle null: permute funding cross-sectionally per row (within-bar permute columns)
    rng = np.random.default_rng(1000+sd)
    fsh = fund.copy()
    vals = fsh.values
    for i in range(vals.shape[0]):
        row = vals[i]
        perm = rng.permutation(row.size)
        vals[i] = row[perm]
    lab["F"]["4h_funding"] = pd.DataFrame(vals, index=fsh.index, columns=fsh.columns)
    csh = agent_on(lab, ["mom14", "4h_funding"], ds, K=5)
    shuf_means.append(np.nanmean(csh)*100)
    lab["F"]["4h_funding"] = fund  # restore
    fund_excess.append((np.nanmean(combo)-np.nanmean(m14))*100)

combo_means = np.array(combo_means); mom14_means = np.array(mom14_means); shuf_means = np.array(shuf_means)
print(f"  mom14 alone        : {mom14_means.mean():.2f}% (10-seed)")
print(f"  mom14 + 4h_funding : {combo_means.mean():.2f}%")
print(f"  shuffle-null combo : {shuf_means.mean():.2f}%")
print(f"  combo vs mom14     : {(combo_means-mom14_means).mean():+.2f}pp  (paired)")
md, p = paired_t(combo_means, mom14_means)
print(f"    paired-t (combo>mom14) seed-level: d={md:+.2f}pp p={p:.3f}")
md2, p2 = paired_t(combo_means, shuf_means)
print(f"    paired-t (combo>shuffle) seed-level: d={md2:+.2f}pp p={p2:.3f}")

print("\n" + "="*70)
print("REFEREE 4: CAUSALITY -- corrupt all FUTURE feature rows, invoke must be identical")
lab2 = load_wide(n=50)
ds = slice_dates(lab2, 50, hold=HOLD, seed=7)
base = agent_on(lab2, ["mom14"], ds, K=5)
# corrupt every feature row strictly after each di by NaNing the tail
labc = load_wide(n=50)
di_test = ds[10]
for k in labc["F"]:
    v = labc["F"][k].copy()
    v.iloc[di_test+1:] = 999.0
    labc["F"][k] = v
corr = invoke(labc, ["mom14"], di_test, HOLD, 5)
clean = invoke(lab2, ["mom14"], di_test, HOLD, 5)
print(f"  invoke @di={di_test}: clean={clean:.6f}  future-corrupted={corr:.6f}  identical={abs(clean-corr)<1e-12}")

print("\n[referee] DONE -- DEV only, OOS/UNSEEN untouched.")
