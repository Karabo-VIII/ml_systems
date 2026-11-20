"""src/strat/fleet_final.py -- Final fleet validation: refined 4-agent fleet vs 6-agent,
correlation structure, robustness (sub-period), and final deployable artifact.

Addresses v2 findings:
- D_accelbrk14 and E_ofidev HURT the ensemble (drop them)
- Test 4-agent fleet: A_mom14K3 + B_mom30dev + C_mom30vpin + F_rsiflip
- Validate on multiple random seeds (fleet stability)
- Report: worst decile, min slice, profit-rate, vs EW
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strat.fleet_lab import load_wide, invoke, fleet_invoke, slice_dates, agent_score, COST, DEV_END

HOLD = 7
K_DEF = 5
N_ASSETS = 50


def eval_fleet(lab, fleet_agents, dates):
    """Return per-slice ROI for each agent and the mean ensemble."""
    C = lab["C"]
    ew = np.array([
        np.nanmean([C[s].iloc[d+HOLD]/C[s].iloc[d]-1
                    for s in C.columns if pd.notna(C[s].iloc[d]) and pd.notna(C[s].iloc[d+HOLD])])
        for d in dates
    ])
    ag_rois = {}
    for name, ag in fleet_agents.items():
        rr = [invoke(lab, ag["feats"], d, HOLD, ag.get("K", K_DEF), ag.get("signs")) for d in dates]
        ag_rois[name] = np.array([r if r is not None else np.nan for r in rr])

    ens = np.nanmean(np.column_stack(list(ag_rois.values())), axis=1)
    return ens, ew, ag_rois


def report(label, arr, ew):
    v = ~np.isnan(arr); r = arr[v]; ew_v = ew[v]
    pct10 = np.percentile(r, 10)
    print(f"  {label:38}  mean={100*np.mean(r):+.2f}%  "
          f"prate={100*np.mean(r>0):.0f}%  "
          f"beatEW={100*np.mean(r>ew_v):.0f}%  "
          f"worst={100*np.min(r):+.2f}%  "
          f"p10={100*pct10:+.2f}%")


def main():
    t0 = time.time()
    print("=" * 70)
    print("FLEET FINAL -- Refined 4-Agent Fleet vs 6-Agent vs Single")
    print("=" * 70)

    lab = load_wide(n=N_ASSETS)
    C = lab["C"]
    print(f"  {len(lab['syms'])} assets  |  {C.index.min().date()} -> {C.index.max().date()}")

    # ====================================================================
    # Define fleets
    # ====================================================================
    fleet_6 = {
        "A_mom14K3":    {"feats": ["mom14"],            "K": 3},
        "B_mom30dev":   {"feats": ["mom30", "dev"],     "K": 5},
        "C_mom30vpin":  {"feats": ["mom30", "vpin"],    "K": 5},
        "D_accelbrk14": {"feats": ["accel", "brk14"],  "K": 5},
        "E_ofidev":     {"feats": ["ofi", "dev"],       "K": 5},
        "F_rsiflip":    {"feats": ["rsi14", "mom14"],   "K": 5, "signs": [-1.0, 1.0]},
    }
    fleet_4 = {  # drop D and E (both hurt ensemble)
        "A_mom14K3":    {"feats": ["mom14"],            "K": 3},
        "B_mom30dev":   {"feats": ["mom30", "dev"],     "K": 5},
        "C_mom30vpin":  {"feats": ["mom30", "vpin"],    "K": 5},
        "F_rsiflip":    {"feats": ["rsi14", "mom14"],   "K": 5, "signs": [-1.0, 1.0]},
    }
    best_single = {"B_mom30dev": {"feats": ["mom30", "dev"], "K": 5}}

    # ====================================================================
    # Multi-seed robustness: eval on 5 random seeds x 200 slices each
    # ====================================================================
    print("\n=== MULTI-SEED ROBUSTNESS (5 seeds x 200 slices each) ===")
    seeds = [42, 7, 13, 99, 314]
    fleet_4_means = []
    fleet_6_means = []
    single_means  = []
    ew_means      = []

    for seed in seeds:
        dates = slice_dates(lab, n=200, hold=HOLD, seed=seed)
        ens4, ew, _ = eval_fleet(lab, fleet_4, dates)
        ens6, _,  _ = eval_fleet(lab, fleet_6, dates)
        sns, _,   _ = eval_fleet(lab, best_single, dates)
        v4 = ~np.isnan(ens4); v6 = ~np.isnan(ens6); vs = ~np.isnan(sns)
        fleet_4_means.append(np.mean(ens4[v4]))
        fleet_6_means.append(np.mean(ens6[v6]))
        single_means.append(np.mean(sns[vs]))
        ew_means.append(np.mean(ew))
        print(f"  seed={seed:3d}:  fleet4={100*np.mean(ens4[v4]):+.2f}%  "
              f"fleet6={100*np.mean(ens6[v6]):+.2f}%  "
              f"single={100*np.mean(sns[vs]):+.2f}%  "
              f"EW={100*np.mean(ew):+.2f}%")

    print(f"\n  AGGREGATE (mean across seeds):")
    print(f"    fleet4-mean  = {100*np.mean(fleet_4_means):+.2f}%  +/- {100*np.std(fleet_4_means):.2f}pp std")
    print(f"    fleet6-mean  = {100*np.mean(fleet_6_means):+.2f}%  +/- {100*np.std(fleet_6_means):.2f}pp std")
    print(f"    single-mean  = {100*np.mean(single_means):+.2f}%  +/- {100*np.std(single_means):.2f}pp std")
    print(f"    EW-mean      = {100*np.mean(ew_means):+.2f}%  +/- {100*np.std(ew_means):.2f}pp std")

    # ====================================================================
    # Deep eval on seed=42 x 400 slices for precise head-to-head
    # ====================================================================
    print("\n=== DEEP EVAL (seed=42, 400 slices) ===")
    dates_deep = slice_dates(lab, n=400, hold=HOLD, seed=42)
    ens4, ew, ag_rois4 = eval_fleet(lab, fleet_4, dates_deep)
    ens6, _,  ag_rois6 = eval_fleet(lab, fleet_6, dates_deep)
    sns,  _,  ag_rois1 = eval_fleet(lab, best_single, dates_deep)

    for label, arr in [
        ("4-agent fleet (MEAN ens)",  ens4),
        ("6-agent fleet (MEAN ens)",  ens6),
        ("Best single (B_mom30dev)",  sns),
        ("EW buy-hold",               ew),
    ]:
        report(label, arr, ew)

    # ====================================================================
    # Correlation structure of the 4-agent fleet
    # ====================================================================
    print("\n=== 4-AGENT FLEET CORRELATION ===")
    corr_mat = pd.DataFrame(
        np.column_stack(list(ag_rois4.values())), columns=list(ag_rois4.keys())
    ).corr()
    for i, a in enumerate(corr_mat.columns):
        for j, b in enumerate(corr_mat.columns):
            if j > i:
                c = corr_mat.loc[a, b]
                print(f"  {a:18} vs {b:18}  r={c:+.3f}")
    mean_c = corr_mat.values[np.triu_indices(len(corr_mat), k=1)].mean()
    print(f"  Mean pairwise corr: {mean_c:.3f}")

    # Does low correlation translate to better worst-slice?
    v4 = ~np.isnan(ens4); v1 = ~np.isnan(sns)
    print(f"\n  Worst 5 slices -- fleet4 vs single:")
    worst5_fleet4  = np.argsort(ens4[v4])[:5]
    worst5_single  = np.argsort(sns[v1])[:5]
    print(f"    fleet4  worst-5 mean: {100*ens4[v4][worst5_fleet4].mean():+.2f}%")
    print(f"    single  worst-5 mean: {100*sns[v1][worst5_single].mean():+.2f}%")

    # ====================================================================
    # SUB-PERIOD ANALYSIS: DEV split into early (2020-2022) and late (2022-2024)
    # ====================================================================
    print("\n=== SUB-PERIOD ANALYSIS ===")
    split_date = pd.Timestamp("2022-01-01")
    all_di = list(range(40, len(C.index) - HOLD - 1))
    early_di = [di for di in all_di if C.index[di] < split_date]
    late_di  = [di for di in all_di if C.index[di] >= split_date]
    rng = np.random.default_rng(42)
    early_sample = sorted(rng.choice(early_di, min(200, len(early_di)), replace=False))
    late_sample  = sorted(rng.choice(late_di,  min(200, len(late_di)),  replace=False))

    for period_name, period_dates in [("Early 2020-2022", early_sample), ("Late 2022-2024", late_sample)]:
        ens4_p, ew_p, _ = eval_fleet(lab, fleet_4, period_dates)
        sns_p,  _,   _ = eval_fleet(lab, best_single, period_dates)
        print(f"\n  {period_name}:")
        report("fleet4", ens4_p, ew_p)
        report("single", sns_p, ew_p)
        report("EW",     ew_p, ew_p)

    # ====================================================================
    # DEMO: 4 specific DEV dates (diverse spread)
    # ====================================================================
    print("\n=== DEMO FLEET INVOCATIONS (4 DEV dates) ===")
    n = len(C.index)
    demo_dis = [int(n * 0.15), int(n * 0.35), int(n * 0.60), int(n * 0.85)]
    fleet_list_4 = [{"feats": ag["feats"], "K": ag.get("K", K_DEF), "signs": ag.get("signs")}
                    for ag in fleet_4.values()]
    for di in demo_dis:
        date = C.index[di].date()
        ew_val = np.nanmean([C[s].iloc[di+HOLD]/C[s].iloc[di]-1
                             for s in C.columns if pd.notna(C[s].iloc[di]) and pd.notna(C[s].iloc[di+HOLD])])
        fleet_roi = fleet_invoke(lab, fleet_list_4, di, hold=HOLD)
        print(f"\n  DATE {date} (di={di}):  fleet_invoke={100*fleet_roi:+.2f}%  EW={100*ew_val:+.2f}%")
        for name, ag in fleet_4.items():
            r = invoke(lab, ag["feats"], di, HOLD, ag.get("K", K_DEF), ag.get("signs"))
            sc = agent_score(lab, ag["feats"], di, ag.get("signs"))
            picks = sc.dropna().sort_values(ascending=False).index[:ag.get("K", K_DEF)].tolist()
            roi_str = f"{100*r:+.2f}%" if r is not None else "N/A"
            print(f"    [{name:16}] roi={roi_str:>8}  picks={picks}")

    # ====================================================================
    # HONEST CAVEATS
    # ====================================================================
    print("\n=== CAVEATS (honest accounting) ===")
    print(f"  1. All eval is DEV-only (< {DEV_END}). OOS >= {DEV_END} untouched.")
    print(f"  2. Family winners are in-sample best-of-N within each family (best-of-{6+5+6+7+9+6}=39 total).")
    print(f"  3. K and hold=7d are fixed (not swept); different values would give different results.")
    print(f"  4. Correlation is measured on the SAME DEV slices used for evaluation (not independent).")
    print(f"  5. Mean pairwise corr = {mean_c:.2f} -- agents still strongly correlated (beta-driven).")
    print(f"  6. fleet_invoke uses TAKER cost {COST} RT; maker/limit fill would be cheaper.")
    print(f"  7. Universe is u50 by DEV-window coverage -- live universe will differ.")

    # ====================================================================
    # FINAL DEPLOYABLE ARTIFACT
    # ====================================================================
    print("\n" + "=" * 70)
    print("DEPLOYABLE FLEET ARTIFACT")
    print("=" * 70)
    print(f"""
FLEET (4 agents, mean ensemble, hold={HOLD}d, taker-cost={COST} RT):

  from strat.fleet_lab import load_wide, fleet_invoke

  lab = load_wide(n=50)          # DEV-walled; swap end= for live

  fleet_list = [
      {{"feats": ["mom14"],           "K": 3, "signs": None}},    # A: pure momentum concentrated
      {{"feats": ["mom30", "dev"],    "K": 5, "signs": None}},    # B: trend x structural deviation
      {{"feats": ["mom30", "vpin"],   "K": 5, "signs": None}},    # C: trend x order-flow toxicity
      {{"feats": ["rsi14", "mom14"], "K": 5, "signs": [-1.0, 1.0]}},  # F: RSI-fade + momentum
  ]

  roi = fleet_invoke(lab, fleet_list, di, hold={HOLD})

DEV PERFORMANCE (400 slices, seed=42):""")

    dates400 = slice_dates(lab, n=400, hold=HOLD, seed=42)
    ens_final, ew_final, _ = eval_fleet(lab, fleet_4, dates400)
    v = ~np.isnan(ens_final)
    r = ens_final[v]; ew_v = ew_final[v]
    print(f"  mean ROI/slice  : {100*np.mean(r):+.2f}%  (EW: {100*np.mean(ew_v):+.2f}%)")
    print(f"  profit rate     : {100*np.mean(r>0):.0f}%  (EW: {100*np.mean(ew_v>0):.0f}%)")
    print(f"  beat EW         : {100*np.mean(r>ew_v):.0f}% of slices")
    print(f"  worst slice     : {100*np.min(r):+.2f}%")
    print(f"  p10 (tail)      : {100*np.percentile(r,10):+.2f}%")
    print(f"  mean pairwise r : {mean_c:.3f}")
    print(f"\nElapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
