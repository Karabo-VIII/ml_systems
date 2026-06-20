"""src/strat/fleet_build2.py -- Fleet v2: forced-diversity selection across info FAMILIES.

Key insight from v1: ALL top-15 candidates are highly correlated (r > 0.75) because they all
ride the same momentum signal. Greedy low-corr selection collapses to 1 agent.

Fix: FAMILY-BASED selection. Define 6 info families:
  A: pure momentum (mom family)
  B: momentum x chimera-dev (structural deviation)
  C: momentum x order-flow (ofi/vpin)
  D: breakout / range
  E: chimera-only (no TI momentum)
  F: sign-flipped / mean-reversion

Pick the best-per-family agent -> a fleet of 6 guaranteed-diverse agents.
Then measure their cross-slice correlation and ensemble performance.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strat.fleet_lab import load_wide, invoke, fleet_invoke, slice_dates, agent_score, COST

RNG_SEED = 42
N_SLICES = 200
HOLD     = 7
K_DEF    = 5
N_ASSETS = 50


def eval_agent(lab, feats, dates, K=K_DEF, signs=None):
    C = lab["C"]
    ew_vals = np.array([
        np.nanmean([C[s].iloc[d+HOLD]/C[s].iloc[d]-1
                    for s in C.columns if pd.notna(C[s].iloc[d]) and pd.notna(C[s].iloc[d+HOLD])])
        for d in dates
    ])
    rois = np.array([
        r if (r := invoke(lab, feats, d, HOLD, K, signs)) is not None else np.nan
        for d in dates
    ])
    v = ~np.isnan(rois)
    if v.sum() < 10:
        return None
    r_v = rois[v]; ew_v = ew_vals[v]
    return {"mean": float(np.mean(r_v)), "profit_rate": float(np.mean(r_v > 0)),
            "beat_ew": float(np.mean(r_v > ew_v)), "rois": rois, "n": int(v.sum())}


def main():
    t0 = time.time()
    print("=" * 70)
    print("FLEET BUILD v2 -- FAMILY-BASED DIVERSE FLEET")
    print("=" * 70)

    lab = load_wide(n=N_ASSETS)
    C = lab["C"]
    dates = slice_dates(lab, n=N_SLICES, hold=HOLD, seed=RNG_SEED)
    print(f"  {len(lab['syms'])} assets  |  {C.index.min().date()} -> {C.index.max().date()}")

    ew = np.array([
        np.nanmean([C[s].iloc[d+HOLD]/C[s].iloc[d]-1
                    for s in C.columns if pd.notna(C[s].iloc[d]) and pd.notna(C[s].iloc[d+HOLD])])
        for d in dates
    ])
    print(f"  EW baseline: mean={100*np.mean(ew):+.2f}%  profit_rate={100*np.mean(ew>0):.0f}%\n")

    # ====================================================================
    # FAMILY GRID: hand-specify diverse info-set candidates
    # Each family captures a distinct slice of signal space.
    # ====================================================================
    families = {
        # -------------------------------------------------------------------
        # A: Pure momentum (varying lookback = real diversity via auto-corr decay)
        # -------------------------------------------------------------------
        "A_mom14":          {"feats": ["mom14"],        "K": K_DEF},
        "A_mom30":          {"feats": ["mom30"],        "K": K_DEF},
        "A_mom7":           {"feats": ["mom7"],         "K": K_DEF},
        "A_mom7K3":         {"feats": ["mom7"],         "K": 3},   # concentrated
        "A_mom14K3":        {"feats": ["mom14"],        "K": 3},
        "A_mom14K8":        {"feats": ["mom14"],        "K": 8},   # broad

        # -------------------------------------------------------------------
        # B: Momentum x structural deviation (chimera dev = price vs realized fair)
        # -------------------------------------------------------------------
        "B_mom30dev":       {"feats": ["mom30", "dev"]},
        "B_mom14dev":       {"feats": ["mom14", "dev"]},
        "B_mom7dev":        {"feats": ["mom7",  "dev"]},
        "B_mom30fdclose":   {"feats": ["mom30", "fdclose"]},
        "B_mom14fdclose":   {"feats": ["mom14", "fdclose"]},

        # -------------------------------------------------------------------
        # C: Momentum x order-flow (ofi / vpin = micro-structure toxicity)
        # -------------------------------------------------------------------
        "C_mom14ofi":       {"feats": ["mom14", "ofi"]},
        "C_mom30ofi":       {"feats": ["mom30", "ofi"]},
        "C_mom30vpin":      {"feats": ["mom30", "vpin"]},
        "C_mom14vpin":      {"feats": ["mom14", "vpin"]},
        "C_mom14ofivpin":   {"feats": ["mom14", "ofi", "vpin"]},
        "C_mom30ofivpin":   {"feats": ["mom30", "ofi", "vpin"]},

        # -------------------------------------------------------------------
        # D: Breakout / range / volatility expansion (NO raw momentum)
        # -------------------------------------------------------------------
        "D_brk14":          {"feats": ["brk14"]},
        "D_brk14volexp":    {"feats": ["brk14", "volexp"]},
        "D_rangepos":       {"feats": ["rangepos"]},
        "D_rangeposofi":    {"feats": ["rangepos", "ofi"]},
        "D_accel":          {"feats": ["accel"]},
        "D_accelbrk14":     {"feats": ["accel", "brk14"]},
        "D_volexpaccel":    {"feats": ["volexp", "accel"]},

        # -------------------------------------------------------------------
        # E: Chimera-only (ZERO TI momentum -- purely structural/microstructure)
        # -------------------------------------------------------------------
        "E_ofi":            {"feats": ["ofi"]},
        "E_vpin":           {"feats": ["vpin"]},
        "E_dev":            {"feats": ["dev"]},
        "E_ofivpin":        {"feats": ["ofi", "vpin"]},
        "E_defdvol":        {"feats": ["dev", "dvol"]},
        "E_ofidev":         {"feats": ["ofi", "dev"]},
        "E_vpinofidev":     {"feats": ["vpin", "ofi", "dev"]},
        "E_dvol":           {"feats": ["dvol"]},
        "E_fdclose":        {"feats": ["fdclose"]},

        # -------------------------------------------------------------------
        # F: Sign-flipped / mean-reversion (overbought-FADE)
        # -------------------------------------------------------------------
        "F_rsiflip":        {"feats": ["rsi14", "mom14"],  "signs": [-1.0, 1.0]},   # sell RSI extremes
        "F_devflip":        {"feats": ["dev", "mom14"],    "signs": [-1.0, 1.0]},   # fade dev overextension
        "F_vpinflip":       {"feats": ["vpin", "mom14"],   "signs": [-1.0, 1.0]},
        "F_rangefade":      {"feats": ["rangepos"],        "signs": [-1.0]},        # pure range fade
        "F_rsionly":        {"feats": ["rsi14"],           "signs": [-1.0]},
        "F_rsifliponly":    {"feats": ["rsi14"]},                                   # rsi momentum (NOT flipped)
    }

    print(f"Evaluating {len(families)} family candidates on {N_SLICES} DEV slices...")
    family_results = {}
    for name, ag in families.items():
        res = eval_agent(lab, ag["feats"], dates, ag.get("K", K_DEF), ag.get("signs"))
        family_results[name] = res
        ag["_res"] = res
        if res:
            flag = "  <<< STRONG" if res["mean"] > 0.025 else ""
            print(f"  {name:22}  mean={100*res['mean']:+6.2f}%  prate={100*res['profit_rate']:.0f}%  beatEW={100*res['beat_ew']:.0f}%{flag}")
        else:
            print(f"  {name:22}  (insufficient data)")

    # ====================================================================
    # PICK BEST-PER-FAMILY
    # ====================================================================
    print("\n=== BEST PER FAMILY ===")
    family_prefix = {"A": [], "B": [], "C": [], "D": [], "E": [], "F": []}
    for name, ag in families.items():
        fam = name[0]
        if fam in family_prefix and ag["_res"] is not None:
            family_prefix[fam].append((name, ag))

    fleet_agents = {}
    for fam, members in family_prefix.items():
        if not members:
            continue
        best = max(members, key=lambda x: x[1]["_res"]["mean"])
        name, ag = best
        fleet_agents[name] = ag
        r = ag["_res"]
        print(f"  Family {fam}: {name:22}  mean={100*r['mean']:+.2f}%  prate={100*r['profit_rate']:.0f}%  beatEW={100*r['beat_ew']:.0f}%")

    # ====================================================================
    # CROSS-CORRELATION OF FLEET
    # ====================================================================
    print(f"\n=== FLEET CROSS-CORRELATION ({len(fleet_agents)} agents) ===")
    roi_mat = {name: ag["_res"]["rois"] for name, ag in fleet_agents.items()}
    names = list(roi_mat.keys())
    mat = pd.DataFrame(np.column_stack(list(roi_mat.values())), columns=names)
    corr = mat.corr()
    print(f"  Pairwise correlations (ideally diverse, < 0.70):")
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j > i:
                c = corr.loc[a, b]
                flag = "<< DIVERSE" if abs(c) < 0.60 else ("moderate" if abs(c) < 0.80 else "")
                print(f"    {a:22} vs {b:22}  r={c:+.3f}  {flag}")

    mean_corr = corr.values[np.triu_indices_from(corr.values, k=1)].mean()
    print(f"\n  Mean pairwise corr = {mean_corr:.3f}")

    # ====================================================================
    # ENSEMBLE COMPARISON: mean / rank-vote / best-single / EW
    # ====================================================================
    print("\n=== ENSEMBLE COMPARISON ===")
    roi_arr = np.column_stack([ag["_res"]["rois"] for ag in fleet_agents.values()])

    # MEAN ensemble
    mean_ens = np.nanmean(roi_arr, axis=1)

    # RANK-VOTE: fire if majority of agents positive
    votes = np.nanmean((roi_arr > 0).astype(float), axis=1)
    vote_roi = np.where(votes >= 0.5, mean_ens, ew)

    # CONVICTION: weight by per-agent historical mean (all DEV, in-sample)
    w_raw = np.array([np.nanmean(ag["_res"]["rois"]) for ag in fleet_agents.values()])
    w = np.maximum(w_raw, 0.0); w = w / (w.sum() + 1e-12)
    conv_ens = np.nansum(roi_arr * w[None,:], axis=1) / \
               np.nansum(~np.isnan(roi_arr) * w[None,:], axis=1)

    # BEST single
    best_name = max(fleet_agents, key=lambda n: fleet_agents[n]["_res"]["mean"])
    best_roi  = fleet_agents[best_name]["_res"]["rois"]

    valid = ~np.isnan(mean_ens) & ~np.isnan(ew)
    for label, arr in [
        ("MEAN-ensemble",    mean_ens),
        ("RANK-VOTE",        vote_roi),
        ("CONV-WEIGHT",      conv_ens),
        (f"BEST-SINGLE({best_name[:16]})", best_roi),
        ("EW-buy-hold",      ew),
    ]:
        v = ~np.isnan(arr)
        r = arr[v]; ew_v = ew[v]
        print(f"  {label:35}  mean={100*np.mean(r):+.2f}%  "
              f"prate={100*np.mean(r>0):.0f}%  "
              f"beatEW={100*np.mean(r>ew_v):.0f}%  "
              f"worst={100*np.min(r):+.2f}%  "
              f"best={100*np.max(r):+.2f}%")

    # ====================================================================
    # DEMO INVOCATIONS ON 4 SPECIFIC DEV DATES
    # ====================================================================
    print("\n=== DEMO FLEET INVOCATIONS (4 specific DEV dates) ===")
    n = len(C.index)
    demo_dis = [
        int(n * 0.15),
        int(n * 0.35),
        int(n * 0.60),
        int(n * 0.85),
    ]
    fleet_list = [{"feats": ag["feats"], "K": ag.get("K", K_DEF), "signs": ag.get("signs")}
                  for ag in fleet_agents.values()]
    for di in demo_dis:
        date = C.index[di].date()
        ew_val = np.nanmean([C[s].iloc[di+HOLD]/C[s].iloc[di]-1
                             for s in C.columns if pd.notna(C[s].iloc[di]) and pd.notna(C[s].iloc[di+HOLD])])
        fleet_roi = fleet_invoke(lab, fleet_list, di, hold=HOLD)
        print(f"\n  DATE {date} (di={di}):  fleet_invoke={100*fleet_roi:+.2f}%  EW={100*ew_val:+.2f}%")
        for name, ag in fleet_agents.items():
            r = invoke(lab, ag["feats"], di, HOLD, ag.get("K", K_DEF), ag.get("signs"))
            sc = agent_score(lab, ag["feats"], di, ag.get("signs"))
            picks = sc.dropna().sort_values(ascending=False).index[:ag.get("K", K_DEF)].tolist()
            roi_str = f"{100*r:+.2f}%" if r is not None else "N/A"
            print(f"    [{name:22}] roi={roi_str}  picks={picks}")

    # ====================================================================
    # AGENT REDUNDANCY: identify which agents ADD vs SUBTRACT
    # ====================================================================
    print("\n=== MARGINAL VALUE: add/drop analysis ===")
    full_mean = float(np.nanmean(mean_ens[valid]))
    full_prate = float(np.nanmean(mean_ens[valid] > 0))
    print(f"  Full fleet ({len(fleet_agents)}):  mean={100*full_mean:+.2f}%  prate={100*full_prate:.0f}%")
    ag_names = list(fleet_agents.keys())
    for drop_name in ag_names:
        keep = [n for n in ag_names if n != drop_name]
        sub = np.nanmean(
            np.column_stack([fleet_agents[n]["_res"]["rois"] for n in keep]), axis=1
        )
        v_sub = ~np.isnan(sub); r_sub = sub[v_sub]
        delta = np.mean(r_sub) - full_mean
        flag = "HELPS" if delta < 0 else "HURTS"  # hurts when dropped -> helps fleet
        print(f"  DROP {drop_name:22}  mean_without={100*np.mean(r_sub):+.2f}%  delta={100*delta:+.2f}pp  -> {flag} fleet")

    # ====================================================================
    # FINAL SUMMARY
    # ====================================================================
    print("\n" + "=" * 70)
    print("DEPLOYABLE FLEET DEFINITION")
    print("=" * 70)
    print(f"\nFleet: {len(fleet_agents)} agents, mean ensemble, K={K_DEF}, hold={HOLD}d")
    print(f"Combine: fleet_invoke(lab, fleet_list, di, hold={HOLD})\n")
    for i, (name, ag) in enumerate(fleet_agents.items()):
        r = ag["_res"]
        print(f"  Agent {i+1} [{name}]")
        print(f"    feats={ag['feats']}  K={ag.get('K',K_DEF)}  signs={ag.get('signs')}")
        print(f"    DEV: mean={100*r['mean']:+.2f}%  profit_rate={100*r['profit_rate']:.0f}%  beat_ew={100*r['beat_ew']:.0f}%")
    print(f"\nfleet_list (copy-paste for fleet_invoke):")
    print(fleet_list)
    print(f"\nElapsed: {time.time()-t0:.1f}s")
    return fleet_agents, fleet_list


if __name__ == "__main__":
    main()
