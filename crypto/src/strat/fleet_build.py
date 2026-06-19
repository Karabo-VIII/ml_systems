"""src/strat/fleet_build.py -- Intelligent fleet search on DEV data.

PIPELINE:
1. Importance screen: rank all 13 single-feature agents (mean ROI, profit-rate, beat-EW)
2. Greedy-forward selection: build 2-TI, 3-TI combos from top singles
3. Chimera cross: best TIs x chimera features (TI x chimera combos)
4. Sign-aware search: try flipping signs for RSI-type features
5. Agent correlation analysis: pick low-corr diverse set
6. Fleet ensemble: mean / rank-vote / conviction-weight vs best-single vs EW
7. Deployable fleet: define agents + demonstrate on 4 specific DEV dates

RWYB: python -m strat.fleet_build
DEV WALL: all eval strictly < 2024-05-15
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from itertools import combinations

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strat.fleet_lab import load_wide, invoke, fleet_invoke, slice_dates, agent_score, COST

RNG_SEED   = 42
N_SLICES   = 200   # evaluation slices
HOLD       = 7
K_DEFAULT  = 5
N_ASSETS   = 50

# -----------------------------------------------------------------------
# EVAL HELPERS
# -----------------------------------------------------------------------

def eval_agent(lab, feats, dates, K=K_DEFAULT, signs=None):
    """Evaluate one agent on given date indices. Returns (mean_roi, profit_rate, beat_ew_rate, rois)."""
    C = lab["C"]
    ew_vals = []
    for d in dates:
        ew = np.nanmean([C[s].iloc[d+HOLD]/C[s].iloc[d]-1
                         for s in C.columns
                         if pd.notna(C[s].iloc[d]) and pd.notna(C[s].iloc[d+HOLD])])
        ew_vals.append(ew)
    ew_vals = np.array(ew_vals)

    rois = []
    for d in dates:
        r = invoke(lab, feats, d, HOLD, K, signs)
        rois.append(r)
    rois = np.array([r if r is not None else np.nan for r in rois])
    valid = ~np.isnan(rois)
    if valid.sum() < 10:
        return None
    r_v = rois[valid]; ew_v = ew_vals[valid]
    return {
        "mean": float(np.mean(r_v)),
        "profit_rate": float(np.mean(r_v > 0)),
        "beat_ew": float(np.mean(r_v > ew_v)),
        "rois": rois,   # full vector (NaN for missing)
        "n": int(valid.sum()),
    }


def score_tuple(res):
    """Single scalar for ranking: mean_roi (primary), profit_rate secondary."""
    if res is None: return -999.0
    return res["mean"] + 0.1 * res["profit_rate"]


# -----------------------------------------------------------------------
# 1. IMPORTANCE SCREEN -- all 13 single-feature agents
# -----------------------------------------------------------------------

def importance_screen(lab, dates):
    feats_all = list(lab["F"].keys())
    print("\n=== STAGE 1: IMPORTANCE SCREEN (single-feature agents) ===")
    results = {}
    for f in feats_all:
        res = eval_agent(lab, [f], dates)
        results[f] = res
        if res:
            print(f"  {f:12}  mean={100*res['mean']:+6.2f}%  prate={100*res['profit_rate']:.0f}%  beatEW={100*res['beat_ew']:.0f}%")
        else:
            print(f"  {f:12}  (insufficient)")

    # rank by mean roi
    ranked = sorted([f for f in feats_all if results[f]], key=lambda f: -results[f]["mean"])
    print(f"\n  Top features by mean ROI: {ranked[:6]}")
    return results, ranked


# -----------------------------------------------------------------------
# 2. GREEDY FORWARD -- build 2-TI, 3-TI combos from top singles
# -----------------------------------------------------------------------

def greedy_forward(lab, dates, base_feats, max_size=3, top_n=8):
    """Greedy forward selection from top_n singles. Returns best combos of size 2..max_size."""
    print(f"\n=== STAGE 2: GREEDY FORWARD (top-{top_n} features, max_size={max_size}) ===")
    pool = base_feats[:top_n]
    selected = [base_feats[0]]  # start with best single
    best_combos = {}

    for size in range(2, max_size + 1):
        best_score = -999.0
        best_add = None
        for f in pool:
            if f in selected:
                continue
            candidate = selected + [f]
            res = eval_agent(lab, candidate, dates)
            s = score_tuple(res)
            if s > best_score:
                best_score = s
                best_add = f
                best_res = res
        if best_add:
            selected = selected + [best_add]
            best_combos[size] = (list(selected), best_res)
            print(f"  size={size}: {selected}  mean={100*best_res['mean']:+.2f}%  prate={100*best_res['profit_rate']:.0f}%  beatEW={100*best_res['beat_ew']:.0f}%")
    return best_combos


# -----------------------------------------------------------------------
# 3. CHIMERA CROSS -- best TIs x chimera features
# -----------------------------------------------------------------------

def chimera_cross(lab, dates, top_tis, chimera_feats=None):
    if chimera_feats is None:
        chimera_feats = ["vpin", "ofi", "dev", "fdclose", "dvol"]
    print(f"\n=== STAGE 3: CHIMERA CROSS (top-3 TIs x chimera) ===")
    tis = top_tis[:3]
    results = {}
    for ti in tis:
        for ch in chimera_feats:
            feats = [ti, ch]
            res = eval_agent(lab, feats, dates)
            key = f"{ti}+{ch}"
            results[key] = (feats, res)
            if res:
                print(f"  {key:25}  mean={100*res['mean']:+6.2f}%  prate={100*res['profit_rate']:.0f}%  beatEW={100*res['beat_ew']:.0f}%")
    # 3-way: best TI + 2 chimera
    for c1, c2 in combinations(chimera_feats, 2):
        feats = [tis[0], c1, c2]
        res = eval_agent(lab, feats, dates)
        key = f"{tis[0]}+{c1}+{c2}"
        results[key] = (feats, res)
        if res:
            print(f"  {key:25}  mean={100*res['mean']:+6.2f}%  prate={100*res['profit_rate']:.0f}%  beatEW={100*res['beat_ew']:.0f}%")
    return results


# -----------------------------------------------------------------------
# 4. SIGN-AWARE SEARCH -- flip signs for mean-reversion / RSI-style features
# -----------------------------------------------------------------------

def sign_search(lab, dates, candidates):
    """Try flipping signs on features that might be mean-reverting."""
    print(f"\n=== STAGE 4: SIGN-AWARE SEARCH ===")
    sign_feats = ["rsi14", "rangepos", "vpin", "dev"]   # features where reversal might help
    results = {}
    for feats in candidates:
        for sf in sign_feats:
            if sf not in feats:
                continue
            signs = [1.0] * len(feats)
            idx = feats.index(sf)
            signs[idx] = -1.0
            res = eval_agent(lab, feats, dates, signs=signs)
            key = f"FLIP({sf}) in {feats}"
            results[key] = (feats, signs, res)
            if res:
                print(f"  {key:45}  mean={100*res['mean']:+6.2f}%  prate={100*res['profit_rate']:.0f}%")
    return results


# -----------------------------------------------------------------------
# 5. AGENT CORRELATION ANALYSIS
# -----------------------------------------------------------------------

def agent_corr_analysis(lab, dates, agent_pool):
    """Compute per-slice return correlation matrix across all agents."""
    print(f"\n=== STAGE 5: AGENT CORRELATION ANALYSIS ({len(agent_pool)} agents) ===")
    # build ROI matrix: rows=dates, cols=agents
    roi_matrix = {}
    for name, ag in agent_pool.items():
        res = eval_agent(lab, ag["feats"], dates, ag.get("K", K_DEFAULT), ag.get("signs"))
        if res is not None:
            roi_matrix[name] = res["rois"]
            ag["_res"] = res

    names = list(roi_matrix.keys())
    mat = np.column_stack([roi_matrix[n] for n in names])  # shape (n_dates, n_agents)
    # handle NaN columns
    corr = pd.DataFrame(mat, columns=names).corr()

    print(f"  Agent count with valid ROIs: {len(names)}")
    print(f"\n  Pairwise correlation (low = diverse):")
    # print upper triangle
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j > i:
                c = corr.loc[a, b]
                flag = "<< LOW-CORR" if abs(c) < 0.50 else ""
                print(f"    {a:30} vs {b:30}  r={c:+.3f}  {flag}")
    return corr, names, {n: agent_pool[n] for n in names}


# -----------------------------------------------------------------------
# 6. FLEET ENSEMBLE COMPARISON
# -----------------------------------------------------------------------

def fleet_ensemble_compare(lab, dates, fleet_agents):
    """Compare: mean ensemble, rank-vote, conviction-weight, best-single, EW."""
    print(f"\n=== STAGE 6: FLEET ENSEMBLE COMPARISON ===")
    C = lab["C"]
    n_ag = len(fleet_agents)

    # compute per-agent ROI vectors
    ag_rois = {}
    for name, ag in fleet_agents.items():
        rr = [invoke(lab, ag["feats"], d, HOLD, ag.get("K", K_DEFAULT), ag.get("signs")) for d in dates]
        ag_rois[name] = np.array([r if r is not None else np.nan for r in rr])

    # EW reference
    ew = np.array([
        np.nanmean([C[s].iloc[d+HOLD]/C[s].iloc[d]-1
                    for s in C.columns if pd.notna(C[s].iloc[d]) and pd.notna(C[s].iloc[d+HOLD])])
        for d in dates
    ])

    # --- MEAN ensemble ---
    mean_stack = np.nanmean(np.column_stack(list(ag_rois.values())), axis=1)

    # --- RANK-VOTE ensemble: at each slice, pick the date where majority > 0 ---
    vote_stack = np.nanmean(
        (np.column_stack(list(ag_rois.values())) > 0).astype(float), axis=1
    )  # fraction of agents positive (used as signal; final ROI = mean when vote > 0.5, else EW)
    vote_roi = np.where(vote_stack >= 0.5, mean_stack, ew)

    # --- CONVICTION-WEIGHT: weight proportional to per-agent trailing mean (last 30 slices) ---
    # simple: fixed weight by historical mean (in-sample, used to see if weighting helps)
    weights_raw = np.array([np.nanmean(v) for v in ag_rois.values()])
    weights_raw = np.maximum(weights_raw, 0.0)
    if weights_raw.sum() > 0:
        weights = weights_raw / weights_raw.sum()
    else:
        weights = np.ones(n_ag) / n_ag
    conv_stack = np.nansum(
        np.column_stack(list(ag_rois.values())) * weights[None, :], axis=1
    ) / np.nansum(~np.isnan(np.column_stack(list(ag_rois.values()))) * weights[None, :], axis=1)

    # --- Best single (by mean) ---
    best_name = max(ag_rois, key=lambda k: np.nanmean(ag_rois[k]))
    best_single = ag_rois[best_name]

    valid = ~np.isnan(mean_stack) & ~np.isnan(ew)
    results_summary = {}
    for label, arr in [("MEAN-ensemble", mean_stack), ("RANK-VOTE", vote_roi),
                        ("CONV-WEIGHT", conv_stack), (f"BEST-SINGLE({best_name[:14]})", best_single),
                        ("EW-buy-hold", ew)]:
        v = ~np.isnan(arr)
        r = arr[v]
        ew_v = ew[v]
        results_summary[label] = {
            "mean": float(np.mean(r)),
            "profit_rate": float(np.mean(r > 0)),
            "beat_ew": float(np.mean(r > ew_v)),
            "worst_slice": float(np.min(r)),
            "best_slice":  float(np.max(r)),
        }
        print(f"  {label:30}  mean={100*np.mean(r):+6.2f}%  prate={100*np.mean(r>0):.0f}%  "
              f"beatEW={100*np.mean(r>ew_v):.0f}%  worst={100*np.min(r):+.2f}%")
    return results_summary, ag_rois


# -----------------------------------------------------------------------
# 7. DEPLOYABLE FLEET DEFINITION + DEMO SLICES
# -----------------------------------------------------------------------

def demo_slices(lab, fleet_agents, demo_di_list):
    """Demonstrate fleet invocation on specific DEV date indices."""
    C = lab["C"]
    print(f"\n=== STAGE 7: DEPLOYABLE FLEET -- DEMO INVOCATIONS ===")
    fleet_list = [{"feats": ag["feats"], "K": ag.get("K", K_DEFAULT), "signs": ag.get("signs")}
                  for ag in fleet_agents.values()]
    ag_names   = list(fleet_agents.keys())

    for di in demo_di_list:
        date = C.index[di].date()
        ew = np.nanmean([C[s].iloc[di+HOLD]/C[s].iloc[di]-1
                         for s in C.columns if pd.notna(C[s].iloc[di]) and pd.notna(C[s].iloc[di+HOLD])])
        fleet_roi = fleet_invoke(lab, fleet_list, di, hold=HOLD)
        print(f"\n  DATE {date} (di={di}):  fleet_invoke={100*fleet_roi:+.2f}%  EW={100*ew:+.2f}%")
        for name, ag in fleet_agents.items():
            r = invoke(lab, ag["feats"], di, HOLD, ag.get("K", K_DEFAULT), ag.get("signs"))
            if r is not None:
                sc = agent_score(lab, ag["feats"], di, ag.get("signs"))
                picks = sc.dropna().sort_values(ascending=False).index[:ag.get("K", K_DEFAULT)].tolist()
                print(f"    [{name:28}] roi={100*r:+.2f}%  picks={picks}")


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------

def main():
    t0 = time.time()
    print("=" * 70)
    print("FLEET BUILD -- DEV-WALLED INTELLIGENT AGENT SEARCH")
    print("=" * 70)

    print("\nLoading DEV data...")
    lab = load_wide(n=N_ASSETS)
    C = lab["C"]
    print(f"  {len(lab['syms'])} assets  |  {C.index.min().date()} -> {C.index.max().date()}")

    dates = slice_dates(lab, n=N_SLICES, hold=HOLD, seed=RNG_SEED)
    print(f"  Evaluation: {len(dates)} random DEV slices (seed={RNG_SEED}, hold={HOLD}d)")

    # EW baseline
    ew = np.array([
        np.nanmean([C[s].iloc[d+HOLD]/C[s].iloc[d]-1
                    for s in C.columns if pd.notna(C[s].iloc[d]) and pd.notna(C[s].iloc[d+HOLD])])
        for d in dates
    ])
    print(f"  EW baseline: mean={100*np.mean(ew):+.2f}%  profit_rate={100*np.mean(ew>0):.0f}%")

    # ---- Stage 1: importance screen ----
    single_results, ranked_feats = importance_screen(lab, dates)

    # ---- Stage 2: greedy forward from top-8 features ----
    greedy_combos = greedy_forward(lab, dates, ranked_feats, max_size=4, top_n=8)

    # ---- Stage 3: chimera cross ----
    ti_feats = [f for f in ranked_feats if f not in {"vpin","ofi","dev","fdclose","dvol"}]
    chimera_results = chimera_cross(lab, dates, ti_feats)

    # ---- Stage 4: sign search on promising combos ----
    candidate_feats_for_sign = [
        ["rsi14", "mom14"],
        ["rsi14", "mom14", "brk14"],
        ["mom14", "rangepos"],
        ["mom7", "rsi14"],
    ]
    sign_results = sign_search(lab, dates, candidate_feats_for_sign)

    # ---- Assemble diverse candidate pool ----
    print("\n=== ASSEMBLING CANDIDATE POOL ===")
    cand_pool = {}
    # all singles
    for f in lab["F"].keys():
        if single_results[f]:
            cand_pool[f"1TI({f})"] = {"feats": [f]}

    # greedy combos
    for size, (feats, res) in greedy_combos.items():
        cand_pool[f"greedy{size}({'+'.join(feats)})"] = {"feats": feats}

    # top chimera crosses
    chim_sorted = sorted(
        [(k, v) for k, v in chimera_results.items() if v[1] is not None],
        key=lambda x: -score_tuple(x[1][1])
    )
    for k, (feats, res) in chim_sorted[:10]:
        cand_pool[f"xchim({k})"] = {"feats": feats}

    # best sign-flipped
    sign_sorted = sorted(
        [(k, v) for k, v in sign_results.items() if v[2] is not None],
        key=lambda x: -score_tuple(x[1][2])
    )
    for k, (feats, signs, res) in sign_sorted[:5]:
        clean_key = k.replace(" ", "").replace(",","_")[:40]
        cand_pool[f"signed({clean_key})"] = {"feats": feats, "signs": signs}

    # diverse hand-crafted agents (different info families)
    cand_pool["pure_breakout(brk14+volexp)"]  = {"feats": ["brk14", "volexp"]}
    cand_pool["pure_momentum(mom7+accel)"]     = {"feats": ["mom7", "accel"]}
    cand_pool["pure_chimera(ofi+vpin+fdclose)"] = {"feats": ["ofi", "vpin", "fdclose"]}
    cand_pool["chimera_only(dev+dvol)"]         = {"feats": ["dev", "dvol"]}
    cand_pool["full_TI(mom14+rsi14+brk14+volexp+accel)"] = {"feats": ["mom14","rsi14","brk14","volexp","accel"]}
    cand_pool["trend_carry(mom30+fdclose)"]     = {"feats": ["mom30", "fdclose"]}
    cand_pool["flow_trend(ofi+mom14+mom7)"]     = {"feats": ["ofi","mom14","mom7"]}
    cand_pool["range_flow(rangepos+ofi+vpin)"]  = {"feats": ["rangepos","ofi","vpin"]}
    cand_pool["K3_mom14"] = {"feats": ["mom14"], "K": 3}
    cand_pool["K8_mom14"] = {"feats": ["mom14"], "K": 8}

    print(f"  Total candidates: {len(cand_pool)}")
    for n, ag in cand_pool.items():
        res = eval_agent(lab, ag["feats"], dates, ag.get("K", K_DEFAULT), ag.get("signs"))
        if res:
            ag["_mean"] = res["mean"]
            print(f"  {n:45}  mean={100*res['mean']:+6.2f}%  prate={100*res['profit_rate']:.0f}%  beatEW={100*res['beat_ew']:.0f}%")

    # ---- Stage 5: correlation analysis on top candidates ----
    # Select top-15 by mean ROI for correlation study
    top15 = sorted(
        [(n, ag) for n, ag in cand_pool.items() if "_mean" in ag],
        key=lambda x: -x[1]["_mean"]
    )[:15]
    top15_pool = {n: ag for n, ag in top15}
    print(f"\n  Top-15 candidates for correlation analysis:")
    for n, ag in top15_pool.items():
        print(f"    {n:45}  mean={100*ag['_mean']:+.2f}%")

    corr, corr_names, valid_agents = agent_corr_analysis(lab, dates, top15_pool)

    # ---- Select low-corr fleet ----
    print("\n=== FLEET SELECTION (greedy low-corr from top agents) ===")
    # Sort by mean; greedily add if max-corr with already-selected < threshold
    CORR_THRESH = 0.75
    sorted_cands = sorted(valid_agents.items(), key=lambda x: -x[1]["_mean"])

    fleet_agents = {}
    for name, ag in sorted_cands:
        if len(fleet_agents) == 0:
            fleet_agents[name] = ag
            continue
        # max correlation with existing fleet members
        max_c = max(abs(corr.loc[name, existing]) for existing in fleet_agents if name in corr.columns and existing in corr.columns)
        if max_c < CORR_THRESH:
            fleet_agents[name] = ag
        if len(fleet_agents) >= 6:
            break

    print(f"  Selected fleet ({len(fleet_agents)} agents, corr_thresh={CORR_THRESH}):")
    for name, ag in fleet_agents.items():
        print(f"    {name:45}  mean={100*ag['_mean']:+.2f}%")

    # ---- Stage 6: ensemble comparison ----
    ens_results, ag_rois = fleet_ensemble_compare(lab, dates, fleet_agents)

    # ---- Stage 7: demo invocations on 4 specific DEV dates ----
    # pick 4 spread across the DEV window
    C = lab["C"]
    n = len(C.index)
    demo_dis = [
        int(n * 0.15),   # early
        int(n * 0.35),   # mid-early
        int(n * 0.60),   # mid-late
        int(n * 0.85),   # late (still DEV)
    ]
    demo_slices(lab, fleet_agents, demo_dis)

    # ---- FINAL SUMMARY ----
    print("\n" + "=" * 70)
    print("FINAL FLEET SUMMARY")
    print("=" * 70)
    print(f"\nDeployable fleet ({len(fleet_agents)} agents):")
    fleet_list = [{"feats": ag["feats"], "K": ag.get("K", K_DEFAULT), "signs": ag.get("signs")}
                  for ag in fleet_agents.values()]
    for i, (name, ag) in enumerate(fleet_agents.items()):
        print(f"  Agent {i+1}: {name}")
        print(f"    feats={ag['feats']}  K={ag.get('K',K_DEFAULT)}  signs={ag.get('signs')}")
        print(f"    DEV mean ROI={100*ag['_mean']:+.2f}%")

    print(f"\nEnsemble comparison (200 DEV slices):")
    for label, r in ens_results.items():
        print(f"  {label:35}  mean={100*r['mean']:+.2f}%  prate={100*r['profit_rate']:.0f}%  "
              f"beatEW={100*r['beat_ew']:.0f}%  worst={100*r['worst_slice']:+.2f}%")

    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")

    # ---- return artifact ----
    return {
        "fleet": fleet_agents,
        "fleet_list": fleet_list,
        "ensemble_results": ens_results,
        "corr": corr,
    }


if __name__ == "__main__":
    main()
