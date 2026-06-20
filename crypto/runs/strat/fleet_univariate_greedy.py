"""fleet_univariate_greedy.py -- QUANT LANE: prune 2^13 feature space via univariate screen + greedy-forward.

Stage 1 (univariate): each of 13 features, BOTH signs, single-agent DEV slice-ROI over >=500 PAIRED random
DEV slices. Rank by paired excess over EW buy-hold. Multiple-comparisons aware (26 configs -> report the
best-of-N inflation via a label-permutation null).

Stage 2 (greedy-forward): start from best univariate agent, add the feature (with best sign) that most improves
mean DEV slice-ROI; repeat to size ~5. Honest about in-sample/greedy overfit.

DEV-WALLED: uses strat.fleet_lab.load_wide (hard cap <= 2024-05-15). No OOS touched. RWYB, real numbers.
No emoji (cp1252). Run: .venv python this_file.py
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "src"   # crypto/src
sys.path.insert(0, str(ROOT))
from strat import fleet_lab as FL   # noqa: E402

N_SLICES = 600
HOLD = 7
K = 5
SEED = 7
RNG = np.random.default_rng(SEED)


def ew_returns(lab, ds, hold=HOLD):
    C = lab["C"]
    out = []
    for d in ds:
        vals = [C[s].iloc[d + hold] / C[s].iloc[d] - 1 for s in C.columns
                if d + hold < len(C.index) and pd.notna(C[s].iloc[d]) and pd.notna(C[s].iloc[d + hold])]
        out.append(np.mean(vals) if vals else np.nan)
    return np.array(out)


def agent_returns(lab, feats, ds, signs=None, hold=HOLD, K=K):
    r = np.array([FL.invoke(lab, feats, d, hold, K, signs) for d in ds], dtype=float)
    return r


def block_bootstrap_p(excess, n_boot=5000, block=10, rng=None):
    """One-sided p that mean(excess) <= 0, via circular block bootstrap (returns autocorrelate)."""
    rng = rng or np.random.default_rng(0)
    x = excess[~np.isnan(excess)]
    n = len(x)
    if n < 20:
        return np.nan, np.nan
    obs = x.mean()
    nb = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        means[b] = x[idx][:n].mean()
    # center the bootstrap dist at 0 to test H0: mean<=0
    centered = means - obs
    p = float(np.mean(centered >= obs))   # fraction of null draws as extreme as observed
    p05 = float(np.percentile(means, 5))
    return p, p05


def summarize(name, r, ew):
    """Paired summary vs EW on the SAME slices."""
    m = ~np.isnan(r) & ~np.isnan(ew)
    rr, ee = r[m], ew[m]
    n = len(rr)
    if n < 20:
        return None
    excess = rr - ee
    pboot, p05 = block_bootstrap_p(excess, rng=np.random.default_rng(hash(name) % 2**31))
    return {
        "name": name, "n": n,
        "mean_roi": float(rr.mean()), "profit_rate": float(np.mean(rr > 0)),
        "ew_mean": float(ee.mean()), "beat_ew_rate": float(np.mean(rr > ee)),
        "excess_mean": float(excess.mean()), "excess_p05": p05, "boot_p": pboot,
        "sharpe_slice": float(rr.mean() / (rr.std() + 1e-12)),
    }


def main():
    print(f"=== FLEET UNIVARIATE SCREEN + GREEDY-FORWARD (DEV-walled <= {FL.DEV_END}) ===")
    lab = FL.load_wide(n=50)
    C = lab["C"]
    print(f"loaded {len(lab['syms'])} assets  {C.index.min().date()} -> {C.index.max().date()}")
    assert C.index.max() < pd.Timestamp(FL.DEV_END), "WALL VIOLATION"

    # ONE paired slice set for the whole screen (fixed seed) -> all comparisons paired
    ds = FL.slice_dates(lab, N_SLICES, hold=HOLD, seed=SEED)
    print(f"PAIRED slice set: {len(ds)} random DEV {HOLD}d slices, top-{K}, seed={SEED}")
    ew = ew_returns(lab, ds)
    ew_m = ~np.isnan(ew)
    print(f"EW buy-hold ref: mean={100*np.nanmean(ew):.2f}%  profit={100*np.mean(ew[ew_m]>0):.0f}%\n")

    feats_all = list(lab["F"].keys())

    # ---------- STAGE 1: UNIVARIATE, BOTH SIGNS ----------
    print("--- STAGE 1: univariate (both signs), ranked by paired excess over EW ---")
    print(f"  {'config':18}{'mean%':>8}{'profit':>8}{'beatEW':>8}{'excess%':>9}{'exP05%':>8}{'bootP':>8}")
    rows = []
    for f in feats_all:
        for sgn, tag in [(1.0, "+"), (-1.0, "-")]:
            r = agent_returns(lab, [f], ds, signs=[sgn])
            s = summarize(f"{f}{tag}", r, ew)
            if s:
                s["feat"] = f; s["sign"] = sgn
                s["_r"] = r  # keep for greedy reuse
                rows.append(s)
    rows.sort(key=lambda d: -d["excess_mean"])
    for s in rows:
        print(f"  {s['name']:18}{100*s['mean_roi']:>8.2f}{100*s['profit_rate']:>7.0f}%"
              f"{100*s['beat_ew_rate']:>7.0f}%{100*s['excess_mean']:>9.2f}"
              f"{100*s['excess_p05']:>8.2f}{s['boot_p']:>8.3f}")

    # MULTIPLE-COMPARISONS: best-of-26. expected max excess under a NULL (random K-of-N pick each slice).
    print("\n  [multiple-comparisons] best-of-N null: 26 univariate configs screened.")
    null_excess_max = best_of_n_null(lab, ds, ew, n_configs=len(rows), n_null=200,
                                     rng=np.random.default_rng(123))
    best = rows[0]
    print(f"    observed best excess = {100*best['excess_mean']:.2f}%  ({best['name']})")
    print(f"    null E[max excess] over {len(rows)} random agents = {100*null_excess_max['mean']:.2f}%"
          f"  (95p {100*null_excess_max['p95']:.2f}%)")
    deflated = best["excess_mean"] > null_excess_max["p95"]
    print(f"    -> best univariate {'CLEARS' if deflated else 'INSIDE'} the best-of-N null band"
          f"  => {'REAL signal' if deflated else 'AMBIGUOUS (could be selection)'}")

    # prune: keep features whose BEST sign has positive excess mean AND boot_p < 0.10
    keep = {}
    for s in rows:
        f = s["feat"]
        if s["excess_mean"] > 0 and s["boot_p"] < 0.10:
            if f not in keep or s["excess_mean"] > keep[f]["excess_mean"]:
                keep[f] = s
    pruned = sorted(keep.values(), key=lambda d: -d["excess_mean"])
    print(f"\n  PRUNED keep-list ({len(pruned)} features w/ +excess & bootP<0.10): "
          f"{[ (s['feat'], '+' if s['sign']>0 else '-') for s in pruned]}")

    # ---------- STAGE 2: GREEDY-FORWARD ----------
    print("\n--- STAGE 2: greedy-forward selection (maximize mean DEV slice-ROI) ---")
    # candidate pool = pruned features with their best sign (fall back to all if prune too aggressive)
    pool = [(s["feat"], s["sign"]) for s in pruned] if len(pruned) >= 3 else \
           [(s["feat"], s["sign"]) for s in rows[:8]]
    greedy_feats, greedy_signs = [], []
    history = []
    best_mean = -1e9
    for step in range(5):
        best_add, best_s = None, None
        for (f, sgn) in pool:
            if f in greedy_feats:
                continue
            cand_f = greedy_feats + [f]
            cand_s = greedy_signs + [sgn]
            r = agent_returns(lab, cand_f, ds, signs=cand_s)
            s = summarize("+".join(cand_f), r, ew)
            if s and s["mean_roi"] > (best_s["mean_roi"] if best_s else -1e9):
                best_s, best_add = s, (f, sgn)
        if not best_add or best_s["mean_roi"] <= best_mean + 1e-5:
            print(f"  step {step+1}: no improvement -> STOP")
            break
        greedy_feats.append(best_add[0]); greedy_signs.append(best_add[1])
        best_mean = best_s["mean_roi"]
        history.append(best_s)
        print(f"  step {step+1}: + {best_add[0]}{'+' if best_add[1]>0 else '-':1}  "
              f"-> agent [{'+'.join(greedy_feats)}]  mean={100*best_s['mean_roi']:.2f}%  "
              f"profit={100*best_s['profit_rate']:.0f}%  beatEW={100*best_s['beat_ew_rate']:.0f}%  "
              f"excess={100*best_s['excess_mean']:.2f}%  exP05={100*best_s['excess_p05']:.2f}%  bootP={best_s['boot_p']:.3f}")

    print(f"\n  BEST GREEDY AGENT: feats={greedy_feats} signs={greedy_signs}")
    if history:
        fin = history[-1]
        print(f"    mean_roi={100*fin['mean_roi']:.2f}%  profit_rate={100*fin['profit_rate']:.0f}%  "
              f"beat_ew={100*fin['beat_ew_rate']:.0f}%  excess={100*fin['excess_mean']:.2f}%  "
              f"excess_p05={100*fin['excess_p05']:.2f}%  boot_p={fin['boot_p']:.3f}  sharpe/slice={fin['sharpe_slice']:.3f}")

    # ---------- ROBUSTNESS: re-run univariate top + greedy on 3 INDEPENDENT slice seeds ----------
    print("\n--- ROBUSTNESS: best univariate + greedy agent on 3 INDEPENDENT slice seeds ---")
    top_uni = (rows[0]["feat"], rows[0]["sign"])
    for sd in [101, 202, 303]:
        ds2 = FL.slice_dates(lab, N_SLICES, hold=HOLD, seed=sd)
        ew2 = ew_returns(lab, ds2)
        ru = agent_returns(lab, [top_uni[0]], ds2, signs=[top_uni[1]])
        su = summarize("uni", ru, ew2)
        rg = agent_returns(lab, greedy_feats, ds2, signs=greedy_signs)
        sg = summarize("greedy", rg, ew2)
        print(f"  seed {sd}: uni[{top_uni[0]}] mean={100*su['mean_roi']:.2f}% excess={100*su['excess_mean']:.2f}% bootP={su['boot_p']:.3f}"
              f"  ||  greedy mean={100*sg['mean_roi']:.2f}% excess={100*sg['excess_mean']:.2f}% bootP={sg['boot_p']:.3f}")

    # write artifact
    out = {
        "dev_end": FL.DEV_END, "n_slices": N_SLICES, "hold": HOLD, "K": K, "seed": SEED,
        "n_assets": len(lab["syms"]),
        "ew_mean": float(np.nanmean(ew)),
        "univariate": [{k: v for k, v in s.items() if not k.startswith("_")} for s in rows],
        "best_of_n_null": null_excess_max,
        "deflated_clears": bool(deflated),
        "pruned_keep": [(s["feat"], "+" if s["sign"] > 0 else "-") for s in pruned],
        "greedy_feats": greedy_feats, "greedy_signs": greedy_signs,
        "greedy_history": [{k: v for k, v in s.items() if not k.startswith("_")} for s in history],
    }
    outp = Path(__file__).resolve().parent / "fleet_univariate_greedy_results.json"
    outp.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[written] {outp}")


def best_of_n_null(lab, ds, ew, n_configs, n_null=200, rng=None):
    """Null: an agent that picks K RANDOM assets each slice (no info). Draw n_configs such agents,
    take the MAX paired-excess-mean; repeat n_null times -> distribution of best-of-N excess under no-skill."""
    rng = rng or np.random.default_rng(0)
    C = lab["C"]
    cols = list(C.columns)
    # precompute fwd returns matrix for all slices x assets
    fwd = np.full((len(ds), len(cols)), np.nan)
    for i, d in enumerate(ds):
        for j, s in enumerate(cols):
            if d + HOLD < len(C.index) and pd.notna(C[s].iloc[d]) and pd.notna(C[s].iloc[d + HOLD]):
                fwd[i, j] = C[s].iloc[d + HOLD] / C[s].iloc[d] - 1
    ew_m = ~np.isnan(ew)
    maxes = []
    for _ in range(n_null):
        best = -1e9
        for _c in range(n_configs):
            roi = np.full(len(ds), np.nan)
            for i in range(len(ds)):
                avail = np.where(~np.isnan(fwd[i]))[0]
                if len(avail) >= K:
                    pick = rng.choice(avail, K, replace=False)
                    roi[i] = fwd[i, pick].mean() - FL.COST
            m = ~np.isnan(roi) & ew_m
            ex = (roi[m] - ew[m]).mean()
            best = max(best, ex)
        maxes.append(best)
    maxes = np.array(maxes)
    return {"mean": float(maxes.mean()), "p95": float(np.percentile(maxes, 95)),
            "p99": float(np.percentile(maxes, 99))}


if __name__ == "__main__":
    main()
