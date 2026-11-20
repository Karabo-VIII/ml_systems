"""src/strat/fleet_evolve.py -- POPULATION-BASED (evolutionary) search over agent configs on the DEV-walled fleet_lab.

ORC lane (2026-06-20): population-based search over agent configs:
  agent = (feature-subset size 1-4) x (K in {3,5,8}) x (per-feature sign in {+1,-1}).
Pipeline: (0) importance screen single features -> preferred signs;
          (1) seed a diverse population (~40 agents);
          (2) fitness = DEV slice-ROI averaged over seeds  (mean_ROI * profit_rate, beat-EW reported);
          (3) keep top-k, MUTATE (add/drop/flip-sign/change-K) + RECOMBINE (subset crossover) ~6 gens;
          (4) report top-10 evolved agents + fitness trajectory + ensemble fleet DEV slice-ROI.

DATA WALL: DEV (<= 2024-05-15) ONLY, enforced by fleet_lab.load_wide. No OOS/UNSEEN. Long-only spot, taker.
Honest about in-sample / best-of-N: fitness is averaged over multiple seeds (different random DEV slices) to
reduce slice-luck; final report runs a fresh HOLDOUT seed-set never used during evolution to flag overfitting.

RWYB:  python -m strat.fleet_evolve         (full run on real DEV numbers)
No emoji. No git commits.
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
from strat import fleet_lab as FL

FEATS = ["mom7", "mom14", "mom30", "rsi14", "brk14", "rangepos", "volexp", "accel",
         "vpin", "ofi", "dev", "fdclose", "dvol"]
KS = [3, 5, 8]
HOLD = 7
RUNS_DIR = ROOT.parent / "runs" / "strat"


# ---------- evaluation primitives ----------
def _ew_curve(lab, ds, hold=HOLD):
    """Equal-weight buy-hold ROI per slice (the reference)."""
    C = lab["C"]
    out = []
    for d in ds:
        rs = [C[s].iloc[d + hold] / C[s].iloc[d] - 1 for s in C.columns
              if pd.notna(C[s].iloc[d]) and pd.notna(C[s].iloc[d + hold])]
        out.append(float(np.mean(rs)) if rs else np.nan)
    return np.array(out)


def eval_agent(lab, agent, seed_slices, ew_by_seed, hold=HOLD):
    """Average an agent's DEV slice metrics across multiple seed-slice sets.
    Returns dict: mean_roi, profit_rate, beat_ew, fitness."""
    means, prates, beats = [], [], []
    for ds, ew in zip(seed_slices, ew_by_seed):
        rr = [FL.invoke(lab, agent["feats"], d, hold, agent.get("K", 5), agent.get("signs")) for d in ds]
        rr = np.array([x for x in rr if x is not None], dtype=float)
        if len(rr) < 5:
            continue
        means.append(rr.mean())
        prates.append(np.mean(rr > 0))
        m = min(len(rr), len(ew))
        beats.append(np.mean(rr[:m] > ew[:m]))
    if not means:
        return {"mean_roi": -9.9, "profit_rate": 0.0, "beat_ew": 0.0, "fitness": -9.9}
    mean_roi = float(np.mean(means))
    profit_rate = float(np.mean(prates))
    beat_ew = float(np.mean(beats))
    # fitness: reward positive mean scaled by how often it wins; mild penalty if it loses money on average.
    # Use mean_roi * profit_rate but keep sign of mean_roi so a negative-mean agent is correctly penalized.
    fitness = mean_roi * profit_rate if mean_roi > 0 else mean_roi * (1 - profit_rate + 0.5)
    return {"mean_roi": mean_roi, "profit_rate": profit_rate, "beat_ew": beat_ew, "fitness": float(fitness)}


# ---------- importance screen ----------
def importance_screen(lab, seed_slices, ew_by_seed):
    """Score each single feature at sign +1 and -1; pick the better sign. Returns sorted list of (feat, sign, fit, mean)."""
    rows = []
    for f in FEATS:
        best = None
        for sgn in (+1.0, -1.0):
            m = eval_agent(lab, {"feats": [f], "K": 5, "signs": [sgn]}, seed_slices, ew_by_seed)
            if best is None or m["mean_roi"] > best[2]:
                best = (f, sgn, m["mean_roi"], m["profit_rate"], m["beat_ew"], m["fitness"])
        rows.append(best)
    rows.sort(key=lambda r: -r[2])
    return rows


# ---------- population genetics ----------
def _key(a):
    return (tuple(a["feats"]), tuple(a["signs"]), a["K"])


def _canon(a):
    """Sort features (and signs together) so permutations are identical genomes; dedup features."""
    pairs = {}
    for f, s in zip(a["feats"], a["signs"]):
        pairs[f] = s  # last wins if duplicate
    feats = sorted(pairs.keys())
    return {"feats": feats, "signs": [pairs[f] for f in feats], "K": a["K"]}


def seed_population(rng, pref_sign, n=40):
    """Diverse starting population: singles, pairs (TIxTI, TIxchimera), triples, a couple quads."""
    pop = []
    ti = ["mom7", "mom14", "mom30", "rsi14", "brk14", "rangepos", "volexp", "accel"]
    chim = ["vpin", "ofi", "dev", "fdclose", "dvol"]
    def mk(feats, K=5):
        return _canon({"feats": list(feats), "signs": [pref_sign[f] for f in feats], "K": K})
    # all singletons (13)
    for f in FEATS:
        pop.append(mk([f]))
    # diverse pairs
    pairs = [("mom14", "rsi14"), ("mom14", "brk14"), ("mom30", "accel"), ("mom7", "volexp"),
             ("mom14", "vpin"), ("mom14", "ofi"), ("brk14", "rangepos"), ("rsi14", "dev"),
             ("mom30", "ofi"), ("brk14", "vpin"), ("accel", "dvol"), ("mom14", "dev")]
    for p in pairs:
        pop.append(mk(p))
    # triples (TIxchimera mixes + pure TI)
    triples = [("mom14", "vpin", "ofi"), ("mom14", "rsi14", "brk14"), ("mom30", "accel", "ofi"),
               ("brk14", "volexp", "vpin"), ("mom14", "mom30", "rangepos"), ("rsi14", "dev", "fdclose"),
               ("mom7", "mom14", "mom30")]
    for t in triples:
        pop.append(mk(t))
    # a few quads
    quads = [("mom14", "rsi14", "brk14", "vpin"), ("mom7", "mom14", "mom30", "accel"),
             ("mom14", "ofi", "vpin", "dev")]
    for q in quads:
        pop.append(mk(q))
    # vary K on a handful
    extra = []
    for a in pop[:12]:
        for K in (3, 8):
            b = dict(a); b["K"] = K; extra.append(_canon(b))
    pop += extra
    # dedup, cap to n by keeping a spread (shuffle then unique)
    seen, uniq = set(), []
    rng.shuffle(pop)
    for a in pop:
        k = _key(a)
        if k not in seen:
            seen.add(k); uniq.append(a)
    return uniq[:n]


def mutate(rng, a, pref_sign):
    """add/drop/flip-sign a feature, or change K. Keep size in [1,4]."""
    b = _canon(a)
    feats = list(b["feats"]); signs = list(b["signs"]); K = b["K"]
    op = rng.choice(["add", "drop", "flip", "K", "swap"])
    if op == "add" and len(feats) < 4:
        cand = [f for f in FEATS if f not in feats]
        if cand:
            f = rng.choice(cand); feats.append(f); signs.append(pref_sign[f])
    elif op == "drop" and len(feats) > 1:
        i = rng.integers(len(feats)); del feats[i]; del signs[i]
    elif op == "flip":
        i = rng.integers(len(feats)); signs[i] = -signs[i]
    elif op == "K":
        K = int(rng.choice([k for k in KS if k != K]))
    elif op == "swap" and len(feats) >= 1:
        cand = [f for f in FEATS if f not in feats]
        if cand:
            i = rng.integers(len(feats)); f = rng.choice(cand); feats[i] = f; signs[i] = pref_sign[f]
    return _canon({"feats": feats, "signs": signs, "K": K})


def crossover(rng, a, b):
    """Union the two feature sets (keeping each parent's sign), trim to <=4, inherit K from a parent."""
    pairs = {}
    for f, s in zip(a["feats"], a["signs"]):
        pairs[f] = s
    for f, s in zip(b["feats"], b["signs"]):
        if f not in pairs:
            pairs[f] = s
    feats = list(pairs.keys())
    if len(feats) > 4:
        feats = list(rng.choice(feats, 4, replace=False))
    K = a["K"] if rng.random() < 0.5 else b["K"]
    return _canon({"feats": feats, "signs": [pairs[f] for f in feats], "K": K})


# ---------- driver ----------
def run(n_assets=50, n_slices=150, n_seeds=3, gens=6, pop_size=40, keep=12, seed=0):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    print(f"[fleet_evolve] loading DEV u{n_assets} (wall <= {FL.DEV_END}) ...")
    lab = FL.load_wide(n=n_assets)
    C = lab["C"]
    print(f"  {len(lab['syms'])} assets, {C.index.min().date()} -> {C.index.max().date()}")
    assert C.index.max() < pd.Timestamp(FL.DEV_END), "WALL VIOLATION"

    # seed-slice sets used DURING evolution (fixed across all agents so fitness is comparable)
    dev_seeds = [11, 22, 33, 44, 55][:n_seeds]
    seed_slices = [FL.slice_dates(lab, n_slices, HOLD, s) for s in dev_seeds]
    ew_by_seed = [_ew_curve(lab, ds) for ds in seed_slices]
    ew_mean = float(np.nanmean([np.nanmean(e) for e in ew_by_seed]))
    print(f"  DEV reference EW mean ROI/slice = {100*ew_mean:.2f}%  ({n_seeds} seeds x {n_slices} slices)\n")

    # (0) importance screen
    print("[screen] single-feature importance (best sign):")
    scr = importance_screen(lab, seed_slices, ew_by_seed)
    pref_sign = {}
    for f, sgn, mr, pr, be, fit in scr:
        pref_sign[f] = sgn
        print(f"    {f:9} sign{int(sgn):+d}  mean {100*mr:+6.2f}%  profit {100*pr:4.0f}%  beatEW {100*be:4.0f}%")
    print()

    # (1) seed population
    pop = seed_population(rng, pref_sign, pop_size)
    print(f"[evolve] start population = {len(pop)} agents; {gens} generations, keep top {keep}\n")

    traj = []
    cache = {}
    def score(a):
        k = _key(a)
        if k not in cache:
            cache[k] = eval_agent(lab, a, seed_slices, ew_by_seed)
        return cache[k]

    best_overall = None
    for g in range(gens):
        scored = sorted(([a, score(a)] for a in pop), key=lambda x: -x[1]["fitness"])
        elites = scored[:keep]
        top = elites[0]
        if best_overall is None or top[1]["fitness"] > best_overall[1]["fitness"]:
            best_overall = top
        gen_best = top[1]
        gen_mean_fit = float(np.mean([s[1]["fitness"] for s in scored]))
        traj.append({"gen": g, "best_fitness": gen_best["fitness"], "best_mean_roi": gen_best["mean_roi"],
                     "best_profit": gen_best["profit_rate"], "best_beat_ew": gen_best["beat_ew"],
                     "pop_mean_fitness": gen_mean_fit, "best_feats": top[0]["feats"],
                     "best_signs": [int(s) for s in top[0]["signs"]], "best_K": top[0]["K"]})
        print(f"  gen {g}: best fit {gen_best['fitness']:.4f} | mean {100*gen_best['mean_roi']:+.2f}% "
              f"profit {100*gen_best['profit_rate']:.0f}% beatEW {100*gen_best['beat_ew']:.0f}% "
              f"| {'+'.join(f'{f}{int(s):+d}' for f,s in zip(top[0]['feats'],top[0]['signs']))} K{top[0]['K']} "
              f"| pop_mean_fit {gen_mean_fit:.4f}")
        if g == gens - 1:
            break
        # next generation: elites + offspring (mutation + crossover) + a couple random injections
        nxt = [e[0] for e in elites]
        elite_agents = [e[0] for e in elites]
        while len(nxt) < pop_size:
            r = rng.random()
            if r < 0.45:  # mutate an elite
                child = mutate(rng, elite_agents[rng.integers(len(elite_agents))], pref_sign)
            elif r < 0.85:  # crossover two elites
                i, j = rng.integers(len(elite_agents)), rng.integers(len(elite_agents))
                child = crossover(rng, elite_agents[i], elite_agents[j])
            else:  # random fresh agent (diversity injection)
                k = int(rng.integers(1, 5))
                fs = list(rng.choice(FEATS, k, replace=False))
                child = _canon({"feats": fs, "signs": [pref_sign[f] for f in fs], "K": int(rng.choice(KS))})
            nxt.append(child)
        # dedup
        seen, uniq = set(), []
        for a in nxt:
            kk = _key(a)
            if kk not in seen:
                seen.add(kk); uniq.append(a)
        pop = uniq

    # final ranking over the full cache (every agent we ever evaluated)
    all_scored = sorted(({"agent": a, **m} for a, m in [(dict(feats=list(k[0]), signs=list(k[1]), K=k[2]), v)
                                                         for k, v in cache.items()]),
                        key=lambda x: -x["fitness"])
    print(f"\n[evolve] evaluated {len(cache)} distinct agents over {gens} gens\n")
    print("TOP-10 EVOLVED AGENTS (DEV slice-ROI, avg over seeds):")
    print(f"  {'genome':46}{'K':>3}{'mean%':>8}{'profit':>8}{'beatEW':>8}{'fit':>8}")
    top10 = all_scored[:10]
    for r in top10:
        a = r["agent"]
        g = "+".join(f"{f}{int(s):+d}" for f, s in zip(a["feats"], a["signs"]))
        print(f"  {g:46}{a['K']:>3}{100*r['mean_roi']:>7.2f}{100*r['profit_rate']:>7.0f}%{100*r['beat_ew']:>7.0f}%{r['fitness']:>8.4f}")

    # (4) build a FLEET = the top-N diverse agents, ensemble it on DEV evolution-seeds AND a fresh holdout seed-set
    fleet = [r["agent"] for r in top10]
    print("\n[fleet] ensembling top-10 as a fleet (mean of per-agent ROI per slice):")
    # on evolution seeds
    fl_means, fl_prates, fl_beats = [], [], []
    for ds, ew in zip(seed_slices, ew_by_seed):
        fr = [FL.fleet_invoke(lab, fleet, d, HOLD) for d in ds]
        fr = np.array([x for x in fr if x is not None], dtype=float)
        fl_means.append(fr.mean()); fl_prates.append(np.mean(fr > 0))
        m = min(len(fr), len(ew)); fl_beats.append(np.mean(fr[:m] > ew[:m]))
    print(f"  EVOLUTION seeds : mean {100*np.mean(fl_means):+.2f}%  profit {100*np.mean(fl_prates):.0f}%  "
          f"beatEW {100*np.mean(fl_beats):.0f}%   (EW ref {100*ew_mean:+.2f}%)")

    # FRESH HOLDOUT seeds (still DEV, but never seen during evolution) -> overfit check
    hold_seeds = [101, 202, 303]
    hs = [FL.slice_dates(lab, n_slices, HOLD, s) for s in hold_seeds]
    hew = [_ew_curve(lab, ds) for ds in hs]
    hew_mean = float(np.nanmean([np.nanmean(e) for e in hew]))
    # fleet on holdout
    hf_means, hf_prates, hf_beats = [], [], []
    for ds, ew in zip(hs, hew):
        fr = [FL.fleet_invoke(lab, fleet, d, HOLD) for d in ds]
        fr = np.array([x for x in fr if x is not None], dtype=float)
        hf_means.append(fr.mean()); hf_prates.append(np.mean(fr > 0))
        m = min(len(fr), len(ew)); hf_beats.append(np.mean(fr[:m] > ew[:m]))
    print(f"  HOLDOUT  seeds  : mean {100*np.mean(hf_means):+.2f}%  profit {100*np.mean(hf_prates):.0f}%  "
          f"beatEW {100*np.mean(hf_beats):.0f}%   (EW ref {100*hew_mean:+.2f}%)  <- never seen during evolution")

    # also: best SINGLE agent on holdout (overfit-vs-robust contrast)
    best_a = top10[0]["agent"]
    bh_means, bh_prates = [], []
    for ds in hs:
        rr = [FL.invoke(lab, best_a["feats"], d, HOLD, best_a["K"], best_a["signs"]) for d in ds]
        rr = np.array([x for x in rr if x is not None], dtype=float)
        bh_means.append(rr.mean()); bh_prates.append(np.mean(rr > 0))
    print(f"  best-single HOLDOUT: mean {100*np.mean(bh_means):+.2f}%  profit {100*np.mean(bh_prates):.0f}%")

    # persist
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "wall": FL.DEV_END, "n_assets": len(lab["syms"]), "hold": HOLD,
        "config": {"n_slices": n_slices, "n_seeds": n_seeds, "gens": gens, "pop_size": pop_size, "keep": keep},
        "ew_dev_mean": ew_mean, "ew_holdout_mean": hew_mean,
        "importance": [{"feat": f, "sign": int(s), "mean": mr, "profit": pr, "beat_ew": be} for f, s, mr, pr, be, fit in scr],
        "trajectory": traj,
        "top10": [{"feats": r["agent"]["feats"], "signs": [int(x) for x in r["agent"]["signs"]], "K": r["agent"]["K"],
                   "mean_roi": r["mean_roi"], "profit_rate": r["profit_rate"], "beat_ew": r["beat_ew"], "fitness": r["fitness"]}
                  for r in top10],
        "fleet_evolution": {"mean": float(np.mean(fl_means)), "profit": float(np.mean(fl_prates)), "beat_ew": float(np.mean(fl_beats))},
        "fleet_holdout": {"mean": float(np.mean(hf_means)), "profit": float(np.mean(hf_prates)), "beat_ew": float(np.mean(hf_beats))},
        "best_single_holdout": {"mean": float(np.mean(bh_means)), "profit": float(np.mean(bh_prates))},
    }
    ts = time.strftime("%Y%m%d_%H%M%S")
    fp = RUNS_DIR / f"fleet_evolve_{ts}.json"
    fp.write_text(json.dumps(out, indent=2))
    print(f"\n[fleet_evolve] DONE in {time.time()-t0:.0f}s -> {fp}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_assets", type=int, default=50)
    ap.add_argument("--n_slices", type=int, default=150)
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--gens", type=int, default=6)
    ap.add_argument("--pop_size", type=int, default=40)
    ap.add_argument("--keep", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(a.n_assets, a.n_slices, a.n_seeds, a.gens, a.pop_size, a.keep, a.seed)
