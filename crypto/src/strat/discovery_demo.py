"""Discovery-factory DEMONSTRATOR -- the propose -> deflate -> verdict loop, end-to-end on synthetic data.

WHAT THIS IS (and is NOT): a compact, RWYB reference that proves the SOTA discovery methodology from
docs/MARKET_FRAMEWORK/06_STRATEGY_RESEARCH.md section D actually COMPOSES and is HONEST -- a cheap formula
PROPOSER generates a candidate family, and the PBO/CSCV deflator (strat/pbo_cscv.py) decides whether the
SELECTION generalizes. It runs on SYNTHETIC data with a planted (or absent) signal so the two-sided behaviour
is checkable. It is NOT a strategy, NOT the production proposer (the real gp_proposer/llm_proposer are
avenue-specific + fitness-designed to avoid the banned IC objective, D13, and are fork-gated to stage-03), and
it makes NO market claim. It is the scaffold the production proposers will mirror.

THE LOOP IT DEMONSTRATES (the whole point of "how to discover new strategies honestly"):
  1. PROPOSE: a cheap generator searches a formula space over features, scoring IN-SAMPLE only -> top-K candidates,
     tracking the TRUE family_n (every candidate evaluated, not just the survivors -- the #1 anti-pattern).
  2. BUILD FAMILY: materialize the top-K as per-bar strategy return streams -> the (T x K) family matrix.
  3. DEFLATE: pbo_cscv on the family -> PBO = P[the IS-best candidate underperforms OUT-OF-SAMPLE].
  4. VERDICT: planted-signal world -> the search finds the real driver + LOW PBO (generalizes);
              pure-noise world -> the search still "finds" a best backtest, but PBO ~ 0.5 (correctly flagged
              as overfit selection). The deflator is what stops a generator-at-scale from smuggling OOS-losers.

Run:  python -m strat.discovery_demo --selftest   (two-sided: planted signal -> low PBO; noise -> ~0.5)
No emoji (cp1252-safe). Composes strat/pbo_cscv.py.
"""
from __future__ import annotations

import sys
from itertools import combinations

import numpy as np

try:
    from .pbo_cscv import pbo_cscv
except ImportError:  # script / module-path execution
    from strat.pbo_cscv import pbo_cscv

__contract__ = {
    "kind": "discovery_demonstrator",
    "inputs": "synthetic feature panel X (T, M) + forward-return y (T,); k survivors; max formula arity",
    "outputs": "dict{found_features, family_n, top_is_ic, pbo_result(dict), verdict}",
    "invariants": [
        "family_n = the TOTAL number of candidate formulas evaluated (not the survivor count) -- the anti-pattern guard",
        "scoring is IN-SAMPLE only at propose time; generalization is judged ONLY by pbo_cscv on the family",
        "demonstrator on synthetic data -- makes NO market claim; not a strategy, not the production proposer",
        "composes strat/pbo_cscv.py unchanged (evaluator independent of the generator)",
    ],
}


def _ic(sig: np.ndarray, y: np.ndarray) -> float:
    """In-sample rank-free Pearson IC between a signal and forward return (propose-time score only)."""
    s = sig - sig.mean()
    t = y - y.mean()
    denom = np.sqrt((s * s).sum() * (t * t).sum())
    return float((s * t).sum() / denom) if denom > 1e-12 else 0.0


def propose_formulas(X: np.ndarray, y: np.ndarray, k: int = 12, max_arity: int = 2):
    """Cheap generator: search single features + signed pairwise sums/diffs, score by IS-only IC.
    Returns (survivors, family_n). survivors = list of (formula_str, build_fn, is_ic) sorted by |IC| desc."""
    T, M = X.shape
    cands = []  # (formula_str, signal_vector, is_ic)
    # arity-1: each feature (and its negation is implicit via IC sign)
    for j in range(M):
        sig = X[:, j]
        cands.append((f"f{j}", j, None, _ic(sig, y)))
    # arity-2: signed combinations f_a +/- f_b
    if max_arity >= 2:
        for a, b in combinations(range(M), 2):
            for sign, sym in ((1.0, "+"), (-1.0, "-")):
                sig = X[:, a] + sign * X[:, b]
                cands.append((f"f{a}{sym}f{b}", a, (b, sign), _ic(sig, y)))
    family_n = len(cands)                                   # the TRUE trial count (anti-pattern guard)
    cands.sort(key=lambda c: abs(c[3]), reverse=True)
    survivors = cands[:k]
    return survivors, family_n


def family_returns(survivors, X: np.ndarray, fwd: np.ndarray) -> np.ndarray:
    """Materialize each survivor as a per-bar long/short strategy return = sign(IS-IC)*sign(signal)*fwd.
    Returns (T, k). The IC sign is the (in-sample) direction; OOS honesty is then judged by pbo_cscv."""
    T = X.shape[0]
    cols = []
    for (_, a, pair, is_ic) in survivors:
        sig = X[:, a].copy()
        if pair is not None:
            b, sgn = pair
            sig = sig + sgn * X[:, b]
        direction = np.sign(is_ic) if is_ic != 0 else 1.0
        cols.append(direction * np.sign(sig) * fwd)
    return np.column_stack(cols)


def run_discovery_demo(X, y, fwd, k: int = 12, S: int = 16) -> dict:
    survivors, family_n = propose_formulas(X, y, k=k)
    fam = family_returns(survivors, X, fwd)
    pbo = pbo_cscv(fam, S=S)
    return {
        "found_top": survivors[0][0],
        "top_is_ic": round(survivors[0][3], 4),
        "family_n": family_n,
        "k_survivors": len(survivors),
        "pbo": pbo["pbo"],
        "pbo_verdict": pbo["verdict"],
        "prob_oos_loss": pbo["prob_oos_loss"],
    }


def _make_world(rng, T=1500, M=8, planted=True):
    """Synthetic world. X = lagged features; fwd = forward return. If planted, f0 (and a touch of f1)
    truly drive fwd; else fwd is pure noise (the generator will still find a 'best backtest')."""
    X = rng.standard_normal((T, M))
    noise = rng.standard_normal(T)
    if planted:
        fwd = 0.45 * X[:, 0] - 0.30 * X[:, 1] + noise          # f0 - f1 is the true driver
    else:
        fwd = noise                                            # no signal: any 'edge' is selection luck
    return X, fwd


def _selftest() -> None:
    print("=== discovery_demo selftest (propose -> deflate; two-sided) ===")
    rng = np.random.default_rng(11)

    Xp, fp = _make_world(rng, planted=True)
    rp = run_discovery_demo(Xp, fp, fp, k=12, S=16)
    print(f"  PLANTED signal -> found={rp['found_top']:7s}  IS-IC={rp['top_is_ic']:+.3f}  "
          f"family_n={rp['family_n']}  PBO={rp['pbo']:.3f} ({rp['pbo_verdict']})  P(oos loss)={rp['prob_oos_loss']:.3f}")

    Xn, fn = _make_world(rng, planted=False)
    rn = run_discovery_demo(Xn, fn, fn, k=12, S=16)
    print(f"  PURE NOISE     -> found={rn['found_top']:7s}  IS-IC={rn['top_is_ic']:+.3f}  "
          f"family_n={rn['family_n']}  PBO={rn['pbo']:.3f} ({rn['pbo_verdict']})  P(oos loss)={rn['prob_oos_loss']:.3f}")

    # the planted signal's true driver involves f0 / f1; the search should surface one of them
    assert ("f0" in rp["found_top"] or "f1" in rp["found_top"]), f"planted search should find f0/f1, got {rp['found_top']}"
    assert rp["pbo"] < 0.10, f"planted-signal discovery should generalize (low PBO), got {rp['pbo']:.3f}"
    assert rn["pbo"] > 0.30, f"pure-noise discovery must be flagged (PBO ~0.5), got {rn['pbo']:.3f}"
    assert rp["pbo"] < rn["pbo"], "planted PBO must be below noise PBO"
    assert rp["family_n"] == rn["family_n"] and rp["family_n"] > rp["k_survivors"], "family_n must be the TRUE trial count"
    print(f"  family_n={rp['family_n']} candidates evaluated, {rp['k_survivors']} survivors deflated")
    print("  ALL PASS -- planted signal discovered + generalizes; noise 'discovery' caught by the deflator.")


if __name__ == "__main__":
    _selftest()
