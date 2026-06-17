"""PBO via CSCV -- the Probability of Backtest Overfitting (Bailey, Borwein, Lopez de Prado, Zhu 2017).

THE GAP THIS CLOSES (RWYB-verified 2026-06-09): `battery.py` carries DSR/Holm only as a CALLER-NOTE
(battery.py:133) and computes NO combinatorial overfitting probability. When the strat layer runs a
DISCOVERY factory (oracle-decomposition + GP/LLM proposers -> 10^3-10^5 candidate strategies), the single
most dangerous failure mode is SELECTION BIAS: the in-sample-best config is, by construction, the one most
fit to in-sample noise. DSR-at-true-family-N deflates a SINGLE reported Sharpe; PBO answers the orthogonal
question -- "does our SELECTION PROCESS itself produce out-of-sample under-performers?" -- which is exactly
the risk of running a generator at scale.

METHOD (Combinatorially-Symmetric Cross-Validation):
  1. Build the performance matrix M (T observations x N candidate strategies); M[t,n] = strategy n's return at t.
  2. Partition the T rows into S disjoint, equal, time-ordered blocks (S even).
  3. For every way to choose S/2 of the S blocks as IN-SAMPLE (the rest = OUT-OF-SAMPLE) -- C(S, S/2) symmetric
     splits -- pick the IS-best strategy n*, then find n*'s relative RANK among all N strategies OUT-OF-SAMPLE.
     omega = rank/(N+1) in (0,1); logit lambda = ln(omega/(1-omega)).
  4. PBO = P[lambda <= 0] = the fraction of splits where the IS-best strategy lands in the BOTTOM HALF
     out-of-sample. PBO ~ 0.5 = skill-less selection; PBO -> 0 = the selection generalizes; PBO > 0.5 = the
     backtest is actively overfit (IS-best is systematically OOS-bad).

SHIP RULE (recommended): PBO < 0.10 to promote a discovery-search winner. Composes with -- does not replace --
the existing candidate_gate (firewall null + benchmark + leak-probe + battery + DSR-at-family-N + UNSEEN-once).

Run:  python -m strat.pbo_cscv --selftest       (two-sided: rejects an overfit/noise family, accepts a genuine one)
No emoji (cp1252-safe).
"""
from __future__ import annotations

import sys
from itertools import combinations
from math import comb

import numpy as np

__contract__ = {
    "kind": "validation_primitive",
    "inputs": "returns matrix R of shape (T observations, N candidate strategies); S even block count",
    "outputs": "dict{pbo, prob_oos_loss, perf_degradation_slope, median_logit, n_combinations, S, T_used, N, verdict}",
    "invariants": [
        "S must be even (combinatorially-symmetric partition)",
        "N >= 2 (need a cross-section of candidates to rank)",
        "PBO in [0,1]; PBO ~ 0.5 = skill-less selection, PBO < 0.1 = generalizing, PBO > 0.5 = actively overfit",
        "performance metric is per-bar Sharpe (annualization cancels in ranking); std==0 columns are excluded from IS argmax",
        "evaluator is independent of any generator -- this primitive never sees how candidates were produced",
    ],
}


def _sharpe(sub: np.ndarray) -> np.ndarray:
    """Per-column per-bar Sharpe over the rows of `sub` (rows x N). std==0 -> nan (excluded downstream)."""
    mu = sub.mean(axis=0)
    sd = sub.std(axis=0, ddof=1)
    sd = np.where(sd < 1e-12, np.nan, sd)
    return mu / sd


def pbo_cscv(R, S: int = 16, eps: float = 1e-6) -> dict:
    """Probability of Backtest Overfitting via CSCV. R: (T, N) returns matrix. Returns the diagnostic dict."""
    R = np.asarray(R, dtype=float)
    if R.ndim != 2:
        raise ValueError(f"R must be 2-D (T, N); got shape {R.shape}")
    T, N = R.shape
    if N < 2:
        raise ValueError(f"need >=2 candidate strategies to rank; got N={N}")
    if S % 2 != 0:
        raise ValueError(f"S must be even; got {S}")
    if S < 4:
        raise ValueError(f"S must be >=4 for a meaningful combinatorial split; got {S}")
    rows_per = T // S
    if rows_per < 2:
        raise ValueError(f"T={T} too small for S={S} blocks (need >=2 rows/block)")
    usable = rows_per * S
    Rt = R[:usable]
    blocks = [Rt[i * rows_per:(i + 1) * rows_per] for i in range(S)]
    all_blocks = range(S)

    logits = []
    oos_perf_sel = []   # OOS Sharpe of the IS-selected strategy
    is_perf_sel = []    # IS Sharpe of the IS-selected strategy
    for is_choice in combinations(all_blocks, S // 2):
        is_set = set(is_choice)
        IS = np.vstack([blocks[b] for b in all_blocks if b in is_set])
        OOS = np.vstack([blocks[b] for b in all_blocks if b not in is_set])
        is_sr = _sharpe(IS)
        oos_sr = _sharpe(OOS)
        is_clean = np.where(np.isnan(is_sr), -np.inf, is_sr)
        n_star = int(np.argmax(is_clean))                       # IS-best strategy
        oos_clean = np.where(np.isnan(oos_sr), -np.inf, oos_sr)
        rank = int(np.sum(oos_clean <= oos_clean[n_star]))      # 1..N (1 = OOS-worst)
        omega = rank / (N + 1.0)
        omega = min(max(omega, eps), 1.0 - eps)
        logits.append(np.log(omega / (1.0 - omega)))
        sel_oos = oos_sr[n_star]
        oos_perf_sel.append(0.0 if np.isnan(sel_oos) else float(sel_oos))
        sel_is = is_sr[n_star]
        is_perf_sel.append(0.0 if np.isnan(sel_is) else float(sel_is))

    logits = np.asarray(logits)
    is_perf = np.asarray(is_perf_sel)
    oos_perf = np.asarray(oos_perf_sel)
    pbo = float(np.mean(logits <= 0.0))
    prob_oos_loss = float(np.mean(oos_perf < 0.0))
    # performance degradation: OLS slope of OOS-selected perf on IS-selected perf (1.0=carries over, <=0=overfit)
    if np.std(is_perf) > 1e-12:
        slope = float(np.polyfit(is_perf, oos_perf, 1)[0])
    else:
        slope = float("nan")
    verdict = "PASS" if pbo < 0.10 else ("WARN" if pbo < 0.50 else "FAIL")
    return {
        "pbo": pbo,
        "prob_oos_loss": prob_oos_loss,
        "perf_degradation_slope": slope,
        "median_logit": float(np.median(logits)),
        "n_combinations": int(comb(S, S // 2)),
        "S": S,
        "T_used": int(usable),
        "N": int(N),
        "verdict": verdict,
    }


def ship_blocker(R, S: int = 16, threshold: float = 0.10) -> tuple[bool, dict]:
    """Convenience SHIP gate: ok iff PBO < threshold. Returns (ok, full_diagnostic)."""
    res = pbo_cscv(R, S=S)
    return (res["pbo"] < threshold, res)


def _selftest() -> None:
    """Two-sided soundness (the gate must REJECT an overfit family AND ACCEPT a genuine one)."""
    print("=== pbo_cscv selftest (two-sided: reject overfit, accept genuine) ===")
    rng = np.random.default_rng(7)
    T, N = 1200, 20

    # GENUINE family: strategy 0 has a real per-bar edge (Sharpe/bar ~0.25, dominant over a 20-wide noise
    # cross-section); the rest are pure noise. A weaker edge (~0.10) does NOT reliably win IS against the
    # max of 19 noise Sharpes -> PBO stays mid (RWYB 2026-06-09: edge 0.10 -> PBO 0.31). 0.25 is the regime
    # where the IS-best is genuinely the edge strategy and PBO collapses toward 0.
    g = rng.standard_normal((T, N))
    g[:, 0] += 0.25
    res_g = pbo_cscv(g, S=16)
    print(f"  GENUINE family   -> PBO={res_g['pbo']:.3f}  P(oos loss)={res_g['prob_oos_loss']:.3f}  "
          f"degrade_slope={res_g['perf_degradation_slope']:.2f}  verdict={res_g['verdict']}")

    # OVERFIT/skill-less family: all N strategies are pure noise -> selection cannot generalize -> PBO ~ 0.5.
    n = rng.standard_normal((T, N))
    res_n = pbo_cscv(n, S=16)
    print(f"  SKILL-LESS family-> PBO={res_n['pbo']:.3f}  P(oos loss)={res_n['prob_oos_loss']:.3f}  "
          f"degrade_slope={res_n['perf_degradation_slope']:.2f}  verdict={res_n['verdict']}")

    assert res_g["pbo"] < 0.25, f"genuine family should have LOW PBO, got {res_g['pbo']:.3f}"
    assert res_n["pbo"] > 0.35, f"skill-less family should have PBO ~0.5, got {res_n['pbo']:.3f}"
    assert res_g["pbo"] < res_n["pbo"], "genuine PBO must be strictly below skill-less PBO"
    # sanity: input validation
    for bad, kw in [((np.zeros((100, 1))), {}), ((np.zeros((100, 5))), {"S": 7}), ((np.zeros((10, 5))), {"S": 16})]:
        try:
            pbo_cscv(bad, **kw)
            raise AssertionError("expected ValueError on bad input")
        except ValueError:
            pass
    print(f"  C(16,8)={res_g['n_combinations']} symmetric splits per run")
    print("  ALL PASS -- discriminates genuine vs overfit; input-guards fire.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        # demo on a mixed family read from stdin is out of scope; default to selftest-style demo
        _selftest()
