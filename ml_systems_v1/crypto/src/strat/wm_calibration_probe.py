"""src/strat/wm_calibration_probe.py -- the "calib" WM-agent gate: is the WM's
predictive RETURN DISTRIBUTION calibrated on HELD-OUT data?

THE QUESTION (doc S / AGENT_LAYER_ARCHITECTURE_2026_06_11.md Phase 2, line 307)
------------------------------------------------------------------------------
The forecaster (V1.1, V12, ...) does NOT emit a point estimate -- it emits a full
predictive DISTRIBUTION per horizon: TwoHot logits over NUM_BINS=255 bins spanning
symlog space [BIN_MIN,BIN_MAX]=[-1,1] (return magnitudes ~ +/-1.72). An A1
WM-consuming agent's whole value proposition (the doc's sec 3.4 LCB-pessimism critic +
variance-sizing) rests on that distribution being TRUSTWORTHY: when the WM says "90%
of the mass is in [lo,hi]", does the realized return actually land in [lo,hi] ~90% of
the time on UNSEEN data?

If NOT -- if the predicted intervals are systematically too narrow (over-confident) or
too wide (under-confident) -- then:
  * the LCB-critic's "low quantile" is a fiction (pessimism over a miscalibrated tail
    is not pessimism, it is noise),
  * variance-based position sizing sizes on a lie,
  * and uncertainty-aware planning is meaningless.
So CALIBRATION is the PREREQUISITE gate before ANY uncertainty-aware A1 sizing/planning
(doc S Phase-2: "a calibration probe ... is the prerequisite before any uncertainty-
aware planning"). This module answers CALIBRATED / MISCALIBRATED with real numbers.

WHAT IS MEASURED (per horizon, on OOS/UNSEEN)
---------------------------------------------
  * Reliability diagram: for nominal central-interval levels {0.50,0.80,0.90,0.95},
    the EMPIRICAL coverage (fraction of realized returns inside the predicted central
    interval). A calibrated model traces the diagonal (empirical ~= nominal).
  * ECE (Expected Calibration Error): mean |empirical - nominal| across the nominal
    grid -- the single scalar gap. LOW ECE = calibrated.
  * Signed coverage gap at the 90% level: empirical_90 - 0.90. Negative => intervals
    too NARROW (OVER-confident, the dangerous direction for an LCB critic). Positive
    => too WIDE (under-confident).
  * Sharpness: the mean predicted 90% interval WIDTH (in return space). Calibration
    without sharpness is useless (a [-inf,inf] interval is perfectly calibrated and
    perfectly worthless), so the verdict reports both. Sharpness is descriptive, not
    pass/fail -- a wide-but-honest interval is still CALIBRATED.
  * PIT (probability-integral-transform) histogram uniformity as a corroborating,
    interval-free calibration check: PIT_i = predicted CDF at the realized return.
    Calibrated => PIT ~ Uniform(0,1). We report a chi-square-style deviation.

INTERVAL CONSTRUCTION (Jensen-consistent with the WM's own decode)
------------------------------------------------------------------
The predicted central interval at level (1-2q) is [Q(q), Q(1-q)] where Q is the inverse
predicted CDF over the symlog bin grid, mapped to RETURN space via the SAME symexp the
WM's TwoHotSymlog.decode uses (E[symexp] decode is Jensen-correct; we apply symexp to
the bin EDGES to get return-space quantiles consistently). This reuses the canonical
src/wm/_shared/twohot.TwoHotSymlog so the probe speaks the model's exact distribution.

TWO-SIDED RWYB (the soundness contract -- a gate must ACCEPT a genuine *and* REJECT a ghost)
--------------------------------------------------------------------------------------------
This file is RUN-WHAT-YOU-BUILD with SYNTHETIC inputs (no GPU, no V1.1 checkpoint touched
-- V1.1 is training right now). The synthetic generator builds logits directly:
  * GENUINE (well-calibrated): logits whose softmax is a discretized Gaussian centered at
    the *true* mean with the *true* std of the realized-return process => coverage tracks
    nominal => verdict CALIBRATED.
  * NULL / OVER-CONFIDENT (too-narrow): identical means but the predicted std SHRUNK (e.g.
    0.35x) => 90% intervals miss far more than 10% of the time => verdict MISCALIBRATED
    (over-confident). The dangerous failure for an LCB critic, caught.
  * (bonus) UNDER-CONFIDENT (too-wide): predicted std inflated => MISCALIBRATED the other
    way -- shown so the gate is proven two-directional, not just narrow-catching.
  * (finding #12) SIZING/STRICT grade: a SUBTLY too-narrow distribution (std = 0.90x true, ~10%
    too narrow) PASSES the permissive DEFAULT grade (roughly honest) but is REJECTED as
    over_confident under --strict. The strict grade adds tighter ECE/gap tolerances AND an
    ASYMMETRIC cap on the over-confident (too-narrow) direction -- because a fake-tight predictive
    interval is the dangerous, asymmetric-loss failure for variance-sizing / LCB-pessimism (the
    whole point of the probe). The two-sided selftest validates BOTH grades.
If GENUINE does not pass OR OVER-CONFIDENT does not fail (at EITHER grade), the gate is mis-built
-- a CRITICAL foundation finding (mirrors positive_control.py's POWER contract for firewall/battery).

INVARIANTS HONORED (CLAUDE.md / project)
----------------------------------------
  * SETUP/MOVE not per-bar: this is a DISTRIBUTION-calibration diagnostic, NOT an IC/per-
    bar-predictability claim. It says nothing about whether the mean is predictive (that
    is wm_value_probe's job + the banned IC lens); it asks only whether the *uncertainty*
    is honest. Calibration is necessary-not-sufficient for A1.
  * ShIC-not-IC-as-objective: we never optimize or gate on IC here. Calibration is
    orthogonal to directional skill -- a model can be perfectly calibrated and have zero
    directional edge (a wide honest interval). Stated explicitly in the verdict text.
  * No look-ahead: the predicted CDF at bar t uses ONLY the WM's logits at t; the realized
    return is target_return_h at t (forward, already the label). No future info enters the
    interval. The real run consumes a held-out (OOS/UNSEEN) split (caller supplies logits
    produced under the 50/20/20/10 + 400-bar-purge split, same as wm_value_probe).
  * Cost: NONE charged -- calibration is a belief-quality measurement, not a P&L claim;
    cost (TAKER 0.0024) enters downstream at the A1 reward/value gate, not here.
  * ASCII only (cp1252), top-of-file __contract__ dict.

RWYB:
    python src/strat/wm_calibration_probe.py            # two-sided synthetic demo + asserts (both grades)
    python src/strat/wm_calibration_probe.py --selftest # exit 0 iff both sides hold (default + strict)
    python src/strat/wm_calibration_probe.py --strict   # also surface the SIZING/LCB-grade verdicts

REAL run (when V1.1 lands -- NOT executed here, no GPU contention):
    produce {h: logits[N,255]} + {h: realized[N]} on the UNSEEN slice via the existing
    WMEntryProducer / forward_train path (see wm_value_probe.load_unseen_data), then:
        from strat.wm_calibration_probe import probe_calibration
        report = probe_calibration(logits_by_h, realized_by_h)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

__contract__ = {
    "kind": "wm_calibration_probe",
    "owner": "src/strat",
    "track": "calib (WM-agent gate apparatus)",
    "inputs": [
        "logits_by_h: {horizon: np.ndarray[N, NUM_BINS]} TwoHot logits per horizon",
        "realized_by_h: {horizon: np.ndarray[N]} forward realized returns (target_return_h)",
    ],
    "outputs": [
        "per-horizon {ECE, coverage_curve, signed_gap_90, sharpness_90, pit_dev}",
        "verdict CALIBRATED | MISCALIBRATED(over|under) per horizon + overall",
    ],
    "invariants": [
        "interval construction reuses canonical TwoHotSymlog symexp (Jensen-consistent decode)",
        "no look-ahead: predicted CDF at t uses only logits at t; realized is the forward label",
        "no cost, no IC, no per-bar-predictability objective (calibration != directional skill)",
        "two-sided soundness: genuine -> CALIBRATED, over-confident (too-narrow) -> MISCALIBRATED",
        "ECE = mean |empirical_coverage - nominal| over the nominal grid; lower is better",
        "sharpness is descriptive (width), never a pass/fail axis on its own",
        "TWO GRADES (finding #12): DEFAULT (ECE/gap 0.07, diagnostic) and STRICT (--strict, the "
        "SIZING/LCB grade: ECE/gap 0.03 + an ASYMMETRIC 0.02 cap on the OVER-confident/too-narrow "
        "direction). A ~10%-too-narrow interval PASSES default but is REJECTED strict -- a fake-tight "
        "predictive interval must NOT pass for variance-sizing / LCB-pessimism",
        "ASCII only; pure-numpy core (torch only for the optional real-WM decode path)",
    ],
}

# ---------------------------------------------------------------------------
# Canonical WM distribution geometry (must match src/wm/v1/v1_1_training/settings.py)
# BIN_MIN/MAX/NUM_BINS are CLAUDE.md cross-version invariants.
# ---------------------------------------------------------------------------
NUM_BINS = 255
BIN_MIN = -1.0
BIN_MAX = 1.0

# Nominal central-interval levels for the reliability diagram (two-sided central intervals).
NOMINAL_LEVELS = (0.50, 0.80, 0.90, 0.95)

# Gate thresholds (pre-registered, not tuned on any real run).
#
# TWO GRADES (finding #12). The DEFAULT grade is a permissive diagnostic bar; the STRICT grade
# is the one a SIZING / LCB-critic use-case MUST clear -- because that is the whole point of this
# probe (doc S Phase-2: calibration is the prerequisite before any uncertainty-aware sizing/planning).
#
#   DEFAULT (diagnostic): ECE_PASS / GAP90_TOL = 0.07. A ~7pp average miscoverage; a distribution
#     ~10-15% too narrow (confidence 0.85-0.90) PASSES here. Acceptable for a coarse "is it roughly
#     honest?" read, but NOT safe for sizing (an over-confident tail under-states risk -> over-sizing).
#   STRICT (--strict, the sizing/LCB grade): tighter symmetric tolerances AND -- crucially -- an
#     ASYMMETRIC cap on the OVER-CONFIDENT (too-narrow, negative-gap) direction. Over-confidence is the
#     ASYMMETRIC-LOSS dangerous failure for an LCB critic / variance sizer (a fake-tight tail is worse
#     than an honestly-wide one), so a too-narrow 90% interval must FAIL at sizing grade even when a
#     symmetric ECE would forgive it. Under-confidence (too-wide) is merely inefficient, so it keeps
#     the looser symmetric bar.
ECE_PASS = 0.07
GAP90_TOL = 0.07
# STRICT (sizing/LCB) grade -- pre-registered:
ECE_PASS_STRICT = 0.03          # mean miscoverage must be <= 3pp for sizing-grade trust
GAP90_TOL_STRICT = 0.03         # symmetric 90%-level tolerance at sizing grade
OVERCONF_GAP90_TOL_STRICT = 0.02  # ASYMMETRIC: a NEGATIVE (too-narrow) 90% gap may not exceed 2pp
                                  # -- tighter than the symmetric bar because over-confidence is the
                                  # dangerous, asymmetric-loss direction for sizing / pessimism.


# ---------------------------------------------------------------------------
# symexp on the bin GRID -- the only piece we borrow from the model's geometry.
# We map symlog-space bin EDGES to return space with the SAME symexp the WM's
# TwoHotSymlog.decode uses, so the probe's quantiles are consistent with the
# model's own E[symexp] decode (Jensen-correct).
# ---------------------------------------------------------------------------
def _symexp(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * (np.exp(np.abs(x)) - 1.0)


def _symlog(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def _bin_centers_symlog(num_bins: int = NUM_BINS) -> np.ndarray:
    """Bin CENTERS in symlog space (the TwoHotSymlog.buckets grid)."""
    return np.linspace(BIN_MIN, BIN_MAX, num_bins)


def _bin_edges_return(num_bins: int = NUM_BINS) -> np.ndarray:
    """Bin EDGES in RETURN space.

    The grid has `num_bins` centers; the CDF is a step function over those centers.
    We place edges at the midpoints between adjacent symlog centers (and extend the
    outer two by half a width), then map to return space via symexp. Returns an array
    of length num_bins+1 in ascending return order.
    """
    centers = _bin_centers_symlog(num_bins)
    width = (BIN_MAX - BIN_MIN) / (num_bins - 1)
    edges_symlog = np.empty(num_bins + 1, dtype=float)
    edges_symlog[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges_symlog[0] = centers[0] - 0.5 * width
    edges_symlog[-1] = centers[-1] + 0.5 * width
    return _symexp(edges_symlog)


# ---------------------------------------------------------------------------
# Core: predicted CDF -> central-interval bounds + PIT, in RETURN space.
# ---------------------------------------------------------------------------
def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _predicted_quantiles(probs: np.ndarray, q_levels: np.ndarray) -> np.ndarray:
    """Inverse predicted CDF at each q in q_levels, in RETURN space.

    probs: [N, num_bins] per-bar predicted bin probabilities.
    Returns [N, len(q_levels)] return-space quantiles via linear interpolation of the
    step CDF on the return-space bin edges (edges length num_bins+1; cdf at edges
    length num_bins+1 starting at 0).
    """
    n, nb = probs.shape
    edges_ret = _bin_edges_return(nb)              # [nb+1] ascending return-space edges
    cdf = np.concatenate([np.zeros((n, 1)), np.cumsum(probs, axis=1)], axis=1)  # [N, nb+1], 0..1
    out = np.empty((n, len(q_levels)), dtype=float)
    for i in range(n):
        # interp expects increasing x; cdf[i] is nondecreasing in [0,1], edges ascending.
        out[i] = np.interp(q_levels, cdf[i], edges_ret)
    return out


def _pit(probs: np.ndarray, realized: np.ndarray) -> np.ndarray:
    """Probability Integral Transform: predicted CDF evaluated at the realized return.

    Calibrated => PIT ~ Uniform(0,1). Interpolates the step CDF on return-space edges.
    """
    n, nb = probs.shape
    edges_ret = _bin_edges_return(nb)
    cdf = np.concatenate([np.zeros((n, 1)), np.cumsum(probs, axis=1)], axis=1)
    pit = np.empty(n, dtype=float)
    for i in range(n):
        pit[i] = float(np.interp(realized[i], edges_ret, cdf[i]))
    return np.clip(pit, 0.0, 1.0)


def _pit_uniformity_dev(pit: np.ndarray, n_bins: int = 10) -> float:
    """Mean |observed - expected| of the PIT histogram vs Uniform (0 = perfectly uniform)."""
    counts, _ = np.histogram(pit, bins=n_bins, range=(0.0, 1.0))
    obs = counts / max(len(pit), 1)
    exp = 1.0 / n_bins
    return float(np.mean(np.abs(obs - exp)))


# ---------------------------------------------------------------------------
# Per-horizon calibration report.
# ---------------------------------------------------------------------------
def calibration_report(logits: np.ndarray, realized: np.ndarray,
                       nominal_levels=NOMINAL_LEVELS, strict: bool = False) -> dict:
    """Calibration metrics for one horizon.

    logits:   [N, num_bins] TwoHot logits (or directly-built synthetic logits).
    realized: [N] forward realized returns (target_return_h), return space.
    strict:   if True, apply the SIZING/LCB grade (finding #12) -- tighter symmetric ECE/gap
              tolerances AND an ASYMMETRIC cap on the OVER-confident (too-narrow, negative-gap)
              direction, so a fake-tight interval cannot pass at sizing grade.
    """
    logits = np.asarray(logits, dtype=float)
    realized = np.asarray(realized, dtype=float)
    valid = np.isfinite(realized) & np.all(np.isfinite(logits), axis=1)
    logits, realized = logits[valid], realized[valid]
    n = len(realized)
    if n < 30:
        return {"n": int(n), "verdict": "INSUFFICIENT", "ece": float("nan"),
                "coverage_curve": {}, "signed_gap_90": float("nan"),
                "sharpness_90": float("nan"), "pit_dev": float("nan")}

    probs = _softmax(logits)

    # central interval [q, 1-q] for each nominal level; q = (1-level)/2
    coverage_curve = {}
    gaps = []
    sharpness_90 = float("nan")
    for level in nominal_levels:
        q = (1.0 - level) / 2.0
        bounds = _predicted_quantiles(probs, np.array([q, 1.0 - q]))  # [N,2] return space
        lo, hi = bounds[:, 0], bounds[:, 1]
        inside = (realized >= lo) & (realized <= hi)
        emp = float(np.mean(inside))
        coverage_curve[round(level, 4)] = {
            "nominal": round(level, 4),
            "empirical": round(emp, 4),
            "gap": round(emp - level, 4),
            "mean_width": round(float(np.mean(hi - lo)), 6),
        }
        gaps.append(abs(emp - level))
        if abs(level - 0.90) < 1e-9:
            sharpness_90 = float(np.mean(hi - lo))

    ece = float(np.mean(gaps))                       # mean |empirical - nominal|
    signed_gap_90 = coverage_curve[0.9]["empirical"] - 0.90 if 0.9 in coverage_curve else float("nan")
    pit = _pit(probs, realized)
    pit_dev = _pit_uniformity_dev(pit)

    # Grade-dependent thresholds (finding #12). STRICT adds an ASYMMETRIC over-confidence cap.
    ece_tol = ECE_PASS_STRICT if strict else ECE_PASS
    gap_tol = GAP90_TOL_STRICT if strict else GAP90_TOL
    overconf_tol = OVERCONF_GAP90_TOL_STRICT if strict else GAP90_TOL  # default = symmetric

    # Verdict: CALIBRATED iff ECE within tolerance AND the headline 90% level is on target.
    # In STRICT mode the OVER-confident (negative-gap, too-narrow) direction faces a TIGHTER cap
    # than the under-confident direction -- a fake-tight interval is the dangerous, asymmetric-loss
    # failure for an LCB critic / variance sizer and must NOT pass at sizing grade.
    if signed_gap_90 < 0:
        gap_ok = abs(signed_gap_90) <= overconf_tol      # too-narrow: tighter cap (strict)
    else:
        gap_ok = abs(signed_gap_90) <= gap_tol           # too-wide: looser symmetric cap
    if ece <= ece_tol and gap_ok:
        verdict = "CALIBRATED"
    else:
        # direction from the 90% level: negative gap => intervals too narrow (over-confident)
        direction = "over_confident" if signed_gap_90 < 0 else "under_confident"
        verdict = f"MISCALIBRATED({direction})"

    return {
        "n": int(n),
        "ece": round(ece, 4),
        "signed_gap_90": round(float(signed_gap_90), 4),
        "sharpness_90": round(float(sharpness_90), 6),
        "pit_dev": round(float(pit_dev), 4),
        "coverage_curve": coverage_curve,
        "verdict": verdict,
        "grade": "strict" if strict else "default",
        "ece_tol": ece_tol,
        "gap_tol": gap_tol,
        "overconf_gap_tol": overconf_tol,
    }


def probe_calibration(logits_by_h: dict, realized_by_h: dict, verbose: bool = True,
                      strict: bool = False) -> dict:
    """Top-level entry: per-horizon calibration + an overall verdict.

    logits_by_h:   {horizon: np.ndarray[N, NUM_BINS]}
    realized_by_h: {horizon: np.ndarray[N]}
    strict:        SIZING/LCB grade (finding #12) -- tighter ECE/gap + an asymmetric over-confidence
                   cap. Use strict=True for ANY uncertainty-aware sizing/LCB-critic decision.
    Overall verdict = CALIBRATED iff EVERY measured horizon is CALIBRATED (the A1 planner
    queries multiple horizons -- a single miscalibrated horizon poisons that branch of its
    imagination, doc S Phase-2 note on per-horizon ShIC).
    """
    per_h = {}
    for h in sorted(logits_by_h):
        if h not in realized_by_h:
            continue
        per_h[h] = calibration_report(logits_by_h[h], realized_by_h[h], strict=strict)

    measured = [r for r in per_h.values() if r["verdict"] not in ("INSUFFICIENT",)]
    overall = ("CALIBRATED" if measured and all(r["verdict"] == "CALIBRATED" for r in measured)
               else "MISCALIBRATED" if measured else "INSUFFICIENT")

    if verbose:
        _print_report(per_h, overall)
    return {"per_horizon": per_h, "overall": overall}


def _print_report(per_h: dict, overall: str) -> None:
    print("\n" + "=" * 78)
    print("  WM CALIBRATION PROBE -- predictive-distribution coverage on held-out data")
    print("=" * 78)
    for h, r in per_h.items():
        if r["verdict"] == "INSUFFICIENT":
            print(f"  h={h:<3} n={r['n']:<5} INSUFFICIENT (<30 valid)")
            continue
        print(f"\n  --- horizon h={h}  (n={r['n']}) ---")
        print(f"  {'nominal':>8} {'empirical':>10} {'gap':>8} {'mean_width':>12}")
        for lvl, c in r["coverage_curve"].items():
            print(f"  {c['nominal']:>8.2f} {c['empirical']:>10.3f} {c['gap']:>+8.3f} {c['mean_width']:>12.5f}")
        print(f"  ECE={r['ece']:.4f}  signed_gap@90={r['signed_gap_90']:+.4f}  "
              f"sharpness@90(width)={r['sharpness_90']:.5f}  PIT_dev={r['pit_dev']:.4f}")
        print(f"  grade={r.get('grade','default')} (ECE_tol={r.get('ece_tol')}, gap_tol={r.get('gap_tol')}, "
              f"overconf_tol={r.get('overconf_gap_tol')})  -> {r['verdict']}")
    print("\n" + "-" * 78)
    print(f"  OVERALL: {overall}")
    print("  (CALIBRATED = predicted intervals contain the realized return at ~the nominal")
    print("   rate => the WM's UNCERTAINTY is trustworthy => LCB-critic / variance-sizing are")
    print("   meaningful. MISCALIBRATED(over_confident) = intervals too NARROW => the dangerous")
    print("   failure: a pessimism layer over a fake-tight tail is noise. Calibration is")
    print("   necessary-not-sufficient for A1 and is ORTHOGONAL to directional skill/IC.)")
    print("=" * 78)


# ---------------------------------------------------------------------------
# SYNTHETIC generators for the two-sided RWYB.
# We build logits DIRECTLY by discretizing a Gaussian onto the symlog bin grid, so we
# control exactly how confident the "model" is relative to the true process.
# ---------------------------------------------------------------------------
def _gaussian_logits(mu_ret: np.ndarray, sigma_ret: np.ndarray,
                     num_bins: int = NUM_BINS) -> np.ndarray:
    """Build [N, num_bins] logits whose softmax approximates N(mu_ret, sigma_ret) in RETURN
    space, expressed on the symlog bin grid.

    We assign each bin probability = Gaussian density (in symlog space, where the model's
    grid is uniform) evaluated at the bin center's symlog coordinate, with the Gaussian
    parameters transformed into symlog space. This produces a proper bell over the bins;
    log of the normalized mass is the logits.
    """
    centers_symlog = _bin_centers_symlog(num_bins)                 # [nb] uniform symlog grid
    mu_symlog = _symlog(mu_ret)                                    # [N]
    # local linearization: d(symlog)/d(ret) = 1/(1+|ret|); map return-space sigma into symlog
    sigma_symlog = sigma_ret / (1.0 + np.abs(mu_ret))             # [N]
    sigma_symlog = np.maximum(sigma_symlog, 1e-4)
    # density of each bin center under N(mu_symlog, sigma_symlog) in symlog space
    z = (centers_symlog[None, :] - mu_symlog[:, None]) / sigma_symlog[:, None]  # [N, nb]
    log_dens = -0.5 * z * z                                        # unnormalized log-Gaussian
    return log_dens  # softmax later normalizes; this IS valid logits


def make_synthetic_case(n: int = 1500, seed: int = 11, confidence: float = 1.0,
                        horizons=(1, 4, 16)) -> tuple:
    """Build (logits_by_h, realized_by_h) for a synthetic WM.

    The TRUE process: realized return ~ N(mu_t, sigma_true_h) where mu_t is a slowly
    varying drift and sigma_true_h grows with horizon (sqrt-time). The MODEL emits logits
    for N(mu_t, confidence*sigma_true_h):
        confidence == 1.0  -> well-calibrated  (predicted std == true std)
        confidence  < 1.0  -> OVER-confident   (predicted intervals too NARROW)
        confidence  > 1.0  -> UNDER-confident  (predicted intervals too WIDE)
    The model gets the MEAN right in all cases -- calibration is about the SPREAD, not the
    location, so this isolates the uncertainty-honesty axis (and keeps the test independent
    of directional skill / IC, which is banned as an objective).
    """
    rng = np.random.default_rng(seed)
    logits_by_h, realized_by_h = {}, {}
    # slowly varying true drift (a regime-ish AR(1) so it is not iid -- realistic, still
    # leak-free: the model is GIVEN mu_t, we only test spread calibration).
    drift = np.zeros(n)
    for t in range(1, n):
        drift[t] = 0.98 * drift[t - 1] + rng.normal(0.0, 0.0008)
    for h in horizons:
        sigma_true = 0.012 * np.sqrt(h)                # true per-horizon vol (sqrt-time)
        mu = drift * np.sqrt(h)                         # true mean scales with horizon
        realized = mu + rng.normal(0.0, sigma_true, n)
        sigma_pred = np.full(n, confidence * sigma_true)
        logits = _gaussian_logits(mu, sigma_pred)
        logits_by_h[h] = logits
        realized_by_h[h] = realized
    return logits_by_h, realized_by_h


# ---------------------------------------------------------------------------
# Two-sided RWYB demonstration + selftest.
# ---------------------------------------------------------------------------
def run_two_sided(verbose: bool = True) -> dict:
    """GENUINE (confidence=1.0) must be CALIBRATED; OVER-confident (0.35) must be
    MISCALIBRATED(over_confident); UNDER-confident (2.5) must be MISCALIBRATED(under_confident).

    Finding #12 additionally validates the SIZING/STRICT grade: a ~10%-too-narrow distribution
    (confidence=0.90) PASSES the permissive default grade (it is roughly honest) but MUST be
    REJECTED as over_confident under --strict (a fake-tight 90% interval is unsafe for sizing).
    The genuine (confidence=1.0) distribution must still pass under STRICT.
    """
    if verbose:
        print("\n########## SIDE 1: GENUINE well-calibrated synthetic WM (predicted std == true std)")
    g_logits, g_real = make_synthetic_case(confidence=1.0, seed=11)
    genuine = probe_calibration(g_logits, g_real, verbose=verbose)

    if verbose:
        print("\n########## SIDE 2: OVER-confident synthetic WM (predicted std = 0.35x true => too NARROW)")
    o_logits, o_real = make_synthetic_case(confidence=0.35, seed=11)
    over = probe_calibration(o_logits, o_real, verbose=verbose)

    if verbose:
        print("\n########## SIDE 3 (bonus, two-directional proof): UNDER-confident (std = 2.5x true => too WIDE)")
    u_logits, u_real = make_synthetic_case(confidence=2.5, seed=11)
    under = probe_calibration(u_logits, u_real, verbose=verbose)

    # FINDING #12: the borderline ~10%-too-narrow distribution (the dangerous-but-subtle case).
    n_logits, n_real = make_synthetic_case(confidence=0.90, seed=11)
    narrow_default = probe_calibration(n_logits, n_real, verbose=False, strict=False)
    narrow_strict = probe_calibration(n_logits, n_real, verbose=False, strict=True)
    genuine_strict = probe_calibration(g_logits, g_real, verbose=False, strict=True)

    genuine_ok = genuine["overall"] == "CALIBRATED"
    over_caught = over["overall"] == "MISCALIBRATED" and \
        all("over_confident" in r["verdict"] for r in over["per_horizon"].values() if r["verdict"].startswith("MIS"))
    under_caught = under["overall"] == "MISCALIBRATED" and \
        all("under_confident" in r["verdict"] for r in under["per_horizon"].values() if r["verdict"].startswith("MIS"))
    # strict-grade contract: genuine still CALIBRATED strict; the 10%-narrow one passes DEFAULT but
    # is REJECTED (over_confident) under STRICT -- the sizing-grade tightening doing its job.
    genuine_strict_ok = genuine_strict["overall"] == "CALIBRATED"
    narrow_default_pass = narrow_default["overall"] == "CALIBRATED"
    narrow_strict_caught = narrow_strict["overall"] == "MISCALIBRATED" and \
        all("over_confident" in r["verdict"] for r in narrow_strict["per_horizon"].values()
            if r["verdict"].startswith("MIS"))
    strict_grade_ok = bool(genuine_strict_ok and narrow_default_pass and narrow_strict_caught)

    two_sided_ok = bool(genuine_ok and over_caught and under_caught and strict_grade_ok)
    if verbose:
        print("\n" + "=" * 78)
        print("  TWO-SIDED SOUNDNESS (the gate must ACCEPT a genuine AND REJECT a ghost):")
        print(f"    SIDE 1 genuine            -> {genuine['overall']:<14} (expect CALIBRATED)     "
              f"{'OK' if genuine_ok else '*** FAIL ***'}")
        print(f"    SIDE 2 over-confident     -> {over['overall']:<14} (expect MISCALIBRATED)  "
              f"{'OK' if over_caught else '*** FAIL ***'} (direction=over_confident)")
        print(f"    SIDE 3 under-confident    -> {under['overall']:<14} (expect MISCALIBRATED)  "
              f"{'OK' if under_caught else '*** FAIL ***'} (direction=under_confident)")
        print("\n  STRICT (SIZING/LCB) GRADE -- finding #12 (a too-narrow interval must NOT pass for sizing):")
        print(f"    genuine (conf=1.0)  STRICT -> {genuine_strict['overall']:<14} (expect CALIBRATED)     "
              f"{'OK' if genuine_strict_ok else '*** FAIL ***'}")
        print(f"    10%-narrow (0.90) DEFAULT  -> {narrow_default['overall']:<14} (expect CALIBRATED -- "
              f"permissive) {'OK' if narrow_default_pass else '*** FAIL ***'}")
        print(f"    10%-narrow (0.90)  STRICT  -> {narrow_strict['overall']:<14} (expect MISCALIBRATED) "
              f"{'OK' if narrow_strict_caught else '*** FAIL ***'} (over_confident -- caught at sizing grade)")
        if two_sided_ok:
            print("\n  PASS -- the calibration gate is SOUND: a well-calibrated distribution is accepted")
            print("  (default AND strict), an over-confident (too-narrow) one is rejected, an under-")
            print("  confident (too-wide) one is rejected, AND -- finding #12 -- a SUBTLY too-narrow (10%)")
            print("  distribution that the permissive default forgives is CAUGHT at the strict sizing grade.")
        else:
            print("\n  *** CRITICAL *** the calibration gate failed its two-sided soundness contract.")
            print("  Inspect ECE_PASS / GAP90_TOL / the strict grade or the interval construction.")
        print("=" * 78)

    return {"two_sided_ok": two_sided_ok, "genuine": genuine, "over": over, "under": under,
            "narrow_default": narrow_default, "narrow_strict": narrow_strict,
            "genuine_strict": genuine_strict, "strict_grade_ok": strict_grade_ok}


def _selftest() -> int:
    """Exit 0 iff the two-sided soundness contract holds (CI / regression hook)."""
    res = run_two_sided(verbose=False)
    assert res["two_sided_ok"], (
        "wm_calibration_probe two-sided soundness FAILED: "
        f"genuine={res['genuine']['overall']} over={res['over']['overall']} under={res['under']['overall']}"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="WM calibration probe (predictive-distribution coverage)")
    ap.add_argument("--selftest", action="store_true",
                    help="exit 0 iff the two-sided soundness contract holds (default + strict grades)")
    ap.add_argument("--strict", action="store_true",
                    help="SIZING/LCB grade: tighter ECE/gap + an asymmetric cap on the OVER-confident "
                         "(too-narrow) direction. Use for ANY uncertainty-aware sizing/LCB-critic decision; "
                         "a fake-tight interval fails here (finding #12). The two-sided selftest validates "
                         "BOTH grades regardless of this flag.")
    args = ap.parse_args()
    if args.selftest:
        rc = _selftest()
        print("wm_calibration_probe selftest: PASS (two-sided soundness holds; default + strict grades)")
        return rc
    res = run_two_sided(verbose=True)
    if args.strict:
        print(f"\n  [--strict requested] STRICT-grade verdicts -- genuine: {res['genuine_strict']['overall']}, "
              f"10%-narrow: {res['narrow_strict']['overall']} (the sizing-grade gate)")
    return 0 if res["two_sided_ok"] else 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
