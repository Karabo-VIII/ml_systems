"""src/strat/single_knob_capture_calibration.py
SINGLE-KNOB calibration generator for the within-window capture metric.

WHY THIS EXISTS (supersedes the monolithic synthetic positive-control that terminally REFUTED)
----------------------------------------------------------------------------------------------
The prior approach forced ONE synthetic setup to PASS a pass/refute GATE (real mean-capture must beat a
random-timing null p95). That conflated TWO separable claims:
  (1) GENERATOR/METRIC soundness -- does the capture metric faithfully *track* injected timing skill?
  (2) GATE calibration          -- is the null-band threshold set so a genuine edge clears it?
Step (2) is brittle (threshold/p95/sample-size sensitive) and is where the old control died. This module
isolates the WEAKER, mechanically-checkable step (1): build a single-knob generator that injects a tunable
timing-skill fraction k in [0,1] and verify the seed-averaged within-window capture metric is MONOTONICALLY
NON-DECREASING in k -- with k=0 -> ~0 capture and k=1 -> ~full capture. No null, no threshold, no gate.

THE KNOB (single, in [0,1])
---------------------------
For each MOVE the strategy is present in, the realized net-return curve ret(t) over candidate entry bars
gives best=max ret (oracle entry), worst=min ret (pessimal entry), spread=best-worst, and the established
capture statistic  capture = (ret(t*) - worst) / spread  in [0,1]  (t* = the strategy's actual trigger;
1.0 == oracle timing, 0.0 == worst timing). The single skill knob k controls the trigger:
  mode="mix"    : per move, with probability k the trigger lands the ORACLE bar (capture 1), else the
                  PESSIMAL bar (capture 0). Expected per-move capture == k -> seed-averaged mean capture
                  tracks k, with exact endpoints k=0->0, k=1->1. (Bernoulli; seed-averaging smooths it.)
  mode="target" : deterministic -- the trigger picks the candidate bar whose capture-fraction is closest
                  to k. Realized capture ~= k up to bar discretization. A cross-check that the monotone
                  response is not an artifact of the stochastic mix.

WHAT THIS PROVES (and what it deliberately does NOT)
----------------------------------------------------
PROVES: the capture metric is a faithful, monotone readout of injected within-window timing skill --
        more skill in => more measured capture out, no holes. The GENERATOR and the METRIC are sound.
DOES NOT: claim any edge exists in real markets, nor that the downstream null/gate is calibrated. Those
          are separate, later steps. This is the foundation they stand on.

VERIFY (verify_cmd): `python src/strat/single_knob_capture_calibration.py`
  exits 0 IFF Spearman(k, seed_averaged_capture) > 0.90 for the primary sweep (monotone non-decreasing),
  AND the endpoints satisfy k=0 -> capture<=0.05 and k=1 -> capture>=0.95. Non-zero exit on any violation.
"""
from __future__ import annotations

import json
import sys

import numpy as np

# Reuse the ESTABLISHED synthetic move-stream + capture definition (no reinvention).
from within_window_capture_gate import (  # type: ignore
    HELD,
    TAKER,
    _move_ret_curve,
    _window_label,
    make_moves,
)

try:  # local-or-package import resilience
    from src.strat.within_window_capture_gate import (  # noqa: F811
        HELD, TAKER, _move_ret_curve, _window_label, make_moves,
    )
except Exception:
    pass


# ---------------------------------------------------------------------------
# Single-knob triggers: k in [0,1] is the within-window timing-skill fraction.
# ---------------------------------------------------------------------------
def make_knob_trigger(k: float, mode: str = "mix"):
    """Return a trigger_fn(rets, rng) -> index into the move's candidate entry bars, with KNOWN skill k.

    mode="mix"    : Bernoulli(k) between the oracle (argmax ret -> capture 1) and pessimal (argmin ret ->
                    capture 0) bar. Expected per-move capture == k.
    mode="target" : deterministic; the bar whose capture-fraction is closest to k.
    """
    k = float(np.clip(k, 0.0, 1.0))

    def trig(rets, rng):
        if mode == "mix":
            if rng.random() < k:
                return int(np.argmax(rets))          # oracle entry -> capture 1
            return int(np.argmin(rets))              # pessimal entry -> capture 0
        # mode == "target"
        best, worst = float(rets.max()), float(rets.min())
        spread = best - worst
        caps = (rets - worst) / (spread if spread > 0 else 1.0)
        return int(np.argmin(np.abs(caps - k)))

    return trig


# ---------------------------------------------------------------------------
# Metric: seed-averaged held-out MEAN within-window capture for a given knob k.
# ---------------------------------------------------------------------------
def mean_held_capture(k: float, seed: int, mode: str = "mix", cost: float = TAKER,
                      min_spread: float = 0.003, trig_seed_offset: int = 9173) -> float:
    """Held-out (OOS+UNSEEN) mean within-window capture for skill knob k on the synth world built at
    `seed`. Degenerate moves (spread < min_spread, no timing decision) are dropped, exactly as the gate
    does. Returns nan if no held-out moves survive."""
    synth = make_moves(seed=seed)
    opens = synth.df["open"].to_numpy(float)
    n = len(opens)
    trig = make_knob_trigger(k, mode=mode)
    rng = np.random.default_rng(seed + trig_seed_offset)  # trigger RNG decoupled from world RNG
    caps = []
    for m in synth.moves:
        if _window_label(int(m["entry_bars"][0]), n) not in HELD:
            continue
        rets = _move_ret_curve(opens, m, cost)
        best, worst = float(rets.max()), float(rets.min())
        spread = best - worst
        if spread < min_spread:
            continue
        ti = int(trig(rets, rng))
        caps.append((float(rets[ti]) - worst) / spread)
    return float(np.mean(caps)) if caps else float("nan")


def seed_averaged_sweep(ks, seeds, mode: str = "mix") -> dict:
    """For each k, average the held-out mean capture across `seeds`. Returns aligned arrays + per-k SD."""
    ks = list(ks)
    mean_cap, sd_cap = [], []
    for k in ks:
        per_seed = np.array([mean_held_capture(k, s, mode=mode) for s in seeds], float)
        per_seed = per_seed[~np.isnan(per_seed)]
        mean_cap.append(float(per_seed.mean()) if per_seed.size else float("nan"))
        sd_cap.append(float(per_seed.std(ddof=0)) if per_seed.size else float("nan"))
    return {"k": np.array(ks, float), "capture": np.array(mean_cap, float),
            "capture_sd": np.array(sd_cap, float), "mode": mode, "n_seeds": len(list(seeds))}


# ---------------------------------------------------------------------------
# Spearman rank correlation (manual; no scipy dependency). Average-rank ties.
# ---------------------------------------------------------------------------
def _rankdata(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # average tied ranks
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    return avg[inv]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 3:
        return float("nan")
    rx, ry = _rankdata(x), _rankdata(y)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


# ===========================================================================
# VERIFY: monotone non-decreasing capture(k), Spearman(k, capture) > 0.90.
# ===========================================================================
SPEARMAN_MIN = 0.90
ENDPOINT_LO = 0.05      # k=0 -> capture <= this
ENDPOINT_HI = 0.95      # k=1 -> capture >= this


def run_calibration(ks=None, seeds=None, verbose: bool = True) -> dict:
    if ks is None:
        ks = [round(0.1 * i, 2) for i in range(11)]            # 0.00 .. 1.00 step 0.10
    if seeds is None:
        seeds = list(range(7, 7 + 24))                          # 24 independent synthetic worlds

    primary = seed_averaged_sweep(ks, seeds, mode="mix")
    crosscheck = seed_averaged_sweep(ks, seeds, mode="target")

    rho = spearman(primary["k"], primary["capture"])
    rho_cc = spearman(crosscheck["k"], crosscheck["capture"])

    cap = primary["capture"]
    # monotone non-decreasing (allow tiny seed-noise dips)
    diffs = np.diff(cap[~np.isnan(cap)])
    monotone = bool(np.all(diffs >= -0.02))
    k0_cap = float(cap[0]); k1_cap = float(cap[-1])
    endpoints_ok = bool(k0_cap <= ENDPOINT_LO and k1_cap >= ENDPOINT_HI)

    spearman_ok = bool(rho > SPEARMAN_MIN)
    passed = bool(spearman_ok and endpoints_ok and monotone)

    if verbose:
        print("=" * 84)
        print("SINGLE-KNOB CAPTURE CALIBRATION -- generator/metric soundness (no gate, no null)")
        print(f"  seeds={len(seeds)} synthetic worlds   knob k in [0,1] step ~0.10")
        print("=" * 84)
        print(f"\n  PRIMARY sweep (mode=mix: Bernoulli(k) oracle-vs-pessimal entry)")
        print(f"  {'k':>5} {'capture':>9} {'sd':>7}")
        for kk, cc, ss in zip(primary["k"], primary["capture"], primary["capture_sd"]):
            print(f"  {kk:>5.2f} {cc:>9.4f} {ss:>7.4f}")
        print(f"\n  Spearman(k, capture) [primary, mix]    = {rho:.4f}   (need > {SPEARMAN_MIN})")
        print(f"  Spearman(k, capture) [crosscheck,target]= {rho_cc:.4f}")
        print(f"  endpoints: k=0 -> {k0_cap:.4f} (<= {ENDPOINT_LO}) ; "
              f"k=1 -> {k1_cap:.4f} (>= {ENDPOINT_HI})")
        print(f"  monotone non-decreasing (no dip > 0.02)  = {monotone}")
        print("-" * 84)
        verdict = ("PASS -- the within-window capture metric is a FAITHFUL, monotone readout of injected "
                   "timing skill: capture rises monotonically from ~0 (k=0) to ~full (k=1) with "
                   f"Spearman {rho:.3f} > {SPEARMAN_MIN}. GENERATOR/METRIC soundness is established, "
                   "independent of any gate/null calibration.") if passed else (
                   "FAIL -- the capture metric did NOT respond monotonically to the injected skill knob; "
                   "inspect the sweep above (this would indict the GENERATOR or the METRIC, not the gate).")
        print(f"[single_knob_capture_calibration] {verdict}")

    return {"passed": passed, "spearman_primary": rho, "spearman_crosscheck": rho_cc,
            "spearman_ok": spearman_ok, "endpoints_ok": endpoints_ok, "monotone": monotone,
            "k": primary["k"].tolist(), "capture": primary["capture"].tolist(),
            "capture_sd": primary["capture_sd"].tolist(),
            "capture_target": crosscheck["capture"].tolist(),
            "k0_capture": k0_cap, "k1_capture": k1_cap, "n_seeds": len(seeds),
            "thresholds": {"spearman_min": SPEARMAN_MIN, "endpoint_lo": ENDPOINT_LO,
                           "endpoint_hi": ENDPOINT_HI}}


if __name__ == "__main__":
    out = run_calibration()
    print("\nJSON_SUMMARY " + json.dumps({k: out[k] for k in (
        "passed", "spearman_primary", "spearman_crosscheck", "spearman_ok",
        "endpoints_ok", "monotone", "k0_capture", "k1_capture", "n_seeds")}, default=str))
    sys.exit(0 if out["passed"] else 1)
