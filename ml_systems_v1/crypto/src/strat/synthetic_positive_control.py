"""src/strat/synthetic_positive_control.py -- TWO-KNOB synthetic positive-control generator + null calibration.

WHY (founding framing): the unit of trading is a SETUP across a MOVE (multiple candles). Two DISTINCT
genuine skills can drive held-out compound return, and each needs its OWN matched null to be measured
fairly without crediting beta/luck:

  (1) TIMING skill  -- given the move you are present in, you enter/exit so as to capture a known
      fraction `f` of the within-window best-vs-worst entry spread. f=0 => no better than a random
      entry inside the SAME move; f=1 => oracle (the best eligible entry).
        => The fair null is the WITHIN-WINDOW (membership-matched) null: hold WHICH moves you are in
           CONSTANT, randomize only the entry timing inside each move. Beating it == real timing value.

  (2) SELECTION skill -- you preferentially pick ABOVE-MEDIAN-drift windows to be present in (with a
      controllable skill `s`: s=0 => random windows; s=1 => always above the drift median). You take a
      NEUTRAL (no-timing) entry inside each picked window, so selection is isolated from timing.
        => The fair null is the HORIZON-MATCHED null: randomize WHICH windows you pick (same count K,
           same holding horizon h, same neutral entry), compound. Beating it == real selection value.

This module is a GENERATOR (synthetic, deterministic/seeded, NO market claim) that lets us:
  A. Confirm the WITHIN-WINDOW null PASSES (admits/detects) a genuine TIMING setup.
  B. Confirm the HORIZON-MATCHED null PASSES (admits/detects) a genuine SELECTION setup.
  C. Record each null's FALSE-NEGATIVE rate at a sweep of KNOWN skill levels -> calibrate over-rejection
     (how strong a genuine edge must be before the gate reliably stops missing it), and report the
     skill=0 rejection rate (== empirical false-POSITIVE / test size, should sit near the nominal alpha).

"Null PASSES the variant" == the matched null-test correctly DETECTS the genuine skill (real compound
beats the null's p95 band) => the gate has POWER for that skill, it is not a reject-everything sieve.

Specificity cross-checks (D) confirm each null is TARGETED: the within-window null does NOT credit a
pure SELECTION setup as timing (membership held constant -> ties the null).

RWYB: `python src/strat/synthetic_positive_control.py`  (exit 0 == all calibration invariants hold).
"""
from __future__ import annotations

import json
import numpy as np


# --------------------------------------------------------------------------------------------------
# WORLD GENERATION (synthetic). A world = W independent multi-candle MOVES (windows). Each window has a
# drift d_w (selection signal lives here) and L noisy bars (timing signal lives inside here).
# --------------------------------------------------------------------------------------------------
def make_world(rng, W=40, L=24, h=6, drift_sd=0.020, noise=0.012, cost=0.0008):
    """Return per-window eligible-entry return arrays + per-window drift.

    For each window w: simple bar returns = d_w + N(0, noise); a trade entered at eligible bar i is held
    for a FIXED horizon h and exits at i+h, so every trade in the study shares the SAME holding horizon
    (horizon-matched by construction). `elig_rets[w]` = net (after `cost`) return of entering at each
    eligible bar i in [0, L-h]. `drift[w]` = the window's per-bar drift (the selection target)."""
    drift = rng.normal(0.0, drift_sd, W)
    elig_rets = []
    for w in range(W):
        bar = drift[w] + rng.normal(0.0, noise, L)
        price = np.concatenate([[1.0], np.cumprod(1.0 + bar)])  # price[k] after k bars
        elig = np.arange(0, L - h + 1)
        rets = price[elig + h] / price[elig] - 1.0 - cost      # net return per eligible entry, horizon h
        elig_rets.append(rets)
    return {"elig_rets": elig_rets, "drift": drift, "W": W, "L": L, "h": h, "cost": cost}


def _compound(per_trade):
    a = np.asarray(per_trade, float)
    return float(np.prod(1.0 + a) - 1.0) if a.size else 0.0


# --------------------------------------------------------------------------------------------------
# KNOB 1 -- TIMING setup: present in EVERY window; inside each, captures fraction f of best-over-random.
# --------------------------------------------------------------------------------------------------
def timing_setup_returns(world, f):
    """Per-window realized trade return of a TIMING setup with skill f in [0,1].

    Within window w, random/no-skill entry has expected return mean(rets_w); the oracle is max(rets_w).
    The skill-f setup realizes  mean + f*(max - mean)  == it captures fraction f of the best-over-average
    timing headroom (f=0 ties the within-window null in expectation; f=1 is the oracle entry). We also
    expose the equivalent fraction of the full best-vs-WORST spread for transparency."""
    out = np.empty(world["W"])
    frac_of_full_spread = np.empty(world["W"])
    for w, rets in enumerate(world["elig_rets"]):
        mean_r, best, worst = rets.mean(), rets.max(), rets.min()
        out[w] = mean_r + f * (best - mean_r)
        spread = best - worst
        frac_of_full_spread[w] = (out[w] - worst) / spread if spread > 0 else np.nan
    return out, float(np.nanmean(frac_of_full_spread))


# --------------------------------------------------------------------------------------------------
# KNOB 2 -- SELECTION setup: picks K windows, preferring ABOVE-median-drift, with skill s in [0,1].
# Inside each picked window it takes a NEUTRAL (no-timing) entry == the MEDIAN eligible-entry return,
# so the setup's value comes ONLY from WHICH windows it is in, not from timing.
# --------------------------------------------------------------------------------------------------
def _neutral_return(rets):
    """No-timing entry == median eligible-entry return (a typical, non-cherry-picked entry in the move)."""
    return float(np.median(rets))


def selection_setup(world, rng, K, s):
    """Return (picked_window_indices, per_trade_neutral_returns) for a SELECTION setup with skill s.

    s controls the probability each pick lands in the ABOVE-median-drift pool: p = 0.5 + 0.5*s
    (s=0 => 0.5, pure chance; s=1 => 1.0, always above median). Picks are without replacement within
    each pool; falls back across pools if a pool is exhausted."""
    drift = world["drift"]
    med = np.median(drift)
    above = list(np.where(drift > med)[0])
    below = list(np.where(drift <= med)[0])
    rng.shuffle(above)
    rng.shuffle(below)
    p_above = 0.5 + 0.5 * s
    picks = []
    for _ in range(K):
        want_above = rng.random() < p_above
        pool = above if want_above else below
        alt = below if want_above else above
        if not pool:
            pool = alt
        if not pool:
            break
        picks.append(pool.pop())
    rets = [_neutral_return(world["elig_rets"][w]) for w in picks]
    return picks, np.asarray(rets, float)


# --------------------------------------------------------------------------------------------------
# NULL 1 -- WITHIN-WINDOW (membership-matched): hold the setup's windows CONSTANT, randomize entry
# timing inside each. Correct null for TIMING skill.
# --------------------------------------------------------------------------------------------------
def within_window_null(world, member_windows, rng, n_books=600):
    """Distribution of compound return when entry TIMING is random inside the SAME windows the setup
    is present in. One random eligible entry per member window, compounded; repeated n_books times."""
    rets_by_w = [world["elig_rets"][w] for w in member_windows]
    books = np.empty(n_books)
    for b in range(n_books):
        draw = [rng.choice(r) for r in rets_by_w]
        books[b] = _compound(draw)
    return books


# --------------------------------------------------------------------------------------------------
# NULL 2 -- HORIZON-MATCHED: randomize WHICH windows are picked (same count K, same horizon h, same
# neutral entry), compound. Correct null for SELECTION skill.
# --------------------------------------------------------------------------------------------------
def horizon_matched_null(world, K, rng, n_books=600):
    """Distribution of compound return when window SELECTION is random: pick K windows uniformly at
    random (without replacement) from all W, take the neutral entry in each, compound; n_books times."""
    W = world["W"]
    neutral = np.array([_neutral_return(r) for r in world["elig_rets"]])
    books = np.empty(n_books)
    for b in range(n_books):
        picks = rng.choice(W, size=min(K, W), replace=False)
        books[b] = _compound(neutral[picks])
    return books


def _detect(real_compound, null_books, alpha=0.05):
    """A one-sided detection at level alpha: real compound beats the null's (1-alpha) quantile band."""
    thresh = float(np.percentile(null_books, 100 * (1 - alpha)))
    return bool(real_compound > thresh), thresh


# --------------------------------------------------------------------------------------------------
# A+B: single-world demonstration that each matched null PASSES (detects) its genuine skill.
# --------------------------------------------------------------------------------------------------
def demonstrate(seed=0, f=0.6, s=0.8, K=20, n_books=600, verbose=True):
    rng = np.random.default_rng(seed)
    world = make_world(rng)

    t_rets, frac_full = timing_setup_returns(world, f)
    members = list(range(world["W"]))
    t_real = _compound(t_rets)
    t_null = within_window_null(world, members, rng, n_books)
    t_detect, t_thr = _detect(t_real, t_null)

    picks, sel_rets = selection_setup(world, rng, K, s)
    s_real = _compound(sel_rets)
    s_null = horizon_matched_null(world, K, rng, n_books)
    s_detect, s_thr = _detect(s_real, s_null)

    if verbose:
        print("=" * 92)
        print(f"A. TIMING setup (f={f}, ~{frac_full*100:.0f}% of best-vs-WORST spread captured) vs WITHIN-WINDOW null")
        print(f"   real_compound = {t_real*100:+.2f}%   null p50={np.percentile(t_null,50)*100:+.2f}%  "
              f"p95={t_thr*100:+.2f}%   -> DETECTED={t_detect}  (within-window null PASSES the timing variant)")
        print(f"B. SELECTION setup (s={s}, picks K={K} above-median-drift windows) vs HORIZON-MATCHED null")
        print(f"   real_compound = {s_real*100:+.2f}%   null p50={np.percentile(s_null,50)*100:+.2f}%  "
              f"p95={s_thr*100:+.2f}%   -> DETECTED={s_detect}  (horizon-matched null PASSES the selection variant)")
    return {"timing_detected": t_detect, "selection_detected": s_detect,
            "timing_frac_full_spread": frac_full}


# --------------------------------------------------------------------------------------------------
# C: false-negative-rate calibration across a sweep of KNOWN skill levels (over-rejection curve).
# --------------------------------------------------------------------------------------------------
def fn_curve_timing(levels, reps=200, n_books=400, K=None, base_seed=1000, alpha=0.05):
    """For each timing skill f, fraction of independent worlds where the within-window null FAILS to
    detect the genuine timing setup (false negative). At f=0 this is the empirical false-POSITIVE rate."""
    rows = []
    for f in levels:
        miss = 0
        for r in range(reps):
            rng = np.random.default_rng(base_seed + r)
            world = make_world(rng)
            t_rets, _ = timing_setup_returns(world, f)
            members = list(range(world["W"]))
            real = _compound(t_rets)
            books = within_window_null(world, members, rng, n_books)
            detected, _ = _detect(real, books, alpha)
            miss += (0 if detected else 1)
        rows.append({"skill_f": round(f, 3), "false_negative_rate": round(miss / reps, 4),
                     "detection_rate": round(1 - miss / reps, 4), "reps": reps})
    return rows


def fn_curve_selection(levels, reps=200, n_books=400, K=20, base_seed=5000, alpha=0.05):
    """For each selection skill s, fraction of independent worlds where the horizon-matched null FAILS to
    detect the genuine selection setup (false negative). At s=0 this is the empirical false-POSITIVE rate."""
    rows = []
    for s in levels:
        miss = 0
        for r in range(reps):
            rng = np.random.default_rng(base_seed + r)
            world = make_world(rng)
            picks, sel_rets = selection_setup(world, rng, K, s)
            real = _compound(sel_rets)
            books = horizon_matched_null(world, K, rng, n_books)
            detected, _ = _detect(real, books, alpha)
            miss += (0 if detected else 1)
        rows.append({"skill_s": round(s, 3), "false_negative_rate": round(miss / reps, 4),
                     "detection_rate": round(1 - miss / reps, 4), "reps": reps})
    return rows


# --------------------------------------------------------------------------------------------------
# D: specificity cross-check -- the within-window null must NOT credit pure SELECTION skill as timing.
# A selection-only setup, when its membership is held constant, takes neutral entries == ties the null.
# --------------------------------------------------------------------------------------------------
def specificity_within_window_vs_selection(reps=200, n_books=400, K=20, s=1.0, base_seed=9000, alpha=0.05):
    """Rate at which the WITHIN-WINDOW null (mis)detects a pure SELECTION-only setup as a timing edge.
    Should sit near alpha (the null is membership-matched, so selection skill cancels out)."""
    false_credit = 0
    for r in range(reps):
        rng = np.random.default_rng(base_seed + r)
        world = make_world(rng)
        picks, sel_rets = selection_setup(world, rng, K, s)   # genuine SELECTION, neutral (no timing) entries
        real = _compound(sel_rets)
        books = within_window_null(world, picks, rng, n_books)  # membership = the SAME picked windows
        detected, _ = _detect(real, books, alpha)
        false_credit += (1 if detected else 0)
    return round(false_credit / reps, 4)


# --------------------------------------------------------------------------------------------------
def main():
    print("\nSYNTHETIC POSITIVE-CONTROL GENERATOR -- two knobs (TIMING, SELECTION) + matched-null calibration\n")
    demo = demonstrate(seed=0, f=0.6, s=0.8, verbose=True)

    print("\n" + "=" * 92)
    print("C1. WITHIN-WINDOW null -- false-negative rate vs TIMING skill f (over-rejection curve):")
    t_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
    t_rows = fn_curve_timing(t_levels, reps=200, n_books=400)
    for row in t_rows:
        print(f"   f={row['skill_f']:.2f}  FN={row['false_negative_rate']:.3f}  detect={row['detection_rate']:.3f}")
    # at skill=0 the DETECTION rate IS the empirical false-positive rate (test size) -- the gate
    # rejecting when there is no skill. (false_negative_rate@0 = 1 - this; using FN here was an
    # inversion bug that flagged a CONSERVATIVE gate as broken. 2026-06-08 overseer correct-as-you-go.)
    fp_timing = t_rows[0]["detection_rate"]

    print("\nC2. HORIZON-MATCHED null -- false-negative rate vs SELECTION skill s (over-rejection curve):")
    s_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
    s_rows = fn_curve_selection(s_levels, reps=200, n_books=400)
    for row in s_rows:
        print(f"   s={row['skill_s']:.2f}  FN={row['false_negative_rate']:.3f}  detect={row['detection_rate']:.3f}")
    fp_select = s_rows[0]["detection_rate"]  # at s=0 detection == false-positive rate (test size)

    print("\nD. SPECIFICITY -- within-window null mis-crediting a pure SELECTION-only setup as timing:")
    spec = specificity_within_window_vs_selection(reps=200, n_books=400, s=1.0)
    print(f"   false-credit rate = {spec:.3f}  (should sit near alpha=0.05; membership-matched cancels selection)")

    # ---- calibration invariants (exit nonzero on violation) ----
    print("\n" + "=" * 92)
    checks = []
    checks.append(("within-window null PASSES timing variant (f=0.6 detected)", demo["timing_detected"]))
    checks.append(("horizon-matched null PASSES selection variant (s=0.8 detected)", demo["selection_detected"]))
    checks.append((f"timing FN->0 at high skill (FN@f=1.0={t_rows[-1]['false_negative_rate']} <= 0.05)",
                   t_rows[-1]["false_negative_rate"] <= 0.05))
    checks.append((f"selection FN->0 at high skill (FN@s=1.0={s_rows[-1]['false_negative_rate']} <= 0.05)",
                   s_rows[-1]["false_negative_rate"] <= 0.05))
    checks.append((f"timing FALSE-POSITIVE size (FP@f=0=={fp_timing} in [0.0,0.15])", 0.0 <= fp_timing <= 0.15))
    checks.append((f"selection FALSE-POSITIVE size (FP@s=0=={fp_select} in [0.0,0.15])", 0.0 <= fp_select <= 0.15))
    checks.append((f"within-window null does NOT credit selection ({spec} <= 0.15)", spec <= 0.15))
    monotone_t = all(t_rows[i]["false_negative_rate"] >= t_rows[i + 1]["false_negative_rate"] - 0.06
                     for i in range(len(t_rows) - 1))
    monotone_s = all(s_rows[i]["false_negative_rate"] >= s_rows[i + 1]["false_negative_rate"] - 0.06
                     for i in range(len(s_rows) - 1))
    checks.append(("timing FN ~monotone-decreasing in skill", monotone_t))
    checks.append(("selection FN ~monotone-decreasing in skill", monotone_s))

    ok = all(c[1] for c in checks)
    for name, passed in checks:
        print(f"   [{'PASS' if passed else 'FAIL'}] {name}")

    summary = {"demo": demo, "timing_fn_curve": t_rows, "selection_fn_curve": s_rows,
               "within_window_credits_selection_rate": spec, "all_invariants_pass": ok}
    print("\nJSON_SUMMARY " + json.dumps(summary, default=str))
    print("\n" + ("ALL CALIBRATION INVARIANTS HOLD" if ok else "*** CALIBRATION INVARIANT VIOLATED ***"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
