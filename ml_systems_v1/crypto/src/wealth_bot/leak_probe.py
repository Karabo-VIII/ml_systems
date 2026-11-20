"""src/wealth_bot/leak_probe.py -- LD-2: shift-sensitivity look-ahead probe.

ADDITIVE diagnostic (Apparatus Lock-Down Spec LD-2, docs/APPARATUS_LOCKDOWN_SPEC_2026_06_04.md).
It does NOT modify the simulate/cost path -- it wraps CanonicalHarness by re-running it with
the signal (and, separately, the filter) columns shifted ONE extra bar into the past.

Rationale: a genuinely past-only signal is only MILDLY sensitive to an extra 1-bar lag (slightly
different entry timing -> modestly different trades). A look-ahead signal COLLAPSES when shifted
back, because its value at bar t secretly encodes information about bar t+1; shifting removes that
and the apparent edge evaporates. So a large compound swing from a single extra bar of lag is the
signature of look-ahead. This replaces the harness's hardcoded Q4 "VERIFIED" string with a measured
value.

Verdict (max_abs_delta = max over the 4 windows of |compound_base - compound_shifted|, in pp):
  > high_pp (20)    -> LEAK_HIGH_CONFIDENCE
  > suspect_pp (5)  -> LEAK_SUSPECT
  else              -> PAST_ONLY_OK

CALIBRATION FINDING (RWYB 2026-06-04, BTC 1d): the ABSOLUTE pp thresholds (5/20) are
CADENCE-DEPENDENT and over-trigger on coarse bars. On daily bars one bar = one large move, so even
a genuinely past-only WMA(10,30) crossover swung +17pp on TRAIN from a 1-bar shift and tripped
LEAK_HIGH_CONFIDENCE (max_abs_delta 33pp) -- a FALSE POSITIVE. The probe still DISCRIMINATES (a
1-bar-forward-leaked control gave 85.6pp >> the legit 33.2pp), but the verdict must be RELATIVE,
not absolute: compare the +1-bar delta against a same-cadence past-only BASELINE (or a shift-spectrum
[+1,+2,+3,...]; a leak shows a DISCONTINUITY at the past/future boundary, a past-only signal degrades
smoothly). Until calibrated per cadence, treat the absolute verdict as ADVISORY only. TODO (supervised):
replace the fixed-pp verdict with the shift-spectrum discontinuity test.

NOTE: also necessary-not-sufficient at the degenerate end -- a config that loses ~100% in every
window yields ~0 delta and passes trivially; interpret alongside a non-degenerate backtest.
"""
from __future__ import annotations

from wealth_bot.harness import CanonicalHarness


def _compounds(results, windows) -> dict:
    return {w: float(results.window_stats[w].compound_pct) for w in windows}


def _run_with_shifted_columns(harness: CanonicalHarness, cols, shift_bars: int):
    """Re-run the harness with the given columns shifted `shift_bars` further into the past.
    Read-only w.r.t. the original harness (operates on a df copy)."""
    df2 = harness.df.copy()
    shifted = []
    for col in cols:
        if col and col in df2.columns:
            df2[col] = df2[col].shift(shift_bars)
            shifted.append(col)
    h2 = CanonicalHarness(
        df2, harness.spec, harness.windows,
        chimera_path=harness.chimera_path,
        command_line=f"leak_probe.shift({shift_bars}):{','.join(shifted)}",
    )
    return h2.run()


def shift_sensitivity_test(harness: CanonicalHarness, shift_bars: int = 1,
                           suspect_pp: float = 5.0, high_pp: float = 20.0) -> dict:
    """Run the LD-2 shift-sensitivity probe on a constructed CanonicalHarness.

    Returns a dict with the verdict, the max absolute compound delta (pp), and the
    per-window deltas for both the signal columns and (independently) the filter column.
    """
    windows = list(harness.WINDOWS)
    base = _compounds(harness.run(), windows)

    # (1) signal columns (fast + slow) shifted one extra bar into the past
    sig_cols = [harness.spec.fast_col, harness.spec.slow_col]
    sig_shift = _compounds(_run_with_shifted_columns(harness, sig_cols, shift_bars), windows)
    sig_delta = {w: base[w] - sig_shift[w] for w in windows}
    sig_max_abs = max(abs(v) for v in sig_delta.values())

    # (2) filter column shifted independently (a separate leak vector per LD-2)
    filt_col = getattr(harness.spec, "filter_col", None)
    if filt_col and filt_col in harness.df.columns:
        filt_shift = _compounds(_run_with_shifted_columns(harness, [filt_col], shift_bars), windows)
        filt_delta = {w: base[w] - filt_shift[w] for w in windows}
        filt_max_abs = max(abs(v) for v in filt_delta.values())
    else:
        filt_delta, filt_max_abs = None, 0.0

    max_abs = max(sig_max_abs, filt_max_abs)
    if max_abs > high_pp:
        verdict = "LEAK_HIGH_CONFIDENCE"
    elif max_abs > suspect_pp:
        verdict = "LEAK_SUSPECT"
    else:
        verdict = "PAST_ONLY_OK"

    return {
        "verdict": verdict,
        "verdict_status": "ADVISORY_CADENCE_SENSITIVE",  # fixed-pp threshold over-triggers on coarse bars; see module docstring CALIBRATION FINDING
        "cadence_caveat": "Absolute pp thresholds are cadence-dependent; on coarse (1d) bars a past-only signal can exceed them. Use a same-cadence past-only baseline or a shift-spectrum discontinuity for a real verdict.",
        "shift_bars": shift_bars,
        "max_abs_delta_pp": round(max_abs, 4),
        "signal": {
            "base_compound_pct": base,
            "shifted_compound_pct": sig_shift,
            "delta_pp": sig_delta,
            "max_abs_delta_pp": round(sig_max_abs, 4),
        },
        "filter": (
            {"delta_pp": filt_delta, "max_abs_delta_pp": round(filt_max_abs, 4)}
            if filt_delta is not None else "no_filter_col"
        ),
        "thresholds_pp": {"suspect": suspect_pp, "high": high_pp},
    }


def shift_spectrum_test(harness: CanonicalHarness, shifts=(0, 1, 2, 3),
                        suspect_floor_pp: float = 3.0, ratio_threshold: float = 3.0) -> dict:
    """CADENCE-ROBUST leak verdict (corrected design, fixes the fixed-pp false-positive).

    Runs the signal at increasing extra lag [0,1,2,3] and looks at the INCREMENTAL compound
    deltas between consecutive lags. A genuinely past-only signal degrades SMOOTHLY (each extra
    lag contributes a similar increment). A leak shows a DISCONTINUITY: the FIRST extra lag (which
    crosses the past/future boundary and removes the future info) produces a delta far larger than
    the later, purely-past lag steps. Verdict = LEAK if d1 > suspect_floor AND d1 > ratio_threshold *
    median(later increments). This is cadence-robust because it compares increments on the SAME
    cadence (the absolute magnitude that scales with cadence cancels in the ratio).

    WARNING -- RWYB 2026-06-05: THIS DESIGN ALSO FAILED on coarse (1d) bars. The leaky control was
    NOT flagged (both PAST_ONLY_OK): incs legit=[17,22,42], leaky=[86,91,75] -- on daily bars even
    past-only->more-past shifts swing compound ~80-90pp, so there is NO clean first-step
    discontinuity; the noise floor drowns it. FINDING: shift-based leak detection (absolute OR
    discontinuity) is unreliable on coarse/high-compounding data. Absolute MAGNITUDE does
    discriminate (~2.6x: leaky 86 vs legit 33) but has no universal threshold. CORRECT DESIGN
    (supervised, next hypothesis): RELATIVE to a known-past-only TWIN built via past_only_indicator()
    -- flag the candidate if its shift-sensitivity >> the twin's. Until then treat ALL verdicts here
    as ADVISORY and use shift probes on FINE bars only."""
    from statistics import median

    windows = list(harness.WINDOWS)
    sig_cols = [harness.spec.fast_col, harness.spec.slow_col]
    comp_by_shift = {}
    for s in shifts:
        if s == 0:
            comp_by_shift[s] = _compounds(harness.run(), windows)
        else:
            comp_by_shift[s] = _compounds(_run_with_shifted_columns(harness, sig_cols, s), windows)

    incs = []  # max-abs compound delta across windows between consecutive shift steps
    for a, b in zip(shifts[:-1], shifts[1:]):
        incs.append(max(abs(comp_by_shift[a][w] - comp_by_shift[b][w]) for w in windows))

    d1 = incs[0] if incs else 0.0
    later = incs[1:] if len(incs) > 1 else [0.0]
    later_med = median(later) if later else 0.0
    if later_med > 1e-9:
        ratio = d1 / later_med
    else:
        ratio = float("inf") if d1 > suspect_floor_pp else 0.0
    is_leak = (d1 > suspect_floor_pp) and (ratio > ratio_threshold)
    verdict = "LEAK_SUSPECT (discontinuity at first lag)" if is_leak else "PAST_ONLY_OK (smooth degradation)"

    return {
        "verdict": verdict,
        "first_step_delta_pp": round(d1, 4),
        "later_steps_median_pp": round(later_med, 4),
        "discontinuity_ratio": (round(ratio, 3) if ratio != float("inf") else "inf"),
        "incremental_deltas_pp": [round(x, 4) for x in incs],
        "compound_by_extra_lag": {s: comp_by_shift[s] for s in shifts},
        "thresholds": {"suspect_floor_pp": suspect_floor_pp, "ratio_threshold": ratio_threshold},
    }


def relative_leak_test(candidate_harness: CanonicalHarness, reference_harness: CanonicalHarness,
                       shift_bars: int = 1, ratio_threshold: float = 2.0) -> dict:
    """CADENCE-ROBUST leak verdict (the corrected design that works — RWYB 2026-06-05).

    The absolute-pp and shift-spectrum verdicts both FAILED on coarse bars because the noise floor of
    shift-sensitivity is large and cadence-dependent. The fix: compare the candidate's shift-sensitivity
    to a KNOWN-PAST-ONLY REFERENCE strategy run on the SAME cadence/asset (e.g. a plain past-only MA
    crossover). Both share the cadence noise floor, so it CANCELS in the ratio. A genuinely past-only
    candidate has ratio ~1; a leak has ratio >> 1 (it collapses far more when shifted because its value
    encodes future info the reference's doesn't).

    Verdict: LEAK_SUSPECT if candidate_delta > ratio_threshold * reference_delta, else PAST_ONLY_OK.
    Validated: legit-vs-legit ratio ~1.0 (OK); 1-bar-forward-leaked-vs-legit ratio ~2.6 (LEAK)."""
    cand = shift_sensitivity_test(candidate_harness, shift_bars)["max_abs_delta_pp"]
    ref = shift_sensitivity_test(reference_harness, shift_bars)["max_abs_delta_pp"]
    if ref > 1e-9:
        ratio = cand / ref
    else:
        ratio = float("inf") if cand > 1.0 else 0.0
    verdict = "LEAK_SUSPECT" if (ratio == float("inf") or ratio > ratio_threshold) else "PAST_ONLY_OK"
    return {
        "candidate_delta_pp": round(cand, 4), "reference_delta_pp": round(ref, 4),
        "ratio": (round(ratio, 3) if ratio != float("inf") else "inf"),
        "ratio_threshold": ratio_threshold, "verdict": verdict,
    }
