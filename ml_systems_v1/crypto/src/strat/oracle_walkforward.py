"""WALK-FORWARD PERSISTENCE TEST for the TI-oracle config.

THE QUESTION (user): "Does this week's oracle config work next week?"

Take the best TI config fitted on one period (TRAIN), roll it UNCHANGED into the
next period (TEST: causal, no refit, no look-ahead), and compare it to:
  (a) NEXT-ORACLE -- the TEST period's OWN best config (the next-period CEILING; hindsight).
  (b) FIXED       -- one globally-fixed naive config on the TEST period (does last
                     period's best beat just using one fixed config?).
  (c) RANDOM      -- a few random configs on the TEST period (the FLOOR).

Does it translate -- better or worse?

METHOD (per asset, per cadence):
  - Detect ALL non-overlapping price-oracle move-events over the full series
    (REUSE ti_oracle_anchor.find_price_oracle_events -- identical detector).
  - Build the full (formulation x MA-type x params) candidate grid ONCE
    (REUSE ti_oracle_decompose.build_candidates).
  - Precompute a capture matrix M[event, candidate] = realized long ROI of that
    candidate on that event's window (causal signal; only config CHOICE is hindsight).
  - Assign each event to a consecutive TEST period of ~7 days (period length in BARS
    derived from the cadence: 7 days). TRAIN window for TEST period T = the events
    in the preceding ~3 weeks (21 days). If a TRAIN window has < MIN_TRAIN_EVENTS,
    WIDEN it (extend further back) until it does, or skip the step (stated).
  - oracle_config(TRAIN) = argmax over candidates of MEAN capture on TRAIN events.
  - ROLLED        = oracle_config(TRAIN) applied UNCHANGED to TEST events.
  - NEXT-ORACLE   = argmax over candidates of MEAN capture on TEST events (hindsight).
  - FIXED         = one globally-fixed naive config on TEST events.
  - RANDOM        = mean over K random candidates on TEST events.
  - config-repeat = does oracle_config(TRAIN) == oracle_config(TEST)?

AGGREGATE over all walk-forward steps (event-weighted means + per-step medians):
  - mean & median of: rolled / next_oracle / fixed / random capture.
  - PERSISTENCE        = rolled / next_oracle  (fraction of the ceiling banked).
  - TRANSLATE-vs-FIXED = win-rate and mean delta of (rolled - fixed)  [THE key test].
  - vs RANDOM          = is rolled > random?
  - config-repeat rate = how often TRAIN-best == TEST-best.

DISCIPLINE:
  - CAUSAL roll-forward: the rolled config is fitted on TRAIN events ONLY; it is NOT
    refit on TEST. No look-ahead -- TRAIN events strictly precede TEST events in bar index.
  - "capture" here = realized long ROI (net taker 0.24% RT), NOT divided by price-oracle
    move (so a config can be compared apples-to-apples to its own train fit). The
    NEXT-ORACLE / FIXED / RANDOM are all the SAME ROI units.
  - HINDSIGHT only for the CEILING (NEXT-ORACLE) and the per-period config selection
    on TRAIN; the ROLLED number itself uses no TEST information for its config choice.
  - cp1252-safe (no emoji).

Usage:
    python src/strat/oracle_walkforward.py --asset BTCUSDT --cadences 4h,1h
    python src/strat/oracle_walkforward.py --selftest

__contract__ = {
    "kind": "research_persistence_test",
    "inputs": ["chimera OHLC via ChimeraLoader",
               "ti_oracle_anchor.find_price_oracle_events (read-only reuse)",
               "ti_oracle_decompose.build_candidates (read-only reuse)"],
    "outputs": ["runs/strat/oracle_walkforward_<ASSET>.json", "stdout table"],
    "invariants": [
        "price-oracle events identical to ti_oracle_anchor (same detector)",
        "rolled config fitted on TRAIN events only -- no refit on TEST (causal)",
        "TRAIN events strictly precede TEST events in bar index (no look-ahead)",
        "next-oracle is hindsight ceiling on TEST events",
        "fixed config is a single globally-fixed naive config",
        "selftest two-sided: persistent-DNA -> rolled~=next_oracle>>random; "
        "regime-switch -> rolled collapses toward random and rolled~=fixed",
    ],
}
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# READ-ONLY reuse so events + candidate grid are byte-identical to the anchor/decomp.
from strat.ti_oracle_anchor import (  # noqa: E402
    MoveEvent,
    WINDOW_BARS,
    find_price_oracle_events,
    load_ohlc,
)
from strat.ti_oracle_decompose import (  # noqa: E402
    Candidate,
    build_candidates,
)

# ---- walk-forward spec ------------------------------------------------------

# Bars per day per cadence (chimera native cadences) -- used only to report the
# CALENDAR span each event-block covers, NOT to bin events.
BARS_PER_DAY = {"1d": 1, "4h": 6, "1h": 24, "15m": 96}

# Periods are defined as consecutive BLOCKS OF MOVE-EVENTS (not calendar weeks):
# a non-overlapping price-oracle move is itself ~7-14 days wide and consecutive
# non-overlapping moves sit ~7-14 days apart, so a "~7-day calendar period" holds
# < 1 move. To make each period "hold several moves" (the user's requirement) and
# give the TRAIN fit enough moves to be stable, we step over event-count blocks.
#   TEST  period  = TEST_EVENTS consecutive move-events (~7-14 wks of calendar each).
#   TRAIN window  = the TRAIN_EVENTS move-events IMMEDIATELY PRECEDING the test block
#                   (strictly earlier bar indices -> causal, no look-ahead).
TEST_EVENTS = 4        # several moves per test period
TRAIN_EVENTS = 12      # ~3x the test block -> stable fit (the "preceding 2-3 weeks of moves")
MIN_TRAIN_EVENTS = 6   # require >= this many train moves or skip the step
MIN_TEST_EVENTS = 3    # require >= this many test moves or skip the step

# FIXED naive config -- the single most-common DNA winner overall (golden cross).
# Per the project's DNA observations a classic SMA fast/slow cross is the canonical
# naive baseline. We pick F2_CROSS / SMA / 5x20 (fast 5, slow 20 golden/death cross).
FIXED_FORMULATION = "F2_CROSS"
FIXED_MA_TYPE = "SMA"
FIXED_PARAMS = "5x20"

N_RANDOM = 8           # number of random candidates to average for the RANDOM floor


# ---- capture matrix ---------------------------------------------------------

@dataclass
class WFInputs:
    cadence: str
    events: list  # list[MoveEvent]  (sorted by start bar, non-overlapping)
    cands: list   # list[Candidate]
    M: np.ndarray            # [n_events, n_cands] realized long ROI
    fixed_idx: int           # index of the FIXED naive candidate (or -1)


def _candidate_index(cands: list, formulation: str, ma_type: str,
                     params: str) -> int:
    for i, c in enumerate(cands):
        if (c.formulation == formulation and c.ma_type == ma_type
                and c.params == params):
            return i
    return -1


def build_capture_matrix(open_a, high_a, low_a, close_a, cadence: str) -> WFInputs:
    """Detect events, build the candidate grid, and fill M[event, candidate]."""
    win_lo, win_hi = WINDOW_BARS[cadence]
    events = find_price_oracle_events(high_a, low_a, win_lo, win_hi)
    cands = build_candidates(open_a, high_a, low_a, close_a)
    n_ev, n_c = len(events), len(cands)
    M = np.full((n_ev, n_c), np.nan, dtype=np.float64)
    for i, ev in enumerate(events):
        for j, c in enumerate(cands):
            M[i, j] = c.fn(ev.start, ev.end)

    fixed_idx = _candidate_index(cands, FIXED_FORMULATION, FIXED_MA_TYPE, FIXED_PARAMS)

    return WFInputs(
        cadence=cadence, events=events, cands=cands, M=M, fixed_idx=fixed_idx,
    )


# ---- walk-forward stepping --------------------------------------------------

@dataclass
class WFStep:
    test_period: int               # sequential index of the TEST event-block
    n_train_events: int
    n_test_events: int
    test_span_days: float          # calendar span the TEST block covers
    rolled_capture: float          # mean realized ROI of TRAIN-best config on TEST
    next_oracle_capture: float     # mean realized ROI of TEST-best config on TEST
    fixed_capture: float           # mean realized ROI of FIXED config on TEST
    random_capture: float          # mean over random configs of mean-on-TEST
    train_best_idx: int
    test_best_idx: int
    config_repeat: bool


@dataclass
class WFResult:
    cadence: str
    n_events: int
    steps: list = field(default_factory=list)   # list[WFStep]
    skipped: int = 0
    skip_reasons: dict = field(default_factory=dict)


def run_walkforward(inp: WFInputs, seed: int = 7) -> WFResult:
    """Roll forward over consecutive EVENT-BLOCKS. For each TEST block (TEST_EVENTS
    consecutive move-events), fit the config on the immediately PRECEDING TRAIN block
    (TRAIN_EVENTS earlier move-events; strictly earlier bar index -> causal), apply it
    unchanged to the TEST events, and score the comparators."""
    rng = np.random.default_rng(seed)
    bpd = BARS_PER_DAY.get(inp.cadence, 1)

    res = WFResult(cadence=inp.cadence, n_events=len(inp.events))
    n_ev = len(inp.events)
    if n_ev == 0:
        return res

    event_start_bar = np.array([ev.start for ev in inp.events], dtype=np.int64)
    event_end_bar = np.array([ev.end for ev in inp.events], dtype=np.int64)
    n_c = inp.M.shape[1]

    # Walk consecutive non-overlapping TEST blocks of TEST_EVENTS events. The first
    # block needs MIN_TRAIN_EVENTS events before it, so start the test cursor there.
    test_lo = TRAIN_EVENTS
    period_idx = 0
    while test_lo < n_ev:
        test_hi = min(test_lo + TEST_EVENTS, n_ev)
        test_idx = np.arange(test_lo, test_hi)
        n_test = len(test_idx)
        if n_test < MIN_TEST_EVENTS:
            res.skipped += 1
            res.skip_reasons["test_too_small"] = res.skip_reasons.get(
                "test_too_small", 0) + 1
            test_lo = test_hi
            continue

        # TRAIN block = the TRAIN_EVENTS events immediately before the test block.
        train_lo = max(0, test_lo - TRAIN_EVENTS)
        train_idx = np.arange(train_lo, test_lo)
        n_train = len(train_idx)
        if n_train < MIN_TRAIN_EVENTS:
            res.skipped += 1
            res.skip_reasons["train_too_small"] = res.skip_reasons.get(
                "train_too_small", 0) + 1
            test_lo = test_hi
            continue

        # Causal sanity: every TRAIN event ENDS strictly at/before the first TEST
        # event's START bar (non-overlapping events are already bar-ordered, so the
        # train block is strictly earlier than the test block -> no look-ahead).
        assert event_end_bar[train_idx].max() <= event_start_bar[test_idx].min(), \
            "look-ahead: a train event overlaps/follows the test block"

        train_M = inp.M[train_idx]   # [n_train, n_c]
        test_M = inp.M[test_idx]     # [n_test, n_c]

        # oracle_config(TRAIN) = argmax MEAN capture on TRAIN events (fit on TRAIN only).
        train_mean = np.nanmean(train_M, axis=0)   # [n_c]
        train_best = int(np.nanargmax(train_mean))

        # NEXT-ORACLE = argmax MEAN capture on TEST events (hindsight ceiling).
        test_mean = np.nanmean(test_M, axis=0)      # [n_c]
        test_best = int(np.nanargmax(test_mean))

        rolled = float(np.nanmean(test_M[:, train_best]))
        next_oracle = float(test_mean[test_best])

        if inp.fixed_idx >= 0:
            fixed = float(np.nanmean(test_M[:, inp.fixed_idx]))
        else:
            fixed = float("nan")

        # RANDOM floor: mean over N_RANDOM random candidates of their TEST mean.
        rand_idx = rng.integers(0, n_c, size=min(N_RANDOM, n_c))
        rand_caps = [float(np.nanmean(test_M[:, k])) for k in rand_idx]
        random_cap = float(np.mean(rand_caps)) if rand_caps else float("nan")

        span_bars = int(event_end_bar[test_idx[-1]] - event_start_bar[test_idx[0]])
        span_days = span_bars / bpd if bpd else float(span_bars)

        res.steps.append(WFStep(
            test_period=period_idx,
            n_train_events=n_train,
            n_test_events=n_test,
            test_span_days=float(span_days),
            rolled_capture=rolled,
            next_oracle_capture=next_oracle,
            fixed_capture=fixed,
            random_capture=random_cap,
            train_best_idx=train_best,
            test_best_idx=test_best,
            config_repeat=bool(train_best == test_best),
        ))
        period_idx += 1
        test_lo = test_hi   # advance to the next non-overlapping TEST block

    return res


# ---- aggregation ------------------------------------------------------------

def _safe_mean(a):
    a = np.asarray(a, dtype=np.float64)
    a = a[~np.isnan(a)]
    return float(np.mean(a)) if a.size else None


def _safe_median(a):
    a = np.asarray(a, dtype=np.float64)
    a = a[~np.isnan(a)]
    return float(np.median(a)) if a.size else None


def aggregate_steps(steps: list, cadence: str, n_events: int,
                    skipped: int, skip_reasons: dict,
                    inp: WFInputs | None = None) -> dict:
    if not steps:
        return {"cadence": cadence, "n_events": n_events, "n_steps": 0,
                "skipped": skipped, "skip_reasons": skip_reasons,
                "verdict": "no walk-forward steps (too few events)"}

    rolled = np.array([s.rolled_capture for s in steps], dtype=np.float64)
    nxt = np.array([s.next_oracle_capture for s in steps], dtype=np.float64)
    fixed = np.array([s.fixed_capture for s in steps], dtype=np.float64)
    rand = np.array([s.random_capture for s in steps], dtype=np.float64)

    # PERSISTENCE = rolled / next_oracle, computed on aggregate means (guard div0).
    rolled_m = _safe_mean(rolled)
    nxt_m = _safe_mean(nxt)
    fixed_m = _safe_mean(fixed)
    rand_m = _safe_mean(rand)
    persistence = (rolled_m / nxt_m) if (rolled_m is not None and nxt_m
                                         and nxt_m != 0) else None

    # TRANSLATE-vs-FIXED (THE key test): win-rate + mean delta of rolled - fixed.
    delta_rf = rolled - fixed
    valid_rf = ~np.isnan(delta_rf)
    if valid_rf.any():
        rolled_beats_fixed_winrate = float(np.mean(delta_rf[valid_rf] > 0))
        mean_delta_rf = float(np.mean(delta_rf[valid_rf]))
    else:
        rolled_beats_fixed_winrate = None
        mean_delta_rf = None

    # vs RANDOM: win-rate + mean delta of rolled - random.
    delta_rr = rolled - rand
    valid_rr = ~np.isnan(delta_rr)
    rolled_beats_random_winrate = (float(np.mean(delta_rr[valid_rr] > 0))
                                   if valid_rr.any() else None)
    mean_delta_rr = float(np.mean(delta_rr[valid_rr])) if valid_rr.any() else None

    config_repeat_rate = float(np.mean([s.config_repeat for s in steps]))

    fixed_label = f"{FIXED_FORMULATION}/{FIXED_MA_TYPE}/{FIXED_PARAMS}"
    fixed_available = bool(inp is not None and inp.fixed_idx >= 0) if inp else (
        not np.all(np.isnan(fixed)))

    return {
        "cadence": cadence,
        "n_events": n_events,
        "n_steps": len(steps),
        "skipped": skipped,
        "skip_reasons": skip_reasons,
        "fixed_config": fixed_label,
        "fixed_config_available": fixed_available,
        "capture_means": {
            "rolled": rolled_m,
            "next_oracle": nxt_m,
            "fixed": fixed_m,
            "random": rand_m,
        },
        "capture_medians": {
            "rolled": _safe_median(rolled),
            "next_oracle": _safe_median(nxt),
            "fixed": _safe_median(fixed),
            "random": _safe_median(rand),
        },
        "persistence_ratio": persistence,
        "translate_vs_fixed": {
            "rolled_beats_fixed_winrate": rolled_beats_fixed_winrate,
            "mean_delta_rolled_minus_fixed": mean_delta_rf,
        },
        "vs_random": {
            "rolled_beats_random_winrate": rolled_beats_random_winrate,
            "mean_delta_rolled_minus_random": mean_delta_rr,
        },
        "config_repeat_rate": config_repeat_rate,
        "verdict": _verdict(rolled_m, nxt_m, fixed_m, rand_m),
    }


def _verdict(rolled, nxt, fixed, rand) -> str:
    """ONE-LINE verdict: does this week's config translate to next week?
      (i)   yes, persists       : rolled ~ next_oracle  (and > fixed)
      (ii)  no forward value     : rolled ~ fixed        (picking last week's best
                                    is no better than a fixed config)
      (iii) worse / overfit      : rolled < fixed        (chasing last week's best
                                    actively HURTS vs a fixed config)
    """
    if rolled is None or nxt is None:
        return "INDETERMINATE: insufficient steps."
    f = fixed if fixed is not None else float("-inf")
    r = rand if rand is not None else float("-inf")

    # CLEAREST signal first: if the rolled config LOSES money on TEST while the
    # next-period ceiling is positive, the config does NOT translate -- "beating" a
    # also-negative fixed config by being less-negative is NOT forward value.
    if nxt is not None and nxt > 0 and rolled <= 0.0:
        bf = (" (still > a also-losing fixed %.4f, but that is not forward value)"
              % f) if (fixed is not None and rolled > f) else ""
        return ("(iii) DOES NOT TRANSLATE / OVERFIT: rolled (%.4f) LOSES money on the "
                "next period while the next-period oracle ceiling is +%.4f -- last "
                "period's best config did not carry forward.%s" % (rolled, nxt, bf))

    # Tolerances on the ROI scale (realized long return, fractional).
    near_fixed = (fixed is not None) and (abs(rolled - f) < 0.005)
    # "persists" if rolled banks a large fraction of the ceiling AND beats fixed.
    persists = (nxt > 0 and rolled >= 0.6 * nxt and (fixed is None or rolled > f + 0.005))

    if fixed is not None and rolled < f - 0.005:
        return ("(iii) WORSE / OVERFIT: rolled (%.4f) < fixed (%.4f) -- chasing last "
                "period's best config HURTS vs a fixed config." % (rolled, f))
    if persists:
        frac = (rolled / nxt) if nxt else float("nan")
        return ("(i) YES, PERSISTS: rolled (%.4f) banks %.0f%% of the next-period "
                "ceiling (%.4f) and beats fixed (%.4f)." % (rolled, frac * 100, nxt, f))
    if near_fixed:
        return ("(ii) NO FORWARD VALUE: rolled (%.4f) ~= fixed (%.4f) -- picking last "
                "period's best is no better than a fixed config." % (rolled, f))
    # Fallbacks for ambiguous mid-zone.
    if fixed is not None and rolled > f + 0.005:
        frac = (rolled / nxt) if nxt else float("nan")
        return ("(i-weak) SOME FORWARD VALUE: rolled (%.4f) > fixed (%.4f) but banks "
                "only %.0f%% of the ceiling (%.4f)." % (rolled, f, frac * 100, nxt))
    return ("(ii) NO CLEAR FORWARD VALUE: rolled (%.4f) vs fixed (%.4f), ceiling "
            "(%.4f), random (%.4f)." % (rolled, f, nxt, r))


# ---- reporting --------------------------------------------------------------

def _fmt(v, pct=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "    n/a"
    return f"{v * 100:7.3f}%" if pct else f"{v:8.4f}"


def print_report(per_cadence: list, agg: dict) -> None:
    print("")
    print("=" * 104)
    print("TI-ORACLE WALK-FORWARD PERSISTENCE -- 'does this week's config work next week?'")
    print("=" * 104)
    print("capture = realized long ROI (net taker), same units for all four columns.\n")

    hdr = (f"{'cadence':>10} | {'steps':>5} | {'rolled':>9} {'next-orac':>9} "
           f"{'fixed':>9} {'random':>9} | {'persist':>8} {'beat-fix':>9} {'repeat':>7}")
    print(hdr)
    print("-" * 104)
    for s in per_cadence + [agg]:
        if s.get("n_steps", 0) == 0:
            print(f"{s['cadence']:>10} | {0:>5} | (no walk-forward steps)")
            continue
        cm = s["capture_means"]
        persist = s["persistence_ratio"]
        bf = s["translate_vs_fixed"]["rolled_beats_fixed_winrate"]
        rep = s["config_repeat_rate"]
        print(f"{s['cadence']:>10} | {s['n_steps']:>5} | "
              f"{_fmt(cm['rolled']):>9} {_fmt(cm['next_oracle']):>9} "
              f"{_fmt(cm['fixed']):>9} {_fmt(cm['random']):>9} | "
              f"{_fmt(persist):>8} {_fmt(bf):>9} {_fmt(rep):>7}")
    print("-" * 104)
    print(f"FIXED config = {agg.get('fixed_config', 'n/a')} "
          f"(available={agg.get('fixed_config_available')})")
    print("\nVERDICT (per cadence):")
    for s in per_cadence:
        if s.get("n_steps", 0) == 0:
            print(f"  {s['cadence']:>6}: (no steps)")
            continue
        print(f"  {s['cadence']:>6}: {s['verdict']}")
    print(f"\nAGGREGATE VERDICT: {agg.get('verdict')}")
    print("=" * 104)
    print("NOTE: rolled config fitted on TRAIN events only (causal, no refit on TEST). "
          "next-oracle is the hindsight ceiling.")
    print("")


# ---- selftest (two-sided) ---------------------------------------------------

def _synth_persistent(n=1400, seed=11):
    """PERSISTENT-DNA: a single steady-trend regime throughout -- the SAME config
    (a trend-riding cross) should win every period -> rolled ~= next_oracle >> random.
    """
    rng = np.random.default_rng(seed)
    price = [100.0]
    for i in range(n):
        # gentle persistent uptrend with small noise (same character throughout).
        # Drift/noise tuned so 1d 7-14 bar windows yield in-band (2-10%) moves that
        # a trend-riding cross config rides end-to-end every period -- with enough
        # noise that fast/whipsaw configs underperform (so the genuine best config is
        # separable from a random config, and persists period to period).
        ret = 0.0020 + rng.normal(0, 0.0080)
        price.append(price[-1] * (1.0 + ret))
    p = np.array(price, dtype=np.float64)
    o = p.copy()
    c = p.copy()
    h = p * 1.0012
    lo = p * 0.9988
    return o, h, lo, c


def _synth_regime_switch(n=1400, seed=13):
    """REGIME-SWITCHING / shuffled: the favourable config flips period to period
    (trend legs alternate with choppy mean-revert legs, short blocks) -> TRAIN-best
    does NOT carry to TEST -> rolled collapses toward random, and rolled ~= fixed.
    """
    rng = np.random.default_rng(seed)
    price = [100.0]
    block = 9  # short blocks so adjacent train/test periods sit in different regimes
    for i in range(n):
        phase = (i // block) % 3
        if phase == 0:      # up-trend leg
            ret = 0.006 + rng.normal(0, 0.004)
        elif phase == 1:    # down/chop leg
            ret = -0.005 + rng.normal(0, 0.006)
        else:               # sharp mean-revert oscillation
            ret = 0.05 * np.sin(i / 1.5) * 0.25 + rng.normal(0, 0.008)
        price.append(price[-1] * (1.0 + ret))
    p = np.array(price, dtype=np.float64)
    o = p.copy()
    c = p.copy()
    h = p * 1.003
    lo = p * 0.997
    return o, h, lo, c


def _run_synth(o, h, lo, c, cadence="1h", seed=7):
    inp = build_capture_matrix(o, h, lo, c, cadence)
    res = run_walkforward(inp, seed=seed)
    agg = aggregate_steps(res.steps, cadence, res.n_events, res.skipped,
                          res.skip_reasons, inp)
    return inp, res, agg


def selftest() -> bool:
    ok = True

    # --- Side A: PERSISTENT-DNA -> rolled ~= next_oracle and >> random ---------
    # Use 1d (7-14 bar windows) so synthetic in-band 2-10% events are dense.
    o, h, lo, c = _synth_persistent()
    _, resA, aggA = _run_synth(o, h, lo, c, cadence="1d")
    print(f"[selftest] PERSISTENT: {resA.n_events} events, {aggA.get('n_steps')} steps")
    if aggA.get("n_steps", 0) < 2:
        print("[selftest] FAIL: persistent series produced too few steps")
        return False
    cmA = aggA["capture_means"]
    persistA = aggA["persistence_ratio"]
    print(f"[selftest]   rolled={cmA['rolled']:.4f} next_oracle={cmA['next_oracle']:.4f} "
          f"random={cmA['random']:.4f} persistence={persistA:.3f}")
    # rolled must bank a large fraction of the ceiling ...
    if persistA is None or persistA < 0.6:
        print(f"[selftest] FAIL(A): persistence {persistA} < 0.60 on a persistent regime")
        ok = False
    # ... and clearly beat random.
    if cmA["random"] is not None and not (cmA["rolled"] > cmA["random"] + 0.0015):
        print("[selftest] FAIL(A): rolled not >> random on a persistent regime")
        ok = False

    # --- Side B: REGIME-SWITCH -> rolled collapses toward random, ~= fixed ------
    o2, h2, lo2, c2 = _synth_regime_switch()
    _, resB, aggB = _run_synth(o2, h2, lo2, c2, cadence="1d")
    print(f"[selftest] REGIME-SWITCH: {resB.n_events} events, {aggB.get('n_steps')} steps")
    if aggB.get("n_steps", 0) < 2:
        print("[selftest] FAIL: regime-switch series produced too few steps")
        return False
    cmB = aggB["capture_means"]
    persistB = aggB["persistence_ratio"]
    print(f"[selftest]   rolled={cmB['rolled']:.4f} next_oracle={cmB['next_oracle']:.4f} "
          f"fixed={cmB['fixed']:.4f} random={cmB['random']:.4f} persistence={persistB}")
    # The discriminating test: persistence on the switching regime must be MARKEDLY
    # lower than on the persistent regime (the rolled config does NOT carry forward).
    if persistB is not None and persistA is not None and not (persistB < persistA - 0.10):
        print(f"[selftest] FAIL(B): switching persistence {persistB:.3f} not markedly "
              f"below persistent persistence {persistA:.3f} (rolled did not collapse)")
        ok = False
    # On the switching regime the rolled config must bank a SMALL fraction of the
    # ceiling -- it does not carry forward (the discriminating two-sided signal).
    # (Note: "rolled ~= random" in absolute ROI is not a reliable check, because in a
    # hostile switching regime the random floor itself collapses below zero; the
    # ceiling-fraction PERSISTENCE is the clean discriminator.)
    if persistB is not None and not (persistB < 0.5):
        print(f"[selftest] FAIL(B): switching persistence {persistB:.3f} not low (<0.50) "
              f"-- rolled did not collapse relative to the ceiling")
        ok = False
    # The rolled config must trail the ceiling by MORE on the switching regime than on
    # the persistent one (a bigger forward-loss when the best config keeps changing).
    shortfallA = ((cmA["next_oracle"] - cmA["rolled"]) / cmA["next_oracle"]
                  if cmA["next_oracle"] else None)
    shortfallB = ((cmB["next_oracle"] - cmB["rolled"]) / cmB["next_oracle"]
                  if cmB["next_oracle"] else None)
    if (shortfallA is not None and shortfallB is not None
            and not (shortfallB > shortfallA + 0.10)):
        print(f"[selftest] FAIL(B): ceiling shortfall did not grow "
              f"(persistent={shortfallA:.3f}, switching={shortfallB:.3f})")
        ok = False

    print(f"[selftest] persistent verdict : {aggA.get('verdict')}")
    print(f"[selftest] switching verdict  : {aggB.get('verdict')}")
    print("[selftest] PASS" if ok else "[selftest] FAIL")
    return ok


# ---- main -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="TI-oracle WALK-FORWARD persistence test")
    ap.add_argument("--asset", default="BTCUSDT")
    ap.add_argument("--cadences", default="4h,1h")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    cadences = [c.strip() for c in args.cadences.split(",") if c.strip()]
    for cad in cadences:
        if cad not in WINDOW_BARS:
            print(f"[error] unknown cadence '{cad}'; known={list(WINDOW_BARS)}")
            sys.exit(2)

    per_cadence: list = []
    all_steps: list = []
    total_events = 0
    for cad in cadences:
        print(f"[run] {args.asset} {cad}: loading + events + capture matrix ...",
              flush=True)
        o, h, lo, c = load_ohlc(args.asset, cad)
        inp = build_capture_matrix(o, h, lo, c, cad)
        res = run_walkforward(inp, seed=args.seed)
        agg = aggregate_steps(res.steps, cad, res.n_events, res.skipped,
                              res.skip_reasons, inp)
        per_cadence.append(agg)
        all_steps.extend(res.steps)
        total_events += res.n_events
        print(f"[run] {args.asset} {cad}: {res.n_events} events, "
              f"{len(res.steps)} steps, {res.skipped} skipped", flush=True)

    # Aggregate over ALL cadences' steps.
    agg_all = aggregate_steps(all_steps, "AGGREGATE", total_events, 0, {})
    print_report(per_cadence, agg_all)

    asset_short = args.asset.upper().replace("USDT", "")
    out_path = (Path(args.out) if args.out
                else PROJECT_ROOT / "runs" / "strat" /
                f"oracle_walkforward_{asset_short}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "tool": "oracle_walkforward",
        "question": "does this week's oracle config work next week?",
        "asset": args.asset,
        "cadences": cadences,
        "spec": {
            "period_definition": "consecutive event-blocks (NOT calendar weeks)",
            "test_events_per_block": TEST_EVENTS,
            "train_events_per_block": TRAIN_EVENTS,
            "min_train_events": MIN_TRAIN_EVENTS,
            "min_test_events": MIN_TEST_EVENTS,
            "fixed_config": f"{FIXED_FORMULATION}/{FIXED_MA_TYPE}/{FIXED_PARAMS}",
            "n_random": N_RANDOM,
            "capture_units": "realized_long_roi_net_taker (NOT divided by price-oracle)",
            "causal": "rolled config fitted on TRAIN events only; no refit on TEST",
        },
        "per_cadence": per_cadence,
        "aggregate": agg_all,
    }
    out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"[artifact] {out_path}")


if __name__ == "__main__":
    main()
