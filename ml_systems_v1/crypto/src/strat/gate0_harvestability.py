"""src/strat/gate0_harvestability.py -- GATE 0: the model-FREE harvestability test.

================================================================================
"DOES EXPLOITABLE STRUCTURE EXIST AT OUR RESOLUTION?" -- the gate that PRECEDES
all agent-building (doc S2.3 of docs/AGENT_LAYER_ARCHITECTURE_2026_06_11.md).
================================================================================

WHY THIS GATE COMES FIRST (the 2-ceiling framing):
    realized_edge <= Ceiling2_signal x Ceiling1_fidelity
A perfect planner over a perfect world-model yields ZERO if Ceiling 2 (the
fundamental signal at our resolution) is zero. Gate 0 measures Ceiling 2 ALONE,
model-FREE on purpose -- so a failed agent can NEVER be blamed on the agent
(a fancy A1/A2 over a Ceiling-2-null resolution is the single most expensive
mistake the project can make; build nothing until this passes).

WHAT IT TESTS (per S2.3), at a given (asset, cadence):
  PRIMARY (harvestability / oracle-gap):
    Build the HINDSIGHT oracle on the multi-candle MOVE (the SETUP unit, NOT a
    per-bar prediction). Then ask: is there a PAST-ONLY conditioner that raises
    realized capture above a REGIME-MATCHED + COST-MATCHED RANDOM-ENTRY null,
      * held-out (the conditioner is fit on TRAIN/VAL only; judged on OOS/UNSEEN),
      * >= 8/10 seeds positive,
      * OOS -> UNSEEN persistent (the held-out margin does not collapse)?
    Oracle ceiling HIGH but NO conditioner narrows the gap to the null => the
    structure is NOT EXTRACTABLE at this resolution (the D45 finding).
  NEGATIVE existence diagnostic (one-sided, IC banned as a primary metric):
    A SHUFFLED-CONTROLLED IC-AT-THE-MOVE-SCALE. IC is BANNED as an objective
    (MEMORY.md founding framing: the unit is a SETUP across a MOVE, not a bar);
    but as a *negative* existence test it is decisive -- a move-scale predictive
    correlation that is indistinguishable from its OWN label-shuffled control
    => existence REFUTED at this resolution. It can only REFUTE, never confirm
    (hence one_sided=True on the output).

VERDICT: a clear  EXISTS / REFUTED  + the numbers (held-out null margin, seed
pass-count, OOS->UNSEEN persistence, shuffled-controlled move-IC + its band).

TWO-SIDEDNESS (a gate must be able to return BOTH outcomes, else it is useless):
    - It MUST be able to say EXISTS  -> validated on a SYNTHETIC series WITH
      injected regime structure (the conditioner genuinely beats random entry).
    - It MUST be able to say REFUTED  -> validated on a RANDOM WALK (no structure;
      the best in-sample conditioner does NOT persist out-of-sample over the null).
    `python src/strat/gate0_harvestability.py` runs both and asserts the pair.

HONEST PRIOR: daily/4h LO crypto is LIKELY  REFUTED  per the HARD dead-list
(D17 cross-sectional IC~0 across 6 architectures; D44 1-5%/day needs IC~0.6,
measured ~0 six ways; D45 entry-timing info lives BELOW the bar). The Gate-0
NULL is itself a HIGH-VALUE deliverable: it bounds Ceiling 2 and redirects
compute to the only lever that can lift it (resolution/data: 1m+liq / tick /
LOB -- D71/D72). But the gate MUST still prove it can detect EXISTS where
structure is genuinely present -- otherwise a REFUTED is just a dead gate.

--------------------------------------------------------------------------------
WHAT THIS REUSES (does not reinvent):
  * src/strat/synthetic_positive_control.py
      - within_window_null  : the REGIME+COST+HORIZON-matched random-entry null
        (membership held constant; randomize only the entry inside each move) --
        the EXACT fair null S2.3 asks for, already two-sided-calibrated there.
      - _compound / _detect : the compound + one-sided p95 detection primitive.
  * src/strat/firewall.py  : random_entry_null -- the SAME cost-matched random-
        ENTRY null principle, on a real CanonicalHarness (used by run_on_chimera
        for the live (asset,cadence) slice once V1.1 lands / data is on disk).
  * src/strat/positive_control.py : the GENUINE past-only SMA-crossover TIMING
        edge construction (regime-switching close; crossover catches the regime)
        -- reused as the EXISTS validation substrate.
  * src/oracle/engine.py   : capture_rate semantics (realized/perfect-entry over
        the multi-candle move) -- mirrored here, model-free, on a plain series so
        Gate 0 needs no chimera on disk for its LOGIC test; run_on_chimera()
        delegates to the real OracleEngine + AdaptiveChooser when data is present.
  * src/oracle/adaptive.py : the PAST-ONLY conditioner principle (pick by past-
        only rolling-validity / in-position-at-D, never by realized capture).

PROJECT INVARIANTS honored:
  * TAKER cost 0.0024 round-trip is the gate value (candidate_gate.TAKER_COST_RT);
    maker is a labeled sensitivity only, never the headline.
  * SETUP / MULTI-CANDLE MOVE is the unit -- NOT per-bar. IC appears ONLY as a
    one-sided NEGATIVE diagnostic at the MOVE scale (never an objective).
  * PAST-ONLY conditioner; walk-forward held-out (the conditioner sees TRAIN/VAL,
    is judged on OOS/UNSEEN); no look-ahead in the conditioner or the oracle's
    per-config signal (only the best-config SELECTION is the allowed hindsight).
  * Shuffled-controlled (ShIC-style) negative control, not raw IC -- consistent
    with the project's anti-memorization 'ShIC-not-IC' discipline.
  * ASCII only (no emoji -- cp1252); top-of-file __contract__.

RWYB: `python src/strat/gate0_harvestability.py`  (exit 0 == two-sided pair holds:
EXISTS on injected structure AND REFUTED on a random walk).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# REUSE the matched-null + detection primitives (do NOT duplicate them).
_THIS = Path(__file__).resolve()
_SRC = _THIS.parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
try:  # package import (normal) vs script run (python src/strat/gate0_harvestability.py)
    from .synthetic_positive_control import within_window_null, _compound, _detect
    from .candidate_gate import TAKER_COST_RT
except ImportError:  # pragma: no cover - script-run path
    from strat.synthetic_positive_control import within_window_null, _compound, _detect
    from strat.candidate_gate import TAKER_COST_RT


__contract__ = {
    "kind": "gate0_harvestability_test",
    "inputs": [
        "a per-window MULTI-CANDLE-MOVE price world (synthetic for the LOGIC test; "
        "real chimera (asset,cadence) slice via run_on_chimera when data is on disk)",
        "strat.synthetic_positive_control.within_window_null (regime+cost+horizon-"
        "matched random-ENTRY null) + _compound/_detect",
        "strat.candidate_gate.TAKER_COST_RT (0.0024 spot taker round-trip)",
        "oracle.engine.OracleEngine + oracle.adaptive.AdaptiveChooser (delegated by "
        "run_on_chimera for the live slice; not needed for the model-free LOGIC test)",
    ],
    "outputs": {
        "callable": "gate0(world, *, n_seeds, n_books, cost, alpha) -> dict with "
                    "verdict in {EXISTS, REFUTED} + numbers + an oracle_ceiling block "
                    "(Ceiling-2 hindsight capture, random floor, conditioner capture, "
                    "gap_closed_fraction)",
        "two_sided_rwyb": "main() runs EXISTS (injected structure) AND REFUTED "
                          "(random walk) and asserts the pair",
    },
    "invariants": [
        "MODEL-FREE: no neural net; a failed agent can never be blamed on the gate",
        "unit is the SETUP across a MULTI-CANDLE MOVE, never a per-bar prediction",
        "PRIMARY test = past-only conditioner capture vs REGIME+COST-matched random-"
        "ENTRY null, held-out, >=8/10 seeds, OOS->UNSEEN persistent",
        "OOS->UNSEEN persistence is NON-tautological: UNSEEN median margin must retain "
        ">= persistence_frac of the OOS median margin (a bare '>0' just re-states seed_ok)",
        "the oracle CEILING (Ceiling-2, model-free hindsight best-entry capture) IS reported "
        "with the gap_closed_fraction (how much of oracle-over-random headroom is closed past-only) "
        "-- the oracle-GAP framing the gate exists to measure (high ceiling + ~0 gap_closed = D45)",
        "the conditioner is PAST-ONLY and fit on TRAIN/VAL only; judged on OOS/UNSEEN",
        "oracle is hindsight ONLY in best-config selection; per-config signal is causal",
        "capture_rate = realized_capture / perfect-entry capture over the move, in [0,1]",
        "NEGATIVE diagnostic = SHUFFLED-controlled IC at the MOVE scale (one-sided; "
        "can only REFUTE existence, never confirm it); IC is BANNED as an objective",
        "TAKER cost 0.0024 is the gate value; maker is a labeled sensitivity only",
        "two-sided: gate0 returns EXISTS on injected structure AND REFUTED on a "
        "random walk (validated in main())",
        "no emoji in prints (cp1252)",
    ],
}

GATE0_LABEL = ("GATE 0 -- model-FREE harvestability: does exploitable structure "
               "exist at this resolution? (precedes ALL agent-building)")


# ==========================================================================
# 1. THE MULTI-CANDLE-MOVE WORLD (the SETUP unit; synthetic for the logic test)
# ==========================================================================
def make_move_world(rng, *, W=48, L=24, structure=True,
                    trend_drift=0.014, noise=0.006, cost=TAKER_COST_RT):
    """Build a world of W independent MULTI-CANDLE MOVES (windows), each L bars.

    This is the SETUP-across-a-MOVE unit (NOT per-bar). Two flavours, switched by
    `structure`, give the gate its two-sidedness:

      structure=True (INJECTED STRUCTURE -> the gate should return EXISTS):
        each move has a hidden MOMENTUM-CONTINUATION onset at a RANDOM bar: the
        move is flat noise (zero drift) until the onset, then a SUSTAINED up-trend
        (`trend_drift`) to the close. This is genuine, past-only-detectable
        structure -- once the up-trend has started, observed up-momentum PREDICTS
        the continued up-leg (momentum continuation, the exact lever the dead-list
        says is absent at daily/4h crypto but we INJECT here to prove gate power).
        A PAST-ONLY momentum trigger fires AFTER the onset and rides the
        continuation to the close; a RANDOM entry is uniform over the whole move,
        so it often lands in the flat pre-onset stretch (or after most of the move)
        and captures less -> the conditioner genuinely beats the matched random-
        entry null. The onset bar is random per move, so the conditioner must READ
        momentum, not memorize a fixed position. (Mirrors the positive_control.py
        'regime the crossover catches' construction, framed per-move.)

      structure=False (RANDOM WALK -> the gate should return REFUTED):
        each move is a driftless random walk (i.i.d. zero-mean bar returns). The
        within-move best entry is pure hindsight noise; NO past-only signal beats a
        random entry out-of-sample. There is NO extractable structure -- continued
        momentum is absent (a random walk has no continuation to ride).

    Returns a dict with, per window w:
      bars[w]       : (L,) per-bar simple returns (the move's path).
      price[w]      : (L+1,) price path, price[0]=1.0 (for oracle capture math).
      The world also carries L, W, cost, and a TRAIN/VAL/OOS/UNSEEN split mask
      (50/20/20/10 over the W windows; UNSEEN reserved -- judged once, never fit).
    """
    bars = []
    prices = []
    for _w in range(W):
        if structure:
            # random onset in the first ~60% of the move so there is always a
            # ridable continuation leg AFTER a past-only trigger can fire.
            onset = int(rng.integers(2, max(3, int(0.6 * L) + 1)))
            drift = np.where(np.arange(L) >= onset, trend_drift, 0.0)
            b = drift + rng.normal(0.0, noise, L)
        else:
            b = rng.normal(0.0, noise, L)  # driftless random walk -- no structure
        p = np.concatenate([[1.0], np.cumprod(1.0 + b)])
        bars.append(b)
        prices.append(p)
    # 50/20/20/10 split over windows (UNSEEN reserved; mirrors the project split).
    n = W
    i_tr, i_va, i_oos = int(0.5 * n), int(0.7 * n), int(0.9 * n)
    split = (["TRAIN"] * i_tr + ["VAL"] * (i_va - i_tr) +
             ["OOS"] * (i_oos - i_va) + ["UNSEEN"] * (n - i_oos))
    return {"bars": bars, "price": prices, "W": W, "L": L, "cost": float(cost),
            "split": split, "structure": bool(structure)}


# ==========================================================================
# 2. HINDSIGHT ORACLE on the MOVE  (capture-rate ceiling; mirrors engine.py)
# ==========================================================================
def _move_oracle_capture(price, cost):
    """Hindsight upper bound on a single move: capture of the BEST entry, and the
    perfect-entry denominator. Mirrors oracle.engine capture_rate semantics
    (realized / perfect, clipped to [0,1]) on a plain price path.

    Entry at bar i, exit at the move end (the SETUP is held to the move's close --
    a fixed, no-skill exit so the test isolates ENTRY structure, not exit timing,
    consistent with the exit-axis-NULL lesson). best_capture = max over i of
    (price[-1]/price[i] - 1 - cost); perfect = best over the whole move.
    """
    p = np.asarray(price, float)
    end = p[-1]
    elig = p[:-1]                      # any entry bar before the close
    caps = end / elig - 1.0 - cost     # net (taker) capture of entering at each bar
    best = float(np.max(caps)) if caps.size else 0.0
    return best


# ==========================================================================
# 3. PAST-ONLY CONDITIONER  (the adaptive-chooser principle; no look-ahead)
# ==========================================================================
def _conditioner_entry(bars, *, lookback=3, thresh=0.0):
    """PAST-ONLY entry index inside a move: the FIRST bar i where the trailing
    `lookback`-bar return (close-to-close, ending at i-1, i.e. ONLY past bars)
    exceeds `thresh` -- an observed up-momentum trigger.

    This is the realizable analog the adaptive chooser uses: a past-only signal,
    no future leak. Returns the entry index i (>=1) or None (no trigger fired).
    The entry FILLS at the trigger bar i (we hold from i to the move close).
    """
    b = np.asarray(bars, float)
    L = b.size
    csum = np.concatenate([[1.0], np.cumprod(1.0 + b)])   # csum[k] = price after k bars
    for i in range(1, L):
        lo = max(0, i - lookback)
        trail = csum[i] / csum[lo] - 1.0                  # trailing return ending at bar i-1->i (past-only)
        if trail > thresh:
            return i
    return None


def _conditioner_capture(price, bars, cost, *, lookback, thresh):
    """Realized (past-only) capture of the conditioner on one move: enter at the
    past-only trigger, hold to the move close, net of taker cost. None if no
    trigger fired (the setup correctly ABSTAINS -- not a forced trade)."""
    i = _conditioner_entry(bars, lookback=lookback, thresh=thresh)
    if i is None:
        return None
    p = np.asarray(price, float)
    return float(p[-1] / p[i] - 1.0 - cost)


def _fit_conditioner(world, train_val_idx, cost, grid=None):
    """PAST-ONLY conditioner fit on TRAIN+VAL ONLY: pick the (lookback, thresh)
    config with the best mean per-move capture over the in-sample windows. The
    chosen config is then frozen and judged on OOS/UNSEEN. No held-out data is
    touched in the fit (no look-ahead across the split)."""
    if grid is None:
        grid = [(lb, th) for lb in (2, 3, 5) for th in (0.0, 0.005, 0.01)]
    best_cfg, best_score = grid[0], -1e18
    for (lb, th) in grid:
        caps = []
        for w in train_val_idx:
            c = _conditioner_capture(world["price"][w], world["bars"][w], cost,
                                     lookback=lb, thresh=th)
            if c is not None:
                caps.append(c)
        score = float(np.mean(caps)) if caps else -1e18
        if score > best_score:
            best_score, best_cfg = score, (lb, th)
    return {"lookback": best_cfg[0], "thresh": best_cfg[1], "in_sample_mean": best_score}


# ==========================================================================
# 4. NEGATIVE DIAGNOSTIC -- shuffled-controlled IC AT THE MOVE SCALE (one-sided)
# ==========================================================================
def shuffled_move_ic(world, *, n_shuffle=200, seed=0):
    """One-sided NEGATIVE existence diagnostic: a SHUFFLED-CONTROLLED IC at the
    MOVE scale. IC is BANNED as a primary/objective metric (the unit is a SETUP
    across a move, not a bar) -- here it is used ONLY to REFUTE existence.

    move-IC = Spearman-rank corr between a PAST-ONLY pre-move momentum feature
    (the trailing return of the FIRST half of each move) and the REALIZED forward
    capture of the SECOND half (entering at the half, holding to the close). This
    is a MOVE-scale (not bar-scale) predictive correlation.

    The shuffled control LABEL-PERMUTES the move outcomes (a random permutation of
    the outcome vector -- NOT an FFT phase-shuffle; it destroys the feature<->outcome
    pairing while preserving both marginals exactly) `n_shuffle` times and forms the
    null band. Returned `move_ic_beats_shuffle` is True ONLY if the real
    move-IC exceeds the shuffled p95 -- the MINIMUM bar for 'a move-scale signal
    might exist'. If it does NOT clear the shuffle, existence is REFUTED by this
    diagnostic (one-sided: clearing the shuffle does NOT by itself confirm a
    tradeable, cost-survivable edge -- that is what the PRIMARY null decides)."""
    rng = np.random.default_rng(seed)
    L = world["L"]
    half = L // 2
    feats, outs = [], []
    for w in range(world["W"]):
        p = np.asarray(world["price"][w], float)
        pre = p[half] / p[0] - 1.0                       # past-only first-half momentum
        post = p[-1] / p[half] - 1.0 - world["cost"]     # realized second-half capture
        feats.append(pre)
        outs.append(post)
    feats = np.asarray(feats, float)
    outs = np.asarray(outs, float)

    def _spearman(a, b):
        if a.size < 3:
            return 0.0
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        ra -= ra.mean(); rb -= rb.mean()
        d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
        return float((ra * rb).sum() / d) if d > 0 else 0.0

    real_ic = _spearman(feats, outs)
    null = np.empty(n_shuffle)
    for k in range(n_shuffle):
        null[k] = _spearman(feats, rng.permutation(outs))
    p95 = float(np.percentile(null, 95))
    return {"move_ic": round(real_ic, 4), "shuffle_p95": round(p95, 4),
            "shuffle_p50": round(float(np.percentile(null, 50)), 4),
            "move_ic_beats_shuffle": bool(real_ic > p95),
            "n_shuffle": int(n_shuffle), "one_sided": True}


# ==========================================================================
# 5. THE GATE -- per-seed held-out conditioner-vs-null + OOS->UNSEEN persistence
# ==========================================================================
def _held_out_margin(world, cfg, cost, held_idx, *, n_books, alpha, seed):
    """For a FROZEN conditioner cfg and a held-out window set, compute:
      real_compound  : the conditioner's compound over the held-out moves it is
                       present in (entering at its past-only trigger; abstaining
                       where no trigger fires -- membership = the moves it trades),
      null p50/p95   : the REGIME+COST+HORIZON-matched random-ENTRY null over the
                       SAME moves (membership held constant; randomize the entry
                       inside each move) -- REUSING within_window_null,
      beats_null     : real_compound > null p95 (one-sided detection, level alpha).
    Returns (beats_null, real_compound, null_p50, null_p95, n_trades).
    """
    rng = np.random.default_rng(seed)
    lb, th = cfg["lookback"], cfg["thresh"]
    member_windows, real_caps = [], []
    # Build a per-move 'eligible-entry net return' table for the matched null
    # (the within_window_null contract: elig_rets[w] = net returns of every
    # eligible entry in move w). For the moves the conditioner is PRESENT in.
    elig_by_w = []
    for w in held_idx:
        c = _conditioner_capture(world["price"][w], world["bars"][w], cost,
                                 lookback=lb, thresh=th)
        if c is None:
            continue                       # ABSTAIN: not present in this move
        member_windows.append(w)
        real_caps.append(c)
        p = np.asarray(world["price"][w], float)
        elig = p[:-1]
        elig_by_w.append(p[-1] / elig - 1.0 - cost)   # net capture of every entry in this move
    n_trades = len(member_windows)
    if n_trades == 0:
        return None, 0.0, None, None, 0
    real_compound = _compound(real_caps)
    # REUSE within_window_null via a minimal world shim carrying elig_rets.
    null_world = {"elig_rets": elig_by_w}
    null_books = within_window_null(
        null_world, list(range(len(elig_by_w))), rng, n_books=n_books)
    beats, thr = _detect(real_compound, null_books, alpha=alpha)
    p50 = float(np.percentile(null_books, 50))
    return bool(beats), float(real_compound), p50, float(thr), n_trades


def gate0(world, *, n_seeds=10, n_books=400, cost=None, alpha=0.05,
          min_seed_pass=8, persistence_tol=1e-4, persistence_frac=0.25):
    """Run GATE 0 on one (asset,cadence) MOVE world. Returns a verdict dict.

    For each of n_seeds:
      1. FIT the past-only conditioner on TRAIN+VAL only (no held-out leak).
      2. Judge it on OOS and on UNSEEN against the regime+cost-matched random-
         ENTRY null (within_window_null), held-out.
      3. A seed PASSES iff the conditioner BEATS the null on BOTH OOS and UNSEEN.
    OOS->UNSEEN PERSISTENCE (NON-TAUTOLOGICAL + SCALE-FAIR collapse test): the median
    (over passing seeds) UNSEEN PER-TRADE margin must clear
        max(persistence_tol, persistence_frac * OOS_per_trade_margin).
    A bare '> 0' on the COMPOUND margin would be TAUTOLOGICAL -- a passing seed already
    has UNSEEN compound margin > 0 (it beat the null p95), so '> 0' just re-states
    seed_ok and detects NO collapse. But comparing COMPOUND margins across OOS (many
    moves) and UNSEEN (few moves) is unfair -- the compound margin scales with move
    count. We therefore use PER-TRADE (geometric) margins -- per_trade =
    (1+compound)^(1/n_trades)-1 for both real and null_p95 -- which ARE comparable
    across window sizes. Requiring the UNSEEN per-trade margin to retain >= persistence_frac
    (default 25%) of the OOS per-trade margin makes an OOS->UNSEEN COLLAPSE (the edge
    present on OOS but evaporating PER-TRADE on the reserved window) actually FAIL. A
    tiny absolute persistence_tol floors the degenerate OOS_per_trade~0 case.

    VERDICT (the PRIMARY is the EXISTS authority, per S2.3):
      EXISTS   iff PRIMARY passes: (seeds_passing >= min_seed_pass) AND
               (OOS->UNSEEN persistent). The conditioner genuinely beats the
               regime+cost-matched random-entry null, held-out and robustly.
      REFUTED  otherwise.
    The shuffled-move-IC is a ONE-SIDED NEGATIVE diagnostic: it is reported and
    CORROBORATES a REFUTED (a move-scale IC indistinguishable from its phase-
    shuffle is independent evidence of no structure), but it NEVER vetoes a passing
    PRIMARY -- clearing the shuffle is neither necessary nor sufficient for a
    tradeable, cost-survivable edge (the PRIMARY null is what decides that).
    """
    cost = float(world["cost"] if cost is None else cost)
    split = np.array(world["split"])
    tr_va = list(np.where((split == "TRAIN") | (split == "VAL"))[0])
    oos = list(np.where(split == "OOS")[0])
    unseen = list(np.where(split == "UNSEEN")[0])

    seed_rows = []
    for s in range(n_seeds):
        # the conditioner FIT is deterministic given the data; the seed varies the
        # NULL sampling (the held-out judgement), the robustness source of variance.
        cfg = _fit_conditioner(world, tr_va, cost)
        oos_beat, oos_real, _, oos_p95, oos_n = _held_out_margin(
            world, cfg, cost, oos, n_books=n_books, alpha=alpha, seed=1000 + s)
        un_beat, un_real, _, un_p95, un_n = _held_out_margin(
            world, cfg, cost, unseen, n_books=n_books, alpha=alpha, seed=2000 + s)
        passed = bool(oos_beat is True and un_beat is True)
        oos_margin = (oos_real - oos_p95) if oos_p95 is not None else None
        un_margin = (un_real - un_p95) if un_p95 is not None else None
        seed_rows.append({
            "seed": s, "cfg": cfg, "passed": passed,
            "oos_real": round(oos_real, 4), "oos_p95": (round(oos_p95, 4) if oos_p95 is not None else None),
            "oos_margin": (round(oos_margin, 4) if oos_margin is not None else None),
            "oos_beats_null": oos_beat, "oos_n_trades": oos_n,
            "unseen_real": round(un_real, 4), "unseen_p95": (round(un_p95, 4) if un_p95 is not None else None),
            "unseen_margin": (round(un_margin, 4) if un_margin is not None else None),
            "unseen_beats_null": un_beat, "unseen_n_trades": un_n,
        })

    seeds_passing = sum(1 for r in seed_rows if r["passed"])
    seed_ok = seeds_passing >= min_seed_pass

    # OOS->UNSEEN PERSISTENCE (NON-TAUTOLOGICAL collapse test, scale-FAIR):
    # the edge must NOT COLLAPSE from OOS to the final, never-fit UNSEEN window. A bare
    # '> 0' on the COMPOUND margin would be tautological (a passing seed already has UNSEEN
    # compound margin > 0). But the COMPOUND margin is NOT comparable across windows --
    # OOS has more moves than UNSEEN, so its compound margin is mechanically larger even
    # for an equally-strong edge. We therefore compare PER-TRADE (geometric) margins, which
    # ARE scale-invariant: per_trade = (1+compound)^(1/n_trades) - 1 for both real and the
    # null p95, and per_trade_margin = geo_real - geo_null_p95. The honest, non-tautological
    # bar: the UNSEEN per-trade margin must retain at least persistence_frac (default 25%) of
    # the OOS per-trade margin -- so a per-bar edge that holds on OOS but evaporates per-trade
    # on UNSEEN FAILS. A tiny absolute persistence_tol floors the degenerate OOS~0 case.
    def _per_trade(compound, n):
        return ((1.0 + compound) ** (1.0 / n) - 1.0) if (n and n > 0) else 0.0

    passing = [r for r in seed_rows if r["passed"]]
    if passing:
        oos_pt, un_pt = [], []
        for r in passing:
            if r["oos_p95"] is not None and r["oos_n_trades"]:
                oos_pt.append(_per_trade(r["oos_real"], r["oos_n_trades"])
                              - _per_trade(r["oos_p95"], r["oos_n_trades"]))
            if r["unseen_p95"] is not None and r["unseen_n_trades"]:
                un_pt.append(_per_trade(r["unseen_real"], r["unseen_n_trades"])
                             - _per_trade(r["unseen_p95"], r["unseen_n_trades"]))
        # reported compound margins (descriptive); the VERDICT uses per-trade margins (scale-fair)
        oos_m = float(np.median([r["oos_margin"] for r in passing if r["oos_margin"] is not None]))
        un_m = float(np.median([r["unseen_margin"] for r in passing if r["unseen_margin"] is not None]))
        oos_pt_m = float(np.median(oos_pt)) if oos_pt else 0.0
        un_pt_m = float(np.median(un_pt)) if un_pt else 0.0
        persistence_bar = max(persistence_tol, persistence_frac * oos_pt_m)
        # NON-tautological + scale-fair: UNSEEN per-trade margin retains a fraction of OOS's.
        persistent = bool(un_pt_m > persistence_bar)
    else:
        oos_m = un_m = oos_pt_m = un_pt_m = None
        persistence_bar = None
        persistent = False

    # ORACLE CEILING (Ceiling-2) + GAP-CLOSED. The whole framing is the ORACLE-GAP:
    # realized_edge <= Ceiling2 x Ceiling1. We report Ceiling-2 (the hindsight best-entry
    # capture the move ADMITS, model-free) over the held-out moves, and how much of the
    # oracle-over-random gap the PAST-ONLY conditioner actually closes:
    #   gap_closed = (conditioner_capture - random_capture) / (oracle_capture - random_capture)
    # A high oracle ceiling with gap_closed ~ 0 is the D45 signature (structure exists in
    # hindsight but is NOT extractable past-only) -- exactly what Gate 0 must surface, not hide.
    held_idx = oos + unseen
    oracle_caps, rand_caps, cond_caps = [], [], []
    cfg_for_ceiling = _fit_conditioner(world, tr_va, cost)
    for w in held_idx:
        price = world["price"][w]
        oracle_caps.append(_move_oracle_capture(price, cost))          # hindsight best-entry (Ceiling-2)
        p = np.asarray(price, float)
        rand_caps.append(float(np.mean(p[-1] / p[:-1] - 1.0 - cost)))  # mean random-entry capture
        cc = _conditioner_capture(price, world["bars"][w], cost,
                                  lookback=cfg_for_ceiling["lookback"],
                                  thresh=cfg_for_ceiling["thresh"])
        if cc is not None:
            cond_caps.append(cc)
    oracle_ceiling = float(np.mean(oracle_caps)) if oracle_caps else 0.0
    random_floor = float(np.mean(rand_caps)) if rand_caps else 0.0
    cond_capture = float(np.mean(cond_caps)) if cond_caps else 0.0
    gap = oracle_ceiling - random_floor
    gap_closed_frac = float((cond_capture - random_floor) / gap) if abs(gap) > 1e-12 else 0.0
    oracle_block = {
        "oracle_ceiling_capture": round(oracle_ceiling, 4),   # Ceiling-2 (model-free hindsight best entry)
        "random_floor_capture": round(random_floor, 4),       # mean random-entry capture (the null floor)
        "conditioner_capture": round(cond_capture, 4),        # the past-only conditioner's realized capture
        "oracle_minus_random_gap": round(gap, 4),             # the extractable-in-hindsight headroom
        "gap_closed_fraction": round(gap_closed_frac, 4),     # how much of that headroom is closed PAST-ONLY
    }

    # NEGATIVE existence diagnostic (one-sided): the shuffled-controlled move-IC.
    # Per S2.3 this is decisive ONLY as an INDEPENDENT route to REFUTED -- a
    # move-scale predictive correlation indistinguishable from its label-shuffle
    # control => existence REFUTED. It is NOT a confirmatory gate: it can NEVER
    # block a PRIMARY pass (clearing the shuffle is neither necessary nor
    # sufficient for a tradeable, cost-survivable edge -- the PRIMARY null decides
    # that). So it only fires when the PRIMARY ALSO fails to find structure, where
    # it corroborates the REFUTED with a second, model-free line of evidence.
    neg = shuffled_move_ic(world, n_shuffle=200, seed=0)
    neg_clears_shuffle = bool(neg["move_ic_beats_shuffle"])

    # PRIMARY is the EXISTS authority (S2.3): a past-only conditioner beats the
    # regime+cost-matched random-entry null, held-out, >= min_seed_pass/10 seeds,
    # OOS->UNSEEN persistent. The negative diagnostic does not veto a primary pass.
    primary_pass = bool(seed_ok and persistent)
    exists = primary_pass

    if exists:
        verdict = "EXISTS"
        corro = ("CORROBORATED by the move-scale IC clearing its shuffle control"
                 if neg_clears_shuffle else
                 "(the one-sided move-IC diagnostic is weak/null but does NOT veto "
                 "a passing PRIMARY -- it can only route to REFUTED independently)")
        reason = (f"PRIMARY PASS: a past-only conditioner beats the regime+cost-"
                  f"matched random-entry null on {seeds_passing}/{n_seeds} seeds "
                  f"(>= {min_seed_pass}) and is OOS->UNSEEN persistent (UNSEEN per-trade "
                  f"margin {round(un_pt_m, 5) if un_pt_m is not None else None} >= "
                  f"{round(persistence_frac, 2)} x OOS per-trade "
                  f"{round(oos_pt_m, 5) if oos_pt_m is not None else None}). " + corro)
    else:
        verdict = "REFUTED"
        bits = []
        if not seed_ok:
            bits.append(f"PRIMARY: seeds_passing {seeds_passing}/{n_seeds} < {min_seed_pass}")
        if not persistent:
            bits.append("PRIMARY: OOS->UNSEEN margin not persistent (held-out edge "
                        "collapses / does not stay positive on the reserved window)")
        # the negative diagnostic corroborates the refusal (independent evidence).
        bits.append("NEGATIVE DIAGNOSTIC: move-scale IC "
                    + ("clears" if neg_clears_shuffle else "does NOT clear")
                    + " its label-shuffle control"
                    + ("" if neg_clears_shuffle else " (corroborates REFUTED)"))
        reason = "no extractable structure at this resolution: " + "; ".join(bits)

    return {
        "verdict": verdict,
        "reason": reason,
        "seeds_passing": seeds_passing,
        "n_seeds": n_seeds,
        "min_seed_pass": min_seed_pass,
        "oos_to_unseen_persistent": persistent,
        "oos_median_margin": (round(float(oos_m), 4) if oos_m is not None else None),
        "unseen_median_margin": (round(float(un_m), 4) if un_m is not None else None),
        "oos_per_trade_margin": (round(float(oos_pt_m), 5) if oos_pt_m is not None else None),
        "unseen_per_trade_margin": (round(float(un_pt_m), 5) if un_pt_m is not None else None),
        "persistence_bar": (round(float(persistence_bar), 5) if persistence_bar is not None else None),
        "persistence_frac": persistence_frac,
        "oracle_ceiling": oracle_block,
        "negative_diagnostic": neg,
        "cost_rt": cost,
        "cost_basis": "TAKER 0.0024 (gate value); maker is a labeled sensitivity only",
        "per_seed": seed_rows,
        "n_books": n_books,
        "alpha": alpha,
    }


# ==========================================================================
# 6. LIVE-SLICE DELEGATION (real chimera; runs when V1.1 lands / data on disk)
# ==========================================================================
def run_on_chimera(asset, cadence, *, dates=None, universe="u50",
                   lookback_days=30, n_books=300, seed=7, verbose=True):
    """Run Gate 0 on a REAL (asset, cadence) chimera slice (read-only).

    This is the production entry: it COMPOSES the existing apparatus rather than
    re-implementing it --
      * oracle.engine.OracleEngine        -> the hindsight capture ceiling on the
                                             multi-candle move (the oracle-gap top),
      * oracle.adaptive.AdaptiveChooser    -> the PAST-ONLY conditioner's picks,
      * strat.firewall.random_entry_null   -> the cost-matched random-ENTRY null
                                             (membership/regime-matched) on a real
                                             CanonicalHarness.
    It is READ-ONLY (loads chimera, never writes data) and CPU-bound (no GPU; does
    NOT touch any WM checkpoint or training run). Returns a verdict dict mirroring
    gate0(). Deferred to the live run -- the LOGIC is proven two-sided by main()
    on synthetic inputs first.

    NOTE (RWYB-honest): this delegates to chimera-coupled modules; it requires the
    asset's parquet on disk. It is intentionally NOT exercised in the offline
    two-sided self-test (which must run on CPU + tiny synthetic inputs with no data
    dependency). Call it explicitly with a real asset once V1.1 has landed.
    """
    from oracle.engine import OracleEngine          # noqa: F401  (delegation point)
    from oracle.adaptive import AdaptiveChooser      # noqa: F401  (delegation point)
    from strat.firewall import random_entry_null     # noqa: F401  (delegation point)
    raise NotImplementedError(
        "run_on_chimera is the LIVE-slice delegation seam (oracle.engine + "
        "oracle.adaptive + strat.firewall on real chimera). It is deferred to the "
        "post-V1.1 live run by design -- the gate LOGIC is validated two-sided on "
        "synthetic CPU inputs by main(). Wire it to a concrete (asset,cadence) "
        "slice when data is on disk; it must remain READ-ONLY + CPU-only and must "
        "NOT touch any WM checkpoint or compete for GPU.")


# ==========================================================================
# 7. RWYB -- TWO-SIDED demonstration (EXISTS on structure; REFUTED on noise)
# ==========================================================================
def _print_verdict(tag, res):
    print(f"\n--- {tag} ---")
    print(f"  VERDICT           : {res['verdict']}")
    print(f"  reason            : {res['reason']}")
    print(f"  seeds_passing     : {res['seeds_passing']}/{res['n_seeds']} "
          f"(need >= {res['min_seed_pass']})")
    print(f"  OOS median margin : {res['oos_median_margin']}   "
          f"UNSEEN median margin: {res['unseen_median_margin']}   "
          f"persistent={res['oos_to_unseen_persistent']}")
    print(f"  per-trade margin  : OOS={res['oos_per_trade_margin']}  UNSEEN={res['unseen_per_trade_margin']}  "
          f"(persistence bar={res['persistence_bar']}, frac={res['persistence_frac']})")
    oc = res["oracle_ceiling"]
    print(f"  oracle ceiling-2  : {oc['oracle_ceiling_capture']}  random_floor={oc['random_floor_capture']}  "
          f"conditioner={oc['conditioner_capture']}")
    print(f"  oracle-gap closed : {oc['gap_closed_fraction']}  (past-only conditioner closes this "
          f"fraction of the oracle-over-random headroom {oc['oracle_minus_random_gap']})")
    neg = res["negative_diagnostic"]
    print(f"  move-scale IC     : {neg['move_ic']}  shuffle_p95={neg['shuffle_p95']}  "
          f"beats_shuffle={neg['move_ic_beats_shuffle']}  (one-sided negative diagnostic)")
    print(f"  cost basis        : {res['cost_basis']}")


def two_sided_selftest(verbose=True):
    """Validate BOTH directions of the gate on synthetic CPU inputs:
       (A) injected structure  -> the gate MUST return EXISTS,
       (B) random walk         -> the gate MUST return REFUTED.
    Returns (ok, exists_res, refuted_res). A gate that cannot do BOTH is useless."""
    rng_a = np.random.default_rng(20260611)
    world_struct = make_move_world(rng_a, W=48, L=24, structure=True)
    exists_res = gate0(world_struct, n_seeds=10, n_books=400)

    rng_b = np.random.default_rng(777)
    world_noise = make_move_world(rng_b, W=48, L=24, structure=False)
    refuted_res = gate0(world_noise, n_seeds=10, n_books=400)

    if verbose:
        print("=" * 84)
        print(GATE0_LABEL)
        print("TWO-SIDED SELF-TEST (synthetic, CPU, model-free -- no GPU, no chimera, "
              "no WM checkpoint)")
        print("=" * 84)
        _print_verdict("(A) INJECTED STRUCTURE  (must => EXISTS)", exists_res)
        _print_verdict("(B) RANDOM WALK         (must => REFUTED)", refuted_res)

    a_ok = (exists_res["verdict"] == "EXISTS")
    b_ok = (refuted_res["verdict"] == "REFUTED")
    ok = bool(a_ok and b_ok)
    if verbose:
        print("\n" + "=" * 84)
        print(f"  [{'PASS' if a_ok else 'FAIL'}] genuine structure detected as EXISTS")
        print(f"  [{'PASS' if b_ok else 'FAIL'}] random walk rejected as REFUTED")
        print("  TWO-SIDED RESULT  : " +
              ("PASS -- the gate can return BOTH EXISTS and REFUTED (calibrated, "
               "not a one-verdict sieve)."
               if ok else
               "*** FAIL *** the gate cannot distinguish structure from noise -- "
               "it is mis-calibrated."))
        print("=" * 84)
    return ok, exists_res, refuted_res


def main():
    ap = argparse.ArgumentParser(description=GATE0_LABEL)
    ap.add_argument("--selftest", action="store_true",
                    help="run the two-sided synthetic self-test (default)")
    ap.add_argument("--asset", default=None,
                    help="(live) run Gate 0 on a real chimera (asset,cadence) slice")
    ap.add_argument("--cadence", default="1d")
    args = ap.parse_args()

    if args.asset:
        # Live-slice path (deferred seam; raises until wired -- see run_on_chimera).
        run_on_chimera(args.asset, args.cadence)
        return 0

    ok, _, _ = two_sided_selftest(verbose=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
