"""src/strat/within_window_capture_gate.py -- the WITHIN-WINDOW BEST-vs-WORST-SPREAD capture gate
+ its SYNTHETIC POSITIVE CONTROL (refute the over-rejection risk before trusting the gate).

THE UNIT (MEMORY.md founding framing): the unit of trading is a SETUP across a MOVE (multiple candles).
Per-bar IC is the WRONG lens. The RIGHT lens for entry-TIMING skill is: GIVEN a move you are present in,
did your TRIGGER land closer to the BEST-possible entry than random timing inside the same move would?

THE GATE (best-vs-worst-spread capture)
---------------------------------------
For each MOVE m the strategy fires in, the move spans a set of candidate entry bars; each bar t gives a
realized net return ret(t) = exit_open / open[t] - 1 - cost (same declarative exit for every candidate
entry, so only the ENTRY TIMING varies). Define:
    best_m  = max_t ret(t)            # the luckiest entry inside the move (an oracle's pick)
    worst_m = min_t ret(t)            # the unluckiest entry
    spread_m = best_m - worst_m       # the timing-decision range available IN this move
    capture_m = (ret(t*) - worst_m) / spread_m   in [0,1]   # t* = the strategy's actual trigger
capture_m = 1.0 means perfect timing (entered at the best bar), 0.0 means worst, 0.5 is the MIDDLE of the
available range. A timing-skilled strategy has held-out MEAN capture systematically ABOVE a random-timing
null. (Degenerate moves with spread_m < eps carry no timing decision and are dropped.)

NULL (random timing inside the SAME moves): for each move, draw the entry bar uniformly from that move's
candidate bars; the held-out mean capture under random timing is Monte-Carlo'd to a distribution. The gate
PASSES iff the strategy's held-out mean capture exceeds the null p95 (i.e. its timing is better than
chance inside the very moves it trades -- this is move-MEMBERSHIP-matched, so it cannot be rewarded for
merely selecting which move to be in; only for WHERE inside the move it triggers). This mirrors
firewall.random_entry_null(membership_matched=True) but as a clean, bounded [0,1] capture statistic.

THE OVER-REJECTION RISK (what this file refutes)
------------------------------------------------
A gate that REJECTS everything is as useless as one that accepts everything. Before trusting a NULL
result from this gate on real data, we must prove the gate has POWER: inject a setup with a KNOWN
within-window timing skill s (its trigger is engineered to land at capture-fraction ~= s in every move),
run it through the gate, and confirm the gate PASSES it -- and calibrate the smallest known skill the gate
can reliably detect (its detection floor). Two-sided: a no-skill (random-timing) strategy must FAIL.

RWYB:  python src/strat/within_window_capture_gate.py
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

WINDOWS = ["TRAIN", "VAL", "OOS", "UNSEEN"]
HELD = ["OOS", "UNSEEN"]
TAKER = 0.0024


# ---------------------------------------------------------------------------
# Synthetic data: a stream of well-defined MOVES with a learnable best-entry bar.
# ---------------------------------------------------------------------------
@dataclass
class SynthMoves:
    df: pd.DataFrame            # date, open, high, low, close
    moves: list                 # list of dicts: {entry_bars: np.ndarray, exit_bar: int}


def make_moves(seed: int = 7, n_moves: int = 320, win_bars: int = 6, gap_bars: int = 2,
               up_drift: float = 0.05, dip_depth: float = 0.04, noise: float = 0.004,
               start: str = "2022-01-01") -> SynthMoves:
    """Build a daily OHLC stream of back-to-back MOVES, each with a clear within-window timing decision.

    Each move = `win_bars` candidate ENTRY bars followed (after the move) by an EXIT bar. Inside the
    window the OPEN price dips to a low at a known bar then recovers; the post-window exit open is up by
    ~`up_drift`. So entering at the dip (best) captures more of the up-move than entering at the window
    edges (worst): a real best-vs-worst SPREAD exists and is driven purely by ENTRY TIMING (the exit is
    identical for every candidate entry). Noise keeps the spread non-degenerate and the null non-trivial.
    Past-only by construction is irrelevant here -- we score timing skill GIVEN move membership, the
    membership is supplied (the synthetic harness IS the move oracle for control purposes).
    """
    rng = np.random.default_rng(seed)
    opens, highs, lows, closes = [], [], [], []
    moves = []
    level = 100.0
    idx = 0
    for _ in range(n_moves):
        # within-move open path: a dip (U/V shape) reaching its low at a random interior bar
        low_bar = int(rng.integers(1, win_bars - 1))           # where the dip bottoms (interior)
        base = level
        win_opens = []
        for k in range(win_bars):
            # V-shape factor: 0 at the edges-ish, -dip_depth at low_bar
            tri = 1.0 - abs(k - low_bar) / max(low_bar, win_bars - 1 - low_bar)
            factor = -dip_depth * tri
            o = base * (1.0 + factor + rng.normal(0.0, noise))
            win_opens.append(o)
        # exit bar AFTER the window, up by up_drift from the move's base level
        exit_open = base * (1.0 + up_drift + rng.normal(0.0, noise))
        # assemble bars: win_bars entry bars, then `gap_bars` filler, exit bar is the first filler bar
        entry_start = idx
        for k in range(win_bars):
            o = win_opens[k]
            c = o * (1.0 + rng.normal(0.0, noise))
            opens.append(o); closes.append(c)
            highs.append(max(o, c) * (1.0 + abs(rng.normal(0, noise))))
            lows.append(min(o, c) * (1.0 - abs(rng.normal(0, noise))))
            idx += 1
        exit_bar = idx                                          # first bar after the window
        for g in range(gap_bars):
            o = exit_open if g == 0 else exit_open * (1.0 + rng.normal(0.0, noise))
            c = o * (1.0 + rng.normal(0.0, noise))
            opens.append(o); closes.append(c)
            highs.append(max(o, c) * (1.0 + abs(rng.normal(0, noise))))
            lows.append(min(o, c) * (1.0 - abs(rng.normal(0, noise))))
            idx += 1
        moves.append({"entry_bars": np.arange(entry_start, entry_start + win_bars), "exit_bar": exit_bar})
        # next move starts from the realized exit level (compounding stream)
        level = closes[-1]
    n = len(opens)
    dates = pd.date_range(start=start, periods=n, freq="D")
    df = pd.DataFrame({"date": dates, "open": np.array(opens), "high": np.array(highs),
                       "low": np.array(lows), "close": np.array(closes)})
    return SynthMoves(df=df, moves=moves)


def _window_label(i: int, n: int) -> str:
    """Chronological 50/15/15/20 split by bar index (UNSEEN = last 20%)."""
    f = i / n
    if f < 0.50:
        return "TRAIN"
    if f < 0.65:
        return "VAL"
    if f < 0.80:
        return "OOS"
    return "UNSEEN"


# ---------------------------------------------------------------------------
# THE GATE
# ---------------------------------------------------------------------------
def _move_ret_curve(opens: np.ndarray, move: dict, cost: float) -> np.ndarray:
    """ret(t) for every candidate entry bar in the move (same exit open for all -> only timing varies)."""
    exit_open = opens[move["exit_bar"]]
    eo = opens[move["entry_bars"]]
    return exit_open / eo - 1.0 - cost


def capture_gate(synth: SynthMoves, trigger_fn, cost: float = TAKER, n_books: int = 2000,
                 seed: int = 11, min_spread: float = 0.003, accept_pctl: float = 95.0) -> dict:
    """Run the best-vs-worst-spread capture gate.

    trigger_fn(rets, rng) -> index INTO move['entry_bars'] chosen by the strategy for that move.
    Returns held-out mean capture, the random-timing null band, and the PASS/FAIL verdict.
    """
    opens = synth.df["open"].to_numpy(float)
    n = len(opens)
    rng = np.random.default_rng(seed)

    # per-move capture for the strategy + the random-timing null mean (analytic per move), by window
    per_window = {w: {"caps": [], "null_means": [], "ret_curves": []} for w in WINDOWS}
    for m in synth.moves:
        rets = _move_ret_curve(opens, m, cost)
        best, worst = float(rets.max()), float(rets.min())
        spread = best - worst
        if spread < min_spread:
            continue                                           # no timing decision in this move
        w = _window_label(int(m["entry_bars"][0]), n)
        ti = int(trigger_fn(rets, rng))
        cap = (float(rets[ti]) - worst) / spread
        per_window[w]["caps"].append(cap)
        # analytic random-timing mean capture for THIS move (uniform over candidate bars)
        per_window[w]["null_means"].append(float(((rets - worst) / spread).mean()))
        per_window[w]["ret_curves"].append(rets)

    # held-out (OOS+UNSEEN) pooled
    held_caps = np.array([c for w in HELD for c in per_window[w]["caps"]], float)
    held_curves = [rc for w in HELD for rc in per_window[w]["ret_curves"]]
    n_held = len(held_caps)
    real_mean_capture = float(held_caps.mean()) if n_held else float("nan")

    # Monte-Carlo random-timing null: re-pick a uniform entry per held move, n_books times -> dist of
    # the held-out MEAN capture under no timing skill.
    null_book_means = np.empty(n_books)
    worsts = np.array([float(rc.min()) for rc in held_curves])
    spreads = np.array([float(rc.max() - rc.min()) for rc in held_curves])
    sizes = np.array([len(rc) for rc in held_curves])
    for b in range(n_books):
        caps_b = np.empty(n_held)
        for j, rc in enumerate(held_curves):
            ti = rng.integers(0, sizes[j])
            caps_b[j] = (rc[ti] - worsts[j]) / spreads[j]
        null_book_means[b] = caps_b.mean()
    null_p50 = float(np.percentile(null_book_means, 50))
    null_p95 = float(np.percentile(null_book_means, accept_pctl))
    null_mean = float(null_book_means.mean())
    # empirical one-sided p-value of the real mean under the random-timing null
    p_value = float((null_book_means >= real_mean_capture).mean()) if n_held else float("nan")

    passes = bool(n_held > 0 and real_mean_capture > null_p95)
    return {
        "n_held_moves": n_held,
        "real_mean_capture": round(real_mean_capture, 4),
        "null_mean_capture": round(null_mean, 4),
        "null_p50": round(null_p50, 4),
        f"null_p{int(accept_pctl)}": round(null_p95, 4),
        "p_value_vs_null": round(p_value, 4),
        "accept_threshold": round(null_p95, 4),
        "PASSES": passes,
        "per_window_mean_capture": {w: (round(float(np.mean(per_window[w]["caps"])), 4)
                                        if per_window[w]["caps"] else None) for w in WINDOWS},
        "per_window_n_moves": {w: len(per_window[w]["caps"]) for w in WINDOWS},
    }


# ---------------------------------------------------------------------------
# Trigger functions with KNOWN within-window timing skill
# ---------------------------------------------------------------------------
def make_skilled_trigger(target_capture: float):
    """A trigger with KNOWN within-window timing skill: in every move it picks the candidate entry bar
    whose capture-fraction is closest to `target_capture`. So held-out mean capture ~= target_capture by
    construction -> a control with a KNOWN skill level."""
    def trig(rets, rng):
        best, worst = rets.max(), rets.min()
        spread = best - worst
        caps = (rets - worst) / (spread if spread > 0 else 1.0)
        return int(np.argmin(np.abs(caps - target_capture)))
    return trig


def random_trigger(rets, rng):
    """No timing skill: uniform random entry inside the move."""
    return int(rng.integers(0, len(rets)))


# ===========================================================================
# RWYB: positive control + two-sided soundness + calibration sweep
# ===========================================================================
def run_positive_control(verbose: bool = True) -> dict:
    synth = make_moves()
    nmoves = len(synth.moves)

    # (A) POSITIVE CONTROL: a KNOWN skill = 0.80 capture must PASS.
    known_skill = 0.80
    skilled = capture_gate(synth, make_skilled_trigger(known_skill))

    # (B) NEGATIVE CONTROL (two-sided): no timing skill (random) must FAIL.
    noskill = capture_gate(synth, random_trigger)

    # (C) CALIBRATION SWEEP: smallest known skill the gate reliably detects (its detection floor),
    #     and confirm monotone detection -> the gate is not arbitrarily over-rejecting.
    sweep = {}
    for s in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.00]:
        r = capture_gate(synth, make_skilled_trigger(s))
        sweep[s] = {"real_mean_capture": r["real_mean_capture"], "thr": r["accept_threshold"],
                    "PASSES": r["PASSES"], "p": r["p_value_vs_null"]}
    detected = [s for s, r in sweep.items() if r["PASSES"]]
    detection_floor = min(detected) if detected else None

    if verbose:
        import json
        print("=" * 80)
        print("WITHIN-WINDOW BEST-vs-WORST-SPREAD CAPTURE GATE -- synthetic positive control")
        print(f"  synthetic stream: {nmoves} moves, daily bars, taker cost={TAKER}")
        print("=" * 80)
        print(f"\n(A) POSITIVE CONTROL -- KNOWN within-window timing skill = {known_skill:.2f} capture:")
        print(json.dumps(skilled, indent=2, default=str))
        print(f"\n(B) NEGATIVE CONTROL -- NO timing skill (uniform random entry inside each move):")
        print(json.dumps(noskill, indent=2, default=str))
        print("\n(C) CALIBRATION SWEEP -- injected known skill -> gate response:")
        print(f"  {'skill':>6} {'real_cap':>9} {'null_p95':>9} {'p':>7}  PASSES")
        for s, r in sweep.items():
            print(f"  {s:>6.2f} {r['real_mean_capture']:>9.4f} {r['thr']:>9.4f} "
                  f"{r['p']:>7.4f}  {r['PASSES']}")
        print(f"\n  detection floor (smallest known skill the gate PASSES): {detection_floor}")

    # SOUNDNESS verdict (two-sided + calibrated)
    pos_passes = bool(skilled["PASSES"])
    neg_rejected = bool(not noskill["PASSES"])
    monotone = detection_floor is not None and all(
        sweep[s]["PASSES"] for s in sweep if s >= detection_floor)
    calibrated = pos_passes and neg_rejected and monotone

    if verbose:
        print("\n" + "-" * 80)
        print("SOUNDNESS (two-sided + calibrated):")
        print(f"  (A) KNOWN-skill (0.80) strategy PASSES the gate         : {pos_passes}")
        print(f"  (B) NO-skill (random timing) strategy is REJECTED       : {neg_rejected}")
        print(f"  (C) detection is MONOTONE above the floor (no holes)    : {monotone}")
        print(f"      detection floor = {detection_floor} capture")
        verdict = ("PASS -- the gate has POWER (accepts a genuine known within-window timing edge) AND "
                   "discriminates (rejects no-skill). The over-rejection risk is REFUTED: a real edge at "
                   "or above the calibrated detection floor is not rejected.") if calibrated else (
                   "CHECK -- the gate did not behave two-sided/monotone; inspect the flags above.")
        print(f"\n[within_window_capture_gate] {verdict}")

    return {"calibrated": calibrated, "pos_passes": pos_passes, "neg_rejected": neg_rejected,
            "monotone": monotone, "detection_floor": detection_floor,
            "skilled": skilled, "noskill": noskill, "sweep": sweep}


if __name__ == "__main__":
    out = run_positive_control()
    import sys
    sys.exit(0 if out["calibrated"] else 1)
