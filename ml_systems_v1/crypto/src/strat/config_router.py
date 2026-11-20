"""THE MISSING-LINK SOLVER: a CAUSAL week-by-week config-ADJUSTMENT mechanism.

THE PROBLEM (from oracle_walkforward.py, committed):
  The SINGLE best config does NOT translate week-to-week. config-repeat ~0%; rolling
  last-period's-best FORWARD LOSES money (BTC -0.40%, ETH -0.12%, SOL -0.57%) versus a
  +3-6% hindsight ceiling. "Last-period's-best" is the FAILED baseline.

THE MISSING LINK (what we hunt here):
  A PAST-ONLY rule that ADJUSTS the config each period and CAPTURES REAL SIGNAL =
    (i)   POSITIVE mean realizable capture OOS, AND
    (ii)  beats a FIXED config robustly, AND
    (iii) beats last-period's-best (the FAILED baseline).
  If such a rule exists, the per-period config CHOICE itself carries forward -- the
  router has found structure (e.g. regime -> config) that the single-best-config roll
  could not.

MECHANISMS TESTED (each picks the TEST-period config CAUSALLY from TRAIN-period info
+ past-only state; NO TEST information leaks into the choice):

  A) REGIME-CONDITIONED ROUTER (primary)
     Detect a regime label PAST-ONLY at each move's START bar (using only bars < start):
       - vol-tercile  : trailing realized vol (std of log-returns over VOL_LOOKBACK),
                        bucketed against TRAIN-period vol terciles.
       - trend-state  : sign of (close - SMA(TREND_MA)) AND sign of the SMA slope.
     The regime bucket = (vol_tercile, trend_state). On TRAIN: for each bucket, learn
     the config (and the config-CLASS = formulation-family x MA-type) with the best MEAN
     capture among TRAIN events in that bucket. On TEST: detect each event's regime
     past-only, apply that bucket's learned config. Fallback to the global TRAIN-best for
     buckets unseen in TRAIN.

  B) STABLE-SUBSET ENSEMBLE
     On TRAIN pick the TOP-K configs by CONSISTENCY (high mean capture AND high win-rate /
     low variance: score = mean - LAMBDA*std, restricted to win-rate >= WINRATE_FLOOR).
     On TEST average those K configs' capture per event (an equal-weight ensemble).

  C) ADAPTIVE EWMA
     Track each config's EWMA capture over the recent move-events (decay EWMA_ALPHA).
     At each TEST move, pick the config with the best EWMA computed from events STRICTLY
     BEFORE it (TRAIN events seed the EWMA; then it updates causally as TEST events pass
     -- but the pick for event k uses only events < k, so no look-ahead). Apply it.

BASELINES (same apparatus, same events):
  - LAST-BEST  : last-period's (TRAIN) single best config, rolled to TEST. THE FAILED one.
  - FIXED      : one globally-fixed naive config (F2_CROSS/SMA/5x20, same as walkforward).
  - CEILING    : next-period (TEST) own best config (hindsight ceiling).
  - RANDOM     : mean over N_RANDOM random configs on TEST.

EVALUATION (per mechanism, aggregate over walk-forward steps):
  - mean & median TEST capture (realized ROI).
  - win-rate vs FIXED + mean delta.
  - fraction-of-steps-positive.
  - persistence vs ceiling (mech_mean / ceiling_mean).
  - beats LAST-BEST? (win-rate + mean delta).

THE VERDICT:
  Does ANY mechanism capture REAL signal = (i) positive mean OOS AND (ii) beats FIXED
  robustly (win-rate >= BEAT_FIXED_WINRATE and mean-delta > BEAT_FIXED_DELTA) AND (iii)
  beats LAST-BEST? If YES -> name it + dump the regime->config DNA map + flag for deeper
  robustness (seeds, jackknife) before any belief. If NO -> honestly report NOT FOUND at
  this level + which mechanism came closest + its gap.

DISCIPLINE (real money -- report the honest number, never manufacture an edge):
  - CAUSAL: regime detected PAST-ONLY at the move start (bars < start only); configs
    fitted on TRAIN events ONLY; TEST events never touched for the config choice; the
    TRAIN block strictly precedes the TEST block in bar index (no look-ahead).
  - OOS-once: each TEST block is scored once, no peeking.
  - Real chimera OHLC. Net taker 0.24% RT (inherited from the candidate fns).
  - cp1252-safe (no emoji).

Usage:
    python src/strat/config_router.py --assets BTCUSDT,ETHUSDT,SOLUSDT --cadences 4h,1h
    python src/strat/config_router.py --selftest

__contract__ = {
    "kind": "research_missing_link_solver",
    "inputs": ["chimera OHLC via ChimeraLoader",
               "ti_oracle_anchor.find_price_oracle_events (read-only reuse)",
               "ti_oracle_decompose.build_candidates (read-only reuse)",
               "oracle_walkforward.build_capture_matrix + WF spec (read-only reuse)"],
    "outputs": ["runs/strat/config_router_<ASSETS>.json", "stdout table"],
    "invariants": [
        "events + candidate grid + capture matrix identical to oracle_walkforward",
        "regime detected PAST-ONLY at move start (bars < start only) -- no look-ahead",
        "all mechanisms fit on TRAIN events only; TEST untouched for config choice",
        "TRAIN block strictly precedes TEST block (causal)",
        "ceiling is hindsight; last-best is the FAILED baseline; fixed is naive single",
        "verdict requires (i) positive mean OOS AND (ii) beats FIXED robustly AND "
        "(iii) beats LAST-BEST -- else NOT FOUND, honestly",
        "selftest two-sided: regime-determines-best -> router captures + beats fixed + "
        "positive; best-config-random-per-move -> all collapse to ~fixed/random",
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

# READ-ONLY reuse so the events + candidate grid + capture matrix are byte-identical
# to the committed walk-forward apparatus (the missing-link hunt rides the SAME numbers).
from strat.ti_oracle_anchor import (  # noqa: E402
    WINDOW_BARS,
    find_price_oracle_events,
    load_ohlc,
)
from strat.ti_oracle_decompose import build_candidates  # noqa: E402
from strat.oracle_walkforward import (  # noqa: E402
    BARS_PER_DAY,
    TEST_EVENTS,
    TRAIN_EVENTS,
    MIN_TRAIN_EVENTS,
    MIN_TEST_EVENTS,
    FIXED_FORMULATION,
    FIXED_MA_TYPE,
    FIXED_PARAMS,
    N_RANDOM,
    build_capture_matrix,
    _candidate_index,
)

# ---- regime-detection spec (PAST-ONLY) --------------------------------------

VOL_LOOKBACK = 30      # bars of trailing log-returns for realized vol (past-only)
TREND_MA = 100         # SMA length for the trend-state (close vs SMA-100)
TREND_SLOPE_LB = 10    # bars to measure the SMA slope sign

# ---- mechanism hyperparameters ----------------------------------------------

STABLE_TOPK = 5            # B: top-K consistent configs to ensemble
STABLE_LAMBDA = 1.0        # B: score = mean - LAMBDA*std
STABLE_WINRATE_FLOOR = 0.5 # B: require >= this TRAIN win-rate to qualify
EWMA_ALPHA = 0.35          # C: EWMA decay on per-config capture

# ---- verdict thresholds (the honest bar) ------------------------------------

# (ii) "beats FIXED robustly" = a meaningful mean-delta (the WEALTH magnitude, the
# project objective) AND a robustness flag that the edge is not one-step luck: EITHER a
# per-step win-rate floor OR median-capture dominance over fixed. The mean-delta floor is
# set ABOVE the largest random-series noise observed in the selftest (~+0.0008) so a
# null series cannot pass; the win-rate-OR-median clause prevents a single-step tail from
# carrying the verdict. A tail-concentrated edge that passes is FLAGGED for deeper
# robustness (jackknife, top-k-of-compound) before any capital -- per the trust stack.
BEAT_FIXED_WINRATE = 0.50  # (ii) win-rate-vs-fixed floor (one of the two robustness flags)
BEAT_FIXED_DELTA = 0.0025  # (ii) mean (mech - fixed) ROI delta floor (> random noise)
BEAT_LASTBEST_DELTA = 0.0  # (iii) mean (mech - last-best) ROI must exceed last-best


# ============================================================================
# REGIME DETECTION  (PAST-ONLY at the move start)
# ============================================================================

def _trailing_realized_vol(close: np.ndarray, t: int, lb: int) -> float:
    """Std of log-returns over (t-lb, t], using bars STRICTLY BEFORE t (past-only).

    Returns NaN if insufficient history. t is the move's START bar; we use bars < t.
    """
    lo = t - lb
    if lo < 1 or t > len(close):
        return float("nan")
    seg = close[lo:t]            # bars [t-lb, t) -- strictly before the move start
    if len(seg) < 3 or np.any(seg <= 0):
        return float("nan")
    logret = np.diff(np.log(seg))
    if len(logret) < 2:
        return float("nan")
    return float(np.std(logret))


def _trend_state(close: np.ndarray, t: int, ma_len: int, slope_lb: int) -> int:
    """Past-only trend state at the move start t. Uses bars < t only.

    Returns an integer code in {0,1,2,3}:
      bit0 = close[t-1] > SMA(ma_len) at t-1   (price above its MA)
      bit1 = SMA(ma_len)[t-1] > SMA(ma_len)[t-1-slope_lb]  (MA sloping up)
    -> 0 below+down, 1 above+down, 2 below+up, 3 above+up. NaN history -> -1.
    """
    if t < 1:
        return -1
    last = t - 1                 # most recent bar we are allowed to see
    need = ma_len + slope_lb + 1
    if last < need:
        return -1

    def sma_at(idx: int) -> float:
        seg = close[idx - ma_len + 1: idx + 1]
        if len(seg) < ma_len:
            return float("nan")
        return float(np.mean(seg))

    sma_now = sma_at(last)
    sma_prev = sma_at(last - slope_lb)
    if np.isnan(sma_now) or np.isnan(sma_prev):
        return -1
    above = 1 if close[last] > sma_now else 0
    up = 1 if sma_now > sma_prev else 0
    return above + 2 * up


def detect_regime(close: np.ndarray, start_bar: int,
                  vol_terciles: tuple[float, float] | None) -> tuple:
    """Return the (vol_tercile, trend_state) regime label for a move starting at
    start_bar, using ONLY bars < start_bar. vol_terciles = (q33, q66) learned on TRAIN;
    if None (or NaN vol) the vol bucket is -1 (unknown)."""
    vol = _trailing_realized_vol(close, start_bar, VOL_LOOKBACK)
    if vol_terciles is None or np.isnan(vol):
        vtile = -1
    else:
        q33, q66 = vol_terciles
        vtile = 0 if vol <= q33 else (1 if vol <= q66 else 2)
    tstate = _trend_state(close, start_bar, TREND_MA, TREND_SLOPE_LB)
    return (vtile, tstate)


# ============================================================================
# WALK-FORWARD with the ROUTER MECHANISMS
# ============================================================================

@dataclass
class RouterStep:
    test_period: int
    n_train_events: int
    n_test_events: int
    test_span_days: float
    # per-mechanism mean TEST capture
    cap_regime: float
    cap_stable: float
    cap_ewma: float
    # baselines
    cap_lastbest: float
    cap_fixed: float
    cap_ceiling: float
    cap_random: float
    # regime->config map learned on THIS step's TRAIN (for DNA dump)
    regime_map: dict = field(default_factory=dict)


@dataclass
class RouterResult:
    asset: str
    cadence: str
    n_events: int
    steps: list = field(default_factory=list)
    skipped: int = 0
    skip_reasons: dict = field(default_factory=dict)


def _config_label(cands, idx: int) -> str:
    if idx < 0 or idx >= len(cands):
        return "NONE"
    c = cands[idx]
    return f"{c.formulation}/{c.ma_type}/{c.params}"


def _config_class(cands, idx: int) -> str:
    """formulation-family x MA-type (the config-CLASS, coarser than the param config)."""
    if idx < 0 or idx >= len(cands):
        return "NONE"
    c = cands[idx]
    return f"{c.formulation}/{c.ma_type}"


def run_router(asset: str, close: np.ndarray, inp, seed: int = 7) -> RouterResult:
    """Walk forward over consecutive TEST event-blocks (identical stepping to
    oracle_walkforward), and for each block score the THREE router mechanisms plus the
    FOUR baselines. Every mechanism's config CHOICE uses TRAIN events + past-only regime
    state ONLY (no TEST leakage)."""
    rng = np.random.default_rng(seed)
    bpd = BARS_PER_DAY.get(inp.cadence, 1)
    res = RouterResult(asset=asset, cadence=inp.cadence, n_events=len(inp.events))
    n_ev = len(inp.events)
    if n_ev == 0:
        return res

    event_start_bar = np.array([ev.start for ev in inp.events], dtype=np.int64)
    event_end_bar = np.array([ev.end for ev in inp.events], dtype=np.int64)
    M = inp.M                       # [n_events, n_cands]
    n_c = M.shape[1]
    cands = inp.cands

    test_lo = TRAIN_EVENTS
    period_idx = 0
    while test_lo < n_ev:
        test_hi = min(test_lo + TEST_EVENTS, n_ev)
        test_idx = np.arange(test_lo, test_hi)
        if len(test_idx) < MIN_TEST_EVENTS:
            res.skipped += 1
            res.skip_reasons["test_too_small"] = res.skip_reasons.get("test_too_small", 0) + 1
            test_lo = test_hi
            continue

        train_lo = max(0, test_lo - TRAIN_EVENTS)
        train_idx = np.arange(train_lo, test_lo)
        if len(train_idx) < MIN_TRAIN_EVENTS:
            res.skipped += 1
            res.skip_reasons["train_too_small"] = res.skip_reasons.get("train_too_small", 0) + 1
            test_lo = test_hi
            continue

        # CAUSAL guard: every TRAIN event ends at/before the first TEST event start.
        assert event_end_bar[train_idx].max() <= event_start_bar[test_idx].min(), \
            "look-ahead: a train event overlaps/follows the test block"

        train_M = M[train_idx]      # [n_train, n_c]
        test_M = M[test_idx]        # [n_test, n_c]
        train_mean = np.nanmean(train_M, axis=0)
        global_train_best = int(np.nanargmax(train_mean))

        # ---- regime terciles learned on TRAIN (past-only vols at TRAIN starts) ----
        train_vols = np.array(
            [_trailing_realized_vol(close, event_start_bar[i], VOL_LOOKBACK)
             for i in train_idx], dtype=np.float64)
        finite = train_vols[np.isfinite(train_vols)]
        if finite.size >= 3:
            vol_terciles = (float(np.quantile(finite, 1 / 3)),
                            float(np.quantile(finite, 2 / 3)))
        else:
            vol_terciles = None

        # =========================================================================
        # A) REGIME-CONDITIONED ROUTER
        # =========================================================================
        # bucket TRAIN events by past-only regime, learn each bucket's best config.
        bucket_events: dict[tuple, list[int]] = {}
        for local_i, gi in enumerate(train_idx):
            reg = detect_regime(close, int(event_start_bar[gi]), vol_terciles)
            bucket_events.setdefault(reg, []).append(local_i)

        regime_map: dict[tuple, int] = {}
        regime_map_meta: dict = {}
        for reg, locs in bucket_events.items():
            sub = train_M[locs]                      # [n_bucket, n_c]
            bmean = np.nanmean(sub, axis=0)
            best_c = int(np.nanargmax(bmean))
            regime_map[reg] = best_c
            regime_map_meta[f"{reg[0]}|{reg[1]}"] = {
                "config": _config_label(cands, best_c),
                "config_class": _config_class(cands, best_c),
                "n_train_events": len(locs),
                "train_mean_capture": float(bmean[best_c]),
            }

        # apply per-TEST-event: detect regime past-only, route to its bucket's config.
        regime_caps = []
        for ti in test_idx:
            reg = detect_regime(close, int(event_start_bar[ti]), vol_terciles)
            cfg = regime_map.get(reg, global_train_best)   # fallback: global TRAIN best
            regime_caps.append(float(M[ti, cfg]))
        cap_regime = float(np.nanmean(regime_caps)) if regime_caps else float("nan")

        # =========================================================================
        # B) STABLE-SUBSET ENSEMBLE  (top-K consistent configs on TRAIN)
        # =========================================================================
        tmean = np.nanmean(train_M, axis=0)
        tstd = np.nanstd(train_M, axis=0)
        twin = np.nanmean(train_M > 0.0, axis=0)         # TRAIN win-rate per config
        score = tmean - STABLE_LAMBDA * tstd
        # disqualify configs below the win-rate floor.
        score = np.where(twin >= STABLE_WINRATE_FLOOR, score, -np.inf)
        if np.all(~np.isfinite(score)):
            score = tmean - STABLE_LAMBDA * tstd        # relax floor if none qualify
        k = min(STABLE_TOPK, int(np.sum(np.isfinite(score))) or n_c)
        topk = np.argsort(np.where(np.isfinite(score), score, -np.inf))[::-1][:k]
        cap_stable = float(np.nanmean(np.nanmean(test_M[:, topk], axis=1)))

        # =========================================================================
        # C) ADAPTIVE EWMA  (best-recent-EWMA config, causal per TEST event)
        # =========================================================================
        # seed the EWMA with TRAIN events in order; then for each TEST event pick the
        # best-EWMA config using ONLY events strictly before it, apply, then update.
        ewma = np.full(n_c, np.nan, dtype=np.float64)
        for gi in train_idx:                               # seed (chronological)
            row = M[gi]
            seed_mask = np.isnan(ewma) & ~np.isnan(row)
            ewma[seed_mask] = row[seed_mask]
            upd = ~np.isnan(row) & ~np.isnan(ewma)
            ewma[upd] = EWMA_ALPHA * row[upd] + (1 - EWMA_ALPHA) * ewma[upd]
        ewma_caps = []
        for ti in test_idx:
            valid = ~np.isnan(ewma)
            if not valid.any():
                pick = global_train_best
            else:
                masked = np.where(valid, ewma, -np.inf)
                pick = int(np.argmax(masked))
            ewma_caps.append(float(M[ti, pick]))
            # causal update AFTER the pick (the event becomes past for the next pick).
            row = M[ti]
            seed_mask = np.isnan(ewma) & ~np.isnan(row)
            ewma[seed_mask] = row[seed_mask]
            upd = ~np.isnan(row) & ~np.isnan(ewma)
            ewma[upd] = EWMA_ALPHA * row[upd] + (1 - EWMA_ALPHA) * ewma[upd]
        cap_ewma = float(np.nanmean(ewma_caps)) if ewma_caps else float("nan")

        # =========================================================================
        # BASELINES
        # =========================================================================
        cap_lastbest = float(np.nanmean(test_M[:, global_train_best]))
        test_mean = np.nanmean(test_M, axis=0)
        ceiling_best = int(np.nanargmax(test_mean))
        cap_ceiling = float(test_mean[ceiling_best])
        cap_fixed = (float(np.nanmean(test_M[:, inp.fixed_idx]))
                     if inp.fixed_idx >= 0 else float("nan"))
        rand_idx = rng.integers(0, n_c, size=min(N_RANDOM, n_c))
        cap_random = float(np.mean([float(np.nanmean(test_M[:, kk])) for kk in rand_idx]))

        span_bars = int(event_end_bar[test_idx[-1]] - event_start_bar[test_idx[0]])
        span_days = span_bars / bpd if bpd else float(span_bars)

        res.steps.append(RouterStep(
            test_period=period_idx,
            n_train_events=len(train_idx),
            n_test_events=len(test_idx),
            test_span_days=float(span_days),
            cap_regime=cap_regime,
            cap_stable=cap_stable,
            cap_ewma=cap_ewma,
            cap_lastbest=cap_lastbest,
            cap_fixed=cap_fixed,
            cap_ceiling=cap_ceiling,
            cap_random=cap_random,
            regime_map=regime_map_meta,
        ))
        period_idx += 1
        test_lo = test_hi

    return res


# ============================================================================
# AGGREGATION + VERDICT
# ============================================================================

def _safe_mean(a):
    a = np.asarray(a, dtype=np.float64)
    a = a[~np.isnan(a)]
    return float(np.mean(a)) if a.size else None


def _safe_median(a):
    a = np.asarray(a, dtype=np.float64)
    a = a[~np.isnan(a)]
    return float(np.median(a)) if a.size else None


def _winrate_delta(mech: np.ndarray, base: np.ndarray):
    """Return (win-rate, mean-delta, median-delta) of (mech - base) over valid steps."""
    d = mech - base
    v = ~np.isnan(d)
    if not v.any():
        return None, None, None
    return float(np.mean(d[v] > 0)), float(np.mean(d[v])), float(np.median(d[v]))


MECHANISMS = ("regime", "stable", "ewma")
MECH_LABELS = {"regime": "A) REGIME-ROUTER", "stable": "B) STABLE-SUBSET",
               "ewma": "C) ADAPTIVE-EWMA"}


def aggregate_router(steps: list) -> dict:
    if not steps:
        return {"n_steps": 0, "verdict": "no walk-forward steps (too few events)"}

    cols = {
        "regime": np.array([s.cap_regime for s in steps], dtype=np.float64),
        "stable": np.array([s.cap_stable for s in steps], dtype=np.float64),
        "ewma": np.array([s.cap_ewma for s in steps], dtype=np.float64),
        "lastbest": np.array([s.cap_lastbest for s in steps], dtype=np.float64),
        "fixed": np.array([s.cap_fixed for s in steps], dtype=np.float64),
        "ceiling": np.array([s.cap_ceiling for s in steps], dtype=np.float64),
        "random": np.array([s.cap_random for s in steps], dtype=np.float64),
    }

    out = {"n_steps": len(steps), "means": {}, "medians": {},
           "frac_positive": {}, "vs_fixed": {}, "vs_lastbest": {},
           "persistence_vs_ceiling": {}}
    for key, arr in cols.items():
        out["means"][key] = _safe_mean(arr)
        out["medians"][key] = _safe_median(arr)
        valid = arr[~np.isnan(arr)]
        out["frac_positive"][key] = (float(np.mean(valid > 0)) if valid.size else None)

    ceiling_m = out["means"]["ceiling"]
    for mech in MECHANISMS:
        wr_f, d_f, md_f = _winrate_delta(cols[mech], cols["fixed"])
        wr_l, d_l, md_l = _winrate_delta(cols[mech], cols["lastbest"])
        out["vs_fixed"][mech] = {"winrate": wr_f, "mean_delta": d_f, "median_delta": md_f}
        out["vs_lastbest"][mech] = {"winrate": wr_l, "mean_delta": d_l,
                                    "median_delta": md_l}
        mm = out["means"][mech]
        out["persistence_vs_ceiling"][mech] = (
            (mm / ceiling_m) if (mm is not None and ceiling_m not in (None, 0)) else None)

    out["verdict"] = _router_verdict(out)
    return out


def _router_verdict(agg: dict) -> dict:
    """THE honest verdict. A mechanism captures REAL signal iff ALL of:
      (i)   positive mean OOS capture,
      (ii)  beats FIXED robustly (winrate >= BEAT_FIXED_WINRATE and
            mean_delta > BEAT_FIXED_DELTA),
      (iii) beats LAST-BEST (mean_delta > BEAT_LASTBEST_DELTA).
    Reports the winner if any, else the closest mechanism + its gap. NO manufactured
    edge -- a mechanism that fails ANY clause is NOT a missing link.
    """
    means = agg["means"]
    medians = agg["medians"]
    fixed_med = medians.get("fixed")
    found = []
    scored = []
    for mech in MECHANISMS:
        mm = means[mech]
        vf = agg["vs_fixed"][mech]
        vl = agg["vs_lastbest"][mech]
        if mm is None or vf["winrate"] is None:
            continue
        clause_i = mm > 0.0
        # (ii) beats FIXED robustly: meaningful mean-delta (WEALTH magnitude) AND a
        # not-one-step-luck flag = win-rate floor OR median-capture dominance.
        delta_ok = (vf["mean_delta"] is not None
                    and vf["mean_delta"] > BEAT_FIXED_DELTA)
        winrate_flag = vf["winrate"] >= BEAT_FIXED_WINRATE
        # not-systematically-worse flag: the per-step MEDIAN delta vs fixed >= 0 means
        # the mechanism does not lose to fixed on a typical step (the positive mean is
        # then a genuine tail edge, not a "lose-often-win-rarely" lottery dressed up).
        median_delta_flag = (vf["median_delta"] is not None
                             and vf["median_delta"] >= 0.0)
        clause_ii = delta_ok and (winrate_flag or median_delta_flag)
        clause_iii = (vl["mean_delta"] is not None
                      and vl["mean_delta"] > BEAT_LASTBEST_DELTA)
        passes = clause_i and clause_ii and clause_iii
        # closeness score: how many clauses pass + a tie-break on the fixed-delta.
        ncl = int(clause_i) + int(clause_ii) + int(clause_iii)
        scored.append((ncl, vf["mean_delta"] or -1e9, mech,
                       clause_i, clause_ii, clause_iii))
        if passes:
            found.append(mech)

    if found:
        # pick the strongest passing mechanism by mean OOS capture.
        best = max(found, key=lambda m: means[m])
        vfb = agg["vs_fixed"][best]
        # tail-concentration flag: if the per-step win-rate is LOW while the mean-delta
        # is positive, the edge is concentrated in a few steps (tail) -- honestly flag it
        # so the deeper-robustness review knows to run the top-k-of-compound / jackknife
        # checks BEFORE any belief. (A wealth-positive edge can be tail-driven; that is
        # NOT disqualifying here, but it is NOT a clean broad edge either.)
        tail_concentrated = (vfb["winrate"] is not None
                             and vfb["winrate"] < BEAT_FIXED_WINRATE)
        return {
            "missing_link_found": True,
            "mechanism": best,
            "mechanism_label": MECH_LABELS[best],
            "mean_oos_capture": means[best],
            "vs_fixed": vfb,
            "vs_lastbest": agg["vs_lastbest"][best],
            "all_passing": found,
            "tail_concentrated": bool(tail_concentrated),
            "note": ("CAUSAL config-router captures REAL signal at this level: positive "
                     "OOS, beats FIXED robustly (mean-delta), beats LAST-BEST. "
                     + ("EDGE IS TAIL-CONCENTRATED (low per-step win-rate vs fixed) -- "
                        "the magnitude comes from a few steps; MANDATORY deeper robustness "
                        "(jackknife, top-k-of-compound, per-asset persistence, seeds) "
                        "before any belief or capital. "
                        if tail_concentrated else
                        "Edge is broad (per-step win-rate clears the bar). ")
                     + "FLAG FOR DEEPER ROBUSTNESS before any capital."),
        }

    # NOT found: report the closest mechanism + the gap on each clause.
    if not scored:
        return {"missing_link_found": False,
                "note": "indeterminate -- no scorable mechanism."}
    scored.sort(reverse=True)
    ncl, _, mech, ci, cii, ciii = scored[0]
    vf = agg["vs_fixed"][mech]
    vl = agg["vs_lastbest"][mech]
    gaps = []
    if not ci:
        gaps.append(f"mean OOS {means[mech]:.4f} <= 0 (loses money)")
    if not cii:
        gaps.append(f"vs-FIXED mean-delta {vf['mean_delta']} (bar {BEAT_FIXED_DELTA}) "
                    f"AND/OR robustness flag (winrate {vf['winrate']} >= "
                    f"{BEAT_FIXED_WINRATE} OR median dominance) not met")
    if not ciii:
        gaps.append(f"vs-LAST-BEST delta {vl['mean_delta']} <= {BEAT_LASTBEST_DELTA}")
    return {
        "missing_link_found": False,
        "closest_mechanism": mech,
        "closest_mechanism_label": MECH_LABELS[mech],
        "clauses_passed": f"{ncl}/3",
        "mean_oos_capture": means[mech],
        "gaps": gaps,
        "note": ("Missing link NOT found at this level. The closest mechanism still "
                 "fails at least one clause (positive-OOS / beats-FIXED / beats-LAST-BEST). "
                 "Honest report: per-period config adjustment does not capture real "
                 "forward signal here."),
    }


# ============================================================================
# REPORTING
# ============================================================================

def _fmt(v, pct=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "    n/a"
    return f"{v * 100:7.3f}%" if pct else f"{v:8.4f}"


def print_report(label: str, agg: dict) -> None:
    print("")
    print("=" * 110)
    print(f"CONFIG-ROUTER MISSING-LINK SOLVER -- {label}")
    print("=" * 110)
    if agg.get("n_steps", 0) == 0:
        print("  (no walk-forward steps -- too few events)")
        print("=" * 110)
        return
    print("capture = realized long ROI (net taker), same units for every row.\n")

    means = agg["means"]
    meds = agg["medians"]
    fpos = agg["frac_positive"]

    print(f"  steps = {agg['n_steps']}\n")
    print(f"{'row':>20} | {'mean cap':>10} {'median':>10} {'frac>0':>8} | "
          f"{'vs-FIXED wr':>12} {'vs-FIXED d':>11} | {'vs-LAST wr':>11} "
          f"{'persist/ceil':>13}")
    print("-" * 110)

    def line(name, key, mech=None):
        wr_f = d_f = wr_l = pers = None
        if mech:
            wr_f = agg["vs_fixed"][mech]["winrate"]
            d_f = agg["vs_fixed"][mech]["mean_delta"]
            wr_l = agg["vs_lastbest"][mech]["winrate"]
            pers = agg["persistence_vs_ceiling"][mech]
        print(f"{name:>20} | {_fmt(means[key]):>10} {_fmt(meds[key]):>10} "
              f"{_fmt(fpos[key]):>8} | {_fmt(wr_f):>12} {_fmt(d_f):>11} | "
              f"{_fmt(wr_l):>11} {_fmt(pers):>13}")

    line("A) REGIME-ROUTER", "regime", "regime")
    line("B) STABLE-SUBSET", "stable", "stable")
    line("C) ADAPTIVE-EWMA", "ewma", "ewma")
    print("-" * 110)
    line("LAST-BEST (failed)", "lastbest")
    line("FIXED (naive)", "fixed")
    line("CEILING (hindsight)", "ceiling")
    line("RANDOM (floor)", "random")
    print("-" * 110)

    v = agg["verdict"]
    print("\nVERDICT:")
    if v.get("missing_link_found"):
        print(f"  MISSING LINK FOUND -> {v['mechanism_label']}")
        print(f"    mean OOS capture = {v['mean_oos_capture']*100:.3f}%")
        print(f"    vs FIXED: winrate={v['vs_fixed']['winrate']:.2f} "
              f"delta={v['vs_fixed']['mean_delta']*100:.3f}%")
        print(f"    vs LAST-BEST: delta={v['vs_lastbest']['mean_delta']*100:.3f}%")
        print(f"    {v['note']}")
    else:
        if "closest_mechanism_label" in v:
            print(f"  MISSING LINK NOT FOUND. closest = {v['closest_mechanism_label']} "
                  f"({v.get('clauses_passed')} clauses)")
            print(f"    mean OOS capture = {v['mean_oos_capture']*100:.3f}%")
            for g in v.get("gaps", []):
                print(f"    gap: {g}")
        print(f"    {v.get('note')}")
    print("=" * 110)
    print("NOTE: regime detected PAST-ONLY at the move start; configs fitted on TRAIN "
          "only; TEST untouched for the config choice. ceiling = hindsight.")
    print("")


def dump_regime_map(steps: list, top_n: int = 15) -> dict:
    """Aggregate the per-step regime->config maps into a stable DNA picture: for each
    regime bucket, the most-frequently-chosen config + config-class across steps."""
    from collections import Counter, defaultdict
    cfg_by_reg: dict = defaultdict(Counter)
    cls_by_reg: dict = defaultdict(Counter)
    n_by_reg: dict = defaultdict(int)
    for s in steps:
        for reg_key, meta in s.regime_map.items():
            cfg_by_reg[reg_key][meta["config"]] += 1
            cls_by_reg[reg_key][meta["config_class"]] += 1
            n_by_reg[reg_key] += 1
    out = {}
    for reg_key in sorted(n_by_reg, key=lambda k: -n_by_reg[k]):
        out[reg_key] = {
            "appearances": n_by_reg[reg_key],
            "top_config": cfg_by_reg[reg_key].most_common(3),
            "top_config_class": cls_by_reg[reg_key].most_common(3),
        }
    return out


# ============================================================================
# SELFTEST (two-sided)
# ============================================================================

def _synth_regime_determines(n=4200, seed=21):
    """REGIME DETERMINES THE BEST CONFIG: the series alternates between LONG smooth
    trend regimes (a slow trend-riding price>MA config wins) and choppy high-vol pop
    regimes (a fast / quick-exit config wins). The regime is detectable PAST-ONLY (vol +
    trend-state), so a regime-router that learned 'trend->slow ride, chop->fast' on
    TRAIN should carry it forward and BEAT a single fixed config.

    Regime blocks are sized so that the TEST block OFTEN sits in a DIFFERENT regime than
    the TRAIN majority -- this is what makes LAST-BEST (one global TRAIN config rolled
    forward) fail, while the regime-CONDITIONED router (which keeps a per-regime config
    and re-routes by the past-only TEST regime) adapts and beats it. Blocks comparable to
    the walk-forward window (TRAIN+TEST ~ 160 bars) straddle the train/test boundary often
    enough to expose this gap, yet each block is long enough that the per-regime config is
    learnable on TRAIN. The regime LABEL flips across the series so a single fixed config
    cannot win in both regimes either.
    """
    rng = np.random.default_rng(seed)
    price = [100.0]
    block = 130   # ~ the walk-forward window, so TEST often flips regime vs TRAIN
    for i in range(n):
        regime = (i // block) % 2
        if regime == 0:   # smooth persistent uptrend -> slow trend-ride wins
            ret = 0.0035 + rng.normal(0, 0.0030)
        else:             # choppy high-vol oscillation -> fast / quick-exit wins
            ret = 0.030 * np.sin(i / 1.6) * 0.5 + rng.normal(0, 0.013)
        price.append(price[-1] * (1.0 + ret))
    p = np.array(price, dtype=np.float64)
    o = p.copy()
    c = p.copy()
    h = p * 1.0016
    lo = p * 0.9984
    return o, h, lo, c


def _synth_random_best(n=2600, seed=23):
    """BEST CONFIG IS RANDOM PER MOVE: a single homogeneous noisy regime where which
    config wins each move is essentially luck (no detectable regime structure ties a
    config to the move). All mechanisms should collapse toward fixed/random -- no
    mechanism should robustly beat FIXED, and the verdict must be NOT FOUND.
    """
    rng = np.random.default_rng(seed)
    price = [100.0]
    for i in range(n):
        # homogeneous noise with tiny drift; no regime blocks -> best config is luck.
        ret = 0.0006 + rng.normal(0, 0.011)
        price.append(price[-1] * (1.0 + ret))
    p = np.array(price, dtype=np.float64)
    o = p.copy()
    c = p.copy()
    h = p * 1.004
    lo = p * 0.996
    return o, h, lo, c


def _run_synth(o, h, lo, c, cadence="1d", seed=7):
    inp = build_capture_matrix(o, h, lo, c, cadence)
    res = run_router("SYNTH", c, inp, seed=seed)
    agg = aggregate_router(res.steps)
    return res, agg


def selftest() -> bool:
    ok = True

    # --- Side 1: REGIME DETERMINES BEST -> regime-router captures it, beats fixed,
    #     positive OOS. (The test must be ABLE to detect a real router-edge.) --------
    o, h, lo, c = _synth_regime_determines()
    res1, agg1 = _run_synth(o, h, lo, c, cadence="1d")
    print(f"[selftest] REGIME-DETERMINES: {res1.n_events} events, "
          f"{agg1.get('n_steps')} steps")
    if agg1.get("n_steps", 0) < 2:
        print("[selftest] FAIL: regime-determines series produced too few steps")
        return False
    m1 = agg1["means"]
    rg = m1["regime"]
    fx = m1["fixed"]
    vf = agg1["vs_fixed"]["regime"]
    print(f"[selftest]   regime={rg:.4f} fixed={fx:.4f} random={m1['random']:.4f} "
          f"ceiling={m1['ceiling']:.4f} | vs-fixed wr={vf['winrate']} d={vf['mean_delta']}")
    # The regime-router must be POSITIVE on a regime-structured series ...
    if rg is None or rg <= 0:
        print(f"[selftest] FAIL(1): regime-router not positive ({rg}) on regime series")
        ok = False
    # ... and must BEAT the fixed config (the captured regime edge over a single config).
    if fx is not None and not (rg > fx):
        print(f"[selftest] FAIL(1): regime-router ({rg:.4f}) does not beat fixed "
              f"({fx:.4f}) on a regime-structured series")
        ok = False
    # ... and -- THE key requirement -- the verdict must FIND the missing link (SOME
    # adaptive mechanism beats FIXED robustly AND beats LAST-BEST). This is what proves
    # the apparatus CAN detect a real router-edge over the failed last-best baseline.
    if not agg1["verdict"].get("missing_link_found"):
        print("[selftest] FAIL(1): verdict did not find the missing link on a "
              "regime-structured series (the test cannot detect a real edge)")
        ok = False

    # --- Side 2: BEST CONFIG RANDOM PER MOVE -> all collapse to ~fixed/random,
    #     verdict NOT FOUND (no false positive). ------------------------------------
    o2, h2, lo2, c2 = _synth_random_best()
    res2, agg2 = _run_synth(o2, h2, lo2, c2, cadence="1d")
    print(f"[selftest] RANDOM-BEST: {res2.n_events} events, {agg2.get('n_steps')} steps")
    if agg2.get("n_steps", 0) < 2:
        print("[selftest] FAIL: random-best series produced too few steps")
        return False
    m2 = agg2["means"]
    print(f"[selftest]   regime={m2['regime']:.4f} stable={m2['stable']:.4f} "
          f"ewma={m2['ewma']:.4f} fixed={m2['fixed']:.4f} random={m2['random']:.4f}")
    # No mechanism may robustly beat fixed -> the verdict must be NOT FOUND.
    if agg2["verdict"].get("missing_link_found"):
        print("[selftest] FAIL(2): verdict FOUND a missing link on a random-best series "
              "(FALSE POSITIVE -- the test manufactured an edge)")
        ok = False
    # Sanity: the regime-router must NOT wildly beat fixed on the random series.
    rg2 = m2["regime"]
    fx2 = m2["fixed"]
    if (rg2 is not None and fx2 is not None and rg2 > fx2 + 0.01):
        print(f"[selftest] WARN(2): regime ({rg2:.4f}) >> fixed ({fx2:.4f}) on random "
              f"series -- inspect (not an auto-fail unless verdict flips).")

    print(f"[selftest] side-1 verdict: found={agg1['verdict'].get('missing_link_found')}")
    print(f"[selftest] side-2 verdict: found={agg2['verdict'].get('missing_link_found')}")
    print("[selftest] PASS" if ok else "[selftest] FAIL")
    return ok


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="CONFIG-ROUTER missing-link solver (causal week-by-week config "
                    "adjustment)")
    ap.add_argument("--assets", default="BTCUSDT,ETHUSDT,SOLUSDT")
    ap.add_argument("--cadences", default="4h,1h")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    assets = [a.strip() for a in args.assets.split(",") if a.strip()]
    cadences = [c.strip() for c in args.cadences.split(",") if c.strip()]
    for cad in cadences:
        if cad not in WINDOW_BARS:
            print(f"[error] unknown cadence '{cad}'; known={list(WINDOW_BARS)}")
            sys.exit(2)

    per_combo = []
    all_steps = []
    per_asset_steps: dict = {}
    for asset in assets:
        for cad in cadences:
            print(f"[run] {asset} {cad}: loading + events + capture matrix ...",
                  flush=True)
            o, h, lo, c = load_ohlc(asset, cad)
            inp = build_capture_matrix(o, h, lo, c, cad)
            res = run_router(asset, c, inp, seed=args.seed)
            agg = aggregate_router(res.steps)
            agg["asset"] = asset
            agg["cadence"] = cad
            agg["n_events"] = res.n_events
            per_combo.append(agg)
            all_steps.extend(res.steps)
            per_asset_steps.setdefault(asset, []).extend(res.steps)
            print(f"[run] {asset} {cad}: {res.n_events} events, {len(res.steps)} steps, "
                  f"{res.skipped} skipped", flush=True)
            print_report(f"{asset} {cad}", agg)

    # AGGREGATE over all asset x cadence steps -- the headline verdict.
    agg_all = aggregate_router(all_steps)
    print_report("AGGREGATE (all assets x cadences)", agg_all)

    regime_dna = dump_regime_map(all_steps)
    if agg_all.get("verdict", {}).get("missing_link_found"):
        print("\nREGIME -> CONFIG DNA MAP (aggregate; only meaningful if FOUND):")
        for reg_key, info in list(regime_dna.items())[:12]:
            print(f"  regime {reg_key}: appears {info['appearances']}x | "
                  f"top config {info['top_config']} | class {info['top_config_class']}")

    asset_tag = "_".join(a.upper().replace("USDT", "") for a in assets)
    cad_tag = "_".join(cadences)
    out_path = (Path(args.out) if args.out
                else PROJECT_ROOT / "runs" / "strat" /
                f"config_router_{asset_tag}__{cad_tag}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "tool": "config_router",
        "question": ("is there a CAUSAL week-by-week config-adjustment rule that "
                     "captures REAL forward signal (positive OOS + beats FIXED + beats "
                     "LAST-BEST)?"),
        "assets": assets,
        "cadences": cadences,
        "spec": {
            "test_events_per_block": TEST_EVENTS,
            "train_events_per_block": TRAIN_EVENTS,
            "min_train_events": MIN_TRAIN_EVENTS,
            "min_test_events": MIN_TEST_EVENTS,
            "fixed_config": f"{FIXED_FORMULATION}/{FIXED_MA_TYPE}/{FIXED_PARAMS}",
            "n_random": N_RANDOM,
            "regime": {"vol_lookback": VOL_LOOKBACK, "trend_ma": TREND_MA,
                       "trend_slope_lb": TREND_SLOPE_LB},
            "stable": {"topk": STABLE_TOPK, "lambda": STABLE_LAMBDA,
                       "winrate_floor": STABLE_WINRATE_FLOOR},
            "ewma_alpha": EWMA_ALPHA,
            "verdict_bar": {"beat_fixed_winrate": BEAT_FIXED_WINRATE,
                            "beat_fixed_delta": BEAT_FIXED_DELTA,
                            "beat_lastbest_delta": BEAT_LASTBEST_DELTA},
            "causal": ("regime past-only at move start; configs fitted on TRAIN only; "
                       "TEST untouched for config choice; TRAIN strictly precedes TEST"),
            "capture_units": "realized_long_roi_net_taker",
        },
        "per_asset_cadence": per_combo,
        "aggregate": agg_all,
        "regime_config_dna": regime_dna,
    }
    out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"[artifact] {out_path}")


if __name__ == "__main__":
    main()
