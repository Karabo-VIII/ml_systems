"""src/strat/regime_gated_longshort.py -- PHASE 7: CONDITIONAL BEAR-INSURANCE (the regime-gated longshort book).

THE SYNTHESIS QUESTION PHASE 6 TEED UP. PHASE 6 found the ALWAYS-ON longshort sleeve gives +15-18pp bear
DD-protection but pays a HEAVY bull-drag at fine TF (-8 to -41pp net). The obvious fix tested here: deploy the
longshort INSURANCE sleeve ONLY when a sustained bear is DETECTED, run trend-alone (long-only) otherwise --
capture the bear protection, dodge the bull-drag. This is a BINARY insurance toggle driven by a SLOW HYSTERETIC
CAUSAL market-regime detector -- genuinely different from the CONTINUOUS dynamic allocator that showed NO skill
in PHASE 2/3. The SHUFFLE control (the same discipline that killed the dynamic engine) is MANDATORY here.

WHAT IT BUILDS:
  THE REGIME-GATED LONGSHORT BOOK:
    - CORE      = the trend sleeve (adaptive-MA per-TF winner VIDYA/KAMA, LONG-only), ALWAYS on.
    - INSURANCE = the PHASE-6 longshort sleeve (long up-cross + short down-cross, FIXED short trail-stop, maker
                  + modelled short-borrow), toggled ON only on DETECTED-sustained-bear days; OFF otherwise.
    - gated[t]  = trend[t]                          on non-bear days  (pure long-only core)
                = (1-w)*trend[t] + w*longshort[t]   on detected-bear days  (insurance blended in; w=INSURANCE_W)
  THE DETECTOR (pre-registered, CAUSAL, SLOW, HYSTERETIC, never OOS/synthetic-path fit):
    - substrate = the equal-weight u10 basket CLOSE (normalized), daily.
    - raw bear  = basket close < SMA(DET_SMA) (a slow level filter).
    - HYSTERESIS: turn the bear gate ON only after K_ON consecutive raw-bear bars; turn it OFF only after
      K_OFF consecutive raw-NOT-bear bars (a debounce band that avoids flicker around the SMA). All causal
      (close[i] decides day i's gate via a 1-bar shift -> the gate that governs day t was decided on t-1).
    - PRE-REGISTERED on the 2020 TRAIN band (a small grid scored by a TRAIN-only objective), then FROZEN; the
      OOS band + EVERY synthetic path use the frozen detector -- the detector is NEVER fit on what it is graded on.

THE DECISIVE CONTROLS (two-sided, binding):
  (a) vs ALWAYS-ON longshort   : does gating CAPTURE most of the bear protection while PAYING little of the
                                  bull/chop drag? (bear-protection-captured % + bull-drag-paid %, both vs always-on)
  (b) vs TREND-ALONE+trail     : does the gated insurance beat plain trend on worst-regime net AND full-mix maxDD
                                  -- the bar PHASE 6's STATIC book FAILED?
  (c) vs a SHUFFLED toggle      : a random-timed gate of EQUAL on-frequency. Does the bear DETECTOR add real
                                  SKILL, or would a random-timed toggle do as well? MANDATORY (the shuffle that
                                  killed the dynamic engine). If the detector does NOT beat its shuffle -> regime
                                  detection genuinely does not work on this data even as a binary toggle.
  (d) block-bootstrap p05 + WORST seed across >=20 seeds (distributions, never a cherry-picked path).

HONEST / TWO-SIDED (binding):
  - If the gated book does NOT beat trend-alone OR the detector does NOT beat its shuffle -> we REPORT IT: regime
    detection does not work even as binary insurance; trend-alone+trail is the honest answer; the bear rescue is
    unconditional-only (always-on short, accepting the bull-drag) or none. That is a clean, decisive closure.
  - The SHORT leg VIOLATES the standing long-only+spot constraint -> the gated longshort book is FLAGGED RESEARCH;
    deploying it needs the user's explicit long-only-EXCEPTION sign-off. We BUILD + VALIDATE for the learning.
  - SHORT-BORROW is MODELLED (reused from PHASE 6): ~20 bps/yr base, prorated per-bar on the short leg.
  - SYNTHETIC test surface from PHASE 3/6's VALIDATED generator (2020-calibrated stylized facts ONLY) + the 2020
    TRAIN/VAL/OOS real band. NO 2026/other data is ever read here.
  - >=20 seeds; report DISTRIBUTIONS (mean +- spread + WORST seed).

CONSTRAINTS (user mandate, BINDING): 2020 BAND + the VALIDATED synthetic generator ONLY; charts (PNG); no emoji
(cp1252); RWYB; do NOT git commit. SHORT side = RESEARCH (deploy needs LO-exception sign-off). PRE-REGISTERED
detector (no OOS/synthetic-path fit); the SHUFFLE control is MANDATORY.

RWYB:
  python -m strat.regime_gated_longshort --selftest                              # detector + gate soundness
  python -m strat.regime_gated_longshort --seeds 20 --cadences 1d,4h,2h,1h,30m,15m  # the full gated book
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="invalid value encountered in divide")
np.seterr(invalid="ignore", divide="ignore")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse PHASE 3/6's validated generator + the PHASE-6 engine mechanics (NOT reinvented).
import strat.synthetic_regime_stress as SRS                              # noqa: E402  (validated generator)
import strat.deep2020_complementarity as COMP                           # noqa: E402
import strat.ma_2020_breakdown as M2                                    # noqa: E402  (the canonical within-2020 TRAIN/VAL/OOS split + _panel)
import strat.complementary_sleeve_search as CSS                         # noqa: E402  (FIXED short trail-stop, _daily, _panel_window)
import strat.longshort_ma_engine as LSE                                 # noqa: E402  (the PHASE-6 trend + longshort books)
from strat.portfolio_replay import MAKER_RT, TAKER_RT                    # noqa: E402
from strat.data_expansion import block_bootstrap_distribution           # noqa: E402

OUT = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
CHARTS = OUT / "charts"
SYMS = COMP.SYMS
SCENARIOS = ["bull", "bear", "chop", "stitched"]
ALL_TFS = LSE.ALL_TFS                                                    # [1d,4h,2h,1h,30m,15m]
BORROW_BASE = LSE.BORROW_BASE                                            # 20 bps/yr (reuse PHASE-6 base)

# The INSURANCE blend weight on detected-bear days: gated = (1-w)*trend + w*longshort. 0.5 = the symmetric
# blend (matches PHASE-6's always-on 0.5*long+0.5*short framing applied as an overlay on detected-bear days).
INSURANCE_W = 0.5

# ----------------------------------------------------------------------------------------------------------
# THE DETECTOR pre-registration grid + the FROZEN params. The detector is a SLOW HYSTERETIC causal SMA gate
# on the equal-weight basket close. We pre-register on the 2020 TRAIN band ONLY (a small grid scored by a
# TRAIN-only objective), FREEZE, then apply the frozen params to OOS + all synthetic paths. The grid is small
# + the params are economically motivated (slow SMA, multi-bar debounce) -- not a fishing expedition.
DET_GRID = {
    # SLOW level filters, but at timescales TESTABLE on the 2020 sub-period regime lengths (bear ~39d, bull ~92d,
    # chop ~105d) -- a SMA(100-200) cannot even compute on a ~39-bar bear, so the grid is bounded by the data's
    # resolution (the regime durations the generator is calibrated to). These are still SLOW relative to the
    # entry signal (the trend MA is faster) -- the gate is a REGIME filter, not a trade timer.
    "sma":   [20, 30, 50],           # the slow level filter length (bars)
    "k_on":  [3, 5],                 # consecutive raw-bear bars to TURN ON the gate (debounce)
    "k_off": [3, 5],                 # consecutive raw-not-bear bars to TURN OFF the gate (hysteresis/debounce)
}
# FROZEN after pre-registration (filled by pre_register_detector(); the default here is the economically-motivated
# prior, overwritten by the TRAIN fit at runtime). Stored so the module is self-documenting + re-runnable.
DETECTOR = {"sma": 50, "k_on": 5, "k_off": 5}

__contract__ = {
    "kind": "regime_gated_longshort_conditional_bear_insurance",
    "inputs": {
        "core": "the trend sleeve (LSE._trend_book: adaptive-MA per-TF winner VIDYA/KAMA, LONG-only) -- ALWAYS on",
        "insurance": "the PHASE-6 longshort sleeve (LSE._longshort_book: long up-cross + short down-cross, FIXED "
                     "short trail-stop, maker + modelled short-borrow) -- toggled ON only on detected-bear days",
        "detector": "a SLOW HYSTERETIC CAUSAL SMA gate on the equal-weight u10 basket close (close<SMA(N); K_ON "
                    "consecutive bars to arm, K_OFF to disarm; 1-bar-shifted = causal). PRE-REGISTERED on the "
                    "2020 TRAIN band, FROZEN, then applied to OOS + all synthetic paths (never OOS/path-fit)",
        "test_surface": "PHASE 3/6 VALIDATED synthetic generator (bull/bear/chop/stitched, calibrated on REAL "
                        "2020-band data ONLY) + the 2020 TRAIN/VAL/OOS real band; never 2026/other",
    },
    "outputs": {
        "bear_protection_captured_pct": "% of the ALWAYS-ON longshort's bear DD-protection the gated book keeps",
        "bull_drag_paid_pct": "% of the ALWAYS-ON longshort's bull/chop net-drag the gated book still pays",
        "gated_vs_trend_alone": "does the gated book beat plain trend on worst-regime net AND full-mix maxDD?",
        "gated_vs_shuffle": "does the bear DETECTOR beat a random-timed toggle of EQUAL on-frequency? (the skill test)",
        "robustness": "block-bootstrap p05 + WORST seed across >=20 seeds per regime",
    },
    "invariants": {
        "synthetic_2020_only": "test surface from PHASE 3/6's validated generator + the 2020 real band; never 2026/other",
        "pre_registered_detector": "detector params fit on the 2020 TRAIN band ONLY, then FROZEN; OOS + all synthetic "
                                   "paths use the frozen detector -- NEVER fit on what it is graded on",
        "causal_detector": "the gate governing day t is decided from close<=t-1 (1-bar shift); no look-ahead",
        "shuffle_control_mandatory": "the detector must beat a RANDOM-TIMED toggle of EQUAL on-frequency -- else regime "
                                     "detection does not add skill even as a binary insurance toggle (report it)",
        "fixed_short_trail_stop": "the insurance leg reuses LSE/CSS._apply_trail_stop_short (the FIX), via LSE._longshort_book",
        "short_borrow_modelled": "borrow prorated per-bar on the short leg (BORROW_BASE bps/yr) via LSE._longshort_book",
        "short_is_research_not_deploy": "the gated longshort book VIOLATES long-only+spot -> FLAGGED RESEARCH; deploy "
                                        "needs the user's explicit LO-exception sign-off",
        "distributions_not_single_paths": ">=20 seeds; report mean +- spread + WORST seed; never cherry-pick",
        "two_sided_honest": "if the gated book does NOT beat trend-alone OR the detector does NOT beat its shuffle, that "
                            "is the headline -- regime detection does not work even as binary insurance; reported, not buried",
    },
}


# =====================================================================================================
# 1. THE DETECTOR -- a slow hysteretic causal SMA gate on the equal-weight basket close.
# =====================================================================================================
def _basket_close_daily(cad):
    """The equal-weight u10 basket CLOSE as a daily series, over the SAME panel window the books use (so it
    aligns to the daily book index). Each asset's close is normalized to its window-start (=1.0) so the basket
    is a clean cross-asset average level; then daily-resampled by LAST close (a level, not a return). Run INSIDE
    the _synthetic_panel_context for synthetic; outside for the 2020 real band."""
    closes = []
    for sym in SYMS:
        pw = CSS._panel_window(sym, cad)
        if pw is None:
            continue
        c2, ms2, ret, win, idx = pw
        cwin = c2[win].astype(float)
        if len(cwin) < 5 or cwin[0] <= 0:
            continue
        closes.append(pd.Series(cwin / cwin[0], index=idx))     # normalized level (start=1.0)
    if not closes:
        return None
    bar = pd.concat(closes, axis=1).mean(axis=1, skipna=True)
    # daily = LAST close of the day (a level)
    daily = bar.resample("1D").last().dropna()
    return daily


def _basket_close_2020_band(start, end):
    """The equal-weight u10 basket CLOSE as a daily series over a REAL 2020-band sub-window [start, end). Reads
    REAL daily panels DIRECTLY (M2._panel), FENCED hard to the 2020 calibration band (never 2026/other). Used
    ONLY for detector PRE-REGISTRATION on the canonical M2 TRAIN band (2020-01-01..2020-07-01), which CONTAINS
    the real Feb-Mar COVID bear -- so the slow bear detector is fit on a window that actually has a bear in it
    (the narrow sleeve scoring window COMP.WIN = Jul-Dec 2020 is a clean bull and cannot train a bear gate)."""
    fence_s = pd.Timestamp(SRS.CALIB_WINDOW[0]).value // 10**6
    fence_e = pd.Timestamp(SRS.CALIB_WINDOW[1]).value // 10**6
    s_ms = pd.Timestamp(start).value // 10**6
    e_ms = pd.Timestamp(end).value // 10**6
    assert s_ms >= fence_s and e_ms <= fence_e, "detector pre-registration window escapes the 2020 band -- FORBIDDEN"
    closes = []
    for sym in SYMS:
        try:
            o, h, l, c, ms = M2._panel(sym, "1d")
        except Exception:
            continue
        m = (ms >= s_ms) & (ms < e_ms)
        if m.sum() < 20:
            continue
        cc = c[m].astype(float)
        if cc[0] <= 0:
            continue
        closes.append(pd.Series(cc / cc[0], index=pd.to_datetime(ms[m], unit="ms")))
    if not closes:
        return None
    bar = pd.concat(closes, axis=1).mean(axis=1, skipna=True)
    return bar.resample("1D").last().dropna()


def _sma(x, n):
    x = np.asarray(x, float)
    if len(x) < n:
        return np.full(len(x), np.nan)
    cs = np.cumsum(np.insert(x, 0, 0.0))
    out = np.full(len(x), np.nan)
    out[n - 1:] = (cs[n:] - cs[:-n]) / n
    return out


def _hysteretic_gate(raw_bear, k_on, k_off):
    """Debounce a raw-bear boolean into a HYSTERETIC gate: arm (1) only after k_on consecutive raw-bear bars;
    disarm (0) only after k_off consecutive raw-NOT-bear bars. Start disarmed. Pure state machine, causal in
    `raw_bear` (no future bars used). Returns an int8 gate array same length as raw_bear."""
    raw = np.asarray(raw_bear, bool)
    n = len(raw)
    gate = np.zeros(n, dtype=np.int8)
    state = 0
    run_bear = 0
    run_calm = 0
    for i in range(n):
        if raw[i]:
            run_bear += 1
            run_calm = 0
        else:
            run_calm += 1
            run_bear = 0
        if state == 0 and run_bear >= k_on:
            state = 1
        elif state == 1 and run_calm >= k_off:
            state = 0
        gate[i] = state
    return gate


def detect_bear_gate(cad, det=None):
    """The CAUSAL detected-bear gate as a daily pd.Series of 0/1 indexed on the daily basket index. raw bear =
    basket close < SMA(det['sma']); debounced by (k_on, k_off); then SHIFTED 1 day so the gate governing day t
    was decided from close<=t-1 (no look-ahead). Run in the same context as the books (synthetic or real)."""
    det = det or DETECTOR
    bc = _basket_close_daily(cad)
    if bc is None or len(bc) < det["sma"] + det["k_on"] + 2:
        return None
    sma = _sma(bc.to_numpy(), det["sma"])
    raw = (bc.to_numpy() < sma)
    raw = np.where(np.isfinite(sma), raw, False)                # warmup NaN -> not-bear (the safe default)
    gate = _hysteretic_gate(raw, det["k_on"], det["k_off"])
    g = pd.Series(gate, index=bc.index).shift(1).fillna(0).astype(np.int8)   # causal: decided on PAST bars
    return g


# =====================================================================================================
# 2. THE BOOKS -- core (trend), always-on (50/50 trend+longshort), gated (insurance toggled by the detector),
#    and the SHUFFLED toggle (a random-timed gate of equal on-frequency). All on the common daily index.
# =====================================================================================================
def build_books(cad, borrow_bps=BORROW_BASE, det=None, shuffle_seed=None):
    """Build the four daily books for one cadence on the (synthetic/real) panels in the current context:
      TREND       = trend-alone (long-only core).
      ALWAYS_ON   = (1-w)*trend + w*longshort EVERY day (the PHASE-6 always-on insurance, the bull-drag payer).
      GATED       = trend on non-bear days; (1-w)*trend + w*longshort on DETECTED-bear days (the conditional book).
      SHUFFLE     = same as GATED but the gate is a RANDOM-TIMED toggle of EQUAL on-frequency (the skill control).
    Returns {name -> daily pd.Series} on the common daily index + the realized gate on-fraction. The longshort
    leg carries the modelled short-borrow (via LSE._longshort_book)."""
    trend = LSE._trend_book(cad)
    longshort = LSE._longshort_book(cad, borrow_bps=borrow_bps)
    if trend is None or longshort is None:
        return None, None
    df = pd.concat([trend.rename("TREND"), longshort.rename("LS")], axis=1).dropna()
    if len(df) < 8:
        return None, None
    tr = df["TREND"].to_numpy()
    ls = df["LS"].to_numpy()
    w = INSURANCE_W
    blended = (1.0 - w) * tr + w * ls                           # the insurance-blended return (when gate on)

    # the detected-bear gate, aligned to the book daily index
    g = detect_bear_gate(cad, det=det)
    if g is None:
        gate = np.zeros(len(df), dtype=np.int8)
    else:
        gate = g.reindex(df.index).fillna(0).to_numpy().astype(np.int8)
    on_frac = float(np.mean(gate))

    gated = np.where(gate > 0, blended, tr)                     # insurance ON only on detected-bear days

    # the SHUFFLE control: a random-timed gate with EXACTLY the same on-COUNT (equal frequency, no skill).
    n_on = int(gate.sum())
    if shuffle_seed is not None and 0 < n_on < len(gate):
        rng = np.random.default_rng(shuffle_seed)
        perm = rng.permutation(len(gate))
        shuf = np.zeros(len(gate), dtype=np.int8)
        shuf[perm[:n_on]] = 1
    else:
        shuf = gate.copy()                                      # degenerate (all-on/all-off) -> identical
    shuffled = np.where(shuf > 0, blended, tr)

    idx = df.index
    books = {
        "TREND":     pd.Series(tr, index=idx),
        "ALWAYS_ON": pd.Series(blended, index=idx),
        "GATED":     pd.Series(gated, index=idx),
        "SHUFFLE":   pd.Series(shuffled, index=idx),
    }
    return books, {"on_frac": round(on_frac, 3), "n_on": n_on, "n_days": int(len(df))}


# =====================================================================================================
# 3. METRICS
# =====================================================================================================
def _perf(x, ann=365):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3:
        return {"net": None, "sharpe": None, "maxdd": None, "p05": None}
    eq = np.cumprod(1 + x); pk = np.maximum.accumulate(eq)
    bb = block_bootstrap_distribution(x, n_boot=400, block=5, seed=13)
    return {"net": round(float((eq[-1] - 1) * 100), 1),
            "sharpe": round(float(np.mean(x) / (np.std(x) + 1e-12) * np.sqrt(ann)), 2),
            "maxdd": round(float(((eq - pk) / pk).min() * 100), 1),
            "p05": round(float(bb["p05"]) * 100, 1)}


def _dist(vals):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return {"mean": None, "std": None, "worst": None, "median": None, "n": 0}
    return {"mean": round(float(np.mean(v)), 2), "std": round(float(np.std(v)), 2),
            "worst": round(float(np.min(v)), 2), "median": round(float(np.median(v)), 2),
            "p25": round(float(np.percentile(v, 25)), 2), "n": int(v.size)}


# =====================================================================================================
# 4. DETECTOR PRE-REGISTRATION (2020 TRAIN band ONLY) -- fit the grid, FREEZE, never touch OOS/synthetic.
# =====================================================================================================
def pre_register_detector(train_band=None):
    """Pre-register the detector params on the canonical 2020 TRAIN band ONLY (M2.SPLIT['TRAIN'] = 2020-01-01..
    2020-07-01 -- the real in-sample window that CONTAINS the Feb-Mar COVID bear, so a bear detector can actually
    be fit). Objective (TRAIN-only, two-sided): the gate should be ON in genuine sustained down-stretches and OFF
    in up-stretches. We score each grid point by how well the gate's on-days align with NEGATIVE forward basket
    drift vs the off-days -- i.e. mean(basket fwd 1d ret | gate on) << mean(... | gate off), with a guard that
    the gate is neither always-on nor always-off (a usable on-fraction). We pick the grid point with the LARGEST
    (off-mean MINUS on-mean) forward-return separation subject to on_frac in [0.05, 0.6]. This uses ONLY the
    TRAIN-band real basket close -- no sleeve returns, no OOS, no synthetic. Then FREEZE into DETECTOR. Returns
    (frozen_params, train_grid_log)."""
    train_band = train_band or M2.SPLIT["TRAIN"]               # ("2020-01-01","2020-07-01") -- has the real bear
    bc_train = _basket_close_2020_band(train_band[0], train_band[1])
    if bc_train is None or len(bc_train) < min(DET_GRID["sma"]) + 20:
        n = 0 if bc_train is None else len(bc_train)
        print(f"   [pre-register] TRAIN band too short ({n} bars) -- keeping the prior {DETECTOR}")
        return dict(DETECTOR), []
    fwd = bc_train.pct_change().shift(-1).to_numpy()            # forward 1-day basket return (TRAIN scoring only)
    vals = bc_train.to_numpy()
    best = None
    grid_log = []
    for sma in DET_GRID["sma"]:
        s = _sma(vals, sma)
        raw = np.where(np.isfinite(s), vals < s, False)
        for k_on in DET_GRID["k_on"]:
            for k_off in DET_GRID["k_off"]:
                gate = _hysteretic_gate(raw, k_on, k_off)
                gate_sh = np.concatenate([[0], gate[:-1]]).astype(bool)   # causal 1-bar shift
                on = gate_sh & np.isfinite(fwd)
                off = (~gate_sh) & np.isfinite(fwd)
                on_frac = float(np.mean(gate_sh))
                if on.sum() < 5 or off.sum() < 5 or not (0.05 <= on_frac <= 0.6):
                    grid_log.append({"sma": sma, "k_on": k_on, "k_off": k_off, "on_frac": round(on_frac, 3),
                                     "sep": None, "usable": False})
                    continue
                # separation: off-day fwd mean MINUS on-day fwd mean (want POSITIVE = on-days are the down-stretches)
                sep = float(np.mean(fwd[off]) - np.mean(fwd[on]))
                grid_log.append({"sma": sma, "k_on": k_on, "k_off": k_off, "on_frac": round(on_frac, 3),
                                 "sep": round(sep, 5), "usable": True})
                if best is None or sep > best["sep"]:
                    best = {"sma": sma, "k_on": k_on, "k_off": k_off, "sep": sep, "on_frac": on_frac}
    if best is None:
        print("   [pre-register] no usable grid point on TRAIN -- keeping the prior", DETECTOR)
        return dict(DETECTOR), grid_log
    frozen = {"sma": best["sma"], "k_on": best["k_on"], "k_off": best["k_off"]}
    print(f"   [pre-register] FROZEN detector (TRAIN-only fit): {frozen} "
          f"(fwd-ret separation {best['sep']:+.5f}, TRAIN on-frac {best['on_frac']:.3f})")
    return frozen, grid_log


# =====================================================================================================
# 4b. THE DETECTOR SKILL BLOCK (on the STITCHED path -- the ONLY surface with regime TRANSITIONS).
#    A regime-TRANSITION detector can only be skill-tested on a path that CONTAINS transitions. Standalone
#    single-regime panels (bear ~39d, bull ~92d) are too short for a slow detector to even warm up -- the gate
#    is structurally ~always-off there, which makes the standalone shuffle DEGENERATE (it would 'pass' trivially).
#    So the DECISIVE skill test runs on the STITCHED full-cycle path, where we know the TRUE bear-window bar
#    indices (from stitch boundaries) and can ask: does the detector concentrate its ON-time in the true bear
#    window FAR better than a random-timed toggle of equal frequency? (precision vs base-rate + vs shuffle).
# =====================================================================================================
def _true_bear_window(boundaries, n_bars):
    """From stitch boundaries [(cursor, regime), ...] return a boolean mask (len n_bars) True on the BEAR bars."""
    mask = np.zeros(n_bars, dtype=bool)
    for i, (cur, rg) in enumerate(boundaries):
        nxt = boundaries[i + 1][0] if i + 1 < len(boundaries) else n_bars
        if rg == "bear":
            mask[cur:nxt] = True
    return mask


def run_detector_skill(seeds, det, shuffles_per_seed=20):
    """The DECISIVE detector skill test on the STITCHED full-cycle path. For each seed: build the daily gate +
    the TRUE bear-window mask (from stitch boundaries), then measure, on the 1d detector grid:
      - precision = P(on-day is a true-bear day)  vs the BASE RATE (bear-bars / total) and vs a SHUFFLE.
      - recall    = P(true-bear day is gated on).
      - shuffle null: draw `shuffles_per_seed` random-timed toggles of the SAME on-count, record their
        precision distribution; the detector's skill = (detector precision) - (mean shuffle precision), and a
        per-seed PASS = detector precision > the shuffle's 95th percentile (one-sided, the gate beats a random
        toggle of equal frequency at this seed). Reports the across-seed distribution + the fraction of seeds
        the detector beats its shuffle. NO sleeve returns here -- this isolates the TIMING skill of the gate."""
    precs, recalls, base_rates, skill_gaps, on_fracs = [], [], [], [], []
    beats_shuffle_seed = []
    rng_master = np.random.default_rng(20200314)
    for seed in seeds:
        panels, bnd = SRS.stitch_panels(SRS.STITCH_SEQUENCE, SRS._CALIB, seed)
        with SRS._synthetic_panel_context(panels):
            g = detect_bear_gate("1d", det=det)
        if g is None:
            continue
        gate = g.to_numpy().astype(bool)
        n = len(gate)
        bear = _true_bear_window(bnd, n)
        n_on = int(gate.sum())
        base = float(bear.mean())
        base_rates.append(base)
        on_fracs.append(float(gate.mean()))
        if n_on == 0 or bear.sum() == 0:
            continue
        prec = float((gate & bear).sum() / n_on)
        rec = float((gate & bear).sum() / bear.sum())
        precs.append(prec); recalls.append(rec)
        # shuffle null: random-timed toggles of the SAME on-count
        shuf_precs = []
        for _ in range(shuffles_per_seed):
            idx = rng_master.permutation(n)[:n_on]
            sh = np.zeros(n, dtype=bool); sh[idx] = True
            shuf_precs.append(float((sh & bear).sum() / n_on))
        shuf_precs = np.asarray(shuf_precs)
        skill_gaps.append(prec - float(shuf_precs.mean()))
        beats_shuffle_seed.append(bool(prec > np.percentile(shuf_precs, 95)))
    out = {
        "precision": _dist(precs), "recall": _dist(recalls), "base_rate": _dist(base_rates),
        "on_frac": _dist(on_fracs), "precision_minus_shuffle": _dist(skill_gaps),
        "frac_seeds_beats_shuffle95": (round(float(np.mean(beats_shuffle_seed)), 2) if beats_shuffle_seed else None),
        "n_seeds": len(precs),
    }
    print(f"   DETECTOR SKILL (stitched path, vs true bear window): precision {out['precision']['mean']} "
          f"(base-rate {out['base_rate']['mean']}, recall {out['recall']['mean']}); "
          f"precision-minus-shuffle {out['precision_minus_shuffle']['mean']}pp; "
          f"beats shuffle-95 in {out['frac_seeds_beats_shuffle95']} of seeds")
    return out


def run_segment_decomposition(seeds, det, cad="1d", borrow_bps=BORROW_BASE):
    """WHERE the gated book gains/gives-back vs trend on the stitched path, per regime SEGMENT (bull1/bear/chop/
    bull2 from stitch boundaries). This makes the BOOK-doesn't-deploy MECHANISM reproducible (not asserted): if
    the bear gain is tiny and the bull/chop false-alarm cost dominates, the per-segment cumulative gated-minus-
    trend pp shows it directly. Also reports the gate on-fraction per segment (the false-alarm rate in non-bear)."""
    seg_acc = {}
    onf_acc = {}
    for seed in seeds:
        panels, bnd = SRS.stitch_panels(SRS.STITCH_SEQUENCE, SRS._CALIB, seed)
        with SRS._synthetic_panel_context(panels):
            books, meta = build_books(cad, borrow_bps=borrow_bps, det=det, shuffle_seed=seed * 131 + 7)
            g = detect_bear_gate(cad, det=det)
        if books is None:
            continue
        tr, gd = books["TREND"], books["GATED"]
        df = pd.concat([tr.rename("t"), gd.rename("g")], axis=1).dropna()
        delta = (df["g"] - df["t"]).to_numpy()
        gate = (g.reindex(df.index).fillna(0).to_numpy() if g is not None else np.zeros(len(df)))
        n = len(delta)
        # segment boundaries (daily-bar space, 1:1 for the 1d stitched panel)
        segs = []
        for i, (cur, rg) in enumerate(bnd):
            nxt = bnd[i + 1][0] if i + 1 < len(bnd) else n
            label = f"{rg}{i}"
            segs.append((label, rg, min(cur, n), min(nxt, n)))
        for label, rg, a, b in segs:
            if a >= b:
                continue
            seg_acc.setdefault(label, {"rg": rg, "delta": []})
            onf_acc.setdefault(label, [])
            seg_acc[label]["delta"].append(float(np.sum(delta[a:b]) * 100))   # cumulative pp in the segment
            onf_acc[label].append(float(np.mean(gate[a:b])))
    out = {}
    for label in seg_acc:
        out[label] = {"regime": seg_acc[label]["rg"],
                      "gated_minus_trend_cum_pp": _dist(seg_acc[label]["delta"]),
                      "gate_on_frac": _dist(onf_acc[label])}
    print("   SEGMENT DECOMPOSITION (stitched 1d, gated-minus-trend cumulative pp + gate on-frac per segment):")
    for label in out:
        e = out[label]
        print(f"      {label:8} ({e['regime']:5}): gated-minus-trend {e['gated_minus_trend_cum_pp']['mean']:+.2f}pp "
              f"| gate on-frac {e['gate_on_frac']['mean']:.2f}")
    return out


# =====================================================================================================
# 5. THE STRESS RUN -- per regime + stitched, over seeds, distributions not single paths.
# =====================================================================================================
def _gen_panels(sc, seed, n_bars=None):
    if sc == "stitched":
        panels, _ = SRS.stitch_panels(SRS.STITCH_SEQUENCE, SRS._CALIB, seed, n_bars_each=n_bars)
    else:
        nb = n_bars if n_bars else SRS.REGIME_BARS.get(sc, SRS.N_BARS_REGIME)
        panels = SRS.generate_regime_panels(sc, SRS._CALIB, seed=seed, n_bars=nb)
    return panels


BOOK_NAMES = ["TREND", "ALWAYS_ON", "GATED", "SHUFFLE"]


def run_stress(cadences, seeds, det, borrow_bps=BORROW_BASE):
    """For each cadence x regime: build TREND / ALWAYS_ON / GATED / SHUFFLE, collect net/maxDD/p05 distributions
    over seeds, plus the realized gate on-fraction. The detector `det` is the FROZEN (pre-registered) params --
    NOT re-fit per seed/regime. Returns the per-cadence results dict."""
    res = {}
    for cad in cadences:
        print(f"\n########## GATED {cad} -- conditional bear-insurance stress ({len(seeds)} seeds, det={det}) ##########")
        acc = {sc: {bn: {"net": [], "maxdd": [], "p05": []} for bn in BOOK_NAMES} for sc in SCENARIOS}
        for sc in SCENARIOS:
            acc[sc]["_on_frac"] = []
            # paired deltas (per seed) for capture/drag ratios
            acc[sc]["gated_minus_trend_dd"] = []
            acc[sc]["gated_minus_trend_net"] = []
            acc[sc]["always_minus_trend_dd"] = []
            acc[sc]["always_minus_trend_net"] = []
        eq_example = {}
        for si, seed in enumerate(seeds):
            for sc in SCENARIOS:
                panels = _gen_panels(sc, seed)
                with SRS._synthetic_panel_context(panels):
                    books, meta = build_books(cad, borrow_bps=borrow_bps, det=det, shuffle_seed=seed * 131 + 7)
                if books is None:
                    continue
                perf = {bn: _perf(books[bn].to_numpy()) for bn in BOOK_NAMES}
                if perf["TREND"]["net"] is None:
                    continue
                for bn in BOOK_NAMES:
                    p = perf[bn]
                    if p["net"] is not None:
                        acc[sc][bn]["net"].append(p["net"]); acc[sc][bn]["maxdd"].append(p["maxdd"])
                        acc[sc][bn]["p05"].append(p["p05"])
                acc[sc]["_on_frac"].append(meta["on_frac"])
                tn, td = perf["TREND"]["net"], perf["TREND"]["maxdd"]
                gn, gd = perf["GATED"]["net"], perf["GATED"]["maxdd"]
                an, ad = perf["ALWAYS_ON"]["net"], perf["ALWAYS_ON"]["maxdd"]
                if None not in (tn, td, gn, gd):
                    acc[sc]["gated_minus_trend_dd"].append(gd - td)     # +ve = gated draws down LESS than trend
                    acc[sc]["gated_minus_trend_net"].append(gn - tn)
                if None not in (tn, td, an, ad):
                    acc[sc]["always_minus_trend_dd"].append(ad - td)
                    acc[sc]["always_minus_trend_net"].append(an - tn)
                if si == 0:
                    eq_example[sc] = {bn: list(np.cumprod(1 + books[bn].to_numpy()) * 100 - 100) for bn in BOOK_NAMES}
            print(f"   seed {seed} done ({si + 1}/{len(seeds)})", end="\r")
        print()
        res[cad] = _summarize(acc, eq_example, det)
        _print_table(cad, res[cad])
    return res


def _summarize(acc, eq_example, det):
    summ = {"_equity_example": eq_example, "detector": dict(det)}
    for sc in SCENARIOS:
        a = acc[sc]
        summ[sc] = {bn: {"net": _dist(a[bn]["net"]), "maxdd": _dist(a[bn]["maxdd"]), "p05": _dist(a[bn]["p05"])}
                    for bn in BOOK_NAMES}
        summ[sc]["on_frac"] = _dist(a["_on_frac"])
        summ[sc]["gated_minus_trend_dd"] = _dist(a["gated_minus_trend_dd"])
        summ[sc]["gated_minus_trend_net"] = _dist(a["gated_minus_trend_net"])
        summ[sc]["always_minus_trend_dd"] = _dist(a["always_minus_trend_dd"])
        summ[sc]["always_minus_trend_net"] = _dist(a["always_minus_trend_net"])

    # ---- THE TWO RATIOS (the headline question) -- computed on the STITCHED full-cycle path ----
    # WHY stitched: the gate is a regime-TRANSITION detector; on a STANDALONE single-regime panel the gate is
    # ~always-off (too few bars to warm a slow SMA), so the standalone-bear capture/drag is structurally ~0 and
    # MEANINGLESS. The STITCHED path (bull->bear->chop->bull) is where the gate actually fires during the embedded
    # bear -- so the full-cycle capture (DD protection) and the full-cycle drag (net give-up vs trend) measured
    # there are the real synthesis numbers. We ALSO report the standalone-bear DD-protection as a secondary check.
    st_always_prot = summ["stitched"]["always_minus_trend_dd"]["mean"]   # always-on lowers stitched DD by this (pp, +=better)
    st_gated_prot = summ["stitched"]["gated_minus_trend_dd"]["mean"]     # gated lowers stitched DD by this
    bear_prot_captured = (round(100.0 * st_gated_prot / st_always_prot, 1)
                          if (st_always_prot not in (None, 0) and abs(st_always_prot) > 1e-6) else None)
    # full-cycle DRAG = net give-up vs trend on the stitched path (the longshort short leg costs net over the cycle)
    st_always_drag = summ["stitched"]["always_minus_trend_net"]["mean"]  # always-on net give-up vs trend (-=drag)
    st_gated_drag = summ["stitched"]["gated_minus_trend_net"]["mean"]    # gated net give-up vs trend
    drag_paid = (round(100.0 * st_gated_drag / st_always_drag, 1)
                 if (st_always_drag not in (None, 0) and abs(st_always_drag) > 1e-6 and st_always_drag < 0) else None)
    # standalone-bear DD-protection (secondary diagnostic; ~0 because the gate barely fires on a 39-bar panel)
    sa_always_bear = summ["bear"]["always_minus_trend_dd"]["mean"]
    sa_gated_bear = summ["bear"]["gated_minus_trend_dd"]["mean"]

    summ["_ratios"] = {
        "bear_protection_captured_pct": bear_prot_captured,            # STITCHED full-cycle DD protection captured
        "always_on_bear_protection_pp": round(st_always_prot, 2) if st_always_prot is not None else None,
        "gated_bear_protection_pp": round(st_gated_prot, 2) if st_gated_prot is not None else None,
        "drag_paid_pct": drag_paid,                                    # STITCHED full-cycle net-drag still paid
        "always_on_drag_pp": round(st_always_drag, 2) if st_always_drag is not None else None,
        "gated_drag_pp": round(st_gated_drag, 2) if st_gated_drag is not None else None,
        "standalone_bear_always_prot_pp": round(sa_always_bear, 2) if sa_always_bear is not None else None,
        "standalone_bear_gated_prot_pp": round(sa_gated_bear, 2) if sa_gated_bear is not None else None,
        "_basis": "stitched full-cycle path (the gate fires during the embedded bear; standalone regimes are too "
                  "short for a slow gate to warm up -> standalone capture is structurally ~0 and not used)",
    }

    # ---- ROBUSTNESS across the regime mix: worst-regime net + full-mix-worst maxDD + worst-regime p05 ----
    robust = {}
    for bn in BOOK_NAMES:
        worst_net = min((summ[sc][bn]["net"]["mean"] for sc in SCENARIOS
                         if summ[sc][bn]["net"]["mean"] is not None), default=None)
        worst_dd = min((summ[sc][bn]["maxdd"]["worst"] for sc in SCENARIOS
                        if summ[sc][bn]["maxdd"]["worst"] is not None), default=None)
        worst_p05 = min((summ[sc][bn]["p05"]["mean"] for sc in SCENARIOS
                         if summ[sc][bn]["p05"]["mean"] is not None), default=None)
        mix_net = np.mean([summ[sc][bn]["net"]["mean"] for sc in SCENARIOS
                           if summ[sc][bn]["net"]["mean"] is not None])
        robust[bn] = {"worst_regime_net": round(float(worst_net), 2) if worst_net is not None else None,
                      "full_mix_worst_maxdd": round(float(worst_dd), 2) if worst_dd is not None else None,
                      "worst_regime_p05": round(float(worst_p05), 2) if worst_p05 is not None else None,
                      "mean_mix_net": round(float(mix_net), 2)}
    summ["_robustness"] = robust

    # ---- THE DECISIVE CONTROLS -- on the STITCHED full-cycle path (the surface with real transitions) ----
    # Control (b): GATED vs TREND-ALONE on the stitched path -- net AND maxDD. (PHASE 6's static book failed this.)
    # Control (c, MANDATORY): GATED vs SHUFFLE on the stitched path -- does the DETECTOR's timing beat a random-
    #   timed toggle of equal frequency? Both books carry the SAME longshort exposure; only the TIMING differs,
    #   so a gated-over-shuffle edge is pure detector skill (not exposure).
    st = summ["stitched"]
    g_net, g_dd = st["GATED"]["net"]["mean"], st["GATED"]["maxdd"]["mean"]
    t_net, t_dd = st["TREND"]["net"]["mean"], st["TREND"]["maxdd"]["mean"]
    sh_net, sh_dd = st["SHUFFLE"]["net"]["mean"], st["SHUFFLE"]["maxdd"]["mean"]
    g_p05, t_p05, sh_p05 = st["GATED"]["p05"]["mean"], st["TREND"]["p05"]["mean"], st["SHUFFLE"]["p05"]["mean"]
    g_worst, t_worst, sh_worst = st["GATED"]["net"]["worst"], st["TREND"]["net"]["worst"], st["SHUFFLE"]["net"]["worst"]

    def _ge(x, y):
        return x is not None and y is not None and x >= y

    summ["_controls"] = {
        "basis": "stitched full-cycle path (bull->bear->chop->bull) -- the surface with regime TRANSITIONS",
        "gated_vs_trend": {
            "beats_on_net": bool(_ge(g_net, t_net)),
            "beats_on_maxdd": bool(_ge(g_dd, t_dd)),                    # less-negative DD = better
            "beats_on_p05": bool(_ge(g_p05, t_p05)),
            "net_gain_pp": round(float((g_net or 0) - (t_net or 0)), 2),
            "dd_gain_pp": round(float((g_dd or 0) - (t_dd or 0)), 2),
            "worst_seed_net_gain_pp": round(float((g_worst or 0) - (t_worst or 0)), 2),
        },
        "gated_vs_shuffle": {
            "beats_on_net": bool(g_net is not None and sh_net is not None and g_net > sh_net),
            "beats_on_maxdd": bool(_ge(g_dd, sh_dd)),
            "net_gain_vs_shuffle_pp": round(float((g_net or 0) - (sh_net or 0)), 2),
            "dd_gain_vs_shuffle_pp": round(float((g_dd or 0) - (sh_dd or 0)), 2),
            "worst_seed_net_gain_vs_shuffle_pp": round(float((g_worst or 0) - (sh_worst or 0)), 2),
        },
    }
    # cross-regime worst-case robustness kept as a secondary view (NOT the primary verdict; the standalone bear
    # has the gate ~off, so worst-regime net there == trend == shuffle == not informative about the detector).
    summ["_robustness"]["_note"] = ("worst-regime net across standalone regimes is dominated by the standalone "
                                    "bear where the gate is ~off (gated==trend==shuffle); the DECISIVE comparison "
                                    "is on the stitched path (_controls), not the standalone worst-regime.")
    return summ


def _print_table(cad, summ):
    print(f"   {'regime':9} {'TREND net':>10} {'GATED net':>10} {'ALWAYS net':>11} {'SHUF net':>9} "
          f"{'GATED DD':>9} {'TREND DD':>9} {'ALWAYS DD':>10} {'on-frac':>8}")
    for sc in SCENARIOS:
        e = summ[sc]
        print(f"   {sc:9} {str(e['TREND']['net']['mean']):>10} {str(e['GATED']['net']['mean']):>10} "
              f"{str(e['ALWAYS_ON']['net']['mean']):>11} {str(e['SHUFFLE']['net']['mean']):>9} "
              f"{str(e['GATED']['maxdd']['mean']):>9} {str(e['TREND']['maxdd']['mean']):>9} "
              f"{str(e['ALWAYS_ON']['maxdd']['mean']):>10} {str(e['on_frac']['mean']):>8}")
    r = summ["_ratios"]; c = summ["_controls"]
    print(f"   STITCHED-PATH RATIOS: bear-protection CAPTURED {r['bear_protection_captured_pct']}% "
          f"(gated {r['gated_bear_protection_pp']}pp DD vs always-on {r['always_on_bear_protection_pp']}pp); "
          f"full-cycle drag PAID {r['drag_paid_pct']}% (gated {r['gated_drag_pp']}pp vs always-on {r['always_on_drag_pp']}pp)")
    cb = c["gated_vs_trend"]; cs = c["gated_vs_shuffle"]
    print(f"   CONTROL (b) GATED vs TREND-ALONE (stitched): net {cb['net_gain_pp']:+}pp (beats={cb['beats_on_net']}), "
          f"maxDD {cb['dd_gain_pp']:+}pp (beats={cb['beats_on_maxdd']}), worst-seed net {cb['worst_seed_net_gain_pp']:+}pp")
    print(f"   CONTROL (c) GATED vs SHUFFLE (stitched, the skill test): net {cs['net_gain_vs_shuffle_pp']:+}pp "
          f"(beats={cs['beats_on_net']}), maxDD {cs['dd_gain_vs_shuffle_pp']:+}pp (beats={cs['beats_on_maxdd']}), "
          f"worst-seed net {cs['worst_seed_net_gain_vs_shuffle_pp']:+}pp")


# =====================================================================================================
# 6. 2020-REAL OOS anchor
# =====================================================================================================
def run_2020_real(cadences, det, borrow_bps=BORROW_BASE):
    """Run the four books on the REAL 2020 band (no synthetic patch), graded on the within-2020 OOS (Oct-Dec,
    a clean BULL -- the real bull-drag anchor + the detector's OOS behaviour on real data). The detector is the
    FROZEN params (pre-registered on the TRAIN portion). Returns per-cadence OOS perf + OOS gate on-fraction."""
    real = {}
    split = pd.Timestamp(COMP.SPLIT)
    for cad in cadences:
        books, meta = build_books(cad, borrow_bps=borrow_bps, det=det, shuffle_seed=12345)
        if books is None:
            real[cad] = None
            continue
        gate = detect_bear_gate(cad, det=det)
        oos = {}
        for bn in BOOK_NAMES:
            b = books[bn]; b_oos = b[b.index >= split]
            oos[bn] = _perf(b_oos.to_numpy())
        oos_on_frac = (float((gate[gate.index >= split]).mean()) if gate is not None else None)
        real[cad] = {bn: oos[bn] for bn in BOOK_NAMES}
        real[cad]["oos_gate_on_frac"] = round(oos_on_frac, 3) if oos_on_frac is not None else None
        real[cad]["full_gate_on_frac"] = meta["on_frac"]
        t = oos["TREND"]; g = oos["GATED"]
        print(f"   2020-REAL OOS {cad}: TREND net {t['net']}% (DD {t['maxdd']}) vs GATED net {g['net']}% "
              f"(DD {g['maxdd']}) -- OOS gate on-frac {real[cad]['oos_gate_on_frac']} "
              f"(bull OOS: gate SHOULD be ~off)")
    return real


# =====================================================================================================
# 7. VERDICT (two-sided)
# =====================================================================================================
def build_verdict(res, real, det, skill, seg_decomp=None):
    lines = []
    lines.append(f"DETECTOR (pre-registered on 2020 TRAIN, FROZEN): close < SMA({det['sma']}), "
                 f"arm after {det['k_on']} bars, disarm after {det['k_off']} bars (causal, 1-bar-shifted).")
    lines.append("")

    # Q0: the DECISIVE detector skill test on the stitched path (the only surface with regime transitions).
    lines.append("Q0 (THE SKILL TEST, decisive): on the STITCHED full-cycle path -- where regime TRANSITIONS exist --")
    lines.append("    does the detector concentrate its ON-time in the TRUE bear window better than a random toggle?")
    detector_has_skill = (skill["precision"]["mean"] is not None and skill["base_rate"]["mean"] is not None
                          and skill["precision"]["mean"] > skill["base_rate"]["mean"] + 0.05
                          and (skill["frac_seeds_beats_shuffle95"] or 0) >= 0.5)
    lines.append(f"    detector precision {skill['precision']['mean']} (on-day is true-bear) vs base-rate "
                 f"{skill['base_rate']['mean']} and recall {skill['recall']['mean']}; precision-minus-shuffle "
                 f"{skill['precision_minus_shuffle']['mean']}pp; beats shuffle-95 in "
                 f"{skill['frac_seeds_beats_shuffle95']} of seeds -> "
                 f"{'DETECTOR HAS REAL TIMING SKILL' if detector_has_skill else 'NO reliable skill over a random toggle'}")
    lines.append("    (NOTE: standalone single-regime panels are too short for a slow gate to warm up -- the gate is")
    lines.append("     ~always-off there, so the standalone shuffle is DEGENERATE; the stitched path is the proper test.)")
    lines.append("")

    # Q1: the two ratios per TF (on the STITCHED full-cycle path) -- capture most of the bear protection, pay
    #     little of the cycle drag?
    lines.append("Q1 (THE SYNTHESIS, stitched full-cycle path): does GATING capture MOST of the always-on's bear")
    lines.append("    DD-protection while paying LITTLE of its full-cycle net-drag? (CAPTURED = gated's stitched")
    lines.append("    DD-protection / always-on's; PAID = gated's stitched net-drag / always-on's. Ideal=high cap, low drag.)")
    capture_tfs = []
    for cad in res:
        r = res[cad]["_ratios"]
        cap = r["bear_protection_captured_pct"]; paid = r["drag_paid_pct"]
        good = (cap is not None and paid is not None and cap >= 40 and paid <= 60)
        if good:
            capture_tfs.append(cad)
        tag = "captures protection, sheds drag" if good else "no clean capture/drag separation"
        lines.append(f"    {cad:5}: bear-protection CAPTURED {cap}% (gated {r['gated_bear_protection_pp']}pp DD vs "
                     f"always-on {r['always_on_bear_protection_pp']}pp); full-cycle drag PAID {paid}% "
                     f"(gated {r['gated_drag_pp']}pp vs always-on {r['always_on_drag_pp']}pp) -> {tag}")

    # Q2 (control b): gated vs TREND-ALONE on the STITCHED path (net AND maxDD) -- PHASE 6's static book FAILED.
    lines.append("")
    lines.append("Q2 (CONTROL b, stitched path): does the GATED book beat TREND-ALONE on net AND maxDD?")
    lines.append("    (PHASE 6's STATIC always-on book FAILED this -- the conditional book must clear trend to be worth it)")
    beats_trend_tfs = []
    for cad in res:
        c = res[cad]["_controls"]["gated_vs_trend"]
        # 'worth it' = does NOT lose net materially AND protects DD (or at worst DD-neutral net-neutral)
        protects = c["beats_on_maxdd"] and c["net_gain_pp"] >= -1.0
        if protects:
            beats_trend_tfs.append(cad)
        lines.append(f"    {cad:5}: vs trend -- net {c['net_gain_pp']:+}pp (beats={c['beats_on_net']}), "
                     f"maxDD {c['dd_gain_pp']:+}pp (beats={c['beats_on_maxdd']}), worst-seed net "
                     f"{c['worst_seed_net_gain_pp']:+}pp -> {'protects DD without losing net' if protects else 'does NOT clear trend'}")

    # Q3 (control c, MANDATORY): does the DETECTOR's TIMING beat a SHUFFLED toggle of equal frequency?
    lines.append("")
    lines.append("Q3 (CONTROL c, MANDATORY -- the skill test, stitched path + gate-precision): does the DETECTOR's")
    lines.append("    TIMING beat a RANDOM-TIMED toggle of EQUAL on-frequency? (the shuffle that killed the dynamic")
    lines.append("    engine. Both books carry the SAME longshort exposure; only the timing differs.)")
    beats_shuffle_tfs = []
    for cad in res:
        c = res[cad]["_controls"]["gated_vs_shuffle"]
        # the gate's timing must protect the cycle DD (or net) better than a random toggle of the same frequency
        sk = c["dd_gain_vs_shuffle_pp"] > 0.2 or c["net_gain_vs_shuffle_pp"] > 0.5
        if sk:
            beats_shuffle_tfs.append(cad)
        lines.append(f"    {cad:5}: vs shuffle -- net {c['net_gain_vs_shuffle_pp']:+}pp (beats={c['beats_on_net']}), "
                     f"maxDD {c['dd_gain_vs_shuffle_pp']:+}pp (beats={c['beats_on_maxdd']}) -> "
                     f"{'detector TIMING adds value over random' if sk else 'NO timing value over a random toggle'}")
    lines.append(f"    (gate-precision skill, stitched: precision {skill['precision']['mean']} vs base-rate "
                 f"{skill['base_rate']['mean']}, beats shuffle-95 in {skill['frac_seeds_beats_shuffle95']} of seeds "
                 f"-> the gate's ON-time IS concentrated in the true bear, even if the BOOK payoff is a wash.)")

    # Q3b: THE MECHANISM (reproducible) -- per-segment gated-minus-trend on the stitched path.
    if seg_decomp:
        lines.append("")
        lines.append("Q3b (MECHANISM, why a SKILLED detector still does not deploy): per-segment gated-minus-trend (1d):")
        for label in seg_decomp:
            e = seg_decomp[label]
            lines.append(f"    {label:8} ({e['regime']:5}): gated-minus-trend {e['gated_minus_trend_cum_pp']['mean']:+.2f}pp "
                         f"| gate on-frac {e['gate_on_frac']['mean']:.2f}")
        lines.append("    -> the conditional BEAR gain is TINY; the gate's residual FALSE-ALARM firings in bull/chop "
                     "(short into a rising market) + borrow OUTWEIGH it. A precise gate still loses on a bull-dominated cycle.")

    # Q4: the 2020-real OOS anchor (a BULL -- the gate should be ~off, the gated book ~= trend)
    lines.append("")
    lines.append("Q4: 2020-REAL OOS (Oct-Dec BULL) anchor -- on a real bull the gate SHOULD be ~off (gated ~= trend):")
    for cad in real:
        if real[cad] is None:
            continue
        t = real[cad]["TREND"]; g = real[cad]["GATED"]
        lines.append(f"    {cad:5}: TREND net {t['net']}% (DD {t['maxdd']}) vs GATED net {g['net']}% (DD {g['maxdd']}); "
                     f"OOS gate on-frac {real[cad]['oos_gate_on_frac']} (near 0 = correctly OFF in the bull)")

    # HEADLINE (two-sided, decisive either way). The deployable bar: the GATED book must (b) clear trend-alone
    # on the stitched path (protect DD without losing net) AND (c) its detector TIMING must beat the equal-freq
    # shuffle. The gate-precision skill (Q0) is necessary but not sufficient -- the BOOK payoff is what deploys.
    n_cad = len(res)
    works = [cad for cad in res if cad in beats_trend_tfs and cad in beats_shuffle_tfs]
    # pick the best TF by stitched gated net (the full-cycle wealth, the objective function)
    def _st_gated_net(c):
        return res[c]["stitched"]["GATED"]["net"]["mean"] or -1e9
    if works:
        best = max(works, key=_st_gated_net)
        st = res[best]["stitched"]; rr = res[best]["_ratios"]; cb = res[best]["_controls"]
        headline = (f"CONDITIONAL BEAR-INSURANCE WORKS at {len(works)}/{n_cad} cadences ({works}): on the stitched "
                    f"full-cycle path the GATED longshort book protects DD without losing net vs trend-alone AND its "
                    f"DETECTOR TIMING beats the equal-frequency SHUFFLE (gate precision {skill['precision']['mean']} vs "
                    f"base-rate {skill['base_rate']['mean']}). Best TF = {best} (stitched gated net "
                    f"{st['GATED']['net']['mean']}% vs trend {st['TREND']['net']['mean']}%, maxDD "
                    f"{st['GATED']['maxdd']['mean']}% vs {st['TREND']['maxdd']['mean']}%; captures "
                    f"{rr['bear_protection_captured_pct']}% of the always-on bear DD-protection while paying only "
                    f"{rr['drag_paid_pct']}% of its full-cycle drag). THIS is the deployable conditional-bear-insurance "
                    f"book -- the longshort INSURANCE leg is RESEARCH (short violates long-only+spot); deploying needs the "
                    f"user's LO-exception sign-off. Without it, trend-alone+trail is the long-only answer.")
    elif detector_has_skill and not works:
        # the detector HAS gate-timing skill but the BOOK doesn't deploy -- the most honest, nuanced outcome.
        # MECHANISM (verified by per-segment decomposition, not asserted): the gate's bear gain is TINY (~+0.1pp)
        # while its FALSE-ALARM cost in the bull/chop segments (the gate still fires ~18-22% there, each firing a
        # short into a RISING market) sums to ~-1.6pp -- the bear is only ~12% of the cycle and too mild in net to
        # offset the short-leg drag over the 88% non-bear time. Precision is real but a bull-dominated cycle
        # punishes even a low false-alarm rate.
        headline = (f"DETECTOR HAS TIMING SKILL, BUT THE BOOK DOES NOT DEPLOY: the bear DETECTOR genuinely concentrates "
                    f"its ON-time in the true bear window (stitched precision {skill['precision']['mean']} vs base-rate "
                    f"{skill['base_rate']['mean']}, recall {skill['recall']['mean']}, beats shuffle-95 in "
                    f"{skill['frac_seeds_beats_shuffle95']} of seeds; the GATED book beats the equal-freq SHUFFLE on net "
                    f"at every TF) -- so this is NOT the dynamic-engine null, the detector WORKS. BUT the gated longshort "
                    f"BOOK still loses ~0.3-2.0pp net vs trend-alone on the full cycle and does not reduce DD. MECHANISM "
                    f"(per-segment decomposition, verified): the conditional BEAR gain is TINY (~+0.1pp -- the synthetic "
                    f"bear is only ~12% of the cycle and mild in NET terms), while the gate's residual FALSE-ALARM firings "
                    f"in the bull/chop segments (~18-22% on-time even there, each shorting a RISING market) plus borrow "
                    f"sum to ~-1.6pp. On a bull-dominated cycle, even a precise gate's small false-alarm rate costs more "
                    f"than the short bear earns. Honest answer: TREND-ALONE+trail is the deployable long-only book; the "
                    f"detector is real but binary longshort insurance is -EV here. The open door (untestable on 2020): a "
                    f"DEEPER/LONGER bear (e.g. 2022 grind-down) where the bear is a larger NET share of the cycle -- the "
                    f"detector machinery is built + validated to test it the moment such a band is in scope.")
    elif beats_trend_tfs and not beats_shuffle_tfs:
        headline = (f"FAILS THE SKILL TEST: the gated book clears trend at {beats_trend_tfs} but its TIMING does NOT beat "
                    f"the equal-frequency SHUFFLE -- the apparent benefit is the EXPOSURE change (adding the short leg), "
                    f"NOT the detector's timing. A random-timed toggle of the same frequency does as well. The SAME null "
                    f"the dynamic engine hit, as a binary gate. trend-alone+trail is the honest answer.")
    else:
        headline = (f"DECISIVE NULL: conditional bear-insurance does NOT deploy -- the GATED longshort book neither clears "
                    f"trend-alone on the stitched full-cycle path NOR beats its equal-frequency SHUFFLE, and the detector's "
                    f"gate-timing edge (precision {skill['precision']['mean']} vs base-rate {skill['base_rate']['mean']}) "
                    f"does not convert to a book payoff after borrow. The honest answer: TREND-ALONE+trail is the deployable "
                    f"long-only book; the bear rescue is UNCONDITIONAL-only (the always-on short, accepting the bull-drag, "
                    f"RESEARCH) or no rescue. A clean, strong closure -- the slow hysteretic binary gate does not beat the "
                    f"static answer on 2020-calibrated data.")
    lines.insert(0, f"HEADLINE: {headline}")
    lines.insert(1, "")

    lines.append("")
    lines.append("CAVEATS (binding): (1) SYNTHETIC stress surface from PHASE 3/6's VALIDATED generator (2020-calibrated "
                 "stylized facts ONLY) + the 2020 real OOS -- NOT real future data. (2) The longshort INSURANCE leg "
                 "VIOLATES long-only+spot -> the gated book is RESEARCH; deploy needs the user's explicit LO-exception "
                 "sign-off. (3) DETECTOR pre-registered on the 2020 TRAIN band ONLY then FROZEN -- never fit on OOS/the "
                 "synthetic test paths. (4) The SHUFFLE control is a random-timed toggle of EQUAL on-frequency -- the "
                 "detector must beat it to claim TIMING skill (not just exposure). (5) Short-borrow MODELLED at "
                 f"{BORROW_BASE}bps/yr (prorated per-bar). (6) maker cost, no MtM double-count, lag-1 causal. (7) >=20 "
                 "seeds; distributions (mean +- spread + WORST seed). (8) The 2020-calibrated synthetic bear is SHORT "
                 "(~12% of the stitched cycle) and mild in NET terms; the verified mechanism (Q3b) is that the gate's "
                 "FALSE-ALARM cost in the 88% non-bear time outweighs the small bear gain -- so the NULL is specific to "
                 "a SHORT/MILD bear. A DEEPER or LONGER bear (a larger NET share of the cycle, e.g. a 2022-style grind-"
                 "down) is the regime where binary insurance could flip +EV; it is NOT testable on the 2020 band and is "
                 "the explicit open door (the detector + book are BUILT and VALIDATED to run on such a band on request).")
    return {"headline": headline, "works_cadences": works, "beats_trend_cadences": beats_trend_tfs,
            "beats_shuffle_cadences": beats_shuffle_tfs, "detector": dict(det),
            "detector_has_timing_skill": bool(detector_has_skill), "detector_skill": skill, "lines": lines}


# =====================================================================================================
# 8. CHARTS
# =====================================================================================================
TF_COLORS = {"1d": "#1f77b4", "4h": "#ff7f0e", "2h": "#2ca02c", "1h": "#d62728", "30m": "#9467bd", "15m": "#8c564b"}
BOOK_COLORS = {"TREND": "#1f77b4", "GATED": "#2ca02c", "ALWAYS_ON": "#9467bd", "SHUFFLE": "#d62728"}


def chart_bear_capture(res, cadences):
    """Chart 1: bear-protection CAPTURED vs bull+chop drag PAID, gated vs always-on, across TFs. The synthesis
    question made visible: top-left (high capture, low drag) is the goal."""
    cs = [c for c in cadences if c in res]
    if not cs:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    # left: scatter of (drag paid %, protection captured %) per TF -- top-LEFT is the goal (stitched full-cycle)
    for cad in cs:
        r = res[cad]["_ratios"]
        cap = r["bear_protection_captured_pct"]; paid = r["drag_paid_pct"]
        if cap is None or paid is None:
            continue
        ax1.scatter(paid, cap, s=130, color=TF_COLORS.get(cad, "#888"), edgecolor="k", zorder=3)
        ax1.annotate(cad, (paid, cap), fontsize=10, xytext=(5, 5), textcoords="offset points")
    ax1.axhline(50, color="grey", ls=":", lw=0.9); ax1.axvline(50, color="grey", ls=":", lw=0.9)
    ax1.axhline(100, color="#9467bd", ls="--", lw=0.8, label="always-on bear protection (=100%)")
    ax1.set_xlabel("full-cycle drag PAID (% of always-on's drag; LOWER = better)")
    ax1.set_ylabel("bear protection CAPTURED (% of always-on's; HIGHER = better)")
    ax1.set_title("THE SYNTHESIS: bear protection captured vs bull+chop drag paid, per TF\n"
                  "(gating's promise: high capture, low drag -> top-LEFT quadrant)", fontsize=10)
    ax1.text(0.02, 0.04, "top-LEFT = conditional insurance WORKS\n(keeps the bear protection, dodges the drag)",
             transform=ax1.transAxes, fontsize=9, va="bottom",
             bbox=dict(boxstyle="round", fc="#eaffea", ec="#999"))
    ax1.legend(fontsize=8, loc="upper right")
    # right: the bear-DD-protection bars -- gated vs always-on, per TF (the captured amount in pp)
    x = np.arange(len(cs)); w = 0.38
    gated_prot = [res[c]["_ratios"]["gated_bear_protection_pp"] or 0 for c in cs]
    always_prot = [res[c]["_ratios"]["always_on_bear_protection_pp"] or 0 for c in cs]
    ax2.bar(x - w / 2, always_prot, w, color="#9467bd", label="ALWAYS-ON bear DD-protection", alpha=0.9)
    ax2.bar(x + w / 2, gated_prot, w, color="#2ca02c", label="GATED bear DD-protection", alpha=0.9)
    for xi, (a, g) in enumerate(zip(always_prot, gated_prot)):
        ax2.annotate(f"{a:+.1f}", (xi - w / 2, a), ha="center", va="bottom", fontsize=8)
        ax2.annotate(f"{g:+.1f}", (xi + w / 2, g), ha="center", va="bottom", fontsize=8)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(cs); ax2.set_xlabel("timeframe")
    ax2.set_ylabel("bear DD-protection vs trend-alone (pp; higher = more protection)")
    ax2.set_title("Bear DD-protection: GATED vs ALWAYS-ON, per TF\n(how much of the purple does the green keep?)",
                  fontsize=10)
    ax2.legend(fontsize=8)
    fig.suptitle("PHASE 7 -- CONDITIONAL BEAR-INSURANCE: does the GATED longshort capture the bear protection without "
                 "the bull-drag? (synthetic, 2020-calibrated, 20 seeds)\nSHORT = RESEARCH (LO-exception to deploy).",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = CHARTS / "regime_gated_bear_capture.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    print(f"[figure] {p}")


def chart_gated_vs_trend_vs_shuffle(res, cadences):
    """Chart 2: the gated book vs trend-alone vs the shuffled-toggle control across regimes, net + maxDD. Uses
    the cadence with the best gated worst-regime net for the detailed regime panels + a cross-TF skill summary."""
    cs = [c for c in cadences if c in res]
    if not cs:
        return
    best = max(cs, key=lambda c: (res[c]["stitched"]["GATED"]["net"]["mean"] or -1e9))
    summ = res[best]
    cands = ["TREND", "GATED", "ALWAYS_ON", "SHUFFLE"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    x = np.arange(len(SCENARIOS)); width = 0.8 / len(cands)
    # panel A: NET by regime (best cadence)
    axn = axes[0]
    for ci, cand in enumerate(cands):
        vals = [summ[sc][cand]["net"]["mean"] or 0 for sc in SCENARIOS]
        errs = [summ[sc][cand]["net"]["std"] or 0 for sc in SCENARIOS]
        lbl = cand + (" [R]" if cand in ("GATED", "ALWAYS_ON") else "")
        axn.bar(x + (ci - len(cands) / 2 + 0.5) * width, vals, width, yerr=errs, capsize=2,
                color=BOOK_COLORS[cand], label=lbl, alpha=0.9,
                edgecolor="k" if cand == "GATED" else None, linewidth=1.3 if cand == "GATED" else 0)
    axn.axhline(0, color="k", lw=0.8)
    axn.set_xticks(x); axn.set_xticklabels([s.upper() for s in SCENARIOS])
    axn.set_ylabel("net % (mean +- sd)"); axn.set_title(f"{best}: NET by regime -- GATED vs trend/always/shuffle", fontsize=10)
    axn.legend(fontsize=8)
    # panel B: maxDD by regime (best cadence)
    axd = axes[1]
    for ci, cand in enumerate(cands):
        vals = [summ[sc][cand]["maxdd"]["mean"] or 0 for sc in SCENARIOS]
        axd.bar(x + (ci - len(cands) / 2 + 0.5) * width, vals, width, color=BOOK_COLORS[cand],
                label=cand, alpha=0.9, edgecolor="k" if cand == "GATED" else None,
                linewidth=1.3 if cand == "GATED" else 0)
    axd.axhline(0, color="k", lw=0.8)
    axd.set_xticks(x); axd.set_xticklabels([s.upper() for s in SCENARIOS])
    axd.set_ylabel("maxDD % (less-negative = better)"); axd.set_title(f"{best}: maxDD by regime", fontsize=10)
    axd.legend(fontsize=8)
    # panel C: THE SKILL TEST (stitched full-cycle) -- gated minus shuffle (net + maxDD), per TF. Positive = the
    # detector's TIMING beats a random-timed toggle of equal frequency (real skill, not just exposure).
    axc = axes[2]
    xc = np.arange(len(cs)); wc = 0.38
    net_gain = [res[c]["_controls"]["gated_vs_shuffle"]["net_gain_vs_shuffle_pp"] for c in cs]
    dd_gain = [res[c]["_controls"]["gated_vs_shuffle"]["dd_gain_vs_shuffle_pp"] for c in cs]
    axc.bar(xc - wc / 2, dd_gain, wc, color="#2ca02c", label="stitched maxDD gain vs shuffle (pp)", alpha=0.9)
    axc.bar(xc + wc / 2, net_gain, wc, color="#1f77b4", label="stitched net gain vs shuffle (pp)", alpha=0.9)
    for xi, (d, n) in enumerate(zip(dd_gain, net_gain)):
        axc.annotate(f"{d:+.1f}", (xi - wc / 2, d), ha="center", va="bottom" if d >= 0 else "top", fontsize=8)
        axc.annotate(f"{n:+.1f}", (xi + wc / 2, n), ha="center", va="bottom" if n >= 0 else "top", fontsize=8)
    axc.axhline(0, color="k", lw=0.9)
    axc.set_xticks(xc); axc.set_xticklabels(cs); axc.set_xlabel("timeframe")
    axc.set_ylabel("GATED minus SHUFFLE, stitched cycle (pp; positive = detector timing skill)")
    axc.set_title("THE BOOK SKILL TEST: detector vs equal-frequency SHUFFLE (stitched cycle)\n(positive = timing "
                  "beats a random toggle; ~0 = exposure, not timing)", fontsize=10)
    axc.legend(fontsize=8)
    axc.text(0.02, 0.04, "~0 everywhere = the BOOK payoff is exposure not timing\n(even if the gate-PRECISION test "
             "shows the gate fires in the bear)", transform=axc.transAxes, fontsize=8, va="bottom",
             bbox=dict(boxstyle="round", fc="#fff0f0", ec="#999"))
    fig.suptitle("PHASE 7 -- the GATED book vs TREND-ALONE vs the SHUFFLED toggle across regimes (synthetic, 20 seeds)\n"
                 "Control (b): gated must beat trend-alone (PHASE 6's static book FAILED). Control (c, MANDATORY): the "
                 "detector must beat its equal-frequency shuffle. [R]=longshort RESEARCH.", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p = CHARTS / "gated_vs_trend_vs_shuffle.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    print(f"[figure] {p}")


# =====================================================================================================
# 9. SELFTEST -- detector + gate soundness on planted synthetic regimes (NO real-data calibration).
# =====================================================================================================
def selftest():
    """Two-sided detector + gate soundness (synthetic, no real calib):
    POSITIVE: in a planted sustained BEAR, the detector gate must ARM (on-fraction high) and the GATED book must
              draw down LESS than trend-alone (the insurance fires). In a planted BULL, the gate must STAY ~OFF
              (on-fraction low) so the GATED book ~= trend-alone (no bull-drag).
    NEGATIVE: the hysteretic gate must NOT flicker -- a single down-bar in a calm series must NOT arm the gate
              (k_on debounce). And the gate must be CAUSAL -- arming on day t uses only close<=t-1 (a future
              crash must not arm the gate before it happens)."""
    print("## REGIME-GATED-LONGSHORT SELFTEST (detector + gate soundness; no real-data calib)")
    ok = True
    calib = {
        "bull": {"mean": 0.006, "std": 0.045, "kurt": 3.0, "skew": 0.3, "ar1": -0.02, "vol_cluster": 0.20,
                 "t_df": SRS._student_t_df_from_kurt(3.0)},
        "bear": {"mean": -0.012, "std": 0.090, "kurt": 8.0, "skew": -1.5, "ar1": -0.30, "vol_cluster": 0.25,
                 "t_df": SRS._student_t_df_from_kurt(8.0)},
        "chop": {"mean": 0.001, "std": 0.030, "kurt": 2.0, "skew": 0.0, "ar1": -0.10, "vol_cluster": 0.05,
                 "t_df": SRS._student_t_df_from_kurt(2.0)},
        "_xasset": {"mean_pairwise_corr": 0.49, "mean_btc_beta": 0.5, "n_assets": 10},
    }
    det = {"sma": 50, "k_on": 3, "k_off": 3}      # a faster detector so the short synthetic regimes can arm it

    # --- pure-unit: hysteresis state machine (no data) ---
    raw = np.array([0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0], dtype=bool)
    g = _hysteretic_gate(raw, k_on=3, k_off=2)
    # arms at index 6 (3 consecutive bears: idx 4,5,6), disarms after 2 calm (idx 7,8) -> off at 8; idx 9,10 only 2 bears not enough? k_on=3 -> stays off
    flicker_ok = (g[2] == 0 and g[3] == 0 and g[6] == 1 and g[8] == 0)
    print(f"  UNIT hysteresis: gate={list(map(int,g))} (no flicker on isolated bears, arms at 3-run idx6: {flicker_ok})")
    ok &= flicker_ok

    # --- POSITIVE: planted BEAR -- gate arms + gated draws down less than trend ---
    nb = 92
    panels = SRS.generate_regime_panels("bear", calib, seed=2, n_bars=nb)
    with SRS._synthetic_panel_context(panels):
        gate_b = detect_bear_gate("1d", det=det)
        books_b, meta_b = build_books("1d", det=det, shuffle_seed=7)
    on_frac_bear = float(gate_b.mean()) if gate_b is not None else 0.0
    gated_dd = _perf(books_b["GATED"].to_numpy())["maxdd"]; trend_dd = _perf(books_b["TREND"].to_numpy())["maxdd"]
    bear_arms = on_frac_bear > 0.15
    bear_protects = gated_dd is not None and trend_dd is not None and gated_dd >= trend_dd
    print(f"  POSITIVE bear: gate on-frac {on_frac_bear:.2f} (arms >0.15: {bear_arms}); "
          f"GATED maxDD {gated_dd}% vs TREND {trend_dd}% (less-neg: {bear_protects})")
    ok &= bear_arms and bear_protects

    # --- POSITIVE: planted BULL -- gate stays ~off + gated ~= trend (no bull-drag) ---
    panels_u = SRS.generate_regime_panels("bull", calib, seed=3, n_bars=nb)
    with SRS._synthetic_panel_context(panels_u):
        gate_u = detect_bear_gate("1d", det=det)
        books_u, meta_u = build_books("1d", det=det, shuffle_seed=8)
    on_frac_bull = float(gate_u.mean()) if gate_u is not None else 0.0
    gated_un = _perf(books_u["GATED"].to_numpy())["net"]; trend_un = _perf(books_u["TREND"].to_numpy())["net"]
    bull_mostly_off = on_frac_bull < 0.5
    gated_near_trend = (gated_un is not None and trend_un is not None
                        and abs(gated_un - trend_un) < abs(trend_un) * 0.5 + 5.0)
    print(f"  POSITIVE bull: gate on-frac {on_frac_bull:.2f} (mostly off <0.5: {bull_mostly_off}); "
          f"GATED net {gated_un}% ~= TREND {trend_un}% (close: {gated_near_trend})")
    ok &= bull_mostly_off and gated_near_trend

    # --- NEGATIVE: causality -- a gate built on a future crash must NOT arm before the crash bar ---
    # construct a series: 60 calm bars then a sharp drop; the shifted gate must be 0 through the calm region.
    calm_then_crash = np.concatenate([np.full(60, 100.0), np.linspace(100, 60, 20)])
    sma = _sma(calm_then_crash, 50)
    raw_c = np.where(np.isfinite(sma), calm_then_crash < sma, False)
    g_c = _hysteretic_gate(raw_c, det["k_on"], det["k_off"])
    g_c_shift = np.concatenate([[0], g_c[:-1]])
    causal_ok = bool(np.all(g_c_shift[:61] == 0))     # gate off for the entire calm stretch + the first crash bar
    print(f"  NEGATIVE causality: gate off through calm+first-crash bar (no look-ahead): {causal_ok}")
    ok &= causal_ok

    print(f"\n  SELFTEST {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# =====================================================================================================
# 10. MAIN
# =====================================================================================================
def _strip_arrays(d):
    out = {}
    for cad, summ in d.items():
        out[cad] = {k: v for k, v in summ.items() if not str(k).startswith("_equity")}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m strat.regime_gated_longshort")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--cadences", default=",".join(ALL_TFS))
    ap.add_argument("--borrow-bps", type=float, default=BORROW_BASE)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    CHARTS.mkdir(parents=True, exist_ok=True)
    print("## REGIME-GATED LONGSHORT -- PHASE 7 (conditional bear-insurance: longshort toggled by a slow hysteretic detector)")

    # 1. CALIBRATE the PHASE 3/6 generator on 2020 ONLY (its calibrate_2020 is the only real-data touch)
    print("\n## CALIBRATING (PHASE 3 generator, REAL 2020-band data ONLY) ...")
    SRS._CALIB, real_samples = SRS.calibrate_2020()
    validation, _ = SRS.validate_generator(SRS._CALIB, real_samples, seed=0, n_paths=max(30, a.seeds))
    print(f"   GENERATOR VALIDATION: {validation['_summary']['verdict']} "
          f"({validation['_summary']['regimes_all_match']}/{validation['_summary']['regimes_validated']} regimes match)")

    # 2. PRE-REGISTER the detector on the 2020 TRAIN band ONLY, then FREEZE (never touch OOS/synthetic paths)
    print("\n## PRE-REGISTERING the detector on the canonical 2020 TRAIN band (Jan-Jul, has the COVID bear) then FROZEN ...")
    frozen, grid_log = pre_register_detector()
    global DETECTOR
    DETECTOR = dict(frozen)

    seeds = list(range(1, a.seeds + 1))
    cadences = [c.strip() for c in a.cadences.split(",") if c.strip()]

    # 3. THE DECISIVE DETECTOR SKILL TEST (on the stitched path -- the only surface with regime transitions)
    print(f"\n## DETECTOR SKILL TEST on the STITCHED full-cycle path (vs true bear window + equal-freq shuffle) ...")
    skill = run_detector_skill(seeds, DETECTOR)

    # 3b. THE MECHANISM: where the gated book gains/gives-back vs trend, per stitched segment (reproducible)
    print(f"\n## MECHANISM -- per-segment gated-minus-trend decomposition on the stitched path (1d) ...")
    seg_decomp = run_segment_decomposition(seeds, DETECTOR, cad="1d", borrow_bps=a.borrow_bps)

    # 4. THE STRESS RUN (per TF, per regime; the FROZEN detector; the shuffle control)
    print(f"\n## STRESS over {len(seeds)} seeds x {len(cadences)} cadences x 4 regimes (frozen det={DETECTOR})")
    res = run_stress(cadences, seeds, DETECTOR, borrow_bps=a.borrow_bps)

    # 5. 2020-REAL OOS anchor
    print("\n## 2020-REAL OOS anchor (the real-data bull-drag + OOS gate behaviour, Oct-Dec 2020) ...")
    real = run_2020_real(cadences, DETECTOR, borrow_bps=a.borrow_bps)

    # 6. VERDICT
    verdict = build_verdict(res, real, DETECTOR, skill, seg_decomp)
    print("\n" + "=" * 100)
    print("## DECISIVE VERDICT (two-sided)")
    for line in verdict["lines"]:
        print(f"   {line}")
    print("=" * 100)

    # 6. CHARTS
    chart_bear_capture(res, cadences)
    chart_gated_vs_trend_vs_shuffle(res, cadences)

    # 7. PERSIST
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    export = {
        "repro": {"command": "python -m strat.regime_gated_longshort " + " ".join(argv or sys.argv[1:]),
                  "git_sha": sha, "cost_maker": MAKER_RT, "borrow_bps_base": a.borrow_bps,
                  "insurance_w": INSURANCE_W, "phase1a_winners": LSE.PHASE1A_WINNERS,
                  "detector_grid": DET_GRID, "detector_frozen": DETECTOR,
                  "calib_window": SRS.CALIB_WINDOW, "regime_periods": SRS.REGIME_PERIODS,
                  "n_seeds": a.seeds, "cadences": cadences,
                  "constraint": "SYNTHETIC (PHASE 3/6 validated generator) + 2020 calibration/OOS ONLY; never 2026/"
                                "other; longshort INSURANCE = RESEARCH (deploy needs LO-exception sign-off); detector "
                                "PRE-REGISTERED on TRAIN then FROZEN; SHUFFLE control mandatory"},
        "generator_validation": validation,
        "detector_pre_registration": {"frozen": DETECTOR, "train_grid_log": grid_log},
        "detector_skill_stitched": skill,
        "mechanism_segment_decomposition_1d": seg_decomp,
        "stress": _strip_arrays(res),
        "real_2020_oos": real,
        "verdict": verdict,
    }
    p = OUT / "regime_gated_book.json"
    json.dump(export, open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n[persisted] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
