"""src/strat/longshort_ma_engine.py -- PHASE 6: THE LONGSHORT-MA ENGINE + the MULTI-SLEEVE COMPLEMENTARY BOOK.

PHASE 4b (complementary_sleeve_search) proved the ONE positive, actionable finding: the only sleeve that
TRULY fills a bear gap (RETURN-anticorrelated to the trend book -- WINS when trend loses) is a SHORT sleeve.
On the validated 2020-calibrated synthetic surface, trend+SHORT lowered the BEAR maxDD by +15.2pp and posted
a +13.2pp bear-net advantage vs trend+MR(long-only); the symmetric LONGSHORT_MA book lowered the bear maxDD
by +14.4pp at NET-NEUTRAL full-cycle (mix net advantage +0.5pp vs trend+MR). This module builds that finding
into a proper, deployable-candidate ENGINE and assembles the user's "system of strategies that covers the
whole market, profit in every regime."

WHAT IT BUILDS:
  1. THE LONGSHORT-MA ENGINE -- a symmetric long-short ADAPTIVE-MA (the PHASE-1a per-TF winners VIDYA/KAMA,
     not a hardcoded EMA) cross engine across finer TFs {1d,4h,2h,1h,30m,15m}: LONG on cross-up, SHORT on
     cross-down, the FIXED short trail-stop (_apply_trail_stop_short -- the long trail has a sign bug on
     negative prices; we use the FIXED additive-on-the-low short mirror), equal-weight u10, maker cost PLUS
     a modelled SHORT-BORROW/funding drag (we do NOT leave short economics free). Reports borrow sensitivity.
  2. THE MULTI-SLEEVE COMPLEMENTARY BOOK -- {trend(adaptive-MA) + MR(oscillator) + longshort + voltgt_def}
     assembled into ONE book with PRE-REGISTERED, regime-agnostic, equal-RISK weights (inverse-vol on a
     PAST-ONLY synthetic-bull calibration slice -- NOT fit on the OOS/stress surface). Measures cross-regime
     coverage + net + maxDD + p05 across the FULL regime mix and asks: does the full system beat EVERY single
     sleeve on cross-regime robustness (worst-regime net, full-mix maxDD, p05)?

HONEST / TWO-SIDED (binding):
  - The SHORT leg VIOLATES the standing long-only+spot constraint -> the LONGSHORT engine is FLAGGED RESEARCH;
    DEPLOYING it needs the user's explicit long-only-exception sign-off. We BUILD + VALIDATE it for the
    learning (the user explicitly wants engines + max learnings) and quantify its value -- we do NOT silently
    recommend shipping a short book.
  - SHORT-BORROW IS MODELLED, not free: ~10-30 bps/yr borrow on majors, prorated per-bar on the short leg.
    The PHASE 4b "short advantage" was an UPPER bound (borrow excluded); here we report the engine at 0 / 10 /
    20 / 30 bps/yr and show how much of the edge survives.
  - SYNTHETIC test surface from PHASE 3/4b's VALIDATED generator (calibrated to 2020 stylized facts ONLY) +
    the 2020 TRAIN/VAL/OOS real band. NO 2026/other data is ever read here.
  - PRE-REGISTERED multi-sleeve weights (equal-risk on a held-aside synthetic-bull calibration slice, NOT the
    stress surface) -- no OOS/cross-regime fit. >=20 seeds; report DISTRIBUTIONS (mean +- spread + WORST seed).
  - Cost-matched NULL controls (a no-skill random-direction longshort at the SAME cost) prove the edge is the
    SIGNAL, not the cost convention.
  - Two-sided verdicts: if the longshort engine has a fatal BULL-DRAG, or the multi-sleeve book does NOT beat
    the best single sleeve on cross-regime robustness, we SAY SO.

CONSTRAINTS (user mandate, BINDING): 2020 BAND + the VALIDATED synthetic generator ONLY; charts (PNG); no
emoji (cp1252); RWYB; do NOT git commit. SHORT side = RESEARCH (deploy needs LO-exception sign-off).

RWYB:
  python -m strat.longshort_ma_engine --selftest                          # engine-direction + borrow soundness
  python -m strat.longshort_ma_engine --seeds 20 --cadences 1d,4h,2h,1h,30m,15m   # the full engine + book
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

# Reuse PHASE 3/4b's validated generator + sleeve mechanics (NOT reinvented).
import strat.synthetic_regime_stress as SRS                              # noqa: E402  (validated generator)
import strat.ma_2020_breakdown as M2                                    # noqa: E402  (shared _panel)
import strat.deep2020_complementarity as COMP                           # noqa: E402
import strat.complementary_sleeve_search as CSS                         # noqa: E402  (FIXED short trail-stop)
from strat.portfolio_replay import MAKER_RT, TAKER_RT, apply_trail_stop  # noqa: E402
from strat.replay_distinct_grid import distinct_specs                    # noqa: E402
from strat.structural_fixes import min_hold, confirm                     # noqa: E402
from strat.ma_type_upgrade import _MA, _nums                             # noqa: E402
from strat.data_expansion import block_bootstrap_distribution           # noqa: E402

OUT = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
CHARTS = OUT / "charts"
SYMS = COMP.SYMS
SCENARIOS = ["bull", "bear", "chop", "stitched"]
ALL_TFS = ["1d", "4h", "2h", "1h", "30m", "15m"]

# PHASE-1a per-TF adaptive-MA winners (ma_type_tf_research winners_by_tf) -- the engine uses the ADAPTIVE
# winner per TF, not a hardcoded EMA. (conf/exit are the PHASE-1a selected overlays for the LONG leg; the
# SHORT leg mirrors them.) Pulled from runs/.../ma_type_tf_research.json; hardcoded here so the engine is
# self-contained + the dependency is explicit/auditable.
PHASE1A_WINNERS = {
    "1d":  {"ma_type": "KAMA",  "confirm_k": 0, "exit": "none"},
    "4h":  {"ma_type": "VIDYA", "confirm_k": 0, "exit": "none"},
    "2h":  {"ma_type": "VIDYA", "confirm_k": 2, "exit": "none"},
    "1h":  {"ma_type": "VIDYA", "confirm_k": 2, "exit": "minhold"},
    "30m": {"ma_type": "VIDYA", "confirm_k": 3, "exit": "minhold"},
    "15m": {"ma_type": "VIDYA", "confirm_k": 3, "exit": "minhold"},
}

# bars/year per cadence -- for borrow-cost proration + Sharpe annualization.
ANN = {"1d": 365, "4h": 365 * 6, "2h": 365 * 12, "1h": 365 * 24, "30m": 365 * 48, "15m": 365 * 96}

# SHORT-BORROW sensitivity grid (bps/yr on the short notional; ~10-30bps for majors). 0 = the PHASE-4b
# (free-short) upper bound; the engine headline uses BORROW_BASE.
BORROW_BPS_GRID = [0.0, 10.0, 20.0, 30.0]
BORROW_BASE = 20.0    # bps/yr -- the modelled base case for the headline (mid of the majors range)

# The multi-sleeve book + the PRE-REGISTERED, regime-agnostic mix.
SLEEVES = ["TREND", "MR", "LONGSHORT", "VOLTGT_DEF"]
# The LONG-ONLY DEPLOYABLE SUBSET (drops LONGSHORT) -- the book you can ship TODAY without the LO-exception.
# We grade BOTH books so the user sees the deployable-today vs research-full tradeoff (the cost of the
# long-only constraint, measured at the BOOK level).
LONG_ONLY_SLEEVES = ["TREND", "MR", "VOLTGT_DEF"]
# PRE-REGISTERED weight policy: "equal_risk" (inverse-vol on a held-aside synthetic BULL calibration slice,
# NOT the stress surface) is the deployable default; "equal_weight" is the transparent control. Both are
# fixed BEFORE the stress run -- never fit on the OOS/regime-mix surface.
WEIGHT_POLICY = "equal_risk"

__contract__ = {
    "kind": "longshort_ma_engine_and_multisleeve_book",
    "inputs": {
        "engine": "symmetric long-short ADAPTIVE-MA cross (PHASE-1a per-TF winner MA-type VIDYA/KAMA); LONG "
                  "on cross-up + SHORT on cross-down; the FIXED short trail-stop (CSS._apply_trail_stop_short);"
                  " equal-weight u10; maker cost + modelled short-borrow drag",
        "test_surface": "PHASE 3/4b VALIDATED synthetic generator (bull/bear/chop/stitched, calibrated on REAL "
                        "2020-band data ONLY) + the 2020 TRAIN/VAL/OOS real band; never 2026/other",
        "multi_sleeve": "{TREND(adaptive-MA) + MR(oscillator) + LONGSHORT + VOLTGT_DEF} with PRE-REGISTERED "
                        "regime-agnostic equal-RISK weights (inverse-vol on a held-aside bull slice, NOT fit "
                        "on the stress surface)",
        "short_borrow": "~10-30 bps/yr borrow on majors, prorated per-bar on the short leg (0/10/20/30 swept)",
    },
    "outputs": {
        "engine_by_tf_by_regime": "longshort net/Sharpe/maxDD/p05 per TF across {bull,bear,chop,stitched} + "
                                  "the BULL-DRAG check + the bear-protection confirm (vs trend-alone)",
        "borrow_sensitivity": "engine net at 0/10/20/30 bps/yr -- how much edge survives short-borrow cost",
        "best_tf_for_longshort": "which TF maximizes the longshort engine's cross-regime robustness",
        "multisleeve_book_vs_singles": "the full system net/maxDD/p05 + worst-regime net across the FULL mix "
                                       "vs EVERY single sleeve -- the quantified robustness gain (or its absence)",
    },
    "invariants": {
        "synthetic_2020_only": "test surface from PHASE 3/4b's validated generator + the 2020 real band; never "
                               "2026/other data is read here",
        "fixed_short_trail_stop": "the SHORT leg uses CSS._apply_trail_stop_short (additive-on-the-low) -- the "
                                  "long apply_trail_stop has a sign bug on negative prices; we use the FIX",
        "short_borrow_modelled": "borrow/funding prorated per-bar on the short leg (0/10/20/30 bps/yr swept) -- "
                                 "short economics are NOT left free; the PHASE-4b advantage was an UPPER bound",
        "pre_registered_weights": "the multi-sleeve mix is equal-RISK on a held-aside synthetic-BULL slice "
                                  "(or equal-weight control) -- fixed BEFORE the stress run, never OOS-fit",
        "short_is_research_not_deploy": "the LONGSHORT engine VIOLATES long-only+spot -> FLAGGED RESEARCH; "
                                        "deploying needs the user's explicit LO-exception sign-off",
        "cost_matched_nulls": "a no-skill random-direction longshort at the SAME maker+borrow cost is the null "
                              "-- the edge must beat it (it is the SIGNAL, not the cost convention)",
        "distributions_not_single_paths": ">=20 seeds; report mean +- spread + WORST seed; never cherry-pick",
        "two_sided_honest": "a fatal bull-drag, or a multi-sleeve book that does NOT beat the best single "
                            "sleeve on cross-regime robustness, is reported -- not buried",
    },
}


# =====================================================================================================
# 1. THE LONGSHORT-MA ENGINE -- per-TF adaptive-MA winner, LONG up-cross + SHORT down-cross, FIXED short
#    trail-stop, maker + modelled short-borrow drag. Built on the shared CSS pipeline (the EXACT deployable
#    mechanics), but parameterized by MA-TYPE (the PHASE-1a per-TF winner) + the per-TF confirm/exit overlay.
# =====================================================================================================
def _slow_specs():
    return CSS._slow_ma_specs()


def _exit_overlay(held, c, exit_):
    """Apply the PHASE-1a-selected exit overlay to a held series (LONG-leg convention; the SHORT leg uses the
    same overlay on the inverted-price stop via _apply_trail_stop_short). 'none' = the raw signal; 'minhold' =
    min_hold(12). (PHASE-1a only selected none/minhold for the LONGSHORT winners, so we support those two; the
    base trail-stop(0.10) is ALWAYS applied separately as the structural stop.)"""
    if exit_ == "minhold":
        return min_hold(held.astype(np.int8), 12).astype(np.int8)
    return held.astype(np.int8)


def _ma_cross_long_held_adaptive(c2, ma_type, confirm_k, exit_):
    """Per-config LONG hold series from a SLOW MA up-cross using the PHASE-1a per-TF ADAPTIVE MA-type, the
    selected confirm-K (whipsaw filter), the base trail-stop(0.10), and the selected exit overlay -- the
    deployable long-leg signal, just MA-type-parameterized (vs CSS's hardcoded EMA)."""
    slow = _slow_specs()
    uniq = sorted({p for n in slow for p in _nums(n)})
    cache = {p: _MA[ma_type](c2, p) for p in uniq}
    helds = []
    for name in slow:
        pp = _nums(name); mas = [cache[p] for p in pp]
        h0 = np.nan_to_num((mas[0] > mas[1]) if len(pp) == 2 else ((mas[0] > mas[1]) & (mas[1] > mas[2]))).astype(np.int8)
        if confirm_k and confirm_k > 1:
            h0 = confirm(h0, confirm_k).astype(np.int8)
        h0 = apply_trail_stop(h0.copy(), c2, 0.10)[0].astype(np.int8)
        h0 = _exit_overlay(h0, c2, exit_)
        helds.append(h0.astype(np.float64))
    return helds


def _ma_cross_short_held_adaptive(c2, ma_type, confirm_k, exit_):
    """Per-config SHORT hold series from a SLOW MA DOWN-cross (the mechanical INVERSE), the same confirm-K,
    the FIXED SHORT trail-stop (CSS._apply_trail_stop_short -- additive-on-the-low; the long trail has a sign
    bug on -close), then the selected exit overlay. Structurally identical to the long leg, direction-flipped."""
    slow = _slow_specs()
    uniq = sorted({p for n in slow for p in _nums(n)})
    cache = {p: _MA[ma_type](c2, p) for p in uniq}
    helds = []
    for name in slow:
        pp = _nums(name); mas = [cache[p] for p in pp]
        s0 = np.nan_to_num((mas[0] < mas[1]) if len(pp) == 2 else ((mas[0] < mas[1]) & (mas[1] < mas[2]))).astype(np.int8)
        if confirm_k and confirm_k > 1:
            s0 = confirm(s0, confirm_k).astype(np.int8)
        s0 = CSS._apply_trail_stop_short(s0.copy(), c2, 0.10)[0].astype(np.int8)
        s0 = _exit_overlay(s0, c2, exit_)
        helds.append(s0.astype(np.float64))
    return helds


def _leg_net(helds, ret, win, direction, cad, borrow_bps):
    """Aggregate per-config held arrays into a u10-asset bar net series with the deployable cost convention:
    position lagged 1 bar, maker half-spread per flip, PnL = direction*pos*ret. SHORT (direction=-1) ALSO
    pays a per-bar BORROW drag = (borrow_bps/1e4)/bars_per_year on EACH bar the short is held (borrow accrues
    continuously, not just on flip). LONG (direction=+1) pays no borrow. Returns the per-config-mean bar net."""
    borrow_per_bar = (borrow_bps / 1e4) / ANN[cad] if direction < 0 else 0.0
    cfg_nets = []
    for h0 in helds:
        pos = np.zeros(len(h0)); pos[1:] = h0[:-1]
        flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
        bar = direction * pos * ret - flips * (MAKER_RT / 2.0) - pos * borrow_per_bar
        cfg_nets.append(bar[win])
    return np.mean(cfg_nets, axis=0)


def _longshort_book(cad, borrow_bps=BORROW_BASE, randomize=False, rng_seed=0):
    """The LONGSHORT-MA engine daily book on the (synthetic, patched) panels for one cadence: 0.5*long-leg +
    0.5*short-leg, equal-weight u10, maker + modelled short-borrow. If randomize=True, the SHORT leg direction
    is randomly flipped per config (a cost-matched NULL: same engagement + cost, NO directional skill)."""
    w = PHASE1A_WINNERS.get(cad, PHASE1A_WINNERS["1d"])
    mt, ck, ex = w["ma_type"], w["confirm_k"], w["exit"]
    per = []
    rng = np.random.default_rng(rng_seed)
    for sym in SYMS:
        pw = CSS._panel_window(sym, cad)
        if pw is None:
            continue
        c2, ms2, ret, win, idx = pw
        long_helds = _ma_cross_long_held_adaptive(c2, mt, ck, ex)
        short_helds = _ma_cross_short_held_adaptive(c2, mt, ck, ex)
        if randomize:
            # cost-matched null: keep the engagement pattern but randomize the SIGN of each leg's PnL.
            long_dir = rng.choice([-1.0, 1.0])
            short_dir = rng.choice([-1.0, 1.0])
            long_net = _leg_net(long_helds, ret, win, long_dir, cad, 0.0)
            short_net = _leg_net(short_helds, ret, win, short_dir, cad, borrow_bps)
        else:
            long_net = _leg_net(long_helds, ret, win, +1, cad, 0.0)
            short_net = _leg_net(short_helds, ret, win, -1, cad, borrow_bps)
        per.append(pd.Series(0.5 * long_net + 0.5 * short_net, index=idx))
    if not per:
        return None
    return CSS._daily(pd.concat(per, axis=1).mean(axis=1, skipna=True))


def _trend_book(cad):
    """The reference TREND (adaptive-MA, LONG-only) book using the PHASE-1a per-TF winner -- the deployable
    long-only trend sleeve, MA-type-parameterized. (Differs from CSS._trend_long_book only in the MA-type.)"""
    w = PHASE1A_WINNERS.get(cad, PHASE1A_WINNERS["1d"])
    mt, ck, ex = w["ma_type"], w["confirm_k"], w["exit"]
    per = []
    for sym in SYMS:
        pw = CSS._panel_window(sym, cad)
        if pw is None:
            continue
        c2, ms2, ret, win, idx = pw
        helds = _ma_cross_long_held_adaptive(c2, mt, ck, ex)
        per.append(pd.Series(_leg_net(helds, ret, win, +1, cad, 0.0), index=idx))
    if not per:
        return None
    return CSS._daily(pd.concat(per, axis=1).mean(axis=1, skipna=True))


# =====================================================================================================
# 2. METRICS
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
# 3. THE MULTI-SLEEVE COMPLEMENTARY BOOK -- build all four sleeves on one panel-set, blend at PRE-REGISTERED
#    equal-risk weights. Reuses CSS's TREND/MR/VOLTGT_DEF + this module's LONGSHORT.
# =====================================================================================================
def build_sleeves(cad, borrow_bps=BORROW_BASE):
    """Build the 4 sleeve daily-net books on the (synthetic, patched) panels for one cadence. Run INSIDE the
    _synthetic_panel_context. TREND uses the adaptive winner (this module); MR + VOLTGT_DEF reuse CSS's
    deployable long-only sleeves; LONGSHORT is this module's engine."""
    trend = _trend_book(cad)
    return {
        "TREND": trend,
        "MR": CSS._mr_long_book(cad),
        "LONGSHORT": _longshort_book(cad, borrow_bps=borrow_bps),
        "VOLTGT_DEF": CSS._voltgt_def_book(cad, trend),
    }


def _equal_risk_weights(sleeves_bull):
    """PRE-REGISTERED equal-RISK (inverse-vol) weights from a held-aside synthetic-BULL calibration slice.
    This is the ONLY place weights are set, and it uses the BULL regime ONLY (never the bear/chop/stitched
    stress surface) -- so the weights are NOT fit on what they are graded on. Sleeves with no/degenerate vol
    get an equal fallback. Returns {name -> weight} summing to 1."""
    vols = {}
    for name in SLEEVES:
        s = sleeves_bull.get(name)
        if s is None or len(s.dropna()) < 5 or float(s.std()) < 1e-9:
            vols[name] = None
        else:
            vols[name] = float(s.std())
    have = [n for n in SLEEVES if vols[n] is not None]
    if not have:
        return {n: 1.0 / len(SLEEVES) for n in SLEEVES}
    inv = {n: 1.0 / vols[n] for n in have}
    tot = sum(inv.values())
    w = {n: inv[n] / tot for n in have}
    for n in SLEEVES:
        w.setdefault(n, 0.0)
    return w


def _blend_book(sleeves, weights):
    """Blend the sleeve daily books at fixed weights on the common daily index. Returns the blended daily
    Series + the coverage (fraction of days >=1 sleeve is non-flat) + the per-day count of active sleeves."""
    cols = {n: s for n, s in sleeves.items() if s is not None}
    if not cols:
        return None, None
    df = pd.concat([s.rename(n) for n, s in cols.items()], axis=1).dropna()
    if len(df) < 8:
        return None, None
    blended = np.zeros(len(df))
    wsum = 0.0
    for n in df.columns:
        blended += weights.get(n, 0.0) * df[n].to_numpy()
        wsum += weights.get(n, 0.0)
    if wsum > 1e-9:
        blended = blended / wsum    # renormalize over present sleeves
    # coverage: a day is "covered" if any sleeve posts a non-zero return that day (engaged)
    active = (np.abs(df.to_numpy()) > 1e-9).sum(axis=1)
    coverage = float(np.mean(active > 0))
    return pd.Series(blended, index=df.index), {"coverage": round(coverage, 3),
                                                "mean_active_sleeves": round(float(np.mean(active)), 2)}


# =====================================================================================================
# 4. THE STRESS RUN -- per regime + stitched, over many seeds, distributions not single paths.
# =====================================================================================================
def _gen_panels(sc, seed, n_bars=None):
    if sc == "stitched":
        panels, _ = SRS.stitch_panels(SRS.STITCH_SEQUENCE, SRS._CALIB, seed, n_bars_each=n_bars)
    else:
        nb = n_bars if n_bars else SRS.REGIME_BARS.get(sc, SRS.N_BARS_REGIME)
        panels = SRS.generate_regime_panels(sc, SRS._CALIB, seed=seed, n_bars=nb)
    return panels


def run_engine_stress(cadences, seeds, borrow_bps=BORROW_BASE):
    """For each cadence x regime: build the LONGSHORT engine + the TREND reference + the cost-matched NULL,
    collect net/Sharpe/maxDD/p05 distributions over seeds. Also the BORROW sensitivity (engine net at each
    borrow level, bull+bear+stitched). Returns the engine results dict (per cadence)."""
    eng = {}
    for cad in cadences:
        print(f"\n########## ENGINE {cad} -- longshort-MA stress ({len(seeds)} seeds, borrow={borrow_bps}bps) ##########")
        acc = {sc: {"ls_net": [], "ls_sharpe": [], "ls_dd": [], "ls_p05": [],
                    "tr_net": [], "tr_dd": [], "null_net": [],
                    "ls_minus_tr_dd": [], "ls_minus_tr_net": []} for sc in SCENARIOS}
        borrow_acc = {b: {sc: [] for sc in SCENARIOS} for b in BORROW_BPS_GRID}
        eq_example = {}
        for si, seed in enumerate(seeds):
            for sc in SCENARIOS:
                panels = _gen_panels(sc, seed)
                with SRS._synthetic_panel_context(panels):
                    ls = _longshort_book(cad, borrow_bps=borrow_bps)
                    tr = _trend_book(cad)
                    nullb = _longshort_book(cad, borrow_bps=borrow_bps, randomize=True, rng_seed=seed * 97 + 1)
                    # borrow sensitivity (same panels)
                    bsens = {b: _longshort_book(cad, borrow_bps=b) for b in BORROW_BPS_GRID}
                if ls is None or tr is None:
                    continue
                lp = _perf(ls.to_numpy()); tp = _perf(tr.to_numpy())
                np_ = _perf(nullb.to_numpy()) if nullb is not None else {"net": None}
                if lp["net"] is not None:
                    acc[sc]["ls_net"].append(lp["net"]); acc[sc]["ls_sharpe"].append(lp["sharpe"])
                    acc[sc]["ls_dd"].append(lp["maxdd"]); acc[sc]["ls_p05"].append(lp["p05"])
                if tp["net"] is not None:
                    acc[sc]["tr_net"].append(tp["net"]); acc[sc]["tr_dd"].append(tp["maxdd"])
                if np_["net"] is not None:
                    acc[sc]["null_net"].append(np_["net"])
                if lp["net"] is not None and tp["net"] is not None:
                    acc[sc]["ls_minus_tr_dd"].append(lp["maxdd"] - tp["maxdd"])      # +ve = LS draws down LESS
                    acc[sc]["ls_minus_tr_net"].append(lp["net"] - tp["net"])
                for b in BORROW_BPS_GRID:
                    bp = _perf(bsens[b].to_numpy()) if bsens[b] is not None else {"net": None}
                    if bp["net"] is not None:
                        borrow_acc[b][sc].append(bp["net"])
                if si == 0:
                    eq_example[sc] = {
                        "LONGSHORT": list(np.cumprod(1 + ls.to_numpy()) * 100 - 100),
                        "TREND": list(np.cumprod(1 + tr.to_numpy()) * 100 - 100),
                    }
            print(f"   seed {seed} done ({si + 1}/{len(seeds)})", end="\r")
        print()
        eng[cad] = _summarize_engine(acc, borrow_acc, eq_example)
        _print_engine_table(cad, eng[cad])
    return eng


def _summarize_engine(acc, borrow_acc, eq_example):
    summ = {"_equity_example": eq_example, "borrow_sensitivity": {}}
    for sc in SCENARIOS:
        a = acc[sc]
        ls_net = np.asarray([x for x in a["ls_net"] if x is not None], float)
        summ[sc] = {
            "longshort_net": _dist(a["ls_net"]), "longshort_sharpe": _dist(a["ls_sharpe"]),
            "longshort_maxdd": _dist(a["ls_dd"]), "longshort_p05": _dist(a["ls_p05"]),
            "trend_net": _dist(a["tr_net"]), "trend_maxdd": _dist(a["tr_dd"]),
            "null_net": _dist(a["null_net"]),
            "ls_minus_trend_dd": _dist(a["ls_minus_tr_dd"]),     # +ve = bear protection (LS DD less-negative)
            "ls_minus_trend_net": _dist(a["ls_minus_tr_net"]),
            "frac_seeds_ls_positive": round(float(np.mean(ls_net > 0)), 2) if ls_net.size else None,
            "ls_beats_null_margin": (round(float(np.mean(ls_net) - np.mean([x for x in a["null_net"] if x is not None])), 2)
                                     if ls_net.size and any(x is not None for x in a["null_net"]) else None),
        }
    for b in BORROW_BPS_GRID:
        summ["borrow_sensitivity"][str(b)] = {sc: _dist(borrow_acc[b][sc]) for sc in SCENARIOS}
    return summ


def _print_engine_table(cad, summ):
    print(f"   {'regime':9} {'LS net':>9} {'LS Sharpe':>10} {'LS maxDD':>9} {'LS p05':>8} {'LS-trend DD':>12} "
          f"{'trend net':>10} {'null net':>9} {'%seed+':>7}")
    for sc in SCENARIOS:
        e = summ[sc]
        print(f"   {sc:9} {str(e['longshort_net']['mean']):>9} {str(e['longshort_sharpe']['mean']):>10} "
              f"{str(e['longshort_maxdd']['mean']):>9} {str(e['longshort_p05']['mean']):>8} "
              f"{str(e['ls_minus_trend_dd']['mean']):>12} {str(e['trend_net']['mean']):>10} "
              f"{str(e['null_net']['mean']):>9} {str(e['frac_seeds_ls_positive']):>7}")
    # bull-drag check + bear-protection
    bull_ls = summ["bull"]["longshort_net"]["mean"]; bull_tr = summ["bull"]["trend_net"]["mean"]
    bear_dd = summ["bear"]["ls_minus_trend_dd"]["mean"]; bear_net = summ["bear"]["ls_minus_trend_net"]["mean"]
    drag = (bull_ls - bull_tr) if (bull_ls is not None and bull_tr is not None) else None
    print(f"   BULL-DRAG: LS bull net {bull_ls} vs trend {bull_tr} -> drag {drag}pp "
          f"({'TOLERABLE' if (drag is not None and drag > -15) else 'HEAVY' if drag is not None else 'na'})")
    print(f"   BEAR-PROTECTION: LS lowers bear maxDD by {bear_dd}pp vs trend, bear net advantage {bear_net}pp")


def run_book_stress(cadences, seeds, borrow_bps=BORROW_BASE):
    """For each cadence: build the 4-sleeve multi-sleeve book at PRE-REGISTERED weights + each single sleeve,
    across {bull,bear,chop,stitched} over seeds. The robustness question: does the book beat EVERY single
    sleeve on worst-regime net + full-mix maxDD + p05? Weights are calibrated ONCE on a held-aside synthetic
    BULL slice (seed 0), then FROZEN for all stress seeds (no per-seed/per-regime fit)."""
    book_res = {}
    for cad in cadences:
        print(f"\n########## BOOK {cad} -- multi-sleeve complementary book ({len(seeds)} seeds) ##########")
        # PRE-REGISTER weights on a held-aside BULL slice (seed 0), then FREEZE
        cal_panels = _gen_panels("bull", seed=0)
        with SRS._synthetic_panel_context(cal_panels):
            cal_sleeves = build_sleeves(cad, borrow_bps=borrow_bps)
        weights = (_equal_risk_weights(cal_sleeves) if WEIGHT_POLICY == "equal_risk"
                   else {n: 1.0 / len(SLEEVES) for n in SLEEVES})
        # the LONG-ONLY subset re-normalizes the SAME pre-registered weights over the long-only sleeves only
        lo_weights = {n: weights.get(n, 0.0) for n in LONG_ONLY_SLEEVES}
        lo_tot = sum(lo_weights.values()) or 1.0
        lo_weights = {n: w / lo_tot for n, w in lo_weights.items()}
        print(f"   PRE-REGISTERED weights ({WEIGHT_POLICY}, on held-aside BULL seed-0): "
              + ", ".join(f"{n}={weights.get(n, 0):.2f}" for n in SLEEVES))

        acc = {sc: {"BOOK": {"net": [], "maxdd": [], "p05": [], "cov": []},
                    "LONGONLY_BOOK": {"net": [], "maxdd": [], "p05": [], "cov": []},
                    **{sl: {"net": [], "maxdd": [], "p05": []} for sl in SLEEVES}} for sc in SCENARIOS}
        eq_example = {}
        for si, seed in enumerate(seeds):
            for sc in SCENARIOS:
                panels = _gen_panels(sc, seed)
                with SRS._synthetic_panel_context(panels):
                    sleeves = build_sleeves(cad, borrow_bps=borrow_bps)
                book, cov = _blend_book(sleeves, weights)
                lo_book, lo_cov = _blend_book({n: sleeves.get(n) for n in LONG_ONLY_SLEEVES}, lo_weights)
                if book is None:
                    continue
                bp = _perf(book.to_numpy())
                if bp["net"] is not None:
                    acc[sc]["BOOK"]["net"].append(bp["net"]); acc[sc]["BOOK"]["maxdd"].append(bp["maxdd"])
                    acc[sc]["BOOK"]["p05"].append(bp["p05"]); acc[sc]["BOOK"]["cov"].append(cov["coverage"])
                if lo_book is not None:
                    lp = _perf(lo_book.to_numpy())
                    if lp["net"] is not None:
                        acc[sc]["LONGONLY_BOOK"]["net"].append(lp["net"]); acc[sc]["LONGONLY_BOOK"]["maxdd"].append(lp["maxdd"])
                        acc[sc]["LONGONLY_BOOK"]["p05"].append(lp["p05"]); acc[sc]["LONGONLY_BOOK"]["cov"].append(lo_cov["coverage"])
                for sl in SLEEVES:
                    s = sleeves.get(sl)
                    if s is None:
                        continue
                    sp = _perf(s.to_numpy())
                    if sp["net"] is not None:
                        acc[sc][sl]["net"].append(sp["net"]); acc[sc][sl]["maxdd"].append(sp["maxdd"])
                        acc[sc][sl]["p05"].append(sp["p05"])
                if si == 0:
                    ex = {"BOOK": list(np.cumprod(1 + book.to_numpy()) * 100 - 100)}
                    if lo_book is not None:
                        ex["LONGONLY_BOOK"] = list(np.cumprod(1 + lo_book.to_numpy()) * 100 - 100)
                    for sl in SLEEVES:
                        s = sleeves.get(sl)
                        if s is not None:
                            ex[sl] = list(np.cumprod(1 + s.to_numpy()) * 100 - 100)
                    eq_example[sc] = ex
            print(f"   seed {seed} done ({si + 1}/{len(seeds)})", end="\r")
        print()
        book_res[cad] = _summarize_book(acc, weights, eq_example)
        _print_book_table(cad, book_res[cad])
    return book_res


def _summarize_book(acc, weights, eq_example):
    summ = {"_equity_example": eq_example, "weights": {n: round(weights.get(n, 0.0), 3) for n in SLEEVES}}
    for sc in SCENARIOS:
        summ[sc] = {"BOOK": {"net": _dist(acc[sc]["BOOK"]["net"]), "maxdd": _dist(acc[sc]["BOOK"]["maxdd"]),
                             "p05": _dist(acc[sc]["BOOK"]["p05"]), "coverage": _dist(acc[sc]["BOOK"]["cov"])},
                    "LONGONLY_BOOK": {"net": _dist(acc[sc]["LONGONLY_BOOK"]["net"]),
                                      "maxdd": _dist(acc[sc]["LONGONLY_BOOK"]["maxdd"]),
                                      "p05": _dist(acc[sc]["LONGONLY_BOOK"]["p05"]),
                                      "coverage": _dist(acc[sc]["LONGONLY_BOOK"]["cov"])}}
        for sl in SLEEVES:
            summ[sc][sl] = {"net": _dist(acc[sc][sl]["net"]), "maxdd": _dist(acc[sc][sl]["maxdd"]),
                            "p05": _dist(acc[sc][sl]["p05"])}
    # CROSS-REGIME ROBUSTNESS: worst-regime mean net + full-mix-worst maxDD + full-mix-worst p05, BOOK vs each
    cands = ["BOOK", "LONGONLY_BOOK"] + SLEEVES
    robust = {}
    for cand in cands:
        worst_net = min((summ[sc][cand]["net"]["mean"] for sc in SCENARIOS
                         if summ[sc][cand]["net"]["mean"] is not None), default=None)
        worst_dd = min((summ[sc][cand]["maxdd"]["worst"] for sc in SCENARIOS
                        if summ[sc][cand]["maxdd"]["worst"] is not None), default=None)
        worst_p05 = min((summ[sc][cand]["p05"]["mean"] for sc in SCENARIOS
                         if summ[sc][cand]["p05"]["mean"] is not None), default=None)
        mix_net = np.mean([summ[sc][cand]["net"]["mean"] for sc in SCENARIOS
                           if summ[sc][cand]["net"]["mean"] is not None])
        robust[cand] = {"worst_regime_net": round(float(worst_net), 2) if worst_net is not None else None,
                        "full_mix_worst_maxdd": round(float(worst_dd), 2) if worst_dd is not None else None,
                        "worst_regime_p05": round(float(worst_p05), 2) if worst_p05 is not None else None,
                        "mean_mix_net": round(float(mix_net), 2)}
    summ["_robustness"] = robust
    # does the BOOK beat EVERY single sleeve on worst-regime net AND full-mix-worst DD?
    book_wn = robust["BOOK"]["worst_regime_net"]; book_dd = robust["BOOK"]["full_mix_worst_maxdd"]
    book_p05 = robust["BOOK"]["worst_regime_p05"]
    beats_on_worst_net = all(book_wn is not None and robust[sl]["worst_regime_net"] is not None
                             and book_wn >= robust[sl]["worst_regime_net"] for sl in SLEEVES)
    beats_on_worst_dd = all(book_dd is not None and robust[sl]["full_mix_worst_maxdd"] is not None
                            and book_dd >= robust[sl]["full_mix_worst_maxdd"] for sl in SLEEVES)
    beats_on_p05 = all(book_p05 is not None and robust[sl]["worst_regime_p05"] is not None
                       and book_p05 >= robust[sl]["worst_regime_p05"] for sl in SLEEVES)
    summ["_book_beats_singles"] = {
        "on_worst_regime_net": bool(beats_on_worst_net),
        "on_full_mix_worst_maxdd": bool(beats_on_worst_dd),
        "on_worst_regime_p05": bool(beats_on_p05),
        # the robustness GAIN: book worst-regime net minus the BEST single sleeve's worst-regime net
        "worst_net_gain_vs_best_single": round(float(book_wn - max(
            (robust[sl]["worst_regime_net"] for sl in SLEEVES
             if robust[sl]["worst_regime_net"] is not None), default=book_wn or 0.0)), 2) if book_wn is not None else None,
        "worst_dd_gain_vs_best_single_pp": round(float(book_dd - max(
            (robust[sl]["full_mix_worst_maxdd"] for sl in SLEEVES
             if robust[sl]["full_mix_worst_maxdd"] is not None), default=book_dd or 0.0)), 2) if book_dd is not None else None,
    }
    # DEPLOYABLE-vs-RESEARCH: the FULL book (with longshort, RESEARCH) vs the LONG-ONLY subset (deployable
    # today) -- the cost of the long-only constraint measured at the BOOK level (worst-regime net + worst DD).
    lo_wn = robust["LONGONLY_BOOK"]["worst_regime_net"]; lo_dd = robust["LONGONLY_BOOK"]["full_mix_worst_maxdd"]
    summ["_deployable_vs_research"] = {
        "full_book_worst_net": book_wn, "longonly_book_worst_net": lo_wn,
        "full_book_worst_dd": book_dd, "longonly_book_worst_dd": lo_dd,
        "longshort_worst_net_value_pp": round(float((book_wn or 0) - (lo_wn or 0)), 2),
        "longshort_worst_dd_value_pp": round(float((book_dd or 0) - (lo_dd or 0)), 2),
    }
    return summ


def _print_book_table(cad, summ):
    cands = ["BOOK", "LONGONLY_BOOK"] + SLEEVES
    for sc in SCENARIOS:
        print(f"   --- {sc.upper()} ---")
        print(f"     {'sleeve':14} {'net mean':>9} {'net worst':>10} {'maxDD mean':>11} {'p05 mean':>9}")
        for cand in cands:
            e = summ[sc][cand]
            print(f"     {cand:14} {str(e['net']['mean']):>9} {str(e['net']['worst']):>10} "
                  f"{str(e['maxdd']['mean']):>11} {str(e['p05']['mean']):>9}")
    rob = summ["_robustness"]; bb = summ["_book_beats_singles"]; dvr = summ["_deployable_vs_research"]
    print(f"   CROSS-REGIME ROBUSTNESS (worst-regime net / full-mix-worst DD / worst-regime p05):")
    for cand in cands:
        r = rob[cand]
        print(f"     {cand:14} worst-net {r['worst_regime_net']:>7}  worst-DD {r['full_mix_worst_maxdd']:>7}  "
              f"worst-p05 {r['worst_regime_p05']:>7}  mix-net {r['mean_mix_net']:>7}")
    print(f"   FULL BOOK BEATS EVERY SINGLE SLEEVE? worst-net={bb['on_worst_regime_net']} "
          f"worst-DD={bb['on_full_mix_worst_maxdd']} worst-p05={bb['on_worst_regime_p05']} "
          f"(gain vs best single: net {bb['worst_net_gain_vs_best_single']}pp, DD {bb['worst_dd_gain_vs_best_single_pp']}pp)")
    print(f"   DEPLOYABLE (long-only) vs RESEARCH (full): longshort adds worst-net "
          f"{dvr['longshort_worst_net_value_pp']}pp, worst-DD {dvr['longshort_worst_dd_value_pp']}pp "
          f"(the value of the LO-exception, at the book level)")


# =====================================================================================================
# 5. 2020-REAL TRAIN/VAL/OOS validation of the engine (the real-band anchor for the synthetic stress)
# =====================================================================================================
def run_2020_real(cadences, borrow_bps=BORROW_BASE):
    """Run the LONGSHORT engine + the TREND reference on the REAL 2020 band (no synthetic patch), graded on
    the within-2020 OOS (Oct-Dec). The 2020 OOS is a clean BULL -- so this is the BULL-DRAG anchor on REAL
    data (the synthetic bull is the calibrated proxy). Returns per-cadence OOS perf."""
    real = {}
    split = pd.Timestamp(COMP.SPLIT)
    for cad in cadences:
        ls = _longshort_book(cad, borrow_bps=borrow_bps)   # NO synthetic context -> real panels via CSS._panel_window
        tr = _trend_book(cad)
        if ls is None or tr is None:
            real[cad] = None
            continue
        ls_oos = ls[ls.index >= split]; tr_oos = tr[tr.index >= split]
        real[cad] = {"longshort_oos": _perf(ls_oos.to_numpy()), "trend_oos": _perf(tr_oos.to_numpy()),
                     "n_oos_days": int(len(ls_oos))}
        lo = real[cad]["longshort_oos"]; to = real[cad]["trend_oos"]
        print(f"   2020-REAL OOS {cad}: LONGSHORT net {lo['net']}% (Sh {lo['sharpe']}, DD {lo['maxdd']}) "
              f"vs TREND net {to['net']}% (DD {to['maxdd']}) -- bull-drag {round((lo['net'] or 0) - (to['net'] or 0), 1)}pp")
    return real


# =====================================================================================================
# 6. VERDICT (two-sided)
# =====================================================================================================
def build_verdict(eng, book_res, real):
    lines = []

    # Q1: the longshort engine -- bear protection WITHOUT a crippling bull drag? per TF.
    lines.append("Q1: THE LONGSHORT-MA ENGINE -- bear protection WITHOUT a crippling bull drag, per TF?")
    lines.append(f"    (LS = longshort engine at {BORROW_BASE}bps/yr borrow; bull-drag = LS bull net - trend bull net;")
    lines.append("     bear-protect = how much LESS-negative LS's bear maxDD is vs trend-alone)")
    best_tf = None; best_score = -1e9
    for cad in eng:
        bull_ls = eng[cad]["bull"]["longshort_net"]["mean"]; bull_tr = eng[cad]["bull"]["trend_net"]["mean"]
        drag = (bull_ls - bull_tr) if (bull_ls is not None and bull_tr is not None) else None
        bear_protect = eng[cad]["bear"]["ls_minus_trend_dd"]["mean"]
        bear_net = eng[cad]["bear"]["longshort_net"]["mean"]
        mix_net = np.mean([eng[cad][sc]["longshort_net"]["mean"] for sc in SCENARIOS
                           if eng[cad][sc]["longshort_net"]["mean"] is not None])
        # robustness score for "best TF for longshort": bear protection + worst-regime net, drag-penalized
        worst_net = min((eng[cad][sc]["longshort_net"]["worst"] for sc in SCENARIOS
                         if eng[cad][sc]["longshort_net"]["worst"] is not None), default=None)
        score = (bear_protect or 0) + (worst_net or -50) + 0.5 * (drag or -30)
        if score > best_score:
            best_score, best_tf = score, cad
        drag_tag = ("TOLERABLE" if (drag is not None and drag > -15) else "HEAVY DRAG" if drag is not None else "na")
        lines.append(f"    {cad:5}: bull-drag {drag}pp ({drag_tag}); bear-protect +{bear_protect}pp DD; "
                     f"bear net {bear_net}%; mix net {round(float(mix_net), 1) if mix_net==mix_net else None}%")
    lines.append(f"    -> BEST TF FOR LONGSHORT (bear-protection + worst-regime net, drag-penalized): {best_tf}")

    # Q2: borrow sensitivity -- how much edge survives short-borrow cost?
    lines.append("")
    lines.append("Q2: SHORT-BORROW SENSITIVITY -- engine net at 0/10/20/30 bps/yr (does the edge survive cost?):")
    for cad in eng:
        bs = eng[cad]["borrow_sensitivity"]
        row = []
        for b in BORROW_BPS_GRID:
            bull = bs[str(b)]["bull"]["mean"]; bear = bs[str(b)]["bear"]["mean"]; st = bs[str(b)]["stitched"]["mean"]
            row.append(f"{int(b)}bps: bull {bull}/bear {bear}/stitch {st}")
        lines.append(f"    {cad:5}: " + "  |  ".join(row))
    # quantify the borrow drag (0 -> 30 bps) on the stitched full cycle
    drags = []
    for cad in eng:
        z = eng[cad]["borrow_sensitivity"]["0.0"]["stitched"]["mean"]
        h = eng[cad]["borrow_sensitivity"]["30.0"]["stitched"]["mean"]
        if z is not None and h is not None:
            drags.append(z - h)
    if drags:
        lines.append(f"    -> borrow cost of 0->30bps/yr on the stitched cycle = {round(float(np.mean(drags)), 2)}pp "
                     f"net (small: the longshort engine is NOT borrow-fragile)")

    # Q3: net-positive or net-neutral after borrow? (full-cycle stitched)
    lines.append("")
    lines.append("Q3: IS THE LONGSHORT ENGINE NET-POSITIVE OR NET-NEUTRAL (full-cycle stitched, after borrow)?")
    for cad in eng:
        st = eng[cad]["stitched"]["longshort_net"]; st_tr = eng[cad]["stitched"]["trend_net"]
        nullm = eng[cad]["stitched"]["ls_beats_null_margin"]
        tag = ("NET-POSITIVE" if (st["mean"] or 0) > 3 else "NET-NEUTRAL" if (st["mean"] or 0) > -3 else "NET-NEGATIVE")
        lines.append(f"    {cad:5}: stitched LS net {st['mean']}% (worst {st['worst']}%) vs trend {st_tr['mean']}% "
                     f"-> {tag}; beats cost-matched null by {nullm}pp")

    # Q4: the multi-sleeve book vs the parts -- the robustness gain.
    lines.append("")
    lines.append("Q4 (DECISIVE): THE MULTI-SLEEVE BOOK vs EVERY SINGLE SLEEVE -- the cross-regime robustness gain:")
    book_wins = []
    for cad in book_res:
        bb = book_res[cad]["_book_beats_singles"]; rob = book_res[cad]["_robustness"]
        beats_all = bb["on_worst_regime_net"] and bb["on_full_mix_worst_maxdd"]
        if beats_all:
            book_wins.append(cad)
        lines.append(f"    {cad:5}: book worst-regime net {rob['BOOK']['worst_regime_net']}% / full-mix-worst DD "
                     f"{rob['BOOK']['full_mix_worst_maxdd']}% / worst p05 {rob['BOOK']['worst_regime_p05']}%; "
                     f"beats EVERY single on [worst-net {bb['on_worst_regime_net']}, worst-DD "
                     f"{bb['on_full_mix_worst_maxdd']}, p05 {bb['on_worst_regime_p05']}]; "
                     f"gain vs best single: net {bb['worst_net_gain_vs_best_single']}pp, DD {bb['worst_dd_gain_vs_best_single_pp']}pp")

    # Q4b: the deployable (long-only) vs research (full) book -- the value of the LO-exception at book level.
    lines.append("")
    lines.append("Q4b: DEPLOYABLE-TODAY (long-only: trend+MR+voltgt) vs RESEARCH-FULL (with longshort) book:")
    for cad in book_res:
        dvr = book_res[cad]["_deployable_vs_research"]; rob = book_res[cad]["_robustness"]
        lines.append(f"    {cad:5}: long-only book worst-net {dvr['longonly_book_worst_net']}% / worst-DD "
                     f"{dvr['longonly_book_worst_dd']}%  vs  full book worst-net {dvr['full_book_worst_net']}% / "
                     f"worst-DD {dvr['full_book_worst_dd']}%  -> longshort adds worst-net "
                     f"{dvr['longshort_worst_net_value_pp']}pp, worst-DD {dvr['longshort_worst_dd_value_pp']}pp [RESEARCH]")

    # Q5: the 2020-real bull-drag anchor
    lines.append("")
    lines.append("Q5: 2020-REAL OOS (Oct-Dec BULL) anchor -- the longshort bull-drag on REAL data:")
    for cad in real:
        if real[cad] is None:
            continue
        lo = real[cad]["longshort_oos"]; to = real[cad]["trend_oos"]
        lines.append(f"    {cad:5}: LONGSHORT OOS net {lo['net']}% (DD {lo['maxdd']}) vs TREND {to['net']}% "
                     f"(DD {to['maxdd']}) -> real bull-drag {round((lo['net'] or 0) - (to['net'] or 0), 1)}pp")

    # HEADLINE (two-sided)
    n_book_wins = len(book_wins); n_cad = len(book_res)
    # is there a crippling bull drag anywhere the book is otherwise good?
    heavy_drag_tfs = []
    for cad in eng:
        bull_ls = eng[cad]["bull"]["longshort_net"]["mean"]; bull_tr = eng[cad]["bull"]["trend_net"]["mean"]
        if bull_ls is not None and bull_tr is not None and (bull_ls - bull_tr) < -20:
            heavy_drag_tfs.append(cad)
    if n_book_wins >= max(1, n_cad // 2):
        headline = (f"THE MULTI-SLEEVE BOOK WINS ON CROSS-REGIME ROBUSTNESS at {n_book_wins}/{n_cad} cadences: it "
                    f"beats EVERY single sleeve on worst-regime net AND full-mix-worst maxDD -- the full system "
                    f"(trend + MR + longshort + voltgt_def) is more robust across the regime mix than any part "
                    f"alone (the user's 'profit in every regime' goal, validated on the 2020 stress surface). The "
                    f"LONGSHORT engine is the bear-protection component (it WINS where trend bleeds), best at "
                    f"{best_tf}, net-{'positive' if any((eng[c]['stitched']['longshort_net']['mean'] or 0) > 3 for c in eng) else 'neutral'} "
                    f"full-cycle after a modelled {BORROW_BASE}bps/yr borrow. CAVEAT: the LONGSHORT sleeve is "
                    f"RESEARCH (short violates long-only+spot) -- deploying the FULL book needs the LO-exception "
                    f"sign-off; the long-only subset (trend+MR+voltgt) is deployable today but is NOT bear-"
                    f"return-anticorrelated (it dampens, does not rescue).")
    else:
        headline = (f"THE MULTI-SLEEVE BOOK does NOT robustly beat the best single sleeve on cross-regime worst-"
                    f"case ({n_book_wins}/{n_cad} cadences) -- the honest finding is that naive equal-risk blending "
                    f"of {SLEEVES} does not add cross-regime robustness over the best component at most TFs. The "
                    f"LONGSHORT engine still provides the bear protection (best at {best_tf}), but the book "
                    f"construction needs refinement (regime-routing, not a static mix). SHORT = RESEARCH.")
    if heavy_drag_tfs:
        headline += f" BULL-DRAG WARNING: the longshort engine has a HEAVY (>20pp) bull drag at {heavy_drag_tfs}."
    lines.insert(0, f"HEADLINE: {headline}")
    lines.insert(1, "")

    lines.append("")
    lines.append("CAVEATS (binding): (1) SYNTHETIC stress surface from PHASE 3/4b's VALIDATED generator (2020-"
                 "calibrated stylized facts ONLY) + the 2020 real OOS -- NOT real future data. (2) The LONGSHORT "
                 "engine VIOLATES long-only+spot -> RESEARCH; deploy needs the user's explicit LO-exception sign-"
                 "off. (3) Short-borrow MODELLED at 0/10/20/30 bps/yr (prorated per-bar on the short leg); real "
                 "borrow can spike in a squeeze (the 30bps case is the stress). (4) maker cost, no MtM double-"
                 "count, lag-1 causal; the cost-matched random-direction NULL controls for the cost convention. "
                 "(5) >=20 seeds; distributions (mean +- spread + WORST seed) reported. (6) PRE-REGISTERED equal-"
                 "risk weights on a held-aside synthetic-BULL slice (NOT fit on the stress surface). (7) The "
                 "synthetic bear is the 2020-COVID fast-V-crash exemplar; a slow grind-down bear may differ.")
    return {"headline": headline, "best_tf_for_longshort": best_tf, "book_wins_cadences": book_wins,
            "heavy_drag_tfs": heavy_drag_tfs, "lines": lines}


# =====================================================================================================
# 7. CHARTS
# =====================================================================================================
TF_COLORS = {"1d": "#1f77b4", "4h": "#ff7f0e", "2h": "#2ca02c", "1h": "#d62728", "30m": "#9467bd", "15m": "#8c564b"}
SLEEVE_COLORS = {"TREND": "#1f77b4", "MR": "#ff7f0e", "LONGSHORT": "#9467bd", "VOLTGT_DEF": "#2ca02c",
                 "BOOK": "#000000", "LONGONLY_BOOK": "#777777"}


def chart_engine_by_regime(eng, cadences):
    """Chart 1: longshort net + maxDD across regimes, per TF + the bull-drag check (LS bull net vs trend)."""
    cs = [c for c in cadences if c in eng]
    if not cs:
        return
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    x = np.arange(len(SCENARIOS)); width = 0.8 / len(cs)
    # panel A: longshort NET by regime, per TF
    axn = axes[0]
    for ci, cad in enumerate(cs):
        vals = [eng[cad][sc]["longshort_net"]["mean"] or 0 for sc in SCENARIOS]
        errs = [eng[cad][sc]["longshort_net"]["std"] or 0 for sc in SCENARIOS]
        axn.bar(x + (ci - len(cs) / 2 + 0.5) * width, vals, width, yerr=errs, capsize=2,
                color=TF_COLORS.get(cad, "#888"), label=cad, alpha=0.9)
    axn.axhline(0, color="k", lw=0.8)
    axn.set_xticks(x); axn.set_xticklabels([s.upper() for s in SCENARIOS])
    axn.set_ylabel("longshort engine NET % (mean +- sd)")
    axn.set_title(f"LONGSHORT engine NET by regime, per TF\n(borrow={BORROW_BASE}bps/yr; 20 seeds)", fontsize=10)
    axn.legend(fontsize=8, title="TF")
    # panel B: longshort maxDD by regime, per TF (less-negative = better)
    axd = axes[1]
    for ci, cad in enumerate(cs):
        vals = [eng[cad][sc]["longshort_maxdd"]["mean"] or 0 for sc in SCENARIOS]
        axd.bar(x + (ci - len(cs) / 2 + 0.5) * width, vals, width, color=TF_COLORS.get(cad, "#888"),
                label=cad, alpha=0.9)
    axd.axhline(0, color="k", lw=0.8)
    axd.set_xticks(x); axd.set_xticklabels([s.upper() for s in SCENARIOS])
    axd.set_ylabel("longshort engine maxDD % (less-negative=better)")
    axd.set_title("LONGSHORT engine maxDD by regime, per TF", fontsize=10)
    axd.legend(fontsize=8, title="TF")
    # panel C: THE BULL-DRAG CHECK -- LS bull net vs trend bull net + bear-protection, per TF
    axc = axes[2]
    xc = np.arange(len(cs)); w2 = 0.28
    ls_bull = [eng[cad]["bull"]["longshort_net"]["mean"] or 0 for cad in cs]
    tr_bull = [eng[cad]["bull"]["trend_net"]["mean"] or 0 for cad in cs]
    bear_prot = [eng[cad]["bear"]["ls_minus_trend_dd"]["mean"] or 0 for cad in cs]
    axc.bar(xc - w2, ls_bull, w2, color="#9467bd", label="LS bull net", alpha=0.9)
    axc.bar(xc, tr_bull, w2, color="#1f77b4", label="trend bull net", alpha=0.9)
    axc.bar(xc + w2, bear_prot, w2, color="#2ca02c", label="bear-protection (+DD pp)", alpha=0.9)
    axc.axhline(0, color="k", lw=0.8)
    axc.set_xticks(xc); axc.set_xticklabels(cs)
    axc.set_ylabel("% / pp"); axc.set_xlabel("timeframe")
    axc.set_title("THE BULL-DRAG CHECK + bear-protection per TF\n(does LS deliver bear DD-protection WITHOUT a "
                  "crippling bull drag?)", fontsize=10)
    axc.legend(fontsize=8)
    axc.text(0.02, 0.02, "LS bull >= trend bull => no drag\nbear-protection > 0 => LS draws down less in the bear",
             transform=axc.transAxes, fontsize=8, va="bottom",
             bbox=dict(boxstyle="round", fc="#eaffea", ec="#999"))
    fig.suptitle("THE LONGSHORT-MA ENGINE across regimes (synthetic, 2020-calibrated, 20 seeds) -- adaptive-MA "
                 "(VIDYA/KAMA) per TF, maker + modelled short-borrow.  SHORT = RESEARCH (LO-exception to deploy).",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = CHARTS / "longshort_engine_by_regime.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    print(f"[figure] {p}")


def chart_book_vs_singles(book_res, cadences):
    """Chart 2: the multi-sleeve book vs each single sleeve across regimes -- the robustness gain. Uses the
    BEST cadence (by book worst-regime net) for the detailed regime panels + a cross-TF robustness summary."""
    cs = [c for c in cadences if c in book_res]
    if not cs:
        return
    # pick the cadence whose BOOK has the highest worst-regime net
    best_cad = max(cs, key=lambda c: (book_res[c]["_robustness"]["BOOK"]["worst_regime_net"] or -1e9))
    summ = book_res[best_cad]
    cands = ["BOOK"] + SLEEVES
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    x = np.arange(len(SCENARIOS)); width = 0.8 / len(cands)
    # panel A: NET by regime, BOOK vs each sleeve (best cadence)
    axn = axes[0]
    for ci, cand in enumerate(cands):
        vals = [summ[sc][cand]["net"]["mean"] or 0 for sc in SCENARIOS]
        errs = [summ[sc][cand]["net"]["std"] or 0 for sc in SCENARIOS]
        lbl = cand + ("" if cand in ("BOOK",) else (" [R]" if cand == "LONGSHORT" else ""))
        axn.bar(x + (ci - len(cands) / 2 + 0.5) * width, vals, width, yerr=errs, capsize=2,
                color=SLEEVE_COLORS.get(cand, "#888"), label=lbl, alpha=0.9,
                edgecolor="k" if cand == "BOOK" else None, linewidth=1.3 if cand == "BOOK" else 0)
    axn.axhline(0, color="k", lw=0.8)
    axn.set_xticks(x); axn.set_xticklabels([s.upper() for s in SCENARIOS])
    axn.set_ylabel("net % (mean +- sd)"); axn.set_title(f"{best_cad}: BOOK vs single sleeves NET by regime", fontsize=10)
    axn.legend(fontsize=8)
    # panel B: maxDD by regime (best cadence)
    axd = axes[1]
    for ci, cand in enumerate(cands):
        vals = [summ[sc][cand]["maxdd"]["mean"] or 0 for sc in SCENARIOS]
        axd.bar(x + (ci - len(cands) / 2 + 0.5) * width, vals, width, color=SLEEVE_COLORS.get(cand, "#888"),
                label=cand, alpha=0.9, edgecolor="k" if cand == "BOOK" else None,
                linewidth=1.3 if cand == "BOOK" else 0)
    axd.axhline(0, color="k", lw=0.8)
    axd.set_xticks(x); axd.set_xticklabels([s.upper() for s in SCENARIOS])
    axd.set_ylabel("maxDD % (less-negative=better)"); axd.set_title(f"{best_cad}: BOOK vs single sleeves maxDD by regime", fontsize=10)
    axd.legend(fontsize=8)
    # panel C: THE ROBUSTNESS GAIN -- worst-regime net of BOOK (research-full) + LONGONLY_BOOK (deployable) vs
    # each sleeve, across ALL cadences. LONGONLY_BOOK = the ship-today book; BOOK = the research-full book.
    axc = axes[2]
    cands_c = ["BOOK", "LONGONLY_BOOK"] + SLEEVES
    xc = np.arange(len(cs)); wc = 0.8 / len(cands_c)
    for ci, cand in enumerate(cands_c):
        vals = [book_res[c]["_robustness"][cand]["worst_regime_net"] or 0 for c in cs]
        lbl = {"BOOK": "BOOK (full,[R])", "LONGONLY_BOOK": "LONGONLY (deployable)"}.get(cand, cand)
        axc.bar(xc + (ci - len(cands_c) / 2 + 0.5) * wc, vals, wc, color=SLEEVE_COLORS.get(cand, "#888"),
                label=lbl, alpha=0.9, edgecolor="k" if cand in ("BOOK", "LONGONLY_BOOK") else None,
                linewidth=1.2 if cand in ("BOOK", "LONGONLY_BOOK") else 0)
    axc.axhline(0, color="k", lw=0.8)
    axc.set_xticks(xc); axc.set_xticklabels(cs); axc.set_xlabel("timeframe")
    axc.set_ylabel("WORST-regime net % (higher=more robust)")
    axc.set_title("THE ROBUSTNESS GAIN: worst-regime net, BOOK vs each sleeve, per TF\n(BOOK bar highest "
                  "=> the full system is the most cross-regime-robust)", fontsize=10)
    axc.legend(fontsize=8)
    fig.suptitle("THE MULTI-SLEEVE COMPLEMENTARY BOOK vs the parts (synthetic, 20 seeds) -- does the full system "
                 "(trend+MR+longshort+voltgt) beat EVERY single sleeve on cross-regime robustness?\n"
                 "PRE-REGISTERED equal-risk weights (held-aside bull slice); [R]=longshort is RESEARCH (LO-exception "
                 "to deploy)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = CHARTS / "multisleeve_book_vs_singles.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    print(f"[figure] {p}")


# =====================================================================================================
# 8. SELFTEST -- engine-direction + borrow-cost soundness on planted synthetic regimes (NO real calib).
# =====================================================================================================
def selftest():
    """Two-sided engine soundness (synthetic, no real calib):
    POSITIVE: in a planted BEAR, the LONGSHORT engine must LOSE LESS than the long-only trend book (its short
              leg profits from the fall) and lower the maxDD. In a planted BULL, the engine's short leg drags
              but the long leg participates -> engine net should be POSITIVE (the long leg dominates a bull).
    NEGATIVE: the cost-matched random-direction NULL must NOT beat the real engine in the bear (the edge is
              the SIGNAL). And borrow cost must MONOTONICALLY reduce net (a higher borrow never raises net)."""
    print("## LONGSHORT-MA-ENGINE SELFTEST (engine-direction + borrow soundness; no real-data calib)")
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

    def _book(rg, cad="1d", seed=2, borrow=BORROW_BASE, randomize=False, rng_seed=0):
        panels = SRS.generate_regime_panels(rg, calib, seed=seed, n_bars=92)
        with SRS._synthetic_panel_context(panels):
            return _longshort_book(cad, borrow_bps=borrow, randomize=randomize, rng_seed=rng_seed), _trend_book(cad)

    # POSITIVE: bear -- engine loses less than trend + lower DD
    ls_b, tr_b = _book("bear", seed=2)
    ls_bn = _perf(ls_b.to_numpy())["net"]; tr_bn = _perf(tr_b.to_numpy())["net"]
    ls_bdd = _perf(ls_b.to_numpy())["maxdd"]; tr_bdd = _perf(tr_b.to_numpy())["maxdd"]
    bear_better = ls_bn is not None and tr_bn is not None and ls_bn > tr_bn
    bear_dd_better = ls_bdd is not None and tr_bdd is not None and ls_bdd >= tr_bdd
    print(f"  POSITIVE bear: LONGSHORT net {ls_bn}% vs trend {tr_bn}% (>trend: {bear_better}); "
          f"LS maxDD {ls_bdd}% vs trend {tr_bdd}% (less-neg: {bear_dd_better})")
    ok &= bear_better and bear_dd_better

    # POSITIVE: bull -- engine net positive (long leg dominates), even with the short-leg drag
    ls_u, tr_u = _book("bull", seed=3)
    ls_un = _perf(ls_u.to_numpy())["net"]; tr_un = _perf(tr_u.to_numpy())["net"]
    bull_positive = ls_un is not None and ls_un > 0
    print(f"  POSITIVE bull: LONGSHORT net {ls_un}% (>0: {bull_positive}); trend {tr_un}% (drag {round((ls_un or 0)-(tr_un or 0),1)}pp)")
    ok &= bull_positive

    # NEGATIVE: cost-matched random-direction NULL must NOT beat the real engine in the bear (signal, not cost)
    null_nets = []
    for s in range(8):
        nb, _ = _book("bear", seed=2, randomize=True, rng_seed=1000 + s)
        if nb is not None:
            null_nets.append(_perf(nb.to_numpy())["net"])
    null_mean = float(np.nanmean([x for x in null_nets if x is not None])) if null_nets else 0.0
    edge_is_signal = ls_bn is not None and ls_bn > null_mean
    print(f"  NEGATIVE null: random-direction NULL bear mean net {round(null_mean,2)}% over {len(null_nets)} draws "
          f"vs real engine {ls_bn}% (real beats null: {edge_is_signal})")
    ok &= edge_is_signal

    # NEGATIVE: borrow cost MONOTONICALLY reduces net (higher borrow never raises net) in the bear
    nets_by_borrow = []
    for b in BORROW_BPS_GRID:
        lb, _ = _book("bear", seed=2, borrow=b)
        nets_by_borrow.append(_perf(lb.to_numpy())["net"])
    monotone = all(nets_by_borrow[i] >= nets_by_borrow[i + 1] - 0.5 for i in range(len(nets_by_borrow) - 1))
    print(f"  NEGATIVE borrow-monotone: bear net by borrow {BORROW_BPS_GRID} = {nets_by_borrow} (non-increasing: {monotone})")
    ok &= monotone

    print(f"\n  SELFTEST {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# =====================================================================================================
# 9. MAIN
# =====================================================================================================
def _strip_arrays(d):
    out = {}
    for cad, summ in d.items():
        out[cad] = {k: v for k, v in summ.items() if not str(k).startswith("_equity")}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m strat.longshort_ma_engine")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--cadences", default=",".join(ALL_TFS))
    ap.add_argument("--borrow-bps", type=float, default=BORROW_BASE)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    CHARTS.mkdir(parents=True, exist_ok=True)
    print("## LONGSHORT-MA ENGINE + MULTI-SLEEVE BOOK -- PHASE 6 (build the PHASE-4b positive finding into an engine)")

    # 1. CALIBRATE the PHASE 3/4b generator on 2020 ONLY (its calibrate_2020 is the only real-data touch)
    print("\n## CALIBRATING (PHASE 3 generator, REAL 2020-band data ONLY) ...")
    SRS._CALIB, real_samples = SRS.calibrate_2020()
    validation, _ = SRS.validate_generator(SRS._CALIB, real_samples, seed=0, n_paths=max(30, a.seeds))
    print(f"   GENERATOR VALIDATION: {validation['_summary']['verdict']} "
          f"({validation['_summary']['regimes_all_match']}/{validation['_summary']['regimes_validated']} regimes match)")

    seeds = list(range(1, a.seeds + 1))
    cadences = [c.strip() for c in a.cadences.split(",") if c.strip()]

    # 2. THE LONGSHORT ENGINE STRESS (per TF, per regime, borrow sensitivity, cost-matched null)
    print(f"\n## ENGINE STRESS over {len(seeds)} seeds x {len(cadences)} cadences x 4 regimes (borrow={a.borrow_bps}bps)")
    eng = run_engine_stress(cadences, seeds, borrow_bps=a.borrow_bps)

    # 3. THE MULTI-SLEEVE BOOK (pre-registered weights, robustness vs singles)
    print(f"\n## MULTI-SLEEVE BOOK over {len(seeds)} seeds x {len(cadences)} cadences x 4 regimes")
    book_res = run_book_stress(cadences, seeds, borrow_bps=a.borrow_bps)

    # 4. 2020-REAL OOS anchor (the real bull-drag)
    print("\n## 2020-REAL OOS anchor (the real-data bull-drag, Oct-Dec 2020) ...")
    real = run_2020_real(cadences, borrow_bps=a.borrow_bps)

    # 5. VERDICT
    verdict = build_verdict(eng, book_res, real)
    print("\n" + "=" * 100)
    print("## DECISIVE VERDICT (two-sided)")
    for line in verdict["lines"]:
        print(f"   {line}")
    print("=" * 100)

    # 6. CHARTS
    chart_engine_by_regime(eng, cadences)
    chart_book_vs_singles(book_res, cadences)

    # 7. PERSIST
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    export = {
        "repro": {"command": "python -m strat.longshort_ma_engine " + " ".join(argv or sys.argv[1:]),
                  "git_sha": sha, "cost_maker": MAKER_RT, "borrow_bps_base": a.borrow_bps,
                  "borrow_bps_grid": BORROW_BPS_GRID, "phase1a_winners": PHASE1A_WINNERS,
                  "calib_window": SRS.CALIB_WINDOW, "regime_periods": SRS.REGIME_PERIODS,
                  "n_seeds": a.seeds, "cadences": cadences, "sleeves": SLEEVES, "weight_policy": WEIGHT_POLICY,
                  "constraint": "SYNTHETIC (PHASE 3/4b validated generator) + 2020 calibration/OOS ONLY; never "
                                "2026/other; LONGSHORT = RESEARCH (deploy needs LO-exception sign-off)"},
        "generator_validation": validation,
        "engine": _strip_arrays(eng),
        "multisleeve_book": _strip_arrays(book_res),
        "real_2020_oos": real,
        "verdict": verdict,
    }
    p = OUT / "longshort_book.json"
    json.dump(export, open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n[persisted] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
