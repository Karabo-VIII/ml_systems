"""src/strat/complementary_sleeve_search.py -- PHASE 4b: THE TRUE-COMPLEMENT SEARCH.

THE OPEN THREAD PHASE 3 SURFACED: the long-only MR sleeve is a BEAR LIABILITY. On the validated synthetic
bear (calibrated to the 2020 COVID crash) the trend+MR STATIC blend's maxDD was -11.3pp WORSE than
trend-alone -- because BOTH sleeves are long, so MR cannot rescue trend's down days; it ADDS losses (it
buys falling knives). The user's CORE vision is "if one is out, the other captures the gap." PHASE 3 proved
a LONG-ONLY complement CANNOT fill a BEAR gap: engagement-anticorrelation (MR engages when trend is flat)
is NOT enough -- what is needed is RETURN-anticorrelation (the complement POSTS POSITIVE RETURNS when trend
LOSES).

THE QUESTION (this module): which candidate sleeve TRULY complements the trend MA book across ALL regimes
-- i.e. has genuinely anti-correlated RETURNS (wins when trend loses), lowers combined maxDD across the
FULL regime mix (not just chop), and posts a POSITIVE return in the bear where trend bleeds? And: HOW MUCH
of the complementarity gap is the LONG-ONLY CONSTRAINT itself -- quantify trend+SHORT (or long-short) vs
trend+MR(long-only) across the mix.

CANDIDATE COMPLEMENTS (vs the deployable trend MA book):
  (a) SHORT_MA      -- inverse-trend: SHORT on MA-cross-DOWN (the mechanical inverse of the trend signal).
  (b) LONGSHORT_MA  -- symmetric trend both directions (long up-cross + short down-cross).
  (c) CASH_GATE     -- cash-defensive regime gate: hold the trend book, but go FLAT (cash) when a PAST-ONLY
                       bear regime is detected. Genuine RETURN gap-fill = not bleeding when trend bleeds.
  (d) VOLTGT_DEF    -- inverse-vol / vol-target defensive overlay on the trend book (scale down in high vol).
  (e) MR_LONG       -- the established long-only MR sleeve (the BEAR-LIABILITY baseline, for contrast).

HOW IT STAYS HONEST (binding):
  - SYNTHETIC test surface ONLY, from PHASE 3's VALIDATED generator (synthetic_regime_stress, 3/3 regimes
    match real-2020 stylized facts). CALIBRATION reads REAL 2020-BAND data ONLY (window-fenced inside that
    module). NO 2026/other data is ever read here.
  - The candidate sleeves run through the EXACT deployable position mechanics (MA-cross signal -> trail-stop
    -> min_hold -> lag-1 -> maker cost) via the SAME _synthetic_panel_context monkeypatch PHASE 3 uses. The
    SHORT sleeve mirrors the long-only trail-stop on the inverted price (a rise against a short triggers the
    stop exactly as a drop against a long does). No MtM double-count (the deployable cost convention).
  - >=20 seeds; report DISTRIBUTIONS (mean +- spread + WORST seed), never a cherry-picked path.
  - TWO-SIDED: the SHORT/long-short sleeves are FLAGGED RESEARCH -- they violate the standing LONG-ONLY +
    spot constraint, so DEPLOYING one needs the user's explicit long-only-exception sign-off. We explore
    them for the LEARNING (the user explicitly expanded scope for max learnings) and QUANTIFY the value of
    relaxing the constraint -- we do NOT silently recommend shipping a short book.

THE DECISIVE METRIC (the KEY one, the PHASE 3 lesson): corr(candidate, trend) on RETURNS PER REGIME. A true
complement is RETURN-anticorrelated (corr < 0, especially in the bear). Engagement-anticorrelation is not
enough -- a long-only complement can be engagement-anticorrelated yet still lose in the bear.

CONSTRAINTS (user mandate, BINDING): synthetic (PHASE 3 generator) + 2020 calibration ONLY; charts (PNG);
no emoji (cp1252); RWYB; do NOT git commit. SHORT sleeve = RESEARCH (deploy needs LO-exception sign-off).

RWYB:
  python -m strat.complementary_sleeve_search --selftest               # sleeve-direction soundness (no calib)
  python -m strat.complementary_sleeve_search --seeds 20 --cadences 1d # the full true-complement search
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse PHASE 3's validated generator + the _panel monkeypatch context (NOT reinvented).
import strat.synthetic_regime_stress as SRS                              # noqa: E402
import strat.ma_2020_breakdown as M2                                    # noqa: E402
import strat.deep2020_complementarity as COMP                           # noqa: E402
from strat.portfolio_replay import MAKER_RT, TAKER_RT, apply_trail_stop  # noqa: E402
from strat.replay_distinct_grid import distinct_specs                    # noqa: E402
from strat.structural_fixes import min_hold                              # noqa: E402
from strat.ma_type_upgrade import _MA, _nums                             # noqa: E402
from strat.data_expansion import block_bootstrap_distribution           # noqa: E402

OUT = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
CHARTS = OUT / "charts"
SYMS = COMP.SYMS

__contract__ = {
    "kind": "complementary_sleeve_search",
    "inputs": {
        "test_surface": "PHASE 3 VALIDATED synthetic generator (synthetic_regime_stress) -- bull/bear/chop/"
                        "stitched, calibrated on REAL 2020-band data ONLY (no 2026/other read here)",
        "reference_book": "the deployable trend MA sleeve (COMP._trend_sleeve mechanics: MA-cross-up -> "
                          "trail-stop -> min_hold -> lag-1 -> maker)",
        "candidates": "SHORT_MA (inverse-trend short) / LONGSHORT_MA (symmetric) / CASH_GATE (past-only "
                      "bear-regime cash gate on trend) / VOLTGT_DEF (vol-target defensive overlay on trend) "
                      "/ MR_LONG (the long-only bear-liability baseline)",
    },
    "outputs": {
        "return_corr_by_regime": "corr(candidate, trend) on RETURNS per regime -- THE KEY metric (a true "
                                 "complement is RETURN-anticorrelated, not just engagement-anticorrelated)",
        "bear_gap_fill": "does the candidate post a POSITIVE return in the bear where trend loses? (genuine "
                         "return gap-fill, vs the long-only MR which loses MORE)",
        "combined_book_by_regime": "trend+candidate net + maxDD + p05 across the FULL regime mix, 20-seed "
                                   "mean +- spread + worst seed",
        "long_only_relaxation_value": "trend+SHORT (or long-short) vs trend+MR(long-only): how much net / "
                                      "maxDD / p05 / worst-seed is the long-only constraint costing us?",
    },
    "invariants": {
        "synthetic_2020_calib_only": "test surface from PHASE 3's generator; real data read ONLY in its "
                                     "calibrate_2020 (window-fenced to 2020); never 2026/other here",
        "exact_deployable_mechanics": "candidates run through the SAME signal->trail->min_hold->lag1->maker "
                                      "pipeline as the deployable trend book; the SHORT trail mirrors the "
                                      "long trail on inverted price; no MtM double-count",
        "return_anticorr_is_the_test": "the PHASE 3 lesson -- engagement-anticorrelation is NOT enough; a "
                                       "true complement WINS (positive return) when trend LOSES",
        "short_is_research_not_deploy": "SHORT/long-short VIOLATE the long-only+spot constraint -> FLAGGED "
                                        "RESEARCH; deploying needs the user's explicit LO-exception sign-off",
        "distributions_not_single_paths": ">=20 seeds; report mean +- spread + WORST seed; never cherry-pick",
        "two_sided_honest": "if NO candidate truly complements across the mix, that is the honest finding; "
                            "if SHORT does, the value of relaxing long-only is QUANTIFIED, not buried",
    },
}

# Candidate sleeve registry. LONG_ONLY flag = does it respect the standing long-only+spot constraint?
CANDIDATES = ["SHORT_MA", "LONGSHORT_MA", "CASH_GATE", "VOLTGT_DEF", "MR_LONG"]
LONG_ONLY = {"SHORT_MA": False, "LONGSHORT_MA": False, "CASH_GATE": True, "VOLTGT_DEF": True, "MR_LONG": True}
SCENARIOS = ["bull", "bear", "chop", "stitched"]


# =====================================================================================================
# 1. THE DEPLOYABLE SIGNAL PIPELINE (shared) -- MA-cross signal -> trail-stop -> min_hold -> lag1 -> maker.
#    This is the EXACT mechanics COMP._trend_sleeve uses; factored here so each candidate varies only the
#    SIGNAL / DIRECTION, not the cost/position convention (so comparisons are apples-to-apples).
# =====================================================================================================
_SLOW_MA_SPECS_CACHE = None


def _slow_ma_specs():
    """The deployable slow EMA family spec names (same as COMP._trend_sleeve). distinct_specs is ~3.6s/call,
    and this is invoked per-asset-per-book, so CACHE it module-level (the specs are deterministic -- no panel
    dependence). This is a pure speed fix; the returned specs are identical to recomputing each time."""
    global _SLOW_MA_SPECS_CACHE
    if _SLOW_MA_SPECS_CACHE is None:
        ma_cfg = {}
        for fam in ("2MA", "3MA"):
            ma_cfg.update(distinct_specs(fam, 0.15, max_n=40))
        _SLOW_MA_SPECS_CACHE = [n for n in ma_cfg if 60 <= max(_nums(n)) < 150]
    return _SLOW_MA_SPECS_CACHE


def _ma_cross_long_held(c2):
    """Per-config LONG hold series from a SLOW MA up-cross (mas[0] > mas[1] [> mas[2]]), trail-stop(0.10),
    min_hold(12) -- the EXACT deployable trend-sleeve signal. Returns a list of per-config held arrays."""
    slow = _slow_ma_specs()
    uniq = sorted({p for n in slow for p in _nums(n)})
    cache = {p: _MA["EMA"](c2, p) for p in uniq}
    helds = []
    for name in slow:
        pp = _nums(name); mas = [cache[p] for p in pp]
        h0 = np.nan_to_num((mas[0] > mas[1]) if len(pp) == 2 else ((mas[0] > mas[1]) & (mas[1] > mas[2]))).astype(np.int8)
        h0 = min_hold(apply_trail_stop(h0.copy(), c2, 0.10)[0].astype(np.int8), 12).astype(np.float64)
        helds.append(h0)
    return helds


def _apply_trail_stop_short(held, close, trail):
    """SHORT-side mirror of portfolio_replay.apply_trail_stop. For a short, the favorable move is DOWN, so we
    track the LOW-WATER (lowest close since entry) and FORCE flat once close rises > trail ABOVE that low
    (a rally against the short). Same episode/re-arm semantics as the long version. Causal (close[i] only).
    NOTE: the long apply_trail_stop CANNOT be reused on -close -- its MULTIPLICATIVE threshold hw*(1-trail)
    is wrong-signed on negative prices (it stopped out every bar; the bug this replaces). This is the correct
    additive-on-the-low symmetric short stop."""
    out = np.asarray(held, dtype=np.int8).copy()
    inpos = False
    lw = 0.0
    stopped = False
    stop_idx = set()
    for i in range(len(held)):
        if not held[i]:
            inpos, stopped = False, False
            out[i] = 0
            continue
        if stopped:
            out[i] = 0
            continue
        if not inpos:
            inpos, lw = True, float(close[i])
        lw = min(lw, float(close[i]))                       # low-water = best price for a short
        if float(close[i]) > lw * (1.0 + trail):            # rallied > trail above the low -> stop the short
            out[i] = 0
            inpos, stopped = False, True
            stop_idx.add(i)
        else:
            out[i] = 1
    return out, stop_idx


def _ma_cross_short_held(c2):
    """Per-config SHORT hold series from a SLOW MA DOWN-cross (the mechanical INVERSE of the trend signal):
    mas[0] < mas[1] [< mas[2]]. Trail-stop is the SHORT mirror (_apply_trail_stop_short: stop when price
    rallies > trail above the low-water), then min_hold(12) -- structurally identical to the deployable trend
    sleeve, just direction-flipped. Returns per-config held arrays (1 = short-engaged that bar)."""
    slow = _slow_ma_specs()
    uniq = sorted({p for n in slow for p in _nums(n)})
    cache = {p: _MA["EMA"](c2, p) for p in uniq}
    helds = []
    for name in slow:
        pp = _nums(name); mas = [cache[p] for p in pp]
        s0 = np.nan_to_num((mas[0] < mas[1]) if len(pp) == 2 else ((mas[0] < mas[1]) & (mas[1] < mas[2]))).astype(np.int8)
        s0 = min_hold(_apply_trail_stop_short(s0.copy(), c2, 0.10)[0].astype(np.int8), 12).astype(np.float64)
        helds.append(s0)
    return helds


def _book_from_helds(helds, ret, win, direction):
    """Aggregate per-config held arrays into a u10-asset bar net series with the deployable cost convention:
    position lagged 1 bar, maker half-spread charged per flip, PnL = direction * pos * ret. direction=+1 for
    long, -1 for short. Returns the per-config-mean bar net over the scored window."""
    cfg_nets = []
    for h0 in helds:
        pos = np.zeros(len(h0)); pos[1:] = h0[:-1]
        flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
        cfg_nets.append((direction * pos * ret - flips * (MAKER_RT / 2.0))[win])
    return np.mean(cfg_nets, axis=0)


def _daily(net_bar: pd.Series) -> pd.Series:
    return net_bar.resample("1D").apply(lambda x: float(np.prod(1 + x) - 1)).dropna()


# =====================================================================================================
# 2. THE CANDIDATE SLEEVES -- each returns a daily-net pd.Series over the (synthetic) window.
#    Run INSIDE the _synthetic_panel_context (so M2._panel yields synthetic panels + windows are widened).
# =====================================================================================================
def _panel_window(sym, cad):
    """Load a (synthetic, via the patched _panel) panel and compute the scored window + per-bar return.
    Returns (c2, ms2, ret, win, idx) or None. Mirrors COMP._trend_sleeve's windowing exactly."""
    s_ms = pd.Timestamp(COMP.WIN[0]).value // 10**6
    e_ms = pd.Timestamp(COMP.WIN[1]).value // 10**6
    try:
        o, h, l, c, ms = M2._panel(sym, cad)
    except Exception:
        return None
    e = int(np.searchsorted(ms, e_ms)); s0 = max(0, int(np.searchsorted(ms, s_ms)) - COMP.WARMUP)
    c2, ms2 = c[s0:e], ms[s0:e]
    if len(c2) < 40:
        return None
    win = ms2 >= s_ms
    if win.sum() < 30:
        return None
    ret = np.zeros(len(c2)); ret[1:] = c2[1:] / c2[:-1] - 1.0
    idx = pd.to_datetime(ms2[win], unit="ms")
    return c2, ms2, ret, win, idx


def _trend_long_book(cad):
    """The REFERENCE trend MA book (re-derived here with the shared pipeline so the candidates that REUSE
    its per-bar net -- CASH_GATE / VOLTGT_DEF -- are bit-consistent with the reference). Equals
    COMP._trend_sleeve's net by construction."""
    per = []
    for sym in SYMS:
        pw = _panel_window(sym, cad)
        if pw is None:
            continue
        c2, ms2, ret, win, idx = pw
        helds = _ma_cross_long_held(c2)
        per.append(pd.Series(_book_from_helds(helds, ret, win, +1), index=idx))
    if not per:
        return None
    net_bar = pd.concat(per, axis=1).mean(axis=1, skipna=True)
    return _daily(net_bar)


def _short_ma_book(cad):
    """(a) SHORT_MA: short on MA-cross-DOWN (inverse-trend). NOT long-only -- RESEARCH."""
    per = []
    for sym in SYMS:
        pw = _panel_window(sym, cad)
        if pw is None:
            continue
        c2, ms2, ret, win, idx = pw
        helds = _ma_cross_short_held(c2)
        per.append(pd.Series(_book_from_helds(helds, ret, win, -1), index=idx))
    if not per:
        return None
    return _daily(pd.concat(per, axis=1).mean(axis=1, skipna=True))


def _longshort_ma_book(cad):
    """(b) LONGSHORT_MA: symmetric trend -- long up-cross + short down-cross, equal-weight. NOT long-only."""
    per = []
    for sym in SYMS:
        pw = _panel_window(sym, cad)
        if pw is None:
            continue
        c2, ms2, ret, win, idx = pw
        long_net = _book_from_helds(_ma_cross_long_held(c2), ret, win, +1)
        short_net = _book_from_helds(_ma_cross_short_held(c2), ret, win, -1)
        per.append(pd.Series(0.5 * long_net + 0.5 * short_net, index=idx))
    if not per:
        return None
    return _daily(pd.concat(per, axis=1).mean(axis=1, skipna=True))


def _bh_eqw_daily(cad):
    """Equal-weight u10 synthetic buy-hold daily return -- the regime-detection substrate for CASH_GATE and
    the realized-vol substrate for VOLTGT_DEF (a market observable, past-only when used)."""
    books = []
    for sym in SYMS:
        pw = _panel_window(sym, cad)
        if pw is None:
            continue
        c2, ms2, ret, win, idx = pw
        books.append(pd.Series(ret[win], index=idx))
    if not books:
        return None
    bar = pd.concat(books, axis=1).mean(axis=1, skipna=True)
    return _daily(bar)


def _cash_gate_book(cad, trend_daily, lookback=10, bear_thresh=-0.02):
    """(c) CASH_GATE: hold the trend book, but go to CASH (0 return) on days a PAST-ONLY bear regime is
    detected. Regime signal = the trailing `lookback`-day mean daily return of the equal-weight u10 book,
    SHIFTED 1 day (causal). If trailing mean < bear_thresh -> bear -> sit in cash that day. LONG-ONLY (cash
    is a valid long-only state). The genuine RETURN gap-fill: it does not bleed when trend bleeds in a bear.
    Honest: a regime gate cannot post POSITIVE return in a bear (cash = 0), but 0 >> trend's negative bleed.
    """
    if trend_daily is None:
        return None
    bh = _bh_eqw_daily(cad)
    if bh is None:
        return None
    df = pd.concat([trend_daily.rename("t"), bh.rename("bh")], axis=1).dropna()
    if len(df) < lookback + 3:
        return None
    trail = df["bh"].rolling(lookback, min_periods=lookback).mean().shift(1)   # causal: decided on PAST bars
    in_bear = (trail < bear_thresh).fillna(False).to_numpy()
    out = df["t"].to_numpy().copy()
    out[in_bear] = 0.0                                                          # cash on detected-bear days
    return pd.Series(out, index=df.index)


def _voltgt_def_book(cad, trend_daily, lookback=10, target_vol=0.02, max_scale=1.0):
    """(d) VOLTGT_DEF: a vol-target DEFENSIVE overlay on the trend book -- scale the trend daily return by
    min(max_scale, target_vol / trailing_realized_vol) using a PAST-ONLY (shifted) realized vol of the
    equal-weight u10 book. In high-vol (bear/crash) the scale drops, cutting exposure; capped at max_scale
    (we DEFEND, never lever up). LONG-ONLY (a scaled-down long is still long). Honest: like CASH_GATE this
    DAMPENS the bear bleed; it does not turn trend's loss into a gain."""
    if trend_daily is None:
        return None
    bh = _bh_eqw_daily(cad)
    if bh is None:
        return None
    df = pd.concat([trend_daily.rename("t"), bh.rename("bh")], axis=1).dropna()
    if len(df) < lookback + 3:
        return None
    rv = df["bh"].rolling(lookback, min_periods=lookback).std().shift(1)        # causal trailing realized vol
    scale = np.clip(target_vol / (rv.to_numpy() + 1e-9), 0.0, max_scale)
    scale = np.where(np.isfinite(scale), scale, max_scale)                      # warmup NaN -> full (defensive default never >1)
    return pd.Series(df["t"].to_numpy() * scale, index=df.index)


def _mr_long_book(cad):
    """(e) MR_LONG: the deployable long-only MR oscillator sleeve (the BEAR-LIABILITY baseline). Reuses
    COMP._mr_sleeve EXACTLY (its daily net). LONG-ONLY."""
    mnet, mexp = COMP._mr_sleeve(cad)
    return mnet


def build_all_books(cad):
    """Build the trend reference + all candidate daily-net books on the (synthetic, patched) panels for one
    cadence. Returns {name -> daily pd.Series} aligned later by the caller. Run INSIDE _synthetic_panel_context."""
    trend = _trend_long_book(cad)
    books = {
        "TREND": trend,
        "SHORT_MA": _short_ma_book(cad),
        "LONGSHORT_MA": _longshort_ma_book(cad),
        "CASH_GATE": _cash_gate_book(cad, trend),
        "VOLTGT_DEF": _voltgt_def_book(cad, trend),
        "MR_LONG": _mr_long_book(cad),
    }
    return books


# =====================================================================================================
# 3. METRICS -- per-regime RETURN-correlation, bear gap-fill, combined book perf, long-only relaxation value.
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


def run_on_panel(panels, cad):
    """Run the trend + all candidates on one synthetic panel-set. Returns a dict per name with:
    aligned daily arrays + standalone _perf, the trend array, plus the combined trend+candidate book
    (50/50 daily blend) _perf and the per-candidate corr-to-trend on RETURNS. All aligned on common days."""
    with SRS._synthetic_panel_context(panels):
        books = build_all_books(cad)
    trend = books["TREND"]
    if trend is None:
        return None
    # align every candidate with trend on the common daily index
    out = {"n_days": None, "TREND": {"arr": None, "perf": None}}
    # first, the trend standalone (aligned to itself)
    base_idx = trend.index
    out["TREND"]["arr"] = trend.to_numpy()
    out["TREND"]["perf"] = _perf(trend.to_numpy())
    out["n_days"] = int(len(trend))
    for name in CANDIDATES:
        b = books.get(name)
        if b is None:
            out[name] = None
            continue
        df = pd.concat([trend.rename("t"), b.rename("c")], axis=1).dropna()
        if len(df) < 8:
            out[name] = None
            continue
        t = df["t"].to_numpy(); c = df["c"].to_numpy()
        corr = float(np.corrcoef(t, c)[0, 1]) if (np.std(t) > 1e-12 and np.std(c) > 1e-12) else 0.0
        combined = 0.5 * t + 0.5 * c                          # 50/50 trend+candidate book
        out[name] = {
            "arr": c, "perf": _perf(c),
            "corr_to_trend": round(corr, 3) if np.isfinite(corr) else None,
            "combined_perf": _perf(combined),
            "combined_arr": combined,
            "trend_arr_aligned": t,
            "n": int(len(df)),
        }
    return out


# =====================================================================================================
# 4. THE STRESS RUN -- per regime + stitched, over many seeds, distributions not single paths.
# =====================================================================================================
def _dist(vals):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return {"mean": None, "std": None, "worst": None, "median": None, "n": 0}
    return {"mean": round(float(np.mean(v)), 2), "std": round(float(np.std(v)), 2),
            "worst": round(float(np.min(v)), 2), "median": round(float(np.median(v)), 2),
            "p25": round(float(np.percentile(v, 25)), 2), "n": int(v.size)}


def run_search(cadences, seeds, n_bars=None):
    """For each cadence: generate {bull,bear,chop,stitched} synthetic panels over `seeds` seeds (PHASE 3
    generator), build the trend + candidate books, collect per-regime distributions of:
      - corr(candidate, trend) on RETURNS,
      - candidate standalone net (the bear gap-fill question: positive in bear?),
      - combined trend+candidate net / maxDD / p05,
      - trend-alone net / maxDD (the baseline the combined must improve on).
    Returns the full results dict."""
    results = {}
    for cad in cadences:
        print(f"\n########## CADENCE {cad} -- true-complement search ({len(seeds)} seeds) ##########")
        # accumulators
        acc = {sc: {"trend": {"net": [], "maxdd": [], "p05": []}} for sc in SCENARIOS}
        for sc in SCENARIOS:
            for name in CANDIDATES:
                acc[sc][name] = {"corr": [], "net": [], "maxdd": [],
                                 "comb_net": [], "comb_maxdd": [], "comb_p05": [],
                                 "comb_minus_trend_dd": [], "comb_minus_trend_net": []}
        equity_example = {}
        for si, seed in enumerate(seeds):
            for sc in SCENARIOS:
                if sc == "stitched":
                    panels, _ = SRS.stitch_panels(SRS.STITCH_SEQUENCE, SRS._CALIB, seed, n_bars_each=n_bars)
                else:
                    nb = n_bars if n_bars else SRS.REGIME_BARS.get(sc, SRS.N_BARS_REGIME)
                    panels = SRS.generate_regime_panels(sc, SRS._CALIB, seed=seed, n_bars=nb)
                res = run_on_panel(panels, cad)
                if res is None:
                    continue
                tp = res["TREND"]["perf"]
                if tp["net"] is not None:
                    acc[sc]["trend"]["net"].append(tp["net"])
                    acc[sc]["trend"]["maxdd"].append(tp["maxdd"])
                    acc[sc]["trend"]["p05"].append(tp["p05"])
                for name in CANDIDATES:
                    r = res.get(name)
                    if r is None or r["perf"]["net"] is None:
                        continue
                    acc[sc][name]["corr"].append(r["corr_to_trend"])
                    acc[sc][name]["net"].append(r["perf"]["net"])
                    acc[sc][name]["maxdd"].append(r["perf"]["maxdd"])
                    cp = r["combined_perf"]
                    acc[sc][name]["comb_net"].append(cp["net"])
                    acc[sc][name]["comb_maxdd"].append(cp["maxdd"])
                    acc[sc][name]["comb_p05"].append(cp["p05"])
                    # paired combined-vs-trend (per seed): how much LESS-negative the combined DD / more net
                    if tp["net"] is not None and cp["maxdd"] is not None and tp["maxdd"] is not None:
                        acc[sc][name]["comb_minus_trend_dd"].append(cp["maxdd"] - tp["maxdd"])
                        acc[sc][name]["comb_minus_trend_net"].append(cp["net"] - tp["net"])
                # stash example equity (first seed) for charts
                if si == 0:
                    ex = {"TREND": list(np.cumprod(1 + res["TREND"]["arr"]) * 100 - 100)}
                    for name in CANDIDATES:
                        r = res.get(name)
                        if r is not None:
                            ex[name] = list(np.cumprod(1 + np.asarray(r["arr"])) * 100 - 100)
                            ex[name + "_COMB"] = list(np.cumprod(1 + np.asarray(r["combined_arr"])) * 100 - 100)
                    equity_example[sc] = ex
            print(f"   seed {seed} done ({si + 1}/{len(seeds)})", end="\r")
        print()
        results[cad] = _summarize(acc, equity_example)
        _print_table(cad, results[cad])
    return results


def _summarize(acc, equity_example):
    summ = {"_equity_example": equity_example}
    for sc in SCENARIOS:
        summ[sc] = {"trend": {"net": _dist(acc[sc]["trend"]["net"]),
                              "maxdd": _dist(acc[sc]["trend"]["maxdd"]),
                              "p05": _dist(acc[sc]["trend"]["p05"])}}
        for name in CANDIDATES:
            a = acc[sc][name]
            # frac of seeds the candidate posts a POSITIVE standalone return in this regime (the gap-fill test)
            nets = np.asarray([x for x in a["net"] if x is not None], float)
            frac_pos = round(float(np.mean(nets > 0)), 2) if nets.size else None
            # frac of seeds the COMBINED book has LESS-negative maxDD than trend-alone (true DD complement)
            dd_red = np.asarray([x for x in a["comb_minus_trend_dd"] if x is not None], float)
            frac_dd_better = round(float(np.mean(dd_red > 0)), 2) if dd_red.size else None
            summ[sc][name] = {
                "corr_to_trend": _dist(a["corr"]),
                "net": _dist(a["net"]),
                "maxdd": _dist(a["maxdd"]),
                "frac_seeds_positive": frac_pos,
                "combined_net": _dist(a["comb_net"]),
                "combined_maxdd": _dist(a["comb_maxdd"]),
                "combined_p05": _dist(a["comb_p05"]),
                "combined_minus_trend_dd": _dist(a["comb_minus_trend_dd"]),
                "combined_minus_trend_net": _dist(a["comb_minus_trend_net"]),
                "frac_seeds_combined_dd_better_than_trend": frac_dd_better,
            }
    return summ


def _print_table(cad, summ):
    for sc in SCENARIOS:
        t = summ[sc]["trend"]
        print(f"   --- {sc.upper()} ---  (trend-alone: net {t['net']['mean']}+-{t['net']['std']}%, "
              f"maxDD {t['maxdd']['mean']}%, p05 {t['p05']['mean']}%)")
        print(f"     {'candidate':13} {'corr2trend':>11} {'cand net':>10} {'%seed+':>7} "
              f"{'comb net':>10} {'comb DD':>9} {'comb-trend DD':>14} {'%DDbetter':>10}")
        for name in CANDIDATES:
            e = summ[sc][name]
            lo = "" if LONG_ONLY[name] else " [RESEARCH/short]"
            print(f"     {name:13} {str(e['corr_to_trend']['mean']):>11} "
                  f"{str(e['net']['mean']):>10} {str(e['frac_seeds_positive']):>7} "
                  f"{str(e['combined_net']['mean']):>10} {str(e['combined_maxdd']['mean']):>9} "
                  f"{str(e['combined_minus_trend_dd']['mean']):>14} "
                  f"{str(e['frac_seeds_combined_dd_better_than_trend']):>10}{lo}")


# =====================================================================================================
# 5. THE DECISIVE QUESTION (two-sided) -- which sleeve truly complements + the value of relaxing long-only.
# =====================================================================================================
def build_verdict(results):
    lines = []
    # aggregate the KEY metrics across cadences
    def agg(sc, name, key, sub="mean"):
        vals = []
        for cad in results:
            e = results[cad][sc].get(name, {}).get(key, {})
            if isinstance(e, dict) and e.get(sub) is not None:
                vals.append(e[sub])
        return float(np.mean(vals)) if vals else None

    lines.append("Q1: WHICH CANDIDATE IS RETURN-ANTICORRELATED TO TREND (the KEY test -- wins when trend loses)?")
    lines.append("    corr(candidate, trend) on RETURNS, per regime (negative = true complement):")
    for name in CANDIDATES:
        row = []
        for sc in SCENARIOS:
            c = agg(sc, name, "corr_to_trend")
            row.append(f"{sc}={c:+.2f}" if c is not None else f"{sc}=na")
        flag = "" if LONG_ONLY[name] else "  [RESEARCH/short]"
        lines.append(f"    {name:13}: " + "  ".join(row) + flag)

    lines.append("")
    lines.append("Q2: THE BEAR GAP-FILL -- does the candidate post a POSITIVE return in the BEAR where trend bleeds?")
    bear_trend_net = agg("bear", "trend", "net") if "bear" in results[list(results)[0]] else None
    # trend-alone bear net (same across candidates) -- pull from any
    tnet_bear = None
    for cad in results:
        tnet_bear = results[cad]["bear"]["trend"]["net"]["mean"]; break
    lines.append(f"    trend-alone BEAR net = {tnet_bear:+.1f}% (it BLEEDS -- this is the gap to fill)")
    for name in CANDIDATES:
        cn = agg("bear", name, "net")
        fp = None
        for cad in results:
            fp = results[cad]["bear"][name]["frac_seeds_positive"]; break
        flag = "" if LONG_ONLY[name] else "  [RESEARCH/short]"
        fills = "GENUINE FILL (positive in bear)" if (cn is not None and cn > 0) else "does NOT fill (loses/flat in bear)"
        lines.append(f"    {name:13}: bear net {cn:+.1f}% (positive in {fp:.0%} of seeds) -> {fills}{flag}"
                     if cn is not None else f"    {name:13}: na")

    lines.append("")
    lines.append("Q3: THE COMBINED trend+candidate BOOK across the FULL regime mix (does it LOWER DD vs trend-alone?):")
    lines.append("    (combined maxDD MINUS trend-alone maxDD; POSITIVE = combined dampens DD; per regime)")
    for name in CANDIDATES:
        row = []
        for sc in SCENARIOS:
            d = agg(sc, name, "combined_minus_trend_dd")
            row.append(f"{sc}={d:+.1f}" if d is not None else f"{sc}=na")
        flag = "" if LONG_ONLY[name] else "  [RESEARCH/short]"
        lines.append(f"    {name:13}: " + "  ".join(row) + "  (pp)" + flag)

    # THE DECISIVE two-sided answer: rank candidates by a FULL-MIX complement score.
    # score = mean over regimes of (combined_minus_trend_dd)  [DD dampening across the mix]
    #         + a bonus for bear return-anticorrelation (the PHASE 3 lesson) ... but keep it transparent:
    # we report the components, and pick the candidate that (i) is RETURN-anticorrelated in the bear AND
    # (ii) lowers combined DD across the mix AND (iii) does not blow up net.
    lines.append("")
    lines.append("Q4 (DECISIVE): which sleeve TRULY complements across regimes?")
    scoreboard = {}
    for name in CANDIDATES:
        bear_corr = agg("bear", name, "corr_to_trend")
        mix_dd = np.mean([agg(sc, name, "combined_minus_trend_dd") or 0.0 for sc in SCENARIOS])
        bear_net = agg("bear", name, "net")
        mix_comb_net = np.mean([agg(sc, name, "combined_net") or 0.0 for sc in SCENARIOS])
        scoreboard[name] = {
            "bear_return_corr": round(bear_corr, 3) if bear_corr is not None else None,
            "mean_combined_dd_dampening_pp": round(float(mix_dd), 2),
            "bear_standalone_net": round(bear_net, 2) if bear_net is not None else None,
            "mean_combined_net": round(float(mix_comb_net), 2),
            "long_only": LONG_ONLY[name],
            # "true complement": return-anticorrelated in bear (<0) AND lowers combined DD across the mix (>0)
            "true_complement_across_mix": bool(
                bear_corr is not None and bear_corr < 0.0 and mix_dd > 0.0),
        }
    # the long-only winners vs the (research) short winners
    lo_true = [n for n in CANDIDATES if scoreboard[n]["true_complement_across_mix"] and LONG_ONLY[n]]
    short_true = [n for n in CANDIDATES if scoreboard[n]["true_complement_across_mix"] and not LONG_ONLY[n]]
    for name in sorted(CANDIDATES, key=lambda n: -(scoreboard[n]["mean_combined_dd_dampening_pp"])):
        s = scoreboard[name]
        flag = "" if s["long_only"] else "  [RESEARCH/short -- needs LO-exception sign-off to deploy]"
        verdict = "TRUE COMPLEMENT" if s["true_complement_across_mix"] else "not a full-mix complement"
        lines.append(f"    {name:13}: bear-corr {s['bear_return_corr']}, mix-DD-dampening "
                     f"{s['mean_combined_dd_dampening_pp']:+}pp, bear-net {s['bear_standalone_net']}% -> {verdict}{flag}")

    # Q5: QUANTIFY the value of relaxing long-only -- trend+SHORT (or long-short) vs trend+MR(long-only).
    lines.append("")
    lines.append("Q5 (QUANTIFIED): the VALUE of relaxing long-only -- trend+SHORT/long-short vs trend+MR(long-only),")
    lines.append("    across the FULL regime mix (mean over bull/bear/chop/stitched):")
    relax = {}
    for ref in ("SHORT_MA", "LONGSHORT_MA"):
        comp_net = np.mean([agg(sc, ref, "combined_net") or 0.0 for sc in SCENARIOS])
        comp_dd = np.mean([agg(sc, ref, "combined_maxdd") or 0.0 for sc in SCENARIOS])
        mr_net = np.mean([agg(sc, "MR_LONG", "combined_net") or 0.0 for sc in SCENARIOS])
        mr_dd = np.mean([agg(sc, "MR_LONG", "combined_maxdd") or 0.0 for sc in SCENARIOS])
        # bear-specific (the regime where long-only fails)
        comp_bear_dd = agg("bear", ref, "combined_maxdd"); mr_bear_dd = agg("bear", "MR_LONG", "combined_maxdd")
        comp_bear_net = agg("bear", ref, "combined_net"); mr_bear_net = agg("bear", "MR_LONG", "combined_net")
        relax[ref] = {
            "mix_net_advantage_vs_trendMR": round(float(comp_net - mr_net), 2),
            "mix_dd_advantage_vs_trendMR_pp": round(float(comp_dd - mr_dd), 2),
            "bear_net_advantage_vs_trendMR": round(float((comp_bear_net or 0) - (mr_bear_net or 0)), 2),
            "bear_dd_advantage_vs_trendMR_pp": round(float((comp_bear_dd or 0) - (mr_bear_dd or 0)), 2),
        }
        lines.append(f"    trend+{ref:12} vs trend+MR(LO): mix net {relax[ref]['mix_net_advantage_vs_trendMR']:+.1f}pp, "
                     f"mix DD {relax[ref]['mix_dd_advantage_vs_trendMR_pp']:+.1f}pp, "
                     f"BEAR net {relax[ref]['bear_net_advantage_vs_trendMR']:+.1f}pp, "
                     f"BEAR DD {relax[ref]['bear_dd_advantage_vs_trendMR_pp']:+.1f}pp  [SHORT = RESEARCH]")

    # HEADLINE
    if short_true and not lo_true:
        headline = (f"LONG-ONLY IS THE BINDING CONSTRAINT ON TRUE COMPLEMENTARITY: the ONLY candidate(s) that are "
                    f"RETURN-anticorrelated to trend in the bear AND lower combined DD across the full mix are "
                    f"{short_true} -- all SHORT/long-short (RESEARCH; deploy needs the LO-exception sign-off). No "
                    f"long-only candidate truly complements in the bear (cash/vol-gates only DAMPEN the bleed; "
                    f"long-only MR ADDS to it). The value of relaxing long-only is quantified in Q5. "
                    f"DEFENSIVE long-only gates (CASH_GATE / VOLTGT_DEF) are the best WITHIN-CONSTRAINT option.")
    elif lo_true:
        headline = (f"A LONG-ONLY TRUE COMPLEMENT EXISTS: {lo_true} is RETURN-anticorrelated to trend in the bear AND "
                    f"lowers combined DD across the mix WITHOUT relaxing long-only. (Short candidates {short_true} may "
                    f"do better -- Q5 quantifies -- but a deployable-today long-only answer exists.)")
    else:
        headline = (f"NO TRUE FULL-MIX COMPLEMENT among the candidates (none is bear-return-anticorrelated AND "
                    f"mix-DD-dampening). The best WITHIN-CONSTRAINT option is the DEFENSIVE long-only gate that "
                    f"dampens the bear bleed; a genuine return-rescue complement requires SHORT exposure (Q5).")
    lines.insert(0, f"HEADLINE: {headline}")
    lines.insert(1, "")

    lines.append("")
    lines.append("CAVEATS (binding): (1) SYNTHETIC test surface from PHASE 3's VALIDATED generator, calibrated to "
                 "2020 stylized facts ONLY -- a stress surface, not real future data. (2) SHORT_MA / LONGSHORT_MA "
                 "VIOLATE the standing long-only+spot constraint -> RESEARCH only; deploying needs the user's "
                 "explicit LO-exception sign-off. (3) maker cost, no MtM double-count, lag-1 causal. (4) >=20 seeds; "
                 "distributions (mean +- spread + WORST seed) reported, no seed cherry-picked. (5) The synthetic "
                 "bear is the 2020-COVID-crash exemplar -- a fast V-crash; a slow grind-down bear may differ. "
                 "(6) Inverse-trend SHORT assumes the same maker fill economics as the long book; a real short has "
                 "borrow/funding costs NOT modeled here -- the Q5 short advantage is an UPPER bound.")
    return {"headline": headline, "scoreboard": scoreboard,
            "long_only_true_complements": lo_true, "short_true_complements": short_true,
            "long_only_relaxation_value": relax, "lines": lines}


# =====================================================================================================
# 6. CHARTS
# =====================================================================================================
COLORS = {"SHORT_MA": "#d62728", "LONGSHORT_MA": "#9467bd", "CASH_GATE": "#2ca02c",
          "VOLTGT_DEF": "#1f77b4", "MR_LONG": "#ff7f0e"}


def chart_return_corr_by_regime(results, cadences):
    """Chart 1: each candidate's RETURN-correlation to trend, per regime (the KEY metric)."""
    cs = [c for c in cadences if c in results]
    if not cs:
        return
    fig, axes = plt.subplots(1, len(cs), figsize=(6.5 * len(cs), 5.2), squeeze=False)
    x = np.arange(len(SCENARIOS)); width = 0.8 / len(CANDIDATES)
    for ci, cad in enumerate(cs):
        ax = axes[0][ci]
        for ni, name in enumerate(CANDIDATES):
            vals = [(results[cad][sc][name]["corr_to_trend"]["mean"]
                     if results[cad][sc][name]["corr_to_trend"]["mean"] is not None else 0) for sc in SCENARIOS]
            errs = [(results[cad][sc][name]["corr_to_trend"]["std"] or 0) for sc in SCENARIOS]
            lbl = name + (" [short/RESEARCH]" if not LONG_ONLY[name] else "")
            ax.bar(x + (ni - len(CANDIDATES) / 2 + 0.5) * width, vals, width, yerr=errs, capsize=2,
                   color=COLORS[name], label=lbl, alpha=0.9)
        ax.axhline(0, color="k", lw=0.8)
        ax.axhspan(-1.0, 0.0, color="#2ca02c", alpha=0.06)
        ax.text(0.02, 0.02, "below 0 = RETURN-anticorrelated = WINS when trend loses\n(the TRUE-complement zone)",
                transform=ax.transAxes, fontsize=8, va="bottom",
                bbox=dict(boxstyle="round", fc="#eaffea", ec="#999"))
        ax.set_xticks(x); ax.set_xticklabels([s.upper() for s in SCENARIOS])
        ax.set_ylabel("corr(candidate, trend) on RETURNS"); ax.set_ylim(-1.0, 1.0)
        ax.set_title(f"{cad}: RETURN-correlation to trend per regime", fontsize=10)
        if ci == 0:
            ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("THE KEY METRIC -- candidate RETURN-correlation to the trend book, per regime (20 seeds, mean +- sd)\n"
                 "PHASE 3 lesson: engagement-anticorrelation is NOT enough; a TRUE complement is RETURN-anticorrelated "
                 "(wins when trend loses). Synthetic, 2020-calibrated.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p = CHARTS / "complement_return_corr_by_regime.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    print(f"[figure] {p}")


def chart_combined_by_regime(results, cadences):
    """Chart 2: trend+each-candidate combined NET + maxDD across regimes (20-seed mean +- spread)."""
    cs = [c for c in cadences if c in results]
    if not cs:
        return
    nrow = len(cs)
    fig, axes = plt.subplots(nrow, 2, figsize=(15, 4.6 * nrow), squeeze=False)
    x = np.arange(len(SCENARIOS)); width = 0.8 / (len(CANDIDATES) + 1)
    for ri, cad in enumerate(cs):
        axn, axd = axes[ri][0], axes[ri][1]
        # trend-alone reference (first bar group)
        t_net = [results[cad][sc]["trend"]["net"]["mean"] or 0 for sc in SCENARIOS]
        t_nets = [results[cad][sc]["trend"]["net"]["std"] or 0 for sc in SCENARIOS]
        t_dd = [results[cad][sc]["trend"]["maxdd"]["mean"] or 0 for sc in SCENARIOS]
        axn.bar(x + (-0.5) * width * 1.0 - (len(CANDIDATES) / 2) * width, t_net, width, yerr=t_nets, capsize=2,
                color="#000000", label="TREND-ALONE", alpha=0.85)
        axd.bar(x - (len(CANDIDATES) / 2) * width, t_dd, width, color="#000000", label="TREND-ALONE", alpha=0.85)
        for ni, name in enumerate(CANDIDATES):
            cn = [results[cad][sc][name]["combined_net"]["mean"] or 0 for sc in SCENARIOS]
            cns = [results[cad][sc][name]["combined_net"]["std"] or 0 for sc in SCENARIOS]
            cd = [results[cad][sc][name]["combined_maxdd"]["mean"] or 0 for sc in SCENARIOS]
            lbl = "trend+" + name + (" [R]" if not LONG_ONLY[name] else "")
            off = (ni - len(CANDIDATES) / 2 + 1) * width
            axn.bar(x + off, cn, width, yerr=cns, capsize=2, color=COLORS[name], label=lbl, alpha=0.9)
            axd.bar(x + off, cd, width, color=COLORS[name], label=lbl, alpha=0.9)
        axn.set_xticks(x); axn.set_xticklabels([s.upper() for s in SCENARIOS]); axn.axhline(0, color="k", lw=0.6)
        axn.set_ylabel("combined net % (mean +- sd)"); axn.set_title(f"{cad}: trend+candidate NET by regime", fontsize=10)
        axd.set_xticks(x); axd.set_xticklabels([s.upper() for s in SCENARIOS]); axd.axhline(0, color="k", lw=0.6)
        axd.set_ylabel("combined maxDD % (less-negative=better)")
        axd.set_title(f"{cad}: trend+candidate maxDD by regime", fontsize=10)
        if ri == 0:
            axn.legend(fontsize=6, ncol=2)
    fig.suptitle("THE COMBINED trend+candidate BOOK across regimes (synthetic, 20 seeds) -- does the candidate LOWER "
                 "combined DD vs trend-alone (black) across the FULL mix, esp. the BEAR? ([R]=short, RESEARCH)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = CHARTS / "complement_combined_by_regime.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    print(f"[figure] {p}")


def chart_longonly_vs_short_value(results, verdict, cadences):
    """Chart 3: the QUANTIFIED value of relaxing long-only -- trend+SHORT/long-short vs trend+MR(long-only),
    net + maxDD advantage across the mix + in the bear specifically."""
    relax = verdict["long_only_relaxation_value"]
    if not relax:
        return
    refs = [r for r in ("SHORT_MA", "LONGSHORT_MA") if r in relax]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6))
    # left: net + DD advantage (mix + bear)
    metrics = [("mix_net_advantage_vs_trendMR", "mix NET adv"),
               ("mix_dd_advantage_vs_trendMR_pp", "mix DD adv"),
               ("bear_net_advantage_vs_trendMR", "BEAR NET adv"),
               ("bear_dd_advantage_vs_trendMR_pp", "BEAR DD adv")]
    x = np.arange(len(metrics)); width = 0.8 / max(1, len(refs))
    for ri, ref in enumerate(refs):
        vals = [relax[ref][m[0]] for m in metrics]
        ax1.bar(x + (ri - len(refs) / 2 + 0.5) * width, vals, width,
                label=f"trend+{ref} vs trend+MR(LO)", color=COLORS.get(ref, "#888"), alpha=0.9)
        for xi, v in zip(x + (ri - len(refs) / 2 + 0.5) * width, vals):
            ax1.annotate(f"{v:+.1f}", (xi, v), ha="center",
                         va="bottom" if v >= 0 else "top", fontsize=8)
    ax1.axhline(0, color="k", lw=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels([m[1] for m in metrics], rotation=15)
    ax1.set_ylabel("advantage of relaxing long-only (pp; positive = SHORT book better)")
    ax1.set_title("VALUE OF RELAXING LONG-ONLY\ntrend+SHORT/long-short vs trend+MR(long-only)", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.text(0.02, 0.97, "positive DD adv = combined book draws down LESS\nwhen the complement can SHORT (research only)",
             transform=ax1.transAxes, fontsize=8, va="top",
             bbox=dict(boxstyle="round", fc="#fff7e6", ec="#999"))
    # right: the bear scoreboard -- bear standalone net of each candidate (the gap-fill, positive = fills)
    sb = verdict["scoreboard"]
    names = CANDIDATES
    bear_nets = [sb[n]["bear_standalone_net"] if sb[n]["bear_standalone_net"] is not None else 0 for n in names]
    cols = [COLORS[n] for n in names]
    bars = ax2.bar(np.arange(len(names)), bear_nets, color=cols, alpha=0.9)
    for xi, (n, v) in enumerate(zip(names, bear_nets)):
        tag = "" if LONG_ONLY[n] else " [R]"
        ax2.annotate(f"{v:+.1f}{tag}", (xi, v), ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    # trend-alone bear net reference line
    t_bear = None
    for cad in results:
        t_bear = results[cad]["bear"]["trend"]["net"]["mean"]; break
    if t_bear is not None:
        ax2.axhline(t_bear, color="k", ls="--", lw=1.0, label=f"trend-alone bear net ({t_bear:+.1f}%)")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xticks(np.arange(len(names))); ax2.set_xticklabels(names, rotation=20, fontsize=8)
    ax2.set_ylabel("candidate STANDALONE bear net % (positive = genuine gap-fill)")
    ax2.set_title("THE BEAR GAP-FILL: candidate standalone net in the BEAR\n(positive = WINS where trend bleeds; "
                  "[R]=short/RESEARCH)", fontsize=10)
    ax2.legend(fontsize=8)
    fig.suptitle("QUANTIFIED: is the LONG-ONLY constraint what blocks TRUE complementarity? (synthetic, 20 seeds)\n"
                 "SHORT = RESEARCH (deploy needs the LO-exception sign-off); short economics here EXCLUDE borrow/funding "
                 "(upper bound)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = CHARTS / "longonly_vs_short_value.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    print(f"[figure] {p}")


# =====================================================================================================
# 7. SELFTEST -- sleeve-direction soundness on planted synthetic regimes (NO real-data calibration).
# =====================================================================================================
def selftest():
    """Two-sided sleeve-direction soundness (synthetic, no real calib):
    POSITIVE: in a planted BEAR, the SHORT sleeve must be RETURN-anticorrelated to (and beat) the long trend
              sleeve, and post a POSITIVE standalone return (it shorts a falling market). CASH_GATE must
              LOSE LESS than trend in the bear. In a planted BULL, SHORT must LOSE (anti-correlated the other
              way). NEGATIVE: in a NULL (no-trend) regime, SHORT must NOT manufacture a positive return."""
    print("## COMPLEMENTARY-SLEEVE-SEARCH SELFTEST (two-sided sleeve-direction soundness; no real-data calib)")
    ok = True
    calib = {
        "bull": {"mean": 0.006, "std": 0.045, "kurt": 3.0, "skew": 0.3, "ar1": -0.02, "vol_cluster": 0.20,
                 "t_df": SRS._student_t_df_from_kurt(3.0)},
        "bear": {"mean": -0.012, "std": 0.090, "kurt": 8.0, "skew": -1.5, "ar1": -0.30, "vol_cluster": 0.25,
                 "t_df": SRS._student_t_df_from_kurt(8.0)},
        "chop": {"mean": 0.001, "std": 0.030, "kurt": 2.0, "skew": 0.0, "ar1": -0.10, "vol_cluster": 0.05,
                 "t_df": SRS._student_t_df_from_kurt(2.0)},
        "null": {"mean": 0.0, "std": 0.02, "kurt": 0.0, "skew": 0.0, "ar1": 0.0, "vol_cluster": 0.0, "t_df": 30.0},
        "_xasset": {"mean_pairwise_corr": 0.49, "mean_btc_beta": 0.5, "n_assets": 10},
    }

    def _books(rg, seed=1, nb=92):
        panels = SRS.generate_regime_panels(rg, calib, seed=seed, n_bars=nb)
        return run_on_panel(panels, "1d")

    # POSITIVE: bear -- SHORT wins where trend loses
    rb = _books("bear", seed=2)
    t_bear = rb["TREND"]["perf"]["net"]
    s_bear = rb["SHORT_MA"]["perf"]["net"]
    s_corr = rb["SHORT_MA"]["corr_to_trend"]
    cash_bear = rb["CASH_GATE"]["perf"]["net"]
    short_wins_bear = s_bear is not None and t_bear is not None and s_bear > t_bear
    short_positive_bear = s_bear is not None and s_bear > 0
    short_anticorr = s_corr is not None and s_corr < 0.3
    cash_loses_less = cash_bear is not None and t_bear is not None and cash_bear >= t_bear
    print(f"  POSITIVE bear: trend net {t_bear}%, SHORT net {s_bear}% (>trend: {short_wins_bear}; >0: {short_positive_bear}); "
          f"corr(short,trend)={s_corr} (<0.3: {short_anticorr}); CASH_GATE {cash_bear}% (>=trend: {cash_loses_less})")
    ok &= short_wins_bear and short_positive_bear and short_anticorr and cash_loses_less

    # POSITIVE: bull -- SHORT loses where trend wins (the symmetry; short is directional)
    ru = _books("bull", seed=3)
    t_bull = ru["TREND"]["perf"]["net"]; s_bull = ru["SHORT_MA"]["perf"]["net"]
    short_loses_bull = s_bull is not None and t_bull is not None and s_bull < t_bull
    print(f"  POSITIVE bull: trend net {t_bull}%, SHORT net {s_bull}% (<trend: {short_loses_bull})")
    ok &= short_loses_bull

    # NEGATIVE: null -- SHORT must NOT manufacture a positive return (no trend either way), across seeds
    null_shorts = []
    for s in range(8):
        rn = _books("null", seed=200 + s)
        if rn is not None and rn["SHORT_MA"] is not None and rn["SHORT_MA"]["perf"]["net"] is not None:
            null_shorts.append(rn["SHORT_MA"]["perf"]["net"])
    null_short_mean = float(np.mean(null_shorts)) if null_shorts else 0.0
    null_no_edge = abs(null_short_mean) < 4.0     # near zero -- after maker cost a no-trend short bleeds slightly
    print(f"  NEGATIVE null: SHORT mean net over {len(null_shorts)} seeds = {null_short_mean:+.2f}% (|.|<4: {null_no_edge})")
    ok &= null_no_edge

    # LONGSHORT in bear should also be anti-correlated-ish + lose less than long-only trend
    ls_bear = rb["LONGSHORT_MA"]["perf"]["net"]
    ls_better = ls_bear is not None and t_bear is not None and ls_bear > t_bear
    print(f"  POSITIVE bear: LONGSHORT net {ls_bear}% (>trend {t_bear}%: {ls_better})")
    ok &= ls_better

    # MR_LONG in bear should be the BEAR-LIABILITY (loses MORE than trend) -- the established PHASE 3 baseline
    mr_bear = rb["MR_LONG"]["perf"]["net"]
    mr_liability = mr_bear is not None and t_bear is not None and mr_bear < t_bear
    print(f"  BASELINE bear: MR_LONG net {mr_bear}% (<trend {t_bear}% = bear-liability: {mr_liability})")
    ok &= mr_liability

    print(f"\n  SELFTEST {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# =====================================================================================================
# 8. MAIN
# =====================================================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m strat.complementary_sleeve_search")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--cadences", default="1d")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    CHARTS.mkdir(parents=True, exist_ok=True)
    print("## COMPLEMENTARY-SLEEVE-SEARCH -- PHASE 4b (true-complement search on PHASE 3's validated synthetic surface)")

    # 1. CALIBRATE the PHASE 3 generator on 2020 ONLY (its calibrate_2020 is the only real-data touch; fenced)
    print("\n## CALIBRATING (PHASE 3 generator, REAL 2020-band data ONLY) ...")
    SRS._CALIB, real_samples = SRS.calibrate_2020()
    validation, _ = SRS.validate_generator(SRS._CALIB, real_samples, seed=0, n_paths=max(30, a.seeds))
    print(f"   GENERATOR VALIDATION: {validation['_summary']['verdict']} "
          f"({validation['_summary']['regimes_all_match']}/{validation['_summary']['regimes_validated']} regimes match)")
    if validation["_summary"]["regimes_all_match"] < 2:
        print("   >> CAVEAT: generator only PARTIALLY validated -- results are SUGGESTIVE, not load-bearing.")

    # 2. THE SEARCH
    seeds = list(range(1, a.seeds + 1))
    cadences = [c.strip() for c in a.cadences.split(",") if c.strip()]
    print(f"\n## SEARCHING over {len(seeds)} seeds x {len(cadences)} cadences x 4 scenarios "
          f"x {len(CANDIDATES)} candidates ...")
    results = run_search(cadences, seeds)

    # 3. VERDICT
    verdict = build_verdict(results)
    print("\n" + "=" * 100)
    print("## DECISIVE VERDICT (two-sided)")
    for line in verdict["lines"]:
        print(f"   {line}")
    print("=" * 100)

    # 4. CHARTS
    chart_return_corr_by_regime(results, cadences)
    chart_combined_by_regime(results, cadences)
    chart_longonly_vs_short_value(results, verdict, cadences)

    # 5. PERSIST
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    export = {
        "repro": {"command": "python -m strat.complementary_sleeve_search " + " ".join(argv or sys.argv[1:]),
                  "git_sha": sha, "cost_maker": MAKER_RT, "cost_taker": TAKER_RT,
                  "calib_window": SRS.CALIB_WINDOW, "regime_periods": SRS.REGIME_PERIODS,
                  "n_seeds": a.seeds, "cadences": cadences, "candidates": CANDIDATES, "long_only": LONG_ONLY,
                  "generator_git_sha_phase3": "1756867 (synthetic_regime_stress)",
                  "constraint": "SYNTHETIC (PHASE 3 validated generator) + 2020 calibration ONLY; never touch "
                                "2026/other; SHORT/long-short = RESEARCH (deploy needs LO-exception sign-off)"},
        "generator_validation": validation,
        "results": _strip_arrays(results),
        "verdict": verdict,
    }
    p = OUT / "complementary_sleeve_search.json"
    json.dump(export, open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n[persisted] {p}")
    return 0


def _strip_arrays(results):
    out = {}
    for cad, summ in results.items():
        out[cad] = {k: v for k, v in summ.items() if not k.startswith("_equity")}
    return out


if __name__ == "__main__":
    raise SystemExit(main())
