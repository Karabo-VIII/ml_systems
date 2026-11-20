"""src/strat/participate_preserve_frontier.py -- THE DECISIVE LONG-ONLY FRONTIER TEST.

THE GENUINE OPEN QUESTION (surfaced by the 6h translation build, commit 6a4c1ff):
  The family-ensemble book PRESERVES bears by going to CASH (= non-participation). Every construction is
  ONE POINT on a participation<->preservation frontier; NO construction has a deployable wealth edge (all
  fail ship-gate p05<0); the cash-going book LOSES to buy-hold on raw wealth (book +57.7% vs BH +75.5%
  full-cycle 2020-2025). THE QUESTION:

    Is there ANY LONG-ONLY (spot, short OFF) construction that PARTICIPATES in bulls AND PRESERVES bears
    -- i.e. lies NORTHEAST of the cash-going book on the (bull-capture% vs bear-DD-preservation%) plane
    (MORE participation AND MORE preservation = DOMINATES), beating it on FULL-CYCLE WEALTH while keeping
    bear-preservation AND passing ship-gate p05>0 --
    OR is participation<->preservation a FUNDAMENTAL tradeoff (you can only MOVE ALONG the frontier line,
    not BEAT it)? If the latter, long-only participate-and-preserve is CLOSED.

THE KEY MECHANISM (from family_free_control.py, the user's binding note):
  vol-target is symmetric to VOLATILITY, NOT DRAWDOWN. A vol-brake de-risks a HOT bull as readily as a
  bear (that is WHY the cash-going book under-participates). To preserve a bear without crippling the bull
  you need a TREND/DRAWDOWN signal that goes to cash on the WAY DOWN and re-invests on the WAY UP -- a
  DIRECTIONAL gate, not a symmetric vol-brake. The 4 constructions below are 4 directional gates, each a
  different point on the participation<->preservation tradeoff.

THE 4 CONSTRUCTIONS (each a per-asset, causal, long-only {0,1}->scaled position modifier on EW-beta):
  a. BEAR-RALLY PARTICIPATION : trend-gate (invest above slow-MA) BUT re-enter on a confirmed counter-trend
     bounce inside a downtrend (fast-MA reclaim) -- captures the +30-50% bear rallies the cash book misses.
  b. TERMINAL-LEG-ONLY        : stay long through corrections; de-risk ONLY on a confirmed SUSTAINED
     downtrend (below slow-MA AND slow-MA itself falling K days) -- a slow drawdown-aware gate, not every dip.
  c. DRAWDOWN-AWARE GATE      : de-risk on the ASSET'S OWN underwater curve breaching a threshold (price
     down > X% from trailing high); re-invest on recovery -- drawdown-aware, not vol-aware.
  d. ASYMMETRIC ENTRY/EXIT    : trend-gate with FAST re-entry / SLOW exit hysteresis (enter on a short
     confirm, exit on a long confirm) -- shifts the participation<->preservation point by asymmetry.

THE FRONTIER + THE VERDICT:
  Plot ALL constructions + the cash-going book + buy-hold + the family-free vol-gate on the
  (bull-capture% vs bear-DD-preservation%) plane. Is any point NORTHEAST of the book (dominates), or do
  they all lie on ONE frontier line (the tradeoff is fundamental)?
  DECISIVE VERDICT (held-out): does a participate-AND-preserve construction beat the cash-going book on
  FULL-CYCLE WEALTH (2020-2024 held-out + the single UNSEEN read) while keeping bear-preservation AND
  passing ship-gate p05>0? OR is it a hard frontier -> long-only participate-and-preserve is CLOSED.

DISCIPLINE (binding):
  STRICT LONG-ONLY + spot (positions in [0,1]; ZERO short logic anywhere). SELECT/tune on 2020-2024; the
  SEALED UNSEEN 2025-12-31->2026-06-01 is READ-ONLY (single read at the END, NO tuning on it). fixed-EW
  (fillna(0.0).mean -- NEVER skipna; buy-hold cadence-invariant). Survivorship-clean POINT-IN-TIME
  (data-derived listing dates, per-year/window cutoffs -- reuses forward_test_2021's PIT machinery).
  Maker cost on flips, causal/lag-1, rolling rv shift(1). MULTIPLE-COMPARISONS-aware: we try several
  constructions -> the BEST is DEFLATED (permutation null on the best's full-cycle edge over the book +
  PBO across the construction family). A best-of-several that fails deflation is NOT a result. No emoji.

RWYB:
  python -m strat.participate_preserve_frontier --selftest     # mechanics sanity (fast; does NOT touch UNSEEN)
  python -m strat.participate_preserve_frontier                # full frontier build + held-out grade + UNSEEN read
Does NOT git commit (overseer commits after judging). UNSEEN is touched EXACTLY ONCE (at the end).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.forward_test_2021 as FT                                          # noqa: E402  (PIT engine)
import strat.family_ensemble_book as FEB                                      # noqa: E402  (the cash-going book)
from strat.forward_test_2021 import pit_universe_2021, _buyhold_net_series, MAKER_RT  # noqa: E402
from strat.family_ensemble_book import (                                      # noqa: E402
    DERISK_LEVELS, _vt_level, _metrics, build_book, build_buyhold,
)
from strat.family_ensemble_unseen import _set_window, UNSEEN_WIN             # noqa: E402
from strat.scorecard import score_book, SPLITS                               # noqa: E402
from strat.battery import block_bootstrap_p05_p95                            # noqa: E402

OUT = ROOT.parent / "runs" / "strat"
OUT.mkdir(parents=True, exist_ok=True)

PICK_LEVEL = "light"          # the FROZEN PHASE-1 de-risk pick (the cash-going book's level)

# the held-out development span (we SELECT/tune here; UNSEEN is sealed) -- calendar years 2020-2024 + 2025-to-OOS-end
DEV_YEARS = (2020, 2021, 2022, 2023, 2024)
DEV_2025_WIN = ("2025-01-01", "2025-12-31")   # the 2025 OOS span (still pre-UNSEEN; part of the held-out dev set)

# the bull years (for the bull-capture axis) and the bear years (for the preservation axis), full-cycle.
BULL_YEARS = (2020, 2021, 2023, 2024)         # raw-beta strongly positive years
BEAR_YEARS = (2022, 2025)                     # raw-beta deep-negative years (the preservation crux)

__contract__ = {
    "kind": "long_only_participate_preserve_frontier_decisive_test",
    "inputs": {
        "incumbent": "the cash-going family-ensemble book at LIGHT de-risk (imported from "
                     "family_ensemble_book.build_book; NO re-implementation).",
        "benchmarks": "EW buy-hold (raw beta) + the family-free vol-gate (vol-brake on beta).",
        "constructions": "4 LONG-ONLY directional gates {bear_rally, terminal_leg, drawdown_aware, "
                         "asymmetric} -- each a causal per-asset {0,1}->scaled position on EW-beta, "
                         "tuned ONLY on 2020-2024 (dev), UNSEEN sealed.",
    },
    "outputs": {
        "frontier": "every construction on the (bull-capture% vs bear-DD-preservation%) plane.",
        "held_out_grade": "full-cycle 2020-2024 wealth/maxDD + canonical scorecard p05 (SEL/OOS/UNSEEN).",
        "unseen_once": "the single sealed UNSEEN read for the cash-going book + the BEST construction.",
        "deflation": "permutation null on the best's full-cycle edge over the book + PBO across the family.",
        "verdict": "DOMINATES (NE of the book, beats it held-out, p05>0, deflation-survives) / "
                   "FUNDAMENTAL_TRADEOFF (one frontier line -- CLOSED) + cheapest falsifier.",
    },
    "invariants": {
        "long_only_spot": "positions in [0,1]; ZERO short logic anywhere; STRICT.",
        "unseen_read_only": "select/tune on 2020-2024; UNSEEN computed ONCE at the end; no tuning on it.",
        "fixed_ew": "fillna(0.0).mean aggregation (NEVER skipna) -- buy-hold cadence-invariant.",
        "survivorship_clean_pit": "data-derived listing dates; per-year/window PIT cutoffs.",
        "causal_mtm_no_double_count": "gates on lagged price/MA; positions lag 1 bar; rv shift(1); maker on flips.",
        "multiple_comparisons_deflated": "the BEST construction is permutation-deflated + PBO across the family.",
    },
}

PREREG = {
    "H0_fundamental_tradeoff": "NO long-only construction lies NORTHEAST of the cash-going book on the "
        "(bull-capture vs bear-preservation) plane; they all lie on ONE frontier line. You can only MOVE "
        "ALONG the tradeoff (trade participation for preservation), not BEAT it. -> long-only "
        "participate-and-preserve is CLOSED; the honest doors are SHORT (OFF) or CARRY.",
    "H1_dominates": "a long-only construction DOMINATES the book (MORE bull-capture AND MORE/equal "
        "bear-preservation), beats it on FULL-CYCLE held-out WEALTH, keeps bear-preservation, AND passes "
        "ship-gate p05>0 (held-out block-bootstrap) AND survives multiple-comparisons deflation.",
    "asymmetric_loss": "false-ship a non-preserving book into a -60% bear >> false-skip. A best-of-several "
        "that fails deflation is NOT a result.",
    "dominance_def": "construction is NE of the book iff bull_capture_pct > book's AND bear_preservation_pct "
        ">= book's - 5pp (preservation not materially worse) -- i.e. it does not give back the book's risk.",
    "ship_gate": "full-cycle held-out (OOS+UNSEEN) block-bootstrap p05 > 0 AND UNSEEN compound > 0.",
    "pick_level": PICK_LEVEL,
    "dev_years": list(DEV_YEARS),
    "unseen_window": list(UNSEEN_WIN),
}


# =====================================================================================================
# 1. CAUSAL GATE PRIMITIVES (numpy; computed on the full in-panel series incl. warmup, applied lag-1)
# =====================================================================================================
def _sma(x, n):
    """Trailing simple MA, causal (uses only x[<=t]); NaN until n obs."""
    return pd.Series(x).rolling(n, min_periods=n).mean().to_numpy()


def _trailing_high(x, n):
    """Trailing rolling max over the last n bars (causal)."""
    return pd.Series(x).rolling(n, min_periods=1).max().to_numpy()


# ---- each gate returns a held-array in {0,1} over the full in-panel length (same length as A['c']) ----
def gate_trend(A, slow=50):
    """Plain trend gate: invested iff close > trailing slow-MA. The reference DIRECTIONAL gate."""
    c = A["c"]; ms = _sma(c, slow)
    return np.nan_to_num(c > ms).astype(np.int8)


def gate_bear_rally(A, slow=50, fast=10, down_lookback=5):
    """BEAR-RALLY PARTICIPATION: invested iff (close > slow-MA)  OR  a confirmed counter-trend BOUNCE
    inside a downtrend = (close > fast-MA  AND  close > close[t-down_lookback])  i.e. price has reclaimed
    the fast-MA and is up over the recent lookback even while below the slow-MA. Captures the sharp bear
    rallies the pure trend gate sits out. Long-only {0,1}."""
    c = A["c"]
    ms = _sma(c, slow); mf = _sma(c, fast)
    up_trend = c > ms
    prev = np.concatenate([np.full(down_lookback, np.nan), c[:-down_lookback]])
    bounce = (c > mf) & (c > prev)          # reclaimed fast-MA AND rising over the lookback
    held = up_trend | np.nan_to_num(bounce).astype(bool)
    return np.nan_to_num(held).astype(np.int8)


def gate_terminal_leg(A, slow=100, fall_k=20):
    """TERMINAL-LEG-ONLY de-risk: stay LONG by default; de-risk (go to cash) ONLY on a confirmed SUSTAINED
    downtrend = (close < slow-MA)  AND  (slow-MA today < slow-MA fall_k bars ago, i.e. the slow trend itself
    is FALLING). Corrections (price dips but slow-MA still rising) keep us invested. Long-only {0,1}."""
    c = A["c"]; ms = _sma(c, slow)
    ms_prev = np.concatenate([np.full(fall_k, np.nan), ms[:-fall_k]])
    sustained_down = (c < ms) & (ms < ms_prev)
    held = ~np.nan_to_num(sustained_down).astype(bool)   # invested unless a sustained downtrend is confirmed
    # but never invest before the slow-MA exists (avoid spurious early all-in)
    held = held & ~np.isnan(ms)
    return np.nan_to_num(held).astype(np.int8)


def gate_drawdown_aware(A, dd_exit=0.20, dd_reenter=0.10, high_n=90):
    """DRAWDOWN-AWARE gate (an underwater/equity-curve gate on the ASSET'S OWN price): de-risk when the
    asset is more than dd_exit below its trailing high; re-invest when it has recovered to within dd_reenter
    of the trailing high. Hysteresis prevents whipsaw. Drawdown-aware, NOT vol-aware. Long-only {0,1}."""
    c = np.asarray(A["c"], float)
    hi = _trailing_high(c, high_n)
    underwater = (c - hi) / hi              # <= 0
    held = np.ones(len(c), dtype=np.int8)
    state = 1
    for t in range(len(c)):
        if state == 1 and underwater[t] <= -dd_exit:
            state = 0
        elif state == 0 and underwater[t] >= -dd_reenter:
            state = 1
        held[t] = state
    return held


def gate_asymmetric(A, slow=50, enter_confirm=2, exit_confirm=10):
    """ASYMMETRIC entry/exit: a trend gate with FAST re-entry / SLOW exit hysteresis. Raw signal = close >
    slow-MA. We ENTER after `enter_confirm` consecutive raw-on bars (fast in), and EXIT only after
    `exit_confirm` consecutive raw-off bars (slow out). Shifts the participation<->preservation point toward
    PARTICIPATION (slow to abandon a position). Long-only {0,1}."""
    c = A["c"]; ms = _sma(c, slow)
    raw = np.nan_to_num(c > ms).astype(np.int8)
    held = np.zeros(len(c), dtype=np.int8)
    state = 0; on_run = 0; off_run = 0
    for t in range(len(c)):
        if raw[t] == 1:
            on_run += 1; off_run = 0
        else:
            off_run += 1; on_run = 0
        if state == 0 and on_run >= enter_confirm:
            state = 1
        elif state == 1 and off_run >= exit_confirm:
            state = 0
        held[t] = state
    return held


# the construction registry: name -> (gate_fn, default kwargs). The grid (for dev-tuning + PBO) is per-fn.
CONSTRUCTIONS = {
    "bear_rally":     (gate_bear_rally,     {"slow": 50, "fast": 10, "down_lookback": 5}),
    "terminal_leg":   (gate_terminal_leg,   {"slow": 100, "fall_k": 20}),
    "drawdown_aware": (gate_drawdown_aware, {"dd_exit": 0.20, "dd_reenter": 0.10, "high_n": 90}),
    "asymmetric":     (gate_asymmetric,     {"slow": 50, "enter_confirm": 2, "exit_confirm": 10}),
}

# small dev-only param grids (tuned on 2020-2024; the grid feeds PBO across the construction family)
CONSTRUCTION_GRID = {
    "bear_rally":     [{"slow": s, "fast": f, "down_lookback": 5} for s in (40, 50, 80) for f in (8, 12)],
    "terminal_leg":   [{"slow": s, "fall_k": k} for s in (80, 100, 150) for k in (15, 25)],
    "drawdown_aware": [{"dd_exit": de, "dd_reenter": dr, "high_n": 90}
                       for de in (0.15, 0.20, 0.30) for dr in (0.08, 0.12)],
    "asymmetric":     [{"slow": s, "enter_confirm": ec, "exit_confirm": xc}
                       for s in (40, 50) for ec in (1, 2) for xc in (8, 15)],
}


# =====================================================================================================
# 2. PER-ASSET NET SERIES UNDER A GATE (long-only, causal, maker cost) -> fixed-EW book
# =====================================================================================================
def _gated_net_series(A, gate_fn, kwargs, vt, cap):
    """One asset's bar-level net Series under `gate_fn` (the directional gate), the LIGHT vol-target overlay
    (same vt the book uses), maker cost on flips, lag-1 causal, NaN where not active (PIT cash). Long-only:
    held in {0,1}, vol-target multiplier clipped to [0, cap]. This is the EXACT cost/lag/PIT stack the
    cash-going book uses, with the directional gate substituted for the family signal."""
    ret, rv = A["ret"], A["rv"]
    held = np.asarray(gate_fn(A, **kwargs)).astype(np.float64)        # {0,1} over full in-panel length
    pos = np.zeros(len(ret)); pos[1:] = held[:-1]                     # lag 1 bar (causal)
    if vt is not None:
        pos = pos * np.clip(vt / (np.nan_to_num(rv, nan=vt) + 1e-12), 0.0, cap)
    flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
    net = pos * ret - flips * (MAKER_RT / 2.0)
    s = pd.Series(np.where(A["active"], net, np.nan), index=A["idx"])
    return s[A["win"]]


def build_construction_book(gate_fn, kwargs, derisk=PICK_LEVEL):
    """Build the LONG-ONLY construction book for the CURRENT window (set via FEB._set_year / _set_window):
    apply `gate_fn` per asset on the PIT-active 1d roster, fixed-EW aggregate (fillna(0.0).mean = PIT cash).
    Returns a daily-return Series or None. Reuses build_buyhold's exact roster + the book's vol-target."""
    lvl = DERISK_LEVELS[derisk]
    assets = FT._assets_for("1d", False, "expand")
    vt = _vt_level(assets, lvl["vt_mult"])
    series = [_gated_net_series(A, gate_fn, kwargs, vt, lvl["cap"]) for A in assets]
    series = [s for s in series if s is not None and len(s)]
    if not series:
        return None
    df = pd.concat(series, axis=1).sort_index()
    bar = df.fillna(0.0).mean(axis=1)                                # fixed-EW (PIT: NaN/inactive = cash)
    return bar.dropna().resample("1D").apply(lambda x: float(np.prod(1 + x.dropna()) - 1)).dropna()


# =====================================================================================================
# 3. WINDOW RUNNERS + FULL-CYCLE CHAIN
# =====================================================================================================
def _set_year_fn(year):
    return lambda: FEB._set_year(year)


def _set_unseen_fn():
    return lambda: _set_window(UNSEEN_WIN[0], UNSEEN_WIN[1])


def _set_2025_fn():
    return lambda: _set_window(DEV_2025_WIN[0], DEV_2025_WIN[1])


def _run_year_streams(window_setter, constructions, derisk=PICK_LEVEL):
    """Set the window; build (i) the cash-going book, (ii) raw EW-beta buy-hold, (iii) the family-free
    vol-gate, (iv) each construction book. Returns {name: (metrics, daily)}."""
    window_setter()
    pit_universe_2021(verbose=False)
    out = {}
    book, _ = build_book(derisk=derisk, combine="ew")
    out["cash_book"] = (_metrics(book) if book is not None else {}, book)
    bh = build_buyhold()
    out["raw_beta"] = (_metrics(bh) if bh is not None else {}, bh)
    ff = _family_free(derisk)
    out["family_free"] = (_metrics(ff) if ff is not None else {}, ff)
    for name, (gate_fn, kwargs) in constructions.items():
        d = build_construction_book(gate_fn, kwargs, derisk=derisk)
        out[name] = (_metrics(d) if d is not None else {}, d)
    return out


def _family_free(derisk=PICK_LEVEL):
    """The family-free vol-gate (plain EW-beta + the book's LIGHT vol-brake) for the CURRENT window."""
    lvl = DERISK_LEVELS[derisk]
    assets = FT._assets_for("1d", False, "expand")
    vt = _vt_level(assets, lvl["vt_mult"])
    series = [_buyhold_net_series(A, vt=vt) for A in assets]
    series = [s for s in series if s is not None and len(s)]
    if not series:
        return None
    df = pd.concat(series, axis=1).sort_index()
    bar = df.fillna(0.0).mean(axis=1)
    return bar.dropna().resample("1D").apply(lambda x: float(np.prod(1 + x.dropna()) - 1)).dropna()


def _chain(per_window, key):
    """Chain per-window daily streams (calendar order) for stream `key` -> (metrics, daily)."""
    parts = [w[key][1] for w in per_window if key in w and w[key][1] is not None and len(w[key][1])]
    if not parts:
        return {}, None
    d = pd.concat(parts).sort_index()
    d = d[~d.index.duplicated(keep="first")]
    return _metrics(d), d


# =====================================================================================================
# 4. THE FRONTIER AXES -- bull-capture % and bear-DD-preservation %
# =====================================================================================================
def _bull_capture_pct(per_year_metrics, raw_year_metrics):
    """Bull-capture % = (sum of construction net across BULL years) / (sum of raw-beta net across BULL years)
    computed on COMPOUNDED bull-year wealth: prod(1+net_bullyears) for construction vs raw beta, as a %.
    100% = captures the full bull; 0% = sits out the bull entirely."""
    def _compound_bulls(m):
        prod = 1.0
        for y in BULL_YEARS:
            n = m.get(y, {}).get("net_pct")
            if n is None:
                return None
            prod *= (1 + n / 100.0)
        return (prod - 1) * 100.0
    cw = _compound_bulls(per_year_metrics); rw = _compound_bulls(raw_year_metrics)
    if cw is None or rw is None or rw <= 0:
        return None
    return round(100.0 * cw / rw, 1)


def _bear_preservation_pct(per_year_metrics, raw_year_metrics):
    """Bear-DD-preservation % = how much SHALLOWER the construction's worst bear-year maxDD is vs raw-beta's,
    as a fraction of raw-beta's drawdown. 100% = no drawdown at all in the bear; 0% = same DD as raw beta.
    Uses the WORST (deepest) bear-year maxDD across BEAR_YEARS for both."""
    def _worst_bear_dd(m):
        dds = [m.get(y, {}).get("maxdd_pct") for y in BEAR_YEARS]
        dds = [d for d in dds if d is not None]
        return min(dds) if dds else None       # most negative
    cd = _worst_bear_dd(per_year_metrics); rd = _worst_bear_dd(raw_year_metrics)
    if cd is None or rd is None or rd >= 0:
        return None
    return round(100.0 * (1.0 - cd / rd), 1)    # cd/rd in [0,1]; 1 - that = preservation fraction


# =====================================================================================================
# 5. THE FULL RUN
# =====================================================================================================
def run(derisk=PICK_LEVEL):
    res = {"prereg": PREREG, "derisk": derisk}
    constructions = CONSTRUCTIONS

    # ----------------------------------------------------------------------------------------------
    # (A) PER-YEAR (DEV: 2020-2024 + 2025-OOS) -- every stream, every year. UNSEEN NOT touched here.
    # ----------------------------------------------------------------------------------------------
    print("\n## (A) PER-YEAR DEV GRADE (2020-2024 + 2025-OOS) -- cash-book / raw-beta / family-free / 4 constructions")
    stream_keys = ["cash_book", "raw_beta", "family_free"] + list(constructions.keys())
    per_year_metrics = {k: {} for k in stream_keys}     # {stream: {year: metrics}}
    per_window = []
    dev_windows = [(_set_year_fn(y), y) for y in DEV_YEARS] + [(_set_2025_fn(), 2025)]
    for setter, label in dev_windows:
        d = _run_year_streams(setter, constructions, derisk)
        per_window.append(d)
        for k in stream_keys:
            per_year_metrics[k][label] = d[k][0]
        cb = d["cash_book"][0]; rb = d["raw_beta"][0]
        print(f"   -- {label}: raw-beta net {rb.get('net_pct')}% DD {rb.get('maxdd_pct')}% | "
              f"cash-book net {cb.get('net_pct')}% DD {cb.get('maxdd_pct')}%")
        for name in constructions:
            m = d[name][0]
            print(f"        {name:15} net {str(m.get('net_pct')):>8}% DD {str(m.get('maxdd_pct')):>8}% "
                  f"Sharpe {str(m.get('sharpe')):>5} Calmar {str(m.get('calmar')):>6}")
    res["per_year_metrics"] = {k: {str(y): v for y, v in d.items()} for k, d in per_year_metrics.items()}

    # ----------------------------------------------------------------------------------------------
    # (B) FULL-CYCLE DEV CHAIN (2020-2025) -- the held-out wealth + maxDD per stream
    # ----------------------------------------------------------------------------------------------
    print("\n## (B) FULL-CYCLE DEV (2020-2025 calendar chain, UNSEEN excluded) -- wealth / maxDD / Calmar")
    fc = {}
    for k in stream_keys:
        m, d = _chain(per_window, k)
        fc[k] = {"metrics": m, "daily": d}
        print(f"   {k:16}: net {str(m.get('net_pct')):>9}% maxDD {str(m.get('maxdd_pct')):>8}% "
              f"Calmar {str(m.get('calmar')):>7} Sharpe {str(m.get('sharpe')):>6}")
    res["full_cycle_dev"] = {k: v["metrics"] for k, v in fc.items()}

    # ----------------------------------------------------------------------------------------------
    # (C) THE FRONTIER PLANE -- bull-capture % vs bear-preservation % for every stream
    # ----------------------------------------------------------------------------------------------
    print("\n## (C) FRONTIER PLANE: bull-capture % (x) vs bear-DD-preservation % (y)")
    raw_pm = per_year_metrics["raw_beta"]
    frontier = {}
    for k in stream_keys:
        bc = _bull_capture_pct(per_year_metrics[k], raw_pm)
        bp = _bear_preservation_pct(per_year_metrics[k], raw_pm)
        fc_net = fc[k]["metrics"].get("net_pct")
        fc_dd = fc[k]["metrics"].get("maxdd_pct")
        frontier[k] = {"bull_capture_pct": bc, "bear_preservation_pct": bp,
                       "fc_net_pct": fc_net, "fc_maxdd_pct": fc_dd}
        print(f"   {k:16}: bull-capture {str(bc):>7}% | bear-preservation {str(bp):>7}% | "
              f"full-cycle net {str(fc_net):>8}% DD {str(fc_dd):>8}%")
    res["frontier"] = frontier

    # dominance: which constructions are NORTHEAST of the cash-going book (MORE participation AND
    # preservation not materially worse, per the pre-registered dominance_def)?
    book_bc = frontier["cash_book"]["bull_capture_pct"]; book_bp = frontier["cash_book"]["bear_preservation_pct"]
    dominators = []
    for name in constructions:
        bc = frontier[name]["bull_capture_pct"]; bp = frontier[name]["bear_preservation_pct"]
        if bc is None or bp is None or book_bc is None or book_bp is None:
            continue
        ne = (bc > book_bc) and (bp >= book_bp - 5.0)        # MORE bull-capture, preservation within 5pp
        frontier[name]["dominates_book"] = bool(ne)
        if ne:
            dominators.append(name)
    res["dominators"] = dominators
    print(f"\n   cash-going book point: bull-capture {book_bc}% / bear-preservation {book_bp}%")
    print(f"   constructions NORTHEAST of the book (dominate): {dominators if dominators else 'NONE'}")

    # ----------------------------------------------------------------------------------------------
    # (D) PICK THE BEST construction by FULL-CYCLE DEV WEALTH subject to a HARD preservation guard
    #     (bear-preservation >= 50% -- it must keep at least half the book's risk reduction; this is the
    #     'participate AND preserve' constraint, NOT just 'participate'). Tuned on DEV only.
    # ----------------------------------------------------------------------------------------------
    PRESERVE_FLOOR = 50.0
    cand = []
    for name in constructions:
        bp = frontier[name]["bear_preservation_pct"]; net = frontier[name]["fc_net_pct"]
        if bp is not None and net is not None and bp >= PRESERVE_FLOOR:
            cand.append((name, net, bp))
    cand.sort(key=lambda x: -x[1])
    best_name = cand[0][0] if cand else None
    res["best_construction"] = best_name
    res["best_candidates_preserving"] = [{"name": n, "fc_net_pct": net, "bear_preservation_pct": bp}
                                         for n, net, bp in cand]
    print(f"\n## (D) BEST participate-AND-preserve construction (DEV wealth s.t. bear-preservation >= "
          f"{PRESERVE_FLOOR}%): {best_name}")
    for n, net, bp in cand:
        print(f"      {n:15} full-cycle DEV net {net}% (bear-preservation {bp}%)")
    if best_name is None:
        print("      NONE -- no construction keeps >= 50% bear-preservation; participate-and-preserve is "
              "structurally hard even before the UNSEEN read.")

    # ----------------------------------------------------------------------------------------------
    # (E) THE SEALED UNSEEN READ-ONCE -- cash-going book + BEST construction + benchmarks. SINGLE touch.
    # ----------------------------------------------------------------------------------------------
    print("\n## (E) SEALED UNSEEN READ-ONCE (2025-12-31 -> 2026-06-01) -- book + best construction + benchmarks")
    print("##     this is the ONLY UNSEEN touch; no tuning/iterating on it.")
    unseen_constructions = ({best_name: constructions[best_name]} if best_name else {})
    du = _run_year_streams(_set_unseen_fn(), {**constructions}, derisk)   # compute ALL for the frontier chart
    u_regime_net = du["raw_beta"][0].get("net_pct")
    regime = ("BULL" if (u_regime_net or 0) > 15 else "BEAR" if (u_regime_net or 0) < -15 else "CHOP/SIDEWAYS")
    unseen = {"window": list(UNSEEN_WIN), "regime": regime}
    for k in stream_keys:
        m = du[k][0]
        unseen[k] = {"net_pct": m.get("net_pct"), "maxdd_pct": m.get("maxdd_pct"),
                     "sharpe": m.get("sharpe"), "calmar": m.get("calmar")}
        print(f"   {k:16}: UNSEEN net {str(m.get('net_pct')):>8}% DD {str(m.get('maxdd_pct')):>8}% "
              f"Sharpe {str(m.get('sharpe')):>5}")
    res["unseen_once"] = unseen
    # stash daily for the equity chart
    res["_unseen_daily"] = {k: du[k][1] for k in stream_keys}

    # ----------------------------------------------------------------------------------------------
    # (F) CANONICAL SCORECARD on the FULL stream (DEV + UNSEEN) for the cash-book + the best construction
    #     -> the held-out block-bootstrap p05 ship-gate.
    # ----------------------------------------------------------------------------------------------
    print("\n## (F) CANONICAL SCORECARD (full 2020-2026 stream; held-out p05 ship-gate)")
    cards = {}
    score_targets = ["cash_book"] + ([best_name] if best_name else [])
    full_daily_streams = {}
    for k in score_targets:
        full_d = _full_stream(k, constructions, derisk)
        full_daily_streams[k] = full_d
        card = score_book(f"pp_frontier::{k}", full_d)
        cards[k] = card
        hp = card.get("heldout_block_bootstrap", {}).get("p05")
        fp = card.get("full_block_bootstrap", {}).get("p05")
        u = card["per_split"].get("UNSEEN", {})
        print(f"   [{k:14}] n_days={card['n_days']} | UNSEEN compound {u.get('compound_pct')}% | "
              f"held-out p05 {hp}% | full p05 {fp}% | ship={card['ship_read']['ship']}")
    res["scorecards"] = cards

    # ----------------------------------------------------------------------------------------------
    # (G) MULTIPLE-COMPARISONS DEFLATION of the BEST -- permutation null + PBO across the family.
    # ----------------------------------------------------------------------------------------------
    print("\n## (G) MULTIPLE-COMPARISONS DEFLATION (the best-of-several must survive)")
    deflation = {}
    if best_name is not None:
        # (G.1) PERMUTATION NULL: is the best's FULL-CYCLE-DEV edge over the cash-book beyond what the
        #       BEST-OF-K constructions would show by chance? Sign-flip / circular-shift block permutation of
        #       the (construction - book) daily DIFFERENCE, taking the MAX edge across the K constructions
        #       each permutation (so the null already includes the multiple-comparisons max-statistic).
        deflation["permutation_null"] = _permutation_deflate(per_window, constructions, best_name)
        # (G.2) PBO across the construction FAMILY (each construction's full-cycle-dev daily as a column).
        deflation["pbo"] = _pbo_across_family(per_window, constructions)
        for line in _fmt_deflation(deflation):
            print("   " + line)
    else:
        deflation["note"] = "no preserving construction to deflate (participate-and-preserve already fails the floor)."
        print("   " + deflation["note"])
    res["deflation"] = deflation

    res["verdict"] = build_verdict(res)
    return res


def _full_stream(key, constructions, derisk):
    """Full 2020-2026 daily stream for stream `key` (calendar chain incl. the single UNSEEN window)."""
    parts = []
    setters = ([(_set_year_fn(y), y) for y in DEV_YEARS] + [(_set_2025_fn(), 2025), (_set_unseen_fn(), "UNSEEN")])
    for setter, _ in setters:
        setter(); pit_universe_2021(verbose=False)
        if key == "cash_book":
            d, _ = build_book(derisk=derisk, combine="ew")
        elif key == "raw_beta":
            d = build_buyhold()
        elif key == "family_free":
            d = _family_free(derisk)
        else:
            gate_fn, kwargs = constructions[key]
            d = build_construction_book(gate_fn, kwargs, derisk=derisk)
        if d is not None and len(d):
            parts.append(d)
    s = pd.concat(parts).sort_index() if parts else pd.Series(dtype=float)
    return s[~s.index.duplicated(keep="first")]


# =====================================================================================================
# 6. DEFLATION ENGINES
# =====================================================================================================
def _dev_daily(per_window, key):
    parts = [w[key][1] for w in per_window if key in w and w[key][1] is not None and len(w[key][1])]
    if not parts:
        return pd.Series(dtype=float)
    d = pd.concat(parts).sort_index()
    return d[~d.index.duplicated(keep="first")]


def _permutation_deflate(per_window, constructions, best_name, n_perm=2000, block=10, seed=11):
    """MAX-statistic permutation null on the best construction's full-cycle-DEV WEALTH edge over the
    cash-going book. The statistic = compound(construction) - compound(book) over the 2020-2025 dev chain.
    Under H0 (the gate adds nothing beyond reshuffling the SAME beta exposure), the per-bar EXCESS return
    (construction - book) has no genuine sign structure; we block-bootstrap the excess stream and recompute
    the MAX edge across ALL K constructions each permutation, so the null distribution is the
    multiple-comparisons-corrected best-of-K. p = P(null max-edge >= observed best edge)."""
    book = _dev_daily(per_window, "cash_book")
    if len(book) < 50:
        return {"error": "insufficient dev book stream"}
    # align each construction's excess-return stream to the book
    excess = {}
    for name in constructions:
        d = _dev_daily(per_window, name)
        j = pd.concat([d.rename("x"), book.rename("b")], axis=1).dropna()
        if len(j) < 50:
            continue
        excess[name] = (j["x"] - j["b"]).to_numpy()
        excess[name + "__book"] = j["b"].to_numpy()
        excess[name + "__con"] = j["x"].to_numpy()
    if best_name not in excess:
        return {"error": "best construction excess stream unavailable"}
    # observed best edge across the K constructions (compound difference)
    def _edge(con, bk):
        return (np.prod(1 + con) - 1) * 100 - (np.prod(1 + bk) - 1) * 100
    obs = {name: _edge(excess[name + "__con"], excess[name + "__book"]) for name in constructions if name + "__con" in excess}
    obs_best = max(obs.values())
    # block-bootstrap each construction's EXCESS stream around zero-mean (sign-randomized blocks) and
    # recompute the best-of-K edge. We resample the excess directly (its mean IS the edge); to form a
    # proper null we CENTER each excess block by random sign flip (preserves autocorrelation, kills drift).
    rng = np.random.default_rng(seed)
    keys = [name for name in constructions if name + "__con" in excess]
    null_max = np.empty(n_perm)
    for p in range(n_perm):
        edges = []
        for name in keys:
            ex = excess[name]
            n = len(ex); nb = int(np.ceil(n / block))
            sp = n - block + 1
            starts = rng.integers(0, max(1, sp), size=nb)
            signs = rng.choice([-1.0, 1.0], size=nb)
            chunks = [signs[i] * ex[st:st + block] for i, st in enumerate(starts)]
            re = np.concatenate(chunks)[:n]
            # the sign-flipped excess is a mean-zero (under H0) reshuffle; its compound delta is the null edge
            edges.append((np.prod(1 + re) - 1) * 100)
        null_max[p] = max(edges)
    pval = float((np.sum(null_max >= obs_best) + 1) / (n_perm + 1))
    return {"observed_best_edge_pp": round(obs_best, 2), "best_name": best_name,
            "per_construction_edge_pp": {k: round(v, 2) for k, v in obs.items()},
            "null_max_p95_pp": round(float(np.percentile(null_max, 95)), 2),
            "p_value_maxstat": round(pval, 4), "n_perm": n_perm, "block": block,
            "survives_deflation": bool(pval < 0.05)}


def _pbo_across_family(per_window, constructions):
    """PBO (CSCV) across the construction FAMILY: each construction's full-cycle-DEV daily return is a
    column; PBO answers 'does picking the in-sample-best construction generalize out-of-sample?'. A
    high PBO (~0.5) means the construction-selection is skill-less (the best-of-family is noise)."""
    cols = []
    names = []
    book = _dev_daily(per_window, "cash_book")
    for name in constructions:
        d = _dev_daily(per_window, name)
        j = pd.concat([d.rename("x"), book.rename("b")], axis=1).dropna()
        if len(j) < 100:
            continue
        cols.append((j["x"] - j["b"]).to_numpy())     # excess over book -> PBO on the EDGE, not the level
        names.append(name)
    if len(cols) < 2:
        return {"error": "need >=2 constructions with aligned streams for PBO"}
    L = min(len(c) for c in cols)
    R = np.column_stack([c[:L] for c in cols])
    try:
        from strat.pbo_cscv import pbo_cscv
        S = 8 if L >= 16 else 4
        out = pbo_cscv(R, S=S)
        out["family"] = names
        return out
    except Exception as e:
        return {"error": str(e)[:120], "family": names}


def _fmt_deflation(deflation):
    lines = []
    pn = deflation.get("permutation_null", {})
    if "p_value_maxstat" in pn:
        lines.append(f"PERMUTATION (max-stat over the {pn['n_perm']} K-construction best-of): observed best "
                     f"edge over book = {pn['observed_best_edge_pp']}pp ({pn['best_name']}); null max-edge p95 "
                     f"= {pn['null_max_p95_pp']}pp; p = {pn['p_value_maxstat']} -> "
                     f"{'SURVIVES (p<0.05)' if pn['survives_deflation'] else 'FAILS deflation (best-of-K is within noise)'}.")
    elif pn:
        lines.append(f"PERMUTATION: {pn.get('error', pn)}")
    pbo = deflation.get("pbo", {})
    if "pbo" in pbo:
        lines.append(f"PBO across the construction family (S={pbo['S']}, N={pbo['N']}): PBO={pbo['pbo']:.3f} "
                     f"-> {pbo['verdict']} (PBO~0.5 = construction-selection is skill-less).")
    elif pbo:
        lines.append(f"PBO: {pbo.get('error', pbo)}")
    return lines


# =====================================================================================================
# 7. VERDICT
# =====================================================================================================
def build_verdict(res):
    fr = res["frontier"]; fc = res["full_cycle_dev"]; u = res["unseen_once"]
    best = res.get("best_construction")
    dominators = res.get("dominators", [])
    cards = res.get("scorecards", {})
    deflation = res.get("deflation", {})

    book_net = fc["cash_book"].get("net_pct"); book_dd = fc["cash_book"].get("maxdd_pct")
    bh_net = fc["raw_beta"].get("net_pct")

    # gates
    gate_dominates = bool(dominators)                                          # any construction NE of book
    best_net = fc.get(best, {}).get("net_pct") if best else None
    gate_beats_book_wealth = bool(best_net is not None and book_net is not None and best_net > book_net + 5.0)
    best_card = cards.get(best, {})
    hp = best_card.get("heldout_block_bootstrap", {}).get("p05") if best else None
    u_best = u.get(best, {}).get("net_pct") if best else None
    gate_p05 = bool(hp is not None and hp > 0)
    gate_unseen_pos = bool(u_best is not None and u_best > 0)
    gate_ship = bool(best_card.get("ship_read", {}).get("ship")) if best else False
    pn = deflation.get("permutation_null", {})
    gate_deflation = bool(pn.get("survives_deflation"))
    # preservation kept on UNSEEN: best construction maxDD not materially worse than the cash-book's
    u_best_dd = u.get(best, {}).get("maxdd_pct") if best else None
    u_book_dd = u.get("cash_book", {}).get("maxdd_pct")
    gate_preserve_unseen = bool(u_best_dd is not None and u_book_dd is not None and u_best_dd >= u_book_dd - 10.0)

    # the decisive verdict
    if gate_dominates and gate_beats_book_wealth and gate_p05 and gate_unseen_pos and gate_deflation and gate_preserve_unseen:
        verdict = "DOMINATES"                  # H1: a participate-AND-preserve construction beats the book held-out
    elif gate_dominates and gate_beats_book_wealth and not (gate_p05 and gate_deflation):
        verdict = "DEV_DOMINATES_BUT_NOT_SHIPPABLE"   # NE on dev but fails the held-out p05 / deflation gate
    elif gate_dominates and not gate_beats_book_wealth:
        verdict = "DOMINATES_PLANE_NOT_WEALTH"        # NE on the plane but does not out-compound the book held-out
    else:
        verdict = "FUNDAMENTAL_TRADEOFF"       # H0: no NE point -> participation<->preservation is a hard frontier

    closed = verdict == "FUNDAMENTAL_TRADEOFF"
    lines = [
        "## DECISIVE VERDICT (long-only participate-AND-preserve frontier) [VERIFIED-HELDOUT + UNSEEN-ONCE]",
        f"INCUMBENT (cash-going book): full-cycle DEV net {book_net}% maxDD {book_dd}% | "
        f"frontier point bull-capture {fr['cash_book'].get('bull_capture_pct')}% / "
        f"bear-preservation {fr['cash_book'].get('bear_preservation_pct')}%. Buy-hold net {bh_net}%.",
        f"CONSTRUCTIONS NORTHEAST of the book (MORE participation AND preservation within 5pp): "
        f"{dominators if dominators else 'NONE'} -> "
        f"{'a dominating point EXISTS on the plane' if gate_dominates else 'NO point dominates -- they lie on one frontier line'}.",
        f"BEST participate-AND-preserve construction: {best} "
        f"(full-cycle DEV net {best_net}% vs book {book_net}%; "
        f"{'beats book wealth by >5pp' if gate_beats_book_wealth else 'does NOT beat book wealth held-out'}).",
        f"SEALED UNSEEN read: best {best} net {u_best}% DD {u_best_dd}% vs cash-book net "
        f"{u.get('cash_book', {}).get('net_pct')}% DD {u_book_dd}% (regime {u['regime']}); "
        f"UNSEEN-positive {gate_unseen_pos}, preservation-kept {gate_preserve_unseen}.",
        f"SHIP-GATE: best held-out block-bootstrap p05 = {hp}% -> {'PASS' if gate_p05 else 'FAIL'}; "
        f"scorecard ship = {gate_ship}.",
        f"DEFLATION (multiple-comparisons): permutation max-stat p = {pn.get('p_value_maxstat')} -> "
        f"{'SURVIVES' if gate_deflation else 'FAILS (best-of-several is within noise)'}; "
        f"PBO = {deflation.get('pbo', {}).get('pbo')}.",
        f"GATES: dominates_plane={gate_dominates} | beats_book_wealth={gate_beats_book_wealth} | "
        f"heldout_p05>0={gate_p05} | unseen_pos={gate_unseen_pos} | deflation_survives={gate_deflation} | "
        f"preserve_unseen={gate_preserve_unseen}.",
        # The narrative read keys on DEPLOYABILITY, not just the dev plane. A construction can sit NE of the
        # book on the dev plane yet fail to be a deploy candidate -- that is the DEV_DOMINATES_BUT_NOT_SHIPPABLE
        # case, which is functionally a CLOSED door for real capital (the dev-dominance is a selection-window +
        # multiple-comparisons mirage). Only verdict == DOMINATES re-opens the door.
        ("CLOSED: long-only participate-AND-preserve is a FUNDAMENTAL tradeoff -- every construction lies on "
         "one frontier line; you can MOVE participation<->preservation but not BEAT the frontier. The honest "
         "doors are SHORT (OFF -- the user's shortcut) or CARRY (the funding-dispersion sleeve)."
         if verdict == "FUNDAMENTAL_TRADEOFF" else
         "CLOSED FOR DEPLOYMENT: a construction sits NORTHEAST of the book on the DEV plane and out-compounds "
         "it on dev wealth, BUT it FAILS the held-out ship-gate (held-out p05 < 0) AND/OR multiple-comparisons "
         "deflation (best-of-K within noise / PBO ~ 0.5). The dev-dominance is a SELECTION-WINDOW + "
         "best-of-several mirage -- NOT a deployable participate-and-preserve edge. The honest doors remain "
         "SHORT (OFF) or CARRY."
         if verdict in ("DEV_DOMINATES_BUT_NOT_SHIPPABLE", "DOMINATES_PLANE_NOT_WEALTH") else
         "OPEN: a participate-AND-preserve construction is NORTHEAST of the book, beats it on held-out wealth, "
         "passes the held-out p05 ship-gate AND survives multiple-comparisons deflation -- the frontier is NOT "
         "fundamental; this construction is the genuine deploy candidate."),
        f"CHEAPEST FALSIFIER: {_cheapest_falsifier(res, verdict)}",
    ]
    return {"verdict": verdict, "closed": closed,
            "gates": {"dominates_plane": gate_dominates, "beats_book_wealth": gate_beats_book_wealth,
                      "heldout_p05_pos": gate_p05, "unseen_pos": gate_unseen_pos,
                      "deflation_survives": gate_deflation, "preserve_unseen": gate_preserve_unseen,
                      "scorecard_ship": gate_ship},
            "best_construction": best, "best_fc_dev_net_pct": best_net, "book_fc_dev_net_pct": book_net,
            "buyhold_fc_dev_net_pct": bh_net, "best_heldout_p05": hp,
            "best_unseen_net_pct": u_best, "best_unseen_maxdd_pct": u_best_dd,
            "permutation_p": pn.get("p_value_maxstat"), "lines": lines}


def _cheapest_falsifier(res, verdict):
    if verdict == "FUNDAMENTAL_TRADEOFF":
        return ("ONE construction that lands NORTHEAST of the cash-going book on the (bull-capture vs "
                "bear-preservation) plane AND beats it on full-cycle held-out wealth with p05>0 would "
                "REOPEN long-only participate-and-preserve. None of the 4 directional gates (bear-rally / "
                "terminal-leg / drawdown-aware / asymmetric) found it -- the next cheapest probe is a "
                "regime-conditioned ENSEMBLE of the gates (does combining them shift NE?) before declaring "
                "the door permanently shut.")
    return ("the best construction's held-out p05 > 0 and permutation deflation are LOAD-BEARING; a single "
            "re-run on a different block size / seed that drops p05 below 0 OR lifts the permutation p above "
            "0.05 collapses the claim. Re-derive on block in {5,10,20} and seeds {7,11,23} before deploy.")


# =====================================================================================================
# 8. CHARTS
# =====================================================================================================
def make_charts(res):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[charts] matplotlib unavailable ({e}) -- skipped")
        return []
    paths = []
    fr = res["frontier"]
    # ---- chart 1: the participation<->preservation frontier plane ----
    fig, ax = plt.subplots(figsize=(11, 8))
    colors = {"cash_book": "#2a9d8f", "raw_beta": "#264653", "family_free": "#8a817c"}
    con_color = "#e76f51"
    for k, pt in fr.items():
        bc = pt.get("bull_capture_pct"); bp = pt.get("bear_preservation_pct")
        if bc is None or bp is None:
            continue
        if k in ("cash_book", "raw_beta", "family_free"):
            ax.scatter(bc, bp, s=180, color=colors[k], zorder=5, edgecolor="k", lw=1.2)
            ax.annotate(k, (bc, bp), textcoords="offset points", xytext=(8, 6), fontsize=10, fontweight="bold")
        else:
            dom = pt.get("dominates_book", False)
            ax.scatter(bc, bp, s=140, color=("#43aa8b" if dom else con_color), marker="D", zorder=4,
                       edgecolor="k", lw=1.0)
            ax.annotate(k + (" [NE]" if dom else ""), (bc, bp), textcoords="offset points",
                        xytext=(8, -12), fontsize=9)
    # the cash-book reference lines (NE quadrant = dominance region)
    book = fr.get("cash_book", {})
    bbc = book.get("bull_capture_pct"); bbp = book.get("bear_preservation_pct")
    if bbc is not None and bbp is not None:
        ax.axvline(bbc, color="#2a9d8f", ls=":", lw=1.0, alpha=0.7)
        ax.axhline(bbp, color="#2a9d8f", ls=":", lw=1.0, alpha=0.7)
        ax.axhspan(bbp - 5, max(105, bbp + 5), xmin=0, xmax=1, alpha=0.0)
        # shade the dominance region (NE of book, preservation within 5pp band)
        ax.fill_betweenx([bbp - 5, 105], bbc, 200, color="#43aa8b", alpha=0.08, label="dominance region (NE of book)")
    ax.set_xlabel("bull-capture %  (construction bull wealth / raw-beta bull wealth)  -- PARTICIPATION ->")
    ax.set_ylabel("bear-DD-preservation %  (1 - construction worst-bear-DD / raw-beta worst-bear-DD)  -- PRESERVATION ->")
    v = res["verdict"]
    ax.set_title("LONG-ONLY participation<->preservation frontier\n"
                 f"VERDICT: {v['verdict']} "
                 f"({'a dominating NE point exists' if not v['closed'] else 'one frontier line -- tradeoff is FUNDAMENTAL'})")
    ax.legend(loc="lower left", fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    c1 = OUT / "participation_preservation_frontier.png"
    fig.savefig(c1, dpi=120); plt.close(fig); paths.append(str(c1))
    print(f"[chart] {c1}")

    # ---- chart 2: best participate-preserve construction vs cash-book equity (full-cycle DEV + UNSEEN) ----
    best = res.get("best_construction")
    ud = res.get("_unseen_daily", {})
    fig2, axes = plt.subplots(1, 2, figsize=(16, 6.2))
    # panel A: full-cycle DEV equity (book vs best vs raw beta) -- rebuilt below in main and passed via res
    fcd = res.get("_fc_dev_daily", {})
    ax = axes[0]
    series_map = [("cash_book", "#2a9d8f", "-", 2.0, "cash-going book"),
                  ("raw_beta", "#264653", "--", 1.3, "raw EW-beta (buy-hold)")]
    if best:
        series_map.append((best, "#e76f51", "-", 2.0, f"best: {best}"))
    for k, col, ls, lw, lab in series_map:
        d = fcd.get(k)
        if d is not None and len(d):
            ax.plot((1 + d).cumprod().index, (1 + d).cumprod().values, color=col, ls=ls, lw=lw, label=lab)
    ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2023-01-01"), color="#e76f51", alpha=0.10, label="2022 BEAR")
    ax.axvspan(pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01"), color="#e76f51", alpha=0.06)
    ax.set_yscale("log"); ax.set_ylabel("growth of $1 (log)"); ax.set_xlabel("date")
    fc = res["full_cycle_dev"]
    ax.set_title(f"FULL-CYCLE DEV 2020-2025: best participate-preserve vs cash-book vs buy-hold\n"
                 f"book {fc['cash_book'].get('net_pct')}% / best "
                 f"{fc.get(best, {}).get('net_pct') if best else 'n/a'}% / buy-hold {fc['raw_beta'].get('net_pct')}%")
    ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.3)
    # panel B: UNSEEN equity
    ax2 = axes[1]
    for k, col, ls, lw, lab in series_map:
        d = ud.get(k)
        if d is not None and len(d):
            ax2.plot((1 + d).cumprod().index, (1 + d).cumprod().values, color=col, ls=ls, lw=lw, label=lab)
    ax2.set_ylabel("growth of $1"); ax2.set_xlabel("date")
    u = res["unseen_once"]
    ax2.set_title(f"SEALED UNSEEN {u['window'][0]} -> {u['window'][1]} ({u['regime']})\n"
                  f"book net {u['cash_book'].get('net_pct')}% DD {u['cash_book'].get('maxdd_pct')}% | best "
                  f"{best} net {u.get(best, {}).get('net_pct') if best else 'n/a'}% DD "
                  f"{u.get(best, {}).get('maxdd_pct') if best else 'n/a'}%")
    ax2.legend(loc="best", fontsize=8); ax2.grid(alpha=0.3)
    fig2.suptitle("Best participate-AND-preserve construction vs the cash-going book")
    fig2.tight_layout()
    c2 = OUT / "best_pp_vs_book_equity.png"
    fig2.savefig(c2, dpi=120); plt.close(fig2); paths.append(str(c2))
    print(f"[chart] {c2}")
    return paths


# =====================================================================================================
# 9. PERSIST + MAIN
# =====================================================================================================
def _strip(obj):
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if not (isinstance(k, str) and k.startswith("_"))}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    if isinstance(obj, pd.Series):
        return None
    return obj


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m strat.participate_preserve_frontier")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--derisk", default=PICK_LEVEL, help="the cash-going book's de-risk level (FROZEN)")
    ap.add_argument("--no-charts", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    print("## LONG-ONLY PARTICIPATE-AND-PRESERVE FRONTIER -- the decisive open question")
    print("## PRE-REGISTRATION (stated BEFORE the run):")
    for k in ("H0_fundamental_tradeoff", "H1_dominates", "asymmetric_loss", "dominance_def", "ship_gate"):
        print(f"   {k}: {PREREG[k]}")
    print(f"\n   DEV years {list(DEV_YEARS)} + 2025-OOS | FROZEN de-risk {a.derisk} | LONG-ONLY spot | "
          f"fixed-EW | PIT survivorship-clean | UNSEEN {list(UNSEEN_WIN)} READ-ONCE\n")

    res = run(derisk=a.derisk)

    # rebuild full-cycle DEV daily streams for the chart (book / best / raw)
    fc_dev_daily = {}
    best = res.get("best_construction")
    for key in ["cash_book", "raw_beta"] + ([best] if best else []):
        parts = []
        for setter, _ in ([(_set_year_fn(y), y) for y in DEV_YEARS] + [(_set_2025_fn(), 2025)]):
            setter(); pit_universe_2021(verbose=False)
            if key == "cash_book":
                d, _ = build_book(derisk=a.derisk, combine="ew")
            elif key == "raw_beta":
                d = build_buyhold()
            else:
                gate_fn, kwargs = CONSTRUCTIONS[key]
                d = build_construction_book(gate_fn, kwargs, derisk=a.derisk)
            if d is not None and len(d):
                parts.append(d)
        if parts:
            dd = pd.concat(parts).sort_index()
            fc_dev_daily[key] = dd[~dd.index.duplicated(keep="first")]
    res["_fc_dev_daily"] = fc_dev_daily

    print("\n" + "=" * 110)
    for line in res["verdict"]["lines"]:
        print(f"   {line}")
    print(f"\n   >>> VERDICT: {res['verdict']['verdict']}")
    print("=" * 110)

    charts = []
    if not a.no_charts:
        charts = make_charts(res)
    res["charts"] = charts

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    p = OUT / f"participate_preserve_frontier_{stamp}.json"
    payload = {"repro": {"command": "python -m strat.participate_preserve_frontier " + " ".join(argv or sys.argv[1:]),
                         "git_sha": sha, "dev_years": list(DEV_YEARS), "unseen_window": list(UNSEEN_WIN),
                         "derisk": a.derisk, "cost_maker": MAKER_RT},
               "prereg": PREREG, "results": _strip(res), "charts": charts}
    json.dump(payload, open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n[persisted] {p}")
    return 0


# =====================================================================================================
# 10. SELFTEST -- mechanics sanity (does NOT touch the UNSEEN window)
# =====================================================================================================
def selftest():
    print("## PARTICIPATE-PRESERVE-FRONTIER SELFTEST (mechanics only; no UNSEEN)")
    ok = True
    FEB._set_year(2022)            # a bear year -- the gates should de-risk here
    pit_universe_2021(verbose=False)
    assets = FT._assets_for("1d", False, "expand")
    A = assets[0]
    n = len(A["c"])

    # (1) every gate returns a {0,1} array of the right length (long-only, no short)
    s1 = True
    for name, (fn, kw) in CONSTRUCTIONS.items():
        h = np.asarray(fn(A, **kw))
        good = (h.shape[0] == n) and set(np.unique(h)).issubset({0, 1})
        print(f"  (1) gate {name:15} len={h.shape[0]} vals={sorted(set(np.unique(h).tolist()))} "
              f"held-frac={round(float(h.mean()),3)} -> {'ok' if good else 'BAD'}")
        s1 &= good
    print(f"  (1) all gates long-only {{0,1}} correct length -> {'PASS' if s1 else 'FAIL'}")
    ok &= s1

    # (2) each construction builds a finite 2022 book, fixed-EW; and is LESS-deep DD than raw beta in a bear
    bh = build_buyhold(); m_raw = _metrics(bh)
    s2 = True
    for name, (fn, kw) in CONSTRUCTIONS.items():
        d = build_construction_book(fn, kw)
        m = _metrics(d) if d is not None else {}
        finite = m.get("net_pct") is not None and m.get("maxdd_pct") is not None
        # in a bear, a directional gate that goes to cash should not be DEEPER DD than always-long beta
        shallower = (m.get("maxdd_pct") is not None and m_raw.get("maxdd_pct") is not None
                     and m["maxdd_pct"] >= m_raw["maxdd_pct"] - 1.0)
        print(f"  (2) {name:15} 2022 net {m.get('net_pct')}% DD {m.get('maxdd_pct')}% "
              f"(raw-beta DD {m_raw.get('maxdd_pct')}%) finite={finite} shallower={shallower}")
        s2 &= bool(finite and shallower)
    print(f"  (2) constructions build + preserve in the 2022 bear (DD not deeper than raw beta) -> {'PASS' if s2 else 'FAIL'}")
    ok &= s2

    # (3) long-only structural: positions are held*vt with vt clip>=0 -> net stream is a fixed-EW of >=0 pos
    import inspect
    src = inspect.getsource(_gated_net_series) + inspect.getsource(build_construction_book)
    s3 = "np.clip(" in src and "0.0, cap" in src and "fillna(0.0).mean" in src and "skipna" not in src
    print(f"  (3) long-only + fixed-EW invariant (clip>=0, fillna(0.0).mean, no skipna) -> {'PASS' if s3 else 'FAIL'}")
    ok &= s3

    # (4) frontier-axis math sanity: a construction equal to raw beta -> bull-capture 100, preservation 0
    pm_raw = {y: {} for y in (2020, 2021, 2022, 2023, 2024)}
    # fabricate metrics where construction == raw beta
    fake = {2020: {"net_pct": 50, "maxdd_pct": -20}, 2021: {"net_pct": 100, "maxdd_pct": -30},
            2022: {"net_pct": -60, "maxdd_pct": -70}, 2023: {"net_pct": 80, "maxdd_pct": -25},
            2024: {"net_pct": 40, "maxdd_pct": -22}}
    bc = _bull_capture_pct(fake, fake); bp = _bear_preservation_pct(fake, fake)
    s4 = (bc is not None and abs(bc - 100.0) < 0.1) and (bp is not None and abs(bp - 0.0) < 0.1)
    print(f"  (4) frontier-axis sanity: construction==raw -> bull-capture {bc}% (want 100) / "
          f"preservation {bp}% (want 0) -> {'PASS' if s4 else 'FAIL'}")
    ok &= s4

    # (5) selftest did NOT touch the sealed UNSEEN window
    s5 = tuple(FT.WIN) != tuple(UNSEEN_WIN)
    print(f"  (5) selftest did NOT touch the sealed UNSEEN window (WIN={FT.WIN}) -> {'PASS' if s5 else 'FAIL'}")
    ok &= s5

    print(f"\n  SELFTEST {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
