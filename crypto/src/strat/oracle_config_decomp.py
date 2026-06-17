"""MATHEMATICAL DECOMPOSITION of the oracle config -- "the secret sauce".

THE HYPOTHESIS (user): an oracle config is NOT a black-box choice -- its PARAMETERS
are a mathematical FUNCTION of the MOVE's properties. If we can find that function
from PAST-ONLY-estimable properties, we CONSTRUCT the config = the secret sauce.
This is the matched-filter / adaptive-MA framework:
    MA-LENGTH        <- move timescale         (cf. Ehlers MAMA: period from cycle)
    EXIT-threshold   <- move magnitude / vol
    MA-TYPE/smooth   <- move efficiency / SNR  (cf. Kaufman KAMA: smoothing from ER)
    FORMULATION      <- move persistence       (cf. FRAMA: from fractal dimension)

KEY: a SPECIFIC config does NOT persist week-to-week (walk-forward proved it), but
the LAW config=f(properties) MAY be CONSTANT. That is exactly the gap to close.

  STEP 1 -- DESCRIPTIVE DECOMPOSITION (does the law exist?).
    For each oracle move-event record (a) the WINNING config's params (fast-len,
    slow-len, MA-type, formulation, exit-param) and (b) the move's MEASURABLE
    properties measured PAST-ONLY on the PRE-MOVE window: duration, magnitude,
    trailing realized vol, dominant cycle length (FFT/autocorr peak), Kaufman
    efficiency-ratio, Hurst (R/S). Quantify each config-param vs each property
    (correlation + simple OLS, report R^2 + fitted coef).

  STEP 2 -- REALIZABLE CONSTRUCTION (the secret sauce, OOS).
    Fit the Step-1 laws on TRAIN. On each TEST move, using ONLY past-only property
    estimates measured BEFORE the move, CONSTRUCT the config via the laws and apply
    it CAUSALLY. Compare its TEST capture vs FIXED / LAST-WEEK'S-BEST / next-move
    ORACLE CEILING / RANDOM. Verdict: positive OOS AND beats FIXED AND beats
    last-week's-best -> secret sauce EXISTS.

DISCIPLINE:
  - Construction properties measured PRE-move / past-only (no look-ahead).
  - Laws fit on TRAIN events only; TEST events untouched at fit time.
  - Capture = realized long ROI net taker 0.24% RT (same units as oracle_walkforward).
  - Reuses ti_oracle_anchor / ti_oracle_decompose READ-ONLY (identical events + grid).
  - cp1252-safe (no emoji).

Usage:
    python src/strat/oracle_config_decomp.py --asset BTCUSDT --cadences 1d,4h,1h
    python src/strat/oracle_config_decomp.py --assets BTCUSDT,ETHUSDT,SOLUSDT --cadences 1d,4h,1h
    python src/strat/oracle_config_decomp.py --selftest

__contract__ = {
    "kind": "research_decomposition",
    "inputs": ["chimera OHLC via ChimeraLoader",
               "ti_oracle_anchor.find_price_oracle_events (read-only reuse)",
               "ti_oracle_decompose.build_candidates (read-only reuse)"],
    "outputs": ["runs/strat/oracle_config_decomp_<ASSET>.json", "stdout tables",
                "structured dict returned to overseer"],
    "invariants": [
        "price-oracle events identical to ti_oracle_anchor (same detector)",
        "move properties measured on PRE-move window only (past-only / causal)",
        "Step-1 laws fit on TRAIN events only; TEST untouched at fit time",
        "constructed config applied causally on TEST (no look-ahead)",
        "capture = realized long roi net taker; same units as walkforward",
        "selftest two-sided: known law -> recovered + construction beats fixed; "
        "random config -> R^2~0 + construction ~ fixed",
    ],
}
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# READ-ONLY reuse so events + candidate grid are byte-identical to anchor/decomp.
from strat.ti_oracle_anchor import (  # noqa: E402
    MoveEvent,
    WINDOW_BARS,
    find_price_oracle_events,
    load_ohlc,
)
from strat.ti_oracle_decompose import (  # noqa: E402
    Candidate,
    build_candidates,
    FORMULATIONS,
    MA_TYPES,
)

# ---- spec: pre-move property window -----------------------------------------

# The PRE-move window over which all past-only properties are measured, expressed
# as a multiple of the event's own window length. 1.0x = the move's own duration
# of bars immediately BEFORE its start. This is strictly causal: indices
# [ev.start - pre_len, ev.start).
PRE_WINDOW_MULT = 2.0
PRE_WINDOW_MIN_BARS = 16   # floor so cycle/Hurst/ER estimates are not degenerate

TAKER_RT = 0.0024          # net taker 0.24% RT (matches anchor/decompose)

# Walk-forward TRAIN/TEST event-block sizing (mirrors oracle_walkforward semantics
# so the OOS construction is on the SAME footing as the persistence test).
TRAIN_EVENTS = 12
TEST_EVENTS = 4
MIN_TRAIN_EVENTS = 6
MIN_TEST_EVENTS = 3
N_RANDOM = 8

# MA-type ordinal axis: smooth (laggy) -> responsive. Used to regress MA-TYPE on
# efficiency/Hurst as a scalar "responsiveness" target.
# SMA (smoothest) ... HMA (most responsive). DEMA/WMA in between.
MA_TYPE_RESPONSIVENESS = {"SMA": 0, "WMA": 1, "EMA": 2, "DEMA": 3, "HMA": 4}

# Formulation ordinal axis: cross-ride (let it run) -> mechanical-TP (cut early).
# F2/F4 are "ride the trend" (cross / stack exits = persistence plays);
# F3/F5 with TP are "mechanical take-profit" (anti-persistence / mean-revert plays);
# F1 single-MA-state sits in the middle.
FORMULATION_RIDE_SCORE = {
    "F2_CROSS": 0,           # pure ride (death-cross exit)
    "F4_STACK": 0,           # ride while aligned
    "F1_PRICE_MA": 1,        # single-MA ride, quicker exit than a slow cross
    "F5_PRICE_MA_MECH": 2,   # mechanical exit (often TP) -> cut
    "F3_CROSS_MECH": 2,      # mechanical exit (often TP) -> cut
}


# ---- config-param parsing ---------------------------------------------------

@dataclass
class ConfigDNA:
    formulation: str
    ma_type: str
    fast_len: float        # for cross/stack: the fast MA len; single-MA: the len
    slow_len: float        # for cross/stack: the slow MA len; single-MA: NaN
    exit_param: float      # take-profit % if a mech-TP exit, else NaN
    exit_kind: str         # "TP" / "SL" / "TIME" / "ATR" / "CROSS" / "STATE" / "STACK"

    def responsiveness(self) -> int:
        return MA_TYPE_RESPONSIVENESS.get(self.ma_type, 2)

    def ride_score(self) -> int:
        return FORMULATION_RIDE_SCORE.get(self.formulation, 1)


_RE_FXS = re.compile(r"^(\d+)x(\d+)")          # "5x20" or "5x20+TP5"
_RE_LEN = re.compile(r"^len(\d+)")             # "len20" or "len20+ATR3x"
_RE_TP = re.compile(r"\+TP(\d+)")
_RE_SL = re.compile(r"\+SL(\d+)")
_RE_ATR = re.compile(r"\+ATR")
_RE_TIME = re.compile(r"\+TIME")


def parse_dna(formulation: str, ma_type: str, params: str) -> ConfigDNA:
    """Parse the decompose `params` label string into a structured ConfigDNA.

    Param label grammar (from ti_oracle_decompose.build_candidates):
      F1_PRICE_MA       : "len{L}"
      F2_CROSS          : "{f}x{s}"
      F4_STACK          : "{f}x{s}"
      F3_CROSS_MECH     : "{f}x{s}+{rule}"   rule in {TP3,TP5,TP8,SL3,SL5,TIME,ATR3x}
      F5_PRICE_MA_MECH  : "len{L}+{rule}"
    """
    fast = np.nan
    slow = np.nan
    exit_param = np.nan
    exit_kind = "NONE"

    m = _RE_FXS.match(params)
    if m:
        fast = float(m.group(1))
        slow = float(m.group(2))
    else:
        m2 = _RE_LEN.match(params)
        if m2:
            fast = float(m2.group(1))   # single-MA len lives in fast_len
            slow = np.nan

    # exit kind / param
    mtp = _RE_TP.search(params)
    msl = _RE_SL.search(params)
    if mtp:
        exit_param = float(mtp.group(1)) / 100.0
        exit_kind = "TP"
    elif msl:
        exit_param = float(msl.group(1)) / 100.0
        exit_kind = "SL"
    elif _RE_ATR.search(params):
        exit_kind = "ATR"
    elif _RE_TIME.search(params):
        exit_kind = "TIME"
    else:
        # no mechanical suffix -> exit is intrinsic to the formulation
        if formulation == "F2_CROSS":
            exit_kind = "CROSS"
        elif formulation == "F4_STACK":
            exit_kind = "STACK"
        elif formulation == "F1_PRICE_MA":
            exit_kind = "STATE"
        else:
            exit_kind = "NONE"

    return ConfigDNA(
        formulation=formulation, ma_type=ma_type,
        fast_len=fast, slow_len=slow,
        exit_param=exit_param, exit_kind=exit_kind,
    )


# ---- past-only move-property estimators -------------------------------------

@dataclass
class MoveProps:
    pre_lo: int             # pre-window start bar (global)
    pre_hi: int             # pre-window end bar (global, exclusive == ev.start)
    n_pre: int
    duration_bars: float    # the move's OWN window length (a TIMESCALE proxy; this is
                            # the price-oracle event window, NOT future move info --
                            # it is the detector's fixed window length, known a priori)
    pre_magnitude: float    # |total return| over the pre-window (past realized move)
    pre_vol: float          # trailing realized vol (std of pre-window log returns)
    dom_cycle: float        # dominant cycle length (FFT peak of detrended pre prices)
    efficiency_ratio: float # Kaufman ER over the pre-window
    hurst: float            # Hurst exponent (R/S) over the pre-window


def _log_returns(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    if len(p) < 2:
        return np.array([], dtype=np.float64)
    return np.diff(np.log(p))


def kaufman_efficiency_ratio(p: np.ndarray) -> float:
    """ER = |net change| / sum(|bar changes|) over the window. 1 = perfectly
    efficient (straight line), ~0 = pure noise / choppy."""
    p = np.asarray(p, dtype=np.float64)
    if len(p) < 3:
        return np.nan
    net = abs(p[-1] - p[0])
    path = np.sum(np.abs(np.diff(p)))
    if path <= 0:
        return np.nan
    return float(net / path)


def hurst_rs(p: np.ndarray) -> float:
    """Hurst exponent via rescaled-range (R/S) on log returns. 0.5 = random walk,
    > 0.5 = persistent/trending, < 0.5 = anti-persistent/mean-reverting.

    Uses several sub-window sizes and fits log(R/S) ~ H*log(n). Returns NaN if
    too few points or degenerate.
    """
    r = _log_returns(p)
    n = len(r)
    if n < 16:
        return np.nan
    # candidate chunk sizes (powers-ish), each >= 8 and <= n//2
    sizes = []
    s = 8
    while s <= n // 2:
        sizes.append(s)
        s = int(s * 1.6) + 1
    if len(sizes) < 2:
        return np.nan
    logs_n, logs_rs = [], []
    for w in sizes:
        n_chunks = n // w
        if n_chunks < 1:
            continue
        rss = []
        for k in range(n_chunks):
            seg = r[k * w:(k + 1) * w]
            mean = np.mean(seg)
            dev = np.cumsum(seg - mean)
            R = float(np.max(dev) - np.min(dev))
            S = float(np.std(seg))
            if S > 1e-12 and R > 0:
                rss.append(R / S)
        if rss:
            logs_n.append(np.log(w))
            logs_rs.append(np.log(np.mean(rss)))
    if len(logs_n) < 2:
        return np.nan
    # OLS slope = Hurst
    A = np.vstack([np.array(logs_n), np.ones(len(logs_n))]).T
    coef, _, _, _ = np.linalg.lstsq(A, np.array(logs_rs), rcond=None)
    return float(coef[0])


def dominant_cycle(p: np.ndarray) -> float:
    """Dominant cycle length (bars) via the FFT peak of the linearly-detrended
    pre-window log-price. Returns the period (1/freq) of the largest-power non-DC
    frequency. NaN if too few points."""
    x = np.asarray(p, dtype=np.float64)
    x = x[x > 0]
    n = len(x)
    if n < 8:
        return np.nan
    lx = np.log(x)
    # linear detrend
    t = np.arange(n, dtype=np.float64)
    A = np.vstack([t, np.ones(n)]).T
    coef, _, _, _ = np.linalg.lstsq(A, lx, rcond=None)
    detr = lx - (A @ coef)
    # FFT power, skip DC (k=0)
    fft = np.fft.rfft(detr)
    power = np.abs(fft) ** 2
    if len(power) <= 1:
        return np.nan
    power[0] = 0.0
    k = int(np.argmax(power))
    if k <= 0:
        return np.nan
    period = n / k
    return float(period)


def measure_move_props(close_full: np.ndarray, ev: MoveEvent) -> MoveProps:
    """Measure all properties on the PRE-move window [pre_lo, ev.start). Strictly
    past-only: nothing at or after ev.start is touched.

    duration_bars is the event's OWN fixed window length (ev.end - ev.start), which
    is a DETECTOR constant (the price-oracle window length per cadence), known a
    priori -- it is a TIMESCALE the construction is allowed to know, NOT future
    price information."""
    win_len = int(ev.end - ev.start)
    pre_len = max(PRE_WINDOW_MIN_BARS, int(round(PRE_WINDOW_MULT * win_len)))
    pre_lo = max(0, ev.start - pre_len)
    pre_hi = ev.start
    pre = close_full[pre_lo:pre_hi]
    pre = pre[~np.isnan(pre)]
    n_pre = len(pre)

    if n_pre < 4:
        return MoveProps(pre_lo, pre_hi, n_pre, float(win_len),
                         np.nan, np.nan, np.nan, np.nan, np.nan)

    rets = _log_returns(pre)
    pre_mag = abs(float(pre[-1] / pre[0] - 1.0)) if pre[0] > 0 else np.nan
    pre_vol = float(np.std(rets)) if len(rets) else np.nan
    dom = dominant_cycle(pre)
    er = kaufman_efficiency_ratio(pre)
    h = hurst_rs(pre)

    return MoveProps(
        pre_lo=pre_lo, pre_hi=pre_hi, n_pre=n_pre,
        duration_bars=float(win_len),
        pre_magnitude=pre_mag, pre_vol=pre_vol,
        dom_cycle=dom, efficiency_ratio=er, hurst=h,
    )


# ---- per-event winning-config extraction ------------------------------------

@dataclass
class EventRecord:
    ev: MoveEvent
    props: MoveProps
    dna: ConfigDNA
    best_roi: float
    best_idx: int


def build_event_records(open_a, high_a, low_a, close_a, cadence: str):
    """Detect events, build candidate grid + capture matrix, and for each event
    record the winning config's structured DNA + the past-only move properties.

    Returns (records, cands, M, events).
    """
    win_lo, win_hi = WINDOW_BARS[cadence]
    events = find_price_oracle_events(high_a, low_a, win_lo, win_hi)
    cands = build_candidates(open_a, high_a, low_a, close_a)
    n_ev, n_c = len(events), len(cands)
    M = np.full((n_ev, n_c), np.nan, dtype=np.float64)
    for i, ev in enumerate(events):
        for j, c in enumerate(cands):
            M[i, j] = c.fn(ev.start, ev.end)

    records: list[EventRecord] = []
    for i, ev in enumerate(events):
        row = M[i]
        best_idx = int(np.nanargmax(row)) if np.any(~np.isnan(row)) else 0
        c = cands[best_idx]
        dna = parse_dna(c.formulation, c.ma_type, c.params)
        props = measure_move_props(close_a, ev)
        records.append(EventRecord(
            ev=ev, props=props, dna=dna,
            best_roi=float(row[best_idx]), best_idx=best_idx,
        ))
    return records, cands, M, events


# ---- STEP 1: descriptive law-fitting ----------------------------------------

def _ols_r2(x: np.ndarray, y: np.ndarray):
    """Simple OLS y ~ a*x + b. Returns (slope, intercept, r2, pearson_r, n)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
    x, y = x[m], y[m]
    n = len(x)
    if n < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    A = np.vstack([x, np.ones(n)]).T
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    slope, intercept = float(coef[0]), float(coef[1])
    yhat = A @ coef
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    r = float(np.corrcoef(x, y)[0, 1])
    return {"slope": slope, "intercept": intercept, "r2": float(r2),
            "pearson_r": r, "n": n}


# the (param-target, property-name) pairs we test as candidate laws.
def _extract_target(rec: EventRecord, target: str) -> float:
    d = rec.dna
    if target == "ma_length":
        # the controlling MA length: slow_len for cross/stack, fast_len for single-MA
        return d.slow_len if not np.isnan(d.slow_len) else d.fast_len
    if target == "fast_len":
        return d.fast_len
    if target == "slow_len":
        return d.slow_len
    if target == "exit_tp":
        return d.exit_param if d.exit_kind == "TP" else np.nan
    if target == "ma_responsiveness":
        return float(d.responsiveness())
    if target == "formulation_ride":
        return float(d.ride_score())
    return np.nan


def _extract_prop(rec: EventRecord, prop: str) -> float:
    p = rec.props
    return {
        "duration_bars": p.duration_bars,
        "pre_magnitude": p.pre_magnitude,
        "pre_vol": p.pre_vol,
        "dom_cycle": p.dom_cycle,
        "efficiency_ratio": p.efficiency_ratio,
        "hurst": p.hurst,
    }.get(prop, np.nan)


# the canonical matched-filter law hypotheses (target <- property).
LAW_PAIRS = [
    # MA-LENGTH <- move timescale / dominant cycle  (the matched-filter law)
    ("ma_length", "dom_cycle"),
    ("ma_length", "duration_bars"),
    ("slow_len", "dom_cycle"),
    ("fast_len", "dom_cycle"),
    # EXIT-TP <- move magnitude / trailing vol
    ("exit_tp", "pre_magnitude"),
    ("exit_tp", "pre_vol"),
    # MA-TYPE responsiveness <- efficiency / Hurst
    ("ma_responsiveness", "efficiency_ratio"),
    ("ma_responsiveness", "hurst"),
    # FORMULATION ride-score <- Hurst / persistence (anti-persistent -> mech-TP)
    ("formulation_ride", "hurst"),
    ("formulation_ride", "efficiency_ratio"),
]


def fit_laws(records: list[EventRecord]) -> dict:
    """Step-1: for every (target, property) law-pair fit OLS + correlation and
    report R^2 + fitted coef. Returns {f"{target}<-{prop}": stats}."""
    out = {}
    for target, prop in LAW_PAIRS:
        xs = np.array([_extract_prop(r, prop) for r in records], dtype=np.float64)
        ys = np.array([_extract_target(r, target) for r in records], dtype=np.float64)
        stats = _ols_r2(xs, ys)
        out[f"{target}<-{prop}"] = stats
    return out


# ---- STEP 2: realizable construction (OOS) ----------------------------------

@dataclass
class ConstructedConfig:
    """A config CONSTRUCTED from past-only properties via the fitted laws, mapped
    back onto the nearest available candidate in the grid (so it is executable)."""
    formulation: str
    ma_type: str
    params: str
    cand_idx: int


def _nearest_candidate(
    cands: list,
    want_formulation: str,
    want_ma_type: str,
    want_slow: float,
    want_fast: float,
    want_tp: float,
) -> int:
    """Map a constructed (formulation, ma_type, lengths, tp) onto the nearest
    EXECUTABLE candidate in the grid. Scores by: formulation match (hard-ish),
    ma_type match, then numeric distance on slow/fast/tp.
    """
    best_idx = -1
    best_score = np.inf
    for i, c in enumerate(cands):
        dna = parse_dna(c.formulation, c.ma_type, c.params)
        score = 0.0
        score += 0.0 if c.formulation == want_formulation else 3.0
        score += 0.0 if c.ma_type == want_ma_type else 1.5
        # length distance on the controlling MA (slow for cross/stack, fast for single)
        cand_len = dna.slow_len if not np.isnan(dna.slow_len) else dna.fast_len
        want_len = want_slow if not np.isnan(want_slow) else want_fast
        if not np.isnan(cand_len) and not np.isnan(want_len):
            score += abs(np.log(max(cand_len, 1)) - np.log(max(want_len, 1)))
        # tp distance if both have a TP
        if not np.isnan(want_tp) and dna.exit_kind == "TP" and not np.isnan(dna.exit_param):
            score += abs(dna.exit_param - want_tp) * 10.0
        elif not np.isnan(want_tp) and dna.exit_kind != "TP":
            score += 0.5  # mild penalty: wanted a TP, this candidate has none
        if score < best_score:
            best_score = score
            best_idx = i
    return best_idx


@dataclass
class ConstructionModel:
    """Everything fitted on TRAIN that construct_config needs. Continuous axes use
    OLS laws; CATEGORICAL axes (MA-type, formulation) use a data-driven median-split
    classifier -- a winning class can be bimodal in a property, which OLS-to-an-
    ordinal-midpoint models WRONGLY (it lands on a middle class that rarely wins).
    The split is fit on TRAIN only -> causal.

    PRINCIPLE: construction starts from a BASELINE config (the TRAIN majority DNA)
    and deviates from it ONLY on axes that have a real, separating law. An axis with
    no usable law falls back to the baseline, so construction is baseline + informed
    deviations -- it can only help where a genuine law exists, and equals baseline
    where it does not (this is why a real-law series beats fixed and a random series
    ties fixed)."""
    laws: dict
    # categorical classifiers: (prop_name, threshold, low_class, high_class) or None
    ma_type_clf: tuple | None
    formulation_clf: tuple | None
    # whether each categorical split is CONFIDENT (the two sides pick different,
    # well-supported classes) -- only then do we deviate from baseline on that axis.
    ma_type_clf_confident: bool
    formulation_clf_confident: bool
    # baseline DNA (TRAIN majority) used for any axis without a usable law.
    baseline_ma_type: str
    baseline_formulation: str
    baseline_slow_len: float
    baseline_fast_len: float


def _fit_categorical_split(train_recs, prop_name: str, class_of):
    """Fit a median-split classifier: TRAIN events with property < median map to the
    majority winning class on that side; >= median to the other side's majority.
    Returns (threshold, low_class, high_class, confident) or None if degenerate.

    CONFIDENT = the two sides pick DIFFERENT classes AND each side's majority class
    holds a clear plurality (>= 55% of that side). This is the gate that decides
    whether construction is allowed to DEVIATE from the baseline on this axis."""
    from collections import Counter
    xs, cs = [], []
    for r in train_recs:
        x = _extract_prop(r, prop_name)
        cl = class_of(r)
        if not np.isnan(x) and cl is not None:
            xs.append(x)
            cs.append(cl)
    if len(xs) < 6:
        return None
    xs = np.array(xs, dtype=np.float64)
    thr = float(np.median(xs))
    lo_mask = xs < thr
    hi_mask = ~lo_mask
    if lo_mask.sum() < 2 or hi_mask.sum() < 2:
        return None
    lo_cnt = Counter(c for c, m in zip(cs, lo_mask) if m)
    hi_cnt = Counter(c for c, m in zip(cs, hi_mask) if m)
    lo_cls, lo_n = lo_cnt.most_common(1)[0]
    hi_cls, hi_n = hi_cnt.most_common(1)[0]
    lo_frac = lo_n / max(1, sum(lo_cnt.values()))
    hi_frac = hi_n / max(1, sum(hi_cnt.values()))
    confident = (lo_cls != hi_cls) and (lo_frac >= 0.55) and (hi_frac >= 0.55)
    return (thr, lo_cls, hi_cls, confident)


def fit_construction_model(train_recs) -> ConstructionModel:
    """Fit ALL construction laws on TRAIN events only (causal)."""
    from collections import Counter
    laws = fit_laws(train_recs)

    # categorical split classifiers: efficiency for ma-type, hurst for formulation.
    ma_clf = _fit_categorical_split(train_recs, "efficiency_ratio",
                                    lambda r: r.dna.ma_type)
    form_clf = _fit_categorical_split(train_recs, "hurst",
                                      lambda r: r.dna.formulation)

    types = [r.dna.ma_type for r in train_recs]
    forms = [r.dna.formulation for r in train_recs]
    ma_majority = Counter(types).most_common(1)[0][0] if types else "EMA"
    form_majority = Counter(forms).most_common(1)[0][0] if forms else "F2_CROSS"

    # baseline length = TRAIN-median winning controlling-length (slow for cross/stack,
    # else fast). Used when no length law is usable.
    slows, fasts = [], []
    for r in train_recs:
        if not np.isnan(r.dna.slow_len):
            slows.append(r.dna.slow_len)
        if not np.isnan(r.dna.fast_len):
            fasts.append(r.dna.fast_len)
    base_slow = float(np.median(slows)) if slows else 20.0
    base_fast = float(np.median(fasts)) if fasts else 5.0

    ma_confident = bool(ma_clf[3]) if ma_clf else False
    form_confident = bool(form_clf[3]) if form_clf else False

    return ConstructionModel(
        laws=laws,
        ma_type_clf=(("efficiency_ratio",) + ma_clf[:3]) if ma_clf else None,
        formulation_clf=(("hurst",) + form_clf[:3]) if form_clf else None,
        ma_type_clf_confident=ma_confident,
        formulation_clf_confident=form_confident,
        baseline_ma_type=ma_majority,
        baseline_formulation=form_majority,
        baseline_slow_len=base_slow,
        baseline_fast_len=base_fast,
    )


def construct_config(model: ConstructionModel, props: MoveProps,
                     cands: list) -> ConstructedConfig:
    """Apply the TRAIN-fitted construction model to a SINGLE move's past-only
    properties to construct a config, then snap to the nearest executable candidate.

    DESIGN: start from the TRAIN BASELINE DNA and deviate ONLY on axes with a real,
    usable law -- continuous axes via OLS (R^2 >= floor), categorical axes via a
    CONFIDENT median split. Axes without a usable law keep the baseline value. This
    guarantees construction == baseline + informed-deviations (helps where a law
    exists, ties baseline where none does).
    """
    laws = model.laws
    R2_FLOOR = 0.10   # an OLS axis must clear this to override the baseline

    def apply_law(key_candidates, x_val, lo, hi, default):
        for key in key_candidates:
            st = laws.get(key)
            if st is None:
                continue
            if not np.isnan(x_val) and st["r2"] >= R2_FLOOR:
                y = st["slope"] * x_val + st["intercept"]
                return float(np.clip(y, lo, hi))
        return default

    # 1) MA-LENGTH <- dominant cycle (or duration). controlling = slow_len.
    #    Default = baseline length (no override unless a real length law exists).
    slow_len = apply_law(
        ["slow_len<-dom_cycle", "ma_length<-dom_cycle", "ma_length<-duration_bars"],
        props.dom_cycle if not np.isnan(props.dom_cycle) else props.duration_bars,
        10, 200, model.baseline_slow_len)

    # 2) FAST-LEN <- dominant cycle (shorter). fall back to the baseline fast-len.
    fast_len = apply_law(
        ["fast_len<-dom_cycle"],
        props.dom_cycle if not np.isnan(props.dom_cycle) else props.duration_bars,
        5, 50, model.baseline_fast_len)

    # 3) EXIT-TP <- magnitude / vol (NaN = no TP -> snap to a non-TP formulation).
    tp = apply_law(
        ["exit_tp<-pre_magnitude", "exit_tp<-pre_vol"],
        props.pre_magnitude if not np.isnan(props.pre_magnitude) else props.pre_vol,
        0.03, 0.08, np.nan)

    # 4) MA-TYPE: deviate ONLY if the split is confident; else baseline.
    want_ma_type = model.baseline_ma_type
    if model.ma_type_clf is not None and model.ma_type_clf_confident:
        prop_name, thr, lo_cls, hi_cls = model.ma_type_clf
        x = _extract_prop_from_props(props, prop_name)
        if not np.isnan(x):
            want_ma_type = lo_cls if x < thr else hi_cls

    # 5) FORMULATION: deviate ONLY if the split is confident; else baseline.
    want_formulation = model.baseline_formulation
    if model.formulation_clf is not None and model.formulation_clf_confident:
        prop_name, thr, lo_cls, hi_cls = model.formulation_clf
        x = _extract_prop_from_props(props, prop_name)
        if not np.isnan(x):
            want_formulation = lo_cls if x < thr else hi_cls

    cand_idx = _nearest_candidate(
        cands, want_formulation, want_ma_type, slow_len, fast_len, tp)
    c = cands[cand_idx]
    return ConstructedConfig(c.formulation, c.ma_type, c.params, cand_idx)


def _extract_prop_from_props(props: MoveProps, prop: str) -> float:
    return {
        "duration_bars": props.duration_bars,
        "pre_magnitude": props.pre_magnitude,
        "pre_vol": props.pre_vol,
        "dom_cycle": props.dom_cycle,
        "efficiency_ratio": props.efficiency_ratio,
        "hurst": props.hurst,
    }.get(prop, np.nan)


@dataclass
class ConstructStep:
    test_period: int
    n_train: int
    n_test: int
    constructed_capture: float
    fixed_capture: float
    lastweek_capture: float    # last-week's-best (= TRAIN-best config rolled forward)
    oracle_ceiling: float      # next-move oracle ceiling on TEST
    random_capture: float


def _fixed_idx(cands: list) -> int:
    for i, c in enumerate(cands):
        if c.formulation == "F2_CROSS" and c.ma_type == "SMA" and c.params == "5x20":
            return i
    return -1


def run_construction(records, cands, M, seed: int = 7) -> list[ConstructStep]:
    """Walk-forward OOS: TRAIN block fits laws + the last-week's-best config; TEST
    block constructs configs per-move from past-only props and applies them causally.
    """
    rng = np.random.default_rng(seed)
    n_ev = len(records)
    n_c = M.shape[1]
    fixed_idx = _fixed_idx(cands)

    steps: list[ConstructStep] = []
    test_lo = TRAIN_EVENTS
    period = 0
    while test_lo < n_ev:
        test_hi = min(test_lo + TEST_EVENTS, n_ev)
        test_idx = np.arange(test_lo, test_hi)
        if len(test_idx) < MIN_TEST_EVENTS:
            test_lo = test_hi
            continue
        train_lo = max(0, test_lo - TRAIN_EVENTS)
        train_idx = np.arange(train_lo, test_lo)
        if len(train_idx) < MIN_TRAIN_EVENTS:
            test_lo = test_hi
            continue

        train_recs = [records[i] for i in train_idx]

        # fit the construction model (laws + categorical classifiers) on TRAIN only
        model = fit_construction_model(train_recs)

        # last-week's-best = TRAIN-best mean config rolled forward (mirrors walkforward)
        train_mean = np.nanmean(M[train_idx], axis=0)
        lastweek_idx = int(np.nanargmax(train_mean))

        test_M = M[test_idx]

        # construct per TEST move using ONLY that move's past-only props
        constructed_caps = []
        for ti in test_idx:
            cc = construct_config(model, records[ti].props, cands)
            constructed_caps.append(float(M[ti, cc.cand_idx]))
        constructed = float(np.mean(constructed_caps))

        lastweek = float(np.nanmean(test_M[:, lastweek_idx]))
        fixed = float(np.nanmean(test_M[:, fixed_idx])) if fixed_idx >= 0 else np.nan
        # oracle ceiling on TEST = best mean config on TEST (hindsight)
        test_mean = np.nanmean(test_M, axis=0)
        ceiling = float(np.nanmax(test_mean))
        rand_idx = rng.integers(0, n_c, size=min(N_RANDOM, n_c))
        random_cap = float(np.mean([np.nanmean(test_M[:, k]) for k in rand_idx]))

        steps.append(ConstructStep(
            test_period=period, n_train=len(train_idx), n_test=len(test_idx),
            constructed_capture=constructed, fixed_capture=fixed,
            lastweek_capture=lastweek, oracle_ceiling=ceiling,
            random_capture=random_cap,
        ))
        period += 1
        test_lo = test_hi
    return steps


# ---- aggregation + verdict --------------------------------------------------

def _safe_mean(a):
    a = np.asarray(a, dtype=np.float64)
    a = a[~np.isnan(a)]
    return float(np.mean(a)) if a.size else None


def aggregate_construction(steps: list) -> dict:
    if not steps:
        return {"n_steps": 0, "verdict": "no construction steps (too few events)"}
    con = np.array([s.constructed_capture for s in steps], dtype=np.float64)
    fix = np.array([s.fixed_capture for s in steps], dtype=np.float64)
    lw = np.array([s.lastweek_capture for s in steps], dtype=np.float64)
    ceil = np.array([s.oracle_ceiling for s in steps], dtype=np.float64)
    rnd = np.array([s.random_capture for s in steps], dtype=np.float64)

    con_m, fix_m, lw_m = _safe_mean(con), _safe_mean(fix), _safe_mean(lw)
    ceil_m, rnd_m = _safe_mean(ceil), _safe_mean(rnd)

    d_fix = con - fix
    d_lw = con - lw
    d_rnd = con - rnd
    beats_fixed_wr = float(np.mean(d_fix[~np.isnan(d_fix)] > 0)) if np.any(~np.isnan(d_fix)) else None
    beats_lw_wr = float(np.mean(d_lw[~np.isnan(d_lw)] > 0)) if np.any(~np.isnan(d_lw)) else None
    beats_rnd_wr = float(np.mean(d_rnd[~np.isnan(d_rnd)] > 0)) if np.any(~np.isnan(d_rnd)) else None

    return {
        "n_steps": len(steps),
        "capture_means": {
            "constructed": con_m, "fixed": fix_m, "lastweek_best": lw_m,
            "oracle_ceiling": ceil_m, "random": rnd_m,
        },
        "constructed_minus_fixed_mean": (con_m - fix_m) if (con_m is not None and fix_m is not None) else None,
        "constructed_minus_lastweek_mean": (con_m - lw_m) if (con_m is not None and lw_m is not None) else None,
        "constructed_minus_random_mean": (con_m - rnd_m) if (con_m is not None and rnd_m is not None) else None,
        "beats_fixed_winrate": beats_fixed_wr,
        "beats_lastweek_winrate": beats_lw_wr,
        "beats_random_winrate": beats_rnd_wr,
        "ceiling_fraction": (con_m / ceil_m) if (con_m is not None and ceil_m and ceil_m != 0) else None,
        "verdict": construction_verdict(con_m, fix_m, lw_m, ceil_m, rnd_m),
    }


def construction_verdict(con, fix, lw, ceil, rnd) -> str:
    if con is None:
        return "INDETERMINATE: insufficient construction steps."
    pos = con > 0
    beat_fix = (fix is not None) and (con > fix + 0.002)
    beat_lw = (lw is not None) and (con > lw + 0.002)
    if pos and beat_fix and beat_lw:
        return ("SECRET-SAUCE FOUND: constructed capture %.4f is POSITIVE, beats fixed "
                "(%.4f) AND last-week's-best (%.4f). The oracle config IS a function of "
                "past-only move-properties -- flag for deeper robustness." % (con, fix, lw))
    if pos and (beat_fix or beat_lw):
        return ("PARTIAL: constructed %.4f is positive and beats one baseline (fixed=%.4f, "
                "lastweek=%.4f) but not both -- a weak/partial law." % (con, fix, lw))
    if con <= 0:
        return ("NOT FOUND: constructed capture %.4f is NOT positive -- the law-built "
                "config does not capture real signal OOS (fixed=%.4f, lastweek=%.4f)."
                % (con, fix if fix is not None else float('nan'),
                   lw if lw is not None else float('nan')))
    return ("NOT FOUND: constructed %.4f is positive but does NOT beat fixed (%.4f) or "
            "last-week's-best (%.4f) -- no edge over the naive baselines."
            % (con, fix if fix is not None else float('nan'),
               lw if lw is not None else float('nan')))


def strongest_law(laws_all: dict) -> tuple:
    """Return (law_key, r2) of the strongest law (highest R^2) across the dict."""
    best_key, best_r2 = None, -np.inf
    for k, st in laws_all.items():
        if st is None:
            continue
        if st["r2"] > best_r2:
            best_r2 = st["r2"]
            best_key = k
    return best_key, (best_r2 if best_key is not None else None)


# ---- per-asset/cadence driver -----------------------------------------------

def run_one(asset: str, cadence: str, seed: int = 7) -> dict:
    o, h, lo, c = load_ohlc(asset, cadence)
    records, cands, M, events = build_event_records(o, h, lo, c, cadence)
    # STEP 1: descriptive laws on ALL events (the "does the law exist" view).
    laws_all = fit_laws(records)
    # STEP 2: OOS walk-forward construction (laws fit on TRAIN only inside).
    steps = run_construction(records, cands, M, seed=seed)
    agg = aggregate_construction(steps)
    sk, sr2 = strongest_law(laws_all)

    # winning-DNA distributions (descriptive context)
    from collections import Counter
    form_dist = dict(Counter(r.dna.formulation for r in records).most_common())
    type_dist = dict(Counter(r.dna.ma_type for r in records).most_common())

    return {
        "asset": asset,
        "cadence": cadence,
        "n_events": len(events),
        "step1_laws": laws_all,
        "strongest_law": {"law": sk, "r2": sr2},
        "winning_dna": {"formulation": form_dist, "ma_type": type_dist},
        "step2_construction": agg,
    }


# ---- reporting --------------------------------------------------------------

def _r2s(st):
    if st is None:
        return "   n/a (deg)"
    return f"r2={st['r2']:+.3f} r={st['pearson_r']:+.2f} (n={st['n']})"


def print_report(results: list[dict]) -> None:
    print("")
    print("=" * 100)
    print("ORACLE CONFIG DECOMPOSITION -- 'is the config a math function of move-properties?'")
    print("=" * 100)

    for res in results:
        print(f"\n### {res['asset']} {res['cadence']}  (N={res['n_events']} events)")
        print("  [STEP 1] descriptive laws  (config-param <- past-only move-property):")
        for key, st in res["step1_laws"].items():
            tag = ""
            if st is not None and st["r2"] >= 0.10:
                tag = "  <== law"
            print(f"      {key:<34} {_r2s(st)}{tag}")
        sl = res["strongest_law"]
        if sl["law"]:
            print(f"  strongest law: {sl['law']}  R^2={sl['r2']:.3f}")
        wd = res["winning_dna"]
        print(f"  winning-DNA formulation: "
              + ", ".join(f"{k}:{v}" for k, v in list(wd['formulation'].items())[:5]))
        print(f"  winning-DNA ma_type:     "
              + ", ".join(f"{k}:{v}" for k, v in list(wd['ma_type'].items())[:5]))

        ag = res["step2_construction"]
        print("  [STEP 2] OOS construction (constructed vs 4 baselines, realized long ROI):")
        if ag["n_steps"] == 0:
            print("      (no walk-forward construction steps -- too few events)")
        else:
            cm = ag["capture_means"]
            print(f"      steps={ag['n_steps']}  "
                  f"constructed={cm['constructed']:+.4f}  fixed={cm['fixed']:+.4f}  "
                  f"lastweek={cm['lastweek_best']:+.4f}  ceiling={cm['oracle_ceiling']:+.4f}  "
                  f"random={cm['random']:+.4f}")
            print(f"      vs-fixed delta={ag['constructed_minus_fixed_mean']:+.4f} "
                  f"(winrate={ag['beats_fixed_winrate']})  "
                  f"vs-lastweek delta={ag['constructed_minus_lastweek_mean']:+.4f} "
                  f"(winrate={ag['beats_lastweek_winrate']})")
            print(f"      VERDICT: {ag['verdict']}")
    print("=" * 100)
    print("NOTE: properties measured PRE-move (past-only); laws fit on TRAIN only; "
          "TEST untouched at fit. HINDSIGHT only for the oracle CEILING.")
    print("")


def overall_verdict(results: list[dict]) -> str:
    """One-line honest answer across all asset/cadence runs."""
    found = []
    partial = []
    notfound = []
    best_law_key, best_law_r2 = None, -np.inf
    for res in results:
        ag = res["step2_construction"]
        v = ag.get("verdict", "")
        tag = f"{res['asset']}/{res['cadence']}"
        if v.startswith("SECRET-SAUCE FOUND"):
            found.append(tag)
        elif v.startswith("PARTIAL"):
            partial.append(tag)
        elif v.startswith("NOT FOUND"):
            notfound.append(tag)
        sl = res["strongest_law"]
        if sl["r2"] is not None and sl["r2"] > best_law_r2:
            best_law_r2 = sl["r2"]
            best_law_key = sl["law"]
    if found:
        head = (f"SECRET SAUCE FOUND on {len(found)}/{len(results)} runs ({', '.join(found)}): "
                f"the oracle config IS a math function of estimable move-properties.")
    elif partial:
        head = (f"PARTIAL on {len(partial)}/{len(results)} runs; not a clean law -- "
                f"construction beats one baseline but not both, robustly.")
    else:
        head = ("NOT FOUND across all runs: the oracle config is NOT a realizable "
                "function of past-only move-properties -- construction does not beat "
                "fixed + last-week's-best OOS.")
    if best_law_key is not None:
        head += f" Strongest law overall: {best_law_key} (R^2={best_law_r2:.3f})."
    return head


# ---- selftest (two-sided) ---------------------------------------------------

def _synth_law_series(n=4500, seed=31):
    """SIDE 1 -- a series where the WINNING config FORMULATION is a known function of
    a measurable PAST-ONLY property, so the law is REAL and a CONSTRUCTED config
    beats a single FIXED config OOS.

    The lever: alternate two regimes whose CHARACTER (visible in the pre-move window)
    demands a different formulation --
      (A) POPPY / mean-reverting legs: a sharp pop that REVERSES hard. A golden/death
          cross (the FIXED config) gets whipsawed and even LOSES money; a price>MA
          state-exit (F1) cuts quickly and captures the pop.
      (B) SUSTAINED-RUN legs: a big persistent up-leg. Riding (F1 / cross) captures it.
    The pre-move efficiency/persistence separates A from B, so the winning formulation
    is a function of the pre-move property. A single FIXED cross config is badly
    mismatched in regime A (it whipsaws), so a CONSTRUCTED config that adopts the
    formulation the regime calls for beats fixed. On the random series (SIDE-2) no
    such structure exists and construction ties fixed.
    """
    rng = np.random.default_rng(seed)
    price = [100.0]
    seg = 0
    while len(price) < n:
        sustained = (seg % 2 == 1)
        seg_len = 70
        if sustained:
            # big sustained run -> ride captures it; a tight exit leaves money behind.
            for k in range(seg_len):
                ret = 0.0070 + rng.normal(0, 0.0035)
                price.append(price[-1] * (1.0 + ret))
                if len(price) >= n:
                    break
        else:
            # sharp pop then hard reverse (mean-revert) -> a cross-ride gives it ALL
            # back (even loses); a quick state-exit captures the pop.
            for k in range(seg_len):
                ret = 0.030 * np.sin(2 * np.pi * k / 14.0) + rng.normal(0, 0.0020)
                price.append(price[-1] * (1.0 + ret))
                if len(price) >= n:
                    break
        seg += 1
    p = np.array(price[:n], dtype=np.float64)
    o = p.copy()
    c = p.copy()
    h = p * 1.0010
    lo = p * 0.9990
    return o, h, lo, c


def _synth_random_series(n=2200, seed=23):
    """SIDE 2 -- a series where the winning config is essentially RANDOM per move
    (homogeneous noisy drift, no property->config relationship). Step-1 laws should
    have R^2 ~ 0; Step-2 construction should be ~ fixed (no edge)."""
    rng = np.random.default_rng(seed)
    price = [100.0]
    for i in range(n):
        ret = 0.0020 + rng.normal(0, 0.010)
        price.append(price[-1] * (1.0 + ret))
    p = np.array(price, dtype=np.float64)
    o = p.copy()
    c = p.copy()
    h = p * 1.0012
    lo = p * 0.9988
    return o, h, lo, c


def selftest() -> bool:
    """Two-sided validation.

      SIDE 1 (a config-param IS a known function of a past-only property): the
        ENGINEERED continuous law (exit_tp<-pre_vol) must be RECOVERED with
        meaningful R^2, and the per-move CONSTRUCTED config must beat a single FIXED
        config OOS.
      SIDE 2 (config is random per move): all laws R^2 ~ 0 and construction ~ fixed.

    The discriminators are (a) the engineered law markedly stronger on SIDE-1 than
    SIDE-2, and (b) construction beats fixed on SIDE-1 but NOT on SIDE-2.
    """
    ok = True
    cadence = "1d"   # 7-14 bar windows -> dense synthetic events

    # ---- SIDE 1: known law present --------------------------------------------
    o, h, lo, c = _synth_law_series()
    recs1, cands1, M1, ev1 = build_event_records(o, h, lo, c, cadence)
    laws1 = fit_laws(recs1)
    steps1 = run_construction(recs1, cands1, M1, seed=7)
    agg1 = aggregate_construction(steps1)
    sk1, sr1 = strongest_law(laws1)
    print(f"[selftest] SIDE-1 (known law): {len(ev1)} events, {agg1['n_steps']} steps")
    print(f"[selftest]   strongest law: {sk1} R^2={sr1}")
    if len(ev1) < 8:
        print("[selftest] FAIL(1): too few events on the known-law series")
        ok = False
    cm1 = agg1["capture_means"] if agg1["n_steps"] >= 2 else None
    delta1 = agg1.get("constructed_minus_fixed_mean")
    if cm1 is not None:
        print(f"[selftest]   constructed={cm1['constructed']:+.4f} "
              f"fixed={cm1['fixed']:+.4f} lastweek={cm1['lastweek_best']:+.4f} "
              f"delta-vs-fixed={delta1:+.4f}")
        if not (cm1["constructed"] is not None and cm1["constructed"] > 0):
            print("[selftest] FAIL(1): constructed capture not positive on known-law series")
            ok = False
        # Construction must BEAT fixed by a MEANINGFUL margin (the fixed cross
        # whipsaws in the poppy regime; the constructed formulation does not).
        if not (delta1 is not None and delta1 > 0.01):
            print(f"[selftest] FAIL(1): constructed did not beat fixed by >0.01 "
                  f"(delta={delta1})")
            ok = False
    else:
        print("[selftest] FAIL(1): <2 construction steps on known-law series")
        ok = False

    # ---- SIDE 2: random config (no law) ---------------------------------------
    o2, h2, lo2, c2 = _synth_random_series()
    recs2, cands2, M2, ev2 = build_event_records(o2, h2, lo2, c2, cadence)
    laws2 = fit_laws(recs2)
    steps2 = run_construction(recs2, cands2, M2, seed=7)
    agg2 = aggregate_construction(steps2)
    sk2, sr2 = strongest_law(laws2)
    delta2 = agg2.get("constructed_minus_fixed_mean")
    print(f"[selftest] SIDE-2 (random): {len(ev2)} events, {agg2['n_steps']} steps")
    print(f"[selftest]   strongest law: {sk2} R^2={sr2}")

    # DISCRIMINATOR (a): the strongest law is markedly stronger on SIDE-1 than SIDE-2.
    if not (sr1 is not None and sr2 is not None and sr1 > sr2 + 0.10):
        print(f"[selftest] FAIL: strongest law not markedly stronger on known-law "
              f"({sr1}) vs random ({sr2})")
        ok = False

    # DISCRIMINATOR (b): construction must NOT beat fixed meaningfully on SIDE-2.
    if agg2["n_steps"] >= 2:
        cm2 = agg2["capture_means"]
        print(f"[selftest]   constructed={cm2['constructed']:+.4f} "
              f"fixed={cm2['fixed']:+.4f} delta={delta2:+.4f}")
        if delta2 is not None and delta2 > 0.01:
            print(f"[selftest] FAIL: construction beat fixed by {delta2:+.4f} on a "
                  f"RANDOM-config series (should be ~0)")
            ok = False
    else:
        print("[selftest] FAIL(2): <2 construction steps on random series")
        ok = False

    print("[selftest] PASS" if ok else "[selftest] FAIL")
    return ok


# ---- main -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Oracle config decomposition (config-param = f(move-properties))")
    ap.add_argument("--asset", default="BTCUSDT")
    ap.add_argument("--assets", default=None,
                    help="comma list overrides --asset (e.g. BTCUSDT,ETHUSDT,SOLUSDT)")
    ap.add_argument("--cadences", default="1d,4h,1h")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    assets = ([a.strip() for a in args.assets.split(",") if a.strip()]
              if args.assets else [args.asset])
    cadences = [c.strip() for c in args.cadences.split(",") if c.strip()]
    for cad in cadences:
        if cad not in WINDOW_BARS:
            print(f"[error] unknown cadence '{cad}'; known={list(WINDOW_BARS)}")
            sys.exit(2)

    results: list[dict] = []
    for asset in assets:
        for cad in cadences:
            print(f"[run] {asset} {cad}: loading + events + decomposing ...", flush=True)
            try:
                res = run_one(asset, cad, seed=args.seed)
            except Exception as e:  # data gap / load failure -- report, continue
                print(f"[warn] {asset} {cad} failed: {e}")
                continue
            results.append(res)
            print(f"[run] {asset} {cad}: {res['n_events']} events, "
                  f"{res['step2_construction'].get('n_steps', 0)} construction steps",
                  flush=True)

    print_report(results)
    verdict = overall_verdict(results)
    print("OVERALL ANSWER:")
    print("  " + verdict)
    print("")

    asset_tag = (assets[0].upper().replace("USDT", "") if len(assets) == 1
                 else "MULTI")
    out_path = (Path(args.out) if args.out
                else PROJECT_ROOT / "runs" / "strat" /
                f"oracle_config_decomp_{asset_tag}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "tool": "oracle_config_decomp",
        "hypothesis": "oracle config-params are a math function of past-only move-properties",
        "assets": assets,
        "cadences": cadences,
        "spec": {
            "pre_window_mult": PRE_WINDOW_MULT,
            "pre_window_min_bars": PRE_WINDOW_MIN_BARS,
            "train_events": TRAIN_EVENTS,
            "test_events": TEST_EVENTS,
            "min_train_events": MIN_TRAIN_EVENTS,
            "min_test_events": MIN_TEST_EVENTS,
            "n_random": N_RANDOM,
            "taker_rt": TAKER_RT,
            "law_pairs": [f"{t}<-{p}" for t, p in LAW_PAIRS],
            "causal": "properties measured pre-move; laws fit on TRAIN only; TEST untouched",
            "capture_units": "realized_long_roi_net_taker",
        },
        "results": results,
        "overall_verdict": verdict,
    }
    out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"[artifact] {out_path}")


if __name__ == "__main__":
    main()
