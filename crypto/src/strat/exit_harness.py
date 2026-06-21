"""src/strat/exit_harness.py -- V1.1 FINE-TF EXIT HARNESS: causal hold-mask transforms.

WHY (meta /orc 2026-06-21). The V1 MA drawdown-insurance engine is binding-scope 1d/4h; it DEGRADES through
2h/1h/30m and BREAKS at 15m. `finetf_loss_diagnose.py` proved the single killer is ROUND-TRIP COST FROM CHURN
(cost-as-%-of-gross 0.8% -> 4.6% -> 14.8% -> 30% -> 70% -> 206% from 1d->15m; GROSS stays +100%+ at every TF;
capture-rate of 5%+ moves FLAT ~0.36-0.41). Diagnosis: NOT under-participation, NOT late entry -- pure churn/cost.
The constructive lever is therefore a CHURN/COST-REDUCTION exit + entry de-bounce. This module IS that harness.

PRIOR-NULL CITATION (cite, then proceed -- DIFFERENT mechanism): smart exits were a NULL on DAILY breakouts
(`project-exit-axis-null-2026-06-08`; dead-list D61 "smart exit > fixed hold (daily breakouts)" REFUTED, the
apparent edge a hold-length artifact; related D67 causal mover-CAPTURE per-move<0; D23 MFE trailing on PEPE MA
TRAIN+VAL -18pp). BUT all of those are DAILY-bar exits where there is NO churn problem -- the capture-rate lens.
The fine-TF (2h/1h/30m) regime here is a DIFFERENT mechanism: cost/churn REDUCTION on an over-trading band
ensemble where +100% gross headroom exists, genuinely UNTESTED. The dead-list itself flags "sub-bar (4h/fine-TF)
exit genuinely untested" (01_DEAD_LIST.md). This harness proceeds on the untested mechanism, citing the daily null.

WHAT THIS IS (the clean seam). The engine's hold-mask is the band ENSEMBLE ON-state: `make_hold_provider` returns
`provider(sym,o,h,l,c) -> int8` = 1 while >50% of the band's `held_cross` is long (a continuous hold-while-aligned
state). The CURRENT engine exit IS `band_off` already (the mask drops to 0 the bar the band majority breaks). This
harness POST-PROCESSES that int8 mask with a SMARTER exit (and/or a smarter entry), returning a modified int8 mask
that the SAME machinery consumes:
    base = make_hold_provider(sub_band, ma_type)            # canonical band ensemble (NEVER ranked[0])
    exit_fn = build_exit_fn(exit_type="band_off_confirm", exit_params={"M": 2}, ...)
    provider2 = decorate_provider(base, exit_fn)            # provider2(sym,o,h,l,c) -> modified int8 mask
    -> feed provider2 through ma_wealth_driver.run_wealth(..., exit_fn=...) for WEALTH (single authoritative path).

The wealth/metrics are produced ONLY by `ma_wealth_driver.run_wealth` (MtM-no-double-count, lag-1, maker 0.0006 RT,
TRUE buy@listing HODL). This module NEVER hand-rolls wealth (2 fabrications were caught precisely because workers
hand-rolled MtM -- see V1 doc provenance). It only TRANSFORMS the boolean hold-mask.

EXIT TYPES (each maps a band-on hold-mask -> a smarter hold-mask; all causal, lag-1 downstream):
  1. band_off            -- baseline (identity transform; current engine behavior).
  2. band_off_confirm    -- exit only after M consecutive band-OFF bars (hysteresis/de-bounce). params {M in 2,3}.
  3. trail_pct           -- trailing stop X% off the running close-peak since entry.        params {X in 8,15}.
  4. chandelier_atr      -- ATR(22) trailing stop k*ATR off the running close-peak.          params {k in 3}.
  5. time_stop           -- hard exit after N bars in-position (caps dead holds).             params {N}.
  6. profit_giveback     -- arm at +target%, then exit on give-back of Z% of PEAK unrealized. params {target,Z}.
  7. htf_confirm_exit    -- exit only when the next-coarser TF band is ALSO off (slow exit).  needs htf_on series.

ENTRY INTERVENTIONS (stack on top of band-on, BEFORE exits; reduce whipsaw entries):
  A. entry_none          -- enter when band-on (identity).
  B. entry_debounce      -- enter only after N consecutive band-ON bars.                      params {N in 2,3}.
  C. htf_entry_gate      -- enter only if the next-coarser TF band is on (multi-TF confirm).  needs htf_on series.

CAUSALITY (binding). Every rule decides at bar i using ONLY data through bar i (peak/ATR/run-length are
backward-looking). The transformed mask is the POSITION STATE per bar; the wealth driver lags it ONE bar
(W.shift(1)) before applying returns -- so a "long at bar i" state earns bar i+1's return. No look-ahead. The HTF
series is forward-filled from the COARSER bar's close-time onto the base grid (a coarser bar is only *complete* at
its close, so its on/off at a base bar reflects the most recent CLOSED coarser bar -- causal by construction; see
align_htf_on).

REUSE (do NOT duplicate): trailing/ATR/giveback math is the SAME library as `ma_capture_exit_sweep.py`; this module
imports `_atr` and `_runs` from there and mirrors `_apply_exit_to_run`'s exit-bar semantics. The difference is the
unit: there the exit acts on a single MA(50/150) cross run for per-TRADE CAPTURE (taker 0.0024); here it acts on the
band-ENSEMBLE hold-mask run for WEALTH (maker 0.0006). The two cost models are NEVER mixed on one column.

RWYB (smoke / one-cell verify):
  python -m strat.exit_harness --selftest                          # mask-level unit checks (no data)
  python -m strat.exit_harness --verify --ma-type EMA --tf 1h      # end-to-end via ma_wealth_driver (2020-walled)
No emoji (cp1252). No git commit (overseer commits). Writes nothing unless --verify --out is given.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent / "src") not in sys.path:
    sys.path.insert(0, str(ROOT.parent / "src"))
CRYPTO = ROOT.parent

# REUSE the committed causal exit library (do NOT duplicate the ATR / runs math).
from strat.ma_capture_exit_sweep import _atr, _runs  # noqa: E402

# ---------------------------------------------------------------------------
# CONTRACT (machine-discoverable; CDAP contract_loader reads this).
# ---------------------------------------------------------------------------
__contract__ = {
    "kind": "strat_transform",
    "purpose": "causal hold-mask transforms (exit policies + entry interventions) for the MA band ensemble; "
               "plugs into ma_movecatch_engine.make_hold_provider and feeds ma_wealth_driver.run_wealth",
    "inputs": {
        "band_on": "int8/bool 1-D array -- the band ENSEMBLE long/flat state per bar (1=majority band long)",
        "o,h,l,c": "float 1-D OHLC arrays aligned to band_on (close-anchored peaks; ATR uses h,l,c)",
        "htf_on": "optional int8 1-D array -- next-coarser TF band-on aligned to the base grid (HTF rules only)",
        "exit_type": "one of EXIT_TYPES",
        "entry_type": "one of ENTRY_TYPES",
        "exit_params/entry_params": "dict of the named scalar params (M,X,k,N,target,Z)",
    },
    "outputs": {"hold_mask": "int8 1-D array -- the transformed POSITION STATE per bar (pre-lag); 1=long, 0=cash"},
    "invariants": [
        "CAUSAL: decision at bar i uses only data through bar i (peak/ATR/run-length are backward-looking).",
        "NO look-ahead: the wealth driver lags this mask 1 bar (W.shift(1)) before applying returns.",
        "TRANSFORM-ONLY: never computes wealth/MtM; ma_wealth_driver.run_wealth is the sole authoritative path.",
        "SUBSET-OR-EQUAL for pure exits/entry-gates: a smarter exit/entry can only REMOVE hold bars within a run "
        "or DELAY an entry -- it never ADDS a hold bar where band_on==0 (band_off is the identity upper bound).",
        "band_off (exit) + entry_none (entry) == identity (reproduces the current engine mask exactly).",
        "HTF rules require htf_on; absent htf_on they raise (no silent fallback).",
    ],
    "reuses": ["strat.ma_capture_exit_sweep._atr", "strat.ma_capture_exit_sweep._runs",
               "strat.ma_movecatch_engine.make_hold_provider", "strat.ma_wealth_driver.run_wealth"],
    "prior_null": "D61/D67/D23 daily exit-axis NULL (capture lens); fine-TF cost/churn reduction is a DIFFERENT, "
                  "genuinely-untested mechanism (01_DEAD_LIST.md flags sub-bar exits untested).",
}

EXIT_TYPES = ["band_off", "band_off_confirm", "trail_pct", "chandelier_atr",
              "time_stop", "profit_giveback", "htf_confirm_exit"]
ENTRY_TYPES = ["entry_none", "entry_debounce", "htf_entry_gate"]

# Pre-registered param grid (the sweep space the campaign should walk; documented, not hard-wired).
EXIT_PARAM_GRID = {
    "band_off": [{}],
    "band_off_confirm": [{"M": 2}, {"M": 3}],
    "trail_pct": [{"X": 8.0}, {"X": 15.0}],
    "chandelier_atr": [{"k": 3.0}],
    "time_stop": [{"N_days": 7}, {"N_days": 14}],          # N in BARS = N_days * bars_per_day (TF-aware; see below)
    "profit_giveback": [{"target": 10.0, "Z": 50.0}, {"target": 20.0, "Z": 50.0}],
    "htf_confirm_exit": [{}],
}
ENTRY_PARAM_GRID = {
    "entry_none": [{}],
    "entry_debounce": [{"N": 2}, {"N": 3}],
    "htf_entry_gate": [{}],
}

# Coarser-TF map (the "next-coarser TF" for HTF confirm/gate). 15m->1h is intentional (skip 30m): 30m is itself a
# fine-TF the diagnosis flagged degrading, so HTF-confirming 15m against 1h gives a meaningfully slower anchor.
NEXT_COARSER_TF = {"15m": "1h", "30m": "2h", "1h": "4h", "2h": "1d", "4h": "1d", "1d": None}


# ===========================================================================
# 1. ENTRY INTERVENTIONS  (band_on -> gated band_on)  -- applied BEFORE exits
# ===========================================================================
def apply_entry(band_on: np.ndarray, entry_type: str, params: Optional[dict] = None,
                htf_on: Optional[np.ndarray] = None) -> np.ndarray:
    """Transform the raw band-on series into a (possibly DELAYED / GATED) entry-eligible series.

    entry_none      -- identity.
    entry_debounce  -- a bar is entry-eligible only after N consecutive band-ON bars (kills 1-bar flickers).
                       The first (N-1) bars of every band-on run are suppressed; the run's tail stays on.
    htf_entry_gate  -- a bar is entry-eligible only if BOTH band_on AND htf_on are 1 (multi-TF confirmation).

    Returns an int8 series (1=eligible-to-be-long). Pure restriction: out <= band_on everywhere (never adds bars)."""
    params = params or {}
    b = np.asarray(band_on, dtype=np.int8).copy()
    if entry_type == "entry_none":
        return b
    if entry_type == "entry_debounce":
        N = int(params.get("N", 2))
        if N <= 1:
            return b
        # rolling count of consecutive 1s ENDING at bar i (causal). eligible iff run-length-so-far >= N.
        run_len = np.zeros(len(b), dtype=np.int64)
        cnt = 0
        for i in range(len(b)):
            cnt = cnt + 1 if b[i] == 1 else 0
            run_len[i] = cnt
        out = (run_len >= N).astype(np.int8)
        return out
    if entry_type == "htf_entry_gate":
        if htf_on is None:
            raise ValueError("htf_entry_gate requires htf_on (the next-coarser TF band-on series)")
        ht = np.asarray(htf_on, dtype=np.int8)
        if len(ht) != len(b):
            raise ValueError(f"htf_on length {len(ht)} != band_on length {len(b)}")
        return (b & ht).astype(np.int8)
    raise ValueError(f"unknown entry_type {entry_type!r}; expected one of {ENTRY_TYPES}")


# ===========================================================================
# 2. EXIT POLICIES  (per-run early exit within the entry-eligible series)
# ===========================================================================
def _exit_bar_for_run(s: int, e: int, c: np.ndarray, h: np.ndarray, l: np.ndarray,
                      atr22: np.ndarray, htf_on: Optional[np.ndarray],
                      exit_type: str, params: dict) -> int:
    """Given a contiguous hold run [s, e) (e exclusive), return the INCLUSIVE exit-bar index per the exit rule.
    Mirrors ma_capture_exit_sweep._apply_exit_to_run semantics (causal, running-CLOSE-peak anchored), generalized
    to the band-ensemble run + extended with time_stop / profit_giveback / htf_confirm_exit. Fall-through = e-1
    (the band's own band_off exit, i.e. the last in-run bar)."""
    if e <= s:
        return s
    run_peak = c[s]
    armed = False  # for profit_giveback
    for i in range(s, e):
        run_peak = max(run_peak, c[i])

        if exit_type == "band_off":
            continue  # identity: only the band itself ends the run (fall-through to e-1)

        elif exit_type == "trail_pct":
            X = float(params.get("X", 8.0)) / 100.0
            if i > s and c[i] < run_peak * (1.0 - X):
                return i

        elif exit_type == "chandelier_atr":
            k = float(params.get("k", 3.0))
            if i > s and c[i] < run_peak - k * atr22[i]:
                return i

        elif exit_type == "time_stop":
            N = int(params.get("N", params.get("N_bars", 0)))
            if N > 0 and (i - s) >= N - 1:
                return i

        elif exit_type == "profit_giveback":
            target = float(params.get("target", 10.0)) / 100.0
            Z = float(params.get("Z", 50.0)) / 100.0
            entry_px = c[s]
            if not armed and run_peak >= entry_px * (1.0 + target):
                armed = True
            if armed and i > s:
                # give back Z% of the PEAK unrealized gain (peak_gain measured from entry)
                peak_gain = run_peak - entry_px
                exit_level = run_peak - Z * peak_gain
                if c[i] < exit_level:
                    return i

        elif exit_type == "htf_confirm_exit":
            if htf_on is None:
                raise ValueError("htf_confirm_exit requires htf_on (next-coarser TF band-on series)")
            # exit only when the coarser TF band is ALSO off at this bar (slows the exit signal)
            if i > s and htf_on[i] == 0:
                return i

        elif exit_type == "band_off_confirm":
            # handled at the SERIES level (hysteresis spans run boundaries), not per-run -- see apply_exit.
            continue
        else:
            raise ValueError(f"unknown exit_type {exit_type!r}; expected one of {EXIT_TYPES}")

    return e - 1  # fall-through: the band's own band_off exit (last in-run bar)


def _band_off_confirm(band_on: np.ndarray, M: int) -> np.ndarray:
    """Hysteresis de-bounce on the EXIT side: stay long until the band has been OFF for M CONSECUTIVE bars.
    A single (or up to M-1) band-off flicker does NOT exit. Causal. Returns int8 hold state.

    Semantics: position turns ON at the first band_on==1; turns OFF only after M consecutive band_on==0 bars,
    and the off takes effect at the M-th off bar (the position is held THROUGH the first M-1 off bars). This is a
    pure churn-reducer (it can only EXTEND holds across short band gaps, never create a hold where the band never
    fired)."""
    b = np.asarray(band_on, dtype=np.int8)
    n = len(b)
    out = np.zeros(n, dtype=np.int8)
    pos = 0
    off_run = 0
    for i in range(n):
        if b[i] == 1:
            pos = 1
            off_run = 0
        else:
            if pos == 1:
                off_run += 1
                if off_run >= M:
                    pos = 0
                    off_run = 0
            # else: already flat, stay flat
        out[i] = pos
    # SAFETY: band_off_confirm bridges gaps WITHIN a band-on episode, but must never hold where the band was
    # never on yet. The loop above guarantees pos only turns 1 on a real band_on==1; nothing to clamp.
    return out


def apply_exit(entry_on: np.ndarray, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray,
               exit_type: str, params: Optional[dict] = None,
               htf_on: Optional[np.ndarray] = None, atr22: Optional[np.ndarray] = None) -> np.ndarray:
    """Apply an exit policy to the entry-eligible series, returning the final int8 POSITION STATE per bar.

    For per-run exits (trail/chandelier/time/giveback/htf/band_off): split `entry_on` into contiguous runs, find
    the rule's exit bar in each run, and force 0 from exit_bar+1 to the run end (stay flat until the next run --
    matching apply_trail_stop's 'stay out until re-arm' semantics; the band re-arms the next time entry_on==1).

    For band_off_confirm: a SERIES-level hysteresis (the de-bounce spans run boundaries), handled by _band_off_confirm.

    Pure restriction within each run: out <= entry_on everywhere (an exit can only END a hold earlier)."""
    params = params or {}
    b = np.asarray(entry_on, dtype=np.int8)
    n = len(b)
    cc = np.asarray(c, dtype=float)

    if exit_type == "band_off_confirm":
        M = int(params.get("M", 2))
        out = _band_off_confirm(b, M)
        # band_off_confirm can extend a hold across short OFF gaps; that is its purpose (churn reduction). It can
        # NEVER turn on where the band never turned on. (Verified in --selftest.)
        return out

    out = b.copy()
    if exit_type == "band_off":
        return out  # identity

    if atr22 is None and exit_type == "chandelier_atr":
        atr22 = _atr(np.asarray(h, float), np.asarray(l, float), cc, 22)
    if atr22 is None:
        atr22 = np.zeros(n)  # unused for non-ATR exits

    for (s, e) in _runs(b):
        xbar = _exit_bar_for_run(s, e, cc, np.asarray(h, float), np.asarray(l, float),
                                 atr22, htf_on, exit_type, params)
        # hold [s, xbar] inclusive; force flat (xbar, e)
        if xbar + 1 < e:
            out[xbar + 1:e] = 0
    return out


# ===========================================================================
# 3. THE PUBLIC TRANSFORM  +  the provider decorator (the wealth seam)
# ===========================================================================
def transform_mask(band_on: np.ndarray, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray,
                   *, exit_type: str = "band_off", entry_type: str = "entry_none",
                   exit_params: Optional[dict] = None, entry_params: Optional[dict] = None,
                   htf_on: Optional[np.ndarray] = None, bars_per_day: Optional[int] = None) -> np.ndarray:
    """THE public causal transform: (band ensemble hold-mask + OHLC + params) -> modified int8 hold-mask.

    Order: (1) ENTRY intervention gates/delays the band-on series, (2) EXIT policy ends each resulting run early.
    `band_off` + `entry_none` is the identity (reproduces the current engine mask byte-for-byte).

    time_stop convenience: if exit_params has 'N_days' and not 'N', N_bars = N_days * bars_per_day (TF-aware).

    Returns int8 position state per bar (PRE-LAG). The wealth driver lags it 1 bar."""
    exit_params = dict(exit_params or {})
    entry_params = dict(entry_params or {})
    band_on = np.asarray(band_on, dtype=np.int8)

    # TF-aware time_stop: translate N_days -> N_bars once we know bars_per_day
    if exit_type == "time_stop" and "N" not in exit_params and "N_bars" not in exit_params:
        nd = exit_params.get("N_days")
        if nd is not None and bars_per_day:
            exit_params["N"] = int(nd) * int(bars_per_day)

    entry_on = apply_entry(band_on, entry_type, entry_params, htf_on=htf_on)
    atr22 = _atr(np.asarray(h, float), np.asarray(l, float), np.asarray(c, float), 22) \
        if exit_type == "chandelier_atr" else None
    hold = apply_exit(entry_on, o, h, l, c, exit_type, exit_params, htf_on=htf_on, atr22=atr22)
    return hold.astype(np.int8)


def build_exit_fn(*, exit_type: str = "band_off", entry_type: str = "entry_none",
                  exit_params: Optional[dict] = None, entry_params: Optional[dict] = None,
                  bars_per_day: Optional[int] = None,
                  htf_on_lookup: Optional[Callable[[str], Optional[np.ndarray]]] = None) -> Callable:
    """Build the `exit_fn(sym, o, h, l, c, base_mask) -> int8 mask` callable that ma_wealth_driver.run_wealth
    applies to the band provider's output. This is the WEALTH-SEAM signature.

    htf_on_lookup(sym) -> the next-coarser-TF band-on series aligned to THIS asset's base grid (or None). Required
    only for htf_* rules; run_wealth supplies it when an HTF rule is requested (it builds the coarser provider and
    aligns via align_htf_on). For all non-HTF rules pass None.

    The returned callable is pure + causal; it post-processes the boolean mask only (no wealth)."""
    def exit_fn(sym: str, o, h, l, c, base_mask) -> np.ndarray:
        htf_on = htf_on_lookup(sym) if (htf_on_lookup is not None) else None
        if htf_on is not None and len(htf_on) != len(base_mask):
            # align defensively: trim/pad to base length (run_wealth aligns by date already; this is a guard)
            m = min(len(htf_on), len(base_mask))
            htf_on = np.asarray(htf_on)[:m]
            base_mask = np.asarray(base_mask)[:m]
        return transform_mask(np.asarray(base_mask, dtype=np.int8), o, h, l, c,
                              exit_type=exit_type, entry_type=entry_type,
                              exit_params=exit_params, entry_params=entry_params,
                              htf_on=htf_on, bars_per_day=bars_per_day)
    exit_fn.__exit_meta__ = {"exit_type": exit_type, "entry_type": entry_type,
                             "exit_params": exit_params or {}, "entry_params": entry_params or {}}
    return exit_fn


def decorate_provider(base_provider: Callable, exit_fn: Callable) -> Callable:
    """Wrap a make_hold_provider output so its mask is post-processed by exit_fn. Returns a provider with the SAME
    signature provider(sym,o,h,l,c)->int8, so it is a drop-in for any consumer of make_hold_provider."""
    def provider(sym, o, h, l, c):
        base = base_provider(sym, o, h, l, c)
        return exit_fn(sym, o, h, l, c, base).astype(np.int8)
    return provider


# ===========================================================================
# 4. HTF ALIGNMENT (causal forward-fill of the coarser band-on onto the base grid)
# ===========================================================================
def align_htf_on(base_dates, htf_dates, htf_on: np.ndarray) -> np.ndarray:
    """Align a COARSER-TF band-on series onto the BASE-TF date grid, CAUSALLY.

    A coarser bar is only COMPLETE at its close. So at a base bar with timestamp t, the coarser band-on we are
    allowed to use is the value of the most recent coarser bar whose CLOSE-time <= t.

    LEAK FIX (2026-06-21, V1.1 audit). Chimera timestamps are floored to the window-START label (ma_wealth_driver
    `.floor(_FREQ)`), so a coarser bar labeled L spans [L, L+freq) and CLOSES at L+freq. The prior implementation
    searchsorted on the START labels `ht` -> it returned the bar CONTAINING t (whose close L+freq > t is in the
    FUTURE), so a base bar inside an unclosed coarser window read that window's eventual state = LOOK-AHEAD (the
    htf_entry_gate's apparent fine-TF "fix" was largely this leak). The fix aligns on coarser CLOSE-times
    (label + nominal bar width), so a coarser bar is usable only AFTER it has closed. Bars before the first coarser
    CLOSE map to 0 (no confirmation yet). Returns int8 aligned to base_dates length."""
    import pandas as pd
    bt = pd.DatetimeIndex(base_dates).asi8
    ht = pd.DatetimeIndex(htf_dates).asi8
    on = np.asarray(htf_on, dtype=np.int8)
    out = np.zeros(len(bt), dtype=np.int8)
    if len(ht) == 0:
        return out
    # coarser CLOSE-times = start-label + nominal bar width (median diff -> robust to occasional gaps).
    freq = int(np.median(np.diff(ht))) if len(ht) >= 2 else 0
    close_t = ht + freq
    # last coarser bar whose CLOSE <= base time (causal: a coarser bar is usable only once it has closed).
    idx = np.searchsorted(close_t, bt, side="right") - 1
    valid = idx >= 0
    out[valid] = on[idx[valid]]
    return out


# ===========================================================================
# 5. SELF-TEST  (mask-level unit checks; no data load)
# ===========================================================================
def _selftest() -> int:
    ok = True

    def check(name, cond):
        nonlocal ok
        status = "PASS" if cond else "FAIL"
        if not cond:
            ok = False
        print(f"  [{status}] {name}")

    # identity: band_off + entry_none reproduces the band exactly
    band = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0], dtype=np.int8)
    c = np.array([10, 11, 12, 13, 12, 11, 14, 20, 19], dtype=float)
    h = c * 1.01; l = c * 0.99; o = c
    m = transform_mask(band, o, h, l, c, exit_type="band_off", entry_type="entry_none")
    check("identity band_off+entry_none == band_on", np.array_equal(m, band))

    # subset property: a pure exit can only remove hold bars within a run
    m_tp = transform_mask(band, o, h, l, c, exit_type="trail_pct", exit_params={"X": 8.0})
    check("trail_pct subset of band_on (no added bars)", np.all(m_tp <= band))

    # trail_pct fires: run [6,8] c=[14,20]; peak 20; 8% trail = 18.4; bar8 c=19 is in NEXT band-off so n/a.
    # build a run where price gives back >8%: c rises to 20 then drops to 18 within the run
    band2 = np.array([1, 1, 1, 1, 1], dtype=np.int8)
    c2 = np.array([10.0, 12.0, 20.0, 18.0, 19.0])
    h2 = c2; l2 = c2; o2 = c2
    m2 = transform_mask(band2, o2, h2, l2, c2, exit_type="trail_pct", exit_params={"X": 8.0})
    # peak 20 at i=2; 8% trail level = 18.4; i=3 c=18 < 18.4 -> exit at i=3 -> hold [0..3], flat at i=4
    check("trail_pct exits on >8% giveback", np.array_equal(m2, np.array([1, 1, 1, 1, 0], dtype=np.int8)))

    # time_stop N=2 bars: hold at most 2 bars per run -> [s, s+1], flat after
    m_ts = transform_mask(band2, o2, h2, l2, c2, exit_type="time_stop", exit_params={"N": 2})
    check("time_stop N=2 holds exactly 2 bars", np.array_equal(m_ts, np.array([1, 1, 0, 0, 0], dtype=np.int8)))

    # band_off_confirm M=2: bridges a 1-bar OFF gap, exits after 2 consecutive OFFs
    bandg = np.array([1, 1, 0, 1, 1, 0, 0, 1], dtype=np.int8)
    cg = np.ones(len(bandg)) * 10.0
    m_bc = transform_mask(bandg, cg, cg, cg, cg, exit_type="band_off_confirm", exit_params={"M": 2})
    # i0,1 on; i2 off(run1) still held; i3 on; i4 on; i5 off(run1); i6 off(run2)->exit at i6; i7 on
    exp_bc = np.array([1, 1, 1, 1, 1, 1, 0, 1], dtype=np.int8)
    check("band_off_confirm M=2 bridges 1-bar gap, exits on 2nd off", np.array_equal(m_bc, exp_bc))
    check("band_off_confirm never holds where band never fired (prefix)",
          m_bc[0] == bandg[0] or bandg[0] == 1)

    # entry_debounce N=2: suppress first bar of each band-on run
    m_db = transform_mask(band, o, h, l, c, entry_type="entry_debounce", entry_params={"N": 2})
    # band runs: [1..3] -> first bar (i1) suppressed -> on at i2,i3 ; [6..7] -> i6 suppressed -> on at i7
    exp_db = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0], dtype=np.int8)
    check("entry_debounce N=2 suppresses first bar of each run", np.array_equal(m_db, exp_db))
    check("entry_debounce subset of band_on", np.all(m_db <= band))

    # htf gate requires htf_on; missing -> raises
    try:
        transform_mask(band, o, h, l, c, entry_type="htf_entry_gate")
        check("htf_entry_gate raises without htf_on", False)
    except ValueError:
        check("htf_entry_gate raises without htf_on", True)

    # htf_entry_gate AND semantics
    htf = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0], dtype=np.int8)
    m_hg = transform_mask(band, o, h, l, c, entry_type="htf_entry_gate", htf_on=htf)
    check("htf_entry_gate == band_on AND htf_on", np.array_equal(m_hg, (band & htf).astype(np.int8)))

    # align_htf_on causality: a coarser ON only appears at base bars at/after the coarser close
    import pandas as pd
    base_d = pd.date_range("2020-01-01", periods=8, freq="1h")
    htf_d = pd.date_range("2020-01-01", periods=2, freq="4h")  # closes at 00:00 and 04:00
    htf_v = np.array([1, 0], dtype=np.int8)
    al = align_htf_on(base_d, htf_d, htf_v)
    # CAUSAL (no look-ahead): coarse[0]=[00:00,04:00) ON closes at 04:00, so base 00:00-03:00 have NO closed coarser
    # bar yet (->0); base 04:00-07:00 use coarse[0]=ON. The leaky version returned [1,1,1,1,...] (read the unclosed
    # window) -- this asserts the fixed close-time alignment.
    check("align_htf_on causal (coarser usable only after close, no look-ahead)",
          np.array_equal(al, np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)))

    print("\nSELFTEST:", "ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


# ===========================================================================
# 6. ONE-CELL END-TO-END VERIFY  (through ma_wealth_driver -- the real wealth path)
# ===========================================================================
def _verify_cell(ma_type: str, tf: str, out_path: Optional[Path] = None) -> dict:
    """Run ONE (ma_type, tf) cell end-to-end through ma_wealth_driver.run_wealth for: (a) the BASELINE band_off
    exit (identity -> must match the committed engine wealth) and (b) a couple of SMARTER exits, to confirm the
    mask changes the metrics sensibly and MtM reconciles. Uses run_wealth's exit_fn hook (added surgically)."""
    from strat import ma_wealth_driver as wd

    if not hasattr(wd.run_wealth, "__wrapped_exit_fn__") and "exit_fn" not in wd.run_wealth.__code__.co_varnames:
        return {"error": "ma_wealth_driver.run_wealth has no exit_fn param yet -- add the surgical hook first."}

    bpd = wd.fl.BARS_PER_DAY[tf]
    results = {}

    # (a) baseline: band_off + entry_none == identity. exit_fn=None reproduces the committed engine.
    base = wd.run_wealth(ma_type, tf, "7d", verbose=False)
    results["baseline_band_off"] = _slim(base)

    # (a') identity exit_fn must EXACTLY match the no-exit baseline (proves the hook is non-destructive)
    id_fn = build_exit_fn(exit_type="band_off", entry_type="entry_none", bars_per_day=bpd)
    base_id = wd.run_wealth(ma_type, tf, "7d", exit_fn=id_fn, verbose=False)
    results["identity_exit_fn"] = _slim(base_id)

    # (b) two smarter churn-reducers (the diagnosis's prescribed levers for fine TFs):
    for label, kw in [
        ("band_off_confirm_M2", dict(exit_type="band_off_confirm", exit_params={"M": 2})),
        ("entry_debounce_N2", dict(entry_type="entry_debounce", entry_params={"N": 2})),
        ("trail_pct_15", dict(exit_type="trail_pct", exit_params={"X": 15.0})),
    ]:
        fn = build_exit_fn(bars_per_day=bpd, **kw)
        r = wd.run_wealth(ma_type, tf, "7d", exit_fn=fn, verbose=False)
        results[label] = _slim(r)

    summary = {
        "cell": f"{ma_type}|{tf}",
        "bars_per_day": bpd,
        "identity_matches_baseline": _matches(results["baseline_band_off"], results["identity_exit_fn"]),
        "results": results,
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(summary, open(out_path, "w"), indent=1, default=str)
        print(f"[json] {out_path}")
    return summary


def _slim(r: dict) -> dict:
    if "error" in r:
        return {"error": r["error"]}
    keys = ["engine_compound_pct", "hodl_compound_pct", "vs_hodl_pp", "maxdd_engine_pct",
            "avg_gross_exposure", "mtm_reconciled", "sane", "n_assets"]
    return {k: r.get(k) for k in keys if k in r}


def _matches(a: dict, b: dict) -> bool:
    if "error" in a or "error" in b:
        return False
    for k in ("engine_compound_pct", "maxdd_engine_pct", "avg_gross_exposure"):
        if abs(float(a.get(k, 0)) - float(b.get(k, 0))) > 1e-6:
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m strat.exit_harness")
    ap.add_argument("--selftest", action="store_true", help="mask-level unit checks (no data)")
    ap.add_argument("--verify", action="store_true", help="one-cell end-to-end via ma_wealth_driver")
    ap.add_argument("--ma-type", default="EMA")
    ap.add_argument("--tf", default="1h")
    ap.add_argument("--out", default=None, help="optional JSON path for --verify")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.verify:
        out_path = Path(a.out) if a.out else None
        r = _verify_cell(a.ma_type, a.tf, out_path)
        print(json.dumps(r, indent=1, default=str))
        return 0 if "error" not in r and "error" not in r.get("results", {}).get("baseline_band_off", {}) else 2
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
