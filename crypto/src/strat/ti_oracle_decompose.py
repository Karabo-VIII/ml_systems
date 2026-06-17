"""TI-ORACLE DECOMPOSITION across FORMULATIONS x MA-TYPES.

The user's framing: "the oracle is not one answer, but a multitude of answers."

The cross-only anchor (ti_oracle_anchor.py) computes the TI-oracle as the max
capture over an SMA/EMA golden/death-cross grid. This tool DECOMPOSES the TI-oracle
into a richer space:

  FORMULATIONS (how the indicator generates the trade):
    F1 PRICE>MA           : long while close > MA(len); exit close < MA. single-MA state.
    F2 CROSS+CROSS-EXIT   : long on fast>slow golden cross; exit on death cross.
    F3 CROSS+MECH-EXIT    : long on golden cross; exit via a mechanical rule, SWEEP
                            varying exits {TP 3/5/8%, SL 3/5%, time-stop=window-end,
                            ATR-trailing 3x}.
    F4 STACK (P>fast>slow): long when close>fast>slow aligned; exit when stack breaks.
    F5 PRICE>MA+MECH-EXIT : F1 entry (close>MA); mechanical exits as in F3.

  MA TYPES (decompose the MA itself):
    SMA, EMA, WMA, HMA (Hull), DEMA.  (pandas_ta if available, else local impl.)

  GRID (tractable):
    fast in {5,10,20,50}, slow in {20,50,100,200} (fast<slow);
    single-MA len in {10,20,50,100}.

For each price-oracle move-event we compute capture (= realized long ROI /
price-oracle move) for EACH candidate, and report which FORMULATION and which
MA-TYPE captures best.

This is an ANCHOR / DECOMPOSITION -- NO pass/fail or null verdict. We measure and
SHOW the multitude.

DISCIPLINE:
  - Reuses ti_oracle_anchor.find_price_oracle_events READ-ONLY so the price-oracle
    events are IDENTICAL to the anchor.
  - Every candidate signal is CAUSAL within the event window: MA warmup may look
    back BEFORE the window (the MA at bar t uses only bars <= t); next-bar-OPEN
    fills; net taker 0.24% RT.
  - Config CHOICE (which formulation/type/params won) is HINDSIGHT by design --
    that IS the oracle.
  - cp1252-safe (no emoji).

Usage:
    python src/strat/ti_oracle_decompose.py --asset BTCUSDT --cadences 1d,4h,1h
    python src/strat/ti_oracle_decompose.py --selftest

__contract__ = {
    "kind": "research_anchor",
    "inputs": ["chimera OHLC via ChimeraLoader",
               "ti_oracle_anchor.find_price_oracle_events (read-only reuse)"],
    "outputs": ["runs/strat/ti_oracle_decompose_<ASSET>.json", "stdout table"],
    "invariants": [
        "price-oracle events identical to ti_oracle_anchor (same detector)",
        "MA signals causal within window (warmup lookback only)",
        "next-bar-open fills",
        "config/formulation/type CHOICE is hindsight by design (the oracle)",
        "no pass/fail or null verdict -- this is a DECOMPOSITION",
    ],
}
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# READ-ONLY reuse of the anchor's price-oracle detector + constants so events are
# byte-identical between anchor and decomposition.
from strat.ti_oracle_anchor import (  # noqa: E402
    MoveEvent,
    WINDOW_BARS,
    PRICE_ROI_LO,
    PRICE_ROI_HI,
    TAKER_RT,
    find_price_oracle_events,
    load_ohlc,
)

# ---- decomposition space ----------------------------------------------------

MA_TYPES = ("SMA", "EMA", "WMA", "HMA", "DEMA")
FAST_GRID = (5, 10, 20, 50)
SLOW_GRID = (20, 50, 100, 200)
SINGLE_LEN_GRID = (10, 20, 50, 100)

# F3 / F5 mechanical-exit sweep ("varying exits").
TP_GRID = (0.03, 0.05, 0.08)      # fixed take-profit
SL_GRID = (0.03, 0.05)            # fixed stop-loss
ATR_TRAIL_MULT = 3.0              # ATR-trailing stop multiple
ATR_LEN = 14

FORMULATIONS = ("F1_PRICE_MA", "F2_CROSS", "F3_CROSS_MECH",
                "F4_STACK", "F5_PRICE_MA_MECH")


# ---- MA primitives (causal) -------------------------------------------------
# pandas_ta if available (project adopted it); else local causal implementations.

try:
    import pandas as _pd
    import pandas_ta as _ta
    _HAVE_PANDAS_TA = True
except Exception:  # pragma: no cover - environment fallback
    _HAVE_PANDAS_TA = False


def _sma_local(x: np.ndarray, n: int) -> np.ndarray:
    n = int(n)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if n <= 0 or len(x) < n:
        return out
    csum = np.cumsum(np.insert(x, 0, 0.0))
    out[n - 1:] = (csum[n:] - csum[:-n]) / n
    return out


def _ema_local(x: np.ndarray, n: int) -> np.ndarray:
    n = int(n)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if n <= 0 or len(x) < n:
        return out
    alpha = 2.0 / (n + 1.0)
    seed = float(np.mean(x[:n]))
    out[n - 1] = seed
    prev = seed
    for i in range(n, len(x)):
        prev = alpha * x[i] + (1.0 - alpha) * prev
        out[i] = prev
    return out


def _wma_local(x: np.ndarray, n: int) -> np.ndarray:
    n = int(n)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if n <= 0 or len(x) < n:
        return out
    w = np.arange(1, n + 1, dtype=np.float64)
    wsum = w.sum()
    for i in range(n - 1, len(x)):
        out[i] = float(np.dot(x[i - n + 1:i + 1], w) / wsum)
    return out


def _hma_local(x: np.ndarray, n: int) -> np.ndarray:
    """Hull MA = WMA( 2*WMA(n/2) - WMA(n), sqrt(n) ). Causal."""
    n = int(n)
    if n <= 1 or len(x) < n:
        return np.full(x.shape, np.nan, dtype=np.float64)
    half = max(1, n // 2)
    sq = max(1, int(round(np.sqrt(n))))
    w_half = _wma_local(x, half)
    w_full = _wma_local(x, n)
    raw = 2.0 * w_half - w_full
    # raw has NaN until w_full warm; WMA over raw needs sq valid points.
    return _wma_local_nan(raw, sq)


def _wma_local_nan(x: np.ndarray, n: int) -> np.ndarray:
    """WMA that tolerates leading NaN in x (HMA inner step)."""
    n = int(n)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if n <= 0 or len(x) < n:
        return out
    w = np.arange(1, n + 1, dtype=np.float64)
    wsum = w.sum()
    for i in range(n - 1, len(x)):
        seg = x[i - n + 1:i + 1]
        if np.isnan(seg).any():
            continue
        out[i] = float(np.dot(seg, w) / wsum)
    return out


def _dema_local(x: np.ndarray, n: int) -> np.ndarray:
    """DEMA = 2*EMA(n) - EMA(EMA(n)). Causal."""
    e1 = _ema_local(x, n)
    # EMA over e1 ignoring leading NaN.
    out = np.full(x.shape, np.nan, dtype=np.float64)
    n = int(n)
    valid = ~np.isnan(e1)
    if valid.sum() < n:
        return out
    first = int(np.argmax(valid))
    sub = e1[first:]
    e2_sub = _ema_local(sub, n)
    e2 = np.full(x.shape, np.nan, dtype=np.float64)
    e2[first:] = e2_sub
    return 2.0 * e1 - e2


def moving_avg(x: np.ndarray, n: int, kind: str) -> np.ndarray:
    """Causal MA of the requested type. Uses pandas_ta when present, else local."""
    if _HAVE_PANDAS_TA:
        s = _pd.Series(x, dtype="float64")
        if kind == "SMA":
            r = _ta.sma(s, length=int(n))
        elif kind == "EMA":
            r = _ta.ema(s, length=int(n))
        elif kind == "WMA":
            r = _ta.wma(s, length=int(n))
        elif kind == "HMA":
            r = _ta.hma(s, length=int(n))
        elif kind == "DEMA":
            r = _ta.dema(s, length=int(n))
        else:
            raise ValueError(f"unknown MA kind {kind}")
        if r is None:
            return np.full(x.shape, np.nan, dtype=np.float64)
        return r.to_numpy(dtype=np.float64)
    # local fallback
    if kind == "SMA":
        return _sma_local(x, n)
    if kind == "EMA":
        return _ema_local(x, n)
    if kind == "WMA":
        return _wma_local(x, n)
    if kind == "HMA":
        return _hma_local(x, n)
    if kind == "DEMA":
        return _dema_local(x, n)
    raise ValueError(f"unknown MA kind {kind}")


def causal_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               n: int = ATR_LEN) -> np.ndarray:
    """Causal Wilder-style ATR (SMA of true range as the seed-free variant).

    out[i] uses bars <= i only. NaN until warm.
    """
    n = int(n)
    tr = np.full(close.shape, np.nan, dtype=np.float64)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    hl = high - low
    hc = np.abs(high - prev_close)
    lc = np.abs(low - prev_close)
    tr = np.maximum(hl, np.maximum(hc, lc))
    tr[0] = high[0] - low[0]
    return _sma_local(tr, n)


# ---- formulation simulators (each returns realized long ROI in window) -------

def _apply_mech_exit(
    open_full: np.ndarray,
    atr: np.ndarray,
    entry_idx: int,
    entry_px: float,
    win_end: int,
    n: int,
    exit_rule: tuple,
) -> tuple[float, int]:
    """Walk forward from entry to find a mechanical exit; return (gross_ret, exit_idx).

    Bars are evaluated at OPEN (causal, conservative: we only see the open price of
    each bar to decide an exit, fill at that same open). The price path between opens
    is unknown to us, so we test the level at each bar's open and at window end.

    exit_rule is one of:
      ("TP", p)         take-profit at +p   -> exit at first open >= entry*(1+p)
      ("SL", p)         stop-loss at -p     -> exit at first open <= entry*(1-p)
      ("TIME", None)    time-stop           -> exit at window end
      ("ATR", mult)     ATR-trailing stop   -> trail high - mult*ATR; exit when open
                                               crosses below the trailing level
    """
    kind = exit_rule[0]
    last = min(win_end, n)
    if kind == "TIME":
        ex = last - 1
        return open_full[ex] / entry_px - 1.0, ex
    if kind == "TP":
        tgt = entry_px * (1.0 + exit_rule[1])
        for t in range(entry_idx + 1, last):
            if open_full[t] >= tgt:
                return open_full[t] / entry_px - 1.0, t
        ex = last - 1
        return open_full[ex] / entry_px - 1.0, ex
    if kind == "SL":
        stop = entry_px * (1.0 - exit_rule[1])
        for t in range(entry_idx + 1, last):
            if open_full[t] <= stop:
                return open_full[t] / entry_px - 1.0, t
        ex = last - 1
        return open_full[ex] / entry_px - 1.0, ex
    if kind == "ATR":
        mult = exit_rule[1]
        peak = entry_px
        for t in range(entry_idx + 1, last):
            px = open_full[t]
            if px > peak:
                peak = px
            a = atr[t]
            if not np.isnan(a):
                trail = peak - mult * a
                if px <= trail:
                    return px / entry_px - 1.0, t
        ex = last - 1
        return open_full[ex] / entry_px - 1.0, ex
    raise ValueError(f"unknown exit rule {kind}")


def f1_price_ma(ma: np.ndarray, open_full: np.ndarray, close_full: np.ndarray,
                win_start: int, win_end: int) -> float:
    """F1 PRICE>MA: long while close>MA; exit when close<MA. Next-bar-open fills."""
    n = len(open_full)
    total = 0.0
    in_pos = False
    entry_px = 0.0
    last = min(win_end, n)
    # state-on-entry: if at win_start close>MA already, enter at win_start open.
    if (win_start < n and not np.isnan(ma[win_start])
            and close_full[win_start] > ma[win_start]):
        if open_full[win_start] > 0:
            entry_px = float(open_full[win_start])
            in_pos = True
    t0 = max(1, win_start - 1)
    for t in range(t0, last):
        if np.isnan(ma[t]):
            continue
        sig_long = close_full[t] > ma[t]
        sig_exit = close_full[t] < ma[t]
        fill = t + 1
        if not in_pos and sig_long:
            if win_start <= fill < last:
                px = float(open_full[fill])
                if px > 0:
                    entry_px = px
                    in_pos = True
        elif in_pos and sig_exit:
            if fill < last:
                exit_px = float(open_full[fill])
            else:
                exit_px = float(open_full[last - 1])
            if entry_px > 0:
                total += (exit_px / entry_px - 1.0) - TAKER_RT
            in_pos = False
    if in_pos and entry_px > 0:
        exit_px = float(open_full[last - 1])
        total += (exit_px / entry_px - 1.0) - TAKER_RT
    return total


def f2_cross(f: np.ndarray, s: np.ndarray, open_full: np.ndarray,
             win_start: int, win_end: int) -> float:
    """F2 golden-cross entry / death-cross exit. (Same logic as the anchor cross.)"""
    above = f > s
    n = len(open_full)
    last = min(win_end, n)
    total = 0.0
    in_pos = False
    entry_px = 0.0
    ws = win_start
    if (ws < n and not np.isnan(f[ws]) and not np.isnan(s[ws])
            and ws - 1 >= 0 and not np.isnan(f[ws - 1]) and not np.isnan(s[ws - 1])
            and above[ws]):
        entry_px = float(open_full[ws])
        if entry_px > 0:
            in_pos = True
    t0 = max(1, win_start - 1)
    for t in range(t0, last):
        if np.isnan(f[t]) or np.isnan(s[t]) or np.isnan(f[t - 1]) or np.isnan(s[t - 1]):
            continue
        golden = (not above[t - 1]) and above[t]
        death = above[t - 1] and (not above[t])
        fill = t + 1
        if not in_pos and golden:
            if win_start <= fill < last:
                px = float(open_full[fill])
                if px > 0:
                    entry_px = px
                    in_pos = True
        elif in_pos and death:
            if fill < last:
                exit_px = float(open_full[fill])
            else:
                exit_px = float(open_full[last - 1])
            if entry_px > 0:
                total += (exit_px / entry_px - 1.0) - TAKER_RT
            in_pos = False
    if in_pos and entry_px > 0:
        exit_px = float(open_full[last - 1])
        total += (exit_px / entry_px - 1.0) - TAKER_RT
    return total


def f3_cross_mech(f: np.ndarray, s: np.ndarray, open_full: np.ndarray,
                  atr: np.ndarray, win_start: int, win_end: int,
                  exit_rule: tuple) -> float:
    """F3 golden-cross entry; mechanical exit (exit_rule). One trade per cross;
    re-arm after a mechanical exit if another golden cross fires in-window."""
    above = f > s
    n = len(open_full)
    last = min(win_end, n)
    total = 0.0
    armed_for_reentry = True
    t = max(1, win_start - 1)
    # state-on-entry
    in_initial = False
    if (win_start < n and not np.isnan(f[win_start]) and not np.isnan(s[win_start])
            and above[win_start] and open_full[win_start] > 0):
        in_initial = True
    while t < last:
        if np.isnan(f[t]) or np.isnan(s[t]) or np.isnan(f[t - 1]) or np.isnan(s[t - 1]):
            t += 1
            continue
        golden = (not above[t - 1]) and above[t]
        entry_idx = -1
        entry_px = 0.0
        if in_initial:
            entry_idx = win_start
            entry_px = float(open_full[win_start])
            in_initial = False
        elif armed_for_reentry and golden:
            fill = t + 1
            if win_start <= fill < last:
                entry_idx = fill
                entry_px = float(open_full[fill])
        if entry_idx >= 0 and entry_px > 0:
            gross, exit_idx = _apply_mech_exit(
                open_full, atr, entry_idx, entry_px, last, n, exit_rule)
            total += gross - TAKER_RT
            t = exit_idx + 1  # resume scanning after the exit
            continue
        t += 1
    return total


def f4_stack(f: np.ndarray, s: np.ndarray, open_full: np.ndarray,
             close_full: np.ndarray, win_start: int, win_end: int) -> float:
    """F4 STACK: long when close>fast>slow aligned; exit when the stack breaks."""
    n = len(open_full)
    last = min(win_end, n)

    def stacked(t):
        if np.isnan(f[t]) or np.isnan(s[t]):
            return False
        return close_full[t] > f[t] > s[t]

    total = 0.0
    in_pos = False
    entry_px = 0.0
    if win_start < n and stacked(win_start) and open_full[win_start] > 0:
        entry_px = float(open_full[win_start])
        in_pos = True
    t0 = max(1, win_start - 1)
    for t in range(t0, last):
        if np.isnan(f[t]) or np.isnan(s[t]):
            continue
        is_stack = stacked(t)
        fill = t + 1
        if not in_pos and is_stack:
            if win_start <= fill < last:
                px = float(open_full[fill])
                if px > 0:
                    entry_px = px
                    in_pos = True
        elif in_pos and not is_stack:
            if fill < last:
                exit_px = float(open_full[fill])
            else:
                exit_px = float(open_full[last - 1])
            if entry_px > 0:
                total += (exit_px / entry_px - 1.0) - TAKER_RT
            in_pos = False
    if in_pos and entry_px > 0:
        exit_px = float(open_full[last - 1])
        total += (exit_px / entry_px - 1.0) - TAKER_RT
    return total


def f5_price_ma_mech(ma: np.ndarray, open_full: np.ndarray, close_full: np.ndarray,
                     atr: np.ndarray, win_start: int, win_end: int,
                     exit_rule: tuple) -> float:
    """F5 PRICE>MA entry (close>MA), mechanical exit. Re-arm on a fresh close>MA
    crossing after a mechanical exit."""
    n = len(open_full)
    last = min(win_end, n)
    total = 0.0
    t = max(1, win_start - 1)
    in_initial = False
    if (win_start < n and not np.isnan(ma[win_start])
            and close_full[win_start] > ma[win_start] and open_full[win_start] > 0):
        in_initial = True
    while t < last:
        if np.isnan(ma[t]) or np.isnan(ma[t - 1]):
            t += 1
            continue
        cross_up = (close_full[t - 1] <= ma[t - 1]) and (close_full[t] > ma[t])
        entry_idx = -1
        entry_px = 0.0
        if in_initial:
            entry_idx = win_start
            entry_px = float(open_full[win_start])
            in_initial = False
        elif cross_up:
            fill = t + 1
            if win_start <= fill < last:
                entry_idx = fill
                entry_px = float(open_full[fill])
        if entry_idx >= 0 and entry_px > 0:
            gross, exit_idx = _apply_mech_exit(
                open_full, atr, entry_idx, entry_px, last, n, exit_rule)
            total += gross - TAKER_RT
            t = exit_idx + 1
            continue
        t += 1
    return total


# ---- candidate enumeration (per cadence, precomputed) -----------------------

@dataclass
class Candidate:
    formulation: str
    ma_type: str
    params: str        # human label of the param tuple
    # closure computing realized long return given (open, close, atr, ev)
    fn: object


def build_candidates(
    open_full: np.ndarray,
    high_full: np.ndarray,
    low_full: np.ndarray,
    close_full: np.ndarray,
) -> list[Candidate]:
    """Precompute every MA the grid needs ONCE per cadence, then bind candidate
    closures. Each closure takes (ev_start, ev_end) and returns realized long ROI.
    """
    ma_cache: dict[tuple[str, int], np.ndarray] = {}

    def get_ma(kind: str, length: int) -> np.ndarray:
        key = (kind, length)
        if key not in ma_cache:
            ma_cache[key] = moving_avg(close_full, length, kind)
        return ma_cache[key]

    atr = causal_atr(high_full, low_full, close_full, ATR_LEN)

    mech_rules: list[tuple] = (
        [("TP", p) for p in TP_GRID]
        + [("SL", p) for p in SL_GRID]
        + [("TIME", None)]
        + [("ATR", ATR_TRAIL_MULT)]
    )

    def _rule_label(r):
        if r[0] == "TIME":
            return "TIME"
        if r[0] == "ATR":
            return f"ATR{r[1]:g}x"
        return f"{r[0]}{int(round(r[1] * 100))}"

    cands: list[Candidate] = []

    for kind in MA_TYPES:
        # F1 PRICE>MA (single-MA state) -- single_len grid
        for length in SINGLE_LEN_GRID:
            ma = get_ma(kind, length)

            def mk_f1(ma=ma):
                return lambda a, b: f1_price_ma(ma, open_full, close_full, a, b)
            cands.append(Candidate("F1_PRICE_MA", kind, f"len{length}", mk_f1()))

        # F2 CROSS / F3 CROSS_MECH / F4 STACK -- (fast,slow) grid
        for fast in FAST_GRID:
            for slow in SLOW_GRID:
                if fast >= slow:
                    continue
                f = get_ma(kind, fast)
                s = get_ma(kind, slow)

                def mk_f2(f=f, s=s):
                    return lambda a, b: f2_cross(f, s, open_full, a, b)
                cands.append(Candidate("F2_CROSS", kind, f"{fast}x{slow}", mk_f2()))

                def mk_f4(f=f, s=s):
                    return lambda a, b: f4_stack(f, s, open_full, close_full, a, b)
                cands.append(Candidate("F4_STACK", kind, f"{fast}x{slow}", mk_f4()))

                for rule in mech_rules:
                    def mk_f3(f=f, s=s, rule=rule):
                        return lambda a, b: f3_cross_mech(
                            f, s, open_full, atr, a, b, rule)
                    cands.append(Candidate(
                        "F3_CROSS_MECH", kind,
                        f"{fast}x{slow}+{_rule_label(rule)}", mk_f3()))

        # F5 PRICE>MA + MECH-EXIT -- single_len grid x mech rules
        for length in SINGLE_LEN_GRID:
            ma = get_ma(kind, length)
            for rule in mech_rules:
                def mk_f5(ma=ma, rule=rule):
                    return lambda a, b: f5_price_ma_mech(
                        ma, open_full, close_full, atr, a, b, rule)
                cands.append(Candidate(
                    "F5_PRICE_MA_MECH", kind,
                    f"len{length}+{_rule_label(rule)}", mk_f5()))

    return cands


# ---- per-event evaluation ---------------------------------------------------

@dataclass
class EventResult:
    price_roi: float
    # best captured ROI overall + winning DNA
    best_roi: float
    best_formulation: str
    best_ma_type: str
    best_params: str
    # best captured ROI per-formulation and per-ma-type (for win-share + mean)
    best_by_formulation: dict           # {formulation: best_roi}
    best_by_ma_type: dict               # {ma_type: best_roi}


def evaluate_event(cands: list[Candidate], ev: MoveEvent) -> EventResult:
    best_roi = -np.inf
    best_f = best_t = best_p = "NONE"
    by_form: dict[str, float] = defaultdict(lambda: -np.inf)
    by_type: dict[str, float] = defaultdict(lambda: -np.inf)
    for c in cands:
        roi = c.fn(ev.start, ev.end)
        if roi > by_form[c.formulation]:
            by_form[c.formulation] = roi
        if roi > by_type[c.ma_type]:
            by_type[c.ma_type] = roi
        if roi > best_roi:
            best_roi = roi
            best_f, best_t, best_p = c.formulation, c.ma_type, c.params
    if best_roi == -np.inf:
        best_roi = 0.0
    return EventResult(
        price_roi=ev.price_roi,
        best_roi=best_roi,
        best_formulation=best_f,
        best_ma_type=best_t,
        best_params=best_p,
        best_by_formulation={k: (v if v != -np.inf else 0.0) for k, v in by_form.items()},
        best_by_ma_type={k: (v if v != -np.inf else 0.0) for k, v in by_type.items()},
    )


# ---- per-cadence driver -----------------------------------------------------

@dataclass
class CadenceDecomp:
    cadence: str
    n_events: int
    events: list[EventResult] = field(default_factory=list)

    # cross-only anchor TI-oracle for the SAME events (apples-to-apples)
    cross_only_ti: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        if self.n_events == 0:
            return {"cadence": self.cadence, "n_events": 0,
                    "overall_decomposed": None, "overall_cross_only": None,
                    "by_formulation": {}, "by_ma_type": {},
                    "winning_dna": {"formulation": {}, "ma_type": {},
                                    "form_x_type": {}, "top_params": {}}}
        pr = np.array([e.price_roi for e in self.events])
        decomp = np.array([e.best_roi for e in self.events])
        cross = np.array(self.cross_only_ti) if self.cross_only_ti else np.array([])

        def cap_means(ti):
            return float(np.mean(ti) / np.mean(pr)) if len(ti) and np.mean(pr) else None

        def cap_med_ev(ti):
            if not len(ti):
                return None
            per = ti / np.where(pr == 0, np.nan, pr)
            return float(np.nanmedian(per))

        # by-formulation: mean of each formulation's best-config capture + win-share
        forms = FORMULATIONS
        by_form = {}
        win_counts_form = Counter(e.best_formulation for e in self.events)
        for fm in forms:
            caps = []
            for e in self.events:
                b = e.best_by_formulation.get(fm)
                if b is None:
                    continue
                caps.append(b / e.price_roi if e.price_roi > 0 else 0.0)
            by_form[fm] = {
                "mean_best_capture": float(np.mean(caps)) if caps else None,
                "mean_best_roi": float(np.mean([e.best_by_formulation.get(fm, 0.0)
                                                for e in self.events])),
                "win_share": win_counts_form.get(fm, 0) / self.n_events,
                "wins": win_counts_form.get(fm, 0),
            }

        by_type = {}
        win_counts_type = Counter(e.best_ma_type for e in self.events)
        for mt in MA_TYPES:
            caps = []
            for e in self.events:
                b = e.best_by_ma_type.get(mt)
                if b is None:
                    continue
                caps.append(b / e.price_roi if e.price_roi > 0 else 0.0)
            by_type[mt] = {
                "mean_best_capture": float(np.mean(caps)) if caps else None,
                "mean_best_roi": float(np.mean([e.best_by_ma_type.get(mt, 0.0)
                                                for e in self.events])),
                "win_share": win_counts_type.get(mt, 0) / self.n_events,
                "wins": win_counts_type.get(mt, 0),
            }

        form_x_type = Counter(f"{e.best_formulation}|{e.best_ma_type}"
                              for e in self.events)
        top_params = Counter(f"{e.best_formulation}|{e.best_ma_type}|{e.best_params}"
                             for e in self.events)

        return {
            "cadence": self.cadence,
            "n_events": self.n_events,
            "price_oracle_mean": float(np.mean(pr)),
            "overall_decomposed": {
                "ti_mean": float(np.mean(decomp)),
                "ti_median": float(np.median(decomp)),
                "capture_ratio_of_means": cap_means(decomp),
                "capture_ratio_median_of_per_event": cap_med_ev(decomp),
            },
            "overall_cross_only": {
                "ti_mean": float(np.mean(cross)) if len(cross) else None,
                "ti_median": float(np.median(cross)) if len(cross) else None,
                "capture_ratio_of_means": cap_means(cross),
                "capture_ratio_median_of_per_event": cap_med_ev(cross),
            },
            "decomposed_minus_cross_capture": (
                (cap_means(decomp) - cap_means(cross))
                if cap_means(decomp) is not None and cap_means(cross) is not None
                else None),
            "by_formulation": by_form,
            "by_ma_type": by_type,
            "winning_dna": {
                "formulation": dict(win_counts_form.most_common()),
                "ma_type": dict(win_counts_type.most_common()),
                "form_x_type": dict(form_x_type.most_common(10)),
                "top_params": dict(top_params.most_common(10)),
            },
        }


def _cross_only_ti_for_event(close_full, open_full, ev: MoveEvent) -> float:
    """Recompute the anchor's SMA/EMA cross-only TI-oracle for ONE event so the
    decomposed space can be compared to the cross-only anchor on identical events.
    Mirrors ti_oracle_anchor: types {SMA,EMA}, golden/death cross, same grid.
    """
    best = -np.inf
    for kind in ("SMA", "EMA"):
        for fast in FAST_GRID:
            for slow in SLOW_GRID:
                if fast >= slow:
                    continue
                f = moving_avg(close_full, fast, kind)
                s = moving_avg(close_full, slow, kind)
                roi = f2_cross(f, s, open_full, ev.start, ev.end)
                if roi > best:
                    best = roi
    return best if best != -np.inf else 0.0


def run_cadence(open_a, high_a, low_a, close_a, cadence: str) -> CadenceDecomp:
    win_lo, win_hi = WINDOW_BARS[cadence]
    events = find_price_oracle_events(high_a, low_a, win_lo, win_hi)
    res = CadenceDecomp(cadence=cadence, n_events=len(events))
    cands = build_candidates(open_a, high_a, low_a, close_a)
    # cross-only MA cache (reused across events): precompute once.
    cross_cache: dict[tuple[str, int], np.ndarray] = {}

    def get_cross_ma(kind, length):
        key = (kind, length)
        if key not in cross_cache:
            cross_cache[key] = moving_avg(close_a, length, kind)
        return cross_cache[key]

    def cross_only(ev):
        best = -np.inf
        for kind in ("SMA", "EMA"):
            for fast in FAST_GRID:
                for slow in SLOW_GRID:
                    if fast >= slow:
                        continue
                    roi = f2_cross(get_cross_ma(kind, fast), get_cross_ma(kind, slow),
                                   open_a, ev.start, ev.end)
                    if roi > best:
                        best = roi
        return best if best != -np.inf else 0.0

    for ev in events:
        res.events.append(evaluate_event(cands, ev))
        res.cross_only_ti.append(cross_only(ev))
    return res


def aggregate(results: list[CadenceDecomp]) -> dict:
    agg = CadenceDecomp(cadence="AGGREGATE", n_events=0)
    for r in results:
        agg.events.extend(r.events)
        agg.cross_only_ti.extend(r.cross_only_ti)
    agg.n_events = len(agg.events)
    return agg.summary()


# ---- reporting --------------------------------------------------------------

def _fmt(v, pct=True):
    if v is None:
        return "   n/a"
    return f"{v * 100:6.2f}%" if pct else f"{v:6.3f}"


def print_report(per_cadence: list[dict], agg: dict) -> None:
    print("")
    print("=" * 100)
    print("TI-ORACLE DECOMPOSITION  (FORMULATIONS x MA-TYPES) -- 'a multitude of answers'")
    print("=" * 100)

    # 1) OVERALL: decomposed vs cross-only
    print("\n[1] OVERALL TI-ORACLE: decomposed space vs cross-only anchor "
          "(capture = TI/price-oracle)")
    hdr = (f"{'cadence':>10} | {'N':>4} | {'decomp cap(mean)':>16} "
           f"{'cross cap(mean)':>16} | {'gain':>8} | {'decomp cap(med/ev)':>18}")
    print(hdr)
    print("-" * 100)
    for s in per_cadence + [agg]:
        if s["n_events"] == 0:
            print(f"{s['cadence']:>10} | {0:>4} | (no events)")
            continue
        d = s["overall_decomposed"]
        c = s["overall_cross_only"]
        gain = s["decomposed_minus_cross_capture"]
        print(f"{s['cadence']:>10} | {s['n_events']:>4} | "
              f"{_fmt(d['capture_ratio_of_means'], pct=False):>16} "
              f"{_fmt(c['capture_ratio_of_means'], pct=False):>16} | "
              f"{_fmt(gain, pct=False):>8} | "
              f"{_fmt(d['capture_ratio_median_of_per_event'], pct=False):>18}")

    # 2) by formulation (aggregate)
    print("\n[2] CAPTURE BY FORMULATION (aggregate): mean best-config capture + win-share")
    print(f"{'formulation':>18} | {'mean capture':>13} | {'win-share':>10} | {'wins':>5}")
    print("-" * 60)
    for fm in FORMULATIONS:
        row = agg["by_formulation"].get(fm, {})
        print(f"{fm:>18} | {_fmt(row.get('mean_best_capture'), pct=False):>13} | "
              f"{_fmt(row.get('win_share'), pct=False):>10} | {row.get('wins', 0):>5}")

    # 3) by MA type (aggregate)
    print("\n[3] CAPTURE BY MA-TYPE (aggregate): mean best-config capture + win-share")
    print(f"{'ma_type':>10} | {'mean capture':>13} | {'win-share':>10} | {'wins':>5}")
    print("-" * 50)
    for mt in MA_TYPES:
        row = agg["by_ma_type"].get(mt, {})
        print(f"{mt:>10} | {_fmt(row.get('mean_best_capture'), pct=False):>13} | "
              f"{_fmt(row.get('win_share'), pct=False):>10} | {row.get('wins', 0):>5}")

    # 4) winning DNA distribution (aggregate)
    print("\n[4] WINNING DNA DISTRIBUTION (aggregate)")
    dna = agg["winning_dna"]
    print("  formulation wins :", ", ".join(f"{k}:{v}" for k, v in dna["formulation"].items()))
    print("  ma_type wins     :", ", ".join(f"{k}:{v}" for k, v in dna["ma_type"].items()))
    print("  top form|type    :")
    for k, v in list(dna["form_x_type"].items())[:8]:
        print(f"      {k}: {v}")
    print("  top form|type|params :")
    for k, v in list(dna["top_params"].items())[:8]:
        print(f"      {k}: {v}")

    # 5) the thesis answer
    print("\n[5] IS THE ORACLE A MULTITUDE?")
    print("   " + multitude_verdict(agg))
    print("=" * 100)
    print("NOTE: DECOMPOSITION (not a gate). HINDSIGHT config choice IS the oracle. "
          "No pass/fail / null verdict.")
    print("")


def multitude_verdict(agg: dict) -> str:
    """One-line answer to 'is the oracle a multitude?' Based on win-concentration of
    formulations and MA-types at the aggregate level."""
    if agg["n_events"] == 0:
        return "no events -- cannot assess."
    form_wins = agg["winning_dna"]["formulation"]
    type_wins = agg["winning_dna"]["ma_type"]
    n = agg["n_events"]
    top_form_share = (max(form_wins.values()) / n) if form_wins else 0.0
    top_type_share = (max(type_wins.values()) / n) if type_wins else 0.0
    n_form_used = sum(1 for v in form_wins.values() if v > 0)
    n_type_used = sum(1 for v in type_wins.values() if v > 0)
    top_form = max(form_wins, key=form_wins.get) if form_wins else "NONE"
    top_type = max(type_wins, key=type_wins.get) if type_wins else "NONE"
    multitude = (top_form_share < 0.80) or (top_type_share < 0.80) or (n_form_used >= 3)
    if multitude:
        return (f"YES -- a multitude. {n_form_used}/{len(FORMULATIONS)} formulations and "
                f"{n_type_used}/{len(MA_TYPES)} MA-types each win >=1 move; top formulation "
                f"'{top_form}' wins only {top_form_share*100:.0f}%, top type '{top_type}' "
                f"{top_type_share*100:.0f}%. Different moves are best-captured by different "
                f"(formulation x type).")
    return (f"NO -- one configuration dominates: '{top_form}' wins {top_form_share*100:.0f}% "
            f"and MA-type '{top_type}' wins {top_type_share*100:.0f}% of moves.")


# ---- selftest ---------------------------------------------------------------

def _synth_mixed(n=600, seed=3):
    """A series with BOTH clean trend legs (favor cross/stack rides) and choppy
    legs with sharp pops (favor mechanical take-profit exits) -- so different
    formulations SHOULD win different events if the decomposition is faithful."""
    rng = np.random.default_rng(seed)
    price = [100.0]
    for i in range(n):
        phase = (i // 30) % 2
        if phase == 0:  # clean trend leg
            drift = 0.004
            ret = drift + rng.normal(0, 0.0010)
        else:           # choppy pop-and-fade leg
            ret = 0.02 * np.sin(i / 2.0) * 0.3 + rng.normal(0, 0.007)
        price.append(price[-1] * (1.0 + ret))
    p = np.array(price, dtype=np.float64)
    o = p.copy()
    c = p.copy()
    h = p * 1.0015
    lo = p * 0.9985
    return o, h, lo, c


def selftest() -> bool:
    ok = True
    # 1) MA causality (full vs truncated overlap identical) for all 5 types.
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(0, 1, 400)) + 100.0
    cut = 300
    for kind in MA_TYPES:
        full = moving_avg(x, 20, kind)
        part = moving_avg(x[:cut], 20, kind)
        a, b = full[:cut], part
        m = ~(np.isnan(a) | np.isnan(b))
        if m.any():
            md = float(np.nanmax(np.abs(a[m] - b[m])))
            if md > 1e-9:
                print(f"[selftest] FAIL: {kind} not causal (full vs truncated diff={md:.2e})")
                ok = False

    # 2) ATR causality.
    h = x * 1.002
    lo = x * 0.998
    atr_full = causal_atr(h, lo, x, ATR_LEN)
    atr_part = causal_atr(h[:cut], lo[:cut], x[:cut], ATR_LEN)
    m = ~(np.isnan(atr_full[:cut]) | np.isnan(atr_part))
    if m.any() and float(np.nanmax(np.abs(atr_full[:cut][m] - atr_part[m]))) > 1e-9:
        print("[selftest] FAIL: ATR not causal")
        ok = False

    # 3) Run the full decomposition on a mixed synthetic series.
    o, hi, lo2, c = _synth_mixed()
    win_lo, win_hi = WINDOW_BARS["1d"]
    events = find_price_oracle_events(hi, lo2, win_lo, win_hi)
    if len(events) == 0:
        print("[selftest] FAIL: synthetic produced no price-oracle events")
        return False
    cands = build_candidates(o, hi, lo2, c)
    res = CadenceDecomp(cadence="1d", n_events=len(events))
    for ev in events:
        res.events.append(evaluate_event(cands, ev))
        res.cross_only_ti.append(_cross_only_ti_for_event(c, o, ev))
    s = res.summary()

    # 4) Decomposed capture must be >= cross-only capture (richer space, hindsight max).
    d = s["overall_decomposed"]["capture_ratio_of_means"]
    cr = s["overall_cross_only"]["capture_ratio_of_means"]
    print(f"[selftest] {len(events)} events; decomp cap(mean)={d:.3f}  "
          f"cross-only cap(mean)={cr:.3f}")
    if d + 1e-9 < cr:
        print("[selftest] FAIL: decomposed capture < cross-only (must be >= by construction)")
        ok = False

    # 5) Each candidate ROI must be finite.
    for ev in events[:3]:
        for cand in cands[:50]:
            r = cand.fn(ev.start, ev.end)
            if not np.isfinite(r):
                print(f"[selftest] FAIL: non-finite candidate ROI ({cand.formulation}/"
                      f"{cand.ma_type}/{cand.params})")
                ok = False
                break

    # 6) Win-share fractions sum to ~1 across formulations.
    share_sum = sum(v["win_share"] for v in s["by_formulation"].values())
    if abs(share_sum - 1.0) > 1e-6:
        print(f"[selftest] FAIL: formulation win-shares sum to {share_sum:.4f} (!=1)")
        ok = False

    print(f"[selftest] multitude verdict: {multitude_verdict(s)}")
    print("[selftest] PASS" if ok else "[selftest] FAIL")
    return ok


# ---- main -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="TI-oracle DECOMPOSITION (formulations x MA-types)")
    ap.add_argument("--asset", default="BTCUSDT")
    ap.add_argument("--cadences", default="1d,4h,1h")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    cadences = [c.strip() for c in args.cadences.split(",") if c.strip()]
    for cad in cadences:
        if cad not in WINDOW_BARS:
            print(f"[error] unknown cadence '{cad}'; known={list(WINDOW_BARS)}")
            sys.exit(2)

    backend = "pandas_ta" if _HAVE_PANDAS_TA else "local"
    print(f"[info] MA backend: {backend}")

    results: list[CadenceDecomp] = []
    per_cadence_summ: list[dict] = []
    for cad in cadences:
        print(f"[run] {args.asset} {cad}: loading + scanning + decomposing ...", flush=True)
        o, h, lo, c = load_ohlc(args.asset, cad)
        res = run_cadence(o, h, lo, c, cad)
        results.append(res)
        s = res.summary()
        per_cadence_summ.append(s)
        print(f"[run] {args.asset} {cad}: {res.n_events} events", flush=True)

    agg = aggregate(results)
    print_report(per_cadence_summ, agg)

    asset_short = args.asset.upper().replace("USDT", "")
    out_path = (Path(args.out) if args.out
                else PROJECT_ROOT / "runs" / "strat" /
                f"ti_oracle_decompose_{asset_short}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "tool": "ti_oracle_decompose",
        "anchor_not_gate": True,
        "decomposition": True,
        "asset": args.asset,
        "cadences": cadences,
        "ma_backend": backend,
        "spec": {
            "price_roi_band": [PRICE_ROI_LO, PRICE_ROI_HI],
            "window_bars": {c: WINDOW_BARS[c] for c in cadences},
            "formulations": list(FORMULATIONS),
            "ma_types": list(MA_TYPES),
            "fast_grid": list(FAST_GRID),
            "slow_grid": list(SLOW_GRID),
            "single_len_grid": list(SINGLE_LEN_GRID),
            "mech_exits": {"tp": list(TP_GRID), "sl": list(SL_GRID),
                           "time_stop": "window_end", "atr_trail_mult": ATR_TRAIL_MULT,
                           "atr_len": ATR_LEN},
            "taker_rt": TAKER_RT,
            "fills": "next_bar_open",
            "events": "non_overlapping (identical to ti_oracle_anchor)",
            "hindsight": "formulation/type/param CHOICE only (signals causal)",
        },
        "per_cadence": per_cadence_summ,
        "aggregate": agg,
        "multitude_verdict": multitude_verdict(agg),
    }
    out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"[artifact] {out_path}")


if __name__ == "__main__":
    main()
