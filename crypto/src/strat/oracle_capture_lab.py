"""src/strat/oracle_capture_lab.py -- the REFRAMED CAPTURE-RATE x CADENCE x COST lab.

THE REFRAME (user, 2026-06-09 -- READ THIS FIRST)
-------------------------------------------------
STOP solving "predict the 25% of daily MOVERS" -- a parallel instance VERIFIED that null on held-out
(all 5 cross-sectional scores coin-flip, capture-vs-oracle ~0, net-negative after cost; persisted in
runs/strat/MOVER_CAPTURE_VERIFY_2026-06-09.md). INSTEAD solve:

    "capture >= 25% of the ORACLE move", where a MOVE = ANY 2-10% price slice at ANY time/cadence
    (incl. intraday), the unit being a SETUP across a MOVE; target = CAPTURE-RATE
    (realized_net / oracle_available_move) >= 25% AFTER realistic cost.

TWO ORACLE FORMS (both an UPPER BOUND, never tradeable):
  (1) PRICE-oracle  = the best long entry->exit on RAW PRICE inside a forward move-window (perfect-timing
      ceiling). Available move = the maximum (sell_high / buy_low - 1) achievable inside the window.
  (2) INDICATOR-oracle = the best CAUSAL adaptive-MA config (SMA/EMA golden->death cross family) whose
      signalled long entry/exit lands inside the window (realizable-FAMILY ceiling). HINDSIGHT is ONLY in
      picking the best config after the fact; each config's cross is computed past-only (no leak).

THE CAUSAL SIGNAL WE TEST (momentum-CONTINUATION, past-only)
-----------------------------------------------------------
SELECTION-ahead is null (the mover-capture verify) and entry-TIMING is fungible (the within-window
proxy), and CONTINUATION dominates over reversal (D53). So the realizable signal here is the simplest
honest momentum-continuation breakout: enter when price is breaking UP and the trend stack is aligned-up
at the close of bar t (all trailing / past-only):
    fire[t] = (close[t] > max(close[t-brk .. t-1]))            # breaking to a new local high (continuation)
              AND (SMA_fast[t] > SMA_slow[t])                  # trend-aligned up
We then chase the SAME single-position non-overlapping book the rest of the apparatus uses.

THE SINGLE-POSITION NON-OVERLAPPING ACCOUNTING (the 2026-06-09 fix -- DO NOT COMPOUND OVERLAPS)
----------------------------------------------------------------------------------------------
Aligned/continuation bars are CONSECUTIVE. You CANNOT hold N overlapping full-reinvest positions under a
fixed-size constraint, so geometric-compounding every fired bar's fixed-horizon trade is INVALID (the bug
that produced 122M%). The book is therefore SINGLE-POSITION, NON-OVERLAPPING: walk fired bars in time
order, take a trade ONLY when flat, enter opens[i+1], exit opens[i+1+h]; after a trade at i the next
entry must be > i+h. This mirrors selection_signal_lab.nonoverlap_book (reused, not re-derived).

THE CAPTURE-RATE (the headline statistic)
-----------------------------------------
For each ORACLE MOVE (a non-overlapping forward window where the price-oracle available move >= move_thr,
default 2%) we ask: how much of that available move did the realizable signal's single-position book
CAPTURE inside that window, AFTER cost?
    capture_price[move]  = realized_net_in_window / price_oracle_available_move
    capture_indicator[m] = realized_net_in_window / indicator_oracle_available_move
Pooled capture-rate (held-out) = sum(realized_net over moves) / sum(oracle_available_move over moves)
                               = total realized net / total oracle move  (a $-weighted capture, robust to
                                 per-move ratio blow-ups when a denominator is tiny). We ALSO report the
median per-move ratio. realized_net is NET of cost; the oracle move is GROSS (the ceiling). So capture
after cost falls as cost rises and as the oracle move shrinks (finer cadence) -> the COST-CLIFF curve.

THE VERDICT THIS PRODUCES
-------------------------
A CAPTURE x CADENCE x COST table (price-oracle + indicator-oracle, taker 0.24% RT + maker ~0.06% RT) per
{1d,4h,1h,15m,dollar} x {BTC,ETH,SOL}. At which cadence (if any) does the causal signal clear 25%-of-oracle
AFTER realistic cost? The COST-CLIFF curve -- as cadence gets finer the oracle move shrinks but cost stays
fixed -- shows whether capture-after-cost RISES (real intraday structure) or COLLAPSES (cost-cliff). That
curve IS the answer to the reframe.

DISCIPLINE (no leak): the signal's grid config is SELECTED on TRAIN+VAL only (50/20 of bars). UNSEEN is
read ONCE to report. The split is by BAR-INDEX quantiles (cadence-agnostic; intraday `date` repeats within
a day so a date-string split would mislabel intraday bars). Purge gap dropped at each boundary.

RWYB:
    python src/strat/oracle_capture_lab.py --selftest            # synthetic two-sided soundness (no data)
    python src/strat/oracle_capture_lab.py --asset BTCUSDT --cadence 4h    # one (asset,cadence)
    python src/strat/oracle_capture_lab.py --assets BTCUSDT ETHUSDT SOLUSDT \
        --cadences 1d 4h 1h 15m dollar                            # the full reframe grid
No emoji (cp1252-safe).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "runs" / "strat"

# Realistic round-trip costs (the brief).
TAKER = 0.0024   # 0.24% round-trip (taker both sides)
MAKER = 0.0006   # ~0.06% round-trip (maker both sides) -- OPTIMISTIC: p_fill 0.21-0.40 makes real maker worse
COSTS = {"taker": TAKER, "maker": MAKER}

WINDOWS = ["TRAIN", "VAL", "OOS", "UNSEEN"]
HELD = ["OOS", "UNSEEN"]

# Reuse the EXACT single-position non-overlapping accounting from selection_signal_lab (do NOT re-derive).
try:
    from .selection_signal_lab import (_net_for_entry, _compound_pct, nonoverlap_book)
except ImportError:  # run as a script
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from selection_signal_lab import (_net_for_entry, _compound_pct, nonoverlap_book)

# The continuation-signal grid (the realizable past-only signal).
BREAKOUTS = [10, 20, 40]          # bars for the prior-high breakout (continuation trigger)
FAST_SLOW = [(10, 50), (20, 100), (10, 20)]   # (fast, slow) SMA trend-alignment filter; fast < slow
HORIZONS = [5, 10, 20]            # bars held per trade (single-position fixed horizon)

# The indicator-oracle MA grid (SMA + EMA golden->death cross family; the realizable-FAMILY ceiling).
ORACLE_FAST = (5, 10, 20)
ORACLE_SLOW = (20, 50, 100)


# ===========================================================================
# Split: BAR-INDEX quantiles 50/20/20/10 with a purge gap (cadence-agnostic).
# ===========================================================================
def index_windows(n: int, purge: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Return (window_label_per_bar, purge_mask) by BAR-INDEX quantiles 50/20/20/10.

    Index-based (not date-string) so intraday cadences (whose `date` repeats within a day) split correctly.
    purge_mask[i]=True drops bar i (inside a purge gap around a boundary) from labelled use.
    """
    i50, i70, i90 = int(0.50 * n), int(0.70 * n), int(0.90 * n)
    wlab = np.empty(n, dtype=object)
    wlab[:i50] = "TRAIN"
    wlab[i50:i70] = "VAL"
    wlab[i70:i90] = "OOS"
    wlab[i90:] = "UNSEEN"
    purge_mask = np.zeros(n, dtype=bool)
    for b in (i50, i70, i90):
        lo, hi = max(0, b - purge), min(n, b + purge)
        purge_mask[lo:hi] = True
    return wlab, purge_mask


# ===========================================================================
# Causal MOMENTUM-CONTINUATION signal (past-only).
# ===========================================================================
def continuation_signal(close: np.ndarray, *, brk: int, fast: int, slow: int) -> np.ndarray:
    """fire[t] = close[t] > max(close[t-brk..t-1])  AND  SMA_fast[t] > SMA_slow[t]. All trailing (past-only)."""
    s = pd.Series(close.astype(float))
    prior_high = s.rolling(brk).max().shift(1)          # max of the PRIOR brk closes (strictly past)
    sf = s.rolling(fast).mean()
    ssl = s.rolling(slow).mean()
    fire = (s > prior_high) & (sf > ssl)
    return fire.fillna(False).to_numpy()


# ===========================================================================
# PRICE-ORACLE moves: non-overlapping forward windows where the best long entry->exit >= move_thr.
#   We scan forward, building each move-window greedily from the last consumed bar; inside a window of
#   length `mw` bars the available move = the best (sell_high / buy_low - 1) achievable buying a low close
#   and selling a later high close (perfect long timing). A window QUALIFIES as an oracle move iff that
#   available move >= move_thr. Windows are NON-OVERLAPPING (each bar belongs to at most one move).
# ===========================================================================
def _best_long_move(opens: np.ndarray, lo: int, hi: int) -> float:
    """Best achievable long (buy a low open, sell a LATER high open) within bars [lo, hi]. open->open so it
    is REALIZABLE at next-bar-open fills (no intrabar high/low look-ahead). Returns gross fraction >= 0."""
    seg = opens[lo:hi + 1]
    if seg.size < 2:
        return 0.0
    run_min = np.minimum.accumulate(seg)               # cheapest buy available up to each bar
    gains = seg / run_min - 1.0                         # sell at each bar vs cheapest prior buy
    return float(max(0.0, gains.max()))


def detect_oracle_moves(opens: np.ndarray, *, move_window: int, move_thr: float,
                        first: int, last: int) -> list[tuple[int, int, float]]:
    """Return non-overlapping (lo, hi, price_oracle_move) windows in [first, last] whose best long move
    >= move_thr. Fixed-length scan: step a `move_window`-bar window; if it qualifies, record it and skip
    past it (non-overlapping); else advance by 1. The price_oracle_move is the perfect-timing ceiling."""
    moves: list[tuple[int, int, float]] = []
    t = first
    while t + move_window <= last:
        lo, hi = t, t + move_window
        mv = _best_long_move(opens, lo, hi)
        if mv >= move_thr:
            moves.append((lo, hi, mv))
            t = hi + 1                                   # non-overlapping
        else:
            t += 1
    return moves


# ===========================================================================
# INDICATOR-ORACLE: best CAUSAL adaptive-MA config whose signalled long lands inside a move-window.
#   Each config is an SMA/EMA golden->death cross. Past-only crosses (uses closes <= t). For a move-window
#   [lo,hi], the indicator-oracle available move = the MAX over configs of the open->open return of the
#   config's long position clipped to the window: enter at the first golden-cross fill at/after lo (open of
#   cross_bar+1), exit at the death-cross fill (or window end), GROSS (the family ceiling). Hindsight is
#   ONLY in choosing the best config; each config's signal is causal.
# ===========================================================================
def _sma(x: np.ndarray, w: int) -> np.ndarray:
    n = len(x); out = np.full(n, np.nan)
    if n < w:
        return out
    c = np.cumsum(np.insert(x, 0, 0.0))
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    """Causal EMA seeded with the SMA of the first `span` points (output[t] depends only on x[0..t]).
    Vectorized via scipy.signal.lfilter when available (fast for the 1e5-1e6-bar fine cadences), else a
    Python loop. NaN until the seed window (t < span-1)."""
    n = len(x); out = np.full(n, np.nan)
    if n < span:
        return out
    a = 2.0 / (span + 1.0)
    seed = x[:span].mean()
    try:
        from scipy.signal import lfilter
        # recursive y[t] = a*x[t] + (1-a)*y[t-1], with y seeded at index span-1 = seed.
        tail = x[span:]
        if tail.size:
            zi = np.array([(1.0 - a) * seed])             # filter state so first output = a*x[span]+(1-a)*seed
            filt = lfilter([a], [1.0, -(1.0 - a)], tail, zi=zi)[0]
            out[span:] = filt
        out[span - 1] = seed
        return out
    except Exception:
        prev = seed; out[span - 1] = prev
        for t in range(span, n):
            prev = a * x[t] + (1.0 - a) * prev
            out[t] = prev
        return out


def _ma_cross_segments(spread: np.ndarray) -> list[tuple[int, int]]:
    """Return list of (golden_bar, death_bar_or_end) long segments from a fast-slow spread (NaN warmup).
    golden at t: spread[t-1]<=0 and spread[t]>0; death at t: spread[t-1]>0 and spread[t]<=0. A segment runs
    from a golden cross to the NEXT death cross (or to the series end if still open). Crosses are detected
    vectorized; the golden->death pairing is a single ordered walk (a position opens at a golden cross and
    closes at the first later death cross; intervening golden crosses are ignored while already long)."""
    n = len(spread)
    if n < 2:
        return []
    prev, cur = spread[:-1], spread[1:]
    valid = np.isfinite(prev) & np.isfinite(cur)
    golden = (np.flatnonzero(valid & (prev <= 0.0) & (cur > 0.0)) + 1).tolist()   # +1 -> index t
    death = (np.flatnonzero(valid & (prev > 0.0) & (cur <= 0.0)) + 1).tolist()
    segs: list[tuple[int, int]] = []
    gi, di, ng, nd = 0, 0, len(golden), len(death)
    while gi < ng:
        g = golden[gi]
        while di < nd and death[di] <= g:    # first death strictly after this golden
            di += 1
        if di < nd:
            d = death[di]
            segs.append((g, d))
            # advance past any golden crosses that occur before this exit (already long)
            while gi < ng and golden[gi] <= d:
                gi += 1
            di += 1
        else:
            segs.append((g, n - 1))          # still open at series end
            break
    return segs


def _build_oracle_ma_grid():
    grid = []
    for fam in ("SMA", "EMA"):
        for f in ORACLE_FAST:
            for s in ORACLE_SLOW:
                if f < s:
                    grid.append((fam, f, s))
    return grid


_ORACLE_GRID = _build_oracle_ma_grid()


class OracleMASegments:
    """Precompute (ONCE per series) the long cross-segments of EVERY SMA/EMA(f,s) config in the oracle grid.

    Computing the 18 MA pairs over the FULL series once is O(grid * n); per-window queries then just clip
    the precomputed segments. This avoids the O(moves * grid * n) recomputation that made fine cadences
    (15m/dollar, ~1e5-1e6 bars) intractable. Each config's segments are CAUSAL (crosses use closes<=t over
    the FULL series; a segment is a (golden_bar, death_bar_or_end) long interval)."""

    def __init__(self, opens: np.ndarray, close: np.ndarray):
        self.opens = opens
        self.n = len(opens)
        self.segs_by_cfg: list[np.ndarray] = []   # each: (k,2) int array of (golden, death/end) bars
        for (fam, f, s) in _ORACLE_GRID:
            ma_f = _sma(close, f) if fam == "SMA" else _ema(close, f)
            ma_s = _sma(close, s) if fam == "SMA" else _ema(close, s)
            spread = ma_f - ma_s
            segs = _ma_cross_segments(spread)
            self.segs_by_cfg.append(np.asarray(segs, dtype=int).reshape(-1, 2) if segs
                                    else np.empty((0, 2), dtype=int))

    def move(self, lo: int, hi: int) -> float:
        """Best (over the grid) GROSS open->open long return realizable CLIPPED to bars [lo,hi]. For each
        config's segment overlapping [lo,hi]: enter at open after the golden cross (or at lo if the cross is
        earlier -- the position is already open entering the window, so the realizable in-window entry is
        max(golden, lo-1)+1), exit at open after min(death, hi). Family ceiling."""
        opens = self.opens
        n = self.n
        best = 0.0
        for segs in self.segs_by_cfg:
            if segs.shape[0] == 0:
                continue
            # segments overlapping [lo, hi]: golden <= hi AND death >= lo
            mask = (segs[:, 0] <= hi) & (segs[:, 1] >= lo)
            for g, d in segs[mask]:
                entry_bar = max(int(g), lo - 1)          # already-open positions enter the window at lo
                ef = entry_bar + 1                       # fill at open after the (clipped) entry
                xf = min(int(d), hi) + 1                  # fill at open after death cross / window end
                if ef >= n or xf >= n or xf <= ef:
                    continue
                r = opens[xf] / opens[ef] - 1.0
                if r > best:
                    best = r
        return float(max(0.0, best))


# ===========================================================================
# REALIZED capture of the signal's single-position non-overlapping book INSIDE each move-window.
#   For a move-window [lo,hi]: take the signal's fired bars whose ENTRY (i+1) lands in [lo, hi], run the
#   non-overlapping book restricted to that window, and SUM the per-trade net (cost-charged) realized
#   return. That sum is the numerator; the oracle move is the denominator -> the capture ratio.
# ===========================================================================
def realized_net_in_window(opens: np.ndarray, fire_bars: np.ndarray, lo: int, hi: int, horizon: int,
                           cost: float, last_valid: int) -> tuple[float, int]:
    """Sum of per-trade net returns of the single-position non-overlapping book whose ENTRY bars (i+1) fall
    in [lo, hi]. Returns (sum_net, n_trades). Additive (not compounded) so it is comparable to the additive
    oracle available-move and cannot explode; a single position never overlaps another by construction.

    fire_bars MUST be sorted ascending (the caller passes a window-sliced sorted array); we slice the fired
    bars whose entry (i+1) is in [lo,hi] -> i in [lo-1, hi-1] via searchsorted (O(log n), not O(n_fires))."""
    if fire_bars.size == 0:
        return 0.0, 0
    a = np.searchsorted(fire_bars, lo - 1, side="left")
    b = np.searchsorted(fire_bars, min(hi - 1, last_valid), side="right")
    cand = fire_bars[a:b]
    if cand.size == 0:
        return 0.0, 0
    bars, nets = nonoverlap_book(opens, cand, horizon, cost, last_valid)
    return float(np.sum(nets)), int(nets.size)


# ===========================================================================
# Evaluate ONE (breakout, fast, slow, horizon) config on ONE series -> capture per window per oracle.
# ===========================================================================
@dataclass
class CaptureResult:
    asset: str
    cadence: str
    brk: int
    fast: int
    slow: int
    horizon: int
    cost_name: str
    # per-window pooled capture stats
    n_moves: dict = field(default_factory=dict)             # qualifying oracle moves per window
    price_oracle_sum: dict = field(default_factory=dict)    # sum of price-oracle available move per window
    indic_oracle_sum: dict = field(default_factory=dict)    # sum of indicator-oracle available move
    realized_sum: dict = field(default_factory=dict)        # sum of signal realized net per window
    n_trades: dict = field(default_factory=dict)            # signal trades inside moves per window
    capture_price: dict = field(default_factory=dict)       # realized_sum / price_oracle_sum per window
    capture_indic: dict = field(default_factory=dict)       # realized_sum / indic_oracle_sum per window
    median_ratio_price: dict = field(default_factory=dict)  # median per-move ratio (price oracle)
    trainval_capture_price: float = 0.0                     # the SELECTION objective (no UNSEEN leak)


def precompute_oracle_moves(opens, close, wlab, purge_mask, oracle_segs, *, move_window, move_thr,
                            last_valid) -> dict:
    """Per window, the list of oracle MOVES (lo, hi, price_move, indic_move). This depends ONLY on the price
    series + window split (NOT on any signal config), so it is computed ONCE per (asset,cadence) and reused
    across all 27 signal configs AND both costs -- the key speed fix (it was the O(moves*configs) hotspot)."""
    moves_by_window: dict[str, list] = {}
    for w in WINDOWS:
        w_bars = np.flatnonzero((wlab == w) & (~purge_mask))
        if w_bars.size < move_window + 2:
            moves_by_window[w] = []
            continue
        first, last = int(w_bars.min()), min(int(w_bars.max()), last_valid)
        raw = detect_oracle_moves(opens, move_window=move_window, move_thr=move_thr, first=first, last=last)
        moves_by_window[w] = [(lo, hi, p_move, oracle_segs.move(lo, hi)) for (lo, hi, p_move) in raw]
    return moves_by_window


def evaluate_capture(df: pd.DataFrame, wlab: np.ndarray, purge_mask: np.ndarray, *,
                     brk: int, fast: int, slow: int, horizon: int, cost: float, cost_name: str,
                     move_window: int, move_thr: float, asset: str, cadence: str,
                     oracle_segs: "OracleMASegments | None" = None,
                     moves_by_window: dict | None = None) -> CaptureResult:
    opens = df["open"].to_numpy(float)
    close = df["close"].to_numpy(float)
    n = len(opens)
    last_valid = n - 2 - horizon
    if last_valid < 2:
        return CaptureResult(asset, cadence, brk, fast, slow, horizon, cost_name)
    if oracle_segs is None:
        oracle_segs = OracleMASegments(opens, close)
    if moves_by_window is None:
        moves_by_window = precompute_oracle_moves(opens, close, wlab, purge_mask, oracle_segs,
                                                  move_window=move_window, move_thr=move_thr,
                                                  last_valid=last_valid)

    fire = continuation_signal(close, brk=brk, fast=fast, slow=slow) & (~purge_mask)
    fire_idx = np.flatnonzero(fire)

    res = CaptureResult(asset, cadence, brk, fast, slow, horizon, cost_name)
    for w in WINDOWS:
        moves = moves_by_window.get(w, [])
        if not moves:
            res.n_moves[w] = 0; res.price_oracle_sum[w] = 0.0; res.indic_oracle_sum[w] = 0.0
            res.realized_sum[w] = 0.0; res.n_trades[w] = 0
            res.capture_price[w] = None; res.capture_indic[w] = None; res.median_ratio_price[w] = None
            continue
        first = moves[0][0]
        last = moves[-1][1]
        p_sum = i_sum = r_sum = 0.0
        n_tr = 0
        per_move_ratios: list[float] = []
        fire_in_w = fire_idx[(fire_idx >= first) & (fire_idx <= last)]
        for (lo, hi, p_move, i_move) in moves:
            r_net, n = realized_net_in_window(opens, fire_in_w, lo, hi, horizon, cost, last_valid)
            p_sum += p_move
            i_sum += i_move
            r_sum += r_net
            n_tr += n
            if p_move > 1e-9:
                per_move_ratios.append(r_net / p_move)
        res.n_moves[w] = len(moves)
        res.price_oracle_sum[w] = round(p_sum, 6)
        res.indic_oracle_sum[w] = round(i_sum, 6)
        res.realized_sum[w] = round(r_sum, 6)
        res.n_trades[w] = n_tr
        res.capture_price[w] = round(r_sum / p_sum, 4) if p_sum > 1e-9 else None
        res.capture_indic[w] = round(r_sum / i_sum, 4) if i_sum > 1e-9 else None
        res.median_ratio_price[w] = round(float(np.median(per_move_ratios)), 4) if per_move_ratios else None

    # SELECTION objective: pooled price-oracle capture on TRAIN+VAL (no UNSEEN leak)
    tv_p = (res.price_oracle_sum.get("TRAIN", 0.0) + res.price_oracle_sum.get("VAL", 0.0))
    tv_r = (res.realized_sum.get("TRAIN", 0.0) + res.realized_sum.get("VAL", 0.0))
    res.trainval_capture_price = round(tv_r / tv_p, 4) if tv_p > 1e-9 else -999.0
    return res


# ===========================================================================
# Grid over one (asset, cadence, cost): pick the TRAIN+VAL-best config, report its held-out capture.
# ===========================================================================
def run_grid(df: pd.DataFrame, *, asset: str, cadence: str, cost: float, cost_name: str,
             move_window: int, move_thr: float, purge: int = 40,
             wlab: np.ndarray | None = None, purge_mask: np.ndarray | None = None,
             oracle_segs: "OracleMASegments | None" = None,
             moves_by_window: dict | None = None
             ) -> tuple[list[CaptureResult], CaptureResult]:
    n = len(df)
    if wlab is None or purge_mask is None:
        wlab, purge_mask = index_windows(n, purge=purge)
    opens = df["open"].to_numpy(float); close = df["close"].to_numpy(float)
    # the indicator-oracle segments depend ONLY on the price series (not the signal config / cost) -> reuse.
    if oracle_segs is None:
        oracle_segs = OracleMASegments(opens, close)
    # the oracle MOVES (and their price+indicator available-move) are config/cost-INDEPENDENT -> precompute
    # ONCE per (asset,cadence) using the most-restrictive last_valid (max horizon) so every config sees the
    # SAME move set. This removes the O(moves*configs) hotspot (was ~100s/cost on 15m).
    if moves_by_window is None:
        last_valid_min = n - 2 - max(HORIZONS)
        moves_by_window = precompute_oracle_moves(opens, close, wlab, purge_mask, oracle_segs,
                                                  move_window=move_window, move_thr=move_thr,
                                                  last_valid=last_valid_min)
    results = []
    for brk, (fast, slow), h in product(BREAKOUTS, FAST_SLOW, HORIZONS):
        if not (fast < slow):
            continue
        r = evaluate_capture(df, wlab, purge_mask, brk=brk, fast=fast, slow=slow, horizon=h,
                             cost=cost, cost_name=cost_name, move_window=move_window, move_thr=move_thr,
                             asset=asset, cadence=cadence, oracle_segs=oracle_segs,
                             moves_by_window=moves_by_window)
        results.append(r)
    # SELECT on TRAIN+VAL capture; require >= 3 qualifying moves with trades on TRAIN+VAL for stability.
    def _tv_trades(r):
        return r.n_trades.get("TRAIN", 0) + r.n_trades.get("VAL", 0)
    eligible = [r for r in results if _tv_trades(r) >= 3]
    pool = eligible or results
    best = max(pool, key=lambda r: r.trainval_capture_price)
    return results, best


# ===========================================================================
# Data loader (reuse ChimeraLoader -- the canonical strategy-facing API).
# ===========================================================================
# Cap the series at the most-recent MAX_BARS for tractability on the fine cadences (the EMA/segment build
# and the move scan are O(n)); the cap keeps the most RECENT bars, so the UNSEEN tail (the verdict segment)
# is fully preserved. The cap is reported in the artifact (n_bars_loaded vs n_bars_used) -- honest, not a gap.
# Dollar and dib cadences use a HIGHER cap so that UNSEEN has enough non-overlapping oracle moves (>=30):
#   dollar: ~1155 bars/day BTC; 150k bars = ~130 days (too little UNSEEN at 10%). Cap raised to 1_500_000.
#   dib:    ~48 bars/day BTC; full series ~108k bars already small; no cap needed (load all).
MAX_BARS = 150_000
MAX_BARS_BY_CADENCE = {"dollar": 1_500_000, "dib": 0}  # 0 = no cap (load full series)


def load_series(asset: str, cadence: str) -> tuple[pd.DataFrame, int]:
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "src" / "pipeline"))
    from pipeline.chimera_loader import ChimeraLoader
    g = ChimeraLoader().load(asset, cadence=cadence, features=["open", "high", "low", "close", "date",
                                                                "timestamp"])
    d = g.to_dict(as_series=False)
    # Use the ms-epoch `timestamp` column for a sortable integer index; `date` is day-resolution
    # (repeats for many intraday/dollar bars). Fall back to `date` if timestamp missing.
    if "timestamp" in d and d["timestamp"] is not None:
        ts_raw = np.asarray(d["timestamp"])
        if np.issubdtype(ts_raw.dtype, np.number) and ts_raw.dtype != object:
            dt = pd.to_datetime(ts_raw, unit="ms")
        else:
            dt = pd.to_datetime(np.asarray(d["date"]))
    else:
        raw = np.asarray(d["date"])
        dt = pd.to_datetime(raw, unit="ms") if np.issubdtype(raw.dtype, np.number) else pd.to_datetime(raw)
    df = pd.DataFrame({"date": dt, "open": np.asarray(d["open"], float),
                       "high": np.asarray(d["high"], float), "low": np.asarray(d["low"], float),
                       "close": np.asarray(d["close"], float)})
    n_loaded = len(df)
    cap = MAX_BARS_BY_CADENCE.get(cadence, MAX_BARS)
    if cap > 0 and n_loaded > cap:
        df = df.iloc[-cap:].reset_index(drop=True)   # keep the most-recent bars -> UNSEEN preserved
    return df, n_loaded


# move_window (bars) per cadence: spans a sensible economic horizon at each bar-type.
#   Time-based cadences: window = a few hours of price action (same rationale as before).
#   Dollar bars:  BTC ~90 bars/h, ETH ~144 bars/h -> 360 bars ~4h (BTC) to ~2.5h (ETH). Consistent 4h-class.
#   Dib bars:     BTC ~16 bars/h, ETH ~25 bars/h   -> 80 bars  ~5h (BTC) to ~3.2h (ETH). Same 4h-class.
#   The OLD dollar=64 was WRONG: each dollar bar aggregates ~1 min of trades; 64 bars = ~1 hour is fine for
#   BTC but only yields 1-3 oracle moves on UNSEEN (the statistical void). 360 yields 297+ on UNSEEN.
MOVE_WINDOW_BY_CADENCE = {"1d": 5, "4h": 12, "1h": 24, "30m": 48, "15m": 64,
                           "dollar": 360, "dib": 80}


# ===========================================================================
# The CAPTURE x CADENCE x COST grid -- the reframe verdict.
# ===========================================================================
def run_reframe(assets: list[str], cadences: list[str], *, move_thr: float = 0.02,
                out_tag: str = "reframe", move_window_override: int | None = None) -> dict:
    print("=" * 118)
    print("ORACLE-CAPTURE LAB -- the REFRAME: capture >= 25% of the ORACLE move (price + indicator), per "
          "cadence x cost")
    print(f"move = a forward window with a best-long price-oracle move >= {move_thr*100:.0f}%; "
          "single-position non-overlapping; UNSEEN-once")
    print("=" * 118)

    table_rows = []
    data_gaps = []
    for asset in assets:
        for cadence in cadences:
            try:
                df, n_loaded = load_series(asset, cadence)
            except Exception as e:
                msg = f"{asset} {cadence}: LOAD ERROR {type(e).__name__}: {str(e)[:80]}"
                print(f"  [GAP] {msg}")
                data_gaps.append(msg)
                continue
            if len(df) < 400:
                msg = f"{asset} {cadence}: only {len(df)} bars (<400) -> SKIP"
                print(f"  [GAP] {msg}")
                data_gaps.append(msg)
                continue
            if n_loaded > len(df):
                print(f"  [CAP] {asset} {cadence}: loaded {n_loaded} bars, capped to most-recent "
                      f"{len(df)} (UNSEEN tail preserved)")
            mw = move_window_override if move_window_override is not None else MOVE_WINDOW_BY_CADENCE.get(cadence, 12)
            # build the split + indicator-oracle segments + oracle MOVES ONCE per (asset,cadence); reuse
            # across both costs (moves are cost/config-independent).
            wlab, purge_mask = index_windows(len(df), purge=40)
            _opens = df["open"].to_numpy(float); _close = df["close"].to_numpy(float)
            oracle_segs = OracleMASegments(_opens, _close)
            _last_valid_min = len(df) - 2 - max(HORIZONS)
            moves_by_window = precompute_oracle_moves(_opens, _close, wlab, purge_mask, oracle_segs,
                                                      move_window=mw, move_thr=move_thr,
                                                      last_valid=_last_valid_min)
            for cost_name, cost in COSTS.items():
                results, best = run_grid(df, asset=asset, cadence=cadence, cost=cost, cost_name=cost_name,
                                         move_window=mw, move_thr=move_thr,
                                         wlab=wlab, purge_mask=purge_mask, oracle_segs=oracle_segs,
                                         moves_by_window=moves_by_window)
                # held-out (UNSEEN) pooled capture for the TRAIN+VAL-selected config
                cap_p_un = best.capture_price.get("UNSEEN")
                cap_i_un = best.capture_indic.get("UNSEEN")
                cap_p_oos = best.capture_price.get("OOS")
                n_moves_un = best.n_moves.get("UNSEEN", 0)
                n_tr_un = best.n_trades.get("UNSEEN", 0)
                row = {
                    "asset": asset, "cadence": cadence, "cost": cost_name,
                    "move_window_bars": mw, "n_bars": len(df), "n_bars_loaded": n_loaded,
                    "cfg": f"brk{best.brk}/f{best.fast}/s{best.slow}/h{best.horizon}",
                    "tv_capture_price": best.trainval_capture_price,
                    "oos_capture_price": cap_p_oos,
                    "unseen_capture_price": cap_p_un,
                    "unseen_capture_indic": cap_i_un,
                    "unseen_n_moves": n_moves_un,
                    "unseen_n_trades": n_tr_un,
                    "unseen_price_oracle_sum": best.price_oracle_sum.get("UNSEEN"),
                    "unseen_realized_sum": best.realized_sum.get("UNSEEN"),
                    "clears_25pct_price": bool(cap_p_un is not None and cap_p_un >= 0.25),
                    "clears_25pct_indic": bool(cap_i_un is not None and cap_i_un >= 0.25),
                }
                table_rows.append(row)
                print(f"  {asset:8} {cadence:6} {cost_name:5} {row['cfg']:22} | "
                      f"TVcap_p={best.trainval_capture_price:>7} | "
                      f"OOScap_p={str(cap_p_oos):>8} | UNSEEN: cap_price={str(cap_p_un):>8} "
                      f"cap_indic={str(cap_i_un):>8} moves={n_moves_un:>4} trades={n_tr_un:>4} "
                      f">=25%price={row['clears_25pct_price']}")

    summary = _emit(table_rows, data_gaps, assets, cadences, move_thr, out_tag)
    return summary


def _emit(table_rows, data_gaps, assets, cadences, move_thr, out_tag) -> dict:
    # ---- the COST-CLIFF curve: median UNSEEN price-oracle capture by cadence, taker vs maker ----
    cadence_order = [c for c in ["1d", "4h", "1h", "30m", "15m", "dollar"] if c in cadences]
    cliff = {}
    for cad in cadence_order:
        cliff[cad] = {}
        for cost_name in COSTS:
            caps = [r["unseen_capture_price"] for r in table_rows
                    if r["cadence"] == cad and r["cost"] == cost_name and r["unseen_capture_price"] is not None]
            orc = [r["unseen_price_oracle_sum"] for r in table_rows
                   if r["cadence"] == cad and r["cost"] == cost_name and r["unseen_price_oracle_sum"] is not None]
            cliff[cad][cost_name] = {
                "median_unseen_capture_price": (round(float(np.median(caps)), 4) if caps else None),
                "n_series": len(caps),
                "median_unseen_price_oracle_sum": (round(float(np.median(orc)), 4) if orc else None),
            }

    print("\n" + "=" * 118)
    print("COST-CLIFF CURVE -- median UNSEEN price-oracle CAPTURE-RATE by cadence (does finer = RISE or COLLAPSE?)")
    print("-" * 118)
    print(f"  {'cadence':8} | {'taker cap':>10} {'taker orcSum':>13} | {'maker cap':>10} {'maker orcSum':>13}")
    for cad in cadence_order:
        tk = cliff[cad]["taker"]; mk = cliff[cad]["maker"]
        print(f"  {cad:8} | {str(tk['median_unseen_capture_price']):>10} "
              f"{str(tk['median_unseen_price_oracle_sum']):>13} | "
              f"{str(mk['median_unseen_capture_price']):>10} "
              f"{str(mk['median_unseen_price_oracle_sum']):>13}")

    # ---- verdict: any (cadence,cost) where median held-out capture clears 25%? ----
    clears = [(r["asset"], r["cadence"], r["cost"], r["unseen_capture_price"])
              for r in table_rows if r["clears_25pct_price"]]
    clears_indic = [(r["asset"], r["cadence"], r["cost"], r["unseen_capture_indic"])
                    for r in table_rows if r["clears_25pct_indic"]]
    cadence_cost_clears = []
    for cad in cadence_order:
        for cost_name in COSTS:
            med = cliff[cad][cost_name]["median_unseen_capture_price"]
            if med is not None and med >= 0.25:
                cadence_cost_clears.append((cad, cost_name, med))

    print("\n" + "=" * 118)
    print("VERDICT")
    print(f"  per-(asset,cadence,cost) cells clearing 25% of PRICE-oracle on UNSEEN : {len(clears)} / "
          f"{len(table_rows)}")
    if clears:
        for a, c, cn, v in clears:
            print(f"      {a} {c} {cn}: {v}")
    print(f"  per-cell clearing 25% of INDICATOR-oracle on UNSEEN                   : {len(clears_indic)}")
    print(f"  (cadence,cost) cells whose MEDIAN held-out capture clears 25%         : "
          f"{cadence_cost_clears if cadence_cost_clears else 'NONE'}")
    if data_gaps:
        print(f"  DATA GAPS                                                           : {len(data_gaps)}")
        for g in data_gaps:
            print(f"      {g}")

    summary = {
        "tag": out_tag,
        "reframe": "capture >= 25% of the ORACLE move (price + indicator), per cadence x cost; UNSEEN-once",
        "move_thr": move_thr,
        "assets": assets, "cadences": cadences,
        "costs": COSTS,
        "signal": "momentum-continuation: close>prior-brk-high AND SMA_fast>SMA_slow (past-only)",
        "accounting": "single-position non-overlapping; realized net summed per move-window vs oracle move",
        "selection": "TRAIN+VAL pooled price-oracle capture (UNSEEN never used to pick a config)",
        "cost_cliff_curve": cliff,
        "rows": table_rows,
        "data_gaps": data_gaps,
        "verdict": {
            "n_cells": len(table_rows),
            "n_cells_clear_25pct_price_UNSEEN": len(clears),
            "n_cells_clear_25pct_indicator_UNSEEN": len(clears_indic),
            "cadence_cost_median_clears_25pct": cadence_cost_clears,
        },
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACT_DIR / f"oracle_capture_{out_tag}.json"
    tmp = out_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    tmp.replace(out_path)
    print(f"\n  artifact: {out_path}")
    return summary


# ===========================================================================
# SELFTEST: two-sided soundness on synthetic data.
#   (A) a GENUINE-capture series: moves are dip->strong-rally; the continuation breakout enters early in
#       each rally and the single-position book captures a large fraction of the oracle move -> HIGH capture
#       that beats a no-skill control.
#   (B) a RANDOM trigger at the same firing rate -> ~0 capture (it does not preferentially sit inside the
#       up-leg of each oracle move).
#   PASS iff genuine capture is materially positive AND clearly exceeds the random control's capture.
# ===========================================================================
def _make_fixture(seed: int = 7, n: int = 5000) -> pd.DataFrame:
    """Repeated RALLY->DECLINE cycles: each move has a clean multi-bar up-leg a continuation breakout can
    ride, FOLLOWED by a symmetric down-leg of comparable magnitude. A momentum-continuation signal fires at
    the BREAKOUT (start of the up-leg) and captures much of it; a RANDOM trigger lands as often in the
    down-leg as the up-leg, so its realized net (cost-charged) is ~0 -> a near-zero no-skill control. The
    oracle move is the up-leg amplitude (a real best-long-timing ceiling), not a vol artifact (vol is flat).
    The down-leg makes random timing genuinely costly -- without it a rising series flatters any trigger."""
    rng = np.random.default_rng(seed)
    rets = np.empty(n)
    t = 0
    while t < n:
        # quiet pre-move drift
        q = int(rng.integers(6, 12)); q = min(q, n - t)
        rets[t:t + q] = rng.normal(0.0, 0.003, q); t += q
        if t >= n:
            break
        # a sharp dip (1-2 bars down) then a sustained RALLY up-leg (the capturable move)
        d = min(2, n - t); rets[t:t + d] = rng.normal(-0.008, 0.003, d); t += d
        if t >= n:
            break
        up = int(rng.integers(28, 40)); up = min(up, n - t)
        rets[t:t + up] = rng.normal(0.006, 0.003, up); t += up    # ~ +17-24% rally, long enough to ride
        if t >= n:
            break
        # a symmetric DECLINE leg of comparable magnitude -> random timing is genuinely penalized here
        dn = int(rng.integers(28, 40)); dn = min(dn, n - t)
        rets[t:t + dn] = rng.normal(-0.005, 0.003, dn); t += dn
    close = 100.0 * np.cumprod(1.0 + rets[:n])
    open_ = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.001, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.001, n)))
    dates = pd.date_range("2018-01-01", periods=n, freq="6h")
    return pd.DataFrame({"date": dates, "open": open_, "high": high, "low": low, "close": close})


def _pooled_capture(df, fire, *, horizon, cost, move_window, move_thr):
    """Pooled price-oracle capture over the WHOLE series for an arbitrary boolean fire array (selftest helper)."""
    opens = df["open"].to_numpy(float); close = df["close"].to_numpy(float)
    n = len(opens); last_valid = n - 2 - horizon
    fire_idx = np.flatnonzero(fire)
    fire_idx = fire_idx[fire_idx <= last_valid]
    moves = detect_oracle_moves(opens, move_window=move_window, move_thr=move_thr, first=move_window,
                                last=last_valid)
    p_sum = r_sum = 0.0; n_tr = 0
    for (lo, hi, p_move) in moves:
        fin = fire_idx[(fire_idx >= lo) & (fire_idx <= hi)]
        r_net, n = realized_net_in_window(opens, fin, lo, hi, horizon, cost, last_valid)
        p_sum += p_move; r_sum += r_net; n_tr += n
    return (r_sum / p_sum if p_sum > 1e-9 else None), len(moves), n_tr, p_sum, r_sum


def _selftest() -> int:
    print("=" * 96)
    print("[oracle_capture_lab --selftest] two-sided soundness (no market data)")
    print("=" * 96)
    df = _make_fixture()
    close = df["close"].to_numpy(float)
    horizon, cost, mw, thr = 10, TAKER, 24, 0.02

    # (A) GENUINE momentum-continuation signal -> rides the up-leg of each move.
    fire_sig = continuation_signal(close, brk=10, fast=10, slow=50)
    cap_a, nm_a, ntr_a, psum_a, rsum_a = _pooled_capture(df, fire_sig, horizon=horizon, cost=cost,
                                                         move_window=mw, move_thr=thr)
    # (B) RANDOM trigger at the SAME firing rate.
    rate = float(fire_sig.mean())
    rng = np.random.default_rng(123)
    fire_rand = rng.random(len(df)) < rate
    cap_b, nm_b, ntr_b, psum_b, rsum_b = _pooled_capture(df, fire_rand, horizon=horizon, cost=cost,
                                                         move_window=mw, move_thr=thr)

    print(f"\n  (A) CONTINUATION signal : capture={cap_a}  n_moves={nm_a} n_trades={ntr_a} "
          f"oracle_sum={round(psum_a,4)} realized_sum={round(rsum_a,4)}")
    print(f"  (B) RANDOM trigger      : capture={cap_b}  n_moves={nm_b} n_trades={ntr_b} "
          f"oracle_sum={round(psum_b,4)} realized_sum={round(rsum_b,4)}")

    # SOUNDNESS: genuine capture is materially positive AND clearly beats the random control.
    genuine_positive = (cap_a is not None and cap_a >= 0.15)
    beats_random = (cap_a is not None and cap_b is not None and cap_a > cap_b + 0.05)
    sane = (cap_a is None or abs(cap_a) < 5.0) and (cap_b is None or abs(cap_b) < 5.0)  # capture is a ratio
    ok = genuine_positive and beats_random and sane
    print("\n" + "-" * 96)
    print("SOUNDNESS (two-sided):")
    print(f"  GENUINE continuation capture >= 0.15 (materially captures the oracle move) : {genuine_positive}")
    print(f"  GENUINE capture > RANDOM capture + 0.05 (discriminates skill from chance)  : {beats_random}")
    print(f"  capture ratios sane (|cap| < 5)                                            : {sane}")
    print(f"\n[oracle_capture_lab --selftest] {'PASS' if ok else 'CHECK'}")
    return 0 if ok else 1


# ===========================================================================
# CLI
# ===========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="Oracle-capture lab (the reframe: capture % of the oracle move)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--asset", type=str, default=None)
    ap.add_argument("--cadence", type=str, default="4h")
    ap.add_argument("--assets", nargs="+", default=None)
    ap.add_argument("--cadences", nargs="+", default=["1d", "4h", "1h", "15m"])
    ap.add_argument("--move-thr", type=float, default=0.02)
    ap.add_argument("--move-window", type=int, default=None,
                    help="Override the per-cadence move_window (bars). If not set, uses "
                         "MOVE_WINDOW_BY_CADENCE (dollar=360, dib=80, 15m=64, 1h=24, 4h=12, 1d=5). "
                         "Dollar-bar appropriate default = 360 (~4h of BTC price action). The old "
                         "hardcoded 64 was WRONG for dollar bars (only 1-3 oracle moves on UNSEEN).")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()

    mw_override = args.move_window  # None -> use per-cadence table

    if args.assets:
        tag = "_".join(a.replace("USDT", "") for a in args.assets) + "__" + "_".join(args.cadences)
        run_reframe(args.assets, args.cadences, move_thr=args.move_thr, out_tag=tag,
                    move_window_override=mw_override)
        return 0

    if args.asset:
        run_reframe([args.asset], [args.cadence], move_thr=args.move_thr,
                    out_tag=args.asset.replace("USDT", "") + "_" + args.cadence,
                    move_window_override=mw_override)
        return 0

    print("specify --selftest, --asset SYM --cadence C, or --assets ... --cadences ...")
    return 1


if __name__ == "__main__":
    sys.exit(main())
