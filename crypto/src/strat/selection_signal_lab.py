"""src/strat/selection_signal_lab.py -- the MULTI-MA-ALIGNMENT SELECTION lab (real chimera data).

THE QUESTION (reframed correctly, per the run brief)
----------------------------------------------------
A parallel instance already established at 1d that ENTRY-TIMING is FUNGIBLE (within-window /
membership-matched null beats_null ~ 0 for momentum / ma_reclaim / donchian). So this lab does NOT
test entry timing. It tests the COMPLEMENTARY dimension -- SELECTION: does a past-only signal
preferentially put us in ABOVE-MEDIAN-forward-drift windows? i.e. "learn from the top-25% performers".

THE SELECTION SIGNAL (a multi-MA-ALIGNMENT filter, all PAST-ONLY)
----------------------------------------------------------------
    present[t] = (close[t] > SMA_long[t]) AND (SMA_fast[t] > SMA_mid[t] > SMA_long[t])
SMAs are trailing means of `close` (no centering, shifted so the value at bar t uses closes <= t).
present[t] = "the trend stack is aligned-up at the close of bar t" -> the setup is CONFIRMED here.

THE FIXED-SIZE / SINGLE-POSITION BASIS (the 2026-06-09 fix -- READ THIS)
-----------------------------------------------------------------------
Aligned bars are CONSECUTIVE / OVERLAPPING (a trend stays aligned for dozens of bars in a row). You
CANNOT hold N overlapping full-reinvest positions under a fixed-size constraint, so geometric-compounding
EVERY fired bar's fixed-horizon trade (the old prod(1+net)-1 over ~445 overlapping bars) is INVALID --
it is ~(1+r)^445 = an astronomical explosion that also poisons the null statistics. The lab is therefore
SELF-CONTAINED and OVERLAP-ROBUST: the SIGNAL's realized return is a SINGLE-POSITION, NON-OVERLAPPING
SEQUENTIAL BOOK -- walk the fired bars in time order, take a trade ONLY when flat (its entry bar is at or
after the prior trade's exit bar), enter opens[i+1], exit opens[i+1+h]; after taking a trade at i the next
entry must be > i+h. Compound ONLY those non-overlapping trades. SIGNAL, ORACLE, and BUY&HOLD are ALL put
on this SAME single-position basis so they are comparable.

THE FOUR MANDATORY HONEST CONTROLS (the whole point) -- all SELF-CONTAINED, all overlap-robust
----------------------------------------------------------------------------------------------
 1. SELECTION null (in-lab): random NON-OVERLAPPING window picks matched on COUNT (the same number of
    non-overlapping trades the signal actually took in that segment), HORIZON, and the past-only REGIME
    bin -> Monte-Carlo n_books draws -> null distribution of the SAME non-overlapping-sequential-compound
    statistic -> p_value + z + edge_pp. Beating it (p<0.05, real>null) = the signal's CHOICE of windows
    adds compound. ALSO report MEAN net-per-trade selected-vs-random (E[net|selected]-E[net|random]) as
    the overlap-robust secondary selection-skill stat (it cannot explode).
 2. BUY & HOLD (beta): does the single-position compound beat passive buy&hold over the SAME UNSEEN
    segment? A filter that only sidesteps drawdowns is the BEAR-ABSTENTION trap, not wealth-add.
 3. ORACLE ceiling: the GREEDY best-first NON-OVERLAPPING positive picks -> the realizable single-position
    hindsight ceiling (same basis) -> the honest gap between the realizable signal and the ceiling.
 4. PBO (pbo_cscv) over the FULL grid family -> Probability of Backtest Overfitting. SHIP needs < 0.10.

DISCIPLINE (no A5 leak): TRAIN+VAL is the only pool used to PICK a config. UNSEEN is read ONCE, only to
report the chosen config. We never argmax over UNSEEN. The selection-best config is chosen on TRAIN+VAL
compound; its UNSEEN numbers + UNSEEN buy&hold + UNSEEN selection-null are the verdict.

THE SPLIT: per-(asset,cadence) bar-count quantiles 50/20/20/10 (TRAIN/VAL/OOS/UNSEEN) with a purge gap
of `purge` bars dropped at each boundary (the trigger column is past-only; the gap guards the rolling-SMA
normalization from straddling a boundary). Dollar bars and 1d bars get their OWN quantile dates so the
fractions are honoured per series (dollar-bar date density != 1d).

RWYB:
    python src/strat/selection_signal_lab.py --selftest         # synthetic two-sided soundness (no data)
    python src/strat/selection_signal_lab.py --asset BTCUSDT --cadence 1d   # one (asset,cadence)
    python src/strat/selection_signal_lab.py --universe u10 --cadences 1d dollar   # the full grid (slow)
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
WINDOWS = ["TRAIN", "VAL", "OOS", "UNSEEN"]
HELD = ["OOS", "UNSEEN"]
TAKER = 0.0024
ARTIFACT_DIR = ROOT / "runs" / "strat"

# Reuse ONLY the window/regime labelling + PBO primitive. The selection null is implemented INSIDE this
# lab (overlap-robust, single-position) -- we deliberately DO NOT import window_selection_metric, whose
# null geometric-compounds every overlapping firing (the explosion this file fixes).
try:
    from .within_window_capture_proxy import Windows, past_only_regime_bins
    from .pbo_cscv import pbo_cscv
except ImportError:  # run as a script
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from within_window_capture_proxy import Windows, past_only_regime_bins
    from pbo_cscv import pbo_cscv

U10 = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
       "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]

# The grid (per the brief)
SMA_LONGS = [50, 100, 200]
FAST_MID = [(10, 20), (10, 50), (20, 50)]   # (fast, mid) pairs; require fast < mid < long
HORIZONS = [5, 10, 20]


# ---------------------------------------------------------------------------
# Split: per-series bar-count quantile dates 50/20/20/10 with a purge gap.
# ---------------------------------------------------------------------------
def quantile_windows(dates: pd.Series, purge: int = 40) -> tuple[Windows, np.ndarray]:
    """Return a Windows (date boundaries at the 50/70/90 bar-count quantiles) + a purge mask.

    purge_mask[i] = True means bar i is INSIDE a purge gap (dropped from labelled use). We carve `purge`
    bars on EACH side of every boundary so a rolling-SMA / regime label cannot straddle a split.
    """
    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    n = len(dates)
    i50, i70, i90 = int(0.50 * n), int(0.70 * n), int(0.90 * n)
    win = Windows(train_end=str(dates.iloc[i50].date()),
                  val_end=str(dates.iloc[i70].date()),
                  oos_end=str(dates.iloc[i90].date()))
    purge_mask = np.zeros(n, dtype=bool)
    for b in (i50, i70, i90):
        lo, hi = max(0, b - purge), min(n, b + purge)
        purge_mask[lo:hi] = True
    return win, purge_mask


# ---------------------------------------------------------------------------
# The SELECTION signal: trend-stack alignment, all PAST-ONLY.
# ---------------------------------------------------------------------------
def alignment_signal(df: pd.DataFrame, *, fast: int, mid: int, long: int) -> pd.Series:
    """present[t] = (close>SMA_long) AND (SMA_fast>SMA_mid>SMA_long). Trailing SMAs (past-only)."""
    close = df["close"].astype(float)
    sf = close.rolling(fast).mean()
    sm = close.rolling(mid).mean()
    sl = close.rolling(long).mean()
    present = (close > sl) & (sf > sm) & (sm > sl)
    return present.fillna(False)


# ---------------------------------------------------------------------------
# Realized per-entry net return: enter opens[i+1], exit opens[i+1+horizon], one round-trip cost.
# ---------------------------------------------------------------------------
def _net_for_entry(opens: np.ndarray, i: int, horizon: int, cost: float) -> float:
    ef = i + 1
    xf = ef + horizon
    return opens[xf] / opens[ef] - 1.0 - cost


def _compound_pct(nets) -> float:
    """Geometric compound (%) of a sequence of per-trade net returns. SAFE here because callers only ever
    pass NON-OVERLAPPING trades (a single-position book), so this cannot explode on overlapping firings."""
    nets = np.asarray(nets, float)
    return float((np.prod(1.0 + nets) - 1.0) * 100) if nets.size else 0.0


# ---------------------------------------------------------------------------
# THE SINGLE-POSITION, NON-OVERLAPPING SEQUENTIAL BOOK (the overlap-robust core).
#   Walk fired bars in time order; take a trade ONLY when flat (entry bar >= prior trade's exit bar).
#   enter=opens[i+1], exit=opens[i+1+h]; after taking a trade at i, the next entry must be > i+h.
# ---------------------------------------------------------------------------
def nonoverlap_book(opens: np.ndarray, fire_bars: np.ndarray, horizon: int, cost: float,
                    last_valid: int) -> tuple[list[int], np.ndarray]:
    """Return (taken_entry_bars, taken_nets) for the single-position non-overlapping book over fire_bars.

    A trade at bar i occupies entry bar i+1 .. exit bar i+1+h. The position is FLAT again only at exit, so
    the NEXT fired bar we can act on must satisfy i_next > i + h (its entry bar i_next+1 > exit bar i+1+h).
    """
    bars = np.sort(np.asarray(fire_bars, dtype=int))
    taken_bars: list[int] = []
    taken_nets: list[float] = []
    next_free = -1  # the earliest bar index we are allowed to enter on
    for i in bars:
        if i < next_free or i > last_valid:
            continue
        taken_bars.append(int(i))
        taken_nets.append(_net_for_entry(opens, int(i), horizon, cost))
        next_free = int(i) + horizon + 1  # next entry must be > i+h
    return taken_bars, np.asarray(taken_nets, float)


def nonoverlap_compound_pct(opens: np.ndarray, fire_bars: np.ndarray, horizon: int, cost: float,
                            last_valid: int) -> tuple[float, int, np.ndarray]:
    """(compound_pct, n_trades, taken_nets) for the single-position non-overlapping book."""
    _, nets = nonoverlap_book(opens, fire_bars, horizon, cost, last_valid)
    return _compound_pct(nets), int(nets.size), nets


# ---------------------------------------------------------------------------
# BUY & HOLD (beta) over a window -- single position, open-to-open, one round-trip cost, held passively
# from the window's first valid entry's fill to its last valid exit's fill. Same single-position basis.
# ---------------------------------------------------------------------------
def buy_hold_compound_pct(opens: np.ndarray, bar_idx: np.ndarray, horizon: int, cost: float) -> float:
    if bar_idx.size < 1:
        return 0.0
    first, last = int(bar_idx.min()), int(bar_idx.max())
    ef = first + 1
    xf = last + 1 + horizon
    if xf >= len(opens) or ef >= len(opens):
        xf = min(xf, len(opens) - 1)
    return float((opens[xf] / opens[ef] - 1.0 - cost) * 100)


# ---------------------------------------------------------------------------
# ORACLE ceiling -- the REALIZABLE single-position ceiling: greedy best-first NON-OVERLAPPING positive
# picks over the candidate entry bars. Same single-position / non-overlapping basis as the signal, so the
# capture ratio is apples-to-apples (NOT a top-quartile compound that secretly stacks overlapping trades).
# ---------------------------------------------------------------------------
def oracle_nonoverlap_pct(opens: np.ndarray, cand_bars: np.ndarray, horizon: int,
                          cost: float, last_valid: int) -> float:
    """Greedy hindsight: repeatedly take the highest-net still-available candidate entry, then block the
    [entry .. exit] band it occupies (no overlap), keep only positive-net picks. The realizable
    single-position compound ceiling -- what perfect SELECTION (with perfect non-overlapping scheduling)
    would have compounded."""
    cand = np.array([int(i) for i in cand_bars if int(i) <= last_valid], dtype=int)
    if cand.size == 0:
        return 0.0
    nets = np.array([_net_for_entry(opens, int(i), horizon, cost) for i in cand], dtype=float)
    order = np.argsort(-nets)  # best-net first
    occupied = np.zeros(int(cand.max()) + horizon + 2, dtype=bool)
    taken_nets: list[float] = []
    for k in order:
        i = int(cand[k])
        if nets[k] <= 0:
            break  # only positive picks help a long-only single-position book
        lo, hi = i, i + horizon  # the band i .. i+h must be free (entry i+1 .. exit i+1+h)
        if occupied[lo:hi + 1].any():
            continue
        occupied[lo:hi + 1] = True
        taken_nets.append(float(nets[k]))
    return _compound_pct(taken_nets)


# ---------------------------------------------------------------------------
# THE IN-LAB SELECTION NULL (self-contained, overlap-robust, COUNT+HORIZON+REGIME matched).
# ---------------------------------------------------------------------------
def _draw_nonoverlap_set(rng, pool_by_bin: dict, target_bins: list, horizon: int,
                         max_tries: int = 60) -> np.ndarray | None:
    """Draw a NON-OVERLAPPING set of random entry bars matched on COUNT (len(target_bins)) and REGIME bin
    (target_bins is the multiset of regime bins of the signal's actual taken trades). Each draw must not
    overlap an already-drawn band [e .. e+h]. Returns the chosen bar indices. If a given target slot cannot
    be placed without overlap after max_tries (the segment is crowded), that slot is SKIPPED (best-effort
    count) rather than aborting the whole book -- this keeps the null distribution well-defined for dense
    triggers; the (rare) shortfall makes the null slightly EASIER to beat, so it is the conservative choice.
    Returns None only if NO slot at all could be placed (degenerate empty pools)."""
    chosen: list[int] = []
    occupied_lo_hi: list[tuple[int, int]] = []

    def _free(e: int) -> bool:
        for lo, hi in occupied_lo_hi:
            if e <= hi and e + horizon >= lo:
                return False
        return True

    for b in target_bins:
        pool = pool_by_bin.get(int(b))
        if pool is None or pool.size == 0:
            continue
        for _ in range(max_tries):
            e = int(rng.choice(pool))
            if _free(e):
                chosen.append(e)
                occupied_lo_hi.append((e, e + horizon))
                break
    if not chosen:
        return None
    return np.array(sorted(chosen), dtype=int)


def selection_null(opens: np.ndarray, taken_bars: list, taken_nets: np.ndarray, *, horizon: int,
                   cost: float, regime_bins: np.ndarray, valid_seg_bars: np.ndarray, last_valid: int,
                   n_books: int, seed: int) -> dict:
    """The SELF-CONTAINED selection null. The signal TOOK `m` non-overlapping trades at `taken_bars`
    (regime bins from `regime_bins`). The null draws, n_books times, a random NON-OVERLAPPING set of `m`
    entry bars from the SAME-window same-regime valid-entry pools (matched on COUNT + HORIZON + REGIME but
    NOT on membership), then forms the SAME single-position non-overlapping-sequential compound statistic
    (chronological order). Returns the compound null band (p/z/edge) AND the mean-net null band."""
    m = len(taken_bars)
    real_comp = _compound_pct(taken_nets)
    real_mean_net = float(np.mean(taken_nets)) if taken_nets.size else None
    out = {"n_trades": m, "real_compound_pct": round(real_comp, 4),
           "real_mean_net": (round(real_mean_net, 6) if real_mean_net is not None else None),
           "null_comp_p50": None, "null_comp_p95": None, "edge_pp": None, "p_value": None, "z": None,
           "beats_null": None, "null_mean_net_p50": None, "mean_net_edge": None, "mean_net_p_value": None,
           "n_books_effective": 0}
    if m == 0:
        return out

    # same-window same-regime pools of VALID entry bars (the non-membership baseline universe)
    seg = np.array([int(i) for i in valid_seg_bars if int(i) <= last_valid], dtype=int)
    pool_by_bin: dict[int, np.ndarray] = {}
    for b in np.unique(regime_bins[seg]) if seg.size else []:
        pool_by_bin[int(b)] = seg[regime_bins[seg] == int(b)]
    target_bins = [int(regime_bins[i]) for i in taken_bars]

    rng = np.random.default_rng(seed)
    comps: list[float] = []
    mean_nets: list[float] = []
    for _ in range(n_books):
        chosen = _draw_nonoverlap_set(rng, pool_by_bin, target_bins, horizon)
        if chosen is None:
            continue
        nets = np.array([_net_for_entry(opens, int(e), horizon, cost) for e in chosen], dtype=float)
        comps.append(_compound_pct(nets))           # chronological (chosen is sorted) single-position book
        mean_nets.append(float(np.mean(nets)))
    comps_a = np.asarray(comps, float)
    mean_a = np.asarray(mean_nets, float)
    out["n_books_effective"] = int(comps_a.size)
    if comps_a.size:
        p50, p95 = float(np.percentile(comps_a, 50)), float(np.percentile(comps_a, 95))
        out["null_comp_p50"] = round(p50, 4)
        out["null_comp_p95"] = round(p95, 4)
        out["edge_pp"] = round(real_comp - p50, 4)
        n_exceed = int(np.sum(comps_a >= real_comp))
        out["p_value"] = round((1 + n_exceed) / (1 + comps_a.size), 5)
        sd = float(comps_a.std(ddof=1)) if comps_a.size > 1 else 0.0
        out["z"] = round(float((real_comp - comps_a.mean()) / sd), 4) if sd > 0 else None
        out["beats_null"] = bool(real_comp > p95)
    if mean_a.size and real_mean_net is not None:
        out["null_mean_net_p50"] = round(float(np.percentile(mean_a, 50)), 6)
        out["mean_net_edge"] = round(real_mean_net - float(np.mean(mean_a)), 6)
        n_exceed_m = int(np.sum(mean_a >= real_mean_net))
        out["mean_net_p_value"] = round((1 + n_exceed_m) / (1 + mean_a.size), 5)
    return out


# ===========================================================================
# Evaluate ONE config on ONE (asset, cadence) df.
# ===========================================================================
@dataclass
class ConfigResult:
    asset: str
    cadence: str
    fast: int
    mid: int
    long: int
    horizon: int
    n_fire: dict                    # raw firings per window (overlapping)
    n_trades: dict                  # NON-OVERLAPPING single-position trades taken per window
    real_compound: dict             # single-position non-overlapping compound % per window
    mean_net: dict                  # mean net-per-trade (selected) per window
    buyhold_compound: dict          # passive compound % per window (same segment)
    sel_null: dict                  # in-lab SELECTION null (held-out OOS+UNSEEN): edge_pp,z,p,mean_net edge
    sel_null_unseen: dict           # in-lab SELECTION null on UNSEEN ONLY
    oracle: dict                    # realizable single-position oracle ceiling per window
    trainval_compound: float        # non-overlap compound on TRAIN+VAL (the SELECTION objective; no leak)
    unseen_series: np.ndarray = field(default=None)  # UNSEEN non-overlapping per-trade nets (PBO family)


def evaluate_config(df: pd.DataFrame, win: Windows, purge_mask: np.ndarray, *,
                    fast: int, mid: int, long: int, horizon: int, cost: float = TAKER,
                    n_regime_bins: int = 3, n_books: int = 2000, seed: int = 11,
                    asset: str = "", cadence: str = "") -> ConfigResult:
    df = df.reset_index(drop=True)
    opens = df["open"].to_numpy(float)
    dates = df["date"]
    n = len(opens)
    last_valid = n - 2 - horizon

    present = alignment_signal(df, fast=fast, mid=mid, long=long).to_numpy()
    # apply the purge gap: a firing inside a purge band is dropped (boundary hygiene)
    present = present & (~purge_mask)

    regime_bins = past_only_regime_bins(df, n_bins=n_regime_bins)
    wlab = np.array([win.label(dates.iloc[i]) for i in range(n)])

    fire_idx = np.array([int(t) for t in np.flatnonzero(present) if t <= last_valid], dtype=int)
    n_fire = {w: int(np.sum(wlab[fire_idx] == w)) for w in WINDOWS} if fire_idx.size else {w: 0 for w in WINDOWS}

    valid_all = np.array([i for i in range(0, last_valid + 1) if not purge_mask[i]], dtype=int)

    # ---- per-window SINGLE-POSITION NON-OVERLAPPING signal book + mean-net + selection null ----
    n_trades, real_compound, mean_net, buyhold_compound, oracle, sel_per_window = {}, {}, {}, {}, {}, {}
    taken_by_window = {}  # cache taken (bars, nets) per window for pooled stats
    for w in WINDOWS:
        fire_w = fire_idx[wlab[fire_idx] == w] if fire_idx.size else np.array([], int)
        seg = valid_all[wlab[valid_all] == w]
        taken_bars, taken_nets = nonoverlap_book(opens, fire_w, horizon, cost, last_valid)
        taken_by_window[w] = (taken_bars, taken_nets)
        real_compound[w] = round(_compound_pct(taken_nets), 4)
        n_trades[w] = int(taken_nets.size)
        mean_net[w] = round(float(np.mean(taken_nets)), 6) if taken_nets.size else None
        buyhold_compound[w] = round(buy_hold_compound_pct(opens, seg, horizon, cost), 4) if seg.size else 0.0
        oc = oracle_nonoverlap_pct(opens, seg, horizon, cost, last_valid) if seg.size else 0.0
        cap = (round(real_compound[w] / oc, 4) if oc not in (0.0,) else None)
        oracle[w] = {"oracle_nonoverlap_compound_pct": round(oc, 4), "signal_vs_oracle_ratio": cap}
        # per-window selection null (matched on COUNT+HORIZON+REGIME, non-overlapping)
        sel_per_window[w] = selection_null(opens, taken_bars, taken_nets, horizon=horizon, cost=cost,
                                           regime_bins=regime_bins, valid_seg_bars=seg, last_valid=last_valid,
                                           n_books=n_books, seed=seed)

    # ---- the SELECTION objective: non-overlap compound on TRAIN+VAL only (no UNSEEN leak) ----
    tv_bars, tv_nets = nonoverlap_book(
        opens, fire_idx[np.isin(wlab[fire_idx], ("TRAIN", "VAL"))] if fire_idx.size else np.array([], int),
        horizon, cost, last_valid)
    trainval_compound = round(_compound_pct(tv_nets), 4)

    # ---- SELECTION null on the HELD pool (OOS+UNSEEN), and on UNSEEN ONLY ----
    held_fire = fire_idx[np.isin(wlab[fire_idx], tuple(HELD))] if fire_idx.size else np.array([], int)
    held_seg = valid_all[np.isin(wlab[valid_all], tuple(HELD))]
    held_bars, held_nets = nonoverlap_book(opens, held_fire, horizon, cost, last_valid)
    sel_null = selection_null(opens, held_bars, held_nets, horizon=horizon, cost=cost,
                              regime_bins=regime_bins, valid_seg_bars=held_seg, last_valid=last_valid,
                              n_books=n_books, seed=seed)
    sel_null["n_windows"] = sel_null.get("n_trades")  # legacy key for the formatter
    sel_u = dict(sel_per_window["UNSEEN"]); sel_u["window"] = "UNSEEN"
    sel_u["n_windows"] = sel_u.get("n_trades")

    # ---- UNSEEN non-overlapping per-trade net series (for the PBO family matrix) ----
    u_bars, u_nets = taken_by_window["UNSEEN"]
    unseen_series = np.asarray(u_nets, float)

    return ConfigResult(asset=asset, cadence=cadence, fast=fast, mid=mid, long=long, horizon=horizon,
                        n_fire=n_fire, n_trades=n_trades, real_compound=real_compound, mean_net=mean_net,
                        buyhold_compound=buyhold_compound, sel_null=sel_null, sel_null_unseen=sel_u,
                        oracle=oracle, trainval_compound=trainval_compound, unseen_series=unseen_series)


# ===========================================================================
# Grid over one (asset, cadence): returns all ConfigResults + the TRAIN+VAL-selected best.
# ===========================================================================
def run_grid_for_series(df: pd.DataFrame, *, asset: str, cadence: str, purge: int = 40,
                        n_books: int = 1500, seed: int = 11) -> tuple[list[ConfigResult], ConfigResult]:
    win, purge_mask = quantile_windows(df["date"], purge=purge)
    results = []
    for long, (fast, mid), h in product(SMA_LONGS, FAST_MID, HORIZONS):
        if not (fast < mid < long):
            continue
        r = evaluate_config(df, win, purge_mask, fast=fast, mid=mid, long=long, horizon=h,
                            n_books=n_books, seed=seed, asset=asset, cadence=cadence)
        results.append(r)
    # SELECT on TRAIN+VAL only (no UNSEEN leak); require a minimum NON-OVERLAPPING trade count for stability.
    eligible = [r for r in results if (r.n_trades["TRAIN"] + r.n_trades["VAL"]) >= 5]
    pool = eligible or results
    best = max(pool, key=lambda r: r.trainval_compound)
    return results, best


# ===========================================================================
# PBO over the full grid family (T x N) using UNSEEN per-window net-return series.
# ===========================================================================
def grid_pbo(results: list[ConfigResult], S: int = 8) -> dict:
    """Build a (T x N) returns matrix from each config's UNSEEN NON-OVERLAPPING per-trade net series, each
    truncated to the common T = min usable length across configs, then run pbo_cscv. PBO answers: does
    picking the IS-best config produce OOS under-performers (selection-bias / backtest overfit)?"""
    series = [r.unseen_series for r in results if r.unseen_series is not None and r.unseen_series.size >= 4]
    if len(series) < 2:
        return {"pbo": None, "verdict": "INSUFFICIENT", "N": len(series),
                "note": "fewer than 2 configs with >=4 UNSEEN non-overlapping trades -> PBO undefined"}
    T = min(len(s) for s in series)
    if T < 2 * S:
        # shrink S to fit; S must stay even and >=4
        S = max(4, (T // 2) - ((T // 2) % 2))
    if T < 8 or S < 4:
        return {"pbo": None, "verdict": "INSUFFICIENT", "N": len(series), "T": T,
                "note": f"common UNSEEN length T={T} too small for CSCV"}
    M = np.column_stack([s[:T] for s in series])
    try:
        res = pbo_cscv(M, S=S)
    except ValueError as e:
        return {"pbo": None, "verdict": "ERROR", "N": len(series), "T": T, "note": str(e)}
    res["family_N"] = len(series)
    return res


# ===========================================================================
# Data loader (reuse ChimeraLoader -- the canonical strategy-facing API)
# ===========================================================================
def load_series(asset: str, cadence: str) -> pd.DataFrame:
    sys.path.insert(0, str(ROOT / "src"))
    from pipeline.chimera_loader import ChimeraLoader
    g = ChimeraLoader().load(asset, cadence=cadence)
    d = g.to_dict(as_series=False)
    raw = np.asarray(d["date"])
    dt = pd.to_datetime(raw, unit="ms") if np.issubdtype(raw.dtype, np.number) else pd.to_datetime(raw)
    return pd.DataFrame({"date": dt, "open": np.asarray(d["open"], float),
                         "high": np.asarray(d["high"], float), "low": np.asarray(d["low"], float),
                         "close": np.asarray(d["close"], float)})


# ===========================================================================
# Reporting
# ===========================================================================
def _fmt_best(r: ConfigResult) -> str:
    sn, su = r.sel_null, r.sel_null_unseen
    bh_u = r.buyhold_compound["UNSEEN"]
    rc_u = r.real_compound["UNSEEN"]
    beats_bh = (rc_u is not None and bh_u is not None and rc_u > bh_u)
    orc_u = r.oracle["UNSEEN"]["oracle_nonoverlap_compound_pct"]
    cap_u = r.oracle["UNSEEN"]["signal_vs_oracle_ratio"]
    return (f"{r.asset:9} {r.cadence:7} f{r.fast}/m{r.mid}/L{r.long} h{r.horizon:<3} | "
            f"TV_comp={r.trainval_compound:>8.2f}% | "
            f"SEL-null(OOS+UN): edge={sn.get('edge_pp')}pp z={sn.get('z')} p={sn.get('p_value')} "
            f"mnEdge={sn.get('mean_net_edge')} mnP={sn.get('mean_net_p_value')} "
            f"ntr={sn.get('n_trades')} beats={sn.get('beats_null')} | "
            f"UNSEEN: sig={rc_u:.2f}% bh={bh_u:.2f}% beats_bh={beats_bh} "
            f"orc={orc_u:.2f}% cap={cap_u} | SEL-null(UN): edge={su.get('edge_pp')}pp p={su.get('p_value')} "
            f"beats={su.get('beats_null')}")


def run_universe(assets: list[str], cadences: list[str], *, n_books: int = 1500, seed: int = 11,
                 out_tag: str = "u10_align") -> dict:
    print("=" * 110)
    print("MULTI-MA-ALIGNMENT SELECTION LAB -- universe grid (SELECTION-skill verdict, no UNSEEN leak)")
    print("=" * 110)
    all_results: list[ConfigResult] = []
    best_per_series: list[ConfigResult] = []
    rows_for_table = []
    for asset in assets:
        for cadence in cadences:
            try:
                df = load_series(asset, cadence)
            except Exception as e:
                print(f"  [SKIP] {asset} {cadence}: load error: {e}")
                continue
            if len(df) < 400:
                print(f"  [SKIP] {asset} {cadence}: only {len(df)} bars (<400)")
                continue
            results, best = run_grid_for_series(df, asset=asset, cadence=cadence, n_books=n_books, seed=seed)
            all_results.extend(results)
            best_per_series.append(best)
            print(f"\n  {asset} {cadence}: {len(df)} bars  TRAIN+VAL-selected best ->")
            print(f"    {_fmt_best(best)}")
            rows_for_table.append(best)

    # ---- PBO over the FULL grid family (all configs across all series, UNSEEN series) ----
    pbo = grid_pbo(all_results, S=8)

    # ---- aggregate UNSEEN verdict over the per-series best configs ----
    n_beats_null_unseen = sum(1 for r in best_per_series if r.sel_null_unseen.get("beats_null"))
    n_beats_bh_unseen = 0
    for r in best_per_series:
        rc, bh = r.real_compound["UNSEEN"], r.buyhold_compound["UNSEEN"]
        if rc is not None and bh is not None and rc > bh:
            n_beats_bh_unseen += 1
    n_beats_both = 0
    for r in best_per_series:
        rc, bh = r.real_compound["UNSEEN"], r.buyhold_compound["UNSEEN"]
        beats_bh = rc is not None and bh is not None and rc > bh
        if r.sel_null_unseen.get("beats_null") and beats_bh:
            n_beats_both += 1

    summary = {
        "tag": out_tag,
        "n_series": len(best_per_series),
        "grid": {"sma_longs": SMA_LONGS, "fast_mid": FAST_MID, "horizons": HORIZONS,
                 "total_configs_evaluated": len(all_results)},
        "unseen_aggregate": {
            "n_series": len(best_per_series),
            "n_beats_selection_null_UNSEEN": n_beats_null_unseen,
            "n_beats_buyhold_UNSEEN": n_beats_bh_unseen,
            "n_beats_BOTH_UNSEEN": n_beats_both,
        },
        "grid_pbo": pbo,
        "best_per_series": [_result_to_dict(r) for r in best_per_series],
        "config": {"cost": TAKER, "split": "50/20/20/10 per-series bar-count quantiles, purge=40",
                   "selection_objective": "TRAIN+VAL compound (UNSEEN never used to pick a config)",
                   "n_books": n_books, "seed": seed},
    }
    print("\n" + "=" * 110)
    print("UNSEEN AGGREGATE (per-series TRAIN+VAL-selected best):")
    print(f"  series                         : {len(best_per_series)}")
    print(f"  beats SELECTION null on UNSEEN : {n_beats_null_unseen}/{len(best_per_series)}")
    print(f"  beats BUY&HOLD on UNSEEN       : {n_beats_bh_unseen}/{len(best_per_series)}")
    print(f"  beats BOTH on UNSEEN           : {n_beats_both}/{len(best_per_series)}  "
          f"(<- the wealth-add verdict)")
    print(f"  GRID PBO                       : {pbo.get('pbo')}  verdict={pbo.get('verdict')}  "
          f"N={pbo.get('N')} T={pbo.get('T_used', pbo.get('T'))}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACT_DIR / f"selection_lab_{out_tag}.json"
    tmp = out_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    tmp.replace(out_path)
    print(f"\n  artifact: {out_path}")
    return summary


def _result_to_dict(r: ConfigResult) -> dict:
    return {"asset": r.asset, "cadence": r.cadence, "fast": r.fast, "mid": r.mid, "long": r.long,
            "horizon": r.horizon, "n_fire": r.n_fire, "n_trades_nonoverlap": r.n_trades,
            "trainval_compound_pct": r.trainval_compound,
            "real_compound_pct": r.real_compound, "mean_net_per_trade": r.mean_net,
            "buyhold_compound_pct": r.buyhold_compound,
            "selection_null_OOS_UNSEEN": r.sel_null, "selection_null_UNSEEN_only": r.sel_null_unseen,
            "oracle": r.oracle}


# ===========================================================================
# SELFTEST: two-sided soundness on a synthetic series where alignment GENUINELY has higher FORWARD drift.
#
# The positive control must be REAL against the SELECTION null (which draws random same-vol-regime windows).
# So forward drift must be conditional on ALIGNMENT, not on a vol regime the null already controls for.
# Construction: a hidden 2-state drift process (BULL drift>0 / FLAT drift~0) whose VOLATILITY is held
# roughly CONSTANT across states (so the past-only vol-tercile bin does NOT separate the states -- the null
# cannot recover the edge from regime alone). The MA stack tracks the hidden state, so alignment fires
# preferentially in BULL bars whose FORWARD-h drift genuinely exceeds a random same-vol-regime bar's. A
# clean trend also makes the alignment fire in long consecutive runs -> the OVERLAP trap the fix defuses.
# ===========================================================================
def _make_fixture(seed: int = 5, n: int = 2400) -> pd.DataFrame:
    """Hidden 2-state (BULL/FLAT) drift, CONSTANT vol across states -> alignment (not vol-regime) carries the
    forward-drift edge, so a GENUINE selection signal beats the same-vol-regime selection null."""
    rng = np.random.default_rng(seed)
    vol = 0.011                                   # SAME vol in both states -> vol-tercile can't separate them
    rets = np.empty(n)
    state = 0                                      # 0 = BEAR/FLAT, 1 = BULL
    t = 0
    while t < n:
        if state == 1:                             # BULL block: strong positive drift, multi-bar -> long aligned runs
            L = int(rng.integers(45, 90))
            mu = 0.010
        else:                                      # BEAR/FLAT block: clearly negative drift (alignment must AVOID it)
            L = int(rng.integers(45, 90))
            mu = -0.006
        L = min(L, n - t)
        rets[t:t + L] = rng.normal(mu, vol, size=L)
        t += L
        state ^= 1                                 # deterministic alternation -> balanced state occupancy
    close = 100.0 * np.cumprod(1.0 + rets[:n])
    open_ = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.002, n)))
    dates = pd.date_range("2018-01-01", periods=n, freq="D")
    return pd.DataFrame({"date": dates, "open": open_, "high": high, "low": low, "close": close})


def _selftest() -> int:
    print("=" * 96)
    print("[selection_signal_lab --selftest] two-sided soundness (no market data)")
    print("=" * 96)
    df = _make_fixture()
    win, purge_mask = quantile_windows(df["date"], purge=20)

    # (A) the alignment SELECTION signal -> fires in BULL bars -> higher forward drift -> beats the null.
    a = evaluate_config(df, win, purge_mask, fast=10, mid=20, long=50, horizon=10,
                        n_books=1500, seed=11, asset="SYN", cadence="syn")
    # (B) a RANDOM trigger at the SAME firing rate, scored through the SAME in-lab selection-null machinery.
    rate = float((alignment_signal(df, fast=10, mid=20, long=50) & (~purge_mask)).mean())
    rng = np.random.default_rng(123)
    dfb = df.copy()
    dfb["rand"] = (rng.random(len(df)) < rate) & (~purge_mask)
    # the random trigger is scored as a TRIGGER column through the SAME book + in-lab selection null.
    b = _eval_trigger_column(dfb, "rand", win, purge_mask, horizon=10, n_books=1500, seed=11)

    a_beats = a.sel_null.get("beats_null")
    a_real = a.sel_null.get("real_compound_pct")
    a_edge = a.sel_null.get("edge_pp")
    a_p = a.sel_null.get("p_value")
    a_mn_edge = a.sel_null.get("mean_net_edge")
    a_mn_p = a.sel_null.get("mean_net_p_value")
    b_beats = b["sel_null"].get("beats_null")
    b_real = b["sel_null"].get("real_compound_pct")
    b_p = b["sel_null"].get("p_value")

    print(f"\n  (A) ALIGNMENT signal  held real_comp={a_real}%  edge={a_edge}pp  z={a.sel_null.get('z')}  "
          f"p={a_p}  beats_null={a_beats}  ntr={a.sel_null.get('n_trades')}")
    print(f"      mean-net edge (selected - random) = {a_mn_edge}  (p={a_mn_p})")
    print(f"  (B) RANDOM trigger    held real_comp={b_real}%  p={b_p}  beats_null={b_beats}")
    print(f"\n  oracle(UNSEEN)={a.oracle['UNSEEN']['oracle_nonoverlap_compound_pct']}%  "
          f"signal(UNSEEN)={a.real_compound['UNSEEN']}%  buyhold(UNSEEN)={a.buyhold_compound['UNSEEN']}%  "
          f"cap_vs_oracle={a.oracle['UNSEEN']['signal_vs_oracle_ratio']}")
    # no explosion guard: a genuine single-position compound must be in a sane range (< a few thousand %)
    sane = all(abs(a.real_compound[w]) < 5000 for w in WINDOWS) and abs(a_real) < 5000

    discriminates = bool(a_beats) and not bool(b_beats)
    ordered = (a_real is not None and b_real is not None and a_real > b_real)
    significant = (a_p is not None and a_p < 0.05)
    mean_net_real = (a_mn_edge is not None and a_mn_edge > 0 and a_mn_p is not None and a_mn_p < 0.05)
    ok = discriminates and ordered and significant and sane
    print("\n" + "-" * 96)
    print("SOUNDNESS (two-sided, overlap-robust):")
    print(f"  ALIGNMENT beats selection-null (p<0.05) AND RANDOM does not : {discriminates and significant}")
    print(f"  ALIGNMENT real compound > RANDOM                            : {ordered}")
    print(f"  mean-net selection edge > 0 and significant                 : {mean_net_real}")
    print(f"  NO explosion (all single-position compounds < 5000%)        : {sane}")
    print(f"\n[selection_signal_lab --selftest] {'PASS' if ok else 'CHECK'}")
    return 0 if ok else 1


def _eval_trigger_column(df: pd.DataFrame, trigger_col: str, win: Windows, purge_mask: np.ndarray, *,
                         horizon: int, cost: float = TAKER, n_regime_bins: int = 3, n_books: int = 1500,
                         seed: int = 11) -> dict:
    """Score an ARBITRARY past-only boolean trigger column through the SAME single-position non-overlapping
    book + in-lab selection null (used by the selftest's RANDOM control). Returns the HELD-pool sel_null."""
    df = df.reset_index(drop=True)
    opens = df["open"].to_numpy(float)
    dates = df["date"]
    n = len(opens)
    last_valid = n - 2 - horizon
    trig = pd.to_numeric(df[trigger_col], errors="coerce").fillna(0.0).to_numpy() > 0.5
    trig = trig & (~purge_mask)
    regime_bins = past_only_regime_bins(df, n_bins=n_regime_bins)
    wlab = np.array([win.label(dates.iloc[i]) for i in range(n)])
    fire_idx = np.array([int(t) for t in np.flatnonzero(trig) if t <= last_valid], dtype=int)
    held_fire = fire_idx[np.isin(wlab[fire_idx], tuple(HELD))] if fire_idx.size else np.array([], int)
    held_seg = np.array([i for i in range(0, last_valid + 1)
                         if (not purge_mask[i]) and wlab[i] in HELD], dtype=int)
    held_bars, held_nets = nonoverlap_book(opens, held_fire, horizon, cost, last_valid)
    sn = selection_null(opens, held_bars, held_nets, horizon=horizon, cost=cost, regime_bins=regime_bins,
                        valid_seg_bars=held_seg, last_valid=last_valid, n_books=n_books, seed=seed)
    return {"sel_null": sn}


# ===========================================================================
# CLI
# ===========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="Multi-MA-alignment SELECTION lab")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--asset", type=str, default=None)
    ap.add_argument("--cadence", type=str, default="1d")
    ap.add_argument("--universe", type=str, default=None, choices=["u10"])
    ap.add_argument("--cadences", nargs="+", default=["1d"])
    ap.add_argument("--n-books", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    if args.selftest:
        return _selftest()

    if args.universe == "u10":
        tag = "u10_" + "_".join(args.cadences)
        run_universe(U10, args.cadences, n_books=args.n_books, seed=args.seed, out_tag=tag)
        return 0

    if args.asset:
        df = load_series(args.asset, args.cadence)
        print(f"loaded {args.asset} {args.cadence}: {len(df)} bars "
              f"{df['date'].min().date()} -> {df['date'].max().date()}")
        results, best = run_grid_for_series(df, asset=args.asset, cadence=args.cadence,
                                            n_books=args.n_books, seed=args.seed)
        print(f"\nTRAIN+VAL-selected best of {len(results)} configs:")
        print(f"  {_fmt_best(best)}")
        pbo = grid_pbo(results, S=8)
        print(f"\ngrid PBO (this series): {pbo.get('pbo')} verdict={pbo.get('verdict')} "
              f"N={pbo.get('N')} T={pbo.get('T_used', pbo.get('T'))}")
        return 0

    print("specify --selftest, --asset SYM, or --universe u10 --cadences 1d dollar")
    return 1


if __name__ == "__main__":
    sys.exit(main())
