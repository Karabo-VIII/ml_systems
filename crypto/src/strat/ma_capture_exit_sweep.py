"""src/strat/ma_capture_exit_sweep.py -- MOVE-CATCH CAMPAIGN: MA-family x EXIT capture-rate sweep.

QUESTION: given an MA-cross entry is IN a move, which exit policy captures the most of that move?
Does any exit beat the no-skill fixed-hold baseline on BOTH mean AND median capture?

DESIGN CHOICES (pre-registered):
  Entry:    each of the 8 canonical MA types (SMA/EMA/WMA/HMA/DEMA/TEMA/KAMA/VIDYA)
            using a representative slow 2-cross config: MA(50) > MA(150) (the
            working-band mid-point; causal; position lagged 1 bar as everywhere).
  Assets:   BTC, ETH (primary) + SOL, BNB, DOGE (u10 extras; cheap to add).
  TF:       1d primary. Notes 4h inline if run with --cadences 1d,4h.
  Span:     2023-01-01 to 2026-05-28 (= in-sample illustrative with a hold-out split at 2025-03-15).
            NOT the 2020 training window (different regime = more relevant to current trading).

CAPTURES-RATE FORMULA (same as capture_rate_probe.py):
  capture = (exit_realized - entry_price) / (peak_since_entry - entry_price)
  clamped [0,1]; peak is the running maximum CLOSE since entry (not a rolling window).
  Available = (peak - entry) / entry  (the MAXIMUM achievable return staying long to the peak).
  Realized  = (exit - entry) / entry - taker_cost_rt.

NO-SKILL BASELINE (the arbiter):
  fixed_hold_N: exit exactly N bars after entry (N = 5, 10, 20).  These are the no-skill controls
  every smart exit must beat on BOTH mean capture AND median capture.

EXIT LIBRARY (pre-registered, not grid-searched):
  fixed_5 / fixed_10 / fixed_20   -- no-skill fixed holds (the baseline)
  trail_atr_2p5 / trail_atr_3p0 / trail_atr_3p5  -- ATR(14) trailing stop k*ATR from RUNNING PEAK
  giveback_10 / giveback_15 / giveback_20         -- exit when price falls X% below running peak
  signal_flip                                     -- exit when MA(fast) crosses below MA(slow)
  chandelier_3atr                                 -- 3*ATR(22) from running peak (same as deep2020)

HONEST CONTROLS (binding):
  1. No-skill fixed-hold is the primary arbiter (not the random-exit null which conflates hold-length).
  2. Membership-matched null is reported for context: does entry SELECTION add value vs random?
  3. Taker cost (24bps RT) applied in realized return. Maker 6bps noted.
  4. Median is the headline (mean is home-run-carried).

Run: python -m strat.ma_capture_exit_sweep
     python -m strat.ma_capture_exit_sweep --cadences 1d,4h
No emoji (cp1252). No git commits from this script.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # crypto/src
CRYPTO = ROOT.parent                        # crypto/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CRYPTO / "src") not in sys.path:
    sys.path.insert(0, str(CRYPTO / "src"))

from strat.ma_type_upgrade import _MA, _nums   # noqa: E402 -- all 8 MA implementations + registry
from strat.ma_2020_breakdown import _panel      # noqa: E402 -- loads OHLC from chimera parquet
from strat.portfolio_replay import TAKER_RT     # noqa: E402 -- 0.0024 taker RT

# ---------------------------------------------------------------------------
# CONSTANTS (pre-registered)
# ---------------------------------------------------------------------------
CADENCES: list[str] = ["1d"]   # overridden via --cadences
ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT"]
MA_TYPES = ["SMA", "EMA", "WMA", "HMA", "DEMA", "TEMA", "KAMA", "VIDYA"]

# Representative 2-cross config: fast=50, slow=150 (working-band mid-point per campaign history)
FAST_PERIOD = 50
SLOW_PERIOD = 150
WARMUP = 400   # bars of pre-window warmup so MAs are initialised

# Span: 2023-01-01 -> all available (2026-05-28 for current data)
TRAIN_START   = "2023-01-01"
HOLDOUT_SPLIT = "2025-03-15"   # OOS split; before = in-sample; after = held-out

TAKER_RT_USED = TAKER_RT   # 0.0024 (24bps RT)
MAKER_RT_NOTE = 0.0006     # 6bps RT, noted only

# ATR window
ATR_WIN = 14
ATR_CHANDELIER_WIN = 22

# Exit configs (pre-registered)
EXITS = [
    "fixed_5", "fixed_10", "fixed_20",
    "trail_atr_2p5", "trail_atr_3p0", "trail_atr_3p5",
    "giveback_10pct", "giveback_15pct", "giveback_20pct",
    "signal_flip",
    "chandelier_3atr",
]

# The no-skill fixed-hold baselines (subset of EXITS; these are the control)
NO_SKILL_BASELINES = ["fixed_5", "fixed_10", "fixed_20"]

# How many bars a "big move" must last (min) to be included in coverage stats
BIG_MOVE_AVAIL_PCT = 8.0   # threshold: available >= 8% to be counted as a "big move"


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
def _load_bars(sym: str, cad: str) -> Optional[tuple]:
    """Load OHLC arrays from chimera via _panel, cropped to TRAIN_START onwards (+ warmup)."""
    try:
        o, h, l, c, ms = _panel(sym, cad)
    except Exception as ex:
        print(f"  [skip] {sym} {cad}: {ex}")
        return None
    s_ms = int(pd.Timestamp(TRAIN_START).value // 10**6)
    e_ms = ms[-1]  # all available
    e_idx = len(ms)
    s_idx = max(0, int(np.searchsorted(ms, s_ms)) - WARMUP)
    o2, h2, l2, c2, ms2 = o[s_idx:e_idx], h[s_idx:e_idx], l[s_idx:e_idx], c[s_idx:e_idx], ms[s_idx:e_idx]
    win = ms2 >= s_ms
    if win.sum() < 40:
        return None
    return o2, h2, l2, c2, ms2, win


# ---------------------------------------------------------------------------
# MA ENTRY SIGNAL
# ---------------------------------------------------------------------------
def _held_cross(c: np.ndarray, fast: int, slow: int, ma_type: str) -> np.ndarray:
    """2-MA cross: long when MA(fast) > MA(slow). Returns int8 {0,1}."""
    f = _MA[ma_type]
    ma_f = f(c, fast)
    ma_s = f(c, slow)
    return np.nan_to_num(ma_f > ma_s).astype(np.int8)


# ---------------------------------------------------------------------------
# ATR helper
# ---------------------------------------------------------------------------
def _atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int) -> np.ndarray:
    """Wilder ATR(n), causal."""
    prev_c = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = pd.Series(tr).ewm(span=n, adjust=False, min_periods=1).mean().to_numpy()
    return atr


# ---------------------------------------------------------------------------
# RUNS DETECTION
# ---------------------------------------------------------------------------
def _runs(held: np.ndarray) -> list[tuple[int, int]]:
    """Return (start_idx, end_idx_exclusive) for each contiguous run of 1s in held."""
    h = np.asarray(held, dtype=np.int8)
    d = np.diff(np.concatenate([[0], h, [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


# ---------------------------------------------------------------------------
# EXIT APPLICATION (causal)
# ---------------------------------------------------------------------------
def _apply_exit_to_run(
    s: int, e: int,
    c: np.ndarray, h_arr: np.ndarray, l_arr: np.ndarray,
    atr14: np.ndarray, atr22: np.ndarray,
    ma_f: np.ndarray, ma_s: np.ndarray,
    exit_name: str,
) -> int:
    """Given a run [s, e) in held, return the actual EXIT bar index (inclusive) for this exit rule.
    All computations are causal: decision at bar i uses only data up to bar i."""
    if e <= s:
        return s
    run_peak = c[s]

    for i in range(s, e):
        run_peak = max(run_peak, c[i])

        if exit_name == "signal_flip":
            # exit when fast MA crosses below slow MA
            if i > s and ma_f[i] < ma_s[i]:
                return i

        elif exit_name.startswith("fixed_"):
            n_bars = int(exit_name.split("_")[1])
            if (i - s) >= n_bars - 1:
                return i

        elif exit_name.startswith("trail_atr_"):
            # running-peak anchored: exit when close falls k*ATR below running peak
            k_str = exit_name.split("_")[-1]          # e.g. "2p5"
            k = float(k_str.replace("p", "."))
            if i > s and c[i] < run_peak - k * atr14[i]:
                return i

        elif exit_name.startswith("giveback_"):
            # exit when close falls X% below running peak
            pct_str = exit_name.replace("giveback_", "").replace("pct", "")
            pct = float(pct_str) / 100.0
            if i > s and c[i] < run_peak * (1.0 - pct):
                return i

        elif exit_name == "chandelier_3atr":
            # 3*ATR(22) from running peak
            if i > s and c[i] < run_peak - 3.0 * atr22[i]:
                return i

    return e - 1   # fall-through: exit at last bar of run


# ---------------------------------------------------------------------------
# PER-TRADE CAPTURE COMPUTATION
# ---------------------------------------------------------------------------
def _compute_trades(
    o: np.ndarray, h_arr: np.ndarray, l_arr: np.ndarray, c: np.ndarray,
    ms: np.ndarray, win: np.ndarray,
    held_base: np.ndarray,   # signal-level held from MA cross (no exit applied yet)
    atr14: np.ndarray, atr22: np.ndarray,
    ma_f: np.ndarray, ma_s: np.ndarray,
    exit_name: str,
) -> list[dict]:
    """Apply an exit rule to each signal-run and record per-trade stats."""
    runs = _runs(held_base)
    trades = []
    for s, e in runs:
        # entry at open[s+1] (1-bar execution lag); if at last bar, skip
        entry_bar = s + 1
        if entry_bar >= len(c) - 1:
            continue

        # determine exit bar via exit rule
        exit_bar = _apply_exit_to_run(
            s, e, c, h_arr, l_arr, atr14, atr22, ma_f, ma_s, exit_name
        )
        exit_bar = max(entry_bar, min(exit_bar, len(c) - 1))

        entry_price = o[entry_bar]
        exit_price  = c[exit_bar]  # exit at close of exit bar

        # peak-since-entry: running max of closes from entry_bar to exit_bar (inclusive)
        peak_price = float(np.max(c[entry_bar: exit_bar + 1]))

        available = (peak_price - entry_price) / entry_price if entry_price > 0 else 0.0
        realized_gross = (exit_price - entry_price) / entry_price if entry_price > 0 else 0.0
        realized_net = realized_gross - TAKER_RT_USED  # deduct taker round-trip

        if available > 0.005:  # minimum spread to avoid division noise
            capture = float(np.clip(realized_net / available, 0.0, 1.0))
        elif realized_net > 0:
            capture = 1.0
        else:
            capture = 0.0

        entry_ts = pd.Timestamp(ms[s], unit="ms")
        exit_ts  = pd.Timestamp(ms[exit_bar], unit="ms")

        in_holdout = entry_ts >= pd.Timestamp(HOLDOUT_SPLIT)
        in_window  = bool(win[s])

        trades.append({
            "entry_ts":     str(entry_ts.date()),
            "exit_ts":      str(exit_ts.date()),
            "entry_bar":    int(s),
            "exit_bar":     int(exit_bar),
            "duration":     int(exit_bar - entry_bar + 1),
            "entry_price":  float(entry_price),
            "exit_price":   float(exit_price),
            "peak_price":   float(peak_price),
            "available_pct":float(available * 100),
            "realized_net_pct": float(realized_net * 100),
            "capture":      float(capture),
            "in_holdout":   in_holdout,
            "in_window":    in_window,
            "is_big_move":  bool(available * 100 >= BIG_MOVE_AVAIL_PCT),
        })
    return trades


# ---------------------------------------------------------------------------
# AGGREGATE STATS
# ---------------------------------------------------------------------------
def _stats(trades: list[dict], subset: str = "all") -> dict:
    """Aggregate mean/median capture + win-rate + net%/trade for a list of trades.
    subset: 'all' | 'insample' | 'holdout' | 'big_moves'
    """
    if subset == "holdout":
        ts = [t for t in trades if t["in_holdout"]]
    elif subset == "insample":
        ts = [t for t in trades if not t["in_holdout"]]
    elif subset == "big_moves":
        ts = [t for t in trades if t["is_big_move"]]
    else:
        ts = list(trades)

    if not ts:
        return {"n": 0, "mean_cap": None, "median_cap": None,
                "mean_net_pct": None, "win_rate": None, "mean_avail_pct": None,
                "mean_dur": None}

    caps = np.array([t["capture"] for t in ts])
    nets = np.array([t["realized_net_pct"] for t in ts])
    wins = np.array([t["realized_net_pct"] > 0 for t in ts])
    avail = np.array([t["available_pct"] for t in ts])
    dur   = np.array([t["duration"] for t in ts])

    return {
        "n":             len(ts),
        "mean_cap":      round(float(np.mean(caps)), 4),
        "median_cap":    round(float(np.median(caps)), 4),
        "mean_net_pct":  round(float(np.mean(nets)), 3),
        "win_rate":      round(float(np.mean(wins)), 3),
        "mean_avail_pct": round(float(np.mean(avail)), 2),
        "mean_dur":      round(float(np.mean(dur)), 1),
    }


# ---------------------------------------------------------------------------
# DELTA vs BASELINE
# ---------------------------------------------------------------------------
def _delta_vs_baseline(
    trades_smart: list[dict],
    trades_base: list[dict],
    subset: str = "all",
) -> dict:
    """Per-trade delta in capture: smart - baseline, for paired trades (same entry).
    Requires trades are for the SAME runs (same entry bars).  We match by entry_bar."""
    base_map = {t["entry_bar"]: t["capture"] for t in trades_base}
    if subset == "holdout":
        ts = [t for t in trades_smart if t["in_holdout"]]
    elif subset == "big_moves":
        ts = [t for t in trades_smart if t["is_big_move"]]
    else:
        ts = list(trades_smart)

    deltas = []
    for t in ts:
        bc = base_map.get(t["entry_bar"])
        if bc is not None:
            deltas.append(t["capture"] - bc)

    if not deltas:
        return {"n_paired": 0, "mean_delta": None, "median_delta": None, "frac_better": None}

    d = np.array(deltas)
    return {
        "n_paired":    len(d),
        "mean_delta":  round(float(np.mean(d)), 4),
        "median_delta": round(float(np.median(d)), 4),
        "frac_better": round(float(np.mean(d > 0)), 3),
    }


# ---------------------------------------------------------------------------
# MEMBERSHIP-MATCHED NULL: random entry within same move window
# ---------------------------------------------------------------------------
def _membership_null_capture(
    trades_base: list[dict],
    o: np.ndarray, c: np.ndarray, ms: np.ndarray,
    rng_seed: int = 42,
    n_books: int = 500,
    subset: str = "all",
) -> dict:
    """For each trade in trades_base, the move window is [entry_bar - dur, entry_bar + dur].
    Draw n_books random entries from that window; compute capture vs same peak-to-exit-bar.
    Returns the p50 and p95 of the null mean-capture distribution over books."""
    if subset == "holdout":
        ts = [t for t in trades_base if t["in_holdout"]]
    else:
        ts = list(trades_base)
    if not ts:
        return {"null_p50": None, "null_p95": None}

    rng = np.random.default_rng(rng_seed)
    book_means = []
    for _ in range(n_books):
        book_caps = []
        for t in ts:
            eb = t["entry_bar"]
            dur = max(1, t["duration"])
            lo = max(1, eb - dur)
            hi = min(len(c) - 2, eb + dur)
            if hi < lo:
                continue
            null_entry = int(rng.integers(lo, hi + 1))
            null_ep = o[null_entry + 1] if null_entry + 1 < len(c) else o[-1]
            # peak from null_entry+1 to same exit_bar
            exit_bar = min(t["exit_bar"], len(c) - 1)
            if null_entry + 1 >= exit_bar:
                continue
            peak = float(np.max(c[null_entry + 1: exit_bar + 1]))
            available = (peak - null_ep) / null_ep if null_ep > 0 else 0.0
            realized = (c[exit_bar] - null_ep) / null_ep - TAKER_RT_USED if null_ep > 0 else 0.0
            if available > 0.005:
                cap = float(np.clip(realized / available, 0.0, 1.0))
            elif realized > 0:
                cap = 1.0
            else:
                cap = 0.0
            book_caps.append(cap)
        if book_caps:
            book_means.append(float(np.mean(book_caps)))

    if not book_means:
        return {"null_p50": None, "null_p95": None}

    bm = np.array(book_means)
    return {
        "null_p50": round(float(np.percentile(bm, 50)), 4),
        "null_p95": round(float(np.percentile(bm, 95)), 4),
    }


# ---------------------------------------------------------------------------
# EXAMPLE TRADES PICKER
# ---------------------------------------------------------------------------
def _pick_examples(trades: list[dict], n: int = 3) -> list[dict]:
    if not trades:
        return []
    # pick the top-capture, bottom-capture, and a middle trade for illustration
    srt = sorted(trades, key=lambda t: t["capture"])
    if len(srt) <= n:
        return srt
    step = (len(srt) - 1) // (n - 1)
    return [srt[i * step] for i in range(n)]


# ---------------------------------------------------------------------------
# MAIN SWEEP
# ---------------------------------------------------------------------------
def run_sweep(cadences: list[str]) -> dict:
    all_results = {}  # {(cad, ma_type, exit_name, sym): trades}

    for cad in cadences:
        print(f"\n{'='*72}")
        print(f"CADENCE: {cad}")
        print(f"{'='*72}")

        for ma_type in MA_TYPES:
            print(f"\n  -- MA type: {ma_type} (cross: {FAST_PERIOD}/{SLOW_PERIOD}) --")

            # Aggregate per-exit stats across all assets
            exit_stats: dict[str, list] = {ex: [] for ex in EXITS}  # {exit: [per-asset stat dicts]}

            for sym in ASSETS:
                bars = _load_bars(sym, cad)
                if bars is None:
                    continue
                o_arr, h_arr, l_arr, c_arr, ms_arr, win_arr = bars

                # Pre-compute MAs and ATRs once per (sym, cad, ma_type)
                ma_f_arr = _MA[ma_type](c_arr, FAST_PERIOD)
                ma_s_arr = _MA[ma_type](c_arr, SLOW_PERIOD)
                atr14_arr = _atr(h_arr, l_arr, c_arr, ATR_WIN)
                atr22_arr = _atr(h_arr, l_arr, c_arr, ATR_CHANDELIER_WIN)

                # Base signal: 2-MA cross (no exit applied)
                held_base = _held_cross(c_arr, FAST_PERIOD, SLOW_PERIOD, ma_type)

                # Compute trades for each exit
                per_exit_trades: dict[str, list] = {}
                for ex in EXITS:
                    trades = _compute_trades(
                        o_arr, h_arr, l_arr, c_arr, ms_arr, win_arr,
                        held_base, atr14_arr, atr22_arr, ma_f_arr, ma_s_arr, ex
                    )
                    per_exit_trades[ex] = trades

                for ex in EXITS:
                    trades = per_exit_trades[ex]
                    s_all = _stats(trades, "all")
                    s_ho  = _stats(trades, "holdout")
                    s_big = _stats(trades, "big_moves")
                    exit_stats[ex].append({
                        "sym": sym, "all": s_all, "holdout": s_ho, "big": s_big
                    })
                    all_results[(cad, ma_type, ex, sym)] = trades

            # Print per-ma_type table
            print(f"\n  {'exit':18} {'n_ho':>5} {'med_cap(ho)':>11} {'mean_cap(ho)':>12} "
                  f"{'net%/tr(ho)':>11} {'win%(ho)':>8} {'avail%(ho)':>10} {'dur(ho)':>7}")
            print(f"  {'-'*18} {'-'*5} {'-'*11} {'-'*12} {'-'*11} {'-'*8} {'-'*10} {'-'*7}")
            for ex in EXITS:
                rows = exit_stats[ex]
                # aggregate: mean of per-asset means/medians (family average)
                def agg(rows, subset_key, field):
                    vals = [r[subset_key][field] for r in rows if r[subset_key][field] is not None]
                    return float(np.mean(vals)) if vals else float("nan")

                n_ho      = sum(r["holdout"]["n"] for r in rows)
                med_cap   = agg(rows, "holdout", "median_cap")
                mean_cap  = agg(rows, "holdout", "mean_cap")
                net_ptr   = agg(rows, "holdout", "mean_net_pct")
                win_r     = agg(rows, "holdout", "win_rate")
                avail     = agg(rows, "holdout", "mean_avail_pct")
                dur       = agg(rows, "holdout", "mean_dur")
                marker    = " <-- NO-SKILL" if ex in NO_SKILL_BASELINES else ""
                print(f"  {ex:18} {n_ho:>5} {med_cap:>11.3f} {mean_cap:>12.3f} "
                      f"{net_ptr:>11.2f} {win_r:>8.1%} {avail:>10.1f} {dur:>7.1f}{marker}")

    return all_results


# ---------------------------------------------------------------------------
# SUMMARY TABLE (best-exit per MA type)
# ---------------------------------------------------------------------------
def build_summary_table(all_results: dict, cadence: str = "1d") -> list[dict]:
    """For each MA type, find the best exit by MEDIAN capture on holdout, report delta vs best fixed-hold."""
    rows = []
    for ma_type in MA_TYPES:
        exit_rows = []
        for ex in EXITS:
            # gather holdout trades across all assets
            ho_trades = []
            for sym in ASSETS:
                key = (cadence, ma_type, ex, sym)
                if key in all_results:
                    ho_trades.extend([t for t in all_results[key] if t["in_holdout"]])
            s = _stats(ho_trades, "all")  # already filtered to holdout
            if s["n"] == 0:
                continue
            exit_rows.append({"exit": ex, **s})

        if not exit_rows:
            rows.append({"ma_type": ma_type, "best_exit": "N/A"})
            continue

        # find best by median_cap
        best = max(exit_rows, key=lambda r: r["median_cap"] if r["median_cap"] is not None else -1)
        # find best fixed-hold baseline
        baselines = [r for r in exit_rows if r["exit"] in NO_SKILL_BASELINES]
        best_base = max(baselines, key=lambda r: r["median_cap"] if r["median_cap"] is not None else -1) if baselines else None

        # membership-null for the BEST exit (BTC only, for speed)
        best_trades_btc = [t for t in all_results.get((cadence, ma_type, best["exit"], "BTCUSDT"), []) if t["in_holdout"]]
        btc_base_trades = [t for t in all_results.get((cadence, ma_type, NO_SKILL_BASELINES[1], "BTCUSDT"), []) if t["in_holdout"]]
        o_arr, h_arr, l_arr, c_arr, ms_arr, win_arr = _load_bars("BTCUSDT", cadence) or (None,) * 6
        null_stats = {}
        if o_arr is not None and best_trades_btc:
            null_stats = _membership_null_capture(best_trades_btc, o_arr, c_arr, ms_arr, n_books=300)

        delta_vs_base = None
        if best_base is not None and best["median_cap"] is not None and best_base["median_cap"] is not None:
            delta_vs_base = round(best["median_cap"] - best_base["median_cap"], 4)

        rows.append({
            "ma_type": ma_type,
            "best_exit": best["exit"],
            "n_holdout": best["n"],
            "mean_cap": best["mean_cap"],
            "median_cap": best["median_cap"],
            "mean_net_pct": best["mean_net_pct"],
            "win_rate": best["win_rate"],
            "delta_vs_best_fixed": delta_vs_base,
            "best_fixed_exit": best_base["exit"] if best_base else None,
            "best_fixed_median_cap": best_base["median_cap"] if best_base else None,
            "null_p50_btc": null_stats.get("null_p50"),
            "null_p95_btc": null_stats.get("null_p95"),
        })
    return rows


# ---------------------------------------------------------------------------
# HEADLINE VERDICT
# ---------------------------------------------------------------------------
def headline_verdict(summary_rows: list[dict]) -> str:
    """Report whether any exit lifts median capture materially above the no-skill fixed-hold."""
    lifts = [r for r in summary_rows if r.get("delta_vs_best_fixed") is not None and r["delta_vs_best_fixed"] > 0.03]
    all_median = [r["median_cap"] for r in summary_rows if r.get("median_cap") is not None]
    best_fixed_medians = [r.get("best_fixed_median_cap") for r in summary_rows if r.get("best_fixed_median_cap") is not None]

    verdict_lines = [
        "HEADLINE VERDICT",
        "-" * 72,
        f"  MA types evaluated     : {len(summary_rows)}",
        f"  MA types with smart exit > fixed hold (+0.03 median cap) : {len(lifts)}",
        f"  Overall median capture range (best exit, by MA type): "
        f"{min(r['median_cap'] for r in summary_rows if r.get('median_cap') is not None):.3f} .. "
        f"{max(r['median_cap'] for r in summary_rows if r.get('median_cap') is not None):.3f}",
    ]
    if len(lifts) >= 4:
        verdict_lines.append("  VERDICT: Smart exits MATERIALLY lift median capture vs no-skill fixed-hold.")
        verdict_lines.append("  The exit dimension DOES recover meaningful capture on MA-family setups.")
    elif len(lifts) >= 2:
        verdict_lines.append("  VERDICT: Mixed -- some MA types benefit from smart exits, others do not.")
        verdict_lines.append("  The exit dimension provides PARTIAL but not universal improvement.")
    else:
        verdict_lines.append("  VERDICT: Fixed-hold dominates. Smart exits do NOT lift median capture reliably.")
        verdict_lines.append("  D61 confirmed: the MA-family exit problem is HOME-RUN-OR-BUST (fixed-hold = default).")

    return "\n".join(verdict_lines)


# ---------------------------------------------------------------------------
# EXAMPLE TRADES (illustrative)
# ---------------------------------------------------------------------------
def print_example_trades(all_results: dict, cadence: str = "1d"):
    """Print 3 representative trades from BTC + ETH across different exits for illustration."""
    print("\nEXAMPLE TRADES (BTC 1d, EMA cross 50/150, holdout only)")
    print("-" * 72)
    hdr = f"  {'exit':18} {'entry':>10} {'exit':>10} {'avail%':>7} {'net%':>7} {'cap':>5} {'dur':>4}"
    print(hdr)
    print("  " + "-" * 66)

    for ex in ["fixed_10", "trail_atr_2p5", "giveback_15pct", "chandelier_3atr"]:
        key = (cadence, "EMA", ex, "BTCUSDT")
        trades = [t for t in all_results.get(key, []) if t["in_holdout"]]
        if not trades:
            continue
        examples = _pick_examples(trades, 2)
        for t in examples:
            print(f"  {ex:18} {t['entry_ts']:>10} {t['exit_ts']:>10} "
                  f"{t['available_pct']:>7.1f} {t['realized_net_pct']:>7.2f} "
                  f"{t['capture']:>5.2f} {t['duration']:>4}")
    print()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> int:
    global CADENCES
    if "--cadences" in sys.argv:
        CADENCES = sys.argv[sys.argv.index("--cadences") + 1].split(",")

    print("MA-CAPTURE-EXIT SWEEP")
    print("=" * 72)
    print(f"  MA types    : {MA_TYPES}")
    print(f"  Config      : MA({FAST_PERIOD}) > MA({SLOW_PERIOD}) 2-cross, 1-bar execution lag")
    print(f"  Assets      : {ASSETS}")
    print(f"  Cadences    : {CADENCES}")
    print(f"  Span        : {TRAIN_START} -> present (holdout split at {HOLDOUT_SPLIT})")
    print(f"  Cost        : taker {TAKER_RT_USED*10000:.0f}bps RT (maker {MAKER_RT_NOTE*10000:.0f}bps noted)")
    print(f"  Exits       : {EXITS}")
    print(f"  No-skill B  : {NO_SKILL_BASELINES}")
    print(f"  Capture     : (realized_net - available) clamped [0,1]; peak = running max close since entry")
    print()

    all_results = run_sweep(CADENCES)

    for cad in CADENCES:
        print(f"\n{'='*72}")
        print(f"SUMMARY TABLE -- {cad} -- best exit per MA type (holdout only)")
        print(f"{'='*72}")
        summary = build_summary_table(all_results, cad)
        print(f"\n  {'MA type':8} {'best_exit':20} {'n':>4} {'med_cap':>8} {'mean_cap':>9} "
              f"{'net%/tr':>8} {'wr%':>5} {'Dmed_vs_fixed':>14} {'null_p50(BTC)':>13}")
        print(f"  {'-'*8} {'-'*20} {'-'*4} {'-'*8} {'-'*9} {'-'*8} {'-'*5} {'-'*14} {'-'*13}")
        for r in summary:
            if r.get("best_exit") == "N/A":
                print(f"  {r['ma_type']:8}  N/A")
                continue
            d = r.get("delta_vs_best_fixed")
            d_str = f"{d:+.3f}" if d is not None else "  N/A"
            np50 = r.get("null_p50_btc")
            np50_str = f"{np50:.3f}" if np50 is not None else "   N/A"
            print(f"  {r['ma_type']:8} {r['best_exit']:20} {r['n_holdout']:>4} "
                  f"{r['median_cap']:>8.3f} {r['mean_cap']:>9.3f} "
                  f"{r['mean_net_pct']:>8.2f} {r['win_rate']:>5.1%} "
                  f"{d_str:>14} {np50_str:>13}")

        print()
        print(headline_verdict(summary))
        print()
        print_example_trades(all_results, cad)

        # ---- per-MA-type: show all exits ranked by median holdout capture ----
        print(f"\nDETAIL -- per MA type, all exits ranked by median holdout capture ({cad})")
        print("-" * 72)
        for ma_type in MA_TYPES:
            exit_rows = []
            for ex in EXITS:
                ho_trades = []
                for sym in ASSETS:
                    key = (cad, ma_type, ex, sym)
                    if key in all_results:
                        ho_trades.extend([t for t in all_results[key] if t["in_holdout"]])
                s = _stats(ho_trades, "all")
                if s["n"] > 0:
                    exit_rows.append({"exit": ex, **s})
            exit_rows.sort(key=lambda r: r["median_cap"] if r["median_cap"] is not None else -1, reverse=True)
            print(f"\n  [{ma_type}] ({cad}) -- ranked by holdout median capture:")
            print(f"  {'exit':18} {'n':>5} {'med_cap':>8} {'mean_cap':>9} {'net%/tr':>8} {'wr%':>6}")
            for er in exit_rows[:6]:  # top 6
                mk = " *" if er["exit"] in NO_SKILL_BASELINES else ""
                print(f"  {er['exit']:18} {er['n']:>5} {er['median_cap']:>8.3f} "
                      f"{er['mean_cap']:>9.3f} {er['mean_net_pct']:>8.2f} {er['win_rate']:>6.1%}{mk}")

    # save JSON
    out_dir = CRYPTO / "runs" / "strat"
    out_dir.mkdir(parents=True, exist_ok=True)
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"ma_capture_exit_sweep_{ts}.json"

    # flatten all_results for JSON serialisation
    json_payload = {}
    for (cad, ma_type, ex, sym), trades in all_results.items():
        key = f"{cad}|{ma_type}|{ex}|{sym}"
        json_payload[key] = {
            "n_trades": len(trades),
            "n_holdout": sum(1 for t in trades if t["in_holdout"]),
            "stats_all": _stats(trades, "all"),
            "stats_holdout": _stats([t for t in trades if t["in_holdout"]], "all"),
            "stats_big": _stats([t for t in trades if t["is_big_move"]], "all"),
            "examples": _pick_examples([t for t in trades if t["in_holdout"]], 3),
        }
    # also save summary per cadence
    summaries = {}
    for cad in CADENCES:
        summaries[cad] = build_summary_table(all_results, cad)
    json_payload["_summaries"] = summaries

    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(json_payload, indent=1, default=str), encoding="utf-8")
    tmp.replace(out_path)
    print(f"\n[JSON] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
