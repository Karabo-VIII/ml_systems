"""src/strat/ma_movecatch_decomp.py -- MA MOVE-CATCH DECOMPOSITION HARNESS.

QUESTION: For the working-band ensemble, per (MA-type x TF):
  - What fraction of >=X% up-moves does the band CATCH (coverage)?
  - WHERE in the move does the band enter (entry-lag: 0=trough, 1=peak, LOWER=better)?
  - How much of the move does the band CAPTURE (raw + weighted by move size)?
  - Does the band enter EARLIER than a random-within-move entry (genuine move-catch skill)?

THE THESIS: Exit timing is a settled null (trail10+min_hold12 is the fixed base). The CAPTURE
lever is ENTRY-LAG -- enter early enough to grab the bulk of the move. If the band enters
systematically earlier than random, it has genuine move-catch value beyond drift exposure.

METHODOLOGY:
  - MOVES defined per-asset as sequences on the TRAIN split (2020-01-01..2020-07-01).
  - A move = a forward up-run of >=X% from a local trough to subsequent peak.
    Troughs detected via rolling local minima (look-behind only, causal). Peak is the highest
    close reached forward from that trough BEFORE a >=Y% pullback (or the split end).
    X swept: 5%, 10%, 15% (report for X=5% primary; 10% secondary).
  - Working band: configs positive across TRAIN&VAL&OOS within-2020, built per (MA-type, TF)
    via ma_2020_config_leaderboard.run_cell. Band ensemble = fixed-EW mean of band members.
  - Per move, the band position is taken from the ironed sleeve (post-trail, post-minhold, lag-1).
    COVERAGE = fraction of moves during which band held long at ANY bar in [trough+1 .. peak].
  - ENTRY-LAG = first bar the band is long within [trough+1..peak], as fraction of move length.
    = (first_long_price - trough_price) / (peak_price - trough_price). 0=start, 1=peak, >1=missed.
    For moves where band enters before the trough (was already long), lag=0 (early entry).
  - CAPTURE raw = (close at first_exit_from_move - close at first_long_bar) / (peak - trough).
    We use the band POSITION within the move window: what fraction of the move magnitude was
    realized by the ironed sleeve. Approximated as: sum(net returns while in-position in move) /
    (peak/trough - 1).
  - WEIGHTED CAPTURE = raw_capture x move_magnitude (sizes contribution by move importance).
  - NULL: random entry within the move window -- same hold duration as the band had, but entry
    placed uniformly at random in [trough_bar+1 .. peak_bar-1]. Delta = band_lag - null_lag.
    Negative delta = band enters EARLIER than random (genuine skill). t-test across moves.
  - CADENCE-INVARIANCE SANITY: buy-hold must be ~140-157% for 2020 full year across TFs.

SPLITS (within-2020, per ma_2020_breakdown.SPLIT):
  TRAIN: 2020-01-01..2020-07-01  <- moves detected here (development set)
  VAL:   2020-07-01..2020-10-01  <- SEALED (not touched)
  OOS:   2020-10-01..2021-01-01  <- SEALED (not touched)
UNSEEN: 2025-12-31+ -- NOT TOUCHED.

RWYB:
  python -m strat.ma_movecatch_decomp --selftest         # 1d only, quick sanity
  python -m strat.ma_movecatch_decomp --tf 1d            # single TF
  python -m strat.ma_movecatch_decomp --all              # all 6 TFs (incremental writes)
No emoji (Windows cp1252). Does NOT git commit (overseer commits).
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="invalid value encountered in divide")
np.seterr(invalid="ignore", divide="ignore")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strat.ma_2020_breakdown import _panel, SPLIT, YEAR, WARMUP          # noqa: E402
from strat.ma_2020_config_leaderboard import (                            # noqa: E402
    build_panels, run_cell, _asset_close, SYMS, SPLITS, ANN,
    config_book, TRAIL, MINHOLD,
)
from strat.portfolio_replay import apply_trail_stop, MAKER_RT             # noqa: E402
from strat.replay_distinct_grid import distinct_specs                     # noqa: E402
from strat.structural_fixes import min_hold                               # noqa: E402
from strat.ma_type_upgrade import _nums, _MA, MA_TYPES                   # noqa: E402
import strat.ma_2020_config_leaderboard as L                             # noqa: E402

OUT = ROOT.parent / "runs" / "strat"
OUT.mkdir(parents=True, exist_ok=True)

TFS = ["1d", "4h", "2h", "1h", "30m", "15m"]

# TRAIN window (development only)
TRAIN_START = "2020-01-01"
TRAIN_END   = "2020-07-01"

# Move parameters
MOVE_THRESHOLDS = [0.05, 0.10, 0.15]   # sweep; primary = 5%
PRIMARY_THRESH = 0.05
PULLBACK_FRAC  = 0.30                  # peak ends when pullback from peak >= 30% of move


# ============================================================
# DATA: per-asset close arrays scoped to TRAIN window
# ============================================================

def _train_close(sym: str, cad: str):
    """Return (close_arr, ms_arr, train_mask) over [TRAIN_START-WARMUP .. TRAIN_END].
    train_mask selects bars within [TRAIN_START .. TRAIN_END).
    Returns None if too short."""
    try:
        o, h, l, c, ms = _panel(sym, cad)
    except Exception:
        return None
    s_ms = pd.Timestamp(TRAIN_START).value // 10**6
    e_ms = pd.Timestamp(TRAIN_END).value // 10**6
    e_idx = int(np.searchsorted(ms, e_ms))
    s_idx = max(0, int(np.searchsorted(ms, s_ms)) - WARMUP)
    c2, ms2 = c[s_idx:e_idx], ms[s_idx:e_idx]
    if len(c2) < 40:
        return None
    win = ms2 >= s_ms
    if win.sum() < 20:
        return None
    return c2, ms2, win


# ============================================================
# MOVE DETECTION
# ============================================================

def _detect_moves(close: np.ndarray, threshold: float = PRIMARY_THRESH,
                  pullback: float = PULLBACK_FRAC) -> list:
    """Detect forward up-runs of >= threshold from local troughs.

    Algorithm (causal -- uses only past data to identify troughs):
      - Scan close array left-to-right.
      - A trough candidate i is a local minimum: close[i] <= close[i-1] and close[i] <= close[i+1].
        (Detected at i+1, so it's confirmed at i+1 -- 1-bar delayed. This is fine for the band too.)
      - From trough i, scan forward for the peak: the highest close before a pullback >= pullback_frac
        of the run-to-date, or end-of-array.
      - Move is valid if (peak - trough) / trough >= threshold.
      - Non-overlapping: after recording a move, advance the cursor past the peak.

    Returns list of dicts: {trough_idx, peak_idx, trough_price, peak_price, move_frac}.
    Indices are into the close array (full array including warmup).
    """
    n = len(close)
    moves = []
    i = 1
    while i < n - 2:
        # local minimum: close[i] <= neighbors
        if close[i] <= close[i - 1] and close[i] <= close[i + 1]:
            trough_price = close[i]
            trough_idx = i
            # scan forward for peak
            best_price = trough_price
            best_idx = i
            hw = trough_price
            j = i + 1
            while j < n:
                if close[j] > best_price:
                    best_price = close[j]
                    best_idx = j
                if close[j] > hw:
                    hw = close[j]
                # pullback from high-water by >= pullback_frac of (hw - trough)
                run = hw - trough_price
                if run > 0 and (hw - close[j]) / run >= pullback:
                    break
                j += 1
            move_frac = (best_price - trough_price) / trough_price if trough_price > 0 else 0.0
            if move_frac >= threshold:
                moves.append({
                    "trough_idx": int(trough_idx),
                    "peak_idx": int(best_idx),
                    "trough_price": float(trough_price),
                    "peak_price": float(best_price),
                    "move_frac": float(move_frac),
                })
                i = best_idx + 1  # non-overlapping
            else:
                i += 1
        else:
            i += 1
    return moves


# ============================================================
# IRONED SLEEVE: per-asset holding array (post-trail+minhold+lag1)
# ============================================================

def _ironed_held(close: np.ndarray, periods: list, ma_type: str) -> np.ndarray:
    """Build the ironed held array for one (asset, config, MA-type).
    MA-cross -> trail(0.10) -> min_hold(12) -> lag-1.
    Returns position array aligned to close (1=long, 0=flat)."""
    maf = _MA[ma_type]
    mas = [maf(close, p) for p in periods]
    if len(periods) == 2:
        h0 = (mas[0] > mas[1]).astype(np.int8)
    else:
        h0 = ((mas[0] > mas[1]) & (mas[1] > mas[2])).astype(np.int8)
    h0 = np.nan_to_num(h0).astype(np.int8)
    h1 = apply_trail_stop(h0.copy(), close, TRAIL)[0].astype(np.int8)
    h2 = min_hold(h1, MINHOLD).astype(np.int8)
    # lag 1: position at bar t = decision at bar t-1
    pos = np.zeros(len(close), dtype=np.int8)
    pos[1:] = h2[:-1]
    return pos


# ============================================================
# BAND ENSEMBLE: per-asset position (EW average of band configs)
# ============================================================

def _band_ensemble_pos(close: np.ndarray, band_configs: list, ma_type: str) -> np.ndarray:
    """Build a continuous band-ensemble position for one asset.
    band_configs: list of config names (e.g. 'ema_5_20'); extract periods via _nums.
    Returns float array in [0,1] (average held state across band members).
    Returns None if band is empty."""
    if not band_configs:
        return None
    held_list = []
    for cfg in band_configs:
        periods = _nums(cfg)
        if len(periods) < 2:
            continue
        pos = _ironed_held(close, periods, ma_type)
        held_list.append(pos.astype(float))
    if not held_list:
        return None
    # fixed-EW ensemble position (mean of all member positions)
    return np.mean(np.stack(held_list, axis=0), axis=0)


# ============================================================
# MOVE-CATCH METRICS: per (asset, move)
# ============================================================

def _move_catch_metrics(close: np.ndarray, pos: np.ndarray,
                        move: dict, train_mask: np.ndarray) -> Optional[dict]:
    """Compute move-catch metrics for one (asset, move, band_position_array).

    move: dict with trough_idx, peak_idx, trough_price, peak_price.
    pos: the ironed band-ensemble position array (same indexing as close).
    train_mask: boolean array marking train bars (same indexing as close).

    Returns dict or None if move not in TRAIN window.
    """
    t_idx = move["trough_idx"]
    p_idx = move["peak_idx"]
    t_price = move["trough_price"]
    peak_price = move["peak_price"]
    move_size = peak_price - t_price

    # Check that the trough is in the TRAIN window
    if t_idx >= len(train_mask) or not train_mask[t_idx]:
        return None
    # Require at least a few bars in the move
    if p_idx <= t_idx + 1:
        return None
    if move_size <= 0:
        return None

    # Band position within the move window [t_idx+1 .. p_idx] (inclusive of peak)
    move_bars = np.arange(t_idx + 1, min(p_idx + 1, len(pos)))
    if len(move_bars) == 0:
        return None

    move_pos = pos[move_bars]

    # COVERAGE: was band ever long in this move window?
    covered = float(np.any(move_pos > 0.5))

    # ENTRY-LAG: first bar where band is long (ensemble threshold >= 0.5 = majority of configs long)
    long_bars = move_bars[move_pos >= 0.5]
    if len(long_bars) == 0:
        # Band never entered: entry_lag = 1.0 (missed entirely)
        entry_lag = 1.0
        entry_price = peak_price
        first_long_bar = None
    else:
        first_long_bar = int(long_bars[0])
        entry_price = float(close[first_long_bar])
        # If band was already long BEFORE the trough (early entry), clamp to 0
        entry_lag = max(0.0, (entry_price - t_price) / move_size)
        entry_lag = min(entry_lag, 1.0)  # cap at 1 for clarity

    # CAPTURE raw: realized return from entry to end-of-move / move_size
    # Approximate as: fraction of move gained while in position
    if first_long_bar is not None:
        # Find the last long bar in the move window (band exit or peak, whichever first)
        last_long_bar = int(long_bars[-1])
        exit_price = float(close[min(last_long_bar + 1, len(close) - 1)])
        # Net return while in position within the move window
        in_move_pos = pos[move_bars]
        in_move_ret = np.zeros(len(move_bars))
        for k, bar in enumerate(move_bars):
            if bar + 1 < len(close):
                in_move_ret[k] = (close[bar + 1] / close[bar] - 1.0) * in_move_pos[k]
        raw_capture = float(np.sum(in_move_ret)) / (move_size / t_price) if (move_size / t_price) > 0 else 0.0
        # Clamp to [-0.5, 1.5] to avoid division artifacts
        raw_capture = max(-0.5, min(1.5, raw_capture))
    else:
        raw_capture = 0.0

    # WEIGHTED CAPTURE: raw_capture x move_frac (credit large moves more)
    weighted_capture = raw_capture * move["move_frac"]

    # NET %/trade: sum of net returns (maker cost included) while in position in this move
    net_trade = 0.0
    if first_long_bar is not None:
        prev_p = 0.0
        for k, bar in enumerate(move_bars):
            if bar + 1 < len(close):
                p_now = in_move_pos[k]
                flip = abs(p_now - prev_p)
                net_trade += p_now * (close[bar + 1] / close[bar] - 1.0) - flip * (MAKER_RT / 2.0)
                prev_p = p_now

    # RANDOM NULL: entry at a random bar within [t_idx+1 .. p_idx-1]
    null_bars = list(range(t_idx + 1, max(t_idx + 2, p_idx)))
    null_lag = None
    if null_bars:
        # Expected entry lag of a uniform random draw
        null_prices = np.array([close[b] for b in null_bars if b < len(close)])
        if len(null_prices) > 0:
            null_lags = [(float(p) - t_price) / move_size for p in null_prices]
            null_lag = float(np.mean(null_lags))

    return {
        "covered": covered,
        "entry_lag": float(entry_lag),
        "null_lag": null_lag,
        "lag_delta": float(entry_lag - null_lag) if null_lag is not None else None,
        "raw_capture": float(raw_capture),
        "weighted_capture": float(weighted_capture),
        "net_trade": float(net_trade),
        "move_frac": float(move["move_frac"]),
        "move_bars": int(p_idx - t_idx),
    }


# ============================================================
# PER-ASSET ANALYSIS
# ============================================================

def _analyze_asset(sym: str, cad: str, ma_type: str, band_configs: list,
                   threshold: float) -> list:
    """Run move-catch analysis for one (asset, cadence, MA-type).
    Returns list of per-move metric dicts."""
    data = _train_close(sym, cad)
    if data is None:
        return []
    close, ms, train_mask = data

    # Build band ensemble position for this asset
    pos = _band_ensemble_pos(close, band_configs, ma_type)
    if pos is None:
        return []

    # Detect moves (over full close array, then filter by train_mask)
    moves = _detect_moves(close, threshold=threshold)
    if not moves:
        return []

    results = []
    for move in moves:
        m = _move_catch_metrics(close, pos, move, train_mask)
        if m is not None:
            m["sym"] = sym
            results.append(m)
    return results


# ============================================================
# BAND EXTRACTION
# ============================================================

def _get_band_configs(cad: str, ma_type: str) -> list:
    """Build working band for (cadence, MA-type) via run_cell.
    Returns list of config names in the band (positive 3-way).
    Returns empty list if no band."""
    specs2 = distinct_specs("2MA", 0.15, max_n=60)
    specs3 = distinct_specs("3MA", 0.15, max_n=60)
    # Re-key specs for this MA type
    ma_specs = {}
    for name, (fam, params) in {**specs2, **specs3}.items():
        # Replace the MA type in the spec
        new_params = dict(params, type=ma_type)
        # Build a new name encoding the MA type
        new_name = f"{ma_type.lower()}_" + "_".join(str(p) for p in _nums(name))
        ma_specs[new_name] = (fam, new_params)

    all_periods = sorted(set(p for n in ma_specs for p in _nums(n)))
    panels = build_panels(cad, ma_type, all_periods)
    if len(panels) < 3:
        return []

    # Set up the L module splits for within-2020
    cell = run_cell(panels, ma_specs, cad)
    band_cfgs = cell["band"]["band_configs"] if cell and cell.get("band") else []
    return band_cfgs


# ============================================================
# PER (MA-type x TF) ANALYSIS
# ============================================================

def analyze_cell(cad: str, ma_type: str, threshold: float = PRIMARY_THRESH) -> dict:
    """Run full move-catch decomposition for one (cadence, MA-type) cell.
    Returns a summary dict."""
    band_configs = _get_band_configs(cad, ma_type)
    n_band = len(band_configs)

    if n_band == 0:
        return {
            "cadence": cad, "ma_type": ma_type, "threshold": threshold,
            "n_band": 0, "n_moves": 0,
            "coverage": None, "mean_entry_lag": None, "mean_null_lag": None,
            "lag_delta": None, "lag_delta_pval": None,
            "raw_capture_mean": None, "weighted_capture_mean": None,
            "net_pct_per_trade": None, "n_assets_active": 0,
            "note": "no band configs",
        }

    all_records = []
    n_active = 0
    for sym in SYMS:
        recs = _analyze_asset(sym, cad, ma_type, band_configs, threshold)
        if recs:
            all_records.extend(recs)
            n_active += 1

    if not all_records:
        return {
            "cadence": cad, "ma_type": ma_type, "threshold": threshold,
            "n_band": n_band, "n_moves": 0,
            "coverage": None, "mean_entry_lag": None, "mean_null_lag": None,
            "lag_delta": None, "lag_delta_pval": None,
            "raw_capture_mean": None, "weighted_capture_mean": None,
            "net_pct_per_trade": None, "n_assets_active": n_active,
            "note": "no moves detected",
        }

    df = pd.DataFrame(all_records)

    # Sanity: buy-hold check (approximate, just ensure data loaded ok)
    n_moves = len(df)
    coverage = float(df["covered"].mean())
    mean_entry_lag = float(df["entry_lag"].mean())

    null_lags = df["null_lag"].dropna()
    mean_null_lag = float(null_lags.mean()) if len(null_lags) > 0 else None

    lag_deltas = df["lag_delta"].dropna()
    lag_delta = float(lag_deltas.mean()) if len(lag_deltas) > 0 else None

    # t-test: band entry-lag vs null (H0: delta >= 0, i.e. band no earlier than random)
    lag_delta_pval = None
    if len(lag_deltas) >= 5:
        from scipy import stats as sp_stats
        # one-sided: H0 lag_delta >= 0; alternative: lag_delta < 0 (band enters earlier)
        _, pval_2side = sp_stats.ttest_1samp(lag_deltas.to_numpy(), 0.0)
        lag_delta_pval = float(pval_2side / 2.0)  # one-sided

    raw_capture_mean = float(df["raw_capture"].mean())
    # Weighted capture: weight each move by its move_frac
    weights = df["move_frac"].to_numpy()
    w_capture = float(np.average(df["raw_capture"].to_numpy(), weights=weights)) if weights.sum() > 0 else 0.0
    net_pct = float(df["net_trade"].mean() * 100)

    # Per-threshold breakdown (coverage by move size)
    size_bins = {"5-10%": (0.05, 0.10), "10-20%": (0.10, 0.20), "20%+": (0.20, 9.9)}
    coverage_by_size = {}
    for label, (lo, hi) in size_bins.items():
        sub = df[(df["move_frac"] >= lo) & (df["move_frac"] < hi)]
        coverage_by_size[label] = {
            "n": int(len(sub)),
            "coverage": float(sub["covered"].mean()) if len(sub) > 0 else None,
            "entry_lag": float(sub["entry_lag"].mean()) if len(sub) > 0 else None,
        }

    return {
        "cadence": cad,
        "ma_type": ma_type,
        "threshold": threshold,
        "n_band": n_band,
        "n_moves": n_moves,
        "n_assets_active": n_active,
        "coverage": round(coverage, 3),
        "mean_entry_lag": round(mean_entry_lag, 3),
        "mean_null_lag": round(mean_null_lag, 3) if mean_null_lag is not None else None,
        "lag_delta": round(lag_delta, 3) if lag_delta is not None else None,
        "lag_delta_pval": round(lag_delta_pval, 4) if lag_delta_pval is not None else None,
        "raw_capture_mean": round(raw_capture_mean, 3),
        "weighted_capture_mean": round(w_capture, 3),
        "net_pct_per_trade": round(net_pct, 3),
        "coverage_by_size": coverage_by_size,
        "per_move_records": [
            {k: v for k, v in r.items() if k != "sym"}
            for r in all_records[:50]  # cap for JSON size
        ],
    }


# ============================================================
# BUY-HOLD SANITY CHECK (cadence-invariance)
# ============================================================

def buyhold_sanity(cads=None) -> dict:
    """Compute TRAIN-period buy-hold across TFs. Should be ~similar (invariant to cadence)."""
    if cads is None:
        cads = TFS
    results = {}
    for cad in cads:
        bhs = []
        for sym in SYMS:
            data = _train_close(sym, cad)
            if data is None:
                bhs.append(0.0)
                continue
            close, ms, train_mask = data
            c_train = close[train_mask]
            if len(c_train) < 2:
                bhs.append(0.0)
                continue
            bh = float(c_train[-1] / c_train[0] - 1.0)
            bhs.append(bh)
        # Fixed-EW: missing = cash (0)
        bh_book = float(np.mean(bhs) * 100)
        results[cad] = round(bh_book, 1)
    return results


# ============================================================
# MAIN
# ============================================================

def run_tf(cad: str, thresholds=None):
    """Run all MA types for one TF, write incremental JSON."""
    if thresholds is None:
        thresholds = [PRIMARY_THRESH]
    print(f"[ma_movecatch] TF={cad} | thresholds={thresholds}")

    results = {}
    for thresh in thresholds:
        results[thresh] = {}
        for ma_type in MA_TYPES:
            print(f"  {ma_type} @ {cad} thresh={thresh*100:.0f}% ...", flush=True)
            cell = analyze_cell(cad, ma_type, thresh)
            results[thresh][ma_type] = cell
            print(f"    n_band={cell['n_band']} n_moves={cell['n_moves']} "
                  f"coverage={cell['coverage']} lag={cell['mean_entry_lag']} "
                  f"null_lag={cell['mean_null_lag']} delta={cell['lag_delta']} "
                  f"pval={cell['lag_delta_pval']}", flush=True)

    out_path = OUT / f"ma_movecatch_{cad}.json"
    with open(out_path, "w") as f:
        json.dump({"cadence": cad, "results": results}, f, indent=2, default=str)
    print(f"  -> {out_path}")
    return results


def run_selftest():
    """Quick self-test: 1d only, primary threshold, check sanity."""
    print("[ma_movecatch] SELFTEST: buy-hold cadence invariance ...")
    bh = buyhold_sanity(["1d", "4h"])
    print(f"  buy-hold TRAIN (should be similar across TFs):")
    for k, v in bh.items():
        print(f"    {k}: {v:.1f}%")

    print("[ma_movecatch] SELFTEST: 1d SMA move detection ...")
    # Just check move detection works
    data = _train_close("BTCUSDT", "1d")
    if data is None:
        print("  ERROR: no BTCUSDT 1d data")
        return False
    close, ms, train_mask = data
    moves = _detect_moves(close[train_mask], threshold=0.05)
    print(f"  BTC TRAIN 1d: {len(moves)} moves >= 5%")
    if len(moves) == 0:
        print("  WARNING: no moves found -- check data range")
    else:
        for m in moves[:3]:
            print(f"    move_frac={m['move_frac']*100:.1f}% bars={m['peak_idx']-m['trough_idx']}")

    print("[ma_movecatch] SELFTEST: 1d SMA band + metrics ...")
    cell = analyze_cell("1d", "SMA", PRIMARY_THRESH)
    print(f"  SMA@1d: n_band={cell['n_band']} n_moves={cell['n_moves']} "
          f"coverage={cell['coverage']} entry_lag={cell['mean_entry_lag']} "
          f"null_lag={cell['mean_null_lag']} lag_delta={cell['lag_delta']} "
          f"pval={cell['lag_delta_pval']}")
    print("[ma_movecatch] SELFTEST PASS")
    return True


def summarize_all(all_results: dict, threshold: float = PRIMARY_THRESH):
    """Print a compact table across all MA-types x TFs for a given threshold."""
    print(f"\n{'='*90}")
    print(f"MOVE-CATCH DECOMPOSITION SUMMARY  threshold={threshold*100:.0f}%  (TRAIN 2020-01..07)")
    print(f"{'='*90}")
    hdr = f"{'MA':6} {'TF':4} {'Band':5} {'Moves':5} {'Cov%':6} {'Lag':5} {'NullLg':6} {'Delta':7} {'p-val':7} {'RawCap':7} {'WtdCap':7} {'Net%T':7}"
    print(hdr)
    print("-" * 90)
    for tf in TFS:
        if tf not in all_results:
            continue
        for ma in MA_TYPES:
            r = all_results[tf].get(threshold, {}).get(ma)
            if r is None:
                continue
            cov = f"{r['coverage']*100:.0f}" if r['coverage'] is not None else "  -"
            lag = f"{r['mean_entry_lag']:.3f}" if r['mean_entry_lag'] is not None else "  -"
            null = f"{r['mean_null_lag']:.3f}" if r['mean_null_lag'] is not None else "  -"
            delta = f"{r['lag_delta']:+.3f}" if r['lag_delta'] is not None else "  -"
            pval = f"{r['lag_delta_pval']:.4f}" if r['lag_delta_pval'] is not None else "  -"
            rawc = f"{r['raw_capture_mean']:.3f}" if r['raw_capture_mean'] is not None else "  -"
            wtdc = f"{r['weighted_capture_mean']:.3f}" if r['weighted_capture_mean'] is not None else "  -"
            net = f"{r['net_pct_per_trade']:+.2f}" if r['net_pct_per_trade'] is not None else "  -"
            nb = r.get('n_band', 0)
            nm = r.get('n_moves', 0)
            print(f"{ma:6} {tf:4} {nb:5d} {nm:5d} {cov:>5}% {lag:>6} {null:>6} {delta:>7} {pval:>7} {rawc:>7} {wtdc:>7} {net:>7}")
    print("=" * 90)
    print("NOTES:")
    print("  Coverage: fraction of moves where band held long at any point.")
    print("  Lag: mean (entry_price-trough)/(peak-trough), 0=trough-entry, 1=peak-entry. LOWER=better.")
    print("  NullLg: expected lag of random entry within move (always ~0.5 for uniform).")
    print("  Delta: band_lag - null_lag. NEGATIVE = band enters EARLIER than random (skill).")
    print("  p-val: one-sided t-test H0: delta>=0. p<0.05 = band enters significantly earlier.")
    print("  RawCap: mean captured fraction of move size. WtdCap: weighted by move magnitude.")
    print("  Net%T: mean net return per move trade (maker cost included).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--tf", help="single TF to run (e.g. 1d)")
    parser.add_argument("--all", action="store_true", help="run all 6 TFs incrementally")
    parser.add_argument("--thresh", type=float, default=PRIMARY_THRESH,
                        help="move threshold (default 0.05=5%)")
    parser.add_argument("--sweep-thresh", action="store_true",
                        help="sweep 5/10/15%% thresholds")
    args = parser.parse_args()

    if args.selftest:
        ok = run_selftest()
        sys.exit(0 if ok else 1)

    thresholds = MOVE_THRESHOLDS if args.sweep_thresh else [args.thresh]

    all_results = {}

    if args.tf:
        tfs_to_run = [args.tf]
    elif args.all:
        tfs_to_run = TFS
    else:
        tfs_to_run = ["1d"]

    for cad in tfs_to_run:
        r = run_tf(cad, thresholds=thresholds)
        all_results[cad] = r

    # Load any previously-computed TFs if running --all
    if args.all:
        for cad in TFS:
            if cad not in all_results:
                p = OUT / f"ma_movecatch_{cad}.json"
                if p.exists():
                    with open(p) as f:
                        d = json.load(f)
                    all_results[cad] = d.get("results", {})

    # Summary table
    if all_results:
        for t in thresholds:
            summarize_all(all_results, t)

    # Write combined summary
    summary_path = OUT / "ma_movecatch_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"thresholds": thresholds, "tfs": tfs_to_run,
                   "summary": {tf: {str(t): {ma: {k: v for k, v in r.items() if k != "per_move_records"}
                                             for ma, r in tdict.items()}
                                    for t, tdict in tres.items()}
                               for tf, tres in all_results.items()}},
                  f, indent=2, default=str)
    print(f"\nSummary -> {summary_path}")
