"""
W2 — Walk-forward WITHIN TRAIN setup discovery (MAXX-INST-2026-05-26-NIGHT).

Brief: docs/dossiers/PEPE_TRAIN_WALK_FORWARD_SURVIVORS_2026_05_27.md
Protocol: TRAIN-ONLY. VAL/OOS/UNSEEN never touched. Past-only at every step.

Methodology:
  TRAIN = first 556 days of PEPE 1d chimera (~2023-05-06 → 2024-11-11).
  Rolling WF: train sub-window = 180d, test sub-window = 90d, step = 30d.
    Fold k (k=0..9): train [k*30, k*30+180), test [k*30+180, k*30+270).
    10 folds in total.
  Per fold:
    - identify mover days (>=2% daily return) in train sub-window
    - sweep MA/EMA/WMA × (fast, slow) on train sub-window mover days
    - score each setup with unified rule: median_capture × sqrt(fire_rate) × win_factor
    - pick top-3 train setups; evaluate same setups on test sub-window mover days
    - record train + test metrics per setup

Survivor rule:
  PRIMARY: train-top-3 in >=6 folds AND test_median_capture > 0 in >=50% of those folds
  FALLBACK: train-top-5 in >=4 folds AND test_median > 0 in >=50% of those folds
  If neither passes -> HONEST NULL

Output:
  runs/mover_day/pepe_train_wf_summary_2026_05_27.json
  runs/mover_day/pepe_train_wf_per_fold_<cadence>_2026_05_27.json
  docs/dossiers/PEPE_TRAIN_WALK_FORWARD_SURVIVORS_2026_05_27.md

canonical_seeds = {bag_seed: 220, feat_seed: 1220, rng_seed: 8120}
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import polars as pl
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CHIM_1D = ROOT / "data/processed/chimera/1d/pepeusdt_v51_chimera_1d_20260522.parquet"
CHIM_PATHS = {
    "30m": ROOT / "data/processed/chimera/30m/pepeusdt_v51_chimera_30m_20260522.parquet",
    "1h":  ROOT / "data/processed/chimera/1h/pepeusdt_v51_chimera_1h_20260522.parquet",
    "4h":  ROOT / "data/processed/chimera/4h/pepeusdt_v51_chimera_4h_20260522.parquet",
}
OUT_DIR = ROOT / "runs/mover_day"
DOSSIER_DIR = ROOT / "docs/dossiers"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DOSSIER_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_SEEDS = {"bag_seed": 220, "feat_seed": 1220, "rng_seed": 8120}

# ----- TRAIN window contract -----
TRAIN_FRAC = 0.50
PURGE_GAP_BARS = 400  # invariant from CLAUDE.md (informational only here; no model leak)

# ----- WF fold geometry (days) -----
WF_TRAIN_DAYS = 180
WF_TEST_DAYS = 90
WF_STEP_DAYS = 30

# ----- Cadence-specific grids (match unify_selection_and_permutation source) -----
CADENCE_CFGS = {
    "30m": {
        "fast_grid": [6, 10, 14, 18, 24, 30],
        "slow_grid": [24, 30, 42, 60, 84, 120],
        "lookback_bars": 144,
        "bar_ms": 30 * 60 * 1000,
    },
    "1h": {
        "fast_grid": [5, 8, 9, 10, 13, 20, 30],
        "slow_grid": [21, 30, 50, 100, 200],
        "lookback_bars": 72,
        "bar_ms": 60 * 60 * 1000,
    },
    "4h": {
        "fast_grid": [3, 5, 7, 9, 12, 15],
        "slow_grid": [9, 12, 18, 24, 36, 48],
        "lookback_bars": 18,
        "bar_ms": 4 * 60 * 60 * 1000,
    },
}

INDICATOR_TYPES = ["SMA", "EMA", "WMA"]

# Survivor thresholds
PRIMARY_TOPK = 3
PRIMARY_MIN_FOLDS = 6
PRIMARY_TEST_POSITIVE_FRAC = 0.50

FALLBACK_TOPK = 5
FALLBACK_MIN_FOLDS = 4
FALLBACK_TEST_POSITIVE_FRAC = 0.50


# =======================================================================
#                        Pure helpers (numpy)
# =======================================================================

def get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT
        ).decode().strip()
    except Exception:
        return "unknown"


def compute_ma(prices: np.ndarray, kind: str, window: int) -> np.ndarray:
    """Pure past-only MA at each index. Returns NaN until window-1 bars accumulated."""
    n = len(prices)
    out = np.full(n, np.nan)
    if n < window:
        return out
    if kind == "SMA":
        csum = np.cumsum(prices)
        out[window - 1:] = (csum[window - 1:] - np.concatenate([[0], csum[:-window]])) / window
    elif kind == "EMA":
        alpha = 2.0 / (window + 1.0)
        seed = prices[:window].mean()
        out[window - 1] = seed
        for i in range(window, n):
            out[i] = alpha * prices[i] + (1 - alpha) * out[i - 1]
    elif kind == "WMA":
        w = np.arange(1, window + 1, dtype=float)
        w_sum = w.sum()
        for i in range(window - 1, n):
            out[i] = (prices[i - window + 1:i + 1] * w).sum() / w_sum
    return out


def first_cross_up(fast_ma: np.ndarray, slow_ma: np.ndarray, start: int, end: int) -> int:
    """Return first index in [max(start,1), end) where fast crosses above slow. -1 if none.

    Cross condition: fast[i-1] <= slow[i-1] AND fast[i] > slow[i].
    """
    s = max(start, 1)
    if s >= end:
        return -1
    prev_f = fast_ma[s - 1:end - 1]
    prev_s = slow_ma[s - 1:end - 1]
    cur_f = fast_ma[s:end]
    cur_s = slow_ma[s:end]
    mask = (
        np.isfinite(prev_f) & np.isfinite(prev_s) &
        np.isfinite(cur_f) & np.isfinite(cur_s) &
        (prev_f <= prev_s) & (cur_f > cur_s)
    )
    if not mask.any():
        return -1
    return s + int(np.argmax(mask))


def unified_score(median_capture: float, fire_rate: float, win_rate: float) -> float:
    """score = median_capture * sqrt(fire_rate) * win_factor.

    win_factor = 1.0 if win_rate >= 0.50 else 0.5.
    Returns -inf if non-finite or zero fires.
    """
    if not np.isfinite(median_capture) or fire_rate <= 0.0:
        return float("-inf")
    win_factor = 1.0 if win_rate >= 0.50 else 0.5
    return median_capture * (fire_rate ** 0.5) * win_factor


def setup_label(setup_tuple) -> str:
    kind, fast, slow = setup_tuple
    return f"{kind} {fast}/{slow}"


# =======================================================================
#                        TRAIN window + folds
# =======================================================================

def load_train_1d() -> Tuple[pl.DataFrame, Dict[str, Any]]:
    df = pl.read_parquet(CHIM_1D).sort("timestamp")
    n = len(df)
    train_end_bar = int(n * TRAIN_FRAC)  # exclusive
    df_train = df[:train_end_bar].with_columns(
        pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("dt")
    )
    # daily return
    closes = df_train["close"].to_numpy()
    daily_ret = np.zeros(len(closes))
    daily_ret[1:] = closes[1:] / closes[:-1] - 1.0
    df_train = df_train.with_columns(pl.Series("daily_return", daily_ret))

    meta = {
        "n_train_days": int(len(df_train)),
        "train_start": str(df_train["dt"][0].date()),
        "train_end": str(df_train["dt"][-1].date()),
        "train_start_ts_ms": int(df_train["timestamp"][0]),
        "train_end_ts_ms": int(df_train["timestamp"][-1]),
    }
    return df_train, meta


def define_folds(n_train_days: int) -> List[Dict[str, int]]:
    """Return list of fold dicts with day indices.

    Fold k: train [k*step, k*step+T_train), test [k*step+T_train, k*step+T_train+T_test).
    Last valid k is the largest where k*step+T_train+T_test <= n_train_days.
    """
    folds = []
    k = 0
    while True:
        tr_lo = k * WF_STEP_DAYS
        tr_hi = tr_lo + WF_TRAIN_DAYS
        te_lo = tr_hi
        te_hi = te_lo + WF_TEST_DAYS
        if te_hi > n_train_days:
            break
        folds.append({
            "fold_id": k,
            "train_day_lo": tr_lo, "train_day_hi": tr_hi,
            "test_day_lo": te_lo, "test_day_hi": te_hi,
        })
        k += 1
    return folds


# =======================================================================
#                  Per-cadence capture matrix builder
# =======================================================================

def build_capture_for_movers(
    cadence: str,
    mover_ts_list: List[int],
    df_cad_ts: np.ndarray,
    df_cad_opens: np.ndarray,
    df_cad_closes: np.ndarray,
    ma_cache: Dict[Tuple[str, int], np.ndarray],
    setups: List[Tuple[str, int, int]],
    lookback_bars: int,
) -> np.ndarray:
    """Capture matrix shape (n_movers, n_setups). NaN if no cross/no clean entry.

    For each mover day:
      - find the cadence bar at/before daily close
      - look back `lookback_bars` for first MA cross-up within [window_start+1, day_close_idx+1)
      - entry = next bar open; capture = daily_close_price / entry_open - 1
    """
    n_movers = len(mover_ts_list)
    n_setups = len(setups)
    cap = np.full((n_movers, n_setups), np.nan)
    for mi, day_ts in enumerate(mover_ts_list):
        day_close_idx = int(np.searchsorted(df_cad_ts, day_ts, side="right") - 1)
        if day_close_idx < 0 or day_close_idx >= len(df_cad_ts):
            continue
        window_start = max(0, day_close_idx - (lookback_bars - 1))
        signal_search_start = window_start + 1
        signal_search_end = day_close_idx + 1
        daily_close_price = df_cad_closes[day_close_idx]
        if daily_close_price <= 0:
            continue
        for si, (kind, fast, slow) in enumerate(setups):
            fma = ma_cache[(kind, fast)]
            sma = ma_cache[(kind, slow)]
            cross_idx = first_cross_up(fma, sma, signal_search_start, signal_search_end)
            if cross_idx == -1:
                continue
            entry_i = cross_idx + 1
            if entry_i > day_close_idx:
                continue
            ent_p = df_cad_opens[entry_i]
            if ent_p <= 0:
                continue
            cap[mi, si] = daily_close_price / ent_p - 1.0
    return cap


def per_setup_stats_from_capture(cap_vec: np.ndarray, n_movers_in_window: int) -> Dict[str, float]:
    """Stats for a single setup column over a window of movers.

    fire_rate = denom n_movers_in_window (i.e., over all mover days, not just those that fired).
    """
    fired = np.isfinite(cap_vec)
    n_fires = int(fired.sum())
    if n_fires == 0 or n_movers_in_window == 0:
        return {
            "n_movers_window": int(n_movers_in_window),
            "n_fires": 0,
            "fire_rate": 0.0,
            "median_capture": float("nan"),
            "mean_capture": float("nan"),
            "win_rate": float("nan"),
            "score": float("-inf"),
            "p10": float("nan"),
            "p90": float("nan"),
            "max_dd_capture": float("nan"),
        }
    arr = cap_vec[fired]
    med = float(np.median(arr))
    fr = float(n_fires / n_movers_in_window)
    wr = float(np.sum(arr > 0) / n_fires)
    return {
        "n_movers_window": int(n_movers_in_window),
        "n_fires": int(n_fires),
        "fire_rate": fr,
        "median_capture": med,
        "mean_capture": float(np.mean(arr)),
        "win_rate": wr,
        "score": unified_score(med, fr, wr),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "max_dd_capture": float(np.min(arr)),
    }


# =======================================================================
#                       Cadence WF mining
# =======================================================================

def mine_cadence_wf(
    cadence: str,
    df_train_1d: pl.DataFrame,
    folds: List[Dict[str, int]],
) -> Dict[str, Any]:
    """Run WF mining for one cadence.

    Returns a dict with per-fold setup-stats (train + test).
    """
    cfg = CADENCE_CFGS[cadence]
    print(f"\n[{cadence}] Loading chimera...")
    df_cad = pl.read_parquet(CHIM_PATHS[cadence]).sort("timestamp")
    # Bound by TRAIN end + small tail
    train_end_ts = int(df_train_1d["timestamp"][-1])
    cap_ts = train_end_ts + 2 * 24 * 3600 * 1000  # +2d cushion
    df_cad = df_cad.filter(pl.col("timestamp") <= cap_ts)
    cad_ts = df_cad["timestamp"].to_numpy()
    cad_opens = df_cad["open"].to_numpy()
    cad_closes = df_cad["close"].to_numpy()
    print(f"[{cadence}] cadence bars in TRAIN tail: {len(cad_ts)}")

    # Precompute MAs ONCE on full TRAIN cadence series (past-only at each index)
    unique_windows = sorted(set(cfg["fast_grid"] + cfg["slow_grid"]))
    ma_cache: Dict[Tuple[str, int], np.ndarray] = {}
    for kind in INDICATOR_TYPES:
        for w in unique_windows:
            ma_cache[(kind, w)] = compute_ma(cad_closes, kind, w)

    # Setup grid
    setups: List[Tuple[str, int, int]] = []
    for kind in INDICATOR_TYPES:
        for fast in cfg["fast_grid"]:
            for slow in cfg["slow_grid"]:
                if fast >= slow:
                    continue
                setups.append((kind, fast, slow))
    n_setups = len(setups)
    print(f"[{cadence}] n_setups={n_setups}, n_folds={len(folds)}")

    # Identify TRAIN mover days from 1d
    daily_ret = df_train_1d["daily_return"].to_numpy()
    ts_1d = df_train_1d["timestamp"].to_numpy()
    # Per CLAUDE.md and W1 baseline: >=2% daily up move
    is_mover = daily_ret >= 0.02
    mover_day_indices = np.where(is_mover)[0]
    mover_day_ts = ts_1d[mover_day_indices]
    print(f"[{cadence}] TRAIN total mover days (>=2%): {len(mover_day_ts)}")

    # Build the FULL capture matrix once over all TRAIN movers (efficient)
    full_cap = build_capture_for_movers(
        cadence=cadence,
        mover_ts_list=mover_day_ts.tolist(),
        df_cad_ts=cad_ts,
        df_cad_opens=cad_opens,
        df_cad_closes=cad_closes,
        ma_cache=ma_cache,
        setups=setups,
        lookback_bars=cfg["lookback_bars"],
    )
    print(f"[{cadence}] full capture matrix shape: {full_cap.shape}, finite frac: {np.isfinite(full_cap).mean():.3f}")

    # For each fold, compute train stats and test stats per setup
    per_fold = []
    for fd in folds:
        # Train sub-window day range
        tr_lo, tr_hi = fd["train_day_lo"], fd["train_day_hi"]
        te_lo, te_hi = fd["test_day_lo"], fd["test_day_hi"]

        # Filter mover indices that fall in each day range
        tr_mask = (mover_day_indices >= tr_lo) & (mover_day_indices < tr_hi)
        te_mask = (mover_day_indices >= te_lo) & (mover_day_indices < te_hi)
        tr_idx_local = np.where(tr_mask)[0]  # indices into full_cap rows
        te_idx_local = np.where(te_mask)[0]
        tr_n_movers = int(tr_mask.sum())
        te_n_movers = int(te_mask.sum())

        # Per-setup stats for train and test
        per_setup = []
        for si, st in enumerate(setups):
            tr_stats = per_setup_stats_from_capture(full_cap[tr_idx_local, si], tr_n_movers)
            te_stats = per_setup_stats_from_capture(full_cap[te_idx_local, si], te_n_movers)
            per_setup.append({
                "setup": setup_label(st),
                "kind": st[0], "fast": st[1], "slow": st[2],
                "train": tr_stats,
                "test": te_stats,
            })
        # Sort descending by train score
        per_setup.sort(key=lambda r: -r["train"]["score"] if np.isfinite(r["train"]["score"]) else float("-inf"))

        # Train ranks
        # (filter out -inf trains -- they didn't fire)
        rank_pos = 1
        for r in per_setup:
            if np.isfinite(r["train"]["score"]):
                r["train_rank"] = rank_pos
                rank_pos += 1
            else:
                r["train_rank"] = None

        per_fold.append({
            "fold_id": fd["fold_id"],
            "train_day_range": [tr_lo, tr_hi],
            "test_day_range": [te_lo, te_hi],
            "train_start_date": str(df_train_1d["dt"][tr_lo].date()) if tr_lo < len(df_train_1d) else None,
            "train_end_date":   str(df_train_1d["dt"][tr_hi - 1].date()) if (tr_hi - 1) < len(df_train_1d) else None,
            "test_start_date":  str(df_train_1d["dt"][te_lo].date()) if te_lo < len(df_train_1d) else None,
            "test_end_date":    str(df_train_1d["dt"][te_hi - 1].date()) if (te_hi - 1) < len(df_train_1d) else None,
            "tr_n_movers": tr_n_movers,
            "te_n_movers": te_n_movers,
            "per_setup": per_setup,
        })
        # Quick top-3 dump
        top3 = [r["setup"] for r in per_setup[:3] if r["train_rank"] is not None]
        print(f"[{cadence}] Fold {fd['fold_id']:2d}: tr_n_mov={tr_n_movers} te_n_mov={te_n_movers} top3={top3}")

    return {
        "cadence": cadence,
        "n_setups": n_setups,
        "setups": [setup_label(s) for s in setups],
        "n_train_movers_total": int(len(mover_day_ts)),
        "folds": per_fold,
    }


# =======================================================================
#                  Survivor aggregation
# =======================================================================

def aggregate_survivors(per_cad_result: Dict[str, Any], top_k: int, min_folds: int, test_pos_frac: float) -> List[Dict[str, Any]]:
    """For one cadence, find all setups appearing in train top-K of at least `min_folds` folds
    AND test_median_capture > 0 in at least `test_pos_frac` of those folds.

    Returns a list of survivor dicts, sorted by (n_folds_top_K, test_median_avg).
    """
    folds = per_cad_result["folds"]
    n_folds = len(folds)
    # Aggregate by setup label
    by_setup: Dict[str, Dict[str, Any]] = {}
    for fd in folds:
        per_setup = fd["per_setup"]
        # train rank assigned for setups that fired
        for r in per_setup:
            rank = r.get("train_rank")
            if rank is None or rank > top_k:
                continue
            lab = r["setup"]
            if lab not in by_setup:
                by_setup[lab] = {
                    "setup": lab,
                    "kind": r["kind"], "fast": r["fast"], "slow": r["slow"],
                    "folds_appeared_top_k": [],  # list of fold dicts with train+test
                }
            by_setup[lab]["folds_appeared_top_k"].append({
                "fold_id": fd["fold_id"],
                "train_start_date": fd["train_start_date"],
                "train_end_date": fd["train_end_date"],
                "test_start_date": fd["test_start_date"],
                "test_end_date": fd["test_end_date"],
                "train_rank": rank,
                "train_median": r["train"]["median_capture"],
                "train_fire_rate": r["train"]["fire_rate"],
                "train_win_rate": r["train"]["win_rate"],
                "train_n_fires": r["train"]["n_fires"],
                "train_score": r["train"]["score"],
                "test_median": r["test"]["median_capture"],
                "test_mean": r["test"]["mean_capture"],
                "test_fire_rate": r["test"]["fire_rate"],
                "test_win_rate": r["test"]["win_rate"],
                "test_n_fires": r["test"]["n_fires"],
                "test_score": r["test"]["score"],
                "tr_n_movers": fd["tr_n_movers"],
                "te_n_movers": fd["te_n_movers"],
            })

    # Survivor filter
    survivors = []
    for lab, blk in by_setup.items():
        appeared = blk["folds_appeared_top_k"]
        n_app = len(appeared)
        if n_app < min_folds:
            continue
        # test_positive fraction among appearances where test fired at least once
        test_meds = [r["test_median"] for r in appeared if np.isfinite(r["test_median"])]
        n_test_evaluable = len(test_meds)
        if n_test_evaluable == 0:
            test_pos_count = 0
            test_pos_pct = 0.0
        else:
            test_pos_count = int(sum(1 for v in test_meds if v > 0))
            test_pos_pct = test_pos_count / n_test_evaluable
        # require test-positive in >= test_pos_frac of EVALUABLE test folds (n_test_evaluable >= 0.5*n_app)
        # Use the cleaner rule: pos count / n_app >= test_pos_frac (no inflation if some test folds had no fires)
        test_pos_pct_strict = test_pos_count / n_app

        # Aggregate stats
        train_med_avg = float(np.median([r["train_median"] for r in appeared if np.isfinite(r["train_median"])]) ) if appeared else float("nan")
        test_med_avg = float(np.median(test_meds)) if test_meds else float("nan")
        test_fire_avg = float(np.mean([r["test_fire_rate"] for r in appeared]))
        test_win_avg = float(np.mean([r["test_win_rate"] for r in appeared if np.isfinite(r["test_win_rate"])])) if any(np.isfinite(r["test_win_rate"]) for r in appeared) else float("nan")
        passes = (n_app >= min_folds) and (test_pos_pct_strict >= test_pos_frac)
        survivors.append({
            "setup": lab,
            "kind": blk["kind"], "fast": blk["fast"], "slow": blk["slow"],
            "n_folds_top_k": n_app,
            "n_test_evaluable": n_test_evaluable,
            "test_positive_count": test_pos_count,
            "test_positive_pct_of_appearances": test_pos_pct_strict,
            "test_positive_pct_of_evaluable": test_pos_pct,
            "train_median_avg_across_folds": train_med_avg,
            "test_median_avg_across_folds": test_med_avg,
            "test_fire_rate_avg_across_folds": test_fire_avg,
            "test_win_rate_avg_across_folds": test_win_avg,
            "passes_survivor_rule": passes,
            "fold_details": appeared,
        })

    # Sort: passes first, then by n_folds_top_k desc, then test_median_avg desc
    survivors.sort(key=lambda s: (
        -int(s["passes_survivor_rule"]),
        -s["n_folds_top_k"],
        -(s["test_median_avg_across_folds"] if np.isfinite(s["test_median_avg_across_folds"]) else float("-inf")),
    ))
    return survivors


def classify_regime_pattern(survivor: Dict[str, Any]) -> str:
    """Heuristic: look at which folds the setup wins vs loses to suggest regime conditioning.

    Returns a one-line summary string.
    """
    wins = [f for f in survivor["fold_details"] if (f["test_median"] is not None and np.isfinite(f["test_median"]) and f["test_median"] > 0)]
    losses = [f for f in survivor["fold_details"] if (f["test_median"] is not None and np.isfinite(f["test_median"]) and f["test_median"] <= 0)]
    if not wins or not losses:
        return f"NO_LOSSES_TO_DIFFERENTIATE (wins={len(wins)}, losses={len(losses)})"
    win_dates = ",".join(f["test_start_date"] for f in wins)
    loss_dates = ",".join(f["test_start_date"] for f in losses)
    return f"WINS_AT={{{win_dates}}}; LOSSES_AT={{{loss_dates}}}"


# =======================================================================
#                          Main
# =======================================================================

def main():
    print("=" * 78)
    print("PEPE TRAIN WALK-FORWARD SURVIVORS — W2 MAXX-INST-2026-05-26-NIGHT")
    print("=" * 78)
    t0 = time.time()

    # Step 1: TRAIN window
    df_train_1d, meta_train = load_train_1d()
    print(f"[setup] TRAIN: {meta_train['train_start']} -> {meta_train['train_end']} ({meta_train['n_train_days']}d)")

    # Step 2: WF folds
    folds = define_folds(meta_train["n_train_days"])
    print(f"[setup] WF folds: {len(folds)} (T_train={WF_TRAIN_DAYS}d, T_test={WF_TEST_DAYS}d, step={WF_STEP_DAYS}d)")
    for fd in folds:
        print(f"   fold {fd['fold_id']:2d}: train[{fd['train_day_lo']:3d}..{fd['train_day_hi']:3d}) test[{fd['test_day_lo']:3d}..{fd['test_day_hi']:3d})")

    # Step 3: Per-cadence WF mining
    per_cad: Dict[str, Any] = {}
    for cadence in ["1h", "30m", "4h"]:
        cad_res = mine_cadence_wf(cadence, df_train_1d, folds)
        per_cad[cadence] = cad_res
        # Persist per-cadence per-fold detail (strip raw capture vectors to save space; we already have aggregates)
        out_path = OUT_DIR / f"pepe_train_wf_per_fold_{cadence}_2026_05_27.json"
        with open(out_path, "w") as f:
            json.dump({
                "_meta": {
                    "git_sha": get_git_sha(),
                    "canonical_seeds": CANONICAL_SEEDS,
                    "cadence": cadence,
                    "wf_config": {
                        "train_days": WF_TRAIN_DAYS, "test_days": WF_TEST_DAYS, "step_days": WF_STEP_DAYS,
                    },
                    "ts_generated": "2026-05-27",
                },
                "result": cad_res,
            }, f, indent=2, default=str)
        print(f"[{cadence}] wrote {out_path.name} ({out_path.stat().st_size/1024:.1f} KB)")

    # Step 4: Survivor aggregation per cadence
    summary: Dict[str, Any] = {
        "_meta": {
            "git_sha": get_git_sha(),
            "canonical_seeds": CANONICAL_SEEDS,
            "train_only": True,
            "train_meta": meta_train,
            "wf_config": {
                "train_days": WF_TRAIN_DAYS, "test_days": WF_TEST_DAYS, "step_days": WF_STEP_DAYS,
                "n_folds": len(folds),
            },
            "primary_rule": {
                "top_k": PRIMARY_TOPK,
                "min_folds": PRIMARY_MIN_FOLDS,
                "test_positive_frac": PRIMARY_TEST_POSITIVE_FRAC,
                "definition": "train-top-K in >=min_folds AND test_median>0 in >=test_pos_frac of appearances",
            },
            "fallback_rule": {
                "top_k": FALLBACK_TOPK,
                "min_folds": FALLBACK_MIN_FOLDS,
                "test_positive_frac": FALLBACK_TEST_POSITIVE_FRAC,
            },
            "elapsed_sec": None,
        },
        "folds": folds,
        "per_cadence": {},
    }

    for cadence, cad_res in per_cad.items():
        prim = aggregate_survivors(cad_res, PRIMARY_TOPK, PRIMARY_MIN_FOLDS, PRIMARY_TEST_POSITIVE_FRAC)
        prim_passes = [s for s in prim if s["passes_survivor_rule"]]
        if not prim_passes:
            fb = aggregate_survivors(cad_res, FALLBACK_TOPK, FALLBACK_MIN_FOLDS, FALLBACK_TEST_POSITIVE_FRAC)
            fb_passes = [s for s in fb if s["passes_survivor_rule"]]
            rule_used = "fallback"
            survivors = fb
            passes_count = len(fb_passes)
        else:
            rule_used = "primary"
            survivors = prim
            fb = None
            passes_count = len(prim_passes)

        summary["per_cadence"][cadence] = {
            "n_setups": cad_res["n_setups"],
            "n_train_movers_total": cad_res["n_train_movers_total"],
            "rule_used": rule_used,
            "n_passing_survivors": passes_count,
            "survivors_primary": prim[:50] if not prim_passes else prim_passes,  # top 50 of primary list OR all passing
            "survivors_fallback": (fb[:50] if (not prim_passes and fb is not None) else None),
            "fallback_was_evaluated": (not prim_passes),
            "honest_null": (not prim_passes) and (fb is None or len([s for s in fb if s["passes_survivor_rule"]]) == 0),
        }
        # Tag regime patterns
        for s in summary["per_cadence"][cadence]["survivors_primary"] or []:
            s["regime_pattern_hint"] = classify_regime_pattern(s)
        if summary["per_cadence"][cadence]["survivors_fallback"]:
            for s in summary["per_cadence"][cadence]["survivors_fallback"]:
                s["regime_pattern_hint"] = classify_regime_pattern(s)

        print(f"\n[{cadence}] rule_used={rule_used}, n_passing_survivors={passes_count}")
        for s in (summary["per_cadence"][cadence]["survivors_primary"] or [])[:10]:
            tag = "[PASS]" if s["passes_survivor_rule"] else "[----]"
            print(f"   {tag} {s['setup']:12s}  top_k={s['n_folds_top_k']:2d}  test_pos={s['test_positive_count']}/{s['n_folds_top_k']} ({s['test_positive_pct_of_appearances']:.0%})  test_med_avg={s['test_median_avg_across_folds']:+.4f}  train_med_avg={s['train_median_avg_across_folds']:+.4f}")

    elapsed = time.time() - t0
    summary["_meta"]["elapsed_sec"] = round(elapsed, 1)

    # Persist summary
    summary_path = OUT_DIR / "pepe_train_wf_summary_2026_05_27.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[OK] Summary -> {summary_path} ({summary_path.stat().st_size/1024:.1f} KB)")
    print(f"[OK] Elapsed: {elapsed:.1f}s")

    return summary


if __name__ == "__main__":
    main()
