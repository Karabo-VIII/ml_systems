"""
W1 task in MAXX-INST-2026-05-26-NIGHT (2026-05-27):
Unify selection rule across 4 cadences (15m, 30m, 1h, 4h) + permutation null tests
on per-cell setup winners + chimera-feature top findings.

Mandate context: W1 (15m+30m) used score = median_capture * sqrt(fire_rate); W2 (1h+4h)
used max median_capture | fire_rate >= 0.50. Numbers not comparable. Multi-comparisons
risk (~2,560 stat tests) acknowledged but never null-tested.

Unified selection rule:
  score = median_capture * sqrt(fire_rate) * (1.0 if win_rate >= 0.50 else 0.5)

Rationale:
  - median_capture: robust to outliers (vs mean)
  - sqrt(fire_rate): penalizes sparse setups without over-penalizing 0.5-1.0 fire rates
  - win_rate gate: only-positive-median is too loose; require >=50% of fires positive

Permutation null protocol:
  Setup permutation (per (cadence, cell)):
    - Within the cell, take the per-day capture vector for the WINNER setup
    - Shuffle row->capture mapping 1000 times
    - For each shuffle: re-rank setups; record the WINNER score from the shuffle
    - p_value = % of null scores >= observed_score
  Chimera permutation (per cell, per (feature, metric) with |corr| >= 0.2):
    - Within the cell, shuffle date->feature_value while keeping capture aligned
    - p_value = % of |null_corr| >= |observed_corr|
  Bonferroni: total N_tests = sum over cadences x cells of (n_setups + 10 features * 8 metrics)

TRAIN-ONLY. Past-only. POST-v8.3 harness style.
canonical_seeds = {bag_seed: 180, feat_seed: 1180, rng_seed: 8090}
"""
from __future__ import annotations

import json
import sys
import subprocess
from pathlib import Path
from collections import defaultdict
from typing import Optional, Dict, List, Any, Tuple

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]

CHIM_PATHS = {
    "15m": ROOT / "data/processed/chimera/15m/pepeusdt_v51_chimera_15m_20260522.parquet",
    "30m": ROOT / "data/processed/chimera/30m/pepeusdt_v51_chimera_30m_20260522.parquet",
    "1h":  ROOT / "data/processed/chimera/1h/pepeusdt_v51_chimera_1h_20260522.parquet",
    "4h":  ROOT / "data/processed/chimera/4h/pepeusdt_v51_chimera_4h_20260522.parquet",
}
CHIM_1D = ROOT / "data/processed/chimera/1d/pepeusdt_v51_chimera_1d_20260522.parquet"

SIG_PATH = ROOT / "runs/mover_day/pepe_train_mover_day_signatures_2026_05_27.json"
REGIME_JSON = {
    "15m": ROOT / "runs/mover_day/pepe_train_regime_conditioned_15m_2026_05_27.json",
    "30m": ROOT / "runs/mover_day/pepe_train_regime_conditioned_30m_2026_05_27.json",
    "1h":  ROOT / "runs/mover_day/pepe_train_regime_conditioned_1h_2026_05_27.json",
    "4h":  ROOT / "runs/mover_day/pepe_train_regime_conditioned_4h_2026_05_27.json",
}
CHIMERA_JSON = {
    cad: ROOT / f"runs/mover_day/pepe_train_regime_chimera_{cad}_2026_05_27.json"
    for cad in ("15m", "30m", "1h", "4h")
}
OUT_DIR = ROOT / "runs/mover_day"
DOSSIER_DIR = ROOT / "docs/dossiers"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DOSSIER_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_SEEDS = {"bag_seed": 180, "feat_seed": 1180, "rng_seed": 8090}
N_PERMS = 1000
MIN_CELL_N = 8

# Cadence grids must match the source mining scripts to ensure identical capture matrix
CADENCE_CFGS = {
    "15m": {
        "fast_grid": [12, 20, 28, 36, 48, 60],
        "slow_grid": [48, 60, 84, 120, 168, 240],
        "bars_per_hour": 4,
        "lookback_bars": 192,
    },
    "30m": {
        "fast_grid": [6, 10, 14, 18, 24, 30],
        "slow_grid": [24, 30, 42, 60, 84, 120],
        "bars_per_hour": 2,
        "lookback_bars": 144,
    },
    "1h": {
        "fast_grid": [5, 8, 9, 10, 13, 20, 30],
        "slow_grid": [21, 30, 50, 100, 200],
        "bars_per_hour": 1,
        "lookback_bars": 72,
    },
    "4h": {
        "fast_grid": [3, 5, 7, 9, 12, 15],
        "slow_grid": [9, 12, 18, 24, 36, 48],
        "bars_per_hour": 0.25,
        "lookback_bars": 18,
    },
}

INDICATOR_TYPES = ["SMA", "EMA", "WMA"]

CHIMERA_FEATURES = [
    "wh_whale_net_usd", "wh_whale_trade_count", "fund_rate_mean", "bs_basis_z30",
    "hbr_eta_buy", "bd_imbalance_l1", "te_btc_imb", "rv_rv_5m", "rv_bpv_5m", "premium_z90",
]
CHIM_METRICS_NUM = ["entry_val", "delta_12h", "zscore_12h", "up_frac", "down_frac"]
CHIM_METRICS_BOOL = ["monotonic_up", "monotonic_down", "spike"]


# --------- helpers ----------

def get_git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return "unknown"


def compute_ma(prices: np.ndarray, kind: str, window: int) -> np.ndarray:
    n = len(prices)
    out = np.full(n, np.nan)
    if kind == "SMA":
        if n >= window:
            csum = np.cumsum(prices)
            out[window - 1:] = (csum[window - 1:] - np.concatenate([[0], csum[:-window]])) / window
    elif kind == "EMA":
        alpha = 2.0 / (window + 1.0)
        if n >= window:
            seed = prices[:window].mean()
            out[window - 1] = seed
            for i in range(window, n):
                out[i] = alpha * prices[i] + (1 - alpha) * out[i - 1]
    elif kind == "WMA":
        if n >= window:
            w = np.arange(1, window + 1, dtype=float)
            w_sum = w.sum()
            for i in range(window - 1, n):
                out[i] = (prices[i - window + 1:i + 1] * w).sum() / w_sum
    return out


def first_cross_up_in_window(fast_ma, slow_ma, start, end):
    s = max(start, 1)
    if s >= end:
        return -1
    prev_f = fast_ma[s - 1:end - 1]
    prev_s = slow_ma[s - 1:end - 1]
    cur_f = fast_ma[s:end]
    cur_s = slow_ma[s:end]
    mask = (~np.isnan(prev_f)) & (~np.isnan(prev_s)) & (~np.isnan(cur_f)) & (~np.isnan(cur_s)) \
        & (prev_f <= prev_s) & (cur_f > cur_s)
    if not mask.any():
        return -1
    return s + int(np.argmax(mask))


def macro_regime(btc_trend: str, risk_state: str) -> str:
    if btc_trend == "bull" and risk_state == "risk_on":
        return "BULL_RISK_ON"
    if btc_trend == "chop" and risk_state == "risk_on":
        return "CHOP_RISK_ON"
    if btc_trend == "bear" or risk_state == "risk_off":
        return "BEAR_OR_RISK_OFF"
    if btc_trend == "bull" and risk_state == "risk_off":
        return "BEAR_OR_RISK_OFF"
    return "OTHER"


def pepe_micro_regime(p: str) -> str:
    if p == "WARMUP":
        return "WARMUP"
    if p.startswith("trending_up"):
        return "pepe_trending_up"
    if p.startswith("trending_down"):
        return "pepe_trending_down"
    if p.startswith("chop"):
        return "pepe_chop"
    return "OTHER"


# --------- unified selection rule ----------

def unified_score(median_capture: float, fire_rate: float, win_rate: float) -> float:
    """score = median_capture * sqrt(fire_rate) * (1.0 if win_rate>=0.50 else 0.5).

    If median_capture is non-finite OR fire_rate==0, returns -inf so it cannot win.
    Treats negative median_capture as acceptable (could be informative) but rare.
    """
    if not np.isfinite(median_capture) or fire_rate <= 0.0:
        return float("-inf")
    win_factor = 1.0 if win_rate >= 0.50 else 0.5
    return median_capture * (fire_rate ** 0.5) * win_factor


# --------- per-cadence capture matrix ----------

def build_capture_matrix(cadence: str, mover_dates: List[str], mover_ts_by_date: Dict[str, int]) -> Dict[str, Any]:
    cfg = CADENCE_CFGS[cadence]
    path = CHIM_PATHS[cadence]
    df = pl.read_parquet(path).sort("timestamp")
    n_total = len(df)

    # TRAIN slice anchor: same as source scripts. Use the 1d TRAIN end ts.
    # We bound by max mover ts + ~7d to avoid scanning unneeded tail.
    last_mover_ts = max(mover_ts_by_date[d] for d in mover_dates if d in mover_ts_by_date)
    bar_ms_lookup = {"15m": 15 * 60_000, "30m": 30 * 60_000, "1h": 60 * 60_000, "4h": 4 * 60 * 60_000}
    bar_ms = bar_ms_lookup[cadence]
    cap_ts = last_mover_ts + 14 * 24 * 3600 * 1000  # 14d tail for any exit work (we won't use here but harmless)
    df_train = df.filter(pl.col("timestamp") <= cap_ts)
    ts = df_train["timestamp"].to_numpy()
    closes = df_train["close"].to_numpy()
    opens = df_train["open"].to_numpy()
    print(f"[{cadence}] train bars used: {len(df_train)} of {n_total}")

    unique_windows = sorted(set(cfg["fast_grid"] + cfg["slow_grid"]))
    ma_cache = {}
    for kind in INDICATOR_TYPES:
        for w in unique_windows:
            ma_cache[(kind, w)] = compute_ma(closes, kind, w)

    setups = []
    for kind in INDICATOR_TYPES:
        for fast in cfg["fast_grid"]:
            for slow in cfg["slow_grid"]:
                if fast >= slow:
                    continue
                setups.append((kind, fast, slow))
    n_setups = len(setups)

    n_movers = len(mover_dates)
    capture = np.full((n_movers, n_setups), np.nan)
    entry_idx_mat = np.full((n_movers, n_setups), -1, dtype=np.int64)

    for mi, d in enumerate(mover_dates):
        day_ts = mover_ts_by_date.get(d)
        if day_ts is None:
            continue
        day_close_idx = int(np.searchsorted(ts, day_ts, side="right") - 1)
        if day_close_idx < 0:
            continue
        window_start = max(0, day_close_idx - (cfg["lookback_bars"] - 1))
        signal_search_start = window_start + 1
        signal_search_end = day_close_idx + 1
        daily_close_price = closes[day_close_idx]

        for si, (kind, fast, slow) in enumerate(setups):
            fma = ma_cache[(kind, fast)]
            sma = ma_cache[(kind, slow)]
            cross_idx = first_cross_up_in_window(fma, sma, signal_search_start, signal_search_end)
            if cross_idx == -1:
                continue
            entry_i = cross_idx + 1
            if entry_i > day_close_idx:
                continue
            ent_p = opens[entry_i]
            if ent_p <= 0:
                continue
            cap = float(daily_close_price / ent_p - 1.0)
            capture[mi, si] = cap
            entry_idx_mat[mi, si] = entry_i

    return {
        "setups": setups,
        "capture": capture,
        "entry_idx": entry_idx_mat,
        "closes": closes,
        "opens": opens,
        "ts": ts,
        "n_setups": n_setups,
        "n_movers": n_movers,
    }


# --------- per-cell stats + winner under unified rule ----------

def setup_label(setup_tuple) -> str:
    kind, fast, slow = setup_tuple
    return f"{kind} {fast}/{slow}"


def per_cell_stats(cap_matrix: np.ndarray, mover_indices: np.ndarray, setups: list) -> List[Dict[str, Any]]:
    sub = cap_matrix[mover_indices, :]
    n_cell = len(mover_indices)
    out = []
    for si, st in enumerate(setups):
        col = sub[:, si]
        fired = np.isfinite(col)
        n_fires = int(fired.sum())
        if n_fires == 0:
            continue
        arr = col[fired]
        med = float(np.median(arr))
        mean = float(np.mean(arr))
        fr = float(n_fires / n_cell)
        wr = float(np.sum(arr > 0) / n_fires)
        out.append({
            "setup": setup_label(st),
            "n_cell": n_cell,
            "n_fires": n_fires,
            "fire_rate": fr,
            "median_capture": med,
            "mean_capture": mean,
            "win_rate": wr,
            "score": unified_score(med, fr, wr),
            "raw_capture_vector": arr.tolist(),
        })
    out.sort(key=lambda r: -r["score"])
    return out


def find_winner_unified(cap_matrix: np.ndarray, mover_indices: np.ndarray, setups: list) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    stats = per_cell_stats(cap_matrix, mover_indices, setups)
    if not stats:
        return None, []
    winner = stats[0]
    return winner, stats[:5]


# --------- permutation null on setup winner ----------

def _best_score_for_indices(cap_matrix: np.ndarray, idxs: np.ndarray, setups: list):
    sub = cap_matrix[idxs, :]
    n_cell, n_setups = sub.shape
    best_score = float("-inf")
    best_med = None
    best_si = -1
    for si in range(n_setups):
        col = sub[:, si]
        fired = np.isfinite(col)
        n_fires = int(fired.sum())
        if n_fires == 0:
            continue
        arr = col[fired]
        med = float(np.median(arr))
        fr = float(n_fires / n_cell)
        wr = float(np.sum(arr > 0) / n_fires)
        sc = unified_score(med, fr, wr)
        if sc > best_score:
            best_score = sc
            best_med = med
            best_si = si
    return best_score, best_med, best_si


def permutation_null_setup(cap_matrix: np.ndarray, mover_indices: np.ndarray,
                            all_mover_indices: np.ndarray, setups: list,
                            n_perms: int, rng: np.random.Generator) -> Dict[str, Any]:
    """Cell-assignment null: under H_0 the score achieved by the BEST setup in THIS cell is no
    larger than the score achieved by the BEST setup in a RANDOMLY-DRAWN subset of mover days
    of the same size from the broader cohort.

    Protocol:
      - Observed: BEST score among `setups` evaluated on `mover_indices` (size n_cell).
      - Null:  draw 1000 random subsamples of size n_cell from `all_mover_indices` (WITHOUT
        replacement; uniform). For each, compute the BEST score.
      - p_value = fraction of null best-scores >= observed.

    Why: shuffling cell labels across all 186 movers (keeping cell sizes) is equivalent to
    sampling-without-replacement subsets at each cell. We just sample the cell that we are
    testing -- so it's exact at the marginal level.

    NOTE: This is a non-trivial null because (a) capture vectors per setup VARY across days
    in the broader cohort, and (b) the n_cell sample drives the median and the within-cell
    fire-rate. Under H_0 of "no cell-specific signal", random subsamples should achieve
    comparable best-setup-scores.
    """
    n_cell = len(mover_indices)
    obs_score, obs_med, obs_si = _best_score_for_indices(cap_matrix, mover_indices, setups)
    obs_setup = setup_label(setups[obs_si]) if obs_si >= 0 else "n/a"
    if obs_si < 0 or not np.isfinite(obs_score):
        return None
    # Per-fire stats for observed
    col = cap_matrix[mover_indices, obs_si]
    obs_arr = col[np.isfinite(col)]
    obs_fire_rate = float(len(obs_arr) / n_cell)
    obs_win_rate = float(np.sum(obs_arr > 0) / len(obs_arr)) if len(obs_arr) else 0.0

    null_scores = np.zeros(n_perms)
    null_meds = np.zeros(n_perms)
    pool = np.asarray(all_mover_indices, dtype=np.int64)
    for p in range(n_perms):
        if n_cell <= len(pool):
            sample_idxs = rng.choice(pool, size=n_cell, replace=False)
        else:
            sample_idxs = rng.choice(pool, size=n_cell, replace=True)
        sc, med, _si = _best_score_for_indices(cap_matrix, sample_idxs, setups)
        null_scores[p] = sc if np.isfinite(sc) else 0.0
        null_meds[p] = med if med is not None else 0.0
    p_score = float(np.mean(null_scores >= obs_score))
    return {
        "null_protocol": "cell_assignment_subsample_without_replacement",
        "observed_winner": obs_setup,
        "observed_score": float(obs_score),
        "observed_median_capture": float(obs_med),
        "observed_fire_rate": float(obs_fire_rate),
        "observed_win_rate": float(obs_win_rate),
        "n_cell": int(n_cell),
        "n_pool": int(len(pool)),
        "n_perms": int(n_perms),
        "null_score_median": float(np.median(null_scores)),
        "null_score_p05": float(np.percentile(null_scores, 5)),
        "null_score_p95": float(np.percentile(null_scores, 95)),
        "null_score_max": float(np.max(null_scores)),
        "p_value_raw": p_score,
    }


# --------- per-cell chimera trajectory extraction ----------

def extract_chim_at_entry(arr_feature: np.ndarray, entry_idx: int, traj_bars: int) -> Dict[str, Any]:
    n = len(arr_feature)
    e_val = float(arr_feature[entry_idx]) if 0 <= entry_idx < n and np.isfinite(arr_feature[entry_idx]) else None
    end = min(entry_idx + traj_bars, n)
    if entry_idx + traj_bars >= n:
        return {
            "entry_val": e_val,
            "delta_12h": None,
            "zscore_12h": None,
            "up_frac": None,
            "down_frac": None,
            "monotonic_up": None,
            "monotonic_down": None,
            "spike": None,
        }
    seg = arr_feature[entry_idx:end]
    valid = seg[np.isfinite(seg)]
    if len(valid) < 2:
        return {
            "entry_val": e_val,
            "delta_12h": None,
            "zscore_12h": None,
            "up_frac": None,
            "down_frac": None,
            "monotonic_up": None,
            "monotonic_down": None,
            "spike": None,
        }
    delta = float(valid[-1] - valid[0])
    mu, sd = float(np.mean(valid)), float(np.std(valid))
    zsc = float((valid[-1] - mu) / sd) if sd > 0 else None
    diffs = np.diff(valid)
    up_frac = float(np.mean(diffs > 0))
    down_frac = float(np.mean(diffs < 0))
    mono_up = bool(np.all(diffs >= 0))
    mono_down = bool(np.all(diffs <= 0))
    spike = bool((np.abs(valid - mu).max() > (2.0 * sd)) if sd > 0 else False)
    return {
        "entry_val": e_val,
        "delta_12h": delta,
        "zscore_12h": zsc,
        "up_frac": up_frac,
        "down_frac": down_frac,
        "monotonic_up": mono_up,
        "monotonic_down": mono_down,
        "spike": spike,
    }


def safe_corr(x: List[float], y: List[float]) -> Optional[float]:
    xa, ya = [], []
    for a, b in zip(x, y):
        if a is None or b is None:
            continue
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        xa.append(float(a))
        ya.append(float(b))
    if len(xa) < 5:
        return None
    xa = np.asarray(xa)
    ya = np.asarray(ya)
    if np.std(xa) == 0 or np.std(ya) == 0:
        return None
    return float(np.corrcoef(xa, ya)[0, 1])


def permutation_null_corr(x: List[float], y: List[float], n_perms: int, rng: np.random.Generator) -> Dict[str, Any]:
    pairs = [(a, b) for a, b in zip(x, y)
             if (a is not None) and (b is not None)
             and np.isfinite(a) and np.isfinite(b)]
    if len(pairs) < 5:
        return {"observed": None, "n_paired": len(pairs), "p_value_raw": None,
                "null_corr_p05": None, "null_corr_p95": None, "null_corr_median": None}
    xa = np.array([p[0] for p in pairs])
    ya = np.array([p[1] for p in pairs])
    if np.std(xa) == 0 or np.std(ya) == 0:
        return {"observed": None, "n_paired": len(pairs), "p_value_raw": None,
                "null_corr_p05": None, "null_corr_p95": None, "null_corr_median": None}
    observed = float(np.corrcoef(xa, ya)[0, 1])
    null = np.empty(n_perms)
    for i in range(n_perms):
        perm = rng.permutation(len(xa))
        c = float(np.corrcoef(xa[perm], ya)[0, 1])
        null[i] = c if np.isfinite(c) else 0.0
    p = float(np.mean(np.abs(null) >= abs(observed)))
    return {
        "observed": observed,
        "n_paired": len(pairs),
        "n_perms": int(n_perms),
        "null_corr_median": float(np.median(null)),
        "null_corr_p05": float(np.percentile(null, 5)),
        "null_corr_p95": float(np.percentile(null, 95)),
        "p_value_raw": p,
    }


# --------- mainflow ----------

def load_movers_with_ts() -> Dict[str, int]:
    df = pl.read_parquet(CHIM_1D).sort("timestamp")
    n = len(df)
    df = df.with_columns(pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("dt"))
    df_train = df[: int(n * 0.50)]
    closes = df_train["close"].to_numpy()
    daily_ret = np.zeros(len(closes))
    daily_ret[1:] = closes[1:] / closes[:-1] - 1.0
    df_train = df_train.with_columns(pl.Series("daily_return", daily_ret))
    movers = df_train.filter(pl.col("daily_return") >= 0.02)
    out = {}
    for row in movers.iter_rows(named=True):
        out[str(row["dt"].date())] = int(row["timestamp"])
    return out


def load_signatures() -> List[Dict[str, Any]]:
    with open(SIG_PATH) as f:
        d = json.load(f)
    return d["signatures"]


def build_per_cadence_results(rng: np.random.Generator):
    print("\n" + "=" * 80)
    print("LOADING TRAIN MOVER DAYS AND SIGNATURES")
    print("=" * 80)
    mover_ts = load_movers_with_ts()
    sigs = load_signatures()
    print(f"  movers: {len(mover_ts)}  signatures: {len(sigs)}")

    # Build per-mover regime tags (ordered)
    mover_dates = sorted(mover_ts.keys())
    sig_by_date = {s["date"]: s for s in sigs}
    cell_for_date = {}
    for d in mover_dates:
        s = sig_by_date.get(d)
        if s is None:
            continue
        macro = macro_regime(s["btc_trend"], s["crypto_risk_state"])
        micro = pepe_micro_regime(s["pepe_self_cell"])
        cell_for_date[d] = f"{macro} x {micro}"

    cells_uniq = sorted(set(cell_for_date.values()))
    print("  unique regime cells:", cells_uniq)

    # --- Per-cadence loop ---
    per_cadence = {}
    for cad in ["15m", "30m", "1h", "4h"]:
        print("\n" + "=" * 80)
        print(f"CADENCE {cad}")
        print("=" * 80)
        cap_pkg = build_capture_matrix(cad, mover_dates, mover_ts)
        setups = cap_pkg["setups"]
        capture = cap_pkg["capture"]
        entry_idx = cap_pkg["entry_idx"]
        print(f"  n_setups={len(setups)}  capture_matrix_shape={capture.shape}")

        # Per-cell winners + permutation null
        per_cell_winners = {}
        per_cell_top5 = {}
        per_cell_perm_null = {}
        for cell in cells_uniq:
            idxs = np.array([i for i, d in enumerate(mover_dates) if cell_for_date.get(d) == cell])
            n_cell = len(idxs)
            if n_cell < MIN_CELL_N:
                per_cell_winners[cell] = {"LOW_N": True, "n": int(n_cell)}
                continue
            winner, top5 = find_winner_unified(capture, idxs, setups)
            if winner is None:
                per_cell_winners[cell] = {"NO_WINNER": True, "n": int(n_cell)}
                continue
            # Strip raw capture vec from saved record but keep elsewhere
            wr = dict(winner)
            wr.pop("raw_capture_vector", None)
            per_cell_winners[cell] = wr
            per_cell_top5[cell] = [{k: v for k, v in r.items() if k != "raw_capture_vector"} for r in top5]
            print(f"  [{cad}] {cell:50s} n={n_cell:3d} winner={winner['setup']:14s} "
                  f"score={winner['score']:+.5f} mc={winner['median_capture']:+.4f} "
                  f"fr={winner['fire_rate']:.3f} wr={winner['win_rate']:.3f}")
            # Permutation null (cell-assignment subsample from all_mover_indices)
            print(f"      running permutation null (N={N_PERMS}) ...")
            all_idxs = np.arange(len(mover_dates))
            perm = permutation_null_setup(capture, idxs, all_idxs, setups, N_PERMS, rng)
            per_cell_perm_null[cell] = perm
            print(f"      observed_score={perm['observed_score']:+.5f}  null_p95={perm['null_score_p95']:+.5f}  p_raw={perm['p_value_raw']:.4f}")
        per_cadence[cad] = {
            "n_setups": len(setups),
            "n_movers_total": len(mover_dates),
            "per_cell_winners": per_cell_winners,
            "per_cell_top5": per_cell_top5,
            "per_cell_perm_null": per_cell_perm_null,
            "capture_matrix_shape": list(capture.shape),
        }
        # Stash arrays for chimera step
        per_cadence[cad]["_cap_pkg"] = cap_pkg
        per_cadence[cad]["_mover_dates"] = mover_dates
        per_cadence[cad]["_cell_for_date"] = cell_for_date
        per_cadence[cad]["_cells_uniq"] = cells_uniq

    return per_cadence


def compare_old_vs_new(per_cadence: dict) -> List[Dict[str, Any]]:
    """Load each cadence's old mining JSON, extract old winner per cell, compare to new winner."""
    comparison_rows = []
    for cad in ["15m", "30m", "1h", "4h"]:
        old = json.load(open(REGIME_JSON[cad]))
        new = per_cadence[cad]["per_cell_winners"]
        if cad in ("15m", "30m"):
            old_cw_dict = old.get("cell_results", {})
            for cell, w in new.items():
                if w.get("LOW_N") or w.get("NO_WINNER"):
                    continue
                old_cell_blob = old_cw_dict.get(cell, {})
                old_w = old_cell_blob.get("winner") if isinstance(old_cell_blob, dict) else None
                if old_w is None:
                    old_setup = "n/a"
                    old_score = None
                else:
                    old_setup = old_w["setup"]
                    old_score = unified_score(old_w["median_capture"], old_w["fire_rate"], old_w["win_rate"])
                comparison_rows.append({
                    "cadence": cad,
                    "regime_cell": cell,
                    "n": w["n_cell"],
                    "old_winner": old_setup,
                    "old_score_under_unified": old_score,
                    "new_winner": w["setup"],
                    "new_score": w["score"],
                    "new_median_capture": w["median_capture"],
                    "new_fire_rate": w["fire_rate"],
                    "new_win_rate": w["win_rate"],
                    "changed": (old_setup != w["setup"]),
                })
        else:  # 1h, 4h
            old_cw = old.get("results", {}).get("cell_winners", {})
            # The 1h/4h cells use | separator; new keys use ' x ' separator. Convert.
            for cell, w in new.items():
                if w.get("LOW_N") or w.get("NO_WINNER"):
                    continue
                # Try both separators in old data
                k_pipe = cell.replace(" x ", "|")
                old_blob = old_cw.get(k_pipe) or old_cw.get(cell)
                if old_blob is None:
                    old_setup = "n/a"
                    old_score = None
                else:
                    old_setup = f"{old_blob['kind']} {old_blob['fast']}/{old_blob['slow']}"
                    old_score = unified_score(old_blob["median_capture"], old_blob["fire_rate"], old_blob["win_rate"])
                comparison_rows.append({
                    "cadence": cad,
                    "regime_cell": cell,
                    "n": w["n_cell"],
                    "old_winner": old_setup,
                    "old_score_under_unified": old_score,
                    "new_winner": w["setup"],
                    "new_score": w["score"],
                    "new_median_capture": w["median_capture"],
                    "new_fire_rate": w["fire_rate"],
                    "new_win_rate": w["win_rate"],
                    "changed": (old_setup != w["setup"]),
                })
    return comparison_rows


def run_chimera_permutation_nulls(per_cadence: dict, rng: np.random.Generator) -> Dict[str, Any]:
    """For each cadence/cell, recompute the winner-setup per-mover captures + chimera trajectory
    metrics, then permutation-null every (feature, metric) with |corr|>=0.2.

    Uses the chimera JSON's cell_records_meta where possible; otherwise recomputes via
    parquet+entry indices from the capture pkg.
    """
    print("\n" + "=" * 80)
    print("CHIMERA FEATURE PERMUTATION NULLS")
    print("=" * 80)
    out = {}
    for cad in ["15m", "30m", "1h", "4h"]:
        cad_out = {}
        # Recompute under unified-winner using parquet
        cfg = CADENCE_CFGS[cad]
        traj_bars = int(round(12 * cfg["bars_per_hour"]))
        chim_path = CHIM_PATHS[cad]
        df = pl.read_parquet(chim_path).sort("timestamp")
        # cap to a sane bound = last mover + 14d (matches build_capture_matrix)
        # but we just use full TRAIN bars since arr indexing is consistent with cap_pkg
        ts = df["timestamp"].to_numpy()
        # We need ts to match cap_pkg["ts"] -- cap_pkg already trimmed; we recompute on full df
        # then index using entry_idx which is local to the trimmed df. To avoid index drift,
        # re-trim here the same way.
        last_mover_ts = max(per_cadence[cad]["_cap_pkg"]["ts"])
        df_train = df.filter(pl.col("timestamp") <= last_mover_ts)
        feat_arrs = {}
        for f in CHIMERA_FEATURES:
            if f in df_train.columns:
                feat_arrs[f] = df_train[f].to_numpy()
            else:
                feat_arrs[f] = None
        cap_pkg = per_cadence[cad]["_cap_pkg"]
        setups = cap_pkg["setups"]
        entry_idx_mat = cap_pkg["entry_idx"]
        capture_mat = cap_pkg["capture"]
        mover_dates = per_cadence[cad]["_mover_dates"]
        cell_for_date = per_cadence[cad]["_cell_for_date"]

        print(f"\n[{cad}] cells:")
        for cell, w in per_cadence[cad]["per_cell_winners"].items():
            if w.get("LOW_N") or w.get("NO_WINNER"):
                cad_out[cell] = {"SKIPPED": "low_n_or_no_winner"}
                continue
            # Locate the new-winner setup index
            winner_label = w["setup"]
            kind, fastslow = winner_label.split(" ")
            fast, slow = fastslow.split("/")
            fast, slow = int(fast), int(slow)
            try:
                si = next(i for i, st in enumerate(setups) if st == (kind, fast, slow))
            except StopIteration:
                cad_out[cell] = {"ERROR": f"winner {winner_label} not in setup grid"}
                continue
            idxs = np.array([i for i, d in enumerate(mover_dates) if cell_for_date.get(d) == cell])
            x_capt = []
            x_dates = []
            x_chim = []
            for mi in idxs:
                ei = int(entry_idx_mat[mi, si])
                cap = capture_mat[mi, si]
                if ei < 0 or not np.isfinite(cap):
                    continue
                cell_chim = {}
                for fname, farr in feat_arrs.items():
                    if farr is None:
                        cell_chim[fname] = None
                    else:
                        cell_chim[fname] = extract_chim_at_entry(farr, ei, traj_bars)
                x_capt.append(float(cap))
                x_dates.append(mover_dates[mi])
                x_chim.append(cell_chim)
            n_fires = len(x_capt)
            # For each feature x metric, run permutation null
            findings = []
            for fname in CHIMERA_FEATURES:
                # Collect feature value vector per metric
                for met in CHIM_METRICS_NUM + CHIM_METRICS_BOOL:
                    fvals = []
                    for ch in x_chim:
                        if ch.get(fname) is None:
                            fvals.append(None)
                        else:
                            v = ch[fname].get(met)
                            if met in CHIM_METRICS_BOOL:
                                if v is None:
                                    fvals.append(None)
                                else:
                                    fvals.append(1.0 if v else 0.0)
                            else:
                                fvals.append(v)
                    obs_corr = safe_corr(fvals, x_capt)
                    if obs_corr is None:
                        continue
                    findings.append({
                        "feature": fname,
                        "metric": met,
                        "observed_corr": obs_corr,
                        "n_paired": int(sum(1 for v in fvals if v is not None and np.isfinite(v))),
                    })
            findings.sort(key=lambda r: -abs(r["observed_corr"]))
            # Permutation null on TOP findings (|corr| >= 0.2) -- bounded compute
            top_findings = [r for r in findings if abs(r["observed_corr"]) >= 0.20]
            # Limit per-cell to top 8 to control compute
            top_findings = top_findings[:8]
            for f in top_findings:
                # Recollect fvals for that (feature, metric)
                fname = f["feature"]; met = f["metric"]
                fvals = []
                for ch in x_chim:
                    if ch.get(fname) is None:
                        fvals.append(None)
                    else:
                        v = ch[fname].get(met)
                        if met in CHIM_METRICS_BOOL:
                            if v is None: fvals.append(None)
                            else: fvals.append(1.0 if v else 0.0)
                        else:
                            fvals.append(v)
                pnull = permutation_null_corr(fvals, x_capt, N_PERMS, rng)
                f.update({
                    "n_perms": pnull["n_perms"],
                    "null_corr_median": pnull["null_corr_median"],
                    "null_corr_p05": pnull["null_corr_p05"],
                    "null_corr_p95": pnull["null_corr_p95"],
                    "p_value_raw": pnull["p_value_raw"],
                })
            cad_out[cell] = {
                "winner_setup": winner_label,
                "n_fires_under_new_winner": n_fires,
                "all_findings": findings,
                "top_findings_with_perm": top_findings,
            }
            print(f"  {cell:50s} fires_under_winner={n_fires}  top_findings_tested={len(top_findings)}")
        out[cad] = cad_out
    return out


def aggregate_bonferroni_n(per_cadence: dict, chimera_out: dict) -> int:
    """Total test count across:
       (a) setup permutation tests (1 per (cadence,cell) with winner) -- this is the #
           of WINNER hypotheses tested.
       (b) chimera feature-metric correlation tests run across ALL features/metrics per
           cell (NOT just top 8) -- this is the # of independent corr tests.
    """
    n = 0
    n_setup = 0
    for cad, blob in per_cadence.items():
        for cell, w in blob["per_cell_winners"].items():
            if w.get("LOW_N") or w.get("NO_WINNER"):
                continue
            n_setup += 1
    n_chim = 0
    for cad, blob in chimera_out.items():
        for cell, r in blob.items():
            if r.get("SKIPPED") or r.get("ERROR"):
                continue
            n_chim += len(r.get("all_findings", []))
    return n_setup + n_chim, n_setup, n_chim


def cross_cadence_robust(per_cadence: dict, chimera_out: dict) -> List[Dict[str, Any]]:
    """A finding is cross-cadence robust if same cell + same (feature, metric) appears at
    >=2 cadences with same-sign and each at least p_raw < 0.10."""
    by_cell_feat = defaultdict(list)  # (cell, feature, metric) -> list of (cadence, observed_corr, p_raw)
    for cad, blob in chimera_out.items():
        for cell, r in blob.items():
            if r.get("SKIPPED") or r.get("ERROR"):
                continue
            for f in r.get("top_findings_with_perm", []):
                key = (cell, f["feature"], f["metric"])
                by_cell_feat[key].append({
                    "cadence": cad,
                    "corr": f["observed_corr"],
                    "p_raw": f.get("p_value_raw"),
                    "n_paired": f.get("n_paired"),
                })
    robust = []
    for key, entries in by_cell_feat.items():
        if len(entries) < 2:
            continue
        signs = set(np.sign(e["corr"]) for e in entries)
        if len(signs) > 1:
            continue
        passing = [e for e in entries if e["p_raw"] is not None and e["p_raw"] < 0.10]
        if len(passing) < 2:
            continue
        robust.append({
            "regime_cell": key[0], "feature": key[1], "metric": key[2],
            "cadences": [e["cadence"] for e in entries],
            "corrs": [e["corr"] for e in entries],
            "p_raws": [e["p_raw"] for e in entries],
            "n_passing_p10": len(passing),
            "all_p_raw_lt_005": all((e["p_raw"] is not None and e["p_raw"] < 0.05) for e in entries),
        })
    robust.sort(key=lambda r: (-r["n_passing_p10"], -max(abs(c) for c in r["corrs"])))

    # Also: same-cell setup-family robustness (SMA/EMA/WMA families at similar fast/slow ratios)
    setups_by_cell = defaultdict(list)
    for cad, blob in per_cadence.items():
        for cell, w in blob["per_cell_winners"].items():
            if w.get("LOW_N") or w.get("NO_WINNER"):
                continue
            perm = blob["per_cell_perm_null"].get(cell, {})
            setups_by_cell[cell].append({
                "cadence": cad,
                "winner": w["setup"],
                "score": w["score"],
                "p_raw": perm.get("p_value_raw") if perm else None,
                "median_capture": w["median_capture"],
            })
    setup_robust = []
    for cell, lst in setups_by_cell.items():
        if len(lst) < 2:
            continue
        # group by indicator family (SMA/EMA/WMA)
        fams = defaultdict(list)
        for e in lst:
            kind = e["winner"].split(" ")[0]
            fams[kind].append(e)
        for kind, els in fams.items():
            if len(els) < 2:
                continue
            passing = [e for e in els if e["p_raw"] is not None and e["p_raw"] < 0.10]
            if len(passing) < 2:
                continue
            setup_robust.append({
                "regime_cell": cell,
                "indicator_family": kind,
                "cadences": [e["cadence"] for e in els],
                "winners": [e["winner"] for e in els],
                "median_captures": [e["median_capture"] for e in els],
                "p_raws": [e["p_raw"] for e in els],
                "n_passing_p10": len(passing),
            })
    return robust, setup_robust


# --------- main ----------

def main():
    rng = np.random.default_rng(CANONICAL_SEEDS["rng_seed"])
    git_sha = get_git_sha()
    print(f"git_sha={git_sha}")
    print(f"canonical_seeds={CANONICAL_SEEDS}")
    print(f"N_PERMS={N_PERMS}  MIN_CELL_N={MIN_CELL_N}")

    per_cadence = build_per_cadence_results(rng)
    comparison = compare_old_vs_new(per_cadence)
    chimera_out = run_chimera_permutation_nulls(per_cadence, rng)

    n_tests_total, n_setup, n_chim = aggregate_bonferroni_n(per_cadence, chimera_out)
    print(f"\nN_TESTS_TOTAL={n_tests_total} (setup tests={n_setup}, chimera corr tests={n_chim})")
    p_bonf = 0.05 / n_tests_total if n_tests_total > 0 else None
    print(f"Bonferroni alpha (FWER=0.05): {p_bonf:.3e}")

    # Apply Bonferroni verdict to setup tests + top chimera findings
    setup_verdicts = []
    for cad, blob in per_cadence.items():
        for cell, perm in blob["per_cell_perm_null"].items():
            if perm is None:
                continue
            p_raw = perm["p_value_raw"]
            p_bonf_val = min(1.0, p_raw * n_tests_total)
            if p_bonf_val < 0.05:
                verdict = "ROBUST"
            elif p_raw < 0.05:
                verdict = "PASSES_RAW"
            else:
                verdict = "NULL_NOT_REJECTED"
            setup_verdicts.append({
                "cadence": cad, "regime_cell": cell,
                "observed_winner": perm["observed_winner"],
                "observed_score": perm["observed_score"],
                "p_value_raw": p_raw,
                "p_bonferroni": p_bonf_val,
                "verdict": verdict,
            })

    chimera_verdicts = []
    for cad, blob in chimera_out.items():
        for cell, rec in blob.items():
            if rec.get("SKIPPED") or rec.get("ERROR"):
                continue
            for f in rec.get("top_findings_with_perm", []):
                p_raw = f.get("p_value_raw")
                if p_raw is None:
                    continue
                p_bonf_val = min(1.0, p_raw * n_tests_total)
                if p_bonf_val < 0.05:
                    v = "ROBUST"
                elif p_raw < 0.05:
                    v = "PASSES_RAW"
                else:
                    v = "NULL_NOT_REJECTED"
                chimera_verdicts.append({
                    "cadence": cad, "regime_cell": cell,
                    "feature": f["feature"], "metric": f["metric"],
                    "observed_corr": f["observed_corr"],
                    "n_paired": f.get("n_paired"),
                    "p_value_raw": p_raw,
                    "p_bonferroni": p_bonf_val,
                    "verdict": v,
                })

    robust_cross, setup_cross = cross_cadence_robust(per_cadence, chimera_out)

    # Strip private fields before save
    for cad in per_cadence:
        for k in ["_cap_pkg", "_mover_dates", "_cell_for_date", "_cells_uniq"]:
            per_cadence[cad].pop(k, None)

    # ---- Save JSON: unified selection ----
    out_unified = {
        "_meta": {
            "task": "W1 unified selection across 4 cadences",
            "instance": "MAXX-INST-2026-05-26-NIGHT",
            "ts_generated": __import__("datetime").datetime.utcnow().isoformat() + "Z",
            "git_sha": git_sha,
            "canonical_seeds": CANONICAL_SEEDS,
            "train_only": True,
            "unified_score_rule": "median_capture * sqrt(fire_rate) * (1.0 if win_rate>=0.50 else 0.5)",
            "min_cell_n": MIN_CELL_N,
        },
        "per_cadence_winners": {cad: per_cadence[cad]["per_cell_winners"] for cad in per_cadence},
        "per_cadence_top5": {cad: per_cadence[cad]["per_cell_top5"] for cad in per_cadence},
        "old_vs_new_comparison": comparison,
    }
    p_unified = OUT_DIR / "pepe_train_unified_selection_2026_05_27.json"
    p_unified.write_text(json.dumps(out_unified, indent=2, default=str))
    print(f"\nWROTE {p_unified}")

    # ---- Save JSON: permutation nulls ----
    out_perm = {
        "_meta": {
            "task": "W1 permutation null tests on setups + chimera correlations",
            "instance": "MAXX-INST-2026-05-26-NIGHT",
            "ts_generated": __import__("datetime").datetime.utcnow().isoformat() + "Z",
            "git_sha": git_sha,
            "canonical_seeds": CANONICAL_SEEDS,
            "train_only": True,
            "n_perms": N_PERMS,
            "n_tests_total": n_tests_total,
            "n_setup_tests": n_setup,
            "n_chimera_corr_tests": n_chim,
            "bonferroni_alpha_FWER_005": p_bonf,
            "permutation_protocol": {
                "setup_null": "CELL-ASSIGNMENT NULL: under H_0 the BEST-setup score for THIS cell is no larger than the BEST-setup score for a randomly-drawn subset of mover days of the same size from the full 186-mover-day pool. Sampling without replacement; uniform; same selection rule applied to the random subset; p-value = fraction of null best-scores >= observed.",
                "chimera_corr_null": "Within the cell, shuffle the date->feature_value pairing while keeping capture vector aligned; |corr| 2-sided p_value = fraction of |null_corr| >= |observed_corr|.",
            },
        },
        "setup_permutation_results_per_cell": {
            cad: per_cadence[cad]["per_cell_perm_null"] for cad in per_cadence
        },
        "chimera_permutation_results_per_cell": chimera_out,
        "setup_verdicts": setup_verdicts,
        "chimera_verdicts": chimera_verdicts,
        "cross_cadence_robust_chimera": robust_cross,
        "cross_cadence_robust_setup_family": setup_cross,
    }
    p_perm = OUT_DIR / "pepe_train_permutation_nulls_2026_05_27.json"
    p_perm.write_text(json.dumps(out_perm, indent=2, default=str))
    print(f"WROTE {p_perm}")

    return per_cadence, chimera_out, comparison, setup_verdicts, chimera_verdicts, robust_cross, setup_cross, n_tests_total


if __name__ == "__main__":
    main()
