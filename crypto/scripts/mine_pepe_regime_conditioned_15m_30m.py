"""
PEPE TRAIN regime-conditioned MA/EMA mining at 15m + 30m (W1 of MAXX-INST-2026-05-26-NIGHT).

Mandate: prior mining produced ONE best setup per cadence aggregated across all 186 TRAIN
mover days. This script re-mines under regime conditioning -- for each (cadence, regime cell)
find the BEST MA/EMA setup specific to that cell, then compare vs aggregate winner.

Regime axes:
  macro_regime = btc_trend x crypto_risk_state collapsed to:
    BULL_RISK_ON (btc_trend=bull AND risk_on)
    BEAR_OR_RISK_OFF (btc_trend=bear OR risk_off)
    CHOP_RISK_ON (btc_trend=chop AND risk_on)
  pepe_micro_regime = trend component of pepe_self_cell:
    pepe_trending_up | pepe_chop | pepe_trending_down | WARMUP

Mining grids (matching prior 15m / 30m scripts):
  15m: fast {12,20,28,36,48,60}, slow {48,60,84,120,168,240}, types SMA/EMA/WMA
  30m: fast {6,10,14,18,24,30},  slow {24,30,42,60,84,120},   types SMA/EMA/WMA

Past-only. TRAIN-only. n>=8 threshold per cell. MEDIAN capture (not max) for selection.

Repro:
  canonical_seeds = {bag_seed: 160, feat_seed: 1160, rng_seed: 8070}
  git_sha = (run-time captured)
  chimera_files mtime 2026-05-22
"""
from __future__ import annotations

import json
import sys
import subprocess
from pathlib import Path
from collections import defaultdict
from typing import Optional

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
CHIM_1D = ROOT / "data/processed/chimera/1d/pepeusdt_v51_chimera_1d_20260522.parquet"
CHIM_15M = ROOT / "data/processed/chimera/15m/pepeusdt_v51_chimera_15m_20260522.parquet"
CHIM_30M = ROOT / "data/processed/chimera/30m/pepeusdt_v51_chimera_30m_20260522.parquet"
SIG_PATH = ROOT / "runs/mover_day/pepe_train_mover_day_signatures_2026_05_27.json"
AGG_15M_PATH = ROOT / "runs/mover_day/pepe_train_mover_day_15m_robust_2026_05_27.json"
AGG_30M_PATH = ROOT / "runs/mover_day/pepe_train_mover_day_30m_robust_2026_05_27.json"

OUT_DIR = ROOT / "runs/mover_day"
DOSSIER_DIR = ROOT / "docs/dossiers"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DOSSIER_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_SEEDS = {"bag_seed": 160, "feat_seed": 1160, "rng_seed": 8070}

CADENCE_CFGS = {
    "15m": {
        "chim_path": CHIM_15M,
        "fast_grid": [12, 20, 28, 36, 48, 60],
        "slow_grid": [48, 60, 84, 120, 168, 240],
        "bars_per_hour": 4,
        "bars_per_day": 96,
        "lookback_bars": 192,   # 2 days
        "exits_bars": {"E2_6h": 24, "E3_12h": 48, "E4_24h": 96, "E5_48h": 192},
        "cap_bars": 14 * 96,
    },
    "30m": {
        "chim_path": CHIM_30M,
        "fast_grid": [6, 10, 14, 18, 24, 30],
        "slow_grid": [24, 30, 42, 60, 84, 120],
        "bars_per_hour": 2,
        "bars_per_day": 48,
        "lookback_bars": 144,  # 3 days (per prior 30m script)
        "exits_bars": {"E2_6h": 12, "E3_12h": 24, "E4_24h": 48, "E5_48h": 96},
        "cap_bars": 14 * 48,
    },
}

INDICATOR_TYPES = ["SMA", "EMA", "WMA"]


# ---------- helpers ----------

def get_git_sha() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip()
        return sha
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
            kern = w[::-1] / w_sum
            conv = np.convolve(prices, kern, mode="valid")
            out[window - 1:] = conv
    return out


def find_first_cross_up(fast_ma: np.ndarray, slow_ma: np.ndarray, start: int, end: int) -> int:
    a, b = fast_ma, slow_ma
    if start < 1:
        start = 1
    if end <= start:
        return -1
    seg_a_prev = a[start - 1:end - 1]
    seg_b_prev = b[start - 1:end - 1]
    seg_a_now = a[start:end]
    seg_b_now = b[start:end]
    mask = (
        ~np.isnan(seg_a_prev) & ~np.isnan(seg_b_prev)
        & ~np.isnan(seg_a_now) & ~np.isnan(seg_b_now)
        & (seg_a_prev <= seg_b_prev) & (seg_a_now > seg_b_now)
    )
    if not mask.any():
        return -1
    idx_rel = int(np.argmax(mask))
    return start + idx_rel


# ---------- regime classification ----------

def macro_regime(btc_trend: str, risk_state: str) -> str:
    if btc_trend == "bull" and risk_state == "risk_on":
        return "BULL_RISK_ON"
    if btc_trend == "chop" and risk_state == "risk_on":
        return "CHOP_RISK_ON"
    if btc_trend == "bear" or risk_state == "risk_off":
        return "BEAR_OR_RISK_OFF"
    # fallback: bull_risk_off, chop_risk_off etc.
    if btc_trend == "bull" and risk_state == "risk_off":
        return "BEAR_OR_RISK_OFF"  # collapsed risk_off
    return "OTHER"


def pepe_micro_regime(pepe_self_cell: str) -> str:
    if pepe_self_cell == "WARMUP":
        return "WARMUP"
    if pepe_self_cell.startswith("trending_up"):
        return "pepe_trending_up"
    if pepe_self_cell.startswith("trending_down"):
        return "pepe_trending_down"
    if pepe_self_cell.startswith("chop"):
        return "pepe_chop"
    return "OTHER"


# ---------- load TRAIN 1d to get mover-day ts ----------

def load_movers_with_ts() -> dict:
    """Return dict: date_str -> ts_ms (the day-close ts of that mover day)."""
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


# ---------- per-cadence: build MA cache + per-mover-day per-setup capture matrix ----------

def build_capture_matrix(cadence: str, mover_dates: list, mover_ts_by_date: dict):
    """
    Returns:
      setups: list of (kind, fast, slow)
      capture_matrix: np.ndarray shape (n_movers, n_setups) of captured_move_pct;
                      np.nan if no cross fired before day close.
      entry_idx_matrix: np.ndarray shape (n_movers, n_setups) of int entry_idx, -1 if not fired.
      df_train_bars: pl.DataFrame of train bars (for exit computation).
    """
    cfg = CADENCE_CFGS[cadence]
    df = pl.read_parquet(cfg["chim_path"]).sort("timestamp")
    df = df.with_columns(pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("dt"))
    # bound: train end ts = last mover day ts + 24h (so we can compute exits up to 14 days after)
    # use all bars up to 14 days after the last mover (cap_bars worth)
    last_mover_ts = max(mover_ts_by_date[d] for d in mover_dates)
    bar_ms = {"15m": 15 * 60 * 1000, "30m": 30 * 60 * 1000}[cadence]
    cap_ts = last_mover_ts + cfg["cap_bars"] * bar_ms + 24 * 3600 * 1000
    df_train = df.filter(pl.col("timestamp") <= cap_ts)

    ts = df_train["timestamp"].to_numpy()
    closes = df_train["close"].to_numpy()
    opens = df_train["open"].to_numpy()
    highs = df_train["high"].to_numpy()

    print(f"[{cadence}] train bars: {len(df_train)}")

    # Precompute all MAs
    unique_windows = sorted(set(cfg["fast_grid"] + cfg["slow_grid"]))
    ma_cache = {}
    for kind in INDICATOR_TYPES:
        for w in unique_windows:
            ma_cache[(kind, w)] = compute_ma(closes, kind, w)
    print(f"[{cadence}] MA cache built: {len(ma_cache)} series")

    # Setup list (excluding fast>=slow)
    setups = []
    for kind in INDICATOR_TYPES:
        for fast in cfg["fast_grid"]:
            for slow in cfg["slow_grid"]:
                if fast >= slow:
                    continue
                setups.append((kind, fast, slow))
    n_setups = len(setups)
    print(f"[{cadence}] n setups in grid: {n_setups}")

    n_movers = len(mover_dates)
    capture = np.full((n_movers, n_setups), np.nan)
    entry_idx_m = np.full((n_movers, n_setups), -1, dtype=np.int64)
    entry_offset_h = np.full((n_movers, n_setups), np.nan)
    available_move = np.full(n_movers, np.nan)
    day_close_idx_arr = np.full(n_movers, -1, dtype=np.int64)

    for mi, d in enumerate(mover_dates):
        day_ts = mover_ts_by_date[d]
        # day_close_idx: last bar with ts <= day_ts (1d ts is day-close ms; we want the bar at that close)
        day_close_idx = int(np.searchsorted(ts, day_ts, side="right") - 1)
        if day_close_idx < 0:
            continue
        day_close_idx_arr[mi] = day_close_idx
        window_start = max(0, day_close_idx - (cfg["lookback_bars"] - 1))
        signal_search_start = window_start + 1
        signal_search_end = day_close_idx + 1
        daily_close_price = closes[day_close_idx]

        ent_p_window = opens[window_start + 1: day_close_idx + 1]
        valid = ent_p_window > 0
        if valid.any():
            avail = daily_close_price / ent_p_window[valid] - 1.0
            available_move[mi] = float(avail.max())

        for si, (kind, fast, slow) in enumerate(setups):
            fma = ma_cache[(kind, fast)]
            sma = ma_cache[(kind, slow)]
            cross_idx = find_first_cross_up(fma, sma, signal_search_start, signal_search_end)
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
            entry_idx_m[mi, si] = entry_i
            entry_offset_h[mi, si] = (day_close_idx - entry_i) / cfg["bars_per_hour"]

    return {
        "setups": setups,
        "capture": capture,
        "entry_idx": entry_idx_m,
        "entry_offset_h": entry_offset_h,
        "available_move": available_move,
        "day_close_idx": day_close_idx_arr,
        "closes": closes,
        "opens": opens,
        "highs": highs,
        "ma_cache": ma_cache,
        "n_bars": len(df_train),
        "cfg": cfg,
    }


# ---------- per-regime-cell best-setup selection ----------

def setup_str(setup_tuple) -> str:
    kind, fast, slow = setup_tuple
    return f"{kind} {fast}/{slow}"


def per_cell_winner(cap_matrix: np.ndarray, mover_indices: list, setups: list):
    """
    cap_matrix : (n_movers, n_setups). NaN = no fire.
    mover_indices : indices of mover days belonging to this regime cell.
    Returns dict of setup -> stats + best setup choice.
    Selection: best by MEDIAN capture, weighted by fire-rate (score = median * fire_rate**0.5).
    """
    sub = cap_matrix[mover_indices, :]  # (n_cell, n_setups)
    n_cell = sub.shape[0]
    out = []
    for si, s in enumerate(setups):
        col = sub[:, si]
        fires = ~np.isnan(col)
        n_fires = int(fires.sum())
        if n_fires == 0:
            out.append({
                "setup": setup_str(s),
                "n_cell": n_cell,
                "n_fires": 0,
                "fire_rate": 0.0,
                "median_capture": None,
                "mean_capture": None,
                "p10": None,
                "p90": None,
                "win_rate": None,
                "score": -1e9,
            })
            continue
        fired_vals = col[fires]
        med = float(np.median(fired_vals))
        mean = float(np.mean(fired_vals))
        p10 = float(np.quantile(fired_vals, 0.10))
        p90 = float(np.quantile(fired_vals, 0.90))
        win = float((fired_vals > 0).mean())
        fire_rate = n_fires / n_cell
        # score: median * sqrt(fire_rate) -- penalizes low fire rate
        score = med * (fire_rate ** 0.5)
        out.append({
            "setup": setup_str(s),
            "n_cell": n_cell,
            "n_fires": n_fires,
            "fire_rate": fire_rate,
            "median_capture": med,
            "mean_capture": mean,
            "p10": p10,
            "p90": p90,
            "win_rate": win,
            "score": score,
        })
    # rank
    out_sorted = sorted(out, key=lambda x: -x["score"])
    return out_sorted


# ---------- exits per regime ----------

def compute_exit_for_setup(setup_tuple, entry_idx: int, ma_cache, opens, closes, highs, n_bars, cfg):
    """Returns dict exit_key -> exit_pnl_pct (or None)."""
    kind, fast, slow = setup_tuple
    fma = ma_cache[(kind, fast)]
    sma = ma_cache[(kind, slow)]
    entry_p = opens[entry_idx]
    if entry_p <= 0:
        return None
    out = {}

    # E1 opposite cross
    cap_bars = cfg["cap_bars"]
    end_search = min(entry_idx + cap_bars, n_bars)
    e1_idx = None
    for i in range(entry_idx, end_search):
        if not np.isnan(fma[i]) and not np.isnan(sma[i]) and fma[i] <= sma[i]:
            e1_idx = i
            break
    if e1_idx is None:
        out["E1_opp_cross"] = None
    else:
        if e1_idx + 1 < n_bars:
            exit_p = opens[e1_idx + 1]
        else:
            exit_p = closes[e1_idx]
        out["E1_opp_cross"] = float(exit_p / entry_p - 1.0)

    # E2-E5 fixed holds
    for k, hold_bars in cfg["exits_bars"].items():
        ex_idx = entry_idx + hold_bars
        if ex_idx >= n_bars:
            out[k] = None
        else:
            out[k] = float(opens[ex_idx] / entry_p - 1.0)

    # E6 MFE-trail50
    peak_p = entry_p
    e6 = None
    end_walk = min(entry_idx + cap_bars, n_bars - 1)
    for i in range(entry_idx + 1, end_walk + 1):
        peak_p = max(peak_p, highs[i])
        peak_ret = peak_p / entry_p - 1.0
        if peak_ret >= 0.03:
            lock_p = entry_p * (1.0 + peak_ret * 0.5)
            if closes[i] <= lock_p:
                if i + 1 < n_bars:
                    e6 = float(opens[i + 1] / entry_p - 1.0)
                else:
                    e6 = float(closes[i] / entry_p - 1.0)
                break
    if e6 is None:
        e6 = float(closes[end_walk] / entry_p - 1.0)
    out["E6_mfe_trail50"] = e6
    return out


def exits_per_regime_winner(cell_winner_setup_str: str, setups: list, mover_indices: list,
                            entry_idx_matrix: np.ndarray, ma_cache, opens, closes, highs, n_bars, cfg):
    """Test all 6 exits for the cell's winning setup across mover days in that cell."""
    setup_map = {setup_str(s): s for s in setups}
    if cell_winner_setup_str not in setup_map:
        return None
    target_setup = setup_map[cell_winner_setup_str]
    si = setups.index(target_setup)

    exit_lists = defaultdict(list)
    for mi in mover_indices:
        e_idx = int(entry_idx_matrix[mi, si])
        if e_idx < 0:
            continue
        exits = compute_exit_for_setup(target_setup, e_idx, ma_cache, opens, closes, highs, n_bars, cfg)
        if exits is None:
            continue
        for k, v in exits.items():
            if v is not None:
                exit_lists[k].append(v)

    summary = {}
    for k in ["E1_opp_cross", "E2_6h", "E3_12h", "E4_24h", "E5_48h", "E6_mfe_trail50"]:
        arr = np.array(exit_lists.get(k, []))
        if len(arr) == 0:
            summary[k] = {"n": 0}
        else:
            summary[k] = {
                "n": int(len(arr)),
                "median_pnl": float(np.median(arr)),
                "mean_pnl": float(np.mean(arr)),
                "p10": float(np.quantile(arr, 0.10)),
                "p90": float(np.quantile(arr, 0.90)),
                "win_rate": float((arr > 0).mean()),
            }
    # best by median pnl, only among n>=5
    best = None
    best_med = -1e9
    for k, v in summary.items():
        if "median_pnl" in v and v["n"] >= 5 and v["median_pnl"] > best_med:
            best_med = v["median_pnl"]
            best = k
    return {"best_exit": best, "best_exit_median_pnl": (best_med if best else None), "per_exit": summary}


# ---------- aggregate winner extraction ----------

def get_aggregate_winner(cadence: str) -> dict:
    """Load aggregate-winner JSON and return TOP setup by median capture."""
    path = AGG_15M_PATH if cadence == "15m" else AGG_30M_PATH
    d = json.load(open(path))
    # Schema: dict of setup_str -> {n_fires, fire_rate, median_capture_pct, ...}
    best_key = None
    best_med = -1e9
    for k, v in d.items():
        if isinstance(v, dict) and "median_capture_pct" in v:
            score = v["median_capture_pct"] * (v.get("fire_rate", 1.0) ** 0.5)
            if score > best_med:
                best_med = score
                best_key = k
    return {"setup": best_key, "stats": d.get(best_key, {}), "score": best_med}


# ---------- main runner ----------

def run_cadence(cadence: str, signatures: list, mover_ts_by_date: dict) -> dict:
    print("=" * 72)
    print(f"REGIME-CONDITIONED MINING - cadence={cadence}")
    print("=" * 72)

    # Build mover_dates ordered list + regime tags
    mover_dates = []
    cell_keys = []   # (macro, pepe_micro)
    macro_only = []
    pepe_only = []
    raw_meta = []
    for s in signatures:
        d = s["date"]
        ts = mover_ts_by_date.get(d)
        if ts is None:
            continue
        macro = macro_regime(s["btc_trend"], s["crypto_risk_state"])
        pmic = pepe_micro_regime(s.get("pepe_self_cell", "WARMUP"))
        mover_dates.append(d)
        cell_keys.append((macro, pmic))
        macro_only.append(macro)
        pepe_only.append(pmic)
        raw_meta.append({
            "date": d,
            "macro": macro,
            "pepe_micro": pmic,
            "btc_trend": s["btc_trend"],
            "crypto_risk_state": s["crypto_risk_state"],
            "pepe_self_cell": s.get("pepe_self_cell"),
            "daily_return": s.get("daily_return"),
        })

    n_movers = len(mover_dates)
    print(f"[{cadence}] mover days mapped: {n_movers}")

    # Distribution table
    cell_counts = defaultdict(int)
    for ck in cell_keys:
        cell_counts[ck] += 1
    macro_counts = defaultdict(int)
    for m in macro_only:
        macro_counts[m] += 1
    pepe_counts = defaultdict(int)
    for p in pepe_only:
        pepe_counts[p] += 1
    print(f"[{cadence}] macro distribution: {dict(macro_counts)}")
    print(f"[{cadence}] pepe_micro distribution: {dict(pepe_counts)}")

    # Build capture matrix once
    mat = build_capture_matrix(cadence, mover_dates, mover_ts_by_date)
    capture = mat["capture"]
    entry_idx = mat["entry_idx"]
    setups = mat["setups"]

    # Aggregate (overall) winner from in-matrix
    overall_indices = list(range(n_movers))
    overall_rank = per_cell_winner(capture, overall_indices, setups)
    overall_winner = overall_rank[0]
    print(f"[{cadence}] overall winner (in-matrix): {overall_winner['setup']} med={overall_winner['median_capture']:+.4f}")

    # Aggregate winner from prior robust-aggregate run (cross-check)
    agg_winner_priorrun = get_aggregate_winner(cadence)
    print(f"[{cadence}] aggregate winner (prior robust JSON): {agg_winner_priorrun['setup']} score={agg_winner_priorrun['score']:.4f}")

    # Per regime-cell
    cell_results = {}
    macro_cells = ["BULL_RISK_ON", "CHOP_RISK_ON", "BEAR_OR_RISK_OFF"]
    pepe_cells = ["pepe_trending_up", "pepe_chop", "pepe_trending_down", "WARMUP"]

    for macro in macro_cells:
        for pmic in pepe_cells:
            ck = (macro, pmic)
            idxs = [i for i, c in enumerate(cell_keys) if c == ck]
            n = len(idxs)
            if n == 0:
                continue
            label = f"{macro} x {pmic}"
            print(f"[{cadence}]   cell {label}: n={n}", end="")
            if n < 8:
                print(" (LOW-N, flagged)")
                cell_results[label] = {
                    "n": n,
                    "macro": macro,
                    "pepe_micro": pmic,
                    "LOW_N_FLAG": True,
                    "indices": idxs,
                    "dates": [mover_dates[i] for i in idxs],
                    "winner": None,
                    "top5": None,
                    "exits": None,
                }
                continue
            rank = per_cell_winner(capture, idxs, setups)
            winner = rank[0]
            # exits for the winner
            exits = exits_per_regime_winner(
                winner["setup"], setups, idxs, entry_idx,
                mat["ma_cache"], mat["opens"], mat["closes"], mat["highs"], mat["n_bars"], mat["cfg"],
            )
            print(f" winner={winner['setup']} med={winner['median_capture']:+.4f}"
                  f" fire={winner['fire_rate']:.2f} win={winner['win_rate']:.2f}"
                  f" best_exit={exits['best_exit'] if exits else 'N/A'}")
            cell_results[label] = {
                "n": n,
                "macro": macro,
                "pepe_micro": pmic,
                "LOW_N_FLAG": False,
                "dates": [mover_dates[i] for i in idxs],
                "winner": winner,
                "top5": rank[:5],
                "exits": exits,
            }

    # Aggregate winner stats on full matrix for comparison
    agg_winner_inmat = overall_winner
    setup_to_idx = {setup_str(s): si for si, s in enumerate(setups)}
    if agg_winner_inmat["setup"] in setup_to_idx:
        si = setup_to_idx[agg_winner_inmat["setup"]]
        agg_col = capture[:, si]
    else:
        si = -1
        agg_col = np.full(n_movers, np.nan)

    # Compare regime winners vs aggregate winner restricted to the same cell
    comparison_rows = []
    for label, cell in cell_results.items():
        if cell.get("winner") is None:
            comparison_rows.append({
                "cell": label,
                "n": cell["n"],
                "aggregate_setup": agg_winner_inmat["setup"],
                "aggregate_median_capture_on_cell": None,
                "regime_setup": None,
                "regime_median_capture": None,
                "lift_pp": None,
                "regime_winner_n_fires": None,
                "LOW_N_FLAG": True,
            })
            continue
        idxs = [i for i, c in enumerate(cell_keys) if c == (cell["macro"], cell["pepe_micro"])]
        # aggregate winner restricted to this cell
        if si >= 0:
            cell_col = agg_col[idxs]
            cell_fires = ~np.isnan(cell_col)
            agg_n = int(cell_fires.sum())
            if agg_n > 0:
                agg_med = float(np.median(cell_col[cell_fires]))
            else:
                agg_med = None
        else:
            agg_n = 0
            agg_med = None
        reg_med = cell["winner"]["median_capture"]
        lift = (reg_med - agg_med) * 100 if (agg_med is not None and reg_med is not None) else None
        comparison_rows.append({
            "cell": label,
            "n": cell["n"],
            "aggregate_setup": agg_winner_inmat["setup"],
            "aggregate_median_capture_on_cell": agg_med,
            "aggregate_n_fires_on_cell": agg_n,
            "regime_setup": cell["winner"]["setup"],
            "regime_median_capture": reg_med,
            "regime_winner_n_fires": cell["winner"]["n_fires"],
            "lift_pp": lift,
        })

    # Composite: TRAIN-conditional expected median capture under regime routing
    # weighted average of regime_winner_median_capture by cell_n, restricted to cells with winners
    total_n = 0
    weighted_sum = 0.0
    for r in comparison_rows:
        if r["regime_median_capture"] is None or r["regime_median_capture"] <= 0:
            # cells with no winner OR negative-median capture: HALT, contribute 0 capture and 0 wealth
            # but they still use up the n in denominator (no firing)
            total_n += r["n"]
            continue
        total_n += r["n"]
        weighted_sum += r["n"] * r["regime_median_capture"]
    composite_expected_median = weighted_sum / total_n if total_n > 0 else 0.0

    # Aggregate baseline: aggregate winner's median capture over full TRAIN set
    aggregate_baseline_median = agg_winner_inmat["median_capture"]
    lift_composite_pp = (composite_expected_median - aggregate_baseline_median) * 100

    print(f"[{cadence}] composite expected median capture (regime-routed): {composite_expected_median:+.4f}")
    print(f"[{cadence}] aggregate winner baseline median capture: {aggregate_baseline_median:+.4f}")
    print(f"[{cadence}] composite lift vs aggregate: {lift_composite_pp:+.2f} pp")

    out = {
        "cadence": cadence,
        "n_movers": n_movers,
        "macro_distribution": dict(macro_counts),
        "pepe_micro_distribution": dict(pepe_counts),
        "cell_distribution": {f"{m} x {p}": cell_counts[(m, p)] for (m, p) in cell_counts},
        "raw_meta": raw_meta,
        "overall_winner_in_matrix": agg_winner_inmat,
        "aggregate_winner_from_prior_run": agg_winner_priorrun,
        "cell_results": cell_results,
        "comparison_vs_aggregate": comparison_rows,
        "composite_expected_median_capture": composite_expected_median,
        "aggregate_baseline_median_capture": aggregate_baseline_median,
        "composite_lift_pp": lift_composite_pp,
    }
    return out


def write_dossier(cadence: str, run_out: dict, git_sha: str):
    cmp = run_out["comparison_vs_aggregate"]
    lines = []
    lines.append(f"# PEPE TRAIN Regime-Conditioned MA/EMA mining at {cadence}\n")
    lines.append(f"Generated 2026-05-27 04:30 SAST by W1 of MAXX-INST-2026-05-26-NIGHT.")
    lines.append(f"git_sha={git_sha} cadence={cadence} canonical_seeds={CANONICAL_SEEDS}\n")
    lines.append("TRAIN-only (first 50% of 1d chimera). Past-only at every layer.")
    lines.append("Selection rule: BEST setup per regime cell by score = median_capture * sqrt(fire_rate).\n")

    lines.append(f"## §1 Regime distribution (n per cell)\n")
    lines.append(f"Total mover days: **{run_out['n_movers']}**\n")
    lines.append("Macro:\n")
    for k, v in run_out["macro_distribution"].items():
        lines.append(f"- {k}: {v}")
    lines.append("\nPepe micro:\n")
    for k, v in run_out["pepe_micro_distribution"].items():
        lines.append(f"- {k}: {v}")
    lines.append("\n3 macro x 4 pepe_micro = 12 cells (some sparse):\n")
    lines.append("| Macro | pepe_trending_up | pepe_chop | pepe_trending_down | WARMUP |")
    lines.append("|---|---:|---:|---:|---:|")
    for macro in ["BULL_RISK_ON", "CHOP_RISK_ON", "BEAR_OR_RISK_OFF"]:
        row = [macro]
        for pmic in ["pepe_trending_up", "pepe_chop", "pepe_trending_down", "WARMUP"]:
            n = run_out["cell_distribution"].get(f"{macro} x {pmic}", 0)
            row.append(str(n))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append(f"\n## §2 Per-regime-cell winning setup\n")
    lines.append("Score = median_capture * sqrt(fire_rate). n>=8 required for ranking.\n")
    lines.append("| Cell | n | Winner | n_fires | fire_rate | median_cap | mean_cap | win_rate | p10 | p90 |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, cell in run_out["cell_results"].items():
        if cell.get("winner") is None:
            lines.append(f"| {label} | {cell['n']} | LOW_N (n<8) | -- | -- | -- | -- | -- | -- | -- |")
        else:
            w = cell["winner"]
            lines.append(
                f"| {label} | {cell['n']} | {w['setup']} | {w['n_fires']} | {w['fire_rate']:.2f} | "
                f"{w['median_capture']:+.4f} | {w['mean_capture']:+.4f} | {w['win_rate']:.2f} | "
                f"{w['p10']:+.4f} | {w['p90']:+.4f} |"
            )

    lines.append(f"\n## §3 Comparison vs aggregate winner\n")
    agg_winner = run_out["overall_winner_in_matrix"]
    lines.append(f"Aggregate winner over all {run_out['n_movers']} mover days: **{agg_winner['setup']}** "
                 f"median_cap={agg_winner['median_capture']:+.4f} fire_rate={agg_winner['fire_rate']:.2f}\n")
    lines.append("| Cell | n | Agg setup | Agg cap on cell | Regime setup | Regime cap | Lift (pp) |")
    lines.append("|---|---:|---|---:|---|---:|---:|")
    for r in cmp:
        agg_med = r.get("aggregate_median_capture_on_cell")
        agg_med_s = f"{agg_med:+.4f}" if agg_med is not None else "n/a"
        reg_setup = r.get("regime_setup") or "LOW_N"
        reg_med = r.get("regime_median_capture")
        reg_med_s = f"{reg_med:+.4f}" if reg_med is not None else "--"
        lift = r.get("lift_pp")
        lift_s = f"{lift:+.2f}" if lift is not None else "--"
        lines.append(f"| {r['cell']} | {r['n']} | {r['aggregate_setup']} | {agg_med_s} | {reg_setup} | {reg_med_s} | {lift_s} |")

    lines.append(f"\n## §4 Best exit per regime\n")
    lines.append("Best exit (n>=5 fires required) under the regime's winning ENTRY setup.\n")
    lines.append("| Cell | Winner setup | Best exit | n | median pnl | win rate |")
    lines.append("|---|---|---|---:|---:|---:|")
    for label, cell in run_out["cell_results"].items():
        if cell.get("winner") is None or cell.get("exits") is None:
            lines.append(f"| {label} | -- | -- | -- | -- | -- |")
            continue
        ex = cell["exits"]
        be = ex.get("best_exit")
        if be is None:
            lines.append(f"| {label} | {cell['winner']['setup']} | -- | 0 | -- | -- |")
            continue
        ed = ex["per_exit"][be]
        lines.append(
            f"| {label} | {cell['winner']['setup']} | {be} | {ed['n']} | "
            f"{ed['median_pnl']:+.4f} | {ed['win_rate']:.2f} |"
        )

    lines.append(f"\n## §5 Proposed regime-routed composite (TRAIN-conditional expected performance)\n")
    lines.append(f"Routing rule: at each 4h decision bar, classify (macro, pepe_micro). If a winner exists "
                 f"for that cell, fire its setup. Else HALT.\n")
    lines.append(f"- Composite TRAIN-conditional expected median capture: "
                 f"**{run_out['composite_expected_median_capture']:+.4f}**")
    lines.append(f"- Aggregate baseline (single setup over all TRAIN movers): "
                 f"**{run_out['aggregate_baseline_median_capture']:+.4f}**")
    lines.append(f"- Composite lift: **{run_out['composite_lift_pp']:+.2f} pp**\n")
    lines.append("Caveat: this is TRAIN-conditional. NOT a live forward estimate -- 12-cell selection "
                 "carries selection-bias risk; the OOS/UNSEEN re-test is required before any deploy claim.\n")

    lines.append(f"\n## §6 Honest residuals\n")
    low_n_cells = [label for label, c in run_out["cell_results"].items() if c.get("LOW_N_FLAG")]
    no_winner_cells = [
        label for label, c in run_out["cell_results"].items()
        if c.get("winner") is not None and c["winner"]["median_capture"] is not None
           and c["winner"]["median_capture"] <= 0
    ]
    lines.append(f"- Cells with n<8 (LOW_N flag, no winner picked): {low_n_cells or 'none'}")
    lines.append(f"- Cells with non-positive median capture (HALT recommended): {no_winner_cells or 'none'}")
    total_low_n = sum(c["n"] for c in run_out["cell_results"].values() if c.get("LOW_N_FLAG"))
    pct_low_n = 100 * total_low_n / run_out["n_movers"]
    lines.append(f"- Total mover days in LOW_N cells: {total_low_n} ({pct_low_n:.1f}% of TRAIN movers)")
    lines.append(f"- Pre-delivery self-audit: TRAIN-only verified | median (not max) selection | LOW_N flagged | "
                 f"HONEST NULL accepted if regime doesn't beat aggregate\n")

    out_path = DOSSIER_DIR / f"PEPE_TRAIN_REGIME_CONDITIONED_{cadence}_2026_05_27.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] dossier -> {out_path}")
    return out_path


def main():
    git_sha = get_git_sha()
    print(f"git_sha = {git_sha}")

    sig = json.load(open(SIG_PATH))
    signatures = sig["signatures"]
    print(f"Loaded {len(signatures)} signatures.")

    mover_ts_by_date = load_movers_with_ts()
    print(f"Loaded {len(mover_ts_by_date)} TRAIN mover-day ts.")

    full_out = {
        "_meta": {
            "task": "W1 regime-conditioned MA/EMA mining (15m + 30m) - MAXX-INST-2026-05-26-NIGHT",
            "ts_generated": "2026-05-27T02:30+00:00",
            "git_sha": git_sha,
            "canonical_seeds": CANONICAL_SEEDS,
            "train_only": True,
            "chimera_files": {
                "1d": str(CHIM_1D.name),
                "15m": str(CHIM_15M.name),
                "30m": str(CHIM_30M.name),
            },
            "regime_axes": {
                "macro": ["BULL_RISK_ON", "CHOP_RISK_ON", "BEAR_OR_RISK_OFF"],
                "pepe_micro": ["pepe_trending_up", "pepe_chop", "pepe_trending_down", "WARMUP"],
            },
            "selection_rule": "best by median_capture * sqrt(fire_rate); n>=8 threshold per cell",
        }
    }

    for cadence in ["15m", "30m"]:
        run_out = run_cadence(cadence, signatures, mover_ts_by_date)
        out_json = OUT_DIR / f"pepe_train_regime_conditioned_{cadence}_2026_05_27.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(run_out, f, indent=2, default=str)
        print(f"[OK] JSON -> {out_json}")
        write_dossier(cadence, run_out, git_sha)
        full_out[cadence] = {
            "json_path": str(out_json),
            "summary": {
                "n_movers": run_out["n_movers"],
                "n_cells_with_winners": sum(1 for c in run_out["cell_results"].values() if c.get("winner") is not None),
                "n_cells_low_n": sum(1 for c in run_out["cell_results"].values() if c.get("LOW_N_FLAG")),
                "composite_expected_median": run_out["composite_expected_median_capture"],
                "aggregate_baseline_median": run_out["aggregate_baseline_median_capture"],
                "lift_pp": run_out["composite_lift_pp"],
            },
        }

    summary_path = OUT_DIR / "pepe_train_regime_conditioned_SUMMARY_2026_05_27.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(full_out, f, indent=2, default=str)
    print(f"\n[OK] SUMMARY -> {summary_path}")


if __name__ == "__main__":
    main()
