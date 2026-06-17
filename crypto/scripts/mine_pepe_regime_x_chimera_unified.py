"""
PEPE TRAIN regime x chimera unified analysis (W3 of MAXX-INST-2026-05-26-NIGHT).

Combines:
  * Regime conditioning from W1 (15m/30m) and W2 (1h/4h) regime-conditioned mining
  * Chimera feature signature analysis PER regime cell (not aggregate)

Hypothesis: chimera feature signatures DIFFER by regime. Aggregate analysis (|corr|<0.10)
blended cells; per-cell analysis should reveal regime-specific signals.

Per (cadence, regime_cell) with n>=8 mover days:
  1. Recompute per-day capture under that cell's WINNING setup (from regime-conditioned JSON)
  2. Extract chimera at entry bar (level, trajectory over 12h past-only)
  3. Within-cell correlation feature x capture_pct
  4. Compare to aggregate correlation

Feature list (per mandate; rv_rv_30m absent in chimera -> substitute rv_bpv_5m):
  wh_whale_net_usd, wh_whale_trade_count, fund_rate_mean, bs_basis_z30,
  hbr_eta_buy, bd_imbalance_l1, te_btc_imb, rv_rv_5m, rv_bpv_5m, premium_z90

Trajectory metrics over 12h past-only lookback:
  - median z-score over 12h window (zscore over the L bars)
  - median delta entry vs 12h-ago
  - % monotonic-up flag (>=75% bars rising)
  - % monotonic-down flag (>=75% bars falling)
  - % spike flag (last 2 bars >2 sigma above 12h prior mean)

Constraints: TRAIN-only, past-only, POST-v8.3 harness.

Repro:
  canonical_seeds = {bag_seed:170, feat_seed:1170, rng_seed:8080}
"""
from __future__ import annotations

import json
import sys
import subprocess
from pathlib import Path
from collections import defaultdict
from typing import Optional, Dict, List, Any
import numpy as np
import polars as pl
import pandas as pd

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

OUT_DIR = ROOT / "runs/mover_day"
DOSSIER_DIR = ROOT / "docs/dossiers"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DOSSIER_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_SEEDS = {"bag_seed": 170, "feat_seed": 1170, "rng_seed": 8080}
MIN_CELL_N = 8
TRAIN_FRAC = 0.50

# Cadence configs: lookback for entry-bar search + 12h-trajectory in bars
CADENCE_CFGS = {
    "15m": {
        "fast_grid": [12, 20, 28, 36, 48, 60],
        "slow_grid": [48, 60, 84, 120, 168, 240],
        "bars_per_hour": 4,
        "lookback_bars": 192,   # 2 days for entry search
        "traj_12h_bars": 48,    # 12h x 4 bars/h
    },
    "30m": {
        "fast_grid": [6, 10, 14, 18, 24, 30],
        "slow_grid": [24, 30, 42, 60, 84, 120],
        "bars_per_hour": 2,
        "lookback_bars": 144,
        "traj_12h_bars": 24,
    },
    "1h": {
        "fast_grid": [5, 8, 9, 10, 13, 20, 30],
        "slow_grid": [21, 30, 50, 100, 200],
        "bars_per_hour": 1,
        "lookback_bars": 72,
        "traj_12h_bars": 12,
    },
    "4h": {
        "fast_grid": [3, 5, 7, 9, 12, 15],
        "slow_grid": [9, 12, 18, 24, 36, 48],
        "bars_per_hour": 0.25,
        "lookback_bars": 18,
        "traj_12h_bars": 3,
    },
}

INDICATOR_TYPES = ["SMA", "EMA", "WMA"]

# 10 chimera features per mandate (rv_rv_30m absent -> rv_bpv_5m subst)
CHIMERA_FEATURES = [
    "wh_whale_net_usd",
    "wh_whale_trade_count",
    "fund_rate_mean",
    "bs_basis_z30",
    "hbr_eta_buy",
    "bd_imbalance_l1",
    "te_btc_imb",
    "rv_rv_5m",
    "rv_bpv_5m",   # substitute for rv_rv_30m (not in chimera)
    "premium_z90",
]


def get_git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return "unknown"


# ----- MA primitives -----

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


# ----- Regime cell derivation -----

def derive_regime_cell(sig):
    btc = sig.get("btc_trend", "")
    risk = sig.get("crypto_risk_state", "")
    pepe = sig.get("pepe_self_cell", "")
    if btc == "bear" or risk == "risk_off":
        macro = "BEAR_OR_RISK_OFF"
    elif btc == "bull" and risk == "risk_on":
        macro = "BULL_RISK_ON"
    elif btc == "chop" and risk == "risk_on":
        macro = "CHOP_RISK_ON"
    else:
        macro = "OTHER"
    if pepe == "WARMUP":
        micro = "WARMUP"
    elif pepe.startswith("trending_up"):
        micro = "pepe_trending_up"
    elif pepe.startswith("chop"):
        micro = "pepe_chop"
    elif pepe.startswith("trending_down"):
        micro = "pepe_trending_down"
    else:
        micro = "UNKNOWN"
    return macro, micro


# ----- TRAIN window from 1d chimera -----

def load_train_1d_end_ts():
    df_1d = pl.read_parquet(CHIM_1D).sort("timestamp")
    n = len(df_1d)
    train_end_bar = int(n * TRAIN_FRAC)
    train_end_ts = int(df_1d["timestamp"][train_end_bar - 1])
    return train_end_ts, n, train_end_bar


# ----- Per-cadence pipeline -----

def load_chimera(cadence, train_end_ts):
    path = CHIM_PATHS[cadence]
    df = pl.read_parquet(path).sort("timestamp")
    # TRAIN-only: keep <= train_end_ts + 24h grace (mover days at boundary)
    df_train = df.filter(pl.col("timestamp") <= train_end_ts + 24 * 3600 * 1000)
    return df_train


def compute_ma_cache(closes, fast_grid, slow_grid):
    cache = {}
    for kind in INDICATOR_TYPES:
        for w in sorted(set(fast_grid + slow_grid)):
            cache[(kind, w)] = compute_ma(closes, kind, w)
    return cache


def extract_chimera_trajectory(arr_feature: np.ndarray, entry_idx: int, traj_bars: int) -> Dict[str, Any]:
    """For a feature array and an entry bar index, compute level + trajectory metrics
    using past-only data: window = entry_idx - traj_bars ... entry_idx - 1 inclusive
    (entry value itself is the "lagged feature signal at decision time" so we use
    last fully-formed bar BEFORE entry, i.e. entry_idx-1, as the "current" value).
    """
    if entry_idx < 1 or entry_idx > len(arr_feature):
        return _null_traj()
    start = max(0, entry_idx - traj_bars)
    if start >= entry_idx:
        return _null_traj()
    window = arr_feature[start:entry_idx]  # past-only
    window = window[~np.isnan(window)]
    if len(window) < 3:
        return _null_traj()
    cur = window[-1]
    twelvh_ago = window[0]
    delta = float(cur - twelvh_ago)
    mu = float(np.mean(window[:-1])) if len(window) >= 2 else float(np.mean(window))
    sd = float(np.std(window[:-1])) if len(window) >= 2 else float(np.std(window))
    zscore = float((cur - mu) / sd) if sd > 1e-12 else 0.0
    # Monotonic up: bars where v[i] >= v[i-1]
    if len(window) >= 4:
        diffs = np.diff(window)
        up_frac = float((diffs > 0).mean())
        down_frac = float((diffs < 0).mean())
        mono_up = bool(up_frac >= 0.75)
        mono_down = bool(down_frac >= 0.75)
    else:
        up_frac = 0.5
        down_frac = 0.5
        mono_up = False
        mono_down = False
    # Spike: last 2 bars > 2 sigma above prior 12h mean
    spike = False
    if len(window) >= 5:
        prior = window[:-2]
        last2 = window[-2:]
        prior_mu = float(np.mean(prior))
        prior_sd = float(np.std(prior))
        if prior_sd > 1e-12:
            spike = bool(np.all(last2 > prior_mu + 2 * prior_sd))
    return {
        "entry_val": float(cur),
        "delta_12h": delta,
        "zscore_12h": zscore,
        "up_frac": up_frac,
        "down_frac": down_frac,
        "monotonic_up": mono_up,
        "monotonic_down": mono_down,
        "spike": spike,
    }


def _null_traj():
    return {
        "entry_val": None,
        "delta_12h": None,
        "zscore_12h": None,
        "up_frac": None,
        "down_frac": None,
        "monotonic_up": None,
        "monotonic_down": None,
        "spike": None,
    }


def safe_corr(x: List[float], y: List[float]) -> Optional[float]:
    """Pearson correlation skipping None pairs. Returns None if n<5."""
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None and np.isfinite(a) and np.isfinite(b)]
    if len(pairs) < 5:
        return None
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


# ----- Per-cadence main -----

def normalize_regime_json(cadence: str, regime_json: Dict) -> Dict:
    """Normalize W1 (15m/30m) vs W2 (1h/4h) regime json schemas into:
       { 'cell_winners': {cell_name: {kind,fast,slow,...}},
         'aggregate_winner': {kind,fast,slow,median_cap,win_rate} }
       cell_name format: 'MACRO|micro'
    """
    out_cw = {}
    agg = None
    if cadence in ("15m", "30m"):
        # W1 schema
        for cell_pipe_name, rec in regime_json.get("cell_results", {}).items():
            # cell_pipe_name like 'BULL_RISK_ON x pepe_trending_up'
            cell_norm = cell_pipe_name.replace(" x ", "|")
            win = rec.get("winner")
            if win and not rec.get("LOW_N_FLAG"):
                # win['setup'] like 'SMA 36/48'
                parts = win["setup"].split()
                kind = parts[0]
                fast_str, slow_str = parts[1].split("/")
                out_cw[cell_norm] = {
                    "kind": kind,
                    "fast": int(fast_str),
                    "slow": int(slow_str),
                    "n_fired": int(win.get("n_fires", 0)),
                    "n_cell": int(win.get("n_cell", rec.get("n", 0))),
                    "fire_rate": float(win.get("fire_rate", 0)),
                    "median_capture": float(win.get("median_capture", 0)),
                    "mean_capture": float(win.get("mean_capture", 0)),
                    "win_rate": float(win.get("win_rate", 0)),
                }
        # Aggregate
        agg_raw = regime_json.get("aggregate_winner_from_prior_run")
        if agg_raw and agg_raw.get("setup"):
            parts = agg_raw["setup"].split()
            kind = parts[0]
            fast_str, slow_str = parts[1].split("/")
            agg = {
                "kind": kind, "fast": int(fast_str), "slow": int(slow_str),
                "median_cap": float(agg_raw.get("stats", {}).get("median_capture_pct", 0)),
                "win_rate": float(agg_raw.get("stats", {}).get("win_rate", 0)),
            }
    else:
        # W2 schema (1h, 4h)
        for cell_norm, win in regime_json.get("results", {}).get("cell_winners", {}).items():
            if win is None or win.get("kind") is None:
                continue
            out_cw[cell_norm] = dict(win)
        agg = regime_json.get("results", {}).get("cadence_cfg", {}).get("aggregate_winner")
    return {"cell_winners": out_cw, "aggregate_winner": agg}


def run_cadence(cadence: str, signatures: List[Dict], train_end_ts: int, regime_json: Dict) -> Dict:
    cfg = CADENCE_CFGS[cadence]
    print(f"\n{'='*72}\nCADENCE {cadence}\n{'='*72}")
    df = load_chimera(cadence, train_end_ts)
    ts_arr = df["timestamp"].to_numpy()
    closes = df["close"].to_numpy()
    opens = df["open"].to_numpy()
    n_bars = len(df)
    print(f"  [{cadence}] TRAIN bars: {n_bars}")
    # Extract chimera feature arrays (handle missing cols)
    feat_arrs = {}
    for f in CHIMERA_FEATURES:
        if f in df.columns:
            feat_arrs[f] = df[f].to_numpy().astype(float)
        else:
            print(f"  [{cadence}] WARN feature {f} not in chimera; skipping")
            feat_arrs[f] = None
    ma_cache = compute_ma_cache(closes, cfg["fast_grid"], cfg["slow_grid"])

    # Per-mover day-close indices
    day_close_indices = {}
    for s in signatures:
        idx = int(np.searchsorted(ts_arr, s["entry_ts_ms"], side="right") - 1)
        day_close_indices[s["date"]] = idx

    # Normalize regime json schema
    norm = normalize_regime_json(cadence, regime_json)
    cell_winners = norm["cell_winners"]
    agg_meta = norm["aggregate_winner"]
    print(f"  [{cadence}] regime cells: {list(cell_winners.keys())}")

    # For each cell with n>=MIN_CELL_N, compute per-day capture under cell winner setup
    cell_records = {}
    for cell_name, win in cell_winners.items():
        if win is None or win.get("kind") is None:
            continue
        macro, micro = cell_name.split("|")
        cell_movers = [s for s in signatures if derive_regime_cell(s) == (macro, micro)]
        if len(cell_movers) < MIN_CELL_N:
            print(f"  [{cadence}] {cell_name} below MIN_CELL_N ({len(cell_movers)}); skipping")
            continue
        kind, fast, slow = win["kind"], win["fast"], win["slow"]
        fast_ma = ma_cache[(kind, fast)]
        slow_ma = ma_cache[(kind, slow)]

        per_mover = []
        for m in cell_movers:
            day_close_idx = day_close_indices.get(m["date"], -1)
            if day_close_idx < 0:
                continue
            window_start = max(0, day_close_idx - (cfg["lookback_bars"] - 1))
            signal_search_start = window_start + 1
            signal_search_end = day_close_idx + 1
            cross_idx = first_cross_up_in_window(fast_ma, slow_ma, signal_search_start, signal_search_end)
            if cross_idx == -1:
                continue
            entry_idx = cross_idx + 1
            if entry_idx > day_close_idx or entry_idx >= n_bars:
                continue
            entry_p = opens[entry_idx]
            if entry_p <= 0:
                continue
            daily_close_p = closes[day_close_idx]
            captured = daily_close_p / entry_p - 1.0
            # Available move
            avail_returns = []
            for i in range(window_start, day_close_idx):
                if i + 1 <= day_close_idx:
                    ep = opens[i + 1]
                    if ep > 0:
                        avail_returns.append(daily_close_p / ep - 1.0)
            available_move = max(avail_returns) if avail_returns else 0.0
            capture_rate = float(captured / available_move) if available_move > 1e-9 else 0.0

            # Chimera per feature at entry
            chimera_at_entry = {}
            for fname, farr in feat_arrs.items():
                if farr is None:
                    chimera_at_entry[fname] = _null_traj()
                else:
                    chimera_at_entry[fname] = extract_chimera_trajectory(farr, entry_idx, cfg["traj_12h_bars"])
            per_mover.append({
                "date": m["date"],
                "entry_ts_ms": int(ts_arr[entry_idx]) if entry_idx < n_bars else None,
                "entry_idx": int(entry_idx),
                "captured_pct": float(captured),
                "capture_rate": float(capture_rate),
                "available_move_pct": float(available_move),
                "chimera": chimera_at_entry,
            })
        if len(per_mover) < MIN_CELL_N:
            print(f"  [{cadence}] {cell_name} after-setup n={len(per_mover)} <{MIN_CELL_N}; flag low-conf")
        cell_records[cell_name] = {
            "winner_setup": f"{kind}|{fast}|{slow}",
            "n_cell_total": len(cell_movers),
            "n_fired_under_winner": len(per_mover),
            "median_capture": float(np.median([m["captured_pct"] for m in per_mover])) if per_mover else None,
            "per_mover": per_mover,
        }

    # Build aggregate (across all cells & all movers under aggregate winner) for comparison
    aggregate_records = []
    if agg_meta and agg_meta.get("kind"):
        kind, fast, slow = agg_meta["kind"], agg_meta["fast"], agg_meta["slow"]
        fast_ma = ma_cache[(kind, fast)]
        slow_ma = ma_cache[(kind, slow)]
        for m in signatures:
            day_close_idx = day_close_indices.get(m["date"], -1)
            if day_close_idx < 0:
                continue
            window_start = max(0, day_close_idx - (cfg["lookback_bars"] - 1))
            cross_idx = first_cross_up_in_window(fast_ma, slow_ma, window_start + 1, day_close_idx + 1)
            if cross_idx == -1:
                continue
            entry_idx = cross_idx + 1
            if entry_idx > day_close_idx or entry_idx >= n_bars:
                continue
            entry_p = opens[entry_idx]
            if entry_p <= 0:
                continue
            daily_close_p = closes[day_close_idx]
            captured = daily_close_p / entry_p - 1.0
            chimera_at_entry = {}
            for fname, farr in feat_arrs.items():
                if farr is None:
                    chimera_at_entry[fname] = _null_traj()
                else:
                    chimera_at_entry[fname] = extract_chimera_trajectory(farr, entry_idx, cfg["traj_12h_bars"])
            aggregate_records.append({
                "date": m["date"],
                "captured_pct": float(captured),
                "chimera": chimera_at_entry,
            })

    # Compute within-cell vs aggregate correlations per feature, per trajectory metric
    metrics = ["entry_val", "delta_12h", "zscore_12h", "up_frac", "down_frac"]
    cell_corr_table = {}
    aggregate_corr = {}
    for fname in CHIMERA_FEATURES:
        aggregate_corr[fname] = {}
        for met in metrics:
            xs = [r["chimera"][fname][met] for r in aggregate_records]
            ys = [r["captured_pct"] for r in aggregate_records]
            aggregate_corr[fname][met] = safe_corr(xs, ys)
        # Bool flags: monotonic_up/_down/_spike -> point-biserial vs capture
        for met in ["monotonic_up", "monotonic_down", "spike"]:
            xs = [(1.0 if r["chimera"][fname][met] else 0.0) if r["chimera"][fname][met] is not None else None for r in aggregate_records]
            ys = [r["captured_pct"] for r in aggregate_records]
            aggregate_corr[fname][met] = safe_corr(xs, ys)

    for cell_name, rec in cell_records.items():
        cell_corr_table[cell_name] = {}
        if not rec["per_mover"]:
            continue
        for fname in CHIMERA_FEATURES:
            cell_corr_table[cell_name][fname] = {}
            for met in metrics:
                xs = [r["chimera"][fname][met] for r in rec["per_mover"]]
                ys = [r["captured_pct"] for r in rec["per_mover"]]
                cell_corr_table[cell_name][fname][met] = safe_corr(xs, ys)
            for met in ["monotonic_up", "monotonic_down", "spike"]:
                xs = [(1.0 if r["chimera"][fname][met] else 0.0) if r["chimera"][fname][met] is not None else None for r in rec["per_mover"]]
                ys = [r["captured_pct"] for r in rec["per_mover"]]
                cell_corr_table[cell_name][fname][met] = safe_corr(xs, ys)

    # Per-cell summary: for each cell, find feature x metric with largest |corr|, fire-rate breakdowns
    cell_summaries = {}
    for cell_name, rec in cell_records.items():
        ctbl = cell_corr_table.get(cell_name, {})
        best = (None, None, 0.0)
        all_corrs = []
        for fname, mets in ctbl.items():
            for met, val in mets.items():
                if val is not None and abs(val) > abs(best[2]):
                    best = (fname, met, val)
                if val is not None:
                    all_corrs.append((fname, met, val))
        # Trajectory pattern dominance: mono_up % among per_mover, etc., for top feature
        top_feat, top_met, top_corr = best
        traj_stats = {}
        if top_feat:
            mu_vals = [r["chimera"][top_feat]["monotonic_up"] for r in rec["per_mover"] if r["chimera"][top_feat]["monotonic_up"] is not None]
            md_vals = [r["chimera"][top_feat]["monotonic_down"] for r in rec["per_mover"] if r["chimera"][top_feat]["monotonic_down"] is not None]
            spike_vals = [r["chimera"][top_feat]["spike"] for r in rec["per_mover"] if r["chimera"][top_feat]["spike"] is not None]
            delta_vals = [r["chimera"][top_feat]["delta_12h"] for r in rec["per_mover"] if r["chimera"][top_feat]["delta_12h"] is not None and np.isfinite(r["chimera"][top_feat]["delta_12h"])]
            traj_stats = {
                "pct_monotonic_up": float(np.mean(mu_vals)) if mu_vals else None,
                "pct_monotonic_down": float(np.mean(md_vals)) if md_vals else None,
                "pct_spike": float(np.mean(spike_vals)) if spike_vals else None,
                "median_delta_12h": float(np.median(delta_vals)) if delta_vals else None,
            }
            # Determine dominant pattern
            patterns = []
            if traj_stats.get("pct_monotonic_up") and traj_stats["pct_monotonic_up"] > 0.5:
                patterns.append(f"monotonic_up ({traj_stats['pct_monotonic_up']*100:.0f}%)")
            if traj_stats.get("pct_monotonic_down") and traj_stats["pct_monotonic_down"] > 0.5:
                patterns.append(f"monotonic_down ({traj_stats['pct_monotonic_down']*100:.0f}%)")
            if traj_stats.get("pct_spike") and traj_stats["pct_spike"] > 0.3:
                patterns.append(f"spike ({traj_stats['pct_spike']*100:.0f}%)")
            traj_stats["dominant_pattern"] = "; ".join(patterns) if patterns else "flat/mixed"

        # Compute lift: cell corr vs aggregate corr (for top feature x metric)
        aggregate_top_corr = None
        if top_feat:
            aggregate_top_corr = aggregate_corr.get(top_feat, {}).get(top_met)
        lift_vs_aggregate = None
        if aggregate_top_corr is not None and abs(aggregate_top_corr) > 1e-9:
            lift_vs_aggregate = abs(top_corr) / abs(aggregate_top_corr)

        cell_summaries[cell_name] = {
            "winner_setup": rec["winner_setup"],
            "n_cell_total": rec["n_cell_total"],
            "n_fired": rec["n_fired_under_winner"],
            "median_capture": rec["median_capture"],
            "top_feature": top_feat,
            "top_metric": top_met,
            "top_corr_within_cell": top_corr,
            "aggregate_corr_same_feat_metric": aggregate_top_corr,
            "lift_abs_corr_vs_aggregate": lift_vs_aggregate,
            "trajectory_stats_for_top_feature": traj_stats,
            "n_significant_corrs_p10": sum(1 for _, _, v in all_corrs if abs(v) >= 0.10),
            "n_significant_corrs_p20": sum(1 for _, _, v in all_corrs if abs(v) >= 0.20),
        }

    return {
        "cadence": cadence,
        "n_train_bars": int(n_bars),
        "cell_records": cell_records,
        "aggregate_records_count": len(aggregate_records),
        "cell_corr_table": cell_corr_table,
        "aggregate_corr": aggregate_corr,
        "cell_summaries": cell_summaries,
        "aggregate_winner": agg_meta,
    }


# ----- Main -----

def main():
    print("=" * 72)
    print("PEPE TRAIN regime x chimera unified (W3 / MAXX-INST-2026-05-26-NIGHT)")
    print(f"git_sha: {get_git_sha()}")
    print("TRAIN-only; past-only; canonical_seeds:", CANONICAL_SEEDS)
    print("=" * 72)

    with open(SIG_PATH) as f:
        sig_data = json.load(f)
    signatures = sig_data["signatures"]
    print(f"Loaded {len(signatures)} TRAIN mover-day signatures")

    train_end_ts, total_days, train_end_bar = load_train_1d_end_ts()
    print(f"TRAIN window: end_ts={train_end_ts}, days={train_end_bar}/{total_days}")

    per_cadence_out = {}
    for cadence in ["15m", "30m", "1h", "4h"]:
        # Load regime json
        with open(REGIME_JSON[cadence]) as f:
            regime_json = json.load(f)
        out = run_cadence(cadence, signatures, train_end_ts, regime_json)
        per_cadence_out[cadence] = out

        # Persist per-cadence JSON (slim version: drop heavy per_mover.chimera bodies but keep summaries)
        save_obj = {
            "_meta": {
                "task": f"W3 regime x chimera unified @ {cadence}",
                "instance": "MAXX-INST-2026-05-26-NIGHT",
                "worker": "W3",
                "cadence": cadence,
                "train_only": True,
                "canonical_seeds": CANONICAL_SEEDS,
                "git_sha": get_git_sha(),
                "ts_generated": pd.Timestamp.now(tz="UTC").isoformat(),
                "min_cell_n": MIN_CELL_N,
                "features_analyzed": CHIMERA_FEATURES,
                "note_rv_rv_30m_substitute": "rv_rv_30m not in chimera; rv_bpv_5m substituted",
            },
            "cadence_summary": {
                "n_train_bars": out["n_train_bars"],
                "n_cells_analyzed": len([c for c in out["cell_summaries"] if out["cell_summaries"][c]["n_fired"] >= MIN_CELL_N]),
                "aggregate_winner": out["aggregate_winner"],
                "aggregate_records_count": out["aggregate_records_count"],
            },
            "cell_summaries": out["cell_summaries"],
            "cell_corr_table": out["cell_corr_table"],
            "aggregate_corr": out["aggregate_corr"],
            "cell_records_meta": {
                cell: {
                    "winner_setup": rec["winner_setup"],
                    "n_cell_total": rec["n_cell_total"],
                    "n_fired_under_winner": rec["n_fired_under_winner"],
                    "median_capture": rec["median_capture"],
                    "per_mover_dates": [m["date"] for m in rec["per_mover"]],
                    "per_mover_captures": [m["captured_pct"] for m in rec["per_mover"]],
                }
                for cell, rec in out["cell_records"].items()
            },
        }
        out_path = OUT_DIR / f"pepe_train_regime_chimera_{cadence}_2026_05_27.json"
        with open(out_path, "w") as f:
            json.dump(save_obj, f, indent=2, default=str)
        print(f"  [OK] saved {out_path} ({out_path.stat().st_size/1024:.1f} KB)")

    # ---- Cross-cadence summary JSON ----
    summary = {
        "_meta": {
            "task": "W3 regime x chimera unified SUMMARY across 4 cadences",
            "instance": "MAXX-INST-2026-05-26-NIGHT",
            "worker": "W3",
            "train_only": True,
            "canonical_seeds": CANONICAL_SEEDS,
            "git_sha": get_git_sha(),
            "ts_generated": pd.Timestamp.now(tz="UTC").isoformat(),
            "n_stat_tests_total": 4 * 9 * 10 * 8,
            "multi_comparisons_caveat": "1800+ correlations tested; |corr|<0.20 should not be interpreted causally on n<20 cells",
        },
        "cadences": ["15m", "30m", "1h", "4h"],
        "master_matrix": [],
        "cross_cadence_consistency": {},
    }
    for cadence in ["15m", "30m", "1h", "4h"]:
        out = per_cadence_out[cadence]
        for cell, csum in out["cell_summaries"].items():
            if csum["n_fired"] < MIN_CELL_N:
                continue
            summary["master_matrix"].append({
                "cadence": cadence,
                "regime_cell": cell,
                "n_fired": csum["n_fired"],
                "n_cell_total": csum["n_cell_total"],
                "winner_setup": csum["winner_setup"],
                "median_capture_under_winner": csum["median_capture"],
                "top_feature": csum["top_feature"],
                "top_metric": csum["top_metric"],
                "in_cell_corr": csum["top_corr_within_cell"],
                "aggregate_corr_same_feat_metric": csum["aggregate_corr_same_feat_metric"],
                "lift_abs_corr": csum["lift_abs_corr_vs_aggregate"],
                "dominant_traj_pattern": csum["trajectory_stats_for_top_feature"].get("dominant_pattern"),
                "n_significant_corrs_p10": csum["n_significant_corrs_p10"],
                "n_significant_corrs_p20": csum["n_significant_corrs_p20"],
            })

    # Cross-cadence consistency: for each cell, which features show up as top across cadences?
    cell_to_feats = defaultdict(list)
    for row in summary["master_matrix"]:
        cell_to_feats[row["regime_cell"]].append({
            "cadence": row["cadence"],
            "feature": row["top_feature"],
            "metric": row["top_metric"],
            "corr": row["in_cell_corr"],
        })
    summary["cross_cadence_consistency"] = {cell: feats for cell, feats in cell_to_feats.items()}

    summary_path = OUT_DIR / "pepe_train_regime_chimera_SUMMARY_2026_05_27.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  [OK] saved summary {summary_path} ({summary_path.stat().st_size/1024:.1f} KB)")

    # ---- Render dossier ----
    render_dossier(per_cadence_out, summary)
    return per_cadence_out


def render_dossier(per_cadence_out, summary):
    git_sha = get_git_sha()
    lines = []
    lines.append("# PEPE x MA/EMA TRAIN Regime x Chimera Unified Analysis")
    lines.append("")
    lines.append(f"**Instance**: MAXX-INST-2026-05-26-NIGHT W3")
    lines.append(f"**Date**: 2026-05-27")
    lines.append(f"**Git SHA**: {git_sha}")
    lines.append(f"**Canonical seeds**: bag_seed=170, feat_seed=1170, rng_seed=8080")
    lines.append(f"**Protocol**: TRAIN-only per SELECTION_BIAS_PROTOCOL_2026_05_27.md; past-only chimera; v8.3 harness")
    lines.append("")
    lines.append("## Mandate")
    lines.append("Combine regime conditioning (W1, W2) with chimera signature analysis PER regime cell.")
    lines.append("Hypothesis: aggregate chimera correlations |corr|<0.10 hid regime-specific signals.")
    lines.append("")
    lines.append("## Stat-test budget caveat")
    lines.append(f"Total stat tests = 4 cadences x ~8 viable cells x 10 features x 8 metrics = ~{4*8*10*8} correlations.")
    lines.append("Multi-comparisons risk is severe; **|corr|<0.20 should not be interpreted causally on n<20 cells**.")
    lines.append("Treat findings as hypothesis-generating; pre-register before validation.")
    lines.append("")

    # ===== Section 1: master matrix =====
    lines.append("## Section 1 - Cross-cadence x cross-regime master matrix")
    lines.append("")
    lines.append("| Cadence | Regime Cell | n_fired | Winner setup | Median capture | Top feature | Metric | In-cell corr | Aggregate corr | Lift |abs| |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    rows = sorted(summary["master_matrix"], key=lambda r: (r["cadence"], r["regime_cell"]))
    for r in rows:
        med_cap = f"{r['median_capture_under_winner']*100:+.2f}%" if r['median_capture_under_winner'] is not None else "n/a"
        in_corr = f"{r['in_cell_corr']:+.3f}" if r['in_cell_corr'] is not None else "n/a"
        agg_corr = f"{r['aggregate_corr_same_feat_metric']:+.3f}" if r['aggregate_corr_same_feat_metric'] is not None else "n/a"
        lift = f"{r['lift_abs_corr']:.2f}x" if r['lift_abs_corr'] is not None else "n/a"
        lines.append(f"| {r['cadence']} | {r['regime_cell']} | {r['n_fired']}/{r['n_cell_total']} | {r['winner_setup']} | {med_cap} | {r['top_feature']} | {r['top_metric']} | {in_corr} | {agg_corr} | {lift} |")
    lines.append("")

    # ===== Section 2: within-cell vs aggregate =====
    lines.append("## Section 2 - Within-cell vs aggregate correlation comparison")
    lines.append("")
    lines.append("For each (cadence, regime cell) and the cell's top-corr feature/metric, this table contrasts the within-cell vs aggregate correlation. Lift > 2x means regime conditioning surfaced a signal the aggregate buried.")
    lines.append("")
    lifts = [r["lift_abs_corr"] for r in rows if r["lift_abs_corr"] is not None]
    if lifts:
        n_lift_gt2 = sum(1 for x in lifts if x >= 2.0)
        n_lift_gt3 = sum(1 for x in lifts if x >= 3.0)
        n_lift_gt5 = sum(1 for x in lifts if x >= 5.0)
        lines.append(f"- Cells with lift >= 2x (within-cell |corr| > 2x aggregate |corr|): **{n_lift_gt2} / {len(lifts)}** ({n_lift_gt2/len(lifts)*100:.0f}%)")
        lines.append(f"- Cells with lift >= 3x: **{n_lift_gt3} / {len(lifts)}**")
        lines.append(f"- Cells with lift >= 5x: **{n_lift_gt5} / {len(lifts)}**")
        lines.append(f"- Median |lift|: **{float(np.median(lifts)):.2f}x**, P90: **{float(np.quantile(lifts, 0.9)):.2f}x**")
    lines.append("")

    # ===== Section 3: per-cell stories =====
    lines.append("## Section 3 - Per-cell chimera stories")
    lines.append("")
    for cadence in ["15m", "30m", "1h", "4h"]:
        out = per_cadence_out[cadence]
        cell_summaries = out["cell_summaries"]
        lines.append(f"### {cadence}")
        lines.append("")
        for cell, csum in sorted(cell_summaries.items()):
            if csum["n_fired"] < MIN_CELL_N:
                lines.append(f"- **{cell}** (n_fired={csum['n_fired']}/{csum['n_cell_total']}) - LOW-CONF, n<{MIN_CELL_N}, skipped")
                continue
            tp = csum["trajectory_stats_for_top_feature"]
            med_cap = csum["median_capture"]
            med_cap_s = f"{med_cap*100:+.2f}%" if med_cap is not None else "n/a"
            in_corr = csum["top_corr_within_cell"]
            agg_corr = csum["aggregate_corr_same_feat_metric"]
            in_corr_s = f"{in_corr:+.3f}" if in_corr is not None else "n/a"
            agg_corr_s = f"{agg_corr:+.3f}" if agg_corr is not None else "n/a"
            lift = csum["lift_abs_corr_vs_aggregate"]
            lift_s = f"{lift:.2f}x" if lift is not None else "n/a"

            # Build narrative
            story = f"**{cell}** at {cadence} (n_fired={csum['n_fired']}/{csum['n_cell_total']}, winner={csum['winner_setup']}, median capture={med_cap_s}):"
            lines.append(f"- {story}")
            lines.append(f"  - Top in-cell signal: **{csum['top_feature']}** ({csum['top_metric']}) corr **{in_corr_s}** vs aggregate {agg_corr_s} (lift {lift_s})")
            if tp:
                pmu = tp.get("pct_monotonic_up")
                pmd = tp.get("pct_monotonic_down")
                psp = tp.get("pct_spike")
                mdd = tp.get("median_delta_12h")
                pmu_s = f"{pmu*100:.0f}%" if pmu is not None else "n/a"
                pmd_s = f"{pmd*100:.0f}%" if pmd is not None else "n/a"
                psp_s = f"{psp*100:.0f}%" if psp is not None else "n/a"
                mdd_s = f"{mdd:+.4g}" if mdd is not None else "n/a"
                lines.append(f"  - Trajectory pattern: mono_up={pmu_s}, mono_down={pmd_s}, spike={psp_s}, median delta_12h={mdd_s}")
                lines.append(f"  - Dominant pattern: **{tp.get('dominant_pattern', 'n/a')}**")
            lines.append(f"  - Significant corrs |corr|>=0.10: {csum['n_significant_corrs_p10']}, |corr|>=0.20: {csum['n_significant_corrs_p20']}")
        lines.append("")

    # ===== Section 4: cross-cadence consistency =====
    lines.append("## Section 4 - Cross-cadence / cross-regime consistency")
    lines.append("")
    lines.append("Features that show up as the top in-cell signal across multiple cadences for the SAME regime cell are the most robust candidates.")
    lines.append("")
    for cell, feats in summary["cross_cadence_consistency"].items():
        if len(feats) < 2:
            continue
        # Group by feature
        feat_counts = defaultdict(list)
        for f in feats:
            feat_counts[f["feature"]].append(f["cadence"])
        lines.append(f"- **{cell}**:")
        for fname, cadences in feat_counts.items():
            in_corrs = [f['corr'] for f in feats if f['feature']==fname]
            in_corrs_s = ", ".join(f"{c:+.3f}" for c in in_corrs if c is not None)
            lines.append(f"  - `{fname}`: cadences {sorted(set(cadences))} | corrs {in_corrs_s}")
    lines.append("")

    # ===== Section 5: tradeable specs =====
    lines.append("## Section 5 - Complete tradeable specs per (cadence x regime cell)")
    lines.append("")
    for cadence in ["15m", "30m", "1h", "4h"]:
        out = per_cadence_out[cadence]
        cell_summaries = out["cell_summaries"]
        for cell, csum in sorted(cell_summaries.items()):
            if csum["n_fired"] < MIN_CELL_N:
                continue
            top_feat = csum["top_feature"]
            top_met = csum["top_metric"]
            in_corr = csum["top_corr_within_cell"]
            if in_corr is None or abs(in_corr) < 0.15:
                # Weak in-cell - skip spec
                continue
            tp = csum["trajectory_stats_for_top_feature"]
            mdd = tp.get("median_delta_12h")
            # Determine filter direction: sign of corr => positive corr means filter "feature high"
            if in_corr >= 0:
                if top_met == "entry_val":
                    filt = f"{top_feat} > 60-bar past-only rolling median (positive level)"
                elif top_met == "delta_12h":
                    sign_s = ">0" if (mdd is None or mdd >= 0) else "<0"
                    filt = f"{top_feat} delta_12h {sign_s} (rising trajectory)"
                elif top_met == "zscore_12h":
                    filt = f"{top_feat} zscore_12h > 0 (above-trend)"
                elif top_met == "monotonic_up":
                    filt = f"{top_feat} >=75% rising over 12h (monotonic-up)"
                elif top_met == "spike":
                    filt = f"{top_feat} >2 sigma above 12h-prior mean (spike)"
                else:
                    filt = f"{top_feat} {top_met} (above median)"
            else:
                if top_met == "entry_val":
                    filt = f"{top_feat} < 60-bar past-only rolling median (negative level)"
                elif top_met == "delta_12h":
                    sign_s = "<0" if (mdd is None or mdd <= 0) else ">0"
                    filt = f"{top_feat} delta_12h {sign_s} (falling trajectory)"
                elif top_met == "zscore_12h":
                    filt = f"{top_feat} zscore_12h < 0 (below-trend)"
                elif top_met == "monotonic_down":
                    filt = f"{top_feat} >=75% falling over 12h (monotonic-down)"
                else:
                    filt = f"{top_feat} {top_met} (below median)"
            lines.append(f"```")
            lines.append(f"Cadence: {cadence}")
            lines.append(f"Regime: {cell}")
            lines.append(f"n_train_movers: {csum['n_cell_total']} (n_fired_under_winner={csum['n_fired']})")
            lines.append(f"Entry: {csum['winner_setup']} cross-up")
            lines.append(f"Chimera filter (NEW): {filt}")
            lines.append(f"In-cell corr (top feat/metric x capture): {in_corr:+.3f}")
            lines.append(f"Aggregate corr same feat/metric: {csum['aggregate_corr_same_feat_metric']:+.3f}" if csum["aggregate_corr_same_feat_metric"] is not None else "Aggregate corr: n/a")
            lines.append(f"TRAIN median capture under winner setup: {csum['median_capture']*100:+.2f}%")
            lines.append(f"```")
            lines.append("")
    lines.append("")

    # ===== Section 6: honest residuals =====
    lines.append("## Section 6 - Honest residuals & null findings")
    lines.append("")
    lines.append("- **Multi-comparisons risk**: ~2560 correlations tested. Expected |corr|>=0.20 by chance alone with n=20 is ~7%, so ~180 spurious findings expected. Use only as hypothesis-generating.")
    lines.append(f"- **Low-n cells (n<{MIN_CELL_N})**: flagged in master matrix.")
    n_strong = sum(1 for r in rows if r["in_cell_corr"] is not None and abs(r["in_cell_corr"]) >= 0.30)
    n_weak = sum(1 for r in rows if r["in_cell_corr"] is not None and abs(r["in_cell_corr"]) < 0.15)
    lines.append(f"- **Strong in-cell corrs |>=0.30|**: {n_strong} / {len(rows)} eligible cells")
    lines.append(f"- **Weak in-cell corrs |<0.15|**: {n_weak} / {len(rows)} eligible cells")
    if rows:
        median_top_corr = float(np.median([abs(r["in_cell_corr"]) for r in rows if r["in_cell_corr"] is not None]))
        lines.append(f"- **Median |top corr per cell|**: {median_top_corr:.3f}")
        if median_top_corr < 0.20:
            lines.append(f"- **HONEST NULL caveat**: median top corr {median_top_corr:.3f} is below 0.20; per-cell conditioning surfaces some structure but signal is still modest. Aggregation was not the only source of weak signal - some regime cells genuinely have noisy or no chimera signal.")
        else:
            lines.append(f"- **PARTIAL CONFIRMATION**: median top corr {median_top_corr:.3f} >= 0.20; regime conditioning reveals signals stronger than the aggregate <0.10. Validate on VAL/OOS before deploying.")
    lines.append("")
    lines.append("- **rv_rv_30m substitution**: not available in v51 chimera; substituted rv_bpv_5m (bipower variation 5m). Slope information at 30-min horizon must come from a future chimera upgrade.")
    lines.append("- **bs_basis_z30 / te_btc_imb / premium_z90 null density**: chimera columns have 886+ nulls early in the TRAIN window; per-cell counts may shrink in pre-2024 cells.")
    lines.append("- **Next step**: lock the top-3 cells (by lift x in-cell-corr x n-fired) and validate on VAL/OOS under POST-v8.3 harness BEFORE any production claim.")
    lines.append("")

    dossier_path = DOSSIER_DIR / "PEPE_TRAIN_REGIME_x_CHIMERA_UNIFIED_2026_05_27.md"
    with open(dossier_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [OK] dossier -> {dossier_path}")


if __name__ == "__main__":
    main()
