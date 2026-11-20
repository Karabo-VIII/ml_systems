"""
PEPE TRAIN regime-conditioned MA/EMA mining (W2 of MAXX-INST-2026-05-26-NIGHT).

Brief: re-mine the 186 TRAIN mover days but condition on (macro_regime, pepe_micro_regime).
For each (cadence, regime cell) find the BEST MA/EMA setup specific to that regime.

W2 covers 1h + 4h cadences (W1 covers 15m + 30m in parallel).

Protocol: docs/SELECTION_BIAS_PROTOCOL_2026_05_27.md - TRAIN-ONLY.
Signatures: runs/mover_day/pepe_train_mover_day_signatures_2026_05_27.json (186 movers tagged).

Phases:
  1. Load signatures and derive macro/micro regime cells.
  2. Per (macro, micro) cell with n>=8: sweep grid, pick best setup by median capture.
  3. Compare per-regime winners to aggregate winners.
  4. Regime-routed composite (weighted by cell n).
  5. 6-exit comparison per regime.

Past-only at every layer. UNSEEN/VAL/OOS untouched.

Repro:
  canonical_seeds = {bag_seed:161, feat_seed:1161, rng_seed:8071}
  git_sha = pre-execution
"""
import json
import sys
from pathlib import Path
import numpy as np
import polars as pl
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CHIM_1D = ROOT / "data/processed/chimera/1d/pepeusdt_v51_chimera_1d_20260522.parquet"
CHIM_1H = ROOT / "data/processed/chimera/1h/pepeusdt_v51_chimera_1h_20260522.parquet"
CHIM_4H = ROOT / "data/processed/chimera/4h/pepeusdt_v51_chimera_4h_20260522.parquet"

SIG_PATH = ROOT / "runs/mover_day/pepe_train_mover_day_signatures_2026_05_27.json"
OUT_DIR = ROOT / "runs/mover_day"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_SEEDS = {"bag_seed": 161, "feat_seed": 1161, "rng_seed": 8071}
GIT_SHA = "1985437"  # most recent commit per gitStatus

TRAIN_FRAC = 0.50

# ---------- Cadence parameter blocks ----------

CADENCE_CONFIGS = {
    "1h": {
        "chim_file": CHIM_1H,
        # Brief grid + we ensure fast=9 is included so the aggregate winner SMA(9,21) is computable
        "fast_grid": [5, 8, 9, 10, 13, 20, 30],
        "slow_grid": [21, 30, 50, 100, 200],
        "bar_hours": 1.0,
        "lookback_bars": 72,   # 3 days
        "exit_horizons_bars": {"E2_6h": 6, "E3_12h": 12, "E4_24h": 24, "E5_48h": 48},
        "exit_cap_bars": 14 * 24,
        "aggregate_winner": {"kind": "SMA", "fast": 9, "slow": 21, "median_cap": 0.0469, "win_rate": 0.81},
    },
    "4h": {
        "chim_file": CHIM_4H,
        "fast_grid": [3, 5, 7, 9, 12, 15],
        "slow_grid": [9, 12, 18, 24, 36, 48],
        "bar_hours": 4.0,
        "lookback_bars": 18,   # 3 days at 4h
        "exit_horizons_bars": {"E2_6h": 2, "E3_12h": 3, "E4_24h": 6, "E5_48h": 12},
        "exit_cap_bars": 14 * 6,
        "aggregate_winner": {"kind": "SMA", "fast": 7, "slow": 9, "median_cap": 0.0470, "win_rate": 0.80},
    },
}

INDICATOR_TYPES = ["SMA", "EMA", "WMA"]
MIN_CELL_N = 8

# ---------- Regime cell derivation ----------

def derive_regime_cell(sig):
    """Derive (macro_regime, pepe_micro_regime) from existing tags.

    Macro:
      BULL_RISK_ON       = btc_trend=='bull' and crypto_risk_state=='risk_on'
      CHOP_RISK_ON       = btc_trend=='chop' and crypto_risk_state=='risk_on'
      BEAR_OR_RISK_OFF   = btc_trend=='bear' OR crypto_risk_state=='risk_off'

    PEPE micro: parse pepe_self_cell which is 'trending_up_x_high_vol' style.
      'trending_up_*'   -> pepe_trending_up
      'chop_*'          -> pepe_chop
      'trending_down_*' -> pepe_trending_down
      'WARMUP'          -> WARMUP (excluded from cell mining)
    """
    btc = sig.get("btc_trend", "")
    risk = sig.get("crypto_risk_state", "")
    pepe = sig.get("pepe_self_cell", "")

    # Macro
    if btc == "bear" or risk == "risk_off":
        macro = "BEAR_OR_RISK_OFF"
    elif btc == "bull" and risk == "risk_on":
        macro = "BULL_RISK_ON"
    elif btc == "chop" and risk == "risk_on":
        macro = "CHOP_RISK_ON"
    else:
        macro = "OTHER"

    # PEPE micro
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


# ---------- MA primitives (cached past-only) ----------

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


# ---------- Phase 1: load mover days + tag regime cells ----------

def load_signatures():
    with open(SIG_PATH) as f:
        sigs = json.load(f)["signatures"]
    print(f"[Phase 1] Loaded {len(sigs)} TRAIN mover-day signatures")
    enriched = []
    for s in sigs:
        macro, micro = derive_regime_cell(s)
        enriched.append({
            "date": s["date"],
            "ts_ms": s["entry_ts_ms"],
            "daily_return": s["daily_return"],
            "hi_move": s.get("hi_move"),
            "btc_trend": s.get("btc_trend"),
            "crypto_risk_state": s.get("crypto_risk_state"),
            "pepe_self_cell": s.get("pepe_self_cell"),
            "btc_vol_state": s.get("btc_vol_state"),
            "btc_30d_quartile": s.get("btc_30d_return_quartile"),
            "macro_regime": macro,
            "pepe_micro_regime": micro,
            "regime_cell": f"{macro}|{micro}",
        })
    # cell distribution
    cells = {}
    for e in enriched:
        cells[e["regime_cell"]] = cells.get(e["regime_cell"], 0) + 1
    print("[Phase 1] Regime-cell distribution:")
    for c, n in sorted(cells.items(), key=lambda kv: -kv[1]):
        flag = "" if n >= MIN_CELL_N else "  [< MIN_CELL_N]"
        print(f"  {c:50s}  n={n:3d}{flag}")
    return enriched, cells


# ---------- Phase 2 helpers: setup sweep per mover day ----------

def compute_per_day_capture(movers_enriched, df_chim, cadence_cfg, ma_cache, ts_arr, closes, opens, day_close_indices):
    """For each mover, sweep grid and record capture per (kind, fast, slow).

    Returns list of dicts keyed by setup tuple -> capture_pct, plus available_move.
    """
    fast_grid = cadence_cfg["fast_grid"]
    slow_grid = cadence_cfg["slow_grid"]
    lookback = cadence_cfg["lookback_bars"]

    per_day = []
    for mi, m in enumerate(movers_enriched):
        day_close_idx = day_close_indices[mi]
        if day_close_idx < 0:
            per_day.append({
                "date": m["date"], "regime_cell": m["regime_cell"],
                "macro_regime": m["macro_regime"], "pepe_micro_regime": m["pepe_micro_regime"],
                "day_close_idx": -1, "available_move_pct": None,
                "setup_results": {}, "status": "NO_BAR",
            })
            continue
        window_start = max(0, day_close_idx - (lookback - 1))
        signal_search_start = window_start + 1
        signal_search_end = day_close_idx + 1
        daily_close_price = closes[day_close_idx]

        # Available move
        avail_returns = []
        for i in range(window_start, day_close_idx):
            if i + 1 <= day_close_idx:
                ent_p = opens[i + 1]
                if ent_p > 0:
                    avail_returns.append(daily_close_price / ent_p - 1.0)
        available_move = max(avail_returns) if avail_returns else 0.0

        setup_results = {}
        for kind in INDICATOR_TYPES:
            for fast in fast_grid:
                for slow in slow_grid:
                    if fast >= slow:
                        continue
                    fast_ma = ma_cache[(kind, fast)]
                    slow_ma = ma_cache[(kind, slow)]
                    cross_idx = first_cross_up_in_window(
                        fast_ma, slow_ma, signal_search_start, signal_search_end
                    )
                    if cross_idx == -1:
                        continue
                    if cross_idx + 1 > day_close_idx:
                        continue
                    entry_idx = cross_idx + 1
                    entry_p = opens[entry_idx]
                    if entry_p <= 0:
                        continue
                    captured = daily_close_price / entry_p - 1.0
                    setup_results[f"{kind}|{fast}|{slow}"] = {
                        "captured_pct": float(captured),
                        "available_pct": float(available_move),
                        "capture_rate": float(captured / available_move) if available_move > 1e-9 else 0.0,
                        "entry_bar_idx": int(entry_idx),
                        "signal_bar_idx": int(cross_idx),
                        "kind": kind, "fast": fast, "slow": slow,
                    }
        per_day.append({
            "date": m["date"], "regime_cell": m["regime_cell"],
            "macro_regime": m["macro_regime"], "pepe_micro_regime": m["pepe_micro_regime"],
            "day_close_idx": int(day_close_idx),
            "available_move_pct": float(available_move),
            "setup_results": setup_results,
            "n_fired": len(setup_results),
            "status": "OK" if setup_results else "NO_SETUP",
        })
    return per_day


def load_chimera_and_cache(cadence, cadence_cfg, df_train_1d, movers_enriched):
    """Load chimera at cadence, restrict to TRAIN window, precompute MA cache, locate day-close indices."""
    df = pl.read_parquet(cadence_cfg["chim_file"]).sort("timestamp")
    df = df.with_columns(pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("dt"))
    train_end_ts = int(df_train_1d["timestamp"][-1])
    df_train = df.filter(pl.col("timestamp") <= train_end_ts + 24 * 3600 * 1000)
    ts_arr = df_train["timestamp"].to_numpy()
    closes = df_train["close"].to_numpy()
    opens = df_train["open"].to_numpy()
    highs = df_train["high"].to_numpy()
    n_bars = len(df_train)
    print(f"[Phase 2 {cadence}] TRAIN {cadence} bars: {n_bars}")

    ma_cache = {}
    for kind in INDICATOR_TYPES:
        for w in sorted(set(cadence_cfg["fast_grid"] + cadence_cfg["slow_grid"])):
            ma_cache[(kind, w)] = compute_ma(closes, kind, w)
    print(f"[Phase 2 {cadence}] MA cache: {len(ma_cache)} series")

    day_close_indices = []
    for m in movers_enriched:
        idx = int(np.searchsorted(ts_arr, m["ts_ms"], side="right") - 1)
        day_close_indices.append(idx)
    return df_train, ts_arr, closes, opens, highs, ma_cache, day_close_indices


# ---------- Phase 2: per-cell winner ----------

def find_per_cell_winners(per_day_setups, cells_dist, cadence_cfg):
    """For each regime cell with n>=MIN_CELL_N, aggregate the setup-results across days
    and pick the setup that maximizes median capture (tiebreak: fire rate, then mean capture).
    """
    fast_grid = cadence_cfg["fast_grid"]
    slow_grid = cadence_cfg["slow_grid"]

    cell_winners = {}
    cell_full_stats = {}

    cells_to_mine = [c for c, n in cells_dist.items() if n >= MIN_CELL_N and "WARMUP" not in c and "UNKNOWN" not in c and "OTHER" not in c]
    print(f"[Phase 2] Cells with n>={MIN_CELL_N} eligible for mining: {len(cells_to_mine)}")
    for c in cells_to_mine:
        print(f"   {c}  n={cells_dist[c]}")

    for cell in cells_to_mine:
        cell_days = [d for d in per_day_setups if d["regime_cell"] == cell]
        n_cell = len(cell_days)
        # Aggregate per setup
        setup_stats = {}
        for kind in INDICATOR_TYPES:
            for fast in fast_grid:
                for slow in slow_grid:
                    if fast >= slow:
                        continue
                    key = f"{kind}|{fast}|{slow}"
                    captures = []
                    capture_rates = []
                    for d in cell_days:
                        if key in d["setup_results"]:
                            captures.append(d["setup_results"][key]["captured_pct"])
                            capture_rates.append(d["setup_results"][key]["capture_rate"])
                    if len(captures) == 0:
                        continue
                    arr = np.array(captures)
                    setup_stats[key] = {
                        "kind": kind, "fast": fast, "slow": slow,
                        "n_fired": int(len(arr)),
                        "n_cell": int(n_cell),
                        "fire_rate": float(len(arr) / n_cell),
                        "median_capture": float(np.median(arr)),
                        "mean_capture": float(np.mean(arr)),
                        "p10": float(np.quantile(arr, 0.10)),
                        "p90": float(np.quantile(arr, 0.90)),
                        "win_rate": float((arr > 0).mean()),
                        "median_capture_rate": float(np.median(capture_rates)),
                    }
        # Pick winner: maximize median_capture with fire_rate >= 0.5 (need it to fire often enough to be useful)
        eligible = [(k, v) for k, v in setup_stats.items() if v["fire_rate"] >= 0.5]
        if not eligible:
            # Fallback: any fire-rate, maximize median_capture
            eligible = list(setup_stats.items())
        if not eligible:
            cell_winners[cell] = None
            cell_full_stats[cell] = setup_stats
            continue
        # Primary sort: median_capture desc; tiebreak: fire_rate desc; secondary: mean_capture
        eligible.sort(key=lambda kv: (-kv[1]["median_capture"], -kv[1]["fire_rate"], -kv[1]["mean_capture"]))
        winner_key, winner_stats = eligible[0]
        cell_winners[cell] = winner_stats
        cell_full_stats[cell] = setup_stats

    return cell_winners, cell_full_stats


# ---------- Phase 3: Comparison vs aggregate ----------

def phase3_aggregate_vs_regime(cell_winners, per_day_setups, cells_dist, cadence_cfg):
    agg = cadence_cfg["aggregate_winner"]
    agg_key = f"{agg['kind']}|{agg['fast']}|{agg['slow']}"

    rows = []
    for cell, stats in cell_winners.items():
        if stats is None:
            continue
        # Aggregate winner's stats restricted to THIS cell's days
        cell_days = [d for d in per_day_setups if d["regime_cell"] == cell]
        agg_captures = [d["setup_results"][agg_key]["captured_pct"] for d in cell_days if agg_key in d["setup_results"]]
        if agg_captures:
            agg_median = float(np.median(agg_captures))
            agg_fire = len(agg_captures) / len(cell_days)
            agg_win = float((np.array(agg_captures) > 0).mean())
        else:
            agg_median = None
            agg_fire = 0.0
            agg_win = None

        lift = None
        if agg_median is not None and abs(agg_median) > 1e-9:
            lift = (stats["median_capture"] - agg_median) / abs(agg_median)
        elif agg_median is None:
            lift = None  # cannot compute
        else:
            lift = stats["median_capture"]  # agg=0 case

        rows.append({
            "regime_cell": cell,
            "n_cell": cells_dist[cell],
            "aggregate_winner": agg_key,
            "aggregate_median_capture_in_cell": agg_median,
            "aggregate_fire_rate_in_cell": agg_fire,
            "aggregate_win_rate_in_cell": agg_win,
            "regime_winner": f"{stats['kind']}|{stats['fast']}|{stats['slow']}",
            "regime_median_capture": stats["median_capture"],
            "regime_fire_rate": stats["fire_rate"],
            "regime_win_rate": stats["win_rate"],
            "lift_vs_aggregate": lift,
        })
    return rows


# ---------- Phase 4: regime-routed composite ----------

def phase4_composite(cell_winners, per_day_setups, cells_dist):
    """Weighted-by-n composite: for each cell, take the regime winner's median_capture and weight by n_cell."""
    total_n = 0
    weighted_sum = 0.0
    fired_total = 0
    captures_all = []  # actual captures from each cell's winner applied to its days

    for cell, stats in cell_winners.items():
        if stats is None:
            continue
        cell_days = [d for d in per_day_setups if d["regime_cell"] == cell]
        n = len(cell_days)
        key = f"{stats['kind']}|{stats['fast']}|{stats['slow']}"
        cell_caps = [d["setup_results"][key]["captured_pct"] for d in cell_days if key in d["setup_results"]]
        fired_total += len(cell_caps)
        captures_all.extend(cell_caps)
        total_n += n
        weighted_sum += stats["median_capture"] * n

    weighted_median = weighted_sum / total_n if total_n > 0 else None
    if captures_all:
        arr = np.array(captures_all)
        composite = {
            "n_total_days_covered": total_n,
            "n_fired_total": fired_total,
            "fire_rate_overall": float(fired_total / total_n) if total_n > 0 else 0.0,
            "composite_median_capture_weighted": float(weighted_median) if weighted_median is not None else None,
            "composite_median_capture_realized": float(np.median(arr)),
            "composite_mean_capture_realized": float(np.mean(arr)),
            "composite_p10_realized": float(np.quantile(arr, 0.10)),
            "composite_p90_realized": float(np.quantile(arr, 0.90)),
            "composite_win_rate_realized": float((arr > 0).mean()),
        }
    else:
        composite = {
            "n_total_days_covered": 0,
            "n_fired_total": 0,
            "fire_rate_overall": 0.0,
            "composite_median_capture_weighted": None,
            "composite_median_capture_realized": None,
            "composite_mean_capture_realized": None,
            "composite_win_rate_realized": None,
        }
    return composite


# ---------- Phase 5: 6-exit per regime ----------

def phase5_exits_per_regime(cell_winners, per_day_setups, cadence_cfg, closes, opens, highs, ma_cache):
    n_bars = len(closes)
    exit_horizons = cadence_cfg["exit_horizons_bars"]
    cap_bars = cadence_cfg["exit_cap_bars"]
    EXIT_NAMES = ["E1_opp_cross"] + list(exit_horizons.keys()) + ["E6_mfe_trail50"]

    per_regime_exits = {}

    for cell, stats in cell_winners.items():
        if stats is None:
            continue
        kind = stats["kind"]; fast = stats["fast"]; slow = stats["slow"]
        fast_ma = ma_cache[(kind, fast)]
        slow_ma = ma_cache[(kind, slow)]
        key = f"{kind}|{fast}|{slow}"

        results = {e: [] for e in EXIT_NAMES}
        cell_days = [d for d in per_day_setups if d["regime_cell"] == cell]
        for d in cell_days:
            if key not in d["setup_results"]:
                continue
            entry_idx = d["setup_results"][key]["entry_bar_idx"]
            entry_p = opens[entry_idx]
            if entry_p <= 0:
                continue
            # E1
            e1_exit_idx = None
            for i in range(entry_idx, min(entry_idx + cap_bars, n_bars)):
                if (not np.isnan(fast_ma[i]) and not np.isnan(slow_ma[i]) and fast_ma[i] <= slow_ma[i]):
                    e1_exit_idx = i
                    break
            if e1_exit_idx is None or e1_exit_idx + 1 >= n_bars:
                results["E1_opp_cross"].append(None)
            else:
                results["E1_opp_cross"].append(float(opens[e1_exit_idx + 1] / entry_p - 1.0))

            # E2-E5
            for ek, kbars in exit_horizons.items():
                ex_idx = entry_idx + kbars
                if ex_idx >= n_bars:
                    results[ek].append(None)
                else:
                    results[ek].append(float(opens[ex_idx] / entry_p - 1.0))

            # E6 MFE-trail50
            peak_p = entry_p
            e6_exit = None
            end_walk = min(entry_idx + cap_bars, n_bars - 1)
            for i in range(entry_idx + 1, end_walk + 1):
                peak_p = max(peak_p, highs[i])
                peak_ret = peak_p / entry_p - 1.0
                if peak_ret >= 0.03:
                    lock_p = entry_p * (1.0 + peak_ret * 0.5)
                    if closes[i] <= lock_p:
                        if i + 1 < n_bars:
                            e6_exit = float(opens[i + 1] / entry_p - 1.0)
                        else:
                            e6_exit = float(closes[i] / entry_p - 1.0)
                        break
            if e6_exit is None:
                e6_exit = float(closes[end_walk] / entry_p - 1.0)
            results["E6_mfe_trail50"].append(e6_exit)

        summary = {}
        for e in EXIT_NAMES:
            arr = np.array([v for v in results[e] if v is not None])
            if len(arr) == 0:
                summary[e] = {"n": 0}
                continue
            summary[e] = {
                "n": int(len(arr)),
                "median_pnl": float(np.median(arr)),
                "mean_pnl": float(np.mean(arr)),
                "p10": float(np.quantile(arr, 0.10)),
                "p90": float(np.quantile(arr, 0.90)),
                "win_rate": float((arr > 0).mean()),
            }
        # Pick best exit by median PnL among those with n >= 5
        eligible = [(e, s) for e, s in summary.items() if s.get("n", 0) >= 5]
        if eligible:
            eligible.sort(key=lambda kv: (-kv[1]["median_pnl"], -kv[1].get("win_rate", 0)))
            best_exit, best_stats = eligible[0]
        else:
            best_exit, best_stats = None, None

        per_regime_exits[cell] = {
            "winner_setup": key,
            "exits": summary,
            "best_exit_by_median_pnl": best_exit,
            "best_exit_stats": best_stats,
        }
    return per_regime_exits


# ---------- Phase 1d: TRAIN window from 1d chimera ----------

def load_train_1d():
    df_1d = pl.read_parquet(CHIM_1D).sort("timestamp")
    n = len(df_1d)
    df_1d = df_1d.with_columns(pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("dt"))
    train_end_bar = int(n * TRAIN_FRAC)
    df_train_1d = df_1d[:train_end_bar]
    return df_train_1d


# ---------- Main per-cadence pipeline ----------

def run_cadence(cadence, cadence_cfg, movers_enriched, cells_dist, df_train_1d):
    print("=" * 72)
    print(f"REGIME-CONDITIONED MINING @ {cadence}")
    print("=" * 72)

    # Load chimera + MA cache + day-close indices
    df_train, ts_arr, closes, opens, highs, ma_cache, day_close_indices = load_chimera_and_cache(
        cadence, cadence_cfg, df_train_1d, movers_enriched
    )

    # Per-day setup sweep
    per_day_setups = compute_per_day_capture(
        movers_enriched, df_train, cadence_cfg, ma_cache, ts_arr, closes, opens, day_close_indices
    )
    n_ok = sum(1 for d in per_day_setups if d["status"] == "OK")
    n_no_setup = sum(1 for d in per_day_setups if d["status"] == "NO_SETUP")
    n_no_bar = sum(1 for d in per_day_setups if d["status"] == "NO_BAR")
    print(f"[Phase 2 {cadence}] {n_ok} OK / {n_no_setup} NO_SETUP / {n_no_bar} NO_BAR of {len(per_day_setups)}")

    # Per-cell winners
    cell_winners, cell_full_stats = find_per_cell_winners(per_day_setups, cells_dist, cadence_cfg)

    print(f"\n[Phase 2 {cadence}] Per-cell winners:")
    for cell, stats in cell_winners.items():
        if stats is None:
            print(f"   {cell}: NO WINNER (all setups had insufficient fires)")
        else:
            print(f"   {cell:50s}  {stats['kind']}({stats['fast']},{stats['slow']})  med={stats['median_capture']:+.4f}  fire={stats['fire_rate']:.2f}  win={stats['win_rate']:.2%}  n_fired={stats['n_fired']}/{stats['n_cell']}")

    # Phase 3
    comparison_rows = phase3_aggregate_vs_regime(cell_winners, per_day_setups, cells_dist, cadence_cfg)
    print(f"\n[Phase 3 {cadence}] Aggregate-vs-regime comparison ({len(comparison_rows)} cells):")
    print(f"   Aggregate winner: {cadence_cfg['aggregate_winner']}")
    for r in sorted(comparison_rows, key=lambda x: -(x["lift_vs_aggregate"] or -999)):
        lift_str = f"{r['lift_vs_aggregate']:+.3f}" if r["lift_vs_aggregate"] is not None else "  N/A"
        agg_med_str = f"{r['aggregate_median_capture_in_cell']:+.4f}" if r['aggregate_median_capture_in_cell'] is not None else "   None"
        print(f"   {r['regime_cell']:50s}  n={r['n_cell']:3d}  agg_med={agg_med_str}  reg_med={r['regime_median_capture']:+.4f}  lift={lift_str}")

    # Phase 4
    composite = phase4_composite(cell_winners, per_day_setups, cells_dist)
    print(f"\n[Phase 4 {cadence}] Regime-routed composite:")
    for k, v in composite.items():
        print(f"   {k:42s}  {v}")

    # Phase 5
    per_regime_exits = phase5_exits_per_regime(cell_winners, per_day_setups, cadence_cfg, closes, opens, highs, ma_cache)
    print(f"\n[Phase 5 {cadence}] Best-exit per regime:")
    for cell, ex in per_regime_exits.items():
        if ex["best_exit_by_median_pnl"]:
            stats = ex["best_exit_stats"]
            print(f"   {cell:50s}  best={ex['best_exit_by_median_pnl']:18s}  med_pnl={stats['median_pnl']:+.4f}  win={stats['win_rate']:.2%}  n={stats['n']}")
        else:
            print(f"   {cell:50s}  NO BEST EXIT (insufficient fires)")

    return {
        "cadence": cadence,
        "cadence_cfg": {
            "fast_grid": cadence_cfg["fast_grid"],
            "slow_grid": cadence_cfg["slow_grid"],
            "lookback_bars": cadence_cfg["lookback_bars"],
            "exit_horizons_bars": cadence_cfg["exit_horizons_bars"],
            "aggregate_winner": cadence_cfg["aggregate_winner"],
        },
        "n_train_bars": int(len(df_train)),
        "per_day_setups_summary": {
            "n_ok": n_ok, "n_no_setup": n_no_setup, "n_no_bar": n_no_bar,
            "total_mover_days": len(per_day_setups),
        },
        "cell_winners": cell_winners,
        "phase3_comparison": comparison_rows,
        "phase4_composite": composite,
        "phase5_exits_per_regime": per_regime_exits,
    }


def main():
    print("=" * 72)
    print("PEPE TRAIN REGIME-CONDITIONED MINING - W2 (1h + 4h)")
    print("MAXX-INST-2026-05-26-NIGHT  |  TRAIN-ONLY (Selection-Bias Protocol)")
    print("=" * 72)

    # Phase 1: load signatures and tag regime cells
    movers_enriched, cells_dist = load_signatures()
    df_train_1d = load_train_1d()
    print(f"\n[Phase 1] TRAIN 1d window: {df_train_1d['dt'][0]} -> {df_train_1d['dt'][-1]} ({len(df_train_1d)} days)")

    out_per_cadence = {}
    for cadence in ["1h", "4h"]:
        cad_cfg = CADENCE_CONFIGS[cadence]
        out_per_cadence[cadence] = run_cadence(cadence, cad_cfg, movers_enriched, cells_dist, df_train_1d)

        # Save JSON per cadence
        out = {
            "_meta": {
                "task": f"W2 regime-conditioned MA/EMA mining @ {cadence}",
                "instance": "MAXX-INST-2026-05-26-NIGHT",
                "worker": "W2",
                "cadence": cadence,
                "train_only": True,
                "selection_protocol": "docs/SELECTION_BIAS_PROTOCOL_2026_05_27.md",
                "canonical_seeds": CANONICAL_SEEDS,
                "git_sha": GIT_SHA,
                "min_cell_n": MIN_CELL_N,
                "n_movers_total": len(movers_enriched),
                "ts_generated": pd.Timestamp.now(tz="UTC").isoformat(),
            },
            "regime_distribution": cells_dist,
            "results": out_per_cadence[cadence],
        }
        out_path = OUT_DIR / f"pepe_train_regime_conditioned_{cadence}_2026_05_27.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\n[OK] {cadence} -> {out_path} ({out_path.stat().st_size/1024:.1f} KB)")

    return out_per_cadence


if __name__ == "__main__":
    main()
