"""
PEPE TRAIN mover-day mining at 30m cadence (W2 of MAXX-INST-2026-05-26-NIGHT).

Replicates scripts/mine_pepe_train_mover_days.py methodology at 30m cadence.

Phase 1: TRAIN PEPE 2%+ UP days (from 1d chimera; same set as 1h analysis)
Phase 2: Per mover day, best MA/EMA setup at 30m
Phase 3: 6 exit policies (scaled to 30m)
Phase 4: V51 chimera signature (10 features) + trajectory metrics

Past-only at every layer. TRAIN-only (first 50% of 30m chimera). VAL/OOS/UNSEEN untouched.

Repro:
  canonical_seeds = {bag_seed:131, feat_seed:1131, rng_seed:8041}
  git_sha = 3adfb8e
  chimera_file = data/processed/chimera/30m/pepeusdt_v51_chimera_30m_20260522.parquet
"""
import json
from pathlib import Path
import numpy as np
import polars as pl
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CHIM_1D = ROOT / "data/processed/chimera/1d/pepeusdt_v51_chimera_1d_20260522.parquet"
CHIM_30M = ROOT / "data/processed/chimera/30m/pepeusdt_v51_chimera_30m_20260522.parquet"

OUT_DIR = ROOT / "runs/mover_day"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_SEEDS = {"bag_seed": 131, "feat_seed": 1131, "rng_seed": 8041}
GIT_SHA = "3adfb8e"

TRAIN_FRAC = 0.50

# 30m sweep grids (scaled per brief)
INDICATOR_TYPES = ["SMA", "EMA", "WMA"]
FAST_GRID = [6, 10, 14, 18, 24, 30]      # 3h/5h/7h/9h/12h/15h
SLOW_GRID = [24, 30, 42, 60, 84, 120]    # 12h/15h/21h/30h/42h/60h

# Lookback for cross search: 72h = 144 30m bars
LOOKBACK_BARS = 144

# Exit horizons in 30m bars
EXIT_HORIZONS = {
    "E2_6h": 12,    # 6h = 12 x 30m
    "E3_12h": 24,
    "E4_24h": 48,
    "E5_48h": 96,
}

# Chimera signature window: +/-48 30m bars = +/-24h
CHIM_WINDOW = 48
# Rolling median for elevation check: 120 30m bars (~60h, comparable to 1h's 60-bar/60h)
ROLL_WIN = 120
# Trajectory slope/z-score window: 24 30m bars (12h) — to detect short-term direction at entry
TRAJ_WIN = 24

CHIMERA_FEATURES = [
    "wh_whale_net_usd",
    "wh_whale_trade_count",
    "fund_rate_mean",
    "bs_basis_z30",
    "hbr_eta_buy",
    "bd_imbalance_l1",
    "te_btc_imb",
    "rv_rv_5m",
    "rv_jv_5m",
    "premium_z90",
]


# ---------- Vectorized MA + cross detection ----------

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
    """Vectorized first cross-up. Returns first index i in [max(start,1), end) where
    fast[i-1]<=slow[i-1] and fast[i]>slow[i]. Returns -1 if none.
    """
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


# ---------- Phase 1 ----------

def phase1_mover_days():
    df_1d = pl.read_parquet(CHIM_1D).sort("timestamp")
    n = len(df_1d)
    df_1d = df_1d.with_columns(
        pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("dt")
    )
    train_end_bar = int(n * TRAIN_FRAC)
    df_train = df_1d[:train_end_bar]

    closes = df_train["close"].to_numpy()
    daily_ret = np.zeros(len(closes))
    daily_ret[1:] = closes[1:] / closes[:-1] - 1.0

    highs = df_train["high"].to_numpy()
    hi_move = np.zeros(len(closes))
    hi_move[1:] = highs[1:] / closes[:-1] - 1.0

    df_train = df_train.with_columns([
        pl.Series("daily_return", daily_ret),
        pl.Series("hi_move", hi_move),
    ])
    df_train = df_train.with_columns(
        pl.col("daily_return").shift(-1).alias("next_day_return")
    )

    movers = df_train.filter(pl.col("daily_return") >= 0.02)
    movers_list = []
    for row in movers.iter_rows(named=True):
        movers_list.append({
            "date": str(row["dt"].date()),
            "ts_ms": row["timestamp"],
            "daily_return": float(row["daily_return"]),
            "hi_move": float(row["hi_move"]),
            "volume": float(row["volume"]),
            "next_day_return": float(row["next_day_return"]) if row["next_day_return"] is not None else None,
        })

    rets = df_train["daily_return"].to_numpy()[1:]
    summary = {
        "n_train_days": int(len(df_train)),
        "train_start": str(df_train["dt"][0].date()),
        "train_end": str(df_train["dt"][-1].date()),
        "n_movers_2pct": int(len(movers)),
        "n_movers_5pct": int((rets >= 0.05).sum()),
        "n_movers_10pct": int((rets >= 0.10).sum()),
        "daily_return_p50": float(np.median(rets)),
        "daily_return_p90": float(np.quantile(rets, 0.90)),
        "daily_return_p99": float(np.quantile(rets, 0.99)),
        "daily_return_max": float(rets.max()),
        "daily_return_min": float(rets.min()),
    }
    print(f"[Phase 1] TRAIN: {summary['train_start']} -> {summary['train_end']} ({summary['n_train_days']} days)")
    print(f"[Phase 1] Movers >=2%: {summary['n_movers_2pct']}, >=5%: {summary['n_movers_5pct']}, >=10%: {summary['n_movers_10pct']}")
    return summary, movers_list, df_train


# ---------- Phase 2 ----------

def phase2_best_setup_per_day(movers_list, df_train_1d):
    df_30m = pl.read_parquet(CHIM_30M).sort("timestamp")
    df_30m = df_30m.with_columns(
        pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("dt")
    )

    train_end_ts = int(df_train_1d["timestamp"][-1])
    # TRAIN slice: bars whose ts <= last 1d bar ts + 24h
    df_train_30m = df_30m.filter(pl.col("timestamp") <= train_end_ts + 24 * 3600 * 1000)

    ts_30m = df_train_30m["timestamp"].to_numpy()
    closes_30m = df_train_30m["close"].to_numpy()
    opens_30m = df_train_30m["open"].to_numpy()
    highs_30m = df_train_30m["high"].to_numpy()

    n_30m = len(df_train_30m)
    print(f"[Phase 2] TRAIN 30m bars: {n_30m}")

    # Sanity check vs total bars
    n_total_30m = len(df_30m)
    canonical_50pct = int(n_total_30m * TRAIN_FRAC)
    print(f"[Phase 2] Total 30m bars: {n_total_30m}; canonical 50%: {canonical_50pct}; using {n_30m} (by ts cutoff)")

    # Precompute MAs once
    ma_cache = {}
    for kind in INDICATOR_TYPES:
        for w in sorted(set(FAST_GRID + SLOW_GRID)):
            ma_cache[(kind, w)] = compute_ma(closes_30m, kind, w)
    print(f"[Phase 2] MA cache: {len(ma_cache)} series")

    per_day_results = []
    for mi, m in enumerate(movers_list):
        day_start_ts = m["ts_ms"]
        day_close_idx = int(np.searchsorted(ts_30m, day_start_ts, side="right") - 1)
        if day_close_idx < 0:
            per_day_results.append({
                "date": m["date"], "daily_return": m["daily_return"],
                "best_setup": None, "captured_move_pct": None,
                "available_move_pct": None, "capture_rate": None,
                "entry_offset_bars": None, "entry_offset_hours": None,
                "entry_bar_idx": None, "day_close_idx": None,
                "status": "NO_BAR",
            })
            continue
        window_start = max(0, day_close_idx - (LOOKBACK_BARS - 1))
        signal_search_start = window_start + 1
        signal_search_end = day_close_idx + 1

        daily_close_price = closes_30m[day_close_idx]

        # Available move (best entry-to-close PnL in window)
        avail_returns = []
        for i in range(window_start, day_close_idx):
            if i + 1 <= day_close_idx:
                ent_p = opens_30m[i + 1]
                if ent_p > 0:
                    avail_returns.append(daily_close_price / ent_p - 1.0)
        available_move = max(avail_returns) if avail_returns else 0.0

        best = {
            "captured_move_pct": -np.inf,
            "setup": None,
            "signal_bar_idx": -1,
            "entry_bar_idx": -1,
            "entry_price": None,
            "entry_offset_bars": None,
        }

        for kind in INDICATOR_TYPES:
            for fast in FAST_GRID:
                for slow in SLOW_GRID:
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
                    entry_p = opens_30m[entry_idx]
                    if entry_p <= 0:
                        continue
                    captured = daily_close_price / entry_p - 1.0
                    if captured > best["captured_move_pct"]:
                        offset_bars = day_close_idx - entry_idx
                        best = {
                            "captured_move_pct": float(captured),
                            "setup": {"kind": kind, "fast": fast, "slow": slow},
                            "signal_bar_idx": int(cross_idx),
                            "entry_bar_idx": int(entry_idx),
                            "entry_price": float(entry_p),
                            "entry_offset_bars": int(offset_bars),
                        }

        if best["setup"] is None:
            per_day_results.append({
                "date": m["date"], "daily_return": m["daily_return"],
                "best_setup": None, "captured_move_pct": None,
                "available_move_pct": float(available_move), "capture_rate": None,
                "entry_offset_bars": None, "entry_offset_hours": None,
                "entry_bar_idx": None, "day_close_idx": int(day_close_idx),
                "status": "NO_SETUP",
            })
        else:
            cr = best["captured_move_pct"] / available_move if available_move > 1e-9 else 0.0
            per_day_results.append({
                "date": m["date"], "daily_return": m["daily_return"],
                "best_setup": best["setup"],
                "captured_move_pct": best["captured_move_pct"],
                "available_move_pct": float(available_move),
                "capture_rate": float(cr),
                "entry_offset_bars": best["entry_offset_bars"],
                "entry_offset_hours": float(best["entry_offset_bars"]) * 0.5,  # 30m bars -> hours approx
                "entry_bar_idx": best["entry_bar_idx"],
                "day_close_idx": int(day_close_idx),
                "status": "OK",
            })

    n_ok = sum(1 for r in per_day_results if r["status"] == "OK")
    n_no_setup = sum(1 for r in per_day_results if r["status"] == "NO_SETUP")
    print(f"[Phase 2] {n_ok}/{len(per_day_results)} mover days had clean setup ({n_no_setup} NO_SETUP)")
    return per_day_results, df_train_30m, ma_cache


# ---------- Phase 3: 6 exits on best-per-day entries ----------

def phase3_exits(per_day_results, df_train_30m, ma_cache):
    closes_30m = df_train_30m["close"].to_numpy()
    opens_30m = df_train_30m["open"].to_numpy()
    highs_30m = df_train_30m["high"].to_numpy()
    n_30m = len(df_train_30m)

    EXITS = ["E1_opp_cross", "E2_6h", "E3_12h", "E4_24h", "E5_48h", "E6_mfe_trail50"]
    results = {e: [] for e in EXITS}
    # 14-day cap = 14*48 = 672 30m bars
    CAP = 14 * 48

    for r in per_day_results:
        if r["status"] != "OK":
            for e in EXITS:
                results[e].append(None)
            continue
        kind = r["best_setup"]["kind"]
        fast = r["best_setup"]["fast"]
        slow = r["best_setup"]["slow"]
        fast_ma = ma_cache[(kind, fast)]
        slow_ma = ma_cache[(kind, slow)]
        entry_idx = r["entry_bar_idx"]
        entry_p = opens_30m[entry_idx]

        # E1: opposite cross
        e1_exit_idx = None
        for i in range(entry_idx, min(entry_idx + CAP, n_30m)):
            if (
                not np.isnan(fast_ma[i]) and not np.isnan(slow_ma[i])
                and fast_ma[i] <= slow_ma[i]
            ):
                e1_exit_idx = i
                break
        if e1_exit_idx is None or e1_exit_idx + 1 >= n_30m:
            results["E1_opp_cross"].append(None)
        else:
            exit_p = opens_30m[e1_exit_idx + 1]
            results["E1_opp_cross"].append(float(exit_p / entry_p - 1.0))

        # E2-E5 fixed holds
        for key, k_bars in EXIT_HORIZONS.items():
            ex_idx = entry_idx + k_bars
            if ex_idx >= n_30m:
                results[key].append(None)
            else:
                results[key].append(float(opens_30m[ex_idx] / entry_p - 1.0))

        # E6: MFE-trail50 (lock 50% MFE once peak > +3%)
        peak_p = entry_p
        e6_exit = None
        end_walk = min(entry_idx + CAP, n_30m - 1)
        for i in range(entry_idx + 1, end_walk + 1):
            peak_p = max(peak_p, highs_30m[i])
            peak_ret = peak_p / entry_p - 1.0
            if peak_ret >= 0.03:
                lock_p = entry_p * (1.0 + peak_ret * 0.5)
                if closes_30m[i] <= lock_p:
                    if i + 1 < n_30m:
                        e6_exit = float(opens_30m[i + 1] / entry_p - 1.0)
                    else:
                        e6_exit = float(closes_30m[i] / entry_p - 1.0)
                    break
        if e6_exit is None:
            e6_exit = float(closes_30m[end_walk] / entry_p - 1.0)
        results["E6_mfe_trail50"].append(e6_exit)

    summary = {}
    for e in EXITS:
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
        print(f"[Phase 3] {e}: n={summary[e]['n']:3d} med={summary[e]['median_pnl']:+.4f} "
              f"mean={summary[e]['mean_pnl']:+.4f} win={summary[e]['win_rate']:.2%}")
    return results, summary


# ---------- Phase 4: V51 chimera signature with TRAJECTORY ----------

def phase4_chimera_signature(per_day_results, df_train_30m):
    df_pd = df_train_30m.to_pandas()
    feat_arr = {f: df_pd[f].to_numpy() for f in CHIMERA_FEATURES if f in df_pd.columns}
    missing = [f for f in CHIMERA_FEATURES if f not in df_pd.columns]
    if missing:
        print(f"[Phase 4] WARNING missing cols: {missing}")

    # past-only rolling median for elevation check (ROLL_WIN=120 ~ 60h)
    rolling_meds = {}
    rolling_means = {}
    rolling_stds = {}
    for f, arr in feat_arr.items():
        s = pd.Series(arr)
        # shift(1) ensures past-only at entry bar i
        rolling_meds[f] = s.shift(1).rolling(ROLL_WIN, min_periods=40).median().to_numpy()
        # For z-score / delta trajectory we use TRAJ_WIN
        rolling_means[f] = s.shift(1).rolling(TRAJ_WIN, min_periods=8).mean().to_numpy()
        rolling_stds[f] = s.shift(1).rolling(TRAJ_WIN, min_periods=8).std().to_numpy()

    per_day_signatures = []
    elevated_counts = {f: 0 for f in feat_arr}
    n_eval = 0

    for r in per_day_results:
        if r["status"] != "OK":
            per_day_signatures.append(None)
            continue
        entry_idx = r["entry_bar_idx"]
        # ±CHIM_WINDOW (48) bars around entry = ±24h
        w_start = max(0, entry_idx - CHIM_WINDOW)
        w_end = min(len(df_train_30m), entry_idx + CHIM_WINDOW + 1)
        # Trajectory window: TRAJ_WIN bars BEFORE entry (12h)
        t_start = max(0, entry_idx - TRAJ_WIN)

        sig = {}
        for f, arr in feat_arr.items():
            entry_val = arr[entry_idx]
            roll_med = rolling_meds[f][entry_idx]
            elevated = False
            if not np.isnan(roll_med) and not np.isnan(entry_val):
                elevated = bool(entry_val > roll_med)

            window_vals = arr[w_start:w_end]
            pre_vals = arr[w_start:entry_idx]
            post_vals = arr[entry_idx:w_end]

            # Trajectory metrics (12h pre-entry)
            traj_vals = arr[t_start:entry_idx]
            # slope: linear regression slope per bar (past-only, ends at entry-1)
            slope = None
            if len(traj_vals) >= 4 and not np.isnan(traj_vals).all():
                mask = ~np.isnan(traj_vals)
                if mask.sum() >= 4:
                    xs = np.arange(len(traj_vals))[mask].astype(float)
                    ys = traj_vals[mask].astype(float)
                    # robust slope = polyfit
                    try:
                        slope_v = np.polyfit(xs, ys, 1)[0]
                        slope = float(slope_v)
                    except Exception:
                        slope = None

            # z-score: entry_val vs past-only rolling mean/std (TRAJ_WIN)
            zscore = None
            rm = rolling_means[f][entry_idx]
            rs = rolling_stds[f][entry_idx]
            if not np.isnan(entry_val) and not np.isnan(rm) and not np.isnan(rs) and rs > 1e-12:
                zscore = float((entry_val - rm) / rs)

            # Delta: entry_val minus value 12h ago (TRAJ_WIN bars)
            delta = None
            past_idx = entry_idx - TRAJ_WIN
            if past_idx >= 0 and not np.isnan(arr[past_idx]) and not np.isnan(entry_val):
                delta = float(entry_val - arr[past_idx])

            pre_mean = float(np.nanmean(pre_vals)) if len(pre_vals) > 0 else None
            post_mean = float(np.nanmean(post_vals)) if len(post_vals) > 0 else None

            sig[f] = {
                "entry_val": float(entry_val) if not np.isnan(entry_val) else None,
                "window_mean": float(np.nanmean(window_vals)) if len(window_vals) > 0 else None,
                "window_median": float(np.nanmedian(window_vals)) if len(window_vals) > 0 else None,
                "pre_mean": pre_mean,
                "post_mean": post_mean,
                "direction": (
                    "increasing" if (pre_mean is not None and post_mean is not None and post_mean > pre_mean)
                    else "decreasing" if (pre_mean is not None and post_mean is not None and post_mean < pre_mean)
                    else "flat"
                ),
                "elevated_vs_60h_median": elevated,
                "rolling_60h_median": float(roll_med) if not np.isnan(roll_med) else None,
                # Trajectory metrics (12h slope/z/delta)
                "trajectory_12h_slope": slope,
                "trajectory_12h_zscore": zscore,
                "trajectory_12h_delta": delta,
            }
            if elevated:
                elevated_counts[f] += 1
        n_eval += 1
        per_day_signatures.append(sig)

    elevation_pct = {f: (elevated_counts[f] / n_eval if n_eval > 0 else 0.0) for f in feat_arr}
    ranked = sorted(elevation_pct.items(), key=lambda kv: -kv[1])
    print(f"[Phase 4] n mover-days for signature: {n_eval}")
    for f, pct in ranked:
        print(f"[Phase 4]   {f:32s}: elevated in {pct:.1%}")

    # Correlations: each (elevated 0/1) and each trajectory metric vs capture_rate
    correlations_elev = {}
    correlations_slope = {}
    correlations_z = {}
    correlations_delta = {}
    for f in feat_arr:
        x_elev, x_slope, x_z, x_delta, y = [], [], [], [], []
        for r, sig in zip(per_day_results, per_day_signatures):
            if r["status"] != "OK" or sig is None:
                continue
            cr = r["capture_rate"]
            if cr is None:
                continue
            ev = sig[f]["elevated_vs_60h_median"]
            sl = sig[f]["trajectory_12h_slope"]
            zs = sig[f]["trajectory_12h_zscore"]
            dl = sig[f]["trajectory_12h_delta"]
            y.append(cr)
            x_elev.append(1.0 if ev else 0.0)
            x_slope.append(sl if sl is not None else np.nan)
            x_z.append(zs if zs is not None else np.nan)
            x_delta.append(dl if dl is not None else np.nan)
        y_arr = np.array(y)

        def safe_corr(xs):
            x_arr = np.array(xs, dtype=float)
            mask = ~np.isnan(x_arr) & ~np.isnan(y_arr)
            if mask.sum() < 5 or np.std(x_arr[mask]) < 1e-12 or np.std(y_arr[mask]) < 1e-12:
                return None
            return float(np.corrcoef(x_arr[mask], y_arr[mask])[0, 1])

        correlations_elev[f] = safe_corr(x_elev)
        correlations_slope[f] = safe_corr(x_slope)
        correlations_z[f] = safe_corr(x_z)
        correlations_delta[f] = safe_corr(x_delta)

    return per_day_signatures, elevation_pct, {
        "elevated_vs_capture": correlations_elev,
        "slope12h_vs_capture": correlations_slope,
        "zscore12h_vs_capture": correlations_z,
        "delta12h_vs_capture": correlations_delta,
    }


# ---------- Main ----------

def main():
    print("=" * 70)
    print("PEPE TRAIN MOVER-DAY MINING @ 30m — W2 MAXX-INST-2026-05-26-NIGHT")
    print("=" * 70)

    p1_summary, movers_list, df_train_1d = phase1_mover_days()
    per_day_results, df_train_30m, ma_cache = phase2_best_setup_per_day(movers_list, df_train_1d)
    exit_results, exit_summary = phase3_exits(per_day_results, df_train_30m, ma_cache)
    per_day_signatures, elevation_pct, correlations = phase4_chimera_signature(per_day_results, df_train_30m)

    output = {
        "_meta": {
            "git_sha": GIT_SHA,
            "cadence": "30m",
            "chimera_file": str(CHIM_30M.relative_to(ROOT)).replace("\\", "/"),
            "canonical_seeds": CANONICAL_SEEDS,
            "train_only": True,
            "n_train_30m_bars": int(len(df_train_30m)),
            "lookback_bars": LOOKBACK_BARS,
            "fast_grid": FAST_GRID,
            "slow_grid": SLOW_GRID,
            "indicator_types": INDICATOR_TYPES,
            "exit_horizons_30m_bars": EXIT_HORIZONS,
            "chim_window_bars": CHIM_WINDOW,
            "roll_win_bars": ROLL_WIN,
            "traj_win_bars": TRAJ_WIN,
        },
        "phase1": p1_summary,
        "movers": movers_list,
        "phase2_per_day": per_day_results,
        "phase3_exits": {
            "summary": exit_summary,
            "per_day": exit_results,
        },
        "phase4_signature": {
            "elevation_pct": elevation_pct,
            "correlations": correlations,
            "per_day_signatures": per_day_signatures,
        },
    }

    out_path = OUT_DIR / "pepe_train_mover_day_30m_2026_05_27.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n[OK] Output -> {out_path}")
    print(f"[OK] Size: {out_path.stat().st_size / 1024:.1f} KB")
    return output


if __name__ == "__main__":
    main()
