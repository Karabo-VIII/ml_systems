"""
PEPE TRAIN mover-day mining at 15m cadence (W1 of MAXX-INST-2026-05-26-NIGHT).

Brief: docs/dossiers/PEPE_TRAIN_MOVER_DAY_PATTERN_15m_2026_05_27.md
Protocol: docs/SELECTION_BIAS_PROTOCOL_2026_05_27.md (TRAIN-ONLY discovery)

Replicates 1h analysis at 15m cadence with appropriately scaled grid:
- fast in {12,20,28,36,48,60} (15m bars = 3h..15h windows)
- slow in {48,60,84,120,168,240} (15m bars = 12h..60h windows)
- 288-bar (3-day) per-mover-day window
- 6 exit policies scaled (E2=6h=24bars, E3=12h=48, E4=24h=96, E5=48h=192)

Past-only at every layer. UNSEEN/VAL/OOS untouched.

Repro:
  canonical_seeds = {bag_seed:130, feat_seed:1130, rng_seed:8040}
  git_sha = 3adfb8e
  chimera_mtime_15m = 2026-05-22
"""
import json
import os
import sys
from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CHIM_1D = ROOT / "data/processed/chimera/1d/pepeusdt_v51_chimera_1d_20260522.parquet"
CHIM_15M = ROOT / "data/processed/chimera/15m/pepeusdt_v51_chimera_15m_20260522.parquet"

OUT_DIR = ROOT / "runs/mover_day"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_SEEDS = {"bag_seed": 130, "feat_seed": 1130, "rng_seed": 8040}
GIT_SHA = "3adfb8e"
CADENCE = "15m"
BARS_PER_DAY = 96  # 15m × 96 = 24h
BARS_PER_HOUR = 4

# Cross-version split contract (CLAUDE.md canonical 50/20/20/10)
TRAIN_FRAC = 0.50

# ---------- Phase 1: TRAIN mover-day inventory (from 1d chimera) ----------

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
    print(f"[Phase 1] TRAIN window: {summary['train_start']} -> {summary['train_end']} ({summary['n_train_days']} days)")
    print(f"[Phase 1] Movers >=2%: {summary['n_movers_2pct']}, >=5%: {summary['n_movers_5pct']}, >=10%: {summary['n_movers_10pct']}")
    return summary, movers_list, df_train


# ---------- Phase 2: best MA/EMA setup per mover day at 15m ----------

INDICATOR_TYPES = ["SMA", "EMA", "WMA"]
FAST_GRID = [12, 20, 28, 36, 48, 60]  # 15m bars (3h..15h)
SLOW_GRID = [48, 60, 84, 120, 168, 240]  # 15m bars (12h..60h)


def compute_ma(prices: np.ndarray, kind: str, window: int) -> np.ndarray:
    """Pure past-only MA. Returns NaN until window-1 bars accumulated."""
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
            # Vectorized via convolve? Use np.convolve for speed
            # weights need to be reversed for np.convolve
            kern = w[::-1] / w_sum
            conv = np.convolve(prices, kern, mode="valid")
            out[window - 1:] = conv
    return out


def find_first_cross_up(fast_ma: np.ndarray, slow_ma: np.ndarray, start: int, end: int):
    """First cross-up index in [start, end). Returns -1 if no cross."""
    # Vectorize: pre-compute mask
    a = fast_ma
    b = slow_ma
    if start < 1:
        start = 1
    if end <= start:
        return -1
    # Slice safely
    seg_a_prev = a[start - 1:end - 1]
    seg_b_prev = b[start - 1:end - 1]
    seg_a_now = a[start:end]
    seg_b_now = b[start:end]
    mask = (
        ~np.isnan(seg_a_prev) & ~np.isnan(seg_b_prev) &
        ~np.isnan(seg_a_now) & ~np.isnan(seg_b_now) &
        (seg_a_prev <= seg_b_prev) & (seg_a_now > seg_b_now)
    )
    idx_rel = np.argmax(mask) if mask.any() else -1
    if idx_rel == -1:
        return -1
    return int(start + idx_rel)


def phase2_best_setup_per_day(movers_list, df_train_1d):
    df_15m = pl.read_parquet(CHIM_15M).sort("timestamp")
    df_15m = df_15m.with_columns(
        pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("dt")
    )
    train_end_ts = int(df_train_1d["timestamp"][-1])
    # include 15m bars up to end of last train day (+24h)
    df_train_15m = df_15m.filter(pl.col("timestamp") <= train_end_ts + 24 * 3600 * 1000)

    ts_15m = df_train_15m["timestamp"].to_numpy()
    closes_15m = df_train_15m["close"].to_numpy()
    opens_15m = df_train_15m["open"].to_numpy()
    highs_15m = df_train_15m["high"].to_numpy()

    n_15m = len(df_train_15m)
    print(f"[Phase 2] TRAIN 15m bars: {n_15m}")

    # Precompute all MAs once
    ma_cache = {}
    for kind in INDICATOR_TYPES:
        for w in sorted(set(FAST_GRID + SLOW_GRID)):
            ma_cache[(kind, w)] = compute_ma(closes_15m, kind, w)
    print(f"[Phase 2] MA cache built ({len(ma_cache)} series)")

    per_day_results = []

    # Per-day window: 3 days = 288 × 15m bars
    # Look at [d-2 days, d] window: 192 bars BEFORE daily close, find first cross-up
    LOOKBACK_BARS = 2 * BARS_PER_DAY  # 192 bars (2 days of pre-cross signal search)

    for m in movers_list:
        day_start_ts = m["ts_ms"]
        day_close_idx = np.searchsorted(ts_15m, day_start_ts, side="right") - 1
        if day_close_idx < 0:
            continue
        window_start = max(0, day_close_idx - (LOOKBACK_BARS - 1))
        signal_search_start = window_start + 1
        signal_search_end = day_close_idx + 1

        daily_close_price = closes_15m[day_close_idx]

        # Available move
        ent_p_window = opens_15m[window_start + 1:day_close_idx + 1]
        valid_mask = ent_p_window > 0
        if valid_mask.any():
            avail_returns = daily_close_price / ent_p_window[valid_mask] - 1.0
            available_move = float(avail_returns.max())
        else:
            available_move = 0.0

        best = {
            "captured_move_pct": -np.inf,
            "setup": None,
            "signal_bar_idx": -1,
            "entry_bar_idx": -1,
            "entry_price": None,
            "entry_offset_bars": None,
            "entry_offset_hours": None,
        }

        for kind in INDICATOR_TYPES:
            for fast in FAST_GRID:
                for slow in SLOW_GRID:
                    if fast >= slow:
                        continue
                    fast_ma = ma_cache[(kind, fast)]
                    slow_ma = ma_cache[(kind, slow)]
                    cross_idx = find_first_cross_up(
                        fast_ma, slow_ma, signal_search_start, signal_search_end
                    )
                    if cross_idx == -1:
                        continue
                    if cross_idx + 1 > day_close_idx:
                        continue
                    entry_idx = cross_idx + 1
                    entry_p = opens_15m[entry_idx]
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
                            "entry_offset_hours": float(offset_bars / BARS_PER_HOUR),
                        }

        if best["setup"] is None:
            per_day_results.append({
                "date": m["date"],
                "daily_return": m["daily_return"],
                "best_setup": None,
                "captured_move_pct": None,
                "available_move_pct": float(available_move),
                "capture_rate": None,
                "entry_offset_bars": None,
                "entry_offset_hours": None,
                "entry_bar_idx": None,
                "day_close_idx": int(day_close_idx),
                "status": "NO_SETUP",
            })
        else:
            cr = best["captured_move_pct"] / available_move if available_move > 1e-9 else 0.0
            per_day_results.append({
                "date": m["date"],
                "daily_return": m["daily_return"],
                "best_setup": best["setup"],
                "captured_move_pct": best["captured_move_pct"],
                "available_move_pct": float(available_move),
                "capture_rate": float(cr),
                "entry_offset_bars": best["entry_offset_bars"],
                "entry_offset_hours": best["entry_offset_hours"],
                "entry_bar_idx": best["entry_bar_idx"],
                "day_close_idx": int(day_close_idx),
                "status": "OK",
            })

    n_ok = sum(1 for r in per_day_results if r["status"] == "OK")
    print(f"[Phase 2] {n_ok}/{len(per_day_results)} mover days had clean setup")
    return per_day_results, df_train_15m, ma_cache


# ---------- Phase 3: exit policies (scaled to 15m) ----------

def phase3_exits(per_day_results, df_train_15m, ma_cache):
    closes_15m = df_train_15m["close"].to_numpy()
    opens_15m = df_train_15m["open"].to_numpy()
    highs_15m = df_train_15m["high"].to_numpy()
    n_15m = len(df_train_15m)

    # Exit horizons in 15m bars (scaled from 1h)
    # E2 6h = 24 bars, E3 12h = 48, E4 24h = 96, E5 48h = 192
    EXITS = ["E1_opp_cross", "E2_6h", "E3_12h", "E4_24h", "E5_48h", "E6_mfe_trail50"]
    EXIT_HOLD_BARS = {"E2_6h": 24, "E3_12h": 48, "E4_24h": 96, "E5_48h": 192}
    CAP_BARS = 14 * BARS_PER_DAY  # 14-day cap (= 1344 bars)

    results = {e: [] for e in EXITS}

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
        entry_p = opens_15m[entry_idx]

        # E1: opposite cross
        e1_exit_idx = None
        end_search = min(entry_idx + CAP_BARS, n_15m)
        for i in range(entry_idx, end_search):
            if (
                not np.isnan(fast_ma[i])
                and not np.isnan(slow_ma[i])
                and fast_ma[i] <= slow_ma[i]
            ):
                e1_exit_idx = i
                break
        if e1_exit_idx is None or e1_exit_idx + 1 >= n_15m:
            results["E1_opp_cross"].append(None)
        else:
            exit_p = opens_15m[e1_exit_idx + 1] if e1_exit_idx + 1 < n_15m else closes_15m[e1_exit_idx]
            results["E1_opp_cross"].append(float(exit_p / entry_p - 1.0))

        # E2-E5: fixed holds
        for key, hold_bars in EXIT_HOLD_BARS.items():
            ex_idx = entry_idx + hold_bars
            if ex_idx >= n_15m:
                results[key].append(None)
            else:
                results[key].append(float(opens_15m[ex_idx] / entry_p - 1.0))

        # E6: MFE-trail50
        peak_p = entry_p
        e6_exit = None
        end_walk = min(entry_idx + CAP_BARS, n_15m - 1)
        for i in range(entry_idx + 1, end_walk + 1):
            peak_p = max(peak_p, highs_15m[i])
            peak_ret = peak_p / entry_p - 1.0
            if peak_ret >= 0.03:
                lock_p = entry_p * (1.0 + peak_ret * 0.5)
                if closes_15m[i] <= lock_p:
                    if i + 1 < n_15m:
                        e6_exit = float(opens_15m[i + 1] / entry_p - 1.0)
                    else:
                        e6_exit = float(closes_15m[i] / entry_p - 1.0)
                    break
        if e6_exit is None:
            e6_exit = float(closes_15m[end_walk] / entry_p - 1.0)
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
        print(
            f"[Phase 3] {e}: n={summary[e]['n']:3d} med={summary[e]['median_pnl']:+.4f} "
            f"mean={summary[e]['mean_pnl']:+.4f} win={summary[e]['win_rate']:.2%}"
        )
    return results, summary


# ---------- Phase 4: V51 chimera signature at best entry (with z-score + delta) ----------

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


def phase4_chimera_signature(per_day_results, df_train_15m):
    df_pd = df_train_15m.to_pandas()
    feat_arr = {f: df_pd[f].to_numpy() for f in CHIMERA_FEATURES if f in df_pd.columns}
    missing = [f for f in CHIMERA_FEATURES if f not in df_pd.columns]
    if missing:
        print(f"[Phase 4] WARNING missing cols: {missing}")

    # Rolling 60-bar median for elevation check (60 15m bars = 15h, slightly different scale than 1h)
    rolling_meds = {}
    # For 15m, 60-bar is ~15h. Use it for consistency with 1h's 60-bar (60h).
    # Optional: also support a 240-bar (60h) for stronger filtering.
    for f, arr in feat_arr.items():
        s = pd.Series(arr)
        rolling_meds[f] = s.shift(1).rolling(60, min_periods=20).median().to_numpy()

    # Trajectory metrics: 12h window = 48 15m bars
    # z-score over past 48 bars (lagged 1 bar), and 12h-delta = feat[t] - feat[t-48]
    Z_WIN = 48  # 12h
    DELTA_LAG = 48  # 12h
    rolling_z = {}
    delta_12h = {}
    for f, arr in feat_arr.items():
        s = pd.Series(arr)
        s_lag = s.shift(1)
        roll_mean = s_lag.rolling(Z_WIN, min_periods=12).mean()
        roll_std = s_lag.rolling(Z_WIN, min_periods=12).std()
        # z = (current - past_mean) / past_std
        rolling_z[f] = ((s - roll_mean) / roll_std.replace(0, np.nan)).to_numpy()
        delta_12h[f] = (s - s.shift(DELTA_LAG)).to_numpy()

    per_day_signatures = []
    elevated_counts = {f: 0 for f in feat_arr}
    n_eval = 0

    for r in per_day_results:
        if r["status"] != "OK":
            per_day_signatures.append(None)
            continue
        entry_idx = r["entry_bar_idx"]
        # Window: ±48 bars (12h) around entry
        w_start = max(0, entry_idx - 48)
        w_end = min(len(df_train_15m), entry_idx + 49)
        sig = {}
        for f, arr in feat_arr.items():
            window_vals = arr[w_start:w_end]
            entry_val = arr[entry_idx]
            roll_med = rolling_meds[f][entry_idx] if not np.isnan(rolling_meds[f][entry_idx]) else None
            elevated = False
            if roll_med is not None and not np.isnan(entry_val):
                elevated = bool(entry_val > roll_med)
            pre_vals = arr[w_start:entry_idx]
            post_vals = arr[entry_idx:w_end]
            pre_mean = float(np.nanmean(pre_vals)) if len(pre_vals) > 0 else None
            post_mean = float(np.nanmean(post_vals)) if len(post_vals) > 0 else None
            z12 = rolling_z[f][entry_idx]
            d12 = delta_12h[f][entry_idx]
            sig[f] = {
                "entry_val": float(entry_val) if not np.isnan(entry_val) else None,
                "window_mean": float(np.nanmean(window_vals)) if len(window_vals) > 0 else None,
                "window_median": float(np.nanmedian(window_vals)) if len(window_vals) > 0 else None,
                "pre_mean_12h": pre_mean,
                "post_mean_12h": post_mean,
                "direction": (
                    "increasing" if (pre_mean is not None and post_mean is not None and post_mean > pre_mean)
                    else "decreasing" if (pre_mean is not None and post_mean is not None and post_mean < pre_mean)
                    else "flat"
                ),
                "elevated_vs_60bar_median": elevated,
                "rolling_60bar_median": float(roll_med) if roll_med is not None else None,
                "z_12h": float(z12) if not np.isnan(z12) else None,
                "delta_12h": float(d12) if not np.isnan(d12) else None,
            }
            if elevated:
                elevated_counts[f] += 1
        n_eval += 1
        per_day_signatures.append(sig)

    elevation_pct = {f: (elevated_counts[f] / n_eval if n_eval > 0 else 0.0) for f in feat_arr}
    ranked = sorted(elevation_pct.items(), key=lambda kv: -kv[1])
    print(f"[Phase 4] n mover-days evaluated for signature: {n_eval}")
    for f, pct in ranked:
        print(f"[Phase 4]   {f:32s}: elevated in {pct:.1%}")

    # Correlations: elevated bool, z_12h, delta_12h vs capture_rate
    correlations_elev = {}
    correlations_z = {}
    correlations_delta = {}
    for f in feat_arr:
        x_e, x_z, x_d, y = [], [], [], []
        for r, sig in zip(per_day_results, per_day_signatures):
            if r["status"] != "OK" or sig is None:
                continue
            cr = r["capture_rate"]
            if cr is None:
                continue
            ev = sig[f]["elevated_vs_60bar_median"]
            z = sig[f]["z_12h"]
            d = sig[f]["delta_12h"]
            if ev is not None:
                x_e.append(1.0 if ev else 0.0)
            if z is not None:
                x_z.append(z)
            if d is not None:
                x_d.append(d)
            y.append(cr)
        # Note: lengths may vary if some had None; align via parallel arrays
        def safe_corr(xs, ys):
            xs = np.array(xs, dtype=float)
            ys = np.array(ys, dtype=float)
            n = min(len(xs), len(ys))
            xs, ys = xs[:n], ys[:n]
            if n < 5 or np.std(xs) == 0 or np.std(ys) == 0:
                return None
            return float(np.corrcoef(xs, ys)[0, 1])
        correlations_elev[f] = safe_corr(x_e, y[:len(x_e)])
        correlations_z[f] = safe_corr(x_z, y[:len(x_z)])
        correlations_delta[f] = safe_corr(x_d, y[:len(x_d)])

    return per_day_signatures, elevation_pct, correlations_elev, correlations_z, correlations_delta


# ---------- Main ----------

def main():
    print("=" * 70)
    print("PEPE TRAIN MOVER-DAY MINING AT 15m — W1 MAXX-INST-2026-05-26-NIGHT")
    print("=" * 70)

    p1_summary, movers_list, df_train_1d = phase1_mover_days()
    per_day_results, df_train_15m, ma_cache = phase2_best_setup_per_day(movers_list, df_train_1d)
    exit_results, exit_summary = phase3_exits(per_day_results, df_train_15m, ma_cache)
    sigs, elev_pct, corr_e, corr_z, corr_d = phase4_chimera_signature(per_day_results, df_train_15m)

    output = {
        "_meta": {
            "git_sha": GIT_SHA,
            "cadence": CADENCE,
            "chimera_file_15m": str(CHIM_15M.name),
            "chimera_mtime_15m": "2026-05-22",
            "canonical_seeds": CANONICAL_SEEDS,
            "train_only": True,
            "fast_grid": FAST_GRID,
            "slow_grid": SLOW_GRID,
            "indicator_types": INDICATOR_TYPES,
            "ts_generated": "2026-05-27T03:30+02:00",
        },
        "phase1": p1_summary,
        "movers": movers_list,
        "phase2_per_day": per_day_results,
        "phase3_exits": {
            "summary": exit_summary,
            "per_day": exit_results,
        },
        "phase4_signature": {
            "elevation_pct": elev_pct,
            "correlation_elevated_vs_capture_rate": corr_e,
            "correlation_z12h_vs_capture_rate": corr_z,
            "correlation_delta12h_vs_capture_rate": corr_d,
            "per_day_signatures": sigs,
        },
    }
    out_path = OUT_DIR / "pepe_train_mover_day_15m_2026_05_27.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n[OK] Output -> {out_path}")
    print(f"[OK] Size: {out_path.stat().st_size / 1024:.1f} KB")
    return output


if __name__ == "__main__":
    main()
