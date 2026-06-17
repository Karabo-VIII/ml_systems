"""
PEPE TRAIN mover-day mining (W1 of MAXX-INST-2026-05-26-NIGHT).

Brief: docs/dossiers/PEPE_TRAIN_MOVER_DAY_PATTERN_2026_05_27.md
Protocol: docs/SELECTION_BIAS_PROTOCOL_2026_05_27.md (TRAIN-ONLY discovery)

Phase 1: Identify TRAIN PEPE 2%+ UP days
Phase 2: For each mover day, find best MA/EMA setup that captures the move
Phase 3: Test 6 exit policies on those setups
Phase 4: Pull V51 chimera signature at best entry

Past-only at every layer. UNSEEN/VAL/OOS untouched.

Repro:
  canonical_seeds = {bag_seed:110, feat_seed:1110, rng_seed:8020}
  git_sha = 3adfb8e
  chimera_mtime_1h = 2026-05-24 21:35:13
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
CHIM_1H = ROOT / "data/processed/chimera/1h/pepeusdt_v51_chimera_1h_20260522.parquet"

OUT_DIR = ROOT / "runs/mover_day"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_SEEDS = {"bag_seed": 110, "feat_seed": 1110, "rng_seed": 8020}
GIT_SHA = "3adfb8e"

# Cross-version split contract (CLAUDE.md canonical 50/20/20/10)
TRAIN_FRAC = 0.50
PURGE_GAP_BARS = 400

# ---------- Phase 1: TRAIN mover-day inventory ----------

def phase1_mover_days():
    df_1d = pl.read_parquet(CHIM_1D).sort("timestamp")
    n = len(df_1d)
    # Convert ms to datetime
    df_1d = df_1d.with_columns(
        pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("dt")
    )

    train_end_bar = int(n * TRAIN_FRAC)  # exclusive
    df_train = df_1d[:train_end_bar]

    # Daily return (close-to-close)
    closes = df_train["close"].to_numpy()
    daily_ret = np.zeros(len(closes))
    daily_ret[1:] = closes[1:] / closes[:-1] - 1.0

    # Intraday high move vs prev close
    highs = df_train["high"].to_numpy()
    hi_move = np.zeros(len(closes))
    hi_move[1:] = highs[1:] / closes[:-1] - 1.0

    df_train = df_train.with_columns([
        pl.Series("daily_return", daily_ret),
        pl.Series("hi_move", hi_move),
    ])
    # Next-day return
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

    # Summary stats
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
    print(f"[Phase 1] Daily return p99: {summary['daily_return_p99']:.4f}, max: {summary['daily_return_max']:.4f}")

    return summary, movers_list, df_train


# ---------- Phase 2: best MA/EMA setup per mover day ----------

INDICATOR_TYPES = ["SMA", "EMA", "WMA"]
FAST_GRID = [5, 8, 10, 12, 15, 20]
SLOW_GRID = [15, 20, 21, 26, 30, 50, 100]


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
        # Seed with SMA of first `window` bars
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


def find_first_cross_up(fast_ma: np.ndarray, slow_ma: np.ndarray, start: int, end: int):
    """Find first index in [start, end) where fast crosses above slow (fast[i-1]<=slow[i-1] and fast[i]>slow[i]).

    Returns the SIGNAL bar index i (the bar where cross is observed). Entry would be at bar i+1 open.
    Returns -1 if no cross.
    """
    for i in range(max(start, 1), end):
        if (
            not np.isnan(fast_ma[i - 1])
            and not np.isnan(slow_ma[i - 1])
            and not np.isnan(fast_ma[i])
            and not np.isnan(slow_ma[i])
            and fast_ma[i - 1] <= slow_ma[i - 1]
            and fast_ma[i] > slow_ma[i]
        ):
            return i
    return -1


def phase2_best_setup_per_day(movers_list, df_train_1d):
    df_1h = pl.read_parquet(CHIM_1H).sort("timestamp")
    df_1h = df_1h.with_columns(
        pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("dt")
    )

    # TRAIN window in 1h bars matches 1d calendar
    train_end_dt = df_train_1d["dt"][-1]
    train_end_ts = int(df_train_1d["timestamp"][-1])  # last train 1d bar ts
    # include 1h bars up to end of that day (+24h)
    df_train_1h = df_1h.filter(pl.col("timestamp") <= train_end_ts + 24 * 3600 * 1000)

    ts_1h = df_train_1h["timestamp"].to_numpy()
    closes_1h = df_train_1h["close"].to_numpy()
    opens_1h = df_train_1h["open"].to_numpy()
    highs_1h = df_train_1h["high"].to_numpy()

    n_1h = len(df_train_1h)
    print(f"[Phase 2] TRAIN 1h bars: {n_1h}")

    # Precompute all MAs once
    ma_cache = {}
    for kind in INDICATOR_TYPES:
        for w in sorted(set(FAST_GRID + SLOW_GRID)):
            ma_cache[(kind, w)] = compute_ma(closes_1h, kind, w)
    print(f"[Phase 2] MA cache built ({len(ma_cache)} series)")

    per_day_results = []

    for m in movers_list:
        day_start_ts = m["ts_ms"]
        # That 1d bar represents day d (close at day_start_ts is day d's close)
        # Find 1h bar where ts equals daily close moment, or last bar of that calendar day
        day_close_idx = np.searchsorted(ts_1h, day_start_ts, side="right") - 1
        if day_close_idx < 0:
            continue
        # Look at last 72 1h bars BEFORE the daily close (i.e., [day_close_idx-71, day_close_idx])
        window_start = max(0, day_close_idx - 71)
        signal_search_start = window_start + 1  # need 1-bar lookback for cross detection
        signal_search_end = day_close_idx + 1  # cross must be observed by daily-close bar

        daily_close_price = closes_1h[day_close_idx]

        # Available move: best from any pre-day-close entry to close (using next-bar-open as entry price)
        # avail = max over i in [window_start, day_close_idx-1] of (daily_close / opens_1h[i+1] - 1)
        avail_returns = []
        for i in range(window_start, day_close_idx):
            if i + 1 <= day_close_idx:
                ent_p = opens_1h[i + 1]
                if ent_p > 0:
                    avail_returns.append(daily_close_price / ent_p - 1.0)
        available_move = max(avail_returns) if avail_returns else 0.0

        best = {
            "captured_move_pct": -np.inf,
            "setup": None,
            "signal_bar_idx": -1,
            "entry_bar_idx": -1,
            "entry_price": None,
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
                    # Entry at next-bar open
                    if cross_idx + 1 > day_close_idx:
                        # signal fires on the daily-close bar itself -> no clean entry before close
                        continue
                    entry_idx = cross_idx + 1
                    entry_p = opens_1h[entry_idx]
                    if entry_p <= 0:
                        continue
                    captured = daily_close_price / entry_p - 1.0
                    if captured > best["captured_move_pct"]:
                        offset_hours = day_close_idx - entry_idx  # hours BEFORE daily close
                        best = {
                            "captured_move_pct": float(captured),
                            "setup": {"kind": kind, "fast": fast, "slow": slow},
                            "signal_bar_idx": int(cross_idx),
                            "entry_bar_idx": int(entry_idx),
                            "entry_price": float(entry_p),
                            "entry_offset_hours": int(offset_hours),
                        }

        if best["setup"] is None:
            per_day_results.append({
                "date": m["date"],
                "daily_return": m["daily_return"],
                "best_setup": None,
                "captured_move_pct": None,
                "available_move_pct": float(available_move),
                "capture_rate": None,
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
                "entry_offset_hours": best["entry_offset_hours"],
                "entry_bar_idx": best["entry_bar_idx"],
                "day_close_idx": int(day_close_idx),
                "status": "OK",
            })

    n_ok = sum(1 for r in per_day_results if r["status"] == "OK")
    print(f"[Phase 2] {n_ok}/{len(per_day_results)} mover days had clean setup")

    return per_day_results, df_train_1h, ma_cache


# ---------- Phase 3: exit policies ----------

def phase3_exits(per_day_results, df_train_1h, ma_cache):
    closes_1h = df_train_1h["close"].to_numpy()
    opens_1h = df_train_1h["open"].to_numpy()
    highs_1h = df_train_1h["high"].to_numpy()
    n_1h = len(df_train_1h)

    # Exit horizons in 1h bars
    EXITS = ["E1_opp_cross", "E2_6h", "E3_12h", "E4_24h", "E5_48h", "E6_mfe_trail50"]

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
        entry_p = opens_1h[entry_idx]

        # E1: opposite cross — first bar i >= entry_idx where fast<=slow
        e1_exit_idx = None
        for i in range(entry_idx, min(entry_idx + 14 * 24, n_1h)):  # cap at 14 days
            if (
                not np.isnan(fast_ma[i])
                and not np.isnan(slow_ma[i])
                and fast_ma[i] <= slow_ma[i]
            ):
                e1_exit_idx = i
                break
        if e1_exit_idx is None or e1_exit_idx + 1 >= n_1h:
            results["E1_opp_cross"].append(None)
        else:
            exit_p = opens_1h[e1_exit_idx + 1] if e1_exit_idx + 1 < n_1h else closes_1h[e1_exit_idx]
            results["E1_opp_cross"].append(float(exit_p / entry_p - 1.0))

        # E2-E5: fixed holds
        for k_hours, key in [(6, "E2_6h"), (12, "E3_12h"), (24, "E4_24h"), (48, "E5_48h")]:
            ex_idx = entry_idx + k_hours
            if ex_idx >= n_1h:
                results[key].append(None)
            else:
                results[key].append(float(opens_1h[ex_idx] / entry_p - 1.0))

        # E6: MFE-trail50 — once unrealized > +3%, lock in 50% of peak
        # Walk forward; track peak unrealized = max(highs since entry)/entry - 1
        # If peak >= 0.03 and current close drops below entry*(1 + peak*0.5) → exit at next-bar open
        peak_p = entry_p
        triggered = False
        e6_exit = None
        cap = 14 * 24
        end_walk = min(entry_idx + cap, n_1h - 1)
        for i in range(entry_idx + 1, end_walk + 1):
            peak_p = max(peak_p, highs_1h[i])
            peak_ret = peak_p / entry_p - 1.0
            if peak_ret >= 0.03:
                triggered = True
                lock_p = entry_p * (1.0 + peak_ret * 0.5)
                if closes_1h[i] <= lock_p:
                    # Exit next-bar open
                    if i + 1 < n_1h:
                        e6_exit = float(opens_1h[i + 1] / entry_p - 1.0)
                    else:
                        e6_exit = float(closes_1h[i] / entry_p - 1.0)
                    break
        if e6_exit is None:
            # If never triggered or never reverted, exit at end_walk close
            e6_exit = float(closes_1h[end_walk] / entry_p - 1.0)
        results["E6_mfe_trail50"].append(e6_exit)

    # Summarize per-exit
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


# ---------- Phase 4: V51 chimera signature at best entry ----------

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


def phase4_chimera_signature(per_day_results, df_train_1h):
    df_pd = df_train_1h.to_pandas()
    feat_arr = {f: df_pd[f].to_numpy() for f in CHIMERA_FEATURES if f in df_pd.columns}
    missing = [f for f in CHIMERA_FEATURES if f not in df_pd.columns]
    if missing:
        print(f"[Phase 4] WARNING missing cols: {missing}")

    # Rolling 60-bar median for each feature (lagged by 1 bar — past-only)
    rolling_meds = {}
    for f, arr in feat_arr.items():
        s = pd.Series(arr)
        # past-only: use only bars 0..i-1
        rolling_meds[f] = s.shift(1).rolling(60, min_periods=20).median().to_numpy()

    per_day_signatures = []
    elevated_counts = {f: 0 for f in feat_arr}
    n_eval = 0

    for r in per_day_results:
        if r["status"] != "OK":
            per_day_signatures.append(None)
            continue
        entry_idx = r["entry_bar_idx"]
        # Window [entry_idx - 6, entry_idx + 6]
        w_start = max(0, entry_idx - 6)
        w_end = min(len(df_train_1h), entry_idx + 7)
        sig = {}
        for f, arr in feat_arr.items():
            window_vals = arr[w_start:w_end]
            entry_val = arr[entry_idx]
            roll_med = rolling_meds[f][entry_idx] if not np.isnan(rolling_meds[f][entry_idx]) else None
            elevated = False
            if roll_med is not None and not np.isnan(entry_val):
                elevated = bool(entry_val > roll_med)
            # direction: mean of pre-entry 6 bars vs post-entry 6 bars
            pre_vals = arr[w_start:entry_idx]
            post_vals = arr[entry_idx:w_end]
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
                "elevated_vs_60bar_median": elevated,
                "rolling_60bar_median": float(roll_med) if roll_med is not None else None,
            }
            if elevated:
                elevated_counts[f] += 1
        n_eval += 1
        per_day_signatures.append(sig)

    # Aggregate: % of mover days where feature is elevated at entry
    elevation_pct = {f: (elevated_counts[f] / n_eval if n_eval > 0 else 0.0) for f in feat_arr}
    ranked = sorted(elevation_pct.items(), key=lambda kv: -kv[1])
    print(f"[Phase 4] n mover-days evaluated for signature: {n_eval}")
    for f, pct in ranked:
        print(f"[Phase 4]   {f:32s}: elevated in {pct:.1%}")

    # Correlation between (elevated bool per feature) and capture_rate
    correlations = {}
    for f in feat_arr:
        x = []  # elevated 0/1
        y = []  # capture_rate
        for r, sig in zip(per_day_results, per_day_signatures):
            if r["status"] != "OK" or sig is None:
                continue
            cr = r["capture_rate"]
            ev = sig[f]["elevated_vs_60bar_median"]
            if cr is None or ev is None:
                continue
            x.append(1.0 if ev else 0.0)
            y.append(cr)
        if len(x) >= 5 and np.std(x) > 0 and np.std(y) > 0:
            correlations[f] = float(np.corrcoef(x, y)[0, 1])
        else:
            correlations[f] = None
    return per_day_signatures, elevation_pct, correlations


# ---------- Main ----------

def main():
    print("=" * 70)
    print("PEPE TRAIN MOVER-DAY MINING — W1 MAXX-INST-2026-05-26-NIGHT")
    print("=" * 70)

    # Phase 1
    p1_summary, movers_list, df_train_1d = phase1_mover_days()

    # Phase 2
    per_day_results, df_train_1h, ma_cache = phase2_best_setup_per_day(movers_list, df_train_1d)

    # Phase 3
    exit_results, exit_summary = phase3_exits(per_day_results, df_train_1h, ma_cache)

    # Phase 4
    per_day_signatures, elevation_pct, correlations = phase4_chimera_signature(per_day_results, df_train_1h)

    # Save raw output
    output = {
        "_meta": {
            "git_sha": GIT_SHA,
            "chimera_mtime_1h": "2026-05-24T21:35:13",
            "canonical_seeds": CANONICAL_SEEDS,
            "train_only": True,
            "ts_generated": "2026-05-27T02:55+02:00",
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
            "correlation_elevated_vs_capture_rate": correlations,
            "per_day_signatures": per_day_signatures,
        },
    }

    out_path = OUT_DIR / "pepe_train_mover_day_2026_05_27.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n[OK] Output -> {out_path}")
    print(f"[OK] Size: {out_path.stat().st_size / 1024:.1f} KB")

    return output


if __name__ == "__main__":
    main()
