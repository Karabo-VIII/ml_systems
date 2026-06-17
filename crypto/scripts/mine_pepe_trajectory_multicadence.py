"""
PEPE TRAIN mover-day TRAJECTORY analysis across 4 cadences (15m, 30m, 1h, 4h).

Brief: docs/dossiers/PEPE_TRAIN_MOVER_DAY_TRAJECTORY_2026_05_27.md
W1 of MAXX-INST-2026-05-26-NIGHT (continuation worker).

User mandate: prior 1h analysis tested STATIC feature value at entry vs capture rate
(max |corr| = 0.09, all weak). User asked the right question: does the chimera bar
trajectory (rising, falling, accelerating, spiking) over the bars LEADING UP TO
entry tell a story?

Phases:
  1. Identify daily mover days (2%+ UP) in TRAIN window (canonical 1d ground truth)
  2. For each cadence: map each mover day to its "entry bar" — the SMA(9,21) cross-up
     within last 72h equivalent before the daily close
  3. For each cadence × feature × entry: compute 8 trajectory metrics on lookback K=12h
  4. Aggregate per cadence and cross-cadence: median values, consistency %, correlation
     with capture_rate

Past-only at every layer. UNSEEN/VAL/OOS untouched.
"""
import json
import sys
from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CHIM = {
    "15m": ROOT / "data/processed/chimera/15m/pepeusdt_v51_chimera_15m_20260522.parquet",
    "30m": ROOT / "data/processed/chimera/30m/pepeusdt_v51_chimera_30m_20260522.parquet",
    "1h": ROOT / "data/processed/chimera/1h/pepeusdt_v51_chimera_1h_20260522.parquet",
    "4h": ROOT / "data/processed/chimera/4h/pepeusdt_v51_chimera_4h_20260522.parquet",
}
CHIM_1D = ROOT / "data/processed/chimera/1d/pepeusdt_v51_chimera_1d_20260522.parquet"

OUT_DIR = ROOT / "runs/audit/MAXX_2026_05_26/data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_SEEDS = {"bag_seed": 120, "feat_seed": 1120, "rng_seed": 8030}
GIT_SHA = "3adfb8e"

# Cross-version split contract (CLAUDE.md canonical 50/20/20/10)
TRAIN_FRAC = 0.50

# Per-cadence lookback K (bars) — all approx 12 hours wall-clock
# 15m: 48 bars = 12h ; 30m: 24 bars = 12h ; 1h: 12 bars = 12h ; 4h: 3 bars = 12h
K_BARS = {"15m": 48, "30m": 24, "1h": 12, "4h": 3}

# Bars-per-day at each cadence (used to map mover days to bars and to size signal-search windows)
BARS_PER_DAY = {"15m": 96, "30m": 48, "1h": 24, "4h": 6}

# Signal-search lookback: 72 hours (3 days) — same as 1h worker
SIGNAL_HOURS_BACK = 72
SIGNAL_LB_BARS = {"15m": SIGNAL_HOURS_BACK * 4, "30m": SIGNAL_HOURS_BACK * 2, "1h": SIGNAL_HOURS_BACK, "4h": SIGNAL_HOURS_BACK // 4}

# 10 chimera features (mapped from mandate)
# Notes: bd_imb does not exist -> bd_imbalance_l1 (best match); rv_rv_30m does not exist -> rv_jv_5m
TRAJ_FEATURES = [
    "wh_whale_net_usd",
    "wh_whale_trade_count",
    "fund_rate_mean",
    "bs_basis_z30",
    "hbr_eta_buy",
    "bd_imbalance_l1",  # mandate said "bd_imb" but only bd_imbalance_l1 exists
    "te_btc_imb",
    "rv_rv_5m",
    "rv_jv_5m",          # mandate said "rv_rv_30m" but only rv_jv_5m exists at this granularity
    "premium_z90",
]

# SMA used for entry detection — same as prior 1h worker (SMA 9/21 won win-rate)
SMA_FAST = 9
SMA_SLOW = 21


def compute_sma(prices: np.ndarray, window: int) -> np.ndarray:
    """Past-only SMA. NaN before window-1."""
    n = len(prices)
    out = np.full(n, np.nan)
    if n >= window:
        csum = np.cumsum(prices)
        out[window - 1:] = (csum[window - 1:] - np.concatenate([[0], csum[:-window]])) / window
    return out


def find_last_cross_up(fast_ma: np.ndarray, slow_ma: np.ndarray, start: int, end: int) -> int:
    """Find LAST cross-up in [start, end). Same definition as prior worker but we take the most-recent
    cross before the close, since that's the entry that just fired before the move.

    Returns -1 if no cross.
    """
    last = -1
    for i in range(max(start, 1), end):
        if (
            not np.isnan(fast_ma[i - 1])
            and not np.isnan(slow_ma[i - 1])
            and not np.isnan(fast_ma[i])
            and not np.isnan(slow_ma[i])
            and fast_ma[i - 1] <= slow_ma[i - 1]
            and fast_ma[i] > slow_ma[i]
        ):
            last = i
    return last


def get_train_mover_days() -> tuple[list, dict]:
    """Return list of mover day timestamps (ms) + daily-return + capture_rate (computed at 1h)
    from canonical 1d ground truth. TRAIN window = first 50%."""
    df_1d = pl.read_parquet(CHIM_1D).sort("timestamp")
    n = len(df_1d)
    train_end = int(n * TRAIN_FRAC)
    df_train = df_1d[:train_end]
    closes = df_train["close"].to_numpy()
    ret = np.zeros(len(closes))
    ret[1:] = closes[1:] / closes[:-1] - 1.0
    df_train = df_train.with_columns(pl.Series("daily_return", ret))

    movers = df_train.filter(pl.col("daily_return") >= 0.02)
    mover_ts = movers["timestamp"].to_list()
    mover_ret = movers["daily_return"].to_list()

    summary = {
        "n_train_days": int(len(df_train)),
        "n_mover_days_2pct": int(len(movers)),
        "train_start_ts": int(df_train["timestamp"][0]),
        "train_end_ts": int(df_train["timestamp"][-1]),
    }
    return list(zip(mover_ts, mover_ret)), summary


def compute_trajectory_metrics(arr: np.ndarray, entry_idx: int, K: int) -> dict:
    """Compute 8 trajectory metrics for feature arr over lookback [entry_idx-K, entry_idx-1].

    STRICTLY past-only: never reads arr[entry_idx] or later.
    Returns dict with all metrics + nan flags for unavailable.
    """
    out = {
        "slope_norm": None,
        "delta": None,
        "monotonic_up": None,
        "monotonic_down": None,
        "flat": None,
        "acceleration": None,
        "spike": None,
        "z_score": None,
    }
    if entry_idx < K + 1:
        return out  # not enough history
    window = arr[entry_idx - K:entry_idx]  # exclusive of entry_idx itself
    if np.any(np.isnan(window)):
        # Try to handle: drop NaN; need at least K//2 valid points
        valid = window[~np.isnan(window)]
        if len(valid) < max(3, K // 2):
            return out
        window = valid

    n_w = len(window)
    x = np.arange(n_w, dtype=float)

    # 1. Slope (linear regression), normalized by feature IQR over the window
    try:
        coeffs = np.polyfit(x, window, 1)
        slope = coeffs[0]
    except np.linalg.LinAlgError:
        slope = 0.0
    iqr = np.percentile(window, 75) - np.percentile(window, 25)
    if iqr > 1e-12:
        out["slope_norm"] = float(slope / iqr)
    else:
        out["slope_norm"] = 0.0 if abs(slope) < 1e-12 else float(np.sign(slope) * 1.0)

    # 2. Delta = window[-1] - window[0]
    out["delta"] = float(window[-1] - window[0])

    # 3. Monotonic-up: ≥60% of NON-ZERO consecutive diffs are positive (handles equal-value bars
    #    common in slow-update features like te_btc_imb and fund_rate_mean). Falls back to "flat"
    #    if <30% of diffs are non-zero (genuinely flat feature in this window).
    diffs = np.diff(window)
    if len(diffs) > 0:
        nz_mask = np.abs(diffs) > 1e-12
        nz_frac = float(nz_mask.mean())
        if nz_frac < 0.30:
            out["monotonic_up"] = False
            out["monotonic_down"] = False
            out["flat"] = True
        else:
            nz_diffs = diffs[nz_mask]
            pct_up = float((nz_diffs > 0).mean())
            pct_dn = float((nz_diffs < 0).mean())
            out["monotonic_up"] = bool(pct_up >= 0.60)
            out["monotonic_down"] = bool(pct_dn >= 0.60)
            out["flat"] = bool(not out["monotonic_up"] and not out["monotonic_down"])

    # 4. Acceleration: 2nd-order polynomial fit, take the quadratic coefficient
    if n_w >= 4:
        try:
            q_coeffs = np.polyfit(x, window, 2)
            # Quadratic term * 2 = approx second derivative
            accel = q_coeffs[0]
            # Normalize by mean-of-window-mag if non-trivial
            ref = np.mean(np.abs(window)) + 1e-12
            out["acceleration"] = float(accel / ref)
        except np.linalg.LinAlgError:
            out["acceleration"] = 0.0

    # 5. Spike: last 2 bars' mean > 2σ above K-2 prior bars' mean
    if n_w >= 4:
        recent = window[-2:].mean()
        prior = window[:-2]
        p_mean = prior.mean()
        p_std = prior.std()
        if p_std > 1e-12:
            out["spike"] = bool((recent - p_mean) / p_std > 2.0)
        else:
            out["spike"] = False

    # 6. Z-score: window[-1] vs window mean/std
    m = window.mean()
    s = window.std()
    if s > 1e-12:
        out["z_score"] = float((window[-1] - m) / s)
    else:
        out["z_score"] = 0.0

    return out


def process_cadence(cad: str, mover_data: list, train_end_ts: int) -> tuple[dict, list]:
    """Process one cadence — identify entry bars at this cadence and compute trajectory metrics."""
    print(f"\n[{cad}] Loading chimera...")
    df = pl.read_parquet(CHIM[cad]).sort("timestamp")
    df = df.with_columns(pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("dt"))

    # TRAIN window: bars up to end of last TRAIN day + 24h margin (to allow within-day entries)
    df_train = df.filter(pl.col("timestamp") <= train_end_ts + 24 * 3600 * 1000)
    n_train = len(df_train)
    print(f"[{cad}] TRAIN bars: {n_train}")

    ts = df_train["timestamp"].to_numpy()
    closes = df_train["close"].to_numpy()
    opens = df_train["open"].to_numpy()

    # Compute SMA(9,21) on closes
    fast_ma = compute_sma(closes, SMA_FAST)
    slow_ma = compute_sma(closes, SMA_SLOW)

    # Pre-load feature arrays
    feat_arrays = {}
    for f in TRAJ_FEATURES:
        if f in df_train.columns:
            feat_arrays[f] = df_train[f].to_numpy()

    K = K_BARS[cad]
    sig_lb = SIGNAL_LB_BARS[cad]

    # For each mover day: find the day's last bar, search backwards for last SMA cross-up,
    # set entry_idx = cross + 1.
    entries = []
    for day_ts, daily_ret in mover_data:
        # Find last bar at this cadence with ts <= day_ts (day close moment)
        day_close_idx = int(np.searchsorted(ts, day_ts, side="right") - 1)
        if day_close_idx < 0 or day_close_idx >= n_train:
            continue
        # Signal-search window: [day_close_idx - sig_lb, day_close_idx]
        sig_start = max(1, day_close_idx - sig_lb)
        sig_end = day_close_idx + 1  # last cross must be observed by day-close bar
        cross_idx = find_last_cross_up(fast_ma, slow_ma, sig_start, sig_end)
        if cross_idx == -1:
            continue
        if cross_idx + 1 > day_close_idx:
            continue  # no clean entry before close
        entry_idx = cross_idx + 1
        # Capture rate: from entry-bar open to day-close-bar close
        entry_p = opens[entry_idx]
        close_p = closes[day_close_idx]
        if entry_p <= 0:
            continue
        captured = close_p / entry_p - 1.0
        # Available move: best entry within signal-search window to day-close
        avail_returns = []
        for i in range(sig_start, day_close_idx):
            ep = opens[i + 1] if i + 1 <= day_close_idx else None
            if ep and ep > 0:
                avail_returns.append(close_p / ep - 1.0)
        avail = max(avail_returns) if avail_returns else 1e-9
        cap_rate = captured / avail if avail > 1e-9 else 0.0

        # Compute trajectory metrics per feature
        traj = {}
        for f, arr in feat_arrays.items():
            traj[f] = compute_trajectory_metrics(arr, entry_idx, K)

        entries.append({
            "day_ts": int(day_ts),
            "daily_return": float(daily_ret),
            "entry_idx": int(entry_idx),
            "day_close_idx": int(day_close_idx),
            "entry_offset_bars": int(day_close_idx - entry_idx),
            "captured_move_pct": float(captured),
            "available_move_pct": float(avail),
            "capture_rate": float(cap_rate),
            "trajectory": traj,
        })

    print(f"[{cad}] Mover-day entries identified: {len(entries)}")

    # Aggregate per feature × trajectory metric
    agg = {}
    capture_rates = np.array([e["capture_rate"] for e in entries])
    for f in feat_arrays:
        agg[f] = {}
        for metric in ["slope_norm", "delta", "monotonic_up", "monotonic_down",
                       "flat", "acceleration", "spike", "z_score"]:
            vals = []
            cr_paired = []
            for e in entries:
                v = e["trajectory"][f][metric]
                cr = e["capture_rate"]
                if v is None:
                    continue
                vals.append(v)
                cr_paired.append(cr)
            if not vals:
                agg[f][metric] = {"n": 0}
                continue
            vals_arr = np.array(vals, dtype=float)
            cr_arr = np.array(cr_paired)
            if metric in ["monotonic_up", "monotonic_down", "flat", "spike"]:
                consistency_pct = float(vals_arr.mean())  # bool->% True
                if vals_arr.std() > 0 and cr_arr.std() > 0:
                    corr = float(np.corrcoef(vals_arr.astype(float), cr_arr)[0, 1])
                else:
                    corr = None
                agg[f][metric] = {
                    "n": int(len(vals)),
                    "consistency_pct_true": consistency_pct,
                    "corr_with_capture": corr,
                }
            else:
                if vals_arr.std() > 1e-12 and cr_arr.std() > 0:
                    corr = float(np.corrcoef(vals_arr, cr_arr)[0, 1])
                else:
                    corr = None
                agg[f][metric] = {
                    "n": int(len(vals)),
                    "median": float(np.median(vals_arr)),
                    "mean": float(np.mean(vals_arr)),
                    "p10": float(np.percentile(vals_arr, 10)),
                    "p90": float(np.percentile(vals_arr, 90)),
                    "corr_with_capture": corr,
                }
    return {
        "cadence": cad,
        "n_entries": len(entries),
        "K_lookback_bars": K,
        "signal_search_bars": sig_lb,
        "aggregate": agg,
    }, entries


def main():
    print("=" * 70)
    print("PEPE TRAJECTORY MULTI-CADENCE — W1 MAXX-INST-2026-05-26-NIGHT")
    print("=" * 70)

    mover_data, summary_1d = get_train_mover_days()
    train_end_ts = summary_1d["train_end_ts"]
    print(f"[1d] TRAIN days: {summary_1d['n_train_days']}, mover days: {summary_1d['n_mover_days_2pct']}")
    print(f"[1d] TRAIN end ts: {train_end_ts}")

    per_cadence_results = {}
    per_cadence_entries = {}
    for cad in ["15m", "30m", "1h", "4h"]:
        agg_result, entries = process_cadence(cad, mover_data, train_end_ts)
        per_cadence_results[cad] = agg_result
        per_cadence_entries[cad] = entries

        # Per-cadence JSON
        out_path = OUT_DIR / f"pepe_train_trajectory_{cad}.json"
        # Drop per-day entries from saved per-cadence file (keep aggregate + a sample)
        cadence_out = {
            "_meta": {
                "git_sha": GIT_SHA,
                "canonical_seeds": CANONICAL_SEEDS,
                "train_only": True,
                "cadence": cad,
                "ts_generated": "2026-05-27T03:20+02:00",
            },
            "aggregate": agg_result,
            "entries": entries[:10],  # first 10 as sample
            "n_entries_total": len(entries),
        }
        with open(out_path, "w") as f:
            json.dump(cadence_out, f, indent=2, default=str)
        print(f"[OK] -> {out_path.name} ({out_path.stat().st_size/1024:.1f}KB)")

    # Cross-cadence summary
    cross = {
        "_meta": {
            "git_sha": GIT_SHA,
            "canonical_seeds": CANONICAL_SEEDS,
            "train_only": True,
            "ts_generated": "2026-05-27T03:20+02:00",
            "1d_summary": summary_1d,
        },
        "per_cadence": {c: per_cadence_results[c] for c in ["15m", "30m", "1h", "4h"]},
        "trajectory_features": TRAJ_FEATURES,
    }
    # Cross-cadence consistency: features where the SAME trajectory metric direction shows up at >=2 cadences
    cross_signals = []
    for f in TRAJ_FEATURES:
        for metric in ["slope_norm", "delta", "monotonic_up", "monotonic_down", "flat", "acceleration", "spike", "z_score"]:
            cadence_signals = []
            for cad in ["15m", "30m", "1h", "4h"]:
                a = per_cadence_results[cad]["aggregate"].get(f, {}).get(metric, {})
                if metric in ["monotonic_up", "monotonic_down", "flat", "spike"]:
                    cons = a.get("consistency_pct_true")
                    corr = a.get("corr_with_capture")
                    if cons is not None:
                        cadence_signals.append({"cad": cad, "consistency": cons, "corr": corr})
                else:
                    med = a.get("median")
                    corr = a.get("corr_with_capture")
                    if med is not None:
                        cadence_signals.append({"cad": cad, "median": med, "corr": corr})
            if cadence_signals:
                cross_signals.append({
                    "feature": f,
                    "metric": metric,
                    "cadences": cadence_signals,
                })
    cross["cross_cadence_table"] = cross_signals

    # Top trajectory signals: |corr| >= 0.15 at any cadence
    top_signals = []
    for f in TRAJ_FEATURES:
        for metric in ["slope_norm", "delta", "monotonic_up", "monotonic_down", "flat", "acceleration", "spike", "z_score"]:
            for cad in ["15m", "30m", "1h", "4h"]:
                a = per_cadence_results[cad]["aggregate"].get(f, {}).get(metric, {})
                corr = a.get("corr_with_capture")
                if corr is not None and abs(corr) >= 0.15:
                    sig_entry = {
                        "feature": f,
                        "metric": metric,
                        "cadence": cad,
                        "corr": corr,
                        "n": a.get("n"),
                    }
                    if "consistency_pct_true" in a:
                        sig_entry["consistency"] = a["consistency_pct_true"]
                    elif "median" in a:
                        sig_entry["median"] = a["median"]
                    top_signals.append(sig_entry)
    cross["top_trajectory_signals"] = sorted(top_signals, key=lambda x: -abs(x["corr"]))

    summary_path = OUT_DIR / "pepe_train_trajectory_summary.json"
    with open(summary_path, "w") as f:
        json.dump(cross, f, indent=2, default=str)
    print(f"\n[OK] Summary -> {summary_path.name} ({summary_path.stat().st_size/1024:.1f}KB)")

    # Print headline findings
    print(f"\n[HEADLINE] Top trajectory signals with |corr| >= 0.15:")
    if not cross["top_trajectory_signals"]:
        print("  NONE — honest NULL declaration: no trajectory metric meets the |corr|>=0.15 bar at any cadence")
    else:
        for s in cross["top_trajectory_signals"][:20]:
            print(f"  [{s['cadence']:>3s}] {s['feature']:25s} × {s['metric']:18s} corr={s['corr']:+.4f} n={s['n']}")

    return cross


if __name__ == "__main__":
    main()
