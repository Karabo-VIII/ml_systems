"""Robustness of fixed setups across all mover days @ 4h cadence."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "runs/mover_day/pepe_train_mover_day_4h_2026_05_27.json"
CHIM_4H = ROOT / "data/processed/chimera/4h/pepeusdt_v51_chimera_4h_20260522.parquet"

with open(DATA) as f:
    d = json.load(f)

df_4h = pl.read_parquet(CHIM_4H).sort("timestamp")
n_train_4h = int(d["_meta"]["n_train_4h_bars"])
ts_4h = df_4h["timestamp"].to_numpy()
closes_4h = df_4h["close"].to_numpy()
opens_4h = df_4h["open"].to_numpy()

train_closes = closes_4h[:n_train_4h]

LOOKBACK = int(d["_meta"]["lookback_bars"])


def compute_ma(prices, kind, window):
    n = len(prices); out = np.full(n, np.nan)
    if kind == "SMA" and n >= window:
        csum = np.cumsum(prices)
        out[window-1:] = (csum[window-1:] - np.concatenate([[0], csum[:-window]])) / window
    elif kind == "EMA" and n >= window:
        alpha = 2.0/(window+1.0); seed = prices[:window].mean(); out[window-1] = seed
        for i in range(window, n):
            out[i] = alpha*prices[i] + (1-alpha)*out[i-1]
    elif kind == "WMA" and n >= window:
        w = np.arange(1, window+1, dtype=float); ws = w.sum()
        for i in range(window-1, n):
            out[i] = (prices[i-window+1:i+1] * w).sum() / ws
    return out


def first_cross_up(fast_ma, slow_ma, start, end):
    s = max(start, 1)
    if s >= end:
        return -1
    prev_f = fast_ma[s-1:end-1]; prev_s = slow_ma[s-1:end-1]
    cur_f = fast_ma[s:end]; cur_s = slow_ma[s:end]
    mask = (~np.isnan(prev_f)) & (~np.isnan(prev_s)) & (~np.isnan(cur_f)) & (~np.isnan(cur_s)) \
        & (prev_f <= prev_s) & (cur_f > cur_s)
    if not mask.any():
        return -1
    return s + int(np.argmax(mask))


per_day = d["phase2_per_day"]
ok = [r for r in per_day if r["status"] == "OK"]
setup_counter = Counter()
for r in ok:
    s = r["best_setup"]
    setup_counter[(s["kind"], s["fast"], s["slow"])] += 1

CANDIDATES = []
for s, _ in setup_counter.most_common(15):
    CANDIDATES.append(s)
# 4h analog of 1h SMA 9/21 -> doesn't directly translate (would be SMA 2.25/5.25)
# Closest matches: SMA 3/9 (12h/36h), SMA 5/12 (20h/48h)
# Closest analog by *time-span* (60h slow): SMA 5/15 (60h slow), SMA 3/15
for s in [("SMA", 3, 9), ("SMA", 5, 12), ("SMA", 3, 12)]:
    if s not in CANDIDATES:
        CANDIDATES.append(s)

ma_cache = {}
for (kind, fast, slow) in CANDIDATES:
    if (kind, fast) not in ma_cache:
        ma_cache[(kind, fast)] = compute_ma(train_closes, kind, fast)
    if (kind, slow) not in ma_cache:
        ma_cache[(kind, slow)] = compute_ma(train_closes, kind, slow)

movers = d["movers"]
N_MOVERS = len(movers)

print(f"\n=== Robustness: fixed setups across all {N_MOVERS} mover days @ 4h ===")
print(f"{'setup':>14s} {'n_fires':>8s} {'fire_rate':>10s} {'med_cap%':>10s} {'mean_cap%':>10s} {'win_rate':>9s} {'med_offset_h':>12s}")
robust_results = {}
for (kind, fast, slow) in CANDIDATES:
    fast_ma = ma_cache[(kind, fast)]
    slow_ma = ma_cache[(kind, slow)]
    captures = []
    offsets_bars = []
    for m in movers:
        day_start_ts = m["ts_ms"]
        day_close_idx = int(np.searchsorted(ts_4h, day_start_ts, side="right") - 1)
        if day_close_idx < 0 or day_close_idx >= n_train_4h:
            continue
        window_start = max(0, day_close_idx - (LOOKBACK - 1))
        cross_idx = first_cross_up(fast_ma, slow_ma, window_start + 1, day_close_idx + 1)
        if cross_idx == -1 or cross_idx + 1 > day_close_idx:
            continue
        entry_idx = cross_idx + 1
        entry_p = opens_4h[entry_idx]
        if entry_p <= 0:
            continue
        daily_close_price = closes_4h[day_close_idx]
        cap = daily_close_price / entry_p - 1.0
        captures.append(cap)
        offsets_bars.append(day_close_idx - entry_idx)

    n_fires = len(captures)
    if n_fires == 0:
        continue
    caps_arr = np.array(captures)
    setup_str = f"{kind} {fast}/{slow}"
    print(f"{setup_str:>14s} {n_fires:>8d} {100*n_fires/N_MOVERS:>9.1f}% "
          f"{100*np.median(caps_arr):>9.3f}% {100*np.mean(caps_arr):>9.3f}% "
          f"{(caps_arr>0).mean():>8.2%} {np.median(offsets_bars)*4.0:>11.1f}h")
    robust_results[setup_str] = {
        "n_fires": int(n_fires),
        "fire_rate": float(n_fires/N_MOVERS),
        "median_capture_pct": float(np.median(caps_arr)),
        "mean_capture_pct": float(np.mean(caps_arr)),
        "win_rate": float((caps_arr>0).mean()),
        "median_offset_bars": float(np.median(offsets_bars)),
        "median_offset_h_approx": float(np.median(offsets_bars))*4.0,
        "p25_capture": float(np.quantile(caps_arr, 0.25)),
        "p75_capture": float(np.quantile(caps_arr, 0.75)),
        "p10_capture": float(np.quantile(caps_arr, 0.10)),
        "p90_capture": float(np.quantile(caps_arr, 0.90)),
    }

OUT = ROOT / "runs/mover_day/pepe_train_mover_day_4h_robust_2026_05_27.json"
with open(OUT, "w") as f:
    json.dump(robust_results, f, indent=2)
print(f"\nSaved -> {OUT}")
