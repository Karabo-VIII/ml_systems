"""Beyond best-per-day: also compute robustness of fixed setups across all mover days."""
import json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "runs/mover_day/pepe_train_mover_day_2026_05_27.json"
CHIM_1H = ROOT / "data/processed/chimera/1h/pepeusdt_v51_chimera_1h_20260522.parquet"

with open(DATA) as f:
    d = json.load(f)

# To compute robustness, we need to re-evaluate FIXED setups across all mover days
# Pull 1h chimera again

df_1h = pl.read_parquet(CHIM_1H).sort("timestamp")
n_train_1h = 13022  # from prior run
ts_1h = df_1h["timestamp"].to_numpy()
closes_1h = df_1h["close"].to_numpy()
opens_1h = df_1h["open"].to_numpy()

# Precompute MAs (only on TRAIN slice)
def compute_ma(prices, kind, window):
    n = len(prices)
    out = np.full(n, np.nan)
    if kind == "SMA":
        if n >= window:
            csum = np.cumsum(prices)
            out[window-1:] = (csum[window-1:] - np.concatenate([[0], csum[:-window]])) / window
    elif kind == "EMA":
        alpha = 2.0/(window+1.0)
        if n >= window:
            seed = prices[:window].mean()
            out[window-1] = seed
            for i in range(window, n):
                out[i] = alpha*prices[i] + (1-alpha)*out[i-1]
    elif kind == "WMA":
        if n >= window:
            w = np.arange(1, window+1, dtype=float); w_sum = w.sum()
            for i in range(window-1, n):
                out[i] = (prices[i-window+1:i+1] * w).sum() / w_sum
    return out

train_closes = closes_1h[:n_train_1h]

# Test candidate setups: top 15 most-common from per-day-best + the deploy candidate SMA 9/21 (PEPE_1h_TRADEABLE)
from collections import Counter
per_day = d["phase2_per_day"]
ok = [r for r in per_day if r["status"] == "OK"]
setup_counter = Counter()
for r in ok:
    s = r["best_setup"]
    setup_counter[(s["kind"], s["fast"], s["slow"])] += 1

CANDIDATES = []
for s, _ in setup_counter.most_common(15):
    CANDIDATES.append(s)
# Add SMA 9/21 (the known candidate from PEPE_1h_TRADEABLE_2026_05_27.md)
if ("SMA", 9, 21) not in CANDIDATES:
    CANDIDATES.append(("SMA", 9, 21))

ma_cache = {}
for (kind, fast, slow) in CANDIDATES:
    if (kind, fast) not in ma_cache:
        ma_cache[(kind, fast)] = compute_ma(train_closes, kind, fast)
    if (kind, slow) not in ma_cache:
        ma_cache[(kind, slow)] = compute_ma(train_closes, kind, slow)

# For each candidate, run the SAME phase 2 logic but with FIXED setup, then aggregate capture_pct across all 186 mover days
movers = d["movers"]

def find_first_cross_up(fast_ma, slow_ma, start, end):
    for i in range(max(start, 1), end):
        if (
            not np.isnan(fast_ma[i-1]) and not np.isnan(slow_ma[i-1])
            and not np.isnan(fast_ma[i]) and not np.isnan(slow_ma[i])
            and fast_ma[i-1] <= slow_ma[i-1]
            and fast_ma[i] > slow_ma[i]
        ):
            return i
    return -1

print("\n=== Robustness: fixed setups across all 186 mover days ===")
print(f"{'setup':>14s} {'n_fires':>8s} {'fire_rate':>10s} {'med_cap%':>10s} {'mean_cap%':>10s} {'win_rate':>9s} {'med_offset_h':>12s}")
robust_results = {}
for (kind, fast, slow) in CANDIDATES:
    fast_ma = ma_cache[(kind, fast)]
    slow_ma = ma_cache[(kind, slow)]
    captures = []
    offsets = []
    for m in movers:
        day_start_ts = m["ts_ms"]
        day_close_idx = np.searchsorted(ts_1h, day_start_ts, side="right") - 1
        if day_close_idx < 0:
            continue
        window_start = max(0, day_close_idx - 71)
        signal_search_start = window_start + 1
        signal_search_end = day_close_idx + 1
        cross_idx = find_first_cross_up(fast_ma, slow_ma, signal_search_start, signal_search_end)
        if cross_idx == -1 or cross_idx + 1 > day_close_idx:
            continue
        entry_idx = cross_idx + 1
        entry_p = opens_1h[entry_idx]
        if entry_p <= 0:
            continue
        daily_close_price = closes_1h[day_close_idx]
        cap = daily_close_price / entry_p - 1.0
        captures.append(cap)
        offsets.append(day_close_idx - entry_idx)

    n_fires = len(captures)
    if n_fires == 0:
        continue
    caps_arr = np.array(captures)
    setup_str = f"{kind} {fast}/{slow}"
    print(
        f"{setup_str:>14s} {n_fires:>8d} {100*n_fires/186:>9.1f}% "
        f"{100*np.median(caps_arr):>9.3f}% {100*np.mean(caps_arr):>9.3f}% "
        f"{(caps_arr>0).mean():>8.2%} {np.median(offsets):>11.1f}h"
    )
    robust_results[setup_str] = {
        "n_fires": int(n_fires),
        "fire_rate": float(n_fires/186),
        "median_capture_pct": float(np.median(caps_arr)),
        "mean_capture_pct": float(np.mean(caps_arr)),
        "win_rate": float((caps_arr>0).mean()),
        "median_offset_h": float(np.median(offsets)),
        "p25_capture": float(np.quantile(caps_arr, 0.25)),
        "p75_capture": float(np.quantile(caps_arr, 0.75)),
    }

# Save robust analysis
OUT = ROOT / "runs/mover_day/pepe_train_mover_day_robust_2026_05_27.json"
with open(OUT, "w") as f:
    json.dump(robust_results, f, indent=2)
print(f"\nSaved -> {OUT}")
