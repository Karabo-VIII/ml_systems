"""Test 6 exit policies on TOP-3 robust setups (fixed setup, all 186 mover days)."""
import json
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "runs/mover_day/pepe_train_mover_day_2026_05_27.json"
CHIM_1H = ROOT / "data/processed/chimera/1h/pepeusdt_v51_chimera_1h_20260522.parquet"

with open(DATA) as f:
    d = json.load(f)

df_1h = pl.read_parquet(CHIM_1H).sort("timestamp")
ts_1h = df_1h["timestamp"].to_numpy()
closes_1h = df_1h["close"].to_numpy()
opens_1h = df_1h["open"].to_numpy()
highs_1h = df_1h["high"].to_numpy()
n_train_1h = 13022

def compute_ma(prices, kind, window):
    n = len(prices); out = np.full(n, np.nan)
    if kind == "SMA" and n >= window:
        csum = np.cumsum(prices); out[window-1:] = (csum[window-1:] - np.concatenate([[0], csum[:-window]])) / window
    elif kind == "EMA" and n >= window:
        alpha = 2.0/(window+1.0); seed = prices[:window].mean(); out[window-1] = seed
        for i in range(window, n): out[i] = alpha*prices[i] + (1-alpha)*out[i-1]
    elif kind == "WMA" and n >= window:
        w = np.arange(1, window+1, dtype=float); w_sum = w.sum()
        for i in range(window-1, n): out[i] = (prices[i-window+1:i+1] * w).sum() / w_sum
    return out

def find_first_cross_up(fast_ma, slow_ma, start, end):
    for i in range(max(start, 1), end):
        if (
            not np.isnan(fast_ma[i-1]) and not np.isnan(slow_ma[i-1])
            and not np.isnan(fast_ma[i]) and not np.isnan(slow_ma[i])
            and fast_ma[i-1] <= slow_ma[i-1] and fast_ma[i] > slow_ma[i]
        ):
            return i
    return -1

train_closes = closes_1h[:n_train_1h]
SETUPS = [("SMA", 20, 21), ("SMA", 12, 15), ("SMA", 9, 21), ("WMA", 5, 15), ("SMA", 8, 26)]

results = {}
for (kind, fast, slow) in SETUPS:
    f_ma = compute_ma(train_closes, kind, fast); s_ma = compute_ma(train_closes, kind, slow)
    exit_pnls = {"E1_opp_cross": [], "E2_6h": [], "E3_12h": [], "E4_24h": [], "E5_48h": [], "E6_mfe_trail50": []}
    n_entries = 0
    for m in d["movers"]:
        day_close_idx = np.searchsorted(ts_1h, m["ts_ms"], side="right") - 1
        if day_close_idx < 0: continue
        ws = max(0, day_close_idx - 71); sss = ws + 1; sse = day_close_idx + 1
        ci = find_first_cross_up(f_ma, s_ma, sss, sse)
        if ci == -1 or ci + 1 > day_close_idx: continue
        ei = ci + 1
        if opens_1h[ei] <= 0: continue
        ep = opens_1h[ei]; n_entries += 1

        # E1
        e1_ex = None
        for i in range(ei, min(ei + 14*24, n_train_1h)):
            if not np.isnan(f_ma[i]) and not np.isnan(s_ma[i]) and f_ma[i] <= s_ma[i]:
                e1_ex = i; break
        if e1_ex is not None and e1_ex + 1 < n_train_1h:
            exit_pnls["E1_opp_cross"].append(opens_1h[e1_ex+1]/ep - 1)

        # E2-E5
        for hours, key in [(6,"E2_6h"),(12,"E3_12h"),(24,"E4_24h"),(48,"E5_48h")]:
            if ei + hours < n_train_1h:
                exit_pnls[key].append(opens_1h[ei+hours]/ep - 1)

        # E6
        peak_p = ep; e6 = None
        end_walk = min(ei + 14*24, n_train_1h - 1)
        for i in range(ei+1, end_walk+1):
            peak_p = max(peak_p, highs_1h[i])
            peak_ret = peak_p/ep - 1
            if peak_ret >= 0.03:
                lock_p = ep*(1 + peak_ret*0.5)
                if closes_1h[i] <= lock_p:
                    if i+1 < n_train_1h:
                        e6 = opens_1h[i+1]/ep - 1
                    else:
                        e6 = closes_1h[i]/ep - 1
                    break
        if e6 is None:
            e6 = closes_1h[end_walk]/ep - 1
        exit_pnls["E6_mfe_trail50"].append(e6)

    setup_key = f"{kind} {fast}/{slow}"
    print(f"\n=== {setup_key} ({n_entries} entries) ===")
    setup_summary = {"n_entries": n_entries}
    for e, vals in exit_pnls.items():
        if not vals: continue
        arr = np.array(vals)
        s = {"n": len(arr), "median": float(np.median(arr)), "mean": float(np.mean(arr)),
             "p10": float(np.quantile(arr, 0.10)), "p90": float(np.quantile(arr, 0.90)),
             "win_rate": float((arr>0).mean()), "p25": float(np.quantile(arr, 0.25)),
             "p75": float(np.quantile(arr, 0.75))}
        setup_summary[e] = s
        print(f"  {e:18s}: n={s['n']:3d} med={s['median']:+.4f} mean={s['mean']:+.4f} p10={s['p10']:+.4f} p90={s['p90']:+.4f} win={s['win_rate']:.2%}")
    results[setup_key] = setup_summary

OUT = ROOT / "runs/mover_day/pepe_train_mover_day_exits_robust_2026_05_27.json"
with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved -> {OUT}")
