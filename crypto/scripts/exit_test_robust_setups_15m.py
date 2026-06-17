"""Test 6 exit policies on TOP-5 robust setups at 15m (fixed setup, all 186 mover days)."""
import json
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "runs/mover_day/pepe_train_mover_day_15m_2026_05_27.json"
CHIM_15M = ROOT / "data/processed/chimera/15m/pepeusdt_v51_chimera_15m_20260522.parquet"
BARS_PER_DAY = 96
BARS_PER_HOUR = 4

with open(DATA) as f:
    d = json.load(f)

df_15m = pl.read_parquet(CHIM_15M).sort("timestamp")
df_1d = pl.read_parquet(ROOT / "data/processed/chimera/1d/pepeusdt_v51_chimera_1d_20260522.parquet").sort("timestamp")
train_end_bar = int(len(df_1d) * 0.50)
train_end_ts = int(df_1d["timestamp"][train_end_bar - 1])
df_train_15m = df_15m.filter(pl.col("timestamp") <= train_end_ts + 24 * 3600 * 1000)

ts_15m = df_train_15m["timestamp"].to_numpy()
closes_15m = df_train_15m["close"].to_numpy()
opens_15m = df_train_15m["open"].to_numpy()
highs_15m = df_train_15m["high"].to_numpy()
n_train_15m = len(df_train_15m)


def compute_ma(prices, kind, window):
    n = len(prices); out = np.full(n, np.nan)
    if kind == "SMA" and n >= window:
        csum = np.cumsum(prices); out[window-1:] = (csum[window-1:] - np.concatenate([[0], csum[:-window]])) / window
    elif kind == "EMA" and n >= window:
        alpha = 2.0/(window+1.0); seed = prices[:window].mean(); out[window-1] = seed
        for i in range(window, n):
            out[i] = alpha*prices[i] + (1-alpha)*out[i-1]
    elif kind == "WMA" and n >= window:
        w = np.arange(1, window+1, dtype=float); w_sum = w.sum()
        kern = w[::-1] / w_sum
        conv = np.convolve(prices, kern, mode="valid")
        out[window-1:] = conv
    return out


def find_first_cross_up(fast_ma, slow_ma, start, end):
    if start < 1:
        start = 1
    if end <= start:
        return -1
    seg_a_prev = fast_ma[start - 1:end - 1]
    seg_b_prev = slow_ma[start - 1:end - 1]
    seg_a_now = fast_ma[start:end]
    seg_b_now = slow_ma[start:end]
    mask = (
        ~np.isnan(seg_a_prev) & ~np.isnan(seg_b_prev) &
        ~np.isnan(seg_a_now) & ~np.isnan(seg_b_now) &
        (seg_a_prev <= seg_b_prev) & (seg_a_now > seg_b_now)
    )
    if not mask.any():
        return -1
    return int(start + np.argmax(mask))


SETUPS = [
    ("WMA", 12, 48),  # top by n_days, 100% fire
    ("SMA", 12, 48),  # 100% fire, highest win
    ("WMA", 20, 48),  # 100% fire
    ("SMA", 36, 48),  # 99.5% fire
    ("SMA", 48, 60),  # 98.9% fire, 84.2% win
]
LOOKBACK_BARS = 2 * BARS_PER_DAY
EXIT_HOLD_BARS = {"E2_6h": 24, "E3_12h": 48, "E4_24h": 96, "E5_48h": 192}
CAP_BARS = 14 * BARS_PER_DAY

results = {}
for (kind, fast, slow) in SETUPS:
    f_ma = compute_ma(closes_15m, kind, fast)
    s_ma = compute_ma(closes_15m, kind, slow)
    exit_pnls = {"E1_opp_cross": [], "E2_6h": [], "E3_12h": [], "E4_24h": [], "E5_48h": [], "E6_mfe_trail50": []}
    n_entries = 0
    for m in d["movers"]:
        day_close_idx = np.searchsorted(ts_15m, m["ts_ms"], side="right") - 1
        if day_close_idx < 0: continue
        ws = max(0, day_close_idx - (LOOKBACK_BARS - 1)); sss = ws + 1; sse = day_close_idx + 1
        ci = find_first_cross_up(f_ma, s_ma, sss, sse)
        if ci == -1 or ci + 1 > day_close_idx: continue
        ei = ci + 1
        if opens_15m[ei] <= 0: continue
        ep = opens_15m[ei]; n_entries += 1

        # E1: opposite cross
        e1_ex = None
        for i in range(ei, min(ei + CAP_BARS, n_train_15m)):
            if not np.isnan(f_ma[i]) and not np.isnan(s_ma[i]) and f_ma[i] <= s_ma[i]:
                e1_ex = i; break
        if e1_ex is not None and e1_ex + 1 < n_train_15m:
            exit_pnls["E1_opp_cross"].append(opens_15m[e1_ex+1]/ep - 1)

        # E2-E5
        for key, hold in EXIT_HOLD_BARS.items():
            if ei + hold < n_train_15m:
                exit_pnls[key].append(opens_15m[ei + hold]/ep - 1)

        # E6 MFE-trail50
        peak_p = ep; e6 = None
        end_walk = min(ei + CAP_BARS, n_train_15m - 1)
        for i in range(ei + 1, end_walk + 1):
            peak_p = max(peak_p, highs_15m[i])
            peak_ret = peak_p/ep - 1
            if peak_ret >= 0.03:
                lock_p = ep*(1 + peak_ret*0.5)
                if closes_15m[i] <= lock_p:
                    if i + 1 < n_train_15m:
                        e6 = opens_15m[i+1]/ep - 1
                    else:
                        e6 = closes_15m[i]/ep - 1
                    break
        if e6 is None:
            e6 = closes_15m[end_walk]/ep - 1
        exit_pnls["E6_mfe_trail50"].append(e6)

    setup_key = f"{kind} {fast}/{slow}"
    print(f"\n=== {setup_key} ({n_entries} entries) ===")
    setup_summary = {"n_entries": n_entries}
    for e, vals in exit_pnls.items():
        if not vals: continue
        arr = np.array(vals)
        s = {
            "n": len(arr), "median": float(np.median(arr)), "mean": float(np.mean(arr)),
            "p10": float(np.quantile(arr, 0.10)), "p90": float(np.quantile(arr, 0.90)),
            "win_rate": float((arr > 0).mean()),
            "p25": float(np.quantile(arr, 0.25)), "p75": float(np.quantile(arr, 0.75)),
        }
        setup_summary[e] = s
        print(f"  {e:18s}: n={s['n']:3d} med={s['median']:+.4f} mean={s['mean']:+.4f} p10={s['p10']:+.4f} p90={s['p90']:+.4f} win={s['win_rate']:.2%}")
    results[setup_key] = setup_summary

OUT = ROOT / "runs/mover_day/pepe_train_mover_day_15m_exits_2026_05_27.json"
with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved -> {OUT}")
