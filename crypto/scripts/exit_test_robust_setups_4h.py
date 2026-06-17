"""Test 6 exits on TOP robust setups (fixed setup x all 186 mover days) @ 4h."""
import json
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "runs/mover_day/pepe_train_mover_day_4h_2026_05_27.json"
CHIM_4H = ROOT / "data/processed/chimera/4h/pepeusdt_v51_chimera_4h_20260522.parquet"

with open(DATA) as f:
    d = json.load(f)

df_4h = pl.read_parquet(CHIM_4H).sort("timestamp")
ts_4h = df_4h["timestamp"].to_numpy()
closes_4h = df_4h["close"].to_numpy()
opens_4h = df_4h["open"].to_numpy()
highs_4h = df_4h["high"].to_numpy()
n_train_4h = int(d["_meta"]["n_train_4h_bars"])
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


train_closes = closes_4h[:n_train_4h]
# Top robust + key candidates from analyze_pepe_mover_robust_4h.py
SETUPS = [
    ("WMA", 3, 9),     # 91.4% fire, 4.37% med, 80.0% win
    ("SMA", 7, 9),     # 84.4% fire, 4.55% med, 80.3% win - high win at high fire
    ("SMA", 5, 9),     # 87.6% fire, 4.16% med, 78.5% win
    ("SMA", 3, 9),     # 87.1% fire, 3.66% med, 79.0% win
    ("SMA", 15, 18),   # 58.1% fire, 4.82% med, 81.5% win - highest win but lower fire
    ("SMA", 9, 12),    # 74.2% fire, 4.26% med, 79.7% win
    ("SMA", 12, 18),   # 50.0% fire, 5.45% med, 77.4% win - highest median
]

CAP = 14 * 6  # 14 days = 84 4h bars
EXIT_HORIZONS = {"E2_6h": 2, "E3_12h": 3, "E4_24h": 6, "E5_48h": 12}

results = {}
for (kind, fast, slow) in SETUPS:
    f_ma = compute_ma(train_closes, kind, fast)
    s_ma = compute_ma(train_closes, kind, slow)
    exit_pnls = {"E1_opp_cross": [], "E2_6h": [], "E3_12h": [], "E4_24h": [], "E5_48h": [], "E6_mfe_trail50": []}
    n_entries = 0
    for m in d["movers"]:
        day_close_idx = int(np.searchsorted(ts_4h, m["ts_ms"], side="right") - 1)
        if day_close_idx < 0 or day_close_idx >= n_train_4h:
            continue
        ws = max(0, day_close_idx - (LOOKBACK - 1))
        ci = first_cross_up(f_ma, s_ma, ws + 1, day_close_idx + 1)
        if ci == -1 or ci + 1 > day_close_idx:
            continue
        ei = ci + 1
        if opens_4h[ei] <= 0:
            continue
        ep = opens_4h[ei]; n_entries += 1

        # E1: opposite cross
        e1_ex = None
        for i in range(ei, min(ei + CAP, n_train_4h)):
            if not np.isnan(f_ma[i]) and not np.isnan(s_ma[i]) and f_ma[i] <= s_ma[i]:
                e1_ex = i
                break
        if e1_ex is not None and e1_ex + 1 < n_train_4h:
            exit_pnls["E1_opp_cross"].append(opens_4h[e1_ex+1]/ep - 1)

        # E2-E5 fixed holds
        for key, k_bars in EXIT_HORIZONS.items():
            if ei + k_bars < n_train_4h:
                exit_pnls[key].append(opens_4h[ei + k_bars]/ep - 1)

        # E6 MFE-trail50
        peak_p = ep; e6 = None
        end_walk = min(ei + CAP, n_train_4h - 1)
        for i in range(ei + 1, end_walk + 1):
            peak_p = max(peak_p, highs_4h[i])
            peak_ret = peak_p/ep - 1
            if peak_ret >= 0.03:
                lock_p = ep*(1 + peak_ret*0.5)
                if closes_4h[i] <= lock_p:
                    if i + 1 < n_train_4h:
                        e6 = opens_4h[i+1]/ep - 1
                    else:
                        e6 = closes_4h[i]/ep - 1
                    break
        if e6 is None:
            e6 = closes_4h[end_walk]/ep - 1
        exit_pnls["E6_mfe_trail50"].append(e6)

    setup_key = f"{kind} {fast}/{slow}"
    print(f"\n=== {setup_key} ({n_entries} entries) ===")
    setup_summary = {"n_entries": n_entries}
    for e, vals in exit_pnls.items():
        if not vals:
            continue
        arr = np.array(vals)
        s = {
            "n": len(arr),
            "median": float(np.median(arr)),
            "mean": float(np.mean(arr)),
            "p10": float(np.quantile(arr, 0.10)),
            "p25": float(np.quantile(arr, 0.25)),
            "p75": float(np.quantile(arr, 0.75)),
            "p90": float(np.quantile(arr, 0.90)),
            "win_rate": float((arr > 0).mean()),
        }
        setup_summary[e] = s
        print(f"  {e:18s}: n={s['n']:3d} med={s['median']:+.4f} mean={s['mean']:+.4f} "
              f"p10={s['p10']:+.4f} p90={s['p90']:+.4f} win={s['win_rate']:.2%}")
    results[setup_key] = setup_summary

OUT = ROOT / "runs/mover_day/pepe_train_mover_day_4h_exits_2026_05_27.json"
with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved -> {OUT}")
