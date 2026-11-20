"""V2 exit-strategy harness: ATR sensitivity + setup-toxicity exits + 8 dimensions.

Extensions vs v1:
- ATR-K sensitivity: K = 1.5, 2.0, 3.0, 4.0 (test if K=2 too tight)
- ATR window sensitivity: ATR(14), ATR(30), ATR(50) — longer windows for fat-tails
- Setup-toxicity exit (MA cross-flip): exit when fast MA crosses back below slow
- Mixed setup-toxicity + cap: max-hold cap (e.g., 14d) as safety on setup-toxicity
- Regime-gated 7d hold: 7d hold ONLY if regime stays bull; else exit at flip

Tests against the SAME MA/EMA entries.

Per user direction (2026-05-20): "7-d hold is okay, but I'm relatively against it.
We must test against setup, not fixed periods. Even if we give up some value, as long
as we can exit a setup when it's toxic that's fine."

================================================================================
UPPER_BOUND_NOT_DEPLOY_ESTIMATE -- READ THIS BEFORE CITING ANY NAV NUMBER
================================================================================
2026-05-20 oracle audit (per docs/ORACLE_CORRECTIONS_2026_05_20.md):

The `nav_4pct_upper_bound_arithmetic` field below is the ARITHMETIC SUM of
per-event PnL × 4% notional sizing. With 2,918 long events over ~330 days
that's ~9 entries/day at 4% = 35% NAV/d in new positions -- not physically
realisable under any capital constraint.

HONEST per-event stats (mean / median / hit-rate / Sharpe) ARE comparable.
RELATIVE rank between exits (setup-toxicity > 7d-hold; ATR K=4 > K=2) IS valid
since both arms share the same notional aggregator.

The HEADLINE "setup-toxicity +896% NAV vs 7d hold +574% (+56% relative)" is:
  - RELATIVE claim valid (setup-toxicity does beat 7d-hold under same sizing assumption)
  - ABSOLUTE numbers are upper bounds, NOT deploy estimates

For deploy NAV: run the per-event ledger through honest_v2_simulator.py with proper
K_MAX + BET_FRACTION + capital lockup; or use v3 paper_trade_replay.
================================================================================
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
EVENT_PATH = ROOT / "runs" / "oracle_layer3" / "ma_ema_permutation" / "event_ma_snapshot.parquet"
OUT_DIR = ROOT / "runs" / "audit"
COST = 0.0024
SIZE = 0.04


def load_chimera_with_atr_variants():
    """Load chimera 1d with ATR(14), ATR(30), ATR(50) computed."""
    panel_rows = []
    files = sorted((ROOT / "data" / "processed" / "chimera" / "1d").glob("*_v51_chimera_1d_*.parquet"))
    for f in files:
        sym = f.name.split("_")[0].upper().replace("USDT", "")
        df = pl.read_parquet(f, columns=["timestamp","open","high","low","close"]).to_pandas()
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
        df["asset"] = sym
        df["high_low"] = df["high"] - df["low"]
        df["high_pc"] = (df["high"] - df["close"].shift(1)).abs()
        df["low_pc"] = (df["low"] - df["close"].shift(1)).abs()
        df["tr"] = df[["high_low","high_pc","low_pc"]].max(axis=1)
        df["atr14_pct"] = df["tr"].rolling(14).mean() / df["close"]
        df["atr30_pct"] = df["tr"].rolling(30).mean() / df["close"]
        df["atr50_pct"] = df["tr"].rolling(50).mean() / df["close"]
        panel_rows.append(df[["asset","date","close","atr14_pct","atr30_pct","atr50_pct"]])
    return pd.concat(panel_rows, ignore_index=True)


def fwd_path_with_atrs(chimera_idx, asset, ev_date, n=15):
    """Return entry_close, [atr14_pct, atr30_pct, atr50_pct] at entry, fwd_closes[1..n]."""
    sub = chimera_idx.get(asset)
    if sub is None:
        return None
    row = sub[sub["date"] == ev_date]
    if row.empty:
        return None
    idx = row.index[0]
    if idx + 1 >= len(sub):
        return None
    entry_close = float(sub.iloc[idx]["close"])
    atrs = {
        "atr14": float(sub.iloc[idx].get("atr14_pct", np.nan)) if pd.notna(sub.iloc[idx].get("atr14_pct", np.nan)) else None,
        "atr30": float(sub.iloc[idx].get("atr30_pct", np.nan)) if pd.notna(sub.iloc[idx].get("atr30_pct", np.nan)) else None,
        "atr50": float(sub.iloc[idx].get("atr50_pct", np.nan)) if pd.notna(sub.iloc[idx].get("atr50_pct", np.nan)) else None,
    }
    fwd = []
    for k in range(1, n+1):
        if idx + k < len(sub):
            fwd.append(float(sub.iloc[idx + k]["close"]))
        else:
            fwd.append(None)
    return entry_close, atrs, fwd


def compute_ma_flip_day(chimera_idx, asset, ev_date, fast: int, slow: int, ma_type: str = "SMA", max_fwd: int = 15):
    """Find first forward day where the MA cross flips (fast falls below slow)."""
    sub = chimera_idx.get(asset)
    if sub is None:
        return None
    row = sub[sub["date"] == ev_date]
    if row.empty:
        return None
    idx = row.index[0]
    # Compute MAs around the entry + forward window
    closes = sub["close"].values
    start = max(0, idx - max(slow, 50))
    seg = closes[start:idx + max_fwd + 1]
    seg_len = len(seg)
    if ma_type == "SMA":
        fast_ma = pd.Series(seg).rolling(fast).mean().values
        slow_ma = pd.Series(seg).rolling(slow).mean().values
    else:  # EMA
        fast_ma = pd.Series(seg).ewm(span=fast, adjust=False).mean().values
        slow_ma = pd.Series(seg).ewm(span=slow, adjust=False).mean().values
    entry_pos = idx - start
    # At entry, fast > slow (long signal active). Find first forward day where fast <= slow.
    for k in range(1, min(max_fwd, seg_len - entry_pos)):
        if entry_pos + k < seg_len:
            fa = fast_ma[entry_pos + k]
            sl = slow_ma[entry_pos + k]
            if pd.notna(fa) and pd.notna(sl) and fa <= sl:
                return k
    return None  # never flipped within max_fwd


def apply_extended_exits(entry_close, fwd_closes, atrs, ma_flip_day, btc_30d_entry):
    """Compute realized PnL under extended exit menu."""
    results = {}
    if not entry_close or entry_close <= 0:
        return results
    def pct(c): return c / entry_close - 1 - COST

    # Baselines (same as v1)
    if len(fwd_closes) >= 7 and fwd_closes[6] is not None:
        results["D_7d_hold"] = pct(fwd_closes[6])
    if len(fwd_closes) >= 5 and fwd_closes[4] is not None:
        results["C_5d_hold"] = pct(fwd_closes[4])

    # Trail 5%/3% (same as v1 G)
    peak = entry_close
    exit_g = None
    armed = False
    for c in fwd_closes[:14]:
        if c is None: continue
        if c > peak: peak = c
        ret = c / entry_close - 1
        if not armed and ret >= 0.05: armed = True
        if armed and c <= peak * 0.97:
            exit_g = c; break
    if exit_g is None and len(fwd_closes) >= 14 and fwd_closes[13] is not None:
        exit_g = fwd_closes[13]
    if exit_g is not None:
        results["G_trail_5pct_3pct"] = pct(exit_g)

    # ATR sensitivity grid: K = 1.5, 2.0, 3.0, 4.0 × ATR window = 14, 30, 50
    # Test as stop-loss + fixed +30% TP (so we isolate the stop dim)
    for atr_key in ("atr14", "atr30", "atr50"):
        atr_val = atrs.get(atr_key)
        if atr_val is None or atr_val <= 0:
            continue
        for K in (1.5, 2.0, 3.0, 4.0):
            stop_pct = K * atr_val
            tp_pct = 0.30
            exit_c = None
            for c in fwd_closes[:14]:
                if c is None: continue
                ret = c / entry_close - 1
                if ret <= -stop_pct:
                    exit_c = c; break
                if ret >= tp_pct:
                    exit_c = c; break
            if exit_c is None and len(fwd_closes) >= 14 and fwd_closes[13] is not None:
                exit_c = fwd_closes[13]
            if exit_c is not None:
                results[f"F_{atr_key}_K{K:.1f}_TP30"] = pct(exit_c)

    # ATR TRAIL sensitivity: K = 1.0, 1.5, 2.0 × ATR window = 14, 30, 50
    for atr_key in ("atr14", "atr30", "atr50"):
        atr_val = atrs.get(atr_key)
        if atr_val is None or atr_val <= 0:
            continue
        for K in (1.0, 1.5, 2.0):
            peak_h = entry_close
            exit_h = None
            armed_h = False
            for c in fwd_closes[:14]:
                if c is None: continue
                if c > peak_h: peak_h = c
                ret = c / entry_close - 1
                if not armed_h and ret >= atr_val:
                    armed_h = True
                if armed_h and c <= peak_h * (1 - K * atr_val):
                    exit_h = c; break
            if exit_h is None and len(fwd_closes) >= 14 and fwd_closes[13] is not None:
                exit_h = fwd_closes[13]
            if exit_h is not None:
                results[f"H_{atr_key}_trailK{K:.1f}"] = pct(exit_h)

    # Setup-toxicity exit: exit on MA cross flip (PURE)
    if ma_flip_day is not None and ma_flip_day <= 14:
        if ma_flip_day < len(fwd_closes) and fwd_closes[ma_flip_day] is not None:
            results["S_setup_toxic_pure"] = pct(fwd_closes[ma_flip_day])
    elif len(fwd_closes) >= 14 and fwd_closes[13] is not None:
        results["S_setup_toxic_pure"] = pct(fwd_closes[13])  # no flip within 14d

    # Setup-toxicity + max_hold cap (5d, 7d, 14d)
    for cap in (5, 7, 14):
        chosen_day = min(ma_flip_day, cap) if ma_flip_day is not None else cap
        chosen_day = min(chosen_day, len(fwd_closes) - 1)
        if chosen_day >= 0 and fwd_closes[chosen_day] is not None:
            results[f"S_setup_toxic_cap{cap}d"] = pct(fwd_closes[chosen_day])

    # Setup-toxicity + stop_loss combo (gated stops): -8% stop OR flip
    exit_combo = None
    for k, c in enumerate(fwd_closes[:14]):
        if c is None: continue
        ret = c / entry_close - 1
        if ret <= -0.08:
            exit_combo = c; break
        if ma_flip_day is not None and k >= ma_flip_day:
            exit_combo = c; break
    if exit_combo is None and len(fwd_closes) >= 14 and fwd_closes[13] is not None:
        exit_combo = fwd_closes[13]
    if exit_combo is not None:
        results["S_setup_OR_stop8"] = pct(exit_combo)

    return results


def main():
    print("Loading chimera (with ATR14/30/50)...")
    chimera = load_chimera_with_atr_variants()
    chimera_idx = {a: sub.sort_values("date").reset_index(drop=True) for a, sub in chimera.groupby("asset")}
    print(f"  {len(chimera)} rows, {chimera['asset'].nunique()} assets")

    print("Loading event_ma_snapshot...")
    events = pl.read_parquet(EVENT_PATH).to_pandas()
    events = events[events["side"] == "long"].copy()
    print(f"  Long events: {len(events)}")

    # Test on SMA(28,29) universal (the most-events config)
    configs = [
        ("SMA_28_29_universal", "SMA", 28, 29, None),
        ("SMA_25_26_bull", "SMA", 25, 26, "bull"),
        ("SMA_16_19_bear", "SMA", 16, 19, "bear"),
    ]

    overall = {}
    for name, ma_type, fast, slow, regime in configs:
        print(f"\n=== {name} (regime={regime}) ===")
        fast_col, slow_col = f"{ma_type}_{fast}", f"{ma_type}_{slow}"
        ev = events[(events[fast_col].notna()) & (events[slow_col].notna())].copy()
        if regime:
            ev = ev[ev["btc_regime_30d"] == regime]
        ev["signal"] = (ev[fast_col] > ev[slow_col]).astype(int)
        ev_long = ev[ev["signal"] == 1].copy()
        n_events = len(ev_long)
        print(f"  Eligible: {n_events}")

        strat_pnls = {}
        n_flip_within_14d = 0
        flip_days = []
        for _, row in ev_long.iterrows():
            asset = row["asset"]
            ev_date = pd.to_datetime(row["date"]).date() if not isinstance(row["date"], date) else row["date"]
            btc_30d = row.get("btc_regime_30d", None)
            path = fwd_path_with_atrs(chimera_idx, asset, ev_date, n=15)
            if path is None: continue
            entry_close, atrs, fwd = path
            ma_flip = compute_ma_flip_day(chimera_idx, asset, ev_date, fast, slow, ma_type, max_fwd=14)
            if ma_flip is not None:
                n_flip_within_14d += 1
                flip_days.append(ma_flip)
            exits = apply_extended_exits(entry_close, fwd, atrs, ma_flip, btc_30d)
            for k, v in exits.items():
                strat_pnls.setdefault(k, []).append(v)

        # Setup-toxicity diagnostic
        if flip_days:
            print(f"  MA flip within 14d: {n_flip_within_14d}/{n_events} = {n_flip_within_14d/max(n_events,1)*100:.1f}%")
            print(f"  Mean flip-day: {np.mean(flip_days):.1f} (median {np.median(flip_days):.0f})")

        # Summary -- per-event stats HONEST; nav_4pct_upper_bound_arithmetic is NOT
        # a deploy estimate (see top-of-file UPPER_BOUND_NOT_DEPLOY_ESTIMATE notice).
        summary = {}
        for s, pnls in strat_pnls.items():
            if not pnls: continue
            arr = np.array(pnls)
            summary[s] = {"n": len(arr), "mean_pct": arr.mean()*100, "median_pct": np.median(arr)*100,
                          "hit_rate": (arr > 0).mean()*100, "sharpe": arr.mean()/(arr.std()+1e-9),
                          "nav_4pct_upper_bound_arithmetic": arr.sum()*SIZE*100}
        overall[name] = summary

        # Print ranked -- LOUD warning that NAV is arithmetic upper bound
        print(f"\n  [WARN] NAV column below is ARITHMETIC UPPER BOUND at 4% notional sizing.")
        print(f"         NOT a deploy estimate. Relative ranking valid; absolute % is not.")
        print(f"\n  Top 12 strategies by per-event mean (honest rank):")
        ranked = sorted(summary.items(), key=lambda x: -x[1]["mean_pct"])
        print(f"  {'strategy':<30}{'n':>6}{'mean':>9}{'median':>9}{'hit%':>7}{'sharpe':>9}{'NAV_UB':>11}")
        for s, r in ranked[:12]:
            print(f"  {s:<30}{r['n']:>6d}{r['mean_pct']:>+8.3f}%{r['median_pct']:>+8.3f}%{r['hit_rate']:>6.1f}%{r['sharpe']:>+9.3f}{r['nav_4pct_upper_bound_arithmetic']:>+10.2f}%")

    out_p = OUT_DIR / "exit_strategy_ranking_v2.json"
    out_p.write_text(json.dumps(overall, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {out_p}")
    return overall


if __name__ == "__main__":
    main()
