"""RSI exhaustion with multi-exit framework.

Two RSI signal classes tested:
  SIG_A — Oversold-bounce: RSI < threshold yesterday AND > threshold today (long entry)
  SIG_B — Momentum-start: RSI crosses up through 50

RSI periods × thresholds:
  periods:    7, 14, 21, 28
  os_thresh:  20, 25, 30, 35

For each config, apply the same multi-exit menu from v2:
  - Fixed-period holds (3d, 5d, 7d)
  - Setup-toxicity: exit when RSI crosses BACK below entry-threshold (oversold-bounce)
                    OR when RSI > 70 (overbought, take profit)
  - ATR variants (best from MA/EMA study: K=3-4 on ATR(30) or ATR(50))
  - Trailing stops

Output:
  runs/audit/rsi_exhaust_multi_exit.json  (full results)
  runs/audit/RSI_EXHAUST_2026_05_20.md   (summary)

================================================================================
UPPER_BOUND_NOT_DEPLOY_ESTIMATE -- READ THIS BEFORE CITING ANY NAV NUMBER
================================================================================
2026-05-20 oracle audit (per docs/ORACLE_CORRECTIONS_2026_05_20.md):

The `nav_4pct_upper_bound_arithmetic` field below is the ARITHMETIC SUM of
per-event PnL × 4% notional sizing. RSI events fire ~3/day × 4% = 12% NAV/d
in new positions -- not physically realisable under any capital constraint.

The headline "RSI(7) os=35 + trail 5%/3% +60% NAV" is an ARITHMETIC UPPER BOUND,
NOT a deploy estimate. The honest per-event mean (~0.3-0.5%) and hit-rate (~50%)
are the useful characterisation. RSI is 10-15x weaker than MA cross at the
PER-EVENT level -- that finding stands.

For deploy: feed the per-event ledger through honest_v2_simulator.py with proper
capacity constraints; or use v3 paper_trade_replay.
================================================================================
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "audit"
COST = 0.0024
SIZE = 0.04


def rsi(closes: np.ndarray, period: int) -> np.ndarray:
    """Wilder's RSI (standard)."""
    d = np.diff(closes, prepend=closes[0])
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    # Wilder smoothing
    up_s = pd.Series(up).ewm(alpha=1/period, adjust=False).mean().values
    dn_s = pd.Series(dn).ewm(alpha=1/period, adjust=False).mean().values
    rs = up_s / (dn_s + 1e-12)
    return 100 - 100 / (1 + rs)


def load_chimera_with_atr():
    """Load chimera 1d with ATR + RSI panels precomputed."""
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
        df["atr30_pct"] = df["tr"].rolling(30).mean() / df["close"]
        df["atr50_pct"] = df["tr"].rolling(50).mean() / df["close"]
        # Precompute RSI for 4 periods
        closes = df["close"].values
        for p in (7, 14, 21, 28):
            df[f"rsi_{p}"] = rsi(closes, p)
        panel_rows.append(df[["asset","date","close","atr30_pct","atr50_pct"] + [f"rsi_{p}" for p in (7,14,21,28)]])
    return pd.concat(panel_rows, ignore_index=True)


def find_rsi_oversold_bounce_events(panel: pd.DataFrame, rsi_period: int, threshold: float,
                                     date_start, date_end):
    """Find (asset, date) where RSI was < threshold yesterday AND > threshold today (long entry)."""
    rsi_col = f"rsi_{rsi_period}"
    events = []
    for asset, sub in panel.groupby("asset"):
        sub = sub.sort_values("date").reset_index(drop=True)
        sub["rsi_prev"] = sub[rsi_col].shift(1)
        crosses = sub[(sub["rsi_prev"] < threshold) & (sub[rsi_col] >= threshold)]
        crosses = crosses[(crosses["date"] >= date_start) & (crosses["date"] <= date_end)]
        for _, row in crosses.iterrows():
            events.append({"asset": asset, "date": row["date"], "entry_close": row["close"],
                            "atr30_pct": row["atr30_pct"], "atr50_pct": row["atr50_pct"],
                            "rsi_at_entry": row[rsi_col]})
    return events


def apply_rsi_exits(panel_idx, asset, ev_date, entry_close, atrs, rsi_period, os_threshold, n=15):
    """Compute realized PnL under each exit strategy for an RSI bounce entry."""
    sub = panel_idx.get(asset)
    if sub is None:
        return {}
    row = sub[sub["date"] == ev_date]
    if row.empty:
        return {}
    idx = row.index[0]
    if idx + 1 >= len(sub):
        return {}

    # Forward closes + forward RSI
    fwd_closes = []
    fwd_rsi = []
    rsi_col = f"rsi_{rsi_period}"
    for k in range(1, n + 1):
        if idx + k < len(sub):
            fwd_closes.append(float(sub.iloc[idx + k]["close"]))
            fwd_rsi.append(float(sub.iloc[idx + k][rsi_col]) if pd.notna(sub.iloc[idx + k][rsi_col]) else None)
        else:
            fwd_closes.append(None)
            fwd_rsi.append(None)

    results = {}
    def pct(c): return c / entry_close - 1 - COST

    # Fixed-period holds
    if len(fwd_closes) >= 3 and fwd_closes[2] is not None: results["B_3d"] = pct(fwd_closes[2])
    if len(fwd_closes) >= 5 and fwd_closes[4] is not None: results["C_5d"] = pct(fwd_closes[4])
    if len(fwd_closes) >= 7 and fwd_closes[6] is not None: results["D_7d"] = pct(fwd_closes[6])

    # Setup-toxicity (RSI crosses BACK below entry threshold OR > 70)
    exit_s = None
    for k, (c, r) in enumerate(zip(fwd_closes[:14], fwd_rsi[:14])):
        if c is None: continue
        if r is not None and (r <= os_threshold or r >= 70):
            exit_s = c; break
    if exit_s is None and len(fwd_closes) >= 14 and fwd_closes[13] is not None:
        exit_s = fwd_closes[13]
    if exit_s is not None:
        results["S_setup_toxic"] = pct(exit_s)

    # Setup-toxicity OR -8% stop
    exit_so = None
    for k, (c, r) in enumerate(zip(fwd_closes[:14], fwd_rsi[:14])):
        if c is None: continue
        if c / entry_close - 1 <= -0.08:
            exit_so = c; break
        if r is not None and (r <= os_threshold or r >= 70):
            exit_so = c; break
    if exit_so is None and len(fwd_closes) >= 14 and fwd_closes[13] is not None:
        exit_so = fwd_closes[13]
    if exit_so is not None:
        results["S_setup_OR_stop8"] = pct(exit_so)

    # ATR variants (best from MA/EMA study: K=4 × ATR(30) or ATR(50))
    for atr_key in ("atr30", "atr50"):
        atr_val = atrs.get(atr_key)
        if atr_val is None or atr_val <= 0: continue
        for K in (3.0, 4.0):
            stop_pct = K * atr_val
            tp_pct = 0.30
            exit_f = None
            for c in fwd_closes[:14]:
                if c is None: continue
                ret = c / entry_close - 1
                if ret <= -stop_pct: exit_f = c; break
                if ret >= tp_pct: exit_f = c; break
            if exit_f is None and len(fwd_closes) >= 14 and fwd_closes[13] is not None:
                exit_f = fwd_closes[13]
            if exit_f is not None:
                results[f"F_{atr_key}_K{K:.0f}_TP30"] = pct(exit_f)

    # Trail 5%/3%
    peak = entry_close
    exit_g = None
    armed = False
    for c in fwd_closes[:14]:
        if c is None: continue
        if c > peak: peak = c
        ret = c / entry_close - 1
        if not armed and ret >= 0.05: armed = True
        if armed and c <= peak * 0.97: exit_g = c; break
    if exit_g is None and len(fwd_closes) >= 14 and fwd_closes[13] is not None:
        exit_g = fwd_closes[13]
    if exit_g is not None:
        results["G_trail_5pct_3pct"] = pct(exit_g)

    return results


def main():
    print("Loading chimera with ATR + RSI panels...")
    panel = load_chimera_with_atr()
    panel_idx = {a: sub.sort_values("date").reset_index(drop=True) for a, sub in panel.groupby("asset")}
    print(f"  {len(panel)} rows, {panel['asset'].nunique()} assets")

    DATE_START = date(2023, 7, 1)
    DATE_END = date(2024, 5, 15)

    configs = [(p, t) for p in (7, 14, 21, 28) for t in (20, 25, 30, 35)]
    all_results = {}
    for rsi_period, os_threshold in configs:
        config_name = f"RSI_{rsi_period}_os_{os_threshold}"
        print(f"\n=== {config_name} ===")
        events = find_rsi_oversold_bounce_events(panel, rsi_period, os_threshold, DATE_START, DATE_END)
        print(f"  events: {len(events)}")
        if len(events) < 100:
            print(f"  TOO FEW EVENTS — skip")
            continue

        strat_pnls = {}
        for ev in events:
            atrs = {"atr30": ev["atr30_pct"], "atr50": ev["atr50_pct"]}
            exits = apply_rsi_exits(panel_idx, ev["asset"], ev["date"], ev["entry_close"],
                                     atrs, rsi_period, os_threshold)
            for k, v in exits.items():
                strat_pnls.setdefault(k, []).append(v)

        # Per-event stats HONEST; nav_4pct_upper_bound_arithmetic is NOT a deploy
        # estimate (see top-of-file UPPER_BOUND_NOT_DEPLOY_ESTIMATE notice).
        summary = {}
        for s, pnls in strat_pnls.items():
            arr = np.array(pnls)
            summary[s] = {"n": len(arr), "mean_pct": arr.mean()*100, "median_pct": np.median(arr)*100,
                          "hit_rate": (arr > 0).mean()*100, "sharpe": arr.mean()/(arr.std()+1e-9),
                          "nav_4pct_upper_bound_arithmetic": arr.sum()*SIZE*100}
        all_results[config_name] = {"n_events": len(events), "summary": summary}

        # LOUD upper-bound warning (per oracle-corrections 2026-05-20)
        print(f"  [WARN] NAV_UB is ARITHMETIC UPPER BOUND at 4% notional sizing -- NOT deploy.")
        print(f"  Top 8 exits by per-event mean (honest rank):")
        ranked = sorted(summary.items(), key=lambda x: -x[1]["mean_pct"])
        print(f"  {'strategy':<22}{'n':>6}{'mean':>9}{'median':>9}{'hit%':>7}{'sharpe':>9}{'NAV_UB':>11}")
        for s, r in ranked[:8]:
            print(f"  {s:<22}{r['n']:>6d}{r['mean_pct']:>+8.3f}%{r['median_pct']:>+8.3f}%{r['hit_rate']:>6.1f}%{r['sharpe']:>+9.3f}{r['nav_4pct_upper_bound_arithmetic']:>+10.2f}%")

    out_p = OUT_DIR / "rsi_exhaust_multi_exit.json"
    out_p.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {out_p}")


if __name__ == "__main__":
    main()
