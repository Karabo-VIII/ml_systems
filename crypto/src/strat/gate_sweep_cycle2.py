"""
gate_sweep_cycle2.py -- Regime-gate sweep on mom14-K5 engine over 2020-2026.

LANE: Bear is dead-beta (selection p=1.000 in Cycle 1). Only lever = EXPOSURE via gate.
Sweep on the Cycle-1 winner: mom14 K5 rebal=3d (as in selftest reference).

Gates tested:
  A. Per-asset SMA gate: sma200 / sma100 / sma50
  B. BTC-market gate: flatten whole book when BTC < its SMA-N (sma200/sma100/sma50)
  C. Breadth gate: scale exposure by fraction of universe above SMA200 (thresholds 0.3 / 0.5)
  D. Vol-target overlay: scale position by TARGET_VOL / realized_vol (target=0.20/0.30)
  E. Combinations: per-asset sma200 + BTC gate

Metric: per-year compound (2020..2025) + full, maxDD, avg_expo.
NO look-ahead. All gates use only data available at the time.

Run: python -m strat.gate_sweep_cycle2   (from crypto/src/)
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")
np.seterr(invalid="ignore", divide="ignore")

import strat.mover_lab as ml

# ── load full history 2020-2026 ──────────────────────────────────────────────
print("[gate_sweep] loading data 2020-01-01 -> 2026-06-01 ...")
ind = ml.load("2020-01-01", "2026-06-01")
C   = ind["C"]

# ── pre-compute extra SMAs we need ──────────────────────────────────────────
sma100 = C.rolling(100, min_periods=100).mean()
sma50  = C.rolling(50,  min_periods=50).mean()
sma200 = ind["sma200"]   # already computed

# Per-asset gate DataFrames (True = allowed to hold)
gate_sma200 = (C > sma200).fillna(False)
gate_sma100 = (C > sma100).fillna(False)
gate_sma50  = (C > sma50).fillna(False)

# BTC-only gates (flatten ALL when BTC below its SMA-N)
btc = "BTCUSDT"
btc_above_200 = (C[btc] > sma200[btc]).fillna(False)
btc_above_100 = (C[btc] > sma100[btc]).fillna(False)
btc_above_50  = (C[btc] > sma50[btc]).fillna(False)


def apply_btc_gate(W: pd.DataFrame, btc_mask: pd.Series) -> pd.DataFrame:
    """Zero entire book on days when BTC-gate is False."""
    return W.multiply(btc_mask.astype(float), axis=0)


def apply_breadth_gate(W: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Scale book by breadth fraction (clamp exposure at 0 below threshold)."""
    breadth = gate_sma200.astype(float).mean(axis=1)   # fraction above sma200
    scale   = (breadth >= threshold).astype(float) * breadth
    return W.multiply(scale, axis=0)


def apply_vol_target(W: pd.DataFrame, target_vol: float, window: int = 20) -> pd.DataFrame:
    """Scale book so realized port vol ~ target_vol.  Cap scale at 1 (no leverage)."""
    R   = ind["R"].reindex(index=W.index, columns=W.columns).fillna(0.0)
    pos = W.shift(1).fillna(0.0)
    port_ret = (pos * R).sum(axis=1)
    realvol  = port_ret.rolling(window, min_periods=5).std() * np.sqrt(365)
    scale    = (target_vol / realvol.replace(0, np.nan)).fillna(1.0).clip(upper=1.0)
    return W.multiply(scale, axis=0)


def build_mom14_k5(gate_df: pd.DataFrame | None = None, rebal: int = 3) -> pd.DataFrame:
    """mom14 K5 strategy with a custom per-asset gate DataFrame (or None = no gate)."""
    score = ind["mom14"]
    if gate_df is None:
        gate_df = pd.DataFrame(True, index=C.index, columns=C.columns)
    # replicate topk_weight but with a custom gate
    K = 5
    W = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    last = -999
    for i, d in enumerate(C.index):
        if i - last >= rebal:
            elig = [s for s in C.columns if bool(gate_df.loc[d, s]) and pd.notna(score.loc[d, s])]
            if elig:
                pick = sorted(elig, key=lambda s: -score.loc[d, s])[:K]
                W.loc[d, :] = 0.0
                for s in pick:
                    W.loc[d, s] = 1.0 / len(pick)
            last = i
        elif i > 0:
            W.iloc[i] = W.iloc[i - 1]
    return W


def evaluate_full(W: pd.DataFrame, label: str) -> dict:
    R = ind["R"].reindex(index=W.index, columns=W.columns).fillna(0.0)
    pos = W.shift(1).fillna(0.0)
    from strat.mover_lab import COST
    turn = pos.diff().abs().fillna(pos.abs()).sum(axis=1)
    bret = (pos * R).sum(axis=1) - turn * (COST / 2.0)
    x    = bret.to_numpy()
    eq   = np.cumprod(1 + x)
    pk   = np.maximum.accumulate(eq)
    maxdd = round(float(((eq - pk) / pk).min() * 100), 1)
    idx   = bret.index

    def comp(s, e):
        mask = (idx >= s) & (idx < e)
        xs   = x[np.asarray(mask)]
        if mask.sum() < 2:
            return None
        return round((float(np.prod(1 + xs)) - 1) * 100, 1)

    avg_expo = round(float(pos.sum(axis=1).mean()), 2)
    return {
        "label":      label,
        "2020":       comp("2020-01-01", "2021-01-01"),
        "2021":       comp("2021-01-01", "2022-01-01"),
        "2022":       comp("2022-01-01", "2023-01-01"),
        "2023":       comp("2023-01-01", "2024-01-01"),
        "2024":       comp("2024-01-01", "2025-01-01"),
        "2025":       comp("2025-01-01", "2026-01-01"),
        "full":       comp("2020-01-01", "2026-06-01"),
        "maxDD":      maxdd,
        "avg_expo":   avg_expo,
    }


# ── build all variants ───────────────────────────────────────────────────────
results = []

print("[gate_sweep] building variants ...")

# 0. Baseline: no gate (raw mom14 K5)
W_base = build_mom14_k5(gate_df=None)
results.append(evaluate_full(W_base, "0.base_no_gate"))

# A. Per-asset SMA gates
for name, gdf in [("sma200", gate_sma200), ("sma100", gate_sma100), ("sma50", gate_sma50)]:
    W = build_mom14_k5(gate_df=gdf)
    results.append(evaluate_full(W, f"A.per_asset_{name}"))
print("  A done")

# B. BTC market gate (whole-book off when BTC < its SMA)
W_a200 = build_mom14_k5(gate_df=gate_sma200)  # per-asset sma200 base
for name, btc_mask in [("btc_sma200", btc_above_200),
                        ("btc_sma100", btc_above_100),
                        ("btc_sma50",  btc_above_50)]:
    # BTC gate applied ON TOP of no per-asset gate
    W_b = apply_btc_gate(W_base.copy(), btc_mask)
    results.append(evaluate_full(W_b, f"B.nogate+{name}"))
    # Also on top of per-asset sma200
    W_b2 = apply_btc_gate(W_a200.copy(), btc_mask)
    results.append(evaluate_full(W_b2, f"B.pasma200+{name}"))
print("  B done")

# C. Breadth gate (scale by % above sma200)
for thr in [0.0, 0.3, 0.5]:
    W_c = apply_breadth_gate(W_base.copy(), threshold=thr)
    results.append(evaluate_full(W_c, f"C.breadth_thr{thr}"))
    W_c2 = apply_breadth_gate(W_a200.copy(), threshold=thr)
    results.append(evaluate_full(W_c2, f"C.pasma200+breadth_thr{thr}"))
print("  C done")

# D. Vol-target overlay (cap at 1, no leverage)
for tv in [0.20, 0.30]:
    W_d = apply_vol_target(W_base.copy(), target_vol=tv)
    results.append(evaluate_full(W_d, f"D.voltgt{int(tv*100)}pct"))
    W_d2 = apply_vol_target(W_a200.copy(), target_vol=tv)
    results.append(evaluate_full(W_d2, f"D.pasma200+voltgt{int(tv*100)}pct"))
print("  D done")

# E. Best combo candidates: per-asset sma200 + BTC sma200 + breadth
W_e1 = apply_btc_gate(W_a200.copy(), btc_above_200)
results.append(evaluate_full(W_e1, "E.pasma200+btcsma200"))

W_e2 = apply_breadth_gate(apply_btc_gate(W_a200.copy(), btc_above_200), threshold=0.3)
results.append(evaluate_full(W_e2, "E.pasma200+btcsma200+breadth0.3"))

W_e3 = apply_vol_target(apply_btc_gate(W_a200.copy(), btc_above_200), target_vol=0.30)
results.append(evaluate_full(W_e3, "E.pasma200+btcsma200+voltgt30"))
print("  E done")

# ── print results ────────────────────────────────────────────────────────────
df = pd.DataFrame(results).set_index("label")

# sort by full-cycle compound descending
df_sorted = df.sort_values("full", ascending=False)

print("\n=== REGIME GATE SWEEP: mom14-K5, 2020-2026 ===")
header = f"{'Label':<42} {'2020':>7} {'2021':>7} {'2022':>7} {'2023':>7} {'2024':>7} {'2025':>7} {'full':>8} {'maxDD':>7} {'expo':>5}"
print(header)
print("-" * len(header))
for lbl, row in df_sorted.iterrows():
    def fmt(v): return f"{v:>7.1f}" if v is not None else f"{'N/A':>7}"
    print(f"{lbl:<42} {fmt(row['2020'])} {fmt(row['2021'])} {fmt(row['2022'])} {fmt(row['2023'])} {fmt(row['2024'])} {fmt(row['2025'])} {fmt(row['full'])} {fmt(row['maxDD'])} {row['avg_expo']:>5.2f}")

# highlight the best on key criteria
print("\n--- ANALYSIS ---")
best_2022 = df_sorted["2022"].idxmax()
best_full  = df_sorted["full"].idxmax()
least_dd   = df_sorted["maxDD"].idxmax()  # maxDD is negative so idxmax = least negative
print(f"Best 2022 (bear protection): {best_2022}  => {df_sorted.loc[best_2022,'2022']:.1f}%")
print(f"Best full-cycle compound:    {best_full}  => {df_sorted.loc[best_full,'full']:.1f}%")
print(f"Least bad maxDD:             {least_dd}   => {df_sorted.loc[least_dd,'maxDD']:.1f}%")

baseline_2022 = df_sorted.loc["A.per_asset_sma200", "2022"] if "A.per_asset_sma200" in df_sorted.index else None
baseline_full  = df_sorted.loc["A.per_asset_sma200", "full"] if "A.per_asset_sma200" in df_sorted.index else None
print(f"\nBaseline (per-asset sma200): 2022={baseline_2022:.1f}%  full={baseline_full:.1f}%")
print(f"Cycle-1 reported: 2022 = -55.0%  (shuffle p=0.010 genuine)")

# save
import json
out_path = Path(__file__).resolve().parents[2] / "runs" / "strat" / "gate_sweep_cycle2.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[gate_sweep] results saved -> {out_path}")
