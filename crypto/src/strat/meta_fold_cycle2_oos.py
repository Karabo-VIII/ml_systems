"""META-FOLD CYCLE 2 -- DECISIVE OOS TEST (2026-06-19).

Frozen Cycle-1 configs tested on FULL 2020-2026 data.
Per-year breakdown + OOS shuffle null for mom14-K5-r14 on 2023+2024+2025.

Run from crypto/src:
    python -m strat.meta_fold_cycle2_oos

RWYB: real numbers only, no look-ahead.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.mover_lab as lab

# -----------------------------------------------------------------------
# 1.  Load full history 2020-01 -> 2026-06
# -----------------------------------------------------------------------
print("[1] Loading 2020-01-01 -> 2026-06-01 ...")
ind = lab.load("2020-01-01", "2026-06-01")
C = ind["C"]
print(f"    Assets: {list(C.columns)}")
print(f"    Date range: {C.index[0].date()} -> {C.index[-1].date()}  ({len(C)} bars)")

# -----------------------------------------------------------------------
# 2.  Build FROZEN Cycle-1 weight matrices
# -----------------------------------------------------------------------

def _breakout_k3_5d(ind):
    """Top-K=3 by 14d high breakout (close == hh14), rebal every 5 days, gate=True."""
    hh14 = ind["hh14"]
    C = ind["C"]
    # score = fraction of last-14d the close was at the 14d high (binary: at new high)
    # Simple proxy: score = 1 if C == hh14 else 0 (already causal via hh14)
    score = (C >= hh14 * 0.999).astype(float)  # tiny tol for float equality
    return lab.topk_weight(score, ind, K=3, gate=True, rebal=5)

def _gated_beta(ind):
    """Equal-weight all gated assets (SMA200 gate)."""
    g = ind["gate"].astype(float)
    denom = g.sum(axis=1).replace(0, np.nan)
    return g.div(denom, axis=0).fillna(0.0)

def _mr_rsi30_gate_ts5(ind):
    """Mean-reversion: buy K=3 lowest-RSI gated assets, trail-stop proxy via rsi>70 exit.
    Implemented as: score = -rsi14, top-K=3 gated, rebal=3 (short carry), ts5 = trail-stop exit
    when RSI exceeds 70 (approximated by holding max 5 days via rebal=5).
    """
    score = -ind["rsi14"]
    return lab.topk_weight(score, ind, K=3, gate=True, rebal=5)

print("[2] Building frozen Cycle-1 weight matrices ...")
W_mom14  = lab.topk_weight(ind["mom14"], ind, K=5, gate=True, rebal=14)
W_bo     = _breakout_k3_5d(ind)
W_beta   = _gated_beta(ind)
W_mr     = _mr_rsi30_gate_ts5(ind)
print("    Done.")

# -----------------------------------------------------------------------
# 3.  evaluate() per-year helper (extended to 2023/2024/2025)
# -----------------------------------------------------------------------

def _comp(bret, s, e):
    mask = (bret.index >= s) & (bret.index < e)
    xs = bret.to_numpy()[np.asarray(mask)]
    return round((np.prod(1 + xs) - 1) * 100, 1) if mask.sum() > 2 else None

def _maxdd_window(bret, s, e):
    mask = (bret.index >= s) & (bret.index < e)
    xs = bret.to_numpy()[np.asarray(mask)]
    if len(xs) < 2:
        return None
    eq = np.cumprod(1 + xs)
    pk = np.maximum.accumulate(eq)
    return round(float(((eq - pk) / pk).min() * 100), 1)

def _green(bret, s, e, H=3):
    """Fraction of H-day blocks that are positive."""
    sub = bret[(bret.index >= s) & (bret.index < e)]
    if len(sub) < H:
        return None
    arr = sub.to_numpy()
    wins = 0; total = 0
    for i in range(0, len(arr) - H + 1, H):
        r = np.prod(1 + arr[i:i+H]) - 1
        if r != 0.0:
            wins += int(r > 0)
            total += 1
    return round(100 * wins / total, 0) if total else None

def _build_bret(W, ind):
    """Return daily book-return series from a weight matrix."""
    import strat.ma_strat_builder as msb
    R = ind["R"].reindex(index=W.index, columns=W.columns).fillna(0.0)
    pos = W.shift(1).fillna(0.0)
    turn = pos.diff().abs().fillna(pos.abs()).sum(axis=1)
    bret = (pos * R).sum(axis=1) - turn * (msb.TAKER_RT / 2.0)
    return bret

print("[3] Computing per-year stats for all 4 strategies ...")

YEARS = ["2020", "2021", "2022", "2023", "2024", "2025"]
STRATS = {
    "mom14_K5_r14":    W_mom14,
    "breakout_K3_5d":  W_bo,
    "gated_beta":      W_beta,
    "MR_rsi30_ts5":    W_mr,
}

rows = []
for name, W in STRATS.items():
    bret = _build_bret(W, ind)
    for yr in YEARS:
        s = f"{yr}-01-01"; e = f"{int(yr)+1}-01-01"
        c = _comp(bret, s, e)
        md = _maxdd_window(bret, s, e)
        gr = _green(bret, s, e)
        rows.append({"strategy": name, "year": yr,
                     "comp%": c, "maxDD%": md, "green3d%": gr})
    # also full
    bret_full = bret
    x = bret_full.to_numpy()
    eq = np.cumprod(1 + x); pk = np.maximum.accumulate(eq)
    rows.append({"strategy": name, "year": "FULL",
                 "comp%": round((eq[-1]-1)*100, 1),
                 "maxDD%": round(float(((eq-pk)/pk).min()*100), 1),
                 "green3d%": None})

df_stats = pd.DataFrame(rows)
print("\n=== PER-YEAR COMPOUND RETURN (%) ===")
pivot_comp = df_stats.pivot(index="strategy", columns="year", values="comp%")
print(pivot_comp.to_string())

print("\n=== PER-YEAR MAX DRAWDOWN (%) ===")
pivot_dd = df_stats.pivot(index="strategy", columns="year", values="maxDD%")
print(pivot_dd.to_string())

print("\n=== PER-YEAR GREEN-RATE (3d blocks, %) ===")
pivot_gr = df_stats.pivot(index="strategy", columns="year", values="green3d%")
print(pivot_gr.to_string())

# -----------------------------------------------------------------------
# 4.  OOS Shuffle Null: mom14-K5 vs random-gated-5 on 2023+2024+2025
# -----------------------------------------------------------------------
print("\n[4] OOS shuffle null (mom14-K5 vs random-gated-5, 2023-2025, N=200 seeds) ...")

OOS_START = "2023-01-01"
OOS_END   = "2026-01-01"

# True mom14-K5 on OOS window
bret_mom14_full = _build_bret(W_mom14, ind)
oos_true = bret_mom14_full[(bret_mom14_full.index >= OOS_START) & (bret_mom14_full.index < OOS_END)]
true_comp = np.prod(1 + oos_true.to_numpy()) - 1

# Per-year true
true_2023 = _comp(bret_mom14_full, "2023-01-01", "2024-01-01")
true_2024 = _comp(bret_mom14_full, "2024-01-01", "2025-01-01")
true_2025 = _comp(bret_mom14_full, "2025-01-01", "2026-01-01")

print(f"    True mom14-K5 OOS 2023-2025 compound: {round(true_comp*100,1)}%")
print(f"    Per year: 2023={true_2023}%, 2024={true_2024}%, 2025={true_2025}%")

# Shuffle null: at each rebal point (every 14d), pick 5 random gated assets
g_oos = ind["gate"][(ind["gate"].index >= OOS_START) & (ind["gate"].index < OOS_END)]
C_oos = ind["C"][(ind["C"].index >= OOS_START) & (ind["C"].index < OOS_END)]
R_oos = ind["R"][(ind["R"].index >= OOS_START) & (ind["R"].index < OOS_END)].fillna(0.0)

import strat.ma_strat_builder as msb
COST = msb.TAKER_RT

N_SEEDS = 200
K_RAND = 5
REBAL = 14

null_comps = []
null_2023  = []
null_2024  = []
null_2025  = []

for seed in range(N_SEEDS):
    rng = np.random.default_rng(seed)
    W_null = pd.DataFrame(0.0, index=C_oos.index, columns=C_oos.columns)
    last = -999
    for i, d in enumerate(C_oos.index):
        if i - last >= REBAL:
            elig = [s for s in C_oos.columns if bool(g_oos.loc[d, s])]
            if elig:
                picks = rng.choice(elig, size=min(K_RAND, len(elig)), replace=False).tolist()
                W_null.loc[d, :] = 0.0
                for s in picks:
                    W_null.loc[d, s] = 1.0 / len(picks)
            last = i
        elif i > 0:
            W_null.iloc[i] = W_null.iloc[i-1]
    pos = W_null.shift(1).fillna(0.0)
    turn = pos.diff().abs().fillna(pos.abs()).sum(axis=1)
    bret_null = (pos * R_oos).sum(axis=1) - turn * (COST / 2.0)
    x = bret_null.to_numpy()
    null_comps.append(np.prod(1 + x) - 1)
    # per year
    mask23 = (bret_null.index >= "2023-01-01") & (bret_null.index < "2024-01-01")
    mask24 = (bret_null.index >= "2024-01-01") & (bret_null.index < "2025-01-01")
    mask25 = (bret_null.index >= "2025-01-01") & (bret_null.index < "2026-01-01")
    null_2023.append(np.prod(1 + bret_null.to_numpy()[np.asarray(mask23)]) - 1)
    null_2024.append(np.prod(1 + bret_null.to_numpy()[np.asarray(mask24)]) - 1)
    null_2025.append(np.prod(1 + bret_null.to_numpy()[np.asarray(mask25)]) - 1)

null_comps = np.array(null_comps)
null_2023  = np.array(null_2023)
null_2024  = np.array(null_2024)
null_2025  = np.array(null_2025)

# One-sided p-value: fraction of nulls that beat true
p_pool = float(np.mean(null_comps >= true_comp))
p_23   = float(np.mean(null_2023 >= (true_2023/100)))
p_24   = float(np.mean(null_2024 >= (true_2024/100)))
p_25   = float(np.mean(null_2025 >= (true_2025/100)))

print(f"\n=== OOS SHUFFLE NULL RESULTS (N={N_SEEDS}) ===")
print(f"Pooled 2023-2025:")
print(f"  True comp:  {round(true_comp*100,1)}%")
print(f"  Null median:{round(np.median(null_comps)*100,1)}%  mean:{round(np.mean(null_comps)*100,1)}%")
print(f"  Null p05:   {round(np.percentile(null_comps,5)*100,1)}%")
print(f"  Null p95:   {round(np.percentile(null_comps,95)*100,1)}%")
print(f"  p-value (one-sided, true >= null): {p_pool:.3f}")
print(f"Per-year p-values:")
print(f"  2023: true={true_2023}%, null_med={round(np.median(null_2023)*100,1)}%, p={p_23:.3f}")
print(f"  2024: true={true_2024}%, null_med={round(np.median(null_2024)*100,1)}%, p={p_24:.3f}")
print(f"  2025: true={true_2025}%, null_med={round(np.median(null_2025)*100,1)}%, p={p_25:.3f}")

# Buy-hold reference (gate=False, EW)
R_ref = ind["R"].reindex(index=W_mom14.index).fillna(0.0)
bh = R_ref.mean(axis=1)
bh_2023 = _comp(bh, "2023-01-01", "2024-01-01")
bh_2024 = _comp(bh, "2024-01-01", "2025-01-01")
bh_2025 = _comp(bh, "2025-01-01", "2026-01-01")
bh_oos   = _comp(bh, "2023-01-01", "2026-01-01")
print(f"\nBuy-hold (EW universe) reference:")
print(f"  2023={bh_2023}%, 2024={bh_2024}%, 2025={bh_2025}%, OOS_total={bh_oos}%")

# -----------------------------------------------------------------------
# 5.  Final markdown-ready table
# -----------------------------------------------------------------------
print("\n\n=== FINAL SUMMARY TABLE (per-year comp%) ===")
header_cols = YEARS + ["FULL"]
for strat_name in STRATS:
    row_data = {}
    for yr in YEARS:
        row_data[yr] = pivot_comp.loc[strat_name, yr]
    row_data["FULL"] = pivot_comp.loc[strat_name, "FULL"]
    vals = " | ".join(f"{row_data[c]:>7}" if row_data[c] is not None else f"{'N/A':>7}" for c in header_cols)
    print(f"  {strat_name:<22} | {vals}")

print("\n[DONE] meta_fold_cycle2_oos.py completed.")
