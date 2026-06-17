"""scripts/probe_dib_flicker.py -- targeted dib-flicker probe.

TASK: determine whether the BTC-dib +26.5% taker / +36.2% maker capture is real
(cross-asset-robust, survives null, survives concentration) or noise.

CHECKS:
  1. DATA: which assets have dib chimera (BTC, ETH, PEPE confirmed).
  2. REPLICATION: run oracle_capture_lab momentum-continuation on ALL dib assets.
     Does +25% replicate cross-asset?
  3. CONCENTRATION (BTC-dib): jackknife drop-top-K (K=1,3,5,10) of the 99 UNSEEN
     moves by realized_net.  Compute n_eff = 1 / sum(w_i^2) where w_i = net_i / sum.
  4. NULL: random-entry firewall at the same firing rate.  Does BTC-dib +26.5%
     beat the null?
  5. SUB-PERIOD: split UNSEEN into first-half / second-half.  Stable or one window?

No edits to oracle_capture_lab.py -- import read-only.
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "strat"))

# ---- import oracle_capture_lab read-only ----
from strat.oracle_capture_lab import (
    load_series, index_windows, continuation_signal,
    detect_oracle_moves, realized_net_in_window,
    OracleMASegments, precompute_oracle_moves,
    run_grid, MOVE_WINDOW_BY_CADENCE, TAKER, MAKER, COSTS, HORIZONS
)
from strat.selection_signal_lab import nonoverlap_book


# ============================================================
# Helpers
# ============================================================

def pooled_capture_with_per_move(df: pd.DataFrame, fire: np.ndarray,
                                  *, horizon: int, cost: float,
                                  move_window: int, move_thr: float,
                                  first: int, last: int
                                  ) -> tuple[float | None, list[dict]]:
    """Compute pooled capture AND per-move detail (for jackknife + sub-period)."""
    opens = df["open"].to_numpy(float)
    n = len(opens)
    last_valid = n - 2 - horizon
    fire_idx = np.flatnonzero(fire)
    fire_idx = fire_idx[fire_idx <= last_valid]
    moves = detect_oracle_moves(opens, move_window=move_window, move_thr=move_thr,
                                 first=first, last=last)
    p_sum = r_sum = 0.0
    per_move = []
    for (lo, hi, p_move) in moves:
        fin = fire_idx[(fire_idx >= lo) & (fire_idx <= hi)]
        r_net, n_tr = realized_net_in_window(opens, fin, lo, hi, horizon, cost, last_valid)
        p_sum += p_move
        r_sum += r_net
        per_move.append({"lo": int(lo), "hi": int(hi), "p_move": float(p_move),
                          "r_net": float(r_net), "n_trades": int(n_tr)})
    cap = round(r_sum / p_sum, 4) if p_sum > 1e-9 else None
    return cap, per_move


def jackknife_capture(per_move: list[dict], *, drop_k: int) -> float | None:
    """Drop the top-K moves by |r_net| (worst-case concentration test).
    Returns pooled capture on the remainder."""
    if not per_move:
        return None
    arr = sorted(per_move, key=lambda x: x["r_net"], reverse=True)  # highest realized first
    kept = arr[drop_k:]
    p_sum = sum(m["p_move"] for m in kept)
    r_sum = sum(m["r_net"] for m in kept)
    return round(r_sum / p_sum, 4) if p_sum > 1e-9 else None


def n_eff(per_move: list[dict]) -> float:
    """Effective number of moves = 1/HHI on realized_net weights.
    If all positive: n_eff = (sum r)^2 / sum r^2. If many near-zero / negative: interpret with care."""
    nets = np.array([m["r_net"] for m in per_move if m["r_net"] > 0])
    if nets.size < 2:
        return float(len(per_move))
    s = nets.sum()
    if s <= 0:
        return float(len(per_move))
    w = nets / s
    return round(1.0 / float((w ** 2).sum()), 2)


# ============================================================
# REPLICATION: run capture for all dib assets, both costs
# ============================================================
DIB_ASSETS = ["BTCUSDT", "ETHUSDT", "PEPEUSDT"]
CADENCE = "dib"
MOVE_THR = 0.02

print("=" * 90)
print("PROBE 1+2: DATA + REPLICATION -- dib oracle-capture for ALL available dib assets")
print("=" * 90)

loaded: dict[str, pd.DataFrame] = {}
for asset in DIB_ASSETS:
    try:
        df, n_loaded = load_series(asset, CADENCE)
        loaded[asset] = df
        print(f"  {asset}: {n_loaded} bars loaded, using {len(df)}")
    except Exception as e:
        print(f"  {asset}: LOAD FAIL -- {e}")

replication_rows = []
best_by_asset_cost: dict[tuple[str, str], object] = {}

for asset, df in loaded.items():
    mw = MOVE_WINDOW_BY_CADENCE.get(CADENCE, 80)
    wlab, purge_mask = index_windows(len(df), purge=40)
    opens = df["open"].to_numpy(float)
    close = df["close"].to_numpy(float)
    oracle_segs = OracleMASegments(opens, close)
    last_valid_min = len(df) - 2 - max(HORIZONS)
    moves_by_window = precompute_oracle_moves(
        opens, close, wlab, purge_mask, oracle_segs,
        move_window=mw, move_thr=MOVE_THR, last_valid=last_valid_min
    )
    for cost_name, cost in COSTS.items():
        results, best = run_grid(
            df, asset=asset, cadence=CADENCE, cost=cost, cost_name=cost_name,
            move_window=mw, move_thr=MOVE_THR,
            wlab=wlab, purge_mask=purge_mask,
            oracle_segs=oracle_segs, moves_by_window=moves_by_window
        )
        best_by_asset_cost[(asset, cost_name)] = best
        cap_un = best.capture_price.get("UNSEEN")
        cap_oos = best.capture_price.get("OOS")
        n_mv = best.n_moves.get("UNSEEN", 0)
        n_tr = best.n_trades.get("UNSEEN", 0)
        row = {
            "asset": asset, "cost": cost_name,
            "cfg": f"brk{best.brk}/f{best.fast}/s{best.slow}/h{best.horizon}",
            "tv_capture": best.trainval_capture_price,
            "oos_capture": cap_oos,
            "unseen_capture": cap_un,
            "unseen_n_moves": n_mv,
            "unseen_n_trades": n_tr,
            "clears_25pct": bool(cap_un is not None and cap_un >= 0.25),
        }
        replication_rows.append(row)
        mark = " <-- CLEARS 25%" if row["clears_25pct"] else ""
        print(f"  {asset:10} {cost_name:5} {row['cfg']:22} | "
              f"TV={row['tv_capture']:>7} OOS={str(cap_oos):>7} UNSEEN={str(cap_un):>7} "
              f"moves={n_mv:>4} trades={n_tr:>4}{mark}")

n_clear = sum(1 for r in replication_rows if r["clears_25pct"])
print(f"\n  REPLICATION: {n_clear}/{len(replication_rows)} (asset,cost) cells clear 25% on UNSEEN")
btc_clear = [r for r in replication_rows if r["asset"] == "BTCUSDT" and r["clears_25pct"]]
eth_clear = [r for r in replication_rows if r["asset"] == "ETHUSDT" and r["clears_25pct"]]
pepe_clear = [r for r in replication_rows if r["asset"] == "PEPEUSDT" and r["clears_25pct"]]
print(f"  BTC clears: {len(btc_clear)}/2   ETH clears: {len(eth_clear)}/2   PEPE clears: {len(pepe_clear)}/2")

# ============================================================
# PROBE 3: CONCENTRATION / JACKKNIFE -- BTC-dib taker UNSEEN
# ============================================================
print("\n" + "=" * 90)
print("PROBE 3: CONCENTRATION -- BTC-dib taker, jackknife drop-top-K UNSEEN moves")
print("=" * 90)

asset_jk = "BTCUSDT"
cost_jk = "taker"
if asset_jk in loaded:
    df_btc = loaded[asset_jk]
    best_btc = best_by_asset_cost.get((asset_jk, cost_jk))
    mw = MOVE_WINDOW_BY_CADENCE.get(CADENCE, 80)
    wlab, purge_mask = index_windows(len(df_btc), purge=40)
    opens = df_btc["open"].to_numpy(float)
    close = df_btc["close"].to_numpy(float)
    horizon = best_btc.horizon
    cost = TAKER

    # Compute UNSEEN bars range
    unseen_mask = (wlab == "UNSEEN") & (~purge_mask)
    unseen_bars = np.flatnonzero(unseen_mask)
    if unseen_bars.size > 0:
        first_un = int(unseen_bars.min())
        last_un = int(unseen_bars.max())
        # Signal with the best config
        fire_best = continuation_signal(close, brk=best_btc.brk, fast=best_btc.fast, slow=best_btc.slow)
        fire_best = fire_best & (~purge_mask)

        cap_full, per_move_un = pooled_capture_with_per_move(
            df_btc, fire_best, horizon=horizon, cost=cost,
            move_window=mw, move_thr=MOVE_THR, first=first_un, last=last_un
        )
        n_moves_un = len(per_move_un)
        print(f"  Full UNSEEN: n_moves={n_moves_un}  pooled_capture={cap_full}")
        nef = n_eff(per_move_un)
        print(f"  n_eff (HHI on positive r_net weights) = {nef}  (out of {n_moves_un} moves)")

        for k in [1, 3, 5, 10]:
            jk = jackknife_capture(per_move_un, drop_k=k)
            pct_removed = round(k / n_moves_un * 100, 1) if n_moves_un else 0
            print(f"  drop top-{k:2d} ({pct_removed}% of moves): capture = {jk}")
    else:
        print("  No UNSEEN bars found for BTC-dib")
else:
    print("  BTC-dib not loaded")

# ============================================================
# PROBE 4: NULL -- BTC-dib vs random-entry firewall
# ============================================================
print("\n" + "=" * 90)
print("PROBE 4: NULL -- BTC-dib taker vs random-entry firewall (same firing rate, same move windows)")
print("=" * 90)

if asset_jk in loaded:
    df_btc = loaded[asset_jk]
    best_btc = best_by_asset_cost.get((asset_jk, cost_jk))
    mw = MOVE_WINDOW_BY_CADENCE.get(CADENCE, 80)
    wlab, purge_mask = index_windows(len(df_btc), purge=40)
    opens = df_btc["open"].to_numpy(float)
    close = df_btc["close"].to_numpy(float)
    horizon = best_btc.horizon
    cost = TAKER

    unseen_mask = (wlab == "UNSEEN") & (~purge_mask)
    unseen_bars = np.flatnonzero(unseen_mask)
    first_un = int(unseen_bars.min()); last_un = int(unseen_bars.max())

    fire_best = continuation_signal(close, brk=best_btc.brk, fast=best_btc.fast, slow=best_btc.slow)
    fire_best = fire_best & (~purge_mask)
    # firing rate on UNSEEN only
    rate = float(fire_best[first_un:last_un + 1].mean())
    print(f"  Signal firing rate on UNSEEN: {rate:.4f}")

    # Signal capture (the real result)
    cap_signal, _ = pooled_capture_with_per_move(
        df_btc, fire_best, horizon=horizon, cost=cost,
        move_window=mw, move_thr=MOVE_THR, first=first_un, last=last_un
    )
    print(f"  Signal capture (UNSEEN): {cap_signal}")

    # Null distribution: 200 random seeds at the same firing rate
    null_caps = []
    rng = np.random.default_rng(42)
    N_NULL = 200
    null_fire_base = np.zeros(len(df_btc), dtype=bool)
    null_fire_base[first_un:last_un + 1] = True  # restrict null to UNSEEN range
    for seed in range(N_NULL):
        rand_fire = (rng.random(len(df_btc)) < rate) & null_fire_base & (~purge_mask)
        cap_r, _ = pooled_capture_with_per_move(
            df_btc, rand_fire, horizon=horizon, cost=cost,
            move_window=mw, move_thr=MOVE_THR, first=first_un, last=last_un
        )
        if cap_r is not None:
            null_caps.append(cap_r)

    null_arr = np.array(null_caps)
    null_mean = float(null_arr.mean())
    null_p95 = float(np.percentile(null_arr, 95))
    null_p99 = float(np.percentile(null_arr, 99))
    beats_p95 = (cap_signal is not None and cap_signal > null_p95)
    beats_p99 = (cap_signal is not None and cap_signal > null_p99)
    print(f"  Null distribution ({N_NULL} random seeds): mean={null_mean:.4f}  p95={null_p95:.4f}  p99={null_p99:.4f}")
    print(f"  Signal beats null p95: {beats_p95}   beats null p99: {beats_p99}")
    # one-sided p-value
    if cap_signal is not None:
        p_val = float((null_arr >= cap_signal).mean())
        print(f"  One-sided null p-value: {p_val:.4f}  (fraction of {N_NULL} nulls >= signal)")
else:
    print("  BTC-dib not loaded")

# ============================================================
# PROBE 5: SUB-PERIOD stability -- split UNSEEN into first/second half
# ============================================================
print("\n" + "=" * 90)
print("PROBE 5: SUB-PERIOD -- BTC-dib taker, first-half vs second-half of UNSEEN")
print("=" * 90)

if asset_jk in loaded:
    df_btc = loaded[asset_jk]
    best_btc = best_by_asset_cost.get((asset_jk, cost_jk))
    mw = MOVE_WINDOW_BY_CADENCE.get(CADENCE, 80)
    wlab, purge_mask = index_windows(len(df_btc), purge=40)
    opens = df_btc["open"].to_numpy(float)
    close = df_btc["close"].to_numpy(float)
    horizon = best_btc.horizon
    cost = TAKER

    unseen_mask = (wlab == "UNSEEN") & (~purge_mask)
    unseen_bars = np.flatnonzero(unseen_mask)
    first_un = int(unseen_bars.min()); last_un = int(unseen_bars.max())
    mid_un = int((first_un + last_un) // 2)

    fire_best = continuation_signal(close, brk=best_btc.brk, fast=best_btc.fast, slow=best_btc.slow)
    fire_best = fire_best & (~purge_mask)

    cap_h1, pm_h1 = pooled_capture_with_per_move(
        df_btc, fire_best, horizon=horizon, cost=cost,
        move_window=mw, move_thr=MOVE_THR, first=first_un, last=mid_un
    )
    cap_h2, pm_h2 = pooled_capture_with_per_move(
        df_btc, fire_best, horizon=horizon, cost=cost,
        move_window=mw, move_thr=MOVE_THR, first=mid_un + 1, last=last_un
    )
    print(f"  UNSEEN first-half  (bars {first_un}-{mid_un}): n_moves={len(pm_h1)} capture={cap_h1}")
    print(f"  UNSEEN second-half (bars {mid_un+1}-{last_un}): n_moves={len(pm_h2)} capture={cap_h2}")
    both_positive = (cap_h1 is not None and cap_h1 > 0) and (cap_h2 is not None and cap_h2 > 0)
    print(f"  Both halves positive: {both_positive}")

# ============================================================
# FINAL VERDICT
# ============================================================
print("\n" + "=" * 90)
print("VERDICT SUMMARY")
print("=" * 90)

# Gate criteria (default-to-noise if ANY fail):
# G1: replicates on >=2 of 3 assets (not BTC-alone)
# G2: survives jackknife (drop top-5 still > 0)
# G3: beats null p95
# G4: both sub-periods positive

print("\n  Replication (G1): does +25% appear on >= 2 of 3 assets?")
assets_clearing = list({r["asset"] for r in replication_rows if r["clears_25pct"]})
print(f"    Assets clearing 25%: {assets_clearing}")
g1 = len(assets_clearing) >= 2
print(f"    G1 PASS: {g1}")

print("\n  (Jackknife + Null + Sub-period results printed above)")
print("  Apply the default-to-noise rule: REAL only if G1 AND G2 (drop-top-5>0) AND G3 (beats p95) AND G4 (both halves positive)")

print("\n  STRUCTURED OUTPUT FOR OVERSEER:")
verdict_data = {
    "probe": "dib_flicker_close",
    "cadence": "dib",
    "dib_assets_with_data": list(loaded.keys()),
    "replication_rows": replication_rows,
    "btc_dib_taker_unseen_original": 0.2653,
    "btc_dib_maker_unseen_original": 0.362,
    "eth_dib_taker_unseen": next((r["unseen_capture"] for r in replication_rows
                                   if r["asset"] == "ETHUSDT" and r["cost"] == "taker"), None),
    "pepe_dib_taker_unseen": next((r["unseen_capture"] for r in replication_rows
                                    if r["asset"] == "PEPEUSDT" and r["cost"] == "taker"), None),
    "n_assets_clearing_25pct": len(assets_clearing),
    "assets_clearing_25pct": assets_clearing,
    "g1_replicates_cross_asset": g1,
}
print(json.dumps(verdict_data, indent=2, default=str))
