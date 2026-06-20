"""src/strat/ma_movecatch_step1.py -- STEP 1 of the MA move-CAPTURE engine (MA-only, contained build).

Wires MA-cross fires into the existing capture_lab and produces the FIRST capture-rate artifact on the full u100
universe, 1d cadence, 7-day window, seeded from the project's canonical EMA|1d band (the fixed-EW ENSEMBLE of
band.band_configs -- NEVER ranked[0] hindsight).

This is IDENTIFICATION only (STEP 1). We report:
  (A) IDENTIFICATION (cost-FREE)  -- mean MFE | ma_cross fires   vs | random entry  (cost-free available-move edge)
  (B) TRADEABLE     (TAKER 0.0024) -- mean realized net | fires  vs | random         (labeled with fleet_lab.COST)
  (C) CAPTURE-EFFY               -- aggregate sum(realized)/sum(MFE)                  (fraction of move kept)

regime_series in capture_lab is REPORTING-only here (NOT an exit gate). Select on DEV (2020-2024-05-15) only; the
load_wide wall (<= 2024-05-15) enforces no OOS/UNSEEN peek. Long-only spot, taker cost, causal triggers. No emoji.

RWYB: python -m strat.ma_movecatch_step1
"""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CRYPTO = ROOT.parent

import strat.fleet_lab as fl
import strat.capture_lab as cl

LEADERBOARD = CRYPTO / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE" / "config_leaderboard.json"
OUT = CRYPTO / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE" / "MA_MOVECATCH" / "step1_ema_1d_7d.json"


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(CRYPTO)).decode().strip()
    except Exception:
        return "unknown"


def load_band_configs():
    """The canonical EMA|1d band-ensemble member set (positive across TRAIN/VAL/OOS), NEVER ranked[0]."""
    d = json.load(open(LEADERBOARD))
    return d["grid"]["EMA|1d"]["band"]["band_configs"]


def pool_mfe_mean(lab, tf, hold, min_move, warm):
    """Mean available-move (MFE) over the RANDOM-ENTRY pool (the cost-free A baseline). Same valid universe as
    evaluate_ti's pool: warm<=di<=n-hold-1, finite price+MFE, MFE>min_move."""
    C = lab["C"]
    MFE = cl.mfe_matrix(C, hold)
    n = len(C.index)
    valid = np.zeros((n, len(C.columns)), dtype=bool)
    valid[warm:n - hold - 1, :] = True
    Ca = C.to_numpy(); MFEa = MFE.to_numpy()
    valid &= np.isfinite(Ca) & np.isfinite(MFEa)
    poolmat = valid & (MFEa > min_move)
    pidx = np.array(np.where(poolmat)).T
    vals = MFEa[pidx[:, 0], pidx[:, 1]]
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()), int(len(vals))


def main() -> int:
    tf = "1d"
    bpd = fl.BARS_PER_DAY[tf]
    hold = 7 * bpd                              # 7 CALENDAR days in bars (7 at 1d)
    min_move = 0.03                             # only windows with a >=3% available up-move to catch
    warm = 40
    band_configs = load_band_configs()
    print(f"[step1] MA move-CATCH -- ma_type=EMA, tf={tf}, window=7d (hold={hold} bars), {len(band_configs)} band configs")
    print(f"[step1] DEV wall <= {fl.DEV_END}; selecting on DEV only (load_wide hard-caps the wall)")

    # FULL u100 at 1d (n=104 -> caps at assets with >=400 1d DEV-window bars).
    lab = fl.load_wide(n=104, tf=tf, end=fl.DEV_END)
    lab["ma_band_configs"] = band_configs       # consumed by capture_lab.fired_matrix's ma_cross branch
    lab["ma_type"] = "EMA"
    C = lab["C"]
    syms = list(C.columns)
    assert C.index.max() < pd.Timestamp(fl.DEV_END), "WALL VIOLATION"
    breadth = len(syms)
    print(f"[step1] admitted {breadth} u100 symbols; range {C.index.min()} -> {C.index.max()}")
    print(f"[step1] BREADTH PROOF (symbol columns): {syms}")
    assert breadth >= 20, f"BREADTH FLOOR FAIL: only {breadth} symbols (<20) -- a null/result here is NOT valid"

    # ---- the capture evaluation (ma_cross vs random-ENTRY null, time-exit, block bootstrap on, regime reporting) ----
    res = cl.evaluate_ti(lab, "ma_cross", tf=tf, hold=hold, exit_kind="time", min_move=min_move, warm=warm,
                         n_null=300, seed=0, by_regime=True, block=True)
    if "note" in res:
        print(f"[step1] INSUFFICIENT: {res}")
        return 1

    # ---- A: cost-FREE identification edge = mean MFE(fired) - mean MFE(random pool) ----
    pool_mfe, pool_n = pool_mfe_mean(lab, tf, hold, min_move, warm)
    A_fired_mfe = res["mean_MFE_fired"]                    # already in %
    A_pool_mfe = round(100 * pool_mfe, 2)
    A_edge_pp = round(A_fired_mfe - A_pool_mfe, 2)

    # ---- B: tradeable net at TAKER (fleet_lab.COST = 0.0024) ----
    B_fired_net = res["mean_realized_net"]
    B_null_net = res["null_realized_net"]
    B_edge_pp = res["edge_vs_random_pp"]
    B_p_vs_random = res["p_vs_random"]

    # ---- C: capture efficiency ----
    C_capture_rate = res["capture_rate"]

    artifact = {
        "step": 1,
        "what": "MA move-CATCH -- ma_cross band-ensemble fires wired into capture_lab; first capture-rate artifact",
        "ma_type": "EMA",
        "tf": tf,
        "window_days": 7,
        "hold_bars": hold,
        "min_move": min_move,
        "warm_bars": warm,
        "ensemble": "fixed-EW over EMA|1d band.band_configs (NEVER ranked[0]); fired = >50% of band long",
        "band_configs": band_configs,
        "n_band_configs": len(band_configs),
        "breadth_count": breadth,
        "breadth_floor_ok": breadth >= 20,
        "admitted_symbols": syms,
        "n_fired": res["n_fired"],
        "random_pool_n": pool_n,
        "cost_model": {"label": "TAKER", "round_trip": fl.COST, "applies_to": "B only (identification A is cost-FREE)"},
        "A_identification_costfree": {
            "mean_MFE_fired_pct": A_fired_mfe,
            "mean_MFE_randompool_pct": A_pool_mfe,
            "edge_pp": A_edge_pp,
            "note": "cost-FREE available-move (MFE) edge: does ma_cross find BIGGER available moves than random entry?",
        },
        "B_tradeable_net_taker": {
            "mean_realized_net_pct": B_fired_net,
            "null_realized_net_pct": B_null_net,
            "edge_vs_random_pp": B_edge_pp,
            "p_vs_random": B_p_vs_random,
            "block_bootstrap": res.get("block"),
            "cost": fl.COST,
            "cost_label": "taker 0.0024 RT (fleet_lab.COST) -- NOT maker",
            "note": "net of taker cost: does entering on ma_cross beat random entry on realized net?",
        },
        "C_capture_efficiency": {
            "capture_rate": C_capture_rate,
            "note": "aggregate sum(realized_net)/sum(MFE) -- fraction of the available move the time-exit kept",
        },
        "by_regime_REPORTING_ONLY": res.get("by_regime"),
        "regime_note": "regime_series is REPORTING-only in capture_lab -- NOT an exit gate. STEP 1 is identification only.",
        "split": "DEV (2020 .. 2024-05-15); OOS/UNSEEN never touched (load_wide wall)",
        "data_range": [str(C.index.min()), str(C.index.max())],
        "git_sha": git_sha(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(artifact, open(OUT, "w"), indent=1, default=str)

    print("\n========== STEP 1 RESULT ==========")
    print(f"  n_fired = {res['n_fired']}   (random pool n = {pool_n})   breadth = {breadth} symbols")
    print(f"  A IDENTIFICATION (cost-FREE):  MFE_fired {A_fired_mfe}%  vs  MFE_pool {A_pool_mfe}%   edge = {A_edge_pp} pp")
    print(f"  B TRADEABLE (TAKER {fl.COST}): net {B_fired_net}%  vs  null {B_null_net}%   edge = {B_edge_pp} pp   p_vs_random = {B_p_vs_random}")
    if res.get("block"):
        b = res["block"]
        print(f"     block-bootstrap (overlap-aware): obs_edge {b['obs_edge_pp']}pp  p05 {b['block_p05_pp']}pp  p_le0 {b['block_p_le0']}  n_eff_dates {b['n_eff_dates']}")
    print(f"  C CAPTURE EFFICIENCY:          capture_rate = {C_capture_rate}")
    print(f"\n  by-regime (REPORTING only, not a gate):")
    for rg, d in (res.get("by_regime") or {}).items():
        if d:
            print(f"    {rg:6} n={d['n']:>6}  realized {d['realized_net']:>6}%  null {d['null_net']:>6}%  edge {d['edge_pp']:>5}pp  p={d['p_vs_random']}")
    print(f"\n[json] {OUT}")
    print(f"[git] {artifact['git_sha']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
