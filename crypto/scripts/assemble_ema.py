"""Assemble engine_EMA.json from the per-cell _ema_cell_<tf>_<w>d.json artifacts (PHASE 2 deliverable).

Reads whatever cells exist (charter: never silently drop; a cell that did not run is recorded with wealth=None
and a limits note). Orders canonical (1d,4h,2h,1h,30m,15m) x (7d,14d). Picks best_cell by WEALTH (engine
compound %, None last). No git commit.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import strat.ma_movecatch_engine as E

MA = "EMA"
TFS = ["1d", "4h", "2h", "1h", "30m", "15m"]
WINDOWS = [7, 14]
OUT = E.OUTDIR

def load_cell(tf, w):
    p = OUT / f"_ema_cell_{tf}_{w}d.json"
    if p.exists():
        return json.load(open(p))
    return {"cell": f"{MA}|{tf}|{w}d", "ma_type": MA, "tf": tf, "window_days": w,
            "note": "NOT RUN (compute-aware: cell did not complete in the available window)", "WEALTH": None}

def wkey(rec):
    w = rec.get("WEALTH")
    if isinstance(w, dict) and w.get("engine_compound_pct") is not None:
        return w["engine_compound_pct"]
    return float("-inf")

def main():
    cells = [load_cell(tf, w) for tf in TFS for w in WINDOWS]
    # best by WEALTH (engine compound %), only among cells with a non-degenerate wealth (> -100)
    cand = [c for c in cells if isinstance(c.get("WEALTH"), dict) and c["WEALTH"].get("engine_compound_pct") is not None
            and c["WEALTH"]["engine_compound_pct"] > -99.0]
    best = max(cand, key=wkey) if cand else None
    artifact = {
        "ma_type": MA,
        "deliverable": "PHASE 2 -- EMA engine across 6 TFs x 2 windows = 12 cells, WHOLE u100, TRAIN+VAL-selected band (OOS sealed)",
        "n_cells": len(cells),
        "tfs": TFS, "windows": WINDOWS,
        "git_sha": E.git_sha(),
        "leaderboard": str(E.LEADERBOARD),
        "split": f"DEV (TRAIN+VAL, <= {E.fl.DEV_END}); OOS(2020 Q4 leaderboard split)+UNSEEN(>={E.fl.DEV_END}) SEALED -- never read in selection",
        "selection_rule": "per cell: band members with TRAIN.net>0 AND VAL.net>0 (OOS field NOT read); fixed-EW ensemble, NEVER ranked[0]",
        "aggregation": "fixed-EW fillna(0.0).mean(axis=1) -- NEVER skipna",
        "wealth_via": "portfolio_replay.run(universe='u100', hold_provider=fixed-EW ma_cross ensemble, wall_end=DEV_END); buy-hold = always-long provider on the SAME u100 panel/window",
        "ranked_by": "WEALTH (engine compound net %), never Sharpe",
        "best_cell_by_wealth": (best["cell"] if best else None),
        "cells": cells,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    outp = OUT / "engine_EMA.json"
    json.dump(artifact, open(outp, "w"), indent=1, default=str)
    print(f"[ARTIFACT] {outp}")
    print(f"[best by wealth] {artifact['best_cell_by_wealth']}")
    # compact table
    print(f"\n{'cell':14s} {'br':>3s} {'A_pp':>6s} {'B_pp':>6s} {'C':>5s} {'wealth%':>10s} {'bh%':>10s} {'vsBH_pp':>9s} {'p05+':>5s} {'betaPass':>8s}")
    for c in cells:
        if "A_identification_costfree" not in c:
            print(f"{c['cell']:14s}  -- {c.get('note','')[:60]}"); continue
        A=c['A_identification_costfree']; B=c['B_tradeable_net_taker']; Cc=c['C_capture_efficiency']
        W=c['WEALTH']; p=c['gate_block_p05']; g=c['gate_beta_in_disguise']
        we = W['engine_compound_pct']; bh = W['buyhold_compound_pct']; vs = W['vs_buyhold_pp']
        wes = f"{we:.1f}" if we is not None else "None"
        bhs = f"{bh:.1f}" if bh is not None else "None"
        vss = f"{vs:.1f}" if vs is not None else "None"
        betapass = (not g['beta_in_disguise'])
        print(f"{c['cell']:14s} {c['breadth_count']:>3d} {A['edge_pp']:>+6.2f} {B['edge_vs_random_pp']:>+6.2f} {Cc['capture_rate']:>5.3f} {wes:>10s} {bhs:>10s} {vss:>9s} {str(p['p05_positive']):>5s} {str(betapass):>8s}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
