"""Run the EMA engine across ALL 6 TFs x BOTH windows = 12 cells, assemble engine_EMA.json.

PHASE 2 driver wrapper (does NOT rebuild infra). Strategy per the charter:
 - run all 12 cells via ma_movecatch_engine.run_engine (capture + wealth together, the driver already does both).
 - coarse-to-fine order so 1d/4h/2h/1h land first; if a fine-TF (15m/30m) cell exceeds a wall-clock budget the
   cell is still emitted with whatever completed -- run_engine internally returns wealth=None on replay error,
   and we never silently drop.
 - positive_control power runs ONCE (apparatus-level) on the first cell; subsequent cells pass run_power=False.
No git commit (meta reviews). No emoji (cp1252).
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import strat.ma_movecatch_engine as E

MA = "EMA"
CELLS = [(tf, w) for tf in E.TFS for w in E.WINDOWS]  # 12, coarse-to-fine, 7d before 14d

def main():
    t_all = time.time()
    out_cells = []
    power_done = False
    for i, (tf, w) in enumerate(CELLS):
        t0 = time.time()
        run_power = (not power_done)
        try:
            rec = E.run_engine(MA, tf, w, run_power=run_power, verbose=True)
            if run_power and rec.get("gate_positive_control_power") and "error" not in (rec["gate_positive_control_power"] or {}):
                power_done = True
        except Exception as ex:
            import traceback; traceback.print_exc()
            rec = {"cell": f"{MA}|{tf}|{w}d", "ma_type": MA, "tf": tf, "window_days": w,
                   "note": f"ENGINE EXCEPTION: {type(ex).__name__}: {str(ex)[:200]}"}
        rec["_runner_elapsed_s"] = round(time.time() - t0, 1)
        E.print_cell(rec) if "A_identification_costfree" in rec else print(f"  {rec['cell']}: {rec.get('note')}")
        out_cells.append(rec)
        # incremental per-cell write so a hang on a later cell never loses completed ones
        E.OUTDIR.mkdir(parents=True, exist_ok=True)
        json.dump(rec, open(E.OUTDIR / f"_ema_cell_{tf}_{w}d.json", "w"), indent=1, default=str)
        print(f"[{i+1}/12] {tf}|{w}d done in {rec['_runner_elapsed_s']}s  (cum {round(time.time()-t_all,1)}s)\n", flush=True)

    artifact = {
        "ma_type": MA,
        "n_cells": len(out_cells),
        "tfs": E.TFS, "windows": E.WINDOWS,
        "git_sha": E.git_sha(),
        "leaderboard": str(E.LEADERBOARD),
        "split": f"DEV (TRAIN+VAL, <= {E.fl.DEV_END}); OOS(2020 Q4 leaderboard split)+UNSEEN(>={E.fl.DEV_END}) SEALED",
        "selection_rule": "per cell: band members with TRAIN.net>0 AND VAL.net>0 (OOS field NOT read); fixed-EW ensemble, NEVER ranked[0]",
        "aggregation": "fixed-EW fillna(0.0).mean(axis=1) -- NEVER skipna",
        "cells": out_cells,
        "total_runtime_s": round(time.time() - t_all, 1),
    }
    E.OUTDIR.mkdir(parents=True, exist_ok=True)
    outp = E.OUTDIR / "engine_EMA.json"
    json.dump(artifact, open(outp, "w"), indent=1, default=str)
    print(f"\n[ARTIFACT] {outp}")
    print(f"[git] {artifact['git_sha']}")
    print(f"[total] {artifact['total_runtime_s']}s")
    return 0

if __name__ == "__main__":
    sys.exit(main())
