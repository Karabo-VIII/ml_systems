"""Run the EMA engine for the 4 FINE-TF cells {30m,15m} x {7d,14d} that the killed main sweep did not reach.

Same engine (ma_movecatch_engine.run_engine), NO infra rebuild. Compute-aware per the charter:
 - n_null reduced 300 -> 150 for the fine-TF block-bootstrap/regime null (cheaper SE; documented as the only
   deviation from the coarse-TF cells). Capture A/B/C, WEALTH, and gates all still run on the FULL u100 panel.
 - run_power=False (apparatus power already established on the 1d cell).
 - each cell written incrementally to _ema_cell_<tf>_<w>d.json so a hang on a later cell never loses an earlier one.
No git commit. No emoji.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import strat.ma_movecatch_engine as E

MA = "EMA"
CELLS = [("30m", 7), ("30m", 14), ("15m", 7), ("15m", 14)]
N_NULL_FINE = 150

def main():
    t_all = time.time()
    for i, (tf, w) in enumerate(CELLS):
        t0 = time.time()
        try:
            rec = E.run_engine(MA, tf, w, run_power=False, n_null=N_NULL_FINE, verbose=True)
            rec["_fine_tf_caveat"] = f"n_null reduced 300->{N_NULL_FINE} for tractability (capture+wealth+gates on FULL u100 panel)"
        except Exception as ex:
            import traceback; traceback.print_exc()
            rec = {"cell": f"{MA}|{tf}|{w}d", "ma_type": MA, "tf": tf, "window_days": w,
                   "note": f"ENGINE EXCEPTION: {type(ex).__name__}: {str(ex)[:200]}"}
        rec["_runner_elapsed_s"] = round(time.time() - t0, 1)
        if "A_identification_costfree" in rec:
            E.print_cell(rec)
        else:
            print(f"  {rec['cell']}: {rec.get('note')}")
        E.OUTDIR.mkdir(parents=True, exist_ok=True)
        json.dump(rec, open(E.OUTDIR / f"_ema_cell_{tf}_{w}d.json", "w"), indent=1, default=str)
        print(f"[fine {i+1}/4] {tf}|{w}d done in {rec['_runner_elapsed_s']}s  (cum {round(time.time()-t_all,1)}s)\n", flush=True)
    print(f"[fine total] {round(time.time()-t_all,1)}s")
    return 0

if __name__ == "__main__":
    sys.exit(main())
