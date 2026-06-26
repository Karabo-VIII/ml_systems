"""ADX|1d OVERFIT PROBE — is the selective edge a SMOOTH plateau or an isolated spike?

The dead-case rework found ADX|1d (lb21-28, thr0.45) is selective + VAL/OOS-consistent +
null-beating. Before promoting it to a validated engine, the edge surface across the
hyperparameter grid must be SMOOTH (robust) not a SPIKE (overfit). Dense grid around the
finding: lookback {16,18,20,22,24,26,28,30} x threshold {0.38,0.42,0.45,0.48,0.52}.

Reports OOS edge for every cell. A robust engine => most neighbours also positive.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CRYPTO = ROOT.parent

from strat.ti_engine_sota import adx_single_held
from strat.ti_deadcase_rework import eval_cell

OUTDIR = CRYPTO / "runs" / "engines" / "V5_dna_report"


def adx_hold(lb, thr):
    def fn(o, h, l, c):
        return adx_single_held(c, lb, thr)
    return fn


def main():
    lookbacks = [16, 18, 20, 22, 24, 26, 28, 30]
    thresholds = [0.38, 0.42, 0.45, 0.48, 0.52]
    print("=" * 96)
    print("ADX|1d OVERFIT PROBE — OOS edge (pp/wk) heatmap. Smooth plateau => robust.")
    print("=" * 96)
    header = "  lb\\thr " + "".join(f"{t:>9.2f}" for t in thresholds)
    print(header); print("  " + "-" * (len(header)))
    grid = {}
    pos = 0; tot = 0
    for lb in lookbacks:
        cells = []
        row = f"  {lb:>5} "
        for thr in thresholds:
            res = eval_cell(adx_hold(lb, thr), "1d", n_null=60)
            if res is None:
                row += f"{'NA':>9}"; continue
            mv, vn, mo, on = res
            grid[f"{lb}_{thr}"] = {"val_edge": mv["edge_pp"], "oos_edge": mo["edge_pp"],
                                   "oos_null": round(on, 0), "oos_deploy": mo["deploy"],
                                   "val_deploy": mv["deploy"]}
            tot += 1; pos += (mo["edge_pp"] > 0)
            mark = "*" if (mo["edge_pp"] > 0 and on >= 95 and mo["deploy"] < 0.70) else " "
            row += f"{mo['edge_pp']:>+8.2f}{mark}"
        print(row)
    print("  " + "-" * (len(header)))
    print(f"  OOS-positive cells: {pos}/{tot} ({pos/tot:.0%}).  '*' = positive + null>=95 + selective")
    print("  (a SMOOTH plateau of * and + around lb22-28/thr0.45 confirms ADX|1d is NOT an overfit spike)")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(OUTDIR / f"adx_overfit_probe_{ts}.json", "w") as f:
        json.dump({"grid": grid, "oos_positive_frac": pos / tot, "timestamp": ts}, f, indent=2, default=str)
    print(f"\n  Saved adx_overfit_probe_{ts}.json")


if __name__ == "__main__":
    main()
