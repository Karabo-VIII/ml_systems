"""MA SLICE STUDY orchestrator — the full A-E matrix in one structured run.

Produces a tiered comparison answering: "Can we trade any 7d window profitably, net of
PROD cost?" across:
  - COST: maker vs taker vs prod (A)
  - MODE: rolling vs fixed slices (D)
  - WHIPSAW control: min-hold + cooldown for short/medium TFs (B)
  - EXIT: canonical default (confirmation-break + time-cap) (C)
  - TIER: SHORT{15m,30m} MEDIUM{1h,2h,4h} LONG{1d} with tier-appropriate lookback/hold (E)

Run is heavy; results saved as JSON + a markdown summary.
"""
from __future__ import annotations

import warnings; warnings.filterwarnings("ignore")
import sys
import json
import time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CRYPTO = ROOT.parent

from strat.ma_slice_engine import run_cell, OUTDIR
from strat.canonical_exits import COST_MAKER, COST_TAKER, COST_PROD

COSTS = {"maker": COST_MAKER, "taker": COST_TAKER, "prod": COST_PROD}

# Tier-appropriate config (E): (lookback_days, slice_days, min_hold_bars, cooldown_bars)
# bars are per-TF; min_hold/cooldown given in BARS at that TF.
TIER_CFG = {
    # SHORT: find good lookback/window; whipsaw control ON (NOT traded yet)
    "15m": {"lookback": 7, "slice": 3, "min_hold": 8, "cooldown": 8},   # 2h min-hold, 2h cooldown
    "30m": {"lookback": 7, "slice": 3, "min_hold": 6, "cooldown": 6},   # 3h min-hold
    # MEDIUM: 7-14d lookback, 7d window, 4h-preferred; mild whipsaw control
    "1h":  {"lookback": 7, "slice": 7, "min_hold": 6, "cooldown": 6},
    "2h":  {"lookback": 7, "slice": 7, "min_hold": 3, "cooldown": 3},
    "4h":  {"lookback": 14, "slice": 7, "min_hold": 0, "cooldown": 0},  # the sweet spot, no whipsaw issue
    # LONG: 14d lookback, 14d hold
    "1d":  {"lookback": 14, "slice": 14, "min_hold": 0, "cooldown": 0},
}

TYPES = ["EMA", "SMA", "WMA", "DEMA", "KAMA", "VIDYA"]  # drop TEMA/HMA (weak OOS)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfs", default="4h,2h,1h,1d,30m,15m")
    ap.add_argument("--costs", default="taker,prod")
    ap.add_argument("--modes", default="rolling,fixed")
    ap.add_argument("--types", default=",".join(TYPES))
    ap.add_argument("--window", default="OOS")
    ap.add_argument("--whipsaw", action="store_true",
                    help="apply tier min-hold/cooldown (B). Off = no whipsaw control (baseline).")
    a = ap.parse_args()

    tfs = [t.strip() for t in a.tfs.split(",")]
    costs = [c.strip() for c in a.costs.split(",")]
    modes = [m.strip() for m in a.modes.split(",")]
    types = [t.strip().upper() for t in a.types.split(",")]

    print("=" * 120)
    print(f"MA SLICE STUDY (A-E) — window={a.window} | whipsaw_control={a.whipsaw}")
    print("=" * 120)
    print(f"  {'type':<6}{'tf':<5}{'tier':<7}{'cost':<6}{'mode':<8}|{'dep':>6}{'edge':>8}"
          f"{'lin/wk':>8}{'Sh':>6}{'7dprof':>7}{'trades':>7}{'cdrag':>8}")
    print("  " + "-" * 100)

    rows = []
    t0 = time.time()
    for tf in tfs:
        cfg = TIER_CFG[tf]
        mh = cfg["min_hold"] if a.whipsaw else 0
        cd = cfg["cooldown"] if a.whipsaw else 0
        for ma in types:
            for cost in costs:
                for mode in modes:
                    r = run_cell(ma, tf, slice_mode=mode, lookback_days=cfg["lookback"],
                                 slice_days=cfg["slice"], cost_rt=COSTS[cost],
                                 min_hold_bars=mh, cooldown_bars=cd, window=a.window)
                    if r["status"] != "OK":
                        print(f"  {ma:<6}{tf:<5}{'':7}{cost:<6}{mode:<8}| {r['status']}")
                        continue
                    r["cost_name"] = cost
                    rows.append(r)
                    print(f"  {ma:<6}{tf:<5}{r['tier']:<7}{cost:<6}{mode:<8}|{r['deploy']:>5.0%}"
                          f"{r['edge_pp']:>+8.2f}{r['lin_wk']:>+8.2f}{r['sharpe']:>6.2f}"
                          f"{r['pct_profitable_7d_slices']:>6.0%}{r['n_trades_est']:>7}"
                          f"{r['cost_drag_wk_pp']:>+8.2f}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "whipsaw" if a.whipsaw else "baseline"
    out = OUTDIR / f"slice_study_FULL_{tag}_{a.window}_{ts}.json"
    with open(out, "w") as f:
        json.dump({"rows": rows, "window": a.window, "whipsaw": a.whipsaw,
                   "tier_cfg": TIER_CFG, "timestamp": ts}, f, indent=2, default=str)
    print(f"\n  Done in {(time.time()-t0)/60:.1f} min. Saved: {out}")


if __name__ == "__main__":
    main()
