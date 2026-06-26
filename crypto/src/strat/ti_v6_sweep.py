"""V6 RIGOROUS SWEEP — all families × all TFs, TRAIN-selected, VAL-evaluated.

Runs the full V6 architecture (CHARTER §ENGINE BUILD RIGOR) for every family on VAL.
Reports the selection rate (how many configs survive TRAIN) and the deploy rate (how
selective the curated ensemble is). This is the honest answer to "which families can
be made selective with full rigor."
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json, time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CRYPTO = ROOT.parent

from strat.ti_engine_rigorous import (
    run_rigorous_cell, GRIDS, TRAIN_END, VAL_END, OOS_2024_END,
    TRAIN_WALL_MS, VAL_WALL_MS, OOS_WALL_MS,
)

OUTDIR = CRYPTO / "runs" / "engines" / "V6_rigorous"
TFS = ["1d", "4h", "2h", "1h", "30m"]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="all")
    ap.add_argument("--tfs", default="1d,4h,2h")
    ap.add_argument("--null", type=int, default=50)
    ap.add_argument("--lookback", type=int, default=7)
    a = ap.parse_args()

    if a.families == "all":
        fams = list(GRIDS.keys())
    else:
        fams = [f.strip().upper() for f in a.families.split(",")]
    tfs = [t.strip() for t in a.tfs.split(",")]

    print("=" * 100)
    print("V6 RIGOROUS SWEEP — TRAIN-selected, VAL-evaluated (CHARTER §ENGINE BUILD RIGOR)")
    print("=" * 100)
    print(f"TRAIN<=2023-07-01 (config selection) | VAL 2023-07-01..2024-05-15 (evaluation)")
    print(f"lookback={a.lookback}d | slice=7d | gate=BTC-SMA100 | cost=0.0006 RT | null={a.null}")
    print(f"Families: {fams}")
    print(f"TFs: {tfs}")

    all_rows = []
    print(f"\n  {'family':<12} {'tf':<4} {'sel':>5} {'dep':>6} {'edge':>8} {'null':>5} "
          f"{'shrp':>6} {'comp':>8} {'MaxDD':>6} t(s)")
    print("  " + "-" * 80)

    for fam in fams:
        for tf in tfs:
            r = run_rigorous_cell(fam, tf, lookback_days=a.lookback,
                                  window_start=TRAIN_END, window_end=VAL_END,
                                  wall_ms=VAL_WALL_MS, n_null=a.null)
            all_rows.append(r)
            if r["status"] == "OK":
                print(f"  {fam:<12} {tf:<4} "
                      f"{r['selection_rate']:>4.0%} {r['deploy']:>5.1%} "
                      f"{r['edge_pp']:>+7.3f} {r['null_pct']:>4.0f} "
                      f"{r['sharpe']:>6.2f} {r['compound']:>+7.0f}% {r['maxdd']:>5.0f}% "
                      f"{r['elapsed_s']:>4.0f}")
            else:
                print(f"  {fam:<12} {tf:<4} {r['status']}")

    # Summary
    ok = [r for r in all_rows if r["status"] == "OK"]
    selective = [r for r in ok if r["deploy"] < 0.70]
    print(f"\n{'='*100}")
    print(f"  {len(ok)} cells OK. {len(selective)} SELECTIVE (deploy<70%).")
    if selective:
        selective.sort(key=lambda x: -x["edge_pp"])
        print(f"\n  SELECTIVE CELLS (the rare/valuable property):")
        for r in selective:
            print(f"    {r['family']:<12}|{r['tf']:<4} edge {r['edge_pp']:+.3f}pp "
                  f"dep {r['deploy']:.1%} null {r['null_pct']:.0f} Sh {r['sharpe']:.2f} "
                  f"sel-rate {r['selection_rate']:.0%}")
    print(f"\n{'='*100}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUTDIR / f"v6_sweep_{ts}.json"
    with open(out, "w") as f:
        json.dump({"rows": all_rows, "lookback_days": a.lookback, "n_null": a.null,
                   "timestamp": ts}, f, indent=2, default=str)
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
