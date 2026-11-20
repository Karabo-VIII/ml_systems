"""
experiments/adaptive_ma/grid_madna_capture.py

GRID SCREEN: rank (cadence x asset) CELLS by how well MA-DNA CAPTURES the oracle.

This is the direct-capture generalization of oracle_dna_1d_u20_runner.py to the full cadence grid
the user asked for ({15m,30m,1h,4h,1d,dollar,range,dib,runs_volume,adaptive_vol} x u100), to decide
which cell to DEEPEN instead of fixing on 4h.

REUSES (does NOT re-implement) the audited capture machinery:
  experiments/adaptive_ma/sol/oracle_dna_shuffled_falsifier.py :: run()
    -> per-cell held-out MA-DNA AUC, capture_skill (realized/available in [0..1]), shuffled-label
       control (genuineness margin), regime-matched firewall, dna_mean_net_pct (the EV per move).

"MA-DNA captures the oracle" = capture_skill_heldout (the realized fraction of the available per-move
edge a logistic over the past-only norm_/xd_ MA-derived chimera features achieves selecting entries).
AUC alone is NOT capture (a high-AUC classifier can have ~0 capture if the oracle hold is too short for
the MA lag -- the central finding). EV per cell = dna_mean_net_pct (mean honest net per DNA-selected
move on held-out, taker 0.0024).

DATA CENSUS (verified on disk 2026-06-06):
  full chimera (norm_/xd_ features) coverage: 15m(104) 30m(77) 1h(104) 4h(104) 1d(104) dollar(104)
  sparse: dib(3: BTC/ETH/PEPE) range(3) ; EMPTY: runs_volume(0) adaptive_vol(0)
  dollar EXCLUDED from capture (4M-bar O(n*H) high-capture DP infeasible -> granularity-degenerate,
  per runs/research/oracle_ceiling_builder.py + oracle_holdtime_prefilter.json).

RWYB: .venv/Scripts/python.exe experiments/adaptive_ma/grid_madna_capture.py --screen
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "runs" / "research"))
sys.path.insert(0, str(ROOT / "experiments" / "adaptive_ma" / "sol"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import oracle_dna_shuffled_falsifier as fal  # noqa: E402

# consistent cross-cadence panel (first 20 of u50; = the existing 1d run's panel, for continuity)
PANEL20 = ["BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ZEC", "TRX", "PEPE", "ADA",
           "LINK", "SUI", "AVAX", "TAO", "FET", "ENJ", "ORDI", "NEAR", "WLD", "ENA"]
TRIO = ["BTC", "ETH", "PEPE"]   # the only assets built for dib/range

CLOCK_CADENCES = ["15m", "30m", "1h", "4h", "1d"]
ALT_CADENCES = ["dib", "range"]            # sparse (TRIO only)
EXCLUDED = {"dollar": "granularity-degenerate (4M-bar O(n*H) high-capture DP infeasible)",
            "runs_volume": "0 chimera builds on disk (feature-dry -> MA-DNA not computable)",
            "adaptive_vol": "0 chimera builds on disk (feature-dry -> MA-DNA not computable)"}


def cell(asset, cadence, n_shuffle, n_books):
    t0 = time.time()
    r = fal.run(asset=asset, cadence=cadence, n_shuffle=n_shuffle, n_books=n_books, verbose=False)
    ho = r["real"]["HELD_OUT"]
    cap = ho["capture_plain"]
    v = r["VERDICT"]
    return {
        "asset": asset, "cadence": cadence, "n_bars": r["n_bars"], "n_features": r["n_features"],
        "oracle_base_rate": r["oracle_base_rate"], "exit_H_bars": r["exit_H_bars"],
        "auc_real": r["real_held_out_auc"],
        "auc_shuf_p95": r["shuffled_control"]["auc"]["p95"],
        "auc_margin": r["real_held_out_auc"] - r["shuffled_control"]["auc"]["p95"],
        "capture_skill": r["real_held_out_capture_skill"],          # <- PRIMARY: "captures the oracle"
        "dna_mean_net_pct": cap.get("dna_mean_net_pct"),            # <- EV per move (DNA-selected)
        "best_mean_net_pct": cap.get("best_mean_net_pct"),          # oracle realizable per-move ceiling
        "chance_mean_net_pct": cap.get("chance_mean_net_pct"),
        "dna_compound_pct": cap.get("dna_compound_pct"),
        "beats_plain_null_p95": r["regime_firewall"]["beats_plain_null_p95"],
        "beats_regime_null_p95": r["regime_firewall"]["beats_regime_null_p95"],
        "apparatus_sound": v["APPARATUS_SOUND"],
        "dna_auc_beats_shuffled": v["dna_auc_beats_shuffled"],
        "dna_genuine": v["DNA_GENUINE_SIGNAL"],
        "secs": round(time.time() - t0, 1),
    }


def _dist(a):
    a = np.array([x for x in a if x is not None and not (isinstance(x, float) and np.isnan(x))], float)
    if len(a) == 0:
        return {"n": 0}
    return {"n": int(len(a)), "mean": float(a.mean()), "median": float(np.median(a)),
            "p25": float(np.percentile(a, 25)), "p75": float(np.percentile(a, 75)), "max": float(a.max())}


def main(n_shuffle, n_books, panel):
    t_start = time.time()
    jobs = [(a, c) for c in CLOCK_CADENCES for a in panel]
    jobs += [(a, c) for c in ALT_CADENCES for a in TRIO]
    print(f"[GRID MA-DNA capture] cells={len(jobs)}  n_shuffle={n_shuffle} n_books={n_books}  "
          f"panel={len(panel)}  clock={CLOCK_CADENCES}  alt={ALT_CADENCES}")
    print(f"  EXCLUDED: " + "; ".join(f"{k} ({v})" for k, v in EXCLUDED.items()))
    rows = []
    for (a, c) in jobs:
        try:
            rows.append(cell(a, c, n_shuffle, n_books))
        except Exception as e:
            rows.append({"asset": a, "cadence": c, "error": repr(e)[:160]})
            print(f"  {c:5} {a:6} ERROR {repr(e)[:90]}")
    ok = [r for r in rows if "error" not in r]

    # per-cadence aggregate
    cad_agg = {}
    for c in CLOCK_CADENCES + ALT_CADENCES:
        cr = [r for r in ok if r["cadence"] == c]
        if not cr:
            continue
        cad_agg[c] = {
            "n_cells": len(cr),
            "capture_skill": _dist([r["capture_skill"] for r in cr]),
            "auc_real": _dist([r["auc_real"] for r in cr]),
            "auc_margin": _dist([r["auc_margin"] for r in cr]),
            "dna_mean_net_pct": _dist([r["dna_mean_net_pct"] for r in cr]),
            "best_mean_net_pct": _dist([r["best_mean_net_pct"] for r in cr]),
            "median_oracle_H_bars": float(np.median([r["exit_H_bars"] for r in cr])),
            "n_genuine": int(sum(1 for r in cr if r["dna_genuine"])),
            "n_apparatus_sound": int(sum(1 for r in cr if r["apparatus_sound"])),
            "n_beats_regime_firewall": int(sum(1 for r in cr if r["beats_regime_null_p95"])),
        }

    blob = {
        "meta": {"panel": panel, "trio": TRIO, "clock_cadences": CLOCK_CADENCES,
                 "alt_cadences": ALT_CADENCES, "excluded": EXCLUDED,
                 "n_shuffle": n_shuffle, "n_books": n_books, "cost_rt": fal.COST_RT,
                 "capture_metric": "capture_skill_heldout (realized/available per-move edge, [0..1])",
                 "ev_metric": "dna_mean_net_pct (mean honest net per DNA-selected move, held-out)",
                 "reused": "experiments/adaptive_ma/sol/oracle_dna_shuffled_falsifier.py::run",
                 "elapsed_secs": round(time.time() - t_start, 1)},
        "cadence_aggregate": cad_agg, "cells": rows,
    }
    outp = Path(__file__).resolve().parent / "grid_madna_capture.json"
    outp.write_text(json.dumps(blob, indent=2), encoding="utf-8")

    # ---- report ----
    print("\n" + "=" * 100)
    print("PER-CADENCE RANK (by median capture_skill, the 'captures-the-oracle' metric)")
    print("=" * 100)
    print(f"  {'cad':6} {'n':>3} {'cap_skill med':>13} {'cap_skill mean':>14} {'AUCmargin med':>13} "
          f"{'EV/move% med':>12} {'oracleH':>7} {'genuine':>8} {'sound':>6} {'beatFW':>7}")
    ranked = sorted(cad_agg.items(), key=lambda kv: -(kv[1]["capture_skill"].get("median") or -9))
    for c, ag in ranked:
        cs, am, ev = ag["capture_skill"], ag["auc_margin"], ag["dna_mean_net_pct"]
        print(f"  {c:6} {ag['n_cells']:>3} {cs.get('median',0):>+13.3f} {cs.get('mean',0):>+14.3f} "
              f"{am.get('median',0):>+13.3f} {ev.get('median',0):>+12.2f} {ag['median_oracle_H_bars']:>7.1f} "
              f"{ag['n_genuine']:>4}/{ag['n_cells']:<3} {ag['n_apparatus_sound']:>3}/{ag['n_cells']:<2} "
              f"{ag['n_beats_regime_firewall']:>3}/{ag['n_cells']:<3}")

    print("\n" + "=" * 100)
    print("TOP-EV CELLS (by capture_skill, then EV/move) -- genuine-only first, then best-of-rest")
    print("=" * 100)
    gen = [r for r in ok if r["dna_genuine"]]
    rank_all = sorted(ok, key=lambda r: (-(r["capture_skill"] if r["capture_skill"] == r["capture_skill"] else -9),
                                         -(r["dna_mean_net_pct"] or -9)))
    print(f"  genuine cells: {len(gen)} / {len(ok)}")
    print(f"  {'cad':6} {'asset':6} {'cap_skill':>9} {'AUC':>6} {'margin':>7} {'EV/move%':>8} "
          f"{'ceil%':>6} {'oH':>3} {'genuine':>7} {'sound':>5} {'beatFW':>6}")
    for r in rank_all[:18]:
        print(f"  {r['cadence']:6} {r['asset']:6} {r['capture_skill']:>+9.3f} {r['auc_real']:>6.3f} "
              f"{r['auc_margin']:>+7.3f} {(r['dna_mean_net_pct'] or 0):>+8.2f} {(r['best_mean_net_pct'] or 0):>6.2f} "
              f"{r['exit_H_bars']:>3} {str(r['dna_genuine']):>7} {str(r['apparatus_sound']):>5} "
              f"{str(r['beats_regime_null_p95']):>6}")
    print(f"\n[OK] wrote {outp}  ({blob['meta']['elapsed_secs']}s)")
    return blob


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", action="store_true")
    ap.add_argument("--n-shuffle", type=int, default=12)
    ap.add_argument("--n-books", type=int, default=120)
    ap.add_argument("--panel", default="", help="comma assets; default PANEL20")
    args = ap.parse_args()
    panel = [x.strip().upper() for x in args.panel.split(",") if x.strip()] or PANEL20
    main(args.n_shuffle, args.n_books, panel)
