"""src/strat/mr_robust.py -- is the orthogonal-diversification benefit (p05 -21->-17) one-window noise?

WHY: mr_diversify.py found a BO+MR ensemble lifts the OOS-heldout p05 (-21.21 -> -17.37). That is ONE
window + ONE bootstrap seed -- the honest caveat. Before believing the diversification lever, check it
REPLICATES: (a) on a SECOND independent held-out span (VAL, not just OOS), and (b) across bootstrap seeds.
If the ensemble p05 > breakout p05 on BOTH spans AND across seeds, the diversification is robust; if only
on OOS-seed-7, it was noise. UNSEEN sealed. RWYB: python -m strat.mr_robust. No emoji (cp1252).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strat.battery import block_bootstrap_p05_p95
from strat.mr_diversify import book_daily

SPANS = {"VAL": ("2024-05-15", "2025-03-15"), "OOS": ("2025-03-15", "2025-12-31")}
SEEDS = [7, 13, 21, 42]


def _slice(d, lo, hi):
    return d[(d.index >= lo) & (d.index < hi)]


def main() -> int:
    print("MR robustness: does the BO+MR diversification (p05 lift) replicate across SPANS + SEEDS?\n")
    bo = book_daily("bo"); mr = book_daily("mr"); ens = book_daily("ens")
    if bo is None or ens is None:
        print("no data"); return 1

    out = {}
    for span, (lo, hi) in SPANS.items():
        print(f"## span {span} ({lo}..{hi})")
        bo_s, mr_s, ens_s = _slice(bo, lo, hi), _slice(mr, lo, hi), _slice(ens, lo, hi)
        comp = {nm: round(float((1 + s).prod() - 1) * 100, 1) for nm, s in
                [("BREAKOUT", bo_s), ("MR", mr_s), ("BO+MR", ens_s)]}
        print(f"   compound%: BREAKOUT {comp['BREAKOUT']}  MR {comp['MR']}  BO+MR {comp['BO+MR']}")
        print(f"   {'seed':>6} {'bo_p05':>9} {'ens_p05':>9} {'ens-bo':>9} {'diversifies?':>13}")
        wins = 0
        for sd in SEEDS:
            bp = block_bootstrap_p05_p95(bo_s.to_numpy(), seed=sd).get("p05")
            ep = block_bootstrap_p05_p95(ens_s.to_numpy(), seed=sd).get("p05")
            d = ep - bp if (bp is not None and ep is not None) else None
            better = d is not None and d > 0
            wins += int(better)
            print(f"   {sd:>6} {bp:>9.2f} {ep:>9.2f} {d:>+9.2f} {'YES' if better else 'no':>13}")
        out[span] = {"compound": comp, "seeds_diversify": f"{wins}/{len(SEEDS)}"}
        print(f"   -> ensemble improves p05 in {wins}/{len(SEEDS)} seeds\n")

    val_w = out["VAL"]["seeds_diversify"]; oos_w = out["OOS"]["seeds_diversify"]
    print(f"VERDICT: diversification p05-lift replicates -- VAL {val_w} seeds, OOS {oos_w} seeds. "
          f"{'ROBUST across both spans' if val_w.startswith(('3','4')) and oos_w.startswith(('3','4')) else 'span/seed-dependent -- treat as suggestive not proven'}")
    jout = ROOT.parent / "runs" / "periods" / "_OOS_CONFIRM" / "mr_robust.json"
    json.dump(out, open(jout, "w"), indent=1, default=str)
    print(f"[json] {jout}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
