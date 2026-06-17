#!/usr/bin/env python3
"""MARKET RESEARCH a4 -- SURVIVORSHIP texture (a partial read on the bias the literature calls the killer).

We cannot see DELISTED coins (the true survivorship bias). But we CAN characterize: (1) the listing-recency of
the surviving 104 (how many are recent = more survivorship-fragile); (2) whether RECENT listings have inflated
move-density vs long-history assets (the 'new-listing pump' bias that would over-state the case). Uses the
per-asset output of move_distribution (runs/research/move_dist_1d_H1.json). Run:
python scripts/research/survivorship_texture.py
No emoji (Windows cp1252).
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "runs", "research", "move_dist_1d_H1.json")


def main():
    if not os.path.exists(SRC):
        print("run move_distribution.py --cadence 1d --horizon 1 first")
        return 1
    pa = json.load(open(SRC))["per_asset"]
    rows = [(a, s["n_bars"], s.get("freq_net_ge_0.05", 0), s.get("freq_net_positive", 0)) for a, s in pa.items()]
    n_bars = np.array([r[1] for r in rows])
    f5 = np.array([r[2] for r in rows])
    fpos = np.array([r[3] for r in rows])

    # cohorts by history length (daily bars): long >=1500 (~4y+), mid 730-1500, recent <730 (<2y)
    long = n_bars >= 1500
    mid = (n_bars >= 730) & (n_bars < 1500)
    recent = n_bars < 730

    def cohort(mask, label):
        if mask.sum() == 0:
            return f"  {label}: (none)"
        return (f"  {label}: n_assets={int(mask.sum())}  median_history_days={int(np.median(n_bars[mask]))}  "
                f"median_freq_net>=5%={np.median(f5[mask]):.3f}  median_freq_net_positive={np.median(fpos[mask]):.3f}")

    print("SURVIVORSHIP TEXTURE (surviving 104; cannot see delisted):")
    print(f"  listing-recency: long-history(>=4y)={int(long.sum())}  mid(2-4y)={int(mid.sum())}  recent(<2y)={int(recent.sum())}")
    print(cohort(long, "LONG  "))
    print(cohort(mid, "MID   "))
    print(cohort(recent, "RECENT"))
    # correlation: does shorter history => higher move-density (new-listing bias)?
    if len(n_bars) > 10:
        corr = float(np.corrcoef(n_bars, f5)[0, 1])
        print(f"  corr(history_length, freq_net>=5%) = {corr:.3f}  (negative => recent listings move MORE => new-listing bias inflates the case)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
