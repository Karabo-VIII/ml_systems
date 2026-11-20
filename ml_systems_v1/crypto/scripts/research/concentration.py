#!/usr/bin/env python3
"""MARKET RESEARCH -k -- CONCENTRATION: is the opportunity broad, or a few fragile survivor names?

If the >=5% up-move 'mover-days' are dominated by a handful of assets, the 'case' is really a bet on a few
volatile small-caps (high survivorship/capacity risk). If spread broadly, it is more robust. Uses the per-asset
output of move_distribution (runs/research/move_dist_1d_H1.json). Run: python scripts/research/concentration.py
No emoji (Windows cp1252).
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "runs", "research", "move_dist_1d_H1.json")


def main():
    if not os.path.exists(SRC):
        print("run: python scripts/research/move_distribution.py --cadence 1d --horizon 1   (first)")
        return 1
    pa = json.load(open(SRC))["per_asset"]
    # mover-days per asset = freq_net_ge_0.05 * n_bars
    md = {a: s.get("freq_net_ge_0.05", 0) * s["n_bars"] for a, s in pa.items()}
    desc = np.array(sorted(md.values(), reverse=True))  # descending: for top-k shares
    total = desc.sum()
    n = len(desc)
    cum = np.cumsum(desc) / total
    # how many assets to reach 50% / 80% of all mover-days
    n50 = int(np.searchsorted(cum, 0.50) + 1)
    n80 = int(np.searchsorted(cum, 0.80) + 1)
    # Gini -- standard formula requires ASCENDING order
    asc = np.sort(desc)
    idx = np.arange(1, n + 1)
    gini = float((2 * (idx * asc).sum()) / (n * asc.sum()) - (n + 1) / n)
    top10_share = float(desc[:10].sum() / total)
    top25_share = float(desc[:25].sum() / total)
    out = {
        "n_assets": n,
        "top10_share_of_mover_days": round(top10_share, 3),
        "top25_share_of_mover_days": round(top25_share, 3),
        "n_assets_for_50pct": n50,
        "n_assets_for_80pct": n80,
        "gini_of_mover_days": round(gini, 3),
        "interpretation": "lower gini / more assets-for-80pct = BROADER (more robust). top10_share>0.5 = concentrated.",
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
