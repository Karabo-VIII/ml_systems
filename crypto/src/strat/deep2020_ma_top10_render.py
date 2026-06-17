"""src/strat/deep2020_ma_top10_render.py -- pool the per-(MA,TF) leaderboards into TOP-10 per MA type.

Reads ma_top10_*.json and renders, for EACH MA type, its top-10 configs POOLED across all cadences (ranked by
held-out OOS Sharpe -- DESCRIPTIVE, not a deployable pick; the VAL-net + drift columns show which are robust
vs OOS-lucky). Sharpe is computed on a daily-resampled series at every cadence, so it is cadence-comparable
(the verification confirmed this). Also prints each MA type's single best config per timeframe.
RWYB: python -m strat.deep2020_ma_top10_render. No emoji (cp1252).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
MA_TYPES = ["EMA", "SMA", "WMA", "HMA", "DEMA", "TEMA", "KAMA", "VIDYA"]
CADS = ["1d", "4h", "2h", "1h"]


def main() -> int:
    data = {}
    for jf in BASE.glob("ma_top10_*.json"):
        for k, v in json.load(open(jf)).items():
            data[k] = v
    if not data:
        print("no ma_top10_*.json found"); return 1

    print("# TOP-10 CONFIGS per MA TYPE (pooled across 1d/4h/2h/1h, by held-out OOS Sharpe) -- 2020, base FULL stack\n")
    print("DESCRIPTIVE (OOS-ranked, in-sample-2020). Robust = strong in BOTH VAL and OOS (small drift); a large")
    print("+drift = OOS-lucky. The honest DEPLOYABLE pick is still the VAL-selected cluster ensemble (MA_REMEDY.md).\n")

    pooled = {}
    for mt in MA_TYPES:
        recs = []
        for cad in CADS:
            for r in data.get(f"{cad}|{mt}", []):
                recs.append({**r, "cad": cad})
        recs.sort(key=lambda r: -r["sharpe"])
        pooled[mt] = recs
        print(f"## {mt} -- top 10 (pooled across cadences)")
        print(f"   {'#':>2} {'TF':>3} {'config':22} {'VALnet':>7} {'OOSnet':>7} {'Sharpe':>7} {'maxDD':>7} "
              f"{'drift':>7} {'tIn':>5} {'turn':>5}")
        for i, r in enumerate(recs[:10], 1):
            print(f"   {i:>2} {r['cad']:>3} {r['cfg']:22} {r['val_net']:>7} {r['net']:>7} {r['sharpe']:>7} "
                  f"{r['maxdd']:>7} {r['drift']:>7} {str(r['time_in']):>5} {str(r['turn']):>5}")
        print()

    # each MA type's best config per timeframe (the per-cell champion)
    print("=" * 90)
    print("## BEST single config per MA type x timeframe (the cell champion, by OOS Sharpe):")
    print(f"   {'MA':6} " + "".join(f"{c:>26}" for c in CADS))
    for mt in MA_TYPES:
        row = f"   {mt:6} "
        for cad in CADS:
            lst = data.get(f"{cad}|{mt}", [])
            if lst:
                b = lst[0]
                row += f"{(b['cfg']+' Sh'+str(b['sharpe'])):>26}"
            else:
                row += f"{'--':>26}"
        print(row)

    # the most ROBUST config per MA type (smallest |drift| among its top-15, i.e. held up VAL->OOS)
    print("\n## MOST ROBUST config per MA type (smallest |drift| among the top-15, strong in BOTH VAL+OOS):")
    for mt in MA_TYPES:
        recs = pooled[mt]
        if not recs:
            continue
        b = min(recs, key=lambda r: abs(r["drift"]))
        print(f"   {mt:6} -> {b['cad']:>3} {b['cfg']:20} |drift| {abs(b['drift']):>4}  VALnet {b['val_net']}  "
              f"OOSnet {b['net']}  Sh {b['sharpe']}  maxDD {b['maxdd']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
