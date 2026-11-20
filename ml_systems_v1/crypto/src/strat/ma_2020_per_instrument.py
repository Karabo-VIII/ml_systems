"""src/strat/ma_2020_per_instrument.py -- per-INSTRUMENT performance per best MA type, VAL+OOS of 2020.

User /orc 2026-06-12: "still within the 2020 window. Performance PER INSTRUMENT per BEST type of MA for the
VAL and OOS windows of 2020 (the last 6 months)." VAL = Jul-Sep 2020, OOS = Oct-Dec 2020.

This DECOUPLES the book into single instruments (no cross-sectional pooling): for each (instrument, MA class)
it builds that ONE asset's family book (equal-weight its slow 2MA+3MA configs, FULL stack, maker) and reports
VAL + OOS compound. Then per instrument the BEST MA type is selected ON VAL (causal, no look-ahead) and its
OOS reported. Per timeframe (sweep, not a default). Reuses the VERIFIED _cells from ma_2020_breakdown.
RWYB: python -m strat.ma_2020_per_instrument --cadences <tf>. No emoji (cp1252).
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

import strat.portfolio_replay as PR
from strat.replay_distinct_grid import distinct_specs
from strat.ma_type_upgrade import _nums, MA_TYPES
import strat.ma_2020_breakdown as B   # reuse _cells (verified), SPLIT, slow-config setup

SPLIT = B.SPLIT  # within-2020: TRAIN/VAL/OOS
CADENCES = ["1d", "4h", "2h", "1h", "30m", "15m"]


def _compound(series, lo, hi):
    s = series[(series.index >= lo) & (series.index < hi)]
    return float(np.prod(1 + s.to_numpy()) - 1) * 100 if len(s) >= 3 else None


def main() -> int:
    global CADENCES
    if "--cadences" in sys.argv:
        CADENCES = sys.argv[sys.argv.index("--cadences") + 1].split(",")
    ma_cfg = {}
    for fam in ("2MA", "3MA"):
        ma_cfg.update(distinct_specs(fam, 0.15, max_n=60))
    PR.STRATS.update(ma_cfg)
    slow = [n for n in ma_cfg if 60 <= max(_nums(n)) < 150]
    print(f"per-instrument: {len(slow)} slow configs x {len(MA_TYPES)} classes x {len(CADENCES)} TF; VAL+OOS 2020\n")

    out = {}   # (cad, sym) -> {ma_type: {val, oos}}
    for cad in CADENCES:
        # per MA class, get all (cfg,sym) cells once, aggregate per asset
        per_asset = {}   # sym -> {ma_type: {val, oos}}
        for ma_type in MA_TYPES:
            cells = B._cells(slow, ma_type, cad)
            bysym = {}
            for (cfg, sym), s in cells.items():
                bysym.setdefault(sym, []).append(s)
            for sym, lst in bysym.items():
                book = pd.concat(lst, axis=1).mean(axis=1, skipna=True)
                v = _compound(book, *SPLIT["VAL"]); o = _compound(book, *SPLIT["OOS"])
                per_asset.setdefault(sym, {})[ma_type] = {"val": None if v is None else round(v, 1),
                                                          "oos": None if o is None else round(o, 1)}
        for sym, d in per_asset.items():
            out[(cad, sym)] = d

        # ---- per-cadence print: instrument x best-MA (VAL-selected) -> VAL/OOS ----
        syms = [s for (c, s) in out if c == cad]
        print(f"########## CADENCE {cad} -- per instrument: best MA (selected on VAL) -> VAL%/OOS% ##########")
        print(f"   {'instrument':11} {'bestMA':7} {'VAL%':>7} {'OOS%':>7}  | {'all-MA OOS (EMA/SMA/WMA/HMA/DEMA/TEMA/KAMA/VIDYA)':>10}")
        for sym in syms:
            d = out[(cad, sym)]
            valid = {mt: v for mt, v in d.items() if v["val"] is not None}
            if not valid:
                print(f"   {sym:11} {'(no 2020-H2 data)':>30}")
                continue
            best = max(valid, key=lambda mt: valid[mt]["val"])
            oosrow = "/".join(str(d[mt]["oos"]) if d[mt]["oos"] is not None else "-" for mt in MA_TYPES)
            print(f"   {sym:11} {best:7} {str(d[best]['val']):>7} {str(d[best]['oos']):>7}  | {oosrow}")
        print()

    jt = "_".join(CADENCES)
    op = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "MA_2020_BREAKDOWN" / f"per_instrument_{jt}.json"
    op.parent.mkdir(parents=True, exist_ok=True)
    json.dump({f"{c}|{s}": d for (c, s), d in out.items()}, open(op, "w"), indent=1, default=str)
    print(f"[json] {op}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
