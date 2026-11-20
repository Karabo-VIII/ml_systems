"""src/strat/deep2020_ma_robust_split.py -- split every MA config into ROBUST vs NON-ROBUST, rank by WEALTH.

User /orc 2026-06-14: "separate the robust list from the non-robust (do the separation, don't throw the list
away). And there's optimising for WEALTH vs Sharpe, right?" -- yes: the binding OBJECTIVE FUNCTION (2026-05-24)
is WEALTH (held-out compound return), NOT Sharpe. The prior leaderboards ranked by Sharpe (which surfaced the
thin-time-in OOS-lucky 1h configs); this re-cut ranks by WEALTH and KEEPS both groups.

Computes ALL grid configs per (MA type, timeframe) (not just the Sharpe-top-15), base FULL stack (trail10 +
min_hold12 + maker), VAL Jul-Sep / OOS Oct-Dec, corrected machinery (fixed-EW + VAL-only vol target). Per config:
  val_net, net (=OOS compound), worst = min(VAL,OOS) [the wealth-ROBUST metric: compound in the WORSE window],
  sharpe, maxdd, drift (=OOS-VAL). robust := |drift| <= DRIFT_TOL (consistent across windows; a ~+5-8 positive
  drift baseline is expected from the fixed-EW VAL cash-drag, so the tolerance allows for it). Exports ALL configs
  with the robust flag for the wealth-ranked render. RWYB: python -m strat.deep2020_ma_robust_split --cadences 1d,4h.
No emoji (cp1252).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.portfolio_replay as PR
from strat.replay_distinct_grid import distinct_specs
from strat.ma_type_upgrade import _MA, _nums, MA_TYPES
from strat.deep2020_ma_remedy import _load_assets, _book, _metrics, _speed

DRIFT_TOL = 10.0       # |OOS-VAL| net <= this => ROBUST (consistent across windows; ~5-8 baseline from VAL cash-drag)


def main() -> int:
    cads = ["1d", "4h", "2h", "1h"]
    if "--cadences" in sys.argv:
        cads = sys.argv[sys.argv.index("--cadences") + 1].split(",")
    ma_cfg = {}
    for fam in ("2MA", "3MA"):
        ma_cfg.update(distinct_specs(fam, 0.15, max_n=40))
    PR.STRATS.update(ma_cfg)
    allcfg = list(ma_cfg)
    allp = sorted({p for n in allcfg for p in _nums(n)})
    print(f"ROBUST SPLIT: {len(allcfg)} configs x {len(MA_TYPES)} MA x {len(cads)} TF | WEALTH-ranked | "
          f"robust := |drift|<={DRIFT_TOL}, 2020\n")

    export = {}
    for cad in cads:
        assets, vt = _load_assets(cad)
        if not assets:
            print(f"## {cad}: no assets"); continue
        for mt in MA_TYPES:
            caches = [{p: _MA[mt](A["c"], p) for p in allp} for A in assets]
            recs = []
            for name in allcfg:
                d = _book(assets, caches, [name], mt, 0.0, False, None)
                m = _metrics(d)
                if m:
                    worst = round(min(m["val_net"], m["net"]), 1)
                    recs.append({"cfg": f"{mt}({','.join(map(str, _nums(name)))})", "speed": _speed(_nums(name)),
                                 "worst": worst, "robust": bool(abs(m["drift"]) <= DRIFT_TOL), **m})
            if not recs:
                continue
            export[f"{cad}|{mt}"] = recs
            nrob = sum(1 for r in recs if r["robust"])
            print(f"   {cad:3} {mt:5}: {len(recs):>3} configs -> {nrob:>2} ROBUST / {len(recs)-nrob:>2} non-robust")

    op = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
    op.mkdir(parents=True, exist_ok=True)
    jt = "_".join(cads)
    json.dump(export, open(op / f"ma_robust_{jt}.json", "w"), indent=1, default=str)
    print(f"\n[json] {op / f'ma_robust_{jt}.json'}  -- ALL configs per (MA, TF) with robust flag; render splits + wealth-ranks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
