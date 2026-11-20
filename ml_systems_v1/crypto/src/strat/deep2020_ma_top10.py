"""src/strat/deep2020_ma_top10.py -- TOP-10 single-config leaderboard per MA type (2020).

Augments MA_REMEDY.md: the GRANULAR view beneath the cluster summary -- the 10 best individual configs per
(MA type, timeframe), base FULL stack (trail10 + min_hold12 + maker), VAL Jul-Sep / OOS Oct-Dec. Reuses the
CORRECTED remedy machinery (fixed-EW alignment + VAL-only vol target), so these numbers inherit the bug fixes.

DESCRIPTIVE leaderboard: ranked by held-out OOS Sharpe so you can SEE the landscape of each MA type. This is
NOT a deployable selection -- ranking on OOS is in-sample snooping; the honest deployable pick remains the
VAL-selected cluster ENSEMBLE in MA_REMEDY.md. The VAL-net + drift columns are the tell: a config strong in
BOTH VAL and OOS (small drift) is robust; one strong only in OOS (large +drift) was OOS-lucky.

Writes per-(cad, MA) top-15 to ma_top10_<cads>.json (the pooled-across-cadence per-MA top-10 is rendered by
deep2020_ma_top10_render.py). RWYB: python -m strat.deep2020_ma_top10 --cadences 1d,4h. No emoji (cp1252).
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
from strat.deep2020_ma_remedy import _load_assets, _book, _metrics, _expo, _speed

KEEP = 15            # store top-15 per (cad, MA) so the render can pool to a clean top-10 across cadences


def main() -> int:
    cads = ["1d", "4h", "2h", "1h"]
    if "--cadences" in sys.argv:
        cads = sys.argv[sys.argv.index("--cadences") + 1].split(",")
    topk = 10
    if "--topk" in sys.argv:
        topk = int(sys.argv[sys.argv.index("--topk") + 1])
    ma_cfg = {}
    for fam in ("2MA", "3MA"):
        ma_cfg.update(distinct_specs(fam, 0.15, max_n=40))
    PR.STRATS.update(ma_cfg)
    allcfg = list(ma_cfg)
    allp = sorted({p for n in allcfg for p in _nums(n)})
    print(f"MA TOP-{topk}: {len(allcfg)} configs x {len(MA_TYPES)} MA x {len(cads)} TF | base FULL stack | "
          f"OOS-Sharpe ranked (DESCRIPTIVE), 2020\n")

    export = {}
    for cad in cads:
        assets, vt = _load_assets(cad)
        if not assets:
            print(f"## {cad}: no assets"); continue
        print(f"########## {cad} -- top {topk} configs per MA type (by held-out OOS Sharpe) ##########")
        for mt in MA_TYPES:
            caches = [{p: _MA[mt](A["c"], p) for p in allp} for A in assets]
            recs = []
            for name in allcfg:
                d = _book(assets, caches, [name], mt, 0.0, False, None)
                m = _metrics(d)
                if m:
                    recs.append({"cfg": f"{mt}({','.join(map(str, _nums(name)))})", "_name": name,
                                 "speed": _speed(_nums(name)), **m})
            if not recs:
                continue
            recs.sort(key=lambda r: -r["sharpe"])
            top = recs[:KEEP]
            for r in top:                                       # time-in only for the kept top-K (cheap)
                tin, turn = _expo(assets, caches, [r["_name"]], mt, 0.0, False, None)
                r["time_in"] = tin; r["turn"] = turn
            print(f"-- {mt} --")
            print(f"   {'#':>2} {'config':22} {'speed':10} {'VALnet':>7} {'OOSnet':>7} {'Sharpe':>7} "
                  f"{'maxDD':>7} {'drift':>7} {'tIn':>5} {'turn':>5}")
            for i, r in enumerate(top[:topk], 1):
                print(f"   {i:>2} {r['cfg']:22} {r['speed']:10} {r['val_net']:>7} {r['net']:>7} {r['sharpe']:>7} "
                      f"{r['maxdd']:>7} {r['drift']:>7} {str(r['time_in']):>5} {str(r['turn']):>5}")
            export[f"{cad}|{mt}"] = [{k: v for k, v in r.items() if k != "_name"} for r in top]
            print()

    op = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
    op.mkdir(parents=True, exist_ok=True)
    jt = "_".join(cads)
    json.dump(export, open(op / f"ma_top10_{jt}.json", "w"), indent=1, default=str)
    print(f"[json] {op / f'ma_top10_{jt}.json'}  -- top-{KEEP} configs per (MA, TF); render pools to per-MA top-{topk}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
