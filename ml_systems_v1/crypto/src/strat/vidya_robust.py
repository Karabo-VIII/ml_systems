"""src/strat/vidya_robust.py -- is VIDYA_FULL's improvement over EMA_FULL real, or one-window?

WHY: vidya_stack found VIDYA_FULL >> EMA_FULL on OOS (p05 -33 -> -15, Sharpe 0.23 -> 0.89). That is ONE
window + ONE seed. The MR lead looked just as good on OOS and REVERSED on VAL -- so before believing VIDYA,
check it REPLICATES: (a) on BOTH held-out spans (VAL + OOS), (b) across bootstrap seeds, for compound,
Sharpe, AND block-bootstrap p05. If VIDYA beats EMA on both spans across seeds, the MA-type upgrade is real;
if only on OOS-seed-7, it was noise. 2MA-slow FULL stack, 4h, UNSEEN sealed. RWYB: python -m strat.vidya_robust.
No emoji (cp1252).
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
from strat.portfolio_replay import MAKER_RT
from strat.replay_distinct_grid import distinct_specs
from strat.battery import block_bootstrap_p05_p95
from strat.ma_type_upgrade import _nums
from strat.vidya_stack import book

SPANS = {"VAL": ("2024-05-15", "2025-03-15"), "OOS": ("2025-03-15", "2025-12-31")}
SEEDS = [7, 13, 21, 42]


def _daily(cfgs, ma_type):
    _, series = book(cfgs, ma_type, "2018-01-01", "2025-12-31", True, MAKER_RT, date_index=True)
    if not series:
        return None
    allc = [x for lst in series.values() for x in lst]
    return pd.concat(allc, axis=1).mean(axis=1, skipna=True).resample("1D").apply(lambda x: float((1 + x).prod() - 1)).dropna()


def _sl(d, lo, hi):
    return d[(d.index >= lo) & (d.index < hi)]


def main() -> int:
    ma_cfg = {}
    for fam in ("2MA", "3MA"):
        ma_cfg.update(distinct_specs(fam, 0.15, max_n=60))
    PR.STRATS.update(ma_cfg)
    slow2 = [n for n in ma_cfg if len(_nums(n)) == 2 and 60 <= max(_nums(n)) < 150]
    print("VIDYA robustness: does VIDYA_FULL > EMA_FULL replicate across SPANS + SEEDS? (2MA-slow)\n")
    ema = _daily(slow2, "EMA"); vid = _daily(slow2, "VIDYA")
    if ema is None or vid is None:
        print("no data"); return 1

    out = {}
    for span, (lo, hi) in SPANS.items():
        e, v = _sl(ema, lo, hi), _sl(vid, lo, hi)
        ec = float((1 + e).prod() - 1) * 100; vc = float((1 + v).prod() - 1) * 100
        esh = float(e.mean() / (e.std() + 1e-12) * np.sqrt(365)); vsh = float(v.mean() / (v.std() + 1e-12) * np.sqrt(365))
        print(f"## span {span} ({lo}..{hi})")
        print(f"   compound%: EMA {ec:+.1f}  VIDYA {vc:+.1f}   |   Sharpe: EMA {esh:+.2f}  VIDYA {vsh:+.2f}")
        print(f"   {'seed':>6} {'ema_p05':>9} {'vid_p05':>9} {'vid-ema':>9} {'VIDYA better?':>14}")
        wins = 0
        for sd in SEEDS:
            ep = block_bootstrap_p05_p95(e.to_numpy(), seed=sd).get("p05")
            vp = block_bootstrap_p05_p95(v.to_numpy(), seed=sd).get("p05")
            d = vp - ep if (ep is not None and vp is not None) else None
            better = d is not None and d > 0
            wins += int(better)
            print(f"   {sd:>6} {ep:>9.2f} {vp:>9.2f} {d:>+9.2f} {'YES' if better else 'no':>14}")
        out[span] = {"ema_compound": round(ec, 1), "vidya_compound": round(vc, 1),
                     "ema_sharpe": round(esh, 2), "vidya_sharpe": round(vsh, 2),
                     "vidya_better_p05_seeds": f"{wins}/{len(SEEDS)}",
                     "vidya_compound_better": vc > ec, "vidya_sharpe_better": vsh > esh}
        print(f"   -> VIDYA p05 > EMA p05 in {wins}/{len(SEEDS)} seeds; compound better={vc>ec}; Sharpe better={vsh>esh}\n")

    both = all(out[s]["vidya_compound_better"] and out[s]["vidya_sharpe_better"] and
               out[s]["vidya_better_p05_seeds"].startswith(("3", "4")) for s in SPANS)
    print(f"VERDICT: VIDYA_FULL > EMA_FULL on BOTH spans (compound+Sharpe+p05-majority)? -> "
          f"{'YES -- the MA-type upgrade is SPAN-ROBUST' if both else 'NO -- span/metric-dependent, treat as suggestive'}")
    jout = ROOT.parent / "runs" / "periods" / "_OOS_CONFIRM" / "vidya_robust.json"
    json.dump(out, open(jout, "w"), indent=1, default=str)
    print(f"[json] {jout}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
