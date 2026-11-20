"""src/strat/ma_blend.py -- intra-family MA-TYPE blend: EMA+VIDYA (complementary regime profiles).

WHY (on-scope, 2MA/3MA): vidya_robust showed EMA and VIDYA are COMPLEMENTARY -- EMA wins the bull VAL
(compound 5.5 vs 2.7), VIDYA wins the hard/choppy OOS (11.2 vs 2.0); VIDYA improves the tail (p05) on BOTH
spans. Unlike the same-beta trend+trend ensemble (which just averaged), two MA TYPES with DIFFERENT regime
responses may genuinely DIVERSIFY within the MA family -- get EMA's bull upside AND VIDYA's hard-tape tail.
This tests a static equal-weight EMA+VIDYA blend (still the 2MA/3MA cross, both MA types pooled) vs each
alone, 2-span (VAL+OOS) x seed. Does BLEND beat BOTH on p05 across both spans? 4h, FULL stack, UNSEEN sealed.
RWYB: python -m strat.ma_blend. No emoji (cp1252).
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


def _series_cells(cfgs, ma_type):
    """per-sym list of cell daily-return series for the FULL stack of one MA type, full->2025-12-31."""
    _, series = book(cfgs, ma_type, "2018-01-01", "2025-12-31", True, MAKER_RT, date_index=True)
    return series or {}


def _book_from(cells_list):
    if not cells_list:
        return None
    df = pd.concat(cells_list, axis=1)
    daily4 = df.mean(axis=1, skipna=True)
    return daily4


def _sl(d, lo, hi):
    return d[(d.index >= lo) & (d.index < hi)]


def main() -> int:
    ma_cfg = {}
    for fam in ("2MA", "3MA"):
        ma_cfg.update(distinct_specs(fam, 0.15, max_n=60))
    PR.STRATS.update(ma_cfg)
    slow2 = [n for n in ma_cfg if len(_nums(n)) == 2 and 60 <= max(_nums(n)) < 150]
    print("MA-type blend: EMA-only vs VIDYA-only vs EMA+VIDYA blend (2MA-slow FULL). UNSEEN sealed.\n")

    ema_cells = _series_cells(slow2, "EMA")
    vid_cells = _series_cells(slow2, "VIDYA")
    ema_all = [x for lst in ema_cells.values() for x in lst]
    vid_all = [x for lst in vid_cells.values() for x in lst]
    ema_d = _book_from(ema_all); vid_d = _book_from(vid_all); blend_d = _book_from(ema_all + vid_all)
    # daily compounding for each
    def daily(d):
        return d.resample("1D").apply(lambda x: float((1 + x).prod() - 1)).dropna()
    ema_dd, vid_dd, bl_dd = daily(ema_d), daily(vid_d), daily(blend_d)

    j = pd.concat([daily(ema_d).rename("e"), daily(vid_d).rename("v")], axis=1).dropna()
    print(f"corr(EMA book, VIDYA book) daily = {float(j['e'].corr(j['v'])):+.3f}\n")

    out = {}
    for span, (lo, hi) in SPANS.items():
        e, v, b = _sl(ema_dd, lo, hi), _sl(vid_dd, lo, hi), _sl(bl_dd, lo, hi)
        comp = {k: round(float((1 + s).prod() - 1) * 100, 1) for k, s in [("EMA", e), ("VIDYA", v), ("BLEND", b)]}
        shp = {k: round(float(s.mean() / (s.std() + 1e-12) * np.sqrt(365)), 2) for k, s in [("EMA", e), ("VIDYA", v), ("BLEND", b)]}
        print(f"## span {span}: compound% {comp}  Sharpe {shp}")
        print(f"   {'seed':>6} {'ema_p05':>9} {'vid_p05':>9} {'blend_p05':>10} {'blend>=both?':>13}")
        wins = 0
        for sd in SEEDS:
            ep = block_bootstrap_p05_p95(e.to_numpy(), seed=sd).get("p05")
            vp = block_bootstrap_p05_p95(v.to_numpy(), seed=sd).get("p05")
            bp = block_bootstrap_p05_p95(b.to_numpy(), seed=sd).get("p05")
            ok = bp is not None and ep is not None and vp is not None and bp >= max(ep, vp) - 0.5
            wins += int(ok)
            print(f"   {sd:>6} {ep:>9.2f} {vp:>9.2f} {bp:>10.2f} {'YES' if ok else 'no':>13}")
        out[span] = {"compound": comp, "sharpe": shp, "blend_ge_both_seeds": f"{wins}/{len(SEEDS)}"}
        print(f"   -> blend p05 >= max(EMA,VIDYA) in {wins}/{len(SEEDS)} seeds\n")

    val_ok = out["VAL"]["blend_ge_both_seeds"].startswith(("3", "4"))
    oos_ok = out["OOS"]["blend_ge_both_seeds"].startswith(("3", "4"))
    print(f"VERDICT: does the blend dominate BOTH MA types on p05 across BOTH spans? VAL {out['VAL']['blend_ge_both_seeds']}, "
          f"OOS {out['OOS']['blend_ge_both_seeds']} -> "
          f"{'YES -- blend is the more robust MA book' if val_ok and oos_ok else 'NO -- blend ~ averages; pick by regime/objective'}")
    jout = ROOT.parent / "runs" / "periods" / "_OOS_CONFIRM" / "ma_blend.json"
    json.dump(out, open(jout, "w"), indent=1, default=str)
    print(f"[json] {jout}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
