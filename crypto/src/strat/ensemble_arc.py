"""src/strat/ensemble_arc.py -- does ARCHETYPE DIVERSIFICATION (breakout + MA) clear the p05 ceiling?

WHY: breakout (breakout_arc.py) beat MA-cross OOS but BOTH still fail absolute robustness (OOS-heldout
block-bootstrap p05 < 0: MA -33, breakout -21). The textbook robustness lever for a fragile-but-positive
book is DIVERSIFICATION across uncorrelated return sources. MA-cross and Donchian breakout capture trends
DIFFERENTLY (cross vs channel-break) -> their books should be imperfectly correlated -> an equal-weight
ENSEMBLE should have a less-negative tail (higher p05) and broader breadth than either alone. This is the
ONE principled, NON-over-mining lever (combine existing families, not mine new internal-price variants).
PRE-REGISTERED: test MA-only vs BREAKOUT-only vs ENSEMBLE; report ALL three (no cherry-pick). The honest
question: does the ensemble p05 cross 0 (robust) or does the ceiling hold even diversified?

4h, equal-weight u10 book of the FULL stack (family + trail10 + minhold12 + maker). UNSEEN sealed.
RWYB: python -m strat.ensemble_arc. No emoji (cp1252).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.portfolio_replay as PR
from strat.portfolio_replay import holding_state, apply_trail_stop, MAKER_RT
from strat.replay_distinct_grid import distinct_specs
from strat.ma_mechanics import _cached_panel
from strat.structural_fixes import min_hold
from strat.scorecard import score_book
from strat.breakout_arc import donchian_held, _configs as _bo_configs

WARMUP = 600
PERIODS = {"Jun2022_bear": ("2022-06-01", "2022-07-01"),
           "VAL": ("2024-05-15", "2025-03-15"), "OOS": ("2025-03-15", "2025-12-31")}


def _nums(n):
    return [int(x) for x in re.findall(r"\d+", n)]


def _ma_full(name, o, c):
    h = holding_state(name, o, c, c, c).astype(np.int8)
    h = apply_trail_stop(h.copy(), c, 0.10)[0].astype(np.int8)
    return min_hold(h, 12).astype(np.float64)


def _bo_full(c, n, m):
    h = donchian_held(c, n, m).astype(np.int8)
    h = apply_trail_stop(h.copy(), c, 0.10)[0].astype(np.int8)
    return min_hold(h, 12).astype(np.float64)


def _cells(which, start, end, date_index=False):
    """list of (net[wm]) arrays for the chosen book; which in {ma,bo,ens}."""
    ma_cfg = {}
    for fam in ("2MA", "3MA"):
        ma_cfg.update(distinct_specs(fam, 0.15, max_n=60))
    PR.STRATS.update(ma_cfg)
    ma_slow = [n for n in ma_cfg if len(_nums(n)) == 2 and 60 <= max(_nums(n)) < 150]
    bo_cfg, bo_slow = _bo_configs()
    s_ms = pd.Timestamp(start).value // 10**6
    e_ms = pd.Timestamp(end).value // 10**6
    syms = [a["symbol"] for a in yaml.safe_load(open(ROOT.parent / "config" / "universes" / "u10.yaml"))["assets"]]
    per_cell, cell_roi, series = [], [], {}
    for sym in syms:
        try:
            o, h, l, c, ms = _cached_panel(sym, "4h")
        except Exception:
            continue
        e_idx = int(np.searchsorted(ms, e_ms))
        s_idx = max(0, int(np.searchsorted(ms, s_ms)) - WARMUP)
        o, c, ms = o[s_idx:e_idx], c[s_idx:e_idx], ms[s_idx:e_idx]
        if len(c) < 30:
            continue
        wm = ms >= s_ms
        if wm.sum() < 10:
            continue
        ret = np.zeros(len(c)); ret[1:] = c[1:] / c[:-1] - 1.0
        weights = []
        if which in ("ma", "ens"):
            weights += [_ma_full(name, o, c) for name in ma_slow]
        if which in ("bo", "ens"):
            weights += [_bo_full(c, *bo_cfg[name]) for name in bo_slow]
        for w in weights:
            pos = np.zeros(len(c)); pos[1:] = w[:-1]
            flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
            net = (pos * ret - flips * (MAKER_RT / 2.0))
            per_cell.append(net[wm])
            cell_roi.append(float(np.cumprod(1 + net[wm])[-1] - 1) * 100)
            if date_index:
                series.setdefault(sym, []).append(pd.Series(net[wm], index=pd.to_datetime(ms[wm], unit="ms")))
    return per_cell, cell_roi, series


def _book_mt(per_cell):
    if not per_cell:
        return {}
    m = min(len(x) for x in per_cell)
    bk = np.mean([x[:m] for x in per_cell], axis=0)
    eq = np.cumprod(1 + bk); peak = np.maximum.accumulate(eq)
    return {"roi": round(float(eq[-1] - 1) * 100, 1), "maxdd": round(float(((eq - peak) / peak).min() * 100), 1)}


def main() -> int:
    print("ensemble arc: MA-only vs BREAKOUT-only vs ENSEMBLE (FULL stack, 4h, maker)\n")
    print(f"   {'book':10}" + "".join(f"{p:>16}" for p in PERIODS) + f"{'OOS %pos':>10}")
    rows = {}
    for which, label in [("ma", "MA"), ("bo", "BREAKOUT"), ("ens", "ENSEMBLE")]:
        line = f"   {label:10}"
        for plabel, (s, e) in PERIODS.items():
            pc, _, _ = _cells(which, s, e)
            mt = _book_mt(pc)
            rows[(label, plabel)] = mt
            line += f"{(str(mt.get('roi'))+'/'+str(mt.get('maxdd'))):>16}" if mt else f"{'--':>16}"
        _, oos_rois, _ = _cells(which, *PERIODS["OOS"])
        bpct = round(100 * float(np.mean(np.array(oos_rois) > 0))) if oos_rois else "?"
        line += f"{str(bpct)+'%':>10}"
        print(line)

    # scorecard p05 per book (UNSEEN sealed: series ends 2025-12-31)
    print("\n[SCORECARD] OOS-heldout block-bootstrap p05 per book (robust iff > 0):")
    sc = {}
    for which, label in [("ma", "MA"), ("bo", "BREAKOUT"), ("ens", "ENSEMBLE")]:
        _, _, series = _cells(which, "2018-01-01", "2025-12-31", date_index=True)
        allcells = [s for lst in series.values() for s in lst]
        if not allcells:
            continue
        book_4h = pd.concat(allcells, axis=1).mean(axis=1, skipna=True)
        daily = book_4h.resample("1D").apply(lambda x: float((1 + x).prod() - 1)).dropna()
        card = score_book(f"FULL_{label}_4h", daily)
        oosp = card["per_split"].get("OOS", {})
        hb = card["heldout_block_bootstrap"]
        sc[label] = {"oos_compound": oosp.get("compound_pct"), "oos_sharpe": oosp.get("sharpe"),
                     "oos_maxdd": oosp.get("maxdd_pct"), "heldout_p05": hb.get("p05"),
                     "unseen_n": card["per_split"].get("UNSEEN", {}).get("n", 0)}
        print(f"   {label:10} OOS compound {oosp.get('compound_pct'):>6}%  Sharpe {oosp.get('sharpe'):>5}  "
              f"maxDD {oosp.get('maxdd_pct'):>6}%  | OOS-heldout p05 {hb.get('p05'):>7}  (UNSEEN n={sc[label]['unseen_n']})")

    ens = sc.get("ENSEMBLE", {}); p05 = ens.get("heldout_p05")
    print(f"\n   VERDICT: ensemble OOS-heldout p05 = {p05} -> "
          f"{'CLEARS the robustness bar (p05>0)!' if (p05 is not None and p05 > 0) else 'still < 0 -- ceiling holds even diversified'}")
    out = ROOT.parent / "runs" / "periods" / "_OOS_CONFIRM" / "ensemble_arc.json"
    json.dump({"rows": {f"{k[0]}|{k[1]}": v for k, v in rows.items()}, "scorecard": sc}, open(out, "w"), indent=1, default=str)
    print(f"[json] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
