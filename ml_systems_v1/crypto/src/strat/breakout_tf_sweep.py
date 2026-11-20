"""src/strat/breakout_tf_sweep.py -- SWEEP breakout across timeframes (HARD RULE: never default one cadence).

WHY: breakout beat MA OOS but breakout_arc.py tested ONLY 4h -- a violation of the standing HARD RULE
(feedback-sweep-all-timeframes-never-default-one). Cadence materially changes the answer (the MA arc:
coarser helps the bear, 1h was MA's sweet spot). Breakout is the most promising lead, so sweep its FULL
stack across {1d, 4h, 1h, 30m} and find the cadence with the best held-out robustness (OOS p05/Sharpe).

FULL breakout stack (slow-N family + trail10 + minhold12 + maker), u10 book, UNSEEN sealed.
RWYB: python -m strat.breakout_tf_sweep. No emoji (cp1252).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strat.portfolio_replay import apply_trail_stop, MAKER_RT
from strat.ma_mechanics import _cached_panel
from strat.structural_fixes import min_hold
from strat.scorecard import score_book
from strat.breakout_arc import donchian_held, _configs as _bo_configs

WARMUP = 600
CADENCES = ["1d", "4h", "1h", "30m"]
PERIODS = {"Jun2022_bear": ("2022-06-01", "2022-07-01"),
           "VAL": ("2024-05-15", "2025-03-15"), "OOS": ("2025-03-15", "2025-12-31")}


def _bo_full(c, n, m):
    h = donchian_held(c, n, m).astype(np.int8)
    h = apply_trail_stop(h.copy(), c, 0.10)[0].astype(np.int8)
    return min_hold(h, 12).astype(np.float64)


def book(slow, cfgs, cadence, start, end, date_index=False):
    s_ms = pd.Timestamp(start).value // 10**6
    e_ms = pd.Timestamp(end).value // 10**6
    syms = [a["symbol"] for a in yaml.safe_load(open(ROOT.parent / "config" / "universes" / "u10.yaml"))["assets"]]
    per_cell, cell_roi, series = [], [], {}
    for sym in syms:
        try:
            o, h, l, c, ms = _cached_panel(sym, cadence)
        except Exception:
            continue
        e_idx = int(np.searchsorted(ms, e_ms))
        s_idx = max(0, int(np.searchsorted(ms, s_ms)) - WARMUP)
        c2, ms2 = c[s_idx:e_idx], ms[s_idx:e_idx]
        if len(c2) < 30:
            continue
        wm = ms2 >= s_ms
        if wm.sum() < 10:
            continue
        ret = np.zeros(len(c2)); ret[1:] = c2[1:] / c2[:-1] - 1.0
        for name in slow:
            n, m = cfgs[name]
            w = _bo_full(c2, n, m)
            pos = np.zeros(len(c2)); pos[1:] = w[:-1]
            flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
            net = (pos * ret - flips * (MAKER_RT / 2.0))
            per_cell.append(net[wm])
            cell_roi.append(float(np.cumprod(1 + net[wm])[-1] - 1) * 100)
            if date_index:
                series.setdefault(sym, []).append(pd.Series(net[wm], index=pd.to_datetime(ms2[wm], unit="ms")))
    if not per_cell:
        return {}, [], None
    m = min(len(x) for x in per_cell)
    bk = np.mean([x[:m] for x in per_cell], axis=0)
    eq = np.cumprod(1 + bk); peak = np.maximum.accumulate(eq)
    return {"roi": round(float(eq[-1] - 1) * 100, 1), "maxdd": round(float(((eq - peak) / peak).min() * 100), 1)}, cell_roi, series


def main() -> int:
    cfgs, slow = _bo_configs()
    print(f"breakout TF sweep: FULL stack, slow-N family={len(slow)}, cadences {CADENCES}\n")
    print(f"   {'cadence':8}" + "".join(f"{p:>16}" for p in PERIODS) +
          f"{'OOS %pos':>10}{'OOS Sharpe':>12}{'OOS p05':>10}")
    rows = {}
    for cad in CADENCES:
        line = f"   {cad:8}"
        for plabel, (s, e) in PERIODS.items():
            mt, _, _ = book(slow, cfgs, cad, s, e)
            rows[(cad, plabel)] = mt
            line += f"{(str(mt.get('roi'))+'/'+str(mt.get('maxdd'))):>16}" if mt else f"{'--':>16}"
        # OOS breadth + scorecard p05 (UNSEEN sealed)
        _, oos_rois, _ = book(slow, cfgs, cad, *PERIODS["OOS"])
        bpct = round(100 * float(np.mean(np.array(oos_rois) > 0))) if oos_rois else "?"
        _, _, series = book(slow, cfgs, cad, "2018-01-01", "2025-12-31", date_index=True)
        sharpe = p05 = None
        if series:
            allcells = [x for lst in series.values() for x in lst]
            b4 = pd.concat(allcells, axis=1).mean(axis=1, skipna=True)
            daily = b4.resample("1D").apply(lambda x: float((1 + x).prod() - 1)).dropna()
            card = score_book(f"bo_{cad}", daily)
            sharpe = card["per_split"].get("OOS", {}).get("sharpe")
            p05 = card["heldout_block_bootstrap"].get("p05")
            rows[(cad, "scorecard")] = {"oos_sharpe": sharpe, "heldout_p05": p05, "oos_breadth_pct": bpct,
                                        "unseen_n": card["per_split"].get("UNSEEN", {}).get("n", 0)}
        line += f"{str(bpct)+'%':>10}{str(sharpe):>12}{str(p05):>10}"
        print(line)

    # best cadence by OOS p05 (closest to / above 0)
    cad_p05 = {cad: rows.get((cad, "scorecard"), {}).get("heldout_p05") for cad in CADENCES}
    valid = {k: v for k, v in cad_p05.items() if v is not None}
    if valid:
        best = max(valid, key=lambda k: valid[k])
        print(f"\n   best cadence by OOS-heldout p05: {best} (p05 {valid[best]})  "
              f"{'-> CLEARS p05>0!' if valid[best] > 0 else '-> still <0, ceiling holds'}")
    out = ROOT.parent / "runs" / "periods" / "_OOS_CONFIRM" / "breakout_tf_sweep.json"
    json.dump({f"{k[0]}|{k[1]}": v for k, v in rows.items()}, open(out, "w"), indent=1, default=str)
    print(f"[json] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
