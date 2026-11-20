"""src/strat/breakout_arc.py -- does the MA-arc PATTERN generalize to a different entry archetype?

WHY (founding mandate, MEMORY.md: "explore ALL indicators broadly+freshly; treat 'exhausted' as a
hypothesis"): the whole fixed-approach arc tested ONE archetype (MA-cross). This runs a structurally
DIFFERENT one -- Donchian breakout (long when close breaks the N-bar high; exit on the M-bar low) --
through the same rigorous gauntlet to test whether the three MA findings GENERALIZE:
  (1) does picking a robust (slower-N) breakout FAMILY beat run-everything, IN and OUT of sample?
  (2) is the trail+min_hold stack a relative drawdown-reducer here too?
  (3) does it clear ABSOLUTE robustness (scorecard p05>0/PBO/breadth) -- or hit the same ceiling?

VARIANTS (4h, equal-weight u10 book, causal MtM):
  NAIVE   all Donchian(N,M) configs, TAKER
  FIXED   slow-N family (entry N in [50,150]), TAKER
  FULL    FIXED + TRAIL(10%) + min_hold(12) + MAKER
Periods: bear(Jun2022) / VAL / OOS (UNSEEN sealed) + a canonical scorecard grade on FULL (series ends
2025-12-31 -> UNSEEN untouched). RWYB: python -m strat.breakout_arc. No emoji (cp1252).
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

from strat.portfolio_replay import apply_trail_stop, TAKER_RT, MAKER_RT
from strat.ma_mechanics import _cached_panel
from strat.structural_fixes import min_hold
from strat.scorecard import score_book

WARMUP = 600
ENTRY_N = [20, 30, 40, 55, 80, 100, 120, 150]
EXIT_M = [10, 20, 30, 50]
PERIODS = {"Jun2022_bear": ("2022-06-01", "2022-07-01"),
           "VAL": ("2024-05-15", "2025-03-15"), "OOS": ("2025-03-15", "2025-12-31")}


def _roll_max(a, n):
    s = pd.Series(a); return s.rolling(n, min_periods=n).max().shift(1).to_numpy()   # causal: excludes t


def _roll_min(a, n):
    s = pd.Series(a); return s.rolling(n, min_periods=n).min().shift(1).to_numpy()


def donchian_held(c, n, m):
    """long-only Donchian: enter when close > prior N-bar high, exit when close < prior M-bar low."""
    hi = _roll_max(c, n); lo = _roll_min(c, m)
    held = np.zeros(len(c), dtype=np.int8)
    cur = 0
    for i in range(len(c)):
        if np.isnan(hi[i]) or np.isnan(lo[i]):
            cur = 0
        elif cur == 0 and c[i] > hi[i]:
            cur = 1
        elif cur == 1 and c[i] < lo[i]:
            cur = 0
        held[i] = cur
    return held


def _configs():
    cfgs = {}
    for n in ENTRY_N:
        for m in EXIT_M:
            if m < n:
                cfgs[f"dc_{n}_{m}"] = (n, m)
    slow = [k for k, (n, m) in cfgs.items() if 50 <= n <= 150]
    return cfgs, slow


def _held_full(c, n, m, full):
    h = donchian_held(c, n, m)
    if full:
        h = apply_trail_stop(h.copy().astype(np.int8), c, 0.10)[0].astype(np.int8)
        h = min_hold(h, 12).astype(np.int8)
    return h


def book(cfgs, names, start, end, full, cost, date_index=False):
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
        c2, ms2 = c[s_idx:e_idx], ms[s_idx:e_idx]
        if len(c2) < 30:
            continue
        wm = ms2 >= s_ms
        if wm.sum() < 10:
            continue
        ret = np.zeros(len(c2)); ret[1:] = c2[1:] / c2[:-1] - 1.0
        for name in names:
            n, m = cfgs[name]
            held = _held_full(c2, n, m, full).astype(np.float64)
            pos = np.zeros(len(c2)); pos[1:] = held[:-1]
            flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
            net = (pos * ret - flips * (cost / 2.0))
            per_cell.append(net[wm])
            cell_roi.append(float(np.cumprod(1 + net[wm])[-1] - 1) * 100)
            if date_index:
                series.setdefault(sym, []).append(pd.Series(net[wm], index=pd.to_datetime(ms2[wm], unit="ms")))
    if not per_cell:
        return {}, [], None
    mlen = min(len(x) for x in per_cell)
    bk = np.mean([x[:mlen] for x in per_cell], axis=0)
    eq = np.cumprod(1 + bk); peak = np.maximum.accumulate(eq)
    mt = {"roi": round(float(eq[-1] - 1) * 100, 1), "maxdd": round(float(((eq - peak) / peak).min() * 100), 1)}
    return mt, cell_roi, series


def main() -> int:
    cfgs, slow = _configs()
    naive = list(cfgs)
    print(f"breakout arc: {len(naive)} Donchian configs; slow-N[50,150] family = {len(slow)}\n")
    print(f"   {'variant':8}" + "".join(f"{p:>16}" for p in PERIODS) + f"{'OOS %posCells':>16}")
    rows = {}
    for vlabel, names, full, cost in [("NAIVE", naive, False, TAKER_RT), ("FIXED", slow, False, TAKER_RT),
                                       ("FULL", slow, True, MAKER_RT)]:
        line = f"   {vlabel:8}"
        for plabel, (s, e) in PERIODS.items():
            mt, rois, _ = book(cfgs, names, s, e, full, cost)
            rows[(vlabel, plabel)] = mt
            line += f"{(str(mt.get('roi'))+'/'+str(mt.get('maxdd'))):>16}" if mt else f"{'--':>16}"
        # OOS breadth (% of cells positive)
        _, oos_rois, _ = book(cfgs, names, *PERIODS["OOS"], full, cost)
        bpct = str(round(100 * float(np.mean(np.array(oos_rois) > 0)))) + "%" if oos_rois else "?"
        line += f"{bpct:>16}"
        print(line)

    # transfer deltas
    print("\n[TRANSFER] FIXED vs NAIVE (family fix) and FULL vs NAIVE, per period (ROI delta)")
    for plabel in PERIODS:
        n = rows[("NAIVE", plabel)].get("roi", np.nan)
        f = rows[("FIXED", plabel)].get("roi", np.nan)
        fu = rows[("FULL", plabel)].get("roi", np.nan)
        print(f"   {plabel:14} NAIVE {n:>6.1f}  FIXED {f:>6.1f} (d{f-n:+.1f})  FULL {fu:>6.1f} (d{fu-n:+.1f})")

    # canonical scorecard on FULL (UNSEEN sealed: series ends 2025-12-31)
    _, _, series = book(cfgs, slow, "2018-01-01", "2025-12-31", True, MAKER_RT, date_index=True)
    if series:
        allcells = [s for lst in series.values() for s in lst]
        book_4h = pd.concat(allcells, axis=1).mean(axis=1, skipna=True)
        daily = book_4h.resample("1D").apply(lambda x: float((1 + x).prod() - 1)).dropna()
        card = score_book("FULL_breakout_4h", daily)
        oosp = card["per_split"].get("OOS", {})
        hb = card["heldout_block_bootstrap"]
        print(f"\n[SCORECARD] FULL breakout (UNSEEN n={card['per_split'].get('UNSEEN',{}).get('n',0)}, sealed):")
        print(f"   OOS compound {oosp.get('compound_pct')}%  maxDD {oosp.get('maxdd_pct')}%  Sharpe {oosp.get('sharpe')}")
        print(f"   OOS-heldout block-bootstrap p05 {hb.get('p05')} (robust iff >0)")
        out = ROOT.parent / "runs" / "periods" / "_OOS_CONFIRM" / "breakout_arc.json"
        json.dump({"rows": {f"{k[0]}|{k[1]}": v for k, v in rows.items()}, "scorecard": card}, open(out, "w"), indent=1, default=str)
        print(f"\n[json] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
