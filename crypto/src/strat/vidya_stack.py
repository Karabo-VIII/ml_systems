"""src/strat/vidya_stack.py -- carry the VIDYA 2MA/3MA upgrade forward: stack it, test it on 3MA, stress the bear.

WHY (on-scope continuation of ma_type_upgrade): VIDYA upgraded the EMA 2MA cross (OOS +4.7 vs -0.0, bear
-0.8 vs -9.9). Three questions before believing it:
  (1) does VIDYA STACK with the drawdown overlays (trail10+minhold12+maker), or do they fight?
  (2) does VIDYA help the fragile 3MA too (3MA needs 3 MAs to align -- does adaptive smoothing help)?
  (3) is the bear-sidestep ROBUST across MULTIPLE bear/crash windows, not just Jun-2022?

Compares EMA vs VIDYA at FIXED + FULL on bear/VAL/OOS, on 2MA AND 3MA, + a multi-bear stress panel + the
OOS scorecard p05 for VIDYA-FULL. 4h, u10 book, UNSEEN sealed. RWYB: python -m strat.vidya_stack. No emoji.
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
from strat.portfolio_replay import apply_trail_stop, TAKER_RT, MAKER_RT
from strat.replay_distinct_grid import distinct_specs
from strat.ma_mechanics import _cached_panel
from strat.structural_fixes import min_hold
from strat.scorecard import score_book
from strat.ma_type_upgrade import held_cross, _nums

WARMUP = 600
TRANSFER = {"Jun2022_bear": ("2022-06-01", "2022-07-01"),
            "VAL": ("2024-05-15", "2025-03-15"), "OOS": ("2025-03-15", "2025-12-31")}
BEARS = {  # multi-bear/crash stress (does VIDYA's sidestep replicate?)
    "2021-05_crash": ("2021-05-01", "2021-06-01"),
    "2022-06_bear":  ("2022-06-01", "2022-07-01"),
    "2022-09_bear":  ("2022-09-01", "2022-10-01"),
    "2022-11_FTX":   ("2022-11-01", "2022-12-01"),
    "2024-08_unwind": ("2024-08-01", "2024-09-01"),
}


def _full_weight(c, periods, ma_type):
    h = held_cross(c, periods, ma_type).astype(np.int8)
    h = apply_trail_stop(h.copy(), c, 0.10)[0].astype(np.int8)
    return min_hold(h, 12).astype(np.float64)


def book(cfgs, ma_type, start, end, full, cost, date_index=False):
    s_ms = pd.Timestamp(start).value // 10**6
    e_ms = pd.Timestamp(end).value // 10**6
    syms = [a["symbol"] for a in yaml.safe_load(open(ROOT.parent / "config" / "universes" / "u10.yaml"))["assets"]]
    per_cell, series = [], {}
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
        for name in cfgs:
            if full:
                w = _full_weight(c2, _nums(name), ma_type)
            else:
                w = held_cross(c2, _nums(name), ma_type).astype(np.float64)
            pos = np.zeros(len(c2)); pos[1:] = w[:-1]
            flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
            net = (pos * ret - flips * (cost / 2.0))[wm]
            per_cell.append(net)
            if date_index:
                series.setdefault(sym, []).append(pd.Series(net, index=pd.to_datetime(ms2[wm], unit="ms")))
    if not per_cell:
        return {}, None
    m = min(len(x) for x in per_cell)
    bk = np.mean([x[:m] for x in per_cell], axis=0)
    eq = np.cumprod(1 + bk); peak = np.maximum.accumulate(eq)
    return {"roi": round(float(eq[-1] - 1) * 100, 1), "maxdd": round(float(((eq - peak) / peak).min() * 100), 1)}, series


def main() -> int:
    ma_cfg = {}
    for fam in ("2MA", "3MA"):
        ma_cfg.update(distinct_specs(fam, 0.15, max_n=60))
    PR.STRATS.update(ma_cfg)
    slow2 = [n for n in ma_cfg if len(_nums(n)) == 2 and 60 <= max(_nums(n)) < 150]
    slow3 = [n for n in ma_cfg if len(_nums(n)) == 3 and 60 <= max(_nums(n)) < 150]
    print(f"VIDYA stack: 2MA-slow={len(slow2)}, 3MA-slow={len(slow3)}\n")

    # (1)+(2) EMA vs VIDYA, FIXED + FULL, on 2MA and 3MA
    for fam_label, cfgs in [("2MA-slow", slow2), ("3MA-slow", slow3)]:
        print(f"## {fam_label}: EMA vs VIDYA, FIXED + FULL stack -- ROI%/maxDD% per period")
        print(f"   {'variant':16}" + "".join(f"{p:>16}" for p in TRANSFER))
        for ma_type in ("EMA", "VIDYA"):
            for full, lvl in [(False, "FIXED"), (True, "FULL")]:
                cost = MAKER_RT if full else TAKER_RT
                line = f"   {ma_type+'_'+lvl:16}"
                for plabel, (s, e) in TRANSFER.items():
                    m, _ = book(cfgs, ma_type, s, e, full, cost)
                    line += f"{(str(m.get('roi'))+'/'+str(m.get('maxdd'))):>16}" if m else f"{'--':>16}"
                print(line)
        print()

    # (3) multi-bear stress: EMA vs VIDYA (FIXED), 2MA-slow -- does the sidestep replicate?
    print("## Multi-bear stress (2MA-slow, FIXED): EMA vs VIDYA ROI% -- does VIDYA sidestep the bear?")
    print(f"   {'bear window':18}{'EMA':>10}{'VIDYA':>10}{'delta':>10}")
    wins = 0
    for blabel, (s, e) in BEARS.items():
        em, _ = book(slow2, "EMA", s, e, False, TAKER_RT)
        vi, _ = book(slow2, "VIDYA", s, e, False, TAKER_RT)
        d = (vi.get("roi", 0) - em.get("roi", 0)) if (em and vi) else None
        wins += int(d is not None and d > 0)
        print(f"   {blabel:18}{str(em.get('roi')):>10}{str(vi.get('roi')):>10}{(f'{d:+.1f}' if d is not None else '?'):>10}")
    print(f"   -> VIDYA less-bad than EMA in {wins}/{len(BEARS)} bear windows")

    # (4) OOS scorecard p05 for VIDYA-FULL vs EMA-FULL (2MA-slow), UNSEEN sealed
    print("\n## OOS scorecard (2MA-slow FULL stack, UNSEEN sealed):")
    sc = {}
    for ma_type in ("EMA", "VIDYA"):
        _, series = book(slow2, ma_type, "2018-01-01", "2025-12-31", True, MAKER_RT, date_index=True)
        if not series:
            continue
        allc = [x for lst in series.values() for x in lst]
        daily = pd.concat(allc, axis=1).mean(axis=1, skipna=True).resample("1D").apply(lambda x: float((1 + x).prod() - 1)).dropna()
        card = score_book(f"2MA_{ma_type}_FULL", daily)
        oosp = card["per_split"].get("OOS", {}); hb = card["heldout_block_bootstrap"]
        sc[ma_type] = {"oos_compound": oosp.get("compound_pct"), "oos_sharpe": oosp.get("sharpe"), "heldout_p05": hb.get("p05"),
                       "unseen_n": card["per_split"].get("UNSEEN", {}).get("n", 0)}
        print(f"   {ma_type+'_FULL':12} OOS compound {str(oosp.get('compound_pct')):>7}%  Sharpe {str(oosp.get('sharpe')):>5}  "
              f"OOS-heldout p05 {str(hb.get('p05')):>7}  (UNSEEN n={sc[ma_type]['unseen_n']})")
    ep = sc.get("EMA", {}).get("heldout_p05"); vp = sc.get("VIDYA", {}).get("heldout_p05")
    if ep is not None and vp is not None:
        print(f"\n   VERDICT: VIDYA-FULL p05 {vp} vs EMA-FULL p05 {ep} -> "
              f"{'CLEARS p05>0 -- robust!' if vp > 0 else ('improves but still <0' if vp > ep else 'no improvement')}")
    out = ROOT.parent / "runs" / "periods" / "_OOS_CONFIRM" / "vidya_stack.json"
    json.dump(sc, open(out, "w"), indent=1, default=str)
    print(f"[json] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
