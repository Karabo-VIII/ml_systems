"""src/strat/deep2020_participation.py -- BLOCK A: how much of the 2020 bull do we actually PARTICIPATE in?

User /orc 2026-06-13 (6h deep-dive): "we are NOT optimally participating within that bull, right?" This
quantifies it rigorously. For each (instrument, timeframe, MA-class family) on the 2020 H2 window (the last
6 months = VAL Jul-Sep + OOS Oct-Dec) it decomposes every bar's LOG return by (position x sign):
  UPSIDE CAPTURED   = sum(pos * logret) over UP bars    -> the bull we rode
  UPSIDE MISSED     = sum((1-pos) * logret) over UP bars -> the bull we sat out  (the participation GAP)
  DOWNSIDE AVOIDED  = sum((1-pos) * logret) over DOWN bars -> the drawdown we dodged (our value-add)
  DOWNSIDE GIVEN-BK = sum(pos * logret) over DOWN bars  -> the drawdown we ate while long
Then the two behavioral rates: UPSIDE-CAPTURE = captured / total_up (want high in a bull) and
DOWNSIDE-AVOIDANCE = avoided / total_down (want high). pos = the FAMILY-AVERAGE position (fraction of the
slow 2MA+3MA family long), FULL stack. Identity: bh_log = captured+missed+avoided+givenback;
strat_log(gross) = captured + givenback. RWYB: python -m strat.deep2020_participation --cadences <tf>.
No emoji (cp1252).
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

import strat.portfolio_replay as PR
from strat.portfolio_replay import apply_trail_stop, MAKER_RT
from strat.replay_distinct_grid import distinct_specs
from strat.structural_fixes import min_hold
from strat.ma_type_upgrade import _MA, _nums, MA_TYPES
from strat.ma_2020_breakdown import _panel

WIN = ("2020-07-01", "2021-01-01")     # 2020 H2 = VAL + OOS
WARMUP = 400
CADENCES = ["1d", "4h", "2h", "1h", "30m", "15m"]
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]


def _family_pos_and_ret(sym, cad, ma_type, slow):
    """family-average position (fraction long, FULL stack) + log returns, on the WIN window."""
    try:
        o, h, l, c, ms = _panel(sym, cad)
    except Exception:
        return None
    w0 = pd.Timestamp(WIN[0]).value // 10**6; w1 = pd.Timestamp(WIN[1]).value // 10**6
    e = int(np.searchsorted(ms, w1)); s0 = max(0, int(np.searchsorted(ms, w0)) - WARMUP)
    c2, ms2 = c[s0:e], ms[s0:e]
    if len(c2) < 40:
        return None
    win = ms2 >= w0
    if win.sum() < 30:
        return None
    uniq = sorted({p for n in slow for p in _nums(n)})
    cache = {p: _MA[ma_type](c2, p) for p in uniq}
    poss = []
    for name in slow:
        pp = _nums(name); mas = [cache[p] for p in pp]
        h0 = np.nan_to_num((mas[0] > mas[1]) if len(pp) == 2 else ((mas[0] > mas[1]) & (mas[1] > mas[2]))).astype(np.int8)
        h1 = apply_trail_stop(h0.copy(), c2, 0.10)[0].astype(np.int8)
        held = min_hold(h1, 12).astype(np.float64)
        pos = np.zeros(len(c2)); pos[1:] = held[:-1]
        poss.append(pos)
    fpos = np.mean(poss, axis=0)                         # fraction of family long
    logret = np.zeros(len(c2)); logret[1:] = np.log(c2[1:] / c2[:-1])
    flips = np.abs(np.diff(np.concatenate([[0.0], fpos])))
    cost_log = flips * (MAKER_RT / 2.0)                  # approx per-bar cost in return units
    return fpos[win], logret[win], cost_log[win]


def _decompose(fpos, logret, cost):
    up = logret > 0; dn = logret < 0
    captured = float(np.sum(fpos[up] * logret[up]))
    missed = float(np.sum((1 - fpos[up]) * logret[up]))
    givenbk = float(np.sum(fpos[dn] * logret[dn]))
    avoided = float(np.sum((1 - fpos[dn]) * logret[dn]))   # negative (down bars we dodged)
    total_up = float(np.sum(logret[up])); total_dn = float(np.sum(logret[dn]))
    strat_gross = captured + givenbk
    strat_net = strat_gross - float(np.sum(cost))
    bh = captured + missed + givenbk + avoided
    return {
        "upside_capture": round(captured / total_up, 3) if total_up > 0 else None,       # frac of the bull ridden
        "downside_avoidance": round(avoided / total_dn, 3) if total_dn < 0 else None,     # frac of drops dodged
        "time_in_mkt": round(float(np.mean(fpos)), 3),
        "strat_net_pct": round((np.exp(strat_net) - 1) * 100, 1),
        "buyhold_pct": round((np.exp(bh) - 1) * 100, 1),
        "capture_vs_bh": round((np.exp(strat_net) - 1) / (np.exp(bh) - 1), 3) if (np.exp(bh) - 1) != 0 else None,
        "cost_pct": round((1 - np.exp(-float(np.sum(cost)))) * 100, 2),
        # log-unit contributions (additive) as % of |bh_log|
        "captured_log": round(captured, 4), "missed_log": round(missed, 4),
        "avoided_log": round(avoided, 4), "givenbk_log": round(givenbk, 4),
    }


def main() -> int:
    global CADENCES
    if "--cadences" in sys.argv:
        CADENCES = sys.argv[sys.argv.index("--cadences") + 1].split(",")
    ma_cfg = {}
    for fam in ("2MA", "3MA"):
        ma_cfg.update(distinct_specs(fam, 0.15, max_n=60))
    PR.STRATS.update(ma_cfg)
    slow = [n for n in ma_cfg if 60 <= max(_nums(n)) < 150]
    print(f"BLOCK A participation: {len(slow)} configs x {len(MA_TYPES)} classes x {len(CADENCES)} TF; 2020 H2\n")

    out = {}
    for cad in CADENCES:
        print(f"########## {cad} -- upside-CAPTURE / downside-AVOID / time-in-mkt / net-vs-BH (family avg over instruments) ##########")
        print(f"   {'MA':6} {'up_capt':>8} {'dn_avoid':>9} {'time_in':>8} {'strat%':>8} {'BH%':>8} {'capt/BH':>8}")
        for mt in MA_TYPES:
            rows = []
            for sym in SYMS:
                r = _family_pos_and_ret(sym, cad, mt, slow)
                if r is None:
                    continue
                d = _decompose(*r)
                out[(cad, mt, sym)] = d
                rows.append(d)
            if not rows:
                continue
            def avg(k):
                vals = [r[k] for r in rows if r[k] is not None]
                return float(np.mean(vals)) if vals else float("nan")
            print(f"   {mt:6} {avg('upside_capture'):>8.2f} {avg('downside_avoidance'):>9.2f} {avg('time_in_mkt'):>8.2f} "
                  f"{avg('strat_net_pct'):>8.1f} {avg('buyhold_pct'):>8.1f} {avg('capture_vs_bh'):>8.2f}")
        print()

    op = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
    op.mkdir(parents=True, exist_ok=True)
    jt = "_".join(CADENCES)
    json.dump({f"{c}|{m}|{s}": d for (c, m, s), d in out.items()}, open(op / f"participation_{jt}.json", "w"), indent=1, default=str)
    print(f"[json] {op / f'participation_{jt}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
