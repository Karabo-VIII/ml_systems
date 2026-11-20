"""src/strat/deep2020_optimal.py -- BLOCK E: optimal PARTICIPATION in a drift-bull (the actionable payoff).

Block D showed 2020 is a positive-DRIFT random walk (no per-bar momentum; ac1<0; Hurst~0.45). If the edge
is DRIFT, the optimal play is to MAXIMIZE time-in-market (capture the drift) and step aside ONLY for the
worst drawdowns -- i.e. LONG-BIASED + a DISASTER-STOP, not a signal-gated cross that sits flat 55-80% of
the time. This tests that hypothesis. Strategies on 2020 H2, per (instrument, TF):
  BUYHOLD            always long                                  (the drift ceiling; 100% participation)
  MA_FAMILY          the EMA slow-family cross + FULL stack       (the trend strat, flat most of the time)
  LONGBIAS_dd20/30   long by default; exit on -20%/-30% from the running peak; RE-ENTER on +10% from the
                     post-exit low (disaster-stop participation)
Report net%, maxDD%, capt-vs-BH, time-in, OOS net% (the grade). The question: does long-bias+disaster BEAT
the MA cross AND approach/keep up with BUYHOLD while cutting maxDD? RWYB: python -m strat.deep2020_optimal.
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
from strat.portfolio_replay import apply_trail_stop, MAKER_RT
from strat.replay_distinct_grid import distinct_specs
from strat.structural_fixes import min_hold
from strat.ma_type_upgrade import _MA, _nums
from strat.ma_2020_breakdown import _panel

WIN = ("2020-07-01", "2021-01-01"); SPLIT = "2020-10-01"
WARMUP = 400
CADENCES = ["1d", "4h", "2h", "1h", "30m", "15m"]
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]


def _disaster(c, dd, recov=0.10):
    """long-biased: exit on -dd from running peak; re-enter on +recov from post-exit low."""
    pos = np.ones(len(c)); state = 1; peak = c[0]; low = np.inf
    for i in range(len(c)):
        if state == 1:
            peak = max(peak, c[i])
            if c[i] <= peak * (1 - dd):
                state = 0; low = c[i]
        else:
            low = min(low, c[i])
            if c[i] >= low * (1 + recov):
                state = 1; peak = c[i]
        pos[i] = state
    return pos


def _ma_family_pos(c, slow):
    uniq = sorted({p for n in slow for p in _nums(n)}); cache = {p: _MA["EMA"](c, p) for p in uniq}
    poss = []
    for name in slow:
        pp = _nums(name); mas = [cache[p] for p in pp]
        h0 = np.nan_to_num((mas[0] > mas[1]) if len(pp) == 2 else ((mas[0] > mas[1]) & (mas[1] > mas[2]))).astype(np.int8)
        held = min_hold(apply_trail_stop(h0.copy(), c, 0.10)[0].astype(np.int8), 12).astype(np.float64)
        poss.append(held)
    return np.mean(poss, axis=0)


def _metrics(pos_held, c, ms, win):
    ret = np.zeros(len(c)); ret[1:] = c[1:] / c[:-1] - 1.0
    pos = np.zeros(len(c)); pos[1:] = pos_held[:-1]
    flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
    net = (pos * ret - flips * (MAKER_RT / 2.0))[win]
    dts = pd.to_datetime(ms[win], unit="ms")
    eq = np.cumprod(1 + net); pk = np.maximum.accumulate(eq); maxdd = float(((eq - pk) / pk).min() * 100)
    oos = dts >= pd.Timestamp(SPLIT)
    oos_net = float((np.prod(1 + net[oos]) - 1) * 100) if oos.sum() > 3 else float("nan")
    return {"net_pct": float((eq[-1] - 1) * 100), "maxdd_pct": maxdd, "time_in": float(np.mean(pos[win])), "oos_net": oos_net}


def main() -> int:
    global CADENCES
    if "--cadences" in sys.argv:
        CADENCES = sys.argv[sys.argv.index("--cadences") + 1].split(",")
    ma_cfg = {}
    for fam in ("2MA", "3MA"):
        ma_cfg.update(distinct_specs(fam, 0.15, max_n=60))
    PR.STRATS.update(ma_cfg)
    slow = [n for n in ma_cfg if 60 <= max(_nums(n)) < 150]
    print(f"BLOCK E optimal participation: {len(CADENCES)} TF; 2020 H2\n")

    out = {}
    for cad in CADENCES:
        print(f"########## {cad} -- BUYHOLD vs MA_FAMILY vs LONGBIAS+disaster (net% / maxDD% / capt-vs-BH / time-in / OOSnet%) ##########")
        print(f"   {'strategy':14} {'net%':>8} {'maxDD%':>8} {'capt/BH':>8} {'time_in':>8} {'OOSnet%':>8}")
        agg = {k: [] for k in ["BUYHOLD", "MA_FAMILY", "LONGBIAS_dd20", "LONGBIAS_dd30"]}
        for sym in SYMS:
            try:
                o, h, l, c, ms = _panel(sym, cad)
            except Exception:
                continue
            w0 = pd.Timestamp(WIN[0]).value // 10**6; w1 = pd.Timestamp(WIN[1]).value // 10**6
            e = int(np.searchsorted(ms, w1)); s0 = max(0, int(np.searchsorted(ms, w0)) - WARMUP)
            c2, ms2 = c[s0:e], ms[s0:e]
            if len(c2) < 60:
                continue
            win = ms2 >= w0
            if win.sum() < 50:
                continue
            bh = _metrics(np.ones(len(c2)), c2, ms2, win)
            agg["BUYHOLD"].append(bh)
            agg["MA_FAMILY"].append(_metrics(_ma_family_pos(c2, slow), c2, ms2, win))
            agg["LONGBIAS_dd20"].append(_metrics(_disaster(c2, 0.20), c2, ms2, win))
            agg["LONGBIAS_dd30"].append(_metrics(_disaster(c2, 0.30), c2, ms2, win))
        bh_net = float(np.mean([m["net_pct"] for m in agg["BUYHOLD"]])) if agg["BUYHOLD"] else float("nan")
        for k in ["BUYHOLD", "MA_FAMILY", "LONGBIAS_dd20", "LONGBIAS_dd30"]:
            ms_ = agg[k]
            if not ms_:
                continue
            net = float(np.mean([m["net_pct"] for m in ms_])); dd = float(np.mean([m["maxdd_pct"] for m in ms_]))
            ti = float(np.mean([m["time_in"] for m in ms_])); oo = float(np.mean([m["oos_net"] for m in ms_]))
            out[(cad, k)] = {"net": round(net, 1), "maxdd": round(dd, 1), "capt_bh": round(net / bh_net, 2) if bh_net else None, "time_in": round(ti, 2), "oos_net": round(oo, 1)}
            print(f"   {k:14} {net:>8.1f} {dd:>8.1f} {(net/bh_net if bh_net else 0):>8.2f} {ti:>8.2f} {oo:>8.1f}")
        print()

    op = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
    op.mkdir(parents=True, exist_ok=True)
    jt = "_".join(CADENCES)
    json.dump({f"{c}|{k}": v for (c, k), v in out.items()}, open(op / f"optimal_{jt}.json", "w"), indent=1, default=str)
    print(f"[json] {op / f'optimal_{jt}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
