"""src/strat/deep2020_calendar_robust.py -- BLOCK Q: is the calendar edge BROAD + not multiple-comparison luck?

The headline finding (calendar beats BH) must be verified before over-claiming. Two tests:
  BREADTH      -- per-asset day-of-week means: how many of u10 have Saturday positive / Tuesday negative?
                  (broad = real structure; 1-2 assets = concentration).
  PLACEBO      -- the calendar tilt sits out the VAL-negative weekdays. Does that BEAT sitting out the SAME
                  NUMBER of RANDOM weekdays (300 perms)? If the calendar-selected days don't beat random
                  day-exclusion, the edge was multiple-comparison luck. p = frac(random OOS-net >= calendar).
RWYB: python -m strat.deep2020_calendar_robust. No emoji (cp1252).
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

from strat.ma_2020_breakdown import _panel

WIN = ("2020-07-01", "2021-01-01"); SPLIT = "2020-10-01"
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _asset_ret(sym):
    w0 = pd.Timestamp(WIN[0]).value // 10**6; w1 = pd.Timestamp(WIN[1]).value // 10**6
    try:
        o, h, l, c, ms = _panel(sym, "1d")
    except Exception:
        return None
    keep = (ms >= w0) & (ms < w1); c2 = c[keep]
    if len(c2) < 50:
        return None
    r = np.zeros(len(c2)); r[1:] = c2[1:] / c2[:-1] - 1
    return pd.Series(r, index=pd.to_datetime(ms[keep], unit="ms"))


def main() -> int:
    # ---- BREADTH ----
    print("BLOCK Q calendar robustness\n## (1) BREADTH -- per-asset day-of-week sign (2020 H2):")
    print(f"   {'sym':6} " + " ".join(f"{d:>5}" for d in DOW))
    sat_pos = tue_neg = n = 0
    for sym in SYMS:
        r = _asset_ret(sym)
        if r is None:
            continue
        n += 1
        m = r.groupby(r.index.dayofweek).mean() * 1e4
        print(f"   {sym.replace('USDT',''):6} " + " ".join(f"{m.get(k,0):>5.0f}" for k in range(7)))
        if m.get(5, 0) > 0:
            sat_pos += 1
        if m.get(1, 0) < 0:
            tue_neg += 1
    print(f"   -> Saturday positive in {sat_pos}/{n} assets;  Tuesday negative in {tue_neg}/{n} assets (bp/day)")

    # ---- PLACEBO ----
    cols = [_asset_ret(s) for s in SYMS]; cols = [c for c in cols if c is not None]
    book = pd.concat(cols, axis=1).mean(axis=1)
    val = book[book.index < pd.Timestamp(SPLIT)]; oos = book[book.index >= pd.Timestamp(SPLIT)]
    valm = val.groupby(val.index.dayofweek).mean()
    bad = [int(k) for k, v in valm.items() if v < 0]
    def net_excl(days):
        s = oos.where(~oos.index.dayofweek.isin(days), 0.0)
        return float(np.prod(1 + s.to_numpy()) - 1) * 100
    cal_net = net_excl(bad); bh_net = float(np.prod(1 + oos.to_numpy()) - 1) * 100
    rng = np.random.default_rng(7); K = len(bad); perms = []
    alldays = list(range(7))
    for _ in range(300):
        perms.append(net_excl(list(rng.choice(alldays, size=max(1, K), replace=False))))
    perms = np.array(perms)
    p = float(np.mean(perms >= cal_net))
    print(f"\n## (2) PLACEBO -- calendar excludes VAL-bad days {bad} ({[DOW[d] for d in bad]}); K={K}")
    print(f"   OOS net: calendar-excluded {cal_net:.1f}%  |  buy-hold {bh_net:.1f}%  |  random-{K}-day-exclusion mean {perms.mean():.1f}%")
    print(f"   permutation p (random >= calendar) = {p:.3f}  -> "
          f"{'calendar selection BEATS random day-exclusion (real, p<0.10)' if p < 0.10 else 'NOT better than random -- multiple-comparison luck'}")
    out = {"sat_pos": f"{sat_pos}/{n}", "tue_neg": f"{tue_neg}/{n}", "bad_days": [DOW[d] for d in bad],
           "cal_net": round(cal_net, 1), "bh_net": round(bh_net, 1), "random_mean": round(float(perms.mean()), 1), "placebo_p": round(p, 3)}
    op = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
    op.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(op / "calendar_robust.json", "w"), indent=1, default=str)
    print(f"\n[json] {op / 'calendar_robust.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
