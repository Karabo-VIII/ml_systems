"""src/strat/deep2020_calendar_tilt.py -- BLOCK O: the CAUSAL calendar tilt (the constructive payoff).

Block N: the day-of-week structure persists (7/7, corr 0.80). This tests it CAUSALLY: classify each weekday
as GOOD/BAD from the FIRST half (Jul-Sep = VAL), then on the SECOND half (Oct-Dec = OOS) run a long-only
strategy that holds full on GOOD-VAL days and sits out (or half-sizes) on BAD-VAL days. Does sitting out the
historically-negative days (Tue/Wed) BEAT flat buy-hold OOS? Long-only spot lev=1 -> the tilt can only
DE-RISK on bad days (cannot over-weight). Compare net / Sharpe / return-per-time-in vs flat BH, + a
vol-target combo. Honest causal grade. RWYB: python -m strat.deep2020_calendar_tilt. No emoji (cp1252).
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
TAKER = 0.0024


def _book(cad):
    w0 = pd.Timestamp(WIN[0]).value // 10**6; w1 = pd.Timestamp(WIN[1]).value // 10**6
    cols = []
    for sym in SYMS:
        try:
            o, h, l, c, ms = _panel(sym, cad)
        except Exception:
            continue
        keep = (ms >= w0) & (ms < w1); c2 = c[keep]
        if len(c2) < 20:
            continue
        r = np.zeros(len(c2)); r[1:] = c2[1:] / c2[:-1] - 1
        cols.append(pd.Series(r, index=pd.to_datetime(ms[keep], unit="ms")))
    return pd.concat(cols, axis=1).mean(axis=1).dropna()


def _stats(r, ann=365):
    x = r.to_numpy()
    if len(x) < 5:
        return {}
    eq = np.cumprod(1 + x); pk = np.maximum.accumulate(eq); dd = float(((eq - pk) / pk).min() * 100)
    sh = float(np.mean(x) / (np.std(x) + 1e-12) * np.sqrt(ann))
    return {"net": round(float(eq[-1] - 1) * 100, 1), "maxdd": round(dd, 1), "sharpe": round(sh, 2), "time_in": round(float(np.mean((x != 0))), 2)}


def main() -> int:
    b = _book("1d")
    val = b[b.index < pd.Timestamp(SPLIT)]; oos = b[b.index >= pd.Timestamp(SPLIT)]
    # classify weekdays GOOD/BAD on VAL
    dow_mean = val.groupby(val.index.dayofweek).mean()
    good = set(int(k) for k, v in dow_mean.items() if v > 0)
    bad = set(range(7)) - good
    print(f"BLOCK O causal calendar tilt -- GOOD weekdays (VAL>0): {sorted(good)}  BAD: {sorted(bad)}\n")
    out = {}
    # OOS strategies
    bh = oos.copy()
    derisk0 = oos.where(oos.index.dayofweek.isin(list(good)), 0.0)        # flat on BAD days
    derisk_half = oos.where(oos.index.dayofweek.isin(list(good)), oos * 0.5)  # half-size on BAD days
    # cost (turnover on day-class changes)
    def cost(mask):
        t = np.abs(np.diff(np.concatenate([[0], mask.astype(int)]))).sum()
        return t * (TAKER / 2)
    g = oos.index.dayofweek.isin(list(good))
    print(f"   {'OOS strategy':16} {'net%':>8} {'maxDD%':>8} {'Sharpe':>7} {'time_in':>8}")
    for label, series, c in [("BUYHOLD", bh, 0.0),
                             ("CAL_flat_bad", derisk0, cost(g) * 100),
                             ("CAL_half_bad", derisk_half, cost(g) * 100)]:
        s = _stats(series); s["net"] = round(s["net"] - c, 1) if s else s
        out[label] = s
        if s:
            print(f"   {label:16} {s['net']:>8} {s['maxdd']:>8} {s['sharpe']:>7} {s['time_in']:>8}")
    bhn = out["BUYHOLD"]["net"]
    print(f"\n   per-time-in: BUYHOLD {bhn:.0f}% | CAL_flat_bad {out['CAL_flat_bad']['net']/max(out['CAL_flat_bad']['time_in'],1e-9):.0f}%")
    beat_sharpe = out["CAL_flat_bad"]["sharpe"] > out["BUYHOLD"]["sharpe"]
    print(f"   VERDICT (causal, VAL->OOS): sitting out historically-bad weekdays {'IMPROVES Sharpe vs BH' if beat_sharpe else 'does not beat BH'} "
          f"(CAL {out['CAL_flat_bad']['sharpe']} vs BH {out['BUYHOLD']['sharpe']}); "
          f"net {out['CAL_flat_bad']['net']} vs {bhn} (de-risking gives up some bull, as expected).")
    op = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
    op.mkdir(parents=True, exist_ok=True)
    json.dump({"good_weekdays": sorted(good), "bad_weekdays": sorted(bad), "oos": out}, open(op / "calendar_tilt.json", "w"), indent=1, default=str)
    print(f"[json] {op / 'calendar_tilt.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
