"""src/strat/deep2020_voltarget.py -- BLOCK J: the best way to HOLD the drift -- vol-targeting vs flat BH.

The deep-dive concluded the only 2020 edge is the drift-beta, best captured by HOLDING it (buy-hold). The
natural risk-management question: is vol-TARGETED holding better risk-adjusted than FLAT buy-hold? Scale
long-only exposure inversely to recent realized vol (target a constant vol), capped [0,1] (no leverage).
  BUYHOLD       exposure = 1 always
  VOLTGT_lo/hi  exposure = clip(target_vol / realized_vol[t-1], 0, 1)   (two target levels)
Compare net% / maxDD% / Sharpe / realized-vol. Vol-targeting should lift Sharpe + cut maxDD (de-risk in
high-vol stretches) for ~similar return. Per instrument + book, 2020 H2. RWYB: python -m strat.deep2020_voltarget.
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

from strat.ma_2020_breakdown import _panel

WIN = ("2020-07-01", "2021-01-01"); SPLIT = "2020-10-01"
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]
VOLWIN = {"1d": 14, "4h": 84}        # ~2 weeks realized-vol lookback


def _series(cad):
    w0 = pd.Timestamp(WIN[0]).value // 10**6; w1 = pd.Timestamp(WIN[1]).value // 10**6
    cols = {}
    for sym in SYMS:
        try:
            o, h, l, c, ms = _panel(sym, cad)
        except Exception:
            continue
        keep = (ms >= w0 - 30 * 86400000) & (ms < w1)
        cols[sym] = pd.Series(c[keep], index=pd.to_datetime(ms[keep], unit="ms"))
    return pd.DataFrame(cols).sort_index()


def _stats(net, cad, ann):
    win = net[net.index >= pd.Timestamp(WIN[0])].to_numpy()
    eq = np.cumprod(1 + win); pk = np.maximum.accumulate(eq); dd = float(((eq - pk) / pk).min() * 100)
    sh = float(np.mean(win) / (np.std(win) + 1e-12) * np.sqrt(ann))
    rv = float(np.std(win) * np.sqrt(ann) * 100)
    return {"net": round(float(eq[-1] - 1) * 100, 1), "maxdd": round(dd, 1), "sharpe": round(sh, 2), "real_vol": round(rv, 1)}


def main() -> int:
    out = {}
    for cad in ["1d", "4h"]:
        ann = {"1d": 365, "4h": 365 * 6}[cad]
        df = _series(cad); vw = VOLWIN[cad]
        ret = df.pct_change()
        rv = ret.rolling(vw).std()                      # realized vol per asset
        # target vol levels = the median realized vol scaled (lo=50%, hi=100% of median asset vol)
        med_rv = float(rv.median().median())
        books = {}
        for label, tv in [("BUYHOLD", None), ("VOLTGT_lo", 0.5 * med_rv), ("VOLTGT_hi", 1.0 * med_rv)]:
            if tv is None:
                exp = pd.DataFrame(1.0, index=df.index, columns=df.columns)
            else:
                exp = (tv / (rv.shift(1) + 1e-12)).clip(0, 1).fillna(0.0)
            bookret = (exp * ret).mean(axis=1)           # equal-weight across assets
            books[label] = bookret
            out[(cad, label)] = _stats(bookret, cad, ann)
        print(f"########## {cad} -- vol-targeting vs flat buy-hold (equal-weight book) ##########")
        print(f"   {'strategy':11} {'net%':>8} {'maxDD%':>8} {'Sharpe':>7} {'realVol%':>9}")
        for label in ["BUYHOLD", "VOLTGT_lo", "VOLTGT_hi"]:
            s = out[(cad, label)]
            print(f"   {label:11} {s['net']:>8} {s['maxdd']:>8} {s['sharpe']:>7} {s['real_vol']:>9}")
        bh = out[(cad, "BUYHOLD")]
        print(f"   -> VOLTGT_hi vs BH: net {out[(cad,'VOLTGT_hi')]['net']-bh['net']:+.1f}pp, "
              f"Sharpe {out[(cad,'VOLTGT_hi')]['sharpe']-bh['sharpe']:+.2f}, maxDD {out[(cad,'VOLTGT_hi')]['maxdd']-bh['maxdd']:+.1f}pp\n")

    op = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
    op.mkdir(parents=True, exist_ok=True)
    json.dump({f"{c}|{l}": v for (c, l), v in out.items()}, open(op / "voltarget.json", "w"), indent=1, default=str)
    print(f"[json] {op / 'voltarget.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
