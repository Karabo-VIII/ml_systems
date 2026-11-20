"""src/strat/deep2020_xsection.py -- BLOCK H: CROSS-SECTIONAL structure -- which assets, not when.

The time-series axis (Blocks A-G) says you cannot time WHEN to be long (no timing skill, drift random walk).
The orthogonal question: can you pick WHICH assets? Each rebalance, rank u10 by trailing-N return and:
  XS_MOM   long the top-3 (recent winners keep winning?)
  XS_REV   long the bottom-3 (recent losers revert?)
  EW       long all (equal-weight = the buy-hold book benchmark)
Compare net% / maxDD / Sharpe over 2020 H2 (+ OOS). Also the cross-sectional DISPERSION (mean daily spread
of asset returns = how much there is to exploit by selection). 1d + 4h, weekly-ish lookback, daily rebalance.
RWYB: python -m strat.deep2020_xsection. No emoji (cp1252).
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
LOOKBACK = {"1d": 7, "4h": 42}     # ~1 week
TOPK = 3


def _panel_df(cad):
    """aligned close DataFrame [time x sym] on the WIN window (+ lookback warmup)."""
    w0 = pd.Timestamp(WIN[0]).value // 10**6; w1 = pd.Timestamp(WIN[1]).value // 10**6
    cols = {}
    for sym in SYMS:
        try:
            o, h, l, c, ms = _panel(sym, cad)
        except Exception:
            continue
        keep = (ms >= w0 - 30 * 86400000) & (ms < w1)
        cols[sym] = pd.Series(c[keep], index=pd.to_datetime(ms[keep], unit="ms"))
    df = pd.DataFrame(cols).sort_index()
    return df[~df.index.duplicated(keep="last")]


def _book(df, mode, lb):
    ret = df.pct_change().fillna(0.0)
    trail = df.pct_change(lb)
    out = []
    idx = df.index
    for i in range(lb + 1, len(idx)):
        r = trail.iloc[i - 1].dropna()                 # causal: rank on PRIOR bar's trailing return
        if len(r) < TOPK + 1:
            out.append(0.0); continue
        if mode == "EW":
            sel = r.index
        elif mode == "XS_MOM":
            sel = r.nlargest(TOPK).index
        else:                                          # XS_REV
            sel = r.nsmallest(TOPK).index
        out.append(float(ret.iloc[i][sel].mean()))
    return pd.Series(out, index=idx[lb + 1:])


def _stats(s, cad):
    win = s[s.index >= pd.Timestamp(WIN[0])]
    net = float(np.prod(1 + win.to_numpy()) - 1) * 100
    eq = np.cumprod(1 + win.to_numpy()); pk = np.maximum.accumulate(eq); dd = float(((eq - pk) / pk).min() * 100)
    sh = float(win.mean() / (win.std() + 1e-12) * np.sqrt({"1d": 365, "4h": 365 * 6}.get(cad, 365)))
    oos = win[win.index >= pd.Timestamp(SPLIT)]
    oosn = float(np.prod(1 + oos.to_numpy()) - 1) * 100 if len(oos) else float("nan")
    return {"net": round(net, 1), "maxdd": round(dd, 1), "sharpe": round(sh, 2), "oos_net": round(oosn, 1)}


def main() -> int:
    out = {}
    for cad in ["1d", "4h"]:
        df = _panel_df(cad)
        if df.shape[1] < 5:
            continue
        lb = LOOKBACK[cad]
        disp = float(df.pct_change().std(axis=1).mean()) * 100      # mean cross-sectional dispersion per bar
        print(f"########## {cad} -- cross-sectional selection (top/bottom-{TOPK} by trailing-{lb}); XS dispersion {disp:.2f}%/bar ##########")
        print(f"   {'strategy':9} {'net%':>8} {'maxDD%':>8} {'Sharpe':>7} {'OOSnet%':>8}")
        for mode in ["EW", "XS_MOM", "XS_REV"]:
            s = _book(df, mode, lb)
            st = _stats(s, cad); out[(cad, mode)] = st
            print(f"   {mode:9} {st['net']:>8} {st['maxdd']:>8} {st['sharpe']:>7} {st['oos_net']:>8}")
        ew = out[(cad, "EW")]["net"]
        print(f"   -> XS_MOM vs EW: {out[(cad,'XS_MOM')]['net']-ew:+.1f}pp   XS_REV vs EW: {out[(cad,'XS_REV')]['net']-ew:+.1f}pp")
        print()

    op = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
    op.mkdir(parents=True, exist_ok=True)
    json.dump({f"{c}|{m}": v for (c, m), v in out.items()}, open(op / "xsection.json", "w"), indent=1, default=str)
    print(f"[json] {op / 'xsection.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
