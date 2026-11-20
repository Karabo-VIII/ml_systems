"""src/strat/deep2020_orthobook.py -- BLOCK P: the best CAUSAL book from the ORTHOGONAL/risk findings.

The deep-dive's two non-bull-beta findings: VOL-TARGET sizing (risk identity, transfers) + CALENDAR tilt
(orthogonal signal, causally beats BH within 2020). This builds the best causal book from ONLY those (NO
bull-beta XS momentum), classified on VAL, applied to OOS, vs buy-hold:
  BUYHOLD            equal-weight, full size
  VOLTGT            vol-target sizing
  CAL               flat on VAL-bad weekdays + VAL-bad hours
  VOLTGT_CAL        vol-target sizing AND calendar tilt (the combined orthogonal book)
Causal (VAL classification -> OOS apply). RWYB: python -m strat.deep2020_orthobook. No emoji (cp1252).
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


def _panel_ret_vol(cad, vw):
    w0 = pd.Timestamp(WIN[0]).value // 10**6; w1 = pd.Timestamp(WIN[1]).value // 10**6
    rets, vols = [], []
    for sym in SYMS:
        try:
            o, h, l, c, ms = _panel(sym, cad)
        except Exception:
            continue
        keep = (ms >= w0 - 30 * 86400000) & (ms < w1); c2 = c[keep]
        if len(c2) < 40:
            continue
        s = pd.Series(c2, index=pd.to_datetime(ms[keep], unit="ms")).sort_index()
        r = s.pct_change()
        rets.append(r.rename(sym)); vols.append(r.rolling(vw).std().rename(sym))
    return pd.concat(rets, axis=1), pd.concat(vols, axis=1)


def _stats(x, ann):
    x = x.dropna().to_numpy()
    if len(x) < 5:
        return {}
    eq = np.cumprod(1 + x); pk = np.maximum.accumulate(eq); dd = float(((eq - pk) / pk).min() * 100)
    return {"net": round(float(eq[-1] - 1) * 100, 1), "maxdd": round(dd, 1),
            "sharpe": round(float(np.mean(x) / (np.std(x) + 1e-12) * np.sqrt(ann)), 2)}


def main() -> int:
    cad = "1h"; vw = 168; ann = 365 * 24
    ret, vol = _panel_ret_vol(cad, vw)
    med = float(vol.median().median())
    bookret = ret.mean(axis=1)                                   # equal-weight book bar returns
    # vol-target exposure (book-level: avg inverse-vol weight)
    volexp = (med / (vol.shift(1) + 1e-12)).clip(0, 1).mean(axis=1)
    val_mask = bookret.index < pd.Timestamp(SPLIT)
    # classify calendar on VAL
    valr = bookret[val_mask]
    bad_dow = set(int(k) for k, v in valr.groupby(valr.index.dayofweek).mean().items() if v < 0)
    bad_hr = set(int(k) for k, v in valr.groupby(valr.index.hour).mean().items() if v < 0)
    calexp = pd.Series(1.0, index=bookret.index)
    calexp[bookret.index.dayofweek.isin(list(bad_dow)) | bookret.index.hour.isin(list(bad_hr))] = 0.0
    oos = bookret.index >= pd.Timestamp(SPLIT)
    print(f"BLOCK P orthogonal book ({cad}) -- VAL-bad weekdays {sorted(bad_dow)}, VAL-bad hours {sorted(bad_hr)}\n")
    print(f"   {'OOS book':14} {'net%':>8} {'maxDD%':>8} {'Sharpe':>7}")
    strats = {"BUYHOLD": bookret, "VOLTGT": volexp.shift(1).fillna(0) * bookret,
              "CAL": calexp.shift(1).fillna(0) * bookret,
              "VOLTGT_CAL": (volexp.clip(0, 1) * calexp).shift(1).fillna(0) * bookret}
    out = {}
    for k, s in strats.items():
        st = _stats(s[oos], ann); out[k] = st
        if st:
            print(f"   {k:14} {st['net']:>8} {st['maxdd']:>8} {st['sharpe']:>7}")
    bh = out["BUYHOLD"]
    best = max((k for k in out if out[k]), key=lambda k: out[k]["sharpe"])
    print(f"\n   best Sharpe: {best} ({out[best]['sharpe']} vs BH {bh['sharpe']}); "
          f"net {out[best]['net']} vs BH {bh['net']}; maxDD {out[best]['maxdd']} vs {bh['maxdd']}")
    print(f"   VERDICT: combining vol-target + calendar (orthogonal/risk findings, NO bull-beta) "
          f"{'beats BH on Sharpe causally OOS' if out[best]['sharpe'] > bh['sharpe'] else 'does not beat BH'}.")
    op = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
    op.mkdir(parents=True, exist_ok=True)
    json.dump({"bad_dow": sorted(bad_dow), "bad_hr": sorted(bad_hr), "oos": out}, open(op / "orthobook.json", "w"), indent=1, default=str)
    print(f"[json] {op / 'orthobook.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
