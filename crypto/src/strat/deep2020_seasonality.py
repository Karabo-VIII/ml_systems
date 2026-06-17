"""src/strat/deep2020_seasonality.py -- BLOCK M: is the 2020 drift calendar-uniform, or is there structure?

We treated the drift as constant. This checks for CALENDAR structure: mean return by DAY-OF-WEEK (1d bars)
and by HOUR-OF-DAY (1h bars), pooled across u10, with a t-stat per bucket (is any day/hour significantly
different from zero / from the overall mean?). A significant day/hour pattern = a NEW exploitable structure
orthogonal to everything tested. RWYB: python -m strat.deep2020_seasonality. No emoji (cp1252).
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

WIN = ("2020-07-01", "2021-01-01")
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _pooled_returns(cad):
    w0 = pd.Timestamp(WIN[0]).value // 10**6; w1 = pd.Timestamp(WIN[1]).value // 10**6
    frames = []
    for sym in SYMS:
        try:
            o, h, l, c, ms = _panel(sym, cad)
        except Exception:
            continue
        keep = (ms >= w0) & (ms < w1)
        c2 = c[keep]; dt = pd.to_datetime(ms[keep], unit="ms")
        if len(c2) < 20:
            continue
        r = np.zeros(len(c2)); r[1:] = c2[1:] / c2[:-1] - 1.0
        frames.append(pd.Series(r[1:], index=dt[1:]))
    return pd.concat(frames) if frames else pd.Series(dtype=float)


def _bucket(s, keyfn, labels):
    rows = {}
    g = s.groupby(s.index.map(keyfn))
    for k, grp in g:
        x = grp.to_numpy()
        rows[k] = {"mean_bp": round(float(np.mean(x)) * 1e4, 1), "t": round(float(np.mean(x) / (np.std(x) / np.sqrt(len(x)) + 1e-12)), 2), "n": len(x)}
    return rows


def main() -> int:
    out = {}
    # DAY OF WEEK (1d)
    s = _pooled_returns("1d")
    dow = _bucket(s, lambda t: t.dayofweek, DOW)
    overall = float(s.mean()) * 1e4
    print(f"BLOCK M seasonality -- 2020 H2; overall mean {overall:.1f} bp/day\n")
    print("## DAY-OF-WEEK mean return (1d, pooled u10):")
    print(f"   {'day':5} {'mean_bp':>8} {'t':>6} {'n':>5}")
    for k in sorted(dow):
        d = dow[k]; sig = "*" if abs(d["t"]) > 2 else ""
        print(f"   {DOW[k]:5} {d['mean_bp']:>8} {d['t']:>6}{sig} {d['n']:>5}")
    out["dow"] = {DOW[k]: v for k, v in dow.items()}

    # HOUR OF DAY (1h)
    s1 = _pooled_returns("1h")
    hod = _bucket(s1, lambda t: t.hour, list(range(24)))
    print(f"\n## HOUR-OF-DAY mean return (1h, pooled u10; UTC); overall {float(s1.mean())*1e4:.1f} bp/h:")
    print(f"   {'hour':5} {'mean_bp':>8} {'t':>6}")
    sigh = [(k, v) for k, v in hod.items() if abs(v["t"]) > 2]
    for k in sorted(hod):
        v = hod[k]; sig = "*" if abs(v["t"]) > 2 else ""
        print(f"   {k:>4}h {v['mean_bp']:>8} {v['t']:>6}{sig}")
    out["hod"] = {str(k): v for k, v in hod.items()}

    nsig_d = sum(1 for v in dow.values() if abs(v["t"]) > 2)
    nsig_h = sum(1 for v in hod.values() if abs(v["t"]) > 2)
    print(f"\n## VERDICT: {nsig_d}/7 days and {nsig_h}/24 hours significant (|t|>2). "
          f"{'CALENDAR STRUCTURE present' if (nsig_d + nsig_h) >= 2 else 'drift is ~calendar-UNIFORM (no exploitable seasonality)'}")
    op = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
    op.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(op / "seasonality.json", "w"), indent=1, default=str)
    print(f"[json] {op / 'seasonality.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
