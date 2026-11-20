"""Settle: (1) cost sensitivity of the gate-vs-engine gap, (2) EW vs cap-weight,
(3) the admissible-strategy FRONTIER (wealth vs maxDD) -- is the engine's low-DD niche real or
can a gate+vol-target dominate it?

Builds many (gate-N, vol-target) points + the 4 engine cells + raw BH on the SAME continuous span,
under taker AND maker, EW AND cap-weight. Prints the Pareto frontier (max wealth at each DD bucket).
No emoji (cp1252).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4] / "src"
sys.path.insert(0, str(ROOT))

import strat.ma_strat_builder as msb
from strat.ma_type_upgrade import _sma

SPAN = ("2020-10-01", "2023-01-01")
TAKER, MAKER = 0.0024, 0.0006


def _ms(span):
    return (int(pd.Timestamp(span[0]).value // 10**6), int(pd.Timestamp(span[1]).value // 10**6))


def _book(assets, held_fn, lo, hi, cost, weights=None):
    """EW (weights=None) or static cap-weight book. held_fn(A)->float array."""
    series = []
    wlist = []
    for A in assets:
        ret, ms = A["ret"], A["ms"]
        held = np.asarray(held_fn(A), dtype=float)
        pos = np.zeros(len(ret)); pos[1:] = held[:-1]
        flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
        net = pos * ret - flips * (cost / 2.0)
        mask = (ms >= lo) & (ms < hi)
        if mask.sum() < 5:
            continue
        s = pd.Series(net[mask], index=pd.to_datetime(ms[mask], unit="ms"))
        series.append(s)
        wlist.append(weights.get(A["sym"], 0.0) if weights else 1.0)
    if not series:
        return None
    df = pd.concat(series, axis=1).fillna(0.0)
    w = np.array(wlist, dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.ones(len(w)) / len(w)
    return (df * w).sum(axis=1).sort_index()


def _netpct(book):
    if book is None:
        return None
    x = book.dropna().to_numpy()
    return round(float(np.prod(1 + x) - 1) * 100.0, 1) if len(x) >= 2 else 0.0


def _maxdd(book):
    if book is None:
        return None
    x = book.dropna().to_numpy()
    if len(x) < 2:
        return 0.0
    eq = np.cumprod(1 + x); pk = np.maximum.accumulate(eq)
    return round(float(((eq - pk) / pk).min() * 100.0), 1)


def gate_held(n):
    return lambda A: (A["c"] > _sma(A["c"], n)).astype(float)


def raw_held(A):
    return np.ones(len(A["c"]), dtype=float)


def gate_vt_held(n, vw, tgt):
    def f(A):
        c = A["c"]; ret = A["ret"]
        g = (c > _sma(c, n)).astype(float)
        rv = pd.Series(ret).rolling(vw, min_periods=max(5, vw // 2)).std().to_numpy()
        rv = np.where(np.isnan(rv) | (rv <= 0), np.nan, rv)
        w = np.nan_to_num(np.minimum(1.0, tgt / rv), nan=0.0)
        wc = np.zeros_like(w); wc[1:] = w[:-1]
        return g * wc
    return f


# approximate static cap weights (rough 2020-10 mcap rank proxy; BTC/ETH dominant)
CAPW = {"BTCUSDT": 0.45, "ETHUSDT": 0.20, "BNBUSDT": 0.08, "XRPUSDT": 0.06, "ADAUSDT": 0.05,
        "SOLUSDT": 0.04, "DOGEUSDT": 0.04, "LINKUSDT": 0.03, "LTCUSDT": 0.03, "AVAXUSDT": 0.02}


def main():
    lo, hi = _ms(SPAN)
    assets = msb._load_all("1d", SPAN[0], SPAN[1])

    # 1) COST SENSITIVITY of the gap (gate-only N=200 vs raw)
    print("## (1) COST sensitivity -- GATE_200 vs RAW_BH, EW")
    for cost, name in ((TAKER, "taker"), (MAKER, "maker"), (0.0, "zero")):
        g = _netpct(_book(assets, gate_held(200), lo, hi, cost))
        r = _netpct(_book(assets, raw_held, lo, hi, cost))
        print(f"  cost={name:6s} GATE_200={g:>8.1f}%  RAW_BH={r:>8.1f}%  ratio={g/r if r else float('nan'):.2f}x")

    # 2) EW vs CAP-WEIGHT
    print("\n## (2) EW vs CAP-WEIGHT (taker)")
    for label, held in (("RAW_BH", raw_held), ("GATE_200", gate_held(200))):
        ew = _netpct(_book(assets, held, lo, hi, TAKER))
        cw = _netpct(_book(assets, held, lo, hi, TAKER, weights=CAPW))
        ewdd = _maxdd(_book(assets, held, lo, hi, TAKER))
        cwdd = _maxdd(_book(assets, held, lo, hi, TAKER, weights=CAPW))
        print(f"  {label:10s} EW={ew:>8.1f}%(dd{ewdd:>6.1f})  CAP={cw:>8.1f}%(dd{cwdd:>6.1f})")

    # 3) FRONTIER: wealth vs maxDD across gates and gate+VT, vs engine reference cells
    print("\n## (3) admissible-strategy FRONTIER (wealth vs maxDD), EW taker")
    pts = []
    pts.append(("RAW_BH", _netpct(_book(assets, raw_held, lo, hi, TAKER)),
                _maxdd(_book(assets, raw_held, lo, hi, TAKER))))
    for n in (75, 100, 125, 150, 175, 200, 250):
        bk = _book(assets, gate_held(n), lo, hi, TAKER)
        pts.append((f"GATE_{n}", _netpct(bk), _maxdd(bk)))
    for n in (150, 200):
        for vw in (20, 30):
            for tgt in (0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05):
                bk = _book(assets, gate_vt_held(n, vw, tgt), lo, hi, TAKER)
                pts.append((f"G{n}_VT{vw}_{tgt}", _netpct(bk), _maxdd(bk)))
    ENGINE = [("ADX", 200.4, -18.8), ("MACD", 186.2, -15.3),
              ("DONCHIAN", 184.0, -13.3), ("SUPERTREND", 194.4, -17.4)]
    for nm, w, d in ENGINE:
        pts.append((f"ENGINE_{nm}", w, d))

    # Pareto: a point is dominated if another has >= wealth AND >= (less negative) maxDD
    def dominated(p, others):
        _, w, d = p
        for _, w2, d2 in others:
            if (w2 >= w and d2 >= d) and (w2 > w or d2 > d):
                return True
        return False
    frontier = [p for p in pts if not dominated(p, pts)]
    frontier.sort(key=lambda p: p[2])   # by maxDD ascending (worst->best)
    print("  PARETO FRONTIER (non-dominated; sorted worst->best DD):")
    print(f"    {'strategy':18s}{'wealth%':>10s}{'maxDD%':>9s}")
    for nm, w, d in frontier:
        tag = "  <-- ENGINE" if nm.startswith("ENGINE") else ""
        print(f"    {nm:18s}{w:>10.1f}{d:>9.1f}{tag}")

    # is any engine cell ON the frontier? and at what DD does a gate/VT match each engine cell's DD?
    print("\n  Engine cells vs best gate/VT at <= same DD:")
    for nm, ew, ed in ENGINE:
        better = [(n2, w2, d2) for n2, w2, d2 in pts
                  if not n2.startswith("ENGINE") and d2 >= ed]   # at least as low DD
        if better:
            bw = max(better, key=lambda x: x[1])
            verdict = "ENGINE WINS" if ew > bw[1] else f"DOMINATED by {bw[0]}"
            print(f"    {nm:12s}(w{ew:.0f},dd{ed:.0f})  best non-engine at dd>={ed:.0f}: "
                  f"{bw[0]}(w{bw[1]:.0f},dd{bw[2]:.0f})  -> {verdict}")


if __name__ == "__main__":
    main()
