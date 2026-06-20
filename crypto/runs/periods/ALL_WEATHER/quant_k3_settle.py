"""K=3 independent derivations of the decisive claims + push gate+VT to the engine's DD band.

D1: gate-only > engine on wealth  (re-derive via raw held arrays, independent of tcs)
D2: gate+VT cannot match engine's wealth at engine's DD  (dense VT sweep into -13..-19% DD)
D3: n_eff + autocorr of the continuous gate book (is p05 trustworthy?)
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
COST = 0.0024
ENGINE = [("ADX", 200.4, -18.8), ("MACD", 186.2, -15.3),
          ("DONCHIAN", 184.0, -13.3), ("SUPERTREND", 194.4, -17.4)]


def _ms(s):
    return (int(pd.Timestamp(s[0]).value // 10**6), int(pd.Timestamp(s[1]).value // 10**6))


def book(assets, held_fn, lo, hi):
    series = []
    for A in assets:
        ret, ms = A["ret"], A["ms"]
        h = np.asarray(held_fn(A), dtype=float)
        pos = np.zeros(len(ret)); pos[1:] = h[:-1]
        flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
        net = pos * ret - flips * (COST / 2.0)
        m = (ms >= lo) & (ms < hi)
        if m.sum() < 5:
            continue
        series.append(pd.Series(net[m], index=pd.to_datetime(ms[m], unit="ms")))
    if not series:
        return None
    return pd.concat(series, axis=1).fillna(0.0).mean(axis=1).sort_index()


def netpct(b):
    x = b.dropna().to_numpy(); return float(np.prod(1 + x) - 1) * 100.0


def maxdd(b):
    x = b.dropna().to_numpy(); eq = np.cumprod(1 + x); pk = np.maximum.accumulate(eq)
    return float(((eq - pk) / pk).min() * 100.0)


def main():
    lo, hi = _ms(SPAN)
    assets = msb._load_all("1d", SPAN[0], SPAN[1])

    # D1: gate-only via three independent gate impls (rolling-mean, cumsum SMA, pandas .mean)
    def gate_rolling(A, n):
        g = pd.Series(A["c"]).rolling(n, min_periods=1).mean().to_numpy()
        return (A["c"] > g).astype(float)
    def gate_cumsum(A, n):
        c = A["c"]; cs = np.cumsum(np.concatenate([[0.0], c]))
        out = np.empty(len(c))
        for i in range(len(c)):
            j = max(0, i - n + 1)
            out[i] = (cs[i + 1] - cs[j]) / (i - j + 1)
        return (c > out).astype(float)
    g1 = netpct(book(assets, lambda A: gate_rolling(A, 200), lo, hi))
    g2 = netpct(book(assets, lambda A: gate_cumsum(A, 200), lo, hi))
    g3 = netpct(book(assets, lambda A: (A["c"] > _sma(A["c"], 200)).astype(float), lo, hi))
    print("## D1: GATE_200 wealth via 3 independent impls")
    print(f"   rolling={g1:.1f}%  cumsum={g2:.1f}%  _sma={g3:.1f}%  (agree={max(abs(g1-g2),abs(g2-g3))<1.0})")
    print(f"   all > engine-max(200.4%)? {min(g1,g2,g3) > 200.4}")

    # D2: dense VT sweep -- find the MAX wealth gate+VT achievable at each engine cell's DD
    def gate_vt(A, n, vw, tgt):
        c = A["c"]; ret = A["ret"]
        g = (c > _sma(c, n)).astype(float)
        rv = pd.Series(ret).rolling(vw, min_periods=max(5, vw // 2)).std().to_numpy()
        rv = np.where(np.isnan(rv) | (rv <= 0), np.nan, rv)
        w = np.nan_to_num(np.minimum(1.0, tgt / rv), nan=0.0)
        wc = np.zeros_like(w); wc[1:] = w[:-1]
        return g * wc
    grid = []
    for n in (100, 125, 150, 175, 200, 250):
        for vw in (15, 20, 30, 40):
            for tgt in np.arange(0.008, 0.06, 0.002):
                b = book(assets, lambda A, n=n, vw=vw, tgt=tgt: gate_vt(A, n, vw, tgt), lo, hi)
                grid.append((netpct(b), maxdd(b), f"G{n}V{vw}t{tgt:.3f}"))
    print("\n## D2: best gate+VT wealth AT OR BELOW each engine cell's DD (dense sweep, n={})".format(len(grid)))
    for nm, ew, ed in ENGINE:
        cands = [(w, d, t) for (w, d, t) in grid if d >= ed]   # DD at least as shallow as engine
        if cands:
            bw, bd, bt = max(cands, key=lambda x: x[0])
            verdict = "ENGINE WINS" if ew > bw else "gate+VT matches/beats"
            print(f"   {nm:11s} engine(w{ew:.0f},dd{ed:.0f})  best gate+VT@dd>={ed:.0f}: "
                  f"w{bw:.0f},dd{bd:.0f} [{bt}] -> {verdict} (engine wealth advantage {ew-bw:+.0f}pp)")

    # D3: n_eff of the continuous gate book
    b = book(assets, lambda A: (A["c"] > _sma(A["c"], 200)).astype(float), lo, hi)
    x = b.dropna().to_numpy(); n = len(x)
    # lag-1 autocorr of daily net
    r1 = np.corrcoef(x[:-1], x[1:])[0, 1]
    # effective sample size (AR1 approx)
    n_eff = n * (1 - r1) / (1 + r1) if abs(r1) < 1 else n
    print(f"\n## D3: gate book n={n} daily bars, lag1-autocorr={r1:.3f}, n_eff~{n_eff:.0f}")
    print(f"   (a 27-month book has only ~{n_eff:.0f} effective obs -> single-window p05 is FRAGILE; "
          f"this is ONE bull-then-bear path, not a resampled distribution)")


if __name__ == "__main__":
    main()
