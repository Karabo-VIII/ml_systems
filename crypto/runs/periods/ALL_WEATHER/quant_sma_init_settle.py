"""Quant referee SETTLEMENT: does the SMA200 min_periods=1 init artifact reverse the gate finding?

Audit 0/1 reproduce GATE_200 = +937.5% with the shipped _sma (min_periods=1) and call it VERIFIED.
Audit 2 says that +937.5% is a partial-window-SMA artifact on late listers (SOL/DOGE/AVAX) and that a
strict SMA200 (NaN until 200 bars -> cash) drops the gate to +324%, BELOW raw BH (+549%) -> reverses.

This script re-derives ALL of it from the loader, no reported number trusted:
  (1) per-asset pre-window bar count (how many bars before 2020-10-01) -> who is a late lister
  (2) GATE_200 wealth under min_periods=1 (shipped) vs strict min_periods=200 (NaN->cash)
  (3) per-asset wealth contribution under both, to localize the difference
  (4) RAW_BH and engine ordering under both regimes
No emoji (cp1252).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4] / "src"
sys.path.insert(0, str(ROOT))

import strat.ma_strat_builder as msb

COST = 0.0024
SPAN = ("2020-10-01", "2023-01-01")
GATE_N = 200


def _ms(span):
    return (int(pd.Timestamp(span[0]).value // 10**6), int(pd.Timestamp(span[1]).value // 10**6))


def _sma_mp1(c, n):
    return pd.Series(c).rolling(n, min_periods=1).mean().to_numpy()


def _sma_strict(c, n):
    return pd.Series(c).rolling(n, min_periods=n).mean().to_numpy()


def _net_window_held(A, held, lo_ms, hi_ms, cost=COST):
    ret, ms = A["ret"], A["ms"]
    pos = np.zeros(len(ret)); pos[1:] = np.asarray(held, dtype=float)[:-1]
    flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
    net = pos * ret - flips * (cost / 2.0)
    mask = (ms >= lo_ms) & (ms < hi_ms)
    if mask.sum() < 5:
        return None
    return pd.Series(net[mask], index=pd.to_datetime(ms[mask], unit="ms"))


def _ew(series_list):
    sl = [s for s in series_list if s is not None]
    if not sl:
        return None
    return pd.concat(sl, axis=1).fillna(0.0).mean(axis=1).sort_index()


def _netpct(book):
    if book is None: return None
    x = book.dropna().to_numpy()
    if len(x) < 2: return 0.0
    return round(float(np.prod(1 + x) - 1) * 100.0, 1)


def _maxdd(book):
    if book is None: return None
    x = book.dropna().to_numpy()
    if len(x) < 2: return 0.0
    eq = np.cumprod(1 + x); pk = np.maximum.accumulate(eq)
    return round(float(((eq - pk) / pk).min() * 100.0), 1)


def _solo_wealth(A, held, lo_ms, hi_ms):
    """compound wealth of a single asset's gated book (not EW-diluted)."""
    s = _net_window_held(A, held, lo_ms, hi_ms)
    if s is None: return None
    x = s.dropna().to_numpy()
    return round(float(np.prod(1 + x) - 1) * 100.0, 0)


def main():
    lo, hi = _ms(SPAN)
    s_ms = int(pd.Timestamp(SPAN[0]).value // 10**6)
    assets = msb._load_all("1d", SPAN[0], SPAN[1])
    print(f"# loaded {len(assets)} assets span {SPAN}")
    print(f"# WARMUP={msb.WARMUP}, GATE_N={GATE_N}")

    # (1) per-asset pre-window bar counts
    print("\n## per-asset pre-window (before 2020-10-01) daily-bar counts")
    print(f"  {'sym':8s}{'pre_bars':>10s}{'late_lister(<200)':>20s}")
    pre = {}
    for A in assets:
        n_pre = int((A["ms"] < s_ms).sum())
        pre[A["sym"]] = n_pre
        flag = "YES" if n_pre < GATE_N else ""
        print(f"  {A['sym']:8s}{n_pre:>10d}{flag:>20s}")

    # (2) book-level wealth/DD under both init regimes
    raw = _ew([_net_window_held(A, np.ones(len(A['c']), dtype=np.int8), lo, hi) for A in assets])
    g_mp1 = _ew([_net_window_held(A, (A['c'] > _sma_mp1(A['c'], GATE_N)).astype(np.int8), lo, hi) for A in assets])
    g_str = _ew([_net_window_held(A, (A['c'] > _sma_strict(A['c'], GATE_N)).astype(np.int8), lo, hi) for A in assets])

    print("\n## BOOK-LEVEL wealth + maxDD, EW-u10, taker 0.0024")
    print(f"  {'book':28s}{'wealth%':>10s}{'maxDD%':>9s}")
    print(f"  {'RAW_BH':28s}{_netpct(raw):>10.1f}{_maxdd(raw):>9.1f}")
    print(f"  {'GATE_200 (min_periods=1)':28s}{_netpct(g_mp1):>10.1f}{_maxdd(g_mp1):>9.1f}")
    print(f"  {'GATE_200 (strict mp=200)':28s}{_netpct(g_str):>10.1f}{_maxdd(g_str):>9.1f}")

    # (3) per-asset SOLO wealth (gate applied to that asset alone) under both regimes
    print("\n## per-asset SOLO gated wealth (not EW-diluted) -- localize the gap")
    print(f"  {'sym':8s}{'pre_bars':>9s}{'mp1%':>10s}{'strict%':>10s}{'delta':>10s}")
    for A in assets:
        w1 = _solo_wealth(A, (A['c'] > _sma_mp1(A['c'], GATE_N)).astype(np.int8), lo, hi)
        ws = _solo_wealth(A, (A['c'] > _sma_strict(A['c'], GATE_N)).astype(np.int8), lo, hi)
        d = None if (w1 is None or ws is None) else round(w1 - ws, 0)
        print(f"  {A['sym']:8s}{pre[A['sym']]:>9d}"
              f"{('' if w1 is None else f'{w1:>10.0f}')}"
              f"{('' if ws is None else f'{ws:>10.0f}')}"
              f"{('' if d is None else f'{d:>10.0f}')}")

    # (4) PIT universe: drop assets with < GATE_N pre-window bars, recompute book ordering
    pit = [A for A in assets if pre[A["sym"]] >= GATE_N]
    print(f"\n## POINT-IN-TIME universe (pre_bars >= {GATE_N}): {[A['sym'] for A in pit]} (n={len(pit)})")
    if pit:
        raw_p = _ew([_net_window_held(A, np.ones(len(A['c']), dtype=np.int8), lo, hi) for A in pit])
        g1_p = _ew([_net_window_held(A, (A['c'] > _sma_mp1(A['c'], GATE_N)).astype(np.int8), lo, hi) for A in pit])
        gs_p = _ew([_net_window_held(A, (A['c'] > _sma_strict(A['c'], GATE_N)).astype(np.int8), lo, hi) for A in pit])
        print(f"  {'RAW_BH(PIT)':28s}{_netpct(raw_p):>10.1f}{_maxdd(raw_p):>9.1f}")
        print(f"  {'GATE_200 mp1(PIT)':28s}{_netpct(g1_p):>10.1f}{_maxdd(g1_p):>9.1f}")
        print(f"  {'GATE_200 strict(PIT)':28s}{_netpct(gs_p):>10.1f}{_maxdd(gs_p):>9.1f}")


if __name__ == "__main__":
    main()
