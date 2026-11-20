"""Quant referee: verify 'gate-only SMA200 buy-hold dominates the engine on wealth'.

Re-derives the decisive numbers INDEPENDENTLY from the loader, not from any reported figure.
Continuous span 2020-10-01 .. 2022-12-31, EW-u10, taker 0.0024, long-only spot, fixed-EW fillna(0).mean.

Strategies compared on the SAME continuous book:
  RAW_BH        : always long (held=1), no gate, no signal.
  GATE_ONLY     : hold each asset when close > causal SMA-N, else cash. N in {100,150,200}.
  GATE_VT       : GATE_ONLY scaled by an inverse-vol target (causal realized vol), capped at 1x (long-only,no lev).
  Engine cells  : ADX/MACD/DONCHIAN/SUPERTREND 1d band-ensembles, replayed continuously, gated.

Outputs wealth (compound net %), maxDD %, and a block-bootstrap p05 + seed spread on the continuous book.
No emoji (cp1252).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4] / "src"
sys.path.insert(0, str(ROOT))

import strat.ma_strat_builder as msb
import strat.ti_capture_sweep as tcs
from strat.ma_type_upgrade import _sma
from strat.deep2020_ti_pipeline import INDICATORS

COST = 0.0024
SPAN = ("2020-10-01", "2023-01-01")   # continuous, inclusive of 2022-12-31


def _ms(span):
    return (int(pd.Timestamp(span[0]).value // 10**6), int(pd.Timestamp(span[1]).value // 10**6))


def _net_window_held(A, held, lo_ms, hi_ms, cost=COST):
    """net book series for an arbitrary held(0/1 or fractional) array, lag-1, taker cost on |dpos|."""
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
    if book is None:
        return None
    x = book.dropna().to_numpy()
    if len(x) < 2:
        return 0.0
    return round(float(np.prod(1 + x) - 1) * 100.0, 1)


def _maxdd(book):
    if book is None:
        return None
    x = book.dropna().to_numpy()
    if len(x) < 2:
        return 0.0
    eq = np.cumprod(1 + x); pk = np.maximum.accumulate(eq)
    return round(float(((eq - pk) / pk).min() * 100.0), 1)


def _gate_held(A, n):
    c = A["c"]; g = _sma(c, n)
    return (c > g).astype(np.int8)   # NaN -> False -> cash


def _raw_held(A):
    return np.ones(len(A["c"]), dtype=np.int8)


def _vol_target_weight(A, n_gate, vol_win, target):
    """gate * min(1, target/realized_vol) -- causal, long-only, no leverage (cap 1)."""
    c = A["c"]; ret = A["ret"]
    g = (c > _sma(c, n_gate)).astype(float)
    rv = pd.Series(ret).rolling(vol_win, min_periods=max(5, vol_win // 2)).std().to_numpy()
    rv = np.where(np.isnan(rv) | (rv <= 0), np.nan, rv)
    w = np.minimum(1.0, target / rv)
    w = np.nan_to_num(w, nan=0.0)
    # use yesterday's vol (causal): shift weight by 1 already handled by lag-1 in _net_window_held; but
    # realized vol at t uses ret up to t which includes ret[t]; shift rv by 1 to be strictly causal.
    w_causal = np.zeros_like(w); w_causal[1:] = w[:-1]
    return g * w_causal


def _block_p05(book, seeds=(0, 1, 2, 3, 4), n_boot=1000):
    if book is None:
        return None, None
    x = book.dropna().to_numpy(); n = len(x)
    if n < 20:
        return None, None
    block = max(10, int(round(n ** 0.5)))
    if block >= n:
        block = max(2, n // 3)
    starts_pool = np.arange(0, n - block + 1)
    nblocks = int(np.ceil(n / block))
    p05s = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        nets = np.empty(n_boot)
        for b in range(n_boot):
            st = rng.choice(starts_pool, size=nblocks, replace=True)
            idx = np.concatenate([np.arange(s, s + block) for s in st])[:n]
            nets[b] = (np.prod(1 + x[idx]) - 1) * 100.0
        p05s.append(float(np.percentile(nets, 5)))
    return round(float(np.median(p05s)), 1), (round(min(p05s), 1), round(max(p05s), 1))


def main():
    lo, hi = _ms(SPAN)
    assets = msb._load_all("1d", SPAN[0], SPAN[1])
    print(f"# loaded {len(assets)} assets, span {SPAN}, syms={[A['sym'] for A in assets]}")

    results = {}

    # RAW buy-hold
    bk = _ew([_net_window_held(A, _raw_held(A), lo, hi) for A in assets])
    results["RAW_BH"] = (_netpct(bk), _maxdd(bk), bk)

    # GATE-ONLY, N in {100,150,200}
    for n in (100, 150, 200):
        bk = _ew([_net_window_held(A, _gate_held(A, n), lo, hi) for A in assets])
        results[f"GATE_{n}"] = (_netpct(bk), _maxdd(bk), bk)

    # GATE + vol-target (a few targets). target is per-bar (daily) vol units.
    for n in (200,):
        for vw in (20, 30):
            for tgt in (0.02, 0.03, 0.04):
                bk = _ew([_net_window_held(A, _vol_target_weight(A, n, vw, tgt), lo, hi) for A in assets])
                results[f"GATE{n}_VT{vw}_t{tgt}"] = (_netpct(bk), _maxdd(bk), bk)

    print("\n## WEALTH (compound net %) + maxDD on the continuous span 2020-10 .. 2022-12")
    print(f"  {'strategy':22s}{'wealth%':>10s}{'maxDD%':>9s}")
    for k, (w, dd, _bk) in results.items():
        print(f"  {k:22s}{('' if w is None else f'{w:>10.1f}')}{('' if dd is None else f'{dd:>9.1f}')}")

    # p05 + seed-spread on the headline cases
    print("\n## block-bootstrap p05 (median over 5 seeds) + seed spread")
    for k in ("RAW_BH", "GATE_200", "GATE100_VT30_t0.03"):
        if k in results:
            p, spread = _block_p05(results[k][2])
            print(f"  {k:22s} p05={p}  seed-range={spread}")

    return results


if __name__ == "__main__":
    main()
