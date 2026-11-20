"""src/strat/deep2020_osc.py -- BATCH 2: the ORTHOGONAL mean-reversion OSCILLATOR family (RSI/Stoch/BB%b/CCI).

User /orc 2026-06-13 "try another family of TIs?": YES but elevated -- a TREND family (MACD/Supertrend) would
be ~0.85 correlated to the MA book (eff-N~1.2, no diversification, per the family finding). The informative
batch is a MECHANISTICALLY ORTHOGONAL family: mean-reversion OSCILLATORS (buy oversold). Parallel to the MA
batch: each oscillator type x a param grid, long-only MR rules, OOS-graded, clustered with top-K, PLUS the
decisive metric -- CORRELATION of the oscillator book to the MA trend book (does it DIVERSIFY?) + does a
TREND+OSC combined book beat trend alone (Sharpe/maxDD).

OSCILLATORS (long-only MR state machine: enter when val<lo (oversold), exit when val>hi (reverted)):
  RSI(n)        lo/hi on RSI            n in {7,14,21}  lo {25,30,35} hi {55,60}
  STOCH(n)      lo/hi on %K             n in {14,21}    lo {15,20}    hi {55,60}
  BBPCT(n,k)    lo/hi on %b (0..1)      n in {14,20}    lo {0.0,0.1}  hi {0.5}  k=2
  CCI(n)        lo/hi on CCI            n in {14,20}    lo {-150,-100} hi {0}
2020 OOS, u10, min_hold(6)+maker. RWYB: python -m strat.deep2020_osc --cadences <tf>. No emoji (cp1252).
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

import strat.portfolio_replay as PR
from strat.portfolio_replay import MAKER_RT
from strat.replay_distinct_grid import distinct_specs
from strat.structural_fixes import min_hold
from strat.ma_type_upgrade import _MA, _nums
from strat.ma_2020_breakdown import _panel

WIN = ("2020-07-01", "2021-01-01"); SPLIT = "2020-10-01"; WARMUP = 400
CADENCES = ["1d", "4h", "1h"]
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]


# ---- oscillator value series (causal) ----
def _rsi(c, n):
    d = np.diff(c, prepend=c[0]); up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    au = pd.Series(up).rolling(n, min_periods=1).mean().to_numpy(); ad = pd.Series(dn).rolling(n, min_periods=1).mean().to_numpy()
    rs = au / (ad + 1e-12); return 100 - 100 / (1 + rs)


def _stoch(c, h, l, n):
    lo = pd.Series(l).rolling(n, min_periods=1).min().to_numpy(); hi = pd.Series(h).rolling(n, min_periods=1).max().to_numpy()
    return 100 * (c - lo) / (hi - lo + 1e-12)


def _bbpct(c, n, k=2.0):
    m = pd.Series(c).rolling(n, min_periods=1).mean().to_numpy(); sd = pd.Series(c).rolling(n, min_periods=1).std().to_numpy()
    lower = m - k * sd; upper = m + k * sd; return (c - lower) / (upper - lower + 1e-12)


def _cci(c, h, l, n):
    tp = (h + l + c) / 3.0; sma = pd.Series(tp).rolling(n, min_periods=1).mean().to_numpy()
    mad = pd.Series(np.abs(tp - sma)).rolling(n, min_periods=1).mean().to_numpy()
    return (tp - sma) / (0.015 * mad + 1e-12)


def _mr_held(val, lo, hi):
    """long when val drops below lo (oversold); hold until val rises above hi (reverted)."""
    held = np.zeros(len(val), np.int8); cur = 0
    for i in range(len(val)):
        if np.isnan(val[i]):
            cur = 0
        elif cur == 0 and val[i] < lo:
            cur = 1
        elif cur == 1 and val[i] > hi:
            cur = 0
        held[i] = cur
    return held


def _grid():
    g = []
    for n in (7, 14, 21):
        for lo in (25, 30, 35):
            g.append(("RSI", n, lo, 58))
    for n in (14, 21):
        for lo in (15, 20):
            g.append(("STOCH", n, lo, 58))
    for n in (14, 20):
        for lo in (0.0, 0.1):
            g.append(("BBPCT", n, lo, 0.5))
    for n in (14, 20):
        for lo in (-150, -100):
            g.append(("CCI", n, lo, 0))
    return g


def _val(kind, c, h, l, n):
    if kind == "RSI": return _rsi(c, n)
    if kind == "STOCH": return _stoch(c, h, l, n)
    if kind == "BBPCT": return _bbpct(c, n)
    if kind == "CCI": return _cci(c, h, l, n)


def _osc_books(cad):
    """per (kind,n,lo) config: OOS daily return book (u10), long-only MR + min_hold(6) + maker."""
    s_ms = pd.Timestamp(WIN[0]).value // 10**6; e_ms = pd.Timestamp(WIN[1]).value // 10**6
    grid = _grid(); per_cfg = {g: [] for g in grid}
    for sym in SYMS:
        try:
            o, h, l, c, ms = _panel(sym, cad)
        except Exception:
            continue
        e = int(np.searchsorted(ms, e_ms)); s0 = max(0, int(np.searchsorted(ms, s_ms)) - WARMUP)
        o2, h2, l2, c2, ms2 = o[s0:e], h[s0:e], l[s0:e], c[s0:e], ms[s0:e]
        if len(c2) < 40:
            continue
        win = ms2 >= s_ms
        if win.sum() < 30:
            continue
        ret = np.zeros(len(c2)); ret[1:] = c2[1:] / c2[:-1] - 1.0
        idx = pd.to_datetime(ms2[win], unit="ms")
        for g in grid:
            kind, n, lo, hi = g
            v = _val(kind, c2, h2, l2, n)
            held = min_hold(_mr_held(v, lo, hi), 6).astype(np.float64)
            pos = np.zeros(len(c2)); pos[1:] = held[:-1]
            flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
            net = (pos * ret - flips * (MAKER_RT / 2.0))[win]
            per_cfg[g].append(pd.Series(net, index=idx))
    books = {}
    for g, cols in per_cfg.items():
        if cols:
            b4 = pd.concat(cols, axis=1).mean(axis=1, skipna=True)
            books[g] = b4.resample("1D").apply(lambda x: float(np.prod(1 + x) - 1)).dropna()
    return books


def _ma_book(cad):
    """EMA slow-family book (the trend reference) -- daily OOS-window returns."""
    ma_cfg = {}
    for fam in ("2MA", "3MA"):
        ma_cfg.update(distinct_specs(fam, 0.15, max_n=40))
    PR.STRATS.update(ma_cfg)
    slow = [n for n in ma_cfg if 60 <= max(_nums(n)) < 150]
    from strat.portfolio_replay import apply_trail_stop
    s_ms = pd.Timestamp(WIN[0]).value // 10**6; e_ms = pd.Timestamp(WIN[1]).value // 10**6
    per = []
    for sym in SYMS:
        try:
            o, h, l, c, ms = _panel(sym, cad)
        except Exception:
            continue
        e = int(np.searchsorted(ms, e_ms)); s0 = max(0, int(np.searchsorted(ms, s_ms)) - WARMUP)
        c2, ms2 = c[s0:e], ms[s0:e]
        if len(c2) < 40:
            continue
        win = ms2 >= s_ms
        if win.sum() < 30:
            continue
        ret = np.zeros(len(c2)); ret[1:] = c2[1:] / c2[:-1] - 1.0
        uniq = sorted({p for n in slow for p in _nums(n)}); cache = {p: _MA["EMA"](c2, p) for p in uniq}
        nets = []
        for name in slow:
            pp = _nums(name); mas = [cache[p] for p in pp]
            h0 = np.nan_to_num((mas[0] > mas[1]) if len(pp) == 2 else ((mas[0] > mas[1]) & (mas[1] > mas[2]))).astype(np.int8)
            h0 = min_hold(apply_trail_stop(h0.copy(), c2, 0.10)[0].astype(np.int8), 12).astype(np.float64)
            pos = np.zeros(len(c2)); pos[1:] = h0[:-1]
            flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
            nets.append((pos * ret - flips * (MAKER_RT / 2.0))[win])
        idx = pd.to_datetime(ms2[win], unit="ms")
        per.append(pd.Series(np.mean(nets, axis=0), index=idx))
    b4 = pd.concat(per, axis=1).mean(axis=1, skipna=True)
    return b4.resample("1D").apply(lambda x: float(np.prod(1 + x) - 1)).dropna()


def _m(d):
    oos = d[d.index >= pd.Timestamp(SPLIT)].to_numpy()
    if len(oos) < 5:
        return None
    eq = np.cumprod(1 + oos); pk = np.maximum.accumulate(eq)
    return {"net": round(float((eq[-1] - 1) * 100), 1), "sharpe": round(float(np.mean(oos) / (np.std(oos) + 1e-12) * np.sqrt(365)), 2),
            "maxdd": round(float(((eq - pk) / pk).min() * 100), 1)}


def main() -> int:
    global CADENCES
    if "--cadences" in sys.argv:
        CADENCES = sys.argv[sys.argv.index("--cadences") + 1].split(",")
    export = {}
    for cad in CADENCES:
        books = _osc_books(cad)
        recs = []
        for g, d in books.items():
            m = _m(d)
            if m:
                recs.append({"kind": g[0], "cfg": f"{g[0]}({g[1]},lo{g[2]})", "oos": d[d.index >= pd.Timestamp(SPLIT)], **m})
        if not recs:
            continue
        ma = _ma_book(cad); ma_oos = ma[ma.index >= pd.Timestamp(SPLIT)]
        osc_fam = pd.concat([r["oos"] for r in recs], axis=1).mean(axis=1, skipna=True)   # equal-weight osc family book
        j = pd.concat([osc_fam.rename("osc"), ma_oos.rename("ma")], axis=1).dropna()
        corr = float(j["osc"].corr(j["ma"]))
        combined = j.mean(axis=1)
        osc_m = _m(osc_fam.rename(0).to_frame().assign(d=osc_fam)["d"]) if False else None
        # metrics directly
        def mm(s):
            x = s.to_numpy(); eq = np.cumprod(1 + x); pk = np.maximum.accumulate(eq)
            return round(float((eq[-1]-1)*100),1), round(float(np.mean(x)/(np.std(x)+1e-12)*np.sqrt(365)),2), round(float(((eq-pk)/pk).min()*100),1)
        on, osh, odd = mm(j["osc"]); mn, msh, mdd = mm(j["ma"]); cn, csh, cdd = mm(combined)
        print(f"\n########## {cad} -- OSCILLATOR (MR) family vs MA (trend) book + DIVERSIFICATION ##########")
        print(f"   {'book':16} {'net%':>7} {'Sharpe':>7} {'maxDD%':>7}")
        print(f"   {'OSC-family':16} {on:>7} {osh:>7} {odd:>7}")
        print(f"   {'MA-family(trend)':16} {mn:>7} {msh:>7} {mdd:>7}")
        print(f"   {'50/50 TREND+OSC':16} {cn:>7} {csh:>7} {cdd:>7}")
        print(f"   corr(OSC, MA) = {corr:+.2f}  -> {'ORTHOGONAL (diversifies)' if corr < 0.4 else 'correlated (limited diversification)'}")
        better = csh > msh and cdd > mdd
        print(f"   combined vs trend-alone: Sharpe {csh} vs {msh}, maxDD {cdd} vs {mdd} -> {'COMBINING HELPS' if better else 'combining does not improve both'}")
        # per-oscillator top-k
        bykind = {}
        for r in recs:
            bykind.setdefault(r["kind"], []).append(r)
        print("   top per oscillator (by Sharpe):")
        for kind, rs in bykind.items():
            t = sorted(rs, key=lambda x: -x["sharpe"])[:2]
            print(f"      {kind:6}: " + " | ".join(f"{x['cfg']}(Sh{x['sharpe']},{x['net']}%)" for t_ in [t] for x in t_))
        export[cad] = {"corr_to_ma": round(corr, 2), "osc_family": {"net": on, "sharpe": osh, "maxdd": odd},
                       "ma_family": {"net": mn, "sharpe": msh, "maxdd": mdd}, "combined": {"net": cn, "sharpe": csh, "maxdd": cdd},
                       "top": {k: [{"cfg": x["cfg"], "sharpe": x["sharpe"], "net": x["net"]} for x in sorted(v, key=lambda x: -x["sharpe"])[:3]] for k, v in bykind.items()}}
    op = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
    op.mkdir(parents=True, exist_ok=True)
    jt = "_".join(CADENCES)
    json.dump(export, open(op / f"oscillators_{jt}.json", "w"), indent=1, default=str)
    print(f"\n[json] {op / f'oscillators_{jt}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
