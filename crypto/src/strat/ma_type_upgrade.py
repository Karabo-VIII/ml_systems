"""src/strat/ma_type_upgrade.py -- UPGRADE the 2MA/3MA: keep the cross structure, swap the MA TYPE.

WHY (user /orc 2026-06-12 halt+correct): we are upgrading the 2MA/3MA we have been dealing with -- NOT
moving to other indicators/families yet (breakout/MR parked). The core MA weakness the building block
exposed is the FIXED-EMA tradeoff: a fast EMA whipsaws in chop, a slow EMA lags in trend. The genuine
ON-FAMILY upgrade is the MA ITSELF -- low-lag types (Hull/DEMA/TEMA) and ADAPTIVE types (KAMA/VIDYA) that
are fast in trend and slow in chop. Same 2-cross / 3-cross structure, upgraded smoothing.

This swaps the MA type into the SAME slow family configs (a,b[,c]) and compares vs the EMA baseline on:
  (1) in-sample mechanics -- whipsaw + net (does adaptive cut the whipsaw that drove the cost drag?)
  (2) transfer -- bear(Jun2022) / VAL / OOS at the FIXED level (pure MA-type effect, no overlay confound)
  (3) best types -- OOS scorecard p05 (does any MA-type UPGRADE the EMA robustness?)
4h, equal-weight u10 book, causal MtM, taker (FIXED level). UNSEEN sealed. RWYB: python -m strat.ma_type_upgrade.
No emoji (cp1252).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.portfolio_replay as PR
from strat.portfolio_replay import TAKER_RT
from strat.replay_distinct_grid import distinct_specs
from strat.ma_mechanics import _cached_panel
from strat.scorecard import score_book

WARMUP = 600
PERIODS = {"Jun2022_bear": ("2022-06-01", "2022-07-01"),
           "VAL": ("2024-05-15", "2025-03-15"), "OOS": ("2025-03-15", "2025-12-31")}
MA_TYPES = ["EMA", "SMA", "WMA", "HMA", "DEMA", "TEMA", "KAMA", "VIDYA"]


# ---- MA implementations (all causal: value at t uses close[:t+1]) ----
def _sma(c, n):
    return pd.Series(c).rolling(n, min_periods=1).mean().to_numpy()


def _ema(c, n):
    return pd.Series(c).ewm(span=n, adjust=False, min_periods=1).mean().to_numpy()


def _wma(c, n):
    w = np.arange(1, n + 1, dtype=float)
    return pd.Series(c).rolling(n, min_periods=1).apply(
        lambda x: np.dot(x, w[-len(x):]) / w[-len(x):].sum(), raw=True).to_numpy()


def _hma(c, n):
    half = max(1, int(n / 2)); sq = max(1, int(np.sqrt(n)))
    raw = 2 * _wma(c, half) - _wma(c, n)
    return _wma(raw, sq)


def _dema(c, n):
    e = _ema(c, n); return 2 * e - _ema(e, n)


def _tema(c, n):
    e = _ema(c, n); e2 = _ema(e, n); e3 = _ema(e2, n)
    return 3 * e - 3 * e2 + e3


def _kama(c, n, fast=2, slow=30):
    c = np.asarray(c, float)
    nn = max(1, min(n, len(c) - 1))                       # clamp window to series length
    change = np.abs(c - np.concatenate([np.full(nn, c[0]), c[:-nn]]))
    vol = pd.Series(np.abs(np.diff(c, prepend=c[0]))).rolling(n, min_periods=1).sum().to_numpy()
    er = np.where(vol > 1e-12, change / vol, 0.0)
    sc = (er * (2 / (fast + 1) - 2 / (slow + 1)) + 2 / (slow + 1)) ** 2
    out = np.empty(len(c)); out[0] = c[0]
    for i in range(1, len(c)):
        out[i] = out[i - 1] + sc[i] * (c[i] - out[i - 1])
    return out


def _vidya(c, n, lookback=9):
    c = np.asarray(c, float)
    up = pd.Series(np.where(np.diff(c, prepend=c[0]) > 0, np.diff(c, prepend=c[0]), 0.0))
    dn = pd.Series(np.where(np.diff(c, prepend=c[0]) < 0, -np.diff(c, prepend=c[0]), 0.0))
    su = up.rolling(lookback, min_periods=1).sum().to_numpy()
    sd = dn.rolling(lookback, min_periods=1).sum().to_numpy()
    cmo = np.where((su + sd) > 1e-12, np.abs(su - sd) / (su + sd), 0.0)
    a = 2 / (n + 1)
    out = np.empty(len(c)); out[0] = c[0]
    for i in range(1, len(c)):
        k = a * cmo[i]
        out[i] = out[i - 1] + k * (c[i] - out[i - 1])
    return out


_MA = {"EMA": _ema, "SMA": _sma, "WMA": _wma, "HMA": _hma, "DEMA": _dema, "TEMA": _tema,
       "KAMA": lambda c, n: _kama(c, n), "VIDYA": lambda c, n: _vidya(c, n)}


def _nums(n):
    return [int(x) for x in re.findall(r"\d+", n)]


def held_cross(c, periods, ma_type):
    """2MA: long when MA(a) > MA(b). 3MA: long when MA(a) > MA(b) > MA(c) (aligned)."""
    f = _MA[ma_type]
    mas = [f(c, p) for p in periods]
    if len(periods) == 2:
        h = (mas[0] > mas[1])
    else:
        h = (mas[0] > mas[1]) & (mas[1] > mas[2])
    return np.nan_to_num(h).astype(np.int8)


def book(slow_cfgs, ma_type, start, end, date_index=False, want_whip=False):
    s_ms = pd.Timestamp(start).value // 10**6
    e_ms = pd.Timestamp(end).value // 10**6
    syms = [a["symbol"] for a in yaml.safe_load(open(ROOT.parent / "config" / "universes" / "u10.yaml"))["assets"]]
    per_cell, series, whip = [], {}, []
    for sym in syms:
        try:
            o, h, l, c, ms = _cached_panel(sym, "4h")
        except Exception:
            continue
        e_idx = int(np.searchsorted(ms, e_ms))
        s_idx = max(0, int(np.searchsorted(ms, s_ms)) - WARMUP)
        c2, ms2 = c[s_idx:e_idx], ms[s_idx:e_idx]
        if len(c2) < 30:
            continue
        wm = ms2 >= s_ms
        if wm.sum() < 10:
            continue
        ret = np.zeros(len(c2)); ret[1:] = c2[1:] / c2[:-1] - 1.0
        for name in slow_cfgs:
            held = held_cross(c2, _nums(name), ma_type).astype(np.float64)
            pos = np.zeros(len(c2)); pos[1:] = held[:-1]
            flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
            net = (pos * ret - flips * (TAKER_RT / 2.0))[wm]
            per_cell.append(net)
            if want_whip:
                # whipsaw = fraction of trades held <= 2 bars
                entries = np.where(np.diff(np.concatenate([[0], held.astype(int)])) == 1)[0]
                exits = np.where(np.diff(np.concatenate([[0], held.astype(int)])) == -1)[0]
                durs = [exits[exits > e][0] - e for e in entries if np.any(exits > e)]
                if durs:
                    whip.append(float(np.mean(np.array(durs) <= 2)) * 100)
            if date_index:
                series.setdefault(sym, []).append(pd.Series(net, index=pd.to_datetime(ms2[wm], unit="ms")))
    if not per_cell:
        return {}, None, None
    m = min(len(x) for x in per_cell)
    bk = np.mean([x[:m] for x in per_cell], axis=0)
    eq = np.cumprod(1 + bk); peak = np.maximum.accumulate(eq)
    mt = {"roi": round(float(eq[-1] - 1) * 100, 1), "maxdd": round(float(((eq - peak) / peak).min() * 100), 1),
          "whip": round(float(np.mean(whip)), 1) if whip else None}
    return mt, series, None


def main() -> int:
    ma_cfg = {}
    for fam in ("2MA", "3MA"):
        ma_cfg.update(distinct_specs(fam, 0.15, max_n=60))
    PR.STRATS.update(ma_cfg)
    slow2 = [n for n in ma_cfg if len(_nums(n)) == 2 and 60 <= max(_nums(n)) < 150]
    slow3 = [n for n in ma_cfg if len(_nums(n)) == 3 and 60 <= max(_nums(n)) < 150]
    print(f"MA-TYPE upgrade: 2MA-slow={len(slow2)}, 3MA-slow={len(slow3)} configs; types {MA_TYPES}\n")

    # (1)+(2): per MA type, in-sample combined-2020 (whipsaw) + transfer (bear/VAL/OOS), 2MA-slow, FIXED level
    print("## 2MA-slow, FIXED level (pure MA-type effect) -- ROI%/maxDD% per period + Jan-2020 whipsaw%")
    print(f"   {'MA_type':8}{'Jan20whip':>11}" + "".join(f"{p:>16}" for p in PERIODS))
    results = {}
    for mt_ in MA_TYPES:
        whp, _, _ = book(slow2, mt_, "2020-01-07", "2020-02-07", want_whip=True)
        line = f"   {mt_:8}{(str(whp.get('whip'))+'%' if whp else '?'):>11}"
        for plabel, (s, e) in PERIODS.items():
            m, _, _ = book(slow2, mt_, s, e)
            results[(mt_, plabel)] = m
            line += f"{(str(m.get('roi'))+'/'+str(m.get('maxdd'))):>16}" if m else f"{'--':>16}"
        print(line)

    # (3): OOS scorecard p05 per MA type (UNSEEN sealed), 2MA-slow FIXED
    print("\n## OOS-heldout block-bootstrap p05 per MA type (2MA-slow, UNSEEN sealed; robust iff >0)")
    sc = {}
    for mt_ in MA_TYPES:
        _, series, _ = book(slow2, mt_, "2018-01-01", "2025-12-31", date_index=True)
        if not series:
            continue
        allc = [x for lst in series.values() for x in lst]
        daily = pd.concat(allc, axis=1).mean(axis=1, skipna=True).resample("1D").apply(lambda x: float((1 + x).prod() - 1)).dropna()
        card = score_book(f"2MA_{mt_}", daily)
        oosp = card["per_split"].get("OOS", {}); hb = card["heldout_block_bootstrap"]
        sc[mt_] = {"oos_compound": oosp.get("compound_pct"), "oos_sharpe": oosp.get("sharpe"), "heldout_p05": hb.get("p05")}
        print(f"   {mt_:8} OOS compound {str(oosp.get('compound_pct')):>7}%  Sharpe {str(oosp.get('sharpe')):>5}  "
              f"OOS p05 {str(hb.get('p05')):>7}")

    # verdict vs EMA baseline
    base = sc.get("EMA", {})
    print(f"\n   EMA baseline: OOS compound {base.get('oos_compound')}%, Sharpe {base.get('oos_sharpe')}, p05 {base.get('heldout_p05')}")
    better = [(t, v) for t, v in sc.items() if t != "EMA" and v.get("heldout_p05") is not None
              and base.get("heldout_p05") is not None and v["heldout_p05"] > base["heldout_p05"]]
    better.sort(key=lambda kv: -kv[1]["heldout_p05"])
    if better:
        print("   MA types that UPGRADE EMA on OOS p05: " +
              ", ".join(f"{t}(p05 {v['heldout_p05']}, Sh {v['oos_sharpe']})" for t, v in better))
    else:
        print("   No MA type upgrades the EMA OOS p05 -- the MA-type axis does not lift robustness")
    out = ROOT.parent / "runs" / "periods" / "_OOS_CONFIRM" / "ma_type_upgrade.json"
    json.dump({"transfer": {f"{k[0]}|{k[1]}": v for k, v in results.items()}, "scorecard": sc}, open(out, "w"), indent=1, default=str)
    print(f"\n[json] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
