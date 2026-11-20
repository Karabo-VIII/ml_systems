"""src/strat/deep2020_exits.py -- BLOCK B: EXIT MECHANISMS -- do we KEEP our returns or give them back?

User /orc 2026-06-13: "include exit mechanisms and grading, because there's no point if we simply give up
all our returns." The participation decomposition (Block A) has a DOWNSIDE-GIVEN-BACK term = the return we
eat while still long in a drop -- the EXIT mechanism is the lever that controls it. So FIX the entry (the
EMA slow-family cross) and sweep EXIT policies, measuring the capture-vs-giveback tradeoff + a GRADE.

EXITS (applied to each config's long runs; family-averaged):
  flip            exit on the MA death-cross (baseline)
  trail5/10/15/20 trailing stop X% from the peak since entry
  tp25 / tp50     take profit at +25% / +50% then flat until next entry
  timestop        exit N bars after entry (~1 week wall-clock per cadence)
  minhold12       do NOT exit before 12 bars (forces participation through wiggles)
  mh12_trail15    min-hold 12 THEN a 15% trail (ride wiggles, cut the big reversal) -- the 'keep gains' combo
  chandelier      trail at 3x ATR(22) from the peak (volatility-scaled)

Per (instrument, TF), family-avg over the EMA slow configs, on 2020 H2 (VAL+OOS): upside-capture /
downside-given-back / net% / maxDD% / time-in + OOS net + OOS block-bootstrap p05 (the GRADE).
RWYB: python -m strat.deep2020_exits --cadences <tf>. No emoji (cp1252).
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
from strat.portfolio_replay import apply_trail_stop, MAKER_RT
from strat.replay_distinct_grid import distinct_specs
from strat.structural_fixes import min_hold
from strat.battery import block_bootstrap_p05_p95
from strat.ma_type_upgrade import _MA, _nums
from strat.ma_2020_breakdown import _panel

WIN = ("2020-07-01", "2021-01-01"); SPLIT = "2020-10-01"
WARMUP = 400
CADENCES = ["1d", "4h", "2h", "1h", "30m", "15m"]
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]
TIMESTOP_BARS = {"1d": 7, "4h": 42, "2h": 84, "1h": 168, "30m": 336, "15m": 672}   # ~1 week
EXITS = ["flip", "trail5", "trail10", "trail15", "trail20", "tp25", "tp50", "timestop", "minhold12", "mh12_trail15", "chandelier"]


def _runs(held):
    """list of (start, end_exclusive) for each contiguous long run in the entry-held series."""
    h = held.astype(np.int8)
    d = np.diff(np.concatenate([[0], h, [0]]))
    starts = np.where(d == 1)[0]; ends = np.where(d == -1)[0]
    return list(zip(starts, ends))


def _tp(held, c, pct):
    h = held.copy()
    for s, e in _runs(held):
        ce = c[s]
        for i in range(s, e):
            if c[i] >= ce * (1 + pct):
                h[i + 1:e] = 0; break
    return h


def _timestop(held, n):
    h = held.copy()
    for s, e in _runs(held):
        if e - s > n:
            h[s + n:e] = 0
    return h


def _chandelier(held, c, hi, lo, k=3.0, per=22):
    tr = np.maximum(hi - lo, np.abs(hi - np.concatenate([[c[0]], c[:-1]])))
    atr = pd.Series(tr).rolling(per, min_periods=1).mean().to_numpy()
    h = held.copy()
    for s, e in _runs(held):
        peak = c[s]
        for i in range(s, e):
            peak = max(peak, c[i])
            if c[i] <= peak - k * atr[i]:
                h[i + 1:e] = 0; break
    return h


def _apply_exit(held, c, hi, lo, exit_, cad):
    h = held.astype(np.int8)
    if exit_ == "flip":
        return h
    if exit_.startswith("trail"):
        return apply_trail_stop(h.copy(), c, int(exit_[5:]) / 100.0)[0].astype(np.int8)
    if exit_ == "tp25":
        return _tp(h, c, 0.25)
    if exit_ == "tp50":
        return _tp(h, c, 0.50)
    if exit_ == "timestop":
        return _timestop(h, TIMESTOP_BARS.get(cad, 7))
    if exit_ == "minhold12":
        return min_hold(h, 12).astype(np.int8)
    if exit_ == "mh12_trail15":
        return apply_trail_stop(min_hold(h, 12).astype(np.int8).copy(), c, 0.15)[0].astype(np.int8)
    if exit_ == "chandelier":
        return _chandelier(h, c, hi, lo)
    return h


def _family(sym, cad, slow):
    try:
        o, h, l, c, ms = _panel(sym, cad)
    except Exception:
        return None
    w0 = pd.Timestamp(WIN[0]).value // 10**6; w1 = pd.Timestamp(WIN[1]).value // 10**6
    e = int(np.searchsorted(ms, w1)); s0 = max(0, int(np.searchsorted(ms, w0)) - WARMUP)
    o, c, hi, lo, ms = o[s0:e], c[s0:e], h[s0:e], l[s0:e], ms[s0:e]
    if len(c) < 40:
        return None
    win = ms >= w0
    if win.sum() < 30:
        return None
    # entry-held per config (EMA cross)
    uniq = sorted({p for n in slow for p in _nums(n)}); cache = {p: _MA["EMA"](c, p) for p in uniq}
    entrys = []
    for name in slow:
        pp = _nums(name); mas = [cache[p] for p in pp]
        entrys.append(np.nan_to_num((mas[0] > mas[1]) if len(pp) == 2 else ((mas[0] > mas[1]) & (mas[1] > mas[2]))).astype(np.int8))
    return o, c, hi, lo, ms, win, entrys


def _decompose(fpos, logret, cost, dates):
    up = logret > 0; dn = logret < 0
    captured = np.sum(fpos[up] * logret[up]); givenbk = np.sum(fpos[dn] * logret[dn])
    avoided = np.sum((1 - fpos[dn]) * logret[dn]); total_up = np.sum(logret[up]); total_dn = np.sum(logret[dn])
    net = fpos * logret - cost
    eq = np.cumprod(np.exp(net)); peak = np.maximum.accumulate(eq); maxdd = float(((eq - peak) / peak).min() * 100)
    oos = dates >= pd.Timestamp(SPLIT)
    oos_net = float((np.exp(np.sum(net[oos])) - 1) * 100) if oos.sum() > 3 else float("nan")
    daily = pd.Series(np.exp(net) - 1, index=dates).resample("1D").apply(lambda x: float(np.prod(1 + x) - 1))
    oos_daily = daily[daily.index >= pd.Timestamp(SPLIT)].dropna().to_numpy()
    p05 = block_bootstrap_p05_p95(oos_daily).get("p05") if len(oos_daily) > 10 else None
    return {"up_capture": round(float(captured / total_up), 3) if total_up > 0 else None,
            "dn_givenback_pct": round(float((np.exp(givenbk) - 1) * 100), 1),
            "net_pct": round(float((np.exp(np.sum(net)) - 1) * 100), 1),
            "maxdd_pct": round(maxdd, 1), "time_in": round(float(np.mean(fpos)), 3),
            "oos_net_pct": round(oos_net, 1), "oos_p05": None if p05 is None else round(p05, 2)}


def main() -> int:
    global CADENCES
    if "--cadences" in sys.argv:
        CADENCES = sys.argv[sys.argv.index("--cadences") + 1].split(",")
    ma_cfg = {}
    for fam in ("2MA", "3MA"):
        ma_cfg.update(distinct_specs(fam, 0.15, max_n=60))
    PR.STRATS.update(ma_cfg)
    slow = [n for n in ma_cfg if 60 <= max(_nums(n)) < 150]
    print(f"BLOCK B exits: EMA entry x {len(EXITS)} exits x {len(slow)} configs x {len(CADENCES)} TF; 2020 H2\n")

    out = {}
    for cad in CADENCES:
        print(f"########## {cad} -- exit x (up_capture / dn_givenback% / net% / maxDD% / time_in / OOSnet% / OOSp05) family-avg ##########")
        print(f"   {'exit':13} {'up_capt':>8} {'givenbk%':>9} {'net%':>8} {'maxDD%':>8} {'time_in':>8} {'OOSnet%':>8} {'OOSp05':>7}")
        fam_cache = {sym: _family(sym, cad, slow) for sym in SYMS}
        for ex in EXITS:
            rows = []
            for sym in SYMS:
                fc = fam_cache[sym]
                if fc is None:
                    continue
                o, c, hi, lo, ms, win, entrys = fc
                logret = np.zeros(len(c)); logret[1:] = np.log(c[1:] / c[:-1])
                poss = []
                for ent in entrys:
                    held = _apply_exit(ent, c, hi, lo, ex, cad).astype(np.float64)
                    pos = np.zeros(len(c)); pos[1:] = held[:-1]; poss.append(pos)
                fpos = np.mean(poss, axis=0)
                flips = np.abs(np.diff(np.concatenate([[0.0], fpos])))
                cost = flips * (MAKER_RT / 2.0)
                dates = pd.to_datetime(ms[win], unit="ms")
                d = _decompose(fpos[win], logret[win], cost[win], dates)
                out[(cad, ex, sym)] = d; rows.append(d)
            if not rows:
                continue
            def a(k):
                v = [r[k] for r in rows if r[k] is not None]; return float(np.mean(v)) if v else float("nan")
            print(f"   {ex:13} {a('up_capture'):>8.2f} {a('dn_givenback_pct'):>9.1f} {a('net_pct'):>8.1f} "
                  f"{a('maxdd_pct'):>8.1f} {a('time_in'):>8.2f} {a('oos_net_pct'):>8.1f} {a('oos_p05'):>7.1f}")
        print()

    op = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
    op.mkdir(parents=True, exist_ok=True)
    jt = "_".join(CADENCES)
    json.dump({f"{c}|{e}|{s}": d for (c, e, s), d in out.items()}, open(op / f"exits_{jt}.json", "w"), indent=1, default=str)
    print(f"[json] {op / f'exits_{jt}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
