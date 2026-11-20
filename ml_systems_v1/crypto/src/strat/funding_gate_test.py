"""src/strat/funding_gate_test.py -- does a FUNDING-based market gate transfer where the price gate failed?

WHY: the regime-gate search (D74) tested only PRICE gates (BTC vs SMA); the BTC-price market gate softened
ONE in-sample bear but did NOT transfer OOS. D74's open thread = "a LEADING market-regime signal". Funding
LEADS price (crowded longs -> correction; negative funding -> bearish positioning), and we have it locally
(data/raw/<SYM>/funding). This tests funding as the gate instrument on the FULL 4h keeper stack, LEAK-FREE
(causal trailing z-score, not the full-history-normalized panel column), transfer-focused (bear/VAL/OOS).

GATES on the FULL stack (FIXED 2MA-slow + TRAIL10 + min_hold12 + MAKER, 4h):
  NONE          no gate
  FUND_POS      long only when BTC trailing-30d-mean funding > 0   (uptrend positioning bias)
  FUND_NOTHOT   long UNLESS BTC funding causal-z > +1.5            (sit out crowded-long euphoria)
  FUND_BOTH     long when (trailing-mean fund > 0) AND (z < +1.5)  (uptrend but not euphoric)
  PRICE_REF     BTC > BTC.SMA100 hysteresis (the D74 price gate, for reference)

Per (gate, period in {Jun2022 bear, VAL, OOS}): book ROI% / maxDD%. UNSEEN sealed. MAKER.
RWYB: python -m strat.funding_gate_test. No emoji (cp1252).
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.portfolio_replay as PR
from strat.portfolio_replay import holding_state, apply_trail_stop, MAKER_RT
from strat.replay_distinct_grid import distinct_specs
from strat.ma_mechanics import _cached_panel
from strat.structural_fixes import min_hold
from strat.complete_stack import _sma, _hysteresis

PERIODS = {
    "Jun2022_bear": ("2022-06-01", "2022-07-01"),
    "VAL":          ("2024-05-15", "2025-03-15"),
    "OOS":          ("2025-03-15", "2025-12-31"),   # UNSEEN (>2025-12-31) NEVER touched
}
WARMUP = 600
GATES = ["NONE", "FUND_POS", "FUND_NOTHOT", "FUND_BOTH", "PRICE_REF"]


def _nums(n):
    return [int(x) for x in re.findall(r"\d+", n)]


_FUND = {}


def _btc_funding_signals():
    """causal BTC funding signals at the funding event clock: (ms, trailing_mean>0, causal_z)."""
    if "x" in _FUND:
        return _FUND["x"]
    fs = sorted(glob.glob(str(ROOT.parent / "data" / "raw" / "BTCUSDT" / "funding" / "*.parquet")))
    df = pl.concat([pl.read_parquet(f) for f in fs]).sort("timestamp")
    ms = df["timestamp"].to_numpy().astype(np.int64)
    f = df["funding_rate"].to_numpy().astype(float)
    s = pd.Series(f)
    # causal trailing window (90 events ~ 30 days at 3/day)
    tmean = s.rolling(90, min_periods=20).mean().to_numpy()
    tstd = s.rolling(90, min_periods=20).std().to_numpy()
    # z uses the PRIOR bar's stats (shift) to be strictly causal at decision time
    z = np.zeros(len(f))
    prev_m = pd.Series(tmean).shift(1).to_numpy()
    prev_s = pd.Series(tstd).shift(1).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (f - prev_m) / (prev_s + 1e-9)
    z = np.nan_to_num(z)
    pos = np.nan_to_num(pd.Series(tmean).shift(1).to_numpy()) > 0   # trailing mean > 0 (causal)
    _FUND["x"] = (ms, pos.astype(bool), z)
    return _FUND["x"]


_BTC_PX = {}


def _btc_price_hyst(cadence):
    if cadence not in _BTC_PX:
        o, h, l, c, ms = _cached_panel("BTCUSDT", cadence)
        _BTC_PX[cadence] = (ms, _hysteresis(c, 100, 0.03))
    return _BTC_PX[cadence]


def _gate_mask(ms_asset, cadence, gate):
    if gate == "NONE":
        return np.ones(len(ms_asset), bool)
    if gate == "PRICE_REF":
        bms, breg = _btc_price_hyst(cadence)
        idx = np.clip(np.searchsorted(bms, ms_asset, side="right") - 1, 0, len(breg) - 1)
        return breg[idx]
    fms, fpos, fz = _btc_funding_signals()
    idx = np.clip(np.searchsorted(fms, ms_asset, side="right") - 1, 0, len(fpos) - 1)
    pos, z = fpos[idx], fz[idx]
    if gate == "FUND_POS":
        return pos
    if gate == "FUND_NOTHOT":
        return z < 1.5
    if gate == "FUND_BOTH":
        return pos & (z < 1.5)
    return np.ones(len(ms_asset), bool)


def _full_weight(name, o, c):
    h = holding_state(name, o, c, c, c).astype(np.int8)
    h = apply_trail_stop(h.copy(), c, 0.10)[0].astype(np.int8)
    return min_hold(h, 12).astype(np.float64)


def book_metrics(slow, cadence, start, end, gate):
    s_ms = pd.Timestamp(start).value // 10**6
    e_ms = pd.Timestamp(end).value // 10**6
    syms = [a["symbol"] for a in yaml.safe_load(open(ROOT.parent / "config" / "universes" / "u10.yaml"))["assets"]]
    per_cell = []
    for sym in syms:
        try:
            o, h, l, c, ms = _cached_panel(sym, cadence)
        except Exception:
            continue
        e_idx = int(np.searchsorted(ms, e_ms))
        s_idx = max(0, int(np.searchsorted(ms, s_ms)) - WARMUP)
        o, c, ms = o[s_idx:e_idx], c[s_idx:e_idx], ms[s_idx:e_idx]
        if len(c) < 20:
            continue
        wm = ms >= s_ms
        if wm.sum() < 10:
            continue
        gmask = _gate_mask(ms, cadence, gate).astype(np.float64)
        ret = np.zeros(len(c)); ret[1:] = c[1:] / c[:-1] - 1.0
        for name in slow:
            w = _full_weight(name, o, c) * gmask
            pos = np.zeros(len(c)); pos[1:] = w[:-1]
            flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
            per_cell.append((pos * ret - flips * (MAKER_RT / 2.0))[wm])
    if not per_cell:
        return {}
    m = min(len(x) for x in per_cell)
    bk = np.mean([x[:m] for x in per_cell], axis=0)
    eq = np.cumprod(1 + bk); peak = np.maximum.accumulate(eq)
    return {"roi": round(float(eq[-1] - 1) * 100, 1), "maxdd": round(float(((eq - peak) / peak).min() * 100), 1)}


def main() -> int:
    allcfg = {}
    for fam in ("2MA", "3MA"):
        allcfg.update(distinct_specs(fam, 0.15, max_n=60))
    PR.STRATS.update(allcfg)
    slow = [n for n in allcfg if len(_nums(n)) == 2 and 60 <= max(_nums(n)) < 150]
    fms, fpos, fz = _btc_funding_signals()
    print(f"funding gate test: {len(slow)} cfg, 4h, MAKER; BTC funding {len(fms)} events, "
          f"frac trailing-mean>0 = {fpos.mean():.2f}, z range [{fz.min():.1f},{fz.max():.1f}]\n")

    results = {}
    print(f"   {'gate':12}" + "".join(f"{p:>18}" for p in PERIODS))
    for gate in GATES:
        row = f"   {gate:12}"
        for plabel, (s, e) in PERIODS.items():
            mt = book_metrics(slow, "4h", s, e, gate)
            results[(gate, plabel)] = mt
            row += f"{(str(mt.get('roi'))+'/'+str(mt.get('maxdd'))):>18}" if mt else f"{'--':>18}"
        print(row)

    print("\n[DELTA vs NONE]  (want bear UP/less-neg AND VAL/OOS not hurt -- the price gate failed VAL/OOS)")
    base = {p: results[("NONE", p)].get("roi", np.nan) for p in PERIODS}
    for gate in GATES[1:]:
        d = {p: results[(gate, p)].get("roi", np.nan) - base[p] for p in PERIODS}
        opens = d["Jun2022_bear"] > 0.3 and d["VAL"] > -1.5 and d["OOS"] > -1.0
        print(f"   {gate:12} d_bear {d['Jun2022_bear']:>+6.1f}  d_VAL {d['VAL']:>+6.1f}  d_OOS {d['OOS']:>+6.1f}   "
              f"{'TRANSFERS' if opens else 'no'}")

    out = ROOT.parent / "runs" / "periods" / "_OOS_CONFIRM" / "funding_gate_test.json"
    json.dump({f"{g}|{p}": m for (g, p), m in results.items()}, open(out, "w"), indent=1, default=str)
    print(f"\n[json] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
