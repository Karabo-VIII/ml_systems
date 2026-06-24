"""Focused random-null for ADX|4h — the most SELECTIVE static candidate (40-47% deploy).

ADX|4h was DEV/OOS consistent (DEV +0.74pp, OOS +1.09pp at thr0.6) and selective.
Selectivity matters: unlike the ~60-65% deploy of RSI/Bollinger, ADX deploys ~44%,
so any edge is more clearly TIMING than being-long. Confirm with random-null at the
same per-asset deploy rate.
"""
from __future__ import annotations

import warnings; warnings.filterwarnings("ignore")
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CRYPTO = ROOT.parent

import strat.fleet_lab as fl
from strat.ma_wealth_driver import _load_asset, _u100_symbols
from strat.ti_static_exhaustion import apply_confirmation, _btc_regime
from strat.ti_static_exhaustion_rest import votes_price

BARS_PER_DAY = fl.BARS_PER_DAY
OOS_START = "2024-05-15"; OOS_END = "2025-01-01"
OOS_WALL_MS = int(pd.Timestamp(OOS_END).value // 10**6)
OUTDIR = CRYPTO / "runs" / "engines" / "V3_ti_families" / "static_randomnull"

FAMILY, TF, THR, CONFIRM = "ADX", "4h", 0.6, False
N_NULL = 200


def build(slice_days=7, lookback_days=7):
    bpd = BARS_PER_DAY[TF]
    slice_bars = slice_days * bpd; lookback_bars = lookback_days * bpd
    cd, hd, dr = {}, {}, {}
    for sym in _u100_symbols():
        rec = _load_asset(sym, TF, OOS_WALL_MS)
        if rec is None:
            continue
        dates, o, h, l, c = rec
        frac = votes_price(FAMILY, o, h, l, c)
        raw = (frac > THR).astype(np.int8)
        held = apply_confirmation(c, raw, lookback_bars, slice_bars) if CONFIRM else raw
        cd[sym] = pd.Series(c, index=dates); hd[sym] = pd.Series(held, index=dates)
        dr[sym] = float(np.mean(held))
    return cd, hd, dr, slice_bars


def linwk(cd, hd, btc_regime, bpd):
    all_dates = sorted(set().union(*[s.index.tolist() for s in cd.values()]))
    ws = pd.Timestamp(OOS_START); we = pd.Timestamp(OOS_END)
    win = [d for d in all_dates if ws <= d < we]
    eng, al = [], []
    for i in range(1, len(win)):
        dp, dc = win[i-1], win[i]
        er, ar = [], []
        for sym in cd:
            if dp in cd[sym].index and dc in cd[sym].index:
                cp = cd[sym][dp]; cc = cd[sym][dc]
                if cp > 0:
                    r = (cc/cp) - 1.0; ar.append(r)
                    if dp in hd[sym].index and hd[sym][dp] == 1:
                        er.append(r)
        e = float(np.mean(er)) if er else 0.0
        if btc_regime is not None:
            d1 = pd.Timestamp(dp).normalize()
            prior = btc_regime.index[btc_regime.index <= d1]
            if len(prior) and int(btc_regime[prior[-1]]) == 0:
                e = 0.0
        eng.append(e); al.append(float(np.mean(ar)) if ar else 0.0)
    eng = np.array(eng); al = np.array(al)
    return float(np.mean(eng)) * bpd * 7 * 100, float(np.mean(al)) * bpd * 7 * 100


def rand_like(hd, dr, slice_bars, rng):
    out = {}
    for sym, hs in hd.items():
        n = len(hs); d = dr[sym]; rh = np.zeros(n, dtype=np.int8)
        i = 0
        while i < n:
            if rng.random() < d:
                rh[i:min(i+slice_bars, n)] = 1
            i += slice_bars
        out[sym] = pd.Series(rh, index=hs.index)
    return out


def main():
    bpd = BARS_PER_DAY[TF]
    print(f"=== ADX|{TF} thr{THR} conf{CONFIRM} — random-null (n={N_NULL}) ===")
    cd, hd, dr, sb = build()
    btc = _btc_regime(OOS_WALL_MS)
    eng, al = linwk(cd, hd, btc, bpd)
    real = eng - al
    avg_deploy = float(np.mean(list(dr.values())))
    print(f"  real OOS edge: {real:+.3f}pp (eng {eng:+.3f} vs AL {al:+.3f}), avg deploy {avg_deploy:.1%}")

    rng = np.random.default_rng(42); nulls = []
    for _ in range(N_NULL):
        rhd = rand_like(hd, dr, sb, rng)
        re, ra = linwk(cd, rhd, btc, bpd)
        nulls.append(re - ra)
    nulls = np.array(nulls)
    pct = float(np.mean(nulls < real)) * 100
    print(f"  null mean {nulls.mean():+.3f}pp (sd {nulls.std():.3f}) -> real beats {pct:.0f}th pct "
          f"{'PASS' if pct >= 95 else 'FAIL'}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(OUTDIR / f"adx_randomnull_{ts}.json", "w") as f:
        json.dump({"family": FAMILY, "tf": TF, "threshold": THR, "confirm": CONFIRM,
                   "real_edge_pp": round(real, 3), "avg_deploy": round(avg_deploy, 3),
                   "null_mean": round(float(nulls.mean()), 3), "null_sd": round(float(nulls.std()), 3),
                   "real_vs_null_pct": round(pct, 1), "pass": bool(pct >= 95)}, f, indent=2)
    print(f"  Saved adx_randomnull_{ts}.json")


if __name__ == "__main__":
    main()
