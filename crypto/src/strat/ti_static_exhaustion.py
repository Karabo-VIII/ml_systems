"""Static Exhaustion Test — pull the static levers the non-MA families never got.

The user's point: if static can't be made to work, dynamic (adaptive) can't either,
because dynamic is just static-with-adaptive-params. So before Phase 2 we must PROVE
the static design space is genuinely exhausted for the non-MA families.

Gap analysis (what the SOTA sweep did NOT test):
  1. VOTE THRESHOLD was fixed at 0.5. Families 6-8 deploy 98-100% => the selectivity
     lever was never engaged. Sweep {0.5, 0.6, 0.7, 0.8}.
  2. DECOUPLED CONFIRMATION: the MA family won with `signal & (close>EMA_lb) & (EMA_lb rising)`
     + a fixed slice-hold. The non-MA families never got this confirmation layer.
  3. RANDOM-NULL was never run for non-MA families (only MA cleared it).

This script tests levers 1+2 on the OOS window (the honest arbiter) for the three
close-based families (RSI/MACD/Bollinger). If NO (threshold, confirmation) combo
produces a SELECTIVE engine (deploy < ~70%) that BEATS always-long OOS, static is
genuinely exhausted for non-MA and we have our answer.
"""
from __future__ import annotations

import warnings; warnings.filterwarnings("ignore")
import json
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CRYPTO = ROOT.parent

import strat.fleet_lab as fl
from strat.ma_wealth_driver import _load_asset, _u100_symbols, MAKER_RT
from strat.ma_type_upgrade import _ema
from strat.ti_engine_sota import (
    rsi_single_held, macd_single_held, bollinger_single_held,
    RSI_CONFIGS, MACD_CONFIGS, BOLLINGER_CONFIGS,
)

BARS_PER_DAY = fl.BARS_PER_DAY
OOS_START = "2024-05-15"
OOS_END = "2025-01-01"
OOS_WALL_MS = int(pd.Timestamp(OOS_END).value // 10**6)
OUTDIR = CRYPTO / "runs" / "engines" / "V3_ti_families" / "static_exhaustion"

FAM = {
    "RSI": (RSI_CONFIGS, lambda c, cfg: rsi_single_held(c, *cfg)),
    "MACD": (MACD_CONFIGS, lambda c, cfg: macd_single_held(c, *cfg)),
    "BOLLINGER": (BOLLINGER_CONFIGS, lambda c, cfg: bollinger_single_held(c, *cfg)),
}


def _btc_regime(wall_ms):
    rec = _load_asset("BTCUSDT", "1d", wall_ms)
    if not rec:
        return None
    d, _, _, _, cl = rec
    sma = pd.Series(cl, index=d).rolling(100).mean()
    s = pd.Series(0, index=d, dtype=int); st = 0
    for i in range(100, len(d)):
        if np.isnan(sma.iloc[i]):
            continue
        if st == 0 and cl[i] > sma.iloc[i] * 1.02:
            st = 1
        elif st == 1 and cl[i] < sma.iloc[i] * 0.98:
            st = 0
        s.iloc[i] = st
    return s


def ensemble_votes(family, close):
    """Return the vote fraction (0..1) across all configs for one asset."""
    configs, fn = FAM[family]
    n = len(close)
    votes = np.zeros(n, dtype=float)
    for cfg in configs:
        try:
            votes += fn(close, cfg).astype(float)
        except Exception:
            continue
    return votes / len(configs)


def apply_confirmation(close, raw_held, lookback_bars, slice_bars):
    """MA-style decoupled confirmation + slice-hold (the architecture MA won with)."""
    fin = np.asarray(close, dtype=float)
    n = len(fin)
    m = np.isfinite(fin)
    out = np.zeros(n, dtype=np.int8)
    if m.sum() < max(5, lookback_bars // 4):
        return out
    cseg = fin[m]
    ema_lb = _ema(cseg, lookback_bars)
    above = cseg > ema_lb
    rising = np.concatenate([[True], ema_lb[1:] >= ema_lb[:-1]])
    rh = np.asarray(raw_held, dtype=bool)[m]
    confirmed = rh & above & rising
    held_seg = np.zeros(len(cseg), dtype=np.int8)
    i = 0; L = len(cseg)
    while i < L:
        if confirmed[i]:
            end = min(i + slice_bars, L)
            j = i
            while j < end and confirmed[j]:
                held_seg[j] = 1
                j += 1
            i = j
        else:
            i += 1
    out[np.where(m)[0]] = held_seg
    return out


def run_cell(family, tf, threshold, confirm, slice_days=7, lookback_days=7):
    bpd = BARS_PER_DAY[tf]
    slice_bars = slice_days * bpd
    lookback_bars = lookback_days * bpd
    btc_regime = _btc_regime(OOS_WALL_MS)
    oos_start_ts = pd.Timestamp(OOS_START)

    close_dict, held_dict = {}, {}
    for sym in _u100_symbols():
        rec = _load_asset(sym, tf, OOS_WALL_MS)
        if rec is None:
            continue
        dates, o, h, l, c = rec
        frac = ensemble_votes(family, np.asarray(c, dtype=float))
        raw = (frac > threshold).astype(np.int8)
        held = apply_confirmation(c, raw, lookback_bars, slice_bars) if confirm else raw
        close_dict[sym] = pd.Series(c, index=dates)
        held_dict[sym] = pd.Series(held, index=dates)

    if len(close_dict) < 10:
        return None
    all_dates = sorted(set().union(*[s.index.tolist() for s in close_dict.values()]))
    oos = [d for d in all_dates if d >= oos_start_ts]
    if len(oos) < 10:
        return None

    eng, al = [], []
    for i in range(1, len(oos)):
        dp, dc = oos[i-1], oos[i]
        er, ar = [], []
        for sym in close_dict:
            if dp in close_dict[sym].index and dc in close_dict[sym].index:
                cp = close_dict[sym][dp]; cc = close_dict[sym][dc]
                if cp > 0:
                    r = (cc/cp) - 1.0
                    ar.append(r)
                    if dp in held_dict[sym].index and held_dict[sym][dp] == 1:
                        er.append(r)
        e = float(np.mean(er)) if er else 0.0
        # regime gate
        if btc_regime is not None:
            d1 = pd.Timestamp(dp).normalize()
            prior = btc_regime.index[btc_regime.index <= d1]
            if len(prior) and int(btc_regime[prior[-1]]) == 0:
                e = 0.0
        eng.append(e)
        al.append(float(np.mean(ar)) if ar else 0.0)

    eng = np.array(eng); al = np.array(al)
    linwk_e = float(np.mean(eng)) * bpd * 7 * 100
    linwk_a = float(np.mean(al)) * bpd * 7 * 100
    deploy = float(np.mean(eng != 0))
    return {
        "family": family, "tf": tf, "threshold": threshold, "confirm": confirm,
        "eng_linwk": round(linwk_e, 3), "al_linwk": round(linwk_a, 3),
        "edge_pp": round(linwk_e - linwk_a, 3), "deploy": round(deploy, 3),
    }


def main():
    print("=" * 92)
    print("STATIC EXHAUSTION TEST — sweeping the levers non-MA families never got")
    print("=" * 92)
    print("Levers: vote threshold {0.5,0.6,0.7,0.8} x confirmation {off,on} (MA-style)")
    print("A cell 'works' if: SELECTIVE (deploy<70%) AND beats always-long OOS (edge>0).")
    print()

    thresholds = [0.5, 0.6, 0.7, 0.8]
    tfs = ["4h", "1h", "1d"]
    all_rows = []
    winners = []

    for family in FAM:
        print(f"\n{'='*92}\n  {family}\n{'='*92}")
        print(f"  {'TF':<5} {'thr':>5} {'conf':>5} {'eng/wk':>9} {'AL/wk':>9} {'edge':>8} {'deploy':>8}  flag")
        print(f"  {'-'*72}")
        for tf in tfs:
            for thr in thresholds:
                for confirm in [False, True]:
                    r = run_cell(family, tf, thr, confirm)
                    if r is None:
                        continue
                    all_rows.append(r)
                    selective = r["deploy"] < 0.70
                    beats = r["edge_pp"] > 0
                    flag = ""
                    if selective and beats:
                        flag = "*** WORKS"
                        winners.append(r)
                    elif beats:
                        flag = "beats(not sel)"
                    elif selective:
                        flag = "sel(not beat)"
                    print(f"  {tf:<5} {thr:>5.1f} {str(confirm):>5} {r['eng_linwk']:>+8.3f}% "
                          f"{r['al_linwk']:>+8.3f}% {r['edge_pp']:>+7.3f} {r['deploy']:>7.1%}  {flag}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(OUTDIR / f"static_exhaustion_{ts}.json", "w") as f:
        json.dump({"rows": all_rows, "winners": winners, "timestamp": ts}, f, indent=2)

    print(f"\n{'='*92}")
    print(f"  SELECTIVE + BEATS-AL cells (deploy<70% AND edge>0): {len(winners)}")
    print(f"{'='*92}")
    if winners:
        for w in sorted(winners, key=lambda x: -x["edge_pp"]):
            print(f"  {w['family']}|{w['tf']} thr{w['threshold']} conf{w['confirm']}: "
                  f"edge {w['edge_pp']:+.3f}pp, deploy {w['deploy']:.1%}")
    else:
        print("  NONE. Pulling the selectivity + confirmation levers did NOT rescue any")
        print("  non-MA family. Static is EXHAUSTED for non-MA TIs at this resolution.")
    print(f"\n  Saved: static_exhaustion_{ts}.json")


if __name__ == "__main__":
    main()
