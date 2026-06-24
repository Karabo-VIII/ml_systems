"""Static closeout — overfit probe + ensemble combination for the validated engines.

Closes the static phase before any Phase 2 work, per the user's gate.

PART A — OVERFIT PROBE (lookback sensitivity):
  A validated edge must be ROBUST to its hyperparameters. A spike at one lookback value
  = overfit; a smooth plateau = real. Bollinger|1d uses the confirmation layer (which
  depends on lookback), so sweep lookback {3,5,7,10,14,21} and check the OOS-edge surface
  is smooth. RSI|4h uses NO confirmation (lookback-independent), so its overfit probe is
  the vote-threshold robustness already shown (beats AL at all 4 thresholds) — noted.

PART B — ENSEMBLE COMBINATION:
  Combine the validated engines (MA WMA|4h + RSI|4h + Bollinger|1d) into one portfolio.
  Each engine produces a daily-equivalent regime-gated return stream over OOS; the
  combined book is the equal-weight-of-engines average. Diversification across orthogonal
  signals should lower vol (smoother equity) even if mean is similar. Report combined
  linear/wk, MaxDD, and edge vs always-long + vs each engine alone.
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
from strat.ma_movecatch_engine import select_band_trainval, make_hold_provider
from strat.ti_engine_multitf import _decoupled_held
from strat.ti_engine_sota import rsi_single_held, bollinger_single_held, RSI_CONFIGS, BOLLINGER_CONFIGS
from strat.ti_static_exhaustion import ensemble_votes, apply_confirmation, _btc_regime

BARS_PER_DAY = fl.BARS_PER_DAY
OOS_START = "2024-05-15"; OOS_END = "2025-01-01"
OOS_WALL_MS = int(pd.Timestamp(OOS_END).value // 10**6)
OUTDIR = CRYPTO / "runs" / "engines" / "V3_ti_families" / "static_closeout"


def _gated_daily_stream(close_dict, held_dict, btc_regime, tf):
    """Return a date-indexed pd.Series of DAILY-equivalent regime-gated EW engine returns
    and the always-long stream, aligned to calendar days (so engines at different TFs can
    be combined). Aggregates intra-day bars by compounding within each calendar day."""
    bpd = BARS_PER_DAY[tf]
    all_dates = sorted(set().union(*[s.index.tolist() for s in close_dict.values()]))
    ws = pd.Timestamp(OOS_START); we = pd.Timestamp(OOS_END)
    win = [d for d in all_dates if ws <= d < we]
    eng_bar, al_bar, bar_day = [], [], []
    for i in range(1, len(win)):
        dp, dc = win[i-1], win[i]
        er, ar = [], []
        for sym in close_dict:
            if dp in close_dict[sym].index and dc in close_dict[sym].index:
                cp = close_dict[sym][dp]; cc = close_dict[sym][dc]
                if cp > 0:
                    r = (cc/cp) - 1.0; ar.append(r)
                    if dp in held_dict[sym].index and held_dict[sym][dp] == 1:
                        er.append(r)
        e = float(np.mean(er)) if er else 0.0
        if btc_regime is not None:
            d1 = pd.Timestamp(dp).normalize()
            prior = btc_regime.index[btc_regime.index <= d1]
            if len(prior) and int(btc_regime[prior[-1]]) == 0:
                e = 0.0
        eng_bar.append(e); al_bar.append(float(np.mean(ar)) if ar else 0.0)
        bar_day.append(pd.Timestamp(dc).normalize())
    eng_s = pd.Series(eng_bar, index=bar_day)
    al_s = pd.Series(al_bar, index=bar_day)
    # compound bars within each calendar day -> daily return
    eng_d = eng_s.groupby(level=0).apply(lambda x: np.prod(1 + x) - 1)
    al_d = al_s.groupby(level=0).apply(lambda x: np.prod(1 + x) - 1)
    return eng_d, al_d


def build_rsi(tf="4h", thr=0.7):
    cd, hd = {}, {}
    for sym in _u100_symbols():
        rec = _load_asset(sym, tf, OOS_WALL_MS)
        if rec is None:
            continue
        dates, o, h, l, c = rec
        frac = ensemble_votes("RSI", np.asarray(c, dtype=float))
        hd[sym] = pd.Series((frac > thr).astype(np.int8), index=dates)
        cd[sym] = pd.Series(c, index=dates)
    return cd, hd


def build_bollinger(tf="1d", thr=0.6, lookback_days=7, confirm=True):
    cd, hd = {}, {}
    bpd = BARS_PER_DAY[tf]
    for sym in _u100_symbols():
        rec = _load_asset(sym, tf, OOS_WALL_MS)
        if rec is None:
            continue
        dates, o, h, l, c = rec
        frac = ensemble_votes("BOLLINGER", np.asarray(c, dtype=float))
        raw = (frac > thr).astype(np.int8)
        held = apply_confirmation(c, raw, lookback_days * bpd, 7 * bpd) if confirm else raw
        hd[sym] = pd.Series(held, index=dates)
        cd[sym] = pd.Series(c, index=dates)
    return cd, hd


def build_ma(ma_type="WMA", tf="4h", lookback_days=7):
    bpd = BARS_PER_DAY[tf]
    sub_band, _ = select_band_trainval(ma_type, tf)
    provider = make_hold_provider(sub_band, ma_type)
    cd, hd = {}, {}
    for sym in _u100_symbols():
        rec = _load_asset(sym, tf, OOS_WALL_MS)
        if rec is None:
            continue
        dates, o, h, l, c = rec
        band_held = provider(sym, o, h, l, c)
        held = _decoupled_held(c, band_held, lookback_days * bpd, 7 * bpd)
        hd[sym] = pd.Series(held, index=dates)
        cd[sym] = pd.Series(c, index=dates)
    return cd, hd


def stats(daily, al_daily, label):
    wk = float(daily.mean()) * 7 * 100
    al_wk = float(al_daily.reindex(daily.index).fillna(0).mean()) * 7 * 100
    eq = np.cumprod(1 + daily.values)
    maxdd = float(np.min((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq))) * 100 if len(eq) > 1 else 0
    vol = float(daily.std()) * 100
    sharpe = float(daily.mean() / (daily.std() + 1e-12)) * np.sqrt(365)
    return {"label": label, "linwk": round(wk, 3), "al_linwk": round(al_wk, 3),
            "edge_pp": round(wk - al_wk, 3), "maxdd": round(maxdd, 1),
            "daily_vol_pct": round(vol, 3), "sharpe_ann": round(sharpe, 2)}


def part_a_overfit_probe():
    print("=" * 84)
    print("PART A — OVERFIT PROBE (lookback sensitivity on Bollinger|1d)")
    print("=" * 84)
    print("A real edge is a SMOOTH plateau across lookback; a spike = overfit.")
    btc = _btc_regime(OOS_WALL_MS)
    print(f"\n  {'lookback_d':>10} {'edge_pp':>9} {'linwk':>8}")
    print(f"  {'-'*32}")
    surface = []
    for lb in [3, 5, 7, 10, 14, 21]:
        cd, hd = build_bollinger(lookback_days=lb)
        eng_d, al_d = _gated_daily_stream(cd, hd, btc, "1d")
        s = stats(eng_d, al_d, f"BB|1d lb{lb}")
        surface.append({"lookback": lb, **s})
        print(f"  {lb:>10} {s['edge_pp']:>+8.3f} {s['linwk']:>+7.3f}%")
    edges = [x["edge_pp"] for x in surface]
    smooth = (min(edges) > 0) and (np.std(edges) < abs(np.mean(edges)))
    print(f"\n  All lookbacks positive: {min(edges) > 0} | edge std {np.std(edges):.3f} "
          f"vs mean {np.mean(edges):.3f}")
    print(f"  --> {'SMOOTH plateau (robust, NOT overfit)' if smooth else 'check: not smooth'}")
    print("\n  Note: RSI|4h uses NO confirmation layer (lookback-independent). Its overfit")
    print("  probe is the vote-threshold robustness already shown (beats AL at all 4 thr).")
    return surface


def part_b_ensemble():
    print("\n" + "=" * 84)
    print("PART B — ENSEMBLE COMBINATION (MA WMA|4h + RSI|4h + Bollinger|1d)")
    print("=" * 84)
    btc = _btc_regime(OOS_WALL_MS)

    print("  Building MA WMA|4h ...")
    ma_cd, ma_hd = build_ma()
    ma_d, ma_al = _gated_daily_stream(ma_cd, ma_hd, btc, "4h")
    print("  Building RSI|4h ...")
    rsi_cd, rsi_hd = build_rsi()
    rsi_d, rsi_al = _gated_daily_stream(rsi_cd, rsi_hd, btc, "4h")
    print("  Building Bollinger|1d ...")
    bb_cd, bb_hd = build_bollinger()
    bb_d, bb_al = _gated_daily_stream(bb_cd, bb_hd, btc, "1d")

    # align on common calendar days
    idx = ma_d.index.union(rsi_d.index).union(bb_d.index)
    ma_a = ma_d.reindex(idx).fillna(0.0)
    rsi_a = rsi_d.reindex(idx).fillna(0.0)
    bb_a = bb_d.reindex(idx).fillna(0.0)
    al_a = ma_al.reindex(idx).fillna(0.0)  # always-long reference (4h-derived)
    combined = (ma_a + rsi_a + bb_a) / 3.0

    rows = [
        stats(ma_d, ma_al, "MA WMA|4h (alone)"),
        stats(rsi_d, rsi_al, "RSI|4h (alone)"),
        stats(bb_d, bb_al, "Bollinger|1d (alone)"),
        stats(combined, al_a, "ENSEMBLE (EW of 3)"),
    ]
    print(f"\n  {'engine':<24} {'linwk':>8} {'edge':>8} {'MaxDD':>8} {'dVol':>7} {'Sharpe':>7}")
    print(f"  {'-'*66}")
    for r in rows:
        print(f"  {r['label']:<24} {r['linwk']:>+7.3f}% {r['edge_pp']:>+7.3f} "
              f"{r['maxdd']:>7.1f}% {r['daily_vol_pct']:>6.3f}% {r['sharpe_ann']:>6.2f}")

    # pairwise correlation of the engine return streams (diversification check)
    corr = pd.DataFrame({"MA": ma_a, "RSI": rsi_a, "BB": bb_a}).corr()
    print(f"\n  Engine return correlations (lower = better diversification):")
    print(corr.round(2).to_string().replace("\n", "\n  "))
    return rows, corr.round(3).to_dict()


def main():
    surface = part_a_overfit_probe()
    rows, corr = part_b_ensemble()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(OUTDIR / f"static_closeout_{ts}.json", "w") as f:
        json.dump({"overfit_probe": surface, "ensemble": rows, "engine_corr": corr,
                   "timestamp": ts}, f, indent=2)
    print(f"\n  Saved: static_closeout_{ts}.json")


if __name__ == "__main__":
    main()
