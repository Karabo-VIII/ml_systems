"""PER-SLICE ROI vs the NORTH STAR mandate (1d / 3d / 7d window profitability).

CHARTER North Star: "PROFITABLE within the smallest time window without decay" —
primary = every 7-day window profitable (per-slice ROI > 0); stretch = 3d and 1d windows.
The metric is CONSISTENCY (fraction of windows > 0), NOT weekly magnitude.

This measures, for the deployable cross-type convergence book (V12, the SOTA landing),
the fraction of NON-OVERLAPPING 1d / 3d / 7d slices that are profitable, on OOS-2024.
Reports the distribution (pctile) per slice length so we see how far from "any window
profitable" we are. 2025 sealed.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
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
from strat.ma_slice_engine import held_rolling, _cached_band_held, btc_regime, OOS_WALL_MS, OOS_START, OOS_2024_END
from strat.canonical_exits import COST_PROD

BARS_PER_DAY = fl.BARS_PER_DAY
OUTDIR = CRYPTO / "runs" / "engines" / "V14_slice_roi"
TIER_CFG = {"1d": {"lookback": 14, "slice": 14}, "4h": {"lookback": 14, "slice": 7}}
TYPES = ["SMA", "EMA", "WMA", "VIDYA", "KAMA"]


def build_book_net(tf, cap=0.15):
    """Cross-type convergence-weighted book (V12 SOTA) -> per-bar net return series + dates."""
    cfg = TIER_CFG[tf]; bpd = BARS_PER_DAY[tf]
    lb, sl = cfg["lookback"]*bpd, cfg["slice"]*bpd
    reg = btc_regime(OOS_WALL_MS)
    closes = {}
    for sym in _u100_symbols():
        rec = _load_asset(sym, tf, OOS_WALL_MS)
        if rec is None:
            continue
        d, o, h, l, c = rec
        closes[sym] = (d, o, h, l, np.asarray(c, float))
    syms = list(closes.keys())
    close_df = pd.DataFrame({s: pd.Series(closes[s][4], index=closes[s][0]) for s in syms}).sort_index()
    idx = close_df.index; T = len(idx); A = len(syms)
    conv = np.zeros((T, A))
    for ma in TYPES:
        hc = {}
        for s in syms:
            d, o, h, l, c = closes[s]
            bh = _cached_band_held(ma, tf, s, o, h, l, c, OOS_WALL_MS)
            hc[s] = pd.Series(held_rolling(c, bh, lb, sl), index=d)
        hd = pd.DataFrame(hc).reindex(idx).fillna(0).reindex(columns=close_df.columns).fillna(0).to_numpy()
        conv += hd
    ret = close_df.pct_change().fillna(0.0).to_numpy()
    vol = close_df.pct_change().rolling(max(20, bpd*5), min_periods=10).std().to_numpy()
    rg = np.ones(T); ridx = reg.index
    for k, dt in enumerate(idx):
        d1 = pd.Timestamp(dt).normalize(); prior = ridx[ridx <= d1]
        rg[k] = int(reg[prior[-1]]) if len(prior) else 0
    mult = (conv / len(TYPES)) * rg[:, None]
    iv = np.where((mult > 0) & np.isfinite(vol) & (vol > 0), 1.0/np.where(vol > 0, vol, np.nan), 0.0)
    iv = np.nan_to_num(iv) * mult
    s = iv.sum(axis=1, keepdims=True)
    w = np.where(s > 0, iv/np.maximum(s, 1e-12), 0.0); w = np.minimum(w, cap)
    Wl = np.vstack([np.zeros((1, A)), w[:-1]])
    gr = np.nansum(Wl*ret, axis=1)
    turn = np.nansum(np.abs(w - np.vstack([np.zeros((1, A)), w[:-1]])), axis=1)
    net = gr - turn*(COST_PROD/2)
    mask = (idx >= pd.Timestamp(OOS_START)) & (idx < pd.Timestamp(OOS_2024_END))
    return net[mask], idx[mask], bpd


def slice_roi(net, bpd, slice_days):
    sb = int(slice_days * bpd)
    if sb < 1:
        sb = 1
    out = []
    i = 0
    while i + sb <= len(net):
        out.append(float(np.prod(1 + net[i:i+sb]) - 1) * 100)
        i += sb
    return np.array(out)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="1d")
    a = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    net, dates, bpd = build_book_net(a.tf)
    print(f"PER-SLICE ROI vs NORTH STAR — cross-type conv book, {a.tf}, OOS-2024 (prod, 15% cap)")
    print(f"Mandate: 'profitable within ANY 7-day window' (per-slice ROI>0). Stretch 3d, 1d.")
    print("=" * 84)
    print(f"  {'slice':<8}{'n':>5}{'% profitable':>14}{'median':>9}{'mean':>8}{'p10':>8}{'worst':>8}")
    print("  " + "-" * 70)
    rows = {}
    for name, days in [("1d", 1), ("3d", 3), ("7d", 7)]:
        sl = slice_roi(net, bpd, days)
        if len(sl) == 0:
            continue
        pct = float(np.mean(sl > 0))
        rows[name] = {"n": len(sl), "pct_profitable": round(pct, 3),
                      "median": round(float(np.median(sl)), 2), "mean": round(float(np.mean(sl)), 2),
                      "p10": round(float(np.percentile(sl, 10)), 2), "worst": round(float(np.min(sl)), 2)}
        print(f"  {name:<8}{len(sl):>5}{pct:>13.0%}{np.median(sl):>+8.2f}%{np.mean(sl):>+7.2f}%"
              f"{np.percentile(sl,10):>+7.2f}%{np.min(sl):>+7.2f}%")
    print("=" * 84)
    print("  North Star primary = 7d %profitable -> 100%. We are at the printed value.")
    with open(OUTDIR / f"slice_roi_{a.tf}_{ts}.json", "w") as f:
        json.dump({"tf": a.tf, "rows": rows, "ts": ts}, f, indent=2, default=str)
    print(f"  Saved slice_roi_{a.tf}_{ts}.json")


if __name__ == "__main__":
    main()
