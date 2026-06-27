"""WITHIN-TYPE TOP-5 CONVERGENCE — the user's exact model.

Each MA type is an INDEPENDENT entity (SMA != EMA != WMA ...). Within EACH type:
  1. Select that type's TOP-5 configs on TRAIN+VAL (frozen, no peeking).
  2. Deploy the 5 SEPARATELY (no voting/averaging into one signal).
  3. Grade signal quality by how many of THAT TYPE'S top-5 converge (agree long) on an asset.
  4. Size position by within-type convergence (0..5).

Compares, PER TYPE, on OOS-2024:
  A) FULL BAND (the current SOTA per V11) — baseline
  B) top5 FLAT       — equal-weight any asset >=1 of the 5 is long
  C) top5 CONV-WEIGHT — size ~ (n_of_5_long / 5)
  D) top5 CONV-GATE>=3 — only enter when >=3 of the 5 agree (high conviction)
  E) top5 CONV-GATE>=4, >=5

ANTI-OVERFIT: the 5 configs are ranked on TRAIN+VAL (2020..2024-05-15) and FROZEN; all
convergence/sizing measured on OOS-2024 (held out). 2025 sealed. Strictly causal.
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
from strat.ma_movecatch_engine import select_band_trainval
from strat.ma_type_upgrade import _nums, _ema, held_cross
from strat.canonical_exits import COST_PROD, resolve_held
from strat.ma_slice_engine import btc_regime, VAL_WALL_MS, OOS_WALL_MS
from strat.ma_topk_edge import rank_configs_trainval, _single_held

BARS_PER_DAY = fl.BARS_PER_DAY
OUTDIR = CRYPTO / "runs" / "engines" / "V13_within_type_conv"
TIER_CFG = {"1d": {"lookback": 14, "slice": 14}, "4h": {"lookback": 14, "slice": 7}}
TYPES = ["SMA", "EMA", "WMA", "VIDYA", "KAMA", "DEMA"]


def eval_book(mult, ret, vol, idx, rg, bpd, cap=0.15):
    """mult [T x A] = conviction multiplier (0..1) per asset-bar (already regime-gated)."""
    T, A = mult.shape
    iv = np.where((mult > 0) & np.isfinite(vol) & (vol > 0), 1.0/np.where(vol > 0, vol, np.nan), 0.0)
    iv = np.nan_to_num(iv) * mult
    s = iv.sum(axis=1, keepdims=True)
    w = np.where(s > 0, iv/np.maximum(s, 1e-12), 0.0); w = np.minimum(w, cap)
    Wl = np.vstack([np.zeros((1, A)), w[:-1]])
    gr = np.nansum(Wl*ret, axis=1)
    turn = np.nansum(np.abs(w - np.vstack([np.zeros((1, A)), w[:-1]])), axis=1)
    net = gr - turn*(COST_PROD/2)
    from strat.ma_slice_engine import OOS_START, OOS_2024_END
    mask = (idx >= pd.Timestamp(OOS_START)) & (idx < pd.Timestamp(OOS_2024_END))
    nw = net[mask]
    lin = float(np.mean(nw))*bpd*7*100
    comp = float(np.prod(1+nw)-1)*100
    cum = np.cumprod(1+nw); dd = float(np.min((cum-np.maximum.accumulate(cum))/np.maximum.accumulate(cum)))*100 if len(cum)>1 else 0
    sh = (float(np.mean(nw))*bpd*365)/(float(np.std(nw))*np.sqrt(bpd*365)) if np.std(nw) > 1e-12 else 0
    return {"lin_wk": round(lin,3), "compound": round(comp,1), "sharpe": round(sh,2), "maxdd": round(dd,1)}


def run_type(ma_type, tf):
    cfg = TIER_CFG[tf]; bpd = BARS_PER_DAY[tf]
    lb, sl = cfg["lookback"]*bpd, cfg["slice"]*bpd
    band = select_band_trainval(ma_type, tf)[0]
    # TRAIN+VAL ranking (no OOS peeking)
    ranked = rank_configs_trainval(ma_type, tf, band, lb, sl)
    top5 = [r[0] for r in ranked[:5]]
    top5_periods = [_nums(c) for c in top5]
    full_periods = [_nums(c) for c in band]

    reg = btc_regime(OOS_WALL_MS)
    closes = {}
    for sym in _u100_symbols():
        rec = _load_asset(sym, tf, OOS_WALL_MS)
        if rec is None:
            continue
        d, o, h, l, c = rec
        closes[sym] = (d, np.asarray(c, float))
    syms = list(closes.keys())
    idx = pd.DataFrame({s: pd.Series(closes[s][1], index=closes[s][0]) for s in syms}).sort_index().index
    close_df = pd.DataFrame({s: pd.Series(closes[s][1], index=closes[s][0]) for s in syms}).sort_index()
    ret = close_df.pct_change().fillna(0.0).to_numpy()
    vol = close_df.pct_change().rolling(max(20, bpd*5), min_periods=10).std().to_numpy()
    T = len(idx); A = len(syms)
    rg = np.ones(T); ridx = reg.index
    for k, dt in enumerate(idx):
        d1 = pd.Timestamp(dt).normalize(); prior = ridx[ridx <= d1]
        rg[k] = int(reg[prior[-1]]) if len(prior) else 0
    rgcol = rg[:, None]

    # convergence count across the top-5 (each config's confirmed+slice held)
    conv = np.zeros((T, A))
    for periods in top5_periods:
        hd = pd.DataFrame({s: pd.Series(_single_held(closes[s][1], periods, ma_type, lb, sl), index=closes[s][0]) for s in syms})
        hd = hd.reindex(idx).fillna(0).reindex(columns=close_df.columns).fillna(0).to_numpy()
        conv += hd
    # full band held (voted majority) for baseline
    votes = np.zeros((T, A))
    for periods in full_periods:
        hd = pd.DataFrame({s: pd.Series(_single_held(closes[s][1], periods, ma_type, lb, sl), index=closes[s][0]) for s in syms})
        hd = hd.reindex(idx).fillna(0).reindex(columns=close_df.columns).fillna(0).to_numpy()
        votes += hd
    full_held = (votes / max(1, len(full_periods)) > 0.5).astype(float)

    out = {"top5_configs": top5}
    out["full_band"] = eval_book(full_held * rgcol, ret, vol, idx, rg, bpd)
    out["top5_flat"] = eval_book((conv >= 1).astype(float) * rgcol, ret, vol, idx, rg, bpd)
    out["top5_convweight"] = eval_book((conv / 5.0) * rgcol, ret, vol, idx, rg, bpd)
    out["top5_gate3"] = eval_book((conv >= 3).astype(float) * rgcol, ret, vol, idx, rg, bpd)
    out["top5_gate4"] = eval_book((conv >= 4).astype(float) * rgcol, ret, vol, idx, rg, bpd)
    out["top5_gate5"] = eval_book((conv >= 5).astype(float) * rgcol, ret, vol, idx, rg, bpd)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", default="SMA,EMA,WMA,VIDYA,KAMA,DEMA")
    ap.add_argument("--tf", default="1d")
    a = ap.parse_args()
    types = [t.strip().upper() for t in a.types.split(",")]
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"WITHIN-TYPE TOP-5 CONVERGENCE — {a.tf}, OOS-2024, prod, 15% cap "
          f"(top5 frozen on TRAIN+VAL, NO peeking)")
    print("=" * 100)
    print(f"  {'type':<7}| {'full band':>12} {'top5 flat':>12} {'convweight':>12} "
          f"{'gate>=3':>12} {'gate>=4':>12} {'gate>=5':>12}")
    print(f"  {'':7}| {'lin/wk Sh':>12}")
    print("  " + "-" * 90)
    allout = {}
    for ma in types:
        r = run_type(ma, a.tf)
        allout[ma] = r
        def c(k):
            m = r[k]; return f"{m['lin_wk']:>+5.2f}({m['sharpe']:.2f})"
        print(f"  {ma:<7}| {c('full_band'):>12} {c('top5_flat'):>12} {c('top5_convweight'):>12} "
              f"{c('top5_gate3'):>12} {c('top5_gate4'):>12} {c('top5_gate5'):>12}")
    print("  " + "-" * 90)
    print("  convweight/gate beating full_band = within-type top-5 convergence adds edge.")
    with open(OUTDIR / f"within_type_conv_{a.tf}_{ts}.json", "w") as f:
        json.dump({"tf": a.tf, "by_type": allout, "ts": ts}, f, indent=2, default=str)
    print(f"\n  Saved within_type_conv_{a.tf}_{ts}.json")


if __name__ == "__main__":
    main()
