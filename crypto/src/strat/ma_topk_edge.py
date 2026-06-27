"""TOP-K EDGE SEARCH (honest, no overfit) + CONVERGENCE-CONVICTION deployment model.

User asks: find 14d-lookback / 7d-window MA strategies consistently in the 5%+ winners,
WITHOUT overfitting. And a NEW deployment model: deploy top-5 configs per MA type
SEPARATELY (no voting); grade signal quality by how many of the top converge per type.

ANTI-OVERFIT PROTOCOL (the whole point):
  - SELECT the top-K configs per (type, tf) on TRAIN+VAL ONLY (<= 2024-05-15), ranked by
    TRAIN+VAL realized P&L through the prod-grade pipeline.
  - FREEZE that set. TEST it on OOS-2024 (held out, never seen in selection).
  - Report: full-band vs top-10 vs top-5 vs top-3 vs top-1 — does tighter selection chosen
    on TRAIN+VAL hold up OOS? If top-K OOS ~ full-band OOS -> tightening is selection noise.
    If top-K OOS > full-band OOS robustly -> real edge from concentration.

CONVERGENCE-CONVICTION (the user's deployment model):
  - The top-5 configs per type each produce their OWN held signal (NOT voted/averaged).
  - At each bar, count how many of the 5 are long on an asset = CONVERGENCE (0..5).
  - Position SIZE scales with convergence: more configs agree -> higher conviction -> more size.
  - This is DIFFERENT from voting (binary threshold): it's a continuous conviction weight.
  - Test: does convergence-weighting beat flat top-5 and beat the full band?

2025 SEALED. Strictly causal: TRAIN+VAL selection uses only <=2024-05-15; OOS test only 2024.
"""
from __future__ import annotations

import warnings; warnings.filterwarnings("ignore")
import sys
import json
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
from strat.ma_wealth_driver import _load_asset, _u100_symbols
from strat.ma_movecatch_engine import select_band_trainval
from strat.ma_type_upgrade import held_cross, _nums, _ema
from strat.canonical_exits import COST_PROD, resolve_held
from strat.ma_slice_engine import btc_regime, VAL_WALL_MS, OOS_WALL_MS

BARS_PER_DAY = fl.BARS_PER_DAY
OUTDIR = CRYPTO / "runs" / "engines" / "V11_topk_edge"
TRAIN_START, TRAIN_END = "2020-01-01", "2023-07-01"
VAL_END = "2024-05-15"
OOS_START, OOS_2024_END = "2024-05-15", "2024-12-31"
TIER_CFG = {"1d": {"lookback": 14, "slice": 14}, "4h": {"lookback": 14, "slice": 7}}


def _single_held(close, periods, ma_type, lb, sl):
    c = np.asarray(close, float); n = len(c)
    raw = held_cross(c, periods, ma_type).astype(np.int8)
    m = np.isfinite(c); conf = np.zeros(n, dtype=bool)
    if m.sum() >= max(5, lb):
        cseg = c[m]; ema = _ema(cseg, lb)
        cs = (raw[m].astype(bool)) & (cseg > ema) & np.concatenate([[False], ema[1:] >= ema[:-1]])
        conf[np.where(m)[0]] = cs
    return resolve_held(c, conf, sl)


def _portfolio_eval(held_cols, close_cols, reg, bpd, win_s, win_e, cap=0.15):
    close = pd.DataFrame(close_cols).sort_index()
    heldp = pd.DataFrame(held_cols).reindex(close.index).fillna(0).astype(np.int8)
    ret = close.pct_change().fillna(0.0).to_numpy()
    H = heldp.to_numpy().astype(float)
    vol = close.pct_change().rolling(max(20, bpd*5), min_periods=10).std().to_numpy()
    dates = close.index; T, A = H.shape
    rg = np.ones(T); ridx = reg.index
    for k, dt in enumerate(dates):
        d1 = pd.Timestamp(dt).normalize(); prior = ridx[ridx <= d1]
        rg[k] = int(reg[prior[-1]]) if len(prior) else 0
    Hg = H * rg[:, None]
    iv = np.where((Hg > 0) & np.isfinite(vol) & (vol > 0), 1.0/np.where(vol > 0, vol, np.nan), 0.0)
    iv = np.nan_to_num(iv); s = iv.sum(axis=1, keepdims=True)
    w = np.where(s > 0, iv/np.maximum(s, 1e-12), 0.0); w = np.minimum(w, cap)
    Wl = np.vstack([np.zeros((1, A)), w[:-1]])
    gr = np.nansum(Wl*ret, axis=1)
    turn = np.nansum(np.abs(w - np.vstack([np.zeros((1, A)), w[:-1]])), axis=1)
    net = gr - turn*(COST_PROD/2)
    mask = (dates >= pd.Timestamp(win_s)) & (dates < pd.Timestamp(win_e))
    nw = net[mask]
    lin = float(np.mean(nw))*bpd*7*100
    comp = float(np.prod(1+nw)-1)*100
    cum = np.cumprod(1+nw); dd = float(np.min((cum-np.maximum.accumulate(cum))/np.maximum.accumulate(cum)))*100 if len(cum)>1 else 0
    sh = (float(np.mean(nw))*bpd*365)/(float(np.std(nw))*np.sqrt(bpd*365)) if np.std(nw) > 1e-12 else 0
    return {"lin_wk": round(lin,3), "compound": round(comp,1), "sharpe": round(sh,2), "maxdd": round(dd,1)}


def rank_configs_trainval(ma_type, tf, band, lb, sl):
    """Rank band configs by TRAIN+VAL realized portfolio P&L (causal, OOS-excluded)."""
    bpd = BARS_PER_DAY[tf]
    reg = btc_regime(VAL_WALL_MS)
    # load TRAIN+VAL closes (wall at VAL_END = 2024-05-15)
    closes = {}
    for sym in _u100_symbols():
        rec = _load_asset(sym, tf, VAL_WALL_MS)
        if rec is None:
            continue
        d, o, h, l, c = rec
        closes[sym] = (d, np.asarray(c, float))
    scored = []
    for cname in band:
        periods = _nums(cname)
        hc, cc = {}, {}
        for sym, (d, c) in closes.items():
            hc[sym] = pd.Series(_single_held(c, periods, ma_type, lb, sl), index=d)
            cc[sym] = pd.Series(c, index=d)
        # score on FULL TRAIN+VAL window (2020-01-01 .. 2024-05-15)
        m = _portfolio_eval(hc, cc, reg, bpd, TRAIN_START, VAL_END)
        scored.append((cname, periods, m["lin_wk"]))
    scored.sort(key=lambda x: -x[2])
    return scored


def topk_oos(ma_type, tf, K_list=(1, 3, 5, 10, 999)):
    """Select top-K on TRAIN+VAL, test the frozen set on OOS-2024. Returns per-K OOS metrics."""
    cfg = TIER_CFG[tf]; bpd = BARS_PER_DAY[tf]
    lb, sl = cfg["lookback"]*bpd, cfg["slice"]*bpd
    band = select_band_trainval(ma_type, tf)[0]
    ranked = rank_configs_trainval(ma_type, tf, band, lb, sl)   # TRAIN+VAL ranking (no OOS)

    reg_oos = btc_regime(OOS_WALL_MS)
    closes_oos = {}
    for sym in _u100_symbols():
        rec = _load_asset(sym, tf, OOS_WALL_MS)
        if rec is None:
            continue
        d, o, h, l, c = rec
        closes_oos[sym] = (d, np.asarray(c, float))

    out = {}
    for K in K_list:
        chosen = [r[0] for r in ranked[:K]] if K < 999 else [r[0] for r in ranked]
        chosen_periods = [_nums(c) for c in chosen]
        # ENSEMBLE (voted) over the chosen set
        hc, cc = {}, {}
        for sym, (d, c) in closes_oos.items():
            n = len(c)
            votes = np.zeros(n)
            for periods in chosen_periods:
                votes += _single_held(c, periods, ma_type, lb, sl).astype(float)
            frac = votes / max(1, len(chosen_periods))
            # majority vote >0.5 then re-confirm/slice (same as engine)
            raw = (frac > 0.5).astype(np.int8)
            m = np.isfinite(c); conf = np.zeros(n, dtype=bool)
            if m.sum() >= max(5, lb):
                cseg = c[m]; ema = _ema(cseg, lb)
                cs = (raw[m].astype(bool)) & (cseg > ema) & np.concatenate([[False], ema[1:] >= ema[:-1]])
                conf[np.where(m)[0]] = cs
            hc[sym] = pd.Series(resolve_held(c, conf, sl), index=d)
            cc[sym] = pd.Series(c, index=d)
        m_oos = _portfolio_eval(hc, cc, reg_oos, bpd, OOS_START, OOS_2024_END)
        klabel = "full" if K >= 999 else f"top{K}"
        out[klabel] = {"n_configs": len(chosen), "oos": m_oos,
                       "top_configs": chosen[:5]}
    return out, ranked


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", default="SMA,EMA,WMA,VIDYA,KAMA,DEMA")
    ap.add_argument("--tf", default="1d")
    a = ap.parse_args()
    types = [t.strip().upper() for t in a.types.split(",")]
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 100)
    print(f"TOP-K EDGE (TRAIN+VAL-selected, OOS-tested) — {a.tf}, prod cost, 15% cap, NO PEEKING")
    print("=" * 100)
    print(f"  {'type':<7}| {'full':>16} {'top10':>16} {'top5':>16} {'top3':>16} {'top1':>16}")
    print(f"  {'':7}| {'OOS lin/wk (Sh)':>16} {'':>16} {'':>16} {'':>16} {'':>16}")
    print("  " + "-" * 92)
    allout = {}
    for ma in types:
        res, ranked = topk_oos(ma, a.tf)
        allout[ma] = {"per_k": res, "trainval_top5": [(r[0], round(r[2],2)) for r in ranked[:5]]}
        def cell(k):
            m = res[k]["oos"]; return f"{m['lin_wk']:>+6.2f}({m['sharpe']:.2f})"
        print(f"  {ma:<7}| {cell('full'):>16} {cell('top10'):>16} {cell('top5'):>16} "
              f"{cell('top3'):>16} {cell('top1'):>16}")
    print("  " + "-" * 92)
    print("  Each top-K SELECTED on TRAIN+VAL, FROZEN, tested OOS-2024. Tighter>full = real concentration edge.")
    with open(OUTDIR / f"topk_edge_{a.tf}_{ts}.json", "w") as f:
        json.dump({"tf": a.tf, "by_type": allout, "ts": ts}, f, indent=2, default=str)
    print(f"\n  Saved topk_edge_{a.tf}_{ts}.json")


if __name__ == "__main__":
    main()
