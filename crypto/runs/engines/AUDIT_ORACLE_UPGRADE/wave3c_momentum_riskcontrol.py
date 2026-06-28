"""WAVE 3C (Option A) -- can we KEEP momentum's selection edge while CUTTING its -52% drawdown?

Momentum top-K has the OOS edge but its de-risked (drawdown-controlled) form reverted to ~EW.
Its failure was concentration drawdown. Attack it directly with PRINCIPLED, low-parameter risk
control on the SAME momentum-top-K selection:
  - EW within top-K (baseline edge)
  - inv-vol within top-K (risk-balance the concentrated names)
  - + vol-target: scale gross exposure by min(1, target_vol / trailing_book_vol) [causal, lag-1]
Win condition (overfit-guarded): a variant must beat EW-all on RETURN-PER-DRAWDOWN on BOTH VAL
AND OOS, with materially shallower maxDD than the -52% top-K, above the random-K null.

de-leaked, PROD cost, 2025 SEALED, long-only spot lev=1.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(ROOT))

import strat.fleet_lab as fl
from strat.ma_wealth_driver import _load_asset, _u100_symbols
from strat.ma_slice_engine import held_rolling, _cached_band_held, causal_regime
from strat.ma_slice_engine import OOS_WALL_MS, OOS_START, OOS_2024_END, VAL_WALL_MS, VAL_START, VAL_END
from strat.canonical_exits import COST_PROD

BARS_PER_DAY = fl.BARS_PER_DAY
TYPES = ["SMA", "EMA", "WMA", "VIDYA", "KAMA"]
WIN = {"OOS": (OOS_WALL_MS, OOS_START, OOS_2024_END), "VAL": (VAL_WALL_MS, VAL_START, VAL_END)}
LB_D, SL_D = 14, 14


def frames(window, tf):
    wall, ws, we = WIN[window]; bpd = BARS_PER_DAY[tf]
    lb, sl = LB_D*bpd, SL_D*bpd
    reg = causal_regime(wall)
    closes = {}
    for sym in _u100_symbols():
        rec = _load_asset(sym, tf, wall)
        if rec is None: continue
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
            bh = _cached_band_held(ma, tf, s, o, h, l, c, wall)
            hc[s] = pd.Series(held_rolling(c, bh, lb, sl), index=d)
        conv += pd.DataFrame(hc).reindex(idx).fillna(0).reindex(columns=close_df.columns).fillna(0).to_numpy()
    ridx = reg.index; rg = np.ones(T)
    for k, dt in enumerate(idx):
        d1 = pd.Timestamp(dt).normalize(); prior = ridx[ridx <= d1]
        rg[k] = int(reg[prior[-1]]) if len(prior) else 0
    eligible = (conv > 0) & (rg[:, None] > 0)
    ret = close_df.pct_change().fillna(0.0).to_numpy()
    mom = close_df.pct_change(lb).to_numpy()
    vol = close_df.pct_change().rolling(max(20, bpd*5), min_periods=10).std().to_numpy()
    mask = (idx >= pd.Timestamp(ws)) & (idx < pd.Timestamp(we))
    return {"ret": ret, "eligible": eligible, "mom": mom, "vol": vol, "mask": mask, "bpd": bpd, "A": A}


def net_from_W(W, ret, cost=COST_PROD):
    Wl = np.vstack([np.zeros((1, W.shape[1])), W[:-1]])
    gr = np.nansum(Wl*ret, axis=1)
    turn = np.nansum(np.abs(W - np.vstack([np.zeros((1, W.shape[1])), W[:-1]])), axis=1)
    return gr - turn*(cost/2)


def vol_target_scale(W, ret, bpd, target_ann=0.60, lbk=None):
    """Scale each bar's gross by min(1, target/trailing_book_vol). Causal: uses book vol up to t-1."""
    lbk = lbk or max(20, bpd*10)
    # unscaled book gross return per bar (lag-1)
    Wl = np.vstack([np.zeros((1, W.shape[1])), W[:-1]])
    g = np.nansum(Wl*ret, axis=1)
    s = pd.Series(g)
    rv = s.rolling(lbk, min_periods=max(10, lbk//3)).std().shift(1).to_numpy() * np.sqrt(bpd*365)
    scale = np.where(np.isfinite(rv) & (rv > 1e-9), np.minimum(1.0, target_ann/rv), 1.0)
    return W * scale[:, None]


def topk_weights(score, elig, K, vol=None, mode="ew"):
    T, A = score.shape; W = np.zeros((T, A))
    s = np.where(elig & np.isfinite(score), score, -np.inf)
    for t in range(T):
        e = np.where(np.isfinite(s[t]))[0]
        if len(e) == 0: continue
        k = min(K, len(e)); top = e[np.argsort(-s[t][e])[:k]]
        if mode == "ew" or vol is None:
            W[t, top] = 1.0/k
        else:  # inv-vol within top-K
            v = vol[t][top]; iv = np.where(np.isfinite(v) & (v > 0), 1.0/v, 0.0)
            W[t, top] = iv/iv.sum() if iv.sum() > 0 else 1.0/k
    return W


def randk_w(elig, K, seed):
    T, A = elig.shape; W = np.zeros((T, A)); rng = np.random.default_rng(seed)
    for t in range(T):
        e = np.where(elig[t])[0]
        if len(e) == 0: continue
        k = min(K, len(e)); W[t, rng.choice(e, k, replace=False)] = 1.0/k
    return W


def ew_all(elig):
    c = elig.sum(axis=1, keepdims=True)
    return np.where(elig & (c > 0), 1.0/np.maximum(c, 1), 0.0)


def metrics(n, bpd, mask):
    n = n[mask]
    if len(n) == 0: return None
    lin = float(np.mean(n))*bpd*7*100
    cum = np.cumprod(1+n)
    dd = float(np.min((cum-np.maximum.accumulate(cum))/np.maximum.accumulate(cum)))*100 if len(cum)>1 else 0
    sh = (float(np.mean(n))*bpd*365)/(float(np.std(n))*np.sqrt(bpd*365)) if np.std(n)>1e-12 else 0
    return {"lin_wk": round(lin,3), "sharpe": round(sh,3), "maxdd": round(dd,1),
            "ret_per_dd": round(lin/abs(dd),4) if abs(dd)>1e-9 else 0}


def main():
    K = 5; N_NULL = 100
    out = {"check": "WAVE3C_momentum_risk_control", "ts": datetime.now().strftime("%Y%m%d_%H%M%S"), "K": K, "tf": {}}
    for tf in ["1d", "4h"]:
        out["tf"][tf] = {}
        fV, fO = frames("VAL", tf), frames("OOS", tf)
        # null + baseline per window
        bw = {}
        for tag, f in (("VAL", fV), ("OOS", fO)):
            ewm = metrics(net_from_W(ew_all(f["eligible"]), f["ret"]), f["bpd"], f["mask"])
            nn = [metrics(net_from_W(randk_w(f["eligible"], K, s), f["ret"]), f["bpd"], f["mask"])["lin_wk"] for s in range(N_NULL)]
            bw[tag] = {"EW_all": ewm, "rand_p95": round(float(np.percentile(nn,95)),3)}
        variants = {}
        for tag, f in (("VAL", fV), ("OOS", fO)):
            W_ew  = topk_weights(f["mom"], f["eligible"], K, mode="ew")
            W_iv  = topk_weights(f["mom"], f["eligible"], K, vol=f["vol"], mode="iv")
            W_ewv = vol_target_scale(W_ew, f["ret"], f["bpd"], 0.60)
            W_ivv = vol_target_scale(W_iv, f["ret"], f["bpd"], 0.60)
            variants[tag] = {
                "mom top5 EW":            metrics(net_from_W(W_ew,  f["ret"]), f["bpd"], f["mask"]),
                "mom top5 inv-vol":       metrics(net_from_W(W_iv,  f["ret"]), f["bpd"], f["mask"]),
                "mom top5 EW + voltgt":   metrics(net_from_W(W_ewv, f["ret"]), f["bpd"], f["mask"]),
                "mom top5 invvol+voltgt": metrics(net_from_W(W_ivv, f["ret"]), f["bpd"], f["mask"]),
            }
        out["tf"][tf] = {"baseline": bw, "variants": variants}
        print(f"\n{'='*104}\n{tf}  momentum top-{K} RISK CONTROL (de-leaked, PROD) | "
              f"EW-all VAL {bw['VAL']['EW_all']['lin_wk']:+.2f}/r{bw['VAL']['EW_all']['ret_per_dd']:.3f}  "
              f"OOS {bw['OOS']['EW_all']['lin_wk']:+.2f}/r{bw['OOS']['EW_all']['ret_per_dd']:.3f}")
        print(f"{'='*104}")
        print(f"  {'variant':<24}{'VAL_lin':>8}{'VAL_Sh':>7}{'VAL_DD':>7}{'VAL_r/DD':>9} | "
              f"{'OOS_lin':>8}{'OOS_Sh':>7}{'OOS_DD':>7}{'OOS_r/DD':>9} | {'both>EW r/DD':>12}")
        ew_v, ew_o = bw["VAL"]["EW_all"]["ret_per_dd"], bw["OOS"]["EW_all"]["ret_per_dd"]
        for name in variants["VAL"]:
            mv, mo = variants["VAL"][name], variants["OOS"][name]
            both = mv["ret_per_dd"] > ew_v and mo["ret_per_dd"] > ew_o
            print(f"  {name:<24}{mv['lin_wk']:>+8.2f}{mv['sharpe']:>+7.2f}{mv['maxdd']:>+7.0f}{mv['ret_per_dd']:>+9.3f} | "
                  f"{mo['lin_wk']:>+8.2f}{mo['sharpe']:>+7.2f}{mo['maxdd']:>+7.0f}{mo['ret_per_dd']:>+9.3f} | {'YES' if both else 'no':>12}")
    with open(HERE / "wave3c_momentum_riskcontrol.json", "w") as fo:
        json.dump(out, fo, indent=2, default=str)
    print(f"\nSaved {HERE/'wave3c_momentum_riskcontrol.json'}")


if __name__ == "__main__":
    main()
