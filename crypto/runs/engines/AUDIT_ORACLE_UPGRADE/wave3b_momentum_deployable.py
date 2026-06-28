"""WAVE 3B -- harden the momentum-selection finding + build the DEPLOYABLE (de-risked) form.

Wave 3 found cross-sectional MOMENTUM is the one causal selection score that beats random-K AND
EW-all on BOTH windows -- but hard top-5 draws down -52..-55% (too deep). This script:
  1. SIGNIFICANCE: momentum top-K vs a 200-draw random-selection null -> exact percentile (is the
     skill real, not 25-draw noise?). Both windows, 1d + 4h.
  2. DEPLOYABLE FORM: a SOFT momentum tilt (weight ~ cross-sectional momentum RANK among eligible,
     per-name cap) that keeps the selection skill but spreads risk -> controlled drawdown. Compare
     EW-all vs soft-tilt(cap33) vs top-quantile(top-third EW) on VAL+OOS.
  3. 4h replication of the top-K skill.

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
    mom = close_df.pct_change(lb).to_numpy()              # causal trailing return
    mask = (idx >= pd.Timestamp(ws)) & (idx < pd.Timestamp(we))
    return {"ret": ret, "eligible": eligible, "mom": mom, "mask": mask, "bpd": bpd, "A": A}


def net(W, ret, cost=COST_PROD):
    Wl = np.vstack([np.zeros((1, W.shape[1])), W[:-1]])
    gr = np.nansum(Wl*ret, axis=1)
    turn = np.nansum(np.abs(W - np.vstack([np.zeros((1, W.shape[1])), W[:-1]])), axis=1)
    return gr - turn*(cost/2)


def metrics(n, bpd, mask):
    n = n[mask]
    if len(n) == 0: return None
    lin = float(np.mean(n))*bpd*7*100
    cum = np.cumprod(1+n)
    dd = float(np.min((cum-np.maximum.accumulate(cum))/np.maximum.accumulate(cum)))*100 if len(cum)>1 else 0
    sh = (float(np.mean(n))*bpd*365)/(float(np.std(n))*np.sqrt(bpd*365)) if np.std(n)>1e-12 else 0
    return {"lin_wk": round(lin,3), "sharpe": round(sh,3), "maxdd": round(dd,1),
            "ret_per_dd": round(lin/abs(dd),4) if abs(dd)>1e-9 else 0}


def topk_w(score, elig, K):
    T, A = score.shape; W = np.zeros((T, A))
    s = np.where(elig & np.isfinite(score), score, -np.inf)
    for t in range(T):
        e = np.where(np.isfinite(s[t]))[0]
        if len(e) == 0: continue
        k = min(K, len(e)); top = e[np.argsort(-s[t][e])[:k]]; W[t, top] = 1.0/k
    return W


def randk_w(elig, K, seed):
    T, A = elig.shape; W = np.zeros((T, A)); rng = np.random.default_rng(seed)
    for t in range(T):
        e = np.where(elig[t])[0]
        if len(e) == 0: continue
        k = min(K, len(e)); W[t, rng.choice(e, k, replace=False)] = 1.0/k
    return W


def ew_w(elig):
    c = elig.sum(axis=1, keepdims=True)
    return np.where(elig & (c > 0), 1.0/np.maximum(c, 1), 0.0)


def softtilt_w(score, elig, cap, power=1.0):
    """Weight ~ cross-sectional momentum RANK among eligible (0..1)^power, capped, normalized."""
    T, A = score.shape; W = np.zeros((T, A))
    for t in range(T):
        e = np.where(elig[t] & np.isfinite(score[t]))[0]
        if len(e) == 0: continue
        sc = score[t][e]
        order = np.argsort(sc); ranks = np.empty(len(e)); ranks[order] = np.arange(1, len(e)+1)
        rw = (ranks/len(e))**power
        rw = rw/rw.sum()
        rw = np.minimum(rw, cap)
        if rw.sum() > 0: rw = rw/rw.sum()
        W[t, e] = rw
    return W


def topq_w(score, elig, q=0.34):
    """Top-q quantile by momentum among eligible, EW."""
    T, A = score.shape; W = np.zeros((T, A))
    for t in range(T):
        e = np.where(elig[t] & np.isfinite(score[t]))[0]
        if len(e) == 0: continue
        k = max(1, int(np.ceil(len(e)*q))); top = e[np.argsort(-score[t][e])[:k]]; W[t, top] = 1.0/k
    return W


def main():
    N_NULL = 200
    out = {"check": "WAVE3B_momentum_hardening_deployable", "ts": datetime.now().strftime("%Y%m%d_%H%M%S"),
           "significance": {}, "deployable": {}}

    # ---- 1+3: significance of momentum top-K vs 200-draw null, 1d + 4h ----
    print("="*96)
    print("SIGNIFICANCE: momentum top-K vs 200-draw random-selection null (de-leaked, PROD)")
    print("="*96)
    for tf in ["1d", "4h"]:
        out["significance"][tf] = {}
        for window in ["VAL", "OOS"]:
            f = frames(window, tf)
            print(f"\n  {tf}/{window}:  {'K':>3}{'mom_lin':>9}{'mom_Sh':>7}{'mom_DD':>7} | "
                  f"{'null_mean':>9}{'null_p95':>9}{'pctile':>7}")
            for K in [5, 10, 15]:
                mw = metrics(net(topk_w(f["mom"], f["eligible"], K), f["ret"]), f["bpd"], f["mask"])
                nulls = []
                for s in range(N_NULL):
                    m = metrics(net(randk_w(f["eligible"], K, s), f["ret"]), f["bpd"], f["mask"])
                    if m: nulls.append(m["lin_wk"])
                nulls = np.array(nulls)
                pct = float(np.mean(nulls < mw["lin_wk"]))*100
                out["significance"][tf][f"{window}_K{K}"] = {**mw, "null_mean": round(float(nulls.mean()),3),
                    "null_p95": round(float(np.percentile(nulls,95)),3), "percentile_vs_null": round(pct,1)}
                print(f"           {K:>3}{mw['lin_wk']:>+9.3f}{mw['sharpe']:>+7.2f}{mw['maxdd']:>+7.0f} | "
                      f"{nulls.mean():>+9.3f}{np.percentile(nulls,95):>+9.3f}{pct:>6.0f}%")

    # ---- 2: deployable soft-tilt forms (1d) ----
    print("\n" + "="*96)
    print("DEPLOYABLE: soft momentum tilt vs EW-all -- keep skill, control drawdown (1d, de-leaked, PROD)")
    print("="*96)
    for tf in ["1d", "4h"]:
        out["deployable"][tf] = {}
        print(f"\n  {tf}:  {'form':<22}{'VAL_lin':>8}{'VAL_Sh':>7}{'VAL_DD':>7}{'VAL_r/DD':>8} | "
              f"{'OOS_lin':>8}{'OOS_Sh':>7}{'OOS_DD':>7}{'OOS_r/DD':>8}")
        fV, fO = frames("VAL", tf), frames("OOS", tf)
        forms = {
            "EW-all (baseline)":      (ew_w(fV["eligible"]), ew_w(fO["eligible"])),
            "mom soft-tilt cap33":    (softtilt_w(fV["mom"], fV["eligible"], 0.33), softtilt_w(fO["mom"], fO["eligible"], 0.33)),
            "mom soft-tilt cap20":    (softtilt_w(fV["mom"], fV["eligible"], 0.20), softtilt_w(fO["mom"], fO["eligible"], 0.20)),
            "mom soft-tilt^2 cap33":  (softtilt_w(fV["mom"], fV["eligible"], 0.33, 2.0), softtilt_w(fO["mom"], fO["eligible"], 0.33, 2.0)),
            "mom top-third EW":       (topq_w(fV["mom"], fV["eligible"], 0.34), topq_w(fO["mom"], fO["eligible"], 0.34)),
        }
        for name, (Wv, Wo) in forms.items():
            mv = metrics(net(Wv, fV["ret"]), fV["bpd"], fV["mask"])
            mo = metrics(net(Wo, fO["ret"]), fO["bpd"], fO["mask"])
            out["deployable"][tf][name] = {"VAL": mv, "OOS": mo}
            print(f"      {name:<22}{mv['lin_wk']:>+8.2f}{mv['sharpe']:>+7.2f}{mv['maxdd']:>+7.0f}{mv['ret_per_dd']:>+8.3f} | "
                  f"{mo['lin_wk']:>+8.2f}{mo['sharpe']:>+7.2f}{mo['maxdd']:>+7.0f}{mo['ret_per_dd']:>+8.3f}")
    with open(HERE / "wave3b_momentum_deployable.json", "w") as fo:
        json.dump(out, fo, indent=2, default=str)
    print(f"\nSaved {HERE/'wave3b_momentum_deployable.json'}")


if __name__ == "__main__":
    main()
