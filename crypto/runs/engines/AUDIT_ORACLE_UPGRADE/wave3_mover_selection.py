"""WAVE 3 -- CAUSAL MOVER-SELECTION: can a price-only score tilt the deployed set toward the
future top movers and beat equal-weight, ROBUSTLY (VAL+OOS), ABOVE a random-selection null?

The oracle says the whole recoverable headroom is name-SELECTION: perfect hindsight top-3 =
+14%/wk @ -24.6% DD (4x the EW engine's return AND a shallower drawdown). This tests whether
ANY of that is CAUSALLY recoverable from internal price data alone -- the user's "capture the
movers (top-% daily performers) ahead of time, using the lookback."

DESIGN (clean isolation of SELECTION):
  - Eligible set each bar = regime-gated (DE-LEAKED) AND cross-type band-confirmed (conv>0) -- the
    SAME deploy timing as the V2 engine. We only change WHICH eligible names get weight.
  - For each causal score S (computed at the decision bar from PAST data only, lag-1 applied):
      top-K by S  (concentrate into predicted movers, EW among them)
    vs baselines:  EW-all (current engine)  and  random-K null (N shuffles among eligible).
  - A score has SKILL iff causal-top-K beats random-K (selection skill) AND beats EW-all
    (concentration pays) on BOTH windows, prod cost, de-leaked.

CAUSAL SCORES (price-only, from OHLC, all known at the decision close):
  mom    = trailing lb-bar return                       (cross-sectional momentum / winners-persist)
  accel  = mom(lb) - mom(lb) shifted lb                  (trend acceleration)
  brk    = close / rolling-max(close, lb)               (breakout proximity, Donchian)
  dist   = close / SMA(close, lb) - 1                    (distance above the band)
  volx   = std(ret, lb) / std(ret, 4*lb)                (volatility expansion -- movers pre-expand)

2025 SEALED. long-only spot lev=1. PROD cost.
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
from strat.canonical_exits import COST_PROD

BARS_PER_DAY = fl.BARS_PER_DAY
TYPES = ["SMA", "EMA", "WMA", "VIDYA", "KAMA"]
# windows from ma_slice_engine constants
from strat.ma_slice_engine import OOS_WALL_MS, OOS_START, OOS_2024_END, VAL_WALL_MS, VAL_START, VAL_END
WIN = {"OOS": (OOS_WALL_MS, OOS_START, OOS_2024_END), "VAL": (VAL_WALL_MS, VAL_START, VAL_END)}
TF = "1d"; LB_D, SL_D = 14, 14


def _frames(window):
    wall, ws, we = WIN[window]; bpd = BARS_PER_DAY[TF]
    lb, sl = LB_D*bpd, SL_D*bpd
    reg = causal_regime(wall)               # DE-LEAKED at source
    closes = {}
    for sym in _u100_symbols():
        rec = _load_asset(sym, TF, wall)
        if rec is None: continue
        d, o, h, l, c = rec
        closes[sym] = (d, o, h, l, np.asarray(c, float))
    syms = list(closes.keys())
    close_df = pd.DataFrame({s: pd.Series(closes[s][4], index=closes[s][0]) for s in syms}).sort_index()
    idx = close_df.index; T = len(idx); A = len(syms)
    # cross-type convergence (band-confirmed breadth), same as build_book_net
    conv = np.zeros((T, A))
    for ma in TYPES:
        hc = {}
        for s in syms:
            d, o, h, l, c = closes[s]
            bh = _cached_band_held(ma, TF, s, o, h, l, c, wall)
            hc[s] = pd.Series(held_rolling(c, bh, lb, sl), index=d)
        conv += pd.DataFrame(hc).reindex(idx).fillna(0).reindex(columns=close_df.columns).fillna(0).to_numpy()
    # regime on grid (de-leaked): bar day D reads last CLOSED 1d bar (reg already shifted)
    ridx = reg.index; rg = np.ones(T)
    for k, dt in enumerate(idx):
        d1 = pd.Timestamp(dt).normalize(); prior = ridx[ridx <= d1]
        rg[k] = int(reg[prior[-1]]) if len(prior) else 0
    eligible = (conv > 0) & (rg[:, None] > 0)        # T x A boolean: deployable names per bar
    C = close_df.to_numpy()
    ret = close_df.pct_change().fillna(0.0).to_numpy()
    # ---- causal scores (computed from PAST closes; cross-sectional each bar) ----
    cdf = close_df
    mom = cdf.pct_change(lb).to_numpy()
    accel = mom - np.vstack([np.full((lb, A), np.nan), mom[:-lb]]) if T > lb else np.full_like(mom, np.nan)
    rollmax = cdf.rolling(lb, min_periods=max(3, lb//2)).max().to_numpy()
    brk = C / np.where(rollmax > 0, rollmax, np.nan)
    sma = cdf.rolling(lb, min_periods=max(3, lb//2)).mean().to_numpy()
    dist = C / np.where(sma > 0, sma, np.nan) - 1.0
    rstd_s = cdf.pct_change().rolling(lb, min_periods=max(3, lb//2)).std().to_numpy()
    rstd_l = cdf.pct_change().rolling(4*lb, min_periods=lb).std().to_numpy()
    volx = rstd_s / np.where(rstd_l > 0, rstd_l, np.nan)
    scores = {"mom": mom, "accel": accel, "brk": brk, "dist": dist, "volx": volx}
    mask = (idx >= pd.Timestamp(ws)) & (idx < pd.Timestamp(we))
    return {"idx": idx, "ret": ret, "eligible": eligible, "scores": scores, "mask": mask, "A": A, "bpd": bpd}


def _net_from_weights(W, ret, cost_rt=COST_PROD):
    """W is T x A target weights (rows sum<=1). Lag-1 application, turnover cost."""
    Wl = np.vstack([np.zeros((1, W.shape[1])), W[:-1]])
    gr = np.nansum(Wl * ret, axis=1)
    turn = np.nansum(np.abs(W - np.vstack([np.zeros((1, W.shape[1])), W[:-1]])), axis=1)
    return gr - turn * (cost_rt / 2)


def _metrics(net, bpd, mask):
    n = net[mask]
    if len(n) == 0: return None
    lin = float(np.mean(n))*bpd*7*100
    cum = np.cumprod(1+n)
    dd = float(np.min((cum-np.maximum.accumulate(cum))/np.maximum.accumulate(cum)))*100 if len(cum)>1 else 0
    sh = (float(np.mean(n))*bpd*365)/(float(np.std(n))*np.sqrt(bpd*365)) if np.std(n)>1e-12 else 0
    sb = 7*bpd; sl=[float(np.prod(1+n[i:i+sb])-1)*100 for i in range(0,len(n)-sb+1,sb)]
    pct7 = sum(1 for x in sl if x>0)/len(sl) if sl else 0
    return {"lin_wk": round(lin,3), "sharpe": round(sh,3), "maxdd": round(dd,1),
            "ret_per_dd": round(lin/abs(dd),4) if abs(dd)>1e-9 else 0, "pct7d": round(pct7,3)}


def _topk_weights(score, eligible, K):
    """Per bar: among eligible names, EW the top-K by score. score NaN -> not selectable."""
    T, A = score.shape
    W = np.zeros((T, A))
    s = np.where(eligible, score, -np.inf)
    s = np.where(np.isfinite(s), s, -np.inf)
    for t in range(T):
        row = s[t]
        elig_t = np.where(np.isfinite(row))[0]
        if len(elig_t) == 0: continue
        k = min(K, len(elig_t))
        top = elig_t[np.argsort(-row[elig_t])[:k]]
        W[t, top] = 1.0/k
    return W


def _randk_weights(eligible, K, seed):
    T, A = eligible.shape
    W = np.zeros((T, A))
    rng = np.random.default_rng(seed)
    for t in range(T):
        elig_t = np.where(eligible[t])[0]
        if len(elig_t) == 0: continue
        k = min(K, len(elig_t))
        pick = rng.choice(elig_t, size=k, replace=False)
        W[t, pick] = 1.0/k
    return W


def _ew_all_weights(eligible):
    cnt = eligible.sum(axis=1, keepdims=True)
    return np.where(eligible & (cnt > 0), 1.0/np.maximum(cnt, 1), 0.0)


def main():
    Ks = [3, 5, 10]
    N_NULL = 25
    out = {"check": "WAVE3_causal_mover_selection", "ts": datetime.now().strftime("%Y%m%d_%H%M%S"),
           "tf": TF, "note": "top-K by causal price score vs random-K null vs EW-all; de-leaked, PROD, 2025 SEALED",
           "windows": {}}
    F = {w: _frames(w) for w in ["VAL", "OOS"]}
    # EW-all baseline + random-K null per window (shared across scores)
    base = {}
    for w in ["VAL", "OOS"]:
        f = F[w]
        ew = _metrics(_net_from_weights(_ew_all_weights(f["eligible"]), f["ret"]), f["bpd"], f["mask"])
        randk = {}
        for K in Ks:
            mets = []
            for s in range(N_NULL):
                m = _metrics(_net_from_weights(_randk_weights(f["eligible"], K, s), f["ret"]), f["bpd"], f["mask"])
                if m: mets.append(m)
            randk[K] = {"lin_wk_mean": round(float(np.mean([m["lin_wk"] for m in mets])),3),
                        "lin_wk_p90": round(float(np.percentile([m["lin_wk"] for m in mets],90)),3),
                        "sharpe_mean": round(float(np.mean([m["sharpe"] for m in mets])),3),
                        "maxdd_mean": round(float(np.mean([m["maxdd"] for m in mets])),1)}
        base[w] = {"EW_all": ew, "randK": randk}
    out["baselines"] = base

    for w in ["VAL", "OOS"]:
        f = F[w]
        print(f"\n{'='*100}\n{w}  causal mover-selection (de-leaked, PROD)  | EW-all: "
              f"lin_wk={base[w]['EW_all']['lin_wk']:+.3f} Sh={base[w]['EW_all']['sharpe']:+.2f} DD={base[w]['EW_all']['maxdd']:+.0f}")
        print(f"{'='*100}")
        print(f"  {'score':>6} {'K':>3} | {'lin_wk':>8}{'Sharpe':>8}{'maxDD':>7}{'ret/DD':>8} | "
              f"{'randK_lin':>9}{'randK_p90':>9} | {'beats_rand':>10}{'beats_EW':>9}")
        print("  " + "-"*92)
        wres = {}
        for sname, score in f["scores"].items():
            for K in Ks:
                W = _topk_weights(score, f["eligible"], K)
                m = _metrics(_net_from_weights(W, f["ret"]), f["bpd"], f["mask"])
                if m is None: continue
                rk = base[w]["randK"][K]
                beats_rand = m["lin_wk"] > rk["lin_wk_p90"]      # above the 90th pct of random selection
                beats_ew = m["lin_wk"] > base[w]["EW_all"]["lin_wk"]
                wres[f"{sname}_K{K}"] = {**m, "rand_lin_mean": rk["lin_wk_mean"], "rand_lin_p90": rk["lin_wk_p90"],
                                        "beats_rand_p90": beats_rand, "beats_EW": beats_ew}
                print(f"  {sname:>6} {K:>3} | {m['lin_wk']:>+8.3f}{m['sharpe']:>+8.2f}{m['maxdd']:>+7.0f}{m['ret_per_dd']:>+8.3f} | "
                      f"{rk['lin_wk_mean']:>+9.3f}{rk['lin_wk_p90']:>+9.3f} | {'Y' if beats_rand else '.':>10}{'Y' if beats_ew else '.':>9}")
        out["windows"][w] = wres
    with open(HERE / "wave3_mover_selection.json", "w") as fo:
        json.dump(out, fo, indent=2, default=str)
    print(f"\nSaved {HERE/'wave3_mover_selection.json'}")


if __name__ == "__main__":
    main()
