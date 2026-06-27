"""A/B TEST — Frozen band vs Rolling-reselected top-K band (MA family only).

QUESTION (user, 2026-06-27): does dynamically reselecting the top-K configs per MA type
every 14d from RECENT realized performance beat the FROZEN TRAIN+VAL band? This keeps the
engine "static" (fixed config grid) but lets it SWAP which configs are active by condition.

DEAD-LIST CONTEXT: D73 found weekly config-PERSISTENCE is anti-predictive for MA
(recent-winner -> OOS median -0.8 to -1.9%/wk). This A/B tests whether that extends to the
14d-lookback / 7d-window / u100 / band resolution, OR whether rolling-reselect helps here.

RIGOR (SOTA comparison — only ONE variable changes: frozen vs rolling band):
  - SAME 120-config universe per (type, tf)
  - SAME confirmation layer, regime gate, prod cost, lag-1, slice/lookback, prod allocator
  - FROZEN arm: the fixed TRAIN+VAL-positive band (114 configs), voted, every bar
  - ROLLING arm: every 14d, score all 120 configs by their realized P&L over the TRAILING
    14d (STRICTLY CAUSAL — only past bars), pick top-K, vote across those for the next 7d
  - RANDOM-RESELECT NULL: same machinery but pick K configs at RANDOM each window (not by
    P&L). If rolling beats frozen but NOT the random-reselect null, the "edge" is just
    churn/variance, not skill. This is the SOTA control that isolates selection skill.

CAUSALITY GUARANTEE: the trailing-window score for window [t, t+7d) uses ONLY bars < t.
The chosen configs are then applied to bars in [t, t+7d). No look-ahead.

Scored per-asset (each asset picks its own best configs from its own recent behavior) —
this matches the user's "select winning candidates per 14d" intent.
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
from strat.ma_movecatch_engine import select_band_trainval, LEADERBOARD
from strat.ma_type_upgrade import held_cross, _nums, _ema
from strat.canonical_exits import COST_PROD
from strat.ma_slice_engine import causal_regime, OOS_WALL_MS, OOS_START, OOS_2024_END

BARS_PER_DAY = fl.BARS_PER_DAY
OUTDIR = CRYPTO / "runs" / "engines" / "V9_rolling_band"
TIER_CFG = {
    "1d": {"lookback": 14, "slice": 14},
    "4h": {"lookback": 14, "slice": 7},
    "2h": {"lookback": 7, "slice": 7},
}


def _full_config_universe(ma_type, tf):
    """All ranked configs for (type,tf) — the rotation universe (~120)."""
    d = json.load(open(LEADERBOARD))
    node = d["grid"].get(f"{ma_type}|{tf}")
    if node is None:
        return []
    return [r["config"] for r in node["ranked"]]


def _config_held_matrix(close, configs, ma_type):
    """[n_configs x T] per-config held arrays for one asset (causal)."""
    c = np.asarray(close, float)
    mats = []
    for cfg in configs:
        periods = _nums(cfg)
        try:
            mats.append(held_cross(c, periods, ma_type).astype(np.int8))
        except Exception:
            mats.append(np.zeros(len(c), dtype=np.int8))
    return np.array(mats)  # [K_all x T]


def _confirm_and_vote(close, votes_frac, lookback_bars, slice_bars, vote_thr=0.5):
    """Apply majority-vote threshold + confirmation (close>EMA & rising) + slice resolve.
    Reuses the canonical resolve via a simple in-line loop (matches ma_slice_engine)."""
    from strat.canonical_exits import resolve_held
    c = np.asarray(close, float)
    n = len(c)
    raw = (votes_frac > vote_thr).astype(np.int8)
    # confirmation
    m = np.isfinite(c)
    conf = np.zeros(n, dtype=bool)
    if m.sum() >= max(5, lookback_bars):
        cseg = c[m]
        ema = _ema(cseg, lookback_bars)
        above = cseg > ema
        rising = np.concatenate([[False], ema[1:] >= ema[:-1]])
        rseg = raw[m].astype(bool)
        conf_seg = rseg & above & rising
        conf[np.where(m)[0]] = conf_seg
    return resolve_held(c, conf, slice_bars)


def build_holds(ma_type, tf, mode, top_k=10, seed=0):
    """Build per-asset held panel under one mode.
    mode: 'frozen' | 'rolling' | 'random'
    Returns (close_df, held_df).
    """
    cfg = TIER_CFG[tf]; bpd = BARS_PER_DAY[tf]
    lookback_bars = cfg["lookback"] * bpd
    slice_bars = cfg["slice"] * bpd

    frozen_band = select_band_trainval(ma_type, tf)[0]
    universe = _full_config_universe(ma_type, tf)
    rng = np.random.default_rng(seed)

    close_cols, held_cols = {}, {}
    for sym in _u100_symbols():
        rec = _load_asset(sym, tf, OOS_WALL_MS)
        if rec is None:
            continue
        dates, o, h, l, c = rec
        c = np.asarray(c, float)
        n = len(c)
        close_cols[sym] = pd.Series(c, index=dates)

        if mode == "frozen":
            # vote across the fixed band every bar
            mats = _config_held_matrix(c, frozen_band, ma_type)
            frac = mats.mean(axis=0) if len(mats) else np.zeros(n)
            held = _confirm_and_vote(c, frac, lookback_bars, slice_bars)
        else:
            # rolling/random: per-window reselect from the FULL universe
            mats_all = _config_held_matrix(c, universe, ma_type)  # [Kall x T]
            rets = np.zeros(n); rets[1:] = np.diff(c) / np.where(c[:-1] > 0, c[:-1], 1.0)
            frac = np.zeros(n)
            # walk windows; for window [w, w+slice), pick configs using bars < w
            w = lookback_bars
            while w < n:
                w_end = min(w + slice_bars, n)
                lb_start = max(0, w - lookback_bars)
                if mode == "rolling":
                    # score each config by trailing realized P&L: sum(held_lag * ret) over [lb_start, w)
                    scores = []
                    for ki in range(len(universe)):
                        hk = mats_all[ki, lb_start:w]
                        hk_lag = np.concatenate([[0], hk[:-1]]) if len(hk) else hk
                        rk = rets[lb_start:w]
                        scores.append(float(np.sum(hk_lag * rk)) if len(hk) else -1e9)
                    chosen = np.argsort(scores)[::-1][:top_k]
                else:  # random
                    chosen = rng.choice(len(universe), size=min(top_k, len(universe)), replace=False)
                # vote across chosen configs for THIS window only
                sub = mats_all[chosen, w:w_end]
                frac[w:w_end] = sub.mean(axis=0) if len(sub) else 0.0
                w = w_end
            held = _confirm_and_vote(c, frac, lookback_bars, slice_bars)

        held_cols[sym] = pd.Series(held, index=dates)

    close_df = pd.DataFrame(close_cols).sort_index()
    held_df = pd.DataFrame(held_cols).reindex(close_df.index).fillna(0).astype(np.int8)
    return close_df, held_df


def evaluate(close_df, held_df, tf, cap=0.15):
    """PROD-grade eval: inv-vol + cap, regime gate, prod cost, OOS-2024 window."""
    bpd = BARS_PER_DAY[tf]
    reg = causal_regime(OOS_WALL_MS)   # de-leaked: bar on day D reads regime[D-1]
    ret = close_df.pct_change().fillna(0.0).to_numpy()
    H = held_df.to_numpy().astype(float)
    vol = close_df.pct_change().rolling(max(20, bpd*5), min_periods=10).std().to_numpy()
    dates = close_df.index; T, A = H.shape
    rg = np.ones(T); ridx = reg.index
    for k, dt in enumerate(dates):
        d1 = pd.Timestamp(dt).normalize(); prior = ridx[ridx <= d1]
        rg[k] = int(reg[prior[-1]]) if len(prior) else 0
    Hg = H * rg[:, None]
    iv = np.where((Hg > 0) & np.isfinite(vol) & (vol > 0), 1.0/np.where(vol > 0, vol, np.nan), 0.0)
    iv = np.nan_to_num(iv); s = iv.sum(axis=1, keepdims=True)
    w = np.where(s > 0, iv/np.maximum(s, 1e-12), 0.0)
    w = np.minimum(w, cap)
    Wl = np.vstack([np.zeros((1, A)), w[:-1]])
    gr = np.nansum(Wl*ret, axis=1)
    turn = np.nansum(np.abs(w - np.vstack([np.zeros((1, A)), w[:-1]])), axis=1)
    net = gr - turn*(COST_PROD/2)
    mask = (dates >= pd.Timestamp(OOS_START)) & (dates < pd.Timestamp(OOS_2024_END))
    nw = net[mask]
    g = np.nansum(np.abs(w), axis=1)[mask]
    lin = float(np.mean(nw))*bpd*7*100
    comp = float(np.prod(1+nw)-1)*100
    cum = np.cumprod(1+nw); dd = float(np.min((cum-np.maximum.accumulate(cum))/np.maximum.accumulate(cum)))*100
    sh = (float(np.mean(nw))*bpd*365)/(float(np.std(nw))*np.sqrt(bpd*365)) if np.std(nw) > 1e-12 else 0
    ag = float(np.mean(g[g > 0])) if (g > 0).any() else 0
    return {"lin_wk": round(lin,3), "compound": round(comp,1), "sharpe": round(sh,2),
            "maxdd": round(dd,1), "avg_gross": round(ag,3)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", default="SMA,EMA,WMA,VIDYA,KAMA,DEMA")
    ap.add_argument("--tfs", default="1d,4h")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--nrandom", type=int, default=10)
    ap.add_argument("--cap", type=float, default=0.15)
    a = ap.parse_args()
    types = [t.strip().upper() for t in a.types.split(",")]
    tfs = [t.strip() for t in a.tfs.split(",")]

    print("=" * 104)
    print(f"A/B — FROZEN band vs ROLLING-reselect top-{a.topk} vs RANDOM-reselect null "
          f"(MA family, OOS-2024, prod cost, {a.cap:.0%} cap)")
    print("=" * 104)
    print(f"  {'cand':<10}| {'FROZEN':>18} | {'ROLLING':>18} | {'RANDOM(meanP5-P95)':>22} | verdict")
    print(f"  {'':10}| {'lin/wk  Sh  MaxDD':>18} | {'lin/wk  Sh  MaxDD':>18} | {'lin/wk band':>22} |")
    print("  " + "-" * 96)

    rows = []
    t0 = time.time()
    for tf in tfs:
        for ma in types:
            cf, hf = build_holds(ma, tf, "frozen")
            mf = evaluate(cf, hf, tf, a.cap)
            cr, hr = build_holds(ma, tf, "rolling", top_k=a.topk)
            mr = evaluate(cr, hr, tf, a.cap)
            # random-reselect null distribution
            rand_lins = []
            for sd in range(a.nrandom):
                cR, hR = build_holds(ma, tf, "random", top_k=a.topk, seed=sd)
                rand_lins.append(evaluate(cR, hR, tf, a.cap)["lin_wk"])
            rand_lins = np.array(rand_lins)
            r_mean, r_p5, r_p95 = float(rand_lins.mean()), float(np.percentile(rand_lins,5)), float(np.percentile(rand_lins,95))
            # verdict: rolling must beat frozen AND beat random P95 to be REAL selection skill
            beats_frozen = mr["lin_wk"] > mf["lin_wk"]
            beats_rand = mr["lin_wk"] > r_p95
            if beats_frozen and beats_rand:
                verdict = "ROLLING WINS (skill)"
            elif beats_frozen:
                verdict = "rolling>frozen but ~random (churn, not skill)"
            elif mr["lin_wk"] >= r_mean:
                verdict = "frozen wins; rolling~random"
            else:
                verdict = "FROZEN WINS (D73 holds)"
            rows.append({"ma": ma, "tf": tf, "frozen": mf, "rolling": mr,
                         "random_mean": round(r_mean,3), "random_p5": round(r_p5,3),
                         "random_p95": round(r_p95,3), "verdict": verdict})
            print(f"  {ma+'|'+tf:<10}| {mf['lin_wk']:>+7.2f} {mf['sharpe']:>4.2f}{mf['maxdd']:>6.0f}% | "
                  f"{mr['lin_wk']:>+7.2f} {mr['sharpe']:>4.2f}{mr['maxdd']:>6.0f}% | "
                  f"{r_mean:>+6.2f} [{r_p5:+.2f},{r_p95:+.2f}] | {verdict}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(OUTDIR / f"rolling_band_ab_{ts}.json", "w") as f:
        json.dump({"rows": rows, "top_k": a.topk, "n_random": a.nrandom, "cap": a.cap,
                   "cost_rt": COST_PROD, "ts": ts}, f, indent=2, default=str)
    nwin = sum(1 for r in rows if "ROLLING WINS" in r["verdict"])
    print("  " + "-" * 96)
    print(f"  ROLLING-WINS-WITH-SKILL: {nwin}/{len(rows)} cells. ({(time.time()-t0)/60:.1f} min)")
    print(f"  Saved rolling_band_ab_{ts}.json")


if __name__ == "__main__":
    main()
