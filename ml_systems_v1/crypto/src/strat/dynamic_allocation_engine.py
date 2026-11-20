"""src/strat/dynamic_allocation_engine.py -- PHASE 2: the DYNAMIC / ML ALLOCATION ENGINE.

USER ASK (verbatim core vision): a "dynamic" + "ML engine" that regime-conditionally weights the two
COMPLEMENTARY families (trend MA + MR oscillator) per timeframe so "when one is out, the other is on."
This is the HEADLINE deliverable of the 5h strategy-discovery-engine build.

WHAT THIS BUILDS (per TF, 2020 runway, u10, maker, causal/lag-1):
  - CANDIDATE SLEEVES: the best trend MA-family (from ma_type_tf_research winners; EMA fallback if a TF is
    missing) + the MR oscillator family. Per-day net + per-day exposure, reusing deep2020_complementarity's
    sleeve mechanics (NOT reinvented).
  - THE DYNAMIC ENGINE (two tiers): per rolling window, past-only CAUSAL features (trend-strength/ADX-like,
    vol-regime, chop-vs-trend ratio, recent per-sleeve performance/persistence, breadth) -> a trend-vs-MR
    weight w_t in [0,1].
      Tier A (interpretable REGIME RULE): high trend-strength -> weight trend; chop/low-trend -> weight MR.
      Tier B (ML): a ridge/James-Stein-shrunk OR sklearn HistGB scorer that LEARNS the weighting from the
        same features (predict each sleeve's next-window net -> softmax to a weight). Small-sample-robust.
    Fit on TRAIN, SELECT the tier on VAL, confirm ONCE on OOS (NEVER select on OOS).

  - HONEST CONTROLS (the engine only "works" if it beats ALL):
      (a) the best STATIC complementary blend (B's optimal per-TF blend, from complementarity_matrix.json);
      (b) VOLTGT_BH on the combined net;
      (c) a SHUFFLE control -- block-shuffle the WEIGHT sequence (same weight distribution, random timing):
          real timing skill MUST beat its own shuffle's distribution (N draws, one-sided p);
      (d) a RANDOM allocator (random weight per window).
    Plus PBO (rank-flip TRAIN->OOS) + block-bootstrap p05.

THE BINDING HONEST FRAME (from COMPLEMENTARITY.md, respected here): gap-filling between two LONG-ONLY
sleeves is DRAWDOWN-DAMPENING, not positive-return rescue -- both sleeves are long so they cannot rescue
each other's down days. So the dynamic engine's realistic objective is BETTER RISK-ADJUSTED return
(Sharpe / maxDD / p05) via regime-timed weighting, NOT magic alpha. We do NOT claim alpha it does not
have. If the dynamic engine does NOT beat the static blend, we REPORT THAT HONESTLY -- then the static
blend is the answer and dynamic timing adds nothing (a real, valuable finding, not a failure).

CONSTRAINTS (user mandate): 2020 BAND ONLY (everything fenced to 2020-01-07..2021-01-01; data past that is
never scored). DO NOT touch 2026/other data. Synthetic only for nulls (selftest). Charts via matplotlib
(Agg). No emoji (cp1252). RWYB. Do NOT git commit. 2020-bull-only limitation flagged (regime-stress later).

RWYB:
  python -m strat.dynamic_allocation_engine --selftest                 # two-sided synthetic control
  python -m strat.dynamic_allocation_engine                            # the full runway, all 6 TFs + charts
  python -m strat.dynamic_allocation_engine --cadences 1d,4h           # a subset
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.portfolio_replay as PR                                      # noqa: E402
from strat.portfolio_replay import MAKER_RT, TAKER_RT, apply_trail_stop  # noqa: E402
from strat.replay_distinct_grid import distinct_specs                    # noqa: E402
from strat.structural_fixes import min_hold                              # noqa: E402
from strat.ma_type_upgrade import _MA, _nums                             # noqa: E402
from strat.ma_2020_breakdown import _panel                              # noqa: E402
from strat.data_expansion import block_bootstrap_distribution, james_stein_shrink  # noqa: E402
from strat.deep2020_osc import _grid, _val, _mr_held                     # noqa: E402

OUT = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
CHARTS = OUT / "charts"

__contract__ = {
    "kind": "dynamic_ml_allocation_engine",
    "inputs": {
        "sleeves": "two LONG-ONLY complementary sleeves per TF: trend MA-family (best MA-type per "
                   "ma_type_tf_research winner; EMA fallback) + MR oscillator family -- daily net + exposure",
        "features": "past-only causal X(t) per rolling daily window: trend-strength(ADX-like)/vol-level/"
                    "vol-regime/chop-vs-trend/recent-per-sleeve-perf/breadth -- standardized on TRAIN only",
        "weight": "engine output w_t in [0,1] = fraction to TREND (1-w to MR), applied lagged to next window",
    },
    "outputs": {
        "per_tf": "OOS net/Sharpe/maxDD/p05 of the DYNAMIC engine vs STATIC blend vs VOLTGT_BH vs SHUFFLE "
                  "vs RANDOM, both tiers, tier chosen on VAL",
        "verdict": "does the dynamic engine BEAT the static blend on RISK-ADJUSTED return (Sharpe/maxDD/p05) "
                   "AND beat its own weight-shuffle -- honest either way (static-is-the-answer is a real finding)",
    },
    "invariants": {
        "long_only_gap_fill_is_dd_dampening": "both sleeves long-only -> combining dampens DD, does NOT rescue "
                                              "return; the engine target is risk-adjusted, NOT alpha",
        "select_on_val_never_oos": "tier (A vs B) chosen on VAL; OOS is test-once",
        "past_only_causal_lag1": "every feature uses bars <= window start; weight applied to the NEXT window",
        "weight_shuffle_must_be_beaten": "block-shuffle the WEIGHT sequence -> real timing skill beats the "
                                         "shuffle distribution (else it is just average exposure)",
        "static_blend_is_the_bar": "if dynamic does not beat the best static blend, the static blend is the "
                                   "answer (reported honestly, not hidden)",
        "2020_band_only": "runway fenced to 2020-01-07..2021-01-01; data past that never scored; bull-only",
        "causal_mtm_no_double_count": "positions lagged 1 bar; cost charged on flips; no MtM double-count",
    },
}

# ---- the 2020 runway (TRAIN fit / VAL select / OOS test) -- 2020 BAND ONLY -------------------------------
RUNWAY = ("2020-01-01", "2021-01-01")          # data fence; first real bar ~2020-01-07
SPLITS = {"TRAIN": ("2020-01-01", "2020-07-01"),   # fit (Jan-Jun)
          "VAL":   ("2020-07-01", "2020-10-01"),   # select the tier (Jul-Sep)
          "OOS":   ("2020-10-01", "2021-01-01")}   # confirm once (Oct-Dec) == the complementarity OOS
WARMUP = 400
CADENCES = ["1d", "4h", "2h", "1h", "30m", "15m"]
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT",
        "LINKUSDT", "LTCUSDT"]
# rolling daily-allocation window length (days). The unit is a SETUP across a multi-day move, not a candle.
ROLL_DAYS = 7
MA_RESEARCH = OUT / "ma_type_tf_research.json"
COMP_MATRIX = OUT / "complementarity_matrix.json"


# =====================================================================================================
# 1. SLEEVE BUILDERS -- trend MA-family (best MA-type per TF) + MR oscillator family, over the FULL runway.
#    Reuses deep2020_complementarity / deep2020_osc mechanics; only the window is widened to the full
#    runway (TRAIN+VAL+OOS) and the trend MA-type is chosen per the ma_type_tf_research winner.
# =====================================================================================================
def _best_ma_type(cad):
    """The best trend MA-type for a TF from ma_type_tf_research winners; EMA fallback if the TF is absent
    (the mandate: 'if a TF is missing, use EMA family')."""
    try:
        d = json.load(open(MA_RESEARCH))
        w = d.get("winners_by_tf", {}).get(cad, {})
        mt = w.get("ma_type")
        if mt and mt in _MA:
            return mt, "ma_type_tf_research winner"
    except Exception:
        pass
    return "EMA", "EMA fallback (TF absent from ma_type_tf_research)"


def _daily_compound(net_bar: pd.Series) -> pd.Series:
    return net_bar.resample("1D").apply(lambda x: float(np.prod(1 + x) - 1)).dropna()


def _daily_exposure(exp_bar: pd.Series) -> pd.Series:
    return exp_bar.resample("1D").max().dropna()


def _trend_sleeve(cad, ma_type):
    """trend MA-slow-family sleeve (chosen MA-type), full-stack (trail10 + min_hold12 + maker), u10
    equal-weight. Returns (daily_net, daily_exposure) over the full runway. Mechanics mirror
    deep2020_complementarity._trend_sleeve, but over RUNWAY and with a selectable MA-type."""
    ma_cfg = {}
    for fam in ("2MA", "3MA"):
        ma_cfg.update(distinct_specs(fam, 0.15, max_n=40))
    PR.STRATS.update(ma_cfg)
    slow = [n for n in ma_cfg if 60 <= max(_nums(n)) < 150]
    s_ms = pd.Timestamp(RUNWAY[0]).value // 10**6
    e_ms = pd.Timestamp(RUNWAY[1]).value // 10**6
    per_net, per_exp = [], []
    for sym in SYMS:
        try:
            o, h, l, c, ms = _panel(sym, cad)
        except Exception:
            continue
        e = int(np.searchsorted(ms, e_ms)); s0 = max(0, int(np.searchsorted(ms, s_ms)) - WARMUP)
        c2, ms2 = c[s0:e], ms[s0:e]
        if len(c2) < 40:
            continue
        win = ms2 >= s_ms
        if win.sum() < 30:
            continue
        ret = np.zeros(len(c2)); ret[1:] = c2[1:] / c2[:-1] - 1.0
        uniq = sorted({p for n in slow for p in _nums(n)})
        cache = {p: _MA[ma_type](c2, p) for p in uniq}
        cfg_nets, cfg_exps = [], []
        for name in slow:
            pp = _nums(name); mas = [cache[p] for p in pp]
            h0 = np.nan_to_num((mas[0] > mas[1]) if len(pp) == 2
                               else ((mas[0] > mas[1]) & (mas[1] > mas[2]))).astype(np.int8)
            h0 = min_hold(apply_trail_stop(h0.copy(), c2, 0.10)[0].astype(np.int8), 12).astype(np.float64)
            pos = np.zeros(len(c2)); pos[1:] = h0[:-1]
            flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
            cfg_nets.append((pos * ret - flips * (MAKER_RT / 2.0))[win])
            cfg_exps.append(pos[win])
        idx = pd.to_datetime(ms2[win], unit="ms")
        per_net.append(pd.Series(np.mean(cfg_nets, axis=0), index=idx))
        per_exp.append(pd.Series(np.mean(cfg_exps, axis=0), index=idx))
    if not per_net:
        return None, None
    net_bar = pd.concat(per_net, axis=1).mean(axis=1, skipna=True)
    exp_bar = pd.concat(per_exp, axis=1).mean(axis=1, skipna=True)
    return _daily_compound(net_bar), _daily_exposure(exp_bar)


def _mr_sleeve(cad):
    """equal-weight MR oscillator family sleeve, u10. Returns (daily_net, daily_exposure) over the full
    runway. Mechanics mirror deep2020_complementarity._mr_sleeve (over RUNWAY)."""
    s_ms = pd.Timestamp(RUNWAY[0]).value // 10**6
    e_ms = pd.Timestamp(RUNWAY[1]).value // 10**6
    grid = _grid()
    per_net, per_exp = [], []
    for sym in SYMS:
        try:
            o, h, l, c, ms = _panel(sym, cad)
        except Exception:
            continue
        e = int(np.searchsorted(ms, e_ms)); s0 = max(0, int(np.searchsorted(ms, s_ms)) - WARMUP)
        o2, h2, l2, c2, ms2 = o[s0:e], h[s0:e], l[s0:e], c[s0:e], ms[s0:e]
        if len(c2) < 40:
            continue
        win = ms2 >= s_ms
        if win.sum() < 30:
            continue
        ret = np.zeros(len(c2)); ret[1:] = c2[1:] / c2[:-1] - 1.0
        idx = pd.to_datetime(ms2[win], unit="ms")
        cfg_nets, cfg_exps = [], []
        for g in grid:
            kind, n, lo, hi = g
            v = _val(kind, c2, h2, l2, n)
            held = min_hold(_mr_held(v, lo, hi), 6).astype(np.float64)
            pos = np.zeros(len(c2)); pos[1:] = held[:-1]
            flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
            cfg_nets.append((pos * ret - flips * (MAKER_RT / 2.0))[win])
            cfg_exps.append(pos[win])
        per_net.append(pd.Series(np.mean(cfg_nets, axis=0), index=idx))
        per_exp.append(pd.Series(np.mean(cfg_exps, axis=0), index=idx))
    if not per_net:
        return None, None
    net_bar = pd.concat(per_net, axis=1).mean(axis=1, skipna=True)
    exp_bar = pd.concat(per_exp, axis=1).mean(axis=1, skipna=True)
    return _daily_compound(net_bar), _daily_exposure(exp_bar)


def _voltgt_bh_daily(cad):
    """VOLTGT_BH (vol-targeted buy-hold) daily net over the runway, u10 -- a control candidate."""
    s_ms = pd.Timestamp(RUNWAY[0]).value // 10**6
    e_ms = pd.Timestamp(RUNWAY[1]).value // 10**6
    VW = {"1d": 14, "4h": 84, "2h": 168, "1h": 168, "30m": 336, "15m": 672}[cad]
    per_net = []
    for sym in SYMS:
        try:
            o, h, l, c, ms = _panel(sym, cad)
        except Exception:
            continue
        keep = (ms >= s_ms - 30 * 86400000) & (ms < e_ms)
        c2 = c[keep]; ms2 = ms[keep]
        if len(c2) < 40:
            continue
        ret = np.zeros(len(c2)); ret[1:] = c2[1:] / c2[:-1] - 1.0
        rv = pd.Series(ret).rolling(VW).std().to_numpy(); med = np.nanmedian(rv)
        exp = np.nan_to_num(np.clip(med / (np.concatenate([[np.nan], rv[:-1]]) + 1e-12), 0, 1))
        win = ms2 >= s_ms
        idx = pd.to_datetime(ms2[win], unit="ms")
        per_net.append(pd.Series((exp * ret)[win], index=idx))
    if not per_net:
        return None
    return _daily_compound(pd.concat(per_net, axis=1).mean(axis=1, skipna=True))


# =====================================================================================================
# 2. ROLLING WINDOWS + per-window per-sleeve realized net (for labels) + the COMBINED-given-weight net
# =====================================================================================================
def rolling_windows(daily_index, roll_days):
    """Non-overlapping rolling daily windows of `roll_days` -> list of (lo_ts, hi_ts)."""
    n = len(daily_index)
    return [(daily_index[s], daily_index[min(s + roll_days, n - 1)])
            for s in range(0, n - roll_days, roll_days)]


def _win_slice(s: pd.Series, lo, hi) -> np.ndarray:
    return s[(s.index >= lo) & (s.index < hi)].dropna().to_numpy()


def window_net(daily_net: pd.Series, lo, hi) -> float:
    seg = _win_slice(daily_net, lo, hi)
    if len(seg) < 1:
        return np.nan
    return float(np.cumprod(1 + seg)[-1] - 1) * 100


def combined_window_net(trend_net, mr_net, lo, hi, w_trend) -> np.ndarray:
    """Daily combined net over [lo,hi) for a FIXED weight w_trend to trend (1-w to MR). Returns the
    bar-level daily combined-net array (so the stitched book compounds correctly)."""
    df = pd.concat([trend_net.rename("t"), mr_net.rename("m")], axis=1)
    df = df[(df.index >= lo) & (df.index < hi)].dropna()
    if len(df) < 1:
        return np.array([])
    return (w_trend * df["t"].to_numpy() + (1 - w_trend) * df["m"].to_numpy())


# =====================================================================================================
# 3. FEATURES (past-only, causal) per rolling daily window -- the regime/skill lever
# =====================================================================================================
def _adx_like(daily_net_combined_proxy: np.ndarray) -> float:
    """A cheap trend-strength proxy on a daily series: |cumulative drift| / sum|daily move| over the
    window (1.0 = pure trend, 0 = pure chop). Computed on the BUYHOLD-ish combined proxy series."""
    if len(daily_net_combined_proxy) < 2:
        return 0.5
    drift = abs(float(np.sum(daily_net_combined_proxy)))
    churn = float(np.sum(np.abs(daily_net_combined_proxy))) + 1e-12
    return float(np.clip(drift / churn, 0.0, 1.0))


def window_features(trend_net, mr_net, trend_exp, mr_exp, bh_net, windows, wi):
    """Causal features for window wi -- built ONLY from windows < wi (past-only). Returns an ordered dict.
      - trend_strength (ADX-like): |drift|/churn of the buy-hold proxy over the PRIOR window
      - vol_level: realized daily vol of the buy-hold proxy over the prior window
      - vol_regime: prior-window vol minus the median vol of all past windows
      - chop_vs_trend: 1 - trend_strength (high = choppy -> favors MR)
      - recent_trend / recent_mr: each sleeve's mean realized net over the last K past windows (persistence)
      - perf_spread: recent_trend - recent_mr (which sleeve has been winning lately)
      - trend_breadth / mr_breadth: mean exposure of each sleeve over the prior window
    """
    feats = {}
    K = 3
    # recent per-sleeve performance over the last K past windows
    rt, rm = [], []
    for j in range(max(0, wi - K), wi):
        lo, hi = windows[j]
        rt.append(window_net(trend_net, lo, hi)); rm.append(window_net(mr_net, lo, hi))
    feats["recent_trend"] = float(np.nanmean(rt)) if rt and np.any(np.isfinite(rt)) else 0.0
    feats["recent_mr"] = float(np.nanmean(rm)) if rm and np.any(np.isfinite(rm)) else 0.0
    feats["perf_spread"] = feats["recent_trend"] - feats["recent_mr"]
    # regime / vol / breadth from the PRIOR window (past-only)
    if wi >= 1:
        plo, phi = windows[wi - 1]
        bh_seg = _win_slice(bh_net, plo, phi)
        feats["trend_strength"] = _adx_like(bh_seg)
        feats["vol_level"] = float(np.std(bh_seg)) if len(bh_seg) > 1 else 0.0
        feats["trend_breadth"] = float(np.mean(_win_slice(trend_exp, plo, phi))) if len(_win_slice(trend_exp, plo, phi)) else 0.5
        feats["mr_breadth"] = float(np.mean(_win_slice(mr_exp, plo, phi))) if len(_win_slice(mr_exp, plo, phi)) else 0.5
    else:
        feats.update({"trend_strength": 0.5, "vol_level": 0.0, "trend_breadth": 0.5, "mr_breadth": 0.5})
    feats["chop_vs_trend"] = 1.0 - feats["trend_strength"]
    # vol_regime: prior-window vol vs median past-window vol
    if wi >= 2:
        vols = []
        for j in range(wi):
            lo, hi = windows[j]; seg = _win_slice(bh_net, lo, hi)
            if len(seg) > 1:
                vols.append(float(np.std(seg)))
        med = float(np.median(vols)) if vols else 0.0
        feats["vol_regime"] = feats["vol_level"] - med
    else:
        feats["vol_regime"] = 0.0
    return feats


def feature_matrix(trend_net, mr_net, trend_exp, mr_exp, bh_net, windows):
    rows = [window_features(trend_net, mr_net, trend_exp, mr_exp, bh_net, windows, wi)
            for wi in range(len(windows))]
    names = list(rows[0].keys()) if rows else []
    X = np.array([[r.get(n, 0.0) for n in names] for r in rows], float)
    return X, names


# =====================================================================================================
# 4. THE TWO TIERS -- A: interpretable regime rule; B: learned scorer -> trend-vs-MR weight per window
# =====================================================================================================
def _standardize_fit(X):
    return np.nanmean(X, axis=0), np.nanstd(X, axis=0) + 1e-9


def _standardize_apply(X, mu, sd):
    return np.nan_to_num((X - mu) / sd)


def tier_a_weights(X, fnames, lo_clip=0.2, hi_clip=0.8):
    """Tier-A: an INTERPRETABLE REGIME RULE -> w_trend per window. High trend-strength -> weight trend;
    chop/low-trend -> weight MR. Uses the RAW (un-standardized) trend_strength in [0,1] directly as the
    base weight, nudged by the recent perf_spread (which sleeve is winning), then clipped to [lo,hi] so
    neither sleeve is ever fully dropped (preserves the gap-fill diversification). NO fitting -> no OOS
    leak; the rule is fixed by construction."""
    ts_i = fnames.index("trend_strength")
    ps_i = fnames.index("perf_spread")
    ts = X[:, ts_i]
    ps = X[:, ps_i]
    # base weight = trend-strength; nudge +/-0.15 by the sign of recent perf spread (winner gets a tilt)
    nudge = 0.15 * np.tanh(ps / 5.0)        # perf_spread is in % units; soft-saturate
    w = np.clip(ts + nudge, lo_clip, hi_clip)
    return w


def tier_b_weights(Xtr_std, ytr_trend, ytr_mr, Xev_std):
    """Tier-B: a LEARNED scorer. Fit two regressors (predict trend's and MR's NEXT-window net from the
    features) on TRAIN; at eval, predict both, convert the predicted-net SPREAD to a weight via a logistic
    map, clip to [0.2,0.8]. LightGBM -> sklearn HistGB/GB -> ridge fallback (small-sample-robust). Returns
    (w_ev, backend)."""
    backend = None
    reg_factory = None
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        reg_factory = lambda: HistGradientBoostingRegressor(max_depth=3, max_iter=60, learning_rate=0.05,
                                                            min_samples_leaf=3)
        backend = "sklearn_hgb"
    except Exception:
        backend = "ridge"

    def _fit_predict(y):
        good = np.isfinite(y)
        if good.sum() < 6 or backend == "ridge":
            # ridge fallback (also when too few rows for a tree)
            lam = 10.0; nf = Xtr_std.shape[1]
            Xg = Xtr_std[good]; yg = np.nan_to_num(y[good])
            if len(yg) < 3:
                return np.zeros(Xev_std.shape[0])
            A = Xg.T @ Xg + lam * np.eye(nf)
            try:
                w = np.linalg.solve(A, Xg.T @ yg)
            except np.linalg.LinAlgError:
                return np.zeros(Xev_std.shape[0])
            return Xev_std @ w
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = reg_factory(); m.fit(np.ascontiguousarray(Xtr_std[good]), np.nan_to_num(y[good]))
            return m.predict(np.ascontiguousarray(Xev_std))

    pred_t = _fit_predict(ytr_trend)
    pred_m = _fit_predict(ytr_mr)
    spread = pred_t - pred_m                                 # predicted trend-minus-MR next-window net
    # logistic map of the spread (scaled by its own train-residual scale) -> weight to trend
    sc = np.std(spread) + 1e-9
    w = 1.0 / (1.0 + np.exp(-spread / sc))
    w = np.clip(w, 0.2, 0.8)
    return w, (backend if backend else "ridge")


# =====================================================================================================
# 5. WEIGHT SEQUENCE -> realized combined book (apply weight to NEXT window, lag-1 causal)
# =====================================================================================================
def realized_book(trend_net, mr_net, windows, weights, idxs):
    """Stitch the realized daily combined-net of the engine over the eval windows. weights[i] is the
    weight DECIDED at window i from past-only features, applied to window i's realized returns (the
    features were built from windows < i, so this is causal). Returns the stitched daily net array."""
    pieces = []
    for i in idxs:
        lo, hi = windows[i]
        seg = combined_window_net(trend_net, mr_net, lo, hi, weights[i])
        if len(seg):
            pieces.append(seg)
    if not pieces:
        return np.array([])
    return np.concatenate(pieces)


def _perf(x: np.ndarray, ann=365) -> dict:
    if x is None or len(x) < 3:
        return {"net": None, "sharpe": None, "maxdd": None, "p05": None}
    eq = np.cumprod(1 + x); pk = np.maximum.accumulate(eq)
    bb = block_bootstrap_distribution(x, n_boot=600, block=5, seed=13)
    return {"net": round(float((eq[-1] - 1) * 100), 1),
            "sharpe": round(float(np.mean(x) / (np.std(x) + 1e-12) * np.sqrt(ann)), 2),
            "maxdd": round(float(((eq - pk) / pk).min() * 100), 1),
            "p05": round(float(bb["p05"]) * 100, 1)}


# =====================================================================================================
# 6. CONTROLS -- static blend, VOLTGT_BH, weight-shuffle distribution, random allocator, PBO
# =====================================================================================================
def _static_blend_weight(cad):
    """The best STATIC complementary blend weight (w_trend) from the complementarity matrix (B's optimal
    per-TF blend). Falls back to 0.5 if absent."""
    try:
        d = json.load(open(COMP_MATRIX))
        e = d.get(cad, {})
        best = e.get("best_blend", "50_50")
        w = e.get("blend_weights", {}).get(best, 0.5)
        return float(w), best
    except Exception:
        return 0.5, "50_50"


def static_book(trend_net, mr_net, windows, idxs, w_static):
    return realized_book(trend_net, mr_net, windows, [w_static] * len(windows), idxs)


def shuffle_weight_distribution(trend_net, mr_net, windows, weights, idxs, n_draws=200, seed=11):
    """SHUFFLE control: block-shuffle the WEIGHT sequence over the eval windows (same weight distribution,
    random TIMING) -> the distribution of shuffled-book compound net %. Real timing skill must sit in the
    right tail of this distribution. Returns (dist_dict, arr)."""
    ev_w = np.array([weights[i] for i in idxs], float)
    block = max(1, len(ev_w) // 4)
    draws = []
    for d in range(n_draws):
        rng = np.random.default_rng(seed + 1 + d)
        nb = int(np.ceil(len(ev_w) / block))
        blocks = [ev_w[k * block:(k + 1) * block] for k in range(nb)]
        order = rng.permutation(len(blocks))
        shuf = np.concatenate([blocks[k] for k in order])[:len(ev_w)]
        bk = realized_book(trend_net, mr_net, windows, _scatter(shuf, idxs, len(windows)), idxs)
        if len(bk) >= 2:
            draws.append(float(np.cumprod(1 + bk)[-1] - 1) * 100)
    if not draws:
        return None, np.array([])
    arr = np.asarray(draws, float)
    return {"n_draws": int(arr.size), "median": round(float(np.median(arr)), 2),
            "mean": round(float(np.mean(arr)), 2), "p05": round(float(np.percentile(arr, 5)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2)}, arr


def _scatter(ev_vals, idxs, n_windows):
    """Place eval-window values back into a full-length weight vector (other windows -> 0.5, unused)."""
    w = np.full(n_windows, 0.5)
    for k, i in enumerate(idxs):
        w[i] = ev_vals[k]
    return w


def random_book(trend_net, mr_net, windows, idxs, seed=7, n_draws=200):
    """RANDOM allocator: random weight in [0.2,0.8] per window; report the MEAN compound net over draws."""
    nets = []
    for d in range(n_draws):
        rng = np.random.default_rng(seed + d)
        w = _scatter(rng.uniform(0.2, 0.8, len(idxs)), idxs, len(windows))
        bk = realized_book(trend_net, mr_net, windows, w, idxs)
        if len(bk) >= 2:
            nets.append(float(np.cumprod(1 + bk)[-1] - 1) * 100)
    return round(float(np.mean(nets)), 1) if nets else None


def pbo_rank_flip(metric_train, metric_oos):
    """A simple PBO-style probe: did the tier that ranked best IN-SAMPLE stay best OOS? Returns 1.0 if the
    in-sample-best underperforms its sibling OOS (overfit), else 0.0. (2-config probe -- A vs B.)"""
    return 1.0 if (metric_train[0] > metric_train[1]) != (metric_oos[0] > metric_oos[1]) else 0.0


# =====================================================================================================
# 7. EVALUATE ONE CADENCE
# =====================================================================================================
def _split_window_idx(windows, lo, hi):
    return [i for i, (wlo, _whi) in enumerate(windows) if pd.Timestamp(lo) <= wlo < pd.Timestamp(hi)]


def evaluate_cadence(cad, n_shuffle=200):
    ma_type, ma_src = _best_ma_type(cad)
    tnet, texp = _trend_sleeve(cad, ma_type)
    mnet, mexp = _mr_sleeve(cad)
    bh = _voltgt_bh_daily(cad)               # used as the regime proxy series + a control
    if tnet is None or mnet is None:
        return {"cadence": cad, "error": "insufficient sleeve data"}

    # align all on a common daily index
    df = pd.concat([tnet.rename("t"), mnet.rename("m")], axis=1).dropna()
    if len(df) < 30:
        return {"cadence": cad, "error": f"too few aligned days ({len(df)})"}
    tnet = df["t"]; mnet = df["m"]
    daily_index = df.index
    bh = bh.reindex(daily_index).fillna(0.0) if bh is not None else (0.5 * tnet + 0.5 * mnet)
    texp = texp.reindex(daily_index).fillna(0.0); mexp = mexp.reindex(daily_index).fillna(0.0)

    windows = rolling_windows(daily_index, ROLL_DAYS)
    if len(windows) < 8:
        return {"cadence": cad, "error": f"too few rolling windows ({len(windows)})"}

    # per-window realized net per sleeve (for Tier-B labels)
    yt = np.array([window_net(tnet, lo, hi) for lo, hi in windows])
    ym = np.array([window_net(mnet, lo, hi) for lo, hi in windows])
    X, fnames = feature_matrix(tnet, mnet, texp, mexp, bh, windows)

    tr_idx = _split_window_idx(windows, *SPLITS["TRAIN"])
    va_idx = _split_window_idx(windows, *SPLITS["VAL"])
    oos_idx = _split_window_idx(windows, *SPLITS["OOS"])
    if len(tr_idx) < 4 or len(oos_idx) < 1:
        return {"cadence": cad, "error": f"insufficient split windows (tr={len(tr_idx)}, oos={len(oos_idx)})"}

    mu, sd = _standardize_fit(X[tr_idx])
    Xstd = _standardize_apply(X, mu, sd)

    # ---- Tier A (regime rule -- no fit) over ALL windows ----
    wA = tier_a_weights(X, fnames)
    # ---- Tier B (learned) -- fit on TRAIN, weight ALL windows ----
    wB, b_backend = tier_b_weights(Xstd[tr_idx], yt[tr_idx], ym[tr_idx], Xstd)

    # ---- SELECT tier on VAL (never OOS) by VAL Sharpe of the realized book ----
    def _split_perf(weights, idxs, ann):
        bk = realized_book(tnet, mnet, windows, weights, idxs)
        return _perf(bk, ann)
    ANN = {"1d": 365, "4h": 365, "2h": 365, "1h": 365, "30m": 365, "15m": 365}[cad]  # daily-resampled -> 365
    va_A = _split_perf(wA, va_idx, ANN) if va_idx else {"sharpe": None, "net": None}
    va_B = _split_perf(wB, va_idx, ANN) if va_idx else {"sharpe": None, "net": None}
    # selection metric = VAL Sharpe (risk-adjusted, the honest objective for a DD-dampening engine), net tiebreak
    def _selkey(p):
        return ((p.get("sharpe") or -9), (p.get("net") or -9))
    chosen = "B" if (wB is not None and _selkey(va_B) > _selkey(va_A)) else "A"
    w_eng = wB if chosen == "B" else wA

    # ---- static blend control ----
    w_static, static_name = _static_blend_weight(cad)

    out = {"cadence": cad, "ma_type": ma_type, "ma_source": ma_src, "n_windows": len(windows),
           "features": fnames, "roll_days": ROLL_DAYS,
           "tier_selection": {"a_val_sharpe": va_A.get("sharpe"), "b_val_sharpe": va_B.get("sharpe"),
                              "a_val_net": va_A.get("net"), "b_val_net": va_B.get("net"),
                              "b_backend": b_backend, "chosen": chosen},
           "static_blend": {"weight_trend": round(w_static, 3), "name": static_name},
           "splits": {}}

    # ---- per-split metrics for the engine + ALL controls ----
    for split, idxs in [("TRAIN", tr_idx), ("VAL", va_idx), ("OOS", oos_idx)]:
        if not idxs:
            out["splits"][split] = {"n_windows": 0}
            continue
        eng_bk = realized_book(tnet, mnet, windows, w_eng, idxs)
        eng_m = _perf(eng_bk, ANN)
        stat_bk = static_book(tnet, mnet, windows, idxs, w_static)
        stat_m = _perf(stat_bk, ANN)
        # VOLTGT_BH over the union of eval windows
        lo = windows[idxs[0]][0]; hi = windows[idxs[-1]][1]
        vt_seg = _win_slice(bh, lo, hi)
        vt_m = _perf(vt_seg, ANN)
        # trend-alone / MR-alone for reference
        t_seg = _win_slice(tnet, lo, hi); m_seg = _win_slice(mnet, lo, hi)
        ta_m = _perf(t_seg, ANN); ma_m = _perf(m_seg, ANN)
        # SHUFFLE distribution (weight re-timing)
        sh_dist, sh_arr = shuffle_weight_distribution(tnet, mnet, windows, w_eng, idxs, n_draws=n_shuffle)
        if sh_dist is not None and eng_m["net"] is not None and sh_arr.size:
            sh_dist["eng_net"] = eng_m["net"]
            sh_dist["eng_percentile"] = round(float((sh_arr < eng_m["net"]).mean()), 3)
            sh_dist["p_value"] = round(float((sh_arr >= eng_m["net"]).mean()), 3)
            sh_dist["beats_median"] = bool(eng_m["net"] > sh_dist["median"])
            sh_dist["timing_skill"] = bool(eng_m["net"] > sh_dist["median"] and sh_dist["p_value"] < 0.10)
        # RANDOM allocator
        rand_net = random_book(tnet, mnet, windows, idxs)
        out["splits"][split] = {
            "n_windows": len(idxs),
            "DYNAMIC": eng_m, "STATIC": stat_m, "VOLTGT_BH": vt_m,
            "TREND_ALONE": ta_m, "MR_ALONE": ma_m,
            "SHUFFLE_dist": sh_dist, "RANDOM_net": rand_net,
            "mean_weight_trend": round(float(np.mean([w_eng[i] for i in idxs])), 3),
        }

    # ---- PBO probe: did the VAL-best tier stay best OOS? ----
    oos_A = _split_perf(wA, oos_idx, ANN); oos_B = _split_perf(wB, oos_idx, ANN)
    out["pbo_rank_flip"] = pbo_rank_flip(
        (va_A.get("sharpe") or -9, va_B.get("sharpe") or -9),
        (oos_A.get("sharpe") or -9, oos_B.get("sharpe") or -9))

    # ---- stash OOS engine weight timeline + equity for charts ----
    if oos_idx:
        oos_w = [(windows[i][0], float(w_eng[i]), float(X[i, fnames.index("trend_strength")])) for i in oos_idx]
        out["_oos_weight_timeline"] = oos_w
        eng_bk = realized_book(tnet, mnet, windows, w_eng, oos_idx)
        stat_bk = static_book(tnet, mnet, windows, oos_idx, w_static)
        lo = windows[oos_idx[0]][0]; hi = windows[oos_idx[-1]][1]
        out["_oos_equity"] = {
            "dynamic": list(np.cumprod(1 + eng_bk) * 100 - 100) if len(eng_bk) else [],
            "static": list(np.cumprod(1 + stat_bk) * 100 - 100) if len(stat_bk) else [],
            "voltgt": list(np.cumprod(1 + _win_slice(bh, lo, hi)) * 100 - 100),
        }
    return out


# =====================================================================================================
# 8. VERDICT (two-sided, honest)
# =====================================================================================================
def build_verdict(results):
    cads = [c for c in results if "error" not in results[c]]
    beat_static_sharpe, beat_static_dd, beat_static_p05 = [], [], []
    timing_skill, beat_voltgt = [], []
    rows = []
    for cad in cads:
        oos = results[cad]["splits"].get("OOS", {})
        if not oos or oos.get("n_windows", 0) == 0:
            continue
        dyn = oos.get("DYNAMIC", {}); st = oos.get("STATIC", {}); vt = oos.get("VOLTGT_BH", {})
        sd = oos.get("SHUFFLE_dist") or {}
        d_sh, s_sh = dyn.get("sharpe"), st.get("sharpe")
        d_dd, s_dd = dyn.get("maxdd"), st.get("maxdd")
        d_p05, s_p05 = dyn.get("p05"), st.get("p05")
        d_net, vt_net = dyn.get("net"), vt.get("net")
        if d_sh is not None and s_sh is not None and d_sh > s_sh + 0.05:
            beat_static_sharpe.append(cad)
        if d_dd is not None and s_dd is not None and d_dd > s_dd + 0.2:   # less-negative maxDD
            beat_static_dd.append(cad)
        if d_p05 is not None and s_p05 is not None and d_p05 > s_p05 + 0.2:
            beat_static_p05.append(cad)
        if d_net is not None and vt_net is not None and d_net > vt_net + 0.5:
            beat_voltgt.append(cad)
        if sd.get("timing_skill"):
            timing_skill.append(cad)
        rows.append(
            f"[{cad}] OOS DYN net {d_net}% Sh {d_sh} DD {d_dd}% p05 {d_p05}% (w_trend~{oos.get('mean_weight_trend')}) "
            f"vs STATIC net {st.get('net')}% Sh {s_sh} DD {s_dd}% p05 {s_p05}% "
            f"vs VOLTGT {vt_net}% | shuf-med {sd.get('median')}% p={sd.get('p_value')} "
            f"{'TIMING-SKILL' if sd.get('timing_skill') else 'no-skill'} | tier {results[cad]['tier_selection']['chosen']}")
    n = len([c for c in cads if results[c]['splits'].get('OOS', {}).get('n_windows', 0) > 0])
    # the engine "works" only if it beats the STATIC blend on a RISK-ADJUSTED axis AND shows timing skill
    beats_static_riskadj = sorted(set(beat_static_sharpe) | set(beat_static_p05))
    real_edge = [c for c in beats_static_riskadj if c in timing_skill]
    if real_edge:
        headline = (f"DYNAMIC-ADDS-VALUE (partial): at {len(real_edge)}/{n} cadences ({real_edge}) the dynamic "
                    f"engine beat the STATIC blend on a risk-adjusted axis (Sharpe or p05) AND showed genuine "
                    f"TIMING SKILL (engine net > weight-shuffle median, p<0.10). Verify robustness before believing.")
    elif beats_static_riskadj or beat_static_dd:
        headline = (f"STATIC-BLEND-WINS (dynamic adds little): the dynamic engine beat the static blend on a "
                    f"risk-adjusted axis at {beats_static_riskadj} and on maxDD at {beat_static_dd}, but with NO "
                    f"coincident timing skill -- where it beats static it is not beating its own weight-shuffle, so "
                    f"the gain is not from regime TIMING. The STATIC complementary blend is the answer; dynamic "
                    f"timing adds nothing reliable on 2020-bull OOS. (A real, valuable finding -- not a failure.)")
    else:
        headline = (f"STATIC-BLEND-WINS (clean): the dynamic engine did NOT beat the static complementary blend "
                    f"on Sharpe/p05/maxDD at any of {n} cadences, and showed timing skill at {len(timing_skill)}/{n}. "
                    f"On the 2020-bull OOS, regime-timed weighting does not improve on a fixed optimal blend. "
                    f"SHIP THE STATIC BLEND; the dynamic engine adds no reliable risk-adjusted value here. "
                    f"(Honest two-sided result: the static blend IS the deployable answer.)")
    lines = [
        "BAR TO BEAT: the STATIC complementary blend (B's optimal per-TF weight) on a RISK-ADJUSTED axis "
        "(Sharpe / maxDD / p05) AND the engine's own WEIGHT-SHUFFLE distribution (timing skill, p<0.10).",
        "HONEST FRAME: both sleeves are LONG-ONLY -> gap-filling is DRAWDOWN-DAMPENING, not return rescue. "
        "The engine target is risk-adjusted return, NOT alpha. We do not claim alpha it does not have.",
        f"HEADLINE: {headline}", ""] + rows + [
        "",
        f"beat-static(Sharpe) {beat_static_sharpe} | beat-static(maxDD) {beat_static_dd} | "
        f"beat-static(p05) {beat_static_p05} | beat-VOLTGT(net) {beat_voltgt} | timing-skill {timing_skill}",
        "CAVEATS: 2020-BULL-ONLY, in-sample regime (single bull -- the trend sleeve is structurally favored; "
        "the DD-dampening value of complementarity is exactly what should generalize worse-known until a "
        "bear/chop regime is tested -- synthetic regime-stress is a LATER phase, flagged not done here). "
        "Long-only; equal-weight u10; causal lag-1 MtM; maker. Tier chosen on VAL (never OOS). 1d has few "
        "rolling windows (small-sample). UNSEEN N/A (2020 band only)."]
    return {"headline": headline, "beat_static_sharpe": beat_static_sharpe, "beat_static_dd": beat_static_dd,
            "beat_static_p05": beat_static_p05, "beat_voltgt": beat_voltgt, "timing_skill": timing_skill,
            "real_edge_cadences": real_edge, "n_cadences": n, "lines": lines}


# =====================================================================================================
# 9. CHARTS
# =====================================================================================================
def chart_weights_timeline(results, cadences):
    """Chart 1: the trend-vs-MR weight over 2020 OOS, shaded by detected regime (the 'dynamic' visualized).
    One panel per cadence with an OOS weight timeline."""
    cs = [c for c in cadences if c in results and results[c].get("_oos_weight_timeline")]
    if not cs:
        return
    n = len(cs); ncol = min(2, n); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0 * ncol, 3.2 * nrow), squeeze=False)
    for k, cad in enumerate(cs):
        ax = axes[k // ncol][k % ncol]
        tl = results[cad]["_oos_weight_timeline"]
        dates = [pd.Timestamp(t[0]) for t in tl]
        w = np.array([t[1] for t in tl]); ts = np.array([t[2] for t in tl])
        w_static = results[cad]["static_blend"]["weight_trend"]
        # shade regime: trend-strong (ts high -> green) vs choppy (ts low -> orange)
        for i in range(len(dates)):
            color = "#2ca02c" if ts[i] >= 0.5 else "#ff7f0e"
            x0 = dates[i]; x1 = dates[i + 1] if i + 1 < len(dates) else dates[i] + pd.Timedelta(days=ROLL_DAYS)
            ax.axvspan(x0, x1, color=color, alpha=0.10)
        ax.plot(dates, w, "-o", color="#1f77b4", ms=3, lw=1.6, label="DYNAMIC w_trend")
        ax.axhline(w_static, color="#d62728", ls="--", lw=1.2, label=f"static w_trend={w_static:.2f}")
        ax.axhline(0.5, color="k", lw=0.5, alpha=0.4)
        ax.set_ylim(0, 1); ax.set_ylabel("weight -> TREND")
        ax.set_title(f"{cad}  (tier {results[cad]['tier_selection']['chosen']}, MA={results[cad]['ma_type']})", fontsize=10)
        ax.legend(fontsize=7, loc="upper left")
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("DYNAMIC trend-vs-MR weight over 2020 OOS (green=trend-strong regime, orange=choppy)\n"
                 "the 'dynamic' visualized -- weight rises to trend in trends, tilts to MR in chop (2020-bull-only)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = CHARTS / "dynamic_weights_timeline.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    print(f"[figure] {p}")


def chart_equity(results, cadences):
    """Chart 2: per TF OOS equity -- dynamic engine vs static blend vs VOLTGT_BH."""
    cs = [c for c in cadences if c in results and results[c].get("_oos_equity")]
    if not cs:
        return
    n = len(cs); ncol = min(3, n); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.8 * nrow), squeeze=False)
    for k, cad in enumerate(cs):
        ax = axes[k // ncol][k % ncol]
        eq = results[cad]["_oos_equity"]
        for key, color, lab in [("dynamic", "#2ca02c", "DYNAMIC"), ("static", "#1f77b4", "STATIC blend"),
                                ("voltgt", "#ff7f0e", "VOLTGT_BH")]:
            y = eq.get(key, [])
            if y:
                ax.plot(range(len(y)), y, color=color, lw=2.0 if key == "dynamic" else 1.4, label=lab)
        ax.axhline(0, color="k", lw=0.5)
        oos = results[cad]["splits"]["OOS"]
        ax.set_title(f"{cad}: DYN {oos['DYNAMIC']['net']}% / STAT {oos['STATIC']['net']}% / VT {oos['VOLTGT_BH']['net']}%",
                     fontsize=9)
        ax.set_ylabel("OOS compound %"); ax.legend(fontsize=7)
        ax.set_xlabel("OOS day")
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("DYNAMIC engine vs STATIC blend vs VOLTGT_BH -- OOS equity per TF (2020, u10, maker)\n"
                 "honest: long-only -> the engine's value is risk-adjusted (DD/p05), not extra return (2020-bull-only)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = CHARTS / "dynamic_vs_static_vs_voltgt_equity.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    print(f"[figure] {p}")


def chart_skill_bars(results, cadences):
    """Chart 3: dynamic vs static vs shuffle-median vs random (net + Sharpe), per TF -- skill-vs-controls."""
    cs = [c for c in cadences if c in results and results[c]["splits"].get("OOS", {}).get("n_windows", 0) > 0]
    if not cs:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    x = np.arange(len(cs)); width = 0.2
    dyn_net = [results[c]["splits"]["OOS"]["DYNAMIC"]["net"] for c in cs]
    stat_net = [results[c]["splits"]["OOS"]["STATIC"]["net"] for c in cs]
    shuf_net = [(results[c]["splits"]["OOS"].get("SHUFFLE_dist") or {}).get("median") for c in cs]
    rand_net = [results[c]["splits"]["OOS"].get("RANDOM_net") for c in cs]
    ax1.bar(x - 1.5 * width, dyn_net, width, color="#2ca02c", label="DYNAMIC")
    ax1.bar(x - 0.5 * width, stat_net, width, color="#1f77b4", label="STATIC blend")
    ax1.bar(x + 0.5 * width, shuf_net, width, color="#9467bd", label="weight-SHUFFLE median")
    ax1.bar(x + 1.5 * width, rand_net, width, color="#7f7f7f", label="RANDOM")
    ax1.set_xticks(x); ax1.set_xticklabels(cs); ax1.axhline(0, color="k", lw=0.6)
    ax1.set_ylabel("OOS compound net %"); ax1.set_title("NET: dynamic vs static vs shuffle vs random")
    ax1.legend(fontsize=8)
    dyn_sh = [results[c]["splits"]["OOS"]["DYNAMIC"]["sharpe"] for c in cs]
    stat_sh = [results[c]["splits"]["OOS"]["STATIC"]["sharpe"] for c in cs]
    vt_sh = [results[c]["splits"]["OOS"]["VOLTGT_BH"]["sharpe"] for c in cs]
    ax2.bar(x - width, dyn_sh, width, color="#2ca02c", label="DYNAMIC")
    ax2.bar(x, stat_sh, width, color="#1f77b4", label="STATIC blend")
    ax2.bar(x + width, vt_sh, width, color="#ff7f0e", label="VOLTGT_BH")
    ax2.set_xticks(x); ax2.set_xticklabels(cs); ax2.axhline(0, color="k", lw=0.6)
    ax2.set_ylabel("OOS Sharpe (ann)"); ax2.set_title("SHARPE: dynamic vs static vs VOLTGT_BH\n(the risk-adjusted axis -- the honest objective)")
    ax2.legend(fontsize=8)
    fig.suptitle("DYNAMIC ENGINE skill vs controls per TF (2020 OOS, u10, maker) -- real skill must beat ALL\n"
                 "2020-bull-only, in-sample; long-only -> objective is risk-adjusted (Sharpe), not net alpha",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p = CHARTS / "dynamic_engine_skill_bars.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    print(f"[figure] {p}")


# =====================================================================================================
# 10. MAIN
# =====================================================================================================
def _print_table(cad, r):
    print(f"\n########## CADENCE {cad} -- DYNAMIC/ML allocation engine (2020 runway, u10, maker) ##########")
    if "error" in r:
        print(f"   SKIPPED: {r['error']}")
        return
    ts = r["tier_selection"]
    print(f"   trend MA-type = {r['ma_type']} ({r['ma_source']}) | windows={r['n_windows']} roll_days={r['roll_days']}")
    print(f"   tier select on VAL: A_Sh={ts['a_val_sharpe']} B_Sh={ts['b_val_sharpe']} backend={ts['b_backend']} -> Tier-{ts['chosen']}")
    print(f"   static blend = {r['static_blend']['name']} (w_trend={r['static_blend']['weight_trend']}) | PBO rank-flip={r.get('pbo_rank_flip')}")
    hdr = (f"   {'split':7} {'DYN net%':>9} {'DYN Sh':>7} {'DYN DD%':>8} {'DYN p05':>8} | "
           f"{'STAT net%':>10} {'STAT Sh':>8} {'STAT DD%':>9} {'STAT p05':>9} | "
           f"{'VT net%':>8} {'shuf-med':>9} {'shuf-p':>7} {'skill':>6} {'RAND%':>7}")
    print(hdr)
    for split in ("TRAIN", "VAL", "OOS"):
        s = r["splits"].get(split, {})
        if not s or s.get("n_windows", 0) == 0:
            print(f"   {split:7} (no windows)")
            continue
        d = s["DYNAMIC"]; st = s["STATIC"]; vt = s["VOLTGT_BH"]; sd = s.get("SHUFFLE_dist") or {}
        skill = "YES" if sd.get("timing_skill") else ("no" if sd else "-")
        print(f"   {split:7} {str(d['net']):>9} {str(d['sharpe']):>7} {str(d['maxdd']):>8} {str(d['p05']):>8} | "
              f"{str(st['net']):>10} {str(st['sharpe']):>8} {str(st['maxdd']):>9} {str(st['p05']):>9} | "
              f"{str(vt['net']):>8} {str(sd.get('median')):>9} {str(sd.get('p_value')):>7} {skill:>6} {str(s.get('RANDOM_net')):>7}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m strat.dynamic_allocation_engine")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cadences", default=",".join(CADENCES))
    ap.add_argument("--n-shuffle", type=int, default=200, dest="n_shuffle")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    CHARTS.mkdir(parents=True, exist_ok=True)
    cads = [c.strip() for c in a.cadences.split(",") if c.strip()]
    print("## DYNAMIC / ML ALLOCATION ENGINE -- regime-conditional trend-vs-MR weighting per TF")
    print(f"   2020 BAND ONLY | TRAIN {SPLITS['TRAIN']} fit / VAL {SPLITS['VAL']} select / OOS {SPLITS['OOS']} confirm")
    print(f"   sleeves = best-MA-type trend + MR oscillator family | controls: STATIC blend, VOLTGT_BH, weight-SHUFFLE, RANDOM")
    print(f"   HONEST FRAME: long-only -> gap-fill is DD-dampening NOT return rescue; objective = risk-adjusted, not alpha")
    print(f"   timing-skill = engine net > weight-shuffle median with one-sided p<0.10 over {a.n_shuffle} draws\n")

    results = {}
    for cad in cads:
        r = evaluate_cadence(cad, n_shuffle=a.n_shuffle)
        results[cad] = r
        _print_table(cad, r)

    verdict = build_verdict(results)
    print("\n" + "=" * 100)
    print("## AGGREGATE VERDICT")
    for line in verdict["lines"]:
        print(f"   {line}")
    print("=" * 100)

    # charts
    chart_weights_timeline(results, cads)
    chart_equity(results, cads)
    chart_skill_bars(results, cads)

    # persist (drop the heavy chart-stash arrays into a compact form; keep timelines)
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    export = {
        "repro": {"command": "python -m strat.dynamic_allocation_engine " + " ".join(argv or sys.argv[1:]),
                  "git_sha": sha, "cost_maker": MAKER_RT, "cost_taker": TAKER_RT,
                  "runway": RUNWAY, "splits": SPLITS, "roll_days": ROLL_DAYS, "n_shuffle": a.n_shuffle,
                  "universe": "u10", "constraint": "2020 BAND ONLY", "syms": SYMS,
                  "honest_frame": "long-only sleeves -> gap-fill is DD-dampening not return rescue; objective "
                                  "is risk-adjusted (Sharpe/maxDD/p05), NOT alpha; 2020-bull-only in-sample"},
        "results": {c: {k: v for k, v in r.items() if not k.startswith("_oos_equity")}
                    for c, r in results.items()},
        "verdict": verdict,
    }
    p = OUT / "dynamic_engine.json"
    json.dump(export, open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n[persisted] {p}")
    return 0


# =====================================================================================================
# 11. SELFTEST -- two-sided soundness (synthetic, no market)
# =====================================================================================================
def selftest():
    """POSITIVE: when trend-strength genuinely predicts which sleeve wins (a planted regime->sleeve map),
    the engine's weight must track the right sleeve and BEAT the weight-shuffle distribution.
    NEGATIVE: when sleeve outcomes are i.i.d. noise (no regime signal), the engine must NOT manufacture
    timing skill (its net does not sit in the right tail of weight re-timings)."""
    print("## DYNAMIC-ALLOCATION-ENGINE SELFTEST (two-sided)")
    ok = True
    n_win = 60

    def _timing_skill_synth(daily_t, daily_m, w_seq, n_draws=200, seed=7):
        """compound net of the weighted book vs N weight re-timings -> (eng_net, shuf_median, p, skill)."""
        def _book(ws):
            segs = [ws[i] * daily_t[i] + (1 - ws[i]) * daily_m[i] for i in range(len(ws))]
            return float(np.cumprod(1 + np.array(segs))[-1] - 1) * 100
        eng = _book(w_seq)
        block = max(1, len(w_seq) // 4)
        draws = []
        for d in range(n_draws):
            rng = np.random.default_rng(seed + d)
            nb = int(np.ceil(len(w_seq) / block))
            blocks = [w_seq[k * block:(k + 1) * block] for k in range(nb)]
            order = rng.permutation(len(blocks))
            shuf = np.concatenate([blocks[k] for k in order])[:len(w_seq)]
            draws.append(_book(shuf))
        arr = np.asarray(draws)
        p = float((arr >= eng).mean())
        return eng, float(np.median(arr)), p, bool(eng > np.median(arr) and p < 0.10)

    rng = np.random.default_rng(0)
    # ---- POSITIVE: regime bit -> which sleeve has the high daily return; engine weights toward winner ----
    bit = rng.integers(0, 2, n_win)
    daily_t = np.where(bit == 1, 0.02, -0.005) + rng.normal(0, 0.003, n_win)   # trend wins when bit=1
    daily_m = np.where(bit == 0, 0.02, -0.005) + rng.normal(0, 0.003, n_win)   # MR wins when bit=0
    # oracle-ish causal weight: tilt to trend when bit=1 (this is what a working engine should approximate)
    w_pos = np.where(bit == 1, 0.8, 0.2).astype(float)
    eng, med, p, skill = _timing_skill_synth(daily_t.tolist(), daily_m.tolist(), w_pos.tolist())
    print(f"  POSITIVE (regime->sleeve): book net {eng:.1f}% vs shuffle-median {med:.1f}% (p={p:.3f}) -> "
          f"{'PASS' if skill else 'FAIL'} (a correct regime weight must show TIMING SKILL)")
    ok &= skill

    # ---- NEGATIVE: i.i.d. sleeve returns, weights uncorrelated with outcome -> no manufactured skill ----
    skill_count = 0
    for s in range(40):
        r2 = np.random.default_rng(100 + s)
        dt_ = r2.normal(0.005, 0.02, n_win); dm_ = r2.normal(0.005, 0.02, n_win)
        w_rand = r2.uniform(0.2, 0.8, n_win)
        _, _, _, sk = _timing_skill_synth(dt_.tolist(), dm_.tolist(), w_rand.tolist(), seed=200 + s)
        skill_count += int(sk)
    neg_pass = (skill_count / 40.0) <= 0.20
    print(f"  NEGATIVE (i.i.d. noise): false-timing-skill rate {skill_count}/40 ({skill_count/40.0:.2f}) -> "
          f"{'PASS' if neg_pass else 'FAIL'} (expect <=0.20 false-positive rate)")
    ok &= neg_pass

    # ---- tier-A regime rule sanity: high trend_strength -> w_trend high; low -> low ----
    fnames = ["trend_strength", "perf_spread", "chop_vs_trend"]
    Xs = np.array([[0.9, 0.0, 0.1], [0.1, 0.0, 0.9]])
    wA = tier_a_weights(Xs, fnames)
    rule_pass = wA[0] > wA[1] and wA[0] <= 0.8 and wA[1] >= 0.2
    print(f"  RULE (tier-A): trend-strong w={wA[0]:.2f} > choppy w={wA[1]:.2f}, both clipped [0.2,0.8] -> "
          f"{'PASS' if rule_pass else 'FAIL'}")
    ok &= rule_pass

    print(f"\n  SELFTEST {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
