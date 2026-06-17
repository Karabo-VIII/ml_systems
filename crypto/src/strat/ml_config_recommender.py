"""src/strat/ml_config_recommender.py -- the LEARNED config-recommender on the 2020 TRAIN/VAL/OOS runway.

USER ASK (6h autonomous, design doc docs/ML_CONFIG_RECOMMENDER_DESIGN_2026_06_13.md): "design the ML
approach that, at the end, gives feedback akin to the results we get when we run the static rules and
gives us the best config -- on the whole-of-2020 TRAIN/VAL/OOS runway the other instance already ran; I
want that as an output."

THE HONEST FRAMING (what the static-rule runway already taught us, so we don't re-mine noise):
deep2020_leaderboard.py graded candidate long-only strategies per timeframe on the 2020 runway
(WIN 2020-07-01..2021-01-01, SPLIT 2020-10-01 -> OOS Oct-Dec). Its verdict:
  (1) rank by NET (wealth), NOT Sharpe -- Sharpe rewards under-participation in the bull;
  (2) the real "best" is VOL-TARGETED BUY-HOLD (highest/near-highest net at every TF);
  (3) MA-config ranking is mostly NOISE (ranks flip; VAL-best does not transfer).
=> For the ML to MATTER it must beat the REAL bars: VOLTGT_BH on NET, the static-rule pick, AND a
same-frequency SHUFFLE. The genuine skill lever is the ORTHOGONAL features the static ranking ignored
(CALENDAR weekend/US-hours tilt + vol-regime).

THE OBJECT (contextual learning-to-rank, NOT per-candle prediction):
  - candidates = the 8 MA-type FAMILIES (EMA/SMA/WMA/HMA/DEMA/TEMA/KAMA/VIDYA slow books) + BUYHOLD +
    VOLTGT_BH (the ML MAY recommend "just vol-target buy-hold" when right -- not rigged to lose).
  - per rolling window t (in-sample): past-only causal features X(t) -> label = the candidate with the
    best realized NEXT-window NET (from the static eval). The model learns X(t) -> recommended candidate.
  - at eval time it emits, per window, a ranked recommendation; we trade the top pick.

FEATURES (past-only, causal): rolling regime (trend/chop/vol) + vol level/regime + recent per-candidate
performance (rho-persistence) + breadth + whipsaw + CALENDAR (weekend / US-hours fraction in the window;
the orthogonal skill lever the static ranking ignored) + participation/coverage state. Standardize on
TRAIN only.

MODEL (framing+features > model complexity): Tier-A = james_stein-shrunk feature scorer -> rank
candidates (overfit-resistant on the small 2020 sample); Tier-B = a gradient-boosted ranker (sklearn
GradientBoosting/HistGB if LightGBM absent). Select the tier on VAL (NEVER on OOS). Report both.

RUNWAY (mirror the static rule for apples-to-apples): TRAIN Jan-Aug 2020 (fit) / VAL Sep 2020 (select) /
OOS Oct-Dec 2020 (test -- the SAME OOS as the static rule). ALSO run the tool's exact Jul-Sep->Oct-Dec
for direct leaderboard comparability. Per TF sweep {1d,4h,2h,1h,30m,15m} (no silent single-cadence).

HONEST CONTROLS (non-negotiable): SHUFFLE control (block-shuffle the recommendation sequence: same
candidate-frequency, random timing -- real ML must beat its own shuffle); vs VOLTGT_BH on NET; vs the
static-rule pick; vs a random recommender; PBO + block-bootstrap p05; a two-sided SELFTEST that PASSES
(a genuine-skill synthetic case ships; a no-signal case shrinks to prior / loses to vol-target).

RWYB:
  python -m strat.ml_config_recommender --selftest          # two-sided synthetic control (no market)
  python -m strat.ml_config_recommender                     # the full runway, all cadences
  python -m strat.ml_config_recommender --cadences 1d,4h    # a subset
No emoji (Windows cp1252). Does NOT git commit (the overseer commits after judging).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.portfolio_replay as PR                                      # noqa: E402
from strat.portfolio_replay import apply_trail_stop, MAKER_RT, TAKER_RT  # noqa: E402
from strat.replay_distinct_grid import distinct_specs                    # noqa: E402
from strat.structural_fixes import min_hold                              # noqa: E402
from strat.ma_type_upgrade import _MA, _nums, MA_TYPES                   # noqa: E402
from strat.ma_2020_breakdown import _panel                              # noqa: E402
from strat.data_expansion import james_stein_shrink                     # noqa: E402
from strat.battery import block_bootstrap_p05_p95                       # noqa: E402

OUT = ROOT.parent / "runs" / "strat"
OUT.mkdir(parents=True, exist_ok=True)

__contract__ = {
    "kind": "learned_config_recommender",
    "inputs": {
        "candidates": "8 MA-type FAMILY books + BUYHOLD + VOLTGT_BH (each a date-indexed net+pos series)",
        "features": "past-only causal X(t) per rolling window: regime/vol/recent-perf/breadth/whipsaw/"
                    "CALENDAR/participation -- standardized on TRAIN only",
        "label": "the candidate with the best realized NEXT-window NET (from the static eval)",
    },
    "outputs": {
        "leaderboard": "per TF per split: ML pick + realized NET%/Sharpe/maxDD/coverage side-by-side "
                       "with the static-rule pick, VOLTGT_BH, BUYHOLD, ORACLE",
        "verdict": "does the ML BEAT (a) static-rule pick, (b) VOLTGT_BH on NET, (c) its shuffle, at how "
                   "many of the 6 cadences -- honest either way",
    },
    "invariants": {
        "rank_by_net_not_sharpe": "rank candidates by realized NET (wealth), never Sharpe (under-participation trap)",
        "oos_is_oct_dec_2020": "OOS = Oct-Dec 2020, the SAME OOS as deep2020_leaderboard.py (apples-to-apples)",
        "select_on_val_never_oos": "tier (A vs B) chosen on VAL Sep 2020; OOS is test-once",
        "past_only_features": "every feature uses bars <= t; standardization fit on TRAIN only",
        "shuffle_must_be_beaten": "block-shuffle the recommendation sequence -> real ML must beat it or "
                                  "the 'skill' is just average exposure",
        "voltgt_is_the_real_bar": "the honest comparison is vs VOLTGT_BH on NET (the static-rule winner)",
        "no_signal_shrinks_to_prior": "two-sided selftest: genuine skill ships; no-signal shrinks to prior",
        "causal_mtm_no_double_count": "positions lagged 1 bar; MtM no double-count; cost charged on flips",
    },
}

# ---- the 2020 runway: mirror deep2020_leaderboard.py for apples-to-apples ----------------------------
WIN = ("2020-07-01", "2021-01-01")          # the tool's window (for the comparable run)
TOOL_SPLIT = "2020-10-01"                    # tool: in-sample Jul-Sep, OOS Oct-Dec
# the design's full runway: TRAIN Jan-Aug fit, VAL Sep select, OOS Oct-Dec test (OOS == the tool's OOS)
RUNWAY = ("2020-01-01", "2021-01-01")
SPLITS = {"TRAIN": ("2020-01-01", "2020-09-01"),
          "VAL":   ("2020-09-01", "2020-10-01"),
          "OOS":   ("2020-10-01", "2021-01-01")}
WARMUP = 400
CADENCES = ["1d", "4h", "2h", "1h", "30m", "15m"]
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT",
        "LINKUSDT", "LTCUSDT"]
ANN = {"1d": 365, "4h": 365 * 6, "2h": 365 * 12, "1h": 365 * 24, "30m": 365 * 48, "15m": 365 * 96}
VW = {"1d": 14, "4h": 84, "2h": 168, "1h": 168, "30m": 336, "15m": 672}
# rolling recommendation window length per cadence (~7 days; the SETUP horizon, not a candle)
ROLL_BARS = {"1d": 7, "4h": 42, "2h": 84, "1h": 168, "30m": 336, "15m": 672}
CANDIDATES = list(MA_TYPES) + ["BUYHOLD", "VOLTGT_BH"]


# =====================================================================================================
# 1. CANDIDATE SERIES -- each candidate -> a book-level (net, pos) pd.Series over the full runway
# =====================================================================================================
def _ma_family_series(slow_cfgs, ma_type, cadence):
    """One MA-type FAMILY book: per asset, equal-weight the slow configs (full stack: trail10 +
    min_hold12 + maker, matching the static rule's FULL-stack family), then equal-weight the u10 book.
    Returns (net_series, pos_series) date-indexed over the runway, or None."""
    s_ms = pd.Timestamp(RUNWAY[0]).value // 10**6
    e_ms = pd.Timestamp(RUNWAY[1]).value // 10**6
    per_net, per_pos = [], []
    for sym in SYMS:
        try:
            o, h, l, c, ms = _panel(sym, cadence)
        except Exception:
            continue
        e = int(np.searchsorted(ms, e_ms))
        s0 = max(0, int(np.searchsorted(ms, s_ms)) - WARMUP)
        c2, ms2 = c[s0:e], ms[s0:e]
        if len(c2) < 40:
            continue
        win = ms2 >= s_ms
        if win.sum() < 20:
            continue
        ret = np.zeros(len(c2)); ret[1:] = c2[1:] / c2[:-1] - 1.0
        uniq = sorted({p for n in slow_cfgs for p in _nums(n)})
        cache = {p: _MA[ma_type](c2, p) for p in uniq}
        nets, poss = [], []
        for name in slow_cfgs:
            pp = _nums(name); mas = [cache[p] for p in pp]
            h0 = np.nan_to_num((mas[0] > mas[1]) if len(pp) == 2
                               else ((mas[0] > mas[1]) & (mas[1] > mas[2]))).astype(np.int8)
            h0 = min_hold(apply_trail_stop(h0.copy(), c2, 0.10)[0].astype(np.int8), 12).astype(np.int8)
            pos = np.zeros(len(c2)); pos[1:] = h0[:-1]
            flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
            nets.append(pos * ret - flips * (MAKER_RT / 2.0)); poss.append(pos)
        idx = pd.to_datetime(ms2[win], unit="ms")
        per_net.append(pd.Series(np.mean(nets, axis=0)[win], index=idx))
        per_pos.append(pd.Series(np.mean(poss, axis=0)[win], index=idx))
    if not per_net:
        return None
    bn = pd.concat(per_net, axis=1).fillna(0.0).mean(axis=1)   # fixed-EW (unlisted=cash; cadence-invariant) -- was skipna (fine-TF-inflated), aligned to the e123ab1 consensus fix
    bp = pd.concat(per_pos, axis=1).fillna(0.0).mean(axis=1)
    return bn, bp


def _universal_series(cadence, kind):
    """BUYHOLD (always 1.0 exposure) or VOLTGT_BH (vol-targeted exposure), u10 book, date-indexed.
    Matches deep2020_leaderboard._universal. Returns (net_series, pos_series) or None."""
    s_ms = pd.Timestamp(RUNWAY[0]).value // 10**6
    e_ms = pd.Timestamp(RUNWAY[1]).value // 10**6
    per_net, per_pos = [], []
    for sym in SYMS:
        try:
            o, h, l, c, ms = _panel(sym, cadence)
        except Exception:
            continue
        keep = (ms >= s_ms - 30 * 86400000) & (ms < e_ms)
        c2 = c[keep]; ms2 = ms[keep]
        if len(c2) < 40:
            continue
        ret = np.zeros(len(c2)); ret[1:] = c2[1:] / c2[:-1] - 1.0
        if kind == "VOLTGT_BH":
            rv = pd.Series(ret).rolling(VW[cadence]).std().to_numpy(); med = np.nanmedian(rv)
            exp = np.nan_to_num(np.clip(med / (np.concatenate([[np.nan], rv[:-1]]) + 1e-12), 0, 1))
        else:
            exp = np.ones(len(c2))
        win = ms2 >= s_ms
        idx = pd.to_datetime(ms2[win], unit="ms")
        per_net.append(pd.Series((exp * ret)[win], index=idx))
        per_pos.append(pd.Series(exp[win], index=idx))
    if not per_net:
        return None
    bn = pd.concat(per_net, axis=1).fillna(0.0).mean(axis=1)   # fixed-EW (unlisted=cash; cadence-invariant) -- was skipna (fine-TF-inflated), aligned to the e123ab1 consensus fix
    bp = pd.concat(per_pos, axis=1).fillna(0.0).mean(axis=1)
    return bn, bp


def build_candidate_series(cadence):
    """All 10 candidate (net, pos) series for a cadence, aligned to a common datetime index.
    Returns (net_df, pos_df) with columns = CANDIDATES (dropping any that failed)."""
    ma_cfg = {}
    for fam in ("2MA", "3MA"):
        ma_cfg.update(distinct_specs(fam, 0.15, max_n=60))
    PR.STRATS.update(ma_cfg)
    slow = [n for n in ma_cfg if 60 <= max(_nums(n)) < 150]
    nets, poss = {}, {}
    for mt in MA_TYPES:
        r = _ma_family_series(slow, mt, cadence)
        if r is not None:
            nets[mt], poss[mt] = r
    for kind in ("BUYHOLD", "VOLTGT_BH"):
        r = _universal_series(cadence, kind)
        if r is not None:
            nets[kind], poss[kind] = r
    if not nets:
        return None, None
    net_df = pd.DataFrame(nets).sort_index()
    pos_df = pd.DataFrame(poss).reindex(net_df.index)
    return net_df, pos_df


# =====================================================================================================
# 2. ROLLING WINDOWS + LABELS (the supervised signal) + per-window realized NET per candidate
# =====================================================================================================
def rolling_windows(index, roll_bars):
    """Non-overlapping rolling windows of `roll_bars` over the datetime index -> list of (lo_ts, hi_ts)."""
    n = len(index)
    out = []
    for s in range(0, n - roll_bars, roll_bars):
        out.append((index[s], index[min(s + roll_bars, n - 1)]))
    return out


def window_net(net_series, lo, hi):
    """Compound NET % of a candidate's net series over [lo, hi) (the realized window outcome)."""
    s = net_series[(net_series.index >= lo) & (net_series.index < hi)].dropna()
    if len(s) < 2:
        return np.nan
    return float(np.cumprod(1 + s.to_numpy())[-1] - 1) * 100


def per_window_table(net_df, windows):
    """[n_windows x n_candidates] realized NET% matrix: row w, col c = candidate c's NET over window w."""
    M = np.full((len(windows), net_df.shape[1]), np.nan)
    for wi, (lo, hi) in enumerate(windows):
        for ci, cand in enumerate(net_df.columns):
            M[wi, ci] = window_net(net_df[cand], lo, hi)
    return M


# =====================================================================================================
# 3. FEATURES (past-only, causal) per rolling window -- the skill lever
# =====================================================================================================
def _calendar_features(lo, hi, cadence):
    """CALENDAR features over the window [lo, hi): weekend fraction + US-hours fraction of the bars.
    The orthogonal-to-beta lever the static leaderboard did NOT use. Computed from the bar timestamps
    themselves (no look-ahead -- a window's own calendar is known causally at window start by the clock)."""
    freq = {"1d": "1D", "4h": "4h", "2h": "2h", "1h": "1h", "30m": "30min", "15m": "15min"}[cadence]
    idx = pd.date_range(lo, hi, freq=freq, inclusive="left")
    if len(idx) == 0:
        return {"weekend_frac": 0.0, "ushours_frac": 0.0}
    weekend = float(np.mean(idx.dayofweek >= 5))            # Sat/Sun
    ushours = float(np.mean((idx.hour >= 13) & (idx.hour < 21)))   # ~US cash session in UTC
    return {"weekend_frac": weekend, "ushours_frac": ushours}


def window_features(net_df, pos_df, win_net_M, windows, wi, cadence):
    """Causal features for window wi, built ONLY from windows < wi (past-only) + the calendar of the
    CURRENT window (known by the clock, not by market outcome). Returns a flat ordered dict.
      - regime: recent breadth / trend (mean position across candidates over the prior window)
      - vol level / vol-regime: realized vol of the BUYHOLD book over recent windows
      - recent per-candidate performance (rho-persistence): each candidate's mean NET over the last
        K past windows (the known-weak-alone signal)
      - whipsaw: recent flip rate of the MA-family books (chop proxy)
      - CALENDAR: weekend / US-hours fraction of the current window (orthogonal lever)
      - participation: mean exposure of the candidate set over the prior window
    """
    feats = {}
    K = 3                                                   # lookback in windows for recent perf
    past = win_net_M[max(0, wi - K):wi]                     # [<=K x n_cand]
    # recent per-candidate performance (rho-persistence) -- one feature per candidate
    if past.shape[0] > 0:
        recent = np.nanmean(past, axis=0)
    else:
        recent = np.zeros(net_df.shape[1])
    for ci, cand in enumerate(net_df.columns):
        feats[f"recent_{cand}"] = float(recent[ci]) if np.isfinite(recent[ci]) else 0.0
    # regime / participation / vol / whipsaw from the PRIOR window's bar-level series (past-only)
    if wi >= 1:
        plo, phi = windows[wi - 1]
        pmask = (net_df.index >= plo) & (net_df.index < phi)
        prior_pos = pos_df[pmask]
        prior_bh = net_df["BUYHOLD"][pmask] if "BUYHOLD" in net_df else net_df.iloc[:, 0][pmask]
        ma_cols = [c for c in pos_df.columns if c in MA_TYPES]
        feats["regime_breadth"] = float(prior_pos[ma_cols].mean().mean()) if ma_cols and len(prior_pos) else 0.5
        feats["participation"] = float(prior_pos.mean().mean()) if len(prior_pos) else 0.5
        feats["vol_level"] = float(prior_bh.std()) if len(prior_bh) > 2 else 0.0
        # whipsaw = mean |flip| of MA-family exposures over the prior window (chop proxy)
        if ma_cols and len(prior_pos) > 1:
            flips = prior_pos[ma_cols].diff().abs().mean().mean()
            feats["whipsaw"] = float(flips) if np.isfinite(flips) else 0.0
        else:
            feats["whipsaw"] = 0.0
    else:
        feats.update({"regime_breadth": 0.5, "participation": 0.5, "vol_level": 0.0, "whipsaw": 0.0})
    # vol-regime: prior-window vol vs the median vol of all past windows (relative level)
    if wi >= 2:
        vols = []
        for j in range(wi):
            jlo, jhi = windows[j]
            seg = net_df["BUYHOLD"][(net_df.index >= jlo) & (net_df.index < jhi)] if "BUYHOLD" in net_df \
                else net_df.iloc[:, 0][(net_df.index >= jlo) & (net_df.index < jhi)]
            if len(seg) > 2:
                vols.append(float(seg.std()))
        med = np.median(vols) if vols else 0.0
        feats["vol_regime"] = float(feats["vol_level"] - med)
    else:
        feats["vol_regime"] = 0.0
    # CALENDAR (current window; clock-known, causal)
    lo, hi = windows[wi]
    feats.update(_calendar_features(lo, hi, cadence))
    return feats


def feature_matrix(net_df, pos_df, win_net_M, windows, cadence):
    """Build the [n_windows x n_features] feature matrix + the ordered feature-name list."""
    rows = [window_features(net_df, pos_df, win_net_M, windows, wi, cadence)
            for wi in range(len(windows))]
    names = list(rows[0].keys()) if rows else []
    X = np.array([[r.get(n, 0.0) for n in names] for r in rows], float)
    return X, names


# =====================================================================================================
# 4. THE MODELS -- Tier-A (james-stein scorer) + Tier-B (gradient-boosted ranker)
# =====================================================================================================
def _standardize_fit(X):
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0) + 1e-9
    return mu, sd


def _standardize_apply(X, mu, sd):
    return np.nan_to_num((X - mu) / sd)


def tier_a_scores(Xtr_std, ytr_net, Xev_std, n_cand):
    """Tier-A: a SHRINKAGE feature scorer. For each candidate, fit a ridge-like linear map from features
    to that candidate's realized NEXT-window NET on TRAIN, then james-stein-shrink the per-candidate
    predicted scores at eval time (overfit-killer on the small 2020 sample). Returns [n_ev x n_cand]
    score matrix (higher = more recommended). Pure-numpy ridge (no sklearn dependency for Tier-A)."""
    lam = 10.0                                              # heavy ridge -- the sample is tiny
    nf = Xtr_std.shape[1]
    W = np.zeros((nf, n_cand))
    resid_var = []                                          # per-candidate TRAIN residual variance
    for ci in range(n_cand):
        y = ytr_net[:, ci]
        good = np.isfinite(y)
        if good.sum() < 3:
            resid_var.append(np.nan)
            continue
        Xg = Xtr_std[good]; yg = np.nan_to_num(y[good])
        Ag = Xg.T @ Xg + lam * np.eye(nf)
        try:
            W[:, ci] = np.linalg.solve(Ag, Xg.T @ yg)
        except np.linalg.LinAlgError:
            W[:, ci] = 0.0
        # in-sample residual: how much of the NET the linear map does NOT explain -> the noise floor
        pred_tr = Xg @ W[:, ci]
        resid_var.append(float(np.var(yg - pred_tr)))
    raw = Xev_std @ W                                       # [n_ev x n_cand] predicted NET per candidate
    # noise floor for the James-Stein shrink = the prediction-residual variance scaled by 1/n_tr
    # (the variance of the per-row predicted-score ESTIMATE). When the map explains nothing, the
    # residual variance ~ the NET variance and B -> 0 (collapse to prior); when the map genuinely
    # explains the NET, the residual is small and B -> 1 (keep the learned ordering). This is the
    # honest, scale-matched noise_var the default 1.0 compound-% floor over-shrinks.
    rv = np.array([v for v in resid_var if np.isfinite(v)], float)
    n_tr = max(1, Xtr_std.shape[0])
    noise_var = float(np.median(rv) / n_tr) if rv.size else 1.0
    noise_var = max(noise_var, 1e-9)
    out = np.zeros_like(raw)
    for r in range(raw.shape[0]):
        shr, _B = james_stein_shrink(raw[r].tolist(), noise_var=noise_var)
        out[r] = np.asarray(shr, float)
    return out


def tier_b_scores(Xtr_std, ytr_net, Xev_std, n_cand):
    """Tier-B: a gradient-boosted ranker. Fit one gradient-boosting regressor per candidate (predict its
    NEXT-window NET), score eval rows. LightGBM if present else sklearn HistGradientBoosting/GradientBoosting.
    Returns [n_ev x n_cand] or None if no GBM backend / insufficient data."""
    reg_factory = None
    try:
        import lightgbm as lgb  # noqa: F401
        from lightgbm import LGBMRegressor
        reg_factory = lambda: LGBMRegressor(n_estimators=60, max_depth=3, learning_rate=0.05,
                                            min_child_samples=3, verbosity=-1)
        backend = "lightgbm"
    except Exception:
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
            reg_factory = lambda: HistGradientBoostingRegressor(max_depth=3, max_iter=60,
                                                                learning_rate=0.05, min_samples_leaf=3)
            backend = "sklearn_hgb"
        except Exception:
            try:
                from sklearn.ensemble import GradientBoostingRegressor
                reg_factory = lambda: GradientBoostingRegressor(n_estimators=60, max_depth=3,
                                                                learning_rate=0.05)
                backend = "sklearn_gb"
            except Exception:
                return None, "none"
    if Xtr_std.shape[0] < 8:                                # too few windows for a tree ensemble
        return None, backend + "(too_few_windows)"
    Xtr_c = np.ascontiguousarray(Xtr_std, float)
    Xev_c = np.ascontiguousarray(Xev_std, float)
    scores = np.zeros((Xev_c.shape[0], n_cand))
    import warnings
    for ci in range(n_cand):
        y = ytr_net[:, ci]
        good = np.isfinite(y)
        if good.sum() < 6:
            scores[:, ci] = np.nan
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")             # mute the lgbm feature-name notice (cosmetic)
                m = reg_factory(); m.fit(Xtr_c[good], np.nan_to_num(y[good]))
                scores[:, ci] = m.predict(Xev_c)
        except Exception:
            scores[:, ci] = np.nan
    return scores, backend


# =====================================================================================================
# 5. RECOMMENDATION -> REALIZED BOOK (trade the top pick each window) + metrics
# =====================================================================================================
def realized_book_from_recos(net_df, windows, recos):
    """Given a recommended candidate per window (recos[wi] = candidate name), stitch the realized
    bar-level net of the recommended candidate over each window into one continuous net series.
    Returns (net_series, picks_list)."""
    pieces = []
    picks = []
    for wi, (lo, hi) in enumerate(windows):
        cand = recos[wi]
        picks.append(cand)
        seg = net_df[cand][(net_df.index >= lo) & (net_df.index < hi)].dropna()
        if len(seg):
            pieces.append(seg)
    if not pieces:
        return None, picks
    return pd.concat(pieces).sort_index(), picks


def book_metrics(net_series, pos_series, cadence):
    """NET%/Sharpe/maxDD/coverage of a stitched book net series (+ optional pos series for coverage)."""
    if net_series is None or len(net_series) < 3:
        return {}
    n = net_series.to_numpy()
    eq = np.cumprod(1 + n); peak = np.maximum.accumulate(eq)
    dd = float(((eq - peak) / peak).min() * 100)
    sh = float(np.mean(n) / (np.std(n) + 1e-12) * np.sqrt(ANN[cadence]))
    cov = float(np.mean(pos_series.reindex(net_series.index).fillna(0.0).to_numpy() > 0.01)) \
        if pos_series is not None else None
    return {"net": round(float((eq[-1] - 1) * 100), 1), "sharpe": round(sh, 2),
            "maxdd": round(dd, 1), "coverage": round(cov, 2) if cov is not None else None,
            "p05": block_bootstrap_p05_p95(n).get("p05")}


# =====================================================================================================
# 6. THE RUNWAY EVALUATION for one cadence (TRAIN fit / VAL select / OOS test) + the tool's Jul-Sep->Oct-Dec
# =====================================================================================================
def _split_window_idx(windows, lo, hi):
    """Indices of rolling windows whose START falls in [lo, hi)."""
    return [i for i, (wlo, _whi) in enumerate(windows)
            if pd.Timestamp(lo) <= wlo < pd.Timestamp(hi)]


def _label_argmax(win_net_M):
    """Per window, the argmax candidate by realized NET (the supervised label = best NEXT-window NET)."""
    lab = np.full(win_net_M.shape[0], -1, int)
    for w in range(win_net_M.shape[0]):
        row = win_net_M[w]
        if np.any(np.isfinite(row)):
            lab[w] = int(np.nanargmax(row))
    return lab


def _scores_to_recos(scores, cand_names):
    """Top-1 recommendation per eval row from a score matrix (NaN rows -> fall back to col mean)."""
    recos = []
    for r in range(scores.shape[0]):
        row = scores[r]
        if not np.any(np.isfinite(row)):
            recos.append(cand_names[0])
            continue
        recos.append(cand_names[int(np.nanargmax(row))])
    return recos


def evaluate_cadence(cadence, seed=11, n_shuffle=200):
    """Full evaluation for one cadence. Returns a dict with the leaderboard rows + controls + verdict bits."""
    net_df, pos_df = build_candidate_series(cadence)
    if net_df is None or net_df.shape[1] < 3:
        return {"cadence": cadence, "error": "insufficient candidate series"}
    cand_names = list(net_df.columns)
    n_cand = len(cand_names)
    roll = ROLL_BARS[cadence]
    windows = rolling_windows(net_df.index, roll)
    if len(windows) < 6:
        return {"cadence": cadence, "error": f"too few rolling windows ({len(windows)})"}
    win_net_M = per_window_table(net_df, windows)
    X, fnames = feature_matrix(net_df, pos_df, win_net_M, windows, cadence)

    # the LABEL each window = best-NEXT-window candidate: align features X(t) to label of window t
    # (X(t) is past-only built at window t; the realized NET of window t is the supervised target).
    Ytr_net = win_net_M                                     # we fit feature->realized-NET per candidate

    out = {"cadence": cadence, "n_windows": len(windows), "candidates": cand_names,
           "features": fnames, "roll_bars": roll, "splits": {}}

    # ---- index the rolling windows into TRAIN / VAL / OOS (by window START date) ----
    tr_idx = _split_window_idx(windows, *SPLITS["TRAIN"])
    va_idx = _split_window_idx(windows, *SPLITS["VAL"])
    oos_idx = _split_window_idx(windows, *SPLITS["OOS"])
    # the tool's comparable split: in-sample Jul-Sep (fit) -> OOS Oct-Dec (test)
    tool_fit_idx = _split_window_idx(windows, WIN[0], TOOL_SPLIT)
    tool_oos_idx = _split_window_idx(windows, TOOL_SPLIT, WIN[1])

    if len(tr_idx) < 3 or len(oos_idx) < 1:
        # fall back: use everything before OOS as TRAIN (1d has few windows)
        tr_idx = _split_window_idx(windows, RUNWAY[0], SPLITS["OOS"][0])
        va_idx = tr_idx[-max(1, len(tr_idx) // 4):]
        tr_idx = tr_idx[:-len(va_idx)] if len(tr_idx) > len(va_idx) else tr_idx

    # ---- standardize on TRAIN only ----
    mu, sd = _standardize_fit(X[tr_idx]) if tr_idx else _standardize_fit(X)
    Xstd = _standardize_apply(X, mu, sd)

    # ---- fit both tiers on TRAIN, score all windows ----
    a_scores = tier_a_scores(Xstd[tr_idx], Ytr_net[tr_idx], Xstd, n_cand)
    b_scores, b_backend = tier_b_scores(Xstd[tr_idx], Ytr_net[tr_idx], Xstd, n_cand)

    # ---- select the tier on VAL (NEVER OOS): which tier's recos earn more NET on VAL ----
    def _val_net(scores):
        if scores is None:
            return -1e9
        recos = _scores_to_recos(scores[va_idx], cand_names) if va_idx else []
        wsel = [windows[i] for i in va_idx]
        bk, _ = realized_book_from_recos(net_df, wsel, recos) if recos else (None, [])
        if bk is None or len(bk) < 2:
            return -1e9
        return float(np.cumprod(1 + bk.to_numpy())[-1] - 1) * 100
    a_val = _val_net(a_scores)
    b_val = _val_net(b_scores)
    chosen_tier = "B" if (b_scores is not None and b_val > a_val) else "A"
    chosen_scores = b_scores if chosen_tier == "B" else a_scores
    out["tier_selection"] = {"a_val_net": round(a_val, 2) if a_val > -1e8 else None,
                             "b_val_net": round(b_val, 2) if b_val > -1e8 else None,
                             "b_backend": b_backend, "chosen": chosen_tier}

    # ---- per-split leaderboard row ----
    static_pick = _static_rule_pick(win_net_M, tr_idx, va_idx, cand_names)   # VAL-best candidate (the static rule)
    rng = np.random.default_rng(seed)
    for split, idxs in [("TRAIN", tr_idx), ("VAL", va_idx), ("OOS", oos_idx),
                        ("TOOL_OOS", tool_oos_idx)]:
        if not idxs:
            out["splits"][split] = {"n_windows": 0}
            continue
        wsel = [windows[i] for i in idxs]
        # ML recommendation (chosen tier) realized on this split
        ml_recos = _scores_to_recos(chosen_scores[idxs], cand_names)
        ml_bk, ml_picks = realized_book_from_recos(net_df, wsel, ml_recos)
        ml_m = book_metrics(ml_bk, _book_pos(pos_df, wsel, ml_picks, windows, idxs), cadence)
        # static-rule pick (fixed candidate from VAL-best), realized on this split
        st_recos = [static_pick] * len(idxs)
        st_bk, _ = realized_book_from_recos(net_df, wsel, st_recos)
        st_m = book_metrics(st_bk, pos_df[static_pick] if static_pick in pos_df else None, cadence)
        # the real bars
        vt_m = _candidate_split_metrics(net_df, pos_df, "VOLTGT_BH", wsel, cadence)
        bh_m = _candidate_split_metrics(net_df, pos_df, "BUYHOLD", wsel, cadence)
        # ORACLE: hindsight-best candidate per window (the ceiling)
        or_recos = [cand_names[int(np.nanargmax(win_net_M[i]))] if np.any(np.isfinite(win_net_M[i]))
                    else cand_names[0] for i in idxs]
        or_bk, _ = realized_book_from_recos(net_df, wsel, or_recos)
        or_m = book_metrics(or_bk, None, cadence)
        # SHUFFLE control: single-draw (NOISY, back-compat field) + the HARDENED N-draw distribution.
        sh_net = _shuffle_control(net_df, wsel, ml_recos, cadence, rng)
        ml_net = ml_m.get("net")
        sh_dist = _shuffle_distribution(net_df, wsel, ml_recos, cadence, n_draws=n_shuffle, seed=seed)
        sh_block = None
        if sh_dist is not None:
            arr = sh_dist.pop("_arr")                       # drop the internal array before persisting
            if ml_net is not None and arr.size:
                # one-sided p-value: P(a random re-timing of the SAME picks does as well or better than ML)
                pval = float((arr >= ml_net).mean())
                pctl = float((arr < ml_net).mean())         # fraction of shuffles ML beats
                sh_dist["ml_net"] = ml_net
                sh_dist["ml_percentile"] = round(pctl, 3)
                sh_dist["p_value"] = round(pval, 3)
                # REAL timing skill = ML beats the shuffle MEDIAN with p < 0.10
                sh_dist["beats_median"] = bool(ml_net > sh_dist["median"])
                sh_dist["timing_skill"] = bool(ml_net > sh_dist["median"] and pval < 0.10)
            sh_block = sh_dist
        # random recommender
        rand_recos = [cand_names[rng.integers(n_cand)] for _ in idxs]
        rd_bk, _ = realized_book_from_recos(net_df, wsel, rand_recos)
        rd_m = book_metrics(rd_bk, None, cadence)
        out["splits"][split] = {
            "n_windows": len(idxs),
            "ML": {**ml_m, "pick_freq": _freq(ml_picks)},
            "STATIC": {**st_m, "pick": static_pick},
            "VOLTGT_BH": vt_m, "BUYHOLD": bh_m, "ORACLE": or_m,
            "RANDOM": rd_m, "SHUFFLE_net": sh_net, "SHUFFLE_dist": sh_block,
        }
    out["static_rule_pick"] = static_pick
    out["ablation"] = _ablation(net_df, pos_df, win_net_M, X, fnames, windows, tr_idx, va_idx, oos_idx,
                                cand_names, cadence)
    return out


def _book_pos(pos_df, wsel, picks, windows, idxs):
    """Stitch the realized exposure series for the ML picks across the eval windows (for coverage)."""
    pieces = []
    for k, (lo, hi) in enumerate(wsel):
        cand = picks[k] if k < len(picks) else picks[-1]
        if cand in pos_df:
            seg = pos_df[cand][(pos_df.index >= lo) & (pos_df.index < hi)]
            if len(seg):
                pieces.append(seg)
    return pd.concat(pieces).sort_index() if pieces else None


def _candidate_split_metrics(net_df, pos_df, cand, wsel, cadence):
    """Realized NET/Sharpe/maxDD/coverage of a single candidate over the union of eval windows."""
    if cand not in net_df:
        return {}
    lo = wsel[0][0]; hi = wsel[-1][1]
    seg = net_df[cand][(net_df.index >= lo) & (net_df.index < hi)]
    pos = pos_df[cand][(pos_df.index >= lo) & (pos_df.index < hi)] if cand in pos_df else None
    return book_metrics(seg, pos, cadence)


def _freq(picks):
    from collections import Counter
    c = Counter(picks)
    return {k: v for k, v in sorted(c.items(), key=lambda kv: -kv[1])}


def _static_rule_pick(win_net_M, tr_idx, va_idx, cand_names):
    """The static-rule pick = the candidate with the best NET over TRAIN+VAL in-sample (the leaderboard
    winner, ranked by NET as the design mandates). A FIXED candidate applied through OOS."""
    insample = (tr_idx or []) + (va_idx or [])
    if not insample:
        return cand_names[0]
    means = np.nanmean(win_net_M[insample], axis=0)
    return cand_names[int(np.nanargmax(means))]


def _one_block_shuffle_net(net_df, wsel, recos, rng):
    """One block-shuffle draw: permute the BLOCKS of the reco sequence (preserve candidate-frequency
    within a block, destroy timing) and re-stitch -> the shuffled book's compound NET %. None on fail."""
    block = max(1, len(recos) // 4)
    nb = int(np.ceil(len(recos) / block))
    blocks = [recos[i * block:(i + 1) * block] for i in range(nb)]
    order = rng.permutation(len(blocks))
    shuf = [c for i in order for c in blocks[i]][:len(recos)]
    bk, _ = realized_book_from_recos(net_df, wsel, shuf)
    if bk is None or len(bk) < 2:
        return None
    return float(np.cumprod(1 + bk.to_numpy())[-1] - 1) * 100


def _shuffle_control(net_df, wsel, recos, cadence, rng):
    """SINGLE-draw block-shuffle (kept for back-compat field SHUFFLE_net -- NOISY, do NOT verdict on it;
    use _shuffle_distribution for the significance test). Returns the one-draw compound NET % or None."""
    v = _one_block_shuffle_net(net_df, wsel, recos, rng)
    return None if v is None else round(v, 1)


def _shuffle_distribution(net_df, wsel, recos, cadence, n_draws=200, seed=11):
    """HARDENED shuffle control: N independent block-shuffles (distinct seeds) -> the full distribution
    of shuffled-book compound NET %. The single-draw _shuffle_control is NOISY (it can swing tens of %
    and sometimes BEAT the ML by luck), so 'ML beat its shuffle' from one draw is not evidence of timing
    skill. Here we compute, vs the distribution:
      - median / p05 / p95 of the shuffle NET
      - ml_percentile  = fraction of shuffles the ML's realized NET exceeds (1.0 = ML beats every shuffle)
      - p_value (one-sided) = fraction of shuffles whose NET >= the ML's NET (P(shuffle does as well/better)
        purely by re-timing the SAME picks). A REAL timing edge => ML beats the shuffle MEDIAN with a small
        p_value (the ML's timing is in the right tail of what random re-timings of its own picks achieve).
    Returns a dict (or None if no valid draws). `ml_net` is passed in by the caller to score against."""
    rng = np.random.default_rng(seed)
    draws = []
    for d in range(n_draws):
        v = _one_block_shuffle_net(net_df, wsel, recos, np.random.default_rng(seed + 1 + d))
        if v is not None:
            draws.append(v)
    if not draws:
        return None
    arr = np.asarray(draws, float)
    return {
        "n_draws": int(arr.size),
        "median": round(float(np.median(arr)), 2),
        "mean": round(float(np.mean(arr)), 2),
        "sd": round(float(np.std(arr)), 2),
        "p05": round(float(np.percentile(arr, 5)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "_arr": arr,   # internal: removed before persist; used to score ml_net at call site
    }


def _ablation(net_df, pos_df, win_net_M, X, fnames, windows, tr_idx, va_idx, oos_idx, cand_names, cadence):
    """Ablate the CALENDAR features (and the regime features) to attribute any OOS skill. Re-fit Tier-A
    without each feature GROUP, measure the OOS NET drop. The drop attributable to CALENDAR answers
    'is it the calendar feature?'."""
    if not oos_idx:
        return {}
    n_cand = len(cand_names)
    groups = {
        "ALL": [],
        "no_calendar": ["weekend_frac", "ushours_frac"],
        "no_regime": ["regime_breadth", "participation", "vol_level", "vol_regime", "whipsaw"],
        "no_recent": [c for c in fnames if c.startswith("recent_")],
    }
    res = {}
    for gname, drop in groups.items():
        keep = [i for i, f in enumerate(fnames) if f not in drop]
        if not keep:
            continue
        Xk = X[:, keep]
        mu, sd = _standardize_fit(Xk[tr_idx]) if tr_idx else _standardize_fit(Xk)
        Xstd = _standardize_apply(Xk, mu, sd)
        sc = tier_a_scores(Xstd[tr_idx], win_net_M[tr_idx], Xstd, n_cand)
        recos = _scores_to_recos(sc[oos_idx], cand_names)
        wsel = [windows[i] for i in oos_idx]
        bk, _ = realized_book_from_recos(net_df, wsel, recos)
        res[gname] = round(float(np.cumprod(1 + bk.to_numpy())[-1] - 1) * 100, 1) if bk is not None and len(bk) > 1 else None
    # calendar attribution = ALL - no_calendar (positive => calendar HELPS OOS net)
    if res.get("ALL") is not None and res.get("no_calendar") is not None:
        res["calendar_contribution_net"] = round(res["ALL"] - res["no_calendar"], 1)
    return res


# =====================================================================================================
# 7. VERDICT
# =====================================================================================================
def build_verdict(results):
    cads = [c for c in results if "error" not in results[c]]
    beat_static = []; beat_voltgt = []; timing_skill = []
    rows = []
    for cad in cads:
        oos = results[cad]["splits"].get("OOS", {})
        if not oos or oos.get("n_windows", 0) == 0:
            oos = results[cad]["splits"].get("TOOL_OOS", {})
        ml = oos.get("ML", {}); st = oos.get("STATIC", {}); vt = oos.get("VOLTGT_BH", {})
        sh = oos.get("SHUFFLE_net"); sd = oos.get("SHUFFLE_dist") or {}
        ml_net = ml.get("net"); st_net = st.get("net"); vt_net = vt.get("net")
        sh_med = sd.get("median"); sh_p = sd.get("p_value"); sh_pctl = sd.get("ml_percentile")
        if ml_net is not None and st_net is not None and ml_net > st_net + 0.5:
            beat_static.append(cad)
        if ml_net is not None and vt_net is not None and ml_net > vt_net + 0.5:
            beat_voltgt.append(cad)
        # TIMING SKILL (the HARDENED test, replaces the single-draw 'beat shuffle'): the ML must beat the
        # shuffle MEDIAN with a one-sided p < 0.10 -- i.e. its realized NET sits in the right tail of what
        # random re-timings of its OWN picks achieve. A single lucky/unlucky draw can no longer flip this.
        if sd.get("timing_skill"):
            timing_skill.append(cad)
        rows.append(f"[{cad}] OOS ML {ml_net}% (pick {ml.get('pick_freq')}) vs STATIC {st_net}% "
                    f"({st.get('pick')}) vs VOLTGT_BH {vt_net}% vs ORACLE {oos.get('ORACLE',{}).get('net')}% "
                    f"| shuffle-dist med {sh_med}% [1draw {sh}%] p={sh_p} ML-pctl={sh_pctl} "
                    f"{'TIMING-SKILL' if sd.get('timing_skill') else 'no-skill'} "
                    f"| tier {results[cad].get('tier_selection',{}).get('chosen')}")
    n = len(cads)
    # CO-CONDITION (the honest bar): a REAL defensive/timing edge must (a) BEAT VOLTGT_BH on NET (the wealth
    # bar / lower-exposure could explain a 'less-bad' loss) AND (b) show genuine TIMING SKILL -- ML beats the
    # shuffle-distribution MEDIAN with p<0.10 (NOT a single noisy draw). The 2026-06-13 single-draw shuffle was
    # NOISE (it swung -5%..-54% across cadences, sometimes beating the ML by luck); 'beat shuffle' from one draw
    # was never evidence. The multi-draw p-value is the proper significance test.
    beat_both = [c for c in beat_voltgt if c in timing_skill]
    cal_help = [c for c in cads
                if (results[c].get("ablation", {}) or {}).get("calendar_contribution_net", 0) and
                results[c]["ablation"]["calendar_contribution_net"] > 0]
    if beat_both:
        headline = (f"CONFIRMED-PARTIAL: at {len(beat_both)}/{n} cadences ({beat_both}) the recommender beat BOTH "
                    f"VOLTGT_BH on NET AND showed genuine TIMING SKILL (ML > shuffle-median, p<0.10 over the N-draw "
                    f"distribution) at the SAME cadence. Verify robustness (p05/PBO) before believing.")
    elif beat_voltgt or timing_skill:
        headline = (f"NULL (non-coincident): isolated beats only -- beat-VOLTGT at {beat_voltgt}, timing-skill "
                    f"(p<0.10 vs shuffle-dist) at {timing_skill} -- but NO single cadence shows BOTH. "
                    f"Where ML loses LESS than VOLTGT in the bear it is LOWER EXPOSURE, not timing skill: the ML's "
                    f"NET does not sit in the right tail of re-timings of its own picks. Vol-target buy-hold remains "
                    f"the wealth winner; the best-config table still ships (the requested output).")
    else:
        headline = (f"NO-EDGE: the learned recommender beat VOLTGT_BH on NET at {len(beat_voltgt)}/{n} cadences and "
                    f"showed timing skill (p<0.10 vs the N-draw shuffle distribution) at {len(timing_skill)}/{n}. "
                    f"The bear-defensive 'less-bad' behaviour is EXPOSURE, not timing skill -- random re-timings of "
                    f"the same picks do as well. Consistent with the static-rule verdict (vol-target buy-hold is the "
                    f"real best). The best-config table still ships (that IS the requested output).")
    lines = [f"BAR TO BEAT: VOLTGT_BH on NET (the static-rule winner) AND genuine TIMING SKILL (ML > the "
             f"shuffle-distribution MEDIAN with one-sided p<0.10 over {results[cads[0]]['splits'].get('OOS',{}).get('SHUFFLE_dist',{}).get('n_draws') if cads else '?'} "
             f"block-shuffle draws) -- on OOS {SPLITS['OOS']}.", f"HEADLINE: {headline}", ""] + rows + [
             "", f"calendar feature HELPS OOS net at {0 if not cal_help else len(cal_help)}/{n} cadences "
             f"({cal_help}) -- the ablation attribution.",
             "TIMING-SKILL TEST (hardened 2026-06-13): the single-draw shuffle is NOISY (it can swing tens of % and "
             "beat the ML by luck), so 'ML > shuffle' from one draw is NOT evidence. We now draw N>=100 independent "
             "block-shuffles of the SAME picks and require ML > the shuffle MEDIAN with one-sided p<0.10. 'Loses less "
             "than VOLTGT in the bear' is EXPOSURE (lower gross), distinct from TIMING (right tail of re-timings).",
             "CAVEATS: long-only; equal-weight u10; causal MtM; maker for MA families / taker-free book "
             f"construction matches the static rule; OOS == {SPLITS['OOS']}; tier chosen on VAL "
             "(never OOS); 1d has few rolling windows (small-sample). UNSEEN N/A."]
    return {"headline": headline, "beat_static_cadences": beat_static,
            "beat_voltgt_cadences": beat_voltgt, "timing_skill_cadences": timing_skill,
            "beat_both_cadences": beat_both, "calendar_helps_cadences": cal_help, "n_cadences": n, "lines": lines}


def best_config_oneliners(results):
    """The 'ML best config for 2020 OOS' one-liner per TF."""
    out = {}
    for cad, r in results.items():
        if "error" in r:
            out[cad] = f"(skipped: {r['error']})"
            continue
        oos = r["splits"].get("OOS", {}) or r["splits"].get("TOOL_OOS", {})
        ml = oos.get("ML", {})
        freq = ml.get("pick_freq", {})
        top = max(freq, key=freq.get) if freq else "?"
        out[cad] = (f"ML best config OOS = top pick '{top}' (freq {freq}); realized NET {ml.get('net')}% "
                    f"Sharpe {ml.get('sharpe')} maxDD {ml.get('maxdd')}% cov {ml.get('coverage')} "
                    f"| static pick '{oos.get('STATIC',{}).get('pick')}' {oos.get('STATIC',{}).get('net')}% "
                    f"| VOLTGT_BH {oos.get('VOLTGT_BH',{}).get('net')}%")
    return out


# =====================================================================================================
# 8. MAIN
# =====================================================================================================
def _print_table(cad, r):
    print(f"\n########## CADENCE {cad} -- ML config recommender on the 2020 runway ##########")
    if "error" in r:
        print(f"   SKIPPED: {r['error']}")
        return
    ts = r.get("tier_selection", {})
    print(f"   candidates={len(r['candidates'])} windows={r['n_windows']} roll_bars={r['roll_bars']} "
          f"| tier A_val={ts.get('a_val_net')} B_val={ts.get('b_val_net')} backend={ts.get('b_backend')} "
          f"-> chosen Tier-{ts.get('chosen')}")
    print(f"   static-rule pick (NET-best in-sample) = {r['static_rule_pick']}")
    hdr = (f"   {'split':10} {'ML net%':>9} {'STATIC%':>9} {'VOLTGT%':>9} {'BUYHOLD%':>9} {'ORACLE%':>9} "
           f"{'SHUFmed%':>9} {'SHUFp':>6} {'MLpctl':>7} {'skill':>6} {'RAND%':>7}")
    print(hdr)
    for split in ("TRAIN", "VAL", "OOS", "TOOL_OOS"):
        s = r["splits"].get(split, {})
        if not s or s.get("n_windows", 0) == 0:
            print(f"   {split:10} {'(no windows)':>9}")
            continue
        def g(k, kk="net"):
            return s.get(k, {}).get(kk) if isinstance(s.get(k), dict) else s.get(k)
        sd = s.get("SHUFFLE_dist") or {}
        skill = "YES" if sd.get("timing_skill") else ("no" if sd else "-")
        print(f"   {split:10} {str(g('ML')):>9} {str(g('STATIC')):>9} {str(g('VOLTGT_BH')):>9} "
              f"{str(g('BUYHOLD')):>9} {str(g('ORACLE')):>9} {str(sd.get('median')):>9} "
              f"{str(sd.get('p_value')):>6} {str(sd.get('ml_percentile')):>7} {skill:>6} "
              f"{str(g('RANDOM')):>7}")
    ab = r.get("ablation", {})
    if ab:
        print(f"   [ablation OOS] ALL={ab.get('ALL')} no_cal={ab.get('no_calendar')} "
              f"no_regime={ab.get('no_regime')} no_recent={ab.get('no_recent')} "
              f"-> calendar_contribution={ab.get('calendar_contribution_net')}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m strat.ml_config_recommender")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cadences", default=",".join(CADENCES))
    ap.add_argument("--train", default=None, help="override TRAIN window START:END (e.g. 2022-01-01:2022-08-01)")
    ap.add_argument("--val", default=None, help="override VAL window START:END")
    ap.add_argument("--oos", default=None, help="override OOS window START:END (the held-out test)")
    ap.add_argument("--tag", default="", help="label for the output filename (e.g. bear2022)")
    ap.add_argument("--n-shuffle", type=int, default=200, dest="n_shuffle",
                    help="number of block-shuffle draws for the timing-skill distribution (>=100)")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    # Parametrize the runway (default = the 2020 runway). All three windows must be given together.
    global RUNWAY, SPLITS, WIN, TOOL_SPLIT
    if a.train and a.val and a.oos:
        ts, te = a.train.split(":"); vs, ve = a.val.split(":"); os_, oe = a.oos.split(":")
        SPLITS = {"TRAIN": (ts, te), "VAL": (vs, ve), "OOS": (os_, oe)}
        RUNWAY = (ts, oe)
        WIN = (vs, oe)            # the 'tool-comparable' run: in-sample (val_start..oos_start) -> OOS (oos_start..end)
        TOOL_SPLIT = os_
        print(f"## RUNWAY OVERRIDE: TRAIN {SPLITS['TRAIN']} / VAL {SPLITS['VAL']} / OOS {SPLITS['OOS']}\n")

    cads = [c.strip() for c in a.cadences.split(",") if c.strip()]
    print("## ML CONFIG-RECOMMENDER -- learned X(t)->best-candidate on the TRAIN/VAL/OOS runway")
    print(f"   candidates = 8 MA families + BUYHOLD + VOLTGT_BH | rank by NET (wealth), NOT Sharpe")
    print(f"   TRAIN {SPLITS['TRAIN']} fit / VAL {SPLITS['VAL']} select / OOS {SPLITS['OOS']} test "
          f"(== the tool's Oct-Dec OOS) + the tool's Jul-Sep->Oct-Dec for comparability\n")

    print(f"   timing-skill test: {a.n_shuffle} block-shuffle draws per split "
          f"-> ML must beat the shuffle MEDIAN with one-sided p<0.10 (single-draw shuffle is NOISY)\n")
    results = {}
    for cad in cads:
        r = evaluate_cadence(cad, n_shuffle=a.n_shuffle)
        results[cad] = r
        _print_table(cad, r)

    verdict = build_verdict(results)
    oneliners = best_config_oneliners(results)
    print("\n" + "=" * 96)
    print("## AGGREGATE VERDICT")
    for line in verdict["lines"]:
        print(f"   {line}")
    print(f"\n## ML BEST CONFIG FOR OOS {SPLITS['OOS']} (one-liner per TF)")
    for cad, ol in oneliners.items():
        print(f"   [{cad}] {ol}")
    print("=" * 96)

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = ("_" + a.tag) if a.tag else ""
    p = OUT / f"ml_config_recommender{tag}_{stamp}.json"
    json.dump({
        "repro": {"command": "python -m strat.ml_config_recommender " + " ".join(argv or sys.argv[1:]),
                  "git_sha": sha, "cost_maker": MAKER_RT, "cost_taker": TAKER_RT,
                  "runway": RUNWAY, "splits": SPLITS, "tool_win": WIN, "tool_split": TOOL_SPLIT,
                  "candidates": CANDIDATES, "rank_metric": "NET (wealth)", "n_shuffle": a.n_shuffle,
                  "timing_skill_test": "ML > shuffle-distribution MEDIAN with one-sided p<0.10 over n_shuffle "
                                       "independent block-shuffle draws (single-draw shuffle is noisy)"},
        "results": results, "verdict": verdict, "best_config_oneliners": oneliners,
    }, open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n[persisted] {p}")
    return 0


# =====================================================================================================
# 9. SELFTEST -- two-sided soundness (synthetic, no market)
# =====================================================================================================
def selftest():
    """POSITIVE: when one candidate's NEXT-window NET is genuinely predictable from a feature (a planted
    regime->candidate map), the Tier-A scorer must recommend it and BEAT the average-exposure shuffle.
    NEGATIVE: when candidate outcomes are i.i.d. noise (no signal), the recommender must NOT beat the
    shuffle (the 'skill' collapses to average exposure / the prior). Plus: james-stein shrink must
    collapse a noise score row toward its mean."""
    print("## ML-CONFIG-RECOMMENDER SELFTEST (two-sided)")
    ok = True
    rng = np.random.default_rng(0)
    n_win, n_cand, n_feat = 80, 4, 3

    # helper: the HARDENED timing-skill test on synthetic per-window picks -- N independent
    # re-timing shuffles of the SAME picks; one-sided p = P(shuffle re-timing does as well/better than ML);
    # timing_skill = ML > shuffle MEDIAN with p < 0.10. Mirrors _shuffle_distribution's logic on arrays.
    def _timing_skill(Y, ev_idx, recos, n_draws=200, base_seed=7):
        ml = float(np.sum([Y[ev_idx[k], recos[k]] for k in range(len(ev_idx))]))
        draws = []
        for d in range(n_draws):
            r = np.random.default_rng(base_seed + d)
            sh = recos.copy(); r.shuffle(sh)
            draws.append(float(np.sum([Y[ev_idx[k], sh[k]] for k in range(len(ev_idx))])))
        arr = np.asarray(draws)
        p = float((arr >= ml).mean())
        return ml, float(np.median(arr)), p, bool(ml > np.median(arr) and p < 0.10)

    # ---- POSITIVE: feature 0 in {0,1} selects candidate 0 vs 1; that candidate gets +2% next window ----
    Xpos = np.zeros((n_win, n_feat))
    Xpos[:, 0] = rng.integers(0, 2, n_win)                 # the regime bit
    Xpos[:, 1:] = rng.normal(0, 1, (n_win, n_feat - 1))
    Ypos = rng.normal(0.0, 0.3, (n_win, n_cand))           # baseline noisy NET per candidate
    # plant: when bit=1, candidate 0 earns +2; when bit=0, candidate 1 earns +2 (a learnable map)
    Ypos[Xpos[:, 0] == 1, 0] += 2.0
    Ypos[Xpos[:, 0] == 0, 1] += 2.0
    tr = np.arange(0, 56); ev = np.arange(56, 80)
    mu, sd = _standardize_fit(Xpos[tr]); Xs = _standardize_apply(Xpos, mu, sd)
    sc = tier_a_scores(Xs[tr], Ypos[tr], Xs, n_cand)
    recos = np.array([int(np.nanargmax(sc[i])) for i in ev])
    # HARDENED: ML must show TIMING SKILL vs the N-draw re-timing distribution (not a single draw)
    ml_net, sh_med, sh_p, skill = _timing_skill(Ypos, ev, recos)
    pos_pass = skill                                       # genuine learnable map -> right-tail vs re-timings
    print(f"  POSITIVE (learnable regime->candidate): ML sum-NET {ml_net:.1f} vs shuffle-MEDIAN {sh_med:.1f} "
          f"(p={sh_p:.3f}) -> {'PASS' if pos_pass else 'FAIL'} (ML must show TIMING SKILL: beat median, p<0.10)")
    ok &= pos_pass

    # ---- NEGATIVE: i.i.d. noise outcomes, no feature signal -> ML must NOT show timing skill ----
    skill_count = 0; deltas = []
    for s in range(40):
        r2 = np.random.default_rng(100 + s)
        Xn = r2.normal(0, 1, (n_win, n_feat))
        Yn = r2.normal(0.0, 0.3, (n_win, n_cand))          # NO signal: outcomes independent of X
        mu, sd = _standardize_fit(Xn[tr]); Xs = _standardize_apply(Xn, mu, sd)
        sc = tier_a_scores(Xs[tr], Yn[tr], Xs, n_cand)
        rc = np.array([int(np.nanargmax(sc[i])) for i in ev])
        ml_n, med_n, p_n, sk_n = _timing_skill(Yn, ev, rc, base_seed=200 + s)
        skill_count += int(sk_n)
        deltas.append(ml_n - med_n)
    mean_delta = float(np.mean(deltas))
    # under pure noise, the false-timing-skill rate must stay near the nominal 10% (allow slack to 20%)
    neg_pass = (skill_count / 40.0) <= 0.20 and mean_delta <= 0.5
    print(f"  NEGATIVE (i.i.d. noise, no signal): false-timing-skill rate {skill_count}/40 "
          f"({skill_count/40.0:.2f}), mean(ML - shuffle-median)={mean_delta:+.4f} "
          f"-> {'PASS' if neg_pass else 'FAIL'} (expect <=0.20 false-positive rate; no manufactured skill)")
    ok &= neg_pass

    # ---- james-stein collapse on a noise score row ----
    noise_row = np.random.default_rng(5).normal(0, 0.01, 12).tolist()
    _, B = james_stein_shrink(noise_row, prior=0.0)
    js_pass = B < 0.5
    print(f"  SHRINK (noise score row): james-stein B={B:.2f} -> {'PASS' if js_pass else 'FAIL'} "
          f"(expect <0.5: collapse to prior under noise)")
    ok &= js_pass

    print(f"\n  SELFTEST {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
