"""MAGNITUDE-SELECTION -- mover problem #1 (SELECTION) reframed through the MAGNITUDE lens.

QUESTION (the deployable spec): can we predict EX-ANTE (using ONLY prior-day state,
observable at/before today's UTC close) WHICH assets will be tomorrow's BIG-MAGNITUDE
movers (|daily return| cross-sectional rank, direction-AGNOSTIC) -- well enough to
tilt a vol-target / straddle basket toward the names that will actually move most?

WHY THIS, NOT DIRECTIONAL SELECTION (which is dead):
  D17 cross-sectional directional IC ~ 0; D67/D72 directional continuation null.
  But MAGNITUDE / vol is robustly persistent and a parallel intraday lane just got
  OOS AUC 0.731 for MAGNITUDE-continuation (and 0.664 mechanism-only, beyond vol).
  A daily-mover characterization (daily_movers_profile) found:
    (a) movers ~80% idiosyncratic;
    (b) prior-day PRECURSOR of a big-magnitude day is WEAK but real -- liquidations
        (effect-d ~0.1), funding-abs (~0.1), everything else negligible;
    (c) magnitude persistence: P(mover tomorrow | mover today) = 38% vs 26% base = 1.45x.
  A weak per-NAME edge (d~0.1) can't PICK the one mover. But the FRESH ANGLE is a
  CROSS-SECTIONAL magnitude-RANK predictor: a weak per-name edge over MANY names +
  vol persistence may give a tradeable magnitude SELECTION even though it cannot pick
  direction. Monetizing it likely needs options (straddles) -- OUT OF SCOPE here; we
  only establish whether the SELECTION SIGNAL EXISTS, held-out.

THE TAUTOLOGY GUARD (the make-or-break check, mirrors the continuation lane):
  Magnitude/vol is persistent, so "yesterday's vol predicts today's vol" is a near-
  tautology that is NOT new information. We therefore run THREE models:
    VOL-ONLY  : prior-day realized-vol / magnitude features only (the tautology floor).
    MECH-ONLY : everything EXCEPT vol/persistence (liq, funding, OI, whale, volume,
                days-since-listed, jumps) -- the genuine extra information.
    FULL      : vol + mechanism.
  The decision number is MECH-ONLY-minus-VOL-ONLY (does positioning add over vol?) and
  FULL-minus-VOL-ONLY (does the combined model beat the tautology?). If FULL ~= VOL-ONLY
  the "signal" is just vol-clustering -- still mildly tradeable, but it's not the
  precursor story; we report it as such honestly.

HONEST HELD-OUT (mandatory, two-sided):
  TRAIN  date <  2024-01-01  : fit models, z-stats, top-decile thresholds, ALL params.
  OOS    2024-01-01..2025-07-01 : scored ONCE.
  UNSEEN date >= 2025-07-01  : touched EXACTLY ONCE at the very end (final confirm only).
  Every feature uses data <= today's close; the LABEL is tomorrow's |return|. No row's
  own-day or future data enters its features. Cross-sectional z / ranking is within-day,
  which uses only that day's contemporaneous (prior-day) features -- causal.
  Shuffled-label null per metric is the false-positive floor; a positive that does not
  beat its shuffled null, or is TRAIN-only, is a NULL and is reported as such.

METRICS:
  magnitude rank-IC : Spearman( predicted score , tomorrow's |return| ) per day, averaged.
  precision@top-k   : among the top-k predicted-magnitude names each day, what fraction
                      are genuine top-quartile |movers| tomorrow? vs base rate + random.
  top-vs-bottom spread : mean tomorrow |return| of top-k predicted minus bottom-k
                         predicted -- the economically meaningful straddle-basket spread.

Run:
  python -m mining.magnitude_selection --universe u50
No emoji (cp1252). Does NOT git commit.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
DAILY_DIR = ROOT / "data" / "processed" / "chimera" / "1d"
OUT = ROOT / "runs" / "mining"
OUT.mkdir(parents=True, exist_ok=True)

__contract__ = {
    "kind": "research",
    "inputs": {"chimera_1d": "data/processed/chimera/1d/*.parquet (daily panel)"},
    "outputs": {"study_json": "runs/mining/magnitude_selection_<tag>_<stamp>.json"},
    "invariants": {
        "train_only_fit": "models, z-stats, thresholds ALL from TRAIN only",
        "causal_features": "every feature uses data <= today's close; label = tomorrow's |return|",
        "unseen_once": "UNSEEN >= 2025-07-01 touched exactly once at the end",
        "shuffled_null": "each metric reports a shuffled-label null as the false-positive floor",
        "tautology_guard": "VOL-ONLY / MECH-ONLY / FULL feature sets isolate vol-clustering from positioning",
    },
}

TRAIN_END = "2024-01-01"
OOS_END = "2025-07-01"
DECILE_TOPK = 0.25      # top quartile = "big mover" (matches daily_movers_profile defn)
MIN_XSEC = 8            # need >= 8 names to rank a cross-section

# ---------------------------------------------------------------------------
# Feature groups. ALL are prior-day state (computed from data <= today's close).
# The LABEL is tomorrow's |return|. None of these reference today's realized return
# beyond what is already in chimera columns dated `today` (which are causal by
# pipeline construction -- they describe the bar that closed today).
# ---------------------------------------------------------------------------
# VOL / MAGNITUDE-PERSISTENCE features (the tautology floor -- "vol predicts vol").
VOL_FEATS = [
    "norm_yz_volatility",     # Yang-Zhang realized vol (per-asset z, trailing/causal)
    "norm_vol_cluster",       # vol-clustering state (causal)
    "norm_vol_ratio",         # short/long vol ratio (causal)
    "rv_rv_5m",               # realized variance (5m, today's bar)
    "rv_bpv_5m",              # bipower variation (today's bar)
    # NOTE: target_vol_20 is EXCLUDED -- it is rolling_std(20).shift(-20) = FORWARD vol
    # (a LABEL, not a feature) and overlaps tomorrow's |return| => look-ahead leak.
    "dv_dvol_close",          # DVOL implied vol (Deribit) -- observed AT today's close (VIX-like, causal)
    "abs_ret_today",          # |today's realized return| -- the raw magnitude-persistence signal
]
# MECHANISM features (the genuine extra information -- positioning / flow / structure).
MECH_FEATS = [
    # liquidations (effect-d ~0.1 precursor)
    "liq_total_usd_log", "liq_long_z30", "liq_short_z30", "liq_capitulation",
    # funding-abs (effect-d ~0.1 precursor) + extremity
    "fund_rate_abs_mean", "fund_rate_z30", "premium_vol30",
    # OI build / divergence
    "norm_oi_change", "norm_oi_price_divergence", "s3_oi_usd_log",
    # whale flow
    "norm_whale", "wh_whale_net_usd_abs_log", "wh_whale_trade_count",
    # volume
    "norm_log_volume",
    # jumps (a jumpy yesterday -> jumpy tomorrow, structurally distinct from smooth vol)
    "rv_jump_frac", "rv_jump_count",
    # days-since-listed (young coins move more)
    "days_since_listed",
    # microstructure / efficiency
    "norm_kyle_lambda", "norm_perm_entropy", "norm_vpin",
]
FULL_FEATS = VOL_FEATS + MECH_FEATS


def _sym_of(path: str) -> str:
    return os.path.basename(path).split("_")[0].upper()


def split_of(date_str: str) -> str:
    if date_str < TRAIN_END:
        return "TRAIN"
    if date_str < OOS_END:
        return "OOS"
    return "UNSEEN"


def _safe_log_abs(x: np.ndarray) -> np.ndarray:
    """log(1+|x|) * sign-agnostic magnitude transform for heavy-tailed USD features."""
    return np.log1p(np.abs(np.asarray(x, dtype=float)))


def load_panel(universe: str) -> pl.DataFrame:
    """Long panel: one row per (date, sym) with the raw feature columns + tomorrow's
    |return| label. ret = close[t]/close[t-1]-1 realized ON `date` (prior-day-relative).
    The LABEL fwd_absret is |close[t+1]/close[t]-1| (tomorrow's magnitude)."""
    flag = {"u10": "is_u10", "u50": "is_u50", "u100": "is_u100"}[universe]
    raw_cols = [
        "date", "close", flag,
        # vol
        "norm_yz_volatility", "norm_vol_cluster", "norm_vol_ratio",
        "rv_rv_5m", "rv_bpv_5m", "dv_dvol_close",
        # liq
        "liq_total_usd", "liq_long_z30", "liq_short_z30", "liq_capitulation",
        # funding
        "fund_rate_abs_mean", "fund_rate_z30", "premium_vol30",
        # oi
        "norm_oi_change", "norm_oi_price_divergence", "s3_oi_usd",
        # whale
        "norm_whale", "wh_whale_net_usd", "wh_whale_trade_count",
        # volume
        "norm_log_volume", "volume_usd",
        # jumps
        "rv_jump_frac", "rv_jump_count",
        # listing age
        "mv_days_since_listed_binance",
        # microstructure
        "norm_kyle_lambda", "norm_perm_entropy", "norm_vpin",
    ]
    frames = []
    for f in sorted(glob.glob(str(DAILY_DIR / "*.parquet"))):
        try:
            head = pl.read_parquet(f, columns=[flag])
        except Exception:
            continue
        if len(head) == 0 or not bool(head[flag][0]):
            continue
        avail = pl.read_parquet_schema(f)
        cols = [c for c in raw_cols if c in avail]
        df = pl.read_parquet(f, columns=cols).sort("date")
        sym = _sym_of(f)
        c = df["close"].to_numpy().astype(float)
        n = len(c)
        ret = np.full(n, np.nan)
        ret[1:] = c[1:] / c[:-1] - 1.0           # today's realized return
        fwd_ret = np.full(n, np.nan)
        fwd_ret[:-1] = c[1:] / c[:-1] - 1.0       # tomorrow's return (shift forward)
        out = {
            "date": df["date"].cast(pl.Utf8).to_numpy(),
            "sym": np.array([sym] * n),
            "ret": ret,
            "fwd_absret": np.abs(fwd_ret),         # the LABEL: tomorrow's magnitude
        }
        # raw feature passthrough (fill missing cols with NaN)
        def col(name):
            return df[name].to_numpy().astype(float) if name in df.columns else np.full(n, np.nan)
        out["norm_yz_volatility"] = col("norm_yz_volatility")
        out["norm_vol_cluster"] = col("norm_vol_cluster")
        out["norm_vol_ratio"] = col("norm_vol_ratio")
        out["rv_rv_5m"] = col("rv_rv_5m")
        out["rv_bpv_5m"] = col("rv_bpv_5m")
        out["target_vol_20"] = col("target_vol_20")
        out["dv_dvol_close"] = col("dv_dvol_close")
        out["abs_ret_today"] = np.abs(ret)         # raw magnitude-persistence signal
        out["liq_total_usd_log"] = _safe_log_abs(col("liq_total_usd"))
        out["liq_long_z30"] = col("liq_long_z30")
        out["liq_short_z30"] = col("liq_short_z30")
        out["liq_capitulation"] = col("liq_capitulation")
        out["fund_rate_abs_mean"] = col("fund_rate_abs_mean")
        out["fund_rate_z30"] = col("fund_rate_z30")
        out["premium_vol30"] = col("premium_vol30")
        out["norm_oi_change"] = col("norm_oi_change")
        out["norm_oi_price_divergence"] = col("norm_oi_price_divergence")
        out["s3_oi_usd_log"] = _safe_log_abs(col("s3_oi_usd"))
        out["norm_whale"] = col("norm_whale")
        out["wh_whale_net_usd_abs_log"] = _safe_log_abs(col("wh_whale_net_usd"))
        out["wh_whale_trade_count"] = col("wh_whale_trade_count")
        out["norm_log_volume"] = col("norm_log_volume")
        out["rv_jump_frac"] = col("rv_jump_frac")
        out["rv_jump_count"] = col("rv_jump_count")
        out["days_since_listed"] = col("mv_days_since_listed_binance")
        out["norm_kyle_lambda"] = col("norm_kyle_lambda")
        out["norm_perm_entropy"] = col("norm_perm_entropy")
        out["norm_vpin"] = col("norm_vpin")
        frames.append(pl.DataFrame(out))
    panel = pl.concat(frames, how="vertical_relaxed")
    # keep rows with a valid label and at least the today-return defined
    panel = panel.filter(pl.col("fwd_absret").is_finite() & pl.col("ret").is_finite())
    # cross-section size per day; drop thin days
    panel = panel.with_columns(pl.len().over("date").alias("xsec_n"))
    panel = panel.filter(pl.col("xsec_n") >= MIN_XSEC)
    # label: tomorrow's cross-sectional |return| rank (pct in (0,1]) + top-quartile flag
    panel = panel.with_columns(
        (pl.col("fwd_absret").rank("ordinal").over("date") / pl.col("xsec_n")).alias("fwd_absret_rank")
    )
    panel = panel.with_columns(
        (pl.col("fwd_absret_rank") > (1.0 - DECILE_TOPK)).alias("is_big_mover_tmrw")
    )
    panel = panel.with_columns(pl.col("date").map_elements(split_of, return_dtype=pl.Utf8).alias("split"))
    return panel


# ---------------------------------------------------------------------------
# Feature matrix builders -- per-asset z-score from TRAIN stats only (no leakage),
# then per-day cross-sectional demean so the model learns RELATIVE magnitude.
# ---------------------------------------------------------------------------
def _zscore_per_asset(panel: pl.DataFrame, feats: list[str]) -> dict:
    """Per-(asset,feature) z using TRAIN-only mean/std; applied to all splits.
    Returns dict feat -> np.array aligned to panel row order."""
    sym = panel["sym"].to_numpy()
    split = panel["split"].to_numpy()
    z = {}
    for f in feats:
        v = panel[f].to_numpy().astype(float)
        zz = np.full(len(v), np.nan)
        for s in np.unique(sym):
            idx = np.where(sym == s)[0]
            tr = idx[split[idx] == "TRAIN"]
            x_tr = v[tr]
            ok = np.isfinite(x_tr)
            if ok.sum() < 30:
                # too few TRAIN obs for this asset -> use that asset's all-time on TRAIN window only
                continue
            mu, sd = np.nanmean(x_tr[ok]), np.nanstd(x_tr[ok])
            if sd > 0:
                zz[idx] = (v[idx] - mu) / sd
        z[f] = zz
    return z


def _build_X(zdict: dict, feats: list[str], rows: np.ndarray) -> np.ndarray:
    return np.column_stack([zdict[f][rows] for f in feats])


def _impute_train_median(Xtr: np.ndarray, *Xs) -> tuple:
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    def fill(X):
        Xf = X.copy()
        bad = ~np.isfinite(Xf)
        Xf[bad] = np.take(med, np.where(bad)[1])
        return Xf
    return (fill(Xtr),) + tuple(fill(x) for x in Xs)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4:
        return np.nan
    from scipy.stats import rankdata
    ra = rankdata(a[ok]); rb = rankdata(b[ok])
    if np.std(ra) == 0 or np.std(rb) == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def daily_rank_ic(scores: np.ndarray, labels_absret: np.ndarray, dates: np.ndarray) -> dict:
    """Per-day Spearman(score, tomorrow's |return|), averaged. This is magnitude-IC."""
    ics = []
    for d in np.unique(dates):
        m = dates == d
        if m.sum() < 4:
            continue
        ic = _spearman(scores[m], labels_absret[m])
        if np.isfinite(ic):
            ics.append(ic)
    ics = np.array(ics)
    if len(ics) == 0:
        return {"mean_ic": np.nan, "n_days": 0, "ic_t": np.nan, "frac_pos": np.nan}
    t = float(np.mean(ics) / (np.std(ics) / np.sqrt(len(ics)))) if np.std(ics) > 0 else np.nan
    return {"mean_ic": float(np.mean(ics)), "median_ic": float(np.median(ics)),
            "n_days": int(len(ics)), "ic_t": t, "frac_pos": float((ics > 0).mean())}


def topk_metrics(scores: np.ndarray, is_big: np.ndarray, absret: np.ndarray,
                 dates: np.ndarray, k_frac: float = DECILE_TOPK, rng=None) -> dict:
    """Per-day: pick top-k by score; report precision (fraction that are true big movers),
    base rate, random-selection precision, and the top-k vs bottom-k mean |move| spread."""
    if rng is None:
        rng = np.random.default_rng(0)
    precs, base_rates, rand_precs = [], [], []
    topk_moves, botk_moves, all_moves = [], [], []
    for d in np.unique(dates):
        m = np.where(dates == d)[0]
        if len(m) < 8:
            continue
        sc = scores[m]; big = is_big[m].astype(bool); ar = absret[m]
        k = max(1, int(round(k_frac * len(m))))
        order = np.argsort(-sc)
        top = m[order[:k]]; bot = m[order[-k:]]
        precs.append(float(is_big[top].mean()))
        base_rates.append(float(big.mean()))
        # random selection of k names that day
        ridx = rng.permutation(len(m))[:k]
        rand_precs.append(float(big[ridx].mean()))
        topk_moves.append(float(np.mean(absret[top])))
        botk_moves.append(float(np.mean(absret[bot])))
        all_moves.append(float(np.mean(ar)))
    precs = np.array(precs); base_rates = np.array(base_rates); rand_precs = np.array(rand_precs)
    topk_moves = np.array(topk_moves); botk_moves = np.array(botk_moves); all_moves = np.array(all_moves)
    if len(precs) == 0:
        return {"degenerate": True}
    spread = topk_moves - botk_moves
    t_spread = float(np.mean(spread) / (np.std(spread) / np.sqrt(len(spread)))) if np.std(spread) > 0 else np.nan
    return {
        "n_days": int(len(precs)),
        "precision_topk": float(np.mean(precs)),
        "base_rate": float(np.mean(base_rates)),
        "random_precision": float(np.mean(rand_precs)),
        "lift_vs_base": float(np.mean(precs) / np.mean(base_rates)) if np.mean(base_rates) > 0 else None,
        "lift_vs_random": float(np.mean(precs) / np.mean(rand_precs)) if np.mean(rand_precs) > 0 else None,
        "mean_absmove_topk": float(np.mean(topk_moves)),
        "mean_absmove_botk": float(np.mean(botk_moves)),
        "mean_absmove_all": float(np.mean(all_moves)),
        "topk_minus_botk_spread": float(np.mean(spread)),
        "spread_t": t_spread,
        "spread_ratio": float(np.mean(topk_moves) / np.mean(botk_moves)) if np.mean(botk_moves) > 0 else None,
    }


# ---------------------------------------------------------------------------
# STATIC vs DYNAMIC decomposition -- THE make-or-break check for tradeability.
#   A cross-sectional |move| ranking can be (a) STATIC: small/volatile coins simply
#   move more than big coins, a constant ranking you write down once (NOT a timing
#   edge, no straddle-rotation alpha), or (b) DYNAMIC: predicting WHICH day a given
#   asset moves more than its OWN average (the genuinely tradeable timing signal).
#   We measure both: a STATIC baseline (rank by each asset's TRAIN-mean |move|, zero
#   daily features) and a WITHIN-ASSET (asset-demeaned) IC that strips the static tier.
# ---------------------------------------------------------------------------
def static_baseline_ic(panel: pl.DataFrame, split_name: str) -> dict:
    """Rank assets each day by their TRAIN-mean |fwd move| (a CONSTANT per-asset score,
    no daily information). If this alone matches the model's IC, the 'signal' is a
    static size/vol-tier ranking, not a precursor edge."""
    split = panel["split"].to_numpy()
    sym = panel["sym"].to_numpy()
    dates = panel["date"].to_numpy()
    absret = panel["fwd_absret"].to_numpy().astype(float)
    tr = split == "TRAIN"
    static = np.full(len(absret), np.nan)
    for s in np.unique(sym):
        idx = sym == s
        static[idx] = np.nanmean(absret[idx & tr])
    mask = split == split_name
    ic = daily_rank_ic(static[mask], absret[mask], dates[mask])
    tk = topk_metrics(static[mask], panel["is_big_mover_tmrw"].to_numpy().astype(int)[mask],
                      absret[mask], dates[mask], rng=np.random.default_rng(11))
    return {"rank_ic": ic, "topk": tk}


def within_asset_ic(scores_sub: np.ndarray, panel: pl.DataFrame, rows: np.ndarray,
                    tr_mask: np.ndarray, seed: int = 0) -> dict:
    """Asset-demean BOTH the score and the label by each asset's SAME-WINDOW (OOS/UNSEEN)
    mean, then per-day rank-IC of the residuals = the CLEAN dynamic timing edge with the
    static size/vol tier fully removed. THIS is the number that must be positive for a
    rotation edge beyond a constant ranking.

    Why in-window (not TRAIN) demeaning: a TRAIN-mean offset does NOT remove per-asset
    LEVEL DRIFT between TRAIN and the eval window (a coin that calmed down, a newly-listed
    coin's age trend). The model's score encodes the static tier and then correlates with
    that leftover per-asset offset -> the within-asset IC is INFLATED by drift, not timing
    (empirically TRAIN-demean gave 0.265 vs in-window 0.056 on u50). This is a DIAGNOSTIC
    decomposition of where existing OOS predictions' IC comes from, not a tradeable signal
    needing the OOS mean ex-ante -- so in-window centering is the correct attribution and
    introduces no look-ahead into any deployable score.

    Returns the clean within-asset IC plus a within-DAY shuffled-label null (permute the
    label deviations inside each day) as its false-positive floor."""
    sym = panel["sym"].to_numpy()
    dates = panel["date"].to_numpy()
    absret = panel["fwd_absret"].to_numpy().astype(float)
    sym_sub = sym[rows]
    sc = scores_sub.astype(float).copy()
    lab = absret[rows].copy()
    dts = dates[rows]
    for s in np.unique(sym_sub):
        m = sym_sub == s
        if m.sum() >= 4:
            sc[m] = sc[m] - np.nanmean(scores_sub[m])
            lab[m] = lab[m] - np.nanmean(absret[rows][m])
    ic = daily_rank_ic(sc, lab, dts)
    # within-day shuffled null: permute label deviations inside each day
    rng = np.random.default_rng(seed + 31)
    nulls = []
    uds = np.unique(dts)
    for _ in range(20):
        labsh = lab.copy()
        for d in uds:
            idx = np.where(dts == d)[0]
            if len(idx) >= 2:
                labsh[idx] = lab[idx][rng.permutation(len(idx))]
        v = daily_rank_ic(sc, labsh, dts)["mean_ic"]
        if np.isfinite(v):
            nulls.append(v)
    ic["shuffled_within_day_null_mean"] = float(np.mean(nulls)) if nulls else None
    ic["shuffled_within_day_null_max"] = float(np.max(nulls)) if nulls else None
    return ic


# ---------------------------------------------------------------------------
# Model fit + scoring for one feature set
# ---------------------------------------------------------------------------
def fit_and_score(panel: pl.DataFrame, feats: list[str], seed: int,
                  score_unseen: bool = False) -> dict:
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor

    split = panel["split"].to_numpy()
    dates = panel["date"].to_numpy()
    y_rank = panel["fwd_absret_rank"].to_numpy().astype(float)   # regress on the rank target
    is_big = panel["is_big_mover_tmrw"].to_numpy().astype(int)
    absret = panel["fwd_absret"].to_numpy().astype(float)

    zdict = _zscore_per_asset(panel, feats)
    rows_all = np.arange(len(panel))
    tr = rows_all[split == "TRAIN"]
    oo = rows_all[split == "OOS"]
    un = rows_all[split == "UNSEEN"]

    Xtr = _build_X(zdict, feats, tr)
    Xoo = _build_X(zdict, feats, oo)
    Xun = _build_X(zdict, feats, un)
    Xtr_i, Xoo_i, Xun_i = _impute_train_median(Xtr, Xoo, Xun)
    ytr = y_rank[tr]

    # Ridge (linear cross-sectional)
    ridge = Ridge(alpha=10.0)
    ridge.fit(Xtr_i, ytr)
    s_oo_ridge = ridge.predict(Xoo_i)

    # GBM
    gbm = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.2, n_iter_no_change=20, random_state=seed)
    gbm.fit(Xtr_i, ytr)
    s_oo_gbm = gbm.predict(Xoo_i)

    def metrics_for(scores, rows):
        ic = daily_rank_ic(scores, absret[rows], dates[rows])
        tk = topk_metrics(scores, is_big[rows], absret[rows], dates[rows],
                          rng=np.random.default_rng(seed + 1))
        return {"rank_ic": ic, "topk": tk}

    tr_bool = split == "TRAIN"
    out = {
        "n_feats": len(feats),
        "n_train": int(len(tr)), "n_oos": int(len(oo)), "n_unseen": int(len(un)),
        "ridge_oos": metrics_for(s_oo_ridge, oo),
        "gbm_oos": metrics_for(s_oo_gbm, oo),
        # DYNAMIC (asset-demeaned, in-window) IC: the static size/vol tier removed. THIS
        # is the number that must be positive (and beat its within-day null) for a
        # tradeable rotation edge beyond a constant size-tier ranking.
        "gbm_oos_within_asset": within_asset_ic(s_oo_gbm, panel, oo, tr_bool, seed),
    }
    out["gbm_oos_within_asset_ic"] = out["gbm_oos_within_asset"]["mean_ic"]

    # shuffled-label null (permute TRAIN rank target; refit GBM; score OOS vs TRUE label)
    rng = np.random.default_rng(seed + 99)
    null_ics, null_precs = [], []
    for k in range(5):
        ysh = ytr.copy(); rng.shuffle(ysh)
        g = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.2, n_iter_no_change=20, random_state=seed + k)
        g.fit(Xtr_i, ysh)
        s = g.predict(Xoo_i)
        null_ics.append(daily_rank_ic(s, absret[oo], dates[oo])["mean_ic"])
        null_precs.append(topk_metrics(s, is_big[oo], absret[oo], dates[oo],
                                       rng=np.random.default_rng(seed + 200 + k))["precision_topk"])
    null_ics = [x for x in null_ics if np.isfinite(x)]
    null_precs = [x for x in null_precs if np.isfinite(x)]
    out["shuffled_null"] = {
        "ic_mean": float(np.mean(null_ics)) if null_ics else None,
        "ic_max": float(np.max(null_ics)) if null_ics else None,
        "precision_mean": float(np.mean(null_precs)) if null_precs else None,
        "precision_max": float(np.max(null_precs)) if null_precs else None,
    }

    # permutation importance on OOS (GBM, against rank-IC)
    base_ic = out["gbm_oos"]["rank_ic"]["mean_ic"]
    imp = {}
    rng2 = np.random.default_rng(seed + 7)
    for j, fn in enumerate(feats):
        Xp = Xoo_i.copy()
        col = Xp[:, j].copy(); rng2.shuffle(col); Xp[:, j] = col
        ic_p = daily_rank_ic(gbm.predict(Xp), absret[oo], dates[oo])["mean_ic"]
        imp[fn] = float((base_ic or 0) - (ic_p or 0))
    out["perm_importance_oos"] = dict(sorted(imp.items(), key=lambda kv: -kv[1]))

    # univariate OOS rank-IC of each raw feature (decomposition: which feature carries it,
    # and is the carrier a STATIC size proxy (oi/volume) or a dynamic precursor?)
    uni = {}
    for f in feats:
        uni[f] = daily_rank_ic(zdict[f][oo], absret[oo], dates[oo])["mean_ic"]
    out["univariate_oos_ic"] = dict(sorted(uni.items(),
                                           key=lambda kv: -(abs(kv[1]) if kv[1] is not None and np.isfinite(kv[1]) else 0)))

    # UNSEEN scored ONCE (final confirm only) -- GBM only, the chosen model arm
    if score_unseen and len(un) > 0:
        s_un_gbm = gbm.predict(Xun_i)
        out["gbm_unseen"] = metrics_for(s_un_gbm, un)
        out["gbm_unseen_within_asset"] = within_asset_ic(s_un_gbm, panel, un, tr_bool, seed)
        out["gbm_unseen_within_asset_ic"] = out["gbm_unseen_within_asset"]["mean_ic"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Cross-sectional magnitude-selection (held-out)")
    ap.add_argument("--universe", default="u50", choices=["u10", "u50", "u100"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or args.universe

    t0 = time.time()
    print(f"loading daily panel for {args.universe} ...")
    panel = load_panel(args.universe)
    n_days = panel.select(pl.col("date").n_unique()).item()
    n_syms = panel.select(pl.col("sym").n_unique()).item()
    tr_n = int((panel["split"].to_numpy() == "TRAIN").sum())
    oo_n = int((panel["split"].to_numpy() == "OOS").sum())
    un_n = int((panel["split"].to_numpy() == "UNSEEN").sum())
    print(f"panel: {len(panel)} asset-days, {n_syms} assets, {n_days} days "
          f"[{panel['date'].min()} -> {panel['date'].max()}]")
    print(f"split: TRAIN {tr_n} / OOS {oo_n} / UNSEEN {un_n}")

    # descriptive: magnitude persistence baseline (P(big mover tomorrow | big mover today))
    pj = panel.sort(["sym", "date"])
    sym = pj["sym"].to_numpy(); big = pj["is_big_mover_tmrw"].to_numpy().astype(bool)
    # today big = the row whose fwd label one step back; reconstruct "big today"
    absret_today = pj["ret"].to_numpy()
    # rank today's |ret| within day to get "big mover today"
    pj2 = pj.with_columns((pl.col("ret").abs().rank("ordinal").over("date") / pl.col("xsec_n")).alias("absret_today_rank"))
    big_today = (pj2["absret_today_rank"].to_numpy() > (1 - DECILE_TOPK))
    base_big = float(big.mean())
    # P(big tomorrow | big today): big_today aligns to same row's fwd label
    cond = float(big[big_today].mean()) if big_today.sum() > 0 else np.nan
    persistence = {"base_rate_big": base_big, "p_big_tmrw_given_big_today": cond,
                   "persistence_lift": float(cond / base_big) if base_big > 0 else None}
    print(f"magnitude persistence: P(big tmrw|big today)={cond*100:.1f}% vs base {base_big*100:.1f}% "
          f"(lift {persistence['persistence_lift']:.2f}x)")

    print("\nfitting VOL-ONLY (tautology floor) ...")
    res_vol = fit_and_score(panel, VOL_FEATS, args.seed)
    print("fitting MECH-ONLY (genuine extra info) ...")
    res_mech = fit_and_score(panel, MECH_FEATS, args.seed)
    print("fitting FULL (vol + mechanism) -- scoring UNSEEN once ...")
    res_full = fit_and_score(panel, FULL_FEATS, args.seed, score_unseen=True)

    # STATIC baseline (rank by asset's TRAIN-mean |move|, ZERO daily info) -- the
    # make-or-break contrast: if the model IC ~= static IC, the 'signal' is a constant
    # size/vol-tier ranking, not a precursor / timing edge.
    static_oos = static_baseline_ic(panel, "OOS")
    static_unseen = static_baseline_ic(panel, "UNSEEN")
    static_ic = static_oos["rank_ic"]["mean_ic"]
    full_within = res_full.get("gbm_oos_within_asset_ic")
    vol_within = res_vol.get("gbm_oos_within_asset_ic")
    print(f"STATIC baseline (no daily info): OOS rankIC={static_ic:+.4f}  |  "
          f"FULL within-asset (dynamic) IC={full_within:+.4f}")

    def ic_of(r, arm="gbm_oos"):
        return r[arm]["rank_ic"]["mean_ic"]
    def prec_of(r, arm="gbm_oos"):
        return r[arm]["topk"].get("precision_topk")
    def spread_of(r, arm="gbm_oos"):
        return r[arm]["topk"].get("topk_minus_botk_spread")

    full_ic = ic_of(res_full); vol_ic = ic_of(res_vol); mech_ic = ic_of(res_mech)
    shuf_ic_max = res_full["shuffled_null"]["ic_max"]
    shuf_prec_max = res_full["shuffled_null"]["precision_max"]
    full_prec = prec_of(res_full); full_spread = spread_of(res_full)
    full_spread_t = res_full["gbm_oos"]["topk"].get("spread_t")

    beats_null_ic = (full_ic is not None and shuf_ic_max is not None and full_ic > shuf_ic_max + 0.005)
    beats_null_prec = (full_prec is not None and shuf_prec_max is not None and full_prec > shuf_prec_max + 0.01)
    mech_adds = (mech_ic is not None and vol_ic is not None and mech_ic > 0)
    full_beats_vol = (full_ic is not None and vol_ic is not None and (full_ic - vol_ic) > 0.005)
    spread_meaningful = (full_spread is not None and full_spread > 0 and
                         np.isfinite(full_spread_t) and full_spread_t > 2.0)

    # --- the make-or-break static-vs-dynamic logic ---
    # (1) The TOTAL (cross-asset) IC must beat random (shuffled null) -- the easy bar.
    # (2) But for a TRADEABLE rotation edge it must beat the STATIC asset-tier baseline,
    #     because a constant 'small coins move more' ranking is not an exploitable spread
    #     over a vol-target that already weights by each asset's own vol level.
    # (3) The DYNAMIC (within-asset) IC must be positive -- predicting WHICH day an asset
    #     moves more than its OWN average. If full_within ~ vol_within, the dynamic edge is
    #     pure vol-persistence (the tautology), with no precursor add.
    full_beats_static = (full_ic is not None and static_ic is not None and (full_ic - static_ic) > 0.01)
    # the dynamic edge must beat its OWN within-day shuffled null (not the cross-asset one)
    dyn_null_max = res_full.get("gbm_oos_within_asset", {}).get("shuffled_within_day_null_max")
    dynamic_beats_null = (full_within is not None and dyn_null_max is not None and
                          np.isfinite(full_within) and full_within > dyn_null_max + 0.005)
    dynamic_positive = (full_within is not None and np.isfinite(full_within) and full_within > 0.01
                        and dynamic_beats_null)
    mech_adds_dynamic = (full_within is not None and vol_within is not None and
                         np.isfinite(full_within) and np.isfinite(vol_within) and
                         (full_within - vol_within) > 0.005)

    verdict = {
        # headline (cross-asset) -- almost certainly positive but mostly STATIC
        "full_oos_rank_ic": full_ic,
        "static_baseline_oos_ic": static_ic,
        "full_minus_static_ic": (full_ic - static_ic) if (full_ic is not None and static_ic is not None) else None,
        "model_beats_static_tier": bool(full_beats_static),
        "headline_is_mostly_static_size_tier": bool(not full_beats_static),
        # DYNAMIC (within-asset, asset-demeaned) -- the actually tradeable timing edge
        "full_within_asset_ic": full_within,
        "vol_within_asset_ic": vol_within,
        "dynamic_within_day_null_max": dyn_null_max,
        "dynamic_within_minus_vol": (full_within - vol_within) if (full_within is not None and vol_within is not None) else None,
        "dynamic_beats_within_day_null": bool(dynamic_beats_null),
        "dynamic_timing_edge_positive": bool(dynamic_positive),
        "dynamic_edge_more_than_vol_persistence": bool(mech_adds_dynamic),
        # nulls / tautology
        "beats_shuffled_null_ic": bool(beats_null_ic),
        "beats_shuffled_null_precision": bool(beats_null_prec),
        "vol_only_oos_rank_ic": vol_ic,
        "mech_only_oos_rank_ic": mech_ic,
        "full_minus_vol_ic": (full_ic - vol_ic) if (full_ic is not None and vol_ic is not None) else None,
        # economic spread (the straddle-basket read)
        "topk_precision_oos": full_prec,
        "topk_lift_vs_random": res_full["gbm_oos"]["topk"].get("lift_vs_random"),
        "topk_minus_botk_spread_oos": full_spread,
        "spread_ratio_oos": res_full["gbm_oos"]["topk"].get("spread_ratio"),
        "spread_t_oos": full_spread_t,
        "spread_economically_meaningful_raw": bool(spread_meaningful),
        # UNSEEN (scored once)
        "unseen_full_rank_ic": ic_of(res_full, "gbm_unseen") if "gbm_unseen" in res_full else None,
        "unseen_static_ic": static_unseen["rank_ic"]["mean_ic"],
        "unseen_within_asset_ic": res_full.get("gbm_unseen_within_asset_ic"),
        "unseen_topk_spread": res_full.get("gbm_unseen", {}).get("topk", {}).get("topk_minus_botk_spread"),
        "unseen_spread_t": res_full.get("gbm_unseen", {}).get("topk", {}).get("spread_t"),
        # THE BOTTOM LINE
        "magnitude_selection_signal_exists": bool(beats_null_ic and full_ic > 0),
        "tradeable_rotation_edge_beyond_static_tier": bool(full_beats_static and dynamic_positive),
    }

    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=str(ROOT)).stdout.strip()
    except Exception:
        sha = "unknown"
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "tool": "magnitude_selection", "git_sha": sha, "seed": args.seed,
        "universe": args.universe,
        "params": {"train_end": TRAIN_END, "oos_end": OOS_END, "topk_frac": DECILE_TOPK,
                   "min_xsec": MIN_XSEC, "vol_feats": VOL_FEATS, "mech_feats": MECH_FEATS},
        "n_assets": n_syms, "n_days": n_days,
        "n_train": tr_n, "n_oos": oo_n, "n_unseen": un_n,
        "magnitude_persistence": persistence,
        "static_baseline": {"oos": static_oos, "unseen": static_unseen},
        "results": {"vol_only": res_vol, "mech_only": res_mech, "full": res_full},
        "verdict": verdict,
        "caveats": [
            "u current membership (survivorship on absolute liq/oi/whale levels)",
            "dv_dvol_close (DVOL implied vol) is BTC/ETH-only -- NaN-imputed for alts (still prior-day where present)",
            "monetizing magnitude-selection needs OPTIONS (straddle/vol-target) -- OUT OF SCOPE; this only "
            "establishes whether the SELECTION SIGNAL exists, not a net-of-cost edge",
            "shuffled-null is the false-positive floor; full IC must beat its max by >0.005",
            "STATIC-vs-DYNAMIC is the load-bearing contrast: the headline cross-asset IC is mostly a "
            "constant size/vol-tier ranking (small coins move more); only the WITHIN-ASSET (asset-demeaned) "
            "IC over the static baseline is a tradeable rotation edge",
            "a vol-target already weights by each asset's own vol level, so it ALREADY captures the static "
            "tier -- the model only adds value to the extent of full_minus_static + the within-asset edge",
        ],
    }
    out_path = OUT / f"magnitude_selection_{tag}_{stamp}.json"
    out_path.write_text(json.dumps(payload, indent=1, default=str))

    # ------------------------------------------------------------------ STORY
    print("\n" + "=" * 84)
    print("MAGNITUDE-SELECTION (cross-sectional ex-ante |return|-rank predictor) -- STORY")
    print("=" * 84)
    print(f"universe {args.universe}: {n_syms} assets, {n_days} days, "
          f"TRAIN {tr_n} / OOS {oo_n} / UNSEEN {un_n} asset-days")
    print(f"magnitude persistence: P(big tmrw|big today)={cond*100:.1f}% vs base {base_big*100:.1f}% "
          f"(lift {persistence['persistence_lift']:.2f}x)")
    print("-" * 84)
    print(f"{'FEATURE SET':<14}{'rankIC GBM':>11}{'rankIC Ridge':>13}{'prec@top-k':>12}"
          f"{'lift vs rnd':>12}{'spread bp':>11}")
    for nm, r in [("VOL-ONLY", res_vol), ("MECH-ONLY", res_mech), ("FULL", res_full)]:
        g = r["gbm_oos"]; rd = r["ridge_oos"]
        print(f"{nm:<14}{g['rank_ic']['mean_ic']:>11.4f}{rd['rank_ic']['mean_ic']:>13.4f}"
              f"{g['topk']['precision_topk']:>12.3f}"
              f"{(g['topk'].get('lift_vs_random') or 0):>12.2f}"
              f"{g['topk']['topk_minus_botk_spread']*1e4:>11.0f}")
    print("-" * 84)
    sn = res_full["shuffled_null"]
    print(f"SHUFFLED NULL (FULL): rankIC mean {sn['ic_mean']:+.4f} max {sn['ic_max']:+.4f}  "
          f"precision mean {sn['precision_mean']:.3f} max {sn['precision_max']:.3f}")
    print("-" * 84)
    print("STATIC vs DYNAMIC (the make-or-break decomposition):")
    print(f"  STATIC baseline (rank by asset TRAIN-mean |move|, ZERO daily info): OOS rankIC = {static_ic:+.4f}")
    print(f"  FULL model cross-asset OOS rankIC                                 : {full_ic:+.4f}")
    print(f"  FULL - STATIC = {verdict['full_minus_static_ic']:+.4f}  "
          f"=> {'model BEATS the static size-tier' if verdict['model_beats_static_tier'] else 'headline IC is ~JUST the static size-tier (small coins move more)'}")
    dnull = verdict.get('dynamic_within_day_null_max')
    print(f"  DYNAMIC (within-asset, IN-WINDOW demeaned, CLEAN) IC: FULL={full_within:+.4f}  VOL-ONLY={vol_within:+.4f}  "
          f"(within-day null max {dnull:+.4f})")
    print(f"     => dynamic edge {'REAL (beats its null)' if verdict['dynamic_beats_within_day_null'] else 'NOT distinguishable from null'}; "
          f"{'mechanism adds over vol-persistence' if verdict['dynamic_edge_more_than_vol_persistence'] else 'edge is ~pure vol-persistence'}")
    print("-" * 84)
    print("TAUTOLOGY GUARD (cross-asset):")
    print(f"  VOL-ONLY rankIC = {vol_ic:+.4f}   MECH-ONLY = {mech_ic:+.4f}   FULL = {full_ic:+.4f}   "
          f"(FULL-VOL = {verdict['full_minus_vol_ic']:+.4f})")
    print("-" * 84)
    g = res_full["gbm_oos"]["topk"]
    print(f"ECONOMIC (FULL, OOS): top-k |move| {g['mean_absmove_topk']*100:.2f}%  "
          f"bot-k |move| {g['mean_absmove_botk']*100:.2f}%  all {g['mean_absmove_all']*100:.2f}%  "
          f"spread {g['topk_minus_botk_spread']*100:.2f}% (t={g['spread_t']:.1f}, ratio {g['spread_ratio']:.2f}x)")
    print("TOP UNIVARIATE OOS rank-IC (which feature carries it; is the carrier a STATIC size proxy?):")
    for fn, iv in list(res_full["univariate_oos_ic"].items())[:8]:
        if iv is None or not np.isfinite(iv):
            continue
        tagm = " [MECH]" if fn in MECH_FEATS else " [vol]"
        size = " <- size/level proxy" if fn in ("s3_oi_usd_log", "norm_log_volume") else ""
        print(f"   {fn:<26}{iv:+.4f}{tagm}{size}")
    print("-" * 84)
    if "gbm_unseen" in res_full:
        u = res_full["gbm_unseen"]
        print(f"UNSEEN (scored once): cross-asset rankIC {u['rank_ic']['mean_ic']:+.4f}  "
              f"static {static_unseen['rank_ic']['mean_ic']:+.4f}  "
              f"within-asset {res_full.get('gbm_unseen_within_asset_ic'):+.4f}  "
              f"spread {u['topk']['topk_minus_botk_spread']*100:.2f}% (t={u['topk']['spread_t']:.1f})")
    print("-" * 84)
    print("VERDICT:")
    print(f"  [1] magnitude-selection SIGNAL exists (beats random/shuffled)? "
          f"{verdict['magnitude_selection_signal_exists']}  (full IC {full_ic:+.4f} vs shuf-max {shuf_ic_max:+.4f})")
    print(f"  [2] model beats the STATIC size-tier (adds over a constant ranking)? "
          f"{verdict['model_beats_static_tier']}  (FULL-STATIC = {verdict['full_minus_static_ic']:+.4f})")
    print(f"  [3] DYNAMIC within-asset timing edge positive? "
          f"{verdict['dynamic_timing_edge_positive']}  (within-asset IC = {full_within:+.4f})")
    print(f"  [4] dynamic edge MORE than vol-persistence tautology? "
          f"{verdict['dynamic_edge_more_than_vol_persistence']}  "
          f"(within FULL-VOL = {verdict['dynamic_within_minus_vol']:+.4f})")
    print(f"  ==> TRADEABLE ROTATION EDGE beyond the static vol-tier? "
          f"{verdict['tradeable_rotation_edge_beyond_static_tier']}")
    print(f"({time.time()-t0:.0f}s)  JSON -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
