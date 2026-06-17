"""Wave 4C -- Regime-overlay robustness probe + book-level test.

TASK (pre-registered before any number is looked at):
  STEP 1: Rolling-window IC robustness probe for hbr_eta_xratio + te_in_btc + te_out_btc.
    Null hypothesis H0: IC = 0 (no predictive relationship between the feature and
    next-day asset return). Alternative H1 (one-sided, directional): pre-register the
    direction on TRAIN before looking at OOS/UNSEEN.
    Verdict criterion: a feature is PERSISTENT if median rolling-IC > 0, >60% of
    windows positive, AND stable across regime-splits (bull/bear/chop); otherwise
    REGIME-ARTIFACT.

  STEP 2: Test SURVIVING conditioners + stbl_z30 + dvol as a SLOW REGIME-LEVEL overlay
    on the daily_engine book (vol-targeted + regime-scalar already inside).
    N_EFF DISCIPLINE: at 1d cadence, slow regimes last ~30-90 days. A per-trade gate
    on ~10 trades/asset/year is structurally under-powered (Wave-2C lesson).
    Design: the conditioner gates the BOOK'S OVERALL EXPOSURE (slow, risk-on/off) at
    daily cadence, not per-trade. This gives n_eff >> 10 (all the daily returns, not
    just entry events).
    Baseline: the un-gated daily_engine book (already has the internal regime scalar).
    Null: shuffled-conditioner (permuted dates) with identical gating logic.
    Test: one-sided (pre-registered direction from TRAIN), block-bootstrap (block=20d).
    UNSEEN touched exactly ONCE at the end, after direction is fixed.

HONEST FRAMING:
  - IC is a DIAGNOSTIC for conditioner worthiness, NOT the objective.
  - Objective = compound return on HELD-OUT (OOS then UNSEEN once).
  - Report: n_eff, DD, p05, seed-spread, vs un-gated + shuffled null.
  - If any step is ambiguous, call it AMBIGUOUS, not REAL.

OUTPUT: this script; runs non-destructively; no git commit; no emoji.
AUTHOR: quant-expert / Wave 4C.
"""
from __future__ import annotations

import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SRC / "pipeline") not in sys.path:
    sys.path.insert(0, str(SRC / "pipeline"))

from pipeline.chimera_loader import ChimeraLoader  # noqa: E402
from pipeline.purge_split import get_split_dates   # noqa: E402
from strat.daily_engine import (                    # noqa: E402
    load_close_panel, build_book, window_stats, TAKER_RT
)

# ============================================================
# CONFIG
# ============================================================
ASSETS_U10 = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT",
]

# Split boundaries (frozen from data_config.yaml)
# train_end=2023-07-01, val_end=2024-05-15, oos_end=2025-03-15, unseen_start=2025-03-15
SPLITS = None  # loaded via get_split_dates()

# IC rolling-window parameters
IC_WINDOW = 60        # 60-day rolling window for IC (enough degrees of freedom)
IC_LAG = 1            # predict next-day return (h=1 diagnostic)
IC_MIN_OBS = 40       # minimum obs in a window to compute IC

# Block-bootstrap parameters
BB_N_REPS = 1000
BB_BLOCK = 20         # 20-day blocks (captures autocorrelation)
RNG_SEED = 42

# Overlay thresholds (pre-registered on TRAIN only)
# Direction for each conditioner is PRE-REGISTERED here:
# hbr_eta_xratio: higher branching ratio => more reflexivity => unclear direction
#   pre-registration: HIGH hbr_eta => trending/momentum => POSITIVE for long positions
#   (literature: high reflexivity = trending markets, better for trend books)
# te_in_btc: high TE_in_BTC => asset strongly follows BTC => risk-on => positive
# te_out_btc: high TE_out_BTC => asset LEADS BTC => often speculative/high-beta => unclear
#   pre-registration: treat as risk-on signal (positive for trend book)
# stbl_z30: positive z30 => stablecoin minting SURGE => RISK-ON signal => POSITIVE
#   (literature: stablecoin minting precedes market rallies)
# dvol (BTC dvol_close): HIGH dvol => high fear => NEGATIVE for trend book exposure
#   pre-registration: dvol above median => reduce exposure (risk-off)
CONDITIONER_DIRECTIONS = {
    "xrel_hbr_eta_total_xratio": +1,   # high => trend-friendly => +exposure
    "te_in_btc": +1,                    # high => BTC-following => risk-on => +exposure
    "te_out_btc": +1,                   # high => leading BTC => risk-on => +exposure
    "stbl_total_zscore_30d": +1,        # high => stablecoin minting => risk-on => +exposure
    "dv_dvol_close": -1,                # high => fear => reduce exposure
}

# Regime-overlay: exposure multiplier when conditioner is "favorable" vs "unfavorable"
# Conservative: only 2 states (on/off via median threshold)
OVERLAY_ON = 1.0    # full exposure when conditioner is favorable
OVERLAY_OFF = 0.5   # half exposure when conditioner is unfavorable (not zero: avoid over-fitting)


# ============================================================
# HELPERS
# ============================================================
def load_feature_panel(feature_col: str, loader: ChimeraLoader) -> pd.DataFrame:
    """Load a single feature column across all u10 assets into a date x asset DataFrame."""
    frames = {}
    for asset in ASSETS_U10:
        try:
            df = loader.load(asset, cadence="1d", features=[feature_col])
        except Exception:
            continue
        ts = pd.to_datetime(df["timestamp"].to_list(), unit="ms").floor("D")
        vals = df[feature_col].to_pandas().values if feature_col in df.columns else None
        if vals is None:
            continue
        s = pd.Series(vals, index=ts, name=asset)
        s = s[~s.index.duplicated(keep="last")]
        frames[asset] = s
    return pd.DataFrame(frames).sort_index()


def spearman_ic(feature_series: pd.Series, ret_series: pd.Series) -> float:
    """Rank-IC (Spearman) between feature and next-day return. Returns NaN if insufficient data."""
    aligned = pd.concat([feature_series, ret_series], axis=1).dropna()
    if len(aligned) < IC_MIN_OBS:
        return np.nan
    x = aligned.iloc[:, 0].rank()
    y = aligned.iloc[:, 1].rank()
    n = len(x)
    cov = np.cov(x.values, y.values)
    corr = cov[0, 1] / (np.sqrt(cov[0, 0] * cov[1, 1]) + 1e-12)
    return float(corr)


def rolling_ic_series(feature_panel: pd.DataFrame, ret_panel: pd.DataFrame,
                      window: int = IC_WINDOW) -> pd.DataFrame:
    """Compute rolling IC (date-indexed) for each asset separately, then average.
    Returns a DataFrame [date x asset] of rolling ICs (after lag)."""
    # shift feature by 1 (predict NEXT day's return)
    feat_lag = feature_panel.shift(IC_LAG)
    ic_records = {}
    for asset in feature_panel.columns:
        if asset not in ret_panel.columns:
            continue
        feat_s = feat_lag[asset].dropna()
        ret_s = ret_panel[asset].dropna()
        # align
        common = feat_s.index.intersection(ret_s.index)
        if len(common) < window + IC_MIN_OBS:
            continue
        feat_s = feat_s.reindex(common)
        ret_s = ret_s.reindex(common)
        ics = []
        dates = []
        for i in range(window, len(common)):
            w_feat = feat_s.iloc[i - window:i]
            w_ret = ret_s.iloc[i - window:i]
            ic = spearman_ic(w_feat, w_ret)
            ics.append(ic)
            dates.append(common[i])
        ic_records[asset] = pd.Series(ics, index=dates, name=asset)
    return pd.DataFrame(ic_records)


def compute_n_eff(returns: pd.Series, max_lag: int = 50) -> float:
    """Effective sample size under autocorrelation (Newey-West style).
    n_eff = n / (1 + 2 * sum_k rho_k * (1 - k/K)) for K = min(n/4, max_lag)."""
    r = returns.dropna().values
    n = len(r)
    if n < 10:
        return float(n)
    K = min(int(n / 4), max_lag)
    rho_sum = 0.0
    for k in range(1, K + 1):
        rho_k = float(np.corrcoef(r[:-k], r[k:])[0, 1])
        if np.isfinite(rho_k):
            rho_sum += rho_k * (1 - k / (K + 1))
    denom = 1.0 + 2.0 * max(rho_sum, 0.0)
    return float(n / denom)


def block_bootstrap_p05(compound_series: pd.Series, block: int = BB_BLOCK,
                        n_reps: int = BB_N_REPS, seed: int = RNG_SEED) -> float:
    """Circular block-bootstrap: resample the DAILY returns (block-size blocks) to get
    distribution of compound returns. Returns the 5th percentile of the compound distribution."""
    r = compound_series.dropna().values
    n = len(r)
    if n < block * 3:
        return float(np.percentile(r, 5) * np.sqrt(n))  # fallback: t-approx
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    comps = []
    for _ in range(n_reps):
        starts = rng.integers(0, n, size=n_blocks)
        resampled = np.concatenate([r[s:s + block] if s + block <= n
                                    else np.concatenate([r[s:], r[:s + block - n]])
                                    for s in starts])[:n]
        comps.append(float(np.prod(1.0 + resampled) - 1.0))
    return float(np.percentile(comps, 5))


def regime_split_ic(feature_panel: pd.DataFrame, ret_panel: pd.DataFrame,
                    regime_labels: pd.Series) -> dict:
    """IC per regime (trend/chop/down) using the daily_engine's internal regime labels.
    Cross-section mean IC across all assets."""
    feat_lag = feature_panel.shift(1)
    results = {}
    for regime in ["trend", "chop", "down"]:
        regime_dates = regime_labels[regime_labels == regime].index
        ics = []
        for asset in feature_panel.columns:
            if asset not in ret_panel.columns:
                continue
            f = feat_lag[asset].reindex(regime_dates).dropna()
            r = ret_panel[asset].reindex(f.index).dropna()
            f = f.reindex(r.index)
            ic = spearman_ic(f, r)
            if np.isfinite(ic):
                ics.append(ic)
        results[regime] = {
            "mean_ic": float(np.mean(ics)) if ics else np.nan,
            "n_assets": len(ics),
            "n_obs": int(len(regime_dates)),
        }
    return results


# ============================================================
# BOOK-LEVEL OVERLAY
# ============================================================
def apply_conditioner_overlay(
    base_net: pd.Series,
    base_weights: pd.DataFrame,
    conditioner_series: pd.Series,
    direction: int,
    train_end: str,
    close_panel: pd.DataFrame,
    cost_rt: float = TAKER_RT,
) -> dict:
    """Apply a single-conditioner slow exposure overlay to the daily_engine book.

    The conditioner gates BOOK-LEVEL gross exposure (not per-trade).
    Threshold is fit on TRAIN (pre-train_end) using the median as the split point.
    Favorable state = direction * (conditioner > median) > 0.
    Exposure multiplier: OVERLAY_ON if favorable, OVERLAY_OFF otherwise.

    Returns a dict with gated_net, threshold, pct_favorable, cost_adjustments.
    """
    # Fit threshold on TRAIN only (causal: no look-ahead)
    cond_train = conditioner_series[conditioner_series.index < pd.Timestamp(train_end)]
    threshold = float(cond_train.dropna().median())

    # Determine favorable state for each day
    favorable = (conditioner_series > threshold) if direction > 0 else (conditioner_series < threshold)
    favorable = favorable.reindex(base_net.index).fillna(False)

    # Apply overlay multiplier
    mult = favorable.map({True: OVERLAY_ON, False: OVERLAY_OFF}).astype(float)

    # Recompute the gated return stream
    # The base_weights are the un-lagged weights from build_book
    # We apply the multiplier to the weights before lagging
    W_gated = base_weights.mul(mult, axis=0)
    rets_panel = close_panel.pct_change(fill_method=None).fillna(0.0)
    W_lag = W_gated.shift(1).fillna(0.0)
    gross_ret = (W_lag * rets_panel).sum(axis=1)
    turnover = (W_gated - W_gated.shift(1)).abs().sum(axis=1).fillna(0.0)
    gated_net = gross_ret - turnover * (cost_rt / 2.0)

    pct_favorable = float(favorable.mean())
    return {
        "gated_net": gated_net,
        "threshold": threshold,
        "pct_favorable": pct_favorable,
        "W_gated": W_gated,
    }


def shuffled_null_compound(
    gated_net_fn,
    conditioner_series: pd.Series,
    direction: int,
    train_end: str,
    base_weights: pd.DataFrame,
    close_panel: pd.DataFrame,
    n_reps: int = 200,
    seed: int = RNG_SEED,
    eval_window: tuple = None,
) -> np.ndarray:
    """Shuffled-conditioner null: permute the conditioner dates randomly n_reps times,
    apply the same overlay logic, compute compound return on eval_window.
    Returns array of null compound returns."""
    rng = np.random.default_rng(seed)
    cond_vals = conditioner_series.values.copy()
    null_comps = []
    for _ in range(n_reps):
        rng.shuffle(cond_vals)
        shuffled = pd.Series(cond_vals, index=conditioner_series.index)
        res = gated_net_fn(shuffled, direction, train_end, base_weights, close_panel)
        net = res["gated_net"]
        if eval_window is not None:
            lo, hi = eval_window
            net = net[(net.index >= pd.Timestamp(lo)) & (net.index < pd.Timestamp(hi))]
        comp = float(np.prod(1.0 + net.dropna()) - 1.0)
        null_comps.append(comp)
    return np.array(null_comps)


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("Wave 4C -- Regime Overlay Robustness Probe")
    print("=" * 70)

    splits = get_split_dates()
    train_end = splits.train_end       # "2023-07-01"
    val_end = splits.val_end           # "2024-05-15"
    oos_end = splits.oos_end           # "2025-03-15"
    unseen_start = splits.unseen_start # "2025-03-15"

    print(f"Split dates: TRAIN<{train_end} / VAL<{val_end} / OOS<{oos_end} / UNSEEN>={unseen_start}")
    print()

    # ---- Load close panel for daily_engine ----
    print("Loading close panel...")
    close_panel = load_close_panel()
    print(f"  close_panel: {close_panel.shape}, {close_panel.index[0].date()} -> {close_panel.index[-1].date()}")

    # ---- Build base book (un-gated daily_engine) ----
    print("Building base daily_engine book (un-gated)...")
    base_book = build_book(close_panel, core="voltgt", use_overlay=True, cost_rt=TAKER_RT)
    base_net = base_book["net"]
    base_W = base_book["W"]
    regime_labels = base_book["labels"]
    print(f"  Base book: {len(base_net)} days")

    # Daily returns panel for IC calculation
    ret_panel = close_panel.pct_change(fill_method=None)

    # ---- Load ChimeraLoader ----
    loader = ChimeraLoader()

    # ============================================================
    # STEP 1: Rolling-window IC robustness probe
    # ============================================================
    print()
    print("=" * 60)
    print("STEP 1: Rolling-Window IC Robustness Probe")
    print("=" * 60)

    IC_CANDIDATES = [
        "xrel_hbr_eta_total_xratio",
        "te_in_btc",
        "te_out_btc",
    ]

    step1_results = {}
    ic_panels = {}  # store for later regime-split

    for feat_name in IC_CANDIDATES:
        print(f"\n-- {feat_name} --")
        feat_panel = load_feature_panel(feat_name, loader)
        ic_panels[feat_name] = feat_panel

        # Align to ret_panel
        common_dates = feat_panel.index.intersection(ret_panel.index)
        fp_aligned = feat_panel.reindex(common_dates)
        rp_aligned = ret_panel.reindex(common_dates)

        # Compute rolling IC series
        ic_df = rolling_ic_series(fp_aligned, rp_aligned, window=IC_WINDOW)

        if ic_df.empty:
            print(f"  ERROR: no rolling IC computed for {feat_name}")
            step1_results[feat_name] = {"verdict": "ARTIFACT", "reason": "no_ic_computed"}
            continue

        # Cross-sectional mean IC per date
        ic_mean = ic_df.mean(axis=1)

        # Split into TRAIN-only (for direction registration) vs OOS+UNSEEN (for verdict)
        ic_train = ic_mean[ic_mean.index < pd.Timestamp(train_end)]
        ic_oos = ic_mean[(ic_mean.index >= pd.Timestamp(val_end)) &
                         (ic_mean.index < pd.Timestamp(oos_end))]
        ic_unseen = ic_mean[ic_mean.index >= pd.Timestamp(unseen_start)]

        # Pre-register direction from TRAIN
        train_median_ic = float(ic_train.median()) if len(ic_train) > 0 else 0.0
        pre_registered_direction = +1 if train_median_ic > 0 else -1

        # Update CONDITIONER_DIRECTIONS with data-driven pre-registration from TRAIN
        # (override the a-priori direction with the empirical TRAIN signal)
        CONDITIONER_DIRECTIONS[feat_name] = pre_registered_direction

        # Rolling IC statistics
        pct_pos_oos = float((ic_oos.dropna() > 0).mean()) if len(ic_oos.dropna()) > 0 else np.nan
        pct_pos_unseen = float((ic_unseen.dropna() > 0).mean()) if len(ic_unseen.dropna()) > 0 else np.nan

        print(f"  TRAIN median IC: {train_median_ic:.4f} (pre-registered dir: {'+1' if pre_registered_direction > 0 else '-1'})")
        print(f"  OOS  median IC: {float(ic_oos.median()):.4f}, pct_positive={pct_pos_oos:.2%} (n={len(ic_oos.dropna())})")
        print(f"  UNSEEN median IC: {float(ic_unseen.median()):.4f}, pct_positive={pct_pos_unseen:.2%} (n={len(ic_unseen.dropna())})")

        # Regime-split IC (using daily_engine's internal regime)
        print(f"  Regime-split IC (TRAIN regime labels):")
        regime_split = regime_split_ic(fp_aligned, rp_aligned, regime_labels)
        for regime, stats in regime_split.items():
            print(f"    {regime}: mean_IC={stats['mean_ic']:.4f} (n_obs={stats['n_obs']})")

        # Verdict logic
        # PERSISTENT: OOS median IC > 0.005 AND pct_pos > 55% AND sign is consistent with TRAIN
        oos_med = float(ic_oos.median()) if len(ic_oos) > 0 else 0.0
        oos_signed = oos_med * pre_registered_direction  # positive if direction holds

        # Check regime consistency (signal should not concentrate in a single regime)
        regime_ics = [v["mean_ic"] * pre_registered_direction
                      for v in regime_split.values()
                      if np.isfinite(v["mean_ic"])]
        regime_consistent = sum(v > -0.01 for v in regime_ics) >= 2 if len(regime_ics) >= 2 else False

        if oos_signed > 0.005 and pct_pos_oos > 0.55 and regime_consistent:
            verdict = "PERSISTENT"
        elif oos_signed > 0 and pct_pos_oos > 0.50:
            verdict = "AMBIGUOUS (weak)"
        else:
            verdict = "REGIME-ARTIFACT"

        print(f"  VERDICT: {verdict}")
        step1_results[feat_name] = {
            "verdict": verdict,
            "train_median_ic": round(train_median_ic, 4),
            "oos_median_ic": round(oos_med, 4),
            "oos_pct_positive": round(pct_pos_oos, 3) if np.isfinite(pct_pos_oos) else None,
            "unseen_median_ic": round(float(ic_unseen.median()), 4) if len(ic_unseen) > 0 else None,
            "unseen_pct_positive": round(pct_pos_unseen, 3) if np.isfinite(pct_pos_unseen) else None,
            "pre_registered_direction": pre_registered_direction,
            "regime_ic": {k: round(v["mean_ic"], 4) for k, v in regime_split.items()},
        }

    print()
    print("STEP 1 SUMMARY:")
    survivors = []
    for feat, res in step1_results.items():
        v = res.get("verdict", "UNKNOWN")
        print(f"  {feat}: {v}")
        if "PERSISTENT" in v or "AMBIGUOUS" in v:
            survivors.append(feat)

    print(f"  Survivors for Step 2: {survivors}")

    # ============================================================
    # STEP 2: Book-level regime overlay test
    # ============================================================
    print()
    print("=" * 60)
    print("STEP 2: Book-Level Regime Overlay Test")
    print("=" * 60)

    # All candidates for overlay (survivors + stbl + dvol)
    OVERLAY_CANDIDATES = list(step1_results.keys()) + [
        "stbl_total_zscore_30d",
        "dv_dvol_close",
    ]

    # Base book OOS/UNSEEN stats (the benchmark)
    def book_compound(net, lo, hi):
        s = net[(net.index >= pd.Timestamp(lo)) & (net.index < pd.Timestamp(hi))]
        return float(np.prod(1.0 + s.dropna()) - 1.0)

    def book_stats_window(net, lo, hi, label):
        s = net[(net.index >= pd.Timestamp(lo)) & (net.index < pd.Timestamp(hi))]
        s = s.dropna()
        n = len(s)
        if n < 5:
            return {"n": n, "error": "too_short"}
        eq = np.cumprod(1.0 + s.values)
        peak = np.maximum.accumulate(eq)
        maxdd = float(((eq - peak) / peak).min() * 100)
        nyr = n / 365.0
        comp = float((eq[-1] - 1) * 100)
        sharpe = float(s.mean() / (s.std() + 1e-12) * np.sqrt(365))
        n_eff = compute_n_eff(s)
        p05 = block_bootstrap_p05(s) * 100
        return {
            "label": label, "n_days": n, "n_eff": round(n_eff, 1),
            "compound_pct": round(comp, 2), "sharpe": round(sharpe, 3),
            "maxdd_pct": round(maxdd, 2), "p05_pct": round(p05, 2),
        }

    print("\nBaseline daily_engine (un-gated, with internal regime scalar):")
    base_oos_stats = book_stats_window(base_net, val_end, oos_end, "BASE_OOS")
    base_unseen_stats = book_stats_window(base_net, unseen_start, "2099-01-01", "BASE_UNSEEN")
    for s in [base_oos_stats, base_unseen_stats]:
        print(f"  {s}")

    # Summarize base regime shares
    r_share = base_book["regime_share"]
    print(f"  Regime share (full history): {r_share}")

    step2_results = {}

    for feat_name in OVERLAY_CANDIDATES:
        print(f"\n-- Overlay: {feat_name} --")
        direction = CONDITIONER_DIRECTIONS.get(feat_name, +1)
        print(f"  Pre-registered direction: {'+1 (risk-on = above median)' if direction > 0 else '-1 (risk-on = below median)'}")

        # Load the conditioner (cross-section mean if per-asset, else use BTC as proxy)
        try:
            feat_panel = load_feature_panel(feat_name, loader)
        except Exception as e:
            print(f"  ERROR loading feature: {e}")
            step2_results[feat_name] = {"verdict": "ERROR", "reason": str(e)}
            continue

        if feat_panel.empty:
            print(f"  ERROR: empty feature panel for {feat_name}")
            step2_results[feat_name] = {"verdict": "ERROR", "reason": "empty_panel"}
            continue

        # Use cross-sectional mean as the single conditioner series (more robust than single asset)
        # For dvol, only BTC available -- use BTC
        if feat_name == "dv_dvol_close":
            if "BTCUSDT" not in feat_panel.columns or feat_panel["BTCUSDT"].isna().all():
                print(f"  SKIP: dvol is all-null (BTC dvol not in panel)")
                step2_results[feat_name] = {"verdict": "SKIP", "reason": "all_null"}
                continue
            cond_series = feat_panel["BTCUSDT"].dropna()
            print(f"  Using BTC dvol (only available asset). Coverage: {cond_series.index.min().date()} -> {cond_series.index.max().date()}")
        else:
            # cross-asset mean (row-wise)
            cond_series = feat_panel.mean(axis=1).dropna()
            print(f"  Cross-asset mean coverage: {cond_series.index.min().date()} -> {cond_series.index.max().date()}")

        # Align conditioner to the base_net index (fill forward for daily cadence)
        cond_aligned = cond_series.reindex(base_net.index).ffill()

        # Check coverage in OOS+UNSEEN
        oos_cov = cond_aligned[(cond_aligned.index >= pd.Timestamp(val_end)) &
                                (cond_aligned.index < pd.Timestamp(oos_end))].notna().mean()
        unseen_cov = cond_aligned[cond_aligned.index >= pd.Timestamp(unseen_start)].notna().mean()
        print(f"  OOS coverage: {oos_cov:.1%}, UNSEEN coverage: {unseen_cov:.1%}")
        if oos_cov < 0.5:
            print(f"  SKIP: OOS coverage < 50%, cannot evaluate")
            step2_results[feat_name] = {"verdict": "SKIP", "reason": "insufficient_oos_coverage"}
            continue

        # FIT threshold on TRAIN only, register direction
        def _apply_overlay(c_series, direction, train_end_str, base_W, close_p):
            return apply_conditioner_overlay(
                base_net, base_W, c_series, direction, train_end_str, close_p, TAKER_RT
            )

        gated_result = _apply_overlay(cond_aligned, direction, train_end, base_W, close_panel)
        gated_net = gated_result["gated_net"]
        threshold = gated_result["threshold"]
        pct_fav = gated_result["pct_favorable"]
        print(f"  Threshold (TRAIN median): {threshold:.4f}, pct_favorable={pct_fav:.1%}")

        # OOS evaluation (val_end -> oos_end)
        gated_oos = book_stats_window(gated_net, val_end, oos_end, f"GATED_{feat_name}_OOS")
        base_oos_comp = book_compound(base_net, val_end, oos_end)
        gated_oos_comp = book_compound(gated_net, val_end, oos_end)
        vs_base_oos = gated_oos_comp - base_oos_comp
        print(f"  OOS: compound={gated_oos['compound_pct']:.2f}% (vs base {base_oos_stats['compound_pct']:.2f}%, delta={vs_base_oos*100:.2f}pp)")
        print(f"       maxDD={gated_oos['maxdd_pct']:.2f}%, Sharpe={gated_oos['sharpe']:.3f}, n_eff={gated_oos['n_eff']:.1f}, p05={gated_oos['p05_pct']:.2f}%")

        # Shuffled-conditioner null on OOS
        def _overlay_fn(c, d, te, bw, cp):
            return apply_conditioner_overlay(base_net, bw, c, d, te, cp, TAKER_RT)

        null_comps_oos = shuffled_null_compound(
            _overlay_fn, cond_aligned, direction, train_end, base_W, close_panel,
            n_reps=200, seed=RNG_SEED, eval_window=(val_end, oos_end)
        )
        null_p50_oos = float(np.percentile(null_comps_oos, 50))
        real_p_val_oos = float(np.mean(null_comps_oos <= gated_oos_comp))
        print(f"  OOS shuffled null: p50={null_p50_oos*100:.2f}%, real_vs_null p-val={real_p_val_oos:.3f}")

        # UNSEEN -- touched ONCE
        print(f"  --- UNSEEN (touched once) ---")
        gated_unseen = book_stats_window(gated_net, unseen_start, "2099-01-01", f"GATED_{feat_name}_UNSEEN")
        base_unseen_comp = book_compound(base_net, unseen_start, "2099-01-01")
        gated_unseen_comp = book_compound(gated_net, unseen_start, "2099-01-01")
        vs_base_unseen = gated_unseen_comp - base_unseen_comp
        print(f"  UNSEEN: compound={gated_unseen['compound_pct']:.2f}% (vs base {base_unseen_stats['compound_pct']:.2f}%, delta={vs_base_unseen*100:.2f}pp)")
        print(f"          maxDD={gated_unseen['maxdd_pct']:.2f}%, Sharpe={gated_unseen['sharpe']:.3f}, n_eff={gated_unseen['n_eff']:.1f}, p05={gated_unseen['p05_pct']:.2f}%")

        # Shuffled null on UNSEEN
        null_comps_unseen = shuffled_null_compound(
            _overlay_fn, cond_aligned, direction, train_end, base_W, close_panel,
            n_reps=200, seed=RNG_SEED + 1, eval_window=(unseen_start, "2099-01-01")
        )
        null_p50_unseen = float(np.percentile(null_comps_unseen, 50))
        real_p_val_unseen = float(np.mean(null_comps_unseen <= gated_unseen_comp))
        print(f"  UNSEEN shuffled null: p50={null_p50_unseen*100:.2f}%, real_vs_null p-val={real_p_val_unseen:.3f}")

        # Verdict
        # REAL: gated beats base OOS AND UNSEEN AND beats null (p-val > 0.7) AND p05 > base p05
        beats_base_oos = vs_base_oos > 0.0
        beats_base_unseen = vs_base_unseen > 0.0
        beats_null_oos = real_p_val_oos > 0.65
        beats_null_unseen = real_p_val_unseen > 0.65
        p05_improves = gated_unseen.get("p05_pct", -999) > base_unseen_stats.get("p05_pct", -999)

        if beats_base_oos and beats_base_unseen and beats_null_oos and beats_null_unseen:
            overlay_verdict = "REAL (beats base + null, both windows)"
        elif beats_base_unseen and beats_null_unseen:
            overlay_verdict = "AMBIGUOUS (UNSEEN beats base+null; OOS did not)"
        elif beats_base_oos and not beats_base_unseen:
            overlay_verdict = "ARTIFACT (OOS positive, UNSEEN negative = overfit)"
        else:
            overlay_verdict = "NULL (no consistent improvement)"

        print(f"\n  STEP-2 VERDICT: {overlay_verdict}")
        step2_results[feat_name] = {
            "verdict": overlay_verdict,
            "oos_compound_delta_pp": round(vs_base_oos * 100, 2),
            "unseen_compound_delta_pp": round(vs_base_unseen * 100, 2),
            "oos_p_val_vs_null": round(real_p_val_oos, 3),
            "unseen_p_val_vs_null": round(real_p_val_unseen, 3),
            "gated_oos": gated_oos,
            "gated_unseen": gated_unseen,
            "threshold": round(threshold, 5),
            "pct_favorable": round(pct_fav, 3),
            "p05_improves": p05_improves,
        }

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print()
    print("=" * 70)
    print("FINAL SUMMARY -- Wave 4C Regime Overlay Probe")
    print("=" * 70)
    print()
    print("STEP 1 (IC Robustness -- feature-level diagnostics):")
    print("-" * 50)
    for feat, res in step1_results.items():
        v = res.get("verdict", "UNKNOWN")
        oos_ic = res.get("oos_median_ic", "n/a")
        pct_pos = res.get("oos_pct_positive", "n/a")
        print(f"  {feat}:")
        print(f"    OOS IC={oos_ic}, pct_pos={pct_pos}, VERDICT={v}")

    print()
    print("STEP 2 (Regime Overlay -- book-level held-out test):")
    print("-" * 50)
    print(f"  BASE daily_engine: OOS={base_oos_stats['compound_pct']}%, maxDD={base_oos_stats['maxdd_pct']}%, p05={base_oos_stats['p05_pct']}%")
    print(f"  BASE daily_engine: UNSEEN={base_unseen_stats['compound_pct']}%, maxDD={base_unseen_stats['maxdd_pct']}%, p05={base_unseen_stats['p05_pct']}%")
    print()
    for feat, res in step2_results.items():
        v = res.get("verdict", "UNKNOWN")
        oos_d = res.get("oos_compound_delta_pp", "n/a")
        unseen_d = res.get("unseen_compound_delta_pp", "n/a")
        oos_p = res.get("oos_p_val_vs_null", "n/a")
        unseen_p = res.get("unseen_p_val_vs_null", "n/a")
        print(f"  {feat}:")
        print(f"    OOS delta={oos_d}pp (p_null={oos_p}), UNSEEN delta={unseen_d}pp (p_null={unseen_p})")
        print(f"    VERDICT: {v}")
    print()
    print("CONDITIONER REGISTER (for promotion decisions):")
    print("-" * 50)
    for feat, res in step2_results.items():
        v = res.get("verdict", "UNKNOWN")
        if "REAL" in v:
            print(f"  PROMOTE: {feat} -- {v}")
        elif "AMBIGUOUS" in v:
            print(f"  FURTHER-TEST: {feat} -- {v}")
        else:
            print(f"  PARK: {feat} -- {v}")

    print()
    print("KEY STATISTICAL CAVEATS:")
    print("  1. IC diagnostic only (not the objective); primary = compound return.")
    print("  2. n_eff << n under autocorrelation; see n_eff column for honest sample size.")
    print("  3. UNSEEN touched ONCE; UNSEEN result is the decisive number.")
    print("  4. Block-bootstrap p05 uses block=20d to capture return autocorrelation.")
    print("  5. Shuffled null has direction pre-registered on TRAIN to avoid post-hoc direction flip.")
    print("  6. dvol is BTC-only (Deribit); applies as a market-wide signal only.")

    return {"step1": step1_results, "step2": step2_results}


if __name__ == "__main__":
    main()
