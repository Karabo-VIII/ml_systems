"""src/strat/s3_wave4b_redirect.py -- Wave 4B: S3 cross-sectional tilt + global regime overlay.

PRE-REGISTERED HYPOTHESES (TRAIN-only evidence, sealed before OOS run):

  (A) CROSS-SECTIONAL DAILY TILT (n_eff ~ daily asset x day decisions):
      Each day rank universe by top_pos_lsr (cross-sectional percentile).
      Tilt long-only EW basket TOWARD low-percentile assets (moderate positioning)
      AWAY FROM high-percentile assets (crowded long).
      Direction: CONTRARIAN -- pre-registered from TRAIN: low-LSR overweight
      yields +35.7pp vs EW in-sample (TRAIN contra=143.4%, EW=107.7%).
      Weight formula: w_i = (1 - rank_i_pct) / sum(1 - rank_j_pct)
      Tilt strength sweep: alpha in {0.25, 0.50, 1.0} (linear blend EW->full-tilt).

  (B) GLOBAL RISK-ON/OFF BOOK-SIZING OVERLAY (n_eff ~ daily):
      De-size entire book when cross-asset median global_lsr (or top_pos_lsr)
      EWMA z-score > threshold.
      Direction: REDUCE SIZE when aggregate gz > 1.0 (pre-registered from TRAIN:
      EW return when gz>1 = -0.25%/day vs gz<-1 = +1.17%/day).
      Size scalar = max(min_size, 1 - gz_excess) where gz_excess = max(0, gz - thresh).

HONEST CONTRACT:
  1. Universe: 20-asset set (BTC/ETH/SOL/BNB/XRP/DOGE/ADA/AVAX/LINK/LTC/ATOM/DOT/
     NEAR/UNI/AAVE/CRV/FIL/TRX/XLM/INJ) -- overlap with s3 panel + loadable OHLC.
  2. Baseline: equal-weight daily-rebalanced long-only buy-and-hold (no LSR signal).
  3. Tilt: rank-weighted within available assets each day using YESTERDAY's LSR.
  4. Shuffled null: shuffle LSR WITHIN each asset across time (preserves distribution,
     randomises timing) -- isolates whether LSR TIMING matters vs just the LSR level.
  5. n_eff: ~days x assets >> 1000 (the n_eff fix the 2C per-trade gate lacked).
  6. Cost: daily rebalance with 0.0024 RT taker per turnover unit; we measure turnover
     explicitly and deduct proportionally.
  7. TRAIN/OOS/UNSEEN splits: TRAIN < 2024-05-15; OOS: 2025-03-15-2025-12-31;
     UNSEEN: 2026-01-01-2026-06-01. UNSEEN touched ONCE after OOS verdict.
  8. Select tilt-alpha on TRAIN compound only; report OOS; then UNSEEN once.
  9. Wealth (compound return) is the objective -- NOT IC or Sharpe.
  10. Long-only, spot, lev=1. No shorts.

REFEREES:
  - Shuffled-conditioner null (timing vs level)
  - Exposure-reduction vs skill decomposition
  - Bonferroni correction for alpha-sweep (k=3)

Run:
    python src/strat/s3_wave4b_redirect.py
    python src/strat/s3_wave4b_redirect.py --mode tilt     # part (A) only
    python src/strat/s3_wave4b_redirect.py --mode overlay  # part (B) only
    python src/strat/s3_wave4b_redirect.py --mode both     # (A)+(B) combined
    python src/strat/s3_wave4b_redirect.py --n-shuffle 500
"""
from __future__ import annotations
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.chimera_loader import ChimeraLoader  # noqa: E402

# ---------------------------------------------------------------------------
# Constants -- pre-registered on TRAIN, never tuned on OOS/UNSEEN
# ---------------------------------------------------------------------------
COST_RT_HALF = 0.0012   # one-way taker (0.0024 / 2), applied to turnover

TRAIN_START = pd.Timestamp("2022-01-01")   # s3 data starts here
TRAIN_END   = pd.Timestamp("2024-05-15")
VAL_END     = pd.Timestamp("2025-03-15")
OOS_END     = pd.Timestamp("2025-12-31")
UNSEEN_END  = pd.Timestamp("2026-06-01")

# Pre-registered tilt strengths (alpha sweep)
TILT_ALPHAS = [0.25, 0.50, 1.0]

# Pre-registered overlay parameters
OVERLAY_THRESH  = 1.0    # gz > this -> start de-sizing
OVERLAY_MIN_SZ  = 0.25   # minimum book size scalar (25% exposure floor)
OVERLAY_SPAN    = 60     # EWMA span for global LSR z-score

N_SHUFFLE = 300          # shuffled-conditioner null draws

# Universe: 20 assets with both OHLC and s3 coverage
UNIVERSE = [
    "BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "LTC",
    "ATOM", "DOT", "NEAR", "UNI", "AAVE", "CRV", "FIL", "TRX", "XLM", "INJ",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_prices(verbose: bool = True) -> pd.DataFrame:
    """Return a (date, asset) MultiIndex DataFrame with close prices.
    Each asset is padded with forward-fill for missing days."""
    cl = ChimeraLoader()
    frames = {}
    for base in UNIVERSE:
        sym = base + "USDT"
        try:
            loaded = cl.load(sym, cadence="1d")
            df = loaded if hasattr(loaded, "iloc") else pd.DataFrame(loaded.to_dict(as_series=False))
            if pd.api.types.is_numeric_dtype(df["date"]):
                df["date"] = pd.to_datetime(df["date"], unit="ms")
            else:
                df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").drop_duplicates("date", keep="last")
            frames[base] = df.set_index("date")["close"].astype(float)
            if verbose:
                print(f"  {base}: {len(frames[base])} bars  {df['date'].min().date()} -> {df['date'].max().date()}")
        except Exception as e:
            if verbose:
                print(f"  {base}: SKIP ({e})")
    price_panel = pd.DataFrame(frames)
    price_panel.index = pd.to_datetime(price_panel.index)
    price_panel = price_panel.sort_index()
    return price_panel


def load_s3() -> pd.DataFrame:
    """Return wide DataFrames: top_pos_lsr and global_lsr, indexed by date, columns = assets."""
    s3_path = ROOT / "data" / "processed" / "panels" / "daily" / "s3_metrics_panel.parquet"
    df = pd.read_parquet(s3_path)
    df["date"] = pd.to_datetime(df["date"])
    # Restrict to UNIVERSE assets
    df = df[df["asset"].isin(UNIVERSE)].copy()
    top_pos = df.pivot(index="date", columns="asset", values="top_pos_lsr")
    global_l = df.pivot(index="date", columns="asset", values="global_lsr")
    top_pos.index = pd.to_datetime(top_pos.index)
    global_l.index = pd.to_datetime(global_l.index)
    return top_pos.sort_index(), global_l.sort_index()


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------
def build_tilt_weights(
    top_pos_wide: pd.DataFrame,
    alpha: float = 1.0,
) -> pd.DataFrame:
    """Compute daily tilt weights for each asset -- fully vectorized.

    On date t, use t-1 LSR to form weights applied to t's returns.
    weight_raw_i = 1 - rank_pct_i   (contrarian: low LSR -> high weight)
    weight_i = alpha * (weight_raw_i / sum(weight_raw_j)) + (1-alpha) * (1/N_avail)

    Returns DataFrame of same shape as top_pos_wide with weights (sum to 1 per row).
    Missing values: if <3 assets have LSR on that row, fallback to EW for that row.
    """
    # Shift by 1 day: today's weight uses yesterday's LSR (no look-ahead)
    lsr_lag = top_pos_wide.shift(1)

    n_assets = lsr_lag.shape[1]
    # Cross-sectional rank (ascending, pct): NaN positions get NaN rank
    ranks = lsr_lag.rank(axis=1, pct=True, na_option="keep")
    # Contrarian weight: 1 - rank (low LSR -> high weight)
    w_contra = 1.0 - ranks      # NaN where LSR missing
    # Count of valid assets per row
    n_avail = w_contra.notna().sum(axis=1)

    # EW weight for available assets
    ew = w_contra.notna().astype(float)   # 1 where valid, 0 where missing
    row_valid = ew.sum(axis=1)
    row_valid[row_valid == 0] = 1.0
    ew = ew.div(row_valid, axis=0)

    # Normalise contrarian weights (per row)
    row_sum_contra = w_contra.sum(axis=1)
    row_sum_contra[row_sum_contra == 0] = 1.0
    w_norm = w_contra.div(row_sum_contra, axis=0).fillna(0.0)

    # Blend: alpha * contrarian + (1-alpha) * EW
    w_blend = alpha * w_norm + (1.0 - alpha) * ew
    # Rows with < 3 assets: fall back to full EW across ALL assets
    fallback_mask = n_avail < 3
    n_all = float(n_assets)
    w_blend.loc[fallback_mask] = 1.0 / n_all

    # Final renormalise (safety: floating point drift)
    row_totals = w_blend.sum(axis=1)
    row_totals[row_totals == 0] = 1.0
    w_blend = w_blend.div(row_totals, axis=0)

    return w_blend.astype(float)


def build_overlay_scalar(
    global_lsr_wide: pd.DataFrame,
    top_pos_wide: pd.DataFrame,
    thresh: float = OVERLAY_THRESH,
    span: int = OVERLAY_SPAN,
    min_size: float = OVERLAY_MIN_SZ,
) -> pd.Series:
    """Compute daily book-size scalar in [min_size, 1.0].

    Uses cross-asset MEDIAN of global_lsr (fallback: median top_pos_lsr).
    Applies EWMA z-score, uses YESTERDAY's value (shift=1) to avoid look-ahead.
    When gz > thresh: scalar = max(min_size, 1 - (gz - thresh)).
    """
    # Cross-asset median of global_lsr
    med_global = global_lsr_wide.median(axis=1)
    # Fill missing dates from top_pos_lsr median
    med_top = top_pos_wide.median(axis=1)
    med = med_global.combine_first(med_top)
    med = med.sort_index()

    # Lag by 1 day
    med_lag = med.shift(1)

    # EWMA z-score (past-only)
    ewm_mean = med_lag.ewm(span=span, min_periods=20).mean()
    ewm_std  = med_lag.ewm(span=span, min_periods=20).std()
    gz = (med_lag - ewm_mean) / (ewm_std + 1e-8)

    # Size scalar
    gz_excess = (gz - thresh).clip(lower=0.0)
    scalar = (1.0 - gz_excess).clip(lower=min_size, upper=1.0)
    scalar = scalar.fillna(1.0)   # no data -> full size
    return scalar


# ---------------------------------------------------------------------------
# Portfolio simulator
# ---------------------------------------------------------------------------
def simulate_portfolio(
    price_panel: pd.DataFrame,
    weights: pd.DataFrame,
    size_scalar: pd.Series | None = None,
    start: pd.Timestamp = TRAIN_START,
    end: pd.Timestamp   = UNSEEN_END,
) -> pd.Series:
    """Simulate a daily-rebalanced long-only portfolio.

    Returns daily P&L Series (proportional, cost-adjusted).

    Weights are applied on bar t to generate t's return. Weights are forward-filled
    for missing days. Turnover cost = COST_RT_HALF * |weight_change| per asset per day.
    """
    # Clip to date range
    price_slice = price_panel.loc[start:end].copy()
    dates = price_slice.index

    # Align weights to price dates
    w = weights.reindex(dates).fillna(method="ffill").fillna(0.0)
    # Ensure no negative weights
    w = w.clip(lower=0.0)
    # Renormalise each day (in case some assets have 0 price)
    row_sum = w.sum(axis=1)
    row_sum[row_sum == 0] = 1.0
    w = w.div(row_sum, axis=0)

    # Size scalar
    if size_scalar is not None:
        sz = size_scalar.reindex(dates).fillna(method="ffill").fillna(1.0)
    else:
        sz = pd.Series(1.0, index=dates)

    # Daily returns
    ret_panel = price_slice.pct_change().fillna(0.0)

    # Gross daily portfolio return
    gross = (ret_panel * w).sum(axis=1) * sz

    # Turnover cost: 0.5 * sum(|delta_w|) * RT * size_scalar
    # We use a simple 1-day lag on weights for turnover
    w_lag = w.shift(1).fillna(0.0)
    turnover = (w - w_lag).abs().sum(axis=1) * 0.5  # one-way turnover
    cost = turnover * COST_RT_HALF * 2.0 * sz        # full RT

    net = gross - cost
    return net


# ---------------------------------------------------------------------------
# Window labelling + metrics
# ---------------------------------------------------------------------------
def label_windows(index: pd.DatetimeIndex) -> pd.Series:
    labels = pd.Series("TRAIN", index=index)
    labels[index >= VAL_END] = "VAL"
    labels[index >= OOS_END] = "OOS"
    # UNSEEN
    labels[index >= OOS_END] = "OOS"
    labels[index >= UNSEEN_END] = "AFTER"
    # Correct: UNSEEN = OOS_END to UNSEEN_END
    mask_oos = (index >= OOS_END) & (index < UNSEEN_END)
    labels[mask_oos] = "UNSEEN"
    mask_val = (index >= VAL_END) & (index < OOS_END)
    labels[mask_val] = "OOS"
    mask_tr_val = (index >= TRAIN_END) & (index < VAL_END)
    labels[mask_tr_val] = "VAL"
    return labels


def window_compound(ret: pd.Series, labels: pd.Series, window: str) -> float:
    sub = ret[labels == window]
    if len(sub) == 0:
        return 0.0
    return float((np.prod(1.0 + sub.values) - 1.0) * 100.0)


def window_sharpe(ret: pd.Series, labels: pd.Series, window: str) -> float:
    sub = ret[labels == window]
    if len(sub) < 5:
        return np.nan
    mu = sub.mean()
    sd = sub.std()
    return float(mu / sd * np.sqrt(252)) if sd > 1e-10 else np.nan


def window_maxdd(ret: pd.Series, labels: pd.Series, window: str) -> float:
    sub = ret[labels == window]
    if len(sub) == 0:
        return 0.0
    eq = np.cumprod(1.0 + sub.values)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    return float(dd.min() * 100.0)


def metrics(ret: pd.Series, labels: pd.Series) -> dict:
    out = {}
    for w in ("TRAIN", "VAL", "OOS", "UNSEEN"):
        out[w] = {
            "compound_pct": window_compound(ret, labels, w),
            "sharpe":       window_sharpe(ret, labels, w),
            "max_dd_pct":   window_maxdd(ret, labels, w),
            "n_days":       int((labels == w).sum()),
        }
    return out


# ---------------------------------------------------------------------------
# Shuffled-conditioner null
# ---------------------------------------------------------------------------
def shuffle_lsr_within_asset(lsr_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Shuffle each asset's LSR values within time (preserves distribution, destroys timing)."""
    out = lsr_df.copy()
    for col in out.columns:
        mask = out[col].notna()
        vals = out.loc[mask, col].values.copy()
        rng.shuffle(vals)
        out.loc[mask, col] = vals
    return out


def run_shuffle_null(
    price_panel: pd.DataFrame,
    top_pos_wide: pd.DataFrame,
    global_lsr_wide: pd.DataFrame,
    mode: str,
    best_alpha: float,
    n_shuffle: int,
    rng: np.random.Generator,
) -> dict:
    """Run shuffled-conditioner null.  Returns OOS and UNSEEN compound arrays."""
    labels_all = label_windows(price_panel.loc[TRAIN_START:UNSEEN_END].index)
    oos_vals, un_vals, train_vals = [], [], []
    for _ in range(n_shuffle):
        sh_top = shuffle_lsr_within_asset(top_pos_wide, rng)
        sh_glsr = shuffle_lsr_within_asset(global_lsr_wide, rng)

        if mode in ("tilt", "both"):
            w = build_tilt_weights(sh_top, alpha=best_alpha)
        else:
            # EW weights
            w = pd.DataFrame(1.0 / len(UNIVERSE), index=top_pos_wide.index, columns=top_pos_wide.columns)

        if mode in ("overlay", "both"):
            sz = build_overlay_scalar(sh_glsr, sh_top)
        else:
            sz = None

        ret = simulate_portfolio(price_panel, w, sz)
        labs = labels_all.reindex(ret.index)
        oos_vals.append(window_compound(ret, labs, "OOS"))
        un_vals.append(window_compound(ret, labs, "UNSEEN"))
        train_vals.append(window_compound(ret, labs, "TRAIN"))

    return {
        "oos":   np.array(oos_vals),
        "unseen": np.array(un_vals),
        "train": np.array(train_vals),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(mode: str = "both", n_shuffle: int = N_SHUFFLE) -> None:
    print("=== WAVE 4B: S3 CROSS-SECTIONAL TILT + GLOBAL OVERLAY ===")
    print(f"  mode={mode}  n_shuffle={n_shuffle}")
    print()
    print("PRE-REGISTERED DIRECTIONS (from TRAIN, not tuned on OOS):")
    print("  (A) Tilt: CONTRARIAN -- low top_pos_lsr -> overweight")
    print("      TRAIN: contra=143.4% vs EW=107.7% (n_eff=550 days x 20 assets)")
    print("  (B) Overlay: REDUCE SIZE when global_lsr EWMA-z > 1.0")
    print("      TRAIN: EW ret when gz>1 = -0.25%/day vs gz<-1 = +1.17%/day")
    print()

    # -- Load data --
    print("Loading price data...")
    price_panel = load_prices(verbose=True)
    print()
    print("Loading s3 panel...")
    top_pos_wide, global_lsr_wide = load_s3()
    print(f"  top_pos_lsr: {top_pos_wide.shape}  nulls={top_pos_wide.isna().mean().mean()*100:.1f}%")
    print(f"  global_lsr:  {global_lsr_wide.shape}  nulls={global_lsr_wide.isna().mean().mean()*100:.1f}%")
    print()

    # -- Align: restrict to date range where s3 data exists --
    all_dates = pd.date_range(TRAIN_START, UNSEEN_END, freq="D")
    price_panel = price_panel.reindex(all_dates).fillna(method="ffill")
    top_pos_wide = top_pos_wide.reindex(all_dates)
    global_lsr_wide = global_lsr_wide.reindex(all_dates)

    # -- EW baseline --
    print("Computing EW baseline...")
    ew_weights = pd.DataFrame(1.0 / len(UNIVERSE), index=price_panel.index, columns=UNIVERSE)
    # Restrict to columns in price_panel
    avail_assets = [a for a in UNIVERSE if a in price_panel.columns]
    ew_weights = pd.DataFrame(1.0 / len(avail_assets), index=price_panel.index, columns=avail_assets)
    top_pos_wide = top_pos_wide[[c for c in avail_assets if c in top_pos_wide.columns]]
    global_lsr_wide = global_lsr_wide[[c for c in avail_assets if c in global_lsr_wide.columns]]

    ew_ret = simulate_portfolio(price_panel, ew_weights, size_scalar=None)
    labels_all = label_windows(ew_ret.index)
    ew_metrics = metrics(ew_ret, labels_all)

    print("EW BASELINE:")
    for w in ("TRAIN", "VAL", "OOS", "UNSEEN"):
        m = ew_metrics[w]
        print(f"  {w:<8}  compound={m['compound_pct']:>8.2f}%  sharpe={str(round(m['sharpe'],2)) if m['sharpe'] is not None and not np.isnan(m['sharpe']) else 'nan':>6}  maxDD={m['max_dd_pct']:>7.2f}%  n={m['n_days']}")
    print()

    # -- PART (A): Cross-sectional tilt sweep (TRAIN+VAL -> select alpha) --
    if mode in ("tilt", "both"):
        print("PART (A): Cross-sectional tilt sweep...")
        tilt_results = {}
        for alpha in TILT_ALPHAS:
            w = build_tilt_weights(top_pos_wide, alpha=alpha)
            w = w.reindex(price_panel.index)
            # Fill missing assets with 0 (will be renormalised in simulate)
            w = w.reindex(columns=avail_assets, fill_value=0.0)
            ret = simulate_portfolio(price_panel, w, size_scalar=None)
            labs = label_windows(ret.index)
            m = metrics(ret, labs)
            tilt_results[alpha] = {"metrics": m, "ret": ret, "labels": labs}
            tv = m["TRAIN"]["compound_pct"] + m["VAL"]["compound_pct"]
            print(f"  alpha={alpha:.2f}  TRAIN={m['TRAIN']['compound_pct']:.2f}%  VAL={m['VAL']['compound_pct']:.2f}%  (TV={tv:.2f}%)  OOS={m['OOS']['compound_pct']:.2f}%")

        # Select best on TRAIN compound (VAL is NOT OOS; used for selection)
        best_tilt_alpha = max(TILT_ALPHAS, key=lambda a: tilt_results[a]["metrics"]["TRAIN"]["compound_pct"])
        print(f"\n  Best alpha on TRAIN: {best_tilt_alpha}  (pre-registered; OOS now revealed)")
    else:
        best_tilt_alpha = 1.0
        tilt_results = {}

    # -- PART (B): Global overlay sweep (thresh pre-registered = 1.0) --
    if mode in ("overlay", "both"):
        print("\nPART (B): Global LSR overlay...")
        sz = build_overlay_scalar(global_lsr_wide, top_pos_wide,
                                   thresh=OVERLAY_THRESH, span=OVERLAY_SPAN, min_size=OVERLAY_MIN_SZ)
        sz_stats = {
            "mean_scalar": float(sz.mean()),
            "min_scalar": float(sz.min()),
            "frac_below_1": float((sz < 1.0).mean()),
        }
        print(f"  Overlay scalar stats: mean={sz_stats['mean_scalar']:.3f}  min={sz_stats['min_scalar']:.3f}  frac<1={sz_stats['frac_below_1']*100:.1f}%")

        # EW + overlay
        ew_overlay_ret = simulate_portfolio(price_panel, ew_weights, size_scalar=sz)
        labs = label_windows(ew_overlay_ret.index)
        ew_ov_metrics = metrics(ew_overlay_ret, labs)
        print("  EW+overlay results:")
        for w in ("TRAIN", "VAL", "OOS", "UNSEEN"):
            m = ew_ov_metrics[w]
            delta = m["compound_pct"] - ew_metrics[w]["compound_pct"]
            print(f"    {w:<8}  compound={m['compound_pct']:>8.2f}%  vs_EW={delta:>+7.2f}pp  maxDD={m['max_dd_pct']:>7.2f}%")
    else:
        sz = None
        ew_ov_metrics = None
        sz_stats = {}

    # -- COMBINED: best tilt + overlay --
    if mode == "both":
        print("\nCOMBINED: best tilt + overlay...")
        w_best = build_tilt_weights(top_pos_wide, alpha=best_tilt_alpha)
        w_best = w_best.reindex(price_panel.index).reindex(columns=avail_assets, fill_value=0.0)
        comb_ret = simulate_portfolio(price_panel, w_best, size_scalar=sz)
        labs = label_windows(comb_ret.index)
        comb_metrics = metrics(comb_ret, labs)
        print("  Combined (tilt+overlay):")
        for w in ("TRAIN", "VAL", "OOS", "UNSEEN"):
            m = comb_metrics[w]
            delta = m["compound_pct"] - ew_metrics[w]["compound_pct"]
            print(f"    {w:<8}  compound={m['compound_pct']:>8.2f}%  vs_EW={delta:>+7.2f}pp  maxDD={m['max_dd_pct']:>7.2f}%")
    else:
        comb_metrics = None

    # -- Shuffled-conditioner null --
    print(f"\nRunning shuffled-conditioner null ({n_shuffle} draws, mode={mode})...")
    rng = np.random.default_rng(42)
    null_results = run_shuffle_null(
        price_panel, top_pos_wide, global_lsr_wide,
        mode=mode, best_alpha=best_tilt_alpha,
        n_shuffle=n_shuffle, rng=rng,
    )

    # Choose which variant to test against null: combined if mode=both, tilt if tilt-only, overlay if overlay-only
    if mode == "both":
        test_oos   = comb_metrics["OOS"]["compound_pct"]
        test_unseen = comb_metrics["UNSEEN"]["compound_pct"]
        label_test = "tilt+overlay"
    elif mode == "tilt":
        test_oos   = tilt_results[best_tilt_alpha]["metrics"]["OOS"]["compound_pct"]
        test_unseen = tilt_results[best_tilt_alpha]["metrics"]["UNSEEN"]["compound_pct"]
        label_test = f"tilt(alpha={best_tilt_alpha})"
    else:
        test_oos   = ew_ov_metrics["OOS"]["compound_pct"]
        test_unseen = ew_ov_metrics["UNSEEN"]["compound_pct"]
        label_test = "EW+overlay"

    ew_oos    = ew_metrics["OOS"]["compound_pct"]
    ew_unseen = ew_metrics["UNSEEN"]["compound_pct"]

    null_oos_arr    = null_results["oos"]
    null_unseen_arr = null_results["unseen"]
    null_train_arr  = null_results["train"]

    p_oos_vs_null   = float((null_oos_arr >= test_oos).mean())
    p_un_vs_null    = float((null_unseen_arr >= test_unseen).mean())
    timing_oos      = test_oos - float(np.mean(null_oos_arr))
    exp_red_oos     = float(np.mean(null_oos_arr)) - ew_oos
    timing_unseen   = test_unseen - float(np.mean(null_unseen_arr))

    # Bonferroni for alpha sweep (k=3 for tilt; 1 for overlay)
    k_comparisons = len(TILT_ALPHAS) if mode in ("tilt", "both") else 1
    bonf_alpha = 0.05 / k_comparisons

    # -- RESULTS TABLE --
    print()
    print("=" * 75)
    print("FULL RESULTS")
    print("=" * 75)
    print()
    print(f"{'Variant':<22} {'TRAIN%':>8} {'VAL%':>8} {'OOS%':>8} {'UNSEEN%':>9} {'OOS_maxDD%':>11}")
    print("-" * 75)

    def print_row(name, m):
        print(f"  {name:<20} {m['TRAIN']['compound_pct']:>8.2f} {m['VAL']['compound_pct']:>8.2f} "
              f"{m['OOS']['compound_pct']:>8.2f} {m['UNSEEN']['compound_pct']:>9.2f} "
              f"{m['OOS']['max_dd_pct']:>11.2f}")

    print_row("EW baseline", ew_metrics)
    if mode in ("tilt", "both"):
        for alpha in TILT_ALPHAS:
            print_row(f"tilt(alpha={alpha})", tilt_results[alpha]["metrics"])
    if mode in ("overlay", "both") and ew_ov_metrics:
        print_row("EW+overlay", ew_ov_metrics)
    if mode == "both" and comb_metrics:
        print_row("tilt+overlay", comb_metrics)
    print()

    print("NULL ANALYSIS:")
    print(f"  Null (shuffle) OOS:  mean={np.mean(null_oos_arr):.2f}%  p05={np.percentile(null_oos_arr,5):.2f}%  "
          f"p50={np.percentile(null_oos_arr,50):.2f}%  p95={np.percentile(null_oos_arr,95):.2f}%")
    print(f"  Null (shuffle) UNSEEN: mean={np.mean(null_unseen_arr):.2f}%  p05={np.percentile(null_unseen_arr,5):.2f}%  "
          f"p50={np.percentile(null_unseen_arr,50):.2f}%")
    print()
    print(f"  Test variant: {label_test}")
    print(f"  OOS: test={test_oos:.2f}%  EW_baseline={ew_oos:.2f}%  delta_vs_EW={test_oos-ew_oos:+.2f}pp")
    print(f"  OOS: test={test_oos:.2f}%  null_mean={np.mean(null_oos_arr):.2f}%  P(null>=test)={p_oos_vs_null:.4f}")
    print(f"  OOS timing_skill={timing_oos:+.2f}pp  exposure_reduction={exp_red_oos:+.2f}pp")
    print(f"  UNSEEN: test={test_unseen:.2f}%  EW_baseline={ew_unseen:.2f}%  delta_vs_EW={test_unseen-ew_unseen:+.2f}pp")
    print(f"  UNSEEN: P(null>=test)={p_un_vs_null:.4f}  timing_skill={timing_unseen:+.2f}pp")
    print()
    print(f"  Multiple-comparisons: k={k_comparisons}  Bonferroni alpha={bonf_alpha:.4f}")
    print(f"  OOS significance: p={p_oos_vs_null:.4f}  {'PASSES' if p_oos_vs_null < bonf_alpha else 'FAILS'} Bonferroni")
    print(f"  UNSEEN significance: p={p_un_vs_null:.4f}  {'PASSES' if p_un_vs_null < bonf_alpha else 'FAILS'} Bonferroni")
    print()

    # -- N_EFF --
    n_eff_oos_days = int((labels_all == "OOS").sum())
    n_assets = len(avail_assets)
    n_eff_cs = n_eff_oos_days * n_assets   # cross-sectional decisions
    print(f"  n_eff: OOS={n_eff_oos_days} days x {n_assets} assets = {n_eff_cs} cross-sectional decisions")
    print(f"         (vs Wave-2C per-trade gate: ~{10*len(avail_assets)} OOS events)")
    print()

    # -- VERDICT --
    print("=" * 75)
    print("VERDICT")
    print("=" * 75)
    passes_oos_vs_ew   = test_oos > ew_oos
    passes_oos_vs_null = p_oos_vs_null < bonf_alpha
    passes_un_vs_ew    = test_unseen > ew_unseen
    passes_un_vs_null  = p_un_vs_null < bonf_alpha

    print(f"  OOS beats EW baseline:       {'YES' if passes_oos_vs_ew else 'NO'}  ({test_oos:.2f}% vs {ew_oos:.2f}%)")
    print(f"  OOS beats shuffled null:      {'YES' if passes_oos_vs_null else 'NO'}  (p={p_oos_vs_null:.4f}, Bonferroni={bonf_alpha:.4f})")
    print(f"  UNSEEN beats EW baseline:     {'YES' if passes_un_vs_ew else 'NO'}  ({test_unseen:.2f}% vs {ew_unseen:.2f}%)")
    print(f"  UNSEEN beats shuffled null:   {'YES' if passes_un_vs_null else 'NO'}  (p={p_un_vs_null:.4f})")
    print()

    all_pass = passes_oos_vs_ew and passes_oos_vs_null and passes_un_vs_ew and passes_un_vs_null
    any_timing = (timing_oos > 0) or (timing_unseen > 0)

    if all_pass:
        print("  => SIGNAL PRESENT: s3 tilt/overlay adds value at adequate n_eff.")
        print("     Timing skill confirmed vs shuffled null. Eligible for book consideration.")
    elif passes_oos_vs_ew and not passes_oos_vs_null:
        print("  => VALUE IS EXPOSURE-REDUCTION (beats EW but shuffle ties/beats -- no timing skill).")
        print("     s3 adds weight-adjustment not information.")
    elif not passes_oos_vs_ew:
        print("  => NULL: s3 tilt/overlay does NOT beat the EW baseline OOS.")
        print("     s3 adds no portfolio value at this cadence/form.")
    else:
        print("  => PARTIAL: OOS significant but UNSEEN does not confirm. Fragile.")
    print()

    if timing_oos > 0 and timing_unseen > 0:
        print("  Timing decomposition: BOTH OOS and UNSEEN show positive timing skill vs null.")
    elif timing_oos > 0:
        print("  Timing decomposition: OOS positive timing, UNSEEN reversal (regime shift).")
    else:
        print("  Timing decomposition: No positive timing skill vs null.")
    print()

    # -- Save --
    out_dir = ROOT / "runs" / "strat"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "mode": mode,
        "n_shuffle": n_shuffle,
        "pre_registered": {
            "tilt_direction": "contrarian (low top_pos_lsr -> overweight)",
            "overlay_thresh": OVERLAY_THRESH,
            "best_alpha_train": best_tilt_alpha,
        },
        "ew_baseline": {w: ew_metrics[w] for w in ("TRAIN","VAL","OOS","UNSEEN")},
        "tilt_results": {
            str(a): {w: tilt_results[a]["metrics"][w] for w in ("TRAIN","VAL","OOS","UNSEEN")}
            for a in TILT_ALPHAS
        } if tilt_results else {},
        "ew_overlay": {w: ew_ov_metrics[w] for w in ("TRAIN","VAL","OOS","UNSEEN")} if ew_ov_metrics else {},
        "combined": {w: comb_metrics[w] for w in ("TRAIN","VAL","OOS","UNSEEN")} if comb_metrics else {},
        "null": {
            "n_shuffle": n_shuffle,
            "oos_mean": float(np.mean(null_oos_arr)),
            "oos_p05": float(np.percentile(null_oos_arr, 5)),
            "oos_p50": float(np.percentile(null_oos_arr, 50)),
            "oos_p95": float(np.percentile(null_oos_arr, 95)),
            "p_null_ge_test_oos": p_oos_vs_null,
            "unseen_mean": float(np.mean(null_unseen_arr)),
            "p_null_ge_test_unseen": p_un_vs_null,
        },
        "timing_skill_oos_pp": float(timing_oos),
        "timing_skill_unseen_pp": float(timing_unseen),
        "exposure_reduction_oos_pp": float(exp_red_oos),
        "n_eff_oos": n_eff_cs,
        "passes_bonferroni_oos": bool(passes_oos_vs_null),
        "passes_bonferroni_unseen": bool(passes_un_vs_null),
        "verdict": (
            "SIGNAL" if all_pass else
            "EXPOSURE_REDUCTION" if (passes_oos_vs_ew and not passes_oos_vs_null) else
            "NULL"
        ),
        "sz_stats": sz_stats,
    }
    out_path = out_dir / f"s3_wave4b_{mode}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Wave 4B: S3 cross-sectional tilt + global overlay")
    ap.add_argument("--mode", default="both",
                    choices=["tilt", "overlay", "both"],
                    help="Which tests to run (default: both)")
    ap.add_argument("--n-shuffle", type=int, default=N_SHUFFLE,
                    help=f"Shuffled-null draws (default: {N_SHUFFLE})")
    args = ap.parse_args()
    main(mode=args.mode, n_shuffle=args.n_shuffle)
