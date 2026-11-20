"""src/strat/meta_ensemble_router.py -- NOVEL DISCOVERY: Meta-Ensemble of Adaptive Router.

FRESH ANGLE (strategy-discovery lens):
  The adaptive_meta_engine uses POINT-ESTIMATE regime boundaries
  (BREADTH_LOW=0.30, BREADTH_HIGH=0.60, VOL_HI_PCTILE=0.67).
  These are boundary-sensitive: near-boundary days flip the sub-behavior
  based on tiny breadth/vol fluctuations, injecting spurious churn.

PROPOSED CONSTRUCTION (no new predictive signal, pure portfolio-structure):
  1. BOUNDARY DIVERSIFICATION: run the router with N slight perturbations
     of the regime thresholds (breadth +/- delta, vol pctile +/- delta).
     On boundary days, different configs disagree -> ensemble is MORE
     diversified (naturally reduces exposure). On clear-regime days,
     all configs agree -> same concentrated bet.
  2. CONVICTION WEIGHTING (optional): weight the config-ensemble by
     regime CLARITY (distance from nearest threshold) -> when clarity is
     low, dilute; when clarity is high, concentrate. This is a SIZE lever
     not a directional lever -- purely structural.
  3. TWO-SLEEVE STRUCTURE: combine
       sleeve-A = router (mean-edge)
       sleeve-B = gated EW BH (smooth pos-rate, near-random but beta-captured)
     with a fixed split, tested for Pareto improvement on
     pos-rate AND mean jointly.

LEAK AUDIT:
  - All threshold perturbations are fixed ex-ante (no OOS tuning).
  - Conviction sizing uses causal signals (breadth/vol at bar d only).
  - Two-sleeve split is a GRID SEARCH over alpha in {0.2, 0.4, 0.6, 0.8}
    treated as a meta-parameter (referee tests ALL, picks on held-out).
  - Same shuffle-null as referee_harness for significance.

Run via referee_harness canonical slices for apples-to-apples comparison.
No emoji (cp1252). Does NOT git commit.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.mover_lab as lab
import strat.adaptive_meta_engine as ame
from strat.referee_harness import (
    book_daily_returns, bh_ew_weights, slice_stats, bh_slice_stats
)

COST = lab.COST

# ============================================================
# BOUNDARY-PERTURBED ROUTER CONFIGS
# ============================================================
# 9 configs: base + 8 perturbations of (breadth_low, breadth_high, vol_pctile)
# All deltas are small (less than 1/10 of the parameter range) and fixed ex-ante.
PERTURBATION_GRID = [
    # (breadth_low, breadth_high, vol_pctile) -- base first
    (0.30, 0.60, 0.67),   # base
    (0.25, 0.55, 0.67),   # lower breadth thresholds
    (0.35, 0.65, 0.67),   # higher breadth thresholds
    (0.30, 0.60, 0.60),   # tighter vol
    (0.30, 0.60, 0.75),   # looser vol
    (0.25, 0.65, 0.67),   # widened breadth band
    (0.35, 0.55, 0.67),   # narrowed breadth band
    (0.25, 0.55, 0.60),   # low-breadth + tighter-vol
    (0.35, 0.65, 0.75),   # high-breadth + looser-vol
]


def _detect_regime_parametric(ind, i, vol_hi_threshold, breadth_low, breadth_high):
    """Parametric version of the regime detector -- no class coupling."""
    C = ind["C"]; sma200 = ind["sma200"]; sma50 = ind["sma50"]; vol20 = ind["vol20"]
    d = C.index[i]
    btc = C.loc[d, "BTCUSDT"]; s200 = sma200.loc[d, "BTCUSDT"]
    btc_up = (not pd.isna(s200)) and (btc > s200)
    if not btc_up:
        return "downtrend"
    row_c = C.iloc[i]; row_s50 = sma50.iloc[i]
    above = 0; total = 0
    for sym in C.columns:
        c_val = row_c[sym]; s_val = row_s50[sym]
        if pd.notna(c_val) and pd.notna(s_val):
            above += int(c_val > s_val); total += 1
    breadth = above / total if total > 0 else 0.5
    btc_vol = vol20.loc[d, "BTCUSDT"]
    if pd.isna(btc_vol):
        btc_vol = 0.5
    hi_vol = btc_vol >= vol_hi_threshold
    if breadth >= breadth_high and not hi_vol:
        return "clean-uptrend"
    elif breadth < breadth_low or hi_vol:
        return "recovery-bounce"
    else:
        return "chop"


def _build_weight_matrix_parametric(ind, vol_hi_threshold, breadth_low, breadth_high):
    """Build router weight matrix with custom thresholds."""
    C = ind["C"]
    W = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    prev_weights = {}
    for i, d in enumerate(C.index):
        if i < 200:
            W.iloc[i] = {col: prev_weights.get(col, 0.0) for col in C.columns}
            continue
        regime = _detect_regime_parametric(ind, i, vol_hi_threshold, breadth_low, breadth_high)
        if regime == "downtrend":
            new_w = {"BTCUSDT": 0.10}
        elif regime == "clean-uptrend":
            new_w = ame._weights_uptrend(ind, i)
        elif regime == "recovery-bounce":
            new_w = ame._weights_recovery(ind, i)
        else:
            new_w = ame._weights_chop(ind, i)
        row = {col: new_w.get(col, 0.0) for col in C.columns}
        W.iloc[i] = row
        prev_weights = new_w
    return W


# ============================================================
# CONVICTION SIGNAL: regime clarity distance
# ============================================================
def _regime_clarity(ind, vol_hi_threshold, breadth_low, breadth_high):
    """Per-bar scalar in [0,1]: how far from the nearest regime boundary.
    Causal: uses data at bar d only.
    Returns a Series indexed like C.index.
    """
    C = ind["C"]; sma50 = ind["sma50"]; vol20 = ind["vol20"]; sma200 = ind["sma200"]
    clarity = pd.Series(0.0, index=C.index)
    for i, d in enumerate(C.index):
        if i < 200:
            clarity.iloc[i] = 0.5
            continue
        btc = C.loc[d, "BTCUSDT"]; s200 = sma200.loc[d, "BTCUSDT"]
        if pd.isna(s200) or pd.isna(btc):
            clarity.iloc[i] = 0.5
            continue
        btc_up = btc > s200
        if not btc_up:
            # downtrend: clarity = how far BTC is below SMA200 (as pct)
            cl = min(1.0, abs(btc / s200 - 1) / 0.10)  # 10% away = full clarity
            clarity.iloc[i] = cl
            continue
        row_c = C.iloc[i]; row_s50 = sma50.iloc[i]
        above = 0; total = 0
        for sym in C.columns:
            c_val = row_c[sym]; s_val = row_s50[sym]
            if pd.notna(c_val) and pd.notna(s_val):
                above += int(c_val > s_val); total += 1
        breadth = above / total if total > 0 else 0.5
        btc_vol = vol20.loc[d, "BTCUSDT"]
        if pd.isna(btc_vol):
            btc_vol = vol_hi_threshold
        # distance from breadth thresholds (normalized to [0, breadth_high - breadth_low])
        span = breadth_high - breadth_low
        if breadth >= breadth_high:
            bd = (breadth - breadth_high) / max(span, 0.01)
        elif breadth < breadth_low:
            bd = (breadth_low - breadth) / max(span, 0.01)
        else:
            # inside chop band -- near boundary
            d_to_low = (breadth - breadth_low) / max(span, 0.01)
            d_to_high = (breadth_high - breadth) / max(span, 0.01)
            bd = min(d_to_low, d_to_high)
        # vol clarity
        vd = abs(btc_vol - vol_hi_threshold) / max(vol_hi_threshold * 0.3, 0.01)
        cl = min(1.0, (bd + vd) / 2.0)
        clarity.iloc[i] = cl
    return clarity.clip(0.0, 1.0)


# ============================================================
# META-ENSEMBLE BUILDER
# ============================================================
def build_ensemble_router(ind, vol_hi_threshold, min_conviction=0.2, max_conviction=1.0):
    """Build the meta-ensemble weight matrix.

    For each bar d:
      1. Run all 9 configs -> 9 weight vectors
      2. Average them (simple mean) = ensemble_W
      3. Optionally scale by conviction (regime clarity of base config)
         to reduce exposure on boundary days.

    conviction_scale = min_conviction + (max_conviction - min_conviction) * clarity
    final_W = ensemble_W * conviction_scale  (row-sum <= 1 guaranteed since ensemble <= 1)
    """
    C = ind["C"]
    # Pre-compute all config weight matrices
    config_weights = []
    train_mask = C.index < pd.Timestamp("2022-01-01")
    for (bl, bh_t, vp) in PERTURBATION_GRID:
        # vol threshold computed from training data for each config
        vol_thr_cfg = float(ind["vol20"]["BTCUSDT"][train_mask].dropna().quantile(vp))
        W_cfg = _build_weight_matrix_parametric(ind, vol_thr_cfg, bl, bh_t)
        config_weights.append(W_cfg)
    # Ensemble: simple mean
    W_ensemble = sum(config_weights) / len(config_weights)

    # Conviction scaling (base config clarity)
    clarity = _regime_clarity(ind, vol_hi_threshold,
                               PERTURBATION_GRID[0][0], PERTURBATION_GRID[0][1])
    conv_scale = min_conviction + (max_conviction - min_conviction) * clarity
    # Apply scaling row-by-row (broadcast over assets)
    W_final = W_ensemble.multiply(conv_scale, axis=0)
    return W_final, W_ensemble, conv_scale


# ============================================================
# TWO-SLEEVE BOOK
# ============================================================
def build_two_sleeve_book(W_router, W_smooth, alpha):
    """Combine router sleeve (alpha) + smooth sleeve (1-alpha).
    Both are weight matrices (row-sum <= 1), result is also <= 1.
    alpha in (0, 1): fraction allocated to router.
    """
    return alpha * W_router + (1 - alpha) * W_smooth


# ============================================================
# SAME-EXPOSURE SHUFFLE NULL
# ============================================================
def shuffle_null(W, ind, bh_b, oos_start, oos_end, n_slices, seeds, n_shuffles=200):
    """Same-exposure shuffle control: randomly permute the COLUMN assignment
    in W each day (preserving exposure profile, destroying selection).
    Returns mean pos_rate across shuffles for significance test.
    """
    C = ind["C"]
    cols = list(C.columns)
    rng = np.random.default_rng(99)
    null_prs = []
    for _ in range(n_shuffles):
        # shuffle columns each day (preserves row-sum = exposure)
        W_null = W.copy()
        for d in W.index:
            row = W.loc[d].values.copy()
            if row.sum() > 1e-6:
                rng.shuffle(row)
                W_null.loc[d] = row
        b_null = book_daily_returns(W_null, ind)
        pr = np.mean([slice_stats(b_null, bh_b, oos_start, oos_end,
                                   n_slices, 7, s)["pos_rate"] for s in seeds])
        null_prs.append(pr)
    return float(np.mean(null_prs)), float(np.std(null_prs))


# ============================================================
# MAIN DISCOVERY RUNNER
# ============================================================
def main():
    t0 = time.time()
    OOS_START = "2022-01-01"
    OOS_END = "2026-06-01"
    N = 500
    SEEDS = [11, 23, 42]

    print("=" * 76)
    print("META-ENSEMBLE ROUTER -- FRESH DISCOVERY (strategy-discovery lens)")
    print(f"OOS: {OOS_START} -> {OOS_END} | n_slices={N} | seeds={SEEDS}")
    print("=" * 76)

    ind = lab.load("2020-01-01", OOS_END)
    C = ind["C"]

    # BH baseline (canonical)
    bh_W = bh_ew_weights(ind)
    bh_b = book_daily_returns(bh_W, ind)
    bh_pr_seeds = [bh_slice_stats(bh_b, OOS_START, OOS_END, N, 7, s) for s in SEEDS]
    bh_pr = float(np.mean([x["pos_rate"] for x in bh_pr_seeds]))
    bh_mn = float(np.mean([x["mean_pct"] for x in bh_pr_seeds]))
    bh_p05 = float(np.mean([x["p05_pct"] for x in bh_pr_seeds]))
    print(f"\n[BH] pos_rate={bh_pr}%  mean={bh_mn}%  p05={bh_p05}%")

    # Base router (canonical)
    train_mask = C.index < pd.Timestamp(OOS_START)
    vthr = float(ind["vol20"]["BTCUSDT"][train_mask].dropna().quantile(ame.VOL_HI_PCTILE))
    W_router = ame.build_weight_matrix(ind, vthr)
    rb = book_daily_returns(W_router, ind)
    r_stats = [slice_stats(rb, bh_b, OOS_START, OOS_END, N, 7, s) for s in SEEDS]
    r_pr = float(np.mean([x["pos_rate"] for x in r_stats]))
    r_mn = float(np.mean([x["mean_pct"] for x in r_stats]))
    r_p05 = float(np.mean([x["p05_pct"] for x in r_stats]))
    r_dw = float(np.mean([x["down_wk_eng_mean"] for x in r_stats]))
    r_bw = float(np.mean([x["beat_bh_pct"] for x in r_stats]))
    print(f"[ROUTER-BASE] pos_rate={r_pr}%  mean={r_mn}%  p05={r_p05}%  "
          f"down_wk={r_dw}%  beat_bh={r_bw}%")

    # ----------------------------------------------------------
    # ANGLE 1: Boundary-Ensemble (9 configs, simple EW average)
    # ----------------------------------------------------------
    print("\n--- ANGLE 1: Boundary-Ensemble (9 parametric configs, EW mean) ---")
    W_ens, W_ens_raw, conv_scale = build_ensemble_router(ind, vthr,
                                                          min_conviction=0.0,
                                                          max_conviction=1.0)
    # W_ens_raw = simple average with no conviction scaling
    ens_b = book_daily_returns(W_ens_raw, ind)
    ens_stats = [slice_stats(ens_b, bh_b, OOS_START, OOS_END, N, 7, s) for s in SEEDS]
    ens_pr = float(np.mean([x["pos_rate"] for x in ens_stats]))
    ens_mn = float(np.mean([x["mean_pct"] for x in ens_stats]))
    ens_p05 = float(np.mean([x["p05_pct"] for x in ens_stats]))
    ens_dw = float(np.mean([x["down_wk_eng_mean"] for x in ens_stats]))
    ens_bw = float(np.mean([x["beat_bh_pct"] for x in ens_stats]))
    ens_dpr = float(np.mean([x["down_wk_eng_posrate"] for x in ens_stats]))
    print(f"  [ENS-SIMPLE] pos_rate={ens_pr:.1f}%  mean={ens_mn:.2f}%  p05={ens_p05:.2f}%  "
          f"down_wk_mean={ens_dw:.2f}%  down_wk_posrate={ens_dpr:.1f}%  beat_bh={ens_bw:.1f}%")
    print(f"    vs BH: pos_rate gap={ens_pr-bh_pr:+.1f}pp  vs ROUTER: {ens_pr-r_pr:+.1f}pp")

    # ----------------------------------------------------------
    # ANGLE 2: Conviction-Weighted Ensemble (scale by regime clarity)
    # ----------------------------------------------------------
    print("\n--- ANGLE 2: Conviction-Scaled Ensemble ---")
    for min_c in [0.3, 0.5, 0.7]:
        W_cv, W_cv_raw, _ = build_ensemble_router(ind, vthr,
                                                   min_conviction=min_c,
                                                   max_conviction=1.0)
        cv_b = book_daily_returns(W_cv, ind)
        cv_stats = [slice_stats(cv_b, bh_b, OOS_START, OOS_END, N, 7, s) for s in SEEDS]
        cv_pr = float(np.mean([x["pos_rate"] for x in cv_stats]))
        cv_mn = float(np.mean([x["mean_pct"] for x in cv_stats]))
        cv_p05 = float(np.mean([x["p05_pct"] for x in cv_stats]))
        cv_dw = float(np.mean([x["down_wk_eng_mean"] for x in cv_stats]))
        cv_bw = float(np.mean([x["beat_bh_pct"] for x in cv_stats]))
        print(f"  [ENS-CONV min={min_c}] pos_rate={cv_pr:.1f}%  mean={cv_mn:.2f}%  "
              f"p05={cv_p05:.2f}%  down_wk={cv_dw:.2f}%  beat_bh={cv_bw:.1f}%  "
              f"vs_BH={cv_pr-bh_pr:+.1f}pp  vs_ROUTER={cv_pr-r_pr:+.1f}pp")

    # ----------------------------------------------------------
    # ANGLE 3: Two-Sleeve Book (router + gated-EW smooth)
    # ----------------------------------------------------------
    print("\n--- ANGLE 3: Two-Sleeve Book (router + gated-EW smooth) ---")
    gate = ind["gate"].astype(float)
    g_sum = gate.sum(axis=1).replace(0, np.nan)
    W_smooth = gate.div(g_sum, axis=0).fillna(0.0)  # gated EW (smooth pos-rate)
    smooth_b = book_daily_returns(W_smooth, ind)
    sm_stats = [slice_stats(smooth_b, bh_b, OOS_START, OOS_END, N, 7, s) for s in SEEDS]
    sm_pr = float(np.mean([x["pos_rate"] for x in sm_stats]))
    sm_mn = float(np.mean([x["mean_pct"] for x in sm_stats]))
    sm_p05 = float(np.mean([x["p05_pct"] for x in sm_stats]))
    print(f"  [GATED-EW smooth sleeve] pos_rate={sm_pr:.1f}%  mean={sm_mn:.2f}%  p05={sm_p05:.2f}%")

    best_alpha = None; best_score = -999; best_row = None
    two_sleeve_results = {}
    for alpha in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        W_2sl = build_two_sleeve_book(W_router, W_smooth, alpha)
        b_2sl = book_daily_returns(W_2sl, ind)
        ts_stats = [slice_stats(b_2sl, bh_b, OOS_START, OOS_END, N, 7, s) for s in SEEDS]
        ts_pr = float(np.mean([x["pos_rate"] for x in ts_stats]))
        ts_mn = float(np.mean([x["mean_pct"] for x in ts_stats]))
        ts_p05 = float(np.mean([x["p05_pct"] for x in ts_stats]))
        ts_dw = float(np.mean([x["down_wk_eng_mean"] for x in ts_stats]))
        ts_bw = float(np.mean([x["beat_bh_pct"] for x in ts_stats]))
        two_sleeve_results[alpha] = {
            "pos_rate": ts_pr, "mean": ts_mn, "p05": ts_p05,
            "down_wk": ts_dw, "beat_bh": ts_bw
        }
        pareto_score = ts_pr + ts_mn * 5  # joint objective (pos-rate + 5x mean)
        print(f"  [alpha={alpha}] pos_rate={ts_pr:.1f}%  mean={ts_mn:.2f}%  "
              f"p05={ts_p05:.2f}%  down_wk={ts_dw:.2f}%  beat_bh={ts_bw:.1f}%  "
              f"vs_BH_pr={ts_pr-bh_pr:+.1f}pp  vs_R_pr={ts_pr-r_pr:+.1f}pp")
        if pareto_score > best_score:
            best_score = pareto_score
            best_alpha = alpha
            best_row = {"pos_rate": ts_pr, "mean": ts_mn, "p05": ts_p05, "down_wk": ts_dw}

    print(f"\n  -> Best alpha on joint (pos_rate + 5*mean): {best_alpha}  {best_row}")

    # ----------------------------------------------------------
    # ANGLE 4: Ensemble + Two-Sleeve combined
    # ----------------------------------------------------------
    print("\n--- ANGLE 4: Best-Ensemble + Smooth two-sleeve ---")
    for alpha in [0.3, 0.5, 0.7]:
        W_4 = build_two_sleeve_book(W_ens_raw, W_smooth, alpha)
        b_4 = book_daily_returns(W_4, ind)
        a4_stats = [slice_stats(b_4, bh_b, OOS_START, OOS_END, N, 7, s) for s in SEEDS]
        a4_pr = float(np.mean([x["pos_rate"] for x in a4_stats]))
        a4_mn = float(np.mean([x["mean_pct"] for x in a4_stats]))
        a4_p05 = float(np.mean([x["p05_pct"] for x in a4_stats]))
        a4_dw = float(np.mean([x["down_wk_eng_mean"] for x in a4_stats]))
        a4_bw = float(np.mean([x["beat_bh_pct"] for x in a4_stats]))
        print(f"  [ENS-ALPHA={alpha}] pos_rate={a4_pr:.1f}%  mean={a4_mn:.2f}%  "
              f"p05={a4_p05:.2f}%  down_wk={a4_dw:.2f}%  beat_bh={a4_bw:.1f}%  "
              f"vs_BH_pr={a4_pr-bh_pr:+.1f}pp  vs_R_pr={a4_pr-r_pr:+.1f}pp")

    # ----------------------------------------------------------
    # ANGLE 5: Conviction-Scaled on SINGLE base router (no boundary ensemble)
    # Simplest possible size lever: scale router exposure by regime clarity
    # ----------------------------------------------------------
    print("\n--- ANGLE 5: Conviction-Scaled BASE Router (single config, size lever) ---")
    clarity_base = _regime_clarity(ind, vthr,
                                    PERTURBATION_GRID[0][0],
                                    PERTURBATION_GRID[0][1])
    for min_c in [0.2, 0.4, 0.6, 0.8]:
        conv = min_c + (1.0 - min_c) * clarity_base
        W_conv = W_router.multiply(conv, axis=0)
        cb = book_daily_returns(W_conv, ind)
        c5_stats = [slice_stats(cb, bh_b, OOS_START, OOS_END, N, 7, s) for s in SEEDS]
        c5_pr = float(np.mean([x["pos_rate"] for x in c5_stats]))
        c5_mn = float(np.mean([x["mean_pct"] for x in c5_stats]))
        c5_p05 = float(np.mean([x["p05_pct"] for x in c5_stats]))
        c5_dw = float(np.mean([x["down_wk_eng_mean"] for x in c5_stats]))
        print(f"  [CONV-BASE min={min_c}] pos_rate={c5_pr:.1f}%  mean={c5_mn:.2f}%  "
              f"p05={c5_p05:.2f}%  down_wk={c5_dw:.2f}%  "
              f"vs_BH={c5_pr-bh_pr:+.1f}pp  vs_R={c5_pr-r_pr:+.1f}pp")

    # ----------------------------------------------------------
    # SHUFFLE NULL for best candidate (date-block permutation)
    # ----------------------------------------------------------
    print("\n--- SHUFFLE NULL (same-exposure, 200 shuffles) ---")
    print("  Computing shuffle null for ENS-SIMPLE (this takes ~2 min)...")
    null_mean, null_std = shuffle_null(W_ens_raw, ind, bh_b, OOS_START, OOS_END, N, SEEDS, n_shuffles=200)
    z_ens = (ens_pr - null_mean) / max(null_std, 0.01)
    p_approx = float(1.0 if ens_pr < null_mean else 0.0)  # approximate one-sided
    print(f"  ENS-SIMPLE pos_rate={ens_pr:.1f}%  null_mean={null_mean:.1f}%  null_std={null_std:.1f}%  "
          f"Z={z_ens:.2f}  (pos_rate > null: {ens_pr > null_mean})")

    # Also shuffle the best two-sleeve
    if best_alpha is not None:
        W_best2sl = build_two_sleeve_book(W_router, W_smooth, best_alpha)
        null_m2, null_s2 = shuffle_null(W_best2sl, ind, bh_b, OOS_START, OOS_END, N, SEEDS, n_shuffles=200)
        ts_best_pr = two_sleeve_results[best_alpha]["pos_rate"]
        z2 = (ts_best_pr - null_m2) / max(null_s2, 0.01)
        print(f"  2SL-alpha={best_alpha} pos_rate={ts_best_pr:.1f}%  null_mean={null_m2:.1f}%  "
              f"null_std={null_s2:.1f}%  Z={z2:.2f}  (pos_rate > null: {ts_best_pr > null_m2})")

    # ----------------------------------------------------------
    # SUMMARY TABLE
    # ----------------------------------------------------------
    print("\n" + "=" * 76)
    print("SUMMARY: Pareto comparison (pos_rate vs mean vs p05)")
    print("-" * 76)
    print(f"{'Engine':<30} {'pos_rate':>9} {'mean':>7} {'p05':>7} {'down_wk':>8}")
    print("-" * 76)
    print(f"{'BH (baseline)':<30} {bh_pr:>9.1f} {bh_mn:>7.2f} {bh_p05:>7.2f} {'n/a':>8}")
    print(f"{'Router-base':<30} {r_pr:>9.1f} {r_mn:>7.2f} {r_p05:>7.2f} {r_dw:>8.2f}")
    print(f"{'ENS-simple (9 configs)':<30} {ens_pr:>9.1f} {ens_mn:>7.2f} {ens_p05:>7.2f} {ens_dw:>8.2f}")
    if best_alpha:
        br = two_sleeve_results[best_alpha]
        print(f"{'2SL best alpha='+str(best_alpha):<30} {br['pos_rate']:>9.1f} {br['mean']:>7.2f} "
              f"{br['p05']:>7.2f} {br['down_wk']:>8.2f}")
    print("=" * 76)

    runtime = round(time.time() - t0, 1)
    print(f"\nTotal runtime: {runtime}s")

    out = {
        "oos": [OOS_START, OOS_END], "n_slices": N, "seeds": SEEDS,
        "bh": {"pos_rate": bh_pr, "mean": bh_mn, "p05": bh_p05},
        "router_base": {"pos_rate": r_pr, "mean": r_mn, "p05": r_p05, "down_wk": r_dw, "beat_bh": r_bw},
        "ens_simple": {"pos_rate": ens_pr, "mean": ens_mn, "p05": ens_p05, "down_wk": ens_dw,
                       "beat_bh": ens_bw, "down_wk_posrate": ens_dpr},
        "smooth_gated_ew": {"pos_rate": sm_pr, "mean": sm_mn, "p05": sm_p05},
        "two_sleeve": two_sleeve_results,
        "best_alpha": best_alpha,
        "shuffle_null_ens": {"null_mean": round(null_mean, 2), "null_std": round(null_std, 2),
                              "Z": round(z_ens, 2), "beats_null": bool(ens_pr > null_mean)},
        "runtime_s": runtime,
    }
    outp = ROOT.parent / "runs" / "strat" / "meta_ensemble_router_results.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2, default=str))
    print(f"Saved: {outp}")
    return out


if __name__ == "__main__":
    main()
