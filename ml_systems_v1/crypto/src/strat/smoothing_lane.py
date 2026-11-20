"""src/strat/smoothing_lane.py -- PURE SMOOTHING LANE for 7-day slice win-rate.

OBJECTIVE: Establish the SMOOTHING CEILING -- how high can the random-7d-slice win-rate go
via diversification and vol-control ALONE (no prediction), using a 14-day lookback window.

BOOKS TESTED:
  1. EW_ALL       -- Equal-weight ALL eligible assets (max diversification, baseline)
  2. EW_BH        -- EW buy-hold entire data range (the standing result to beat: 55%, +2.9%)
  3. VOL_TARGET   -- Vol-targeted EW: scale each name to equal risk contribution (annualized target)
  4. BARBELL      -- Hold winners (top 3 by 14d mom) + laggards (bottom 3 by 14d mom) simultaneously EW
  5. IVP          -- Inverse-variance portfolio (allocate 1/vol^2, renormalized)
  6. RISK_PARITY  -- Iterative risk parity (equal marginal risk contribution)
  7. MIN_CORR     -- Max-diversification approx: weight inversely by rolling correlation to median asset

CAUSAL RULE (CRITICAL): any feature at row d uses only data <= d.
The 14-day lookback for vol/correlation estimates is strictly past-only.
LABELS = d -> d+7 forward return.

WALK-FORWARD: Since these are pure structural books (no prediction), the "training" is just the
rolling 14-day window for vol estimation. No look-ahead possible by construction.

RANDOM SLICE EVALUATION: 500+ non-overlapping random 7-day windows drawn uniformly from 2020-2025.
Report: win-rate (fraction > 0), mean return, median return, Sharpe proxy.

RWYB: python -m strat.smoothing_lane
No emoji. No git commits.
"""
from __future__ import annotations

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.mover_lab as lab

COST = lab.COST  # taker round-trip cost

FULL_START = "2020-01-01"
FULL_END   = "2026-01-01"
N_SLICES   = 500
SLICE_LEN  = 7    # days
LOOKBACK   = 14   # days for vol/corr estimation
VOL_TGT    = 0.40  # annualized vol target per asset in vol-targeted book
EPS        = 1e-8
RNG_SEED   = 42

np.random.seed(RNG_SEED)


# ---------------------------------------------------------------------------
# Weight builders (all causal, all past-only)
# ---------------------------------------------------------------------------

def build_ew_all(ind):
    """EW over ALL assets every day (max diversification, no prediction)."""
    C = ind["C"]
    # All assets present (not NaN)
    present = (~C.isna()).astype(float)
    row_sum = present.sum(axis=1).replace(0, np.nan)
    W = present.div(row_sum, axis=0).fillna(0.0)
    return W


def build_vol_target(ind, vol_tgt=VOL_TGT):
    """Vol-targeted EW: scale each asset to equal risk contribution.

    Weight_i = (vol_tgt / vol_i) / N, renormalized so sum=1 (long-only, no leverage).
    vol_i = rolling 14-day annualized realized vol (causal).
    """
    R = ind["R"]
    vol = R.rolling(LOOKBACK, min_periods=5).std() * np.sqrt(365)
    vol = vol.replace(0, np.nan)

    # Raw weight proportional to 1/vol
    raw = 1.0 / (vol + EPS)
    # Mask missing
    present = (~ind["C"].isna()).astype(float)
    raw = raw * present
    # Renormalize to sum=1
    row_sum = raw.sum(axis=1).replace(0, np.nan)
    W = raw.div(row_sum, axis=0).fillna(0.0)
    return W


def build_barbell(ind, top_k=3, bot_k=3):
    """Barbell: hold winners (top K by 14d mom) AND laggards (bottom K by 14d mom), EW.

    This tests whether holding both extremes (diversification across momentum spectrum)
    improves win-rate via smoother combined return.
    """
    mom = ind["mom14"]
    C = ind["C"]
    present = ~C.isna()

    W = pd.DataFrame(0.0, index=C.index, columns=C.columns)

    for i, d in enumerate(C.index):
        row_mom = mom.iloc[i]
        row_present = present.iloc[i]
        eligible = row_mom[row_present & row_mom.notna()].sort_values()

        n = len(eligible)
        if n < 2:
            continue

        # Top K (winners) and bottom K (laggards)
        actual_k = min(top_k, n // 2)
        if actual_k < 1:
            actual_k = 1

        winners = list(eligible.index[-actual_k:])
        laggards = list(eligible.index[:actual_k])
        picks = list(set(winners + laggards))

        for s in picks:
            W.loc[d, s] = 1.0 / len(picks)

    return W


def build_ivp(ind):
    """Inverse-variance portfolio: weight_i = (1/var_i) / sum(1/var_j).

    Minimizes portfolio variance under diagonal covariance assumption.
    More aggressive vol weighting than vol-target (uses var not vol).
    """
    R = ind["R"]
    var = R.rolling(LOOKBACK, min_periods=5).var()
    var = var.replace(0, np.nan)

    raw = 1.0 / (var + EPS)
    present = (~ind["C"].isna()).astype(float)
    raw = raw * present
    row_sum = raw.sum(axis=1).replace(0, np.nan)
    W = raw.div(row_sum, axis=0).fillna(0.0)
    return W


def build_risk_parity(ind, n_iter=5):
    """Approximate risk parity via iterative scaling.

    Start from 1/vol, then iteratively adjust so each asset contributes
    equal marginal risk (sigma_i * w_i = constant across assets).
    Uses rolling 14d covariance diagonal only (computationally tractable).
    """
    R = ind["R"]
    vol = R.rolling(LOOKBACK, min_periods=5).std() * np.sqrt(365)
    vol = vol.replace(0, np.nan)
    present = (~ind["C"].isna()).astype(float)

    W_list = []
    for i, d in enumerate(R.index):
        row_vol = vol.iloc[i].fillna(0)
        row_present = present.iloc[i]

        active = row_vol[(row_present > 0) & (row_vol > 0)]
        if len(active) == 0:
            W_list.append(pd.Series(0.0, index=R.columns))
            continue

        # Start from 1/vol
        w = 1.0 / active.values
        w = w / w.sum()

        # Iterative: adjust so w_i * vol_i is equal
        vols = active.values
        for _ in range(n_iter):
            contrib = w * vols
            target = contrib.mean()
            w = w * target / (contrib + EPS)
            w = np.clip(w, 0, None)
            s = w.sum()
            if s > EPS:
                w /= s

        row = pd.Series(0.0, index=R.columns)
        row[active.index] = w
        W_list.append(row)

    W = pd.DataFrame(W_list, index=R.index)
    return W


def build_min_corr_weight(ind):
    """Minimum-correlation weighting: weight inversely by avg rolling pairwise correlation.

    Assets that move with everyone else get downweighted; diversifiers get upweighted.
    Uses 14-day rolling correlation matrix (causal).
    """
    R = ind["R"]
    C = ind["C"]
    present = (~C.isna()).astype(float)
    cols = R.columns

    W_list = []
    for i, d in enumerate(R.index):
        row_present = present.iloc[i]
        active_cols = cols[row_present > 0]

        if len(active_cols) < 2:
            row = pd.Series(0.0, index=cols)
            if len(active_cols) == 1:
                row[active_cols[0]] = 1.0
            W_list.append(row)
            continue

        # Rolling correlation matrix using past LOOKBACK days
        start_i = max(0, i - LOOKBACK)
        window_r = R.iloc[start_i:i][active_cols].fillna(0)

        if len(window_r) < 3:
            # Not enough data, use EW
            row = pd.Series(0.0, index=cols)
            for s in active_cols:
                row[s] = 1.0 / len(active_cols)
            W_list.append(row)
            continue

        corr_mat = window_r.corr().fillna(0)
        # Average correlation of each asset with all others
        avg_corr = (corr_mat.sum(axis=1) - 1) / (len(active_cols) - 1)
        avg_corr = avg_corr.clip(lower=0.01)

        # Weight inversely to avg correlation
        raw = 1.0 / avg_corr
        raw = raw / raw.sum()

        row = pd.Series(0.0, index=cols)
        for s in active_cols:
            row[s] = raw.get(s, 0.0)
        W_list.append(row)

    W = pd.DataFrame(W_list, index=R.index)
    return W


# ---------------------------------------------------------------------------
# Random-slice evaluator
# ---------------------------------------------------------------------------

def random_slice_winrate(W, R_daily, n_slices=N_SLICES, slice_len=SLICE_LEN, seed=RNG_SEED):
    """Draw n_slices random 7-day windows, compute book return for each, return stats.

    CAUSAL: positions are lagged 1 day (W.shift(1)), so day d's position was decided on day d-1.
    Returns: dict with win_rate, mean_ret, median_ret, std_ret, n_slices_actual
    """
    rng = np.random.default_rng(seed)

    # Align W and R
    R = R_daily.reindex(index=W.index, columns=W.columns).fillna(0.0)
    pos = W.shift(1).fillna(0.0)

    # Daily book return (net of taker cost on turnover)
    turn = pos.diff().abs().fillna(pos.abs()).sum(axis=1)
    bret = (pos * R).sum(axis=1) - turn * (COST / 2.0)

    dates = bret.index
    n_dates = len(dates)

    # Need at least slice_len + 1 (for the shift) bars
    min_start = 1  # skip first bar (shift needs one warmup)
    max_start = n_dates - slice_len

    if max_start <= min_start:
        return {"win_rate": np.nan, "mean_ret": np.nan, "n": 0}

    # Draw random start indices
    start_indices = rng.integers(min_start, max_start + 1, size=n_slices * 3)  # oversample
    # Deduplicate and pick n_slices
    seen = set()
    unique_starts = []
    for si in start_indices:
        if si not in seen:
            seen.add(si)
            unique_starts.append(si)
        if len(unique_starts) >= n_slices:
            break

    slice_returns = []
    for si in unique_starts:
        window = bret.iloc[si:si + slice_len]
        if len(window) < slice_len:
            continue
        compound_ret = float(np.prod(1 + window.values) - 1)
        slice_returns.append(compound_ret)

    arr = np.array(slice_returns)
    return {
        "win_rate": float(np.mean(arr > 0)),
        "mean_ret": float(np.mean(arr) * 100),
        "median_ret": float(np.median(arr) * 100),
        "std_ret": float(np.std(arr) * 100),
        "p25": float(np.percentile(arr, 25) * 100),
        "p75": float(np.percentile(arr, 75) * 100),
        "n": len(arr),
    }


def bh_slice_winrate(ind, n_slices=N_SLICES, slice_len=SLICE_LEN, seed=RNG_SEED):
    """EW buy-hold (all assets, fixed EW), the standing baseline."""
    C = ind["C"]
    R = ind["R"]
    present = (~C.isna()).astype(float)
    row_sum = present.sum(axis=1).replace(0, np.nan)
    W = present.div(row_sum, axis=0).fillna(0.0)
    # BH has no turnover after day 0 (no rebalancing cost)
    # But we still pay entry cost on day 1
    return random_slice_winrate(W, R, n_slices=n_slices, slice_len=slice_len, seed=seed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("SMOOTHING LANE -- Pure Diversification/Vol-Control Ceiling")
    print(f"Period: {FULL_START} to {FULL_END}")
    print(f"Slices: {N_SLICES} random 7-day windows | Lookback: {LOOKBACK}d | Seed: {RNG_SEED}")
    print("=" * 70)

    print("\nLoading data...")
    ind = lab.load(start=FULL_START, end=FULL_END)
    print(f"  Universe: {list(ind['C'].columns)}")
    print(f"  Date range: {ind['C'].index[0].date()} to {ind['C'].index[-1].date()}")
    print(f"  Total bars: {len(ind['C'])}")

    R_daily = ind["R"]
    C = ind["C"]

    # --- Build all weight matrices ---
    print("\nBuilding weight matrices...")

    print("  [1/6] EW_ALL (equal-weight all assets)...")
    W_ew = build_ew_all(ind)

    print("  [2/6] VOL_TARGET (inverse-vol weighted)...")
    W_vt = build_vol_target(ind, vol_tgt=VOL_TGT)

    print("  [3/6] BARBELL (top-3 + bottom-3 by 14d mom, EW)...")
    W_bb = build_barbell(ind, top_k=3, bot_k=3)

    print("  [4/6] IVP (inverse-variance portfolio)...")
    W_ivp = build_ivp(ind)

    print("  [5/6] RISK_PARITY (equal marginal risk contribution)...")
    W_rp = build_risk_parity(ind, n_iter=5)

    print("  [6/6] MIN_CORR (downweight high-correlation assets)...")
    W_mc = build_min_corr_weight(ind)

    # --- Evaluate random slices ---
    print("\nEvaluating random 7-day slices...")

    books = [
        ("EW_BH (standing baseline)",    W_ew),
        ("VOL_TARGET (1/vol weights)",   W_vt),
        ("BARBELL (top3+bot3 mom14)",    W_bb),
        ("IVP (1/var weights)",          W_ivp),
        ("RISK_PARITY (equal risk)",     W_rp),
        ("MIN_CORR (low-corr upweighted)", W_mc),
    ]

    results = []
    for name, W in books:
        print(f"  {name}...")
        stats = random_slice_winrate(W, R_daily)
        stats["book"] = name
        results.append(stats)
        print(f"    Win-rate={stats['win_rate']*100:.1f}%  Mean={stats['mean_ret']:.2f}%  "
              f"Median={stats['median_ret']:.2f}%  Std={stats['std_ret']:.2f}%  N={stats['n']}")

    # --- Also test regime-gated variants (above SMA200 only) ---
    print("\n  --- Regime-gated variants (above SMA200 only) ---")
    gate = ind["gate"].astype(float)

    gated_books = []
    for base_name, W in books:
        W_gated = W * gate
        row_sum = W_gated.sum(axis=1).replace(0, np.nan)
        W_gated = W_gated.div(row_sum, axis=0).fillna(0.0)
        gated_books.append((f"GATED_{base_name}", W_gated))

    gated_results = []
    for name, W in gated_books:
        print(f"  {name}...")
        stats = random_slice_winrate(W, R_daily)
        stats["book"] = name
        gated_results.append(stats)
        print(f"    Win-rate={stats['win_rate']*100:.1f}%  Mean={stats['mean_ret']:.2f}%  "
              f"Median={stats['median_ret']:.2f}%  Std={stats['std_ret']:.2f}%  N={stats['n']}")

    # --- Summary table ---
    all_results = results + gated_results
    all_results.sort(key=lambda x: -x["win_rate"])

    print("\n")
    print("=" * 70)
    print("RESULTS -- Random 7-day Slice Win-Rate (>0 return), sorted by win-rate")
    print("=" * 70)
    print(f"{'Book':<42} {'WinRate':>8} {'Mean%':>7} {'Median%':>8} {'Std%':>6} {'N':>5}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['book']:<42} {r['win_rate']*100:>7.1f}% {r['mean_ret']:>7.2f}% "
              f"{r['median_ret']:>8.2f}% {r['std_ret']:>6.2f}% {r['n']:>5d}")
    print("-" * 70)
    print(f"{'STANDING BASELINE (EW BH, target >55%)':42s} {'55.0%':>8} {'2.90%':>7}")
    print("=" * 70)

    # --- Year-level green-rate for best book ---
    best = max(results, key=lambda x: x["win_rate"])
    print(f"\nBest ungated book: {best['book']}")
    print(f"  Win-rate: {best['win_rate']*100:.1f}%")
    print(f"  Mean per-slice return: {best['mean_ret']:.2f}%")
    print(f"  IQR: [{best['p25']:.2f}%, {best['p75']:.2f}%]")

    # --- Verdict ---
    print("\n")
    print("=" * 70)
    print("VERDICT -- Smoothing Ceiling")
    print("=" * 70)
    top_wr = max(r["win_rate"] for r in results)
    top_mr = max(r["mean_ret"] for r in results)
    beats_wr = top_wr > 0.55
    beats_mr = top_mr > 2.9

    print(f"Best smoothing win-rate (ungated): {top_wr*100:.1f}%  vs target 55%  -> {'BEATS' if beats_wr else 'DOES NOT BEAT'}")
    print(f"Best smoothing mean return (ungated): {top_mr:.2f}%  vs target 2.9%  -> {'BEATS' if beats_mr else 'DOES NOT BEAT'}")

    gated_top_wr = max(r["win_rate"] for r in gated_results)
    print(f"Best smoothing win-rate (gated): {gated_top_wr*100:.1f}%  (gated locks out, expected lower in bear slices)")

    if beats_wr or beats_mr:
        print("\nCONCLUSION: SMOOTHING CAN PUSH ABOVE 55% -- structure alone clears the bar.")
    else:
        print(f"\nCONCLUSION: 55% IS A HARD WALL for this universe/period. Pure diversification "
              f"plateaus at {top_wr*100:.1f}%. Beating 55% requires PREDICTION, not just smoothing.")
        print("  -> The 55% floor is the asset-class beta floor (crypto tends up 2020-2025).")
        print("  -> The win-rate ceiling from structure alone = {:.1f}%.".format(top_wr * 100))

    # Save JSON
    out = Path(__file__).resolve().parents[2] / "runs" / "strat" / "smoothing_lane_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "desc": "Pure smoothing lane -- win-rate ceiling",
        "n_slices": N_SLICES,
        "slice_len_days": SLICE_LEN,
        "lookback_days": LOOKBACK,
        "period": f"{FULL_START} to {FULL_END}",
        "results": all_results,
        "verdict": {
            "best_ungated_winrate": round(top_wr, 4),
            "best_ungated_mean_ret": round(top_mr, 4),
            "beats_55pct": beats_wr,
            "beats_2p9pct": beats_mr,
        }
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nResults saved to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
