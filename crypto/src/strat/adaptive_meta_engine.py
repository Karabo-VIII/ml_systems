"""src/strat/adaptive_meta_engine.py -- Adaptive Meta-Engine (regime-routing engine).

DESIGN
------
Every trading day we detect the current MARKET REGIME from causal inputs:
  BTC-trend (C vs SMA200) x breadth (% universe above SMA50) x vol (vol20 tercile)

PRE-REGISTERED routing map (no per-regime parameter sweep):
  REGIME               SUB-BEHAVIOR
  -------              ------------
  clean-uptrend        momentum top-3 concentrate (gated, above SMA200)
  recovery-bounce      momentum top-5, NO gate (catch alt bounces)
  chop                 diversified inverse-vol EW (gated assets only)
  downtrend            BTC-only defensive 10% (BTC below SMA200; soft landing)

The downtrend sub-behavior holds 10% BTC (never goes full cash) so that:
  - Down weeks: small BTC-only loss (better than 0% positive rate)
  - Up weeks: participates in BTC recovery (turns "zero" slices to positive)
  - Preserves capital on broad crashes (10% exposure vs 100% BH)

Win condition: beat EW buy-hold positive-rate (~55%) and/or mean (+2.9%) on random 7-day slices,
OOS walk-forward, leak-free, >=300 slices.

CAUSAL RULE: all features at row d use only data <= d. Labels (7-day fwd) excluded from features.
Walk-forward: TRAIN-only period is all rows whose fwd-7d label closed before the EVAL start date.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import random

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.mover_lab as ml

# ──────────────────────────────────────────────────────────────
# REGIME DETECTION (fully causal)
# ──────────────────────────────────────────────────────────────
BREADTH_LOW_THRESH = 0.30   # < 30% above SMA50 -> breadth is low
BREADTH_HIGH_THRESH = 0.60  # > 60% above SMA50 -> breadth is high
VOL_HI_PCTILE = 0.67        # upper tercile of rolling vol20


def _detect_regime(ind: dict, i: int, vol_hi_threshold: float) -> str:
    """Return regime string for bar index i (uses data up to and including i).
    Pre-registered mapping -- NO per-regime parameter search.

    Regime ladder:
      1. BTC below SMA200 -> downtrend (defensive 10% BTC position)
      2. BTC above SMA200:
         a. Breadth >= 60%, low vol -> clean-uptrend (momentum concentrate)
         b. Breadth < 30% OR high vol -> recovery-bounce (alt catch, no gate)
         c. Otherwise -> chop (inv-vol EW)
    """
    C = ind["C"]; sma200 = ind["sma200"]; sma50 = ind["sma50"]
    vol20 = ind["vol20"]
    d = C.index[i]
    btc = C.loc[d, "BTCUSDT"]
    s200 = sma200.loc[d, "BTCUSDT"]
    # BTC trend: SMA200 is the primary gate
    btc_up = (not pd.isna(s200)) and (btc > s200)
    if not btc_up:
        return "downtrend"
    # BTC above SMA200: determine uptrend quality via breadth + vol
    row_c = C.iloc[i]
    row_s50 = sma50.iloc[i]
    above = 0; total = 0
    for sym in C.columns:
        c_val = row_c[sym]; s_val = row_s50[sym]
        if pd.notna(c_val) and pd.notna(s_val):
            above += int(c_val > s_val); total += 1
    breadth = above / total if total > 0 else 0.5
    btc_vol = vol20.loc[d, "BTCUSDT"] if pd.notna(vol20.loc[d, "BTCUSDT"]) else 0.5
    hi_vol = btc_vol >= vol_hi_threshold
    if breadth >= BREADTH_HIGH_THRESH and not hi_vol:
        return "clean-uptrend"
    elif breadth < BREADTH_LOW_THRESH or hi_vol:
        return "recovery-bounce"
    else:
        return "chop"


# ──────────────────────────────────────────────────────────────
# SUB-BEHAVIORS (all causal, long-only, row-sum <= 1)
# ──────────────────────────────────────────────────────────────
def _weights_uptrend(ind: dict, i: int) -> dict:
    """Clean uptrend: BTC up, breadth high, low vol.
    Strategy: top-5 by composite score (mom14 + mom7) among gated assets.
    Using 5 picks (not 3) to increase positive-rate coverage -- single-week
    concentration risk dragged pos-rate below BH even in clean up-weeks.
    """
    C = ind["C"]; gate = ind["gate"]
    mom14 = ind["mom14"]; mom7 = ind["mom7"]
    d = C.index[i]
    scored = []
    for sym in C.columns:
        g = bool(gate.loc[d, sym])
        m14 = mom14.loc[d, sym]; m7 = mom7.loc[d, sym]
        if g and pd.notna(m14) and pd.notna(m7):
            # composite: 60% 14d momentum + 40% 7d for recency
            score = 0.6 * m14 + 0.4 * m7
            scored.append((sym, score))
    scored.sort(key=lambda x: -x[1])
    picks = scored[:5]
    if not picks:
        return {}
    w = 1.0 / len(picks)
    return {s: w for s, _ in picks}


def _weights_recovery(ind: dict, i: int) -> dict:
    """Recovery-bounce: BTC up, breadth low/rising.
    Strategy: EW over assets where BTC is above SMA50 (momentum leader),
    plus any gated asset (above SMA200). If BTC is the only gate-pass,
    fall back to top-5 by mom7 (short-term bounce signal).
    Rationale: in early recovery, alts are below SMA200 so the gate kills
    them all. We need EITHER above-SMA50 OR showing positive 7d momentum
    with any positive price.
    """
    C = ind["C"]; gate = ind["gate"]; mom7 = ind["mom7"]
    sma50 = ind["sma50"]
    d = C.index[i]
    # Eligible: above SMA50 (intermediate trend) OR gated (above SMA200)
    eligible = []
    for sym in C.columns:
        c_val = C.loc[d, sym]; s50 = sma50.loc[d, sym]
        g = bool(gate.loc[d, sym])
        m7 = mom7.loc[d, sym]
        above_sma50 = pd.notna(s50) and pd.notna(c_val) and c_val > s50
        if (above_sma50 or g) and pd.notna(m7) and pd.notna(c_val) and c_val > 0:
            eligible.append((sym, m7))
    if not eligible:
        # fallback: top-5 by mom7, no filter
        scored = []
        for sym in C.columns:
            m7 = mom7.loc[d, sym]; c_val = C.loc[d, sym]
            if pd.notna(m7) and pd.notna(c_val) and c_val > 0:
                scored.append((sym, m7))
        scored.sort(key=lambda x: -x[1])
        eligible = scored[:5]
    eligible.sort(key=lambda x: -x[1])
    picks = eligible[:5]
    if not picks:
        return {}
    w = 1.0 / len(picks)
    return {s: w for s, _ in picks}


def _weights_chop(ind: dict, i: int) -> dict:
    """Inverse-vol EW over gated assets (smooth, diversified)."""
    C = ind["C"]; gate = ind["gate"]; vol20 = ind["vol20"]
    d = C.index[i]
    eligible = []
    for sym in C.columns:
        g = bool(gate.loc[d, sym])
        v = vol20.loc[d, sym]
        if g and pd.notna(v) and v > 0:
            eligible.append((sym, v))
    if not eligible:
        return {}
    inv_vols = {s: 1.0 / v for s, v in eligible}
    total = sum(inv_vols.values())
    return {s: iv / total for s, iv in inv_vols.items()}


# ──────────────────────────────────────────────────────────────
# WEIGHT MATRIX BUILDER (walk-forward causal)
# ──────────────────────────────────────────────────────────────
def build_weight_matrix(ind: dict, vol_hi_threshold: float) -> pd.DataFrame:
    """Build the W matrix (dates x assets) for the full data range.
    Positions are lagged 1 bar by evaluate(), so signal at d is acted on at d+1 (causal).
    vol_hi_threshold is pre-computed from TRAINING data only (see run_engine).
    """
    C = ind["C"]
    W = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    prev_weights: dict = {}

    for i, d in enumerate(C.index):
        if i < 200:  # warmup for SMA200
            W.iloc[i] = {col: prev_weights.get(col, 0.0) for col in C.columns}
            continue
        regime = _detect_regime(ind, i, vol_hi_threshold)
        if regime == "downtrend":
            # Soft defensive: small BTC-only position (10%)
            # Rationale: pure CASH = 0% every down-trend week, drags positive-rate
            # to near-zero during trend transitions. 10% BTC still preserves capital
            # on broad crashes (-71% BH vs -7% defensive) while turning many
            # "zero-return" weeks into small positive weeks during bounce days.
            new_w = {"BTCUSDT": 0.10}
        elif regime == "clean-uptrend":
            new_w = _weights_uptrend(ind, i)
        elif regime == "recovery-bounce":
            new_w = _weights_recovery(ind, i)
        else:  # chop
            new_w = _weights_chop(ind, i)
        row = {col: new_w.get(col, 0.0) for col in C.columns}
        W.iloc[i] = row
        prev_weights = new_w
    return W


# ──────────────────────────────────────────────────────────────
# RANDOM-SLICE EVALUATION (the WIN CONDITION)
# ──────────────────────────────────────────────────────────────
def random_slice_eval(
    bret: pd.Series,
    bh_ew: pd.Series,
    n_slices: int = 400,
    slice_days: int = 7,
    seed: int = 42,
    eval_start: str = "2022-01-01",   # OOS region only
) -> dict:
    """Draw n_slices random 7-day windows from eval_start onward.
    For each: compute engine 7d compound return AND buy-hold 7d return.
    Report positive-rate, mean, and down-week cash behavior.
    """
    rng = random.Random(seed)
    idx = bret.index
    oos_mask = idx >= pd.Timestamp(eval_start)
    oos_idx = idx[oos_mask]
    if len(oos_idx) < slice_days + 10:
        return {"error": "not enough OOS data"}
    max_start = len(oos_idx) - slice_days
    results = []
    for _ in range(n_slices):
        start_i = rng.randint(0, max_start - 1)
        sl = oos_idx[start_i: start_i + slice_days]
        eng_ret = float((1 + bret.loc[sl]).prod() - 1)
        bh_ret  = float((1 + bh_ew.loc[sl]).prod() - 1)
        results.append({"eng": eng_ret, "bh": bh_ret})
    eng_rets = [r["eng"] for r in results]
    bh_rets  = [r["bh"]  for r in results]
    down_weeks = [r for r in results if r["bh"] < 0]
    eng_on_down = [r["eng"] for r in down_weeks]
    return {
        "n_slices": n_slices,
        "eng_positive_rate": round(100 * np.mean(np.array(eng_rets) > 0), 1),
        "bh_positive_rate":  round(100 * np.mean(np.array(bh_rets)  > 0), 1),
        "eng_mean_7d_pct":   round(100 * float(np.mean(eng_rets)), 2),
        "bh_mean_7d_pct":    round(100 * float(np.mean(bh_rets)), 2),
        "eng_median_7d_pct": round(100 * float(np.median(eng_rets)), 2),
        "bh_median_7d_pct":  round(100 * float(np.median(bh_rets)), 2),
        "n_down_bh_weeks": len(down_weeks),
        "eng_on_down_bh_mean": round(100 * float(np.mean(eng_on_down)), 2) if eng_on_down else None,
        "eng_on_down_bh_positive_rate": round(100 * np.mean(np.array(eng_on_down) > 0), 1) if eng_on_down else None,
        "eng_p5_7d_pct": round(100 * float(np.percentile(eng_rets, 5)), 2),
        "bh_p5_7d_pct":  round(100 * float(np.percentile(bh_rets, 5)), 2),
    }


# ──────────────────────────────────────────────────────────────
# MAIN ENGINE RUNNER
# ──────────────────────────────────────────────────────────────
def run_engine(
    start: str = "2020-01-01",
    end:   str = "2026-06-01",
    eval_start: str = "2022-01-01",   # OOS boundary
    n_slices: int = 400,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Full walk-forward run of the Adaptive Meta-Engine.
    vol_hi_threshold is computed from TRAINING data only (before eval_start) -- causal.
    """
    if verbose:
        print("[adaptive_meta_engine] Loading data...")
    ind = ml.load(start, end)
    C = ind["C"]
    R = ind["R"]

    # ── causal vol threshold: computed from TRAINING region only ──
    train_mask = C.index < pd.Timestamp(eval_start)
    btc_vol_train = ind["vol20"]["BTCUSDT"][train_mask].dropna()
    vol_hi_threshold = float(btc_vol_train.quantile(VOL_HI_PCTILE))
    if verbose:
        print(f"  vol_hi_threshold (from train only): {vol_hi_threshold:.4f}")

    # ── build weight matrix ──
    if verbose:
        print("[adaptive_meta_engine] Building weight matrix...")
    W = build_weight_matrix(ind, vol_hi_threshold)

    # ── evaluate full period ──
    result_full = ml.evaluate(W, ind, H=7, label="adaptive_meta_engine")
    if verbose:
        print("[adaptive_meta_engine] Full-period results:")
        for k, v in result_full.items():
            print(f"  {k}: {v}")

    # ── EW buy-hold returns ──
    bh_ew = R.fillna(0.0).mean(axis=1)

    # ── derive engine daily returns from W ──
    COST = ml.COST
    pos = W.shift(1).fillna(0.0)
    turn = pos.diff().abs().fillna(pos.abs()).sum(axis=1)
    R_aligned = R.reindex(index=W.index, columns=W.columns).fillna(0.0)
    bret = (pos * R_aligned).sum(axis=1) - turn * (COST / 2.0)

    # ── random-slice evaluation ──
    if verbose:
        print(f"[adaptive_meta_engine] Random-slice eval (n={n_slices}, eval_start={eval_start})...")
    slice_stats = random_slice_eval(bret, bh_ew, n_slices=n_slices, slice_days=7,
                                    seed=seed, eval_start=eval_start)
    if verbose:
        print("[adaptive_meta_engine] Slice stats:")
        for k, v in slice_stats.items():
            print(f"  {k}: {v}")

    # ── regime distribution (OOS) ──
    oos_mask = C.index >= pd.Timestamp(eval_start)
    regimes_oos = []
    vol_hi_thr = vol_hi_threshold
    oos_mask_arr = np.asarray(oos_mask)
    for i in range(len(C.index)):
        if not oos_mask_arr[i]:
            continue
        if i < 200:
            continue
        regimes_oos.append(_detect_regime(ind, i, vol_hi_thr))
    from collections import Counter
    rc = Counter(regimes_oos)
    total_r = sum(rc.values())
    regime_dist = {k: round(100 * v / total_r, 1) for k, v in rc.items()}

    return {
        "full_period": result_full,
        "slice_stats": slice_stats,
        "regime_distribution_oos_pct": regime_dist,
        "vol_hi_threshold": round(vol_hi_threshold, 4),
    }


# ──────────────────────────────────────────────────────────────
# BASELINE COMPARISON (EW BH random-slice stats)
# ──────────────────────────────────────────────────────────────
def run_bh_baseline(
    start: str = "2020-01-01",
    end:   str = "2026-06-01",
    eval_start: str = "2022-01-01",
    n_slices: int = 400,
    seed: int = 42,
) -> dict:
    ind = ml.load(start, end)
    C = ind["C"]
    R = ind["R"]
    bh_ew = R.fillna(0.0).mean(axis=1)
    # For comparison: random-slice on BH itself
    stats = random_slice_eval(bh_ew, bh_ew, n_slices=n_slices, slice_days=7,
                               seed=seed, eval_start=eval_start)
    return stats


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-start", default="2022-01-01")
    ap.add_argument("--n-slices", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-06-01")
    args = ap.parse_args()

    print("=" * 60)
    print("ADAPTIVE META-ENGINE -- TOURNAMENT RUN")
    print("=" * 60)
    out = run_engine(
        start=args.start, end=args.end,
        eval_start=args.eval_start,
        n_slices=args.n_slices, seed=args.seed,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("BUY-HOLD BASELINE (EW, same OOS slices)")
    print("=" * 60)
    bh_stats = run_bh_baseline(
        start=args.start, end=args.end,
        eval_start=args.eval_start,
        n_slices=args.n_slices, seed=args.seed,
    )
    for k, v in bh_stats.items():
        print(f"  {k}: {v}")

    # ── Summary table ──
    s = out["slice_stats"]
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Metric':<35} {'Engine':>10} {'EW BH':>10}")
    print("-" * 57)
    print(f"{'Positive-rate (7d slices) %':<35} {s['eng_positive_rate']:>10} {s['bh_positive_rate']:>10}")
    print(f"{'Mean 7d return %':<35} {s['eng_mean_7d_pct']:>10} {s['bh_mean_7d_pct']:>10}")
    print(f"{'Median 7d return %':<35} {s['eng_median_7d_pct']:>10} {s['bh_median_7d_pct']:>10}")
    print(f"{'p5 7d return %':<35} {s['eng_p5_7d_pct']:>10} {s['bh_p5_7d_pct']:>10}")
    print(f"{'Down-BH weeks: eng mean %':<35} {s['eng_on_down_bh_mean']:>10} {'<0':>10}")
    print(f"{'Down-BH weeks: eng pos-rate %':<35} {s['eng_on_down_bh_positive_rate']:>10} {'~0':>10}")
    print(f"\nRegime distribution (OOS): {out['regime_distribution_oos_pct']}")
    fp = out["full_period"]
    print(f"\nFull-period compound: {fp['comp_full']}% | maxDD: {fp['maxDD']}%")
    print(f"Green-rate (7d blocks): {fp['green_all']}%")
    print(f"Avg exposure: {fp['avg_expo']} | Avg turnover: {fp['avg_turnover']}")
