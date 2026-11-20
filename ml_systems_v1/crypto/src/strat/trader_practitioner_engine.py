"""src/strat/trader_practitioner_engine.py -- PRACTITIONER LENS: within-week dynamic management.

EXPERT ANGLE (fresh, C1+C2 missed it):
C1 = price cross-sectional direction forecasting (DEAD).
C2 = chimera exogenous direction forecasting (DEAD).
C3 TRADER ANGLE = WITHIN-POSITION DYNAMIC MANAGEMENT.

The question is NOT "which assets will go up this week?"
The question is "given the router's existing position, how do I MANAGE it intra-week
to extract more positive-rate outcomes?"

PRACTITIONER THESIS:
- A real trader who is long 58% (router average) will NOT sit through a -7% week if it can be avoided.
- If the week opens with 2 down days, the week is more likely to close negative (momentum persists short-term).
- If the week is already +3% by day 3, partial profit capture locks a positive outcome even if days 4-7 give back.

This is structurally different from direction forecasting: we are NOT predicting the week at t=0.
We are ADJUSTING based on within-week REALIZED P&L, which is causal (we know day t's return before
making day t+1's decision).

TWO LEVERS TESTED (both strictly causal, long-only, daily bar mechanic):

LEVER A: TRAILING WEEK-STOP.
  Monitor running 5-bar P&L from any week start.
  If running P&L <= -STOP_THR (e.g. -3%), scale exposure to FLOOR for remaining days of week.
  Effect: convert deep-red weeks into small-red weeks -> pos-rate-neutral but tail-improvement.
  Implementation: track a "week start" equity level, compute running return vs start, apply scale.

LEVER B: PARTIAL PROFIT LOCK.
  If running week P&L >= TAKE_THR (e.g. +2.5%), scale down to LOCK_SCALE (e.g. 50% of normal).
  Effect: lock green outcomes more reliably -> pos-rate lift.
  Implementation: once the threshold is crossed, stay at LOCK_SCALE for the rest of the week.

LEVER C: ENTRY DELAY (momentum recency).
  On Mon/Tue if the daily return is negative (opening down), delay rebalance by 1 bar.
  A trader's "don't buy the gap down, wait for a bounce" rule.
  Effect: small improvement in entry price on re-entries; prevents buying into short-term sell-offs.

LEVER D: BARBELL FLOOR.
  Always maintain a small BTC floor (e.g. 5%) even in downtrend regime.
  The router goes to 10% BTC in downtrend already; this tests whether RAISING the floor helps pos-rate.
  Effect: during downtrend bounce weeks, 10% -> 15-20% BTC catches more bounces.

COMBINATION: A+B stacked on the ROUTER (not on BH, the router is already the winner).

AUDIT PROTOCOL:
- All levers applied POST the router weight matrix (no re-fit, no lookforward).
- Causal: adjustment at day d uses only returns through day d-1 (the lag-1 mechanic handles this).
- Actually: week-stop needs the WITHIN-WEEK P&L. Since positions are acted at d+1 and we see R at d+1,
  the stop fires at d+1 using: running_pnl = prod(1 + bret[week_start:d+1]) - 1 (causal at d+1).
- Scale = multiplicative on the W row: W_adjusted[d+1] = W[d+1] * scale_factor.
- Referee re-derives via book_daily_returns (same canonical mechanic).
- Date-block permutation test on the ADJUSTMENT DELTA to verify the management overlay is real.

LEAK CHECKLIST:
  [x] No lookahead (adjustment uses only past-intra-week returns from positions already held).
  [x] Thresholds set from TRAINING data only (pre-2022), NOT swept OOS.
  [x] The router W is prebuilt with its own causal convention (pos lagged 1 bar).
  [x] Adjustment is a multiplier on W (doesn't change the asset selection, only size).
  [x] Cost applies to the SIZE CHANGES (turnover on reduced weight = real cost).

No emoji (cp1252). Does NOT git commit.
"""
from __future__ import annotations
import sys
import json
import time
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
# TRAINING-DATA THRESHOLD CALIBRATION
# ============================================================
def calibrate_thresholds(ind: dict, W_router: pd.DataFrame, train_end: str) -> dict:
    """Derive STOP and TAKE thresholds from TRAINING data only (before train_end).
    We want thresholds that capture the meaningful tails without over-triggering.
    Approach: look at the distribution of 5-day rolling P&L on the router in training.
    STOP = 20th percentile of 5d router returns (the bad weeks we'd want to exit).
    TAKE = 70th percentile of 5d router returns (the weeks where profit already in hand).
    These are soft targets, NOT optimised for OOS (no sweep).
    """
    bret = book_daily_returns(W_router, ind)
    mask = bret.index < pd.Timestamp(train_end)
    train_bret = bret[mask]

    # 5-bar rolling compound returns (non-overlapping proxy)
    vals = []
    arr = train_bret.values
    for i in range(0, len(arr) - 5, 5):
        r5 = float(np.prod(1 + arr[i:i + 5]) - 1)
        vals.append(r5)
    vals = np.array(vals)

    p20 = float(np.percentile(vals, 20))
    p70 = float(np.percentile(vals, 70))

    # Clip to practical ranges (don't set insane thresholds)
    stop_thr = max(-0.06, min(-0.01, p20))   # between -1% and -6%
    take_thr = max(0.01, min(0.05, p70))      # between +1% and +5%

    return {
        "stop_thr": round(stop_thr, 4),
        "take_thr": round(take_thr, 4),
        "raw_p20": round(float(p20), 4),
        "raw_p70": round(float(p70), 4),
        "n_blocks": len(vals),
    }


# ============================================================
# LEVER A: TRAILING WEEK-STOP (scale down on running loss)
# ============================================================
def apply_week_stop(W_router: pd.DataFrame, ind: dict,
                    stop_thr: float = -0.03,
                    floor_scale: float = 0.15) -> pd.DataFrame:
    """Apply trailing week-stop to the router weight matrix.

    Logic (fully causal):
    - Track a "week block" (Mon-Fri or just every 5 bars from OOS start).
    - Within the week, monitor running compound P&L from week-start position.
    - If running P&L (from HELD positions) <= stop_thr, scale W down to floor_scale
      for ALL remaining bars of the week.
    - Week resets at the start of each new Mon/5-bar block.

    Note: 'running P&L from held positions' = using the ACTUAL returns on ALREADY-HELD positions
    (W[d-1] acted at d), which is causal.
    """
    C = ind["C"]
    R = ind["R"].reindex(index=W_router.index, columns=W_router.columns).fillna(0.0)
    W_adj = W_router.copy()

    dates = list(C.index)
    # Group bars into Monday-anchored weeks
    dtidx = pd.DatetimeIndex(dates)

    # Use ISO week for natural Mon-Sun grouping
    week_ids = dtidx.isocalendar().week.values.astype(int) * 10000 + dtidx.year.values

    stopped = False
    week_running_ret = 1.0
    cur_week = None

    for i, d in enumerate(dates):
        wk = week_ids[i]
        if wk != cur_week:
            # New week: reset
            cur_week = wk
            week_running_ret = 1.0
            stopped = False

        if stopped:
            W_adj.iloc[i] = W_router.iloc[i] * floor_scale
            # Still accumulate P&L at the reduced scale (but for stop purposes,
            # we track the full portfolio daily return as-executed)
            # At floor_scale, the daily return from this row is approximately:
            # position = W_adj[i-1] (shifted), return = W_adj[i-1] * R[i]
            # We approximate: just track the floor-scale version
        else:
            # Compute today's contribution from yesterday's position (already held)
            if i > 0:
                pos_yesterday = W_adj.iloc[i - 1]  # the position entering today
                day_ret = float((pos_yesterday * R.iloc[i]).sum())
                week_running_ret *= (1 + day_ret)
                running_pnl = week_running_ret - 1.0
                if running_pnl <= stop_thr:
                    stopped = True
                    W_adj.iloc[i] = W_router.iloc[i] * floor_scale

    return W_adj


# ============================================================
# LEVER B: PARTIAL PROFIT LOCK (scale down on running gain)
# ============================================================
def apply_profit_lock(W_router: pd.DataFrame, ind: dict,
                      take_thr: float = 0.025,
                      lock_scale: float = 0.50) -> pd.DataFrame:
    """Apply partial profit-lock to the router weight matrix.

    Logic (fully causal):
    - Track running week P&L.
    - If running P&L (from HELD positions) >= take_thr, scale W down to lock_scale.
    - Once locked, stays at lock_scale for the rest of the week.
    - Resets each Monday/5-bar block.

    The positive-rate effect: if a week is +2.5% by day 3, locking to 50% means
    days 4-5 can only halve the gain at most -> more weeks close positive.
    """
    C = ind["C"]
    R = ind["R"].reindex(index=W_router.index, columns=W_router.columns).fillna(0.0)
    W_adj = W_router.copy()

    dates = list(C.index)
    dtidx = pd.DatetimeIndex(dates)
    week_ids = dtidx.isocalendar().week.values.astype(int) * 10000 + dtidx.year.values

    locked = False
    week_running_ret = 1.0
    cur_week = None

    for i, d in enumerate(dates):
        wk = week_ids[i]
        if wk != cur_week:
            cur_week = wk
            week_running_ret = 1.0
            locked = False

        if locked:
            W_adj.iloc[i] = W_router.iloc[i] * lock_scale
        else:
            if i > 0:
                pos_yesterday = W_adj.iloc[i - 1]
                day_ret = float((pos_yesterday * R.iloc[i]).sum())
                week_running_ret *= (1 + day_ret)
                running_pnl = week_running_ret - 1.0
                if running_pnl >= take_thr:
                    locked = True
                    W_adj.iloc[i] = W_router.iloc[i] * lock_scale

    return W_adj


# ============================================================
# LEVER A+B COMBINED
# ============================================================
def apply_combined(W_router: pd.DataFrame, ind: dict,
                   stop_thr: float = -0.03,
                   floor_scale: float = 0.15,
                   take_thr: float = 0.025,
                   lock_scale: float = 0.50) -> pd.DataFrame:
    """Apply both stop and profit-lock simultaneously.
    Priority: if BOTH fire (unusual), stop takes precedence (floor < lock).
    """
    C = ind["C"]
    R = ind["R"].reindex(index=W_router.index, columns=W_router.columns).fillna(0.0)
    W_adj = W_router.copy()

    dates = list(C.index)
    dtidx = pd.DatetimeIndex(dates)
    week_ids = dtidx.isocalendar().week.values.astype(int) * 10000 + dtidx.year.values

    stopped = False
    locked = False
    week_running_ret = 1.0
    cur_week = None

    for i, d in enumerate(dates):
        wk = week_ids[i]
        if wk != cur_week:
            cur_week = wk
            week_running_ret = 1.0
            stopped = False
            locked = False

        scale = 1.0
        if stopped:
            scale = floor_scale
        elif locked:
            scale = lock_scale

        W_adj.iloc[i] = W_router.iloc[i] * scale

        if not stopped and i > 0:
            pos_yesterday = W_adj.iloc[i - 1]
            day_ret = float((pos_yesterday * R.iloc[i]).sum())
            week_running_ret *= (1 + day_ret)
            running_pnl = week_running_ret - 1.0
            if running_pnl <= stop_thr:
                stopped = True
            elif running_pnl >= take_thr and not locked:
                locked = True

    return W_adj


# ============================================================
# LEVER D: BTC FLOOR BOOST in downtrend
# ============================================================
def apply_btc_floor_boost(W_router: pd.DataFrame, ind: dict,
                           btc_floor: float = 0.20) -> pd.DataFrame:
    """In downtrend regime, boost BTC floor from 10% to btc_floor.
    Rationale: the 10% BTC floor is already in the router for downtrends,
    but bounce weeks in downtrend (BTC up even when below SMA200 short-term)
    can be partially captured with a higher floor.
    Note: this HURTS in continued downtrend (bigger loss on BTC down days),
    so the threshold must be calibrated from training.
    """
    C = ind["C"]
    sma200 = ind["sma200"]
    W_adj = W_router.copy()

    for i, d in enumerate(C.index):
        btc = C.loc[d, "BTCUSDT"]
        s200 = sma200.loc[d, "BTCUSDT"]
        in_downtrend = (not pd.isna(s200)) and (btc <= s200)
        if in_downtrend:
            # Boost BTC weight to btc_floor (router already has 0.10)
            row = W_router.iloc[i].copy()
            if row.sum() < 0.15:  # only modify clearly low-exposure rows
                row["BTCUSDT"] = btc_floor
                W_adj.iloc[i] = row

    return W_adj


# ============================================================
# PERMUTATION TEST: is the management overlay adding real value?
# ============================================================
def permutation_test_overlay(bret_base: pd.Series, bret_managed: pd.Series,
                              oos_start: str, oos_end: str,
                              n_perm: int = 1000, seed: int = 42) -> dict:
    """Test whether the managed-vs-base daily return DELTA is significant.
    We test: mean(delta) > 0 OOS.
    Null: permute the delta (date-block permutation, block=21 days = 1 month).
    This is conservative: if the overlay just times volatility, it will not survive
    block permutation that preserves autocorrelation structure.
    """
    delta = (bret_managed - bret_base)
    mask = (delta.index >= pd.Timestamp(oos_start)) & (delta.index < pd.Timestamp(oos_end))
    delta_oos = delta[mask].values
    observed = float(delta_oos.mean())

    rng = np.random.default_rng(seed)
    n = len(delta_oos)
    block = 21  # 1 month
    null_means = []
    for _ in range(n_perm):
        # Block permutation: shuffle blocks
        n_blocks = n // block
        perm = np.concatenate([
            delta_oos[rng.integers(0, n_blocks) * block:
                      rng.integers(0, n_blocks) * block + block]
            for _ in range(n_blocks)
        ])[:n]
        null_means.append(float(perm.mean()))
    null_means = np.array(null_means)
    p_one_sided = float((null_means >= observed).mean())

    return {
        "observed_mean_delta_bps": round(observed * 10000, 2),
        "null_mean": round(float(null_means.mean()) * 10000, 2),
        "p_one_sided": round(p_one_sided, 4),
        "n_oos": int(n),
        "significant_p05": bool(p_one_sided < 0.05),
    }


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    OOS_START = "2022-01-01"
    OOS_END = "2026-06-01"
    TRAIN_END = OOS_START
    N = 500
    SEEDS = [11, 23, 42]

    print("=" * 76)
    print("PRACTITIONER ENGINE -- within-week dynamic management overlay")
    print("EXPERT ANGLE: POSITION MANAGEMENT, not direction forecasting")
    print(f"OOS: {OOS_START} -> {OOS_END} | N={N} slices | seeds={SEEDS}")
    print("=" * 76)

    print("\n[DATA] Loading...")
    ind = lab.load("2020-01-01", OOS_END)
    C = ind["C"]

    # --- Build canonical router (baseline) ---
    print("[ROUTER] Building baseline weight matrix (causal, training vol-threshold)...")
    train_mask = C.index < pd.Timestamp(OOS_START)
    vthr = float(ind["vol20"]["BTCUSDT"][train_mask].dropna().quantile(ame.VOL_HI_PCTILE))
    W_router = ame.build_weight_matrix(ind, vthr)

    bh_W = bh_ew_weights(ind)
    bh_b = book_daily_returns(bh_W, ind)
    router_b = book_daily_returns(W_router, ind)

    print("[BASELINE] Router vs BH:")
    for s in SEEDS:
        rs = slice_stats(router_b, bh_b, OOS_START, OOS_END, N, 7, s)
        bs = bh_slice_stats(bh_b, OOS_START, OOS_END, N, 7, s)
        print(f"  seed={s}: router pos_rate={rs['pos_rate']}% mean={rs['mean_pct']}% "
              f"p05={rs['p05_pct']}% | BH pos_rate={bs['pos_rate']}% mean={bs['mean_pct']}%")

    # --- Calibrate thresholds from TRAINING data only ---
    print("\n[CALIBRATE] Deriving thresholds from TRAINING data only (pre-2022)...")
    thresh = calibrate_thresholds(ind, W_router, TRAIN_END)
    print(f"  stop_thr={thresh['stop_thr']} take_thr={thresh['take_thr']}")
    print(f"  raw_p20={thresh['raw_p20']} raw_p70={thresh['raw_p70']} n_blocks={thresh['n_blocks']}")
    STOP = thresh["stop_thr"]
    TAKE = thresh["take_thr"]

    results = {
        "oos": [OOS_START, OOS_END],
        "n_slices": N,
        "seeds": SEEDS,
        "calibration": thresh,
        "strategies": {},
    }

    def run_strategy(name, W, description):
        b = book_daily_returns(W, ind)
        stats_by_seed = [slice_stats(b, bh_b, OOS_START, OOS_END, N, 7, s) for s in SEEDS]
        pr = [x["pos_rate"] for x in stats_by_seed]
        mn = [x["mean_pct"] for x in stats_by_seed]
        p05 = [x["p05_pct"] for x in stats_by_seed]
        bw = [x["beat_bh_pct"] for x in stats_by_seed]
        dw = [x["down_wk_eng_mean"] for x in stats_by_seed]
        dpr = [x["down_wk_eng_posrate"] for x in stats_by_seed]
        expo = float((W.sum(axis=1) > 0).loc[C.index >= OOS_START].mean())

        # Delta permutation test vs router
        perm = permutation_test_overlay(router_b, b, OOS_START, OOS_END)

        out = {
            "description": description,
            "pos_rate": round(float(np.mean(pr)), 1),
            "pos_rate_seeds": pr,
            "mean_pct": round(float(np.mean(mn)), 2),
            "p05_pct": round(float(np.mean(p05)), 2),
            "beat_bh": round(float(np.mean(bw)), 1),
            "down_wk_mean": round(float(np.mean([x for x in dw if x is not None])), 2) if any(x is not None for x in dw) else None,
            "down_wk_posrate": round(float(np.mean([x for x in dpr if x is not None])), 1) if any(x is not None for x in dpr) else None,
            "avg_expo": round(expo, 2),
            "perm_test_vs_router": perm,
        }
        print(f"\n[{name}] {description}")
        print(f"  pos_rate={out['pos_rate']}% (seeds {pr})")
        print(f"  mean={out['mean_pct']}% | p05={out['p05_pct']}% | beat_bh={out['beat_bh']}%")
        print(f"  down_wk_mean={out['down_wk_mean']}% | down_wk_posrate={out['down_wk_posrate']}%")
        print(f"  avg_expo={out['avg_expo']}")
        print(f"  delta vs router: mean={perm['observed_mean_delta_bps']}bps p={perm['p_one_sided']} sig={perm['significant_p05']}")
        results["strategies"][name] = out
        return b

    # --- LEVER A: Week-stop ---
    print("\n" + "=" * 60)
    print("LEVER A: TRAILING WEEK-STOP")
    W_stop = apply_week_stop(W_router, ind, stop_thr=STOP, floor_scale=0.15)
    run_strategy("lever_A_stop", W_stop,
                 f"Router + week-stop (stop={STOP:.3f} -> 15% floor)")

    # --- LEVER B: Profit lock ---
    print("\n" + "=" * 60)
    print("LEVER B: PARTIAL PROFIT LOCK")
    W_lock = apply_profit_lock(W_router, ind, take_thr=TAKE, lock_scale=0.50)
    run_strategy("lever_B_lock", W_lock,
                 f"Router + profit-lock (take={TAKE:.3f} -> 50% scale)")

    # --- LEVER A+B Combined ---
    print("\n" + "=" * 60)
    print("LEVER A+B: COMBINED STOP + LOCK")
    W_combo = apply_combined(W_router, ind, stop_thr=STOP, floor_scale=0.15,
                             take_thr=TAKE, lock_scale=0.50)
    run_strategy("lever_AB_combined", W_combo,
                 f"Router + stop({STOP:.3f}->15%) + lock({TAKE:.3f}->50%)")

    # --- LEVER D: BTC floor boost ---
    print("\n" + "=" * 60)
    print("LEVER D: BTC FLOOR BOOST in downtrend")
    W_btc = apply_btc_floor_boost(W_router, ind, btc_floor=0.20)
    run_strategy("lever_D_btc_floor", W_btc,
                 "Router + BTC floor 20% in downtrend (vs 10% default)")

    # --- COMBO A+B+D ---
    print("\n" + "=" * 60)
    print("LEVER A+B+D: FULL PRACTITIONER STACK")
    # Apply D first (changes the base), then A+B on top
    W_d = apply_btc_floor_boost(W_router, ind, btc_floor=0.20)
    W_full = apply_combined(W_d, ind, stop_thr=STOP, floor_scale=0.15,
                            take_thr=TAKE, lock_scale=0.50)
    run_strategy("lever_ABD_full_stack", W_full,
                 "Router + BTC-floor-20% + stop + lock (full practitioner)")

    # --- Robustness: try tighter/looser thresholds (pre-registered sensitivity, not sweep) ---
    print("\n" + "=" * 60)
    print("SENSITIVITY: tighter stop / looser take (robustness check, NOT a sweep)")
    # Tighter stop (half the training-derived threshold)
    stop_tight = max(-0.04, STOP * 0.5)
    take_loose = min(0.06, TAKE * 1.5)
    W_sens = apply_combined(W_router, ind, stop_thr=stop_tight, floor_scale=0.15,
                            take_thr=take_loose, lock_scale=0.50)
    run_strategy("sensitivity_tight_stop",  W_sens,
                 f"Sensitivity: stop={stop_tight:.3f} take={take_loose:.3f}")

    # Looser stop (1.5x the training-derived threshold)
    stop_loose = min(-0.01, STOP * 1.5)
    take_tight = max(0.01, TAKE * 0.7)
    W_sens2 = apply_combined(W_router, ind, stop_thr=stop_loose, floor_scale=0.15,
                             take_thr=take_tight, lock_scale=0.50)
    run_strategy("sensitivity_loose_stop", W_sens2,
                 f"Sensitivity: stop={stop_loose:.3f} take={take_tight:.3f}")

    # --- Final comparison table ---
    print("\n" + "=" * 76)
    print("SUMMARY TABLE (OOS 2022-2026, N=500 slices per seed)")
    print(f"{'Strategy':<28} {'pos_rate':>9} {'mean':>7} {'p05':>8} {'beat_bh':>8} {'down_wk':>8} {'expo':>6} {'delta_sig':>10}")
    print("-" * 88)
    # BH baseline
    bh_stats = [bh_slice_stats(bh_b, OOS_START, OOS_END, N, 7, s) for s in SEEDS]
    bh_pr = round(float(np.mean([x["pos_rate"] for x in bh_stats])), 1)
    bh_mn = round(float(np.mean([x["mean_pct"] for x in bh_stats])), 2)
    bh_p05 = round(float(np.mean([x["p05_pct"] for x in bh_stats])), 2)
    print(f"{'EW BH (baseline)':<28} {bh_pr:>9} {bh_mn:>7} {bh_p05:>8} {'N/A':>8} {'N/A':>8} {'1.00':>6} {'N/A':>10}")
    # Router
    rs_all = [slice_stats(router_b, bh_b, OOS_START, OOS_END, N, 7, s) for s in SEEDS]
    rpr = round(float(np.mean([x["pos_rate"] for x in rs_all])), 1)
    rmn = round(float(np.mean([x["mean_pct"] for x in rs_all])), 2)
    rp05 = round(float(np.mean([x["p05_pct"] for x in rs_all])), 2)
    rbw = round(float(np.mean([x["beat_bh_pct"] for x in rs_all])), 1)
    rdw = round(float(np.mean([x["down_wk_eng_mean"] for x in rs_all if x["down_wk_eng_mean"] is not None])), 2)
    print(f"{'Router (validated)':<28} {rpr:>9} {rmn:>7} {rp05:>8} {rbw:>8} {rdw:>8} {'0.58':>6} {'p=0.000':>10}")
    for name, out in results["strategies"].items():
        dw_str = f"{out['down_wk_mean']:.2f}" if out["down_wk_mean"] is not None else "N/A"
        pt = out["perm_test_vs_router"]
        sig_str = f"p={pt['p_one_sided']:.3f}" + ("*" if pt["significant_p05"] else "")
        print(f"{name:<28} {out['pos_rate']:>9} {out['mean_pct']:>7} {out['p05_pct']:>8} "
              f"{out['beat_bh']:>8} {dw_str:>8} {out['avg_expo']:>6} {sig_str:>10}")

    runtime = round(time.time() - t0, 1)
    results["runtime_s"] = runtime
    outp = ROOT.parent / "runs" / "strat" / "practitioner_engine_results.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved: {outp}  ({runtime}s)")

    # Honest verdict
    print("\n" + "=" * 76)
    print("VERDICT PROTOCOL")
    print("=" * 76)
    sig_wins = [(n, v) for n, v in results["strategies"].items()
                if v["perm_test_vs_router"]["significant_p05"]]
    pareto_wins = [(n, v) for n, v in results["strategies"].items()
                   if v["pos_rate"] > rpr or v["mean_pct"] > rmn or v["p05_pct"] > rp05]
    print(f"  Strategies with significant delta vs router (p<0.05): {len(sig_wins)}")
    for n, v in sig_wins:
        print(f"    {n}: delta={v['perm_test_vs_router']['observed_mean_delta_bps']}bps p={v['perm_test_vs_router']['p_one_sided']}")
    print(f"  Strategies Pareto-improving on any metric: {len(pareto_wins)}")
    for n, v in pareto_wins:
        print(f"    {n}: pos_rate={v['pos_rate']}% mean={v['mean_pct']}% p05={v['p05_pct']}%")
    if not sig_wins and not pareto_wins:
        print("  RESULT: ALL levers FAIL -- management overlay adds no value OOS.")
        print("  The router is the ceiling for internal-daily-LO data. CONVERGE.")
    elif not sig_wins:
        print("  RESULT: Pareto improvement without significance -- noise / variance reduction, not alpha.")
        print("  Could be useful operationally (smoother equity) but not a durable edge.")
    else:
        print("  RESULT: At least one lever shows significant improvement. Report for user review.")
    print(f"\nTotal runtime: {runtime}s")

    return results


if __name__ == "__main__":
    main()
