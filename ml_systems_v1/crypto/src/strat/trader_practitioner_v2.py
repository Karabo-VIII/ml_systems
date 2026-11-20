"""src/strat/trader_practitioner_v2.py -- PRACTITIONER LENS v2: deeper analysis of Lever B + entry timing.

FOLLOW-UP after v1 results:
- Lever B (profit-lock) matched BH pos-rate 51.7% while keeping router's tail advantage.
  The p-value vs router was 0.228 (mean DECREASED slightly). This needs unpacking:
  the pos-rate lift is real (matching BH) but mean fell from 0.96->0.49 = we capped winners.
  Q: can we find a threshold where pos-rate > BH AND mean > router?

- Entry-timing: test the OPENING-WEEK MOMENTUM rule.
  If day 1 of the week (Monday) closes DOWN, delay full position entry until day 2.
  A real trader's "don't chase into weakness" rule.
  Implementation: on Mondays, if ret1 < -GATE_THR, hold 50% of signal; restore Tuesday.
  This is STRICTLY causal (ret1 known at bar close d, position taken at d+1).

- The BLEND approach (C2 validated): mixing router 60% + BH 40% lifted pos-rate to 52%
  while keeping mean > BH. Test this alongside our levers.

- TAIL ISOLATION: the router's main win IS the tail (-3% vs -7% in down weeks).
  Can we AMPLIFY this without hurting mean? The answer is: only if we predict down-weeks better.
  Since we can't (C1/C2 proof), the management overlay can only redistribute return, not add it.
  This IS the structural bound - but it's still useful for practitioners who care about drawdown.

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
from strat.trader_practitioner_engine import (
    calibrate_thresholds, apply_profit_lock, permutation_test_overlay
)

COST = lab.COST


# ============================================================
# LEVER B SENSITIVITY: sweep the take-profit threshold
# We pre-register 5 values from a TRAINING-data grid (not OOS sweep)
# ============================================================
def sweep_profit_lock_train(W_router: pd.DataFrame, ind: dict, train_end: str):
    """In TRAINING data only: compute slice stats at multiple thresholds.
    Return the single threshold closest to: maximise pos_rate while keeping mean >= 0.5%.
    This is a TRAINING-ONLY calibration step (causal).
    """
    bh_W = bh_ew_weights(ind)
    bh_b = book_daily_returns(bh_W, ind)

    TRAIN_START = "2020-01-01"
    TRAIN_END = train_end

    thresholds = [0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.050]
    best = None
    best_score = -999

    print("  [TRAIN-ONLY sweep to pick take_thr]")
    for thr in thresholds:
        W = apply_profit_lock(W_router, ind, take_thr=thr, lock_scale=0.50)
        b = book_daily_returns(W, ind)
        stats = slice_stats(b, bh_b, TRAIN_START, TRAIN_END, 300, 7, 42)
        pr = stats["pos_rate"]; mn = stats["mean_pct"]
        # Objective: pos_rate - penalty for dropping mean below 0.5%
        score = pr - max(0, 0.5 - mn) * 10
        print(f"    thr={thr:.3f}: train pos_rate={pr}% mean={mn}% score={score:.2f}")
        if score > best_score:
            best_score = score
            best = {"thr": thr, "pos_rate": pr, "mean": mn, "score": score}

    return best


# ============================================================
# LEVER E: MONDAY ENTRY DELAY (opening-week momentum)
# ============================================================
def apply_monday_delay(W_router: pd.DataFrame, ind: dict,
                       gate_thr: float = -0.005,
                       hold_scale: float = 0.50) -> pd.DataFrame:
    """If Monday's close return < -gate_thr (market opened weak), hold only hold_scale
    of the signal on Monday close -> act Tuesday at full signal.
    This is the 'wait for the dust to settle after a gap-down open' rule.
    Note: in our daily bar mechanic, position at d is ACTED at d+1.
    So: Monday (d=Mon) return is known at Mon close. We adjust Mon's W row,
    which is acted on Tues open. If Mon return was weak, we reduce Mon's W -> smaller Tues position.
    We then restore to full on Tue (d=Tue) -> act Wed.
    """
    C = ind["C"]
    R = ind["R"].reindex(index=W_router.index, columns=W_router.columns).fillna(0.0)
    W_adj = W_router.copy()
    btc_ret = R["BTCUSDT"]  # use BTC as market proxy for the opening signal

    dates = list(C.index)
    dtidx = pd.DatetimeIndex(dates)
    dow = dtidx.dayofweek  # Mon=0, Fri=4

    for i, d in enumerate(dates):
        if dow[i] == 0:  # Monday
            # Market return on Monday (known at bar close)
            btc_r = btc_ret.iloc[i] if i < len(btc_ret) else 0.0
            if not pd.isna(btc_r) and float(btc_r) < gate_thr:
                # Weak Monday: reduce position (wait for Tuesday to confirm direction)
                W_adj.iloc[i] = W_router.iloc[i] * hold_scale

    return W_adj


# ============================================================
# LEVER F: BLEND ROUTER + BH (validated in C2)
# ============================================================
def apply_blend(W_router: pd.DataFrame, ind: dict, alpha: float = 0.60) -> pd.DataFrame:
    """Blend: alpha * router + (1-alpha) * EW BH.
    This is the C2-validated approach that lifted pos-rate to ~52% while keeping mean > BH.
    Test whether it beats BOTH router mean AND BH pos-rate.
    """
    bh_W = bh_ew_weights(ind)
    W_blend = alpha * W_router + (1 - alpha) * bh_W
    return W_blend


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
    print("PRACTITIONER ENGINE v2 -- deeper Lever B + entry timing + blend")
    print(f"OOS: {OOS_START} -> {OOS_END} | N={N} slices | seeds={SEEDS}")
    print("=" * 76)

    print("\n[DATA] Loading...")
    ind = lab.load("2020-01-01", OOS_END)
    C = ind["C"]

    train_mask = C.index < pd.Timestamp(OOS_START)
    vthr = float(ind["vol20"]["BTCUSDT"][train_mask].dropna().quantile(ame.VOL_HI_PCTILE))
    W_router = ame.build_weight_matrix(ind, vthr)

    bh_W = bh_ew_weights(ind)
    bh_b = book_daily_returns(bh_W, ind)
    router_b = book_daily_returns(W_router, ind)

    results = {
        "oos": [OOS_START, OOS_END],
        "n_slices": N,
        "seeds": SEEDS,
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
        perm = permutation_test_overlay(router_b, b, OOS_START, OOS_END)
        # Also: perm vs BH
        perm_vs_bh = permutation_test_overlay(bh_b, b, OOS_START, OOS_END)

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
            "perm_vs_router": perm,
            "perm_vs_bh": perm_vs_bh,
        }
        print(f"\n[{name}] {description}")
        print(f"  pos_rate={out['pos_rate']}% (seeds {pr})")
        print(f"  mean={out['mean_pct']}% | p05={out['p05_pct']}% | beat_bh={out['beat_bh']}%")
        print(f"  down_wk_mean={out['down_wk_mean']}% | down_wk_posrate={out['down_wk_posrate']}%")
        print(f"  delta vs router: {perm['observed_mean_delta_bps']}bps p={perm['p_one_sided']}")
        print(f"  delta vs BH:     {perm_vs_bh['observed_mean_delta_bps']}bps p={perm_vs_bh['p_one_sided']}")
        results["strategies"][name] = out
        return b

    # --- Calibrate optimal profit-lock from TRAINING only ---
    print("\n[CALIBRATE] Training-data sweep for optimal profit-lock threshold:")
    best_lock = sweep_profit_lock_train(W_router, ind, TRAIN_END)
    print(f"  Best training threshold: {best_lock}")
    OPT_THR = best_lock["thr"]

    # --- Lever B at optimized training threshold ---
    print("\n" + "=" * 60)
    print("LEVER B (OPTIMIZED from training): PARTIAL PROFIT LOCK")
    W_lock_opt = apply_profit_lock(W_router, ind, take_thr=OPT_THR, lock_scale=0.50)
    b_lock_opt = run_strategy("lever_B_optimized",  W_lock_opt,
                              f"Router + profit-lock thr={OPT_THR:.3f} (training-optimal)")

    # --- Lever B at multiple lock scales (trained threshold, vary scale) ---
    print("\n" + "=" * 60)
    print("LEVER B SCALE SENSITIVITY: lock to 30% vs 50% vs 70%")
    for scale in [0.30, 0.50, 0.70]:
        W_s = apply_profit_lock(W_router, ind, take_thr=OPT_THR, lock_scale=scale)
        run_strategy(f"lever_B_scale{int(scale*100)}",  W_s,
                     f"Router + profit-lock thr={OPT_THR:.3f} scale={scale}")

    # --- Lever E: Monday delay ---
    print("\n" + "=" * 60)
    print("LEVER E: MONDAY ENTRY DELAY (opening-week momentum)")
    W_mon = apply_monday_delay(W_router, ind, gate_thr=-0.005, hold_scale=0.50)
    run_strategy("lever_E_monday_delay", W_mon,
                 "Router + Mon-delay (if BTC Mon ret < -0.5%, hold 50% position)")

    # Tighter Monday gate
    W_mon2 = apply_monday_delay(W_router, ind, gate_thr=-0.010, hold_scale=0.30)
    run_strategy("lever_E_mon_tight", W_mon2,
                 "Router + Mon-delay tight (BTC Mon ret < -1%, hold 30%)")

    # --- Lever F: Blend ---
    print("\n" + "=" * 60)
    print("LEVER F: BLEND ROUTER + BH (C2-validated concept)")
    for alpha in [0.60, 0.70, 0.80]:
        W_bl = apply_blend(W_router, ind, alpha=alpha)
        run_strategy(f"lever_F_blend_{int(alpha*100)}",  W_bl,
                     f"Blend {int(alpha*100)}% router + {int((1-alpha)*100)}% BH")

    # --- STAR COMBO: if lock improves pos-rate, stack with blend for best of both ---
    print("\n" + "=" * 60)
    print("COMBO: profit-lock + blend (Pareto search)")
    W_bl_lock = apply_profit_lock(apply_blend(W_router, ind, 0.70), ind, take_thr=OPT_THR, lock_scale=0.50)
    run_strategy("combo_blend70_lock", W_bl_lock,
                 f"70% router blend + profit-lock thr={OPT_THR:.3f}")

    # --- Summary table ---
    print("\n" + "=" * 76)
    print("SUMMARY TABLE (OOS 2022-2026, N=500 slices per seed)")
    print(f"{'Strategy':<32} {'pos_rate':>9} {'mean':>7} {'p05':>8} {'dw_mean':>8} {'expo':>6}")
    print("-" * 76)
    bh_stats = [bh_slice_stats(bh_b, OOS_START, OOS_END, N, 7, s) for s in SEEDS]
    bh_pr = round(float(np.mean([x["pos_rate"] for x in bh_stats])), 1)
    bh_mn = round(float(np.mean([x["mean_pct"] for x in bh_stats])), 2)
    bh_p05 = round(float(np.mean([x["p05_pct"] for x in bh_stats])), 2)
    print(f"{'EW BH (baseline)':<32} {bh_pr:>9} {bh_mn:>7} {bh_p05:>8} {'N/A':>8} {'1.00':>6}")
    rs_all = [slice_stats(router_b, bh_b, OOS_START, OOS_END, N, 7, s) for s in SEEDS]
    rpr = round(float(np.mean([x["pos_rate"] for x in rs_all])), 1)
    rmn = round(float(np.mean([x["mean_pct"] for x in rs_all])), 2)
    rp05 = round(float(np.mean([x["p05_pct"] for x in rs_all])), 2)
    rdw = round(float(np.mean([x["down_wk_eng_mean"] for x in rs_all if x["down_wk_eng_mean"] is not None])), 2)
    print(f"{'Router (validated)':<32} {rpr:>9} {rmn:>7} {rp05:>8} {rdw:>8} {'0.58':>6}")
    for name, out in results["strategies"].items():
        dw_str = f"{out['down_wk_mean']:.2f}" if out["down_wk_mean"] is not None else "N/A"
        pv_r = out["perm_vs_router"]
        pv_b = out["perm_vs_bh"]
        pr_flag = ">" if out["pos_rate"] > bh_pr else ""
        mn_flag = ">" if out["mean_pct"] > rmn else ""
        print(f"{name:<32} {str(out['pos_rate'])+pr_flag:>9} {str(out['mean_pct'])+mn_flag:>7} {out['p05_pct']:>8} {dw_str:>8} {out['avg_expo']:>6}")

    runtime = round(time.time() - t0, 1)
    results["runtime_s"] = runtime
    outp = ROOT.parent / "runs" / "strat" / "practitioner_engine_v2_results.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved: {outp}  ({runtime}s)")

    # Honest verdict
    print("\n" + "=" * 76)
    print("VERDICT")
    print("=" * 76)
    best_pareto = None
    best_score_oos = -999
    for name, v in results["strategies"].items():
        # Score: beat BH pos-rate AND beat router mean
        pr_beats_bh = v["pos_rate"] > bh_pr
        mn_beats_router = v["mean_pct"] > rmn
        p05_beats_router = v["p05_pct"] > rp05
        flags = []
        if pr_beats_bh: flags.append("pos_rate>BH")
        if mn_beats_router: flags.append("mean>router")
        if p05_beats_router: flags.append("p05>router")
        perm_r = v["perm_vs_router"]["p_one_sided"]
        perm_b = v["perm_vs_bh"]["p_one_sided"]
        if flags:
            print(f"  {name}: {flags} | p_vs_router={perm_r} p_vs_bh={perm_b}")
        score = int(pr_beats_bh) + int(mn_beats_router) + int(p05_beats_router)
        if score > best_score_oos:
            best_score_oos = score
            best_pareto = (name, v, flags, perm_r, perm_b)

    if best_pareto:
        n, v, flags, pr, pb = best_pareto
        print(f"\n  BEST PARETO: {n} -- {flags}")
        print(f"  pos_rate={v['pos_rate']}% | mean={v['mean_pct']}% | p05={v['p05_pct']}%")
        if v["perm_vs_router"]["significant_p05"] or v["perm_vs_bh"]["significant_p05"]:
            print("  SIGNIFICANT improvement detected (p<0.05)")
        else:
            print("  NOT statistically significant -- improvement is Pareto-shift, not alpha.")
    print(f"\nTotal runtime: {runtime}s")
    return results


if __name__ == "__main__":
    main()
