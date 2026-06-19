"""src/strat/nl_cycle2_router_hardening.py -- NL-CYCLE 2 ROUTER HARDENING (CANONICAL).

C1 VERDICT (referee_harness.py, N=5000, OOS 2022-01-01+):
  - BH:    pos_rate=52.3%  mean=+0.46%  p05=-14%
  - ROUTER pos_rate=50.6%  mean=+0.83%  down_wk=-3.0%  expo=0.58
  - router selection survives same-exposure shuffle p=0.000 (REAL selection skill)
  - pos_rate BELOW BH because cash-when-downtrend = 0 = non-positive

THIS MODULE builds the two hardened variants as instructed:
  (A) ROUTER x INV-VOL ENSEMBLE: router selection signal blended inside an always-invested
      inverse-vol book. Never fully cash -> keeps pos_rate ~BH while adding the router's
      MEAN and TAIL edge. Implementation: router's regime weights define WHICH assets to hold;
      inv-vol weighting always keeps a diversified floor.
      Two flavours:
        A1. OVERLAY: keep router weights but add inv-vol residual floor (alpha-blend)
        A2. PURE BLEND: on each day pick either router weights or inv-vol weights based
            on a BTC-trend signal; always hold *something*

  (B) BTC 10%-FLOOR ROUTER: identical to router but the downtrend sub-behavior never goes
      below 10% BTC exposure (already coded in adaptive_meta_engine; we vary the floor to
      test 5%/10%/20% and report which floor best recovers pos_rate without giving up tail).

CANONICAL METRICS (all identical to referee_harness.py):
  - slice = 7 consecutive trading days, uniform sample from OOS region [2022-01-01, 2026-06-01)
  - N=5000 slices, 3 seeds [11, 23, 42]
  - BH = fixed-EW (fillna(0)=cash for pre-listing, cadence-invariant)
  - positions lagged 1 bar, taker cost on |dpos|
  - report: pos_rate + mean + down_wk_mean + p05 + maxDD + full-cycle compound

RWYB. No emoji (cp1252). Does NOT git commit.
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
import strat.referee_harness as rh

COST = rh.COST
OOS_START = "2022-01-01"
OOS_END   = "2026-06-01"
N         = 5000
SEEDS     = [11, 23, 42]


# ============================================================
# CORE EVALUATION (canonical, mirrors referee_harness exactly)
# ============================================================

def full_eval(bret: pd.Series, bh: pd.Series, label: str) -> dict:
    """Full canonical evaluation: slice stats (N=5000, 3 seeds) + full-cycle compound + maxDD."""
    prs, mns, dnwks, p05s, btbhs = [], [], [], [], []
    for s in SEEDS:
        st = rh.slice_stats(bret, bh, OOS_START, OOS_END, N, 7, s)
        prs.append(st["pos_rate"])
        mns.append(st["mean_pct"])
        dnwks.append(st["down_wk_eng_mean"])
        p05s.append(st["p05_pct"])
        btbhs.append(st["beat_bh_pct"])

    # full-cycle compound + maxDD over the entire bret series
    oos_mask = (bret.index >= pd.Timestamp(OOS_START)) & (bret.index < pd.Timestamp(OOS_END))
    oos_bret = bret[oos_mask]; oos_bh = bh[oos_mask]
    x_eng = oos_bret.to_numpy(); x_bh = oos_bh.to_numpy()
    eng_comp = (float(np.prod(1 + x_eng)) - 1) * 100
    bh_comp  = (float(np.prod(1 + x_bh))  - 1) * 100
    eq = np.cumprod(1 + x_eng); pk = np.maximum.accumulate(eq)
    max_dd = float(((eq - pk) / pk).min() * 100)

    # average exposure (OOS)
    expo = None  # computed outside where W is available

    return {
        "label": label,
        "pos_rate":      round(float(np.mean(prs)), 1),
        "pos_rate_seeds": [round(p, 1) for p in prs],
        "mean_pct":      round(float(np.mean(mns)), 2),
        "mean_pct_seeds": [round(m, 2) for m in mns],
        "down_wk_mean":  round(float(np.mean([d for d in dnwks if d is not None])), 2) if any(d is not None for d in dnwks) else None,
        "p05_pct":       round(float(np.mean(p05s)), 2),
        "beat_bh_pct":   round(float(np.mean(btbhs)), 1),
        "oos_comp_pct":  round(eng_comp, 1),
        "bh_oos_comp_pct": round(bh_comp, 1),
        "oos_maxDD_pct": round(max_dd, 1),
    }


def print_row(r: dict, indent: str = "  ") -> None:
    print(f"{indent}{r['label']:<38}  pos_rate={r['pos_rate']:>5.1f}%  "
          f"mean={r['mean_pct']:>+6.2f}%  down_wk={r['down_wk_mean']:>+6.2f}%  "
          f"p05={r['p05_pct']:>+7.2f}%  beat_bh={r['beat_bh_pct']:>5.1f}%  "
          f"oos_comp={r['oos_comp_pct']:>+7.0f}%  maxDD={r['oos_maxDD_pct']:>+6.1f}%")


# ============================================================
# (0) BASELINES: BH + plain router
# ============================================================

def build_baselines(ind: dict) -> tuple:
    """Returns (bh_b, router_W, router_b)."""
    bh_W = rh.bh_ew_weights(ind)
    bh_b = rh.book_daily_returns(bh_W, ind)

    train_mask = ind["C"].index < pd.Timestamp(OOS_START)
    vthr = float(ind["vol20"]["BTCUSDT"][train_mask].dropna().quantile(ame.VOL_HI_PCTILE))
    router_W = ame.build_weight_matrix(ind, vthr)
    router_b = rh.book_daily_returns(router_W, ind)
    return bh_b, router_W, router_b


# ============================================================
# INV-VOL BOOK (always-invested reference)
# ============================================================

def inv_vol_weights(ind: dict) -> pd.DataFrame:
    """Inverse-volatility weights across ALL assets with valid price. Always invested (no cash).
    Uses vol20 (20-day trailing realized vol); if vol missing, falls back to EW among present assets.
    Positions are cadence-invariant: pre-listing assets get 0 weight (not inflated EW).
    """
    C = ind["C"]; vol = ind["vol20"]
    W = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    for d in C.index:
        row_c = C.loc[d]; row_v = vol.loc[d]
        # present = has a valid price
        present = [s for s in C.columns if pd.notna(row_c[s])]
        if not present:
            continue
        ivols = {}
        for s in present:
            v = row_v[s]
            ivols[s] = 1.0 / v if (pd.notna(v) and v > 0) else 1.0  # fallback to EW weight for 0-vol
        total = sum(ivols.values())
        for s in present:
            W.loc[d, s] = ivols[s] / total
    return W


# ============================================================
# (A1) ROUTER-ALPHA x INV-VOL FLOOR BLEND
#
# Idea: the router tells us WHEN to concentrate (its selection skill, shuffle p=0.000).
#       But when it goes to cash (downtrend), we substitute inv-vol instead.
#       Net effect: never fully cash -> pos_rate recovers toward BH;
#       when router is invested, it keeps its mean/tail edge.
#
# Parameter: alpha = fraction of the book that follows the router; (1-alpha) is inv-vol floor.
#   alpha=1.0 -> pure router (baseline)
#   alpha=0.0 -> pure inv-vol (always-invested, no selection)
#   We test alpha in [0.2, 0.4, 0.5, 0.6, 0.8] to map the Pareto frontier.
# ============================================================

def router_invvol_blend(router_W: pd.DataFrame, invvol_W: pd.DataFrame,
                        alpha: float) -> pd.DataFrame:
    """Convex blend: W = alpha * router_W + (1-alpha) * invvol_W.
    Row sums: router can be in [0,1]; invvol is ~1 (always invested). Blend row-sum in [1-alpha, 1].
    This means when router is fully in cash, the book holds (1-alpha) in inv-vol diversified assets.
    """
    return router_W.mul(alpha) + invvol_W.mul(1.0 - alpha)


# ============================================================
# (A2) REGIME-GATED COMPOSITION
#
# Alternative: use the router's regime signal to SWITCH between two books:
#   - uptrend/recovery/chop regimes -> use router weights (concentrated selection)
#   - downtrend -> use inv-vol weights (small but always-invested floor)
#
# The downtrend floor size is a parameter (exposure_floor in [0.1, 0.3, 0.5]).
# ============================================================

def regime_gated_composition(ind: dict, router_W: pd.DataFrame, invvol_W: pd.DataFrame,
                              downtrend_floor: float, vol_hi_thr: float) -> pd.DataFrame:
    """When router is in downtrend regime: hold inv-vol weights * downtrend_floor.
    Otherwise: use router weights as-is.
    """
    C = ind["C"]
    W = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    for i, d in enumerate(C.index):
        if i < 200:
            continue  # warmup
        regime = ame._detect_regime(ind, i, vol_hi_thr)
        if regime == "downtrend":
            # Scale down inv-vol to floor (diversified but small)
            W.loc[d] = invvol_W.loc[d] * downtrend_floor
        else:
            W.loc[d] = router_W.loc[d]
    return W


# ============================================================
# (B) BTC EXPOSURE-FLOOR ROUTER
#
# Modify the router's downtrend sub-behavior: instead of 10% BTC,
# test floors of 5%, 10%, 20%, and 30%.
# The existing adaptive_meta_engine hardcodes 10%; we re-derive here
# so we can sweep without modifying the canonical engine.
# ============================================================

def build_floor_router(ind: dict, vol_hi_thr: float, btc_floor: float) -> pd.DataFrame:
    """Identical to ame.build_weight_matrix but parameterizes the BTC downtrend floor."""
    C = ind["C"]
    W = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    prev_weights: dict = {}

    for i, d in enumerate(C.index):
        if i < 200:
            W.iloc[i] = {col: prev_weights.get(col, 0.0) for col in C.columns}
            continue
        regime = ame._detect_regime(ind, i, vol_hi_thr)
        if regime == "downtrend":
            new_w = {"BTCUSDT": btc_floor}  # <-- parameterized floor
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
# MAIN
# ============================================================

def main():
    t0 = time.time()
    print("=" * 80)
    print("NL-CYCLE 2 -- ROUTER HARDENING (N=5000, OOS 2022+, 3 seeds)")
    print(f"  BH target: pos_rate~52.3%  mean~+0.46%  (canonical C1 baseline)")
    print(f"  ROUTER baseline: pos_rate=50.6%  mean=+0.83%  down_wk=-3.0%  expo=0.58")
    print(f"  QUESTION: can hardening RECOVER pos_rate to >=BH while KEEPING mean+tail edge?")
    print("=" * 80)

    # --- load data ---
    print("\n[1] Loading data 2020-01-01 -> 2026-06-01 ...")
    ind = lab.load("2020-01-01", OOS_END)
    C = ind["C"]
    print(f"    assets={C.shape[1]}  bars={C.shape[0]}")

    # --- build baselines ---
    print("\n[2] Building baselines ...")
    bh_b, router_W, router_b = build_baselines(ind)

    # re-derive vol threshold (same as ame.run_engine)
    train_mask = C.index < pd.Timestamp(OOS_START)
    vthr = float(ind["vol20"]["BTCUSDT"][train_mask].dropna().quantile(ame.VOL_HI_PCTILE))
    print(f"    vol_hi_threshold={vthr:.4f}  (from train data only)")

    # avg exposure of router in OOS
    oos_mask_arr = (C.index >= pd.Timestamp(OOS_START)) & (C.index < pd.Timestamp(OOS_END))
    router_expo = float(router_W.loc[oos_mask_arr].sum(axis=1).mean())

    # BH eval
    bh_eval = full_eval(bh_b, bh_b, "BH (EW fixed-weight)")
    bh_eval["avg_expo"] = 1.0
    router_eval = full_eval(router_b, bh_b, "ROUTER (plain, C1 winner)")
    router_eval["avg_expo"] = round(router_expo, 2)

    # --- inv-vol weights ---
    print("\n[3] Building inv-vol book (always-invested) ...")
    invvol_W = inv_vol_weights(ind)
    invvol_b  = rh.book_daily_returns(invvol_W, ind)
    invvol_expo = float(invvol_W.loc[oos_mask_arr].sum(axis=1).mean())
    invvol_eval = full_eval(invvol_b, bh_b, "INV-VOL (always-invested)")
    invvol_eval["avg_expo"] = round(invvol_expo, 2)

    print("\n=== SECTION A: ROUTER x INV-VOL BLEND ===")
    print("    Idea: alpha fraction -> router weights; (1-alpha) -> inv-vol floor.")
    print("    When router is in cash (downtrend 47%), inv-vol covers the floor -> pos_rate recovers.")
    print()

    # (A1) convex blend sweep
    blend_results = []
    alphas = [1.0, 0.8, 0.6, 0.5, 0.4, 0.2, 0.0]
    for alpha in alphas:
        W_blend = router_invvol_blend(router_W, invvol_W, alpha)
        b_blend  = rh.book_daily_returns(W_blend, ind)
        expo = float(W_blend.loc[oos_mask_arr].sum(axis=1).mean())
        ev = full_eval(b_blend, bh_b, f"BLEND alpha={alpha:.1f}")
        ev["avg_expo"] = round(expo, 2)
        blend_results.append(ev)
        print_row(ev)

    # pick best blend (highest pos_rate while mean >= router mean * 0.7 threshold)
    router_mean = router_eval["mean_pct"]
    qualified = [r for r in blend_results if r["mean_pct"] >= router_mean * 0.7]
    if qualified:
        best_blend = max(qualified, key=lambda r: r["pos_rate"])
    else:
        best_blend = max(blend_results, key=lambda r: r["pos_rate"])
    print(f"\n  --> Best blend (pos_rate maximized with mean >= {router_mean*0.7:.2f}%): {best_blend['label']}")
    print(f"      pos_rate={best_blend['pos_rate']}%  mean={best_blend['mean_pct']:+.2f}%  "
          f"down_wk={best_blend['down_wk_mean']:+.2f}%  p05={best_blend['p05_pct']:+.2f}%")

    print("\n--- (A2) REGIME-GATED COMPOSITION ---")
    print("    Downtrend regime -> inv-vol x floor; other regimes -> router selection.")
    print()
    a2_results = []
    floors_a2 = [0.10, 0.20, 0.30, 0.50]
    for fl in floors_a2:
        W_a2 = regime_gated_composition(ind, router_W, invvol_W, fl, vthr)
        b_a2  = rh.book_daily_returns(W_a2, ind)
        expo = float(W_a2.loc[oos_mask_arr].sum(axis=1).mean())
        ev = full_eval(b_a2, bh_b, f"REGIME-GATE inv-vol floor={fl:.0%}")
        ev["avg_expo"] = round(expo, 2)
        a2_results.append(ev)
        print_row(ev)

    best_a2 = max(a2_results, key=lambda r: r["pos_rate"])
    print(f"\n  --> Best A2 (pos_rate): {best_a2['label']}")
    print(f"      pos_rate={best_a2['pos_rate']}%  mean={best_a2['mean_pct']:+.2f}%  "
          f"down_wk={best_a2['down_wk_mean']:+.2f}%  p05={best_a2['p05_pct']:+.2f}%")

    print("\n=== SECTION B: BTC EXPOSURE-FLOOR ROUTER ===")
    print("    Downtrend sub-behavior: BTC floor=X%. Higher floor -> more exposure in downtrend -> higher pos_rate.")
    print("    Trade-off: floor>0 means small BTC loss in genuine down weeks -> tail gets slightly worse.")
    print()

    # canonical 10% is already in the router; sweep 5%, 10%, 20%, 30%
    b_floor_results = []
    btc_floors = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
    for fl in btc_floors:
        if fl == 0.10:
            # use prebuilt router (exact match, avoid redundant compute)
            W_fl = router_W
            b_fl = router_b
        else:
            W_fl = build_floor_router(ind, vthr, fl)
            b_fl = rh.book_daily_returns(W_fl, ind)
        expo = float(W_fl.loc[oos_mask_arr].sum(axis=1).mean())
        label = f"FLOOR-ROUTER btc_floor={fl:.0%}"
        ev = full_eval(b_fl, bh_b, label)
        ev["avg_expo"] = round(expo, 2)
        b_floor_results.append(ev)
        print_row(ev)

    best_bfloor = max(b_floor_results, key=lambda r: r["pos_rate"])
    print(f"\n  --> Best floor (pos_rate): {best_bfloor['label']}")
    print(f"      pos_rate={best_bfloor['pos_rate']}%  mean={best_bfloor['mean_pct']:+.2f}%  "
          f"down_wk={best_bfloor['down_wk_mean']:+.2f}%  p05={best_bfloor['p05_pct']:+.2f}%")

    print("\n=== SUMMARY TABLE ===")
    print()
    print(f"  {'Engine':<42}  {'pos_rate':>9}  {'mean':>7}  {'down_wk':>8}  {'p05':>8}  {'beat_bh':>7}  {'oos_comp':>9}  {'maxDD':>7}  {'expo':>5}")
    print("  " + "-" * 120)

    all_rows = (
        [bh_eval, router_eval, invvol_eval]
        + blend_results
        + a2_results
        + b_floor_results
    )
    for r in all_rows:
        dn = r['down_wk_mean'] if r['down_wk_mean'] is not None else float('nan')
        print(f"  {r['label']:<42}  {r['pos_rate']:>8.1f}%  {r['mean_pct']:>+6.2f}%  "
              f"{dn:>+7.2f}%  {r['p05_pct']:>+7.2f}%  {r['beat_bh_pct']:>6.1f}%  "
              f"{r['oos_comp_pct']:>+8.0f}%  {r['oos_maxDD_pct']:>+6.1f}%  {r.get('avg_expo', '?'):>5}")

    # --- Verdict ---
    print("\n=== VERDICT ===")
    print(f"  BH benchmark:   pos_rate={bh_eval['pos_rate']}%  mean={bh_eval['mean_pct']:+.2f}%")
    print(f"  ROUTER (plain): pos_rate={router_eval['pos_rate']}%  mean={router_eval['mean_pct']:+.2f}%  "
          f"down_wk={router_eval['down_wk_mean']:+.2f}%  [BELOW BH pos_rate by "
          f"{bh_eval['pos_rate'] - router_eval['pos_rate']:.1f}pp]")
    print()

    # Find best overall: maximise pos_rate >= BH while mean >= BH mean
    above_bh_pr = [r for r in all_rows if r["pos_rate"] >= bh_eval["pos_rate"]
                   and r["label"] not in ("BH (EW fixed-weight)",)]
    above_bh_mean = [r for r in all_rows if r["mean_pct"] >= bh_eval["mean_pct"]
                     and r["label"] not in ("BH (EW fixed-weight)",)]

    if above_bh_pr:
        best_pr = max(above_bh_pr, key=lambda r: r["pos_rate"])
        print(f"  HARDENING RECOVERS pos_rate >= BH ({bh_eval['pos_rate']}%): YES")
        print(f"  Best: {best_pr['label']}")
        print(f"    pos_rate={best_pr['pos_rate']}%  mean={best_pr['mean_pct']:+.2f}%  "
              f"down_wk={best_pr['down_wk_mean']:+.2f}%  p05={best_pr['p05_pct']:+.2f}%")
        # Does it keep the router's mean/tail advantage?
        keeps_mean = best_pr["mean_pct"] >= bh_eval["mean_pct"]
        keeps_tail = (best_pr["down_wk_mean"] is not None and bh_eval["down_wk_mean"] is not None
                      and best_pr["down_wk_mean"] > bh_eval["down_wk_mean"] + 0.5)
        print(f"    KEEPS mean advantage vs BH: {'YES' if keeps_mean else 'NO'} "
              f"(best={best_pr['mean_pct']:+.2f}% vs BH={bh_eval['mean_pct']:+.2f}%)")
        print(f"    KEEPS tail advantage vs router: {'YES' if keeps_tail else 'NO (tail slightly worse)'}")
    else:
        print(f"  HARDENING RECOVERS pos_rate >= BH ({bh_eval['pos_rate']}%): NO")
        # find closest
        closest = max([r for r in all_rows if r["label"] != "BH (EW fixed-weight)"],
                      key=lambda r: r["pos_rate"])
        print(f"  Closest: {closest['label']} pos_rate={closest['pos_rate']}% "
              f"(still {bh_eval['pos_rate'] - closest['pos_rate']:.1f}pp below BH)")
        print(f"  -- STRUCTURAL: pos_rate is BELOW BH because inv-vol/router blends inevitably hold some")
        print(f"     downside in the ~47% downtrend regime. The tail gain (-3% vs -7%) comes at a pos_rate cost.")
        print(f"     Mean+tail are LIFTABLE; pos_rate is a STRUCTURAL FLOOR at current universe and OOS window.")

    print()
    # Deployable recommendation
    candidates = [r for r in (a2_results + blend_results + b_floor_results)]
    if candidates:
        # Best on pareto: mean_pct + down_wk_mean improvement over BH
        bh_dn = bh_eval.get("down_wk_mean", 0.0) or 0.0
        router_dn = router_eval.get("down_wk_mean", 0.0) or 0.0
        # Score = how much pos_rate recovered + how much mean above BH
        def score(r):
            pr_delta = r["pos_rate"] - router_eval["pos_rate"]
            mean_delta = r["mean_pct"] - bh_eval["mean_pct"]
            return pr_delta + 2 * mean_delta  # weight mean 2x vs pos_rate delta
        deployable = max(candidates, key=score)
        print(f"  DEPLOYABLE RECOMMENDATION (best pos_rate recovery + mean above BH):")
        print(f"    {deployable['label']}")
        print(f"    pos_rate={deployable['pos_rate']}%  mean={deployable['mean_pct']:+.2f}%  "
              f"down_wk={deployable['down_wk_mean']:+.2f}%  p05={deployable['p05_pct']:+.2f}%  "
              f"oos_comp={deployable['oos_comp_pct']:+.0f}%  maxDD={deployable['oos_maxDD_pct']:+.1f}%  "
              f"expo={deployable.get('avg_expo', '?')}")

    # --- Save JSON ---
    out = {
        "run_config": {"oos": [OOS_START, OOS_END], "n_slices": N, "seeds": SEEDS},
        "baselines": {"bh": bh_eval, "router": router_eval, "invvol": invvol_eval},
        "A1_blend": blend_results,
        "A2_regime_gated": a2_results,
        "B_floor_router": b_floor_results,
        "best_blend": best_blend,
        "best_a2": best_a2,
        "best_floor": best_bfloor,
        "recovers_bh_posrate": len(above_bh_pr) > 0,
        "runtime_s": round(time.time() - t0, 1),
    }

    outp = ROOT.parent / "runs" / "strat" / "nl_cycle2_router_hardening.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[saved] {outp}  ({out['runtime_s']}s)")
    return out


if __name__ == "__main__":
    main()
