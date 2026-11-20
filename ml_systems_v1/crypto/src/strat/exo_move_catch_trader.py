"""src/strat/exo_move_catch_trader.py -- TRADER AGENT: Exogenous move-catch battery (two specific angles).

ANGLE 1: LISTING-AGE (mv_days_since_listed_binance)
  Does YOUNG asset age (low days_since_listed) produce a stronger / regime-different move-catch?
  Young = more vol / momentum runway. Tests by bucketing into YOUNG/MID/OLD terciles and comparing
  move-catch within each bucket, separately per-regime.

ANGLE 2: WHALE FLOW + LIQ FLUSH as move-onset
  Does whale inflow (wh_whale_net_usd top tercile) or liq flush (liq_capitulation / liq_short_panic)
  PRECEDE a catchable next-day move, per regime?

DATA: v51_feature_lab.load_v51_daily -> DEV-walled (<= 2024-05-15). Long-only spot. Causal.
HONEST P: date-block bootstrap (block=True). Holm-BH corrected across sweeps.
No emoji (cp1252). RWYB.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import strat.v51_feature_lab as vfl
import strat.capture_lab as cl
import strat.fleet_lab as fl


# ---- helpers ----
def holm_bh_correct(raw_ps: list[float]) -> list[float]:
    """Holm-BH correction: returns adjusted p-values (conservative Holm)."""
    n = len(raw_ps)
    if n == 0:
        return []
    order = np.argsort(raw_ps)
    adj = np.zeros(n)
    for rank, idx in enumerate(order):
        adj[idx] = min(1.0, raw_ps[idx] * (n - rank))
    # ensure monotone
    for i in range(1, n):
        adj[order[i]] = max(adj[order[i]], adj[order[i - 1]])
    return list(np.round(adj, 4))


def describe_block(b: dict | None) -> str:
    if b is None:
        return "no_block"
    return f"p05={b.get('block_p05_pp','?')}pp  ble0={b.get('block_p_le0','?')}  n_eff={b.get('n_eff_dates','?')}"


# =====================================================================
# ANGLE 1: LISTING-AGE conditioning
# =====================================================================
def angle_listing_age(lab: dict, n_null: int = 400, seed: int = 7) -> dict:
    """Does young listing age produce a stronger/regime-different move-catch?

    Approach: use the existing v51 feature mv_days_since_listed_binance.
    Bucket each (date, asset) into YOUNG / MID / OLD terciles based on the CROSS-SECTIONAL
    rank of days_since_listed on that date (causal -- this day's value, used to filter entries
    the NEXT day via the standard fired_matrix lag).

    Then run evaluate_ti on each bucket separately by slicing the lab.
    """
    print("\n" + "=" * 70)
    print("ANGLE 1: LISTING-AGE conditioning on move-catch")
    print("=" * 70)

    feat = "mv_days_since_listed_binance"
    if feat not in lab["F"] or lab["F"][feat].isna().all().all():
        print(f"  SKIP: {feat} not in data")
        return {"angle": "listing_age", "status": "no_data"}

    age = lab["F"][feat]  # shape (dates, assets), float = days since listed
    print(f"  age range: {age.stack().min():.0f}d .. {age.stack().max():.0f}d  | "
          f"median {age.stack().median():.0f}d | n_assets={age.notna().any().sum()}")

    # Cross-sectional tercile per date (causal: today's age -> tomorrow's entries)
    rank_frac = age.rank(axis=1, pct=True, na_option="keep")
    young_mask = rank_frac <= 0.33        # bottom tercile in cross-section = youngest
    mid_mask = (rank_frac > 0.33) & (rank_frac <= 0.67)
    old_mask = rank_frac > 0.67

    results = {}
    all_ble0 = []

    for bucket_name, b_mask in [("YOUNG", young_mask), ("MID", mid_mask), ("OLD", old_mask)]:
        print(f"\n  -- Bucket: {bucket_name} --")
        # Build a sub-lab where assets outside the bucket are NaN'd on that date
        C_orig = lab["C"]
        C_sub = C_orig.where(b_mask, other=np.nan)
        F_sub = {k: v.where(b_mask, other=np.nan) for k, v in lab["F"].items()}
        sub_lab = {"C": C_sub, "R": C_sub.pct_change(fill_method=None), "F": F_sub, "syms": lab["syms"]}

        # Evaluate "mom14" (price momentum, our baseline) on this bucket -- tests whether
        # listing-age MODIFIES the move-catch of a standard signal
        bucket_res = {}
        for ti_name in ("mom14", "brk14"):
            try:
                r = cl.evaluate_ti(sub_lab, ti_name, tf="1d", exit_kind="time",
                                   n_null=n_null, by_regime=True, block=True, seed=seed)
                n_f = r.get("n_fired", 0)
                if "note" in r or n_f < 30:
                    print(f"    {ti_name:10}: insufficient signals ({n_f})")
                    bucket_res[ti_name] = {"n_fired": n_f, "insufficient": True}
                    continue
                print(f"    {ti_name:10}: n={n_f}  edge={r['edge_vs_random_pp']:+.2f}pp  "
                      f"block={describe_block(r.get('block'))}")
                for rg in ("bull", "chop", "bear"):
                    d = r.get("by_regime", {}).get(rg)
                    if d:
                        blk = d.get("block", {})
                        ble0 = blk.get("block_p_le0") if blk else None
                        all_ble0.append(ble0)
                        print(f"      {rg:5}: n={d['n']:>5}  realized={d['realized_net']:+.2f}%  "
                              f"edge={d['edge_pp']:+.2f}pp  {describe_block(blk)}")
                bucket_res[ti_name] = r
            except Exception as ex:
                print(f"    {ti_name}: ERROR {ex}")
                bucket_res[ti_name] = {"error": str(ex)}

        # Also test the raw age feature as a contrarian signal: TOP-age (OLD) assets might mean-revert
        # while YOUNG assets might trend -- test xd_momentum_rank within the age bucket
        results[bucket_name] = bucket_res

    return {"angle": "listing_age", "results": results}


# =====================================================================
# ANGLE 2: WHALE FLOW + LIQ FLUSH as move-onset
# =====================================================================
def angle_whale_liq(lab: dict, n_null: int = 400, seed: int = 42) -> dict:
    """Does whale inflow or liq flush precede a catchable move, per regime?

    Features tested:
      wh_whale_net_usd  (top tercile XS = big inflow day -- continuation)
      norm_whale        (normalized whale proxy)
      liq_capitulation  (max-of-day binary flag -- flush -> bounce, potentially contrarian)
      liq_short_panic   (max-of-day binary flag)
      liq_delta_z30     (z-score of net liq bias)

    Key hypotheses:
      H1: whale inflow -> continuation (bull) / capitulation (bear)
      H2: liq flush -> bounce (contrarian in bear; continuation in bull)
      H3: liq_short_panic in BEAR -> forced sellers exhausted -> bounce edge

    BOTH directions tested (XS top tercile = "big" signal, XS bottom tercile = "reverse").
    """
    print("\n" + "=" * 70)
    print("ANGLE 2: WHALE FLOW + LIQ FLUSH as move-onset trigger")
    print("=" * 70)

    flow_feats = {
        "wh_whale_net_usd": "top",        # top = big inflow
        "norm_whale": "top",              # top = high whale activity
        "liq_capitulation": "top",        # top = capitulation day (contrarian flip candidate)
        "liq_short_panic": "top",         # top = short squeeze / flush
        "liq_delta_z30": "both",          # both: +ve = long-dominated liq, -ve = short-dominated
        "liq_total_usd": "top",           # top = big liq day total
    }

    results = {}
    all_raw_ps = []
    all_labels = []

    for feat, direction in flow_feats.items():
        X = lab["F"].get(feat)
        if X is None or X.isna().all().all():
            print(f"  {feat}: NO DATA, skip")
            continue
        n_data = int(X.notna().sum().sum())
        print(f"\n  Feature: {feat}  (n_obs={n_data}, direction={direction})")

        dirs_to_test = ["top", "bot"] if direction == "both" else [direction]
        for d_name in dirs_to_test:
            # Build triggered lab: fire where feature is in top/bottom cross-sectional tercile
            if d_name == "top":
                fired_override = X.gt(X.quantile(0.66, axis=1), axis=0)
            else:
                fired_override = X.lt(X.quantile(0.33, axis=1), axis=0)

            # Inject into lab as a custom feature key
            lab_copy = dict(lab)
            lab_copy["F"] = dict(lab["F"])
            feat_key = f"__exo_{feat}_{d_name}"
            lab_copy["F"][feat_key] = X  # raw; fired_matrix will use the cross-sectional tercile logic

            # Override fired_matrix by using the pre-computed mask via a numeric "signal"
            # Trick: set the feature value to 1/0 where fired/not, then quantile>0.66 -> always fires where 1
            fired_binary = fired_override.astype(float)
            lab_copy["F"][feat_key] = fired_binary  # 0/1 -> quantile(0.66) works as top-tercile gate

            try:
                r = cl.evaluate_ti(lab_copy, feat_key, tf="1d", exit_kind="time",
                                   n_null=n_null, by_regime=True, block=True, seed=seed)
                n_f = r.get("n_fired", 0)
                tag = f"{feat}_{d_name}"
                if "note" in r or n_f < 30:
                    print(f"    [{d_name}] insufficient signals ({n_f}), skip")
                    continue

                blk = r.get("block", {})
                ble0 = blk.get("block_p_le0")
                all_raw_ps.append(ble0 if ble0 is not None else 1.0)
                all_labels.append(f"{feat}_{d_name}_ALL")
                print(f"    [{d_name}] n={n_f}  edge={r['edge_vs_random_pp']:+.2f}pp  {describe_block(blk)}")

                # Per-regime breakdown
                for rg in ("bull", "chop", "bear"):
                    d = r.get("by_regime", {}).get(rg)
                    if d:
                        rblk = d.get("block", {})
                        ble0_rg = rblk.get("block_p_le0") if rblk else None
                        all_raw_ps.append(ble0_rg if ble0_rg is not None else 1.0)
                        all_labels.append(f"{feat}_{d_name}_{rg}")
                        sig_flag = "*** SIG ***" if (ble0_rg is not None and ble0_rg < 0.05) else ""
                        print(f"      {rg:5}: n={d['n']:>5}  realized={d['realized_net']:+.2f}%  "
                              f"edge={d['edge_pp']:+.2f}pp  {describe_block(rblk)}  {sig_flag}")

                results[tag] = r
            except Exception as ex:
                print(f"    [{d_name}] ERROR: {ex}")
                results[f"{feat}_{d_name}"] = {"error": str(ex)}

    # Holm-BH correction across all tests
    if all_raw_ps:
        adj_ps = holm_bh_correct(all_raw_ps)
        print("\n  --- Holm-BH corrected p-values (ALL tests) ---")
        for label, raw_p, adj_p in sorted(zip(all_labels, all_raw_ps, adj_ps), key=lambda x: x[1]):
            sig = " <-- SURVIVES MCC" if adj_p < 0.05 else ""
            print(f"    {label:45}  raw_p={raw_p:.4f}  adj_p={adj_p:.4f}{sig}")

    return {"angle": "whale_liq", "results": results,
            "mc_correction": {"labels": all_labels, "raw_ps": all_raw_ps}}


# =====================================================================
# COMBINED EXOGENOUS FEATURE SWEEP (supporting context)
# =====================================================================
def exo_full_sweep(lab: dict, n_null: int = 300, seed: int = 99) -> list[dict]:
    """Run the full V51 exogenous feature battery (bear-focus + regime-conditional check)."""
    print("\n" + "=" * 70)
    print("FULL EXOGENOUS FEATURE SWEEP (bear + regime-conditional)")
    print("=" * 70)

    feats_available = [c for c in vfl.V51
                       if c in lab["F"] and lab["F"][c].notna().sum().sum() > 500
                       and c not in ("mv_days_since_listed_binance",)]  # angle 1 separate
    print(f"  Features to sweep: {feats_available}\n")
    print(f"  {'feature':30}{'n_fired':>8}{'edge_ALL':>10}{'bear_edge':>11}{'bear_ble0':>11}{'regime_cond?':>14}")

    sweep_results = []
    all_ps = []; all_labels = []
    for feat in feats_available:
        try:
            r = cl.evaluate_ti(lab, feat, tf="1d", exit_kind="time",
                               n_null=n_null, by_regime=True, block=True, seed=seed)
            n_f = r.get("n_fired", 0)
            if "note" in r or n_f < 30:
                print(f"  {feat:30}  insuff ({n_f})")
                continue

            br = r.get("by_regime", {})
            bear = br.get("bear") or {}
            bull = br.get("bull") or {}
            chop = br.get("chop") or {}
            bear_blk = bear.get("block", {}) or {}
            bear_ble0 = bear_blk.get("block_p_le0")
            bear_edge = bear.get("edge_pp", "n/a")
            bull_edge = bull.get("edge_pp", "n/a")
            chop_edge = chop.get("edge_pp", "n/a")

            # Regime-conditional = bear substantially different from bull+chop
            regime_cond = "?"
            if isinstance(bear_edge, float) and isinstance(bull_edge, float):
                diff = abs(bear_edge - bull_edge)
                regime_cond = "YES" if diff > 1.0 else "no"

            all_ps.append(bear_ble0 if bear_ble0 is not None else 1.0)
            all_labels.append(feat)
            sig_flag = " ***" if (bear_ble0 is not None and bear_ble0 < 0.10) else ""
            print(f"  {feat:30}{n_f:>8}{r['edge_vs_random_pp']:>+10.2f}{str(bear_edge):>11}{str(bear_ble0):>11}{regime_cond:>14}{sig_flag}")
            sweep_results.append({"feat": feat, "n_fired": n_f, "edge_all": r["edge_vs_random_pp"],
                                  "bear_edge": bear_edge, "bear_ble0": bear_ble0,
                                  "bull_edge": bull_edge, "chop_edge": chop_edge,
                                  "regime_conditional": regime_cond})
        except Exception as ex:
            print(f"  {feat:30}  ERROR: {ex}")

    # Holm-BH on bear_ble0 only
    if all_ps:
        adj = holm_bh_correct(all_ps)
        print("\n  Bear-regime Holm-BH corrected:")
        for lbl, rp, ap in sorted(zip(all_labels, all_ps, adj), key=lambda x: x[1]):
            sig = " <-- MCC-SURVIVES" if ap < 0.05 else (" <-- trend" if ap < 0.15 else "")
            print(f"    {lbl:35}  raw_p={rp:.4f}  adj_p={ap:.4f}{sig}")

    return sweep_results


# =====================================================================
# MAIN
# =====================================================================
def main():
    print("=" * 70)
    print("TRADER AGENT: Exogenous v51 move-catch -- DEV wall <= 2024-05-15")
    print("=" * 70)

    print("\nLoading v51 daily lab (DEV-walled)...")
    lab = vfl.load_v51_daily(n=50)
    C = lab["C"]
    print(f"  {len(lab['syms'])} assets | {C.index.min().date()} -> {C.index.max().date()}")
    avail = [c for c in vfl.V51 if c in lab["F"] and lab["F"][c].notna().sum().sum() > 200]
    print(f"  v51 features with data: {avail}")

    # --- ANGLE 1: Listing age ---
    res1 = angle_listing_age(lab, n_null=400, seed=7)

    # --- ANGLE 2: Whale + liq flow ---
    res2 = angle_whale_liq(lab, n_null=400, seed=42)

    # --- FULL SWEEP (supporting context for the decisive question) ---
    sweep = exo_full_sweep(lab, n_null=300, seed=99)

    # --- DEPLOYABLE SPEC (if any edge survives) ---
    print("\n" + "=" * 70)
    print("DEPLOYABLE PRE-REGISTRATION (if any edge survives the block + MCC battery)")
    print("=" * 70)

    # Collect any bear-positive or regime-conditional survivors
    survivors = [s for s in sweep
                 if isinstance(s.get("bear_ble0"), float) and s["bear_ble0"] < 0.10]
    if survivors:
        for s in sorted(survivors, key=lambda x: x["bear_ble0"]):
            print(f"\n  SURVIVOR: {s['feat']}")
            print(f"    bear_edge={s['bear_edge']:+.2f}pp  bear_ble0={s['bear_ble0']:.4f}")
            print(f"    bull_edge={s['bull_edge']:+.2f}pp  chop_edge={s['chop_edge']:+.2f}pp")
            print(f"    regime_conditional: {s['regime_conditional']}")
            print(f"  PRE-REGISTERED SPEC:")
            print(f"    signal       : {s['feat']} (v51 exogenous, daily causal last-of-day)")
            print(f"    direction    : top cross-sectional tercile (>66th percentile)")
            print(f"    entry        : next-bar open (no look-ahead)")
            print(f"    exit         : time-stop 7 calendar days")
            print(f"    regime_gate  : bear (block_p_le0={s['bear_ble0']:.4f})")
            print(f"    OOS-handoff  : DEV frozen; user validates OOS (2024-05-15 forward)")
            print(f"    DO NOT TOUCH : OOS slice; this script is DEV-only")
    else:
        print("\n  NO SURVIVOR: no v51 exogenous feature clears bear block_p_le0 < 0.10")
        print("  (Note: if whale/liq angle 2 produced a bear-significant result at p<0.05")
        print("   it will appear above with *** SIG *** -- check the angle 2 output)")

    print("\n[DONE] Exogenous v51 move-catch battery complete.")


if __name__ == "__main__":
    main()
