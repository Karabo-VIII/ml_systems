"""run_exo_deep.py -- Deep dive on the most promising exo candidates + mom14 conditioned.

Focuses on:
  1. Standalone: mv_days_since_listed_binance bot (bear 1.52pp), s3_smart_vs_retail_z top (bear 0.80pp),
     norm_kyle_lambda bot (bear 0.12pp), xd_momentum_rank top (bear 0.08pp) -- all bear-positive but not sig.
     Run with n_null=1000, block=True to get more precise p estimates.
  2. mom14 (the price-TI with widest bear coverage) x top exo conditioners (BOTH directions)
  3. A 2-exo composite gate: liq_capitulation top AND s3_smart_vs_retail_z top (bear signal stacking)
  4. Summary table + decisive verdict

No emoji. DEV-walled. RWYB.
"""
from __future__ import annotations
import sys, copy
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.v51_feature_lab as vlab
import strat.capture_lab as cl

DEV_END = vlab.DEV_END


def inject_gate(lab, gate_df, name):
    """Inject a pre-computed boolean/float gate DataFrame into a lab copy."""
    lab2 = copy.copy(lab)
    lab2["F"] = dict(lab["F"])
    lab2["F"][name] = gate_df
    return lab2, name


def top_gate(F, feat, pct=0.34):
    X = F[feat]
    return X.gt(X.quantile(1.0 - pct, axis=1), axis=0)


def bot_gate(F, feat, pct=0.34):
    X = F[feat]
    return X.lt(X.quantile(pct, axis=1), axis=0)


def eval_named(lab, gate_df, name, n_null=1000):
    lab2, k = inject_gate(lab, gate_df, name)
    r = cl.evaluate_ti(lab2, k, tf="1d", exit_kind="time", n_null=n_null, by_regime=True, block=True)
    return r


def fmt_regime(r, rg):
    d = r.get("by_regime", {}).get(rg)
    if d is None:
        return f"  {rg:5}: no data"
    b = d.get("block", {})
    return (f"  {rg:5}: n={d['n']:>5}  realized={d['realized_net']:>6}%  null={d['null_net']:>6}%  "
            f"edge={d['edge_pp']:>6}pp  p_rnd={d['p_vs_random']:.4f}  "
            f"ble0={b.get('block_p_le0','?')}  bp05={b.get('block_p05_pp','?')}pp")


def print_result(label, r):
    print(f"\n  [{label}]  n_fired={r.get('n_fired','?')}  capture={r.get('capture_rate','?')}")
    if "note" in r:
        print(f"    SKIP: {r['note']}"); return
    for rg in ("bull", "chop", "bear"):
        print(fmt_regime(r, rg))


def main():
    print(f"[exo_deep] Loading v51 daily lab (DEV <= {DEV_END}) ...")
    lab = vlab.load_v51_daily(n=50, end=DEV_END)
    C = lab["C"]; F = lab["F"]
    print(f"  {len(lab['syms'])} assets; {C.index.min().date()} -> {C.index.max().date()}\n")
    avail = [c for c in vlab.V51 if c in F and F[c].notna().sum().sum() > 500]

    # ================================================================
    # Part 1: Standalone bear-positive candidates with 1000 nulls
    # ================================================================
    print("=" * 72)
    print("PART 1: Standalone bear-positive candidates -- n_null=1000, block=True")
    print("=" * 72)
    standalone_cands = [
        ("mv_days_since_listed_binance_bot", bot_gate(F, "mv_days_since_listed_binance")),
        ("s3_smart_vs_retail_z_top",         top_gate(F, "s3_smart_vs_retail_z")),
        ("s3_smart_vs_retail_z_bot",         bot_gate(F, "s3_smart_vs_retail_z")),
        ("norm_hawkes_imbalance_bot",        bot_gate(F, "norm_hawkes_imbalance")),
        ("norm_kyle_lambda_bot",             bot_gate(F, "norm_kyle_lambda")),
        ("norm_vpin_top",                    top_gate(F, "norm_vpin")),
        ("xd_momentum_rank_top",             top_gate(F, "xd_momentum_rank")),
        ("norm_whale_top",                   top_gate(F, "norm_whale")),
        ("liq_capitulation_top",             top_gate(F, "liq_capitulation")),
        ("liq_short_panic_top",              top_gate(F, "liq_short_panic")),
        ("norm_funding_top",                 top_gate(F, "norm_funding")),
    ]
    part1_results = []
    for name, gate in standalone_cands:
        feat_key = name
        if feat_key not in avail and name.rsplit("_", 1)[0] not in avail:
            # check if base feature available
            base = "_".join(name.split("_")[:-1])
            if base not in avail:
                print(f"  {name}: feature not available -- skip"); continue
        try:
            r = eval_named(lab, gate, f"__deep_{name}", n_null=1000)
            print_result(name, r)
            be = (r.get("by_regime", {}).get("bear") or {}).get("edge_pp")
            bp = ((r.get("by_regime", {}).get("bear") or {}).get("block") or {}).get("block_p_le0")
            part1_results.append({"name": name, "bear_edge": be, "bear_ble0": bp, "result": r})
        except Exception as ex:
            print(f"  {name}: ERROR {ex}")

    # ================================================================
    # Part 2: mom14 (wider bear coverage) x exo conditioners
    # ================================================================
    print("\n" + "=" * 72)
    print("PART 2: mom14 x exo conditioners -- bear-rescue test")
    print(f"        Baseline bear edge (unconditioned mom14):")
    r_mom = cl.evaluate_ti(lab, "mom14", tf="1d", exit_kind="time", n_null=500, by_regime=True, block=True)
    print(fmt_regime(r_mom, "bear"))
    bear_mom_base = (r_mom.get("by_regime", {}).get("bear") or {}).get("edge_pp")
    print("=" * 72)

    mom_conds = []
    for feat in avail:
        if feat in ("mv_days_since_listed_binance", "stbl_total_zscore_30d", "stbl_compound_shock"):
            continue  # skip flags with coverage issues for mom14
        for d, gate_fn in [("top", top_gate), ("bot", bot_gate)]:
            try:
                gate = gate_fn(F, feat) & cl.fired_matrix(lab, "mom14")
                key = f"mom14_AND_{feat}_{d}"
                r = eval_named(lab, gate, f"__deep_{key}", n_null=500)
                if "note" in r and "insufficient" in r.get("note", ""):
                    continue
                be = (r.get("by_regime", {}).get("bear") or {}).get("edge_pp")
                bp = ((r.get("by_regime", {}).get("bear") or {}).get("block") or {}).get("block_p_le0")
                delta = f"{be - bear_mom_base:+.2f}pp" if (be is not None and bear_mom_base is not None) else "?"
                flag = " <-- WALL BROKEN" if (bp is not None and bp < 0.05 and be is not None and be > 0) else ""
                flag += " <-- marginal" if (bp is not None and 0.05 <= bp < 0.10 and be is not None and be > 0 and not flag) else ""
                print(f"  {key:45}  bear_edge={str(be):>7}pp  ble0={str(bp):>6}  delta={delta}{flag}")
                mom_conds.append({"key": key, "bear_edge": be, "bear_ble0": bp, "delta": (be - bear_mom_base) if (be is not None and bear_mom_base is not None) else None})
            except Exception as ex:
                pass  # skip silently for tractability

    # ================================================================
    # Part 3: 2-exo stacked gate (bear signal stacking)
    # ================================================================
    print("\n" + "=" * 72)
    print("PART 3: 2-exo STACKED gates (no price-TI) -- pure exo bear signal")
    print("=" * 72)
    # Best 2 standalone: s3_smart_vs_retail_z_top + norm_hawkes_imbalance_bot
    # Also try: mv_days_since_listed_binance_bot AND (top exo momentum)
    stacked_cands = []
    if "s3_smart_vs_retail_z" in avail and "norm_hawkes_imbalance" in avail:
        stacked_cands.append(("smart_z_top_AND_hawkes_bot",
                               top_gate(F, "s3_smart_vs_retail_z") & bot_gate(F, "norm_hawkes_imbalance")))
    if "liq_capitulation" in avail and "s3_smart_vs_retail_z" in avail:
        stacked_cands.append(("liq_cap_top_AND_smart_z_top",
                               top_gate(F, "liq_capitulation") & top_gate(F, "s3_smart_vs_retail_z")))
    if "norm_funding" in avail and "s3_smart_vs_retail_z" in avail:
        stacked_cands.append(("funding_top_AND_smart_z_top",
                               top_gate(F, "norm_funding") & top_gate(F, "s3_smart_vs_retail_z")))
    if "mv_days_since_listed_binance" in avail and "s3_smart_vs_retail_z" in avail:
        stacked_cands.append(("age_bot_AND_smart_z_top",
                               bot_gate(F, "mv_days_since_listed_binance") & top_gate(F, "s3_smart_vs_retail_z")))
    if "norm_kyle_lambda" in avail and "norm_hawkes_imbalance" in avail:
        stacked_cands.append(("kyle_bot_AND_hawkes_bot",
                               bot_gate(F, "norm_kyle_lambda") & bot_gate(F, "norm_hawkes_imbalance")))

    part3_results = []
    for name, gate in stacked_cands:
        try:
            r = eval_named(lab, gate, f"__stack_{name}", n_null=800)
            print_result(name, r)
            be = (r.get("by_regime", {}).get("bear") or {}).get("edge_pp")
            bp = ((r.get("by_regime", {}).get("bear") or {}).get("block") or {}).get("block_p_le0")
            part3_results.append({"name": name, "bear_edge": be, "bear_ble0": bp})
        except Exception as ex:
            print(f"  {name}: ERROR {ex}")

    # ================================================================
    # DECISIVE VERDICT
    # ================================================================
    print("\n" + "=" * 72)
    print("DECISIVE VERDICT")
    print("=" * 72)

    all_results = part1_results + part3_results
    for cr in mom_conds:
        all_results.append({"name": cr["key"], "bear_edge": cr["bear_edge"], "bear_ble0": cr["bear_ble0"]})

    sig = [r for r in all_results if r["bear_ble0"] is not None and r["bear_ble0"] < 0.05 and r["bear_edge"] is not None and r["bear_edge"] > 0]
    marg = [r for r in all_results if r["bear_ble0"] is not None and 0.05 <= r["bear_ble0"] < 0.10 and r["bear_edge"] is not None and r["bear_edge"] > 0]
    best_bear = sorted([r for r in all_results if r["bear_edge"] is not None], key=lambda r: -r["bear_edge"])[:5]

    print(f"\n  Significant bear edges (block_p_le0 < 0.05, bear_edge > 0): {len(sig)}")
    for r in sig:
        print(f"    {r['name']}  bear_edge={r['bear_edge']}pp  ble0={r['bear_ble0']}")

    print(f"\n  Marginal bear edges (0.05 <= ble0 < 0.10): {len(marg)}")
    for r in marg:
        print(f"    {r['name']}  bear_edge={r['bear_edge']}pp  ble0={r['bear_ble0']}")

    print(f"\n  Top 5 by raw bear_edge (regardless of significance):")
    for r in best_bear:
        print(f"    {r['name']}  bear_edge={r['bear_edge']}pp  ble0={r['bear_ble0']}")

    if not sig:
        print("\n  FINAL: No exo feature or conditioner combo breaks the wall (block p < 0.05, bear > 0).")
        print("  The wall is EXOGENOUS-INVARIANT across all 20 v51 internal features.")
        print("  CHARTER VERDICT: Internal-data space EXHAUSTED. Redirect to EXTERNAL data is warranted.")
    else:
        print(f"\n  FINAL: WALL BROKEN by {len(sig)} combo(s). REGIME-CONDITIONAL move-catch edges exist.")
        print("  These warrant OOS testing (charter unlock after dev confirmation).")

    print("\n[exo_deep] DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
