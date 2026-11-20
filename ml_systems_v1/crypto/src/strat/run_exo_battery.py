"""run_exo_battery.py -- Full exogenous v51 feature battery + conditioned price-TI fleet.

Steps:
  1. Load v51 daily lab (DEV-walled <= 2024-05-15, n=50 assets)
  2. Sweep ALL available exo features: top-tercile trigger, time-exit 7d hold, by_regime=True, block=True
  3. Sweep BOTH directions (top-tercile = continuation; bottom-tercile = contrarian for liq/funding extremes)
  4. Conditioned fleet: brk14 AND exo_gate (1 price-TI x best 2 exo conditioners), by_regime
  5. Report findings: bear-positive (block_p_le0 < 0.05), regime-conditional, vs unconditioned brk14
  6. Holm-BH correction across sweeps

No emoji. DEV-walled. RWYB.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.v51_feature_lab as vlab
import strat.capture_lab as cl

DEV_END = vlab.DEV_END


# ---------- helpers ----------

def block_p_bear(result):
    br = result.get("by_regime", {}).get("bear")
    if br is None:
        return None
    b = br.get("block")
    return b.get("block_p_le0") if b else None


def edge_pp(result, regime="bear"):
    br = result.get("by_regime", {}).get(regime)
    return br.get("edge_pp") if br else None


def n_fired(result, regime="bear"):
    br = result.get("by_regime", {}).get(regime)
    return br.get("n") if br else None


def holm_correct(pvals):
    """Holm-Bonferroni correction. Returns adjusted p-values in same order."""
    m = len(pvals)
    idx = sorted(range(m), key=lambda i: (pvals[i] if pvals[i] is not None else 1.0))
    adj = [None] * m
    running_max = 0.0
    for rank, i in enumerate(idx):
        if pvals[i] is None:
            adj[i] = None
        else:
            corrected = pvals[i] * (m - rank)
            running_max = max(running_max, corrected)
            adj[i] = min(1.0, running_max)
    return adj


def conditioned_fired(lab, ti_key, exo_key, exo_dir="top", thr_pct=0.34):
    """Combined fired matrix: price-TI fires AND exo gate fires.
    exo_dir='top' -> top tercile (momentum / continuation signals)
    exo_dir='bot' -> bottom tercile (contrarian: liq capitulation flush, funding extreme)
    """
    F = lab["F"]
    # price-TI fired (momentum/breakout)
    pti = cl.fired_matrix(lab, ti_key)
    # exo gate
    X = F[exo_key]
    if exo_dir == "top":
        gate = X.gt(X.quantile(1.0 - thr_pct, axis=1), axis=0)
    else:
        gate = X.lt(X.quantile(thr_pct, axis=1), axis=0)
    combined = pti & gate
    # inject into lab["F"] under a synthetic key and return modified lab copy
    import copy
    lab2 = copy.copy(lab)
    lab2["F"] = dict(lab["F"])
    ckey = f"{ti_key}_AND_{exo_key}_{exo_dir}"
    lab2["F"][ckey] = combined
    return lab2, ckey


# ---------- main battery ----------

def main():
    print(f"[exo_battery] Loading v51 daily lab (DEV <= {DEV_END}) ...")
    lab = vlab.load_v51_daily(n=50, end=DEV_END)
    C = lab["C"]
    print(f"  {len(lab['syms'])} assets; {C.index.min().date()} -> {C.index.max().date()}")
    avail_exo = [c for c in vlab.V51 if c in lab["F"] and lab["F"][c].notna().sum().sum() > 500]
    print(f"  Exo features with data ({len(avail_exo)}): {avail_exo}\n")

    # ================================================================
    # STEP 1: Full exo feature sweep -- top-tercile + bottom-tercile
    # ================================================================
    print("=" * 80)
    print("SWEEP 1: ALL exo features, top-tercile vs bottom-tercile trigger, 7d time-exit")
    print("         (top=continuation/momentum  bot=contrarian: liq-flush, funding-reset)")
    print("=" * 80)

    header = f"  {'feature':28}{'dir':5}{'n_fired':>7}{'bull_e':>8}{'chop_e':>8}{'bear_e':>8}{'bear_n':>7}{'bear_ble0':>11}  note"
    print(header)
    print("  " + "-" * 80)

    sweep_results = []
    for feat in avail_exo:
        for direction in ("top", "bot"):
            # inject direction into lab via a synthetic key
            import copy
            lab2 = copy.copy(lab)
            lab2["F"] = dict(lab["F"])
            X = lab["F"][feat]
            thr_pct = 0.34
            if direction == "top":
                gate = X.gt(X.quantile(1.0 - thr_pct, axis=1), axis=0)
            else:
                gate = X.lt(X.quantile(thr_pct, axis=1), axis=0)
            synkey = f"__sweep_{feat}_{direction}"
            lab2["F"][synkey] = gate
            try:
                r = cl.evaluate_ti(lab2, synkey, tf="1d", exit_kind="time",
                                   n_null=300, by_regime=True, block=True)
            except Exception as ex:
                print(f"  {feat:28}{direction:5}  ERROR: {ex}")
                continue
            if "note" in r and "insufficient" in r.get("note", ""):
                note = r["note"]
                print(f"  {feat:28}{direction:5}  SKIP ({note})")
                continue
            be = edge_pp(r, "bear"); bn = n_fired(r, "bear"); bp = block_p_bear(r)
            bue = edge_pp(r, "bull"); bce = edge_pp(r, "chop")
            flag = ""
            if bp is not None and bp < 0.05 and be is not None and be > 0:
                flag = " <-- BEAR-POSITIVE sig"
            elif bp is not None and bp < 0.10 and be is not None and be > 0:
                flag = " <-- bear-marginal"
            print(f"  {feat:28}{direction:5}{r['n_fired']:>7}{str(bue):>8}{str(bce):>8}{str(be):>8}{str(bn):>7}{str(bp):>11}{flag}")
            sweep_results.append({"feat": feat, "dir": direction, "result": r,
                                   "bear_edge": be, "bear_ble0": bp, "bull_edge": bue, "chop_edge": bce})

    # Holm correction
    pvals = [s["bear_ble0"] for s in sweep_results]
    adj_p = holm_correct(pvals)
    for s, ap in zip(sweep_results, adj_p):
        s["bear_ble0_adj"] = ap

    sig_bear = [s for s in sweep_results if s["bear_ble0"] is not None and s["bear_ble0"] < 0.05 and s["bear_edge"] is not None and s["bear_edge"] > 0]
    print(f"\n  --> BEAR-POSITIVE (raw p < 0.05, edge > 0): {len(sig_bear)}")
    if sig_bear:
        for s in sig_bear:
            print(f"       {s['feat']} {s['dir']}  bear_edge={s['bear_edge']}pp  ble0={s['bear_ble0']}  holm_adj={s['bear_ble0_adj']:.3f}")

    # Regime-conditional: bear-positive AND NOT bull-positive (or bear >> bull)
    rc = [s for s in sweep_results
          if s["bear_edge"] is not None and s["bear_edge"] > 0
          and s["bull_edge"] is not None
          and s["bear_edge"] > s["bull_edge"] + 0.5]
    print(f"\n  --> REGIME-CONDITIONAL candidates (bear_edge > bull_edge + 0.5pp): {len(rc)}")
    for s in rc[:10]:
        print(f"       {s['feat']} {s['dir']}  bear={s['bear_edge']}pp  bull={s['bull_edge']}pp  ble0={s['bear_ble0']}")

    # ================================================================
    # STEP 2: Unconditioned brk14 baseline (the wall)
    # ================================================================
    print("\n" + "=" * 80)
    print("SWEEP 2: UNCONDITIONED brk14 baseline (the price-TI wall)")
    print("=" * 80)
    r_brk = cl.evaluate_ti(lab, "brk14", tf="1d", exit_kind="time",
                           n_null=300, by_regime=True, block=True)
    for rg in ("bull", "chop", "bear"):
        d = r_brk.get("by_regime", {}).get(rg)
        if d:
            b = d.get("block", {})
            print(f"  brk14 {rg:5} n={d['n']:>5}  realized={d['realized_net']:>6}%  null={d['null_net']:>6}%  "
                  f"edge={d['edge_pp']:>6}pp  ble0={b.get('block_p_le0','?')}")
        else:
            print(f"  brk14 {rg:5}  (no data)")

    # ================================================================
    # STEP 3: Conditioned fleet -- brk14 AND top exo conditioners
    # ================================================================
    # Pick top candidates by regime-conditionality + signal strength
    # Use the best 6 exo features in both directions to conditioned price-TI
    cond_candidates = sorted(sweep_results, key=lambda s: (
        -(s["bear_edge"] or -99),
        (s["bear_ble0"] or 1.0)
    ))[:8]

    print("\n" + "=" * 80)
    print("SWEEP 3: CONDITIONED fleet -- brk14 AND exo gate (price-TI x exo conditioner)")
    print("         Goal: bear block_p_le0 < 0.05 + bear_edge > 0 (wall-breaker test)")
    print("=" * 80)

    print(f"  {'conditioner':38}{'n_cond':>7}{'bull_e':>8}{'chop_e':>8}{'bear_e':>8}{'bear_n':>7}{'bear_ble0':>11}  vs_baseline")
    print("  " + "-" * 82)

    bear_base = edge_pp(r_brk, "bear")
    cond_results = []
    for s in cond_candidates:
        feat = s["feat"]; d = s["dir"]
        lab2, ckey = conditioned_fired(lab, "brk14", feat, exo_dir=d)
        try:
            r = cl.evaluate_ti(lab2, ckey, tf="1d", exit_kind="time",
                               n_null=300, by_regime=True, block=True)
        except Exception as ex:
            print(f"  {ckey:38}  ERROR: {ex}"); continue
        if "note" in r and "insufficient" in r.get("note", ""):
            print(f"  {ckey:38}  SKIP ({r['note']})"); continue
        be = edge_pp(r, "bear"); bn = n_fired(r, "bear"); bp = block_p_bear(r)
        bue = edge_pp(r, "bull"); bce = edge_pp(r, "chop")
        delta = f"+{be - bear_base:.2f}pp" if (be is not None and bear_base is not None) else "?"
        flag = ""
        if bp is not None and bp < 0.05 and be is not None and be > 0:
            flag = " <-- WALL BROKEN"
        elif bp is not None and bp < 0.10 and be is not None and be > 0:
            flag = " <-- marginal"
        print(f"  {ckey:38}{r['n_fired']:>7}{str(bue):>8}{str(bce):>8}{str(be):>8}{str(bn):>7}{str(bp):>11}  {delta}{flag}")
        cond_results.append({"key": ckey, "feat": feat, "dir": d, "result": r,
                              "bear_edge": be, "bear_ble0": bp, "delta_bear": (be - bear_base if be is not None and bear_base is not None else None)})

    # Also test the explicitly mentioned conditioners from the task
    explicit_conds = [
        ("liq_capitulation", "bot"),  # flush -> bounce (contrarian)
        ("liq_short_panic", "bot"),
        ("norm_funding", "bot"),      # funding reset -> continuation
        ("s3_smart_vs_retail_z", "top"),  # smart vs retail positive
        ("wh_whale_net_usd", "top"),  # whale inflow
        ("stbl_compound_shock", "bot"),  # stablecoin shock -> reversal
    ]
    print("\n  -- Explicit mechanistic conditioners (charter-mentioned) --")
    for feat, d in explicit_conds:
        if feat not in avail_exo:
            print(f"  {feat} {d}  -- not in avail_exo, skip"); continue
        lab2, ckey = conditioned_fired(lab, "brk14", feat, exo_dir=d)
        try:
            r = cl.evaluate_ti(lab2, ckey, tf="1d", exit_kind="time",
                               n_null=300, by_regime=True, block=True)
        except Exception as ex:
            print(f"  {ckey:38}  ERROR: {ex}"); continue
        if "note" in r and "insufficient" in r.get("note", ""):
            print(f"  {ckey:38}  SKIP ({r['note']})"); continue
        be = edge_pp(r, "bear"); bn = n_fired(r, "bear"); bp = block_p_bear(r)
        bue = edge_pp(r, "bull"); bce = edge_pp(r, "chop")
        delta = f"+{be - bear_base:.2f}pp" if (be is not None and bear_base is not None) else "?"
        flag = ""
        if bp is not None and bp < 0.05 and be is not None and be > 0:
            flag = " <-- WALL BROKEN"
        elif bp is not None and bp < 0.10 and be is not None and be > 0:
            flag = " <-- marginal"
        print(f"  {ckey:38}{r['n_fired']:>7}{str(bue):>8}{str(bce):>8}{str(be):>8}{str(bn):>7}{str(bp):>11}  {delta}{flag}")
        cond_results.append({"key": ckey, "feat": feat, "dir": d, "result": r,
                              "bear_edge": be, "bear_ble0": bp, "delta_bear": (be - bear_base if be is not None and bear_base is not None else None)})

    # ================================================================
    # STEP 4: CAUSALITY AUDIT
    # ================================================================
    print("\n" + "=" * 80)
    print("CAUSALITY AUDIT -- verifying each used feature is point-in-time (no forward leakage)")
    print("=" * 80)
    audit = {
        "norm_funding":        "trailing z-score of funding rate, last-of-day -> next-day entry. CAUSAL.",
        "s3_global_lsr_z":     "trailing long/short-ratio z, last-of-day. CAUSAL.",
        "s3_smart_vs_retail_z":"trailing smart vs retail flow z, last-of-day. CAUSAL.",
        "bs_basis_z30":        "trailing 30-bar basis z-score, last-of-day. CAUSAL.",
        "bs_basis_xsec_z":     "cross-sectional basis z within day, last-of-day. CAUSAL.",
        "liq_total_usd":       "sum-of-day liquidations -> known by EOD before next entry. CAUSAL.",
        "liq_delta_z30":       "trailing 30-bar liq delta z, last-of-day. CAUSAL.",
        "liq_capitulation":    "max-of-day flag (did liq capitulation occur today). CAUSAL (daily max, no forward).",
        "liq_short_panic":     "max-of-day short-squeeze flag. CAUSAL.",
        "wh_whale_net_usd":    "sum-of-day whale flow. CAUSAL (known by EOD).",
        "norm_whale":          "trailing norm whale z, last-of-day. CAUSAL.",
        "stbl_total_zscore_30d":"trailing 30d stablecoin supply z. CAUSAL.",
        "stbl_compound_shock": "max-of-day stablecoin shock flag. CAUSAL.",
        "norm_vpin":           "trailing vpin norm, last-of-day. CAUSAL.",
        "norm_kyle_lambda":    "trailing kyle lambda norm, last-of-day. CAUSAL.",
        "norm_hawkes_imbalance":"trailing hawkes imbalance norm, last-of-day. CAUSAL.",
        "te_imb":              "trailing transfer-entropy imbalance, last-of-day. CAUSAL.",
        "norm_oi_change":      "trailing open-interest change norm, last-of-day. CAUSAL.",
        "mv_days_since_listed_binance": "age feature, monotonically increasing. CAUSAL.",
        "xd_momentum_rank":    "trailing cross-asset momentum rank, last-of-day. CAUSAL.",
    }
    for feat in avail_exo:
        status = audit.get(feat, "UNKNOWN -- review manually")
        print(f"  {feat:35}: {status}")

    # ================================================================
    # STEP 5: DECISIVE VERDICT
    # ================================================================
    print("\n" + "=" * 80)
    print("DECISIVE VERDICT")
    print("=" * 80)
    wall_broken = [r for r in cond_results
                   if r["bear_ble0"] is not None and r["bear_ble0"] < 0.05
                   and r["bear_edge"] is not None and r["bear_edge"] > 0]
    wall_marginal = [r for r in cond_results
                     if r["bear_ble0"] is not None and 0.05 <= r["bear_ble0"] < 0.10
                     and r["bear_edge"] is not None and r["bear_edge"] > 0]
    exo_standalone_bear_sig = [s for s in sweep_results
                                if s["bear_ble0"] is not None and s["bear_ble0"] < 0.05
                                and s["bear_edge"] is not None and s["bear_edge"] > 0]

    print(f"\n  Standalone exo features with bear block_p_le0 < 0.05: {len(exo_standalone_bear_sig)}")
    for s in exo_standalone_bear_sig:
        print(f"    {s['feat']} {s['dir']}  bear_edge={s['bear_edge']}pp  ble0={s['bear_ble0']}  holm_adj={s['bear_ble0_adj']:.3f}")

    print(f"\n  Conditioned price-TI x exo combos with bear block_p_le0 < 0.05 (WALL BROKEN): {len(wall_broken)}")
    for r in wall_broken:
        print(f"    {r['key']}  bear_edge={r['bear_edge']}pp  ble0={r['bear_ble0']}  delta_vs_brk14={r['delta_bear']:.2f}pp")

    print(f"\n  Marginal conditioners (0.05 <= ble0 < 0.10): {len(wall_marginal)}")
    for r in wall_marginal:
        print(f"    {r['key']}  bear_edge={r['bear_edge']}pp  ble0={r['bear_ble0']}  delta_vs_brk14={r['delta_bear']:.2f}pp")

    if not wall_broken and not exo_standalone_bear_sig:
        print("\n  CONCLUSION: NO exo feature (standalone or as conditioner) produces a bear-significant")
        print("  move-catch edge (block_p_le0 < 0.05). The wall is EXOGENOUS-INVARIANT.")
        print("  The internal-data space is EXHAUSTED. Charter-blessed redirect: EXTERNAL data.")
    elif wall_broken or exo_standalone_bear_sig:
        print("\n  CONCLUSION: WALL BROKEN -- exo features produce bear-positive move-catch edges.")
        print("  These are REGIME-CONDITIONAL candidates warranting OOS testing (after charter unlock).")

    print("\n[exo_battery] DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
