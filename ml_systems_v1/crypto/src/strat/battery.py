"""src/strat/battery.py -- the reusable robustness BATTERY (Lens A/B/C), the foundation's
validation GATE. Consolidated on the kept CanonicalHarness; numpy-only + self-contained.

PROVENANCE: ported 2026-06-05 from runs/staging/battery_rebuilt_2026_06_04.py (the discover/trader
skills reference src/strat/battery.py; the path was archived in the 2026-06-04 reset). This is the
turnkey home. Hardened against the 2026-06-05 apparatus red-audit (see docs/APPARATUS_AUDIT_2026_06_05.md):
  - F6 (HIGH, FIXED): block-bootstrap start-index off-by-one excluded the last observation from every
    resample. Fixed `sp = a.size - block + 1` so the final block can include the last element.
  - F7 (MEDIUM, CLARIFIED): the `(p05 or -1) > 0` guard was a confusing Python truthiness pitfall;
    rewritten as an explicit `p05 is not None and p05 > 0`.

FOUNDATION infrastructure: every future candidate (any avenue in docs/AVENUE_MAP_2026_06_04.md) is
judged here. Upstream gates (cost lens / leak probe / firewall) live in sibling modules:
  - cost lens   -> src/strat/fill_model.py
  - leak probe  -> src/wealth_bot/leak_probe.py (relative_leak_test = the cadence-robust verdict)
  - firewall    -> src/strat/firewall.py
  - integrated  -> src/strat/candidate_gate.py (chains all of the above + this battery)

Lens A (strict/institutional): all-4-positive AND n>=15 AND n_eff>=15 AND jk2>0 AND jk3>0 AND p05>0
  AND maxDD<30%  [+ DSR/Holm @ true family-N supplied by caller].
Lens B (pragmatic 'money-in-bank'): all-4-positive AND UNSEEN>0 AND jk2>0 AND jk3>0 AND maxDD<30%
  AND n_eff>=8 (concentration floor -- a 1-2-trade-dominated book is NOT bankable).
Lens C (temporal barometer): UNSEEN>0 AND n_months>=3 AND monthly-positive>=0.60 AND worst_month>-10%.

NOTE on maxDD: this module gates at 30% (the published ceiling). The PROJECT binding floor is 20%
(CLAUDE.md). The candidate_gate / solving phase applies the stricter 20% bar; 30% here is the
loosest admissible ceiling for the diagnostic verdict.
"""
from __future__ import annotations

import numpy as np


# ---- primitives ----------------------------------------------------------
def compound(rets) -> float:
    a = np.asarray(rets, float)
    return float((np.prod(1.0 + a) - 1.0) * 100) if a.size else 0.0


def jackknife(rets, k: int) -> float:
    """Compound % after removing the k largest-|return| trades (overfit cap)."""
    a = np.asarray(rets, float)
    if a.size <= k:
        return 0.0
    drop = np.argsort(np.abs(a))[-k:] if k > 0 else np.array([], dtype=int)
    keep = np.delete(a, drop)
    return compound(keep)


def herfindahl_neff(rets) -> float:
    a = np.abs(np.asarray(rets, float))
    s = a.sum()
    return float(s * s / np.sum(a * a)) if s > 0 else float(len(a))


def block_bootstrap_p05_p95(rets, block: int = 5, n: int = 2000, seed: int = 7) -> dict:
    """Stationary block-bootstrap of the compound %. F6 FIX (2026-06-05): the start index upper bound
    is `a.size - block + 1` so `rng.integers(0, sp)` can yield `a.size - block`, letting the last block
    include the final observation. The previous `sp = a.size - block` silently dropped the last element
    from every resample (small but systematic bias on all percentiles)."""
    a = np.asarray(rets, float)
    if a.size < block * 2:
        return {"p05": None, "p50": None, "p95": None}
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(a.size / block))
    sp = a.size - block + 1  # F6: +1 so the final block (start = a.size-block) is reachable
    cs = [(np.prod(1.0 + np.concatenate([a[st:st + block] for st in rng.integers(0, sp, size=nb)])[:a.size]) - 1.0) * 100
          for _ in range(n)]
    c = np.array(cs)
    return {"p05": round(float(np.percentile(c, 5)), 2), "p50": round(float(np.percentile(c, 50)), 2),
            "p95": round(float(np.percentile(c, 95)), 2)}


def expectancy(rets) -> float:
    a = np.asarray(rets, float)
    return float(a.mean() * 100) if a.size else 0.0


def win_rate(rets) -> float:
    a = np.asarray(rets, float)
    return float((a > 0).mean()) if a.size else 0.0


def profit_factor(rets) -> float:
    a = np.asarray(rets, float)
    g = a[a > 0].sum(); l = -a[a < 0].sum()
    return float(g / l) if l > 0 else float("inf") if g > 0 else 0.0


def monthly(entry_pnl_pairs) -> dict:
    """entry_pnl_pairs: list of (entry_timestamp-like, net_pnl). Returns monthly-positive rate + worst month."""
    import pandas as pd
    if not entry_pnl_pairs:
        return {"n_months": 0, "mpos": 0.0, "worst_month_pct": 0.0}
    df = pd.DataFrame(entry_pnl_pairs, columns=["ts", "pnl"])
    df["ts"] = pd.to_datetime(df["ts"])
    g = df.groupby(df["ts"].dt.to_period("M"))["pnl"].apply(lambda s: (np.prod(1.0 + s.to_numpy()) - 1.0) * 100)
    return {"n_months": int(len(g)), "mpos": float((g > 0).mean()), "worst_month_pct": float(g.min())}


# ---- the gate ------------------------------------------------------------
def evaluate(unseen_returns, comps: dict, unseen_maxdd_pct: float,
             entry_pnl_pairs=None, family_n=None, all_4_positive=None) -> dict:
    """Lens A/B/C verdict on an UNSEEN-touched-once candidate. comps = {TRAIN,VAL,OOS,UNSEEN} compound %.
    unseen_maxdd_pct as a NEGATIVE %. Caller supplies family_n for the DSR note + the random-entry-null verdict.

    NOTE: `n` and `n_eff` are UNSEEN-trade counts (the held-out gate intent). A strategy with many TRAIN
    trades but <15 UNSEEN trades fails Lens A on n alone -- that is intentional held-out strictness, not a bug."""
    r = np.asarray(unseen_returns, float)
    n = int(r.size)
    neff = herfindahl_neff(r)
    jk2, jk3 = jackknife(r, 2), jackknife(r, 3)
    bb = block_bootstrap_p05_p95(r)
    p05 = bb["p05"]
    p05_ok = (p05 is not None and p05 > 0)  # F7: explicit, no truthiness pitfall
    if all_4_positive is None:
        all_4_positive = all(comps.get(w, 0) > 0 for w in ("TRAIN", "VAL", "OOS", "UNSEEN"))
    dd_ok = unseen_maxdd_pct > -30.0
    mon = monthly(entry_pnl_pairs) if entry_pnl_pairs else {"n_months": 0, "mpos": 0.0, "worst_month_pct": 0.0}

    # n_eff floor on Lens B too (2026-06-05 self-test fix): jackknife drops by COUNT, not by mass, so a
    # 1-2-trade-dominated book can pass jk3>0 yet have n_eff~1 -> dangerous for real capital. Guard it.
    concentration_flag = neff < 8.0
    lens_A = bool(all_4_positive and n >= 15 and neff >= 15 and jk2 > 0 and jk3 > 0 and p05_ok and dd_ok)
    lens_B = bool(all_4_positive and comps.get("UNSEEN", 0) > 0 and jk2 > 0 and jk3 > 0 and dd_ok and neff >= 8.0)
    lens_C = bool(comps.get("UNSEEN", 0) > 0 and mon["n_months"] >= 3 and mon["mpos"] >= 0.60 and mon["worst_month_pct"] > -10.0)

    return {
        "n": n, "n_eff": round(neff, 1), "jk2": round(jk2, 1), "jk3": round(jk3, 1),
        "p05": p05, "p50": bb["p50"], "p95": bb["p95"], "unseen_maxdd_pct": round(unseen_maxdd_pct, 1),
        "all_4_positive": bool(all_4_positive), "monthly": mon, "concentration_flag": bool(concentration_flag),
        "lens_A_strict": lens_A, "lens_B_pragmatic": lens_B, "lens_C_temporal": lens_C,
        "dsr_note": f"caller must supply DSR/Holm at true family_n={family_n}" if family_n else "DSR/Holm @ family-N NOT supplied",
        "firewall_note": "must ALSO beat the cost-matched random-entry null on held-out (src/strat/firewall.py)",
        "verdict": "SHIP-TIER (Lens A)" if lens_A else "PRAGMATIC (Lens B)" if lens_B else "PROVISIONAL (Lens C)" if lens_C else "FAIL",
    }


def evaluate_setup_chaser(unseen_returns, comps: dict, unseen_maxdd_pct: float,
                          flat_benchmark_mean=None, pf_min: float = 1.3, dd_max: float = 30.0) -> dict:
    """Gate for a SELECTIVE setup-chaser (few big trades): positive expectancy + PF + UNSEEN>0 + DD +
    SELECTIVITY (beats flat exposure). jk/n_eff are DIAGNOSTICS here (few-big-trades is the design).

    D13 DISCIPLINE (dead-list): setup SELECTION must be done on TRAIN+VAL only; UNSEEN is touched once
    for the verdict. NEVER gate setup selection on UNSEEN>0 (the archived chaser's selection leak)."""
    r = np.asarray(unseen_returns, float)
    exp = expectancy(r); pf = profit_factor(r); wr = win_rate(r)
    selective = (flat_benchmark_mean is None) or (r.mean() > flat_benchmark_mean)
    gate = bool(exp > 0 and pf >= pf_min and comps.get("UNSEEN", 0) > 0 and unseen_maxdd_pct > -dd_max and selective)
    return {"expectancy_pct": round(exp, 3), "profit_factor": round(pf, 2), "win_rate": round(wr, 2),
            "n_eff_diag": round(herfindahl_neff(r), 1), "jk3_diag": round(jackknife(r, 3), 1),
            "selective_vs_flat": bool(selective), "gate_pass": gate}


# ---- RWYB self-test (synthetic; validates the gate logic without data) ----
def _selftest():
    rng = np.random.default_rng(0)
    print("[battery selftest]")
    # (a) a robust positive book: many small-positive trades, all windows positive
    good = rng.normal(0.01, 0.02, 40)
    res = evaluate(good, {"TRAIN": 20, "VAL": 8, "OOS": 6, "UNSEEN": 9}, -12.0, family_n=100)
    print(f"  GOOD  -> verdict={res['verdict']}  n_eff={res['n_eff']} jk3={res['jk3']} p05={res['p05']} lensA={res['lens_A_strict']}")
    # (b) a concentration ghost: one huge trade carries it (low n_eff, jk3 collapses)
    ghost = np.concatenate([rng.normal(-0.005, 0.01, 12), [0.9]])
    res2 = evaluate(ghost, {"TRAIN": 50, "VAL": 1, "OOS": 1, "UNSEEN": 80}, -5.0, family_n=6000)
    print(f"  GHOST -> verdict={res2['verdict']}  n_eff={res2['n_eff']} jk3={res2['jk3']} conc_flag={res2['concentration_flag']} (should FAIL)")
    # (c) setup-chaser: few big trades, positive expectancy
    sc = np.array([0.15, -0.04, 0.22, -0.03, 0.18, -0.05])
    res3 = evaluate_setup_chaser(sc, {"UNSEEN": 50}, -8.0, flat_benchmark_mean=0.0)
    print(f"  CHASER-> gate_pass={res3['gate_pass']}  exp={res3['expectancy_pct']}% PF={res3['profit_factor']} (n_eff diag={res3['n_eff_diag']})")
    assert res["lens_B_pragmatic"] and not res2["lens_A_strict"] and not res2["lens_B_pragmatic"], "selftest invariant broken"
    print("[battery selftest] OK -- gate logic exercised (ship passes; ghost fails A+B; chaser gated).")


if __name__ == "__main__":
    _selftest()
