"""src/strat/selftest_all.py -- one-shot DATA-FREE regression runner for the apparatus.

Runs the self-tests that need NO external chimera data (synthetic only), asserts their invariants, and
exits nonzero on any failure -- so a future instance (or CI) can regression-check the gate with a single
command after any edit. Covers: battery (ship/ghost/chaser), DSR/Holm gate (ship-fails-Holm -> halt),
the full integrated gate POWER check (positive_control), and STEP-5 benchmark-excess.

The DATA-DEPENDENT smokes (firewall / fill_model / candidate_gate / discover on real chimeras) are run
separately -- see src/strat/README.md -- because they require materialized data.

Run: python src/strat/selftest_all.py   (exit 0 = all apparatus invariants hold)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    results = []

    # 1. battery: ship passes Lens B, ghost fails A+B, chaser gated (internal asserts)
    try:
        from strat import battery
        battery._selftest()
        results.append(("battery (ship/ghost/chaser)", True, ""))
    except Exception as e:  # noqa: BLE001
        results.append(("battery (ship/ghost/chaser)", False, repr(e)))

    # 2. DSR/Holm gate: weak ship-claim @ family_N=200 must HALT (exit 2); null round -> warn (exit 1)
    try:
        from audit import check_dsr_holm
        rc = check_dsr_holm._selftest()
        results.append(("dsr_holm (ship-fails-Holm -> halt)", rc == 0, f"selftest rc={rc}"))
    except Exception as e:  # noqa: BLE001
        results.append(("dsr_holm (ship-fails-Holm -> halt)", False, repr(e)))

    # 3. integrated gate POWER: a synthetic genuine timing edge must pass the frequency-independent check
    try:
        from strat.positive_control import run_positive_control
        r = run_positive_control(verbose=False)
        ok = bool(r["has_power"])
        results.append(("positive_control (gate HAS power)", ok,
                        f"has_power={r['has_power']} ships={r['ships']} beats_held={r['beats_held']}"))
    except Exception as e:  # noqa: BLE001
        results.append(("positive_control (gate HAS power)", False, repr(e)))

    # 4. STEP-5 benchmark-excess: the same synthetic edge must beat the beta-matched static on held-out
    try:
        from strat.positive_control import make_edge_frame
        from strat.benchmark import benchmark_excess
        from wealth_bot.harness import CanonicalHarness, StrategySpec, WindowSpec, sma_past_only
        df = make_edge_frame()
        df["sma_fast"] = sma_past_only(df["close"], 2)
        df["sma_slow"] = sma_past_only(df["close"], 5)
        spec = StrategySpec(fast_col="sma_fast", slow_col="sma_slow", signal="crossover", filter_col=None,
                            exit_policy="signal_flip_or_filter", cost_rt=0.0024, use_funding=False,
                            funding_scale=0.0, max_hold_bars=None, max_hold_ext_bars=None)
        win = WindowSpec(train_end="2024-05-15", val_end="2025-03-15", oos_end="2025-12-31", unseen_end="2026-05-22")
        b = benchmark_excess(CanonicalHarness(df, spec, win, chimera_path="selftest_bench"))
        results.append(("benchmark_excess (edge beats beta)", bool(b["beats_beta_held"]),
                        f"beats_beta_held={b['beats_beta_held']}"))
    except Exception as e:  # noqa: BLE001
        results.append(("benchmark_excess (edge beats beta)", False, repr(e)))

    # 5. WM calibration probe (calib track): two-sided soundness -- a well-calibrated synthetic
    #    predictive distribution -> CALIBRATED, an over-confident (too-narrow) one -> MISCALIBRATED.
    try:
        from strat.wm_calibration_probe import run_two_sided
        r = run_two_sided(verbose=False)
        results.append(("wm_calibration_probe (two-sided soundness)", bool(r["two_sided_ok"]),
                        f"genuine={r['genuine']['overall']} over={r['over']['overall']} under={r['under']['overall']}"))
    except Exception as e:  # noqa: BLE001
        results.append(("wm_calibration_probe (two-sided soundness)", False, repr(e)))

    print("=" * 70)
    print("src/strat apparatus regression (data-free)")
    print("=" * 70)
    all_ok = True
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
        all_ok = all_ok and ok
    print("=" * 70)
    print(f"RESULT: {'ALL PASS' if all_ok else '*** FAILURES ***'}  ({sum(1 for _,o,_ in results if o)}/{len(results)})")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
