"""End-to-end selftest of the solutioning pipeline -- the framework verifying ITSELF (a SOTA requirement: the
evaluator/store must be testable, not asserted). Runs the full init->record->run->gate->advance->doctor cycle on a
throwaway workspace, asserts the SOTA invariants (lineage captured, machine gates flip correctly, status==gate.passed,
dedup-on-path, doctor catches a planted break), then cleans up. Exit 0 = healthy; exit 1 = a regression.

Run:  python -m framework.selftest   (or python src/framework/selftest.py)
Wireable into CDAP / a pre-commit hook so the store machinery can't silently regress. No emoji.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from framework import pipeline as P  # noqa: E402

M, I = "_selftest", "_ws"
CHECKS: list[tuple[str, bool]] = []


def _ck(name, cond):
    CHECKS.append((name, bool(cond)))


def run() -> int:
    ws = P.ws_dir(M, I)
    if ws.exists():
        shutil.rmtree(ws, ignore_errors=True)
    try:
        # 1. init is idempotent + atomic + creates the manifest
        P.init(M, I)
        _ck("init creates manifest", P.manifest_path(M, I).exists())
        P.init(M, I)  # idempotent (must not clobber)

        # 2. record captures LINEAGE (git_sha + artifact hash)
        art = P.record(M, I, "00_research", "docs/SOLUTIONING_PIPELINE.md", "doc", "selftest")
        _ck("record captures git_sha lineage", art["lineage"].get("git_sha", "n/a") != "n/a")
        _ck("record captures artifact_sha256", len(art["lineage"].get("artifact_sha256", "")) > 0)

        # 3. dedup on PATH (record same path twice -> 1 row)
        P.record(M, I, "00_research", "docs/SOLUTIONING_PIPELINE.md", "doc", "second")
        man = P._read(M, I)
        n_same = sum(1 for a in man["stages"]["00_research"]["artifacts"] if a["path"] == "docs/SOLUTIONING_PIPELINE.md")
        _ck("dedup on path (1 row)", n_same == 1)

        # 4. manual gate requires override; status DERIVES from gate.passed (single source)
        P.gate(M, I, "00_research")  # manual gate w/o override -> stays unpassed
        _ck("manual gate w/o override does NOT pass", not P._read(M, I)["stages"]["00_research"]["gate"]["passed"])
        P.gate(M, I, "00_research", manual_override=True, evidence="selftest")
        man = P._read(M, I)
        _ck("manual-override passes + passed_by=human", man["stages"]["00_research"]["gate"]["passed"]
            and man["stages"]["00_research"]["gate"].get("passed_by") == "human")
        _ck("status DERIVED from gate.passed (==done)", man["stages"]["00_research"]["status"] == "done")

        # 5. advance only when gate passed
        P.gate(M, I, "01_mining", manual_override=True, evidence="selftest")
        adv = P.advance(M, I)  # current=00 (passed) -> 01
        _ck("advance moves to next stage", adv["ok"] and "01_mining" in adv["msg"])

        # 6. MACHINE gate: 02_engine runs a real command (CDAP). pass iff exit in pass_exit.
        g = P.gate(M, I, "02_engine")
        _ck("machine gate ran (passed_by=machine)", g.get("passed_by") == "machine")
        _ck("machine gate evidence has exit code", isinstance(g.get("evidence"), dict) and "exit" in g["evidence"])

        # 7. RUN registry + ship_run_exists predicate (the strat gate)
        gb = P.gate(M, I, "03_strat")  # no SHIP run yet
        _ck("03 gate FAILS before a SHIP run", not gb["passed"])
        P.run(M, I, "03_strat", "cand_x", "NULL", metrics={"unseen": -0.1})
        P.run(M, I, "03_strat", "cand_y", "SHIP", metrics={"unseen": 0.14})
        ga = P.gate(M, I, "03_strat")
        _ck("03 gate PASSES after a SHIP run (machine)", ga["passed"] and ga.get("passed_by") == "machine")
        _ck("run registry persisted", len(P._read_runs(M, I)) == 2)

        # 8. doctor catches a planted MISSING artifact + does not crash
        P.record(M, I, "04_bot", "docs/__NO_SUCH_FILE__.md", "doc", "planted")
        rep, n = P.doctor()
        _ck("doctor catches planted missing path", n >= 1 and "__NO_SUCH_FILE__" in rep)
    finally:
        shutil.rmtree(P.ws_dir(M, I), ignore_errors=True)
        P.registry()  # refresh the real registry (drops the throwaway)

    failed = [n for n, ok in CHECKS if not ok]
    # SCOPE-HONEST banner: this selftest covers ONLY the solutioning-pipeline STORE
    # (init/record/gate/advance/doctor over a workspace manifest). It runs ZERO
    # model-engine code -- a reader must NOT read "N/N passed" as "the engine is
    # verified". The model engine has its own separate selftests:
    #   python -m framework.router    --selftest   (routing cascade)
    #   python -m framework.solve     --selftest   (plan_solve over canonical problems)
    #   python -m framework.general_trainer --controls   (Layer-B trainer two-sided control)
    print(f"=== framework STORE selftest: {len(CHECKS)-len(failed)}/{len(CHECKS)} passed "
          f"(solutioning-pipeline store ONLY -- NOT the model engine) ===")
    for n, ok in CHECKS:
        print(f"  {'PASS' if ok else 'FAIL'}  {n}")
    print("  NOTE: model-engine verification is separate -- run the router / solve / "
          "general_trainer selftests (see module header).")
    if failed:
        print(f"\nFAILED: {failed}")
    return 0 if not failed else 1


def run_engine() -> int:
    """Run the MODEL-ENGINE selftests (router / solve-planner+executor / two-sided trainer controls)
    -- the code the STORE selftest does NOT touch. Returns the total failure count (0 = all green).

    This is what makes `framework.selftest --full` an honest "the engine is verified" command, instead
    of the store-only "14/14" that previously implied more than it checked. Trains a few small CPU
    models (the two-sided controls), so it is slower (~1-2 min) than the store selftest."""
    fails = 0
    print("\n" + "=" * 72)
    print("  MODEL-ENGINE selftests (router + solve + two-sided trainer controls)")
    print("=" * 72)

    # 1. router -- routing cascade (fast, no training)
    print("\n[engine 1/3] framework.router._selftest")
    try:
        from framework import router as _R
        fails += int(_R._selftest(verbose=True))
    except Exception as exc:  # noqa: BLE001 -- a broken import is itself a failure to report
        print(f"  [FAIL] router selftest raised {type(exc).__name__}: {exc}")
        fails += 1

    # 2. solve -- planner over canonical problems + the executor two-sided control
    print("\n[engine 2/3] framework.solve._selftest (planner + executor)")
    try:
        from framework import solve as _S
        fails += int(_S._selftest(verbose=True))
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] solve selftest raised {type(exc).__name__}: {exc}")
        fails += 1

    # 3. general_trainer -- two-sided soundness (positive learns, negative does NOT hallucinate)
    print("\n[engine 3/3] framework.general_trainer.run_controls (two-sided soundness)")
    try:
        from framework import general_trainer as _GT
        res = _GT.run_controls(verbose=False)
        ok = bool(res.get("verdict", {}).get("overall_pass", False))
        v = res.get("verdict", {})
        print(f"  {'PASS' if ok else 'FAIL'}  controls: pos={v.get('positive_control_pass')} "
              f"neg={v.get('negative_control_pass')} boundary={v.get('boundary_pass')}")
        if not ok:
            fails += 1
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] general_trainer controls raised {type(exc).__name__}: {exc}")
        fails += 1

    print(f"\n=== model-engine selftests: {'ALL PASS' if not fails else str(fails) + ' FAILED'} ===")
    return fails


def run_full() -> int:
    """STORE selftest + MODEL-ENGINE selftests aggregated -- the single honest 'everything verified'
    command. Returns 0 only if BOTH the store machinery AND the model engine are green."""
    store_rc = run()
    engine_fails = run_engine()
    total_fail = (1 if store_rc else 0) + engine_fails
    print("\n" + "=" * 72)
    print(f"  FRAMEWORK FULL VERIFY: store={'PASS' if not store_rc else 'FAIL'}  "
          f"engine={'PASS' if not engine_fails else str(engine_fails) + ' FAIL'}  "
          f"-> {'ALL GREEN' if not total_fail else 'FAILURES PRESENT'}")
    print("=" * 72)
    return 0 if not total_fail else 1


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        prog="framework.selftest",
        description=("STORE selftest (default) or --full to ALSO verify the model engine "
                     "(router/solve/trainer-controls). The default tests ONLY the solutioning-pipeline "
                     "store; --full is the honest 'is the whole engine verified' command."),
    )
    ap.add_argument("--full", action="store_true",
                    help="Also run the model-engine selftests (router/solve/two-sided controls). "
                         "Slower (~1-2 min: trains small control models).")
    ap.add_argument("--engine", action="store_true",
                    help="Run ONLY the model-engine selftests (skip the store selftest).")
    args = ap.parse_args()

    if args.engine:
        raise SystemExit(1 if run_engine() else 0)
    if args.full:
        raise SystemExit(run_full())
    raise SystemExit(run())
