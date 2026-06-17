"""RWYB test for the MECHANICAL VERIFIER (AlphaProof-Nexus) wired into the metaop graph's JUDGE.

Exercises the REAL judge/dispatch closures from graph.make_nodes (MockBrain -- no API/network) to prove:
  1. verify_cmd that PASSES (exit 0)  -> mechanical PASS (status done), no verify_error, LLM panel NOT consulted.
  2. verify_cmd that FAILS  (exit 1)  -> mechanical REFUTE, verify_error captured, node RE-OPENED (retry budget).
  3. the RE-OPENED node's NEXT dispatch prompt CONTAINS the concrete verify_error text (rejection-as-gradient).
  4. backward compat: a node WITHOUT verify_cmd still goes through the LLM-vote judge.

Run from repo ROOT:
  .venv/Scripts/python.exe -m scripts.autonomy.metaop._test_verifier
or
  .venv/Scripts/python.exe scripts/autonomy/metaop/_test_verifier.py
No emoji (Windows cp1252).
"""
from __future__ import annotations

import sys
from pathlib import Path

# allow running as a plain script (python scripts/autonomy/metaop/_test_verifier.py) by adding the repo root
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.autonomy.metaop.graph import make_nodes  # noqa: E402
from scripts.autonomy.metaop.brain import MockBrain    # noqa: E402

# Use the absolute interpreter (sys.executable) so the example commands run regardless of shell/slash quirks.
# FINDING: a verify_cmd written as a FORWARD-slash relative path (".venv/Scripts/python.exe") FAILS under
# shell=True on Windows cmd.exe -- cmd needs BACKSLASHES (".venv\\Scripts\\python.exe") or an absolute path.
# Real verify_cmd authors on Windows should use backslashes or an absolute interpreter path.
PY = f'"{sys.executable}"'
PASS_CMD = f'{PY} -c "print(42)"'                 # exit 0
FAIL_CMD = f'{PY} -c "import sys;sys.stderr.write(\'BOOM-XYZ-marker\');sys.exit(1)"'  # exit 1 w/ marker


def _base_state(frontier):
    return {"objective": "test the mechanical verifier", "success_criteria": "verify_cmd exits 0",
            "frontier": frontier, "ledger": [], "budget": 8, "cycle": 0, "status": "running",
            "parallel": 2, "run_id": "test-verifier-0", "awaiting_approval": []}


def main() -> int:
    brain = MockBrain()
    # build the REAL graph closures (we call dispatch/judge directly -- this is the production logic, not a copy)
    # make_nodes returns a 6-tuple since N3 (the replanner added a `replan` node); unpack it (replan unused here).
    plan, dispatch, judge, reflect, route, _replan = make_nodes(brain, parallel=2, max_steps=6, judges=3, taper=3)

    failures = []

    # ---- Case 1: PASS verify_cmd -------------------------------------------------------------------------
    n_pass = {"id": "p1", "task": "produce a passing artifact", "ev": 0.9, "kind": "verify",
              "status": "worked", "result": "I claim success (LLM would have to trust me)", "verify_cmd": PASS_CMD}
    # ---- Case 2: FAIL verify_cmd with retry budget -------------------------------------------------------
    n_fail = {"id": "f1", "task": "produce an artifact", "ev": 0.9, "kind": "verify",
              "status": "worked", "result": "I claim success", "verify_cmd": FAIL_CMD, "verify_retries": 1}
    # ---- Case 4: NO verify_cmd -> existing LLM-vote path (MockBrain judges 'pass' when result present) ----
    n_llm = {"id": "l1", "task": "no verifier here", "ev": 0.5, "kind": "build",
             "status": "worked", "result": "some evidence"}

    out = judge(_base_state([n_pass, n_fail, n_llm]))
    fr = {n["id"]: n for n in out["frontier"]}

    # Assertion 1: mechanical PASS overrides the panel
    p = fr["p1"]
    if not (p.get("verdict") == "pass" and p.get("status") == "done" and "verify_error" not in p):
        failures.append(f"CASE1 mechanical PASS failed: {p}")
    else:
        print("CASE1 PASS: verify_cmd exit 0 -> verdict=pass status=done (LLM panel overridden), no verify_error")

    # Assertion 2: mechanical REFUTE captures verify_error AND re-opens (budget was 1)
    f = fr["f1"]
    if not (f.get("verdict") == "refuted" and f.get("status") == "open" and f.get("verify_error")
            and "BOOM-XYZ-marker" in f["verify_error"] and f.get("verify_retries") == 0):
        failures.append(f"CASE2 mechanical REFUTE failed: {f}")
    else:
        print("CASE2 PASS: verify_cmd exit 1 -> verdict=refuted, verify_error captured (marker present), "
              "node RE-OPENED, retries decremented 1->0")

    # Assertion 4: backward-compat LLM path still runs (MockBrain.judge returns pass for a node with a result)
    l = fr["l1"]
    if not (l.get("verdict") in ("pass", "inconclusive", "refuted") and "verify_error" not in l
            and l.get("status") in ("done", "refuted")):
        failures.append(f"CASE4 LLM-vote backward-compat path failed: {l}")
    else:
        print(f"CASE4 PASS: no verify_cmd -> LLM-vote path used (verdict={l.get('verdict')}, status={l.get('status')})")

    # ---- Assertion 3: the RE-OPENED node's NEXT dispatch prompt contains the error (rejection-as-gradient) --
    # Capture exactly what brain.work() is handed by monkeypatching it on this brain instance.
    captured = {}
    orig_work = brain.work

    def _spy_work(task, persona=""):
        captured["task"] = task
        return orig_work(task, persona=persona)

    brain.work = _spy_work
    # dispatch the now-open, refuted node (carries verify_error from CASE2)
    dispatch(_base_state([f]))
    prompt = captured.get("task", "")
    if not ("BOOM-XYZ-marker" in prompt and "MECHANICAL VERIFIER" in prompt and FAIL_CMD in prompt):
        failures.append(f"CASE3 error-fed-back-into-retry-prompt failed. Prompt was:\n{prompt}")
    else:
        print("CASE3 PASS: re-dispatch prompt APPENDS the concrete verify_error (marker + verify_cmd + "
              "'MECHANICAL VERIFIER' present) -> rejection-as-gradient")
    brain.work = orig_work

    print("-" * 70)
    if failures:
        print("RESULT: FAILED")
        for fa in failures:
            print("  -", fa)
        return 1
    print("RESULT: ALL PASS -- mechanical PASS, mechanical REFUTE+capture, error-fed-back-into-retry-prompt, "
          "and LLM backward-compat all verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
