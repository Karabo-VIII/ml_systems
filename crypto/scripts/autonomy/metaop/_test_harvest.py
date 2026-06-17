"""RWYB regression test for H4 (2026-06-09): MONOTONIC skill-library auto-harvest on mechanical pass.

Exercises the REAL shim judge closure (scripts.autonomy.metaop.graph.make_nodes -> injects _crypto_harvester)
with a MockBrain to prove:
  1. a BUILD node with a verify_cmd that PASSES (exit 0) is auto-REGISTERED into the skill library
     (entry name metaop_<id>, kind tool, tags include 'auto_harvest', tested_on carries the passing verify_cmd).
  2. a VERIFY-kind node that passes mechanically is NOT harvested (verify/diverge assert nothing reusable).
  3. a node whose verify_cmd FAILS (refuted) is NOT harvested.

The real registry is NOT touched: skill_library.INDEX_PATH is redirected to a temp file for the test.

Run from repo ROOT:
  .venv/Scripts/python.exe scripts/autonomy/metaop/_test_harvest.py
No emoji (Windows cp1252).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# the shim adds scripts/autonomy to sys.path at import; ensure skill_library is importable here too
sys.path.insert(0, str(ROOT / "scripts" / "autonomy"))

from scripts.autonomy.metaop.graph import make_nodes  # noqa: E402  (shim -> injects _crypto_harvester)
from scripts.autonomy.metaop.brain import MockBrain    # noqa: E402
import skill_library                                   # noqa: E402

PY = f'"{sys.executable}"'
PASS_CMD = f'{PY} -c "print(1)"'              # exit 0
FAIL_CMD = f'{PY} -c "import sys;sys.exit(1)"'  # exit 1


def _state(frontier):
    return {"objective": "test harvest", "success_criteria": "passes harvested",
            "frontier": frontier, "ledger": [], "budget": 8, "cycle": 0, "status": "running",
            "parallel": 4, "run_id": "test-harvest-0", "awaiting_approval": []}


def main() -> int:
    fails = []
    tmpd = tempfile.mkdtemp(prefix="h4_skill_")
    tmp_index = Path(tmpd) / "INDEX.json"
    # redirect the registry to the temp file so the real INDEX.json is untouched
    orig_index = skill_library.INDEX_PATH
    skill_library.INDEX_PATH = tmp_index
    try:
        brain = MockBrain()
        _p, _d, judge, _r, _ro, _rp = make_nodes(brain, parallel=4, max_steps=6, judges=3, taper=3)

        nodes = [
            {"id": "buildA", "task": "build a reusable tool", "ev": 0.9, "kind": "build",
             "status": "worked", "result": "ok", "verify_cmd": PASS_CMD},
            {"id": "verifyB", "task": "verify a claim", "ev": 0.9, "kind": "verify",
             "status": "worked", "result": "ok", "verify_cmd": PASS_CMD},
            {"id": "buildC", "task": "build that fails its check", "ev": 0.9, "kind": "build",
             "status": "worked", "result": "ok", "verify_cmd": FAIL_CMD},
        ]
        judge(_state(nodes))

        assets = {a["name"]: a for a in (json.loads(tmp_index.read_text(encoding="utf-8"))["assets"]
                                         if tmp_index.exists() else [])}

        # 1. the passing BUILD node was harvested
        if "metaop_buildA" not in assets:
            fails.append(f"buildA (mechanical pass build) was NOT harvested. registry names: {list(assets)}")
        else:
            a = assets["metaop_buildA"]
            if a.get("kind") != "tool":
                fails.append(f"buildA harvested with kind={a.get('kind')!r}, expected 'tool'")
            if "auto_harvest" not in a.get("tags", []):
                fails.append(f"buildA missing 'auto_harvest' tag: {a.get('tags')}")
            if "print(1)" not in (a.get("tested_on") or ""):
                fails.append(f"buildA tested_on does not carry the passing verify_cmd: {a.get('tested_on')!r}")
            if not fails:
                print("CASE1 PASS: passing BUILD node auto-registered (kind=tool, auto_harvest tag, "
                      "tested_on carries the verify_cmd)")

        # 2. the VERIFY-kind pass was NOT harvested
        if "metaop_verifyB" in assets:
            fails.append("verifyB (verify kind) WAS harvested but should be excluded")
        else:
            print("CASE2 PASS: verify-kind mechanical pass NOT harvested (asserts nothing reusable)")

        # 3. the FAILING build was NOT harvested
        if "metaop_buildC" in assets:
            fails.append("buildC (verify_cmd FAILED -> refuted) WAS harvested but should not be")
        else:
            print("CASE3 PASS: failing build (refuted) NOT harvested")

    finally:
        skill_library.INDEX_PATH = orig_index
        try:
            tmp_index.unlink()
            os.rmdir(tmpd)
        except OSError:
            pass

    print("-" * 70)
    if fails:
        print("RESULT: FAILED")
        for f in fails:
            print("  -", f)
        return 1
    print("RESULT: ALL PASS -- mechanical pass on a reusable-kind node is harvested; verify/failed are not. "
          "Monotonicity hook live.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
