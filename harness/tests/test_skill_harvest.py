"""VOYAGER HARVEST: when a build passes the MECHANICAL verifier, the harvested SKILL.md must capture the actual
verified ARTIFACT (re-usable code in ./artifact/), not just an episodic note. Pins the 2026-07-03 harvest upgrade.
Also checks the legacy 1-arg harvester still works (duck-typed cwd)."""
from __future__ import annotations

from pathlib import Path

from metaop.brain import MockBrain
from metaop.graph import build
from metaop.skills import discover, skill_harvester


def test_harvest_captures_verified_artifact(tmp_path):
    skills_dir = tmp_path / "skills"
    build_cwd = tmp_path / "build"
    build_cwd.mkdir()
    (build_cwd / "adder.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")

    app = build(MockBrain(), parallel=1, judges=1, taper=1, channel="test", cwd=str(build_cwd),
                harvester=skill_harvester(str(skills_dir)))
    seed = {"id": "n1", "task": "build: adder.py exposing add(a,b)", "ev": 1.0, "kind": "build", "status": "open",
            "verify_cmd": 'python -c "import adder; assert adder.add(2,3)==5"', "verify_retries": 0}
    init = {"objective": "x", "success_criteria": "y", "frontier": [seed], "ledger": [], "budget": 1, "cycle": 0,
            "status": "running", "parallel": 1, "run_id": "t-harv", "awaiting_approval": []}
    for _ in app.stream(init, {"configurable": {"thread_id": "t-harv"}}, stream_mode="values"):
        pass

    skills = discover(str(skills_dir))
    assert skills, "a verified build must be harvested into a skill"
    artifacts = list(skills_dir.glob("*/artifact/adder.py"))
    assert artifacts, "the harvested skill must contain the actual verified artifact file (Voyager, not a note)"
    assert artifacts[0].read_text() == "def add(a, b):\n    return a + b\n"
    skill_md = next(iter(skills_dir.glob("*/SKILL.md"))).read_text()
    assert "RE-RUNNABLE" in skill_md and "adder.add(2,3)==5" in skill_md


def test_legacy_one_arg_harvester_still_called(tmp_path):
    seen = {}

    def legacy_harvester(node):  # one positional arg -- the crypto shim shape; must still be invoked
        seen["id"] = node.get("id")

    app = build(MockBrain(), parallel=1, judges=1, taper=1, channel="test", cwd=str(tmp_path),
                harvester=legacy_harvester)
    seed = {"id": "n1", "task": "build: x", "ev": 1.0, "kind": "build", "status": "open",
            "verify_cmd": 'python -c "assert 1+1==2"', "verify_retries": 0}
    init = {"objective": "x", "success_criteria": "y", "frontier": [seed], "ledger": [], "budget": 1, "cycle": 0,
            "status": "running", "parallel": 1, "run_id": "t-legacy", "awaiting_approval": []}
    for _ in app.stream(init, {"configurable": {"thread_id": "t-legacy"}}, stream_mode="values"):
        pass
    assert seen.get("id") == "n1", "a legacy 1-arg harvester(node) must still be called (backward compat)"
