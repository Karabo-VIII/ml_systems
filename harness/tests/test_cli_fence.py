"""CLI-LAYER FENCE: under bypassPermissions only PreToolUse hooks can block a tool, so the no-key CLI worker
(CliBrain) wires metaop._cli_fence. Its decision must block destructive Bash + protected-file writes and allow
safe ops. Pins the 2026-07-03 round-2 fix for the unfenced default-path gap. (The decision logic is tested here;
the CLI plumbing is probe-gated + degradable.)"""
from __future__ import annotations

import pytest

from metaop import _cli_fence

_G = "g" + "it"
BLOCK = [
    {"tool_name": "Bash", "tool_input": {"command": "r" "m -rf /"}},
    {"tool_name": "Bash", "tool_input": {"command": "r" "m -fr ."}},
    {"tool_name": "Bash", "tool_input": {"command": _G + " push origin +main"}},
    {"tool_name": "Bash", "tool_input": {"command": _G + " push -f"}},
    {"tool_name": "Bash", "tool_input": {"command": "sudo rm x"}},
    {"tool_name": "Write", "tool_input": {"file_path": ".env"}},
    {"tool_name": "Edit", "tool_input": {"file_path": ".git/config"}},
]
ALLOW = [
    {"tool_name": "Bash", "tool_input": {"command": "python test.py"}},
    {"tool_name": "Bash", "tool_input": {"command": _G + " status"}},
    {"tool_name": "Bash", "tool_input": {"command": _G + " commit -m x"}},
    {"tool_name": "Write", "tool_input": {"file_path": "adder.py"}},
    {"tool_name": "Read", "tool_input": {"file_path": "/etc/hosts"}},
    {},
]


@pytest.mark.parametrize("data", BLOCK)
def test_fence_blocks(data):
    block, reason = _cli_fence.decide(data)
    assert block is True and reason, f"should block: {data}"


@pytest.mark.parametrize("data", ALLOW)
def test_fence_allows(data):
    block, _reason = _cli_fence.decide(data)
    assert block is False, f"should allow: {data}"


def test_cli_fence_wiring_produces_valid_settings(tmp_path, monkeypatch):
    """Regression: _install_cli_fence must actually produce a settings file wiring all 4 tool matchers to the fence.
    The original round-2 code referenced `sys` without importing it -> the exception was swallowed and the fence
    NEVER wired (silent dead code). This exercises the wiring WITHOUT needing the claude CLI."""
    import json

    from metaop.brain import CliBrain
    monkeypatch.setenv("HARNESS_WORKSPACE", str(tmp_path / "ws"))
    b = CliBrain.__new__(CliBrain)          # bypass __init__ (which needs the claude exe); test the method directly
    b.exe = "claude"
    path = b._install_cli_fence()
    assert path, "the fence settings file must be written (not silently None)"
    settings = json.loads(open(path, encoding="utf-8").read())
    hooks = settings.get("hooks", {}).get("PreToolUse", [])
    matchers = {h.get("matcher") for h in hooks}
    assert {"Bash", "Write", "Edit", "MultiEdit"} <= matchers, f"all tool surfaces must be fenced, got {matchers}"
    assert "_cli_fence" in hooks[0]["hooks"][0]["command"], "the hook must invoke metaop._cli_fence"
