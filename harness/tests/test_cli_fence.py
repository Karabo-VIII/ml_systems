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
