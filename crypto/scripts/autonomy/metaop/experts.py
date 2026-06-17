"""Operator EXPERTS -- crypto-consumer SHIM over the canonical harness.metaop.experts (G-A dedup 2026-06-07).

The persona machinery (frontmatter strip, kind->role default, persona load) lives ONCE in harness/metaop/experts.py
(project-agnostic: a persona is any <name>.md in a persona dir). The crypto loop adds two specifics this shim injects:

  - ALIASES: short role name -> the project's registered .claude/agents/<file>.md (e.g. auditor -> expert-auditor),
    so the planner picks 'auditor' and we resolve the real expert-auditor.md persona.
  - the persona dir is the crypto repo's .claude/agents.

available() returns the ALIAS keys (the role names the planner chooses from). load()/for_node()/persona_for_node()
resolve through the ALIASES against .claude/agents. KIND_DEFAULT + _strip_frontmatter are reused from the harness.
No emoji (Windows cp1252).
"""
from __future__ import annotations

from pathlib import Path

from harness.metaop import experts as _h  # canonical engine (KIND_DEFAULT, _strip_frontmatter, load)
from harness.metaop.experts import KIND_DEFAULT, _strip_frontmatter  # noqa: F401  re-export for parity/compat

ROOT = Path(__file__).resolve().parents[3]
AGENTS = ROOT / ".claude" / "agents"
_AGENTS_STR = str(AGENTS)

# short alias -> registered agent file (without .md)
ALIASES = {
    "architect": "expert-architect", "auditor": "expert-auditor", "oracle": "expert-oracle",
    "pipeline": "expert-pipeline", "researcher": "expert-researcher", "trader": "expert-trader",
    "trainer": "expert-trainer", "validator": "expert-validator", "recon": "recon", "scout": "scout-strat",
}


def available() -> list:
    return sorted(ALIASES.keys())


def load(name: str) -> str:
    """Return the expert's persona body, resolving the crypto ALIAS, from .claude/agents. '' if unknown/missing.
    Delegates the actual file read + frontmatter strip to the canonical harness loader (override=.claude/agents)."""
    if not name:
        return ""
    return _h.load(ALIASES.get(name, name), _AGENTS_STR)


def for_node(node: dict) -> str:
    """Which expert runs this node: explicit node['expert'], else the kind default; clamped to a known alias."""
    name = node.get("expert") or KIND_DEFAULT.get(node.get("kind", "build"), "researcher")
    return name if name in ALIASES else "researcher"


def persona_for_node(node: dict) -> tuple[str, str]:
    """(expert_name, persona_text) for a node."""
    name = for_node(node)
    return name, load(name)


if __name__ == "__main__":
    print("available:", available())
    for k in ("auditor", "researcher", "oracle"):
        print(f"{k}: {len(load(k))} chars persona")
    print("for_node verify:", for_node({"kind": "verify"}), "| explicit:", for_node({"expert": "trader"}))
