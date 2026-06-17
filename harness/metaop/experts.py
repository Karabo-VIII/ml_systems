"""Harness EXPERTS -- optional specialist PERSONAS attached to loop workers/judges in expert mode.

Project-agnostic version. The original read the crypto repo's `.claude/agents/*.md` files. Here a "persona" is
just a markdown file in a PERSONA DIRECTORY you point the harness at (HARNESS_PERSONAS env var, or `persona_dir=`).
Each `<name>.md` is a system-prompt body (YAML frontmatter is stripped). The harness ships ZERO personas by
default -- expert mode then degrades gracefully to generic workers (load() returns ''), so nothing breaks; supply
your own persona dir to specialize. No emoji (Windows cp1252).

Persona file naming: drop a file `<alias>.md` (e.g. `auditor.md`, `researcher.md`) in your persona dir. The
kind->alias default routing below means a node of kind 'verify' looks for `auditor.md`, 'build' for
`researcher.md`, 'diverge' for `oracle.md`. Missing files are simply skipped (generic worker).

ALIASES seam (optional): a consumer whose persona files are named differently from the role name (e.g. the crypto
repo registers `expert-auditor.md` for the `auditor` role) can pass an `aliases` map {role: filename-stem}. The
default is no aliasing (the role name IS the filename stem), so the harness stays project-agnostic.
"""
from __future__ import annotations

import os
from pathlib import Path

# fallback when plan doesn't assign an expert: node.kind -> persona alias
KIND_DEFAULT = {"build": "researcher", "verify": "auditor", "diverge": "oracle"}


def persona_dir(override: str | None = None) -> Path | None:
    base = override or os.environ.get("HARNESS_PERSONAS")
    if not base:
        return None
    p = Path(base)
    return p if p.exists() else None


def available(override: str | None = None, aliases: dict | None = None) -> list:
    if aliases:  # a consumer supplying an alias map chooses from the ROLE names (the keys), not the file stems.
        return sorted(aliases.keys())
    d = persona_dir(override)
    if not d:
        return sorted(set(KIND_DEFAULT.values()))  # the canonical role names, even with no files present
    return sorted(p.stem for p in d.glob("*.md"))


def _strip_frontmatter(text: str) -> str:
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            return parts[2].strip()
    return text.strip()


def load(name: str, override: str | None = None, aliases: dict | None = None) -> str:
    """Return the persona's system-prompt body, or '' if unknown/missing/no persona dir configured. If `aliases` is
    given, the role `name` is mapped to its registered filename stem first (default: name IS the stem)."""
    if not name:
        return ""
    d = persona_dir(override)
    if not d:
        return ""
    stem = (aliases or {}).get(name, name)
    fn = d / f"{stem}.md"
    if not fn.exists():
        return ""
    try:
        return _strip_frontmatter(fn.read_text(encoding="utf-8"))[:6000]
    except Exception:
        return ""


def for_node(node: dict, aliases: dict | None = None) -> str:
    """Which persona role runs this node: explicit node['expert'], else the kind default. If `aliases` is given, an
    unknown role is clamped to 'researcher' (so the planner can't pick a role with no registered persona)."""
    name = node.get("expert") or KIND_DEFAULT.get(node.get("kind", "build"), "researcher")
    if aliases and name not in aliases:
        return "researcher"
    return name


def persona_for_node(node: dict, override: str | None = None, aliases: dict | None = None) -> tuple[str, str]:
    """(persona_name, persona_text) for a node. persona_text is '' when no persona file is found."""
    name = for_node(node, aliases)
    return name, load(name, override, aliases)


if __name__ == "__main__":
    print("available:", available())
    print("for_node verify:", for_node({"kind": "verify"}), "| explicit:", for_node({"expert": "trader"}))
    print("load auditor (likely empty, no persona dir):", repr(load("auditor")))
