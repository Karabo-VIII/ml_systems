"""Closest-neighbor matcher for /command — typos AND aliases.

Two layers:
  1. **Alias table** (.claude/skills/_common/SKILL_ALIASES.yaml)
     Explicit "debate" -> "dialectic", "consult" -> "oracle", etc.
     Matches EXACTLY and returns similarity 1.0.
  2. **Fuzzy match** (difflib.SequenceMatcher / Ratcliff-Obershelp)
     Catches typos like "oarcle" -> "oracle". Threshold 0.55.

The alias layer fires FIRST so that semantic synonyms route reliably
even when they have arbitrary edit distance from the canonical name.

Usage:
    python scripts/fuzzy_slash_match.py oarcle
    -> oracle  (0.833, fuzzy)

    python scripts/fuzzy_slash_match.py debate
    -> dialectic  (1.000, alias)

    python scripts/fuzzy_slash_match.py xyz
    -> NO_MATCH

The output is TSV: <input>\\t<resolved>\\t<similarity>\\t<source>
where <source> is "alias", "fuzzy", or "NO_MATCH".
"""
from __future__ import annotations
import sys
from difflib import SequenceMatcher
from pathlib import Path


def list_skills(skills_root: Path) -> list[str]:
    """Enumerate available skill names (dir names under .claude/skills/)."""
    if not skills_root.exists():
        return []
    names = []
    for p in skills_root.iterdir():
        if not p.is_dir():
            continue
        if p.name.startswith("_"):
            continue
        if (p / "SKILL.md").exists():
            names.append(p.name)
    return sorted(names)


def load_aliases(aliases_path: Path) -> tuple[dict[str, str], list[str]]:
    """Load alias -> canonical map + built-ins list from SKILL_ALIASES.yaml.

    Returns:
        (alias_map, builtins)
        alias_map: {alias: canonical_skill_name}
        builtins: list of built-in skill names (not in .claude/skills/)

    The special key `__builtins__` collects bare skill names (no aliases)
    that are shipped with the harness. They become resolvable targets so
    that alias mappings like "settings" -> "update-config" don't fail.
    """
    if not aliases_path.exists():
        return {}, []
    rev: dict[str, str] = {}
    builtins: list[str] = []
    current_canonical: str | None = None
    for line in aliases_path.read_text(encoding="utf-8").splitlines():
        s = line.rstrip()
        if not s or s.lstrip().startswith("#"):
            continue
        # Top-level "canonical:" line
        if not s.startswith(" ") and not s.startswith("\t") and s.endswith(":"):
            current_canonical = s[:-1].strip()
            continue
        # Indented "  - alias" line
        stripped = s.strip()
        if stripped.startswith("- ") and current_canonical:
            alias_raw = stripped[2:].strip()
            # Strip inline comments
            if "#" in alias_raw:
                alias_raw = alias_raw.split("#", 1)[0].strip()
            if alias_raw:
                if current_canonical == "__builtins__":
                    builtins.append(alias_raw.lower())
                else:
                    rev[alias_raw.lower()] = current_canonical
    return rev, builtins


def fuzzy_match(query: str, candidates: list[str],
                threshold: float = 0.55) -> tuple[str, float] | None:
    """Ratcliff-Obershelp fuzzy match. Returns None below threshold."""
    if not candidates:
        return None
    query = query.lower().strip().lstrip("/")
    if not query:
        return None
    if query in candidates:
        return (query, 1.0)
    scored = [(c, SequenceMatcher(None, query, c).ratio()) for c in candidates]
    scored.sort(key=lambda x: -x[1])
    best, best_sim = scored[0]
    if best_sim < threshold:
        return None
    return (best, best_sim)


def resolve(query: str, candidates: list[str],
            aliases: dict[str, str]) -> tuple[str, float, str] | None:
    """Two-layer resolve. Aliases first, then fuzzy.

    Returns (canonical, similarity, source) or None.
    """
    if not query:
        return None
    q = query.lower().strip().lstrip("/")
    if not q:
        return None
    # Layer 0: exact match against canonical
    if q in candidates:
        return (q, 1.0, "exact")
    # Layer 1: alias table
    if q in aliases:
        target = aliases[q]
        # Only return if the alias target is a live skill
        if target in candidates:
            return (target, 1.0, "alias")
    # Layer 2: fuzzy
    fm = fuzzy_match(q, candidates)
    if fm is None:
        return None
    return (fm[0], fm[1], "fuzzy")


def main():
    if len(sys.argv) < 2:
        print("Usage: fuzzy_slash_match.py <query> [<query2> ...]", file=sys.stderr)
        sys.exit(2)
    project_root = Path(__file__).resolve().parent.parent
    skills_root = project_root / ".claude" / "skills"
    aliases_path = skills_root / "_common" / "SKILL_ALIASES.yaml"
    candidates = list_skills(skills_root)
    if not candidates:
        print(f"NO_SKILLS_FOUND at {skills_root}", file=sys.stderr)
        sys.exit(1)
    aliases, builtins = load_aliases(aliases_path)
    # Combined candidate set: project skills + harness built-ins.
    # The matcher resolves to either; the harness's Skill tool will then
    # invoke whichever (built-ins still work via the same Skill(skill="run")
    # interface).
    candidates_all = sorted(set(candidates) | set(builtins))
    for q in sys.argv[1:]:
        r = resolve(q, candidates_all, aliases)
        if r is None:
            print(f"{q}\tNO_MATCH\t0.000\tnone")
        else:
            name, sim, src = r
            print(f"{q}\t{name}\t{sim:.3f}\t{src}")


if __name__ == "__main__":
    main()
