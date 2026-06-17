"""harness/metaop/skills.py -- give a (generic / LOCAL) model SKILLS + LOCALISED CONTEXT, the way Claude Code does.

THE MECHANISM (model-agnostic, no magic): a skill / a rule file / a memory is just *structured text injected into
the prompt*, plus a *relevance SELECTOR* that respects a finite context budget. This module is that mechanism,
portable and dependency-free.

  - A SKILL is a markdown file with frontmatter (`name`, `description`) and a body. Either layout works:
        <skills_dir>/<name>/SKILL.md      (Claude-Code / open "Agent Skills" convention)
        <skills_dir>/<name>.md            (flat)
  - PROGRESSIVE DISCLOSURE (3 tiers): the cheap MANIFEST (name + description per skill, ~tens of tokens each) is
    always available; a skill's BODY is loaded ONLY when that skill is SELECTED; bundled reference files / scripts
    are read/run on demand (left to the worker's tools). So you can have many skills but pay for the 1-3 relevant.
  - The SELECTOR is MECHANICAL by default (token-overlap / BM25-lite over name+description). This is the SOTA choice
    for a SMALL local model: a frontier model can self-route a big manifest, but a 7B cannot -- it burns its limited
    reasoning on tool-disambiguation. So we hand the 7B only the top-k pre-selected skills. Pass `embedder=` (any
    callable that ranks) to upgrade to dense/semantic retrieval (e.g. a local nomic-embed-text via memory.py).
  - LOCALISED CONTEXT (the CLAUDE.md / project-instruction equivalent): `context_pack()` concatenates project files
    (invariants, a dead-list, founding framing) under a hard char budget.

WIRING (ZERO graph changes -- uses the existing host hooks in graph.py):
    from metaop.skills import skills_recaller, context_framer
    app = build(brain,
                recaller=skills_recaller(skills_dir, k=3),       # selected skills -> payload["recall"]
                framer=context_framer(["CONTEXT.md", "DEAD_LIST.md"]))  # project context -> payload["framing"]
The planner already consumes payload["recall"] + payload["framing"], so the model gets relevant skills + project
context with nothing else to change. Or set a Brain's `domain=context_pack([...])` for always-on context.

No third-party deps. No emoji (Windows cp1252).
"""
from __future__ import annotations

import os
import re
from pathlib import Path


# --------------------------------------------------------------------------- frontmatter + discovery
def parse_frontmatter(md: str) -> tuple[dict, str]:
    """Split a markdown doc into ({frontmatter}, body). Frontmatter is a leading `---`-delimited block of simple
    `key: value` lines (no YAML dependency -- we only need name/description; nested YAML values are returned raw).
    Missing/blank frontmatter -> ({}, full text)."""
    if not md.startswith("---"):
        return {}, md.strip()
    parts = md.split("---", 2)
    if len(parts) < 3:
        return {}, md.strip()
    fm: dict = {}
    for line in parts[1].splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, val = line.split(":", 1)
        fm[key.strip().lower()] = val.strip().strip('"').strip("'")
    return fm, parts[2].strip()


def _skill_files(skills_dir: str | os.PathLike) -> list[Path]:
    """Every skill file under skills_dir: <name>/SKILL.md (dir style) + <name>.md (flat). Deterministic order."""
    d = Path(skills_dir)
    if not d.is_dir():
        return []
    found = sorted(d.glob("*/SKILL.md")) + sorted(p for p in d.glob("*.md") if p.name.lower() != "readme.md")
    return found


def discover(skills_dir: str | os.PathLike) -> list[dict]:
    """Read every skill -> [{name, description, body, path}]. `name` falls back to the dir/file stem; `description`
    to the first non-empty body line. Unreadable files are skipped (never raises)."""
    out = []
    for fp in _skill_files(skills_dir):
        try:
            fm, body = parse_frontmatter(fp.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        stem = fp.parent.name if fp.name.upper() == "SKILL.MD" else fp.stem
        name = fm.get("name") or stem
        desc = fm.get("description") or (body.splitlines()[0].strip() if body.strip() else "")
        out.append({"name": name, "description": desc[:1024], "body": body, "path": str(fp)})
    return out


# --------------------------------------------------------------------------- selection (mechanical, embedding-optional)
_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


def _overlap_score(query_toks: set, doc_text: str) -> float:
    """BM25-lite: fraction of the doc's distinct tokens that the query hits, length-normalised so a short, on-point
    description isn't buried by a long one. Zero deps; good enough to put the right skill in the top-3."""
    dts = _tokens(doc_text)
    if not dts:
        return 0.0
    dset = set(dts)
    hits = sum(1 for t in dset if t in query_toks)
    return hits / (len(dset) ** 0.5)


def select(objective: str, skills_dir: str | os.PathLike, k: int = 3, embedder=None,
           use_body: bool = False) -> list[dict]:
    """Return the top-k skills most relevant to `objective` (most-relevant first). Default = mechanical token-overlap
    over name+description (+body if use_body). Pass `embedder(query, docs)->list[float]` (one score per doc, same
    order) to use dense/semantic ranking instead -- the small-model RAG upgrade. k<=0 returns all, ranked."""
    skills = discover(skills_dir)
    if not skills:
        return []
    if embedder is not None:
        docs = [f"{s['name']}. {s['description']}" + (("\n" + s["body"]) if use_body else "") for s in skills]
        try:
            scores = list(embedder(objective, docs))
        except Exception:
            scores = None
        if scores and len(scores) == len(skills):
            ranked = sorted(zip(skills, scores), key=lambda x: x[1], reverse=True)
            return [s for s, _ in (ranked if k <= 0 else ranked[:k])]
    q = set(_tokens(objective))
    scored = [(s, _overlap_score(q, s["name"] + " " + s["description"] + (("\n" + s["body"]) if use_body else "")))
              for s in skills]
    scored.sort(key=lambda x: x[1], reverse=True)
    chosen = [s for s, sc in scored if sc > 0]
    if not chosen:                       # nothing matched -> fall back to declaration order (better than empty)
        chosen = [s for s, _ in scored]
    return chosen if k <= 0 else chosen[:k]


# --------------------------------------------------------------------------- manifest + digest (progressive disclosure)
def manifest(skills_dir: str | os.PathLike) -> str:
    """Tier-0: one `- name: description` line per skill. The cheap always-on routing table (NOT the bodies)."""
    sk = discover(skills_dir)
    if not sk:
        return ""
    return "AVAILABLE SKILLS (load the body of a relevant one):\n" + "\n".join(
        f"- {s['name']}: {s['description']}" for s in sk)


def digest(objective: str, skills_dir: str | os.PathLike, k: int = 3, body_chars: int = 1800) -> str:
    """A recaller-shaped string: the Tier-0 manifest (always) + the BODIES of the top-k selected skills (Tier-1,
    progressive disclosure, each truncated to body_chars). This is exactly what a relevant SKILL.md injection looks
    like -- and it never exceeds ~ k*body_chars + manifest, so it fits a small model's window."""
    man = manifest(skills_dir)
    if not man:
        return ""
    chosen = select(objective, skills_dir, k=k)
    blocks = [f"### SKILL: {s['name']}\n{s['description']}\n\n{(s['body'] or '').strip()[:body_chars]}"
              for s in chosen]
    sel = "\n\n".join(blocks)
    return man + ("\n\n--- SELECTED (most relevant to this objective) ---\n\n" + sel if sel else "")


# --------------------------------------------------------------------------- localised context (CLAUDE.md-equivalent)
def context_pack(context_files: list[str] | None, max_chars: int = 4000, per_file_chars: int = 2500) -> str:
    """Concatenate project-context files (the CLAUDE.md / rules / dead-list equivalent) under a HARD char budget.
    Each file is head-truncated to per_file_chars; the whole pack to max_chars (small-model budgeting). Missing
    files are skipped silently. Returns '' if nothing readable."""
    if not context_files:
        return ""
    chunks = []
    for f in context_files:
        try:
            txt = Path(f).read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            continue
        if txt:
            chunks.append(f"=== {Path(f).name} ===\n{txt[:per_file_chars]}")
    return ("\n\n".join(chunks))[:max_chars]


# --------------------------------------------------------------------------- drop-in host hooks for build(...)
def skills_recaller(skills_dir: str | os.PathLike, k: int = 3, embedder=None):
    """Return a `recaller(objective) -> str` for build(recaller=...): selects the top-k relevant skills and returns
    their manifest + bodies (progressive disclosure). Never raises into the loop."""
    def _recall(objective: str) -> str:
        try:
            if embedder is not None:
                chosen = select(objective, skills_dir, k=k, embedder=embedder)
                man = manifest(skills_dir)
                blocks = [f"### SKILL: {s['name']}\n{s['description']}\n\n{(s['body'] or '').strip()[:1800]}"
                          for s in chosen]
                return man + ("\n\n--- SELECTED ---\n\n" + "\n\n".join(blocks) if blocks else "")
            return digest(objective, skills_dir, k=k)
        except Exception:
            return ""
    return _recall


def _slug(text: str, maxlen: int = 48) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return s[:maxlen].strip("-") or "skill"


def skill_harvester(skills_dir: str | os.PathLike):
    """Return a `harvester(node)` for build(harvester=...): when a node passes the MECHANICAL verifier (ground truth),
    AUTHOR a SKILL.md capturing the verified capability into skills_dir -- so the harness GROWS its OWN skill library
    from what it proves (the Voyager monotonicity hook + the literal 'augment itself from within its own loop'). The
    new skill is then selectable on future runs. Idempotent (re-verifying refreshes the same skill, never duplicates);
    never raises into the loop."""
    def _harvest(node: dict) -> None:
        try:
            task = (node.get("task") or "").strip()
            task = re.sub(r"^(build|verify|diverge)\s*:\s*", "", task, flags=re.I).strip()  # drop the node-kind prefix
            if not task:
                return
            name = _slug(task)
            d = Path(skills_dir) / name
            d.mkdir(parents=True, exist_ok=True)
            vc = node.get("verify_cmd") or ""
            res = str(node.get("result") or "")[:600]
            body = ("A capability this harness BUILT and mechanically VERIFIED (harvested automatically).\n\n"
                    f"Task: {task}\n\n"
                    + (f"Verified by (exit 0 == pass): `{vc}`\n\n" if vc else "")
                    + (f"Notes / last result:\n{res}\n" if res else ""))
            (d / "SKILL.md").write_text(
                f"---\nname: {name}\ndescription: {task[:240]} (a verified capability harvested by the harness; "
                f"reuse its approach).\n---\n{body}", encoding="utf-8")
        except Exception:
            pass
    return _harvest


def context_framer(context_files: list[str] | None, max_chars: int = 4000):
    """Return a `framer(objective) -> dict` for build(framer=...): injects the project context pack as
    payload['framing']['project_context']. Never raises into the loop."""
    def _frame(_objective: str) -> dict:
        try:
            pack = context_pack(context_files, max_chars=max_chars)
            return {"project_context": pack} if pack else {}
        except Exception:
            return {}
    return _frame


if __name__ == "__main__":  # tiny self-test against a throwaway skills dir
    import sys
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    (tmp / "pdf").mkdir()
    (tmp / "pdf" / "SKILL.md").write_text(
        "---\nname: pdf-extract\ndescription: Extract text and tables from PDF files. Use when a task mentions PDFs"
        " or documents.\n---\nUse pdfplumber. Steps: 1) open 2) iterate pages 3) extract.\n", encoding="utf-8")
    (tmp / "sql.md").write_text(
        "---\nname: sql-tune\ndescription: Optimize slow SQL queries and add indexes. Use for database performance.\n"
        "---\nEXPLAIN ANALYZE first; add covering indexes.\n", encoding="utf-8")
    print("manifest:\n", manifest(tmp))
    print("\nselect('my report is a slow pdf'):", [s["name"] for s in select("my report is a slow pdf", tmp, k=1)])
    print("select('the database query is slow'):", [s["name"] for s in select("the database query is slow", tmp, k=1)])
    ok = (select("extract a pdf document", tmp, k=1)[0]["name"] == "pdf-extract"
          and select("index my database query", tmp, k=1)[0]["name"] == "sql-tune")
    print("\nSELF-TEST:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
