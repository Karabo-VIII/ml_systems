"""SKILL ROUTER -- hand an expert skill's mandate to the SHARED fill_window metaop loop, with the SKILL'S OWN
EXPERTISE injected, so the loop PLANS / DISPATCHES / JUDGES through that lens.

The point (the thing the user asked to confirm): /audit and /trader can receive the SAME objective but DECOMPOSE and
solve it DIFFERENTLY -- the auditor attacks it with adversarial red-team checks (gradient flow, leakage, invariants,
cross-version), the trader attacks it with sizing / risk / execution / portfolio nodes -- because each skill's own
SKILL.md protocols are injected as the brain's planning lens (domain + plan_instruction). The loop machinery (the
plan->dispatch->judge->reflect->route->replan graph, the mechanical verifier, fill_window no-idle-stop) is SHARED;
only the LENS changes per skill.

WHY this lives in the crypto layer (not harness/metaop): reading .claude/skills/<skill>/SKILL.md is THIS project's
convention; the harness stays project-agnostic. The router is the thin adapter: skill name -> lens -> shared loop.

Usage:
  python scripts/autonomy/skill_route.py <skill> "<objective>" "<success criteria>" \
      [--backend cli|ollama|mock] [--budget 12] [--parallel 1] [--no-fill-window] [--show-lens]
  --show-lens : print the injected domain + plan_instruction and EXIT (no loop run) -- the confirm/inspect path.

No emoji (Windows cp1252).
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKILLS_DIR = ROOT / ".claude" / "skills"
AGENTS_DIR = ROOT / ".claude" / "agents"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/autonomy -> so `metaop` (the crypto shim) imports
from harness.metaop.brain import make_brain          # noqa: E402
# Use the CRYPTO-SHIM build (scripts/autonomy/metaop/graph), NOT the agnostic harness build: the shim auto-wires the
# depth/breadth framer (problem_framing: anti-impossible rail + depth/breadth axes) + recaller (resourcefulness:
# decompose-the-ideal) + recorder (hypothesis_register: don't re-mine refuted veins). Without this, a skill-routed
# run got the skill lens + planner-prompt breadth but MISSED those narrow-mindedness countermeasures. 2026-06-08.
from metaop.graph import build                       # noqa: E402  (crypto shim -> framer/recaller/recorder auto-wired)


def available_skills() -> list[str]:
    """Skill names that carry a SKILL.md (excludes _common and any dir without one)."""
    out = []
    for d in sorted(SKILLS_DIR.iterdir()) if SKILLS_DIR.exists() else []:
        if d.is_dir() and not d.name.startswith("_") and (d / "SKILL.md").exists():
            out.append(d.name)
    return out


def _read_skill_md(skill: str) -> str:
    p = SKILLS_DIR / skill / "SKILL.md"
    if not p.exists():
        raise FileNotFoundError(f"no SKILL.md for skill '{skill}' at {p} (have: {', '.join(available_skills())})")
    return p.read_text(encoding="utf-8")


def _frontmatter_field(md: str, field: str) -> str:
    m = re.search(rf"^{re.escape(field)}:\s*(.+)$", md, re.M)
    return m.group(1).strip() if m else ""


def _body_after_frontmatter(md: str) -> str:
    """Return the SKILL.md body (after the closing '---' of the YAML frontmatter)."""
    parts = md.split("---", 2)
    return (parts[2] if len(parts) >= 3 else md).strip()


def _esc_braces(s: str) -> str:
    """Double every brace so the string survives the .format(domain=...) contract in Brain._decide_sys /
    _build_decide_sys (a lone { or } from a SKILL.md code snippet would otherwise crash str.format)."""
    return s.replace("{", "{{").replace("}", "}}")


# common non-ASCII -> ASCII so the lens is safe in EVERY consumer (cp1252 console print, the cli-subprocess
# transport, logs) -- the project's no-non-ASCII-in-prints invariant. Meaning is preserved; stragglers are dropped.
_ASCII_MAP = {
    "≥": ">=", "≤": "<=", "→": "->", "←": "<-", "×": "x", "…": "...",
    "–": "-", "—": "--", "‘": "'", "’": "'", "“": '"', "”": '"',
    "•": "*", "±": "+-", "≈": "~", "·": ".",
    # any remaining non-ASCII (emoji, rare glyphs) is dropped by the encode('ascii','ignore') fallback below.
}


def _ascii(s: str) -> str:
    for k, v in _ASCII_MAP.items():
        s = s.replace(k, v)
    return s.encode("ascii", "ignore").decode("ascii")


def load_skill_lens(skill: str, max_protocol_chars: int = 1800) -> dict:
    """Build the planning LENS for a skill from its SKILL.md: a `domain` (the expert identity, injected into every
    decide-system prompt) and a `plan_instruction` (the DECOMPOSITION approach: attack the objective the way THIS
    expert would, grounded in the skill's own protocols). These are the two brain seams that make the SAME objective
    decompose differently per skill."""
    md = _read_skill_md(skill)
    desc = _ascii(_frontmatter_field(md, "description") or f"the {skill} expert")
    body = _ascii(_body_after_frontmatter(md))
    persona = next((ln.strip() for ln in body.splitlines() if ln.strip()), desc)
    domain = f"the {skill.upper()} expert. {desc}"[:600]
    protocols = body[:max_protocol_chars]
    plan_instruction = _esc_braces(
        f"You are the {skill.upper()} expert. {persona}\n"
        f"DECOMPOSE the objective into frontier nodes THROUGH THE {skill.upper()} LENS -- attack it the way a {skill} "
        f"expert would, applying YOUR OWN domain protocols below, NOT a generic decomposition. Each node should be a "
        f"concrete step a {skill} expert would take, with a verify_cmd where the artifact can be mechanically checked.\n"
        f"--- {skill} protocols (your decomposition toolkit) ---\n{protocols}")
    return {"skill": skill, "domain": domain, "persona": persona, "plan_instruction": plan_instruction,
            "description": desc, "protocol_chars": len(protocols)}


def build_routed_brain(skill: str, backend: str = "mock", cwd: str | None = None):
    """Construct a brain whose PLANNING LENS is the skill's expertise (domain + plan_instruction injected)."""
    lens = load_skill_lens(skill)
    brain = make_brain(backend, domain=lens["domain"], cwd=cwd)
    brain.set_plan_instruction(lens["plan_instruction"])   # the skill's decomposition approach -> the plan node
    return brain, lens


def route(skill: str, objective: str, success: str, backend: str = "cli", budget: int = 12,
          parallel: int = 1, fill_window: bool = True, recursion_limit: int = 400, thread: str | None = None):
    """Hand the objective to the SHARED fill_window loop with the skill's lens. expert_mode + the .claude/agents
    persona dir let dispatch workers also attach the matching expert persona. Returns the final loop state."""
    brain, lens = build_routed_brain(skill, backend)
    app = build(brain, parallel=parallel, expert_mode=True,
                persona_dir=str(AGENTS_DIR) if AGENTS_DIR.exists() else None,
                channel=skill, fill_window=fill_window)
    tid = thread or f"skill-{skill}-{int(time.time())}"
    init = {"objective": objective, "success_criteria": success, "frontier": [],
            "budget": budget, "cycle": 0, "status": "running", "parallel": parallel, "run_id": tid,
            "ledger": [], "awaiting_approval": [], "drain_empty": 0, "stall_cycles": 0,
            "replan_count": 0, "done_count": 0}
    print(f"=== SKILL-ROUTED loop  skill={skill}  brain={brain.name}  backend={backend}  "
          f"fill_window={fill_window}  budget={budget} ===")
    print(f"    lens.domain: {lens['domain'][:120]}")
    return app.invoke(init, {"recursion_limit": recursion_limit, "configurable": {"thread_id": tid}})


def main(argv=None):
    ap = argparse.ArgumentParser(description="Route an expert skill's mandate through the shared metaop loop.")
    ap.add_argument("skill", help=f"skill name -- one of: {', '.join(available_skills())}")
    ap.add_argument("objective", nargs="?", default="", help="the one-line objective")
    ap.add_argument("success", nargs="?", default="", help="verifiable success criteria")
    ap.add_argument("--backend", default="cli", choices=["auto", "sdk", "mock", "cli", "api", "ollama", "cascade", "litellm"])
    ap.add_argument("--budget", type=int, default=12)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--no-fill-window", dest="fill_window", action="store_false",
                    help="disable the no-idle-stop window-filling (default ON for a skill-routed run)")
    ap.add_argument("--show-lens", action="store_true",
                    help="print the injected domain + plan_instruction for this skill and EXIT (no loop run)")
    a = ap.parse_args(argv)

    if a.show_lens:
        lens = load_skill_lens(a.skill)
        print(f"=== SKILL LENS: {a.skill} ===")
        print(f"[domain]\n{lens['domain']}\n")
        print(f"[plan_instruction]\n{lens['plan_instruction'][:1400]}\n...")
        return 0
    if not a.objective:
        ap.error("objective is required unless --show-lens is given")
    route(a.skill, a.objective, a.success or a.objective, backend=a.backend, budget=a.budget,
          parallel=a.parallel, fill_window=a.fill_window)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
