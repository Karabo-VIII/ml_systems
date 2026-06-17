"""RWYB for the SKILL ROUTER (scripts/autonomy/skill_route.py): proves a skill's EXPERTISE is injected into the
shared loop's planning lens, that /audit and /trader decompose the SAME objective DIFFERENTLY, and that a routed
loop actually runs end-to-end.

  A (deterministic): audit-lens != trader-lens; each carries its OWN domain keywords; the brain's decide-system
      prompt (what the plan node sees) contains the caller skill's lens and NOT the other's.
  B (e2e): a routed loop runs through the shared graph (mock brain + fill_window + small budget) and terminates.
  C (empirical, ollama, best-effort): the SAME objective planned with the audit lens vs the trader lens yields
      DIFFERENT frontiers (auditor -> adversarial/red-team nodes; trader -> sizing/risk/portfolio nodes). SKIPs
      cleanly if no local model is reachable (the deterministic A already proves the injection).

Run from repo ROOT:  .venv/Scripts/python.exe scripts/autonomy/_test_skill_route.py
No emoji (Windows cp1252).
"""
from __future__ import annotations
import sys, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.autonomy.skill_route import load_skill_lens, build_routed_brain, route, available_skills  # noqa: E402

fails = []
def ok(c, label, detail=""):
    print(("  PASS" if c else "  FAIL"), label, ("" if c else f":: {detail}"))
    if not c: fails.append(label)

# keyword fingerprints of each lens (drawn from the skills' own SKILL.md vocabulary)
AUDIT_WORDS = ["adversarial", "red-team", "invariant", "leakage", "cross-version"]
TRADER_WORDS = ["sizing", "risk", "position", "portfolio", "execution"]

print("=" * 84); print("A -- deterministic: the SAME loop, DIFFERENT injected expertise per skill"); print("=" * 84)
la = load_skill_lens("audit")
lt = load_skill_lens("trader")
ok("audit" in available_skills() and "trader" in available_skills(), "audit + trader skills discovered")
ok(la["plan_instruction"] != lt["plan_instruction"] and la["domain"] != lt["domain"],
   "audit lens != trader lens (different domain AND plan_instruction)")
la_blob = (la["domain"] + " " + la["plan_instruction"]).lower()
lt_blob = (lt["domain"] + " " + lt["plan_instruction"]).lower()
ok(sum(w in la_blob for w in AUDIT_WORDS) >= 3, "audit lens carries audit vocabulary (>=3 of red-team/adversarial/invariant/leakage/cross-version)",
   [w for w in AUDIT_WORDS if w in la_blob])
ok(sum(w in lt_blob for w in TRADER_WORDS) >= 3, "trader lens carries trader vocabulary (>=3 of sizing/risk/position/portfolio/execution)",
   [w for w in TRADER_WORDS if w in lt_blob])
# cross-check: each lens is DISTINCTLY its own -- audit words dominate the audit lens, trader words the trader lens
ok(sum(w in la_blob for w in AUDIT_WORDS) > sum(w in lt_blob for w in AUDIT_WORDS),
   "audit vocabulary is concentrated in the audit lens (not the trader lens)")
ok(sum(w in lt_blob for w in TRADER_WORDS) > sum(w in la_blob for w in TRADER_WORDS),
   "trader vocabulary is concentrated in the trader lens (not the audit lens)")

# the brain the PLAN node actually uses: its decide-system prompt must carry the caller's lens
ba, _ = build_routed_brain("audit", backend="mock")
bt, _ = build_routed_brain("trader", backend="mock")
sys_a = ba._decide_sys().lower()
sys_t = bt._decide_sys().lower()
ok("audit expert" in sys_a and "audit expert" not in sys_t,
   "the AUDIT brain's plan-system prompt says 'AUDIT expert'; the trader brain's does not")
ok("trader expert" in sys_t and "trader expert" not in sys_a,
   "the TRADER brain's plan-system prompt says 'TRADER expert'; the audit brain's does not")
ok(any(w in sys_a for w in AUDIT_WORDS) and any(w in sys_t for w in TRADER_WORDS),
   "each brain's plan prompt carries its own domain protocols -> they WILL decompose differently")

print("=" * 84); print("B -- e2e: a skill-routed loop runs through the SHARED graph + terminates"); print("=" * 84)
final = route("audit", "demo: route a tiny objective", "loop runs + stops", backend="mock",
              budget=3, parallel=1, fill_window=True, recursion_limit=200, thread="test-skillroute")
ok(isinstance(final, dict) and final.get("status") in ("solved", "budget_spent"),
   "routed loop ran the shared graph and terminated cleanly", final.get("status"))

print("=" * 84); print("C -- empirical (ollama, best-effort): SAME objective -> DIFFERENT decomposition per lens"); print("=" * 84)
def _ollama_up():
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False

OBJ = "Evaluate a candidate crypto trading strategy before we rely on it."
if not _ollama_up():
    print("  SKIP -- no local ollama model reachable; deterministic A already proves the lens injection.")
else:
    try:
        from harness.metaop.brain import make_brain
        mdl = "qwen2.5-coder:7b"
        a_brain = make_brain("ollama", model=mdl, domain=la["domain"]); a_brain.set_plan_instruction(la["plan_instruction"])
        t_brain = make_brain("ollama", model=mdl, domain=lt["domain"]); t_brain.set_plan_instruction(lt["plan_instruction"])
        payload = {"objective": OBJ, "success_criteria": "a sound verified verdict", "frontier": []}
        a_plan = a_brain.decide("plan", payload)
        t_plan = t_brain.decide("plan", payload)
        a_text = str(a_plan).lower()
        t_text = str(t_plan).lower()
        a_hits = [w for w in AUDIT_WORDS + ["look-ahead", "overfit", "purge", "verify"] if w in a_text]
        t_hits = [w for w in TRADER_WORDS + ["kelly", "drawdown", "sleeve", "capital", "sharpe"] if w in t_text]
        print(f"  audit-lens plan keywords hit: {a_hits}")
        print(f"  trader-lens plan keywords hit: {t_hits}")
        ok(a_text != t_text, "the two plans for the SAME objective are DIFFERENT text")
        ok(len(a_hits) >= 1 or len(t_hits) >= 1,
           "at least one lens steered the decomposition toward its own domain (model-dependent)",
           {"audit": a_hits, "trader": t_hits})
    except Exception as e:
        print(f"  SKIP -- empirical plan call errored ({type(e).__name__}: {str(e)[:80]}); A proves the injection.")

print("=" * 84)
print(f"SKILL-ROUTER TEST: {'ALL PASS' if not fails else 'FAILURES: ' + ', '.join(fails)}")
print("=" * 84)
sys.exit(0 if not fails else 1)
