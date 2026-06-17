# Shared Standards (slim _common, 2026-05-28)

Every curated skill references this file instead of the old ~1,500-line ceremony
layer (PROTOCOL_COMPOSITION / DIMENSION_SURFACE / SOLVER), all archived 2026-05-30
to `archive/skills/_common/` — PROTOCOL_COMPOSITION's surviving rules were folded
into `AUTONOMOUS_RUNNER.md` (§4 cross-skill summoning + the value-needle test).
Keep it short. CLAUDE.md + MEMORY.md are auto-loaded — do not duplicate their
content here.

## Standing rules (apply on every invocation)

1. **Real capital is at stake.** No academic answers. Every recommendation/code
   change must survive contact with real markets. If it wouldn't, don't ship it.
2. **Read before editing.** Never modify code you haven't read — even "obvious" edits.
3. **Run what you build (RWYB).** Every code change runs against real data before
   commit. Document the run command + result in the commit body. No silent failures.
4. **Verify after changing.** `py_compile` at minimum; smoke-test the changed code
   path (a test that imports but never calls the change is not a test).
5. **Honesty / no inflation.** Before reporting any number: check for look-ahead,
   K-selection on future returns, MtM double-count, compound-math drift, leverage
   drift, survivorship, multi-testing. Report robustness (DD, p05, n_eff) alongside
   returns. Single-seed ML claims are unverified — require N≥10-seed median + p05/p95.
6. **Self-audit before delivering anything** (commit OR analysis/recommendation):
   claim-tagging (VERIFIED / REPORTED / INFERRED), look-ahead check, gate-spec
   consistency, repro block (git SHA + chimera SHA + seeds + config) on ship claims.
7. **Surface 🔴 CRITICAL inline** as a ≤1-sentence caveat. Run the red-team check
   mentally; do NOT print a visible red-team section unless asked.
8. **No emoji in Python print/log statements** (Windows cp1252 crashes).
9. **WEALTH not Sharpe.** Optimize compound return under the robustness CONSTRAINT
   (10/10 seeds positive on UNSEEN, p05 > 0, max DD < 30%). Sharpe is tiebreak only.
10. **LO + SPOT + LEV=1** is a hard North Star bound. Any deviation = automatic reject.
11. **CDAP gate.** Commits run `python src/audit/check_invariants.py` (exit 2 = block).
    Fix the finding; bypass only with `SKIP_CDAP=1 SKIP_CDAP_REASON='...≥20 chars...'`
    and document the reason in the commit body.
12. **Cross-version / caller propagation.** When changing a signature, constant, or
    schema, grep ALL callers/siblings and update every one. Sibling-skip is the #1
    silent-failure source (see CLAUDE.md Cross-Version Training Invariants).
13. **Non-linear, self-improving operation** (binding on every skill, 2026-06-04 user mandate).
    Work the problem as a re-rankable frontier, not a fixed plan: loop back, re-frame, and
    re-sequence the moment a result demands it (the AUTONOMOUS_RUNNER build→run→learn→pivot
    lattice is the canonical shape; the posture applies to EVERY invocation, timed or not).
    Experience compounds — READ-FORWARD memory + dead-list + reusable-asset register at start,
    WRITE-FORWARD every learning as it happens, never re-pay for a lesson already learned. Treat
    ALL prior conclusions (incl. archived "dead"/"works" verdicts) as HYPOTHESES to verify under
    the current sound apparatus, never inherited facts; a verdict reached under a since-fixed
    apparatus bug is re-opened. When a real, recurring capability gap appears, propose/create a
    sub-skill or tool rather than re-improvising — improving the project's own machinery (skills,
    directives, artifacts, the act→observe→feedback loop) is a first-class deliverable.

## Wall-clock grounding

Any wall-clock claim (date, ETA, "X days ago", "last modified") must be grounded:
run `date` at turn start when it matters, and tag the claim VERIFIED (checked now),
REPORTED (from a file/log), or INFERRED (derived). Session-mid date rollover is the
canonical trigger for silent drift — re-check `date` if a session spans midnight.

## Delegation budget (hard limits, all skills)

- ≤2 Opus sub-agents in parallel; ≤9 Sonnet sub-agents in parallel; +1 META (you);
  ≤12 concurrent at any moment. Serial chains are unbounded.
- Sub-agent model is EXPLICIT every spawn (`model="opus"` or `model="sonnet"`). No Haiku.
- Prefer direct Grep/Glob/Read over spawning agents for simple lookups.
- VERIFY every Sonnet output against actual code before acting (agents hallucinate).

## Escalation

`normal` handles simple/direct work. Escalate to `apex` or `audit` review when a
task touches >3 files, crosses expert domains, would take >10 min to do right, involves
cross-version propagation, or modifies a CLAUDE.md invariant. High-stakes claims
(promotion, deploy, resource allocation) route to `decide` (dialectic). Pre-commit
review of strategy/training/cost-model diffs or ≥5-file diffs routes to `audit`.
