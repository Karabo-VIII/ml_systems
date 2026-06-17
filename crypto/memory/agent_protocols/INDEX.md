# Agent Protocols Index

> Layer-3 protocols that extend skill behavior. Referenced by
> `.claude/skills/LAYER2_UNCONSTRAINED.md`. Each protocol is ADDITIVE — never
> overrides Layer-0/1/2 rules. Read this index at session start; load specific
> protocols on demand.

## 🔴 Top-priority warnings

**WEB TOOLS ARE FIRST-CLASS — load at session item-1.** Per user directive
2026-05-14 ("110% capability unlocked"). Bootstrap via
`ToolSearch("select:WebSearch,WebFetch")` BEFORE any Read/Grep/Bash work.
Web tools (WebSearch + WebFetch) behave identically to out-of-the-box
Opus where they're always available. Use proactively for every factual
claim, API spec, 2024-2026 SOTA citation, market-state check. Failure to
bootstrap is a Layer-2 violation. Past sessions routinely under-complied
with this directive — enforcement strengthened 2026-05-14 in:
- `.claude/skills/LAYER2_UNCONSTRAINED.md` (top + §"Amplified Capabilities")
- All 14 `.claude/skills/*/SKILL.md` files (header banner)
- `CLAUDE.md` (project invariant)

**SONNET sub-agents produce sub-par results in non-trivial work.** Per user
directive 2026-05-13. They are READ-ONLY scouts whose output must be
VERIFIED by Opus before any action. Never trust Sonnet:
- Numeric claims (verify against source files)
- Architectural recommendations (verify against actual code)
- "Done" / "complete" verdicts (verify with `py_compile` + smoke test)
- Bug findings (Opus reproduces, then fixes)
- Cross-file pattern claims (Opus runs `grep` to confirm)

Use Sonnet for BREADTH (parallel file enumeration, multi-source literature
sweep, large-surface inventory). Never use Sonnet for the DECISION or the
CODE CHANGE. See `sonnet_integration_safety.md`.

---

## Protocols (mandatory when trigger fires)

| Protocol | File | Trigger |
|---|---|---|
| Pre-action adversarial debate | [pre_action_debate.md](pre_action_debate.md) | Any diff touching >5 files OR strategy/training/cost-model/DAG code OR irreversible op |
| Calibrated uncertainty | [calibrated_uncertainty.md](calibrated_uncertainty.md) | EVERY numerical claim in user-facing output |
| Cross-instance awareness | [cross_instance_awareness.md](cross_instance_awareness.md) | Session start + every ~30 min of work |
| Constraint propagation | [constraint_propagation.md](constraint_propagation.md) | User changes any constraint, scope, target, or invariant |
| Test-first | [test_first.md](test_first.md) | Non-trivial code work (sleeve adapters, ML training, pipeline builders) |
| Sonnet integration safety | [sonnet_integration_safety.md](sonnet_integration_safety.md) | Before AND after any Sonnet sub-agent invocation |
| Fix-log read before edit | [fix_log_read_before_edit.md](fix_log_read_before_edit.md) | Before editing any model version OR strategy sleeve |

## Existing related protocols (NOT duplicated here)

| Protocol | Where | Scope |
|---|---|---|
| 4-agent research delegation (2 Sonnet → Oracle → Auditor → Opus) | `memory/research_delegation_protocol.md` | High-blast-radius architecture decisions |
| CDAP pre-commit invariant audit | `src/audit/check_invariants.py` | Every commit |
| RED TEAM double-audit | `docs/DOUBLE_AUDIT_PROTOCOL.md` | Post-commit |
| Browser directive (A/B/C gates) | `docs/BROWSER_DIRECTIVE.md` | Every @browser-tagged prompt |
| Auto-memory schema | inline in skills | Continuous |

## When protocols conflict

Order of precedence:
1. CLAUDE.md hard invariants (cannot violate)
2. LAYER2_UNCONSTRAINED.md (universal amplifier)
3. Skill-specific SKILL.md rules
4. These protocols (additive)
5. Memory feedback entries

If a Layer-4 protocol conflicts with Layer-1 (skill rules) — Layer-1 wins. If
within Layer-4 protocols conflict — pre_action_debate.md wins (it's the
adversary gate).

## Maintenance

- Add new protocols when a recurring failure mode is identified
- Mark protocols deprecated (not deleted) if they prove counterproductive
- Each protocol must have: trigger, steps, verification, escalation path
- Each protocol file is ~80-150 lines max
