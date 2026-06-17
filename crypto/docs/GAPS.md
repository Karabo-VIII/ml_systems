# GAPS — open issues register

CLAUDE.md §13/§14 (DOUBLE_AUDIT_PROTOCOL) file 🟠 HIGH / 🟡 MEDIUM findings here.
🔴 CRITICAL halts the commit and is fixed inline, not filed. This file was created
2026-06-06 (it was referenced by CLAUDE.md but did not exist — a dangling ref, now closed).

## OPEN

### 🟠 HIGH — stale `src/strategy/` paths in CDAP registry + trader skill (P2)
The 2026-06-05 reset archived `src/strategy/` → `archive/strategy/` and rebuilt the layer
at `src/strat/` + `src/wealth_bot/bot/`, but the registry + docs were not repointed.
- `config/_invariants.yaml`: ~6 registered invariants point `files:` at non-existent paths
  (`src/strategy/risk_controller.py`, `src/strategy/sleeves/...`, etc.). Now WARN-surfaced
  by the 2026-06-06 silent-no-op guard fix (commit 8b5f41a).
- `.claude/skills/trader/`: `DAILY_OPS.md:108`, `PRE_DEPLOY_CHECKLIST.md:20,80`,
  `RISK_PLAYBOOK.md:74,153`, `SIZING_THEORY.md:19` cite the dead path, contradicting
  `SKILL.md:59`'s own note that the layer moved.
- Verified current homes: kill-switches/position_sizer → `src/wealth_bot/bot/`; regime →
  `src/wealth_bot/regime_router/`; cost calib → `config/maker_cost_calibration.yaml`;
  DSR → caller contract in `src/strat/battery.py`; `risk_controller.py` NOT BUILT.
- FIX: repoint + re-escalate kill-switch/decay rules to their declared `critical` once the
  canonical post-reset risk apparatus is decided. Attended (changes trust-stack behavior).

### 🟠 HIGH — CLAUDE.md cites 3 dead binding-contract spec homes (P1b, attended)
RWYB-verified MISSING 2026-06-06:
- `CLAUDE.md:49` → `.claude/skills/_common/OPERATIONAL_DIRECTIVES.md` (universal self-audit spec).
- `CLAUDE.md:49` → `.claude/skills/maxx/SKILL.md` (per-commit gate spec; **no `maxx` skill dir exists**).
- `CLAUDE.md:71` → `memory/feedback_run_what_you_build.md` (RWYB Layer-1 rule; archived at reset).
- FIX: decide restore-from-`archive/restart_2026_06_04/` vs re-home vs rewrite-as-post-reset.
  Do NOT rush — CLAUDE.md is the highest-blast-radius control surface.

### 🟡 MEDIUM — other dangling refs (recon-confirmed 2026-06-06)
`STATE.md:31/34` dossier + coordination docs (archived), `CLAUDE.md` `docs/V50_TO_V51_FIXES.md`
(archived). Low-stakes; repoint or strike when next editing those files.

## RESOLVED
- 2026-06-06 (8b5f41a) — CDAP `must_match` invariants silently no-op'd on missing-file
  globs (false-clean gate). Empty-glob guard added to `check_invariants.py`. 8 silent
  invariants surfaced. Full report: `runs/autonomy/FINDINGS_skill_directive_audit_2026_06_06.md`.
- 2026-06-06 — recon false positive struck: `docs/BROWSER_DIRECTIVE.md` EXISTS.
