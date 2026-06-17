# The Brain Operating Model — the hardened modus operandi (so the user stops re-steering)

> **Mandate (user, 2026-06-06):** *"Go through all chats of all instances exhaustively + external research; build a
> modus operandi of skills/directives/harnesses; harden + make each robust. The BRAIN of the project (Claude), not the
> strat/WM layers. I can't be correcting and steering over and over. Give me an operational design that — when
> implemented + operated through — yields the frontier + SOTA outcomes."* Built by mining **1,411 cross-instance user
> turns** + a RWYB audit of all skills/directives/harnesses + external SOTA research (Workflow `we03sta4w`, 9 agents);
> **overseer-judged (RWYB-verified the load-bearing claims).**

## The core principle — corrections become MECHANISMS, not reminders

You keep re-steering because the rules are **prose that drifts and gets forgotten**. The fix: **every correction you've
had to make more than once becomes a gate/check/hook a future instance cannot skip.** Prose is advisory; a mechanism is
binding. This is the one idea that makes "you stop steering" real.

## The 8 most-repeated corrections → their mechanical fix (RWYB-mined, with provenance)

| # | recurring correction (times) | mechanical fix (the binding mechanism) |
|---|---|---|
| 1 | **Honest reporting / no inflated-or-mismatched numbers** (8+ in one stretch; the cause of every trust-stack layer: +468%, +501%, the +120% ceiling, the `btc_ret_same_day` t+1 leak) | `src/audit/check_report_claims.py` as a **CDAP sub-check + PreToolUse on Write/Edit to report `*.md`**: every ROI/Sharpe/return number must carry a claim-tag (VERIFIED/REPORTED/INFERRED) + reconcile backtest==simulator before quoting |
| 2 | **Premature "done/concluded" while gaps open** (pushed back repeatedly; `/un` "ship or concede" re-invoked 15+×) | **Stop-hook closure-block**: any response claiming done/complete must emit a machine-checkable `{open_CRITICALs, gaps_with_plan, unverified_claims}` block or it's not a stop |
| 3 | **Wall-clock / elapsed-time fabrication** (canonical 2026-06-03: claimed "~5h in", real ~1h12m, + a fabricated ledger) | **WIRE `.claude/hooks/intent_clock.py`** (BUILT but unwired — RWYB-verified absent from settings.json) into UserPromptSubmit → a VERIFIED `date` injected every turn |
| 4 | **Train-vs-OOS framing slip / leakage** (corrected 3× consecutively; "don't touch unseen") | **`split_discipline` CDAP invariant**: discovery/fit runs must declare `{fit_window, eval_window}`; HALT (exit 2) if fit touches unseen or eval overlaps train |
| 5 | **Silent failures / exit-0-on-partial** ("end to end" ×51; Phase2 silent-drop-72/87, rc=0 on empty feeds) | Make **`dispatch.run_per_task` the MANDATORY producer wrapper** (CDAP invariant) — it already `sys.exit(2)`-on-zero-ok + captures per-task errors; + per-stage USABLE assertion (row-floor + required-cols non-null) before DAG advance |
| 6 | **Deferring the call / parking on "awaiting review"** ("you're the expert, only consult on blockers" ×4) | **Stop-hook ownership clause**: a response ending in a question/park-state on a git-revertible action within an open envelope → converted to a decision-and-act |
| 7 | **Non-PowerShell shell syntax on a PowerShell box** (`for f in…` ParserError; `tail` not found) | **PostToolUse guard on Bash** (detect `for-in`/`tail`/`/dev/null`/backtick) + `scripts/check_docs_shell.py` CDAP-warn on POSIX-only docs |
| 8 | **Slow / non-parallel / resource-underutilizing runs** (155 speed mentions; babysitting stuck procs) | Promote the **`loop_health.py` liveness FSM** (WRITING-OK / SLOWNESS / HUNG via dual checkpoint+lane stall) — every long run goes through it + a progress heartbeat |

## The operating model (every instance, every query)

`ORCHESTRATE (thin) → DISPATCH to a durable solver (never self-execute the bulk) → MONITOR (the 60s liveness FSM) →
JUDGE adversarially (RWYB; refuse false victory) → COMMIT (git is the revert net)`. The instance owns the *plan +
judgment*; the work + state live in **external durable state** (frontier + checkpoints + files). See
[ORCHESTRATOR_ARCHITECTURE.md](ORCHESTRATOR_ARCHITECTURE.md).

## The hardening plan (EV-ranked — the frontier; built incrementally, overseer-judged)
1. `[S]` WIRE `intent_clock.py` → settings.json (verified unwired) — wall-clock fabrication
2. `[S]` Seed the 2 dead read-forward lanes (`memory/trader/post_mortems.md`, oracle) — redoing-done-work
3. `[M]` PreCompact hook snapshots rolling state (CONSTRAINTs/OPEN_Qs/CRITICALs) — state-drift on compaction
4. `[M]` `check_report_claims.py` (CDAP + PreToolUse) — honest-reporting (#1)
5. `[M]` Stop-hook closure-block — false-victory / premature done
6. `[M]` `split_discipline` CDAP invariant — train-vs-OOS leakage
7. `[S]` Stop-hook ownership clause — deferring-the-call
8. `[M]` Promote `DIRECTIVES_REGISTER` to one-row-per-rule with `ENFORCED-BY|GAP` + `check_register_coverage` — fragmentation/one-source-of-truth
9. `[M]` `dispatch.run_per_task` mandatory wrapper + USABLE assertion — silent failures
10. `[S]` PowerShell-syntax guard (PostToolUse) + `check_docs_shell.py` — non-PowerShell
11. `[S]` Document the narrate CLI surface — skill contract mismatch
12. `[S]` Name the `skill_library` consolidation owner — unowned-registry governance

## Reliability disciplines — the honest version of "no failures allowed"

Zero failures is impossible (claiming it would be the false-victory you've taught me to refuse). The achievable,
rigorous target is **failures that are rare, caught fast, and NEVER REPEATED** — enforced by: **RWYB** (verify by
running, never assert), **adversarial self-audit** before delivery, **two-sided verification** (a gate must accept a
positive control, not just reject), the **liveness FSM** (the watcher), and the **mechanism-per-recurring-correction**
rule above. Each new repeated correction adds a new mechanism (monotonic) — the brain hardens itself over time.

*Mining: `runs/staging/brain_audit/user_mandate_corpus.md` (1,411 turns). Audit + research + synthesis: Workflow
`we03sta4w` (9 agents, 1M tokens). RWYB-verified: intent_clock unwired, dispatch.run_per_task + loop_health exist.*
