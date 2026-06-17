<!-- STAGING PLACEMENT (per Brain-Upgrade Guardrails 2026-06-05):
  trust_level: TRUST_CRITICAL (it is a governing directive)
  proposed_home: .claude/skills/_common/AF_BL_RF.md  — loaded ON-DEMAND by trader/discover/decide skills,
                 NOT pasted into the always-loaded CLAUDE.md. (19 always-on lenses would worsen the
                 instruction-budget bloat the audit identified — SOTA: ~150-instruction reliable ceiling.)
  review_status: PENDING
  provenance: WF1 (5-lens synthesis) wf_56a1df87-c72, 2026-06-05. Anchored to docs/APPARATUS_LOCKDOWN_SPEC,
              FOUNDATION_2026_06_04, RETEST_PLAN. -->

# META-DIRECTIVE: APPARATUS-FIRST, BEAT-LAZY, REPORT-FIRST (AF-BL-RF)

> On-demand governing directive for STRATEGY work. Composes with — does not replace — AUTONOMOUS_RUNNER,
> CDAP/the 6-layer trust-stack, the `decide` skill, and the report-first user preference.
> Trust-critical: edits require human review.

## North Star
Optimize for **WEALTH = robust held-out compound return** (NOT Sharpe, NOT IC) under LO+spot+lev=1,
ROBUST = 10/10 seeds positive on UNSEEN, block-bootstrap p05>0, maxDD<30%.

The governing constraint is **epistemic, not strategic**: *no number, verdict, sizing decision, or
self-edit is trustworthy until the instrument that produced it is independently verified sound — a
result from a broken instrument is not weak evidence, it is NO evidence.*

Sequencing is load-bearing and irreversible:
1. **Fix the apparatus FIRST** (LD-1..LD-5).
2. Prove the candidate **beats a cost-matched random-entry null AND a beta-matched static hold on a
   bear-inclusive holdout**.
3. **THEN** state + falsify the mechanism, compute capacity + ruin-path, and only then size.

A false-positive edge reaching real capital is **categorically worse** than a discarded real edge.
Every gate resolves toward **NULL** under genuine uncertainty.

## The Workhorse Pick
**Single-agent reflexion loop (AUTONOMOUS_RUNNER build→run→learn→pivot over an EV-ranked frontier),
WRAPPED by a scoped adversarial judge-panel.**

The panel (≥2 independent skills, majority vote) convenes **only on load-bearing events**: a number
entering a SHIP claim/leaderboard, a reversal/paradigm-shift, a dead-list re-open, or an apparatus
verdict. Its binding upgrade is an **independent re-deriver** — one slot recomputes the number from
raw-data + spec + apparatus-contract *without reading the original agent's output* (a reviewer inherits
framing errors; a blind re-deriver cannot), and one fixed slot is the **apparatus-trust role**.

- **Rationale:** multi-agent costs ~15x and ~40% of pilots fail on orchestration; single-agent is
  superior on tightly-coupled apparatus/strategy coding (most of the work). Reflexion-for-execution +
  panel-as-gate is exactly what produced the project's most trustworthy output (2026-06-04 BTC 1d R12:
  a lone Opus oracle over-reached on a reversal; a 2-skill audit+validator panel caught all four errors
  in one pass). The re-deriver structurally breaks the three-artifact inflation pipeline.
- **Honest failure modes:** (1) panel convened *prematurely* on the old apparatus → false-positive
  consensus → guard: apparatus-first is a hard precondition; (2) convened *too broadly* → reverts to the
  15x cost trap → guard: panel only on the four load-bearing events; (3) *shared blind spot* → guard: a
  permanent apparatus-trust slot + the known-answer CANARY suite.

## Routing Function
| Task shape | Pattern | Multi-agent? |
|---|---|---|
| Apparatus/harness/cost/loader/pipeline code; per-candidate eval; coupled multi-file edit | Single-agent reflexion loop | **WITHHELD** |
| A NUMBER entering a SHIP claim / leaderboard / headline | Independent re-deriver + apparatus-trust slot; >1pp/window disagreement → INVESTIGATION-GATE | Yes (scoped) |
| Load-bearing REVERSAL (kill/exhaust/proven/re-open/paradigm) | Adversarial judge-panel ≥2 skills, blind verdicts, reconcile | Yes (binding) |
| Breadth-first DISCOVERY / literature / SOTA / many candidates | Fan-out Sonnet scouts ≤9, gather; log USABLE/PARTIAL/WRONG | **Yes** (~90% better) |
| Promotion / deploy / capital / "overfit or beta-in-disguise?" | `decide` BULL/BEAR/NULL, default NULL under ambiguity | Yes |
| Lookup / single-file edit / status / chat | `normal` / direct | **No** |
| Honest diagnosis / open fork / user-owned call | Report-first single-agent synthesis; open decisions as plain enumeration | No |

## Always-On Lenses (the deduped union of corrections)
1. **Harvestability > Existence** — existence-proof and harvestability-proof are separate experiments; every edge node carries a cost-feasibility sub-node.
2. **Cadence-Feasibility Preflight** — ~5-min taker break-even arithmetic per (ASSET×TI×CADENCE) before mining; infeasible → already-dead with the math shown.
3. **Beta-Separation (Beat-Lazy)** — required beta-excess field per regime window; beats-static-only-in-bear = label drawdown-control beta, not the wealth objective.
4. **Cost-Matched Random-Entry Null (firewall)** — beat K matched-entry/holding/cost nulls on EVERY window (LD-4, THE primary gate; replaces no-op DSR-only).
5. **Bear-Inclusive Holdout (pre-registered)** — bear + bull window, pooling rule pre-registered before results (LD-5).
6. **Family-N includes Aggregation DoF** — `_sweep_manifest.json` with `n_variants_tested` × `n_aggregation_compositions_tried`; `n_trials = max(written, manifest)` (LD-3).
7. **Mechanism + Pre-Registered Falsifier** — "exists because X; dies if Y", Y testable, result a verifiable JSON field. No sizing before mechanism.
8. **Apparatus-State Tag on Every Number + Canary regression** — cost/p_fill/DSR-state/regime/git-SHA tags; 30-day known-answer canary; failed canary → all recent positives APPARATUS-UNVERIFIED.
9. **Capacity & Liquidity Degradation** — turnover at $10k/$50k/$200k vs bear-regime ADV, +1bp per 1% ADV; below taker baseline at $50k = toy.
10. **Ruin-Path + Kelly Conservatism** — DD duration (>60d below -15%), recovery multiple, consecutive-loss count; pre-deploy sizing memo at min(quarter-Kelly, 25%/signal).
11. **Survivorship Quantification** — current-87 vs full-at-the-time delta; report return ± survivorship_haircut_pp; haircut > alpha = noise.
12. **Base-Rate / Prior before Sweep** — prior <~5% (linear TI on efficient majors) caps budget at 1-2 confirmatory tests; redirect freed compute.
13. **Provenance-Anchored Write-Forward + Refutation-vs-Coverage** — git SHA + run-path + real-`date` VERIFIED timestamp on every write; THIN-REFUTATION (apparatus=BROKEN or coverage<30%) blocks only the same sub-variant.
14. **Opportunity-Cost + VOI Convergence** — at 25/50/75% ask "highest-EV work NOT in my frontier?"; when confirmed ≥3 ways with stable verdict, write an "evidence sufficient, decision is yours" handoff.
15. **Orchestration-Failure & Agent-Velocity Ledger** — log spawned-agents / USABLE-PARTIAL-WRONG / failure-mode / EV-per-wall-clock-min; adjust fan-out + model assignment after 10 entries or 3 same-type failures.
16. **Structural-Break / Decay Monitor** — CONFIRMED carries last-verified timestamp + decay trigger (60-day rolling OOS Sharpe < 0.5 → SUSPECT + re-verify).
17. **Indicator Look-Ahead graduation** — known-delayed columns must run +1-bar-shifted; promoted candidates carry `leak_probe_status:PENDING_SUPERVISED` until the shift-spectrum probe graduates.
18. **Pre-Registered Search Protocol** — JSON sidecar before any sweep enumerating every axis + grid size; CDAP rejects SHIP claims whose family-N < the protocol product.
19. **Staging-Area Lifecycle** — header (date_staged/trigger/trust_level/review_status/expiry); PushNotify PENDING TRUST_CRITICAL >7 days.

## Stop Conditions
- **IDLE-STOP:** frontier empty or all nodes below floor → park with wake-conditions and STOP.
- **VALUE-FLOOR:** below-floor nodes enter already-closed; ≥3 cycles below floor → IDLE-STOP or +k/−k lattice-expand.
- **REFUTATION (monotonic):** REFUTED under a sound apparatus with ≥30% coverage = never re-mined; under broken apparatus or <30% coverage = RE-TEST-REQUIRED.
- **COST-INFEASIBLE:** taker break-even > observed move distribution → stop before any experiment.
- **VOI-EXHAUSTED:** confirmed ≥3 independent ways, stable verdict → stop analyzing, hand off to user.
- **ASYMMETRIC-LOSS NULL default:** unresolved conflict → conservative verdict stands, candidate held at NULL.

## Provenance
Synthesis of 5 expert lenses (Trading-Risk-Capital, Adversarial-Auditor-Epistemics, ML-Research-Validation,
Systems-Orchestration, Decision-Science-Metacognition), all of which independently selected
adversarial-verify/judge-panel + apparatus-first. Clock-grounded: `date` Fri Jun 5 2026 SAST.
