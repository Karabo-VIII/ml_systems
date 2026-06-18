# Autonomous Runner (lean) — n±k Objective Neighborhood + build→run→learn→pivot

Invoked when a directive grants an autonomy mandate ("go for N hours", "no consult",
"agentic work", `/loop`, `/schedule`). Composes with `STANDARDS.md` — every standing rule
(RWYB, honesty/no-inflation, self-audit, wall-clock, delegation budget) still applies.

**Goal of this protocol:** solve the primary objective AND its high-value neighborhood, while
(a) keeping the non-linear "pivot when you learn something" strength, and (b) never idling once
the primary is exhausted. Keep it lean — this is a checklist, not ceremony.

## 1. Turn-1 — MAP the objective neighborhood (divergence)

Define **n** = the primary objective: what success concretely looks like + how it is measured.
Then map the NEIGHBORHOOD as a small lattice — two axes, both valences:

```
direction:  -k = FOUNDATIONAL  (prerequisites / assumptions / the method+data that produce n)
            +k = DERIVED        (consequences / the general class n is an instance of / adjacent uses)
valence:    +  = OPPORTUNITY    (an adjacent win to capture)
            -  = FALSIFIER      (how n is wrong / breaks / costs — the red-team objective)

        |  foundational (-k)                       |  derived (+k)
  ±1    | -: is the METHOD / DATA sound? (falsify) | +: adjacent solutions that fall out of n
        | +: reusable engines / harness / battery  | -: does n survive cost / regime / held-out?
  ±2    | -: is the PARADIGM / SPEC mis-framed?    | +: the GENERAL principle / class n belongs to
        | +: a better METHODOLOGY                  | -: even-if-it-works systemic fragility
```

The two **−k falsifier** nodes (is the method sound? is the spec mis-framed?) are FIRST-CLASS,
not afterthoughts — the highest-value finds usually live there. (Worked example, 2026-05-29 run:
"positive-in-every-window long-only-spot is an incoherent spec" was an n−2 falsifier, worth more
than any single strategy found.) The **+k** nodes are the solution-space expanders (generalize,
reuse); the **−k** nodes are the discovery expanders (find where you're wrong). Both matter.

Then set **GOAL_BOUNDS**: time/token budget; a per-node **VALUE FLOOR** (min EV to keep working a
node); wall-clock anchor (`date`); explicit stop conditions. Outward/irreversible ops still need
user OK (STANDARDS) even under autonomy.

> **⏱ VERIFIED-TIME (BINDING — per [`WALL_CLOCK.md`](WALL_CLOCK.md); 2026-06-03).** A timed mandate records its
> **VERIFIED start** with one `date` call up front, and EVERY later "elapsed / X-hours-in / time-left / am-I-near-
> budget" figure is `(fresh date) − start` with the subtraction shown — you have **no internal clock**, felt time
> and work-volume are NOT time. **Re-`date` at the start of each build→run→learn cycle and before any progress
> report.** Every learnings-ledger entry's timestamp is a real `date` reading at that moment, or omit the time
> (write "C7", not an invented "C7 (19:40)"). The 25/50/75%-budget checkpoints (§4) are computed from VERIFIED
> elapsed, never estimated. Incident: 2026-06-03 an instance claimed "~5h in" at ~1h12m and fabricated a whole
> ledger of timestamps — which also biased the strategy (false "frontier exhausted, little time left").

## 2. Execute — build→run→learn→pivot (the cycle, canonical)

Maintain an **EV-ranked FRONTIER** (a live priority queue of open nodes; seed it from the lattice).
Each cycle:

- **BUILD** — the minimal artifact to test the current top-EV node. Reuse before writing.
- **RUN** — on real data, honest (RWYB; look-ahead / survivorship / multi-test checks; shuffle-null
  + held-out for any edge claim; select on TRAIN+VAL, confirm OOS+UNSEEN once). Active-monitor long
  jobs (poll the output file; do not block on completion notifications, which can hang).
- **LEARN** — update beliefs: mark the node CONFIRM / REFUTE / EXHAUST. Push newly-discovered
  neighbors INTO the lattice + frontier (learning expands the map). Re-rank the frontier by EV.
- **PIVOT** — move to the next top-EV node when: the node is refuted/exhausted, OR its marginal
  value-per-cycle drops below the next node's EV (diminishing returns), OR a higher-EV neighbor
  appeared. Non-linear pivoting is EXPECTED; the frontier is what makes it principled, not lucky.

**Value-needle test (every cycle):** did this cycle move the PRIMARY needle or a high-EV neighbor's?
If three cycles in a row only move low-value nodes → you are over-mining; pivot UP an order (to a
+2 generalization or a −2 reframe) or stop.

## 3. Exhaustion / idle / STOP (fixes the time-loop inefficiency)

- A node **BLOCKED** on external state (a rebuild, CI, a deploy) is **PARKED** with a wake-condition
  — removed from the active frontier, not ticked on. Work other frontier nodes meanwhile.
- **STOP (do not reschedule)** when the frontier is empty (all nodes confirmed/refuted/parked) OR
  every remaining node is below the value floor. "Use the remaining time" is NOT an objective —
  burning the clock to fill hours is the over-mining / multiple-comparisons trap this project
  diagnosed. **Honest early-stop > busywork.**
  - **TIMED-RUN OVERRIDE (user iron-clad 2026-06-02, [[feedback_no_sleep_autonomous]]):** when the
    mandate is TIMED ("go for N hours", an explicit clock budget), do NOT idle-stop or sleep — keep
    working GENUINELY-valuable nodes until the budget is spent. The reconciliation with "no busywork":
    when the primary vein is exhausted, PIVOT UP an order (a +2 generalization, a −2 reframe) or to a
    sibling objective, AND when compute is saturated do NON-compute work in PARALLEL (WebSearch /
    read-only scout agents / build the next experiment / write the deliverable / checkpoint memory).
    Honest-stop still applies to UNtimed mandates and to fabricating fake nodes — never manufacture a
    result to fill time; widen the frontier instead. (This composes with §5: read-forward keeps the
    lattice genuinely open so a timed run never has to choose between idling and busywork.)
- **One PushNotification per STATE change only:** newly blocked on a user decision, ending the loop,
  or a major update that changes the plan. Progress you made yourself is not a trigger.

## 4. Checkpoints

- Self-summon a second opinion (`decide`) at ~25/50/75% of budget, on a posterior plateau,
  or on whiplash (a pivot that reverses a prior conclusion).
- Capture durable findings to `crypto/docs/` + memory BEFORE context risk — not only at the end.

## 5. Self-improving loop — experience compounds across cycles AND sessions

§2's build→run→learn→pivot learns WITHIN a run; this is the loop that makes each cycle, and each new
session, START smarter than the last. The one rule: **never re-pay for a lesson already learned.**

- **READ-FORWARD at every start.** Before mapping the lattice (§1), load prior experience: memory
  (`MEMORY.md` + relevant topic notes), the dead-list / failure catalog (what's REFUTED — do NOT
  re-mine), the reusable-asset register (harnesses / batteries / tools already built — reuse, don't
  rebuild), and the live frontier doc. The lattice is SEEDED from accumulated knowledge, not a blank
  slate; a node already marked REFUTED enters the frontier already-dead (a guardrail), not fresh work.
  (This session: the `returns-signal-frontier-exhausted` note + dead-list is exactly what stopped a
  re-mine and let the run jump straight to the de-inflation + pivot.)
- **WRITE-FORWARD every cycle.** Each LEARN appends to the durable record AS IT HAPPENS — a REFUTE →
  the dead-list WITH the falsifier that killed it (so it can't silently return); a CONFIRM → a reusable
  asset + the honest number; a new neighbor → the frontier. The next cycle reads this; the posterior
  only moves forward.
- **FEEDBACK IS A FIRST-CLASS INPUT.** User corrections, oracle / audit / red-team findings are folded
  into the operating model immediately AND persisted (a memory feedback-note, or a CLAUDE.md invariant
  if standing). A reframe that redirects the run (e.g. "returns, not alpha") becomes a permanent lens,
  not a one-off.
- **MONOTONIC GUARANTEE.** The system gets strictly smarter: refuted veins stay closed, validated tools
  accumulate, the honest-number ledger tightens. If a later cycle CONTRADICTS a prior conclusion
  (whiplash), that is a checkpoint (§4) — reconcile and update the record, don't silently flip. The
  durable memory IS the improving agent; any one session is just its current step.

## 6. Consensus + claim-integrity guards (added 2026-06-04 from a live self-eval)

Three refinements surfaced by a 2026-06-04 run in which a single load-bearing oracle over-reached
(cherry-picked the bearish evidence, ignored a contradicting result in the same archive, laundered beta
as "alpha", and inherited archived conclusions in violation of the session's reset) — and a 2-skill
consensus panel (audit + validator) caught all of it. The act→observe→feedback lesson, now binding:

- **WHIPLASH → MANDATORY CONSENSUS (not discretionary).** Any single agent verdict that REVERSES a prior
  conclusion or recommends an architecture/paradigm change is NOT adopted on its own word. Convene a
  consensus panel of ≥2 INDEPENDENT skills (e.g. `audit` + `validator`) and reconcile before acting.
  A single agent is unreliable on load-bearing reversals; the panel is the guard. (Strengthens §4: whiplash
  already triggers a second opinion — this makes the panel BINDING for reversals.)
- **ARCHIVED-CONCLUSION TAG.** When any agent cites a prior/archived conclusion ("X is dead / exhausted /
  refuted"), it MUST be tagged **RE-TEST-REQUIRED**, never stated as fact — especially after a reset whose
  premise is "prior conclusions are hypotheses, not facts." A claim becomes "fact" only when reproduced under
  the CURRENT, verified-sound apparatus.
- **BROKEN-APPARATUS FLAG ON THE DEAD-LIST.** A negative produced under a since-fixed apparatus bug (wrong
  cost model, a no-op gate, a loader bug) is a **FALSE-NEGATIVE CANDIDATE**, not a settled refutation. Every
  dead-list entry carries the apparatus-state it was found under; fixing an apparatus bug RE-OPENS every
  negative that depended on it. (A broken measurement instrument produces false negatives, not only false
  positives — the dead-list is only as trustworthy as the apparatus that wrote it.)
