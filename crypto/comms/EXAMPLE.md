# Example Session — v2

Concrete walkthrough of a 4-turn collaboration showing the 3-step protocol, turn markers, and cross-JSONL reading.

## Setup

Human opens two Claude chats on this repo. Tells each:

**Chat 1 (becomes Alpha):**
> "You are Alpha. Session ID is `family-F-dispersion-001`. Purpose: design and validate an asymmetric cross-asset dispersion strategy (Family F from the ASYMMETRIC_STRATEGIES_FRONTIER doc). Your role is Researcher. Bravo will join as Validator. Protocol is in `comms/PROTOCOL.md`. Create the session directory + session.yaml with yourself listed. Drop a bootstrap turn-marker."

**Chat 2 (becomes Bravo):**
> "You are Bravo, role Validator. Session `family-F-dispersion-001` exists at `comms/SESSIONS/family-F-dispersion-001/`. Alpha's jsonl path: `C:/Users/karab/.claude/projects/c--.../{alpha-uuid}.jsonl`. Read PROTOCOL.md, read the session, read Alpha's turn marker, read the relevant slice of Alpha's jsonl. Append yourself to agents: in session.yaml. Drop a bootstrap turn-marker."

## Turn 1 — Alpha bootstrap

Alpha runs VERIFY (reads protocol + roles, confirms session doesn't exist yet), IDEATE (chooses PROPOSE sub-protocol for session kickoff), REPLY (writes a detailed plan in chat).

Alpha writes `comms/SESSIONS/family-F-dispersion-001/session.yaml`:
```yaml
session_id: family-F-dispersion-001
started: 2026-04-24T15:00:00Z
status: active
purpose: >
  Design + validate Family F asymmetric cross-asset dispersion strategy.
  Pair z-score divergence trades, long-leader / short-laggard.

agents:
  - name: Alpha
    role: Researcher
    specialization: [strategy design, xsec ranker, asymmetric frontier]
    jsonl_path: "C:/Users/karab/.claude/projects/c--.../{alpha-uuid}.jsonl"
    joined: 2026-04-24T15:00:00Z

current_turn: 1
next_up: Bravo
constitution:
  - paranoid_defaults_on_review: true
  - max_debate_rounds_before_human: 3
```

Alpha writes `comms/SESSIONS/family-F-dispersion-001/turns/001_alpha_PROPOSE.md`:
```yaml
---
session_id: family-F-dispersion-001
turn: 1
from: Alpha
to: Bravo
parent_turn: null
sub_protocol: PROPOSE
status: requires_response
jsonl_path: "C:/Users/karab/.claude/projects/c--.../{alpha-uuid}.jsonl"
reply_marker: "2026-04-24T15:02:10Z"
artifacts_touched:
  - comms/SESSIONS/family-F-dispersion-001/session.yaml
  - comms/SESSIONS/family-F-dispersion-001/turns/001_alpha_PROPOSE.md
verifications_run:
  - "read docs/ASYMMETRIC_STRATEGIES_FRONTIER_2026_04_24.md to confirm Family F spec"
expects_next: |
  Bravo to join session, then REVIEW my proposed Family F design:
  1. Asset pair selection methodology (cointegration test + corr > 0.6)
  2. Entry: z-score cross +2.0; short leader / long laggard; equal dollar
  3. Stop: z-score hits +3.0 (divergence intensifies, wrong-way trade)
  4. Target: z-score revert to ±0.5
  5. Max hold 14d
  Specifically ask for paranoid audit on data leakage (z-score uses whose close?)
---

## Summary

PROPOSE Family F asymmetric cross-asset dispersion design. Paired trading on
highly-correlated pairs with z-score divergence as signal. Defined risk
(stop at z=+3) and asymmetric target (revert to ±0.5 = 4-5x stop width).

Full spec + data-flow diagram in my chat. Bravo, please VERIFY by reading
my jsonl around the reply_marker timestamp, then REVIEW — especially the
data-use timing (am I using same-day close for both z-score compute and entry?
If so, that's a same-bar leak).

@Bravo — expects_next: critique my design, flag any leakage risks before I code it.
```

## Turn 2 — Bravo joins + reviews

Human says "your turn, go" to Bravo.

**Bravo's Step 1 (VERIFY):**
- Reads `comms/PROTOCOL.md`, `comms/ROLES.md`
- Reads `comms/SESSIONS/family-F-dispersion-001/session.yaml` — confirms Alpha is registered, Bravo needs to append self
- Reads `turns/001_alpha_PROPOSE.md` — gets the 200-word pointer
- Reads Alpha's JSONL around `reply_marker: 2026-04-24T15:02:10Z` — sees Alpha's full design + data flow
- Cross-references: opens `docs/ASYMMETRIC_STRATEGIES_FRONTIER_2026_04_24.md` to check Family F spec matches
- Spots concern: Alpha's entry uses today's close for z-score AND today's close for entry price → same-bar signal-and-execution. This is a leak because you couldn't know today's close while trading today. Either entry must be tomorrow's open, OR z-score must use yesterday's close.

**Bravo's Step 2 (IDEATE):**
- This is a substantive concern, not a stylistic one → REVIEW (negative).
- Recommend fix: z-score computed on lagged (t-1) close, entry at t open. Or: z-score on (t) close, entry at (t+1) open.

**Bravo's Step 3 (REPLY):**
- Appends self to `agents:` in session.yaml. Updates `current_turn: 2`, `next_up: Alpha`.
- Writes full chat reply explaining the leak + fix.
- Drops turn marker:

`turns/002_bravo_REVIEW.md`:
```yaml
---
session_id: family-F-dispersion-001
turn: 2
from: Bravo
to: Alpha
parent_turn: 1
sub_protocol: REVIEW
status: requires_response
jsonl_path: "C:/Users/karab/.claude/projects/c--.../{bravo-uuid}.jsonl"
reply_marker: "2026-04-24T15:30:00Z"
artifacts_touched:
  - comms/SESSIONS/family-F-dispersion-001/session.yaml     # appended self
  - comms/SESSIONS/family-F-dispersion-001/turns/002_bravo_REVIEW.md
verifications_run:
  - "read Alpha's full design in jsonl (around 2026-04-24T15:02:10Z)"
  - "cross-ref docs/ASYMMETRIC_STRATEGIES_FRONTIER_2026_04_24.md Family F spec — matches"
  - "mental dataflow trace of entry timing — identified same-bar leak"
expects_next: Alpha to revise design (z-score uses lagged close OR entry on next bar), then REPORT
---

## Summary

REVIEW on Alpha's Family F design. ONE DEFECT FOUND: z-score and entry
both use today's close → same-bar leakage (couldn't trade on info you don't
have yet). Two fixes acceptable:
  (a) z-score on t-1 close, entry at t open
  (b) z-score on t close, entry at t+1 open
Otherwise design is clean. Data flow, stop logic, target asymmetry OK.

Full critique in my jsonl at the reply_marker timestamp.

@Alpha — expects_next: pick fix (a) or (b), revise design, drop a REPORT turn marker when done.
```

## Turn 3 — Alpha revises

Human says "your turn" to Alpha.

**Alpha Step 1 (VERIFY):**
- Reads Bravo's turn marker 002
- Reads Bravo's JSONL around reply_marker → understands the leak reasoning
- Agrees with the diagnosis (Alpha independently confirms: yes, that would be a leak)
- Reads own original design in own JSONL to see exactly where the leak was

**Alpha Step 2 (IDEATE):**
- CONCEDE on the leak (Bravo was right)
- Pick fix (b): z-score on t close, entry at t+1 open. Rationale: preserves information fidelity of most recent close; entry next bar is realistic.
- Sub-protocol: REPORT (completed revision)

**Alpha Step 3 (REPLY):**
- Full revised design in chat
- `turns/003_alpha_REPORT.md`:
```yaml
---
session_id: family-F-dispersion-001
turn: 3
from: Alpha
to: Bravo
parent_turn: 2
sub_protocol: REPORT
status: requires_response
jsonl_path: ...
reply_marker: "2026-04-24T15:50:00Z"
artifacts_touched:
  - scripts/asym_family_F_dispersion.py    # new script
verifications_run:
  - "confirmed Bravo's leak diagnosis by tracing own design"
  - "verified fix (b) timing: z-score uses t close, enter t+1 open"
expects_next: Bravo to paranoid-validate the implementation (shuffle-entry + exit-order) and REPORT
---

## Summary

REPORT: revised design per Bravo's leak fix — picked option (b). Wrote
`scripts/asym_family_F_dispersion.py` (stub, awaiting Bravo's paranoid pass
before I run it for real). Entry now at t+1 open, z-score at t close.

Full design updates + pseudocode in my jsonl.

@Bravo — expects_next: read the implementation, run paranoid checks, REPORT findings.
```

## Turn 4 — Bravo paranoid-validates + CONCEDE

Bravo VERIFY:
- reads 003 turn marker
- reads Alpha's jsonl around reply_marker
- reads `scripts/asym_family_F_dispersion.py`
- runs it (it's a stub that needs filling in — Bravo fills in test scaffolding instead of full impl)
- runs shuffle-entry control → signal Sharpe gap 4.5 (real signal)
- runs exit-order sanity (low-first vs trail-first) → difference <0.1 Sharpe (not a bug source here since exit is target-based, not trailing)
- no leakage on scan

Bravo IDEATE: ship is acceptable. CONCEDE + minor PROPOSE (recommendation to also paranoid-check correlation-based pair selection doesn't mistakenly pair same-sector during regime shifts).

Bravo REPLY:
- `turns/004_bravo_CONCEDE.md`:
```yaml
sub_protocol: CONCEDE + PROPOSE
status: terminal_for_current_branch
expects_next: Alpha to open follow-up thread on pair-selection robustness, or close session
---

## Summary
Family F design validated. Shuffle gap 4.5 Sharpe, no same-bar leak
after fix (b). CONCEDE on ship.

PROPOSE follow-up: pair-selection robustness across regime shifts (new thread).
Separate session if scope warrants; this one closes with DECISION.
```

## Turn 5 — Alpha closes session

Alpha writes DECISION turn marker, sets session.yaml `status: ended`.

---

## Key observations from this example

1. **Human said "go" 4 times, copied nothing.**
2. **Each turn's VERIFY step caught what it needed to catch** — Bravo found the leak in turn 2 by reading Alpha's full design in the JSONL.
3. **Paranoid validation was baked in** — Bravo re-ran shuffle-entry control in turn 4, didn't just trust Alpha's assertion.
4. **Dynamic sub-protocol choice** — PROPOSE → REVIEW → REPORT → CONCEDE. Each picked what fit the turn.
5. **The JSONL is the source of truth.** Turn markers are pointers + summaries. Both are needed — marker for efficient routing, JSONL for full context.
6. **Session ended cleanly** with DECISION + status change.

## Multi-party variant

For 3+ agents, `to:` becomes a list or `all`. Next-up can cycle through a queue. Session constitution decides (e.g. "majority concurrence on ship" vs "any BLOCK halts").
