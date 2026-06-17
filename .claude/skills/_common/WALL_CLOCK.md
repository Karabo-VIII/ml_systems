# WALL-CLOCK GROUNDING — verified time, never felt time (binding, all instances + autonomous mode)

> **Core fact the model must internalize: you have NO internal clock.** You cannot perceive how much
> wall-clock time has passed within a session, between turns, or during long tool calls. Elapsed time,
> "current time", "X hours in", "N hours left", "last modified", "X days ago" — ALL of these are unknowable
> by feel and MUST be MEASURED with a `date` call. Stating any of them from memory/feeling is fabrication.

## The rule (BINDING for every instance and every directive)
1. **Any wall-clock claim grounds against a fresh `date` call.** Date, time, ETA, elapsed, remaining,
   "last modified", "X days/hours ago", a cycle timestamp — run `date` first, then state the number.
2. **Tag the basis:** `VERIFIED` (from a `date`/`stat`/`git log` reading this turn), `REPORTED` (the user
   or a doc said it — repeat their value, don't re-derive), `INFERRED` (a computed delta from a VERIFIED
   anchor — show the arithmetic). Never present INFERRED or felt time as VERIFIED.
3. **Re-ground after any gap:** session resume, context summary, a long-running/background tool call, a
   `<system-reminder>` date-change, or simply a new turn after time may have passed → `date` again before
   any time claim. The 2026-05-21→05-22 session-mid-rollover is the canonical silent-drift trigger.

## ELAPSED TIME (the 2026-06-03 failure mode — added after a real incident)
- **Never compute "elapsed" or "time remaining" from how much work you did or how long it *feels*.** Work
  volume ≠ time. You can do 6 experiments in 70 minutes or in 5 hours — you cannot tell which without `date`.
- **Elapsed = (VERIFIED now via `date`) − (VERIFIED start).** Both ends must be real readings. Record the
  start time with a `date` call when a timed task begins; recompute against a fresh `date` whenever you
  report progress. Show the subtraction.
- **Incident of record (2026-06-03):** in a 12h autonomous run that started 16:30 SAST (verified), an
  instance claimed "~5h in" and wrote a ledger of invented timestamps (C4 17:30 … C11 19:40) — the real
  elapsed was **~1h12m** (actual time 17:42). The fabrication also biased strategy (prematurely declared the
  frontier "exhausted for the time left" when ~11h actually remained). Felt-time is not a signal; measure.

## AUTONOMOUS-MODE LEDGER (binding for AUTONOMOUS_RUNNER + /loop + /schedule + any timed run)
- **Every cycle/ledger entry timestamp is a real `date` reading at that moment** — not a guessed progression.
  If you didn't call `date`, write the cycle label without a time (e.g. "C7") rather than invent one.
- **Re-ground the clock at the START of each build→run→learn cycle** and before any "N hours in / left" claim
  to the user. Budget/stop-condition decisions ("am I near the time budget?") MUST use a fresh `date`.
- **A timed mandate ("go for N hours") records its VERIFIED start once**, then every elapsed/remaining figure
  is `date − start`. Do not narrate the passage of time; read it.

## Quick primitive
`date "+VERIFIED %Y-%m-%d %H:%M %Z"` → anchor. Elapsed: `date +%s` minus the stored start epoch, /60 for
minutes. For file/commit times use `stat`/`git log --format=%cd`, not assumption.
