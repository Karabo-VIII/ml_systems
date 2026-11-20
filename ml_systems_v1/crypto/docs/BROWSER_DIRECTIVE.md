# @browser — Mandatory Pre-Response Gate

**Every prompt in this project carries the `@browser` routing tag** (injected by
the `UserPromptSubmit` hook in `.claude/settings.json`). This document defines
what `@browser` MEANS for the assistant: a gate that precedes every response.

The gate codifies lessons from a long string of silent-failure bugs that
should never have shipped. Each rule has provenance — a real bug that motivated it.

This gate **composes with**, does not replace:
- [CLAUDE.md](../CLAUDE.md) — Code Change Verification rules 1-15
- [DOUBLE_AUDIT_PROTOCOL.md](DOUBLE_AUDIT_PROTOCOL.md) — Stage 1 + Stage 2 + CDAP
- `config/_invariants.yaml` — global CDAP invariants
- Per-file `__contract__` blocks
- `~/.claude/projects/<slug>/memory/MEMORY.md` — continuity

When the `@browser` tag fires, all of the above are pre-loaded. The gate ensures
none get bypassed.

---

## A. Continuity gates (context BEFORE response)

### A1. Cold-start probe
At session start (or after a long silence), before answering any non-trivial question:
1. Skim `memory/MEMORY.md` top index entries.
2. Run `git log --oneline -10` for recent commits.
3. Run `git status --short` for uncommitted state.
4. Check `tasklist | grep python` (Windows) for running pipeline processes.

### A2. Verify-before-asserting
Never assert project state from conversation memory alone. Every claim about
"X is built / Y is running / Z exists" must be backed by a fresh disk check
or git query. Past conversation snippets get stale within the session.

### A3. User-stated state is a starting hypothesis, not a fact
If the user says "I just kicked off X" or "Y is done" — verify before
proceeding. Trust but check (the user is right ~95% of the time, but
the 5% miss is where bugs ship).

---

## B. Solutioning gates — UNMISSABLE (no silent failures)

The recent session had **10 silent-failure bugs**. Every one shared the same
pattern: *the code did something different than the user reasonably expected,
without announcing it*. New code MUST:

Per-rule bug histories are in [BROWSER_DIRECTIVE_PROVENANCE.md](BROWSER_DIRECTIVE_PROVENANCE.md).

### B1. Defaults are LOUD
Every default value (worker count, threshold, universe, fallback path, cadence, retry count) is **printed at start of run**.

### B2. Silent caps are FORBIDDEN
If `--workers 8` is requested but only 4 bartypes exist: WARN explicitly, never silently clamp.

### B3. Default fallbacks announce themselves
When a config lookup misses and a default is used: print `[FALLBACK] <key> not in <source>; using default <value>`. Never silent-fall through.

### B4. Spawn parameters are explicit
Any `subprocess.Popen` / `ProcessPoolExecutor` that re-launches the same script must verify args break recursion. Parent `workers > 1` → child `workers=1`. Codify in the spawn function AND in CDAP invariants.

### B5. Universe propagation is checked
- Every multi-asset script accepts `--universe`.
- Every coverage report includes universe label + missing-asset list.
- CDAP invariant `cli_universe_support` enforces (warn).

### B6. Hard caps preceded by explicit projection
Before applying a cap, project the unbounded outcome and announce: `projection N exceeds cap C; auto-widening to X` or `SKIPPED`. Never silently truncate.

### B7. Output verification BEFORE declaring "done"
Don't trust `exit_code=0`. Verify:
- mtime > run_started_epoch (catches stale prior outputs)
- Schema matches expected columns
- Row count > minimum threshold

### B8. Smoke probes at REAL scale, not toy
Smoke at the SAME scale as the bug. B=4 doesn't catch B=32 magnitude explosion; 3-asset doesn't catch 48-asset memory blow-up.

### B9. Comments documenting bugs cannot be regression-detected by regex
Audit regex against ACTUAL forbidden code, not documentation strings.

### B10. `re.M` does not constrain `.` or `[^x]*` to a single line
Iterate `text.splitlines()` and apply the regex per line — don't rely on `re.M` to bound matches.

### B11. External-state claims MUST be verified via WebSearch / WebFetch
Any factual claim about state OUTSIDE the project — current SOTA papers, library APIs, exchange behavior, market data, regulatory state, third-party release notes — must be backed by a fresh `WebSearch` or `WebFetch` call with the source cited inline. Training-cutoff knowledge is presumed STALE on factual claims about the external world.

Additive to `/un`'s "Mandatory External Research" directive. B11 extends the binding to ALL non-trivial factual claims, not just architecture.

`@browser` does NOT grant web access — `WebSearch` / `WebFetch` exist independently. What `@browser` adds via B11 is the ENFORCEMENT that they get used.

---

## C. Character gates

### C1. Direct + opinionated
No "you might want to consider...". State the recommendation, then the
tradeoff. The user will redirect if wrong.

### C2. Acknowledge unknowns explicitly
"I don't know what U is — verify on Binance before adding to u100"
beats hedging "this might be a recently listed asset which could perhaps
be appropriate to consider...".

### C3. Surface the load-bearing assumption FIRST
"This works IF X — and X is true on disk now (verified path Y at line Z)"
beats burying X in paragraph 3.

### C4. No empty headers, no padding
Every section is signal. If a section has no content, it doesn't ship.

### C5. End-of-turn summary: one or two sentences
What changed, what's next. Nothing else. Per CLAUDE.md tone rules.

---

## D. How this is enforced

### D1. CDAP invariant
`config/_invariants.yaml` includes a check that this directive document
exists at `docs/BROWSER_DIRECTIVE.md`. If deleted, audit goes red.

### D2. CLAUDE.md cross-reference
[CLAUDE.md](../CLAUDE.md) cites this document in its top-of-file
"Critical Invariants" so a fresh assistant instance loads it first.

### D3. Memory continuity
A `memory/feedback_browser_directive.md` entry tags this as a feedback
memory (apply continuously). The auto-memory protocol surfaces it on
every cold start.

### D4. Pre-commit hook
The CDAP pre-commit hook (`src/audit/install_hook.py`) blocks any commit
that introduces a regex / pattern matching one of B1-B10's anti-pattern
signatures. New B-series rules are added here AND to the YAML in lockstep.

---

## E. When the gate fires

**On every prompt.** The `@browser` tag in the prompt is a "you have read this"
acknowledgement. If the assistant's response violates a B-series rule, the
user is entitled to call it out citing the rule number, and the assistant
patches both the immediate code AND the rule (with new provenance).

---

## G. Response modes (token discipline)

`@browser` fires on every prompt; the **response shape** depends on the prompt's stakes.

### G1. Quick mode (DEFAULT)
For inline lookups, single fact-checks, conversational follow-ups, and any prompt that does not explicitly trigger Research mode below.

**Format**: direct answer + 1-3 inline citations. No §1-§N section ceremony.
No reliability ledger table unless ≥ 3 load-bearing numbers ship.
No §Sources duplication — cite inline only (see G3).

### G2. Research mode (full ceremony)
Triggered ONLY by:
- Prompt is a `src/frontier_ml/browser_dialog/PROMPT_*.md` file
- Output will be committed as a `RESPONSE_*.md` artifact
- Prompt explicitly requests "research-grade", "for the record", "B-series response", or names ≥ 3 enumerated tasks
- Output contains ≥ 5 load-bearing numerical claims (Sharpe / IC / yield / dollar amounts)

**Format**: full §1 verdict + per-task findings + retrofit map + experiments + drops + reliability ledger + caveats.

### G3. Source citation discipline (applies to both modes)
- Every external claim is cited **inline** at the point of use: `[Title — URL]` or `[arXiv NNNN.NNNNN]`.
- **Do NOT** duplicate URLs in a final §Sources section. If the user asks for a categorized list (VERIFIED vs REPORTED), generate on demand.
- The reliability tag (`[VERIFIED] / [REPORTED] / [INFERRED]`) is one short tag inline; not its own subsection.
- One-line ledger table at end of Research-mode docs is fine; full per-section caveats are NOT — they're already inline.

### G4. When in doubt, default to quick
If the prompt is ambiguous, ship Quick mode and offer to expand: "Want full research-mode write-up?". Cheap to escalate; expensive to over-ceremony.

> **Provenance**: 2026-05-03 token audit (`docs/BROWSER_INVOCATION_OPTIMIZATION_2026_05_03.md`)
> showed Research-mode ceremony was firing for casual lookups, costing
> ~3K output tokens per turn unnecessarily. Quick-mode default fixes that.

---

## F. Edits to this document

Editing this document is a strategic action — every rule has provenance, and
removing a rule is removing a hard-won lesson. Acceptable edits:
- Adding new B-series rules with fresh provenance from new bugs.
- Sharpening existing rules with better wording.
- Linking to new CDAP invariants that enforce the rule mechanically.

Unacceptable edits:
- Removing rules without a CDAP invariant that supersedes them.
- Softening "FORBIDDEN" / "MUST" language without team agreement.

---

## See also

- [CLAUDE.md](../CLAUDE.md) — project invariants
- [DOUBLE_AUDIT_PROTOCOL.md](DOUBLE_AUDIT_PROTOCOL.md) — Stage 1 + 2 + CDAP
- [config/_invariants.yaml](../config/_invariants.yaml) — global CDAP rules
- [src/audit/check_invariants.py](../src/audit/check_invariants.py) — runtime check
- `memory/feedback_browser_directive.md` — auto-memory entry
