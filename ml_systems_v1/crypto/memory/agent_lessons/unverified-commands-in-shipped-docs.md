# Unverified commands in shipped runbook docs

**Date observed**: 2026-05-14
**Severity**: high (user noticed; would have wasted real compute time if launched)
**Frequency**: pattern-recurring (similar to "rationalize-then-bury" failure mode)

## Context

While producing `docs/COMMANDS_TO_RUN_2026_05_14.md` as a deliverable for the
user to run, I wrote commands like:

```bash
python src/pipeline/fetch_all.py --kind klines_1m --universe u59 ...
```

without actually running `--help` against `fetch_all.py` first. Reality:
- The flag is `--trade-mode klines`, not `--kind klines_1m`
- `--universe` only accepts `u10/u50/u100` (not `u59`)
- `make_dataset_v51.py` doesn't exist — it's `make_dataset.py`

User caught it: "the commands did not show up: COMMANDS_TO_RUN §1".

## What went wrong

When writing a runbook, I assumed CLI flags based on what would be sensible
rather than what actually exists in the target script. Three of the five
commands in the initial draft had wrong flags or wrong file names.

## Root cause

A specific anti-pattern: **plausible-looking commands that haven't been
tested.** This is more dangerous than wrong code because:
- Code that doesn't compile fails immediately in CI/CDAP
- A wrong CLI command in a runbook ships silently and only fails when the
  user runs it — wasting their time AND undermining trust in the doc

## How to apply

1. **Every command in a runbook must have its `--help` checked** before
   shipping. No exceptions.
2. **If a CLI flag is uncertain, explicitly mark it `UNVERIFIED` with a
   "run --help first" note** (this is what §8 of the rewritten doc now does)
3. **For long-form runbooks, the verification step is:**
   ```bash
   for cmd in $(grep -E "^python " docs/RUNBOOK.md); do
     python <script> --help 2>/dev/null | head -3
   done
   ```
   Verify each script's actual flag surface BEFORE writing the runbook.
4. **Prefer copying real shell sessions over composing commands.** If you
   ran the command successfully, paste that. If you didn't, mark it
   UNVERIFIED.

## Related
- `agent_protocols/test_first.md` — generalized "verify-before-ship" discipline
- `agent_protocols/calibrated_uncertainty.md` — VERIFIED/REPORTED/INFERRED tagging
  applies to CLI flags too: only mark VERIFIED if you ran `--help`
- This session's earlier `silent-skip-buried-in-report.md` — same family of
  "produced docs without verifying the underlying claim" failures
