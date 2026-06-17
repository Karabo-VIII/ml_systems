---
name: debug-failure
description: Diagnose and fix a failing command, test, or verify_cmd by reading the actual error and iterating. Use when a build is refuted, a test fails, a traceback appears, or a verify_cmd exits non-zero.
---
# Fix it from the real error (rejection-as-gradient)

A refutation is not a dead end -- it is a directed signal. The concrete error tells you exactly what to fix.

## Steps
1. **Read the actual error** -- the stderr/traceback, not a guess. Find the failing line + the assertion/exception.
2. **Form ONE hypothesis** about the cause (off-by-one, wrong type, missing import, bad edge case).
3. **Make the smallest change** that addresses it. Do not rewrite everything.
4. **Re-run the exact same verifier.** Did the error change? Progress. Same error? Your hypothesis was wrong -- form
   a new one from the new evidence.
5. **Repeat** until exit 0. Then emit `final` immediately.

## Tactics
- Reproduce minimally: run the smallest snippet that triggers the failure.
- Add a `print`/assert at the suspected line to confirm the actual values.
- Check the obvious first: imports, file paths (relative to the build cwd), off-by-one, empty/edge inputs.
- If two fixes in a row do not change the error, you are guessing -- step back and re-read the error from scratch.
- Empty string / zero / None are the usual edge cases a test will catch -- handle them explicitly.
