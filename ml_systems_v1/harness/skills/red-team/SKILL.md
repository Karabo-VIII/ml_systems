---
name: red-team
description: Before finalizing, adversarially attack your own artifact -- hunt the edge cases and failure modes a passing happy-path test missed, add checks for them, and fix what breaks. Use after a build's verify passes but before you emit final, especially on anything load-bearing.
---
# Attack your own build before you call it done

A green happy-path test proves the artifact works on the cases you THOUGHT of. It says nothing about the ones
you didn't. Before finalizing, switch stance from builder to attacker: "what input makes this produce a wrong
answer silently, or crash?" Then prove it can't.

## Steps
1. **Stance flip.** Assume the artifact is broken. Your job is to find HOW, not to confirm it works.
2. **Enumerate failure modes** for this kind of artifact:
   - boundary / empty / zero / negative / huge inputs
   - malformed or wrong-type input (does it fail loudly or silently?)
   - off-by-one, division by zero, integer vs float, unicode/encoding
   - state/order dependence; idempotency; concurrent or repeated calls
   - the spec's IMPLICIT requirements the happy path skipped
3. **Write a check for each plausible mode** -- extend the verifier with these adversarial cases (assert the
   CORRECT behavior, including "raises on bad input").
4. **Run it.** Each failure is a real bug found before delivery -- fix the artifact (not the test) and re-run.
5. **Finalize** only when the adversarial checks pass too. If a mode is out of scope, say so explicitly rather
   than leaving it silently uncovered.

## Example
Built `divide(a, b)` with `assert divide(6, 2) == 3`. Red-team adds: `divide(1, 0)` must raise (not return inf);
`divide(-6, 2) == -3`; `divide(7, 2) == 3.5` (not 3). The div-by-zero case usually exposes a real bug.

## Why
- The happy-path author and the adversary are different mindsets; running both on your own work catches what one
  alone misses.
- Edge-case failures found pre-delivery are cheap; found in use (or in a capital decision) they are not.
- Pairs with `write-verifiable`: that proves it works; this proves it does not break.
