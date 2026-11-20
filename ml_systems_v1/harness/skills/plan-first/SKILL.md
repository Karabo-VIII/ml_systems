---
name: plan-first
description: Before building anything non-trivial, decompose the task into a short ordered list of VERIFIABLE sub-goals, each with its own mechanical check. Use for any multi-step task -- it is the think-before-build step that pairs with write-verifiable.
---
# Decompose into verifiable sub-goals before you build

A big task fails silently when you build it all at once and only check at the end. Break it into the smallest
steps that each END in a command that proves the step worked. Plan first, then build each step with
`write-verifiable`.

## Steps
1. **Restate the goal in one line** + the single command that will prove the WHOLE thing works (the final
   `verify_cmd`). If you cannot name that command, the goal is still too vague -- sharpen it first.
2. **List 2-6 ordered sub-goals.** Each sub-goal is independently checkable. Order them so each builds on the
   last and nothing forward-references.
3. **Attach a check to every sub-goal** -- an assertion, a `--help` that must exit 0, a tiny script. A
   sub-goal with no check is not a sub-goal; merge or split it.
4. **Build them one at a time** with `write-verifiable`. Run each sub-goal's check before starting the next.
   A failing early check is cheap; a failing final check after building everything is expensive.
5. **Finish** when the final `verify_cmd` from step 1 passes. Do not expand scope mid-build.

## Example
Goal: "a CLI that sums numbers from a file." Final check: `echo "1 2 3" > n.txt && python sum_cli.py n.txt` -> `6`.
- sub-goal 1: `read_numbers(path)` returns a list of floats -- check: assert on a temp file.
- sub-goal 2: `total(nums)` sums them -- check: `assert total([1,2,3]) == 6`.
- sub-goal 3: wire the CLI (argv -> read -> total -> print) -- check: the final command above.

## Why
- Small verified steps localize the error to the step you just wrote (rejection-as-gradient, tightly scoped).
- The plan IS the set of `verify_cmd`s -- it hands `write-verifiable` its tests for free.
- A weak model that cannot hold the whole task in its head CAN finish one checkable step at a time.
