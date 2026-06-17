---
name: write-verifiable
description: Build an artifact together with a MECHANICAL test that proves it works (exit 0 == pass). Use for any "create/build/implement/write a function/script/module" task -- it is the harness's core honest-build discipline.
---
# Build it, then let a real command prove it

A build is only "done" when a real command says so -- not when you claim it. This is how the harness stays honest
(especially with a weaker local model whose self-report you cannot trust).

## Steps
1. **Write the artifact** with the `write_file` tool (e.g. `palindrome.py`).
2. **Write a tiny verifier** that exercises it and exits non-zero on failure -- assertions are perfect:
   ```python
   # verify.py
   from palindrome import is_palindrome
   assert is_palindrome("A man, a plan, a canal: Panama") is True
   assert is_palindrome("hello") is False
   print("VERIFIED")
   ```
3. **Run it** with `run_python` / `run_shell`. Exit 0 = ground-truth PASS. If it fails, READ the traceback, fix the
   artifact, and re-run -- the error is your gradient.
4. **Finalize.** Once the verifier passes, emit your `final` -- do not keep tinkering.

## Why
- `exit 0` overrides any LLM judgement -- no vote can fake a green build.
- A non-zero exit captures the concrete error, which is fed back into the next attempt (rejection-as-gradient).
- When run via the harness, attach the test as the node's `verify_cmd` (or use `run.py --verify-cmd "..."`); a
  passing verify also HARVESTS the capability into the skill library (with `--harvest`).
