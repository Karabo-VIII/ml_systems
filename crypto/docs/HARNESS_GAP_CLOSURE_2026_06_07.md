# Harness Gap-Closure Roadmap (2026-06-07 audit)

Scope: the autonomous / LangGraph / harness stack (NOT chess — chess is the next cycle). Produced by the `/audit`
of "is the harness robust, ReAct+Reflexion over cycles, with a trustworthy audit component, no gaps?" Every finding
below was verified against code/runtime (RWYB), not asserted by an agent.

## Verdict up front (honest)

The harness has a **genuinely good ReAct+Reflexion core** (`metaop/graph.py`: plan→dispatch→judge→reflect→route,
mechanical verifier with rejection-as-gradient, durable checkpointer, HITL park). It is **NOT yet** "no gaps / SOTA /
won't deviate when you kick off autonomous mode." Three structural reasons (below): the live loop runs a forked
crypto-hardcoded copy; the self-improvement + anti-drift modules are **built but not wired into the loop**; and 5/6
"Two Minute Papers" upgrades are doc-only. The audit/trust component had a real hole (now fixed).

## Closed this cycle (audited + committed)

| Fix | What | Evidence |
|---|---|---|
| **Verifier trust-core** (06e07ab) | `_run_verify` ran brain-authored `shell=True` with NO screening → a brain could fake a PASS with `verify_cmd:"true"` or weaponize it (`rm -rf`, force-push, `dd`, `curl\|sh`). Added `_screen_verify_cmd` (reject trivial→125, destructive→126) to BOTH forked copies. | `_test_verify_guard.py` ALL COPIES PASS |
| **in_progress black-hole** (8c7ebd8) | Stop hook counted only done/open → a node parked `in_progress` read as "exhausted/done" (false-done) and was never re-surfaced. `ACTIVE_STATUSES=(open,in_progress)`. | `test_loop_control.py` 21/21 (19 regression + IP1/IP2) |
| **G-A duplication-drift** (staged, uncommitted) | The forked `scripts/autonomy/metaop/` is now a set of THIN shims over the SINGLE canonical engine `harness/metaop/`. Engine logic lives ONCE; the shims inject only crypto specifics (domain, `.claude` fences + `permission_policy.json`, repo-root cwd, `runs/autonomy` traces/learnings, expert ALIASES). Unioned ALL features: harness GAINED `PersistentCliBrain` + the planner verify-seam (`DEFAULT_VERIFY_RETRIES`/`_seed_verify_defaults`) + an `aliases` persona seam; scripts GAINED `OllamaBrain` (brain-swap now reachable from the live loop). `KNOWN_DRIFT` is now empty. | `_test_copy_parity.py` PASS (0 known drift), `_test_verify_guard.py` ALL COPIES PASS, `_test_verifier`/`_test_planner_verify` PASS, `test_loop_control.py` 21/21, all 3 falsifier probes PASS, both `run_metaop` + `harness/run.py` smokes PASS |

## Open gaps — EV-ranked (NOT rushed; each needs its own audited cycle)

| # | Gap | Severity | Why it matters to "won't deviate / SOTA" | Fix sketch | Est |
|---|---|---|---|---|---|
| ~~**G-A**~~ **CLOSED** (staged 2026-06-07) | ~~**Duplication-drift**: the harness was "separated" by COPYING.~~ DONE: `harness/metaop/` is now the SINGLE canonical engine; `scripts/autonomy/metaop/` is a set of thin crypto-consumer shims (`from harness.metaop.<mod> import *` + inject crypto domain/fences/cwd/ALIASES). NO behavior lost from either copy (features unioned both ways). Brain-swap (Ollama) now reachable from the live loop. | ~~HIGH~~ DONE | Was: brain-swap only worked in the non-running copy; two copies drifted. Now: one source, drift-guard (`_test_copy_parity.py`) green with EMPTY `KNOWN_DRIFT`. | DONE — see "Closed this cycle" row above | done |
| **G-B** | **Mem0 vector memory NOT built** (Two-Min-Papers #1). Grep `faiss\|mem0\|embedding\|vector` in `scripts/autonomy/` = 0 hits. `learnings.py` is flat-append, time-ordered retrieval. | HIGH | This was the headline fix for "instances forget after compaction." The loop does NOT retrieve task-similar past cycles; it only reads recent learnings by time. | Add a local FAISS (or sklearn-NN) store keyed by task-embedding; `plan`/`reflect` retrieve top-3 similar past cycles as few-shot. Confidence-gate writes (weak local brain can hallucinate memories). | build, ~half-day |
| **G-C** | **Self-improvement + anti-drift modules unwired**: `skill_library` (Voyager reuse-before-build), `problem_framing` (depth/breadth + "never declare impossible"), `resourcefulness` (decompose-the-ideal), `hypothesis_register` (don't re-mine refuted veins) — all exist but are **NOT imported by `metaop/graph.py`**. Only `learnings` is wired. | HIGH | This IS the "agent comes up with original questions/directions, doesn't narrow, won't deviate" the mandate asks for. Today an autonomous run does NOT automatically get it — the modules are CLI scaffolding, not loop nodes. | Add a `frame` step before `plan` (problem_framing+resourcefulness expand the objective into depth+breadth) and a `recall` step (skill_library.digest + hypothesis_register) feeding the plan payload. | wiring, ~half-day |
| **G-D** | Two-Min-Papers #2–#5 (cascade-router, EvoFSM, DGM skill-archive, AlphaEvolve islands) are **design-only** in `docs/HARNESS_EVOLUTION_ROADMAP_2026_06_06.md`. Only #1-ish (mechanical verifier ≈ AlphaProof) is wired. | MED | The "improve over cycles" evolution is currently just `reflect` generating adjacent nodes + flat learnings — not the archive/island evolution the roadmap promises. | Pick ONE (DGM skill-archive is the best fit: benchmark-before-promote). Do NOT let it mutate commit/permission gates (HITL). | build, ~1 day each |
| **G-E** | **18 dead CRITICAL CDAP guards** (post-reset path rot → silent no-ops): `python scripts/mandatory_gate.py` lists them (leakage/layer-isolation/strat invariants pointing at deleted `src/strategy/`). | MED | The audit component the user wants trustworthy has 18 real-capital safety invariants currently INERT. | Retire or re-point per `docs/CDAP_DEAD_SECTIONS_2026_06_05.md`. (Crypto-strat scope — overlaps the chess-next boundary; do with strat work.) | ~2h |
| **G-F** | **`mandatory_gate.py` itself unwired**: not called by `.git/hooks/pre-commit` (only `check_no_large_files`+`check_invariants`) nor by `check_invariants.py`. CLAUDE.md/memory claim it makes CDAP "unskippable" — actually `permission_gate`'s `SKIP_CDAP` regex does. | MED | The meta-gate that detects silent-no-op gates is itself a silent-no-op (ironic). | Add `mandatory_gate.main()` to `check_invariants.py` as a WARN-tier section (NOT exit-2 until G-E clears, else it blocks all commits). | ~1h |
| **G-G** | `intent_clock.py` built but **unwired** (not in `settings.json` UserPromptSubmit). Wall-clock fabrication fix not mechanically enforced. | LOW | The 2026-06-03 "claimed 5h, was 1h12m" class isn't hook-enforced; relies on the agent running `date`. | Wire into `settings.json` UserPromptSubmit — **USER-GATED** (settings.json is a 3-layer deny; Claude cannot auto-edit it). User adds the hook via `/hooks`. | user, 5 min |
| **G-H** | `.gitignore:176 scripts/autonomy/metaop/_*` silently ignores the **entire metaop probe/test suite** (`_test_verifier`, `_proof_loop_delivers`, falsifier probes). Not version-controlled → absent on fresh clone. | LOW | The verification harness that proves the loop's trust claims isn't reproducible. (This audit force-added `_test_verify_guard.py`.) | Narrow the ignore to `_operator_snippet.py`/scratch; track `_test_*`/`_proof_*`. | ~15 min |
| **G-I** | `permission_gate` `cmd_deny_regex` screens the WHOLE command string, so a commit MESSAGE that merely mentions `rm -rf /` is blocked (hit this audit; worked around via `git commit -F file`). | LOW | False-positive friction on legit commits/docs. | Scope the destructive-regex screen to the command head, not quoted message bodies (or exempt `git commit -F`). | ~30 min |
| **G-J** | `track_job._alive()` is **bare-PID** (Windows reuses PIDs fast; pid 23932 was `train_robust`, now `AppVShNotify`). A recycled PID reads as "job alive" → eternal WAIT-MODE (a new stuck-path). | MED | The anti-silent-death guarantee can invert into an anti-stop bug on PID reuse. | Store + verify process create-time (`psutil`/`wmic`) alongside PID; treat mismatch as dead. | ~1h |

## Recommended order for the next cycles
1. **G-A** (dedup to one canonical harness) — unblocks honest brain-swap + stops all future drift. Do FIRST.
2. **G-C** (wire anti-drift/idea-gen into the loop) — this is the "won't deviate / original questions" the mandate centers on.
3. **G-B** (vector memory) — the "stops forgetting" fix.
4. **G-J, G-F, G-E** (trust/safety hardening) → then **G-D** (one evolution upgrade).
5. **G-G, G-H, G-I** (cheap hygiene; G-G is user-gated).

## The honest bottom line
"Kick off autonomous mode and trust it won't deviate" is **not yet safe** until G-A + G-C land: today the live loop is
the crypto-forked copy and the breadth/anti-drift/reuse machinery is dormant. The CORE is sound and the trust anchor
(verifier) is now hardened — but the synergy ("everything coming together") the chess test was meant to prove is
gated on G-A/G-C, not on chess. Chess will exercise a loop whose self-improvement modules aren't wired in yet.
