# Harness — Wants vs Status gap map (2026-06-07, N1 of the 6h SOTA-integration run)

Mined from **all 373 prior transcripts (2234 user messages)** via 5 parallel scouts → deduplicated wants register,
then **status verified against current code** (RWYB — scout claims were NOT trusted; each status checked vs the file).
Scope = the autonomy/solutioning ENGINE (not crypto-strat domain). Legend: ✅ built+verified · 🟡 partial/weak · ❌ missing.

## A. What the user wants (deduped) × status

| # | Want (recurring across chats) | Status | Evidence / gap | Action |
|---|---|---|---|---|
| 1 | **Loop-as-PROGRAM** (durable LangGraph StateGraph, not a re-invoked chat) | ✅ | `harness/metaop/graph.py` real langgraph + SqliteSaver durable checkpointer | keep |
| 2 | **Planner** decomposes objective → frontier | 🟡 | `graph.py:176` plan is **ONE-SHOT** (`if frontier: return {}`); thin generic prompt | N7 |
| 3 | **REPLANNER** (revise/prune plan on failure, not just append) | ❌ | `route` never returns to plan; `reflect` only appends adjacent. **THE #1 fragility** | **N3** |
| 4 | **Pluggable brain** (swap Claude/Ollama/… one line, auto-fallback) | 🟡 | hand-rolled Brain classes; **LiteLLM not integrated**; no auto-fallback | **N4** |
| 5 | **Brain INSTALL-ensure** (ollama+model bootstrap on fresh clone) | ❌ | no install/ensure script | **N4** |
| 6 | **Compounding memory, MONOTONIC, cross-session** | 🟡 | `learnings.jsonl` (time) + TF-IDF `similar_for_plan` (G-B); **Mem0 not integrated** | **N5** |
| 7 | **Mechanical verifier** (independent, unspoofable proof) | ✅ | `graph._run_verify` + `_screen_verify_cmd` (trivial/destructive rejected) | keep |
| 8 | **Independent audit gate before "solved"** | ✅ | CDAP + `mandatory_gate` now dispatched (G-F); overseer-RWYB pattern | keep |
| 9 | **Drift-prevention / breadth / framing / resourcefulness** | 🟡 | modules WIRED into plan (G-C: framer/recaller) — but **advisory context to a one-shot planner**, not enforced; depends on N3+N7 | N3/N7 |
| 10 | **Don't re-mine REFUTED veins** (hypothesis register) | ✅ | `hypothesis_register` recorder wired in judge (G-C) | keep |
| 11 | **EV-ranked frontier + value-needle + goal-bounds preflight** | 🟡 | `frontier.json` + Stop-hook select-by-EV exist; preflight/value-needle are **prose, not code** | N7-adjacent |
| 12 | **3-loop concurrency** (solver + meta + project-evolution) | 🟡 | metaop = solver loop; meta/evolution are cadence-prose in OVERSEER, not a running concurrent program | defer (design) |
| 13 | **Self-summon oracle/auditor at cadence (25/50/75%, plateau)** | 🟡 | prose in OVERSEER/AUTONOMOUS_RUNNER; not a mechanical loop node | defer |
| 14 | **60s health diagnostics + no-silent-hang** | ✅ | `watcher.py` + `loop_health.py` + Stop-hook WAIT-MODE + `proc_liveness` (G-J) | keep |
| 15 | **No approval prompts / no hang** | ✅ | `permission_gate` allow-all-except-deny + bypassPermissions + cd-wedge guard | keep |
| 16 | **Overseer ≠ executor; loop-child can't commit** | ✅ | `permission_gate` METAOP_LOOP/HARNESS_WORKER commit-fence + `tools.HARD_DENY` | keep |
| 17 | **Cross-process durable RESUME** | ✅ | SqliteSaver + lease locks (`manager` + `proc_liveness`) | keep |
| 18 | **EVAL / fitness harness** (task-solve-rate) — keystone | ❌ | none; blocks DSPy + OpenEvolve | **N6** |
| 19 | **DSPy planner-prompt compilation** | ❌ | not integrated (needs N6) | N9 (later) |
| 20 | **OpenEvolve self-evolution** | ❌ | doc-only (needs N6) | N9 (later) |
| 21 | **Two-Minute-Papers #2-5** (cascade-router/EvoFSM/DGM/AlphaEvolve) | ❌ | doc-only | N9 (later) |
| 22 | **Steward-not-initiator + authorize irreversible** | ✅ | harness Stop/loop prompts + the git-revert-net model | keep (note tension w/ "Claude manages") |
| 23 | **One canonical engine, no fork-drift** | ✅ | G-A dedup + `_test_copy_parity` firewall | keep |
| 24 | **Version-controlled verification suite** | ✅ | G-H (gitignore negation tracks `_test_*`/`_proof_*`) | keep |

## B. The honest verdict
The **mechanics** the user asked for (durable loop, verifier, audit gates, health/no-hang, no-prompt, resume, dedup,
memory-seed, anti-drift wiring) are **built + verified**. The **"thinks for itself" core is still weak in 4 places**,
all integration-not-reinvention:
- **N3 REPLANNER** (the #1 gap — one-shot planner can't recover from a bad plan) → LangGraph plan-execute pattern.
- **N4 LiteLLM** brain layer + install-ensure → robust swap/fallback/bootstrap.
- **N5 Mem0** → real compounding memory (replace TF-IDF).
- **N6 EVAL harness** → the keystone that later unlocks DSPy (N7/N9) + OpenEvolve (N9).

## C. "Missing but useful" ideas surfaced by the mining (not yet in the system)
- **Goal-bounds PRE-FLIGHT as code** (budget/value-floor/wall-clock-anchor/stop-conditions checked mechanically before each run) — currently prose.
- **Self-summon cadence as loop nodes** (oracle/auditor at 25/50/75% + plateau) — currently prose.
- **Build→run→learn→pivot** as an explicit FSM (maps directly onto the replanner, N3).
- **Value-needle test each cycle** (is this node moving the needle?) — pairs with the eval harness (N6).

This file is N1. Build order by EV: N3 → N4 → N5 → N6 → N7, each dispatch→RWYB→commit; N9 (DSPy/OpenEvolve) if the window allows.
