# Harness Evolution Roadmap — SOTA-grounded self-improvement upgrades (2026-06-06)

> **Provenance.** User mandate 2026-06-06: *"get all the ideas from [Two Minute Papers], all the papers it references,
> so that we can improve our harness and thinking and self-evolution and looping methodology."* Sourced from an
> agent-literature survey (RWYB: every paper has an arXiv ID / venue below) + a parallel Two-Minute-Papers channel
> mining pass. Composes with [LOCAL_AUTONOMY_ARCHITECTURE](LOCAL_AUTONOMY_ARCHITECTURE_2026_06_06.md) — these are the
> *intelligence* upgrades to the local harness whose *execution plane* (fast SDK brain + fence) is already built.

The harness today: a LangGraph StateGraph (`scripts/autonomy/metaop/graph.py`) + flat-append memory
(`rolling_ledger.py`) + a static skill library (`skill_library.py`) + a fast fenced brain (`AgentSdkBrain`). The SOTA
says the next gains are in **memory, compute-routing, and self-evolution** — each maps to a specific change here.

## Prioritized upgrades (ROI × implementation-cost × safety-given-our-gates)

| # | Upgrade | Paper (RWYB source) | Concrete change in OUR harness | Honest caveat |
|---|---|---|---|---|
| 1 | **Mem0-style memory** | Mem0, arXiv 2504.19413 (SOLID, prod) | Replace flat-append `rolling_ledger.py` with an extract→update cycle + a **vector retrieval store (FAISS) indexed by TASK SIMILARITY** (not time); retrieve top-3 similar past cycles as few-shot before each cycle. Points at a LOCAL dir (no cloud). Fixes "instances forget after compaction." | Extraction quality depends on the model; with a weak local brain it can hallucinate memories — add a confidence gate before write. |
| 2 | **Cascade routing + confidence-gated judge** | LLM-Cascade DT 2605.06350 + CAMEL 2602.20670 (PROMISING) | A `router` node before `dispatch`: cheap Sonnet first, **escalate to Opus only when confidence < τ** (token-negentropy / log-prob margin). `judge` node accepts a high-margin single-token verdict without reflection; invokes full Opus CoT only on low margin. | ~70-80% Opus-budget cut is on NLP benchmarks; τ needs empirical calibration on our code/numeric-validation tasks, not theory. |
| 3 | **EvoFSM FSM-diff evolution** | EvoFSM, arXiv 2601.09465 (PROMISING) | The LangGraph graph IS a finite state machine. After N failed cycles, the orchestrator emits **Flow operators** (ADD/DELETE/REWIRE edge — macro) + **Skill operators** (rewrite ONE node's prompt — micro); the experience pool = the 3-lane memory tagged with the FSM diff. **SAFEST self-improvement given our gate invariants** (controllable, inspectable diffs vs free-form code mutation). | arXiv-only Jan-2026, no independent replication; gains shown on QA/games not code. Add a forgetting policy (pool grows unboundedly). |
| 4 | **DGM-style skill archive** | Darwin-Gödel Machine 2505.22954 (SOLID) | Upgrade `skill_library.py` from static named fns → a **population archive**: each skill = file + fitness score (held-out task-solve rate) + parent pointer; a new skill is **benchmarked before promotion**, kept only on improvement; sample diverse archive members to avoid local optima. The `reflect` node becomes the mutation engine. | Each mutation re-runs a benchmark (expensive); needs an automated eval harness PER DOMAIN. **NEVER let it mutate the commit/permission gates — HITL there.** |
| 5 | **AlphaEvolve island evolution** | AlphaEvolve 2506.13131 (SOLID, prod) | For DISCOVERY (engine variants): keep a population (5-10) of engine variants, mutate via the meta-agent, score with an **automated fitness fn** (Elo for chess; held-out Sharpe/return for crypto), keep top-k. **Sonnet = breadth mutations (many, fast); Opus = deep architectural proposals (few)**. Island model resists collapse better than single-lineage DGM. | Gemini-gated results; expect 10-30% (REPORTED, paper) of headline numbers in a reimpl. The fitness fn is the hard part, not the mutation engine. |
| 6 | **Kronos feature extractor** | Kronos 2508.02739 (PROMISING) | CRYPTO-only: use the financial-TS foundation model as a **frozen feature extractor** (fine-tune a thin head) feeding the V-series WM — not a competitor architecture. | Pre-trained on time-based K-line; our dollar bars need a tokenizer adaptation. +93% RankIC (REPORTED, paper) is vs a WEAK TSFM baseline — do not cite without that flag. |

## The honest meta-lesson (across all of them)
**The fitness/eval function is the hard part, not the mutation engine.** DGM + AlphaEvolve are the most production-
evidenced self-improvement results, and both hinge on an *automated domain evaluator*. We already have natural
evaluators (chess perft/Elo; crypto held-out compound/Sharpe via `src/strat`) — that is our enabler. The other lesson:
self-improvement has real failure rates (REPORTED, paper — Gödel Agent: 4% terminations, 92% temporary perf drops) — **every self-update
needs a snapshot+rollback, and must never touch the gates.**

## Build order (lowest-risk → highest-ceiling)
1. **Mem0 memory** (item 1) — highest ROI, drop-in, fixes the forgetting problem. → frontier `he_mem0_memory`
2. **Cascade routing + gated judge** (item 2) — cuts Opus budget every cycle, no quality loss. → `he_cascade_router`
3. **EvoFSM FSM-diff** (item 3) — safest self-improvement under our gates. → `he_evofsm_evolution`
4. **DGM skill archive** (item 4) — highest capability ceiling; gate it behind a per-domain eval harness. → `he_dgm_skill_archive`
5. **AlphaEvolve islands** (item 5) — for discovery once single-frontier search saturates. → `he_alphaevolve_islands`

## Channel additions (Two Minute Papers — faithful mining, 2026-06-06)
Anchor video "DeepMind's New AI Found A Strange New Way To Think" = **AlphaProof Nexus** (arXiv 2605.22763): an
LLM↔formal-verifier loop — Gemini proposes a step, **Lean 4 mechanically verifies**, a REJECTION is fed back as the next
prompt. Solved 9/353 open Erdős problems (REPORTED, paper) at ~a few hundred USD each. **Single highest-leverage idea for us.**

**THE KEY SYNTHESIS (why `chess_validate` just failed + the #1 fix).** Our metaop `judge` node is an LLM VOTE
(inconclusive/refuted) — NOT a mechanical verifier that RUNS the artifact and feeds the concrete error back. AlphaProof
Nexus shows the grounding oracle (Lean) + rejection-as-gradient is what makes the loop actually DELIVER. **Our backtester
/ a unit test / `python perft.py` IS our Lean.** Fix: the judge must RUN the artifact and feed the concrete failure
("perft.py missing" / "depth-3 got X, want 8902") back as the next worker prompt → iterate. This is the leading
explanation for the validation failure (the loop reasoned but got only vote-feedback, never a mechanical error to act on).
→ frontier `he_verifier_loop` (NEW, #1 priority).

Other channel ideas worth banking: **Titans surprise-weighted memory** (arXiv 2501.00663) — weight ledger entries by
surprise/contradiction, not chronology (refines `he_mem0_memory`); **PARC bootstrap** (arXiv 2505.04002) — generate →
backtest-filter → retrain, for cold-start discovery; **Emergent Misalignment from reward-hacking** (Anthropic, arXiv
2511.18397) — EMPIRICAL backing for this roadmap's "a self-update must NEVER touch its own eval/gate" caveat (reward-
hacking generalizes to sabotage/deception); **test-time compute scaling** — a `reasoning_budget` per node (more on high-EV).

Sources: [AlphaProof Nexus 2605.22763](https://arxiv.org/html/2605.22763v1) · [Titans 2501.00663](https://arxiv.org/abs/2501.00663) · [PARC 2505.04002](https://arxiv.org/abs/2505.04002) · [Emergent Misalignment 2511.18397](https://arxiv.org/abs/2511.18397) · [Mem0 2504.19413](https://arxiv.org/abs/2504.19413) · [LLM Cascades 2605.06350](https://arxiv.org/html/2605.06350) · [CAMEL 2602.20670](https://arxiv.org/abs/2602.20670) · [EvoFSM 2601.09465](https://arxiv.org/abs/2601.09465) · [Darwin-Gödel Machine 2505.22954](https://arxiv.org/abs/2505.22954) · [AlphaEvolve 2506.13131](https://arxiv.org/abs/2506.13131) · [Kronos 2508.02739](https://arxiv.org/abs/2508.02739) · [Memory survey 2603.07670](https://arxiv.org/html/2603.07670v1) · [ADAS 2408.08435](https://arxiv.org/abs/2408.08435) · [Gödel Agent 2410.04444](https://arxiv.org/abs/2410.04444)
