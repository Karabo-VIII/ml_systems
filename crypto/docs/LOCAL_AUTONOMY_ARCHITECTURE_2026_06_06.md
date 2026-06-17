# Local Autonomy Architecture — the harness that outlives the model (2026-06-06)

> **Provenance.** User mandate 2026-06-06, verbatim intent: *"I need the solution LOCALLY because I need the harness
> more than anything (in case Claude is no longer my subscription). I do need the loop to run autonomously and
> self-improve and one-shot instructions because I need the engines built one-shot (with Claude managing questions,
> ideas, execution, queries, etc) end to end."* This supersedes the cloud-hosted Managed-Agents recommendation — that
> is Anthropic-locked + per-token-billed, the OPPOSITE of a harness that survives a vendor change.

## The verified diagnosis (independent meta-oracle, 2026-06-06)
The prior autonomy work optimized the **control plane** (6 watcher fixes, Stop hook, gates) while the **execution
plane** was broken. Two structural truths, both RWYB-verified:
1. **The interactive Claude Code instance is turn-gated by the harness** — the Stop hook is self-`/loop`, it never runs
   between turns. No hook can make it autonomous. (So "the loop is dead → I type a new message" is the GROUND TRUTH,
   not a bug to fix.)
2. **The `claude -p`-per-node brain is ~300x too slow** to be a real solver (our own rolling ledger said so); the
   detached metaop loop hung 16 min blocked on one cold `claude -p` plan call. Producing 2 small files then hanging is
   "the machinery compiles", not "a solver works".

## SOTA-2026 grounding (what real implementations use)
- **LangGraph** = the LOCAL, **model-agnostic** orchestration runtime: durable execution via checkpointers (state saved
  after every node → crash/restart resumes from the last checkpoint), self-hostable, streaming, HITL. This is the
  micro-level reasoning-flow engine. ([LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview))
- **Temporal** = the 2026 standard for *crash-proof* **durable agent execution** (survives infra failure, waits days for
  approval; OpenAI uses it for Codex). The winning pattern is **LangGraph for reasoning + Temporal for the durable
  lifecycle**. Heavier; LangGraph's checkpointer saves BETWEEN nodes only (mid-node crash loses intra-node work) — that
  gap is the one thing Temporal fixes. ([Temporal vs LangGraph 2026](https://cordum.io/blog/temporal-vs-langgraph), [decision guide](https://agentmarketcap.ai/blog/2026/04/08/langgraph-vs-temporal-long-running-agent-workflows-2026))
- **LangSmith** = observability only; **LangGraph Platform** (`langgraph deploy`, Mar 2026) = the self-hosted server. Both
  local-capable + model-portable.

## The aligned architecture (LOCAL, portable, durable, autonomous, self-improving)
**Keep our LangGraph metaop graph as the harness; fix the brain + the driver.** What we already have, verified
installed: `langgraph` + `langgraph-checkpoint-sqlite` (durable, local) + a pluggable `Brain` interface
(`scripts/autonomy/metaop/brain.py`: `decide()`/`work()` with Anthropic/Cli/PersistentCli impls). The harness is the
durable asset; only the brain IMPL is model-specific.

| Layer | Component | Status | Survives Claude leaving? |
|---|---|---|---|
| Orchestration | LangGraph graph (plan→dispatch→judge→reflect→route) `metaop/graph.py` | BUILT, good | YES (model-agnostic) |
| Durability | SqliteSaver checkpointer (resume across restart/crash) | BUILT | YES (local file) |
| **Brain (the gap)** | pluggable `Brain` interface; impl = best available model | interface BUILT; **fast impl MISSING** | YES (swap impl) |
| Autonomous driver | a LOCAL daemon that steps the graph unattended to completion | **flaky (hung on slow brain)** | YES |
| Self-improvement | reflect/learn nodes + skill_library + 3-lane memory | BUILT | YES |
| Observability | dashboard.html (NRT) + loop_health | BUILT | YES |

**The honest crux — the brain.** A brain that is simultaneously (a) Claude-quality, (b) free, (c) survives-Claude-leaving
does NOT exist. The resolution: the HARNESS is the insurance (local, model-agnostic, yours); the BRAIN is hot-swappable.
- **Today's brain = Claude** via a FAST path (Agent SDK in-process OR a persistent `--resume` CLI session) on the
  SUBSCRIPTION (not per-token API) — fixes the 300x cold-start tax. (Note: from 2026-06-15 `claude -p`/Agent-SDK on
  subscription draws a separate monthly Agent-SDK credit — budget it.)
- **Tomorrow's brain (if Claude leaves) = swap the impl** to the best-then-available (a local model via Ollama/vLLM, or
  another API). Intelligence degrades gracefully; the harness is untouched.

## Build plan (multi-cycle; honest about what is unproven)
1. **Fast brain** — replace cold `claude -p`-per-node with a fast impl behind the `Brain` interface (Agent SDK or a
   persistent session). KILLS the hang + the 300x tax. *Prereq the user controls: a brain backend (subscription auth or
   key).* [the crux]
2. **Robust local driver** — a daemon that steps the graph unattended with a per-node TIMEOUT + fallback (never hang),
   writes durable checkpoints, exits cleanly. Not tied to the interactive session.
3. **Prove it one-shot** — the litmus: the AUTONOMOUS loop (not the attended instance) builds something real end-to-end
   from one instruction. (The chess engine was built by the ATTENDED instance — that distinction is the whole ballgame;
   do not blur it again.)
4. **Graceful model-swap** — wire an Ollama/local-model `Brain` impl as the portability proof (works, weaker).
5. **(Defer) Temporal** — only if mid-node-crash durability becomes a real pain; SqliteSaver between-node durability is
   enough for now.

## What we STOP doing (per the verdict)
Stop shipping watcher/hook fixes. Stop calling the `claude -p` loop the "walk-away solution". Stop conflating "the
machinery runs" with "the work gets done". The Stop hook stays ONLY as an attended-session convenience, never sold as
true autonomy.

Sources: [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) · [Temporal vs LangGraph 2026](https://cordum.io/blog/temporal-vs-langgraph) · [LangGraph vs Temporal decision guide](https://agentmarketcap.ai/blog/2026/04/08/langgraph-vs-temporal-long-running-agent-workflows-2026) · [Claude Managed Agents (rejected: cloud-locked)](https://platform.claude.com/docs/en/managed-agents/overview)
