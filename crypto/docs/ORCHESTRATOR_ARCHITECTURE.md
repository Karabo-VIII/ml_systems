# Every Instance is an Orchestrator over a Persistent Solver — the architecture

> **Mandate (user, 2026-06-06):** *"Make every instance the orchestrator: whichever query I submit must be solved by
> the langgraph instance; the main invoked instance/skill just passes instructions, monitors, gives feedback, guides.
> Because I keep going through loop and loop and never achieve the true autonomy and self-reliance I'm looking for."*
> Designed via multi-agent SOTA-2026 research (Workflow `w8vcvvylc`, 7 agents) + a RWYB repo audit; **overseer-judged.**

## The diagnosis — why the loop dies (one root cause)

The Claude instance is **both planner AND executor**, and a chat instance is **linear (plan → act → halt)** with the
work bound to a finite, ephemeral context the turn-end discards. So when my turn ends, the work stops and you re-prompt.
The "loop" today is the **Stop hook re-injecting text into the *same* interactive instance** — not a separate durable
worker. Hence `loops_alive=[]`: the watcher guards a worker that was never started.

## The SOTA-2026 convergence (all sources agree on the fix)

Move the durable loop **OUT of the LLM into a PROGRAM/RUNTIME**; make the orchestrator a thin planner whose every
artifact lives in **external durable state**.
- **Orchestrator-worker separation** (Anthropic multi-agent research system): the lead agent *never executes* — it
  plans, **saves the plan to Memory**, decomposes into contracted subtasks, dispatches subagents that **persist outputs
  to the filesystem and return lightweight references**, then synthesizes.
- **Durable execution** (LangGraph checkpointer · Temporal deterministic-replay · ACP checkpoint-on-delegation): the
  *runtime* (the graph), not the LLM, is the awake loop; it resumes from the last checkpoint after a crash / pause / turn-end.
- **Dormant-session warm-restart** (Claude Agent SDK): sessions persist on disk and resume by id.

## The key finding — the durable solver ALREADY EXISTS (do not re-invent)

`scripts/autonomy/meta_graph.py` is a genuine **LangGraph `StateGraph`** (`plan → dispatch → judge → reflect → route`)
with a `MemorySaver` checkpointer and a **pluggable Claude brain** (Mock / CliBrain / AnthropicBrain). Its own docstring
states the exact thesis the SOTA confirms: *"the awake loop cannot be the LLM — it must be a PROGRAM… the GRAPH holds
state, dispatches, judges, reflects, routes, and CHECKPOINTS so it survives + resumes."* The metaop manager
(`run_metaop.py`/`manager.py`) wraps it with a per-thread **SqliteSaver** durable path + cross-process resume (verified).

**So the gap is NOT a missing runtime — it is WIRING:** today the interactive instance self-executes (or dispatches to
*ephemeral* Agent/Workflow subagents) instead of running the **persistent** metaop loop in the background and only
monitoring it.

## Overseer corrections to the research (RWYB)

1. The "no working brain" blocker is **overstated**: `CliBrain` works via the bundled `claude.exe` (`find_claude()`) —
   this session's metaop loops ran on it and produced real commits. The real constraint is **speed (~300×)**, not "no
   brain." `AnthropicBrain` would be faster but needs `langchain_anthropic` + an API key (genuine, optional).
2. The **automata-theory lens** is the right correctness discipline (user, 2026-06-06): model the orchestration as an
   **explicit FSM** — `IDLE · DISPATCHED · WORKING · BLOCKED(wake) · JUDGING · DONE` — with a **liveness invariant**
   (the watcher = the monitor). The "empty-frontier → STOP" bug was an *unintended accepting state* an explicit FSM
   catches at design time.

## Target architecture

```
EVERY query → main instance = THIN ORCHESTRATOR (FSM):
  sharpen objective → DISPATCH to the persistent metaop loop (background, durable) → MONITOR (watcher/loop_health)
   → JUDGE returns (RWYB) → GUIDE / re-feed → commit. NEVER self-executes the primary work.
PERSISTENT SOLVER = meta_graph/metaop run in the BACKGROUND (run_in_background / windowless daemon), SqliteSaver-durable,
  surviving the orchestrator's turn ending; re-ticked by the Stop-hook driver + the watcher's stall/dead exit codes.
SHARED DURABLE STATE = frontier.json (the plan) + per-thread SqliteSaver checkpoints + externalized artifacts (files;
  nodes carry lightweight references, never the bulk work).
```

## Implementation roadmap (EV-ranked; the new frontier — built incrementally, overseer-judged)

| EV | effort | step | SOTA pattern |
|----|--------|------|--------------|
| 0.90 | S | spawn a background metaop thread from `launch_autonomy --mode attended` (durable) so the solver runs after the turn ends | orchestrator-worker + durable |
| 0.88 | M | rewire the Stop hook from "re-feed text to the same instance" → "**resume the durable metaop process**" when one exists | durable execution / resume |
| 0.85 | S | default ALL launches to `--durable` (SqliteSaver), not MemorySaver — true cross-restart durability | LangGraph checkpointer |
| 0.82 | M | watcher stall/dead exit codes → **automatic manager resume** (not just "wake the overseer") | Temporal/ACP auto-resume |
| 0.78 | M | a pre-dispatch shim so EVERY non-trivial query routes to the persistent loop, not just `/orc` | every-instance-orchestrator |
| 0.70 | S | enforce externalized artifacts (workers write files; frontier nodes hold references) | artifact pattern |
| 0.66 | S | OVERSEER approval = an **async checkpoint** (`awaiting_approval`), not a turn boundary | HITL-as-checkpoint |
| 0.60 | S | speed: prefer `AnthropicBrain` (install `langchain_anthropic` + key) OR the `--backend persistent` warm session over per-node `claude -p` | brain speed |
| 0.50 | L | (optional) an EXTERNAL scheduler tick (cron/daemon) for unattended cross-machine, no human re-prompt | external scheduler |

## Honest limits
- **Speed:** the persistent loop on `CliBrain` is slow (~300×). True unattended autonomy is *reliable but slow* until a
  faster brain (`AnthropicBrain`/warm-session) is wired.
- **The external-scheduler gap:** with no open Claude session AND no daemon, nothing re-ticks the loop — the Stop-hook
  keep-going needs an *open session*; a true cron/daemon scheduler (step 9) closes the last mile but adds a surface.
- **Not a magic brain:** this makes the loop *persistent + durable*, not *smarter* — the reasoning quality is still the
  fixed Claude brain. The win is decoupling the work from any single ephemeral turn.

*Sources: Anthropic "How we built our multi-agent research system"; LangGraph durable-execution/persistence docs;
Temporal agentic-workflow durability; humanlayer Agent Control Plane; Claude Agent SDK session persistence. Full
research: Workflow `w8vcvvylc`.*
