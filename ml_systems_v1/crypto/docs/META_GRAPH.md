# Meta-Graph — the genuine persistent meta-orchestrator (LangGraph) (2026-06-05)

> The fix for the real gap: a chat instance is **linear** (plan → act → halt). The "awake loop" cannot be the
> LLM — it must be a **PROGRAM**. `scripts/autonomy/meta_graph.py` is that program: a LangGraph state-machine
> whose **runtime is the awake loop**, calling a (pluggable) Claude "brain" as a stateless decision-function per
> node. It supersedes the prose-`OVERSEER.md` + Stop-hook re-invocation trick for **unsupervised** runs.

## Why this exists (the gap, named)
- A linear instance + a Stop hook = halt → re-invoke → halt. It only *looks* continuous; nothing is awake.
- The thing that stays awake must be a **program** that holds state, dispatches work, judges results, reflects
  lessons into evolution, **generates adjacent problems dynamically**, routes, and **checkpoints** (survives +
  resumes). That is exactly what a LangGraph `StateGraph` runtime does. The LLM is a node it consults.

## Architecture
```
State (MetaState): objective, success_criteria, frontier[], ledger[] (append-only lessons), budget, cycle, status

START → plan → dispatch → judge → reflect → route ─(open nodes & budget)→ dispatch   ← the awake loop
                                                  └─(solved | budget spent)→ END
```
- **plan** — seed the n±k frontier from the objective (once).
- **dispatch** — pick the top-EV open node; run the **brain** (worker) on it.
- **judge** — evaluate the result against `success_criteria` (refuse false victory).
- **reflect** — distil a lesson → append to `ledger`; **GENERATE ADJACENT PROBLEMS** (new frontier nodes) →
  this is the dynamic "produce adjacent problems" behaviour a linear instance can't sustain.
- **route** — conditional edge: continue while open nodes + budget; else END.
- **checkpointer** — `MemorySaver` (in-process) or **`SqliteSaver`** (durable, cross-process). The state survives;
  the program resumes — *not* a chat instance.

## The pluggable brain (the GRAPH is real regardless of which brain runs)
| backend | when | how |
|---|---|---|
| `AnthropicBrain` | `ANTHROPIC_API_KEY` set | anthropic SDK, real Claude |
| `CliBrain` | `claude` on PATH | `claude -p` headless (real Claude, no API key) |
| `MockBrain` | else (default) | deterministic, role-aware — proves the loop with **no credentials** |
Each node calls `brain.think(role, payload) -> dict`. Swapping the brain swaps the intelligence; the loop is
unchanged. (The repo currently has neither key nor CLI → the demo runs on `MockBrain`, which is why the loop is
fully exercised without secrets. For real autonomy, set `ANTHROPIC_API_KEY` or put `claude` on PATH.)

## Run it
```
pip install langgraph langgraph-checkpoint-sqlite          # one-time
# awake loop, mock brain, generates adjacent problems, terminates itself:
python scripts/autonomy/meta_graph.py --objective "..." --budget 8
# durable cross-process resume (proves it SURVIVES, not a chat instance):
python scripts/autonomy/meta_graph.py --durable --budget 2 --thread t1      # partial
python scripts/autonomy/meta_graph.py --durable --resume --budget 6 --thread t1   # loads checkpoint + CONTINUES
# real Claude:
ANTHROPIC_API_KEY=... python scripts/autonomy/meta_graph.py --backend api --objective "..."
```
Verified 2026-06-05: the loop runs end-to-end, dynamically generates adjacent nodes, hits both termination modes
(solved / budget), and **resumes across separate processes carrying the frontier + lesson-ledger** (5 cycles,
5 lessons across the boundary).

## How it relates to the rest of the framework
- **Supersedes** (for unsupervised runs): the prose `OVERSEER.md` discipline + the `autonomy_loop.py` Stop-hook
  re-invocation trick + the crude `autonomy_driver.py` (`while True` + `claude -p`). Those simulated a loop inside
  a linear instance; this **is** the loop, as a program.
- **Keeps**: the n±k / EV-frontier / verify-gate / write-forward *concepts* — they are now graph nodes + State,
  not prose. `frontier.json` ≈ State; `OVERSEER.md` ≈ the node contracts; `memory/` ≈ the durable ledger/store.
- **Interactive vs autonomous**: this graph is the **unsupervised** orchestrator (a standalone process). The
  Claude Code chat remains for **supervised** work. "Deploy the agent framework on a sub-instance" = the graph
  spawns Claude worker-calls; the graph (not a chat) is the persistent meta.

## Honest limits / next
- The `MockBrain` proves the *machinery*, not the *intelligence* — real judgment needs a real brain (key/CLI).
- `judge`/`reflect`/`plan` node prompts are minimal contracts; harden them when wiring the real brain.
- Worker nodes are currently single calls; for parallel fan-out, a node can spawn sub-graphs or use the Workflow/
  sub-agent tools. The graph shape (state + conditional cyclic edges + checkpointer) is the durable foundation.
