# Orchestration Layer — how we call & use LangGraph + autonomous mode (2026-06-12)

> Scope: this maps the **orchestration layer** end-to-end — how the project's LangGraph (metaop) implementation and
> autonomous mode are *called and used* — and documents the **local brain wiring** (Qwen on Ollama). It is a reader's
> map, not new machinery. Source: the `orchestration-layer-map` workflow (3 parallel readers) + RWYB on this box.
> Companion: [`docs/SYSTEM_TOPOLOGY.md`](SYSTEM_TOPOLOGY.md) (the control-surface hierarchy), `MEMORY.md` autonomy entries.

---

## 0. TL;DR (the one thing to get right)

**"Autonomous mode" is TWO substrates that share state, chosen by `--mode`:**

| | **ATTENDED** (default) | **UNATTENDED** (`--mode unattended` / `--spawn-loops`) |
|---|---|---|
| Loop driver | the **Stop hook** (`.claude/hooks/autonomy_loop.py`) re-feeds the present Claude | detached **LangGraph metaop** processes (`run_metaop.py`) |
| LangGraph runs? | **NO** — the StateGraph is not spawned | **YES** — `plan→dispatch→judge→reflect→route` with a SqliteSaver checkpoint |
| Who solves | the present Claude as Tier-0 OVERSEER, dispatching to in-harness `Agent` workers | `claude -p` / brain workers inside the graph |
| Survives session end | no (needs the live session) | yes (durable checkpoint + detached procs) |
| Queue artifact | `runs/autonomy/frontier.json` (flat node list the hook reads) | `OpState.frontier` (in-checkpoint, **separate object**) |

**The #1 trap:** `frontier.json` (Stop-hook queue) and `OpState.frontier` (LangGraph in-memory frontier) are **different,
unsynchronized objects**. The metaop graph never reads `runs/autonomy/frontier.json`. Don't assume one shared queue.

**The brain** is a pluggable seam *separate* from the graph: the **graph is the awake loop, the brain is what thinks
per node.** Swapping the model touches only `make_brain` (one function). The local brain is **Qwen-Coder on Ollama**;
as of 2026-06-12 the default is **`qwen2.5-coder:7b`** (was `:3b`) — the most capable that fits this 8GB GPU as a
single resident model. See §5.

---

## 1. Entry points

| Entry | File | Role |
|---|---|---|
| **`/orc` skill** | `.claude/skills/orc/` | the user-facing default modus operandi; adopts an objective, stands up the 3 loops. Drives `launch_autonomy.py`. |
| **`launch_autonomy.py`** | `scripts/autonomy/launch_autonomy.py:228` (main) | THE 3-loop gate. Writes `frontier.json`, arms the Stop hook (`AUTONOMY_ON`), spawns the watcher + meta/evolution loops; in `unattended` also spawns the metaop solver loops. **Now also selects + exports the local brain model (`OLLAMA_MODEL`).** |
| **`run_metaop.py`** | `scripts/autonomy/run_metaop.py` | 19-line PYTHONPATH-free shim → `metaop.manager.main()`. Every detached metaop process starts here. Verbs: `launch/resume/status/approve/stop/learnings`. |
| **`metaop.manager`** | `scripts/autonomy/metaop/manager.py` | supervisory CLI over the graph: `launch` (lease → `make_brain` → `apply_champion` → build graph → `app.stream`), `resume` (reload SqliteSaver), `approve` (HITL), `stop` (kill tree, keep checkpoint). |
| **`autonomy_loop.py`** (Stop hook) | `.claude/hooks/autonomy_loop.py:196` | the mechanical keep-going engine. On every Stop event, if armed + work remains, returns `{"decision":"block","reason":<next OVERSEER cycle>}` so the harness re-feeds work. **A summary is not a stop.** This IS the attended-mode loop. |
| **`watcher.py`** + `ensure_watcher.py` | `scripts/autonomy/watcher.py:157` | the absolute 60s liveness watcher (loop-3 substrate). Discovers locks, checks process liveness, writes `*.flag` signals; singleton-guarded; self-respawns. Monitor-only (no auto-relaunch). |
| **`permission_gate.py`** (PreToolUse hook) | `.claude/hooks/permission_gate.py:62` | hot-reloadable perms + **worker fence**: when `METAOP_LOOP=1`/`HARNESS_WORKER=1` is in env it DENIES git commit/push and control-surface writes — loop children stage, the OVERSEER commits. |

---

## 2. The LangGraph implementation (REAL, not prose)

Lives **once** in **`harness/metaop/graph.py:672-687`** — a `langgraph.graph.StateGraph(OpState)`.
`scripts/autonomy/metaop/graph.py` is a thin **crypto shim** (re-exports `make_nodes`/`build` + injects
`cwd=ROOT`, `workspace=runs/autonomy`, `persona_dir=.claude/agents`, and the four anti-drift host hooks
`_crypto_framer/_crypto_recaller/_crypto_recorder/_crypto_harvester`). **Edit the canonical `harness/metaop/graph.py`,
not the shim.**

**State** — `OpState` (TypedDict, `graph.py:34`): `objective, success_criteria, frontier[], ledger (Annotated +reducer),
budget, cycle, status, parallel, run_id, awaiting_approval` + replanner bookkeeping (`done_count, stall_cycles,
replan_count, replan_reason, drain_empty`).

**Nodes** (closures from `make_nodes`, `graph.py:338`):
- **plan** — `brain.decide('plan')` → EV-ranked frontier of `build/verify/diverge` nodes; runs `_self_critique_plan`
  (adds a missing falsifier/generalization — the n±k breadth guard) + `_seed_verify_defaults` (build nodes get
  `verify_retries=2`, a `verify_missing` flag if no `verify_cmd`). One-shot guard: skips if frontier already seeded.
- **dispatch** — parks irreversible nodes for HITL (`_is_irreversible` → `awaiting_approval`), then runs up to
  `parallel` highest-EV nodes **concurrently** in a `ThreadPoolExecutor` calling `brain.work(task)`.
  Rejection-as-gradient: a prior `verify_error` is appended to the task so the worker fixes the **real** error.
- **judge** — **MECHANICAL VERIFIER FIRST**: a node's `verify_cmd` runs from `build_cwd`; **exit 0 = ground-truth
  pass** (overrides the LLM panel) + harvest into the skill library; exit≠0 = `refuted` (re-opened if retries remain,
  else terminally recorded as a dead vein). No `verify_cmd` → an N-judge LLM vote with **H3 evidence-typing** (an
  unverified LLM "pass" on a checkable node auto-downgrades to `inconclusive`).
- **reflect** — `brain.decide('reflect')` → a transferable lesson (persisted to a learnings channel) + adjacent nodes;
  tracks `stall_cycles`; sets `status='solved'` only when no open AND no refuted nodes remain.
- **replan** — `brain.decide('replan')` → revised frontier merged KEEP/PRUNE/ADD via `_merge_replan` (LangGraph
  plan-execute pattern). Serves both stall/failure recovery and `fill_window` drain-replanning. Cap `max_replans=3`.
- **budget** — terminal sink → END.

**Edges** (`graph.py:677-686`): `START→plan→dispatch→judge→reflect`; `reflect` has conditional edges via `route()`
→ `{dispatch, replan, budget, END}`; `replan→dispatch`; `budget→END`. Steady cycle = `dispatch→judge→reflect→route`.

**Checkpointer**: `MemorySaver` by default; `manager._checkpointer` swaps in
`langgraph.checkpoint.sqlite.SqliteSaver` at `runs/autonomy/metaop_<thread>.db` when `--durable` → cross-process
resume via `app.get_state(cfg)` / `app.update_state(cfg)` keyed on `configurable.thread_id`.

**Observability**: every node/decision appends a JSONL line to `runs/autonomy/traces/<run_id>.jsonl` (`_trace`).

---

## 3. Autonomous mode — arming, driving, releasing

**Two arming switches** (Stop hook):
- **A. `runs/autonomy/AUTONOMY_ON` exists** → frontier-driven loop (**the live path** — `launch_autonomy` writes this
  and deletes `autonomous_mode.json`, so this is what actually runs; prompt-free because it's outside `.claude/`).
- **B. `.claude/autonomous_mode.json` `{autonomous:true, envelope_end}`** → timed loop (authority file per W3).
  `AUTONOMY_ON` arms **only** when `autonomous_mode.json` is absent. Mutually exclusive.

**The Stop hook is the engine.** After every response it: (1) allow-stop if `stop_hook_active` / `AUTONOMY_LOOP=off`
/ neither switch active; (2) computes `loop_active`; (3) **P4 SAFE-default 6h window** anchored on first fire +
persisted to `loop_progress.json` (so the switch path can't loop forever); (4) **global anti-stuck gate** — stall
measured in wall-clock idle TIME (`STALL_IDLE_SECONDS=1800`, `STALL_LIMIT=3`), with an exemption for a legit in-flight
build; (5) if work remains → `block()` with the full OVERSEER-cycle instruction; (6) releases on window expiry /
`budget.spent ≥ max_cycles` or `HARD_CEIL=500` / frontier exhausted — **but never silently if a tracked long job is
alive**: it converts to a bounded WAIT-MODE (one health check then end the turn — the anti-silent-death guarantee).

**Survival**: (a) detached metaop + SqliteSaver checkpoint (unattended, cross-session); (b) `ensure_watcher.ensure()`
called from BOTH the Stop hook and the UserPromptSubmit hook every turn; (c) watcher self-respawn + singleton lock;
(d) long jobs registered via `track_job.py` are seen by hook + watcher; (e) all state files written atomically
(tmp + `os.replace`) so a concurrent reader never sees a torn file (the H2 hardening).

---

## 4. The brain seam (pluggable per-node intelligence)

Interface (`harness/metaop/brain.py`): `decide(role, payload, persona) -> dict` (plan/judge/reflect/replan/
plan_critique), `act(...)` (one ReAct step), `work(task, persona) -> {ok, result}`.

`scripts/autonomy/metaop/brain.py` is a **shim** that (1) re-exports every Brain class, (2) injects `CRYPTO_DOMAIN`
into the `{domain}` prompt slot (LONG-ONLY spot/perp; objective = robust held-out COMPOUND return; SETUP across a
multi-candle MOVE — per-bar IC NOT the target), and (3) at import sets `os.environ["METAOP_LOOP"]=1` so every worker
child is fenced by `permission_gate`.

**`make_brain(kind, domain, model)`** (`harness/metaop/brain.py:778`) selects the backend:

| kind | backend | notes |
|---|---|---|
| `auto` | AgentSdkBrain → AnthropicBrain → CliBrain → MockBrain | prefers in-process Claude (no key); MockBrain is deterministic, no creds |
| `cli` / `persistent` | CliBrain / PersistentCliBrain | headless `claude -p`, existing auth; persistent carries ONE session across nodes. **The spawned loops use `--backend cli`.** |
| `api` | AnthropicBrain (or via LiteLLM) | needs `ANTHROPIC_API_KEY`; default Claude model `claude-opus-4-8` |
| **`ollama`** | **OllamaBrain (or via LiteLLM)** | **LOCAL open-source model — proof the harness outlives Claude** |
| `litellm` | LiteLLMBrain | unified gateway, auto-fallback primary → local ollama → MockBrain |
| `cascade` | CascadeBrain | cheap→strong escalation; `set_node_context` gives it the node's `verify_cmd` so cheap-accept uses the same ground truth as judge |

**Model override env vars**: `HARNESS_MODEL` (Claude), `LITELLM_MODEL` (litellm string), `OLLAMA_MODEL` (ollama tag).
The graph never imports a model — it only depends on the Brain interface, so a model swap touches only `make_brain`.
`apply_champion(brain)` (`manager.py:162`) GATED-installs an evolved planner prompt (`_PLAN_INSTRUCTION` = the
DSPy/evolve seam) IFF `runs/autonomy/evolve/champion.json` exists AND beats baseline.

---

### 4.1 Tooling — does the brain have hands? (YES, RWYB-verified 2026-06-12)

A raw LLM is just text. The harness gives EVERY brain **hands** via a real tool executor —
**`harness/metaop/tools.py`** (`class Tools`) — driven by the brain's **`work(task)` ReAct loop**
(`act → tool call → observe → … → final`). Seven tools, all SSRF/fence-guarded:

| tool | what it does |
|---|---|
| `run_shell(command)` | run a shell command in the build cwd (300s cap, stdout+stderr captured) |
| `run_python(code)` | write a scratch `.py` and execute it (multiline-safe) |
| `read_file(path)` / `write_file(path,content)` / `list_dir(path)` | filesystem in the build cwd |
| `web_search(query)` | public-web search — Brave API if `BRAVE_API_KEY` set, else DuckDuckGo instant-answer (entity-only) |
| `fetch_url(url)` | fetch + strip any public http(s) page (SSRF-guarded: no localhost/private/metadata) |

**Backend matters:**
- **Local brains** (Ollama `qwen2.5-coder:7b` / LiteLLM) → exactly these **7 harness tools** via the ReAct loop.
- **Claude brains** (AgentSdk / Cli / Anthropic) → ALSO the **full Claude Code tool surface** (Read/Write/Bash/Grep/…)
  on top, each with its own PreToolUse safety fence (`AgentSdkBrain._safety_fence_hook`).

**Always-on safety fences** (independent of backend): `HARD_DENY` (shell — `rm -rf /`, `git push --force`,
`git reset --hard`, `mkfs`, `sudo`, fork-bomb, …) + `HARD_FILE_DENY` (writes to `.git/`, `.env`, `.ssh/`, `id_rsa`)
+ the SSRF guard on web + the **outer `permission_gate` worker fence** (a loop worker with `METAOP_LOOP=1` cannot
commit/push or touch control surfaces — the overseer commits). So the brain can DO real work but cannot do
irreversible/destructive things.

**RWYB (2026-06-12):** `Tools` self-test — all 7 execute, both fences fire (`git push --force` denied, `.git/`
write denied). Local 7b `work()` — ran `python --version` end-to-end and returned `Python 3.11.9` (22s). Web —
`fetch_url(example.com)` + `web_search(bitcoin)` returned real content; `fetch_url(localhost:11434)` SSRF-refused.

> **One real gap:** `web_search` is **entity-only without `BRAVE_API_KEY`** (DDG instant-answer ≠ general web). For
> the local brain to do genuine web research (the project's "web tools are first-class" invariant), set
> `BRAVE_API_KEY` (2000/mo free tier). `fetch_url` already reaches any public page regardless.

---

## 5. Local brain wiring — Qwen-Coder on Ollama (the 2026-06-12 change)

**Is a Qwen brain wired? YES.** Ollama (0.30.7) serves `qwen2.5-coder:7b` (4.7GB) and `qwen2.5-coder:3b` (1.9GB),
both pulled; `nomic-embed-text` is the Mem0 embedder. The `ollama`/`litellm` backends route the metaop brain to it.

**What changed (user mandate: "wire the most capable model we can run locally").** The default was `qwen2.5-coder:3b`;
it is now **`qwen2.5-coder:7b`** — the most capable model that fits this RTX 4060 (8188 MiB) **as a single resident
model**. Two layers:

1. **Engine default** (`harness/metaop/brain.py`): the four `OLLAMA_MODEL` fallbacks / `LITELLM_DEFAULT_OLLAMA` now
   default to `qwen2.5-coder:7b` (env vars still override). Parity-safe: the copy-parity firewall checks *symbols*,
   not default values.
2. **VRAM-aware selector** (`scripts/autonomy/ensure_brain.py:best_local_model` + `MODEL_LADDER`): picks the
   most-capable PULLED model whose footprint fits the GPU's **total** VRAM (gate on total, not free, because the brain
   should own the GPU — one resident model, not two). `launch_autonomy.py` calls it and **exports `OLLAMA_MODEL`** so
   every spawned loop inherits the choice; `--brain-model` forces one. On an 8GB card → `7b`; degrades to `3b` on a
   <6.5GB GPU; falls back gracefully with no NVIDIA probe.

**Why "single resident model" matters (the diagnosed failure).** With BOTH `3b` and `7b` loaded simultaneously
(~6.9GB) plus other GPU processes, the 8GB VRAM oversubscribed and a `7b` call thrashed/timed out (>300s). The fix is
to standardize on ONE model: with only `7b` resident (4.7GB) the GPU has headroom and inference is healthy. The
selector + the single default enforce this.

**Run it / verify it:**
```
python scripts/autonomy/ensure_brain.py                       # auto-selects the best-fitting pulled model + live test
python scripts/autonomy/ensure_brain.py --model qwen2.5-coder:7b   # force 7b
python scripts/autonomy/launch_autonomy.py --objective "..." --brain-model qwen2.5-coder:7b   # force for the loops
```
`ensure_brain.py` exit 0 ⟺ a live `LiteLLMBrain.decide()` returned a real dict from the local model (the RWYB anchor).

> **VRAM caveat for concurrent instances.** 8GB fits ONE 7b with headroom — it does NOT fit two models. If another
> instance/loop holds a model (or heavy GPU compute), either let ollama LRU-evict (5-min keep-alive) or `ollama stop`
> the stale one before the 7b loop runs. Don't co-load 3b + 7b.

---

## 6. Known gaps / fragilities (from the map — carry forward, don't re-discover)

1. **Dual loop / dual frontier** — attended mode runs NO LangGraph; `frontier.json` ≠ `OpState.frontier` (unsynced).
2. **Window is advisory on the switch path** — `--hours N` is recorded but the Stop hook enforces its own 6h
   SAFE-default (+ `budget.max_cycles`/`HARD_CEIL`), not N hours, unless the OVERSEER self-disarms on `stop_conditions`.
3. **Shim duplication** — `harness/metaop/` (canonical) vs `scripts/autonomy/metaop/` (shims), kept in sync by
   `_test_copy_parity.py`. Easy to edit the wrong file; the parity firewall warns but is a maintenance cost.
4. **`verify_missing` → belief** — an unverified node downgrades to `inconclusive` but still terminates (`status=done`),
   so a run can report "solved" on LLM-believed-but-unverified nodes. The `evidence_type` tag is the only signal.
5. **`fill_window` no-idle-stop** — drained frontier triggers ≤2 drain-replans for "adjacent work"; on a weak brain
   this risks low-value busywork (the over-mining trap) bounded only by the cycle budget.
6. **Commit-lease still prose** — `permission_gate` fences loop workers, but a concurrent loop's `git add -A` can still
   sweep the overseer's staged files (CONFIRMED recurring). Mechanical commit-lease is design-ready, not in code.
7. **Watcher is monitor-only + Windows-fragile** — writes `*.flag`, doesn't auto-relaunch; PID/singleton logic leans
   on PowerShell `Get-CimInstance` + create-time guards that weaken off-Windows / without `proc_liveness.created_at`.
8. **env-var fence dependency** — commit safety rests on `METAOP_LOOP`/`HARNESS_WORKER` being inherited; a future
   backend that doesn't thread `os.environ` would silently open the fence.

---

## 7. Quick reference — how to drive it

```bash
# Stand up the 3-loop autonomous mode (attended = present Claude is the OVERSEER):
python scripts/autonomy/launch_autonomy.py --objective "<one line>" --success "<verifiable>" [--hours N]

# Cross-session / unattended (also spawns the durable LangGraph metaop solver loops):
python scripts/autonomy/launch_autonomy.py --objective "..." --mode unattended

# Report / resume the current launch:
python scripts/autonomy/launch_autonomy.py --status

# Drive the LangGraph metaop directly (the StateGraph):
python scripts/autonomy/run_metaop.py launch  --objective "..." --backend cli --durable --fill-window
python scripts/autonomy/run_metaop.py status   --thread <slug>
python scripts/autonomy/run_metaop.py resume   --thread <slug>-r
python scripts/autonomy/run_metaop.py approve  --thread <slug> --node <id>   # HITL release
python scripts/autonomy/run_metaop.py stop     --thread <slug>

# Ensure / verify the local brain (RWYB):
python scripts/autonomy/ensure_brain.py                 # auto-select best-fitting model + live test
```

Disarm autonomous mode: `rm runs/autonomy/AUTONOMY_ON` (or set `autonomous=false`).
