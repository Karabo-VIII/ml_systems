# METAOP — the SOTA autonomous meta-operator (2026-06-05)

> The complete, production-shaped version of the awake-loop orchestrator. A **LangGraph program** that stays
> awake, plans, dispatches **tool-using worker agents in parallel**, **adversarially verifies**, **reflects +
> generates adjacent problems**, **escalates irreversible actions for approval (HITL)**, traces everything, and
> **survives + resumes** durably. The graph is the META that never sleeps; the brain (Claude) thinks per node;
> workers do real, fenced work. Package: `scripts/autonomy/metaop/`.

## Components (each verified independently)
| module | role | SOTA property |
|---|---|---|
| `tools.py` | the executor (shell / python / read / write / list) | every op screened by the **live deny-fence** (rm -rf, force-push, secrets, settings → blocked) |
| `brain.py` | pluggable intelligence (`decide` for nodes, `act` for ReAct) | `AnthropicBrain` (real Claude, retried, JSON) · `CliBrain` (`claude -p`) · `MockBrain` (proves machinery, no creds) |
| `worker.py` | a **ReAct tool-using agent** that executes one node | brain → tool → observe → loop → final; **real work, fenced** |
| `graph.py` | the **awake-loop** StateGraph | parallel dispatch · adversarial verify (N judges, majority) · reflect→adjacent · observability · HITL · durable |
| `manager.py` | **launch / status / resume / approve** | the supervisory surface — "call + manage it as META" |

## The loop
```
START → plan → dispatch(‖ workers, retry) → judge(N-vote) → reflect(+adjacent) → route ─(open&budget)→ dispatch
                       │ irreversible? → park to awaiting_approval (HITL)                └─(solved|spent)→ END
```
Verified 2026-06-05 (MockBrain + real tools): end-to-end loop; **parallel** dispatch; **adversarial** verdicts
(trace shows per-node votes); **dynamic adjacent-problem generation**; **observability** (JSONL trace + `status`);
**HITL** (a "DEPLOY REAL capital" node was parked, not run); **durable cross-process resume** (budget 2 → resume 6
→ solved, 5 lessons carried).

## Run + manage it (the META interface)
```
pip install langgraph langgraph-checkpoint-sqlite anthropic     # one-time (in requirements.txt)
# Shell-agnostic launcher (PowerShell / cmd / bash -- no PYTHONPATH needed):
python scripts/autonomy/run_metaop.py launch  --backend cli --objective "..." --budget 8 --parallel 2 --durable --thread t1
python scripts/autonomy/run_metaop.py status  --thread t1            # watch the trace + checkpoint
python scripts/autonomy/run_metaop.py resume  --thread t1 --budget 16   # continue (survives processes)
python scripts/autonomy/run_metaop.py approve --thread t1 --node <id>   # release a parked irreversible action
```
(Equivalent module form, if you prefer: PowerShell `$env:PYTHONPATH="scripts/autonomy"; python -m metaop.manager ...`;
bash `PYTHONPATH=scripts/autonomy python -m metaop.manager ...`. The launcher avoids the shell-specific env-var syntax.)

## The ONE activation (genuine thinking)
**ZERO-SETUP real brain (default).** `find_claude()` auto-detects the Claude Code VS Code extension's bundled
`native-binary/claude.exe`, so `--backend cli` thinks with **real Claude using your existing subscription auth —
no npm, no API key, no extra billing.** VERIFIED LIVE 2026-06-05 (`--backend cli`): real `plan` produced a genuine
5-node n±k frontier; a `claude -p` worker ran `git rev-parse` AND cross-checked it against `git log` (RWYB);
`judge` returned a real verdict; `reflect` dynamically generated 2 adjacent problems; durable + traced.
Backends, in priority: `--backend api` (ANTHROPIC_API_KEY, fastest/metered) → `--backend cli` (bundled CLI,
free, default) → `MockBrain` (no Claude at all; proves machinery). Nothing else changes between them — only what
*thinks* per node.

## Operational hardening (v1, 2026-06-05)
- **Thread LEASE (race-proof):** a thread can't be double-launched. `launch` acquires `runs/autonomy/locks/<thread>.lock`
  via **atomic `O_EXCL`** create; a second/simultaneous launch is REFUSED (exit 2); dead-owner locks self-reclaim;
  `launch` also reaps stale locks on start. (Closes W6 — provenance: a double-launch crashed two runs on the shared
  SQLite checkpoint, burning quota.)
- **Clean `stop` + auto-cleanup:** `stop --thread X` kills the run's whole **process tree** (the python + its
  `claude -p` workers, `taskkill /T`), releases the lease, reaps stale locks, and **preserves the durable
  checkpoint** (resume later). No more PID-hunting.
- **Project-wide compounding learnings:** `reflect` writes every lesson to `runs/autonomy/learnings.jsonl`; `plan`
  reads the recent ones into its prompt — so lessons persist and compound **across different objectives/threads**,
  not just within one run. Inspect with `... run_metaop.py learnings`.
- **CLI-brain robustness:** per-call timeout + retry; a slow/hung `claude -p` degrades to a handled `_error`
  (judge→inconclusive, dispatch→retry) instead of stalling or silently nulling the loop.
- **Hardened node prompts:** project-aware (long-only / compound-return / setup-over-a-move), anti-false-victory
  judging, RWYB workers, mandatory −k falsifier + +k generalization in planning.

## Manager commands
```
python scripts/autonomy/run_metaop.py launch    --backend cli --objective "..." --budget N --parallel K --durable --thread T
python scripts/autonomy/run_metaop.py status    --thread T          # live trace + checkpoint
python scripts/autonomy/run_metaop.py stop       --thread T          # clean tree-kill + cleanup (checkpoint kept)
python scripts/autonomy/run_metaop.py resume     --thread T --budget M   # continue from the durable checkpoint
python scripts/autonomy/run_metaop.py approve    --thread T --node ID     # release a parked irreversible action (HITL)
python scripts/autonomy/run_metaop.py learnings                       # the compounding project-wide memory
```

## Two execution variants (`--mode`)
Same loop, two ways to execute the work:
- **`--mode plain`** (default): generic `claude -p` workers.
- **`--mode expert`**: each node is routed to a `.claude/agents/*` specialist (`plan` assigns an `expert` per node;
  fallback `kind→expert`: build→researcher, verify→auditor, diverge→oracle). The worker/judge **adopts that
  expert's persona** (its system prompt), so the autonomous loop is as specialized as the interactive expert bench.
  Verify-nodes are judged through `expert-auditor`.

**Learnings per variant — separate or pooled** (`--learnings-channel`, default = the mode name):
- Default → `plain` and `expert` keep **separate** improvement loops (lanes `learnings/plain.jsonl`, `learnings/expert.jsonl`).
- Set both to one name (e.g. `--learnings-channel meta`) → **one pooled meta-lessons store** across variants.
```
python scripts/autonomy/run_metaop.py launch --backend cli --mode expert --objective "..." --durable --thread X
python scripts/autonomy/run_metaop.py launch --backend cli --mode plain  --learnings-channel meta ...   # pool
```

## Tool calling
- **CLI workers (`--backend cli`, what you use):** each `claude -p` worker is a **full Claude Code agent with the
  entire native toolset** (Read/Write/Edit/Bash/Grep/…), gated by the project's `permission_gate` hook + the
  worker's safety prompt (no commit/deploy/irreversible). Real, complete tool calling.
- **API workers (`--backend api`):** the raw LLM has no native tools, so `tools.py` is its hands — a fenced
  executor (`run_shell`/`run_python`/`read_file`/`write_file`/`list_dir`) screened by the same deny-fence.
- **Expert mode + tools:** persona-injection gives the expert's *knowledge + stance* while the worker keeps the
  full toolset; enforcing an expert's *own* `tools:` restriction (e.g. recon = read-only) would require spawning
  the real registered subagent — a documented future refinement.

## How it fits the project
- **Supersedes** (for UNSUPERVISED runs): the `meta_graph.py` prototype, the prose `OVERSEER.md`, the
  `autonomy_loop.py` Stop-hook re-invocation trick, and the crude `autonomy_driver.py`. Those *simulated* a loop
  inside a linear chat; METAOP **is** the loop, as a tool-using, self-verifying, durable program.
- **Two layers of "META":** METAOP (the program) is the persistent META that never sleeps. The Claude Code chat
  (me) is the **supervisory layer**: launch it, watch the trace, `approve` gated actions, `resume` it. For fully
  unattended runs, schedule `launch`/`resume` (cron) and drain `awaiting_approval` when present.
- **Drives the real project:** workers use `tools.py` to run the actual analyses (e.g. `scripts/research/*.py`,
  `src/strat`), read/write artifacts, and commit — so with a real brain it can run the research/strategy phases
  end-to-end, gated (deploy-real-capital escalates via HITL).

## Honest limits / next
- MockBrain proves the **machinery**, not the **intelligence** — real judgment needs the CLI/key (above).
- Node prompts are now project-aware + hardened (v1); they'll keep improving as real runs accumulate.
- The CLI brain is intrinsically **heavy** (each node = a full `claude -p` session) — that's the cost of using
  your subscription with no API key; `--backend api` is the lighter/faster path if you add a key.
- Parallelism is intra-node thread-pool fan-out; for heavy parallel sub-graphs, a dispatch node can use LangGraph
  `Send`/sub-graphs. **Single-machine** (SQLite), now lease-guarded against concurrent thread clashes; the durable
  checkpoint is the basis for distributing later.
