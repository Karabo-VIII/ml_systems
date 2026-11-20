# harness -- a local, model-portable, self-improving engine-builder

A small, **standalone, project-agnostic** autonomous loop that builds verifiable artifacts in *any* domain. You
own it. It outlives any one model and any one project: the model is a pluggable `Brain` behind a tiny interface,
and nothing in the package imports a host repository.

It is a [LangGraph](https://github.com/langchain-ai/langgraph) state graph implementing the classic agentic loop:

```
plan -> dispatch -> judge -> reflect -> route ──(loop until solved / budget spent / frontier empty)
```

- **plan** -- the Brain seeds an EV-ranked *frontier* of nodes (a `build`, a `verify`/falsifier, a `diverge`/generalize).
- **dispatch** -- runs up to `parallel` nodes concurrently; nodes flagged irreversible are *parked* for human approval (HITL).
- **judge** -- a **mechanical verifier first** (see below), else an adversarial N-judge LLM vote that refuses false victory.
- **reflect** -- distils one transferable *lesson* (persisted to a learnings channel) and may generate *adjacent* problems.
- **route** -- continue, or stop on solved / budget / empty frontier.

Every node and decision is written to a JSONL **trace** you can watch live. State can be made **durable**
(SqliteSaver) so a run survives and resumes across processes.

## Why this exists

This is the autonomy harness extracted from a larger project so it is *yours* and *portable*. The core idea: an
honest, verifier-grounded build loop where **rejection is a gradient** -- when a build fails its mechanical check,
the concrete error is fed straight back into the next attempt. The intelligence (the model) is swappable; the loop,
the memory, and the verifier contract are the durable assets.

## Install

**Pip-installable (recommended).** From the `harness/` dir (or the extracted repo root) the `pyproject.toml`
makes it a package with a `harness` console script:

```bash
pip install .                 # core (langgraph only) + the `harness` / `metaop` console scripts
pip install ".[all]"          # + durable resume + the litellm gateway + anthropic + mem0 (a local-first stack)
pip install ".[durable]"      # just add the sqlite checkpointer for --durable cross-process resume
```

Or run in place with no install — just two dependencies (a Python 3.11+ env):

```bash
pip install langgraph langgraph-checkpoint-sqlite   # sqlite checkpointer only needed for --durable runs
```

Optional extras for the real backends (`pip install ".[litellm]"` / `.[sdk]` / `.[anthropic]` / `.[memory]`):

```bash
pip install litellm              # --backend ollama / the unified gateway (local model, no Claude, no key)
pip install claude-agent-sdk     # for --backend sdk (in-process Claude; uses your existing CLI auth)
pip install anthropic            # for --backend api (needs ANTHROPIC_API_KEY)
# or have `claude` on PATH       # for --backend cli
```

The default **MockBrain backend needs none of these** -- it drives the full machinery (including real tool work)
deterministically, so you can prove the loop end-to-end with zero credentials.

## Run

```bash
# smallest possible run -- no API key, no network -- proves the whole loop end-to-end
python harness/run.py --objective "trivial" --backend mock

# a build node verified by a MECHANICAL command (exit 0 == ground-truth pass, overrides any LLM vote)
python harness/run.py --objective "make hi.py print hi" --backend mock \
    --verify-cmd "python -c \"print('ok')\""

# real Claude (in-process SDK) building inside a chosen target directory
python harness/run.py --objective "build a CSV summariser CLI" --backend sdk --cwd /path/to/project
```

Key flags: `--backend {mock|sdk|api|cli|auto}`, `--verify-cmd "<shell>"`, `--verify-retries N`, `--cwd <dir>`
(where the worker/verifier run), `--workspace <dir>` (where the harness keeps its own traces/learnings/checkpoints),
`--domain "<text>"` (injected into the brain prompts to specialize task-flavor), `--budget`, `--parallel`,
`--judges`, `--taper`.

For long-lived / resumable runs use the manager:

```bash
python -m harness.metaop.manager launch --objective "..." --budget 8 --backend mock --durable --thread t1
python -m harness.metaop.manager status --thread t1
python -m harness.metaop.manager resume --thread t1 --budget 16
python -m harness.metaop.manager approve --thread t1 --node a3   # release a parked irreversible node
python -m harness.metaop.manager stop   --thread t1             # checkpoint preserved
```

## Out of the box: complete & self-evolving

A fresh install runs end-to-end **and** can augment itself from within its own loop — the only thing left for you is
to ask it to. Everything below ships in this package.

**1. Solve (with skills + project context).** A starter `skills/` library + a `CONTEXT.md` template ship here; the
harness selects the relevant skills per objective and injects them at plan time:
```bash
python -m metaop.manager launch --objective "build X" --skills-dir ./skills --context ./CONTEXT.md \
    --backend cli --durable
```

**2. Self-augment — grow the skill library from what it verifies.** Add `--harvest`: every build that passes its
mechanical verifier is authored into `skills/` as a new `SKILL.md`, selectable on the next run (Voyager monotonicity).
```bash
python -m metaop.manager launch --objective "..." --skills-dir ./skills --harvest --backend cli
```

**3. Self-evolve — improve HOW it plans.** Optimize the planner prompt against the honest solve-rate fitness and
install the winner as the champion (the next run auto-applies it; an elitism floor never installs a worse one):
```bash
python -m metaop.manager evolve --backend cli --generations 3 --pop-size 4   # one evolution run
```

**3b. Continuous self-evolution (hands-off, no asking).** Start the daemon once and it keeps evolving the planner
round after round on its own -- each round seeds from the best-so-far champion, so the planner gets monotonically
better with no further input. Bounded by `--rounds`/`--max-minutes`; resumable (the champion persists on disk, so
stop/restart continues where it left off):
```bash
python -m metaop.manager improve --backend cli --rounds 20         # 20 hands-off rounds
python -m metaop.manager improve --backend cli --max-minutes 60    # run continuously for an hour
```
This is the standalone equivalent of an always-on evolution loop, shipped IN the harness. (The planner signal needs
no external objectives -- it scores against the built-in benchmark; skill-growth from real work is lever 2.)

**Just ask it.** Because the **`self-evolve`** skill ships in `skills/`, the loop knows the playbook above: when you
give it an objective like *"augment yourself"* / *"add a CSV-summary skill"* / *"get better at planning"*, the
selector surfaces `self-evolve` (and `author-skill`) and it runs the right lever itself. The machinery — skill
selector + harvester, planner evolution (`evolve.py`) + champion install (`champion.py`), compounding memory
(`learnings.py`/`memory.py`) — is all present and wired; you supply the objective.

## The Brain interface (wiring a new model)

A `Brain` is the model-portability seam. Implement three methods and register it in `make_brain`:

```python
class Brain:
    def decide(self, role, payload, persona="") -> dict:   # role in {"plan","judge","reflect"} -> ONE JSON object
        ...
    def act(self, task, tools_schema, history) -> dict:    # ONE ReAct step -> {"action":"tool",...} | {"action":"final","result":...}
        ...
    def work(self, task, persona="") -> dict:              # do a node's task end-to-end -> {"ok": bool, "result": str}
        ...
```

- `decide` is one-shot structured reasoning for the graph nodes. The expected JSON shapes per role are documented
  in `harness/metaop/brain.py` (the `_DECIDE_SYS_T` contract).
- `act` is a single step of a tool-using ReAct loop (used by `Worker` + the API backend). Return either a tool call
  (the harness runs it via `Tools` and feeds back the observation) or a `final` result.
- `work` does a whole node end-to-end. SDK/CLI backends (which are themselves full agents) implement `work` directly
  and stub `act`; raw-LLM backends (API/Mock) implement `act` and get `work` via the ReAct loop.

The system prompts carry a `{domain}` slot, so the harness is domain-neutral by default; pass `domain=` to
`make_brain` (or `--domain`) to specialize without editing any code.

**Local models already ship.** `OllamaBrain` (pure-stdlib, hits a local Ollama server) and `LiteLLMBrain` (the unified
gateway: any provider, auto-fallback primary -> local ollama -> MockBrain) are built in -- `--backend ollama` runs the
whole loop on a local open-source model with **no Claude and no API key** (verified end-to-end on `qwen2.5-coder:7b`).
`CascadeBrain` adds a cheap->strong escalation router. To add a brand-new backend, subclass `Brain`, implement
`decide`/`act`/`work` (the tolerant `_extract_json` in `brain.py` parses loose model JSON), and add a branch in
`make_brain`. `MockBrain` (pure/no-creds) and `AgentSdkBrain` (a real in-process agent) are the two reference copies.

## The verify_cmd (mechanical verifier) contract

Any node may carry a `verify_cmd` (a shell command string). When present, `judge` **runs it mechanically from the
build cwd** and treats it as ground truth:

- **exit 0 -> PASS.** This *overrides* the LLM judge panel -- no vote can fake a green build.
- **exit != 0 -> REFUTED.** The last ~1500 chars of stdout+stderr are captured on the node and, if a
  `verify_retries` budget remains, the node is re-opened and the **concrete error is appended to the next dispatch
  prompt** so the worker fixes the real failure (rejection-as-gradient).

This is the heart of honest autonomy: a build is only "done" when a real command says so. `verify_cmd` runs from
`--cwd` (your target project), times out at 120s, and a timeout/launch-failure counts as a refutation.

## Safety fence

`Tools` always denies irreversible/destructive shell ops (`rm -rf /`, force-push, `mkfs`, `sudo`, fork bombs, ...)
and refuses to write control-surface files (`.git/`, `.env`, ssh keys), regardless of caller config. The SDK
backend additionally denies commit/push/deploy via a PreToolUse hook (stage your work; let a human review + commit).
You can tighten further with `extra_cmd_deny` / `extra_file_deny` when constructing `Tools`.

## Files

```
harness/
  run.py                 # clean CLI entrypoint: one objective through the loop (+ optional --verify-cmd)
  README.md              # this file
  metaop/
    __init__.py          # package exports
    config.py            # the ONLY filesystem coupling: a configurable WORKSPACE (traces/learnings/checkpoints)
    graph.py             # the LangGraph awake loop (plan/dispatch/judge/reflect/route) + mechanical verifier
    brain.py             # the pluggable Brain interface + MockBrain / AgentSdkBrain / AnthropicBrain / CliBrain
    tools.py             # the worker's hands (shell/python/read/write/list) behind a hard safety fence
    worker.py            # the reference ReAct worker loop (used by act()-based brains)
    experts.py           # optional specialist personas for expert mode (point at your own persona dir)
    learnings.py         # append-only, per-channel lessons memory (experience compounds across runs)
    skills.py            # SKILL.md skills + localised context for ANY model (manifest + mechanical selector +
                         #   progressive disclosure + context pack); drop-in recaller/framer hooks. See SKILLS.md
    manager.py           # launch/status/resume/approve/stop a durable run (+ --skills-dir / --context)
  SKILLS.md              # how generic/local models work with skills + localised context, and how to use skills.py
```

**Give your model skills + localised context** (the Claude-Code translation, on any model incl. a local 7B):
`python -m metaop.manager launch --objective "..." --skills-dir ./skills --context ./CONTEXT.md` injects the top-k
relevant `SKILL.md` bodies + your project context at plan time, with a **mechanical selector** sized for a small
model (a frontier model self-routes a big manifest; a 7B can't). Full explainer + SOTA + usage in **[SKILLS.md](SKILLS.md)**.

Run artifacts live under the **workspace** (default `./.harness_runs/`), never inside the package.

## Standalone status & extraction

**This package (`harness/`) is already standalone and domain-agnostic.** Verified 2026-06-12: nothing under
`harness/` imports a host repository (the engine uses only relative imports + `langgraph`), the full
planner-first solver runs generically via `python -m metaop.manager launch ...` (its `build` comes from
`metaop.graph`, not any project shim), and it solves with `--backend mock` (zero creds) and `--backend ollama`
(a local model, no Claude). You can call it like a CLI **today**, three ways:

```bash
harness launch --objective "build X" --backend cli --durable --domain "your domain"   # after `pip install .`
python -m metaop.manager launch --objective "build X" --backend ollama --durable      # no install, extracted
python -m harness.metaop.manager launch --objective "build X" --backend ollama         # no install, in THIS repo
```

…and programmatically: `from metaop.manager import launch` (or copy the ~20-line `run.py` pattern).

**What is the standalone core vs. what stays with the host project.** The harness is the *engine*; a host project
*parameterizes* it without editing it — via `make_brain(domain=...)` and the optional `build(framer=, recaller=,
recorder=, harvester=)` hooks. In THIS repo the crypto project's glue lives OUTSIDE `harness/` and stays behind:

| Standalone (lives in `harness/`, extract this) | Host-project glue (stays here, do NOT extract) |
|---|---|
| `metaop/` engine + `run.py` + `pyproject.toml` + this README | `scripts/autonomy/metaop/` — the crypto **shim** (injects domain + framer/recaller/recorder/harvester) |
| pluggable `Brain` (mock/sdk/api/cli/ollama/litellm/cascade) | `scripts/autonomy/*.py` — launchers, watcher, skill_library, loop_health (the always-on driver) |
| mechanical verifier + safety fence + durable resume | `.claude/` hooks (Stop-hook loop, permission_gate) + `.claude/skills/` (the Claude-Code attended UX) |

The host-project layer is OPTIONAL: it is how *this* repo runs the harness always-on inside Claude Code. A
standalone user just calls `metaop.manager` directly; they can re-create their own thin shim + driver if they want
an always-on loop in their own project.

**Extract it to its own repo (one command).** `git subtree split` carves `harness/` out with its history intact,
making its contents the new repo root (so the package is the top-level `metaop`, `pyproject.toml` is at root, and
the imports above resolve):

```bash
git subtree split -P harness -b harness-standalone     # branch whose root == harness/ contents
# then, in a fresh empty repo:
#   git pull <this-repo-path> harness-standalone
#   pip install .            # -> `harness` console script + `metaop` package
#   harness launch --objective "..." --backend cli --durable
```

(The crypto project keeps its own `harness/` copy untouched; the split is a copy, not a move.)

