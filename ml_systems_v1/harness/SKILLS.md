# Skills + localised context for a generic / local model

> How do "skills, MD files, localised context" (à la Claude Code) actually work, and how do you give them to YOUR
> model — including a small local one (e.g. `qwen2.5-coder:7b`)? This is the harness's answer + the built mechanism
> (`metaop/skills.py`). Sourced from a 2026-06 SOTA survey (Anthropic Agent Skills, aider/Cursor/Cline/Continue,
> CrewAI/AutoGen, LangChain/LlamaIndex, OpenAI Assistants) + small-model context research, verified against the code.

## 1. The one idea (there is no magic)

Everything reduces to **structured text injected into the model's prompt, plus a relevance SELECTOR that respects a
finite context budget.** A "skill", a "rule file", a "memory", a "persona" — all are just text that some component
decides to place in the window *this turn*. The engineering is entirely: *which* subset of your knowledge/capability
goes in front of the model, and *how you choose it*, because the window is a shared finite resource.

Three injection primitives, and the SOTA selector in front of each:

| Primitive | What it is | Selector |
|---|---|---|
| **Always-on text** (rules / `CLAUDE.md`) | prepended every turn | none (or glob-scoped) |
| **Retrieved text** (skills, docs / RAG) | injected only when relevant | **manifest + progressive disclosure**, or embedding-RAG |
| **Tool schemas** | name+description+params; model emits a call, you run it | the model picks from the advertised set |

## 2. Progressive disclosure — the core pattern

A skill is a markdown file with frontmatter (`name`, `description`) + a body. You **advertise cheaply, load lazily**:

| Tier | Loaded | Cost | Content |
|---|---|---|---|
| 0 — manifest | always | ~tens of tok/skill | `name` + `description` (the routing key) |
| 1 — body | when the skill is **selected** | <5k tok | the `SKILL.md` body |
| 2 — resources/scripts | on demand | ~unbounded | bundled refs read; **scripts executed → only stdout costs tokens, never the source** |

So you can have 50 skills but pay for the 1–3 relevant ones. The `description` is the whole ballgame — write it
**third person, stating what it does AND when to use it** ("Extracts text from PDFs. Use when a task mentions PDFs"),
because it's the selector signal. (`SKILL.md` is now a cross-vendor open standard — Codex CLI, Gemini CLI, Cursor.)

`CLAUDE.md` is the *always-on* primitive: project instructions concatenated into the prompt every turn (in Claude
Code, as a user message, to preserve prompt-cache — a caching detail, not a requirement). Keep it small; it's a tax
on every call. Memory files are progressive disclosure applied to memory (a small always-loaded index + topic files
pulled on demand).

## 3. The small-model translation (the part that's different for a 7B)

**Claude's selector is the model itself** — it reads the whole manifest and self-routes. **A 7B cannot do this
reliably** (a large manifest is noise it can't attend to; it burns its limited reasoning on tool-disambiguation
instead of execution). So for a local model you **invert it: make the selector mechanical/embedding-based, and hand
the model only the top-k pre-selected skills.** Research is consistent: retrieving **K=3** relevant tools and
injecting only those *beats* dumping all of them — both on tokens (~99% reduction) and on task success.

SOTA stack for a 7B (~4–8k *usable* context, even if "128k" is advertised — long-context degrades hard):
- **Mechanical selector**, not LLM-self-select: token-overlap/BM25 (zero-dep) → upgrade to dense+rerank
  (`nomic-embed-text` on **CPU** so the 8GB VRAM stays with the generator; hybrid dense+BM25 catches exact
  identifiers; rerank to **top-3**).
- **Contextual Retrieval** (prepend a one-line "this is from skill X" blurb before embedding) — biggest single
  retrieval-quality win.
- **Hard per-section token budget** (system / skills / context / task), compress-or-truncate before assembly.
- **Primacy + recency**: role+invariants FIRST, output-contract LAST (beats "lost in the middle").
- **Strict output formats** (GBNF/grammar-constrained decoding) — specifically a small-model reliability lever
  (Qwen2.5-Coder-7B: 0% → 75% schema-valid under grammar constraints).

## 4. What the harness ships — `metaop/skills.py`

The mechanism above, portable and dependency-free:

```python
from metaop.skills import select, digest, manifest, context_pack, skills_recaller, context_framer

manifest(skills_dir)                 # Tier-0: "- name: description" per skill (the cheap routing table)
select(objective, skills_dir, k=3)   # MECHANICAL top-k by relevance (token-overlap; pass embedder= for dense RAG)
digest(objective, skills_dir, k=3)   # manifest + the SELECTED skill bodies (progressive disclosure, budgeted)
context_pack(["CONTEXT.md", ...])    # the CLAUDE.md-equivalent: project files concatenated under a hard char budget
```

It reads either layout: `<skills_dir>/<name>/SKILL.md` (Claude-Code style) or `<skills_dir>/<name>.md` (flat).
Selection is mechanical by default (right for a 7B); pass `embedder=` for semantic retrieval.

## 5. Wiring it (zero graph changes)

`skills_recaller` / `context_framer` are drop-in for the engine's existing host hooks, so the selected skills land in
`payload["recall"]` and the project context in `payload["framing"]` — both of which the planner already consumes:

```python
from metaop.graph import build
from metaop.skills import skills_recaller, context_framer
app = build(brain, recaller=skills_recaller("skills/", k=3),
                   framer=context_framer(["CONTEXT.md", "DEAD_LIST.md"]))
```

…or straight from the CLI (the manager full-solver):

```bash
python -m metaop.manager launch --objective "detect the market regime and ride the trend" \
    --skills-dir ./skills --context ./CONTEXT.md --backend ollama --durable
```

The planner is then handed the top-3 relevant `SKILL.md` bodies + the project context every cycle — Claude-Code-style
skills + localised context, on **any** model, with mechanical selection sized for a small one.

## 6. Tool-skills vs knowledge-skills

`metaop/skills.py` is **knowledge** (text that shapes reasoning — the `SKILL.md`/`CLAUDE.md` translation you asked
for). For **executable** skills (a verified script the loop can re-run), the harness has a separate register pattern
(the host's `harvester` hook → a tool/asset library). Keep the split: *put flexible guidance in text (skills.py); put
deterministic/large work behind a tool or a script.*

## 7. Build order (smallest first)

1. **Done:** `metaop/skills.py` (manifest + mechanical selector + progressive-disclosure digest + context pack +
   the two host-hook closures), wired into `metaop/manager.py` via `--skills-dir` / `--context`. RWYB-verified.
2. **Next (optional):** the embedding selector — pass an `embedder=` backed by a local `nomic-embed-text` (reuse
   `metaop/memory.py`) so selection is semantic, not just lexical. The seam is already there (`select(embedder=...)`).
3. **Then (optional):** grammar-constrained decoding for the local backend (GBNF) so the 7B's JSON is always valid.
