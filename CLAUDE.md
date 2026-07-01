# ml_systems — parent workspace (ONE git repo, shared Claude config)

This directory is a **parent workspace** holding four self-contained sub-projects under one git repo and
one shared `.claude/`. From here you can focus on any one sub-project, or ship one off on its own.

> **🟢 OPERATING MODEL (2026-06-20) — READ FIRST for crypto work.** The user's binding intent + live state live in
> **[`crypto/CHARTER.md`](crypto/CHARTER.md)** (what the user wants + the permanent constraints) and
> **[`crypto/CURRENT.md`](crypto/CURRENT.md)** (where we are + the surface to cover next). Both are auto-injected
> every turn by the prompt router. A give-up / scope-collapse **gate** in
> [`.claude/hooks/autonomy_loop.py`](.claude/hooks/autonomy_loop.py) mechanically blocks a close that collapses the
> breadth surface (e.g. u100 → BTC) or quits at a verdict without attempting the full surface. Historical knowledge
> (dead-list, lessons, survivors, archive, git bundle) is mapped in [`crypto/KNOWLEDGE_MAP.md`](crypto/KNOWLEDGE_MAP.md).
> This is the 4-pillar operating model that fixes "instances can't build constructively on prior work + my intent."

> **🟢 WALL-CLOCK GROUNDING (binding, ALL instances + all three sub-projects).** You have NO internal clock — current
> time, elapsed, "X hours in / left", "last modified", "N days ago" are unknowable by feel and MUST be measured with a
> fresh `date` (files via `stat` / `git log`). Tag every time claim **VERIFIED / REPORTED / INFERRED**; for elapsed show
> the subtraction (fresh `date` − VERIFIED start). Re-ground after any gap (resume, context summary, long/background
> tool call, new turn). Timed / autonomous / `/loop` / `/schedule` runs record a VERIFIED start once, then `date − start`
> on every progress claim — never narrate the passage of time, read it. Full spec + the 2026-06-03 fabrication incident:
> [`.claude/skills/_common/WALL_CLOCK.md`](.claude/skills/_common/WALL_CLOCK.md).

| Sub-project | Path | What it is | Its own doc |
|---|---|---|---|
| **Crypto** | [`crypto/`](crypto/) | The V4 crypto system (pipeline, world-models, strat, mining, oracle, wealth_bot, audit, framework). | **[`crypto/CLAUDE.md`](crypto/CLAUDE.md) — READ THIS for any crypto work; all the binding crypto invariants live there.** |
| **Games** | [`games/`](games/) | Self-contained games engine — AlphaZero + MuZero core (`az/`), classical engines (`chess_engine/`), chess/connect-4/atari. Zero crypto imports. | [`games/CLAUDE.md`](games/CLAUDE.md) |
| **Harness** | [`harness/`](harness/) | Project-agnostic LangGraph metaop engine (plan/dispatch/judge/reflect/route). Installed globally (`metaop`/`harness` on PATH). Zero crypto imports. | [`harness/README.md`](harness/README.md) |
| **Forex** | [`forex/`](forex/) | Production-grade FX + metals data pipeline (Dukascopy ticks + HistData M1 + OANDA v20 → bid/ask time bars → FX-aware quality gates + metrics). Instrument universe u7/u28/u52. Zero crypto imports. | **[`forex/CLAUDE.md`](forex/CLAUDE.md) — READ for any forex work.** |

**Shared at this root** (not owned by any one sub-project):
- `.claude/` — agents, skills, `_common`, hooks, settings (governs all three sub-projects).
- `models/` — GGUF LLM brains for the local-model harness (gitignored). Crypto's ML model weights live under `crypto/models/`.
- `.venv/` — shared virtualenv (gitignored).

**Layout rule (so paths self-correct):** each sub-project's `src/ config/ data/ runs/ scripts/ docs/` descend together
under its own root, so in-code `Path(__file__).parents[N]` root-resolution still lands on the sub-project root. The
only cross-sub-project code dependency is one-way: `crypto/scripts/autonomy → harness.metaop`.

> For crypto invariants, anti-fragile training rules, CDAP, the market-decomposition harness, and everything else
> that used to be at the repo root: see **[`crypto/CLAUDE.md`](crypto/CLAUDE.md)**.

Provenance: 3-way repo split (`repo-split-3way`), 2026-06-17. Migration runbook: [`crypto/docs/SPLIT_RUNBOOK.md`](crypto/docs/SPLIT_RUNBOOK.md).
