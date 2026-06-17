# ml_systems — parent workspace (ONE git repo, shared Claude config)

This directory is a **parent workspace** holding three self-contained sub-projects under one git repo and
one shared `.claude/`. From here you can focus on any one sub-project, or ship one off on its own.

| Sub-project | Path | What it is | Its own doc |
|---|---|---|---|
| **Crypto** | [`crypto/`](crypto/) | The V4 crypto system (pipeline, world-models, strat, mining, oracle, wealth_bot, audit, framework). | **[`crypto/CLAUDE.md`](crypto/CLAUDE.md) — READ THIS for any crypto work; all the binding crypto invariants live there.** |
| **Games** | [`games/`](games/) | Self-contained games engine — AlphaZero + MuZero core (`az/`), classical engines (`chess_engine/`), chess/connect-4/atari. Zero crypto imports. | [`games/CLAUDE.md`](games/CLAUDE.md) |
| **Harness** | [`harness/`](harness/) | Project-agnostic LangGraph metaop engine (plan/dispatch/judge/reflect/route). Installed globally (`metaop`/`harness` on PATH). Zero crypto imports. | [`harness/README.md`](harness/README.md) |

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
