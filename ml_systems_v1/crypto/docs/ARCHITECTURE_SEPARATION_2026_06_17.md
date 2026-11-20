# Architecture separation — implemented 2026-06-17 (ONE git repo, global harness)

User intent (2026-06-17): separate the project globally — the **harness** (global, usable by every project + by
Claude), the **crypto project**, the **games engine**, and the **generic engine** — sharing the global Claude
agents/skills/configs, with **no reference/import breaks**, and **everything kept in one git repository**.

Recon finding that shaped the approach: the codebase was **already ~90% decoupled** (the harness is self-contained,
crypto `src/` is a clean leaf, `projects/chess_zero` is self-contained). So the "massive undertaking" reduced to a
small, low-risk set of changes — NOT a physical relocation of `src/` (which would only rewrite `parents[N]` path-math
for cosmetic gain). Minimal-churn was chosen deliberately.

## The one-repo component map (logical separation, physical co-location)
| Component | Location (in this ONE repo) | Role | Coupling |
|---|---|---|---|
| **Harness** | `harness/` | project-agnostic LangGraph metaop engine (plan/dispatch/judge/reflect/route) | zero crypto imports; **installed globally** (below) |
| **Crypto** | `src/` (pipeline, wm, strat, mining, oracle, wealth_bot, audit) + `config/` + `data/` | the crypto system (a clean leaf) | depends only on shared infra inside `src/`; one *optional, graceful* read of `scripts/autonomy/cross_pollination_bus` |
| **Generic engine** | `src/framework/` | market-agnostic solutioning pipeline (research→…→deploy; Layer A/B/C router) | self-contained; references games via a **graceful dynamic import** (optional) |
| **Games engine** | `projects/chess_zero/` | chess/connect-4/atari (AlphaZero+MuZero core in `az/`, classical in `chess_engine/`) | **fully self-contained** — zero crypto imports |
| **Crypto autonomy wiring** | `scripts/autonomy/` | the crypto-specific consumer of the harness (`metaop` shim + domain injection + the 3 loops) | imports `harness.metaop`; stays beside crypto |
| **Shared Claude layer** | `.claude/` (agents, skills, `_common`, hooks, settings) | shared by ALL in-repo components automatically (one repo → one `.claude`) | — |
| **Models** | `models/` (gitignored) | shared GGUFs for the local brain | — |

## Global harness (the headline) — IMPLEMENTED
The harness is installed **editable into the system Python user-site** (NOT the project `.venv`):
```
C:\Users\karab\AppData\Local\Programs\Python\Python311\python.exe -m pip install --user -e harness
# console scripts -> %APPDATA%\Python\Python311\Scripts  (added to USER PATH)
```
Result: `harness` / `metaop` CLI on PATH for **every** project + shell; `import metaop` works system-wide and
resolves to this repo's `harness/metaop` (editable → tracks the repo). Subcommands:
`launch | resume | status | approve | stop | learnings | evolve | improve`.

**Collision-safety (the key design point):** the repo `.venv` does NOT see user-site packages, so **inside this repo
`import metaop` still resolves to the crypto shim `scripts/autonomy/metaop`** (verified) — the global install adds the
harness as a global tool **without breaking any in-repo import**. Reproduce/verify:
```
.venv\Scripts\python -c "import sys;sys.path.insert(0,r'scripts/autonomy');import metaop;print(metaop.__file__)"
# -> scripts\autonomy\metaop\__init__.py   (the crypto shim — correct, no collision)
```
Uninstall (reverse): `<system-python> -m pip uninstall metaop-harness` + remove the Scripts dir from USER PATH.

## Dedup — RESOLVED
`projects/chess_zero` is the **canonical** games engine. The external sibling repo
`c:/Users/karab/Documents/coding/games_engine` (github.com/Karabo-VIII/games-engine) was **harmonized INTO**
`projects/chess_zero` on 2026-06-13 (commit `7e57c9a`), which then added the Windows no-window fixes + the test
suite — so the in-repo copy is strictly ahead. **Action for the user:** the external `games_engine` repo/dir is now
superseded and can be archived/deleted (it is outside this git repo, so left untouched here).

## Why "one git repo" (not separate repos / submodules) is right here
The components share *ideas* (via the in-repo `.claude` skills + the `cross_pollination_bus` JSONL), not pinned code
*versions*. One repo gives cross-component inference with zero broken imports and no submodule friction. The harness
is the only thing that needs to reach OTHER projects — solved by the global install, not a repo split.

## No-broken-refs verification (RWYB)
- in-repo `metaop` → crypto shim (no collision) ✓
- `harness.metaop` importable in-repo ✓; `metaop` importable globally (system) ✓
- `projects/chess_zero` self-contained (zero crypto imports) ✓
- games engine invariants gate + the metaop `agent_eval` selftest re-run green after the change.
