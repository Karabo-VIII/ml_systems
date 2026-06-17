# games_engine -- project guide

Three engines that ACTUALLY PLAY their games (real programs, not an LLM): Chess + Connect-4
(AlphaZero), Atari (DQN / MuZero on MinAtar). A shared AlphaZero/MuZero core lives in `az/`.
Self-contained: depends only on torch / python-chess / numpy / minatar (see requirements.txt).

## Run
- `python run_engines.py`                          -- all three engines, rendered move-by-move.
- `python play_chess.py` / `play_connect4.py` / `play_atari.py`  -- one engine at a time.
- Guides: docs/RUN_ENGINES.md (commands + what you see), docs/MOVE_TO_OWN_REPO.md (repo setup).

## Layout (binding -- the import root is the repo root; all packages import as top-level names)
- Repo root holds the COMMAND SURFACE only: `run_engines.py`, `play_chess.py`, `play_connect4.py`,
  `play_atari.py` (the turnkey demos) + `run_tests.py`, `run_invariants_check.py` (the gates).
- `az/`            -- the shared neural core (AlphaZero + MuZero + DQN) + `checkpoints/`. Internal
                      modules import each other relatively (`from .net import ...`).
- `chess_engine/` -- the classical chess engine: `engine.py` (negamax/alpha-beta), `uci.py`, `perft.py`,
                      `play_human.py`. Imported as `from chess_engine.engine import Engine`.
- `selfplay/`     -- AlphaZero self-play TRAINING ops: `play.py` (learn/watch loop), `run_engine.py`
                      (recipe launcher), `watchdog.py` (unattended-run guard).
- `tests/`        -- the engine test suite (`_test_*.py`); run via `python run_tests.py`.
- `docs/`         -- the guides (RUN_ENGINES, SELF_PLAY_GUIDE, MOVE_TO_OWN_REPO, ...).
- NOTE: this project was lifted out of a monorepo; the old `projects.chess_zero.*` import prefix and
  `projects/chess_zero/` path prefix are GONE. Use top-level package names (`az.*`, `chess_engine.*`,
  `selfplay.*`) and run `-m` modules from the repo root (e.g. `python -m az.train_robust`).

## Test gates
- `python run_tests.py`             -- the engine test suite (auto-discovers tests/_test_*.py).
- `python run_invariants_check.py`  -- catastrophic-correctness gate (move-gen, value-sign, no-hang).

## Conventions (binding)
- NO emoji / non-ASCII in print statements (Windows cp1252 crashes).
- RWYB (run-what-you-build): verify every change by actually RUNNING it before declaring done.
- Honest ceilings: report real measured strength. Chess is compute-bound (weak-but-real, NOT a master);
  Connect-4 is competent; the Atari DQN beats random on 3 real MinAtar games. Do not overclaim.
- Trained weights (the "brains") ARE COMMITTED: az/checkpoints/*.pt + az/robust_from_bootstrap/champion.pt are
  tracked in git (other/transient *.pt stay gitignored). Re-commit after each retrain to override the repo copy.
  The demo also runs WITHOUT them (untrained-net fallback, labelled). Regenerate via az/train_*.py.

## Specialist agents (.claude/agents/)
expert-architect (architecture), expert-trainer (training loops), expert-auditor (red-team review),
expert-validator (claim/evidence checks), expert-researcher (literature), expert-oracle (first-principles),
recon (fast read-only scan). Dispatch them for the matching work.
