# Moving games_engine into its own repository

> NOTE: this engine has ALREADY been lifted out of the monorepo into its own standalone directory
> (with its own git repo). This guide is kept for reference / re-hosting it elsewhere.

This project is **self-contained** — it imports nothing from the parent
crypto project (verified: zero `src.*` / crypto-pipeline imports). Its only dependencies are
`torch`, `python-chess`, `numpy`, `minatar` (see `requirements.txt`). So it can be lifted out cleanly.

## Option A — copy (simplest, recommended)
```
cp -r projects/chess_zero  /path/to/games_engine      # or copy the folder however you like
cd /path/to/games_engine
git init
python -m venv .venv
.venv\Scripts\activate            # Windows   (or: source .venv/bin/activate)
pip install -r requirements.txt   # install a torch build for your platform first if needed
python run_engines.py             # all three engines, or: python play_chess.py
```

## Option B — git subtree split (preserves the code's history)
```
git subtree split --prefix=projects/chess_zero -b games_engine-export
# then add the new repo as a remote and push that branch to it
```
Note: a subtree split only carries files that are **in git history**. The trained brains are now
**committed** (see below), so they ARE included by a split.

## The trained checkpoints (important — these make the engines play *well*)
The demo loads trained weights from:
- `az/checkpoints/connect4.pt`               (~1.5 MB)
- `az/checkpoints/atari.pt`                  (MuZero / CatchEnv, ~0.4 MB)
- `az/checkpoints/atari_minatar.pt`          (DQN / Breakout, ~1.6 MB)
- `az/checkpoints/atari_minatar_space_invaders.pt`, `..._asterix.pt`
- `az/robust_from_bootstrap/champion.pt`     (chess AlphaZero champion, ~42 MB)

These brains are **committed to the repo**: `.gitignore` ignores `*.pt` in general but un-ignores
`az/checkpoints/*.pt` and `az/robust_from_bootstrap/*.pt`, so a fresh `git clone` plays well out of the
box. Re-commit them after each retrain to override the repo copy:
`git add az/checkpoints az/robust_from_bootstrap/*.pt && git commit -m "update brains"`.
- The chess `champion.pt` is ~41 MB — fine for plain git (GitHub warns >50 MB, blocks >100 MB). If you
  re-commit it often and want to avoid history bloat, track it with **Git LFS** instead.
- **Without the checkpoints the demo still RUNS** — each engine falls back to an untrained net and
  says so on screen (`[checkpoint missing -- playing with an UNTRAINED net]`). To regenerate them,
  use the training drivers in `az/` (`train_connect4_gpu.py`, `train_minatar.py`, `train_robust.py`).

## What's in here
| Path | What |
|---|---|
| `run_engines.py` | play **all three** engines, rendered (the "click run" demo) |
| `play_chess.py` / `play_connect4.py` / `play_atari.py` | run **one** engine at a time |
| `docs/RUN_ENGINES.md` | the run guide (commands + what you see + honest strength) |
| `az/` | the neural engine core: `net.py`, `mcts.py`, `encoding.py`, the games (`connect4.py`, `muzero_rl.py`, `dqn_minatar.py`, `minatar_env.py`, `game_adapter.py`, `chess_adapter.py`), training drivers, `checkpoints/` |
| `chess_engine/` | the classical chess engine: `engine.py` (negamax/alpha-beta) + `uci.py` + `perft.py` + `play_human.py` |
| `selfplay/` | self-play TRAINING ops: `play.py` (learn/watch), `run_engine.py` (recipe launcher), `watchdog.py` |
| `tests/` | the engine test suite (`_test_*.py`) -- run via `python run_tests.py` |
| `run_tests.py`, `run_invariants_check.py` | the test gates (`python run_tests.py`) |
| `requirements.txt`, `pyproject.toml`, `.gitignore` | packaging |

Nothing here depends on the crypto project — `grep -rn "from src\|import src" .` returns nothing.
