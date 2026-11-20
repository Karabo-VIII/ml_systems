# Run the engines — click run, watch them play

One command plays all three engines, rendered move-by-move in the terminal:

```
python run_engines.py
```

That's it. No setup, no training wait — it loads the already-trained checkpoints and plays.

### Watch GENUINE strength — `--strong`

```
python run_engines.py --strong
```

The default run shows the AlphaZero **net** (honest, but the net is the weak component). `--strong`
instead showcases the genuinely strong engines, **measured**:
- **Chess:** the in-repo **classical engine (~1600 Elo)** competing against **Stockfish** (one of the
  world's strongest, Elo-capped to a fair level — the honest "compete against the best"). Needs a
  Stockfish binary (see [PLAY_HUMAN.md](../PLAY_HUMAN.md)); without one it plays strong self-play.
- **Connect-4:** the **net + perfect-endgame solver HYBRID** — **MEASURED +~134 Elo** over the bare
  net (it plays a provably-optimal mid/endgame and never misses a 1-move win/block).
- **Atari:** the trained **DQN** on real MinAtar Breakout (the champion scores ~191, 337x random).

## What you'll see

| Engine | What it does | Watch for |
|---|---|---|
| **CHESS** (AlphaZero) | the trained champion plays **itself** (self-play) | real, legal, varied chess move-by-move on an ASCII board |
| **CONNECT-4** (AlphaZero) | the trained net (PUCT search) vs a 1-ply win/block heuristic | it drops pieces to build/block 4-in-a-row and usually **wins** |
| **ATARI** (DQN) | the trained agent plays **real MinAtar Breakout** (a scaled Atari game) | the paddle `=` tracks the ball `o` and smashes the brick wall `#`, score climbing |

(The Atari slot defaults to real MinAtar **Breakout**; `--atari-game all` plays all three trained games
(Breakout, Space Invaders, Asterix). A model-based variant — MuZero planning over a *learned* model on
CatchEnv — is one flag away: `--atari-mode catch`.)

## Options

```
python run_engines.py --engine chess      # just one engine (chess|connect4|atari|all)
python run_engines.py --games 3           # more games each
python run_engines.py --delay 0.8         # slower, easier to watch
python run_engines.py --fast              # no delay (quick)
python run_engines.py --no-render         # scores only
python run_engines.py --device cuda       # use the GPU
python run_engines.py --atari-game all     # play all 3 real MinAtar games
python run_engines.py --atari-mode catch  # MuZero/CatchEnv instead of MinAtar
```

## Honest strength (what "running" means here)

These are the **SOTA algorithms** (AlphaZero + MuZero) genuinely running and playing — trained as far
as a single RTX 4060 affords. They are **not** superhuman; that is compute-bound (orders of magnitude
more self-play), and we don't fake it:

- **Connect-4** — genuinely competent: **W40/L0 vs random** and **W33/D2/L5 vs a sharp 1-ply
  win/block heuristic** (it started out *losing 0:24* — GPU training taught it real win/block tactics).
  Not perfect play (Connect-4 is a solved first-player win), but clearly skilled.
- **Atari** — a real Atari agent (DQN) that genuinely learned to play **three** MinAtar games (the
  standard scaled-Atari research benchmark, 10×10 pixel grids), each reload-verified:
  **Breakout ~191 vs random 0.57 (≈300×)**, **Space Invaders 44 vs 3.6 (≈12×)**, **Asterix 1.17 vs 0.40
  (≈2.9×)**. The default shows Breakout; `--atari-game all` plays all three. *Also available*
  (`--atari-mode catch`): the **MuZero** variant planning entirely over a *learned* latent model on
  CatchEnv (+1.0 vs random −0.55, 0 environment steps inside the search — the defining MuZero property).
  Full pixel-Atari to human level is compute-bound; MinAtar is the honest, genuinely-learned stand-in.
- **Chess** — a **weak-but-real** AlphaZero learner: it **crushes random (100%)** but loses to a
  classical minimax (even depth-1). Chess mastery needs vastly more self-play than a 4060 session, so
  the demo shows **self-play** (the net genuinely playing itself) rather than rigging a win.

## How they were built (reuse, not reinvention)

All three share ONE engine. The neural AlphaZero search (`az/mcts.py::NeuralMCTS`) runs over a generic
`GameAdapter` contract, so chess, Connect-4 (and TicTacToe) all plug into the *same* search. Atari uses
the single-agent MuZero (`az/muzero_rl.py`) — the same idea with a **learned** model instead of a
simulator. Training drivers: `az/train_connect4_gpu.py`, `az/train_atari_gpu.py`; reload check:
`az/verify_checkpoints.py`; CI gate: `az/_test_run_engines.py`.
