# games_engine — Chess, Connect-4, Atari

Three engines that **actually play** their games — real programs, not an LLM. A shared
AlphaZero/MuZero core (`az/`) drives all three, with a turnkey terminal demo. **Self-contained**:
no dependency on any other project (only `torch`, `python-chess`, `numpy`, `minatar` —
see [`requirements.txt`](requirements.txt)).

## Run it

```
python run_engines.py       # play ALL THREE engines, rendered move-by-move (the demo)
python play_chess.py        # just chess      (AlphaZero champion self-play)
python play_connect4.py     # just Connect-4  (trained net vs a 1-ply win/block heuristic)
python play_atari.py        # just Atari      (real MinAtar Breakout; --game all = 3 games)
```

Add **`--web`** to any of these to also watch in a **live browser UI** (real graphics, auto-refreshing,
no server) -- e.g. `python run_engines.py --web`. See **[WATCH_UI.md](WATCH_UI.md)**.

**Play against the AI yourself** -- `python selfplay/play.py net-play --mcts` (chess vs the champion) or
`python play_connect4.py --human` (Connect-4 vs the trained net). See **[PLAY_HUMAN.md](PLAY_HUMAN.md)**.

Run guide: **[docs/RUN_ENGINES.md](docs/RUN_ENGINES.md)**.   Lift this into its own repo: **[docs/MOVE_TO_OWN_REPO.md](docs/MOVE_TO_OWN_REPO.md)**.

## The three engines (honest strength)

| Engine | Algorithm | Plays | Measured strength |
|---|---|---|---|
| **Chess** | AlphaZero | champion **self-play** | crushes random (100%); weak-but-real vs a classical minimax — chess mastery is compute-bound |
| **Connect-4** | AlphaZero | net vs 1-ply win/block heuristic | **W40/L0 vs random, W33/D2/L5 vs the heuristic** (was losing 0:24 before training) |
| **Atari** | DQN (+ MuZero) | 3 real **MinAtar** games | Breakout **~300×** random, Space Invaders **~12×**, Asterix **~2.9×**; a MuZero/CatchEnv variant is at `--atari-mode catch` |

These are SOTA **algorithms** genuinely running + playing, trained as far as one RTX 4060 affords —
not superhuman (that is compute-bound), and we don't pretend otherwise.

---

# Chess engine — in depth

A chess engine that **actually plays chess** — its own program, not an LLM, not
Claude. This is a **capability proof**: an AlphaZero-lineage engine built from a
classical baseline up to the neural-MCTS frontier.

Two layers:

1. **Phase 1 — a real classical engine** (`chess_engine/engine.py`): negamax + alpha-beta +
   iterative deepening + a material/PST/mobility evaluation + MVV-LVA move
   ordering + quiescence search. This genuinely *plays* (crushes a random mover,
   finds forced mates).
2. **Phase 2 — the AlphaZero frontier** (`az/`): a residual CNN (policy + value
   heads), PUCT-guided MCTS, and the self-play → train loop, implemented to the
   real structure (arXiv:1712.01815). **The self-play → train loop RUNS and
   demonstrably LEARNS**: a bounded ~5-minute demo on an RTX 4060 drove the
   AlphaZero loss from **8.55 → 2.48** over 4 iterations (see `az/train_demo.py`
   and `az/train_metrics.json`). Playing strength is honestly *compute-bounded* —
   a strong AZ needs days-to-weeks on this GPU — but the bootstrap (net → MCTS
   targets → gradient steps → better net) is real and observable.

Legal move generation, castling, en-passant, check/checkmate, draw rules are all
delegated to **python-chess** (`pip install python-chess`). We do not reinvent
chess rules; we add *search* and *evaluation* on top.

---

## Install / environment

Windows, project venv:

```
.venv\Scripts\python.exe -m pip install python-chess
```

`torch` and `numpy` are required only for the Phase-2 `az/` scaffold (already
present in this project's venv: torch 2.7.1, numpy 1.25.2). Phase 1 needs only
`python-chess`.

---

## Phase 1 — the classical engine

### Run it

```bash
# The litmus test: engine (depth 4) vs a random mover, 20 games, print game 1.
.venv\Scripts\python.exe selfplay\play.py vs-random --games 20 --depth 4 --print-game

# Engine vs engine self-play to a finished game.
.venv\Scripts\python.exe selfplay\play.py vs-self --depth 4 --print-game

# Best move for any position (simple CLI; FEN in, SAN + eval + PV out).
.venv\Scripts\python.exe selfplay\play.py position --fen "<FEN>" --depth 4

# Nodes/time-per-move sanity (confirms alpha-beta is pruning).
.venv\Scripts\python.exe selfplay\play.py bench --depth 4
```

### What's inside (`chess_engine/engine.py`)

- **Negamax + alpha-beta pruning** with **iterative deepening** (depth 1..N,
  time-bounded; keeps the best move from the last completed depth on timeout).
- **Evaluation** (side-to-move relative): material values + **piece-square
  tables** for every piece (separate king mid/endgame tables) + **mobility**
  (legal-move differential) + bishop-pair + a check/king-safety nudge +
  endgame detection that re-centralises the king.
- **Move ordering**: **MVV-LVA** (captures first, big victim / small attacker),
  promotions and checks prioritised — this is what makes alpha-beta actually
  prune.
- **Quiescence search**: at the leaves, search only captures/promotions until
  the position is quiet, so the static eval is applied to a stable position
  (mitigates the horizon effect).

### Strength / correctness (RWYB — measured, see below)

- Finds forced mates: `Qxf7#` (Scholar's Mate) and `Rd8#` (back-rank) are both
  found instantly and scored as mate.
- On Kiwipete it finds the tactical pawn-win `Bxa6` (+116cp).
- Alpha-beta prunes: depth-4 startpos ≈ 19k nodes (a naive minimax would be
  ~millions).
- **vs random, depth 4, 20 games: see "RWYB results" below.**

---

## Phase 2 — the AlphaZero frontier (`az/`)

Import-clean and `py_compile`-clean. The reusable building blocks
(`encoding`/`net`/`mcts`/`selfplay`) keep their library guard (`RUN_TRAINING=False`
in `selfplay.py`, so *importing* never launches a multi-hour run), but the loop is
**no longer scaffold-only**: `az/train_demo.py` actually RUNS a bounded self-play →
train curriculum and learns (loss **8.55 → 2.48** over 4 iters; metrics in
`az/train_metrics.json`, checkpoints in `az/az_demo_checkpoints/`).

### Run the training demo

```bash
.venv\Scripts\python.exe -m az.train_demo
```

~5 min on an RTX 4060. The loss going DOWN across iterations is the learning
signal; win-rate-vs-random stays near 0 at this tiny compute budget — reported
honestly (strength is compute-bounded, not a pipeline failure).

### Visualize self-play (see the mechanics from first principles)

```bash
.venv\Scripts\python.exe az\visualize_selfplay.py
# then open az/selfplay_viz.html in a browser
# (add --terminal for an ASCII/console view; --sims / --max-plies to tune the run)
```

Plays ONE self-play game with the trained net + MCTS and renders a self-contained,
no-CDN, steppable HTML viewer: for each ply it shows the board (inline SVG), the
net's RAW policy prior side-by-side with the MCTS-refined visit distribution `pi`
(so you SEE search refining the prior — the core AZ insight), the value head's
estimate `v`, the move played, and the final outcome `z` (the value target). Raw
per-ply data is also written to `az/selfplay_viz_data.json`.

### Kick it off + WATCH it learn (the current learn pipeline)

```bash
# ONE command: trains in the background (self-play AND vs the engine) and opens a
# TRUE browser visualizer (real board graphics) showing the agent play, live.
.venv\Scripts\python.exe selfplay\play.py learn-watch
```

Full guide: **[docs/SELF_PLAY_GUIDE.md](docs/SELF_PLAY_GUIDE.md)**. In short: the strength comes from a
**supervised bootstrap** (`az/bootstrap_supervised.py` imitates the classical engine → the net beats
random, ~70% / 0 losses); `learn-watch` then refines it via **dual learning** — self-play AND vs the
engine (`--train-opponent mix`; pluggable to a UCI engine like Stockfish via `--engine-path`) — behind a
**champion gate** (`az/train_robust.py`) that guarantees the playable net **never regresses** below the
bootstrap and **climbs** on draw-aware progress vs the engine (with `--curriculum` to raise the teacher).
The board renders in a real browser visualizer (`az/live_viz.py`, inline SVG; `--no-viz` for the terminal).
Resumable/ACID (atomic per-iter checkpoint + champion sidecar + `--supervise` auto-restart + a per-dir lock).
Honest strength eval: `az/eval_bootstrap.py` (reports `adjudicated_fraction`). Honest ceiling: it crushes
random + plays coherent winning chess; beating a *strong* engine via self-play needs more compute.

New modules since Phase-2's first draft: `az/bootstrap_supervised.py` (imitation bootstrap),
`az/train_robust.py` (hardened self-play loop + champion gate + dual-learning + curriculum + pruning),
`az/eval_bootstrap.py` (honest eval), `az/uci_engine.py` (pluggable classical/UCI teacher),
`az/live_viz.py` (the true browser visualizer). `play.py` modes: `human`/`watch`/`net-play`/`net-watch`/
`learn`/`learn-watch`.

| File | What it is | Status |
|------|-----------|--------|
| `az/encoding.py` | board → 19 input planes; move ↔ 4672-way policy index (AlphaZero's 8×8×73 scheme) | runnable, verified: every legal move encodes to a unique index, 0 collisions across startpos / Kiwipete / promotion positions |
| `az/net.py` | residual CNN: stem + 6 residual blocks + **policy head** (4672 logits) + **value head** (tanh scalar) | runnable forward pass (≈10M params at C=64/6 blocks); shapes verified |
| `az/mcts.py` | **PUCT-guided MCTS** using the net's prior + value (no rollouts), Dirichlet root noise, visit-count policy | runnable against the net (now exercised with the TRAINED demo checkpoints, not just an untrained net) |
| `az/selfplay.py` | self-play → replay buffer → train; **loss = CE(policy, MCTS visits) + MSE(value, outcome z) + L2** | `generate_selfplay_game` + `train_step` + `train_loop` all runnable; the import-guard (`RUN_TRAINING`) only stops training on *import* |
| `az/train_demo.py` | bounded, time-boxed self-play → train demo that **RUNS and learns** (loss 8.55 → 2.48 / 4 iters) | runs in ~5 min on an RTX 4060; writes `train_metrics.json` + `az_demo_checkpoints/net_iter*.pt` |
| `az/visualize_selfplay.py` | first-principles **self-play VISUALIZER** (prior vs MCTS-refined `pi`, value `v`, outcome `z`) | runs one game; emits self-contained `selfplay_viz.html` + `selfplay_viz_data.json` |

### The loss (exactly the paper)

```
L = (z - v)^2  -  pi^T log p  +  c||theta||^2
    value MSE     policy CE       L2 weight decay
```

where `pi` = MCTS visit distribution, `z` = game outcome in {+1,0,-1} from each
state's player's perspective, `p,v` = net policy/value.

### Path from the classical engine to a trained AlphaZero

1. **Use `engine.py` as the strength yardstick.** A freshly-initialised net +
   MCTS plays randomly (priors ≈ uniform). The classical engine is the bar the
   learned agent must clear, then beat.
2. **Self-play curriculum** (`selfplay.train_loop`): each iteration plays a batch
   of self-play games (MCTS with ~800 sims, temperature=1 for the first ~30
   plies then greedy, Dirichlet root noise for exploration), appends
   `(planes, pi, z)` triples to a replay buffer, then takes gradient steps on the
   loss above; checkpoint; repeat. The net improving makes MCTS stronger, which
   produces better targets — the bootstrap.
3. **Scale the knobs** from the scaffold defaults to real values: net
   C=256 / 19–20 residual blocks; MCTS ~800 sims/move; thousands of self-play
   games per iteration; many iterations.
4. **Compute reality.** DeepMind trained on thousands of TPUs in hours. On a
   single RTX-4060 (8 GB) this is a *long* project: expect a much smaller net,
   far fewer sims, and days-to-weeks for even modest strength. The scaffold is
   structured so the only thing standing between it and a real (if small) run is
   compute + `RUN_TRAINING=True`. Practical accelerators if pursued: start from
   the classical engine's evaluations as a value-target warm-start, use a smaller
   board-feature net, and parallelise self-play.

### Run the scaffold smoke tests (no training)

```bash
.venv\Scripts\python.exe -m az.net        # forward-pass shapes
.venv\Scripts\python.exe -m az.mcts       # MCTS returns a legal move
.venv\Scripts\python.exe -m az.selfplay   # data-gen + 1 train_step
```

---

## RWYB results

Real run: `play.py vs-random --games 20 --depth 4 --time 1.5 --seed 0 --print-game`
(full log in `_vs_random_d4.log`). The `--time 1.5` caps per-move time at 1.5s
with depth 4 as the iterative-deepening ceiling; without it the run is identical
in outcome but slower.

**Engine (depth 4) vs random mover, 20 games:**

```
W/D/L = 20/0/0  out of 20   (score 20.0/20 = 100.0%)
every game ended in CHECKMATE delivered by the engine
engine played White in 10/20 and Black in 10/20 (colours alternated)
engine: 296 moves, 3,686,762 nodes, 463.5s
        avg 12,455 nodes/move, 1.57s/move, ~7,950 nps
```

The engine **crushes random 100%** — it genuinely plays, it does not merely make
legal moves. Sample (game 1, engine = White, mate in 16 full moves):

```
1. e4 h5 2. d4 h4 3. Bf4 Rh6 4. Bxh6 d6 5. Bg5 b6 6. Bxh4 f6 7. Qh5+ g6
8. Qxg6+ Kd7 9. Qxg8 e5 10. Qf7+ Be7 11. Bxf6 Nc6 12. Bb5 a6 13. Bxe7 Qf8
14. Bxf8+ Kd8 15. Bxc6 Be6 16. Qe8#
```

**Correctness verified:**
- Only legal moves: python-chess generates them; an explicit per-move
  `move in board.legal_moves` assertion also passed over a driven game.
- Games terminate properly: all 20 ended in real checkmate; a separate depth-2
  batch and a depth-2 self-play game also terminated in checkmate; draw paths
  (stalemate / insufficient material / 75-move / repetition) are handled.
- Tactics: finds `Qxf7#` (Scholar's Mate) and `Rd8#` (back-rank) instantly and
  scores them as mate; finds `Bxa6` (+116cp pawn-win) on Kiwipete.
- Alpha-beta prunes: ~19k nodes at depth-4 startpos (naive minimax ≈ millions);
  ~12.5k nodes/move averaged across the 20-game match.

**Two real bugs found and fixed during RWYB** (the litmus did its job):
1. **push/pop not exception-safe under the time limit.** When `TimeUp` fired
   between `board.push(move)` and `board.pop()`, the move was left on the board,
   corrupting it for the rest of the game. Fixed with `try/finally` around every
   push/pop in `_negamax`, `_quiesce`, `_search_root`. Verified: board FEN is
   byte-identical before/after a timed-out search at limits 0.01–1.0s.
2. **`_extract_pv` ran under the (already-expired) deadline** and could raise
   `TimeUp` out of `search()`. Fixed by clearing the deadline before PV
   extraction and guarding it. (Both bugs only manifest with `--time`; the pure
   depth-bounded path never hit them, which is why the first 4 games ran clean
   before the time-limited rerun exposed them.)

---

## Files

```
games_engine/
  run_engines.py            # turnkey demo: all three engines rendered move-by-move ("click run")
  play_chess.py play_connect4.py play_atari.py   # one engine at a time
  run_tests.py              # test-gate runner (auto-discovers tests/_test_*.py)
  run_invariants_check.py   # catastrophic-correctness gate (move-gen / value-sign / no-hang)
  README.md                 # this file
  chess_engine/             # the classical (non-neural) chess engine
    engine.py               #   negamax + alpha-beta + iterative deepening + evaluation
    uci.py                  #   UCI adapter (play it from a chess GUI)
    perft.py                #   move-generation perft check
    play_human.py           #   human-vs-engine console play
  selfplay/                 # AlphaZero self-play TRAINING ops
    play.py                 #   game runners (vs-random/vs-self/position/bench) + learn-watch
    run_engine.py           #   recipe launcher    watchdog.py  # unattended-run guard
  az/                       # the shared neural core (AlphaZero + MuZero + DQN)
    encoding.py net.py mcts.py selfplay.py        # planes, residual CNN, PUCT MCTS, train loop
    connect4.py muzero.py muzero_rl.py dqn_minatar.py minatar_env.py   # games + model-based RL
    train_*.py bootstrap_supervised.py train_robust.py   # training drivers
    visualize_selfplay.py live_viz.py             # self-play visualizers
    checkpoints/            #   trained weights (*.pt, committed -- the small "brains")
    robust_from_bootstrap/  #   the chess AlphaZero champion (champion.pt ~41MB, committed)
  tests/                    # the engine test suite (_test_*.py)  ->  python run_tests.py
  docs/                     # RUN_ENGINES.md, SELF_PLAY_GUIDE.md, MOVE_TO_OWN_REPO.md, ...
```
