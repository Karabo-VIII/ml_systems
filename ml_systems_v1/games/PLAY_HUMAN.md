# Play against the AI (human vs AI)

Two engines are turn-based, so you can play them yourself against the **trained** net.
(Atari/MinAtar is single-player real-time, so there's no "human vs AI" there.)

## Chess -- you vs the AlphaZero champion

```
python selfplay/play.py net-play --mcts
```
- Loads the committed **champion.pt** by default and plays it with net + MCTS search.
- You are **White** (you move first). Add `--net-white` to let the AI move first.
- Enter moves as **UCI** (`e2e4`, `g1f3`) or **SAN** (`e4`, `Nf3`, `O-O`).
  Other commands: `board`, `fen`, `moves`, `resign`, `quit`, `help`.
- Strength/speed knob: `--mcts-sims 200` (stronger, slower) ... `--mcts-sims 32` (faster).
- Drop `--mcts` for a faster net-only (no-search) opponent.
- Honest note: the champion crushes random but is a *weak-but-real* learner vs a strong
  classical engine -- chess mastery is compute-bound. It plays real, legal, coherent chess.

(Want to play the **classical** minimax engine instead of the neural net?
`python chess_engine/play_human.py` -- pick a depth with `--depth`. Measured ~1600 Elo vs
Stockfish (0.375 vs Stockfish@1700 with the upgraded eval) -- a genuinely strong club/expert engine.)

### Play the BEST -- Stockfish

```
python chess_engine/play_human.py --stockfish               # you vs Stockfish 16.1 (~1600 Elo cap)
python chess_engine/play_human.py --stockfish --sf-elo 2200 # crank it up (range 1320-3190)
```
This is the honest "compete against the best": Stockfish is one of the strongest engines in the
world and runs great on a laptop, Elo-capped so it is a fair, calibrated opponent. (A from-scratch
engine cannot reach this level on a 4060 -- see [docs/STRENGTH_ROADMAP.md](docs/STRENGTH_ROADMAP.md).)
Requires a Stockfish binary in `engines/` (or set `STOCKFISH_PATH`).

## Connect-4 -- you vs the HYBRID (net + perfect-endgame solver)

```
python play_connect4.py --human
```
- You play the **strongest engine in the repo**: the trained net plays the opening, then a
  **provably-perfect bitboard solver** takes over the mid/endgame (and it never misses a 1-move
  win or fails to block one). **MEASURED +~134 Elo over the bare net** (W20-D1-L9 over 30 games).
- You are **X** and move first; the AI is **O**. Add `--ai-first` to let it move first.
- Drop a disc by typing a **column number 0-6**; `q` to quit.
- The board prints after every move with the column numbers along the bottom.
- (Want the bare net instead? It is still there -- the hybrid just wraps it.)

## Notes
- The boards render in ASCII and auto-fall-back from unicode on terminals that can't encode
  it (Windows cp1252), so they never crash.
- If a 2-hour soak (`soak_run.py`) is still running, it shares the GPU -- the AI will just
  think a little slower. You can stop the soak any time (close its window / kill the process).
