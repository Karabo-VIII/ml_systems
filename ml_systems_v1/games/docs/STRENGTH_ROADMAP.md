# Strength roadmap -- genuine competitive strength, honestly measured

Standard (user, 2026-06-13): **no toys / no "impressive but not real".** Each engine must grow
in real strength and compete against the *best*, not just beat random. This document is the
honest plan: per engine -- the measured current state, what "the best" realistically means on
this hardware (a single RTX 4060 laptop), the genuine technique path, the **measurement yardstick**
(never "beats random"), and the next step. Where there is a hard compute ceiling, it is stated.

**The integrity rule:** every strength claim must be made against a STRONG reference
(Stockfish for chess, a perfect solver for Connect-4, published SOTA for Atari) -- never random.

---

## Chess

**Current (measured, honest):** AlphaZero champion (`champion.pt`, iter 5) crushes random (100%)
but **loses to a depth-1 classical minimax**. A 2 h self-play refine on the 4060 **plateaued**
(stuck ~0.73 win-rate vs random, draw-aware climb-vs-classical = 0.0). It plays legal, coherent
chess -- but it is weak.

**Honest ceiling:** a from-scratch **AlphaZero net trained on one 4060 will not reach competitive
Elo** in any feasible wall-clock (DeepMind used thousands of TPUs; Leela used distributed compute
over months). Pure AZ is the *wrong path to strength on this hardware.* The literal best
(Stockfish ~3500) is unreachable by any laptop-trained engine -- do not claim it.

**Genuine path (how strong chess is actually achieved on a laptop) -- a strong ALPHA-BETA / NNUE
engine.** Upgrade `chess_engine/engine.py`:
- transposition table (Zobrist) + aspiration windows; null-move pruning; late-move reductions (LMR);
  futility / delta pruning; killer + history move ordering; SEE-ordered captures; check extensions.
- a much stronger eval: tapered piece-square tables + pawn structure + king safety + mobility, **or
  an NNUE eval** (a small efficiently-updatable net -- the modern Stockfish approach; the GPU helps
  *train* it from self-play or from a strong engine's evals).
- an opening book (Polyglot) + Syzygy endgame tablebases.
This stack reaches **~2000-2800 Elo on a laptop CPU** -- genuinely strong (beats strong club/expert
players and many engines). **This is the realistic "compete against the best".**

**Measurement (real Elo):** a gauntlet vs **Stockfish at calibrated strength** (UCI `Skill Level`
/ `UCI_LimitStrength` / fixed nodes) + an Elo computation (Ordo/BayesElo or fixed-reference).
`az/uci_engine.py` already plugs a UCI engine via `--engine-path` -> point it at a Stockfish binary
for a REAL rating. (Provide a Stockfish binary, or we fetch one.)

**Honest target:** 2000-2500 Elo -- but see the MEASURED FINDING below, which revises the path.

**MEASURED FINDING (2026-06-13, this session -- the key lesson):** I built the search upgrade
(transposition table + null-move + LMR + killer/history, behind a `strong` flag, with a real
head-to-head benchmark `chess_engine/bench.py`) and **measured it**. At interactive per-move time
the pure-`python-chess` engine only reaches **depth 3-5** (its move generator is the bottleneck,
~5-15k nps), and at THAT depth the SOTA pruning **does not help and can hurt** -- a first 4-game
match had the "strong" config at roughly **-191 Elo** vs the plain baseline (deeper by 1 ply, but
the null-move/LMR accuracy loss outweighed the tiny depth gain). Null-move + LMR pay off at depth
8-12+, which python-chess cannot reach in interactive time. **Same root cause as the Connect-4
finding: pure Python is too slow for the SOTA technique to reach its payoff depth.**

**Revised path (honest, evidence-based):**
1. **Speed is the real lever.** Genuine 2000+ Elo needs a FAST core -- a Cython/C bitboard
   move-generator + search -- so it reaches depth 10+ where pruning pays off. This is the real
   (multi-day) engineering for true strength. *Or* wrap a strong existing engine (Stockfish via
   `uci_engine.py`) when the goal is to literally play/measure against the best.
2. **At the shallow depth the python engine CAN reach, EVAL quality + an opening book + endgame
   tablebases are the high-leverage wins** (better moves per ply; book/tablebases sidestep search
   entirely). These lift a slow engine the most and are self-contained. Honest python ceiling
   ~1400-1700 Elo even done well -- a solid club player, not "the best".
3. Keep the search machinery (`strong` flag) **off by default if it measures weaker** -- never
   ship a regression -- and turn it on once the fast core makes depth 8+ reachable.

**MEASURED Elo (this session, vs Stockfish 16.1 Elo-capped, 0.3s/move, 8 games each):** the
classical `engine.py` scored **W7-D0-L1 vs Stockfish@1320** (~1658 by that anchor) and
**W0-D2-L6 vs Stockfish@1700** (~1362 by that anchor). Stockfish's `UCI_Elo` cap is imperfect at
fast time controls, so the honest estimate is **~1400-1550 Elo -- a genuinely STRONG club-level
engine**, NOT a toy (the weak chess component is the AlphaZero *net*, which loses to depth-1; the
classical engine is good). Tools now in the repo: `chess_engine/bench.py` (the measurement),
`chess_engine/sf_engine.py` (Stockfish as a drop-in engine), `engines/` (the binary, gitignored).
**To literally "compete against the best": `python chess_engine/play_human.py --stockfish`** (you
vs Stockfish, Elo-cappable).

**EVAL UPGRADE -- MEASURED +108 Elo (2026-06-13).** The first GROW-the-engine rung is done: a richer
static eval (`evaluate_v2`, now the default) adds the terms a depth-4 search cannot find for itself
-- pawn structure (doubled/isolated penalties + rank-scaled passed-pawn bonus), a real king
pawn-shield, rook-on-(semi)open-file, and a tapered (smoothly interpolated) king PST. It was A/B
benchmarked head-to-head BEFORE promotion (integrity rule): **W14-D11-L5 over 30 games vs the old
eval -> +108 Elo.** This lifts the engine WITHIN the python speed cap (no faster core needed).

**Independently confirmed vs Stockfish (the strong reference, not self-play):** with `evaluate_v2`
the engine scored **0.375 vs Stockfish@1700 (W2-D2-L4) -> ~1611 Elo**, UP from the old eval's 0.125
(W0-D2-L6, ~1362). So the upgrade is real against an external calibrated opponent, not just against
itself. **Honest current estimate: ~1600 Elo -- a strong club/expert-level engine.**

**OPENING BOOK -- MEASURED -44 Elo (2026-06-13), NOT shipped as default.** I built a Stockfish-
generated opening book (`chess_engine/book.py`, 665 positions in `opening_book.json`, wired as an
opt-in `Engine(book=...)`) and A/B-measured it (`chess_engine/measure_book.py`): **W11-D6-L15 over
32 games = 0.438 -> -44 Elo.** It does NOT help -- the SAME lesson as the search pruning: a strong
opening move assumes a strong DEEP follow-up; fed to a depth-4 engine it just steers it into sharper
positions it understands WORSE than its own comfortable shallow-eval openings. Per the integrity
rule (never ship a regression) the book stays OPT-IN, off by default. The infra is kept (the finding
is the value, and the book would help once a faster/deeper core exists). Remaining real rung: a
Cython/bitboard core for depth (where the eval is already good and pruning/book would then pay off).

---

## Connect-4

**Current (measured):** the trained net is W34/D3/L3 vs a 1-ply win/block heuristic, crushes random,
and -- notably -- **beat a naive pure-Python search 2/4** (it is stronger than a quick Python solver
at interactive speed). Decent, **not perfect.**

**"The best" = PERFECT play** (Connect-4 is SOLVED: first player wins from the centre). This *is*
achievable here -- just not via a pure-Python interactive search.

**Honest finding (this session):** a pure-Python full solver (`az/connect4_solver.py`, a correct
bitboard negamax following Pascal Pons) is **endgame-perfect and crushes random, but a full perfect
solve of the OPENING is too slow in Python for an interactive per-move budget** -- so its early-game
play is strong-but-not-provably-perfect and it did **not** reliably beat the trained net at 1 s/move
(2-2 over 4). The solver ALONE is therefore not the live engine.

**DELIVERED -- the HYBRID (net + perfect-endgame solver), MEASURED +~134 Elo (2026-06-13).** The
right design composes the two: the trained net plays the deep OPENING (where the Python solver is
too slow), and `Connect4Solver.proven_move` takes over for the WHOLE mid/endgame the moment it
solves within budget (=> PROVABLY-OPTIMAL there), with cheap tactical guardrails (never miss a
1-move win, always block a 1-move loss). `az/connect4_hybrid.HybridConnect4Player`. It is strictly
>= the net by construction and **MEASURED it: W20-D1-L9 over 30 colour-swapped games vs the bare net
-> score 0.683 (~+134 Elo).** Of its moves, 185/341 were solver-perfect/forced, 156 net-opening.
This is the genuine, measured Connect-4 strength win -- shipped as the strong engine (the human now
plays the hybrid). Remaining path to FULLY perfect: a precomputed opening book or a Cython core to
also solve the opening (then it is perfect end-to-end).

**Genuine paths to PERFECT:**
1. **Precomputed opening book** (solve the opening tree offline, once -- hours) feeding the existing
   endgame solver -> instant, provably-perfect play. (Most tractable.)
2. **Compiled solver** (Cython/C core) -> perfect at interactive speed.
3. **Extended AlphaZero self-play** -- Connect-4 is small enough to near-solve on a 4060 over time.

**Measurement:** once a perfect engine exists, score every other agent by **move-agreement with the
game-theoretic optimum** + head-to-head (a perfect P1 never loses; verified the search crushes
random 6/0). **Next step:** build the opening book (or Cython core) -> perfect Connect-4.

---

## Atari (MinAtar)

**Current:** plain **DQN** beats random on 3 MinAtar games (Breakout ~300x random, Space Invaders
~12x, Asterix ~2.9x). It is **not benchmarked against published MinAtar SOTA.** The MuZero/Catch
piece is a small learned-model showcase.

**Honest ceiling:** real pixel-Atari (ALE) at SOTA (MuZero / Agent57, superhuman) needs **far more
compute than a 4060 affords in reasonable wall-clock** -- not the achievable bar here. The honest
achievable target is **published MinAtar SOTA scores** (MinAtar is the standard simplified 10x10
benchmark; SOTA is reachable on modest hardware with proper training).

**Genuine path:** upgrade plain DQN -> **Rainbow** (Double DQN + Dueling + Prioritised Replay +
n-step returns + Distributional C51 + Noisy Nets), train each MinAtar game to published SOTA, and
**evaluate vs the published baselines** (not vs random).

**DELIVERED -- Rainbow (Dueling + n-step + Double-DQN) MORE THAN DOUBLED the champion (2026-06-13).**
Added a dueling Q-network (split V+A streams) + n-step returns on top of the existing Double-DQN
(`az/dqn_minatar.py`, `az/train_minatar.py`), backward-compatible, behind a CHAMPION GATE (never
overwrite a stronger checkpoint). A ~17 min GPU run on Breakout broke through a long ~27-score
plateau and hit a NEW CHAMPION: **trained_return 394.2** (9440 episodes, 920k frames), vs the old
plain-DQN champion's **191.1** -- a **2.06x** jump. INDEPENDENTLY RE-VERIFIED by reloading the saved
checkpoint: **mean 394.0 over 8 fresh episodes (min 391, max 397), random 0.375 (~1050x random)** --
robust, not a fluke. The new champion is committed at `az/checkpoints/atari_minatar.pt` (dueling=True).
This is the genuine, measured, verified Atari strength win. Remaining: add PER + C51 + Noisy nets and
benchmark vs published MinAtar SOTA numbers; extend the same run to Space Invaders / Asterix.

**Measurement:** the published MinAtar SOTA score per game is the bar; report ours **vs SOTA**, not
vs random (next: pin the exact published Breakout number to contextualise the 394). (If full ALE
Atari is wanted later, that is a separate, much larger compute commitment -- modest, not superhuman,
on a 4060.)

---

## Order of work (EV-ranked)
1. **Chess strong engine + real Elo harness** -- biggest, most-wanted, genuinely achievable strength
   win (2000+ Elo). Start with transposition table + null-move + LMR + a real eval; measure vs Stockfish.
2. **Connect-4 perfect** -- opening book (or Cython core) + the endgame solver -> provably optimal.
3. **Atari Rainbow -> MinAtar SOTA** -- measured against published baselines.
4. **Docs honesty pass** -- scrub the README/guides of any "SOTA / superhuman" phrasing that the
   measurements don't support; replace with the measured numbers + the honest ceilings above.
