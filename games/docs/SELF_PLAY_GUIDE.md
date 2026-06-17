# Chess self-play — verified state + how to kick it off (deliverable A)

**Status (2026-06-07, overseer RWYB-verified, turnkey):** done end-to-end, nothing pending. The agent plays
purposeful, winning chess; the self-play loop is champion-gated (it can never get worse than the strong base)
and disk-bounded. The only thing left is to run the one command below.

## ▶ Click play (the one command — 2026-06-08)
```bash
# preflight (venv/cuda/python-chess/seed/lock/disk), then launch the standard 2h recipe:
python selfplay/run_engine.py                 # standard (2h, mix+distill+auto-balance+curriculum)
python selfplay/run_engine.py --recipe quick  # 15-min smoke
python selfplay/run_engine.py --list          # show recipes
python selfplay/run_engine.py --doctor        # preflight only
```
`run_engine.py` is the launcher: it runs a **--doctor preflight** (auto-clears a stale lock, checks GPU/deps/seed/
disk) and then starts the engine with sane defaults. Under the hood it calls the underlying command, still available
directly:
```bash
python selfplay/play.py learn-watch --temperature 1.0
```
It seeds the strong bootstrap net, launches training in the background, and streams games live while the **champion
gate** guarantees the playable net never regresses below the strong base; old checkpoints are pruned so disk stays
bounded; Ctrl-C stops it cleanly.

**Pre-restart safety (2026-06-09, RWYB-verified — see [docs/PRE_RESTART_AUDIT_2026_06_09.md](docs/PRE_RESTART_AUDIT_2026_06_09.md)):**
the trainer now runs a **pre-training invariant gate** (`run_invariants_check.py`, exit 2 = halt) before any run; the
train step has a **NaN/Inf guard + gradient clipping + value-loss weighting**; every eval carries its **95% Wilson CI**
(with a loud WARN when the gate would decide on noise) plus a **forgetting axis** (current net vs the frozen seed); and
self-play health (distinct openings, decisiveness, grad-norm, nan-count) is logged each iter. For a long UNATTENDED run,
wrap the trainer in the **out-of-process watchdog** (catches a hang/segfault the in-process supervisor can't):
```bash
python selfplay/watchdog.py --ckpt-dir robust_dual --max-stall-s 900 --max-hours 8 \
    -- python -m az.train_robust --ckpt-dir robust_dual --supervise <your flags>
# run the invariant gate standalone any time:
python run_invariants_check.py   # exit 0 = safe to train
```

**What's new (2026-06-08, all RWYB-verified, test-gate green 4/4):**
- **Dense teacher distillation** (default ON for mix/teacher): the net imitates the teacher's move at EVERY teacher
  move (one-hot policy target), not just the sparse game outcome — denser learning signal. `--no-teacher-distill` to off.
- **`--auto-balance`**: self-tunes throughput (workers/games/steps) per unit time from the `--max-hours` budget so you
  don't hand-tune; the learning contract (gate/anchor-kl/curriculum/lr) stays fixed.
- **Un-bloated checkpoints**: `net_iterN.pt` is now weights-only (~42MB); the optimizer+replay-buffer live in one
  rolling `train_state.pt` (was ~700MB-and-growing per file). Old checkpoints still load (backward-compat).
- **Self-checks**: `python run_tests.py` runs the engine test gate; the engine sweeps orphan
  `*.tmp` at startup and auto-emits a learning report at the end.

Verified end-to-end: seeds strong -> game 1 a checkmate in 21 plies -> champion held at 1.0 vs random while
degrading candidates were rejected.

**Honest level (a fact, not a TODO):** "very good" = it crushes random and plays coherent, winning chess (learned
by imitating the classical engine). It will NOT beat the classical engine via self-play on this hardware -- that is
a genuine compute ceiling (days-weeks on a 4060), not a task pending for you. The loop is safe to run as long as you
like; it never degrades the player.

## The honest story
Pure self-play **from scratch** hit a compute ceiling on the 4060 — over 107 iterations the loss learned
(3.38 → 1.91) but strength-vs-random stayed ~0 (`az/strength_curve.json`). The resourceful pivot:
**bootstrap the net by supervised imitation of the strong classical engine** (the in-budget "oracle" that
beats random 100%). Imitation is far more sample-efficient than RL-from-scratch, so the net gains real
strength in budget; self-play then refines from that strong base.

## Verified numbers (independently re-run by the overseer, not the builder)
| Condition | Result | Meaning |
|---|---|---|
| net-only vs **random** | 35W/15D/**0L** over 50 games (70% win, 85% score) | **real strength** — never loses to random (baseline was ~0) |
| net-only vs random (30 short games) | 28/1/1 = 93% | shorter games convert material wins instead of drawing by repetition |
| net-only vs **classical depth-1** | 0/50 | the **honest ceiling** — imitation approaches but can't exceed the teacher |
| `net-watch` smoke | **checkmate in 27 plies** | plays coherent, winning chess (1.e4, development, queen infiltration, promotion mate) |
| refine-from-strong-base | seed weights **120/120 identical** to the bootstrap net | self-play genuinely continues from the trained weights, not a fresh net |

The net reproduces the teacher's exact move ~35% of the time (vs ~3% random). MCTS strictly helps vs random
(100%) but not yet vs the classical engine (the priors aren't strong enough for a few sims to out-search alpha-beta).

## How to use it (run from the repo root)

```bash
# WATCH the learned net play (net-only by default; add --mcts for search; --opponent classical|random)
python selfplay/play.py net-watch --opponent random
python selfplay/play.py net-watch --opponent classical --mcts --mcts-sims 24

# PLAY against the net yourself (you are White here; --ascii for a text board)
python selfplay/play.py net-play --net-white --ascii

# WATCH IT LEARN (the integrated experience): kicks off training in the background AND streams a net-vs-net
# self-play game move-by-move LIVE in the foreground, from successively-trained checkpoints, with the strength curve.
python selfplay/play.py learn-watch --temperature 1.0
#   --temperature 1.0 = each self-play game VARIES (observe variation; 0 = deterministic, games repeat)
#   --mcts (both sides search) | --move-delay 0.3 | --board (ASCII) | --max-games N | --no-train (just watch an existing run)
#   THROUGHPUT (control the rate of parallel instances -> more learning per unit time):
#     --train-opponent self --parallel-games 8
#     generates self-play games in GPU-BATCHED groups of 8 (one net forward does up to 8 leaf evals
#     -> ~Nx fewer GPU round-trips; ~5.9x fewer launches measured at G=6). Raise it + watch VRAM.
#     (Applies to opponent='self', the batchable case; mix/teacher stay sequential.)

# KICK OFF SELF-PLAY LEARNING headless (self-play -> train -> eval, resumes from the bootstrap, separate dir)
python selfplay/play.py learn
#   -> writes net_iterN.pt + a strength curve. The CHAMPION GATE (default ON) keeps the playable net MONOTONIC --
#      it can NEVER drop below the bootstrap (see below). Use --no-champion-gate to watch raw self-play fall.

# REGENERATE the bootstrap from scratch (~20 min; only if you want to retrain the base)
python az/bootstrap_supervised.py --channels 80 --n-blocks 8 \
    --target-positions 20000 --self-games 220 --vs-random-games 180 --workers 8 --epochs 6
#   (epochs 6 ~ the generalization optimum; 12 overfits — val move-match peaked at epoch 6)

# RE-EVAL strength honestly (4 conditions: net-only/+MCTS x random/classical)
python -m az.eval_bootstrap \
    --ckpt az/bootstrap_checkpoints/net_bootstrap.pt --games 50 --classical-depth 1
```

## Opening diversity — vary the starting conditions (2026-06-09)
**Symptom the user spotted: "it plays the same way."** Every self-play game used to start from the IDENTICAL
startpos; the only variation was Dirichlet root noise + a short temperature window, which washes out on a
sharply-peaked imitation net. So games funnelled into one rote line, the net reinforced one repertoire (bad
learned habits), and the VALUE head rarely saw diverse decisive positions — self-play can't teach what it never
visits. This is the textbook AlphaZero/Leela failure mode; the textbook fix is to **start each self-play game from
a distinct, sound opening** (Leela uses randomized opening books for exactly this).
- **`--opening-mode {startpos,book,random,mixed}`** (DEFAULT `mixed`) + **`--opening-plies N`** (default 4). `book` =
  a curated library of 64 sound, balanced openings across a broad ECO spread; `random` = N guarded random plies
  (rejected+resampled if the start is already lost); `mixed` = book line + guarded random jitter (max diversity
  without ever starting from a blundered position). The opening plies are **NOT training samples** — only the net's
  own searched moves are. **EVAL stays on startpos**, so the strength curve is unchanged as a yardstick. Module:
  [`az/openings.py`](az/openings.py) (self-test: `python -m az.openings`).
- **The DIVERSITY GAUGE** (so this can never silently die again): every iter the trainer records
  `selfplay_distinct_starts` + `selfplay_decisive_frac` in `strength_curve.json`, prints them per-iter, WARNs loudly
  if diversity collapses, and the end-of-run learning report flags `*** DIVERSITY DEAD ***`. The prior 111-iter run
  had no such instrument, so collapsed openings looked identical to a healthy run on loss/win-rate alone.

## Honest ceiling + what's next
- The net **imitates** the classical engine, so it crushes random but **loses to the alpha-beta teacher** (depth-1).
  No master-level claim.
- **Naive self-play degrades the bootstrap — but the CHAMPION GATE (default ON) now stops that from reaching you.**
  Measured: 4 raw self-play iters take the net from **70% win / 0 losses vs random** down to **~35% win / ~37.5%
  losses** (RL distribution-shift away from the strong teacher-imitation). The **champion gate** (now built into
  `learn`/`learn-watch`) only promotes a candidate that does NOT regress on winrate-vs-random, refines each
  candidate FROM the champion, and generates self-play BY the champion — so the playable net (`latest.pt`) is
  **MONOTONIC: it can never drop below the bootstrap.** Verified independently: with the gate ON, candidates at
  0.50 / 0.67 / 0.50 were all REJECTED and the champion held at 0.833. **So "kick off → it never gets worse" is now
  TRUE; the curve is flat-or-up, never down.** (`--no-champion-gate` shows the raw fall.)
- **Making it actually CLIMB past the bootstrap (the remaining frontier — compute-bound, not a 4060 quick win):**
  refine vs the classical TEACHER (`--selfplay-opponent teacher|mix`) and/or anchor to the bootstrap policy
  (`--anchor-kl`), with more GPU time. Both are scaffolded and run end-to-end; climbing is a compute question, not a
  code one. The champion gate means experimenting here is safe — a bad run can't drop the playable net.
- **Watch the RIGHT axis for progress vs the engine (2026-06-08).** The live visualizer now plots a **third,
  draw-aware "climb score" series (blue)** = `(wins + 0.5·draws)/games` vs the classical engine. Progress vs a
  stronger engine shows up as losing → **DRAWING** first, which leaves the win-rate (orange) pinned at 0 but RAISES
  the blue score from 0 toward 0.5 — so the blue line is the one that moves before the win-rate does, and it is the
  axis the champion/curriculum gate actually climbs on.
- **A MOVING teacher (curriculum, 2026-06-08).** `--curriculum` bumps the self-play teacher depth by 1 each time the
  net masters the current depth (draw-aware score ≥ `--curriculum-threshold` vs the *current* teacher), capped at
  `--curriculum-max-depth`. It advances exactly once per crossing (latched) and the climbed depth survives
  crash-resume and `--supervise` restart — so a long unattended run keeps raising the bar instead of saturating
  against a fixed depth.
- **LEARN FASTER — the big lever: `--selfplay-workers N` (2026-06-08).** The diagnosis behind "it learns slowly":
  self-play is **CPU-bound** (the MCTS tree ops + python-chess move generation dominate; the net eval is microseconds),
  so a single self-play process pegs **1 of ~20 cores** and the GPU sits at ~20-40% (never the bottleneck). The fix is
  multiprocess self-play **actors** (`az/selfplay_pool.py`): `--selfplay-workers 16` runs 16 worker PROCESSES
  generating games in parallel → **measured 13.5× throughput** (0.13 → 1.76 games/s) → ~13.5× more iters/hour → the
  strength curve climbs ~13.5× faster. Use with `--train-opponent self`. This is the #1 "learn faster" knob; combine
  with cheap eval (`--eval-games 4`) so eval doesn't become the new bottleneck.
    ```bash
    python selfplay/play.py learn-watch --train-opponent self --selfplay-workers 16 \
        --selfplay-sims 64 --games-per-iter 96 --eval-games 4 --temperature 1.0
    python -m az.learning_report --ckpt-dir robust_fast   # iters/hr, games/s, curve slope, parabolic?
    ```
- **`--parallel-games N`** (GPU leaf-batching) trims GPU round-trips (~5.9× fewer launches) — a secondary win since the
  GPU was never the bottleneck; `--selfplay-workers` is the dominant lever for THIS (CPU-bound) workload.
- **HONEST CEILING for "world-class" learning.** Faster learning raises iters/hour, not the asymptote. Beating a
  *strong* engine via AlphaZero self-play needs datacenter-scale compute (AZ used thousands of TPUs); a single 4060
  in hours does thousands of games, not the millions AZ-strength needs. So the LEARNING ENGINE here is world-class-grade
  (multiprocess actors + GPU batching + champion-gate flywheel + curriculum + self-evolving autonomy) and the curve
  climbs as fast as the hardware allows — but the absolute strength ceiling is compute-bound, not a code defect.
- **The ONE lever that raises the ceiling on a single box: a STRONGER TEACHER (better data, not more compute).** The
  current base imitates classical *depth-1* (weak). Re-bootstrap by imitating a much stronger teacher so the net
  *starts* far stronger, then dual-refine. Now turnkey + non-destructive (`--ckpt-dir` writes a NEW dir, never
  clobbers the good base):
    ```bash
    # multi-hour: stronger-teacher imitation bootstrap (classical depth-4) -> its own dir
    python -m az.bootstrap_supervised --gen-depth 4 --ckpt-dir bootstrap_d4 \
        --target-positions 20000 --workers 16          # ~hours: depth-4 gen is ~1 pos/s/worker
    # then dual-refine FROM the stronger base, fast + visual:
    python selfplay/play.py learn-watch --ckpt-dir robust_d4 --train-opponent mix \
        --selfplay-workers 16 --anchor-kl 1.0 --selfplay-sims 64
    ```
  (Stockfish is an even stronger teacher if installed: pass `--engine-path stockfish.exe` to the trainer's teacher.)
- A known tuning lever: the shipped checkpoint is the epoch-11 (slightly over-fit) net; an early stop ~epoch 6
  (or label smoothing / more positions) would likely raise held-out move-match and net-only strength.
- Data generation is CPU-bound (the pure-Python engine ~4.5 pos/s/worker); ~8 workers scaled best on this box.

## Files
- `az/bootstrap_supervised.py` — classical-oracle labelled positions → CE(policy)+MSE(value) imitation training (resumable)
- `az/eval_bootstrap.py` — honest colour-alternated 4-condition strength eval
- `play.py` — `net-play` / `net-watch` (both `--mcts`, `--live`), `learn`, and `learn-watch` (kick-off + live self-play watch + the curve; `--temperature` for varied games)
- `az/train_robust.py` — hardened self-play loop (checkpoint/resume, OOM guard + bounded floor-retry, `--max-hours`, `--ckpt-dir`) + the CHAMPION GATE (`--champion-gate`, default ON, monotonic) + climb scaffold (`--selfplay-opponent`, `--anchor-kl`) + the MOVING-teacher curriculum (`--curriculum`, persisted across restart) + throughput knobs (`--selfplay-workers` multiprocess actors, `--parallel-games` GPU batching)
- `az/selfplay_pool.py` — MULTIPROCESS self-play actors (the #1 "learn faster" lever): N CPU worker processes generate games in parallel, **13.5× measured** at N=16 on 20 cores (overseer-RWYB). `--selfplay-workers N`
- `az/learning_report.py` — quantifies a run: iters/hour, self-play games/s, champion-floor monotonicity, and the strength-curve slope + whether it's ACCELERATING (parabolic). `python -m az.learning_report --ckpt-dir <dir>`
- `az/batched_selfplay.py` — GPU-batched, GAME-parallel self-play: G games in lockstep, leaf evals batched into one net forward (the `--parallel-games` engine). Game-parallel, not tree-parallel → per-game search is identical to `mcts.MCTS`; ~5.9× fewer GPU launches at G=6 (overseer-RWYB)
- `az/live_viz.py` — TRUE browser visualizer (`live.html`, real piece SVG, auto-refresh, no server/CDN) with the draw-aware climb-score series

Checkpoints (`az/bootstrap_checkpoints/net_bootstrap.pt`, `.npz`) are local regenerable artifacts (gitignored), not committed.
