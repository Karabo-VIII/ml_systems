# End-to-End Playbook — Designing & Building a Self-Play Bot

> The **HOW** layer. Companions: [SELFPLAY_SOTA_FEATURES.md](SELFPLAY_SOTA_FEATURES.md) = the *what* (the canonical
> feature set), [PRE_RESTART_AUDIT_2026_06_09.md](PRE_RESTART_AUDIT_2026_06_09.md) = the *applied audit*. This doc is
> the **lifecycle**: how to go from idea → design → implementation → training → trustworthy evaluation → operation,
> for an AlphaZero-lineage self-play bot. Grounded in the real `chess_zero` build (file cites are clickable) and the
> bugs that actually bit us (the Lessons appendix — a playbook that only lists the happy path is sub-SOTA).

The whole thing is one sentence: **a network proposes, search improves the proposal, self-play turns the improved
proposal into training data, and the better network proposes better next time** — wrapped in enough engineering that
it doesn't lie to you or fall over. Everything below is in service of that loop not deceiving you.

```
                 ┌─────────────────────────── the flywheel ───────────────────────────┐
   bootstrap →   NET ──propose──▶ MCTS ──improve──▶ SELF-PLAY ──(s, π, z)──▶ REPLAY ──train──▶ NET'
   (optional)     ▲                (PUCT+noise)      (diverse openings)      BUFFER            │
                  └──────────────────── gate: promote only if it didn't regress ◀─────────────┘
                                         (everything else = making this honest + fast)
```

---

## Phase 0 — Frame it before any code (the highest-leverage step)

| Decision | Options → consequence |
|---|---|
| **Rules known or learned?** | Known (chess, Go) → **AlphaZero** (search the real game). Unknown / pixels (Atari) → **MuZero** (learn a dynamics model, plan in latent space). Imperfect-information (poker) → **ReBeL/CFR-hybrid**. Pick the *smallest* that fits. |
| **Compute budget** | This sets the ENTIRE design. Datacenter → tabula-rasa is fine. **Single GPU (our case)** → you cannot brute-force; you will lean on a **bootstrap**, low sims, small net, and the highest-ROI-per-sim levers. Decide this honestly first. |
| **Success criteria** | "Working" must be *verifiable*: beats baseline X by margin Y on N games with a CI, robust across seeds. Write it down. A vague objective yields confident-wrong autonomy. |
| **Honest ceiling** | State up front what's reachable on your hardware. AZ used thousands of TPUs; a laptop GPU does thousands of games, not the millions AZ-strength needs. The bot will be *grade*-world-class (the scaffold), not *strength*-world-class. Say so. |

> **Lesson:** the single biggest predictor of a wasted month is a fuzzy objective + an unstated compute ceiling.

---

## Phase 1 — Design: the four pillars

### 1a. Algorithm core (the flywheel)
Self-play data generation → MCTS visit-counts as the improved policy target → gradient steps → better net → repeat.
Model-free when rules are known (you search the real game); model-based (MuZero) when they aren't.

### 1b. Search — MCTS / PUCT ([`az/mcts.py`](../az/mcts.py))
- **PUCT** `Q + c·P·√ΣN/(1+N)` — the net's prior `P` guides exploration; `Q` exploits. This is what makes search tractable on a huge action space.
- **Dirichlet root noise** + **temperature schedule** (τ=1 for the first ~30 plies, then greedy) — the *only* exploration in vanilla AZ. **Both wash out on a peaked net** → you also need opening diversity (Phase 3).
- **Value-sign negation per ply on backup** — the #1 catastrophic silent bug. A value is from the side-to-move's view; negate it each level up. Verified by the invariant gate's I1.
- **Low-sim reality (single GPU):** PUCT has no policy-improvement guarantee at ~64 sims. **Gumbel root selection + sequential halving** does — highest search-quality ROI when sims are scarce. (Roadmap lever.)

### 1c. Network ([`az/net.py`](../az/net.py))
- **Backbone:** residual tower (skip connections are load-bearing: +600 Elo vs shallow in AGZ). Add **squeeze-excite or global-pooling bias** for ~free Elo. Transformers (lc0 BT4) are EMERGING but heavier per node.
- **Policy head:** spatial `8×8×73 = 4672` (AZ scheme). **Value head:** scalar tanh is the baseline; **WDL 3-way** is better for draw-heavy games (chess) and is the prerequisite for contempt.
- **Keep `value_loss_weight ≤ 0.5`** — the large early value-MSE otherwise drowns the policy head (value-head overfit).

### 1d. Encoding — the representation contract ([`az/encoding.py`](../az/encoding.py))
- **board → planes:** piece planes + history + side-to-move + castling rights + repetition + 50-move counter. (We use `HISTORY_STEPS=1`; the paper uses 8 — a known depth gap.)
- **move ↔ index bijection:** every legal move maps to a unique index, *including* the special cases — **en passant** (rides the normal diagonal-pawn planes), **castling** (king moves 2 squares), **queen promotion** (queen-move planes), **under-promotion** (9 dedicated planes). **This bijection MUST be verified** (0 collisions, no legal move → None): the invariant gate's I3 does it before every run.

---

## Phase 2 — Implementation: the module map + contracts

The real `chess_zero` decomposition (a clean separation you can mirror for any bot of this nature):

| Module | Contract |
|---|---|
| [`engine.py`](../engine.py) | classical baseline (negamax/αβ) — the strength yardstick **and** the bootstrap teacher |
| [`az/encoding.py`](../az/encoding.py) | board↔planes, move↔index (bijective, verified) — the representation |
| [`az/net.py`](../az/net.py) | residual CNN, policy+value heads, batched inference (`predict_many`) |
| [`az/mcts.py`](../az/mcts.py) | PUCT search over the net (the single source of truth for the search math) |
| [`az/selfplay.py`](../az/selfplay.py) | the reference self-play→data→train primitives + the paper loss |
| [`az/batched_selfplay.py`](../az/batched_selfplay.py) | GPU-batched, game-parallel self-play (throughput) |
| [`az/selfplay_pool.py`](../az/selfplay_pool.py) | **multiprocess** self-play actors (the real throughput lever; CPU-bound MCTS) |
| [`az/openings.py`](../az/openings.py) | opening diversity (book/random/mixed) — vary the starting conditions |
| [`az/train_robust.py`](../az/train_robust.py) | the production loop: self-play → train → eval → **champion gate** → checkpoint → telemetry |
| [`az/bootstrap_supervised.py`](../az/bootstrap_supervised.py) | imitation bootstrap from the classical teacher |
| [`az/eval_bootstrap.py`](../az/eval_bootstrap.py) | honest multi-condition strength eval |
| [`run_invariants_check.py`](../run_invariants_check.py) | the **pre-training invariant gate** (correctness pre-flight, exit 2 = halt) |
| [`watchdog.py`](../watchdog.py) | out-of-process liveness guard (catches hangs the in-process supervisor can't) |

> **Discipline:** one module = one source of truth for its concern. The batched/pool generators **reuse** `mcts.py`'s
> PUCT math rather than re-implement it — so a search fix lands everywhere at once.

---

## Phase 3 — Data generation (self-play): diversity is non-negotiable

Each iteration generates **many games** (`games_per_iter`, 8–64) → appends `(planes, π, z)` to a **FIFO replay
buffer** → trains. Get these right or the data lies to the net:

- **Opening diversity** ([`az/openings.py`](../az/openings.py)). Fixed startpos + a peaked net collapses self-play to
  one rote line; the value head never sees diverse decisive positions. Start every game from a **distinct sound
  opening** (book + guarded jitter), for self-play AND teacher games. This is the "so the bot doesn't think there's
  one starting position" fix — it's the *training* data; the eval/demo stays on startpos for a comparable yardstick.
- **Target integrity:** store the **full MCTS visit distribution** `π` (not a one-hot of the played move), and the
  played move must come from the **same search** that produced `π`. `z` is stored from **each position's mover's
  perspective**. (Invariant gate I2/I8.)
- **Replay window sizing:** too large = stale targets; too small = variance/forgetting. Bound it; sample sparsely
  per game to de-correlate.
- **Opponents:** `self` (diversity, but pure self-play *degrades* an imitation net), `teacher` (high-quality data
  from a stronger bot), `mix` (both). Use `mix` + an **anchor-KL** to refine without drifting off the strong base.
- **Throughput levers (single GPU):** multiprocess CPU actors (self-play is CPU-bound — **spawn, not fork**, with
  CUDA), GPU leaf-batching, and KataGo's **playout-cap-randomization** + **forced-playouts** (roadmap).

---

## Phase 4 — Training

- **Loss** = CE(policy, π) + `w`·MSE(value, z) + L2, with `w ≤ 0.5`. ([`az/selfplay.py`](../az/selfplay.py))
- **Optimizer/schedule:** SOTA is SGD+momentum with a warmup→step-drop LR schedule (Adam chases the non-stationary
  MCTS target). We currently use Adam+flat-LR (a conscious deviation; switch the two together).
- **Numerical safety** (no silent corruption): NaN/Inf guard before `.backward()`, gradient clipping (+ log the
  grad-norm), mixed precision (bf16 on Ampere). All in [`az/train_robust.py`](../az/train_robust.py).
- **Anti-pathology machinery** (each pathology needs a *detector*, see Phase 6):
  - **Champion gate** — promote a candidate only if it doesn't regress (monotonic floor); generate self-play from
    the champion. Stops a noisy/degraded net from poisoning future data.
  - **Anchor-KL** — penalty toward the bootstrap policy; refine from strength without drifting.
  - **Curriculum** — raise the teacher's depth as the net masters the current one (a moving target).

---

## Phase 5 — Bootstrap-first (the compute-poor's decisive move)

Pure RL-from-scratch is datacenter-bound (our 107-iter from-scratch run learned the *loss* but stayed ~0 strength).
The resourceful pivot: **imitation-bootstrap** the net by supervised learning on a strong, in-budget teacher (the
classical engine, or Stockfish), so it *starts* strong; then **dual-refine** from there (`mix` + anchor-KL + gate).
Imitation is far more sample-efficient than RL-from-scratch — it's how a single-GPU build gets real strength in
budget. ([`az/bootstrap_supervised.py`](../az/bootstrap_supervised.py)) **The one lever that raises the ceiling on
one box is a *stronger teacher*, not more compute.**

---

## Phase 6 — Evaluation: make the metrics trustworthy (the part that prevents self-deception)

A self-play win-rate is a **lie about absolute strength** — two co-adapting agents both degrade while their mutual
rate stays ~50%. Build the eval so it can't fool you:

- **Draw-aware score** `(W+0.5D)/G`, **colour-balanced** (alternate colours; White is ~+32 Elo), **deterministic**
  (τ=0, no Dirichlet at eval).
- **Confidence intervals / SPRT.** A 15-game eval has a ±0.25 win-rate CI — *noise*. Log the **95% Wilson CI** on
  every number and WARN when the gate is deciding on noise; size the gate eval accordingly. ([`az/train_robust.py`](../az/train_robust.py) `wilson_ci95`)
- **Fixed external baseline** — eval vs a *non-co-adapting* opponent (a frozen engine / Stockfish at fixed depth),
  not only vs random/self.
- **Forgetting axis** — periodically play the current net vs the **frozen seed**; a sustained <0.5 means it's losing
  to its own starting point = catastrophic forgetting the vs-random axis *cannot* see. (We caught exactly this on the
  first real run.)
- **Gating** — best-of-N promotion (≥55%), seeded from a *large enough* eval that a lucky read can't lock the floor.

---

## Phase 7 — Observability & numerical safety (no silent failures)

The rule: **every silent failure mode must have a non-silent detector.** A SOTA system is one whose instruments
catch what the headline metric hides.

| Failure mode | Detector (built) |
|---|---|
| sign/encoding/terminal/target regression | **invariant gate** ([`run_invariants_check.py`](../run_invariants_check.py)), exit 2 = halt, runs before every train + in CI |
| NaN corrupts weights silently | guard before `.backward()` → skip+count+abort |
| gradient blow-up | grad-norm logged per iter |
| opening collapse | **diversity gauge** (distinct-starts, decisive-frac) + WARN |
| gate deciding on noise | Wilson CI + wide-CI WARN |
| catastrophic forgetting | vs-frozen-seed axis + WARN |
| a hung-but-alive run (GPU at 0%) | **heartbeat + out-of-process [`watchdog.py`](../watchdog.py)** |
| self-play Elo inflation | overlay the external-baseline curve |

---

## Phase 8 — Infra & reproducibility (survive multi-day, unattended)

- **Atomic checkpoint writes** (tmp+rename); **split the buffer/optimizer into ONE rolling `train_state.pt`** so
  `net_iterN.pt` stays weights-only (~42 MB) and pruning keeps `iter0 + last 3` — **bounded at ~1.9 GB regardless of
  iteration or resume count** (the old format hit 74 GB; verified across 3 stop/resume cycles).
- **Resume correctness** (never pair a stale buffer with a newer net), **instance lock** (no two trainers on one
  dir), **crash supervisor** + the out-of-process watchdog, **throughput metrics**, and **gitignore the regenerable
  artifacts** (a `robust_*/` glob — a concurrent `git add -A` must never sweep GB of checkpoints).

---

## Phase 9 — Compute reality & the honest ceiling

Faster iterations raise **iters/hour, not the asymptote.** On one GPU the strength ceiling is compute-bound — that's
a fact, not a bug. The engineering (multiprocess actors, batching, gate flywheel, curriculum) makes the curve climb
*as fast as the hardware allows*; it does not make a laptop a datacenter. The honest deliverable is a *grade*-SOTA
engine + a clearly-stated ceiling, and the one lever that moves the ceiling on one box (a stronger teacher).

---

## Phase 10 — The META-process: how to actually DRIVE the build

The bots above are artifacts; this is the *method* that produces them without lying to yourself:

1. **Bootstrap → refine → gate** (Phase 5) — start strong, refine safely, never regress the playable net.
2. **RWYB (Run-What-You-Build)** — every change is run on real data before you believe it. The watchdog Windows-exe
   bug, the kill-loop, the checkpoint-bloat claim — all caught *by running*, not by reading.
3. **Build the canon, then audit against it** — write the SOTA feature set first ([SELFPLAY_SOTA_FEATURES.md](SELFPLAY_SOTA_FEATURES.md)),
   then adversarially audit your code against it (multiple agents), then **verify every finding against the actual
   code** (agents hallucinate). Result: you find the *real* gaps (we found correctness was clean; the gaps were
   safety-nets + eval-trust).
4. **A/B each strength lever against a *trustworthy* baseline** — build the measurement layer (CIs, forgetting,
   diversity) *first*, so when you add Gumbel / playout-cap / contempt, the deltas are real, not noise.
5. **Correct-as-you-go, commit, don't park** — fix weaknesses the moment you find them; keep the tree clean.

---

## Appendix — Lessons catalog (the bugs that bit us; don't repeat them)

| # | Lesson | Fix |
|---|---|---|
| L1 | **Fixed startpos collapses self-play** (the net "thinks there's one opening") | book/random/mixed opening diversity, in self-play AND teacher games |
| L2 | **Pure self-play degrades an imitation net** | refine with `mix` + anchor-KL + the champion gate; watch the forgetting axis |
| L3 | **Self-play win-rate hides absolute decline** | a fixed external baseline + the vs-frozen-seed forgetting axis |
| L4 | **15-game gate decisions are noise (±0.25)** | Wilson CI on every number + a wide-CI WARN + a larger seed eval |
| L5 | **Checkpoint bloat to 74 GB** (buffer+optimizer in every file) | split into one rolling `train_state.pt`; prune to iter0+last3 |
| L6 | **A hung-but-alive run looks healthy** | heartbeat + out-of-process watchdog |
| L7 | **Watchdog ckpt-dir mismatch → kill-loop a healthy trainer** | watchdog refuses to kill on a *never-appearing* heartbeat (mismatch ≠ hang); heartbeat during the seed-eval warmup |
| L8 | **A NaN silently corrupts the net to uniform policy** | guard the loss before `.backward()` |
| L9 | **Windows: `Popen(relative_exe, cwd=…)` fails; spawn≠fork with CUDA** | resolve the exe to absolute; `mp.get_context("spawn")` |
| L10 | **A future encoding refactor could silently break en-passant / value-sign** | the invariant gate (I1–I12) runs before every train + in CI |

> Provenance: built across the 2026-06-09 chess_zero hardening sessions (commits `5080b07` → `de1aabc`). Every claim
> here is one we ran, not one we assumed.
