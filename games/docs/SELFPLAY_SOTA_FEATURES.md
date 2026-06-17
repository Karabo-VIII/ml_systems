# SOTA Self-Play System — Canonical Feature Checklist + Research Foundation

> Built 2026-06-09 by a 4-researcher fan-out (web-grounded) over the AlphaZero lineage, to define
> *every* feature a SOTA self-play RL game-playing system should have, before we restart `chess_zero`
> training. Companion audit (our-status vs this list): [PRE_RESTART_AUDIT_2026_06_09.md](PRE_RESTART_AUDIT_2026_06_09.md).

## How to read this

- **Tier** = relevance to OUR target (single RTX 4060, 8 GB, AlphaZero-style chess):
  `MUST` (must-have at single-GPU scale) · `GPU-OPT` (must-have *because* compute is scarce — biggest ROI on small HW) ·
  `OPT` (nice-to-have) · `DC` (datacenter-only; skip).
- **Maturity**: `EST` (established in published SOTA) · `EMG` (emerging / recent).
- **Detector** = the gauge/test that reveals the failure when the feature is missing or broken. This column is the
  point of the whole exercise: a SOTA system is one whose *silent* failure modes each have a *non-silent* detector.
- Canonical systems abbreviated: **AZ** AlphaZero (1712.01815), **AGZ** AlphaGo Zero (Nature 2017), **MZ** MuZero
  (1911.08265), **EZ** EfficientZero (2111.00210), **Gumbel** Gumbel MuZero/AZ (Danihelka 2022), **lc0** Leela Chess
  Zero, **KG** KataGo (1902.10565), **AS** AlphaStar (Nature 2019), **ELF** ELF OpenGo (1902.04522), **fishtest**
  Stockfish SPRT framework.

---

## A. Algorithm Core

| # | Feature | Tier | Why / failure-mode if absent | Systems |
|---|---------|------|------------------------------|---------|
| A1 | Pure self-play data generation (net → MCTS targets → train → better net) | MUST | The flywheel itself; without it there is no learning | AZ/AGZ/MZ/lc0/KG |
| A2 | Async/overlapped self-play ↔ training (actor/learner separation) | MUST | Serial alternation idles the GPU during CPU-bound MCTS; 2–3× slower wall-clock | AZ/KG/MZ |
| A3 | FIFO replay buffer, rolling window | MUST | Train-on-one-game → forgetting/instability; unbounded → staleness | AZ/MZ/lc0 |
| A4 | Replay window **sized** (and ideally sublinear growth) | MUST | Too large = stale targets (Elo plateau); too small = variance/forgetting. AZ ~1M positions; KG sublinear; ELF found *small* buffer = big speedup (less staleness) | AZ/KG/ELF |
| A5 | PUCT selection `Q + c·P·√ΣN/(1+N)` | MUST | No prior-guided search → degenerate on large action spaces | AZ/AGZ/MZ |
| A6 | `c_puct` schedule (grows with sims) | OPT | Fixed constant miscalibrates explore/exploit at varying sim budgets | AZ/lc0 |
| A7 | Visit-count policy target `π ∝ N^{1/τ}` | MUST | Training on raw prior defeats search; net never improves | all |
| A8 | Value target = game outcome z ∈ {−1,0,+1} | MUST | No terminal signal → value head → constant 0 | AZ |
| A9 | **Gumbel root selection + sequential halving** | GPU-OPT | At LOW sim counts (the single-GPU reality) PUCT has no policy-improvement guarantee; Gumbel does — biggest search-quality ROI when sims are scarce | Gumbel |
| A10 | Tree reuse across moves (keep played-child subtree) | MUST | Rebuilding the tree every move throws away most of the search budget | lc0 |
| A11 | Batched NN leaf eval | MUST | One-eval-per-leaf pins GPU <5%; games/hr collapses | lc0/all |
| A12 | Virtual loss (if any tree/root parallelism) | MUST* | Parallel sims pick identical paths; parallelism buys no diversity (*only if parallel) | AZ/lc0/ELF |
| A13 | FPU (first-play-urgency reduction on unvisited children) | OPT | Search wastes sims on 0-visit high-prior moves; narrower trees | lc0/KG |
| A14 | Transposition / graph search (DAG) | OPT | Re-evaluates equivalent positions; memory waste (complex; tree-reuse is the simpler 80%) | MC-GraphSearch |
| A15 | N-step bootstrapped / WDL-distribution value target | OPT (MUST for draw-heavy chess: WDL) | Scalar z is high-variance; WDL separates draw from win (needed for contempt) | MZ/EZ; lc0/KG (WDL) |
| A16 | Reanalyze (re-run MCTS on old trajectories with latest net) | GPU-OPT | When self-play is the bottleneck, stale targets dominate; reanalyze refreshes them | MZ-Reanalyze/EZ |
| A17 | Model-based learned dynamics (MuZero) | DC / N-A | Only needed when rules are unknown; chess rules are known | MZ/EZ |

## B. Exploration & Game Diversity

| # | Feature | Tier | Why / **Detector** | Systems |
|---|---------|------|--------------------|---------|
| B1 | Dirichlet root noise (chess α≈0.3, ε≈0.25) | MUST | Forces minimal root exploration. **Detector:** root move-1 entropy → ~0 bits = collapse | AZ/lc0/KG |
| B2 | Temperature schedule (τ=1 first ~30 ply, then →0) | MUST | Diversity early, strong targets late. **Detector:** draw% >80% at move 10 = τ too low too early | AZ/lc0/KG |
| B3 | **Opening diversity / varied start positions** | MUST | Fixed startpos + peaked net → self-play monoculture; value head never sees diverse decisive lines (the failure we hit). **Detector:** distinct-starts/games, first-move entropy <1 bit | AZ(20-ply sampled)/lc0(books, FRC)/KG |
| B4 | **Playout cap randomization** (p≈0.25 full-N, else fast-n) | GPU-OPT | Decouples value-sample rate from policy quality; ~1.37× ROI. **Detector:** value loss stalls while policy improves (asymmetric heads) | KG |
| B5 | Forced playouts + policy-target pruning (k=2) | GPU-OPT | Stops premature pruning of low-prior good moves; prunes the forced visits back out of the target. **Detector:** visit-fraction tracks prior with no correction | KG |
| B6 | Resignation threshold + false-positive playthrough fraction (~10–20% played out) | MUST | Aggressive resign starves endgame data → self-fulfilling "lost". **Detector:** %games by resign vs mate; KR-vs-K tablebase value <0.5 | AZ/AGZ/KG |
| B7 | Diverse opponent pool / checkpoint sampling (PFSP-lite) | OPT (lite: MUST) | Training only vs latest-self → non-transitive cycling / forgetting. **Detector:** winrate vs frozen old checkpoint <60% | AS/PSRO/NFSP |
| B8 | Draw avoidance / contempt / search-contempt (N_scl≈5) | GPU-OPT (chess) | Chess self-play over-draws → sterile signal. **Detector:** (W+L)/D < 0.3; >60% of value outputs within ±0.05 of 0 | search-contempt(2025)/Stockfish |

## C. Network Architecture & Heads

| # | Feature | Tier | Why / failure-mode | Systems |
|---|---------|------|--------------------|---------|
| C1 | Residual tower w/ skip connections | MUST | Deep nets without skips don't train; shallow lacks capacity (+600 Elo in AGZ ablation) | AZ/AGZ/lc0/MZ |
| C2 | Squeeze-Excite **or** global-pooling bias | GPU-OPT | Injects board-wide context into local convs; ~free Elo (~1.6× in KG ablation) | lc0(SE)/KG(global pool) |
| C3 | BatchNorm (folded) / batch-renorm | MUST | Deep ResNet won't train stably without normalization | AZ/AGZ/lc0 |
| C4 | Full input planes: piece planes + **history (T≈8)** + side-to-move + castling + **repetition** + 50-move counter | MUST | Missing history/repetition planes → repetition-draw decisions become random; net can't see the rule state | AZ(119)/lc0(112) |
| C5 | Side-relative board flip (always from mover's view) | MUST | Without it the net learns W and B from scratch separately; doubles difficulty | AZ/AGZ/lc0 |
| C6 | Spatial policy head (8×8×73 = 4672) incl. underpromotions | MUST | Flat head loses spatial priors; underpromotion encoding bugs are common | AZ/lc0 |
| C7 | **WDL 3-way value head** | MUST (chess) | Scalar conflates win/draw; can't do contempt or calibrate draw-heavy positions | lc0/KG |
| C8 | Moves-left head (MLH) | OPT (prod MUST) | Prefer shorter wins; avoid aimless shuffling in won positions | lc0 |
| C9 | Auxiliary heads (opponent-policy, short-horizon value, ownership/score for Go) | OPT | Denser signal → sample efficiency (+30–60 Elo in KG); ownership/score are Go-specific | KG/lc0-BT |
| C10 | L2 / weight decay (c≈1e-4) | MUST | BN can't bound weight scale → unbounded growth → late collapse | AZ/lc0/KG |
| C11 | **No** symmetry augmentation for chess (only Go is 8-fold symmetric) | MUST | Rotating/reflecting a chess board makes inputs inconsistent with move targets → corrupts training | AZ (explicit) |
| C12 | Transformer / attention backbone (+ smolgen) | OPT/EMG | Long-range piece interactions; lc0 BT4 +270 policy Elo — but larger/slower per node | lc0-BT4 |

## D. Training Mechanics & Optimization

| # | Feature | Tier | Why / failure-mode | Systems |
|---|---------|------|--------------------|---------|
| D1 | **SGD + (Nesterov) momentum 0.9**, not Adam | MUST | Adam chases non-stationary MCTS-target noise → instability / value divergence | AZ/KG/lc0/MZ |
| D2 | LR schedule: warmup → piecewise step drops | MUST | Flat LR plateaus early; full LR from step 0 on a random net → early loss spikes | AZ/KG/lc0 |
| D3 | Reduced LR for first N samples (early stabilization) | GPU-OPT | Weakest net sees highest-variance targets; KG uses 3× lower LR for first 5M samples | KG |
| D4 | Batch size scaled to HW (AZ 4096 → 512–1024 on 4060) + LR co-scaled | MUST | Too small = gradient variance; too large = OOM; must co-scale LR | AZ/MZ/KG |
| D5 | Gradient clipping (global norm) | MUST | One outlier batch corrupts weights for many steps; tighter (1–5) under fp16 | lc0 |
| D6 | **Mixed precision (bf16 preferred on Ampere/4060)** | GPU-OPT | 2–3× throughput + half VRAM. bf16 keeps fp32 exponent range → no overflow-NaN | lc0/AMP |
| D7 | Dynamic loss scaling + overflow-skip (if fp16) | MUST (fp16) | fp16 underflow → silent zero-gradient updates; overflow → NaN. **Detector:** scaler scale <1, skip-ratio >5% | GradScaler/lc0 |
| D8 | Loss-component weights tuned (policy/value/aux); **value_loss_weight ≤ ~0.5** | MUST | Value head (large early MSE) dominates → policy underfits / value overfits (lc0 dropped to 0.25) | lc0/KG |
| D9 | Replay ratio (grad-samples per new sample) in band ~1–4, monitored | MUST | Too high = overfit stale data; too low = GPU starved. On 1 GPU it stays low naturally — but monitor | AZ/MZ/ELF |
| D10 | Stochastic Weight Averaging / weight EMA | OPT | Flatter minima, lower Elo variance, fewer noisy gate rejections | KG/lc0 |
| D11 | Training-step / cadence defined in **gradient steps** (not wall-clock) | MUST | Undefined cadence → non-reproducible, non-comparable Elo curves | KG/lc0 |

## E. Training Pathologies & Their Detectors (the silent-failure core)

| # | Pathology | Mitigation | **Detector** (how it stops being silent) |
|---|-----------|-----------|-------------------------------------------|
| E1 | Catastrophic forgetting | Historical-checkpoint replay (~5–10% games) + adequate buffer | Winrate vs a 100–200k-step-old checkpoint *drops* |
| E2 | Non-transitivity / strategy cycling | Checkpoint pool / PFSP (full PSRO=DC) | Round-robin Elo among 10 checkpoints is non-monotone; pairwise matrix unsortable |
| E3 | Value-head overfitting / memorization | value_loss_weight↓, ≤30 positions/game, aux short-horizon heads | Value loss ↓ while playing strength flat/↓; correlation to a fixed eval set drops |
| E4 | Policy/entropy collapse onto one move | Dirichlet + temperature + FPU; (modified PUCT) | Root entropy <1 bit in typical middlegames; first-move Gini→1 |
| E5 | Draw collapse / over-drawing | Search-contempt / contempt δ; opening imbalance | (W+L)/D < 0.3; median self-play game length *rising* over training |
| E6 | **Self-play distribution shift / drift** | **Fixed external baseline eval** + bounded staleness + gate + refine-from-champion | External-baseline Elo ↓ while self-play winrate stays ~50% — invisible to self-play alone |
| E7 | Replay staleness / window pathologies | FIFO + sparse per-game sampling + generation tags | >30% of batch is >5 generations old; value-target inconsistency |
| E8 | Over-resignation / endgame blindspot | Calibrated playthrough fraction; soft visit-cap finish; min game length | >85% games end by resign; KR-vs-K tablebase positions valued <0.5 |
| E9 | Mode collapse onto one opening | Opening diversity (B1–B3) | First-move entropy <1 bit; >80% games share a 3-ply prefix |
| E10 | Reward/return sparsity | MCTS value bootstrap (core) + aux short-horizon value | Value loss >0.4 after 50k games while policy sharpens |
| E11 | Gating failure (promote a regressed net) | Best-of-N gate (≥55%, or KG 100/200) + SWA candidate | Sawtooth external Elo; self-play winrate pinned 50% while external oscillates |
| E12 | Multi-head training instability | Per-head loss weights + reduced initial LR + grad clip | Per-head loss diverge; grad-norm spikes 10× |

## F. Evaluation & Strength Measurement

| # | Feature | Tier | Why / **Detector** | Systems |
|---|---------|------|--------------------|---------|
| F1 | Draw-aware score `(W+0.5D)/G` | MUST | Raw win% biased when draw rates shift. Log W/D/L separately | fishtest/BayesElo |
| F2 | Colour-balanced/paired play (alternate colours) | MUST | White ≈ +32 Elo edge; imbalance fakes +15–30 Elo on short matches. **Detector:** assert |white_games_A − white_games_B| ≤ 1 | fishtest |
| F3 | SPRT (sequential, α=β=0.05) **or** confidence intervals | MUST | Fixed-N is noisy: a 200-game eval has ±40–50 Elo CI — meaningless for small gains. **Detector:** require LLR to cross a bound, or report ±CI | fishtest |
| F4 | Pentanomial (paired-game) scoring | OPT | ~15% variance reduction; faster SPRT | fishtest |
| F5 | BayesElo/Ordo + **error bars**, never raw win% headline | MUST | Elo from different draw regimes is incomparable; CI>30 Elo = untrustworthy | CCRL/KG |
| F6 | **Fixed external baseline opponent** (not just self-play winrate) | MUST | Co-adapting agents both degrade while mutual winrate stays 50% (one study: 51.9pp generalization drop invisible in self-play). The single most important anti-silent-failure eval | lc0/AGZ/KG |
| F7 | Gating eval before promotion (best-of-N, reduced sims, no noise) | MUST | A noisy 48%-net promoted poisons future data → regression cycle | AGZ/KG/ELF |
| F8 | Opening-balanced eval suite (same opening both sides) | MUST | All-from-startpos eval → correlated, draw-inflated, hides strategic gaps | fishtest |
| F9 | Deterministic eval mode (τ=0, no Dirichlet) | MUST | Stochastic eval → ±10–20 Elo run-to-run, non-reproducible | AGZ/KG |
| F10 | Minimum game count / never headline Elo from <~200–400 games | MUST | +15 Elo on 50 games is noise 60% of the time | Henderson/fishtest |
| F11 | Evaluate vs a **pool** of past checkpoints (not only latest opponent) | MUST | Forgetting is invisible if you only test vs N−1 | AGZ/lc0 |

## G. Reproducibility & Systems Infrastructure

| # | Feature | Tier | Why / failure-mode | Systems |
|---|---------|------|--------------------|---------|
| G1 | Full RNG seeding (python/numpy/torch/cuda) | MUST | Unseeded runs diverge from step 0; results unreproducible | PyTorch best practice |
| G2 | Checkpoint contents = weights + optimizer + scheduler + step + RNG | MUST | Resume without optimizer/momentum → loss spike at every resume | KG/lc0/MZ |
| G3 | **Atomic checkpoint write (tmp + os.replace)** | MUST | Crash mid-write → corrupt-but-correctly-named file → unrecoverable resume | KG |
| G4 | Checkpoint rotation / retention | MUST | Unbounded checkpoints fill the disk → silent training stall | lc0/KG |
| G5 | Replay buffer **separated** from net checkpoint | MUST | Bundling bloats every file; pairs a stale buffer with a new net on resume | MZ/KG |
| G6 | Resume correctness: never pair stale buffer w/ newer net (reanalyze or flush) | MUST | Off-policy targets diverge policy silently (Elo dips post-resume) | MZ |
| G7 | Actor/learner separation pattern (SEED-RL-style for 1 GPU: CPU actors + central batched inference) | MUST | Serial self-play idles GPU; one-eval-per-call pins GPU ~5% | Ape-X/IMPALA/SEED |
| G8 | Weight broadcast to actors (poll newest model file) | MUST | Stale actors raise effective replay-lag; value loss won't drop | MZ/KG |
| G9 | Multiprocess CPU self-play actors (spawn, not fork w/ CUDA) | MUST | MCTS+move-gen is CPU-bound; the #1 throughput lever on 1 GPU | KG/ELF |
| G10 | Inference server / central batched eval for actors | GPU-OPT | Batches actor leaf-evals → near-peak GPU; else GPU idle | KG/ELF/SEED |
| G11 | **Instance lock** (no 2 trainers on one dir) | MUST | Two trainers race on checkpoints/curve/buffer → corruption / split-brain | KG-practice |
| G12 | Crash-recovery / auto-restart supervisor | MUST | Multi-day runs WILL hit transients; a death at hr40 loses a day unattended | KG/supervisord |
| G13 | Disk/bloat management | MUST | Self-play + checkpoints + shuffles grow unbounded → full disk → stall | KG/lc0 |
| G14 | Throughput metrics (games/hr, steps/s, GPU-util) | MUST | A 2× throughput regression is otherwise invisible until "feels slow" days later | KG/ELF/SEED |

## H. Observability, Monitoring & Numerical Safety

| # | Feature | Tier | **Detector / action** | Systems |
|---|---------|------|------------------------|---------|
| H1 | **NaN/Inf guard on loss before `.backward()`** | MUST | `assert not isnan/isinf(loss)`; on hit skip batch + log + LR↓ + counter; abort if >5/100. A single NaN → uniform-policy net, silently | GradScaler/prod RL |
| H2 | Loss scaling + overflow detection (fp16) | MUST(fp16) | Monitor `scaler.get_scale()`; skip-ratio >5% = instability | AMP/lc0 |
| H3 | Gradient-norm monitor (pre- and post-clip) | MUST | Inf pre-clip → clipped to 0 = silent no-op batch. Track EMA; 10× spike predicts collapse | Henderson/lc0 |
| H4 | Activation-magnitude monitor (unbounded growth) | MUST | `act.abs().max()` on final residual; >3× baseline precedes NaN by many steps | (2506.15544) |
| H5 | Dormant/dead-neuron fraction | OPT/EMG | Non-stationary RL → loss of plasticity; dormant>20% = capacity gone, no loss signal | Sokar 2302.12902 |
| H6 | Per-iter structured log: policy/value loss, **policy entropy**, value MSE, draw rate, LR, grad-norm, NaN-count | MUST | The composite early-warning; without it a broken run burns the whole budget undiagnosed | lc0/KG |
| H7 | **Self-play diversity gauges: distinct openings, (W+L)/D decisiveness, root entropy** | MUST | The exact instrument that catches collapse (we just added distinct-starts + decisive-frac) | AZ-diversity/KG |
| H8 | Heartbeat / liveness for unattended runs | MUST | A deadlocked-but-alive process runs GPU at 0% for hours "successfully". Watchdog on heartbeat + GPU-util | MLOps |
| H9 | Strength-curve artifact detection (self-play Elo inflation vs real) | MUST | Overlay external-baseline Elo on self-play Elo; divergence = inflation | lc0 (explicit) |
| H10 | Checkpoint integrity (reload + NaN-scan + checksum after save) | MUST | A silently-corrupt checkpoint poisons all post-resume training | prod practice |
| H11 | "Advertises-live-but-dead" detection (PID reuse) | MUST | A stale PID file makes a supervisor believe a dead run is alive (we hit this) | MLOps |

## I. Correctness Invariants & Anti-Leakage (presumed-broken-until-tested)

| # | Invariant | **Detector / unit test** | Severity if wrong |
|---|-----------|---------------------------|-------------------|
| I1 | **Value sign negated per ply in MCTS backup** (side-to-move relative) | Position 1 ply from mate → parent Q strongly negative; assert `value=-value` each backup level | CATASTROPHIC — Q→0, 50% everywhere, value loss stuck ~0.25 |
| I2 | **Training value target z stored from each position's mover's perspective** | 3-move forced mate: z at mating pos=+1, one before=−1 | CATASTROPHIC — half the value labels inverted |
| I3 | Policy↔move-index bijection, no collisions (esp. underpromotions) | Round-trip `idx(move(idx))`==idx over 1000 positions; no 2 legal moves share an index | CRITICAL — silent move-encoding defects |
| I4 | Illegal-move mask applied to **logits** then renormalize (not after softmax) | 1-legal-move position → that move prob=1.0 after mask | CRITICAL — prob bleeds to illegal moves |
| I5 | Board push/pop exception-safe (try/finally) | Zobrist hash of board == pre-MCTS hash after the sim loop | HIGH — corrupt board poisons all later sims |
| I6 | Terminal value correctness (mate=−1 STM, stalemate/3-fold/50-move/insufficient=0) | Minimal test position per condition | CATASTROPHIC — e.g. stalemate scored as mate teaches play-into-stalemate |
| I7 | Repetition via state hash incl. castling/EP (not FEN string) | Pre/post-castling positions hash differently; 3-fold detected on 3rd occurrence | HIGH — false draws in won positions |
| I8 | **Search-vs-played-move consistency**: store full π (visit dist), played move sampled from the *same* search | `sum(π)≈1`, `π.shape==(4672,)`, max(π)<1 for multi-legal positions (else a one-hot was stored) | CRITICAL — one-hot target wastes ~30× sample efficiency; a second re-search disagrees with the labelled π |
| I9 | Eval-set / replay-buffer leakage prevention | `eval_fens ∩ buffer_fens == ∅` | MEDIUM — inflated eval vs real strength |
| I10 | No future-information leakage in input features (only s_{t−N..t}) | Feature fn for s_t never reads s_{t+1..T}; z appended post-game | HIGH — value "knows" the result (G-AUDIT-011 class) |
| I11 | Bit-exact reproducible replay (seed + determinism flags) | Two same-seed runs → identical checkpoints to iter 10 | MEDIUM — "flaky" Elo, undebuggable |
| I12 | Game-length cap + correct adjudication (never-hang) | assert game_len ≤ MAX; alert on >300-move games | HIGH — one buggy game hangs a worker forever |
| I13 | **Pre-training invariant gate** running I1–I12 in <60s, exit 2 = halt | `run_invariants_check.py` before every training start | The meta-control: turns all of the above into a mechanical pre-flight |

---

## Single-GPU priority stack (what actually moves the needle on a 4060)

1. **Correctness first (I1, I2, I6, I8)** — a sign bug or one-hot-target silently wastes the entire run.
2. **Numerical safety (H1, D7/H2, H3)** — NaN/underflow silently corrupts weights / zeroes gradients.
3. **Anti-silent-failure eval (F6, F3/F5, F9)** — a *fixed external baseline* + CIs/SPRT; self-play Elo lies.
4. **Observability (H6, H7, H8, H9)** — log the composite + diversity gauges + heartbeat; a broken run must scream.
5. **Search ROI (A9 Gumbel, B4 playout-cap-randomization, B5 forced-playouts)** — the highest strength-per-sim levers when compute is scarce.
6. **Diversity & anti-pathology (B3, B6, E5 contempt, E6 drift control)** — keep self-play informative.
7. **Heads & signal (C7 WDL, D8 value_loss_weight, C9/E10 aux short-horizon value)** — richer, better-calibrated targets.
8. **Infra hygiene (G3 atomic, G5/G6 buffer-split+resume, G11 lock, G12 supervisor, G14 throughput)** — survive multi-day unattended.

## Datacenter-only (consciously skip at our scale)
40-block / large-transformer backbones · thousands-of-actor farms · full PSRO/AlphaStar league · MuZero learned-dynamics (rules are known) · Go-specific ownership/score heads.

---

## Research foundation (consolidated sources)

**Core algorithms:** AlphaZero [1712.01815](https://arxiv.org/abs/1712.01815) · AlphaGo Zero [Nature 2017](https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf) · MuZero [1911.08265](https://arxiv.org/abs/1911.08265) · EfficientZero [2111.00210](https://arxiv.org/abs/2111.00210) · Gumbel MuZero/AZ [ICLR 2022](https://openreview.net/forum?id=bERaNdoegnO) · Sampled MuZero [2104.06303](https://arxiv.org/abs/2104.06303).
**Search / diversity:** MC Graph Search [2012.11045](https://arxiv.org/abs/2012.11045) · lc0 PUCT/opening-diversity [issue #913](https://github.com/LeelaChessZero/lc0/issues/913) · Search-contempt [2504.07757](https://arxiv.org/html/2504.07757v1) · Creative Chess w/ AZ (diversity) [2308.09175](https://arxiv.org/abs/2308.09175) · asymmetric draw rules [2604.03683](https://arxiv.org/pdf/2604.03683).
**KataGo techniques (playout-cap-rand, forced-playouts, global-pooling, aux heads):** [1902.10565](https://ar5iv.labs.arxiv.org/html/1902.10565) · [KataGoMethods.md](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md) · [SelfplayTraining.md](https://github.com/lightvector/KataGo/blob/master/SelfplayTraining.md).
**lc0 (WDL, MLH, transformer/BT4, training):** [tech explanation](https://lczero.org/dev/wiki/technical-explanation-of-leela-chess-zero/) · [BT4/transformer](https://lczero.org/blog/2024/02/transformer-progress/) · [WDL-rescale/contempt](https://lczero.org/blog/2023/07/the-lc0-v0.30.0-wdl-rescale/contempt-implementation/) · [training](https://lczero.org/blog/2018/10/lc0-training/).
**Distributed / infra:** Ape-X [1803.00933](https://arxiv.org/abs/1803.00933) · IMPALA [1802.01561](https://arxiv.org/abs/1802.01561) · SEED RL [1910.06591](https://ar5iv.labs.arxiv.org/html/1910.06591) · ELF OpenGo [1902.04522](https://ar5iv.labs.arxiv.org/html/1902.04522).
**Population / non-transitivity:** AlphaStar [Nature 2019](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf) · PSRO [1711.00832](https://arxiv.org/abs/1711.00832) · NFSP [1603.01121](https://arxiv.org/abs/1603.01121) · JiangJun (non-transitivity) [2308.04719](https://arxiv.org/abs/2308.04719) · PBT-for-exploitability [2208.05083](https://arxiv.org/abs/2208.05083).
**Evaluation / testing:** fishtest SPRT [chessprogramming](https://www.chessprogramming.org/Sequential_Probability_Ratio_Test) · [Fishtest math](https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-Mathematics.html) · BayesElo [Coulom](https://www.remi-coulom.fr/Bayesian-Elo/) · Match statistics [chessprogramming](https://www.chessprogramming.org/Match_Statistics).
**Stability / monitoring / reproducibility:** Deep RL that Matters [1709.06560](https://arxiv.org/abs/1709.06560) · Stable Gradients at Scale [2506.15544](https://arxiv.org/html/2506.15544v1) · Dormant Neuron Phenomenon [2302.12902](https://arxiv.org/pdf/2302.12902) · Plasticity-loss survey [2411.04832](https://arxiv.org/pdf/2411.04832) · Mixed-precision [1710.03740](https://arxiv.org/pdf/1710.03740) · PyTorch AMP [docs](https://docs.pytorch.org/docs/stable/amp).
**Correctness references:** lc0 AZ primer [search/alphazero](https://lczero.org/dev/lc0/search/alphazero/) · JoshVarty AZ+MCTS [tutorial](https://joshvarty.github.io/AlphaZero/) · "Improving the Training Target" [Oracle/Abrams](https://medium.com/oracledevs/lessons-from-alphazero-part-4-improving-the-training-target-6efba2e71628) · Repetitions/Zobrist [chessprogramming](https://www.chessprogramming.org/Repetitions) · Reproducing AZ on Tablut [2604.05476](https://arxiv.org/html/2604.05476v1).
