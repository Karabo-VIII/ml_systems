# chess_zero — Pre-Restart Weakness Audit (2026-06-09)

> Method: 4 web-grounded researchers built the canonical SOTA feature set
> ([SELFPLAY_SOTA_FEATURES.md](SELFPLAY_SOTA_FEATURES.md)); 2 Opus auditors + 1 validator + 1 trainer-expert
> audited our code against it; **every load-bearing claim was then re-verified against the actual source by the
> overseer** (grep/read + the Opus auditor's empirical probe). Citations are `file:line` in `projects/chess_zero/az/`.

## Headline verdict

- **Correctness is CLEAN (the catastrophic class).** All the silent-killer invariants — value-sign negation per ply
  in MCTS backup (I1), training-`z` from the mover's perspective (I2), terminal values (I6), policy↔index bijection
  incl. underpromotions (I3), illegal-move masking on logits (I4), push/pop safety (I5, via copy-per-sim), no
  future-leakage (I10), search-vs-played-move consistency (I8) — **HOLD in the production path**, confirmed by an
  empirical probe (mate-in-1 → root Q correct; 800-position bijection walk → 0 collisions; etc.). **We are not
  carrying a catastrophic training bug into the restart.**
- **The real exposure is SILENT-FAILURE SAFETY NETS + EVAL TRUST, not bugs.** The long-running production loop
  `train_robust.py` has **no NaN/Inf guard, no gradient clipping, and no mixed precision** — even though the team's own
  `bootstrap_supervised.py` already uses AMP. The champion gate makes promote/reject decisions on **15-game** evals
  (±0.25 win-rate CI). And several high-ROL SOTA strength features are absent (WDL head, Gumbel, playout-cap-rand, SE).
- One **demo-only** real defect: `selfplay.py:88-92` double-searches (stored π ≠ played move). **Not on the restart
  path** (`play.py learn` → `train_robust` → `selfplay_pool` → batched/guarded generators, all single-search). Restart
  via `train_robust`, never `train_demo`.

---

## STATUS (updated 2026-06-09 — Tier 1 LANDED + RWYB-verified)

The entire silent-failure safety + measurement layer is **DONE and committed** (the "no bugs / no
silent failures before restart" mandate):

| Item | Status | Commit |
|------|--------|--------|
| S1 NaN/Inf loss guard (skip+count+abort) | ✅ DONE | `5080b07` |
| S2 gradient clipping + grad_norm/nan_skipped telemetry | ✅ DONE | `5080b07` |
| S3 value_loss_weight=0.5 | ✅ DONE | `5080b07` |
| S4 pre-training invariant gate (run_invariants_check.py, exit 2 + CI) | ✅ DONE | `5080b07` |
| S5a Wilson CI on every eval + wide-CI WARN + larger seed eval (floor-lock fix) | ✅ DONE | `5080b07` |
| S5b forgetting/drift detector (current net vs frozen seed + CI) | ✅ DONE | `2d0c8b9` |
| S6 heartbeat + out-of-process watchdog (catches HANGS) | ✅ DONE | `0b44530` |

RWYB: invariant gate 8/8, test gate 6/6, every path exercised. Correctness was already CLEAN; these
add the safety nets + make every failure mode (NaN, gradient blow-up, opening collapse, gate-noise,
forgetting, hang) **loud instead of silent**. The one remaining S5 sub-item — a *strong absolute*
external baseline (Stockfish/higher-depth) beyond random + classical-d1 + frozen-seed — is OPTIONAL
(the frozen-seed axis already makes drift/forgetting visible).

**Remaining (next phase, NOT a silent-failure fix):** T1 mixed-precision (throughput), and the
Tier-3 **search levers** (Gumbel / playout-cap-randomization / tree-reuse / search-contempt). These
are *algorithmic strength* changes to load-bearing production search — each needs a real A/B run on
the now-trustworthy metrics to validate (a compute decision tied to the restart), not a blind ship.

## TIER 1 — Fix BEFORE restart (silent-failure safety; low-risk, reversible) — ✅ ALL LANDED (see Status above)

| # | Gap | Severity | Evidence (verified) | Fix |
|---|-----|----------|---------------------|-----|
| S1 | **No NaN/Inf guard on loss before `.backward()`** | SILENT/CATASTROPHIC | `train_robust.py` grep `isnan/isfinite` = 0 hits; `selfplay.py:137` `loss=policy_loss+value_loss`→`backward()`; loop `train_robust.py` catches only OOM | Guard in `train_step` + `train_step_anchored`: if `not torch.isfinite(loss)` → skip step, count, abort if >N/window; log `nan_count` |
| S2 | **No gradient clipping + no grad-norm monitor** | SILENT/HIGH | grep `clip_grad` = 0 hits | `clip_grad_norm_(net.parameters(), 5.0)` between backward/step; return + log `grad_norm` (an Inf pre-clip → 0 grad = silent no-op; EMA spike predicts collapse) |
| S3 | **`value_loss_weight = 1.0`** (equal) — value MSE dominates early | SILENT/MED (drives E3 value-overfit) | `selfplay.py:137`, `train_robust.py:1062` `loss=policy_loss+value_loss` | Add `cfg.value_loss_weight` (0.5); `loss = policy_loss + w*value_loss` (lc0 dropped to 0.25) |
| S4 | **No pre-training invariant gate** (I13) | the meta-control | no `run_invariants_check.py` exists | Add `run_invariants_check.py` asserting I1–I12 in <60s, `exit 2`=halt; run at launch. Turns every correctness invariant into a mechanical pre-flight so a future refactor can't regress silently |
| S5 | **Eval trust: 15-game gate decisions / no CI** (F3/F5/F10) | SILENT/HIGH | `eval_games=30` → `half=15`; 95% Wilson CI at p=0.5 ≈ ±0.25; no CI logged anywhere | Log Wilson CI per eval block + WARN when too wide; raise gate/seed eval N; **fix the seed-eval floor-lock** (seed champion from a *larger* eval so a lucky 15-game read doesn't lock the monotonic floor too high → permanent REJECT) |
| S6 | **In-process supervisor + no heartbeat** (H8/G12) | SILENT/HIGH (unattended) | `supervise()` is a `while True` catching exceptions only; a deadlock/segfault/SIGKILL kills the supervisor too; GPU 0% "successfully" | Heartbeat file each iter + an out-of-process watchdog/launcher that restarts on stale heartbeat |

## TIER 2 — Throughput / robustness (before restart or early)

| # | Gap | Evidence | Fix | ROI |
|---|-----|----------|-----|-----|
| T1 | **No mixed precision in the prod loop** (D6) | AMP present in `bootstrap_supervised.py:309/323/347`, absent in `train_robust.py` | Wrap forward+loss in `torch.autocast("cuda", bfloat16)` (bf16 = no overflow-NaN, no scaler needed); persist scaler/scheduler in ckpt | 2–3× throughput + ½ VRAM on the 4060 |
| T2 | Self-play pool silently returns < n_games | `selfplay_pool.py` `if res: games.extend(res)` then `games[:n_games]` | WARN/raise when `len(games) < n_games` | closes a silent under-generation hole |
| T3 | Throughput metric reads 0 if stdout not in train.log | `learning_report.py:41` regexes `train.log` | read timing from the curve's `timing_s` field instead | makes games/hr trustworthy |
| T4 | RNG not fully seeded for bit-exactness | no `cuda.manual_seed_all`, no cudnn determinism flags | add for debug runs (perf cost in prod) | reproducible debugging |

## TIER 3 — SOTA strength features (roadmap; mostly your call — some need decisions)

Ranked by ROI-on-a-4060. **Verified absent** unless noted. Several interact with the existing scalar-value bootstrap
checkpoint, so they're staged for your direction rather than auto-applied.

| # | Feature | Tier | Evidence | Effort | Note / decision |
|---|---------|------|----------|--------|-----------------|
| R1 | **WDL 3-way value head** (A15/C7) | MUST(chess) | `net.py:71` scalar `tanh` | M | Prereq for contempt; **changes checkpoint schema → the current scalar bootstrap won't load into a WDL net** (needs a fresh bootstrap or a head graft). Decision needed |
| R2 | **Gumbel root selection + sequential halving** (A9) | GPU-OPT | `mcts.py` PUCT only | M | Biggest search-quality ROI at our low (~64) sim budget; policy-improvement guarantee PUCT lacks |
| R3 | **Playout-cap randomization** (B4) + **forced playouts + policy-target pruning** (B5) | GPU-OPT | absent | M | ~1.37× sample ROI (KataGo); decouples value-sample rate from policy quality |
| R4 | **SE / global-pooling bias block** (C2) | GPU-OPT | `net.py:30-43` plain ResBlock | S | ~free Elo (~1.6× KG ablation); also schema change |
| R5 | **Tree reuse across moves** (A10) | MUST | `mcts.py` fresh root each move | M | Stops throwing away the search budget every move |
| R6 | **Search-contempt / draw penalty** (B8/E5) | GPU-OPT(chess) | absent | M | Chess over-draws; needs WDL (R1) to calibrate |
| R7 | **Stronger fixed external baseline** (F6/E6) + **past-checkpoint eval** (F11/E1/E2) | MUST | only vs random + classical-d1 (weak ceiling); no historical-self eval | S–M | Stockfish at fixed movetime as the permanent anchor + eval vs `net_iter0` every N iters → makes drift/forgetting non-silent. **High value, low risk — strong candidate to also do before restart** |
| R8 | **Input history depth** (C4) | MUST | `encoding.py:32` `HISTORY_STEPS=1`, `N_INPUT_PLANES=19` (paper: T=8/119) | M | Net can't see piece history / full repetition context (terminal draw detection still correct via python-chess). Schema change |
| R9 | **Reanalyze** (A16) | GPU-OPT | absent | M | Refresh stale targets when self-play is the bottleneck |
| R10 | **Optimizer: Adam → SGD+Nesterov + LR schedule/warmup** (D1/D2) | MUST | `train_robust.py` Adam, flat `lr=1e-3` | M | SOTA uses SGD+momentum. **Caveat:** Adam partly *compensates* for the missing LR schedule — switch the two together, with tuning, not piecemeal. Decision needed |
| R11 | Moves-left head (C8), aux short-horizon value head (C9), checkpoint-pool PFSP-lite (B7/E2), resignation+playthrough (B6) | OPT | absent | M each | Secondary; revisit after R1–R7 |

## What is already SOTA-solid (do NOT re-touch)

Champion gate (monotonic floor + draw-aware climb tie-break, logic verified correct) · curriculum (moving teacher) ·
anchor-KL · teacher distillation · multiprocess CPU actors (13.5× verified) · GPU leaf-batching · **opening diversity
(book/random/mixed) + the diversity gauge (distinct-starts/decisive-frac) — added 2026-06-09** · Dirichlet noise ·
atomic checkpoint writes · buffer/net split + stale-buffer resume guard · instance lock + PID-alive check ·
deterministic greedy eval · checkpoint rotation. The **infra layer is the strongest part of the codebase.**

## Recommended restart sequence

1. Land **Tier 1 (S1–S5)** + **R7** (fixed strong baseline + past-checkpoint eval) — all low-risk, all turn a silent
   failure into a loud one. (S6/T1 next.)
2. Run `run_invariants_check.py` (S4) — must exit 0.
3. Decide the **schema-changing strength batch (R1 WDL + R4 SE + R8 history)** as ONE re-bootstrap, since each
   invalidates the current scalar bootstrap checkpoint. Either commit to a fresh stronger bootstrap or keep the
   current net and add only the non-schema levers (R2 Gumbel, R3 playout-cap, R5 tree-reuse, R6 contempt).
4. Restart via `train_robust` (never `train_demo`), watch the diversity gauge + the new grad-norm/nan-count + the
   external-baseline curve.
