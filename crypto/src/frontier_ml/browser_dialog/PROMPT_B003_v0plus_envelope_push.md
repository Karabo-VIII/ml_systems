# B003 — V0+ envelope push: lit review + first-principles novel ideas to lift V1.x past IC 0.067 / ShIC 0.037

> **Status:** OPEN  •  **Sent:** 2026-05-02
> **For:** an `@browser`-routed Claude Code session with WebSearch / WebFetch.
> **From:** the trained V1.x family in `src/wm/v1/v{0,1_0,1_1,1_4,1_6}/`.
> **Tone:** /un — direct, ship-or-concede, no polite hedging.

## Mission framing

The V1.x family (V1.0 / V1.1 / V1.4 / V1.6, all 2M params) is the project's
**proven SHIP-tier alpha**. Current envelope:

| Metric | V1.0 | V1.1 | V1.4 | V1.6 | Family record |
|---|---|---|---|---|---|
| IC (h=1) | 0.066 | 0.067 | 0.068 | 0.062 | **+0.068** (V1.4) |
| ShIC (h=1) | 0.032 | 0.033 | 0.031 | 0.033 | **+0.037** (recent) |
| Avg IC across family | ~0.05 |  |  |  | (cohort baseline) |
| Avg ShIC | ~0.030 |  |  |  | |

User's framing (verbatim): *"changes you applied to V1 have resulted in
IC 0.037, the highest I've seen. Even though IC is 0.05 or so across the
board. If they apply a literature review layer on top of that, as well as
look at novel ideas from first principles, we can get somewhere."*

**The number 0.037 in the user's note refers to the family's recent ShIC
record** (anti-fragility metric — IC on shuffled labels). The IC ~0.05
average is the family's central tendency on raw IC. We want both lifted.

The V1.x family is on **dollar bars** at chimera_legacy resolution, with
f34 features (33 norm_* + 1 xd_* cross-asset). Architecture: 2M-param
Transformer + RSSM with multi-horizon TwoHot return prediction at
h={1, 4, 16, 64}. Pattern P + Q bundle (5 dead features dropped, recon-
loss-clamp at 0.5) is the most recent improvement.

The V1.x family has been incrementally tuned; we believe most "obvious"
upgrades (focal loss, label smoothing, larger batch, more layers, LR
schedules) have been tested and rejected. The user wants:

1. **Comprehensive literature review** of techniques the V1.x family has
   NOT yet tried that 2024-2026 SOTA reports as IC-positive on time-series.
2. **First-principles novel ideas** — combinations or new tricks that
   aren't in published literature but follow from fundamentals of the
   problem.

The goal is not "incremental +0.005 IC". The goal is to **find a +0.020 to
+0.040 IC lift** that lets the V1.x family approach Headline (IC > 0.10)
WITHOUT requiring foundation-tier compute.

## Tasks (priority-ordered)

### Task 1 — Literature: novel anti-fragility / ShIC-lift techniques

The V1.x ShIC record is 0.037 (vs IC 0.067 → ratio 0.55, comfortably above
our 0.5 anti-fragility floor). To push ShIC further:

1. What 2024-2026 techniques specifically address the gap between
   train-loss minimization and shuffled-IC robustness? Look at:
   - Sharpness-Aware Minimization (SAM) and follow-ons
   - Wavelet-based feature regularizers
   - Mixup / cutmix / time-domain augmentation specifically for time-series
   - Label noise injection (Pearl 2024 / Chen 2025 tabular)
   - Causal regularization (CausalForecasting 2025)
2. Are there 2025-2026 distributional regression techniques (CRPS,
   energy distance, Sinkhorn loss) that lift ShIC over the V1.x's plain
   TwoHot CE?
3. What about adversarial training vs noise (we tried JEPA-style time-shuffle
   adversarial in V6, dropped per LITERATURE.md Hole 2; but that was on
   FOUNDATION not V1.x — does it lift V1.x ShIC?)?

For each technique: **(name) | (paper / repo) | (claim) | (compute on
4060 for V1.x retrain) | (project-specific applicability)**.

### Task 2 — Literature: novel IC-lift techniques on V1.x architecture

The V1.x family already uses:
- Transformer encoder (d_model=256, 8 heads, 3 layers)
- RSSM categorical bottleneck (24×24 codes)
- Multi-horizon TwoHot prediction
- Cross-feature attention (V1.4 only — FeatureAttentionBlock)
- KL anneal / Gumbel / ATME dream (V1.6)

What 2024-2026 work has reported ≥ +0.01 IC on similar small-cap
(≤ 5M params) time-series predictors? Look at:

1. **Conformal prediction** (Romano 2019 → Bhatnagar 2024) — wraps a base
   predictor in a calibration layer that shifts the loss surface. Has
   been shown to lift downstream Sharpe; not yet tested on our V1.x.
2. **MDN heads** (Mixture Density Networks, Bishop 1994 → Khorshid 2023)
   instead of TwoHot — predict K-component Gaussian mixture. Smoother
   gradient, can fit fat tails better.
3. **Quantile regression heads** (Wang 2017 → Lim 2024) — predict at
   p ∈ {0.05, 0.5, 0.95} jointly, optimize check-loss. Better tail
   behavior than TwoHot.
4. **Energy-based models** for return distribution (LeCun 2025 EBM-Trans).
5. **Information bottleneck regularization** (Tishby → Alemi 2024) on
   the latent code.

### Task 3 — First-principles novel ideas (not in published lit)

Reason from the structure of our problem:
- Inputs: 34 normalized features over 96 dollar bars
- Outputs: TwoHot return distribution at 4 horizons
- Target: IC ≈ 0.07 (current) → 0.10+ (Headline)
- Anti-fragility: ShIC > IC × 0.5

What novel architectural / loss / training-protocol ideas could plausibly
push ShIC past 0.05 + IC past 0.10 at 2M params on dollar-bar input?

Candidates to investigate from first principles:

1. **Asymmetric returns asymmetric loss**. Crypto returns are skewed
   (more upside than downside in bull regimes; opposite in bear). Should
   the loss have a regime-conditioned asymmetric Huber delta?
2. **Multi-asset shared backbone with per-asset heads**. Currently V1.x
   trains per-asset. A shared backbone + per-asset prediction head would
   let assets cross-pollinate without crashing model size. Already in V12
   architecturally; has it been tested on V1.x family?
3. **Memory bank for non-stationary features**. The V1.x sequence
   length is 96 bars. Add a slow-memory bank (NeurIPS 2024 RetNet /
   Hyena) of N=10000 historical "regime exemplars" the model can
   attend to. Effective context = 96 fast + 10K slow.
4. **Self-supervised pretraining ON V1.x** at 2M params. Currently the
   V1.x family trains end-to-end on the supervised loss. Run a 1-epoch
   self-supervised pretrain (masked feature reconstruction OR
   horizon-prediction over unlabeled data), then finetune on the
   labeled return objective. Foundation-style at small scale.
5. **Contrastive sequence augmentation**. For each window, add a
   contrastive loss against a temporally-jittered or noise-augmented
   version of itself. Forces invariance to small temporal perturbations
   (which IS the thing distinguishing real signal from memorization).
6. **Gradient surgery** between horizons. The 4 horizon heads have
   conflicting gradients on shared backbone params. Apply PCGrad
   (Yu 2020) or CAGrad (Liu 2021) to reconcile. May lift IC at all
   horizons jointly.
7. **Frequency-domain features**. Add DFT/wavelet-domain features as
   additional channels. The V1.x has time-domain only.

For each candidate: **(idea) | (mechanism) | (expected effect on IC and
ShIC, separately) | (compute cost on 4060) | (risk of breaking
anti-fragility)**.

### Task 4 — Hybrid: combine 1+2+3 into a coherent V2.x design

If all three task lines yielded promising candidates, propose a coherent
**V2.x architecture** (still ≤ 5M params, fits 4060) that combines the
top 3-5 ideas. Specify:

- New module composition (which V1.x blocks to keep, which to replace)
- Loss function (combined supervised + auxiliary terms)
- Training curriculum (single-stage vs multi-stage SSL → finetune)
- Expected IC and ShIC envelope (ranges with rationale)
- Compute budget for one full retrain on 4060

### Task 5 — What V1.x family is missing that the foundation prong has

The foundation prong (Prong 1 in `src/frontier_ml/foundation/`) has things
the V1.x family doesn't:
- Cross-asset attention (V1.x doesn't see other assets jointly; only via xd_*)
- Lead-lag JEPA contrastive (V1.x has no contrastive loss)
- 8x sequence length (S=512 vs V1.x S=96)
- Adaptive log-spaced bins (V1.x uses 255-uniform)
- Larger d_model (768 vs 256)

Of those, **which transfer down to a 2M-param V1.x retrain** without
breaking the budget? Specifically:
- Could V1.x be retrained with cross-asset attention (panel input) at
  2M params?
- Could V1.x adopt adaptive bins now that we have `adaptive_bins.py`?
- Could V1.x adopt JEPA contrastive (small-batch lead-lag) at 2M params?

For each: implementation feasibility (yes/no/maybe), expected IC delta,
risk of breaking V1.x's existing record.

## Output format

Return one document with these sections:

1. **Executive verdict** (≤ 200 words). Top three V1.x upgrades (one
   from each of Tasks 1-3) ranked by expected IC + ShIC lift per GPU-hour.
2. **Anti-fragility / ShIC-lift inventory** (Task 1 deliverable).
3. **IC-lift inventory at 2M params** (Task 2 deliverable).
4. **First-principles candidates** (Task 3 deliverable) — each with
   independent IC and ShIC estimates and a "what would have to be true
   for this to work" critique.
5. **Proposed V2.x design** (Task 4 deliverable) — coherent stack
   combining the top picks.
6. **Foundation-to-V1.x technique transfer** (Task 5 deliverable) — table
   of which foundation-prong tricks ship down to 2M-param family.
7. **Top 5 next V1.x retrains in priority order**, each with the full
   parameter set the trainer would receive.

## Operational constraints

- Hardware: 1× RTX 4060 (8.59 GB VRAM), i9 20 cores, 32 GB RAM.
- V1.x retrain budget: ~4-6 GPU-hours per version per pattern.
- ShIC > IC × 0.5 must hold; do NOT recommend anything that lifts IC
  while crashing ShIC ratio.
- No emojis in any output that touches Python files (Windows cp1252).
- The user's stated bar: a +0.02 to +0.04 IC lift on the V1.x family,
  not incremental +0.005.
- Reference materials in this repo:
  - `CLAUDE.md` (cross-version invariants, tier ladder, Pattern P+Q)
  - `src/frontier_ml/LITERATURE.md` (8 holes already closed)
  - `src/wm/v1/v1_*/world_model.py` (current V1.x architectures)
  - `memory/fix_logs/INDEX.md` (cross-cutting bug patterns)

## Time / cost budget

- WebSearch calls: 10-15.
- WebFetch calls: 3-6 (pick the 2025-2026 papers most relevant to a 2M-param
  time-series predictor on dollar-bar crypto data).
- Output budget: 2500-4500 words.

## Confidence tagging (mandatory per memory/feedback_search_reliability_protocol.md)

Every load-bearing numerical claim (IC, ShIC, Sharpe, %-improvement, dates,
param counts, money) MUST be tagged inline:

- `[VERIFIED]` — raw source fetched (arxiv abstract page, GitHub raw README,
  HuggingFace `/api/models/<id>` JSON, paper HTML/PDF body — NOT a
  summarized WebFetch response).
- `[REPORTED]` — sourced from a WebSearch snippet OR a summarized WebFetch;
  not yet re-checked against raw text.
- `[INFERRED]` — derived/computed/extrapolated; not directly stated.

End the response with a **Reliability ledger** counting VERIFIED vs
REPORTED vs INFERRED across all numerical claims. If REPORTED > 0 on any
decision-gating number (one that a V1.x retrain depends on), surface those
in a `## Caveats` section so the reader knows precisely which numbers to
re-check before retraining.

Never let a recommendation sound more confident than its verification
level: REPORTED foundations → REPORTED-grade recommendations.

## Stop conditions

- If the literature search returns no technique that has been MEASURED to
  add ≥ +0.01 IC on a comparable small-cap predictor, drop the polite
  framing and recommend dropping V1.x track entirely in favor of the
  foundation prong. The user accepts that as a valid outcome.
- If first-principles candidates all have caveats or risks that exceed
  expected lift, say so. We do not need 7 weak ideas; one strong idea is
  better.
- If foundation-prong techniques transfer cleanly down (Task 5), prioritize
  those over novel-from-first-principles ones — proven over speculative.
