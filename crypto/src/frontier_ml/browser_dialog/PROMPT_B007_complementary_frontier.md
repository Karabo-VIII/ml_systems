# B007 — Complementary frontier approaches (microstructure + calibration + multi-asset + continual)

> **Status:** OPEN  •  **Sent:** 2026-05-02
> **For:** an `@browser`-routed Claude Code session with WebSearch / WebFetch.
> **From:** the cohort state after B001-B006 — V1.x family wired with 6 upgrades,
> V15-V21 SOTA scaffolds, B006 modules (TTT/Koopman/Born-Again) built, V9+V18 archived.
> **Tone:** /un — direct, ship-or-concede, no polite hedging.
> **Isolation:** per `memory/feedback_search_reliability_protocol.md` §9.
>  Conclusions from B001-B006 must NOT be cited as evidence; they're context.

## Mission framing

The B001-B006 cohort already covered architectures + training-loss tricks +
state-space models + JEPA + diffusion + foundation-class transfer.
**This prompt targets COMPLEMENTARY tools** — what wraps, deploys, calibrates,
specializes, or maintains the models we already have.

User's framing (verbatim): *"frontier approaches that will complement our models,
both foundation and extra."*

## What we have (context, not citations)

7 V1.x-aligned trainers wired with `--sam --fraug --pcgrad --mtp
--adaptive-bins --mdn`. SOTA backbones for PatchTST (V15), DreamerV3 (V16),
TD-MPC2 (V17), TFT-GNN hybrid (V20), Mamba-NODE hybrid (V21), sparse-MoE
(V11'). VICReg module ready for V6. B006 modules: Test-Time Training,
Koopman Neural Forecaster, Born-Again self-distillation.

What we're missing — and what this prompt asks browser to surface — is
the COMPLEMENTARY layer: stuff that turns these predictors into
deployable, calibrated, regime-robust trading signals.

## Tasks (priority-ordered; each task = an independent investigation)

### Task 1 — Crypto microstructure techniques 2024-2026

What 2024-2026 crypto-research-specific techniques does the literature
report as IC-positive (or Sharpe-positive) on top of dollar-bar features?

Specifically:
1. **Order-book imbalance models** — LOBSTER-style L2 features beyond simple
   bid/ask spread; queue-position dynamics; depth-weighted mid (DWM) signals.
2. **Liquidation-cascade prediction** — Binance / Bybit liquidation feeds as
   leading indicators; 2025 papers reporting Sharpe lift from liquidation
   "shock" features.
3. **MEV-aware features** — sandwich attacks, JIT liquidity, priority gas
   bidding patterns as exogenous signals for L1 (ETH/SOL) prices.
4. **Funding-rate / basis arbitrage as feature** — perp-spot basis,
   cross-venue funding spreads, funding momentum (V1.x has norm_funding +
   norm_funding_momentum but uses them as regression inputs, not as
   regime / arbitrage signals).
5. **Stablecoin flow features** — USDT/USDC mint-burn, off-exchange flow,
   exchange netflow.

For each: **(technique) | (paper / repo with arXiv ID or URL) | (claimed
Sharpe / IC lift) | (data prerequisite — does our chimera_legacy have the
inputs, or is new ingest needed?) | (reliability tier per protocol)**.

### Task 2 — Inference-time calibration + deployment post-processing

We have V1.x's TwoHot / MDN return distributions and can decode to
expected-return + variance. What 2025-2026 work has shown to ADD IC or
deployment-tier value via a post-prediction layer?

1. **Conformal prediction wrappers** — beyond the surface mention in B003;
   specifically conformal *quantile* prediction for trading sizing.
   Bhatnagar 2024, Zhang 2025 (CPTC), Gibbs & Candes 2024.
2. **Isotonic regression / Platt scaling** for IC calibration on TwoHot bin
   probabilities. Has anyone published IC-lift numbers from this?
3. **Distribution alignment** — taking a distribution-aware predictor and
   refitting its quantiles to match empirical return quantiles via mass-
   transport (Sinkhorn / OT). Recent 2025 work on Wasserstein-aligned
   forecasters.
4. **Ranking / ordinal heads** — pairwise rank loss, ListNet, NDCG-on-IC.
   We have a small pairwise loss in V1.x; is the field doing better in 2025?
5. **Classifier-free guidance** for diffusion-class predictors (V14).

Output: **technique | paper | claim | impl difficulty | applicability to
TwoHot+MDN heads we have**.

### Task 3 — Multi-asset specialization (50-100 asset universe)

V12 has cross-asset attention but trains per-asset. The foundation prong
sees u100 jointly. What's the 2025-2026 SOTA for handling 50-100 assets:

1. **Asset-as-token** — treat each asset as a sequence position; the
   transformer attends over assets explicitly. Some papers ML4Trading 2025.
2. **Asset-conditioned LoRA / adapters** — frozen shared backbone +
   per-asset rank-r adapter (~1-10K params per asset). PEFT-style
   specialization at inference.
3. **MoE-over-assets** — route each asset to a sparse subset of experts;
   distinct from MoE-over-tokens (V11' covers the latter).
4. **Asset embeddings learned via contrastive** — beyond simple lookup
   embedding; learned via lead-lag InfoNCE or asset-cluster contrastive.
5. **Asset-cluster meta-learning** — group similar assets (BTC / ETH / large-
   caps / small-caps), train meta-learner that adapts within cluster.

Output: per-technique recommendation with applicability assessment and
expected lift envelope (INFERRED is fine; VERIFIED-where-possible).

### Task 4 — Anti-memorization 2025-2026 (beyond SAM/FrAug/PCGrad)

Our anti-fragility floor is ShIC > IC × 0.5. What 2025-2026 techniques
specifically target the memorization-vs-generalization gap on financial
time-series?

1. **Label-noise injection** (Chen 2025 / tabular literature) — add
   calibrated noise to training labels; tested on regression with IC?
2. **Causal regularization** (CausalForecasting 2025) — penalize
   non-causal feature dependencies in the loss.
3. **Batch decorrelation** — explicit penalty on Pearson correlation
   between in-batch predictions and any single feature beyond a threshold.
4. **Sample-reweighting via gradient matching** — Yu et al. 2025-style
   re-weighting that down-weights samples whose gradient direction conflicts
   with the OOS gradient direction.
5. **Stochastic depth / DropPath at high rates** for time-series
   transformers (V1.x family).

For each: **VERIFIED claim** OR **INFERRED transfer to crypto regime**.

### Task 5 — Continual / online learning for non-stationary regimes

Crypto regimes shift (2022 bear → 2023 recovery → 2024 ETF → 2025 maturation
→ 2026 chop). What's beyond TTT for maintaining IC over months of regime drift?

1. **Continual learning with replay buffers** — store K representative
   windows from past regimes; periodically re-mix into current training.
   2025 papers on financial continual learning specifically.
2. **Drift detection + retraining triggers** — Page-Hinkley test, ADWIN,
   newer 2024-2025 detectors. When to retrain vs adapt vs hold.
3. **Online gradient updates** with conservative learning rates — formal
   stability guarantees for time-series.
4. **Test-time fine-tuning** beyond TTT — SAR, EATA, surgical fine-tuning.
5. **Mixture of online and offline models** — old (frozen) + new (adapting)
   ensemble.

### Task 6 — Score-based / EBM / flow-matching for return distributions

Our V1.x uses TwoHot bins; V14 uses diffusion; B003 R3 added MDN. What
NEWER (2025-2026) distributional heads exist that the cohort doesn't yet
include?

1. **Energy-based models for forecasting** (Du 2024 / LeCun 2025).
2. **Flow-matching for time-series** (Lipman 2024 + 2025 extensions).
3. **Score-based regression** (alternative to diffusion that's faster).
4. **Implicit quantile networks** (IQN; trained with quantile-Huber loss).
5. **Continuous distributional CRPS** with learned quantile networks.

For each: would it BEAT our adaptive-bin TwoHot or skewed-Student-t MDN
on tail capture, in published benchmarks?

### Task 7 — Open-weights finance-specialist models (not general TSFM)

B001 evaluated Kronos. Are there OTHER open-weights finance-specialist
models we missed? Looking for things between general-purpose TSFM
(Chronos / TimesFM) and crypto-specialist (Kronos):

1. **FinTSB** — open dataset + benchmark suite for time-series in finance.
2. **Time-LLM** — repurposes LLMs for time-series; has anyone tested on
   crypto?
3. **ChatTime / ChronoX / FinChrono / etc.** — finance-flavored chat-LLM-on-
   time-series hybrids.
4. Any 2025-2026 release between Kronos's 12B-K-line corpus scale and a
   crypto-only specialist.

## Output format

Return one document with these sections:

1. **Executive verdict** (≤ 250 words). Top three COMPLEMENTARY techniques
   per the foundational/extra framing established by prior dialogs:
   foundational = changes IC/ShIC ceiling; extra = polish; deployment =
   wraps existing predictor.
2. **Per-task findings** — Tasks 1-7, each with surfaced techniques + reliability tags.
3. **Module-by-module retrofit map** — table mapping each transferable
   technique to which existing module file it modifies / wraps / extends.
4. **Top 5 next experiments**, ordered by EV / GPU-h ratio.
5. **What gets DROPPED from current plan** — if any of these complementary
   techniques DOMINATE existing items, name them.
6. **Reliability ledger** per protocol §10.

## Operational constraints

- Hardware: 1× RTX 4060 (8.59 GB VRAM), i9 20 cores, 32 GB RAM
- Project goal: Headline IC > 0.10 / ShIC > 0.05
- Anti-fragile invariants: ShIC > IC × 0.5, walk-forward CV with purge
- No emojis in any output that touches Python files
- Specialist mindsets via Skill tool, not parallel Agent subagents

## Time / cost budget

- WebSearch calls: 12-18 (broad-then-narrow over 7 tasks)
- WebFetch calls: 4-7 (the 1-2 most-relevant papers per task that have
  abstract claims worth verifying)
- Output budget: 3000-5000 words

## Stop conditions

- If a task returns NO 2025-2026 work materially relevant to our cohort,
  say so and skip. We don't need synthesized novelty that isn't there.
- If one technique dominates (e.g. "conformal quantile wrappers add +0.01
  IC universally"), say it without polite hedging.
- If multi-asset specialization techniques all assume infrastructure we
  don't have (e.g. live LOB feeds at < 1ms latency), say so and concede.
