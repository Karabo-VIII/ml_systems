# B006 — New frontiers probe (expand arsenal while V1 trains)

> **Status:** OPEN  •  **Sent:** 2026-05-02
> **For:** an `@browser`-routed Claude Code session with WebSearch / WebFetch.
> **From:** the trained V1.x baseline (currently retraining) and current 3-prong
> foundation/distillation/multimodal stack.
> **Tone:** /un — direct, ship-or-concede.
> **Isolation:** per `memory/feedback_search_reliability_protocol.md` §9. This
>  prompt is INDEPENDENT of B001-B005. Findings from prior dialogs MUST NOT
>  be cited as evidence in this response.

## Mission framing

V1 baseline is training. **Expand the arsenal in parallel** — surface
2024-2026 frontier paradigms that the prior B001-B005 dialogs did NOT cover.
The B001-B005 cohort already covered: foundation models (Chronos / Kronos /
TimesFM / MOMENT / PatchTST), MTP, Hybrid-Mamba, Native-multimodal pretrain,
SAM, FrAug, MDN heads, PCGrad, SSL pretrain, JEPA + VICReg, Mamba-3 QKNorm,
FSQ bottleneck, diffusion, NODE, DreamerV3, TD-MPC2, Chronos-2.

Gaps to probe (NEW frontiers):

1. **Test-Time Training (TTT)** — adapt model weights at inference on the
   test sequence itself. Sun et al. 2024 / Wang 2025. Particularly relevant
   for non-stationary crypto regimes.
2. **Modern optimizers beyond Adam + SAM** — Lion (Chen 2023), Adan
   (Xie 2024), Lookahead (Zhang 2019), Sophia (Liu 2024). Any with
   measured benefit on small-cap time-series predictors?
3. **Architectures NOT in our cohort** — Liquid Neural Networks / CfC
   (Hasani 2022), Hyena Hierarchy (Poli 2023+), S4/S5/S6 (Smith 2023),
   Deep Equilibrium Models (Bai 2019+).
4. **Alternative output-distribution heads** — Energy-Based Models (EBM)
   for return distribution, Koopman-operator forecasters, Spectral /
   Fourier neural operators, Score-based regression (alternative to
   diffusion or MDN).
5. **Self-distillation / Born-Again Networks** — train V1.x, use as
   teacher, train V1.x v2 from same data. Furlanello 2018 + 2024-2025
   updates. Distinct from cross-version distillation.
6. **Test-time adaptation for distribution shift** — TENT (Wang 2021),
   TTA++ (2024-2025) for inference-time normalization / batch-statistic
   adaptation. Crypto regimes shift; this is a direct fit.
7. **Hierarchical / hyperbolic embeddings** — Poincaré (Nickel 2017),
   hyperbolic transformers (2024-2025). Tree-like structure of
   BTC → ETH → alt-cap. Applicable to cross-asset embedding space?
8. **Causal discovery as feature engineering** — Granger / NOTEARS /
   PCMCI 2024-2025. Identify which features Granger-cause returns vs
   are spurious.

## Tasks (priority-ordered)

### Task 1 — Test-Time Training (TTT) for time-series

Search:
1. TTT (Sun et al. NeurIPS 2024 follow-on work) applied to time-series.
2. Has anyone published TTT for financial / crypto with measured IC lift?
3. Compute cost on 4060 — is TTT viable when each test bar updates weights?

For each: paper, claim, IC delta if measured, applicability to V1.x at 2M
params and seq_len 96.

### Task 2 — Modern optimizers (Lion / Adan / Sophia / Lookahead)

Search:
1. Lion optimizer 2024-2025 follow-ons — beats AdamW on which benchmarks?
2. Adan / Sophia / Adafactor for time-series specifically.
3. Lookahead wrapper combined with SAM — any synergy or conflict?

### Task 3 — Architectures NOT in our cohort

Search:
1. Liquid Neural Networks / Closed-form Continuous-time (CfC) for
   time-series — is the 2024-2025 evidence positive on financial data?
2. Hyena Hierarchy for sub-quadratic time-series (distinct from Mamba SSM).
3. S5 / S6 (post-Mamba SSM variants) — any lift over Mamba on financial?
4. Deep Equilibrium Models (DEQ) — implicit-depth models; do they beat
   explicit-depth Transformers at <5M params?

### Task 4 — Alternative output-distribution heads

Search:
1. Energy-Based Models (EBM) for return distribution — LeCun 2025 EBM-Trans;
   anyone applied to financial?
2. Koopman-operator forecasters — physics-inspired; deterministic operator
   on observable space; has anyone validated on crypto?
3. Spectral / Fourier neural operators for time-series — typically used in
   PDE solving but applicable to financial? FNO / SFNO 2024-2025.
4. Score-based regression — alternative to MDN/diffusion; predicts gradient
   of log-density.

### Task 5 — Self-distillation / Born-Again Networks at V1.x scale

Search:
1. Self-distillation 2024-2025 follow-ons — does the "student becomes new
   teacher" loop converge for time-series predictors?
2. Has anyone published Born-Again on financial / small-cap predictors?
3. Compute cost — would 3 generations of V1.x self-distillation be worth
   ~12 GPU-h?

### Task 6 — Test-time adaptation (TENT / TTA / SAR)

Search:
1. TENT (Wang ICLR 2021) entropy-minimization at test time — has been
   extended to time-series in 2024-2025?
2. SAR / ConjugatePL / other TTA methods — applicability to non-stationary
   crypto regimes.
3. Does TTA conflict with our anti-memorization (ShIC) invariant?

### Task 7 — Causal-discovery feature engineering

Search:
1. PCMCI / NOTEARS for time-series feature selection 2024-2025.
2. Granger-causal feature pruning — does dropping non-Granger-causal
   features improve IC at <5M params?
3. Has any 2024-2026 paper measured causal-discovery-pruned features
   beating raw features on financial IC?

### Task 8 — Hyperbolic / hierarchical embeddings

Search:
1. Hyperbolic transformer 2024-2025 — useful for asset hierarchy
   (BTC dominance → ETH → alts)?
2. Poincaré embedding for cross-asset representation.
3. Has anyone published this on crypto?

## Output format

Return one document with these sections:

1. **Executive verdict** (≤ 250 words). Top three NEW frontiers worth
   adding to the arsenal, with FOUNDATIONAL/EXTRA classification per the
   2026-05-02 user-mandated framing.
2. **TTT / test-time adaptation findings** (Tasks 1 + 6).
3. **Optimizer landscape 2024-2025** (Task 2).
4. **Architecture frontiers NOT in cohort** (Task 3).
5. **Output-head alternatives** (Task 4).
6. **Self-distillation feasibility** (Task 5).
7. **Causal discovery for features** (Task 7).
8. **Hyperbolic / hierarchical embeddings** (Task 8).
9. **Top 5 next experiments** in priority order.

## Confidence tagging (mandatory per protocol §1-§8)

Every load-bearing numerical claim MUST be tagged:
- `[VERIFIED]` — raw source fetched.
- `[REPORTED]` — snippet/summary only.
- `[INFERRED]` — derived/extrapolated.

End with **Reliability ledger**. Surface REPORTED-grade gating numbers in
`## Caveats`.

## Result-quality isolation (mandatory per protocol §9)

DO NOT cite B001-B005 conclusions as evidence. Each search budget is fresh.

## Operational constraints

- Hardware: 1× RTX 4060 (8.59 GB VRAM), i9 20 cores, 32 GB RAM.
- ShIC > IC × 0.5 must hold.
- Goal: IC > 0.10 AND ShIC > 0.05 (per user mandate).
- Classify each surfaced upgrade as FOUNDATIONAL (capacity / architecture /
  target-representation change → moves IC ceiling) or EXTRA (additive on
  top → moves ShIC floor).
- No emojis in any output that touches Python files.

## Time / cost budget

- WebSearch calls: 8-12. Quality > volume.
- WebFetch calls: 3-5. Prioritize the most-promising 2024-2026 paper per
  task category.
- Output budget: 2500-4000 words.

## Stop conditions

- If a category returns no measurable evidence on financial / time-series
  data, mark CONCEDE for that task and proceed.
- If one frontier dominates the rest, name it without hedging.
- If all 8 tasks return concedes, drop the framing and recommend
  "no new frontier worth probing in 2026-05; consolidate B001-B005 stack."
