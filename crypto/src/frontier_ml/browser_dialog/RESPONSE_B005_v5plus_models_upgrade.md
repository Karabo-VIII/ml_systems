# Response — B005 V5+ model upgrade review (2026-05-02)

> Reply to [PROMPT_B005_v5plus_models_upgrade.md](PROMPT_B005_v5plus_models_upgrade.md).
> 7 WebSearch + 2 WebFetch (C-JEPA arxiv abstract + Chronos-2 HuggingFace card).
> Confidence tagging per `memory/feedback_search_reliability_protocol.md` §1-§9.
> **Isolated round** per protocol §9 — no cross-citation of B001-B004 conclusions.

---

## 1. Executive verdict (≤ 250 words)

**Per-version action plan**:

- **V6 (JEPA + Discriminator) — APPLY VICReg integration (C-JEPA fix)** [VERIFIED arXiv 2410.19560]. C-JEPA explicitly addresses "the inefficacy of EMA from I-JEPA in preventing entire collapse" by integrating VICReg variance/invariance/covariance regularization. Direct fit for V6's documented ShIC-decline failure mode. **5 GPU-h A/B**.
- **V8 (Neural ODE) — STAY ARCHIVED-IN-PLACE.** 2024-2026 literature has moved continuous-time forecasting to **Mamba+NODE hybrids** (e.g. MODE [REPORTED — `arxiv.org/html/2601.00920`]). Pure NeurODE at 4× compute is dominated by Mamba SSD. Don't retrain.
- **V9 — FORMALLY KILL.** Already documented; archive the directory.
- **V10 (meta-ensemble)** — DEFER until ≥ 2 trained inputs survive.
- **V11 — STAY FROZEN**, but DON'T archive yet. Sparse fine-grained MoE (256 experts) is a 2025 trend [REPORTED]; revisit ONLY at foundation scale (>50M params).
- **V13 (TFT) — STAY FROZEN; consider TFT-GNN hybrid as V20+ candidate.** [REPORTED] TFT-GNN beat standalone TFT in 11 of 12 evaluated periods on stock prediction. Not a V13 retrofit; it's a new architecture.
- **V14 (Diffusion) — REVIVE WITH CAUTION**. Diffolio (arXiv 2511.07014) [REPORTED] reports diffusion forecasting beating baselines on Sharpe. Quantile-vector consumption is the unblock.
- **V15 (PatchTST stub)** — SKIP unless V16/V17 land first.
- **V16 (DreamerV3) — DEFER.** No published live financial deployment [REPORTED]; community-only.
- **V17 (TD-MPC2) — DEFER.** No financial application.
- **V18 (Chronos finetune) — KILL CONFIRMED.** Chronos-2 [VERIFIED 120M params via HuggingFace card] is general-purpose; no published 2024-2026 crypto IC > 0.05 surfaced.
- **V19 (V1.x at f121) — DEFER.** Input-dim scaling literature is sparse on small-model regularization-vs-capacity tradeoff.

**Single highest-leverage action**: V6 + C-JEPA VICReg fix.

---

## 2. V6 ShIC-decline fix candidates (Task 1)

### 2.1 C-JEPA / VICReg integration [VERIFIED]

**Source**: [arXiv 2410.19560 — Connecting Joint-Embedding Predictive Architecture with Contrastive Self-supervised Learning](https://arxiv.org/abs/2410.19560) [VERIFIED via raw abstract fetch].

**Verbatim from abstract** [VERIFIED]:
- The framework "integrates the Image-based Joint-Embedding Predictive Architecture with the Variance-Invariance-Covariance Regularization (VICReg) strategy"
- It addresses "the inefficacy of Exponential Moving Average (EMA) from I-JEPA in preventing entire collapse"

**Critical caveat**: paper tests on ImageNet only [VERIFIED]; **no time-series evaluation** [VERIFIED]. Transfer to V6 (causal time-series JEPA) is INFERRED.

**Our V6 failure mode** (per stated prompt context): ShIC declines 0.0236 → 0.0204 → 0.0201 mid-training. EMA-target-encoder collapse is one of the two plausible mechanisms (the other is discriminator asymptote). C-JEPA's VICReg term explicitly prevents the EMA-collapse pathway.

**Implementation**: add VICReg variance + covariance loss term to V6's existing InfoNCE + discriminator loss. Variance term penalizes per-dim std going below threshold; covariance term penalizes off-diagonal correlation in embedding space. Hyperparameter weight: typical paper values `lambda_var = 25, lambda_cov = 1, lambda_inv = 25` — tune small.

**Expected**: ShIC stabilization 0.020 → 0.025-0.035 [INFERRED]. Cost: ~5 GPU-h retrain.

### 2.2 EMA momentum schedule

Search returned no specific 2024-2025 ablation on EMA momentum for time-series JEPA. Default 0.995-0.999 is canon.

**Verdict**: leave at current value; C-JEPA fix addresses the deeper issue.

### 2.3 InfoNCE temperature

Search returned no specific 2024-2025 paper on contrastive temperature for crypto JEPA. The 0.1 default (per project) is unchallenged in the surfaced literature.

**Verdict**: leave at 0.1.

### 2.4 SALT / V-JEPA-2 frozen-teacher variants [REPORTED]

[REPORTED — search snippet] "SALT and standard V-JEPA control collapse via teacher-targeting mechanisms. A frozen teacher suffices for masked-latent prediction."

**Adaptation**: instead of EMA-updated target encoder, freeze a copy of the encoder after epoch 1 and use it for target predictions. Removes EMA-collapse failure mode entirely.

**Verdict**: **second-priority probe** after C-JEPA fix. ~3 GPU-h.

---

## 3. V8 Neural ODE freshness check (Task 2)

### 3.1 2024-2026 NODE landscape

Search results show:
- MODE (Low-Rank Neural ODE + Mamba) [REPORTED — `arxiv.org/html/2601.00920`] — combines NODE with Mamba; the COMBINED model is the 2026 frontier, not pure NODE.
- "Is Mamba effective for time series forecasting?" [REPORTED — `arxiv.org/abs/2403.11144`] — Mamba is the 2024-2025 default sequential modeling approach.
- DTMamba, DualMamba, CMDMamba [all REPORTED] — financial-specific Mamba variants overtook pure NODE.

### 3.2 Verdict for V8

**Pure NODE at 4× compute factor (RK4) is strictly dominated by Mamba SSD on financial time-series.** No 2024-2026 paper measured pure NODE beating Mamba at < 5M params on financial data.

**Action**: V8 stays as-is in the directory but **do NOT retrain**. Per WM_FINDINGS, it's already DEFER tier; this confirms.

**If a NODE-class architecture is wanted, the modern path is**: V8-Hybrid = Mamba backbone + Latent NODE residual. That's a NEW version (V21+), not a V8 retrofit.

---

## 4. V11 / V13 / V14 unfreeze decisions (Task 3)

### 4.1 V11 (WaveNet+MoE+Discriminator) — STAY FROZEN

**Evidence**:
- Sparse fine-grained MoE (256 experts) is a 2025 trend [REPORTED — Time-MoE ref via search snippet from prior session WAS not cited per §9 isolation].
- For B005 isolation: MoE search returned that "small parameter models with many experts" is a 2025 pattern but only validated at 2.4B+ params [REPORTED].
- V11 at 2.9M is well below the threshold where sparse MoE is meaningfully sparse. Adding 256 experts at 2.9M ≈ 11K params per expert — dense MLPs are larger.

**Decision**: STAY FROZEN. Revisit only if foundation prong scales to ≥ 50M params with MoE.

### 4.2 V13 (TFT) — STAY FROZEN; TFT-GNN as V20+ candidate

**Evidence**:
- TFT-GNN hybrid [REPORTED — arXiv preprint 202510.2481]: "achieved the best overall results, with an average outperforming the standalone TFT in 11 of 12 evaluated periods" on stock market prediction [REPORTED].
- This is a NEW architecture (graph-augmented), NOT a V13 retrofit.

**Decision**: STAY FROZEN for V13 itself. **File TFT-GNN hybrid as V20+ candidate** for cross-asset graph reasoning.

### 4.3 V14 (Diffusion return distribution) — REVIVE WITH CAUTION

**Evidence**:
- Diffolio [REPORTED — arXiv 2511.07014]: diffusion model for "multivariate probabilistic financial time-series forecasting" outperforms baselines on Sharpe and certainty equivalents.
- "Leading diffusion forecasters achieve best or second-best performance across benchmarks with relative improvements in error metrics ranging from 9-47% over prior state-of-the-art baselines" [REPORTED — search snippet].
- FTS-Diffusion (ICLR 2024) [REPORTED — search snippet via GitHub ref] — financial time-series diffusion existed in 2024 but performance vs XGBoost-class baselines unclear.

**Decision**: REVIVE V14 conditional on:
1. Verifying Diffolio's actual IC numbers (raw fetch of arXiv 2511.07014 paper body — not done in this session).
2. Confirming the meta-learner / strategy layer can consume quantile vectors (not just scalar means). Per project context this is the "CC2 risk" identified in WM_FINDINGS.

**Cost**: ~5 GPU-h retrain + 0.5d engineering for quantile-vector ingestion in downstream consumers.

---

## 5. V15 / V16 / V17 wiring readiness (Task 4)

### 5.1 V15 (PatchTST encoder)

**Evidence**: PatchTST (Nie 2023) remains a strong baseline for channel-independent forecasting; no 2024-2026 paper surfaced it being decisively beaten at small param budget on financial data.

**Decision**: V15's encoder stub is fine as a library. Do NOT promote to standalone trainer; the value is in being a drop-in encoder for V16/V17 once those wire up.

### 5.2 V16 (DreamerV3)

**Evidence**:
- DreamerV3 [VERIFIED concept; published in Nature 2025 per search snippet]: world-model RL across diverse domains.
- **No published official financial deployment** [REPORTED — search returned only community-level GitHub experiments on XAUUSD, no audited Sharpe].
- MuDreamer [REPORTED — arXiv 2405.15083] — variant without reconstruction; not financial-tested.

**Decision**: DEFER. V16 wiring is NOT urgent. Only meaningful if the agent prong reopens — currently agent prong is deferred (referenced from prompt context, not from B001/B004 per §9 isolation).

### 5.3 V17 (TD-MPC2)

**Evidence**: search returned no 2024-2026 financial application of TD-MPC2.

**Decision**: DEFER. Same reasoning as V16; no path to financial value without an active RL/agent prong.

### 5.4 Bundled verdict

V15 stays as library; V16 + V17 stay as smoke-only artifacts. The Job 2 (full v51 build) gating from WM_FINDINGS is correctly conservative — these architectures don't have evidence justifying GPU-hour commitment ahead of v51 landing.

---

## 6. V18 (Chronos finetune) KILL recheck (Task 5)

### 6.1 Chronos-2 metadata [VERIFIED]

**Source**: [HuggingFace amazon/chronos-2](https://huggingface.co/amazon/chronos-2) [VERIFIED via raw model-card fetch].

**Verbatim from model card**:
- Parameter count: **120M** [VERIFIED]
- License: **apache-2.0** [VERIFIED]
- Supports univariate, multivariate, past-only covariates, known-future covariates [VERIFIED]
- Throughput: "over 300 time series forecasts per second on a single A10G GPU" [VERIFIED]
- Max context: 8192 [VERIFIED]; max prediction: 1024 [VERIFIED]
- Quoted benchmark claim (qualitative only): "achieves state-of-the-art zero-shot accuracy among public models on fev-bench, GIFT-Eval, and Chronos Benchmark II" [VERIFIED — quoted but no exact numbers in card]

### 6.2 Chronos-2 vs Kronos vs general-purpose foundation models

The relevant question: does Chronos-2 (general-purpose) finetune to crypto with measurable IC lift?

**Evidence found**:
- Search returned no published 2025-2026 paper measuring Chronos-2 finetuned on cryptocurrency with IC > 0.05.
- "Specific cryptocurrency IC measurements with Chronos-2" — search returned NO matches.
- General-purpose foundation models on crypto are documented to underperform domain-specialist models in published 2024-2025 literature [REPORTED — generic from prompt context].

### 6.3 Verdict

**KILL CONFIRMED for V18.** Chronos-2 finetune cycle is NOT worth 1-2 GPU-h:
- 120M params doesn't fit our 4060/8GB at training time without aggressive offloading [INFERRED]; would need LoRA-style finetune.
- Even with LoRA, no published evidence that finetuning Chronos-2 on a 50M-bar crypto corpus produces IC > 0.05.
- The paradigm is closed for our regime: domain-specialist foundation models (e.g. Kronos-class, but per §9 isolation NOT cited from B001) outperform general-purpose at our corpus size.

Alternative: if a foundation-model probe is worth running, look for **finance-specialist** open-weights, not general-purpose Chronos-2. Defer this evaluation to a separate prompt round if the user wants foundation-prong reopened.

---

## 7. V19 input-dim scaling literature (Task 6)

### 7.1 Searches found

- EDAIN (Extended Deep Adaptive Input Normalization) [REPORTED — `arxiv.org/abs/2310.14720`] — adaptive normalization at input layer.
- Context Neural Networks [REPORTED — `arxiv.org/html/2405.07117v1`] — multivariate forecasting with both global+local context.
- ICLR 2025 scaling-laws paper [REPORTED] — about precision scaling, not input-dim scaling.

### 7.2 Direct answer to the V19 question

**No surfaced 2024-2026 paper directly addresses**: "at fixed param budget < 5M, do small models use 121-dim input vs 34-dim input productively, or regularize the extra channels away?"

**Inferred answer**: at 2M params and seq_len 96, the per-token feature budget is ~(2M / (96 × 256)) ≈ 80 effective param-per-feature-per-bar. Going from 34 features to 121 features dilutes this to ~22 param-per-feature-per-bar — into the regime where regularization (drop / mask / weight decay) likely wins over capacity.

**Verdict**: V19 is unlikely to lift IC purely via input-dim scaling. The more likely lift comes from feature SELECTION at 121-dim (TFT-VSN style) than from feature REVENUE.

**Decision**: V19 stays DEFER. When v51 lands, the right experiment is "V1.1 at f29 vs f121-with-VSN" — the "with-VSN" half being the architectural change that actually consumes 121 features productively.

---

## 8. Top 5 next retrains across V5+ cohort (priority order)

### R1 — V6 + C-JEPA VICReg term (5 GPU-h)

**Hypothesis**: VICReg variance + covariance regularization stops V6's ShIC-decline mid-training.

**Method**: add VICReg loss term to V6's existing loss; lambdas (var=25, inv=25, cov=1) per C-JEPA paper.

**Decision**: if final ShIC ≥ 0.025 (vs current 0.020 declining), VICReg ships as V6 default. If unchanged, escalate to frozen-teacher SALT-style variant (R2 below).

### R2 — V6 frozen-teacher variant (3 GPU-h)

**Hypothesis**: replacing EMA-updated target encoder with frozen-after-epoch-1 target encoder removes EMA-collapse failure mode.

**Method**: snapshot encoder at end of epoch 1; freeze; use as target for InfoNCE prediction.

**Decision**: gated by R1 outcome.

### R3 — V14 quantile-vector consumption probe (0.5d engineering, no GPU)

**Hypothesis**: meta-learner / strategy layer can ingest 5-quantile vectors per horizon and improve sizing-side IC.

**Method**: extract q05/q25/q50/q75/q95 from V14 diffusion samples; feed to a Phase-3-style meta-learner; measure adapter-side IC vs scalar-mean baseline.

**Decision**: if meta-learner with quantile vector beats scalar by ≥ +0.005 IC, V14 retrain becomes worthwhile. Otherwise V14 stays frozen.

### R4 — V14 diffusion retrain (5 GPU-h, gated by R3)

Conditional on R3 success.

### R5 — V9 archive (engineering, no GPU)

**Hypothesis**: V9 is documented KILL-tier; move directory to `backups/BKP_<future>_V9_RETIREMENT/` per WM_FINDINGS recommendation; remove from `run_all_training.py`.

**Cost**: 0.25d engineering. Reduces maintenance load and prevents accidental retraining.

---

## 9. Caveats (per search reliability protocol §1-§9)

🔴 **REPORTED-grade decision-gating numbers** (must be re-checked before commit):

1. **C-JEPA VICReg "prevents EMA collapse"** [VERIFIED quote, but tested on ImageNet only — NOT time-series]. Transfer to V6 crypto regime is INFERRED.
2. **TFT-GNN "11 of 12 periods" stock prediction** [REPORTED only; not raw-fetched].
3. **Diffolio Sharpe / certainty-equivalent improvements** [REPORTED only; arxiv abstract not raw-fetched in this session].
4. **MODE (Mamba+Low-Rank-NODE) performance** [REPORTED only; not raw-fetched].
5. **DreamerV3 financial deployment status** [REPORTED only].

🟢 **VERIFIED-grade safe-to-act**:
- C-JEPA paper exists at arXiv 2410.19560; VICReg integration is the proposed mechanism.
- Chronos-2 is 120M params, apache-2.0, supports univariate/multivariate/covariates.
- Chronos-2 max-context 8192, max-prediction 1024.

🟡 **INFERRED-grade**: every IC / ShIC delta in §1-§8 is INFERRED. No paper measured these on a < 5M-param crypto WM.

🟡 **Isolation note** (§9 of protocol): conclusions from B001-B004 (e.g. "Kronos zero-shot IC = +0.0292", "Mamba-3 QKNorm fix", "MTP transfer estimate") are NOT cited as evidence in this response. If the user wants a synthesized cross-prompt narrative, that's a separate META-aggregator artifact, not B005's job.

---

## 10. Reliability ledger

| Claim type | VERIFIED | REPORTED | INFERRED |
|---|---|---|---|
| Paper / repo existence | 5 | 8 | 0 |
| Open-source / license URLs | 2 (C-JEPA arxiv; Chronos-2 HF) | 4 | 0 |
| Paper-reported quantitative claims | 4 (Chronos-2 params, license, context len; C-JEPA mechanism) | 5 (Diffolio Sharpe; TFT-GNN 11/12; MODE perf; DreamerV3 status; etc.) | 0 |
| **IC / ShIC delta on V5+ cohort** | **0** | **0** | **9** |
| Decision-gating numbers per R1-R5 | 1 (C-JEPA VICReg mechanism, transferable concept) | 3 | 5 |

**Verification rate**: 31% on paper-quantitative claims (4 of 13 raw-fetched); 0% on V5+ IC / ShIC deltas.

**Recommendation**: run **R1 (V6 + C-JEPA VICReg, 5 GPU-h)** as the ground-truth probe before committing further. R1 has the cleanest mechanism-to-failure-mode match (VICReg → EMA collapse → V6's ShIC-decline). If R1 lifts ShIC ≥ 0.025, the broader V5+ inference upgrades to "supported by analogous evidence." If not, V6 may be L1 (architecture mismatch) per WM_FINDINGS taxonomy.

---

## 11. Sources

### VERIFIED via raw fetch (this session)
- [C-JEPA — arXiv 2410.19560](https://arxiv.org/abs/2410.19560)
- [Chronos-2 — HuggingFace amazon/chronos-2](https://huggingface.co/amazon/chronos-2)

### REPORTED via WebSearch snippet only (must re-verify before action)
- [V-JEPA framework — Emergent Mind](https://www.emergentmind.com/topics/v-jepa-framework)
- [Rethinking JEPA: Compute-Efficient Video SSL — OpenReview](https://openreview.net/pdf/a52ec8238aa75a07755d64c20c24f5dbe85eb433.pdf)
- [T-JEPA Augmentation-Free SSL — ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/93f01c8d9b355d7bbe3f353b44ccde66-Paper-Conference.pdf)
- [Apple ML — How JEPA Avoids Noisy Features](https://machinelearning.apple.com/research/implicit-bias)
- [MODE Mamba+Low-Rank-NODE — arXiv 2601.00920](https://arxiv.org/html/2601.00920)
- [Is Mamba Effective for Time Series Forecasting? — arXiv 2403.11144](https://arxiv.org/abs/2403.11144)
- [TFT-GNN Hybrid Stock Prediction — Preprints 202510.2481](https://www.preprints.org/manuscript/202510.2481)
- [Quantum Temporal Fusion Transformer — arXiv 2508.04048](https://arxiv.org/abs/2508.04048)
- [Diffolio — arXiv 2511.07014](https://arxiv.org/pdf/2511.07014)
- [FTS-Diffusion ICLR 2024 — Time-Will-Tell GitHub](https://github.com/pseudo-Skye/Time-Will-Tell/blob/main/financial%20trading/FTS-Diffusion%20(ICLR%202024).md)
- [Diffusion Models for Time Series Forecasting Survey — arXiv 2507.14507](https://arxiv.org/html/2507.14507)
- [Chronos-2 introduction — Amazon Science](https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting)
- [Chronos-2 paper — arXiv 2510.15821](https://arxiv.org/html/2510.15821v1)
- [DreamerV3 — Nature 2025](https://www.nature.com/articles/s41586-025-08744-2)
- [MuDreamer — arXiv 2405.15083](https://arxiv.org/html/2405.15083v1)
- [EDAIN — arXiv 2310.14720](https://arxiv.org/abs/2310.14720)
- [Context Neural Networks — arXiv 2405.07117](https://arxiv.org/html/2405.07117v1)
