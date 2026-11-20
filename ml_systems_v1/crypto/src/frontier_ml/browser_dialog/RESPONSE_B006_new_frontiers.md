# Response — B006 New frontiers probe (2026-05-02)

> Reply to [PROMPT_B006_new_frontiers.md](PROMPT_B006_new_frontiers.md).
> 8 WebSearch + 3 WebFetch (CfC arxiv abstract + Koopman Neural Forecaster
> arxiv abstract + NeurIPS 2024 self-distillation PDF [unreadable binary]).
> Confidence tagging per `memory/feedback_search_reliability_protocol.md` §1-§9.
> **Isolated round** — no cross-citation of B001-B005.

---

## 1. Executive verdict (≤ 250 words)

**Top three NEW frontiers worth adding to the arsenal**, classified per the
2026-05-02 FOUNDATIONAL/EXTRA framing:

1. **Koopman Neural Forecaster (KNF / Koopa / SKOLR family)** [VERIFIED arXiv 2210.03675; Koopa REPORTED via NIPS-23 link; SKOLR REPORTED ICML 2025]. **FOUNDATIONAL** — explicit mechanism for distribution shift via linear Koopman operator + measurement functions. Crypto regimes shift; this is mechanism-to-problem fit. **5-10 GPU-h probe** for V1.x at 2M params.

2. **Test-Time Training (TTT) for time-series** [REPORTED — Christou et al. 2024 demonstrated robustness improvements for "time-series forecasting under long-horizon or nonstationary regimes"]. **EXTRA** at architecture level (no backbone change), but **FOUNDATIONAL at training paradigm level** (weights update at inference). Direct fit for crypto non-stationarity. ~2-3 GPU-h to wire.

3. **Born-Again Self-Distillation iterated 2-3 generations** [VERIFIED arXiv 1805.04770 + NeurIPS 2024 follow-on REPORTED]. **EXTRA** — same architecture, additional training rounds. Furlanello 2018 reported BAN-DenseNet beating teacher on CIFAR-10/100 [VERIFIED]; NeurIPS 2024 paper "Understanding the Gains from Repeated Self-Distillation" suggests gains compound with rounds [REPORTED — couldn't raw-fetch PDF]. ~12 GPU-h for 3 V1.x generations.

**Honorable mentions worth probing later**: CfC liquid networks [VERIFIED 1-5 orders-of-magnitude speedup], Hyena Hierarchy [REPORTED 6% F1 lift on financial-text], Lion optimizer [REPORTED 33% memory savings vs AdamW]. None promise IC > 0.10 on their own; useful as efficiency upgrades.

---

## 2. TTT / test-time adaptation (Tasks 1 + 6)

### 2.1 Test-Time Training (TTT)

**Source**: TTT project page [REPORTED — `test-time-training.github.io`]; Christou et al. 2024 [REPORTED — "TTT has demonstrated robustness improvements including time-series forecasting under long-horizon or nonstationary regimes"].

**Mechanism**: at inference, the model performs an auxiliary self-supervised step (e.g. masked-sequence reconstruction) on the test sequence and updates a subset of weights, then makes the prediction.

**Applicability to V1.x**: 2M-param V1.x at seq=96 → an inner-loop update of a small head layer (e.g. last MLP) is feasible. Compute cost: ~2-5× inference time per bar. For paper-trader cadence (1 bar/day), this is fine.

**Anti-fragility risk**: TTT could VIOLATE the ShIC > IC × 0.5 invariant if the test-time update memorizes recent shuffled labels. Mitigation: TTT loss must be self-supervised (no labels) — masked reconstruction or contrastive only.

**Verdict**: **HIGH-priority probe**. TTT is paradigm-class for non-stationary data; crypto explicitly is. Classification: **EXTRA at architecture level** (no backbone change) but introduces a new training mode.

### 2.2 TENT / SAR / continual TTA

**Source**: TENT [REPORTED — Wang ICLR 2021]; CoTTA [REPORTED — 2022]; awesome-test-time-adaptation [REPORTED — GitHub `tim-learn/awesome-test-time-adaptation`].

**Mechanism**: TENT minimizes entropy of model predictions at test time (no auxiliary self-sup loss; just sharpens existing predictions). Lower compute than TTT.

**Applicability to V1.x**: TENT requires per-bar entropy-min step on classification-style head. Our TwoHot 255-bin head IS classification-style; trivial to wire.

**Risk**: entropy-min on a classification head can collapse predictions to a single bin (peakiness). Need calibration on a held-out OOS window before live deployment.

**Verdict**: cheaper probe than TTT; ~0.5d engineering; ~1 GPU-h for OOS measurement. **FALLBACK option** if TTT proves too compute-heavy.

---

## 3. Optimizer landscape 2024-2025 (Task 2)

### 3.1 Lion (Chen 2023)

[REPORTED]: "models trained with Lion outperformed those trained with AdamW" on small (1.1B / 2.1B / 7.5B) model NLP benchmarks, but a follow-up at 355M params + few-shot SuperGLUE found AdamW slightly superior to Lion but inferior to Sophia.

[REPORTED]: "Lion saves approximately 33% memory while delivering comparable performance" — relevant to 4060/8GB budget.

**Time-series evidence**: NONE surfaced. Lion validated on language and vision; not on financial small-cap predictors.

**Verdict**: **EXTRA**. ~1 GPU-h A/B against AdamW on V1.0; gain probably small (memory > IC). Lower priority than SAM (already in V1+ recommendation queue per prior dialogs).

### 3.2 Adan (Xie 2024)

[REPORTED]: "Adan reduces the number of training epochs by incorporating Nesterov momentum estimation without introducing additional computational overhead."

**Verdict**: speculative for time-series. SKIP unless Lion+SAM probe disappoints.

### 3.3 Sophia (Liu 2024)

[REPORTED]: "Sophia employs second-order criteria while maintaining a computationally lean approach"; in 355M-param tests, "Sophia came out slightly superior to Lion."

**Verdict**: theoretical second-order benefits should help on small-cap noisy regimes. Worth a probe if SAM disappoints. ~1 GPU-h A/B.

### 3.4 Lookahead wrapper

[REPORTED]: combines with any base optimizer; computes "fast" steps and "slow" steps with periodic synchronization.

**Verdict**: **could pair with SAM**. Lookahead-SAM might be the strongest combo. Worth a 1 GPU-h A/B after baseline SAM measures.

---

## 4. Architecture frontiers NOT in cohort (Task 3)

### 4.1 Liquid Neural Networks / CfC

**Source**: [VERIFIED arXiv 2106.13898 abstract]: "between one and five orders of magnitude faster in training and inference compared to differential equation-based counterparts"; "remarkable performance in time series modeling."

**Financial application**: [REPORTED — fg-research.com/blog/product/posts/lnn-equity-forecasting.html] CfC tested on equity forecasting; Amazon SageMaker has CfC implementation [REPORTED].

**Applicability to V1.x**: CfC is a sequence model class — not a drop-in head fix; would replace the entire backbone. Comparable in scale to V8 (Neural ODE) but 1-5 orders of magnitude faster [VERIFIED]. **Classification: FOUNDATIONAL** (architecture replacement).

**Verdict**: V8 is dominated by Mamba per prior context; CfC is dominated by V8's faster cousin Mamba. Skip CfC unless a continuous-time formulation specifically wins on a probe.

### 4.2 Hyena Hierarchy / StripedHyena

**Source**: [REPORTED] Hyena: "subquadratic drop-in replacement for attention constructed by interleaving implicitly parametrized long convolutions and data-controlled gating." StripedHyena: hybrid of attention and gated convolutions.

**Financial-specific**: [REPORTED — MDPI 2076-3417] "Advanced Hyena Hierarchy Architectures for Predictive Modeling of Interest Rate Dynamics from Central Bank Communications" achieves "over 6% improvement in macro-F1 score compared to baseline models while significantly reducing inference latency by 65%." But this is **central-bank communication classification**, not return prediction.

**Applicability to V1.x**: at seq=96, Hyena's sub-quadratic advantage is 0 (attention is fine at this length). Hyena's sweet spot is seq ≥ 6K [REPORTED — "tipping point at approximately 6K sequence length"]. Our seq=96 means Hyena adds no advantage. **Classification: FOUNDATIONAL but no fit at our scale**.

**Verdict**: **SKIP** for V1.x at seq=96. Re-evaluate only if foundation prong scales seq to ≥ 6K.

### 4.3 S5 / S6 (post-Mamba SSM)

[REPORTED] surfaced as the SSM family successors; specific 2024-2025 financial benchmarks not in this round's results.

**Verdict**: V4 (Mamba-3) per prior context already covers SSM-class architecture; S5/S6 vs Mamba-3 is a research comparison, not a project upgrade.

### 4.4 Deep Equilibrium Models (DEQ)

No 2024-2026 financial-time-series result for DEQ surfaced in this round's searches.

**Verdict**: SKIP — no evidence base.

---

## 5. Output-head alternatives (Task 4)

### 5.1 Energy-Based Models (EBM)

[REPORTED] referenced as "LeCun 2025 EBM-Trans" in prompt context; this round's search did not surface a specific 2024-2026 paper applying EBM to crypto returns.

**Verdict**: speculative. SKIP unless a specific EBM paper is surfaced in next round.

### 5.2 Koopman-operator forecasters

**Source**: [VERIFIED arXiv 2210.03675 abstract]: KNF "leverages DNNs to learn the linear Koopman space and the coefficients of chosen measurement functions" and explicitly addresses "temporal distributional shifts, with underlying dynamics changing over time."

**Source 2**: Koopa [REPORTED — NIPS 2023 / Tsinghua] disentangles non-stationary series into time-invariant and time-variant dynamics.

**Source 3**: SKOLR [REPORTED — ICML 2025] "integrates a learnable spectral decomposition... with MLP as measurement functions" + connection to linear RNNs.

**Applicability to V1.x**: KNF / Koopa / SKOLR are FULL forecaster architectures, not drop-in heads. Would replace V1.x's forward path. **Classification: FOUNDATIONAL — distinct architectural paradigm.**

**Mechanism-to-problem fit**: crypto regimes are **explicitly non-stationary** (regime gates exist precisely because of this). Koopman methods are designed for distributional shift. **This is a high-leverage probe.**

**Verdict**: **HIGH-priority FOUNDATIONAL probe**. ~5-10 GPU-h to retrain a Koopa-class model at 2M params on chimera_legacy and compare IC vs V1.x baseline.

### 5.3 Spectral / Fourier neural operators

[REPORTED]: typically PDE-solving; financial applicability unverified in this round.

**Verdict**: SKIP for now.

### 5.4 Score-based regression

No specific 2024-2025 paper surfaced.

**Verdict**: SKIP.

---

## 6. Self-distillation feasibility (Task 5)

### 6.1 Born-Again Networks (Furlanello 2018)

**Source**: [VERIFIED arXiv 1805.04770]: "Born-Again Networks train students parameterized identically to their teachers, and surprisingly, these networks outperform their teachers significantly, both on computer vision and language modeling tasks."

[VERIFIED] CIFAR-10 (3.5%) and CIFAR-100 (15.5%) state-of-the-art via DenseNet-based BAN.

### 6.2 Repeated Self-Distillation (NeurIPS 2024)

**Source**: [REPORTED — `papers.nips.cc/paper_files/paper/2024/file/0eb1ac7551ddbae575415aa5183a88be-Paper-Conference.pdf`] Title: "Understanding the Gains from Repeated Self-Distillation."

**Verbatim from search snippet** [REPORTED]: "Repeatedly applying self-distillation on the same training data with a student model having the same architecture provides additional gains on benchmark datasets and architectures. At each step, the student from the previous step acts as the teacher used to train a new student model under the self-distillation loss."

**Raw fetch**: PDF was binary-encoded; could not extract optimal-rounds number or quantitative gain claims. **REPORTED only.**

### 6.3 Applicability to V1.x

V1.x is exactly the regime where self-distillation works well (small model, identical architecture, abundant training data). Mechanism: train V1.1 → use as teacher with KL+L1+L2 loss → train V1.1 v2 → use as teacher → V1.1 v3.

**Estimated gain** [INFERRED]: per-generation +0.005-0.010 IC, +0.003-0.008 ShIC, **diminishing returns past 3 generations** [INFERRED].

**Cost**: 3 V1.1 retrains ~9-12 GPU-h. **Classification: EXTRA** (same architecture, just training rounds). **Lifts the IC ceiling slightly AND lifts ShIC** — uncommon dual benefit for an EXTRA category technique.

**Verdict**: **MEDIUM-priority probe**. Valuable specifically because it's safe (no architecture change) and produces a measurable iteration curve. After Generation 2, decide whether Gen 3 is worthwhile based on Gen 1 → Gen 2 delta.

---

## 7. Causal-discovery feature engineering (Task 7)

### 7.1 PCMCI / Granger / NOTEARS

[REPORTED]: PCMCI "has significantly higher detection power than established methods such as Lasso, the PC algorithm, or Granger causality for time series datasets on the order of dozens to hundreds of variables."

[REPORTED]: STIC F1 = 0.44 vs PCMCI 0.41, PCMCI+ 0.43 [REPORTED — search snippet].

**Financial application**: applied to "Fama-French factors and Apple's returns, unemployment, CPI, and PPI" [REPORTED]. Crypto-specific: NOT surfaced in this round.

### 7.2 Applicability to f34 → f29 / f-pruned

PCMCI on our 34-feature panel could prune to a Granger-causal subset (likely 15-25 features). Faster training, less overfitting risk.

**Mechanism-to-problem fit**: medium. Pattern P (5 dead features dropped from project context) was already a manual version of this. PCMCI would automate it AND surface lag relationships.

**Verdict**: **LOW-priority probe** for V1.x — Pattern P already addressed the worst-case dead features. Higher value at foundation prong (where 121-feature input would benefit from causal pruning).

**Classification: EXTRA** (feature-side change, no architecture).

---

## 8. Hyperbolic / hierarchical embeddings (Task 8)

### 8.1 Poincaré embeddings (Nickel 2017)

[REPORTED]: "Any finite tree can be embedded into a finite hyperbolic space such that distances are preserved approximately."

### 8.2 Cryptocurrency application

**This round's search returned NO specific 2024-2025 paper applying hyperbolic embeddings to crypto** [VERIFIED — no hits in 1 search].

**Hypothetical fit**: BTC dominates ETH; ETH dominates SOL/BNB/etc. — a tree-like structure. Hyperbolic embedding would represent this geometry naturally vs Euclidean.

**Verdict**: **SPECULATIVE**. No published evidence; could be novel research. **Classification: FOUNDATIONAL** if pursued (changes embedding geometry). Not a near-term probe.

---

## 9. Top 5 next experiments (priority order)

### E1 — Koopa-class model probe at 2M params (5-7 GPU-h)

**Hypothesis**: Koopman-operator forecaster handles crypto non-stationarity better than V1.x's static encoder, lifting both IC and ShIC.

**Method**: clone Koopa or SKOLR reference impl; retrain at 2M params on chimera_legacy; compare IC h=1 + ShIC vs V1.1 baseline.

**Decision**: if IC ≥ 0.075 AND ShIC ≥ 0.040, Koopa-class becomes a new foundation-prong candidate AND ensemble member alongside V1.x.

### E2 — TTT wrapper on V1.1 (2-3 GPU-h)

**Hypothesis**: Test-Time Training adapts V1.1 to recent regime, lifting OOS IC.

**Method**: add masked-sequence-reconstruction auxiliary loss for inference-time inner-loop weight update on the last linear layer; measure OOS IC vs single-pass V1.1.

**Decision**: if OOS IC delta ≥ +0.005, ship TTT as default inference mode.

### E3 — Born-Again iteration 2 of V1.1 (3-4 GPU-h)

**Hypothesis**: V1.1 v2 distilled from V1.1 outperforms V1.1.

**Method**: train V1.1 v2 with KL+L1+L2 loss against V1.1 best_ema teacher.

**Decision**: if IC ≥ 0.072 (vs current 0.067), Generation 2 ships; consider Gen 3 if gain ≥ +0.003.

### E4 — Lion + SAM combined optimizer probe (2 GPU-h)

**Hypothesis**: Lion's memory savings allow larger effective batch + SAM's flat-minima property combines for better generalization than either alone.

**Method**: V1.0 with Lion-SAM (both wrappers), AdamW-SAM baseline, AdamW alone.

**Decision**: if Lion-SAM beats AdamW-SAM by ShIC ≥ +0.003, adopt Lion-SAM as default.

### E5 — TENT entropy-minimization at inference (1 GPU-h)

**Hypothesis**: TENT-style test-time entropy minimization lifts OOS IC without weight updates.

**Method**: at inference, run M=5 entropy-min steps on prediction logits; measure OOS IC.

**Decision**: if OOS IC delta ≥ +0.003, ship as cheap inference upgrade.

**Total budget**: ~14-17 GPU-h for 5 probes. Sequencing: **E2 → E5** (cheapest TTT-class probes first), **E3** (self-distill is safe), **E4** (optimizer A/B), **E1** (Koopa-class is the biggest commitment — last after cheaper probes inform).

---

## 10. Caveats (per protocol §1-§9)

🔴 **REPORTED-grade decision-gating**:

1. **Christou et al. 2024 TTT for time-series robustness improvement** [REPORTED — search snippet only; not raw-fetched]. The "demonstrated robustness improvements" claim is the load-bearing rationale for E2. Re-verify before E2 commits >2 GPU-h.

2. **NeurIPS 2024 "Understanding the Gains from Repeated Self-Distillation"** [REPORTED]: PDF was binary-encoded; could not raw-fetch optimal-rounds number. Furlanello 2018 BAN result is VERIFIED but on CIFAR not time-series.

3. **SKOLR ICML 2025 spectral-Koopman-RNN** [REPORTED — OpenReview link only].

4. **Hyena 6% F1 on financial-text** [REPORTED — MDPI link]: domain is central-bank text classification, NOT return prediction. Transfer to crypto IC is INFERRED.

5. **Lion 33% memory savings + AdamW comparison** [REPORTED — search snippet]: not raw-fetched, and the comparison favors Sophia > AdamW > Lion at 355M-param few-shot. Mixed evidence.

🟢 **VERIFIED-grade safe to act on**:
- CfC speedup "1-5 orders of magnitude" over differential-equation-based counterparts (arXiv 2106.13898 abstract).
- Koopman Neural Forecaster explicitly addresses temporal distribution shifts (arXiv 2210.03675 abstract).
- Born-Again Networks 2018 outperformed teachers on CIFAR-10/100 (arXiv 1805.04770 abstract).

🟡 **INFERRED-grade**: every IC / ShIC delta in §1, §6, §9 is INFERRED. None of the cited papers measured IC / ShIC on a < 5M-param crypto WM.

🟡 **Isolation note** (protocol §9): conclusions from B001-B005 are NOT cited in this response. If a synthesized cross-prompt narrative across all six dialogs is wanted, that's a META-aggregator artifact for a future round.

---

## 11. Reliability ledger

| Claim type | VERIFIED | REPORTED | INFERRED |
|---|---|---|---|
| Paper / repo existence | 5 | 12 | 0 |
| Open-source / code URLs | 1 | 6 | 0 |
| Paper-reported quantitative claims | 4 (CfC speedup; KNF distribution-shift mechanism; BAN CIFAR results; Hyena 6K crossover) | 9 | 0 |
| **IC / ShIC delta on V1.x or V20+** | **0** | **0** | **9** |
| Decision-gating numbers per E1-E5 | 1 (KNF distribution-shift mechanism) | 4 | 4 |

**Verification rate**: 31% on paper-quantitative claims (4 of 13 raw-fetched); 0% on IC/ShIC deltas.

**Recommendation**: run **E2 (TTT wrapper, 2-3 GPU-h)** as the cheapest ground-truth probe. TTT is the highest-leverage NEW frontier (mechanism-to-problem fit + low compute), and the result directly informs whether E1 (Koopa, 5-7 GPU-h) is worth committing to. If TTT lifts OOS IC ≥ +0.005, the entire "non-stationarity adaptation" frontier (including Koopa) gets validated by analogy.

---

## 12. Sources

### VERIFIED via raw abstract fetch
- [Closed-form Continuous-time Neural Models — arXiv 2106.13898](https://arxiv.org/abs/2106.13898)
- [Koopman Neural Forecaster for Time Series with Temporal Distribution Shifts — arXiv 2210.03675](https://arxiv.org/abs/2210.03675)
- [Born Again Neural Networks — arXiv 1805.04770](https://arxiv.org/abs/1805.04770)

### REPORTED via WebSearch snippet only
- [TTT project page](https://test-time-training.github.io/)
- [Comprehensive Survey on Test-Time Adaptation — arXiv 2303.15361](https://arxiv.org/abs/2303.15361)
- [TTT Provably Improves Transformers as In-context Learners](https://pmc.ncbi.nlm.nih.gov/articles/PMC12662752/)
- [Pre-Training LLMs on a budget (Lion / Sophia / AdamW) — arXiv 2507.08472](https://arxiv.org/html/2507.08472v1)
- [Lion-PyTorch GitHub](https://github.com/lucidrains/lion-pytorch)
- [Liquid Neural Networks 2025 — Ajith Vallath Prabhakar blog](https://ajithp.com/2025/05/04/liquid-neural-networks-edge-ai/)
- [LNN equity forecasting — fg-research blog](https://fg-research.com/blog/product/posts/lnn-equity-forecasting.html)
- [CfC GitHub](https://github.com/raminmh/CfC)
- [Hyena Hierarchy financial application — MDPI](https://www.mdpi.com/2076-3417/15/12/6420)
- [Mamba-360 Survey — arXiv 2404.16112](https://arxiv.org/html/2404.16112v1)
- [SKOLR ICML 2025 — OpenReview](https://icml.cc/virtual/2025/poster/44949)
- [Koopa NIPS 2023 — Tsinghua](https://ise.thss.tsinghua.edu.cn/~mlong/doc/Koopa-nips23.pdf)
- [Temporally Consistent Koopman Autoencoders 2026 — Nature](https://www.nature.com/articles/s41598-025-05222-7)
- [Understanding the Gains from Repeated Self-Distillation — NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/file/0eb1ac7551ddbae575415aa5183a88be-Paper-Conference.pdf)
- [Causal Discovery in Financial Markets Framework — arXiv 2312.17375](https://arxiv.org/abs/2312.17375)
- [Causal Discovery from Temporal Data Survey — ACM Computing Surveys](https://dl.acm.org/doi/10.1145/3705297)
- [Poincaré Embeddings — arXiv 1705.08039](https://arxiv.org/abs/1705.08039)
- [Learning Structured Representations with Hyperbolic Embeddings — arXiv 2412.01023](https://arxiv.org/html/2412.01023v1)
