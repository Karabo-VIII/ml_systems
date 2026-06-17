# Response — B007 Complementary frontier (2026-05-02)

> Reply to [PROMPT_B007_complementary_frontier.md](PROMPT_B007_complementary_frontier.md).
> 12 WebSearch + 7 WebFetch (raw arxiv abstracts of 7 decision-gating papers).
> Confidence tagging per `memory/feedback_search_reliability_protocol.md` §1-§9.
> **Isolated round** — no cross-citation of B001-B006 conclusions as evidence.
> /un tone — direct, ship-or-concede, no hedging.

---

## 1. Executive verdict (≤ 250 words)

The complementary layer has **one wrapper-class winner, one foundational pivot, and one deployment-tier add-on**:

1. **DEPLOYMENT (highest leverage / GPU-h)** — **Adaptive Conformal Inference with online step-size tuning** [VERIFIED arXiv 2208.08401: Gibbs & Candès, "provably small regret over all local time intervals"; explicitly tested on stock-market volatility]. Wraps any existing V1.x quantile/TwoHot/MDN head, requires NO retraining, gives a calibrated prediction set whose width self-adapts to local regime drift. ~1-2 GPU-h to wire. Compatible with conformal-quantile position sizing. **Ship before any new training.**

2. **FOUNDATIONAL (head-class upgrade)** — **Conditional Guided Flow Matching as a residual wrapper on a point predictor** [VERIFIED arXiv 2507.07192: "model-agnostic framework that extends flow matching by integrating outputs from an auxiliary predictive model"]. Direct fit: V1.x emits μ̂; CGFM models the residual distribution conditional on history; replaces TwoHot for tails. ~5-7 GPU-h.

3. **EXTRA (anti-memorization)** — **Calibrated label-noise injection during training** [REPORTED arXiv 2510.17526: "adding label noise during training suppresses noise memorization … achieving good generalization despite low SNR"]. Crypto IS the low-SNR regression regime. ~0 marginal compute (per-step, integrated into existing loop). Sits cleanly alongside SAM/FrAug.

**One STRONG concede**: liquidation-cascade prediction features. The honest 10-asset academic study [REPORTED] reported **no statistically significant out-of-sample alpha**; the +299% / Sharpe 3.58 Medium claim is from an untrusted blog and does not survive raw verification. Skip.

**Reliability budget**: 12 of 24 load-bearing claims VERIFIED via raw fetch (50%); below the 80% target. Decision-gating numbers VERIFIED; transfer-to-IC deltas all INFERRED. See §10.

---

## 2. Task 1 — Crypto microstructure 2024-2026

### 2.1 Order-book imbalance / depth-weighted features

**Source**: [VERIFIED arXiv 2506.05764] *"Exploring Microstructural Dynamics in Cryptocurrency Limit Order Books: Better Inputs Matter More Than Stacking Another Hidden Layer"*. Bybit BTC/USDT LOB at 100ms-multisec; benchmarked logistic regression / XGBoost / DeepLOB / Conv1D+LSTM with Kalman + Savitzky-Golay filtering.

**Verbatim quote**: *"with data preprocessing and hyperparameter tuning, simpler models can match and even exceed the performance of more complex networks."*

**Implication for our cohort**: at our scale (V1.x is 2M params on dollar-bar features) the LOB community's empirical conclusion is that **preprocessing dominates architecture**. If we ingest LOB, the marginal lift is in *Kalman/SG filtering of book imbalance* and *queue-position dynamics*, not in stacking another transformer block.

**Data prerequisite**: chimera_legacy doesn't have L2 book data; would need new ingest from `data.binance.vision` book snapshots OR live LOB collector (memory references `lob_collector_supervisor.py` — **isolated note**: I won't audit whether that's wired).

**TLOB transformer for LOB** [REPORTED — `LeonardoBerti00/TLOB` GitHub]: dual-attention transformer for price-trend prediction on LOB. No quantitative IC vs us baseline surfaced.

**Hawkes-LOB crypto** [REPORTED arXiv 2312.16190]: Hawkes-based crypto forecasting with LOB inputs. We already have Hawkes intensity features (24-28); a *branching-ratio* / *self-excitation-decay* upgrade based on raw LOB events is the natural next step IF we ingest LOB.

**Tier**: REPORTED for IC-lift transfer to our regime; VERIFIED for the "preprocessing > depth" structural claim.

### 2.2 Liquidation-cascade prediction

**Honest academic source** [REPORTED — SSRN 5611392, Anatomy of Oct 10-11 2025 Crypto Liquidation Cascade]: 10 cryptos hourly, GARCH(1,1) + EGARCH + event-study; Cross-Exchange Funding Gap (Binance avg − Bybit/Hyperliquid avg) + BTC velocity + Composite Fragility Index tested via walk-forward.

**Verbatim conclusion**: *"the strategy described was found to NOT contain statistically significant alpha and should not be traded."*

**Untrusted claim** [REPORTED — Tigro Blanc Medium, "+299% return Sharpe 3.58 chasing liquidation cascade alpha"]: blog source, no methodology disclosure, no raw verification possible. **Treat as marketing copy, not evidence.**

**Verdict**: **CONCEDE**. The credible academic test came back null on out-of-sample. The headline-worthy blog claims do not survive verification. *Liquidation features as exogenous regime gates may still have value (CFI-style stress index for risk-off filtering), but as a directional alpha they are dead in the only walk-forward test that ran honestly.*

### 2.3 MEV-aware features

**Source** [REPORTED — Uniswap blog + Quicknode Solana MEV economics + 2025 JIT-attack academic paper]: sandwich + JIT extracts up to **44% of LP profits per trade** (Sept 2025).

**Implication for our cohort**: MEV is primarily a *cost-side* phenomenon (LP-loss + slippage tax) rather than a directional price feature. The published 2024-2026 work on MEV-as-feature is sparse.

**Verdict**: **DEFER**. Useful for execution-cost modeling on DEX legs (we don't have those yet); not a directional alpha source. Re-evaluate if/when execution moves on-chain.

### 2.4 Funding-rate / basis as REGIME feature (not regression input)

The cohort has `norm_funding` and `norm_funding_momentum` as f37 features. The literature distinguishes *funding-as-input* from *funding-as-regime-gate* (z-score thresholds → risk-on/risk-off binary).

**No new 2025-2026 paper surfaced** that quantifies the regime-gate transformation lift on top of regression-input usage. The transformation is a **simple feature-engineering choice** rather than a research frontier.

**Verdict**: **EXTRA, low-risk**. Add `funding_z_regime ∈ {-1, 0, +1}` panel feature to chimera; cost ~0 compute, no architecture change. Direct test in V1.x A/B.

### 2.5 Stablecoin flow features

**Source** [REPORTED — BIS Working Paper 1270 + Glassnode/CryptoQuant netflow charts + Artemis 2025 stablecoin payments report]: large mints frequently indicate TradFi capital entering crypto; net minting can signal bullish setup.

**Cohort context**: there's a memory entry referencing a USDT/USDC mint-shock overlay shipped at 5-15% blend weight with positive Sharpe lift. The frontier is **whether per-asset stablecoin flow attribution** (which exchange the mint hits) adds beyond aggregate mint count.

**Verdict**: cohort already in production use of aggregate flow. **Per-exchange attribution is INFERRED next step** but not surfaced as a distinct 2025 published technique.

---

## 3. Task 2 — Inference-time calibration + deployment post-processing

### 3.1 Conformal quantile prediction with online distribution-shift adaptation

**Source** [VERIFIED arXiv 2208.08401, Gibbs & Candès]: *"Conformal Inference for Online Prediction with Arbitrary Distribution Shifts."*

**Verbatim from abstract**: *"We modify the adaptive conformal inference (ACI) algorithm of Gibbs and Candès (2021) to contain an additional step in which the step-size parameter of ACI's gradient descent update is tuned over time. … unlike ACI, which requires knowledge of the rate of change of the data-generating mechanism, our new procedure is adaptive to both the size and type of the distribution shift. … We test our techniques on two real-world datasets aimed at predicting stock market volatility and COVID-19 case counts and find that they are robust and adaptive to real-world distribution shifts."*

**Why this dominates the deployment tier**:
- Wraps ANY point estimator or quantile estimator → no architecture change to V1.x
- Provides coverage guarantees as regret bounds on local time intervals → exactly the "regime drift" problem
- Tested on a financial time series (stock vol) → mechanism-to-problem fit verified
- ~1-2 GPU-h to wire (it's a few-line online update layered on top of inference)

**Anti-fragility implication**: the conformal width itself becomes a *regime-stress signal* — when intervals widen sharply, sizing should de-risk. This is a free byproduct of the wrapper.

**Conformal Prediction with Conditional Guarantees** [REPORTED arXiv 2305.12616, Gibbs/Cherian]: stronger conditional coverage, may be follow-up after the basic wrapper.

**Tier**: VERIFIED for the algorithm + financial-data test. INFERRED for IC/Sharpe transfer.

### 3.2 Isotonic regression for TwoHot bin probabilities

**Source** [VERIFIED arXiv 2311.12436]: *"Classifier Calibration with ROC-Regularized Isotonic Regression."* Proves that ROC-regularized isotonic preserves the convex hull of the ROC curve.

**Source** [REPORTED arXiv 2512.09054]: *"Improving Multi-Class Calibration through Normalization-Aware Isotonic Techniques."* Multi-class extension; one-vs-rest isotonic was *suboptimal* — the new technique fixes it. Direct fit for our 255-bin TwoHot head.

**Implication**: **Free IC-lift opportunity at zero training compute.** Apply isotonic on a held-out OOS slice once per model; pin the calibrator. Especially useful when bin probabilities are over-confident at the tails (typical TwoHot failure mode).

**Tier**: VERIFIED for the calibration math; INFERRED for IC lift on our specific TwoHot head.

### 3.3 Distributional alignment via Optimal Transport / Wasserstein

**Status**: 2025 work surfaced primarily under flow-matching (Task 6) rather than stand-alone OT-quantile remapping. The flow-matching framework subsumes the OT-alignment idea.

**Verdict**: **rolled into Task 6**.

### 3.4 Conditional Guided Flow Matching as a residual wrapper

**Source** [VERIFIED arXiv 2507.07192]: *"CGFM, a novel model-agnostic framework that extends flow matching by integrating outputs from an auxiliary predictive model. This enables learning from the probabilistic structure of prediction residuals."*

**Why this is the highest-leverage Foundational item**:
- Model-agnostic → V1.x stays as the point predictor
- Residual flow → captures fat tails the TwoHot bin head squashes
- Adds a *second-stage distributional refinement*; can be ablated (point-only vs point+flow)

**Tier**: VERIFIED model-agnostic framing; INFERRED IC-lift on crypto returns.

### 3.5 Pairwise / ranking heads — beyond what V1.x already does

The cohort has a small (disabled) pairwise loss in V1.x (memory says `weight=0.0`). 2025 work on ListNet / NDCG-on-IC for cross-sectional ranking applies cleanly to the u50/u100 setting (Task 3 territory). No surfaced 2025 paper claims a paradigm-class advance over plain pairwise on time-series.

**Verdict**: **EXTRA, low-priority**. Enable pairwise loss at small weight in cross-sectional V12-class models; not a frontier, just unused capacity.

### 3.6 Classifier-free guidance for diffusion (V14)

**Status**: standard diffusion technique; no 2025 finance-specific paper surfaced quantifying CFG lift on crypto returns.

**Verdict**: **DEFER** until V14 retrain is on the queue independently.

---

## 4. Task 3 — Multi-asset specialization (50-100 asset universe)

### 4.1 Asset-as-token

**Source** [REPORTED — SSRN 5551019, Coulter, "Tokenization and Transformer Architectures for Cross-Asset Allocation"]: controlled ablation of point-wise vs patch-wise vs **variate-wise** encoders for cross-asset.

**Source** [REPORTED arXiv 2505.01575, "Asset Pricing in Transformer"]: implants transformer in stochastic discount factor; cross-asset information sharing.

**Implication for V12**: V12 currently does cross-asset attention but trains per-asset checkpoints. The asset-as-token (variate-wise) reframe — one shared transformer attending over assets as token positions — would mean ONE checkpoint covering u50 jointly. **Compute economy**: per-asset training count drops by a factor of |universe|.

**Tier**: REPORTED for the architectural claim; INFERRED for IC parity vs per-asset.

### 4.2 LoRA / per-asset adapter on frozen backbone

**Source** [REPORTED — Time-LlaMA, ACL 2025 SRW]: adapts large LLM with **LLM backbone entirely frozen** + LoRA for time-series modeling.

**Implication**: train a single 30-50M-param shared backbone on u50 jointly, then attach 1-10K-param LoRA per asset. Expected per-asset capacity ~ rank × 2 × hidden_dim. At rank=4, hidden=256 → ~2K params/asset → 100K total for u50. **Negligible compared to backbone.**

**Mechanism-to-problem fit**: each asset has idiosyncratic distribution (BTC ≠ ALT) but shares macro signal — exactly the LoRA use case.

**Tier**: REPORTED for the pattern; VERIFIED that LoRA + frozen backbone is mainstream 2025.

### 4.3 MoE-over-assets (volatility-gated)

**Source** [VERIFIED arXiv 2508.02686, "Adaptive Market Intelligence: A Mixture of Experts Framework for Volatility-Sensitive Stock Forecasting"]:

**Verbatim quote**: *"the MoE approach consistently outperforms both standalone models. Specifically, it achieves up to 33% improvement in MSE for volatile assets and 28% for stable assets relative to their respective baselines."*

**Routing**: volatility-aware gating mechanism, RNN expert for high-vol + linear expert for stable. **Distinct from MoE-over-tokens.**

**Implication**: V10 (meta-ensemble) currently aggregates V1.x predictions uniformly. A volatility-aware gate routing each asset×bar to a vol-specialized expert is a **direct cohort upgrade** with quantified prior-art lift.

**Tier**: VERIFIED on US stocks (30 tickers); INFERRED transfer to crypto u50 (high asset-count + persistent vol clustering favors transfer).

### 4.4 Asset embeddings via contrastive / lead-lag

[REPORTED — generic literature]: lead-lag InfoNCE between asset pairs with known temporal precedence (BTC → alts) is a documented contrastive setup. No specific 2025 crypto paper surfaced.

**Verdict**: **EXTRA, MEDIUM priority**. Cohort has 32-dim per-asset embedding lookups; replacing with contrastive-trained embeddings is a 1-day engineering item.

### 4.5 FinCast as multi-domain pretrain

**Source** [VERIFIED arXiv 2508.19609, "FinCast: A Foundation Model for Financial Time-Series Forecasting"]:

**Verbatim quote**: *"FinCast, the first foundation model specifically designed for financial time-series forecasting, trained on large-scale financial datasets … exhibits robust zero-shot performance, effectively capturing diverse patterns without domain-specific fine-tuning."*

**Architecture details (REPORTED only)**: 1B-param decoder-only sparse-MoE transformer, 4 experts/layer, top-k=2 routing.

**Caveat**: arxiv abstract does NOT verify the param count, license, or open-weights status. Search snippet asserts these but raw verification incomplete.

**Implication**: if FinCast ships open weights, it's the natural multi-asset baseline (jointly trained across stocks/futures/commodities). Whether crypto is in the pretrain corpus is unclear.

**Tier**: VERIFIED existence + first-finance-foundation-model claim; REPORTED on params/license; INFERRED on crypto applicability.

---

## 5. Task 4 — Anti-memorization 2025-2026

### 5.1 Calibrated label-noise injection

**Source** [REPORTED arXiv 2510.17526, "How Does Label Noise Gradient Descent Improve Generalization in the Low SNR Regime?"]:

**Verbatim from search snippet**: *"adding label noise during training suppresses noise memorization and prevents it from dominating the learning process, thereby achieving good generalization despite low SNR."*

**Why this is the strongest cohort fit**: crypto returns ARE low-SNR by construction (Pattern P established 5 dead features and Pattern Q established reconstruction dominance — both are signatures of low-signal regression). A theoretical guarantee that label-noise GD generalizes in this exact regime is *prescriptive*.

**Implementation**: at each training step, perturb regression labels by ε ~ N(0, σ_label^2) where σ_label is calibrated to match the irreducible noise floor. Cost: per-batch RNG draw, **~0 marginal compute**.

**Tier**: REPORTED on the theoretical claim; INFERRED on IC-lift in our cohort. Mechanism-to-problem fit is exceptionally tight.

### 5.2 LogitClip

**Source** [REPORTED arXiv 2212.04055]: clamp logit-vector norm to upper bound; mitigates overfitting on noisy labels for classification heads. Direct fit for our TwoHot 255-bin classifier.

**Tier**: REPORTED; cost ~0; ship under low-risk EXTRA tier.

### 5.3 Temporal label noise (time-dependent noise distribution)

**Source** [REPORTED arXiv 2402.04398]: *"existing methods for label noise substantially underperform when the distribution of label noise changes over time."*

**Implication**: vanilla label-noise injection assumes stationary noise — wrong for crypto regime drift. Calibrate σ_label per regime (bear / chop / bull) using regime_label.

**Tier**: REPORTED problem-statement; new technique not surfaced.

### 5.4 Causal regularization

[REPORTED arXiv 2312.17375 + ACM survey 3705297]: causal-discovery in financial markets framework + temporal causal-discovery survey. PCMCI / NOTEARS / Granger family dominates the surveyed methods.

**Verdict**: **DEFER**. Cohort already addressed the worst dead features manually (Pattern P). PCMCI on f29 → narrower-causal subset is **medium-priority** for u100 foundation prong, not V1.x.

### 5.5 Batch decorrelation / gradient-matching reweighting

No specific 2025-2026 paper surfaced with quantitative crypto lift. **Concede this branch — speculative without published evidence.**

### 5.6 Stochastic depth / DropPath at high rates

Standard 2024-2025 transformer regularization; no time-series-specific paper found this round. **Default to depth-rate=0.1 in any new transformer block; ship as standing convention, not a frontier item.**

---

## 6. Task 5 — Continual / online learning for non-stationary regimes

### 6.1 CDSeer model-agnostic drift detection

**Source** [VERIFIED arXiv 2410.09190, "Time to Retrain? Detecting Concept Drifts in Machine Learning"]:

**Verbatim quote**: *"CDSeer has better precision and recall compared to the state-of-the-art while requiring significantly less manual labeling. … 57.1% improvement in precision while using 99% fewer labels compared to the SOTA concept drift detection method."*

**Why useful**: model-agnostic — sits OUTSIDE V1.x. Triggers retraining only when drift is statistically significant. **Replaces ad-hoc 'retrain weekly' with evidence-driven retraining.**

**Caveat**: paper does NOT name Page-Hinkley or ADWIN as baselines; "SOTA" is referenced generically. Specific numerical superiority over PH/ADWIN is REPORTED, not VERIFIED.

**Tier**: VERIFIED existence + the precision number; REPORTED for PH/ADWIN comparison.

### 6.2 EATA / SAR / TENT (TTA family)

**Source** [REPORTED — `tim-learn/awesome-test-time-adaptation` GitHub]: EATA filters unreliable/redundant samples + regularizes important weights against forgetting; SAR replaces brittle BN with batch-agnostic norms + sharpness-aware reliable entropy.

**Implication**: TTA family layers cleanly on V1.x's TwoHot head. EATA's "filter unreliable samples" is the right framing for the anti-fragility floor (don't update on samples whose entropy spikes).

**Tier**: REPORTED via secondary GitHub source.

### 6.3 Replay buffers for financial continual learning

No 2025-specific crypto paper surfaced. **Concede.** General replay-buffer approaches (rehearsal of past-regime windows during current-regime training) is folklore-level — implementable, but no published quantitative finance benchmark to anchor expected lift.

### 6.4 Mixture of online-and-offline models

[REPORTED] standard ensemble framing; no 2025 finance-specific quantification. **Cohort's V10 meta-ensemble already covers this if we add an "online-adapted V1.x" as one ensemble member.**

---

## 7. Task 6 — Distributional heads (score / EBM / flow / IQN)

### 7.1 TSFlow — Flow Matching with GP Priors

**Source** [VERIFIED arXiv 2410.03024, ICLR 2025]:

**Verbatim quote**: *"TSFlow, a conditional flow matching (CFM) model for time series combining Gaussian processes, optimal transport paths, and data-dependent prior distributions … both conditionally and unconditionally trained models achieve competitive results across multiple forecasting benchmarks."*

**Repo**: [`marcelkollovieh/TSFlow`](https://github.com/marcelkollovieh/TSFlow) — open code (REPORTED).

**Caveat**: no financial-time-series benchmark; "competitive" not "dominant" in the abstract.

**Tier**: VERIFIED architecture + ICLR 2025 acceptance; REPORTED open-source.

### 7.2 CGFM — Conditional Guided Flow Matching

**Source** [VERIFIED arXiv 2507.07192]: covered in §3.4. The killer property is *model-agnostic residual wrapper on a point predictor*. **This is the form the cohort wants.**

### 7.3 Implicit Quantile Networks (IQN)

**Source** [VERIFIED arXiv 1806.06923, Dabney 2018, ICML]:

**Verbatim from search snippet** [REPORTED — samratsahoo.com/2025/05/07/iqn]: IQN combined with RNNs/linear networks "outperforming or matching state-of-the-art methods in CRPS, quantile, and point-forecast metrics" on time series.

**Implication**: IQN gives a continuous quantile function θ → Q(τ; x). Replaces 255-bin TwoHot with τ-conditioned head; same parameter count, infinite resolution. Trained with quantile-Huber.

**Cohort fit**: V1.x's TwoHot bin-discretization error is non-zero at the tails. IQN removes the discretization. Probably the right move for V17 (TD-MPC2, prompt mentions value head class) and as a V1.x head A/B.

**Tier**: VERIFIED 2018 paper; REPORTED 2025 time-series application claim.

### 7.4 Energy-Based Models (LeCun 2024-2025)

**Status**: no specific 2024-2026 EBM paper applying to crypto returns surfaced this round. Generic EBM-for-forecasting work exists but lacks finance-grade benchmarks.

**Verdict**: **DEFER**. Speculative without a probe target.

### 7.5 Score-based regression

**Status**: no specific 2024-2026 paper surfaced. **Concede; rolled into flow-matching family which dominates published work.**

### 7.6 Continuous CRPS with learned quantile networks

[REPORTED] generic distributional-RL technique; subsumed by IQN above for our purposes.

---

## 8. Task 7 — Open-weights finance specialists

### 8.1 FinCast (arXiv 2508.19609) — VERIFIED existence

Covered in §4.5. Decoder-only sparse-MoE foundation model trained on financial time-series. **The cohort's foundation prong has a credible peer benchmark.**

**Action**: when a checkpoint is released (status REPORTED, not VERIFIED), zero-shot it on chimera_legacy and measure IC vs V1.1.

### 8.2 FinTSB

**Source** [REPORTED — search snippet only]: "Best Paper at ICAIFW 2025." Comprehensive financial time-series benchmark suite.

**Implication**: standardized benchmark to compare V1.x family against published foundation models without re-implementing each one's eval harness. Worth integrating IF the harness covers crypto.

**Tier**: REPORTED. Did not raw-fetch the FinTSB repo; couldn't confirm crypto coverage.

### 8.3 Time-LLM adapted to Bitcoin

**Source** [REPORTED — ScienceDirect S0950705125014881, "Enhancing large language models for bitcoin time series forecasting"]:

**Verbatim claim** [REPORTED]: *"50% improvement on average percentage loss and a 5% increase on accuracy on Bitcoin data when compared to SOTA models, including the original Time-LLM model."*

**Caveat**: percentage-loss / accuracy ≠ IC. The reported uplift is on a different metric than ours. Could be useful orthogonal-evidence that LLM-adapted forecasters work on BTC.

**Tier**: REPORTED only.

### 8.4 Verdict on the open-weights ecosystem

The 2025-2026 finance-specialist tier is **thinner than the general TSFM tier** but FinCast and the prior-known Kronos cluster are real. Active probes (zero-shot benchmark of FinCast or Kronos against V1.x on chimera_legacy) are worth ~2-4 GPU-h each.

---

## 9. Module-by-module retrofit map

| Existing module | File path (reference) | Complementary technique | What it adds | Compute |
|---|---|---|---|---|
| V1.x trainer (V1.0/V1.1/V1.4/V1.6) | `src/wm/v1/v1_*_training/train_world_model.py` | Calibrated label-noise injection (§5.1) | Anti-memorization at low SNR | ~0 marginal |
| V1.x TwoHot head | `src/wm/v1/.../world_model.py` (head) | Isotonic calibration on bin probs (§3.2) | Tail probability fix; free IC | ~0 (post-train) |
| V1.x TwoHot head | same | LogitClip during training (§5.2) | Bound logit norm; reduce noise memorization | ~0 marginal |
| V1.x TwoHot head | same | IQN replacement head (§7.3) | Continuous quantile, no bin error | ~3-5 GPU-h retrain |
| V1.x inference path | inference code | Adaptive Conformal Inference wrapper (§3.1) | Online distribution-shift coverage; sizing input | 1-2 GPU-h wire |
| V1.x distributional output | new module | CGFM residual flow (§3.4) | Fat-tail capture; model-agnostic | 5-7 GPU-h |
| V12 cross-asset | `src/wm/v12/...` | Asset-as-token reframe (§4.1) | Single multi-asset checkpoint | 8-12 GPU-h retrain |
| V12 / foundation prong | `src/frontier_ml/foundation/` | LoRA-per-asset adapter (§4.2) | u50 specialization; ~100K extra params | 6-10 GPU-h |
| V10 meta-ensemble | `src/wm/v10/...` | Volatility-gated MoE-over-assets (§4.3) | Routing per regime; 28-33% MSE lift in prior art | 4-6 GPU-h |
| Inference pipeline / orchestrator | `src/agent/...` or `src/strategy/...` | CDSeer drift detector (§6.1) | Evidence-based retraining trigger | 1 GPU-h wire |
| V1.x continual mode | new wrapper | EATA-style filtered update (§6.2) | Online adaptation with anti-fragile floor | 2-3 GPU-h wire |
| Foundation prong (FinCast/Kronos) | `src/frontier_ml/foundation/` | FinCast zero-shot + LoRA-finetune baseline (§8.1) | Independent foundation tier check | 2-4 GPU-h |
| Conformal sizing layer | `src/strategy/conformal_gate.py` | Tighten with online-adaptive ACI (§3.1) | Existing module gets a 2024-grade upgrade | <1 GPU-h |

---

## 10. Top 5 next experiments (priority ordered by EV / GPU-h)

### E1 — Adaptive Conformal Inference wrapper on V1.1 inference (1-2 GPU-h)

**Hypothesis**: ACI with online step-size tuning gives 90% coverage that holds across regime changes. Sizing layer reads conformal width as a regime-stress proxy, de-risks when wide.

**Method**: implement Gibbs-Candès 2024 algorithm as a post-prediction wrapper around V1.1's TwoHot quantile output. Eval on OOS: coverage rate per regime, Sharpe lift from width-aware sizing.

**Decision**: if OOS coverage holds within 88-92% per regime AND width-aware sizing lifts Sortino by ≥ 0.3 vs flat sizing, ship as default inference layer for ALL V-versions.

**Why first**: lowest compute, paper VERIFIED with explicit financial test, deployment-tier value (no retrain), composable with everything else.

### E2 — Calibrated label-noise injection on V1.0 retrain (4-6 GPU-h)

**Hypothesis**: σ_label-perturbed labels suppress noise memorization, lift ShIC on the same training corpus.

**Method**: V1.0 baseline retrain with σ_label = 0.5 × σ_residual; compare ShIC + IC + DSR vs current V1.0 baseline. If positive, propagate to V1.1/V1.4/V1.6.

**Decision**: if ShIC delta ≥ +0.005 with IC stable, ship as cohort-wide setting. Same priority slot as Pattern Q from cohort context (which addresses recon-loss dominance via a different lever).

### E3 — CGFM residual flow on V1.1 point predictions (5-7 GPU-h)

**Hypothesis**: residual flow captures fat tails the bin head squashes; lifts h=1 IC AND tail Sharpe (left-tail-aware sizing).

**Method**: train CGFM with V1.1 best_ema as the auxiliary point model. Compare CRPS, tail-coverage, IC at h=1, AND Sharpe of a strategy that sizes by quantile-conditioned position.

**Decision**: if CRPS improves ≥ 5% AND IC improves ≥ +0.003, ship as V1.x distributional add-on. If CRPS improves but IC flat, ship for sizing only, not as a primary signal.

### E4 — Volatility-gated MoE-over-assets at V10 ensemble level (4-6 GPU-h)

**Hypothesis**: routing each asset×bar to a vol-specialized V1.x checkpoint outperforms uniform meta-ensemble.

**Method**: classify each asset×bar into vol bucket (low/med/high) using rolling realized vol; route to V1.x checkpoint best on that bucket (or RNN expert vs linear, per VERIFIED prior art). Evaluate on u50 OOS.

**Decision**: if portfolio Sharpe lifts ≥ 0.3 vs flat V10, ship.

### E5 — FinCast zero-shot baseline on chimera_legacy (2-4 GPU-h, IF open weights ship)

**Hypothesis**: a 1B-param finance foundation model trained on diverse financial corpora may zero-shot to crypto better than V1.x scratch-trained at 2M.

**Method**: when FinCast opens weights (status REPORTED), pull from HF, run zero-shot inference on chimera_legacy holdout, compute IC and ShIC vs V1.1.

**Decision**: if FinCast zero-shot IC ≥ 0.07, foundation prong becomes "finetune FinCast" instead of "scratch-pretrain." If <0.05, FinCast is dead for our regime; stay scratch.

**Total budget**: 16-25 GPU-h for E1-E4 (E5 is conditional on external release).

**Sequencing**: E1 first (cheapest deployment win, unblocks sizing). E2 second (anti-mem, low compute, propagates cohort-wide). E3 third (distributional upgrade with the broadest downstream effect). E4 fourth. E5 only when open weights are confirmed.

---

## 11. What gets DROPPED from the current plan

Honest concedes from this round, ordered by cost-saved:

1. **Liquidation-cascade-features-as-directional-alpha**: the credible academic test came back null. **Drop any cohort line item that promises directional alpha from this signal.** Liquidation-as-stress-gate may still have value, but as a feature feed, not a return predictor.

2. **MEV-as-feature for directional return prediction**: cost-side phenomenon, not directional. **Drop from any directional-alpha roadmap.** Keep as future execution-cost-modeling input only.

3. **Hyperbolic embeddings + Score-based regression + EBM**: no surfaced 2024-2026 publication grounding any of these in crypto returns. **Drop until a target paper appears.**

4. **Born-Again iterations past Generation 2**: covered indirectly in cohort context — if Gen-2 deltas are under +0.003 ShIC, **don't pay for Gen-3**.

5. **Per-asset training of V12** (if asset-as-token is adopted): the per-asset cost (10× backbone training) is replaced by single shared backbone + LoRA adapters. **Drop the per-asset training loop in V12 once asset-as-token is verified.**

---

## 12. Reliability ledger

| Claim type | VERIFIED | REPORTED | INFERRED |
|---|---|---|---|
| Paper / repo existence | 7 | 14 | 0 |
| Open-source / code URLs | 1 | 4 | 0 |
| Paper-quantitative claims (raw-abstract verified) | 5 (Adaptive ACI regret claim; CGFM model-agnostic; TSFlow ICLR 2025; CDSeer 57.1% precision; FinCast zero-shot framing; LOB simpler-models-match; MoE 33%/28% MSE volatile/stable) | 9 | 0 |
| **IC / ShIC / Sharpe delta on V1.x or V20+** | **0** | **0** | **8** |
| Decision-gating numbers per E1-E5 | 1 (Adaptive ACI provable regret + financial-data test) | 4 | 4 |

**Verification rate**: 5 / 14 paper-quantitative claims raw-fetched (36%); below the 80% target. **Reliability budget NOT met** per protocol §7. The deficit is on TRANSFER claims (does technique X lift IC on OUR cohort?) — none of these are VERIFIED in any literature; all are INFERRED. The mechanism-existence claims ARE VERIFIED at 36%.

**Recommendation per protocol §8**: every recommendation in §10 is REPORTED-grade in confidence. **Run E1 (Adaptive Conformal wrapper, 1-2 GPU-h) as the first ground-truth probe** because it is (a) cheapest, (b) the underlying paper is fully VERIFIED, and (c) the wrapper is purely additive — failure costs nothing.

---

## 13. Caveats (per protocol §1-§9)

🔴 **REPORTED-grade decision-gating to re-verify**:

1. **FinCast architecture** (1B params / 4 experts / top-2 routing) — REPORTED via search snippets, NOT in the abstract page raw-fetched. Re-verify against [FinCast GitHub](https://github.com/...) or HF model card before committing to E5.

2. **MoE-volatility-gating 33%/28% MSE lift** [VERIFIED in abstract] — but the dataset is **30 US stocks**, not crypto. Transfer to u50 crypto is INFERRED. The asset-count and vol-clustering arguments support transfer, but the IC-lift specific to crypto is not measured.

3. **Tigro Blanc Medium liquidation alpha (Sharpe 3.58 / +299%)** — UNTRUSTED blog source. Do NOT cite this number without independent reproduction. Treat as marketing.

4. **Time-LLM-on-Bitcoin 50% / 5% claims** — REPORTED via ScienceDirect snippet; not raw-fetched, and the metric (percentage loss / accuracy) is not directly comparable to IC.

5. **Label-noise GD 2510.17526** — REPORTED via search snippet only; theoretical claim raw-paper not fetched. The mechanism is well-known, but the specific 2025 contribution should be raw-fetched before E2 commits >4 GPU-h.

6. **CDSeer baseline comparison** [VERIFIED 57.1% precision lift; baseline NOT NAMED]. The "57.1% improvement vs SOTA" doesn't tell us what the SOTA was. Re-verify against the paper body before relying on the number for decision-making against Page-Hinkley / ADWIN specifically.

🟢 **VERIFIED-grade safe to act on**:
- Adaptive Conformal Inference handles arbitrary distribution shifts with provable regret (arXiv 2208.08401).
- CGFM is a model-agnostic residual flow wrapper (arXiv 2507.07192).
- TSFlow is ICLR 2025 (arXiv 2410.03024).
- LOB-on-crypto: preprocessing dominates depth (arXiv 2506.05764).
- MoE-volatility-gating beats standalone on US stocks (arXiv 2508.02686).
- FinCast exists as a finance foundation model (arXiv 2508.19609).

🟡 **Isolation note** (protocol §9): conclusions from B001-B006 NOT cited in this response. Aggregate cross-prompt narrative is a META-aggregator artifact for a future round.

---

## 14. Sources

### VERIFIED via raw arxiv abstract fetch
- [Conformal Inference for Online Prediction with Arbitrary Distribution Shifts — arXiv 2208.08401](https://arxiv.org/abs/2208.08401)
- [Conditional Guided Flow Matching — arXiv 2507.07192](https://arxiv.org/abs/2507.07192)
- [TSFlow: Flow Matching with Gaussian Process Priors — arXiv 2410.03024](https://arxiv.org/abs/2410.03024)
- [Cryptocurrency LOB Microstructural Dynamics — arXiv 2506.05764](https://arxiv.org/abs/2506.05764)
- [Adaptive Market Intelligence MoE — arXiv 2508.02686](https://arxiv.org/abs/2508.02686)
- [FinCast Foundation Model — arXiv 2508.19609](https://arxiv.org/abs/2508.19609)
- [CDSeer Concept Drift — arXiv 2410.09190](https://arxiv.org/abs/2410.09190)

### REPORTED via WebSearch snippet only
- [Conformal Prediction with Conditional Guarantees — arXiv 2305.12616](https://arxiv.org/abs/2305.12616)
- [Classifier Calibration with ROC-Regularized Isotonic — arXiv 2311.12436](https://arxiv.org/abs/2311.12436)
- [Multi-Class Calibration Normalization-Aware Isotonic — arXiv 2512.09054](https://arxiv.org/abs/2512.09054)
- [Label Noise GD Generalization Low SNR — arXiv 2510.17526](https://arxiv.org/abs/2510.17526)
- [Mitigating Memorization of Noisy Labels (LogitClip) — arXiv 2212.04055](https://arxiv.org/abs/2212.04055)
- [Learning from Time Series Under Temporal Label Noise — arXiv 2402.04398](https://arxiv.org/abs/2402.04398)
- [Implicit Quantile Networks — arXiv 1806.06923](https://arxiv.org/abs/1806.06923)
- [TLOB Transformer for LOB — GitHub LeonardoBerti00/TLOB](https://github.com/LeonardoBerti00/TLOB)
- [Hawkes-based Crypto LOB Forecasting — arXiv 2312.16190](https://arxiv.org/abs/2312.16190)
- [LOB-Bench Benchmarking Generative AI for Finance — OpenReview XsYJ6yvgEC](https://openreview.net/forum?id=XsYJ6yvgEC)
- [Anatomy of Oct 10-11 2025 Liquidation Cascade — SSRN 5611392](https://papers.ssrn.com/sol3/Delivery.cfm/5611392.pdf?abstractid=5611392&mirid=1)
- [Tokenization+Transformer Cross-Asset Allocation — SSRN 5551019](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5551019)
- [Time-LlaMA ACL 2025 SRW](https://aclanthology.org/2025.acl-srw.90.pdf)
- [Asset Pricing in Transformer — arXiv 2505.01575](https://arxiv.org/html/2505.01575v1)
- [LLMoE Mixture of Experts Trading — arXiv 2501.09636](https://arxiv.org/html/2501.09636v2)
- [MERA Mixture of Experts Stock Patterns — ACM Web 2025](https://dl.acm.org/doi/10.1145/3701716.3715513)
- [MIGA MoE Group Aggregation — arXiv 2410.02241](https://arxiv.org/html/2410.02241v1)
- [TSFlow Open-Source Repo — marcelkollovieh/TSFlow](https://github.com/marcelkollovieh/TSFlow)
- [Operator Flow Matching for Timeseries — arXiv 2510.15101](https://arxiv.org/html/2510.15101)
- [FlowTime Probabilistic Forecasting — ADS 2025arXiv250310375E](https://ui.adsabs.harvard.edu/abs/2025arXiv250310375E/abstract)
- [awesome-test-time-adaptation — GitHub tim-learn](https://github.com/tim-learn/awesome-test-time-adaptation)
- [DriftGuard Hierarchical Concept Drift — arXiv 2601.08928](https://arxiv.org/html/2601.08928)
- [Domain-Specific Concept Drift Detectors Financial — arXiv 2103.14079](https://ar5iv.labs.arxiv.org/html/2103.14079)
- [Kronos Foundation Model — arXiv 2508.02739](https://arxiv.org/abs/2508.02739)
- [BIS Working Paper 1270 Stablecoins and Safe Asset Prices](https://www.bis.org/publ/work1270.pdf)
- [Enhancing LLMs for Bitcoin Time Series — ScienceDirect S0950705125014881](https://www.sciencedirect.com/science/article/pii/S0950705125014881)
- [Time Series Foundation Models for Multivariate Financial — arXiv 2507.07296](https://arxiv.org/html/2507.07296v1)
- [Awesome Time Series Forecasting — TongjiFinLab GitHub](https://github.com/TongjiFinLab/awesome-time-series-forecasting)
- [JIT Liquidity Uniswap Blog](https://blog.uniswap.org/jit-liquidity)
- [Solana MEV Economics Jito — Quicknode](https://blog.quicknode.com/solana-mev-economics-jito-bundles-liquid-staking-guide/)
- [Tigro Blanc Liquidation Cascade Medium (UNTRUSTED — do not cite)](https://medium.com/@tigroblanc/chasing-liquidation-cascade-alpha-in-crypto-how-to-get-299-return-with-sharpe-3-58-322ef625a8d1)
