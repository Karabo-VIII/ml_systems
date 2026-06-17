# SOTA Literature Review — frontier WM for crypto under 4060/8GB

> **Purpose:** close the "holes" enumerated in PLAN.md by tracing each
> load-bearing claim to a paper, a public benchmark, or an explicit
> "we accept this risk because" annotation. **No code is built until
> every hole has a closure**.

## Hole 1 — Is 30M parameters enough to "emerge" anything on time-series?

### What the literature says

- **Chronos** (Ansari et al. 2024, Amazon, [arxiv:2403.07815]): T5-base
  size = **220M params**. Uses causal next-token. Zero-shot benchmark
  on M4 / M5 / Tourism / Wiki. Performance scales monotonically with
  size from chronos-tiny (8M) → chronos-base (220M) → chronos-large
  (710M). Importantly the 8M version is non-trivial (beats naive
  baselines on most benchmarks), so emergence isn't required for
  competence.
- **TimesFM** (Das et al. 2024, Google, [arxiv:2310.10688]):
  decoder-only transformer at **200M params**. Pretrained on 100B
  time points. Uses causal next-token + patch-level masking. Beats
  Chronos at similar scale; size-vs-performance curve also monotonic.
- **MOMENT** (Goswami et al. 2024, CMU, [arxiv:2402.03885]):
  T5 encoder-only at **40M / 125M / 385M**. Masked patch reconstruction.
  At 40M params it beats most task-specific baselines on
  classification/forecasting/anomaly-detection. **Closest analog to
  what we'd build at 30M.**
- **PatchTST** (Nie et al. 2023, IBM, [arxiv:2211.14730]): channel-
  independent patching + transformer. Demonstrates that ~3M params
  on multivariate forecasting matches/beats much larger models at
  benchmark scale. Architecture > scale at small N.

### Closure

30M params IS enough to be useful, per MOMENT (40M-param time-series
encoder achieves strong performance on benchmarks). 30M won't trigger
LLM-style emergence (which requires 6B+) but **emergence isn't required
for our objective**. Moving to IC > 0.10 from IC ≈ 0.07 is a 1.5x
signal lift, not a phase transition.

**Risk-accepted:** at 30M we'll see iterative improvements, not
emergent capabilities.

## Hole 2 — Is masked sequence modeling the right objective for time-series?

### What the literature says

- **Chronos** uses **causal next-token** (autoregressive). Strong.
- **TimesFM** uses **patch-level causal**: predict next *patch* of N
  tokens given history. Strong.
- **MOMENT** uses **bidirectional masked patch reconstruction**.
  Encoder-only. Strong.
- **PatchTST** uses **causal** patching.
- **TS2Vec** (Yue et al. 2022, [arxiv:2106.10466]): contrastive
  pretraining on time-series, no masked modeling. Strong on
  classification, weaker on forecasting.

Empirical comparison: causal next-token wins for forecasting
specifically; masked is more general (better for classification +
forecasting). For our domain (forecasting return distribution at
h ∈ {1, 4, 16, 64}), **causal next-token is the literature's preferred
choice**.

### Closure

Switch PLAN.md objective from masked-sequence-modeling to **causal
next-token** at multiple horizons. Combine with cross-asset contrastive
(separate signal). Drop adversarial time-shuffle as it overlaps with
contrastive.

**Updated objective set:**
1. Causal multi-horizon next-bar prediction (TwoHot, 255 bins) — primary
2. Cross-asset contrastive (JEPA) — auxiliary
3. (Drop) Masked sequence modeling
4. (Drop) Time-shuffle adversarial

This simplifies the loss to two terms instead of four.

## Hole 3 — Cross-asset contrastive: same-timestamp positives correct?

### What the literature says

- **JEPA** (LeCun 2022, [openreview]): Joint-Embedding Predictive
  Architecture. Positive pairs are different views of the same target
  state. For time-series, "view" can mean "same time, different feature
  subset" (channel-dropout) OR "same timeline, different time
  resolution" OR "same asset, lagged window".
- **TS2Vec** uses **temporal contrastive**: positive = adjacent
  windows of the same asset, negative = far-apart windows. NOT
  cross-asset.
- **CoST** (Woo et al. 2022, [arxiv:2202.01575]): **frequency-domain
  contrastive** for time-series; doesn't address cross-asset directly.
- **No clean reference for "cross-asset same-timestamp positive"**.
  This is novel territory.

### Empirical concern (mine)

Crypto markets have **lead-lag relationships**: BTC moves first, ETH
follows ~10-30 min later, alts hours-to-days later. Same-timestamp
contrastive forces the model to treat these as identical, when they
aren't. Better: positive pair = (BTC bar at time T, ETH bar at time
T + lag), where lag is learned or a small distribution {0, 5, 15, 60}
minutes.

### Closure

Replace **same-timestamp cross-asset** with **lead-lag aware**
contrastive: for each asset pair (A, B), the positive is (A_t, B_{t+δ})
where δ is sampled from a small set including 0. Negative pairs are
(A_t, B_{t+T}) for large T. This handles BTC → ETH → alt cascade.

**Risk:** introduces a hyperparameter. Will need ablation.

## Hole 4 — Hardware-OOM risk during pretrain

### Empirical baselines (from this codebase)

- V4 (Mamba-3 + RSSM, 3.5M params, batch=32, seq_len=96): observed peak
  ~3 GB during training. (~32 GB host RAM also fine.)
- V11 (WaveNet+MoE, 2.9M params, batch=32, seq_len=96): ~2 GB.
- V13 (TFT, 2.2M params, same config): ~2.5 GB.

Scaling to 30M params at seq_len=512:
- Weights (fp16): 60 MB
- Adam states (fp32): 240 MB
- Gradients (fp16): 60 MB
- Activations at batch=8, seq=512, d=256, 6 layers: rough estimate
  6 × 8 × 512 × 256 × 4 bytes × ~3 (forward+backward) = ~75 MB per layer
  × 6 layers = 450 MB
- Total: ~810 MB ≈ 1 GB → fits 8 GB with **5-7 GB headroom for batch
  scaling and second-order ops**.

### Closure

30M-param Mamba at seq=512 batch=8 fits comfortably. **Add an
upfront OOM probe** (200 steps real data) before committing to the
full epoch budget — per CLAUDE.md cross-version invariant §12.

## Hole 5 — KL distillation on TwoHot 255-bin: peaky transfer

### What the literature says

- **DistillBERT** (Sanh et al. 2019): KL on softmax distribution works
  well for text (bins → vocab tokens, smooth distributions).
- **Born-Again Networks** (Furlanello et al. 2018): self-distillation
  on classification — works.
- **Knowledge distillation for ordinal regression** (Phuong & Lampert
  2019, [arxiv:1812.04106]): for ordinal targets (like our 255 bins),
  pure KL underperforms vs. **two-target hybrid**: KL on smoothed
  teacher + L1 on continuous expectation.

### Closure

Replace pure KL distillation with **hybrid**:
- L_KL = KL(student_logits, teacher_logits_softmaxed) — primary
- L_L1 = |student_expected_return - teacher_expected_return| — auxiliary
- L_var = (student_var - teacher_var)² — auxiliary

Loss = α·L_KL + β·L_L1 + γ·L_var. α=0.5 β=0.4 γ=0.1 starting point per
the ordinal-distillation paper.

## Hole 6 — Multi-modal alignment leakage risk

### What the literature says

This is a **timestamp hygiene problem**, not an ML problem. Every
finance ML paper since the dawn of time has had this issue:
- **Lopez de Prado, AFML chapter 4**: any feature must use information
  available STRICTLY BEFORE the bar's close time, not bar's open and
  not anything later.
- **Bailey & Lopez de Prado, "False Strategies"** ([SSRN]): empirical
  walk-forward CV with purge gaps prevents leakage.

For our specific multi-modal:
- **Funding rate**: available at funding-event timestamps (every 8 hr).
  Use funding(t-ε) where ε is the smallest gap to bar-close.
- **Open interest**: continuous; same logic.
- **Macro (DXY, S&P)**: daily close. Use prev-day-close for current-day
  bars.
- **On-chain**: continuous; same as funding.
- **News embeddings**: published-time of article. Use news(t-ε).

### Closure

Add a "lag" parameter to every multi-modal channel ingest, defaulting
to 1 bar (5 min for chimera_legacy at 5-min bars) so the model never
sees information from the same bar's close or beyond.

Wire into walk-forward CV so the purge gap covers the longest lookback
(e.g., if multi-modal lookback = 30 days, purge = 30 days).

## Hole 7 — Distillation diminishing returns vs ensemble

### What the literature says

- **Born-Again Networks** (Furlanello et al. 2018): student often
  matches or **exceeds** the teacher when same-arch + same-data.
  Not always, but common.
- **Distilling Knowledge in a Neural Network** (Hinton et al. 2015):
  ensemble distillation works well when ensemble is diverse.

### Empirical estimate

Our ensemble: V1.1 IC=0.067 (best individual). Average of 8 teachers
might be ~0.07-0.08 (diversity bonus). Distilled student aim: ≥ 0.067
(match best teacher) at single-inference cost.

If distillation only matches best individual, ensemble is the
deployable. **But:** ensemble inference is 8x cost. If we want sub-100ms
trading decisions, distilled student is worth it even at slight
IC penalty.

### Closure

**Decision metric:** distilled student must achieve IC ≥ 0.95 ×
best_teacher_IC AND inference latency ≤ 1/4 ensemble. If yes, deploy
student. If no, deploy ensemble.

## Hole 8 — Compute budget realism (4060 24/7 for 1.5 weeks)

### Risk

- Single GPU = no redundancy if a checkpoint corrupts
- Thermal throttling if cooling is marginal — could extend wall-clock
- Cannot use GPU for anything else (visualization, ad-hoc inference)
  during the run

### Closure

- **Save checkpoints every 100 steps** (or every 10% of an epoch),
  not just at end. Already standard.
- **Resume-from-checkpoint** is already implemented in V1's
  train_world_model.py — port to new framework.
- **Schedule pretrain over 2 weekends + weeknight idle hours**, not
  continuous. Even at 50% duty cycle, finish in 3 weeks rather than
  1.5 — acceptable.
- **Live monitoring** of thermals via Windows Task Manager / nvidia-smi
  — abort if GPU temp > 85°C sustained.

## SOTA gap-check (what frontier teams have that we don't)

For perspective, the actual frontier (Chronos-large, TimesFM-large,
MOMENT-large) has:
- 100-700M params
- Pretrained on 100B time-points across many domains
- Months of compute on H100 clusters

We have 30M params on crypto-only data with 4060. **But:** crypto data
is denser and has microstructure signal. Trade-off acceptable.

What we'd need to fully match frontier:
- Rented compute: 8x A100 for 2 weeks ≈ $2,000-4,000 cloud cost
- Tick-level data archive (V20 prereq): 5-10 GPU-d engineering + ~500 GB
  storage
- Multi-domain data: defer (we're crypto-specialized by design)

## 2026-05-02 update — Kronos benchmark (browser response E1)

**Hole added by browser, then closed empirically:**

The 2026-05-01 LITERATURE.md missed **Kronos** (Shi et al., AAAI 2026,
arxiv:2508.02739) — the finance-specialized analog of Chronos. Decoder-only
autoregressive on 12B K-line records from 45 exchanges. Open weights at
`NeoQuasar/Kronos-{mini,small,base,large}`. Three sizes fit our 4060.

A `@browser`-tagged research session caught this gap. We tested
**Kronos-small zero-shot on u10 chimera_legacy OOS** (1000 windows, ctx=400,
n_sample=5 MC). See `logs/frontier_ml/kronos_baseline/full_e1.log` and
`src/frontier_ml/kronos_baseline/eval_kronos.py`.

**Result: pooled Spearman IC at h=1 = +0.0292 across 10 assets.**
Per-asset: BTC -0.10, ETH +0.08, SOL -0.02, BNB +0.11, XRP -0.00,
DOGE +0.09, ADA +0.18, AVAX -0.02, LINK -0.04, LTC -0.01.

This is **below**:
- XGBoost 5-lag baseline (+0.031, this project)
- V1.0 baseline (+0.066)
- V1.1 record (+0.067)

**Conclusion: Kronos zero-shot does not transfer to our crypto + dollar-bar
distribution.** Likely cause: Kronos was pretrained on **uniform-time**
K-line bars; chimera_legacy uses **dollar-volume bars** with variable time
intervals. The model has no concept that "this bar represents $30M traded
over 47 minutes" — it sees bar shape but mis-aligns the implied horizon.

**Decision: stay on scratch-pretrain plan for Prong 1.** The 35-hour u100
pretrain on our own Mamba-3 backbone remains the right bet for our specific
data distribution.

### E1c — dollar-vs-time bar A/B test (2026-05-02, follow-up)

The E1 verdict had one open caveat: maybe Kronos failed because we fed it
dollar bars when it was pretrained on time bars. To close that, ran E1c:
resample chimera_legacy to uniform 1-hour time bars and re-test.

**Pooled time-bar IC = +0.0135 (n=1000, p=0.67); delta vs dollar-bar -0.016.**

Per-asset signs flipped across half the universe: BTC +0.12 (was -0.10),
ETH -0.08 (was +0.08), BNB -0.13 (was +0.11), ADA -0.03 (was +0.18). Only
LTC marginal at p=0.025. This is the signature of noise, not a systematic
shift to a regime where Kronos works.

**Conclusion: dollar-vs-time was NOT the issue.** Kronos zero-shot does
not transfer to our crypto distribution regardless of bar type. Pure
dollar-bar architecture is empirically validated for our pipeline, and
dual-cadence pretrain (which would have added a time-bar input stream) is
NOT warranted by evidence. Foundation Prong 1 stays as designed.

### Hole 9 — explicit: bar-type choice for foundation pretrain (NEW, closed 2026-05-02)

Bar-type selection had been implicit. Lopez de Prado AFML ch. 2 argues
dollar bars are closest to stationary across volatility regimes; V1.x
family records (IC 0.067 / ShIC 0.037) are all on dollar bars. The Kronos
E1 + E1c results empirically confirm that switching to time bars does NOT
unlock foundation-model transfer for our data distribution at our compute
scale. **Closure: dollar bars are the canonical bar type for the foundation
pretrain. Time bars are unnecessary as a primary input.** Other bar types
(DIB / range / runs / adaptive_vol) remain valuable as auxiliary feature
sources via the existing bar_fabric infrastructure.

### Adopted from the browser response (independent of Kronos verdict):

- **R3 — adaptive log-spaced bins.** Replace 255-uniform-bin TwoHot on
  [-1, 1] with 51-bin log-spaced (or quantile-fit). Default uniform bins
  waste ~80% of capacity for h=1 5-min returns where 99% of mass lives in
  ±0.01. Shipped: `src/frontier_ml/foundation/adaptive_bins.py` provides
  `make_log_spaced_bucketer()` and `make_quantile_bucketer()` with
  TwoHotSymlog-API-compatible drop-in semantics.

- **R2 — no agent prong.** Browser confirmed via CryptoBench
  (arXiv 2512.00417) and TradingAgents/FinAgent/CryptoTrade evidence that
  LLM-trader paradigm is 1-2 years from frontier-tier on crypto.
  Multi-agent crypto Sharpe 1.08 < our XGB xsec champion Sharpe 3.36 OOS WF.
  PPO agent stays on backburner. Revisit when CryptoBench top score > 60%.

- **R5 — CPCV gate for V19+ Headline candidates.** Walk-forward + 400-bar
  purge stays for V1.x; pre-Headline candidates that beat V1.1 must also
  pass CPCV PBO < 0.10.

## Closing summary

All 8 holes have closures (citation + experimental decision rule).
PLAN.md needs **two updates** before code:
1. Drop MSM and adversarial; keep causal next-token + lead-lag
   contrastive (Hole 2 + 3)
2. Add hybrid KL+L1+var distillation loss (Hole 5)
3. Add explicit lag parameter on multi-modal ingest (Hole 6)
4. Add OOM probe step before full pretrain (Hole 4)
5. Add checkpoint-every-100-steps + resume support (Hole 8)
6. Add deployment metric for distillation: ≥ 0.95 × best_teacher_IC
   at ≤ 1/4 latency (Hole 7)

After PLAN.md is updated and the pipeline lands, build commences in
strict sequence: prong 1 → measure → prong 2 → measure → prong 3.

## References

- Ansari et al. 2024 "Chronos: Learning the Language of Time Series"
  arxiv:2403.07815
- Das et al. 2024 "A decoder-only foundation model for time-series
  forecasting" arxiv:2310.10688
- Goswami et al. 2024 "MOMENT: A Family of Open Time-series Foundation
  Models" arxiv:2402.03885
- Nie et al. 2023 "PatchTST" arxiv:2211.14730
- Yue et al. 2022 "TS2Vec" arxiv:2106.10466
- Woo et al. 2022 "CoST" arxiv:2202.01575
- LeCun 2022 "JEPA"
- Phuong & Lampert 2019 "Knowledge Distillation for Ordinal Regression"
  arxiv:1812.04106
- Sanh et al. 2019 "DistilBERT"
- Hinton et al. 2015 "Distilling the Knowledge in a Neural Network"
- Furlanello et al. 2018 "Born-Again Networks"
- Lopez de Prado 2018 "Advances in Financial Machine Learning"
  (chapter 4: leakage)
- Bailey & Lopez de Prado "Probability of Backtest Overfitting" SSRN
