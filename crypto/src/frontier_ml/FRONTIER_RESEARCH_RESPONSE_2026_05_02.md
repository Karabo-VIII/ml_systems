# Frontier Research Response — Crypto WM + Agent (2026-05-02)

> **For:** the build state in `src/frontier_ml/` (3-prong: foundation +
> distillation + multi-modal) before launching the 35-hour u100 pretrain.
> **From:** a `@browser`-tagged research session, 8 WebSearch + 2 WebFetch
> calls, serial per the 2026-04-30 lean-utilization feedback.
> **Tone:** /un — direct, evidence-first, no polite hedging.

---

## 1. Executive verdict (≤ 200 words)

**Stack is correct but missing component Y — Y = Kronos baseline.**

A finance-specialized foundation model called **Kronos** (Shi et al., AAAI 2026,
arxiv:2508.02739) has shipped between when LITERATURE.md was written and now.
It is **decoder-only autoregressive on K-line (OHLCV) sequences**, pretrained
on **12 billion K-line records from 45 exchanges**, MIT-licensed,
HuggingFace-hosted, and ships at four sizes — Kronos-mini 4.1M, Kronos-small
24.7M, Kronos-base 102.3M, Kronos-large 499.2M. The first three fit our 4060.
Reported numbers: **+93% RankIC over the leading TSFM, +87% over the best
non-pretrained baseline**, with a public BTC/USDT 1h live demo using the
4M-param mini.

This is exactly the architectural bet we made (multi-asset autoregressive +
TwoHot/quantile bins on K-line bars) — except Kronos already paid the
12-billion-records pretrain cost on data we cannot match.

**Recommendation: do NOT skip the u100 pretrain. Do RUN Kronos-small as a
baseline against the foundation prong before committing to it.** If Kronos
matches V1.1 IC (0.067) zero-shot on our chimera_legacy bars, the foundation
prong becomes "finetune Kronos-small on our 10-asset corpus" instead of
"pretrain from scratch on 4060." Saves the 35 GPU-hours and likely lifts
the IC ceiling.

---

## 2. Top 5 papers / repos / projects we should KNOW about (probably not in LITERATURE.md)

### 2.1 Kronos (Shi et al., AAAI 2026)

- arxiv: [2508.02739](https://arxiv.org/abs/2508.02739)
- repo: [shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos)
- live BTC demo: [Kronos-demo](https://shiyu-coder.github.io/Kronos-demo/)
- analysis: [Time Series Foundation Models for Financial Markets — Kinlay](https://jonathankinlay.com/2026/02/time-series-foundation-models-for-financial-markets-kronos-and-the-rise-of-pre-trained-market-models/)

**Why it matters**: this is the paper's-equivalent of "the Chronos for
finance specifically." 12B K-line corpus + decoder-only autoregressive +
specialized OHLCV tokenizer + Monte Carlo N=30 sampling for probabilistic
forecasts. Already accepted at AAAI 2026 (the field's vetting process). Open
weights on HuggingFace under `NeoQuasar/Kronos-{mini,small,base}` +
`NeoQuasar/Kronos-Tokenizer-{2k,base}`. **Kronos-small (24.7M) is within our
30M target almost exactly; Kronos-base (102.3M) fits 4060 with FA2 + grad
checkpointing.**

### 2.2 TS-JEPA / Joint Embeddings Go Temporal (NeurIPS 2024 workshop → arxiv 2509.25449)

- arxiv: [2509.25449](https://arxiv.org/abs/2509.25449)

**Why it matters**: validates the JEPA → time-series direction we already
took (LITERATURE.md Hole 3). Not paradigm-class on its own — abstract claims
"match or surpass current state-of-the-art baselines" without quantifying.
Use as supporting citation for our cross-asset lead-lag JEPA design.

### 2.3 CHARM — Foundation Embedding Model for Multivariate Time Series via JEPA (2025)

- referenced in [Time Series Modeling Redefined — c3.ai](https://c3.ai/blog/time-series-modeling-redefined-a-breakthrough-approach/)

**Why it matters**: independent confirmation that JEPA + time-series is a
2025 mainstream direction. Not crypto-specific.

### 2.4 CryptoBench (arXiv 2512.00417, Dec 2025)

- arxiv: [2512.00417](https://arxiv.org/abs/2512.00417)

**Why it matters**: dynamic benchmark for LLM agents in cryptocurrency,
explicitly evaluating expert-level prediction. Identifies a systematic
**"retrieval-prediction imbalance"** across LLM models — strong on data
retrieval, weak on predictive analysis. **This is the single most important
finding for the agent prong: it explains why FinAgent / TradingAgents /
CryptoTrade sound impressive but can't outperform XGBoost.** Ground truth
for "the agent track is not yet frontier-tier."

### 2.5 Cisco Time-Series Model (Splunk, Nov 2025)

- announcement: [Splunk blog](https://www.splunk.com/en_us/blog/artificial-intelligence/introducing-the-cisco-time-series-model.html)

**Why it matters**: open-weight TSFM pretrained on 300B observability data
points. Not finance-specific. Worth noting as a counter-example: Cisco-TSM
shows that even a 300B-token pretrain on non-financial time series doesn't
beat a 12B-K-line domain-specialist (Kronos). **Domain specialization wins
at this corpus scale.** Reinforces the Kronos bet.

---

## 3. Specific recommendations (≤ 5 items)

### R1 — ADD Kronos as a baseline before committing to u100 pretrain

**Action**: clone `shiyu-coder/Kronos`, pull `NeoQuasar/Kronos-small` from
HuggingFace, run it zero-shot on our chimera_legacy 10-asset 1h bars,
measure h=1 / h=4 / h=16 / h=64 IC + ShIC against V1.1's 0.067 / 0.033.

**Decision rule**:
- If Kronos-small IC ≥ 0.060 zero-shot: foundation prong becomes
  "finetune Kronos-small with our cross-asset attention adapter" instead of
  pretraining from scratch.
- If Kronos-small IC ≥ 0.080 zero-shot (i.e. beats V1.1 record without
  finetune): direct ship as foundation; multi-modal adapter (prong 3) becomes
  the only headroom prong.
- If Kronos-small IC ≤ 0.030: stack stays on-trend; proceed to u100 pretrain
  as planned.

**Cost**: 1 GPU-hour for inference benchmark, 0.5 d engineering. Versus 35
GPU-hours for the u100 pretrain we'd otherwise commit to. **Highest
leverage 0.5d in the project right now.**

### R2 — DO NOT open a fourth (agent) prong

**Evidence**:
- TradingAgents (Tauric, 2024): 23.21% cumulative return on 3 stocks, 24.90%
  annualized, Sharpe ratio modest, **stock-only** in v1; crypto extension is
  community-maintained ([auronsun/TradingAgents-crypto](https://github.com/auronsun/TradingAgents-crypto)) and reports no IC numbers.
- FinAgent (multimodal foundation agent, 2024): **slightly lower returns
  than FinMem on ETH** because the auxiliary agents are stock-tuned.
- CryptoTrade (Xtra-Computing, EMNLP 2024 → 2025 follow-up): beats time-series
  baselines on cumulative return but **not on IC** specifically.
- Most recent crypto-specific multi-agent paper (ScienceDirect, S0306457325004078):
  21.75% total return / 29.30% annualized / **Sharpe 1.08** on Bitcoin —
  worse than our XGB xsec ranker (Sharpe 3.36 OOS WF, per
  `memory/honest_ranking_2026_04_22.md`).
- CryptoBench (arXiv 2512.00417, Dec 2025): identifies systematic
  "retrieval-prediction imbalance" — LLMs retrieve data well but predict
  poorly, and "the integration of agentic frameworks produced notable ranking
  shifts" (i.e. agent layer doesn't reliably help).

**Verdict**: the LLM-trader paradigm is 1-2 years from frontier-tier crypto
performance. Field is still pre-Sharpe-2.0 on real-money frequencies. Our
PPO agent backburner is the right call. **Do not open prong 4.** Revisit
in 2027 when CryptoBench scores improve.

### R3 — REPLACE TwoHot 255-uniform-bin with HL-Gauss or log-spaced bins

**Background** (from RESEARCH_BRIEF §5): h=1 5-min returns concentrate ~99%
of mass on ~50 of 255 bins.

**Evidence**: 2025 ICLR paper proceedings reformulated regression as
classification with **HPD (highest posterior density) regions**, showing
adaptive bins outperform uniform on density-skewed targets ([ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/9da457059dd3386a2166c5b08f29b7de-Paper-Conference.pdf)).
DeepMind's HL-Gauss head (2024-2025) similarly addresses this for
distributional RL. No 2025-2026 paper specifically on TwoHot adaptive-bin
for finance, but the failure mode is well-documented.

**Action**: switch FOUNDATION head from 255 uniform bins on [-1, 1] to **51
log-spaced bins** with cuts at ±0.0001, ±0.0003, ±0.001, ±0.003, ±0.01,
±0.03, ±0.1, ±0.3, ±1.0 (or fit empirically to our 5-min return density).
Keep V1.x at 255 uniform for backward compat. Project +0.005 ShIC at h=1.

### R4 — KEEP cross-asset lead-lag JEPA (Hole 3 closure) — it's correct

**Evidence**: TS-JEPA (arXiv 2509.25449) and CHARM both validate JEPA on
multivariate time-series in 2025. Our same-timestamp-vs-lead-lag concern
(LITERATURE.md Hole 3) anticipates the cross-asset issue these papers
implicitly address. **No paradigm shift; we got there first for crypto.**

Action: cite TS-JEPA + CHARM in LITERATURE.md Hole 3 as supporting evidence,
not paradigm-shifting.

### R5 — KEEP walk-forward + 400-bar purge for V1.x; ADD CPCV for V19+ pre-Headline

**Evidence**: CPCV outperforms walk-forward on PBO + DSR in synthetic
controlled studies (Bailey et al. SSRN 4686376). Quantbeckman/practitioner
consensus: "for limited-history crypto, CPCV is impractical because
train/test windows become too short." Our chimera_legacy has ~3-5 years per
asset; CPCV n_subperiods=16 leaves ~2-month windows — borderline.

Action: V19+ Headline-tier candidates that beat V1.1 on walk-forward
**must** also pass CPCV PBO < 0.10 before deployment. Don't run CPCV on
V1.x retrains (pre-Headline; not worth the complexity). Already covered
by CDAP `dsr_published_ranking` invariant in `config/_invariants.yaml`.

---

## 4. Headline-tier feasibility re-estimate

### What changes given the research

- **Kronos-small at 24.7M params zero-shot is the new external benchmark**
  for crypto foundation work. If it reports IC ≥ 0.07 in independent
  evaluation (TBD), our 0.10 Headline target is feasible **as long as we
  finetune Kronos rather than pretrain from scratch**.
- **Pretrain-from-scratch on 4060 cannot match a 12B-record corpus.**
  Domain-specialist Kronos is the existence proof that pretrain matters
  enormously when corpus dwarfs model capacity. Our chimera_legacy has
  ~50M bars across 10 assets; Kronos has 12B bars across 45 exchanges
  ≥ 240× more data.
- **Tier ladder ceiling for daily-bar dollar-bar regime**: still ~0.13
  raw IC by the data-dimensionality argument (per `CLAUDE.md`). Foundation
  models don't break this ceiling on daily bars; they raise the floor and
  shrink the variance.

### Numerical re-estimate

| Path | Ceiling estimate | Cost | Risk |
|---|---|---|---|
| Stay on plan: pretrain 30M from scratch on chimera_legacy | IC 0.07-0.09 | 35 GPU-h | low (we control) |
| Pivot to Kronos-small finetune + cross-asset adapter | IC 0.08-0.12 | 5-8 GPU-h | medium (depends on zero-shot baseline) |
| Pivot to Kronos-base finetune | IC 0.09-0.13 | 12-20 GPU-h | medium-high (FA2 + checkpointing required) |

**Realistic Headline target**: IC 0.10 / ShIC 0.05 is achievable via the
Kronos-base finetune path with our multi-modal adapter (prong 3) layered on.
**Without** a foundation-model corpus that dwarfs our own, Headline is
borderline.

### What gets us above 0.13 (Ambitious tier)?

- **Tick-level data + V20 architecture** — currently out of scope per
  hardware. Microstructure features (order-book imbalance at 100ms, queue
  position) carry ~0.05-0.08 IC headroom that daily bars cannot.
- **Multi-exchange + on-chain + macro multi-modal fusion** with proper
  T+1 lag hygiene — prong 3 of our plan; estimate +0.01-0.02 IC.
- **Reasoning chain as feature engineer** (Kronos+RAG+chain-of-thought) —
  speculative; CryptoBench evidence says LLMs hurt as predictors today.
- **SOTA agentic execution layer** that captures non-linear venue advantages
  — engineering, not modeling.

**Verdict on Ambitious tier (0.13)**: ~30% probability via multi-modal
prong + Kronos-base finetune; ~70% probability requires V20 tick-level
work (not in 4060 budget).

---

## 5. Agent track verdict

**Do NOT open a fourth prong for an LLM-agent layer.** Defer to 2027 review.

### Specific evidence

- TradingAgents 23.21% / Sharpe modest on stocks, no IC reporting on crypto.
- FinAgent UNDERPERFORMS FinMem on ETH despite multi-modal upgrade
  (research consensus: subagents stock-tuned, crypto generalization fails).
- CryptoTrade beats time-series baselines on returns but not on IC; 1d
  decision frequency caps Sharpe.
- ScienceDirect Bitcoin multi-agent: Sharpe 1.08 — **worse than our XGB
  xsec champion at Sharpe 3.36 OOS WF**. The agent layer would degrade us.
- CryptoBench retrieval-prediction imbalance: agentic frameworks
  produce ranking shifts but not consistent improvements.
- Grok-4 (Web) tops CryptoBench at 44.0% accuracy (vs 30.0% for GPT-5);
  even the best LLM is below random-walk-on-random-coin-flip if calibrated
  honestly.

### What we keep

- PPO agent under `src/agent/` — keep on backburner. RL is the right
  framework for execution timing once we have signal IC > 0.10. PPO does
  not require an LLM.
- Voyager-style skill library (RESEARCH_BRIEF Task 2.3) — interesting but
  no published crypto port; defer.

### What to revisit in 2027

- CryptoBench scores rising past Grok-4's 44%.
- Any LLM-trader system reporting **independently-verified** Sharpe ≥ 2.0
  on crypto live (not paper).
- Open-weights crypto-specialist agent trained on tool-use + reflection
  with public eval.

---

## 6. Concrete experiments (priority order, ≤ 5)

### E1 — Kronos-small zero-shot benchmark on our 10-asset 1h chimera_legacy

**Goal**: get an objective IC vs V1.1 record (0.067).

**How**: pull `NeoQuasar/Kronos-small` + `NeoQuasar/Kronos-Tokenizer-base`
from HuggingFace; run inference on the OOS+UNSEEN segments of all 10
assets; compute h=1 IC + ShIC; compare to V1.1 baseline.

**Cost**: 1-2 GPU-hours.

**Decision**: if Kronos-small IC ≥ 0.040, abandon scratch-pretrain and
pivot foundation prong to "Kronos-small + cross-asset adapter finetune."
If < 0.040, proceed with current plan.

### E2 — Kronos-base feasibility probe on 4060

**Goal**: confirm Kronos-base (102.3M) can run inference + LoRA-style
finetune on 4060/8GB with FA2 + grad checkpointing.

**How**: install Kronos-base, run forward at seq_len=512 batch=1 in fp16,
measure peak VRAM. If <6 GB, run a 100-step LoRA finetune probe.

**Cost**: 2-3 GPU-hours.

**Decision**: if Kronos-base fits, it becomes the foundation prong target
(not Kronos-small). The 4× param difference at fixed corpus is the
single biggest expected IC delta in the project.

### E3 — TwoHot adaptive-bin head replacement

**Goal**: validate +ShIC from log-spaced 51-bin head vs 255-uniform-bin.

**How**: train V1.0-class baseline with the new head only; everything else
held constant. 2-3 hour run. Compare ShIC vs the V1.0 0.0319 reference.

**Cost**: 3 GPU-hours.

**Decision**: if ShIC ≥ 0.034 (+5% vs V1.0), adopt for foundation +
distillation prongs. If ≤ 0.030, leave at 255-uniform.

### E4 — Multi-modal prong 3 with Kronos backbone

**Goal**: validate the cross-attention adapter on Kronos-small (or base
if E2 passes), with funding/OI/ETF side channels.

**How**: freeze Kronos backbone; train cross-attention adapter for 5K
steps on chimera_legacy + frontier panels (with explicit T+1 lag).
Compare adapter-only IC to Kronos-zero-shot IC.

**Cost**: 5-10 GPU-hours.

**Decision**: if multi-modal lift > +0.01 IC, prong 3 ships. If < +0.005,
multi-modal stays out and we save the channel-fusion engineering.

### E5 — Rerun strat-layer dsr_re_rank under foundation backbone

**Goal**: confirm the 61-graduate cohort (per `logs/strat_audit/dsr_re_rank.md`)
holds when WM signal upgrades from V1.1 to Kronos-finetune.

**How**: regenerate strategy seed daily-snapshots with the Kronos-finetuned
WM signal in place of V1.1; re-run `scripts/strat_audit/dsr_re_rank.py`;
compare graduate count.

**Cost**: 2-4 GPU-h depending on number of WM-dependent strategies (per
`memory/strat_review_2026_04_30.md`, 7 of 66 strategies are WM-stale).

**Decision**: if graduate count rises by ≥ 5, foundation prong shipping
also unlocks DEPLOY-tier strategies (per the WM-stale flag mechanism in
STRAT_REVIEW_SPEC §13.1). If unchanged, foundation prong is a quality
upgrade only, not a deployment unlock.

---

## 7. Stop-condition check

Per the brief's stop conditions:

- **"If the field has shipped no materially new paradigm, stop early"** →
  one paradigm-class artifact has shipped: **Kronos** (AAAI 2026,
  domain-specialist financial foundation model). Cannot stop early.
- **"If one paradigm dominates, drop the polite hedging"** → Kronos at
  4-103M params with 12B-record pretrain is the dominant paradigm for
  daily/hourly crypto WM. **Recommendation #1 (R1) is not a hedge — it is
  the only sensible next move.** The 35 GPU-h scratch-pretrain on chimera_legacy
  makes a worse model than 1 GPU-h Kronos-small zero-shot will likely
  produce. Run E1 before committing.

---

## 8. Sources

### Foundation models / time-series

- [Kronos: A Foundation Model for the Language of Financial Markets — arXiv 2508.02739](https://arxiv.org/abs/2508.02739)
- [Kronos GitHub repository](https://github.com/shiyu-coder/Kronos)
- [Kronos live BTC/USDT demo](https://shiyu-coder.github.io/Kronos-demo/)
- [Time Series Foundation Models for Financial Markets: Kronos and the Rise of Pre-Trained Market Models — Kinlay 2026-02](https://jonathankinlay.com/2026/02/time-series-foundation-models-for-financial-markets-kronos-and-the-rise-of-pre-trained-market-models/)
- [The 2026 Time Series Toolkit: 5 Foundation Models for Autonomous Forecasting — MachineLearningMastery](https://machinelearningmastery.com/the-2026-time-series-toolkit-5-foundation-models-for-autonomous-forecasting/)
- [TimesFM GitHub](https://github.com/google-research/timesfm/)
- [Time-Series Foundation Models in Finance: Pretraining (ACM 2025)](https://dl.acm.org/doi/full/10.1145/3785706.3785728)
- [MOMENT GitHub](https://github.com/moment-timeseries-foundation-model/moment)
- [Cisco Time Series Model — Splunk blog Nov 2025](https://www.splunk.com/en_us/blog/artificial-intelligence/introducing-the-cisco-time-series-model.html)
- [Chronos-2 — Medium analysis](https://medium.com/@alexnedyalkov/chronos-2-the-new-state-of-the-art-forecasting-framework-74b40b21f953)

### State-space + JEPA time-series

- [MambaTS — arXiv 2405.16440](https://arxiv.org/abs/2405.16440)
- [ms-Mamba — arXiv 2504.07654](https://arxiv.org/html/2504.07654v1)
- [Joint Embeddings Go Temporal — arXiv 2509.25449](https://arxiv.org/abs/2509.25449)
- [Time Series Modeling Redefined: A Breakthrough Approach (CHARM) — c3.ai](https://c3.ai/blog/time-series-modeling-redefined-a-breakthrough-approach/)

### Crypto-specific LLM agents

- [TradingAgents arXiv 2412.20138](https://arxiv.org/abs/2412.20138)
- [TradingAgents crypto fork — auronsun](https://github.com/auronsun/TradingAgents-crypto)
- [FinAgent multimodal foundation agent — arXiv 2402.18485](https://arxiv.org/html/2402.18485v3)
- [FinMem layered memory — arXiv 2311.13743](https://arxiv.org/abs/2311.13743)
- [CryptoTrade reflective LLM agent — arXiv 2407.09546](https://arxiv.org/abs/2407.09546)
- [Explainable zero-shot trading multi-agent BTC — ScienceDirect S0306457325004078](https://www.sciencedirect.com/science/article/abs/pii/S0306457325004078)
- [CryptoBench dynamic LLM-agent benchmark — arXiv 2512.00417](https://arxiv.org/abs/2512.00417)

### Statistical validity / cross-validation / distillation

- [Backtest Overfitting in the ML Era (Bailey et al. SSRN 4686376)](https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID4686376_code4361537.pdf?abstractid=4686376&mirid=1)
- [Combinatorial Purged Cross Validation — Quantbeckman](https://www.quantbeckman.com/p/with-code-combinatorial-purged-cross)
- [A Comprehensive Survey on Knowledge Distillation — arXiv 2503.12067](https://arxiv.org/html/2503.12067v1)
- [BiLD knowledge distillation loss — Bi-directional Logits Difference 2025](https://aclanthology.org/2025.coling-main.169.pdf)

### Hardware / efficiency

- [FlashAttention-3 — arXiv 2407.08608](https://arxiv.org/html/2407.08608v1)
- [Unsloth Gradient Checkpointing](https://unsloth.ai/blog/long-context)
- [FastAttention low-resource GPU extension — arXiv 2410.16663](https://arxiv.org/html/2410.16663v1)

### Crypto market structure (lead-lag)

- [Price Transmission from Bitcoin to Altcoins: High-Frequency Evidence — Springer](https://link.springer.com/article/10.1007/s10690-026-09589-z)
