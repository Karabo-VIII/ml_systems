# Frontier-tier WM Plan — 4060/8GB constraint version

> **Status (2026-05-02):** BUILD PHASE. Holes 1-8 closed via LITERATURE.md.
> 6 literature-driven updates applied below. Pipeline gate green; Prong 1
> implementation underway.

## Literature-driven updates (post-LITERATURE.md, 2026-05-02)

These supersede earlier sections where they conflict:

1. **Objective set simplified to 2 terms** (Hole 2): drop masked-sequence-
   modeling and time-shuffle adversarial. Keep causal multi-horizon next-
   token (TwoHot 255 bins, h ∈ {1, 4, 16, 64}) as primary + lead-lag
   cross-asset contrastive as auxiliary. Per Chronos / TimesFM / MOMENT
   evidence: causal next-token wins for forecasting.
2. **Cross-asset contrastive uses lead-lag positives** (Hole 3): positive
   pair = (A_t, B_{t+δ}) with δ sampled from {0, 1, 3, 12} bars; negative
   = (A_t, B_{t+T}) for large T. Handles BTC → ETH → alt cascade.
3. **Distillation loss is hybrid** (Hole 5): α·KL + β·L1(expected return)
   + γ·L2(variance), starting α=0.5 β=0.4 γ=0.1.
4. **Multi-modal channels carry explicit lag parameter** (Hole 6):
   default 1 bar; walk-forward purge ≥ longest lookback.
5. **OOM probe BEFORE full pretrain** (Hole 4): 200 steps real-data at
   batch=8 seq=512 on actual model; commit epoch budget only after pass.
6. **Distillation deployment metric** (Hole 7): student must hit
   IC ≥ 0.95 × best_teacher_IC AND latency ≤ 1/4 ensemble; else deploy
   ensemble.

Backbone components are REUSED from `src/wm/v4/v4_training/components.py`
(`Mamba3SSM`, `Mamba3Block`, `TwoHotSymlog`, `RMSNorm`) — V4 already
validated the architecture; we scale it (6 layers, d_model=256,
d_state=16) rather than reimplementing.

## TL;DR

Three-pronged bet, sized for one 4060:

1. **Foundation pretraining** at 30M params on u100 + 1y aggTrades with
   masked + cross-asset contrastive + multi-horizon objectives.
   Then per-asset linear-probe + light fine-tune.
2. **Distillation** from the ensemble of V1/V3/V4/V6/V11-V14 already
   trained, into a single 5-10M deployable model.
3. **Multi-modal alignment**: macro + on-chain + funding side-channel
   added to the foundation backbone via cross-attention adapter.

NOT pursuing on this hardware:
- Tick-level Performer/Hyena (V20) — needs >> 8 GB unless heavily
  truncated; defer until we have rented compute
- 100M+ pretraining
- End-to-end RL with full latent imagination at scale

## Prong 1: Foundation pretraining (30M params)

### Architecture

A 30M-param **Mamba-3 backbone** + **shallow cross-asset attention head**:

- Backbone: 6-layer Mamba-3 (~24M params at d_model=256, d_state=16)
  - Why Mamba over Transformer: linear in sequence length → can do
    seq_len=512+ on 8 GB; transformer would cap at ~256
  - Why Mamba-3 specifically: complex-valued state via RoPE, SSD chunks,
    QK-Norm — the V4 line already validates the architecture works
- Cross-asset head: 2-layer attention over u10/u50 asset embeddings
  (~4M params at d=128, n_heads=4, n_assets=50)
- Total: ~28M params. fp16 = 56 MB weights. Adam states ~225 MB. Activations
  at batch=8, seq=512: ~700 MB. Total ~1 GB → fits 8 GB comfortably.

### Self-supervised objectives (multi-task)

All four jointly during pretrain:

1. **Masked sequence modeling (MSM)** — mask 15% of tokens (5-min OHLCV +
   flow_imbalance), predict back. Standard BERT-style.
2. **Multi-horizon next-window prediction** — predict h ∈ {1, 4, 16, 64}
   bar returns + variance. Same as V1.x's TwoHot prediction. Direct
   transfer to downstream task.
3. **Cross-asset contrastive (JEPA-style)** — for each (asset, time)
   anchor, positive = same time across other u10 assets, negative =
   different time same asset. Encourages cross-asset structure in
   embeddings.
4. **Time-shuffle adversarial** — V6's existing JEPA pattern. Auxiliary
   discriminator distinguishes "real sequential window" vs "time-shuffled
   window". Backbone is rewarded for fooling discriminator.

Joint loss: w₁·L_MSM + w₂·L_horizon + w₃·L_contrastive + w₄·L_adv with
λ-tuned weights. Start from {0.3, 0.5, 0.1, 0.1} based on V6 + V1
experience.

### Data + corpus

- u100 (87 active assets after the universe cleanup)
- 5-min OHLCV from raw aggTrades (already in chimera_legacy at 41 cols)
- ~5 years × 87 assets × 288 bars/day = ~46M (asset, bar) pairs
- At seq_len=512 stride=64 → ~700K training windows
- After train/val/oos/unseen 50/20/20/10 split: 350K train windows
- At batch=8 → 44K steps/epoch
- Compute: ~12 hr/epoch on 4060 (Mamba is fast at d_state=16)
- Target: 5-10 epochs = 60-120 hr = 2.5-5 days wall-clock for full pretrain

### Downstream fine-tune

After pretraining, freeze backbone + attach small head per task:
- Linear probe: 1-layer head, ~100K params, 1 hr/asset
- Light fine-tune: 2-layer head + last 2 backbone layers unfrozen,
  ~3M effective params, 2 hr/asset
- Goal: linear-probe IC > current V1.0 IC (0.066) on the same OOS window

## Prong 2: Distillation from existing ensemble

### Source teachers

Already-trained checkpoints (post-2026-04-29 audit):
- V1.0 / V1.1 / V1.4 / V1.6 (transformer + RSSM, 2M each)
- V3 (WaveNet, 2M)
- V4 (Mamba-3 + RSSM, 3.5M)
- V6 (JEPA, 1M)
- V11 (combo, 3M)
- V12 (cross-asset attention, 1M)

Teacher logits (TwoHot return distribution, 255 bins, h={1,4,16,64})
across all checkpoints → averaged or learned-gated ensemble probability.

### Student

5-10M-param distilled model. Either:
- Reuse V1.4 architecture (FeatureAttention block) as student
- OR a fresh 5M Mamba

Loss: KL divergence between student logits and ensemble teacher logits +
direct return Huber on raw targets. λ-tuned.

### Compute

- Teacher inference: cached one-time (each teacher predicts on training
  windows once, results saved to disk)
- Student training: 1 epoch = ~3 hr on 4060
- Total: ~30-50 hr for proper distillation

### Why this is high-leverage on our hardware

Distillation transfers architectural diversity into a single small model.
We can't run an ensemble in production (deployment latency). Distilled
student gets diversity benefit at single-inference cost.

## Prong 3: Multi-modal alignment

### Side channels

Currently we use price-only inputs. Multi-modal additions:
1. **Funding rate** (already in chimera as norm_funding)
2. **Open interest delta** (already in chimera as norm_oi_change)
3. **Macro**: DXY, S&P, BTC dominance — daily fetch from Yahoo or
   Coinbase Pro pre-market
4. **On-chain**: large-tx counts, exchange netflow — DefiLlama or
   CryptoQuant has free tier
5. **News embeddings**: already-computed daily sentiment (e.g.,
   Tradingview News API or Coinbase events) → embed via a frozen
   small text model (sentence-BERT, runs on CPU)

### Architecture

Cross-attention adapter on top of foundation backbone:
- Backbone produces sequence embeddings for each asset
- Side channels produce per-day vectors aligned to chimera dates
- Cross-attention: backbone embeddings query into side-channel KV
- ~2M extra params; backbone stays frozen during multi-modal fine-tune

### Why on this hardware

Side-channels are tiny (10-50 features each). Cross-attention adapter is
small (~2M params). Total system stays under 35M. Still fits 8 GB.

## Compute budget summary

| Phase | Time | Status |
|---|---|---|
| Foundation pretrain (Prong 1) | 60-120 hr | NOT YET |
| Per-asset linear probe (10 u10) | 10 hr | NOT YET |
| Per-asset light fine-tune (10 u10) | 20 hr | NOT YET |
| Distillation (Prong 2) | 30-50 hr | NOT YET |
| Multi-modal adapter (Prong 3) | 20-40 hr | NOT YET |
| **TOTAL** | **140-240 hr** | ~6-10 days wall-clock |

If we target 2 weeks for the full sprint, we have headroom for retries
and ablations.

## What we're explicitly NOT doing on 4060

- ❌ Tick-level pretraining (V20) — defer until rented compute
- ❌ 100M+ foundation backbone — won't fit
- ❌ Long-context (>1024) sequences — fits only with seq_len truncation
- ❌ Multi-task RL with full DreamerV3 imagination at scale — need 24+ GB
- ❌ Diffusion over full return distribution at 1000 steps — too slow
- ❌ Online learning with continuous gradient updates — not needed for
  daily-cadence trading

## Decision points before building

1. **Does linear-probe foundation IC > V1.0 IC after 2 epochs of
   pretrain?** If no after 2 epochs, the foundation approach isn't
   working at our scale; pivot to distillation-only.
2. **Does cross-asset contrastive add IC vs cross-asset-disabled
   ablation?** If no, drop it; saves 25% of pretrain compute.
3. **Does multi-modal adapter add ≥ +0.01 IC at p_value < 0.10 vs
   no-adapter ablation?** Any LESS than that is noise on our sample.
4. **Does distilled student match or beat the ensemble's average IC?**
   If no, ensemble is the deployable; distillation isn't worth it.

Each decision point ends with a measurable outcome, not just "looks
good in TensorBoard."

## Holes (poked deliberately — see LITERATURE.md for resolution)

> ⚠️ **Holes** are unresolved questions whose answers determine whether
> a load-bearing claim above stands. Each hole MUST be closed (with
> citation or experiment) before building.

1. **Is 30M parameters enough to "emerge" anything on time-series?**
   LLM emergence happens at 6B-100B+. Sub-billion-parameter models in
   text are non-emergent. Time-series is denser (more bits per token)
   but no one has demonstrated emergent capabilities at <100M for
   crypto. Resolution: pre-train, measure, decide. Risk-accepted: this
   is exploratory.
2. **Is masked sequence modeling actually the right objective for
   numerical time-series?** BERT-MSM is text-tested, not time-series-
   tested. Chronos / TimesFM use causal next-token, not MSM. Need to
   compare both during pretrain.
3. **Cross-asset contrastive: is positive-pairing at "same timestamp"
   correct?** Crypto markets are correlated but with delays
   (BTC → ETH leader-follower). Same-timestamp positives may bias
   the model toward instantaneous correlation, missing lead-lag.
4. **Hardware-OOM risk during pretrain.** A 30M-param Mamba with
   seq_len=512 at batch=8 — measured peak memory? We've seen V4 (3.5M)
   use ~3 GB. 30M scales to ~6-7 GB at batch=8. Not far from the cap.
   Need empirical probe before committing to the full epoch budget.
5. **What does ensemble distillation look like on TwoHot logits?** KL
   divergence on 255-bin distributions has known issues (peaky teacher
   distributions don't transfer cleanly). Alternative: distill on the
   continuous expectation + variance, not the bin softmax.
6. **Multi-modal alignment via cross-attention: does it leak future
   info?** Macro / on-chain / news must be aligned to BAR-CLOSE time,
   not bar-open or any future timestamp. Subtle bug surface.
7. **Distillation diminishing returns**: with 8 teachers each at
   IC=0.06-0.07, average ensemble probably gets ~0.075. A distilled
   student would aim for similar. Whether it beats V1.1's IC=0.067
   record in walk-forward is unclear.
8. **Compute budget realism**: 6-10 days of GPU time on a single 4060
   is essentially 24/7 utilization. Power, thermals, and not being
   able to use the GPU for anything else for 1.5 weeks is a real cost.

See `LITERATURE.md` for citations against each hole.

## Provenance

Plan v0 — 2026-05-01 — drafted under user mandate to "build top-tier
WM for our problem within 4060/8GB hardware". Written before pipeline
fully complete; pipeline must land first per build-order.
