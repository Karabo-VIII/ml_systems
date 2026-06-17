# WM Headline-Tier Upgrade Plan — Per-Architecture (2026-04-30)

> **Mandate** (CLAUDE.md "Indisputable Operating Lens"): every active
> architecture in `src/wm/v*/` must have a documented upgrade plan
> targeting **IC > 0.10 / ShIC > 0.05** (the agent-teaching tier where
> the WM signal IS the alpha, not just a position-sizing input).
>
> **Companion**: [WM_HIGH_TIER_TARGETS_2026_04_30.md](WM_HIGH_TIER_TARGETS_2026_04_30.md)
> for the broader framework + 5 NEW V20-V24 proposals.
> **This doc**: per-existing-architecture concrete upgrades.
>
> **Stance**: pessimistic-conservative. Some upgrades will fail.
> The point is to BUILD THE LADDER toward Headline tier explicitly,
> not to assume any single upgrade will land us there.

---

## 0. Cross-cutting upgrades (apply to most architectures)

These knobs lift the signal-capacity ceiling regardless of architecture.
They're listed once here and referenced in per-version specs.

**STATUS UPDATE 2026-04-30 (post-evening commit)**: ALL 7 cross-cutting
upgrades are now SHIPPED as shared modules under
[src/wm/_shared/headline_components.py](../src/wm/_shared/headline_components.py).
Each has its own smoke test; all 6 module-level smokes PASS. The
remaining work is per-version trainer wiring + V20 tick-chimera build.

```
$ python src/wm/_shared/headline_components.py
[CC-H3 cross-asset]  smoke PASS  (262,912 params)
[CC-H1 multi-res]    smoke PASS  (78,098 params, causal verified)
[CC-H2 linear-attn]  smoke PASS  (65,920 params, causal verified)
[CC-H5 quantile]     smoke PASS  (55,132 params, pinball loss = 0.13)
[CC-H6 regime]       smoke PASS  (1,187,316 params; soft_blend math verified)
[CC-H7 dream]        smoke PASS  (auxiliary loss helper)
ALL CC-H1 / CC-H2 / CC-H3 / CC-H5 / CC-H6 / CC-H7 SMOKE PASS
```

### CC-H1 — Multi-resolution context (1-bar + 4-bar + 16-bar)

**Mechanism**: instead of a single 96-bar context at h=1 dollar bars,
stack three encoders at three resolutions. Concat the latents.

**Expected lift**: +0.005-0.010 IC, +0.002-0.005 ShIC across V1.x/V3/V4.
Captures features at multiple frequencies in one model.

**Cost**: +20-30% wall-clock per epoch; ×3 encoder params.

**Where it applies**: V1.0/V1.1/V1.4/V1.6 (Transformer encoders),
V3 (WaveNet), V4 (Mamba).

### CC-H2 — Sequence length 96 → 256 bars

**Mechanism**: 2-3 days clock at 4-bar/day → 6-9 days clock at 256
bars. Captures funding-rate regime, weekly mean-reversion.

**Expected lift**: +0.003-0.008 IC.

**Cost**: O(seq²) for V1.x Transformer = 7× attention cost. Free
for V4 Mamba (linear). Moderate for V3 WaveNet (depends on dilations).

**Where it applies**: V1.0/V1.1/V1.4/V1.6 with bottom-up swap to
linear-attention (Hyena/Performer); V3 with extended dilations to 32;
V4 directly (Mamba scales).

### CC-H3 — Cross-asset attention head

**Mechanism**: light cross-asset attention layer ABOVE the per-asset
encoder, before the prediction heads. Each asset's representation
sees the other 9 assets' representations at the same timestamp.

**Expected lift**: +0.005-0.012 IC. The cross-asset signal is real
(BTC leads alts; alt-season is a regime); single-asset V1.x can't
exploit it directly.

**Cost**: 1 attention layer × 10 assets = trivial; ~5% wall-clock.

**Where it applies**: ALL versions. This is the cheapest single upgrade
on the list.

### CC-H4 — Anti-mem stack: KL ↑, ATME ↑, XD ↑

**Mechanism**: tighten the anti-memorization knobs:
- `WM_FREE_NATS = 1.0 → 1.5-2.0` (KL latent forced to throw away more info)
- `TEMPORAL_CTX_DROP = 0.15 → 0.25` (ATME zeros more bars)
- `XD_DROPOUT_RATE = 0.7 → 0.85` (drop ~12 of 34 features per batch)

**Expected effect**: ShIC +0.005-0.012 at cost of IC -0.005-0.010.
The point is RATIO improvement (ShIC/IC from ~0.49 → 0.65+).

**Where it applies**: V1.x has ATME+KL+XD; V3/V6 have XD only.

### CC-H5 — Distributional output (q05/q50/q95) instead of point

**Mechanism**: replace TwoHot point estimate with quantile heads
(p10, p50, p90) trained with quantile loss. Conformal calibration on
top.

**Expected lift**: tradeable Sharpe via better sizing (use q90-q10 for
position-size scaling), not raw IC. But strategy-side meta-learner
needs to consume quantile vector (CC2 from STRAT_SOTA_REVIEW).

**Where it applies**: V14 already does (diffusion); V1.x can be
retrofitted via `quantile_heads.py` (already exists in src/strategy/).

### CC-H6 — Predicted-regime conditional training

**Mechanism**: train per-regime auxiliary heads (separate decoders for
bear/neutral/bull). Loss = base loss + λ × per-regime CE. At inference,
soft-blend across regimes via the regime gate.

**Expected lift**: +0.003-0.008 IC, **+0.05+ on Sharpe in regime shift
windows** (the headline benefit isn't IC, it's robustness).

**Where it applies**: ALL. Cheap.

### CC-H7 — Auxiliary loss on dream-rollout (V1.6's `dream_step`)

**Mechanism**: V1.6 has `dream_step` defined but it's not in the loss.
Add a 1-2 step rollout at training time + reconstruction + return
prediction. Forces the latent to be predictively useful, not just
encoding-useful.

**Expected lift**: +0.003-0.007 ShIC.

**Where it applies**: V1.6 first (already has the path), then port
to V1.0/V1.1/V1.4/V3/V4/V6.

---

## 1. V0 — Linear / non-linear baseline

**Status**: SHIP-AS-BENCHMARK; 65/100 in current scoresheet.

**Headline plan**: V0 by definition does NOT pursue Headline tier; it
**defines the floor** that every neural WM must beat by 3× IC. **No
upgrade plan needed.**

But: V0 needs explicit DSR + PBO computation per the CC4 finding (gap
in scoresheet is the missing rigor, not the missing IC). Action:
add DSR/PBO to `linear_baseline.py` output.

**Decision**: V0 stays at SHIP-AS-BENCHMARK; close DSR gap; no Headline
ambition.

---

## 2. V1.0 — Transformer + RSSM reference

**Current**: IC 0.067, ShIC 0.032 (Trader tier).

**Headline upgrades** (in priority order):

| # | Upgrade | Expected delta | Cost (GPU-d) | Risk |
|---|---|---|---|---|
| H1 | CC-H3 cross-asset head | IC +0.005-0.012 | 0.5 | low |
| H2 | CC-H1 multi-resolution stack | IC +0.005-0.010 | 1.0 | medium |
| H3 | RSSM 24×24 → 32×32 | IC +0.002-0.005 | 1.5 | low; previously tested marginal |
| H4 | CC-H6 regime-conditional heads | Sharpe +0.05 | 1.0 | low |
| H5 | Pattern P+Q at f29 (CC4 baseline) | IC +0.005, ShIC +0.005 | 0.5 | already coded |

**Bundled V1.0-Headline retrain**: H1+H2+H4+H5 = 3 GPU-d → projected
IC ≥ 0.085 / ShIC ≥ 0.042 (high end of band, gates Headline).
Acceptance: must beat `IC ≥ 0.080` AND `ShIC ≥ 0.040` AND ratio ≥ 0.50
on 3-window walk-forward.

**Failure path**: if V1.0-Headline lands at IC < 0.075, the
representation is the bottleneck → escalate to V21 (multi-asset MASTER)
which is the lowest-risk Headline-target arch.

---

## 3. V1.1 — XD anti-memorization (current record)

**Current**: IC 0.067, ShIC 0.033 (Trader tier; record).

**Headline upgrades**:

| # | Upgrade | Expected delta | Cost (GPU-d) | Risk |
|---|---|---|---|---|
| H1 | CC-H3 cross-asset head | IC +0.008-0.015 | 0.5 | low |
| H2 | CC-H4 anti-mem ↑ (XD 0.7→0.85, KL 1.0→1.5, ATME 0.15→0.25) | ShIC +0.008-0.015, IC -0.003-0.008 | 0.5 | medium |
| H3 | CC-H1 multi-resolution stack | IC +0.005-0.010 | 1.0 | medium |
| H4 | CC-H6 regime-conditional heads | Sharpe +0.05 | 1.0 | low |
| H5 | Pattern P+Q at f29 | IC +0.005, ShIC +0.005 | 0.5 | already coded |

**Bundled V1.1-Headline retrain**: H1+H2+H4+H5 = 2.5 GPU-d → projected
**IC ≥ 0.082, ShIC ≥ 0.045**. ShIC moves from 0.033 → 0.045 (+0.012)
puts ratio at 0.55 (was 0.49). **Highest-EV individual model upgrade
in this plan** because XD's anti-mem stack is already mature; CC-H4
unlocks the next ShIC tier directly.

**If V1.1-Headline ShIC ≥ 0.050**: V1.1 hits Headline tier proper.
This is the **most plausible single-architecture path to Headline**.

---

## 4. V1.4 — FeatureAttentionBlock (iTransformer)

**Current**: IC 0.068 (best raw), ShIC 0.031 (Trader tier).

**V1.4's distinctive lever**: cross-FEATURE attention. Extending it
to cross-FEATURE×cross-ASSET makes it the natural Headline candidate
within V1.x family.

**Headline upgrades**:

| # | Upgrade | Expected delta | Cost (GPU-d) | Risk |
|---|---|---|---|---|
| H1 | Extend FeatAttn → cross-feature × cross-asset (joint attention over the [bar, feature, asset] tensor) | IC +0.012-0.020 | 1.5 | medium-high |
| H2 | CC-H4 anti-mem ↑ | ShIC +0.005-0.010 | 0.5 | low |
| H3 | CC-H6 regime heads | Sharpe robust | 1.0 | low |
| H4 | Pattern P+Q at f29 | +0.005 each | 0.5 | already coded |

**V1.4-Headline cost**: 3.5 GPU-d. Projected **IC ≥ 0.085, ShIC ≥
0.040**.

**The V1.4 vs V1.1 race**: at Headline tier, the question is whether
cross-feature attention (V1.4) or anti-mem-stack (V1.1) is the
dominant lever. Per existing data V1.4 has higher raw IC, V1.1 has
higher ShIC. **Train both at Headline-spec, ensemble in V10. Expected
ensemble lift ≥ 0.005 ShIC if pairwise ρ < 0.85.**

---

## 5. V1.6 — All anti-memorization (KL+Gumbel+ATME+dream)

**Current**: IC 0.062, ShIC 0.033 (Trader tier; lowest raw IC in V1.x).

**V1.6 is the natural CC-H7 testbed**: dream_step exists in code but
isn't in the loss. Activating it is the V1.6-specific Headline play.

**Headline upgrades**:

| # | Upgrade | Expected delta | Cost (GPU-d) | Risk |
|---|---|---|---|---|
| H1 | CC-H7 dream-rollout in loss (V1.6's signature; auxiliary 1-2 step rollout) | IC +0.003, ShIC +0.005-0.012 | 1.0 | medium |
| H2 | CC-H3 cross-asset head | IC +0.005-0.010 | 0.5 | low |
| H3 | CC-H4 anti-mem ↑ (already aggressive; tune Gumbel temperature instead) | ShIC +0.003 | 0.5 | low |
| H4 | Mechanism ablation: train V1.6 with ONE of {KL, Gumbel, ATME, dream} disabled at a time, identify which is load-bearing | answer dependent | 4 GPU-d (4 ablation runs) | informational |

**V1.6-Headline cost**: 2 GPU-d for upgrade; 4 GPU-d ADDITIONAL for
ablation (deferred). Projected **IC ≥ 0.072, ShIC ≥ 0.045**.

**V1.6 is the dream-rollout R&D lab** — even if V1.6-Headline doesn't
hit the 0.10 IC bar, the dream-rollout finding ports to V1.0/V1.1/V1.4
adding ShIC across the cohort.

---

## 6. V3 — WaveNet-GRU + RSSM

**Current**: untrained at f29; --clean variant memorized.

**V3's distinctive lever**: dilated causal convolutions handle
multi-resolution natively. Push the dilations further.

**Headline upgrades**:

| # | Upgrade | Expected delta | Cost (GPU-d) | Risk |
|---|---|---|---|---|
| H1 | Pattern P+Q + train FULL at f29 (baseline first) | establish baseline | 1.5 | medium |
| H2 | Extend dilations [1,2,4,8] → [1,2,4,8,16,32,64] (covers seq_len 256) | IC +0.005-0.012 | 0.5 | low (just settings) |
| H3 | CC-H3 cross-asset head | IC +0.005-0.010 | 0.5 | low |
| H4 | CC-H4 anti-mem (V3 lacks ATME currently; add per-sample temporal drop) | ShIC +0.008-0.015 | 1.0 | medium |
| H5 | CC-H6 regime heads | Sharpe robust | 1.0 | low |

**V3-Headline cost**: 4.5 GPU-d (incl. baseline H1). Projected
**IC ≥ 0.080, ShIC ≥ 0.040**.

**Risk**: V3 has no current IC measurement at f29; the entire upgrade
path is hypothetical until the f29 baseline runs. **Sequence as: H1
first; if IC ≥ 0.060 baseline, proceed with H2-H5.**

---

## 7. V4 — Mamba-3 SSM + RSSM

**Current**: ShIC declining mid-training (0.0164 → 0.0132); ratio
=0.266 (memorization signature).

**V4's distinctive lever**: **linear-time sequence complexity**. Mamba
scales to 1024+ bars almost free. The current 96-bar setting wastes
the architecture.

**Headline upgrades**:

| # | Upgrade | Expected delta | Cost (GPU-d) | Risk |
|---|---|---|---|---|
| H1 | Pattern Q (already in settings) + B=32 stress probe per CLAUDE.md | establish stable training | 0.5 | low |
| H2 | seq_len 96 → 512 (Mamba's home turf) | IC +0.010-0.020 | 1.5 | medium (memory) |
| H3 | CC-H3 cross-asset head | IC +0.005-0.010 | 0.5 | low |
| H4 | Switch RK4-style step → Mamba-2 IO-aware kernel if not present | wall-clock -30% | 0.5 | medium |
| H5 | CC-H4 anti-mem ↑ | ShIC +0.005-0.012 | 0.5 | low |
| H6 | CC-H6 regime heads | Sharpe robust | 1.0 | low |

**V4-Headline cost**: 4.5 GPU-d. Projected **IC ≥ 0.085, ShIC ≥ 0.045**.

**Argument**: V4 has the only architecture in the cohort whose linear
sequence scaling makes seq_len=512 free. That's a structural advantage
nothing else has. If V4-Headline lands, it justifies its compute even
though V1.1 has been the historical favorite.

---

## 8. V6 — JEPA + Adversarial discriminator

**Current**: IC 0.024 (gate fail on ShIC decline, NOT signal-absence
per FINDINGS_COMPLEMENT spot-check).

**V6's distinctive lever**: no reconstruction (JEPA = joint-embedding
predictive). No Pattern Q dependency. Discriminator is the regularizer.

**Headline upgrades**:

| # | Upgrade | Expected delta | Cost (GPU-d) | Risk |
|---|---|---|---|---|
| H1 | Discriminator stability: spectral norm + R1 regularization (GAN-stability) | training stable | 0.5 | low |
| H2 | Discriminator on RESIDUAL not encoder output (V6 fix log idea) | ShIC +0.005-0.012 | 1.0 | medium |
| H3 | CC-H3 cross-asset head | IC +0.005-0.010 | 0.5 | low |
| H4 | CC-H4 anti-mem ↑ | ShIC +0.005-0.010 | 0.5 | low |
| H5 | Pattern P f29 | IC +0.005 | 0.5 | settings only |

**V6-Headline cost**: 3 GPU-d. Projected **IC ≥ 0.075, ShIC ≥ 0.040**.

**Failure mode**: discriminator collapses → V6-Headline fails. Apply
H1 first; if disc-loss ratio doesn't stabilize ~0.5, archive V6.

---

## 9. V8 — Neural ODE (RK4)

**Current**: untrained on this dataset; 4× compute per step.

**V8's distinctive lever**: continuous-time fits dollar bars
(event-time samples). Theoretical advantage; empirical question.

**Headline upgrades**:

| # | Upgrade | Expected delta | Cost (GPU-d) | Risk |
|---|---|---|---|---|
| H1 | Switch RK4 → Tsit5 (more efficient, same accuracy) | wall-clock -40% | 0.5 | low |
| H2 | Adjoint-method backprop (memory-efficient) | enables longer seq_len | 1.0 | medium |
| H3 | Learned step-size (current: fixed) | IC +0.005-0.010 | 1.0 | medium |
| H4 | CC-H3 cross-asset head | IC +0.005 | 0.5 | low |

**V8-Headline cost**: 3 GPU-d. Projected **IC ≥ 0.065, ShIC ≥ 0.035**.

**Honest assessment**: V8 unlikely to clear Headline tier because the
underlying continuous-time advantage on dollar bars is small. **If
H1+H2 baseline doesn't show IC > 0.050, KILL V8 and reallocate the
compute**.

---

## 10. V9 — GRU + 3-MoE (ARCHIVED for regime-leak)

**Decision**: Resurrect as V23 (proper-MoE with PREDICTED-regime gate,
not leaky-label gate). Per WM_HIGH_TIER_TARGETS V23 spec: 3-5 GPU-d.
Headline target: IC ≥ 0.080.

**Action**: Do NOT retrain V9 in current form. New file at
`src/wm/v23/v23_training/` when ready (separate session).

---

## 11. V10 — Meta-ensemble router

**Headline plan**: V10 doesn't generate IC; it **multiplies IC of
its inputs**. The Headline question for V10 is: "do its inputs have
ρ < 0.7 such that ensemble lift > 0.005 ShIC?"

**Action**: Wait for V1.x-Headline (V1.0/V1.1/V1.4/V1.6 at headline
spec), V21 cross-asset MASTER, and V24 TimesFM LoRA. Once ≥3 are
trained, run pairwise correlation. If diversity exists, V10 ensemble
should lift to **portfolio ShIC ≥ 0.05** (THE definition of Headline
in ensemble form).

---

## 12. V11 — WaveNet + MoE + Discriminator

**Current**: untrained; combined V3+V6+V9 architecture. V9's regime-leak
is INHERITED.

**Headline upgrades**:

| # | Upgrade | Expected delta | Cost (GPU-d) | Risk |
|---|---|---|---|---|
| H1 | Drop the V9 MoE component (use 1 expert); inherit V3+V6 only | training stable | 1.0 | medium |
| H2 | Apply V3's dilations + V6's discriminator stability fixes | tracks V3 + V6 progress | follow V3+V6 |  |
| H3 | If H1 stable, ADD a non-leaky predicted-regime MoE (V23-style) | IC +0.005 | 1.5 | high |

**V11-Headline cost**: 3.5 GPU-d. Projected **IC ≥ 0.080, ShIC ≥
0.040** — only if V3-Headline AND V6-Headline both land.

**Honest assessment**: V11 inherits both V3's stationarity assumption
AND V6's discriminator instability. Three failure modes stacked. **Defer
until V3-Headline and V6-Headline succeed independently.**

---

## 13. V12 — Cross-Asset Attention

**Current**: untrained; **forward_multi_asset is dead code** in
standard runner (per `world_model.py:267-271` explicit comment).

**V12 IS the Headline candidate of the frozen-evaluation cohort.**
Its architectural innovation (joint cross-asset processing) is exactly
the representation-level change that V1.x cannot do.

**Headline upgrades**:

| # | Upgrade | Expected delta | Cost (GPU-d) | Risk |
|---|---|---|---|---|
| H1 | **Fix the dataloader** to provide synchronized multi-asset batches (this unblocks the architecture) | enables training | 1.0d harness | medium |
| H2 | Train V12-FULL at f34 baseline | IC ≥ 0.060 expected | 2.0 | medium |
| H3 | Hierarchical attention: cross-asset at bar-level + temporal at sequence-level | IC +0.010-0.020 | 1.5 | medium-high |
| H4 | CC-H4 anti-mem ↑ | ShIC +0.005 | 0.5 | low |

**V12-Headline cost**: 5 GPU-d (incl. harness). Projected
**IC ≥ 0.090, ShIC ≥ 0.040**.

**Of all existing architectures, V12 has the highest ceiling for
Headline tier** — joint multi-asset processing is genuinely orthogonal
to V1.x. **Recommend V12 fix as the post-V1.0 priority**.

---

## 14. V13 — Temporal Fusion Transformer

**Current**: untrained; FROZEN-eval; ACTIVE_HORIZONS restored to
[1,4,16,64].

**V13's distinctive lever**: VSN per-timestep variable selection.
Hard top-k=8 of 34 features.

**Headline upgrades**:

| # | Upgrade | Expected delta | Cost (GPU-d) | Risk |
|---|---|---|---|---|
| H1 | VSN_TOP_K 8 → 12-16 (less aggressive bottleneck) | IC +0.005 | 0.5 | low |
| H2 | Add cross-asset VSN layer (asset-level variable selection) | IC +0.010 | 1.5 | medium |
| H3 | Pattern P+Q at f29 baseline | IC +0.005 | 0.5 | settings |
| H4 | CC-H6 regime heads | Sharpe robust | 1.0 | low |

**V13-Headline cost**: 3.5 GPU-d. Projected **IC ≥ 0.075, ShIC ≥
0.038**.

**TFT was designed for hourly/daily data with calendar features**;
dollar bars don't have calendar structure. V13's ceiling on this
dataset is genuinely lower than V12's. **Run only if V12-Headline
succeeds and there's surplus compute**.

---

## 15. V14 — Diffusion Return Distribution

**Current**: untrained; 50 inference steps × 32 samples = ~500× V1.x
inference cost.

**V14's distinctive lever**: full distribution output. Strategy-side
can size on quantile structure (q05/q50/q95).

**Headline upgrades**:

| # | Upgrade | Expected delta | Cost (GPU-d) | Risk |
|---|---|---|---|---|
| H1 | Reduce DIFFUSION_INFERENCE_STEPS 50 → 10 (DDIM with classifier-free guidance) | inference 5× faster | 0.5 | low |
| H2 | Reduce DIFFUSION_N_SAMPLES 32 → 8 | inference 4× faster | 0.5 | low |
| H3 | Pattern P+Q at f29 baseline | IC ≥ 0.060 expected | 1.5 | medium |
| H4 | Distributional Sharpe via meta-learner accepting q-vector input (CC2 from STRAT_SOTA_REVIEW) | tradeable Sharpe +0.10 | 1.0 (strat-side) | low |
| H5 | CC-H3 cross-asset conditioning | IC +0.005-0.010 | 1.0 | medium |

**V14-Headline cost**: 4.5 GPU-d (1.5 of which is strategy-side wiring).
Projected **IC ≥ 0.075, ShIC ≥ 0.040, distributional Sharpe ≥ 2.5**.

**The Headline argument for V14**: even at IC = 0.075 (sub-Headline raw
IC), the **distributional Sharpe** can exceed standard Sharpe by 0.30+
because position sizing on full distribution dominates point-estimate
sizing in fat-tail regimes. **V14-Headline pursued via Sharpe metric,
not IC metric** — which is consistent with the lens (deployable
real-capital model, not academic IC chase).

---

## Aggregate sequencing — 8 weeks of Headline pursuit

If the user signs off on the Headline lens being applied universally,
the GPU plan is roughly:

**Weeks 1-2 (V1.x cohort Headline retrain — 12-15 GPU-d)**:
- V1.0-Headline (3 GPU-d)
- V1.1-Headline (2.5 GPU-d) — **HIGHEST EV, run first**
- V1.4-Headline (3.5 GPU-d)
- V1.6-Headline (2 GPU-d)
- + V1.6 ablation (4 GPU-d, parallel)

**Week 3 (V12 fix + train — 5 GPU-d)**:
- V12 dataloader harness fix (1 GPU-d)
- V12-FULL baseline (2 GPU-d)
- V12-Headline (2 GPU-d)

**Week 4 (V3/V4/V6 Headline — 12 GPU-d)**:
- V3-Headline (4.5)
- V4-Headline (4.5)
- V6-Headline (3)

**Week 5 (Aspirational architectures from WM_HIGH_TIER_TARGETS — 10
GPU-d)**:
- V21 multi-asset MASTER (3-4)
- V24 TimesFM LoRA (2-3)
- V14-Headline (4.5)

**Week 6 (Decision gate)**:
- DSR/PBO across all Headline retrains
- Pairwise V1.x-Headline correlation
- V10 ensemble of survivors
- KILL list — anything that didn't clear Trader tier on Headline-spec
  retrain

**Weeks 7-8 (deployment)**:
- HEADLINE-TIER survivors enter V10 ensemble
- Anti-correlated sleeves (per F4 in STRAT_PHASE5_SPEC) constructed
- Real-capital deployment with calibrated p_fill, capacity caps,
  reflexivity-cascade gate

**Total**: ~50-55 GPU-d across 8 weeks for the full headline pursuit.
Compare to ~30 GPU-d for the conservative SHIP-cohort retrain.

The Headline pursuit costs **2× the conservative path** and yields
**a model class that was previously absent** — the agent-teaching tier
where the WM signal IS the alpha.

---

## Failure-mode acceptance

If after the 50-55 GPU-d Headline pursuit, **no model clears
IC ≥ 0.10 / ShIC ≥ 0.05**:

1. **Don't deploy at sub-Headline as if it were Headline** — that's
   the Tier-A multi-strat trap. Calibrate sleeve weights against the
   actual numbers.
2. **The lens still applies**: every model gets retrained against the
   Trader-tier acceptance criteria with the Headline-target hypothesis
   documented in D10.
3. **Escalate to V20 (tick-level)**: the only architectural class that
   credibly targets IC > 0.10 if dollar-bar representations are
   structurally capped.

The lens isn't "every model must hit Headline" — it's **"every model
must be BUILT FOR Headline AND scored against it, even if it lands at
SHIP"**. That changes the design space and the rubric incentive.

---

## See also

- [CLAUDE.md "Indisputable Operating Lens"](../CLAUDE.md) — the mandate
- [WM_HIGH_TIER_TARGETS_2026_04_30.md](WM_HIGH_TIER_TARGETS_2026_04_30.md) — V20-V24 NEW architecture proposals
- [WM_SCORESHEET_MERGED_2026_04_29.md](WM_SCORESHEET_MERGED_2026_04_29.md) — D1 super-tier (+5 bonus)
- [WM_FINDINGS_2026_04_29.md](WM_FINDINGS_2026_04_29.md) — current per-version status
- [STRAT_PHASE5_SPEC_2026_04_30.md](STRAT_PHASE5_SPEC_2026_04_30.md) — anti-correlated sleeve construction (downstream consumer of Headline-tier outputs)
