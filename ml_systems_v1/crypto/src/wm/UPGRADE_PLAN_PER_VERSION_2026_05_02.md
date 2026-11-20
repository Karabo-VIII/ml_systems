# Per-Version Upgrade Plan — 2026-05-02

> Detailed evaluation per-version: **upgrade to SOTA** OR **archive**.
> Per user 2026-05-02: don't archive prematurely; develop each model to
> the very best of what it is. If after evaluation it's strictly
> dominated, archive — but only after honest upgrade-attempt.

This document is the **canonical roadmap** for second-wave wiring +
defer-tier evaluation + V15-V19 SOTA implementation. Each section names
specific architecture changes, expected lift envelopes, and ship/archive
verdicts grounded in browser dialog evidence (B001-B006).

## Wave 1 — V1.x sibling wiring (mechanical port from V1.1)

| Version | Status | Effort | Notes |
|---|---|---|---|
| **V1.1** | ✅ FULLY WIRED | done | 6 flags functional |
| **V1.0** | ✅ FULLY WIRED (this turn) | done | model + trainer; smoke PASS on baseline/mtp/mdn/adaptive |
| **V1.4** | 🟡 PARTIAL (model __init__ + forward_train branched) | ~1 hr remaining | get_loss return_components + MDN paths + trainer wiring |
| **V1.6** | ⏳ pending | ~2-3 hrs | KL anneal + Gumbel + ATME + dream interaction; PCGrad may conflict with KL anneal schedule. **Test PCGrad on V1.6 only AFTER V1.0/V1.1 results land** |

## Wave 2 — Different-architecture wiring

### V12 (Cross-Asset Attention, 841K params)

**Verdict: HIGH-PRIORITY UPGRADE** (best architectural fit for foundation alignment).

Wire-up plan:
1. Port the same `__init__` hooks + forward_train branches + get_loss
   `return_components` pattern as V1.0/V1.1
2. **Special consideration**: V12's cross-asset attention emits per-asset
   embeddings; PCGrad surgery should treat per-asset losses as separate
   tasks alongside per-horizon → 4 horizons × 10 assets = 40 tasks
   (heavy memory). **Default to per-horizon only** (4 tasks), per-asset
   as opt-in `--pcgrad-per-asset`
3. MTP particularly natural fit: cross-asset attention output → MTPHead
   chain
4. SAM applies cleanly

Effort: ~3-4 hrs. Likely the highest-leverage second-wave wiring.

### V4 (Mamba-3 + RSSM, 3.5M params)

**Verdict: UPGRADE WITH ROOT-CAUSE INVESTIGATION**.

V4 has QKNorm in code (lines 15, 196, 241, 294 of components.py) so the
B004 R1 fix is already done. But ShIC declines mid-training despite
this. Two hypotheses:

1. **Batch-size curriculum** (per B004): SSD chunk-based scan may need
   batch=64 vs current 32. Easy to test.
2. **Different RSSM bottleneck size** vs V1.x's 24×24: V4 uses 32×32
   (`SSM_D_STATE`). Possibly over-parameterized at 3.5M for our data.
3. **FSQ replacement** (B004 R2): replace 24×24 categorical RSSM with
   FSQ projection; arXiv 2309.15505 verified "no codebook collapse."
   ~5 GPU-h probe.

Wire-up plan:
1. Port the V1.x flag pattern (6 flags) into V4 trainer
2. Add `--batch-curriculum` flag (32 → 64 grad-accum) for B004 §3.3 test
3. Add `--fsq-bottleneck` flag for B004 R2 test
4. Run probe matrix: V4 baseline / V4+SAM / V4+SAM+batch64 / V4+FSQ

Effort: ~6-8 hrs (model file is more complex due to Mamba-3 internals).

### V3 (WaveNet, 1.9M params)

**Verdict: SUBSET-UPGRADE**. Single-head architecture (no per-horizon
multi-task). PCGrad and MTP n/a. SAM, FrAug, MDN, Adaptive bins apply.

Wire-up plan:
1. Port `__init__` hooks for adaptive_bins / MDN
2. Port forward path branches (single-head simplifies; MDN replaces
   the head's bin output with parametric distribution)
3. Trainer: 4 flags only (`--sam`, `--fraug`, `--mdn`, `--adaptive-bins`)

Effort: ~3 hrs. Lower priority than V12/V4.

## Wave 3 — Defer-tier evaluation (NOT premature archive)

### V8 (Neural ODE + RSSM)

**Per B005 §3**: pure NODE strictly dominated by Mamba SSD on financial
small-cap. 2024-2026 frontier: Mamba+NODE hybrids (MODE arXiv 2601.00920).

**Honest upgrade attempt before archive**:
- **Option A**: keep NODE; add B005 R1 SSL-pretrain (1 epoch masked
  reconstruction) + finetune. ~6 GPU-h. If lift ≥ +0.005 IC over baseline,
  V8 ships as ensemble member.
- **Option B**: replace pure NODE with Mamba-NODE hybrid (V8' = V4 backbone
  + latent NODE residual). This is effectively a NEW version (V21+),
  not V8 retrofit. **3-5 days engineering**.

**Verdict**: try Option A as 1-shot probe (6 GPU-h). If passes → V8
stays in cohort. If not → ARCHIVE V8 to `backups/BKP_<future>_V8_NODE_ARCHIVE/`
and file Mamba-NODE hybrid as new V21 proposal.

### V10 (Meta-Ensemble Aggregator)

**Verdict: NO UPGRADE NEEDED YET (not premature; structural defer)**.

V10 is an aggregator over trained inputs. Until V1.x retrains (post-upgrade)
land + V4 retrains land + V12 retrains land, V10 has nothing to aggregate
that's different from current. Keep V10 frozen; revisit when ≥ 2 upgrade-
validated inputs exist.

**Optional enhancement** (does NOT block): replace V10's current weighted-
average aggregator with a small **stacked meta-learner** (Ridge or LightGBM)
using leave-one-out CV. Per AFML chapter 9, stacking outperforms uniform
averaging when constituent models are diverse. ~2 GPU-h work; no model
training needed (just inference + LightGBM fit).

### V11 (WaveNet + MoE + Discriminator)

**Verdict: STAY FROZEN** per B005, BUT consider **upgrading the MoE
layer to 2025-style sparse-MoE**.

Current V11: 3 dense experts at 2.9M total → ~970K per expert.

Modern frontier sparse MoE (DeepSeek-V3 / Mixtral / Llama 4):
- 16-32 experts at smaller per-expert size
- Top-2 routing (dense uses all 3)
- Auxiliary load-balancing loss

**At V11's scale (2.9M)**, sparse MoE doesn't help (per B005 — "below
the threshold where sparse MoE is meaningfully sparse"). True. BUT V11
could be **scaled to 8M** with 16 small experts at 500K each, top-2
routing. That keeps total params manageable while testing the modern
recipe.

**Honest upgrade attempt**: scale V11 to V11' at 8M with 16-expert MoE +
top-2 routing + load-balancing loss (Shazeer 2017 / Switch Transformer).
~8 GPU-h probe. If IC > V11 baseline by ≥ +0.01 → ship V11' as new
ensemble member. If not → V11 stays frozen.

### V13 (Temporal Fusion Transformer, 2.2M params)

**Verdict: STAY FROZEN per B005; build TFT-GNN as V20+**.

TFT itself is mature; the 2025 frontier is **TFT-GNN hybrid** [REPORTED
preprint 202510.2481] — outperformed standalone TFT in 11/12 evaluated
periods on stock prediction.

**Honest upgrade attempt before archive**:
- Keep V13 as the canonical TFT-VSN reference
- File **V20 = TFT-GNN hybrid** as new architecture (graph-attention
  layer over u100 assets + V13's VSN machinery)
- ~5 days engineering (graph-attention layer + PyG dependency + V13
  encoder reuse)

**Verdict**: don't archive V13; it's the canonical VSN reference. Do
build V20 = TFT-GNN as new version.

### V14 (Diffusion return distribution)

**Verdict: REVIVE WITH CAUTION** per B005 R3.

Two-gate revival:
1. **Quantile-vector consumption probe** (no GPU; strategy layer): can
   the meta-learner ingest 5-quantile vectors per horizon and beat
   scalar-mean baseline by ≥ +0.005 IC?
2. **Diffolio number verification** (WebFetch): arXiv 2511.07014 paper
   body for actual IC numbers before committing GPU-h.

If both PASS → V14 retrain with `--sam` + `--fraug` (PCGrad/MTP/MDN
n/a — V14 IS a parametric distribution model). ~5 GPU-h.

If either FAILS → V14 stays frozen. Don't archive yet — diffusion is a
2025 frontier worth keeping the codebase warm for.

## Wave 4 — V15-V19 SOTA implementation (per user: not stick models)

### V15 (PatchTST encoder stub) → **V15 SOTA: PatchTST production**

Current state: encoder stub at `src/wm/v15/patchtst_encoder.py`.

**SOTA implementation plan**:
- Full PatchTST per Nie 2023 (arXiv 2211.14730)
- Patch length = 16 dollar bars
- Stride = 8 (50% overlap)
- Channel-independent: per-feature transformer encoder
- Multi-horizon TwoHot heads at h={1,4,16,64}
- ~3M params at d_model=256, 4 layers
- Wire all 6 V1.x upgrade flags

Compute: 1-2 weeks engineering (full architecture + trainer). ~5-8 GPU-h
per training run.

### V16 (DreamerV3) → **V16 SOTA: DreamerV3 for crypto WM**

Current state: stub.

**SOTA implementation plan**:
- DreamerV3 per Hafner 2025 Nature
- Replace V1.x's RSSM with DreamerV3's recurrent state-space model
- Symlog targets (DreamerV3 standard)
- Predict reward + value + return distribution
- World-model + actor + critic (3-component training)
- ~4M params

⚠ **Operational caveat**: DreamerV3 requires reward feedback loops; for
crypto, the "reward" is realized return → can be supervised. Per B005
no published live financial deployment; this would be exploratory.

Compute: 2-3 weeks engineering. ~12 GPU-h per training run.

**Honest assessment**: V16 may not lift IC for OUR project before 2027.
The framework is for sequential decision-making with rewards; we have
that signal but our IC ceiling on dollar bars (~0.13 per CLAUDE.md) may
not need imagination/rollouts.

### V17 (TD-MPC2) → **V17 SOTA: TD-MPC2 model-based RL**

Current state: stub.

**SOTA implementation plan**:
- TD-MPC2 per Hansen 2024 (arXiv 2310.16828)
- Latent dynamics model + value function + policy
- Cross-entropy planner at inference (search over latent actions)
- ~3M params

⚠ **Operational caveat**: same as V16. RL framework; benefit on
predictive IC unclear.

Compute: 2-3 weeks engineering. ~10 GPU-h per training run.

### V18 (Chronos finetune) → **V18 KILL CONFIRMED**

Per B001 + B005: domain-mismatch (general-purpose foundation model on
crypto); empirically rejected via E1 + E1c (Kronos as analogous, IC +0.029).
**Archive V18 directory**; do NOT spend SOTA effort here.

### V19 (V1.x at f121) → **V19 SOTA: f121 with TFT-style VSN**

Current state: stub.

**SOTA implementation plan**:
- Take V1.4's FeatureAttentionBlock + scale to 121 input features
- Add TFT-style Variable Selection Network (VSN) so per-bar feature
  selection prevents the curse of dimensionality at 121-dim input
- Per B005 §7: at 2M params + S=96 + 121 features, capacity is
  regularization-bound, not capacity-bound → VSN is the right add
- Wire 6 V1.x upgrade flags

Compute: 1 week engineering. ~5 GPU-h per training run.

**Per B005 V19 verdict**: defer until v51 (with f121 features) lands.
Once it does, V19 = V1.4 + VSN + 121 features is a solid SOTA target.

## Composite roadmap (priority order)

| Phase | Items | GPU-h | Engineering hrs |
|---|---|---|---|
| **1A** (this turn done) | V1.0 wiring | done | done |
| **1B** (next session) | V1.4 + V1.6 wiring | 0 | ~4 |
| **1C** (next session) | V12 wiring (HIGH PRIORITY) | 0 | ~4 |
| **2** (V1.x probe outcomes determine) | V4 + V3 wiring | 0 | ~10 |
| **3** (after V1.x probes land) | V1.x SAM probe + propagation | 12-24 | ~0 |
| **4** (gated on V1.x success) | V8 SSL probe / V11' MoE upgrade / V14 revival | 20-40 | ~10 |
| **5** (project-tier-frontier) | V19 (f121+VSN), V15 (PatchTST), V20 (TFT-GNN) | 30-50 | ~80 |
| **6** (R&D-tier) | V16 (DreamerV3), V17 (TD-MPC2), V21 (Mamba-NODE) | 40-80 | ~150 |

**Total to clear Phases 1-3 (the practical Headline path)**: ~24 GPU-h
+ ~18 engineering hrs.

## What to archive — only after upgrade attempts fail

| Version | Archive trigger |
|---|---|
| V8 | If SSL-pretrain probe fails to lift IC ≥ +0.005 |
| V9 | Already KILL CONFIRMED; archive now |
| V11 | If sparse-MoE V11' upgrade fails to beat V11 baseline by ≥ +0.01 |
| V18 | KILL CONFIRMED; archive now |

V10, V13, V14, V15-V19 do NOT archive — they're either structurally
useful (V10 aggregator, V13 VSN reference) or they're frontier targets
worth implementing properly.

## Reliability caveat (carried through every recommendation)

0% VERIFIED on IC/ShIC deltas across browser dialogues B001-B006. All
lift estimates are INFERRED extrapolations from non-IC published metrics.
**Probes after baseline f29 finishes decide. Do not commit large compute
on INFERRED estimates alone.**
