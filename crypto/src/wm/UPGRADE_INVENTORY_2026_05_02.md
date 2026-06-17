# Upgrade Inventory — 2026-05-02

> Single-pane status of which model versions received the V1.x upgrades
> from the 2026-05-02 browser dialogue rounds (B001-B007). Source modules at
> [`src/frontier_ml/v1_upgrades/`](../frontier_ml/v1_upgrades/).
>
> Each upgrade is **opt-in via CLI flag**, default OFF. Versions that
> don't carry the flags here have NOT had the wiring patched in yet —
> they're candidates for "second-wave" wiring once the V1.1 probe
> outcomes land.

## Upgrade matrix

| Version | SAM | FrAug | PCGrad | MTP | Adaptive bins | MDN | VICReg | Status |
|---|---|---|---|---|---|---|---|---|
| **V1.1** | ✅ flag | ✅ flag | ✅ flag | ✅ flag | ✅ flag | ✅ flag | n/a | **PROBE-READY** |
| V1.0 | ⏳ pending | ⏳ pending | ⏳ pending | ⏳ pending | ⏳ pending | ⏳ pending | n/a | needs wiring |
| V1.4 | ⏳ pending | ⏳ pending | ⏳ pending | ⏳ pending | ⏳ pending | ⏳ pending | n/a | needs wiring |
| V1.6 | ⏳ pending | ⏳ pending | ⏳ pending | ⏳ pending | ⏳ pending | ⏳ pending | n/a | needs wiring |
| V3 (WaveNet) | n/a (transformer-only) | ⏳ pending (input-side) | n/a (single-head) | ⏳ pending | ⏳ pending | ⏳ pending | n/a | needs wiring |
| V4 (Mamba-3) | ⏳ pending | ⏳ pending (input-side) | ⏳ pending | ⏳ pending | ⏳ pending | ⏳ pending | n/a | needs wiring; **B004 R1 QKNorm already in code** |
| V6 (JEPA) | ⏳ pending | ⏳ pending | n/a (single objective) | n/a | n/a | n/a | ⏳ **PRIMARY upgrade per B005 R1** | needs VICReg wire |
| V8 (Neural ODE) | ⏳ low priority | ⏳ low | n/a | ⏳ low | ⏳ low | ⏳ low | n/a | DEFER (per B005: NODE dominated by Mamba) |
| V9 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | **KILL** per B005 |
| V10 (Meta-ensemble) | n/a (no signal layer) | n/a | n/a | n/a | n/a | n/a | n/a | DEFER until ≥ 2 trained inputs |
| V11 (WaveNet+MoE+Disc) | ⏳ low priority | ⏳ low | n/a | n/a | ⏳ low | ⏳ low | n/a | STAY FROZEN per B005 |
| V12 (Cross-Asset Attn) | ⏳ pending | ⏳ pending | ⏳ pending | ⏳ pending | ⏳ pending | ⏳ pending | n/a | needs wiring; high-priority due to xattn alignment with foundation |
| V13 (TFT) | ⏳ pending | ⏳ pending | ⏳ pending | n/a | ⏳ pending | ⏳ pending | n/a | STAY FROZEN per B005 |
| V14 (Diffusion) | n/a (denoiser-style) | ⏳ pending | n/a | n/a | n/a | n/a | n/a | REVIVE WITH CAUTION per B005 R3 |
| V15 (PatchTST stub) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | library only |
| V16 (DreamerV3) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | DEFER (no fin deployment evidence) |
| V17 (TD-MPC2) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | DEFER (no fin app) |
| V18 (Chronos finetune) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | KILL CONFIRMED per B005 |
| V19 (V1.x at f121) | ⏳ via V1.x | ⏳ via V1.x | ⏳ via V1.x | ⏳ via V1.x | ⏳ via V1.x | ⏳ via V1.x | n/a | DEFER until v51 lands |

Legend: ✅ = wired + smoke-passed • ⏳ = pending wire-up • n/a = not applicable (architectural mismatch or model dead)

## Foundational vs Structural classification

Per the 2026-05-02 deliberation (and aligned with B003's EV/GPU-h ranking
modulo metric):

- 🔴 **FOUNDATIONAL** (move the IC ceiling): **SAM, PCGrad, MTP, MDN**
- 🟡 **STRUCTURAL** (free polish, additive): **Adaptive bins, FrAug**
- 🟢 **V6-SPECIFIC**: **VICReg**

Path to Headline (IC > 0.10 / ShIC > 0.05) for V1.x:
```
baseline f29           IC 0.067, ShIC 0.033
+ SAM                  +0.005-0.010 ShIC
+ PCGrad               +0.005-0.010 IC
+ MTP                  +0.005-0.015 IC
+ MDN                  +0.005-0.020 IC (tail capture)
+ adaptive bins +FrAug +0.005-0.015 ShIC (polish)
=====================  IC 0.085-0.105 / ShIC 0.050-0.060 [INFERRED]
```

## Module locations

```
src/frontier_ml/v1_upgrades/
├── sam.py                     B003 R1: Sharpness-Aware Minimization
├── pcgrad.py                  B003 4.6: gradient surgery for multi-horizon
├── fraug.py                   B003 R2: FFT-mask augmentation
├── mtp_head.py                B002 R1: Multi-Token Prediction sequential head
├── mdn_head.py                B003 R3: NormalMDNHead + SkewedStudentTHead
├── vicreg.py                  B005 R1: VICReg variance/invariance/covariance
├── adaptive_conformal.py      B007 E1: ACI online-tuned coverage wrapper
├── label_noise.py             B007 E2: calibrated Gaussian label-noise injector
├── isotonic_calibrator.py     B007 §3.2: per-bin isotonic post-hoc TwoHot calibration
├── logit_clip.py              B007 §5.2: bounded-norm logits (anti-memorization)
├── iqn_head.py                B007 §7.3: implicit-quantile continuous head
├── test_time_training.py      B006: TTT for non-stationarity
├── koopman.py                 B006: Koopman linear-evolution wrapper
├── born_again.py              B006: self-distillation
└── integration.py             apply_v1_upgrades(model, **flags) entry point

src/frontier_ml/foundation/adaptive_bins.py    B001 R3: log-spaced bins
```

## B007 module verdicts (2026-05-02 browser response)

| Module | Tier | When to use | Decision gate |
|---|---|---|---|
| `adaptive_conformal.py` | RISK-MGMT (NOT default sizing) | Inference wrapper around any V*.x quantile/TwoHot/MDN head | **TESTED V1.1 BTC VAL 535K windows: dSortino -0.035 across 4 sizing paradigms; CONCEDE default-deployment.** Keep as zero-cost coverage tool for risk dashboards / position caps; do NOT use as alpha sizing. |
| `label_noise.py` | EXTRA (anti-mem) | V1.x retrain flag `--label-noise` | ShIC delta ≥ +0.005 with IC stable → cohort-wide |
| `isotonic_calibrator.py` | DEPLOYMENT | Post-train, pin per checkpoint on OOS slice | ECE drop > 50% AND IC non-degraded |
| `logit_clip.py` | EXTRA (anti-mem) | V1.x retrain flag `--logit-clip` | A/B vs SAM-only baseline |
| `iqn_head.py` | FOUNDATIONAL | Replacement for TwoHot 255-bin head | CRPS lift ≥ 5% AND IC ≥ +0.003 → ship to V1.x + V17 |
| `cgfm_residual.py` | UTILITY (NOT V1.x add) | Residual-flow distributional refinement | **TESTED V1.1 BTC VAL h=1+h=64: IC delta -0.03 to -0.22; CRPS 22x worse vs fair bin-baseline.** V1.1's 255-bin symlog distribution is already a strong distributional model; CGFM underfits the narrow conditional density. Module stays shipped (useful for future predictors with weaker distributional heads); do NOT promote to V1.x add. |

**Concedes from B007** (drop from cohort plan):
1. Liquidation-cascade-features-as-directional-alpha (academic test came back null)
2. MEV-as-feature for directional return (cost-side phenomenon, not directional)
3. Hyperbolic embeddings + Score-based regression + EBM (no 2024-2026 crypto evidence)
4. Born-Again iterations past Generation 2 (deltas under +0.003 ShIC unworth)
5. Per-asset training of V12 IF asset-as-token reframe is adopted

**B007 cross-finding (post-E1+E3 concedes)**:
Distributional inference-time wrappers on V1.x are not the IC bottleneck.
V1.1 baseline IC at h=1 is +0.039 and at h=64 is +0.211 -- already strong.
ACI/CGFM mean-of-samples add sampling noise that washes out the
directional signal. The next yield comes from training-paradigm
changes (E2 label-noise, SAM/MTP/MDN flags wired but untrained), not
from inference-time distributional augmentations.

**Queued for next-round build** (not yet shipped):
- CDSeer drift detector (orchestrator-level, B007 §6.1)
- Asset-as-token reframe of V12 (8-12 GPU-h retrain, B007 §4.1)
- LoRA-per-asset adapter on shared backbone (6-10 GPU-h, B007 §4.2)
- Volatility-gated MoE-over-assets at V10 (4-6 GPU-h, B007 E4)
- FinCast zero-shot baseline (E5, conditional on weights release)

## Models that got upgrades this round

**V1.1 only.** Trainer flags `--sam`, `--fraug`, `--pcgrad`, `--mtp`,
`--adaptive-bins`, `--mdn` are functional. world_model.py refactored to
expose per-horizon loss components (PCGrad), MTP-aware forward, and MDN-
aware loss + inference paths.

## Models that did NOT get upgrades — probing list

**Priority order for second-wave wiring** (after V1.1 probe outcomes land):

1. **V1.0** — same architecture as V1.1; minimal-effort port (literal
   diff of trainer + world_model.py). If V1.1+SAM lifts ShIC by ≥+0.005,
   port to V1.0 immediately and re-validate the V1.0 reference baseline.
2. **V1.4** — FeatureAttentionBlock variant; same trainer pattern. Port
   after V1.0.
3. **V1.6** — has KL anneal + Gumbel + ATME + dream; some upgrades may
   conflict (especially PCGrad with KL anneal). Port carefully.
4. **V12 (Cross-Asset Attention)** — already aligned architecturally
   with foundation prong; SAM + PCGrad + MTP could meaningfully lift V12.
5. **V4 (Mamba-3)** — orthogonal architecture; SAM/PCGrad/MDN should
   transfer. MTP requires per-horizon-head equivalent in V4's structure.
   B004 R1 noted V4 already has QKNorm — investigate why ShIC still
   declines.
6. **V3 (WaveNet)** — single-head causal-conv; PCGrad/MTP n/a, but SAM
   and MDN apply. FrAug applies at input.
7. **V14 (Diffusion)** — REVIVE WITH CAUTION per B005 R3. Quantile-
   vector consumption probe at strategy layer first.

## Models with NO upgrade plan (defer / kill)

- **V8** — Neural ODE strictly dominated by Mamba per B005 §3
- **V9** — KILL CONFIRMED per B005
- **V11, V13** — STAY FROZEN per B005
- **V15, V16, V17** — library / experimental stubs
- **V18 (Chronos)** — KILL CONFIRMED per B001 + B005

## Provenance

Browser dialogues 2026-05-02:
- B001 ([RESEARCH_BRIEF](../frontier_ml/RESEARCH_BRIEF_2026_05_02.md) → [RESPONSE](../frontier_ml/FRONTIER_RESEARCH_RESPONSE_2026_05_02.md))
- [B002 V1.x lab overlay](../frontier_ml/browser_dialog/RESPONSE_B002_frontier_lab_overlay.md)
- [B003 V0+ envelope push](../frontier_ml/browser_dialog/RESPONSE_B003_v0plus_envelope_push.md)
- [B004 V2/V5 archived decision](../frontier_ml/browser_dialog/RESPONSE_B004_v2_v5_models_upgrade.md)
- [B005 V5+ models upgrade](../frontier_ml/browser_dialog/RESPONSE_B005_v5plus_models_upgrade.md)
- [B006 new frontiers](../frontier_ml/browser_dialog/RESPONSE_B006_new_frontiers.md)
- [B007 complementary frontier](../frontier_ml/browser_dialog/RESPONSE_B007_complementary_frontier.md) — ACI / label-noise / isotonic / logit-clip / IQN modules shipped 2026-05-02

Reliability caveat preserved across all responses: 0% VERIFIED on IC/ShIC
deltas. All probe lifts are INFERRED. Decision rule per probe:
**ShIC ≥ 0.038 (+0.005 vs current 0.033 record) → propagate to other V1.x.**
