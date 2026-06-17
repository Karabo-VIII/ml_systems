# V6 — Causal JEPA + Adversarial Time Shuffling (SOTA-2026)

> **Role in cohort**: JEPA-based architecture. Tests whether joint-embedding
> predictive learning (no reconstruction; latent-target distillation) beats
> reconstruction-based RSSM on dollar-bar regimes.
>
> **Status**: V6 not yet retrained on SOTA-2026 defaults. Was ⚠ UNDERSIZED
> (2.23M params) pre-2026-05-07 capacity bump; now 4.4M (HEADLINE_MODE).

## Purpose

V6 abandons V1.x's "reconstruct obs from latent" objective and adopts JEPA
(Joint-Embedding Predictive Architecture, LeCun 2022): the encoder predicts
**target encoder's latent** at future timesteps, not raw observations. The
target encoder is an EMA-momentum copy of the context encoder (frozen).

The bet: reconstruction wastes capacity reconstructing noise (returns are
inherently noisy). JEPA latent-space prediction is more efficient — the
encoder only needs to capture features that PREDICT future latents.

Bonus: **adversarial time discriminator** as anti-memorization. The encoder
must produce latents that the discriminator can't distinguish from
time-shuffled latents. If the encoder leaks temporal structure, the
discriminator wins; if the encoder learns truly time-invariant features,
the discriminator can't.

## Architecture (SOTA-2026 post-upgrade)

```
Obs (B, T, F) + asset_emb (frozen target sees UNMASKED obs)
                                                    │
  ┌─ context branch (TRAINED) ─┐    ┌─ target branch (EMA, FROZEN) ─┐
  │ obs_proj → CausalGRU       │    │ target_obs_proj → target_encoder │
  │ → context_latent_proj      │    │ → target_latent_proj           │
  │ → ctx_latent [B, T, 192]   │    │ → tgt_latent [B, T, 192]       │
  └────────────────────────────┘    └────────────────────────────────┘
            │                                       │ (stop_gradient)
            │ RegimeFiLM(ctx_latent, regime_probs)──┤
            ▼                                       │
  ┌─ VIB ─────────────────────────────────────────┐ │
  │ ctx_latent[192] → vib_mu/logvar[48]           │ │
  │ → z_vib[48] → vib_expand → feat[192]          │ │
  └───────────────────────────────────────────────┘ │
            │                                       │
            ├── predictor(feat) → pred_latent ──────┤── InfoNCE + VICReg
            │                                       │
            ├── recon_head(feat) → recon          (encoder gradient via VIB path)
            ├── time_discriminator(ctx_latent_pre_VIB) → real_score
            │   (residual target per disc fix)
            │
            └── feat.detach() → heads
                 ├── return_trunk → return_heads (TwoHot 255 bins × 4 horizons)
                 ├── regime_head → 3-class
                 ├── CC-H5 quantile_heads (q05..q95 × 4 horizons)
                 └── CC-H6 regime_cond_heads (3 × 4 = 12 per-regime decoders)
```

### Anti-memorization (multi-mechanism)

1. **InfoNCE (per-timestep contrastive)** — predicts target latent, separates from negatives
2. **VICReg** — invariance + variance + covariance regularization on pred vs target
3. **Variational Information Bottleneck (VIB)** — KL-constrained low-bit latent
4. **Adversarial Time Discriminator** — penalizes any temporal dependence in latent
5. **`feat.detach()` on heads** — return loss can't drive encoder memorization
6. **Forecast head as encoder anchor** — predict obs[t+h] from feat (queued generalization fix)
7. **RegimeFiLM (SOTA-2026)** — encoder-level regime conditioning

### Design rationale

- **Why JEPA over RSSM**: JEPA captures predictive features without
  reconstructing noise; for high-noise return targets, this is a more
  efficient use of capacity.
- **Why EMA target encoder**: stable target distribution; prevents the
  representation-collapse pathology where the predictor and target collapse
  to a constant.
- **Why VIB**: hard rate constraint on temporal info flowing into the heads.
  Without it, the encoder + discriminator pair can game each other while the
  decoder leaks temporal structure through the cardinality of the latent.
- **Why disc-target = ctx_latent (NOT feat)**: per the 2026-05-16 fix shipped
  in commit 8afb3e1, the discriminator must target the PRE-VIB encoder
  residual. Targeting post-VIB feat gives the GAN pressure no useful gradient
  signal (the bottleneck removed the temporal info before the disc sees it).
- **Why `feat.detach()` on heads**: V6's return loss is 7.4× weighted; without
  detach, that gradient dominates and drives the encoder to memorize
  return-correlated temporal structure. Detach forces encoder gradient to
  come from InfoNCE/VICReg/disc/recon ONLY.
- **Why d_model=320 / n_layers=4 (post 2026-05-07 bump)**: pre-bump 2.23M was
  UNDERSIZED per iron-clad audit. Post-bump 4.4M clears the 4M iron-clad floor.
- **Why TEMPORAL_CTX_DROP=0.0**: JEPA has no h_seq/z_post split where ATME
  applies; the bottleneck mechanism is different (VIB + disc + InfoNCE).
- **Why RegimeFiLM on ctx_latent (not feat)**: pre-VIB is the correct
  encoder-level conditioning point; identity-at-init means no early disruption.

## Files

```
src/wm/v6/v6_training/
├── settings.py              # config (incl. HEADLINE_MODE disc fixes)
├── components.py            # CausalGRU + TimeDiscriminator + JEPA-specific
├── world_model.py           # CausalJEPAWorldModel (627 lines)
├── train_world_model.py     # full trainer w/ disc-aware gradient flow
├── validate_world.py
└── adapter.py / snapshot_ensemble.py / ncl_model.py
```

### Sub-versions

| Sub | Hypothesis |
|---|---|
| `v6_training` | base (canonical) |
| `v6_1_training` | alt config |
| `v6_2_training` | alt config |
| `v6_3_training` | alt config |

## Usage

```bash
# Train (SOTA-2026 defaults — HEADLINE_MODE on by default, gives disc spectral_norm + R1 + residual target)
python src/wm/v6/v6_training/train_world_model.py --features 29

# Legacy mode
V6_HEADLINE_MODE=0 python src/wm/v6/v6_training/train_world_model.py --features 29

# Validate
python src/wm/v6/v6_training/validate_world.py
```

## Key settings (SOTA-2026 vs pre-upgrade)

| Setting | Pre | Now (SOTA-2026) |
|---|---|---|
| `WM_D_MODEL` | 256 | **320** (post 2026-05-07 capacity bump) |
| `WM_N_LAYERS` | 3 | **4** (post 2026-05-07) |
| `WM_FREE_NATS` | 1.0 | **1.5** (CC-H4) |
| `XD_DROPOUT_RATE` | 0.7 | (unchanged; JEPA has no XD-split anyway) |
| `HEADLINE_MODE` | OFF | **ON by default** (disc spectral_norm + R1 + residual target) |
| `USE_QUANTILE_HEADS` | n/a | **True** (CC-H5) |
| `USE_REGIME_COND_HEADS` | n/a | **True** (CC-H6) |
| `REGIME_AWARENESS_MODE` | n/a | **"film"** (CC-H6 + FiLM on ctx_latent) |
| Disc target | feat (post-VIB) | **ctx_latent (pre-VIB)** [fixed 2026-05-16] |

## Known gaps / queued

| # | Item | Status |
|---|---|---|
| 1 | CC-H3 cross-asset | hook injected; needs MultiAssetDataset |
| 2 | V6 first SOTA-2026 retrain | GPU-day allocation pending; pre-flight probe via wm_deep_audit |
| 3 | Duplicate XD_DROPOUT_RATE in settings.py (lines 160 + 177) | COSMETIC — V6 doesn't use XD-split path |
| 4 | Discriminator stability under HEADLINE_MODE | needs first SOTA training to verify D-loss-ratio stays ~0.5 |
