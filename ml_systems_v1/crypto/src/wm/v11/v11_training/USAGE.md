# V11 — Microstructure Feature Extractor (WaveNet + Hurst-MoE + Disc)

> **Role in cohort**: V3 + V6 + V9 hybrid (post-2026-04-30 plan §12).
> Iron-clad ✓ at 4.02M params.

## Purpose

V11 combines V3's WaveNet TCN, V6's TimeShuffleDiscriminator, and V9's
Mixture-of-Experts (with the V9 regime-leak fixed — V11 uses Hurst gating,
not label gating). The result is a **single-purpose return-prediction
architecture** that drops V1.x's reconstruction + RSSM in favor of "every
parameter serves return prediction".

The bet: V1.x's reconstruction loss + RSSM bottleneck waste capacity on
generative side-tasks. V11 strips them, replaces RSSM with VIB (cheaper
bottleneck), and adds Hurst-gated expert routing to specialize per-regime.

## Architecture (SOTA-2026)

```
Obs (B, T=96, F) + asset_emb
  └── obs_encoder → Linear → d_model=256
       └── WaveNet-TCN (6 layers, dilations [1,2,4,8,16,32])
            └── RegimeGatedExperts (Hurst-gated: trending vs reverting TCN)
                 └── FeatureAttention (post-encoder cross-feature)
                      └── h_seq [B, T, 256]
                           ├── RegimeFiLM (h_seq-only gate, identity-at-init)
                           ├── VIB to_mu / to_logvar → z [B, T, z_dim] → feat
                           │    └── ATME on feat (per-sample 0.30)
                           ├── return_trunk + return_heads (TwoHot 255 bins)
                           ├── regime_head → 3-class
                           ├── CC-H5 quantile_heads (q05..q95)
                           ├── CC-H6 regime_cond_heads
                           └── TimeShuffleDiscriminator on h_seq (separate optimizer)
```

### Anti-memorization

1. **VIB** (cheap bottleneck; replaces RSSM)
2. **TimeShuffleDiscriminator** (V6's adversarial penalty on temporal dependence)
3. **ATME 0.30** on post-VIB feat
4. **Hurst-gated experts** (trending vs reverting paths; per-regime specialization)
5. **XD dropout 0.85** (SOTA-2026)
6. **HEADLINE_MODE** drops V9 MoE leak (1 expert instead of 3)
7. **RegimeFiLM** (SOTA-2026 encoder-level conditioning)

### Design rationale

- **Why drop RSSM**: V11's bet is that "every param serves return". RSSM's
  categorical bottleneck is great anti-memo but its decoder consumes capacity.
  VIB is the lighter equivalent.
- **Why MoE with 1 expert (HEADLINE_MODE)**: V9's 3-expert MoE used label-leaked
  regime gating, which leaked target into prediction. V11 fixes this by using
  Hurst feature (pre-computed, no leak). Default HEADLINE_MODE=1 expert
  effectively drops MoE; multi-expert is queued for predicted-regime gate.
- **Why feature_attn AFTER TCN, not before**: TCN captures temporal patterns;
  feature_attn then mixes across features on the temporal-aware representation.
  Reverse order (V1.4 pattern) is a separate variant.
- **Why no reconstruction**: per WM_HEADLINE_UPGRADE_PLAN §12 — V11 is the
  "no waste" architecture. Recon loss = wasted gradient on noise.

## Files

```
src/wm/v11/v11_training/
├── settings.py
├── world_model.py           # MicrostructureWorldModel (798 lines)
├── train_world_model.py     # full trainer w/ disc-aware gradient flow
└── components.py            # 0 lines (V11 defines its own inline)
```

## Usage

```bash
# Train (SOTA-2026 defaults — HEADLINE_MODE ON, CC-H5/H6, FiLM)
python src/wm/v11/v11_training/train_world_model.py --features 29

# Legacy mode
V11_HEADLINE_MODE=0 python src/wm/v11/v11_training/train_world_model.py --features 29

# Validate
python src/wm/v11/v11_training/validate_world.py
```

## Key settings

| Setting | Value | Notes |
|---|---|---|
| `WM_D_MODEL` | 256 | |
| Cohort invariants | canonical | (BIN_*, WM_BATCH_SIZE, ACTIVE_HORIZONS, etc.) |
| `XD_DROPOUT_RATE` | **0.85** | SOTA-2026 (was 0.7) |
| `VIB_KL_WEIGHT` | 0.05 | KL weight on VIB z |
| `DISC_WEIGHT` | (set in module) | adversarial loss weight |
| `HEADLINE_MODE` | **ON** by default | drops V9 MoE leak; activates V3+V6 headline knobs |
| `USE_QUANTILE_HEADS` | True | CC-H5 |
| `USE_REGIME_COND_HEADS` | True | CC-H6 |
| `REGIME_AWARENESS_MODE` | "film" | RegimeFiLM on h_seq pre-VIB |

## Known gaps / queued

| # | Item | Status |
|---|---|---|
| 1 | CC-H3 cross-asset | not yet wired; needs MultiAssetDataset |
| 2 | MoE expansion to predicted-regime gate (3 experts) | queued (high-risk; current Hurst-1-expert is the safe baseline) |
| 3 | Discriminator stability under HEADLINE_MODE | needs first SOTA training to verify |
