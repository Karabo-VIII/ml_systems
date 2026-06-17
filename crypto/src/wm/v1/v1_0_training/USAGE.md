# V1.0 — Transformer-RSSM Reference (Canonical Anchor)

> **Role in cohort**: the canonical Trader-tier anchor. Every V1.x variant
> (V1.1 / V1.4 / V1.6) compares against V1.0. Last verified: IC 0.067 / ShIC 0.032.

## Purpose

V1.0 is the **reference Transformer + RSSM** dollar-bar world model. It's the
simplest design in the V1.x family — no XD-split, no FeatureAttention, no Dream
rollouts. Just the canonical anti-memo stack: causal Transformer encoder + RSSM
categorical bottleneck + per-sample ATME + TwoHot return heads.

It's the **iron-clad baseline** — if a V1.x variant doesn't beat V1.0's
IC/ShIC, the variant's distinctive lever isn't paying off.

## Architecture

```
Obs (B, T, F) + asset_id
  └── Linear(F + asset_emb → d_model=256) → RMSNorm → SiLU → Dropout
       └── (causal shift: predict t from t-1)
            └── 3× CausalTransformerBlock(d_model=256, n_heads=8, d_ff=768, SwiGLU)
                 └── h_seq [B, T, 256]
                      ├── prior_head(h_seq) ─→ prior_logits (B, T, 576)
                      └── posterior_head(h_seq, obs) ─→ post_logits → z_post (B, T, 576)
                           └── feat = [h_seq, z_post] (B, T, 832)
                                ├── decoder → recon (B, T, base_dim)
                                ├── regime_head → 3-class
                                └── return_trunk → 4× return_heads (TwoHot 255 bins)
```

### Anti-memorization (the 5-layer defense)

1. **RSSM 24×24 categorical bottleneck** — only 9.2 bits/timestep ceiling
2. **Per-sample ATME** — 15% of samples get obs-only posterior (`TEMPORAL_CTX_DROP=0.15`)
3. **Block-masking 15%** of bars during training
4. **Causal shift** — input at t shows obs[t-1]; eliminates same-bar leakage
5. **Free-nats KL floor (max-formulation)** — prevents KL collapse + Kendall log_var explosion

### Design rationale

- **Why Transformer over LSTM/GRU**: parallel training (no recurrence in time), better gradient flow over 96+ bars, native compatibility with RoPE + FlashAttention
- **Why RSSM 24×24 not 32×32**: hardcap at 9.2 bits/timestep forces the encoder to throw away noise. Bigger RSSM = more memorization capacity (tested; ShIC drops at 32×32)
- **Why d_model=256**: matches 5M-param budget on RTX 4060 (8GB VRAM) with B=32, T=96
- **Why 3 layers**: empirically optimal for 96-bar context; 6 layers overfit per ShIC
- **Why SwiGLU over GeLU**: LLaMA convention; better gradient flow
- **Why TwoHot over softmax classification**: continuous distribution-aware loss; better calibration than 1-hot CE
- **Why 4 horizons [1, 4, 16, 64]**: dollar-bar timescale; missing h16/h64 caused ShIC decay per V11 audit
- **Why free-nats max-formulation**: clamp(kl-fn, min=0) allowed KL=0 → Kendall log_var drifted to -∞ → weight explosion. max(kl, fn) keeps KL contributing to weighting.

## Files

```
src/wm/v1/v1_0_training/
├── settings.py              # config + canonical invariants
├── components.py            # RMSNorm, RotaryEmb, CausalTransformerBlock, TwoHotSymlog, SwiGLU, MLPHead
├── world_model.py           # TransformerWorldModel (architecture + forward_train + get_loss)
├── train_world_model.py     # training loop (EMA, ShIC tracking, ckpt save/load)
├── validate_world.py        # post-training eval (IC, ShIC, regime acc, by-horizon)
└── adapter.py, snapshot_ensemble.py, ncl_model.py  # .X / .E / .D variants (V1.0 has NONE active)
```

## Usage

### Train

```bash
# Default: f29 (Pattern P, no dead features)
python src/wm/v1/v1_0_training/train_world_model.py --features 29

# Other feature counts
python src/wm/v1/v1_0_training/train_world_model.py --features 13      # legacy base
python src/wm/v1/v1_0_training/train_world_model.py --features 41      # full v50
python src/wm/v1/v1_0_training/train_world_model.py --features 121     # full v51 frontier

# Seeds for ensemble
python src/wm/v1/v1_0_training/train_world_model.py --features 29 --seed 0
python src/wm/v1/v1_0_training/train_world_model.py --features 29 --seed 1

# Headline mode (CC-H4 anti-mem ↑)
V1_HEADLINE_MODE=1 python src/wm/v1/v1_0_training/train_world_model.py --features 29
```

### Validate

```bash
python src/wm/v1/v1_0_training/validate_world.py
```

## Key settings

| Setting | Value | Notes |
|---|---|---|
| `WM_D_MODEL` | 256 | Transformer width |
| `WM_N_HEADS` | 8 | 32-dim per head |
| `WM_N_LAYERS` | 3 | Causal layers |
| `WM_D_FF` | 768 | FFN inner dim (3× d_model; SwiGLU sized for parity with 4× d_model GeLU) |
| `RSSM_LATENT_DIM` | 24 | Categorical groups |
| `RSSM_CLASSES` | 24 | Classes per group → 24² = 576 flat states |
| `WM_SEQ_LEN` | 96 | Bars per training sequence |
| `WM_BATCH_SIZE` | 32 | Anti-mem (small batch = implicit grad noise) |
| `WM_FREE_NATS` | 1.0 | KL floor |
| `TEMPORAL_CTX_DROP` | 0.15 | Per-sample ATME |
| `DIRECT_RETURN_WEIGHT` | 3.0 | Huber-dominance regularizer |
| `betas` | (0.9, 0.95) | AdamW (LLaMA convention) |

## Last known metrics (from `models/wm/v1/v1_0/`)

- **IC = 0.067 / ShIC = 0.032** at f29 — Trader tier (CLAUDE.md ladder)
- Per `WM_HEADLINE_UPGRADE_PLAN §2`, projected V1.0-Headline (H1+H2+H4+H5) =
  IC ≥ 0.085 / ShIC ≥ 0.042 at 3 GPU-d cost

## Known gaps / queued

| # | Item | Status |
|---|---|---|
| 1 | CC-H3 cross-asset head | BLOCKED on MultiAssetDataset (~1.5 weeks) |
| 2 | CC-H1 multi-resolution stack | NOT WIRED (queued) |
| 3 | CC-H6 regime-conditional heads | NOT WIRED (queued; V3+V4 have it) |
| 4 | Pattern P+Q at f29 | SHIPPED via settings |
| 5 | CC-H4 anti-mem ↑ | settings flag `V1_HEADLINE_MODE=1` (default OFF) |
