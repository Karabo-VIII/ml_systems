# World Models Summary -- 9 Independent Architectures

## Overview

Nine world model architectures (V1-V9), each with 3-4 variants, sharing the same 24-feature data pipeline. All predict multi-horizon returns (h=1,4,16,64) using TwoHot encoding (255 bins, [-5, 5]) with vol-adjusted targets.

---

## Architecture Comparison

| Feature | V1 Transformer | V2 JEPA | V3 WaveNet | V4 Mamba | V5 Hybrid | V6 JEPA+Adv | V7 ViT | V8 Neural ODE | V9 MoE |
|---------|---------------|---------|------------|----------|-----------|-------------|--------|--------------|---------|
| **Core** | Causal Transformer | CausalGRU + InfoNCE | Dilated TCN + GRU | Selective SSM | Mamba + Attention | CausalGRU + GAN | Patch ViT | ODE + RK4 | 3-Expert GRU |
| **Complexity** | O(T^2) | O(T) | O(T) | O(T) | O(T + (T/4)^2) | O(T) | O((T/4)^2) | O(T * substeps) | O(T * experts) |
| **Latent** | RSSM 24x24 | 192-dim continuous | RSSM 24x24 | RSSM 32x32 | RSSM 32x32 | 192-dim continuous | RSSM 24x24 | RSSM 24x24 | RSSM 24x24/expert |
| **d_model** | 256 | 192 | varies | 384 | 384 | 192 | 320 | 256 | 320 |
| **Default Features** | 13 (V1.0) / 22 | 22 | 22 | 22 | 22 | 22 | 22 | 22 | 22 |
| **dream_step** | GRU wrapper | GRU wrapper | Native GRU | GRU wrapper | GRU wrapper | GRU wrapper | GRU wrapper | GRU wrapper | Native GRU |

---

## Shared Settings (All V1-V9)

### Data Pipeline
- **24 features**: 18 base + 6 cross-asset (XD)
- **10 assets**: BTC, ETH, SOL, BNB, XRP, DOGE, ADA, AVAX, LINK, LTC (all USDT pairs)
- **Dollar bars**: Variable bar sizes per asset (~5 min frequency)
- **Sequence length**: 96 bars (~24h of BTC data)

### Prediction
- **4 horizons**: h=1, h=4, h=16, h=64 steps ahead
- **TwoHot Symlog**: 255 bins over [-5, 5], focal gamma 1.0
- **Targets**: voladj preferred (auto-detected), raw fallback
- **Regime head**: 3-class (bear/neutral/bull) from SMA-200 labels

### Training
- **AMP**: Automatic mixed precision for 8GB VRAM
- **EMA**: 0.995 decay for best model tracking
- **Gradient clipping**: 1.0
- **Full resumability**: model + optimizer + scheduler + scaler states
- **Validation gates**: Rec MSE < 0.10, IC > 0.015, KL 0.01-15.0, ShIC ratio > 0.3

### Asset Conditioning
- Learned 32-dim asset embeddings (10 assets)
- Shared temporal dynamics with asset-specific scaling

---

## Feature System

Models select features by name from `settings.FEATURE_LIST`:

| Count | Composition | Available In |
|-------|-------------|-------------|
| 13 | Original base features | All models (V1.0 fixed, others via --features 13) |
| 17 | Extended base (+ whale, efficiency, return_4, return_16) | All variants via --features 17 |
| 18 | 13 base + 5 XD | Legacy compatibility |
| 20 | 17 base + 3 Tier 1 (return_kurtosis, bar_duration, funding_momentum) | All variants via --features 20 |
| 22 | 17 base + 5 XD | All variants via --features 22 |
| 25 | 20 base + 5 XD (full feature set) | Default for V1.1+, V2-V9 |

BASE_DIM separates base from XD features for anti-memorization techniques (13 for --features 13/18, 20 for --features 20/25, 17 for --features 17/22).

---

## V1 Variants (6 + ensemble)

| Variant | Key Change | Status |
|---------|------------|--------|
| V1.0 | Reference architecture (13 features) | Needs retrain (V51) |
| V1.1 | Anti-memorization (ShIC tracking) | Needs retrain |
| V1.2 | KL anneal ablation | Needs retrain |
| V1.3 | Gumbel tau ablation | Needs retrain |
| V1.4 | FeatureAttention ablation | Needs retrain |
| V1.5 | Cross-sectional MA distance (6 XD) | Needs retrain |
| V1.6 | Best-of-V1 (ATME, dream loss, KL anneal) | Never trained |
| V1.E | Cross-model ensemble | Needs retrained models |

## V2-V9 Variants (4 each)

Each version has: base (fixed 22 features) + .1 (anti-memorization) + .2 (ablation A) + .3 (ablation B).

**None of V2-V9 have been trained yet.** All checkpoints need V51 settings (bins [-5,5], voladj targets).

---

## Validation Gates

| Metric | V1/V3-V5/V7-V9 (RSSM) | V2/V6 (JEPA) |
|--------|----------------------|-------------|
| Reconstruction MSE | < 0.10 | N/A |
| Information Coefficient | > 0.015 | > 0.015 |
| KL Divergence | 0.01 - 15.0 | N/A |
| Shuffled IC / Contiguous IC | > 0.3 | > 0.3 |
| Val/Train Loss Ratio | < 2.0 | < 2.0 |
| Contrastive Accuracy | N/A | > 0.5 |
| Embedding Diversity | N/A | std > 0.1 |

---

## Training Commands

```bash
# V1 (recommended starting point)
python src/wm/v1/v1_0_training/train_world_model.py                    # V1.0 base (13 features)
python src/wm/v1/v1_6_training/train_world_model.py --features 22      # V1.6 best-of-V1

# V2-V9 base versions
python src/v{N}/v{N}_training/train_world_model.py

# V2-V9 variants (with feature selection)
python src/v{N}/v{N}_1_training/train_world_model.py --features 22
```

---

## Checkpoint Files

Each version saves to: `models/v{N}/v{N}_{variant}/base/`

| File | Purpose |
|------|---------|
| `*_wm_latest.pt` | Full checkpoint (resume training) |
| `*_wm_weights.pt` | Weights only (fast inference) |
| `*_wm_best_ema.pt` | Best EMA model (agent training) |
| `*_wm_epoch_*.pt` | Top-3 epoch checkpoints |

---

## Next Steps

1. **Retrain V1.0 f13**: Baseline with V51 settings (fastest)
2. **Train V1.6 f22**: Best-of-V1 consolidation
3. **Train V3 base**: Highest priority untrained architecture (working dream_step + multi-scale)
4. **Build ensemble**: V1.E cross-model ensemble after retraining
5. **Train agent**: `python src/agent/train_agent.py --ensemble --steps 2000000 --sav`

For per-version details, see `src/v{N}/USAGE.md`.
