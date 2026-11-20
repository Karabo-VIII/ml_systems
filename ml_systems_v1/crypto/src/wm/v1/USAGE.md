# V1 World Models -- Usage Guide

## 2026-05-02 Frontier-ML upgrades — V1.1 only (probe-ready)

The V1.1 trainer carries **6 opt-in upgrade flags** wired against
modules at [`src/frontier_ml/v1_upgrades/`](../../frontier_ml/v1_upgrades/).
**V1.0 / V1.4 / V1.6 do NOT have these flags yet** — second-wave wiring
gated on V1.1 probe outcomes. See
[../UPGRADE_INVENTORY_2026_05_02.md](../UPGRADE_INVENTORY_2026_05_02.md)
for the cross-version matrix.

| Flag | Class | Source | Lift target |
|---|---|---|---|
| `--sam` | 🔴 FOUNDATIONAL | B003 R1 | +0.005-0.010 ShIC + +0.005-0.010 IC |
| `--pcgrad` | 🔴 FOUNDATIONAL | B003 4.6 | +0.005-0.015 IC across horizons |
| `--mtp` | 🔴 FOUNDATIONAL | B002 R1 | +0.005-0.015 IC |
| `--mdn` (`--mdn-mode {normal,skewed_t}`) | 🔴 FOUNDATIONAL | B003 R3 | +0.005-0.020 IC (tail capture) |
| `--adaptive-bins` | 🟡 STRUCTURAL | B001 R3 | +0.003-0.008 ShIC |
| `--fraug` | 🟡 STRUCTURAL | B003 R2 | +0.003-0.010 IC + +0.005-0.015 ShIC |

⚠ **Reliability caveat**: 0% VERIFIED on IC/ShIC deltas; all values are
INFERRED extrapolations from non-IC published metrics. Probes after
baseline f29 finishes decide. Decision rule per probe: **ShIC ≥ 0.038
(+0.005 vs current 0.033 record) → propagate to V1.0/V1.4/V1.6**.

### Composition order (each composes on prior)

```
python src/wm/v1/v1_1_training/train_world_model.py --features 29                                                          # baseline
python src/wm/v1/v1_1_training/train_world_model.py --features 29 --sam                                                   # +SAM
python src/wm/v1/v1_1_training/train_world_model.py --features 29 --sam --pcgrad                                          # +PCGrad
python src/wm/v1/v1_1_training/train_world_model.py --features 29 --sam --pcgrad --mtp                                    # +MTP
python src/wm/v1/v1_1_training/train_world_model.py --features 29 --sam --pcgrad --mtp --mdn --mdn-mode skewed_t          # +MDN
python src/wm/v1/v1_1_training/train_world_model.py --features 29 --sam --pcgrad --mtp --mdn --mdn-mode skewed_t --adaptive-bins --fraug   # full stack
```

Constraint notes:
- `--sam` disables AMP (eager fp32; ~2× wall-clock per step)
- `--pcgrad` disables AMP (per-task backward incompatible with scaler)
- `--mtp` and `--mdn` are mutually exclusive (different output-space contracts)
- `--adaptive-bins` keeps bin COUNT (255) but changes PLACEMENT (denser near zero)

## Architecture: Transformer-RSSM

Reference architecture: Transformer encoder (d_model=256, 8 heads, 3 layers) + RSSM latent space (24x24 categorical). Uses RoPE, FlashAttention, SwiGLU, RMSNorm.

## Active Variants (4 models)

| Variant | Dir | Features | Key Change |
|---------|-----|----------|------------|
| V1.0 base | `v1_0_training/` | 13 (fixed) | Reference architecture |
| V1.1 | `v1_1_training/` | 13/18/21/25/30/37 | Flexible features, XD anti-memorization |
| V1.4 | `v1_4_training/` | 13/18/21/25/30/37 | FeatureAttentionBlock (iTransformer cross-feature attention) |
| V1.6 | `v1_6_training/` | 13/18/21/25/30/37 | Best-of-V1 (KL anneal, Gumbel tau, ATME, dream consistency) |

**Archived** (in `src/wm/v1/archive/`): V1.2 (KL only), V1.3 (Gumbel only), V1.5 (xd_ma_distance, zero value), V1.7 (GBT baseline, never trained)

## Feature Counts Explained

- **13**: Original V1.0 base features (BASE_DIM=13)
- **18**: 13 base + 5 extended (ma_distance, whale, efficiency, return_4, return_16)
- **21**: 18 + 3 Tier 1 (return_kurtosis, bar_duration, funding_momentum)
- **25**: 21 + 4 Hawkes (intensity, buy/sell intensity, imbalance)
- **30**: 25 + 5 IC-boost Tier 2 (momentum_accel, vol_price_corr, vol_ratio, flow_persistence, oi_price_divergence)
- **37**: Full (30 base + 7 cross-asset XD features)

## Multi-Head Feature Ablation

Opt-in with `--ablation` (V1.1 only). Each training run trains the primary head plus
smaller-subset heads in parallel. Ablation heads share the encoder/RSSM but have separate
return MLPs. IC is tracked per-head to measure marginal feature contribution.

### Recommended Training (2 runs cover all subsets)

```powershell
python src/wm/v1/v1_1_training/train_world_model.py --features 37 --ablation  # f37 + ablation heads
python src/wm/v1/v1_1_training/train_world_model.py --features 18 --ablation  # f18 + f13
```

Ablation runs use `_abl` checkpoint namespace (separate from standalone training).

## Shared Modules (v1/ root)

| File | Purpose |
|------|---------|
| `cross_ensemble.py` | V1.E cross-model ensemble (loads V1.0, V1.1, V1.6) |
| `validate_ensemble.py` | Ensemble validation with per-model and combined metrics |

## Training

```powershell
# V1.0 base (13 features, no ablation -- fixed architecture)
python src/wm/v1/v1_0_training/train_world_model.py

# V1.1 with 13 features (baseline comparison)
python src/wm/v1/v1_1_training/train_world_model.py --features 13

# V1.1 with 37 features (default)
python src/wm/v1/v1_1_training/train_world_model.py

# V1.1 with ablation (trains f37 + ablation heads, slower)
python src/wm/v1/v1_1_training/train_world_model.py --ablation

# V1.1 with CRPS loss (A/B test)
python src/wm/v1/v1_1_training/train_world_model.py --features 13 --loss-type crps

# V1.4 FeatureAttention (iTransformer cross-feature attention)
python src/wm/v1/v1_4_training/train_world_model.py --features 13

# V1.6 best-of-V1 (ATME, KL anneal, Gumbel tau, dream consistency)
python src/wm/v1/v1_6_training/train_world_model.py --features 13
```

## Key Settings (shared across V1 variants)

- d_model=256, n_heads=8, n_layers=3, d_ff=768
- RSSM: 24x24 categorical (576-dim flat)
- TwoHot: 255 bins, range [-1.0, 1.0], NO focal/smoothing (plain compute_loss)
- Targets: raw returns (default), voladj deprecated (vol shortcut)
- ACTIVE_HORIZONS: [1, 4, 16, 64] (h16/h64 act as multi-scale regularizers, removal kills ShIC)
- Pairwise ranking loss: weight=0.1, 256 pairs, h=1 only
- Sequence length: 96 bars
- Batch size: 32
- Epochs: 200, steps/epoch: 2000
- Cosine LR schedule

## Validation Gates

| Gate | Threshold |
|------|-----------|
| Reconstruction MSE | < 0.12 |
| Information Coefficient | > 0.015 |
| KL Divergence | 0.01 - 15.0 |
| Shuffled IC / Contiguous IC (h1) | > 0.3 |
| Val/Train Loss Ratio | < 2.0 |

## V1 Training Results

**NOTE**: All existing V1 checkpoints are INCOMPATIBLE with current settings (bins [-1,1] + raw targets). Must retrain from scratch.

Current results (post-SOTA-fix):

| Variant | Stop | Best ShIC | Gates |
|---------|------|-----------|-------|
| V1.0 f13 | ShIC-stop | 0.0257 | ALL PASS |
| V1.1 f13 | ShIC-stop | 0.0261 | ALL PASS |

## V1.E Ensemble

Cross-model ensemble of active V1 models:

```powershell
# Validate ensemble
python src/wm/v1/validate_ensemble.py

# Custom ensemble composition
python src/wm/v1/validate_ensemble.py --models v1_0 v1_1_f13 v1_1 v1_6

# Train agent on ensemble
python src/agent/train_agent.py --ensemble --steps 2000000 --sav
```

Default models: `v1_0, v1_1_f13, v1_1, v1_4, v1_6`

## Agent Training on V1 Models

```powershell
# Single model
python src/agent/train_agent.py --world-model v1_0 --features 13
python src/agent/train_agent.py --world-model v1_1 --features 37

# V1.E ensemble (recommended for best signal diversity)
python src/agent/train_agent.py --ensemble --steps 2000000 --sav

# With stress augmentation
python src/agent/train_agent.py --ensemble --augment --steps 2000000 --sav

# Resume interrupted training
python src/agent/train_agent.py --ensemble --resume
```

## Checkpoints

- **Location**: `models/wm/v1/v1_{N}/base/`
- **Format**: `{prefix}_wm_best_ema.pt` containing `{"model_state_dict": ...}`
- **Naming**: `v1_1_f25_abl_wm_best_ema.pt` (ablation), `v1_1_f25_wm_best_ema.pt` (standalone)
- **Collision guard**: Resume validates `n_features` and `use_ablation` match checkpoint metadata

## Critical Notes

- **RevIN OFF by default**: RevIN causes temporal memorization (ShIC=-0.001 vs ShIC=0.028 without)
- **Raw return targets**: Default. Voladj creates vol shortcut (IC=0.10 voladj vs 0.017 raw)
- **TwoHot bins [-1.0, 1.0]**: Fixed from [-5,5] which caused memorization
- **No emoji in print()**: Windows cp1252 will crash
- **NUM_WORKERS=0**: Required on Windows
- **torch.compile DISABLED for V1.1**: NaN collapse at epochs 3-5 with f13
- **All checkpoints need retraining**: Bins [-1,1] + raw targets require fresh training
