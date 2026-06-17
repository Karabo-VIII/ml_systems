# Quick Start Guide -- Training World Models

## Prerequisites

1. Processed data in `data/processed/` as `*_v50_chimera.parquet` files (24 features + 10 targets per asset)
2. Run `python src/pipeline/inspect_dataset.py --strict` to verify data health before training

---

## File Structure Overview

```
v4_crypto_stystem/
    config/
        data_config.yaml              # Shared data config (10 assets, dollar bar sizes)
    data/
        processed/                    # 10 chimera parquet files (24 features each)
            BTCUSDT_v50_chimera.parquet
            ETHUSDT_v50_chimera.parquet
            SOLUSDT_v50_chimera.parquet
            BNBUSDT_v50_chimera.parquet
            XRPUSDT_v50_chimera.parquet
            DOGEUSDT_v50_chimera.parquet
            ADAUSDT_v50_chimera.parquet
            AVAXUSDT_v50_chimera.parquet
            LINKUSDT_v50_chimera.parquet
            LTCUSDT_v50_chimera.parquet
    src/
        pipeline/                     # Data pipeline (fetch, build, inspect)
        v1/                           # Transformer-RSSM (6 variants + ensemble)
            v1_0_training/            # Reference (13 features fixed)
            v1_1_training/            # Anti-memorization
            v1_2_training/            # KL anneal ablation
            v1_3_training/            # Gumbel tau ablation
            v1_4_training/            # FeatureAttention ablation
            v1_5_training/            # Cross-sectional MA distance (6 XD)
            v1_6_training/            # Best-of-V1 (ATME, dream loss)
            cross_ensemble.py         # V1.E ensemble
            validate_ensemble.py      # Ensemble validation
        v2/                           # JEPA Contrastive (4 variants)
        v3/                           # WaveNet-GRU (4 variants)
        v4/                           # Mamba SSM (4 variants)
        v5/                           # Hybrid Mamba-Attention (4 variants)
        v6/                           # JEPA + Adversarial (4 variants)
        v7/                           # ViT Patch-based (4 variants)
        v8/                           # Neural ODE (4 variants)
        v9/                           # Mixture of Experts (4 variants)
        agent/                        # PPO trading agent
        analysis/                     # Backtesting & strategy lab
    models/v{1-9}/                    # Auto-created checkpoints
    logs/v{1-9}/                      # Auto-created training logs
```

---

## Training Individual Models

### V1: Transformer-RSSM (reference architecture)

```bash
# V1.0 base (13 features, fastest, good for baseline comparison)
python src/wm/v1/v1_0_training/train_world_model.py

# V1.1 anti-memorization (22 features default)
python src/wm/v1/v1_1_training/train_world_model.py --features 22

# V1.6 best-of-V1 (ATME, dream consistency, KL anneal)
python src/wm/v1/v1_6_training/train_world_model.py --features 22
```

### V2-V9: Alternative Architectures

```bash
# V2: JEPA Contrastive (continuous 192-dim latent)
python backups/BKP_20260429_MODEL_HARMONIZATION/v2/v2_training/train_world_model.py

# V3: WaveNet-GRU (multi-scale temporal, HIGHEST PRIORITY untrained)
python src/wm/v3/v3_training/train_world_model.py

# V4: Mamba SSM (O(T) linear complexity)
python src/wm/v4/v4_training/train_world_model.py

# V5: Hybrid Mamba-Attention
python backups/BKP_20260429_MODEL_HARMONIZATION/v5/v5_training/train_world_model.py

# V6: JEPA + Adversarial TimeDiscriminator
python src/wm/v6/v6_training/train_world_model.py

# V7: ViT Patch-based
python backups/BKP_20260429_MODEL_HARMONIZATION/v7/v7_training/train_world_model.py

# V8: Neural ODE + RK4
python src/wm/v8/v8_training/train_world_model.py

# V9: Mixture of Experts (3 regime-specialized)
python src/wm/v9/v9_training/train_world_model.py
```

All base versions use their full default feature set (22 features for V2-V9). Variant versions (.1, .2, .3) support `--features 13|17|18|22`.

---

## Feature System

Pipeline produces 24 features per asset. Models select by name:

| Feature Count | Composition | Used By |
|--------------|-------------|---------|
| 13 | Original V1.0 base features | V1.0 (fixed), all variants (via --features 13) |
| 17 | Extended base (13 + whale, efficiency, return_4, return_16) | Via --features 17 |
| 18 | 13 base + 5 XD cross-asset | Legacy compatibility |
| 22 | 17 base + 5 XD (default for V1.1-V1.4, V1.6, V2-V9) | Default |
| 19 | 13 base + 6 XD | V1.5 legacy |
| 23 | 17 base + 6 XD (default for V1.5) | V1.5 default |

## Target System

- **Voladj targets** (preferred): `target_voladj_{1,4,16,64}` -- vol-adjusted symlog returns
- **Raw targets** (fallback): `target_return_{1,4,16,64}` -- raw returns
- **TwoHot encoding**: 255 bins over [-5, 5] with focal gamma 1.0
- Auto-detection: models prefer voladj targets when available

---

## Resuming Training

All models support full resumability. If training is interrupted, run the same command again:

```bash
python src/wm/v1/v1_0_training/train_world_model.py
```

The trainer will automatically detect and resume from the latest checkpoint.

---

## Monitoring Training

### Live Progress

Each epoch shows:
```
Ep  45 | Loss: 1.2340 | Rec: 0.0523 | KL: 2.15 | r1:0.345 | r4:0.378 | r16:0.401 | r64:0.423 | Mask: 0.18 | LR: 3.0e-04
```

### Validation (Every 5 Epochs)

```
  -- VAL | Loss: 1.1987 | Rec: 0.0498 | IC1:0.0234 | IC4:0.0267 | IC16:0.0289 | IC64:0.0312 | KL: 2.08 | GATE PASS | BEST
```

---

## Validation Gates

### V1/V3-V5/V7-V9 (RSSM Models)

| Gate | Threshold |
|------|-----------|
| Reconstruction MSE | < 0.10 |
| Information Coefficient | > 0.015 |
| KL Divergence | 0.01 - 15.0 |
| Shuffled IC / Contiguous IC | > 0.3 |
| Val/Train Loss Ratio | < 2.0 |

### V2/V6 (JEPA Models)

| Gate | Threshold |
|------|-----------|
| Contrastive Accuracy | > 0.5 |
| Information Coefficient | > 0.015 |
| Embedding Diversity (std) | > 0.1 |

---

## Common Issues & Solutions

### "No data files found"

```bash
python src/pipeline/make_dataset.py            # builds v51 chimera (primary)
# python src/pipeline/make_dataset_legacy.py   # legacy v50 chimera (V1-V14 only)
```

### CUDA out of memory

Edit `settings.py` in the specific version:
```python
WM_BATCH_SIZE = 32  # or 24
```

### Training diverges (loss -> NaN)

Trainers automatically skip bad batches. If persistent:
```python
WM_LR = 1e-4  # Lower learning rate in settings.py
```

### Gate never passes

1. Train longer (gates can pass after 50+ epochs)
2. Try a different architecture
3. Check data quality: `python src/pipeline/inspect_dataset.py --strict`

---

## Next Steps After Training

Once a model passes validation gates:

1. **Verify checkpoint**: `ls models/v1/v1_0/base/`
2. **Validate ensemble** (V1 models): `python src/wm/v1/validate_ensemble.py`
3. **Train agent**: `python src/agent/train_agent.py --ensemble --steps 2000000 --sav`
4. **Run backtest**: `python src/analysis/strategy_lab.py --sweep --wm-ensemble`

---

## Hardware

- GPU: RTX 4060 8GB (one model at a time)
- RAM: 32GB
- Training: ~7-10 min/epoch for V1 on RTX 4060

For detailed architecture docs, see each version's USAGE.md in `src/v{N}/USAGE.md`.
