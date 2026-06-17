# V6 Training — Run Instructions

All commands run from project root: `c:\Users\karab\Documents\coding\v4_crypto_stystem`

## Prerequisites

Data must exist: `data/processed/{SYMBOL}_v51_chimera.parquet` for all 10 assets.
If not, run the pipeline first:
```
python src/pipeline/fetch_all.py
python src/pipeline/make_dataset_legacy.py
```

---

## Checkpoint Directory Structure

```
models/wm/v6/
  base/           # V6.0 base world model
  adapter/        # V6.X FiLM adapter
  ensemble/       # V6.E multi-seed ensemble (seed models + master checkpoint)
  ncl/            # V6.D NCL diversity
```

---

## V6.0 — Base World Model (CausalJEPA)

**Architecture:** CausalGRU encoder (d=256, 3 layers) + JEPA contrastive (d_latent=192) + VICReg + TimeDiscriminator
**Training:** 180 epochs x 300 steps, batch=40, seq_len=96, AdamW lr=2.5e-4
**Key difference from RSSM versions:** Dual optimizer (main + discriminator). get_loss returns 4-tuple.

```bash
# Train from scratch (or resume from checkpoint automatically)
python src/wm/v6/v6_training/train_world_model.py
```

**Outputs:**
- `models/wm/v6/base/v6_wm_latest.pt` — latest checkpoint (resumes from here)
- `models/wm/v6/base/v6_wm_best_ema.pt` — best EMA model (by validation loss)
- `logs/v6/v6_training.log` — training log

---

## V6.0 Validation

```bash
# Validate best EMA model (default)
python src/wm/v6/v6_training/validate_world.py

# Validate latest checkpoint
python src/wm/v6/v6_training/validate_world.py --latest

# Validate both best + latest, compare
python src/wm/v6/v6_training/validate_world.py --both

# Validate a specific checkpoint file
python src/wm/v6/v6_training/validate_world.py --model models/wm/v6/base/v6_wm_best_ema.pt

# Robust validation (overfitting detection)
python src/wm/v6/v6_training/validate_world.py --robust

# Robust validation for specific horizon
python src/wm/v6/v6_training/validate_world.py --robust --horizon 16
```

**Validation gates:** Contrastive acc > 0.3, IC > 0.015, ShIC/IC > 0.3, Val/Train < 2.0, Embed std > 0.05

---

## V6.X — FiLM Adapter (Regime-Adaptive)

**Architecture:** FiLM adapter (~15K params) on frozen V6.0 base
**Training:** 30 epochs x 500 steps, batch=64, AdamW lr=1e-3
**Requires:** Trained V6.0 base model (`models/wm/v6/base/v6_wm_best_ema.pt`)
**Note:** Adapter operates on ctx_latent (d=192), not cat(vit_out, z_post) like V7/V8.

```bash
# Train adapter on frozen V6.0 (resumes from adapter checkpoint if exists)
python src/wm/v6/v6_training/train_adapter.py
```

**How it works:**
1. Loads frozen V6.0 base + RevIN from `models/wm/v6/base/v6_wm_best_ema.pt`
2. Builds context vector from training data tail (rolling IC, regime, volatility)
3. Trains small FiLM adapter that modulates V6.0's return trunk (input: ctx_latent d=192)

**Outputs:**
- `models/wm/v6/adapter/v6_adapter_latest.pt` — adapter weights
- `models/wm/v6/adapter/v6_adapter_best.pt` — best adapter weights
- `logs/v6/v6_adapter_training.log` — training log

---

## V6.E — Multi-Seed Ensemble

**Architecture:** K=5 independent V6.0 models trained with different random seeds
**Training:** 5 back-to-back full training runs (180 epochs each), different basin per seed
**Key params:** Seeds=[42, 1337, 2024, 7777, 31415], top-3 by IC at inference

**Why multi-seed?** Independent seeds land in different loss landscape basins, giving low inter-model correlation (rho ~0.3-0.5). Ensemble IC boost: `IC_ensemble = IC_single * sqrt(K / (1 + (K-1)*rho))`, typically ~30-50% improvement.

```bash
# Train all 5 seeds back-to-back (resumes from last incomplete seed)
python src/wm/v6/v6_training/train_snapshot.py
```

**How it works:**
1. Loads data ONCE, shared across all seeds
2. For each seed: sets all random states, creates fresh model/optimizer/discriminator_optimizer/RevIN/EMA
3. Trains full 180 epochs with vanilla cosine LR (same as V6.0)
4. Saves final model (EMA weights + RevIN) per seed
5. Master checkpoint tracks which seeds completed (robust resume across interruptions)
6. VRAM cleanup between seeds

**Resume behavior:**
- If interrupted mid-seed: resumes that seed from its per-seed checkpoint
- If interrupted between seeds: skips completed seeds, starts next one
- Master checkpoint (`v6e_master.pt`) tracks all progress

**Outputs:**
- `models/wm/v6/ensemble/v6_seed_0.pt` through `v6_seed_4.pt` — final seed models (EMA + RevIN)
- `models/wm/v6/ensemble/v6e_master.pt` — master checkpoint with seed metrics
- `models/wm/v6/ensemble/v6e_seed_{N}_latest.pt` — per-seed training checkpoint (cleaned up after completion)
- `logs/v6/v6_seed_ensemble_training.log` — training log

**Inference (after training):**
```python
from snapshot_ensemble import SnapshotEnsemble
ensemble = SnapshotEnsemble().to(DEVICE)  # auto-loads top-3 seeds by IC
outputs = ensemble.forward_train(obs_seq, asset_id)
```

---

## V6.D — Multi-Head NCL Diversity

**Architecture:** V6.0 backbone + K=5 parallel return prediction heads with NCL diversity loss
**Training:** 200 epochs x 300 steps, batch=40, AdamW lr=3e-4, ncl_lambda=0.5

```bash
# Train from scratch (full model: backbone + 5 heads)
python src/wm/v6/v6_training/train_ncl.py

# Warm-start backbone from trained V6.0, train everything end-to-end
python src/wm/v6/v6_training/train_ncl.py --load-backbone models/wm/v6/base/v6_wm_best_ema.pt

# Freeze backbone, only train the 5 diversity heads
python src/wm/v6/v6_training/train_ncl.py --load-backbone models/wm/v6/base/v6_wm_best_ema.pt --freeze-backbone
```

**How it works:**
1. Replaces V6.0's single return head with K=5 parallel heads (each: trunk + per-horizon projections)
2. NCL penalty forces heads to make diverse (negatively correlated) errors
3. Inference averages all 5 heads for improved IC via diversity
4. `--load-backbone` initializes shared backbone from V6.0 (recommended)
5. `--freeze-backbone` only trains the 5 heads (faster, less VRAM, good for fine-tuning)

**Outputs:**
- `models/wm/v6/ncl/v6d_wm_latest.pt` — latest checkpoint
- `models/wm/v6/ncl/v6d_wm_best_ema.pt` — best EMA model
- `logs/v6/v6_ncl_training.log` — training log

---

## Variant Validation

After training each variant, validate it:

```bash
# V6.E — Multi-seed ensemble
python src/wm/v6/v6_training/validate_snapshot.py

# V6.X — FiLM adapter (compares base vs adapted IC side-by-side)
python src/wm/v6/v6_training/validate_adapter.py

# V6.D — NCL diversity (per-head IC + diversity metrics)
python src/wm/v6/v6_training/validate_ncl.py
```

All validators produce JSON output with per-asset, per-horizon metrics. Validation gates are the same as V6.0.

---

## Full Training Pipeline (Recommended Order)

```
Step  Variant   Command                                                   Depends On
----  --------  --------------------------------------------------------  ----------
1     V6.0      python src/wm/v6/v6_training/train_world_model.py               Data only
2     V6.0      python src/wm/v6/v6_training/validate_world.py --robust         Step 1
3     V6.X      python src/wm/v6/v6_training/train_adapter.py                   Step 1
4     V6.X      python src/wm/v6/v6_training/validate_adapter.py                Step 3
5     V6.E      python src/wm/v6/v6_training/train_snapshot.py                  Data only
6     V6.E      python src/wm/v6/v6_training/validate_snapshot.py               Step 5
7     V6.D      python src/wm/v6/v6_training/train_ncl.py --load-backbone models/wm/v6/base/v6_wm_best_ema.pt
8     V6.D      python src/wm/v6/v6_training/validate_ncl.py                    Step 7
```

Steps 3-4 (V6.X) and 5-6 (V6.E) are independent and can run in parallel.
Step 7 (V6.D) benefits from warm-starting from V6.0 but can also train from scratch.

## VRAM Notes (RTX 4060, 8GB)

- V6.0: ~4-5 GB (dual optimizer adds ~5% overhead vs single-optimizer versions)
- V6.X: ~3-4 GB (frozen base, tiny adapter)
- V6.E: ~4-5 GB per seed (same architecture as V6.0, seeds run sequentially)
- V6.D: ~5-6 GB (5 return heads add ~2M params)
- All use automatic mixed precision (torch.amp)

## Reproducibility

All trainers accept `--seed` for deterministic initialization:
```bash
python src/wm/v6/v6_training/train_world_model.py --seed 42
```

V6.E (train_snapshot.py) manages seeds automatically via `ENSEMBLE_SEEDS` in settings.py.

---

## Anti-Memorization Variant (V6.1)

V6.1 adds XD anti-memorization defenses to V6:
- Decoder restricted to 13 base features (no XD temporal fingerprints)
- 70% dropout + 0.3 noise on XD features [13:17] during training
- `--features 13|18` flag for ablation testing

See `src/wm/v6/v6_1_training/RUN.md` for full run instructions.

```bash
# Quick start
python src/wm/v6/v6_1_training/train_world_model.py --features 18    # XD anti-memorization
python src/wm/v6/v6_1_training/train_world_model.py --features 13    # Control (no XD)
python src/wm/v6/v6_1_training/validate_world.py --features 18
```
