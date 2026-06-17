# V8 Training — Run Instructions

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
models/wm/v8/
  base/           # V8.0 base world model
  adapter/        # V8.X FiLM adapter
  ensemble/       # V8.E multi-seed ensemble (seed models + master checkpoint)
  ncl/            # V8.D NCL diversity
```

---

## V8.0 — Base World Model (Neural ODE-RSSM)

**Architecture:** Neural ODE (dh/dt = f_theta(h, t, obs), RK4 solver, substeps=2) + RSSM categorical latent (24x24)
**Training:** 200 epochs x 300 steps, batch=32, seq_len=96, AdamW lr=1e-4 (slower for ODE stability)
**Key feature:** Continuous-time dynamics. Lower LR and longer warmup (10 epochs) required for ODE stability.

```bash
# Train from scratch (or resume from checkpoint automatically)
python src/wm/v8/v8_training/train_world_model.py
```

**Outputs:**
- `models/wm/v8/base/v8_wm_latest.pt` — latest checkpoint (resumes from here)
- `models/wm/v8/base/v8_wm_best_ema.pt` — best EMA model (by validation loss)
- `logs/v8/v8_training.log` — training log

**IMPORTANT — Checkpoint Loading:**
Always use `CheckpointManager.load_latest(revin=)` to load V8 checkpoints. Do NOT use raw `torch.load` as it will miss RevIN state. This was a known bug and has been fixed in all V8 scripts.

---

## V8.0 Validation

```bash
# Validate best EMA model (default)
python src/wm/v8/v8_training/validate_world.py

# Validate latest checkpoint
python src/wm/v8/v8_training/validate_world.py --latest

# Validate both best + latest, compare
python src/wm/v8/v8_training/validate_world.py --both

# Validate a specific checkpoint file
python src/wm/v8/v8_training/validate_world.py --model models/wm/v8/base/v8_wm_best_ema.pt

# Robust validation (overfitting detection)
python src/wm/v8/v8_training/validate_world.py --robust

# Robust validation for specific horizon
python src/wm/v8/v8_training/validate_world.py --robust --horizon 16
```

**Validation gates:** IC > 0.015, ShIC/IC > 0.3, Val/Train < 2.0, Recon MSE < 0.10, KL in [0.01, 15.0]
**V8-specific logged metric:** `dynamics_reg` (||f(h,t,obs)||^2) -- should be small and stable.

---

## V8.X — FiLM Adapter (Regime-Adaptive)

**Architecture:** FiLM adapter (~15K params) on frozen V8.0 base
**Training:** 30 epochs x 500 steps, batch=64, AdamW lr=1e-3
**Requires:** Trained V8.0 base model (`models/wm/v8/base/v8_wm_best_ema.pt`)
**Note:** Adapter operates on cat(h_seq, z_post) with feat_dim=832 (256+576).

```bash
# Train adapter on frozen V8.0 (resumes from adapter checkpoint if exists)
python src/wm/v8/v8_training/train_adapter.py
```

**How it works:**
1. Loads frozen V8.0 base + RevIN from `models/wm/v8/base/v8_wm_best_ema.pt` via CheckpointManager
2. Builds context vector from training data tail (rolling IC, regime, volatility)
3. Trains small FiLM adapter that modulates V8.0's return trunk (input: cat(h_seq, z_post), d=832)

**Outputs:**
- `models/wm/v8/adapter/v8_adapter_latest.pt` — adapter weights
- `models/wm/v8/adapter/v8_adapter_best.pt` — best adapter weights
- `logs/v8/v8_adapter_training.log` — training log

---

## V8.E — Multi-Seed Ensemble

**Architecture:** K=5 independent V8.0 models trained with different random seeds
**Training:** 5 back-to-back full training runs (200 epochs each), different basin per seed
**Key params:** Seeds=[42, 1337, 2024, 7777, 31415], top-3 by IC at inference

**Why multi-seed?** Independent seeds land in different loss landscape basins, giving low inter-model correlation (rho ~0.3-0.5). Ensemble IC boost: `IC_ensemble = IC_single * sqrt(K / (1 + (K-1)*rho))`, typically ~30-50% improvement.

```bash
# Train all 5 seeds back-to-back (resumes from last incomplete seed)
python src/wm/v8/v8_training/train_snapshot.py
```

**How it works:**
1. Loads data ONCE, shared across all seeds
2. For each seed: sets all random states, creates fresh model/optimizer/RevIN/EMA
3. Trains full 200 epochs with vanilla cosine LR (same as V8.0)
4. Saves final model (EMA weights + RevIN) per seed
5. Master checkpoint tracks which seeds completed (robust resume across interruptions)
6. VRAM cleanup between seeds

**Resume behavior:**
- If interrupted mid-seed: resumes that seed from its per-seed checkpoint
- If interrupted between seeds: skips completed seeds, starts next one
- Master checkpoint (`v8e_master.pt`) tracks all progress

**Outputs:**
- `models/wm/v8/ensemble/v8_seed_0.pt` through `v8_seed_4.pt` — final seed models (EMA + RevIN)
- `models/wm/v8/ensemble/v8e_master.pt` — master checkpoint with seed metrics
- `models/wm/v8/ensemble/v8e_seed_{N}_latest.pt` — per-seed training checkpoint (cleaned up after completion)
- `logs/v8/v8_seed_ensemble_training.log` — training log

**Inference (after training):**
```python
from snapshot_ensemble import SnapshotEnsemble
ensemble = SnapshotEnsemble().to(DEVICE)  # auto-loads top-3 seeds by IC
outputs = ensemble.forward_train(obs_seq, asset_id)
```

---

## V8.D — Multi-Head NCL Diversity

**Architecture:** V8.0 backbone + K=5 parallel return prediction heads with NCL diversity loss
**Training:** 200 epochs x 300 steps, batch=32, AdamW lr=1e-4, ncl_lambda=0.5

```bash
# Train from scratch (full model: backbone + 5 heads)
python src/wm/v8/v8_training/train_ncl.py

# Warm-start backbone from trained V8.0, train everything end-to-end
python src/wm/v8/v8_training/train_ncl.py --load-backbone models/wm/v8/base/v8_wm_best_ema.pt

# Freeze backbone, only train the 5 diversity heads
python src/wm/v8/v8_training/train_ncl.py --load-backbone models/wm/v8/base/v8_wm_best_ema.pt --freeze-backbone
```

**How it works:**
1. Replaces V8.0's single return head with K=5 parallel heads (each: trunk + per-horizon projections, head_dim=384)
2. NCL penalty forces heads to make diverse (negatively correlated) errors
3. Inference averages all 5 heads for improved IC via diversity
4. `--load-backbone` initializes shared backbone from V8.0 via CheckpointManager (recommended)
5. `--freeze-backbone` only trains the 5 heads (faster, less VRAM, good for fine-tuning)

**Outputs:**
- `models/wm/v8/ncl/v8d_wm_latest.pt` — latest checkpoint
- `models/wm/v8/ncl/v8d_wm_best_ema.pt` — best EMA model
- `logs/v8/v8_ncl_training.log` — training log

---

## Variant Validation

After training each variant, validate it:

```bash
# V8.E — Multi-seed ensemble
python src/wm/v8/v8_training/validate_snapshot.py

# V8.X — FiLM adapter (compares base vs adapted IC side-by-side)
python src/wm/v8/v8_training/validate_adapter.py

# V8.D — NCL diversity (per-head IC + diversity metrics)
python src/wm/v8/v8_training/validate_ncl.py
```

All validators produce JSON output with per-asset, per-horizon metrics. Validation gates are the same as V8.0 (IC > 0.015, ShIC/IC > 0.3, Val/Train < 2.0, Recon MSE < 0.10).

---

## Full Training Pipeline (Recommended Order)

```
Step  Variant   Command                                                   Depends On
----  --------  --------------------------------------------------------  ----------
1     V8.0      python src/wm/v8/v8_training/train_world_model.py               Data only
2     V8.0      python src/wm/v8/v8_training/validate_world.py --robust         Step 1
3     V8.X      python src/wm/v8/v8_training/train_adapter.py                   Step 1
4     V8.X      python src/wm/v8/v8_training/validate_adapter.py                Step 3
5     V8.E      python src/wm/v8/v8_training/train_snapshot.py                  Data only
6     V8.E      python src/wm/v8/v8_training/validate_snapshot.py               Step 5
7     V8.D      python src/wm/v8/v8_training/train_ncl.py --load-backbone models/wm/v8/base/v8_wm_best_ema.pt
8     V8.D      python src/wm/v8/v8_training/validate_ncl.py                    Step 7
```

Steps 3-4 (V8.X) and 5-6 (V8.E) are independent and can run in parallel.
Step 7 (V8.D) benefits from warm-starting from V8.0 but can also train from scratch.

## VRAM Notes (RTX 4060, 8GB)

- V8.0: ~4-5 GB (RK4 with substeps=2 runs 4 function evaluations per sub-step per bar; peak VRAM is higher than GRU versions during backward pass)
- V8.X: ~3-4 GB (frozen base, tiny adapter; adapter feat_dim=832)
- V8.E: ~4-5 GB per seed (same architecture as V8.0, seeds run sequentially)
- V8.D: ~5-6 GB (5 return heads, head_dim=384)
- All use automatic mixed precision (torch.amp)
- If VRAM is tight, reduce ODE_SUBSTEPS to 1 in settings.py (coarser integration, less memory)

## Reproducibility

All trainers accept `--seed` for deterministic initialization:
```bash
python src/wm/v8/v8_training/train_world_model.py --seed 42
```

V8.E (train_snapshot.py) manages seeds automatically via `ENSEMBLE_SEEDS` in settings.py.

---

## Anti-Memorization Variant (V8.1)

V8.1 adds XD anti-memorization defenses to V8:
- Posterior/decoder restricted to 13 base features (no XD temporal fingerprints)
- 70% dropout + 0.3 noise on XD features [13:17] during training
- `--features 13|18` flag for ablation testing

See `src/wm/v8/v8_1_training/RUN.md` for full run instructions.

```bash
# Quick start
python src/wm/v8/v8_1_training/train_world_model.py --features 18    # XD anti-memorization
python src/wm/v8/v8_1_training/train_world_model.py --features 13    # Control (no XD)
python src/wm/v8/v8_1_training/validate_world.py --features 18
```
