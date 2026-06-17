# V3.1 Training — Run Instructions (WaveNet-GRU-RSSM + XD Anti-Memorization)

All commands run from project root: `c:\Users\karab\Documents\coding\v4_crypto_stystem`

## What's Different from V3

V3.1 adds XD anti-memorization defenses to the V3 WaveNet-GRU-RSSM architecture:

- `--features 18` (default): All 18 features. Posterior/decoder restricted to 13 base features. XD features [13:17] get 70% dropout + 0.3 noise during training.
- `--features 13`: Base features only. Identical behavior to V3 (base_dim == input_dim, no XD paths).

This allows A/B testing: train with 13 features as control, 18 features to test anti-memorization.

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
models/wm/v3_1/
  base/           # V3.1.0 base world model
  adapter/        # V3.1.X FiLM adapter
  ensemble/       # V3.1.E multi-seed ensemble
  ncl/            # V3.1.D NCL diversity
```

---

## V3.1.0 — Base World Model

```bash
# Train with 18 features + XD anti-memorization (default)
python src/wm/v3/v3_1_training/train_world_model.py

# Train with 13 base features only (ablation control)
python src/wm/v3/v3_1_training/train_world_model.py --features 13

# Specific seed
python src/wm/v3/v3_1_training/train_world_model.py --features 18 --seed 42
```

**Outputs:**
- `models/wm/v3_1/base/v3_1_f{13|18}_wm_latest.pt` — latest checkpoint
- `models/wm/v3_1/base/v3_1_f{13|18}_wm_best_ema.pt` — best EMA model
- `logs/v3_1/v3_1_f{13|18}_training.log` — training log

---

## V3.1.0 Validation

```bash
python src/wm/v3/v3_1_training/validate_world.py --features 18
python src/wm/v3/v3_1_training/validate_world.py --features 13
python src/wm/v3/v3_1_training/validate_world.py --features 18 --robust
python src/wm/v3/v3_1_training/validate_world.py --features 18 --latest
python src/wm/v3/v3_1_training/validate_world.py --features 18 --both
```

**Validation gates:** IC > 0.015, ShIC/IC > 0.3, Val/Train < 2.0, Recon MSE < 0.10

---

## V3.1.X — FiLM Adapter

**Requires:** Trained V3.1.0 base model

```bash
python src/wm/v3/v3_1_training/train_adapter.py
```

---

## V3.1.E — Multi-Seed Ensemble

```bash
python src/wm/v3/v3_1_training/train_snapshot.py
```

---

## V3.1.D — NCL Diversity

```bash
python src/wm/v3/v3_1_training/train_ncl.py
python src/wm/v3/v3_1_training/train_ncl.py --load-backbone models/wm/v3_1/base/v3_1_f18_wm_best_ema.pt
```

---

## Variant Validation

```bash
python src/wm/v3/v3_1_training/validate_snapshot.py     # V3.1.E
python src/wm/v3/v3_1_training/validate_adapter.py      # V3.1.X
python src/wm/v3/v3_1_training/validate_ncl.py          # V3.1.D
```

---

## Checkpoint Naming Convention

```
v3_1_{feat_tag}_wm_{type}.pt
  feat_tag: f13 or f18
  type: best_ema, latest
```

## Ablation Test Matrix

```bash
python src/wm/v3/v3_1_training/train_world_model.py --features 13   # Control (base features only)
python src/wm/v3/v3_1_training/train_world_model.py --features 18   # +XD anti-memorization
```

Primary metric: Shuffled IC (ShIC). Gate: ShIC > 0.015.
