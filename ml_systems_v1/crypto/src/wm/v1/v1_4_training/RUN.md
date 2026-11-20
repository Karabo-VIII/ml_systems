# V1.4 Training — Run Instructions (Transformer-RSSM Feature-Decoupled + FeatureAttentionBlock)

All commands run from project root: `c:\Users\karab\Documents\coding\v4_crypto_stystem`

## What's Different from V1.1

V1.4 is a feature-decoupled ablation variant of V1.1 with an iTransformer-style FeatureAttentionBlock added:

- Same Transformer-RSSM architecture as V1.1 (d_model=256, 8 heads, 3 layers, RSSM 24x24).
- `--features 18` (default): All 18 features. Posterior/decoder restricted to 13 base features. XD features [13:17] get 70% dropout + 0.3 noise during training.
- `--features 13`: Base features only (base_dim == input_dim, no XD paths).
- FeatureAttentionBlock: iTransformer-style cross-feature attention that runs over the feature dimension at each time step independently before temporal encoding. Each feature is projected to a 32-dim token; 4-head attention learns cross-feature interactions. This runs before the causal Transformer, allowing the model to learn feature interactions prior to temporal processing.

Uses its own checkpoint directory (`models/wm/v1_4/`) separate from V1.1.

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
models/wm/v1_4/
  base/           # V1.4.0 base world model
  adapter/        # V1.4.X FiLM adapter
  ensemble/       # V1.4.E multi-seed ensemble
  ncl/            # V1.4.D NCL diversity
```

---

## V1.4.0 — Base World Model

RevIN disabled by default (causes temporal memorization); opt-in via `--revin`.

```bash
# Train with 18 features + XD anti-memorization + FeatureAttentionBlock (default, no RevIN)
python src/wm/v1/v1_4_training/train_world_model.py

# Train with 13 base features only (ablation control)
python src/wm/v1/v1_4_training/train_world_model.py --features 13

# Specific seed
python src/wm/v1/v1_4_training/train_world_model.py --features 18 --seed 42

# Enable RevIN (opt-in, causes memorization -- A/B testing only)
python src/wm/v1/v1_4_training/train_world_model.py --revin
```

**Outputs:**
- `models/wm/v1_4/base/v1_4_f{13|18}_wm_latest.pt` — latest checkpoint (no RevIN)
- `models/wm/v1_4/base/v1_4_f{13|18}_wm_best_ema.pt` — best EMA model (no RevIN)
- `models/wm/v1_4/base/v1_4_f{13|18}_revin_wm_*.pt` — RevIN variants (if `--revin` used)
- `logs/v1_4/v1_4_f{13|18}_training.log` — training log

---

## V1.4.0 Validation

```bash
python src/wm/v1/v1_4_training/validate_world.py --features 18
python src/wm/v1/v1_4_training/validate_world.py --features 13
python src/wm/v1/v1_4_training/validate_world.py --features 18 --robust
python src/wm/v1/v1_4_training/validate_world.py --features 18 --latest
python src/wm/v1/v1_4_training/validate_world.py --features 18 --both

# Validate a RevIN model (must match training config)
python src/wm/v1/v1_4_training/validate_world.py --features 18 --revin
```

**Validation gates:** IC > 0.015, ShIC/IC > 0.3, Val/Train < 2.0, Recon MSE < 0.10, KL in [0.01, 15.0]

---

## V1.4.X — FiLM Adapter

**Requires:** Trained V1.4.0 base model

```bash
python src/wm/v1/v1_4_training/train_adapter.py
```

---

## V1.4.E — Multi-Seed Ensemble

```bash
python src/wm/v1/v1_4_training/train_snapshot.py
```

---

## V1.4.D — NCL Diversity

```bash
python src/wm/v1/v1_4_training/train_ncl.py
python src/wm/v1/v1_4_training/train_ncl.py --load-backbone models/wm/v1_4/base/v1_4_f18_wm_best_ema.pt
```

---

## Variant Validation

```bash
python src/wm/v1/v1_4_training/validate_snapshot.py     # V1.4.E
python src/wm/v1/v1_4_training/validate_adapter.py      # V1.4.X
python src/wm/v1/v1_4_training/validate_ncl.py          # V1.4.D
```

---

## Checkpoint Naming Convention

```
v1_4_{feat_tag}{revin_tag}_wm_{type}.pt
  feat_tag: f13 or f18
  revin_tag: "" (default, no RevIN) or "_revin" (opt-in)
  type: best_ema, latest
```

## Ablation Test Matrix

```bash
python src/wm/v1/v1_4_training/train_world_model.py --features 13   # Control (base features only)
python src/wm/v1/v1_4_training/train_world_model.py --features 18   # +XD anti-memorization + FeatureAttentionBlock
```

Primary metric: Shuffled IC (ShIC). Gate: ShIC > 0.015.
Compare V1.4 (cross-feature attention before temporal encoding) vs V1.1 (temporal encoding only) to measure the benefit of explicit feature interaction modeling.
