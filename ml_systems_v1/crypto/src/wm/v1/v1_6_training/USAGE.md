# V1.6 — Transformer-RSSM + Dream + Gumbel ("Best of V1")

> **Role in cohort**: anti-memo MAX + CC-H7 testbed. Last verified
> IC ≈ 0.062 / ShIC ≈ 0.033 (Trader tier; lowest raw IC in V1.x, highest ShIC).

## Purpose

V1.6 = V1.0 + ALL proven V1.x anti-mem techniques + dream-rollout R&D lab:

- V1.0 base (Transformer + RSSM)
- V1.2's KL annealing + base_dim posterior + XD augmentation
- V1.3's Gumbel τ annealing (runtime parameter)
- V3-V9's ATME temporal context dropout (0.15)
- ACTIVE_HORIZONS [1,4,16,64] + pairwise ranking loss
- **`dream_step`** — defined but not yet trained in loss (CC-H7 candidate)

V1.6's pitch: "lowest raw IC, but highest ShIC ratio". When ShIC > IC × 0.5,
the model has signal that survives shuffle = real signal, not memorization.
V1.6 is the V1.x family's most robust performer per ShIC.

## Architecture

V1.0 + many additions:

| Component | What it does |
|---|---|
| **Gumbel τ annealing** | Categorical posterior uses Gumbel-softmax; τ schedules 1.0 → 0.5 over training (soft → sharp) |
| **Dream step** | `dream_gru + dream_proj`: from (h_seq, z_post) predict next (h, z) without observations. Currently DEFINED in `world_model.py` but NOT used in loss. CC-H7 = activate the dream rollout as auxiliary loss. |
| **Per-sample ATME 0.15** | Per-CLAUDE.md canonical |
| **KL annealing** | Linear ramp KL weight 0 → 1 over `KL_ANNEAL_EPOCHS` |
| **Base-dim posterior** | Posterior reads only `base_dim` features (V1.1's idea, propagated here) |
| **XD augmentation** | Dropout + noise on XD channels (same as V1.1) |

### Design rationale

- **Why combine everything**: V1.6 is the "max anti-mem" reference. If even
  V1.6 over-memorizes, the problem is structural (features, not architecture).
- **Why dream_step is defined but unused**: CC-H7 says "activate the dream
  rollout in loss to force latent z_post to be predictively useful, not just
  encoding-useful". V1.6 is the natural testbed but the activation is queued
  (needs `dream_rollout_loss` from `_shared/headline_components`).
- **Why Gumbel τ annealing instead of fixed τ**: soft categorical at start
  allows exploration; sharp at end matches inference distribution. Mitigates
  early posterior collapse.
- **Why lowest raw IC**: aggressive anti-mem trades raw IC for ShIC ratio.
  The COHORT wants the highest-ratio model in ensemble, not the highest IC.

## Files

```
src/wm/v1/v1_6_training/
├── settings.py
├── components.py            # shared bricks
├── world_model.py           # TransformerWorldModel + dream_step + Gumbel τ
├── train_world_model.py     # 1002 lines — Gumbel annealing schedule + KL anneal
├── validate_world.py        # uses τ_inference = GUMBEL_TAU_END (sharp)
└── adapter.py / snapshot_ensemble.py / ncl_model.py
└── train_adapter.py / train_snapshot.py / train_ncl.py
└── validate_adapter.py / validate_snapshot.py / validate_ncl.py
```

V1.6 is by-design slim — no `train_diversity.py` (V1.1 owns that lab).

## Usage

```bash
# Train base
python src/wm/v1/v1_6_training/train_world_model.py --features 29

# Headline mode
V1_HEADLINE_MODE=1 python src/wm/v1/v1_6_training/train_world_model.py --features 29

# Variants
python src/wm/v1/v1_6_training/train_adapter.py    --features 29
python src/wm/v1/v1_6_training/train_snapshot.py   --features 29
python src/wm/v1/v1_6_training/train_ncl.py        --features 29

# Validate
python src/wm/v1/v1_6_training/validate_world.py
```

## Headline path (per WM_HEADLINE_UPGRADE_PLAN §5)

V1.6 is the **dream-rollout testbed**:
- **H1**: CC-H7 — activate dream_rollout_loss (auxiliary 1-2 step rollout) — projected ShIC +0.005-0.012
- H2: CC-H3 cross-asset head (~0.5 GPU-d)
- H3: CC-H4 anti-mem ↑ (already aggressive; tune Gumbel τ instead)
- H4: Mechanism ablation: train V1.6 with ONE of {KL, Gumbel, ATME, dream}
       disabled at a time to identify the load-bearing component

V1.6-Headline projection: IC ≥ 0.072 / ShIC ≥ 0.045 at 2 GPU-d. Even if V1.6
doesn't hit Headline (0.10), the **dream-rollout finding ports back to
V1.0/V1.1/V1.4** — V1.6 is the laboratory.

## Key settings (additions vs V1.0)

| Setting | Value | Notes |
|---|---|---|
| `GUMBEL_TAU_START` | 1.0 | Initial (soft) |
| `GUMBEL_TAU_END` | 0.5 | Final (sharp) |
| `GUMBEL_TAU_ANNEAL_EPOCHS` | 50 | Schedule length |
| `KL_ANNEAL_EPOCHS` | 20 | Linear ramp of KL weight |
| `TEMPORAL_CTX_DROP` | 0.15 | Per-sample ATME (canonical) |
| Dream | defined; NOT in loss | CC-H7 wire pending |

## Known gaps / queued

| # | Item | Status |
|---|---|---|
| 1 | CC-H7 dream-rollout in loss | NOT WIRED (V1.6's signature play) |
| 2 | Per-mechanism ablation runs | QUEUED (4 GPU-d, plan §5 H4) |
| 3 | Gumbel τ tuning beyond annealing | OPEN |
| 4 | CC-H3 / CC-H1 / CC-H6 | Same as V1.0 |
