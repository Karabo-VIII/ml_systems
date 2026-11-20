# V3 World Models -- Usage Guide

## 2026-05-02 Frontier-ML upgrade status: ⏳ **NOT YET WIRED**

V3 has not received the V1.x upgrade flags. The dilated-causal-conv +
GRU architecture means:

| Upgrade | Applicability | Wiring effort |
|---|---|---|
| `--sam` | ✅ applicable | port from V1.1 trainer (1-2 hrs) |
| `--fraug` | ✅ applicable (input-side) | trivial (1 hr) |
| `--mdn` | ✅ applicable | medium (head replacement; 2 hrs) |
| `--pcgrad` | ❓ partial | V3 currently single-head per horizon; needs per-horizon split |
| `--mtp` | ❓ partial | requires multi-head V3 variant |
| `--adaptive-bins` | ✅ applicable | trivial (drop-in bucketer swap; 1 hr) |

**Priority**: SECOND-WAVE wire-up after V1.1 probe outcomes confirm the
INFERRED estimates. Per B004: V3 architecture is mature; gains come from
training-stack changes, not architecture. Re-train at f29 with current
invariants + future SAM+FrAug+MDN flags.

See [../UPGRADE_INVENTORY_2026_05_02.md](../UPGRADE_INVENTORY_2026_05_02.md)
for the cross-version matrix.

## Architecture: WaveNet-GRU + RSSM

Temporal Convolutional Network (dilated causal convolutions [1,2,4,8]) feeding into CausalGRU, with RSSM 24x24 categorical latent. Multi-scale temporal processing with 31-bar receptive field.

## Variants

| Variant | Dir | Features | Key Change |
|---------|-----|----------|------------|
| V3 base | `v3_training/` | 25 (fixed) | WaveNet + GRU reference |
| V3.1 | `v3_1_training/` | 13/17/18/20/22/25 | Anti-memorization |
| V3.2 | `v3_2_training/` | 13/17/18/20/22/25 | Alt Dilations [1,3,9,27] |
| V3.3 | `v3_3_training/` | 13/17/18/20/22/25 | WaveNet-Only (no GRU) |

## Training

```powershell
# V3 base (22 features) -- HIGHEST PRIORITY untrained model
python src/wm/v3/v3_training/train_world_model.py

# V3.1 anti-memorization (default 22 features)
python src/wm/v3/v3_1_training/train_world_model.py --features 25
```

## Training Agent on V3 Models

```powershell
python src/agent/train_agent.py --world-model v3 --features 25
python src/agent/train_agent.py --world-model v3 --features 25 --dual-stream
```

## Key Settings

- TwoHot: 255 bins, range [-1, 1], NO focal/smoothing
- Targets: raw returns (default, voladj deprecated)
- BASE_DIM: 20 (for 25-feature mode)

## Key Strengths

- **Working dream_step**: Native GRU state evolution (no wrapper needed)
- **Multi-scale**: Dilated convolutions capture different temporal scales
- **Best practical architecture** per SOTA audit (working dream + multi-scale)

## Critical Notes

- **NOT YET TRAINED**: No validated checkpoints exist
- **Priority 1**: Best candidate for next training run
- **RevIN OFF by default**
