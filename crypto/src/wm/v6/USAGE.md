# V6 World Models -- Usage Guide

## 2026-05-02 Frontier-ML upgrade status: ⏳ **VICReg PRIMARY (B005 R1)**

V6 has its own dedicated upgrade per B005 R1: **C-JEPA VICReg fix** to
address the documented EMA-target-encoder collapse failure mode (ShIC
declines 0.0236 → 0.0204 → 0.0201 mid-training).

| Upgrade | Applicability | Wiring effort | Priority |
|---|---|---|---|
| **VICReg** (B005 R1) | ✅ JEPA-specific; primary fix | medium (3-5 hrs) | 🔴 HIGH (V6's only retrain justification) |
| `--sam` | ✅ applicable | port from V1.1 (1-2 hrs) | 🟡 second |
| `--fraug` | ✅ applicable | trivial (1 hr) | 🟡 second |
| `--pcgrad` | ❌ n/a (single objective) | — | n/a |
| `--mtp` | ❌ n/a (no multi-horizon TwoHot heads) | — | n/a |
| `--mdn` | ❌ partial (V6's bucketer is single-horizon if any) | — | low |
| `--adaptive-bins` | ❌ partial | — | low |

**Module ready** at [`../../frontier_ml/v1_upgrades/vicreg.py`](../../frontier_ml/v1_upgrades/vicreg.py).
Smoke PASS: healthy / collapsed / high-corr embeddings discriminate correctly.

**Wiring path for VICReg**: V6's existing JEPA loss already has an
InfoNCE term. Add VICReg variance + covariance terms with weights
(λ_var=25, λ_cov=1, λ_inv=25 from C-JEPA paper). The variance term
specifically penalizes embedding collapse — directly targets the failure
mode.

⚠ **Reliability**: VICReg paper tested ImageNet only; transfer to
time-series JEPA is INFERRED. 5 GPU-h probe; decision rule: ShIC ≥ 0.025
(vs current 0.020 declining) → VICReg ships as V6 default.

See [../UPGRADE_INVENTORY_2026_05_02.md](../UPGRADE_INVENTORY_2026_05_02.md)
for the cross-version matrix.

## Architecture: JEPA + Adversarial (TimeDiscriminator)

JEPA (V2-style) with adversarial anti-memorization via TimeDiscriminator. 3-layer CausalGRU, continuous 192-dim latent, GAN-style training with 2 optimizers.

## Variants

| Variant | Dir | Features | Key Change |
|---------|-----|----------|------------|
| V6 base | `v6_training/` | 25 (fixed) | JEPA + TimeDiscriminator |
| V6.1 | `v6_1_training/` | 13/17/18/20/22/25 | Anti-memorization (ShIC) |
| V6.2 | `v6_2_training/` | 13/17/18/20/22/25 | No Discriminator |
| V6.3 | `v6_3_training/` | 13/17/18/20/22/25 | Adv Weight Schedule (0->0.15) |

## Training

```powershell
python src/wm/v6/v6_training/train_world_model.py
python src/wm/v6/v6_1_training/train_world_model.py --features 25
```

## Training Agent

```powershell
python src/agent/train_agent.py --world-model v6 --features 25
```

## Key Settings

- TwoHot: 255 bins, range [-1, 1], NO focal/smoothing
- Targets: raw returns (default, voladj deprecated)
- BASE_DIM: 20 (for 25-feature mode)

## Key Notes

- **No posterior_head**: Agent uncertainty defaults to 0.5 (JEPA continuous latent)
- **get_loss returns 4-tuple**: `(total, loss_dict, l_disc, outputs)` -- different from V1-V5
- **2 optimizers**: Main + discriminator (GAN instability risk)
- **NOT YET TRAINED**
