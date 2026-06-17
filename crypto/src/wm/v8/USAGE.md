# V8 World Models -- Usage Guide

## 2026-05-02 Frontier-ML upgrade status: ⏸ **DEFER (per B005)**

V8 (Neural ODE + RSSM) is strictly dominated by Mamba SSD (V4) per B005
§3 inventory. The 2024-2026 frontier moved continuous-time forecasting
to **Mamba+NODE hybrids** (e.g. MODE arXiv 2601.00920) — pure NODE at
4× compute factor (RK4) is no longer competitive.

**Action**: do NOT wire V1.x upgrade flags into V8. Stays as-is in the
directory; do NOT retrain. Per WM_FINDINGS, it's already DEFER tier;
B005 confirms.

If a NODE-class architecture is wanted, the modern path is **Mamba
backbone + Latent NODE residual** — which is a NEW version (V21+),
not a V8 retrofit.

See [../UPGRADE_INVENTORY_2026_05_02.md](../UPGRADE_INVENTORY_2026_05_02.md)
for the cross-version matrix.

## Architecture: Neural ODE + RSSM

Continuous-time dynamics via ODE + RK4 solver. Correct for irregular dollar bars where time between bars varies. RSSM 24x24 categorical latent.

## Variants

| Variant | Dir | Features | Key Change |
|---------|-----|----------|------------|
| V8 base | `v8_training/` | 25 (fixed) | ODE + RK4 solver |
| V8.1 | `v8_1_training/` | 13/17/18/20/22/25 | Anti-memorization |
| V8.2 | `v8_2_training/` | 13/17/18/20/22/25 | Euler Solver (4 substeps) |
| V8.3 | `v8_3_training/` | 13/17/18/20/22/25 | No Dynamics Regularization |

## Training

```powershell
python src/wm/v8/v8_training/train_world_model.py
python src/wm/v8/v8_1_training/train_world_model.py --features 25
```

## Training Agent

```powershell
python src/agent/train_agent.py --world-model v8 --features 25
```

## Key Settings

- d_model=256, RSSM 24x24 (flat_dim=576)
- TwoHot: 255 bins, range [-1, 1], NO focal/smoothing
- Targets: raw returns (default, voladj deprecated)
- BASE_DIM: 20 (for 25-feature mode)

## Key Notes

- 760 MLP evaluations per step (RK4 is expensive)
- h0 bottleneck: entire sequence compressed to initial condition
- **NOT YET TRAINED**
