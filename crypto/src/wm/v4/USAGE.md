# V4 World Models -- Usage Guide

## 2026-05-02 Frontier-ML upgrade status: ⏳ **NOT YET WIRED**

V4 has not received the V1.x upgrade flags but is the **highest-priority
second-wave target** because Mamba+RSSM is structurally similar to V1.x
(RSSM bottleneck + multi-horizon heads).

| Upgrade | Applicability | Wiring effort |
|---|---|---|
| `--sam` | ✅ applicable | port from V1.1 trainer (1-2 hrs) |
| `--fraug` | ✅ applicable (input-side) | trivial (1 hr) |
| `--pcgrad` | ✅ applicable | port get_loss `return_components` pattern (2-3 hrs) |
| `--mtp` | ✅ applicable | swap return_heads for MTPHead (1-2 hrs) |
| `--mdn` | ✅ applicable | head replacement + loss path (2-3 hrs) |
| `--adaptive-bins` | ✅ applicable | trivial bucketer drop-in (1 hr) |
| QKNorm/BCNorm (B004 R1) | ✅ **already in code** at lines 15, 196, 241, 294 of `components.py` | done |

⚠ **Open question (B004 R1)**: V4 already has QKNorm + RoPE + complex-
valued state per Mamba-3 paper. Yet `WM_FINDINGS` documents a persistent
ShIC decline 0.0236 → 0.0204 → 0.0201 mid-training. The QKNorm fix the
browser flagged is THERE; root cause must be different. Investigate
before retrain (likely: batch-size curriculum or LR schedule, per B004).

See [../UPGRADE_INVENTORY_2026_05_02.md](../UPGRADE_INVENTORY_2026_05_02.md)
for the cross-version matrix.

## Architecture: Mamba SSM + RSSM

State Space Model (selective scan, d_state=16) with RSSM 32x32 categorical latent. O(T) linear complexity with selective forgetting gates.

## Variants

| Variant | Dir | Features | Key Change |
|---------|-----|----------|------------|
| V4 base | `v4_training/` | 13/18/30/37 | Mamba SSM reference |
| V4.1 | `v4_1_training/` | 13/18/30/37 | Anti-memorization |
| V4.2 | `v4_2_training/` | 13/18/30/37 | d_state=32 (vs 16) |
| V4.3 | `v4_3_training/` | 13/18/30/37 | expand=1 (vs 2) |

## Training

### Base World Model (train this FIRST)

```powershell
# V4 base f13 (recommended starting point)
python src/wm/v4/v4_training/train_world_model.py --features 13

# V4 base f37 (full feature set)
python src/wm/v4/v4_training/train_world_model.py --features 37

# With RevIN (experimental only, disabled by default)
python src/wm/v4/v4_training/train_world_model.py --features 13 --revin
```

Feature options: `--features 13` (legacy), `18` (extended), `30` (base), `37` (full+XD, default).

### Variant Training (only after base passes gates)

```powershell
# FiLM Adapter (.X) -- ~5-25K params on frozen base
python src/wm/v4/v4_training/train_adapter.py --features 13

# Snapshot Ensemble (.E) -- cyclical cosine LR
python src/wm/v4/v4_training/train_snapshot.py --features 13

# NCL Diversity (.D) -- K=5 parallel heads
python src/wm/v4/v4_training/train_ncl.py --features 13
```

### Validation & Diagnostics

```powershell
# Full validation suite (IC, ShIC, reconstruction, dream coherence)
python src/wm/v4/v4_training/validate_world.py --features 13

# With RevIN
python src/wm/v4/v4_training/validate_world.py --features 13 --revin
```

## Checkpoints

Checkpoints are feature-tagged to prevent collision:
- `models/wm/v4/v4/base/v4_f13_wm_best_ema.pt` -- best EMA model for f13
- `models/wm/v4/v4/base/v4_f37_wm_best_ema.pt` -- best EMA model for f37
- `models/wm/v4/v4/base/v4_f13_wm_epoch_*.pt` -- epoch snapshots

Resume is automatic: the training script detects existing checkpoints and resumes. A collision guard validates that the checkpoint's `n_features` matches the current `--features` argument.

The `shic_decline_count` counter persists across resumes -- a memorizing model cannot reset its countdown by restarting.

## Key Settings

- d_model=384, RSSM 32x32 (flat_dim=1024)
- TwoHot: 255 bins, range [-1, 1], NO focal/smoothing
- Targets: raw returns (`target_return_*`, voladj deprecated)
- ACTIVE_HORIZONS: [1, 4, 16, 64]
- STEPS_PER_EPOCH: 2000
- Batch size: 32
- Sequence length: 96

## Key Notes

- JIT loop implementation (not CUDA kernel) -- slower than theoretical
- AMP dtype fix: SSM hidden state forced to float32 (prevents float16 corruption under mixed precision)
- **NOT YET TRAINED**
- **Priority: Highest** among V2-V9 (cleanest SSM, different inductive bias from V1/V3)
