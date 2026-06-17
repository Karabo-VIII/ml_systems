---
name: V8 bf16 ODE speed fix (2026-04-16)
description: V8 Neural ODE was running at 2000-2400s/epoch due to fp32 solver + per-stage time encoding. bf16 autocast + precomputed time features cut step time from 1.05s to 0.65s (1.62x). Probe PASS (100 steps, 0 NaN, h_max bounded).
type: project
---

## Fix summary

**Problem**: V8 epoch time 2011-2381s (33-40 min), GPU util only 27%. Python-loop kernel-launch overhead dominated.

**Root causes**:
1. `torch.amp.autocast("cuda", enabled=False)` forced fp32 throughout the RK4 solver (96 × 4 = 384 MLP evals per forward, all fp32)
2. `_encode_time` allocated fresh sin/cos tensors on every dynamics call (~1500+ tensor ops per forward)

**Fix** (2 files, no param-shape change, checkpoint-compatible):

1. `src/wm/v8/v8_training/world_model.py` — switched fp32-disabled autocast to bf16 autocast with `cache_enabled=False`:
   ```python
   with torch.amp.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
       h_seq = self.solver(h0.to(torch.bfloat16), obs_for_ode.to(torch.bfloat16), t)
   h_seq = h_seq.float()  # back to fp32 for downstream heads
   ```
   - `cache_enabled=False` is CRITICAL — without it, the bf16 cast of `dynamics.net` weights persists in the autocast cache and clashes when `dynamics_regularization` runs under outer fp16 autocast, producing "mat1 Half vs mat2 BFloat16" errors.

2. `src/wm/v8/v8_training/components.py` — RK4Solver precomputes time features once per forward instead of 4× per stage:
   - `ODEDynamics.encode_time()` is now a static helper returning [4]
   - `ODEDynamics.forward()` accepts EITHER scalar t OR pre-encoded [4]/[B,4] (backward compatible)
   - `RK4Solver.forward()` builds a `[n_points, 4]` table upfront and indexes into it in the hot loop

3. `src/wm/v8/v8_training/world_model.py dynamics_regularization` — pre-encodes time and casts obs_t to h_t.dtype for consistent Linear input dtypes.

**Probe result** (`src/wm/v8/v8_training/probe_bf16.py`, 100 steps at B=32):
- 0 NaN/inf losses
- h_seq.abs.max bounded 375-396 (bf16 max is 3.4e38 — comfortable margin)
- Loss trajectory: 168 → 61 → 38 (normal descent)
- Step time: **0.65s median (was 1.05s) = 1.62× speedup**

**Expected epoch wall**: 1240s (20 min) — down from 2000-2400s.
**Expected 200-epoch total**: 82h — down from 133h. Saves ~51 GPU-hours.

## Why: Apply

This fix applies to any model where ODE/RK4/sequential MLP dominates forward time. V8.1/V8.2/V8.3 share the same `components.py` and `world_model.py` structure — propagate the same change if those variants are trained.

Sibling variants to propagate:
- `src/wm/v8/v8_1_training/` (FiLM adapter on base V8)
- `src/wm/v8/v8_2_training/`
- `src/wm/v8/v8_3_training/`

## How to apply: restart training

Current checkpoint (epoch 4) is still valid — no parameter shape change.
Just kill the running training (the one at epoch 5 when this was diagnosed),
edit nothing else, relaunch with the same `train_world_model.py` command.
The solver path is transparent to training-loop logic.

If problem manifests in live training: revert both files via git.
