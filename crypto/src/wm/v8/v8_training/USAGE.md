# V8 — Neural ODE Continuous-Time Dynamics (SOTA-2026)

> **Role in cohort**: continuous-time dynamics via Neural ODE. Tests whether
> ODE integration captures irregular dollar-bar timing better than
> discrete-step models.
>
> **Status**: V8 not yet retrained on SOTA-2026 defaults. Per
> `WM_VERSION_VERDICTS_2026_05_16`: **DEPRIORITIZE — kill-or-resize decision
> pending first SOTA training**.

## Purpose

V8 models market dynamics as a continuous-time ODE:
```
dh/dt = f_θ(h, t, obs_t)
```
Solved numerically with **rk2** (midpoint method, 2-stage; downsized from
rk4 4-stage). Per-bar h advances via fixed-step integration.

The theoretical bet: dollar bars trigger at irregular wall-clock intervals,
so a continuous-time model SHOULD capture irregular timing better than a
discrete-step model. But empirically the lift is small — and the compute
cost is real (3 dynamics-fn evals per bar vs V1.x's 3 transformer-layer
forwards over the same compute).

## Architecture (SOTA-2026 post-upgrade)

```
Obs (B, T=96, F) + asset_emb
  └── Linear(F + 32 → d_model=256) → RMSNorm → SiLU
       └── obs_emb [B, T, 256]
            └── h0 (initial state)
                 └── rk2_solver(dh/dt = f_θ(h, t, obs_t), substeps=1)
                      │  └── Per bar: 3 fn evals at t/(t+0.5dt)/(t+dt)
                      └── h_seq [B, T, 256] (ODE-integrated)
                           ├── RegimeFiLM (pre-RSSM, identity-at-init)
                           ├── RSSM prior_head(h_seq) → prior_logits
                           ├── RSSM posterior_head(h_seq, obs) → post_logits → z_post
                           │    └── (per-sample ATME 0.15 mask)
                           └── feat_heads = [h_seq.detach(), z_post]
                                ├── decoder → recon
                                ├── regime_head → 3-class
                                ├── return_trunk → return_heads (TwoHot)
                                ├── forecast_heads → predict obs[t+h]
                                ├── CC-H5 quantile_heads
                                └── CC-H6 regime_cond_heads
```

### Speed profile

| Configuration | Per-bar fn evals | Per-step (T=96 × B=32) | vs V1.x baseline |
|---|---|---|---|
| Pre-2026 (rk4 substeps=2) | 8 | 24,576 | ~256× |
| Pre-2026-05-16 (rk2 substeps=1) | 3 | 9,216 | ~96× |
| **Current SOTA-2026** (rk2 substeps=1 + HEADLINE_MODE) | 3 | 9,216 | ~96× |
| Queued (tsit5 + adjoint) | adaptive | varies | ~50× |
| V1.x reference (transformer×3) | n/a | ~3 fwds total | 1× |

The ~96× slowdown is the **fundamental cost** of V8. Even Tsit5 adaptive
solver wouldn't bring it under ~30× V1.x. V8 has to demonstrate ≥0.05 IC
to justify the GPU spend; if not, KILL per plan §9.

### Anti-memorization (SOTA-2026)

1. **RSSM 24×24** categorical bottleneck
2. **Per-sample ATME 0.15** (V1.6-class, was 0.40 batch-level)
3. **Free-nats 1.5** (was 1.0; CC-H4)
4. **XD dropout 0.85** (was 0.7; CC-H4)
5. **Block masking 15%**
6. **Forecast head** as encoder anchor (+88% train_IC validated on V4 same pattern)
7. **`h_seq.detach()` on heads** — same anti-memo philosophy as V6

### Design rationale

- **Why rk2 over rk4**: 2026-04-27 audit downsized; rk2 (midpoint) gives 2nd-
  order accuracy with 2 stages instead of rk4's 4. Halves compute with
  minimal accuracy loss for our regime.
- **Why substeps=1**: previously substeps=2 (4 stages × 2 = 8 evals/bar);
  substeps=1 cuts to 3 evals/bar (rk2's 2 + the discrete-step boundary call).
- **Why bf16 in solver**: 2026-05-10 fix — fp32 ODE solver was the per-bar
  bottleneck. bf16 with cache_enabled=False (CRITICAL: prevents cached weight
  conflicts under nested autocast).
- **Why `nan_to_num` on forecast h_seq**: bf16 ODE solver can produce NaN/inf
  h_seq values that cascade through forecast_heads. Defensive cast prevents
  100%-NaN training collapse.
- **Why HEADLINE_MODE default ON**: V8 not yet trained on SOTA-2026 defaults;
  the headline-mode settings (tsit5 adaptive, adjoint, learned-step-size) are
  the version's intended best-case. Tsit5 kernel + adjoint wiring NOT yet
  implemented; flags are scaffolds.

## Files

```
src/wm/v8/v8_training/
├── settings.py              # config (rk2, substeps=1, headline mode flags)
├── components.py            # ODEDynamics, RK2/RK4/EulerSolver, RMSNorm
├── world_model.py           # NeuralODEWorldModel (with SOTA-2026 wiring)
├── train_world_model.py     # full trainer
└── (variants v8_1/2/3: alt-experiment slots)
```

### Sub-versions

| Sub | Hypothesis |
|---|---|
| `v8_training` | base (rk2 + substeps=1) |
| `v8_1_training` | RK4 substeps=2 (legacy reference) |
| `v8_2_training` | Euler substeps=4 (~½ compute test) |
| `v8_3_training` | alt config |

## Usage

```bash
# Train (SOTA-2026 defaults)
python src/wm/v8/v8_training/train_world_model.py --features 29

# Legacy
V8_HEADLINE_MODE=0 python src/wm/v8/v8_training/train_world_model.py --features 29

# Validate
python src/wm/v8/v8_training/validate_world.py
```

## Kill-or-resize verdict

Per `WM_HEADLINE_UPGRADE_PLAN §9`:
> "V8 unlikely to clear Headline tier because the underlying continuous-time
> advantage on dollar bars is small. If H1+H2 baseline doesn't show IC > 0.050,
> KILL V8 and reallocate the compute."

**Recommended decision protocol**:
1. Run V8 first SOTA-2026 training (~3 GPU-d est at rk2 substeps=1)
2. Measure: contiguous IC at h=1 after 12 epochs
3. **IF IC < 0.050 → archive V8** to `backups/BKP_V8_ODE_ARCHIVE_<date>/`
4. **IF IC ≥ 0.050 → invest in tsit5 + adjoint kernel** to bring wall-clock under
   30× V1.x; then a full Headline retrain
5. **IF IC ≥ 0.080 → V8-Headline becomes a real contender** (unlikely)

## Known gaps / queued

| # | Item | Status |
|---|---|---|
| 1 | tsit5 adaptive solver kernel | settings flag exists; impl queued |
| 2 | Adjoint-method backprop | settings flag exists; impl queued |
| 3 | Learned step size | settings flag exists; impl queued |
| 4 | CC-H3 cross-asset | hook injected; needs MultiAssetDataset |
| 5 | First SOTA-2026 training | required for kill-or-resize decision |
| 6 | Duplicate XD_DROPOUT_RATE in settings (lines 166 + 183) | COSMETIC |
