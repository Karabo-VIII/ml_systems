# V14 — WaveNet Conditioner + DDPM Diffusion (Distributional Output)

> **Role in cohort**: ONLY cohort member with native distributional output.
> Predicts full P(return | condition) instead of a point estimate.
>
> **Status**: V14 first launch was BLOCKED until 2026-05-16 by a 3-tuple
> unpack crash at `train_world_model.py:163` (fixed in commit `8afb3e1`).
> Now launchable + SOTA-2026 wired.

## Purpose

V14 reframes return prediction as **conditional density estimation**.
Instead of predicting a single number (or even TwoHot's 255-bin
discretization), V14's DDPM denoiser learns the full conditional
distribution `P(return_h | obs[t-T:t], regime, ...)` and SAMPLES from it
at inference.

The bet: returns are heavy-tailed + skewed. A model that outputs a
distribution (and can be queried for quantiles) is strictly more useful for
risk-aware sizing than a point estimate.

Bonus: V14 has DUAL output paths — DDPM samples for the meta-learner +
fast TwoHot heads for V1-compatible IC measurement. Best of both worlds.

## Architecture (SOTA-2026)

```
Obs (B, T=96, F) + asset_emb
  └── obs_encoder → Linear → d_model=256
       └── WaveNet condition encoder (fp32 for stability)
            └── condition [B, T, 256]
                 │
                 ├── RegimeFiLM (h_seq path; identity-at-init)
                 │     └── h_seq → VIB → feat (TwoHot path)
                 │           ├── return_heads (TwoHot 255 bins)
                 │           ├── regime_head
                 │           ├── CC-H5 quantile_heads (NATIVE FIT for V14)
                 │           └── CC-H6 regime_cond_heads
                 │
                 └── DDPM denoiser (unmodulated condition)
                       ├── Trained: 100-step noise schedule
                       └── Inference: DDIM 10-step (HEADLINE_MODE) or 50-step
                            └── N_SAMPLES = 32 (or 8 in HEADLINE_MODE)
                                 └── output: P(return) — distributional
```

### Anti-memorization

1. **VIB** stochastic compression on TwoHot path
2. **DDPM noise schedule** (training adds noise → encoder can't memorize
   exact obs→return mapping)
3. **XD dropout 0.85** (SOTA-2026)
4. **RegimeFiLM** (TwoHot path only; DDPM denoiser sees unmodulated condition
   for stable diffusion training)
5. **Score head RMSNorm** (May-9 fix — was missing)

### Design rationale

- **Why diffusion**: heavy-tailed returns are exactly what diffusion is best
  at modeling. Standard Gaussian regression assumes thin tails (wrong); DDPM
  learns the empirical distribution shape.
- **Why dual paths (TwoHot + DDPM)**: TwoHot gives fast V1-compatible IC
  measurement during training (no sampling overhead). DDPM gives full
  distribution for inference. Best of both.
- **Why FiLM only on TwoHot path**: DDPM training is SENSITIVE — perturbing
  the condition signal mid-training can destabilize the denoiser. Keeping
  DDPM-conditioning fixed (unmodulated) preserves training stability.
- **Why CC-H5 native fit**: V14's whole architectural pitch is distributional
  output. CC-H5 quantile heads give explicit q05..q95 outputs without the
  DDPM sampling cost. The meta-learner can read either; quantile-head is fast.
- **Why HEADLINE_MODE inference steps 50→10**: DDIM allows fewer steps with
  minimal quality loss. 10 steps = 5x faster inference, suitable for production.
- **Why score-head RMSNorm**: per V14 fix log, was missing initially; led to
  score head magnitude drift during training. RMSNorm bounds output magnitude
  before the final 1-d projection.

## Files

```
src/wm/v14/v14_training/
├── settings.py
├── world_model.py           # DiffusionWorldModel (~530 lines)
├── train_world_model.py     # full trainer (3-tuple unpack FIXED 2026-05-16)
└── components.py            # 0 lines (V14 defines inline; DDPM denoiser, score head)
```

## Usage

```bash
# Train (SOTA-2026 defaults — HEADLINE_MODE ON; DDIM 10-step inference)
python src/wm/v14/v14_training/train_world_model.py --features 29

# Legacy mode (50-step inference)
V14_HEADLINE_MODE=0 python src/wm/v14/v14_training/train_world_model.py --features 29

# Validate
python src/wm/v14/v14_training/validate_world.py
```

## Key settings

| Setting | Value | Notes |
|---|---|---|
| `DIFFUSION_STEPS` | 100 | training noise schedule depth |
| `DIFFUSION_INFERENCE_STEPS` | 50 → **10** (HEADLINE_MODE) | DDIM steps |
| `DIFFUSION_N_SAMPLES` | 32 → **8** (HEADLINE_MODE) | per-bar samples |
| `XD_DROPOUT_RATE` | **0.85** | SOTA-2026 |
| `VIB_KL_WEIGHT` | 0.10 | VIB anti-mem on TwoHot path |
| `HEADLINE_MODE` | **ON** by default | DDIM inference + quantile head + N_SAMPLES=8 |
| `USE_QUANTILE_HEADS` | True | CC-H5 (V14's NATIVE FIT) |
| `USE_REGIME_COND_HEADS` | True | CC-H6 |
| `REGIME_AWARENESS_MODE` | "film" | TwoHot path only |

## V14 specific — TwoHot vs DDPM at inference

| Output | When to use | Cost |
|---|---|---|
| **TwoHot** | Fast: per-bar IC measurement; meta-router conditioning | 1 forward |
| **CC-H5 quantile** | Risk-aware sizing in meta-learner | 1 forward |
| **DDPM full sample** | Diagnostic: actual return distribution at a bar | 10-50 forwards (HEADLINE 10) |

## Known gaps / queued

| # | Item | Status |
|---|---|---|
| 1 | 3-tuple unpack crash at train:163 | ✓ FIXED 2026-05-16 (commit 8afb3e1) |
| 2 | Score head RMSNorm | ✓ wired 2026-05-09 |
| 3 | CC-H3 cross-asset | hook injected on TwoHot path; needs MultiAssetDataset |
| 4 | Classifier-free guidance (CFG) | `HEADLINE_CFG_SCALE = 1.5` flag exists; impl queued |
| 5 | V14 first SOTA training | GPU-d allocation pending |

## Speed reality

V14 training: WaveNet condition encoder is cheap; DDPM denoiser is N×
forward passes through a 3-layer net. At T=96 × 100 noise levels × B=32,
the inner loop is the bottleneck. Estimated 1.5-2 GPU-d for full SOTA-2026
training (per WM_COHORT_RETRAIN_SCHEDULE).

Inference: TwoHot path is V1.x-fast. DDPM at HEADLINE_MODE 10 DDIM steps =
~50ms/bar vs ~5ms for V1.1 — 10x slower but fast enough for production
when only called once-per-batch.
