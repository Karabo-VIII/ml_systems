# V6 World Model: CausalJEPA

## Architecture

**CausalGRU + JEPA Contrastive Learning + VICReg + Adversarial Time Discriminator**

```
Input (18 features) --> Linear(18+32 -> 256) + RMSNorm + SiLU
    --> CausalGRUEncoder (3 layers, d=256) [online / context branch]
    --> Linear(256 -> 192) + RMSNorm  [context latent projector]
    --> PredictorNetwork (2 layers, d=192)
    --> Heads: Returns (TwoHot x4) | Regime (3-class focal) | Aux Reconstruction

Target branch (EMA-updated, no gradients):
    --> CausalGRUEncoder (3 layers, d=256) [target encoder]
    --> Linear(256 -> 192) + RMSNorm  [target latent projector]

Adversarial:
    --> TimeDiscriminator (d=128, 3 layers) distinguishes real vs time-shuffled latents
```

| Component              | Config                                          |
|------------------------|-------------------------------------------------|
| Hidden dim (d_model)   | 256                                             |
| Latent dim (d_latent)  | 192                                             |
| GRU layers             | 3 (unidirectional CausalGRU -- not BiGRU)       |
| Predictor layers       | 2                                               |
| Asset embedding        | 32-dim learned per asset                        |
| Return encoding        | TwoHot Symlog, 255 bins, [-1, 1]                |
| JEPA EMA decay         | 0.996 (target encoder momentum)                 |
| Full-model EMA decay   | 0.995 (for stable validation/ShIC)              |
| Discriminator hidden   | 128-dim, 3 layers                               |
| JEPA temperature       | 0.1 (InfoNCE)                                   |
| VICReg weights         | sim=25.0, var=25.0, cov=1.0                     |
| Adversarial weight     | 0.10 (encoder vs discriminator)                 |
| Grad penalty           | 10.0 (WGAN-GP for discriminator stability)      |
| Dropout                | 0.22                                            |

## Key Design Decisions

- **CausalGRU (not BiGRU)**: The most critical V6 fix. BiGRU (as in V2) peeks at future timesteps, learning spurious temporal correlations absent at inference. CausalGRU representations at time t depend only on t' <= t.
- **JEPA (not RSSM reconstruction)**: Self-supervised objective is predicting future latent embeddings (InfoNCE) rather than reconstructing input pixels. More sample-efficient for high-dimensional feature inputs.
- **Time Discriminator**: Adversarial penalty forces the encoder to produce latents indistinguishable from time-shuffled versions, removing any remaining temporal memorization.
- **Dual optimizer**: The discriminator has a separate optimizer (DISC_LR_MULT=1.0 of base LR). The main optimizer drives the encoder, predictor, and return heads.
- **Target encoder EMA (JEPA_EMA_DECAY=0.996)**: Must update every step (not every epoch) after optimizer.step() to maintain momentum encoder stability.
- **No KL annealing, no Gumbel tau**: V6 uses a continuous latent space (d_latent=192) instead of RSSM categorical (24x24). No Gumbel-Softmax needed.
- **Kendall weight corridors**: Contrastive loss clamped from below (at most 1.0x weight) so it cannot steal gradients from return heads. Return heads clamped from above (at least 7.4x weight).
- **RevIN**: Per-sequence normalization (Kim et al. ICLR 2022) before forward pass.
- **Label smoothing**: 0.05 through TwoHot loss pipeline.
- **Auxiliary reconstruction head**: Light MSE decoder (d_latent -> d_model -> 18) with small fixed weight (0.1) to anchor latents to the input space.

## Loss Function

V6 has a 4-tuple return from `get_loss`: `(total, loss_dict, l_disc, outputs)`.
The discriminator loss is optimized with a separate optimizer.

Multi-objective encoder loss (uncertainty-weighted, Kendall et al.):

```
L_encoder = exp(-s0)*L_InfoNCE  [contrastive, clamped s0 >= 0]
           + 0.1*L_VICReg       [fixed weight regularizer]
           + 0.1*L_recon        [fixed weight auxiliary]
           + sum(exp(-si)*L_ret_h)  [returns, clamped si <= -2.0]
           + exp(-sN)*L_regime  [focal, clamped sN <= -1.0]
           + 0.10*L_adv         [encoder fools discriminator, fixed weight]
           + 3.0*L_huber        [direct return regression]

L_disc = BCE(real) + BCE(fake) + 10.0 * grad_penalty (WGAN-GP)
```

- **InfoNCE**: Per-timestep contrastive alignment of predicted vs target latent (memory-safe chunked computation)
- **VICReg**: Variance + covariance regularization to prevent representation collapse
- **Adversarial encoder loss**: Encoder wants to fool discriminator (minimize LAMBDA_ADV=0.10)
- **Discriminator loss**: Binary cross-entropy + WGAN-GP gradient penalty (optimized separately)
- **Return prediction**: TwoHot cross-entropy at horizons [1, 4, 16, 64] with label smoothing (0.05)
- **Direct Huber return**: Bypasses TwoHot discretization bottleneck (weight=3.0)
- **Regime classification**: 3-class (bear/neutral/bull) focal loss (gamma=2.0)

## Training

| Parameter          | Value                          |
|--------------------|--------------------------------|
| Batch size         | 40                             |
| Sequence length    | 96 bars                        |
| Learning rate      | 2.5e-4                         |
| Weight decay       | 1.2e-1                         |
| Epochs             | 180                            |
| Steps/epoch        | 300                            |
| Patience           | 40                             |
| LR schedule        | 5-epoch warmup + cosine decay  |
| Augmentation       | Gaussian noise (0.02) + feature dropout (10%) |
| Mask ratio         | 0.15 -> 0.35 (ramp over 40 epochs) |
| Temporal jitter    | +/- 4 bars                     |

## Production Features

- Dual optimizer: main (encoder + heads) and discriminator trained independently
- Full checkpoint resumability (model + optimizer + discriminator optimizer + scaler)
- EMA model tracking (decay=0.995) for stable evaluation and ShIC computation
- Target encoder EMA updated every step (JEPA_EMA_DECAY=0.996)
- JEPA-specific validation: contrastive accuracy logged alongside IC and ShIC
- Gradient norm logging with NaN detection per loss component
- Windows-safe DataLoader (num_workers=0 on Windows)
- Memory cleanup every 10 epochs

## Validation Gates

The model must pass all gates before agent training begins:

| Gate                       | Threshold    |
|----------------------------|--------------|
| Contrastive accuracy       | > 0.3        |
| Information Coefficient    | > 0.015      |
| Shuffled IC / Contiguous IC| > 0.3        |
| Val/Train loss ratio       | < 2.0        |
| Embedding std              | > 0.05       |

Note: V6 does not use KL divergence gates (no RSSM categorical latent).

## Files

| File                    | Purpose                                                     |
|-------------------------|-------------------------------------------------------------|
| `settings.py`           | All hyperparameters and configuration                        |
| `components.py`         | CausalGRUEncoder, PredictorNetwork, TimeDiscriminator, InfoNCELoss, VICRegLoss, TwoHotSymlog, SwiGLU, MLPHead, RMSNorm |
| `world_model.py`        | CausalJEPAWorldModel with get_loss() (4-tuple) and encode_sequence() |
| `train_world_model.py`  | V6.0 base training loop with dual optimizer + resumability   |
| `train_snapshot.py`     | V6.E multi-seed ensemble orchestrator                        |
| `train_adapter.py`      | V6.X FiLM adapter trainer                                   |
| `train_ncl.py`          | V6.D NCL diversity trainer                                  |
| `adapter.py`            | FiLM adapter module (operates on ctx_latent, d=192)         |
| `ncl_model.py`          | DiversityWorldModel (K=5 heads on V6 backbone)              |
| `snapshot_ensemble.py`  | V6.E ensemble inference wrapper                              |
| `validate_world.py`     | V6.0 base validation                                        |
| `validate_snapshot.py`  | V6.E ensemble validation                                    |
| `validate_adapter.py`   | V6.X adapter validation                                     |
| `validate_ncl.py`       | V6.D NCL diversity validation                               |

## Usage

```bash
cd src/wm/v6/v6_training
python train_world_model.py
```

Checkpoints saved to `models/wm/v6/`. Training resumes automatically from latest checkpoint.

## Strengths and Trade-offs

**Strengths:**
- Adversarial time shuffling provides strong anti-memorization guarantee beyond ShIC stopping
- JEPA objective is more sample-efficient than pixel/feature reconstruction for high-dim inputs
- CausalGRU processes variable-length sequences with O(T) memory (vs O(T^2) for attention)
- Dual-branch momentum encoder provides stable contrastive targets without a large queue

**Trade-offs:**
- Dual optimizer adds training complexity; discriminator must stay balanced with encoder
- Continuous latent space (d_latent=192) lacks the discrete structure of RSSM categorical
- No KL annealing: latent regularization must be managed entirely through VICReg
- get_loss returns 4-tuple `(total, loss_dict, l_disc, outputs)` -- different from RSSM versions (3-tuple)
