# V8 World Model: Neural ODE-RSSM

## Architecture

**Neural ODE Continuous-Time Dynamics + RSSM Categorical Latent**

```
Input (18 features) --> Linear(18+32 -> 256) + RMSNorm + SiLU + Dropout
    h0 = encoded[:, 0, :]   [B, 256] -- initial hidden state from first timestep
    obs_for_ode = obs_proj(obs + asset_emb)  [B, T, 18] -- obs conditioning for ODE

Neural ODE:
    dh/dt = f_theta(h, t, obs_t)   -- observation-conditioned dynamics
    Solved via RK4: substeps=2 (effective dt = 0.5), MLP [256, 512, 512, 256]
    h_seq = RK4Solver(h0, obs_for_ode, t)   [B, T, 256]

RSSM:
    Prior:     MLPHead(h_seq -> 576)
    Posterior: MLPHead(cat(h_seq, obs) -> 576)
    z_post = Gumbel-Softmax(post_logits, tau annealed 1.0->0.5)

Heads:
    feat = cat(h_seq, z_post)   [B, T, 832]
    Reconstruction | Returns (TwoHot x4) | Regime (3-class focal)
```

| Component              | Config                                           |
|------------------------|--------------------------------------------------|
| Hidden dim (d_model)   | 256                                              |
| ODE dynamics MLP       | [256, 512, 512, 256] layers                      |
| ODE solver             | RK4 (Runge-Kutta 4th order)                      |
| ODE substeps           | 2 (effective step size = 0.5)                    |
| RSSM latent            | 24 distributions x 24 classes = 576 flat         |
| Return head dim        | 384                                              |
| Regime head dim        | 256                                              |
| Asset embedding        | 32-dim learned per asset                         |
| Return encoding        | TwoHot Symlog, 255 bins, [-1, 1]                 |
| Dynamics regularization| lambda=0.01 (penalize ||f(h,t,obs)||^2)          |
| Smoothness regularization| lambda=0.005 (penalize non-smooth dynamics)    |
| Dropout                | 0.15                                             |

## Key Design Decisions

- **Continuous-time dynamics via Neural ODE**: Market state evolves along a smooth continuous trajectory dh/dt = f_theta(h, t, obs_t), rather than discrete GRU steps. The dynamics network f_theta is a 4-layer MLP [256, 512, 512, 256] conditioned on both the current hidden state and the current observation.
- **Observation-conditioned ODE**: The dynamics function takes the current observation as input at each integration step, not just at t=0. This allows the ODE to update its trajectory as new data arrives, bridging the gap between purely continuous dynamics and recurrent architectures.
- **RK4 solver with substeps=2**: Fixed-step Runge-Kutta 4 integrator with 2 sub-steps per bar interval (effective dt=0.5). More sub-steps increase accuracy for stiff dynamics but use more VRAM. ODE_SUBSTEPS=1 is the original single-step setting.
- **Sinusoidal time encoding**: The dynamics network receives a sinusoidal time encoding at each integration step for temporal awareness.
- **Initial state from first timestep**: Unlike RNNs that process sequentially, the ODE takes h0 from the encoded first observation and integrates forward. This makes the model sensitive to good initialization at t=0.
- **RSSM on ODE output**: The continuous ODE hidden state h_seq feeds into categorical RSSM prior/posterior heads, providing stochastic modeling on top of the continuous dynamics. KL annealed over 20 epochs.
- **Gumbel tau annealing**: Temperature annealed 1.0 -> 0.5 over 50 epochs for sharper categorical samples at end of training.
- **Dynamics regularization**: Additional loss term ||f(h,t,obs)||^2 (lambda=0.01) penalizes overly complex dynamics, encouraging smooth trajectories. Computed at 8 randomly sampled timesteps per step.
- **Slower LR for ODE stability**: lr=1e-4 (half of most other versions) and longer warmup (10 epochs vs 5) to prevent ODE instability during early training.
- **Kendall weight corridors**: Reconstruction clamped from below (at most 1.0x), returns clamped from above (at least 7.4x), regime at least 2.7x.
- **RevIN**: Per-sequence normalization (Kim et al. ICLR 2022) before forward pass.
- **Label smoothing**: 0.05 through TwoHot loss pipeline.

## Loss Function

Multi-task uncertainty-weighted loss (Kendall et al.) plus dynamics regularization:

```
L_total = exp(-s0)*L_rec         [rec, clamped s0 >= 0, at most 1.0x]
        + exp(-s1)*L_kl * anneal [KL, annealed 0->1 over 20 epochs]
        + sum(exp(-si)*L_ret_h)  [returns, clamped si <= -2.0, at least 7.4x]
        + exp(-sN)*L_regime      [focal, clamped sN <= -1.0, at least 2.7x]
        + 3.0 * L_huber          [direct Huber return regression]
        + 0.01 * L_dynamics      [||f(h,t,obs)||^2 at 8 random timesteps]
```

- **Reconstruction**: MSE on input features (in RevIN-normalized space), decoded from cat(h_seq, z_post)
- **KL divergence**: Categorical RSSM prior vs posterior (free nats = 1.0, annealed over 20 epochs)
- **Return prediction**: TwoHot cross-entropy at horizons [1, 4, 16, 64] with label smoothing (0.05)
- **Direct Huber return**: Bypasses TwoHot discretization bottleneck (weight=3.0)
- **Regime classification**: 3-class (bear/neutral/bull) focal loss (gamma=2.0)
- **Dynamics regularization (V8-specific)**: Penalizes ||f(h,t,obs)||^2 to encourage simple, smooth trajectories (lambda=0.01)

## Training

| Parameter          | Value                           |
|--------------------|---------------------------------|
| Batch size         | 32                              |
| Sequence length    | 96 bars                         |
| Learning rate      | 1e-4 (slower for ODE stability) |
| Weight decay       | 8e-2                            |
| Epochs             | 200                             |
| Steps/epoch        | 300                             |
| Patience           | 40                              |
| LR schedule        | 10-epoch warmup + cosine decay  |
| Augmentation       | Gaussian noise (0.02) + feature dropout (10%) |
| Mask ratio         | 0.10 -> 0.25 (ramp over 40 epochs) |

Note: LR is 1e-4 (half of V1/V7/V9) and warmup is 10 epochs (double V1/V7/V9). ODE dynamics require more careful early-training stability management.

## Production Features

- Full checkpoint resumability (model + optimizer + scaler + training state)
- EMA model tracking (decay=0.995) for stable evaluation
- KL annealing (0 -> 1.0 over 20 epochs) and Gumbel tau annealing (1.0 -> 0.5 over 50 epochs)
- Dynamics regularization loss logged separately (`dynamics_reg` in loss_dict)
- Use CheckpointManager.load_latest(revin=) for checkpoint loading (not manual torch.load)
- Gradient norm logging with NaN detection per loss component
- Windows-safe DataLoader (num_workers=0 on Windows)
- Memory cleanup every 10 epochs

## Validation Gates

The model must pass all gates before agent training begins:

| Gate               | Threshold  |
|--------------------|------------|
| Reconstruction MSE | < 0.10     |
| Information Coeff  | > 0.015    |
| KL (collapse)      | > 0.01     |
| KL (explosion)     | < 15.0     |
| Shuffled IC / IC   | > 0.3      |
| Val/Train loss     | < 2.0      |

## Files

| File                    | Purpose                                                          |
|-------------------------|------------------------------------------------------------------|
| `settings.py`           | All hyperparameters and configuration                             |
| `components.py`         | ODEDynamics, RK4Solver, TwoHotSymlog, SwiGLU, MLPHead, RMSNorm  |
| `world_model.py`        | NeuralODEWorldModel with get_loss() and dynamics_regularization() |
| `train_world_model.py`  | V8.0 base training loop with full resumability                   |
| `train_snapshot.py`     | V8.E multi-seed ensemble orchestrator                            |
| `train_adapter.py`      | V8.X FiLM adapter trainer                                        |
| `train_ncl.py`          | V8.D NCL diversity trainer                                       |
| `adapter.py`            | FiLM adapter module (operates on cat(h_seq, z_post), d=832)     |
| `ncl_model.py`          | DiversityWorldModel (K=5 heads on V8 backbone)                  |
| `snapshot_ensemble.py`  | V8.E ensemble inference wrapper                                  |
| `validate_world.py`     | V8.0 base validation                                             |
| `validate_snapshot.py`  | V8.E ensemble validation                                         |
| `validate_adapter.py`   | V8.X adapter validation                                          |
| `validate_ncl.py`       | V8.D NCL diversity validation                                    |

## Usage

```bash
cd src/wm/v8/v8_training
python train_world_model.py
```

Checkpoints saved to `models/wm/v8/`. Training resumes automatically from latest checkpoint.

**Important:** When loading V8 checkpoints manually, always use CheckpointManager.load_latest(revin=) to correctly restore RevIN state. Direct torch.load will miss RevIN parameters.

## Strengths and Trade-offs

**Strengths:**
- Continuous-time dynamics naturally handles irregularly-spaced data and smooth market state evolution
- ODE trajectory is differentiable end-to-end -- dynamics regularization directly penalizes complexity
- Observation-conditioning at each ODE step allows real-time trajectory correction as new bars arrive
- Smooth latent trajectories are theoretically more robust to small input perturbations

**Trade-offs:**
- Slower per-step training than GRU/Transformer due to RK4 integration (substeps x 4 function evaluations per bar)
- Sensitive to initial h0 quality; bad first-bar encoding can corrupt the entire trajectory
- ODE stability requires lower LR (1e-4) and longer warmup (10 epochs) vs other versions
- Stochastic ODE sampling is less common in practice; checkpoint loading requires CheckpointManager (not raw torch.load)
