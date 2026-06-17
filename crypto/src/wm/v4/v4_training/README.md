# V4 World Model: Mamba-RSSM

## Architecture

**Mamba Selective State Space Model + RSSM with Multi-Horizon Prediction**

```
Input (18 features) --> Linear(18+32 -> 384) + RMSNorm + SiLU
    --> Causal shift (predict t from t-1)
    --> 2x MambaBlock (d=384, d_state=16, expand=2)
        Each: RMSNorm --> TitaniumSSM (depthwise conv + selective scan + SiLU gate) --> Residual
    --> RSSM Prior/Posterior (32x32 categorical, 1024 flat)
    --> Heads: Reconstruction | Returns (TwoHot x4) | Regime (3-class focal)
```

| Component         | Config                               |
|-------------------|--------------------------------------|
| Hidden dim        | 384                                  |
| SSM state dim     | 16                                   |
| Mamba layers      | 2                                    |
| Expansion factor  | 2 (inner dim = 768)                  |
| Depthwise conv    | kernel=4, groups=d_inner (local mix) |
| RSSM latent       | 32 distributions x 32 classes = 1024 |
| Asset embedding   | 32-dim learned per asset              |
| Return encoding   | TwoHot Symlog, 255 bins, [-1, 1]     |
| Return head dim   | 384 (wider trunk for return priority) |
| Dropout           | 0.10                                 |

## Key Design Decisions

- **Mamba SSM (O(T) complexity)**: Linear-time selective state space replaces O(T^2) attention; the model learns data-dependent state transitions rather than pairwise position interactions
- **TitaniumSSM with JIT**: Custom selective scan compiled with `torch.jit.script` for GPU efficiency; selective scan loop executes in fused CUDA kernels
- **Depthwise conv for local context**: kernel=4 depthwise convolution inside TitaniumSSM provides local feature mixing before the SSM step
- **Largest RSSM in the ensemble**: 32x32=1024 flat latent (vs V3's 576 or V1's 576) gives greater stochastic state capacity
- **No action conditioning during WM training**: Actions are noise; policy is learned in Phase 2 via PPO agent dreaming in the learned world model
- **RevIN**: Per-sequence normalization (Kim et al. ICLR 2022) before forward pass
- **Label smoothing**: 0.05 through TwoHot loss pipeline
- **Kendall weight corridors**: Asymmetric clamping prevents reconstruction from hogging gradients; returns always get at least exp(2)=7.4x weight
- **Direct return regression**: Huber loss on decoded predictions (weight=3.0) bypasses the TwoHot discretization bottleneck for smoother gradients
- **Block masking curriculum**: Self-supervised objective ramping from 10% to 25% over 40 epochs
- **KL annealing**: KL weight ramps 0->1 over 20 epochs to stabilize early latent learning
- **Gumbel tau annealing**: Temperature 1.0->0.5 over 50 epochs for sharper categorical latents

## Loss Function

Multi-task uncertainty-weighted loss (Kendall et al.):

```
L_total = exp(-s0)*L_rec + exp(-s1)*L_kl + sum(exp(-si)*L_return_h) + exp(-sN)*L_regime + 3.0*L_direct_ret
```

- **Reconstruction**: MSE on input features (in RevIN-normalized space)
- **KL divergence**: Categorical RSSM prior vs posterior (free nats = 1.0, annealed over 20 epochs)
- **Return prediction**: TwoHot cross-entropy at horizons [1, 4, 16, 64] with label smoothing (0.05)
- **Regime classification**: 3-class (bear/neutral/bull) focal loss (gamma=2.0)
- **Direct return regression**: Huber loss on decoded TwoHot predictions (weight=3.0)

## Training

| Parameter          | Value    |
|--------------------|---------:|
| Batch size         | 48       |
| Sequence length    | 96 bars  |
| Learning rate      | 3e-4     |
| Weight decay       | 1e-2     |
| Epochs             | 100      |
| Steps/epoch        | 500      |
| Patience           | 30       |
| LR schedule        | 5-epoch warmup + cosine decay |
| Augmentation       | Gaussian noise (0.02) + feature dropout (10%) |

## Production Features

- Full checkpoint resumability (model + optimizer + scaler + training state)
- EMA model tracking (decay=0.995) for stable evaluation
- Gradient norm logging with NaN detection per loss component
- Windows-safe DataLoader (num_workers=0 on Windows)
- Memory cleanup every 10 epochs
- Validation on ALL data (no batch limit)
- JIT-compiled selective scan for GPU performance (compiled at first run)

## Validation Gates

The model must pass all gates before agent training begins:

| Gate                  | Threshold  |
|-----------------------|------------|
| Reconstruction MSE    | < 0.10     |
| Information Coeff     | > 0.015    |
| KL (collapse)         | > 0.01     |
| KL (explosion)        | < 15.0     |
| ShIC / Contiguous IC  | > 0.3      |
| Val/Train loss ratio  | < 2.0      |

## Files

| File                    | Purpose                                                              |
|-------------------------|----------------------------------------------------------------------|
| `settings.py`           | All hyperparameters and Mamba SSM configuration                      |
| `components.py`         | TitaniumSSM (JIT selective scan), MambaBlock, TwoHotSymlog, SwiGLU, MLPHead |
| `world_model.py`        | MambaWorldModel with get_loss() and dream_step()                     |
| `train_world_model.py`  | V4.0 base training loop with full resumability                       |
| `train_snapshot.py`     | V4.E multi-seed ensemble orchestrator                                |
| `train_adapter.py`      | V4.X FiLM adapter trainer                                            |
| `train_ncl.py`          | V4.D NCL diversity trainer                                           |
| `adapter.py`            | FiLM adapter module                                                   |
| `ncl_model.py`          | DiversityWorldModel (K=5 heads on V4 backbone)                       |
| `snapshot_ensemble.py`  | V4.E ensemble inference wrapper                                       |
| `agent.py`              | PPO agent for dreamer-style policy learning (Phase 2)                |
| `train_agent.py`        | Agent training loop (Phase 2, after WM gates pass)                   |
| `validate_world.py`     | V4.0 base validation                                                  |
| `validate_snapshot.py`  | V4.E ensemble validation                                              |
| `validate_adapter.py`   | V4.X adapter validation                                               |
| `validate_ncl.py`       | V4.D NCL diversity validation                                         |

## Usage

```bash
cd src/wm/v4/v4_training
python train_world_model.py
```

Checkpoints saved to `models/wm/v4/`. Training resumes automatically from latest checkpoint.

After world model passes all gates, train the PPO agent:
```bash
python src/wm/v4/v4_training/train_agent.py
```

## Strengths & Trade-offs

**Strengths:**
- O(T) linear complexity — fastest training per epoch of all models
- Selective state space captures both local (depthwise conv) and long-range (SSM) dependencies
- JIT compilation delivers production-grade GPU throughput for the selective scan loop
- Largest RSSM capacity (32x32=1024) and widest hidden dim (384) in the ensemble
- PPO agent support for full dreamer-style policy learning (only model with Phase 2 agent)

**Trade-offs:**
- JIT compilation at first run adds one-time startup overhead (~30s)
- Custom SSM implementation less battle-tested than Transformer attention
- Larger RSSM (32x32 vs V3's 24x24) may be slightly overparameterized for current data volume
- Shorter training budget (100 epochs vs 200) due to higher per-step compute from 500 steps/epoch
