# V3 World Model: WaveNet-GRU Hybrid

## Architecture

**WaveNet-style Gated Dilated Convolutions + GRU + RSSM**

```
Input (18 features) --> Linear(18+32 -> 96) + RMSNorm + SiLU
    --> Causal shift (predict t from t-1)
    --> WaveNet TCN: 4 layers [96 -> 128 -> 192 -> 256], dilations [1,2,4,8]
        Each layer: tanh(conv_filter) * sigmoid(conv_gate) --> skip + residual + RMSNorm
    --> MultiScaleAggregator: Sum all skip connections --> 1x1 conv stack
    --> CausalGRU: 2-layer GRU (256 hidden), RMSNorm output
    --> RSSM Prior/Posterior (24x24 categorical, 576 flat)
    --> Heads: Reconstruction | Returns (TwoHot x4) | Regime (3-class focal)
```

| Component          | Config                                   |
|--------------------|------------------------------------------|
| TCN channels       | [96, 128, 192, 256] (progressive)        |
| TCN kernel size    | 3                                        |
| TCN dilations      | [1, 2, 4, 8] (31-step receptive field)   |
| GRU hidden         | 256, 2 layers                            |
| RSSM latent        | 24 distributions x 24 classes = 576      |
| Asset embedding    | 32-dim learned per asset                  |
| Return encoding    | TwoHot Symlog, 255 bins, [-1, 1]         |
| Return head dim    | 384 (wider trunk for return priority)    |
| TCN dropout        | 0.20                                     |
| GRU/head dropout   | 0.15                                     |

## Key Design Decisions

- **WaveNet gated activations**: `tanh(filter) * sigmoid(gate)` outperforms ReLU convolutions for sequential financial data; the gate controls information flow at each timestep
- **Causal shift before WaveNet**: The input is shifted by one timestep before entering the TCN to prevent the target observation from leaking into the convolution receptive field
- **Multi-scale skip aggregation**: Each TCN layer captures a different temporal scale (3, 7, 15, 31 steps), all skip connections are projected to 256-dim and summed for a richer final representation
- **GRU over LSTM**: Fewer parameters (3 gates vs 4) with comparable performance; orthogonal init for recurrent weights improves gradient flow
- **Progressive channel expansion**: 96->128->192->256 provides fine-grained early features and richer late features without large upfront cost
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
| Weight decay       | 5e-2     |
| Epochs             | 200      |
| Steps/epoch        | 300      |
| Patience           | 40       |
| LR schedule        | 5-epoch warmup + cosine decay |
| Augmentation       | Gaussian noise (0.02) + feature dropout (10%) |

## Production Features

- Full checkpoint resumability (model + optimizer + scaler + training state)
- EMA model tracking (decay=0.995) for stable evaluation
- Gradient norm logging with NaN detection per loss component
- Windows-safe DataLoader (num_workers=0 on Windows)
- Memory cleanup every 10 epochs
- Validation on ALL data (no batch limit)

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
| `settings.py`           | All hyperparameters and TCN/GRU configuration                        |
| `components.py`         | WaveNetBlock, WaveNetTCN, MultiScaleAggregator, CausalGRU, TwoHotSymlog, SwiGLU, MLPHead |
| `world_model.py`        | WaveNetGRUWorldModel with get_loss() and dream_step()                |
| `train_world_model.py`  | V3.0 base training loop with full resumability                       |
| `train_snapshot.py`     | V3.E multi-seed ensemble orchestrator                                |
| `train_adapter.py`      | V3.X FiLM adapter trainer                                            |
| `train_ncl.py`          | V3.D NCL diversity trainer                                           |
| `adapter.py`            | FiLM adapter module                                                   |
| `ncl_model.py`          | DiversityWorldModel (K=5 heads on V3 backbone)                       |
| `snapshot_ensemble.py`  | V3.E ensemble inference wrapper                                       |
| `validate_world.py`     | V3.0 base validation                                                  |
| `validate_snapshot.py`  | V3.E ensemble validation                                              |
| `validate_adapter.py`   | V3.X adapter validation                                               |
| `validate_ncl.py`       | V3.D NCL diversity validation                                         |

## Usage

```bash
cd src/wm/v3/v3_training
python train_world_model.py
```

Checkpoints saved to `models/wm/v3/`. Training resumes automatically from latest checkpoint.

## Strengths & Trade-offs

**Strengths:**
- Hierarchical temporal processing: TCN captures multi-scale local patterns, GRU handles long-range sequential dynamics
- O(T) computation — no attention matrices
- Multi-scale features from skip connections capture both fast (1-bar) and slow (31-bar) dynamics simultaneously
- WaveNet gating is well-suited for financial time series; the gate acts as a learned filter for relevant signal
- Largest effective receptive field of all non-attention models (TCN 31 steps + GRU unlimited)

**Trade-offs:**
- More architectural components than V1 or V4 (TCN + aggregator + GRU + RSSM), more failure points
- Fixed TCN receptive field per layer (though GRU compensates with long-range memory)
- Channel transition projections add minor parameter overhead
- Smaller RSSM latent (24x24=576 vs V4's 32x32=1024) limits stochastic state capacity
