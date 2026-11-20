# V1 World Model: Transformer-RSSM

## Architecture

**Causal Transformer + Recurrent State Space Model (RSSM)**

```
Input (13 features) --> Linear(13+32 -> 256) --> Sinusoidal PosEnc
    --> 3x CausalTransformerBlock (d=256, 8 heads, d_ff=768)
    --> RSSM Prior/Posterior (24x24 categorical, 576 flat)
    --> Heads: Reconstruction | Returns (TwoHot x4) | Regime (3-class)
```

| Component         | Config                              |
|-------------------|-------------------------------------|
| Hidden dim        | 256                                 |
| Attention heads   | 8 (32 dim/head)                     |
| Transformer layers| 3                                   |
| FFN inner dim     | 768 (SwiGLU gating)                 |
| RSSM latent       | 24 distributions x 24 classes = 576 |
| Asset embedding   | 32-dim learned per asset             |
| Return encoding   | TwoHot Symlog, 65 bins, [-2, 2]     |
| Dropout           | 0.15 (attention + FFN)              |

## Key Design Decisions

- **Right-sized for ~57K bars**: 256d/3L vs larger configs that would overfit on ~2,100 sequences
- **Causal self-attention**: Strict triangular mask prevents future information leakage
- **Xavier init for attention, He init for FFN**: Proven initialization for transformer stability
- **Causal shift**: Model predicts timestep t from observations at t-1
- **Block masking curriculum**: Self-supervised objective ramping from 10% to 25% over 40 epochs

## Loss Function

Multi-task uncertainty-weighted loss (Kendall et al.):

```
L_total = exp(-s0)*L_rec + exp(-s1)*L_kl + sum(exp(-si)*L_return_h) + exp(-sN)*L_regime
```

- **Reconstruction**: MSE on input features
- **KL divergence**: Categorical RSSM prior vs posterior (free nats = 1.0)
- **Return prediction**: TwoHot cross-entropy at horizons [1, 4, 16, 64]
- **Regime classification**: 3-class (bear/neutral/bull) cross-entropy

## Training

| Parameter          | Value    |
|--------------------|---------:|
| Batch size         | 32       |
| Sequence length    | 96 bars  |
| Learning rate      | 2e-4     |
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

| Gate              | Threshold  |
|-------------------|------------|
| Reconstruction MSE| < 0.10     |
| Information Coeff | > 0.015    |
| KL (collapse)     | > 0.01     |
| KL (explosion)    | < 15.0     |

## Files

| File                  | Purpose                                        |
|-----------------------|------------------------------------------------|
| `settings.py`         | All hyperparameters and configuration           |
| `components.py`       | CausalTransformerBlock, PositionalEncoding, TwoHotSymlog, SwiGLU, MLPHead |
| `world_model.py`      | TransformerWorldModel class with get_loss() and dream_step() |
| `train_world_model.py`| Production training loop with full resumability |

## Usage

```bash
cd src/wm/v1/v1_training
python train_world_model.py
```

Checkpoints saved to `models/wm/v1/`. Training resumes automatically from latest checkpoint.

## Strengths & Trade-offs

**Strengths:**
- Global context via self-attention (can attend to any past timestep)
- Well-understood architecture with extensive literature
- Clean interpretability of attention patterns

**Trade-offs:**
- O(T^2) attention complexity (mitigated by T=96 sequence length)
- Smaller batch size (32 vs 48) to fit 8GB VRAM with attention matrices
- Slower per-step than SSM-based models (V4)
