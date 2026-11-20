# Trading Agent -- Usage Guide

## Overview

PPO trading agent that uses a frozen world model as its market representation. The agent observes world model predictions (return forecasts, regime probabilities, uncertainty) plus portfolio state, and outputs target positions per asset.

## Prerequisites

1. A **trained world model checkpoint** (e.g., `v1_0_f13_wm_best_ema.pt`)
2. **Processed data** in `data/processed/` (chimera parquet files with 24 features)
3. Python environment with torch, numpy, scipy, tqdm

## Quick Start

```powershell
# Train agent on V1.E ensemble (recommended)
python src/agent/train_agent.py --ensemble --steps 2000000 --sav

# Train on single V1.0 base world model (13 features)
python src/agent/train_agent.py --world-model v1_0

# Train on V1.1 (default 22 features)
python src/agent/train_agent.py --world-model v1_1 --features 25

# Train on V3 WaveNet (25 features)
python src/agent/train_agent.py --world-model v3 --features 25

# SPOT mode (default: long-only, no funding costs)
python src/agent/train_agent.py --ensemble --spot --steps 2000000
```

## All Options

```powershell
python src/agent/train_agent.py \
    --world-model v1_1       # World model variant (v1_0, v1_1, ..., v9_3)
    --features 25             # 13 (V1.0 base), 17, 18, 20, 22, 25 (V1.1+/V2+)
    --revin                   # Enable RevIN (OFF by default -- causes memorization)
    --steps 2000000           # Total training steps (default: 2M)
    --seed 42                 # Random seed
    --dual-stream             # Use DualStream policy (confidence gate + Hebbian)
    --crucible                # Enable Crucible adversarial training
    --augment                 # Enable stress augmentation (15% of episodes)
    --sav                     # Run SAV robustness test after training
    --eval-only               # Evaluate existing agent (no training)
    --ensemble                # Use V1.E cross-model ensemble
    --ensemble-models v1_0,v1_2  # Custom ensemble model list
    --resume                  # Resume training from latest checkpoint
    --spot                    # SPOT mode (long-only, no funding, higher fees)
    --margin                  # SPOT margin mode (allows shorting)
    --decision-interval 64    # Decide every N bars (default: every bar)
```

## Policy Types

### Baseline (ActorCritic, ~93K params)
Standard MLP PPO with separate actor and critic networks.
```powershell
python src/agent/train_agent.py --ensemble
```

### DualStream (~161K params)
Alpha stream (what to trade) + risk stream (when not to trade) with:
- Confidence gate: sigmoid suppression when risk stream detects confusion
- Hebbian plasticity: fast online adaptation via outer-product fast weights
- Return predictor: auxiliary supervision on per-asset return predictions
```powershell
python src/agent/train_agent.py --ensemble --dual-stream
```

## V1.E Ensemble (Recommended)

The ensemble combines heterogeneous V1 models for better signal diversity:
- Default models: `v1_0, v1_1_f13, v1_1, v1_4, v1_6`
- Feature ordering: `ENSEMBLE_FEATURE_LIST` in `cross_ensemble.py` (single source of truth)
- Models with non-contiguous feature layouts use `feat_indices` for routing

```powershell
# Standard ensemble training with SAV robustness test
python src/agent/train_agent.py --ensemble --steps 2000000 --sav

# With stress augmentation
python src/agent/train_agent.py --ensemble --augment --steps 2000000 --sav

# Custom model composition
python src/agent/train_agent.py --ensemble --ensemble-models v1_0,v1_1,v1_4
```

## Cost Models

### SPOT (with --spot flag)
- Fee: 0.10% per side (taker)
- Slippage: 0.02%
- Funding: none
- Positions: long-only [0, MAX_POS] (or [-MAX_POS, +MAX_POS] with --margin)

### Futures (default)
- Fee: 0.04% per side (taker)
- Slippage: 0.01%
- Funding: 0.01% per 8 hours
- Positions: [-MAX_POS, +MAX_POS]

## Stress Augmentation

15% of training episodes get synthetic stress scenarios:
- **flash_crash**: sudden large negative returns (5-30 bars)
- **vol_spike**: realized vol 2-4x expected (10-50 bars)
- **squeeze**: momentum followed by sharp reversal
- **cost_shock**: effective transaction costs 2-5x normal

```powershell
python src/agent/train_agent.py --ensemble --augment
```

## Environment Details

- **Observation**: 101 dims = 10 assets x 10 (4 return preds + 3 regime + 1 uncertainty + 1 position + 1 unrealized PnL) + 1 (cash)
- **Action**: 10 target positions in [-0.20, +0.20] per asset (or [0, +0.20] in SPOT mode)
- **Gross exposure cap**: sum(|positions|) <= 1.0 (100% of capital)
- **Episode**: 256 bars (~2.7 days of BTC), 96 bars warmup context
- **Costs**: 6 bps per trade (5 taker + 1 slippage) in futures; 12 bps in SPOT
- **Dollar bar timing**: 96 bars ~ 24 hours (4 bars/hour), used for Sharpe annualization
- **Precomputed features**: World model runs ONCE per asset at episode start (not per step)
- **No look-ahead**: unrealized PnL uses previous bar's realized return only
- **Agent targets**: Always uses raw returns (`target_prefix="target_return"`), never voladj

## Decision Interval

With `--decision-interval N`, the agent observes every bar but only executes trades every N bars:
- Between decisions, positions are held constant
- Matches the 1-2 day holding period where profitability exists
- Default: every bar (N=1)

## Evaluation

```powershell
# Evaluate existing agent (in-sample + out-of-sample)
python src/agent/train_agent.py --ensemble --eval-only

# With SAV robustness test
python src/agent/train_agent.py --ensemble --eval-only --sav
```

### Metrics
- **sharpe**: Annualized Sharpe ratio (sqrt(252 * 96) annualization)
- **max_drawdown**: Worst peak-to-trough drawdown
- **win_rate**: Fraction of episodes where portfolio value > initial capital
- **mean_turnover**: Average |position changes| per step
- **mean_cost**: Total transaction + funding costs per episode
- **mean_confidence**: (DualStream only) confidence gate activation

### Validation Gates
- **OOS/IS Sharpe ratio**: > 0.5 (out-of-sample retains >50% of in-sample edge)
- **SAV stability**: > 0.7 (robust to weight perturbation)

## Drawdown Circuit Breaker

Progressive risk management during episodes:
- **15% drawdown**: Halve all positions (one-time, with transaction costs)
- **25% drawdown**: Terminate episode immediately

## World Model Compatibility

| Model | Class | Default Features | dream_step |
|-------|-------|------------------|------------|
| V1 Transformer | TransformerWorldModel | 13 (V1.0) / 25 (V1.1, V1.4, V1.6) | GRU-based |
| V2 JEPA | JEPAWorldModel | 25 | GRU-based |
| V3 WaveNet | WaveNetGRUWorldModel | 25 | Native GRU |
| V4 Mamba | MambaWorldModel | 25 | GRU-based |
| V5 Hybrid | HybridMambaAttentionWorldModel | 25 | GRU-based |
| V6 JEPA+Adv | CausalJEPAWorldModel | 25 | GRU-based |
| V7 ViT | ViTWorldModel | 25 | GRU-based |
| V8 Neural ODE | NeuralODEWorldModel | 25 | GRU-based |
| V9 MoE | MoEWorldModel | 25 | Native GRU |

## Outputs

- **Best checkpoint**: `models/agent/agent_{tag}_best.pt`
- **Latest checkpoint**: `models/agent/agent_{tag}_latest.pt` (for resume)
- **Final model**: `models/agent/agent_{tag}_final.pt`
- **Results JSON**: `logs/agent/agent_{tag}_{timestamp}.json`

## Hardware

- RTX 4060 (8GB): World model ~2GB frozen, agent ~1MB, leaves room for batch
- Training: ~2M steps takes ~90-100 min with V1.E ensemble
