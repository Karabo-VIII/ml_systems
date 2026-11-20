# V17 — TD-MPC2 (decoupled WM + MPPI planner)

**Status**: architecture + smoke test built. Trainer + actual training pending Job 2.

**Source**: Hansen et al. 2024, "TD-MPC2: Scalable, Robust World Models for Continuous Control" (arxiv 2310.16828).

**Files**:
- `v17_training/td_mpc2.py` — WM (encoder + dynamics + reward + value + continue) + MPPIPlanner

**Smoke test**: `python src/wm/v17/v17_training/td_mpc2.py`
- 554,020 params
- Forward + backward + 1-step plan output

**Architecture**:
- **Encoder**: obs → latent z (64-dim by default)
- **Dynamics**: (z, a) → z_next (latent transition; deterministic)
- **Reward**: (z, a) → r (scalar MSE)
- **Value**: z → V(z) (scalar)
- **Continue**: z → continue (sigmoid)
- **Slow EMA target value**: tau=0.05
- **NOT generative** (no reconstruction loss; value-equivalent only)

**MPPI Planner**:
- 64 sample trajectories × 5-step horizon (configurable)
- 6 refinement iterations
- Softmax(return / temperature) weighting
- Iteratively re-fits action mean + std

**vs DreamerV3** (V16):
| | V16 (Dreamer) | V17 (TD-MPC2) |
|---|---|---|
| WM type | Generative + RSSM | Value-equivalent (no recon) |
| Latent | Categorical 32×32 | Continuous 64-dim |
| Agent | Amortized actor-critic | MPPI planning at inference |
| Compute (training) | RSSM expensive | Cheaper (no recon) |
| Compute (inference) | Cheap (one-shot policy) | Expensive (MPPI loop) |
| Scaling | Cleanest at 200M | Cleanest 80M-1B per Hansen 2024 |

**Math correctness**: validated by `tests/test_model_math.py`:
- MPPI plan finite + correct shape

**Pending work** (M2/M3 budget remaining):
- Trainer script using TrainingLoader
- Reward-equivalence ablation (consistency-only loss)
- MPPI vs random shooting comparison
- Validation harness

**Source of truth**: docs/MODEL_LAYER_OPTION_B_2026_04_26.md
