# V16 — DreamerV3 World Model

**Status**: architecture + smoke test built. Trainer + actual training pending Job 2 (full 53-asset v51 build).

**Source**: Hafner et al. 2023, "Mastering Diverse Domains through World Models" (arxiv 2301.04104).

**Files**:
- `v16_training/dreamer_v3.py` — full WM (RSSM + 4 heads + symlog/TwoHot + KL balance)

**Smoke test**: `python src/wm/v16/v16_training/dreamer_v3.py`
- Forward + backward + imagination rollout
- 2,398,746 params total; 31/49 params get gradient on backward (rest are unused buffers / target-side tensors)

**Architecture**:
- **RSSM**: GRUCell recurrent + 32×32 categorical latent (gumbel-softmax straight-through)
- **4 heads**: reconstruction (MSE on symlog), reward (MSE), continue (sigmoid), return (TwoHot symlog 255 bins)
- **KL balance**: 0.8 prior weight, 0.2 posterior weight (per Hafner sec. 4)
- **Free bits**: 1.0 nat per latent dim (prevents posterior collapse)
- **Asset conditional**: per-asset embedding (32-dim) concatenated to obs

**Public methods**:
- `forward_train(obs, actions, asset_ids, returns)` -> dict with loss components + decoded predictions
- `imagine_rollout(init_state, actions)` -> dict with feat / rewards / continues; used by DreamerV3 agent for latent imagination

**Downstream**: `src/agent/dreamer_v3_agent.py` consumes `imagine_rollout` for actor-critic in latent space (frozen V16 + trainable actor + critic).

**Math correctness**: validated by `tests/test_model_math.py`:
- symlog/symexp roundtrip (max err 7.6e-06)
- TwoHot encode/decode roundtrip
- imagine continues in [0, 1]

**Pending work** (M2 budget remaining):
- Trainer script (`v16_training/train.py`) using TrainingLoader
- Asset-conditional ablation
- Free-bits sweep
- Validation harness (IC, ShIC) on the WM heads

**Source of truth**: docs/MODEL_LAYER_OPTION_B_2026_04_26.md
