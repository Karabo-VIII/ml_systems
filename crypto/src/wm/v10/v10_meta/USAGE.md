# V10 — Multi-Brain Meta-Ensemble Router

> **Role in cohort**: NOT a base model. V10 is a tiny learned router (~10K
> params) that combines outputs from FROZEN V1-V9 base models into a single
> production forecast. The router is the "ensemble manager".

## Purpose

V10 is the production-time **ensemble manager**. Given a set of trained base
models (V1.0, V1.1, V1.4, V1.6, V3, V4, V6, V8 — whichever have current
ckpts), V10 learns context-dependent weights:

```
forecast(t) = sum over base_models: weight_m(context_t) × base_m.predict(t)
```

Where `weight_m(context)` is a softmax over a tiny MLP that reads
recent-window features (rolling IC per base, regime label, volatility regime).

The bet: **base models are good in different regimes**. V1.1 dominates bull,
V1.6 dominates chop (more anti-mem), V3 has WaveNet's multi-scale advantage
in fast regimes, V4 wins in long-context Mamba contexts. A learned router
captures this without designing the regime-to-model map by hand.

**Per CLAUDE.md "Indisputable Operating Lens"**: V10 doesn't generate IC; it
multiplies IC of its inputs. The Headline question for V10 is "do its inputs
have pairwise ρ < 0.85 such that ensemble lift > 0.005 ShIC?"

## Architecture

```
Context (rolling IC per base, regime label, vol regime, ...)
  └── tiny MLP (~10K params) → softmax → weights[B, T, n_bases]
                                          │
Base model k=1..N predictions:            │
  base_1.predict(obs) → ret_1[B, T]       │
  base_2.predict(obs) → ret_2[B, T]   ────┤── weighted sum → final_ret[B, T]
  ...                                     │
  base_N.predict(obs) → ret_N[B, T]       │
```

V10 does NOT do its own per-bar inference of base models. It assumes
pre-computed base predictions are stored in features, then learns the
context-dependent mixing.

### Design rationale

- **Why a learned router vs fixed avg**: fixed-weight ensembles ignore that
  V1.1 is better in some regimes and V1.6 is better in others. Router
  context features (regime, vol, recent IC) let the weights ADAPT.
- **Why tiny MLP**: the router is one layer deep on a small feature vector.
  Bigger router = overfits the regime-to-model map (a 10K-param router was
  empirically optimal at u10).
- **Why softmax**: ensures weights are non-negative and sum to 1 (no
  base-model can be "negatively weighted" — would amplify error).
- **Why `strict=False` on ALL base loads**: V10 loads ckpts from MULTIPLE
  architectures (V1.0, V1.1, V1.4, V1.6, V3, V4, V6, V8). Any architectural
  drift between commit-time and inference-time would break ckpt loading
  without `strict=False`. Plus WARN-print on non-trivial mismatch (shipped
  2026-05-16 commit 8afb3e1).

## Files

```
src/wm/v10/v10_meta/
├── settings.py              # router config + paths to base ckpts
├── ensemble_model.py        # Router + base-model loading orchestration
├── meta_ensemble.py         # ensemble-time inference (production callable)
└── train_meta.py            # router training loop
```

## Usage

### Train the router

```bash
# Requires base models trained + their predictions cached
python src/wm/v10/v10_meta/train_meta.py
```

### Inference (production)

```python
from src.wm.v10.v10_meta.meta_ensemble import MetaEnsemble

# Loads all configured base models + router
ensemble = MetaEnsemble.from_config()

# Run forward — returns weighted ensemble prediction
result = ensemble.predict(obs_seq, asset_id)
```

## Key settings

| Setting | What |
|---|---|
| `MODELS_TO_ENSEMBLE` | list of (version_path, ckpt_path) tuples |
| `META_LR` | router learning rate (small) |
| `META_EPOCHS` | router training epochs |
| `META_BATCH_SIZE` | router batch (router is tiny; larger batches OK) |
| `NUM_BINS / BIN_MIN / BIN_MAX` | inherited from base models (must match) |
| `target_prefix` | "target_return" (matches cohort canonical) |

## Retrain cadence

V10 router must be retrained **every time a base model's ckpt is rotated**.
The router's weights are calibrated to the specific base predictions; new
base ckpts = stale router.

**When to retrain**:
- After V1.1-Headline retrain (record-holder; primary)
- After V3-Headline or V4-Headline ships
- After V6-Headline (if V6 clears its kill-or-resize gate)
- NOT after V8 (since V8 is on kill-or-resize watch)

## What V10 is NOT

| Misconception | Reality |
|---|---|
| "V10 has its own encoder" | NO — it's a router only |
| "V10 needs CC-H5 / CC-H6 / FiLM" | NO — those apply to base architectures; V10 routes them |
| "V10 has anti-memo gates" | NO — anti-memo is the base models' job; V10 inherits whatever base produces |
| "V10 has WM_BATCH_SIZE" | NO — meta has its own META_BATCH_SIZE |
| "V10 needs assert_canonical" | NO — V10's settings are router-specific, not WM cross-version invariants |

## Known gaps / queued

| # | Item | Status |
|---|---|---|
| 1 | `strict=False` + WARN print on multi-arch loads | ✓ shipped (commit 8afb3e1) |
| 2 | target_prefix in settings | ✓ shipped (commit 30bd4d3) |
| 3 | V10 router retrain after V3/V4 Headline | QUEUED — fires after V3/V4 first SOTA training |
| 4 | Pairwise ρ measurement between base models | OPEN — should run after each base ckpt rotates to verify ensemble diversity |
| 5 | "Headline portfolio ShIC" gate | OPEN — V10 ensemble's job is to lift cohort to ShIC ≥ 0.05 portfolio-tier |
