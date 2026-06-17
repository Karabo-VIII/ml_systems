# V23 — xLSTM (Beck et al., NeurIPS 2024)

> **Role in cohort**: Extended LSTM with sLSTM (scalar) + mLSTM (matrix)
> blocks. The 2024 "LSTM revival" that aims to compete with attention.
>
> **Status**: SOTA-2026 wired (CC-H5/H6/FiLM + cohort flags). Not yet
> trained on the new defaults.

## Purpose

The xLSTM paper revisits LSTM with two key innovations:
1. **sLSTM**: scalar gating with exponential gates (more expressive than sigmoid)
2. **mLSTM**: matrix-valued memory cell (parallelizable like attention)

Stacking sLSTM and mLSTM blocks gives a model that combines LSTM's
sequential inductive bias with attention's parallelism. Beck et al. show
competitive results vs transformers on long-horizon tasks.

V23 ports the architecture to dollar-bar return prediction.

## Architecture (SOTA-2026)

```
Obs (B, T, F=29) + asset_emb
  └── obs_encoder → Linear → d_model
       └── xLSTM stack (interleaved sLSTM + mLSTM blocks)
            └── post_norm → h_seq [B, T, D]
                 ├── RegimeFiLM (h_seq pre-VIB; identity-at-init)
                 └── VIB to_mu/to_logvar → feat
                      ├── ATME per-sample 0.15 → feat
                      │    ├── return_trunk + return_heads (TwoHot)
                      │    ├── regime_head
                      │    ├── CC-H5 quantile_heads
                      │    └── CC-H6 regime_cond_heads
```

### Design rationale

- **Why xLSTM vs vanilla LSTM**: vanilla LSTM gates clip to [0,1] via sigmoid,
  bounding state update magnitude. xLSTM's exponential gates allow much
  larger updates → captures fast transitions (e.g., regime changes).
- **Why mLSTM matrix memory**: scalar LSTM state limits expressive capacity;
  matrix state allows the model to remember structured information without
  growing parameter count exponentially.
- **Why interleave sLSTM + mLSTM**: sLSTM captures local dynamics; mLSTM
  captures longer-range context. Paper's recipe alternates them.

## Files

```
src/wm/v23/v23_training/
├── settings.py
├── world_model.py           # xLSTMWorldModel
└── train_world_model.py
```

## Usage

```bash
# Train (SOTA-2026 defaults)
python src/wm/v23/v23_training/train_world_model.py --features 29

# Validate
python src/wm/v23/v23_training/validate_world.py
```

## Key settings (SOTA-2026)

| Setting | Value | Notes |
|---|---|---|
| Cohort invariants | canonical | |
| `XD_DROPOUT_RATE` | **0.85** | SOTA-2026 (was 0.7) |
| `TEMPORAL_CTX_DROP` | 0.15 | per-sample ATME |
| `betas` | (0.9, 0.95) | fixed 2026-05-16 |
| `USE_QUANTILE_HEADS` | True | CC-H5 (NEW today) |
| `USE_REGIME_COND_HEADS` | True | CC-H6 (NEW today) |
| `REGIME_AWARENESS_MODE` | "film" | RegimeFiLM h_seq pre-VIB (NEW today) |

## Known gaps / queued

| # | Item | Status |
|---|---|---|
| 1 | Per-version Headline plan section | NOT in WM_HEADLINE_UPGRADE_PLAN (D10 fail per wm_audit) |
| 2 | CC-H3 cross-asset | hook deferred — needs MultiAssetDataset |
| 3 | First SOTA training to measure IC | GPU-d pending |
| 4 | xLSTM-specific stability checks | needs first training (sLSTM exponential gates can NaN under fp16) |

## Risks

xLSTM's exponential gates are KNOWN to be numerically fragile. The paper
trains in bf16; V23 should follow. Under fp16, the exp() can overflow on
large gate inputs. If first training shows NaN losses, force fp32 forward
through the xLSTM stack (similar to V4's Mamba complex-state fp32 fix).
