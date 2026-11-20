# V23 — xLSTM (extended LSTM with matrix memory)

**Status**: backbone scaffold + smoke test built. Trainer-wiring pending.

**Source**: Beck et al. NeurIPS 2024, "xLSTM: Extended Long Short-Term Memory" ([arXiv:2405.04517](https://arxiv.org/abs/2405.04517)).

## Why V23 specifically

Recurrent SOTA alternative to V6's GRU JEPA. xLSTM closes the capacity gap with transformers via:
- **Exponential gating** (replaces sigmoid; allows revising past memory)
- **Matrix memory** (mLSTM block — parallel formulation, transformer-like throughput)
- **Stabilized state** via normalization on the cell state

Compute cost is linear in T (vs O(T²) for transformer attention) — cheap to train, cheap to scale.

## Architecture (faithful to NeurIPS 2024 paper §3-4)

- **sLSTMBlock**: scalar cell, exponential input/forget gates with normalizer state n_t for overflow protection. Per paper §3.1.
- **mLSTMBlock**: matrix C_t cell with parallel (q, k, v) projections + outer-product memory updates + associative recall via h_t = (C_t @ q_t) / max(|n_t^T q_t|, 1). Per paper §3.2.
- **Stack**: alternating sLSTM + mLSTM blocks (paper Table 1 default).

## Files

- `xlstm_backbone.py` — sLSTM + mLSTM blocks + backbone class + smoke test (~340 LOC)

## Smoke test

```powershell
python src/wm/v23/xlstm_backbone.py
```

Verifies forward + backward at B=4, T=96, F=29.

## Iron-clad properties

- ✅ Architecture faithful to published reference (sLSTM + mLSTM with stabilized exp gating)
- ✅ V1.x-compatible interface: `forward_train(obs_seq, asset_id)` returns dict with `return_logits`, `regime_logits`, `h_seq`
- ✅ Sized for ~5-7M params at d_model=256, n_layers=6
- ✅ Anti-memorization: ATME 0.15 + exponential gate decay (mLSTM has no temporal-replay shortcut)
- ✅ TwoHot symlog 255 bins, [-1, 1] (CLAUDE.md invariants)
- ✅ ACTIVE_HORIZONS=[1, 4, 16, 64] hardcoded as default

## Trainer wiring (pending, ~2 days)

Same pattern as V13: add `v23_training/` dir with `settings.py` + `train_world_model.py`, register in `src/run_all_training.py`.
