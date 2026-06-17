# V22 — iTransformer (channel-tokenized cross-feature attention)

**Status**: backbone scaffold + smoke test built. Trainer-wiring pending.

**Source**: Liu et al. ICLR 2024, "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting" ([arXiv:2310.06625](https://arxiv.org/abs/2310.06625)).

## Why V22 specifically

V12 (Cross-Asset Attention) is structurally blocked because dollar-bar data is NOT timestamp-synchronized across assets. iTransformer **inverts the transformer's tokenization**: each feature is a token, attention runs across features. Cross-asset modeling becomes a feature-attention problem with **no synchronization requirement**.

This is the cleanest fix for V12's design issue without rebuilding a multi-asset dataloader.

## Architecture (faithful to ICLR 2024 paper §3)

1. **Inverted embedding**: `[B, T, F] -> [B, F, D]` via `Linear(T, D)`. Each feature's full time-series becomes a D-dim token.
2. **Transformer encoder**: N layers of multi-head self-attention OVER features.
3. **Inverted projection**: `[B, F, D] -> [B, F, T]` via `Linear(D, T)`.
4. **Bar aggregation**: feature-mean of decoded time-series → `[B, T]` per-bar signal.
5. **Per-bar return heads**: TwoHot logits at horizons [1, 4, 16, 64].

Asset conditioning via prepended asset token (paper §4.2 covariate extension).

## Files

- `itransformer_backbone.py` — backbone class + smoke test (~310 LOC)

## Smoke test

```powershell
python src/wm/v22/itransformer_backbone.py
```

Verifies forward + backward + correct output shapes at B=4, T=96, F=29.

## Iron-clad properties

- ✅ Architecture faithful to published reference
- ✅ d_model=256 / n_layers=4 / n_heads=8 sized for ~5M params at F=29 (above 4M floor)
- ✅ Anti-memorization: per-feature compression + ATME 0.15 + permutation-invariant aggregation
- ✅ TwoHot symlog 255 bins, [-1, 1] (CLAUDE.md invariants)
- ✅ ACTIVE_HORIZONS=[1, 4, 16, 64] hardcoded as default

## Trainer wiring (pending, ~2 days)

To turn V22 into a live V-version:
1. Add `src/wm/v22/v22_training/settings.py` (mirror V13)
2. Add `src/wm/v22/v22_training/train_world_model.py` (mirror V13)
3. Add `iTransformerBackbone.get_loss()` method (mirror V13's pattern, reusing TwoHotSymlog)
4. Register in `src/run_all_training.py` MODELS list at f29
5. Pre-train gate + smoke + first f29 run
