# V24 — TimesNet (FFT multi-period + 2D inception)

**Status**: backbone scaffold + smoke test built. Trainer-wiring pending.

**Source**: Wu et al. ICLR 2023, "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis" ([arXiv:2210.02186](https://arxiv.org/abs/2210.02186)).

## Why V24 specifically

Crypto markets carry strong cyclical structure that 1D temporal models capture only implicitly:
- **8-hour funding rate cycle** (Binance perp)
- **24-hour UTC daily cycle** (US/Asia open/close shifts)
- **7-day weekly cycle** (weekend effect, Friday options expiry)

TimesNet **detects** these cycles via FFT, then **reshapes** the 1D series into a 2D tensor with rows=intra-period position, cols=cycle index. Inception 2D convolutions then capture both relations directly. This is the cleanest "explicit cyclical structure" model in our family.

## Architecture (faithful to ICLR 2023 paper §3)

1. **FFT period detection**: top-K frequencies from amplitude spectrum (Algorithm 1).
2. **2D reshape**: for each period p, `[B, T, D] -> [B, D, p, ceil(T/p)]` with right-padding.
3. **Inception 2D conv**: parallel kernels `(1, 3, 5)` capture multi-scale 2D patterns.
4. **Aggregate**: softmax(amplitude)-weighted sum across the K period-specific outputs.
5. **Stacked TimesBlocks** (paper default 2-3 blocks).

## Files

- `timesnet_backbone.py` — TimesBlock + InceptionBlock2D + backbone class + smoke (~330 LOC)

## Smoke test

```powershell
python src/wm/v24/timesnet_backbone.py
```

Verifies forward + backward at B=4, T=96, F=29. Periods are detected per-batch from the FFT amplitude spectrum.

## Iron-clad properties

- ✅ Architecture faithful to published reference (FFT period detection, 2D reshape, inception conv, amplitude-weighted aggregate)
- ✅ V1.x-compatible interface: `forward_train(obs_seq, asset_id)` returns dict with `return_logits`, `regime_logits`, `h_seq`
- ✅ Sized for ~3-5M params at d_model=192, n_blocks=3
- ✅ Anti-memorization: 2D conv has bounded RF per layer; period detection is data-driven (no fixed cycle assumption); ATME 0.15
- ✅ TwoHot symlog 255 bins, [-1, 1] (CLAUDE.md invariants)
- ✅ ACTIVE_HORIZONS=[1, 4, 16, 64] hardcoded as default

## Trainer wiring (pending, ~2 days)

Same pattern as V13: add `v24_training/` dir with `settings.py` + `train_world_model.py`, register in `src/run_all_training.py`.
