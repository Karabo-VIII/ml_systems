---
name: architecture_analysis
description: Why V1 Transformer beats V2-V9 and what to do about it -- capacity vs signal analysis
type: project
---

# Architecture Analysis: Why V1 Wins and Path Forward

## The Evidence (f13, post-audit, raw targets)

| Model | Arch | Params | Best ShIC | Peak IC1 | ShIC/IC | Memorization |
|-------|------|--------|-----------|----------|---------|-------------|
| V1.0 | Transformer+RSSM | ~2M | 0.0302 | 0.030 | ~1.0 | 0% |
| V1.1 | Transformer+RSSM | ~2.5M | 0.0284 | 0.028 | ~1.0 | 0% |
| V1.6 | Transformer+RSSM | ~3M | 0.0302 | 0.030 | ~1.0 | 0% |
| V3 | WaveNet-GRU+RSSM | ~5M | 0.0159 | 0.044 | 0.38 | 62% |
| V4 | Mamba-SSM+RSSM | 5.4M | 0.0167 | 0.042 | 0.40 | 60% |
| V9 | MoE(3)+RSSM | 9.1M | 0.0126 | 0.034 | 0.38 | 62% |

## Root Cause: Capacity-Signal Mismatch

With IC~0.03, there are only ~2M params of real cross-sectional signal to learn from 13 features.
Any model capacity beyond ~2M fills with temporal autocorrelation patterns (memorization).

**Key insight:** V3/V4/V9 achieve HIGHER contiguous IC (0.034-0.044 vs 0.030) but the extra
signal is temporal memorization, not generalizable cross-sectional information. ShIC (which
shuffles time order) exposes this.

## Why "Better Architecture" Didn't Help

1. **Mamba (V4)**: Selective state spaces are excellent at long-range temporal dependencies.
   But that's exactly what we DON'T want -- temporal patterns = memorization in financial data.

2. **WaveNet (V3)**: Dilated causal convolutions excel at temporal pattern capture.
   Same problem -- the architecture's strength is the problem's weakness.

3. **MoE (V9)**: 3 regime experts should help if regimes are cross-sectionally distinct.
   But with 9.1M params total, each expert memorizes its own temporal patterns.

4. **Transformers (V1)**: Self-attention is position-invariant by design (needs positional
   encoding to even see order). This makes it naturally resistant to temporal memorization.
   Combined with small size (~2M), it can ONLY learn cross-sectional patterns.

## Path Forward: Two Strategies

### Strategy 1: Shrink V3/V9 to V1 Scale (~2M params)
Reduce d_model, layers, latent dims. Test if their architectural innovations contribute
ensemble diversity at the right capacity. Cheapest experiment.

Target configs:
- V3-tiny: d_model=128, 1 WaveNet layer, RSSM 16x16 (~1.5M params)
- V9-tiny: d_model=128, 2 experts (not 3), RSSM 16x16 (~2M params)

### Strategy 2: Cross-Sectional Architectures
Process features across assets rather than across time. Fundamentally cannot memorize
temporal patterns.

Ideas:
- iTransformer: treats each feature as a token, attention across features not time
- Cross-asset attention: treats each asset as a token, shares info across market
- Tabular models: XGBoost/LightGBM on same features (zero temporal capacity)

### Strategy 3: Ensemble of Tiny Diverse Models
5 different 1-2M param architectures ensembled:
- V1.0 (Transformer)
- V1.4 (FeatureAttention)
- V3-tiny (WaveNet-tiny)
- V9-tiny (MoE-tiny)
- XGBoost/LightGBM (tabular)

## Training Priority (Revised 2026-03-21)

1. **V1.0 f13** -- retrain with all SOTA fixes (baseline verification)
2. **V1.1 f13** -- retrain (compare to V1.0)
3. **V1.4 f13** -- FeatureAttentionBlock (cross-feature attention hypothesis)
4. **V1.6 f13** -- all techniques (KL anneal, Gumbel, ATME)
5. **V1.E ensemble** -- from best V1 checkpoints
6. **V3-tiny / V9-tiny** -- shrunk architectures (if V1.E needs more diversity)
7. **WM-filtered strategy evaluation** -- Donchian + Bollinger with WM filter
