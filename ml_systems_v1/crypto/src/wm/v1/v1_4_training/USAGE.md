# V1.4 — Transformer-RSSM + iTransformer FeatureAttention

> **Role in cohort**: cross-feature attention variant; ensemble diversity
> partner to V1.1. Last verified IC ≈ 0.068 / ShIC ≈ 0.031 (Trader tier).

## Purpose

V1.4 = V1.0 + **FeatureAttentionBlock** (iTransformer-style cross-feature
attention). Before the temporal Transformer stack, V1.4 runs a self-attention
across the FEATURE axis — letting each feature "see" the others at the same
timestamp.

The bet: V1.0's per-feature linear projection treats features independently
through the encoder. Real signal probably emerges from interactions (e.g.,
"funding rate × volume z-score" is more predictive than either alone).
Cross-feature attention captures this without explicit feature engineering.

In ensemble: V1.4's distinct lever (cross-feature) vs V1.1's (XD anti-mem)
gives **uncorrelated errors**. V10 meta-router can ensemble them for IC lift.

## Architecture

V1.0 + ONE addition:

```
After obs_encoder, BEFORE causal Transformer:
    h = FeatureAttentionBlock(obs_emb)
      └── Self-attention over [feature_count] tokens (inverted from V1.0)
      └── Each feature attends to other features at same timestep
```

Original iTransformer paper (Liu et al., ICLR 2024): inverts axes so feature
becomes a "token sequence" of length F. This puts cross-feature interaction
INSIDE the attention computation rather than pre-encoder.

V1.4's implementation does ONE feature-attention layer (not the full L-layer
stack from the paper), then concatenates with V1.0's standard temporal stack.
This is "pre-encoder feature mixing" rather than full iTransformer.

### Design rationale

- **Why one feature-attn layer, not L**: V1.4's job is ensemble diversity, not
  outperforming V1.1 alone. One layer is enough to introduce the feature-
  interaction inductive bias. Full iTransformer is V22's territory.
- **Why BEFORE temporal stack**: feature mixing should happen first; temporal
  modeling then operates on mixed features. (Post-temporal feature attention
  was tested in earlier rounds; ShIC dropped — feature mixing leaks temporal
  context if applied after temporal Transformer.)
- **Why kept the rest of V1.0 intact**: minimal-knob change for clean A/B
  comparison. V1.4 vs V1.0 IC delta = the cross-feature lift.

## Files

Same layout as V1.1 (16-17 .py); slightly smaller `train_world_model.py`
(948 vs 1254 lines) because V1.4 doesn't have the V1.1-specific
`train_diversity.py` + `train_ensemble_gating.py`.

## Usage

```bash
# Train
python src/wm/v1/v1_4_training/train_world_model.py --features 29

# Validate
python src/wm/v1/v1_4_training/validate_world.py

# Variants
python src/wm/v1/v1_4_training/train_adapter.py    --features 29   # FiLM .X
python src/wm/v1/v1_4_training/train_snapshot.py   --features 29   # snapshot .E
python src/wm/v1/v1_4_training/train_ncl.py        --features 29   # NCL .D
python src/wm/v1/v1_4_training/validate_adapter.py
python src/wm/v1/v1_4_training/validate_snapshot.py
python src/wm/v1/v1_4_training/validate_ncl.py

# Headline
V1_HEADLINE_MODE=1 python src/wm/v1/v1_4_training/train_world_model.py --features 29
```

## Last known metrics

- **IC = 0.068 / ShIC = 0.031** (Trader tier; raw IC slightly above V1.1)
- Pairwise ρ with V1.1 known < 0.85 → ensemble lift candidate

## Headline path (per WM_HEADLINE_UPGRADE_PLAN §4)

V1.4's distinctive lever extends cleanly:
- **H1**: Extend FeatAttn → cross-feature × cross-asset (joint attention over
  the [bar, feature, asset] tensor) — projected IC +0.012-0.020
- H2: CC-H4 anti-mem ↑
- H3: CC-H6 regime heads
- H4: Pattern P+Q at f29
- **Bundled V1.4-Headline = 3.5 GPU-d → projected IC ≥ 0.085 / ShIC ≥ 0.040**

The full feature×asset joint attention is the natural Headline candidate within
V1.x family. Requires MultiAssetDataset for the asset axis.

## Known gaps / queued

Same as V1.1: CC-H3 needs MultiAssetDataset; CC-H1 / CC-H6 / CC-H7 not wired
yet; V1_HEADLINE_MODE settings present, retrain to validate.
