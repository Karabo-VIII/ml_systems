# V10 Meta-Ensemble -- Usage Guide

## 2026-05-02 Frontier-ML upgrade status: ⏸ **DEFER (per B005)**

V10 is the meta-ensemble layer that aggregates predictions from trained
WM versions. Per B005: **defer until ≥ 2 trained inputs survive** the
upgrade probes.

V1.x upgrade flags don't apply directly to V10 (no signal-layer of its
own). The right move is:
1. Run V1.1 + foundational upgrade probes (SAM / PCGrad / MTP / MDN)
2. Whatever survives in V1.x family + V4 (Mamba, B004 R1) feeds V10
3. V10 ensemble ROI: per B004, inferred ρ between V1.x ↔ V4 ↔ V3 is
   ~0.70-0.80 → ensemble lift modest (~10% over best single member)

## Architecture: Meta-Ensemble Aggregator

V10 reads checkpoints from V1.x / V3 / V4 / V6 / V11 / V12 / V14, runs
inference on the same OOS windows, and combines via:
- Average (uniform)
- Weighted by per-version recent-OOS IC
- Stacked (small linear regression with leave-one-out CV)

## Usage (when ≥ 2 inputs are upgrade-validated)

```
python src/wm/v10/v10_meta/train_meta.py \
    --inputs v1_1 v4 \
    --weighting weighted_ic \
    --features 29
```

See [../UPGRADE_INVENTORY_2026_05_02.md](../UPGRADE_INVENTORY_2026_05_02.md)
for the cross-version matrix.
