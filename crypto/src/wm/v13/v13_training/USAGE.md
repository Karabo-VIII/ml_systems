# V13 — Temporal Fusion Transformer (TFT)

> **Role in cohort**: Google's TFT (Lim et al., 2021) adapted to dollar-bar
> crypto. Interpretable cross-feature attention with hard top-k variable
> selection (VSN).

## Purpose

V13 tests whether **interpretable attention + hard top-k feature gating**
beats opaque dense models on dollar-bar returns. TFT's distinctive
mechanism is the Variable Selection Network (VSN): a softmax over the F
features that hard-selects only the top-K most relevant per timestep, then
runs attention on those.

The bet: "at this bar, VPIN and flow matter; ignore the rest" — explicit
feature gating produces a model that:
- Generalizes better (sparser features = less overfitting)
- Is interpretable (you can see which features fired per prediction)
- Has fewer effective params for the same architectural capacity

## Architecture (SOTA-2026)

```
Obs (B, T, F) + asset_emb
  └── VSN per timestep: softmax over F features → keep top-K=8 (default)
       └── GRN (Gated Residual Network) ×N to mix selected features
            └── Multi-head causal attention (2 layers, interpretable)
                 └── post_attn_grn → h_seq [B, T, 256]
                      ├── RegimeFiLM (h_seq-only gate, identity-at-init)
                      ├── period_emb (crypto-native 8h/24h/7d cycles)
                      └── VIB to_mu/to_logvar → feat
                           ├── ATME (per-sample 0.15)
                           ├── return_trunk + return_heads (TwoHot 255 bins)
                           ├── regime_head
                           ├── CC-H5 quantile_heads (SOTA-2026)
                           └── CC-H6 regime_cond_heads (SOTA-2026)
```

### Anti-memorization

1. **VSN hard top-K=8** (HEADLINE_MODE bumps to 16) — sparser features
2. **VIB** stochastic compression
3. **ATME 0.15** per-sample (V1.x canonical)
4. **GRN gated residuals** with **RMSNorm** (fixed 2026-05-16: was nn.LayerNorm which leaked via affine bias)
5. **XD dropout 0.85** (SOTA-2026)
6. **RegimeFiLM** (SOTA-2026 encoder-level conditioning)
7. **CryptoPeriodEmbedding** (hard-coded 8h/24h/7d cycles → encoder can't memorize them)

### Design rationale

- **Why TFT**: published SOTA for time-series. ICLR 2023 results showed it
  outperforming WaveNet + LSTM on irregular-sample regimes.
- **Why VSN hard top-K (vs soft attention everywhere)**: hard gating creates
  an explicit sparsity prior. Empirically generalizes better than dense
  attention when feature count exceeds 30.
- **Why VSN_TOP_K=8 default**: smaller-than-features (we have 29-121
  features); HEADLINE_MODE bumps to 16 for finer-grained selection.
- **Why USE_QUANTILE_LOSS=False (TFT-native quantile reverted)**: 2026-05-10
  attempt to use TFT's native quantile output as primary regressor
  REGRESSED (ShIC dropped). Kept as auxiliary CC-H5 head instead.
- **Why period_emb**: TFT doesn't natively encode known periodicities;
  hard-coding 8h funding / 24h UTC / 7d weekly makes structural cycles
  explicit so the encoder doesn't waste capacity learning them.
- **Why GRN with RMSNorm (was LayerNorm)**: LayerNorm's learned bias DOF
  enables temporal memorization through the affine path. RMSNorm drops
  it. Fixed 2026-05-16 (commit `8afb3e1`).

## Files

```
src/wm/v13/v13_training/
├── settings.py
├── world_model.py           # TFTWorldModel (550+ lines)
├── train_world_model.py     # full trainer
└── components.py            # 0 lines (V13 defines inline)
```

## Usage

```bash
# Train (SOTA-2026 defaults — VSN_TOP_K=16, CC-H5/H6, FiLM)
python src/wm/v13/v13_training/train_world_model.py --features 29

# Legacy
V13_HEADLINE_MODE=0 python src/wm/v13/v13_training/train_world_model.py --features 29

# Validate (uses attention_weights for per-bar interpretability)
python src/wm/v13/v13_training/validate_world.py
```

## Key settings

| Setting | Value | Notes |
|---|---|---|
| `VSN_TOP_K` | 8 default; 16 with HEADLINE_MODE | hard feature-gate cardinality |
| `XD_DROPOUT_RATE` | **0.85** | SOTA-2026 |
| `TEMPORAL_CTX_DROP` | 0.15 | per-sample ATME |
| `USE_QUANTILE_LOSS` | False | TFT-native primary path REVERTED 2026-05-10 |
| `USE_QUANTILE_HEADS` | True | CC-H5 auxiliary (different from above) |
| `HEADLINE_MODE` | **ON** by default | VSN_TOP_K → 16, cross-asset VSN ON |
| `USE_REGIME_COND_HEADS` | True | CC-H6 |
| `REGIME_AWARENESS_MODE` | "film" | RegimeFiLM after attention, pre-VIB |

## Known gaps / queued

| # | Item | Status |
|---|---|---|
| 1 | TFT-native quantile loss as PRIMARY | REVERTED — kept as CC-H5 aux |
| 2 | Cross-asset VSN (asset-level variable selection) | flag exists; impl queued |
| 3 | CC-H3 cross-asset | needs MultiAssetDataset |
| 4 | V13 first SOTA-2026 training | GPU-d allocation pending |
