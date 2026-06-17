# V3 — WaveNet-GRU Hybrid + RSSM (SOTA-2026 baseline)

> **Role in cohort**: convolutional alternative to V1.x's attention. Tests
> whether dilated causal convolutions + GRU sequential state capture
> dollar-bar dynamics better than self-attention.
>
> **Status: not yet trained on SOTA defaults**. All upgrades shipped 2026-05-16.

## Purpose

V3 explores a **convolutional + recurrent** stack instead of pure attention:

- **WaveNet TCN** — dilated causal convolutions with gated activations (tanh × sigmoid)
- **MultiScaleAggregator** — combines skip connections from all WaveNet layers
- **CausalGRU** — sequential dynamics on top of WaveNet features
- **RSSM 24×24** — same categorical bottleneck as V1.x for fair comparison

The bet: convolution captures local temporal patterns more parameter-
efficiently than attention; GRU adds genuine recurrent state (vs Transformer's
stateless attention).

## Architecture (SOTA-2026 post-upgrade)

```
Obs (B, T=256, F) + asset_emb
  └── Linear(F + 32 → 96) → RMSNorm → SiLU
       └── (causal shift)
            └── 7× WaveNetBlock(channels=[96,128,192,256,256,256,256],
                                    dilations=[1,2,4,8,16,32,64])
                 │                  └─ Receptive field: 255 bars (= seq_len 256)
                 └── MultiScaleAggregator → agg_out [B, T, 256]
                      └── 2-layer CausalGRU → h_seq [B, T, 256]
                           ├── RegimeFiLM (h_seq-only gate, identity-at-init)
                           ├── RSSM prior_head(h_seq) → prior_logits (B, T, 576)
                           ├── RSSM posterior_head(h_seq, obs) → post_logits → z_post
                           │    └── (per-sample ATME 0.15 mixes both paths)
                           └── feat = [h_seq, z_post] (B, T, 832)
                                ├── decoder → recon
                                ├── regime_head → 3-class
                                ├── return_trunk → return_heads (TwoHot 255 bins)
                                ├── CC-H5 quantile_heads (q05..q95 per horizon)
                                └── CC-H6 regime_cond_heads (3 × 4 = 12 per-regime heads)
```

### Anti-memorization (SOTA-2026)

1. **RSSM 24×24** categorical bottleneck (9.2 bits/timestep ceiling)
2. **Per-sample ATME 0.15** — was 0.40 batch-level (legacy ⚠ fixed today)
3. **Free-nats 1.5** — was 1.0; CC-H4 KL floor
4. **XD dropout 0.85** + heavy noise — was 0.7; CC-H4
5. **Block masking 15%** during training
6. **Causal shift** — predict t from t-1
7. **VIB** for clean-variant memorization defense (`USE_VIB` flag)

### Design rationale

- **Why WaveNet over LSTM**: parallel training, no gradient vanishing, native
  dilated multi-scale, half the params of equivalent-RF LSTM
- **Why GRU on top**: WaveNet captures patterns; GRU adds true recurrent state
  that compresses long-context information into a hidden vector
- **Why 7 dilations [1..64]**: receptive field = (kernel-1) × (sum of
  dilations) + 1 = 2 × 127 + 1 = 255 bars matches seq_len 256
- **Why seq_len 256 (was 96)**: 6 days of dollar bars at ~ 4 bars/day; captures
  weekly funding cycle
- **Why per-sample ATME (was batch-level)**: iron-clad audit ⚠ flagged batch-
  level as legacy. Per-sample matches V1.x canonical (CLAUDE.md). Costs 1
  extra posterior forward per batch.
- **Why USE_FORECAST_HEAD=True**: V3 was AutopsyMode-queued for memorization
  in --clean variant; forecast head as encoder anchor is the same root-cause
  fix that resolved V22/V25 memorization.
- **Why CC-H5 quantile heads**: auxiliary distributional output; downstream
  meta-learner sizes positions on q90-q10 spread, not point estimate
- **Why CC-H6 regime-conditional**: Sharpe stability across regime shifts
  (+0.05 Sharpe in shift windows per WM_HEADLINE_UPGRADE_PLAN §0 CC-H6)
- **Why FiLM (REGIME_AWARENESS_MODE="film")**: lightweight encoder-level
  regime conditioning (1.5K params, identity-at-init) supplements CC-H6's
  decoder-only conditioning

## Files

```
src/wm/v3/v3_training/
├── settings.py
├── components.py            # WaveNetTCN, MultiScaleAggregator, CausalGRU, ...
├── world_model.py           # WaveNetGRUWorldModel (with all SOTA-2026 wiring)
├── train_world_model.py     # 977-line trainer
├── validate_world.py
└── adapter.py / snapshot_ensemble.py / ncl_model.py / etc.
```

### Sub-versions (ablation slots)

| Sub | Architecture variant | Purpose |
|---|---|---|
| `v3_training` | base (canonical) | Production target |
| `v3_1_training` | base config | Production-canonical V3 ablation slot |
| `v3_2_training` | dilations `[1,3,9,27]` | Wider-but-sparser receptive field test |
| `v3_3_training` | NO GRU (WaveNet-only) | Tests if GRU's seq state is redundant given dilations |

The sub-versions test individual hypotheses; v3 base + v3_1 are essentially identical (canonical production target).

## Usage

```bash
# Train base (SOTA-2026 defaults — HEADLINE_MODE on, CC-H5 + CC-H6 + FiLM)
python src/wm/v3/v3_training/train_world_model.py --features 29

# Legacy mode (pre-2026-05-16 baseline)
V3_HEADLINE_MODE=0 python src/wm/v3/v3_training/train_world_model.py --features 29

# Ablation runs
python src/wm/v3/v3_1_training/train_world_model.py --features 29     # canonical
python src/wm/v3/v3_2_training/train_world_model.py --features 29     # alt dilations
python src/wm/v3/v3_3_training/train_world_model.py --features 29     # no GRU

# Validate
python src/wm/v3/v3_training/validate_world.py

# Pre-training empirical probe (REQUIRED before first SOTA training)
python scripts/probe_v3_v4_pretrain.py --version v3
```

## Key settings (SOTA-2026 vs pre-upgrade)

| Setting | Pre | Now (SOTA-2026) |
|---|---|---|
| `WM_SEQ_LEN` | 96 | **256** |
| `TCN_DILATIONS` | [1,2,4,8,16,32] | **[1,2,4,8,16,32,64]** |
| `TCN_CHANNELS` | 6 layers | **7 layers** |
| `TEMPORAL_CTX_DROP` | 0.40 (batch) | **0.15 (per-sample)** |
| `WM_FREE_NATS` | 1.0 | **1.5** |
| `XD_DROPOUT_RATE` | 0.7 | **0.85** |
| `HEADLINE_MODE` | env-var "0" | env-var **"1"** |
| `USE_QUANTILE_HEADS` | n/a | **True** (CC-H5) |
| `USE_REGIME_COND_HEADS` | n/a | **True** (CC-H6) |
| `REGIME_AWARENESS_MODE` | n/a | **"film"** |
| `target_prefix` | missing | **"target_return"** (added by deep-audit framework) |
| AdamW betas | (0.9, 0.95) ✓ | unchanged ✓ |

## Last known metrics (pre-upgrade; not yet retrained on SOTA defaults)

- IC ≈ baseline pending — V3 was untrained at f29 per WM_HEADLINE_UPGRADE_PLAN §6
- Per plan §6, V3-Headline projected **IC ≥ 0.080 / ShIC ≥ 0.040** at 4.5 GPU-d

## Pre-training pre-flight (REQUIRED)

```bash
# Per CLAUDE.md §12 empirical probe
python scripts/probe_v3_v4_pretrain.py --version v3 --steps 200
```

Validates: forward shape at seq_len 256; VRAM fits 8GB at B=32; loss
decreasing; h_seq stable; pinball loss positive; ATME math correct.

## Known gaps / queued

| # | Item | Status |
|---|---|---|
| 1 | CC-H3 cross-asset head | hook injected (no-op); needs MultiAssetDataset |
| 2 | CC-H1 multi-resolution stack | NOT WIRED (V3 already has multi-scale agg) |
| 3 | Regime label quality | regime_quality.py flagged persistence + cross-asset gaps |
| 4 | V3 pre-SOTA-upgrade ckpts | OBSOLETE (architecture changed; will retrain cold) |
