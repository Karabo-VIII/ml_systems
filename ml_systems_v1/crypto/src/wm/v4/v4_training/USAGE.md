# V4 — Mamba-3 SSM + RSSM (SOTA-2026 baseline)

> **Role in cohort**: state-space model alternative to Transformer. Tests
> whether selective SSMs with linear-time sequence scaling outperform
> attention on dollar-bar dynamics.
>
> **Status: not yet trained on SOTA defaults**. All upgrades shipped 2026-05-16.

## Purpose

V4 explores **Mamba-3** — the 2026 selective state-space model with:

- **Complex-valued dynamics** via data-dependent RoPE on B/C parameters
- **Trapezoidal discretization** (2nd-order accurate state update)
- **SSD chunk-based parallel scan** (replaces sequential RNN for-loop)
- **QK-Norm on B/C** for training stability

The single architectural advantage no other cohort member has: **linear-time
sequence complexity**. Mamba scales to 1024+ bars almost free, vs Transformer's
O(seq²) attention cost. Post-2026-05-16 upgrade: seq_len=512 (was 96 — that
setting wasted the architecture).

## Architecture (SOTA-2026 post-upgrade)

```
Obs (B, T=512, F) + asset_emb
  └── Linear(F + 32 → d_model=320) → RMSNorm → SiLU
       └── (causal shift)
            └── 4× MambaBlock(d_model=320, complex SSD, IO-aware kernel)
                 └── post_ssm_norm → h_seq [B, T, 320]
                      ├── RegimeFiLM (h_seq-only gate, identity-at-init)
                      ├── RSSM prior_head(h_seq) → prior_logits (B, T, 576)
                      ├── RSSM posterior_head(h_seq, obs) → post_logits → z_post
                      │    └── (per-sample ATME 0.20 mixes both paths)
                      └── feat = [h_seq, z_post] (B, T, 896)
                           ├── decoder → recon
                           ├── regime_head → 3-class
                           ├── return_trunk → return_heads (TwoHot 255 bins)
                           ├── CC-H5 quantile_heads (q05..q95 per horizon)
                           └── CC-H6 regime_cond_heads (3 × 4 = 12)
```

### Anti-memorization (SOTA-2026)

1. **RSSM 24×24** categorical bottleneck
2. **Per-sample ATME 0.20** — already per-sample (V1.6-class) since 2026-05-10
3. **Free-nats 2.0** — already aggressive per round-4 bump
4. **XD dropout 0.85** + heavy noise (CC-H4, upgraded from 0.7)
5. **Block masking 15%**
6. **Causal shift**
7. **post_ssm_norm** clamps SSD output (which grows ~4x per layer)
8. **Forecast head** as encoder anchor (V4-validated +88% train_IC)

### Design rationale

- **Why Mamba over Transformer**: linear-time on seq_len = no quadratic attn
  cost = can use long contexts (512+ bars) cheaply
- **Why complex-valued dynamics (Mamba-3)**: theoretical capacity for
  oscillatory/cyclical patterns (funding cycles, weekly seasonality)
- **Why fp32 forward through Mamba**: SSD output uses complex states; AMP/bf16
  caused numerical drift. `with autocast(enabled=False)` enforces fp32.
- **Why seq_len 512 (was 96)**: Mamba is linear-time on seq, so 512 bars is
  ~5x cost vs 96 (not 27x like Transformer). 512 bars ≈ 6 days of BTC dollar
  bars = enough to see weekly funding regime.
- **Why WM_CHUNK_SIZE=16**: SSD parallel scan operates on chunks; must divide
  WM_SEQ_LEN evenly. 512/16 = 32 chunks; fits VRAM.
- **Why USE_FORECAST_HEAD=True**: V4 probe-validated +88% train_IC over baseline.
  Anchors h_seq to feature-faithful future prediction.
- **Why CC-H5 / CC-H6 / FiLM**: same rationale as V3 — auxiliary distributional
  output + regime conditioning at both decoder and encoder.

## Files

```
src/wm/v4/v4_training/
├── settings.py
├── components.py            # MambaBlock + RMSNorm + TwoHotSymlog
├── world_model.py           # MambaWorldModel (with all SOTA-2026 wiring)
├── train_world_model.py     # ~866 lines
├── validate_world.py
├── evaluate_v4_oos.py       # extra: OOS evaluation runner
└── adapter.py / snapshot_ensemble.py / ncl_model.py / etc.
```

### Sub-versions

| Sub | Hypothesis | Purpose |
|---|---|---|
| `v4_training` | base (canonical) | Production target |
| `v4_1_training` | stronger batch-ATME (0.40) | Tests if higher ATME helps |
| `v4_2_training` | alt config | (alt-experiment slot) |
| `v4_3_training` | alt config | (alt-experiment slot) |

Production target is the base; ablations are alt-experiment slots.

## Usage

```bash
# Train base (SOTA-2026 defaults — seq_len 512, CC-H5 + CC-H6 + FiLM)
python src/wm/v4/v4_training/train_world_model.py --features 29

# Legacy mode (pre-upgrade, seq_len 96)
V4_HEADLINE_MODE=0 python src/wm/v4/v4_training/train_world_model.py --features 29

# Validate
python src/wm/v4/v4_training/validate_world.py

# OOS evaluation
python src/wm/v4/v4_training/evaluate_v4_oos.py

# Pre-training empirical probe (REQUIRED before first SOTA training)
python scripts/probe_v3_v4_pretrain.py --version v4
```

## Key settings (SOTA-2026 vs pre-upgrade)

| Setting | Pre | Now (SOTA-2026) |
|---|---|---|
| `WM_SEQ_LEN` | 96 | **512** (Mamba's linear-time advantage) |
| `WM_CHUNK_SIZE` | 16 | unchanged (32 SSD chunks at seq_len 512) |
| `WM_D_MODEL` | 320 | unchanged (post-2026-05-07 capacity bump) |
| `WM_N_LAYERS` | 4 | unchanged |
| `TEMPORAL_CTX_DROP` | 0.20 | unchanged (already per-sample) |
| `WM_FREE_NATS` | 2.0 | unchanged (already aggressive) |
| `XD_DROPOUT_RATE` | 0.7 | **0.85** (CC-H4) |
| `HEADLINE_MODE` | env-var "0" | env-var **"1"** |
| `USE_QUANTILE_HEADS` | n/a | **True** (CC-H5) |
| `USE_REGIME_COND_HEADS` | n/a | **True** (CC-H6) |
| `REGIME_AWARENESS_MODE` | n/a | **"film"** |
| `target_prefix` | missing | **"target_return"** |
| AdamW betas | drift to (0.9, 0.999) | **(0.9, 0.95)** fixed cohort-wide today |
| `HEADLINE_INTEGRATOR` | n/a | "tsit5" flag (Mamba-2 IO-aware kernel; not yet wired) |
| `HEADLINE_ADJOINT_BACKPROP` | n/a | True flag (memory-efficient backprop; not yet wired) |

## Last known metrics (pre-upgrade — V4 base capacity-fixed May-7)

- Pre-fix V4 (3.47M params): ShIC declining mid-training (memorization signature)
- Post-fix V4 (7.05M params): ✓ iron-clad per May-7 audit; not yet trained on SOTA defaults
- Per plan §7, V4-Headline projected **IC ≥ 0.085 / ShIC ≥ 0.045** at 4.5 GPU-d

## Pre-training pre-flight (REQUIRED)

```bash
python scripts/probe_v3_v4_pretrain.py --version v4 --steps 200
```

**Memory-critical**: V4 at seq_len=512 + B=32 + complex-fp32 forward is the
heaviest configuration in the cohort. Probe must confirm peak VRAM < 7.5GB
on RTX 4060 (8GB).

## Known gaps / queued

| # | Item | Status |
|---|---|---|
| 1 | CC-H3 cross-asset head | hook injected; needs MultiAssetDataset |
| 2 | Mamba-2 IO-aware kernel (`HEADLINE_INTEGRATOR='tsit5'`) | settings flag; kernel wiring queued |
| 3 | Adjoint backprop (memory-efficient long-seq) | settings flag; wiring queued |
| 4 | CC-H1 multi-resolution encoder | NOT WIRED |
| 5 | Pre-SOTA-upgrade ckpts | OBSOLETE (architecture changed) |
