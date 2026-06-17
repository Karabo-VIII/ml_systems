# V22 — iTransformer (Inverted Attention)

> **Role in cohort**: published 2024 SOTA (Liu et al., ICLR 2024). Tests
> whether INVERTED attention (features-as-tokens, NOT bars-as-tokens) beats
> conventional sequence transformers on dollar bars.
>
> **Status**: post-2026-05-16 fixes — USE_PERIOD_EMB gated, AdamW betas
> canonical, CC-H5/H6/FiLM wired. **Memorization issue** identified
> 2026-05-10 (IC≈+0.21 / ShIC=0 — no encoder anchor); **Path A** untested
> joint experiment ready (USE_CROSS_FEAT_ATTN=True + USE_FORECAST_HEAD=True).

## Purpose

The iTransformer paper's core insight: time-series transformers should treat
each FEATURE as a token (a sequence over time), not each TIMESTEP as a token
(a sequence over features). This INVERSION:

- Reduces attention complexity from O(T² × d) to O(F² × d)
- Lets the model learn cross-feature interactions natively (which is the
  whole point of multivariate forecasting)
- Aligns with crypto-WM where F (29-121 features) << T (96-512 bars)

V22 is the cohort's bet on this architectural reframing.

## Architecture (SOTA-2026)

```
Obs (B, T, F=29) + asset_emb
  └── Patch-embedding: each feature's T-series → D-dim token  [B, F, D]
       └── Asset token prepended → [B, F+1, D]
            └── N× InvertedAttentionLayer (attention OVER FEATURES, not bars)
                 └── feat_tokens [B, F+1, D]
                      └── Inverted projection: [B, F, D] → [B, F, T]
                           └── Per-bar slice: [B, T, F] → bar_proj → h_seq [B, T, D]
                                ├── RegimeFiLM (h_seq pre-period_emb + pre-VIB)
                                ├── period_emb (gated by USE_PERIOD_EMB; default OFF)
                                ├── RateBudgetVIB → feat_vib
                                │    └── ATME per-sample 0.15 → feat
                                │         ├── return_trunk + return_heads
                                │         ├── regime_head
                                │         ├── CC-H5 quantile_heads
                                │         └── CC-H6 regime_cond_heads
                                └── feat_T → forecast head (V22's encoder anchor; gated)
```

### The memorization problem (open)

Per `WM_HEADLINE_UPGRADE_PLAN_2026_04_30.md` and audit 2026-05-10:
- V22 has NO encoder anchor (`recon=torch.zeros(B,T,1)` stub at line 380)
- Without anchor, encoder minimizes loss by memorizing temporal position
- Result: contiguous IC ≈ +0.21 but **ShIC = 0.000** (pure memorization)
- USE_FORECAST_HEAD was wired 2026-05-10 then REVERTED after sign-flip regression
- Diagnosis: forecast head alone doesn't fix it; need USE_CROSS_FEAT_ATTN=True ALSO

**Path A (untested joint experiment)**:
```
USE_CROSS_FEAT_ATTN = True
USE_FORECAST_HEAD = True
FORECAST_WEIGHT = 0.5
# Run 3-epoch CUDA validation; check ic1 sign + pred_std/real_std ratio
```

**Path B (alternative)**: replace `recon=zeros` with real RSSM-style decoder
+ `RECON_WEIGHT=1.0`. Standard V1.x pattern.

## Files

```
src/wm/v22/v22_training/
├── settings.py              # USE_PERIOD_EMB flag (2026-05-16); USE_FORECAST_HEAD reverted; betas canonical
├── world_model.py           # iTransformerWorldModel + InvertedAttentionLayer
├── train_world_model.py     # full trainer
```

## Usage

```bash
# Train (SOTA-2026 defaults — period_emb gated OFF, CC-H5/H6/FiLM ON)
python src/wm/v22/v22_training/train_world_model.py --features 29

# Path A retrain (joint cross_feat_attn + forecast_head — the open experiment)
# Edit settings.py:
#   USE_CROSS_FEAT_ATTN = True
#   USE_FORECAST_HEAD = True
# Then run as above.
```

## Key settings (SOTA-2026)

| Setting | Value | Notes |
|---|---|---|
| `WM_D_MODEL` | 320 | larger than V1.x (256) |
| `USE_PATCH_EMBEDDING` | True | PatchTST-style |
| `USE_CROSS_FEAT_ATTN` | False | **Path A flips this to True** |
| `USE_FORECAST_HEAD` | False | **Path A flips this to True** |
| `USE_PERIOD_EMB` | **False** | Gated 2026-05-16 (mirrors V25 ablation) |
| `USE_SPECTRAL_NORM_EMBED` | True | Round-10 anti-mem |
| `USE_INPUT_VIB` | True | Round-10 — VIB upstream of transformer |
| `TEMPORAL_CTX_DROP` | 0.15 | per-sample ATME (canonical) |
| `XD_DROPOUT_RATE` | **0.85** | SOTA-2026 (was 0.7) |
| `WM_FREE_NATS` | 2.0 | aggressive (round-4 bump) |
| `betas` | (0.9, 0.95) | fixed 2026-05-16 (was default) |
| `USE_QUANTILE_HEADS` | True | CC-H5 |
| `USE_REGIME_COND_HEADS` | True | CC-H6 |
| `REGIME_AWARENESS_MODE` | "film" | RegimeFiLM h_seq pre-VIB |

## Known gaps / queued

| # | Item | Status |
|---|---|---|
| 1 | Path A joint experiment (cross_feat_attn=True + forecast_head=True) | UNTESTED — open for next GPU allocation |
| 2 | Path B decoder anchor (real recon, not zeros) | alternative path |
| 3 | CC-H3 cross-asset | hook injected; needs MultiAssetDataset |
| 4 | First post-fix SOTA training | GPU-d pending |
