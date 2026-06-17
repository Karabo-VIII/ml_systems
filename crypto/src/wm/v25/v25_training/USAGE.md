# V25 — Frontier Crypto WM (First-Principles Synthesis)

> **Role in cohort**: V22 iTransformer + frontier components (regime_ffn,
> spectral_norm proj, adversarial regime, tail-Huber, period_emb ablated).
> Most-engineered version in the cohort.
>
> **Status**: same memorization-state as V22 (ShIC=0; encoder anchor
> missing); 17.76M params with ~12M wasted on dead regime_ffn paths;
> SOTA-2026 wired today.

## Purpose

V25 is the **frontier-components synthesis**. Takes V22's iTransformer
backbone and adds every shipping anti-mem mechanism the cohort knows:

- **Spectral norm on embedding proj** (Round-10 Lipschitz bound)
- **Input VIB** (rate-budget bottleneck UPSTREAM of transformer)
- **3-way regime_ffn ModuleList** (encoder-side; per-regime FFN branches)
- **Tail-adaptive Huber** (heavy-tail return loss)
- **Adversarial regime upweighting** (rare regimes get loss weight)
- **Asset token + period_emb ABLATED** (period_emb=False per V25 fix)
- **Forecast head** (same as V22; same revert applies)

The bet: piling on every anti-mem mechanism the literature offers should
crack the memorization wall.

## Architecture (SOTA-2026)

```
Obs (B, T, F=29)
  └── Patch embed [B, F, D]  (spectral-norm proj)
       └── Input VIB (upstream of transformer)
            └── Asset token prepended → InvertedAttention × N
                 └── feat_T → bar_proj → h_seq [B, T, D]
                      ├── RegimeFiLM (NEW 2026-05-16; pre-regime_ffn pre-VIB)
                      ├── regime_gate_per_bar → regime_dist [B, T, 3]
                      ├── regime_ffn[3] gated by USE_REGIME_FFN (default False; 12M dead params)
                      ├── RateBudgetVIB on h_seq → feat_vib
                      │    └── ATME per-sample 0.15 → feat
                      │         ├── return_trunk + return_heads (tail-Huber loss)
                      │         ├── regime_head (with adversarial weighting in loss)
                      │         ├── CC-H5 quantile_heads (NEW today)
                      │         └── CC-H6 regime_cond_heads (NEW; COMPLEMENTARY to regime_ffn)
                      └── feat_T → forecast head (encoder anchor; reverted)
```

### The memorization problem (open — same as V22)

V25 inherits V22's no-encoder-anchor problem:
- `recon=torch.zeros(B, T, 1)` stub at line ~666
- `USE_FORECAST_HEAD=False` (precautionary revert; same as V22)
- IC ≈ +0.21 / ShIC = 0.000 (memorization signature)

V25 has MORE anti-mem defenses than V22 (spectral norm, dropout 0.25,
period_emb ablated) but the same FUNDAMENTAL anchor gap. Per the
expert-architect audit earlier this session: "additional defenses may
prevent the sign-flip regression V22 suffered, but root cause is
unchanged."

**Path A (untested joint)**:
```
USE_CROSS_FEAT_ATTN = True
USE_FORECAST_HEAD = True
FORECAST_WEIGHT = 0.1  # lower than V22's 0.5 due to deeper net
ASSET_TOKEN_DROP_PROB = 0.30  # add (V22 has this; V25 missing)
```

### Architectural concerns

| Concern | Detail |
|---|---|
| 17M params, 12M dead | `regime_ffn` ModuleList(3) with USE_REGIME_FFN=False default → 2 of 3 paths consume optimizer memory but contribute zero. **Refactor saves ~3.5M + ~150MB VRAM**. Deferred to next V25 retrain. |
| Local CryptoPeriodEmbedding + RateBudgetVIB | V25 defines these inline; V22 imports from `_shared/frontier_components`. V25's RateBudgetVIB.__init__ has `beta_init` kwarg that V22's doesn't. Fixes to `_shared` don't propagate. |
| Missing ASSET_TOKEN_DROP_PROB | V22 has 0.30; V25 doesn't define. Asset-token leak risk. |

## Files

```
src/wm/v25/v25_training/
├── settings.py              # 282 lines; richest cohort flags
├── world_model.py           # V25FrontierWorldModel (773 lines)
├── train_world_model.py     # full trainer with bf16 path + meta-learner swarm
```

## Usage

```bash
# Train (SOTA-2026 defaults — period_emb ablated; CC-H5/H6/FiLM ON)
python src/wm/v25/v25_training/train_world_model.py --features 29

# Path A retrain (joint cross_feat_attn + forecast_head):
# Edit settings.py to flip both flags + add ASSET_TOKEN_DROP_PROB
# Then run as above.
```

## Key settings (SOTA-2026)

| Setting | Value | Notes |
|---|---|---|
| `WM_D_MODEL` | 320 | |
| `WM_N_LAYERS` | 6 | deeper than V22 (4) |
| `USE_PATCH_EMBEDDING` | True | |
| `USE_CROSS_FEAT_ATTN` | False | **Path A flips True** |
| `USE_FORECAST_HEAD` | False | **Path A flips True** |
| `USE_PERIOD_EMB` | **False** | ablated 2026-05-10 |
| `USE_SPECTRAL_NORM_PROJ` | True | |
| `USE_INPUT_VIB` | False | (was True; reverted Phase-14.8 — IC collapsed from +0.622 to +0.031) |
| `USE_REGIME_FFN` | False | 12M dead params (refactor queued) |
| `TEMPORAL_CTX_DROP` | 0.15 | per-sample |
| `XD_DROPOUT_RATE` | **0.85** | SOTA-2026 (was 0.7) |
| `WM_FREE_NATS` | 2.0 | |
| `WM_DROPOUT` | 0.25 | higher than cohort (V25-specific) |
| `betas` | (0.9, 0.95) | fixed 2026-05-16 |
| `USE_QUANTILE_HEADS` | True | CC-H5 (NEW today) |
| `USE_REGIME_COND_HEADS` | True | CC-H6 (NEW; complementary to regime_ffn) |
| `REGIME_AWARENESS_MODE` | "film" | RegimeFiLM h_seq pre-VIB (NEW today) |

## Known gaps / queued

| # | Item | Status |
|---|---|---|
| 1 | Path A joint experiment | UNTESTED — same as V22 |
| 2 | 12M dead regime_ffn refactor | DEFERRED (changes ckpt shape) |
| 3 | _shared component migration | DEFERRED (signature divergence — `beta_init` kwarg) |
| 4 | ASSET_TOKEN_DROP_PROB | MISSING (V22 has 0.30; V25 should add) |
| 5 | CC-H3 cross-asset | hook deferred — needs MultiAssetDataset |
| 6 | First post-fix SOTA training | GPU-d pending |
| 7 | Per-version Headline plan section | NOT in WM_HEADLINE_UPGRADE_PLAN (D10 fail per wm_audit) |

## V25 design philosophy

V25 represents "throw everything at the problem":
- Inverted attention from iTransformer (V22)
- Spectral norm + input VIB from anti-mem literature
- Regime-conditional everything (gates + per-regime FFN + adversarial weighting)
- Tail-adaptive Huber for heavy-tail returns

If V25 (with Path A applied) doesn't clear Headline, the conclusion is
likely that DAILY-bar-class architectures can't reach IC > 0.10 without
a structural change (tick-level data, multi-asset path, etc.). V25 is the
cohort's strongest single-asset / daily-bar test of "can we anti-mem our
way to Headline?"
