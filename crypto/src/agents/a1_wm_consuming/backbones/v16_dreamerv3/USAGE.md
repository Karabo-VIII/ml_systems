# V16 — DreamerV3 (Library Backbone)

> **Role in cohort**: World-model RL architecture (Hafner et al., Nature 2025)
> adapted to supervised crypto IC. **Library-only**; no trainer yet.
>
> **Verdict (2026-05-16)**: BUILD-VIABLE — capacity bump needed first (3.23M
> → 5M+ to clear iron-clad floor); then 1-2 weeks of trainer work. Real
> architectural novelty worth pursuing.

## Purpose

DreamerV3 is the most-cited 2024-25 world-model paper outside the LLM space.
Its claim to fame: a single hyperparameter set works across 150+ domains
(robotics, games, control). V16 adapts the world-model component for
supervised crypto return prediction:

- **Drop the actor-critic loop** (no RL agent; we have realized returns)
- **Keep the RSSM dynamics** (deterministic GRU + categorical 32×32 latent)
- **Keep the multi-head decoder** (recon + reward + continuation + value)
- **Use return as dense supervised signal** instead of sparse RL reward

The bet: DreamerV3's RSSM is more capable than V1.x's 24×24 RSSM
(32×32=1024 states vs 576 states = ~10 bits/step ceiling), and the
deterministic GRU + stochastic z separation matches the published recipe
exactly.

## Architecture (current backbone)

```
Obs (B, T, F=34) + asset_id
  └── Encoder (2-layer GELU, d_model=256)
       └── For t in 0..T-1:
            ├── prior_logits = prior_proj(h_t)
            ├── post_logits = posterior_proj([h_t, obs_emb[t]])
            ├── z_post = gumbel_softmax(post_logits)        # straight-through
            └── h_{t+1} = GRUCell([z_post, obs_emb[t]] proj, h_t)
            
   feat = [h_seq, z_post_seq]  (B, T, d_hidden + flat_dim)
        ├── decoder → recon (B, T, F)
        ├── continue_head → sigmoid (B, T, 1)
        └── return_heads × {1,4,16,64} → TwoHot logits
```

## Smoke (2026-05-16 verified)

```
[v16-dreamer] params: 3,230,751 (3.23M)
[v16-dreamer] return_logits OK; recon: (2, 32, 34)
[v16-dreamer] continue: (2, 32, 1)
[v16-dreamer] backward OK
[v16-dreamer] PASS smoke
```

## Files

```
src/wm/v16/
├── __init__.py
├── README.md
├── dreamerv3_backbone.py    # V16DreamerWM + DreamerRSSM + smoke()
└── v16_training/            # EMPTY (no trainer built yet)
```

## Status: BUILD-VIABLE (with caveats)

| Axis | Assessment |
|---|---|
| Architecture quality | ✅ Faithful to Hafner 2025 paper; published SOTA |
| Code quality | ✅ Clean implementation; smoke + backward pass |
| Capacity | ⚠ 3.23M (below 4M iron-clad floor) — need d_model 256 → 320 OR n_categories 32 → 48 |
| Anti-memo | ⚠ Has RSSM + categorical bottleneck; needs ATME wiring |
| Speed | ⚠ Per-step GRU (sequential, no parallelism) — slow vs V1.x Transformer |
| Trainer | ❌ NOT BUILT (empty `v16_training/`) |
| Cohort fit | ✅ Uses V4's TwoHotSymlog + RMSNorm (consistent) |

## To convert V16 from LIBRARY to PRODUCTION WM

Required work (~1.5-2 weeks):

1. **Capacity bump** (~1 hour): bump `d_model` 256 → 320, `n_categories` 32 → 32 keeps; check params land >4M. Or extend `feat_dim` via d_hidden 256 → 384.

2. **Settings.py** (~2 hours): per-cohort canonical invariants + Headline flags (USE_QUANTILE_HEADS, USE_REGIME_COND_HEADS, REGIME_AWARENESS_MODE, XD_DROPOUT, ATME 0.15).

3. **Trainer** (~3-4 days): adapt V4's `train_world_model.py` pattern. V16 needs:
   - DreamerV3's full loss: `L = recon + reward + continuation + value(symlog) + KL(post||prior)` with free-bits KL
   - All terms with symlog targets (the Dreamer signature)
   - Per-step recurrence makes batching slower; use shorter seq (T=48?) for speed

4. **Validate world** (~1 day): post-training IC/ShIC measurement.

5. **SOTA-2026 wiring** (~1 day): add CC-H5/H6/FiLM following V3/V4 pattern. RegimeFiLM would condition `h` between RSSM steps.

6. **First SOTA training + measure** (~2-3 GPU-d): does V16 clear V1.1 record?

## Headline projection (per WM_HEADLINE_UPGRADE_PLAN tier framework)

If V16 gets to 5M params + full SOTA-2026 wiring:
- Expected IC: 0.060-0.080 (DreamerV3's recurrent latent should outperform V1.x's stateless attention on long-context dynamics)
- Expected ShIC: 0.030-0.045 (RSSM's 10-bit ceiling is anti-mem-strong)
- Verdict: **plausible Trader-tier**; uncertain if it crosses Headline (0.10)

## Known risks

| Risk | Mitigation |
|---|---|
| Sequential GRU = slow training | Use T=48 instead of 96; precompute encoder outputs |
| 32×32 categorical sampling unstable in fp16 | Force fp32 forward (like V4 does) |
| DreamerV3's continuation head is meaningless for non-episodic markets | Set continuation target to 1.0; ignore loss |
| Categorical sampling needs Gumbel-softmax tau annealing | Borrow V1.6's GUMBEL_TAU schedule |

## Cross-references

- Paper: Hafner et al., "Mastering Diverse Domains through World Models", Nature 2025
- `docs/WM_VERSION_INVENTORY_2026_04_29.md` Tier 4 — V16 listed as "DreamerV3: placeholder; trainer pending"
- V4's `components.py` — shared `TwoHotSymlog`, `RMSNorm`
- V1.6's `GUMBEL_TAU` annealing pattern (borrow if categorical sampling needs it)
