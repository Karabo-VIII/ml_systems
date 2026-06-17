# V17 — TD-MPC2 Latent Dynamics (Library Backbone)

> **Role in cohort**: Latent dynamics + value head architecture
> (Hansen et al., arXiv 2310.16828, 2024). **Library-only**; no trainer yet.
>
> **Verdict (2026-05-16)**: CONDITIONAL-ARCHIVE — current 0.73M params is
> severely undersized (4.5x below iron-clad floor). The architecture is
> RL-oriented; supervised-IC adaptation is a stretch. **Recommended: defer
> until V11/V12/V13/V14 SOTA trainings reveal compelling gaps; archive if
> cohort is sufficient.**

## Purpose

TD-MPC2 was designed for model-predictive RL planning: at inference, run
CEM (Cross-Entropy Method) search over imagined latent trajectories. V17
adapts the latent dynamics + value head for supervised prediction:

- **Skip CEM planning** (we have realized returns; no need to search)
- **Use the latent dynamics** as a temporal model (residual updates)
- **Use the value head** as a multi-horizon prediction proxy

The bet: latent rollout with CONTINUOUS z (no categorical sampling) gives
cleaner gradient flow than DreamerV3's categorical RSSM, and the smaller
param count may regularize against memorization.

## Architecture (current backbone)

```
Obs (B, T, F=34) + asset_id
  └── Encoder (3-layer GELU, d_z=128)
       └── For t in 0..T-1:
            ├── if t==0: z = z_obs[0]
            ├── else: z = 0.5 * (step_dynamics(z_prev) + z_obs[t])    # filter-style
            └── return_heads × {1,4,16,64}(z) → TwoHot logits

Plus value_head(z[-1]) for cumulative-return forecast.
```

## Smoke (2026-05-16 verified)

```
[v17-tdmpc2] params: 732,284 (0.73M)
[v17-tdmpc2] return_logits OK; value: (4, 255)
[v17-tdmpc2] dyn_gate grad: -0.0003  (rollout learning)
[v17-tdmpc2] PASS smoke
```

## Status: CONDITIONAL ARCHIVE

| Axis | Assessment |
|---|---|
| Architecture quality | ✓ Faithful to Hansen 2024; published SOTA |
| Code quality | ✓ Clean implementation; smoke passes; dyn_gate gets gradient |
| Capacity | ❌ 0.73M = **4.5x BELOW iron-clad floor**. Native bumps still cap at ~2M |
| Anti-memo | ⚠ No RSSM bottleneck; no VIB; relies on filter-style averaging |
| Speed | ✅ Fast (continuous latent, no sampling) |
| Trainer | ❌ NOT BUILT |
| Cohort fit | ⚠ RL-oriented; supervised IC is a forced adaptation |

## Honest verdict

**Don't build V17 as a production WM** unless V11/V13/V14 SOTA-2026
trainings show compelling failure modes that TD-MPC2 architecture would
specifically address. Reasons:

1. **Native capacity ceiling**: TD-MPC2's design has d_z=128, d_hidden=256
   by construction. Bumping to d_z=256, d_hidden=512 gets ~2-3M params —
   still below the iron-clad 4M floor. Going larger DEPARTS from the
   published TD-MPC2 design (validating its IC claims becomes harder).
2. **Filter-style hardcoded mix** (line 127: `z = 0.5 * (z_pred + z_obs[t])`)
   — this isn't learned. A real version would have a learned gating
   network. But adding that drifts from the published recipe.
3. **No bottleneck**: V17 has no categorical RSSM, no VIB. Anti-mem
   relies on the small param count alone. For dollar-bar returns (very
   noisy), this likely allows the encoder to memorize.
4. **Value head is mostly redundant**: `value_logits` predicts cumulative
   future return at the LAST timestep. For our cohort the multi-horizon
   return heads already give per-h predictions. Value head doesn't add
   new signal.

## When to RECONSIDER V17

V17 would become interesting if:

- V11/V13/V14 SOTA-2026 trainings all fail to clear ShIC ratio > 0.40
  (suggesting all their dense architectures over-fit)
- A novel filter-style learned-gating approach is published
- TD-MPC2's CEM planning becomes relevant (e.g. for execution-time
  optimization vs prediction-time IC)

Until any of these fire, treat V17 as **archived in-place**.

## Files

```
src/wm/v17/
├── __init__.py
├── README.md
├── tdmpc2_backbone.py       # V17TDMPC2WM + smoke()
└── v17_training/            # EMPTY (no trainer; recommend NOT building)
```

## If you choose to build a V17 trainer anyway (NOT RECOMMENDED)

Required work (~1-1.5 weeks):

1. Capacity bump: d_z 128 → 256, d_hidden 256 → 512 → ~2.5M params (still below floor)
2. Add learned filter gate (replace 0.5 hardcoded mix)
3. Add VIB bottleneck on z (anti-mem)
4. Settings.py with cohort invariants + Headline flags
5. Trainer following V4 pattern
6. First SOTA training; measure IC

Expected outcome: IC 0.040-0.060 (Sizer tier at best). NOT a Headline contender.

## Cross-references

- Paper: Hansen et al., "TD-MPC2: Scalable, Robust World Models for
  Continuous Control", arXiv 2310.16828, 2024
- `docs/WM_VERSION_INVENTORY_2026_04_29.md` Tier 4 — V17 listed as
  "TD-MPC2: placeholder; trainer pending; DORMANT"
- V16/V21 (sibling library stubs) — same status pattern but better viability
