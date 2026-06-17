# 100-Asset Blanketing Protocol (2026-05-23T09:08)

Tiered priors for cold-start on OOS assets:

- **Tier 1 (TRAIN-specialist)**: per-asset basket from existing catalog. Highest confidence.
- **Tier 2 (Bucket-prior)**: engine recipes appearing in >=3 assets of the SAME DNA bucket. Apply blindly to new assets in that bucket.
- **Tier 3 (Universe-prior)**: engine recipes appearing in >=10 assets cross-bucket. Apply to any new asset.

## Tier 2 (bucket-prior): 2 recipes

Per (bucket, indicator, config, regime, hold) — recipes shared across bucket-members.


### Bucket: VOLATILE (top 10 by mean compound)

| n_assets | indicator | config | regime | hold | mean_compound | mean_catch | assets |
|---:|---|---|---|---:|---:|---:|---|
| 3 | MA_state_EMA_above | period_20 | chop | 1 | +25.4% | 46.0% | DYDX,OP,UNI |

### Bucket: DEGEN (top 10 by mean compound)

| n_assets | indicator | config | regime | hold | mean_compound | mean_catch | assets |
|---:|---|---|---|---:|---:|---:|---|
| 3 | ETF_flow_z | t_0.5 | bull | 1 | +31.7% | 52.1% | DASH,FLOKI,ZEC |

## Tier 3 (universe-prior): 0 recipes

Recipes that work across >=10 cross-bucket assets — apply to any new asset.

| n_assets | n_buckets | indicator | config | regime | hold | mean_comp | mean_catch |
|---:|---:|---|---|---|---:|---:|---:|

## Cold-start protocol for a new OOS asset

1. Identify asset's DNA bucket (BLUE/STEADY/VOLATILE/DEGEN) + sector + liquidity tier
2. Pull Tier 2 bucket-prior recipes (above table per bucket) — apply blindly
3. Pull Tier 3 universe-prior recipes — apply additively
4. After 90+ days of OOS asset history: run discovery_scout on the asset and assess if asset-specific engines beat the priors
5. Promote asset-specific engines into the asset's Tier-1 basket; demote priors

## Honest limit

- The 50 OOS-only assets WILL have lower expected engine quality than the 50 TRAIN-included assets by a factor of ~0.3-0.7×.
- Composition layer should WEIGHT Tier-1 fires > Tier-2 fires > Tier-3 fires when ranking daily picks.
- The expanded catalog provides priors; OOS asset coverage is best-effort, not best-quality.