# Optimal config per timeframe — is it different? (2026-06-11)

> User /orc question: *"I suspect optimal config per time frame will be different?"* Tested by running
> the walk-forward POOLED config selection (`src/strat/train_ma_walkforward.py --cadence`) at 4h / 1h /
> 1d on u10, TRAIN walk-forward (holdout sealed). 2-MA grid.

## Answer: YES — and uniquely, the difference is STABLE (unlike per-asset/cluster/regime)
Per-cadence POOLED winner (the config that wins pooled across all assets, per walk-forward fold):

| Cadence | Optimal 2-MA config | Across the 4 folds | Wall-clock horizon of the slow MA |
|---|---|---|---|
| **1d** | EMA **50/100** | identical all 4 folds | 100 days |
| **4h** | EMA **50/200** | identical all 4 folds | 200×4h ≈ 33 days |
| **1h** | EMA 50/200 | 3/4 folds (1 SMA) | 200×1h ≈ 8.3 days |

Two things matter here:
1. **The optimal config genuinely differs by timeframe** — daily wants the *faster* pair (50/100),
   intraday wants the *slower* pair (50/200). Counter-intuitively the finer cadence wants the
   *longer* lookback: more bars of intrabar noise need a longer filter to avoid whipsaw. Your
   intuition is correct.
2. **It is STABLE within each cadence** — the same winner repeats across all 4 walk-forward folds.
   This is the crucial contrast: per-ASSET, per-CLUSTER, and per-REGIME config choices did NOT repeat
   fold-to-fold (they flipped on noise / didn't beat pooled — see MA_GRANULARITY_DISCOVERY and
   regime_dna). **Cadence is the first granularity axis where the per-group optimum is consistent.**
   The reason is mechanical, not lucky: bar-duration scales the meaning of a fixed lookback, so each
   cadence has a genuinely different "right" horizon — and that mapping is structural, so it persists.

## The honest bound: stable ≠ tradeable (the cost wall caps the practical value)
- The per-cadence stability is REAL, but the sub-day cadences are not tradeable NET of cost (this
  session + the fork's `intraday_oracle`: 4h/1h causal capture is negative; the cross-instrument
  generalization lab found NO family clears a concentration-robust bar at 4h/1h on u50).
- So the practically-usable config is the **daily** winner (EMA 50/100), which is already what the
  regime-gated trend book uses. The sub-day winners (EMA 50/200) are correct answers to a question the
  cost wall makes moot.
- Implication for a multi-cadence book: IF sub-day ever becomes tradeable (external data /
  maker-only / a real discriminator), the right config is cadence-specific — EMA 50/100 daily,
  EMA 50/200 at 4h. Until then, daily is the only cadence that pays.

## Where this sits in the granularity ladder (now complete)
- per-ASSET config: **dead** (config choice is noise — regime_dna_lab).
- per-CLUSTER (asset_dna): **not robust** (u10 flicker didn't replicate at u50).
- per-REGIME (config-switch): **hurts** (adds losing DOWN-regime configs; regime-GATE instead).
- per-CADENCE: **real + stable** (the one axis that differentiates consistently) — but the value is
  bounded by the cost wall to the daily cadence.
- → **POOLED-per-cadence** is the honest architecture: one config per timeframe, pooled across all
  assets, with daily the only currently-tradeable cadence.

Scope: u10, 2-MA grid, TRAIN walk-forward, holdout sealed. u50 confirmation of the per-cadence
stability is the next check. Repro:
`python -m strat.train_ma_walkforward --universe u10 --folds 5 --family 2MA --cadence 4h` (and 1h/1d).
