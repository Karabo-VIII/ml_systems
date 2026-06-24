# Tasks: TI Engine Rebuild

## Task 1: Setup infrastructure
- [x] Update CHARTER.md with reframed goal
- [x] Update CURRENT.md with session state
- [x] Create Kiro spec (requirements, design, tasks)
- [ ] Create output directory structure
- [ ] Verify decoupled engine imports work at sub-day TFs

## Task 2: Build the multi-TF runner
- [ ] Write `crypto/src/strat/ti_engine_multitf.py` — the unified runner
- [ ] Wire: load chimera at TF → band ensemble → decoupled signal → slice evaluation
- [ ] Handle 2h cadence (resampled from 1h, existing recipe in fleet_lab)
- [ ] Per-cell output: JSON with slice metrics + aggregates

## Task 3: Run initial cells (EMA × all TFs)
- [x] EMA × 30m × {7d, 14d} lookback × 7d slice
- [x] EMA × 1h × {7d, 14d} lookback × 7d slice
- [x] EMA × 2h × {7d, 14d} lookback × 7d slice
- [x] EMA × 4h × {7d, 14d} lookback × 7d slice
- [x] EMA × 1d × {7d, 14d} lookback × 7d slice

## Task 4: Expand to HMA and DEMA
- [x] HMA × all TFs × {7d, 14d} × 7d
- [x] DEMA × all TFs × {7d, 14d} × 7d
- [x] SMA × 2h/4h × {7d, 14d} × 7d (added)
- [x] WMA × 2h/4h × {7d, 14d} × 7d (added)

## Task 5: Analysis and comparison
- [x] Cross-TF comparison table
- [x] Recency decay analysis (7d vs 14d lookback) → 7d wins at 2h, 14d wins at 4h for DEMA
- [x] Identify best cells → DEMA|4h|7d (58% prof, 59% beats AL), WMA|4h (59% beats AL)
- [x] vs always-long baseline → 4h has genuine edge (56-59%), 2h is marginal (52-53%)
- [ ] Random-entry null test (remaining)
- [ ] Cost sensitivity analysis (remaining)

## Task 5b: KEY FINDING
The 2h "sweet spot" is partially illusory — high median is from bullish market, not timing.
The REAL timing edge is at **4h cadence** (DEMA/WMA: 59% of weeks beat always-long).
This is consistent with V1: the engine is a TIMING engine, 4h gives optimal resolution.

## Task 6: Iterate on winners (if any)
- [ ] Try shorter slice (3d) on best TF
- [ ] Try signal refinement (add vol-confirmation or momentum filter)
- [ ] Run broader universe check (do results hold across ≥20 assets?)

## Task 7: Documentation and commit
- [ ] Write findings into progress_log.md
- [ ] Commit working code + results
- [ ] Update CURRENT.md with honest verdict
