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
- [ ] EMA × 30m × {7d, 14d} lookback × 7d slice
- [ ] EMA × 1h × {7d, 14d} lookback × 7d slice
- [ ] EMA × 2h × {7d, 14d} lookback × 7d slice
- [ ] EMA × 4h × {7d, 14d} lookback × 7d slice
- [ ] EMA × 1d × {7d, 14d} lookback × 7d slice

## Task 4: Expand to HMA and DEMA
- [ ] HMA × all TFs × {7d, 14d} × 7d
- [ ] DEMA × all TFs × {7d, 14d} × 7d

## Task 5: Analysis and comparison
- [ ] Cross-TF comparison table
- [ ] Recency decay analysis (7d vs 14d lookback)
- [ ] Identify best cells (if any)
- [ ] vs always-long and random-entry baselines
- [ ] Cost sensitivity: at what cost level does each cell break even?

## Task 6: Iterate on winners (if any)
- [ ] Try shorter slice (3d) on best TF
- [ ] Try signal refinement (add vol-confirmation or momentum filter)
- [ ] Run broader universe check (do results hold across ≥20 assets?)

## Task 7: Documentation and commit
- [ ] Write findings into progress_log.md
- [ ] Commit working code + results
- [ ] Update CURRENT.md with honest verdict
