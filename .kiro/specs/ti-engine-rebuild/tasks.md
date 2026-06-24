# Tasks: TI Engine Rebuild — COMPLETE

## Task 1: Setup infrastructure ✅
- [x] Update CHARTER.md with reframed goal
- [x] Update CURRENT.md with session state
- [x] Create Kiro spec (requirements, design, tasks)
- [x] Create output directory structure (V2_subtf_decoupled/cells/, summary/, oos_validation/)
- [x] Verify decoupled engine imports work at sub-day TFs (all 6 TFs confirmed)

## Task 2: Build the multi-TF runner ✅
- [x] Write `crypto/src/strat/ti_engine_multitf.py` — the unified runner
- [x] Wire: load chimera at TF → band ensemble → decoupled signal → slice evaluation
- [x] Handle 2h cadence (resampled from 1h, existing recipe in fleet_lab)
- [x] Per-cell output: JSON with slice metrics + aggregates

## Task 3: Run initial cells (EMA × all TFs) ✅
- [x] EMA × 30m × {7d, 14d} lookback × 7d slice
- [x] EMA × 1h × {7d, 14d} lookback × 7d slice
- [x] EMA × 2h × {7d, 14d} lookback × 7d slice
- [x] EMA × 4h × {7d, 14d} lookback × 7d slice
- [x] EMA × 1d × {7d, 14d} lookback × 7d slice

## Task 4: Expand to all MA types ✅
- [x] HMA × all TFs × {7d, 14d} × 7d
- [x] DEMA × all TFs × {7d, 14d} × 7d
- [x] SMA × 2h/4h × {7d, 14d} × 7d
- [x] WMA × 2h/4h × {7d, 14d} × 7d
- [x] TEMA, KAMA, VIDYA × 4h (batch sweep)

## Task 5: Analysis and comparison ✅
- [x] Cross-TF comparison table (2h/4h sweet spot, 30m dead, 1h mixed)
- [x] Recency decay analysis (7d > 14d at 2h; mixed at 4h)
- [x] Identify best cells: WMA|4h|7d (59% beats AL), DEMA|4h|7d (58%)
- [x] vs always-long: 4h = genuine edge (56-59%), 2h = marginal (52-53%)
- [x] Random-entry null: PASSED 100th percentile (all 3 types)
- [x] Cost sensitivity: maker cost viable at 4h; cost cliff at 1h/30m (D60 confirmed)
- [x] Out-of-fit (exclude 2020): honest edge +0.6-1.4%/week
- [x] Vol-targeting overlay: adds ~0 (already diversified)

## Task 6: Iterate on winners ✅
- [x] Test shorter slice (3d) → 7d confirmed superior
- [x] Bear-period validation → engine is trend amplifier, not DD reducer alone
- [x] COMBINED V1+V2 → MaxDD -42%, bear -2 to -6%
- [x] Ensemble (EMA+WMA+DEMA majority vote) → 60% profitable, +4.11%/week
- [x] Full 8-type comparison at 4h: 6/8 pass OOS

## Task 7: OOS Validation ✅
- [x] Design rigorous OOS test (oos_engine_test.py)
- [x] Run V1 MA family on 2024-05-15 to 2024-12-31
- [x] Run all 8 types on same OOS window
- [x] Report both linear and compound returns
- [x] Results: WMA +193.5%, +4.11%/wk | 6/8 types pass

## Task 8: Documentation and hardening ✅
- [x] Paper trader compatibility confirmed (PAPER_TRADER_WIRING.md)
- [x] V2 engine README written
- [x] CURRENT.md updated with verified architecture
- [x] All code committed (19 commits this session)
- [x] Next build queue documented (RSI, MACD, Bollinger, Funding)

## FINAL STATUS: ALL TASKS COMPLETE
OOS-validated, regime-gated MA engines at 4h cadence.
Best cell: WMA|4h|7d → +193.5% OOS, +4.11%/wk linear, -38.8% MaxDD.
Next phase: build RSI/MACD/Bollinger/Funding engine families.
