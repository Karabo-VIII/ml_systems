# Tasks: TI Families Build (8 Families) — COMPLETE

> NOTE (2026-06-24): This checklist predates the SOTA rebuild. All 8 families were
> built via the SOTA framework (`ti_engine_sota.py` + `ti_engine_sota_v11.py`):
> config ENSEMBLE (majority vote) + V1 BTC SMA100 regime GATE + maker COST + decoupled
> lookback/slice. DEV sweeps live in `crypto/runs/engines/V3_ti_families/<FAMILY>/`.
> All families were then put through OOS validation (2024-05-15→2025-01-01) — a step
> BEYOND the original DEV-only scope — after the user flagged the DEV numbers as
> "too good to be true."
>
> HONEST VERDICT (see OOS_VALIDATION_RESULT.md + FORENSIC_AUDIT_RESULT.md):
> The DEV compounds were inflated ~9.5x by the EW-rebalancing premium. OOS exposed
> that NO non-MA family carries a coherent edge — each beats always-long at only one
> scattered timeframe (noise), and families 6-8 deploy 98-100% (always-long-with-tilt).
> The MA family remains the ONLY OOS-validated, random-null-beating, deployable set.

## Family 1: RSI — COMPLETE (HONEST NULL)
- [x] V1: Define RSI signal logic (oversold/overbought bands, trend confirmation)
- [x] V1: Build config ENSEMBLE generator (RSI_CONFIGS, 18 configs)
- [x] V1: Run multi-TF sweep {1d, 4h, 2h, 1h, 30m}
- [x] V1.1: Iterate — trend-confirmation mode (RSI>50+rising vs oversold bounce)
- [x] Validate: DEV strong @4h but OOS = noise (beats AL only @4h, +0.42pp). D50/D51/D52/D53.
- [x] Commit + document

## Family 2: MACD — COMPLETE (NULL OOS)
- [x] V1: Define MACD signal logic (signal-line cross)
- [x] V1: Build config ENSEMBLE generator (MACD_CONFIGS, 15 configs)
- [x] V1: Run multi-TF sweep
- [x] V1.1: Iterate (ensemble vote threshold)
- [x] Validate: DEV +94K% inflated; OOS beats AL only @1h (+1.47pp), single-asset BTC LOST to buy-hold
- [x] Commit + document

## Family 3: Bollinger Bands — COMPLETE (NULL OOS)
- [x] V1: Define BB signal logic (band-riding above middle)
- [x] V1: Build config ENSEMBLE generator (BOLLINGER_CONFIGS, 12 configs)
- [x] V1: Run multi-TF sweep
- [x] V1.1: Iterate
- [x] Validate: OOS beats AL only @1d (+0.99pp, low slice count)
- [x] Commit + document

## Family 4: Stochastic Oscillator — COMPLETE (FAIL)
- [x] V1: Define Stochastic signal logic (%K/%D cross, zones)
- [x] V1: Build config ENSEMBLE generator (STOCHASTIC_CONFIGS, 12 configs)
- [x] V1: Run multi-TF sweep
- [x] V1.1: Iterate
- [x] Validate: fails on DEV (no selectivity)
- [x] Commit + document

## Family 5: ADX/DMI — COMPLETE (1d-only, marginal)
- [x] V1: Define ADX signal logic (efficiency-ratio proxy, trend strength)
- [x] V1: Build config ENSEMBLE generator (ADX_CONFIGS, 12 configs)
- [x] V1: Run multi-TF sweep
- [x] V1.1: Iterate
- [x] Validate: works 1d only, marginal
- [x] Commit + document

## Family 6: Supertrend — COMPLETE (vectorized V1.1; NULL OOS)
- [x] V1: Define Supertrend signal logic (ATR-based flip)
- [x] V1: Build config ENSEMBLE generator (SUPERTREND_CONFIGS)
- [x] V1: Run multi-TF sweep
- [x] V1.1: VECTORIZED ATR (ti_engine_sota_v11.supertrend_held_vec)
- [x] Validate: OOS deploy ~98% (always-long-with-tilt, not selective). +0.76pp @4h is residual.
- [x] Commit + document

## Family 7: Funding-Rate Extreme — COMPLETE (real feature; NULL OOS)
- [x] V1: Define funding signal logic (z-score contrarian)
- [x] V1: Build signal generator — V1 used price PROXY
- [x] V1: Run multi-TF sweep
- [x] V1.1: Wired REAL chimera norm_funding + vectorized (ti_engine_sota_v11)
- [x] Validate: OOS loses to AL at every TF even with real feature. NULL.
- [x] Commit + document

## Family 8: OI-Price Divergence — COMPLETE (real feature; NULL OOS)
- [x] V1: Define OI-divergence signal logic
- [x] V1: Build signal generator — V1 used price PROXY
- [x] V1: Run multi-TF sweep
- [x] V1.1: Wired REAL chimera norm_oi_price_divergence + vectorized (ti_engine_sota_v11)
- [x] Validate: OOS beats AL only @1d (+0.30pp). NULL otherwise.
- [x] Commit + document

## Final
- [x] Comprehensive registry of all families (ranked, per-TF) — OOS_VALIDATION_RESULT.md
- [x] Forensic audit of the numbers — FORENSIC_AUDIT_RESULT.md
- [x] Update CURRENT.md (verified verdict + Phase 2 pointer)
- [x] Phase 2 dynamic-engine plan — docs/PHASE2_DYNAMIC_ENGINES_PLAN.md (not executed)

## OUTCOME
All 8 families built (SOTA), DEV-swept, AND OOS-validated. Non-MA TIs do not survive
OOS. MA family is the sole deployable engine set. Next: Phase 2 dynamic engines
(awaiting user go-ahead) — see docs/PHASE2_DYNAMIC_ENGINES_PLAN.md.
