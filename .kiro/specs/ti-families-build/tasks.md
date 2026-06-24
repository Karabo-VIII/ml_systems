# Tasks: TI Families Build (8 Families)

## Family 1: RSI
- [ ] V1: Define RSI signal logic (oversold/overbought bands, trend confirmation)
- [ ] V1: Build band-config generator using pandas_ta RSI
- [ ] V1: Run multi-TF sweep {1d, 4h, 2h, 1h, 30m}
- [ ] V1.1: Iterate on best TF (threshold tuning, signal refinement)
- [ ] Validate: per-slice metrics, vs always-long, random-null (DEV only)
- [ ] Commit + document results

## Family 2: MACD
- [ ] V1: Define MACD signal logic (signal-line cross, histogram)
- [ ] V1: Build band-config generator
- [ ] V1: Run multi-TF sweep
- [ ] V1.1: Iterate
- [ ] Validate (DEV only)
- [ ] Commit + document

## Family 3: Bollinger Bands
- [ ] V1: Define BB signal logic (squeeze breakout, band-riding)
- [ ] V1: Build band-config generator
- [ ] V1: Run multi-TF sweep
- [ ] V1.1: Iterate
- [ ] Validate (DEV only)
- [ ] Commit + document

## Family 4: Stochastic Oscillator
- [ ] V1: Define Stochastic signal logic (%K/%D cross, zones)
- [ ] V1: Build band-config generator
- [ ] V1: Run multi-TF sweep
- [ ] V1.1: Iterate
- [ ] Validate (DEV only)
- [ ] Commit + document

## Family 5: ADX/DMI
- [ ] V1: Define ADX signal logic (trend strength threshold + DI cross)
- [ ] V1: Build band-config generator
- [ ] V1: Run multi-TF sweep
- [ ] V1.1: Iterate
- [ ] Validate (DEV only)
- [ ] Commit + document

## Family 6: Supertrend
- [ ] V1: Define Supertrend signal logic (ATR-based flip)
- [ ] V1: Build band-config generator
- [ ] V1: Run multi-TF sweep
- [ ] V1.1: Iterate
- [ ] Validate (DEV only)
- [ ] Commit + document

## Family 7: Funding-Rate Extreme
- [ ] V1: Define funding signal logic (z-score contrarian + chimera features)
- [ ] V1: Build signal generator from chimera fund_rate_z30
- [ ] V1: Run multi-TF sweep
- [ ] V1.1: Iterate
- [ ] Validate (DEV only)
- [ ] Commit + document

## Family 8: OI-Price Divergence
- [ ] V1: Define OI-divergence signal logic (norm_oi_price_divergence)
- [ ] V1: Build signal generator from chimera features
- [ ] V1: Run multi-TF sweep
- [ ] V1.1: Iterate
- [ ] Validate (DEV only)
- [ ] Commit + document

## Final
- [ ] Comprehensive registry of all new families (ranked, per-TF)
- [ ] Update ENGINE_BUILDS.md with new versions
- [ ] Update CURRENT.md
