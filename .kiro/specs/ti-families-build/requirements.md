# Requirements: TI Families Build (8 Families, Sequential)

## Goal
Build 8 non-MA TI engine families sequentially. Each family produces engines at
{1d, 4h, 2h, 1h, 30m} using the same decoupled architecture proven by the MA family.

## Build Order
1. RSI (Relative Strength Index)
2. MACD (Moving Average Convergence Divergence)
3. Bollinger Bands
4. Stochastic Oscillator
5. ADX/DMI (Average Directional Index)
6. Supertrend
7. Funding-Rate Extreme (crypto-native)
8. OI-Price Divergence (crypto-native)

## Per-Family Deliverables
- V1: Initial engine build (signal logic + band configs at all 5 TFs)
- V1.1: Iterate/refine based on DEV results (tune thresholds, fix issues)
- Validation: DEV-window per-slice metrics (NO OOS — sealed for later)
- Commit: code + results + per-TF performance table

## Architecture (same as MA, adapted per TI)
- Signal: TI-specific entry/exit logic (NOT MA crossover)
- Lookback: 7d (confirmed optimal from MA work)
- Slice: 7d trade horizon
- Regime gate: V1 BTC SMA100 (the verified Layer 1)
- Cost: maker 0.0006 RT
- Universe: u100 (≥20 assets must participate)
- Data: chimera at all cadences (already available)

## Constraints
- NO OOS data (2024-05-15 onwards is SEALED until all families are built)
- DEV window only (2020-01 to 2024-05-15)
- TRAIN+VAL for config selection, never OOS
- Same verified accounting (lag-1, maker cost, MtM reconciled)
- Use pandas_ta for indicator computation where available

## Success Criteria (per family, DEV-only)
- At least 1 TF with >55% slices profitable
- At least 1 TF that beats always-long majority of weeks
- Random-entry null passed at the best TF
- If ALL TFs fail: document as honest null, move to next family
