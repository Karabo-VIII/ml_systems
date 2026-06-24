# Requirements: TI Engine Rebuild for Sub-Day Profitability

## Goal
Rebuild and extend the verified V1 MA move-catch engines to sub-day cadences (30m, 1h, 2h, 4h)
using the decoupled 7d-slice / 14d-lookback architecture, measuring per-slice profitability.

## Requirements

### R1: Multi-Timeframe Engine Execution
- Run the decoupled MA engine at {30m, 1h, 2h, 4h, 1d}
- MA types: {EMA, HMA, DEMA} (top 3 from V1 verified results)
- Lookback windows: {7d, 14d} (signal horizon)
- Slice/hold windows: {3d, 7d} (trade horizon)
- Universe: u100 (≥20 symbols must participate)

### R2: Per-Slice Profitability Metrics
- For each (TF × MA_type × lookback × slice) cell, report:
  - % of 7d slices with positive net return
  - Median per-slice ROI (net of maker cost)
  - Mean per-slice ROI
  - MaxDD within any single slice
  - Number of trades per slice (activity)
  - vs always-long same-TF same-slice baseline

### R3: Recency Decay Analysis
- Compare 7d-lookback vs 14d-lookback signal quality
- Hypothesis: recent crossovers (last 7d) should produce better current-window results
- Measure: conditional slice ROI given signal age (0-7d old vs 7-14d old)

### R4: Cross-TF Learning
- Identify which TF produces the best per-slice profitability
- Test whether faster TFs (30m/1h) give MORE FREQUENT but SMALLER wins
- vs slower TFs (4h/1d) give FEWER but LARGER wins
- Find the optimal compounding TF (frequency × magnitude - cost)

### R5: Honest Evaluation Gates
- Beat the always-long baseline at same TF (not just "be positive")
- Beat a random-entry null (enter random, exit at slice boundary)
- Block-bootstrap p05 > 0 on per-slice returns
- No look-ahead in any signal (lag-1 causality verified)
- Maker cost applied on every turnover event

### R6: Cost Reality
- 30m/1h: maker 0.0006 RT per turnover (verified cost model)
- If activity is too high (>2 entries per slice), net after cost becomes critical
- D60 killed 1h at TAKER cost — re-test at MAKER cost with proper decoupling

## Success Criteria
- At least ONE cell (TF × MA × lookback × slice) with:
  - ≥60% slices profitable
  - Median slice ROI > +0.5%
  - Beats always-long by > 0
  - Passes the random-entry null test
- OR: honest documentation that all cells are null (with per-cell evidence)

## Prior Art
- V1 engines (committed, verified): 1d/4h only, coupled lookback/hold (degenerate)
- V1.1 fine-TF (refuted D81): look-ahead leak caught, de-leaked = dominated by coarse
- Decoupled architecture (uncommitted): built but never run at sub-day
- Key insight: V1.1 used EXIT optimization, not LOOKBACK decoupling. Different approach.
