# V3 TI Families — FORENSIC AUDIT RESULT (2026-06-24)

Ran `(tmp)/_forensic_audit.py` on the MACD 4h engine (the +94,352% headline).
The user was right to be skeptical. Here is the honest decomposition.

## [1] Look-ahead: CLEAN
Truncation test PASSED. `held[:5000]` is identical whether computed on the full
series or on a series truncated at bar 5000. The signal at bar `i` uses only
data up to bar `i`. **No look-ahead bias.**

## [2] Rebalancing premium: THE INFLATION SOURCE
The EW-rebalanced-every-bar method (what the engine does) on the same held names:
- Compound: **+6,078%**
- Linear (sum of bar returns): **+640.5%**
- Mean per bar: +0.0670%
- **Compound / Linear ratio: 9.5x** — this is the compounding magnification

The 9.5x gap between compound and linear is the rebalancing premium + path
compounding. The headline 5-digit compounds are NOT realistic deployable returns.

## [3] Single-asset BTC sanity (MACD 12/26/9): THE HONEST NUMBER
- BTC MACD-held compound: **+162%**
- BTC buy-and-hold: **+655%**
- BTC deploy rate: 49.9%

On a single asset (no rebalancing premium), MACD timing **UNDERPERFORMS** buy-and-hold
on BTC over the DEV window (+162% vs +655%). The signal is in cash ~50% of the time
and misses upside. This is the most sobering and honest data point.

## [4] What this means
The 5-digit headline compounds (+47K-122K%) come from, in order of magnitude:
1. **EW rebalancing premium across 56 assets** (rebalanced every bar) — dominant, ~9.5x
2. **Regime gate avoiding the 2022 bear** — large
3. **The 2020-2021 mega-bull compounding** — large
4. **The actual TI timing signal** — the SMALLEST contributor

## Verdict
- The architecture is sound (no look-ahead, cost applied, causal).
- The headline compound numbers are **inflated artifacts** of EW-rebalancing + multi-year
  compounding, NOT realistic per-asset deployable returns.
- **Report linear returns and single-asset numbers as the honest baseline.**
- The real question — does the TI signal beat random/buy-hold? — is answered by OOS
  validation and the single-asset comparison, NOT the headline compound.
- On the BTC single-asset test, MACD timing LOST to buy-and-hold. This must be
  weighed heavily before deploying any non-MA family.
