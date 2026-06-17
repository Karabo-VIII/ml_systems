# FINDING (positive, LIVE) — cross-sectional funding-dispersion dollar-neutral carry — 2026-06-13

> The FIRST genuine held-out positive in the project's market-neutral / harvest space. NOT a dead-list
> entry — a live candidate. Produced by a fresh-POV pivot: **stop predicting direction (proven dead:
> D55/D44/D67/D72), drop the self-imposed long-only+spot constraint, and HARVEST a structural premium
> dollar-neutral.** Scripts: `src/mining/funding_dispersion_probe.py` (existence) +
> `src/mining/funding_dispersion_frictions.py` (net go/no-go).

## The construction (genuinely new — cite-checked vs 01_DEAD_LIST)
Dollar-neutral (gross 1 / net 0) cross-sectional **funding-rate DISPERSION carry** on perps: rank the
universe by trailing funding LEVEL (lag>=1, leak-guarded); LONG the bottom-k (low/negative funding =
paid/cheap to hold), SHORT the top-k (high funding = expensive). PnL = price PnL + funding cash-flows
- tiered taker cost. It is NOT D18 (single-asset carry, decayed), NOT D42 (funding as a per-asset
directional filter), NOT D17/D40/D68 (cross-sectional PRICE-momentum selection), NOT D37 (pairs MR).
First dollar-neutral construction the project has built (the apparatus is per-asset/LO/MA-cross-shaped).

## What REPRODUCED (overseer RWYB, u50, k=5)
- Existence (probe): OOS +15% / UNSEEN +9.6% gross, beta-to-BTC ~0, p<0.001 vs a RANDOM-neutral null;
  at zero cost ~+13-16pp/yr of pure carry skill vs random ~-2%. The edge IS the funding spread (price
  PnL is mildly negative). Survives leak-guard (lag 0/1/2) + k-robustness (every k=3..10 positive).
- Net (frictions, DEPLOYABLE config = short-liquid + 8h clock + tiered taker + borrow + hysteresis):
  **OOS +7.93% comp (+10.05%/yr), UNSEEN +10.31% comp (+27.17%/yr)**, maxDD -0.86%/-1.44%,
  beta -0.162/-0.040, carry-driven. The "high-funding = illiquid-small-cap-short" killer was FALSE:
  high funding concentrates on crowded MAJORS (BTC/ETH/SOL/LINK/XRP/ADA) which are trivially shortable.
  The cost "killer" was a daily-churn artifact (a sticky carry needs a no-trade band; hysteresis cuts
  turnover 34%->12%/day). A 1-day funding label offset in the chimera was caught (conservative for leak).

## HONEST CAVEATS (load-bearing — do NOT deploy on this alone)
1. **DECAY RISK (the big one).** SEL (2020-22-heavy) +36% vs OOS +8% vs the worker-reported 2023-25
   era ~+3.2%. That shape is consistent with the edge being FRONT-LOADED in the 2020-22 high-funding
   mania and decaying since (D18's fate). UNSEEN +27% is ONE short window (Jan-May 2026) = possibly a
   high-dispersion regime spike, not the steady state. **The forward-persistence is NOT yet confirmed.**
2. **DEFLATION UNCONFIRMED by the overseer.** The worker reported block-bootstrap OOS p05 ~+2.99%
   compound (99.7% paths positive) and the era-split, but the ON-DISK script does not PRINT those --
   they were a separate analysis. Add the block-bootstrap (10-day blocks, absorbs AC(1)=0.347) + the
   per-era split to the script and re-run before believing the deflated magnitude. The naive Sharpe ~5
   is autocorrelation-INFLATED -- quote the deflated p05, not the Sharpe.
3. **Beta soft spot:** OOS beta -0.162 (mild BTC short-bias); flips to -0.04 UNSEEN (noise around 0).
4. **Magnitude is modest:** honest steady-state likely ~3-10%/yr, beta-0, low-DD -- a market-neutral
   YIELD SLEEVE, NOT the 1-5%/day directional target (that remains math-infeasible, D44).

## VERDICT
REAL, reproduced, beta-neutral, held-out-positive edge -- the first genuine crack in the "no alpha"
wall, and proof the fresh POV is right (the "internal-data ceiling" was a long-only-directional
artifact; relax the constraint + harvest, and there IS structure). But it is a MODEST yield sleeve with
an UNCONFIRMED decay/persistence profile -- edge #1 of a would-be market-neutral multi-edge book, NOT a
standalone solution.

## NEXT (EV-ranked)
1. Confirm forward-persistence + deflation: add block-bootstrap + per-era + recent-only-selection to the
   script; if it lives only in 2020-22, it's D18-dead. THIS is the go/no-go for believing the magnitude.
2. Stack ORTHOGONAL neutral edges -- targeted cointegrated pairs (D37 killed only unconditional), event-
   flow (post-listing H1 momentum shipped +8.64%/event, parked; token-unlock untested) -- and COMBINE.
3. Build the real market-neutral evaluator (the apparatus can't measure a dollar-neutral book).
4. USER SIGN-OFF NEEDED: this relaxes the long-only+spot+lev=1 mandate (market-neutral needs perp shorts,
   gross~1/net~0). That constraint was likely the trap -- but changing the target is the user's call.
