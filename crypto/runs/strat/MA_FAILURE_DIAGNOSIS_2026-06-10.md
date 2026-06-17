# Why "setups → ride → risk-exit" failed flat — diagnosis (2026-06-10)

Multi-agent workflow (internalise the other instance's 108-approach MA decomposition + our nulls → diagnose →
adversarially verify). The adversarial pass found a **real methodology error**, not just a story.

## One-line
We spent days perfecting the **WHEN-to-enter (entry trigger, axis B1)** — the single corner the MA literature names
as the **weakest, most data-mined** use of a moving average — when the only robust uses are the **slow** ones
(B2 regime-filter, B3 trailing-exit, D4 diversified portfolio), and those, done perfectly, pay the **trend-following
premium (~25–48%/yr drawdown-managed), not 2x/yr.**

## The clean split

### Wrong corner (OUR error — fixable)
- All ~10 of our experiments tested the **entry trigger / capture / config-selection (B1×D1)**: oracle-capture,
  walk-forward config, vol→config, vol→length matched-filter — every one null. That is the **predicted** outcome:
  B1-standalone is "lag HURTS, rarely survives cost+OOS." The entry bar inside a multi-candle move carries ~no info;
  being *in* the move is what pays. We re-derived a known result the hard way.
- **The real catch (adversarial):** our price-oracle move-windows were **clamped at the high**, which pre-banks the
  move → makes any "let it ride" exit look worthless **by construction**. So **we never honestly tested
  exit-as-convexity (B3)** — the literature's strongest mechanistic claim ("the lag that dooms the entry HELPS the
  exit: cut losers, let winners run into the fat right tail = positive skew"). That is the one genuinely-open, cheap
  test we owe ourselves.

### Real ceiling (NOT our error — structural, not fixable by effort)
- The robust roles **were** tested properly (mostly by the parallel instance): the **200-day regime book** (Calmar
  19.8 > phase-shift null p95 10.8) and the **vol-scaled TSMOM ensemble** (41% vs 21% exposure-matched null) both
  **beat their nulls — genuine.** Both deliver the **same object: drawdown-managed trend premium ~25–48%/yr**, beats
  buy&hold on return AND drawdown — **structurally short of 2x/yr.**
- 2x/yr (+100%) appeared in **1 of 7 years** (bull 2021, +519%) and was **pure beta**. No alpha lifts the six non-bull years.
- **Structural reason:** academic TSMOM earns its edge from **diversification across uncorrelated instruments**
  (bonds/FX/metals/equities). Crypto is **one-factor** (BTC-beta ~0.55 → ~1 in the cascades that matter). A 50-coin
  "portfolio" is **one bet wearing 50 tickers.** More breadth buys **smoothness** (Calmar u10→u100 1.18×→8×), not
  **return** (the return ceiling doesn't move). The alpha-manufacturing mechanism can't be harvested here.

## What the literature says to do differently
Stop using the MA as a predict-the-turn entry. Use its 3 robust roles: **B2** (regime filter, halves DD, risk-not-alpha)
· **B3** (Chandelier/ATR trailing exit off the period-high, gated by the regime filter — positive-skew harvesting) ·
**D4** (vol-scaled multi-asset TSMOM ensemble — the only role with real post-cost OOS evidence, MOP 2012; a single-asset
50/200 cross inherits NONE of it). Volume (VWMA) is the only genuinely independent confluence axis.

## The single highest-EV next test (concrete) — the one we botched
`A=daily/4h spot, u50 × B2(200-day regime gate, long only when price>200DMA) + B3(Chandelier ATR trail, 22-bar high,
3.0×ATR, NO window clamp) × C=maker fills (empirical p_fill 0.25–0.40) × D4(vol-scaled equal-risk ensemble)`.
**Honest expectation:** lifts our measured ~0 realized capture to the **TOP of the 25–48% band with better drawdown**
— it does NOT reach 2x/yr.

## The bottom line
2x/yr robust is **not a research result** under LO+spot+lev=1 at daily/4h — it requires **relaxing a constraint**
(the A/B/C fork): leverage/convexity (options VRP), cadence/instrument (sub-hour event-clock cascade entries + maker),
or paid leading data (OI/funding cascade timing). Each costs capital, effort, or tail-risk. The honest, real, robust
deliverable the research supports is the **~25–48%/yr drawdown-managed trend book** — good, but not 2x.
