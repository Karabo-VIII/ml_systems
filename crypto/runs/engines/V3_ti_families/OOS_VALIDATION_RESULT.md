# V3 TI Families — OOS VALIDATION RESULT (2026-06-24)

Held-out window: **2024-05-15 → 2025-01-01** (7.5 months, never used in config selection).
Configs are frozen hand-chosen grids. Reporting LINEAR/wk (the honest read) + edge vs always-long.

## The verdict: the DEV numbers did NOT hold out of sample

The "too good to be true" 5-digit DEV compounds were inflated artifacts (EW
rebalancing premium ~9.5x + multi-year compounding + regime gate). Out of sample,
the non-MA TI families are **inconsistent and mostly fail to beat always-long.**

### Edge vs always-long (linear %/wk, combined regime-gated engine)

| Family    | 4h      | 2h      | 1h      | 1d      | 30m     | Beats AL? |
|-----------|---------|---------|---------|---------|---------|-----------|
| RSI       | **+0.42** | -1.30 | -1.11 | -0.59 | -0.67 | 1/5 TFs (4h only) |
| MACD      | -0.49   | -0.16   | **+1.47** | -0.19 | -2.79 | 1/5 TFs (1h only) |
| Bollinger | -1.51   | -0.99   | -0.56   | **+0.99** | -2.90 | 1/5 TFs (1d only) |

Each family beats always-long at exactly ONE scattered timeframe — and a different
one each time (RSI@4h, MACD@1h, Bollinger@1d). There is **no coherent sweet spot**
like the MA family's clean 4h dominance. This pattern (one isolated win, rest losses)
is exactly what you'd expect from noise, not a real edge.

### Single-asset BTC (no rebalancing premium — the honest per-asset number)

In almost every cell, the engine UNDERPERFORMS BTC buy-and-hold over OOS:
- RSI 4h: engine +43% vs BH +51%
- MACD 4h: engine +24% vs BH +51%
- Bollinger 4h: engine +52% vs BH +51% (≈tie — the only non-loss at 4h)

The signals spend ~50-70% of the window in cash and miss upside.

## Contrast with the MA family (the gold standard)
- MA OOS: +193.5% compound, **+4.11%/wk**, consistent 4h sweet spot, 59% slices profitable
- MA passed random-null at the 100th percentile (genuine timing, not just the gate)
- Non-MA families: scattered ~+0.4 to +1.5%/wk edges at single TFs, mostly negative elsewhere

## Conclusion / recommendation
1. **MACD@1h** (+1.47pp edge) and **RSI@4h** (+0.42pp) are the only non-MA cells worth a
   second look, but neither is convincing enough to deploy on this evidence alone.
2. Bollinger@1d's +0.99pp is on very few slices (low statistical weight).
3. **The MA family remains the only validated, deployable engine set.**
4. Do NOT deploy RSI/MACD/Bollinger as standalone engines. If used at all, they belong
   only as minority confirmation features inside an MA-anchored ensemble — and even that
   needs its own validation.

This is the conservative, honest read the user asked for. The numbers were indeed
too good to be true; OOS exposes them.


---

# V1.1 Families 6-8 (Supertrend / Funding / OI) — OOS RESULT (2026-06-24)

Items 2-3 done: signal functions vectorized; **Funding/OI now use the REAL chimera
features** (`norm_funding`, `norm_oi_price_divergence`) instead of price proxies.
Re-ran the same OOS gauntlet. Results:

### Edge vs always-long (combined regime-gated, linear %/wk)

| Family       | 4h      | 1h      | 1d      | 30m     | Deploy | Beats AL? |
|--------------|---------|---------|---------|---------|--------|-----------|
| Supertrend   | **+0.76** | +0.17 | **+0.41** | -3.42 | ~98%   | 3/4 TFs but |
| Funding(real)| -2.03   | -0.55   | -0.94   | -0.20   | ~99%   | 0/4 TFs   |
| OI-div(real) | -2.28   | -2.34   | **+0.30** | -4.85 | ~99%   | 1/4 TFs   |

### The decisive finding: deploy rate ~98-100%
Supertrend/Funding/OI all deploy **98-100% of the time**. With the regime gate on,
they are effectively **always-long-when-bull with a tiny tilt** — they are NOT
selective engines. Supertrend's +0.76pp at 4h is just the small residual of a
near-always-long book, not a timing edge. The real chimera Funding/OI features did
NOT rescue the families: real-feature Funding still loses to AL at every TF.

### Funding momentum note
`norm_funding_momentum` (dead at the panel level, D-listed) DOES carry per-asset data
(range [-1.30, 1.11] on BTC). It's dead as a *cross-sectional selector*, not per-asset.
Not pursued further given the family-level failure.

## FINAL VERDICT ACROSS ALL 8 NON-MA FAMILIES
None of RSI / MACD / Bollinger / Stochastic / ADX / Supertrend / Funding / OI-divergence
shows a coherent, cross-timeframe edge that survives OOS. Each produces at most one
isolated positive-TF cell (consistent with noise / always-long residual). **The MA
family (+4.11%/wk OOS, 100th-pct random-null, clean 4h sweet spot) remains the ONLY
validated deployable engine set.** Non-MA TIs should not be deployed standalone.

This is the conservative answer the user asked for after flagging the numbers twice.
The headline DEV compounds were inflated; OOS exposes them.
