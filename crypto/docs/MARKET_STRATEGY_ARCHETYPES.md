# Market Strategy Archetypes -- MASTER MAP

> Generated from `src/narrate/strategy_archetypes.py`. The map exists to **select the right MODE for our mandate** and stop trading per-candle. Edit the structured source, not this file.

## Our mandate (the selection constraints)

- **per_move_net_target**: 2-5%+ net
- **hold_band**: hours to <7 days
- **constraints**: LONG-ONLY, SPOT, leverage=1
- **primary_resolution**: 4h / 1d (cost-clearing favors these)
- **unit**: per-setup (multi-candle MOVE), NEVER per-candle
- **exit**: OUT OF SCOPE here -- a separate decomposable domain (trailing/fixed/volatility)
- **objective**: robust held-out compound return (wealth), not Sharpe, not per-bar IC

## Selection verdict

PRIMARY MODE = swing (multi-day MOVE) + breakout as the entry-signal engine; compose intraday-momentum (fast end), position-trend (MAs as a MODE filter, not a per-candle trigger), and event-driven (liquidation-gated entries). AVOID scalping + HFT (per-candle / infra-gated = the trap). Mean-reversion only in confirmed ranges. Funding-carry is the separate beta+yield sleeve.

- **Primary mode:** swing, breakout
- **Composable entry layers:** intraday_momentum, position_trend, event_driven
- **Conditional (regime-gated):** mean_reversion
- **Misfit traps (AVOID):** hft_mm, scalping

## The archetypes (full characteristics)

### Swing trading  --  *PRIMARY*
- **Timescale:** hours-to-days (1-7d)  |  **Unit:** per-setup (multi-candle)  |  **Suitable regime:** trend / transitional
- **Edge source:** Capture a discrete multi-day MOVE/leg between structure points.
- **Entry signal (what we'd hunt):** Pullback-into-trend, breakout-and-hold, support reclaim, momentum-with-confirmation -- a SETUP that forms over several bars and resolves over days.
- **Instruments:** 4h-1d bars, swing MAs, structure (HH/HL), momentum + vol-expansion, positioning.
- **Crypto fit:** Crypto swings are large (5-15%+ legs) and BTC-led; funding/liquidation context filters entries; weekends thin.
- **Failure modes:** Regime flips mid-swing; beta (BTC) overrides the idiosyncratic thesis.
- **Capital profile:** Low-moderate turnover; cost-robust; the sweet spot for taker fees.
- **Our fit:** EXACT match: 2-5%+ net per MOVE, hold hours-to-7d, long-only spot, daily/4h. Our primary mode.

### Breakout / volatility expansion  --  *PRIMARY*
- **Timescale:** hours-days  |  **Unit:** per-setup  |  **Suitable regime:** transitional (coil->expansion)
- **Edge source:** Enter as price exits a coil/range with volume; capture the expansion leg.
- **Entry signal (what we'd hunt):** Range-break + volatility expansion + volume/flow confirmation -- a setup that triggers on the break.
- **Instruments:** Donchian/range, ATR/vol-squeeze, RVOL, OFI confirmation.
- **Crypto fit:** Crypto's vol-clustering makes coils->expansions clean; liquidation cascades often IGNITE the break.
- **Failure modes:** False breakouts (fakeouts) in chop; needs volume/flow confirmation to avoid traps.
- **Capital profile:** Low-moderate turnover; cost-robust.
- **Our fit:** Strong fit + composes with swing: breakout is a primary ENTRY-SIGNAL family for our multi-day moves.

### Intraday momentum / day trading  --  *composable*
- **Timescale:** minutes-hours (flat by ~1 day)  |  **Unit:** per-setup  |  **Suitable regime:** trend / high-vol
- **Edge source:** Ride an intraday impulse/trend within the day; exit before overnight.
- **Entry signal (what we'd hunt):** Intraday breakout / momentum ignition / VWAP reclaim -- a multi-bar setup within a session.
- **Instruments:** 5m-1h bars, intraday MAs, VWAP, RVOL, opening-range.
- **Crypto fit:** Crypto has no 'session', so 'intraday' = a chosen window; funding accrues; BTC-led impulses dominate.
- **Failure modes:** Chop whipsaws; no session structure to anchor; overnight gap risk is absent but vol is continuous.
- **Capital profile:** High turnover; moderate cost sensitivity.
- **Our fit:** Multi-candle setup, but horizon shorter than our hours-to-7d band; viable on the fast end (4h).

### Position / trend-following (HTF)  --  *composable*
- **Timescale:** days-weeks-months  |  **Unit:** per-setup (slow)  |  **Suitable regime:** strong trend
- **Edge source:** Ride large secular trends; let winners run; few large trades.
- **Entry signal (what we'd hunt):** Trend-regime confirmation (MA stack, breakout of higher timeframe range) -- slow MA-based entry.
- **Instruments:** 1d-1w bars, long MAs, Donchian/Turtle breakouts, trend filters.
- **Crypto fit:** Crypto's power-law trends make this lucrative but drawdowns are violent; funding cost over long holds matters.
- **Failure modes:** Whipsaw in ranges; huge give-back at trend ends; long flat periods.
- **Capital profile:** Very low turnover; cost-insensitive; drawdown-tolerant capital needed.
- **Our fit:** Adjacent: hold can exceed 7d, but the ENTRY logic (trend confirmation) overlaps our swing mode. MAs live here, as a MODE filter -- not a per-candle trigger.

### Event-driven (liquidations/unlocks/listings)  --  *composable*
- **Timescale:** minutes-days  |  **Unit:** per-event  |  **Suitable regime:** any (catalyst-gated)
- **Edge source:** Trade discrete catalysts: liquidation cascades, token unlocks, listings, funding resets.
- **Entry signal (what we'd hunt):** A specific EVENT fires (liq cascade > threshold, listing, unlock date) -> conditional entry.
- **Instruments:** liquidation flags, calendar (unlocks), exchange events.
- **Crypto fit:** Crypto-native and rich: liquidation cascades, exchange listings, unlock cliffs, ETF flows.
- **Failure modes:** Events are rare (small n), crowded, and reflexive; hard to backtest with significance.
- **Capital profile:** Bursty turnover; high variance.
- **Our fit:** Composes as an ENTRY-CONDITION layer on swing (e.g. enter the swing only after a capitulation event).

### Mean reversion  --  *conditional*
- **Timescale:** hours-days  |  **Unit:** per-setup  |  **Suitable regime:** range / low-trend
- **Edge source:** Fade statistical extremes back to a mean; sell strength / buy weakness in a range.
- **Entry signal (what we'd hunt):** Stretch beyond bands (z-score/Bollinger), exhaustion, RSI extreme + reversal confirmation.
- **Instruments:** Bollinger/z-score, RSI, VWAP reversion, vol-of-vol.
- **Crypto fit:** Crypto mean-reverts hard intraday but TRENDS on the daily; fading a crypto breakout is dangerous (reflexivity).
- **Failure modes:** Trends destroy it (the gap risk is unbounded long-only); 'catching a falling knife' in cascades.
- **Capital profile:** Moderate turnover; regime-fragile.
- **Our fit:** Conditional fit: works in ranging regimes only. Useful as a SUB-mode the narrator can flag, not the primary.

### Funding/basis carry (delta-neutral)  --  *out of scope*
- **Timescale:** days-weeks  |  **Unit:** continuous (held)  |  **Suitable regime:** any (neutral)
- **Edge source:** Harvest perp funding / futures basis while hedged delta-neutral; market-agnostic yield.
- **Entry signal (what we'd hunt):** Funding/basis exceeds a threshold -> put on the carry; not a directional entry.
- **Instruments:** funding rate, basis, OI -- crypto-native.
- **Crypto fit:** Purely crypto: there is no equity analogue. This is BETA+YIELD, the project's known-robust sleeve.
- **Failure modes:** Funding flips; basis collapse / deleveraging events; exchange/counterparty risk.
- **Capital profile:** Low directional risk; capital-intensive; yield-like.
- **Our fit:** Out of THIS scope (we hunt directional long entries), but it IS the project's known beta+yield sleeve -- note it.

### Statistical arbitrage / relative value  --  *out of scope*
- **Timescale:** hours-days  |  **Unit:** per-setup (cross-sectional)  |  **Suitable regime:** any (market-neutral)
- **Edge source:** Exploit cross-sectional mispricings: pairs, baskets, lead-lag, cross-exchange.
- **Entry signal (what we'd hunt):** Cross-sectional z-score / rank divergence / lead-lag signal -> long-short or relative entry.
- **Instruments:** cross-asset rank (xrel_*), pairs spreads, lead-lag.
- **Crypto fit:** Crypto lead-lag (BTC->alts) is real but strongest intraday; cross-exchange + pairs are crowded.
- **Failure modes:** Crowded; relationships break; daily lead-lag is weak (we tested: null held-out).
- **Capital profile:** Moderate turnover; needs short leg (we are long-only -> limited).
- **Our fit:** Limited by long-only-spot; cross-sectional RANK is still a useful narrator CONTEXT, not a standalone strategy here.

### HFT / market-making  --  *AVOID (trap)*
- **Timescale:** microseconds-seconds  |  **Unit:** continuous  |  **Suitable regime:** any (thrives on volume)
- **Edge source:** Latency + queue priority + spread capture + inventory management; rebate harvesting.
- **Entry signal (what we'd hunt):** Continuous quoting; entry is a resting order filled by adverse flow, not a 'signal'.
- **Instruments:** L2 book dynamics, queue position, microprice, OFI.
- **Crypto fit:** Crypto's fragmented venues + maker rebates make MM viable but co-location/infra-gated; toxic flow (VPIN) is the enemy.
- **Failure modes:** Adverse selection, latency arms race, inventory blowups in cascades.
- **Capital profile:** Massive turnover, infra-bound, cost edge IS the strategy.
- **Our fit:** Infra/latency-gated; not a signal-hunt; impossible without co-location. Out of scope.

### Scalping  --  *AVOID (trap)*
- **Timescale:** seconds-minutes (1-a few bars)  |  **Unit:** per-candle  |  **Suitable regime:** range / high-liquidity
- **Edge source:** Tiny repeated edges on micro-moves; capture the next 1-2 bars repeatedly.
- **Entry signal (what we'd hunt):** Micro-pullback / order-flow flip / 1-2 bar reversal -- a per-CANDLE trigger.
- **Instruments:** 1m/tick bars, order flow, micro-MAs, VWAP bands.
- **Crypto fit:** Crypto 24/7 + thin books make scalping costly (taker fees + slippage eat the tiny edge); funding bleeds positions.
- **Failure modes:** Costs/slippage dominate the tiny edge; per-candle noise-fitting; over-trading.
- **Capital profile:** Very high turnover; cost-fragile; needs maker fills.
- **Our fit:** THE TRAP we fell into (scalp oracle = 2-bar wiggles). Per-candle, cost-fragile, mismatched to 2-5%/move.
