<!-- Research-grounded decomposition: 12 clusters, 108+ approaches, 14 agents (web-grounded). /orc ultracode workflow 2026-06-10. -->

I'll synthesize this into a clean, exhaustive decomposition. This is a pure synthesis task over the provided 133-approach set, so I'll write the deliverable directly.

# How People Actually Trade Moving Averages — The Definitive Decomposition

A moving average is just a smoothing operator. Every real "MA strategy" is a **point in a 4-dimensional choice space**: *what signal mode you read off the line, what role the line plays, how the line is constructed, and how you compose it with everything else.* Naming a strategy ("golden cross", "20-EMA pullback") is just naming a specific corner of that cube. The 133 approaches collapse to **a few dozen genuinely distinct mechanics dressed in ~100 names** — most of the apparent variety is type-shopping and period-shopping on the same axis.

---

## PART 1 — The Orthogonal Decomposition Axes (the real knobs)

### Axis A — SIGNAL MODE (how the line produces a decision)

There are only **five** primitive ways to extract a decision from an MA. Everything else is a combination.

| Mode | The read | The premise | Fires best in | Failure mode |
|------|----------|-------------|---------------|--------------|
| **A1. Cross** | Price crosses the MA, or fast MA crosses slow MA (incl. MACD/PPO = MA-difference cross) | A crossing = momentum regime change | Trend onset | Whipsaw in range (the line is sliced repeatedly); lags the turn by ~N/2 bars |
| **A2. Slope / direction** | First derivative of the MA (rising/falling, steepness, curvature roll-over) | A rising MA = trend in force; flat = equilibrium | Trend (as a filter) | Slope is a smoothed derivative of a smoothed price → confirms late, flips in chop |
| **A3. Distance / extension (mean-reversion)** | Quantified gap of price from the MA: %, k·σ (Bollinger), k·ATR (Keltner/STARC), z-score, %B, price/MA ratio (Mayer) | The MA is "fair value"; a stretch is a temporary dislocation that reverts | Range / stationary spread | Catastrophic in a trend — price "hugs the band" and the fader is run over |
| **A4. Dynamic S/R (pullback / bounce / retest)** | Price returns *to* the MA and reacts (bounce, rejection, break-and-retest polarity flip) | Order/cost-basis clustering + reflexive crowd-watching make the line a moving level | Established trend (pullback phase) | "Support" is descriptive not causal; in fast moves price slices through; selection-biased in hindsight |
| **A5. Ribbon alignment** | The *order & separation* of many stacked MAs (stack order, fan width, compression/expansion) | Agreement across horizons = conviction; compression = coiled energy | Trend (as a regime read) | Dominated by the two extreme lines → reduces to a 2-MA cross "with extra ink"; tangles in chop |

**Key insight:** A3 (extension) and A1/A2/A4 (trend) are *opposite trades*. The single most important hidden variable in any MA strategy is **which regime you assume**, because the same band touch is a "buy the dislocation" (range) or "you're about to be steamrolled" (trend).

### Axis B — ROLE (what job the line does in the system)

The same MA, same period, can play any of these. The role determines whether lag *helps* or *hurts*.

| Role | What it decides | Does lag help? | Evidence quality |
|------|-----------------|----------------|------------------|
| **B1. Entry-trigger** | WHEN to get in | Hurts (enter late, near mid-move) | Weak standalone |
| **B2. Trend-filter / regime gate** | WHETHER signals are allowed (risk-on/off) | Neutral-to-helps (ignores noise) | **Strongest** — the one robust use |
| **B3. Exit / trailing-stop** | WHEN to get out | **Helps** (ignores noise, rides trend) | Robust risk-discipline (not alpha) |
| **B4. S/R level** | WHERE to enter/place stops | Helps (tight stops, better R:R) | Discretionary, partly self-fulfilling |
| **B5. Position-sizing** | HOW BIG (conviction = ribbon width / slope / vol-scaling / Mayer ratio) | Neutral | Sound as risk mgmt; generates no timing edge |

**The single most important realization in the entire dataset:** *the lag that makes an MA a bad entry-trigger (B1) makes it a good exit/trail (B3) and a good regime filter (B2).* The robust uses of MAs are all "slow" roles.

### Axis C — MA CONSTRUCTION (the operator itself)

Three sub-knobs, in descending order of how much they actually matter:

- **C1. Period** (dominant knob): short = responsive/noisy, long = smooth/laggy. Clusters: **9/10, 20/21, 50, 100, 200** (daily); **20W/21W, 200W** (weekly crypto); Fibonacci **8/13/21/34/55/89**; round numbers **50/100/200**.
- **C2. Type / weighting kernel** (second-order — see Part 3): the lag-vs-noise Pareto frontier. Every operator is one point on it:
  - **Plain:** SMA (laggiest/smoothest) → EMA (recency-weighted default) → WMA/LWMA → SMMA/RMA/Wilder's (the invisible engine inside RSI/ATR/ADX).
  - **Volume-aware:** VWMA (rolling, volume-weighted price), MA-of-volume (participation baseline).
  - **Lag-reduced (overshoot in chop):** DEMA, TEMA, T3, HMA, ZLEMA, "zero-lag" family.
  - **Regime-adaptive (vary the smoothing constant):** KAMA (efficiency ratio), VIDYA (momentum/vol), FRAMA (fractal dimension), MAMA/FAMA (Hilbert phase), McGinley Dynamic, JMA.
  - **Period-adaptive (vary the period):** Ehlers dominant-cycle MAs (distinct from KAMA/VIDYA).
  - **Alternative kernels:** ALMA (Gaussian-offset), triangular, median (outlier-robust, non-linear), sine-weighted.
- **C3. Input & anchoring** (an underused axis that materially changes signals):
  - **Centerline source:** close vs HL2/HLC3 (Ichimoku midpoints, pivots, Heikin-Ashi averaged bars).
  - **Anchoring:** rolling vs **session-anchored (VWAP)** vs **event-anchored (AVWAP** from swing low / ATH / halving).
  - **Displacement:** shift the line ±k bars (Ichimoku Senkou, Alligator, DPO) to cut premature crosses.
  - **Input transform:** raw price vs **log price** vs **fractionally-differenced** price (AFML) — changes every downstream signal, especially in crypto's multiplicative moves.

### Axis D — COMPOSITION (how it's wired to the rest)

| Mode | What it adds | Honest note |
|------|--------------|-------------|
| **D1. Standalone** | Simplicity | Rarely survives cost + out-of-sample |
| **D2. Confluence-with-X** | Orthogonal info (X = RSI, MACD, volume, candlestick, Fib, structure) | **Volume is the only genuinely independent axis** (non-price-derived); stacking more *price-derived* oscillators (MACD+EMA) is correlated redundancy disguised as "two confirmations." Each added condition = another free parameter = overfit surface |
| **D3. Multi-timeframe (HTF filter + LTF trigger)** | HTF MA gates direction, LTF MA times entry | **The most defensible composition** — removes counter-trend chop. But it's risk-reduction, not new alpha; doubles lag; HTF-bar-before-close is a classic look-ahead leak |
| **D4. Cross-sectional / portfolio** | MA as a *factor* across a universe: rank by distance-above-own-MA, % of names above 200DMA (breadth), TSMOM ensemble | This is where the *real* academic edge lives (diversification is the antidote to single-rule data-snooping) |

> **The thesis, made concrete:** *Every* MA strategy = (one A) × (one+ B) × (one C-triple) × (one D). "Golden cross" = **A1 cross × B2/B1 × {SMA, 50/200, close} × D1**. "20-EMA pullback" = **A4 dynamic-S/R × B1 × {EMA, 20, close} × D2 (candle)**. "Faber GTAA" = **A1/A2 × B2 × {SMA, 10-month, monthly close} × D4 (asset basket)**. There is no other content.

---

## PART 2 — Canonical Named Approaches Mapped to the Axes

| # | Named approach | A (mode) | B (role) | C (type / period / anchor) | D (compose) | Best regime | Honest evidence |
|---|----------------|----------|----------|----------------------------|-------------|-------------|-----------------|
| 1 | **Price vs 200-day** | A1/A2 | B2 filter | SMA / 200 / rolling | D1/D3 | Trend | **Robust risk-reducer**, not alpha. Cuts DD/vol; *underperforms* B&H on raw return in bulls. Faber GTAA core |
| 2 | **Golden / Death Cross (50/200)** | A1 cross | B2/B1 | SMA / 50,200 | D1 | Trend | Famous, **heavily lagging** (~33 signals/66yr, ~350-day hold). Halves max DD; lags B&H return. Whipsaws in ranges (2015-16, mid-2023). Media event > edge |
| 3 | **9/21 EMA cross** | A1 cross | B1 | EMA / 9,21 / LTF | D2+D3 | Trend | Crypto day-trader default. **Most false signals in chop**; fee/funding-sensitive sub-1h; needs HTF filter |
| 4 | **12/26 EMA (= MACD)** | A1 cross | B1/B2 | EMA / 12,26,9 | D2 | Trend | IS MACD. Same lag/whipsaw, milder. Histogram = earlier+noisier. No standalone alpha |
| 5 | **20/50 EMA** | A1 cross | B1/B2 | EMA / 20,50 | D1/D2 | Trend | Swing workhorse; fewer whipsaws, more lag. Period/market-dependent |
| 6 | **Triple-MA (5/10/20, 8/21/55)** | A1+A5 | B1+B2 | EMA / stacked | D1 | Trend | 3rd MA = built-in filter → fewer false signals, **more lag, fewer/later trades**. Discipline tool |
| 7 | **GMMA / Guppy ribbon** | A5 align | B2+B1 | EMA / 3-15 & 30-60 | D1/D2 | Trend | ~competitive with B&H after lag; **6 lines per group are near-collinear** (illusory confirmation). Long-fan = the real read |
| 8 | **EMA ribbon (crypto)** | A5 align | B2 | EMA / 10-60 sets | D1/D3 | Trend | Clean *regime filter*; as a trigger reduces to a 2-MA cross. Serial whipsaw in crypto chop |
| 9 | **Ribbon squeeze / compression** | A5 (width) | B1 | EMA dense | D2 (vol) | Range→trend | Compressions DO precede expansions, but **direction is a coin-flip** — edge is in the volume/ATR filter, not the ribbon |
| 10 | **20/21-EMA pullback ("buy the dip")** | A4 S/R | B1 | EMA / 9,20,21 | D2 (candle) | Trend | **Single most popular discretionary entry.** High win-rate *inside a confirmed trend*; "falling knife" = #1 retail loss; heavy per-asset curve-fit |
| 11 | **50-EMA/SMA bounce** | A4 S/R | B1/B4 | 50 / EMA or SMA | D2 (Fib) | Trend | Deeper pullback filters noise; the break lags. Most-watched 4h/daily crypto line |
| 12 | **Break-and-retest (polarity flip)** | A4 (flip) | B1 | 50/200 | D2/D3 | Trend transition | Catches regime changes a bounce misses; discretionary, fakeout/stop-run prone at the level |
| 13 | **Bollinger band-touch fade** | A3 extension | B1+B4 | SMA+2σ / 20 | D2 (RSI+ADX) | **Range** | Works in range (PF ~1.6 ADX<20), **blows up in trend (PF ~-0.7)**. Fix = close-back-inside + ADX gate. σ assumes normality (crypto fat tails) |
| 14 | **%B oscillator** | A3 | B1+B5 | 20,2 | D2 | Range | Quant-friendly normalization; **does NOT fix trend-fade** (can pin >1 through a whole uptrend) |
| 15 | **Bollinger squeeze / TTM** | A3→break | B2 detector | BB inside Keltner | D2 (momentum) | Range→trend | Flags low-vol reliably; **does NOT predict direction**. The "don't fade now" off-switch |
| 16 | **Keltner channel** | A3 / A4 | B1/B4/B3 | EMA+ATR / 20 | D2 (slope) | **Trend** (pullback) | ATR bands smoother/laggier than σ. Best use = trend-pullback-to-EMA, NOT outer-band fade |
| 17 | **z-score / extension reversion** | A3 | B1+B5 | anchor MA + rolling σ | D2 (Hurst gate) | Stationary spread | **Most defensible reversion form IF stationarity tested** (Hurst<0.5, ADF, half-life). Engine of pairs/stat-arb |
| 18 | **Connors RSI(2) < MA200** | A3 (gated) | B2+B1 | SMA200 filter + RSI2 | D2 | Trend-pullback | **Best-documented short-term reversion** *because* the 200-MA stops it fighting the trend. Crowded/decayed; stop-less = fat left tail; weak crypto transfer |
| 19 | **KAMA / AMA** | A1/A2 | B2 | adaptive (ER 10, 2/30) | D1/D2 | All (self-tuning) | **Most principled whipsaw fix** — moves along the frontier with regime. Lags reversals after a trend; 3 params = mining surface; modest standalone edge |
| 20 | **HMA** | A2 slope | B1/B3 | WMA-stack / 9,16,21,55 | D3 (200 gate) | Trend | Genuinely smooth-for-its-lag; **overshoots & whipsaws in chop** (lag-reduced ≠ noise-adaptive). Needs regime gate |
| 21 | **DEMA / TEMA / T3** | A1/A2 | B1/B2 | recombined EMAs | D2 | Trend | Each lag-reduction step buys earlier signals, pays in whipsaw. **TEMA crosses price far more often** = near-untradeable in chop standalone |
| 22 | **ZLEMA / zero-lag** | A1/A2 | B1 | de-lagged EMA | D3 | Trend | Lowest lag = **maximum overshoot/noise**. "Zero-lag" oversells; lag traded for noise, not free |
| 23 | **ALMA** | A1/A2 | B1/B2 | Gaussian (offset.85, σ6) | D1/D2 | All | Leads EMA, fewer whips than HMA; **NOT vol-adaptive** (fixed knobs). 3-param overfit surface |
| 24 | **VWMA** | A1/A2 | B2 confirm | volume-weighted / 20,50 | D2 (volume) | Trend (confirmed) | Adds **orthogonal participation axis**; best use = divergence. Crypto volume dirty (wash trading, fragmentation) |
| 25 | **VWAP (session)** | A4/A2 | B2/B4 | session-anchored | D1 | Intraday all | Institutional execution benchmark, partly **self-fulfilling**. Distinct from VWMA (cumulative-from-anchor) |
| 26 | **Anchored VWAP** | A4 S/R | B4 | event-anchored (low/ATH/halving) | D1/D2 | All | Cost-basis proxy from a meaningful event; modern, increasingly popular (Shannon) |
| 27 | **Ichimoku TK/Kijun + cloud** | A1 cross | B1+B2+B4 | **HL2 midpoint** / 9,26,52 | D2 | Trend | Midpoint (not close) construction; cloud=filter, Kijun=dynamic S/R. Same lag; heavily param-mined |
| 28 | **Bill Williams Alligator** | A5 align | B2+B1 | displaced SMMA / 13,8,5 | D1 | Trend | Displacement + median-price + SMMA. "Sleeping vs eating" = ribbon compression renamed |
| 29 | **SuperTrend / Hull-suite flip** | A1 flip | **B1+B3** (always-in) | ATR-band HL2 / HMA | D1 | Trend | **Whipsaw machine in range** (flips & bleeds). (period,mult) heavily curve-fit. Crypto-TV staple |
| 30 | **Chandelier exit** | — | **B3** | ATR off period-high / 22,3.0 | D1 | Trend | Best-engineered trail (keyed to extreme, not average). Multiplier (3→5 on vol) = mining knob |
| 31 | **Parabolic SAR** | A1 flip | B3 (stop-rev) | accelerating extreme / 0.02-0.20 | D1 | Trend | Tight accelerating trail; late on crashes, flips constantly in range |
| 32 | **MA-crossback exit (close below 20/50)** | A1 | **B3** | EMA/SMA 10/20/50 | D1 | Trend | **The default retail exit.** Real discipline; lagging give-back; range whipsaw. Needs ADX>25 |
| 33 | **MA ratchet trail / scale-out** | A4/A3 | B3+B5 | EMA 20/21 | D3 | Trend | Locks gains, never loosens; trim into extension above the 21, reload on retest. Discretionary |
| 34 | **BMSB (20W SMA + 21W EMA)** | A4 S/R | B2 regime | weekly / 20W,21W | D3 | Macro bull | **Iconic crypto band.** Respected 2017/21 support; lagging weekly; 20-vs-21 cosmetic; **n≈3-4 cycles = overfit to BTC** |
| 35 | **200-week MA** | A4 floor | B4+B5 | SMA / 200W | D2 (on-chain) | Bear-bottom | Strong historical bottom hit-rate; **2022 broke the clean pattern**; tiny sample; BTC-only |
| 36 | **On-chain cost-basis MAs (Realized Price, MVRV, Mayer)** | A3 ratio | B4+B5+B2 | economic averages | D2 | Macro extremes | **Real economic mechanism** (capitulation) > pure pattern; thresholds (MVRV 3.5, Mayer 2.4) drift/curve-fit; BTC-centric |
| 37 | **Mayer Multiple (price/200DMA ratio)** | A3 ratio | B5 sizing | price÷SMA200 | D1 | Macro | Ratio (not band) for accumulation/distribution; dimensionless OB/OS |
| 38 | **Faber GTAA (10-month SMA)** | A1/A2 | B2 allocation | SMA / 10-month / monthly | D4 basket | Trend | Canonical academic TAA timing; **crash-avoidance**, similar return ~half DD |
| 39 | **TSMOM / managed-futures trend** | A1/A2 (ensemble) | **B1+B5** | EMA-crossover ensemble + vol-scale | **D4** multi-asset | Trend (crisis-alpha) | **Strongest evidence in the family** (MOP 2012, 58 futures, survives costs *via diversification*). A single 50/200 cross is NOT TSMOM and inherits none of its robustness. Post-2012 decay debate |
| 40 | **Cross-sectional MA-momentum rank** | A3 (rel.) | B1 factor | per-name MA, ranked | D4 | Dispersion | Core quant cross-sectional momentum; the listed TSMOM's counterpart |
| 41 | **% above 200DMA breadth** | A2 (aggregate) | B2 internals | SMA per-name → % | D4 breadth | All | Market-internals confirmation; bearish breadth divergence as risk-off |
| 42 | **MACD histogram + signal + divergence** | A1 + divergence | B1+momentum | EMA-of-EMA-diff / 12,26,9 | D2 | Trend / range | MA-of-MA-difference. Divergence **unreliable in strong trends**; no edge over plain 12/26 cross |
| 43 | **MA-of-indicator (%D, MA-of-RSI, MA-of-volume)** | A1 cross | B1 confirm | SMA/EMA on derived series | D2 | All | Universal (%D = SMA of %K; "volume > its average"). Entire missing branch — MA applied off-price |
| 44 | **AO / Coppock / TRIX / DPO** | A1/A3 | B1 | SMA/WMA spreads, triple-smooth ROC, displaced detrend | D1 | Trend/cycle | AO huge in crypto; Coppock famous for index bottoms; DPO/TRIX = distinct MA-derived oscillators |
| 45 | **N-bar / %-band confirmation filter** | wrapper | B1 refine | any pair + filter (BLL 1% band) | D1 | All | The **standard whipsaw-mitigation wrapper**; the academic BLL band-filter version |
| 46 | **Adaptive-period (Ehlers cycle-tuned)** | A1 | B1/B2 | period set to dominant cycle | D1 | All | Adapts *period* (not alpha) — distinct from KAMA/VIDYA; DSP niche |
| 47 | **McGinley Dynamic** | A1 | B2/B4 | self-adjusting tracking line / 10-20 | D1 | All | Auto-speeds in down-moves; named adaptive line distinct from KAMA |
| 48 | **Heikin-Ashi color-flip** | A1/A2 | B1+B3 | recursive 2-bar OHLC average | D1 | Trend | MA baked into bar construction; color flip = trend/exit. Very high crypto use |
| 49 | **MA on log / frac-diff input** | any | B2 | any MA, transformed input | D1 | All | The *input* axis (raw vs log vs frac-diff, AFML) materially changes crypto signals — underused |

---

## PART 3 — What the Research Actually Concludes

**The honest verdict, distilled across every source:**

### Has evidence (use these)
1. **MA as a trend-FILTER / regime gate (B2).** Being above/below the 200-day (or Faber's 10-month SMA) materially shifts the forward return/vol distribution and **cuts drawdown ~half**. Robust to the exact period (50 vs 100 vs 200 all behave similarly) — that *period-insensitivity is the signature of a structural effect, not an over-fit*. It is **risk-reduction / beta-control, not alpha**: it underperforms buy-and-hold on raw return in sustained bulls.
2. **MA as an EXIT / trailing stop (B3).** Converts the family's trend-persistence into a disciplined "let winners run." The lag that kills entries *helps* here. Robust risk practice; not a return source.
3. **Time-Series Momentum (TSMOM, MOP 2012).** The academically respectable generalization — **the only member with strong post-cost OOS evidence**, and it earns it specifically through **diversification across ~58 instruments + vol-scaling**, not through any single rule. Best performance in market extremes (crisis-alpha / convex payoff). *Crucial caveat: a single-asset 50/200 cross is NOT TSMOM and inherits none of its robustness.*
4. **On-chain cost-basis MAs (Realized Price, capitulation bands).** Have a *real economic mechanism* (seller exhaustion) rather than pure pattern-fitting — their edge over price-only MAs. Still tiny-sample (3-4 BTC cycles), BTC-centric, zone-not-signal.

### Folklore (no robust standalone edge)
1. **Precise crossovers as standalone triggers.** Brock-Lakonishok-LeBaron (1992) found MA rules profitable on the DJIA 1897-1986 — but **Sullivan-Timmermann-White (1999, White's Reality Check bootstrap) showed that's largely data-snooping once the full universe of rules + costs is priced in, and fresh OOS (1987-2011) found no predictive power.** Crossovers lag the turn and whipsaw in ranges.
2. **The "best" exact period / type.** SMA-vs-EMA-vs-WMA-vs-DEMA-vs-HMA is **second-order**: period and role dominate type. Type-shopping across ~15 near-identical operators is a **classic data-mining surface** (many shots at a spurious in-sample win). Fibonacci periods (21,55) show **no evidence of beating neighbors** (20,50) beyond chance + mild self-fulfillment.
3. **Mean-reversion-TO-the-MA (band fades) without a regime gate.** Works in ranges, *catastrophic in trends* — "price hugs the band." The fix (close-back-inside + ADX/Hurst gate) is mandatory, and the edge then lives in the *gate*, not the band.
4. **Lag-reduced MAs as a free lunch.** DEMA/TEMA/HMA/ZLEMA reduce lag *by overshooting* — they trade one failure mode (late) for another (whipsaw harder in chop). "Zero-lag" is a misnomer.
5. **Multi-indicator confluence as "independent confirmation."** MACD + EMA + RSI are all lagging price transforms → **correlated, not independent**. More conditions = fewer, later signals + a larger overfit surface. *Volume is the only genuinely orthogonal confirmer* (and crypto volume is dirty).

### Universal caveats (apply to ALL of the above)
- **Lag** ≈ N/2 bars — you enter after the bottom, exit after the top.
- **Whipsaw in ranges** — false-signal rates of **57-76%** in some S&P configs; the family bleeds in chop and shines only in sustained trends.
- **Cost cliff** — fast/low-TF systems are killed by fees + funding + slippage (esp. crypto perps); the whole edge can be a fixed-backtest artifact.
- **Parameter data-snooping** — every period, type, threshold, band-width, and multiplier is a free knob; the more knobs, the easier the in-sample win and the worse the OOS result. **Reflexivity** (the 200-day is "real" partly because >70% of institutional PMs watch it) is the only thing giving specific levels extra realness — and it's a small effect.

---

## PART 4 — How People MOSTLY Trade MAs (naive approaches, ranked by real-world prevalence)

This is the direct answer to *"how do people actually trade moving averages?"* — ranked from most to least common in actual retail/crypto practice:

1. **"Price above/below the 200-day" as a bull/bear switch.** The single most-used technical concept anywhere. Risk-on above, defensive below. (A1/A2 × B2)
2. **"Buy the dip to the 20/21/50 EMA" in an uptrend.** The dominant *discretionary* entry across stocks, FX, and crypto. The "21-EMA bounce" is near-universal in crypto. (A4 × B1)
3. **The Golden / Death Cross (50/200).** The most *famous* setup — driven as much by media headlines as by trading. (A1 × B2/B1)
4. **Fast EMA crossover (9/21) on lower timeframes.** The default crypto day-trader / scalper trigger. (A1 × B1)
5. **"Exit / trail with the close below the 20 or 50."** The default trend-trade exit and stop discipline. (A1 × B3)
6. **EMA ribbon / stack as an at-a-glance trend read** (incl. exchange-default 7/25/99, GMMA, 5-8-13 scalp). (A5 × B2)
7. **Bollinger Band touch-fade** (the default mean-reversion tool — and the default way to blow up by fading a trend). (A3 × B1)
8. **MACD line/signal cross** (a disguised 12/26 EMA cross that ships by default on every platform). (A1 × B1)
9. **MA + RSI / MACD / volume "confluence" stacks** ("it feels rigorous"). (any × D2)
10. **Multi-timeframe MA bias** (HTF 200 gates LTF entries) — the "pro" framing the disciplined minority graduate to. (any × D3)
11. **Crypto-macro lines: BMSB (20W/21W) and the 200-week MA** as cycle regime/accumulation gauges. (A4 × B2/B4)
12. **SuperTrend flip** as an always-in directional system (crypto TradingView culture). (A1 × B1+B3)

> **Bottom line:** Most people trade MAs as **A1 (crosses)** and **A4 (pullbacks)** in the **B1 (entry-trigger)** role, standalone or with correlated confluence — which is precisely the **weakest, most data-snooped corner of the choice space.** The evidence says the value is in the *other* corners: MAs as a **B2 regime filter** and **B3 exit/trail**, and — at the portfolio level — as a **D4 vol-scaled, multi-asset TSMOM ensemble.** The robust uses are all the "slow, boring, filter/risk" uses; the popular uses are all the "fast, exciting, predict-the-turn" uses. That inversion is the whole story.