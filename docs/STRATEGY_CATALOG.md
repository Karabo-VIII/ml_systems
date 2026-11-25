# Strategy catalog: crypto and FX

Every strategy family this project knows about, in one list: what the rule is, which market it fits, whether it needs a short side, whether code for it exists anywhere, and what the earlier projects measured.

**This is a fresh start.** The "Prior" column reports what older code measured on older data builds. Those numbers were not re-run on this pipeline. They decide the order of work. They do not decide what is true: a row is believed or dropped only after it has been run here, on gold from `pipeline/`, through `strategy/engine.py`.

Written 2026-11-25. Companion to `docs/STRATEGY_LAYER.md` (the engine and the first three strategies).

**Status: inventory only.** Nothing in this catalog has been tested on the new pipeline, and nothing will be until the methodology in section 6 is agreed.

## 0. The whole table at a glance, by layer

A strategy is built in layers. Each layer adds one kind of thing on top of the layer below, and is judged against the layer below.

| Layer | What it is | What is on the table | Where in this catalog |
|---|---|---|---|
| **0. Baselines** | No skill at all. The floor everything is compared with. | Cash; buy and hold; equal-weight basket rebalanced monthly; fixed BTC/cash mix; random entries; a day-of-week rule kept as a known-useless control | PM-01, PM-02, SE-01 |
| **1. Static: naive indicator rules** | One technical indicator, one fixed setting, trading its own rule with no filter. | Crypto: 63 rule families with 111 textbook settings. FX: 38 rules (the same families made long-short). | Sections A, B, C, H |
| **2. Static: rule plus fixed improvements** | The same rule with a fixed, non-adaptive change to how it holds or exits. | Minimum hold (12 bars); 10% trailing stop; 6-bar cooldown after an exit; 3-bar confirmation before entry; own 200-bar trend filter; limit-order entry; exit types; sizing types; top-K or strength-weighted selection; fixed drawdown stop or volatility cap | PM-05, PM-06, PM-07, XS-01, XS-04, VO-01, VO-04 |
| **3. Static: non-indicator books** | Fixed rules that are not a chart indicator. | Grids; DCA and rebalancing; cross-sectional ranking (momentum, reversal, value); carry; FX style blend; pairs | MR-11 to MR-13, sections D and E, PM-02 to PM-04 |
| **4. Dynamic** | Hand-written logic that changes exposure or the choice of rule as market conditions change. No learning. | On/off gates, one per kind of information: BTC regime, breadth, volatility regime, statistical regime, order-flow size, funding, macro and sentiment, calendar, FX session, FX risk-off. Graded (partial) versions of the gates. Routing between rule families or books. Core plus satellite. | Section G, XS-11, XS-12, SE-02 |
| **5. ML on top of rules** | A trained model that adjusts a rule from layers 1 to 4. The rule still decides the basics. | Take-or-skip filter on a rule's trades; reweighting the coins a rule holds; a learned dial for how strong that reweighting is; learned routing between books; learned parameter choice; learned regime label | ML-01, ML-03, ML-04, ML-05, RG-09 |
| **6. ML alone** | A model that decides positions with no rule underneath. | Return prediction (ridge, boosted trees); sequence and world models; reinforcement learning; formula search | ML-02, ML-06, ML-07 |
| **7. Execution** | How an order is placed. Applies to every layer. | Market order at the next bar; limit order; scheduled slicing | PM-07 |

Every cell in that table is then multiplied by four axes:

| Axis | Options on the table |
|---|---|
| Instrument | Crypto: 10, 50 or 100 coins. FX: 7 majors, then crosses and metals. |
| Timeframe | 1m, 5m, 30m, 1h, 2h, 4h, 6h, 8h, 1d (all built by the pipeline), plus dollar bars on crypto |
| Side | Long or cash; long-short; market neutral |
| Parameters | One textbook setting; a small set of clearly different settings; a dense grid |

Inventory counts from the old registries, for scale:

| Registry | Unit | Count |
|---|---|---|
| Indicators | A measurement (RSI, ATR, ...) | 96: 34 had a trading rule, 26 were proposed and 25 of those were later coded, 23 are regime or sizing reads and not entry rules, 13 were excluded (for example indicators that redraw history) |
| Strategy rules (crypto) | A tradeable rule built on an indicator | 63 rules, 111 textbook settings |
| Strategy rules (FX) | The same, long-short | 38 indicator rules plus 5 portfolio rules |
| Distinct settings | Settings at least 15% apart on every parameter | 852 across 7 families (mostly two- and three-average crosses) |
| "Fixes" (layer 2) | A fixed hold or exit change | 10 variants built from 6 parts: minimum hold, trail, cooldown, confirmation, 200-bar filter, limit-order cost |
| Dynamic gates (layer 4) | One on/off gate per information family | 9 binary on crypto and 2 on FX, plus 5 graded or routed forms |
| ML forms (layer 5) | A model on a rule | 6 |

## 1. Where the strategies were found

| Place | What is there |
|---|---|
| `strategy/signals.py` (live) | 3 strategies: `regime_gate`, `donchian_breakout`, `vol_scaled_momentum`. |
| `docs/STRATEGY_LAYER.md` (live) | Family notes, library licences, a 7-item shortlist. |
| `ml_systems_v1/crypto/docs/` (archive, in this repo) | `TI_MASTER_CATALOG_2026_06_11.md` (about 110 indicators in 8 families), `CANONICAL_STRATEGY_DIMENSION_REGISTRY_2026_06_11.md` (bar types, cadences, entries, exits), `MARKET_FRAMEWORK/01_DEAD_LIST.md` (76 tested-and-failed ideas, D01–D76), `STRATEGY_PLAYBOOK.md`, `MARKET_STRATEGY_ARCHETYPES.md`, `STRATEGY_LEADERBOARD_2026_06_11.md`. |
| `ml_systems_v1/crypto/src/strat/` (archive) | 293 one-off research modules ("labs"): grid harness, mover capture, exits, rotation, funding, MA studies. The later rule library (`ladder/`) is there only as compiled `.pyc`. |
| `ml_harness`, the sibling repo (github.com/Karabo-VIII/ml_harness, HEAD `2c5475e`, 2026-11-20) | The readable rule libraries. Crypto: `crypto/src/strat/ladder/signals.py` (39 rule families) plus tranche 8 and 9 (25 more), registrations in `crypto/config/strat_ladder/*.yaml`, `crypto/config/ti_canonical_configs.yaml`. FX: `forex/src/strat/signals.py` (5 portfolio rules F0–F4 and 38 indicator rules T1–T38), results in `forex/docs/`. Findings for the 2020 u10 runs in `docs/findings/`. |
| Git history of this repo | `git show d3130d53:crypto/runs/strat/ladder/FINAL_MODELS.md` (the last finalized model per timeframe). |

Rule: read the archive for ideas and definitions. Do not import it. A strategy enters `strategy/` by being written again against gold and tested there.

## 2. How to read the tables

- **Mkt**: C = crypto, FX = forex, both.
- **Side**: LO = works as long-or-cash. LS = needs a short side (native on an FX pair; on crypto it needs perps). MN = market neutral (long and short legs together). OPT = needs options.
- **Code**: LIVE = in `strategy/` today. C-lib = rule exists in the old crypto library. FX-lib = rule exists in the old FX library (T-number given). Lab = a one-off archive module. New = never built.
- **Prior**: the old measurement, shortened. Blank means none was found. D-numbers refer to the archive dead list.

Context for every prior:

- Old crypto work was **long-only spot, no leverage**, 77 coins, taker cost 24 bps round trip. The short side of every crypto rule is untested.
- Old FX work was **long-short**, 7 majors, 2010–2024, measured spreads, mostly on 4h bars.
- Data built on this pipeline today: crypto u10 for calendar 2020 only (`data/u10`, 1m to 1d plus dollar bars). No real FX data yet: the Dukascopy download is throttled on this machine, so FX rows are code-ready and data-blocked.

## 3. The catalog

### A. Trend following (one asset, its own price history)

Bet: a move that has started continues. Many small losses, a few large wins. Loses in sideways markets.

| ID | Strategy | Rule | Mkt | Side | Code | Prior |
|---|---|---|---|---|---|---|
| TR-01 | Slow regime gate | Long while close is above its 200-bar average, else cash | both | LO/LS | LIVE `regime_gate` | The regime-gated trend book was the one full-cycle survivor of the June 2026 re-grade. It preserved capital in bear markets and did not earn in them. |
| TR-02 | SMA cross | Long while fast SMA is above slow SMA | both | LO/LS | C-lib `ma`, FX T1 | 4h 10/50 passed 3 of 4 folds. As a daily entry trigger it failed (D63). |
| TR-03 | EMA cross | Same with exponential averages | both | LO/LS | C-lib `ema`, FX T6 | 12/26 earned less than its own cost-free market exposure. |
| TR-04 | VWMA cross | Same with volume-weighted averages | both | LO/LS | C-lib `vwma`, FX T7 (tick count stands in for volume) | 4h 10/50 and 20/100 passed 3 of 4 folds. 2h 20/100 was the 2h sleeve of the final static book. |
| TR-05 | Other average types | WMA, DEMA, TEMA, ZLEMA, Hull, Kaufman adaptive, ALMA, VIDYA crosses | both | LO/LS | C-lib `hma_x`, `kama_x`, `wma_x`, `dema_x`, `tema_x`, `zlema_x`, `alma`; FX T30, T31 | Run for u10 2020. Results are in `ml_harness/docs/findings/`, not summarised here. |
| TR-06 | Triple MA, MA stack with slope | Cross plus a rising slow average | both | LO/LS | C-lib `golden_slope`, FX T26 | |
| TR-07 | MACD | MACD line above its signal line | both | LO/LS | C-lib `macd`, FX T8 | 2020: 0.76x buy-and-hold at 1d, drawdown −10%. |
| TR-08 | TRIX | Sign of the triple-smoothed rate of change | both | LO/LS | C-lib `trix`, FX T17 | |
| TR-09 | Rate of change (time-series momentum) | Long while the n-bar return is positive | both | LO/LS | C-lib `roc`, FX T3 | |
| TR-10 | Vol-scaled time-series momentum | Sign of the trailing return, sized to a volatility target | both | LO/LS | LIVE `vol_scaled_momentum`; C-lib `tsmom_va`; FX T25 and F4 | Crypto 2026-06: 41% a year against 21% for an exposure-matched random book. FX F4 ensemble: failed after costs (0.908 against a bar of 1.085 in the same run, turnover 65x a year). |
| TR-11 | ADX / DMI | +DI above −DI and ADX above a threshold | both | LO/LS | C-lib `adx_trend`, FX T19 | |
| TR-12 | Aroon | Aroon-up above Aroon-down | both | LO/LS | C-lib `aroon`, FX T18 | |
| TR-13 | Vortex | VI+ above VI− | both | LO/LS | C-lib `vortex`, FX T37 | 2020: about 0.6x buy-and-hold at 2h–4h. |
| TR-14 | Supertrend | ATR-band trend state | both | LO/LS | C-lib `supertrend`, FX T14 | 2020: about 0.6x buy-and-hold, drawdown 5–13%. |
| TR-15 | Parabolic SAR | Wilder stop-and-reverse state | both | LO/LS | C-lib `psar`, FX T29 | 2020: 0.68x buy-and-hold at 1d. |
| TR-16 | Ichimoku | Close above the cloud | both | LO/LS | C-lib `ichimoku`, FX T28 | |
| TR-17 | Regression slope, Kalman velocity | Sign of a fitted slope or filtered velocity | both | LO/LS | C-lib `kalman_vel` | |
| TR-18 | Oscillators read as trend | RSI above 55, stochastic above a level, CCI above a level, Ultimate Oscillator above 50, Schaff, TSI, Awesome, KST, Fisher, RVI, Stoch-RSI, Elder Ray, Balance of Power | both | LO/LS | C-lib `rsi_trend`, `stoch_trend`, `cci_trend`, `ult_osc`, `elder_ray`, `bop`, `stc`, `tsi`, `awesome_osc`, `kst`, `fisher`, `rvi`, `stoch_rsi`; FX T4, T15, T16, T32, T34, T36 | FX stochastic-trend at 4h won on EURUSD 2010 alone, then read 0.395 over 2010–2024. One year is not a verdict. |
| TR-19 | Pivot trend | Close above the prior day's pivot point | both | LO/LS | C-lib `pivot_trend`, FX T35 | |
| TR-20 | Trend on other bar types | The same rules on Renko, range or Heikin-Ashi bars | C | LO | Lab `alt_bar_trend_lab` | Held-out −20 to −53% a year. Renko and Heikin-Ashi redraw the current bar. |
| TR-21 | Trend on dollar bars | The same rules on dollar bars | C | LO/LS | New here (dollar bars are built) | |

### B. Channels and breakouts

Bet: price leaving a band keeps going. A channel used this way is trend following. The opposite use (fading the band) is section C.

| ID | Strategy | Rule | Mkt | Side | Code | Prior |
|---|---|---|---|---|---|---|
| CH-01 | Donchian / Turtle | Long on a close above the prior N-bar high, out below the prior M-bar low (20/10, 55/20) | both | LO/LS | LIVE `donchian_breakout`; C-lib `donchian`, `hl_breakout`; FX T2, T12, T22 | |
| CH-02 | Near-high breakout | Long while close is within 2% of the n-bar high | both | LO/LS | C-lib `brk`, FX T9 | The 111-bar version was in the market 18% of the time with an upside-to-downside capture ratio of 1.57. |
| CH-03 | ATR channel | Enter above SMA(n) + k·ATR(n), exit below SMA(n) | both | LO/LS | C-lib `atr_channel` | The old champion. 1d 20×2.0: validation 3.64x at −24.5% drawdown. Passed at four cadences (1d, 12h, 8h, 6h). |
| CH-04 | Keltner breakout | Break of EMA ± k·ATR, exit at the mid-line | both | LO/LS | C-lib `keltner`, FX T13 | 1d 20×2.0 was one of two top-graded cells out of 208. |
| CH-05 | Bollinger band ride | Long above the upper band, exit at the mid-line | both | LO/LS | C-lib `boll_trend`, FX T5 | |
| CH-06 | Squeeze breakout | Band width in its lowest quarter, then a 2-sigma break, held a fixed time | both | LO/LS | C-lib `squeeze_break`, FX T24 | |
| CH-07 | Regression-channel break | Close above the fitted line plus k residual deviations | both | LO/LS | C-lib `linreg_chan`, FX T38 | |
| CH-08 | Volatility breakout | Entry on a range expansion (move larger than k·ATR, narrow-range bar, inside bar) | both | LO/LS | Lab `volbreak_sweep` | No cross-asset winner; results depended on the period (D41). |
| CH-09 | Session breakout | Break of the Asian-session range at the London open | FX | LS | New | |
| CH-10 | Swing and fractal breakouts | Break of the last confirmed swing high | both | LO/LS | C-lib `swing_hl`, `fractals` | |
| CH-11 | 52-week-high proximity | Long near the yearly high | both | LO/LS | New | |
| CH-12 | Chandelier trail | Trend entry with an ATR trailing stop off the running high | both | LO/LS | Lab `family1_chandelier_trail` | Held-out −24% a year. A trailing stop added about 2 points a trade and did not rescue a weak entry. |

### C. Mean reversion and range trading (including grids)

Bet: a stretched price comes back. Many small wins, a rare large loss. Loses in trends.

| ID | Strategy | Rule | Mkt | Side | Code | Prior |
|---|---|---|---|---|---|---|
| MR-01 | Bollinger fade | Buy below the lower band, sell at the mid or upper band | both | LO/LS | New | 2020 crypto: under 0.5x buy-and-hold, with the lowest drawdown of any family. |
| MR-02 | RSI dip-buy | Buy RSI below a level, hold N bars | both | LO/LS | C-lib `dip_buy`, FX T10 | Level 30 lost to costs; level 20 was negative held-out (D50, D51). |
| MR-03 | RSI turn bounce | Enter when RSI crosses back up through the level, trail out | both | LO | C-lib `rsi_bounce` | Re-tested 2026-09 on 25 coins at 4h: did not beat random entries. |
| MR-04 | Stochastic, Williams %R, CCI, MFI fades | Buy oversold readings | both | LO/LS | Lab `deep2020_osc` | Same picture as MR-01. |
| MR-05 | Connors RSI | Composite short-term oversold score | both | LO/LS | New | Listed as tested-null in the old catalog. |
| MR-06 | Z-score reversion | Buy when close is k deviations under its rolling mean | both | LO/LS | New (crypto gold has `norm_deviation`) | |
| MR-07 | VWAP reversion | Fade distance from a rolling VWAP | C | LO/LS | C-lib `vwap_roll` (trend form only) | |
| MR-08 | Pullback in trend | Buy a dip while the slow trend is up | both | LO/LS | C-lib `pullback`, FX T23 | |
| MR-09 | Fibonacci retracement | Retrace into a fib level of the swing, enter when the move resumes | both | LO/LS | C-lib `fibo_retrace`, FX T27 | |
| MR-10 | Pivot and support/resistance range | Buy support, sell resistance inside a range | both | LO/LS | New | |
| MR-11 | Spot grid | Ladder of buy orders inside a price band; each fill is sold one step higher | both | LO | Lab `grid_harness.py` | Every tested setup lost held-out: 12 configs on BTC, ETH, SOL at 15m–1h, −3 to −18% even with perfect fills (D70). Dips kept falling and the stop sold at a loss. |
| MR-12 | Long-short or neutral grid | Grid that also sells short above the mid-price (futures grid) | both | LS | New | Not tested. |
| MR-13 | Grid variants | Percentage spacing, no upper bound ("infinity"), ATR-scaled spacing | both | LO/LS | New | |
| MR-14 | Drawn-channel trading | Buy the lower rail of a sloped channel, sell the upper | both | LO/LS | New | |
| MR-15 | Buy the extreme | Buy a liquidation spike or a volume climax | C | LO | Lab | Worse than random entry (D48, D49). |
| MR-16 | Liquidation-cascade fade | Buy after the forced selling is exhausted | C | LO | Lab | 0 of 77 beat random at 4h (D47). Needs a liquidation series the new pipeline does not carry. |
| MR-17 | Weekend-gap trade | Follow or fade the Sunday reopen gap | FX | LS | FX exploration E1f | No edge in either direction at 1h or 4h. |
| MR-18 | Asian-range fade | Fade moves inside the quiet session | FX | LS | New | |

### D. Cross-sectional: ranking instruments against each other

Bet: relative strength or weakness across the universe persists or reverses.

| ID | Strategy | Rule | Mkt | Side | Code | Prior |
|---|---|---|---|---|---|---|
| XS-01 | Momentum rotation | Rank by trailing return, hold the top K, cash for the rest | both | LO | Lab `momentum_rotation_lab`; C-lib selectors | Selection was real in one test window and no better than random in the next. Concentrating into the top K raised returns and drawdown together (0 of 19 improved both). |
| XS-02 | Dual momentum | Top-K by relative return, held only if own return is positive | both | LO | New | |
| XS-03 | Cross-sectional reversal | Buy last week's losers, sell the winners | both | MN/LO | New | The old playbook's top unbuilt idea. Within-group reversal failed 0 of 72 at 4h (D53). |
| XS-04 | Strength-weighted sizing | Weight by signal strength instead of equally | both | LO | C-lib selectors | |
| XS-05 | Low-volatility tilt | Overweight the calmest names | C | LO | Lab | No effect. |
| XS-06 | Funding tilt | Overweight low or negative funding | C | LO | Lab `funding_tilt_longonly` | Worse than equal weight in every cell tested. |
| XS-07 | Pairs and cointegration | Trade the spread between two related instruments | both | MN | New for FX | Crypto, 91 pairs: lost 8% a trade in training (D37). FX crosses were not tested. |
| XS-08 | Currency-strength momentum | Rank the 8 currencies by 1/3/12-month momentum, long the top 3, short the bottom 3 | FX | LS | FX-lib F1, F1v2 | F1 failed on turnover (90x a year). The monthly-rebalance fix reached 0.930 against the 1.067 bar. |
| XS-09 | FX value | Long cheap currencies by PPP or real exchange rate | FX | LS | New | Deferred: no PPP data. |
| XS-10 | FX style blend | One third carry, one third trend, one third value, monthly | FX | LS | FX-lib F0 | The FX bar every other rule was compared with: 1.067 over 2010–2022 with measured spreads, drawdown −8.2%. |
| XS-11 | Dispersion-routed rotation | Switch rule family by how spread-out returns are | C | LO | C-lib `d3` | Near miss: 4.77x on validation, missed the drawdown bar by 0.1 point. |
| XS-12 | Core plus satellite | A slow core book and a small fast sleeve | C | LO | C-lib `d4`, Lab | No gain. |
| XS-13 | BTC-dominance rotation | Shift between BTC and alts on dominance | C | LO | New | |
| XS-14 | Triangle residual | Trade a cross against the rate implied by two majors | FX | MN | New | Needs crosses, which the fetch does not build yet. |

### E. Carry and yield

Bet: collect a rate difference and hope price does not move against it by more.

| ID | Strategy | Rule | Mkt | Side | Code | Prior |
|---|---|---|---|---|---|---|
| CA-01 | FX carry | Long high-rate currencies, short low-rate | FX | LS | FX-lib F2 (`carry_pct` is on FX gold) | Best of its group: 1.174, drawdown −7.2%, turnover 3x a year. Failed the fold test (4 of 7) and validation against the bar. Rates only existed from 2015. |
| CA-02 | Carry with a filter | Carry held only when trend agrees or risk is on | FX | LS | New | A risk-off gate on the style blend added nothing. |
| CA-03 | Crypto funding carry | Long spot, short the perp, collect funding | C | MN | New | About zero after costs since 2024 (D18). This is Binance's "Arbitrage" bot. |
| CA-04 | Basis trade | Trade the perp-minus-spot premium | C | MN | C-lib `perp_basis` (as a signal) | |
| CA-05 | Funding extreme as a signal | Fade or follow extreme funding | C | LO/LS | Lab | High funding was followed by higher returns, the reverse of the theory (D42). Held-out −4 to −9% (D54). |

### F. Volatility

| ID | Strategy | Rule | Mkt | Side | Code | Prior |
|---|---|---|---|---|---|---|
| VO-01 | Volatility targeting | Scale any book by target vol over realised vol | both | any | Inside LIVE `vol_scaled_momentum` | As a cap on the gated champion it cut drawdown and cut wealth more. |
| VO-02 | Vol-regime switch | Run a trend rule only in a chosen volatility bucket | both | any | C-lib family gate | Run for u10 2020; see `ml_harness/docs/findings/FAMILY_VOL_REGIME_*`. |
| VO-03 | Risk parity, inverse-vol weights | Equal risk per instrument | both | any | New | |
| VO-04 | Drawdown stop | Go to cash after a book drawdown, re-enter on recovery | both | any | C-lib `s3` | 0 of 4 improved return per unit of drawdown. |
| VO-05 | Selling volatility | Covered calls, short straddles | C | OPT | New | Covered calls cost 24 points of yearly return (D38). No options data. |
| VO-06 | Buying volatility | Straddle on a volatility expansion | C | OPT | New | Imitations built from perps lose to costs (D64, D65). |

### G. Market-regime overlays (a switch on top of another book)

| ID | Strategy | Rule | Mkt | Side | Code | Prior |
|---|---|---|---|---|---|---|
| RG-01 | BTC regime gate | Hold any crypto book only while BTC is above its 50-day average, with a one-day lag | C | LO | C-lib `d1_btc_gate` | The strongest old result. On the champion: 3.94x at −20.5%. It also lifted a plain equal-weight basket, so the gate carries value on its own. An earlier 100-day version failed out of sample (D74). |
| RG-02 | Breadth throttle | Scale exposure by the share of coins above their own average | C | LO | C-lib `d2` | Helped a raw basket; added nothing on top of RG-01. |
| RG-03 | Graded exposure | Partial instead of on/off gating | C | LO | C-lib `d1b` | Near miss. |
| RG-04 | Statistical regime | Trade trend rules only when efficiency ratio or Hurst says "trending" | both | any | C-lib family gate | Run for u10 2020. |
| RG-05 | Funding regime | Stand aside when funding is extreme | C | LO | C-lib family gate | Earlier test: +5.5 points in one window, −16.7 in the next. |
| RG-06 | Flow-magnitude gate | Stand aside on abnormal order-flow size | C | LO | C-lib family gate | Run for u10 2020. |
| RG-07 | Macro and sentiment gate | Stablecoin supply, ETF flows, DXY, VIX, fear-and-greed | both | any | C-lib family gate | Cut drawdown, cost return (D69). |
| RG-08 | Multi-state router | Classify the regime, route to the matching strategy | both | any | Lab | Routing diluted results (D05, D33). |
| RG-09 | Switch-aware book routing | Choose which book to run each bar | C | LO | C-lib `dt_route2` | With hindsight the ceiling was 55.8x. A learned policy reached 3.60x at −15.2%. |

### H. Volume and order flow

| ID | Strategy | Rule | Mkt | Side | Code | Prior |
|---|---|---|---|---|---|---|
| FL-01 | Volume-trend indicators | OBV, MFI, Chaikin money flow, accumulation/distribution, force index, ease of movement, Klinger, price-volume trend | C (FX by tick count) | LO/LS | C-lib `obv_trend`, `mfi_trend`, `force_index`, `cmf`, `accdist`, `emv`, `klinger`, `pvt`; FX T20, T21, T33 | |
| FL-02 | Volume-profile level | Price against the highest-volume price of a window | C | LO/LS | C-lib `vp_poc` | |
| FL-03 | Cumulative volume delta, taker ratio | Running buy-minus-sell volume | C | LO/LS | C-lib `cvd`, `taker_ratio` | |
| FL-04 | Order-flow direction | Trade the sign of extreme flow imbalance | C | LO/LS | Lab | A real effect of about 9 bps a trade, under the 24 bps cost (D75). |
| FL-05 | VPIN, Kyle lambda as a risk input | Size down when flow is toxic | C | any | New | |
| FL-06 | Lead-lag | BTC leads alts; USD strength leads pairs | both | LO/LS | Lab; FX exploration E2f | None at 15m or slower in crypto (D46). Dead at 1h and 4h in FX. |
| FL-07 | Open-interest change | Trade OI build-up or flush | C | LO/LS | New | Needs `--include-oi`. |
| FL-08 | Order book and footprint | Book imbalance, absorption, market making | C | — | — | Needs trades-at-price or a book. Not on gold. |

### I. Sessions and the calendar

| ID | Strategy | Rule | Mkt | Side | Code | Prior |
|---|---|---|---|---|---|---|
| SE-01 | Day of week | Long on chosen weekdays | both | LO/LS | C-lib `dow`, FX T11 | Built as a no-skill control. It topped one 10-month crypto window, which shows how little one window proves. |
| SE-02 | FX session gate | Trade only in London–New York hours, or stay flat in Tokyo | FX | LS | FX study | On the style blend both gates lost (0.83 and 0.81 against 1.067). |
| SE-03 | Turn of month | Hold risk around the month end | FX | LS | FX exploration E4f | Inside the noise band. |
| SE-04 | Hour of day, weekend, funding-time effects | Crypto intraday seasonality | C | LO/LS | C-lib calendar gate | Run for u10 2020. |
| SE-05 | Fix and month-end flows | London 4pm fix, rebalancing flows | FX | LS | New | |

### J. Events

| ID | Strategy | Rule | Mkt | Side | Code | Prior |
|---|---|---|---|---|---|---|
| EV-01 | Economic releases | Trade payrolls, CPI, rate decisions | FX | LS | New | Needs an event calendar. |
| EV-02 | Listings, unlocks, ETF flows | Crypto catalysts | C | LO/LS | New | Needs data. |
| EV-03 | Riding intraday movers | Join a coin already up several percent | C | LO | Lab | Continuers could be identified, and the trade still lost: affordable orders did not fill on the fast ones (D72, D76). |

### K. Position management (components and bot types, not signals)

| ID | Item | What it is | Code | Prior |
|---|---|---|---|---|
| PM-01 | Buy and hold | Constant exposure | LIVE `buy_and_hold` | The benchmark. |
| PM-02 | Rebalancing | Fixed weights, re-set monthly (Binance "Rebalancing" bot) | C-lib `s0_baselines` | Used as the floor every strategy had to beat. |
| PM-03 | DCA | Buy a fixed amount on a schedule (Binance "DCA", "Auto-Invest") | New | A way to accumulate, not a signal. |
| PM-04 | Averaging down | Buy more as price falls, larger each time (martingale) | New | Can lose the whole account in a trend. Test only with a hard loss cap. |
| PM-05 | Exits | Fixed hold; percent trail; ATR trail; take-profit; triple barrier; signal flip; regime-based; exit on a finer clock | Lab `exit_*` | Smart exits did not beat a fixed hold on daily breakouts (D61). Take-profit, regime-based and finer-clock exits were never tested. |
| PM-06 | Sizing | Equal weight; vol target; fractional Kelly; conviction; scaling in and out | Lab | Larger Kelly fraction and conviction sizing both lowered held-out results (D25, D26). |
| PM-07 | Execution | Limit-order entry, TWAP | C-lib `p1_fill_shadow` | Limit entry was worth +5% to +14% of wealth by timeframe on paper. Measured fill rate was 21–40%. |

### L. Machine learning on top of rules

| ID | Strategy | Rule | Code | Prior |
|---|---|---|---|---|
| ML-01 | Meta-labeling | A model decides which of a rule's trades to take | C-lib `ml1`, `ml2` | No gain. |
| ML-02 | Return prediction | Ridge or boosted trees predict the next return | Lab | Coin-flip accuracy across 6 model types (D16, D17). |
| ML-03 | Cross-sectional tilt on a rule book | The rule decides when to be in; a model reweights the coins held | C-lib `ml3` | The last finalized models. 1d: 4.68x at −18.7% on validation against 3.94x for the rule alone. |
| ML-04 | Regime classifier | Predict the regime, trade it | Lab | Underperformed (D28). |
| ML-05 | Config selector | Predict the best parameters for the next period | Lab | Failed five separate ways (D73). |
| ML-06 | Sequence and world models | Read the last L bars, predict the next | archive `wm/` | Diagnostic only. |
| ML-07 | Formula search | Genetic or LLM-proposed rules with an overfitting check | Not built | |

### M. Arbitrage and infrastructure-bound (listed so they are not mistaken for candidates)

Triangular and cross-exchange arbitrage, market making, latency trading, on-chain MEV. These need execution speed or venues this project does not have.

## 4. Where "channel" strategies sit, and Binance bots

A channel is a tool: two lines around price. It is used in two opposite ways.

| Use | Bet | Rows | Wins when | Loses when |
|---|---|---|---|---|
| Trade the break | Price leaving the band keeps going | Section B (Donchian, Keltner, ATR channel, Bollinger ride, squeeze) | Trends | Sideways markets |
| Fade inside the band | Price stays in the band | Section C (grid, Bollinger fade, range trading) | Sideways markets | Trends |

Binance's range bots are the second kind. A Spot Grid or Futures Grid bot is given a lower and an upper price and trades back and forth between them, so it is range trading, MR-11 to MR-13.

| Binance bot | Category here |
|---|---|
| Spot Grid | Range mean reversion, long only (MR-11) |
| Futures Grid (long, short, neutral) | The same with leverage and a short side (MR-12) |
| Rebalancing | Portfolio management (PM-02) |
| Spot DCA, Auto-Invest | Accumulation schedule (PM-03, PM-04) |
| Arbitrage | Funding carry (CA-03) |
| TWAP, volume participation | Execution (PM-07) |

Why grid bots look good on the bot marketplace: the page ranks running bots by return, so it shows the ones whose range happened to hold. Bots whose range broke were closed or liquidated and are not on the page. Many of the top ones are leveraged, short-lived, and started in a quiet stretch. The numbers are real for those bots and say little about a grid started today.

The archive tested both uses on crypto. The break side produced the best results it ever had (CH-03, CH-04). The fade side as a long-only grid lost in every setup (MR-11). The parts never tested are the long-short grid (MR-12), grids on FX, and grids switched on only when a trend filter says the market is flat. FX crosses such as AUDNZD or EURCHF range more than coins do, so that is where a grid has its best chance.

## 5. What the archive measured overall

Priors, in order of how much evidence stood behind them.

1. **Costs set the timeframe floor.** At market-order cost, crypto rules at 1h and 30m were 0 of 52 positive; FX rules at 30m were wiped out. Crypto survivors stopped at 2h. On FX, 4h was the least bad cadence and still had no survivor.
2. **In long-only crypto the gain came from being out of the market at the right times.** The market-level gate (RG-01) did more than any choice of entry rule.
3. **Slow band-style trend rules were the most robust single rules** (CH-03, CH-04, TR-04) at 2h to 1d.
4. **Buying dips and extremes lost** in every form tried on crypto (MR-02, MR-11, MR-15, MR-16).
5. **Per-coin tuning was noise.** One pooled setting per timeframe held up; a setting per coin or per regime did not.
6. **On FX majors, single-pair indicator rules did not survive**: 0 of 168 across 26 families and four cadences, 2010–2024. The open FX work was at portfolio level: carry (CA-01), currency-strength ranking (XS-08), value (XS-09).
7. **One window proves little.** The no-skill weekday rule topped a 10-month crypto test.

Gaps in the old evidence, where a fresh run adds the most:

- The short side on crypto: every rule in sections A to C, long-short.
- FX with real tick data on this pipeline, including the 11 trend rules (T28–T38) whose old results were not written up.
- Mean reversion and grids outside long-only crypto (MR-01, MR-06, MR-12, MR-13 on FX crosses).
- Exits never tested (PM-05) and dollar-bar versions (TR-21).
- Multi-year crypto history. Only 2020 is built here.

## 6. Methodology: to agree before any test

No strategy is run until this section is settled. Part 6.1 lists the rules the earlier projects ended up with, most of them from the owner's own decisions at the time. They are proposed as the starting point, to confirm or change. Part 6.2 lists what has not been decided for this project.

### 6.1 Proposed rules (carried over, to confirm)

| # | Rule | What it prevents |
|---|---|---|
| 1 | **Declare before running.** Each run is written down first: rules, settings, instruments, dates, timeframes, costs. The file is committed before any result exists. | Choosing the test after seeing the answer |
| 2 | **Naive means naive.** A layer-1 rule trades its own rule exactly as a person following it would. No filter is hidden inside it. | A "simple rule" that is secretly a tuned one |
| 3 | **Report every row.** A run publishes all its cells. A losing row stays in the table as knowledge. | Showing only the winner |
| 4 | **Each layer is judged against the layer below.** Random entries, then static, then dynamic, then ML. A layer earns its place only by adding to the one under it. | Calling something an improvement because it beat the wrong benchmark |
| 5 | **Buy and hold is not the yardstick.** Each row reports the share of rallies kept next to the share of declines taken, and time in market during each. Never one number without the other. (Owner decision, 2026-09-14.) | Rewarding a rule for simply staying long in a rising year |
| 6 | **The unit is the family, not the setting.** Settings closer than 15% on every parameter count as one. A family is described by the spread of its results, not its best member. | Mistaking a lucky setting for a strategy |
| 7 | **Count the trials.** Every run records how many distinct things were tried, so the result can be discounted for it. | Finding a winner by trying enough times |
| 8 | **No verdict from one window.** Results are read across several pre-declared market periods. The final hold-out period stays sealed, and each look at it is logged. | A one-year or one-regime fluke |
| 9 | **The random-entry control is matched.** Same time in market and same number of trades as the rule it is compared with. | A control that loses only because it pays more fees |
| 10 | **Structure before results.** Before any return is shown: prove the rule gives the same decision at bar t whether or not later data exists, and that positions obey the stated rule. A new checker is trusted only after it catches a defect planted on purpose. | Look-ahead and silent bugs producing good-looking numbers |
| 11 | **One engine, one cost model, next-bar fills.** Every layer runs through the same accounting. | Layers that cannot be compared |
| 12 | **ML is anchored.** With the model's influence set to zero the book must equal the rule underneath exactly. Only walk-forward, out-of-sample model scores count. | A model that looks good because it saw the answers |
| 13 | **Families are parked, not deleted.** A family is set aside only when its whole spread fails with enough data to say so, and the note records what would reopen it. | Throwing away an idea on thin evidence |

### 6.2 Open decisions

| # | Decision | Why it matters | Current state |
|---|---|---|---|
| A | **The layer stack.** Is section 0 the right set of layers and the right order? | It fixes what gets compared with what. | Proposed in section 0. |
| B | **Sides.** Long-or-cash only, or also long-short, for crypto research? | Doubles the table. FX is long-short either way. | Old crypto work was long-only spot. |
| C | **Data windows.** Which years are for building, which for checking, which stay sealed? | Nothing can be selected fairly until this is fixed. | `config/pipeline.yaml` has train to 2023-07-01, validation to 2024-05-15, out-of-sample to 2025-03-15, unseen after. Only 2020 is built. |
| D | **Instruments.** 10 coins first, or 50 or 100? Which FX set? | Breadth changes what cross-sectional rules can show, and coins that died must be included for an honest result. | u10 for 2020 on disk. No real FX data. |
| E | **Timeframes.** All nine plus dollar bars, or a coarse set first? | Fine timeframes multiply cost and runtime. | All nine are built. |
| F | **What "works" means at each layer.** Beat matched random entries? Keep more of rallies than it takes of declines? A wealth or drawdown bar? | It is the pass mark. Set it before seeing results. | Not set. |
| G | **Settings per rule.** Textbook only (1 to 4 per rule), a distinct set, or a dense grid? | More settings means a larger discount for trials. | Old naive runs used textbook settings. |
| H | **Book form.** Each instrument on its own, or one book that equal-weights every instrument whose signal is on? | The two give different answers for the same rule. | The live engine is single-instrument. |
| I | **Costs.** Market-order fee plus slippage for crypto; measured spread for FX. Is limit-order entry reported beside it or left out? | Costs decide which timeframes survive. | `config/pipeline.yaml`: crypto 10 bps fee + 2 bps slippage per side. |
| J | **What ML is allowed to sit on.** Every rule, or only rules that passed their own layer? | The old project admitted only survivors. | Not set. |
| K | **Order of work across layers.** Finish a layer everywhere before starting the next, or take one family through all layers first? | Breadth-first gives the full map sooner; depth-first gives one finished product sooner. | Not set. |

### 6.3 What has to exist before the first test

Listed so the scope is visible. None of it is built.

1. A rule registry in `strategy/`: one function per catalog ID, each checked for warm-up, no look-ahead, and correct direction on a made-up rising and falling series.
2. A multi-instrument book engine. The live engine handles one instrument.
3. A runner that reads a declared run, executes every cell, and writes every row with the trial count.
4. The matched random-entry control and the baselines of layer 0.
5. The rally and decline measurement of rule 5.
6. More data: further crypto years, and FX ticks once the download works.
