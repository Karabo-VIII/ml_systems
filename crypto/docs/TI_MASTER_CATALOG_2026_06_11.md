# Master Technical-Indicator Catalog — classified by family (2026-06-11)

> User ask (/orc): *"we have a master list of technical indicators, correct? Add on to that, and then
> classify each family of TIs (Trend, Vol[ume], Volatility, etc)."*
>
> **Before this doc, there was NO master list** — only RSI/MACD/Bollinger coded
> (`src/oracle/indicators_ta.py`) + RSI/MA/MACD/Bollinger/ADX/ATR documented (`docs/TI_HARNESS.md`).
> This is the comprehensive, crypto-aware catalog. Machine-readable mirror:
> [`config/ti_master_catalog.yaml`](../config/ti_master_catalog.yaml).

**Columns:** `pandas_ta` = the library fn if available (our backend; same math as the oracle registry).
`HAVE` = **C**oded (in `indicators_ta.py`) / **X**=in chimera feature store / **—**=don't have /
**G**=crypto-data gap (needs ingest). `LA` = look-ahead safety: **S**afe (causal), **R**isky
(repaint/param-sensitive), **U**nsafe (centered/future-peeking by construction). `dead` = dead-list
status (do not re-mine as a standalone edge; ✓=tested-null, blank=untested-as-edge).

---

## 1. TREND / DIRECTION (where price is going, smoothed)
| Indicator | measures | params | pandas_ta | HAVE | LA | dead |
|---|---|---|---|---|---|---|
| SMA | mean price | n=10..200 | `sma` | X | S | D63 (naive MA cross HARD) |
| EMA | exp-weighted mean | n=12..200 | `ema` | C | S | D63 |
| WMA / HMA / DEMA / TEMA / ZLMA | lag-reduced means | n | `wma/hma/dema/tema/zlma` | — | S | D63-class |
| KAMA (Kaufman adaptive) | ER-scaled adaptive MA | n=10, fast2/slow30 | `kama` | — | S | |
| ALMA / McGinley | smoothed adaptive MA | — | `alma/mcgd` | — | S | |
| MACD | fast-slow EMA spread + signal | 12/26/9 | `macd` | C | S | D45-class (entry-timing) |
| ADX / DMI | trend STRENGTH + ±DI direction | 14 | `adx` | X(doc) | S | |
| Aroon | bars-since-high/low up/down | 14/25 | `aroon` | — | S | |
| Supertrend | ATR-banded trend flip | 10, 3×ATR | `supertrend` | — | R (repaints intrabar) | |
| Parabolic SAR | trailing stop-and-reverse | 0.02/0.2 | `psar` | — | R | |
| Ichimoku | multi-line cloud (lead spans **shifted forward**) | 9/26/52 | `ichimoku` | — | **U** (senkou shifted +26 = future on chart; the *signal* is causal, the cloud plot is not) | |
| Donchian channel (mid) | N-bar high/low midline | 20 | `donchian` | X | S | |
| Vortex (VI+/VI−) | trend via true-range swings | 14 | `vortex` | — | S | |
| TRIX | triple-EMA ROC | 15 | `trix` | — | S | |
| Schaff Trend Cycle | MACD run through stochastic | 23/50 | `stc` | — | R | |
| Linear-reg slope / forecast | OLS slope over window | 14..100 | `linreg` | partial(X) | S | |
| Chande Kroll stop | volatility trailing stop | 10/1/9 | `cksp` | — | S | |

## 2. MOMENTUM / OSCILLATORS (speed & overbought/oversold)
| Indicator | measures | params | pandas_ta | HAVE | LA | dead |
|---|---|---|---|---|---|---|
| RSI | avg-gain/avg-loss ratio | 14 | `rsi` | C | S | D50 thr30 / D51 thr20 HARD |
| Stochastic %K/%D | close vs N-bar range | 14/3/3 | `stoch` | — | S | |
| Stochastic RSI | stochastic of RSI | 14/14/3 | `stochrsi` | — | S | |
| CCI | dev from typical-price MA | 20 | `cci` | — | S | |
| Williams %R | inverted stochastic | 14 | `willr` | — | S | |
| ROC / Momentum | %-change over n | 10..20 | `roc/mom` | X | S | D67-class (per-move) |
| CMO (Chande) | momentum oscillator | 14 | `cmo` | — | S | |
| TSI (true strength) | double-smoothed momentum | 25/13 | `tsi` | — | S | |
| Ultimate Oscillator | 3-timeframe momentum | 7/14/28 | `uo` | — | S | |
| Awesome Oscillator | 5/34 median-price SMA diff | 5/34 | `ao` | — | S | |
| Fisher Transform | Gaussianized price → turns | 9 | `fisher` | — | R | |
| RVI (relative vigor) | close-open vs range | 14 | `rvi` | — | S | |
| Connors RSI | RSI + streak + percentrank | 3/2/100 | `crsi`* | — | S | ✓ tested null (ideation top-3, held-out NULL) |
| KST / Coppock | long-cycle momentum sums | — | `kst/coppock` | — | S | |
| Balance of Power / Elder Ray | buy vs sell pressure | — | `bop/eri` | — | S | |

## 3. VOLATILITY (how much price moves — the PREDICTABLE channel per D55)
| Indicator | measures | params | pandas_ta | HAVE | LA | dead |
|---|---|---|---|---|---|---|
| ATR | avg true range | 14 | `atr` | X(doc)+C(family_regime) | S | |
| Bollinger Bands / %B / Bandwidth | SMA ± k·σ | 20/2 | `bbands` | C | S | D70 (LO grid HARD) |
| Keltner Channel | EMA ± k·ATR | 20/2 | `kc` | — | S | |
| Std-dev (rolling) | return dispersion | 20 | `stdev` | X | S | |
| Donchian width | N-bar range size | 20 | `donchian` | X | S | |
| Historical/Realized Vol | annualized σ of returns | 30 | — | X (`norm_yz_volatility`) | S | |
| Yang-Zhang | OHLC drift-robust vol | 30 | — | X (`norm_yz_volatility`) | S | |
| Garman-Klass / Rogers-Satchell / Parkinson | OHLC-range vol estimators | n | — | partial | S | |
| Chaikin Volatility | EMA of H−L range | 10 | — | — | S | |
| Ulcer Index | depth+duration of drawdown | 14 | `ui` | — | S | |
| Mass Index | range-ratio reversal bulge | 25 | `massi` | — | S | |
| Choppiness Index | trend vs range (0-100) | 14 | `chop` | — | S | (the chop/efficiency screen — see §5 ER) |
| RVI-vol / vol-of-vol | σ of σ | — | — | X (`norm_vol_cluster`) | S | |

## 4. VOLUME / FLOW (conviction behind the move)
| Indicator | measures | params | pandas_ta | HAVE | LA | dead |
|---|---|---|---|---|---|---|
| OBV | cumulative signed volume | — | `obv` | — | S | |
| Chaikin Money Flow | volume-weighted close position | 20 | `cmf` | — | S | |
| MFI (money flow index) | volume-weighted RSI | 14 | `mfi` | — | S | |
| VWAP / VWMA | volume-weighted price | session/n | `vwap/vwma` | X (`vwap_*`) | S (rolling); R (anchored repaints) | |
| Accumulation/Distribution | close-loc × volume cum | — | `ad` | — | S | |
| Force Index | price-change × volume | 13 | `efi` | — | S | |
| Ease of Movement | move per unit volume | 14 | `eom` | — | S | |
| Klinger Oscillator | volume-force trend | 34/55 | `kvo` | — | S | |
| PVT / NVI / PVI | price-volume trend variants | — | `pvt/nvi/pvi` | — | S | |
| Volume Profile / POC | volume-by-price node | window | — | X (`lob_*` proxy) | S | |
| Amihud illiquidity | |return|/volume | 30 | — | — (time-bar add) | S | (06§C #3 candidate add) |
| Kyle's lambda | price impact per flow | window | — | X (`norm_kyle_lambda`) | S | |
| CVD (cumulative volume delta) | cum (buy−sell aggressor) | — | — | X (`buy_vol/sell_vol`) | S | |
| Taker buy/sell ratio | aggressor imbalance | — | — | X | S | |

## 5. STATISTICAL / CYCLE (the regime + math layer)
| Indicator | measures | params | pandas_ta | HAVE | LA | dead |
|---|---|---|---|---|---|---|
| Hurst exponent | trend/mean-revert/random (0.5=random) | 100+ | — | X (`hurst_regime`) | R (full-sample variants UNSAFE) | crypto Hurst≈0.5 (D-lesson) |
| Autocorrelation (ret, |ret|) | memory; |ret| AC1 0.18→0.33 (vol clusters) | 1..30 | — | X | S | |
| Z-score / normalization | std-devs from mean | 20..200 | `zscore` | X (all `norm_*`) | R (full-sample = UNSAFE, G-AUDIT-011) | |
| Kaufman Efficiency Ratio | directional/path ratio (chop screen) | 20 | — | X (computed in config_map/regime_dna) | S | M2-tested (regime-math, tail-driven) |
| Linear-reg R²/slope | trend quality | n | `linreg` | partial | S | |
| Hilbert Transform (cycle, dom-period) | dominant cycle length/phase | — | `ht_*` (talib) | — | R | |
| DFT/FFT power | spectral energy by frequency | — | — | — | R (windowed = leakage care) | |
| Permutation entropy | predictability/complexity | m=3..5 | — | X (`norm_perm_entropy`) | S | |
| Fractal dimension (Katz/Higuchi) | roughness | — | — | X (`norm_fd_close` = frac-diff, NOT fractal-dim — label drift, 06§C) | S | |
| Kalman velocity/state | causal level+velocity | — | — | — (06§C #2 candidate add) | S | |
| Realized skewness / kurtosis (5m) | tail asymmetry/fatness | 5m | — | — (06§C #1 TOP add; 5m pipeline exists) | S | lottery-reversal candidate |
| DFA (detrended fluctuation) | long-range correlation | — | — | — | R | |
| Frac-diff (López de Prado) | stationary-yet-memory series | d∈(0,1) | — | X (`norm_fd_close`) | S | |

## 6. STRUCTURE / SUPPORT-RESISTANCE (price geometry)
| Indicator | measures | params | pandas_ta | HAVE | LA | dead |
|---|---|---|---|---|---|---|
| Pivot Points (classic/Fib/Camarilla/Woodie/DeMark) | S/R from prior H/L/C | daily | `pivots`* | — | S (prior-period = causal) | |
| Fibonacci retracement/extension | swing-ratio levels | swing | — | — | R (needs swing pick) | |
| Swing high/low | local extrema | k-bar | — | — | R (confirmed-late) | |
| Williams Fractals | 5-bar reversal pattern | 5 | — | — | R (2-bar lag to confirm) | |
| ZigZag | filtered swing legs | %thr | — | — | **U** (REPAINTS — last leg redraws; never use as a live signal) | A-class artifact |
| Murrey Math / round numbers | psychological levels | — | — | — | S | |
| Donchian channel levels | N-bar H/L | 20 | `donchian` | X | S | |
| Prior-day/week H/L, VWAP bands | session reference levels | — | — | X (resample) | S | |

## 7. CRYPTO-NATIVE — DERIVATIVES / POSITIONING (no equity analog; the real edge surface)
| Indicator | measures | source | HAVE | LA | dead |
|---|---|---|---|---|---|
| Funding rate (+ z-score) | perp-spot tether; crowding | 8h, on disk | X (`fp_*`, `fund_rate_z30`) | S | D18 carry decayed / D42 guard sign-inverted / D54 reversion SCOPED |
| Open Interest + OI-delta | leveraged $ outstanding; fuel | 5m metrics, on disk | X (`oi_*`, `oi_d1h/4h/24h`) | S | oi_d24h marginal (D71 thread) |
| Long/Short Ratio (global + top-trader) | positioning crowding/skew | 5m metrics | X (`long_short_ratio`, `top_*_lsr`) | S | D12 LSR regime-dependent |
| Perp basis / premium | perp−spot %; leverage demand | derived | X (`basis_*`) | S | |
| Liquidations (long/short, cascade) | forced-flow events | 1m proxy (liq_subbar) | X (`liq_*`) | S | D47/D48/D52/D71 HARD (buy-the-extreme/signature) |
| Liquidation HEATMAP / cluster proximity | where liq levels sit AHEAD (magnet) | **Coinglass (paid)** | G (proxy-tested NULL 2026-06-11, AUC 0.49) | S | SKIP (free proxy no lift) |
| Estimated leverage ratio | OI/market-cap | derived | partial | S | |
| Options: DVOL / skew / term / gamma | implied-vol surface | Deribit | G (`dv_dvol` 98% null off BTC/ETH) | S | D38 covered-call −24pp; D64/D65 gamma HARD |

## 8. CRYPTO-NATIVE — ON-CHAIN / MACRO (slow regime context)
| Indicator | measures | source | HAVE | LA | dead |
|---|---|---|---|---|---|
| Exchange netflow (in/out) | coins to/from CEX (sell/accum pressure) | Glassnode/CryptoQuant | G (06§C, 48-72h lead) | S | untested (external) |
| Stablecoin supply + SSR | dry powder / buying capacity | on disk (`stbl_*`) | X | S | D69 macro-flow = preservation not alpha |
| MVRV / MVRV-Z / SOPR / NUPL | on-chain valuation/profit-taking | Glassnode | G | S | untested (external) |
| Realized cap / NVT / active addr | network value/usage | Glassnode | G | S | |
| Exchange reserves / whale flows | supply on exchanges; large moves | on disk (`wh_whale_*`) | X | S | D69-class |
| ETF flows (BTC/ETH spot) | institutional demand | farside (on disk) | X (`etf_*`) | S | D69 (coincident not predictive) |
| BTC dominance / Altcoin Season Index | rotation regime | derived | X | S | broke 2024-25 (lagging by construction) |
| Fear & Greed Index | sentiment composite | alternative.me | partial | S | contrarian-extreme only (lore) |

---

## How to use this catalog (the load-bearing notes)
1. **HAVE vs DON'T-HAVE:** only RSI/MACD/Bollinger are *coded* (`indicators_ta.py`); ATR/MA/ADX are
   in the TI-harness config; most §1-6 classics are **pandas_ta one-liners we don't yet wire**; the
   crypto-native §7-8 is where our real data advantage is (most on disk).
2. **Look-ahead landmines (LA=U/R):** **ZigZag and centered-DPO REPAINT** (A-class artifacts — never
   a live signal); **Ichimoku's cloud is plotted 26 bars ahead** (the plot misleads, the signal is
   causal); **full-sample z-score / Hurst** are G-AUDIT-011 leaks — always rolling/causal.
3. **dead-list reality:** the *standalone* edge of price-TI families (MA, RSI-threshold, micro-z,
   grids) is HARD-refuted (D50/D52/D55/D63/D67/D70). **This catalog is NOT a list of edges** — it's
   the configurable lens-set. Per the project's findings, TI value (if any) is as a *regime/exit/
   conditioning* layer or a *cross-sectional* input, never a standalone entry trigger.
4. **The genuinely-underexploited families** (low dead-list coverage): **Volatility §3** (the one
   PREDICTABLE channel per D55 — bet size not direction), **Statistical §5** (realized skew/kurt,
   Kalman, efficiency-ratio adaptivity), and **crypto-native §7-8** beyond the dead liquidation-spike
   cells. These are where a NEW TI-based study should look.

Coverage: 8 families, ~110 indicators. Source: domain canon + `pandas_ta` + our chimera dictionary +
the 73-entry dead-list. RWYB where HAVE-status was checked against `src/oracle/indicators_ta.py` and
`docs/CHIMERA_FEATURE_DICTIONARY.md`.
