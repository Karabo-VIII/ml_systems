# Chimera Feature Dictionary (2026-06-09)

**Purpose.** Understand every chimera feature individually -- what it MEANS, what it TELLS US on its own, AND how its
interpretation CHANGES when pegged to a particular asset archetype (does funding rate mean the same on BTC as on a
meme? -- no). Per-feature meanings are generated from the source-of-truth `src/narrate/feature_map.py` (218 curated
features, 100% column coverage, 12 families); the asset-conditional layer is synthesized from the 2026-06-09 research
scouts (see [CRYPTO_MARKET_UNDERSTANDING.md](CRYPTO_MARKET_UNDERSTANDING.md) for citations).

**How to read.** Each family below gives: (a) what it tells us on its own, (b) the ASSET-CONDITIONAL read (BTC vs
large-alt vs mid/DeFi vs meme vs stablecoin), (c) every feature with its meaning + polarity (+1 high=bullish,
-1 high=bearish, 0 contextual). For a single (asset, period, timeframe) VIEW of these features, use the decomposer:
`python -m mining.decompose --asset <SYM> --cadence <TF> --start <d> --end <d> --plots`.

## The headline: the SIGNAL x ARCHETYPE interpretation matrix
The same reading means different things by archetype (synthesized from the derivatives-mechanics + archetype scouts):

| Signal | BTC (beta/reserve) | ETH/large-cap L1 | Mid-cap / DeFi | MEME | Stablecoin |
|---|---|---|---|---|---|
| **Funding rate** | crowded institutional leverage; strong contrarian at extremes | same, noisier; narrative-driven spikes revert faster | thin perp -> a 'high funding' may be ONE whale, not crowding | often manipulated; may be the insider short-leg of a pump | N/A -- deviation = depeg/credit signal, not direction |
| **Basis / premium** | reliable leverage gauge (CME+perp) | reliable, faster-moving | perp-only, noisy | meaningless multi-day | deviation = depeg risk |
| **Open interest** | true leverage thermometer ($60-70B) | reliable, ~1/3 of BTC | fragile (single-actor) | unreliable; vanishes on rugs | N/A |
| **Liquidations** | causal + BOUNDED -> contrarian bounce | larger cascades (higher beta) | violent/fast (thin) | usually TERMINAL, not a bounce | protocol-specific only |
| **Whale flow** | institutional, trackable, gameable | protocol/staking flows | TVL-flow = real; spot ambiguous | ADVERSARIAL (insider exit) | treasury/peg arb |
| **Social / attention** | coincident-lagging noise | leads 1-3d on narratives | partnership/audit = content | IS the signal (peak = exit) | depeg-rumor tail only |
| **Order-book depth** | real liquidity gauge | moderate | thin, spoofable | synthetic/spoofed | deep; deterioration = depeg |
| **Implied vol (DVOL)** | high (Deribit) | high (Deribit) | N/A | N/A | N/A |

**Cross-cutting rule:** trust order-flow / depth / whale / positioning signals IN PROPORTION TO LIQUIDITY DEPTH. They
are information on BTC and a manipulation surface on memes. Stablecoins invert all return-based logic (peg process).

---

## Price structure & trend  (`structure`, 7 features)
*Question it answers:* Where is price relative to its own history and moving averages -- trending, ranging, stretched?

**On its own:** Where price sits relative to its own trend/structure + how trendy vs choppy the tape is.

**Asset-conditional:** Math is archetype-invariant, but RELIABILITY scales with depth: trend/Hurst/MA-distance are clean on BTC/ETH (deep, continuous) and gappy/whipsawy on memes (thin, reflexive). On stablecoins all price-structure features are near-zero noise around the peg (ignore).

| feature | meaning | polarity |
|---|---|---|
| `norm_ma_distance` -- Distance to moving average | How far price sits above/below its moving average, in sigma. High = stretched up; low = stretched down. | +1 bullish-high |
| `xd_ma_distance` -- Cross MA-distance | Stretch vs MA relative to the cross-section. | +1 bullish-high |
| `norm_deviation` -- Price deviation | Deviation of price from its local trend baseline. | +1 bullish-high |
| `norm_efficiency` -- Trend efficiency (fractal) | Kaufman-style efficiency: how directional vs choppy the path is. High = clean trend; low = chop/range. | 0 contextual |
| `norm_fd_close` -- Fractal dimension | Path roughness; high = jagged/mean-reverting texture. | 0 contextual |
| `norm_perm_entropy` -- Permutation entropy | Complexity/randomness of the recent path. High = disordered; low = structured/predictable texture. | 0 contextual |
| `hurst_regime` -- Hurst regime | Persistence label: >0.5 trending, ~0.5 random walk, <0.5 mean-reverting. | 0 contextual |

## Momentum & returns  (`momentum`, 8 features)
*Question it answers:* Which way and how hard has price been moving, and is that move accelerating or exhausting?

**On its own:** Direction + speed of recent price/flow drift across horizons.

**Asset-conditional:** BTC momentum is institutional + slow (ETF/treasury flows); large-alt momentum is narrative-reflexive (ETF/staking rumors); MEME momentum is pure attention reflexivity -- violent, self-fulfilling, and the exit is when attention peaks. Same return value, opposite durability.

| feature | meaning | polarity |
|---|---|---|
| `norm_return_1` -- Return (fast) | Most-recent-bar normalized return. Sign = direction. | +1 bullish-high |
| `norm_return_4` -- Return (mid) | 4-bar normalized return. | +1 bullish-high |
| `norm_return_16` -- Return (slow) | 16-bar normalized return = the prevailing drift. | +1 bullish-high |
| `norm_momentum_accel` -- Momentum acceleration | Is momentum building or fading (2nd derivative). High = accelerating up. | +1 bullish-high |
| `norm_return_kurtosis` -- Return kurtosis | Tail-heaviness of recent returns. High = jumpy/fat-tailed regime. | 0 contextual |
| `norm_flow_persistence` -- Flow persistence | How autocorrelated recent order flow is. | +1 bullish-high |
| `xd_momentum_rank` -- Cross momentum rank | This asset's momentum percentile vs the cross-section. High = a relative leader. | +1 bullish-high |
| `norm_funding_momentum` -- Funding momentum *(crypto-specific)* | Trend/momentum in the funding rate series; rising = crowd increasing conviction. | -1 bearish-high |

## Volatility & activity  (`volatility`, 15 features)
*Question it answers:* Is the market compressed or expanded, calm or violent, and is volatility clustering?

**On its own:** How big moves are + are getting (realized, jumps, implied) -- the predictable channel.

**Asset-conditional:** DVOL/implied-vol features exist ONLY for BTC/ETH (Deribit) -- null elsewhere. Realized-vol/jump features are universal but the BASELINE differs ~4x by tier (BTC ~50-60% ann -> meme >200%); a 'high vol' z-score must be read against the asset's own regime, not a global bar.

| feature | meaning | polarity |
|---|---|---|
| `norm_yz_volatility` -- Yang-Zhang volatility | Range-based realized volatility. High = violent/expanded; low = compressed (coil). | 0 contextual |
| `norm_vol_cluster` -- Volatility clustering | Is vol clustering (GARCH-like persistence). | 0 contextual |
| `norm_vol_ratio` -- Vol ratio (fast/slow) | Short-vs-long vol. High = vol expanding now relative to baseline. | 0 contextual |
| `norm_vol_price_corr` -- Vol-price correlation | Leverage effect: negative = vol rises as price falls (typical risk-off). | -1 bearish-high |
| `norm_log_volume` -- Log volume | Activity level. High = heavy participation. | 0 contextual |
| `norm_bar_duration` -- Bar duration | For event bars: how long this bar took to form. | 0 contextual |
| `dv_dvol_close` -- DVOL index close *(crypto-specific)* | Deribit implied-vol index close; high = options market pricing elevated uncertainty. | 0 contextual |
| `dv_dvol_high` -- DVOL index high *(crypto-specific)* | Intrabar high of DVOL; captures IV spike during the bar. | 0 contextual |
| `dv_dvol_low` -- DVOL index low *(crypto-specific)* | Intrabar low of DVOL; lower = options market relatively calm. | 0 contextual |
| `rv_rv_5m` -- Realized variance (5-min) *(crypto-specific)* | 5-minute realized variance (continuous component + jumps); high = active bar. | 0 contextual |
| `rv_bpv_5m` -- Bipower variation (5-min) *(crypto-specific)* | Jump-robust realized variance (continuous only); separates smooth from jump vol. | 0 contextual |
| `rv_jv_5m` -- Jump variation (5-min) *(crypto-specific)* | RV minus bipower = jump-only variance; high = price jumped in this bar. | 0 contextual |
| `rv_jump_frac` -- Jump fraction *(crypto-specific)* | Jump variation as pct of total RV; high = most of the move was a jump (gap/news). | 0 contextual |
| `rv_jump_count` -- Jump count *(crypto-specific)* | Number of statistically significant intrabar jumps. | 0 contextual |
| `rv_jump_signed_var` -- Signed jump variation *(crypto-specific)* | Jump variance with sign of the jump; positive = up-jump, negative = down-jump. | +1 bullish-high |

## Order flow & microstructure  (`orderflow`, 25 features)
*Question it answers:* Who is in control of the tape -- aggressive buyers or sellers -- and how toxic/intense is the flow?

**On its own:** Who is pressing -- aggressive buy vs sell flow, trade intensity, informed-flow toxicity.

**Asset-conditional:** Hawkes/VPIN/flow-imbalance/tick features are GENUINE microstructure on BTC (real two-sided flow) but MANIPULABLE on thin meme/micro books (wash trading, spoof-driven taker prints). Trust order-flow signals in proportion to book depth; on memes treat them as manipulation surface, not information.

| feature | meaning | polarity |
|---|---|---|
| `norm_vpin` -- VPIN (flow toxicity) *(crypto-specific)* | Volume-synchronized probability of informed trading. High = toxic/one-sided flow = informed pressure. | 0 contextual |
| `norm_flow_imbalance` -- Order-flow imbalance *(crypto-specific)* | Signed aggressor imbalance. High = buyers lifting offers; low = sellers hitting bids. | +1 bullish-high |
| `norm_hawkes_intensity` -- Hawkes intensity *(crypto-specific)* | Self-exciting trade arrival rate (clustering). | 0 contextual |
| `norm_hawkes_buy_intensity` -- Hawkes buy intensity *(crypto-specific)* | Self-exciting BUY arrival intensity. | +1 bullish-high |
| `norm_hawkes_sell_intensity` -- Hawkes sell intensity *(crypto-specific)* | Self-exciting SELL arrival intensity. | -1 bearish-high |
| `norm_hawkes_imbalance` -- Hawkes imbalance *(crypto-specific)* | Buy-vs-sell intensity imbalance. | +1 bullish-high |
| `norm_kyle_lambda` -- Kyle's lambda (impact) *(crypto-specific)* | Price impact per unit flow = illiquidity. High = thin/impactful; small flow moves price. | 0 contextual |
| `norm_tick_count` -- Tick count | Number of trades; participation/fragmentation of the tape. | 0 contextual |
| `hbr_eta_total` -- Hawkes branching ratio (total) *(crypto-specific)* | Aggregate excitability of trade arrivals; close to 1 = near-critical self-exciting regime. | 0 contextual |
| `hbr_eta_buy` -- Hawkes branching ratio (buys) *(crypto-specific)* | Buy-side excitability; high = buy flow is self-reinforcing (momentum). | +1 bullish-high |
| `hbr_eta_sell` -- Hawkes branching ratio (sells) *(crypto-specific)* | Sell-side excitability; high = sell flow is self-reinforcing (momentum down). | -1 bearish-high |
| `hbr_eta_imbalance` -- Hawkes branching imbalance *(crypto-specific)* | Signed buy-minus-sell excitability. Positive = buy side more self-exciting. | +1 bullish-high |
| `hbr_n_trades` -- Hawkes trade count *(crypto-specific)* | Number of trades used in Hawkes fit; proxy for tape activity intensity. | 0 contextual |
| `buy_vol` -- Buy volume | Aggressor buy volume for the bar; higher = demand pressure. | +1 bullish-high |
| `sell_vol` -- Sell volume | Aggressor sell volume for the bar; higher = supply pressure. | -1 bearish-high |
| `tick_count` -- Tick count | Raw trade count; high = active/fragmented tape. | 0 contextual |
| `norm_spread_bps` -- Quoted spread (bps) | Bid-ask spread in basis points; high = thin/illiquid book. | 0 contextual |
| `norm_hl_spread` -- High-low spread | Bar high-low range as a fraction; measures realized intrabar volatility/whip. | 0 contextual |
| `norm_cs_spread` -- Cross-spread (composite) *(crypto-specific)* | Composite spread signal across venues; elevated = fragmented/stressed liquidity. | 0 contextual |
| `lob_spread_bps_mean` -- LOB quoted spread mean (bps) *(crypto-specific)* | Average bid-ask spread over the bar; high = costly to trade/thin book. | 0 contextual |
| `lob_spread_bps_p90` -- LOB quoted spread p90 (bps) *(crypto-specific)* | 90th-percentile spread over the bar; captures worst-case execution cost. | 0 contextual |
| `lob_kyle_lambda_mean` -- LOB Kyle-lambda mean *(crypto-specific)* | Mean price impact per unit flow estimated from the LOB; high = illiquid. | 0 contextual |
| `lob_kyle_lambda_abs_max` -- LOB Kyle-lambda abs max *(crypto-specific)* | Peak price impact event within the bar; captures flash liquidity shocks. | 0 contextual |
| `lob_bgf_kyle_lambda_mean` -- BGF Kyle-lambda mean *(crypto-specific)* | Price impact from BGF composite venues; high = those venues are illiquid. | 0 contextual |
| `lob_bgf_spread_bps_mean` -- BGF quoted spread mean (bps) *(crypto-specific)* | Mean spread from BGF composite; wider = off-exchange execution is costly. | 0 contextual |

## Liquidity & order book  (`liquidity`, 33 features)
*Question it answers:* How deep and balanced is the book, and is it thinning out (fragile) right now?

**On its own:** How deep/tight/fragile the book is -- the cost-of-execution + slippage gauge.

**Asset-conditional:** LOB depth / spread / venue-count are a real liquidity gauge on BTC/ETH (multi-layer books) and largely SYNTHETIC on memes (spoofed walls, 10-30% spreads in thin hours -- a '$500k wall' may be the entire visible book). Depth deterioration on a stablecoin's book is a depeg warning.

| feature | meaning | polarity |
|---|---|---|
| `bd_imbalance_l1` -- Book imbalance L1 *(crypto-specific)* | Top-of-book bid/ask size imbalance. High = bid-heavy (support); low = ask-heavy (resistance). | +1 bullish-high |
| `bd_imbalance_l5` -- Book imbalance L5 *(crypto-specific)* | 5-level depth imbalance. | +1 bullish-high |
| `bd_thin_book_frac` -- Thin-book fraction *(crypto-specific)* | Fraction of time the book was thin. High = fragile, gap-prone book. | 0 contextual |
| `bd_total_depth_l5_mean` -- Total depth L5 *(crypto-specific)* | Aggregate resting depth. Low = fragile. | 0 contextual |
| `bd_notional_skew` -- Notional skew *(crypto-specific)* | Skew of resting notional bid vs ask. | +1 bullish-high |
| `xex_cb_bn_spread_bps` -- Coinbase-Binance spread (bps) *(crypto-specific)* | Price spread between Coinbase and Binance; elevated = cross-venue stress/arbitrage. | 0 contextual |
| `xex_by_bn_spread_bps` -- Bybit-Binance spread (bps) *(crypto-specific)* | Price spread between Bybit and Binance; elevated = fragmented liquidity. | 0 contextual |
| `xex_ok_bn_spread_bps` -- OKX-Binance spread (bps) *(crypto-specific)* | Price spread between OKX and Binance; elevated = cross-venue dislocation. | 0 contextual |
| `xex_cb_bn_spread_bps_right` -- Coinbase-Binance spread (lagged) *(crypto-specific)* | Prior-period Coinbase-Binance spread; used to detect persistent dislocation. | 0 contextual |
| `xex_by_bn_spread_bps_right` -- Bybit-Binance spread (lagged) *(crypto-specific)* | Prior-period Bybit-Binance spread. | 0 contextual |
| `xex_ok_bn_spread_bps_right` -- OKX-Binance spread (lagged) *(crypto-specific)* | Prior-period OKX-Binance spread. | 0 contextual |
| `xex_cb_bn_z30` -- Coinbase-Binance spread z-score *(crypto-specific)* | Spread vs 30-bar norm; high z = unusual dislocation, potential cascade risk. | 0 contextual |
| `xex_spread_dispersion` -- Cross-exchange spread dispersion *(crypto-specific)* | Variance of spreads across venue pairs; high = market is fragmented/dislocated. | 0 contextual |
| `xex_max_abs_spread` -- Max absolute cross-exchange spread *(crypto-specific)* | Largest single spread across monitored venue pairs; extreme = stress peak. | 0 contextual |
| `xex_n_venues_active` -- Venues active count *(crypto-specific)* | Number of exchanges with active markets; lower = liquidity concentration risk. | 0 contextual |
| `lob_l1_imb_mean` -- LOB L1 imbalance mean *(crypto-specific)* | Mean top-of-book bid/ask imbalance; high = persistent bid pressure. | +1 bullish-high |
| `lob_l1_imb_std` -- LOB L1 imbalance std *(crypto-specific)* | Variability of top-of-book imbalance; high = unstable/rapidly flipping book. | 0 contextual |
| `lob_l5_imb_mean` -- LOB L5 imbalance mean *(crypto-specific)* | Mean 5-level depth imbalance; broader view of order-book skew. | +1 bullish-high |
| `lob_l5_imb_std` -- LOB L5 imbalance std *(crypto-specific)* | Variability of 5-level depth imbalance. | 0 contextual |
| `lob_top_pressure_mean` -- LOB top-of-book pressure mean *(crypto-specific)* | Mean notional pressure at best bid/ask; high = strong resting order wall. | 0 contextual |
| `lob_count_imb_mean` -- LOB order-count imbalance mean *(crypto-specific)* | Imbalance by order count (not notional); captures HFT quote skew. | +1 bullish-high |
| `lob_run_length_p50` -- LOB run length median *(crypto-specific)* | Median consecutive-same-sign depth snapshots; high = persistent book skew. | 0 contextual |
| `lob_n_bars` -- LOB snapshot count | Number of book snapshots in the bar; lower = sparser data coverage. | 0 contextual |
| `bd_depth_l1pct_mean` -- Book depth L1 pct mean *(crypto-specific)* | Mean depth at L1 as pct of total; higher = concentrated near-touch liquidity. | 0 contextual |
| `bd_depth_l1pct_p90` -- Book depth L1 pct p90 *(crypto-specific)* | 90th-pct L1-depth concentration; captures thin-book tail events. | 0 contextual |
| `bd_notional_l1pct_mean` -- Notional depth L1 pct mean *(crypto-specific)* | Mean notional concentration at L1; proxy for maker-side commitment near touch. | 0 contextual |
| `bd_total_depth_l5_p10` -- Total depth L5 p10 *(crypto-specific)* | 10th-pct of 5-level depth; captures worst-case book depth during the bar. | 0 contextual |
| `bd_depth_at_02pct` -- Depth at 0.2% from mid *(crypto-specific)* | Resting notional within 20bp of mid; directly translates to market-impact capacity. | 0 contextual |
| `bd_n_snapshots` -- Book snapshot count | Number of book snapshots taken; lower = sparse coverage. | 0 contextual |
| `bd_bgf_imbalance_l1` -- BGF book imbalance L1 *(crypto-specific)* | Bybit/Gate/FTX-composite top-of-book imbalance; cross-venue bid/ask skew. | +1 bullish-high |
| `lob_bgf_l1_imb_mean` -- BGF LOB L1 imbalance mean *(crypto-specific)* | Mean top-of-book imbalance from BGF composite; supplementary venue signal. | +1 bullish-high |
| `lob_bgf_top_pressure_mean` -- BGF top-of-book pressure mean *(crypto-specific)* | Mean notional at best bid/ask across BGF venues. | 0 contextual |
| `lob_bgf_count_imb_mean` -- BGF order-count imbalance mean *(crypto-specific)* | Mean order-count imbalance from BGF composite; HFT skew on secondary venues. | +1 bullish-high |

## Funding, open interest & basis  (`derivatives`, 28 features)
*Question it answers:* What are leveraged traders paying to hold, how crowded is the perp, and is the futures basis stressed?

**On its own:** Leverage + positioning + carry -- funding, basis, OI, premium (the reflexivity fuel).

**Asset-conditional:** THE most archetype-dependent family. FUNDING: on BTC = crowded institutional leverage with real mean-reversion at extremes (contrarian); on a MEME = often one whale dominating a thin perp, or the insider short-leg of a pump-and-dump. BASIS/premium: reliable leverage gauge on BTC (CME+perp), noise on memes (perp-only). OI: a true leverage thermometer on BTC; on memes it can be one actor and vanish overnight. On stablecoins, any funding/basis deviation = a depeg/credit signal, not direction.

| feature | meaning | polarity |
|---|---|---|
| `norm_funding` -- Funding rate *(crypto-specific)* | Perp funding. High positive = longs paying (crowded long, often a fade); negative = shorts paying. | -1 bearish-high |
| `fund_rate_z30` -- Funding z-score *(crypto-specific)* | Funding vs its 30-bar norm. Extreme = crowded. | -1 bearish-high |
| `fund_sign_flip` -- Funding sign flip *(crypto-specific)* | Funding flipped sign = positioning regime change. | 0 contextual |
| `norm_oi_change` -- OI change *(crypto-specific)* | Open-interest change. Rising OI + rising price = new longs. | +1 bullish-high |
| `norm_oi_price_divergence` -- OI-price divergence *(crypto-specific)* | OI and price disagreeing = potential squeeze fuel. | 0 contextual |
| `bs_basis_pct` -- Futures basis % *(crypto-specific)* | Perp/quarterly basis. High = bullish carry/greed. | +1 bullish-high |
| `bs_basis_z30` -- Basis z-score *(crypto-specific)* | Basis vs norm; extreme = stressed leverage. | +1 bullish-high |
| `bs_basis_panic` -- Basis panic *(crypto-specific)* | Basis collapse flag = deleveraging/risk-off. | -1 bearish-high |
| `bs_basis_frenzy` -- Basis frenzy *(crypto-specific)* | Basis blow-off flag = leveraged greed. | +1 bullish-high |
| `bs_basis_delta_1d` -- Basis 1-day change *(crypto-specific)* | Change in futures basis over 1 day; rising = carry/leverage expanding. | +1 bullish-high |
| `bs_basis_delta_3d` -- Basis 3-day change *(crypto-specific)* | Change in futures basis over 3 days; trend in carry appetite. | +1 bullish-high |
| `bs_basis_xsec_z` -- Basis cross-section z *(crypto-specific)* | This asset's basis vs the cross-section; high = outsized carry vs peers. | +1 bullish-high |
| `bs_basis_bull_shock` -- Basis bull shock *(crypto-specific)* | Sudden spike up in basis = leveraged-long rush; often precedes local top. | -1 bearish-high |
| `bs_basis_bear_shock` -- Basis bear shock *(crypto-specific)* | Sudden collapse in basis = deleveraging/panic exit; contrarian signal. | +1 bullish-high |
| `fund_rate_mean` -- Funding rate mean (cross-venue) *(crypto-specific)* | Average funding rate across venues; high positive = crowded long, fade signal. | -1 bearish-high |
| `fund_rate_max` -- Funding rate max (cross-venue) *(crypto-specific)* | Maximum funding across venues; captures the most extreme long-crowd venue. | -1 bearish-high |
| `fund_rate_min` -- Funding rate min (cross-venue) *(crypto-specific)* | Minimum funding across venues; deeply negative = crowded short, squeeze fuel. | +1 bullish-high |
| `fund_rate_abs_mean` -- Funding absolute mean *(crypto-specific)* | Mean absolute funding; high = strong directional crowding (either side). | 0 contextual |
| `fund_extreme_long_count` -- Extreme-long funding count *(crypto-specific)* | Number of venues with extreme positive funding; high = broadly crowded long. | -1 bearish-high |
| `fund_extreme_short_count` -- Extreme-short funding count *(crypto-specific)* | Number of venues with extreme negative funding; high = broadly crowded short. | +1 bullish-high |
| `fund_avg_apr` -- Funding APR *(crypto-specific)* | Annualized funding rate; high positive = expensive to hold long (bearish fade). | -1 bearish-high |
| `fund_n_settlements` -- Funding settlement count *(crypto-specific)* | Number of 8h settlements in the bar; normally 1 for 8h bars, <1 = data gap. | 0 contextual |
| `premium_vol30` -- Premium volatility (30-bar) *(crypto-specific)* | Volatility of the perp basis premium; high = unstable carry regime. | 0 contextual |
| `premium_persistence30` -- Premium persistence (30-bar) *(crypto-specific)* | Autocorrelation of premium; high = carry regime is sticky. | +1 bullish-high |
| `premium_extreme_count30` -- Premium extreme count (30-bar) *(crypto-specific)* | Count of extreme premium readings; elevated = repeated leverage surges. | 0 contextual |
| `premium_z90` -- Premium z-score (90-bar) *(crypto-specific)* | Premium vs 90-bar norm; extreme = stretched carry vs historical baseline. | 0 contextual |
| `premium_apr` -- Premium APR *(crypto-specific)* | Annualized premium yield; equivalent to funding APR on the spot-futures basis. | -1 bearish-high |
| `fp_fund_panel` -- Funding panel flag *(crypto-specific)* | Binary/count indicating asset is in the multi-venue funding data panel. | 0 contextual |

## Liquidations & forced flow  (`liquidation`, 13 features)
*Question it answers:* Is there forced selling/buying (cascades, capitulation, short squeezes) happening or building?

**On its own:** Forced flow -- where leverage is breaking and which way it overshoots.

**Asset-conditional:** On BTC/ETH liquidation cascades are causal-but-BOUNDED (deep spot support) -> the post-cascade bounce is a contrarian-long regularity (capitulation). On thin alts/memes a liquidation is often a TERMINAL event (50%+ in minutes), NOT a bounce setup. Cross-collateral DeFi adds an on-chain liquidation layer for DeFi tokens.

| feature | meaning | polarity |
|---|---|---|
| `liq_long_usd` -- Long liquidations $ *(crypto-specific)* | Forced long sells. Spikes mark down-cascades. | -1 bearish-high |
| `liq_short_usd` -- Short liquidations $ *(crypto-specific)* | Forced short buys. Spikes mark squeezes. | +1 bullish-high |
| `liq_delta_z30` -- Liquidation delta z *(crypto-specific)* | Net long-vs-short liq pressure (z). | -1 bearish-high |
| `liq_capitulation` -- Capitulation flag *(crypto-specific)* | Long-liquidation capitulation event (contrarian long). | +1 bullish-high |
| `liq_short_panic` -- Short panic flag *(crypto-specific)* | Short-side panic / squeeze event. | +1 bullish-high |
| `liq_long_spike` -- Long-liq spike *(crypto-specific)* | Outsized long liquidation burst. | -1 bearish-high |
| `liq_short_spike` -- Short-liq spike *(crypto-specific)* | Outsized short liquidation burst (squeeze). | +1 bullish-high |
| `liq_delta_usd` -- Liquidation delta (USD) *(crypto-specific)* | Net long-minus-short liquidation USD; positive = more longs getting stopped. | -1 bearish-high |
| `liq_total_usd` -- Total liquidation (USD) *(crypto-specific)* | Combined long + short liquidations; high = forced de-leveraging on both sides. | 0 contextual |
| `liq_long_z30` -- Long liquidation z-score *(crypto-specific)* | Long liquidations vs 30-bar norm; extreme = outsized forced selling event. | -1 bearish-high |
| `liq_short_z30` -- Short liquidation z-score *(crypto-specific)* | Short liquidations vs 30-bar norm; extreme = outsized squeeze event. | +1 bullish-high |
| `liq_long_xsec_z` -- Long liquidation cross-section z *(crypto-specific)* | This asset's long liq vs cross-section; idiosyncratic cascade signal. | -1 bearish-high |
| `liq_short_xsec_z` -- Short liquidation cross-section z *(crypto-specific)* | This asset's short liq vs cross-section; idiosyncratic squeeze signal. | +1 bullish-high |

## Positioning (long/short ratio, smart vs retail)  (`positioning`, 14 features)
*Question it answers:* How are accounts positioned -- crowded long or short, smart money vs retail?

**On its own:** Who is on which side -- long/short ratios, smart vs retail, taker imbalance.

**Asset-conditional:** Top-trader LSR / smart-vs-retail divergence is meaningful on BTC (deep, real institutional participants). On memes 'top traders' may be a few coordinated wallets -> the smart-money read breaks down; taker imbalance on thin books is easily manufactured by a single whale testing liquidity.

| feature | meaning | polarity |
|---|---|---|
| `s3_global_lsr` -- Global long/short ratio *(crypto-specific)* | All-account LSR. High = crowd is long. | -1 bearish-high |
| `s3_top_pos_lsr` -- Top-trader position LSR *(crypto-specific)* | Top traders' positioning. High = big long. | +1 bullish-high |
| `s3_taker_lsr` -- Taker LSR *(crypto-specific)* | Aggressor buy/sell ratio. | +1 bullish-high |
| `s3_smart_vs_retail` -- Smart vs retail *(crypto-specific)* | Top-trader minus retail positioning gap. | +1 bullish-high |
| `s3_smart_bullish` -- Smart bullish *(crypto-specific)* | Smart money net bullish flag. | +1 bullish-high |
| `s3_smart_extreme_long` -- Smart extreme long *(crypto-specific)* | Smart money crowded long (fade risk). | -1 bearish-high |
| `s3_global_lsr_z` -- Global LSR z *(crypto-specific)* | Crowd long/short extremity (z). | -1 bearish-high |
| `s3_oi_usd` -- Open interest (USD) *(crypto-specific)* | Total open interest in USD; rising = new leverage entering (watch direction). | 0 contextual |
| `s3_top_acct_lsr` -- Top-account long/short ratio *(crypto-specific)* | Largest accounts' LSR; more reliable than global LSR (less noise). | +1 bullish-high |
| `s3_smart_bearish` -- Smart bearish flag *(crypto-specific)* | Smart money net bearish signal; top traders leaning short. | -1 bearish-high |
| `s3_smart_extreme_short` -- Smart extreme short *(crypto-specific)* | Smart money at extreme short (contrarian: squeeze potential). | +1 bullish-high |
| `s3_smart_vs_retail_z` -- Smart vs retail z-score *(crypto-specific)* | Z-scored divergence between smart and retail positioning; extreme = max divergence. | +1 bullish-high |
| `s3_top_pos_lsr_z` -- Top-trader position LSR z-score *(crypto-specific)* | Top-trader LSR vs its own norm; captures regime shifts in smart positioning. | +1 bullish-high |
| `s3_top_pos_lsr_xsec_z` -- Top-trader position LSR cross-section z *(crypto-specific)* | This asset's top-trader LSR vs cross-section; isolates idiosyncratic smart positioning. | +1 bullish-high |

## Whale & large-trade flow  (`whale`, 6 features)
*Question it answers:* Are large players net buying or selling via outsized prints?

**On its own:** Large-actor flow -- accumulation vs distribution (institutional on BTC, insider on memes).

**Asset-conditional:** On BTC whale flow is institutional + trackable but gameable; ~15-40% of alerts are internal/custodial transfers (noise). On MEMES whale flow is ADVERSARIAL -- the whale is typically the deployer/insider executing an exit; 'whale accumulation' is frequently the pump leg before the dump. DeFi: TVL-flow whales carry genuine fundamental signal (protocol inflow = accumulation).

| feature | meaning | polarity |
|---|---|---|
| `wh_whale_net_usd` -- Whale net $ *(crypto-specific)* | Net large-trade USD flow. Positive = whales net buying. | +1 bullish-high |
| `wh_whale_buy_usd` -- Whale buy $ *(crypto-specific)* | Large-print buying. | +1 bullish-high |
| `wh_whale_sell_usd` -- Whale sell $ *(crypto-specific)* | Large-print selling. | -1 bearish-high |
| `norm_whale` -- Whale (normalized) *(crypto-specific)* | Normalized whale-flow signal. | +1 bullish-high |
| `wh_whale_trade_count` -- Whale trade count *(crypto-specific)* | Number of large-print trades; high = many whale participants active. | 0 contextual |
| `wh_whale_trade_count_500k` -- Whale trade count (>$500k) *(crypto-specific)* | Count of trades exceeding $500k notional; captures true institutional-size flow. | 0 contextual |

## Cross-asset & relative context  (`cross_asset`, 58 features)
*Question it answers:* What is BTC (the market beta) doing, and how does this asset rank cross-sectionally right now?

**On its own:** How the asset sits vs BTC + the rest of the universe -- beta + cross-sectional rank.

**Asset-conditional:** These encode BTC-beta + cross-sectional RANK. For BTC they are self-referential (~identity). For large-alts they capture the ~65-70% BTC-led component. For memes BTC explains <30% of variance -> cross-asset features under-describe them (idiosyncratic reflexivity dominates). Cross-sectional rank (xrel_*) is where the 1-week reversal + dispersion edge lives.

| feature | meaning | polarity |
|---|---|---|
| `xd_btc_return` -- BTC return | Market beta: what BTC is doing right now. | +1 bullish-high |
| `xd_btc_volatility` -- BTC volatility | BTC vol = market-wide risk temperature. | 0 contextual |
| `xd_cross_return_mean` -- Cross return mean | Average move across the universe (breadth). | +1 bullish-high |
| `xd_funding_spread` -- Funding spread *(crypto-specific)* | This asset's funding vs the cross-section. | -1 bearish-high |
| `xrel_rv_rv_5m_xrank` -- RV (5m) cross-rank *(crypto-specific)* | Rank of this asset's 5-min realized vol across the universe; high = most volatile. | 0 contextual |
| `xrel_rv_rv_5m_xpct10` -- RV (5m) cross-pct10 *(crypto-specific)* | Pct of peers with lower 5m RV; high = relatively volatile. | 0 contextual |
| `xrel_rv_rv_5m_xratio` -- RV (5m) cross-ratio *(crypto-specific)* | This asset's RV vs cross-section mean; >1 = vol premium vs peers. | 0 contextual |
| `xrel_rv_bpv_5m_xrank` -- BPV (5m) cross-rank *(crypto-specific)* | Rank of this asset's bipower variation; separates jump risk from continuous vol. | 0 contextual |
| `xrel_rv_bpv_5m_xpct10` -- BPV (5m) cross-pct10 *(crypto-specific)* | Pct of peers with lower bipower variation. | 0 contextual |
| `xrel_rv_bpv_5m_xratio` -- BPV (5m) cross-ratio *(crypto-specific)* | BPV relative to cross-section mean. | 0 contextual |
| `xrel_hbr_eta_total_xrank` -- HBR excitability cross-rank *(crypto-specific)* | Rank of Hawkes branching ratio vs peers; high = most self-exciting tape. | 0 contextual |
| `xrel_hbr_eta_total_xpct10` -- HBR excitability cross-pct10 *(crypto-specific)* | Pct of peers with lower Hawkes excitability. | 0 contextual |
| `xrel_hbr_eta_total_xratio` -- HBR excitability cross-ratio *(crypto-specific)* | This asset's Hawkes excitability vs cross-section mean. | 0 contextual |
| `xrel_hbr_n_trades_xrank` -- HBR trade-count cross-rank *(crypto-specific)* | Rank of trade count vs the universe; high = most active tape. | 0 contextual |
| `xrel_hbr_n_trades_xpct10` -- HBR trade-count cross-pct10 *(crypto-specific)* | Pct of peers with lower trade count. | 0 contextual |
| `xrel_hbr_n_trades_xratio` -- HBR trade-count cross-ratio *(crypto-specific)* | Trade count relative to cross-section mean. | 0 contextual |
| `xrel_liq_long_usd_xrank` -- Long-liquidation cross-rank *(crypto-specific)* | Rank of long-liquidation intensity vs peers; high = most deleveraged. | -1 bearish-high |
| `xrel_liq_long_usd_xpct10` -- Long-liquidation cross-pct10 *(crypto-specific)* | Pct of peers with lower long-liquidation intensity. | -1 bearish-high |
| `xrel_liq_long_usd_xratio` -- Long-liquidation cross-ratio *(crypto-specific)* | This asset's long liq vs cross-section mean; >1 = idiosyncratic selling. | -1 bearish-high |
| `xrel_wh_whale_net_usd_xrank` -- Whale net cross-rank *(crypto-specific)* | Rank of whale net flow vs the universe; high = most whale-bought. | +1 bullish-high |
| `xrel_wh_whale_net_usd_xpct10` -- Whale net cross-pct10 *(crypto-specific)* | Pct of peers with lower whale net flow. | +1 bullish-high |
| `xrel_wh_whale_net_usd_xratio` -- Whale net cross-ratio *(crypto-specific)* | This asset's whale net vs cross-section mean. | +1 bullish-high |
| `xrel_lob_kyle_lambda_mean_xrank` -- LOB Kyle-lambda cross-rank *(crypto-specific)* | Rank of price-impact vs the universe; high = most illiquid. | 0 contextual |
| `xrel_lob_kyle_lambda_mean_xpct10` -- LOB Kyle-lambda cross-pct10 *(crypto-specific)* | Pct of peers with lower price impact. | 0 contextual |
| `xrel_lob_kyle_lambda_mean_xratio` -- LOB Kyle-lambda cross-ratio *(crypto-specific)* | This asset's price impact vs cross-section mean. | 0 contextual |
| `xd_cross_vol_mean` -- Cross volatility mean | Average realized volatility across the universe; high = risk-on/volatile crypto environment. | 0 contextual |
| `te_in` -- Transfer entropy in *(crypto-specific)* | Information flowing INTO this asset from others; high = being led by peers. | 0 contextual |
| `te_out` -- Transfer entropy out *(crypto-specific)* | Information flowing OUT of this asset to others; high = this asset is leading. | 0 contextual |
| `te_in_btc` -- Transfer entropy in (BTC) *(crypto-specific)* | Information flowing from BTC specifically; high = this asset is BTC-led. | 0 contextual |
| `te_out_btc` -- Transfer entropy out (BTC) *(crypto-specific)* | Information flowing from this asset toward BTC; high = this asset is leading BTC. | 0 contextual |
| `te_imb` -- Transfer entropy imbalance *(crypto-specific)* | te_out minus te_in; positive = net information leader in the network. | 0 contextual |
| `te_btc_imb` -- Transfer entropy imbalance (BTC pair) *(crypto-specific)* | TE imbalance vs BTC specifically; positive = leading BTC. | 0 contextual |
| `etf_btc_etf_total_usdm` -- BTC ETF AUM (USD millions) *(crypto-specific)* | Total BTC spot-ETF AUM; rising = institutional accumulation. | +1 bullish-high |
| `etf_btc_etf_total_z30` -- BTC ETF AUM z-score (30-bar) *(crypto-specific)* | ETF AUM vs 30-bar norm; high = rapid institutional inflow. | +1 bullish-high |
| `etf_btc_etf_total_7d_z` -- BTC ETF AUM z-score (7-day) *(crypto-specific)* | Short-horizon ETF AUM z; captures fresh inflow/outflow momentum. | +1 bullish-high |
| `etf_btc_etf_inflow_shock` -- BTC ETF inflow shock *(crypto-specific)* | Binary: large BTC ETF inflow event; strong institutional demand pulse. | +1 bullish-high |
| `etf_btc_etf_outflow_shock` -- BTC ETF outflow shock *(crypto-specific)* | Binary: large BTC ETF outflow event; institutional selling pressure. | -1 bearish-high |
| `etf_btc_etf_mega_inflow` -- BTC ETF mega inflow *(crypto-specific)* | Binary: extreme BTC ETF inflow (top-tail); rare but high-impact demand shock. | +1 bullish-high |
| `etf_btc_etf_mega_outflow` -- BTC ETF mega outflow *(crypto-specific)* | Binary: extreme BTC ETF outflow; rare institutional distribution event. | -1 bearish-high |
| `etf_eth_etf_total_usdm` -- ETH ETF AUM (USD millions) *(crypto-specific)* | Total ETH spot-ETF AUM; correlated institutional demand signal. | +1 bullish-high |
| `etf_eth_etf_total_z30` -- ETH ETF AUM z-score (30-bar) *(crypto-specific)* | ETH ETF AUM vs norm; ETH institutional positioning proxy. | +1 bullish-high |
| `etf_eth_etf_inflow_shock` -- ETH ETF inflow shock *(crypto-specific)* | Binary: large ETH ETF inflow; correlated with broad crypto risk appetite. | +1 bullish-high |
| `etf_eth_etf_outflow_shock` -- ETH ETF outflow shock *(crypto-specific)* | Binary: large ETH ETF outflow; correlated risk-off signal. | -1 bearish-high |
| `etf_any_inflow_shock` -- Any-ETF inflow shock *(crypto-specific)* | Binary: at least one BTC or ETH ETF inflow shock; broad institutional demand. | +1 bullish-high |
| `etf_both_inflow_shock` -- Both-ETF inflow shock *(crypto-specific)* | Binary: simultaneous BTC and ETH ETF inflow; highest-conviction institutional buying. | +1 bullish-high |
| `stbl_total_zscore_30d` -- Stablecoin supply z-score (30d) *(crypto-specific)* | Total stablecoin supply vs 30-day norm; high = fresh dry powder entering crypto. | +1 bullish-high |
| `stbl_total_delta_7d_pct` -- Stablecoin supply 7d change % *(crypto-specific)* | 7-day pct change in stablecoin supply; positive = liquidity expanding. | +1 bullish-high |
| `stbl_total_delta_30d_pct` -- Stablecoin supply 30d change % *(crypto-specific)* | 30-day pct change in stablecoin supply; trend in crypto dry-powder. | +1 bullish-high |
| `stbl_usdt_zscore_30d` -- USDT supply z-score (30d) *(crypto-specific)* | USDT supply vs norm; dominant stablecoin inflow = bullish fuel. | +1 bullish-high |
| `stbl_usdt_delta_7d_pct` -- USDT supply 7d change % *(crypto-specific)* | 7-day USDT supply change; fresh USDT minting = demand signal. | +1 bullish-high |
| `stbl_usdc_zscore_30d` -- USDC supply z-score (30d) *(crypto-specific)* | USDC supply vs norm; institutional stablecoin expansion. | +1 bullish-high |
| `stbl_usde_zscore_30d` -- USDe supply z-score (30d) *(crypto-specific)* | Ethena USDe supply vs norm; yield-bearing stablecoin demand (crypto-native). | +1 bullish-high |
| `stbl_dai_zscore_30d` -- DAI supply z-score (30d) *(crypto-specific)* | DAI supply vs norm; DeFi-native stablecoin expansion proxy. | +1 bullish-high |
| `stbl_stable_shock` -- Stablecoin supply shock *(crypto-specific)* | Binary: rapid stablecoin supply expansion; sudden liquidity injection. | +1 bullish-high |
| `stbl_stable_crash` -- Stablecoin supply crash *(crypto-specific)* | Binary: rapid stablecoin supply contraction; dry-powder draining = risk-off. | -1 bearish-high |
| `stbl_stable_shock_strong` -- Stablecoin strong supply shock *(crypto-specific)* | Binary: extreme stablecoin inflow event; high-conviction liquidity surge. | +1 bullish-high |
| `stbl_usdt_shock` -- USDT supply shock *(crypto-specific)* | Binary: rapid USDT minting event; strong institutional on-ramp signal. | +1 bullish-high |
| `stbl_compound_shock` -- Compound stablecoin shock *(crypto-specific)* | Composite shock across multiple stablecoins simultaneously; peak liquidity signal. | +1 bullish-high |

## Attention & social  (`social`, 1 features)
*Question it answers:* Is retail attention rising on this asset?

**On its own:** Attention/narrative intensity -- reflexive fuel (THE signal on memes, noise on BTC).

**Asset-conditional:** Attention INVERTS by archetype: on BTC social/attention is coincident-to-lagging noise (a sentiment gauge, not a trigger). On MEMES attention IS the fundamental -- the only asset -- and social velocity leads price with a short lag; the catch: peak social = insider exit timing. DeFi: a partnership/audit post can carry real content.

| feature | meaning | polarity |
|---|---|---|
| `soc_wiki_views` -- Wiki/search attention *(crypto-specific)* | Retail attention proxy. Rising = reflexive interest. | +1 bullish-high |

## Regime labels (precomputed)  (`regime`, 10 features)
*Question it answers:* What regime does the pipeline already label this bar as (trend persistence, Hurst, asset DNA)?

**On its own:** The precomputed market-state label (trend/Hurst/DNA) the asset is in.

**Asset-conditional:** Regime labels (SMA200 / Hurst / DNA) are COHORT-WIDE, BTC-driven -- a regime call is really a market-state call. regime_label encodes price-vs-trend position (above/below MA). Same label means the same thing across assets BECAUSE it's the market regime, not an asset property; but its TRADING implication differs (a bull regime in a meme is far more fragile than in BTC).

| feature | meaning | polarity |
|---|---|---|
| `regime_label` -- Regime label | Pipeline's precomputed regime tag for this bar. | 0 contextual |
| `asset_dna` -- Asset DNA | Static behavioral cluster the asset belongs to. | 0 contextual |
| `is_u10` -- U10 universe flag | Asset is in the top-10 liquid universe. | 0 contextual |
| `is_u50` -- U50 universe flag | Asset is in the top-50 liquid universe. | 0 contextual |
| `is_u100` -- U100 universe flag | Asset is in the top-100 liquid universe. | 0 contextual |
| `mv_days_since_listed_binance` -- Days listed on Binance | Asset listing age on Binance; newer = higher information asymmetry/volatility. | 0 contextual |
| `mv_days_since_listed_bybit` -- Days listed on Bybit | Asset listing age on Bybit. | 0 contextual |
| `mv_days_since_listed_okx` -- Days listed on OKX | Asset listing age on OKX. | 0 contextual |
| `mv_n_venues_listed` -- Number of venues listed | Cross-venue listing count; higher = more liquid/established asset. | 0 contextual |
| `mv_is_multi_venue` -- Multi-venue flag | Binary: asset is listed on multiple major venues. | 0 contextual |
