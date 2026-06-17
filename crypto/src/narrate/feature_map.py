"""src/narrate/feature_map.py -- the canonical DECOMPOSITION of chimera into families + human-readable reads.

This is the "all of chimera, decomposed and explained" layer: every chimera column is assigned to a FAMILY, and
the interpretable ones carry a `Feature` record (what it measures, which DIRECTION means what, and whether it is a
CRYPTO-SPECIFIC signal that has no clean analogue in equities/FX). The narrator uses this to turn raw normalized
columns into sentences a human can read.

Design notes:
  - Chimera `norm_*` columns are z-scored (std ~= 1), so a value is itself a "how many sigma" read.
  - `polarity`: +1 = higher is more BULLISH / more upward-pressure; -1 = higher is more BEARISH / down-pressure;
     0 = MAGNITUDE-only (higher = more of the thing, no directional sign -- e.g. volatility, toxicity, intensity).
  - `crypto_specific=True` flags signals that exist BECAUSE crypto is a 24/7, perp-dominated, highly-reflexive,
     retail-leveraged market (funding, liquidations, basis, LSR positioning, on-chain whale flow). See crypto_context.
  - Coverage is exhaustive at the FAMILY level (every column maps to a family via classify()); the curated
     `FEATURES` dict gives rich reads for the most decision-relevant ~70 columns. Uncurated columns still get a
     generic family read.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Feature:
    col: str
    family: str
    title: str
    desc: str
    polarity: int = 0          # +1 bullish-up, -1 bearish-up, 0 magnitude-only
    crypto_specific: bool = False


@dataclass(frozen=True)
class Family:
    key: str
    title: str
    question: str              # the question this family answers about the market
    crypto_note: str = ""


# ---------------------------------------------------------------------------
# FAMILY taxonomy -- the spine of the decomposition. Ordered as a human would read a market.
FAMILIES: dict[str, Family] = {f.key: f for f in [
    Family("structure", "Price structure & trend",
           "Where is price relative to its own history and moving averages -- trending, ranging, stretched?"),
    Family("momentum", "Momentum & returns",
           "Which way and how hard has price been moving, and is that move accelerating or exhausting?"),
    Family("volatility", "Volatility & activity",
           "Is the market compressed or expanded, calm or violent, and is volatility clustering?"),
    Family("orderflow", "Order flow & microstructure",
           "Who is in control of the tape -- aggressive buyers or sellers -- and how toxic/intense is the flow?",
           "Hawkes intensity, VPIN and Kyle-lambda are computed from trade prints; crypto's 24/7 tape makes these "
           "continuously meaningful (no session gaps)."),
    Family("liquidity", "Liquidity & order book",
           "How deep and balanced is the book, and is it thinning out (fragile) right now?",
           "Crypto books are fragmented across venues and thin relative to notional turnover -- depth shocks move "
           "price hard."),
    Family("derivatives", "Funding, open interest & basis",
           "What are leveraged traders paying to hold, how crowded is the perp, and is the futures basis stressed?",
           "Perpetual funding + basis are the dominant crypto-native positioning signals; there is no equity analogue."),
    Family("liquidation", "Liquidations & forced flow",
           "Is there forced selling/buying (cascades, capitulation, short squeezes) happening or building?",
           "Liquidation cascades are a defining crypto reflexivity mechanism -- leverage unwinds feed back into price."),
    Family("positioning", "Positioning (long/short ratio, smart vs retail)",
           "How are accounts positioned -- crowded long or short, smart money vs retail?",
           "Exchange long/short-ratio and top-trader positioning are crypto-native crowding gauges."),
    Family("whale", "Whale & large-trade flow",
           "Are large players net buying or selling via outsized prints?",
           "On-chain / large-print whale flow is observable in crypto in a way it is not in most markets."),
    Family("cross_asset", "Cross-asset & relative context",
           "What is BTC (the market beta) doing, and how does this asset rank cross-sectionally right now?",
           "BTC-beta dominance: most alts inherit direction from BTC; idiosyncratic moves are the exception."),
    Family("social", "Attention & social",
           "Is retail attention rising on this asset?",
           "Retail attention (search/wiki/social) is a meaningful crypto reflexivity input."),
    Family("regime", "Regime labels (precomputed)",
           "What regime does the pipeline already label this bar as (trend persistence, Hurst, asset DNA)?"),
]}

FAMILY_ORDER = list(FAMILIES.keys())


# ---------------------------------------------------------------------------
def _F(col, family, title, desc, polarity=0, crypto=False):
    return Feature(col, family, title, desc, polarity, crypto)


# Curated reads for the most decision-relevant columns. (Uncurated cols still resolve to a family via classify().)
FEATURES: dict[str, Feature] = {f.col: f for f in [
    # --- structure / trend
    _F("norm_ma_distance", "structure", "Distance to moving average",
       "How far price sits above/below its moving average, in sigma. High = stretched up; low = stretched down.", +1),
    _F("xd_ma_distance", "structure", "Cross MA-distance", "Stretch vs MA relative to the cross-section.", +1),
    _F("norm_deviation", "structure", "Price deviation", "Deviation of price from its local trend baseline.", +1),
    _F("norm_efficiency", "structure", "Trend efficiency (fractal)",
       "Kaufman-style efficiency: how directional vs choppy the path is. High = clean trend; low = chop/range.", 0),
    _F("norm_fd_close", "structure", "Fractional differentiation (log-close)",
       "Frac-diff of log-close (d=0.4, Lopez de Prado): a STATIONARY transform that preserves long-run trend memory. "
       "NOT fractal dimension -- the prior 'path roughness' label was wrong (computed by frac_diff_fast, sota_shared_logic_v50.py:474).", 0),
    _F("norm_perm_entropy", "structure", "Permutation entropy",
       "Complexity/randomness of the recent path. High = disordered; low = structured/predictable texture.", 0),
    _F("hurst_regime", "structure", "Hurst regime",
       "Persistence label: >0.5 trending, ~0.5 random walk, <0.5 mean-reverting.", 0),
    # --- momentum
    _F("norm_return_1", "momentum", "Return (fast)", "Most-recent-bar normalized return. Sign = direction.", +1),
    _F("norm_return_4", "momentum", "Return (mid)", "4-bar normalized return.", +1),
    _F("norm_return_16", "momentum", "Return (slow)", "16-bar normalized return = the prevailing drift.", +1),
    _F("norm_momentum_accel", "momentum", "Momentum acceleration",
       "Is momentum building or fading (2nd derivative). High = accelerating up.", +1),
    _F("norm_return_kurtosis", "momentum", "Return kurtosis",
       "Tail-heaviness of recent returns. High = jumpy/fat-tailed regime.", 0),
    _F("norm_flow_persistence", "momentum", "Flow persistence", "How autocorrelated recent order flow is.", +1),
    _F("xd_momentum_rank", "momentum", "Cross momentum rank",
       "This asset's momentum percentile vs the cross-section. High = a relative leader.", +1),
    # --- volatility
    _F("norm_yz_volatility", "volatility", "Yang-Zhang volatility",
       "Range-based realized volatility. High = violent/expanded; low = compressed (coil).", 0),
    _F("norm_vol_cluster", "volatility", "Volatility clustering", "Is vol clustering (GARCH-like persistence).", 0),
    _F("norm_vol_ratio", "volatility", "Vol ratio (fast/slow)",
       "Short-vs-long vol. High = vol expanding now relative to baseline.", 0),
    _F("norm_vol_price_corr", "volatility", "Vol-price correlation",
       "Leverage effect: negative = vol rises as price falls (typical risk-off).", -1),
    _F("norm_log_volume", "volatility", "Log volume", "Activity level. High = heavy participation.", 0),
    _F("norm_bar_duration", "volatility", "Bar duration", "For event bars: how long this bar took to form.", 0),
    # --- orderflow / microstructure
    _F("norm_vpin", "orderflow", "VPIN (flow toxicity)",
       "Volume-synchronized probability of informed trading. High = toxic/one-sided flow = informed pressure.", 0, True),
    _F("norm_flow_imbalance", "orderflow", "Order-flow imbalance",
       "Signed aggressor imbalance. High = buyers lifting offers; low = sellers hitting bids.", +1, True),
    _F("norm_hawkes_intensity", "orderflow", "Hawkes intensity", "Self-exciting trade arrival rate (clustering).", 0, True),
    _F("norm_hawkes_buy_intensity", "orderflow", "Hawkes buy intensity", "Self-exciting BUY arrival intensity.", +1, True),
    _F("norm_hawkes_sell_intensity", "orderflow", "Hawkes sell intensity", "Self-exciting SELL arrival intensity.", -1, True),
    _F("norm_hawkes_imbalance", "orderflow", "Hawkes imbalance", "Buy-vs-sell intensity imbalance.", +1, True),
    _F("norm_kyle_lambda", "orderflow", "Kyle's lambda (impact)",
       "Price impact per unit flow = illiquidity. High = thin/impactful; small flow moves price.", 0, True),
    _F("norm_tick_count", "orderflow", "Tick count", "Number of trades; participation/fragmentation of the tape.", 0),
    # --- liquidity / book depth
    _F("bd_imbalance_l1", "liquidity", "Book imbalance L1",
       "Top-of-book bid/ask size imbalance. High = bid-heavy (support); low = ask-heavy (resistance).", +1, True),
    _F("bd_imbalance_l5", "liquidity", "Book imbalance L5", "5-level depth imbalance.", +1, True),
    _F("bd_thin_book_frac", "liquidity", "Thin-book fraction",
       "Fraction of time the book was thin. High = fragile, gap-prone book.", 0, True),
    _F("bd_total_depth_l5_mean", "liquidity", "Total depth L5", "Aggregate resting depth. Low = fragile.", 0, True),
    _F("bd_notional_skew", "liquidity", "Notional skew", "Skew of resting notional bid vs ask.", +1, True),
    # --- derivatives: funding / OI / basis
    _F("norm_funding", "derivatives", "Funding rate",
       "Perp funding. High positive = longs paying (crowded long, often a fade); negative = shorts paying.", -1, True),
    _F("fund_rate_z30", "derivatives", "Funding z-score", "Funding vs its 30-bar norm. Extreme = crowded.", -1, True),
    _F("fund_sign_flip", "derivatives", "Funding sign flip", "Funding flipped sign = positioning regime change.", 0, True),
    _F("norm_oi_change", "derivatives", "OI change", "Open-interest change. Rising OI + rising price = new longs.", +1, True),
    _F("norm_oi_price_divergence", "derivatives", "OI-price divergence",
       "OI and price disagreeing = potential squeeze fuel.", 0, True),
    _F("bs_basis_pct", "derivatives", "Futures basis %", "Perp/quarterly basis. High = bullish carry/greed.", +1, True),
    _F("bs_basis_z30", "derivatives", "Basis z-score", "Basis vs norm; extreme = stressed leverage.", +1, True),
    _F("bs_basis_panic", "derivatives", "Basis panic", "Basis collapse flag = deleveraging/risk-off.", -1, True),
    _F("bs_basis_frenzy", "derivatives", "Basis frenzy", "Basis blow-off flag = leveraged greed.", +1, True),
    # --- liquidation
    _F("liq_long_usd", "liquidation", "Long liquidations $", "Forced long sells. Spikes mark down-cascades.", -1, True),
    _F("liq_short_usd", "liquidation", "Short liquidations $", "Forced short buys. Spikes mark squeezes.", +1, True),
    _F("liq_delta_z30", "liquidation", "Liquidation delta z", "Net long-vs-short liq pressure (z).", -1, True),
    _F("liq_capitulation", "liquidation", "Capitulation flag", "Long-liquidation capitulation event (contrarian long).", +1, True),
    _F("liq_short_panic", "liquidation", "Short panic flag", "Short-side panic / squeeze event.", +1, True),
    _F("liq_long_spike", "liquidation", "Long-liq spike", "Outsized long liquidation burst.", -1, True),
    _F("liq_short_spike", "liquidation", "Short-liq spike", "Outsized short liquidation burst (squeeze).", +1, True),
    # --- positioning (LSR / smart vs retail)
    _F("s3_global_lsr", "positioning", "Global long/short ratio", "All-account LSR. High = crowd is long.", -1, True),
    _F("s3_top_pos_lsr", "positioning", "Top-trader position LSR", "Top traders' positioning. High = big long.", +1, True),
    _F("s3_taker_lsr", "positioning", "Taker LSR", "Aggressor buy/sell ratio.", +1, True),
    _F("s3_smart_vs_retail", "positioning", "Smart vs retail", "Top-trader minus retail positioning gap.", +1, True),
    _F("s3_smart_bullish", "positioning", "Smart bullish", "Smart money net bullish flag.", +1, True),
    _F("s3_smart_extreme_long", "positioning", "Smart extreme long", "Smart money crowded long (fade risk).", -1, True),
    _F("s3_global_lsr_z", "positioning", "Global LSR z", "Crowd long/short extremity (z).", -1, True),
    # --- whale flow
    _F("wh_whale_net_usd", "whale", "Whale net $", "Net large-trade USD flow. Positive = whales net buying.", +1, True),
    _F("wh_whale_buy_usd", "whale", "Whale buy $", "Large-print buying.", +1, True),
    _F("wh_whale_sell_usd", "whale", "Whale sell $", "Large-print selling.", -1, True),
    _F("norm_whale", "whale", "Whale (normalized)", "Normalized whale-flow signal.", +1, True),
    _F("wh_whale_trade_count", "whale", "Whale trade count",
       "Number of large-print trades; high = many whale participants active.", 0, True),
    _F("wh_whale_trade_count_500k", "whale", "Whale trade count (>$500k)",
       "Count of trades exceeding $500k notional; captures true institutional-size flow.", 0, True),
    # --- cross-asset
    _F("xd_btc_return", "cross_asset", "BTC return", "Market beta: what BTC is doing right now.", +1),
    _F("xd_btc_volatility", "cross_asset", "BTC volatility", "BTC vol = market-wide risk temperature.", 0),
    _F("xd_cross_return_mean", "cross_asset", "Cross return mean", "Average move across the universe (breadth).", +1),
    _F("xd_funding_spread", "cross_asset", "Funding spread", "This asset's funding vs the cross-section.", -1, True),
    # --- social
    _F("soc_wiki_views", "social", "Wiki/search attention", "Retail attention proxy. Rising = reflexive interest.", +1, True),
    # --- regime labels
    _F("regime_label", "regime", "Regime label", "Pipeline's precomputed regime tag for this bar.", 0),
    _F("asset_dna", "regime", "Asset DNA", "Static behavioral cluster the asset belongs to.", 0),
    _F("is_u10", "regime", "U10 universe flag", "Asset is in the top-10 liquid universe.", 0),
    _F("is_u50", "regime", "U50 universe flag", "Asset is in the top-50 liquid universe.", 0),
    _F("is_u100", "regime", "U100 universe flag", "Asset is in the top-100 liquid universe.", 0),

    # --- orderflow: Hawkes branching-ratio (hbr_*)
    _F("hbr_eta_total", "orderflow", "Hawkes branching ratio (total)",
       "Aggregate excitability of trade arrivals; close to 1 = near-critical self-exciting regime.", 0, True),
    _F("hbr_eta_buy", "orderflow", "Hawkes branching ratio (buys)",
       "Buy-side excitability; high = buy flow is self-reinforcing (momentum).", +1, True),
    _F("hbr_eta_sell", "orderflow", "Hawkes branching ratio (sells)",
       "Sell-side excitability; high = sell flow is self-reinforcing (momentum down).", -1, True),
    _F("hbr_eta_imbalance", "orderflow", "Hawkes branching imbalance",
       "Signed buy-minus-sell excitability. Positive = buy side more self-exciting.", +1, True),
    _F("hbr_n_trades", "orderflow", "Hawkes trade count",
       "Number of trades used in Hawkes fit; proxy for tape activity intensity.", 0, True),

    # --- orderflow: raw volumes and spreads
    _F("buy_vol", "orderflow", "Buy volume",
       "Aggressor buy volume for the bar; higher = demand pressure.", +1),
    _F("sell_vol", "orderflow", "Sell volume",
       "Aggressor sell volume for the bar; higher = supply pressure.", -1),
    _F("tick_count", "orderflow", "Tick count",
       "Raw trade count; high = active/fragmented tape.", 0),
    _F("norm_spread_bps", "orderflow", "Quoted spread (bps)",
       "Bid-ask spread in basis points; high = thin/illiquid book.", 0),
    _F("norm_hl_spread", "orderflow", "High-low spread",
       "Bar high-low range as a fraction; measures realized intrabar volatility/whip.", 0),
    _F("norm_cs_spread", "orderflow", "Cross-spread (composite)",
       "Composite spread signal across venues; elevated = fragmented/stressed liquidity.", 0, True),

    # --- derivatives: futures basis extras (bs_basis_*)
    _F("bs_basis_delta_1d", "derivatives", "Basis 1-day change",
       "Change in futures basis over 1 day; rising = carry/leverage expanding.", +1, True),
    _F("bs_basis_delta_3d", "derivatives", "Basis 3-day change",
       "Change in futures basis over 3 days; trend in carry appetite.", +1, True),
    _F("bs_basis_xsec_z", "derivatives", "Basis cross-section z",
       "This asset's basis vs the cross-section; high = outsized carry vs peers.", +1, True),
    _F("bs_basis_bull_shock", "derivatives", "Basis bull shock",
       "Sudden spike up in basis = leveraged-long rush; often precedes local top.", -1, True),
    _F("bs_basis_bear_shock", "derivatives", "Basis bear shock",
       "Sudden collapse in basis = deleveraging/panic exit; contrarian signal.", +1, True),

    # --- derivatives: funding extras (fund_*)
    _F("fund_rate_mean", "derivatives", "Funding rate mean (cross-venue)",
       "Average funding rate across venues; high positive = crowded long, fade signal.", -1, True),
    _F("fund_rate_max", "derivatives", "Funding rate max (cross-venue)",
       "Maximum funding across venues; captures the most extreme long-crowd venue.", -1, True),
    _F("fund_rate_min", "derivatives", "Funding rate min (cross-venue)",
       "Minimum funding across venues; deeply negative = crowded short, squeeze fuel.", +1, True),
    _F("fund_rate_abs_mean", "derivatives", "Funding absolute mean",
       "Mean absolute funding; high = strong directional crowding (either side).", 0, True),
    _F("fund_extreme_long_count", "derivatives", "Extreme-long funding count",
       "Number of venues with extreme positive funding; high = broadly crowded long.", -1, True),
    _F("fund_extreme_short_count", "derivatives", "Extreme-short funding count",
       "Number of venues with extreme negative funding; high = broadly crowded short.", +1, True),
    _F("fund_avg_apr", "derivatives", "Funding APR",
       "Annualized funding rate; high positive = expensive to hold long (bearish fade).", -1, True),
    _F("fund_n_settlements", "derivatives", "Funding settlement count",
       "Number of 8h settlements in the bar; normally 1 for 8h bars, <1 = data gap.", 0, True),

    # --- derivatives: premium / implied basis
    _F("premium_vol30", "derivatives", "Premium volatility (30-bar)",
       "Volatility of the perp basis premium; high = unstable carry regime.", 0, True),
    _F("premium_persistence30", "derivatives", "Premium persistence (30-bar)",
       "Autocorrelation of premium; high = carry regime is sticky.", +1, True),
    _F("premium_extreme_count30", "derivatives", "Premium extreme count (30-bar)",
       "Count of extreme premium readings; elevated = repeated leverage surges.", 0, True),
    _F("premium_z90", "derivatives", "Premium z-score (90-bar)",
       "Premium vs 90-bar norm; extreme = stretched carry vs historical baseline.", 0, True),
    _F("premium_apr", "derivatives", "Premium APR",
       "Annualized premium yield; equivalent to funding APR on the spot-futures basis.", -1, True),

    # --- liquidity: cross-exchange spreads (xex_*)
    _F("xex_cb_bn_spread_bps", "liquidity", "Coinbase-Binance spread (bps)",
       "Price spread between Coinbase and Binance; elevated = cross-venue stress/arbitrage.", 0, True),
    _F("xex_by_bn_spread_bps", "liquidity", "Bybit-Binance spread (bps)",
       "Price spread between Bybit and Binance; elevated = fragmented liquidity.", 0, True),
    _F("xex_ok_bn_spread_bps", "liquidity", "OKX-Binance spread (bps)",
       "Price spread between OKX and Binance; elevated = cross-venue dislocation.", 0, True),
    _F("xex_cb_bn_spread_bps_right", "liquidity", "Coinbase-Binance spread (lagged)",
       "Prior-period Coinbase-Binance spread; used to detect persistent dislocation.", 0, True),
    _F("xex_by_bn_spread_bps_right", "liquidity", "Bybit-Binance spread (lagged)",
       "Prior-period Bybit-Binance spread.", 0, True),
    _F("xex_ok_bn_spread_bps_right", "liquidity", "OKX-Binance spread (lagged)",
       "Prior-period OKX-Binance spread.", 0, True),
    _F("xex_cb_bn_z30", "liquidity", "Coinbase-Binance spread z-score",
       "Spread vs 30-bar norm; high z = unusual dislocation, potential cascade risk.", 0, True),
    _F("xex_spread_dispersion", "liquidity", "Cross-exchange spread dispersion",
       "Variance of spreads across venue pairs; high = market is fragmented/dislocated.", 0, True),
    _F("xex_max_abs_spread", "liquidity", "Max absolute cross-exchange spread",
       "Largest single spread across monitored venue pairs; extreme = stress peak.", 0, True),
    _F("xex_n_venues_active", "liquidity", "Venues active count",
       "Number of exchanges with active markets; lower = liquidity concentration risk.", 0, True),

    # --- liquidity: LOB aggregates (lob_*)
    _F("lob_l1_imb_mean", "liquidity", "LOB L1 imbalance mean",
       "Mean top-of-book bid/ask imbalance; high = persistent bid pressure.", +1, True),
    _F("lob_l1_imb_std", "liquidity", "LOB L1 imbalance std",
       "Variability of top-of-book imbalance; high = unstable/rapidly flipping book.", 0, True),
    _F("lob_l5_imb_mean", "liquidity", "LOB L5 imbalance mean",
       "Mean 5-level depth imbalance; broader view of order-book skew.", +1, True),
    _F("lob_l5_imb_std", "liquidity", "LOB L5 imbalance std",
       "Variability of 5-level depth imbalance.", 0, True),
    _F("lob_spread_bps_mean", "orderflow", "LOB quoted spread mean (bps)",
       "Average bid-ask spread over the bar; high = costly to trade/thin book.", 0, True),
    _F("lob_spread_bps_p90", "orderflow", "LOB quoted spread p90 (bps)",
       "90th-percentile spread over the bar; captures worst-case execution cost.", 0, True),
    _F("lob_top_pressure_mean", "liquidity", "LOB top-of-book pressure mean",
       "Mean notional pressure at best bid/ask; high = strong resting order wall.", 0, True),
    _F("lob_count_imb_mean", "liquidity", "LOB order-count imbalance mean",
       "Imbalance by order count (not notional); captures HFT quote skew.", +1, True),
    _F("lob_run_length_p50", "liquidity", "LOB run length median",
       "Median consecutive-same-sign depth snapshots; high = persistent book skew.", 0, True),
    _F("lob_kyle_lambda_mean", "orderflow", "LOB Kyle-lambda mean",
       "Mean price impact per unit flow estimated from the LOB; high = illiquid.", 0, True),
    _F("lob_kyle_lambda_abs_max", "orderflow", "LOB Kyle-lambda abs max",
       "Peak price impact event within the bar; captures flash liquidity shocks.", 0, True),
    _F("lob_n_bars", "liquidity", "LOB snapshot count",
       "Number of book snapshots in the bar; lower = sparser data coverage.", 0),

    # --- liquidity: book depth extras (bd_*)
    _F("bd_depth_l1pct_mean", "liquidity", "Book depth L1 pct mean",
       "Mean depth at L1 as pct of total; higher = concentrated near-touch liquidity.", 0, True),
    _F("bd_depth_l1pct_p90", "liquidity", "Book depth L1 pct p90",
       "90th-pct L1-depth concentration; captures thin-book tail events.", 0, True),
    _F("bd_notional_l1pct_mean", "liquidity", "Notional depth L1 pct mean",
       "Mean notional concentration at L1; proxy for maker-side commitment near touch.", 0, True),
    _F("bd_total_depth_l5_p10", "liquidity", "Total depth L5 p10",
       "10th-pct of 5-level depth; captures worst-case book depth during the bar.", 0, True),
    _F("bd_depth_at_02pct", "liquidity", "Depth at 0.2% from mid",
       "Resting notional within 20bp of mid; directly translates to market-impact capacity.", 0, True),
    _F("bd_n_snapshots", "liquidity", "Book snapshot count",
       "Number of book snapshots taken; lower = sparse coverage.", 0),
    _F("bd_bgf_imbalance_l1", "liquidity", "BGF book imbalance L1",
       "Bybit/Gate/FTX-composite top-of-book imbalance; cross-venue bid/ask skew.", +1, True),

    # --- liquidity: BGF LOB aggregates
    _F("lob_bgf_l1_imb_mean", "liquidity", "BGF LOB L1 imbalance mean",
       "Mean top-of-book imbalance from BGF composite; supplementary venue signal.", +1, True),
    _F("lob_bgf_kyle_lambda_mean", "orderflow", "BGF Kyle-lambda mean",
       "Price impact from BGF composite venues; high = those venues are illiquid.", 0, True),
    _F("lob_bgf_spread_bps_mean", "orderflow", "BGF quoted spread mean (bps)",
       "Mean spread from BGF composite; wider = off-exchange execution is costly.", 0, True),
    _F("lob_bgf_top_pressure_mean", "liquidity", "BGF top-of-book pressure mean",
       "Mean notional at best bid/ask across BGF venues.", 0, True),
    _F("lob_bgf_count_imb_mean", "liquidity", "BGF order-count imbalance mean",
       "Mean order-count imbalance from BGF composite; HFT skew on secondary venues.", +1, True),

    # --- positioning: s3_* extras
    _F("s3_oi_usd", "positioning", "Open interest (USD)",
       "Total open interest in USD; rising = new leverage entering (watch direction).", 0, True),
    _F("s3_top_acct_lsr", "positioning", "Top-account long/short ratio",
       "Largest accounts' LSR; more reliable than global LSR (less noise).", +1, True),
    _F("s3_smart_bearish", "positioning", "Smart bearish flag",
       "Smart money net bearish signal; top traders leaning short.", -1, True),
    _F("s3_smart_extreme_short", "positioning", "Smart extreme short",
       "Smart money at extreme short (contrarian: squeeze potential).", +1, True),
    _F("s3_smart_vs_retail_z", "positioning", "Smart vs retail z-score",
       "Z-scored divergence between smart and retail positioning; extreme = max divergence.", +1, True),
    _F("s3_top_pos_lsr_z", "positioning", "Top-trader position LSR z-score",
       "Top-trader LSR vs its own norm; captures regime shifts in smart positioning.", +1, True),
    _F("s3_top_pos_lsr_xsec_z", "positioning", "Top-trader position LSR cross-section z",
       "This asset's top-trader LSR vs cross-section; isolates idiosyncratic smart positioning.", +1, True),

    # --- social: soc_* extras (currently only wiki; note others may appear later)
    # (soc_wiki_views already curated above; keep open for future soc_ columns)

    # --- cross_asset: xrel_* relative ranks
    _F("xrel_rv_rv_5m_xrank", "cross_asset", "RV (5m) cross-rank",
       "Rank of this asset's 5-min realized vol across the universe; high = most volatile.", 0, True),
    _F("xrel_rv_rv_5m_xpct10", "cross_asset", "RV (5m) cross-pct10",
       "Pct of peers with lower 5m RV; high = relatively volatile.", 0, True),
    _F("xrel_rv_rv_5m_xratio", "cross_asset", "RV (5m) cross-ratio",
       "This asset's RV vs cross-section mean; >1 = vol premium vs peers.", 0, True),
    _F("xrel_rv_bpv_5m_xrank", "cross_asset", "BPV (5m) cross-rank",
       "Rank of this asset's bipower variation; separates jump risk from continuous vol.", 0, True),
    _F("xrel_rv_bpv_5m_xpct10", "cross_asset", "BPV (5m) cross-pct10",
       "Pct of peers with lower bipower variation.", 0, True),
    _F("xrel_rv_bpv_5m_xratio", "cross_asset", "BPV (5m) cross-ratio",
       "BPV relative to cross-section mean.", 0, True),
    _F("xrel_hbr_eta_total_xrank", "cross_asset", "HBR excitability cross-rank",
       "Rank of Hawkes branching ratio vs peers; high = most self-exciting tape.", 0, True),
    _F("xrel_hbr_eta_total_xpct10", "cross_asset", "HBR excitability cross-pct10",
       "Pct of peers with lower Hawkes excitability.", 0, True),
    _F("xrel_hbr_eta_total_xratio", "cross_asset", "HBR excitability cross-ratio",
       "This asset's Hawkes excitability vs cross-section mean.", 0, True),
    _F("xrel_hbr_n_trades_xrank", "cross_asset", "HBR trade-count cross-rank",
       "Rank of trade count vs the universe; high = most active tape.", 0, True),
    _F("xrel_hbr_n_trades_xpct10", "cross_asset", "HBR trade-count cross-pct10",
       "Pct of peers with lower trade count.", 0, True),
    _F("xrel_hbr_n_trades_xratio", "cross_asset", "HBR trade-count cross-ratio",
       "Trade count relative to cross-section mean.", 0, True),
    _F("xrel_liq_long_usd_xrank", "cross_asset", "Long-liquidation cross-rank",
       "Rank of long-liquidation intensity vs peers; high = most deleveraged.", -1, True),
    _F("xrel_liq_long_usd_xpct10", "cross_asset", "Long-liquidation cross-pct10",
       "Pct of peers with lower long-liquidation intensity.", -1, True),
    _F("xrel_liq_long_usd_xratio", "cross_asset", "Long-liquidation cross-ratio",
       "This asset's long liq vs cross-section mean; >1 = idiosyncratic selling.", -1, True),
    _F("xrel_wh_whale_net_usd_xrank", "cross_asset", "Whale net cross-rank",
       "Rank of whale net flow vs the universe; high = most whale-bought.", +1, True),
    _F("xrel_wh_whale_net_usd_xpct10", "cross_asset", "Whale net cross-pct10",
       "Pct of peers with lower whale net flow.", +1, True),
    _F("xrel_wh_whale_net_usd_xratio", "cross_asset", "Whale net cross-ratio",
       "This asset's whale net vs cross-section mean.", +1, True),
    _F("xrel_lob_kyle_lambda_mean_xrank", "cross_asset", "LOB Kyle-lambda cross-rank",
       "Rank of price-impact vs the universe; high = most illiquid.", 0, True),
    _F("xrel_lob_kyle_lambda_mean_xpct10", "cross_asset", "LOB Kyle-lambda cross-pct10",
       "Pct of peers with lower price impact.", 0, True),
    _F("xrel_lob_kyle_lambda_mean_xratio", "cross_asset", "LOB Kyle-lambda cross-ratio",
       "This asset's price impact vs cross-section mean.", 0, True),

    # --- cross_asset: xd_* extras
    _F("xd_cross_vol_mean", "cross_asset", "Cross volatility mean",
       "Average realized volatility across the universe; high = risk-on/volatile crypto environment.", 0),

    # --- liquidation: extras
    _F("liq_delta_usd", "liquidation", "Liquidation delta (USD)",
       "Net long-minus-short liquidation USD; positive = more longs getting stopped.", -1, True),
    _F("liq_total_usd", "liquidation", "Total liquidation (USD)",
       "Combined long + short liquidations; high = forced de-leveraging on both sides.", 0, True),
    _F("liq_long_z30", "liquidation", "Long liquidation z-score",
       "Long liquidations vs 30-bar norm; extreme = outsized forced selling event.", -1, True),
    _F("liq_short_z30", "liquidation", "Short liquidation z-score",
       "Short liquidations vs 30-bar norm; extreme = outsized squeeze event.", +1, True),
    _F("liq_long_xsec_z", "liquidation", "Long liquidation cross-section z",
       "This asset's long liq vs cross-section; idiosyncratic cascade signal.", -1, True),
    _F("liq_short_xsec_z", "liquidation", "Short liquidation cross-section z",
       "This asset's short liq vs cross-section; idiosyncratic squeeze signal.", +1, True),

    # --- volatility: derived vol extras
    _F("dv_dvol_close", "volatility", "DVOL index close",
       "Deribit implied-vol index close; high = options market pricing elevated uncertainty.", 0, True),
    _F("dv_dvol_high", "volatility", "DVOL index high",
       "Intrabar high of DVOL; captures IV spike during the bar.", 0, True),
    _F("dv_dvol_low", "volatility", "DVOL index low",
       "Intrabar low of DVOL; lower = options market relatively calm.", 0, True),

    # --- volatility: realized vol sub-components (rv_*)
    _F("rv_rv_5m", "volatility", "Realized variance (5-min)",
       "5-minute realized variance (continuous component + jumps); high = active bar.", 0, True),
    _F("rv_bpv_5m", "volatility", "Bipower variation (5-min)",
       "Jump-robust realized variance (continuous only); separates smooth from jump vol.", 0, True),
    _F("rv_jv_5m", "volatility", "Jump variation (5-min)",
       "RV minus bipower = jump-only variance; high = price jumped in this bar.", 0, True),
    _F("rv_jump_frac", "volatility", "Jump fraction",
       "Jump variation as pct of total RV; high = most of the move was a jump (gap/news).", 0, True),
    _F("rv_jump_count", "volatility", "Jump count",
       "Number of statistically significant intrabar jumps.", 0, True),
    _F("rv_jump_signed_var", "volatility", "Signed jump variation",
       "Jump variance with sign of the jump; positive = up-jump, negative = down-jump.", +1, True),

    # --- cross_asset: transfer-entropy (te_*)
    _F("te_in", "cross_asset", "Transfer entropy in",
       "Information flowing INTO this asset from others; high = being led by peers.", 0, True),
    _F("te_out", "cross_asset", "Transfer entropy out",
       "Information flowing OUT of this asset to others; high = this asset is leading.", 0, True),
    _F("te_in_btc", "cross_asset", "Transfer entropy in (BTC)",
       "Information flowing from BTC specifically; high = this asset is BTC-led.", 0, True),
    _F("te_out_btc", "cross_asset", "Transfer entropy out (BTC)",
       "Information flowing from this asset toward BTC; high = this asset is leading BTC.", 0, True),
    _F("te_imb", "cross_asset", "Transfer entropy imbalance",
       "te_out minus te_in; positive = net information leader in the network.", 0, True),
    _F("te_btc_imb", "cross_asset", "Transfer entropy imbalance (BTC pair)",
       "TE imbalance vs BTC specifically; positive = leading BTC.", 0, True),

    # --- cross_asset: ETF flow (etf_*)
    _F("etf_btc_etf_total_usdm", "cross_asset", "BTC ETF AUM (USD millions)",
       "Total BTC spot-ETF AUM; rising = institutional accumulation.", +1, True),
    _F("etf_btc_etf_total_z30", "cross_asset", "BTC ETF AUM z-score (30-bar)",
       "ETF AUM vs 30-bar norm; high = rapid institutional inflow.", +1, True),
    _F("etf_btc_etf_total_7d_z", "cross_asset", "BTC ETF AUM z-score (7-day)",
       "Short-horizon ETF AUM z; captures fresh inflow/outflow momentum.", +1, True),
    _F("etf_btc_etf_inflow_shock", "cross_asset", "BTC ETF inflow shock",
       "Binary: large BTC ETF inflow event; strong institutional demand pulse.", +1, True),
    _F("etf_btc_etf_outflow_shock", "cross_asset", "BTC ETF outflow shock",
       "Binary: large BTC ETF outflow event; institutional selling pressure.", -1, True),
    _F("etf_btc_etf_mega_inflow", "cross_asset", "BTC ETF mega inflow",
       "Binary: extreme BTC ETF inflow (top-tail); rare but high-impact demand shock.", +1, True),
    _F("etf_btc_etf_mega_outflow", "cross_asset", "BTC ETF mega outflow",
       "Binary: extreme BTC ETF outflow; rare institutional distribution event.", -1, True),
    _F("etf_eth_etf_total_usdm", "cross_asset", "ETH ETF AUM (USD millions)",
       "Total ETH spot-ETF AUM; correlated institutional demand signal.", +1, True),
    _F("etf_eth_etf_total_z30", "cross_asset", "ETH ETF AUM z-score (30-bar)",
       "ETH ETF AUM vs norm; ETH institutional positioning proxy.", +1, True),
    _F("etf_eth_etf_inflow_shock", "cross_asset", "ETH ETF inflow shock",
       "Binary: large ETH ETF inflow; correlated with broad crypto risk appetite.", +1, True),
    _F("etf_eth_etf_outflow_shock", "cross_asset", "ETH ETF outflow shock",
       "Binary: large ETH ETF outflow; correlated risk-off signal.", -1, True),
    _F("etf_any_inflow_shock", "cross_asset", "Any-ETF inflow shock",
       "Binary: at least one BTC or ETH ETF inflow shock; broad institutional demand.", +1, True),
    _F("etf_both_inflow_shock", "cross_asset", "Both-ETF inflow shock",
       "Binary: simultaneous BTC and ETH ETF inflow; highest-conviction institutional buying.", +1, True),

    # --- cross_asset: stablecoin supply (stbl_*)
    _F("stbl_total_zscore_30d", "cross_asset", "Stablecoin supply z-score (30d)",
       "Total stablecoin supply vs 30-day norm; high = fresh dry powder entering crypto.", +1, True),
    _F("stbl_total_delta_7d_pct", "cross_asset", "Stablecoin supply 7d change %",
       "7-day pct change in stablecoin supply; positive = liquidity expanding.", +1, True),
    _F("stbl_total_delta_30d_pct", "cross_asset", "Stablecoin supply 30d change %",
       "30-day pct change in stablecoin supply; trend in crypto dry-powder.", +1, True),
    _F("stbl_usdt_zscore_30d", "cross_asset", "USDT supply z-score (30d)",
       "USDT supply vs norm; dominant stablecoin inflow = bullish fuel.", +1, True),
    _F("stbl_usdt_delta_7d_pct", "cross_asset", "USDT supply 7d change %",
       "7-day USDT supply change; fresh USDT minting = demand signal.", +1, True),
    _F("stbl_usdc_zscore_30d", "cross_asset", "USDC supply z-score (30d)",
       "USDC supply vs norm; institutional stablecoin expansion.", +1, True),
    _F("stbl_usde_zscore_30d", "cross_asset", "USDe supply z-score (30d)",
       "Ethena USDe supply vs norm; yield-bearing stablecoin demand (crypto-native).", +1, True),
    _F("stbl_dai_zscore_30d", "cross_asset", "DAI supply z-score (30d)",
       "DAI supply vs norm; DeFi-native stablecoin expansion proxy.", +1, True),
    _F("stbl_stable_shock", "cross_asset", "Stablecoin supply shock",
       "Binary: rapid stablecoin supply expansion; sudden liquidity injection.", +1, True),
    _F("stbl_stable_crash", "cross_asset", "Stablecoin supply crash",
       "Binary: rapid stablecoin supply contraction; dry-powder draining = risk-off.", -1, True),
    _F("stbl_stable_shock_strong", "cross_asset", "Stablecoin strong supply shock",
       "Binary: extreme stablecoin inflow event; high-conviction liquidity surge.", +1, True),
    _F("stbl_usdt_shock", "cross_asset", "USDT supply shock",
       "Binary: rapid USDT minting event; strong institutional on-ramp signal.", +1, True),
    _F("stbl_compound_shock", "cross_asset", "Compound stablecoin shock",
       "Composite shock across multiple stablecoins simultaneously; peak liquidity signal.", +1, True),

    # --- misc / catch-all: market maturity (mv_*)
    _F("mv_days_since_listed_binance", "regime", "Days listed on Binance",
       "Asset listing age on Binance; newer = higher information asymmetry/volatility.", 0),
    _F("mv_days_since_listed_bybit", "regime", "Days listed on Bybit",
       "Asset listing age on Bybit.", 0),
    _F("mv_days_since_listed_okx", "regime", "Days listed on OKX",
       "Asset listing age on OKX.", 0),
    _F("mv_n_venues_listed", "regime", "Number of venues listed",
       "Cross-venue listing count; higher = more liquid/established asset.", 0),
    _F("mv_is_multi_venue", "regime", "Multi-venue flag",
       "Binary: asset is listed on multiple major venues.", 0),

    # --- misc: fund_panel label
    _F("fp_fund_panel", "derivatives", "Funding panel flag",
       "Binary/count indicating asset is in the multi-venue funding data panel.", 0, True),

    # --- norm_* extras not yet curated
    _F("norm_funding_momentum", "momentum", "Funding momentum",
       "Trend/momentum in the funding rate series; rising = crowd increasing conviction.", -1, True),
]}


# ---------------------------------------------------------------------------
_PREFIX_FAMILY = [
    ("liq_", "liquidation"), ("bd_", "liquidity"), ("s3_", "positioning"),
    ("wh_", "whale"), ("xd_", "cross_asset"), ("xrel_", "cross_asset"),
    ("fund", "derivatives"), ("bs_basis", "derivatives"), ("soc_", "social"),
    ("hbr_", "orderflow"), ("xex_", "liquidity"),
]
_KEYWORD_FAMILY = [
    (("sma", "ema", "wma", "ma_distance", "deviation", "efficiency", "hurst", "fractal", "fd_", "entropy", "trend"), "structure"),
    (("return", "momentum", "roc", "rsi", "stoch", "accel", "persistence"), "momentum"),
    (("vol", "atr", "yz_", "garch", "range", "log_volume", "duration", "bar_dur"), "volatility"),
    (("vpin", "hawkes", "flow_imbalance", "kyle", "tick", "ofi", "cvd", "aggress", "micro", "spread"), "orderflow"),
    (("depth", "book", "imbalance", "thin", "notional"), "liquidity"),
    (("oi", "funding", "basis", "perp"), "derivatives"),
    (("liq",), "liquidation"), (("lsr", "smart", "retail", "position"), "positioning"),
    (("whale",), "whale"), (("btc", "cross", "xsec", "xrank", "xpct", "xratio", "rank"), "cross_asset"),
    (("wiki", "social", "sentiment", "views"), "social"),
    (("regime", "dna", "is_u"), "regime"),
]
_META = {"timestamp", "date", "bar_id", "symbol", "asset", "open", "high", "low", "close", "volume",
         "volume_usd", "tick_seq", "returns_clean"}


def classify(col: str) -> str:
    """Map ANY chimera column to a family key. Forward/target columns -> 'target' (excluded from narration)."""
    cl = col.lower()
    if col in FEATURES:
        return FEATURES[col].family
    if cl in _META:
        return "meta"
    if cl.startswith("target") or "voladj" in cl or cl.endswith("_raw") or "_fwd" in cl:
        return "target"
    for pre, fam in _PREFIX_FAMILY:
        if cl.startswith(pre):
            return fam
    if cl.startswith("norm_") or cl.startswith("xd_"):
        for keys, fam in _KEYWORD_FAMILY:
            if any(k in cl for k in keys):
                return fam
        return "structure"  # generic norm_ default = structural
    for keys, fam in _KEYWORD_FAMILY:
        if any(k in cl for k in keys):
            return fam
    return "misc"


def group_columns(cols) -> dict[str, list[str]]:
    """Group a column list into families, in FAMILY_ORDER (then meta/target/misc)."""
    out: dict[str, list[str]] = {k: [] for k in FAMILY_ORDER}
    for extra in ("meta", "target", "misc"):
        out[extra] = []
    for c in cols:
        out.setdefault(classify(c), []).append(c)
    return {k: v for k, v in out.items() if v}


def describe(col: str) -> Feature | None:
    return FEATURES.get(col)


def coverage_report(cols) -> dict:
    """How much of a chimera schema we explicitly curate vs only family-bucket (the 'max coverage' meter)."""
    grouped = group_columns(cols)
    tradeable = [c for c in cols if classify(c) not in ("meta", "target")]
    curated = [c for c in tradeable if c in FEATURES]
    return {"n_cols": len(cols), "n_tradeable": len(tradeable), "n_curated": len(curated),
            "curated_pct": round(100 * len(curated) / max(1, len(tradeable)), 1),
            "families_present": {k: len(v) for k, v in grouped.items()}}
