# Crypto Market Understanding (2026-06-09)

**Purpose.** The grounded model of *how the crypto market actually works* — its nature, its mathematical characteristics,
the mechanics of its signals, and the strategies that are evidenced to work — so the strategy engine is built on
understanding, not guesswork. Synthesized from 5 cited web-research passes (2026-06-09) + cross-validated against this
project's own empirical mining ([CHIMERA_MINING_FINDINGS_2026_06_08.md](CHIMERA_MINING_FINDINGS_2026_06_08.md)).

> **Provenance + honesty.** External claims carry a source URL; extraordinary or single-source claims are tagged
> [REPORTED] (cited but not independently verified) or [UNCERTAIN]. Claims this project has independently measured are
> tagged [RWYB-OURS]. This is a literature synthesis — a strategy still earns belief only by clearing `candidate_gate`.

---

## I. The nature of the market (structure & microstructure)
1. **24/7/365, no close, no circuit breakers.** Vol can erupt at any hour with no institutional desk coverage and no
   cooling halt; there are no overnight gaps but cascades propagate in real time. → position sizes must be smaller per
   unit vol; risk is continuous. ([BusinessToday](https://www.businesstoday.in/markets/story/bt-explainer-crypto-futures-and-options-vs-equity-futures-and-options-what-are-key-differences-what-traders-must-know-533966-2026-05-29))
2. **Perpetual swaps are the dominant instrument** — >90% of derivatives volume; 2025 derivatives volume ~$85.7T,
   ~$264.5B/day (~10x spot on peak days); top-10 venue OI ~$145B year-end. ([CoinGlass 2025](https://www.coinglass.com/learn/2025-annual-report-en), [CoinGecko](https://www.coingecko.com/learn/rise-of-perpetuals-and-perp-dexs)) The
   perp **funding rate** is the central price-anchoring + sentiment mechanism (see §III).
3. **Stablecoins are the settlement spine** — USDT (~$175B) + USDC (~$75B) = 93% of stablecoin cap; nearly all margin/
   PnL is USDT/USDC-denominated → a depeg is a *systemic* risk with no equity parallel. ([Crystal](https://crystalintelligence.com/thought-leadership/usdt-maintains-dominance-while-usdc-faces-headwinds/))
4. **Venue fragmentation** — top-5 venues ~80% of OI but price discovery runs across 20+ venues simultaneously →
   persistent cross-venue divergence (edge on liquid names, trap on thin ones). Maker ~0.02% / taker ~0.055% typical. ([BitMEX](https://www.bitmex.com/blog/state-of-crypto-perps-2025))
5. **Reflexivity & the liquidation loop** — levered longs → OI up → funding positive → a 5-10% adverse move triggers
   margin calls → forced sells deepen the move → next tier liquidates → cascade. 2025 saw **~$150B liquidated**; the
   Oct 10-11 2025 event alone = ~$19B reported ($30-40B est.), with OI having spiked ~20% in the prior 48h (a
   measurable warning). ([SSRN cascade](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5611392), [CryptoSlate](https://cryptoslate.com/how-150-billion-was-liquidated-from-crypto-market-in-2025-driving-bitcoin-crash/))
6. **On-chain transparency is a structural info edge** — exchange in/outflows, whale wallets, stablecoin mint/burn,
   ETF flows are publicly observable in ~real time (no insider-trading bar). Caveat: ~15-40% of "whale" alerts are
   internal/custodial transfers; the signal is rich but noisy + increasingly gamed. ([Yellow](https://yellow.com/research/etfs-vs-crypto-whales-who-controls-bitcoin-markets-in-2025), [Chainalysis via scout])
7. **MEV is a real on-chain tax** — Ethereum MEV >$3B/yr; sandwich attacks 51.6% of 2025 ETH MEV; >90% of arb routes
   through private relays (PBS). Any DEX-venue strategy must model it explicitly. ([ESMA](https://www.esma.europa.eu/sites/default/files/2025-07/ESMA50-481369926-29744_Maximal_Extractable_Value_Implications_for_crypto_markets.pdf))
8. **Institutionalization is real but partial** — spot BTC ETFs (2024) shifted price discovery (CME > Binance BTC OI);
   ETF custodians + treasuries hold ~1.05M BTC. Yet the retail-leverage *tail* still produces $19B liquidation days.
   Institutions dampen reflexive cycles; retail leverage supplies the volatility texture. ([Grayscale](https://research.grayscale.com/reports/2026-digital-asset-outlook-dawn-of-the-institutional-era))

## II. Mathematical / statistical stylized facts (what a quant must respect)
> **[MEASURED 2026-06-09]** These facts are now COMPUTED on our data, not just cited — see
> [`ECONOMETRIC_SIGNATURE.md`](ECONOMETRIC_SIGNATURE.md) + the tool `python -m mining.econometric_signature`. u10 4h
> whole-series confirms: excess kurtosis 9–108 (BTC 19, DOGE 108), Hill α 1.96–3.2 (cubic band), GARCH persistence
> ≈1.0 near-integrated, ν 3–4, Hurst(ret)≈0.5 / Hurst(|ret|) 0.80–0.84, ADF-stationary, GJR leverage mild-equity on
> majors → inverted on DOGE. The numbers AGREE with the literature below; the archetype split (§IV) is now quantified.
1. **Fat tails.** Daily BTC excess kurtosis ~6-26 (period-dependent; vs ~5-8 for S&P 500); tail index alpha ~2-3.5
   (matured toward the equity "cubic law" alpha~3); GARCH-Student-t innovations fit ν≈3-4. Tails ~2-4x heavier than
   equities; aggregation toward Gaussian is *slow* (kurtosis decays with horizon at exponent ~-0.62). ([PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0246209), [arXiv 1803.08405](https://arxiv.org/pdf/1803.08405))
2. **Volatility clustering + long memory.** The single most robust fact. GARCH(1,1) baseline; EGARCH/GJR + FIGARCH
   (long-memory) fit better. BTC GARCH persistence alpha+beta ~1.0 (near-integrated → shocks decay slowly); Hurst of
   |returns| > 0.5 (long memory in vol). **[RWYB-OURS: our mining measured vol AC1(|ret|) 0.18 (1d) -> 0.33 (15m),
   100% of assets vol-persistent — exactly this.]** ([Frontiers 2025](https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2025.1567626/full), [MDPI Entropy](https://www.mdpi.com/1099-4300/24/10/1410))
3. **Leverage effect is inverted/absent in BTC** — unlike equities, positive shocks often raise vol *more* than
   negative (TGARCH gamma ~-0.058 over full sample); period- and coin-dependent. [UNCERTAIN — active debate.] ([PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0246209))
4. **Direction is near-unpredictable; Hurst of raw returns ~0.5.** Daily/weekly return ACF statistically
   insignificant; micro-scale *negative* AC (bid-ask bounce, Roll); both intraday momentum and intraday reversal exist
   conditional on jumps/liquidity. **[RWYB-OURS: GBM held-out next-bar DIRECTION AUC 0.51-0.53 ~= logistic at every
   TF; median Hurst <0.5; AC1<0 strengthening to 30m (99% assets) — matches the literature exactly.]** ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7850481/), [ScienceDirect intraday](https://www.sciencedirect.com/science/article/abs/pii/S1062940822000833))
5. **One-factor (BTC) cross-section.** A single market factor explains most cross-asset variance; alt variance ~65-70%
   BTC-led in trending regimes, ~35% idiosyncratic; eigenstructure broadened post-Terra (ETH/SOL gained centrality).
   **[RWYB-OURS: median pairwise corr ~0.55 at every TF, BTC-beta ~1.19, 46/50 in one cluster.]** ([Coinbase](https://www.coinbase.com/institutional/research-insights/research/monthly-outlook/monthly-outlook-august-2024), [arXiv 2501.09911](https://arxiv.org/pdf/2501.09911))
6. **Cross-sectional reversal (1wk) + momentum (1-6mo) — with a LIQUIDITY-TIER SIGN-FLIP (load-bearing).** Short-term
   cross-sectional *reversal* at the 1-week horizon is the most replicated directional anomaly (past-week winners
   underperform); momentum emerges at 1-6 months (~3%/wk gross long-short, pre-cost). **CRITICAL: the daily/short-term
   reversal is driven by the ILLIQUIDITY of small-caps — the most liquid/tradeable coins show daily MOMENTUM, not
   reversal** (verified 2026-06-09 across 1,160-3,600-coin studies). So the reversal edge lives in the thin-cap tail
   (worst execution/manipulation); liquid majors flip to momentum. NEVER trade "reversal" in liquid majors. ([Liu et al.](https://www.sciencedirect.com/science/article/abs/pii/S1544612321002208), [Up or down? ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1057521921002349), [Emerald CAFR](https://www.emerald.com/cafr/article/27/4/493/1271913/Unravelling-cross-sectional-patterns-in))
7. **Regime-switching + seasonality.** Markov-switching GARCH finds 2-3 vol regimes (calm/moderate/explosive), weeks-
   to-months durations; vol peaks in US hours; a **"turn-of-the-candle"** effect (disproportionate returns at the 0/15/
   30/45-min boundaries used by systematic traders); 8h funding settlement (00/08/16 UTC) imprints modest patterns;
   weekends quieter. **[RWYB-OURS: our seasonality found vol concentrates 14-16 UTC (US open) + 00 UTC (funding),
   weekends quieter — matches.]** ([PMC turn-of-candle](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10015199/), [MDPI IJFS](https://www.mdpi.com/2227-7072/14/5/103))
8. **Predictable: volatility, funding/basis extremes, on-chain activity, sentiment extremes, 1wk x-sectional reversal.
   NOT reliably predictable: unconditional daily direction, jump timing, macro-direction.** ([Frontiers](https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2025.1567626/full), [Fulgur funding](https://medium.com/@fulgur.ventures/bitcoin-funding-rates-and-price-predictability-27ce95535af1))

## III. Derivatives & on-chain signal mechanics (how to READ each)
- **Funding rate** = Premium Index + clamp(Interest - Premium, +-0.05%); settled ~8h (Hyperliquid 1h). Positive =
  longs pay shorts = crowded long (a *contrarian sell* at extremes >0.1%/8h ~130% ann.); negative streaks mark
  capitulation (contrarian buy). Also a **carry** source (long spot / short perp). ([BingX](https://bingx.com/en/support/articles/14857605906575), [QuantJourney](https://quantjourney.substack.com/p/funding-rates-in-crypto-the-hidden))
- **Basis** (futures-spot) = leverage appetite gauge; contango (positive) normal in bulls, backwardation = distress;
  the **cash-and-carry** trade harvests it. CME BTC basis peaked ~25% (Feb 2024) -> ~4.5% (Dec 2025), 93% of days
  below the 5% breakeven (decayed by ETF arb). ([CME](https://www.cmegroup.com/openmarkets/equity-index/2025/Spot-ETFs-Give-Rise-to-Crypto-Basis-Trading.html), [CoinDesk basis](https://www.coindesk.com/markets/2025/12/03/bitcoin-futures-return-to-deepest-backwardation-since-ftx-collapse))
- **Open interest** — rising-OI+rising-price = conviction; rising-OI+falling-price = distribution/shorts; falling-OI =
  unwind; OI+funding *together* = the crowding thermometer (high OI + high funding = most fragile). ([Gate](https://www.gate.com/crypto-wiki/article/how-to-interpret-crypto-derivatives-market-signals-funding-rates-open-interest-and-liquidation-data-explained-20251227))
- **Liquidations** — causal flow that *overshoots* then mechanically exhausts → the post-cascade bounce is a
  structural regularity (capitulation = contrarian-long). CoinGlass aggregates; under-counts in extreme events. ([CryptoSlate](https://cryptoslate.com/how-150-billion-was-liquidated-from-crypto-market-in-2025-driving-bitcoin-crash/))
- **Long/short ratio + taker imbalance** — top-trader vs global divergence = smart-vs-retail; taker buy/sell extreme =
  exhaustion. ([AInvest LSR](https://www.ainvest.com/news/bitcoin-long-short-ratio-sentinel-market-reversals-2025-2509/))
- **On-chain/flow** — exchange inflow = sell-prep, outflow = conviction hold; **stablecoin supply expansion = fresh
  capital** (a top macro-bull gauge, esp. with ETF inflows); ETF flows are now the single most explanatory BTC
  variable post-2024. ([Glassnode](https://insights.glassnode.com/the-week-onchain-week-29-2025/), [CoinGlass ETF](https://www.coinglass.com/bitcoin-etf))
- **Implied vol (DVOL, Deribit)** = crypto VIX; DVOL/19 ~ expected daily % move; **IV>RV (variance risk premium) is
  the normal state** → vol-sellers harvest it (but short gamma = "pennies in front of a steamroller"). BTC/ETH only. ([Deribit DVOL](https://insights.deribit.com/exchange-updates/dvol-deribit-implied-volatility-index/))

## IV. Asset archetypes — the SAME signal means DIFFERENT things (see the dictionary's matrix)
Five archetypes with distinct vol/liquidity/marginal-trader/survivorship → every signal must be read per-archetype.
| Archetype | Ann. vol | BTC-beta | Marginal trader | Survivorship |
|---|---|---|---|---|
| **BTC** (reserve/beta) | ~50-60% | 1.0 | institutional/ETF | permanent |
| **ETH / large-cap L1** | ~70-100% | 1.2-1.8 | crypto-native inst. + ETF | low churn |
| **Mid-cap / DeFi** | ~100-200% | 1.5-2.5 | retail + DeFi funds | high (unlock overhang) |
| **Meme** | >200% (daily 12-53%) | <0.3 explanatory | retail + bots + single-whale | maximal (97% fail; 68% Solana rugs <72h) |
| **Stablecoin** | <1% (tail to -30/-65%) | ~0 | arbitrageurs | depeg tail |
([CoinLaw memes](https://coinlaw.io/memecoin-statistics/), [S&P BTC vol](https://www.spglobal.com/en/research-insights/special-reports/bitcoin-volatility-trends-deep-dive), [BIS stablecoin](https://www.bis.org/publ/work1270.pdf))
**The headline answer to "does funding mean the same on BTC vs a meme?": NO.** On BTC funding = crowded *institutional*
leverage with real mean-reversion; on a meme it may be one whale dominating both sides of a thin book, or the insider
short-leg of a pump-and-dump. Full signal×archetype matrix in [CHIMERA_FEATURE_DICTIONARY.md](CHIMERA_FEATURE_DICTIONARY.md).

## V. Strategy families — the evidence-based catalog (detail + our-fit in the playbook)
| Family | Edge / mechanism | Regime needed | TF / instrument | Robustness 2024-26 | Retail-accessible |
|---|---|---|---|---|---|
| Trend / TSMOM (MA, breakout) | ride autocorrelated drift | sustained trend | 6h-1d / perp | **Conditional/Medium — naive daily long-only MA is OUR VERIFIED NULL; only vol-scaled TSMOM + hard regime filter survived, must beat buy-and-hold Calmar on mixed-regime UNSEEN** | Yes (needs regime filter) |
| Cross-sectional momentum/reversal | 1wk reversal **(illiquid tail only; liquid majors = momentum)** | high dispersion | 1d-1wk / perp L/S | Medium | Partial (short-leg borrow; tier-consistent) |
| Mean-reversion / stat-arb / pairs | cointegration spread reversion | stable relationships | 1h-6h / spot+perp | Medium | Yes (**maker fills mandatory**) |
| Funding / basis carry | harvest funding/basis | positive funding | perp / delta-neutral | **Low (ETF-arb decayed)** | Yes (capital-heavy) |
| Volatility (target/breakout/VRP) | vol predictable; sell IV-RV gap | calm (VRP) / expansion (breakout) | overlay / options(Deribit) | Medium | Partial (options BTC/ETH only) |
| Liquidation/event fade | overshoot exhaustion / event bias | high OI / scheduled event | min-h / perp | Medium (not codified) | Yes (overlay) |
| Market-making / HFT / latency | spread + speed | fragmented liquid | sub-ms / all | High for incumbents | **No (co-location moat)** |
| On-chain / flow / sentiment | flows lead price | structural flows | 1d-1wk / spot | Low in isolation | Yes (overlay only) |

**Cost is the dominant constraint.** Realistic round-trip ~**0.15-0.30%** at taker rates; one study found only 6/22
popular strategies net-profitable after fees (all RSI mean-reversion). **Maker routing (~0.05-0.10% round-trip) is the
single biggest lever** — it flips many marginal strategies positive. ([StratProof](https://stratproof.com/blog/paper-trading-22-strategies-real-fees))
**Top picks for a retail-to-midsize systematic trader:** (1) trend-following 6h-1d maker-routed + regime filter; (2)
stat-arb/pairs maker-routed with cointegration gate; (3) liquidation/event as a *conditional sizing overlay*. **Most
decayed/closed:** funding carry (ETF-arb), pure HFT/MM (co-location). Extraordinary single-source: a leveraged BTC
funding-carry algo claims Sharpe 6.1 / 16% / <2% DD [REPORTED [SSRN 5292305](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5292305)].

## VI. The convergence (why this is trustworthy) + what it means for us
Our independent mining and the external literature **agree on every major point**: vol is predictable + clusters
(strengthening intraday); direction is ~random (Hurst ~0.5, AUC ~0.5) linearly AND nonlinearly; the market is
one-factor (BTC, corr ~0.55); intraday is mean-reverting (negative AC, bid-ask bounce); the cross-section reverses at
~1 week; funding/basis extremes mean-revert; vol concentrates at US-open + funding times. This convergence is the
foundation: **we are not fighting an unknown market — we know its shape.** The implication carried into the playbook:
*build for volatility/magnitude + cross-sectional reversal + carry/flow conditioning, with maker execution and regime
gating; do NOT build naive long-only directional trend at daily resolution (the data + the literature both say it's
the one thing that doesn't work after costs).*

## Sources
All inline above. Primary anchors: CoinGlass 2025 annual; BIS WP-1087 (crypto carry) + WP-1270 (stablecoins); SSRN
5611392 (Oct-2025 cascade); PLOS ONE 2021 (BTC vol/multifractality); Frontiers 2025 (long-memory GARCH); Liu et al.
(x-sectional reversal); Emerald CAFR 2024 (4-factor); Deribit DVOL; CoinLaw (meme survivorship); arXiv 2501.09911
(institutional correlation). Synthesized 2026-06-09 from 5 web-research scouts; cross-validated vs our
CHIMERA_MINING_FINDINGS.
