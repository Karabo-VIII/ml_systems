# 04 — Market Model (what we KNOW about the market + its data)

The substrate. Established facts about crypto's structure + the chimera data representation — the things any strategy
taps into. STATUS: ESTABLISHED (multi-confirmed, often RWYB-ours) · CONTESTED · OPEN. The decomposition-dimension axes
(the "fundamental constituents") are in the [README](README.md); this doc is what we know *within* them. Theory detail +
citations in [CRYPTO_MARKET_UNDERSTANDING.md](../CRYPTO_MARKET_UNDERSTANDING.md) and
[CHIMERA_MINING_FINDINGS_2026_06_08.md](../CHIMERA_MINING_FINDINGS_2026_06_08.md).

## A. The opportunity surface
| Finding | What it says | Evidence |
|---|---|---|
| Opportunity is abundant + durable | ~15 assets/day move ≥5%; 97.8% of days have ≥1; regime-stable | move_distribution.py, 104-asset 2020-26 |
| LO-harvestable ≈ half | ~7.6 UP-movers/day ≥5% (~5/day after beta-strip) | long_only_skew.py |
| Mostly idiosyncratic, NOT just beta | BTC explains ~30% of a typical alt's daily var; ~65% of movers survive beta-strip, but residual still ~0.27 cross-corr | beta_confound.py |
| Moves ≠ edge (random entry = coin flip) | next-day up-excursion net-taker 46.8% positive, median ~0% | random-entry null |
| Opportunity broad not concentrated | top-10 assets = 20% of up-mover-days; 57 assets to reach 80% (Gini 0.36); TAO/ZEC/RENDER/FET/WIF lead | movers_analysis; MARKET_RESEARCH §3b |
| Opportunity is bursty | mover-count lag-1 autocorr 0.26; high-opp day follows high 66% of time; LO right-tail favorable (P95 up 13.5 vs down 11.3) | timing_asymmetry.py |
| Survivorship is real + unquantifiable | within survivors move-density age-independent; the delisted left-tail is unobservable; "momentum is an illusion net of survivorship" (Grobys 2025) | MARKET_RESEARCH §4,§6 |

## B. The governing dynamics (the 3 facts that bind everything)
| Finding | What it says | Evidence |
|---|---|---|
| **Direction unpredictable at every TF** | Hurst ~0.5; AC1 −0.018(1d)→−0.053(30m); GBM AUC 0.508–0.531 ≈ logistic; corr 0.013–0.048 = noise | CHIMERA_MINING §P11 (RWYB) |
| **Volatility/magnitude IS predictable + sharpens intraday** | AC1(\|ret\|) 0.18→0.33; 100% assets vol-persistent; feature→\|ret\| corr 0.04–0.07; `norm_oi_price_divergence` dominant | CHIMERA_MINING §P7–9 (RWYB) |
| **Vol-expansion → 1.5–2.0× bigger next move at COIN-FLIP direction** | up-rate 0.49 every TF; "something big coming, direction unknown" | CHIMERA_MINING §P12 (RWYB) |
| **One-factor (BTC) market at every resolution** | median pairwise corr ~0.55; BTC-beta ~1.19; 46/50 one cluster; no lead-lag ≥15m (best-lag 0) | CHIMERA_MINING §F2 (RWYB) |
| **MR strongest in bear/below-MA regime** | regime0 AC1 −0.060(1d), −0.076(15m) vs weakest in bull | CHIMERA_MINING seasonality (RWYB) |
| **Cross-sectional 1wk reversal — LIQUIDITY-TIER SIGN-FLIP** | reversal driven by illiquid small-caps; **liquid majors MOMENTUM not reversal** | verified 2026-06-09; Liu et al. |

## C. The cost / execution reality
| Finding | What it says | Evidence |
|---|---|---|
| Cost is the binding constraint | taker RT ~0.15–0.30%; only 6/22 popular strats net-+ after fees; 30m gross −89.5% in MA sweep | FOUNDATION §7-C; StratProof |
| Maker is the #1 lever — but p_fill 0.21–0.40 | live = 50–75% of fixed-backtest; adverse-sel high; maker can't make sub-daily viable alone | maker_cost_calibration |
| Intraday vol seasonality | concentrates 14–16 UTC (US open) + 00 UTC (funding); weekends quieter; turn-of-candle at 0/15/30/45m | CHIMERA_MINING; PMC turn-of-candle |

## D. Asset archetypes (the same signal means different things)
| Archetype | Ann vol | BTC-beta | Marginal trader | Survivorship | Signal reading |
|---|---|---|---|---|---|
| BTC | 50–60% | 1.0 | institutional/ETF | permanent | funding/whale/OI = real institutional; signals trustworthy |
| ETH/large-cap L1 | 70–100% | 1.2–1.8 | crypto-native inst.+ETF | low | same, noisier; narrative-reflexive |
| mid-cap / DeFi | 100–200% | 1.5–2.5 | retail+DeFi funds | high (unlock overhang) | TVL-flow real; spot whale ambiguous |
| **MEME** | >200% (daily 12–53%) | <0.3 explanatory | retail+bots+single-whale | maximal (97% fail; 68% Solana rugs <72h) | funding/whale/depth = MANIPULATION surface, not info |
| stablecoin | <1% (tail −30/−65%) | ~0 | arbitrageurs | depeg tail | return-based signals INVERT (peg process) |
**Rule:** trust order-flow/depth/whale/positioning IN PROPORTION TO LIQUIDITY DEPTH. (Full matrix: feature dictionary.)

## E. The data substrate (chimera)
| Finding | What it says | Evidence |
|---|---|---|
| 215 features, 12 families, 27-dim effective | 40 dense feats need 27 PCs for 90% var (PC1 ~10%, stable ALL cadences); 2/780 pairs >\|0.8\| | CHIMERA_MINING §F3 (RWYB) |
| Features are NOT the bottleneck | rich + non-redundant + cadence-stable; the open problem is the regime-conditioned cost-aware MAPPING | CHIMERA_MINING §F3 |
| Family-uneven sparsity | norm/xd/stbl/fund dense (0–7% missing) universe-wide; LOB 91%, xex 97%, DVOL 98% missing (major-only) | CHIMERA_MINING §F5 |
| Dollar bars = modeling tool not move-source | equal-activity → tiny per-bar moves; value is statistical (stationarity) | MARKET_RESEARCH §3b |
| Time bars derived from dollar bars (~75s floor) | "intraday" study without native loaders runs on DAILY (the load_panel bug); resolution floor ~75s BTC | FOUNDATION §7-A |
| SMA-200 is the durable ex-ante regime label | parameter-free, no look-ahead; HMM Viterbi carries hidden look-ahead (post-hoc only) | FOUNDATION §7-F |
| 5 empirical GMM regimes (BTC-cohort) | quiet-chop 49%, topping/distribution 20% (−0.035/bar trap), uptrend 13%, downtrend-bounce 12%, euphoria 5%; persistence rises intraday | CHIMERA_MINING §F4 (RWYB) |
| Asset-DNA buckets (BLUE/STEADY/DEGEN/VOLATILE) | precomputed; median top-Sharpe BLUE 1.20 > STEADY > DEGEN > VOLATILE 0.86 | asset_dna_u100 |
| Universe tiers u10/u50/u100 declarative | is_u10/50/100 + asset_dna inline in chimera; 104 assets u100, 77 at 30m | CLAUDE.md |

## F. Structural mechanics (the perp/on-chain machinery)
| Finding | What it says | Evidence |
|---|---|---|
| Perps dominant (>90% deriv vol); funding is the price-anchor | 2025 deriv ~$85.7T; funding 8h settlement = crowding/sentiment gauge | CRYPTO_MARKET_UNDERSTANDING §I |
| Liquidation cascades: causal-but-bounded on majors, terminal on memes | post-cascade bounce = structural regularity on BTC/ETH; OI spike precedes (measurable warning) | CRYPTO_MARKET §III; SSRN |
| Liquidation cascade is the standout vol-coincident avenue | fires ~5–6% of days; same-day \|move\| 1.24×; forward-predictive = OPEN (reverse-causality caveat) | MARKET_RESEARCH §5 |
| Funding carry decayed | CME basis ~25%(Feb-24)→~4.5%(Dec-25); 93% of days <5% breakeven (ETF arb) | CRYPTO_MARKET §III |
| Stablecoin supply = macro liquidity regime | expansion = fresh capital (top bull gauge w/ ETF); ETF flows = most explanatory BTC variable post-2024 | CRYPTO_MARKET §I,III |
| Institutionalization partial; dampens reflexivity but retail leverage = vol texture | CME>Binance BTC OI; ~1.05M BTC in ETF/treasury; yet $19B retail-liq days persist | Grayscale |
| MEV = on-chain tax (~$3B/yr ETH) | only matters for DEX-venue execution | ESMA |

## G. The binding foundation conclusion
**At daily/4h/dollar resolution under LO+spot+lev=1, there is NO verified active-trading alpha — everything robust is
beta + yield** (confirmed 6 independent ways). This is the limit of the *constraint*, NOT a global ceiling. The
unexplored frontier is: relaxing the constraint (long-short, leverage) OR finer resolution (sub-15m / event-clock /
info-driven bars, whose chimeras are mostly EMPTY) OR the vol/magnitude channel (options/convexity, ingest absent).
This is the **A/B/C fork** (see [05_OPEN_THREADS.md](05_OPEN_THREADS.md) thread 5) — the project's blocking decision.

**Methodology pointer:** the oracle-decomposition method (treat ROI as oracle-attainment, decompose its DNA, build a
capture-rate proxy, with min-move-net as the scalp-vs-swing design variable) is the canonical way to test any new
avenue empirically before committing — it inverts bottom-up trigger-guessing.
