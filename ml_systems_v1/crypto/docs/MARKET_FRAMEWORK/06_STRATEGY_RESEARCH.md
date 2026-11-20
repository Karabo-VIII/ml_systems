# 06 — Strategy Research: the strat-layer founding survey (2026-06-09)

**What this is.** The exhaustive, SOTA-cited research that founds **STAGE 03 (strat)** of the
[Solutioning Pipeline](../SOLUTIONING_PIPELINE.md). The user's next step is the strat layer — *using the decomposed
data to mine for strategies and technical indicators*. Before writing a single strategy, this doc consolidates: (A) a
crypto **strategy taxonomy** (CeFi + DeFi/microstructure) mapped to our dead-list + chimera + maker-cost reality; (B) the
**chimera coverage + gap map**; (C) **SOTA technical indicators** vs what we already have; (D) the SOTA of **strategy
DISCOVERY methods** — how to find *new* strategies honestly; (E) the **prioritized first-build shortlist**.

> Built 2026-06-09 by a 5-scout `/orc` research run (4 Sonnet surveys + 1 Opus discovery-methods node), each cited and
> RWYB-mapped to our own code. **No strategy is built yet** — this is the research/ingredient layer (stage-03 gate is
> still `FALSE`, blocked on the A/B/C fork, thread #1 in [05_OPEN_THREADS](05_OPEN_THREADS.md)). Provenance:
> user mandate *"do research on crypto strategies as they relate/differ from our chimera set… exhaustive + SOTA…
> as well as approaches on how to discover new ones."*

**Read [01_DEAD_LIST](01_DEAD_LIST.md) first.** Many families below are near-neighbours of D01–D63. This doc tags each
against that list so the strat layer never re-mines a refuted vein as if it were live.

> **Number convention:** all Sharpe / CAGR / yield figures are **REPORTED** from the cited literature unless tagged
> VERIFIED (we measured it, RWYB) or INFERRED. Null-rates / coverage-% are RWYB measurements from our own parquet.

---

## The 4 binding constraints (every recommendation is filtered through these)
1. **Maker-or-it-doesn't-ship.** Taker round-trip ≈ **0.24%** (`candidate_gate.TAKER_COST_RT = 0.0024`); real maker
   `p_fill ≈ 0.21–0.40` (not 0.80). Sub-daily-at-taker is a cost cliff (30m gross −89.5%, D60). (D43, D50, D60, A7)
2. **One-factor (BTC) market.** corr ~0.55, beta ~1.2, no exploitable lead-lag ≥15m → **time the market, don't pick the
   asset**. Cross-sectional models that assume 300–3000 names are statistically hollow on 10 coins. (Market model #3; D17)
3. **Direction is unpredictable; magnitude/vol is predictable.** AUC 0.508–0.531, Hurst(ret)≈0.5, IC≈0 six ways (D55,
   D44, D17). Vol clusters: Hurst(|ret|) 0.80–0.84, AC1|ret| 0.18→0.33 (RWYB econometric_signature). The honest edge
   channel is convexity/vol/sizing, **not** a next-bar direction bet.
4. **Daily/4h + LO+spot+lev=1 floor = beta+yield**, not verified active alpha (confirmed 6 ways; Market model #33). This
   is the *constraint's* limit, not a global ceiling — relaxing it is exactly the A/B/C fork.

---

## A. Crypto strategy taxonomy — organized by viability AT OUR LAYER

Merged CeFi + DeFi/microstructure survey. Status = our dead-list verdict; **Viability** = build-readiness under the 4
constraints. Full per-family theses, sources, and decay numbers are in the source-scout appendix (Part F).

### TIER 1 — viable at our layer, buildable now (data in hand, maker-compatible)
| Family | Edge thesis | Our chimera coverage | Dead-list status | Note |
|---|---|---|---|---|
| **Vol-targeting / vol-managed overlay** | Scale position inversely with realized vol → higher Sharpe (vol clusters, returns don't scale with vol). Moreira–Muir 2017; Habeli 2024 (crypto). | `norm_yz_volatility`, `rv_rv_5m`, regime vol-state — full | **No refutation** (D02 is about voladj *targets*, not sizing) | **Zero-execution overlay on everything.** The single least-contested win. |
| **Cross-sectional reversal / relative-value (liquidity-tier-aware)** | Illiquid tail reverts; liquid majors persist. Maker-filled, daily-formation, tier-separated. | `xd_momentum_rank`, `xrel_*` ranks across families | **REFUTED-HARD as *naive pooled* alpha** (D17, D40, D53); the **tier-separated, market-neutral, maker** version is UNEXPLORED | Survivorship-clean universe mandatory (Grobys 2025 inflates 4×). Short-leg = borrow + manipulation guards. |
| **Funding-extreme *contrarian signal*** (NOT the carry trade) | Funding extremes → price mean-reversion; a conditioner, not a yield trade. | `fund_rate_z30`, `bs_basis_panic/frenzy`, `fund_extreme_long_count` (derivatives, 0.5% null) | D42 (sign-care: high funding → *higher* fwd ret via momentum), D18/D54 scoped | Use as an overlay/gate on a base strategy, never asserted alone. |

### TIER 2 — conditional / gated (viable but needs a load-bearing filter or finer cadence)
| Family | Edge thesis | Coverage | Dead-list status | Gate / condition required |
|---|---|---|---|---|
| **Regime-gated TSMOM / managed-futures** | Vol-scaled, regime-filtered trend (AdaptiveTrend Sharpe 2.41 vs naive 0.65; REPORTED). | `structure`, `momentum`, `regime_label` | **Naive = REFUTED-HARD** (D63, D21); SOTA vol-scaled+gated = conditional | Regime filter must *independently* improve a gated metric; must beat buy&hold **Calmar** in MIXED-regime UNSEEN (not just abstain in bear). |
| **Funding-carry yield sleeve** | Delta-neutral spot-long/perp-short carry — but compressed to ~4–5% APY post-ETF (BIS WP1087; REPORTED). | derivatives family | **REFUTED-SCOPED as primary alpha** (D18) | Survives only as a thin yield sleeve in elevated-funding regimes; event-spike arb (~17% of obs) as tactical overlay. |
| **Liquidation-cascade fade** | Post-cascade exhaustion bounce on majors (mechanism real; same-day |move| 1.24× on liq days). | `liq_*` (13 feats, 0.1% null at time-bars) | **REFUTED-HARD at daily/4h** (D47, D48, D49 — buy-the-extreme is an anti-edge) | Needs **sub-4h event-clock** entry + a strict "exhaustion confirmed" gate (NOT a spike-buy) + ideally pre-event limit placement from OI-delta. This is the A/B/C fork. |

### TIER 3 — data-blocked but *addable* (not infra-bound; an ingest sprint unlocks them)
| Family | Why blocked | Ingest route | Priority |
|---|---|---|---|
| **Pre-cascade liquidation-heatmap proximity** | We have the cascade *aftermath*; not the pre-event level map | Coinglass API → `liq_cluster_proximity_pct` | **High** — turns a lagging signal into a leading one, no HFT |
| **On-chain exchange netflow / stablecoin-to-exchange** | We have aggregate `stbl_*` supply + CEX-side `wh_whale_*`, not on-chain wallet→exchange flow | Glassnode / CryptoQuant API | High — documented 48–72h lead; cleaner than supply z-score |
| **Cash-and-carry / CME basis + COT positioning** | We have perp-basis, not CME term structure or CFTC large-trader COT | CFTC COT (free, weekly) / CME DataMine (paid) | Medium — institutional-positioning signal, post-ETF relevant |
| **Options surface (Deribit beyond DVOL)** | We have `dv_dvol_*` (98% null off BTC/ETH); no skew/term/gamma | Deribit / tardis.dev | Medium — the convexity/vol channel (constraint #3) lives here; Fork-B candidate |

### TIER 4 — structurally closed (infra-bound; not our game)
MEV (sandwich/backrun/JIT — mempool + Flashbots/Jito), CEX↔DEX arbitrage ($234M/19mo across 19 searchers, top-3 take
75%), market-making / HFT spread-capture (co-location moat; our p_fill 0.21–0.40 already reflects second-tier queue
position), passive AMM/LP provision (on-chain position management + IL modeling). **All require execution infrastructure
we do not have and will not build.** Use their CEX-side echoes (`xex_*` cross-exchange spreads) only as *stress/regime*
features, never as execution triggers.

---

## A2. The convexity / magnitude channel — the honest edge, and why it's options-gated (decision-relevant for the fork)

Our market model's sharpest fact: **direction is unpredictable but magnitude/vol IS predictable** (Hurst|ret| 0.80–0.84,
GARCH persistence ~1.0, vol-expansion → 1.5–2× move at coin-flip direction). So "trade magnitude, not direction" is the
theoretically-honest edge. A dedicated deep-dive (cited) asked: **can predictable magnitude be monetized under Fork A
(maker, daily/4h, no options)?** The rigorous answer is **no, not as standalone alpha** — and two tempting "Fork-A
convexity" ideas are explicit traps:

- **Synthetic long-gamma via perp re-hedging is a FALSE analogy (the key falsification).** Spot+perp has **zero gamma** —
  delta is fixed at ±1.0, there is no second derivative. "Re-hedging a delta-neutral spot/perp book on each move" is NOT
  gamma scalping; it's a direction-agnostic taker scalp. The arithmetic: pros re-hedge 4–8×/day; at ~0.24% taker RT that's
  ~2%/day cost vs a ~0.035%/day realized-vs-implied vol edge — **cost ≈ 57× the edge.** Maker-only re-hedging hits the D43
  p_fill 0.21–0.40 wall (fills fail exactly during the fast moves you need). → see **D64**.
- **Vol-expansion "perp straddle" (simultaneous long+short to capture the move) is a cost trap.** Two entries at the same
  price = immediate offset + 2× taker RT (~0.48%); staggering the legs re-introduces a one-bar directional bet at coin-flip
  odds. → see **D65**. (Vol-expansion is a valid *sizing* signal, not a standalone entry — D41/F9.)
- **Dispersion/correlation** is structurally thin in a one-factor (BTC-beta ~0.55) market with no index-options product. → **D66**.

**What survives:** **vol-targeting as a SIZING OVERLAY** on an existing directional edge (SOTA = HAR-RV / smoothed-SV
scaling, not naive inverse-vol; Moreira–Muir; crypto-momentum vol-management confirmed). It improves risk-adjusted
return but **does not generate alpha on its own** — it shapes the loss profile of whatever edge (or beta) it scales.

**The fork implication (rigorous):** predictable magnitude is real but **its standalone monetization requires options
(Deribit ingest = Fork B)**; the smallest Fork-B step is DVOL futures (BTC/ETH, no full chain needed). Under **Fork A**,
the only correct architecture is *a (however small) directional/relative-value edge × vol-targeting sizing* — and if no
directional edge exists at daily/4h, vol-targeting merely shapes the beta. This tightens the fork: **Fork A's ceiling is
"directional-edge × vol-sizing"; genuine magnitude-alpha is Fork B.** VRP (sell IV>RV, 71% of days) is a Fork-B *yield*
sleeve, not return-enhancement (covered-calls REFUTED, D38: −24pp).

## B. Chimera coverage + gap map (the "what we have to work with" baseline)

**12 families, 218 curated features** (`src/narrate/feature_map.py`), RWYB null-rates from live parquet.

**Densest / most crypto-native (build here first):**
- **derivatives** (28 feats, 0.5% null BTC, clean to 15m): funding z, basis panic/frenzy, OI-price divergence, premium carry.
- **liquidation** (13, 0.1% null at time-bars): forced-flow magnitude + z + capitulation + x-sectional idiosyncratic.
- **cross_asset ETF/stablecoin sub-block** (14, dense post-2024 at 1d/4h): the macro-liquidity regime signals (most
  explanatory BTC variable post-2024).
- **structure / momentum / volatility / orderflow / whale**: dense across all time-bars.

**The explicit gaps (precise):**
| Gap | Ingest route | Infra-bound? |
|---|---|---|
| Raw LOB depth beyond L5 (L10/L20, cancels, queue position) | Binance `depth@100ms` collector | No — ~1–2d producer, storage-heavy |
| On-chain / mempool (netflow, SOPR, UTXO age, active addr) | Glassnode / Dune / Nansen | No if API available (~2–3d) |
| Options full surface (Deribit IV skew/term/gamma) | Deribit / tardis.dev | No — ~3–5d, large volume |
| MEV / mempool priority | Flashbots / Etherscan | **Yes** — DEX-execution only, out of scope |
| Miner flows (BTC) | Glassnode / CryptoQuant | No, but low value for LO-perp universe |
| CME futures basis / COT | CFTC COT (free) / CME DataMine (paid) | No — CFTC is free+weekly |
| Social depth (X/Reddit/LunarCrush) | LunarCrush / Santiment | No (~1d) — only `soc_wiki_views` today (62% null even BTC) |
| Macro / TradFi (SPY/VIX/DXY/rates) | FRED / Yahoo (free) | No — trivial daily ingest |
| Sub-5m tick microstructure | raw tick store | Partially infra-bound |

**Resolution cliff (RWYB):** derivatives + liquidation survive cleanly through 15m on **time-bars** but go **83–100%
null on dollar/dib/range bars** (liquidation = 100% null in dollar-bar). Positioning is 0% null on majors but **100% null
on memes** (PEPE). Dollar/dib/range chimeras are currently **OHLCV + structure/momentum only** — every domain family
(derivatives, liquidation, whale, positioning, social) is hollow there. *Info-bar strategies cannot yet use the
crypto-native signal families.*

---

## B2. The Fork-B data-ingest shopping list (concrete feasibility — grounds the fork's data-engineering half)

The Tier-3 gaps are "addable, not infra-bound" — but at very different cost/EV. Priced (cited) by EV-per-engineering-day,
with the honest dead-list collision per source. **The headline: Fork-B's cheapest, cleanest first step is FREE.**

| Rank | Source | Cost | Effort | Unlocks | Dead-list collision (the honest counter) |
|---|---|---|---|---|---|
| **1** | **CFTC COT** (CME BTC/ETH large-trader positioning) | **FREE** (public API + `cot_reports` lib) | ~1–2d | institutional-positioning **regime conditioner** (post-ETF; CME OI at records) | **NONE** — cleanest profile; but weekly cadence + 3d lag → a regime switch/multiplier, not a bar-level signal |
| **2** | **CryptoQuant** stablecoin-to-exchange *directional* flow | ~$25–100/mo (API tier) | ~2–3d | the directional complement to our supply-only `stbl_*` (`onchain_exchange_netflow`) | **D55-adjacent** — academic edge is *intraday* (1–6h); daily lead is INFERRED not verified → test in `candidate_gate` immediately, **drop if null** |
| **3** | **Coinglass** liquidation heatmap | $29/mo | ~3–5d | pre-cascade `liq_cluster_proximity_pct` (turns our aftermath-only `liq_*` into a leading signal) | **D47/D48/D49 HARD at daily/4h** — only helps WITH a sub-4h cadence sprint; daily-only ingest = a better input to a still-dead strategy |
| 4 | **DefiLlama** TVL/bridge flow | FREE | ~1d (reuses `src/pipeline/ingest/defillama_stable_flows.py`) | macro DeFi regime | **TVL-price circularity** (G-AUDIT-011 class) + overlaps existing `stbl_*` → low incremental value |
| 5 | **Deribit** full options surface (IV skew/term/gamma) | ~$700+/mo (tardis.dev) | ~5–7d | the convexity/VRP unlock (the §A2 magnitude channel) | **D64/D65/D38** — convexity-via-perp refuted; VRP = *yield* sleeve only; BTC/ETH-only; Fork-B-deep |
| 6 | **Glassnode** on-chain (netflow/SOPR/miner) | **$999+/mo** (API tier) | ~3–5d | richer on-chain flow | cost-prohibitive for a daily signal we partly replicate with `stbl_*`; same D55-adjacent risk as #2 |
| 7 | **LunarCrush/Santiment** social depth | tiered | ~1–2d | replaces near-hollow `soc_wiki_views`; meme coverage | **D12 SCOPED** (social ≈ 0 predictive in bull); small effect (Cohen's d≈0.21) |

**The fork takeaway:** Fork B does NOT require a big upfront spend. The rational sequence is **COT (free, clean) → test
CryptoQuant netflow cheaply (drop if daily-null) → only then consider the sub-4h cadence sprint + Coinglass** (the
cascade avenue) **or the options surface** (the convexity avenue) — both of which are the genuine capital/effort
commitments. Every ingest is still gated by `candidate_gate` + `pbo_cscv`; "addable" data ≠ an edge (most feed a
dead-list-refuted or daily-null strategy).

## C. SOTA technical indicators — HAVE-IT / DON'T-HAVE / LOOK-AHEAD-RISK

Our panel is already SOTA-deep on causal microstructure + vol: **HAVE** fractional-differencing (FFD, causal, `frac_diff_fast`
d=0.4), Kaufman efficiency (`norm_efficiency`), VPIN (`norm_vpin`), Kyle's lambda ×4 variants, Hawkes intensity + branching
ratio (full suite), order-flow imbalance, Yang-Zhang vol (`norm_yz_volatility`), realized variance + **bipower** (jump-robust)
from 5m, DVOL, permutation entropy, Hurst regime, transfer entropy. `econometric_signature.py` adds GARCH/DFA-Hurst/Hill/BNS
as **whole-series signature diagnostics** (not per-bar tradeables).

**Highest-value CAUSAL indicators we DON'T have (worth adding for the strat layer):**
1. **Realized skewness / kurtosis from 5m** (`rv_skew_5m`, `rv_kurt_5m`) — pure engineering add (the 5m pipeline already
   powers `rv_rv_5m`); RS predicts crypto returns via lottery-preference reversal (Lee/Wang 2024). **Top pick.**
2. **Kalman-filter velocity state** — causal state-space level+velocity, regime-adaptive vs static-window momentum.
3. **Amihud illiquidity on time-bars** — currently dropped (correctly) for dollar-bars where volume is constant; valid on
   1d/4h, interpretable on assets without full LOB (memes/micro-caps).

**Look-ahead traps to avoid (G-AUDIT-011 class):** non-causal wavelets / EMD / Hilbert-MESA — most published crypto TI
results using these leak future info via full-signal envelope fitting. **Causal wavelets were probed in-house and found
NULL at daily resolution** (don't re-mine — D-context). Only add with a verified streaming-causal implementation.

> **Data-dictionary drift (RWYB, FIXED 2026-06-09):** `norm_fd_close` is *computed* as fractional-differentiation
> (`frac_diff_fast`, `sota_shared_logic_v50.py:474`) but was *labeled* "Fractal dimension" in `feature_map.py:97` — two
> different concepts sharing the column label. Corrected the dictionary entry to "Fractional differentiation (log-close)"
> so a strategy reading the feature map gets the right semantics.

---

## D. Strategy DISCOVERY methods — how to find NEW strategies (the load-bearing section)

**The one-sentence thesis (Opus discovery node, verified vs our code):** the published alpha-mining frontier
(AlphaForge, AlphaAgent, AlphaGen, QuantEvolve, MCTS/GFlowNet) is almost entirely **cross-sectional, equity, daily,
IC-optimized, and un-deflated** — a *different problem* from ours. **Their generators are adoptable; their evaluation is
not.** Our `candidate_gate` is already stricter than every honesty control in any of those papers. The winning move is to
**bolt their cheap, high-throughput formula GENERATORS onto our gate as the back-end — never the reverse.**

### Ranked menu of discovery methods to adopt
| Rank | Method | Why it fits us | Integration path | Status |
|---|---|---|---|---|
| **1** | **Oracle-decomposition** (ours, built) | Top-down, returns-native (not IC), per-setup, crypto-tuned — it *is* "generate then deflate" in our objective | Already wired: `oracle_ceiling_builder` → DNA fit (shuffle + positive control) → `candidate_gate` | **HAVE** — make it the default front-end |
| **2** | **Genetic-programming formula proposer** (gplearn/gpquant) | Cheapest, highest-throughput proposer; its #1 failure (multiple testing) is exactly what DSR-at-true-N kills | New `src/strat/gp_proposer.py`: GP over **lookahead-safe** chimera features → materialize survivors as past-only columns → `evaluate_candidate(family_n = TOTAL GP trials)` | **BUILD** |
| **3** | **LLM hypothesis proposer (judge-STRIPPED)** | Imports cross-domain priors, reads our chimera dict + dead-list, proposes conditioners a human wouldn't | New `src/strat/llm_proposer.py`: prompt = feature dict + D01–D63 + "propose a past-only conditioner + falsifiable mechanism"; **LLM emits NO verdict** — mechanism tested by our falsifier | **BUILD** |
| **4** | **GFlowNet diversity sampler** (AlphaSAGE import) | Counters GP mode-collapse → a genuinely diverse family → honest DSR `Var(SR_trials)` | Optional upgrade to #2's sampler once GP saturates | DEFER |
| **5** | **AlphaForge dynamic recombination** | Regime-conditional reweighting of a *shipped* book | Only after ≥2 independently-gated uncorrelated edges; gate against D29/D30/D33 (correlated-aggregation refutations) | DEFER |
| **NEW** | **PBO / CSCV primitive** (Bailey–López de Prado) | **Our one true methodological gap** — battery carries DSR only as a *caller-note* (`battery.py:133`), computes no PBO | **`src/strat/pbo_cscv.py`** (BUILT 2026-06-09): T×N candidate-family matrix → C(S,S/2) symmetric IS/OOS splits → PBO = P(IS-best is OOS-underperformer). Integration point = the **discovery-search layer** (the proposers call it on their candidate family before promoting; PBO < 0.10), NOT per-candidate battery (battery evaluates one strategy across windows — wrong granularity for PBO). | **BUILT — selftest two-sided PASS (genuine PBO 0.00 vs skill-less 0.39)** |

**Net recommendation:** adopt #1 (have) + #2 + #3 + the PBO primitive. ~3 small modules, all feeding the *unchanged*
`candidate_gate`. Skip #4/#5 until the basics saturate.

### The discovery anti-pattern dead-list (what NOT to do)
1. Report the survivor's IC/Sharpe **without deflating by the TRUE trial count** (N = full search size 10⁵, not the count
   you kept) — the #1 sin of the whole literature.
2. **Optimize IC/RankIC** as the objective (D13 HARD — our unit is a multi-candle setup, not a per-bar prediction).
3. **Cross-sectional factor models on 10 assets** (FactorVAE/HIST/AlphaForge assume 300–3000 names; D17/D37 HARD).
4. **Let an LLM (or any model) JUDGE overfitting/validity** — mechanism must be *empirically falsified*, never asserted by narrative.
5. **Future info in training** (prior-posterior / FactorVAE-style) without auditing the inference path (A2 / G-AUDIT-011 class).
6. **Combine correlated factors and call it diversification** (D29 HARD — jackknife collapse).
7. **Select on / re-read the held-out set across the search** (UNSEEN-once invariant; A5).
8. **Same-bar-close / inline-backtest fill** when materializing a formula (A3, +50–80pp) — route every candidate through `CanonicalHarness` (next-bar fills).
9. **Trust maker p_fill = 0.8** (D43 HARD) — every candidate runs at taker 0.0024 by default.
10. **No memory of failures → re-mine dead veins** — wire D01–D63 + `skill_library` as the generator **pre-filter**.
11. **Count a backtest with 5+ tuned knobs as one trial** — "Pseudo-Mathematics": ~5 trials buys a Sharpe-1 strategy from noise on 7yr daily data (REPORTED, Bailey et al.).

### SOTA discovery verdict
For a **maker-cost, daily/4h, one-factor, 10-asset crypto** market, the real frontier is **not a better generator but a
better *deflator wrapped around a cheap, diverse generator*.** Keep oracle-decomposition as the returns-native front-end;
bolt a high-throughput GP proposer + a judge-stripped LLM proposer onto it; close the PBO/CSCV gap. The edge is
*throughput-of-honest-trials* — our gate is already stricter than the literature, so feed it 10³–10⁵ diverse cheap
candidates and let the deflated gate find the rare survivor, **accepting that the honest base-rate of survival is very low
and a null at daily/4h is the most likely (and still valuable) outcome.**

---

## E. The prioritized strat-layer entry shortlist (the first builds)

Dead-list-aware, cost-aware, ordered by EV-under-constraints. **None of these is a strategy yet** — they are the ranked
candidates the stage-03 build phase should take through `candidate_gate`, *after* the A/B/C fork is chosen.

1. **Vol-targeting overlay** (Tier 1) — build first; zero-execution; goes on top of everything; no dead-list risk.
2. **Liquidity-tier-aware cross-sectional reversal/RV book** (Tier 1) — the top *standalone* candidate the corpus points
   to; maker-filled, daily-formation, survivorship-clean, tier-separated. Discovery method: oracle-decomposition + GP proposer.
   **DEAD-LIST RECONCILIATION (do not skip — this is a near-neighbor of a HARD refutation):** D53 (within-group
   relative-value *reversion*) is **0/72 at 4h, HARD — "continuation dominates"**; D49/D52 say cross-sectional/micro
   *reversal* is an anti-edge; D40 says cross-sectional *momentum* is HARD as a standalone. So the only genuinely-open
   cell is narrow and carries an **elevated burden of proof**: the *reversal* leg is **illiquid-tail-ONLY + daily-formation**
   (the tier+cadence D53 didn't test), and the **liquid-majors leg must be MOMENTUM, not reversal** (D40/D53: continuation
   persists there). A "cross-sectional reversal" applied to liquid majors, or within-group RV at 4h, would be re-mining a
   refuted direction. Mandatory: pre-register the tier boundary + run through `pbo_cscv` (search-level) THEN `candidate_gate`.
3. **Regime-gated vol-scaled trend** (Tier 2) — second; the regime filter is load-bearing and must pass the independent-
   improvement + mixed-regime-Calmar test. Discovery: oracle-decomposition.
4. **Funding-extreme contrarian conditioner** (Tier 1 signal) — not standalone; an overlay/gate that must improve a base
   strategy's metric. Discovery: GP/LLM proposer over the derivatives family.
5. **(Fork-B, data-sprint first)** sub-4h event-clock **liquidation-cascade fade** + **pre-cascade heatmap proximity**
   (Coinglass ingest) — the highest residual-return avenue, but it requires the cadence/data investment the A/B/C fork decides.

**Cross-cutting build infrastructure (do alongside #1–#2):** `gp_proposer.py`, `llm_proposer.py` (judge-stripped), and
**`pbo_cscv.py`** — the discovery factory + the honesty primitive that lets us run it at scale safely.
**`pbo_cscv.py` is BUILT (2026-06-09)** — the discovery-search-level deflator (CSCV → PBO), selftest two-sided PASS
(rejects a skill-less family, accepts a genuine one); the two proposers remain TODO and will call it on their
candidate families before promoting a winner through `candidate_gate`.

**The fork, restated by this research:** everything viable at our *current* resolution+data is yield/relative-value/vol
(Tiers 1–2). The residual higher-return avenues (sub-4h cascades, options convexity, finer microstructure) all require
relaxing the constraints — finer cadence or new instruments (Tier 3 ingest / Fork B). This survey does **not** find a
missed signal hiding in the existing daily/4h universe; it confirms the fork is a structural data+cadence decision.

---

## F. Sources (consolidated, all RWYB-tagged in the scout appendices)
**Strategies (CeFi):** Han/Kang/Ryu 2024 (SSRN 4675565); Grobys 2025 survivor-momentum; Liu et al. liquidity-tier;
AdaptiveTrend (arXiv 2602.11708); Moreira–Muir 2017 (SSRN 2773438); Habeli 2024 (SSRN 5090097); BIS WP1087 crypto-carry.
**DeFi/microstructure:** CEX-DEX value (arXiv 2507.13023); LOB patterns (arXiv 2602.00776); JIT liquidity (AFT 2025);
VPIN-jumps (ScienceDirect S0275531925004192); Oct-2025 cascade (SSRN 5611392); two-tier funding (MDPI Math 2025).
**Discovery:** AlphaForge (arXiv 2406.18394); AlphaAgent (arXiv 2502.16789); LLM-MCTS (arXiv 2505.11122); AlphaGen
(github RL-MLDM/alphagen); QuantEvolve (arXiv 2510.18569); FactorMiner (arXiv 2602.14670); **Deflated Sharpe** &
**PBO/CSCV** (davidhbailey.com/dhbpapers: deflated-sharpe, backtest-prob).
**Indicators:** López de Prado FFD (Hudson&Thames); Lee/Wang 2024 realized-moments (Scheller/GT); higher-order moments
(ScienceDirect S105905602030294X); Kalman crypto (PyQuantLab); Yang-Zhang (FlashAlpha).

*(Full URL lists + per-family decay numbers are preserved in the 2026-06-09 scout outputs; this section is the index.)*
