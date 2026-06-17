# Strategy Playbook (2026-06-09)

**Purpose.** The synthesis that turns *understanding* into *executable strategy*: which strategy archetypes to build,
why (grounded in BOTH our empirical mining AND the external literature), which chimera features feed them
(asset-conditionally), what regime/instrument/timeframe they need, and the non-negotiable validation discipline that
makes "no failures" real. This is the menu we pick from when we touch the first strategy.

Built on: [CRYPTO_MARKET_UNDERSTANDING.md](CRYPTO_MARKET_UNDERSTANDING.md) (cited market model) +
[CHIMERA_FEATURE_DICTIONARY.md](CHIMERA_FEATURE_DICTIONARY.md) (feature meanings + asset-conditional) +
[CHIMERA_MINING_FINDINGS_2026_06_08.md](CHIMERA_MINING_FINDINGS_2026_06_08.md) (our RWYB decomposition) + the
`src/strat/candidate_gate` validation apparatus.

## 0. The foundation (what we KNOW — our mining and the literature agree)
- **Direction is ~unpredictable** at canonical TFs, linearly AND nonlinearly (our GBM AUC 0.51-0.53 ≈ logistic; the
  literature's Hurst≈0.5, insignificant return ACF). → directional bets are the weakest foundation.
- **Volatility/magnitude IS predictable** and clusters, strengthening intraday (our AC1|ret| 0.18→0.33; GARCH long
  memory). → the surest edge channel.
- **One-factor market** (corr ~0.55, BTC-beta ~1.2, no exploitable lead-lag ≥15m). → time/regime > asset-selection.
- **Cross-section reverses at ~1 week**; dispersion is abundant (~10 movers>5%/day). → relative-value, not absolute
  direction.
- **Funding/basis extremes mean-revert**; **liquidations overshoot then exhaust** (bounded on majors). → reflexivity is
  tradeable as a *conditioner*.
- **Cost is the binding constraint** (~0.15-0.30% taker round-trip); **maker execution (~0.05-0.10%) is the #1 lever.**

## 1. Design principles (binding for every strategy we build)
1. **Trade magnitude/convexity, not direction** — build around vol-expansion, breakout, and variance, where the
   predictability actually lives; only take directional exposure where a *conditioner* (regime, carry, flow, x-sectional
   rank) tilts the coin, never raw price-MA.
2. **Time the market; don't pick the asset** — in a one-factor market, the exposure switch (on/off/scale) matters more
   than which alt. Regime-gate everything.
3. **Maker execution or it doesn't ship** — at finer TFs the cost cliff is fatal (our 30m gross −89.5%). Limit/maker
   fills are a hard requirement for any sub-daily strategy.
4. **Asset-archetype-aware features** — a feature's trust scales with liquidity depth (dictionary §matrix). Never read a
   meme's funding/whale/depth like BTC's. Exclude stablecoins/pegged from return-based signals.
5. **Convexity beats prediction** — "a bigger move is coming, direction unknown" (our vol-expansion: next move 1.5-2.0×
   larger, up-rate ~0.49) is monetised with long-gamma/straddle structures, not a directional entry.
6. **Honest validation or it's a failure** — every candidate clears `candidate_gate` net of realistic costs, beats the
   beta-matched passive, survives 10 seeds + UNSEEN-once, before any belief. No exceptions (this is the "no failures").

## 1.5 Regime taxonomy + detection recipe (the switch every archetype gates on)
"Time the market" needs a concrete, chimera-computable regime definition. Synthesizing OUR empirical 5-regime GMM
([CHIMERA_MINING_FINDINGS](CHIMERA_MINING_FINDINGS_2026_06_08.md)) with the literature's regime axes (vol / trend /
crowding / liquidity), a regime is a point in **3 orthogonal axes**, each detectable from chimera:

| Axis | States | Detect from chimera (past-only) |
|---|---|---|
| **Trend** | bull / bear / chop | `regime_label` (price vs MA), BTC `close>SMA200` (market risk-on), `norm_efficiency` + `hurst_regime` (trend vs chop) |
| **Volatility** | calm / normal / explosive | `rv_*` / `norm_yz_volatility` vs its expanding median (vol-expansion); DVOL level (BTC/ETH) |
| **Crowding** | neutral / crowded-long / crowded-short / capitulation | `fund_rate_*` extremes, `s3_oi_*` (OI) + OI-delta, `liq_*` spikes, `positioning` (taker/LSR) |
| *(macro liquidity)* | risk-on / risk-off | `stbl_*` supply expansion, `etf_*` flows (the slow conditioner) |

**The 5 empirical regimes (our GMM, BTC-cohort) mapped to the axes + what each enables:**
| Regime (share) | Trend × Vol | Crowding tell | Enables |
|---|---|---|---|
| Quiet-chop (49%) | chop × calm | neutral | A2 (VRP harvest), B (x-sectional RV), stand-aside for trend |
| Uptrend-momentum (13%) | bull × high-vol | building long | D (regime-gated trend), A1 (convexity) |
| Euphoria-blowoff (5%) | strong-bull × extreme-vol | crowded-long (funding/OI extreme) | A1 convexity + funding-contrarian FADE setup; trim trend |
| Downtrend-bounce (12%) | bear × high-vol | capitulation (liq spike, funding flip) | C (liquidation fade, majors), contrarian-long |
| Topping/distribution (20%) | above-MA but bleeding (−0.035/bar) | crowded-long unwinding | DE-RISK / exit conditioner (the trap regime) |

**Operational rule:** compute the (trend, vol, crowding) triple each bar from the features above (all past-only); route
to the archetype its row enables; in quiet-chop or topping, the correct action is often *stand aside / de-risk*, not
trade. This regime router IS the "time the market, don't pick the asset" lever — and it must itself be validated
(a regime that doesn't improve a gated metric is decoration). The existing `src/wealth_bot/regime_router/` +
`regime_classifier.py` is the seam to build it on.

## 2. The ranked archetype menu (what to build, in EV order for a retail-to-midsize systematic trader)
Each: edge thesis → regime → TF/instrument → features (asset-conditional) → failure mode → gate bar.

### A. Volatility / convexity regime engine  ★ highest SIGNAL-conviction, but A1/A2 are BLOCKED on data (see banner)
> **BUILDABILITY BANNER (verified 2026-06-09):** A1/A2 require a **Deribit options-CHAIN ingest that does NOT yet
> exist.** Chimera has **zero options-chain features** (no strikes / IV surface / greeks) — only the `dv_dvol_*` *index*
> (3 cols, **98% missing**, BTC/ETH only; ingest `src/pipeline/ingest/deribit_dvol.py`). `config/deployment_ranking.yaml`
> marks "Deribit options ingest" an open **blocker** (:1735) and the options track "still aspirational" (:1638). So A1/A2
> are a **data-pipeline prerequisite, not a first-strategy.** Only **A3 (vol-targeting overlay)** is buildable today on
> dense features. → for the FIRST build, prefer **B or A3**; pursue A1/A2 only after standing up options ingest.
- **Thesis:** vol is predictable + clusters; a vol-expansion bar predicts a ~2× larger next move with **no directional
  bias (our up-rate after expansion = 0.49 = coin flip).** This splits into THREE distinct builds — get the distinction
  right or it fails:
  - **A1 — true convexity (the honest magnitude edge):** long-gamma / straddle on **BTC/ETH options (Deribit)** entered
    on vol-expansion — profits from the bigger move regardless of direction. This is the ONLY build that directly
    monetises "bigger move, unknown direction."
  - **A2 — VRP harvest:** SELL the implied-vs-realized gap (IV>RV is the normal state) in calm regimes — short gamma,
    strictly risk-capped/defined-risk (the tail is fatal otherwise).
  - **A3 — vol-targeting OVERLAY (not standalone alpha):** scale ANY other strategy's position by 1/realized-vol. This
    improves risk-adjusted return regime-agnostically; it is a sizing overlay, not an edge.
  - **WARNING — do NOT build a directional perp "breakout on vol-expansion":** going long on an up-breakout is a
    directional bet, and our data says post-expansion direction is ~0.49 (a coin flip). A vol-expansion is a
    *magnitude* signal, not a *direction* signal. Breakout-on-perps only works if paired with an independent
    directional conditioner (regime/carry/flow), and even then must clear the gate net of cost.
- **Regime:** A1 on vol-expansion onset; A2 in calm (low DVOL, IV>RV); A3 always. **TF:** 1h-1d. **Instrument:** Deribit
  options (BTC/ETH only) for A1/A2; A3 is instrument-agnostic.
- **Features:** `volatility` family (rv_*, jumps, norm_yz_volatility, DVOL on BTC/ETH), `norm_vol_cluster`,
  `norm_vol_ratio`; regime + `derivatives` (OI/funding) as the expansion conditioner — note `norm_oi_price_divergence`
  is the *dominant* magnitude predictor only in a relative sense; its absolute handle is ~0.05 corr (a weak tilt, one
  input to the vol-expansion gate, NOT a standalone trigger).
- **Failure mode:** short-gamma tail ("pennies before a steamroller") — cap it; options spreads wide on alts (BTC/ETH
  only). **Gate bar:** positive net-of-cost UNSEEN, beats passive, 10-seed.

### B. Cross-sectional reversal / relative-value (market-neutral)  ★ high
- **Thesis:** 1-week cross-sectional reversal is the most-replicated directional anomaly (past-week winners
  underperform); dispersion is abundant; market-neutral strips the one-factor beta. 
- **Regime:** high-dispersion (altseason / idiosyncratic). **TF:** daily formation, weekly hold. **Instrument:** perp
  long-short basket (top vs bottom decile by prior-week return), beta-neutral.
- **Features:** `cross_asset` rank features (`xrel_*`, `xd_momentum_rank`, `xd_cross_return_mean`), `momentum` returns;
  liquidity filter (exclude thin/meme from the short leg manipulation risk).
- **CRITICAL buildability caveat (verified 2026-06-09):** the short-term reversal is **driven by the ILLIQUIDITY of
  small-caps — the most liquid/tradeable coins exhibit daily MOMENTUM, not reversal** ([ScienceDirect "Up or down?"](https://www.sciencedirect.com/science/article/pii/S1057521921002349),
  1,160-3,600 coin studies). So the reversal edge lives exactly where execution is hardest (thin books, manipulation,
  maker-fill uncertainty). The build must either (a) trade reversal in the small-cap tail with strict maker fills +
  manipulation guards, or (b) trade daily MOMENTUM in the liquid majors — opposite signs by liquidity tier. Pick one
  deliberately; do NOT pool them.
- **Failure mode:** factor collapse in uniform crashes (all coins move together); short-leg borrow/cost; crowding; the
  liquidity-tier sign-flip above. **Gate bar:** market-neutral Sharpe + net-of-cost positive on UNSEEN; survivorship-
  clean universe; liquidity-tier-consistent.

### C. Liquidation-cascade / reflexivity fade (conditional overlay)  ★ high as an overlay
- **Thesis:** liquidations are causal-but-bounded; the post-cascade bounce is a structural regularity on majors
  (capitulation = contrarian-long). Use as a *sizing/timing overlay*, not a standalone.
- **Regime:** high-OI + funding extreme + a liquidation spike. **TF:** minutes-hours. **Instrument:** perp (majors only
  — on memes a liquidation is terminal, NOT a bounce).
- **Features:** `liquidation` family (liq_*, capitulation/squeeze flags), `derivatives` (OI delta, funding extreme),
  `positioning` (taker exhaustion). Strictly BTC/ETH/large-cap.
- **Failure mode:** catching a falling knife (premature entry before exhaustion); contagion. **Gate bar:** event-study
  positive expectancy net-of-slippage; majors-only; no meme application.

### D. Regime-gated trend / managed-futures (the SOTA version of what we refuted)  ◆ medium
- **Thesis:** naive daily long-only MA is a verified null (our arc + the literature's "negative in bear, positive in
  bull"). The SOTA version is **vol-scaled TSMOM with a hard regime filter** (trade trend only in confirmed trending +
  risk-on regimes; flat/cash otherwise) — i.e. capital-preservation + selective participation, NOT always-on.
- **Regime:** confirmed trend (high efficiency + BTC risk-on). **TF:** 6h-1d. **Instrument:** perp/spot, maker-routed.
- **Features:** `structure` (efficiency, MA-distance, Hurst), `regime` labels, `cross_asset` BTC-regime; sized by
  inverse `volatility`.
- **Failure mode:** whipsaw in chop (the regime filter exists to kill this); our honest caveat — the DD-control "win"
  must beat buy-and-hold on Calmar in a MIXED-regime UNSEEN, not just abstain in a downtrend. **Gate bar:**
  Calmar-vs-buy-and-hold + 10-seed + mixed-regime UNSEEN.

### E. Funding / basis carry (delta-neutral)  ◇ low-now (decayed) but capital-stable
- **Thesis:** harvest persistent positive funding/basis (long spot / short perp). Structurally decayed by ETF arb
  (CME basis ~4.5%, 93% of days <5% breakeven) → sub-5% APY baseline, only event-spikes pay. 
- **Regime:** elevated funding (bull/euphoria). **TF:** 8h funding cadence. **Instrument:** spot+perp delta-neutral.
- **Features:** `derivatives` (funding_*, bs_basis_*, premium_*), OI. BTC/ETH (deep, real carry).
- **Failure mode:** negative funding in bears (you pay); CEX counterparty risk; decayed returns. **Gate bar:** net of
  funding-pay periods + fees positive; treat as a yield sleeve, not alpha.

### F. On-chain / flow conditioning (overlay only)  ◇ low in isolation
- **Thesis:** stablecoin-supply expansion + ETF inflows = the macro liquidity regime; exchange flows + whale =
  accumulation/distribution. No standalone edge; a strong *regime/exposure conditioner* on A-D.
- **Features:** `stbl_*`, `etf_*`, `wh_*` (whale, asset-conditional!), `s3_*` (OI/LSR), `soc_wiki_views` (memes only).
- **Failure mode:** noise (15-40% internal transfers), lag, reflexivity. **Gate bar:** must improve a base strategy's
  gated metric, not asserted alone.

## 3. The feature → edge map (which families power which archetype)
| Family | Primary edge it informs | Best archetype(s) |
|---|---|---|
| volatility, structure(efficiency/Hurst) | magnitude/convexity + regime | A, D |
| cross_asset (xrel/rank) | cross-sectional reversal | B |
| liquidation, derivatives(OI/funding) | reflexivity fade + crowding | C, A |
| derivatives(funding/basis/premium) | carry + crowding contrarian | E, A |
| positioning, orderflow | exhaustion + flow (depth-weighted trust) | C, B |
| stbl/etf/whale/social (flow) | macro liquidity + accumulation regime | F (overlay on all) |
| regime | the on/off/scale switch | ALL (gate) |

## 4. The dead-list (do NOT build — verified or structurally closed)
- **Naive daily long-only MA / price-only directional trend** — verified null (our oracle→decomposer→TF-sweep arc);
  the literature agrees it's negative after costs in non-trending regimes.
- **Any next-bar DIRECTION predictor at canonical TF without convexity** — AUC ~0.5 linear and nonlinear; the features
  predict how-big, not which-way.
- **Sub-daily anything with taker fills** — the cost cliff (30m gross −89.5%) kills it; maker-only.
- **Reading meme funding/whale/depth/OI like BTC** — manipulation surface, not information (dictionary §matrix).
- **Funding carry as a primary alpha** — ETF-arb decayed it to a thin yield sleeve.
- **HFT / market-making / latency arb** — co-location moat; closed to us.

## 5. The validation discipline (the "no failures" guarantee)
Every candidate, before any belief, MUST:
1. Run through `src/strat/candidate_gate.evaluate_candidate` (backtest → leak check → firewall random-entry null →
   `benchmark_excess` beta-matched passive + `bear_preserved` → `battery`).
2. **Beat the beta-matched passive** (not relabel beta as alpha) at realistic (maker where claimed) cost.
3. Pass the **battery**: block-bootstrap p05>0, maxDD<30%, jackknife-robust, **10/10 seeds positive on UNSEEN**.
4. **UNSEEN touched once** (single verdict, no peeking); MIXED-regime UNSEEN for any DD-control claim.
5. Report honestly: synthetic-vs-real caveats, asymmetric loss (false-positive > false-negative), no silent target
   reframing. A candidate that fails is REFUTED and recorded (don't re-mine).

**The path to the first strategy (buildable-now EV, NOT just signal conviction):** **A has the strongest *signal* but
A1/A2 are blocked on a non-existent options ingest** — so the FIRST build should be **B (cross-sectional reversal/RV,
dense-feature, market-neutral)** or **A3 (vol-targeting overlay)** — both buildable today on the dense feature families.
Wire its features (per the dictionary, asset-archetype-aware: trust depth-weighted, exclude memes/stablecoins from the
manipulable signals), build it **maker-routed + regime-gated** (§1.5), and run it through §5. Sequence: ship B/A3 first;
stand up the Deribit options-chain ingest in parallel to unlock A1/A2 (the highest-signal engine) next. The
understanding above is what makes the FIRST build a high-probability hit rather than a guess.
