# Research: profitable systematic-engine methodologies for our case (2026-06-14)

Deep-research harness (5 angles, 105 agents, ~50 sources fetched, 3-vote adversarial verification).
Scope: methods that ADD net-of-cost, capacity-aware, implementable edge on top of our vol-target/regime/carry
book, given internal-price directional alpha is empirically exhausted. All performance numbers below are
[REPORTED] from the cited external sources (gross/in-sample unless stated); our job is to re-verify net-of-cost
on our own held-out data before believing any of them.

## The headline (what the evidence says for OUR exact configuration)
1. **Our architecture is LITERATURE-VALIDATED.** A trend/regime DEFENSIVE overlay + a CARRY satellite + multi-sleeve
   construction is exactly what the peer-reviewed evidence endorses. We built the right thing.
2. **Our HONEST CAVEATS are CORROBORATED.** Gross Sharpes ~halve net of costs; the overlay buys crisis/drawdown
   protection, NOT return; gross/in-sample edges vanish under real implementability. (This validates the
   ENGINE_METHODOLOGY disciplines wholesale.)
3. **One real DECAY RISK in our satellite is now CONFIRMED** (re-verify on recent data).
4. **One genuine RETURN-ADDITIVE frontier with NET-OF-COST OOS evidence: EXTERNAL ORDER-FLOW DATA.**

## The 7 verified findings (claim | confidence | sources)
- **F1 [high, 3-0] Trend/managed-futures overlay = "crisis alpha" / "dual mandate".** Convex smile payoff (best in
  extreme up AND down), ~zero equity correlation, protective in slow multi-month bears (goes short as markets fall);
  LAGS in fast V-crashes. => validates our regime overlay's role. Hurst/Ooi/Pedersen "Demystifying Managed Futures"
  (JOIM); Asness "Raisons d'etre" (AQR).
- **F2 [high, 3-0] Gross trend Sharpes HALVE net of costs.** Diversified 12m TSMOM gross Sharpe [REPORTED 1.6-1.8] ->
  ~1.0 net after 2-and-20 (~6%/yr) + [REPORTED 1-4%/yr] transaction costs; costs HIGHER for alt assets (crypto's
  regime). => any ported Sharpe needs a real net haircut. AQR Trends-Everywhere (JOM); Demystifying (Pedersen).
- **F3 [high, 3-0] CARRY is a distinct, orthogonal, model-free return source** (works XS + TS across asset classes,
  NOT subsumed by value/momentum/TSMOM). Combining uncorrelated carry sleeves lifts Sharpe [REPORTED ~0.7 -> ~1.1]. =>
  theoretical grounding for our funding-carry satellite + the multi-sleeve diversification gain. Koijen/Moskowitz/
  Pedersen/Vrugt "Carry" (JFE 2018).
- **F4 [high, 3-0] Crypto funding/basis carry is a real, grounded crypto-native edge** (short-perp/long-spot harvests
  positive funding; cash-and-carry basis [REPORTED ~7%/yr 2019-24, sometimes >40%/yr], NOT explained by rates) -- BUT
  every headline Sharpe is GROSS/in-sample/frictionless. CEPR/VoxEU crypto-carry; ScienceDirect.
- **F5 [high, 3-0] Funding-carry alpha is DECAYING HARD + capacity-limited (DEPLOYMENT-CRITICAL).** Perp funding-carry
  Sharpe [REPORTED 6.45 full-sample -> 4.06 in 2024 -> NEGATIVE in 2025]. The Jan-2024 spot-BTC-ETF causally
  compressed crypto carry ~3pp across exchanges (+5pp on CME) = [REPORTED 36% / 97%] declines of mean carry -> it's
  MARGINING FRICTIONS, not fundamentals, so it compresses as institutional capital enters. arXiv 2510.14435; CEPR.
- **F6 [high, 2-1] EXTERNAL-DATA alpha is the ONE angle with NET-of-cost OOS evidence beating price-only.** "World
  order flow" (FX order flows across 11 currencies) has strong OOS predictive power for DAILY crypto returns,
  DOMINATES fundamentals, strongest with non-linear ML; realistic LONG-ONLY [REPORTED alpha ~0.87%/day, t=3.18,
  Sharpe ~2.0, break-even cost 1.07%/day]. ScienceDirect S1386418126000029.  <!-- REPORTED (external source) -->
- **F7 [high, 3-0] The dominant PITFALL: gross/in-sample edges VANISH under implementability.** Cross-exchange BTC
  "arb" [REPORTED 68.5%/wk -> 3.8%/wk] on implementable venues; best crypto TS-momentum nets [REPORTED Sharpe ~1.51]
  (max from ANY momentum ~1.5) vs >2.0 routine in equities; fat tails + intra-hold liquidations strip significance.
  arXiv 2510.14435; ResearchGate momentum-under-realistic-assumptions.

## How this maps to OUR engine (the actionable synthesis)
- **KEEP the regime overlay** (F1) -- it is the literature's crisis-alpha dual-mandate; our finding "risk not return"
  matches F1/F2 exactly. No change.
- **RE-VERIFY the funding-dispersion satellite NET-of-cost on 2024-2025 (F5).** Our memory already shows the decay
  signature (SEL +36 / OOS +8 / recent +3); F5 gives the MECHANISM (post-ETF margin-friction compression) and the
  warning (negative in 2025). NUANCE: ours is cross-sectional DISPERSION (spread across names), which may persist
  longer than the average-basis LEVEL that F5 measures -- but this MUST be checked on our venues/costs (open Q1).
- **THE RETURN FRONTIER = EXTERNAL ORDER-FLOW DATA (F6).** This is the one method with net-of-cost OOS evidence that
  beats price-only -- exactly the "external data" direction every internal dead-end pointed to. Source world/FX
  order-flow (or a crypto proxy: aggregated exchange CVD / spot-perp flow / stablecoin flows) and test the ~2.0
  long-only signal on our daily universe under our real costs (open Q2).
- **The pitfall discipline (F7) IS our methodology** -- net-of-cost, OOS, capacity-aware, two-sided controls. The
  research independently re-derives the ENGINE_METHODOLOGY disciplines.

## The two highest-EV next experiments (the open questions)
1. **Satellite decay re-verification:** net-of-cost, capacity-aware, multi-year-OOS Sharpe of the funding-dispersion
   satellite on OUR tradable venues (Binance/Bybit) under our real fee/funding/slippage -- does dispersion survive
   the 2024-25 ETF-driven carry compression, or is it decaying too?
2. **External order-flow build:** source world/FX order flow (or a crypto flow proxy) and test whether the [REPORTED
   ~2.0] long-only net Sharpe replicates OOS on our daily crypto universe and survives our costs. This is the one
   evidence-backed path to RETURN (not just risk-reshaping).

Refuted/flagged: headline carry Sharpes (6-23) are gross/in-sample/frictionless; naive crypto momentum is net-weak
(~1.5 max); cross-exchange arb is mostly non-implementable. Treat ALL external numbers as REPORTED until re-verified
on our held-out data + cost model.
