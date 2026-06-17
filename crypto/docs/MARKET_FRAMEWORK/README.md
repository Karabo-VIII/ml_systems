# Consolidated Market Framework — Research Layer (2026-06-09)

**What this is.** The project's failure mode has been: *each instance pursues a theory, falsifies it, and leaves it
hanging — then the next instance re-mines the same dead vein.* This directory is the antidote: a **single consolidated
place** for every idea/approach/theory ever pursued (current + the 524-file `archive/restart_2026_06_04/`), with its
status, evidence, and lesson — so we can **decompose any market to its fundamental constituents** and never re-pay for
a lesson already learned. This is the *research/theory layer* of the framework (no strategy is being built yet; we are
assembling the ingredients we will tap into later).

> Built 2026-06-09 by consolidating 6 parallel archaeology passes over the whole project + archive, integrated with the
> 2026-06-08/09 mining + research-synthesis work. Every entry cites its source. RWYB where a number is ours.
>
> **This is STAGE 00 (research/decomposition) of the [Solutioning Pipeline](../SOLUTIONING_PIPELINE.md)** — the
> repeatable, market-agnostic working model (research → mining → engine → strat → bot → execution → deployment) whose
> single store is [`workspaces/`](../../workspaces/) + [`REGISTRY.md`](../../workspaces/REGISTRY.md).

## How to navigate
| Doc | What it holds |
|---|---|
| [01_DEAD_LIST.md](01_DEAD_LIST.md) | **The single most valuable artifact** — 63 REFUTED/null/exhausted theories (D01–D63) + 8 measurement artifacts that invalidated whole result-sets, each with the falsifying test, the number, and the SCOPE of refutation (hard-mechanism vs scoped-to-resolution). Read before proposing anything. |
| [02_APPROACH_LEDGER.md](02_APPROACH_LEDGER.md) | Every STRATEGY/edge theory (53) + every WORLD-MODEL version (V0–V25) ever built — claim, what was done, status, key number, lesson. The "what's been tried" map. |
| [03_METHODOLOGY.md](03_METHODOLOGY.md) | The hard-won VALIDATION discipline (35 gates/principles) — what each enforces, the bug that birthed it, and whether it's still binding. The "how we decide something is real." |
| [04_MARKET_MODEL.md](04_MARKET_MODEL.md) | What we actually KNOW about the crypto market + its data (38 established findings) + the **decomposition-dimension axes** (the fundamental constituents). The substrate model. |
| [05_OPEN_THREADS.md](05_OPEN_THREADS.md) | The 19 threads left HANGING — started, never resolved — with a "worth revisiting?" read. The user's core ask: nothing valuable forgotten. |
| [06_STRATEGY_RESEARCH.md](06_STRATEGY_RESEARCH.md) | **The strat-layer founding survey (2026-06-09)** — exhaustive SOTA crypto strategy taxonomy (CeFi + DeFi/microstructure) by viability-at-our-layer, chimera coverage+gap map, SOTA technical indicators (have/don't-have/look-ahead), **strategy-DISCOVERY methods** (GP/LLM/oracle-decomp + the PBO/CSCV gap), and the prioritized first-build shortlist. Founds STAGE 03. |

Companion theory docs (the 2026-06-09 research synthesis, kept alongside): [CRYPTO_MARKET_UNDERSTANDING.md](../CRYPTO_MARKET_UNDERSTANDING.md),
[CHIMERA_FEATURE_DICTIONARY.md](../CHIMERA_FEATURE_DICTIONARY.md), [STRATEGY_PLAYBOOK.md](../STRATEGY_PLAYBOOK.md),
[CHIMERA_MINING_FINDINGS_2026_06_08.md](../CHIMERA_MINING_FINDINGS_2026_06_08.md).

## The market, decomposed into its fundamental constituents (the framework's spine)
Any strategy = a point in this constituent space. This is the decomposition lattice the project converged on.
**Each axis is now catalogued canonically** (not just listed here): the SIGNAL axis (#4) →
[CANONICAL_FACTOR_REGISTRY](../CANONICAL_FACTOR_REGISTRY_2026_06_11.md) (A=TI ~110, B=chimera ~218, C=frontier 141);
the OTHER 7 axes (chart-type, cadence, instrument, regime, method, approach, **entry/exit policy**) →
[CANONICAL_STRATEGY_DIMENSION_REGISTRY](../CANONICAL_STRATEGY_DIMENSION_REGISTRY_2026_06_11.md). These registries
are *enforced* (not just listed) by the **discovery preflight** ([`src/framework/discovery_contract.py`](../../src/framework/discovery_contract.py)):
a stage-03 gate that flags any silently-omitted axis (a missing timeframe, an untested exit *family*) and
canonicalizes near-duplicate configs (MA(28,29) ≈ MA(27,30)) — so discovery is dimensionally complete + search-space
canonical *by construction*, and the intelligence is freed for edge-finding.

| # | Constituent axis | Sub-dimensions | Coverage |
|---|---|---|---|
| 1 | **Chart / bar-type** | time, dollar (coarse ~6676/asset & fine ~75s), dollar-imbalance (dib), runs-tick, runs-volume, range, adaptive-vol, Heikin-Ashi, Renko | time-bars explored; info-driven bars mostly raw-only (chimeras sparse/empty) |
| 2 | **Resolution / cadence** | 1d, 4h, 1h, 30m, 15m, dollar-coarse, fine-dollar/event-clock, tick | 1d/4h explored; ≤1h cost-walled; sub-bar/tick unexplored |
| 3 | **Instrument** | per-asset over u10/u50/u100; perp vs spot; options (Deribit BTC/ETH) | most of u100 unexplored; options ingest absent |
| 4 | **Signal / indicator** | price-TI (MA/EMA/RSI/MACD/Boll, 127+); whale/flow; OI/funding/basis; liquidations; ETF/stablecoin macro; LOB; cross-asset/TE; WM signal | standalone price-TI dead; most non-price gates unexplored |
| 5 | **Regime** | trend (SMA-200), volatility (calm/expansion/euphoria), crowding (funding/OI/capitulation), macro-liquidity (stbl/ETF); 5-state GMM (empirical) | SMA-200 gate explored; vol/crowding/multi-axis largely open |
| 6 | **Method** | static rules, dynamic/regime-adaptive, ML (only as meta-labeler), self-improving rotation, world-model | static explored; ML-as-generator dead; ML-as-meta-labeler open |
| 7 | **Approach / portfolio** | per-asset specialist+combine, cross-sectional/breadth-pooled, regime-gated portfolio, setup-chaser book, oracle-decomposition | regime-gated explored (the floor); x-sectional + oracle open |
| 8 | **Entry / exit policy** | trailing stop, time-stop, fixed target, signal-flip, triple-barrier, managed-RSI exit | trailing adds ~2pp/trade; exit < entry-selectivity in importance |
| + | **Cross-cutting lenses** | actor (institutional/retail/DeFi/MM), sector (L1/Meme/DeFi/AI…), hold-cadence, capital-velocity | archetype matrix in the feature dictionary; sector conditioning open |

## The 10 hard-won lessons (the cross-cutting truths — read these first)
1. **Direction is unpredictable at every canonical TF — linearly AND nonlinearly.** GBM AUC 0.508–0.531 ≈ logistic; Hurst≈0.5; IC≈0 six ways. (D55, D44, D17) → don't build a next-bar direction predictor.
2. **Volatility/magnitude IS predictable and strengthens intraday** (AC1|ret| 0.18→0.33; vol-expansion → 1.5–2.0× bigger move at coin-flip direction). The honest edge channel is convexity/vol, not direction.
3. **Crypto is a one-factor (BTC) market at every resolution** (corr ~0.55, beta ~1.2, no lead-lag ≥15m) → time the market, don't pick the asset.
4. **Cost is the binding constraint; maker is the #1 lever — but maker p_fill is 0.21–0.40, not 0.80.** Sub-daily at taker is a cost cliff (30m gross −89.5%). (D43, D50, D60, A7)
5. **"Buy-the-extreme" is an ANTI-edge** (liq-spike, vol-climax, deep-oversold all fire mid-cascade; random-entry BEATS them). Wait for the *reversal confirmation*, never the extreme. (D48–D52)
6. **Concentration is THE failure mode.** Almost every "edge" was 1–3 trades carrying the whole compound; jackknife K=2/K=3 and n_eff killed them. Robustness = the BOOK, not one config. (PEPE campaign, D29)
7. **ML is NOT an alpha source — only defensible as a META-LABELER on an already-proven exo-gate** (AUC ≈ 0.50 as a generator). (D16, D17, WM register)
8. **At daily/4h/dollar + LO+spot+lev=1, the honest floor is beta+yield (~13–26% CAGR), NOT active timing alpha** — confirmed 6 independent ways. This is the *constraint's* limit, not a global ceiling. (Market model #33)
9. **Look-ahead is the silent inflator** (MtM double-count 5×, K-on-future-returns 14×, same-bar-close fill +50–80pp, tautology cells = 100% hit, sub-daily `load_panel` collapse). Every headline must pass the 4-bounds + firewall + UNSEEN-once discipline. (Methodology; A1–A8)
10. **The IC/per-bar-predictability paradigm is ARCHIVED.** The unit of trading is the SETUP across a MULTI-CANDLE MOVE; IC survives only as a within-WM diagnostic (>0.015), never an objective. (D13)

## The 3-era arc (how the project got here)
1. **2026-04 (ensemble build):** a rule+ML ensemble (xsec ranker + frontier flow sleeves + meta-labeler) hit 80–100%+ CAGR on the 2025-bull aligned window — then was systematically deflated by bug-fixes (MtM 5×, YAML-vs-v3, cost-model). The WM layer built a 25-version cohort chasing the IC "Headline" tier (never reached; dollar-bar ceiling ~IC 0.07).
2. **2026-05 (exhaustion):** a massive per-asset/per-indicator search (9,900 MA pairs, TA-SML, mover-oracle, the PEPE gold-standard dossier across 14 dimensions) — every avenue produced promising in-sample numbers that collapsed under honest 4-window OOS + shuffle-null + drop-top-jackknife + DSR. Convergence: price-TA, per-asset flow, order-book, WM-signal, and cross-sectional momentum all ≈ 0 persistent OOS edge.
3. **2026-06 (reset + foundation):** the IC-primary objective was declared wrong; the unit became the SETUP across a MOVE; the apparatus was rebuilt (`src/strat/`); the opportunity surface was re-derived (≈7.6 up-movers/day ≥5%); and the 2026-06-08/09 mining established the three governing facts (direction unpredictable, vol predictable, one-factor). **No active-trading system has been built/validated post-reset** — the pre-reset "~26% CAGR backbone" and "4h RSI-bounce satellite" are DESIGNS carried over as hypotheses (their numbers are pre-reset, apparatus-inflated per the A1–A8 artifacts, archived 2026-06-04), to be re-established under the current apparatus before they can be cited as real.

## Current state (what's actually live vs aspirational)
- **Built + operational — THE WORKING MODEL (what we go back to):** the `src/strat/` gate apparatus (candidate_gate → firewall → battery → benchmark → two-sided positive control) + CDAP; the `src/oracle/` + `src/mining/` decomposition/research engines; the `src/framework/` solutioning pipeline + unified workspace store. All RWYB-verified. This is the foundation; no strategy depends on it yet.
- **DESIGNED but NOT built or validated** (no implementation in `src/strat/`; pipeline stage `03_strat` gate = FALSE): the beta+yield regime-timed backbone, the 4h RSI breadth-bounce satellite, the bear-short sleeve. The ~13–26% CAGR is the *beta ceiling* (a market fact under LO+spot+lev=1), **not** a validated system result; the prior ~26% figure was apparatus-inflated (archived 2026-06-04) and must be re-established under the current apparatus before being cited.
- **Built-but-deprioritized:** the WM layer (V1.1, IC 0.067/ShIC 0.033, used diagnostically only); the research-synthesis docs.
- **The blocking decision:** the **A/B/C fork** — accept the beta+yield ceiling (Fork A) vs invest in sub-bar/tick + info-driven-bar data engineering to chase the unproven frontier (Fork B). Every campaign converges on "you must pick this." See [05_OPEN_THREADS.md](05_OPEN_THREADS.md) thread 5.

## How to use this framework
Before proposing ANY strategy: (1) check [01_DEAD_LIST.md](01_DEAD_LIST.md) — is it (or a near-neighbor) already refuted, and is the refutation HARD or SCOPED? (2) locate it in the constituent lattice above — which axes is it new on? (3) check [03_METHODOLOGY.md](03_METHODOLOGY.md) — which gates it must pass. (4) if it survives, it's a candidate for the (future) build phase, validated through `candidate_gate`. The honest edge frontier the whole corpus points to: **vol/magnitude + convexity, cross-sectional relative-value (tier-aware), liquidation-fade overlay on majors — all maker-routed, regime-gated, archetype-aware.**
