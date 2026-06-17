# 02 — Approach Ledger (every strategy + every world-model tried)

The "what's been tried" map. **Part A** = strategy/edge theories; **Part B** = world-model versions. Status legend:
SHIPPED (validated/operational) · QUALIFIED (real but caveated/satellite) · INCONCLUSIVE · HANGING · FALSIFIED (see
[01_DEAD_LIST.md](01_DEAD_LIST.md) for the kill detail) · DECAYED · ARCHIVED. Numbers pre-2026-06-04 are on the
pre-reset apparatus (often inflated by the A1–A8 artifacts) — treat as historical.

## Part A — Strategy / edge theories

| # | Approach | Status | Key number | Lesson | Source |
|---|---|---|---|---|---|
| 1 | Regime-timed quality-beta + yield (DEF12 backbone) | **DESIGNED / pre-reset** (NOT re-validated; no `src/strat/` impl) | +25.9% CAGR / DD −28% (pre-reset, apparatus-inflated) | simplicity wins; every "enhancement" refuted; cash-gate = structural bear protection; **must be re-built+validated post-reset before "deployable"** | DEPLOYABLE_STRAT_2026_05_31 |
| 2 | Pooled 4h RSI-oversold bounce (breadth MR) | **DESIGNED / pre-reset** (satellite; NOT re-validated) | OOS +6.4%; beats null +14pp; UNSEEN ~flat (pre-reset) | edge is SELECTIVITY (~2/day = the confirmed turn); no 2nd uncorrelated edge survived (17 tested) | DEPLOYABLE_BREADTH_SYSTEM |
| 3 | Bear trend-following short | **QUALIFIED** (bear participation, not wealth-add) | UNSEEN +0.40%/trade; book −5.1pp CAGR, +5.6pp bear-participation | a hedge at a wealth+DD cost; first bear-participation edge | all_regime_bear_short |
| 4 | Breadth-bounce trailing-exit (κ lever) | **DEPLOYED IMPROVEMENT** | trail_0.05 OOS +3.61% vs managed +1.58% (+2pp) | trailing exit ~doubles per-trade in a regime-routed book; standalone breaches DD | capture_lever_breadth_exit |
| 5 | Cross-sectional XGB/CatBoost ranker | SHIPPED (2025-era) → **FALSIFIED** on 2024-26 bear | K=3 +98%/13mo BUT 2026 IC≈0 | gains were 2025-bull beta; prediction only in the tail; ρ≈0 globally | xsec_ranker_breakthrough |
| 6 | Meta-labeler gate (CatBoost) | **SHIPPED** (as a gate) | prod_meta_combined CAGR 48.9% Sh 2.69 | without the gate, 10/16 strategies DEAD; the meta IS the edge on rule-swing | DEAD_STRATEGIES §A |
| 7 | PEPE × MA/EMA × whale (gold-standard dossier) | **INCONCLUSIVE / refuted-under-stress** | best EMA30+whale UNSEEN +60%; but n_eff<15, DSRp>0.91 | concentration is the invariant failure; mechanism real, harvestability fails rigor | PEPE_MA_EMA_CLOSURE |
| 8 | PEPE dollar slow-SMA × whale (lone survivor) | **CANDIDATE w/ caveats** (ship pending) | UNSEEN +71%, jk3 +26.7, mech-falsifier PASS; n_eff=8, n=11 | the right clock (coarse dollar) is necessary; whale-alone fails OOS; rare even then | pepe_dollar_slowsma_whale |
| 9 | PEPE dollar MACD × LOB-rising (setup-chaser) | **CANDIDATE** (~2/14 assets) | UNSEEN +2.43%/trade; plateau 6/6; n=19 | setup-chaser gate not a loosening; 0 majors survive | STRAT_LAYER_CONSOLIDATION §10 |
| 10 | Exhaustive single-asset price-TA (MA/EMA/RSI/ADX) | **FALSIFIED** (D14) | u100 0 survivors; ETH 0/1152 | price-history features carry ~0 persistent forward discrimination (info-theoretic) | OPENBOOK_ETH_SEARCH |
| 11 | Cross-sectional LO momentum | **FALSIFIED** (D40) | 0/72; TRAIN +1000s% bull-beta, OOS neg | CS LO momentum = dead beta; vol-adj leaders partial | crypto_strategy_basis |
| 12 | Liquidation-reversion LO / market-neutral | **FALSIFIED LO** (D48) / candidate MN | LO 0/8; null POSITIVE; MN survives 3/4 | reversion premium real but market-neutral; LO buys into cascade | STRAT_LAYER §E |
| 13 | Funding carry | **DECAYED** (D18) | full +13%/yr → UNSEEN +3.1% gross, −5.4% net | arbed away post-2024; worse than staking yield | STRAT_LAYER §I |
| 14 | Post-pop risk-off reversion | **FALSIFIED** (D19) | shuffle-null p=0.176; bear confound | symmetry test = the diagnostic | STRAT_LAYER §J |
| 15 | DOGE smart-vs-retail conditioner | **FALSIFIED harvestable** (D36) | discriminates pctile 1.0 but jk3 0/120 | discrimination ≠ harvestability | event_study_discriminator |
| 16 | WM (V1.1) as tradable signal | **FALSIFIED** (D15) | h64 0/5; h1 spread ~8× < cost | sub-bp IC doesn't aggregate to tradable under cost | event_study_discriminator |
| 17 | DIB flow-imbalance sleeve | SHIPPED (2026-04) | DIB Sh 3.62, IC +0.19; vol_runs 92% redundant | DIB is time-series not x-sectional; flow persistence ≈0 in 2026 | frontier_final_state |
| 18 | Stablecoin mint-shock sleeve | SHIPPED (2026-04) | aligned Sh 1.50 | true-flow (new capital) works in bull; bear-death risk | frontier_final_state |
| 19 | ETF flow-shock sleeve | SHIPPED (2026-04) | aligned Sh 1.10 | structural (BTC+ETH only); regime-dependent | frontier_final_state |
| 20 | Post-listing H1 momentum | SHIPPED (2026-04) | OOS +8.64%/event n=85; live +1.50% (5.76× attenuation) | new info-channel, orthogonal; needs real-time listing poller | listing_h1_momentum_alpha |
| 21 | TA-SML (TA × supervised ML) | **HANGING** | 4h VOLATILE OOS IC +0.127; 9-day replay +16% | chop fails at cost; 9-day validation = the fragility | ta_sml_build |
| 22 | Setup-driven day-trader (9 setups) | **HANGING** | April UNSEEN −0.07%/d; ceiling 0.3–0.6%/d | priors over-stated 60–90×; Phase-2 stack unvalidated | day_trader_phase8 |
| 23 | Mover oracle (LGBM magnitude+direction) | **SHIPPED as overlay** | AUC 0.672; full-UNSEEN 0.04%/d (misses 1%/d) | signal exists but 1%/d needs leverage/sub-day/new-channel | mover_oracle |
| 24 | Engine of specialists + production blends | SHIPPED (2026-04, pre-reset) | v7_frontier CAGR 82% Sh 4.45 (PRE-FIX) | all numbers pre-reset/inflated; apparatus later rebuilt | MEMORY_HISTORY |
| 25 | Adaptive learning bot (6 ML architectures) | **FALSIFIED** (D17) | IC −0.008@1d; real bot ~26%/yr | 1%/day = IC 0.60 needed; we measure ~0 = efficiency | adaptive_bot_prediction_dead |
| 26 | HFT / sub-bar (tick) path | **MAPPED, not pursued** | +7727% at 0 cost; −60% at lag-1; breakeven ~5bps | edge is sub-bar microstructure; needs co-location/maker/L2 | HFT_PATH_SCOPING |
| 27 | MA/EMA 9,900-permutation Layer-3 | **INCONCLUSIVE / filter-tier** | best universal SMA Sh 0.07; chop kills | MA cross is filter-tier, needs regime routing | ma_ema_permutation_layer3 |
| 28 | Per-asset MA permutation (u100×4 cadences) | **INCONCLUSIVE** | TRX 100% hit (suspicious); 0 survivors under honest gate | Phase 3 train+val only; OOS required | ma_ema_permutation_layer3 |
| 29 | Regime-router (STRICT_LO_SETUP60) | **REFUTED / apparatus-inflated** (D05) | +20.25% OOS later found 3–5× inflated | helps bear→bull transitions, not sustained bull | DEAD_STRATEGIES #2 |
| 30 | Breadth-based regime gate | **REFUTED** | redundant w/ per-asset SMA50 | per-asset SMA50 already encodes breadth | STRAT_LAYER dead-list |
| 31 | Composite accumulation gate (whale+OI+funding) | **REFUTED** | doesn't beat trend-alone; OI hurts | weak-signal composites don't pool | STRAT_LAYER §E |
| 32 | Order-book flow persistence | **FALSIFIED** | TV→OOS corr +0.011/−0.052 | OB snapshots carry no persistent forward signal | STRAT_LAYER §G |
| 33 | Continuous TSMOM V3 sizing | **REFUTED overstated** (D21) | battery fails jk 1/3 seeds; DD −41.8% | binary 12mo screen modest-real; continuous not the answer | STRAT_LAYER §H |
| 45 | Vol-adjusted persistent leaders | **QUALIFIED** | 3/4 windows +; vanishes UNSEEN bear | optional tilt; raw leadership = vol persistence | crypto_strategy_basis |
| 50 | Beta-hedge (long bounce / short BTC) | **INCONCLUSIVE / OOS-only** | OOS +2.96%/trade PF7.5 (~60% real alpha) | isolates real alpha but bear sample too thin | per_trade_multi_tf |
| 53 | Trend-scale (proportional regime exposure) | **SHIPPED (DD-efficiency)** | +0.6–2.9pp CAGR, DD −8–9pp lower | risk-efficiency gain not free alpha (early "+10pp" was apples-oranges) | DEPLOYABLE_BREADTH |

*(Also tried + FALSIFIED — see dead-list: 8 alpha formulas F1–F8 (D41-adjacent), funding-divergence/top-trader, wiki
pageviews, calendar Q4, kelly vol-break (D06), te_leadlag (D08), perp L/S (D04), MA×RSI (D20), mn_momentum retracted,
chop-MR (D49-adjacent), symmetric L/S short-fade (D56), flow-surge (D59). Full strategy detail in
`archive/.../docs/STRAT_LAYER_CONSOLIDATION_2026_06_02.md` + `WEALTH_BOT_FAILURE_CATALOG.md`.)*

### Strategy arc (5 lines)
2026-04 explosive ensemble (80–100% CAGR on 2025-bull) → deflated by bug-fixes. 2026-05 massive per-asset/indicator
exhaustion (9,900 MA pairs, TA-SML, mover-oracle, PEPE dossier) → all collapse under honest OOS+null+jackknife+DSR.
2026-05-late convergence: price-TA, per-asset flow, order-book, WM-signal, x-sectional momentum all ≈0 persistent OOS;
only durable money = disciplined beta+yield + the moderate-oversold breadth MR (~26%/yr). 2026-06 all-regimes: a first
bear-participation short edge (hedge not wealth-add); HFT path mapped (real sub-bar edge, needs co-location). At the
reset: backbone + breadth-bounce validated, a setup-chaser book partially built (1 robust PEPE survivor), IC-primary
declared wrong.

## Part B — World-model versions (V0–V25)

| Version | Architecture / idea | Status | Key result | Lesson |
|---|---|---|---|---|
| **Anti-fragile stack** | ShIC>IC×0.5, no-RevIN, no-focal, raw targets, bins[-1,1], DIRECT_RETURN_WEIGHT=3.0, ACTIVE_HORIZONS[1,4,16,64] | **SHIPPED — permanent invariant** | the project's most durable contribution; survived every reset | the natural "improvements" (RevIN, focal, voladj, wide bins, LR-on-decline) are ALL anti-anti-fragile for return heads |
| V0 | Linear/Ridge/GBT/MLP baseline | SHIPPED (benchmark) | IC≈0.018 floor | the L3 data-vs-architecture fault discriminator |
| V1.0 | Transformer + RSSM (24×24 categorical bottleneck) | **SHIPPED** (iron-clad anchor) | IC 0.066 / ShIC 0.032 (Trader tier) | the canonical anchor; beat both IC AND ShIC to earn a place |
| **V1.1** | V1.0 + XD cross-dim dropout | **SHIPPED — project record** | IC 0.073 / ShIC 0.032 | XD is the cheapest anti-mem upgrade; the live WM |
| V1.4 | FeatureAttention / iTransformer | SHIPPED | IC 0.068 (best raw IC) | cross-feature attn raises IC not ShIC |
| V1.6 | All anti-mem (KL+Gumbel+ATME+dream) | SHIPPED (just clears) | IC 0.062 / ShIC 0.033; dream never in loss | stacking without ablation wastes compute; CC-H7 dream-rollout never pulled |
| V2/V5/V7 | JEPA / TBPTT / discriminator variants | ARCHIVED 2026-04-29 | — | superseded; discriminator instability is the failure mode |
| V3 | WaveNet + GRU + RSSM | INCONCLUSIVE | clean-variant IC 0.28/ShIC 0.0002 = Pattern-I memorization (RSSM stripped) | multi-scale promising; needs clean baseline w/ bottleneck |
| V4 | Mamba-3 SSM + RSSM | INCONCLUSIVE (ShIC declining) | IC 0.048/ShIC 0.016 @ep30 declining | h_seq magnitude explosion found only at B=32 stress; A_log uniform init mandatory |
| V6 | Causal JEPA + adversarial | INCONCLUSIVE (ShIC-decline) | IC 0.024/ShIC 0.020 declined; InfoNCE saturated | JEPA sound; adversarial stability (spectral-norm/R1/TTUR) is the gap |
| V8 | Neural ODE (RK4) | DEFER (never trained) | 256× compute; fp16 overflow | RK4 = worst compute profile; Mamba dominates |
| V9 | GRU + 3-expert MoE (ground-truth regime gate) | **KILLED** | IC 0.007/ShIC 0.007 | ground-truth regime label = look-ahead LEAK; V23 is the predicted-regime redo |
| V10 | Meta-ensemble router | DEFERRED (L4 dep) | never built; pairwise ρ never measured | the highest-EV 0.5-day work (CC1) — never run |
| V11–V14 | WaveNet-MoE / cross-asset / TFT / diffusion | INCONCLUSIVE / blocked | V12 cross-asset path is DEAD CODE (runs single-asset); V11–14 gate-fail ShIC 0.000 | V12 = highest single-model upside, blocked on MultiAssetDataset harness (never built) |
| V15–V19 | PatchTST/DreamerV3/TD-MPC2/Chronos/f121 stubs | STUB / KILLED / DEFERRED | Chronos zero-shot fails IC>0.015 | foundation-model zero-shot below threshold; LoRA path unbuilt |
| **V20** | Tick-level Performer/Hyena + LOB micro-features | **PROPOSED / never built** | expected IC 0.10–0.15 at h=5–30s | the honest conclusion: IC>0.10 needs tick REPRESENTATION, not architecture swaps |
| V21 | Multi-asset MASTER cross-asset spatio-temporal | SCAFFOLD only | projected IC 0.08–0.12 | blocked on same MultiAssetDataset gap as V12 |
| V22 / V25 | iTransformer / frontier (V22+regime+spectral) | INCONCLUSIVE — NEEDS RETRAIN | IC +0.21 / ShIC 0.000 = PURE MEMORIZATION (recon=zeros stub) | a model with IC 0.21/ShIC 0.000 is WORSE than 0.06/0.03; the ShIC=0 was misread as "Champion" |
| V23 / V24 | xLSTM / TimesNet | INCONCLUSIVE (need plan) | no IC numbers; FiLM wired | real SOTA archs wired before establishing they pass the ShIC gate |
| CC-H1–H7 + SAM/PCGrad/MTP | headline upgrade modules (multi-res, seq-len, cross-asset attn, anti-mem tightening, distributional, regime-cond, dream-rollout) | SHIPPED as modules, mostly NOT WIRED | CC-H4 (anti-mem tighten) was the clearest ShIC path, never run; CC-H7 dream-in-loss never executed | the cheapest Headline lifts were modules nobody wired into a version |
| Pattern L | HIGH_MAG conformal gating | **SHIPPED — empirically validated** | IC 0.014→0.042 gated (3.09× BTC); 4/4 assets | the WM signal is sparse — present only when |prediction| is large; LOW_WIDTH gating fails (0.58×) |

### WM arc (5 lines)
A Transformer+RSSM family (V1.x) converged at IC≈0.067/ShIC≈0.033 (Trader tier) and never reached the Headline tier
(IC>0.10) — diagnosed as an information-theoretic dollar-bar ceiling (~0.07–0.08), a REPRESENTATION problem not an
architecture one. The 25-version cohort proliferated (Mamba, JEPA, diffusion, xLSTM, TimesNet) but the alternatives
either memorized (ShIC→0) or were never trained; the cheapest real lifts (CC-H4 anti-mem tighten, CC-H7 dream-in-loss,
V10 ensemble ρ, V12 cross-asset harness) were never executed. The **anti-fragile philosophy** (ShIC gate, no-RevIN,
no-focal, raw targets) is the durable contribution. At the 2026-06-04 reset the entire IC-as-primary paradigm was
ARCHIVED — the unit of trading became the SETUP across a MOVE, and the WM layer has been structurally intact but
diagnostic-only since. The genuine path to IC>0.10 (V20 tick-level) was the honest conclusion but needs sub-second data
that isn't ingested.
