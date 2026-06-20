# META-FOLD LEDGER — self-folding strategy-discovery loop (2026-06-19)

User directive: "Fold cycles upon yourself. After a run, run as meta with initial run + conclusions as input,
over and over." Each CYCLE = a mover_lab exploration workflow whose design is informed by the CUMULATIVE
conclusions below. Each cycle must REFINE, not repeat: deepen winners, kill losers, test the lessons,
forward-validate (lock 2020 / prove 2021-22), push the greedy frontier, widen to u50 when earned, and
adversarially verify any too-good result. CONVERGENCE = 2 consecutive cycles with no new edge -> final synthesis + STOP.
Lab: src/strat/mover_lab.py. Constraints: long-only spot, no leverage, no shorts, taker, RWYB, internal data only.

---

## CYCLE 1 — broad landscape (6 ways to win) — RUNNING (workflow wdanrsq3d)
Focus: map distinct styles — momentum×flexible-holding, daily-movers, mean-reversion, vol-breakout, exit-study,
greedy-concentration — each on mover_lab, measured by 3d-checkpoint green-rate + compound + maxDD + where-it-wins.
Status: DONE (workflow wdanrsq3d, refereed w/ shuffle-null + jackknife + independent re-derivation).
WINNERS (real): `mom14 K5 r14` +1274% full / 2022 -55% / maxDD -72% / **shuffle p=0.010** (genuine bull skill);
  `breakout K3 5d gated` +1516% / green21 61%. LOSERS/luck: top-1, mom30-K1 (+1850%), ret1-top1 (+6087%),
  vol_breakout +973% (jackknife ±1500pp on one name / non-replicating). RISK-SLEEVE: `MR rsi30+gate` 2022 -18.7%
  maxDD -36% (cash-avoidance, not alpha). LESSONS: flexible-exit(ATR-trail) >> flush; breadth > concentration;
  daily-movers amplify bull but lose risk-adjusted; **BEAR IS DEAD-BETA for ALL trend/mover styles (2022 shuffle
  p=1.000, zero selection skill) — only lever is EXPOSURE (the gate)**. CAVEAT: all in-sample on the ONE 2021
  supercycle, best-of-N=194 -> the decisive test is a FRESH 2023+ path.

## CYCLE 2 — FORWARD-VALIDATE the survivors on unseen 2023-2025 + attack the open levers — RUNNING (workflow wok9amqx9)
Focus (folds on C1): (1) OOS 2023/2024/2025 of the 2 survivors + beta + MR sleeve — does the bull skill PERSIST on
fresh paths (the make-or-break)? (2) BEAR LEVER — harden the gate (SMA100/50, BTC-trend, breadth, vol-target) to cut
the -55%/-64% bear, the one open lever. (3) U50 widen (deeper bench for momentum). (4) EXIT optimization (ATR-trail,
the confirmed lever) forward-validated on 2023-25. (5) Verify lane (jackknife + shuffle on the OOS years; kill single-
path luck). Status: launching.

Status: DONE (wok9amqx9, refereed via independent re-derivation cycle2_referee.py).
- OOS 2023-25: mom14-K5 selection skill REAL at FAST rebal (r3 OOS p=0.010-0.028; r1 p=0.005) but DIES at slow r14
  (p=0.154 — C1 froze the wrong config) AND does NOT survive multiple-comparisons correction across the rebal DoF
  (Holm clears only r1). Per-year 2024 clean (p=0.015), 2023 dead, 2025 wash. **breakout-K3 DIED OOS (p=0.169).**
- BEAR LEVER (strongest finding): **BTC-market SMA200 gate** (book->cash when BTC<itsSMA200) = 2022 **0.0%**
  (BTC sub-SMA200 all 365d), maxDD -76->-56%, full ~free (+103pp). BUT it is a TREND filter: HURTS 2025 chop
  (-26% vs -20%); per-asset gate DEAD (-64.6% at 0.47 expo).
- u50 DEAD (p=0.000 was a bull-magnitude category error; 2025 edge = single ZEC; coverage caveat). EXIT: ATR-trail >
  flush OOS but k is NOISE (IS/OOS rho 0.437) -> use k~3 generic; let-run/TP/flip/time all LOSE to flush OOS.
- DEPLOYABLE PLAY: mom14-K5 **r3** + **BTC-market gate** + ATR-trail-k3 = de-risked beta w/ a thin breadth-dependent
  selection tilt (+7,812% full / -56% DD); survives jackknife NOT MC = gated-beta book, NOT verified alpha.

## CYCLE 3 — NARROW + DECISIVE (90% converged): pre-registered selection test + the 2025 chop hole — RUNNING (workflow wu7fizcnf)
(1) Make-or-break: PRE-REGISTER mom14-K5-r3 (NO sweep — that DoF killed significance); test the OOS selection edge vs
random-gated-5 via MOVING-BLOCK BOOTSTRAP (not the iid shuffle) + leave-one-year-out walk-forward. Clears 0.05 pre-registered?
(2) The 2025 chop hole: dual-condition exposure (BTC-SMA200 AND breadth-%above-SMA50) to separate clean-downtrend(cut) from
chop. DO NOT re-mine u50/exit-tuning/per-asset-gates/slow-rebal (closed). If (1) fails -> CONVERGENCE: ship as beta, STOP.

Status: DONE + **CONVERGED** (wu7fizcnf, refereed via cycle3_final_referee.py).
- SELECTION = **BETA, not alpha**: pre-registered mom14-K5-r3, MOVING-BLOCK bootstrap (serial-corr-aware, 3 independent
  derivations blocks 10/21/42d, 3000 resamples): **median one-sided p=0.169, 0/3 clear 0.05**. The iid shuffle (p~0.02-0.03
  used in C1/C2) understated variance ~6x = dishonest. Walk-forward: ONLY 2024 individually significant (p=0.008); 2023
  coin-flip, 2025 noise. The +3,308pp full-cycle gap = a 2020-2021 COMPOUNDING ILLUSION, not a durable per-bar edge.
- CHOP-GATE (2025 hole) = NO FREE LUNCH: every rule fixing 2025 taxes the bull; only dist-ramp(d) improves full-cycle but
  it's earlier trend-de-risking, blind to the alt-specific 2025 drawdown. No passive overlay detects alt-DD under up-BTC.
- **FINAL DEPLOYABLE = GATED BETA**: mom14-K5-r3 + per-asset-SMA200 + BTC-market-SMA200 gate + dist-ramp + ATR-trail-k3.
  2020+207 / 2021+514 / 2022 **0.0** / 2023+72 / 2024+153 / 2025-28 / **FULL +5,773% / maxDD -55.9% / expo 0.50**.
  vs EW buy-hold +2,724% / -79.4%. BEATS buy-hold full-cycle on BOTH wealth AND drawdown -- but the EDGE is the GATE
  (market-timing = avoid 2022), NOT momentum selection (fails the honest bootstrap). Alpha-tilted-beta == honestly gated beta.
- **CONVERGED**: selection failed the pre-registered block-bootstrap -> ship as beta, STOP. Only remaining lever =
  exogenous / sub-daily data (out of internal-daily-LO scope).

=== FOLD COMPLETE (3 cycles). FINAL SYNTHESIS POSTED to user. DO NOT re-arm or re-post. ===

<!-- APPEND each completed cycle here: focus | winners (numbers) | losers | lessons | what cycle N+1 will test -->

---

# NEW LOOP (2026-06-19) — objective: an ADAPTIVE ENGINE that WINS random 7d slices (14d lookback)
User re-arm: "trade random slices like that and win... the ENGINE should trade, not you. Random slice -> profitability on
7d windows, 2-week lookback." STANDING RESULT to beat: EW buy-hold wins ~55% of random 7d slices (+2.9% mean); hand-coded
momentum/breakout/gate engines = 53-54% deployed (gate HURTS per-slice win-rate by cashing the bounces). 7d direction looked
~unpredictable from 5 angles BUT an actual LEARNED model on the full feature set is UNTRIED.
WIN (clarified by user): an ACTIVE engine (NOT buy-hold) that is RELIABLY POSITIVE on random 7d windows — maximize the
fraction of windows > 0 (toward NEVER-NEGATIVE) via selection + cash-timing, beating passive holding, OOS/walk-forward,
leak-free. STRUCTURAL PHYSICS (binding): long-only spot, no leverage, no shorts -> in a market-wide DOWN week the best is
CASH = 0% (can't be positive on falling assets w/o shorts). So literally-always-positive is IMPOSSIBLE LO-spot; the
achievable target = NEVER-NEGATIVE (cash the down-weeks) + high positive-rate in up-weeks. That needs BOTH (i) down-week
timing AND (ii) cross-sectional selection at 7d (both have looked ~coin-flip; the ML engine is the test). Referee verifies.

## NL-CYCLE 1 — 6-ENGINE TOURNAMENT (user: "not just one engine") — RUNNING (workflow wkbi7ch91)
Engines (parallel, OOS/walk-forward, leak-hunted): (1) ml_gbm meta-labeler; (2) ml_ensemble (logistic+RF+MLP x 3 labels,
stacked); (3) smartweight smoothing (inverse-vol/risk-parity/quality); (4) wide universe u30/u50; (5) **regime_router** =
the ADAPTIVE META-ENGINE that detects regime {uptrend/recovery-bounce/chop/downtrend} and ROUTES to the best sub-behavior
(momentum-concentrate / momentum-no-gate-catch-bounces / diversified-smooth / cash) -- the literal 'adapts to conditions'
engine; (6) non_momentum (reversal/range/vol/RSI composite). Referee = tournament judge, re-derives the winner OOS strict-
walk-forward, hunts ML leak. WIN = an ACTIVE engine reliably-positive ABOVE buy-hold (~55%/+2.9%) OOS leak-free; never-negative.

---
# EXPANDED SCOPE + 6h AUTONOMOUS MANDATE (user, 2026-06-19, deadline Sat 05:25 SAST)
"Expand scope, use the WHOLE project, formalise the fold framework (canonical: docs/META_FOLD_FRAMEWORK.md), run 6h
autonomous, expect collapse/early-stopping. Skin in the game: losing is out of the picture given timeframes, chart types,
chimera, technical indicators, different skills & experts." => the fold now searches the WHOLE space and collapses it:
- FOLD DIMENSIONS to attack (kill dead ones fast = the collapse): {7d-frame} x {u10->u50} x {1d,4h,2h,1h,30m,15m + dollar/
  DIB/range bars} x {price-TIs + CHIMERA families: funding/basis/ETF/on-chain/order-flow/LOB/TE/stablecoin/DVOL} x
  {hand-rule/ML/ensemble/regime-router/expert} x {expert-discover/quant/trader/oracle/pipeline lanes}.
- BIGGEST UNTAPPED LEVER = CHIMERA features into the ML/router engines (price-only is the current tournament; exogenous
  signal is where the one prior held-out-positive lived = funding-dispersion). Next cycle after the price tournament.
- WIN bar unchanged: an ACTIVE engine reliably-positive ABOVE buy-hold on random 7d slices, OOS/walk-forward, leak-free,
  never-negative. Referee re-derives + hunts leak every cycle. Early-stop on convergence (collapse) -- do not pad to 6h.
- NL-Cycle 1 = the 6-engine price tournament (wkbi7ch91, RUNNING). Folds: C2=chimera-feature engines; C3=multi-TF/bar-types;
  C4=expert-skill lanes; converge.

## NL-CYCLE 1 RESULT (wkbi7ch91 DONE, canonical leak-free referee_harness.py, N=5000, OOS 2022+) + NL-C2 PLAN
- LEADERBOARD pos-rate: EW buy-hold **52.3%**/+0.46% mean; inv-vol 52.9% (TIE, pure smoothing perm-p 0.71); **regime-router
  50.6%/+0.83% mean/down-wk -3.0%/58% expo** (BEST mean+tail, BELOW BH pos-rate); ML 46% (anti-selective); gated ~40% (cash kills pos-rate).
- NO active engine beats BH pos-rate OOS leak-free. ML: NO LEAK (strict-WF AUC 0.509) NO SKILL (date-block perm p=0.146; iid p=0.018
  = autocorrelation trap). 7d direction UNFORECASTABLE from PRICE features.
- STRUCTURAL: pos-rate is the WRONG objective LO-spot (cash=0=non-positive -> gating LOWERS pos-rate to 30.4%); only MEAN+TAIL liftable.
  "55%" was a bull-window artifact (canonical BH 52.3% on 2022+).
- ROUTER = real de-risked beta: selection survives same-exposure shuffle p=0.000; MEAN +0.83 vs +0.46 (+80%); TAIL -3% vs -7% (2022 -8% vs -71%).
- NL-C2 (the whole-project expansion; referee named EXOGENOUS signal as the only pos-rate lever): (1) CHIMERA-feature ML meta-labeler
  (funding/basis/ETF-flow/on-chain/order-flow/stablecoin/DVOL/TE; skip xex_) -- does exogenous signal FORECAST 7d direction where price
  can't (date-block-perm AUC)? = the RIGHT pos-rate lever (predict UP, don't cash). (2) chimera regime-conditioner on the router (mean/tail).
  (3) HARDEN router (x inv-vol + BTC 10% floor to recover pos-rate ~BH w/o forfeiting tail). (4) revisit funding-dispersion. Referee
  re-derives + date-block-permutation. If chimera ALSO fails -> internal+exogenous exhausted for 7d direction -> re-baseline to wealth/maxDD/mean.

## NL-CYCLE 2 — chimera exogenous features + router hardening — RUNNING (workflow wg38jr7wi)

## NL-CYCLE 2 RESULT (wg38jr7wi DONE, refereed, leak-hunted) + NL-C3 PLAN
- CHIMERA LEAK HUNT: **NO LEAK** (4 vectors clean: date bit-identical, no forward-fill, z-scores backward-looking, ETF zero-weight).
  Strict-WF AUC reproduced 0.4705 vs lane 0.4709.
- CHIMERA 7d DIRECTION: **NO** (exog-only AUC 0.47, block-perm two-sided p=0.15 = indistinguishable from 0; sub-0.5 = base-rate artifact).
  pos-rate 38-46% < BH. EXOGENOUS also can't forecast 7d. Conditioner on router = NOISE (Δpos -1.1pp, p=0.95 worse; redundant w/ price-regime).
- ROUTER HARDENING = hard Pareto TRILEMMA: pos-rate OR mean+tail, never both (downtrend=47% OOS, cash=non-positive). Deployable points:
  Router plain 50.6%/+0.83% mean/p05 -9.83%/+211% comp; BLEND α=0.6 52.0%/+0.66%/+99% comp (closest to BH pos-rate keeping mean>BH).
- FUNDING: real but tiny (rank-IC -0.063, ICIR -0.18, p=7e-13, low-fund outperforms) -- LO-blocked; +7.9% prior was large-universe LS;
  u10 insufficient dispersion. ≥30-name MARKET-NEUTRAL falsifier separates "no edge" from "LO-blocked harvestable" (MN=no-shorts=off-scope).
- VERDICT: **7d pos-rate NOT beatable above BH 52.3% with internal OR exogenous signal -- both EXHAUSTED for 7d direction** (structural:
  LO-spot can't be positive in down-weeks w/o cash, cash=non-positive). The ROUTER is the validated winner on MEAN+TAIL (shuffle p=0.0000).
- NL-C3 (FINAL whole-project sweep = the "different skills & experts" dimension): EXPERT lanes (expert-discover/oracle/trader each PROPOSE+TEST
  a novel angle the price+chimera cycles missed) + chimera-as-EXIT/SIZING on router winners (untested placement) + multi-TF entry-timing within
  the 7d window. If all fold -> CONVERGE: router = ship-grade internal-data ceiling; pos-rate needs market-neutral/external (off-scope).

## NL-CYCLE 3 — FINAL whole-project sweep: EXPERT lanes + multi-TF/exit threads — RUNNING (workflow wfxp03z74)

## NL-CYCLE 3 RESULT (wfxp03z74 DONE) — CONVERGED (capstone)
- NO expert angle beat BH pos-rate or improved the router leak-free. Trader "profit-lock 52.3%" used a LEAKED threshold (0.010 OOS-peeked;
  real calibrated 0.0338 -> pos 50.8-51.7% = ties BH). MC: 0 discoveries (best perm-vs-BH p=0.052; BH needs <=0.005). The real pos-rate lift
  is MECHANICAL (Pareto reshaping: buy pos-rate by selling mean), not alpha. multitf-4h caught a min-close leak then failed null p=1.0.
- STRUCTURAL CEILING PROVEN 3 ways. **CASH THEOREM**: cash=0=non-positive, so week-timing can NEVER raise pos-rate above the in-market basket
  up-rate (~51-52%) -- DIMENSION-INVARIANT (holds for any TF/bar-type/signal, so chart-types can't escape it either). P(7d>0)>50% for BTC only
  (51.5%); all alts <50% (pump-and-bleed). EW-basket up-rate 51.15%; best in-sample-cherry basket 52.83% (needs the dead direction signal).
  **>52.3% reliably-positive is STRUCTURALLY IMPOSSIBLE long-only-spot.**
- DEPLOYABLE = the ROUTER (adaptive_meta_engine.py): pos 50.73% / mean +0.96% / p05 -8.86% / down-wk -3.03% / expo 0.58 vs BH 51.73% /
  +0.43% / -13.14% / -6.50% / 1.00. Trades -1pp pos-rate for +0.53pp mean + 4.28pp tighter tail; real selection skill (shuffle p=0.0000). Ship plain router.
- VERDICT: internal long-only-spot daily = de-risked-beta ceiling, NOT pos-rate alpha. Whole project exhausted (price/chimera/multi-TF/experts,
  all leak-free). Only frontiers = market-neutral funding (>=30 names, LO-blocked) + external-event data -- both off the user's constraints.
=== FOLD CONVERGED (new-loop C1->C3, ~1h of the 6h budget = the early collapse the user predicted). FINAL SYNTHESIS POSTED. DO NOT re-arm. ===

## CORRECTION (user /orc challenge, 2026-06-20): engine traded BTC not MOVERS = FRAMING COLLAPSE
- FAILURE MODE: the prior fold optimized POS-RATE/BEAR-SURVIVAL (where the per-asset SMA200 GATE wins), NOT the user's MOVE-CAPTURE
  objective. The gate STRUCTURALLY deletes the movers: #1 forward-7d mover GATED-OUT 39% of days; mover is BTC only 9% of days; 38% of
  top-3 movers excluded. Every engine held majors. = framing_collapse + literal-over-spirit.
- COVERAGE: ~90% on direction/pos-rate; ~50% on the actual MOVE-CAPTURE objective. NOT >90% on the real ask.
- ENGINES: ~24 variants built, ZERO were a true UNGATED move-capture engine judged on CAPTURE (daily-movers lane killed on risk-adjusted DD).
- FIX (NL-C4 RUNNING wayjkpshd): UNGATED move-capture (rank all assets by move-signal, no per-asset gate, ride movers) + MARKET-LEVEL
  circuit-breaker (scale TOTAL exposure, never exclude a mover) judged on CAPTURE + slice-profitability + bear-survival; + mover-ID + capture-oracle.

## ORC FLEET-SEARCH C1 (wqwvsdhza DONE, expert-auditor refereed, DEV-walled leak-verified)
- WALL HELD: load_wide max date 2024-05-14 < 2024-05-15; causality verified 2 ways; OOS/UNSEEN untouched. ✓ (the #1 no-repeat discipline)
- Best agents (10-seed, 200 slices): EW 1.90% | mom14-K5 2.93% | evolve-champ(brk14+mom30+volexp-)K3 3.23% | 4-agent FLEET 2.44%.
- FLEET beats EW robustly (+0.54pp, 10/10 seeds) but does NOT beat the best single agent mom14 (-0.48pp, 9/10) -- ensemble only buys
  lower variance + better worst-slice. Profit-rate ~52% (≈EW) -> edge is mean-MAGNITUDE not hit-rate.
- INFO SETS: TI momentum/breakout (mom14/mom30/brk14) = the whole signal = DE-RISKED BETA not concentratable (K=1 excess NEGATIVE,
  peaks K=3-5). 1d chimera = ZERO univariate edge, dilutive. multi-TF 4h price = redundant. 4h-funding NOT significant (+0.18pp p=0.231).
- OVERFIT: 300+ configs on same DEV slices; mom14 excess-over-EW block-bootstrap p=0.124 (n.s.). Honest = drawdown-de-risked momentum
  beta (beats EW by losing less in down weeks; 2022-08-16 -6.2% vs EW -12.2%); expect OOS shrinkage. Deployable: fleet_final.py 4-agent +2.44% DEV.
- C2 (folding): the ADAPTIVE regime-conditional fleet (route agents by regime = the "adapts to conditions" vision, untested) + u100 wider bench;
  strict overfit discipline (pre-register, multi-seed, block-bootstrap). If it doesn't beat static momentum robustly -> converge on the de-risked-beta deployable for OOS handoff.

## NL-C4 MOVE-CAPTURE (wayjkpshd DONE, quant-refereed) -- STRONG LEAD but on OLD OOS harness (pre-data-discipline)
- Ungated mover engine (mover-rank top-K + MARKET circuit-breaker) TRADES THE MOVERS (2025-05-15: ETH/DOGE/LTC; gate would delete
  DOGE/AVAX/SOL). Gate-exclusion of #1 mover = 57% of days (worse than 39%). BEATS router+BH on MEAN (+1.11 vs +0.96 vs +0.43) +
  CAPTURE (17.3% vs 14.8%); LOSES pos-rate (46% vs 52%) = concentration cost. Market-CB recovers bear (-69%->-8/-16% maxDD) w/o excluding movers.
- **DECISIVE NEW EVIDENCE = same-exposure SHUFFLE control: real mover-selection +1.11%/17.3% vs random-same-exposure +0.08%±0.19%/9.5% ->
  z=5.3(mean) z=6.2(capture) p≈0.0000.** => cross-sectional mover ranking adds REAL SELECTION ALPHA above equal-exposure random, NOT just timing.
- CAVEATS: (a) ran on OOS 2022+ (referee_harness) = HELD-OUT, pre-discipline -> MUST re-prove on DEV (<=2024-05-15). (b) Referee FALSIFIER:
  run the shuffle STRATIFIED BY REGIME -- if selection alpha vanishes in bull (everything rises) it's a beta artifact, not skill. UNTESTED = make-or-break.
  (c) vol_exp "p=0.032" FAILS multiple-comparisons (best-of-5); composite engine survives (shuffle z=5-6). capture mean(eng/oracle) broken -> use aggregate.
- RECONCILE vs C1-fleet (DEV, "de-risked beta, K=1 negative"): different NULL -- C1 compared vs EW (beta); NL-C4 vs random-SAME-EXPOSURE (selection).
  The selection null is the cleaner skill test. => DECISIVE CYCLE: re-run move-capture engine + same-exposure shuffle + REGIME-STRATIFICATION on DEV-walled fleet_lab.

## DECISIVE CYCLE -- move-capture selection-alpha on DEV (regime-stratified shuffle) -- RUNNING (workflow wh20zajez)

---

## DECISIVE CYCLE -- move-capture selection-alpha on DEV (regime-stratified shuffle) -- VERDICT: ARTIFACT (2026-06-20 01:58 SAST, workflow wh20zajez)

**Pre-registered decisive test FAILED -> internal-data 1d mover-selection CLOSED (converge).**

The NL-C4 lead (ungated mover engine beats random-same-exposure by z=5-6) was re-proven on DEV-walled fleet_lab
(<=2024-05-15) with the regime-stratified same-exposure shuffle + honest non-overlapping (stride=7) z + moving-block
bootstrap. Referee (expert-quant) independently re-derived (`quant_referee_mover_dev.py`).

- **DEV WALL HELD** (verified 3 layers; every result JSON caps at 2024-05-14; oos_validate never called).
- **The z=5-6 was a POOLING + OVERLAPPING-WINDOW artifact.** Honest pooled z = 1.5 (n_eff=197), and it is ENTIRELY bull.
- **DECISIVE (regime-stratified paired same-exposure, K=3 derivations unanimous):**
  - bull: +2.11pp, honest z=1.74, block-boot p05=+0.25pp, frac>0=0.96  -- marginally real, bull-ONLY
  - chop: -0.08pp, z=-0.83, p05=**-1.20pp**, frac>0=0.22  -- ARTIFACT (selection NEGATIVE)
  - bear: -0.16pp, z=-0.56, p05=**-0.29pp**, frac>0=0.30  -- ARTIFACT (selection NEGATIVE)
  Outside a confirmed bull tape, mover-selection ACTIVELY UNDERPERFORMS the same-exposure no-skill control.
- **Aggregate capture** (honest, dollar-wtd, K=3): engine 7.91% vs shuffle 7.14% = +0.77pp, INSIDE seed noise (5.5/8.3/7.7).
- **Identification AUC = 0.549** (p=1.6e-5): real but economically trivial (random=0.5). Real-but-unharvestable -- the
  project's recurring signature, now confirmed for mover-selection.
- **Deployable** mover_capture_K3 full-DEV: comp +3784% but **maxDD -86.5%** -> FAILS the <30% DD bar = leveraged-into-bull
  beta, NOT a risk-controlled book. vol_hi calibration-sensitive (q75-pre2022=0.83 vs q80-1H=0.046) = not robust.
- **One dissenting lane REFUTED:** move_capture_fleet.py:578 `z_delta` = difference of two BH-relative t-stats, never forms
  the paired same-exposure diff -> its "chop z=+2.22 REAL" has no valid null. Distrust any z=5-6 without the regime split.

**The honest residue:** within a CONFIRMED bull tape the composite picks ~+2-3.8pp/7d above a same-exposure market
portfolio (survives beta-decontamination so not PURE beta-loading) -- a regime-GATED bull-beta ENHANCER, sub-2sigma,
the same participate-preserve / de-risked-beta wall, NOT the breakthrough. The -86% DD is fixable ONLY by regime-gating
to cash in chop/bear (= the participate-preserve book, with the mover-selection as its bull leg).

**FOLD -> next cycle (FRAME-BROADLY, the user's explicit not-explored axis):** the entire fleet campaign has been 1d-only.
The user REPEATEDLY asked for SUB-DAILY (<1d) + multi-timeframe + chart-types ("24 4h bars < 24 days"). Chimera has
15m/30m/1h/4h + dollar/dib/range on disk. CYCLE = extend fleet_lab to sub-daily TFs and re-run the move-capture +
regime-stratified test there: does faster-cadence move-capture beat the same-exposure shuffle ACROSS regimes (where 1d
failed), or is the de-risked-beta wall TIMEFRAME-INVARIANT? This is the genuine widening, not a manufactured cycle.

_Sub-daily cycle RUNNING (workflow w3bbolocj, launched 2026-06-20 02:0x SAST): 4h/1h move-capture + regime-stratified shuffle + exit-mechanism lever + TF-invariance referee. Foundation: fleet_lab TF-parametric, smoke-verified @4h._

---

## SUB-DAILY CYCLE -- move-capture @ 4h/1h -- VERDICT: ARTIFACT, the wall is TIMEFRAME-INVARIANT (2026-06-20 02:2x SAST, workflow w3bbolocj)

**The de-risked-beta wall holds at EVERY cadence. Sub-daily is the same wall at a faster clock. CONVERGE on selection-vs-shuffle.**

Per-regime selection-alpha under the HONEST hold-to-maturity null (block-boot p05), across TFs:
| regime | 1d | 4h | 1h |
|---|---|---|---|
| bull | +0.93 REAL | +1.72 REAL | +0.87 REAL |
| chop | -0.11 ARTIFACT | -0.30 ARTIFACT | -0.55 ARTIFACT |
| bear | -0.40 ARTIFACT | -0.16 ARTIFACT | -0.27 ARTIFACT |
Only BULL survives at any TF = long-beta concentration in a rising tape. Chop/bear NEGATIVE everywhere.

**METHODOLOGY BUG FOUND + FIXED (bigger than this cycle): the same-exposure random-K SHUFFLE is a BROKEN NULL.**
Re-drawing K names every bar, the random control eats a per-bar reshuffle-variance penalty that smoother books don't.
Quantified (meta_tf_stress): 4h-chop control -10.99 bp/bar vs real book -2.75 bp/bar -- BOTH negative gross; the
"+3.36pp alpha" was ENTIRELY the control being churn-penalized, NOT skill. The NL-C4 z=5-6 lived on this broken null.
**The HONEST null = HOLD-TO-MATURITY (top-K, hold the slice, ~1 RT) + 4 cross-checks, all of which killed it:**
  1. hold-to-maturity null: chop/bear ARTIFACT at every TF (decisive)
  2. regime-label shuffle: "alpha" survives shuffling regime labels = regime-INDEPENDENT churn (4h chop +4.86 ~ bull +4.89)
  3. reverse-score: WORST-K stays positive (+4.3pp chop 4h) = direction-blind concentration/compounding, not selection
  4. calendar-day scaling: chop alpha/day grows -0.02 -> +0.48 -> +1.98 pp/day with bar-count = compounding-of-churn fingerprint
Shift-2 retention ~100% rules out feature look-ahead -- the artifact is the RESHUFFLE mechanic. Cheapest falsifier for any
revived sub-daily selection: re-run the cell with the hold-to-maturity null (S3 in meta_tf_invariance_audit.py); p05 stays <0.

OTHER: exit mechanisms do NOT rescue chop/bear (worsen bear; lone chop positive flips REAL->ARTIFACT on a 1pp trail-width
change, killed by BH q=0.10). Cost is the secondary executioner (16->40->80%/yr taker drag 1d->4h->1h). ID AUC DEGRADES with
cadence (0.549->0.522->0.487, anti-predictive @1h) -- "more bars to confirm onset" REFUTED.

**FOLD -> CONVERGENCE on selection-vs-shuffle (internal TIs x {1d,4h,1h} x {fixed,trail,target,time,ATR} exhausted).**
PIVOT to the USER'S BINDING DIRECTIVE + the referee's lever #2, which is the SAME axis: the **TI move-CATCH thesis judged by
CAPTURE-RATE** (realized/available move within the signal window) -- a DIFFERENT, churn-IMMUNE null (per-signal entry/exit,
no per-bar reselection). NOT selection-vs-random-portfolio. This is the charter's deferred move-CATCH product space
(TI x [Chimera] x Asset x TF x exit-mechanism), and it sidesteps the exact artifact that broke every shuffle lane.
EXTERNAL data (Coinbase/Upbit) stays DEFERRED behind internal move-CATCH per the standing directive.

_Move-CATCH capture-rate cycle RUNNING (workflow w9wj1w70h, launched 2026-06-20 02:1x SAST): tests whether the chop momentum/breakout move-catch edge (churn-immune random-ENTRY null; mom14 +0.98pp/1d chop, p~0 iid) survives the full adversarial battery (date-block moving-block bootstrap + reverse-score + regime-label shuffle + calendar-invariance) -- REAL non-bull edge or wall under a new lens. Lanes: chop_battery / product_sweep / chimera_conditioner / exit_mechanism + expert-quant referee. Foundation: capture_lab.py (DEV-walled, aggregate capture, regime split), verified 1d+4h._

_OVERSEER independent read (2026-06-20 02:2x SAST, while w9wj1w70h runs) -- NON-OVERLAPPING (stride=hold) + reverse-score +
regime-label-shuffle check on the chop move-catch lead, 1d time-exit, DEV-walled:_
- _chop edge SURVIVES independent (non-overlapping) samples: mom14 +0.98pp p(<=0)=0.003 (p05 +0.41), brk14 +2.42pp p~0 (p05 +1.37), rsi14 +1.42pp p=0.0003. Bear NEGATIVE (mom14 -0.38, brk14 -2.13) -- wall holds in bear._
- _REVERSE-SCORE (worst-momentum mom14<0) LOSES in chop (-1.41pp, p=1.0) => the edge is DIRECTION-SENSITIVE = REAL trend-continuation, NOT the direction-blind concentration churn that killed the sub-daily shuffle lane (where reverse-score stayed positive). This is the key contrast: the churn-immune null + direction-sensitivity distinguishes a real momentum-catch from the prior artifact._
- _Regime-label shuffle does NOT vanish (+2.14pp) => the edge is GENERAL (positive in bull AND chop), not chop-SPECIFIC. The claim sharpens from "chop edge" to "a general direction-sensitive momentum/breakout move-catch, positive in non-bear, that the BROKEN shuffle null had masked in chop." CAVEAT held for referee: the MFE>3% conditioning isolates timing-skill-GIVEN-a-move; the UNCONDITIONAL tradeable P&L (when the TI fires, how often a move occurs + net) is the open deployability question for the exit/sweep lanes._

---

## MOVE-CATCH CAPTURE-RATE CYCLE -- VERDICT: SAME WALL UNDER A NEW LENS (2026-06-20 02:5x SAST, workflow w9wj1w70h)

**The chop move-catch "edge" is up-regime continuation, NOT regime-conditional. De-risked-beta wall re-confirmed via capture-rate.**

The churn-immune random-ENTRY null DID show chop momentum/breakout beating random entry (mom14 +0.98pp, brk14 +2.42pp,
honest date-block p_le0 ~0.02) -- survives block-bootstrap AND reverse-score (direction-sensitive). BUT the two decisive kills:
- **REGIME-LABEL SHUFFLE (done correctly: pool re-drawn under rotated labels) FAILS.** Real chop edge sits BELOW the
  label-destroyed mean (p_shuf>=real = 0.79-0.83 for mom14/brk14/rsi14). Destroying regime labels makes the edge BIGGER
  (rotation imports bull entries) = the signature of NON-regime-conditional up-regime continuation, not a chop signal.
- **CALENDAR-INVARIANCE FAILS.** mom14 chop +0.98 (1d) -> +0.54 (4h) -> -0.15pp (1h); brk14 +2.42 -> +0.69 -> -0.23.
  Edge decays to NEGATIVE with bar density = overlap/selection signature, not a stable per-day economic edge.
- Holm across the non-bull family: only rsi14_chop_4h survives block-p (0.048) but FAILS regime-shuffle (p=0.55) = moot.
- BEAR: 0 survivors, all reverse-score direction FALSE = genuinely anti-predictive. brk14 bear -2.1..-2.9pp (bear-traps).
- Chimera CONDITIONERS add nothing (<=+0.02pp; 3-gate stack drops chop +0.98->+0.08pp by thinning).
- Mechanism exits: time-stop has the HIGHEST capture (0.59-0.63); trail/target REDUCE it (crypto tails); none rescue bear.
The only real component = a faint UNIVERSAL momentum-continuation tilt (reverse-score) positive in up-regimes, negative in
bear = the de-risked-beta wall exactly. NOT a breach. Methodology fix applied: wired the date-block bootstrap into
capture_lab.evaluate_ti(block=True) (iid deflated SE 2-3.6x); reproduces referee (mom14 chop n_eff=317).

**FOLD -> the LAST untested INTERNAL axis (charter-mandated before external): the WASTED v51 EXOGENOUS features.**
The "chimera dead" verdict was on 4 features (vpin/dev/fd) at 1d. The dollar-bar files carry the v51 chimera (~250 cols):
funding, basis, liquidations, ETF/whale flow, stablecoin shocks, order-flow (kyle/hawkes), transfer-entropy, listing-age.
Built v51_feature_lab (resamples these to a causal daily grid -> capture_lab's hardened battery). FIRST READ: exogenous
features show DIFFERENT regime patterns than price-TIs -- norm_funding positive in ALL regimes incl bear +0.54; s3_smart_
vs_retail_z FLIPS to +0.82 in bear; liq_capitulation/short_panic +3.2pp chop. None yet bear-significant. CYCLE wob5hmhhu
runs the full battery + BOTH directions + causality audit: does any EXOGENOUS feature break the wall (regime-conditional
or bear-positive)? If no -> internal EXHAUSTED, external (Coinbase/Upbit) is the charter-blessed redirect.

_v51 exogenous move-CATCH cycle RUNNING (workflow wob5hmhhu, launched 2026-06-20 02:5x SAST)._
