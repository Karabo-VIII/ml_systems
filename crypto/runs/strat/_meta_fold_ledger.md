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
