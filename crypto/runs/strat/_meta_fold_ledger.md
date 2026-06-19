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
