# CURRENT — live state (Kiro-written each cycle; committed; <2KB)

PHASE: V6 RIGOROUS STATIC COMPLETE — READY FOR PHASE 2 DYNAMIC
UPDATED: 2026-06-25

## VERIFIED ARCHITECTURE
Layer 1: V1 regime gate (1d BTC SMA100, hysteresis) → bull/bear switch
Layer 2: V2 4h decoupled engine (MA band + 7d lookback) → position timing

Results (DEV window 2020-2024):
  EMA combined:  +171,610%, MaxDD -42%, bear 2022 = -2%
  WMA combined:  +229,783%, MaxDD -48%, bear 2022 = -3%
  DEMA combined: +691,448%, MaxDD -45%, bear 2022 = -6%

Random-entry null: PASSED (100th percentile, all 3 types)
Per-week: 56-59% profitable, median +3-4%/week, deployment 35% in 4h layer

## ACTIVE CAMPAIGN
STATIC PHASE — CLOSED + EXHAUSTED across 21 TI families (2026-06-24).
3 SELECTIVE validated engines: MA WMA|4h, Bollinger|1d, RSI|4h (deploy MA+RSI; Boll
0.96 corr w/ MA = redundant). PLUS a bench of 8 NON-SELECTIVE long-tilt overlays from
V4 (CCI/WILLR/VWAP|1d strongest, 100th-pct null, +2-3pp OOS but deploy 80-96% = refine
WHICH longs, not WHETHER long). See runs/engines/CONSOLIDATED_ENGINE_REGISTRY.md.
Phase 2 (dynamic) has a working foundation + concrete target (turn V4 tilts selective).
NOT started — user gate.

## V5 DNA + HYPER-AUDIT (2026-06-25) — full TI×TF performance DNA, no exclusions
Pipeline AUDITED 19/19 (causal/walled/cost/null/rebal-premium all verified) →
runs/engines/V5_dna_report/PIPELINE_AUDIT_RESULT.md. Key audit finding: rebalancing
premium is WINDOW-dependent — ~9.5x on full-DEV but only ~1.1x on OOS-2024, so OOS-2024
compounds are ~honest. Look-ahead probe decisive (cheat +65%→+4217%; production causal).

Full DNA: 26 families × 3 TFs (1d/4h/2h) on VAL (2023-07-01..2024-05-15) + OOS-2024
(2024-05-15..2024-12-31, 2025 SEALED). → runs/engines/V5_dna_report/DNA_REPORT.md (every
cell, ranked by transparent 0-6 robustness score). Findings:
- 25/78 cells DEV/OOS-consistent; 21 also beat OOS random-null ≥95th (genuine timing).
- ONLY ADX is genuinely SELECTIVE (~70% deploy) — but it FAILS VAL/OOS consistency.
  Every other consistent cell deploys 96-100% (always-long-with-tilt under the gate).
- Top robust tilts (5/6, consistent + 100th-pct null, non-selective): HMA|1d (+2.66pp OOS),
  WILLR|1d (+1.29), Bollinger|1d (+1.29), TEMA|1d (+1.12), VIDYA|4h, SMA|4h, VWAP|1d,
  MFI|1d, Supertrend|4h, WMA|4h, PSAR|4h. 1d and 4h carry the signal; 2h is noise (sign-flips).
- This RECONCILES the earlier "Bollinger|1d / RSI|4h validated" claim: under this larger
  harness Bollinger|1d holds (robust tilt), but it is NOT selective — the V3 "selective"
  label conflated low-deploy with the confirmation-layer slice-hold. Honest status: robust
  NON-SELECTIVE tilt, same as ~13 others. RSI|4h is OOS-only here (VAL edge -0.64) → weaker
  than previously stated. NOTHING is both selective AND consistent except via the regime gate.

## DEAD/NULL REWORK (2026-06-25) — DONE
ADX rework OVERTURNED its V3 "null" verdict: a proper lookback×threshold sweep reveals
ADX|1d (lb22-28, thr0.42-0.48) is the program's FIRST genuinely SELECTIVE (deploy ~40-66%)
+ VAL/OOS-consistent + 100th-pct-null engine. Overfit probe: NOT a spike — a directional
gradient (lookback≥22 consistently positive, 25/40 cells OOS+). HONEST caveat: lookback-
sensitive (dies <lb20), OOS edge (+0.6-1.1pp/wk) < VAL. Promoted VALIDATED-SELECTIVE
(PROVISIONAL): deploy only as ensemble over the robust pocket, re-confirm on 2025 when opened.
Stochastic: confirmed DEAD, but the scary -7.8pp was an INVERTED-LOGIC artifact (flip →
-0.44); both directions ≤0 OOS = genuinely no signal. → DEADCASE_REWORK_RESULT.md

## ROSTER NOW (audited)
- SELECTIVE validated: ADX|1d (provisional) + MA decoupled-
  BAND V2 (WMA|4h, the canonical selective winner from ma_wealth_driver).
- ROBUST NON-SELECTIVE tilts (consistent + 100th-pct null, deploy 96-100%): HMA|1d, WILLR|1d,
  Bollinger|1d, TEMA|1d, VIDYA|4h, SMA|4h, VWAP|1d, MFI|1d, Supertrend|4h, WMA|4h, PSAR|4h
  (+~10 more at score 4). These refine WHICH longs under the gate.
- The regime gate remains the dominant value lever (D78).

## NORTH STAR RECKONING (2026-06-27) — we FAIL the primary mandate on CONSISTENCY
The mandate (CHARTER) = "profitable within ANY 7-day window" (per-slice ROI>0, target ~100%),
NOT weekly magnitude (D77 retired that). Measured the deployable cross-type conv book (1d,
prod, OOS-2024): only **34% of 7d windows profitable** (median +0.00%, mean +2.34% = right-
skewed: 34% big winners carry it, 66% small losers). 3d=39%, 1d=32%.
VERDICT: trend-following is STRUCTURALLY low-hit-rate (lose small often, win big rarely) —
it CANNOT self-satisfy "any 7d window profitable" no matter the tuning. Both ceilings now
confirmed: MAGNITUDE ~+1.7%/wk, CONSISTENCY ~34% of 7d windows.
→ runs/engines/V14_slice_roi/NORTH_STAR_RECKONING.md
PATH FORWARD: the North Star REQUIRES anti-correlated complements (mean-reversion / market-
neutral / orthogonal signal families) that win in the choppy windows trend bleeds in. The
MA engine is ONE component (bull-trend harvester), not the system. Next campaign: build the
anti-correlated complements; measure per-7d-window consistency of the COMBINED book.
Tested user idea: reselect top-K configs per MA type every 14d from trailing P&L vs the
FROZEN TRAIN+VAL band. SOTA rigor (random-reselect null isolates selection skill).
RESULT: FROZEN WINS 11/12 cells; rolling-by-recent-P&L performs AT/BELOW random-reselect
(no skill), several cells COLLAPSE (KAMA|1d +1.58→+0.63). 5th independent confirmation of
D73 (config performance anti-persistent). DEPLOY FROZEN BAND. The 1/12 "win" (VIDYA|1d) is
within multiple-comparison noise. → runs/engines/V9_rolling_band/ROLLING_BAND_AB_RESULT.md

## PROD-GRADE DEPLOYABLE VERDICT (2026-06-27, real allocator, 15% cap, prod cost)
Top deployable MA engines (inv-vol + 15% cap, OOS-2024): SMA|1d +1.67/wk Sh1.57 DD-26%,
EMA|1d +1.68 Sh1.57, SMA|4h +1.60 Sh1.54, VIDYA|1d +1.57 Sh1.54, KAMA|4h +1.57 Sh1.55.
The per-name cap is the return dial: 15%=+1.4/wk, 33%=+1.6, 50%=+2.1, EW(no cap)=+3.4 raw.
3-5%/wk target requires high concentration (low/no cap). See V8 DEPLOYMENT_NOTES.md.

## ACTIVE CAMPAIGN: V6 RIGOROUS REBUILD — COMPLETE (2026-06-25)
All 12 non-MA families × 5 TFs rebuilt with full V1.1 rigor (CHARTER §ENGINE BUILD RIGOR).
57 cells evaluated. RESULT: ONLY ADX|1d is genuinely selective (67% deploy, +2.26pp/wk, 100th
null, Sharpe 3.38). Every other family deploys 98-100% at all TFs — the TRAIN filter cannot
reduce them because their signal logic is structurally always-on under a bull gate. Sub-daily
(1h/30m) universally bleeds from cost+noise.
→ runs/engines/V6_rigorous/V6_RESULT.md + v6_sweep_20260625_203231.json (1d/4h/2h) +
  v6_sweep_20260625_223748.json (1h/30m)

## ENGINE ROSTER (rigorous, final, OOS-validated 2026-06-26)
SELECTIVE + OOS-VALIDATED:
  1. MA V2 BAND (WMA|4h) — the canonical winner (ma_wealth_driver, 35% deploy, +4.11%/wk OOS)
  2. MA × 4h EFFICIENCY-GATED (m=0, eff>0.25) — 7/8 types OOS+ (60-72% deploy, +0.3-1.25pp)
     Best: DEMA|4h (+1.25pp, 100th null, Sh 2.02), VIDYA (+0.85), SMA (+0.91), EMA (+0.90)
  3. ADX|1d (V6 discovery, 67% deploy, +2.26pp VAL, Sharpe 3.38, OOS probe pending)
DYNAMIC ADAPTATION (m>0 vov-threshold): NULL OOS (VAL-fit, does not transfer)
MA × 2h eff-gate: NULL OOS (VAL-fit, extraordinary selectivity did not generalize)
NON-SELECTIVE tilts (1d only, 98-100% deploy): PSAR, OBV, DONCHIAN, RSI (tiny edges)
Static TI engine build COMPLETE AND EXHAUSTED. Dynamic adaptation (vov-threshold) REFUTED.

## WHAT EXISTS (verified, committed)
- V1 MA drawdown-insurance: 8 types × {1d, 4h} = VERIFIED (halves DD, -35% vs -82%)
- V1.1 fine-TF exits: REFUTED (look-ahead leak caught, de-leaked = null, D81)
- V1.2 combinatrix: 30 engines × 275 combos = disciplined NULL (D82)
- Decoupled 7d/14d architecture: BUILT (uncommitted) but NEVER RUN at 30m/1h/2h
- Chimera data: available at ALL cadences (15m/30m/1h/4h/1d), 104 assets, 243 features

## THIS SESSION (2026-06-24) — TI FAMILY VERIFICATION (honest read)
- [x] Forensic audit of SOTA numbers → FORENSIC_AUDIT_RESULT.md
      No look-ahead (clean). BUT compound inflated ~9.5x by EW-rebalancing premium.
      Single-asset BTC: MACD timing LOST to buy-hold (+162% vs +655%).
- [x] OOS validation RSI/MACD/Bollinger (2024-05-15→2025-01-01) → OOS_VALIDATION_RESULT.md
      Each beats AL at only ONE scattered TF (noise signature). NOT deployable.
- [x] Vectorized families 6-8 + wired REAL chimera features (norm_funding,
      norm_oi_price_divergence) → ti_engine_sota_v11.py. OOS: Supertrend/Funding/OI
      deploy 98-100% (always-long-with-tilt, not selective). NOT deployable.
- [x] Phase 2 dynamic-engine plan (thorough, not executed) → docs/PHASE2_DYNAMIC_ENGINES_PLAN.md

VERDICT (CORRECTED 2026-06-24 after static-exhaustion): My first "all non-MA null"
read was UNDER-POWERED (one threshold, no confirmation layer). Pulling the proper
levers (vote-threshold sweep + MA-style confirmation) + the gold-standard bar
(DEV/OOS consistency + random-null >=95th pct) REVEALED 2 NEW validated static engines:
  - RSI|4h: +0.77pp OOS, DEV+OOS consistent, 100th-pct random-null  [VALIDATED]
  - Bollinger|1d: +2.38pp OOS, DEV+OOS consistent, 100th-pct null    [VALIDATED]
  - ADX|4h: +1.09pp OOS, selective (6.9% deploy), but 93rd pct       [BORDERLINE]
  - MACD|1h: big OOS but NEGATIVE DEV -> OOS-fit                      [REJECTED]
  - Stochastic/Supertrend/Funding/OI: no selective edge              [NULL]
See STATIC_EXHAUSTION_RESULT.md. The user's "static before dynamic" gate is satisfied:
static demonstrably works (MA + RSI|4h + Bollinger|1d). Still TODO on static: lookback/
overfit probe on the 3 validated cells + validate their ENSEMBLE combination OOS.

## STANDING FINDINGS (from prior session)
- No daily-bar SELECTION alpha (D17/D55/D77 + 6 new experiments)
- Vol IS predictable (top feature: vol-rank, rv_rv_5m)
- Continuation > reversion at daily. REVERSAL at sub-daily (multi-TF analysis)
- Regime timing is the only verified value lever (V1, V3 selection)
- Trailing stops create illusion of alpha (vol-breakout audit)
- Oracle ceiling: 10.4%/day available (empirically real opportunity)

## SPLIT CONVENTIONS
TI/MA: 2020=DEV, 2021=IRON, 2022=no-touch BEAR.
Convention B: TRAIN<=2024-05-15 / VAL / OOS / UNSEEN (>=2024-05-15 sealed).

## V8 SLICE STUDY (2026-06-27) — PROD costs + rolling/fixed + whipsaw + canonical exits
Full A-E study. → runs/engines/V8_slice_study/SLICE_STUDY_RESULT.md + canonical_exits.py +
ma_slice_engine.py + ma_slice_study.py. KEY FINDINGS (OOS-2024, PROD taker+slip cost):
- COSTS NOW PROD-GRADE: taker 0.0024 + slippage = 0.0028 RT (prior reports were maker-optimistic).
- ROLLING >> FIXED slices: fixed force-exits at window boundaries, loses >half the edge at 1d
  (rolling +3.79 vs fixed +1.72). DEPLOY ROLLING (let positions run, exit on signal/time-cap).
- 4h is the SWEET SPOT: SMA|4h +4.02%/wk prod, Sharpe 2.12, cost-resilient. 6 types +3.4-4.0.
- 1d (14d/14d) best slice-profitability: 50-56% of 7d windows profitable, VIDYA +3.99, ~390 trades.
- 2h survives w/ whipsaw control (SMA +3.21); 1h marginal (+1.4 best); 30m/15m DEAD (cost wall,
  drag 4-5.7pp/wk, 9-10k trades).
- WHIPSAW CONTROL (min-hold+cooldown) works at 1h/2h: cuts trades 10-25%, lifts edge, rescues
  several 1h cells from negative. Recommended for 1h/2h; unnecessary at 4h/1d.
- EXITS: canonical_exits.py = confirmation-break + time-cap default, NO stops (D23/RED-TEAM
  destructive), take-profit/trailing available but OFF. Whipsaw modifiers for high-freq.

## CANONICAL EXIT LIBRARY (new, 2026-06-27)
crypto/src/strat/canonical_exits.py — THE exit module (was missing). Menu: TIME_CAP +
SIGNAL_BREAK (robust default), TAKE_PROFIT/VOL_TARGET/REGIME (scoped), TRAILING/STOP_LOSS
(risky, D23/RED-TEAM). apply_min_hold_cooldown for whipsaw. resolve_held = the resolver.

<!-- ============================================================================ -->
## CORRECTION (Claude, 2026-06-28): same-day regime look-ahead leak fixed
<!-- Appended by Claude; KIRO-owned body above is unchanged. -->
The DEPLOYABLE capped headline at L87-89 above (SMA|1d +1.67, EMA|1d +1.68, SMA|4h +1.60;
"cap dial 15%=+1.4 / 33%=+1.6 / 50%=+2.1 / EW=+3.4") was inflated by a SAME-DAY 1d-regime
LOOK-AHEAD (the V1.1 align_htf_on bug class): gating consumers broadcast the day-D 1d-BTC-SMA100
CLOSE regime onto intraday bars of the SAME day D, which isn't known until ~23:59. Confirmed +
falsified, then fixed DRY via a shared `causal_regime(wall_ms) = btc_regime.shift(1)` helper
(a bar on day D now reads the last CLOSED 1d bar = regime[D-1]).
- Impact is MODEST: ~1.7% of OOS-2024 bars flip; the edge retains ~85% (NOT the V1.1 catastrophe).
- **De-leaked headline (prod cost, OOS-2024): 1d cap15 +1.71 -> +1.564/wk** (Sh 1.46, DD -28.7%).
  De-leaked cap dial (1d): 15%=+1.56 / 33%=+1.82 / 50%=+2.19 / EW=+3.47 (flag 50%/EW as window-luck).
  4h cap15 +1.52 -> +1.30/wk. All cells stay positive after de-leak.
- Evidence: `runs/engines/AUDIT_ORACLE_UPGRADE/leakcheck.json`,
  `runs/engines/AUDIT_ORACLE_UPGRADE/audit_deployable.json`,
  `runs/engines/AUDIT_ORACLE_UPGRADE/wave2_concentration_frontier.json`;
  regression: `runs/engines/AUDIT_ORACLE_UPGRADE/leakfix_regression.py` (6/6 PASS).
<!-- ============================================================================ -->
## SOLVE-FOR-SOTA verdict (Claude, 2026-06-29) — the honest frontier
User: "solve so we get SOTA." Stacked the 5-lens reachable levers on the V5 base (WF deploy-
timing + momentum top-8). runs/engines/STRAT_TIERS/strat_sota_v6.py + strat_sota_validate.py.
- NEW levers do NOT improve V5: continuous Pwf-expo ~=base (DD -20->-17), vol-target HURTS
  (lev=1 can only de-risk; gate already does -> +168->+89%), residual-mom neutral. Convex
  tail-exit reasoned-out (it raises concentration -> worsens the p05 problem). V5 IS the frontier.
- V5 OOS: +168% 2024 (+3.55%/wk mean = in the 2-4% target), +14% 2025 bear, +215% all.
- RIGOROUS VALIDATION = the ceiling: BEATS temporal-shuffle null @ 100th pctile (WHEN-structure
  is REAL) BUT FAILS robustness floor: DSR 0.45 (<0.95, deflated by 24-config search) + block-
  boot p05<=0 every window (OOS-2024 p05 -0%/p50 +160%/p95 +894% = extreme right-skew; rides a
  few mega trend-weeks). => real-but-LUMPY selective timing edge. For a RETAIL WEALTH account
  (high median + bounded DD -17..-25%) it's the right trade; for institutional always-on p05>0
  it does not certify. Gap to +12-14%/wk oracle = un-capturable directional/which-name (rank-IC~0).
- NO FALSE VICTORY: SOTA-within-constraints = a genuine but fragile timing edge, NOT a robust
  2-4%/wk. Beyond this needs data the user has ruled out, or prediction skill price data lacks.
<!-- ============================================================================ -->
## RED-TEAM CORRECTION (Claude, 2026-06-29) -> VERDICT: PAPER-TRADE-ONLY
Adversarial verification (workflow wbw77ivx2, 4 lenses) REFUTED the over-kind "real-but-lumpy
frontier" framing above. Look-ahead CLEAN, but every pre-ship STAT gate FAILS:
- Newey-West(L3) HAC p=0.087 (the iid p=0.046 ignored autocorrelation ACF1-3 +0.13/.19/.35) ->
  NOT significant even before the search penalty. DSR 0.13@n100 / 0.07@n200 (n=24 under-counts).
- top-2 deployed weeks = 77% of OOS-ALL compound (remove top-5 -> NEGATIVE). p05<0 all windows.
- COMPONENT DECOMP: breadth-alone +57% / Pwf-timing-alone -7% / full +170%. The BREADTH gate is the
  REAL de-risking edge; the learned-timing overlay alone is NEGATIVE (only the lumpy mega-week coincidence).
- "10/10 seeds" = GBM-subsample stability, NOT robustness (bear 7/10, 2.2x swing).
- Reproducibility: unseeded-GBM bug fixed (seed=0); stale v6/pullback JSONs regenerated; 6 siblings seeded.
=> V5 FROZEN (hash aa5f7422, seed=0) for PAPER-FORWARD ONLY. Pre-registered gate + 2% cap + kill-switch:
   runs/engines/STRAT_TIERS/DEPLOY_RUNBOOK.md. Honest deployable = breadth-gated momentum (de-risk); timing magnitude unsettled.
<!-- ============================================================================ -->
## AUDIT (Claude, 2026-06-29; /audit + refutation wp5tv37do, all 6 findings CONFIRMED) -> de-risking-sleeve-only
Hardened the verdict. (1) The breadth-ALONE sleeve (the claimed deployable) FAILS robustness too: NW p=0.21, p05=-55%,
top-2 wks=121%. (2) De-concentration sweep (top8/16/30/all): return-alpha lumpy + NOT robust at EVERY K -> lumpiness is
structural. (3) ONLY robust value = BEAR/CHOP-CONDITIONAL drawdown reduction (sleeve bear maxDD -30% vs BTC-BH -50%; but
OOS-2024 BULL sleeve -36% WORSE than BTC -24% -> DD-insurance is bear-only). NO robust return-alpha exists within
LO-spot/u100/1d. Refinements: "Pwf-alone -7%" is a degenerate no-breadth baseline (Pwf|breadth adds +113pp in-sample =
lumpy 2-wk interaction); survivorship bias -> backtests overstated; fixed cap-renorm confound + share guards + 6 sibling
seeds. ONE-LINE: did NOT get SOTA results; honest deployable = bear/chop drawdown-insurance sleeve, full V5 paper-trade-only.
<!-- ============================================================================ -->
## MA FAMILY robustness AUDIT (Claude, 2026-06-29; user q "are the trend-riders robust?"; refutation wlirg1ui9 all-confirmed)
Ran the battery on the ACTUAL product (SMA/EMA/WMA/VIDYA/KAMA + CONV, breadth-gated inv-vol cap33 14d/7d rolling). Verdict:
ROBUST AS DE-RISKED BETA, NOT AS RETURN-ALPHA.
- NO significant alpha vs BTC-BH: OOS-2024 CONV +65% (vs BTC +40%) but NW p=0.12, p05=-17%, top-2 wks=86%, 0/6 types clear
  even the uncorrected 0.05 bar (best EMA 0.109). ONE-WEEK-FRAGILE: drop top week -> below BTC; 2yr compound -> NEGATIVE w/o top-2 wks.
- BEAR DD-halving IS robust + MECHANICAL: 48/52 cash weeks flat, sat out Feb-Apr 2025 crash via causally-lagged breadth gate
  (no look-ahead; leaky run_cell cross-grid path bypassed). maxDD -33% vs BTC -50%.
- BULL DD-match is partly path-luck (NOT mechanical); SMA/WMA actually WORSE than BTC in bull. Deployed-week win ~50% = coin-flip
  (gate cuts SIZE, doesn't pick winners); ~62-71% zero-yield cash.
Fixes: vol pct_change fill_method=None (was 'pad'); deployed-win-rate added; per-week-list naive-compound overstates ~13pp (use
daily stream). Artifact: runs/engines/STRAT_TIERS/strat_sota_ma_family_robust.{py,json}. DEPLOY for risk-adjusted profile, NEVER alpha.
## QUANT referee (exposure-matched, ab037947) -> CORRECTED: MA family IS an EFFICIENT de-risked-beta sleeve
First cut (CAPITAL-match e*underlying) said "DD-inefficient / just lower exposure" -- REFUTED: capital-match penalizes ANY
gated book (97%-when-on timer vs diluted constant-e hold). Exposure-INVARIANT Calmar: sleeve 2.76 vs BTC 1.69/EW 0.93 = wins
2/2 bull, 1/2 bear; vol+downside-matched DD favor sleeve 4/4; vs the actual EW-u100 universe the sleeve DOMINATES every axis
both windows (Sharpe 1.82 vs 1.08 / -0.36 vs -0.65; maxDD -24 vs -42 / -33 vs -75). Sharpe edge POSITIVE but UNDERPOWERED
(cash 62-72% wks; "not sig"=no power not no effect); cost+survivorship bias AGAINST sleeve. Residual cautions: lower exposure
does most of the absolute DD-LEVEL; bull outperformance lumpy/one-week-fragile (unchanged). Artifact: strat_quant_exposure_matched.{py,json}.
<!-- ============================================================================ -->
