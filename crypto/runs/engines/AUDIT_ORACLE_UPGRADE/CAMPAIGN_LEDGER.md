# AUDIT + ORACLE + UPGRADE CAMPAIGN — LIVE LEDGER (Claude; state-saving)

## MANDATE (user, 2026-06-28)
1. Work the PROPOSED ITEM: the long-only ORACLE + ceiling gap (best-achievable cash-vs-deploy in the ALLOWED space,
   NOT a perfect predictor; reuse src/oracle/). Decompose the gap DNA = where the realizable engine leaves money.
2. VALIDATE + AUDIT all of Kiro's V2-V15 work THOROUGHLY (adversarial). Foundation before upgrading.
3. UPGRADE the profitability capture: ~1.5%/wk (15% cap, OOS-2024) is a FLOOR; the user relaxed the 3d/1d ROI goals.
   Push capture ABOVE the floor within LONG-ONLY (cap/concentration is the dial: 15->1.4, 50->2.1, EW->3.4). The
   oracle gap tells WHERE the headroom is.

## SEQUENCING: AUDIT (foundation) + ORACLE (ceiling) in parallel -> then UPGRADE (close the gap).
## BINDING: long-only spot lev=1 (no market-neutral even as framing); 2025 SEALED; rank by WEALTH; RWYB; no-headline-compound.

## THE MAKE-OR-BREAK CHECK (do FIRST): the V1 1d-regime-gate -> V2 4h-engine ALIGNMENT.
Does the 4h engine read the CLOSED 1d regime bar (causal) or the in-progress/same-day one (look-ahead leak)?
This is the EXACT bug Claude found in V1.1 (align_htf_on start-label leak). If leaked, the deployable +1.4-1.6%/wk
shifts. Falsifier: strict-lag the 1d regime by one full 1d bar; does the edge survive?

## WAVE LOG (newest last)
- 00:30 SAST — Campaign opened. Recon done (Kiro builders src/strat/ma_*.py; deployable ma_wealth_driver; oracle
  infra src/oracle/). Launching Wave 1: LeakCheck -> (Audit || Oracle) -> Synthesize.
- 00:38 SAST — LEAKCHECK DONE. VERDICT: **LEAKED** (same-day 1d-regime leak; the V1.1 bug class, MODEST impact).
  MECHANISM (src/strat/ma_slice_engine.py:166,192-197): `btc_regime` builds the 1d-BTC-SMA100 hysteresis series on
  CLOSE-time bars floored to day-start, then broadcasts onto the 4h/1d engine grid via
  `norm=dates.normalize(); ra=regime.reindex(idx).ffill().reindex(norm)`. An intraday 4h bar on day D normalizes to
  day-D 00:00 and reads regime[D] = the day-D DAILY-CLOSE regime, unknown until ~23:59 of day D = SAME-DAY LOOK-AHEAD.
  Traced on the 2024-06-18 transition: all six intraday 4h bars (incl. 04:00/08:00/12:00) assigned the regime derived
  from day-D's full close. SYSTEMIC: every V8-V15 deployable report (ma_prod_report_v2 -> the +1.67/+1.68 capped
  figures, ma_comprehensive_report, ma_convergence, ma_slice_roi, ma_config_breakdown, etc.) imports btc_regime and
  replicates the identical leaked broadcast.
  FALSIFIER (strict-lag the 1d regime one full bar; OOS-2024 PROD cost; runs/engines/AUDIT_ORACLE_UPGRADE/leakcheck.json):
  leak touches only **1.7% of OOS bars** (the ~6 transition days). Edge SURVIVES de-leak (all 9 cells still positive),
  loses **~0.56 pp/wk (~15% relative)**. SMA|4h +4.02->+3.41; VIDYA|1d +3.99->+3.52. NOT the V1.1 catastrophe
  (+879%->+109%) -- a uniform ~15% haircut. Deployable capped headline ~+1.4-1.6%/wk de-leaks to ~+1.2-1.4%/wk.
  ACTION FOR UPGRADE WAVE: fix in ma_slice_engine.btc_regime broadcast (shift the daily regime +1 bar before ffill);
  re-baseline CURRENT.md + registry headlines down ~15%; the qualitative North-Star/de-risk conclusions are unchanged.
- 00:58 SAST — ORACLE-GAP DONE (oracle_gap_study.py -> oracle_gap.json + ORACLE_GAP.md). u100 OOS-2024, PROD cost,
  DE-LEAKED regime. 3 levels (%/wk, slice-prof): L1 perfect-cash per-slice +12.0%/wk @94% (1d) / +11.8 @94% (4h)
  = the mandate-headline absolute long-only ceiling; L2 best-realizable causal (hindsight type, EW) +3.5/+3.4
  (==leakcheck lag1, cross-validated); L3 V2 frozen-EW de-leaked +3.29/+3.41 (capped deployable ~+1.2-1.4%/wk).
  GAP SPLIT: UPGRADE headroom V2->realizable ceiling = +10.8/+8.9 pp/wk (RECOVERABLE = concentration dial, the cap);
  un-capturable info limit realizable->perfect = +10.1/+31.4 pp/wk (needs DIRECTIONAL PREDICTION = refuted vein).
  GAP DNA (per-bar, sums exactly): late/early ~+0.0-0.2 (NON-issue), under-concentration +8.9-10.5 (biggest
  recoverable @1d), wrong-cash-calls +10-31 (un-capturable, dominates @4h). VERDICT: the only realizable upgrade is
  CONCENTRATION (raise the per-name cap), NOT entry/exit timing; the rest is the structural long-only ceiling
  (= North-Star 34% hit-rate). Anti-fab: never headline per-bar L1/L2b compound (rebal premium, 1e8% @4h meaningless).

## WAVE 2 — UPGRADE (the user's headline ask: push capture above the ~1.5%/wk floor, LONG-ONLY)
- 01:15 SAST — PART A: DE-LEAKED concentration frontier (wave2_concentration_frontier.py/.json,
  wave2_frontier_robust.py/.json). The oracle said "concentration (the cap) is the recoverable lever." TESTED IT
  with two skeptical gates and CAUGHT AN OVERFIT MIRAGE:
  * Fine cap sweep (1d OOS, de-leaked): cap15 +1.56 (Sh1.46,DD-29) -> cap33 +1.82 (Sh1.39,DD-33) -> cap50 +2.19
    (DD-35) -> EW +3.47 (Sh1.70,DD-32). cap-bind frac at cap15 = only 5% of live name-weights clipped: ALL the
    "headroom" is un-clipping a few high-conviction LOW-VOL names. The cap does NOT control drawdown (DD ~30% flat
    across all caps) -- the REGIME GATE does. The cap is single-name TAIL INSURANCE, not a DD dial.
  * CROSS-WINDOW (the kill): on VAL the cap->EW lift REVERSES. 1d VAL: cap15 +1.84 (Sh1.86) -> cap33 +1.96 (Sh1.86)
    -> EW +1.61 (Sh1.45, DD-49). EW is the WORST cell on VAL (lower ret, worst Sharpe, -49% DD). The "+3.5%/wk" is
    OOS-2024-ONLY luck. 4h same pattern.
  * FRAGILITY (the smoking gun): 69% of the 1d OOS cap15->EW lift is ONE name (ZENUSDT +55% extra gross), 88% top-3,
    98% top-5. EW = bet the book on ZEN; it mooned in 2024-H2 and lost on VAL.
  VERDICT: robust, cross-window-stable operating point = **cap ~33%**: +1.82%/wk OOS / +1.96%/wk VAL, de-leaked,
  no single-name dependence, modest DD cost (~+4pp). cap50/EW REJECTED (window-luck + single-name tail). The honest
  capture upgrade = leaked-cap15 ~+1.4-1.6%/wk -> DE-LEAKED cap33 ~+1.8-2.0%/wk, robust. The CURRENT.md cap dial's
  high end (50->2.1, EW->3.4) is an OOS mirage; the ROBUST dial tops at ~cap33. Within the MA-band mechanism cap33 is
  ~the robust ceiling; higher requires a DE-CORRELATED second mechanism (Part B: ADX|1d sleeve).
- 01:35 SAST — PART B: ADX|1d diversification blend (wave2_diversification.py/.json). MA-band@cap33 + ADX|1d
  ensemble (lb{24,26,28}x thr{0.42,0.45}), BOTH de-leaked + PROD + EW + same dates. corr(MA,ADX) = +0.44 VAL /
  +0.62 OOS. VAL blend looks great (MA50/ADX50 Sh2.96 DD-26.7) but that is VAL-FIT (ADX alone +5.45/wk VAL vs
  +2.60 OOS = the audit's VAL>>OOS shrinkage). On OOS (honest): MA50/ADX50 lifts Sharpe 1.39->1.68 but DEEPENS DD
  -32.8->-37.6; return-per-DD FLAT (0.056->0.059). VERDICT: ADX is NOT a robust frontier-lift -- the benefit is
  VAL-fit; on the honest window it buys Sharpe by accepting deeper drawdown. Stays PROVISIONAL/gated on 2025 unseal
  (matches the audit's "do not size"). MA-band@cap33 remains the robust deployable answer.

## WAVE 2 FINAL VERDICT (the upgrade answer)
The robust, defensible capture upgrade = **DE-LEAKED MA-band @ cap33 ~+1.8-2.0%/wk** (OOS +1.82 Sh1.39 DD-33;
VAL +1.96 Sh1.86 DD-43), cross-window-robust, no single-name dependence, above the 1.5%/wk floor. The leaked-cap15
headline (~+1.4-1.6%/wk) was both (a) leak-inflated ~15% and (b) cap-suppressed; de-leaking lowers it, loosening
the cap to 33 raises it net ABOVE the floor robustly. Beyond cap33 the MA mechanism's extra return is EITHER
single-name window-luck (EW = 69% ZENUSDT) OR a VAL-fit/gated second mechanism (ADX). REFINEMENT of the oracle: the
oracle's "+10.5pp/wk recoverable via concentration" conflated robust-recoverable (the small cap15->cap33 step) with
single-name-luck-recoverable (the rest). True robust recoverable headroom within the MA mechanism is ~cap33. Higher
robust capture needs crypto-native data (user-gated) -- the internal price-TI ceiling is reached.

## LEAK FIX DONE + COMMITTED (b6ea9336): causal_regime() helper DRY across ~26 gating consumers, regression 6/6 PASS,
## docs re-baselined. Verified by overseer (regression, parity, import, non-ASCII, append-only).

## WAVE 3 — "MORE VALUE FROM MA DEPLOYMENT" (user 2026-06-28: better deployment not new data; capture movers ahead of
## time via the lookback; dynamic/adaptive per-TI-type). INTERNAL price-TI only (user: no new data sources).
- RECON: oracle L2b (perfect hindsight top-3) = +14.0/wk @1d DD-24.6 / +12.3 @4h DD-23.2 => the recoverable headroom
  is NAME-SELECTION (mover capture), 4x return AND shallower DD. Type-selection thin (L2a spread SMA3.29..VIDYA3.52,
  ~+0.2pp; adaptive-MA refuted). Kiro V15 lookback ablation: confirmation-lb is a MINOR dial (lb14 1.67->lb28 1.92).
  => the one open lever = CAUSAL mover-SELECTION, not type/config/lookback.
- WAVE 3 (wave3_mover_selection.py/.json): 5 causal price scores (mom/accel/brk/dist/volx) x top-K{3,5,10}, tilt the
  regime+band-confirmed eligible set, vs random-K null + EW-all, BOTH windows. ONLY cross-sectional MOMENTUM beats
  random-p90 AND EW-all on BOTH VAL+OOS (dist/accel/volx = VAL-fit; brk = OOS-fit). mom K5: VAL +3.45 Sh1.85 / OOS
  +4.06 Sh1.75 (vs EW +1.74/+3.10). The theoretically-expected survivor (documented crypto factor).
- WAVE 3B HARDENING (wave3b_momentum_deployable.py/.json): 200-draw null + deployable soft-tilt + 4h.
  SIGNIFICANCE: mom top-5 = 100th pctile VAL, but only **94th OOS** (+4.06 vs null p95 +4.18 -> just shy of 0.05),
  degrades at K10/K15. DEPLOYABLE soft-tilt(cap20/33): clear risk-adj WIN on VAL (Sh1.39->1.92, DD-52->-47) but
  REVERTS to ~EW on OOS (lin 3.10->3.00, Sh down, DD worse). The OOS edge lives ONLY in hard top-5 (DD-52); on ret/DD
  even that is ~EW (0.078 vs 0.073). 4h: random-K bleeds (neg null), momentum stays positive (avoids the bleed).
  VERDICT: momentum mover-selection is REAL + cross-window-skilled but you are PAID ~FAIRLY for the concentration risk
  (= "magnitude is priced"). NOT a free de-risked upgrade. PROVISIONAL/regime-conditional (pays in high cross-sectional
  DISPERSION = VAL; adds little in uniform melt-up = OOS-2024). Gated on 2025 unseal, same bucket as ADX but better-
  principled. Confirmed deployable remains cap33 EW ~+1.8-2%/wk. Internal price-TI ceiling effectively REACHED; the
  one live thread = momentum-tilt + 2025 confirmation (+ maybe dispersion-conditioning, but that risks overfit on 2 windows).

## WAVE 3C/D/E -- "TRY DIFFERENT OPTIONS RIGOROUSLY" (user 2026-06-28). Both deployment axes, found 2 real levers.
- 3C OPTION A momentum risk-control (wave3c_*): the RIGHT control = inv-vol WITHIN hard top-5 (not the 3B soft-tilt,
  which diluted the signal). mom top5 inv-vol beats EW-all on return-per-drawdown on ALL 4 cells (1d VAL .068/OOS .086;
  4h VAL .039/OOS .069 vs EW .033/.073/.019/.062). RAISES return-efficiency but DEEPENS absolute DD (-48..-56). vol-
  target kills the edge. = robust risk-ADJUSTED win, not a DD win.
- 3D OPTION B gate timing (wave3d_*): cross-sectional BREADTH gate (frac of names>own SMA100, hyst .5/.4) vs single-BTC.
  At 4h = clean WIN both windows (DD -35->-27 OOS, -48->-30 VAL, flat return, better r/DD). At 1d = lower DD but trades
  a little return (~wash on r/DD OOS .063 vs .070). btc_sma50 worse; 'none' (ungated) worse (gate adds value). Breadth
  catches turns the BTC-only gate misses -> the DRAWDOWN win the de-risk mandate wants. Principled, low-overfit.
- 3E SYNTHESIS (wave3e_*): 2x2 gate{btc100,breadth} x selection{EW,mom-invvol}. COMBO breadth+mom-invvol roughly
  DOUBLES return AND lifts r/DD 1.5-2.5x on BOTH windows BOTH TFs (1d OOS +3.71 Sh1.75 r.103 vs base +2.01 r.070; 4h OOS
  +3.56 r.086 vs +1.70 r.048). COST: OOS DD deepens to ~-36..-41 (from -28..-35). Params NOT grid-searched (K5/cap33/
  breadth.5-.4 = defaults) -> overfit-limited. VERDICT: breadth gate = CONFIDENT DD-cut (adopt, esp 4h); momentum
  inv-vol = PROVISIONAL return-booster (94th-pct OOS, gated on 2025); COMBO = max risk-adj return at deeper DD.
  "At the ceiling" was WRONG -- real deployment value existed in HOW we deploy (gate) + WHICH names (momentum), exactly
  as the user predicted. Confirmed deployable spectrum: de-risk = breadth+EW@4h; max-capture = breadth+mom-invvol (gated).

## WAVE 3F + V3 LOCK -- "you're the expert, maximise WEALTH on a retail account" (user 2026-06-28).
- 3F wealth-robustness (wave3f_*): COMBO evaluated on COMPOUND (terminal wealth), param-robustness sweep
  K{3,5,8,10} x cap{.25,.33,.5} x breadth{(.5/.4),(.45/.35),(.55/.45)} = 36 combos. **36/36 beat baseline
  (btc100+EW33) compound on BOTH windows, BOTH TFs** -> NOT default-param-luck. 1d OOS compound: median +148%,
  range [+96,+185], ALL > baseline +68%. K8 = sweet spot (OOS +177% DD-29; DD IMPROVES with K: K3 -37 -> K8/10 -28/-29).
  cap IRRELEVANT (K8 inv-vol weights never reach 0.25). Momentum vs random UNDER breadth gate now SIGNIFICANT:
  97th pctile OOS-1d / 100th 4h (breadth gate sharpened it from 94th).
- V3 LOCKED (src/strat/deployable_v3.py): L1 breadth gate (0.5/0.4, 100d) + L2 cross-type band + L3 momentum top-8
  inv-vol. Verified: 1d OOS compound +170% (Sh1.94, DD-29) / VAL +615% (Sh3.45, DD-23); 4h OOS +173% / VAL +528%.
  ~2.5x baseline wealth at similar drawdown, ROBUST across params + both windows, momentum significant. Emits the
  latest target book (what to hold). TRADEOFF: 7dprof 22-25% (vs 34%) = LUMPIER (wealth-max trade). FINAL GATE: 2025.
  K = consistency/wealth dial (K10 or V3/baseline blend = more consistent at less wealth).

## WAVE 4 -- 2025-Q1 UNSEAL (user-authorized final OOS gate; data ends 2025-03-31 = all 2025 OOS that exists).
wave4_2025q1_unseal.py/.json. All MA-family candidates FROZEN, de-leaked, PROD, window 2025-01-01..04-01 (90 1d bars,
67/77 names). Q1-2025 = a BRUTAL alt quarter (BH EW-u100 -48.6%, BH BTC -11.8%; alts cratered in the Feb-Mar reversal).
RESULTS (1d, ranked by wealth): BH BTC -11.8% > conv breadth+EW -14.6% (deploy 19%) > btc+cap15 -16.4% > ...
V3 breadth+mom8 -21.7% > btc+cap33 (old baseline) -24.5% > btc+mom8 -34.8% > BH EW-u100 -48.6%.
THREE VERDICTS:
 1. BREADTH GATE = CONFIRMED. Best engine (-14.6%, deploy 19% vs btc-gate 51%/-24.5%); de-risk held on a real stress
    quarter, beat the old BTC gate ~10pp. Lever B validated across 3 windows (VAL/OOS-2024/OOS-2025Q1).
 2. MOMENTUM = REFUTED on fresh OOS. Adding it made everything WORSE (breadth+EW -14.6 -> +mom8 -21.7; btc+cap33 -24.5
    -> +mom8 -34.8). Textbook MOMENTUM CRASH: concentrated into Jan winners right as they reversed hardest Feb-Mar. The
    OOS-2024 +170%/2.5x was regime-specific (2024-H2 melt-up), exactly as the 94th-pctile borderline warning predicted.
    *** V3-AS-LOCKED MUST NOT BE DEPLOYED. *** Momentum is a regime-conditional bet that crashes in reversals.
 3. SOBERING: NOTHING beat BH BTC (-11.8%). The whole alt-selection apparatus underperformed BTC in this risk-off
    quarter; the gate->cash is the right floor but still lagged BTC. Retail wealth implication: BTC core hold matters.
REVISED DEPLOYABLE: breadth gate + conv band + EW (NO momentum) = the validated de-risk core. Momentum SHELVED.

## WAVE 5 -- PROD-GRADE BACKTEST: BTC-core + breadth-gated-alt-satellite, primary Q1-2025 (/orc, user 2026-06-28).
prod_backtest_q1_2025.py/.json. Elevated "prod grade" beyond cost: NEXT-OPEN fills (open-to-open, lag-1), de-leaked
breadth gate, POINT-IN-TIME listing universe, differential slippage (BTC core 0.0024 / alt 0.0028, +1.5x variant),
turnover costed. DATA CAVEAT: volume feed UNRELIABLE (BTC 1d ~26 "coins"=~$2M, off ~4 orders) -> NO dollar-ADV filter
built; u100 top-100 IS the liquidity screen (flagged). Structure: w_btc*BTC-core + (1-w_btc)*alt-satellite; BTC core
HODL or breadth-GATED; alt sleeve = conv band + breadth gate + EW cap33 (validated de-risk core, NO momentum).
RESULTS (compound, by window):
 - Q1-2025 (PRIMARY): BH alt-basket -50.5%, BH BTC -12.9%. GATED structures PRESERVED CAPITAL: gated-BTC core +1.5%
   (c2c) / +3.9% (o2o), 50/50-gated -6.8%/-1.7%, alt-satellite alone -14.6%/-7.3%. Verdict robust to BOTH fill models
   (magnitude execution-sensitive in this small window; qualitative win robust). The de-risk structure is VALIDATED
   PROD-GRADE on the fresh OOS: positive/flat where BTC AND alts both fell hard.
 - VAL: alt-basket +138%, alt-satellite +122%, core50-gated +106%, BH BTC +101%. (alt-bull -> alt-heavy wins.)
 - OOS24: alt-satellite +65% (> BH BTC +41% > alt-basket +33% -> gate adds value), core25-HODL +62%, core50-gated +45%.
3-WINDOW combined wealth: alt-satellite alone 3.39x (highest, but Q1 crash exposure) > core25-gated 3.18x > core50-gated
2.94x > BH BTC 2.48x > BH alt-basket 1.57x (Q1 crash destroys it). VERDICT: the BTC-core+alt-satellite structure WORKS
all-weather. The GATING (cash in risk-off) is the key protective lever (applied to both sleeves); the BTC/alt split is
the bull-capture (alt) vs crash-defense (gated-BTC) dial. Retail wealth+survivability recommendation: ~25-50% gated-BTC
core + breadth-gated-alt-satellite = positive/flat in EVERY window incl. the Q1 crash, shallow DD (-13 to -22%); pure
alt-satellite if maximizing wealth and accepting crash exposure. CAVEATS: 1 crash quarter (small sample); execution-
sensitive in Q1; volume-feed unreliable (liquidity = u100 screen only).

## WAVE 6 -- CORRECTION + PROD-GRADE MA FAMILY (user /orc 2026-06-28: "don't just trade BTC; the spec is the MA family").
CORRECTION: my Wave-5 "BTC-core" drift COLLAPSED the breadth surface (u100->BTC) + was OFF-SPEC. Deployable = the MA
FAMILY OF ENGINES on the FULL u100, canonical 14d lookback / 7d trading window, ROLLING, regime-gated. prod_ma_family_q1_2025.py.
PROD-GRADE (next-open fills, de-leaked, prod cost, PIT, turnover, u100 77 names):
 - VAL: MA family +118-130% (~= BH alt-basket +138).
 - OOS24: MA family +54-87% (KAMA|4h +87, EMA|1d +74, CONV +65-72) -- BEATS BH alt-basket +33 AND BH BTC +41.
 - Q1-2025 (crash): MA family PRESERVED CAPITAL -- CONV|1d breadth cap33 -7.3% (next-open)/-14.6% (close-close),
   KAMA|1d -5.1/-15.3 -- vs BH alt-basket -50.5%, BH BTC -12.9%. ROBUST = preserved capital vs basket (any fills);
   beat-BTC is EXECUTION-SENSITIVE (7-10pp o2o-vs-c2c swing in this gappy quarter) -> do NOT claim beat-BTC.
 - 1d is the deployable resolution (4h bled -16 to -21% in the Q1 chop). breadth gate > btc gate. cap33 ~= cap15.
VERDICT: the MA family is an all-weather capital-preservation product on the FULL universe -- captures most of the
bull (VAL), beats the basket in chop (OOS24), preserves capital in the crash (Q1). NO BTC core needed; BTC-core was a
solution to a non-problem. DEPLOYABLE = cross-type CONV book (or per-type ensemble) @ 1d, breadth gate, cap33, 7d/14d,
EW inv-vol, NO momentum. CAVEATS: 1 crash quarter; execution-sensitive in Q1; volume-feed unreliable (u100 = the screen).

## WAVE 7 -- PER-TYPE WIRING + PAPER-TRADE FORWARD on the REST of OOS (user /orc 2026-06-28).
SAFETY CHECK handled via MEMORY (user-endorsed, not gate-edit): added recall-based pre-recommendation safety check to
CLAUDE_CHARTER rule 6 (anti-collapse + long-only + wealth + no-headline + verify-at-canonical-spec) -- closes the
"broad-work-then-BTC-collapse" hole the gate is blind to. CORRECTED my earlier false claim: data does NOT end Q1-2025
(that was the WALL); real data runs to ~2026-05-28 -> 2025-Q2..2026-05 = 423 1d bars of GENUINELY-UNSEEN OOS (77/77 names).
WIRING: wealth_bot/bot paper trader is SINGLE-POSITION (non-overlapping, (TI,ASSET) framework) -> wrong arch for the
cross-sectional u100 portfolio (forcing it = the collapse just corrected). Built the PORTFOLIO paper-trade runner
(paper_trade_ma_family.py) reusing wealth_bot CONVENTIONS (next-open fills, prod cost, $-equity, journal/telemetry).
OFFICIAL per-type candidates (locked): u100, 1d, 14d/7d rolling, breadth-gated, cap33, EW inv-vol, NO momentum.
FORWARD RESULT (2025-Q2..2026-05, $10k start, prod-grade): KAMA -5.9% ($9409) > CONV -10.8% > SMA -12.5% > EMA -13.1%
> WMA -15.3% > VIDYA -16.7%  vs  BH alt-basket -36.7% ($6329) and BH BTC -13.7%. maxDD: family -22..-27% vs basket
-67% vs BTC -50%. The de-risk product WORKS over a long unseen window (a 2025 bull + a 2026 bear): preserved ~26pp vs
the basket + halved the drawdown; KAMA+CONV beat BTC too. QUARTERLY (CONV vs basket): Q2-25 -19.9 vs +2.3 (WHIPSAW --
the real weakness: lagged a rotational rising quarter), Q3-25 +11.4 vs +32.6 (captured some bull, lagged), Q4-25 -0.8
vs -43.9 (PRESERVED in the crash), Q1-26 0.0 vs -30.1 (PRESERVED), Q2-26 +0.8 vs +19.0 (lagged recovery).
VERDICT: the MA family is a genuine all-weather CAPITAL-PRESERVATION product on the full universe -- it does NOT make
money in a net-down cycle (long-only can't; basket -37%) but loses far less (-6 to -17%) at half the drawdown, and
preserves capital in crashes. Cost = lags bull/rotational quarters (Q2-25 -20% whipsaw). Deployable = CONV (type-
agnostic) or the per-type set; KAMA best on THIS window but type-selection is fragile (1 realization). CAVEATS:
one cycle; Q2-25 whipsaw is a real weakness; volume feed unreliable (u100 = the liquidity screen).
