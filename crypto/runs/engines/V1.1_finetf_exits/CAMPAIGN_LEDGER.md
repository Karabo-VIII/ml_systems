# V1.1 FINE-TF CAMPAIGN — LIVE LEDGER (state-saving; survives context loss / resume)

> This is the durable spine of the 4h autonomous fine-TF campaign. Any instance resuming (after a context
> summary, a restart, or a stall) READS THIS FIRST to recover the thread, then re-grounds the clock with a
> fresh `date` and checks the live workflow IDs below. Write-forward every LEARN as it happens (autonomous-runner §5).

## WALL-CLOCK (measure, never feel)
- VERIFIED start: **2026-06-21 02:27:02 SAST** (`date`).
- Budget: 4h → target end ~**06:27 SAST** (INFERRED = start + 4h).
- On every progress claim: fresh `date` − VERIFIED start; show the subtraction. Re-ground after any gap.

## THE MANDATE (user, 2026-06-21)
Make the OTHER timeframes work. Only 15m may be conceded — and even it "has some trends going for it".
- **Goal (acid test):** in a **7-day slice across timeframes, the engines must be PROFITABLE.**
- Allowed: **mechanical interventions** + an **EXIT-MECHANISM HARNESS (types of exits)** — explicitly authorized.
- **Rework the original 1d/4h engines** too (enhancements / resilience; exits likely not tried exhaustively).
- **Iron on 2021** (learn each engine's strengths/weaknesses). **DO NOT touch 2022+** (sealed OOS).
- This is the **V1.1** build (on top of V1 = `docs/engines/V1_ma_drawdown_insurance.md`, committed dd1c7b8f).

## SPLIT DISCIPLINE (BINDING)
2020 = DEV (select here). 2021 = IRON (out-of-2020-sample; allowed). **2022+ = SEALED — never load.**
Convention-B OOS/UNSEEN (post-2024-05-15) also sealed. Assert max-bar before every load.

## THE FINE-TF THESIS
Coarse TFs work; 2h→15m degrade. Failure is NOT missing signal (forensic shows 4h/2h capture movers) — it is
**whipsaw churn**: "exit = band-off" flickers → cost drag + missed re-entry. Levers: smarter EXITS (ride the move,
exit on a real stop not a band-flicker), entry de-bounce (hysteresis), higher-TF confirmation. All mechanical.
Prior daily exit-axis was a NULL (different regime); fine-TF whipsaw-reduction is genuinely untested.

## WAVE PLAN (build → run → learn → pivot)
- **Wave 1 (LIVE):** diagnose fine-TF loss (2020) → build `src/strat/exit_harness.py` (8 exit types + entry
  de-bounce + HTF-confirm) → sweep on 2020 DEV, select per (engine,TF) on TRAIN/VAL → adversarial verify.
  Engines {EMA,HMA,TEMA,KAMA} × TFs {2h,1h,30m,15m}. Metric = % of rolling 7d slices net-profitable.
- **Wave 2 (pending Wave-1 learn):** iron Wave-1 winners on **2021** (per-engine strengths/weaknesses) +
  retrofit best exits onto the 1d/4h originals (resilience) + expand to all 8 engines.
- **Wave 3 (if needed):** dynamic regime→exit switching; whatever Wave-2 learning surfaces.

## REUSE (no reinvention)
Band = `config_leaderboard.json` fixed-EW ensemble (never ranked[0]). Hold = `ma_movecatch_engine`. Wealth =
`ma_wealth_driver` ONLY (verified MtM/lag-1/maker0.0006/TRUE-HODL). Exit infra to extend: `ma_capture_exit_sweep.py`.

## LIVE WORKFLOWS (mid-flight intervention)
- `wf_e5401786-223` (task `w2vpra2x8`) — Wave 1. STATUS: **running** (launched 02:27 SAST).
  - Watch: `/workflows`. Output artifacts land in `runs/engines/V1.1_finetf_exits/` incrementally (salvageable on stall).
  - Intervene: TaskStop to kill; Workflow({scriptPath:…, resumeFromRunId:"wf_e5401786-223"}) to resume (cached prefix).
  - Script: `…/workflows/scripts/v1-1-finetf-exit-harness-wave1-wf_e5401786-223.js`.

## DECISION / LEARN LOG (newest last)
- 02:27 SAST — Campaign opened. V1 committed dd1c7b8f. Wave 1 launched. Out-of-fit honesty folded into V1 doc.
- 11:42 SAST (resumed after ~9h idle; workflow rate-limited 02:42, only ~15min real work) — Wave 1 PARTIAL:
  diagnose + exit_harness.py BUILT & VERIFIED (selftest 11/11; exit_fn hook non-destructive; on EMA|1h de-bounce
  lifts compound +885%->+1216%, entry_debounce improves maxDD -47%->-43% == churn-thesis confirmed). SWEEP (16
  cells) rate-limited (16-way LLM fan-out = the mistake). DIAGNOSIS: single fine-TF killer = round-trip COST FROM
  CHURN (cost/gross 0.8%->206% across 1d->15m; gross +100%+ at every TF; capture FLAT ~0.38). NOT participation.
- PIVOT: the sweep is DETERMINISTIC compute -> run it as ONE in-process script, never 16 agents. Also: run_wealth
  runs full Conv-B DEV (2020->2024-05) which TOUCHES 2022+ -> add a two-sided eval-window so 2022+ is NEVER loaded
  (user: 2020 select / 2021 iron / 2022+ SEALED). Metric = % of 7d slices net-profitable (the acid test).
- 11:53 SAST (in-process sweep) — TASK 1 DONE + GATED: ma_wealth_driver.run_wealth now takes eval_start/eval_end
  (two-sided window; defaults None,None == exact prior behavior, clamped <= DEV_END) + returns engine_equity +
  engine_net_stream (dated, JSON). HARD IDENTITY GATE PASSED: run_wealth(EMA,1d,7d) no-window == committed
  engine_EMA_clean.json EXACTLY (3609.82% / -37.5% / mtm_reconciled, 0.0000% err). 2022-SEALED proven: with
  eval_end=2021-01-01 the panel is 2020-01-03..2020-12-31 only (cp.index.max() < 2021-01-01 asserted in-code).
  TASK 2 LAUNCHED: scripts/sweep_finetf_exits.py (ONE process), 16 cells {EMA,HMA,TEMA,KAMA}x{2h,1h,30m,15m},
  curated exit x entry grid (10 exits incl. HTF x 4 entries incl. HTF), acid metric slice7d_profit_rate over 2020.
  EMA|1h smoke: baseline slice7d=0.50 -> band_off_confirm_M2+entry_debounce_N2 = 0.5577 (FIXED). HTF wired +
  works: htf_entry_gate on EMA|1h = slice7d 0.6923 / net +249% / maxDD -18% (strong churn-quality filter). Run
  incremental to runs/.../sweep_2020/<ENG>_<TF>.json + _summary.json (salvageable).
- 12:00 SAST — run_wealth eval-window VERIFIED (default identity EMA|1d maxDD -37.5% matches committed; 2020-eval
  range 2020-01-03..2020-12-31, 2022+ SEALED True). Smoke EMA|1h: baseline slice7d 0.50 -> 0.73 (chandelier_atr_3 +
  htf_entry_gate), maxDD -47%->-10% == FIXED. Dominant lever so far: htf_entry_gate (enter 1h only when 4h band on).
  Full 16-cell 2020 sweep launched (PID 1865, scripts/sweep_finetf_exits.py). ~30-60min.
- 13:20 SAST — 2020 SWEEP COMPLETE (16/16). RESULT (2020 DEV, in-sample): 2h/1h/30m ALL 4/4 FIXED; 15m HMA+TEMA
  FIXED, EMA only IMPROVED (net -15.7%), KAMA done. THE LEVER: htf_entry_gate (enter fine TF only when next-coarser
  TF band agrees) in 14/15 winners; trades/wk HEALTHY (HMA|30m 7.7, TEMA|15m 11.6) -> NOT under-participation.
  Standouts: TEMA|30m slice7d 0.94 net +700%, HMA|30m 0.90 maxDD -6.9%, TEMA|15m 0.85, HMA|1h 0.85.
  CAVEAT: 2020 is the SELECT set (in-sample). Real test = 2021 iron.
- 13:24 SAST — Wave 2 launched: scripts/iron_2021.py (PID 5041) freezes each 2020 winner, runs OOS on 2021
  [2021-01-01,2022-01-01) (2022+ SEALED), vs band_off baseline. Verdict per cell: HOLDS_OOS / PARTIAL_OOS / FAILS_OOS.
  5-min PushNotification ping cadence active (ScheduleWakeup 270s).
- 13:56 SAST — iron BUG: best_exit stored the LABEL not exit_type; 11/16 cells errored. FIX: reconstruct exact
  build_exit_fn kwargs from best_label via the canonical EXIT_COMBOS/ENTRY_COMBOS maps (+ compute tpw from the
  net-stream since run_wealth doesn't inject _trades_per_week). All 16 labels resolve. EARLY OOS (5 valid cells):
  ALL 5 HOLD_OOS -- TEMA|15m 0.85->0.98, HMA|15m 0.73->0.92, TEMA|1h 0.87->0.87, KAMA|2h 0.62->0.60, KAMA|15m
  0.42->0.66. Confirmed: EVERY winner uses htf_entry_gate (KAMA|2h htf_confirm_exit) -- HTF confirmation = the
  universal fine-TF fix; exit policy varies per cell. Iron relaunched (full 16) PID-bg, waiter bt30w3u69.
- 14:07 SAST — 2021 OOS IRON COMPLETE (16/16). RESULT: ALL 16 cells HOLDS_OOS (2h/1h/30m/15m each 4/4); most
  IMPROVE vs 2020 (TEMA|15m 0.85->0.98, HMA|30m 0.90->0.94, HMA|15m 0.73->0.92); every cell beats its 2021 band_off
  baseline (lift +0.02..+0.36pp); tpw healthy 0.5-5.5; maxDD -5..-15%. Even 15m holds (user said concede-able).
  SKEPTIC CAVEATS (must verify before claiming victory): (1) 2021 was a BULL year -> absolute slice7d flattered
  (baseline also 0.55-0.66); real signal = LIFT over baseline (modest in EMA|2h +0.04, KAMA|30m +0.02). (2) The fix
  IS htf_entry_gate = defer fine-TF entry to coarser-TF confirmation -> does the fine TF ADD value over the coarse
  engine alone, or has it just become the 4h engine? (3) net% (2640%, 5414%) = bull-ride, NOT headline. (4) leak:
  HTF-alignment causality + 2022-seal need adversarial check. (5) 2022 bear is SEALED -> cannot OOS-test the bear.
  NEXT: dispatch adversarial auditor on these vectors; then 1d/4h retrofit; then synthesize + commit V1.1.
- 14:20 SAST — RED-TEAM CAUGHT A LEAK (3rd would-be fabrication the operating model caught). htf_entry_gate had
  LOOK-AHEAD: align_htf_on searchsorted on coarser START-labels (floored), so a fine bar inside an UNCLOSED coarser
  window read its eventual state. De-leak collapses HMA|1h +879%->+109%, TEMA|30m +2641%->+88%. FIXED align_htf_on
  to align on coarser CLOSE-times (label+freq); selftest updated to causal; verified de-leaked HMA|1h = +116%/0.53
  matching auditor. HONEST VERDICT: V1.1 fine-TF is NOT a wealth edge -- de-leaked it underperforms always-long
  (slice7d) AND the coarse engine (net); only maxDD survives = drawdown-insurance, REDUNDANT with V1 coarse.
  V1 UNAFFECTED (never used HTF wiring). Per-cell exit tuning = noise (universal rule ~ tuned). Running full
  de-leaked 16-cell eval (htf vs always-long vs coarse) for the honest documented table -> then commit V1.1 honest.
- 14:49 SAST — HONEST de-leaked 16-cell table (deleaked_2021/): htf-gate beats always-long slice7d only 1/16,
  beats coarse engine net 0/16, halves maxDD 16/16. VERDICT: fine-TF exit-harness REFUTED as a wealth edge --
  DOMINATED by the coarse engine (0/16 net), loses to dumb bull-rider on slice7d (15/16); only drawdown-insurance
  survives (redundant with V1). EARNED NULL (full u100 surface x 4 fine TFs x 8 exits x 2 windows, de-leaked).
  Deployable scope STAYS 1d/4h (V1). Now running 1d/4h exit-retrofit (do exits help the ORIGINALS? expect null per
  D61) -> then commit: align_htf_on leak-fix + exit_harness + run_wealth eval-window + honest V1.1 doc + dead-list.
- 14:58 SAST — 1d/4h RETROFIT done: exits improve V1 baseline OOS only 3/8 (TEMA|1d, HMA|4h, TEMA|4h via
  band_off_confirm hysteresis de-bounce; rest flat-to-worse; KAMA hurt by price stops). Consistent with D61.
  VERDICT: no V1 upgrade; de-bounce is an OPTIONAL marginal tweak, not robust. V1 band_off STANDS.
- CAMPAIGN COMPLETE. V1.1 = earned NULL (D81) + a real align_htf_on leak FIX. Fine TFs NOT a wealth edge; deploy
  stays 1d/4h. Committing: leak-fix + exit_harness + run_wealth eval-window + drivers + V1.1 doc + dead-list + CURRENT.
