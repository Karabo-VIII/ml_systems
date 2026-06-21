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
