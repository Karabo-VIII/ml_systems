# Adaptive-MA Overseer Log (8h autonomous run, ends 2026-06-06 07:03)

## Check-in 1 --  -- LAUNCH
- Armed 8h autonomous envelope. Brief: docs/ADAPTIVE_MA_BRIEF_2026_06_05.md.
- Launched BOTH rigs --backend cli --budget 40 --parallel 2 --judges 1 --durable:
  - rig-E EXPERT (thread ama-expert, experts attached, lane=expert)
  - rig-P PLAIN  (thread ama-plain, generic, lane=plain)
- Isolation: per-thread checkpoint DBs (metaop_<thread>.db) committed 2c89150.
- HEALTH: both leases held, both PLANNED (frontier seeded), workers dispatching, ~15 claude.exe.
- Plan: ~20-min synthesis check-ins; review output dirs, commit good work, manage, fold learnings.

## Check-in 2 -- 23:18 -- FIRST RESULTS (honest)
- rig-E (expert) FAST: built adaptive_ma.py + run_u100.py; ran u100 backtest (69 assets). Features = rolling
  realized-vol + efficiency-ratio + 252d percentile -> 2D bucket -> MA-config map. Honest train/val/oos/unseen
  splits, taker 0.0024, exit=opposite_cross.
  - EARLY NULL for adaptation: on TRAIN adaptive UNDERperforms fixed (winrate .395 vs .447; per-trade exp 13.8%
    vs 24.2%; only 16/69 assets adaptive>fixed). BTC UNSEEN adaptive comp -16.56 (loses).
  - OVERSEER CAVEAT: compound figures (1000s) are INFLATED -> almost certainly full-capital-per-trade / no real
    sizing. Trust winrate + per-trade-expectancy + UNSEEN, NOT headline compound. (compound%=output-not-target.)
- rig-P (plain): cycle 1 done (adaptive_ma_plain.py; n1+n3 pass; +3 adjacent). Behind rig-E on output.
- Health: 14 claude.exe, both rigs active, no crashes. On-brief + honest.
- ACTIONS/NOTES: (1) note for synthesis: adaptation not yet earning keep vs fixed. (2) improvement to make later
  (won't disrupt running rigs): have reflect ALSO read learnings so mid-run overseer guidance is picked up.

## Check-in 3 -- ~23:20 -- AUDIT (look-ahead)
- Red-teamed rig-E adaptive_ma.py (206 lines): RIGOROUSLY CAUSAL -- every feature .shift(1), trailing past-only
  percentile self-norm, past-only sma/ema, per-bar prefix re-featurization. Look-ahead falsifier ADDRESSED for rig-E
  (auditor persona shows). Inflated compound = sizing (full-capital/trade), NOT leakage. Honest metrics stand:
  adaptation not yet > fixed.

## Check-in 4 -- 23:24 -- BOUNDED GOAL + PARALLEL RESEARCHER
- User r2: target = 2-5
## Check-in 4/5 -- 23:24-23:28 -- bounded goal committed; rig-E closed cycle-1
- Brief r2 (bounded 2-5pct/move, 4h, anti-overfit) committed. Parallel researcher (scout-strat) running.
- rig-E closed cycle-1 (~18min, thorough, 10 files). rig-P cycle-1 done (2 files). Both entering cycle-2.
- Auditing plain-rig code for look-ahead rigor (expert-vs-plain comparison). (log printf-% bug fixed -> heredoc.)

## Check-in 6 -- ~23:32 -- REDIRECT (evolution step) -- bidirectional loop converged
- Researcher (scout-strat) returned: 1d MA-cross REFUTED (0/69 beat random-entry firewall on held-out; UNSEEN
  per-trade exp -2.09pct adaptive / -2.50pct fixed, both NEG; all 6 fixed beat the adaptive switcher). Timing =
  noise/beta; adaptation adds trades not quality. Cost is NOT the killer -- entry timing is.
- PRESCRIPTION (committed to RESEARCHER_REPORT_1.md + injected into both learnings lanes): ER as a HARD GATE (not
  switcher), 4h cadence, ATR-trail exit (not opposite-cross), breakout-confirm, MINIMAL 3-DOF, full overfit audit.
- ACTIONS: committed 1d baseline (control); stopped both 1d rigs (checkpoints preserved); injected prescription;
  RELAUNCHED fresh rigs ama2-expert + ama2-plain on the bounded 4h ER-gated objective (budget 30, reuse infra).
- ~29 min elapsed: launch -> 1d baseline -> self-falsified -> researcher refuted -> redirected to 4h. Tight loop.

## Check-in 7 -- ~23:40 -- 3-LOOP ARCHITECTURE (user directive: meta gets its own loop)
- Made reflect READ the learnings lane (mid-run [META]/[OVERSEER] guidance now reaches running rigs). Committed.
- Isolation formalized (STRUCTURE.md): each loop its own folder (expert/ plain/ meta/), no main-file mingling,
  promote-to-src/strat integration path defined.
- LAUNCHED 3 loops: rig-E2 (expert builder), rig-P2 (plain builder), META loop (ama-meta) = my autonomous
  meta-instance that each cycle reads rig output, AUDITS overfit/look-ahead, RESEARCHES, formulates adjacent
  objectives, and WRITES [META] guidance to BOTH rig lanes + META_LOOP_LOG. I am now pure ORCHESTRATOR.
- My role: monitor all 3 @1-min, guide, pass info, decide pivots, commit verified work. Meta-work is the META loop.

## ~00:36 -- HUMAN INTERVENTION: full-grid correction
- User: "@4h - wrong. What about other timeframes, and chart type. Whole project in play, just MA is instrument."
- I had wrongly fixed cadence=4h. CORRECTED: explore the FULL grid -- cadence in {15m,30m,1h,4h,1d,dollar,range,
  dib,runs_volume,adaptive_vol} x u100 (all verified present in data/processed/chimera/). MA = the indicator only.
- Found pre-computed capturable_4h_catalog.parquet + capturable_win_catalog.parquet (likely oracle/capturable
  catalogs to reuse). Injected correction to sol+meta lanes; relaunched both loops on the full-grid objective.
- Added rigor guard: META must DSR/Holm-correct across the ~10 cadences (grid-scan = multiple-comparisons risk).
