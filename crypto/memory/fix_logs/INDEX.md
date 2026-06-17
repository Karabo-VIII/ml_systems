# Fix Log Index

**Last updated:** 2026-05-26 (MAXX gap-closure session; Patterns Q/R/S/T/U/v8.5 amendments)

## Per-Model Fix Logs
| Model | File | Last Updated |
|-------|------|-------------|
| V1.0 | `memory/fix_logs/v1_0.md` | 2026-03-18 |
| V1.1 | `memory/fix_logs/v1_1.md` | 2026-03-18 |
| V2.0 | `memory/fix_logs/v2_0.md` | 2026-03-18 |
| V3.0 | `memory/fix_logs/v3_0.md` | 2026-03-17 |
| V4.0 | `memory/fix_logs/v4_0.md` | 2026-03-18 |
| V5.0 | `memory/fix_logs/v5_0.md` | 2026-03-18 |
| V6.0 | `memory/fix_logs/v6_0.md` | 2026-03-17 |
| V7.0 | `memory/fix_logs/v7_0.md` | 2026-03-18 |
| V8.0 | `memory/fix_logs/v8_0.md` | 2026-03-18 |
| V9.0 | `memory/fix_logs/v9_0.md` | 2026-03-18 |
| V12.0 | `memory/fix_logs/v12_0.md` | 2026-04-08 (VIB fix) |
| V14.0 | `memory/fix_logs/v14_0.md` | 2026-04-08 (VIB fix) |

## NEW Cross-Cutting Pattern: Clean-Variant Information Bottleneck Stripped (CRITICAL, 2026-04-08)
- **What:** Clean refactor stripped the V1.0 RSSM categorical latent (the load-bearing anti-memorization mechanism) but kept the "ATME" name. ATME implementation `feat = h_seq * atme_mask` is a no-op channel-zeroing applied AFTER temporal encoding, not actual context erasure.
- **Empirical signature:** Cont IC1 reaches 0.20-0.30 in <10 epochs (5-10x V1 healthy), ShIC ≈ 0 (1000x worse than gate min), ratio < 0.001.
- **Confirmed in:** V3-clean f34 2026-04-08 (IC1=0.2729, ShIC=0.0002, ratio=0.0007)
- **Models at risk:** V3-clean, V12 (forward_train single-asset path), V14 (TwoHot path)
- **Models safe:** V6-clean (TimeShuffleDiscriminator), V11 (discriminator), V8 (RSSM), V9-clean (GRU implicit), V13 (VSN feature gating)
- **Fix (commit 9973ca0):** Variational Information Bottleneck (VIB). Project encoder output to (mu, logvar) of small latent z (z_dim << d_model), sample z~N(mu,σ), KL(z||N(0,I)) regularizer annealed via existing KL_ANNEAL_EPOCHS. The non-recurrent analog of RSSM compression pressure.
- **Prevention:** Any new "clean" variant that removes a recurrent latent MUST add an explicit information bottleneck. Audit: (1) does an explicit rate-limited latent exist? (2) does the return head read from it, or directly from the encoder?

## Cross-Cutting Bug Pattern R: Mechanism-Claim Empirically False (CRITICAL, 2026-05-25)
- **What:** A ship-candidate's claimed mechanism is contradicted by trade-level data. The aggregate compound looks structural; the verbal explanation of *why* is plausible but unverified against the actual filtered trade set. Reviewers and downstream artifacts (leaderboard, learnings, closure docs) inherit the false claim.
- **Concrete instance (2026-05-25):** INST-A's `P4_route_basis_pos_only` claimed *"filter strips top-tail-dependent trades; remaining 9 trades are robust to top-trade removal AND realistic execution costs."* RED-team auditor traced the trade list and found the filter KEPT the same top 3 trades as ABC_AND (+24.71% / +23.57% / +21.63%, early-Jan to mid-Feb 2026) and DROPPED 10 diversifying smaller trades — the OPPOSITE of the claim. The "+2.25% combined K=2+S9 stress pass" was driven by a single trade (+21.63%); K=3 → −5%.
- **Detection signal:** any commit-message / dossier / audit JSON line of shape *"filter strips X"* / *"mechanism: Y"* / *"X is the reason it works"*. When seen:
  1. Identify candidate's `per_trade_returns_sorted_desc`.
  2. Identify what filter KEPT vs DROPPED.
  3. Diff against prior unfiltered candidate's top-trade list.
  4. If same top trades: mechanism is selection, not robustness.
- **Fix (in-session, binding):**
  1. Pull `per_trade_returns_sorted_desc` from audit JSON.
  2. Compute `top_3_pct_of_compound`.
  3. If >70%: do NOT ship mechanism claim. Rewrite in leaderboard, learnings, closure decision.
  4. Mark `mechanism_falsifier_check.verified_by` in audit JSON.
- **Prevention (binding 2026-05-25 this commit):**
  - `src/wealth_bot/framework/claim_contract.py` enforces `mechanism_falsifier_check` field
  - `src/audit/check_wealth_bot_claims.py` halts commit (CDAP exit 2) when `top_3_pct > 70% AT n<30 AND verified_by=NOT_YET_VERIFIED`
  - Framework SR1.3 Mechanism-Verification Rule
  - Auditor brief item 9
  - TURN_END_CHECKLIST item 4
- **Related:** Pattern Q (top-trade concentration disclosure)

## Cross-Cutting Bug Pattern Q: Top-Trade Concentration Hidden (CRITICAL, 2026-05-25)
- **What:** A candidate reports strong UNSEEN compound (e.g. +60-80%) but compound is concentrated in 2-3 trades. Audit JSON does NOT disclose per-trade distribution. Reviewers see only aggregate. Strict gates (jackknife K=2, K=3) would collapse the candidate but are not run.
- **Concrete instance (2026-05-25):** Every (PEPE, MA/EMA) candidate in the session — 1-strat LGBM, EMA30_dist+whale, EMA(12,26)+OR, ABC_AND, 33/33/33 sleeves, P4_route_basis_pos_only — had top-3 trades carrying ≥95% of UNSEEN compound. Several promoted to ship-tier or near-ship-tier before jackknife revealed structural top-trade-dominance.
- **Detection signal:** audit JSON where `n_unseen < 30` AND UNSEEN compound ≥ +50% AND lacks `per_trade_returns_sorted_desc` OR `top_3_pct_of_compound` field.
- **Fix (in-session):**
  1. Sort per-trade returns desc.
  2. Compute `top_3_pct = 100 × (prod(1+r_top3)−1) / (prod(1+r_all)−1)`.
  3. If >70%: do NOT promote to SHIP without:
     - Jackknife K=2 + K=3 + K=5 compound report
     - Combined K=2 + S9 +0.88%/side stress compound report
     - Mechanism-falsifier check (Pattern R)
  4. Sample-size discipline: `min(baseline_compound, combined_K2_plus_S9_compound) ≥ ship_threshold`, NOT `baseline ≥ ship_threshold` alone.
- **Prevention (binding 2026-05-25 this commit):**
  - `src/wealth_bot/framework/claim_contract.py` REQUIRES `per_trade_returns_sorted_desc` + `top_3_pct_of_compound` + `jackknife` + `combined_K2_plus_S9_pct`
  - CDAP `check_wealth_bot_claims.py` (exit 2)
  - Sample-size discipline now applied to STRESSED compound (claim_contract.passes_stressed_gate)
  - Auditor brief items 8 + 10 + 11
- **Related:** Pattern R (mechanism-claim empirically false)

## Cross-Cutting Bug Pattern P: aggTrades scale + sort regressions (CRITICAL, 2026-05-13)
- **What:** Binance aggTrades raw format changed twice in 2025-Q3 and 2026-Q1:
  (1) timestamps switched from 13-digit ms to 16-digit us (microseconds);
  (2) rows arrive unsorted (~45% out-of-order on 2026-03-04 BTC). Bar builders
  passed timestamps through unmodified and assumed sorted input -> 2026 range
  bars produced 32M rows for BTC in 128 days (median tick_count = 2) instead
  of ~30/day at 0.5% threshold. Cumsum-based bars (DIB) less visibly broken
  but subtly wrong (cumsum order matters).
- **Empirical signature:** AltBarLoader validator: `timestamp_ms out of 13-digit
  ms range, min=1577836800594 max=1778283284707756`. F3 audit density check:
  6,644 bars/day BTC > 500 upper bound. Tick-to-tick price moves: 2.92% (2026
  unsorted) vs 0.0003% (2024 sorted).
- **Confirmed in:** All 4 alt-bar builders (dib/range/runs/adaptive_vol),
  output for 2025-Q3 onwards.
- **Files at risk (not yet patched):** 7 other aggTrades consumers in
  `src/pipeline/features/` and `src/pipeline/ingest/` -- impact chimera_v51
  for VAL/OOS/UNSEEN periods; TRAIN unaffected.
- **Fix:** `src/pipeline/bars/_aggtrades_utils.py::prepare_aggtrades(df, ts_col)`
  -- normalizes ts scale + sorts. Idempotent. All 4 bar builders patched.
  Detail log: `memory/fix_logs/pipeline_aggtrades_us_unsort_2026_05_13.md`.
- **Prevention:** Mandate `prepare_aggtrades()` call site immediately after
  any `pl.read_parquet(aggTrades_path)`. AltBarLoader / ChimeraLoader
  validators MUST reject ts outside [1.5e12, 2.0e12].

## Cross-Cutting Bug Pattern O: Backtest Aggregation Inflation (CRITICAL, 2026-04-19)
- **What:** When aggregating N dollar bars into one candle, using MAX(feature) for threshold conditions makes those conditions non-selective. MAX(z-scored VPIN) over 144 bars fires on 82-95% of candles because at least one of 144 bars almost always exceeds the threshold. The reported "conditional edge" is actually unconditional drift.
- **Empirical signature:** fire_rate > 50% on conditions that should be rare events (e.g., "VPIN spike" firing on 82% of candles is not a spike). Inflated edge that collapses when aggregation method is corrected.
- **Confirmed in:** sub_day_resolution_discovery.py (2026-04-18). DEGEN "net edge +3.64%" became 0% (zero viable combos) after switching from MAX to LAST aggregation. Documented in `docs/phase2_corrected_findings.md`.
- **Models/scripts at risk:** Any analysis that aggregates bar-level features into coarser candles and applies threshold conditions. Includes sub_day_speculator.py, cross_sectional_research.py, brain2_rotation.py, and any future multi-resolution backtester.
- **Fix:** Use LAST bar value (or MEAN) for threshold conditions, never MAX. Added Backtest Validity Gate to Layer 2 (LAYER2_UNCONSTRAINED.md) with mandatory selectivity check, sample size floor, cost realism, and statistical significance requirements.
- **Prevention:** Every backtest output must include fire_rate per condition. If fire_rate > 50%, the condition is measuring unconditional drift, not conditional alpha. Layer 2 Backtest Validity Gate is now BLOCKING for any analysis claiming trading edge.

## Cross-Cutting Bug Patterns

### DIRECT_RETURN_WEIGHT=3.0 (ALL versions)
- Affects: V1.0-V1.6, V2, V3, V4, V5, V6, V7, V8, V9 (ALL had 3.0 from creation)
- Pattern: DRW=3.0 creates Huber shortcut bypassing TwoHot bottleneck. Accelerates memorization.
- Fix: Set to 1.0 in all versions. Fixed 2026-03-17 (V3/V4/V6), 2026-03-18 (V1/V2/V5/V7/V8/V9).

### Steps-per-epoch Undertraining (ALL V2-V9)
- Affects: V2/V5/V7/V9 (300), V8 (400), V3/V6 (300), V4 variants (300-500)
- Pattern: 4-6x less data exposure per epoch. ShIC checks fire before model learns anything.
- Fix: Increased WM_STEPS_PER_EPOCH and DIVERSITY_STEPS_PER_EPOCH to 2000 in all versions.

### TwoHotSymlog Default Parameters [-5,5]
- Affects: V2, V5, V6, V8 components.py (all variants), V1.4 components.py
- Pattern: Class default params were [-5,5] (legacy voladj). If constructed without explicit args, produces wrong bins.
- Fix: Changed defaults to [-1,1]. V7/V9 checked — no issue or already correct.

### Missing strict=False on load_state_dict (ALL V2-V9)
- Affects: V2, V5, V7, V8, V9 (fixed 2026-03-18), V3/V6 (fixed 2026-03-17)
- Pattern: Without strict=False, architecture changes crash on checkpoint load.
- Fix: Added strict=False to both model and ema_model load_state_dict calls.

### shic_decline_count Not Persisted (ALL V2-V9)
- Affects: V2, V5, V7, V8, V9 (fixed 2026-03-18), V3/V6 (fixed 2026-03-17)
- Pattern: Without persistence, restarts reset counter = infinite ShIC declines allowed.
- Fix: Added to checkpoint save/load, expanded return to 6-tuple.

### Missing n_features + Collision Guard (ALL V2-V9)
- Affects: V2, V5, V7, V8, V9 (fixed 2026-03-18), V3/V4/V6 (fixed 2026-03-17)
- Pattern: Without collision guard, loading f13 checkpoint for f37 model silently corrupts.
- Fix: Save n_features in checkpoint, reject on mismatch in load_latest.

### ATME z_post Leakage (V3, V4, V5, V8)
- Affects: V4 (fixed 2026-03-17), V3 (fixed 2026-03-17), V5/V8 (partial mitigation 2026-03-18)
- Pattern: ATME zeros h_seq in feat_heads, but z_post = f(posterior(h_seq, obs_seq)) already carries temporal info. ATME is cosmetic.
- Fix (V3/V4): Obs-only posterior in ATME mode + h_seq.detach() in normal mode.
- V5/V8: TEMPORAL_CTX_DROP increased 0.15->0.40, SEQ_SHUFFLE_PROB 0.20->0.30. Full architectural fix deferred.

### Stale Argparse Choices (ALL V2-V9 variants)
- Affects: ~60 variant scripts across V2-V9 (train_adapter, train_ncl, train_snapshot, validate_world)
- Pattern: choices=[13,18] blocked f30/f37 training; validate default=22 not in choices.
- Fix: choices=[13,18,30,37], validate default=13.

### Temporal Memorization via Weak Regularization
- Affects: V4 (confirmed), potentially V3/V5/V9 (untested)
- Pattern: More expressive architectures (Mamba, WaveNet, MoE) need V1.0-equivalent regularization: batch=32, weight_decay=5e-2, dropout=0.15

### JEPA Shared obs_proj Bug (V6)
- Affects: V6 (all variants), potentially V2 (same JEPA architecture)
- Pattern: Target encoder branch used online obs_proj instead of EMA-updated copy.
- Fix: Added `target_obs_proj` with EMA update, freeze, and copy logic.

### NaN Recovery Optimizer Missing RevIN Params
- Affects: V1.1 (fixed 2026-03-18), V1.6 (fixed 2026-03-18)
- Pattern: NaN recovery recreates optimizer with only model.parameters(), dropping RevIN params.
- Fix: Include RevIN params in reconstructed optimizer.

### ACTIVE_HORIZONS [1,4] Bug (RESOLVED)
- Affected: V1.1 (Mar 13 training run)
- Pattern: ACTIVE_HORIZONS set to [1,4] zeroed h16/h64 loss, removing multi-scale regularization.
- Fix: Reverted to [1,4,16,64] across all versions.

### Pattern S — Close-only trail breach inflation (2026-05-25 R45/R46 + 2026-05-26 R51)
- Affects: `scripts/wealth_bot/r45_cde_iga_wide_depth.py` (E45_2/E45_3 chandelier), `scripts/oracle/r46_run_all_depth.py` (E46_1/E46_3 trailing_atr/ratchet_stop), `scripts/wealth_bot/r51_depth_deepening.py` (E51_1 atr_trail, MFE-profit-lock)
- Pattern: Trailing-stop / chandelier / MFE-lock exit policies use close-only breach detection (`closes[j] <= trail_level`) instead of intra-bar `lows[j] <= trail_level`. Even when intra-bar detection is added, the gap-down fill model must also include the `exit_price = trail_level if highs[j] >= trail_level else closes[j]` guard — otherwise the simulator fills at the trail price on bars that gapped down PAST the trail (impossible in live trading; price was below trail at open, must fill at OPEN or worse — closes[j] is the conservative proxy when OHLC is bar-aggregated).
- Symptom: TRAIN compound inflated 100-1000× in gap-prone bars; UNSEEN typically less affected (UNSEEN regimes tend to be lower-vol). Diagnostic: TRAIN/UNSEEN ratio > 8× should auto-flag for harness audit.
- Fix: canonical post-fix in `scripts/oracle/r46_run_all_depth.py:203-225 (trailing_atr) + :262-285 (ratchet_stop)`. Pattern S audited via Auditor 17 (R46 case) and Auditor 19 (R51 case).
- Impact pre-fix → post-fix:
  - R46 E46_1 trail-2xATR(7): TRAIN +2426.78% → +2.82% / UNSEEN +81.68% → +22.79% (SHIP_CANDIDATE → REFUTED)
  - R46 E46_3 ratchet-ATR2to1@3%: TRAIN +8263.67% → +20.68% / UNSEEN +126.52% → +34.44% (SHIP_CANDIDATE → REFUTED)
  - R51 E51_1 atr_trail-1.0x: VAL +1563.72% → -10.33% / UNSEEN +90.61% → +38.52% (SHIP_CANDIDATE → REFUTED)
- Sister patterns: R45 chandelier line 467 `max(lows[j], trail_level)` exceeds bar high when trail > high (different bug, same family). DO-NOT-COPY banner added 2026-05-25.

### Pattern T — Harness inheritance regression (R51 inherited R45/R46 PARTIAL-fix only)
- Affects: any new exit-policy script that copies from an INCOMPLETE-fix predecessor instead of the canonical post-fix `r46_run_all_depth.py`
- Pattern: R51 author copied the `lows[j] <= trail_level` intra-bar detection (correct) but forgot the `closes[j]` gap-down fallback (incomplete). Result: SHIP_CANDIDATE that wouldn't have been flagged with full post-fix harness.
- Fix: New exit-policy scripts MUST import the harness from `r46_run_all_depth.py` directly OR pass a line-by-line diff against r46 lines 203-285 in the commit body before review.
- Prevention: pre-commit hook (queued) to grep for `lows[j] <= trail_level` without subsequent `highs[j] >=` guard in the same function body.
- **2026-05-26 (R12 sim fix, commit `850d05a`)**: `scripts/wealth_bot/r12_instrument_variant.py` used `entry_p = closes[i]` (same-bar close fill); Q4 scored LEAKY. Post-fix: UNSEEN compound +48.90% LEAKY -> +39.65% realistic (-9.25pp inflation). G1 ship gate verdict flipped REFUTED.

### Pattern U -- Inline indicator computation (BANNED v8.3+)
- **Symptom**: Script hand-rolls `wma()`, `ema()`, `sma()`, `atr()`, or `df['sma'] = closes.rolling(N).mean()` without `import pandas_ta as ta`.
- **Risk**: Past-only correctness depends on author discipline; some inline implementations include same-bar reference (look-ahead-by-1-bar) or wrong-direction rolling windows.
- **Fix at API**: v8.3 framework SM10 binding -- use `pandas_ta` for all indicators, OR use `framework.data_loader.wma_past_only()` family. CanonicalHarness structurally bans inline.
- **Coverage as of 2026-05-26**: ~32 scripts in `scripts/wealth_bot/` have inline indicator definitions and have NOT been migrated. 10 of these also have active Pattern T (same-bar fill). See `src/wealth_bot/harness.py::MIGRATION_BACKLOG` for FREEZE-flagged list.
- **Provenance**: Pattern U codified by commit `3d69711` (2026-05-26, framework v8.3); enforced at API level via CanonicalHarness 833-line module.

### FM-PSEUDO-VB-FORWARD-CLOSE (look-ahead pattern, 2026-05-26 Auditor 22 MED-1)
- Affects: any "synthetic bar" implementation that groups underlying bars into volume / dollar / range buckets and then assigns the GROUP-AGGREGATE close (or other group-level statistic) BACK to all member bars
- Mechanism: at bar t, the function computes `last_idx = max index where group_ids==group_ids[t]`, which can be IN THE FUTURE relative to t. Assigning `closes[last_idx]` to bars BEFORE last_idx within the same group leaks the future close to those earlier bars. A subsequent `.shift(1)` is insufficient because the leaked value persists across the shift.
- Symptom: synthetic-bar-derived signals fire EARLIER than they would in live trading (the bar t signal already "knows" the group's final close)
- Fix: use a RUNNING within-group statistic. At bar t, the "current group representative close" must be closes[t] itself OR `cummax(closes within group up to t)` — never the group's terminal close.
- Reference fix: `scripts/wealth_bot/r54_depth_deepening.py:998-1020` (post-Auditor-22 fix, 2026-05-26 ~04:30 SAST)
- Impact: A54_2 was already REFUTED on G2 fail (TRAIN -26.34%), so NO ship-decision impact in current dossier. Defensive immunization against future copy-paste of the pattern.
- Sister patterns: synthetic Renko bricks (similar group-aggregate risk); imbalance bars (Lopez de Prado — already documented in dossiers/README.md as deferred-to-build).

### FM-ATR-TIGHT-REGIME (failure mode, not bug pattern; 3 data points 2026-05-25 → 2026-05-26)
- Mechanism: Tight ATR trailing stops (≤1.0× ATR) are INTRINSICALLY regime-fragile. The same mechanism that amplifies compounding in trending bull markets (rapid re-entries riding the trend) generates catastrophic whipsaw losses in sideways/declining markets (every minor pullback triggers stop + adverse re-entry). NOT a tunable parameter — the failure is structural.
- Data points:
  - C8 R39 E39_1 trailing-ATR-0.75x: TRAIN -17.7% (R49b REFUTED on G2)
  - R46 E46_1 trail-2xATR(7) post-Pattern-S-fix: TRAIN +2.82% / UNSEEN +22.79% (REFUTED)
  - R51 E51_1 atr_trail-1.0x post-Pattern-S-fix: TRAIN -30.99% / UNSEEN +38.52% (REFUTED)
- Implication: do NOT propose new tighter-ATR variants as a "fix" for the ATR-2.0x family. The family is exhausted. Future work should target ALTERNATIVE exit families (trailing-percent-of-MFE, time-stop-conditional-on-volatility, partial-scale-out tiers) rather than ATR-tightening.
- Catalog status: 3rd confirmation = MATURE failure mode; documented in dossier §E exit policy row + framework §F.8 lesson #8 (added).

## Audit Status
**All versions audited as of 2026-03-18.**
- V1.0-V1.6: Audited 2026-03-18
- V2 (all): Audited 2026-03-18
- V3 (all): Audited 2026-03-17
- V4 (all): Audited 2026-03-17/18
- V5 (all): Audited 2026-03-18
- V6 (all): Audited 2026-03-17
- V7 (all): Audited 2026-03-18
- V8 (all): Audited 2026-03-18
- V9 (all): Audited 2026-03-18
