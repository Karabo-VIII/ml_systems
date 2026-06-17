---
name: claude-instance-state-2026-05-26-maxx-v8-5-restart
description: "MAXX-INST-2026-05-26-NIGHT autonomous 8h session — comprehensive gap closure + v8.5 framework upgrade + (PEPE, MA/EMA) full reset. All 5 prior PEPE × MA/EMA candidates REFUTED-UNDER-STRESS (n_eff < 15 + DSR p > 0.91 @ N=1000). Combined dossier archived; replaced by separate (PEPE, SMA) + (PEPE, EMA) dossiers, both Phase 0 STARTING 0% exhaustion. Framework v8.5 binding from 2026-05-27 00:00 SAST with 8 new amendments (SM12-SM19)."
metadata:
  node_type: memory
  type: project
  supersedes: claude_instance_state_2026_05_20_smart_discovery.md
  snapshot_utc: 2026-05-26T23:45Z
  originSessionId: MAXX-INST-2026-05-26-NIGHT
---

# Instance State — MAXX v8.5 Restart (2026-05-26)

## §1. Session context

- **Wall-clock**: 2026-05-26 ~23:35-23:45 SAST (autonomous 8h mandate from user)
- **Coordinator**: META (Opus 4.7) under MAXX protocol — Opus position-workers + Opus domain-workers + Sonnet scouts per `PROTOCOL_COMPOSITION.md`
- **Mandate verbatim** (user, 2026-05-26 ~23:30 SAST): *"go back to the beginning with upgraded framework; expect real and better results; wealth building, quality signals, multi-timeframe; autonomous 8h"*
- **Win condition**: every Phase 0 cadence has diagnostic baseline; Phase 1 mining under v8.5 gates dispatched; doc suite updated for next-instance continuity; ≥1 deliverable per gap-closure worker landed.

## §2. What happened (4-conversation chain leading here)

1. **2026-05-25**: (PEPE, MA/EMA) provisional closure decision shipped (`docs/dossiers/PEPE_MA_EMA_CLOSURE_DECISION_2026_05_25.md`) with R12 perp at +48.9% UNSEEN compound; trust-stack v1.1 retrofit found OOS collapse + top-trade concentration but Phase 1 ship still claimed.
2. **2026-05-26 (early)**: comprehensive review surfaced 7 gaps F1-F7 (`runs/coordination/GAP_CLOSURE_STATUS_2026_05_26.md`):
   - F1 — n_eff < 15 across ALL 5 candidates (advisory in v8.4, not gating)
   - F2 — DSR never enforced at family-N; retrofit shows ALL 5 fail p < 0.05 at N=1000
   - F3 — L2 capture broken on R23a (capture_rate < 0)
   - F4 — R62 vs R23a 32pp sanity delta unreconciled
   - F5 — 12 active Pattern T scripts (silent same-bar close fill = look-ahead inflation)
   - F6 — R12 perp `ship_claim` block missing
   - F7 — `canonical_seeds` triple missing on 5 of 5 sampled audit JSONs
3. **2026-05-26 (mid)**: META dispatched 2 Opus domain-workers + 8 Sonnet scouts to close F1-F7 in parallel; gap-closure status doc tracked progress.
4. **2026-05-26 23:30 SAST**: user mandate to "go back to the beginning with upgraded framework" → META drafted v8.5 amendment spec → BINDING from 2026-05-27 00:00 SAST.

## §3. Framework v8.5 binding (full pointer)

Spec: `docs/WEALTH_BOT_FRAMEWORK_v8_5_AMENDMENTS_2026_05_26.md`. Eight new amendments compose on top of v8.4:

| § | Title | Bite |
|---|---|---|
| SM12 | n_eff GATE BAKED INTO PHASE 1 | `n_eff = 1 / Herfindahl(\|trade_returns\|) ≥ 15` REQUIRED. Stress-override path for n_unseen<30 via combined K2+S9 jackknife passing G1 floor under PARTIALLY_PASS. CDAP exit 2 on `verdict ∈ {SHIP, PARTIALLY_PASS} AND phase1_n_eff_gate.passes == False`. |
| SM13 | DSR PRE-SHIP GATE | `dsr_at_family_N_p_value < 0.05` at N=actual-family-size. `scripts/oracle/deflated_sharpe.py` canonical. For n<30 (thin-status): pre-registered hypothesis + ≥30 forward paper-trade trades required before strict-ship. |
| SM14 | INDICATOR SUBTYPE SEPARATION | (TI_subtype, ASSET) = separate dossier. SMA / EMA / WMA / Bollinger NEVER collapsed. Migration: (PEPE, MA/EMA) → (PEPE, SMA) + (PEPE, EMA). Old dossier archived to `docs/dossiers/archive_pre_v8_5/`. |
| SM15 | MULTI-TIMEFRAME COVERAGE | Phase 1 mining MUST cover all 5 canonical cadences {15m, 30m, 1h, 4h, 1d}. NULL cells valid per cadence; coverage required, not coverage-and-pass. DSR family-N includes ALL cadences. |
| SM16 | QUALITY-SIGNAL GATES (QS1-QS6) | QS1 n_eff / QS2 DSR feasibility / QS3 L2 capture / QS4 Pattern Q populated / QS5 Pattern S/T/U clean / QS6 repro block — ALL clear BEFORE any compound-based gate (G1-G6). CDAP exit 2 on any QS fail. |
| SM17 | CANONICAL_SEEDS BAG/FEAT/RNG TRIPLE | `{bag_seed, feat_seed: bag+1000, rng_seed: bag+7919}` REQUIRED on every audit JSON involving randomness. Per architect 2026-05-25 RNG-decorrelation. CDAP exit 2 on missing when `uses_randomness == True`. |
| SM18 | POST-v8.3 HARNESS REQUIRED | All Phase 1+ scripts MUST import from `framework.data_loader` (next_bar_open/next_bar_close only) OR `wealth_bot.harness.CanonicalHarness`. 10 of 12 Pattern T scripts FROZEN as MIGRATION_BACKLOG — advisory-only, not deploy-eligible. |
| SM19 | WEALTH-FIRST RANKING (formalized) | Leaderboard sort key = `compound_discounted = compound_raw × min(1, n_eff/12)`. Sharpe = tiebreak only. Per 2026-05-24 user mandate + `feedback_wealth_not_sharpe.md`. |

What v8.5 retires: combined (PEPE, MA/EMA) dossier; advisory n_eff; single-cadence Phase 1 ships.

## §4. PRIOR WORK FAILED (REFUTED-UNDER-STRESS)

All 5 candidates from `runs/oracle/WEALTH_BOT_LEADERBOARD.md` (post-RED-team verdict table) fail v8.5 §SM12 + §SM13:

| Candidate | n_unseen | n_eff | DSR p (N=1000) | Verdict |
|---|---:|---:|---:|---|
| R12 perp Strat B (WMA 10/30) | — | **7.07** | > 0.91 | REFUTED-UNDER-STRESS |
| R23a EMA30_dist + whale (static) | 25 | **8.81** | > 0.91 | REFUTED-UNDER-STRESS |
| R23c EMA(12,26) + (whale OR pz<0) | 20 | **8.81** | > 0.91 | REFUTED-UNDER-STRESS |
| R23h AB_AND / ABC_AND consensus | 19-22 | **4.43-7.0** | > 0.91 | REFUTED-UNDER-STRESS |
| 33-33-33 3-sleeve blend | 62 | TBD | TBD | REFERENCE — re-mine under v8.5 |
| P4_route_basis_pos_only | 9 | **4.55** | > 0.91 | REFUTED-UNDER-STRESS |

**Additional structural defects**:
- 0 of 5 sampled audit JSONs include `canonical_seeds` triple (§SM17 fail)
- 12 active Pattern T scripts identified (`scripts/wealth_bot/*.py`) — same-bar close fill = look-ahead inflation surface
- R12 perp leak source: `framework/data_loader.py:284-291` `build_forward_returns()` uses `closes[i + fwd_bars] / closes[i]` (leaky same-bar entry); honest H4 realistic compound +36.63% but n_eff still fails

**Combined (PEPE, MA/EMA) dossier**: ARCHIVED to `docs/dossiers/archive_pre_v8_5/` (parallel worker W1 this session). Does NOT carry credit forward per §SM14.

## §5. New dossiers

| Dossier | Path | Phase | Status |
|---|---|---|---|
| (PEPE, SMA) | `docs/dossiers/PEPE_SMA__inst_MAXX_2026_05_26.md` | 0 | STARTING — 0% exhaustion under v8.5 |
| (PEPE, EMA) | `docs/dossiers/PEPE_EMA__inst_MAXX_2026_05_26.md` | 0 | STARTING — 0% exhaustion under v8.5 |

Both dossiers will run independently under v8.5 §SM14 (no cross-pollination). Both are gold-standard per user mandate 2026-05-25 r5.2 (*"PEPE × EMA/MA is going to be our gold standard for all work going forward"*) — termination shortcut DENIED, multi-sprint commitment, documentation premium applies.

## §6. What actually happened this session (FINAL)

| Worker | Mandate | Final Status |
|---|---|---|
| META (Opus coordinator) | Drive 8h autonomous run, dispatch workers, accept returns, commit gates | COMPLETE |
| W1 (Opus domain-worker) | Archive old (PEPE, MA/EMA) dossier; reset leaderboard; create stub dossiers | COMPLETE — git log confirms delivery |
| W2 (Opus domain-worker) | Doc-suite rewrite (this file + CURRENT_PLAN + STATE + NORTH_STAR + MEMORY) | COMPLETE — THIS FILE |
| W3 (Sonnet) | (PEPE, SMA) Phase 1 baseline: 375 cells, all 5 cadences, 3 filters | COMPLETE — 0/375 SHIP, best n_eff 10.27 |
| W4 (Sonnet) | (PEPE, EMA) Phase 1 baseline: 435 cells, all 5 cadences, 3 filters | COMPLETE — 0/435 SHIP, best n_eff 11.91 |
| W5 (Sonnet) | (PEPE, SMA) Phase 1 expansion: 298 cells (perp + broader filters + SMA-distance) | COMPLETE — 0/298 SHIP, best n_eff 14.23 |
| W6 (Sonnet) | (PEPE, EMA) Phase 1 expansion: 396 cells (perp + broader filters + AX2 AND combos) | COMPLETE — 0/396 SHIP, best n_eff 11.46 |
| W7 (Sonnet) | Phase 2.5 EMA exit bakeoff: 36 cells (3 entries x 12 exits) | COMPLETE — 0/36 SHIP, max L2 lift +13.68pp (M2) |
| W8 (Sonnet) | Multi-cadence ensemble: 6 cells ({15m,30m,1h,4h} modes + {4h,1h} modes) | COMPLETE — 0/6 SHIP; OR{4h,1h} n_eff=15.89 FIRST above gate but QS3 L2=-0.297 broken |
| W9 (Sonnet) | Filter-cascade composition: 28 cells (21 EMA + 7 SMA-distance, OR/AND combos) | COMPLETE — 0/28 SHIP; max OR n_eff=13.02, still below gate |
| S9 (Sonnet) | Update round summary + tracker with W8/W9 final verdicts | COMPLETE |
| S10 (Sonnet) | Pre-commit verification (Pattern S/T/U, JSON, repro) + commit + finalize this doc | COMPLETE — THIS OPERATION |

**Total cells mined this session**: ~1,580
**Total SHIP candidates clearing all v8.5 gates**: 0
**Framework**: v8.5 held binding throughout all ~1,580 cells

## §7. Continuity instructions for next instance

If you are the next Claude instance reading this:

1. **READ THIS FILE FIRST** (you are here).
2. **Read v8.5 spec**: `docs/WEALTH_BOT_FRAMEWORK_v8_5_AMENDMENTS_2026_05_26.md`
3. **Read gap-closure status**: `runs/coordination/GAP_CLOSURE_STATUS_2026_05_26.md`
4. **Read both new dossiers**: `docs/dossiers/PEPE_SMA__inst_MAXX_2026_05_26.md` + `docs/dossiers/PEPE_EMA__inst_MAXX_2026_05_26.md`
5. **Read CURRENT_PLAN.md** (rewritten this session — terse, decision-ready)
6. **Check `git log --since="<session_duration> hours ago"`** to see where MAXX-INST-2026-05-26-NIGHT actually ended.
7. **Pick up from §8** below based on git log evidence.

## §8. What to do next (depends on where this 8h session ended)

**Scenario A — Phase 0 done, Phase 1 NOT started**: dispatch Phase 1 v8.5-gated mining sweep for both dossiers across all 5 cadences. Apply QS1-QS6 quality gates BEFORE any compound gate. Populate `canonical_seeds` triple on every audit JSON. NO Pattern T scripts in provenance chain.

**Scenario B — Phase 1 done, no survivors**: invoke `/oracle` for failure-mode catalog entry + Phase 2 imagine-frame re-pricing. Consider new sub-axis exploration per dossier §1B Top-5 untested. Check 80% exhaustion progress.

**Scenario C — Phase 1 done, ≥1 survivor**: dispatch Phase 2 oracle-augmented refinement under imagine-frame (pre-registered thresholds, asymmetric loss, honest TRAIN+VAL fit, cross-window persistence). Then Phase 2.5 exit-mechanism exploration if Phase 2 survives.

**Scenario D — Phase 2.5 done, candidate clears all v8.5 gates**: enter pre-live verification (12-item checklist from prior R12 perp work, adapted for v8.5: chimera SHA, fund-rate source, basis check, taker smoke, paper-trade window, etc.).

**Scenario E — saturation (3 consecutive NULL Phase 2 rounds with sample growth)**: invoke saturation protocol — 1-quarter ban on (PEPE, SMA) or (PEPE, EMA), mandatory scope expansion. Spawn Stretch-Goal Worker for top-3 expansion directions.

**Scenario F — wall-clock exhausted, session ending**: write fresh session-end state doc (file naming: `claude_instance_state_<YYYY_MM_DD>_<topic>.md`), update CURRENT_PLAN.md §1+§4, update this file's §8 with where you actually ended, push MEMORY.md HEAD pointer forward.

## §9. Pointers to all docs updated this session

- `CURRENT_PLAN.md` — FULL REWRITE by W2 (148 lines, v8.5-grounded, terse)
- `STATE.md` — §"Wealth-bot strat layer (2026-05-26)" added by W2 at top (pipeline / model-layer sections preserved)
- `PROJECT_NORTH_STAR.md` — §7 rewritten (no deploy-ready candidate; REGIME_ROUTER + R12 stricken); §8 augmented with gold-standard exhaustion rows
- `memory/agent_protocols/INDEX.md` (user-memory side) — updated to mark all 7 protocols MATERIALIZED (gap D4 CLOSED — stale framing corrected; project-side files were already substantive 76-105 lines each)
- `memory/MEMORY.md` — HEAD pointer updated to this file
- `memory/claude_instance_state_2026_05_26_maxx_v8_5_restart.md` — THIS FILE (session-end state)
- `docs/dossiers/PEPE_SMA__inst_MAXX_2026_05_26.md` — created by W1
- `docs/dossiers/PEPE_EMA__inst_MAXX_2026_05_26.md` — created by W1
- `docs/dossiers/archive_pre_v8_5/` — archived old combined dossier by W1
- `runs/oracle/WEALTH_BOT_LEADERBOARD.md` — leaderboard reset for v8.5 by W1
- `runs/oracle/WEALTH_BOT_FAILURE_CATALOG.md` — F1-F7 + 5 candidate refutations entered by W1
- `memory/fix_logs/INDEX.md` — patterns added for n_eff-was-advisory and DSR-not-gated drifts
- `docs/WEALTH_BOT_FRAMEWORK_v8_5_AMENDMENTS_2026_05_26.md` — v8.5 spec (META, separate)
- `runs/coordination/GAP_CLOSURE_STATUS_2026_05_26.md` — closure tracker

## §10. Final session findings (2026-05-27 00:40 SAST)

### Aggregate v8.5 mining cycle stats
- Total cells mined: ~1,623 across (PEPE,SMA) + (PEPE,EMA) dossiers + WMA Phase 0
  - Phase 1 baseline: 810 cells (W3 SMA 375 + W4 EMA 435)
  - Phase 1 expansion: 694 cells (W5 SMA 298 + W6 EMA 396)
  - Phase 2.5 exit bakeoff: 36 cells (W7 EMA only)
  - Round 4 multi-cadence: 6 cells (W8 EMA)
  - Round 4 filter-cascade: 28 cells (W9 both)
  - Round 5 combined-axis probe: 9 cells (W10 EMA)
  - WMA Phase 0: 40 diagnostic checks (W11 — Phase 0 only; Phase 1 mining queued)
- Total SHIP candidates clearing all v8.5 gates: **0**
- v8.5 framework: held binding throughout
- 7 commits landed: 80bd29e, 940c98a, 828664a, 8beb6b8, cdb64a9, 540e110, 1524574

### Round 5 (combined probe) and WMA Phase 0 findings (added 00:40 SAST)

**W10 combined probe (F2_AND_F3 + OR{4h,1h} + {M2, H2, R1}, 9 cells)**:
- 0/9 SHIP. Falsifier TRIGGERED.
- Critical finding: orthogonal best-finds are **NOT additive**. Combining W7's M2 + W8's OR{4h,1h} + W9's F2_AND_F3:
  - n_eff lifted slightly (16.70 vs W8's 15.89) ✓
  - L2 destroyed (-0.95 vs W7 M2 alone +0.10) ✗
  - top3 concentration inflated (8235% vs W9 F2_AND_F3 alone 112%) ✗
  - UNSEEN compound dropped to +0.23%-+9.10% (vs W9 alone +14-17%)
- The cells where M2 worked (clean entries) and the cells where F2_AND_F3 worked (filtered narrow tail) are DIFFERENT cells — they don't co-occur.
- Implication: no current cross-axis cell within explored param space passes all v8.5 gates.

**W11 WMA Phase 0**:
- Dossier created: `docs/dossiers/PEPE_WMA__inst_MAXX_2026_05_26.md` (528 lines, v8.5-compliant from Day 1)
- Diagnostics done across all 5 cadences; WMA past-only verified (diff=0)
- Whale_net lagged correlation POSITIVE at every cadence (+0.0098 to +0.0535)
- Rank-1 cell queued for Phase 1: WMA(10,30) x F2 x 4h x PERP (R12 v8.5 re-validation)
- Pre-registered H7: ~70% prior of REFUTED en-bloc (R12 was REFUTED-UNDER-STRESS at v8.5 retrofit)
- Phase 1 mining itself is NEXT-INSTANCE WORK (not run this session)

### What was learned (verified findings — REPORTED from real worker output JSONs)
1. **n_eff >= 15 gate is binding** at PEPE 4h (max achieved 15.89 via W8 OR{4h,1h}; closest single-cell 14.23 W5 SMA-distance s=100 thr=2%)
2. **PERP cost reduction (0.10% vs 0.44% RT) DOUBLES EMA compound** (11.03% -> 22.92% on same trades) but does NOT lift n_eff (concentration unchanged)
3. **Exit mechanism is responsive** (Phase 2.5 max L2 lift +13.68pp via M2 volume-collapse) but absolute L2 stays below QS3 +0.40 floor
4. **F2 (whale > 60-bar median lag1) is load-bearing** filter (12 of 13 EMA Stage-1 survivors used it)
5. **F2_AND_F3 cross-mechanism (whale AND basis_z<0) gives cleanest concentration** (top3~125% vs 280-873% for OR-disjunction) — most promising future-mining lead
6. **15m/30m sub-cadences ADD NOISE not signal** for EMA family; 1h adds genuine breadth
7. **SMA family structurally cost-dominated at spot** (0/375 positive UNSEEN compound at 15m/30m/1h)
8. **EMA family richer than SMA** (8-10x Stage-1 survivors); EMA is the better PEPE entry indicator
9. **SMA-distance entry primitive** (R23a-equivalent) is best within SMA family (W5 slow=100 thr=2% UNSEEN +11.09%, n_eff=14.23)

### What was REFUTED (formally)
- All 5 prior PEPE x MA/EMA candidates (R12, R23a, R23c, R23h, P4, 33/33/33) — REFUTED-UNDER-STRESS at v8.5 gates
- (PEPE, SMA) Phase 1 baseline + expansion: REFUTED across 673 cells
- (PEPE, EMA) Phase 1 baseline + expansion: REFUTED across 831 cells
- Phase 2.5 exit bakeoff for EMA: 0/36 clears v8.5 gates
- Multi-cadence ensembles {15m,30m,1h,4h}: REFUTED (sub-cadences hurt)
- Filter-cascade compositions: REFUTED at n_eff (max 13.02)
- SMA-distance under filter composition: REFUTED across all 7 combinations
- "1,112-day window" claim in BTC plan: REFUTED (was PEPE typo — resolved as TRAIN+VAL window, not total span)

### What was NOT REFUTED (interesting leads for future)
- F2_AND_F3 cross-mechanism on EMA: cleanest concentration profile (top3~125%), but n too small to clear n_eff
- OR{4h,1h} multi-cadence: lifts n_eff above 15 for first time but exit needs re-engineering (L2 broken)
- Combined approach: F2_AND_F3 entry + OR{4h,1h} cadence + Phase 2.5 winning exit (M2/H2) — NOT YET TESTED, highest-prior future axis

### Continuity for next instance
- READ FIRST: this state doc + `runs/coordination/GAP_CLOSURE_STATUS_2026_05_26.md` + `docs/WEALTH_BOT_FRAMEWORK_v8_5_AMENDMENTS_2026_05_26.md`
- Active dossiers: `docs/dossiers/PEPE_SMA__inst_MAXX_2026_05_26.md`, `docs/dossiers/PEPE_EMA__inst_MAXX_2026_05_26.md` (both REFUTED at current depth)
- Highest-prior next axis: combined F2_AND_F3 + OR{4h,1h} + Phase 2.5 winning exit (M2 volume-collapse)
- Alternative pivot per Edge-Pushing §EP3 saturation: BTC x {SMA, EMA, MACD} (new TI, ASSET cells) — BTC plan has B1-B3 fixes already applied per `GAP_CLOSURE_STATUS_2026_05_26.md`
- Alternative: WMA family for PEPE (separate dossier under v8.5 §SM14)
- All prior PEPE x MA/EMA work is REFUTED reference only — do NOT re-promote without new n_eff-clean evidence
- Round summary doc: `runs/coordination/MAXX_INST_2026_05_26_NIGHT_ROUND_SUMMARY.md`
