# Current Plan (Living Document)

**Last updated**: 2026-05-28 ~01:10 SAST by AUTONOMOUS_trader_2026_05_28_0032 (5h trader autonomous mandate, IN-FLIGHT)

## 🔴 LIVE AUTONOMOUS RUN (2026-05-28)

- **Run**: `runs/audit/AUTONOMOUS_trader_2026_05_28_0032/REPORT.md` — full ledger of 13 hypotheses H1-H13 + cross-asset transfer H14.
- **Best candidate (updated)**: **H18 QUINTUPLE-SLEEVE** = H12 + R11 (3-MA SMA(9,25,99) signal-flip from prior commit e3eb6aa) at 20% each (total 100% bankroll). **+33.5% combined OOS+UNSEEN**, OOS +25.7%, UNSEEN **+6.3% (both positive)**, n_eff 70, max DD -14.0%, all QS1-QS6 pass with margin, cost-robust to 100bps. **DSR p = 0.0473 at N=1 (PASSES v8.5 SHIP GATE)** — first design in dossier to do so. Deploy spec: `docs/dossiers/PEPE_H18_QUINT_SLEEVE_DEPLOY_SPEC_2026_05_28.md`. Live monitor: `scripts/wealth_bot/h18_live_signal.py`.
- **Prior H12**: 4-sleeve (no R11), +21.2%, DSR p=0.13. Superseded by H18.
- **User-facing menu** (REPORT §8): (A) Ship H12 paper-trade 30d zero-risk, (B) U1 pre-2023 chimera backfill, (C) U2 rotate to BONK (only other memecoin where H12 spec is profitable, +43%), (D) accept "good enough" + monitor.
- **Deploy spec**: `docs/dossiers/PEPE_MA_DISTANCE_Z_H12_PAPER_TRADE_SPEC_2026_05_28.md`
- **3-MA blind spot** (user Q): answered empirically in H5 — 3-MA stack (9/25/99) produces −42.8% on PEPE (vs 2-MA distance-z +21.2%). 3-MA buys late; 2-MA distance-z catches the catalyst extension.

---

**Previous last-update**: 2026-05-26 23:45 SAST by INST-MAXX-2026-05-26-NIGHT (autonomous 8h mandate)

> The next instance reads this FIRST at session start. Replaces relying on memory recall for cross-session continuity. All entries below reflect the v8.5 framework reset on 2026-05-26 — earlier project state has been archived; this is the live truth.

## 🔴 READ FIRST AFTER COMPACTION (in order)

1. **[memory/claude_instance_state_2026_05_26_maxx_v8_5_restart.md](memory/claude_instance_state_2026_05_26_maxx_v8_5_restart.md)** — most recent HEAD instance state (this session)
2. **[docs/WEALTH_BOT_FRAMEWORK_v8_5_AMENDMENTS_2026_05_26.md](docs/WEALTH_BOT_FRAMEWORK_v8_5_AMENDMENTS_2026_05_26.md)** — v8.5 binding spec (n_eff gate, DSR gate, indicator-subtype separation, multi-timeframe mandate, quality-signal gates, canonical_seeds triple, post-v8.3 harness, wealth-first ranking)
3. **[runs/coordination/GAP_CLOSURE_STATUS_2026_05_26.md](runs/coordination/GAP_CLOSURE_STATUS_2026_05_26.md)** — F1-F7 gap state from the comprehensive review that triggered the v8.5 reset
4. **[PROJECT_NORTH_STAR.md](PROJECT_NORTH_STAR.md)** — mission + ROI target + objective function (compound, not Sharpe)
5. **[CLAUDE.md](CLAUDE.md)** — invariants
6. **[docs/WEALTH_BOT_DEVELOPMENT_FRAMEWORK.md](docs/WEALTH_BOT_DEVELOPMENT_FRAMEWORK.md)** — full framework (v8.5 amendments compose on top)

## §1. Active dossiers

| Dossier | Phase | Status | Path |
|---|---|---|---|
| (PEPE, SMA) | Phase 0 | STARTING under v8.5 | `docs/dossiers/PEPE_SMA__inst_MAXX_2026_05_26.md` |
| (PEPE, EMA) | Phase 0 | STARTING under v8.5 | `docs/dossiers/PEPE_EMA__inst_MAXX_2026_05_26.md` |

Both at **0% exhaustion** under v8.5 gates. Prior (PEPE, MA/EMA) work is REFERENCE ONLY — does not count as credit toward exhaustion (per §SM14 indicator-subtype separation).

## §2. Framework state

**Binding**: **v8.5** (effective 2026-05-27 00:00 SAST) per `docs/WEALTH_BOT_FRAMEWORK_v8_5_AMENDMENTS_2026_05_26.md`. Key new gates:

- **§SM12** — n_eff ≥ 15 BAKED INTO Phase 1 (was advisory in v8.4)
- **§SM13** — DSR p < 0.05 at family-N as ship gate (canonical `scripts/oracle/deflated_sharpe.py`)
- **§SM14** — Indicator subtypes (SMA / EMA / WMA / Bollinger) are SEPARATE dossiers, never collapsed
- **§SM15** — Phase 1 mining MUST cover all five canonical cadences {15m, 30m, 1h, 4h, 1d}
- **§SM16** — Quality-signal gates (QS1-QS6) clear BEFORE compound gates (G1-G6)
- **§SM17** — `canonical_seeds: {bag_seed, feat_seed, rng_seed}` REQUIRED on any audit JSON involving randomness
- **§SM18** — Phase 1+ scripts MUST import from `framework.data_loader` or `wealth_bot.harness.CanonicalHarness`. 10 pre-v8.3 Pattern T scripts FROZEN as advisory-only.
- **§SM19** — Cross-bot leaderboard ranks by `compound_discounted = compound_raw × min(1, n_eff/12)`. Sharpe = tiebreak only.

Composes with: v8.4 entry/exit separation (SM11), Phase 2.5 exit-mechanism exploration, all earlier wealth-bot framework SR1-SR1.2 dossier-format mandates.

## §3. What was refuted 2026-05-26

Comprehensive review (`GAP_CLOSURE_STATUS_2026_05_26.md` §1 F1+F2) found **ALL 5 prior PEPE × MA/EMA leaderboard candidates** fail v8.5 gates:

| Candidate | n_unseen | n_eff | n_eff gate | DSR p @ N=1000 | DSR gate | Verdict |
|---|---:|---:|---|---:|---|---|
| R12 perp Strat B | — | **7.07** | FAIL (<15) | > 0.91 | FAIL | REFUTED-UNDER-STRESS |
| R23a EMA30_dist+whale (static) | 25 | **8.81** | FAIL (<15) | > 0.91 | FAIL | REFUTED-UNDER-STRESS |
| R23c EMA(12,26)+(whale OR pz<0) | 20 | **8.81** | FAIL (<15) | > 0.91 | FAIL | REFUTED-UNDER-STRESS |
| R23h AB_AND / ABC_AND consensus | 19-22 | **4.43-7.0** | FAIL | > 0.91 | FAIL | REFUTED-UNDER-STRESS |
| 33-33-33 3-sleeve blend | 62 | TBD | TBD | — | TBD | REFERENCE — re-mine under v8.5 |
| P4_route_basis_pos_only | 9 | **4.55** | FAIL | > 0.91 | FAIL | REFUTED-UNDER-STRESS |

Additional structural defects: 0 of 5 sampled audit JSONs include `canonical_seeds` triple; 12 active Pattern T scripts identified (same-bar close fill = look-ahead inflation surface).

**Old combined (PEPE, MA/EMA) dossier**: archived to `docs/dossiers/archive_pre_v8_5/` by parallel worker (W1) this session. Historical reference only — does NOT carry credit forward.

## §4. Current active mining (this session)

**Mandate**: autonomous 8h MAXX coordinator session, user-directed *"go back to beginning with upgraded framework; expect real and better results; wealth building, quality signals, multi-timeframe"* (2026-05-26 ~23:30 SAST).

**Plan**:

| Phase | Action | Cadences | Status |
|---|---|---|---|
| Phase 0 | Per-cadence diagnostics for (PEPE, SMA) + (PEPE, EMA) | {15m, 30m, 1h, 4h, 1d} | Dispatched turn-1 |
| Phase 1 | Mining under v8.5 gates (n_eff ≥ 15, DSR @ N=family, QS1-QS6 clear) | All 5 cadences MANDATED | Next turn |
| Phase 2 | Oracle-augmented refinement under imagine-frame (if any candidate survives Phase 1) | per surviving cadence | Conditional |
| Phase 2.5 | Exit-mechanism exploration per v8.4 (SM11 entry/exit separation) | per surviving cadence | Conditional |

Workers spawned this session: META (Opus 4.7 coordinator) + W1 (archive/leaderboard reset) + W2 (this — doc-suite rewrite) + Phase-0 sonnet scouts per cadence.

## §5. Next-session prompt (read FIRST)

When a new Claude instance starts:

1. **Wall-clock anchor**: run `date` and `git log --since="6 hours ago"`. Read any commits.
2. **Read this file** (CURRENT_PLAN.md) for §1 active dossiers + §4 in-flight work.
3. **Read `memory/claude_instance_state_2026_05_26_maxx_v8_5_restart.md`** — most recent HEAD; tells you where this session ended.
4. **Read `docs/WEALTH_BOT_FRAMEWORK_v8_5_AMENDMENTS_2026_05_26.md`** — v8.5 gates that bind any new work.
5. **Read `runs/coordination/GAP_CLOSURE_STATUS_2026_05_26.md`** — what was refuted and why.
6. **Read both new dossiers**: `docs/dossiers/PEPE_SMA__inst_MAXX_2026_05_26.md` + `docs/dossiers/PEPE_EMA__inst_MAXX_2026_05_26.md`.
7. **Then**: continue from §8 of the instance state doc (depends on where this 8h session ended).

## §6. Stricken / superseded entries

The following references in prior CURRENT_PLAN versions are STRICKEN (do not act on):

- All R32-era F4/F5/F6/F7 sprint references (alt-bar exhaust, MA specialist factory, P8 listing momentum) — >>40 R-rounds stale; if needed, re-derive from current state.
- R31 math fixes, R28 pipeline integrity, R30 specialist factory — historical; CURRENT_PLAN no longer carries pre-v8.5 round logs (see git history).
- "REGIME_ROUTER_STRICT_LO_SETUP60 +20.25% COMP" — superseded; was prior project-state framing, not part of the v8.5 wealth-bot stack.
- "Track 1-4 in-flight work tracks" from 2026-05-15 plan — all dead; v8.5 reset replaces with the §4 in-flight plan above.
- "Active running processes" table from 2026-05-15 — all DEAD per the May 15 audit; new processes are session-bound to this 8h MAXX run.

## §7. Update protocol for this file

At session end, every Claude instance MUST:

1. Update "Last updated" timestamp and session ID
2. Update §1 active dossiers (add new, mark closed)
3. Update §4 in-flight work (move completed to a §4.5 "Completed this session")
4. Update §5 next-session prompt if continuity instructions change
5. Add to §3 if a new refutation lands

If this file > 300 lines, archive older sessions to `docs/plan_archive/`.

---

**Provenance for this rewrite**: built by W2 (Opus domain-worker) under META MAXX-INST-2026-05-26-NIGHT autonomous 8h mandate. Replaces 2026-05-15 plan (11 days stale, ~238 lines, R31-era). Source docs cited inline: v8.5 amendment spec, GAP_CLOSURE_STATUS, WEALTH_BOT_LEADERBOARD (numbers in §3 verbatim from leaderboard post-RED-team verdict table).
