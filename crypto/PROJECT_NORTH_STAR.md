# PROJECT NORTH STAR — v4 Crypto System

> **CANONICAL FRAMING DOC** — read this FIRST every session. Stop the loop of re-establishing the goal.

> Auto-memory binding: every future Claude instance should treat this as load-bearing context, paralleling CLAUDE.md (invariants) and STATE.md (current state).

---

## 1. MISSION (north star)

**Capture 1-10% daily crypto moves under NO LEVERAGE + LONG-ONLY + spot constraints, profitably and sustainably.**

Per `memory/project_mission_2026_05_13.md` (2026-05-13 user mandate):
- PREDICT / DETECT / EXTRACT mode — every component must fit one of three
- Deploy unit = % of daily ≥5% movers correctly pre-called AND profitably exited
- "Drift gate": if proposed work doesn't fit PREDICT/DETECT/EXTRACT, PAUSE

---

## 2. EMPIRICAL PREMISE (our data, not external)

Per `memory/move_frequency_2025_2026.md` + reverified 2026-05-17:
- **100% of days in 2025-2026 have ≥1 asset moving ≥5%** (on full u87)
- **Avg 20 assets/day move ≥5%**, 35 assets/day ≥3%, 6 assets/day ≥10%
- Regime-stable (bear/bull/chop/crash all ≥21% asset-days)
- DEGEN/meme tail (KAT/GUN/NEIRO/PENGU/ENA): 13-26% days ≥10%
- **Opportunity set is 365 days/yr**, NOT 80-150
- **Bottleneck = conviction-calling signal quality**, not opportunity scarcity

**Implication**: ROI targets of 0.5-1%/d under LO are not a market-availability problem. They're a SIGNAL-QUALITY problem.

---

## 3. ROI TARGET (concrete, not aspirational)

| Tier | Daily ROI (LO + spot + lev=1) | 4-month COMP | Status |
|---|---|---|---|
| Floor (deploy threshold) | +0.05%/d | +6% | Any blend must clear |
| Current best (empirical) | +0.16%/d | +20.25% | REGIME_ROUTER_STRICT_LO_SETUP60 |
| Stretch (next milestone) | +0.30%/d | +40% | Cycle 4 target |
| Headline | +0.50%/d | +75% | Sub-day or per-asset ML required |
| **Project Goal** | **+1-5%/d** | **+200-1500%** | Structural reframe required (per-asset ML, sub-day, etc.) |

The +1-5%/d goal is NOT "best-published-quant-fund-Sharpe-translated-to-daily-return". It's the user-mandated empirical target derived from §2's opportunity surface. The methodology choice is the open variable.

### 3.1 OBJECTIVE FUNCTION (canonical, 2026-05-24 user mandate)

> **"We're optimising for wealth, not sharpe or any of the other metrics.
> We want to build pure returns and pure ROI, nothing more, nothing less.
> The strategy itself has to be robust, and if it has good sharpe that's good,
> but we not building for that necessarily."** — user, 2026-05-24

Decision-precedence under this objective:

1. **Optimize**: held-out compound return (UNSEEN compound %, paper-trade-realistic final equity, $X → $Y outcome). All ranking, selection, and "winner" calls use this metric.
2. **Constrain (robust)**:
   - N-seed audit: 10/10 seeds positive on UNSEEN (or N-1/N with explicit justification)
   - Block-bootstrap p05 on the compound > 0 (so the lower bound is also wealth-creating)
   - Max DD < 30% (capital preservation; harder caps OK)
   - Multi-window consistency (per §11): walk-forward not over-fit
3. **Nice-to-have (not the goal)**: Sharpe, Sortino, Calmar. A bot with Sharpe 1.5 and compound +70% **wins** over a bot with Sharpe 2.0 and compound +50%, provided both pass the robustness constraints in (2).
4. **Implication for design choices**:
   - DO NOT recommend selectivity-for-Sharpe at the cost of held-out compound.
   - DO recommend selectivity when it improves BOTH compound and Sharpe.
   - When 2 bots are equal on compound, prefer the one with higher Sharpe and lower DD.

Provenance: this clarification resolved the 2026-05-24 INST-C session ambiguity where the multi-cadence GATED rule (Sharpe 1.92 / compound +50%) was initially proposed as the deploy candidate over the 4h SOTA (Sharpe 1.45 / compound +70%). Under the corrected objective, 4h SOTA wins on compound by +20pp and is the deploy candidate.

---

## 4. HARD CONSTRAINTS (never relax silently)

1. **NO LEVERAGE** (lev = 1.0 always; leveraged variants are INFO ONLY)
2. **LONG-ONLY** (spot; shorts are EXIT/BLOCK inhibits via STAYOUT pattern)
3. **v3 paper_trade_replay = CANONICAL TRUTH** (yaml claims inflate 7.6x; pre-fix v51 also inflated 5-8x; only v3 is honest)
4. **RWYB Layer-1**: every code change run-tested with real data before commit (per `memory/feedback_run_what_you_build.md`)
5. **LAYER-0 unconstrained mindset**: published benchmarks not our concern; unblock OWN limitations; setup identification > raw threshold; reconfig > rebuild (per `memory/feedback_unconstrained_2026_05_17.md`)

---

## 5. WHAT'S NOT OUR CONCERN

- "Best published quant-fund Sharpe is X" → ANCHOR-against, not anchor-for
- "Published HFT alpha caps at Y" → measure OUR data; our premise is empirical
- "Other crypto funds report Z" → irrelevant; we have our own dataset + measurement infrastructure

**The empirical premise (§2) is what binds us. External benchmarks do not.**

---

## 6. ARCHITECTURE LAYERS

```
LAYER-0 (mindset): UNCONSTRAINED (never accept ceiling as terminal)
LAYER-1 (process): RWYB (run-what-you-build; no silent failures)
LAYER-2 (correctness): RED TEAM after every impl; CDAP pre-commit; connector_integrity_crawler
LAYER-3 (canonical): v3 paper_trade_replay; v51 chimera; MakerCostModel
LAYER-4 (deploy): NO LEVERAGE + LONG-ONLY + spot
LAYER-5 (strategy): blends → sleeves → indicators → ML
LAYER-6 (data): chimera v51 (194 cols × 87 assets × multi-cadence)
LAYER-7 (pipeline): 100% closed/upgraded per user 2026-05-17
```

---

## 7. CURRENT DEPLOYABLE (as of 2026-05-26T23:45 SAST)

**NO deploy-ready candidate exists.**

As of the 2026-05-26 v8.5 framework reset, all prior deploy-candidate claims are superseded or REFUTED-UNDER-STRESS:

- ~~**REGIME_ROUTER_STRICT_LO_SETUP60** (+20.25% COMP Jan-Apr 2026, Sh +3.29)~~ — **SUPERSEDED** by the wealth-bot framework track (v8.5); this candidate was strat-layer / blend-era, predates the wealth-bot 2-phase framework, and was not re-priced under v8.5 quality-signal gates (QS1-QS6). It is NOT deploy-ready under the current framework. May be re-evaluated as a separate non-wealth-bot deploy track, but does not satisfy current ship contract.
- ~~**PEPE R12 perp Strat B** (+48.9% UNSEEN compound)~~ — **REFUTED-UNDER-STRESS** 2026-05-26 per `GAP_CLOSURE_STATUS_2026_05_26.md` §1 F1+F2. Fails v8.5 §SM12 (n_eff = 7.07 < 15) AND §SM13 (DSR p > 0.91 @ N=1000). Same-bar close-fill leak source identified in `framework/data_loader.py:284-291`; honest H4 realistic compound +36.63% but underlying n_eff still fails gate.
- ~~All other PEPE × MA/EMA candidates (R23a / R23c / R23h / 33-33-33 / P4)~~ — REFUTED-UNDER-STRESS or REFERENCE per `runs/oracle/WEALTH_BOT_LEADERBOARD.md` post-RED-team verdict table.

**(PEPE, SMA) + (PEPE, EMA) v8.5 mining is IN PROGRESS** under MAXX-INST-2026-05-26-NIGHT autonomous session. Both dossiers at Phase 0, 0% exhaustion. Next deploy candidate (if any) requires:

- All v8.5 quality-signal gates (QS1-QS6) clear BEFORE compound gates
- n_eff ≥ 15 AND DSR p < 0.05 at family-N
- Multi-cadence coverage {15m, 30m, 1h, 4h, 1d}
- Canonical seeds triple (bag/feat/rng) in audit JSON
- Post-v8.3 harness (no Pattern T scripts in provenance chain)

Per `docs/WEALTH_BOT_FRAMEWORK_v8_5_AMENDMENTS_2026_05_26.md` + `CURRENT_PLAN.md §4`.

---

## 8. OPEN STRUCTURAL QUESTIONS (each opens +X% lift potential)

| Question | Status | Lift potential |
|---|---|---|
| **Gold-standard dossier exhaustion: (PEPE, SMA)** | **0%** — Phase 0 STARTING under v8.5 (2026-05-26) | Gating: gold-standard must hit ≥80% (or all floors MET) before next-TI dossier opens per r5.1 |
| **Gold-standard dossier exhaustion: (PEPE, EMA)** | **0%** — Phase 0 STARTING under v8.5 (2026-05-26); separate from SMA per §SM14 indicator-subtype mandate | Same gating as above |
| Per-asset ML (DNA + regime + oracle context) | Not yet built | HIGH — universe-wide hides per-asset heterogeneity (TAO Sh 4.6, others 0) |
| Sub-day cadence native measurement | v8.5 §SM15 makes this MANDATORY for Phase 1; not yet executed under v8.5 | HIGH — required, not optional |
| Capacity-scale TAO composite pattern | Deferred until (PEPE, SMA) + (PEPE, EMA) reach ≥80% exhaustion | MEDIUM — Sh 2.56 already; just need multi-asset replication |
| Constraint creativity (inverse-tokens for spot-shorts) | Not yet explored | MEDIUM — captures short-leg alpha without violating LO |
| WM cohort retrain on clean v51 | 32-40 GPU-hrs queued | LOW-MED — V1.x at Trader-tier; could lift to Headline |
| V20 tick-level Performer/Hyena | Multi-month commit | HIGH but slow — Capacity-tier IC>0.20 target |

---

## 9. ROADMAP (rough phasing)

| Cycle | Focus | Status |
|---|---|---|
| 1 | Crawler upgrade + WM v51 migration | ✅ DONE |
| 2 | Path A discovery + STAYOUT + sweep harness | ✅ DONE |
| 3 | Reconfig stacking + TA_SML retrain + UNCOND L3 | ✅ DONE — new deploy +20.25% |
| **4** | **Per-asset ML + Sub-day cadence + capacity-scale TAO** | **NEXT** |
| 5 | WM cohort retrain + Track B sweep | GPU-bound |
| 6+ | V20 tick infrastructure | Long-term |

---

## 10. CROSS-REFERENCES (auto-loaded context)

- `CLAUDE.md` — invariants + lens
- `STATE.md` — current state (versions/paths/tables)
- `memory/feedback_unconstrained_2026_05_17.md` — LAYER-0 mindset
- `memory/feedback_run_what_you_build.md` — LAYER-1 RWYB
- `memory/project_mission_2026_05_13.md` — original mission doc
- `memory/move_frequency_2025_2026.md` — empirical premise
- `memory/paper_trade_v3_deploy_gate_2026_05_09.md` — v3 = canonical truth
- `memory/MEMORY.md` — STATE REPLAY CHAIN (point-in-time snapshots)
- `memory/claude_instance_state_2026_05_17_autonomous_cycle3.md` — current HEAD #12

---

## 11. METHODOLOGY (load-bearing — read AFTER §10)

**The HOW of advancing §1 → §8**: `STRAT_DISCOVERY_METHODOLOGY.md`

**SETUP DETECTION methodology** (canonical 2026-05-18, supersedes Forward+Backward as primary):
- Detection NOT prediction (3x ML predictor failures confirmed information-horizon mismatch)
- 1d avoidance layer (heuristic gates: btc_7d_ret < -15%, intraday breadth circuit-breaker)
- 4h/1h detection layer (signature matching on v51 microstructure metrics + indicator confluence)
- Conditional gate (cluster/regime/DNA selects active signatures, NOT predictive)
- See `STRAT_DISCOVERY_METHODOLOGY.md §5a-prime` for full architecture

Legacy: **INTEGRATED FORWARD+BACKWARD methodology** (now secondary; composition framework only):
- **Phase 0 (backward)**: outcome catalog — what was achievable per day with hindsight
- **Phase 1 (backward)**: condition discovery — what distinguishes HIGH-EV days
- **Phase 2 (bridge)**: day-class detector — predict today's class prospectively
- **Phase 3 (forward)**: indicator exhaustion CONDITIONAL on day-class (smaller search)
- **Phase 4 (specific backward)**: reverse-engineer our existing winners
- **Phase 5 (compose)**: HIGH-day gate + multi-indicator confluence + per-sleeve classifier + correlation-aware portfolio
- **Phase 6 (verify)**: walk-forward + DSR + OOS final gate + UNSEEN paper trade

Forward alone is bounded; backward alone is ungrounded. Integrated = each phase informs the next.

Includes:
- 19-dim asset-day classification taxonomy
- 7 chart types (dollar / range / runs / adaptive-vol / DIB / time)
- Per-indicator exhaustion checklist (§6 of methodology doc)
- Per-phase acceptance gates
- ROI capture math decomposition

## 12. ONE-LINE BINDING

> "Capture 1-10% daily moves under LO+spot+lev=1, empirically grounded in our 365-day opportunity set; +1-5%/d is the user-mandated target; published benchmarks irrelevant; methodology is the open variable (see `STRAT_DISCOVERY_METHODOLOGY.md`)."

If a future Claude instance reads this and proposes work that doesn't advance §1 or §8, PAUSE. Re-read this doc + the methodology doc.
