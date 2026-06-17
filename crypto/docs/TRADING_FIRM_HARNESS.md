# The Trading-Firm Harness — the target the project is wired to BE

> **Mandate (user, 2026-06-06):** *"I want a project that is a harness for solving the market, market engines, etc. —
> basically wired to be the best of a trading firm. This is before we even touch a single line of [strategy] code or
> go back to the MA problem."* Build the **apparatus** to world-class-firm standard first; the strategy (MA, etc.) is
> what the harness then *solves*, not the harness itself.

This doc is the **decompose-the-ideal target**: state what a world-class systematic crypto trading firm's harness IS,
then reverse-engineer our project toward it. The gap-map + EV-ranked roadmap are populated by the multi-agent design
run (`Workflow trading-firm-harness`, run wq3u9dvq1) and committed by the overseer after RWYB verification.

## The ideal: a world-class systematic crypto firm as a layered harness

A trading firm is a **pipeline of functions with a meta-layer (PM/CIO) over it**. Data feeds intelligence; intelligence
feeds forecasts; forecasts feed *decisions* (the probabilistic spine); decisions feed portfolio construction; the
portfolio is executed under a cost model; risk bounds everything; monitoring closes the loop; knowledge compounds; and
the PM/CIO layer allocates *research effort and capital-at-risk* across it all.

| # | Function | What the BEST firm's harness does | Our home (to be RWYB-verified) |
|---|---|---|---|
| 1 | **Alpha Research & Discovery** | hypothesis pipeline; conditioner/factor search; idea→ship-candidate | `discover` skill, `src/strat/battery.py` |
| 2 | **Market Data & Features** | point-in-time data, dollar/volume bars, feature eng., health gates | `src/pipeline/*`, chimera, `pre_train_gate.py` |
| 3 | **Market Intelligence & Regime** | regime detection, microstructure (VPIN/Kyle/Hawkes), tape narration | `src/narrate/*`, `narrate` skill |
| 4 | **Probabilistic Forecasting** | *distributional* forecasts, the WM, calibration, uncertainty | `src/wm/*`, `src/agent/*` |
| 5 | **Oracle / Ideal Decomposition** | best-achievable-within-constraints oracle → decompose DNA → reverse-engineer proxy | `docs/ORACLE_DECOMPOSITION`, `resourcefulness.py` |
| 6 | **Backtesting & Validation** | walk-forward + purge, deflated Sharpe, block-bootstrap, CSCV, overfit control | `src/anti_fragile.py`, `src/strat/battery.py`, CDAP |
| 7 | **Portfolio Construction** | sizing, fractional Kelly, risk-parity, cross-sleeve allocation | `src/wealth_bot/bot/position_sizer.py` |
| 8 | **Risk Management** | kill-switches, DD/VaR/CVaR, exposure limits, regime-risk, circuit breakers | `src/wealth_bot/bot/risk_manager.py` |
| 9 | **Execution & Cost** | cost models, slippage, p_fill, venue, microstructure-aware execution | `MakerCostModel`, `src/analysis/execution_sim.py` |
| 10 | **Strategy Lifecycle & Capital** | incubation→paper→live gates, sleeve lifecycle, capital tiers, decay retirement | `docs/WEALTH_BOT_DEVELOPMENT_FRAMEWORK.md`, `trader` |
| 11 | **Decision Governance** | promotion gates, BULL/BEAR/NULL committee, HITL for real capital | `decide` skill, `audit` skill, CDAP |
| 12 | **Monitoring & Live Ops** | live perf monitoring, edge-decay detection, alerting, liveness | `watcher.py`, `loop_health.py` |
| 13 | **Knowledge & Memory** | learnings, dossiers, failure catalog, anti-compaction memory, skill library | `rolling_ledger.py`, `skill_library.py`, `memory/` |
| 14 | **PM / CIO Meta-Layer** | research-effort + capital-at-risk allocation; what-to-work-on | `orc`/OVERSEER, the autonomy framework |
| 15 | **Probabilistic / Decision Spine** | expectancy, EV, Kelly-under-uncertainty, forecast→bet decision theory | **likely SCATTERED — the lead gap** |

## Verified target architecture — 9 layers + a meta/PM layer (from the design run, RWYB-grounded)

`L0 DATA/FEATURE SPINE` (strong; MISSING a bitemporal/as-of layer) → `L1 MARKET INTELLIGENCE/REGIME` (strong narrate +
cross-sectional rotation; MISSING a *persisted probabilistic* RegimeState) → `L2 PROBABILISTIC FORECASTING` (WM with
CRPS as a training loss; MISSING any proper-scoring **deploy gate** + a forecast object the decision layer consumes) →
`L3 DISCOVERY & FALSIFICATION` (**world-class** — `candidate_gate` chains cost→leak→firewall→battery→benchmark into one
verdict; MISSING the factory connective tissue) → `L4 DECISION/SIZING` (**the keystone gap**) → `L5 PORTFOLIO` →
`L6 RISK` → `L7 EXECUTION/COST` → `L8 MONITORING` — with the `META/PM` (orc/OVERSEER) over it.

## Gap-map — the 6 biggest gaps (RWYB-verified, adversarially)
1. **NO DECISION-THEORETIC SPINE** (highest leverage) — no module taking {calibrated forecast dist + regime posterior +
   cost + asymmetric loss} → sized position. *Partially closed this run: `src/firm/decision_spine.py`.*
2. **FORECAST→DECISION EDGE SEVERED** — `src/strat` consumes ZERO world-model forecasts; the WM-prod stack and the
   discovery/falsification stack are two disconnected signal worlds.
3. **GOVERNANCE/PORTFOLIO IS CODE-PRESENT BUT STATE-ABSENT** — meta_allocator, risk_parity, lifecycle, drift, half_life
   are written but their `runs/` dirs are EMPTY (never run productively). "A harness that has never produced state is
   not yet a harness."
4. **NO BITEMPORAL / AS-OF DATA LAYER** — look-ahead caught reactively (per-feature `shift(1)` + a correlation detector),
   not structurally; no `feature_publication_lag.yaml` / `asof_join.py` / knowable-at timestamp.
5. **NO DISCOVERY FACTORY** — can rigorously KILL one hand-built candidate, but cannot TRACK/GENERATE/DISPATCH a
   population (no hypothesis register, no mechanical dead-catalog, no spec→materialize→run→verdict loop).
6. **NO PROBABILISTIC REGIME STATE + NO PROPER-SCORING DEPLOY GATE** — regime is a hard-coded string (no posteriors);
   CRPS/ECE primitives exist but ZERO are gated before capital is risked.

## EV-ranked build roadmap (apparatus only; overseer builds + RWYB + commits)
| EV | effort | build | status |
|----|--------|-------|--------|
| 0.95 | M | **`decision_node` — the probabilistic decision spine** ({forecast posterior, regime posterior, cost, asymmetric loss} → EV + sized position) | **CORE BUILT** (`src/firm/decision_spine.py`); extend w/ regime-posterior + asymmetric loss |
| 0.88 | M | **forecast→strat fusion adapter** (give `candidate_gate` a leak-safe WM-forecast/regime conditioner) | open |
| 0.85 | M | **`hypothesis_register.py` + `runs/discovery/`** (spec→materialize→run→verdict + dead-catalog) | open |
| 0.82 | L | **`asof_join.py` + `feature_publication_lag.yaml` + CDAP invariant** (structural look-ahead defense) | open |
| 0.78 | M | **RegimeState producer** (probabilize `recalibrate()` + BOCPD transition, persist posteriors) | open |
| 0.74 | S | **calibration deploy-gate** (wire CRPS/ECE/pinball into `check_deploy_gates.py`) | open — next |
| 0.72 | M | **run the governance loop productively** (populate the empty `runs/` for allocator/lifecycle/drift) | open |
| 0.66 | S | **universe breadth/dispersion/risk-on-off roll-up** object | open |
| 0.60 | L | **dual-mode ingest spine** (wire `lob_depth_collector` to the feature contract) | open |
| 0.55 | S | **EV-frontier persistence** for the meta/PM layer (ledger-backed `frontier.json`) | open |

## Probabilistic / decision-spine verdict
**SCATTERED — there is NO coherent decision spine today; what exists is a world-class FALSIFICATION spine
masquerading as one.** `candidate_gate` answers *"is this candidate real?"* (binary SHIP/NOT), not *"given my beliefs
and costs, how much do I bet?"*. Quarter-Kelly lives in `position_sizer` fed by *rolling realized* win/loss (not a
forecast posterior); the WM computes distributions but ZERO proper-scoring rule gates them and `src/strat` consumes
none. **THE ONE BUILD: `decision_node`** — the keystone that gives the forecast layer a consumer, promotes Kelly from
rolling-stat-fed to posterior-fed, and turns the falsification gate into the front-end of a real decision policy.
*Started this run as `src/firm/decision_spine.py` (cost gate + continuous Kelly + uncertainty haircut + NO-TRADE default
+ Brier); the extension to regime-posterior + asymmetric loss + the forecast→strat edge is the remaining keystone work.*

---
*Design run: `Workflow trading-firm-harness` (wq3u9dvq1) — 31 agents, 15 functions mapped + adversarially verified.
The verify pass caught real corrections (the WS collector exists-but-unwired; `asset_rotation` IS cross-sectional;
`regime_classifier` HAS `recalibrate()`) — RWYB working as designed.*

## Built this run (3h harness run, 2026-06-06) — the firm's DECISION→ALLOCATION chain, wired end-to-end
The harness now composes **`market_state` → `decision_spine` → `portfolio`**, with `hypothesis_register` as discovery memory:
- **`src/firm/decision_spine.py`** (roadmap #1, ev 0.95) — the keystone: {forecast(μ,σ) + cost + **regime_posterior** +
  **asymmetric loss** } → sized bet via cost-gate → confidence-floor → continuous Kelly → uncertainty-haircut → risk cap,
  **NO-TRADE as default**, + Brier calibration. RWYB ALL PASS.
- **`src/firm/portfolio.py`** (L5) — N spine-bets → one risk-budgeted, correlation-aware, gross/per-name-capped book.
- **`src/firm/market_state.py`** (#8, ev 0.66) — universe breadth/dispersion/risk-on-off → favourability that **feeds the
  spine's `regime_posterior`** (proven: spine bets 0.103 risk-on vs 0.053 stressed).
- **`scripts/autonomy/hypothesis_register.py`** (#3, ev 0.85) — discovery-factory register + **monotonic dead-catalog**
  (seeded with the project's 6 real refuted veins; refuses re-mining).
- All RWYB-verified, committed+pushed, harvested to the skill library. **Remaining roadmap (open):** forecast→strat
  fusion adapter (#2), as-of join (#4), RegimeState producer (#5), calibration deploy-gate (#6), governance loop
  productively (#7), dual-mode ingest (#9), EV-frontier persistence (#10).

---
*Method: this is decompose-the-ideal (`resourcefulness.py`) applied to the firm itself — the ideal harness is the
oracle; we reverse-engineer the project toward it. Breadth = the 15 functions; depth = each function's gap.*
