# Strategy Discovery Methodology — V4 Crypto System

> **CANONICAL METHODOLOGY DOC** — load-bearing for all strategy work. Read AFTER `PROJECT_NORTH_STAR.md` (which is the WHY); this is the HOW.

> Companion to: `PROJECT_NORTH_STAR.md` (mission), `CLAUDE.md` (invariants), `STATE.md` (current state).

---

## 1. The problem we're solving

**Find combinations of (asset state, chart type, indicator, horizon, entry/exit logic) where a tradable edge exists, capture them as sleeves, compose into a portfolio under LO+spot+lev=1, and clear +1-5%/d ROI floor.**

Key constraints:
- No strategy works universally (except possibly microstructure / order-flow primitives)
- Edge is conditional on regime/DNA/state/chart-type
- Discovery must avoid conditional-event-day bias (per Cycle 3 finding: 4.7x median deflation)
- Cost realism: maker discipline, multi-day hold to amortize 0.05-0.24% RT
- Acceptance gate: v3 paper_trade_replay = canonical truth (yaml inflates 7.6x)

---

## 2. Empirical premise (binds methodology)

Per `memory/move_frequency_2025_2026.md` (reverified 2026-05-17):
- 95-100% of 2025-2026 days have ≥1 asset moving ≥5% (1-day cc)
- 20 movers/day on average ≥5%; 6 movers ≥10%; 35 movers ≥3%
- Top 25% of movers (5/day) = our capture surface
- **At 30% identification × 60% capture rate × 5% gross × 5 picks/day × 3-day amortization → +5-7%/d gross at idealized rates**
- **At current ~10% identification × 40% capture → ~0.16%/d (where we are)**

The methodology gap (identification 10% → 30%; capture 40% → 60%) is where the +1-5%/d unlock lives.

---

## 3. Classification taxonomy (the SLICING axes)

Every (asset, date) row can be classified along ~19 dimensions. The classification panel is the foundational data structure.

### A. Static / slow-moving (asset-level)
1. **DNA bucket** (BLUE/STEADY/VOLATILE/DEGEN — volatility signature + market cap)
2. **Sector / narrative cohort** (L1, DeFi, meme, AI, RWA, privacy, BTC-class, etc.)
3. **Liquidity tier** (per `config/universes/u100.yaml::liquidity_tier`)
4. **Listing recency** (asset lifecycle phase — new-listing momentum is structural)
5. **Multi-correlation profile**: β to BTC + β to ETH + β to sector-leader (3D not 1D)
6. **Asset volatility cluster** (per-asset historical vol regime, separate from market regime)

### B. Dynamic / fast-moving (asset-day state)
7. **Asset's own regime** (trending up / trending down / ranging / consolidating)
8. **Asset's drawdown phase** (peak / early correction / capitulation / recovery)
9. **Asset's volatility cluster TODAY** (rolling 7d vol z-score, not historical)

### C. Market-level regime
10. **BTC regime** (bear/bull/chop/crash via BTC 30d/7d/1d)
11. **Daily cluster** (L2 cluster_id K=6 from `runs/oracle_layer2/`)
12. **Cross-asset cohort co-movement** (today's sector-internal correlation tightness)

### D. Microstructure regime
13. **Order-flow regime** (Kyle λ quantile, VPIN quantile, Hawkes branching state)
14. **Liquidity regime** (LOB depth, bd_*, spread quantile)
15. **Funding regime** (perp funding extremity, applies even to spot-only trades)

### E. Flow regime
16. **ETF flow regime** (BTC/ETH ETF inflow vs outflow phase)
17. **Stablecoin supply regime** (stbl_* growth/contraction)
18. **Whale activity regime** (wh_* accumulation/distribution)

### F. Cross-asset / lead-lag
19. **Mover-cohort state** (asset's recent role: leader / follower / cool-down)

### Implementation
- Build `data/processed/classification_panel.parquet` from `chimera v51` + panels
- Output shape: ~600K rows (87 assets × ~2300 days)
- Cluster reduction: UMAP/K-means → 15-30 natural asset-day STATES
- Each state has its own characteristic indicator/chart-type response profile

---

## 4. Chart type framework

Multiple chart types give DIFFERENT views of the same price evolution. Same indicator behaves differently across chart types.

| Chart type | Captures | Best for | Already built |
|---|---|---|---|
| Dollar bars | Activity-clock; edge over time-clock retailers | Tick-volume confirmation | ✅ (chimera v51 base) |
| Time bars (1d / 4h / 1h / 15m) | Calendar-aligned reference | Daily reporting / regime gates | ✅ (chimera v51 cadence views) |
| Range bars | Price-range constant | Breakout / volatility expansion | ✅ (`bars/range_bars_fast.py`) |
| Run bars (volume) | Persistence | Trend continuation | ✅ (`bars/runs_bars.py`) |
| Run bars (tick) | Tick-count persistence | Micro-momentum | ✅ |
| Adaptive vol bars | Volatility-adaptive | Regime-shift detection | ✅ (`bars/adaptive_vol_bars.py`) |
| DIB bars (dollar imbalance) | Buy/sell pressure asymmetry | Order-flow alpha | ✅ (`bars/dib_bars_fast.py`) |

**Discovery axis**: same indicator (e.g., MA cross) applied to range bars vs time bars vs DIB bars produces 3 different signal profiles. Each may have its own (asset × regime × DNA) sweet-spot map.

---

## 5a-prime. METHODOLOGY UPGRADE 2026-05-18 — SETUP DETECTION (canonical, supersedes 5a as primary)

> **THE structural reframe** after 3 consecutive ML predictor failures (Phase 2 day-class, C0 detector, C34 detector — all collapse TRAIN AUC ~0.85+ → OOS AUC ~0.51). The cause: information-horizon mismatch. Macro catalysts that drive cluster transitions resolve INTRADAY, not from previous-day close features.

### Pivot: detection NOT prediction

| Mode | Information horizon | Empirical status |
|---|---|---|
| ❌ PREDICTION (old) | yesterday's close → tomorrow's cluster/outcome | 3x FAILED (Phase 2, C0, C34 detectors) |
| ✅ DETECTION (new) | TODAY's intraday metric state → setup happening NOW | Tractable; signatures are mathematically distinctive |

### Core insight

When an asset moves +1-5%, the v51 microstructure metrics show DISTINCTIVE PATTERNS during the move:
- **Hurst**: drops from ~0.5 → ~0.7+ (trending mode begins)
- **VPIN**: spikes (toxic flow) at breakout, then declines
- **Kyle's λ**: jumps (institutional flow enters; price impact increases)
- **Hawkes branching η**: > 0.8 (self-excitation; cascading flow)
- **OBV**: divergence vs price
- **Bollinger squeeze**: contraction → expansion

These are EMPIRICALLY DETECTABLE during the move, not from yesterday's close. We mine them on outcome-catalog HIGH-day winners and build a SIGNATURE LIBRARY.

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│ 1d AVOIDANCE LAYER                                            │
│   - btc_7d_ret < -15% → 3-day cool-down (simple heuristic)    │
│   - intraday circuit-breaker (noon UTC breadth check)         │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ 4h/1h DETECTION LAYER (this is where alpha lives)             │
│   - For each bar t, for each asset:                           │
│     * Compute current v51 metric state                        │
│     * Match against signature library (per cluster + DNA)     │
│     * Layer with technical indicator confirmation             │
│     * Fire long if match strength > threshold                 │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ CONDITIONAL GATE (cluster/regime/DNA — selects which          │
│   signatures are ACTIVE; not predictive)                      │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ EXECUTION + EXIT TRACKING                                     │
│   - Hold while signature persists                             │
│   - Exit on signature breakdown OR SL/TP/max-hold             │
└──────────────────────────────────────────────────────────────┘
```

### Phase structure (overrides §5a numbered phases as primary)

**Phase D-1: Signature mining** (backward, oracle-anchored)
For each HIGH-day winner in outcome_catalog (TRAIN segment, ~775 days × 21 winners ≈ 16K events):
- Identify the bar t where the asset's biggest 4h/1h move began (peak velocity)
- Extract v51 metric state at t-6, t-3, t-1, t (6h, 3h, 1h, 0h before move)
- Compute multi-metric signature vector
- Group by (cluster, DNA, regime) → signature centroids per cell

**Phase D-2: Anti-signature mining**
For LOW/NEG day picks (oracle's worst), same procedure → distinguishes winning from losing setups.

**Phase D-3: Signature matcher**
For each 4h/1h bar, compute distance to nearest centroid by cell. Top-K matches per day = trade candidates.

**Phase D-4: Technical indicator overlay**
Layer top-N indicator dossier survivors as confirmation:
- If MA(13,34) cross fires AND signature match > threshold → highest conviction
- If only one fires → reduced size
- If neither → no trade

**Phase D-5: Walk-forward verification**
Per §7b discipline:
- TRAIN: signature mining + matcher training
- VAL: hyperparameter tuning (threshold, K, hold bars)
- OOS: final dev gate (bear-stress)
- UNSEEN: burn-once paper trade

Plus WALK-FORWARD WITHIN TRAIN+VAL+OOS: 6-8 sub-quarters, per-quarter performance, regime-balanced.

**Phase D-6: Composition** (per §5a Phase 5; unchanged)

### Acceptance gates (per phase D)
- D-1: ≥5 distinguishable signature centroids per cluster × DNA bucket (otherwise too few HIGH-day samples)
- D-2: anti-signatures statistically separated from win-signatures (KS or Mahalanobis distance)
- D-3: matcher OOS precision@top-decile ≥ 50% (when match strong, asset wins next bar)
- D-4: confluence (matcher + indicator) precision ≥ 60%
- D-5: walk-forward 6-8 quarters: ≥4 quarters positive Sh ≥ 1.0
- D-6: composed v3 COMP ≥ +15% / Sh ≥ 2.0 / DD ≥ -8% on TRAIN+VAL+OOS (NOT UNSEEN)

### Why this might work where prediction failed
- No information-horizon mismatch (we detect concurrent with the move, not predict it)
- Microstructure metrics ARE distinctive during moves (mathematically documented in v51 features)
- Conditional gate prevents wrong-regime application (don't fire C1 detectors in C0)
- Technical indicators provide independent confirmation layer
- Real-time exit on signature breakdown avoids holding losers

### What this REPLACES
- §5a Phase 2 (day-class detector) — REMOVED from primary path; empirically infeasible from daily close
- §5a Phase 3 (indicator exhaust conditional on day-class) — RE-FRAMED as Phase D-4 (indicators as detection confirmation, not prediction)
- C0/C34 ML predictors — REPLACED by simple heuristic gate (btc_7d_ret < -15%) + intraday circuit-breaker

---

## 5a. METHODOLOGY UPGRADE 2026-05-17 — INTEGRATED FORWARD+BACKWARD (now SECONDARY — composition framework only)

> **This is the official methodology going forward.** Supersedes pure forward-only exhaustion. Replaces "indicator-first" with "outcome-anchored hypothesis-driven indicator exhaustion conditioned on day-class."

### Why integrated > forward-only

Forward-only (bottom-up: indicator → exhaust → wrap → hope) is bounded by what we test and doesn't ask "what would have to be true for +X%/d?" Per cycle 4 honest verdict: forward-only delivered scaffolding + negative findings but no demonstrable path to +1-5%/d.

Backward-only (top-down: outcome → conditions → engineer) generates hypotheses but doesn't ground them in feature exhaustion.

INTEGRATED: each is conditional on the other. Use backward to identify WHICH conditions matter; use forward to test WHICH indicators detect those conditions reliably.

### The 6-phase integrated cycle (run each cycle of strat work)

**Phase 0 — OUTCOME CATALOG** (backward; once, then refresh quarterly)
For each historical day (TRAIN+VAL+OOS):
- Compute the **oracle-optimal** LO portfolio return with K=5 picks, 1-3 day hold, post-cost (i.e., if we had perfect foresight, what return was AVAILABLE today?)
- Label day: HIGH (≥+2%), MED (+0.5 to +2%), LOW (-0.5 to +0.5%), NEG (<-0.5%)
- Output: `data/processed/outcome_catalog.parquet` with day-class + winning-pick metadata
- This is the GROUND TRUTH for "what's available."

**Phase 1 — CONDITION DISCOVERY** (backward)
What features distinguish HIGH days from MED/LOW/NEG?
- Statistical post-mortem on Phase 0 catalog
- Use classification panel (19 dims) + chimera v51 features
- Output: ranked feature-importance list + condition vocabulary

**Phase 2 — DAY-CLASS DETECTOR** (backward → forward bridge)
Train LightGBM: predict today's day-class given yesterday's features (no lookahead)
- Walk-forward TRAIN+VAL → OOS held-out test
- Acceptance: HIGH-day top-decile precision ≥ 60%
- Output: `src/oracle/day_class_filter.py::DayClassFilter`

**Phase 3 — FORWARD INDICATOR EXHAUSTION CONDITIONAL ON DAY-CLASS** (forward, hypothesis-grounded)
For each detected HIGH day, which indicator combinations pick winners?
- Use existing sweep harness (Phase C of cycle 4) but PARTITIONED by day-class
- Per-indicator dossier (§6 checklist) now includes per-day-class slice
- Output: indicator-class winners conditional on HIGH days (vs MED/LOW)

**Phase 4 — REVERSE-ENGINEER OUR EXISTING WINNERS** (backward, specific not generic)
For each currently-best blend (e.g., REGIME_ROUTER_STRICT_LO_SETUP60), examine:
- Days it WON big → what was different about those days?
- Days it LOST → what was different?
- Where did it miss HIGH days the oracle catalog flagged?
- Output: targeted improvement list for each deployed blend

**Phase 5 — COMPOSITION** (integrated)
- HIGH-day gate (from Phase 2 detector)
- ON HIGH days: multi-indicator confluence (from Phase 3) selects top-K assets
- Per-sleeve setup classifier (Phase D from cycle 4) provides per-asset conviction
- Correlation-aware Markowitz-lite (per_day_composition.py) builds portfolio
- OFF HIGH days: cash OR fallback regime-conditional sleeves

**Phase 6 — HONEST VERIFICATION** (§7b discipline, unchanged)
- Walk-forward 6-8 quarters from TRAIN+VAL
- OOS final dev gate (bear-stress)
- UNSEEN burn-once per blend
- DSR correction for n_trials

### What makes this faster

| Old (forward-only) | New (integrated) |
|---|---|
| Exhaust every indicator class blindly | Test indicators CONDITIONAL on day-class (much smaller search) |
| Hope composition meets target | KNOW target is achievable from oracle catalog |
| Discover failure modes blind | KNOW failure modes from existing-winner reverse-engineering |
| Sleeves don't share intelligence | Each phase informs the next |

### What this REPLACES from the old §5

The old "indicator-first exhaustive" section is now ONE PIECE of Phase 3 — still valuable, but always conditional on day-class. Pure indicator exhaustion without day-class partitioning is now ONLY allowed if we don't yet have outcome catalog (Phase 0 not yet built).

### Acceptance gates per phase
- Phase 0: catalog covers ≥95% of (asset, date) in TRAIN+VAL+OOS; HIGH-day rate 15-30% (sanity check)
- Phase 1: ≥5 features with statistically-significant HIGH-day association (p<0.01)
- Phase 2: HIGH-day precision@10 > 60% on OOS (held-out)
- Phase 3: per-indicator dossier has per-day-class slice with non-degenerate sample
- Phase 4: per-existing-blend improvement list with specific actionable changes
- Phase 5: composed blend COMP ≥ +25% / Sh ≥ 2.0 / DD ≥ -6% (4mo)
- Phase 6: walk-forward + DSR + OOS final dev gate cleared

---

## 5b. Discovery methodology — INDICATOR-FIRST EXHAUSTIVE (legacy; now Phase 3 of integrated)

### Principle
For each indicator class, exhaustively measure performance across the FULL classification × chart-type × parameter × horizon × side product space. Identify winners. Wrap winners as sleeve(s). Move to next indicator class.

This is empirical (data shows what works) rather than prescriptive (predetermined structure).

### Per-indicator-class procedure

```
For each indicator_class in registry:  # MA-cross, RSI, BB, Donchian, OBV, VPIN, Hawkes, Kyle-λ, YZ-vol, ETF-flow, Liquidation-cascade, Whale-flow, etc.
    For each asset in u100:
        For each chart_type in {dollar, range, runs_vol, runs_tick, adaptive_vol, DIB, 1d_time, 4h_time, 1h_time, 15m_time}:
            For each cadence/horizon:
                For each param_combo (e.g., MA periods 1-100 × 1-100):
                    For each side (long / short_as_stayout / no-trade):
                        For each regime_filter in {none, BTC_regime, asset_drawdown_phase, microstructure_regime}:
                            Measure UNCONDITIONALLY (not just on event days):
                              - n_fires (across full 2020-2026 history)
                              - mean_pnl_per_fire (net of taker + maker cost)
                              - hit_rate
                              - Sharpe (annualized, daily-aggregated)
                              - max_dd_per_fire
                              - capture_pct of move
                              - per-(asset-day-state) Sharpe slice
```

### Acceptance gate per cell (per indicator class)
- n_fires ≥ 30 across the test window
- Unconditional Sharpe ≥ 1.5
- mean_pnl_net ≥ +0.5% per fire (covers ~2x cost)
- hit_rate ≥ 45%
- Stability: positive in ≥3 of 4 quarters

### Sleeve wrapping pattern (after per-indicator exhaust)
For each surviving cell-cluster (e.g., "MA(13,34) on dollar-bar TAO during BTC-bull regime — Sh 2.8 / n=87 fires"):
1. Generate sleeve config: indicator + chart_type + asset_filter + regime_filter + params + side + hold + upgrades (SL/TP/breakeven)
2. Wrap via `oracle_wrapper_sleeve.py` (existing pattern, extended for chart-type axis)
3. Register in `production_blends.yaml` as `<INDICATOR>_<ASSET-OR-COHORT>_<STATE>_LO`
4. v3 paper_trade_replay 4 monthly seeds — TRUE acceptance gate
5. If v3 PASS (COMP ≥ +6%, Sh ≥ 1.5, DD ≥ -8%): promote to active deploy pool
6. Move to next indicator class

### What this avoids
- Conditional-event-day bias (per Cycle 3 Path A finding) — measure unconditional
- Universe-wide averaging (per cycle 3 STRICT_LO_SETUP60 finding) — measure per-asset, aggregate at composition layer
- Path A v2 "single-indicator-threshold weak" — wrap per natural cell-cluster, possibly multi-indicator confluence

---

## 6. Information-exhaustion checkpoint per indicator class

Before moving to next indicator class, complete this checklist for the current class:

- [ ] Per-asset-day-state Sharpe distribution mapped (which states does this indicator help?)
- [ ] Per-chart-type best-form identified (which chart this indicator likes most?)
- [ ] Per-cadence horizon stability tested (does it work at 4h, 1h, 15m?)
- [ ] Per-asset DNA preference profiled (does it favor DEGEN vs BLUE?)
- [ ] Per-regime conditional performance documented (when off, when on?)
- [ ] Surviving cells → sleeve configs written
- [ ] v3 verify gate run on top 5-10 surviving sleeves
- [ ] Sleeve scoresheet entry created
- [ ] Negative-result honest log entry (which (state, chart) combinations DON'T work for this indicator)
- [ ] Move-to-next certified

This produces an INDICATOR DOSSIER per class — a permanent artifact future Claude instances can read instead of re-running the exhaust.

---

## 7. Composition layer (after N indicator classes processed)

Once 5-10 indicator classes have surviving sleeves:

1. **Setup classifier per sleeve**: LightGBM predicts "today this sleeve's setup is live for asset A". Built per-sleeve, not universal.
2. **Per-day routing**: for each asset A on date D, query each sleeve's setup_prob. Build a confidence-ranked candidate list.
3. **Correlation-aware portfolio**: pick top-K with low pairwise correlation (Markowitz-lite with DD cap).
4. **Capacity scaling**: replicate single-asset high-conviction signals across multiple-asset cohorts where DNA matches.

---

## 7b. SPLIT DISCIPLINE — non-negotiable

> **Per user 2026-05-17 critical-question check** + per `CLAUDE.md` invariants. Inviolable; supersedes any time-saving shortcut.

### Segment definitions (canonical)

| Segment | Window | Purpose | Iteration policy |
|---|---|---|---|
| **TRAIN** | 2020-01-01 → 2024-12-31 (~1820 days, all 4 regimes balanced) | Model fitting, parameter sweeps, indicator exhaustion | UNLIMITED iteration |
| **VAL** | 2025-01-01 → 2025-09-30 (~273 days, chop-heavy) | Hyperparameter tuning, candidate winnowing | BOUNDED iteration with DSR tracking |
| **OOS** | 2025-10-01 → 2025-12-31 (~92 days, bear-skewed: 95% non-bull) | FINAL validation gate before paper trade | Each touch counted; max 3 retries per blend |
| **UNSEEN** | 2026-01-01 → 2026-04-30 (~120 days) | **RESERVED for paper_trade_replay deploy gate ONLY** | Burn-once per blend; never used for development |

### Critical insight (validated 2026-05-17)
**OOS alone is regime-skewed** (95% non-bull; bear 35 / chop 34 / crash 19 / bull 4). A blend tuned only on OOS will be bear-overfit.

**TRAIN+VAL is regime-diverse** (combined ~2090 days; all 4 regimes well-represented).

### Validation methodology (mandatory before any deploy claim)

1. **Walk-forward 6-8 quarters from TRAIN+VAL** (e.g., 2024-Q1/Q2/Q3/Q4 + 2025-Q1/Q2/Q3 = 7 quarters covering bull/chop/bear/crash):
   - Per-quarter Sharpe ≥ 1.0 in ≥3 of 7 quarters (regime-balanced robustness)
   - Per-quarter COMP positive in ≥4 of 7 quarters
   - No single quarter DD < -10%
2. **OOS final dev gate**: blend passes walk-forward, then v3 on 2025-Q4 (92 days bear-stress):
   - COMP ≥ +3% (lower bar because bear-skewed)
   - Sh ≥ 1.0
   - DD ≥ -6%
3. **UNSEEN paper trade replay**: ONLY after walk-forward + OOS pass:
   - Per `memory/paper_trade_v3_deploy_gate_2026_05_09.md` 7 G-guarantees
   - Single read per blend; result is the deploy decision

### Multi-comparison correction (DSR)

Per `_invariants.yaml::dsr_published_ranking` (now ENFORCED):
- Track n_trials = number of v3 variants tested in the search
- Apply Deflated Sharpe Ratio (Bailey-Lopez-de-Prado)
- DSR ≥ 0.90 required for deploy claim
- Cycle 3 finalists: ~25-30 v3 variants tested → DSR deflation ~30-40% on chosen blend

### What this fixes
- Cycle 3 iteration burned UNSEEN through ~10 v3 batches; finding magnitudes likely overstated 1.5-2x
- Going forward: discovery + iteration on TRAIN+VAL+OOS; UNSEEN reserved
- Regime-balanced validation prevents bear-only / bull-only overfit
- DSR correction prevents multi-comparison artifact

### Per-batch contract (every v3 batch must declare)
1. Which split is being tested (TRAIN-quarter / VAL-quarter / OOS / UNSEEN)
2. How many variants are in the search (for DSR n=)
3. Whether the batch is dev-iteration (TRAIN/VAL/OOS) or deploy-gate (UNSEEN)

If UNSEEN is being touched: requires sign-off that this is the FINAL blend going to paper trade — not "let me try one more variant".

---

## 8. Acceptance gates (per phase + per artifact)

| Artifact | Gate |
|---|---|
| Classification panel | ≥95% (asset, date) coverage; 15-30 natural states post-clustering |
| Per-indicator exhaust | Reproducible; UNCONDITIONAL measurement; n_fires ≥30 per cell |
| Surviving cells | Sh ≥ 1.5 / IC ≥ 0.015 / hit ≥ 0.45 / +mean_pnl per fire |
| Sleeve wrapping | v3 paper_trade_replay 4-monthly cells |
| Deployable sleeve | COMP ≥ +6% (over 4mo) / Sh ≥ 1.5 / DD ≥ -8% / fires ≥ 10/mo |
| Per-asset ML | UNSEEN AUC > 0.55 / top-decile precision > 60% |
| Composed blend | COMP ≥ +25% (4mo) at composition layer / Sh ≥ 2.0 / DD ≥ -6% |
| FINAL DEPLOY | Per `memory/paper_trade_v3_deploy_gate_2026_05_09.md`: 7 G-guarantees + walk-forward + DSR(n=N_universe) + 5-Q replay |

---

## 9. Math — what we need to lift

ROI capture decomposition under LO+spot+lev=1:

```
Daily_gross = (movers_pred × P(direction_correct) × capture_pct × move_avg) ÷ NAV
            - (n_trades × cost_per_RT ÷ N_assets)
            - regime_drawdown_drag

Target +0.5%/d:
  movers_pred = 5/day (top 25% of 20 movers)
  P(direction_correct) ≥ 0.40
  capture_pct ≥ 0.50 (of average 5% move)
  cost_per_RT ≤ 0.10% (maker discipline)
  → Daily_gross = 5 × 0.40 × 0.50 × 5% × (0.05 NAV/asset) - 5 × 0.10% × 0.05
                = 5 × 0.50% - 5 × 0.005% = +2.475% - 0.025% = +2.45%

Hmm, that's actually +2.45%/d at idealized rates.
Current actual ~+0.16%/d ≈ 6.5% of theoretical.
```

The methodology gap is:
- Identification rate (current ~10% → target 30%) = 3x lift
- Capture rate (current ~40% → target 60%) = 1.5x lift
- Cost discipline (current taker 0.24% RT → target maker 0.05% RT) = 4.8x cost reduction
- Compounded: ~22x — clears the path from 0.16%/d to 1-3%/d

---

## 10. Phase plan — revised per indicator-first methodology

### Phase A: Foundation (1 cycle)
**Build the classification panel** (Section 3) + audit chart-type framework (Section 4).
Deliverable: `data/processed/classification_panel.parquet` + 15-30 natural states + chart-type compatibility matrix.

### Phase B: Indicator-first exhaustive sweep (iterative — 1 indicator class per sub-cycle)
For each indicator class, run Section 5 procedure. Produce indicator dossier per Section 6.

Priority order (per existing Oracle L3 evidence + theoretical importance):
1. **Order-flow primitives** (Kyle λ, VPIN, Hawkes branching) — most universal per Section 1
2. **Microstructure** (OFI, LOB imbalance, spread)
3. **Trend / momentum** (MA crosses, RSI, MACD) — most populated existing data
4. **Volatility** (BB, YZ-vol, ATR breakouts)
5. **Liquidation flow** (liq_*)
6. **Whale / flow** (wh_*, etf_*, stbl_*)
7. **Range / breakout** (Donchian, range bars structural)
8. **Cross-asset / cohort** (cohort_co_movement)
9. **Calendar / seasonality** (DoW, MoM, options expiry)
10. **Composite / multi-indicator confluence** (e.g., TAO 2-of-4 pattern proven Sh 2.56)

Each indicator class: ~1 session if data is clean and we have parallelism. Iterate.

### Phase C: Sleeve dossier + v3 verify (continuous per Phase B)
Each indicator class's surviving cells → wrapped sleeves → v3-verified. Build deployment pool incrementally.

### Phase D: Setup classifier + composition (1 cycle after Phase B has ≥5 indicator classes)
Per-sleeve setup classifier; correlation-aware composition; capacity scaling.

### Phase E: Sub-day cadence pivot (1 cycle)
Extend the v3 measurement to sub-day. Re-measure top sleeves at 4h cadence.

### Phase F: Continuous iteration
- New indicator classes added → run Section 5 procedure
- Negative findings (indicator/state doesn't work) → permanent dossier entry
- Setup classifier retrained when new sleeves added
- Deployment pool refreshed

---

## 11. What this methodology AVOIDS

- ❌ Universe-wide aggregation that hides per-asset edge (TAO Sh 4.6 averaged into noise)
- ❌ Conditional-event-day bias (single-indicator threshold on event days only)
- ❌ Single-state strategies (RSI universally — doesn't work; RSI in microstructure-regime-X on DIB chart for VOLATILE-DNA = may work)
- ❌ Hand-coded sleeve design (slow, biased by what we think to code) — replaced by exhaustive discovery
- ❌ Restarting from scratch every cycle (dossiers persist)

---

## 12. What this methodology ENABLES

- ✅ Permanent indicator dossiers — never re-run a class once gate-checked
- ✅ Per-(asset-day-state) edge surface mapped empirically
- ✅ Multiple chart types treated as 1st-class discovery axis
- ✅ Composition built bottom-up from honest cell-level evidence
- ✅ Setup classifier per-sleeve (not universal) — closes Path A bias gap
- ✅ Capacity-scaling pattern (TAO Sh 2.56 → multi-asset replication) productionizes high-precision signals
- ✅ Cycle-over-cycle compounding (new indicator class → new sleeve pool → composition lift)

---

## 13. Status (as of 2026-05-17)

### Done
- ✅ MA/EMA cross indicator: 9,900 cells exhausted (per `memory/ma_ema_permutation_layer3_2026_05_12.md`); top: SMA(13,34) universal Sh ~0.37/event; per-asset DNA (TRX bear long Sh 2.49)
- ✅ Liquidation cascade: cells in L3 panel (`liq_short_spike`, `liq_capitulation`, `liq_short_panic`)
- ✅ 10 indicator classes scaffolded in `runs/oracle_layer3/` (RSI, Bollinger, Donchian, OBV, VPIN, Hawkes, Kyle, YZ-vol, ETF-flow, Liquidation)
- ✅ Per-asset DNA cards: u50 in `runs/oracle_layer3_asset_dna/`
- ✅ Unconditional L3 panel rebuild: `runs/oracle_layer3/indicator_perf_panel_UNCONDITIONAL.parquet`
- ✅ Setup classifier (universe-wide): `src/oracle/setup_filter.py` (AUC 0.6157)
- ✅ Auto-sleeve generator: `scripts/oracle/auto_sleeve_generator.py`
- ✅ STAYOUT pattern, SETUP_FILTER gate, extra_params dispatch — all wired

### Partially done (needs to follow methodology fully)
- ⏸ Chart-type axis NOT yet exhausted (only dollar bars in L3 panel)
- ⏸ Many indicator classes have cells but no SLEEVE DOSSIER per Section 6 checklist
- ⏸ Setup classifier is universe-wide, not per-sleeve

### Not started
- 🔲 Classification panel (Section 3 — 19 dims combined)
- 🔲 Per-asset ML layer
- 🔲 Composition layer (correlation-aware + capacity-scaled)
- 🔲 Sub-day cadence native v3

---

## 14. References

- `PROJECT_NORTH_STAR.md` — mission, ROI tiers, premise
- `CLAUDE.md` — invariants
- `STATE.md` — current state
- `memory/move_frequency_2025_2026.md` — empirical premise
- `memory/feedback_unconstrained_2026_05_17.md` — LAYER-0 mindset
- `memory/ma_ema_permutation_layer3_2026_05_12.md` — first indicator-first exhaust
- `memory/asset_dna_u100_2026_05_12.md` — per-asset DNA evidence
- `memory/oracle_layer3_framework_2026_05_12.md` — discovery framework spec
- `runs/audit/AUTONOMOUS_CYCLE3_REPORT_2026_05_17.md` — latest cycle report
- `runs/audit/PATH_A_HONEST_FINDINGS_2026_05_17.md` — conditional-bias finding
- `runs/audit/PATH_A_V2_UNCONDITIONAL_RESULTS_2026_05_17.md` — unconditional measurement
- `runs/audit/TAO_COMPOSITE_2026_05_17.md` — capacity-bound multi-indicator success

---

## 15. One-line binding

> "For each indicator class, exhaust the (asset × state × chart × param × horizon) space unconditionally; wrap honest survivors as sleeves; v3-verify; build a permanent indicator dossier; iterate; composition layer routes per-day via per-sleeve setup classifiers under correlation-aware Markowitz-lite."

Future Claude instances: PAUSE before writing new strategy code. Check the indicator dossier for the indicator class you intend to use. If a dossier exists, READ IT FIRST. Don't re-run an exhausted class.
