# Trading-System Foundation — Fresh Start (2026-06-04)

> **Status:** LIVING DOC — research campaign in progress. Pre-code foundation.
> **Mandate (user, /trader autonomous):** "research the problem precisely and thoroughly, multiple
> angles, lay down the framework and approaches we'll use, before touching a single line of code.
> Explore the whole infra (chart types, info types, density, gaps). Don't be narrow (prior instances
> anchored to 4h and died of narrow-mindedness). Also test whether the operational agent framework
> is active/usable/needs refinement."
> **VERIFIED wall-clock start:** 2026-06-04 21:25:02 SAST.
> **Hard constraint:** LONG-ONLY + SPOT + LEVERAGE=1. Objective: WEALTH (compound %) under robustness.

This doc is the write-forward anchor (AUTONOMOUS_RUNNER §5). The architecture in §1–§6 is settled from
four design turns; §7 holds live research findings; §8 evaluates the agent framework.

> **🟠 FOUNDATION-PHASE RE-FRAME (2026-06-05, user correction).** The goal of this work is to SET THE
> FOUNDATION — the avenue-map + apparatus + methodology + operating framework — **NOT to SOLVE whether an
> edge exists.** §11–24 below drifted into SOLVING (verdicts like "no active alpha" / "deploy the
> preservation floor"). Those are **4h/daily-LO-specific prior-experience signals — the KNOWN-HARD starting
> premise — NOT global conclusions.** Conclusions are **HELD OPEN** (re-test, don't inherit). The foundation
> DELIVERABLE is [`AVENUE_MAP_2026_06_04.md`](AVENUE_MAP_2026_06_04.md) (the full exploration space + per-avenue
> status + the apparatus to test each). Read §11–24 as *"what the 4h/daily-LO corner showed + the apparatus we
> built,"* not as *"the answer."*

---

## 1. The objective, framed edge-first (not target-first)

The #1 prior failure was **target-first**: declaring +1–5%/day with no edge, on the least-tradeable
substrate, validated by an untrustworthy apparatus. We invert it.

- **Goal in plain terms (user):** *enter, leave, profit* — chase a setup, capture a meaningful move,
  cut instantly if wrong, rotate to the next. Across timeframes and holding periods.
- **The only non-negotiable test:** the *activity of timing* must beat the *laziness of not timing*,
  net of realistic cost — because if it doesn't, buy-and-hold (or a fixed partial allocation) is the
  better profit machine. "Beat lazy + beat buy-and-hold" is not alpha-purism; it's the literal test
  that the timing is doing anything. (A no-edge timer is strictly dominated by a static partial hold:
  lower exposure return minus trading cost.)
- **Where long-only timing profit actually lives:** drawdown-avoidance at regime *transitions*
  (bull→bear→recovery), NOT out-trading a bull (buy-hold wins there) and NOT scalping chop (cost wins
  there). Implication: trend/timing edges MUST be tested across a full cycle incl. the 2022 bear —
  which sits in TRAIN, since UNSEEN (2025-03+) is bull/recovery only.
- **Return is an OUTPUT**, read off a proven edge — never an input target.

## 2. The architecture: a portfolio of per-asset specialists

**Unit of discovery:** a cell in the grid **TI / gated-TI × ASSET × REGIME × TIMEFRAME/CHART-TYPE.**
**Unit of deployment:** a validated per-asset sleeve ("trade this asset as if it's the only one you can").
**Portfolio:** ~u100 specialists; opportunity-coverage emerges from breadth (at any time some subset is
"firing"). This is the "top-25%-of-movers" engine, bottom-up not top-down.

**Decouple → combine (user's principle):** find profitable *standalone* per-asset cells first; combine
*after*. Different TIs on the same asset are *decoupled* return sources (intra-asset diversification).
This was partly VALIDATED pre-reset: cross-sleeve mean pairwise monthly corr ≈ −0.06; pooling lifts
n_eff ≈ 11 → 108. Bottleneck = SUPPLY of validated sleeves, not aggregation.

**Refinement the evidence demands (⚖️ trader judgment):** standalone bare-TI mining was a lottery
(~0 survivors on majors); where an edge was found it lived in an **exo-conditioner/GATE** (e.g. whale
flow) with the TI as the *structural frame* (slow trend → window-consistency + DD control). So the ship
unit is flexible — `ASSET × [TI-signal]` OR `ASSET × [GATE] × [TI-structure] × [CLOCK]` — and we
*instrument it to learn which carries the edge*. Re-test TI×ASSET from scratch (don't inherit the
"lottery" verdict), but measure gate-vs-TI attribution.

**Capital velocity is a first-class metric:** 2.5% in 1 day ≫ 2.5% in 6 days (6× capital efficiency).
Track return-per-deployed-capital-day AND return-per-calendar-day; the bridge is opportunity density
(why breadth matters — broad scanning keeps capital deployed).

**The Benedict exit layer (method-agnostic, bolts onto every sleeve):** hard stop (the "I'm wrong" exit,
~2–2.5% per-trade risk cap) + time-stop (the "this is dead, free the capital" exit — the velocity engine).

## 3. The eight prior failure points → installed guards

1. Target-first incompatible w/ constraint → edge-first, return is output.
2. Opportunity-existence ≠ harvestability → ex-ante selection + net-of-cost gate.
3. Cost floor violated by cadence → cost-realism gate PER cadence (cost-feasibility frontier).
4. Long-only = beta confound → benchmark-excess, per regime.
5. Untrustworthy numbers / selection leak → DSR@family-N + leak test + no-UNSEEN-reselect.
6. Discipline mistaken for alpha → mechanism-first; sizing gated last.
7. Depth not breadth / over-mining → portfolio after gate; saturation stop-rule; dead-list.
8. Unreliable autonomy (fabrication) → wall-clock grounding; pre-delivery self-audit.

## 4. Apparatus — trust state (3 holes to spec, NOT yet fixed)

5 of 8 failure modes are guarded in the kept harness (MtM double-count, close-only-breach, no-UNSEEN-
reselect, n_eff floor, cost charged). Three are surprises-in-waiting; precise fix-specs in §7-C:
1. 🔴 Cost+fill optimism: harness default 0.10% < real 0.24% taker, AND assumes 100% fill (no p_fill;
   empirical 0.21–0.40). Naive use inflates ~2–3×.
2. 🟠 Look-ahead not auto-detected: fill-at-next-open is enforced, but indicator past-only-ness +
   full-history standardization are caller-trusted (no runtime check).
3. 🟠 DSR/Holm not auto-applied: exists (`check_dsr_holm.py`) but caller must run it across the grid.

## 5. Methods, placed along discover→combine

- **Static/dynamic rules** — the DISCOVERY engine for per-cell edges. Start here.
- **ML** — NOT a generator (worst track record). Defensible as (a) meta-labeler on a §4-survivor,
  (b) allocator over already-validated sleeves ("pick PEPE vs BTC" lives here, at COMBINE).
- **Self-improving bot** — decay-aware rotation over the validated sleeve library (lifecycle). The
  decay-detection primitive is the valuable, buildable part. After a library exists.
- **WM (AlphaZero analog)** — research, OFF critical path. Disanalogy: markets are partial-info,
  non-stationary, NO perfect simulator; self-play exploits the learned simulator's errors, not the
  market's (prior RL: Sharpe −9.66). Gated on the unsolved simulator-fidelity problem.

**Governing principles:** (P1) methods don't create edge, they express it — prove edge dumb-and-simple,
add sophistication only to capture more of a KNOWN edge. (P2) a richer feature set is a bigger overfit
surface, not free alpha — discover parsimonious, refine rich. (P3) validation must match the holding
period + cost hurdle.

**Build order:** 0 lock apparatus → 1 prove ONE survivor → 2 measure decoupling → 3 scale grid
(family-corrected) → 4 combine + ML allocate + decay-rotate → X WM research in parallel.
**→ Superseded by the actionable phased plan in [`RETEST_PLAN_2026_06_04.md`](RETEST_PLAN_2026_06_04.md): Phase 0 apparatus → Phase 1 honest base → Fork A (bank beta) / B (fine-resolution capture) / C (tick).**

## 6. Open decisions (user's; parked, not forced)

1. Target vs constraint (keep LO+spot+lev=1 and let return float, or relax).
2. Benchmark definition (buy-hold BTC / equal-weight basket / exposure-matched static).
3. Operationalizing "the move" + "capture" (the ex-ante selection rule).
4. Universe scope (u10 clean vs u50/u100 breadth+survivorship/capacity risk).

---

## 7. Research campaign — live findings
_(populated per cycle; each entry tagged CONFIRM / REFUTE / EXHAUST / OPEN with citations)_

### 7-A Infra / chart-type / info-density map — DONE (Wave 1)
- **Time bars (1d/4h/1h/30m/15m) are DERIVED from dollar bars** (last dollar bar per time bucket; `make_dataset.py:212`). Dollar bars are the PRIMARY substrate. There is no separate raw 4h fetch — the resolution floor is the dollar bar.
- **BTC dollar bars = ~1,155/day (~75-second), NOT the "288/5-min" target** in `data_config.yaml:41` — thresholds drifted as volume grew. SOL ~1,970/day, PEPE ~1,297/day.
- **Coverage:** 104 assets have dollar+1d/4h/1h/15m chimera; 77 also have 30m (27 missing from an early batch); 5m/1m exist in code but NOT materialized.
- **5 information-driven alt-bar types exist as RAW bars for 87 assets (2020–2026):** `dib` (dollar-imbalance, ~200/day), `runs_tick` (~30/day), `runs_volume` (BROKEN — 2023-only, pathological PEPE threshold), `range` (vol-driven), `adaptive_vol` (~50/day, vol-scaled). **BUT chimera-enriched versions exist only for BTC/ETH/PEPE (dib, range); runs_tick/runs_volume/adaptive_vol chimeras are EMPTY.** Using alt bars on the universe requires building chimeras (compute cost).
- Exploratory `data/processed/alt_bars/`: Heikin-Ashi + Renko for PEPE only, outside the pipeline.
- **Anti-4h-anchoring takeaway:** the substrate is far richer than "4h time bars." 15m exists universe-wide; the dollar floor is ~75s; a whole information-driven bar family is built-raw but under-enriched. CONFIRM: timeframe/chart-type must be a swept axis, not a default.

### 7-B Feature / info-source taxonomy — DONE (Wave 1)
- **184 cols = 41 canonical + 142 frontier + 1 bar-grain.** Most are look-back-safe (rolling-z or `shift(1)`).
- **Best GATE candidates (where edge tends to live):** `s3_smart_vs_retail`/`_z` (smart-vs-retail positioning — the closest analog to the PEPE whale gate), `wh_whale_net_usd` + `xrel_wh_whale_net_usd_xpct10`, `hbr_eta_imbalance` (Hawkes buy/sell asymmetry), `liq_capitulation`/`liq_short_panic`, `bs_basis_bull/bear_shock`, `etf_*`/`stbl_*` macro shocks, `rv_jump_frac`, `xex_spread_dispersion`.
- **Best STRUCTURAL signals:** `regime_label`/`norm_ma_distance` (SMA-200), `norm_fd_close` (frac-diff), `xd_ma_distance` (x-sec trend), `norm_return_4/16`, `norm_efficiency`.
- **Ex-ante REGIME labels:** `regime_label` (SMA-200, 0/1/2), `norm_efficiency`, `norm_vol_cluster`/`norm_yz_volatility`, `xd_cross_return_mean` (breadth), `s3` positioning, `fund_rate_z30`.
- **🟠 Look-ahead flags:** `xd_btc_return`/`xd_btc_volatility` are SAME-BAR pass-throughs (not ex-ante at intraday). Daily-silver cols (`bd_`/`te_`/`hbr_`/`lob_`/`mv_`) need a **+1d lag in live** (only available end-of-day). The `no_full_history_standardization` CDAP rule is DISABLED.
- **Coverage gaps:** `dv_dvol_*` = BTC/ETH only; `xex_*` = 5 assets (BTC/ETH/SOL/XRP/DOGE); `soc_wiki_views` ≈ empty. Dead linear features: `norm_funding`, `hurst_regime`, `norm_perm_entropy`.

### 7-C Apparatus lock-down spec — DONE (Wave 1) — 🔴 WORSE THAN THOUGHT
- **Hole 1 (cost+fill):** harness `cost_rt=0.0010` (maker, not 0.0024 taker) and **zero p_fill modeling** (`harness.py:175,654`). `config/maker_cost_calibration.yaml` shows empirical **p_fill 0.21–0.40 AND adverse_selection 0.96–1.00** — i.e. when a maker order DOES fill it is almost always picked off — **but the yaml is never read by the harness.** **Implication: maker execution is effectively dead for us; the working cost assumption must be TAKER 0.24%.** Fix-spec: a `FillModel{mode: taker(0.0024,p_fill 1.0) | maker_pessimistic(0.0010,0.30,adv 0.96) | maker_calibrated | ideal}`, Monte-Carlo over p_fill, report median/p05/p95.
- **Hole 2 (leak):** `_validate_df()` checks only that signal columns EXIST, not that they were shifted; `Q4_look_ahead_integrity` is a hardcoded `"VERIFIED"` string (`harness.py:487,762`). Fix-spec: `shift_sensitivity_test(harness, shift_bars=1)` — shift signal one extra bar; `max_abs_delta > 5pp` ⇒ `LEAK_SUSPECT`, `> 20pp` ⇒ `LEAK_HIGH`. Also shift `filter_col` independently.
- **Hole 3 (DSR):** 🔴 the DSR/Holm gate is a **STRUCTURAL NO-OP** — severity is always `"warn"`, so exit-2 is unreachable; and `n_trials = len(written JSONs)`, NOT the true grid size, so the correction is far too weak (`check_dsr_holm.py:165,183`). Fix-spec: `severity="critical"` for claimed-ships that fail Holm; a `_sweep_manifest.json` sidecar declaring true `n_variants_tested`; `n_trials = max(written, manifest)`. **This means prior "DSR-protected" claims were never hard-gated — consistent with how the +36.8% selection-leak shipped.**

### 7-D Dead-list + reusable-asset register — DONE (Wave 1) — READ-FORWARD
**DEAD-LIST (do NOT re-mine; each killed by a named test):** D1 standalone price-TI mining (0 survivors, ~127 indicators × all cadences); D2 exo-conditioners HURT in UNSEEN (whale-gate t=−2.82; bare MA beats them); D3 order-book flow reversion (catastrophic OOS −17..−40%); D4 funding carry (fees > carry; forward-decayed); D5 risk-off reversion (bear confound); D6 pairs stat-arb (crypto pairs trend); D7 vol-climax reversion (−22%, buys into cascade); D8 breakout regime-gating (sign-flips); D9 ML-gated/meta-labeler (AUC 0.495–0.505 = null; ranker IC ~0); D10 RL/PPO (Sharpe −9.66); D11 within-cluster relval; D12 continuous-TSMOM (oracle-caught overstatement); D13 per-asset setup-chaser SELECTION LEAK (gated on UNSEEN>0; clean = below-chance 0.08–0.15); D14 WM-as-signal (~3 OOM below cost); D15 XS-momentum standalone (0/144; chases beta into crashes); D16 flow-surge (anti-edge clean); D17 chop-MR/bull-cont/rel-strength regime cells; D18 **DOGE whale-gate INVERTS** (PEPE's gate is anti-predictive on DOGE).
**🔴 INFRA BUG (landmine):** `strat.xsec_momentum.load_panel` silently floors any sub-daily cadence to daily (`floor('D')`) → every "4h" study importing it actually ran on DAILY data. Conclusions didn't flip on clean re-run, but the sub-daily search was shallow. Use native loaders.
**REUSABLE ASSETS (in archive — port, don't rebuild):** R1 `battery.py` (`evaluate`/`evaluate_setup_chaser`/`evaluate_portfolio`, Lens A/B/C); R2 `event_study_discriminator.py` + R3 `discriminator_null_calib.py` (discrimination + shuffle-null); R4 `u100_specialist_scan.py` (MA/EMA sweep × universe, 3 lenses); R5 `dollar_ladder.py` (dollar-coarseness sweep; the **n_eff↔jk3 tension law**: finer cadence ↑n_eff but ↓jk3); R6 the TI_ASSET methodology doc; R7 `kill_test.py` (cost-matched random-entry null firewall), `pooled_book_sim.py` (capital-constrained, replaces inflated equal-weight), `neighborhood_probe.py` (plateau vs spike), `decay_monitor.py`.
**LONE SURVIVOR (PEPE whale-gated slow-SMA, coarse dollar ~6676 bars):** UNSEEN +71.2% but **n=11, n_eff≈8, VAL monthly-positive only 36%**, resolution is a hidden hyperparameter (5500–6676 bars only), PEPE-idiosyncratic (inverts on DOGE), needs clean-data re-verify. Provisional, NOT ship.
**TIMEFRAME:** 4h was the empirical sweet spot for the breadth bounce (1h dead by cost, 1d too sparse) — but for per-asset chasers 4h-TIME was a blind wrong default; PEPE worked on coarse-DOLLAR (~4h-equiv by count), not 4h time bars.

### 7-E SOTA: bars / labeling / setup-trading — DONE (Wave 1)
- **Info-driven bars:** dollar bars are already the right choice; imbalance/run bars give only marginal, contested gains and are operationally unstable (exploding bar count). Don't over-invest in building all alt-bar chimeras. Possible mild upgrade: volume-weighted dollar bars.
- **Triple-barrier + meta-labeling (López de Prado):** directly applicable to the setup framework (TP=target move, SL=invalidation, vertical=max hold). MUST use **purged k-fold CV** + **ATR-scaled barriers**. Meta-labeling improves precision but only on a primary signal that already has edge.
- **Cross-sectional momentum:** established (30d lookback / 7d rebal); **long-only is better in crypto** (Han 2023: 85% of long-only specs positive); BUT net-of-cost shrinks sharply and it's CONTESTED (Grobys 2025 "is it an illusion?" for large-caps). **Best role: a UNIVERSE FILTER** (trade setups only in top-quintile recent performers), not a standalone strategy. Volume-weight the ranking.
- **Trend-following:** value is **drawdown-avoidance in bears, NOT bull capture** (Grayscale: 50d-MA Sharpe 1.9 vs 1.3, improvement ENTIRELY from 2018/2022 DD reduction; LOWER total return in 2020–21 bull). **Best role: a PORTFOLIO-LEVEL trend gate** (go flat when BTC/asset in confirmed downtrend). 28d/5d TSMOM a concrete start.

### 7-F SOTA: regime detection / ML / RL / WM — DONE (Wave 1)
- **Regime:** SMA-200 (price vs long MA) is the most durable ex-ante gate — parameter-free, century of OOS, cannot look ahead; ATR/vol bands also robust. **HMMs almost universally carry hidden look-ahead** (Viterbi uses the full sequence); BOCPD is theoretically correct but fragile in crypto. Use SMA-200/vol gate live; HMM only post-hoc.
- **ML:** standalone signal generators decay fast and fail live (LightGBM RankIC 0.072→0.010 across one regime shift); ~50% of high-in-sample models retain OOS (coin-flip). Meta-labeling **filters** false positives, does **not generate** alpha (and can have *lower* Sharpe than one end-to-end model). Use ML only as a meta-labeler/sizer on a first-principles edge.
- **RL/WM:** essentially **no credible published live-trading wins**; the **simulator-fidelity problem is unresolved/structural** (agents exploit the learned simulator's simplifications; LiveTradeBench: backtest-top agents did WORSE live). DreamerV3 is a games benchmark, not trading. Use the WM as a representation/predictor gated by OOS IC/ShIC (what the project already does) — NOT as a live policy.

### 7-G 🔶 ECL TENSION (flagged for the paradigm checkpoint)
SOTA literature is moderately POSITIVE on long-only trend/momentum in crypto (Grayscale, Han 2023, AdaptiveTrend long-only ~34%/Sharpe 2.12) — while the project's own dead-list is BRUTALLY negative on the same family (D15 XS-momentum 0/144; trend conditioners refuted). This empirical-contradicts-literature gap must be adjudicated (oracle, Wave 2): is the project's refutation an artifact (standalone-signal framing vs universe-filter role; the load_panel sub-daily bug; missing family-correction), or does the lit have look-ahead/cost-blind issues (Grobys 2025 says momentum is an "illusion" net of cost)? This determines whether the per-asset-specialist paradigm survives.

## 8. Agent-framework evaluation (META, running log)
- **21:25 SAST (T+0)** — runner protocol loaded & applied; wall-clock anchored; TIMED-RUN override resolves the "utilise the time vs honest-stop" tension cleanly. Lattice + GOAL_BOUNDS + frontier produced; TodoWrite is the live frontier.
- **21:41 SAST (T+16.5min)** — Wave 1 (6 Sonnet scouts, parallel) returned in ~16 min: complete infra map, feature taxonomy, a sharper-than-expected apparatus audit (DSR gate is a no-op; maker adverse-selection ~1.0), a full dead-list+reusable register (READ-FORWARD working as designed — prevents re-mining D1–D18), and two cited SOTA sweeps. **Framework verdict so far: ACTIVE and HIGH-YIELD.** Parallel Sonnet fan-out is the right tool for breadth; wall-clock grounding + write-forward both functioning. Refinement candidates noted for §final.
- **21:51 SAST (T+26.7min)** — Wave 2 returned: Opus oracle (paradigm reframe, §9) + opportunity/capture verification (§10). User EXPANDED the mandate mid-run: authorized to close gaps in own directive + project directives + all artifacts, gated by multi-skill CONSENSUS. Convening a consensus panel (auditor + validator, Opus) on the reframe before any artifact edits. Whiplash handled as a checkpoint per runner §4 (reconcile, don't silently flip).

## 9. 🔶 PARADIGM REFRAME (Wave 2 oracle — PENDING CONSENSUS, do not treat as settled)
**Oracle verdict:** the per-asset-specialist *cell* (`TI/gated-TI × ASSET × REGIME × TIMEFRAME`) as the **unit of discovery** is mis-framed and refuted-of-the-right-thing. Move the center of gravity from the CELL to the CROSS-SECTION.
- **Evidence:** the archived 2026-06-02 campaign ran the architecture's own named bottleneck (stage-5: find decoupled survivors to pool) with a CORRECTED firewall (tiered cost + cost-matched random-ENTRY null) → `2nd-EDGE SEARCH = ALL NEGATIVE` except ONE breadth-pooled effect. Supply ≈ 1–2, not the ~10 the pooling math needs. D13's below-random persistence (0.08–0.15 vs 0.25) = **anti-persistence** → the per-asset grid IS the leak surface. The lone PEPE survivor (n=11, n_eff≈8, VAL mpos 36%) was selected from ~6000 cells with the DSR gate disabled → most likely the multiple-comparisons ghost.
- **Center of gravity, ranked by robustness-adjusted wealth:** (1) **portfolio-level regime-gated diversified long-only book** (the "backbone," ~26% CAGR robust — and the SOTA lit's strongest result, drawdown-avoidance); (2) **cross-sectional pooled satellites** — incl. the ONE genuinely-untested node: long-only XS-momentum as a *universe FILTER* (Han 2023 spec) on the FIXED native-sub-daily loader (prior tests had the `load_panel` calendar-day bug); (3) leverage the backbone (a sizing dial, ruin risk); (4, LAST) per-asset cell mining.
- **Inverted unit:** `REGIME-GATE (whole-book on/off) × CROSS-SECTIONAL-MECHANISM (pooled ≥50 names) × CAPTURE-POLICY (enter-on-confirmation, time/trail stop, cut fast)`. Per-asset mining demoted to a **candidate-generator** for pooled mechanisms, never a ship unit.
- **Keep as GOLD:** discrimination≠harvestability epistemics; the cost-matched random-ENTRY null firewall (retire the no-op DSR-only gate); the capture-style discipline.
- **Most-likely-fails-again mode:** multiple-comparisons laundering through the AGGREGATION stage (cell-selection now gated, book-COMPOSITION un-gated). **Guardrail:** family-N must include aggregation DoF; pre-register pooling weights (equal/vol-parity) BEFORE touching held-out.
- **⚠️ CAVEAT (why this is PENDING, not adopted):** the BEAR case leans on ARCHIVED results the reset says to **re-test, not inherit**; a broken apparatus produces false *negatives* too. So the recommendation is to **re-test the key claims fresh with the fixed apparatus** — not to adopt the archived "dead" verdict on faith. What justifies shifting the PRIMARY bet + build order *now* is the *structural* argument (cell = a ~6000-wide multiple-comparison leak surface; cross-section/regime-gate has independent SOTA support), which holds regardless of the archived numbers.

## 10. Opportunity premise + capture ceiling (Wave 2 — evidence-grounded)
- **Premise is COMPUTED** (`analyze_daily_move_frequency.py` → `move_frequency_2025_2026.{md,json}`): 100% of days have ≥1 asset moving ≥5%; ~19.6/day ≥5% (u87, 2025); ~35/day ≥3%; regime-stable (all regimes ≥21% asset-days). CONFIRMED — but **close-to-close**, **survivor-universe**, and the "~20/day" drops to ~11 on a 60-asset universe.
- **Survivorship:** 17 assets purged; the upward bias on long-only backtests is acknowledged across docs but **never quantified** (no script measures the delta). Treat every long-only return as biased high.
- **UNSEEN reality (corrects an earlier assumption):** 2026 H1 is a **broad ALT BEAR** (equal-weight universe −13 to −36%; BTC bull-leaning) — NOT a clean bull. Plus a split-discipline ambiguity: "UNSEEN 2025-03-15+" actually spans OOS+UNSEEN (~14 months). Good news: a held-out that includes a bear CAN test drawdown-avoidance (the backbone's mechanism).
- **Capture ceiling (the gap that matters):** perfect-top-1 ≈ **+15.9%/day**, perfect-top-5 ≈ +9.6%/day. Best HONEST realized ≈ **+0.44%/day ≈ 2.8% of the perfect-top-1 ceiling**; best OOS-enhanced ≈ 5–6% mover-hit at Sharpe ~2.4–2.6. Published systematic-arb ceiling ≈ 5–15% of perfect. **The moves exist; net-of-cost capture is ~30–50× smaller.** This quantifies failure-point #2 (existence ≠ harvestability).

## 11. ⚖️ CONSENSUS VERDICT on the reframe (oracle + auditor + validator)
> **🟢 GROUND-ZERO RECALIBRATION (user, 2026-06-04 ~T+77min):** nothing in this section is a settled FACT.
> We are at GROUND ZERO — no approach, no edge, **no ceiling** is established; prior work is EXPERIENCE, not a
> binding conclusion. The apparatus that produced these reads was BROKEN (so it can produce false NEGATIVES too),
> which means even *"it's all beta / ~13–22% ceiling"* is a **HYPOTHESIS TO RE-TEST with the fixed apparatus**, not
> a verdict. The "A/B fork" below is therefore a **RESEARCH-PRIORITY question** (what to re-test first), NOT a
> strategic surrender to beta. The only fixed points are the founding frame (setups-across-moves, IC banned,
> LO+spot+lev=1, WEALTH, beat-lazy+beat-buy-hold) and the apparatus/methodology we are building.
> **Autonomous envelope updated: 8 hours from ~22:45 SAST 2026-06-04 → ends ~06:45 SAST 2026-06-05.**

Convened per the user's consensus mandate. The panel AMENDED the oracle (the auditor + validator independently caught it over-reaching). Read every point below as a **prior-experience hypothesis to re-test**, not a fact:
1. **Build-order: center the FIRST bet on the regime-gated portfolio; demote per-asset cell-mining — APPROVED (3/3).**
2. 🔴 **The "backbone" is BETA + YIELD, NOT alpha.** Project's OWN decomposition: ~+17.4pp regime-timed beta + ~5pp yield + **0 timing-alpha** (the kill_test literally labels the timing layer "BETA-IN-DISGUISE"). Honest FORWARD CAGR ≈ **13–22%** (the "~26%" was a bull-weighted full-cycle seed set + optimistic all-stakeable yield). DD<30% is regime-conditional (−31..−41% through the 2022 mega-bear). **BTC buy-and-hold BEATS it in every bull** (+174% vs +25–48% in 2023) — its only genuine value is drawdown-control / bear-preservation, NOT return-maximization.
3. ❌ **REJECT "per-asset is dead" as FACT.** It violates the reset (re-test, don't inherit), AND the apparatus that produced the negatives was broken (the `load_panel` sub-daily→daily bug, the no-op DSR gate, maker-not-taker cost) — a broken apparatus produces false NEGATIVES too. Per-asset = **UNPROVEN, not refuted.** Unreconciled in the same archive: a leak-free MAJORS book (BTC/ETH/BNB/PEPE) **SURVIVED at +81.1% UNSEEN / 4-of-4 positive / 30bps** vs the below-chance broad scan → genuinely open.
4. Cross-section satellite is **THIN + weakly-confirmed** (bear-robustness rests on a 1-month/15-trade holdout; the thr25 "beats-null-not-beta" proof is not artifact-verifiable; it dilutes wealth vs backbone-alone).

**🔴 THE LOAD-BEARING TRUTH (honest-failure surface — must NOT be silently reframed):** at daily / 4h / dollar-bar resolution, on LO+spot+lev=1, the corpus shows **NO verified active-trading alpha — everything robust is beta + yield.** The original goal ("enter, leave, profit" active trading; +1–5%/day) is **UNPROVEN at this resolution.** Centering on the backbone is, honestly, a *concession that the active edge has not been found here* — not the discovery of one.

**THE FORK (user's decision — surfaced, not resolved):**
- **Fork A — accept the beta+yield ceiling (~13–22%/yr)** as the deployable base. Robust, but it's drawdown-control beta, not active trading, and a wealth-max objective must weigh that buy-hold beats it in bulls.
- **Fork B — pursue the genuinely-UNPROVEN active-alpha frontier** with the FIXED apparatus: (i) re-test the per-asset MAJORS book fresh (the +81% leak-free thread is the one unresolved candidate); and/or (ii) go to finer resolution (sub-day / tick) where the corpus suggests alpha may live (orthogonal to this work).

**MANDATORY before adopting ANY verdict (consensus condition):** fresh re-test with the FIXED loader (correct cadence), TAKER baseline + maker p_fill∈[0.25,0.50] sensitivity, family-N/DSR correction *including aggregation DoF*, a PRE-REGISTERED mechanical universe, and a **BEAR-INCLUSIVE holdout (2022 + 2026)** — and explicitly reconcile the +81% leak-free majors book vs the below-chance broad scan.

**Leak-free majors thread — verified & RECONCILED (2026-06-04, T+85min):** the one unresolved "+81% leak-free majors book" was forensically characterized → **DEPRIORITIZED.** The +81% is PEPE-driven (+48% on n=19, on the *broken dollar-bar loader*); strip PEPE → ~+22% (overlaps the beta book). No mechanism portability (4 different TI/conditioner/clock combos), the asset universe was pre-selected, no family-N, and the matching broad clean scan is **below chance (0.08–0.15 vs 0.25)** — i.e. 4 lucky draws from a below-chance universe. Selection code itself is structurally clean (TRAIN+VAL+OOS select, UNSEEN held out once), and the BTC/ETH 4h sleeves (+5/+8%, NOT on broken data) are the only sub-threads not clearly noise. **Net:** the per-asset-majors sub-option of Fork B is LOW priority; if Fork B is pursued, the higher-EV path is finer-resolution (sub-day/tick), and the only per-asset question worth a clean isolated check is "does BTC/ETH trend@4h add anything incremental over the beta book?" (answer via `market_replay`, not another scan). Sources: `leak_free_harvest.log:12-15`, `pa_clean_shard*.json`, `pa_mechtest.json:53`, `kill_test.json:17`.

## 12. Dead-list triage — VERIFIED at ground zero (2026-06-04, T+101min)
A forensic re-examination of all 18 dead-list items (D1–D18) against their actual test code / loader / cadence / cost: **ZERO items RE-OPEN.** This is the ground-zero "re-test, don't inherit" process run to completion on the negatives — and they hold.
- The 3 most `load_panel`-sensitive studies — D7 vol-climax, D11 within-cluster relval, D12 TSMOM@4h — were EACH either re-tested with a native loader and stayed dead (the bug was caught and corrected in-place for D7/D11), or had an identical clean 1d result (D12). Verified, not assumed.
- **Directional insight (important):** the apparatus bugs are **FALSE-POSITIVE generators, not false-negative ones.** The maker-not-taker cost bug is OPTIMISTIC — anything dead at 0.0010 is deader at 0.0024; the no-op DSR gate shipped false POSITIVES (the +36.8% leak). **So fixing the apparatus matters for trusting FUTURE POSITIVES, not for re-opening past negatives.**
- **Therefore the ground-zero state is asymmetric:** prior NEGATIVES are largely trustworthy (they died for structural reasons — cost walls, mechanism laws, regime decay, overfit); prior POSITIVES (the "backbone-as-alpha," the +81% majors) are the ones that were apparatus-inflated and need fresh, trustworthy validation.
- Only loose thread: a ~15-min housekeeping check of the April-2026 DIB BTC+ETH sleeve at taker cost (likely survives at lower Sharpe ~2.5; not frontier-best). Not a re-mine.

**Implication for the fork (sharpened):** Fork B's "re-test the dead daily/4h/dollar veins fresh" is **LOW EV — those veins are genuinely closed.** The genuinely UNEXPLORED, highest-EV-for-active-alpha frontier is **finer-resolution information-driven bars** — `dib` / `runs_tick` / `runs_volume` / `adaptive_vol` + fine dollar (~75s/bar) — which EXIST as raw bars for 87 assets but were **never enriched into chimeras** (only BTC/ETH/PEPE for dib/range; runs/adaptive_vol chimeras are EMPTY) and so were **never properly mined under any apparatus.** That substrate, plus the tick/representation (WM) path, is where Fork B should point. The daily-bar dead-list is not the frontier; the un-built fine-resolution bars are.

## 13. Agent-framework evaluation (the explicit meta-deliverable)
**Verdict: ACTIVE, USABLE, HIGH-YIELD — and it self-corrected, which is the strongest signal.** In ~100 min it produced a full infra/feature/apparatus map, a verified dead-list triage, two SOTA sweeps, a paradigm reframe, and a 3-skill consensus that caught the reframe's own errors.
**Worked as designed:**
- **Wall-clock grounding** — 5 re-`date`s, zero fabricated elapsed (the 2026-06-03 failure did not recur).
- **READ-FORWARD dead-list** — prevented re-mining D1–D18; the triage then verified they're genuinely closed.
- **n±k lattice** — the **−2 falsifier** ("is the paradigm mis-framed?") was the single highest-value node, exactly as the runner predicts.
- **Parallel Sonnet fan-out** — 6 scouts in ~16 min; the right tool for breadth.
- **Consensus mechanism** — a single Opus oracle over-reached (cherry-picked, ignored a contradicting result, laundered beta as alpha, inherited archived conclusions); the audit+validator panel caught all four. **One agent unreliable on a load-bearing reversal; the panel reliable.**
- **Write-forward + TIMED-RUN override** — durable docs each cycle; "utilise the time vs honest-stop" resolved cleanly.
**Refinements APPLIED this run** (the act→observe→feedback loop improving itself): AUTONOMOUS_RUNNER §6 (mandatory consensus for whiplash; archived-conclusion→RE-TEST tag; broken-apparatus→false-negative flag); STANDARDS rule 13 (non-linear self-improving, inherited by all skills); discover+trader stale-path banners.
**Residual refinement candidates (proposed):** (1) a CDAP rule that skill file-references must resolve (the reset archived `src/strat/` but left skills pointing at it — a "no-surprises" class of bug); (2) apparatus-trust as a hard Phase-0 gate before any claim (now in RETEST_PLAN); (3) consider a dedicated `verify`/`re-test` sub-skill if the re-test phase recurs enough to warrant it.

## 14. Current state & turnkey next actions
**State (ground zero, verified):** dead-list closed (structural); proven floor = beta+yield ~13–22% (needs honest re-validation, is NOT alpha); no verified active alpha at daily/4h/dollar; apparatus has 3 holes (specced). Companion docs: [`APPARATUS_LOCKDOWN_SPEC`](APPARATUS_LOCKDOWN_SPEC_2026_06_04.md), [`RETEST_PLAN`](RETEST_PLAN_2026_06_04.md).
**Turnkey next:** Phase 0 (fix apparatus — the cost/simulate-path change is trust-critical → SUPERVISED; the additive leak-probe + random-entry-null can be built safely) → Phase 1 (honest base re-validation, β-matched) → user picks Fork A/B/C.
**Awaiting user:** the A/B/C fork (sets the build order; foundation supports all three).

## 15. Phase-1 honest benchmark — FRESH ground-zero confirmation (2026-06-04, RWYB)
Computed fresh on real BTC 1d at TAKER cost (0.24% round-trip): 200-day-MA regime gate vs buy-and-hold vs exposure-matched static hold, per window incl. the 2022 bear. **This is a ground-zero VERIFICATION (not an inherited claim).**

| Window | buy&hold | MA-gated | β-matched static | avg_exp | BH maxDD | gated maxDD |
|---|---|---|---|---|---|---|
| TRAIN (incl 2022 bear) | +546.3% | +574.2% | +274.4% | 0.61 | −76.6% | −64.2% |
| VAL (bull) | +36.3% | +3.8% | +28.6% | 0.75 | −26.0% | −26.5% |
| OOS (chop) | +5.4% | −0.4% | +5.0% | 0.68 | −32.1% | −19.7% |
| UNSEEN (2026 alt bear) | −17.0% | +0.0% | +0.0% | 0.00 | −35.1% | +0.0% |

**The timing-alpha test (MA-gated vs exposure-matched static):** TRAIN gate +574% > static +274% (timing added value — but entirely from dodging the 2022 bear, + lower DD); VAL gate +3.8% ≪ static +28.6% (timing DESTROYED value); OOS gate ≈ static (no return-alpha, some DD benefit); UNSEEN gate fully-cash 0% vs buy-hold −17% (pure bear-avoidance). **Conclusion (empirical, fresh, taker-cost):** the regime gate's entire edge is **drawdown-avoidance at major bear turns**; in bull/chop it LOSES to lazy. It beats buy-hold in total return only when a big bear is in-window. Independently re-confirms "backbone = drawdown-control beta, NOT timing-alpha," and operationalizes the benchmark every active strategy must beat: *in bull/chop the bar is buy-hold (hard); the gate's only honest edge is bear-avoidance.* Caveats: single asset/param, illustrative β-match (static avg-exposure). Script: `runs/staging/benchmark_2026_06_04.py`.

**Portfolio version (5th confirmation, equal-weight majors BTC/ETH/SOL/BNB/XRP, daily-rebalanced, taker):** TRAIN(incl bear) gate +1707% > β-matched +760% (timing adds via bear-dodge, DD −52% vs −79%) but < buy-hold +1988%; VAL gate +18.4% ≪ β-matched +32.6% ≪ buy-hold +44.6%; OOS gate **−1.8%** < β-matched +12.4% < buy-hold +16.5%; UNSEEN(2026 alt-bear) gate **−0.4%** vs buy-hold **−27.4%** (avg_exp 0.02 → went to cash). **Honest Fork-A floor: capital-PRESERVATION through bears (−2% vs −15% over OOS+UNSEEN), NOT return generation; loses to lazy in bull/chop.** Script: `runs/staging/benchmark_multi_2026_06_04.py`.

## 16. Apparatus pieces BUILT + RWYB-validated this run (STAGED for integration)
Two additive apparatus tools built + validated on real data (staged per the integrate-at-end directive; they do NOT touch the canonical harness path):
- **Leak-probe (LD-2)** `src/wealth_bot/leak_probe.py` — DISCRIMINATES (leaked control 85.6pp ≫ legit 33.2pp), but RWYB revealed the fixed-pp thresholds are cadence-mis-calibrated (false-positive on 1d); verdict tagged ADVISORY; the shift-spectrum discontinuity verdict is the corrected design (supervised).
- **Random-entry-null firewall (LD-4)** `runs/staging/random_entry_null_2026_06_04.py` — the consensus "THE gate," generic harness-wrapping version (kill_test was bespoke). VALIDATED + a FRESH bonus confirmation: BTC 1d R12 (WMA whale-gated) does NOT beat random entries in ANY window (TRAIN +135% vs null p95 +522%; OOS/UNSEEN below null p95) → **BETA-IN-DISGUISE, a THIRD independent method agreeing with the consensus + the benchmark.** (Caveat: unrestricted-random null is harsh for a regime-gated strat; a regime-matched null is the fairer-stricter variant, noted in the module.)
**Three independent methods now agree** the proven configs are beta, not timing-alpha: the consensus panel, the Phase-1 benchmark (§15), and the firewall (§16).
- **FillModel (LD-1)** `runs/staging/fill_model_2026_06_04.py` — post-hoc cost+fill realism (additive). RWYB on BTC R12: `ideal_ref` (old 0.10%/100%-fill) reproduces the harness default (gross-recovery correct); **taker (0.24%) cuts TRAIN 135%→82% and flips OOS −1.6%→−5.7%** (the silent inflation the 0.10% default hid); **maker_pessimistic (p_fill 0.30, adv 0.96) collapses to ~0/neg in every window — empirically confirming maker execution is dead for us** (a 4th independent agreement). Calibration of maker p_fill/adv is PROVISIONAL (flagged).
**Four independent methods now agree** the proven configs are beta, not timing-alpha: consensus panel, Phase-1 benchmark (§15), the firewall (§16), and the FillModel taker/maker stress (§16).
**Apparatus state:** cost-realism (FillModel) ✓, beta-in-disguise gate (firewall) ✓ — the consensus's PREFERRED gate, which **retires the no-op DSR-only gate** (so DSR-fix is DEPRIORITIZED, not pending). Remaining (supervised): the leak-probe relative-twin verdict (needs a suspect candidate to validate against), the firewall regime-matched-null variant, and wiring both into the candidate pipeline.

## 17. Fine-resolution frontier — CHARACTERIZED (Fork-B decision-support, 2026-06-05)
Probed dollar bars (finest substrate, ~30–90s/bar) for exploitable structure + cost-clearing fuel — the triage's "one genuinely unexplored frontier," explored before the user commits to Fork B:
- **Majors are EFFICIENT** (BTC lag-1 autocorr −0.014, SOL −0.013): no cheap linear structure; cost-clearing MFE-fuel only at long holds (BTC K=60 → 24% of bars >0.72% MFE; ~0 at K=1–5). At long holds fine-res buys nothing over coarse bars (same wall-clock). → **Fork B on majors = no-man's-land** (the cost-wall that killed 1h/15m).
- **Memecoins show structure + fuel** (PEPE lag-1 −0.10; 17% of bars clear cost in ONE bar; 32% have >0.72% MFE within 20 bars). This is where Fork-B promise concentrates.
- **🔴 The trap:** PEPE's −0.10 lag-1 autocorr is most likely **bid-ask bounce / microstructure noise** (price oscillating across the spread → spurious negative autocorr), NOT harvestable reversion — buying it = paying the spread. Prior dead-list D3 (order-book flow) and D7 (vol-climax) found fine-res microstructure reversion **catastrophic / knife-catching.** **Fuel ≠ harvestability.**
- **Net Fork-B read:** surface promise only on high-vol memecoins, where the structure is most likely a microstructure cost-trap → a LOW-probability, cost-wall-bound bet, consistent with the consensus's "least-bad remaining frontier." Not a clear win. Script: `runs/staging/fine_resolution_probe_2026_06_04.py`.

## 18. Verification frontier EXHAUSTED — every prior "positive" freshly tested (2026-06-05)
All prior positive claims re-verified fresh under the fixed apparatus; **none is a robust active edge:**
- **Backbone (regime gate)** → beta / drawdown-control (benchmark §15 + firewall §16 + portfolio §15).
- **Leak-free majors +81%** → PEPE-driven on the broken dollar loader; broad clean scan below-chance (forensic).
- **Breadth RSI-bounce satellite** → firewall (fresh, 10 assets, 4h, taker): beats the random-entry null on OOS (real_exp **+1.77%** > null p95 +0.37% — confirms the prior "+6.6% OOS" was real THERE) but **FAILS UNSEEN** (real_exp **−0.31%**, negative, below null) → **OOS-regime-luck, NOT robust held-out.** `runs/staging/breadth_firewall_2026_06_04.py`.
- **Fine-res frontier** → majors efficient; memecoin "structure" likely bid-ask noise (§17).
**CONCLUSION (airtight, 6 independent verifications):** at daily/4h/dollar resolution under LO+spot+lev=1, there is **NO verified robust active-trading alpha.** The only robust thing is beta + yield (capital-preservation). The verification frontier is exhausted; **the A/B/C fork is the genuine remaining decision (user's call).**

## 19. Fork-A PROVEN FLOOR — battery-tested (2026-06-05, Phase 1 build-out)
Built the regime-gated majors book properly and ran the robustness battery (per-window + maxDD + block-bootstrap p05 + SMA-length lone-peak check):
- **v1** (BTC-whole-book gate, EW majors): **FAILED** — maxDD −54%, block-bootstrap p05 −49.7%, SMA-param-fragile (OOS +10.9% → −4.5% across 150/200/250).
- **v2** (per-asset 200-MA gate + vol-target vt0.015 + 3% yield): vol-targeting **FIXES drawdown → maxDD −17% (passes the <20% binding gate)**, BUT held-out (OOS+UNSEEN) compound only **+1.1%** with **block-bootstrap p05 −16.7% → FAILS p05>0**; avg exposure 0.19 (heavy de-lever, capital-inefficient).
- **Honest Fork-A verdict (battery-tested):** the proven floor is a **drawdown-control beta vehicle (maxDD −17%) that does NOT robustly generate positive return forward (p05<0)** — "don't lose much," NOT "reliably make money." The "~13–22% CAGR" was bull-weighted full-cycle; recent held-out is ~flat with a negative tail. (Did NOT over-tune target_dvol to force p05>0 — that would fit the held-out.) Scripts: `runs/staging/honest_base_book_2026_06_04.py` (v1), `..._v2_2026_06_04.py` (v2).
- **Sharpens the fork:** even Fork A only PRESERVES capital forward → the choice is starker: A (preserve, ~flat) / B (chase low-prob fine-res active alpha) / C (tick, high-cost).

## 20. Fork-B genuine attempt — lone survivor re-tested with FIXED apparatus (2026-06-05)
Re-tested the prior "lone survivor" (PEPE whale-gated slow-SMA, coarse dollar ~6676 bars) under TAKER 0.24% + the firewall + per-window incl. bear (never freshly re-tested before):
- **Whale-gated:** TRAIN +88.7% / VAL −5.8% / OOS +11.0% / UNSEEN +58.0% (n=11, DD −5%, win 73%). all_4_positive = **FALSE** (VAL neg).
- **Mechanism falsifier PASSES decisively:** bare SMA no-gate = TRAIN +35 / VAL −47 / OOS −34 / UNSEEN −19 → the whale gate lifts EVERY window. The gate is a **real discriminator**, not noise.
- **Firewall:** beats the random-entry null on **UNSEEN** (+58% > null p95 +32%) but **NOT on OOS** (+11% < null p95 +105% — random full-participation in the OOS bull beat being selective). Verdict: fails held-out.
- **Honest Fork-B read:** the best existing active candidate has a REAL mechanism + bear-regime value + low DD, but is THIN (n=11), regime-dependent (bear-value only), PEPE-specific, and FAILS the strict gates (all-4-positive + firewall-on-OOS). **DISCRIMINATION ≠ HARVESTABILITY, re-confirmed fresh on the lone survivor.** Fork B yields NO ship-tier edge on existing data; the whale-gate's bear-value is the one thread worth a supervised deepening if Fork B is chosen. Script: `runs/staging/forkb_pepe_retest_2026_06_04.py`.

## 21. Fork-B generalization — whale-gate is PEPE-IDIOSYNCRATIC, NOT poolable (2026-06-05)
Tested the whale-gated slow-SMA across 6 memecoins (PEPE/DOGE/SHIB/FLOKI/BONK/WIF), taker + per-asset firewall + falsifier:
- ONLY PEPE is UNSEEN-positive (+58%) AND beats-null. DOGE UNSEEN −22% (fails — confirms D18 inversion), FLOKI −7%, BONK −26%, WIF −26%, SHIB 0 trades.
- **POOLED UNSEEN = −37.2%** (n=27, mean/trade −1.32%) — PEPE's +58% is swamped by the others' losses → the gate does NOT pool.
- **Decisive:** the lone survivor is PEPE-idiosyncratic, thin (n=11), non-replicating, **non-poolable → NOT a deployable edge.** PEPE's +58% is most likely PEPE-specific microstructure/luck, not a generalizable mechanism. Script: `runs/staging/forkb_whale_generalize_2026_06_04.py`.
- **AIRTIGHT combined fork picture:** Fork A = DD-control beta (preserves; p05<0 forward, no robust profit); Fork B = no poolable active edge on existing data. The binding issue keeps returning to the **CONSTRAINT** (LO+spot+lev=1) — the genuine paths to active WEALTH are a constraint change (leverage/shorts — risk conversation) or new data/resolution (Fork C tick), and tick active-alpha typically NEEDS the very tools the constraint forbids. Both are user decisions. (Fork-C feasibility researched next to verify, not assert.)

## 22. DEFINITIVE CONCLUSION — the binding limit is the CONSTRAINT, not the resolution (2026-06-05)
Fork-C (tick) researched (SOTA) + repo-recon'd:
- **Fork C is infeasible under LO+spot+lev=1 + retail infra:** every tick edge (latency-arb, LOB-imbalance, maker-rebates, MEV, triangular) requires a FORBIDDEN tool — shorts, leverage, market-maker volume-status ($10–100M/mo), or colocation/low-latency — or is competed away before retail arrives. The one long-only-compatible tick signal (LOB-imbalance) is eaten by 0.1% taker at sub-second holds and is decaying to ~0 even for perp/low-latency operators.
- **And it's a months-long build from scratch:** NO usable tick/LOB data on disk (aggTrades ≈ 3ms *tape*, NOT order-book; bookTicker dir empty; LOB collector built-but-never-run, `data/lob/` empty; depth = 30s cadence). V20 does NOT exist (no `src/wm/v20/`). A tick system = new data acquisition + V20 + tick pipeline + sub-second execution.
- **🔴 The binding limit is the CONSTRAINT (LO+spot+lev=1), not the resolution.** Research on comparable data: relaxing to LONG-SHORT on EXISTING daily/4h data ≈ **2× returns (224% vs 108%) + half DD (62% vs 84%) + Sharpe 1.18→1.96** on the SAME signals — infra-cheap, research-backed. Tick-under-constraint is high-cost + infeasible.
- **THE fork, reframed:** the highest-EV path to active WEALTH is a **CONSTRAINT decision** — relax LO+spot+lev=1 toward long-short / modest perps on existing data (accepting ruin-risk: a risk conversation) — NOT tick (Fork C, infeasible) and NOT more LO-spot mining (exhausted). Reconnects to post-mortem #1 (target⊥constraint) and #4 (long-only=beta). Quantifying the relaxation EV on OUR data next (decision-support, NOT a deployable LO candidate).

## 23. Constraint-relaxation on OUR data — naive long-short is NOT a freebie (2026-06-05)
Tested relaxation (regime book but SHORT the bear vs LO long-or-cash), majors, vol-targeted, taker + funding:
- **LO:** TRAIN +224 / VAL +4.4 / OOS +6.4 / UNSEEN +0.8 ; full DD −19%, held p05 −16%.
- **LS:** TRAIN +95 / VAL −9.7 / OOS −6.3 / **UNSEEN +11.9** ; full DD **−29%**, held p05 **−32%**.
- The bear-capture is **REAL** (UNSEEN +11.9% vs +0.8% — shorting the 2026 bear works), BUT the naive sign-flip long-short is **WORSE everywhere else** (loses the bull/recovery: TRAIN +95 vs +224, OOS −6.3 vs +6.4) AND **riskier** (DD −29% vs −19%, p05 −32% vs −16%).
- **Honest verdict:** the generic "long-short ≈ 2× returns" does NOT replicate with a NAIVE regime-sign-flip on our data — it trades bull-capture for bear-capture + adds risk. The relaxation lever is REAL (bear-capture) but NOT free: capturing it net-positive needs a SOPHISTICATED long-short (cross-sectional) + ruin-risk tolerance. (Cross-sectional version tested next.) Script: `runs/staging/constraint_relax_longshort_2026_06_04.py`.

## 24. Constraint-relaxation COMPLETE verdict — does NOT deliver the generic "2×" on our data (2026-06-05)
Tested the sophisticated long-short (cross-sectional momentum, long top-3 / short bottom-3 majors — the research's "2×" construction):
- **XS-LS momentum:** TRAIN +74 / VAL −12 / OOS −3.6 / UNSEEN −25.8 ; held-out **−28.4%**, maxDD **−49.6%**, p05 **−44.2%**, worst_day **−21.7%**. FAILS badly. Script: `runs/staging/constraint_relax_xsec_ls_2026_06_04.py`.
- **Combined relaxation verdict (naive-LS §23 + XS-LS §24):** relaxing LO→long-short does NOT replicate the generic "≈2× returns" on OUR majors. Naive regime-LS captures the bear (UNSEEN +11.9%) but loses the bull + is riskier; XS-LS-momentum is negative held-out with −50% DD + a −22% squeeze day. Crypto majors are too BTC-correlated for a cross-sectional spread; the bear-capture benefit is real but **gated by ex-ante regime-detection quality** (hard — HMMs look-ahead, §7-F).
- **🔴 AIRTIGHT bottom line across the WHOLE run:** under LO+spot+lev=1 OR relaxed-to-long-short, on existing daily/4h data, there is **NO robust active-wealth engine.** The LO floor PRESERVES (no robust forward profit); long-short relaxation does NOT deliver + adds ruin/squeeze tails; tick (Fork C) is infeasible under the constraint. **The one honest deployable is the LO preservation floor** (beta+yield, maxDD −17%, ~flat-forward) — harvest the secular bull when it comes, preserve through bears. Robust ACTIVE wealth is not demonstrable on this data/resolution with these methods. That is the ground-zero truth, RWYB-verified ~8 ways.
