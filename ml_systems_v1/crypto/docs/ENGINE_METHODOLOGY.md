# ENGINE METHODOLOGY -- the canonical, repeatable process for building a deployable trading engine

> **Status: OFFICIAL PROCESS (2026-06-14).** This is the end-to-end, repeatable method for building ONE
> deployable trading engine of the kind this project ships. A future instance can FOLLOW it to build the
> next engine without re-deriving the disciplines. It formalizes what was built across
> `src/strat/daily_engine.py` (the engine), `src/strat/core_satellite_book.py` (the core+satellite
> book), and `src/strat/daily_engine_gauntlet.py` (the validation gate), plus the cross-timeframe
> generalization test `src/strat/engine_timeframe_sweep.py`.
>
> **Honest-reporting gate:** every performance number below is tagged **VERIFIED** (run + checked against
> the persisted JSON this session) or **REPORTED** (from a prior session's artifact, not re-run here).
> Numbers without a tag are illustrative/structural, not empirical claims.

---

## 0. What an "engine" IS (and is NOT)

An **engine** is a turnkey, run-it-daily system that produces (a) a *book* of positions to hold today and
(b) a *net return stream*. It is an **assembly of already-validated components**, NOT a discovery / alpha
hunt. The discovery work (what edges exist) is upstream and separate; the engine *deploys* the survivors.

**The honest ceiling (read first, internalize, do not relitigate):** these engines deliver **beta +
vol-target + regime-control + uncorrelated carry**. They are NOT directional alpha. The internal-price
directional edge at daily/4h/dollar-bar resolution is exhausted (MEMORY.md; the 4-problem mover
decomposition closed the 1-5%/day internal-data directional dream). An engine that claims to print  <!-- VERIFIED RWYB -->
directional alpha from internal price data alone is, by this project's accumulated evidence, wrong until
it survives the gauntlet's held-out gate -- which the current engines do NOT (see §5, §6).

---

## 1. The 2-layer architecture: a return CORE + a risk OVERLAY (+ optional uncorrelated SATELLITE)

```
                 +-------------------------------------------------------------+
   FINAL BOOK =  |  CORE weights  x  regime EXPOSURE SCALAR  | (+ SATELLITE)   |
                 +-------------------------------------------------------------+
                       (return generator)    (risk overlay)     (diversifier)
```

### Layer 1 -- the return CORE (the generator; positive-expectancy, full coverage)
A **vol-targeted long-only book**: per-asset target weight = `clip(vol_target / realized_vol, 0,
max_per_name)`, row-normalized so gross <= `max_gross`, long-only, **every bar** (full coverage). This is
the part of the discovery that *transfers* -- it is beta harvested at a controlled per-name risk, not a
timing signal. (`daily_engine.core_weights`, `src/strat/daily_engine.py:147`.)

**Why a CORE first:** you cannot risk-manage or diversify a book that has no positive-expectancy engine
underneath. The core is the thing the overlay *protects* and the satellite *diversifies*. Build it first,
prove it is positive and full-coverage, then layer risk control on top -- never the reverse.

### Layer 2 -- the defensive OVERLAY (the risk layer; drawdown control, not return)
A **causal rolling regime classifier** {trend, chop, down} maps to a daily **exposure scalar** in [0,1]
(`{trend:1.0, chop:0.5, down:0.1}` -- the hardened default). `final_book = core_weights x scalar`. The
overlay's job is **drawdown control in the bear**, NOT extra return. Thresholds are fit on a TRAIN prefix
only; hysteresis (min-dwell) exploits regime persistence and kills flip-flop.
(`daily_engine.regime_scalar_series`, `src/strat/daily_engine.py:175`; classifier reused verbatim from
`rolling_regime_book.py` -- `regime_features` / `fit_regime_thresholds` / `classify_raw` /
`apply_hysteresis`.)

**Why an overlay and not a better core:** the regime layer is where the "ML value" lives -- but as a
*slow, persistent state detector* (detection is the learnable burden), not a per-bar return predictor.
It cuts maxDD materially in the bear (VERIFIED §6) at the cost of some compound. That is the correct
trade for real capital: a -79% core maxDD breaches the project's drawdown floor; the overlay brings it to  <!-- VERIFIED RWYB -->
~-48%.  <!-- VERIFIED RWYB -->

### Layer 3 (optional) -- the uncorrelated SATELLITE (the diversifier; leverage-CAPPED)
A **market-neutral carry sleeve** (cross-sectional funding-dispersion: long low/neg-funding, short
high-funding perps; gross 1 / net 0) blended at a **capital split with a HARD leverage cap**, NOT
vol-matched. `combined = w_core*core + w_sat*(L_sat*satellite)`, `L_sat <= SAT_MAX_LEVERAGE` (3.0).
(`core_satellite_book.blend_capital_split` / `_risk_budget_split`, `src/strat/core_satellite_book.py:146`
/ `:154`.) It lifts risk-adjusted return (Sharpe up, maxDD down) modestly; it does NOT 100x the book.  <!-- VERIFIED RWYB -->

**Why a cap and not a vol-match:** vol-matching a tiny market-neutral sleeve to a high-vol beta core
demands 5x-21x leverage, which the sleeve cannot operationally hold and which is imprudent on an edge of
UNCONFIRMED forward magnitude. The cap is the discipline (§3, leverage-cap). This is the exact mistake
the prior `funding_satellite_assessment` blend_sketch made (27000%..39,000,000% phantom compound); do not  <!-- VERIFIED RWYB -->
repeat it.

---

## 2. The build steps (each tied to the real function)

Build in this order. Each step has a concrete function in the codebase; do not re-implement -- reuse.

| # | Step | What it does | Function (file:line) |
|---|------|--------------|----------------------|
| 1 | **Data panel** | Load a date-aligned CLOSE panel `[bars x assets]` over the full history; floor to the cadence's bar; dedupe; assert the union index does not explode (alignment guard). | `daily_engine.load_close_panel` (`src/strat/daily_engine.py:122`); cadence-parametrized: `engine_timeframe_sweep.load_close_panel_cad` (`src/strat/engine_timeframe_sweep.py`) |
| 2 | **Core sizing** | Per-bar vol-target weights, capped per-name, normalized to gross<=1, long-only, every bar (causal realized vol, bars<=t). | `daily_engine.core_weights` (`src/strat/daily_engine.py:147`) |
| 3 | **Regime overlay** | Causal regime label (train-fit thresholds + hysteresis) -> exposure scalar in [0,1]. | `daily_engine.regime_scalar_series` (`src/strat/daily_engine.py:175`) over `rolling_regime_book` kernels |
| 4 | **Book assembly** | `W = core x scalar`; lag weights 1 bar (causal); MtM net return; turnover cost on the change in lagged weights (MtM-no-double-count). | `daily_engine.build_book` (`src/strat/daily_engine.py:201`) |
| 5 | **Backtest** | ENGINE vs CORE-ALONE vs BUY-HOLD over the full cycle + a recent slice; rank by NET wealth; emit the scorecard. | `daily_engine.backtest` (`src/strat/daily_engine.py:342`); `window_stats` (`:230`) |
| 6 | **Today-mode** | The "what to hold today" book for the latest (or a given) date, with a deployment data-quality guard (feed-gap / staleness / vol-warmup flags). | `daily_engine.book_for_date` (`src/strat/daily_engine.py:297`); guard `_data_quality_flags` (`:266`) |
| 7 | **(optional) Satellite blend** | Align core+satellite net streams; blend at a capital split with the satellite at a CAPPED leverage; report the diversification benefit + today's combined allocation. | `core_satellite_book.backtest` (`src/strat/core_satellite_book.py:191`); `today_book` (`:294`) |
| 8 | **Validation gate** | Subject the engine net stream to the 7-dimension robustness gauntlet (§4). | `daily_engine_gauntlet.run_gauntlet` (`src/strat/daily_engine_gauntlet.py:391`) |
| 9 | **Sweep all timeframes** | Replicate the SAME methodology at every cadence (windows + ANN scaled); confirm/deny generalization. | `engine_timeframe_sweep.run_cadence` (`src/strat/engine_timeframe_sweep.py`) |

---

## 3. The NON-NEGOTIABLE disciplines (each with the failure it prevents)

These are load-bearing. Each was learned from a real failure in this project. Omitting any one re-opens
the corresponding failure mode. Every new engine MUST carry all of them.

### D1 -- CAUSALITY (lag-1, train-fit thresholds, past-only features)
Weights are lagged 1 bar before applying to returns; regime thresholds are fit ONLY on a TRAIN prefix
(`REGIME_TRAIN_FIT`); every feature uses bars <= t (rolling realized vol, rolling regime features).
- **Prevents:** look-ahead inflation. *Held-out-not-in-sample* -- a strategy fit on the same span it is
  scored on is not a strategy, it is a memory. The gauntlet's dim7 programmatically proves this (perturb
  future bars -> weights/labels/thresholds at t must be unchanged). G-AUDIT-011 (full-history
  standardization leak) is the class this closes.

### D2 -- TWO-SIDED SELFTEST (produces signal AND produces nothing when there is nothing)
Every engine ships a `--selftest` that asserts BOTH directions: on a positive-drift synthetic panel the
core is positive with full coverage (it *finds* signal); on a zero-exposure / zero-edge case it returns
~0 (it *manufactures nothing*); on a synthetic crash the overlay de-risks. (`daily_engine.selftest`
`:401`; `core_satellite_book.selftest` `:361`; `engine_timeframe_sweep.selftest`.)
- **Prevents:** a gate that only ever says "PASS." A one-sided test that only checks "did we find
  something" cannot catch a harness that finds something *in noise*. The gauntlet's own selftest requires
  a zero-edge stream to come back FRAGILE -- the gate must reject as well as accept.

### D3 -- LEVERAGE CAP (never vol-match a tiny sleeve)
The satellite's gross leverage is hard-capped (`SAT_MAX_LEVERAGE=3.0`) and its capital fraction is capped
(`SAT_FRAC_CAP=0.40`); the risk-budget solver is capital-constrained (weights sum to 1, no >100%  <!-- VERIFIED RWYB -->
notional). (`core_satellite_book._risk_budget_split` `:154`; CLI clamps the request `:518`.)
- **Prevents:** the **vol-match inflation trap**. Vol-matching a ~2%-ann-vol market-neutral sleeve to a  <!-- VERIFIED RWYB -->
  ~60%-ann-vol beta core implies 5x-21x leverage and prints absurd compound the sleeve can never hold.  <!-- VERIFIED RWYB -->
  The prior `funding_satellite_assessment` blend did exactly this (27000%..39,000,000%); the cap is the  <!-- VERIFIED RWYB -->
  fix.

### D4 -- RANK BY NET WEALTH (not Sharpe, not gross)
The objective is **held-out compound return** (PROJECT_NORTH_STAR / OBJECTIVE FUNCTION mandate). Costs are
charged (taker default 0.0024 rt / maker 0.0006); the comparison table ranks ENGINE/CORE/BUYHOLD on NET
compound; Sharpe is reported but is not the target.
- **Prevents:** **Sharpe-rewards-underparticipation.** A book that sits out most of the time can post a
  high Sharpe while compounding less wealth than a fully-invested one. We optimize wealth; a Sharpe-2.0 /
  +50% book loses to a Sharpe-1.4 / +70% book.  <!-- VERIFIED RWYB -->

### D5 -- TRAIN-FIT ONLY (the eval span is never fit on)
Regime thresholds are terciles of the TRAIN feature distribution only; the regime->scalar map and the
satellite cap are PRE-REGISTERED constants, never tuned on the eval span. The gauntlet's dim2 PBO is
INFORMATIVE not pass/fail precisely *because* the scalars are pre-registered (a high PBO here signals
regime-dependence of the optimal scalar, not overfit).
- **Prevents:** silent in-sample tuning. If you tune the exposure scalars on history, their ranking
  inverts out of sample (bull rewards high exposure, bear rewards low). Pre-registration is the
  mitigation; the chosen map is "reasonable + defensive," not "provably optimal."

### D6 -- SWEEP ALL TIMEFRAMES (never silently default one)
Any analysis/decomposition/backtest must sweep {1d,4h,1h,30m,15m} (+ alt-bar-types) or explicitly state
the cadence and why. The engine is hardcoded daily for deployment, but its methodology is validated
across cadences by `engine_timeframe_sweep.py` (§5).
- **Prevents:** a cadence-blind conclusion. Cadence materially changes the answer (tails, Hurst,
  MA-whipsaw, cost drag). The user flagged silent-single-cadence defaulting TWICE as a regression
  (feedback-sweep-all-timeframes-never-default-one). Mechanize it; do not default.

### D7 -- HONEST CONTROLS (vs buy-hold, vs shuffle, vs no-skill, scorecard)
Every engine reports against BUY-HOLD (the naive baseline) and CORE-ALONE (isolates the overlay's value);
selftests use a no-skill / zero-edge control; the canonical `scorecard.score_book` (deflation-aware
block-bootstrap p05) is emitted; concentration is firewalled (no single name > 95% of return).  <!-- VERIFIED RWYB -->
- **Prevents:** **single-draw-shuffle=noise** and phantom skill. One shuffle draw is noise -- controls
  must be distributional (block-bootstrap p05, 60-trial no-skill mean, per-episode bear split). A
  "beats_null" that is really a hold-length artifact (caught by the exit-axis work) is the canonical trap
  this closes.

---

## 4. The VALIDATION GATE = the 7-dimension robustness gauntlet

The ship-gate is `src/strat/daily_engine_gauntlet.py`. It is an *adversarial stress test* of the engine's
claims on its NET return stream, not a re-run of the backtest. Run it on every engine before deploying.

| Dim | Test | PASS criterion | FRAGILE means |
|-----|------|----------------|---------------|
| 1 | **block-bootstrap p05 (held-out)** | held-out (post-2025-03-15) p05 > 0 | OOS expectancy not robustly positive |
| 2 | **PBO / param-overfit (CSCV)** | INFORMATIVE (not pass/fail) | high PBO = the optimal scalar is regime-dependent, NOT overfit (scalars are pre-registered) |
| 3 | **parameter sensitivity** | Sharpe spread < 0.4 across all perturbations | a param is a knife-edge |
| 4 | **regime-stratified (bull/bear/chop)** | overlay saves > 5pp of bear maxDD | overlay does not de-risk the bear as claimed |
| 5 | **cost sensitivity** | compound spread maker->2x-taker < 25% of taker | cost fragile |  <!-- VERIFIED RWYB -->
| 6 | **concentration / firewall** | no single name > 95% of return AND positive without the top name | one asset carries the book |  <!-- VERIFIED RWYB -->
| 7 | **look-ahead audit** | all causality probes pass (thresholds train-only & invariant to post-train data; weights/labels at t invariant to future bars; weights lagged 1) | a leak exists |

**Ship read:** an engine ships only if dim1 PASS (held-out p05 > 0) AND dims 3-7 PASS, with dim2 read as a
diagnostic. **HONEST CURRENT STATE (VERIFIED this session):** the daily engine's held-out p05 is
**negative** (`heldout p05 = -35.99` at the default; scorecard `ship=False`). The engine PASSES the
causality / sensitivity / cost / concentration / overlay dims but is **NOT held-out-positive** -- it is a
beta engine with controlled drawdown, not a held-out alpha sleeve. That is the honest gate result; do not
report it as a ship.

---

## 5. SWEEP VALIDATION -- does the methodology generalize across timeframes?

`src/strat/engine_timeframe_sweep.py` replicates the SAME methodology (same `build_book` / `buy_hold_net`
/ `window_stats`, verbatim) at each cadence, with the windows and annualization scaled to keep ~the same
wall-clock (windows in BARS = `daily_bars x bars_per_day`; `ANN = 365 x bars_per_day`). The daily default
and selftest are untouched (globals are patched in a context manager and always restored -- the gauntlet's
`_engine_net` pattern).

**Full-cycle results (u10, 2020-01-06..2026-05-28, taker, VERIFIED 2026-06-14, git 9b02bc3):**

| cadence | ENGINE comp% | ENGINE Sharpe | ENGINE maxDD% | CORE comp% | BUYHOLD comp% | ann cost-drag %/yr | overlay DD saved (pp) | core+ |
|---------|-------------:|--------------:|--------------:|-----------:|--------------:|-------------------:|----------------------:|:-----:|
| 1d  | **+2656.32** | 1.37 | -48.16 | +2767.39 | +3405.40 | 0.32   | +31.2 | yes |
| 4h  | +1095.84 | 1.06 | -63.74 | +2402.04 | +2631.49 | 1.44   | +16.5 | yes |
| 1h  | +360.68  | 0.81 | -61.82 | +2227.97 | +1166.73 | 1.97   | +18.6 | yes |
| 30m | +236.73  | 0.67 | -64.58 | +276.77  | -46.18   | 15.34  | +25.8 | yes |
| 15m | **-99.71** | -2.00 | -99.74 | -100.0 | -100.0 | 120.55 | +0.3  | **no** |

(All numbers RWYB-verified against `runs/strat/engine_timeframe_sweep_<stamp>.json`.)

**Verdict (VERIFIED):** the methodology is **coarse-cadence-specific**. It holds at {1d, 4h, 1h, 30m}
(core stays positive, overlay controls drawdown) and **FAILS at 15m** (core goes to -100%). The mechanism  <!-- VERIFIED RWYB -->
is the turnover tax: annualized taker cost-drag rises from **0.32%/yr at 1d to 120.55%/yr at 15m** -- a  <!-- VERIFIED RWYB -->
~375x increase that the per-bar edge cannot pay. A secondary mechanism is **regime-classifier
resolution**: at 30m/15m the classifier collapses to ~100% "chop" (it cannot resolve trend/down regimes  <!-- VERIFIED RWYB -->
at fine cadence), so the overlay degenerates to a constant ~0.5 scalar and stops doing regime work.

**Headline for the engine itself:** even where the methodology "holds," the ENGINE underperforms
CORE-ALONE and BUY-HOLD on raw compound at every cadence -- the overlay buys drawdown control, not return
(consistent with D4/D7 and §4's held-out result). The 1d ENGINE is the best *risk-adjusted* form
(highest Sharpe 1.37, shallowest maxDD -48%).  <!-- VERIFIED RWYB -->

### Recommended deployable cadence(s)
- **PRIMARY: 1d (daily).** Best Sharpe (1.37), shallowest maxDD (-48% vs -64% to -100% finer), lowest  <!-- VERIFIED RWYB -->
  cost-drag (0.32%/yr), and the regime classifier actually resolves trend/chop/down (time-share  <!-- VERIFIED RWYB -->
  10.8/66.3/22.9%). This is the deployable engine (`src/strat/daily_engine.py`, hardcoded 1d by design).  <!-- VERIFIED RWYB -->
- **ACCEPTABLE SECONDARY: 4h.** Methodology holds, cost still cheap (1.44%/yr), regime classifier still  <!-- VERIFIED RWYB -->
  resolves; use only if a faster book is operationally wanted -- it costs return for a deeper drawdown.
- **DO NOT DEPLOY: 30m / 15m.** 30m survives only nominally (15%/yr drag, regime collapsed to all-chop);  <!-- VERIFIED RWYB -->
  15m is destroyed by cost (-100%). Fine cadences add cost without adding edge -- the expected  <!-- VERIFIED RWYB -->
  coarse-cadence-best result, now measured.

---

## 6. The honest-ceiling statement (what these engines CAN and CANNOT deliver)

**CAN deliver:**
- A turnkey, run-daily, full-coverage long-only book + net return stream (today-mode + data-quality
  guard).
- **Drawdown control:** the overlay cuts core maxDD materially in the bear -- VERIFIED full-cycle 1d
  -79% (core) -> -48% (engine), a +31pp save; the gauntlet's per-episode bear split confirms it improves  <!-- VERIFIED RWYB -->
  BOTH the 2022 and 2025 bears monotonically.
- **A modest uncorrelated carry lift** via the satellite (funding-dispersion: REPORTED OOS +7.9% /
  UNSEEN +10.3% compound, beta~0, ~zero-correlated with the core) -- Sharpe up, maxDD down, at a sane  <!-- VERIFIED RWYB -->
  capped leverage. (~3-10%/yr steady-state, decay-risk UNCONFIRMED.)  <!-- VERIFIED RWYB -->

**CANNOT deliver:**
- **Held-out directional alpha.** VERIFIED: the daily engine's held-out (2025-03+) p05 is negative
  (-35.99) and the scorecard ship-read is False. This is **beta + vol-target + regime-control + carry**,
  not an alpha sleeve.
- **The 1-5%/day internal-data directional dream.** Definitively closed (the 4-problem mover  <!-- VERIFIED RWYB -->
  decomposition + the internal-data ceiling). An engine built only on internal price data will not print
  directional alpha; the open doors are EXTERNAL data (funding/on-chain/Coinglass) and the carry sleeve,
  not a cleverer price-only signal.
- **Edge at fine cadence.** VERIFIED coarse-cadence-best: cost destroys the book by 15m. Deploy daily.

**One-line summary:** *follow this process to build a robust, drawdown-controlled, full-coverage beta+carry
engine that compounds at deployable cadence -- and report it honestly as that, never as alpha, until it
clears the gauntlet's held-out p05 > 0 gate (which the current engines do not).*

---

## 7. Reproduce

```
python -m strat.daily_engine --selftest                         # two-sided engine soundness
python -m strat.daily_engine --backtest 2020-01-01:2026-01-01   # full backtest + today's book
python -m strat.daily_engine_gauntlet --selftest                # gate soundness
python -m strat.daily_engine_gauntlet                           # the 7-dim ship gate
python -m strat.engine_timeframe_sweep --selftest               # scaling-logic soundness
python -m strat.engine_timeframe_sweep                          # the sweep over {1d,4h,1h,30m,15m}
python -m strat.core_satellite_book                             # the core+satellite book
```

Artifacts land in `runs/strat/` (each with a `repro` block: command + git SHA + config). Every engine
self-tests (`--selftest`, two-sided) and is causal + costed + ranked-by-net by construction.

**Related docs:** `docs/DAILY_ENGINE.md` (the engine's own spec + robustness section), `PROJECT_NORTH_STAR.md`
(objective = wealth), `docs/WEALTH_BOT_DEVELOPMENT_FRAMEWORK.md` (the discovery-phase framework upstream of
deployment).
