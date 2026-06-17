# Oracle System Reference (2026-06-08)

This document describes the crypto oracle system as it actually exists in the committed codebase.
No numbers are invented; all figures are drawn from committed code, docstrings, and
`docs/ORACLE_DNA_FINDINGS_2026_06_08.md`.

---

## 1. Overview

The system is built around three distinct engines with clearly separated roles.

### Engine 1: ORACLE (hindsight upper bound)

`src/oracle/engine.py` (`OracleEngine`) and its predecessor `src/oracle/ma_oracle_engine.py`
(`MAOracleEngine`).

For a query date D the oracle ranks the top-N performers in the universe by trailing return, then
for each asset selects the best (indicator config, entry day) pair that maximized realized captured
return to D. The per-config signal (MA cross computation) is **causal** (uses only
`closes[:d_idx+1]`, never a future bar). The hindsight is only in the best-config selection after
the fact. This is the allowed oracle move: it defines a hindsight upper bound on what an MA-family
long entry could have captured. Every output row carries `hindsight=True`.

The oracle is a **descriptive ceiling, not a tradeable signal.**

### Engine 2: MA ADAPTIVE (realizable, past-only chooser)

`src/oracle/adaptive.py` (`AdaptiveChooser`).

The honest forward analog of the oracle. At decision date D it uses only `closes[:d_idx+1]` and
chimera features as-of D. It selects a config per asset using its **past-only rolling-validity
score**: the mean capture rate of completed golden-to-death cross round-trips whose entries fall in
a trailing validity window. It then reports whether that config is in-position at D (a live golden
cross with no later death cross) as a forward entry signal. There is no `captured_return` column
because the model does not know the future move.

Three **mechanisms** are registered (`src/oracle/adaptive.py:MECHANISMS`):
- `rolling_validity` -- the core: pick the highest past-only validity config that is in-position at D.
- `regime_cond` -- same as `rolling_validity` but gated by a past-only BTC trend+vol regime;
  abstains entirely (cash) when BTC is risk-off.
- `state_cond` -- conditions on a cross-sectional market-state read; scales down the number of
  names held when tape favourability is below 0.5.

### Engine 3: SIDE-BY-SIDE COMPARISON

`src/oracle/compare.py` (`OracleVsModel`).

Runs both engines for the same date, joins on symbol, and adds per-row grading: what did the
model's past-only chosen config actually capture (hindsight eval of an already-made past-only pick,
`capture_of_config`)? The columns `oracle_capture`, `model_realized_capture`, `config_MATCH`, and
`capture_GAP` (= oracle minus model) expose the gap directly. A one-line summary is attached to
every result frame. The ceiling invariant is checked: any row where the model beats the oracle is
flagged as a likely bug (the oracle picks the max-capture config, so the model cannot exceed it
without an error).

### The design goal

Call either engine independently or run the side-by-side comparison. The oracle provides the "right
answer" in hindsight; the adaptive engine provides the realizable model choice; the gap is the
honest measure of how far the model's selection is from the upper bound.

---

## 2. How to call it

**Unified entry point: `src/oracle/run.py`** — one command with subcommands so you do not need the
per-module CLIs. `compare` is the default subcommand (the headline "call either side by side"):

```
python src/oracle/run.py doctor  --universe u10                      # preflight (imports + data range)
python src/oracle/run.py compare --date 2026-05-20 --universe u10    # HEADLINE: model vs oracle side-by-side
python src/oracle/run.py oracle  --date 2026-05-20 --universe u10    # oracle (hindsight) only
python src/oracle/run.py model   --date 2026-05-20 --mechanism rolling_validity   # adaptive (past-only) only
python src/oracle/run.py dna     --date 2026-05-20 --universe u10    # decouple the surrounding DNA
python src/oracle/run.py sweep   --start 2026-03-01 --end 2026-05-25 --step-days 7 # exhaustive leaderboard
```

The per-module `__main__` CLIs below remain available as the underlying calls.

**Windows environment setup (run from repo root):**

```
set PYTHONPATH=src;.
set PYTHONIOENCODING=utf-8
```

### Headline command: side-by-side comparison

```
python src/oracle/compare.py --date 2026-05-20 --universe u10 --cadence 1d
    --validity-window 365 --mechanism rolling_validity --lookback 30 --top-n 25
```

Prints the oracle config vs model config per asset, the model's realized capture,
`config_MATCH`, `capture_GAP`, and a summary line.

### Compare-grid (exhaustive leaderboard over cadence x window x mechanism)

```
python src/oracle/compare.py --start 2026-05-01 --end 2026-05-20 --step-days 5
    --grid --universe u10
```

Returns a leaderboard sorted by `model_mean_capture` descending.

### Robustness check (does the model's selection beat random?)

Append `--robustness` to any `compare.py` call with `--start/--end`. Runs a matched-count
random-selection null (300 draws by default) and reports `beats_random_selection`,
`random_null_p95`, and the chaser gate verdict.

### Oracle only (hindsight upper bound)

```
python src/oracle/engine.py --date 2026-05-20 --universe u50 --indicator ma
    --cadence 1d --lookback 30 --top-n 25
    --out runs/oracle/engine_2026-05-20.csv
```

Optionally append `--reconcile` to print a hand reconciliation for the rank-1 asset (proves the
golden cross and the captured return are bit-identical).

### Adaptive engine only (realizable, past-only)

```
python src/oracle/adaptive.py --date 2026-05-20 --universe u50 --cadence 1d
    --validity-window 365 --mechanism rolling_validity --lookback 30 --top-n 25

# Run all three mechanisms side by side:
python src/oracle/adaptive.py --date 2026-05-20 --all
```

### DNA decoupling (attach features, regime, chart-type context)

```
python src/oracle/dna.py --date 2026-05-20 --universe u10 --indicator ma
    --cadence 1d --lookback 30 --top-n 25 --chart-types 1d,dollar
    --out runs/oracle/dna_u10_ma_1d_2026-05-20.parquet
```

### Panel / incremental sweep store

```
python src/oracle/panel.py --start 2026-01-01 --end 2026-06-01 --step-days 7
    --universe u10 --indicator ma --cadence 1d --lookback 30 --top-n 25
```

Writes to `runs/oracle/panel/panel_<universe>_<indicator>_<cadence>.parquet`. Subsequent runs
skip already-stored dates by default (`--no-skip` to force re-run).

### Verify the DNA finding (reproducibility)

```
python src/oracle/verify_dna_finding.py
# exit 0 = finding reproduces within tolerance + causality holds
```

---

## 3. Architecture and dimensions

### Indicator plug-in REGISTRY

`src/oracle/engine.py:INDICATOR_REGISTRY` is the single extension point.

Currently registered:

| Key | Status | Implementation |
|-----|--------|----------------|
| `ma` | Implemented | `MAIndicator`: 16 configs (SMA+EMA x fast{5,10,20} x slow{20,50,100}, fast < slow). Reuses `_sma/_ema/_crosses` from `ma_oracle_engine.py`. |
| `rsi` | Registered placeholder | Config grid stub only; `signal()` raises `NotImplementedError`. |
| `macd` | Registered placeholder | Config grid stub only; `signal()` raises `NotImplementedError`. |
| `bollinger` | Registered placeholder | Config grid stub only; `signal()` raises `NotImplementedError`. |

Adding a new indicator family requires implementing the `Indicator` protocol (two methods:
`config_grid() -> list[dict]`, `signal(dates, closes, cfg) -> dict`) and registering the instance
in `INDICATOR_REGISTRY`. The rest of the system (engine, adaptive, compare, dna, panel) picks it up
without further changes.

### Cadence-generality

All engines accept `--cadence 1d` (native daily bar) or any event cadence (dollar, dib, range,
runs_*, ...). For event cadences the engine aggregates to a daily close series (last bar close on
each calendar date) before applying the indicator. This mirrors the logic in
`src/oracle/decomposer.py:_daily_series`. The adapter is in
`src/oracle/engine.py:OracleEngine._daily_series`.

### Validity windows and the driver

The `rolling_validity` driver iterates over a list of validity windows (default `[180, 365]` days)
in order. For each window it scores in-position configs by the mean capture rate of completed
golden-to-death round-trips whose entry falls in `[D - window, D]`. The first window that yields at
least `min_valid_trades` (default 3) qualifying configs wins. When no window qualifies the driver
falls back to `bounded_oneshot` (max captured return among in-position configs).

The `bounded_oneshot` driver bypasses validity scoring and picks purely by realized captured return.

### The three adaptive mechanisms

`src/oracle/adaptive.py:AdaptiveChooser._mech_rolling_validity` -- the core forward analog of the
oracle driver (past-only validity score).

`src/oracle/adaptive.py:AdaptiveChooser._mech_regime_cond` -- regime-gated via a past-only BTC
trend+vol regime function (`btc_regime_series` loaded by file path from
`runs/staging/h1_regime_overlay_2026_06_08.py`; gracefully degrades to `regime_unavailable` if the
file is absent).

`src/oracle/adaptive.py:AdaptiveChooser._mech_state_cond` -- cross-sectional state gate via
`firm.market_state.compute_state`; scales the number of names held by tape favourability.

### DNA decoupling dimensions

`src/oracle/dna.py:decouple()` attaches four groups of columns to every engine row:

- **ENGINE**: the raw oracle output (sym, config, entry, captured return, capture rate, etc.).
- **TIMING**: `query_date`, `peak_date` per chart type (day of max close in `[entry_date, D]`).
- **REGIME**: `btc_regime_at_entry`, `btc_regime_risk_on` computed past-only from BTC closes up to
  and including `entry_date` (SMA200 + 30d-vol vs expanding median).
- **FEATURES**: `ctx_entry__<col>` and `ctx_peak__<col>` -- chimera v50/v51 feature vectors as-of
  `entry_date` and `peak_date` respectively, via
  `src/oracle/decomposer.py:OracleDecomposer._features_as_of`.
- **CHART**: per-chart-type `best_config`, `entry_date`, `days_back`, `captured_return`,
  `capture_rate`, `peak_date` for each chart type in the `chart_types` list.

### The robustness spine each candidate runs

`src/strat/` houses the machinery that grades any candidate strategy. Every candidate that clears
the oracle layer faces:

- **`firewall.py` (`random_entry_null`)**: cost-matched random-entry null. Three modes: plain
  (draws from all window bars), `regime_matched` (draws only from gate-ON bars, isolating timing
  from regime selection), `membership_matched` (draws from within the same multi-candle move
  window, isolating trigger timing from move selection).
- **`dual_null_evaluator.py`**: decomposes edge into NULL-1 SELECTION (which moves to be in) and
  NULL-2 TIMING (entry precision inside a move); both yield empirical p-values + effect sizes on
  held-out data.
- **`synthetic_positive_control.py`**: two-sided soundness gate: confirms the null admits genuine
  timing skill and genuine selection skill before being applied to a real candidate. A gate that
  rejects everything (or passes everything) is not a gate.
- **`battery.py`**: three lenses (Lens A strict/institutional, Lens B pragmatic, Lens C temporal
  barometer). Lens A requires all-4-positive, n>=15, jk2>0, jk3>0, block-bootstrap p05>0,
  maxDD<30%. Lens B: all-4-positive, UNSEEN>0, jk2>0, n_eff>=8. Lens C: UNSEEN>0, >=3 months,
  60%+ monthly positive.
- **`benchmark.py`**: beta-matched passive benchmark (the beta-honest comparison; a strategy must
  beat the same-exposure passive hold, not just beat zero).
- **`candidate_gate.py`**: chains cost lens, leak probe, firewall, and battery into a single gate.

---

## 4. The methodology

The oracle-decomposition methodology is documented in `docs/ORACLE_DECOMPOSITION_2026_06_06.md`.

### Step 1: construct the oracle (the ceiling)

Run `OracleEngine.oracle()` for the target date and universe. This produces, per asset, the
hindsight-best (config, entry) pair and the realized `capture_rate = captured_return /
perfect_return`, where `perfect_return = (close[D] - min_close_in_window) /
min_close_in_window` is the perfect-entry oracle over the same lookback window.

### Step 2: decompose the DNA (what does the oracle exploit?)

Run `dna.decouple()` to attach chimera features, BTC regime, and chart-type context to every
oracle move. The resulting panel (`runs/oracle/dna_panel.parquet`) records 47 `ctx_entry__*`
chimera feature values per (asset, date) row.

For the executed DNA analysis: AUC of each feature against the GOOD-vs-POOR capture-rate
classification on the full panel identifies which entry-day features predict capture quality. The
`verify_dna_finding.py` script reproduces the result.

### Step 3: diffuse the noise (the realizable ceiling)

The fraction of oracle capture that causal features predict is the realizable ceiling. The
unpredictable remainder is perfect-foresight luck. The DNA analysis quantifies which causal
conditions co-occur with high vs low capture, and which are cohort/date effects vs per-asset
signals.

### Step 4: build and validate the proxy (capture-rate KPI)

**Capture rate = realized / oracle-max** is the primary L2 KPI (cost-free, capital-free). Any
candidate entry policy is graded against the oracle ceiling. Validation runs through the full
robustness spine described in Section 3.

---

## 5. Honest findings (what the RWYB runs showed)

All numbers in this section are drawn from `docs/ORACLE_DNA_FINDINGS_2026_06_08.md` and the
overseer-RWYB records in those docs. VERIFIED means independently recomputed by the overseer.

### Genuine multi-timeframe leaderboard (overseer RWYB, u10, 6 dates 2026-03..05)

`model_mean_capture` vs `oracle_mean_capture` (the hindsight ceiling), per cadence x mechanism:

| cadence | oracle ceiling | model (rolling_validity) | model (state_cond) | model (regime_cond) | config-match |
|---|---|---|---|---|---|
| **1d**  | ~0.35 | ~0.10-0.13 | ~0.10 | 0.0 (abstains) | ~0.42 |
| **4h**  | ~0.25 | ~0.10-0.11 | ~0.08-0.09 | 0.0 (abstains) | ~0.33-0.40 |
| **1h**  | ~0.19 | ~0.06 | ~0.02 | 0.0 (abstains) | ~0.55-0.60 |

The model captures ~30-40% of the oracle ceiling at 1d/4h, less at 1h. `rolling_validity` is the best
mechanism; `regime_cond` abstained entirely in this BTC-risk-off window; `state_cond` trails. The oracle
ceiling itself shrinks at finer cadences (more capturable move at the daily scale).

### Robustness verdict -- does the model's selection beat a RANDOM config?

Per (asset, date), model's chosen-config capture vs the mean capture of all 16 grid configs (a no-skill
chooser). EDGE = model - random; win-rate = fraction of picks where model > random:

| cadence | model_mean | random-config baseline | win-rate | verdict |
|---|---|---|---|---|
| **1d** | 0.246 | 0.124 | **50%** | NO EDGE (coin-flip; the higher mean is a few big wins) |
| **4h** | 0.330 | 0.304 | **65%** (n=20) | modest EDGE -- the faint selection skill is at 4h |

CAVEAT (the chess small-sample lesson): 13/20 at 4h is NOT statistically strong (one-sided binomial p~0.13).
Suggestive, not proven. But it points the future SOTA truly-adaptive engine at **4h + rolling-validity**.

### Performance note (an honest engine limitation)

Native rolling-validity at FINE cadences is O(configs x crosses x bars x dates): 1h ~280s per
(window,mechanism) combo, dollar slower. 1d/4h are practical; an exhaustive 1h/dollar sweep is not (yet).
Optimization (vectorize the per-config native scan / cache crosses) is an open perf item -- the capability
exists (any cadence runs), only the speed at fine cadences is limiting.

### The model is far below the oracle ceiling

In the compare harness on the u10 universe, 2026-05-20 (reported in the compare-module docs):
`model_mean_capture` of approximately 0.035 vs oracle `~0.258` is the order of magnitude in the
findings; exact numbers for a specific date run should be read from the output of
`compare.py --date <D> --universe u10` directly, as these figures depend on the universe state on
that date.

The config-match rate (oracle chosen config equals model chosen config) is approximately 10-70%
depending on date. In the compare grid, the model's selection does not beat random: in the executed
robustness check the model's mean capture (approximately 0.334) was below the random null p95
(approximately 0.408), meaning `beats_random_selection = False`.

### The DNA finding: REGIME-driven, not asset-feature-driven (VERIFIED)

From `docs/ORACLE_DNA_FINDINGS_2026_06_08.md`, verified by `src/oracle/verify_dna_finding.py`:

- 47 entry-day chimera features ranked by class-separation (GOOD vs POOR capture rate): **max
  |AUC - 0.5| = 0.117, median 0.031**. Only 1/47 clears the noise bar.
- The single winner `ctx_entry__xd_btc_volatility` had raw AUC 0.6064, which collapsed to 0.5329
  after date-demeaning. This is a **per-DATE / cohort effect** (high-BTC-vol days were good days
  for the entire top-25 cohort), not a within-cohort per-asset discriminator.
- **108/235 = 46%** of DNA panel rows have `capture_rate` exactly 0.0 (zero-inflated; confirmed
  VERIFIED).

**Conclusion:** whether a top-performer's MA crossover captures its move is governed by the market
regime of the day, not by per-asset chimera features.

### H1 regime overlay validation: NULL on alpha (VERIFIED)

Tested in `runs/staging/h1_regime_overlay_2026_06_08.py` against the `candidate_gate` battery.
Long-only SMA(30/50) on u10 with a BTC-regime overlay, held-out (mean-asset compound, net taker):

- ARM A (always-on): UNSEEN **-31.0%**
- ARM B (BTC-regime overlay): UNSEEN **0.0%**
- ARM C (passive buy-and-hold): UNSEEN **-28.5%**

The overlay's 0% UNSEEN is **trivially explained by full cash**: UNSEEN was a pure BTC downtrend
(risk-on fraction = 0.0 the entire window). ARM B2 (classic `BTC > SMA200` only) produced
identical UNSEEN results to the full overlay. Under the exposure-matched passive baseline (0%
exposure gives 0% return), the overlay `beats_beta_held = 0/10 = NULL`. The bare MA driver failed
the gate on all 10 seeds.

**The oracle is a hindsight upper bound, not alpha.** It is the foundation for understanding what
a future SOTA truly-adaptive engine would need to capture, not a deployable strategy.

---

## 6. What is covered vs what is open

### Covered

- MA/EMA indicator family (16 configs: SMA+EMA x fast{5,10,20} x slow{20,50,100}) -- fully
  implemented in `MAIndicator` (`src/oracle/engine.py`), reusing the verified causal primitives
  from `src/oracle/ma_oracle_engine.py`.
- Cadence-generality: 1d native and any event cadence via daily-close aggregation.
- Window-generality: configurable `validity_windows` (default 180 and 365 days, tried in order).
- Three adaptive mechanisms: `rolling_validity`, `regime_cond`, `state_cond`.
- Side-by-side comparison harness with ceiling invariant check
  (`src/oracle/compare.py:OracleVsModel`).
- Exhaustive grid sweep and leaderboard (`compare_grid`).
- Robustness check: matched-count random-selection null vs model selection
  (`OracleVsModel.robustness`).
- DNA decoupling across four column groups: engine, timing, regime, features, chart types
  (`src/oracle/dna.py:decouple`).
- Incremental panel store with atomic writes (`src/oracle/panel.py:build_panel`).
- Reproducible DNA verifier (`src/oracle/verify_dna_finding.py`, exit 0 = reproduced).
- Robustness spine in `src/strat/`: firewall (three null modes), dual-null evaluator, synthetic
  positive control (two-sided soundness), battery (three lenses), benchmark, candidate gate.

### Open (registry slots and the realizable adaptive engine)

The `INDICATOR_REGISTRY` in `src/oracle/engine.py` has three **registered-but-not-implemented**
slots: `rsi`, `macd`, `bollinger`. Their `config_grid()` returns a stub; calling `signal()` on any
of them raises `NotImplementedError`. Building a real `RSIIndicator`, `MACDIndicator`, or
`BollingerIndicator` that implements the `Indicator` protocol and replacing the placeholder in the
registry is the mechanical extension path.

The larger open problem is the **SOTA truly-adaptive engine**: a realizable chooser that actually
beats random selection on held-out data. The current `AdaptiveChooser` with any mechanism does not
clear this bar at 1d daily resolution (DNA finding: capture is regime-driven, not per-asset-feature-
driven; H1 regime overlay is a trivial bear-abstention null). The oracle system gives the honest
gap measurement and the DNA decomposition that redirects where to look; it is the foundation, not
the destination.

---

## File map

| File | Role |
|------|------|
| `src/oracle/ma_oracle_engine.py` | v1 hindsight oracle: verified causal primitives (`_sma`, `_ema`, `_crosses`, `_last_idx_le`, `_to_date`, `rank_top_performers`); the source-of-truth for MA logic |
| `src/oracle/engine.py` | Generalized oracle engine: `OracleEngine`, `MAIndicator`, `INDICATOR_REGISTRY`, plug-in `Indicator` protocol, cadence/driver-general |
| `src/oracle/adaptive.py` | Realizable chooser: `AdaptiveChooser`, three mechanisms, `MECHANISMS` registry |
| `src/oracle/compare.py` | Side-by-side harness: `OracleVsModel`, `capture_of_config`, `compare_grid`, `robustness` |
| `src/oracle/dna.py` | DNA decoupling: `decouple()`, `_BTCRegimeCache`, four column groups |
| `src/oracle/decomposer.py` | v2 cross-dimensional decomposer: `OracleDecomposer` (reused by `dna.py` for `_features_as_of` and `_driver_for_cadence`) |
| `src/oracle/panel.py` | Incremental panel store: `build_panel`, `load_panel`, `date_range` |
| `src/oracle/verify_dna_finding.py` | Reproducibility verifier for the DNA finding (exit 0 = reproduced) |
| `src/strat/firewall.py` | Random-entry null: three modes (plain, regime-matched, membership-matched) |
| `src/strat/dual_null_evaluator.py` | Two-null evaluator: selection + timing p-values and effect sizes |
| `src/strat/synthetic_positive_control.py` | Two-sided soundness: confirms gates admit genuine skill before being applied |
| `src/strat/battery.py` | Three-lens robustness battery (Lens A/B/C) |
| `src/strat/benchmark.py` | Beta-matched passive benchmark |
| `src/strat/candidate_gate.py` | Integrated gate: chains all strat-layer checks |
| `docs/ORACLE_DECOMPOSITION_2026_06_06.md` | Methodology: construct / decompose / diffuse / build |
| `docs/ORACLE_DNA_FINDINGS_2026_06_08.md` | DNA finding: REGIME-driven not asset-feature-driven (VERIFIED) |
| `runs/oracle/dna_panel.parquet` | 235 (asset, date) DNA records; corpus for the finding |
| `runs/oracle/dna_ranked_features.csv` | 47 `ctx_entry__*` features ranked by AUC |
