# DATA-EXPANSION — Canonical Limited-Data Toolkit

**Module:** [`src/strat/data_expansion.py`](../src/strat/data_expansion.py)
**Status:** CANONICAL. The single source of truth for the six limited-data estimation techniques.
Any future model trained / selected / estimated on limited data uses THESE — do not re-implement.
**Provenance:** extracted 2026-06-12 from `src/strat/config_selector_jan2feb.py` (the Jan→Feb 2020
period-level regime→config selector), where the six techniques correctly (a) avoided overfitting calm
January data and (b) correctly found only a cost-avoidance effect into the COVID-crash month — i.e. they
did not hallucinate an edge that was not there.

---

## The binding honest caveat (read first)

These techniques make a limited-data **ESTIMATE robust and overfit-resistant**. They **DO NOT
manufacture signal**. If the underlying process has no edge, a correct application returns *"no confident
pick"* — James-Stein `B → 0`, robust stats `~ 0`. That is the tool **working**, not failing.

The empirical proof is the source: on calm-January-trained / COVID-February-evaluated MA configs, the
toolkit's verdict was *"the only thing learned is that slow-and-rare MA configs survive cost — NOT a
regime→config map, NOT a profitable bot into the crash"*. The random-selection control was beaten only
because the toolkit reliably **avoided** the cost-bleeding configs, not because it found alpha. A
crash-month loss was reported as a loss, not spun. The tools surfaced exactly that, and nothing more.

---

## The six techniques

All functions take **generic inputs** (numpy arrays, group labels, score dicts) and are **pure-numpy**
with no project imports. Every randomized routine takes a `seed` and is reproducible.

### 1. `cross_sectional_pool(per_entity_samples)` — the biggest N-multiplier
- **What:** concatenate `{entity → samples}` into one stream; treat each cross-section entity (asset /
  instrument / symbol) as an independent regime sample.
- **When:** several comparable entities observed over the SAME limited period, and you want a population
  estimate of a statistic you believe is shared (e.g. *"does this config survive cost on average"*).
- **Inputs:** `Mapping[entity → 1-D array]`.
- **Outputs:** `{pooled, n_entities, n_eff_naive, n_eff_corr_adjusted, rho_bar, note}`.
- **Honest caveat:** independence is an ASSUMPTION. Cross-section entities co-move (crypto especially),
  so the TRUE effective N is below the naive pooled count. The function returns BOTH `n_eff_naive` (the
  optimistic count) AND a Kish-deflated `n_eff_corr_adjusted = n / (1 + (m̄−1)·ρ̄)`. **Never cite the
  naive count as independent samples when entities co-move.**

### 2. `subperiod_windows(index_or_len, window, step)` — more samples, honestly down-weighted
- **What:** produce `(start, stop)` sub-window slices over a single series, with an HONEST `n_eff` that
  down-weights for overlap.
- **When:** a single series is your only sample of a period and you want several sub-period reads (e.g.
  weekly windows of a one-month series) as **SUPPORTING** evidence — never independent primary samples.
- **Inputs:** an int length (or any sized object), a `window`, a `step`.
- **Outputs:** `{slices, n_windows, overlap_fraction, n_eff, note}` where
  `n_eff = n_windows · (1 − overlap_fraction)`.
- **Honest caveat:** this is the easiest technique to ABUSE. Overlapping windows are NOT independent;
  the function REFUSES to report their raw count as independence. `n_eff` corrects only for explicit
  overlap, so it is an UPPER bound (even non-overlapping windows of a time series are autocorrelated).

### 3. `block_bootstrap_distribution(returns, n_boot=400, block=3, stat='median', seed=0)` — a distribution + a robust stat
- **What:** resample a return stream in **contiguous blocks** (preserving short-range autocorrelation),
  compound each resample, and summarize the resulting distribution; rank by a ROBUST stat (median / p25),
  never the in-sample max.
- **When:** you have a SHORT stream of per-event (per-trade) returns and want a robust central estimate +
  downside spread, instead of the single optimistic realized compound.
- **Inputs:** 1-D returns (decimal), `n_boot`, `block` (= your autocorr horizon), `stat`, `seed`.
- **Outputs:** `{distribution, median, p25, p05, mean, robust, n}`.
- **Honest caveat:** preserves only SHORT-range dependence; a stream with no losers cannot produce a
  loss; a degenerate (empty / length-1) stream returns a point mass.

### 4. Synthetic regime paths — `fit_regime_generator` / `simulate_regime_paths` / `score_across_paths`
- **What:** fit `{drift, vol, ar1}` from a return stream; simulate `K` synthetic future-like paths
  (AR(1) + drift/vol, vectorized across paths); score a candidate across them and take a robust
  (median / p25) aggregate.
- **When:** you want to stress a candidate across MANY plausible continuations of the observed regime,
  not just the one realized path.
- **Inputs:** `fit_regime_generator(returns)` → params; `simulate_regime_paths(params, n_bars, K, seed,
  p0)` → `{returns, prices, K, n_bars}`; `score_across_paths(scorer_fn, paths, on, stat)` →
  `{robust, median, p25, mean, K}`.
- **Honest caveat:** AR(1)+drift+vol is the simplest generator (Gaussian innovations, single lag) — no
  fat tails, no vol clustering, no regime switches. **Alternatives:** block-bootstrap-of-paths (preserves
  the empirical marginal + short autocorr) and IAAFT (preserves amplitude distribution + power spectrum).
  Use AR(1) as the cheap default; the robustness is only as good as the generator.

### 5. `james_stein_shrink(scores, prior=None, noise_var=None)` — THE overfit-killer
- **What:** shrink each candidate's score toward a prior (default = grand / cross-sectional mean) by a
  **data-driven** factor `B ∈ [0,1]`:
  `shrunk = prior + B·(raw − prior)`, `B = 1 − (k−2)·σ² / Σ(raw − prior)²` (clipped).
  - spread ≫ noise → `B → 1` → keep the raw scores (differences are **SIGNAL**; trust the winner).
  - spread ≈ noise → `B → 0` → collapse to the prior (differences are **NOISE**; pick nothing).
- **When:** ALWAYS, before taking an argmax over candidate scores estimated on limited data (config
  ranking, hyperparameter selection, per-asset picks). It is the overfit-killer.
- **Inputs:** `Mapping[name → score]` OR 1-D array; optional `prior`; optional `noise_var` (per-score
  estimation-noise variance σ², in the SAME units² as the scores).
- **Outputs:** `(shrunk, B)` — same container type as input (dict→dict, array→array).
- **Honest caveat:** σ² must be in the same units² as the scores. The default (a 1.0-unit² floor) assumes
  scores are in compound-% units — pass `noise_var` explicitly for any other scale (e.g. the squared
  standard error of each score, or the squared half-IQR of a per-candidate bootstrap). `B` is a SCALAR
  applied to all candidates (classic James-Stein), not per-candidate.

### 6. `regime_bucket` / `hierarchical_pool_by_bucket` — generalize, don't memorize
- **What:** label each sample with a coarse `(trend × vol)` regime bucket, then pool samples WITHIN
  bucket (lower-dimensional, generalizable) instead of estimating a separate value per entity.
- **When:** you want a regime→X map that generalizes (learn per bucket), not a per-entity map that
  memorizes the in-sample winner for each individual entity.
- **Inputs:** `regime_bucket(returns, vol_terciles_breaks)` → `{bucket, trend, vol_bucket, ...}` (use the
  `vol_terciles(values)` helper to get the cross-sectional tercile breaks);
  `hierarchical_pool_by_bucket(per_entity_samples, bucket_fn)` →
  `{bucket → {pooled, n_entities, n_samples, entities}}`.
- **Honest caveat:** pooling assumes entities within a bucket are exchangeable (same data-generating
  regime) — that is the whole bet of regime-conditioning. The label is COARSE by design (that coarseness
  is what makes it generalize). Buckets with few entities still over-fit — check `n_entities` before
  trusting a bucket's pick.

---

## How to use in a NEW limited-data model (the recipe)

You have a tiny dataset (one period, few entities) and a grid of candidate configs/models to choose from.

1. **Pool across entities (technique 1).** `cross_sectional_pool({entity: samples})`. Use
   `n_eff_corr_adjusted` (NOT the naive count) as your honest sample size. This is your biggest N gain.

2. **Get a robust per-candidate estimate, not the max (technique 3).** For each candidate, take its
   pooled per-event return stream and run `block_bootstrap_distribution(stream, block=<autocorr horizon>)`.
   Rank by `robust` (median or p25), never the realized compound.

3. **Stress across plausible futures (technique 4).** `fit_regime_generator(returns)` →
   `simulate_regime_paths(...)` → `score_across_paths(scorer, paths, stat='p25')`. Prefer candidates that
   are robust across the synthetic distribution, not just on the one realized path.

4. **(Optional) Add sub-period reads as supporting evidence only (technique 2).**
   `subperiod_windows(len, window, step)`; weight them by `n_eff`, never by `n_windows`.

5. **Combine the robust signals into one score per candidate, then SHRINK (technique 5 — the keystone).**
   Average the robust stats from steps 2–4 into one score per candidate. Set `prior = median(scores)` and
   `noise_var =` your estimation-noise proxy (e.g. the squared half-IQR of the per-candidate bootstrap;
   floor it so shrinkage never fully disengages). Run `james_stein_shrink(scores, prior, noise_var)`.
   **If `B` is near 0, STOP — the data does not support a confident pick.** Only argmax the shrunk scores
   when `B` is meaningfully above 0.

6. **Generalize via regime, don't memorize per entity (technique 6).** Label each entity with
   `regime_bucket(...)`, pool with `hierarchical_pool_by_bucket(...)`, and learn the best (shrunk)
   candidate PER BUCKET. Then ask the honest diagnostic: **does the map differentiate** (distinct
   candidate per bucket = genuine regime skill) **or collapse** (one candidate for all buckets = a
   single-axis dominance effect, NOT regime navigation)? Report which.

7. **Validate against a no-skill control.** Always compare the picked candidate's held-out result to a
   RANDOM-selection baseline and to a naive default. Beating random is the bar; approaching an oracle is
   the goal; a held-out loss is a loss.

---

## RWYB / selftest

```
python src/strat/data_expansion.py        # two-sided selftest (no market data)
```

The two-sided selftest proves each technique sound:
- **`james_stein_shrink` (keystone):** a wide-spread REAL winner → `B` high (trust); i.i.d. NOISE → `B`
  low / heavy shrink (pick nothing).
- **`block_bootstrap_distribution`:** the bootstrap PRESERVES the marginal (mean within tolerance of the
  realized compound) while giving a non-degenerate spread.
- **`simulate_regime_paths`:** the synthetic paths PRESERVE the input regime stats (drift / vol / ar1
  within tolerance).
- **`subperiod_windows`:** `n_eff` is DOWN-weighted vs the naive count under overlap (proves it does not
  fake independence); exact under no-overlap.
- **`cross_sectional_pool`:** correlated entities deflate `n_eff` below the naive count; independent
  entities keep it.
- **`regime_bucket` / `hierarchical_pool_by_bucket`:** up/down streams label correctly; same-regime
  entities pool together.

The caller `src/strat/config_selector_jan2feb.py` imports the canonical functions (single source of
truth) and its own two-sided selftest + a real Jan→Feb run still pass:

```
python src/strat/config_selector_jan2feb.py --selftest
python src/strat/config_selector_jan2feb.py --cadences 4h
```
