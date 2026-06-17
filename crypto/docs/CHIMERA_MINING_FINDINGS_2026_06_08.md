# Chimera Feature-Space Mining — Findings (2026-06-08/09, 6h autonomous)

**Mandate:** *"decompose crypto and all the chimera features across all assets and different time frames... a mining
exercise... regime, clusters, trends... etc."* **Scope:** u100 (104 assets; 77 at 30m) x {1d, 4h, 1h, 30m, 15m} x the
full 243-col chimera v51 schema. **Method/plan:** [CHIMERA_MINING_PLAN_2026_06_08.md](CHIMERA_MINING_PLAN_2026_06_08.md).
**Status:** DESCRIPTIVE / unsupervised. Every number below is overseer-RWYB (recomputed this session from the artifacts).
**Reproduce:** `python src/mining/chimera_mine.py` then `python src/mining/analyze.py` (artifacts in `runs/mining/`).

> **HONESTY RAIL (binding).** This is a map of the *terrain*, NOT an edge. Descriptive structure != tradeable alpha.
> Bar-level autocorrelation is partly bid-ask bounce / microstructure noise, not a cost-survivable signal. No claim
> here has passed `candidate_gate`. The prior adaptive-MA edge-search at 1d/4h/1h/30m is a VERIFIED-HONEST NULL
> ([[project-oracle-decomposer-dna-2026-06-08]]); this exercise explains *why* and points where to look next.

## Executive summary — the 6 findings
0. **(THE DEEPEST ONE) Crypto's next move is predictable in MAGNITUDE, not DIRECTION — at every timeframe.** Across the
   40 dense features the best linear handle on the next bar's *size* is ~0.04-0.07 (small but real, led by
   `norm_oi_price_divergence`, rising intraday), while the best handle on its *direction/sign* is **0.013-0.048 =
   noise**. Volatility is persistent for 100% of assets (AC1|ret| 0.18 at 1d -> 0.33 at 15m); direction is a coin flip.
   *This single split explains the whole prior null: long-only directional search asks the one question the data can't
   answer. The honest edge frontier is volatility/magnitude (breakout, vol-targeting, options), not direction.*
1. **No trend-persistence at ANY canonical timeframe — and it gets *more* mean-reverting intraday.** Median Hurst is
   **below 0.5 at every cadence** (1d 0.487 -> 15m 0.474); the count of trending assets (H>0.55) **collapses** from 20
   at 1d to 0-2 intraday. Median return AC1 goes **−0.018 (1d) -> −0.053 (30m)**, with **99% of assets AC1<0 at 30m** (VERIFIED).
   *Crypto at daily-to-intraday scale is random-walk-to-mean-reverting.* This is the structural root cause of the
   adaptive-MA null AND the finer-TF cost-cliff (trend-following fights the data + pays more costs).
2. **Crypto is a one-factor market at every resolution.** Median pairwise return correlation is **~0.55 at all five
   cadences**; BTC-beta median **~1.19** (alts ~1.2x BTC vol; VERIFIED); 46 of 50 co-moving assets fall in **one** cluster.
   Implication: *within long-crypto, asset-selection diversifies little — market/regime TIMING dominates selection.*
3. **The chimera feature space is RICH and non-redundant — the features are not the bottleneck.** 40 dense features
   (norm_/xd_) need **27 principal components for 90%** of variance (PC1 only ~10%) — and this is **identical across all
   cadences**. Only **2 of 780** feature pairs exceed |corr|>0.8. The information content is high and stable; the
   bottleneck is the *mapping* from features to a cost-survivable edge (and trend-following is the wrong mapping).
4. **Five interpretable market regimes (GMM, BIC-selected k=5 at every cadence)**, dominated (~39-49%) by a
   *quiet/low-vol slightly-down* regime; the others are downtrend-bounce, uptrend-momentum, euphoria-blowoff (~5%,
   extreme vol), and a *topping/distribution* regime (above-trend but bleeding). Regime **persistence rises** toward
   finer TF (regime stays put 44%/bar at 1d -> 85%/bar at 15m) — intraday regimes are identifiable and sticky.
5. **Feature availability is cadence-stable but family-uneven.** norm_(33)/xd_(7)/stbl_(13)/fund_(10) are dense
   (~0-7% missing) across ALL assets/timeframes; microstructure is sparse — LOB **91%**, cross-exchange **97%**, DVOL
   **98%** missing (recent/major-venue only). Any universe-wide signal must live in the dense families; LOB/DVOL-based
   ideas are major-asset-only.

## Cross-cadence tables (RWYB)

### Trend character — Hurst / variance-ratio (trend vs mean-revert)
| cadence | median Hurst | median VR(5) | trending H>0.55 | random-walk | mean-rev H<0.45 | n |
|--------|----|----|----|----|----|----|
| 1d  | 0.487 | 0.951 | 20 | 51 | 26 | 97 |
| 4h  | 0.491 | 0.966 | 1  | 88 | 15 | 104 |
| 1h  | 0.482 | 0.925 | 2  | 93 | 9  | 104 |
| 30m | 0.475 | 0.893 | 0  | 67 | 10 | 77 |
| 15m | 0.474 | 0.897 | 0  | 93 | 11 | 104 |

### Mean-reversion strength (return AC1; <0 = reverts)
| cadence | median AC1 | frac assets AC1<0 | median VR(5) |
|--------|----|----|----|
| 1d  | −0.018 | 0.65 | 0.951 |
| 4h  | −0.025 | 0.82 | 0.966 |
| 1h  | −0.028 | 0.90 | 0.925 |
| 30m | −0.053 | **0.99** | 0.893 |
| 15m | −0.035 | 0.95 | 0.897 |

### Co-movement / BTC structure (returns; fine cadences daily-resampled for the corr matrix)
| cadence | median pairwise corr | BTC-beta median | n assets |
|--------|----|----|----|
| 1d  | 0.551 | 1.194 | 50 |
| 4h  | 0.555 | 1.181 | 50 |
| 1h  | 0.553 | 1.194 | 50 |
| 30m | 0.564 | 1.167 | 41 |
| 15m | 0.549 | 1.197 | 50 |

*(Lead-lag at the daily-resampled resolution is contemporaneous — best lag 0 for all alts vs BTC. Native intraday
lead-lag, e.g. BTC->alt at minute scale, is NOT tested here and is an open follow-up.)*

### Feature structure — PCA effective dimensionality on 40 dense features
| cadence | PC1 % | PC2 % | PCs for 70% | PCs for 90% | pairs |corr|>0.8 (of 780) |
|--------|----|----|----|----|----|
| 1d  | 10.1 | 9.8 | 17 | 27 | 2 |
| 4h  | 10.3 | 9.8 | 17 | 27 | 2 |
| 1h  | 10.2 | 10.0 | 17 | 27 | 2 |
| 30m | 10.4 | 10.2 | 16 | 26 | 2 |
| 15m | 10.4 | 10.2 | 17 | 26 | 2 |

PC1 (the dominant factor) loads on a **trading-activity axis**: norm_bar_duration (+0.34) vs norm_tick_count (+0.28),
norm_whale (−0.29), norm_hawkes_intensity / buy / sell (−0.27..−0.28). i.e. the first axis separates *slow, thin,
low-activity* bars from *fast, whale-driven, high-Hawkes-intensity* bars.

### Discovered regimes (GMM) at 1d — the interpretable 5
| regime | share | ret/bar | vol | trend(vs SMA50) | reading |
|--------|----|----|----|----|----|
| quiet-chop | 0.49 | +0.0005 | 0.036 | −0.07 | dominant; low-vol drift below trend |
| downtrend-bounce | 0.12 | +0.020 | 0.073 | −0.24 | high-vol relief rallies in deep downtrends |
| topping/distribution | 0.20 | **−0.035** | 0.052 | +0.02 | above trend but bleeding (the trap regime) |
| uptrend-momentum | 0.13 | +0.020 | 0.071 | +0.26 | high-vol clean uptrend |
| euphoria-blowoff | 0.05 | +0.014 | 0.125 | +0.39 | extreme-vol parabola |

## What this implies for the next edge-search (the bridge back to returns)
- **Stop fighting the data with trend-following.** The single most consistent finding is sub-0.5 Hurst everywhere and
  strengthening mean-reversion intraday. The natural hypothesis to test next is **mean-reversion / fade-the-move**
  (esp. 30m-1h where AC1<0 holds for 90-99% of assets) — BUT the cost-cliff (30m gross −89.5% in the MA sweep, VERIFIED) means
  this is ONLY viable with maker/limit execution + strong abstention; it must clear `candidate_gate` net of realistic
  costs before any belief. *Descriptive AC1<0 is not yet a tradeable edge — bid-ask bounce confound + costs.*
- **Time the market, don't pick the asset.** One-factor structure (corr 0.55, beta 1.2) means a long basket ~= leveraged
  BTC; the lever that matters is **regime/exposure timing** (when to be on vs cash), not which alt. The 5-regime
  decomposition + the "topping/distribution" regime (−0.035/bar while above trend; VERIFIED) is the most promising *exit/de-risk*
  conditioner.
- **The features are not the limit.** 27-dim non-redundant representation that's stable across timeframes means the
  chimera carries real, independent information; the open problem is a *nonlinear, regime-conditioned, cost-aware*
  mapping to a multi-bar setup — not "more features."

## Deep mining (P7-P9) — the volatility-vs-direction split (the deepest finding)
A second pass (`src/mining/deep_mine.py`) decomposed *predictability itself*. The result is the single most useful
thing in this report: **crypto's next move is predictable in MAGNITUDE, not in DIRECTION — at every timeframe.**

### Volatility clustering — the one robustly predictable structure (strengthens intraday)
| cadence | median AC1(\|ret\|) | median AC1(ret^2) | frac assets >0 | frac >0.1 |
|--------|----|----|----|----|
| 1d  | 0.184 | 0.111 | 1.00 | 0.85 |
| 4h  | 0.243 | 0.140 | 1.00 | 1.00 |
| 1h  | 0.287 | 0.184 | 1.00 | 1.00 |
| 30m | 0.328 | 0.272 | 1.00 | 1.00 |
| 15m | 0.333 | 0.285 | 1.00 | 1.00 |

**Volatility is persistent for 100% of assets at every cadence, and the persistence STRENGTHENS toward intraday**
(AC1|ret| 0.18 at 1d -> 0.33 at 15m). Today's vol predicts tomorrow's — this is where the predictability actually
lives (vol-targeting, breakout-on-expansion, straddle/option structures), and intraday is its home.

### Feature -> next bar: MAGNITUDE is weakly predictable, DIRECTION is ~noise (every TF)
| cadence | max\|corr\| with next \|ret\| | max\|corr\| with next sign | dominant predictor |
|--------|----|----|----|
| 1d  | 0.044 | 0.013 | norm_oi_price_divergence / norm_bar_duration |
| 4h  | 0.048 | 0.028 | norm_oi_price_divergence |
| 1h  | 0.064 | 0.048 | norm_oi_price_divergence |
| 30m | 0.068 | 0.047 | norm_oi_price_divergence |
| 15m | 0.064 | 0.040 | norm_oi_price_divergence |

Across the full 40 dense features, the best linear handle on next-bar **magnitude** is ~0.04-0.07 (small but real, and
rising intraday), led consistently by **norm_oi_price_divergence** (open-interest building against price). The best
handle on next-bar **direction** is **0.013-0.048 = essentially noise** at every timeframe. *This is the deep reason a
long-only directional search is null: the chimera predicts how big, not which way.*

### Movers premise (opportunity) — confirmed fresh
Median **~10 assets move >5%/day**, **3 up-movers >5%**, 2 move >10%; **~84% of days have >=1 up-mover >5%** (stable
across cadences, daily-resampled, ~57 assets/day). Opportunity (cross-sectional dispersion) is abundant — the binding
constraint is ex-ante *direction*, not lack of movement.

### MR / trend candidate lists + regime feature signatures
- **Most mean-reverting (1d AC1):** XUSD/DUSD (**pegged stablecoins -> artifact, exclude**), then TRX, HBAR, ASTER,
  TRUMP. **Most trending (Hurst):** ZBT, FET, HBAR, FLOKI, SEI, DEXE (newer momentum names). Actionable only after the
  stablecoin/bid-ask-bounce caveats.
- **regime_label is a trend-position state:** its feature signature is dominated by norm_ma_distance (z −0.59 in the
  bear state, +0.65 in the bull state) and norm_fd_close — i.e. the existing 3-state regime essentially encodes
  price-above-vs-below its MA, confirmed from the microstructure features.

## Predictability ceiling (P11) — direction is unpredictable LINEARLY *and* nonlinearly; no lead-lag
The decisive test: a held-out, time-split (train first 70% / test last 30%, pooled across assets) gradient-boosting
model on the 40 dense features, scored on next-bar **direction (AUC)** and **magnitude (R2)**, vs a logistic baseline.

| cadence | direction AUC (GBM) | direction AUC (logistic) | magnitude R2 (GBM) |
|--------|----|----|----|
| 1d  | 0.508 | 0.504 | 0.001 |
| 4h  | 0.515 | 0.513 | 0.016 |
| 1h  | 0.525 | 0.525 | 0.025 |
| 30m | 0.531 | 0.527 | 0.005 |
| 15m | 0.529 | 0.527 | 0.044 |

- **Direction AUC is 0.50-0.53 at every timeframe, and the GBM ~= the logistic** — nonlinearity buys essentially
  nothing. The tiny intraday uptick (30m AUC 0.531) is far too weak to survive the measured cost-cliff (30m gross
  −89.5%). *Direction is intrinsically unpredictable here, not merely linearly unpredictable.* This closes the door on
  "we just need a better/nonlinear model for direction."
- **Magnitude R2 rises toward intraday (0.001 -> 0.044)** — the normalized features predict *size* increasingly well at
  finer cadence (and lagged raw vol predicts it much better still, AC1 up to 0.33). The predictable channel is real.
- **No lead-lag at canonical resolution.** Native-grid (floored-timestamp) BTC-vs-alt cross-correlation is
  **contemporaneous (median best-lag 0, BTC-leads <=3%) at every cadence incl. native 15m** — no exploitable
  BTC->alt timing edge at >=15m granularity (sub-minute is untested and beyond the chimera's canonical cadences).

**Synthesis (the whole mining in one line):** *crypto at canonical resolutions is a one-factor market whose direction
is unpredictable (linearly and nonlinearly) but whose volatility/magnitude is robustly predictable and sharpens
intraday — so the honest edge frontier is volatility/magnitude (breakout, vol-targeting, options-like convex
structures) executed with maker fills, NOT directional trend- or asset-selection.*

## Conditional structure (P12) — the actionable capstone: the vol-expansion setup
The mining's one concretely-actionable, descriptive **setup**: condition on a volatility-expansion bar (local 20-bar
\|ret\| > 1.5x its past-only expanding median) and measure the NEXT bar.

| cadence | next \|ret\| after EXPANSION (bps) | after CALM (bps) | magnitude ratio | up-rate (exp / calm) |
|--------|----|----|----|----|
| 1d  | 582 | 386 | 1.51 | 0.492 / 0.484 |
| 4h  | 270 | 149 | 1.81 | 0.493 / 0.484 |
| 1h  | 150 | 76  | 1.97 | 0.488 / 0.480 |
| 30m | 105 | 52  | **2.02** | 0.495 / 0.481 |
| 15m | 84  | 42  | **2.01** | 0.484 / 0.474 |

**A vol-expansion bar predicts a next move 1.5x (1d) to ~2.0x (intraday) larger — with the up-rate pinned at a coin
flip (0.48-0.49) at every timeframe.** This is the magnitude-not-direction thesis made into a setup: *"something big
is coming, direction unknown"* = a long-gamma / straddle / breakout-convexity trade, not a directional one. The effect
strengthens monotonically intraday, mirroring the vol-clustering result — and is the single most promising honest lead
in this entire exercise (it still must clear `candidate_gate` net of costs, with a convex/maker structure, before belief).

### Intraday seasonality (UTC) + regime-conditioned mean-reversion
- **Volatility concentrates at 14-16 UTC (US equity open) and 00 UTC (UTC day boundary / funding settlement)**; calmest
  at 04-05 and 10 UTC. Hour 14 UTC has the most negative mean return, hour 22 the most positive (consistent 1h & 15m).
  Weekends are quieter (mean \|ret\| 334-367 vs ~440 bps weekday).
- **Mean-reversion is strongest in the bear / below-MA regime** (regime0 AC1 −0.060 at 1d, −0.071/−0.076 at 30m/15m)
  and weakest in the bull regime — so any fade/MR idea should be regime-gated to chop/down states.

## Caveats / honesty
- Descriptive only; nothing gate-validated. AC1<0 at bar level is partly microstructure (bid-ask bounce).
- Co-movement/lead-lag computed on daily-resampled returns for fine cadences (memory-safe) — native intraday lead-lag
  untested. Clustering used the ~50 assets with sufficient overlapping history (newer listings dropped) — survivorship.
- Median lifetime max-DD across u100 is **−94.6%** (1d, VERIFIED) — alt survivorship is brutal; any backtest must use the
  point-in-time universe, not today's survivors.
- GMM regimes are fit on pooled standardized [ret, rolling-vol, trend] — a deliberately simple, interpretable basis;
  a richer regime model (HMM on the dense feature PCs) is an open follow-up.

## Artifacts (reproduce)
`runs/mining/`: `corpus_<cad>.parquet` (per-asset structure), `feature_catalog_<cad>.csv` (per-col health),
`regimes_<cad>.json`, `asset_clusters_<cad>.json`, `structure_<cad>.json`. Engine: `src/mining/chimera_mine.py` +
`src/mining/analyze.py`. Plan: `docs/CHIMERA_MINING_PLAN_2026_06_08.md`.
