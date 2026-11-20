# Chimera Feature-Space Mining — Plan (2026-06-08, 6h autonomous)

**Mandate (user, 2026-06-08 23:58 SAST):** *"decompose crypto and all the chimera features across all assets and
different time frames. You have 6 hours. Plan properly and come back to me with findings. this is a mining exercise of
sorts. Add things like regime, clusters, trends... etc."*

**Window:** 2026-06-08 23:59:36 -> 2026-06-09 05:59:36 SAST (VERIFIED anchor). Autonomous (AUTONOMY_ON + watcher 61052).

## What this is (and isn't)
- **IS:** an UNSUPERVISED / DESCRIPTIVE mining of the chimera feature space — characterize its *structure* (regime,
  clusters, trends, factor/redundancy) across all assets x canonical timeframes. The deliverable is a **findings
  report + reproducible artifacts**, not a deployable strategy.
- **ISN'T:** a supervised edge-search. The 1d/4h/1h/30m adaptive-MA edge-search is a VERIFIED-HONEST NULL
  ([[project-oracle-decomposer-dna-2026-06-08]]); this exercise maps the *terrain* that a future edge-search would
  exploit. **Honesty rail:** descriptive co-occurrence != predictive alpha; any "feature X precedes move Y" finding is
  hypothesis-generating and must be labelled as such (no look-ahead, no gate-claim without the candidate_gate).

## Scope (the breadth grid)
- **Assets:** all u100 available per cadence (104 chimera files at 1d/4h/1h/30m/15m).
- **Timeframes:** 1d, 4h, 1h, 30m, 15m (universe-wide coverage; dollar/dib excluded — sparse u10 coverage).
- **Features:** the full 243-col chimera v51 schema, partitioned into families: norm_* (33 microstructure/flow),
  xd_* (7 cross-asset), regime (2), returns (18), vol (20), sma (6), targets (14, used only as descriptive outcomes,
  never as inputs to a predictive claim), and the OHLCV/meta remainder.

## Reuse (don't reinvent — [[project-harness-integrate-not-reinvent-2026-06-07]])
- `src/narrate/feature_map.py` + `crypto_context.py` — feature semantics / families.
- `src/wealth_bot/regime_router/regime_classifier.py` — existing regime labels.
- `src/oracle/decomposer.py` — per-asset/cadence series loader + chimera context join.
- `pipeline.chimera_loader.ChimeraLoader` — canonical data access.

## Method — 6 phases (each emits a verifiable artifact; overseer RWYB-judges before the report uses it)
**Memory-safe rule:** stream ONE chimera file at a time, select only needed columns, emit per-(asset,cadence)
AGGREGATES to a compact corpus. Never hold the full 15m x u100 panel in memory (that OOM'd the TF-sweep).

- **P1 Feature inventory + health** -> `runs/mining/feature_catalog.csv`: per cadence, every column's presence-across-
  u100, missing-fraction, mean/std (std~1 health), constant/degenerate flags, family tag.
- **P2 Mining corpus** -> `runs/mining/corpus_<cadence>.parquet`: per (asset,cadence) row of aggregates — return
  moments (mean/std/skew/kurt), realized vol, max-DD, autocorr(1), variance-ratio, Hurst, trend fraction
  (time above SMA200), and the mean/std/last-decile of each feature family. The base table for P3-P5.
- **P3 Regime decomposition** -> `runs/mining/regimes_<cadence>.{parquet,json}`: (a) rule regimes (bull/bear via
  SMA200; vol hi/lo) time-shares + transition matrices; (b) UNSUPERVISED discovered regimes (GMM/k-means on
  [ret, realized-vol, trend-strength, dispersion]) — k chosen by BIC/silhouette; per-regime return/vol profile +
  persistence. Cohort (BTC-driven) vs per-asset.
- **P4 Asset clustering** -> `runs/mining/asset_clusters_<cadence>.{csv,json}`: (a) return-correlation hierarchical
  clusters (co-movement blocks; BTC-beta vs idiosyncratic); (b) feature-signature clusters (which assets have similar
  microstructure DNA); (c) BTC/ETH lead-lag (cross-correlation at +-k lags).
- **P5 Trend character + feature structure** -> `runs/mining/structure_<cadence>.{csv,json}`: (a) trend-vs-mean-revert
  map per asset x TF (Hurst, variance-ratio, AC sign); (b) feature redundancy (|corr| blocks) + PCA -> effective
  dimensionality (# PCs for 90% var) + top-PC interpretation (is PC1 a market factor? PC2 a vol factor?).
- **P6 Findings report** -> `docs/CHIMERA_MINING_FINDINGS_2026_06_08.md`: synthesize P1-P5 into ranked, honestly-
  caveated findings + "what this implies for the next edge-search" (the bridge back to returns). Commit.

## Execution
Build `src/mining/chimera_mine.py` (reusable engine: P1+P2 streaming corpus builder) + `src/mining/analyze.py`
(P3-P5 on the compact corpus). Run cadence-by-cadence (1d->4h->1h->30m->15m; heavy cadences backgrounded with
incremental writes). Overseer builds + RWYB-judges each artifact; Agent workers parallelize interpretation. Stop-hook
continues the overseer across turns until the report is committed or the window closes.

## Stop conditions / honesty
- Window end 05:59:36, OR the report is committed + all 5 artifact families RWYB-verified.
- IDLE-STOP: if a phase saturates (no new structure), move to the next axis; don't grind.
- Every number in the report is RWYB-recomputed by the overseer; descriptive-not-predictive caveat on every
  "feature precedes move" statement; no emoji (cp1252).
