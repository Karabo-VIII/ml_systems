# Pipeline — V51 v2

The pipeline produces a unified per-asset chimera that's the input to all
training, ranker, and strategy work. This document describes how the layers
fit together, what's where, and how to run each piece.

> **📌 UNIFIED ONE-LINER (2026-05-22)**: To refresh + validate the WHOLE data layer
> in a single command:
>
> ```powershell
> python src/pipeline/refresh_full.py --universe u100 [--force]
> ```
>
> This composes: registry_contract_test → CDAP check_invariants → refresh.py
> --all (DAG walk; includes pre_train_gate) → validate_chimera --asset BTC →
> CDAP check_invariants. Exit codes: 0 clean / 1 warn / 2 CRITICAL. Streams
> producer output via the homogeneous progress convention.
>
> **📌 CANONICAL REFERENCE (2026-05-22)**: For the comprehensive end-to-end
> spec — every producer + every validator + every CDAP invariant + the
> progress convention + troubleshooting — see
> [`docs/DATA_LAYER_CANONICAL_REFERENCE_2026_05_22.md`](../../docs/DATA_LAYER_CANONICAL_REFERENCE_2026_05_22.md).
> This README covers architecture + flow + adding-a-feature; the canonical
> reference covers everything else.
>
> **📌 Convention** (every producer's stdout): see
> [`docs/PIPELINE_PROGRESS_CONVENTION_2026_05_22.md`](../../docs/PIPELINE_PROGRESS_CONVENTION_2026_05_22.md).
> All 25 producers emit lines in the form `[<module>] [<PHASE>] <message>`;
> `refresh.py` streams subprocess stdout to the user's terminal in real time.

## TL;DR

```
RAW DATA (bronze)             SILVER (panels)              GOLD (model-ready)
────────────────────          ──────────────────          ─────────────────────
data/raw/                     data/features/               data/processed/
data/raw_external/            data/bars/                     <SYM>/v51.parquet
                                                              + v51_1d/4h/1h/15m
                                                            <sym>usdt_v50_chimera.parquet (legacy)
```

```
        BarFabric            ChimeraLoader                 Strategy / Model
       ────────────         ───────────────              ───────────────────
        bf.load(             loader.load(                 strategies.run(
          sym, type)            sym, cadence)                ranker.train(
                                                                ChimeraLoader().load_universe('u50'))
```

## Directory layout (canonical, 2026-04-25)

```
data/
├── _config/
│   ├── feature_registry.yaml       # 11 sources × 80 features
│   └── universes_index.yaml        # u10/u50/u100 references
├── _manifests/
│   └── v51_<SYMBOL>.json           # per-build lineage + checksums
├── universes/
│   ├── u10.yaml                    # 10 deep-liquid (BTC, ETH, SOL, ...)
│   ├── u50.yaml                    # 50 audited deployable
│   └── u100.yaml                   # 70+ extended
├── raw/                            # BRONZE: Binance native
│   └── <SYMBOL>/{aggTrades,funding,metrics,bookTicker}/<DATE>.parquet
├── raw_external/                   # BRONZE: non-Binance
│   ├── farside/                    # ETF flows
│   ├── defillama/                  # stablecoin supply
│   ├── binance_futures_panels/     # cross-asset funding panels
│   ├── deribit/                    # DVOL
│   ├── coinbase_okx_bybit/         # cross-exchange klines
│   └── wikipedia/                  # pageviews
├── features/                       # SILVER: feature panels
│   ├── _global/                    # multi-asset wide-format panels
│   │   ├── s3_features_long.parquet
│   │   ├── basis_features_long.parquet
│   │   ├── liq_features_long.parquet
│   │   ├── whale_activity_daily.parquet
│   │   ├── hawkes_branching_daily.parquet
│   │   └── ...
│   └── <SYMBOL>/                   # per-asset silver
│       └── frontier_daily.parquet  # 80 frontier features daily
├── bars/                           # SILVER: bar fabric (per-asset)
│   └── <SYMBOL>/
│       ├── dib/<YEAR>.parquet
│       ├── runs_tick/all.parquet
│       ├── runs_volume/all.parquet
│       ├── range/<YEAR>.parquet
│       └── adaptive_vol/all.parquet
├── processed/                      # GOLD: model-ready
│   ├── <sym>usdt_v50_chimera.parquet  # LEGACY flat (V1-V14 inference)
│   └── <SYMBOL>/                       # NEW per-asset subdir for v51
│       ├── v51.parquet                 # full chimera, 154 cols, dollar bars
│       ├── v51_1d.parquet              # daily cadence
│       ├── v51_4h.parquet
│       ├── v51_1h.parquet
│       └── v51_15m.parquet
└── lob/                            # streaming LOB collector output
    └── <SYMBOL>/<DATE>/<HH>.parquet
```

## How it flows

1. **fetch_all.py** pulls Binance aggTrades/funding/metrics → `data/raw/<SYM>/`
2. **frontier ingest scripts** pull Farside/DeFiLlama/Deribit/Wiki → `data/raw_external/<source>/`
3. **frontier feature scripts** compute panels → `data/features/_global/<panel>.parquet`
4. **frontier_consolidator.py** joins all silver into per-asset → `data/features/<SYM>/frontier_daily.parquet`
5. **make_dataset_legacy.py** legacy v50 chimera → `data/processed/<sym>usdt_v50_chimera.parquet`
6. **make_dataset.py** extends v50 with frontier + v50 fixes + cadences + manifest →
   `data/processed/<SYM>/v51*.parquet` + `data/_manifests/v51_<SYM>.json`

## Strategy/model code reads via ChimeraLoader

```python
from pipeline.chimera_loader import ChimeraLoader

loader = ChimeraLoader()

# Single asset, default cadence (dollar bars + 80 frontier features)
df = loader.load("BTCUSDT", cadence="dollar")

# Daily cadence with metadata
result = loader.load("BTCUSDT", cadence="1d", with_meta=True)
print(result.asset_dna, result.is_u50, result.n_features)

# Universe panel (all U50 assets, 1d cadence)
panel = loader.load_universe("u50", cadence="1d",
                              features=["close", "norm_return_1", "etf_btc_etf_total_z30"])

# Date filter + universe filter combined
df = loader.load("BTCUSDT", cadence="4h",
                 date_range=("2025-01-01", "2026-04-15"),
                 universe="u50")  # raises if BTC isn't in u50
```

## Runtime tools

| Script | Purpose |
|--------|---------|
| `feature_registry.py` | Loader + disk validator for `config/feature_registry.yaml` |
| `universe_loader.py` | Loader for U10/U50/U100 from `config/universes/*.yaml` |
| `frontier_consolidator.py` | Build per-asset silver `<SYM>/frontier_daily.parquet` |
| `make_dataset.py` | Build v51 chimera + 4 cadence views + manifest |
| `bar_fabric.py` | Unified loader for any bar type (dollar/DIB/runs/range/adaptive_vol/v51 cadences) |
| `chimera_loader.py` | Strategy-facing single API; replaces `pl.read_parquet` calls |
| `purge_split.py` | Leak-proof train/val/oos/unseen splits with 5%-cap purge |
| `data_health_check.py` | Audit registry coverage + freshness + universe + DNA consistency |
| `validate_chimera.py` | 14-check validator per asset |
| `cross_asset_consistency.py` | xd_* feature sanity (BTC ref, momentum_rank z-score) |
| `pipeline_e2e_test.py` | 12-stage end-to-end smoke test |
| `cadence_correctness.py` | Verify cadence views match dollar-bar resampling |
| `v50_backward_compat.py` | Verify V1-V14 still loads cleanly + v51 identity-matches v50 |
| `pre_train_gate.py` | Compose 5 validators into single CI hook |
| `strategy_parity_test.py` | Verify v51 features identical to raw frontier sources |

## Adding a new feature

1. Edit `config/feature_registry.yaml` — add a new source spec.
2. Run `python src/pipeline/feature_registry.py` to validate the spec.
3. Run `python src/pipeline/frontier_consolidator.py --asset BTC` to test silver.
4. Run `python src/pipeline/make_dataset.py --asset BTC` to test gold.
5. Run `python src/pipeline/validate_chimera.py --asset BTC` to confirm.

No code changes needed for additive feature sources.

## Two YAMLs, two schemas — do not confuse (2026-05-20)

There are **two** feature YAMLs in `config/`. They serve different consumers and have incompatible schemas. CDAP enforces the split — see `config/_invariants.yaml::required_patterns::feature_registry_yaml_has_sources_key`.

| File | Schema | Consumers | Writers |
|---|---|---|---|
| `config/feature_registry.yaml` | `sources:` + `chimera_v51:` (pipeline-spec) | `src/pipeline/feature_registry.py` and downstream (frontier_consolidator, validate_chimera, make_dataset, data_health_check, pipeline_e2e_test, transfer_entropy_panel) | **HAND-EDIT ONLY**; never overwritten by scripts. |
| `config/feature_catalog.yaml` | `meta:` + `prefix_families:` (metadata catalog) | (no auto-consumers yet — discoverable via `src/pipeline/feature_catalog.py`) | `scripts/audit/build_feature_registry.py` regenerates this; safe to overwrite. |

**Provenance (2026-05-20)**: commit `c59c4e7` accidentally pointed the metadata-catalog builder at `feature_registry.yaml`, breaking 6 pipeline consumers with `KeyError: 'sources'`. Fix in `7fa8bbd` restored the pipeline-spec YAML and rerouted the builder to `feature_catalog.yaml`. CDAP invariants added to prevent recurrence: `feature_registry.yaml` must contain `sources:` and `chimera_v51:`, and must NOT contain `prefix_families:` or top-level `meta:`.

**What lives in `feature_catalog.yaml` and how to read it** — see `src/pipeline/feature_catalog.py::FeatureCatalog.load()`. Per-feature: `is_z_scored`, `preserves_magnitude`, `is_cross_asset`, `expected_range`, `ks_winner_v_nonmover`, source script, semantic class. Useful for: ML feature-import sanity checks (catch the C4 `norm_*` trap), feature-selection at train time, post-hoc audits.

**Note on registry scope**: the pipeline registry tracks the 16 sources joined into frontier_silver / chimera v51 (s3, liq, stbl, etf, lob, bd, bs, rv, te, hbr, wh, mv, xex, dv, soc, fp = 120 features). Post-chimera enrichment stages — `add_xrel_features.py` (xrel_*, 21 cols), xd_* (7 cols), tick_*, etc. — write directly to the chimera parquet and are NOT registered. Use `feature_catalog.yaml` to discover those.

## Pre-train CI hook

Before any training session:

```
python src/pipeline/pre_train_gate.py --asset BTC --quick
```

Exit 0 = clean. Exit 1 = warns only (proceed with caution). Exit 2 = hard fail (stop).

## V50 fixes captured in V51 v2

See `docs/V50_TO_V51_FIXES.md` for the full catalog. Highlights:
- `tick_seq` column for zero-diff timestamp tiebreaking
- `target_return_<h>_raw` (uncapped) alongside v50's clipped versions
- `returns_clean` without silent fill_null(0) on first bar
- `is_u10/u50/u100` + `asset_dna` columns inline (no code lookups needed)
- per-asset manifest with input checksums + lineage
- 4 cadence materializations (1d/4h/1h/15m) instead of single cadence
