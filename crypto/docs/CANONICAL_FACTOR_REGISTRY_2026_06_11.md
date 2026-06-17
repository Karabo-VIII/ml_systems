# Canonical Factor Registry — the whole factor universe in one place (2026-06-11)

> User ask (/orc): *"add these [the new TIs] to the canonical list. The factor list also included
> chimera v51 features, and another dimension I forgot — fan out the whole list."*
>
> **The "another dimension" you forgot = the 19 FRONTIER FEATURE SOURCES** (`config/feature_registry.yaml`)
> — the engineered non-price silver features (derivatives, on-chain, microstructure, options, cross-
> exchange…) that get assembled INTO the chimera v51 gold dataset. So the factor universe is THREE
> dimensions, now consolidated here as the single canonical index:

| Dim | What | Where it lives | Count | Nature |
|---|---|---|---|---|
| **A** | **Technical Indicators** | [TI_MASTER_CATALOG](TI_MASTER_CATALOG_2026_06_11.md) + [config/ti_master_catalog.yaml](../config/ti_master_catalog.yaml) | 8 families, ~110 | **CONFIGURED** on raw OHLC — you pick family+params (a read doesn't exist until configured) |
| **B** | **Chimera v51 features** | [CHIMERA_FEATURE_DICTIONARY](CHIMERA_FEATURE_DICTIONARY.md); gold parquet | 12 families, ~218 | **PRECOMPUTED** gold (fixed, no choice) — the assembled feature surface |
| **C** | **Frontier feature sources** | [config/feature_registry.yaml](../config/feature_registry.yaml) | 19 sources, 141 | **SILVER** per-source engineered features that FEED chimera v51 |

**~469 factors total** across the 3 dimensions (with intentional conceptual overlap — see relationship map).

---

## Dimension A — Technical Indicators (the canonical indicator list, NEW 2026-06-11)
8 families, ~110 indicators, each tagged pandas_ta / HAVE / look-ahead / dead-list. Full:
[TI_MASTER_CATALOG_2026_06_11.md](TI_MASTER_CATALOG_2026_06_11.md). Families: **Trend** (20),
**Momentum** (15), **Volatility** (14), **Volume** (15), **Statistical/Cycle** (11),
**Structure/S-R** (7), **Crypto-Derivatives** (7), **Crypto-Onchain/Macro** (7). This is THE
canonical configured-indicator list; only RSI/MACD/Bollinger are coded today (`indicators_ta.py`),
the rest are pandas_ta one-liners awaiting wiring.

## Dimension B — Chimera v51 features (the precomputed gold surface)
12 families, ~218 features. Full dictionary (per-feature, asset-conditional reads):
[CHIMERA_FEATURE_DICTIONARY.md](CHIMERA_FEATURE_DICTIONARY.md). Family breakdown:
| family | n | what |
|---|---|---|
| `structure` | 7 | price structure & trend |
| `momentum` | 8 | returns & momentum (overlaps Dim-A Momentum) |
| `volatility` | 15 | vol & activity (overlaps Dim-A Volatility) |
| `orderflow` | 25 | order-flow microstructure (buy/sell aggression, CVD, Kyle λ) |
| `liquidity` | 33 | order-book / liquidity (LOB proxy, depth) |
| `derivatives` | 28 | funding / OI / basis (overlaps Dim-A Crypto-Derivatives) |
| `liquidation` | 13 | forced-flow / cascades |
| `positioning` | 14 | long/short ratio, smart-vs-retail |
| `whale` | 6 | large-trade flow |
| `cross_asset` | 58 | cross-asset / relative context (BTC-beta, xrel ranks, TE lead-lag) |
| `social` | 1 | attention (wiki pageviews) |
| `regime` | 10 | precomputed regime labels (SMA-200, vol-state, GMM) |

## Dimension C — Frontier feature sources (the silver feeds = "the dimension you forgot")
19 sources, 141 features (`config/feature_registry.yaml`). These are the per-source engineered
features that get joined into chimera v51:
| source | n | source | n |
|---|---|---|---|
| `s3_features` | 14 | `etf_flows` | 13 |
| `liq_features` | 13 | `stable_flow` | 13 |
| `lob_proxy_daily` | 12 | `book_depth_profile_daily` | 11 |
| `funding_features` | 10 | `basis_features` | 9 |
| `xex_features` | 7 | `te_panel` | 6 |
| `rv_jumps` | 6 | `hawkes_branching` | 5 |
| `premium_features` | 5 | `whale_activity` | 5 |
| `multi_venue_features` | 5 | `cross_exchange_spreads` | 3 |
| `dvol` | 3 | `wiki_pageviews` | 1 |
| `funding_panel` | 0 (panel join) | | |

## How the three dimensions relate (the map)
```
  raw OHLCV + ticks ──┬──> Dimension A: TI lens (configured: pick family+params)  [~110]
                      │
  19 SILVER sources ──┴──> Dimension C: frontier features (engineered per-source) [141]
        │                                          │
        └──────────────── assembled into ─────────┴──> Dimension B: chimera v51 GOLD [~218]
```
- **A is a LENS** (configured on OHLC) — chimera's `momentum`/`volatility` families are a *fixed*
  subset of what A can compute with chosen params; A's crypto-native families (derivatives/onchain)
  overlap chimera's `derivatives`/`positioning`/`liquidation`/`whale`.
- **C is the silver feed** — most of C is already baked into B; C matters when you want the raw
  per-source feature (or to add a NEW source).
- **B is what strategies read today** (the gold parquet via `ChimeraLoader`); A is what we'd *wire*
  for configurable indicator studies; C is the ingestion layer.

## Honest cross-cutting notes (so this is decision-useful, not just a list)
1. **Most of the universe is on disk** (B fully, C fully, A's crypto-native families) — the gap is A's
   classic TI families (unwired pandas_ta one-liners) + the 06§C causal adds (realized skew/kurt 5m,
   Kalman velocity, Amihud-on-time-bars).
2. **The factor universe is NOT a list of edges** — standalone price-TI is HARD-refuted (D50/D52/D55/
   D63/D67/D70); chimera-feature entry-timing is null (D45); positioning-magnet proxy gave AUC 0.49
   (Coinglass SKIP). Factors earn their keep as regime/exit/conditioning layers or cross-sectional
   inputs, not standalone triggers.
3. **The underexploited slices** (lowest dead-list coverage, where a NEW study should look): A's
   **Volatility + Statistical** families (D55's predictable channel + adaptivity), B's `orderflow`/
   `liquidity` microstructure beyond the dead liquidation cells, and the 06§C causal feature adds.
4. **Look-ahead landmines** carried from the TI catalog: ZigZag / centered-DPO REPAINT; full-sample
   z-score/Hurst leak (G-AUDIT-011); Ichimoku cloud is plotted forward. Every factor must be causal.

## The companion: the OTHER 7 axes
This registry catalogs the **SIGNAL axis** (axis 4 of the 8-axis lattice). The other seven axes — chart/bar-type,
cadence, instrument, regime, method, approach, and **entry/exit policy** — are catalogued in
[CANONICAL_STRATEGY_DIMENSION_REGISTRY_2026_06_11.md](CANONICAL_STRATEGY_DIMENSION_REGISTRY_2026_06_11.md)
(+ `config/strategy_dimension_registry.yaml`). Together the two registries fully catalog the strategy decomposition
space: **factors = what you read; dimensions = how you slice, condition, and harvest.**

Canonical entry point: this doc indexes A+B+C. Update it when a source/indicator/feature is added.
Machine-readable: `config/ti_master_catalog.yaml` (A) + `config/feature_registry.yaml` (B+C).
