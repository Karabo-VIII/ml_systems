# Data Contract — New Crypto Pipeline Layer

**Status:** accepted 2026-11-20. Open questions in §8 are closed by the Decisions section below. Those decisions are owner-changeable; they live in config, not in scattered literals.  
**Date:** 2026-11-20 (Africa/Johannesburg).  
**Scope:** what we fetch, what we compute, how every series is timestamped and joined, and how we prove charts and ML features cannot be misaligned or silently empty.  
**Reference (read-only):** the archived tree at `crypto/archive/pipeline_v1/` (moved from `crypto/src/pipeline/`) and `crypto/config/`. File references below use the old `crypto/src/pipeline/` paths.  
**Non-goals for this document:** implementation, FX ingest (interface only), model training, live execution.

---


## Decisions (owner-accepted 2026-11-20, owner-changeable)

These close §8. Change them in `crypto/src/pipeline_v2/contract/` (and the split block below). Do not silently substitute a different rule in code.

The v1 tree now lives at `crypto/archive/pipeline_v1/` (reference only). The draft suggested `crypto/src/_archive/pipeline_legacy_2026_11_20/`; the archive used here is `crypto/archive/pipeline_v1/`.

1. **Bar types.** Core bars are true time bars: a 1-minute base aggregated to 5m, 15m, 1h, 4h, and 1d with real OHLCV (open = first, high = max, low = min, close = last; volumes and trade counts are sums), plus dollar bars. Tick bars and volume bars are optional and are built only when requested. DIB, runs, range, and adaptive-vol are research-only and are omitted from pipeline v2.
2. **Universe.** The first universe is U10 (`crypto/config/universes/u10.yaml`: BTC, ETH, SOL, BNB, XRP, DOGE, ADA, AVAX, LINK, LTC).
3. **History start.** 2020-01-01 UTC, or the listing date in `crypto/config/asset_launch_dates.json` when that date is later. Every run is bounded by `--start` and `--end`.
4. **Timestamps.** `ts_ms` int64 is the canonical clock on bars, features, panels, and gold datasets. Raw trades also keep `ts_us` and `agg_trade_id`. `ts_ms = ts_us // 1000`.
5. **Dollar-bar thresholds.** v2 uses the current fixed table, versioned at `crypto/src/pipeline_v2/contract/dollar_thresholds.yaml` (BTC $2,000,000, ETH $700,000, SOL $200,000, BNB $300,000, XRP $350,000, DOGE $400,000, ADA $100,000, AVAX $80,000, LINK $70,000, LTC $50,000). Causal trailing-median recalibration is a TODO. It is not applied.
6. **Labels.** The default training labels are uncapped `target_return_{1,4,16,64}`. Clipped copies and `target_voladj_*` (the denominator is future realized volatility) are optional and off unless requested. Vol-adjusted columns are labels only, never features.
7. **Splits and embargo.** The frozen boundaries stay: train `ts` date `< 2023-07-01`, val `< 2024-05-15`, oos `< 2025-03-15`, unseen `>= 2025-03-15`. The embargo is `max(horizon, window)` bars on that file's bar type, dropped from the tail of the earlier segment and the head of the later one. `horizon` is 64. `window` is the longest trailing lookback of the enabled features (fractional-difference window + z-score window, 1200 bars with the default windows). The draft's extra 7-day calendar floor and the old 400-bar cap with a silent 5% shrink are not used. If a segment is shorter than the embargo, the gate fails rather than shrinking the embargo.
8. **Bar clock and joins.** A bar covering `(t_open, t_close]` is labelled `ts_ms = t_close`. Every non-native series joins by a backward as-of on `knowable_at <= ts_ms`.

Also fixed while closing the draft, because two sentences disagreed:

- A daily UTC aggregate of day D is knowable at **(D+1) 00:00:00.000 UTC** and not before. Implementation: `event_time = D 00:00:00 UTC` and `lag_ms = 86_400_000`. The draft's other wording (`event = D 23:59:59.999` plus another full day) would publish the value only at the end of D+1 and double-count the lag. That wording is not used.
- Funding is knowable at `calc_time` (`lag_ms = 0`). No second 8-hour lag is stacked on top.
- `norm_oi_change` and `norm_oi_price_divergence` stay out of the default feature set until a full-history lookahead retest. `--include-oi` turns them on. When on, open interest is as-of joined with a 5-minute publication lag and is not 0-filled.


## 0. Key decisions in this draft (for approval)

1. **UTC everywhere.** All stored timestamps are timezone-naive Unix instants interpreted as UTC.
2. **Canonical time unit is milliseconds** (`ts_ms`: int64). Raw aggTrades also keep `ts_us` (source microseconds) and `agg_trade_id` as the trade primary key. Every bar, feature, and dataset row is labeled in `ts_ms`.
3. **Bar label = close.** A bar covering `(t_open, t_close]` is stamped `ts_ms = t_close`. Features may be joined onto a bar only when `knowable_at <= ts_ms`.
4. **True time bars are first-class**, built from 1-minute (or tick) data with proper OHLCV aggregation. The old v51 “1d/4h/1h/15m cadence views” are **not** time bars: they take the *last dollar bar in the bucket* (`make_dataset.py` `materialize_cadence`, `group_by(ck).last()`, and `feature_registry.yaml` `cadence_materializations.agg: last`). Those views must not be used as chart or ML time series.
5. **Core bar set:** 1m → 5m/15m/1h/4h/1d time bars, plus dollar bars. Tick/volume bars are in the core if wanted for flow work. DIB, runs, range, and adaptive-vol bars are optional research fabrics, not required for the default chart/ML path.
6. **Joins are bitemporal.** `knowable_at = event_time + publication_lag_ms`. The old `asof_join.py` already defines this, but **it is not wired into** `frontier_consolidator.py` or `make_dataset.py` (only a self-test in `asof_join.py`). The new pipeline will use it for every non-native series.
7. **No silent fill of prices or returns with 0.** Warmup and missing joins stay null. Downstream must mask them.
8. **Default build universe for the first solid cut: U10** (`config/universes/u10.yaml`: BTC, ETH, SOL, BNB, XRP, DOGE, ADA, AVAX, LINK, LTC). Schema must not assume a single asset (cross-asset features are undefined or null until ≥2 peers exist).
9. **History start:** 2020-01-01 UTC for assets listed by then; otherwise the listing date in `config/asset_launch_dates.json`. Fetch and compute always take explicit `--start` / `--end`.
10. **Primary ML labels are raw forward returns** `target_return_{1,4,16,64}` (uncapped + optionally clipped copies). Vol-adjusted targets (`target_voladj_*`) use *future* realized vol as the denominator (`sota_shared_logic_v50.py` `add_strategic_targets`); they are optional and must not be the default training target (this matches the existing CLAUDE.md invariant).

---

## 1. Global conventions

| Rule | Contract | Old pipeline |
|---|---|---|
| Timezone | UTC only. No local-time stamps, no exchange “session” tz. | Generally UTC (e.g. `make_dataset_legacy.py` uses `timezone.utc` for the date suffix). |
| Canonical unit | `ts_ms` int64 Unix milliseconds on **all** bars, features, panels, and gold datasets. | Mixed: Vision aggTrades are microseconds (16-digit); funding/metrics are milliseconds. `standardise_time_column` / `normalize_ts_to_ms` / `detect_ts_unit` convert on read. Easy to miss. |
| Raw trade clock | Keep `ts_us` (int64) and `agg_trade_id` (int64) on bronze trades. Derive `ts_ms = ts_us // 1000`. | `_parse_aggtrades_zip` stores the raw `timestamp` only and **drops** `agg_trade_id` (`fetch_all.py`). |
| Sort | Every file is sorted ascending by its primary key before write. Trades: `(asset, ts_us, agg_trade_id)`. Bars/features: `(asset, ts_ms)` or `(asset, date)` for daily panels. | Vision CSVs are sorted; `_parse_aggtrades_zip` calls `.unique(...)` **without** `maintain_order=True`, which shuffles every daily parquet. Downstream `prepare_aggtrades` sorts; any consumer that forgets to sort is wrong. Metrics files are also unsorted on write. |
| Dedupe | Trades: unique on `agg_trade_id` (keep last). Bars: unique on `(asset, ts_ms)` (keep last). Daily panels: unique on `(asset, date)`. | Unique on `(timestamp, price, qty)` — can collapse distinct trades and drop the id. |
| Bar open/close | Interval is left-open, right-closed: `(t_open, t_close]`. `ts_open_ms` and `ts_ms` (= close) are both stored. | Dollar bars stamp `timestamp = last trade` (close) only (`generate_dollar_bars`). Alt bars store `bar_start_ts` and `bar_end_ts`. Time “cadences” are not intervals at all (see §3). |
| Symbol | Canonical `BTCUSDT`. CLI accepts `BTC`, `BTC/USDT`, `BTCUSDT`. Column `asset` is always the canonical form. Paths use the same token. | Mix of `BTC/USDT` (config), `BTC` (`--asset`), `BTCUSDT` (dirs), `btcusdt` (filenames). `layout.normalize_asset` already does this; the new layer will do it once, at the edge. |
| Missing data | **Never** `fill_null(0)` on price, volume, or returns. Feature nulls during warmup or failed joins stay null. A manifest records expected vs written row counts. | `calculate_v50_features` forward-fills then 0-fills many raw series; `returns` is `pct_change().fill_null(0)` (v51 adds `returns_clean` without that fill). `chimera_v51.missing_value_policy: forward_fill` with `forward_fill_max_days: 7` (`feature_registry.yaml`). `long_short_ratio` is literally `1.0` when the column is absent (`fetch_all.standardize_metrics`). |
| Publication / as-of | Every non-native feature has `event_time` and `knowable_at = event_time + lag_ms` from a lag registry. Attach to a bar iff `knowable_at <= bar.ts_ms` (backward as-of). Lags round **up**. | Lag registry exists (`config/feature_publication_lag.yaml`: ETF/wiki/onchain/whale = 1d, funding = 8h) and `asof_join.asof_join_knowable` implements the rule, but producers do not call it. v51 uses a blanket `date + 1 day` then join on calendar date (`make_dataset.attach_frontier`). Funding/OI use backward as-of with **24h / 48h tolerance** and **no lag** (`make_dataset_legacy.py`). |
| Writes | Atomic: write `{path}.tmp.{pid}` → fsync → rename. Optional sidecar `{path}.sha256`. Idempotent given the same `(asset, start, end, schema_version, params_hash)`. | `parquet_io.atomic_write_parquet` exists and is used by many producers; not universal (some pandas `to_parquet` paths remain). |
| Layout | See §7. Partition bronze by `asset/yyyy-mm-dd`. Silver/gold are per-asset files dated by **data-end date** (`YYYYMMDD`), never by the run’s `--start` year. | `layout.py` is dated snapshots. Alt-bar filenames use `args.start[:4]`, so 2026 data was written as `BTCUSDT_dib_2023.parquet` (default `--start 2023-01-01`). Hawkes registry path omits the date suffix (`hawkes_branching_daily.parquet` vs `hawkes_branching_daily_20261118.parquet`). |
| Schema version | Every parquet carries metadata `contract_version`, `schema_name`, `producer`, `params_hash`. Breaking changes bump `schema_name` (e.g. `trades_v1` → `trades_v2`). | Informal `__contract__` dicts in some modules; CDAP `config/_invariants.yaml`; v51 manifest (`make_dataset.py`). No uniform parquet metadata. |
| Determinism | Same inputs + params → bit-identical output (except recorded `built_at`). | Generally attempted; shuffled unique() and unsorted writes already violate this at bronze. |

**Clock policy for lags (recommendations, overridable in the lag registry):**

| Source family | Default `lag_ms` | Rationale |
|---|---|---|
| Own trades / own time or dollar bars | 0 | Knowable at close of the bar. |
| Binance Vision daily metrics / funding settlement | funding: 0 at `calc_time` if we treat settlement as contemporaneous; otherwise 8h. **Decision: 0 at `ts_ms = calc_time`, do not apply a second 8h lag on top.** | Old lag YAML uses 8h (the interval), which double-counts if event_time is already the settlement instant. |
| ETF flows, Wikipedia, DeFiLlama, whale daily aggregates | 86_400_000 (1 day) | Same as `feature_publication_lag.yaml`. |
| Daily silver panels built from a *full UTC day* of data (s3, liq, rv, hawkes, lob daily, …) | 86_400_000 | A day-D aggregate is not knowable until D+1 00:00 UTC (old `attach_frontier` comment, 2026-05-24). |
| Cross-exchange mid prices | venue-specific; default 0 if the candle close is the event, plus 60_000 ms safety. | — |

---

## 2. Raw sources (tier 0)

Prefer `https://data.binance.vision` (S3, no API key). `api.binance.com` and `fapi.binance.com` return HTTP 451 from at least some environments (observed 2026-11-20). API fallbacks are optional and must not be required for a complete bronze build.

### 2.1 Inventory and verdict

| Source | Old producer | URL / method | Granularity | Fields (contract) | Lag | Keep / fix / drop | What the old code got wrong |
|---|---|---|---|---|---|---|---|
| Spot aggTrades | `fetch_all.py` `sync_trades` | `data.binance.vision/data/spot/daily/aggTrades/{SYM}/{SYM}-aggTrades-YYYY-MM-DD.zip` | per trade | `agg_trade_id` i64, `ts_us` i64, `ts_ms` i64, `price` f64 (USDT), `qty` f64 (base), `is_buyer_maker` bool, `is_best_match` bool (optional) | 0; daily zip usually T+1 (T itself 404 until published — e.g. 2026-11-19 was missing on 2026-11-20) | **KEEP, FIX** | Drops trade id; stores unsorted after `unique()`; no `--end-date`; `run_pipeline` cannot pass `--start-date` (defaults to `config/data_config.yaml` `2020-01-01`). |
| 1m spot klines | `fetch_all.py` `sync_klines` (`--trade-mode klines\|both`); also `ingest/binance_spot_klines.py` (monthly, used for basis) | Vision `spot/daily/klines/{SYM}/1m/` and monthly 1d klines | 1m / 1d | `ts_open_ms`, `ts_ms` (close), OHLCV, `quote_volume`, `n_trades`, `taker_buy_base`, `taker_buy_quote` (Binance kline schema) | 1m daily zip ~T+1 | **KEEP, PROMOTE** | 1m klines are optional and unused as the time-bar base. Daily klines are only a basis input. The ML “time cadences” were not built from these. |
| UM futures funding | `fetch_all.py` `sync_funding` | Official URL is **daily** `futures/um/daily/fundingRate/…` — **that prefix is now empty**. Live path is **monthly** `futures/um/monthly/fundingRate/{SYM}/{SYM}-fundingRate-YYYY-MM.zip`. API fallback `fapi.binance.com/fapi/v1/fundingRate` (451 here). | 8h settlement | `ts_ms` (calc_time, ms), `funding_rate` f64 (fraction, not percent), `funding_interval_hours` i32 | Settlement time = event_time; zip published after month-end for the current month (Nov 2026 monthly was 404 mid-month); intra-month must be allowed to be incomplete or filled from a non-blocked fallback | **KEEP, FIX** | Looks only at the dead daily path; parser expects `calcTime`/`fundingRate` but current monthly CSV is `calc_time`/`last_funding_rate` (`_parse_funding_zip`). Wide panel (`ingest/binance_funding_panel.py`) skips assets with &lt;90 days and then prints “no raw funding found”. |
| UM futures metrics (OI) | `fetch_all.py` `sync_metrics`; also `ingest/binance_s3_metrics.py` (longer Vision history) | `data.binance.vision/data/futures/um/daily/metrics/{SYM}/{SYM}-metrics-YYYY-MM-DD.zip` (works) | 5 min (288 rows/day) | Keep whatever Vision actually ships: at least `ts_ms`, `sum_open_interest_value` (USDT). Do **not** invent columns. | ~T+1 for the daily zip | **KEEP, FIX** | `standardize_metrics` maps OI value correctly, but if no long/short ratio column exists it writes `long_short_ratio = 1.0` for every row. That dead series later looks like a real feature. Files written unsorted. |
| Book depth snapshots | `ingest/binance_vision_book_depth_profile.py` | Vision `futures/um/daily/bookDepth/{SYM}/` | periodic snapshots | Keep snapshot ts, bid/ask notional per level as the ingest already derives; daily profile is a silver aggregate | ~T+1 | **KEEP** (optional for v1 core; needed for `bd_*`) | Works on Vision when `--start` is modest. Not on the runner’s critical path cleanly. |
| BookTicker | Manifest slot in `fetch_all.py` (`bookTicker`) | Vision has `futures/um/daily/bookTicker/` | tick of top of book | — | — | **DROP from v1** unless a producer is written | Declared in the fetch manifest schema; no `sync_bookTicker` implementation found. |
| Farside ETF flows | `ingest/etf_flows.py` | `https://farside.co.uk/bitcoin-etf-flow-all-data/` and ethereum equivalent (HTML scrape, no key) | daily (USD millions, per issuer + total) | `date` (UTC date), per-issuer flows f64, `total_usdm` f64 | 1 day | **KEEP, FIX** | Works, but ETH headers can be dirty (`ETHA 0.25%`, `Total Unnamed: 12_level_1`). Normalize issuer names in the contract schema. |
| DeFiLlama stables | `ingest/defillama_stable_flows.py` | `https://stablecoins.llama.fi` | daily | date, aggregate supply / flow features as produced (`stable_flows_daily`, `stable_flow_features`) | 1 day (conservative; revisions happen) | **KEEP** | Works end-to-end. Frontier used its calendar (2017-11-29 onward), leaving years of nulls for other families — join calendar must be the **asset’s** bar calendar, not the longest source. |
| Deribit DVOL | `ingest/deribit_dvol.py` | `https://www.deribit.com/api/v2/public/get_volatility_index_data` | daily OHLC of the vol index | `date`, `asset` (BTC/ETH), `dv_dvol_{open,high,low,close}` | 0 at index close, plus 1h safety optional | **KEEP** | Works. |
| Wikipedia pageviews | `ingest/wikipedia_pageviews.py` | Wikimedia REST pageviews | daily | `date`, `asset`, `soc_wiki_views` i64 | 1 day | **KEEP** | Works. |
| Cross-exchange spreads | `ingest/cross_exchange_spreads.py` + `features/xex_features_daily.py` | Coinbase / Bybit / OKX public candles vs Binance | daily (derived) | mid/close per venue, spread vs Binance in bps | 1 day or 60s + close | **FIX or defer** | Bybit 403 then `KeyError: 'date'` on an empty frame aborts the whole job; Coinbase/OKX never finish. `feature_registry.yaml` 1.6.0 already removed raw `cross_exchange_spreads` from `sources_to_join` because of a name collision with `xex_features` (duplicate cols, 67–73% nulls). |
| Multi-venue listings | `ingest/multi_venue_listings.py` | Binance `fapi` exchangeInfo (451), Bybit (403), OKX (works) | event catalogue | `venue`, `symbol`, `onboard_ts_ms`, `contract_type` | 0 | **KEEP, FIX** | `--smoke` can pass without Binance. `--once` returned OKX-only data; derived `is_multi_venue` was 0. Exit 0 must not mean “all venues fetched”. |
| Liquidations (approx) | `ingest/liquidations_approx.py` | derived from local aggTrades (not the official liq feed) | daily | approximate liq notional from large aggressive trades | 0 (from own trades) + 1d if used as a daily panel | **KEEP as a *proxy*, labeled as such** | Honest name already says approx. Must not be presented as exchange liquidation prints. |
| Whale activity | `ingest/whale_activity.py` | derived from local aggTrades ($ size thresholds) | daily | counts / notionals of large prints | 1d as a daily panel | **KEEP** | Works from local trades. |
| Token unlocks | `build_panels.py` points at `ingest/coingecko_unlocks.py` | — | — | — | — | **DROP** | File does not exist. Comments already say the idea was a heuristic proxy, “DO NOT trade live”. |

### 2.2 Bronze storage (proposed)

```
data/bronze/
  trades/spot/{ASSET}/yyyy-mm-dd.parquet      # aggTrades, contract trades_v1
  klines/spot_1m/{ASSET}/yyyy-mm-dd.parquet     # 1m OHLCV, contract kline_1m_v1
  funding/um/{ASSET}/yyyy-mm.parquet            # monthly source, also exploded daily for convenience
  metrics/um/{ASSET}/yyyy-mm-dd.parquet         # OI etc., only real columns
  book_depth/um/{ASSET}/yyyy-mm-dd.parquet      # optional
  external/{source}/...                         # etf, defillama, deribit, wiki, listings
```

Every bronze file includes `source_url`, `retrieved_at_ms`, `schema_name=…_v1`, and a per-day fetch manifest (`ok | missing_not_published | error`).

---

## 3. Bars (tier 1)

### 3.1 Shared bar schema (`bars_v1`)

Every bar type writes the same core columns so charts and features can share code:

| Column | Dtype | Meaning |
|---|---|---|
| `asset` | str | Canonical symbol |
| `bar_type` | str | `time_1m`, `time_5m`, `time_15m`, `time_1h`, `time_4h`, `time_1d`, `dollar`, `tick`, `volume`, `dib`, `runs_tick`, `runs_volume`, `range`, `adaptive_vol` |
| `bar_id` | i64 | Monotone unique id per `(asset, bar_type)`, 0-based in the file’s history |
| `ts_open_ms` | i64 | Start of interval (first trade ts, or bucket open) |
| `ts_ms` | i64 | **Close label** (last trade ts, or bucket close) |
| `open, high, low, close` | f64 | USDT. Invariant: `high >= max(open, close)`, `low <= min(open, close)`, `high >= low` |
| `volume` | f64 | Base-asset quantity |
| `volume_usd` | f64 | Sum of `price * qty` (or VWAP × volume) |
| `vwap` | f64 | `volume_usd / volume` (null if volume=0) |
| `buy_volume`, `sell_volume` | f64 | Base qty from taker side. Binance: `is_buyer_maker=True` ⇒ **sell** aggressor (already the convention in `generate_dollar_bars`) |
| `buy_usd`, `sell_usd` | f64 | Same split in USDT |
| `n_trades` | i64 | Trade count in the bar (must reconcile to bronze for time/dollar/tick/volume/DIB/runs) |
| `signed_usd` | f64 | `buy_usd - sell_usd` |

No bar file is valid unless it is sorted by `ts_ms`, `bar_id` is unique and non-decreasing, and `ts_ms` is monotone.

### 3.2 Types

| Bar type | Definition | Threshold / params | Calibration (causal only) | Timestamp | Verdict |
|---|---|---|---|---|---|
| **Time 1m (base)** | Bucket trades (or official 1m klines, cross-checked) into UTC minutes. Aggregate OHLCV / vwap / buy-sell / n_trades. | Clock. Empty minutes are **omitted** (crypto is 24/7 but a silent minute is a gap, not a zero bar), except we record gap flags. | None | `ts_open_ms = bucket_start`, `ts_ms = bucket_start + 60_000` | **CORE (new).** Old pipeline can fetch 1m klines but does not build a time-bar fabric from them. |
| **Time 5m / 15m / 1h / 4h / 1d** | *Aggregate* 1m bars (or trades) into the coarser bucket: open=first, high=max, low=min, close=last, volumes and trade counts **sum**. | Clock | None | Close of the bucket (e.g. the 14:00–15:00 UTC hour is labeled 15:00:00.000) | **CORE (new).** Replaces v51 cadence views. **Old bug:** `materialize_cadence` / registry `agg: last` copies the last *dollar bar* in the window. A “1h” row’s volume is one dollar bar’s volume (~$2M for BTC), not the hour’s volume. Targets were later recomputed at cadence grain (`recompute_cadence_targets`), but OHLC/volume/features were not. Charts and features on those files are misaligned by construction. |
| **Dollar** | Close a bar when cumulative `price*qty` since last close ≥ threshold (`generate_dollar_bars`: `floor((cum_usd + residual) / threshold)` with residual carried across chunks). | Per-asset fixed USD threshold from a config table. Old BTC: `$2,000,000` aiming at ~5-minute bars at median ~$500M/day (`data_config.yaml`). | Fixed thresholds, **recalibrated causally** on a trailing window that ends before the bar (e.g. prior 30d median daily volume / 288). Never use future volume to set the threshold of a historical bar. Adaptive mid-history changes must be dated and versioned. | Close = last trade `ts_ms`; also store `ts_open_ms` | **CORE.** This is the old gold base (`make_dataset_legacy.py`). Keep as the default *event-time* fabric for microstructure features. |
| **Tick** | Close when trade count since last close ≥ N. | Per-asset N, target ~same bars/day as dollar (old tick-runs target was ~30/day in `alt_bar_thresholds.yaml` — that is a different “runs” notion; see below). | Causal trailing median ticks/day / target bars/day | Close = last trade | **CORE-OPTIONAL.** Simple and chartable. Not first-class in the old gold path. |
| **Volume** | Close when cumulative base qty ≥ V. | Per-asset V, same calibration style | Causal trailing median | Close = last trade | **CORE-OPTIONAL.** |
| **DIB** (dollar imbalance) | AFML: accumulate signed USD; close and **reset** when `|accum| ≥ threshold` (`bars/dib_bars_fast.py`, `imbalance_bar_ids`). | Per-asset USD; old hardcoded BTC `$2,000,000`, also YAML `dib_threshold_usd` (BTC `5,920,200` in `alt_bar_thresholds.yaml` — two sources already disagree). | Causal; pick **one** table. Target ~200 bars/day was the YAML goal. | `bar_start_ts` / `bar_end_ts` today; new schema uses `ts_open_ms` / `ts_ms` | **OPTIONAL.** Used by `lob_proxy_panel.py`. The file itself says other alt types “currently have no chimera consumer.” |
| **Runs (tick / volume)** | Old code: close when cumulative signed ticks or volume exceeds a threshold (`runs_bars.py` — a simplification of AFML “expected run length”). | YAML `tick_runs_threshold` / `vol_runs_threshold` (BTC tick 49,335; vol 564). Recalibrated 2026-05-13 after a bad proxy. | Causal trailing | close | **OPTIONAL.** Research only until a strategy actually consumes them. |
| **Range** | Close when `|price - bar_open| ≥ range_threshold`. BTC was 0.5% of price (`RANGE_THRESHOLDS`). | Fraction of open (or absolute). Auto-widen if estimated bars/day > 10k. | Causal (threshold from prior data only) | close | **OPTIONAL.** |
| **Adaptive-vol dollar** | Dollar threshold scaled by `T_base * max(σ_day/σ_30d, 0.3)` (`adaptive_vol_bars.py`). | `T_base` from YAML (`adaptive_vol_base_threshold_usd`, BTC ~23.7M). σ must be **causal** (no same-day or future vol). | Causal EWMA / trailing 30d | close | **OPTIONAL.** File notes it “currently re-reads the full date range” and was “orphan compute (no chimera consumer).” |

### 3.3 Recommended default for charts and ML

- **Charts (default):** `time_1h` and `time_1d`, with a control to switch to dollar or 15m. These are real clock bars, so a 1h candle is a 1h candle.
- **ML default fabric:** dollar bars (microstructure, existing 34-feature physics) **and** `time_1h` / `time_1d` (for daily/hourly features and any model that should think in clock time).
- **Do not** train or plot against old v51 `v51_1h.parquet` et al. once the new fabric exists.

---

## 4. Derived metrics and features

Alignment rule for this section, unless a row says “native to the bar”:

> Join source series S onto bar fabric B by backward as-of on `knowable_at(S) <= B.ts_ms`. If S is a daily UTC aggregate of day D, `event_time = D 23:59:59.999 UTC` and `lag_ms = 1 day` (or equivalently event_time = (D+1) 00:00:00 and lag 0). Never join “same calendar date” onto an intraday bar without that lag.

Rolling windows are in **bars of the fabric they are computed on**, and are causal (`rolling_*` ending at t, no centered windows). Z-scoring uses a trailing window (old `WINDOW_ADAPTIVE = 200`, `WINDOW_FAST = 20`, `WINDOW_REGIME = 200` in `sota_shared_logic_v50.py`). Z-scores are computed **inside each split after purge**, or with an expanding/trailing window that does not peek into a later split.

### 4.1 On-bar physics (old v50 / `sota_shared_logic_v50.py`)

Computed on the dollar-bar fabric today; in the new layer they may also be computed on time bars (same formulas, window in that fabric’s bars). Inputs are the bar’s OHLCV, buy/sell, n_trades, plus as-of funding and OI.

| Family / name | Formula (as implemented) | Inputs | Window | Cadence | Alignment | Leak / quality risk |
|---|---|---|---|---|---|---|
| `norm_deviation` | Z of EMA-spread vol regime | close | 200 | bar | native | Low if rolling. |
| `norm_fd_close` | Fractional difference of close, then rolling z | close | fd window + 200 | bar | native | Old comments say this was fixed to rolling z (no full-history z). Keep that. |
| `norm_vpin` | Volume-synchronized informed-trading proxy | buy/sell vol | short bucket | bar | native | Needs the correct buy/sell convention. |
| `norm_flow_imbalance` | Z of buy−sell volume | buy/sell | 200 | bar | native | — |
| `norm_vol_cluster` | Vol-of-vol (rolling std of a deviation series) | close | 50 then 200 | bar | native | — |
| `norm_funding` | Z of joined funding rate | funding | 200 | bar | as-of funding at `calc_time` | Dead/constant if funding missing (observed on a short BTC window with no November funding). Do not 0-fill. |
| `norm_tick_count` | Z of `n_trades` | n_trades | 200 | bar | native | — |
| `norm_log_volume` | Z of log volume | volume | 200 | bar | native | — |
| `norm_hl_spread` | Rogers-Satchell RV, then z | OHLC | ~6? + 200 | bar | native | Kept; YZ is the upgrade. |
| `hurst_regime` | R/S Hurst, then z | log returns | 200 | bar | native | — |
| **`norm_oi_change`** | `pct_change` of forward-filled OI (capped at 5× rolling median 50), then z | metrics OI | 50 + 200 | bar | as-of OI, **lag required** | **FLAG.** `validate_chimera.py` fired `lookahead_corr` of 0.209 vs `target_return_1` (threshold 0.20) on a 32-day BTC sample (2026-10-20–11-18). Possible causes: (a) OI as-of with 48h tolerance and no publication lag attaches a print that already contains same-bar / same-day price reaction; (b) short-sample noise; (c) forward-fill + `fill_null(0)` on `pct_change`. **Must be retested on full history with the new as-of rule before this feature is kept in the default set.** |
| `norm_return_{1,4,16}` | Lagged returns: `returns.shift(1)` etc. | close | 1/4/16 | bar | native | Correctly lagged. Do not fill 0. |
| `norm_spread_bps` | HL-range spread proxy, then z | OHLC | 200 | bar | native | Superseded conceptually by `norm_cs_spread`. |
| `norm_ma_distance` | Distance to SMA(200), z | close | 200 | bar | native | Warmup: first 200 bars null, not 0. |
| `norm_whale` | Z of volume / n_trades | volume, n_trades | 200 | bar | native | — |
| `norm_efficiency` | Efficiency ratio (net move / path) | close | 10 | bar | native | — |
| `norm_return_kurtosis` | Rolling excess kurtosis | returns | 50 | bar | native | — |
| `norm_bar_duration` | Z of log(Δt between bar closes) | ts_ms | 200 | bar | native | Meaningful on dollar/imbalance bars; almost constant on time bars — **do not emit on time bars** (would be dead). |
| `norm_funding_momentum` | Funding vs trailing mean (after a 2026-05-19 slope fix) | funding | 30 + 200 | bar | same as funding | Dead if funding missing. |
| Hawkes intensity quartet | Tick/buy/sell rate vs EMA; imbalance | n_trades, buy/sell | short EMA | bar | native (this is the *on-bar* Hawkes, distinct from the daily `hbr_*` panel) | — |
| `norm_momentum_accel` | 2nd difference of price / returns | close | fast | bar | native | — |
| `norm_vol_price_corr` | Rolling corr(log volume, \|return\|) | volume, returns | 20 | bar | native | — |
| `norm_vol_ratio` | Short/long realized-vol ratio | returns | 20 / 200 | bar | native | — |
| `norm_flow_persistence` | Rolling corr of flow with its lag | flow imbalance | 20 | bar | native | — |
| `norm_oi_price_divergence` | OI building while price is flat | OI, close | fast | bar | same as OI | Inherit OI lag discipline. |
| `norm_yz_volatility` | Yang-Zhang RV | OHLC, previous close | 20–200 | bar | native | Preferred vol feature. |
| `norm_cs_spread` | Corwin-Schultz | HL | 200 | bar | native | Preferred spread feature. |
| `norm_perm_entropy` | Permutation entropy of returns | returns | 100, m=3 | bar | native | — |
| `norm_kyle_lambda` | Price impact vs signed volume | close, signed usd | 50 | bar | native | — |
| `regime_label` | 0/1/2 from SMA(200) distance | close | 200 | bar | native | Label, not a feature for training unless explicitly listed. |

**Cross-asset (`xd_*`, `make_dataset_legacy.py` phase 2):** `xd_btc_return`, `xd_btc_volatility` (BTC pass-through), `xd_funding_spread`, `xd_cross_return_mean`, `xd_cross_vol_mean`, `xd_ma_distance`, `xd_momentum_rank`. Joined with 10-minute backward tolerance (`MAX_JOIN_STALENESS_MS`). **Dubious / dead with one asset** (observed: all peer-relative `xd_*` had std=0 on BTC-only). Contract: compute only when the peer set for that timestamp has ≥2 (or ≥3 for ranks) assets; otherwise null, not 0. Do not run a “BTC-only universe” and then treat xd as live features.

### 4.2 Daily / slow silver families (old frontier / `feature_registry.yaml`)

Old expected new features: 139 + 1 bar-grain, for a claimed v51 total of 181 (`feature_registry.yaml` 1.6.0). These are daily (or sparse) panels joined onto bars.

| Family (prefix) | What it is | Inputs | Window / cadence | Alignment | Keep? | Notes / leak |
|---|---|---|---|---|---|---|
| `hbr_*` (5) | Daily Hawkes branching ratios | aggTrades | daily | 1d lag | KEEP | Registry path missing date suffix (health check false-miss). |
| `s3_*` (14) | Futures “S3 metrics” / top-trader derived panel | Vision metrics | daily | 1d lag | KEEP | Distinct from raw 5m OI used in `norm_oi_change`. |
| `bs_*` (9) | Basis (perp vs spot) | 1d spot klines + perp close | daily | 1d lag | KEEP | Needs true daily close, not last dollar bar of the day. |
| `funding_*` (10) | Per-asset funding features from settlements | bronze funding | daily / 8h | knowable at calc_time | KEEP | Depends on the funding fetch fix. |
| `premium_*` (5) | Derived from basis | basis panel | daily | 1d lag | KEEP if basis kept | Missing in the 2026-11-20 smoke (not built). |
| `xex_*` (7) | Cross-exchange spreads vs Binance | Coinbase/Bybit/OKX | daily | 1d or 60s | DEFER until fetch is robust | Name-collision history; Bybit crash. |
| `liq_*` (13) | Features on the liquidation *proxy* | approx liq panel | daily | 1d | KEEP, labeled proxy | — |
| `wh_*` (5) | Large-print (“whale”) activity | aggTrades | daily | 1d | KEEP | — |
| `soc_*` (1) | Wikipedia views | Wikimedia | daily | 1d | KEEP | — |
| `dv_*` (3) | Deribit DVOL | Deribit | daily | see §2 | KEEP | BTC/ETH only; other assets may use BTC DVOL as a market feature (document that). |
| `stbl_*` (13) | DeFiLlama stablecoin flow shocks | DeFiLlama | daily | 1d | KEEP | Shock flags can be “frozen” (old validator warnings) — rare events, not necessarily a bug. |
| `etf_*` (13) | BTC/ETH ETF flows + z-scores | Farside | daily | 1d | KEEP | Rolling z must be causal. Old validator warned `etf_btc_etf_total_7d_z` train/test mean drift (leak *or* regime). Retest with trailing z only. |
| `fp_*` (1) | Wide cross-asset funding panel | all assets’ funding | daily | calc_time | KEEP | Needs ≥ peers; 90-day gate in the old wide producer was too rigid for smoke/new listings. |
| `rv_*` (6) | Realized vol / jumps | trades or bars | daily | 1d | KEEP | — |
| `te_*` (6) | Transfer entropy panel | multi-asset returns | rolling, daily | 1d; window must be causal | DEFER / FIX | Old producer raised “0 rows” on BTC-only short history — needs a real peer set. |
| `mv_*` (5) | Days since listed, multi-venue flags | listings catalogue | daily | listing time is event_time, lag 0 | KEEP | Quality depends on venue fetch completeness. |
| `lob_*` (12 daily + optional bar-grain) | LOB *proxy* from trades (not a real LOB) | aggTrades / DIB | daily or bar | daily: 1d; bar-grain: backward as-of, 30 min cap (`attach_bargrain`) | KEEP as proxy | — |
| `bd_*` (11 daily + `bd_bgf_imbalance_l1`) | Book-depth profile from Vision | bookDepth | daily / bar | daily 1d; bar-grain 30 min as-of | KEEP (optional v1) | — |
| `xrel_*` (21, not in registry) | Post-chimera enrichment (`add_xrel_features.py`) | chimera | daily | Old code shifts an extra day on top of `attach_frontier` (+2 total — comment in `make_dataset.py`) | Re-specify with a single lag; do not double-shift | Not in `feature_registry.yaml` (known gap). |

**Dubious / dead (do not carry forward as-is):**

- `long_short_ratio ≡ 1.0` (invented column).
- Peer-relative `xd_*` and `fp_*` / TE features on a 1-asset universe.
- `norm_bar_duration` on fixed time bars.
- `target_voladj_*` as a **feature** (they are labels and they use future vol).
- v51 cadence files used as if they were time bars.
- Silent 0-filled first returns / missing funding.

---

## 5. ML output contract

### 5.1 Gold dataset (`dataset_v1`)

One file (or hive partition) per `(asset, bar_type, schema_version)`, plus a universe panel builder that concatenates assets with an `asset` column.

**Identity / market columns:** `asset`, `bar_type`, `bar_id`, `ts_open_ms`, `ts_ms`, `open`, `high`, `low`, `close`, `vwap`, `volume`, `volume_usd`, `buy_volume`, `sell_volume`, `buy_usd`, `sell_usd`, `n_trades`, `signed_usd`.

**Feature columns:** only names listed in a versioned feature list YAML (replacement for the two current YAMLs: `feature_registry.yaml` for producers, `feature_catalog.yaml` for metadata — keep that split). Each feature has `dtype`, `native_cadence`, `lag_ms`, `warmup_bars`, `family`.

**Metadata:** `is_u10`, `is_u50`, `is_u100`, `asset_dna` (from universe files). `listing_date`, `is_warmup` (true while any rolling window is incomplete).

**Targets (all explicit, all use future close, all null on the last `h` bars):**

| Name | Definition | Overlap / leak notes |
|---|---|---|
| `target_return_h` | `(close[t+h] - close[t]) / close[t]` for `h ∈ {1,4,16,64}` | Overlapping for h&gt;1 (standard). Do not also use these as features. Optional clipped copies (`clip` 15% for h=1, 50% otherwise — old v50) are named `target_return_h_clipped`. Default training target is the **uncapped** series (old v51 `target_return_h_raw`). |
| `target_voladj_h` (optional) | `symlog( return_h / realized_vol_h )` where `realized_vol_h` is the **forward** rolling std (`shift(-h)`). | **Uses future volatility.** Allowed only as a label, never as a feature. Default **off** for training (CLAUDE.md: voladj is a vol shortcut). |
| `target_vol_h` (optional) | Forward realized vol over h bars | Label only. |

`h` is in **bars of this file’s `bar_type`**. A 4h time-bar file’s `target_return_1` is the next 4 hours, not the next dollar bar. (The old cadence bug was exactly the opposite, then patched only for targets.)

### 5.2 Purging, embargo, splits

Frozen calendar splits remain a good idea so growing data does not move boundaries (`config/data_config.yaml`):

| Segment | Old boundary | Contract |
|---|---|---|
| Train | `ts_ms` date &lt; 2023-07-01 | Keep as default, but confirm against the chosen history and bar type |
| Val | 2023-07-01 ≤ d &lt; 2024-05-15 | — |
| OOS | 2024-05-15 ≤ d &lt; 2025-03-15 | — |
| Unseen | d ≥ 2025-03-15 | Paper/live after `live_start` 2025-09-03 if still desired |

**Purge / embargo:** drop `purge_bars` at the **tail of the earlier** segment and the **head of the later** one. `purge_bars` is defined on the **training fabric**. 400 dollar bars (~5 min) ≈ 1.4 days; 400 hourly bars ≈ 17 days. The old helper caps purge at 5% of a segment and warns (`purge_split.py`) — that cap made 1d purge ~10 bars, not 400. **New rule:** purge is `max(horizon_max, rolling_window_max)` bars, or a **minimum calendar embargo** (proposed: 7 days), whichever is larger. No silent 5% shrink without failing the gate.

Normalization fitted on train must not be applied from a window that includes val (this is why purge exists). Prefer trailing/expanding z-scores that never look ahead, so the split leak is structurally impossible.

### 5.3 Manifest (`manifest_v1`)

Written atomically next to each gold file (and a universe-level index):

- `contract_version`, `schema_name`, `producer_git_sha`, `params_hash`
- `asset` / `universe`, `bar_type`, `feature_list_id`
- `ts_ms_min`, `ts_ms_max`, `n_rows`, `n_feature_cols`, `n_null_by_col` (or a hash of the null vector)
- Input files: path, sha256, row count (bronze trades, funding, metrics, each silver panel)
- `lags_applied` (family → lag_ms)
- `split_boundaries`, `purge_bars`, `embargo_days`
- `built_at_ms`, `code_version`

Old v51 manifest already has checksums, universe membership, cadence views, and a `fixes_applied` list (`make_dataset.py`); we keep that spirit and make input lineage mandatory.

---

## 6. Validation gates (must be green before anything is “usable”)

Exit 0 = pass, 1 = warn (usable with explicit ack), 2 = hard fail (do not train, do not show as the default chart source).

| Gate | Rule |
|---|---|
| Schema | Required columns, dtypes, `contract_version`, symbol form. No invented constants (std=0 and unique=1 on a non-indicator column → fail, e.g. LS ratio ≡ 1). |
| Sort / uniqueness | `ts_ms` (or `ts_us` for trades) monotone non-decreasing; primary key unique. |
| Bronze integrity | Trade ids unique; `high/low` N/A; price and qty &gt; 0; `is_buyer_maker` boolean. Daily Vision row counts vs a tolerance for expected presence (T+1 lag allowed for “today”). |
| Bar OHLC | `high >= low`, `high >= max(open,close)`, `low <= min(open,close)`, volumes ≥ 0, `buy+sell ≈ volume` (or `buy_usd+sell_usd ≈ volume_usd`) within a tight relative tolerance. |
| Time-bar reconciliation | For each day and asset: `sum(time_1m.volume) == time_1d.volume` (and same for 5m/15m/1h/4h) within 1e-9 relative or 1e-8 abs. Same for `n_trades` and `volume_usd`. **Dollar / DIB / range bars are not required to sum to clock volume in a fixed window** (they are event-time); instead, `sum(n_trades)` over all dollar bars in a UTC day = that day’s bronze trade count. |
| Raw vs bars | `sum(time_* .n_trades)` and `sum(dollar.n_trades)` over [start, end] equals bronze trade count in the same range (last open dollar/DIB bar may be excluded if incomplete). |
| Gaps | Time bars: missing minutes/hours only where bronze has no trades; gap list is written. Unexpected holes in mid-day liquid BTC → fail. |
| As-of / lag | Positive-control test (already sketched in `asof_join.py` self-test): a series with lag L must not appear on a bar with `ts_ms < event_time + L`. Also: no same-UTC-day daily aggregate on an intraday bar. |
| Lookahead correlation | For every *feature* (not target): Spearman/Pearson vs `target_return_1` on **purged train and on unseen**, with a documented threshold (old: 0.20). Failures block the feature, not the whole dataset, unless a core feature fails. **Re-run `norm_oi_change` on full history.** |
| Cross-cadence features | A daily feature joined onto 1m/1h/1d must be constant across all intra-day bars of the day it becomes knowable, and null before that. |
| Split / purge | No row in two segments; embargo lengths honored; 1d vs dollar purge converted correctly. |
| Universe | Cross-asset features null (not zero) when peer count is insufficient. Health checks are scoped to the requested universe (U10 missing peers should not fail a BTC-only smoke unless `--universe u10` was asked). |
| Idempotence | Second run with the same params does not change sha256 of outputs (except `built_at`). |

---

## 7. Proposed layout, FX hook, and archive plan

### 7.1 New module layout (small)

Proposed root: `crypto/src/pipeline_v2/` (name TBD) with a sibling contract at `/workspace/new_pipeline/` until it is moved into the repo.

```
pipeline_v2/
  contract/          # this document + lag registry + feature list + bar-threshold YAML
  fetch/             # source adapters: vision.py, http.py, farside.py, defillama.py, deribit.py, wiki.py
  bars/              # time.py, dollar.py, tick_volume.py, optional dib.py, runs.py, range.py, adaptive.py
  features/          # physics.py (v50-like), daily/ (one module per family), cross_asset.py
  align/             # knowable as-of join (port of asof_join.py, actually used)
  dataset/           # assemble gold, targets, splits, manifests, loader
  validate/          # gates from §6
  cli.py             # the one orchestrator
```

**One CLI:**

```text
python -m pipeline_v2.cli run \
  --start 2020-01-01 --end 2026-11-18 \
  --assets BTCUSDT ETHUSDT \
  --universe u10 \
  --stages fetch,bars,features,dataset,validate \
  --bar-types time_1m,time_1h,time_1d,dollar
```

`--dry-run`, `--status`, `--force`, and per-stage logs are required. `run_pipeline` / `refresh.py` / `refresh_full.py` / `cli.py` will not be carried over as three competing orchestrators.

### 7.2 FX later (interface only — no implementation now)

```python
class MarketSource(Protocol):
    market: str            # "crypto" | "fx" | ...
    def list_instruments(self) -> list[Instrument]: ...
    def fetch_ticks_or_bars(self, instrument, start, end) -> TradeFrame | BarFrame: ...
    def calendar(self) -> TradingCalendar:   # crypto: 24/7; FX: sessions/weekends
    def default_lags(self) -> dict[str, int]: ...
```

Bronze layout is already `data/bronze/.../{market}/{instrument}/`. Time bars, as-of joins, and validation are market-agnostic. Crypto-only series (funding, OI, DVOL, ETF) simply do not register for FX instruments. The old `framework/adapter.py` already gestures at this; there is **no FX ingest in the repo today** (EURUSDT etc. in `universe_screener.py` are Binance crypto-fiat pairs).

### 7.3 Archive plan (after approval, not in this step)

1. Move `crypto/src/pipeline/` → `crypto/src/_archive/pipeline_legacy_2026_11_20/`.
2. Leave a short `README` there: “reference only; do not import from production.”
3. Existing `crypto/src/_archive/frontier/` and `crypto/src/pipeline/_archived/` travel with it (already dead or duplicated).
4. Do not move `crypto/config/` wholesale (universes, listing dates, dollar-bar sizes are useful starting points); copy-and-trim into `pipeline_v2/contract/`.
5. No Git push in this workstream until the user asks. Downstream `src/strat`, `src/wm`, `ChimeraLoader` stay pointed at the old gold until a cutover PR switches the loader to `dataset/` and we re-emit gold under the new contract.

---

## 8. Open questions (resolved)

Resolved by the Decisions section at the top of this file. The original questions are kept so the choice is auditable. Proposed defaults are in parentheses; the owner choices match those defaults except the embargo, which is `max(horizon, window)` bars with no extra 7-day floor.

1. **Bar types that must exist for charts and ML?** (Default: true time bars 1m→5m/15m/1h/4h/1d **and** dollar bars. DIB/runs/range/adaptive-vol optional, built on request.)
2. **First universe to make rock-solid?** (Default: U10. U50/U100 after the gates pass on U10.)
3. **History start?** (Default: 2020-01-01 UTC, or listing date if later. Fetch is always bounded by `--end`.)
4. **Timestamp unit?** (Default: milliseconds canonical; microseconds retained only on raw trades with `agg_trade_id`.)
5. **Dollar-bar size policy?** (Default: keep the current per-asset fixed sizes as v1, e.g. BTC $2M, but recalibrate causally and version the table. Or switch immediately to trailing median-volume calibration.)
6. **Default training labels?** (Default: uncapped `target_return_{1,4,16,64}` only. Keep voladj as an optional labeled column, not the default.)
7. **Split dates?** (Default: keep the frozen 2023-07-01 / 2024-05-15 / 2025-03-15 cuts, but replace the 400-bar/5% cap with `max(horizon, window)` bars **or** a 7-day calendar embargo.)

---

## Appendix A — Old path map (reference)

| Concern | Primary files |
|---|---|
| Fetch (Vision + API fallback) | `crypto/src/pipeline/fetch_all.py` |
| External ingest | `crypto/src/pipeline/ingest/*` |
| Bar fabric | `crypto/src/pipeline/build_bars.py`, `bars/*.py`, `make_dataset_legacy.generate_dollar_bars` |
| v50 features / targets | `sota_shared_logic_v50.py`, `make_dataset_legacy.py` |
| Frontier silver / registry | `frontier_consolidator.py`, `config/feature_registry.yaml`, `config/feature_publication_lag.yaml` |
| v51 gold + cadence views | `make_dataset.py` (`attach_frontier`, `materialize_cadence`, `recompute_cadence_targets`) |
| Join helper (unused in producers) | `asof_join.py` |
| Splits | `purge_split.py`, `config/data_config.yaml` |
| Loader | `chimera_loader.py` |
| Gates | `validate_chimera.py`, `data_health_check.py`, `pipeline_e2e_test.py`, `pre_train_gate.py` |
| Layout | `layout.py`, `parquet_io.py` |
| Universes / sizes / thresholds | `config/universes/*.yaml`, `config/data_config.yaml`, `config/alt_bar_thresholds.yaml`, `config/asset_launch_dates.json` |

## Appendix B — Empirical notes from the 2026-11-20 BTC smoke (not part of the contract, evidence for the “fix” column)

Short window 2026-10-18–2026-11-18, Vision OK, API 451. Official funding fetch: 0 days (daily path + schema). Metrics LS ratio constant 1.0. AggTrades parquets shuffled. v50 dollar chimera built (18,642 bars). v51 cadence files written via `.last()`. Pre-train gate hard-failed (`norm_oi_change` lookahead 0.209; dead funding features; e2e 8/12 after a second panel pass). Logs under `/workspace/pipeline_eval_logs/`.
