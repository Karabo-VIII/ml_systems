# Data contract

**Version:** 2.2.0 (`pipeline.constants.CONTRACT_VERSION`; changes in §10).  
**Live code:** the flat package `pipeline/` plus `strategy/`. **Config:** `config/pipeline.yaml` and `config/asset_launch_dates.json`.  
**Snapshot:** `ml_systems_v1/` is a read-only archive. Do not import it.  
**Scope:** what we fetch, which bar fabrics exist, how every series is timestamped and joined, how gold is written, and how a run is proven usable. Training and live execution are out of scope.

---

## 1. Invariants

These bind any change to `pipeline/`. Change policy in config, not as a silent literal in code.

- **UTC.** Stored timestamps are Unix instants interpreted as UTC. FX session flags use `Asia/Tokyo`, `Europe/London`, and `America/New_York`.
- **`ts_ms` int64** is the canonical clock on bars, features, and gold. Raw crypto trades also keep `ts_us` and `agg_trade_id`. `ts_ms = ts_us // 1000`. FX ticks are already in `ts_ms`.
- **Close-labelled bars.** A bar covering `[t_open, t_close)` is stamped `ts_ms = t_close`, the same convention as a Binance kline (open_time = `t_open`, next open_time = `t_close`). A trade stamped exactly on a boundary is the first trade of the bar that closes one width later, so a bar never holds data from after its own stamp, and the first millisecond of a month belongs to the new month's first bar. Event bars (dollar, tick, volume) are stamped at the close of the trade that completes them. Time bars are real OHLCV aggregates (open = first, high = max, low = min, close = last; volumes and trade counts sum). They are not the last event bar in a clock bucket.
- **UTC midnight alignment.** Every canonical time-bar width divides 24 hours, and `close_of` uses the Unix epoch (midnight UTC). 6h closes at 00:00/06:00/12:00/18:00, 8h at 00:00/08:00/16:00, 2h on the even UTC hours. Coarser time bars use the clock window `ts_open_ms = ts_ms - width`, even when some minutes in the bucket are missing.
- **As-of joins.** A non-native series attaches only when `knowable_at <= bar.ts_ms`. `knowable_at = event_time + publication_lag_ms`. A negative lag is rejected. A daily UTC aggregate of day D is knowable at **(D+1) 00:00:00 UTC** (`event_time = D 00:00:00 UTC`, `lag_ms = 86_400_000`).
- **No invented constants.** Do not fill price, volume, or returns with 0. Do not synthesize a column the source did not ship. Warmup stays null. FX has no traded notional and no aggressor: `volume_usd` and the side columns stay null; `volume` and `n_trades` are the tick count.
- **Atomic writes.** Parquet goes to a temp file, is fsynced, then `os.replace`d. A `.sha256` sidecar is written next to every parquet. Required columns are checked before the replace. A half-written partition is never treated as done.
- **Monthly, resumable stages.** Fetch, bars, features, dataset, and validate stream one month (or one hour, for Dukascopy) at a time. Re-running the same command skips fresh partitions. Ctrl+C is safe.

---

## 2. Universes, history, labels, splits

- **Crypto universe.** Default `u10`: BTC, ETH, SOL, BNB, XRP, DOGE, ADA, AVAX, LINK, LTC (`config/pipeline.yaml` `universes.u10`).
- **FX universe.** Default `majors`: EURUSD, GBPUSD, USDJPY, AUDUSD, NZDUSD, USDCAD, USDCHF. Crosses live under `fx.crosses` and are included only when requested.
- **History start.** 2020-01-01 UTC, or the listing date in `config/asset_launch_dates.json` when that date is later. Every run is bounded by `--start` and `--end`. FX uses the same floor.
- **Labels.** Default training labels are uncapped `target_return_{1,4,16,64}`. Clipped copies (`--clipped`) and `target_voladj_*` (`--voladj`; the denominator is future realized volatility) are optional and off unless requested. Vol-adjusted columns are labels only, never features.
- **Splits.** Train `ts` date `< 2023-07-01`, val `< 2024-05-15`, oos `< 2025-03-15`, unseen `>= 2025-03-15`. Crypto and FX share these cuts.
- **Embargo.** `max(horizon, feature lookback)` in that cadence’s bars, and at least 7 calendar days. Feature windows are capped at 90 days of clock time. The 7-day floor and 90-day cap are under `windows` in `config/pipeline.yaml`. Rows inside the embargo are labelled `embargo` and belong to no segment. The embargo is not shrunk when a segment is short.

---

## 3. Bar fabrics (the views this pipeline unlocks)

Default `--bar-types` is the canonical set below. Tick and volume bars are built only when named on the command line. DIB, runs, range, and adaptive-vol are research-only and are not produced.

| Fabric | Markets | What it is | What it is for |
|---|---|---|---|
| Time bars: 1m, 5m, 30m, 1h, 2h, 4h, 6h, 8h, 1d | crypto and FX | True clock OHLCV from 1m (crypto trades or FX ticks) | Charts and any model that should think in clock time. Default chart views: 1h and 1d. |
| Dollar bars | crypto only | Event-time bars that close on the trade that makes cumulative notional reach the threshold; that trade is the last of the bar and the next bar restarts at zero, so every completed bar holds at least its threshold | Microstructure and flow features. Default ML fabric alongside 1h/1d. Not built for FX (no traded notional). |
| Tick bars | crypto, optional | Event-time bars of a fixed trade count | Flow work when requested. |
| Volume bars | crypto, optional | Event-time bars of a fixed traded quantity | Flow work when requested. |

**Dollar-bar size (`dollar_thresholds_v2`, adaptive and causal).** The threshold of UTC day D is `median(notional of the last 30 observed complete days before D) / 288`, so a bar is on average a 5-minute slice of that asset's own recent activity and the fabric carries a comparable number of bars per day in 2020 and in 2026. A static size could not do this: the table below gives about 60 bars a day for BTC in 2020 but several thousand in 2024. Day D's own trades never enter day D's size (no lookahead). A bar that is open at midnight keeps the size it opened with. On an asset's first day there is no history, so the static bootstrap table in `config/pipeline.yaml` under `dollar_thresholds.assets` is used (BTC $2,000,000, ETH $700,000, SOL $200,000, BNB $300,000, XRP $350,000, DOGE $400,000, ADA $100,000, AVAX $80,000, LINK $70,000, LTC $50,000). `threshold_overrides` (tests only) switch to a constant size. Missing days are not zero-filled: the window is the last 30 days that have trades.

Shared bar schema `bars_v1`: `asset`, `bar_type`, `bar_id`, `ts_open_ms`, `ts_ms`, `open`, `high`, `low`, `close`, `vwap`, `volume`, `volume_usd`, `buy_volume`, `sell_volume`, `buy_usd`, `sell_usd`, `n_trades`, `signed_usd`. FX adds quote fields (`bid_close`, `ask_close`, `spread_mean`, `spread_max`) and leaves aggressor and `volume_usd` null.

Time bars do not straddle UTC month boundaries (widths divide 24h; months start at midnight). Dollar bars can straddle months and also days, so they carry state. `silver/bars/_state/dollar_<ASSET>.json` (+ `.parquet` for the open bar's trade fragments) holds the open bar, its running total and size, the next `bar_id`, the 30-day notional history, and `through`, the last month whose dollar partition is written. Resume rules: a dollar month counts as done only when its partition is fresh **and** `state.through >= month`; months before the first undone one keep their files; the rest are rebuilt in order from the saved state (a crash between writing a partition and saving the state is therefore redone, never double counted); if the state is ahead of, or unrelated to, the last good month the whole chain is rebuilt from the start. The fragments are written before the JSON and checked against it by hash, so a torn write is refused. `chain_start` is part of every dollar partition's params, so a different `--start` invalidates the chain. Trades are never fed to the accumulator in a month it is not rebuilding. `bar_id` is the close index for time bars (`ts_ms // width`) and a running counter for dollar bars; it is unique across the partitions of an asset.

---

## 4. Sources and bronze

**Crypto** is Binance Vision (`pipeline.fetch_crypto`). Past months use monthly zips; the current month uses daily files. Daily bronze files are still written. Series: spot aggTrades, spot 1m klines, UM funding (monthly), UM metrics (daily), UM 1d klines (basis). Official funding lives at the monthly Vision prefix; the daily funding prefix is not requested.

**FX** is Dukascopy hourly bi5 (`pipeline.fetch_fx`):

`https://datafeed.dukascopy.com/datafeed/<PAIR>/<YYYY>/<MM>/<DD>/<HH>h_ticks.bi5`

`MM` is 0-indexed. The body is LZMA-alone. Each record is 20 bytes, big-endian: `uint32` ms from the hour start, `uint32` ask, `uint32` bid, `float32` ask volume, `float32` bid volume. Ask/bid integers are divided by `fx.price_divisor` (`100000` for the majors, `1000` when the symbol contains `JPY`). A pip is `0.0001`, or `0.01` for JPY. Dukascopy’s float volumes are not used as traded volume. Mid is `(bid + ask) / 2`. Empty hours are cached as an empty parquet. 503, 429, timeouts, and connection errors retry with backoff (`fx.max_retries`, `fx.min_interval_s`). A finished hour is not downloaded again unless `--force`.

The Friday–Sunday close is the FX calendar. Missing 1-minute stamps that fall on Saturday, or on Friday after / Sunday before 17:00 America/New_York, are not data-gap failures. The bound follows US daylight-saving time. A hole inside the week still counts.

**External** (`pipeline.fetch_external`), as-of joined with the lags in config: Farside ETF flows (BTC, ETH), DefiLlama stable supply, Deribit DVOL, Wikipedia pageviews.

Path patterns, schema names, and producers are in the [artifact registry](#9-artifact-registry).

---

## 5. Features

Alignment, unless a series is native to the bar: backward as-of on `knowable_at(S) <= B.ts_ms`. Rolling windows are in bars of the fabric they are computed on, and are causal. Warmup is null.

**Crypto physics** (same formulas on dollar and time fabrics; `norm_bar_duration` is not emitted on time bars because the clock is fixed): `norm_deviation`, `norm_fd_close`, `norm_vpin`, `norm_flow_imbalance`, `norm_vol_cluster`, `norm_funding`, `norm_tick_count`, `norm_log_volume`, `norm_hl_spread`, `hurst_regime`, `norm_return_{1,4,16}`, `norm_spread_bps`, `norm_ma_distance`, `norm_whale`, `norm_efficiency`, `norm_return_kurtosis`, `norm_funding_momentum`, Hawkes intensity columns, `norm_momentum_accel`, `norm_vol_price_corr`, `norm_vol_ratio`, `norm_flow_persistence`, `norm_oi_price_divergence`, `norm_yz_volatility`, `norm_cs_spread`, `norm_perm_entropy`, `norm_kyle_lambda`. `norm_oi_change` is computed only with `--include-oi`. Audit result (2.2.0): it does not leak. Vision `metrics` stamps the START of a 5-minute window (`create_time`) and the value describes the state at T+5m, so the open-interest lag is 10 minutes (`lags.families.open_interest = 600000`, 5 minutes plus one window of slack); the earlier 5-minute lag was exactly marginal. Every feature is trailing or lagged: perturbing bronze after time T leaves every feature at bars `<= T` bit-identical (`tests/test_audit_v22.py`).

**Daily / slow series** are built per month and as-of joined: funding and metrics daily panels, spot–perp basis, whale/liquidation summaries from trades (`wh_*`, `liq_*`), plus the external series above. Cross-asset columns (`xd_btc_return`, `xd_btc_volatility`, `xd_funding_spread`, `xd_cross_return_mean`, `xd_cross_vol_mean`, `xd_ma_distance`, `xd_momentum_rank`) are a per-month cohort pass; they stay null when peer count is insufficient.

**FX features** (windows in bars of that frame): `log_ret_1`, `ret_{1,3,5,10,20,50}`, `rv_{10,20,50}`, `parkinson_{10,20,50}`, `garman_klass_20`, `atr_pips_{10,20,50}`, `spread_pips`, `rel_spread_bps`, `spread_z_100`, `tick_intensity_z_100`, `efficiency_ratio_{20,50}`, session flags `is_tokyo` / `is_london` / `is_newyork` / `is_ldn_ny_overlap` (a bar is in session when `(t_open, t_close]` overlaps Tokyo 09:00–18:00 Asia/Tokyo, London 08:00–17:00 Europe/London, or New York 08:00–17:00 America/New_York), `weekend_gap_pips` (only on the first bar after a hole that contains a Saturday), and `usd_strength_ret` (mean of signed `ret_1` across USD legs in the run; null unless at least two legs are present). Not ported, because this feed lacks the inputs: triangular residuals, carry, a wider USD index, and Amihud illiquidity.

Dropped on purpose (do not reintroduce): invented `long_short_ratio = 1.0`, token unlocks, bookTicker, xex, te, and vol-adjusted targets used as features.

---

## 6. Gold, manifests, and loading

Stages write monthly silver and gold partitions named `<ASSET>_YYYY-MM.parquet`. `load_gold` concatenates every partition for an asset (including any leftover legacy `<ASSET>_YYYYMMDD.parquet` files). Read gold through `pipeline.dataset.load_gold`.

Each gold parquet has a sibling `<name>.manifest.json` (`schema_name` `dataset_v1`) with contract version, producer `pipeline`, asset, bar type, `params_hash` (equal to the parquet footer), `ts_ms` range, row count, feature null fractions, `null_by_design` (columns that are null for a documented reason: warm-up, funding/OI absent, ETF before launch), `inputs` (families of non-native series attached, with the columns and the lag applied), `lags_applied`, `split_boundaries`, embargo length, `sha256` of the parquet, and `built_at_ms`. Month M's gold params include the sha256 of month M+1's features (targets look up to 64 bars ahead), so gold is rebuilt when the next month appears. Idempotent re-runs must not change the sha256 of outputs except `built_at`.

---

## 7. Validation

A run is not “usable” until these gates are considered. In-pipeline validate runs after dataset; `scripts/validate_run.py` is the independent check on a finished `data/` tree. `python -m pipeline.cli status` (or `mlsys status`) reports completeness without fetching.

| Gate | Rule |
|---|---|
| Time-bar reconciliation | For each asset, `sum(time_1m.volume)` equals each coarser time bar’s volume (and the same for `n_trades` and `volume_usd`) within 1e-9 relative or 1e-8 abs. |
| Raw vs bars | `sum(time_*.n_trades)` and `sum(dollar.n_trades)` over the range equals bronze trade count. The last open dollar bar may be excluded if incomplete. |
| Klines | 1m bars built from trades equal Binance's own 1m klines minute by minute (OHLC 1e-9, volume 1e-8, quote volume 1e-6 relative). Zero-trade filler candles are skipped and must be empty. This is the gate that catches a wrong bucket edge. |
| Dollar | The threshold schedule is recomputed from bronze only; every bar's `volume_usd` is at least the threshold it opened with; counts and notional reconcile to bronze (completed bars plus the open tail). |
| Manifests | Every gold manifest has all required keys, the current contract version, a sha256 equal to the file and its sidecar, and row count, ts range and `params_hash` equal to the parquet. |
| Gaps | Time bars: unexpected mid-day holes in liquid BTC fail. FX weekend holes (see §4) are not failures. Isolated FX quiet minutes warn when the longest hole is short. |
| As-of / lag | The lags are read from `lags.families` in `config/pipeline.yaml` (version `lags_v2`: funding 0, funding_daily and daily panel 1 day, open_interest 10 min, etf and wikipedia 2 days). A series with lag L must not appear on a bar with `ts_ms < event_time + L`. No same-UTC-day daily aggregate on an intraday bar. |
| Lookahead correlation | For every feature (not target): correlation vs `target_return_1` with a 0.20 threshold. Failures block the feature, not the whole dataset, unless a core feature fails. Short windows can fire noisily. |
| Cross-cadence features | A daily feature joined onto an intraday bar is null before it is knowable, and constant across the intra-day bars of the day it becomes knowable. |
| Split / embargo | No row in two segments; embargo lengths honored. |
| Universe | Cross-asset features null (not zero) when peer count is insufficient. |
| Idempotence | A second run with the same params does not change output sha256 (except `built_at`). |

---

## 8. CLI and layout

One CLI: `python -m pipeline.cli` or the console script `mlsys`. Subcommands: `run`, `status`, `validate`, and `inspect`. `--market crypto` (Binance Vision) or `--market fx` (Dukascopy). Stages: `fetch,bars,features,dataset,validate`. Default stages end with `validate`, which also runs the independent checks (`mlsys validate`; `--skip-network` skips venue close cross-checks). Default bar types are the canonical set (FX drops dollar). `--force` rebuilds partitions and resets dollar-bar state. Progress is printed with tqdm; a log is written under `runs/`.

Crypto and FX share bars, align, dataset, and validate. There is no second package.

---

## 9. Artifact registry

This section is the inventory that keeps the live tree from growing silently. **Every tracked file outside `ml_systems_v1/` must have a row in the first table.** Adding, renaming, or deleting a tracked file means updating that row in the same commit. `tests/test_artifacts.py` runs `git ls-files`, ignores `ml_systems_v1/`, and fails if a tracked file is missing from the table or a listed path no longer exists.

The second table lists on-disk data families. Those paths are created at runtime under `--data-root` (default `data/`) or `runs/`. They are gitignored and must not be committed. Each parquet has a sibling `.sha256` sidecar written by `pipeline.io`.

### Tracked files (outside the archive)

<!-- begin:tracked-files -->
| path | purpose |
| --- | --- |
| `.gitignore` | Ignore rules for data, runs, build artifacts, and OS junk |
| `AGENTS.md` | Agent working rules for this tree |
| `CLAUDE.md` | Pipeline invariants that bind changes to `pipeline/` |
| `README.md` | Setup, CLI, and run examples |
| `config/asset_launch_dates.json` | Per-asset listing dates used as the history floor |
| `config/pipeline.yaml` | Universes, dollar thresholds, windows, lags, FX settings, and splits |
| `docs/DATA_CONTRACT.md` | This contract, including the artifact registry |
| `docs/STRATEGY_LAYER.md` | Backtest engine and the first three strategies |
| `pipeline/__init__.py` | Package docstring and `__version__` |
| `pipeline/align.py` | As-of joins on `knowable_at` |
| `pipeline/bars.py` | Time, dollar, tick, and volume bar builders |
| `pipeline/calendar.py` | Day and month iterators for chunked runs |
| `pipeline/chunked.py` | Monthly bar, feature, and dataset writers |
| `pipeline/cli.py` | CLI entry for `run` and `status` |
| `pipeline/config.py` | YAML and JSON config loaders |
| `pipeline/constants.py` | Contract version, cadences, and shared constants |
| `pipeline/dataset.py` | Gold assembly, manifests, and `load_gold` |
| `pipeline/features.py` | Crypto physics, daily, cross-asset, and FX features |
| `pipeline/fetch_crypto.py` | Binance Vision fetch (trades, klines, funding, metrics, perp) |
| `pipeline/fetch_external.py` | Farside, DefiLlama, Deribit DVOL, and Wikipedia pageviews |
| `pipeline/fetch_fx.py` | Dukascopy hourly tick fetch |
| `pipeline/io.py` | Atomic parquet/JSON writes and sha256 sidecars |
| `pipeline/inspect.py` | Text and optional HTML inspection of built views |
| `pipeline/orchestrator.py` | Run, resume, validate, and status orchestration |
| `pipeline/partitions.py` | Skip-fresh checks and partition readers |
| `pipeline/paths.py` | Bronze, silver, gold, and state path helpers |
| `pipeline/progress.py` | tqdm progress and `runs/` logs |
| `pipeline/symbols.py` | Symbol canonicalization |
| `pipeline/validate.py` | In-pipeline validation gates |
| `pipeline/validate_run.py` | Independent PASS/WARN/FAIL checks used by `mlsys validate` and the last stage of `run` |
| `pyproject.toml` | Package metadata, dependencies, and the `mlsys` console script |
| `scripts/install_pre_commit.sh` | Installs the pre-commit size and binary guard |
| `scripts/pre_commit_guard.py` | Blocks large and data files from being committed |
| `scripts/validate_run.py` | Thin wrapper that calls `pipeline.validate_run.main` |
| `strategy/__init__.py` | Strategy package exports |
| `strategy/engine.py` | Backtest engine |
| `strategy/signals.py` | The first three strategy functions |
| `tests/conftest.py` | Shared pytest fixtures |
| `tests/fixtures.py` | Offline fixture builders |
| `tests/test_artifacts.py` | Asserts every tracked file outside the archive is in this registry |
| `tests/test_audit_v22.py` | Crypto-layer audit regressions: bar edges, dollar bars and resume, lags, manifests, splits, validators |
| `tests/test_backtest.py` | Backtest engine tests |
| `tests/test_bars.py` | Bar construction and cadence tests |
| `tests/test_external_and_loader.py` | External ingest and gold loader tests |
| `tests/test_features_align.py` | Feature and as-of join tests |
| `tests/test_fx.py` | Offline FX pipeline tests |
| `tests/test_inspect.py` | Inspect summaries and scoped independent validation |
| `tests/test_fx_smoke.py` | Optional live Dukascopy smoke |
| `tests/test_network_smoke.py` | Optional live Vision smoke |
| `tests/test_parse_and_contract.py` | Parser and contract tests |
| `tests/test_resume.py` | Interrupt/resume and status tests |
| `tests/test_run.py` | Offline crypto run and idempotence tests |
| `tests/test_validate_run.py` | Tests for `scripts/validate_run.py` |
<!-- end:tracked-files -->

### On-disk data artifacts (not tracked)

| Family | Path pattern | Partitioning | Schema | Producer |
|---|---|---|---|---|
| Crypto trades | `bronze/trades/spot/<ASSET>/<YYYY-MM-DD>.parquet` | daily | `trades_v1` | `pipeline.fetch_crypto` |
| Spot 1m klines | `bronze/klines/spot_1m/<ASSET>/<YYYY-MM-DD>.parquet` | daily | `kline_1m_v1` | `pipeline.fetch_crypto` |
| UM funding | `bronze/funding/um/<ASSET>/<YYYY-MM>.parquet` | monthly | `funding_v1` | `pipeline.fetch_crypto` |
| UM metrics | `bronze/metrics/um/<ASSET>/<YYYY-MM-DD>.parquet` | daily | `metrics_v1` | `pipeline.fetch_crypto` |
| UM 1d klines | `bronze/klines/um_1d/<ASSET>/<YYYY-MM-DD>.parquet` | daily | `kline_1d_um_v1` | `pipeline.fetch_crypto` |
| Fetch manifests | `bronze/_manifest/<source>/<ASSET>.json` | per source and asset | JSON | `pipeline.fetch_crypto` |
| FX ticks | `bronze/fx_ticks/<PAIR>/<YYYY-MM-DD>/<HH>.parquet` | hourly | `fx_ticks_v1` | `pipeline.fetch_fx` |
| DefiLlama stables | `bronze/external/defillama/stable_supply.parquet` | single file | `external_defillama_v1` | `pipeline.fetch_external` |
| Deribit DVOL | `bronze/external/deribit/dvol.parquet` | single file | `external_dvol_v1` | `pipeline.fetch_external` |
| Farside ETF flows | `bronze/external/farside/etf_flows.parquet` | single file | `external_etf_v1` | `pipeline.fetch_external` |
| Wikipedia pageviews | `bronze/external/wikipedia/pageviews.parquet` | single file | `external_wiki_v1` | `pipeline.fetch_external` |
| Silver bars | `silver/bars/<bar_type>/<ASSET>_YYYY-MM.parquet` | monthly | `bars_v1` | `pipeline.chunked` |
| Daily trade panel | `silver/bars/_daily/<ASSET>_YYYY-MM.parquet` | monthly | `daily_panel_v1` | `pipeline.chunked` |
| Gap list | `silver/bars/_gaps/<ASSET>_<YYYYMMDD>.json` | stamped at the month-end close | JSON | `pipeline.chunked` |
| Dollar-bar state | `silver/bars/_state/dollar_<ASSET>.parquet` and `.json` | per asset | `dollar_state_v1` | `pipeline.chunked` |
| Silver features | `silver/features/<bar_type>/<ASSET>_YYYY-MM.parquet` | monthly | `features_v1` | `pipeline.chunked` |
| Gold dataset | `gold/<bar_type>/<ASSET>_YYYY-MM.parquet` | monthly | `dataset_v1` | `pipeline.chunked` |
| Gold manifest | `gold/<bar_type>/<ASSET>_YYYY-MM.manifest.json` | sibling of each gold parquet | `dataset_v1` payload | `pipeline.dataset` |
| Run log | `runs/<market>_<start>_<end>_<YYYYMMDDTHHMMSSZ>.log` | one file per run | text | `pipeline.progress` |

`load_partitions` also reads leftover legacy names `<ASSET>_YYYYMMDD.parquet` if they are still on disk. New writes use the monthly `YYYY-MM` name only.

---

## 10. Changelog

**2.2.0** (crypto-layer audit). Every silver and gold file built under 2.1.0 is rejected as stale (`partition_fresh` now compares the file's `contract_version`); bronze is unchanged and reusable.

- Bars are `[t_open, t_close)` stamped at close, not `(t_open, t_close]`. Under 2.1.0 a trade exactly on a minute boundary went into the earlier bar, so about 1% of 1m bars disagreed with Binance klines, and a trade at 00:00:00.000 on the 1st sat in the previous month's last bar (duplicated `ts_ms` across two partitions). Applies to FX ticks too.
- Event bars close on the crossing trade and restart at zero (2.1.0 put the crossing trade in the next bar, and bars as small as half the threshold appeared).
- Dollar size is adaptive and causal (`dollar_thresholds_v2`, see §3). Resume is state-versioned (`through`), torn writes are detected.
- `bar_id` unique across partitions. `bs_basis` on every cadence (was 1d only). Daily-panel (`s3_*`) features no longer depend on `--include-oi`. Feature reads are capped at the frame's own end (was O(months^2)).
- Lags: open interest 5 -> 10 minutes, ETF and Wikipedia 1 -> 2 days; all lags from config, written to the manifest.
- Manifests: `inputs`, `lags_applied`, `split_boundaries`, `null_by_design`, `params_hash` now populated. Gold depends on the next month's features.
- Spot kline `open_time` in microseconds (from 2025-01-01) is normalised to ms.
- Validator: streaming (no full-frame loads), new klines, dollar and manifest gates.
