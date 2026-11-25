# Data contract

**Version:** 2.6.0 (`pipeline.constants.CONTRACT_VERSION`; changes in §10).  
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
- **As-of joins.** A non-native series attaches only when `knowable_at <= bar.ts_ms`. `knowable_at = event_time + publication_lag_ms`. A negative lag is rejected. A daily UTC aggregate of day D is knowable at **(D+1) 00:00:00 UTC** (`event_time = D 00:00:00 UTC`, `lag_ms = 86_400_000`) at the earliest; when the day's last input row is itself published later, that lag is added (the Vision metrics mean: its last row is stamped D 23:55 and is knowable 10 minutes after its stamp, so the mean is knowable at (D+1) 00:05, `lags.families.metrics_daily = 86_700_000`).
- **No invented constants.** Do not fill price, volume, or returns with 0. Do not synthesize a column the source did not ship. Warmup stays null, and so does a value that is undefined (a flat window's z-score, correlation, Hurst exponent or efficiency ratio). No size comes from a static table: dollar bars calibrate on their chain's first day, and `wh_*` is null on an asset's first day. FX has no traded notional and no aggressor: `volume_usd` and the side columns stay null; `volume` and `n_trades` are the tick count.
- **Finite memory.** Every feature at bar t is a function of the last `max_lookback` bars only. Exponential averages are evaluated over a finite window (`windows.ema`), not recursively, and a feature month is computed on a frame that holds at least `max_lookback` bars before the month. So a bar's value does not depend on which partition holds it or where a run started, and the embargo (which is `max_lookback` long) covers everything a feature can see.
- **Atomic writes.** Parquet goes to a temp file, is fsynced, then `os.replace`d. A `.sha256` sidecar is written next to every parquet. Required columns are checked before the replace. A half-written partition is never treated as done.
- **Monthly, resumable stages.** Fetch, bars, features, dataset, and validate stream one month (or one hour, for Dukascopy) at a time. Re-running the same command skips fresh partitions. Ctrl+C is safe: the fetch manifest is saved after every month, and a bronze file whose sidecar matches is adopted even when the manifest never recorded it.
- **Freshness follows inputs.** A partition is fresh only when its params match, and the params of every silver and gold partition carry a digest of the inputs it read (the `.sha256` sidecars of the bronze days, bar months, daily panels, funding, metrics and perp files, plus the external rows in its window, and for gold the feature months of the asset and its cohort) and the settings that shape its values (dollar and whale sizing for bars; the cadence's scaled `windows` and the `lags` table for features; target horizons, target clock and cross-asset windows for gold). A refetched day, funding published after a month was built, a rebuilt upstream month, or an edited window or lag therefore rebuilds exactly what depends on it, and nothing else.

---

## 2. Universes, history, labels, splits

- **Crypto universe.** Default `u10`: BTC, ETH, SOL, BNB, XRP, DOGE, ADA, AVAX, LINK, LTC (`config/pipeline.yaml` `universes.u10`).
- **FX universe.** Default `majors`: EURUSD, GBPUSD, USDJPY, AUDUSD, NZDUSD, USDCAD, USDCHF. Crosses live under `fx.crosses` and are included only when requested.
- **History start.** 2020-01-01 UTC, or the listing date in `config/asset_launch_dates.json` when that date is later. Every run is bounded by `--start` and `--end`. FX uses the same floor.
- **Labels.** Default training labels are uncapped `target_return_{1,4,16,64}`. On crypto time bars the horizon is clock time: the target of the bar closing at `t` is `close(t + h * width) / close(t) - 1`, where `close(x)` is the last close at or before `x` (the last trade price; a minute with no trades has no bar and leaves the price unchanged), and it is null when `t + h * width` is past the frame's last bar. 2.4.x counted rows, so on an illiquid coin's 1m bars (DOGE 2020: 36% of minutes without a trade) 16% of `target_return_1` spanned more than a minute. On dollar bars the horizon is `h` bars; FX also counts bars, because the weekend close is not a quiet minute and the next achievable exit is the reopen. Clipped copies (`--clipped`) and `target_voladj_*` (`--voladj`; the denominator is the realized volatility of the next `h` bars) are optional and off unless requested. Vol-adjusted columns are labels only, never features.
- **Splits.** Train `ts` date `< 2023-07-01`, val `< 2024-05-15`, oos `< 2025-03-15`, unseen `>= 2025-03-15`. Crypto and FX share these cuts.
- **Embargo.** `max(horizon, feature lookback)` in that cadence’s bars, and at least 7 calendar days, on each side of every split edge. The feature lookback is the *chained* one: every `norm_*` feature is an input window followed by a `z`-bar z-score, so the lookback is `longest input window + 1 + z` (and `2 * ma + 1` for the regime label). Three inputs are themselves a chain of two windows and count both: the EMA deviation followed by its `vol_cluster`-bar std (`ema + vol_cluster - 1`), one diff followed by the Hawkes EMA (`ema + 1`), and the open-interest median followed by the divergence EMAs (`oi_median + ema`). Windows are scaled per cadence so that this chained total is at most 90 days of clock time: `z` takes at most half of the cap, every input window at most the rest, and the two windows of a chain share it (1d: `z` 45, `ema` 23, `vol_cluster` 23, `oi_median` 22, the others 44; 8h: `ema` 86; 6h: `ema` 131). Fine cadences (1m to 1h) are unchanged. The 7-day floor and 90-day cap are under `windows` in `config/pipeline.yaml`. Resulting embargo per side: 1m and 5m 7 days, 30m 25 days, 1h 50 days, 2h to 1d 90 days, dollar 7 days. On 1d this leaves about 139 days of val and 124 of oos (2.3.0 chained two 90-day windows into 180 days and left both empty). The length uses a global width, not a per-month median, so a split on a month boundary is labelled the same way in both partitions: the clock width for time bars, and the design width (one day / `target_bars_per_day`, 5 minutes) for adaptive dollar bars. Rows inside the embargo are labelled `embargo` and belong to no segment. The embargo is not shrunk when a segment is short.

---

## 3. Bar fabrics (the views this pipeline unlocks)

Default `--bar-types` is the canonical set below. Tick, volume, DIB, runs, range, and adaptive-vol bars are not produced; naming one on the command line is an error.

| Fabric | Markets | What it is | What it is for |
|---|---|---|---|
| Time bars: 1m, 5m, 30m, 1h, 2h, 4h, 6h, 8h, 1d | crypto and FX | True clock OHLCV from 1m (crypto trades or FX ticks) | Charts and any model that should think in clock time. Default chart views: 1h and 1d. |
| Dollar bars | crypto only | Event-time bars that close on the trade that makes cumulative notional reach the threshold; that trade is the last of the bar and the next bar restarts at zero, so every completed bar holds at least its threshold | Microstructure and flow features. Default ML fabric alongside 1h/1d. Not built for FX (no traded notional). |

**Dollar-bar size (`dollar_thresholds_v3`, adaptive and causal).** The threshold of UTC day D is `median(notional of the last 30 observed complete days before D) / 288`, so a bar is on average a 5-minute slice of that asset's own recent activity and the fabric carries a comparable number of bars per day in 2020 and in 2026. A static size could not do this (a 2024-era table gave BTC about 60 bars a day in 2020 and DOGE a first bar of 45 hours). Day D's own trades never enter day D's size (no lookahead). A bar that is open at midnight keeps the size it opened with. The first day of a chain (the asset's first traded day in the run) has no history, so it has no size: it is a **calibration day**. Its trades seed the history and close no bar; they are counted with the open tail as trades in no completed bar (`DollarAccumulator.excluded_trades`, the manifest's `dollar_open_tail_trades`), so the reconciliation gates still add up. This holds for any asset, in u10 or not; no size is read from a table. `threshold_overrides` (tests only) switch to a constant size. Missing days are not zero-filled: the window is the last 30 days that have trades.

Shared bar schema `bars_v1`: `asset`, `bar_type`, `bar_id`, `ts_open_ms`, `ts_ms`, `open`, `high`, `low`, `close`, `vwap`, `volume`, `volume_usd`, `buy_volume`, `sell_volume`, `buy_usd`, `sell_usd`, `n_trades`, `signed_usd`. FX adds quote fields (`bid_close`, `ask_close`, `spread_mean`, `spread_max`) and leaves aggressor and `volume_usd` null.

Time bars do not straddle UTC month boundaries (widths divide 24h; months start at midnight). Dollar bars can straddle months and also days, so they carry state. After each month the accumulator is saved twice: a per-month snapshot `silver/bars/_state/dollar_<ASSET>/<YYYY-MM>.json` (+ `.parquet` for the open bar's trade fragments) and the latest state `silver/bars/_state/dollar_<ASSET>.json` (+ `.parquet`). A state holds the open bar, its running total and size, the next `bar_id`, the 30-day notional history, and `through`, the month it was taken after. Resume rules: a dollar month counts as done only when its partition is fresh (same `chain_start`, same bronze digest, and the same previous-month snapshot it continued from, `prev_state`) **and** its snapshot exists, is intact, and names that partition's sha256 (`partition_sha`); months before the first undone one keep their files; the rest are rebuilt in order from the snapshot of the month before. Extending `--end`, or a corrected bronze day in one month, therefore rebuilds only from that month on, and because each month names the state it continued from, a rebuild interrupted halfway through that cascade still rebuilds every later month on the next run. A crash between writing a partition and saving its snapshot is redone, never double counted. A month in which no bar closed still writes an (empty) partition, so it counts as done. The fragments are written before the JSON and checked against it by hash, so a torn write is refused. `chain_start` is part of every dollar partition's params, so a different `--start` invalidates the chain. Trades are never fed to the accumulator in a month it is not rebuilding. `bar_id` is the close index for time bars (`ts_ms // width`) and a running counter for dollar bars; it is unique across the partitions of an asset.

---

## 4. Sources and bronze

**Crypto** is Binance Vision (`pipeline.fetch_crypto`). Past months use monthly zips, streamed to disk and parsed in row chunks so a large month stays well under 1.5 GB of RAM; the current month uses daily files. Daily bronze files are still written. Series: spot aggTrades, spot 1m klines, UM funding (monthly), UM metrics (daily; Vision publishes no monthly metrics), UM 1d klines (basis; monthly zips for past months). Official funding lives at the monthly Vision prefix; the daily funding prefix is not requested.

Fetch bookkeeping (`bronze/_manifest/<source>/<ASSET>.json`, saved after every month): `ok` (file written, sha256 recorded), `empty` (a successfully parsed monthly zip had no rows that day), `absent` (Vision answered 404 for a period that ended more than 7 days ago, including a monthly zip, which Vision ships for every month a symbol traded; a month that ended more recently falls back to the daily files), `missing_not_published` (a recent 404, retried next run), `error` (retried next run). `empty` and `absent` are not fetched again unless `--force`. A bronze file whose `.sha256` sidecar matches is adopted as `ok` even if the manifest does not list it (an interrupted run, or bronze copied from another root). Transient failures (network errors, 429, 5xx) are retried four times with exponential backoff; a 404 is an answer and is not retried.

**FX** is Dukascopy hourly bi5 (`pipeline.fetch_fx`):

`https://datafeed.dukascopy.com/datafeed/<PAIR>/<YYYY>/<MM>/<DD>/<HH>h_ticks.bi5`

`MM` is 0-indexed. The body is LZMA-alone. Each record is 20 bytes, big-endian: `uint32` ms from the hour start (`< 3,600,000`), `uint32` ask, `uint32` bid, `float32` ask volume, `float32` bid volume. Ask/bid integers are divided by `fx.price_divisor` (`100000` for the majors, `1000` when the symbol contains `JPY`). A pip is `0.0001`, or `0.01` for JPY. Dukascopy’s float volumes are not used as traded volume. Mid is `(bid + ask) / 2`. A crossed quote (ask < bid) is kept as shipped and counted by the validator; a non-positive price makes the hour fail (not cached, reported, the range continues). An empty hour (404/204) is cached as an empty parquet only once it ended more than 6 hours ago, so an hour that is simply not published yet is fetched again. Hours wholly inside the weekend close under either daylight-saving regime (Friday 22:00 to Sunday 20:00 UTC) are cached as empty, closed hours without a request (`fx.skip_closed_hours`); the hours at the edges are fetched. Requests are spaced adaptively (`fetch_fx._Pacer`): Dukascopy throttles a fast client with 503s, timeouts and resets, so the spacing doubles on each of those (up to 60 s) and shrinks by 15% after each answered request, never below `fx.min_interval_s` (1 s); each hour has `fx.max_retries` tries, then the range continues and the hour is reported. A finished hour is not downloaded again unless `--force`. Records are decoded vectorised; ticks keep the feed's order (a stable sort), so two ticks in the same millisecond can not swap a bar's open or close. FX bar months and the FX daily panel carry a digest of the month's tick hours, so a refetched hour rebuilds what it feeds.

The Friday–Sunday close is the FX calendar. Missing 1-minute stamps that fall on Saturday, or on Friday after / Sunday before 17:00 America/New_York, are not data-gap failures. The bound follows US daylight-saving time. A hole inside the week still counts.

**External** (`pipeline.fetch_external`), as-of joined with the lags in config: Farside ETF flows (BTC, ETH; from 2024), DefiLlama stable supply, Deribit DVOL (BTC, ETH; from 2021, requested in 500-day windows), Wikipedia pageviews (u10 only: the article titles are mapped for those ten), and for FX the central-bank policy rates of the eight currency areas (BIS `WS_CBPOL`, daily, the rate in force each day; the euro area is the ECB main refinancing rate as BIS reports it; fetched from 45 days before the run's start). Each source is one file, and every fetch merges into it keyed by (asset, day), newest fetch winning, so a run over a short range never erases the rest of the history. Farside is scraped from HTML and is the most fragile source; a failed external fetch is reported and does not stop the run.

Path patterns, schema names, and producers are in the [artifact registry](#9-artifact-registry).

---

## 5. Features

Alignment, unless a series is native to the bar: backward as-of on `knowable_at(S) <= B.ts_ms`. Rolling windows are in bars of the fabric they are computed on, and are causal. Warmup is null. Window lengths are `windows` in `config/pipeline.yaml` (`slow_span` is the slow EMA of `norm_vol_ratio`, `ema` the window every EMA is evaluated over, `oi_median` the open-interest cap), scaled per cadence as in §2. On crypto 1m bars a window counts bars, and a minute without a trade has no bar (§3), so on an illiquid coin a 200-bar window covers more than 200 minutes.

**z-score.** Every `norm_*` column (and `hurst_regime`) is `rolling_z(x, z) = clip((x - mean) / std, -5, 5)`, with the mean and sample std taken over the defined values among the last `z` rows of its input `x`; at least 4/5 of them (`ceil(4z/5)`) must be defined. It is null where `x` is null, when fewer values are defined, and when the window is flat (`std <= 1e-10 * (|mean| + std)`). An occasional undefined input (the efficiency ratio of a flat path) therefore blanks only its own row: requiring every value of the window blanked the next `z` rows too (16% of DOGE 2020 dollar-bar efficiency). There is no additive epsilon: 2.4.x divided by `std + 1e-5`, which shrank fine-cadence z-scores whose std is of that order by up to 6%.

**EMA.** `ewm(x, span)` is `sum_k w_k x[t-k] / sum_k w_k` over `k = 0 .. ema - 1` with `w_k = (1 - a)^k`, `a = 2 / (span + 1)`: a normalised EMA truncated at `ema` bars (240 on fine cadences, where it drops less than 1e-8 of a span-26 EMA's weight). Null until `ema` values and wherever the window holds a null.

**Crypto physics** (same formulas on dollar and time fabrics; `norm_bar_duration`, the z-score of the log seconds since the previous bar, is emitted on dollar bars only). Each is the z-score of the input named:

| Column | Input |
|---|---|
| `norm_deviation` | `(ewm(close, 12) - ewm(close, 26)) / ewm(close, 26)` |
| `norm_vol_cluster` | rolling std of that deviation over `vol_cluster` bars |
| `norm_fd_close` | fractional difference (`d = frac_diff_d`) of log close over the `frac_diff` bars before the bar; null until `frac_diff + z` bars |
| `norm_vpin` | `|buy_volume - sell_volume| / volume` of the bar |
| `norm_flow_imbalance` | `(buy_volume - sell_volume) / volume` |
| `norm_tick_count` | `n_trades` |
| `norm_log_volume` | `ln(volume)` |
| `norm_rs_volatility` | Rogers–Satchell per-bar volatility `sqrt(ln(H/C) ln(H/O) + ln(L/C) ln(L/O))` (was `norm_hl_spread`) |
| `hurst_regime` | rescaled-range Hurst exponent of the `hurst` log returns before the bar; null on a flat window (2.4.x wrote 0.5) |
| `norm_return_{1,4,16}` | the 1-, 4- and 16-bar return ending one bar before this one (`close[t-1] / close[t-1-k] - 1`) |
| `norm_ma_distance` | `close / mean(close, ma) - 1` |
| `norm_whale` | mean trade size, `volume / n_trades` |
| `norm_efficiency` | Kaufman efficiency `|close[t] - close[t-n]| / sum(|close diff|)` over `efficiency` bars; null on a flat path (2.4.x added 1e-12 and wrote 0) |
| `norm_return_kurtosis` | excess kurtosis of `close.pct_change()` over one window of `kurtosis` bars (population moments) |
| `norm_hawkes_intensity` | trades per second since the previous bar over its `ewm(·, hawkes_span)`, minus 1 |
| `norm_hawkes_imbalance` | the same surge for buy volume minus that for sell volume |
| `norm_momentum_accel` | `ret4[t] - ret4[t-4]` |
| `norm_vol_price_corr` | Pearson correlation of `ln(volume)` and `|return|` over `fast` bars |
| `norm_vol_ratio` | `ewm(rs, 10) / ewm(rs, slow_span) - 1` of the Rogers–Satchell volatility; null when the slow leg is 0 |
| `norm_flow_persistence` | Pearson correlation of the flow imbalance with its previous bar over `fast` bars |
| `norm_yz_volatility` | Yang–Zhang over `yz` bars: `sqrt(var(ln(O/C_prev)) + k var(ln(C/O)) + (1 - k) mean(RS))`, sample variances, `k = 0.34 / (1.34 + (n + 1) / (n - 1))` |
| `norm_cs_spread` | Corwin–Schultz spread from two adjacent blocks of `cs_block` bars: `beta = ln(H1/L1)^2 + ln(H2/L2)^2`, `gamma = ln(H12/L12)^2`, `alpha = (sqrt(2 beta) - sqrt(beta)) / (3 - 2 sqrt 2) - sqrt(gamma / (3 - 2 sqrt 2))`, `S = 2 (e^alpha - 1) / (1 + e^alpha)`; a negative S is kept, and a block without range is a defined input (`ln(H/L) = 0`; a flat stretch gives `S = 0`) |
| `norm_perm_entropy` | normalised order-3 permutation entropy of the returns over `perm_entropy` bars, returns rounded to 12 decimals so equal returns tie exactly |
| `norm_kyle_lambda` | Kyle's lambda: OLS slope of log return on signed USD volume (taker buy minus taker sell notional) over the `kyle` bars before the bar |

The two Pearson correlations centre both series on the same window's means. 2.4.x centred each product on per-row trailing means and divided by the full-window standard deviations, which is not a correlation (0.14 correlated with the true one at 1d). 2.4.x also halved the Corwin–Schultz `beta` (the paper sums the two blocks) and returned a correlation for Kyle's lambda, which drops the `sd(return) / sd(flow)` scale that makes it a price impact.

`norm_oi_change` and `norm_oi_price_divergence` are computed only with `--include-oi`. Open interest is first capped at 5× its `oi_median`-bar rolling median; `_oi_chg` is the pct_change of that capped series; `norm_oi_change` is the trailing z-score of `_oi_chg`. `norm_oi_price_divergence` is the trailing z-score of `ewm(|_oi_chg|, 10) / ewm(|close.pct_change()|, 10) - 1` (null when the return leg is 0), i.e. how much absolute OI change outruns absolute return, not a price−OI correlation. Audit result (2.2.0): neither leaks. Vision `metrics` stamps the START of a 5-minute window (`create_time`) and the value describes the state at T+5m, so the open-interest lag is 10 minutes (`lags.families.open_interest = 600000`, 5 minutes plus one window of slack); the earlier 5-minute lag was exactly marginal. Every feature is trailing or lagged: perturbing bronze after time T leaves every feature at bars `<= T` bit-identical (`tests/test_audit_v22.py`).

Funding is carried as `funding_rate` (the last settled rate, lag 0), the daily `funding_day_{mean,sum,last}` and `xd_funding_spread`. There is no z-scored funding column: 2020 funding sat at exactly 0.01% for weeks, so a 200-bar z-score of it was null on 43–84% of rows and, where defined on 1m and dollar bars, its denominator was the old epsilon, not the data.

**Daily / slow series** are built per month and as-of joined: funding and metrics daily panels (`s3_*`: the mean of each numeric Vision metrics column over the UTC day, lag 1 day + 5 minutes), spot–perp basis, whale summaries from trades (`wh_large_{count,usd,buy_usd,sell_usd}`, `wh_max_print_usd`, and `wh_threshold_usd`, the size in force that day), realized volatility of the day's 1m closes (`rv_realized`, bipower `rv_bpv`, jump `rv_jv`, `rv_jump_frac`), the Hawkes branching ratio of the day's per-minute trade counts (`hbr_*`), plus the external series above. Whale size (`daily_thresholds_v3`) for UTC day D is the 0.999-quantile of per-trade notional over the last 30 observed complete days before D, from a log histogram with 64 bins per decade and log-linear interpolation inside the bin (within a fraction of 3.7%); day D's own trades never move day D's size. On an asset's first day there is no size and the `wh_*` counts and sums are null (`wh_max_print_usd` is measured). Cross-asset columns (`xd_btc_return`, `xd_btc_volatility`, `xd_funding_spread`, `xd_cross_return_mean`, `xd_cross_vol_mean`, `xd_ma_distance`, `xd_momentum_rank`) are a per-month cohort pass; they stay null when peer count is insufficient. `xd_momentum_rank` is the cohort percentile of the bar's return in [0, 1] (the share of the other names with a lower return, ties counting half; 1 is the best), so it means the same thing when SOL and AVAX join mid-2020 (2.4.x wrote a raw 1..N rank).

**FX features** (windows in bars of that frame; an FX bar is trading time, five days in seven): `log_ret_1`, `ret_{1,3,5,10,20,50}`, `rv_{10,20,50}` (sample std of `ret_1`; the first bar after a weekend carries the weekend move, which `weekend_gap_pips` isolates), `parkinson_{10,20,50}`, `garman_klass_20`, `atr_pips_{10,20,50}` (mean true range), `spread_pips`, `rel_spread_bps`, `spread_z` and `tick_intensity_z` (z-scores over 100 bars, cut on coarse cadences so no FX lookback spans more than 90 calendar days: 63 bars on 1d; they were `spread_z_100` / `tick_intensity_z_100`), `efficiency_ratio_{20,50}`, session flags `is_tokyo` / `is_london` / `is_newyork` / `is_ldn_ny_overlap` (a bar is in session when `(t_open, t_close]` overlaps Tokyo 09:00–18:00 Asia/Tokyo, London 08:00–17:00 Europe/London, or New York 08:00–17:00 America/New_York), and `weekend_gap_pips` (only on the first bar after a hole that contains a Saturday).

The FX analogues of the crypto cross-asset and slow series:

- **Currency strength** (`base_strength_ret`, `quote_strength_ret`; was `usd_strength_ret`). Each currency's log return against USD at a bar comes from its USD pair (`log_ret_1` of XXXUSD, or minus that of USDXXX; USD's own is 0). With N currencies present, the strength of C is its mean log return against the other N - 1, `(N r_C - sum r) / (N - 1)`; a pair carries its base's and its quote's, so one of them is the USD strength for every major. Null below three currencies. A cross adds no currency.
- **Daily realized variance** (`rv_realized`, `rv_bpv`, `rv_jv`, `rv_jump_frac`, as on crypto): from the UTC day's 1m mids, built with the bars into `silver/bars/_daily/<PAIR>_YYYY-MM.parquet`, joined with the `daily_panel` lag (one day). `rv_n_bars` says how many 1m bars the figure rests on: a UTC day can be partial (the Sunday reopen is two or three hours, and Monday's bars see it).
- **Carry** (`carry_pct`): the policy rate of the base currency's area minus the quote's (percent a year: what holding the pair long earns, the FX counterpart of crypto funding), from the BIS rates, joined with the `policy_rate` lag (one day). BIS ships a missing day as `NaN` (New Zealand and Canada weekends, UK holidays); those rows are dropped, so the last published rate carries. Null when either area has no rate. A step series: constant between rate decisions is real, so the constant gates skip it (an all-null carry is still reported).

Targets on FX count bars, not clock time (§2). FX feature months warm up by bars (at least `fx_lookback + 1` before the month). The 50-bar FX windows are part of the feature names and are not cut: a `max_lookback_days` too short to hold them on a cadence is a configuration error. Not ported: triangular residuals (the feed has the majors only) and Amihud illiquidity (no traded volume).

Dropped on purpose (do not reintroduce; reasons in `pipeline.constants.DROPPED_FAMILIES`): invented `long_short_ratio = 1.0`, token unlocks, bookTicker, xex, te, vol-adjusted targets used as features, and in 2.5.0 `norm_funding` and `norm_funding_momentum` (above), `liq_proxy_*` (spot prints above a static size: spot has no liquidations, and it duplicated `wh_*`), `norm_spread_bps` (range times one plus the flow imbalance, not a spread; r = 0.91 with the RS volatility) and `norm_hawkes_{buy,sell}_intensity` (r = 0.96–0.98 with `norm_hawkes_intensity`; `norm_hawkes_imbalance` keeps the split). Renamed in 2.5.0 (`pipeline.constants.RENAMED_COLUMNS`): `norm_hl_spread` → `norm_rs_volatility`.

---

## 6. Gold, manifests, and loading

Stages write monthly silver and gold partitions named `<ASSET>_YYYY-MM.parquet`. The months of one asset need not have the same columns: a series enters with the month its source begins (on u10 2020: BTC's `s3_*` metrics means with Vision's first metrics day, in September; DOGE's funding and basis columns with its perpetual, in July), and a gold month whose targets and embargo read into such a month carries those columns, null on its own rows. `load_gold` concatenates every partition for an asset (including any leftover legacy `<ASSET>_YYYYMMDD.parquet` files) in time order, null where a month lacks a column; `columns=` returns the same selection, and a column that no partition has raises. Read gold through `pipeline.dataset.load_gold`: a plain multi-file parquet read fails on the differing schemas.

Each gold parquet has a sibling `<name>.manifest.json` (`schema_name` `dataset_v1`) with contract version, producer `pipeline`, asset, bar type, `params_hash` (equal to the parquet footer), `ts_ms` range, row count, feature null fractions, `null_by_design` (columns that are null for a documented reason: warm-up, funding/OI absent, ETF before launch), `inputs` (families of non-native series attached, with the columns and the lag applied), `lags_applied`, `split_boundaries`, embargo length, `cohort` (the assets used as cross-asset peers, or null on a single-asset build), `sha256` of the parquet, and `built_at_ms`. Gold params carry the embargo length, the split dates, the cohort, and a digest of every feature month the partition reads: its own month, the warm-up before it, the months after it that the targets and the embargo reach (so gold is rebuilt when month M+1 appears), and the same months of every cohort member. Cross-asset columns use every cohort member that has features for the month, whether or not that member's gold was stale; an asset whose history does not reach a month (listed later) is simply not in that month's cohort. Idempotent re-runs must not change the sha256 of outputs except `built_at`.

---

## 7. Validation

A run is not “usable” until these gates are considered. In-pipeline validate runs after dataset over the run's `--start`/`--end` only; `mlsys validate` (or `scripts/validate_run.py`) is the independent check on a finished `data/` tree. Both work in bounded memory: one asset and one calendar year at a time (1m, 5m and dollar in the in-pipeline gates; every fabric in the independent checks), with results merged per gate (worst severity kept, labelled with the year it came from). Dollar bars, which carry state across years, and the manifests are checked once over the whole range. `mlsys status` reports completeness without fetching (a day recorded `empty` or `absent` counts as complete) and exits 1 when anything is incomplete.

| Gate | Rule |
|---|---|
| Time-bar reconciliation | For each asset, `sum(time_1m.volume)` equals each coarser time bar’s volume (and the same for `n_trades` and `volume_usd`) within 1e-9 relative or 1e-8 abs. |
| Raw vs bars | `sum(time_*.n_trades)` and `sum(dollar.n_trades)` over the range equals bronze trade count (row counts from parquet footers). For dollar bars the trades in no completed bar are added back: the last open bar and the calibration day (recorded as `calibration_day` in the dollar state). A leftover beyond the calibration day larger than about three typical bars fails (a missing month of the fabric). On a sub-range that starts after the chain, bars that opened before the range are left out and their in-range trades count as leftover. |
| Klines | 1m bars built from trades equal Binance's own 1m klines minute by minute (OHLC 1e-9, volume 1e-8, quote volume 1e-6 relative). Days with klines but no trade file are included, so a missing trade day shows up. Zero-trade filler candles are skipped and must be empty. A traded minute missing from the kline file is a venue kline outage (bars exist only where trades exist, and trades are the source of truth); such minutes and the candles on the two edges of each outage are reported as a warning with the outage start (2020-12-21 13:20-13:47 UTC is one, on every asset). Scattered disagreements fail above 1e-4 of compared minutes. This is the gate that catches a wrong bucket edge. |
| Dollar | The threshold schedule is recomputed from bronze only, from the chain start; no bar opens on the calibration day (it has no size); every bar's `volume_usd` is at least the threshold it opened with; counts and notional reconcile to bronze (completed bars plus the open tail plus the calibration day). |
| Manifests | Every gold manifest has all required keys, the current contract version, a sha256 equal to the file and its sidecar, and row count, ts range and `params_hash` equal to the parquet. |
| Gaps | Crypto: a missing 1m minute is lost data when Binance's own 1m kline shows trades in it, or when its day has no kline file; it is venue downtime (an exchange maintenance window) when the day's klines exist and show no trade. Lost holes longer than 15 minutes fail. Without any klines, unexpected holes in BTC fail. FX weekend holes (see §4, including the bar that closes at the Sunday 17:00 New York reopen) and Christmas Day and New Year's Day (New York date) are not failures. Isolated FX quiet minutes warn when the longest hole is short. |
| Bronze coverage | Every day of an asset's range has a bronze trade file, or Vision's own monthly file shipped no rows that day (`empty`, a warning). Any other missing day fails: a day missing from both trades and bars passes every count-based gate. |
| As-of / lag | The lags are read from `lags.families` in `config/pipeline.yaml` (version `lags_v3`: funding 0, funding_daily and daily panel 1 day, metrics_daily 1 day + 5 min, open_interest 10 min, etf and wikipedia 2 days). The independent check recomputes the expected value of each daily series on 1h gold as the last event with `event_time + L <= ts_ms` (whale panel from bronze on up to three 20-day blocks of each year, funding and metrics panels from bronze, DefiLlama and ETF from their bronze files) and requires equality. No same-UTC-day daily aggregate on an intraday bar. |
| Lookahead correlation | A smoke alarm. For every feature (not target): max(|Pearson|, |Spearman|) vs `target_return_1` with a 0.20 threshold, reported with n and z = |corr|·√n. A core crypto feature fails the run only when it is over the threshold and z ≥ 5 (five standard errors under independence); otherwise it warns, because on a short frame (a coin listed in September, a 1d cadence) a 0.2 correlation is noise. Any other feature (FX included, since 1m FX returns have real short-lag autocorrelation) warns. Nothing is removed from gold. |
| Causality (truncation proof) | Crypto, in-pipeline: for two bars of the last feature month of each asset × cadence × chunk, the features are recomputed on the same warm frame from only what is knowable at the bar's close (bars up to it, daily-panel rows whose lag has passed, funding/metrics/perp read to that day). Every stored column must equal the recompute (1e-9 relative; a column absent from the truncated frame counts as null). Any difference fails. External series are covered by the as-of gate. The cross-sectional columns, which another stage builds from several assets, are proven by `mlsys certify` (below). |
| Cross-cadence features | A daily feature joined onto an intraday bar is null before it is knowable, and constant across the intra-day bars of the day it becomes knowable. |
| Split / embargo | No row in two segments; embargo lengths honored and identical across the month partitions that share a split edge. |
| Universe | Cross-asset features null (not zero) when peer count is insufficient. |
| Constants | A core feature that is entirely null fails unless the frame is warmup: shorter than the lookback, or fewer than 50 rows past `is_warmup` (a coin listed late in the range). A single repeated value over 50+ rows is an invented constant. |
| Dead columns | Over the whole range, a column that is entirely null is reported dead (a warning) unless its source has nothing for that asset in range (ETF flows before 2024, DVOL before 2021, metrics before September 2020, no Wikipedia article, no cross-asset cohort). |
| Idempotence | A second run with the same params does not change output sha256 (except `built_at`). Tested, not re-checked on every run. |
| FX reference cross-check (`mlsys crosscheck --market fx`, and the FX part of `mlsys validate`) | Every pair's 1m mid at the ECB fixing, 14:15 Frankfurt time (CET or CEST), against the ECB euro reference rate of that business day (official, `data-api.ecb.europa.eu`; a non-EUR pair is the cross of two per-EUR rates). Median absolute gap: PASS at most 5 pips, WARN up to 10, FAIL above (a 0.1% price-scale error is about 11 pips). The lag test compares at -60..+60 minutes: a clock or daylight-saving shift shows up as a gap at an offset beyond 15 minutes under half the gap at the fixing, and fails. It is applied to all days and to the CET days and the CEST days separately (at least 3 days each), since a clock error confined to one daylight-saving regime hides in a pooled median. Our mid must come from a bar at most 5 minutes old. An unavailable reference is a WARN; a pair without 1m bars is reported, not skipped. |
| Public-API cross-check (`mlsys crosscheck`, on demand, not a `run` stage) | Gold vs channels the build never read. Binance REST klines (`api.binance.com`) for every time cadence: OHLC equal (1e-9), base and taker-buy volume 1e-8, quote and taker-buy quote 1e-6 relative; our `n_trades` (aggTrades) never above the venue's trade count; a venue-traded bar missing from ours fails; a venue kline outage found on 1m explains (warns) the coarser bars that contain it, and scattered disagreement fails above 1e-4. Dollar bars lie inside the REST 1m high/low envelope of the minutes they span and their notional equals the REST quote volume over them. Funding: every bronze event equals Binance's `fapi/v1/fundingRate` and every gold `funding_rate` equals the last venue event at or before the close. A second venue (OKX, or KuCoin where OKX did not list the coin) on 1d (and 1h for BTC, ETH): median close deviation warns over 0.5%, fails over 2%, and the return correlation must peak at lag 0 (a bar stamped a period off peaks at ±1). Responses are cached under `crosscheck/`; `--refresh` fetches every response live instead and reports whether each cached copy still equals the venue (it never overwrites one). The summary is `crosscheck/report.json`. |
| Venue cross-check | Our 1d closes equal Vision's spot 1d klines (one monthly zip per past month). A day Vision cannot serve is a WARN, not a mismatch; `--skip-network` compares against the last bronze 1m kline instead. |
| Certification (`mlsys certify`, on demand, before a range is promoted) | One command, one PASS/WARN/FAIL line per step. `code`: the git commit of the code and any uncommitted change (WARN). `status`: every stage complete. `reproduce`: the range is rebuilt in place from bronze with `--force` (bars, features, dataset), and every silver and gold partition of it must equal the stored file byte for byte; a content fingerprint (row hashes in order) tells a changed value from a bytes-only difference. The rebuild is refused, and nothing is written, when it would rewrite data built from another range or asset list: stored bars outside the range in its first or last month, a dollar chain that calibrated on another day, or a gold month whose manifest names another cohort. `validate`: the validate stage, with its report kept. `causality`: the cross-sectional columns recomputed for two bars (the middle and the last) of the first month, the last month and every month in which an asset starts, with every input cut at the bar, must equal the stored values bit for bit: crypto gold `xd_*` (the dataset stage, rerun on feature months cut at the bar, with the cohort the stored month's manifest names) and every FX feature column (the FX features stage builds the currency strengths across pairs; bars are cut at the bar, daily-panel and policy-rate rows at the instant their lag has passed). `crosscheck`: the public-API cross-check, `--refresh` to fetch live. `--no-rebuild` and `--skip-network` turn their steps into a WARN. The certificate `certify/<market>_<start>_<end>.json` records the commit, the library versions, every check with the validate and cross-check details, and digests of the content and of the bytes of the range's partitions. Certify with the range and assets the data was built with. |

---

## 8. CLI and layout

One CLI: `python -m pipeline.cli` or the console script `mlsys`. Subcommands: `run`, `status`, `validate`, `inspect`, `crosscheck`, and `certify` (§7). `inspect --report DIR --universe u10 --start ... --end ...` writes an index plus one page per asset that draws every view (closes on one axis, per-day notional per view, bars per day, funding and external series, feature coverage and scale heatmaps) and the latest cross-check results; it needs the `viz` extra. `--market crypto` (Binance Vision) or `--market fx` (Dukascopy). Stages: `fetch,bars,features,dataset,validate`. Default stages end with `validate`, which also runs the independent checks (`mlsys validate`; `--skip-network` skips venue close cross-checks). Default bar types are the canonical set (FX drops dollar). `--force` rebuilds partitions and resets dollar-bar state. Exit codes: 0 pass, 1 warn (or, for `status`, incomplete), 2 fail, including a bad argument or config; 130 after Ctrl+C. Progress is printed with tqdm; a log is written under `runs/` (or `$MLSYS_RUNS_DIR`, which the test suite points at a temp directory).

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
| `pipeline/calendar.py` | Day and month iterators for chunked runs, and whether a range holds an FX weekend |
| `pipeline/chunked.py` | Monthly bar, feature, and dataset writers |
| `pipeline/certify.py` | `mlsys certify`: completeness, an in-place rebuild from bronze compared byte for byte, validation, truncation proofs of the cross-sectional columns, the public-API cross-check, and the certificate |
| `pipeline/cli.py` | CLI entry for `run`, `status`, `validate`, `inspect`, `crosscheck`, and `certify` |
| `pipeline/crosscheck.py` | `mlsys crosscheck`: built views vs public APIs (crypto: Binance REST klines and funding, OKX/KuCoin closes; FX: ECB reference rates) |
| `pipeline/config.py` | YAML and JSON config loaders |
| `pipeline/constants.py` | Contract version, cadences, and shared constants |
| `pipeline/dataset.py` | Gold assembly, manifests, and `load_gold` |
| `pipeline/features.py` | Crypto physics, daily, cross-asset, and FX features |
| `pipeline/fetch_crypto.py` | Binance Vision fetch (trades, klines, funding, metrics, perp) |
| `pipeline/fetch_external.py` | Farside, DefiLlama, Deribit DVOL, Wikipedia pageviews, and BIS policy rates (FX carry) |
| `pipeline/fetch_fx.py` | Dukascopy hourly tick fetch |
| `pipeline/io.py` | Atomic parquet/JSON writes and sha256 sidecars |
| `pipeline/inspect.py` | Text and optional HTML inspection of built views; `--report` writes an every-view QA page per asset |
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
| `tests/test_certify.py` | `mlsys certify` on synthetic crypto and FX roots: byte-identical reproduction, an edited stored value, a cross-asset and a currency-strength value read from the next bar (planted), and the refusals that leave stored files untouched |
| `tests/test_crosscheck.py` | Public-API cross-check against a fake venue (planted defects) and the every-view report |
| `tests/test_contract_v26.py` | Contract 2.6.0 regressions: FX windows and embargo inside 90 calendar days, vectorised and order-stable tick decode, closed-hour skip, adaptive pacing, tick-digested bar freshness, currency strength, daily realized variance and carry with their lags, the ECB reference check against a planted clock shift and price-scale error, correctly rounded daily funding and metrics aggregates, a guard that every sort feeding `unique(keep="last")` is stable, and complete gate and column-check reporting |
| `tests/test_contract_v25.py` | Contract 2.5.0 regressions: corrected features against their textbook formulas (Pearson, Corwin–Schultz, Kyle's lambda, Yang–Zhang, z-score), finite memory / partition independence per cadence, drops and renames, clock-time targets, dollar calibration day, whale size without a static table, metrics lag; z-score null tolerance, Corwin-Schultz on flat blocks, dollar trade count when only other bars were rebuilt, sub-range dollar validation, freshness on edited windows/lags, FX warm-up by bars |
| `tests/test_complete_v24.py` | Contract 2.4.x regressions: chained-lookback cap, fetch adoption/absent/retry, external merge, dollar snapshots, input digests, listings, cohort, FX re-run, validator pieces; 2.4.1: every feature within its declared lookback per cadence, significance-aware lookahead, late-listing warmup, kline outages, causality gate vs a planted leak |
| `tests/test_bars.py` | Bar construction and cadence tests |
| `tests/test_external_and_loader.py` | External ingest and gold loader tests |
| `tests/test_followup_v23.py` | Fetch streaming/empty-day cache, adaptive whale size, global dollar embargo |
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
| BIS policy rates | `bronze/external/bis/policy_rates.parquet` | single file, keyed (area, day) | `external_bis_policy_v1` | `pipeline.fetch_external` |
| Silver bars | `silver/bars/<bar_type>/<ASSET>_YYYY-MM.parquet` | monthly | `bars_v1` | `pipeline.chunked` |
| Daily panel | `silver/bars/_daily/<ASSET>_YYYY-MM.parquet` (crypto: whale, realized variance, Hawkes; FX: realized variance of the mids) | monthly | `daily_panel_v1` | `pipeline.chunked` |
| Gap list | `silver/bars/_gaps/<ASSET>_<YYYYMMDD>.json` | stamped at the month-end close | JSON | `pipeline.chunked` |
| Dollar-bar state | `silver/bars/_state/dollar_<ASSET>.parquet` and `.json` (latest) | per asset | `dollar_state_v1` | `pipeline.chunked` |
| Dollar-bar snapshots | `silver/bars/_state/dollar_<ASSET>/<YYYY-MM>.parquet` and `.json` | one per month | `dollar_state_v1` | `pipeline.chunked` |
| Silver features | `silver/features/<bar_type>/<ASSET>_YYYY-MM.parquet` | monthly | `features_v1` | `pipeline.chunked` |
| Gold dataset | `gold/<bar_type>/<ASSET>_YYYY-MM.parquet` | monthly | `dataset_v1` | `pipeline.chunked` |
| Gold manifest | `gold/<bar_type>/<ASSET>_YYYY-MM.manifest.json` | sibling of each gold parquet | `dataset_v1` payload | `pipeline.dataset` |
| Run log | `runs/<market>_<start>_<end>_<YYYYMMDDTHHMMSSZ>.log` | one file per run | text | `pipeline.progress` |
| Cross-check cache | `crosscheck/binance_rest/<interval>/<ASSET>_YYYY-MM.parquet`, `crosscheck/binance_fapi/funding/`, `crosscheck/{okx,kucoin}/<bar_type>/` | monthly (REST klines), per range otherwise; settled periods only | `crosscheck_v1` | `pipeline.crosscheck` |
| Cross-check report | `crosscheck/report.json` | latest run | JSON | `pipeline.crosscheck` |
| ECB reference cache | `crosscheck/ecb/EXR_<start>_<end>.parquet` | per range, settled periods only | `crosscheck_v1` | `pipeline.crosscheck` |
| Certificate | `certify/<market>_<start>_<end>.json` | one per market and range, latest run | JSON | `pipeline.certify` |

`load_partitions` also reads leftover legacy names `<ASSET>_YYYYMMDD.parquet` if they are still on disk. New writes use the monthly `YYYY-MM` name only.

---

## 10. Changelog

**2.6.0** (the FX layer brought to the crypto layer's standard). FX silver and gold built under 2.5.0 are stale. Crypto content is unchanged except for the last bits of the daily funding and metrics means and of four cross-asset means (see Idempotence and Cross-sectional sums); its partitions are rebuilt because the contract version is part of freshness.

- Windows and embargo: FX features have their own lookback (`features.fx_lookback`), in FX bars and capped at 90 calendar days of trading time like crypto's: the two z-scores (`spread_z`, `tick_intensity_z`, were `_100`) are cut to 63 bars on 1d (100 1d bars spanned about 140 calendar days). The FX embargo is `max(horizon, lookback)` in FX bars with the 7-calendar-day floor (`features.fx_embargo_bars`); 2.5.0 used crypto's windows and 101 bars, about 141 calendar days on 1d.
- Fetch: adaptive pacing against Dukascopy's throttle; the weekend close is cached without requests (about a quarter of all hours); vectorised decode (byte-identical to the record loop); stable tick order. FX bar months and the daily panel digest their tick hours (a refetched hour rebuilds them; 2.5.0 FX bars had no input digest).
- New FX series: currency strength for the base and the quote of every pair (replaces `usd_strength_ret`), the daily realized variance of the 1m mids (`rv_*`), and `carry_pct` from BIS policy rates; both slow series as-of joined with a one-day lag (`lags.families.policy_rate`).
- Validation: the FX external check compares the 1m mid at the ECB fixing (14:15 Frankfurt) with the ECB reference rate on every business day for every pair, with a lag test on all days and on each daylight-saving regime alone (2.5.0 compared an end-of-day mid with the fixing on three days and failed only above 100 pips, so it could not see a clock error). `mlsys crosscheck --market fx` runs the same check on demand, with `--refresh`. The dead-column check knows the FX z-score windows, reports an all-null `carry_pct` (only its constancy is exempt), and excuses an all-null `weekend_gap_pips` only when the range holds no weekend; a 5-bar return on a 5-bar frame is warm-up (it needs 6 rows). A constant `rv_n_bars` (1440 on every full weekday) is excused the same way, in the in-pipeline gate and in the validator. The column check lists all-null and constant columns together and counts past eight (a dead column used to hide every constant one in the frame), and the run log keeps every finding at a gate's worst severity (it kept one per gate, so a second constant column or a later chunk's failure did not appear).
- Idempotence: the first u10 2020 rebuild under 2.6.0 matched 2.5.0 in 3,245 of 3,255 silver and gold partitions bit for bit; the other 10 (BTC 1m and 5m, October to December) differed only in the `s3_*` daily means, by about 1e-16 relative. Polars' group-by can sum the same group along a different path from one run to the next. The daily funding and metrics aggregates are now correctly rounded sums (`math.fsum` in `features._per_day`), which no frame layout or row order can change, not even among rows sharing a timestamp. Against the group-by they move `funding_day_mean`, `funding_day_sum` and the `s3_*` means by at most 1.4e-15 relative, on the days where the group-by sum was not correctly rounded (about one funding day in ten, most metrics days); no other column changes, and nothing downstream derives from them (`xd_funding_spread` uses the native rate). Two consecutive forced rebuilds are bit-identical in all 3,255 partitions.
- Cross-sectional sums: the `xd_*` means and the FX currency basket are left folds over the peers in name order (`features._fold_sum`). polars' `sum_horizontal` rounds differently with the number of columns, so a gold month whose cohort includes a peer listing later in the month (its column null before then; AVAX in September 2020) moved the other assets' `xd_*` values before the listing by an ulp, and the result also depended on the order the frames arrived in. A truncation proof of `xd_*` (rebuild from every asset's data up to the bar) now matches the stored gold bit for bit. `xd_funding_spread`, `xd_cross_return_mean`, `xd_cross_vol_mean` and `xd_ma_distance` move by at most 6e-15 of the column's standard deviation (relative changes are larger only where a spread or a difference of means is itself near zero); the other `xd_*` columns do not change.
- Stable sorts: every sort that feeds a keep-last de-duplication, an as-of join or a first/last value now asks for a stable order (`maintain_order=True`): the as-of join's de-duplication of equal knowable instants, the trade, kline, funding, metrics and external parsers, the readers of funding, metrics, perp, bar, feature and gold months, and the FX tick reader and its validator twin. Polars 1.44 happened to keep tie order already, so no output changes; the guarantee no longer depends on it. A test fails on any unstable sort that feeds `unique(keep="last")`, and it is shown to catch a planted one.
- Review fixes before release: BIS `NaN` days dropped (they would have made `carry_pct` NaN on every Sunday-evening and Monday bar of NZDUSD and USDCAD), an ECB outage or a weekend-only range is a WARN instead of a crash, `rv_n_bars` in FX gold.
- Report: per-day activity is the quote-tick count where there is no notional (FX); FX pages draw `carry_pct` and `rv_realized`.
- Loader: `load_gold(columns=...)`, and the cross-check's gold reader, raised when a requested column was missing from any month (BTC's `s3_*` before mid-2020, for example). Each partition now gives the requested columns it has, and the rest are null, as in the full load. `ts_ms` is always read for the time order and the de-duplication, and returned only when requested. A column that no partition has still raises. Stored data does not change.
- Certification: `mlsys certify` (§7) runs the checks a range needs before it is promoted as one command and writes a certificate. It adds the truncation proofs of the cross-sectional columns (crypto gold `xd_*`, every FX feature column; before, no gate covered FX features) and the byte-for-byte rebuild from bronze. `write_dataset_month` and `write_fx_features_month` take a `sink` that receives the computed frames instead of writing them, and a `cut_ms` that drops every input after the bar; a cut frame is never written. Stored data does not change.

**2.5.0** (feature audit of the u10 2020 build: every column recomputed independently from bronze, gold bars and public APIs). Silver and gold built under 2.4.x are stale; bronze is unchanged and reusable.

- Corrected formulas. `norm_vol_price_corr` and `norm_flow_persistence` are Pearson correlations (2.4.x centred each product on per-row trailing means: 0.14–0.89 correlated with the true value). `norm_cs_spread` sums the two blocks' squared log ranges as Corwin–Schultz do (2.4.x halved beta: 0.31–0.79 correlated) and is the spread `S`, not `alpha`. `norm_kyle_lambda` is the regression slope, not a correlation. `norm_yz_volatility` uses sample variances as Yang–Zhang do.
- z-scores have no additive epsilon (it shrank fine-cadence z-scores by up to 6%) and are null on a flat window. Undefined values are null, not constants: Hurst on a flat window (was 0.5), efficiency on a flat path (was 0), the vol-ratio and OI-divergence legs at 0 (was a 1e-12 denominator).
- Finite memory. EMAs are evaluated over `windows.ema` bars, and each two-window chain (EMA deviation then its std, diff then Hawkes EMA, OI median then EMAs) counts both windows in the lookback and shares the 90-day budget. A feature month is computed on a frame with at least `max_lookback` bars before it (`chunked.feature_warm_start`), not a fixed number of days, so a quiet stretch of dollar bars cannot leave it short. 2.4.x recursive EMAs restarted on each month's warm-up frame: at 1d `norm_vol_cluster` differed by up to 1.7 z (p99) depending on the partition, `norm_vol_ratio` 0.2, `norm_deviation` 0.04–0.1, Hawkes 0.02.
- Dropped: `norm_funding`, `norm_funding_momentum` (16–57% coverage after warm-up; the epsilon was the whole denominator at 1m and on dollar bars), `liq_proxy_*`, `norm_spread_bps`, `norm_hawkes_buy_intensity`, `norm_hawkes_sell_intensity`. Renamed: `norm_hl_spread` → `norm_rs_volatility`. 26 physics columns (was 31).
- `xd_momentum_rank` is a cohort percentile in [0, 1] (was a raw rank whose scale changed when SOL and AVAX listed).
- Targets on crypto time bars use clock time (§2).
- `s3_*` lag is 1 day + 5 minutes: the day's mean includes the window stamped 23:55, which is knowable at 00:05 under the open-interest lag (2.4.x attached it at 00:00: a 5-minute leak at 1/288 weight).
- z-scores use the defined values among their last `z` inputs (at least 4/5 of them), so an undefined input blanks only its own row; Corwin–Schultz is defined on a block without range (`S = 0` on a flat stretch, was null on 9.4% of DOGE 2020 dollar bars).
- Freshness: feature params carry the scaled windows and the lag table, gold params the target horizons, target clock and cross-asset windows, so editing them in config rebuilds what they shape without a contract bump. FX feature months also warm up by bars (at least 101 before the month), not 92 calendar days.
- Validate: the bars stage hands `None` (not 0) to validate when no month reported the dollar fabric's excluded trades, so validate reads the saved state (adding a cadence on a rerun used to fail the trade-count gate); the dollar schedule starts at the chain's recorded calibration day, and bars that opened before a validated sub-range are left out of it.
- No static sizes. Dollar bars calibrate on their chain's first day instead of using a 2024-era table (DOGE 2020's first bar spanned 45 hours); `wh_*` counts are null on an asset's first day instead of using a static whale size. The unused per-asset tick/volume tables and the `liq_proxy_usd` table are removed from config. Whale size is read from a 64-bin-per-decade histogram with in-bin interpolation (was 16 bins, bin midpoint).
- Permutation entropy rounds returns to 12 decimals before ranking, so equal returns tie exactly (rounding noise decided 0.25% of BNB dollar-bar values).
- FX shares `rolling_z`, so `spread_z_100` and `tick_intensity_z_100` lose the epsilon too; FX gold built under 2.4.x is stale.

**2.4.1** (found by the u10 2020 build). Silver and gold built under 2.4.0 are stale; bronze is unchanged and reusable.

- `norm_return_kurtosis` is a true rolling excess kurtosis over one window. 2.4.0 centred each 4th power on a different trailing mean, so it was not a kurtosis and needed 2k-1 bars: at 1d its first value came at bar 131, past the declared 90-bar lookback and so past the embargo. A test now checks that every feature's first value comes within `max_lookback` at every cadence.
- Validate: a causality gate (truncation proof) on every asset × cadence. The lookahead correlation fails only when significant (z ≥ 5); on SOL, AVAX (listed Aug/Sep 2020) and XRP 1d it fired on 45-276 rows and the truncation proof showed the features causal. An all-null slow feature on a late-listed coin is warmup, not dead.
- Klines: a venue kline-file outage (traded minutes absent from Binance's klines, e.g. 2020-12-21 13:20-13:47 UTC on every asset) is a warning; only scattered disagreements count toward the 1e-4 failure rate. 2.4.0 failed SOL and AVAX on the same outage that warned on full-year coins, because their shorter range made the same 25 minutes a larger fraction.
- Tooling (no data change, partitions stay fresh): `mlsys crosscheck` compares gold with public APIs the build never read (Binance REST klines on every cadence, Binance funding history, OKX/KuCoin closes), and `inspect --report` draws every view of every asset for review.
- Round-2 review: dollar snapshots record the partition sha they follow and the next month's params carry the previous snapshot's hash (an interrupted month no longer leaves a stale chain); a monthly-zip 404 marks the month absent only once it ended over a week ago (otherwise the daily files are tried); a day with no kline file counts its missing minutes as lost, not downtime; dead-column exemptions are data-driven per source; the bronze coverage gate; FX holidays narrowed to Christmas Day and New Year's Day.

**2.4.0** (full-history rebuild). Silver and gold built under 2.3.0 are stale; bronze is unchanged and reusable.

- Embargo: the lookback is the chained one (input window + z-score) and is capped at 90 days of clock time. FX features keep fixed windows of up to 100 bars, so the FX embargo is at least 101 bars (and padded for the five-day trading week). 2.3.0 capped each window alone, so 1d chained to 180 days and left val and oos empty (8h: 157 days, 4h: 123). New `windows.slow_span` replaces two hard-coded 50-bar spans so they are capped too.
- Gold: the frame is padded by the embargo span on both sides (2.3.0 padded the later side by the target horizon only, so a month inside the embargo but further than the horizon from the edge was labelled `train`). Event-bar embargo uses the design width, not the on-disk median (which re-read the whole fabric for every month).
- Freshness follows inputs: bars, features, and gold params carry a digest of their inputs (§1), including the trailing bronze that sets the daily whale size and the slow series read through the day after the month. The external digest covers only the rows an asset can join.
- Dollar bars keep one state snapshot per month; extending a run rebuilds only from the first stale month (2.3.0 rebuilt the chain from the start on every catch-up run). A month with no completed bar writes an empty partition.
- Dataset: an asset listed after a month no longer makes that month fail for the whole cohort; cross-asset peers include members whose gold is fresh; the cohort is in the params and the manifest.
- Fetch: retries with backoff on transient errors; the manifest is saved every month; finished files are adopted from their sidecars; old 404s and missing monthly zips are recorded `absent` and not refetched; perp 1d uses monthly zips.
- External sources merge instead of overwrite; Deribit is requested in windows.
- FX: empty hours are cached only once settled; a crossed quote no longer aborts a pair's fetch; a cached month counts its ticks (a repeated FX run used to fail); `usd_strength_ret` is computed over the whole cohort.
- Dollar bootstrap for assets outside the u10 table (u50/u100 used to raise and lose their time bars too).
- Validation runs in bounded memory (one asset and one year at a time) over the run range; as-of lags come from config (the ETF check assumed one day and failed on every 2024+ run); the venue cross-check uses monthly zips and treats an unavailable day as a warning; the dollar tail check catches a missing month; klines include days without a trade file; FX holidays and the Sunday reopen bar are calendar, not holes.
- CLI: `status` exits 1 when incomplete and counts `empty`/`absent` days as complete; bad arguments exit 2; `tick`/`volume` bar types are rejected instead of silently producing nothing; test run logs go to `$MLSYS_RUNS_DIR`.

**2.3.0** (crypto follow-up). Silver and gold built under 2.2.0 are stale; bronze is unchanged and reusable.

- Monthly Vision zips are streamed to disk and parsed in 250,000-row chunks. A legitimately empty day (no rows in a successfully parsed monthly zip) is recorded in the fetch manifest as `empty` and is not refetched.
- Whale size is adaptive and causal (`daily_thresholds_v2`): the 0.999-quantile of per-trade notional over the trailing 30 complete days, with the static table as first-day bootstrap. `wh_threshold_usd` is stored on the daily panel.
- Embargo length for event bars is taken from the global fabric, not a per-month median, so a split on a month boundary is labelled consistently.
- Contract text for `norm_oi_price_divergence` now matches the code: it is `--include-oi` only, and is the z-score of smoothed |OI change| over smoothed |return|, minus one.

**2.2.0** (crypto-layer audit). Every silver and gold file built under 2.1.0 is rejected as stale (`partition_fresh` now compares the file's `contract_version`); bronze is unchanged and reusable.

- Bars are `[t_open, t_close)` stamped at close, not `(t_open, t_close]`. Under 2.1.0 a trade exactly on a minute boundary went into the earlier bar, so about 1% of 1m bars disagreed with Binance klines, and a trade at 00:00:00.000 on the 1st sat in the previous month's last bar (duplicated `ts_ms` across two partitions). Applies to FX ticks too.
- Event bars close on the crossing trade and restart at zero (2.1.0 put the crossing trade in the next bar, and bars as small as half the threshold appeared).
- Dollar size is adaptive and causal (`dollar_thresholds_v2`, see §3). Resume is state-versioned (`through`), torn writes are detected.
- `bar_id` unique across partitions. `bs_basis` on every cadence (was 1d only). Daily-panel (`s3_*`) features no longer depend on `--include-oi`. Feature reads are capped at the frame's own end (was O(months^2)).
- Lags: open interest 5 -> 10 minutes, ETF and Wikipedia 1 -> 2 days; all lags from config, written to the manifest.
- Manifests: `inputs`, `lags_applied`, `split_boundaries`, `null_by_design`, `params_hash` now populated. Gold depends on the next month's features.
- Spot kline `open_time` in microseconds (from 2025-01-01) is normalised to ms.
- Validator: streaming (no full-frame loads), new klines, dollar and manifest gates.
