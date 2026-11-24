# ml_systems

The live tree is one data pipeline for crypto and FX. The previous crypto system, games, harness, forex, and Claude project config are a read-only snapshot in [`ml_systems_v1/`](ml_systems_v1/README.md).

```
pipeline/          flat package: fetch, bars, features, dataset, validate, inspect, one CLI
strategy/          backtest engine and the first three strategy functions
config/            pipeline.yaml and asset listing dates
docs/              DATA_CONTRACT.md (schemas, invariants, artifact registry), STRATEGY_LAYER.md
tests/             offline tests, plus network smokes that are skipped by default
scripts/           pre-commit guard; validate_run.py is a thin wrapper for `mlsys validate`
```

`data/` and `runs/` are created when a command writes them. They are gitignored. There is no `crypto/` tree and no empty skeleton directories.

## Setup

Python 3.11 or newer (tested on 3.13). Do not install `numpy==1.26` (it does not build on 3.13).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1` and use the same `pip install -e ".[dev]"`. Optional HTML charts: `pip install -e ".[dev,viz]"`. The install provides the `mlsys` command; `python -m pipeline.cli` keeps working from the repository root.

Install the pre-commit guard (files over 5 MB, and `.parquet` / `.pkl` / `.pt` / `.npz` and the same family). It does not run the old CDAP audit.

```bash
sh scripts/install_pre_commit.sh
```

## Data layer

This is the path to test end to end before any strategy work. One `run` command fetches, builds every view, writes gold, and finishes with in-pipeline gates plus the independent `validate` checks. Default stages are `fetch,bars,features,dataset,validate`. Schemas, lags, and the file registry live in [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md).

### What is fetched

**Crypto** (`--market crypto`), from Binance Vision. Past months use monthly zips; the current month uses daily files. Daily bronze files are always written.

- Spot aggTrades, spot 1m klines
- UM funding (monthly), UM metrics / open interest (daily, often missing in 2020), UM 1d klines (basis)
- Externals, as-of joined with the publication lags in `config/pipeline.yaml` (one day for DefiLlama and Deribit, two for Farside and Wikipedia): Farside ETF flows (BTC/ETH; US spot ETF history starts 2024), DefiLlama stablecoin supply, Deribit DVOL (from 2021), Wikipedia pageviews (u10)

A symbol listed after `--start` begins at its Binance listing date in `config/asset_launch_dates.json` (and never before 2020-01-01). SOL lists on 2020-08-11, AVAX on 2020-09-22.

**FX** (`--market fx`), from Dukascopy hourly tick files. Majors: EURUSD, GBPUSD, USDJPY, AUDUSD, NZDUSD, USDCAD, USDCHF. `volume` / `n_trades` are the tick count. There is no traded notional and no aggressor side.

### How each view is processed

```
bronze  raw trades, ticks, klines, funding, metrics, externals
  -> silver/bars     time bars at each canonical cadence, plus crypto dollar bars
  -> silver/features  physics, daily, cross-asset (crypto) or FX features, as-of joined
  -> gold             features + labels + splits + embargo, one monthly partition per view
```

Stages stream month by month (Dukascopy hour by hour), skip work already on disk, and are safe to interrupt with Ctrl+C and re-run with the same command. A partition is rebuilt when anything it was built from changes (a refetched day, a late funding file, a newly fetched month), so extending `--end` or re-running after a failed fetch rebuilds only what is affected. Progress goes to the terminal and to `runs/`.

### What each view is for

| View | Markets | Role |
|---|---|---|
| Time bars 1m, 5m, 30m, 1h, 2h, 4h, 6h, 8h, 1d | crypto and FX | True clock OHLCV. Charts and clock-time models. Default chart views: 1h and 1d. |
| Dollar bars | crypto only | Event-time bars that close on the trade that reaches a notional set per asset and day from its own trailing 30-day median (see `dollar_thresholds` and the contract §3). Microstructure and flow; default ML fabric alongside 1h/1d. |

Tick, volume, DIB, runs, range, and adaptive-vol bars are not produced. All time bars align to 00:00 UTC.

### Where files land

Under `--data-root` (default `data/`):

- `bronze/trades/spot/<ASSET>/<YYYY-MM-DD>.parquet`, `bronze/klines/...`, `bronze/funding/um/<ASSET>/<YYYY-MM>.parquet`, `bronze/fx_ticks/<PAIR>/<YYYY-MM-DD>/<HH>.parquet`, `bronze/external/...`
- `silver/bars/<bar_type>/<ASSET>_YYYY-MM.parquet` and `silver/features/<bar_type>/...`
- `gold/<bar_type>/<ASSET>_YYYY-MM.parquet` plus a `.manifest.json` and a `.sha256` sidecar
- `runs/<market>_<start>_<end>_<timestamp>.log`

Read gold with `load_gold` in `pipeline/dataset.py`.

### Commands (PowerShell)

From the repo root, venv activated. `mlsys` and `python -m pipeline.cli` are equivalent.

Fetch, build every view, and validate (this is the data-layer test). Crypto assets listed after 2020 start at their listing date. Re-run the same command after Ctrl+C; it skips fresh partitions.

```powershell
mlsys run --market crypto --start 2020-01-01 --end 2020-12-31 --universe u10 --data-root data
python -m pipeline.cli run --market crypto --start 2020-01-01 --end 2020-12-31 --universe u10 --data-root data

mlsys run --market fx --start 2020-01-01 --end 2020-12-31 --universe majors --data-root data
```

`--skip-network` skips the Vision / Frankfurter close cross-checks in the final validate stage (in-pipeline gates still run). A short smoke, and a completeness / standalone validate / inspect:

```powershell
mlsys run --market crypto --start 2026-10-01 --end 2026-10-03 --assets BTCUSDT --data-root data
mlsys run --market fx --start 2024-01-16 --end 2024-01-16 --assets EURUSD --data-root data

mlsys status --data-root data --start 2020-01-01 --end 2020-12-31 --universe u10 --market crypto
mlsys validate --data-root data --market crypto --start 2020-01-01 --end 2020-12-31 --skip-network
python scripts/validate_run.py --data-root data --market crypto --start 2020-01-01 --end 2020-12-31 --skip-network

mlsys inspect --asset BTCUSDT --start 2020-03-11 --end 2020-03-13 --data-root data
mlsys inspect --asset BTCUSDT --start 2020-03-11 --end 2020-03-13 --data-root data --html crash.html
```

To review a finished build, compare it with public APIs and draw every view:

```bash
mlsys crosscheck --universe u10 --start 2020-01-01 --end 2020-12-31 --data-root data/u10
mlsys inspect --report data/u10/report --universe u10 --start 2020-01-01 --end 2020-12-31 --data-root data/u10
```

`run` ends with one `SUMMARY  PASS=...  WARN=...  FAIL=...  EXIT=...` line (exit 0 pass, 1 warn, 2 fail). `status` exits 0 when the range is complete and 1 when it is not. `inspect --html` needs `pip install -e ".[viz]"`.

A recent 7-day window (Monday 9 Nov 2026 to Sunday 15 Nov 2026; US and UK standard time, after the 2026 DST fallbacks):

```powershell
mlsys run --market crypto --start 2026-11-09 --end 2026-11-15 --universe u10 --data-root data
mlsys run --market fx --start 2026-11-09 --end 2026-11-15 --universe majors --data-root data
```

The backtest engine and the first three strategies are in `strategy/`. How to call them, and the research behind the shortlist, is [`docs/STRATEGY_LAYER.md`](docs/STRATEGY_LAYER.md). That is a later step than the data-layer `run` above.

## Tests

```bash
python -m pytest tests -q -m 'not network'
python -m pipeline.cli --help   # or: mlsys --help
```

Network tests, both off by default:

```bash
PIPELINE_V2_NETWORK=1 python -m pytest tests/test_network_smoke.py -q
PIPELINE_FX_NETWORK=1 python -m pytest tests/test_fx_smoke.py -q
```
