# Pipeline v2

The data layer that feeds charts and ML. It replaces `crypto/src/pipeline` (now `crypto/archive/pipeline_v1`, reference only). One command builds bronze, real time bars, dollar bars, causal features, a gold dataset, and the validation gates.

The contract is `crypto/docs/DATA_CONTRACT.md`. Owner-changeable numbers live in `contract/`.

## What you get

| Stage | Output |
|---|---|
| fetch | `data/pipeline_v2/bronze/` — Vision aggTrades, 1m klines, monthly funding, metrics |
| bars | `silver/bars/{bar_type}/{ASSET}_{YYYYMMDD}.parquet` |
| features | `silver/features/...` — physics, daily panels, cross-asset |
| dataset | `gold/{bar_type}/{ASSET}_{YYYYMMDD}.parquet` plus a `.manifest.json` |
| validate | exit 0 pass, 1 warn, 2 do not train |

`YYYYMMDD` is the UTC date of the last bar close, not the year in `--start`.

Core bar types: `time_1m`, `time_5m`, `time_15m`, `time_1h`, `time_4h`, `time_1d`, `dollar`. Time bars are real OHLCV aggregates of the 1-minute base (volumes and trade counts sum). Tick and volume bars are optional (`--bar-types`). DIB, runs, range, and adaptive-vol are not built.

## Setup

Python 3.12 or 3.13. Do not install `numpy==1.26` — it does not build on 3.13.

```bash
cd crypto
python3 -m venv .venv
source .venv/bin/activate
pip install -r src/pipeline_v2/requirements.txt
export PYTHONPATH=src
```

`PYTHONPATH=src` is required. The package is `pipeline_v2`.

## Smoke run (a few days of BTC)

This hits `data.binance.vision`. No API key. Funding is the monthly zip. A day that is not published yet is recorded as `missing_not_published` and does not abort the run.

```bash
PYTHONPATH=src python -m pipeline_v2.cli run \
  --start 2026-10-01 --end 2026-10-03 \
  --assets BTCUSDT \
  --stages fetch,bars,features,dataset,validate \
  --bar-types time_1m,time_1h,time_1d,dollar \
  --data-root data/pipeline_v2
```

Expect, under `data/pipeline_v2`:

```
bronze/trades/spot/BTCUSDT/2026-10-01.parquet
bronze/funding/um/BTCUSDT/2026-10.parquet
bronze/metrics/um/BTCUSDT/2026-10-01.parquet
bronze/_manifest/{trades,funding,metrics,klines_1m,perp_1d}/BTCUSDT.json
silver/bars/time_1m/BTCUSDT_20261003.parquet
silver/bars/dollar/BTCUSDT_20261003.parquet
gold/time_1h/BTCUSDT_20261003.parquet
gold/time_1h/BTCUSDT_20261003.manifest.json
```

The stamp can be `20261004` when the last 1-minute bar closes at the next midnight. Exit 0 means the gates passed. Exit 2 means do not chart or train on that file.

`--dry-run` prints the plan and writes nothing. `python -m pipeline_v2.cli status --data-root data/pipeline_v2` lists gold files.

## Full U10 rebuild

Disk: spot aggTrades from 2020 for ten assets is hundreds of gigabytes. The builder streams one day at a time. It does not hold the raw history in memory. Give the machine the disk, then:

```bash
PYTHONPATH=src python -m pipeline_v2.cli run \
  --start 2020-01-01 --end 2026-11-18 \
  --universe u10 \
  --stages fetch,bars,features,dataset,validate \
  --bar-types time_1m,time_5m,time_15m,time_1h,time_4h,time_1d,dollar \
  --data-root data/pipeline_v2
```

History starts at `max(2020-01-01, listing date, --start)`. Re-running without `--force` skips a bronze file whose sha256 still matches the manifest. That check is per file, not per directory. `--force` rewrites.

U10 is BTC, ETH, SOL, BNB, XRP, DOGE, ADA, AVAX, LINK, LTC. Dollar thresholds are the fixed table in `contract/dollar_thresholds.yaml` (BTC $2,000,000, and so on). Recalibrating them on a trailing volume window is a TODO. Do not edit a threshold in code.

## Labels, splits, features

Default labels are uncapped `target_return_{1,4,16,64}`. The last `h` bars are null. `--voladj` adds `target_voladj_*`, which divide by future volatility. Those columns are labels only. `--clipped` adds the old 15% / 50% copies.

Splits stay frozen: train before 2023-07-01, val before 2024-05-15, oos before 2025-03-15, unseen from 2025-03-15. The embargo is `max(horizon, window)` bars (64 and 1200 with the default windows), taken from the tail of the earlier segment and the head of the later one. It is not capped at 5%. On daily bars, 1200 bars is a long embargo. That is the rule. Change the windows in `contract/windows.yaml` if you want a shorter one. The gate fails if a segment is too short to honour it, rather than shrinking it.

`norm_oi_change` is off. `--include-oi` turns it and `norm_oi_price_divergence` on, with a 5-minute publication lag and no zero-fill. It still needs a full-history lookahead retest before it is a default feature.

`norm_bar_duration` is computed on dollar, tick, and volume bars only.

Cross-asset `xd_*` columns are null when fewer than two assets are knowable at that bar, and the rank is null below three. They are not filled with 0.

## Tests

```bash
cd crypto/src
PYTHONPATH=src python -m pytest pipeline_v2/tests -q
```

The suite is offline. The Vision smoke is skipped unless you opt in:

```bash
PIPELINE_V2_NETWORK=1 PYTHONPATH=src python -m pytest pipeline_v2/tests/test_network_smoke.py -q
```

## Not in v2

FX (the `MarketSource` protocol is the slot; there is no adapter). ETF, Wikipedia, DeFiLlama, and Deribit fetches. Cross-exchange spreads (an empty venue used to crash the job; the defensive parser exists and is not on the default path). Transfer entropy. Book depth and the LOB proxy. Multi-venue listings. Token unlocks. DIB, runs, range, and adaptive-vol bars.

The strat and world-model layers still import the old package name `pipeline`. See the pull request for how to point them at the archive while they still read old gold, and at `pipeline_v2` once this gold is rebuilt.
