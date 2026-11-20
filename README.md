# ml_systems

The live tree is one data pipeline for crypto and FX. The previous crypto system, games, harness, forex, and Claude project config are a read-only snapshot in [`ml_systems_v1/`](ml_systems_v1/README.md).

```
pipeline/          flat package: fetch, bars, features, dataset, validate, one CLI
strategy/          backtest engine and the first three strategy functions
config/            pipeline.yaml and asset listing dates
docs/              DATA_CONTRACT.md, STRATEGY_LAYER.md
tests/             offline tests, plus network smokes that are skipped by default
scripts/           pre-commit guard
```

`data/` and `runs/` are created when a command writes them. They are gitignored. There is no `crypto/` tree and no empty skeleton directories.

## Setup

Python 3.12 or 3.13. Do not install `numpy==1.26` (it does not build on 3.13).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

From the repository root, `python -m pipeline.cli` works without `PYTHONPATH`.

Install the pre-commit guard (files over 5 MB, and `.parquet` / `.pkl` / `.pt` / `.npz` and the same family). It does not run the old CDAP audit.

```bash
sh scripts/install_pre_commit.sh
```

## Tests and commands

```bash
python -m pytest tests -q -m 'not network'
python -m pipeline.cli --help
```

Crypto smoke (a few days of BTC from Binance Vision):

```bash
python -m pipeline.cli run \
  --market crypto \
  --start 2026-10-01 --end 2026-10-03 \
  --assets BTCUSDT \
  --stages fetch,bars,features,dataset,validate \
  --bar-types time_1m,time_1h,time_1d,dollar
```

FX smoke (one pair, one day, Dukascopy). The feed throttles; hours already on disk are skipped.

```bash
python -m pipeline.cli run \
  --market fx \
  --start 2024-01-16 --end 2024-01-16 \
  --assets EURUSD \
  --stages fetch,bars,features,dataset,validate \
  --bar-types time_1m,time_1h
```

Full U10 rebuild (large download; the builder streams one day at a time):

```bash
python -m pipeline.cli run \
  --market crypto \
  --start 2020-01-01 --end 2026-11-18 \
  --universe u10 \
  --stages fetch,bars,features,dataset,validate \
  --bar-types time_1m,time_5m,time_15m,time_1h,time_4h,time_1d,dollar
```

Output goes to `data/`. Read it with `load_gold` in `pipeline/dataset.py`. The contract is [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md). The backtest engine and the first three strategies are in `strategy/`. How to call them, and the research behind the shortlist, is [`docs/STRATEGY_LAYER.md`](docs/STRATEGY_LAYER.md).

Independent checks on a built `data/` tree (1m→1d reconciliation, trade counts, gaps, sessions, spreads, dollar-bar notionals, as-of joins, and a few daily closes against Vision / Frankfurter):

```bash
python scripts/validate_run.py --data-root data
```

A recent 7-day window (Monday 9 Nov 2026 to Sunday 15 Nov 2026; US and UK standard time, after the 2026 DST fallbacks):

```bash
python -m pipeline.cli run --market crypto --start 2026-11-09 --end 2026-11-15 --universe u10 --stages fetch,bars,features,dataset,validate --bar-types time_1m,time_5m,time_15m,time_1h,time_4h,time_1d,dollar

python -m pipeline.cli run --market fx --start 2026-11-09 --end 2026-11-15 --universe majors --stages fetch,bars,features,dataset,validate --bar-types time_1m,time_5m,time_15m,time_1h,time_4h,time_1d
```

Network tests, both off by default:


```bash
PIPELINE_V2_NETWORK=1 python -m pytest tests/test_network_smoke.py -q
PIPELINE_FX_NETWORK=1 python -m pytest tests/test_fx_smoke.py -q
```
