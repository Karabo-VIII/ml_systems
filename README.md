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

Python 3.11 or newer (tested on 3.13). Do not install `numpy==1.26` (it does not build on 3.13).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1` and use the same `pip install -e ".[dev]"`. The install provides the `mlsys` command; `python -m pipeline.cli` keeps working from the repository root.

Install the pre-commit guard (files over 5 MB, and `.parquet` / `.pkl` / `.pt` / `.npz` and the same family). It does not run the old CDAP audit.

```bash
sh scripts/install_pre_commit.sh
```

## Tests and commands

```bash
python -m pytest tests -q -m 'not network'
python -m pipeline.cli --help   # or: mlsys --help
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

The default `--bar-types` is the canonical set: 1m, 5m, 30m, 1h, 2h, 4h, 6h, 8h, and 1d time bars, plus dollar bars for crypto (FX uses the same time bars, without dollar). All time bars align to 00:00 UTC. Stages stream month by month so memory stays bounded, skip work already on disk, and are safe to interrupt (Ctrl+C) and re-run with the same command. Progress and a log go to the terminal and to `runs/`.

Output goes to `data/`. Read it with `load_gold` in `pipeline/dataset.py`. The contract is [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md). The backtest engine and the first three strategies are in `strategy/`. How to call them, and the research behind the shortlist, is [`docs/STRATEGY_LAYER.md`](docs/STRATEGY_LAYER.md).

Check completeness without fetching:

```bash
python -m pipeline.cli status --data-root data --start 2020-01-01 --end 2026-11-20 --universe u10 --market crypto
python -m pipeline.cli status --data-root data --start 2020-01-01 --end 2026-11-20 --universe majors --market fx
```

Full-history run on a Windows PC (PowerShell). Crypto assets listed after 2020 start at their launch date; FX starts at 2020-01-01. These are large downloads — especially Dukascopy FX, which is hourly and resumable. Re-run the same command after Ctrl+C:

```powershell
python -m pipeline.cli run --market crypto --start 2020-01-01 --end 2026-11-20 --universe u10 --data-root data
python -m pipeline.cli run --market fx --start 2020-01-01 --end 2026-11-20 --universe majors --data-root data

python -m pipeline.cli status --data-root data --start 2020-01-01 --end 2026-11-20 --universe u10 --market crypto
python -m pipeline.cli status --data-root data --start 2020-01-01 --end 2026-11-20 --universe majors --market fx

python scripts/validate_run.py --data-root data
```

The same commands work in bash if you prefer, without `--data-root data` when you are happy with the default `data/` root.

Independent checks on a built `data/` tree (1m→1d reconciliation, trade counts, gaps, sessions, spreads, dollar-bar notionals, as-of joins, and a few daily closes against Vision / Frankfurter):

```bash
python scripts/validate_run.py --data-root data
```

A recent 7-day window (Monday 9 Nov 2026 to Sunday 15 Nov 2026; US and UK standard time, after the 2026 DST fallbacks):

```bash
python -m pipeline.cli run --market crypto --start 2026-11-09 --end 2026-11-15 --universe u10

python -m pipeline.cli run --market fx --start 2026-11-09 --end 2026-11-15 --universe majors
```

Network tests, both off by default:


```bash
PIPELINE_V2_NETWORK=1 python -m pytest tests/test_network_smoke.py -q
PIPELINE_FX_NETWORK=1 python -m pytest tests/test_fx_smoke.py -q
```
