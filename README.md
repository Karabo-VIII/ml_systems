# ml_systems

The live tree is a crypto data pipeline. Everything that used to sit at the repository root — the old crypto system, games, harness, forex, and Claude project config — is a read-only snapshot in [`ml_systems_v1/`](ml_systems_v1/README.md).

```
crypto/src/pipeline/     the pipeline package
crypto/docs/DATA_CONTRACT.md
crypto/config/           U10/U50/U100 universes and listing dates
crypto/data/             pipeline output (gitignored)
crypto/runs/             run output (gitignored)
scripts/pre_commit_guard.py
```

`games/` and `harness/` are only in the snapshot.

## Setup

Python 3.12 or 3.13. Do not install `numpy==1.26` (it does not build on 3.13).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r crypto/src/pipeline/requirements.txt
export PYTHONPATH=crypto/src
```

Install the pre-commit guard (large files over 5 MB, and `.parquet` / `.pkl` / `.pt` / `.npz` and the same family). It does not run the old CDAP audit.

```bash
sh scripts/install_pre_commit.sh
```

## Tests and commands

```bash
PYTHONPATH=crypto/src python -m pytest crypto/src/pipeline/tests -q -m 'not network'
PYTHONPATH=crypto/src python -m pipeline.cli --help
```

Smoke (a few days of BTC from Binance Vision):

```bash
PYTHONPATH=crypto/src python -m pipeline.cli run \
  --start 2026-10-01 --end 2026-10-03 \
  --assets BTCUSDT \
  --stages fetch,bars,features,dataset,validate \
  --bar-types time_1m,time_1h,time_1d,dollar
```

Full U10 rebuild (large download; the builder streams one day at a time):

```bash
PYTHONPATH=crypto/src python -m pipeline.cli run \
  --start 2020-01-01 --end 2026-11-18 \
  --universe u10 \
  --stages fetch,bars,features,dataset,validate \
  --bar-types time_1m,time_5m,time_15m,time_1h,time_4h,time_1d,dollar
```

Output goes to `crypto/data/pipeline/`. Read it with `pipeline.dataset.loader.load_gold`. Details are in [`crypto/src/pipeline/README.md`](crypto/src/pipeline/README.md).

Optional network test: `PIPELINE_V2_NETWORK=1 PYTHONPATH=crypto/src python -m pytest crypto/src/pipeline/tests/test_network_smoke.py -q`.
