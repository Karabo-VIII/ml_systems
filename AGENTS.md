# Agents

Follow `CLAUDE.md`. Work on `dev`. Do not touch `main`. Do not import or edit `ml_systems_v1/` except to read it.

Install with `pip install -e ".[dev]"`. The live code is the flat package `pipeline/` plus `strategy/` (backtest engine and the first three strategies). Config is `config/pipeline.yaml`. Invariants: UTC, `ts_ms` int64, bars covering `[t_open, t_close)` labelled at the close, as-of joins on `knowable_at <= ts_ms`, no invented constants or zero-filled prices and returns, atomic parquet writes. Crypto and FX share bars, align, dataset, and validate. `--market crypto` uses Binance Vision. `--market fx` uses Dukascopy bi5. FX `volume` is the tick count. FX `volume_usd` and the aggressor side stay null.

Read gold with `load_gold` in `pipeline/dataset.py`. Run tests with `python -m pytest tests -q -m 'not network'` from the repository root. A folder is added only when something is stored in it. `data/` and `runs/` are created at runtime and are not committed.

Every tracked file outside `ml_systems_v1/` must have a row in the artifact registry in `docs/DATA_CONTRACT.md`. Adding, renaming, or deleting a tracked file means updating that row in the same commit. `tests/test_artifacts.py` enforces this.
