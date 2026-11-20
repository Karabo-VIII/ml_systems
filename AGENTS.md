# Agents

Follow `CLAUDE.md`. Work on `dev`. Do not touch `main`. Do not import `ml_systems_v1/`.

The live code is `crypto/src/pipeline`. Invariants that must hold: UTC, `ts_ms` int64, bars labelled at the close of `(t_open, t_close]`, as-of joins on `knowable_at <= ts_ms`, no invented constants or zero-filled prices and returns, atomic parquet writes.

Read gold with `pipeline.dataset.loader.load_gold`. Run tests with `PYTHONPATH=crypto/src python -m pytest crypto/src/pipeline/tests -q -m 'not network'`.
