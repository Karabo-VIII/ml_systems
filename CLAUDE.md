# Pipeline invariants

These bind any change to `pipeline/`. The full contract is `docs/DATA_CONTRACT.md`. The snapshot in `ml_systems_v1/` is reference only. Do not import it. Do not edit it.

- **UTC.** Stored timestamps are Unix instants interpreted as UTC. FX session flags use `Asia/Tokyo`, `Europe/London`, and `America/New_York` so daylight-saving time is not a fixed offset.
- **`ts_ms` int64** is the canonical clock on bars, features, and gold datasets. Raw crypto trades also keep `ts_us` and `agg_trade_id`. `ts_ms = ts_us // 1000`.
- **Close-labelled bars.** A bar covering `(t_open, t_close]` is stamped `ts_ms = t_close`. Time bars are real OHLCV aggregates (volumes and trade counts sum). They are not the last event bar in a clock bucket. Crypto and FX share this.
- **As-of joins.** A non-native series attaches only when `knowable_at <= bar.ts_ms`. `knowable_at = event_time + publication_lag_ms`. A negative lag is rejected. A daily aggregate of day D is knowable at (D+1) 00:00:00 UTC.
- **No invented constants.** Do not fill price, volume, or returns with 0. Do not synthesize a column the source did not ship (the old `long_short_ratio = 1.0` is the example). Warmup stays null. FX has no traded notional and no aggressor: `volume_usd` and the side columns stay null, and `volume` is the tick count.
- **Atomic writes.** Parquet goes to a temp file, is fsynced, then `os.replace`d. Required columns are checked before the replace. Dukascopy hours are cached one file per hour so a retry does not refetch a finished hour.
- **One CLI.** `python -m pipeline.cli` or `mlsys`. `run` takes `--start`, `--end`, `--assets` or `--universe`, `--market` (`crypto` or `fx`), and `--force`.
- **Embargo.** `max(horizon, feature lookback)` in that cadence's bars, and at least 7 calendar days. Feature windows are capped at 90 days of clock time. Crypto and FX use the same splits.
- **Layout.** A directory exists only when it holds something. Do not add an empty package, an empty `crypto/` tree, or a committed `data/` or `runs/` directory.
- **Artifact registry.** Every tracked file outside `ml_systems_v1/` is listed in `docs/DATA_CONTRACT.md`. Update that list in the same commit as the file change. Install the package with `pip install -e ".[dev]"`.

The pre-commit guard is `scripts/pre_commit_guard.py`. The old CDAP audit stays inside `ml_systems_v1/` and must not be wired to this tree.
