# Pipeline invariants

These bind any change to `crypto/src/pipeline`. The full contract is `crypto/docs/DATA_CONTRACT.md`. The snapshot in `ml_systems_v1/` is reference only. Do not import it.

- **UTC.** Stored timestamps are Unix instants interpreted as UTC.
- **`ts_ms` int64** is the canonical clock on bars, features, and gold datasets. Raw trades also keep `ts_us` and `agg_trade_id`. `ts_ms = ts_us // 1000`.
- **Close-labelled bars.** A bar covering `(t_open, t_close]` is stamped `ts_ms = t_close`. Time bars are real OHLCV aggregates (volumes and trade counts sum). They are not the last event bar in a clock bucket.
- **As-of joins.** A non-native series attaches only when `knowable_at <= bar.ts_ms`. `knowable_at = event_time + publication_lag_ms`. A negative lag is rejected. A daily aggregate of day D is knowable at (D+1) 00:00:00 UTC.
- **No invented constants.** Do not fill price, volume, or returns with 0. Do not synthesize a column the source did not ship (the old `long_short_ratio = 1.0` is the example). Warmup stays null.
- **Atomic writes.** Parquet goes to a temp file, is fsynced, then `os.replace`d. Required columns are checked before the replace.
- **One CLI.** `python -m pipeline.cli run` takes `--start`, `--end`, `--assets` or `--universe`, and `--force`.
- **Embargo.** `max(horizon, feature lookback)` in that cadence's bars, and at least 7 calendar days. Feature windows are capped at 90 days of clock time.

The pre-commit guard is `scripts/pre_commit_guard.py`. The old CDAP audit stays inside `ml_systems_v1/` and must not be wired to this tree.
