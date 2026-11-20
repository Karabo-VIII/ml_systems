"""Monthly, skip-fresh bar / feature / dataset writers. Memory stays one chunk at a time."""

from __future__ import annotations

from datetime import date, datetime, timezone, timedelta
from pathlib import Path

import polars as pl

from pipeline.bars import DollarAccumulator, bars_to_time, build_fx_ladder, gaps_ms
from pipeline.bars import merge_partial_time, trades_to_time
from pipeline.calendar import clip_month, iter_days, iter_months, month_first, month_last
from pipeline.config import (
    Windows,
    dollar_threshold,
    embargo_bars_for,
    liq_proxy_usd,
    load_embargo_policy,
    load_listing_dates,
    load_windows,
    scale_windows,
    whale_large_usd,
)
from pipeline.constants import DAY_MS, TIME_WIDTH_MS
from pipeline.dataset import (
    add_targets,
    add_universe_columns,
    apply_embargo,
    assign_split,
    manifest_payload,
    utc_now_ms,
)
from pipeline.features import add_cross_asset, rows_to_frame, summarize_day
from pipeline.fetch_crypto import iter_days as _iter_days  # noqa: F401
from pipeline.fetch_fx import read_fx_ticks
from pipeline.io import atomic_write_json, atomic_write_parquet, file_sha256, params_hash
from pipeline.partitions import partition_fresh, partition_params, sidecar_ok
from pipeline.paths import (
    bars_month_path,
    dollar_state_meta_path,
    dollar_state_path,
    features_month_path,
    gap_path,
    gold_manifest_path,
    gold_month_path,
    trades_path,
)


def _day_ms(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)


def month_bounds_ms(first: date, last: date) -> tuple[int, int]:
    """Bars whose close is in (first 00:00, last+1 00:00]."""
    return _day_ms(first), _day_ms(last + timedelta(days=1))


def _in_month(df: pl.DataFrame, first: date, last: date) -> pl.DataFrame:
    lo, hi = month_bounds_ms(first, last)
    return df.filter((pl.col("ts_ms") > lo) & (pl.col("ts_ms") <= hi))


def write_frame(df: pl.DataFrame, path: Path, schema_name: str, params: dict) -> str:
    atomic_write_parquet(
        df, path, schema_name=schema_name, params=params,
        sort_cols=["ts_ms"] if "ts_ms" in df.columns else None,
        required_cols=[c for c in ("ts_ms", "asset", "bar_type") if c in (df.columns if df is not None else [])] or None,
    )
    return file_sha256(path)


def _daily_path(root: Path, asset: str, year: int, month: int) -> Path:
    return root / "silver" / "bars" / "_daily" / f"{asset}_{year:04d}-{month:02d}.parquet"


def _tail_trades(acc: DollarAccumulator) -> int:
    if not acc._partials:
        return 0
    return int(pl.concat(acc._partials, how="diagonal_relaxed")["n_trades"].sum() or 0)


def count_bronze_trades(root: Path, asset: str, first: date, last: date) -> int:
    n = 0
    for day in iter_days(first, last):
        path = trades_path(root, asset, day)
        if path.exists():
            n += pl.read_parquet(path, columns=["ts_ms"]).height
    return n


def save_dollar_state(root: Path, asset: str, acc: DollarAccumulator) -> None:
    meta = {
        "asset": asset,
        "threshold": acc.threshold,
        "cum": acc.cum,
        "n_partials": len(acc._partials),
        "excluded": _tail_trades(acc),
    }
    atomic_write_json(dollar_state_meta_path(root, asset), meta)
    if acc._partials:
        tail = pl.concat(acc._partials, how="diagonal_relaxed")
        atomic_write_parquet(
            tail, dollar_state_path(root, asset),
            schema_name="dollar_state_v1",
            params={"asset": asset, "threshold": acc.threshold, "cum": acc.cum},
        )
    else:
        path = dollar_state_path(root, asset)
        if path.exists():
            path.unlink()
        sidecar = path.with_suffix(path.suffix + ".sha256")
        if sidecar.exists():
            sidecar.unlink()


def load_dollar_state(root: Path, asset: str, threshold: float) -> DollarAccumulator:
    acc = DollarAccumulator(threshold)
    meta_path = dollar_state_meta_path(root, asset)
    path = dollar_state_path(root, asset)
    if not meta_path.exists():
        return acc
    import json
    meta = json.loads(meta_path.read_text())
    if abs(float(meta.get("threshold", threshold)) - threshold) > 1e-9:
        return acc
    acc.cum = float(meta.get("cum", 0.0))
    if path.exists() and sidecar_ok(path):
        acc._partials = [pl.read_parquet(path)]
    return acc


def bars_params(asset: str, bar_type: str, first: date, last: date, extra: dict | None = None) -> dict:
    return partition_params(asset=asset, bar_type=bar_type, start=first, end=last, extra=extra)


def build_crypto_month(cfg, asset: str, first: date, last: date, acc: DollarAccumulator) -> dict:
    """Stream one month of bronze trades into 1m / coarser / dollar bars. Bounded memory."""
    partials = []
    daily_rows = []
    n_bronze = 0
    for day in iter_days(first, last):
        path = trades_path(cfg.data_root, asset, day)
        if not path.exists():
            continue
        trades = pl.read_parquet(path)
        sort_cols = ["ts_us", "agg_trade_id"] if "ts_us" in trades.columns else ["ts_ms"]
        trades = trades.sort(sort_cols)
        n_bronze += trades.height
        one_m = trades_to_time(trades, TIME_WIDTH_MS["time_1m"], asset, "time_1m")
        partials.append(one_m)
        acc.add(trades)
        daily_rows.append(summarize_day(
            trades, one_m, asset=asset, day=day,
            whale_usd=whale_large_usd(asset), liq_usd=liq_proxy_usd(asset),
        ))
        del trades
    bars_1m = merge_partial_time(partials, asset, "time_1m", TIME_WIDTH_MS["time_1m"])
    del partials
    ladder = {"time_1m": bars_1m}
    for name in cfg.bar_types:
        if name in TIME_WIDTH_MS and name != "time_1m":
            ladder[name] = bars_to_time(bars_1m, TIME_WIDTH_MS[name], asset, name)
    dollar, excluded = acc.emit_completed(asset)
    out_bars = {k: v for k, v in ladder.items() if k in cfg.bar_types}
    if "dollar" in cfg.bar_types:
        out_bars["dollar"] = dollar
    return {
        "bars": out_bars,
        "ladder": ladder,
        "daily": rows_to_frame(daily_rows),
        "n_bronze": n_bronze,
        "dollar_excluded": excluded,
        "gaps": gaps_ms(bars_1m),
    }


def write_bars_month(cfg, asset: str, year: int, month: int, acc: DollarAccumulator, session=None) -> dict:
    first, last = clip_month(year, month, *(_asset_range(cfg, asset)))
    extra = {"threshold": (cfg.threshold_overrides or {}).get(asset) or dollar_threshold(asset)} if "dollar" in cfg.bar_types else {}
    needed = []
    for bt in cfg.bar_types:
        path = bars_month_path(cfg.data_root, bt, asset, year, month)
        params = bars_params(asset, bt, first, last, extra if bt == "dollar" else None)
        if partition_fresh(path, params, force=cfg.force):
            continue
        needed.append(bt)
    if not needed:
        # Still keep dollar accumulator in sync when skipping a fresh month.
        if "dollar" in cfg.bar_types:
            acc_loaded = load_dollar_state(cfg.data_root, asset, acc.threshold)
            acc.cum = acc_loaded.cum
            acc._partials = acc_loaded._partials
        return {
            "cached": True,
            "n_bronze": count_bronze_trades(cfg.data_root, asset, first, last),
            "dollar_excluded": _tail_trades(acc),
            "gaps": [],
        }
    packed = build_crypto_month(cfg, asset, first, last, acc)
    for bt, df in packed["bars"].items():
        params = bars_params(asset, bt, first, last, extra if bt == "dollar" else None)
        path = bars_month_path(cfg.data_root, bt, asset, year, month)
        if df.is_empty():
            continue
        write_frame(df, path, "bars_v1", params)
    daily = packed["daily"]
    if daily is not None and not daily.is_empty():
        dpath = _daily_path(cfg.data_root, asset, year, month)
        write_frame(daily, dpath, "daily_panel_v1", bars_params(asset, "daily", first, last))
    if packed["gaps"]:
        # Stamp with month end close.
        atomic_write_json(
            gap_path(cfg.data_root, asset, month_bounds_ms(first, last)[1]),
            {"missing_ts_ms": packed["gaps"][:10000], "n_gaps": len(packed["gaps"])},
        )
    if "dollar" in cfg.bar_types:
        save_dollar_state(cfg.data_root, asset, acc)
    packed["cached"] = False
    return packed


def _asset_range(cfg, asset: str) -> tuple[date, date]:
    listings = load_listing_dates()
    from pipeline.config import effective_start
    start = effective_start(asset, cfg.start, listings) if cfg.market != "fx" else max(cfg.start, date(2020, 1, 1))
    return start, cfg.end


def load_bar_months(root: Path, asset: str, bar_type: str, start: date, end: date) -> pl.DataFrame:
    from pipeline.partitions import load_partitions
    from pipeline.paths import bars_month_path
    frames = []
    for y, m in iter_months(start, end):
        path = bars_month_path(root, bar_type, asset, y, m)
        if path.exists() and sidecar_ok(path):
            frames.append(pl.read_parquet(path))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed").sort("ts_ms").unique(subset=["ts_ms"], keep="last", maintain_order=True)


def load_daily_months(root: Path, asset: str, start: date, end: date) -> pl.DataFrame:
    frames = []
    for y, m in iter_months(start, end):
        path = _daily_path(root, asset, y, m)
        if path.exists():
            frames.append(pl.read_parquet(path))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def feature_lookback_days(cfg, bar_type: str, n_bars: int | None = None) -> int:
    windows = cfg.windows or load_windows()
    floor_days, max_days = load_embargo_policy()
    scaled = scale_windows(windows, bar_type, None, max_lookback_days=max_days)
    width = TIME_WIDTH_MS.get(bar_type) or 5 * 60_000
    bars = scaled.max_lookback
    return max(floor_days, int((bars * width) / DAY_MS) + 2)


def write_features_month(cfg, asset: str, bar_type: str, year: int, month: int, attach_fn) -> str:
    """Compute features for one month with a lookback warmup; write only the month's rows."""
    first, last = clip_month(year, month, *(_asset_range(cfg, asset)))
    params = partition_params(
        asset=asset, bar_type=bar_type, start=first, end=last,
        extra={"include_oi": cfg.include_oi, "stage": "features"},
    )
    path = features_month_path(cfg.data_root, bar_type, asset, year, month)
    if partition_fresh(path, params, force=cfg.force):
        return "cached"
    lookback = feature_lookback_days(cfg, bar_type)
    warm_start = first - timedelta(days=lookback)
    bars = load_bar_months(cfg.data_root, asset, bar_type, warm_start, last)
    if bars.is_empty():
        return "empty"
    daily = load_daily_months(cfg.data_root, asset, warm_start, last)
    featured = attach_fn(cfg, asset, warm_start, {bar_type: bars}, daily)
    df = featured[bar_type]
    month_df = _in_month(df, first, last)
    if month_df.is_empty():
        return "empty"
    write_frame(month_df, path, "features_v1", params)
    return "written"


def load_feature_months(root: Path, asset: str, bar_type: str, start: date, end: date) -> pl.DataFrame:
    frames = []
    for y, m in iter_months(start, end):
        path = features_month_path(root, bar_type, asset, y, m)
        if path.exists() and sidecar_ok(path):
            frames.append(pl.read_parquet(path))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed").sort("ts_ms").unique(subset=["ts_ms"], keep="last", maintain_order=True)


def write_dataset_month(cfg, assets: list[str], bar_type: str, year: int, month: int, meta: dict, *, cross_asset: bool = True) -> dict[str, str]:
    """Cross-asset, targets, splits, embargo for one month; write gold partitions."""
    windows: Windows = cfg.windows or load_windows()
    splits = cfg.splits
    floor_days, max_days = load_embargo_policy()
    width = TIME_WIDTH_MS.get(bar_type) or 5 * 60_000
    embargo = embargo_bars_for(bar_type, windows, width if bar_type in TIME_WIDTH_MS else None,
                               calendar_floor_days=floor_days, max_lookback_days=max_days)
    lookback_days = max(floor_days, int((embargo * width) / DAY_MS) + 2)
    horizon_days = max(1, int((max(windows.horizons) * width) / DAY_MS) + 2)
    status = {}
    # Determine the union range covering all assets this month.
    # Load a padded window, join cross-asset, then write each asset's month slice.
    frames: dict[str, pl.DataFrame] = {}
    ranges = {}
    for asset in assets:
        first, last = clip_month(year, month, *(_asset_range(cfg, asset)))
        ranges[asset] = (first, last)
        params = partition_params(
            asset=asset, bar_type=bar_type, start=first, end=last,
            extra={"stage": "dataset", "include_oi": cfg.include_oi, "voladj": cfg.voladj, "clipped": cfg.clipped},
        )
        path = gold_month_path(cfg.data_root, bar_type, asset, year, month)
        if partition_fresh(path, params, force=cfg.force):
            status[asset] = "cached"
            continue
        warm = first - timedelta(days=lookback_days)
        tail = last + timedelta(days=horizon_days)
        feat = load_feature_months(cfg.data_root, asset, bar_type, warm, tail)
        if feat.is_empty():
            status[asset] = "empty"
            continue
        frames[asset] = feat
    todo = {a: f for a, f in frames.items()}
    if todo and cross_asset:
        todo = add_cross_asset(todo, bar_type=bar_type, fast=windows.fast, ma=min(windows.ma, max(windows.fast, 2)))
    listings = load_listing_dates()
    for asset, df in todo.items():
        first, last = ranges[asset]
        params = partition_params(
            asset=asset, bar_type=bar_type, start=first, end=last,
            extra={"stage": "dataset", "include_oi": cfg.include_oi, "voladj": cfg.voladj, "clipped": cfg.clipped},
        )
        df = add_targets(df, windows.horizons, clipped=cfg.clipped, voladj=cfg.voladj)
        df = assign_split(df, splits)
        diffs = df["ts_ms"].diff().drop_nulls() if df.height > 2 else None
        median = int(diffs.median()) if diffs is not None and diffs.len() else None
        embargo_n = embargo_bars_for(
            bar_type, windows, median, calendar_floor_days=floor_days, max_lookback_days=max_days,
        )
        df = apply_embargo(df, embargo_n)
        df = add_universe_columns(df, asset, listings.get(asset))
        month_df = _in_month(df, first, last)
        if month_df.is_empty():
            status[asset] = "empty"
            continue
        path = gold_month_path(cfg.data_root, bar_type, asset, year, month)
        digest = write_frame(month_df, path, "dataset_v1", params)
        feats = [c for c in month_df.columns if c.startswith(("norm_", "xd_", "wh_", "liq_", "rv_", "hbr_", "funding_day", "s3_", "bs_")) or c == "hurst_regime"]
        man = manifest_payload(
            asset=asset, bar_type=bar_type, df=month_df, feature_cols=feats, params=params,
            embargo_bars=embargo_n, inputs=[], lags={},
            built_at_ms=utc_now_ms(),
        )
        man["sha256"] = digest
        packed = meta.get(asset, {})
        man["dollar_open_tail_trades"] = packed.get("dollar_excluded")
        man["n_bronze_trades"] = packed.get("n_bronze")
        atomic_write_json(gold_manifest_path(path), man)
        status[asset] = "written"
    return status


def build_fx_month(cfg, asset: str, first: date, last: date) -> dict:
    ticks = read_fx_ticks(cfg.data_root, asset, first, last)
    if ticks.is_empty():
        return {"bars": {}, "n_ticks": 0, "gaps": []}
    # Include a 1-day pad of previous ticks? Month boundaries do not split time bars.
    ladder = build_fx_ladder(ticks, asset, [b for b in cfg.bar_types if b in TIME_WIDTH_MS])
    return {"bars": ladder, "n_ticks": ticks.height, "gaps": gaps_ms(ladder.get("time_1m", pl.DataFrame()))}


def write_fx_bars_month(cfg, asset: str, year: int, month: int) -> dict:
    first, last = clip_month(year, month, *(_asset_range(cfg, asset)))
    needed = []
    for bt in cfg.bar_types:
        path = bars_month_path(cfg.data_root, bt, asset, year, month)
        params = bars_params(asset, bt, first, last)
        if not partition_fresh(path, params, force=cfg.force):
            needed.append(bt)
    if not needed:
        return {"cached": True, "n_ticks": 0, "gaps": []}
    packed = build_fx_month(cfg, asset, first, last)
    for bt, df in packed["bars"].items():
        if df.is_empty():
            continue
        path = bars_month_path(cfg.data_root, bt, asset, year, month)
        params = bars_params(asset, bt, first, last)
        write_frame(df, path, "bars_v1", params)
    packed["cached"] = False
    return packed


def write_fx_features_month(cfg, assets: list[str], bar_type: str, year: int, month: int) -> dict[str, str]:
    """FX physics + USD-strength for one month, one cadence."""
    from pipeline.config import fx_pip
    from pipeline.features import add_usd_strength, compute_fx_features
    from datetime import timedelta

    status: dict[str, str] = {}
    lookback = feature_lookback_days(cfg, bar_type)
    frames: dict[str, pl.DataFrame] = {}
    ranges = {}
    for asset in assets:
        first, last = clip_month(year, month, *(_asset_range(cfg, asset)))
        ranges[asset] = (first, last)
        params = partition_params(
            asset=asset, bar_type=bar_type, start=first, end=last,
            extra={"stage": "features", "market": "fx"},
        )
        path = features_month_path(cfg.data_root, bar_type, asset, year, month)
        if partition_fresh(path, params, force=cfg.force):
            status[asset] = "cached"
            continue
        warm = first - timedelta(days=lookback)
        bars = load_bar_months(cfg.data_root, asset, bar_type, warm, last)
        if bars.is_empty():
            status[asset] = "empty"
            continue
        frames[asset] = compute_fx_features(bars, fx_pip(asset))
    if frames:
        frames = add_usd_strength(frames)
    for asset, df in frames.items():
        first, last = ranges[asset]
        params = partition_params(
            asset=asset, bar_type=bar_type, start=first, end=last,
            extra={"stage": "features", "market": "fx"},
        )
        month_df = _in_month(df, first, last)
        if month_df.is_empty():
            status[asset] = "empty"
            continue
        path = features_month_path(cfg.data_root, bar_type, asset, year, month)
        write_frame(month_df, path, "features_v1", params)
        status[asset] = "written"
    return status
