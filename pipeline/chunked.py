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
    dollar_spec,
    embargo_bars_for,
    liq_proxy_usd,
    load_embargo_policy,
    load_lags,
    load_listing_dates,
    load_windows,
    scale_windows,
    whale_spec,
)
from pipeline.constants import CONTRACT_VERSION, DAY_MS, TIME_WIDTH_MS
from pipeline.dataset import (
    add_targets,
    add_universe_columns,
    apply_embargo,
    assign_split,
    input_families,
    manifest_payload,
    utc_now_ms,
)
from pipeline.features import (
    add_cross_asset, load_whale_hists, rows_to_frame, summarize_day,
    trade_notional_hist, whale_threshold,
)
from pipeline.fetch_crypto import iter_days as _iter_days  # noqa: F401
from pipeline.fetch_fx import read_fx_ticks
from pipeline.io import atomic_write_json, atomic_write_parquet, file_sha256, params_hash, read_parquet_metadata
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


def count_bronze_trades(root: Path, asset: str, first: date, last: date) -> int:
    n = 0
    for day in iter_days(first, last):
        path = trades_path(root, asset, day)
        if path.exists():
            n += pl.read_parquet(path, columns=["ts_ms"]).height
    return n


def save_dollar_state(root: Path, asset: str, acc: DollarAccumulator, through: str) -> None:
    """Persist the accumulator after month ``through`` ("YYYY-MM") is fully written.

    The marker is what makes resume safe: a dollar partition only counts as done
    when the state says it has been folded in (``state.through >= month``). A
    crash between writing the partition and this call leaves ``through`` one
    month behind, and the next run rebuilds that month from this very state.
    """
    acc.through = through
    meta, tail = acc.state()
    meta["asset"] = asset
    meta["contract_version"] = CONTRACT_VERSION
    path = dollar_state_path(root, asset)
    # Two files make one state. Write the fragments first and the JSON (which names
    # them) last, so a crash in between is detectable: the footer hash of the
    # fragments is checked against the JSON on read.
    if tail is not None:
        atomic_write_parquet(
            tail, path,
            schema_name="dollar_state_v1",
            params=_state_tail_params(meta),
        )
    atomic_write_json(dollar_state_meta_path(root, asset), meta)
    if tail is None:
        if path.exists():
            path.unlink()
        sidecar = path.with_suffix(path.suffix + ".sha256")
        if sidecar.exists():
            sidecar.unlink()


def _state_tail_params(meta: dict) -> dict:
    return {
        "asset": meta.get("asset"),
        "through": meta.get("through"),
        "next_id": meta.get("next_id"),
        "cum": meta.get("cum"),
        "n_partials": meta.get("n_partials"),
    }


def read_dollar_state(root: Path, asset: str, spec) -> DollarAccumulator | None:
    """The saved accumulator, or ``None`` when absent, unreadable, or built for another spec."""
    import json

    meta_path = dollar_state_meta_path(root, asset)
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, ValueError):
        return None
    if (
        abs(float(meta.get("threshold", -1)) - spec.bootstrap) > 1e-9
        or meta.get("target_bars_per_day") != spec.target_bars_per_day
        or int(meta.get("window_days", -1)) != spec.window_days
    ):
        return None
    if meta.get("contract_version") != CONTRACT_VERSION:
        return None
    path = dollar_state_path(root, asset)
    frag = None
    if int(meta.get("n_partials", 0)) > 0:
        if not sidecar_ok(path):
            return None
        try:
            if read_parquet_metadata(path).get("params_hash") != params_hash(_state_tail_params(meta)):
                return None  # torn write: fragments and JSON come from different months
        except Exception:
            return None
        frag = pl.read_parquet(path)
    return DollarAccumulator.restore(meta, frag)


def new_accumulator(spec) -> DollarAccumulator:
    return DollarAccumulator(
        spec.bootstrap, target_bars_per_day=spec.target_bars_per_day, window_days=spec.window_days,
    )


def dollar_spec_for(cfg, asset: str):
    return dollar_spec(asset, (cfg.threshold_overrides or {}).get(asset))


def dollar_extra(cfg, asset: str, start: date) -> dict:
    """Partition params of a dollar month. ``chain_start`` ties every month to one origin.

    Dollar bar edges depend on where accumulation began, so a different start
    date must invalidate the whole fabric, not just its first month.
    """
    return {"dollar": dollar_spec_for(cfg, asset).as_params(), "chain_start": start.isoformat()}


def plan_dollar(cfg, asset: str, months: list[tuple[int, int]], start: date) -> tuple[int, DollarAccumulator | None]:
    """How many leading months are already consistent, and the accumulator to continue from.

    Returns ``(f, acc)``. Months ``< f`` keep their dollar partitions. Months
    ``>= f`` must be rebuilt, in order, starting from ``acc``. When nothing can
    be reused, ``f == 0`` and ``acc`` is fresh. When every month is reusable,
    ``f == len(months)`` and ``acc`` is the saved state (it is not fed again).
    """
    spec = dollar_spec_for(cfg, asset)
    if cfg.force or "dollar" not in cfg.bar_types:
        return 0, new_accumulator(spec)
    state = read_dollar_state(cfg.data_root, asset, spec)
    extra = dollar_extra(cfg, asset, start)
    f = 0
    for y, m in months:
        first, last = clip_month(y, m, *(_asset_range(cfg, asset)))
        path = bars_month_path(cfg.data_root, "dollar", asset, y, m)
        fresh = partition_fresh(path, bars_params(asset, "dollar", first, last, extra))
        if not (fresh and state is not None and state.through is not None and state.through >= f"{y:04d}-{m:02d}"):
            break
        f += 1
    if f == len(months):
        return f, state
    if f == 0:
        return 0, new_accumulator(spec)
    prev = f"{months[f - 1][0]:04d}-{months[f - 1][1]:02d}"
    if state is not None and state.through == prev:
        return f, state
    # The state is ahead of (or unrelated to) the last good month; there is nothing
    # to continue from, so the whole fabric is rebuilt.
    return 0, new_accumulator(spec)


def bars_params(asset: str, bar_type: str, first: date, last: date, extra: dict | None = None) -> dict:
    return partition_params(asset=asset, bar_type=bar_type, start=first, end=last, extra=extra)


def build_crypto_month(cfg, asset: str, first: date, last: date, acc: DollarAccumulator | None) -> dict:
    """Stream one month of bronze trades into 1m / coarser / dollar bars. Bounded memory.

    ``acc`` is ``None`` when the dollar fabric is not being rebuilt for this month;
    trades are then never fed to it, so a cached month can not be counted twice.
    """
    partials = []
    daily_rows = []
    n_bronze = 0
    wspec = whale_spec(asset)
    whale_hists = load_whale_hists(cfg.data_root, asset, first, wspec.window_days)
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
        if acc is not None:
            acc.add(trades)
        thresh = whale_threshold(whale_hists, wspec)
        daily_rows.append(summarize_day(
            trades, one_m, asset=asset, day=day,
            whale_usd=thresh, liq_usd=liq_proxy_usd(asset),
        ))
        usd = (trades["price"] * trades["qty"]).to_numpy()
        whale_hists.append(trade_notional_hist(usd))
        whale_hists = whale_hists[-wspec.window_days:]
        del trades
    bars_1m = merge_partial_time(partials, asset, "time_1m", TIME_WIDTH_MS["time_1m"])
    del partials
    ladder = {"time_1m": bars_1m}
    for name in cfg.bar_types:
        if name in TIME_WIDTH_MS and name != "time_1m":
            ladder[name] = bars_to_time(bars_1m, TIME_WIDTH_MS[name], asset, name)
    out_bars = {k: v for k, v in ladder.items() if k in cfg.bar_types}
    excluded = None
    if acc is not None and "dollar" in cfg.bar_types:
        dollar, excluded = acc.emit_completed(asset)
        out_bars["dollar"] = dollar
    return {
        "bars": out_bars,
        "ladder": ladder,
        "daily": rows_to_frame(daily_rows),
        "n_bronze": n_bronze,
        "dollar_excluded": excluded,
        "gaps": gaps_ms(bars_1m),
    }


def write_bars_month(
    cfg, asset: str, year: int, month: int, acc: DollarAccumulator | None, session=None,
    *, start: date | None = None, rebuild_dollar: bool = False,
) -> dict:
    """Write the month's bar partitions that are missing or stale.

    ``rebuild_dollar`` says the dollar partition for this month must be rebuilt
    from ``acc`` (see :func:`plan_dollar`). Otherwise the accumulator is untouched.
    """
    first, last = clip_month(year, month, *(_asset_range(cfg, asset)))
    start = start or _asset_range(cfg, asset)[0]
    want_dollar = "dollar" in cfg.bar_types
    extra = dollar_extra(cfg, asset, start) if want_dollar else {}
    needed = []
    for bt in cfg.bar_types:
        if bt == "dollar":
            if rebuild_dollar:
                needed.append(bt)
            continue
        path = bars_month_path(cfg.data_root, bt, asset, year, month)
        if partition_fresh(path, bars_params(asset, bt, first, last), force=cfg.force):
            continue
        needed.append(bt)
    if not needed:
        return {
            "cached": True,
            "n_bronze": count_bronze_trades(cfg.data_root, asset, first, last),
            "dollar_excluded": acc.tail_trades() if acc is not None else None,
            "gaps": [],
        }
    use_acc = acc if (want_dollar and rebuild_dollar) else None
    packed = build_crypto_month(cfg, asset, first, last, use_acc)
    for bt, df in packed["bars"].items():
        if bt not in needed or df.is_empty():
            continue
        params = bars_params(asset, bt, first, last, extra if bt == "dollar" else None)
        write_frame(df, bars_month_path(cfg.data_root, bt, asset, year, month), "bars_v1", params)
    daily = packed["daily"]
    if daily is not None and not daily.is_empty():
        dpath = _daily_path(cfg.data_root, asset, year, month)
        write_frame(
            daily, dpath, "daily_panel_v1",
            bars_params(asset, "daily", first, last, extra={"whale": whale_spec(asset).as_params()}),
        )
    gpath = gap_path(cfg.data_root, asset, month_bounds_ms(first, last)[1])
    if packed["gaps"]:
        # Stamp with month end close.
        atomic_write_json(gpath, {"missing_ts_ms": packed["gaps"][:10000], "n_gaps": len(packed["gaps"])})
    elif gpath.exists():
        gpath.unlink()  # a rebuild that closed the holes must not leave the old list behind
    if use_acc is not None:
        save_dollar_state(cfg.data_root, asset, use_acc, f"{year:04d}-{month:02d}")
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


def _next_features_digest(cfg, asset: str, bar_type: str, year: int, month: int) -> str | None:
    """Digest of the *next* month's feature partition, or None when it does not exist.

    Gold targets look ``max(horizons)`` bars ahead, so the last rows of month M need
    month M+1. A gold partition built before M+1 existed has null targets there and
    must be rebuilt once M+1 appears. Putting this digest in the partition params
    makes that happen and nothing else.
    """
    ny, nm = (year + 1, 1) if month == 12 else (year, month + 1)
    path = features_month_path(cfg.data_root, bar_type, asset, ny, nm)
    if not sidecar_ok(path):
        return None
    return path.with_suffix(path.suffix + ".sha256").read_text().strip()


def _dataset_params(cfg, asset: str, bar_type: str, first: date, last: date, year: int, month: int) -> dict:
    return partition_params(
        asset=asset, bar_type=bar_type, start=first, end=last,
        extra={
            "stage": "dataset", "include_oi": cfg.include_oi, "voladj": cfg.voladj,
            "clipped": cfg.clipped,
            "next_features": _next_features_digest(cfg, asset, bar_type, year, month),
        },
    )



def _embargo_padding(cfg, asset: str, bar_type: str, windows, floor_days: int, max_days: int) -> tuple[int, int, int]:
    """Global embargo length and the calendar padding needed to apply it.

    Time bars use the fixed clock width. Event bars (dollar, tick, volume) use
    the median duration of the *entire* fabric already on disk, so a split that
    sits on a month boundary is labelled the same way in both partitions. The
    lookback is long enough to hold that many bars, not a 5-minute guess.
    """
    if bar_type in TIME_WIDTH_MS:
        median = TIME_WIDTH_MS[bar_type]
    else:
        start, end = _asset_range(cfg, asset)
        bars = load_bar_months(cfg.data_root, asset, bar_type, start, end)
        median = None
        if bars.height > 2:
            diffs = bars["ts_ms"].diff().drop_nulls()
            if diffs.len():
                median = int(diffs.median())
    embargo_n = embargo_bars_for(
        bar_type, windows, median, calendar_floor_days=floor_days, max_lookback_days=max_days,
    )
    width = median if median and median > 0 else 5 * 60_000
    if bar_type in TIME_WIDTH_MS:
        width = TIME_WIDTH_MS[bar_type]
    lookback_days = max(floor_days, int((embargo_n * width) / DAY_MS) + 2)
    horizon_days = max(1, int((max(windows.horizons) * width) / DAY_MS) + 2)
    return embargo_n, lookback_days, horizon_days


def write_dataset_month(cfg, assets: list[str], bar_type: str, year: int, month: int, meta: dict, *, cross_asset: bool = True) -> dict[str, str]:
    """Cross-asset, targets, splits, embargo for one month; write gold partitions."""
    windows: Windows = cfg.windows or load_windows()
    splits = cfg.splits
    floor_days, max_days = load_embargo_policy()
    status = {}
    # Determine the union range covering all assets this month.
    # Load a padded window, join cross-asset, then write each asset's month slice.
    frames: dict[str, pl.DataFrame] = {}
    ranges = {}
    embargo_for: dict[str, int] = {}
    for asset in assets:
        first, last = clip_month(year, month, *(_asset_range(cfg, asset)))
        ranges[asset] = (first, last)
        params = _dataset_params(cfg, asset, bar_type, first, last, year, month)
        path = gold_month_path(cfg.data_root, bar_type, asset, year, month)
        if partition_fresh(path, params, force=cfg.force):
            status[asset] = "cached"
            continue
        embargo_n, lookback_days, horizon_days = _embargo_padding(
            cfg, asset, bar_type, windows, floor_days, max_days,
        )
        embargo_for[asset] = embargo_n
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
        params = _dataset_params(cfg, asset, bar_type, first, last, year, month)
        df = add_targets(df, windows.horizons, clipped=cfg.clipped, voladj=cfg.voladj)
        df = assign_split(df, splits)
        embargo_n = embargo_for[asset]
        df = apply_embargo(df, embargo_n)
        df = add_universe_columns(df, asset, listings.get(asset))
        month_df = _in_month(df, first, last)
        if month_df.is_empty():
            status[asset] = "empty"
            continue
        path = gold_month_path(cfg.data_root, bar_type, asset, year, month)
        digest = write_frame(month_df, path, "dataset_v1", params)
        feats = [
            c for c in month_df.columns
            if c.startswith(("norm_", "xd_", "wh_", "liq_", "rv_", "hbr_", "funding_day", "s3_", "bs_",
                             "etf_", "dv_", "soc_", "stbl_"))
            or c in ("hurst_regime", "funding_rate", "oi_value")
        ]
        inputs, lags = input_families(month_df.columns, load_lags())
        man = manifest_payload(
            asset=asset, bar_type=bar_type, df=month_df, feature_cols=feats, params=params,
            embargo_bars=embargo_n, inputs=inputs, lags=lags,
            built_at_ms=utc_now_ms(), splits=splits,
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
