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
from pipeline.partitions import inputs_digest, partition_fresh, partition_params, sidecar_ok
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
    """Rows in the bronze trade files of ``[first, last]`` (from parquet footers)."""
    import pyarrow.parquet as pq

    n = 0
    for day in iter_days(first, last):
        path = trades_path(root, asset, day)
        if path.exists():
            n += int(pq.read_metadata(path).num_rows)
    return n


def bronze_digest(root: Path, asset: str, first: date, last: date) -> str:
    """Digest of the bronze trade days a bar month is built from."""
    return inputs_digest(trades_path(root, asset, d) for d in iter_days(first, last))


def _snapshot_paths(root: Path, asset: str, through: str) -> tuple[Path, Path]:
    base = dollar_state_path(root, asset).parent / f"dollar_{asset}"
    return base / f"{through}.parquet", base / f"{through}.json"


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
    # The sha256 of the dollar partition this state follows. A snapshot whose
    # partition was rewritten afterwards (crash between the two writes, or a
    # rebuild from a different upstream) no longer matches and is refused.
    meta["partition_sha"] = getattr(acc, "partition_sha", None)
    # One snapshot per month (``_state/dollar_<ASSET>/<YYYY-MM>``) lets a later run
    # resume from the month before the first stale one, instead of from 2020 when
    # only the current partial month changed. The flat ``dollar_<ASSET>`` pair is
    # the latest state, kept for inspection.
    snap_frag, snap_json = _snapshot_paths(root, asset, through)
    for frag_path, json_path in (
        (snap_frag, snap_json),
        (dollar_state_path(root, asset), dollar_state_meta_path(root, asset)),
    ):
        # Two files make one state. Write the fragments first and the JSON (which
        # names them) last, so a crash in between is detectable: the footer hash of
        # the fragments is checked against the JSON on read.
        if tail is not None:
            atomic_write_parquet(
                tail, frag_path,
                schema_name="dollar_state_v1",
                params=_state_tail_params(meta),
            )
        atomic_write_json(json_path, meta)
        if tail is None:
            if frag_path.exists():
                frag_path.unlink()
            sidecar = frag_path.with_suffix(frag_path.suffix + ".sha256")
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


def read_dollar_state(root: Path, asset: str, spec, through: str | None = None) -> DollarAccumulator | None:
    """The saved accumulator, or ``None`` when absent, unreadable, or built for another spec.

    ``through`` selects the snapshot taken after that month; ``None`` is the latest.
    """
    import json

    if through is None:
        meta_path = dollar_state_meta_path(root, asset)
        path = dollar_state_path(root, asset)
    else:
        path, meta_path = _snapshot_paths(root, asset, through)
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
    if through is not None and meta.get("through") != through:
        return None
    if through is not None:
        y, m = int(through[:4]), int(through[5:7])
        part = bars_month_path(root, "dollar", asset, y, m)
        side = part.with_suffix(part.suffix + ".sha256")
        try:
            if not side.exists() or meta.get("partition_sha") != side.read_text().strip():
                return None
        except OSError:
            return None
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
    ``f == len(months)`` and ``acc`` is that last month's state (it is not fed again).

    A month is consistent when its partition is fresh (same chain origin, same
    bronze) and the snapshot taken after it exists and is intact. The chain then
    continues from the snapshot of month ``f - 1``, so only months from the first
    stale one onward are rebuilt.
    """
    if "dollar" not in cfg.bar_types:
        return 0, None
    spec = dollar_spec_for(cfg, asset)
    if cfg.force:
        return 0, new_accumulator(spec)
    f = 0
    last_state = None
    for y, m in months:
        ym = f"{y:04d}-{m:02d}"
        path = bars_month_path(cfg.data_root, "dollar", asset, y, m)
        if not partition_fresh(path, dollar_params(cfg, asset, y, m, start)):
            break
        state = read_dollar_state(cfg.data_root, asset, spec, through=ym)
        if state is None:
            break
        last_state = state
        f += 1
    if f == 0 or last_state is None:
        return 0, new_accumulator(spec)
    return f, last_state


def dollar_params(cfg, asset: str, year: int, month: int, start: date) -> dict:
    """Params of one dollar month: chain origin, the month's bronze, and the state it
    continued from (the previous month's snapshot). A rebuilt earlier month therefore
    makes every later month stale, even if a run was interrupted halfway through the
    cascade."""
    first, last = clip_month(year, month, *(_asset_range(cfg, asset)))
    py, pm = (year - 1, 12) if month == 1 else (year, month - 1)
    prev = None
    if (py, pm) >= (start.year, start.month):
        _frag, prev_json = _snapshot_paths(cfg.data_root, asset, f"{py:04d}-{pm:02d}")
        prev = file_sha256(prev_json) if prev_json.exists() else "-"
    return bars_params(
        asset, "dollar", first, last,
        {**dollar_extra(cfg, asset, start), "bronze": bronze_digest(cfg.data_root, asset, first, last),
         "prev_state": prev},
    )


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
    bronze = {"bronze": bronze_digest(cfg.data_root, asset, first, last)}
    wspec = whale_spec(asset)
    whale_lookback = first - timedelta(days=int(wspec.window_days) * 4 + 30)
    daily_params = bars_params(asset, "daily", first, last, extra={
        "whale": wspec.as_params(), **bronze,
        # Whale sizes come from the trailing days' trades, so those days are inputs too.
        "whale_history": bronze_digest(cfg.data_root, asset, whale_lookback, first - timedelta(days=1)),
    })
    needed = []
    for bt in cfg.bar_types:
        if bt == "dollar":
            if rebuild_dollar:
                needed.append(bt)
            continue
        path = bars_month_path(cfg.data_root, bt, asset, year, month)
        if partition_fresh(path, bars_params(asset, bt, first, last, bronze), force=cfg.force):
            continue
        needed.append(bt)
    if not needed and not partition_fresh(_daily_path(cfg.data_root, asset, year, month), daily_params, force=cfg.force):
        if count_bronze_trades(cfg.data_root, asset, first, last):
            needed.append("_daily")  # the daily panel comes out of the same pass
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
        if bt not in needed:
            continue
        if bt == "dollar":
            # Written even when no bar closed this month: the partition (empty or
            # not) is what marks the month as folded into the chain.
            params = dollar_params(cfg, asset, year, month, start)
        elif df.is_empty():
            continue
        else:
            params = bars_params(asset, bt, first, last, bronze)
        digest = write_frame(df, bars_month_path(cfg.data_root, bt, asset, year, month), "bars_v1", params)
        if bt == "dollar" and use_acc is not None:
            use_acc.partition_sha = digest
    daily = packed["daily"]
    if daily is not None and not daily.is_empty():
        dpath = _daily_path(cfg.data_root, asset, year, month)
        write_frame(daily, dpath, "daily_panel_v1", daily_params)
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
            df = pl.read_parquet(path)
            if df.height:
                frames.append(df)
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


def _months_overlapping(first: date, last: date) -> list[tuple[int, int]]:
    return list(iter_months(first, last))


_EXTERNAL_CACHE: dict = {}


def _external_slice_digest(root: Path, lo: date, hi: date, asset: str | None = None) -> str:
    """Digest of the external rows (all sources) whose event day is in ``[lo, hi]``.

    External series are single files that grow every run, so their file hash
    would invalidate every month. Only the rows a month can see matter.
    """
    import hashlib
    from pipeline.fetch_external import external_dir, load_external

    h = hashlib.sha256()
    lo_ms, hi_ms = _day_ms(lo), _day_ms(hi + timedelta(days=1))
    for source in ("defillama", "deribit", "farside", "wikipedia"):
        files = sorted(external_dir(root, source).glob("*.parquet")) if external_dir(root, source).exists() else []
        key = (str(root), source, tuple((f.name, inputs_digest([f])) for f in files))
        df = _EXTERNAL_CACHE.get(key)
        if df is None:
            df = load_external(root, source)
            if len(_EXTERNAL_CACHE) > 16:
                _EXTERNAL_CACHE.clear()
            _EXTERNAL_CACHE[key] = df
        if df.is_empty() or "event_time_ms" not in df.columns:
            h.update(f"{source}:-\n".encode())
            continue
        part = df.filter((pl.col("event_time_ms") >= lo_ms) & (pl.col("event_time_ms") < hi_ms))
        if asset is not None and "asset" in part.columns:
            # Another asset's rows never reach this asset's features, so they must not
            # invalidate them (BTC's DVOL is joined onto every asset).
            part = part.filter(pl.col("asset").is_in([asset, "BTCUSDT"]))
        part = part.sort([c for c in ("event_time_ms", "asset") if c in part.columns])
        h.update(f"{source}:".encode())
        h.update(part.write_csv().encode())
    return h.hexdigest()[:32]


def features_inputs_digest(cfg, asset: str, bar_type: str, warm_start: date, last: date) -> str:
    """Everything a crypto feature month reads: bars (+1d for basis), the daily
    trade panel, funding, metrics, perp 1d, and the external rows in range."""
    from pipeline.paths import funding_path, metrics_path, perp_1d_path

    root = cfg.data_root
    months = _months_overlapping(warm_start, last)
    paths = []
    for y, m in months:
        paths.append(bars_month_path(root, bar_type, asset, y, m))
        if bar_type != "time_1d":
            paths.append(bars_month_path(root, "time_1d", asset, y, m))
        paths.append(_daily_path(root, asset, y, m))
    # The frame's last bar closes at 00:00 of the next day, so slow series are
    # read through ``last + 1``.
    for y, m in _months_overlapping(warm_start, last + timedelta(days=1)):
        paths.append(funding_path(root, asset, y, m))
    for d in iter_days(warm_start, last + timedelta(days=1)):
        paths.append(metrics_path(root, asset, d))
        paths.append(perp_1d_path(root, asset, d))
    return inputs_digest(paths) + _external_slice_digest(root, warm_start - timedelta(days=3), last + timedelta(days=1), asset)


def write_features_month(cfg, asset: str, bar_type: str, year: int, month: int, attach_fn) -> str:
    """Compute features for one month with a lookback warmup; write only the month's rows."""
    first, last = clip_month(year, month, *(_asset_range(cfg, asset)))
    lookback = feature_lookback_days(cfg, bar_type)
    warm_start = first - timedelta(days=lookback)
    params = partition_params(
        asset=asset, bar_type=bar_type, start=first, end=last,
        extra={
            "include_oi": cfg.include_oi, "stage": "features",
            "inputs": features_inputs_digest(cfg, asset, bar_type, warm_start, last),
        },
    )
    path = features_month_path(cfg.data_root, bar_type, asset, year, month)
    if partition_fresh(path, params, force=cfg.force):
        return "cached"
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
            df = pl.read_parquet(path)
            if df.height:
                frames.append(df)
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed").sort("ts_ms").unique(subset=["ts_ms"], keep="last", maintain_order=True)


def _dataset_params(cfg, asset: str, bar_type: str, first: date, last: date,
                    embargo_n: int, inputs: str, cohort: list[str] | None) -> dict:
    """Gold params. ``inputs`` digests every feature month the partition reads
    (its own, with the warm-up and the look-ahead of the targets, and the peers'
    when cross-asset columns are built), so gold is rebuilt when month M+1
    appears, when a feature month is rebuilt, or when the peer set changes."""
    return partition_params(
        asset=asset, bar_type=bar_type, start=first, end=last,
        extra={
            "stage": "dataset", "include_oi": cfg.include_oi, "voladj": cfg.voladj,
            "clipped": cfg.clipped, "embargo_bars": int(embargo_n),
            "cohort": cohort, "inputs": inputs,
            "splits": [str(x) for x in (cfg.splits.train_end, cfg.splits.val_end, cfg.splits.oos_end)] if cfg.splits else None,
        },
    )


def nominal_bar_ms(cfg, asset: str, bar_type: str) -> int:
    """Clock width used to size the embargo of one fabric.

    Time bars: the fixed width. Dollar bars: the design width, one day divided
    by ``target_bars_per_day`` (5 minutes at 288), which is global and stable as
    months are added. A constant-size override (tests) has no design width, so
    the observed median of the fabric on disk is used.
    """
    if bar_type in TIME_WIDTH_MS:
        return TIME_WIDTH_MS[bar_type]
    if bar_type == "dollar":
        spec = dollar_spec_for(cfg, asset)
        if spec.target_bars_per_day:
            return int(DAY_MS / float(spec.target_bars_per_day))
    start, end = _asset_range(cfg, asset)
    diffs = []
    for y, m in iter_months(start, end):
        path = bars_month_path(cfg.data_root, bar_type, asset, y, m)
        if sidecar_ok(path):
            ts = pl.read_parquet(path, columns=["ts_ms"])["ts_ms"]
            if ts.len() > 1:
                diffs.append(ts.diff().drop_nulls())
    if diffs:
        med = pl.concat(diffs).median()
        if med and med > 0:
            return int(med)
    return 5 * 60_000


def _embargo_padding(cfg, asset: str, bar_type: str, windows, floor_days: int, max_days: int) -> tuple[int, int]:
    """Global embargo length (bars) and the calendar padding needed to apply it.

    The padding is loaded on both sides of the month so that a split edge up to
    one embargo away is inside the frame: the tail of the earlier segment is
    flagged in the months before the edge, the head of the later one in the
    months after it. Event bars get three times the nominal span because their
    density varies.
    """
    width = nominal_bar_ms(cfg, asset, bar_type)
    embargo_n = embargo_bars_for(
        bar_type, windows, width, calendar_floor_days=floor_days, max_lookback_days=max_days,
    )
    if getattr(cfg, "market", "crypto") == "fx":
        # FX features use fixed windows of up to FX_MAX_WINDOW bars at every cadence,
        # so the embargo must cover them (it can exceed 90 days on 6h-1d FX).
        from pipeline.features import FX_MAX_WINDOW
        embargo_n = max(embargo_n, FX_MAX_WINDOW + 1)
    span_days = int((embargo_n * width) / DAY_MS) + 2
    if bar_type not in TIME_WIDTH_MS:
        span_days *= 3
    elif getattr(cfg, "market", "crypto") == "fx":
        span_days = span_days * 3 // 2 + 3  # five trading days per seven
    horizon_days = int((max(windows.horizons) * width) / DAY_MS) + 2
    return embargo_n, max(floor_days, span_days, horizon_days)


def write_dataset_month(cfg, assets: list[str], bar_type: str, year: int, month: int, meta: dict, *, cross_asset: bool = True) -> dict[str, str]:
    """Cross-asset, targets, splits, embargo for one month; write gold partitions.

    An asset whose history does not reach this month (listed later) is skipped,
    not an error for the whole cohort. Cross-asset columns use every cohort
    member that has features for the month as a peer, whether or not that
    member's own gold is stale, and the cohort is part of the params.
    """
    windows: Windows = cfg.windows or load_windows()
    splits = cfg.splits
    floor_days, max_days = load_embargo_policy()
    status: dict[str, str] = {}
    ranges: dict[str, tuple[date, date, date, date]] = {}
    embargo_for: dict[str, int] = {}
    params_for: dict[str, dict] = {}
    members = []
    for asset in assets:
        a_start, a_end = _asset_range(cfg, asset)
        if (year, month) < (a_start.year, a_start.month) or (year, month) > (a_end.year, a_end.month):
            continue
        first, last = clip_month(year, month, a_start, a_end)
        embargo_n, pad_days = _embargo_padding(cfg, asset, bar_type, windows, floor_days, max_days)
        ranges[asset] = (first, last, first - timedelta(days=pad_days), last + timedelta(days=pad_days))
        embargo_for[asset] = embargo_n
        members.append(asset)
    cohort = sorted(members) if cross_asset and len(members) > 1 else None
    stale = []
    for asset in members:
        first, last, warm, tail = ranges[asset]
        readers = cohort or [asset]
        digest = inputs_digest(
            features_month_path(cfg.data_root, bar_type, a, y, m)
            for a in readers for y, m in iter_months(warm, tail)
        )
        params = _dataset_params(cfg, asset, bar_type, first, last, embargo_for[asset], digest, cohort)
        params_for[asset] = params
        path = gold_month_path(cfg.data_root, bar_type, asset, year, month)
        if partition_fresh(path, params, force=cfg.force):
            status[asset] = "cached"
        else:
            stale.append(asset)
    if not stale:
        return status
    frames: dict[str, pl.DataFrame] = {}
    for asset in (members if cohort else stale):
        first, last, warm, tail = ranges[asset]
        feat = load_feature_months(cfg.data_root, asset, bar_type, warm, tail)
        if not feat.is_empty():
            frames[asset] = feat
    todo = frames
    if todo and cohort:
        todo = add_cross_asset(todo, bar_type=bar_type, fast=windows.fast, ma=min(windows.ma, max(windows.fast, 2)))
    listings = load_listing_dates()
    for asset in stale:
        if asset not in todo:
            status[asset] = "empty"
            continue
        df = todo[asset]
        first, last, _warm, _tail = ranges[asset]
        params = params_for[asset]
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
        man["cohort"] = cohort
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


def count_fx_ticks(root: Path, asset: str, first: date, last: date) -> int:
    """Ticks in the bronze hour files of ``[first, last]`` (from parquet footers)."""
    import pyarrow.parquet as pq
    from pipeline.fetch_fx import iter_hours
    from pipeline.paths import fx_hour_path

    n = 0
    for hour in iter_hours(first, last):
        path = fx_hour_path(root, asset, hour.date(), hour.hour)
        if path.exists():
            n += int(pq.read_metadata(path).num_rows)
    return n


def write_fx_bars_month(cfg, asset: str, year: int, month: int) -> dict:
    first, last = clip_month(year, month, *(_asset_range(cfg, asset)))
    needed = []
    for bt in cfg.bar_types:
        path = bars_month_path(cfg.data_root, bt, asset, year, month)
        params = bars_params(asset, bt, first, last)
        if not partition_fresh(path, params, force=cfg.force):
            needed.append(bt)
    if not needed:
        return {"cached": True, "n_ticks": count_fx_ticks(cfg.data_root, asset, first, last), "gaps": []}
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
    """FX physics + USD-strength for one month, one cadence.

    ``usd_strength_ret`` averages the USD legs of every pair in the run, so the
    whole cohort is loaded whenever any member is stale, and the cohort plus the
    digest of every pair's bars are part of the params.
    """
    from pipeline.config import fx_pip
    from pipeline.features import add_usd_strength, compute_fx_features

    status: dict[str, str] = {}
    lookback = feature_lookback_days(cfg, bar_type)
    ranges = {}
    members = []
    for asset in assets:
        a_start, a_end = _asset_range(cfg, asset)
        if (year, month) < (a_start.year, a_start.month) or (year, month) > (a_end.year, a_end.month):
            continue
        first, last = clip_month(year, month, a_start, a_end)
        ranges[asset] = (first, last, first - timedelta(days=lookback))
        members.append(asset)
    cohort = sorted(members)
    params_for = {}
    stale = []
    for asset in members:
        first, last, warm = ranges[asset]
        digest = inputs_digest(
            bars_month_path(cfg.data_root, bar_type, a, y, m)
            for a in cohort for y, m in iter_months(warm, last)
        )
        params = partition_params(
            asset=asset, bar_type=bar_type, start=first, end=last,
            extra={"stage": "features", "market": "fx", "cohort": cohort, "inputs": digest},
        )
        params_for[asset] = params
        path = features_month_path(cfg.data_root, bar_type, asset, year, month)
        if partition_fresh(path, params, force=cfg.force):
            status[asset] = "cached"
        else:
            stale.append(asset)
    if not stale:
        return status
    frames: dict[str, pl.DataFrame] = {}
    for asset in members:
        first, last, warm = ranges[asset]
        bars = load_bar_months(cfg.data_root, asset, bar_type, warm, last)
        if not bars.is_empty():
            frames[asset] = compute_fx_features(bars, fx_pip(asset))
    if frames:
        frames = add_usd_strength(frames)
    for asset in stale:
        if asset not in frames:
            status[asset] = "empty"
            continue
        first, last, _warm = ranges[asset]
        month_df = _in_month(frames[asset], first, last)
        if month_df.is_empty():
            status[asset] = "empty"
            continue
        path = features_month_path(cfg.data_root, bar_type, asset, year, month)
        write_frame(month_df, path, "features_v1", params_for[asset])
        status[asset] = "written"
    return status
