"""The one orchestrator. Stages: fetch, bars, features, dataset, validate."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from pipeline.align import asof_join_knowable
from pipeline.bars import DollarAccumulator, tick_bars, volume_bars
from pipeline.bars import build_fx_ladder, build_time_ladder, gaps_ms, merge_partial_time, trades_to_time
from pipeline.constants import (
    ALL_BAR_TYPES,
    CORE_BAR_TYPES,
    DAY_MS,
    OI_LAG_MS,
    TIME_WIDTH_MS,
)
from pipeline.config import (
    fx_pip,
    Splits,
    Windows,
    dollar_threshold,
    effective_start,
    embargo_bars_for,
    liq_proxy_usd,
    load_embargo_policy,
    load_listing_dates,
    load_splits,
    load_windows,
    optional_threshold,
    scale_windows,
    universe_symbols,
    whale_large_usd,
)
from pipeline.dataset import (
    add_targets,
    add_universe_columns,
    apply_embargo,
    assign_split,
    manifest_payload,
    utc_now_ms,
)
from pipeline.features import FX_FEATURES, XD_COLUMNS, add_cross_asset, add_usd_strength, compute_fx_features
from pipeline.features import (
    basis_daily,
    funding_daily,
    metrics_daily,
    rows_to_frame,
    summarize_day,
)
from pipeline.features import compute_physics
from pipeline.fetch_external import fetch_all_external, load_external
from pipeline.fetch_crypto import (
    fetch_funding,
    fetch_klines_1m,
    fetch_metrics,
    fetch_perp_1d,
    fetch_trades,
    iter_days,
)
from pipeline.io import atomic_write_json, atomic_write_parquet, file_sha256, params_hash
from pipeline.paths import (
    bars_path,
    features_path,
    funding_path,
    gap_path,
    gold_manifest_path,
    gold_path,
    metrics_path,
    perp_1d_path,
    trades_path,
)
from pipeline.symbols import canonical_fx_symbol, canonical_symbol
from pipeline.validate import (
    gate_trade_count,
    gate_time_reconcile,
    validate_frame,
)


# Order is load-bearing. The CDAP dag check greps this exact literal.
STAGE_ORDER = ("fetch", "bars", "features", "dataset", "validate")


def print_coverage_report(rows: list[dict]) -> None:
    """One line per written gold file. The coverage invariant greps this."""
    for row in rows:
        print(
            f"COVERAGE -- asset={row.get('asset')} bar_type={row.get('bar_type')} "
            f"rows={row.get('rows')} status={row.get('status')}"
        )


@dataclass
class RunConfig:
    start: date
    end: date
    assets: list[str] = field(default_factory=list)
    universe: str | None = None
    stages: list[str] = field(default_factory=lambda: list(STAGE_ORDER))
    bar_types: list[str] = field(default_factory=lambda: list(CORE_BAR_TYPES))
    data_root: Path = field(default_factory=lambda: Path("data/pipeline"))
    force: bool = False
    dry_run: bool = False
    include_oi: bool = False
    voladj: bool = False
    clipped: bool = False
    windows: Windows | None = None
    splits: Splits | None = None
    fetch_timeout: int = 120
    market: str = "crypto"
    # Tests pass a small notional so a short fixture still closes bars.
    # Production leaves this empty and uses dollar_thresholds.yaml.
    threshold_overrides: dict[str, float] | None = None


def resolve_assets(cfg: RunConfig) -> list[str]:
    if cfg.market == "fx":
        if cfg.assets:
            return [canonical_fx_symbol(a) for a in cfg.assets]
        return universe_symbols(cfg.universe or "majors")
    if cfg.assets:
        return [canonical_symbol(a) for a in cfg.assets]
    if cfg.universe:
        return universe_symbols(cfg.universe)
    raise ValueError("pass assets or a universe")


def feature_columns(df: pl.DataFrame) -> list[str]:
    prefixes = ("norm_", "xd_", "wh_", "liq_", "rv_", "hbr_", "funding_day", "s3_", "bs_")
    cols = [c for c in df.columns if c.startswith(prefixes) or c == "hurst_regime"]
    cols += [c for c in FX_FEATURES if c in df.columns and c not in cols]
    return cols


def _read_funding(root: Path, asset: str, start: date, end: date) -> pl.DataFrame:
    frames = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        path = funding_path(root, asset, y, m)
        if path.exists():
            frames.append(pl.read_parquet(path))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    if not frames:
        return pl.DataFrame(schema={"ts_ms": pl.Int64, "funding_rate": pl.Float64})
    return pl.concat(frames, how="diagonal_relaxed").sort("ts_ms").unique(subset=["ts_ms"], keep="last", maintain_order=True)


def _read_metrics(root: Path, asset: str, start: date, end: date) -> pl.DataFrame:
    frames = []
    for day in iter_days(start, end):
        path = metrics_path(root, asset, day)
        if path.exists():
            frames.append(pl.read_parquet(path))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed").sort("ts_ms")


def _read_perp(root: Path, asset: str, start: date, end: date) -> pl.DataFrame:
    frames = []
    for day in iter_days(start, end):
        path = perp_1d_path(root, asset, day)
        if path.exists():
            frames.append(pl.read_parquet(path))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed").sort("ts_ms")


def _join_funding(bars: pl.DataFrame, funding: pl.DataFrame) -> pl.DataFrame:
    if funding.is_empty():
        return bars.with_columns(pl.lit(None, dtype=pl.Float64).alias("funding_rate"))
    return asof_join_knowable(bars, funding.select(["ts_ms", "funding_rate"]), ["funding_rate"], 0)


def _join_oi(bars: pl.DataFrame, metrics: pl.DataFrame) -> pl.DataFrame:
    if metrics.is_empty() or "sum_open_interest_value" not in metrics.columns:
        return bars
    feat = metrics.select([
        "ts_ms",
        pl.col("sum_open_interest_value").alias("oi_value"),
    ]).drop_nulls(["oi_value"])
    return asof_join_knowable(bars, feat, ["oi_value"], OI_LAG_MS)


def _join_daily(bars: pl.DataFrame, panel: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    if panel.is_empty() or not cols:
        return bars
    present = [c for c in cols if c in panel.columns]
    if not present:
        return bars
    feat = panel.select(["event_time_ms", *present])
    return asof_join_knowable(
        bars, feat, present, DAY_MS,
        feature_time_col="event_time_ms",
    )


def build_bars_for_asset(cfg: RunConfig, asset: str, start: date) -> dict:
    """Stream bronze days into time bars, dollar bars, and a daily panel."""
    windows = cfg.windows or load_windows()
    override = (cfg.threshold_overrides or {}).get(asset)
    threshold = float(override) if override is not None else dollar_threshold(asset)
    partials = []
    acc = DollarAccumulator(threshold)
    tick_chunks: list[pl.DataFrame] = []
    vol_chunks: list[pl.DataFrame] = []
    daily_rows = []
    n_bronze = 0
    want_tick = "tick" in cfg.bar_types
    want_vol = "volume" in cfg.bar_types
    for day in iter_days(start, cfg.end):
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
        if want_tick or want_vol:
            # Optional fabrics are small relative to a day; keep the day frame only
            # when the caller asked for them.
            if want_tick:
                tick_chunks.append(trades)
            if want_vol:
                vol_chunks.append(trades)
        daily_rows.append(summarize_day(
            trades, one_m, asset=asset, day=day,
            whale_usd=whale_large_usd(asset), liq_usd=liq_proxy_usd(asset),
        ))
        del trades

    bars_1m = merge_partial_time(partials, asset, "time_1m", TIME_WIDTH_MS["time_1m"])
    # Coarser clocks from the merged 1m bars. build_time_ladder would re-bin trades;
    # here the 1m frame is already the base, so aggregate from it.
    from pipeline.bars import bars_to_time
    ladder = {"time_1m": bars_1m}
    for name in cfg.bar_types:
        if name in TIME_WIDTH_MS and name != "time_1m":
            ladder[name] = bars_to_time(bars_1m, TIME_WIDTH_MS[name], asset, name)
    dollar, excluded = acc.finish(asset)
    out_bars: dict[str, pl.DataFrame] = {}
    for name in cfg.bar_types:
        if name in ladder:
            out_bars[name] = ladder[name]
    if "dollar" in cfg.bar_types:
        out_bars["dollar"] = dollar
    if want_tick and tick_chunks:
        ticks, _ = tick_bars(pl.concat(tick_chunks), int(optional_threshold(asset, "tick_trades_per_bar")), asset)
        out_bars["tick"] = ticks
    if want_vol and vol_chunks:
        vols, _ = volume_bars(pl.concat(vol_chunks), optional_threshold(asset, "volume_base_qty_per_bar"), asset)
        out_bars["volume"] = vols
    return {
        "bars": out_bars,
        "ladder": ladder,
        "daily": rows_to_frame(daily_rows),
        "n_bronze": n_bronze,
        "dollar_excluded": excluded,
        "gaps": gaps_ms(bars_1m),
        "windows": windows,
    }


def _attach_slow_features(cfg: RunConfig, asset: str, start: date, bars: dict[str, pl.DataFrame], daily: pl.DataFrame) -> dict[str, pl.DataFrame]:
    funding = _read_funding(cfg.data_root, asset, start, cfg.end)
    metrics = _read_metrics(cfg.data_root, asset, start, cfg.end) if cfg.include_oi else pl.DataFrame()
    fund_daily = funding_daily(funding, asset)
    met_daily = metrics_daily(metrics, asset) if cfg.include_oi else pl.DataFrame()
    perp = _read_perp(cfg.data_root, asset, start, cfg.end)
    spot_1d = bars.get("time_1d")
    basis = basis_daily(spot_1d, perp, asset) if spot_1d is not None else pl.DataFrame()
    windows = cfg.windows or load_windows()
    _floor_days, max_days = load_embargo_policy()
    out = {}
    daily_cols = [c for c in daily.columns if c not in ("asset", "event_time_ms")]
    fund_cols = [c for c in fund_daily.columns if c not in ("asset", "event_time_ms")]
    met_cols = [c for c in met_daily.columns if c not in ("asset", "event_time_ms")]
    basis_cols = [c for c in basis.columns if c not in ("asset", "event_time_ms")]
    stables = load_external(cfg.data_root, "defillama")
    dvol = load_external(cfg.data_root, "deribit")
    etf = load_external(cfg.data_root, "farside")
    wiki = load_external(cfg.data_root, "wikipedia")
    btc_dvol = pl.DataFrame()
    if not dvol.is_empty() and "asset" in dvol.columns and "dv_dvol_close" in dvol.columns:
        btc_dvol = (
            dvol.filter(pl.col("asset") == "BTCUSDT")
            .select(["event_time_ms", pl.col("dv_dvol_close").alias("dv_btc_dvol_close")])
        )
    stable_cols = [c for c in stables.columns if c != "event_time_ms"]
    dvol_cols = [c for c in ("dv_dvol_open", "dv_dvol_high", "dv_dvol_low", "dv_dvol_close") if c in dvol.columns]
    etf_cols = [c for c in etf.columns if c.startswith("etf_")]
    wiki_cols = [c for c in wiki.columns if c.startswith("soc_")]
    for bar_type, frame in bars.items():
        df = _join_funding(frame, funding)
        if cfg.include_oi:
            df = _join_oi(df, metrics)
        diffs = frame["ts_ms"].diff().drop_nulls() if frame.height > 2 else None
        median = int(diffs.median()) if diffs is not None and diffs.len() else None
        scaled = scale_windows(windows, bar_type, median, max_lookback_days=max_days)
        df = compute_physics(df, scaled, include_oi=cfg.include_oi)
        df = _join_daily(df, daily, daily_cols)
        df = _join_daily(df, fund_daily, fund_cols)
        df = _join_daily(df, met_daily, met_cols)
        df = _join_daily(df, basis, basis_cols)
        df = _join_daily(df, stables, stable_cols)
        if dvol_cols and not dvol.is_empty():
            own = dvol.filter(pl.col("asset") == asset)
            df = _join_daily(df, own, dvol_cols)
        df = _join_daily(df, btc_dvol, ["dv_btc_dvol_close"] if "dv_btc_dvol_close" in btc_dvol.columns else [])
        if etf_cols and not etf.is_empty():
            own_etf = etf.filter(pl.col("asset") == asset) if "asset" in etf.columns else etf
            df = _join_daily(df, own_etf, etf_cols)
        if wiki_cols and not wiki.is_empty():
            own_wiki = wiki.filter(pl.col("asset") == asset)
            df = _join_daily(df, own_wiki, wiki_cols)
        out[bar_type] = df
    return out


def _write_frame(df: pl.DataFrame, path: Path, schema_name: str, params: dict) -> str:
    atomic_write_parquet(
        df, path, schema_name=schema_name, params=params,
        sort_cols=["ts_ms"],
        required_cols=["ts_ms", "asset", "bar_type"],
    )
    return file_sha256(path)


def run(cfg: RunConfig) -> int:
    if cfg.market == "fx":
        return run_fx(cfg)
    assets = resolve_assets(cfg)
    listings = load_listing_dates()
    windows = cfg.windows or load_windows()
    splits = cfg.splits or load_splits()
    cfg.windows = windows
    cfg.splits = splits
    unknown = [b for b in cfg.bar_types if b not in ALL_BAR_TYPES]
    if unknown:
        raise ValueError(f"unknown bar types {unknown}; omitted research fabrics are not built")

    plan = []
    for asset in assets:
        start = effective_start(asset, cfg.start, listings)
        plan.append((asset, start))
        if start > cfg.end:
            print(f"[plan] {asset} listing/start {start} is after end {cfg.end}; skip")

    if cfg.dry_run:
        print(f"[dry-run] stages={','.join(cfg.stages)} bar_types={','.join(cfg.bar_types)} root={cfg.data_root}")
        for asset, start in plan:
            print(f"[dry-run] {asset} effective_start={start.isoformat()} end={cfg.end.isoformat()} include_oi={cfg.include_oi}")
        return 0

    worst = 0
    # Pass 1: fetch, bars, per-asset features (no cross-asset yet).
    built: dict[str, dict[str, pl.DataFrame]] = {}
    meta: dict[str, dict] = {}
    external_done = False
    for asset, start in plan:
        if start > cfg.end:
            continue
        try:
            if "fetch" in cfg.stages:
                print(f"[fetch] {asset} {start} -> {cfg.end}")
                for fn in (fetch_trades, fetch_klines_1m, fetch_funding, fetch_metrics, fetch_perp_1d):
                    report = fn(cfg.data_root, asset, start, cfg.end, force=cfg.force, timeout=cfg.fetch_timeout)
                    print(f"[fetch] {report.source} ok={report.n_ok} error={report.n_error}")
                if not external_done:
                    fetch_all_external(
                        cfg.data_root, [a for a, s in plan if s <= cfg.end],
                        cfg.start, cfg.end, timeout=cfg.fetch_timeout,
                    )
                    external_done = True
            if not any(s in cfg.stages for s in ("bars", "features", "dataset", "validate")):
                continue
            print(f"[bars] {asset}")
            packed = build_bars_for_asset(cfg, asset, start)
            if packed["n_bronze"] == 0:
                print(f"[bars] {asset} has no bronze trades in range")
                worst = max(worst, 1)
                continue
            meta[asset] = packed
            if "bars" in cfg.stages or "features" in cfg.stages or "dataset" in cfg.stages:
                ts_max = max(int(df["ts_ms"].max()) for df in packed["bars"].values() if df.height)
                for bar_type, df in packed["bars"].items():
                    if df.is_empty():
                        continue
                    params = {"asset": asset, "bar_type": bar_type, "start": start.isoformat(), "end": cfg.end.isoformat()}
                    path = bars_path(cfg.data_root, bar_type, asset, int(df["ts_ms"].max()))
                    _write_frame(df, path, "bars_v1", params)
                    print(f"[bars] wrote {path} rows={df.height}")
                atomic_write_json(gap_path(cfg.data_root, asset, ts_max), {"missing_ts_ms": packed["gaps"][:10000], "n_gaps": len(packed["gaps"])})
            featured = _attach_slow_features(cfg, asset, start, packed["bars"], packed["daily"])
            built[asset] = featured
        except Exception as exc:
            print(f"[error] {asset}: {type(exc).__name__}: {exc}")
            worst = 2

    if "features" not in cfg.stages and "dataset" not in cfg.stages and "validate" not in cfg.stages:
        return worst

    # Pass 2: cross-asset, then dataset + validate per bar type.
    by_type: dict[str, dict[str, pl.DataFrame]] = {bt: {} for bt in cfg.bar_types}
    for asset, frames in built.items():
        for bt, df in frames.items():
            by_type.setdefault(bt, {})[asset] = df

    reports = []
    coverage: list[dict] = []
    for bt, frames in by_type.items():
        if not frames:
            continue
        frames = add_cross_asset(frames, bar_type=bt, fast=windows.fast, ma=min(windows.ma, max(windows.fast, 2)))
        for asset, df in frames.items():
            if df.is_empty():
                continue
            df = add_targets(df, windows.horizons, clipped=cfg.clipped, voladj=cfg.voladj)
            df = assign_split(df, splits)
            diffs = df["ts_ms"].diff().drop_nulls() if df.height > 2 else None
            median = int(diffs.median()) if diffs is not None and diffs.len() else None
            floor_days, max_days = load_embargo_policy()
            embargo = embargo_bars_for(
                bt, windows, median,
                calendar_floor_days=floor_days, max_lookback_days=max_days,
            )
            df = apply_embargo(df, embargo)
            listing = load_listing_dates().get(asset)
            df = add_universe_columns(df, asset, listing)
            params = {
                "asset": asset,
                "bar_type": bt,
                "start": cfg.start.isoformat(),
                "end": cfg.end.isoformat(),
                "include_oi": cfg.include_oi,
                "voladj": cfg.voladj,
                "clipped": cfg.clipped,
                "windows": windows.__dict__ if hasattr(windows, "__dict__") else {},
                "embargo_bars": embargo,
                "splits": {
                    "train_end": splits.train_end.isoformat(),
                    "val_end": splits.val_end.isoformat(),
                    "oos_end": splits.oos_end.isoformat(),
                    "unseen_start": splits.unseen_start.isoformat(),
                },
            }
            # Windows is a dataclass; __dict__ exists.
            params["windows"] = {
                "z": windows.z, "fast": windows.fast, "frac_diff": windows.frac_diff,
                "hurst": windows.hurst, "ma": windows.ma, "horizons": list(windows.horizons),
            }
            ph = params_hash(params)
            params["params_hash"] = ph
            ts_max = int(df["ts_ms"].max())
            if "features" in cfg.stages or "dataset" in cfg.stages:
                fpath = features_path(cfg.data_root, bt, asset, ts_max)
                _write_frame(df, fpath, "features_v1", params)
            if "dataset" in cfg.stages or "validate" in cfg.stages:
                gpath = gold_path(cfg.data_root, bt, asset, ts_max)
                digest = _write_frame(df, gpath, "dataset_v1", params)
                feats = feature_columns(df)
                man = manifest_payload(
                    asset=asset, bar_type=bt, df=df, feature_cols=feats, params=params,
                    embargo_bars=embargo, inputs=[], lags={
                        "funding": 0, "daily_panel": DAY_MS, "open_interest": OI_LAG_MS if cfg.include_oi else None,
                    },
                    built_at_ms=utc_now_ms(),
                )
                man["sha256"] = digest
                man["dollar_open_tail_trades"] = meta.get(asset, {}).get("dollar_excluded")
                man["n_bronze_trades"] = meta.get(asset, {}).get("n_bronze")
                atomic_write_json(gold_manifest_path(gpath), man)
                print(f"[dataset] {gpath} rows={df.height} features={len(feats)} sha={digest[:12]}")
            if "validate" in cfg.stages:
                packed = meta.get(asset, {})
                report = validate_frame(
                    df, feature_cols=feature_columns(df), embargo_bars=embargo,
                    n_assets_in_run=len(built), asset=asset,
                    missing_minutes=packed.get("gaps") if bt == "time_1m" else None,
                    min_history=windows.max_lookback,
                )
                if bt == "time_1m" and packed.get("ladder"):
                    report.results.append(gate_time_reconcile(packed["ladder"]))
                    report.results.append(gate_trade_count(
                        int(packed["ladder"]["time_1m"]["n_trades"].sum()),
                        int(packed["n_bronze"]),
                        what="time_1m",
                    ))
                if bt == "dollar":
                    report.results.append(gate_trade_count(
                        int(df["n_trades"].sum()) if df.height else 0,
                        int(packed.get("n_bronze", 0)),
                        excluded_tail=int(packed.get("dollar_excluded", 0)),
                        what="dollar",
                    ))
                reports.append((asset, bt, report))
                print(f"[validate] {asset} {bt} exit={report.code}")
                for item in report.results:
                    if item.severity != "pass":
                        print(f"  {item.severity}: {item.name}: {item.detail}")
                worst = max(worst, report.code)
                coverage.append({
                    "asset": asset, "bar_type": bt, "rows": df.height,
                    "status": "pass" if report.code == 0 else ("warn" if report.code == 1 else "fail"),
                })
    if coverage:
        print_coverage_report(coverage)
    return worst


def _fx_weekday_gaps(missing: list[int]) -> list[int]:
    """The Friday-Sunday close is the FX calendar, not a missing-data hole."""
    kept = []
    for ts in missing:
        dt = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
        if dt.weekday() == 5:
            continue
        if dt.weekday() == 6 and dt.hour < 22:
            continue
        if dt.weekday() == 4 and dt.hour >= 21:
            continue
        kept.append(ts)
    return kept


def run_fx(cfg: RunConfig) -> int:
    """FX uses the same stages. Fetch is Dukascopy hours; bars and features are shared code."""
    from pipeline.fetch_fx import fetch_fx_range, read_fx_ticks

    assets = resolve_assets(cfg)
    windows = cfg.windows or load_windows()
    splits = cfg.splits or load_splits()
    cfg.windows = windows
    cfg.splits = splits
    bad = [b for b in cfg.bar_types if b not in TIME_WIDTH_MS]
    if bad:
        raise ValueError(f"FX builds time bars 1m through 1d; not {bad}")
    if cfg.dry_run:
        print(f"[dry-run] market=fx stages={','.join(cfg.stages)} bar_types={','.join(cfg.bar_types)} root={cfg.data_root}")
        for asset in assets:
            start = effective_start(asset, cfg.start)
            print(f"[dry-run] {asset} effective_start={start.isoformat()} end={cfg.end.isoformat()}")
        return 0

    worst = 0
    ladders: dict[str, dict[str, pl.DataFrame]] = {}
    tick_counts: dict[str, int] = {}
    for asset in assets:
        start = effective_start(asset, cfg.start)
        if start > cfg.end:
            print(f"[plan] {asset} start {start} is after end {cfg.end}; skip")
            continue
        if "fetch" in cfg.stages:
            print(f"[fetch] {asset} {start} -> {cfg.end}")
            n = fetch_fx_range(
                cfg.data_root, asset, start, cfg.end,
                force=cfg.force, timeout=cfg.fetch_timeout,
            )
            print(f"[fetch] dukascopy hours={n}")
        ticks = read_fx_ticks(cfg.data_root, asset, start, cfg.end)
        tick_counts[asset] = ticks.height
        ladder = build_fx_ladder(ticks, asset, cfg.bar_types)
        ladders[asset] = ladder
        if "bars" in cfg.stages:
            for bt, frame in ladder.items():
                if frame.is_empty():
                    print(f"[bars] {asset} {bt} empty")
                    continue
                path = bars_path(cfg.data_root, bt, asset, int(frame["ts_ms"].max()))
                atomic_write_parquet(
                    frame, path, schema_name="bars_v1",
                    params={"asset": asset, "bar_type": bt, "market": "fx"},
                )
                print(f"[bars] {asset} {bt} rows={frame.height}")

    featured: dict[str, dict[str, pl.DataFrame]] = {a: {} for a in ladders}
    for bt in cfg.bar_types:
        cohort = {}
        for asset, ladder in ladders.items():
            frame = ladder.get(bt)
            if frame is None or frame.is_empty():
                continue
            cohort[asset] = compute_fx_features(frame, fx_pip(asset))
        cohort = add_usd_strength(cohort)
        for asset, frame in cohort.items():
            featured[asset][bt] = frame
            if "features" in cfg.stages:
                path = features_path(cfg.data_root, bt, asset, int(frame["ts_ms"].max()))
                atomic_write_parquet(
                    frame, path, schema_name="features_v1",
                    params={"asset": asset, "bar_type": bt, "market": "fx"},
                )

    floor_days, cap_days = load_embargo_policy()
    coverage = []
    for asset, by_type in featured.items():
        for bt, df in by_type.items():
            scaled = scale_windows(windows, bt, max_lookback_days=cap_days)
            embargo = embargo_bars_for(
                bt, windows, calendar_floor_days=floor_days, max_lookback_days=cap_days,
            )
            df = add_targets(df, scaled.horizons, clipped=cfg.clipped, voladj=cfg.voladj)
            df = add_universe_columns(df, asset, None)
            df = assign_split(df, splits)
            df = apply_embargo(df, embargo)
            if "dataset" in cfg.stages:
                gpath = gold_path(cfg.data_root, bt, asset, int(df["ts_ms"].max()))
                digest = _write_frame(
                    df, gpath, "dataset_v1",
                    {"asset": asset, "bar_type": bt, "market": "fx", "embargo_bars": embargo},
                )
                feats = feature_columns(df)
                man = manifest_payload(
                    asset=asset, bar_type=bt, df=df, feature_cols=feats,
                    params={"market": "fx", "pip": fx_pip(asset)},
                    embargo_bars=embargo, inputs=[],
                    lags={"ticks": 0}, built_at_ms=utc_now_ms(),
                )
                man["sha256"] = digest
                atomic_write_json(gold_manifest_path(gpath), man)
                print(f"[dataset] {gpath} rows={df.height} features={len(feats)}")
            if "validate" in cfg.stages:
                gaps = None
                if bt == "time_1m" and "time_1m" in ladders[asset]:
                    gaps = _fx_weekday_gaps(gaps_ms(ladders[asset]["time_1m"]))
                report = validate_frame(
                    df, feature_cols=feature_columns(df), embargo_bars=embargo,
                    n_assets_in_run=len(ladders), asset=asset,
                    missing_minutes=gaps, min_history=max(scaled.max_lookback, 100),
                )
                if bt == "time_1m":
                    report.results.append(gate_time_reconcile(ladders[asset]))
                    report.results.append(gate_trade_count(
                        int(ladders[asset]["time_1m"]["n_trades"].sum()),
                        int(tick_counts[asset]),
                        what="fx_ticks",
                    ))
                print(f"[validate] {asset} {bt} exit={report.code}")
                for item in report.results:
                    if item.severity != "pass":
                        print(f"  {item.severity}: {item.name}: {item.detail}")
                worst = max(worst, report.code)
                coverage.append({
                    "asset": asset, "bar_type": bt, "rows": df.height,
                    "status": "pass" if report.code == 0 else ("warn" if report.code == 1 else "fail"),
                })
    if coverage:
        print_coverage_report(coverage)
    return worst


def status(root: Path) -> int:
    gold = root / "gold"
    if not gold.exists():
        print(f"[status] no gold under {gold}")
        return 0
    files = sorted(gold.rglob("*.parquet"))
    print(f"[status] {len(files)} gold files under {gold}")
    for path in files:
        print(f"  {path.relative_to(root)} bytes={path.stat().st_size}")
    return 0
