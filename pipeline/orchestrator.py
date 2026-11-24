"""The one orchestrator. Stages: fetch, bars, features, dataset, validate."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

import polars as pl

from pipeline.align import asof_join_knowable
from pipeline.bars import DollarAccumulator, tick_bars, volume_bars
from pipeline.bars import build_fx_ladder, build_time_ladder, gaps_ms, merge_partial_time, trades_to_time
from pipeline.constants import (
    ALL_BAR_TYPES,
    CORE_BAR_TYPES,
    DAY_MS,
    TIME_WIDTH_MS,
)
from pipeline.config import (
    fx_pip,
    Splits,
    Windows,
    effective_start,
    embargo_bars_for,
    lag_ms,
    liq_proxy_usd,
    load_embargo_policy,
    load_listing_dates,
    load_splits,
    load_windows,
    optional_threshold,
    scale_windows,
    universe_symbols,
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
from pipeline.calendar import clip_month, iter_days, iter_months
from pipeline.chunked import (
    load_bar_months,
    plan_dollar,
    load_feature_months,
    write_bars_month,
    write_dataset_month,
    write_features_month,
    write_fx_bars_month,
)
from pipeline.fetch_crypto import (
    fetch_funding,
    fetch_klines_1m,
    fetch_metrics,
    fetch_perp_1d,
    fetch_trades,
)
from pipeline.io import atomic_write_json, atomic_write_parquet, file_sha256, params_hash
from pipeline.partitions import partition_fresh, sidecar_ok
from pipeline.paths import (
    bars_month_path,
    bars_path,
    features_month_path,
    features_path,
    funding_path,
    gap_path,
    gold_manifest_path,
    gold_month_path,
    gold_path,
    metrics_path,
    perp_1d_path,
    trades_path,
)
from pipeline.progress import RunSession
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
    data_root: Path = field(default_factory=lambda: Path("data"))
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
    skip_network: bool = False


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
    return asof_join_knowable(bars, feat, ["oi_value"], lag_ms("open_interest"))


def _join_daily(bars: pl.DataFrame, panel: pl.DataFrame, cols: list[str], family: str) -> pl.DataFrame:
    """As-of join of a daily series. Its lag comes from ``lags.families[family]``."""
    if panel.is_empty() or not cols:
        return bars
    present = [c for c in cols if c in panel.columns]
    if not present:
        return bars
    feat = panel.select(["event_time_ms", *present])
    return asof_join_knowable(
        bars, feat, present, lag_ms(family),
        feature_time_col="event_time_ms",
    )


def _attach_slow_features(cfg: RunConfig, asset: str, start: date, bars: dict[str, pl.DataFrame], daily: pl.DataFrame) -> dict[str, pl.DataFrame]:
    # Read only as far as the frame goes. Anything later could never be knowable
    # for these bars, and reading to cfg.end made every partition scan the whole run.
    last_ts = max(int(f["ts_ms"].max()) for f in bars.values() if not f.is_empty())
    read_end = min(cfg.end, datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc).date())
    funding = _read_funding(cfg.data_root, asset, start, read_end)
    # The daily metrics panel (s3_*) is a slow, day-lagged series and is always built
    # when Vision shipped metrics. Only the 5-minute OI feature is opt-in (--include-oi).
    metrics = _read_metrics(cfg.data_root, asset, start, read_end)
    fund_daily = funding_daily(funding, asset)
    met_daily = metrics_daily(metrics, asset)
    perp = _read_perp(cfg.data_root, asset, start, read_end)
    # Basis needs true daily spot closes whatever cadence is being featured, so the
    # 1d silver bars are read here when the frame being built is not the 1d one.
    spot_1d = bars.get("time_1d")
    if spot_1d is None:
        loaded = load_bar_months(cfg.data_root, asset, "time_1d", start, read_end)
        spot_1d = loaded if not loaded.is_empty() else None
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
        df = _join_daily(df, daily, daily_cols, "daily_panel")
        df = _join_daily(df, fund_daily, fund_cols, "funding_daily")
        df = _join_daily(df, met_daily, met_cols, "metrics_daily")
        df = _join_daily(df, basis, basis_cols, "basis")
        df = _join_daily(df, stables, stable_cols, "defillama")
        if dvol_cols and not dvol.is_empty():
            own = dvol.filter(pl.col("asset") == asset)
            df = _join_daily(df, own, dvol_cols, "deribit")
        df = _join_daily(df, btc_dvol, ["dv_btc_dvol_close"] if "dv_btc_dvol_close" in btc_dvol.columns else [], "deribit")
        if etf_cols and not etf.is_empty():
            own_etf = etf.filter(pl.col("asset") == asset) if "asset" in etf.columns else etf
            df = _join_daily(df, own_etf, etf_cols, "etf")
        if wiki_cols and not wiki.is_empty():
            own_wiki = wiki.filter(pl.col("asset") == asset)
            df = _join_daily(df, own_wiki, wiki_cols, "wikipedia")
        out[bar_type] = df
    return out


def _write_frame(df: pl.DataFrame, path: Path, schema_name: str, params: dict) -> str:
    atomic_write_parquet(
        df, path, schema_name=schema_name, params=params,
        sort_cols=["ts_ms"],
        required_cols=["ts_ms", "asset", "bar_type"],
    )
    return file_sha256(path)


CHUNKED_VALIDATE = ("time_1m", "time_5m", "dollar")


def _year_chunks(lo: date, hi: date) -> list[tuple[date, date]]:
    out = []
    for y in range(lo.year, hi.year + 1):
        out.append((max(lo, date(y, 1, 1)), min(hi, date(y, 12, 31))))
    return out


def _load_gold_range(root: Path, bar_type: str, asset: str, lo: date, hi: date, columns: list[str] | None = None) -> pl.DataFrame:
    """Gold partitions of ``[lo, hi]`` only, clipped to bars closing in that range."""
    frames = []
    for y, m in iter_months(lo, hi):
        path = gold_month_path(root, bar_type, asset, y, m)
        if sidecar_ok(path):
            df = pl.read_parquet(path, columns=columns) if columns else pl.read_parquet(path)
            if df.height:
                frames.append(df)
    if not frames:
        return pl.DataFrame()
    lo_ms = int(datetime(lo.year, lo.month, lo.day, tzinfo=timezone.utc).timestamp() * 1000)
    hi_ms = int(datetime(hi.year, hi.month, hi.day, tzinfo=timezone.utc).timestamp() * 1000) + DAY_MS
    out = pl.concat(frames, how="diagonal_relaxed").sort("ts_ms")
    return out.filter((pl.col("ts_ms") > lo_ms) & (pl.col("ts_ms") <= hi_ms))


def _merge_gate(results: dict, item, label: str) -> None:
    rank = {"pass": 0, "warn": 1, "fail": 2}
    prev = results.get(item.name)
    if prev is None or rank[item.severity] > rank[prev[0]]:
        results[item.name] = (item.severity, f"[{label}] {item.detail}")


CAUSALITY_CUTS = 2


def _causality_gate(cfg: "RunConfig", asset: str, bar_type: str, chunk_end: date) -> GateResult | None:
    """Truncation proof on the last feature month of a chunk.

    For a few bars of that month, the features are recomputed from only the
    data knowable at the bar's close (bars up to it, daily panel rows whose lag
    has passed, funding/metrics/perp read to that day) on the same warm frame
    the features stage used. Every stored column must match. A correlation gate
    can only suspect a leak; this proves its absence for the sampled bars.
    External series are covered by the as-of control check instead.
    """
    import dataclasses
    import math
    from datetime import timedelta

    from pipeline.chunked import _asset_range, feature_lookback_days, load_daily_months
    from pipeline.validate import GateResult

    y, m = chunk_end.year, chunk_end.month
    try:
        first, last = clip_month(y, m, *_asset_range(cfg, asset))
    except ValueError:
        return None
    path = features_month_path(cfg.data_root, bar_type, asset, y, m)
    if not path.exists():
        return None
    stored = pl.read_parquet(path).sort("ts_ms")
    if stored.height < 2:
        return None
    warm_start = first - timedelta(days=feature_lookback_days(cfg, bar_type))
    bars = load_bar_months(cfg.data_root, asset, bar_type, warm_start, last).sort("ts_ms")
    daily = load_daily_months(cfg.data_root, asset, warm_start, last)
    lag = lag_ms("daily_panel")
    picks = sorted({stored.height // 2, stored.height - 1})[:CAUSALITY_CUTS]
    bad: dict[str, float] = {}
    cols = [c for c in stored.columns if c != "ts_ms"]
    for i in picks:
        cut_ts = int(stored["ts_ms"][i])
        cut_day = datetime.fromtimestamp(cut_ts / 1000, tz=timezone.utc).date()
        sub = dataclasses.replace(cfg, end=cut_day)
        b = bars.filter(pl.col("ts_ms") <= cut_ts)
        d = daily.filter(pl.col("event_time_ms") + lag <= cut_ts) if "event_time_ms" in daily.columns else daily
        out = _attach_slow_features(sub, asset, warm_start, {bar_type: b}, d)[bar_type]
        row = out.filter(pl.col("ts_ms") == cut_ts)
        if row.height != 1:
            bad["<row>"] = float("inf")
            continue
        want, got = stored.row(i, named=True), row.row(0, named=True)
        for c in cols:
            # A column the truncated frame lacks (e.g. no daily row knowable yet)
            # is null there; a stored non-null value then fails.
            a, v = want.get(c), got.get(c)
            if isinstance(a, float) and isinstance(v, float):
                if math.isnan(a) and math.isnan(v):
                    continue
                if abs(a - v) <= 1e-9 * max(1.0, abs(a)):
                    continue
                bad[c] = max(bad.get(c, 0.0), abs(a - v))
            elif a != v:
                bad[c] = float("inf")
    month = f"{y:04d}-{m:02d}"
    if bad:
        worst = sorted(bad.items(), key=lambda kv: -kv[1])[:6]
        return GateResult("causality", "fail",
                          f"{len(bad)} column(s) change when the future is cut off at {len(picks)} bar(s) of {month}: {worst}")
    return GateResult("causality", "pass",
                      f"{len(cols)} columns at {len(picks)} bar(s) of {month} match a recompute from data up to the bar")


def _validate_asset(cfg: "RunConfig", asset: str, start: date, n_assets: int, packed: dict, session) -> int:
    """In-pipeline gates for one asset over the run's range, in bounded chunks.

    Fine fabrics (1m, 5m, dollar) are validated one calendar year at a time;
    coarser ones in one piece (they are small and their 90-day embargoes can
    straddle a year end). Bronze counts and gaps are recomputed here when the
    bars stage did not run in this invocation.
    """
    from pipeline.chunked import _embargo_padding, count_bronze_trades, count_fx_ticks
    from pipeline.validate_run import _fx_weekend_expected

    windows = cfg.windows or load_windows()
    floor_days, max_days = load_embargo_policy()
    worst = 0
    rows_seen: dict[str, int] = {}
    for bt in cfg.bar_types:
        embargo, _pad = _embargo_padding(cfg, asset, bt, windows, floor_days, max_days)
        scaled = scale_windows(windows, bt, None, max_lookback_days=max_days)
        chunks = _year_chunks(start, cfg.end) if bt in CHUNKED_VALIDATE else [(start, cfg.end)]
        results: dict = {}
        n_rows = 0
        for c0, c1 in chunks:
            df = _load_gold_range(cfg.data_root, bt, asset, c0, c1)
            if df.is_empty():
                continue
            n_rows += df.height
            missing = None
            if bt == "time_1m":
                one_m = load_bar_months(cfg.data_root, asset, "time_1m", c0, c1)
                missing = gaps_ms(one_m) if not one_m.is_empty() else []
                if cfg.market == "fx":
                    missing = [t for t in missing if not _fx_weekend_expected(t)]
                elif missing:
                    # Exchange downtime is not a hole: keep minutes the venue traded, and
                    # minutes on days with no venue klines (unknown counts as lost).
                    from pipeline.validate_run import classify_missing
                    split = classify_missing(cfg.data_root, asset, missing, c0, c1)
                    if split is not None:
                        missing = split[0]
            report = validate_frame(
                df, feature_cols=feature_columns(df), embargo_bars=embargo,
                n_assets_in_run=n_assets, asset=asset,
                missing_minutes=missing, min_history=max(scaled.max_lookback, 100),
            )
            label = f"{c0}..{c1}"
            for item in report.results:
                _merge_gate(results, item, label)
            if cfg.market != "fx":
                proof = _causality_gate(cfg, asset, bt, c1)
                if proof is not None:
                    _merge_gate(results, proof, label)
            if bt == "time_1m":
                ladder = {}
                for name in cfg.bar_types:
                    if name in TIME_WIDTH_MS:
                        frame = load_bar_months(cfg.data_root, asset, name, c0, c1)
                        if not frame.is_empty():
                            ladder[name] = frame
                if "time_1m" in ladder:
                    _merge_gate(results, gate_time_reconcile(ladder), label)
                    if cfg.market == "fx":
                        n_src, what = count_fx_ticks(cfg.data_root, asset, c0, c1), "fx_ticks"
                    else:
                        n_src, what = count_bronze_trades(cfg.data_root, asset, c0, c1), "time_1m"
                    _merge_gate(results, gate_trade_count(int(ladder["time_1m"]["n_trades"].sum()), n_src, what=what), label)
        if not n_rows:
            session.info(f"[validate] {asset} {bt} no gold")
            continue
        rows_seen[bt] = n_rows
        if bt == "dollar":
            # Dollar bars carry state across months: count over the whole range.
            d = _load_gold_range(cfg.data_root, "dollar", asset, start, cfg.end, columns=["ts_ms", "n_trades"])
            n_bronze = count_bronze_trades(cfg.data_root, asset, start, cfg.end)
            if packed.get("dollar_excluded") is None:
                # The bars stage did not run in this call: the open tail is in the
                # snapshot taken after the last month.
                from pipeline.chunked import dollar_spec_for, read_dollar_state
                state = read_dollar_state(
                    cfg.data_root, asset, dollar_spec_for(cfg, asset),
                    through=f"{cfg.end.year:04d}-{cfg.end.month:02d}",
                )
                packed["dollar_excluded"] = state.tail_trades() if state is not None else 0
            _merge_gate(results, gate_trade_count(
                int(d["n_trades"].sum()) if d.height else 0, n_bronze,
                excluded_tail=int(packed.get("dollar_excluded") or 0), what="dollar",
            ), f"{start}..{cfg.end}")
        code = max(({"pass": 0, "warn": 1, "fail": 2}[sev] for sev, _ in results.values()), default=0)
        session.info(f"[validate] {asset} {bt} rows={n_rows} exit={code}")
        for name, (sev, detail) in results.items():
            if sev != "pass":
                session.info(f"  {sev}: {name}: {detail}")
        worst = max(worst, code)
        packed.setdefault("_coverage", []).append({
            "asset": asset, "bar_type": bt, "rows": n_rows,
            "status": "pass" if code == 0 else ("warn" if code == 1 else "fail"),
        })
    return worst



def _independent_validate(cfg: RunConfig, worst: int) -> int:
    """Run scripts-compatible independent checks and print one PASS/WARN/FAIL summary."""
    from pipeline.validate_run import Report, run_validation

    report = Report()
    if worst == 0:
        report.add("in_pipeline", "PASS", "cadence gates passed")
    elif worst == 1:
        report.add("in_pipeline", "WARN", "cadence gates reported warnings; see the run log")
    else:
        report.add("in_pipeline", "FAIL", "cadence gates failed; see the run log")
    report = run_validation(
        cfg.data_root,
        start=cfg.start,
        end=cfg.end,
        assets=list(resolve_assets(cfg)),
        market=cfg.market,
        skip_network=bool(cfg.skip_network),
        report=report,
        z_window=(cfg.windows.z if cfg.windows is not None else None),
        threshold_overrides=cfg.threshold_overrides,
    )
    print()
    print("=== data layer validation ===")
    report.print()
    return max(worst, report.code)


def run(cfg: RunConfig) -> int:
    if cfg.market == "fx":
        return run_fx(cfg)
    assets = resolve_assets(cfg)
    listings = load_listing_dates()
    windows = cfg.windows or load_windows()
    splits = cfg.splits or load_splits()
    cfg.windows = windows
    cfg.splits = splits
    unknown = [b for b in cfg.bar_types if b not in CORE_BAR_TYPES]
    if unknown:
        raise ValueError(
            f"bar types {unknown} are not built by this pipeline; "
            f"use any of {', '.join(CORE_BAR_TYPES)}"
        )

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
    meta: dict[str, dict] = {}
    session = RunSession(
        market="crypto", start=cfg.start, end=cfg.end, stages=cfg.stages,
    )
    active = [(a, s) for a, s in plan if s <= cfg.end]
    try:
        if "fetch" in cfg.stages:
            external_done = False
            for asset, start in active:
                try:
                    session.info(f"[fetch] {asset} {start} -> {cfg.end}")
                    days = list(iter_days(start, cfg.end))
                    with session.bar(f"fetch {asset}", len(days) * 5, "file") as bar:
                        def _prog(kind, _bar=bar):
                            _bar.update(1, cached=(kind == "cached"), written=(kind == "written"), failed=(kind == "failed"))
                        for fn in (fetch_trades, fetch_klines_1m, fetch_funding, fetch_metrics, fetch_perp_1d):
                            report = fn(
                                cfg.data_root, asset, start, cfg.end,
                                force=cfg.force, timeout=cfg.fetch_timeout,
                                **({"progress": _prog} if fn in (fetch_trades, fetch_klines_1m) else {}),
                            )
                            # funding/metrics/perp have no progress hook; count here
                            if fn not in (fetch_trades, fetch_klines_1m):
                                bar.update(max(report.n_ok, 1), cached=report.n_cached, written=max(0, report.n_ok - report.n_cached), failed=report.n_error)
                            session.info(
                                f"[fetch] {asset} {report.source} ok={report.n_ok} cached={report.n_cached} error={report.n_error}"
                            )
                            if report.n_error:
                                worst = max(worst, 1)
                    if not external_done:
                        fetch_all_external(
                            cfg.data_root, [a for a, s in active],
                            cfg.start, cfg.end, timeout=cfg.fetch_timeout,
                        )
                        external_done = True
                    session.asset_summary(f"[fetch] {asset} done")
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    session.error(asset, "fetch", exc)
                    worst = 2

        if "bars" in cfg.stages:
            for asset, start in active:
                try:
                    months = list(iter_months(start, cfg.end))
                    # Months below f keep their dollar partitions; the rest are rebuilt in
                    # order from `acc` (see chunked.plan_dollar). Time bars are independent.
                    f, acc = plan_dollar(cfg, asset, months, start)
                    n_bronze = 0
                    gaps = []
                    excluded = 0
                    with session.bar(f"bars {asset}", len(months), "month") as bar:
                        for i, (year, month) in enumerate(months):
                            packed = write_bars_month(
                                cfg, asset, year, month, acc, session,
                                start=start, rebuild_dollar=(i >= f),
                            )
                            n_bronze += int(packed.get("n_bronze") or 0)
                            if packed.get("dollar_excluded") is not None:
                                excluded = int(packed.get("dollar_excluded") or 0)
                            if packed.get("cached"):
                                bar.update(1, cached=True)
                            else:
                                bar.update(1, written=True)
                                gaps.extend(packed.get("gaps") or [])
                    meta[asset] = {"n_bronze": n_bronze, "dollar_excluded": excluded, "gaps": gaps}
                    session.asset_summary(f"[bars] {asset} months={len(months)} bronze_trades={n_bronze}")
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    session.error(asset, "bars", exc)
                    worst = 2

        if "features" in cfg.stages:
            for asset, start in active:
                try:
                    units = [(bt, y, m) for bt in cfg.bar_types for y, m in iter_months(start, cfg.end)]
                    with session.bar(f"features {asset}", len(units), "part") as bar:
                        for bt, year, month in units:
                            try:
                                st = write_features_month(cfg, asset, bt, year, month, _attach_slow_features)
                                bar.update(1, cached=(st == "cached"), written=(st == "written"), skipped=(st == "empty"))
                            except Exception as exc:
                                session.error(asset, f"features {bt} {year}-{month:02d}", exc)
                                bar.update(1, failed=True)
                                worst = 2
                    session.asset_summary(f"[features] {asset} done")
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    session.error(asset, "features", exc)
                    worst = 2

        if "dataset" in cfg.stages:
            # One month at a time across the cohort so cross-asset stays bounded.
            all_months = []
            for _, start in active:
                for ym in iter_months(start, cfg.end):
                    if ym not in all_months:
                        all_months.append(ym)
            cohort = [a for a, s in active]
            with session.bar("dataset", len(all_months) * max(len(cfg.bar_types), 1), "part") as bar:
                for year, month in all_months:
                    for bt in cfg.bar_types:
                        try:
                            statuses = write_dataset_month(cfg, cohort, bt, year, month, meta)
                            n_c = sum(1 for s in statuses.values() if s == "cached")
                            n_w = sum(1 for s in statuses.values() if s == "written")
                            n_e = sum(1 for s in statuses.values() if s == "empty")
                            bar.update(1, cached=n_c, written=n_w, skipped=n_e)
                        except KeyboardInterrupt:
                            raise
                        except Exception as exc:
                            session.error(",", f"dataset {bt} {year}-{month:02d}", exc)
                            bar.update(1, failed=True)
                            worst = 2

        if "validate" in cfg.stages:
            coverage = []
            for asset, start in active:
                try:
                    packed = meta.setdefault(asset, {})
                    worst = max(worst, _validate_asset(cfg, asset, start, len(active), packed, session))
                    coverage.extend(packed.pop("_coverage", []))
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    session.error(asset, "validate", exc)
                    worst = 2
            if coverage:
                print_coverage_report(coverage)
            worst = _independent_validate(cfg, worst)
    except KeyboardInterrupt:
        session.info("[run] interrupted (Ctrl+C). Re-run the same command to resume.")
        session.finish(130)
        raise
    return session.finish(worst)


def _fx_weekday_gaps(missing: list[int]) -> list[int]:
    """The Friday-Sunday close (and the year-end holidays) is the FX calendar, not a hole.

    Close and reopen are 17:00 America/New_York, so the UTC hour follows US DST.
    The rule is ``validate_run._fx_weekend_expected``, shared with the independent checks.
    """
    from pipeline.validate_run import _fx_weekend_expected
    return [ts for ts in missing if not _fx_weekend_expected(ts)]


def run_fx(cfg: RunConfig) -> int:
    """FX uses the same stages. Fetch is Dukascopy hours; bars and features are shared code."""
    from pipeline.fetch_fx import fetch_fx_range

    assets = resolve_assets(cfg)
    windows = cfg.windows or load_windows()
    splits = cfg.splits or load_splits()
    cfg.windows = windows
    cfg.splits = splits
    bad = [b for b in cfg.bar_types if b not in TIME_WIDTH_MS]
    if bad:
        raise ValueError(f"FX builds time bars only; not {bad}")
    if cfg.dry_run:
        print(f"[dry-run] market=fx stages={','.join(cfg.stages)} bar_types={','.join(cfg.bar_types)} root={cfg.data_root}")
        for asset in assets:
            start = max(effective_start(asset, cfg.start), date(2020, 1, 1))
            print(f"[dry-run] {asset} effective_start={start.isoformat()} end={cfg.end.isoformat()}")
        return 0

    worst = 0
    session = RunSession(market="fx", start=cfg.start, end=cfg.end, stages=cfg.stages)
    from pipeline.chunked import write_fx_features_month, write_dataset_month
    active = []
    for asset in assets:
        start = max(effective_start(asset, cfg.start), date(2020, 1, 1))
        if start > cfg.end:
            session.info(f"[plan] {asset} start {start} is after end {cfg.end}; skip")
            continue
        active.append((asset, start))
    names = [a for a, _ in active]
    tick_counts: dict[str, int] = {}
    gaps: dict[str, list] = {}
    try:
        if "fetch" in cfg.stages:
            for asset, start in active:
                session.info(f"[fetch] {asset} {start} -> {cfg.end}")
                try:
                    n = fetch_fx_range(
                        cfg.data_root, asset, start, cfg.end,
                        force=cfg.force, timeout=cfg.fetch_timeout,
                    )
                    session.info(f"[fetch] {asset} dukascopy hours={n}")
                    session.asset_summary(f"[fetch] {asset} hours={n}")
                except RuntimeError as exc:
                    session.error(asset, "fetch", exc)
                    worst = max(worst, 1)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    session.error(asset, "fetch", exc)
                    worst = 2

        if "bars" in cfg.stages:
            for asset, start in active:
                try:
                    months = list(iter_months(start, cfg.end))
                    n_ticks = 0
                    asset_gaps = []
                    with session.bar(f"bars {asset}", len(months), "month") as bar:
                        for year, month in months:
                            packed = write_fx_bars_month(cfg, asset, year, month)
                            if packed.get("cached"):
                                bar.update(1, cached=True)
                            else:
                                bar.update(1, written=True)
                                n_ticks += int(packed.get("n_ticks") or 0)
                                asset_gaps.extend(packed.get("gaps") or [])
                    tick_counts[asset] = n_ticks
                    gaps[asset] = asset_gaps
                    session.asset_summary(f"[bars] {asset} months={len(months)} ticks={n_ticks}")
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    session.error(asset, "bars", exc)
                    worst = 2

        if "features" in cfg.stages:
            all_months = []
            for _, start in active:
                for ym in iter_months(start, cfg.end):
                    if ym not in all_months:
                        all_months.append(ym)
            with session.bar("features", len(all_months) * max(len(cfg.bar_types), 1), "part") as bar:
                for year, month in all_months:
                    for bt in cfg.bar_types:
                        try:
                            statuses = write_fx_features_month(cfg, names, bt, year, month)
                            n_c = sum(1 for s in statuses.values() if s == "cached")
                            n_w = sum(1 for s in statuses.values() if s == "written")
                            bar.update(1, cached=n_c, written=n_w, skipped=sum(1 for s in statuses.values() if s == "empty"))
                        except KeyboardInterrupt:
                            raise
                        except Exception as exc:
                            session.error(",", f"features {bt} {year}-{month:02d}", exc)
                            bar.update(1, failed=True)
                            worst = 2

        if "dataset" in cfg.stages:
            all_months = []
            for _, start in active:
                for ym in iter_months(start, cfg.end):
                    if ym not in all_months:
                        all_months.append(ym)
            with session.bar("dataset", len(all_months) * max(len(cfg.bar_types), 1), "part") as bar:
                for year, month in all_months:
                    for bt in cfg.bar_types:
                        try:
                            statuses = write_dataset_month(cfg, names, bt, year, month, {}, cross_asset=False)
                            n_c = sum(1 for s in statuses.values() if s == "cached")
                            n_w = sum(1 for s in statuses.values() if s == "written")
                            bar.update(1, cached=n_c, written=n_w, skipped=sum(1 for s in statuses.values() if s == "empty"))
                        except KeyboardInterrupt:
                            raise
                        except Exception as exc:
                            session.error(",", f"dataset {bt} {year}-{month:02d}", exc)
                            bar.update(1, failed=True)
                            worst = 2

        if "validate" in cfg.stages:
            coverage = []
            for asset, start in active:
                try:
                    packed: dict = {}
                    worst = max(worst, _validate_asset(cfg, asset, start, len(active), packed, session))
                    coverage.extend(packed.pop("_coverage", []))
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    session.error(asset, "validate", exc)
                    worst = 2
            if coverage:
                print_coverage_report(coverage)
            worst = _independent_validate(cfg, worst)
    except KeyboardInterrupt:
        session.info("[run] interrupted (Ctrl+C). Re-run the same command to resume.")
        session.finish(130)
        raise
    return session.finish(worst)


def status(root: Path | None = None, cfg: "RunConfig | None" = None) -> int:
    """Inventory of gold files, or a completeness table when ``cfg`` has a range."""
    if cfg is None:
        if root is None:
            raise ValueError("pass a data root or a RunConfig")
        gold = root / "gold"
        if not gold.exists():
            print(f"[status] no gold under {gold}")
            return 0
        files = sorted(gold.rglob("*.parquet"))
        print(f"[status] {len(files)} gold files under {gold}")
        for path in files:
            print(f"  {path.relative_to(root)} bytes={path.stat().st_size}")
        return 0
    return _status_range(cfg)


def _status_range(cfg: RunConfig) -> int:
    """Without fetching, report how complete each asset/stage is for the range."""
    from pipeline.config import effective_start, load_listing_dates, universe_symbols
    from pipeline.partitions import partition_params, partition_fresh
    from pipeline.paths import (
        bars_month_path,
        features_month_path,
        fx_hour_path,
        gold_month_path,
        klines_1m_path,
        trades_path,
    )
    from pipeline.fetch_fx import iter_hours

    root = cfg.data_root
    assets = resolve_assets(cfg)
    listings = load_listing_dates() if cfg.market != "fx" else {}
    print(
        f"[status] market={cfg.market} {cfg.start} -> {cfg.end} "
        f"stages={','.join(cfg.stages)} bar_types={','.join(cfg.bar_types)} root={root}"
    )
    any_incomplete = 0
    for asset in assets:
        if cfg.market == "fx":
            start = max(cfg.start, date(2020, 1, 1))
        else:
            start = effective_start(asset, cfg.start, listings)
        if start > cfg.end:
            print(f"  {asset:12s}  skipped (start {start} after end)")
            continue
        parts = [f"{asset:12s}"]
        if "fetch" in cfg.stages:
            if cfg.market == "fx":
                hours = list(iter_hours(start, cfg.end))
                have = sum(1 for h in hours if fx_hour_path(root, asset, h.date(), h.hour).exists())
                total = len(hours)
            else:
                # A day Vision answered with no rows (empty) or a 404 long after the
                # fact (absent) is complete: there is nothing more to fetch.
                import json as _json
                from pipeline.paths import manifest_path
                man = manifest_path(root, "trades", asset)
                arts = _json.loads(man.read_text()).get("artifacts", {}) if man.exists() else {}
                days = list(iter_days(start, cfg.end))
                have = sum(
                    1 for d in days
                    if trades_path(root, asset, d).exists()
                    or arts.get(d.isoformat(), {}).get("status") in ("empty", "absent")
                )
                total = len(days)
            parts.append(f"fetch {have}/{total}")
            if have < total:
                any_incomplete = 1
        months = list(iter_months(start, cfg.end))
        for stage, path_fn in (
            ("bars", bars_month_path),
            ("features", features_month_path),
            ("dataset", gold_month_path),
        ):
            if stage not in cfg.stages and not (stage == "dataset" and "validate" in cfg.stages):
                if stage not in cfg.stages:
                    continue
            if stage == "validate":
                continue
            if stage not in cfg.stages and stage != "dataset":
                continue
            if stage not in cfg.stages:
                continue
            have = 0
            total = 0
            missing = []
            for y, m in months:
                first, last = clip_month(y, m, start, cfg.end)
                for bt in cfg.bar_types:
                    total += 1
                    path = path_fn(root, bt, asset, y, m)
                    params = partition_params(asset=asset, bar_type=bt, start=first, end=last)
                    # Status is conservative: file+sidecar is enough (params may differ by extra keys).
                    if sidecar_ok(path):
                        have += 1
                    else:
                        missing.append(f"{bt}:{y:04d}-{m:02d}")
            parts.append(f"{stage} {have}/{total}")
            if have < total:
                any_incomplete = 1
        print("  " + "  ".join(parts))
    print(f"[status] {'complete' if any_incomplete == 0 else 'incomplete'}")
    return 0 if any_incomplete == 0 else 1


