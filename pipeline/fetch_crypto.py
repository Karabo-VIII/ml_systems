"""Market source interface. Crypto is implemented. FX is a slot, not an adapter."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class MarketSource(Protocol):
    """What a venue must provide.

    An FX adapter can implement this later (session calendar, weekend closes).
    No FX adapter is shipped. Crypto-only series (funding, open interest) simply
    do not register for instruments that are not crypto.
    """

    market: str

    def list_instruments(self) -> list[str]:
        ...

    def fetch_ticks_or_bars(self, instrument: str, start: str, end: str) -> list[pl.DataFrame]:
        ...

    def calendar(self) -> str:
        """``24/7`` for crypto. FX would return a session calendar id."""
        ...

    def default_lags(self) -> dict[str, int]:
        ...

"""Binance Vision fetch. Vision is primary. The REST API is not required.

Funding lives at the monthly prefix. The daily funding prefix is empty.
An empty or unexpected body is a recorded miss, not a crash.
"""


import io
import json
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from pipeline.io import atomic_write_json, atomic_write_parquet, file_sha256, params_hash
from pipeline.paths import (
    funding_path,
    klines_1m_path,
    manifest_path,
    metrics_path,
    perp_1d_path,
    trades_path,
)
from pipeline.symbols import canonical_symbol
from pipeline.calendar import (  # re-exported
    clip_month,
    is_current_or_future_month,
    iter_days,
    iter_months,
)

VISION = "https://data.binance.vision/data"
USER_AGENT = "pipeline-v2"


def trades_url(symbol: str, day: date) -> str:
    sym = canonical_symbol(symbol)
    ds = day.isoformat()
    return f"{VISION}/spot/daily/aggTrades/{sym}/{sym}-aggTrades-{ds}.zip"


def klines_1m_url(symbol: str, day: date) -> str:
    sym = canonical_symbol(symbol)
    ds = day.isoformat()
    return f"{VISION}/spot/daily/klines/{sym}/1m/{sym}-1m-{ds}.zip"


def funding_url(symbol: str, year: int, month: int) -> str:
    """Monthly UM funding. The daily prefix is dead and is not requested."""
    sym = canonical_symbol(symbol)
    ym = f"{year:04d}-{month:02d}"
    return (
        f"{VISION}/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{ym}.zip"
    )


def metrics_url(symbol: str, day: date) -> str:
    sym = canonical_symbol(symbol)
    ds = day.isoformat()
    return f"{VISION}/futures/um/daily/metrics/{sym}/{sym}-metrics-{ds}.zip"


def perp_1d_url(symbol: str, day: date) -> str:
    sym = canonical_symbol(symbol)
    ds = day.isoformat()
    return f"{VISION}/futures/um/daily/klines/{sym}/1d/{sym}-1d-{ds}.zip"


def trades_month_url(symbol: str, year: int, month: int) -> str:
    sym = canonical_symbol(symbol)
    ym = f"{year:04d}-{month:02d}"
    return f"{VISION}/spot/monthly/aggTrades/{sym}/{sym}-aggTrades-{ym}.zip"


def klines_1m_month_url(symbol: str, year: int, month: int) -> str:
    sym = canonical_symbol(symbol)
    ym = f"{year:04d}-{month:02d}"
    return f"{VISION}/spot/monthly/klines/{sym}/1m/{sym}-1m-{ym}.zip"


def metrics_month_url(symbol: str, year: int, month: int) -> str:
    sym = canonical_symbol(symbol)
    ym = f"{year:04d}-{month:02d}"
    return f"{VISION}/futures/um/monthly/metrics/{sym}/{sym}-metrics-{ym}.zip"


def perp_1d_month_url(symbol: str, year: int, month: int) -> str:
    sym = canonical_symbol(symbol)
    ym = f"{year:04d}-{month:02d}"
    return f"{VISION}/futures/um/monthly/klines/{sym}/1d/{sym}-1d-{ym}.zip"


def http_get(url: str, timeout: int = 120) -> tuple[int, bytes, str | None]:
    """Return (status, body, error). HTTP errors do not raise."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return int(resp.status), resp.read(), None
    except urllib.error.HTTPError as exc:
        return int(exc.code), b"", f"HTTP {exc.code}"
    except Exception as exc:  # network, timeout, decode
        return 0, b"", f"{type(exc).__name__}: {exc}"


def _zip_bytes(content: bytes) -> bytes:
    if not content:
        raise ValueError("empty body")
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        if not names:
            raise ValueError("zip has no files")
        return zf.read(names[0])


def _has_header(raw: bytes) -> bool:
    line = raw.split(b"\n", 1)[0].strip()
    if not line:
        return False
    first = line.split(b",", 1)[0].strip().strip(b'"')
    try:
        int(first)
        return False
    except ValueError:
        return True


def _csv(raw: bytes, columns: list[str] | None = None) -> pl.DataFrame:
    if not raw.strip():
        raise ValueError("empty csv")
    if _has_header(raw):
        return pl.read_csv(io.BytesIO(raw), infer_schema_length=1000)
    if columns is None:
        raise ValueError("headerless csv requires columns")
    return pl.read_csv(
        io.BytesIO(raw),
        has_header=False,
        new_columns=columns,
        infer_schema_length=1000,
    )


def _as_bool(df: pl.DataFrame, col: str) -> pl.DataFrame:
    if col not in df.columns:
        return df
    dtype = df.schema[col]
    if dtype == pl.Boolean:
        return df
    return df.with_columns(
        pl.col(col).cast(pl.Utf8).str.to_lowercase().is_in(["true", "1", "t"]).alias(col)
    )


def _ts_us_and_ms(ts: pl.Expr) -> tuple[pl.Expr, pl.Expr]:
    """Vision aggTrades are microseconds. Older files are milliseconds."""
    ts = ts.cast(pl.Int64)
    ts_us = (
        pl.when(ts > 100_000_000_000_000).then(ts)
        .when(ts > 100_000_000_000).then(ts * 1000)
        .otherwise(ts * 1_000_000)
    )
    ts_ms = ts_us // 1000
    return ts_us, ts_ms


_AGG_HEADERLESS = [
    "agg_trade_id", "price", "qty", "first_trade_id",
    "last_trade_id", "timestamp", "is_buyer_maker", "is_best_match",
]


def parse_aggtrades(content: bytes) -> pl.DataFrame:
    """Parse a Vision aggTrades zip or raw CSV.

    Sorted by ``(ts_us, agg_trade_id)`` and deduped on ``agg_trade_id``.
    ``unique()`` without an explicit sort is not used: it shuffled v1 files.
    """
    raw = content
    if content[:2] == b"PK":
        raw = _zip_bytes(content)
    df = _csv(raw, _AGG_HEADERLESS)
    lower = {c.lower().strip(): c for c in df.columns}
    rename = {}
    aliases = {
        "agg_trade_id": "agg_trade_id",
        "price": "price",
        "quantity": "qty",
        "qty": "qty",
        "transact_time": "timestamp",
        "timestamp": "timestamp",
        "is_buyer_maker": "is_buyer_maker",
        "is_best_match": "is_best_match",
    }
    for src, dst in aliases.items():
        if src in lower and lower[src] != dst:
            rename[lower[src]] = dst
        elif src in lower and src != dst:
            rename[lower[src]] = dst
    if rename:
        df = df.rename(rename)
    need = ["agg_trade_id", "price", "qty", "timestamp", "is_buyer_maker"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"aggTrades missing columns {missing}; got {df.columns}")
    df = _as_bool(df, "is_buyer_maker")
    if "is_best_match" in df.columns:
        df = _as_bool(df, "is_best_match")
    ts_us, ts_ms = _ts_us_and_ms(pl.col("timestamp"))
    cols = [
        pl.col("agg_trade_id").cast(pl.Int64),
        ts_us.alias("ts_us"),
        ts_ms.alias("ts_ms"),
        pl.col("price").cast(pl.Float64),
        pl.col("qty").cast(pl.Float64),
        pl.col("is_buyer_maker").cast(pl.Boolean),
    ]
    if "is_best_match" in df.columns:
        cols.append(pl.col("is_best_match").cast(pl.Boolean))
    df = df.select(cols)
    df = df.filter((pl.col("price") > 0) & (pl.col("qty") > 0))
    df = df.sort(["ts_us", "agg_trade_id"]).unique(
        subset=["agg_trade_id"], keep="last", maintain_order=True,
    )
    return df.sort(["ts_us", "agg_trade_id"])


def parse_funding(content: bytes) -> pl.DataFrame:
    """Monthly Vision funding. Columns are ``calc_time`` / ``last_funding_rate``.

    Older files that still say ``calcTime`` / ``fundingRate`` are accepted.
    No column is invented.
    """
    raw = content
    if content[:2] == b"PK":
        raw = _zip_bytes(content)
    df = _csv(raw, None)
    if df.is_empty():
        return pl.DataFrame(schema={
            "ts_ms": pl.Int64,
            "funding_rate": pl.Float64,
            "funding_interval_hours": pl.Int32,
        })
    lower = {c.lower().strip(): c for c in df.columns}
    time_key = next((k for k in ("calc_time", "calctime", "fundingtime", "timestamp") if k in lower), None)
    rate_key = next((k for k in (
        "last_funding_rate", "funding_rate", "fundingrate", "funding_rate",
    ) if k in lower), None)
    if time_key is None or rate_key is None:
        raise ValueError(f"funding columns not recognised: {list(df.columns)}")
    interval_key = next((k for k in ("funding_interval_hours", "fundingintervalhours") if k in lower), None)
    ts = pl.col(lower[time_key])
    # calc_time may be an int ms or a datetime string
    if df.schema[lower[time_key]] == pl.Utf8 or df.schema[lower[time_key]] == pl.String:
        ts_ms = ts.str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False).dt.timestamp("ms").cast(pl.Int64)
    else:
        ts_ms = ts.cast(pl.Int64)
        ts_ms = pl.when(ts_ms > 100_000_000_000_000).then(ts_ms // 1000).otherwise(ts_ms)
    out_cols = [
        ts_ms.alias("ts_ms"),
        pl.col(lower[rate_key]).cast(pl.Float64).alias("funding_rate"),
    ]
    if interval_key:
        out_cols.append(pl.col(lower[interval_key]).cast(pl.Int32).alias("funding_interval_hours"))
    else:
        out_cols.append(pl.lit(None, dtype=pl.Int32).alias("funding_interval_hours"))
    df = df.select(out_cols).drop_nulls(["ts_ms", "funding_rate"])
    return df.sort("ts_ms").unique(subset=["ts_ms"], keep="last", maintain_order=True)


_KLINE_HEADERLESS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "n_trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]


def parse_klines(content: bytes, *, width_ms: int = 60_000) -> pl.DataFrame:
    raw = content
    if content[:2] == b"PK":
        raw = _zip_bytes(content)
    df = _csv(raw, _KLINE_HEADERLESS)
    lower = {c.lower().strip(): c for c in df.columns}
    # headerless path already has our names; headered path uses binance names
    def col(*names: str) -> str:
        for n in names:
            if n in lower:
                return lower[n]
            if n in df.columns:
                return n
        raise ValueError(f"kline missing {names}; got {df.columns}")
    open_name = col("open_time", "open time")
    df = df.select([
        pl.col(open_name).cast(pl.Int64).alias("ts_open_ms"),
        pl.col(col("open")).cast(pl.Float64),
        pl.col(col("high")).cast(pl.Float64),
        pl.col(col("low")).cast(pl.Float64),
        pl.col(col("close")).cast(pl.Float64),
        pl.col(col("volume")).cast(pl.Float64),
        pl.col(col("quote_volume", "quote_asset_volume")).cast(pl.Float64).alias("quote_volume"),
        pl.col(col("n_trades", "number_of_trades", "count")).cast(pl.Int64).alias("n_trades"),
        pl.col(col("taker_buy_base", "taker_buy_base_asset_volume", "taker_buy_volume")).cast(pl.Float64).alias("taker_buy_base"),
        pl.col(col("taker_buy_quote", "taker_buy_quote_asset_volume", "taker_buy_quote_volume")).cast(pl.Float64).alias("taker_buy_quote"),
    ])
    df = df.with_columns((pl.col("ts_open_ms") + width_ms).alias("ts_ms"))
    return df.sort("ts_open_ms").unique(subset=["ts_open_ms"], keep="last", maintain_order=True)


def parse_metrics(content: bytes) -> pl.DataFrame:
    """Keep the columns Vision actually shipped. Do not invent long_short_ratio."""
    raw = content
    if content[:2] == b"PK":
        raw = _zip_bytes(content)
    df = _csv(raw, None)
    if df.is_empty():
        return pl.DataFrame(schema={"ts_ms": pl.Int64})
    df = df.rename({c: c.strip() for c in df.columns})
    lower = {c.lower(): c for c in df.columns}
    time_key = next((k for k in ("create_time", "createtime", "timestamp", "time") if k in lower), None)
    if time_key is None:
        raise ValueError(f"metrics missing a time column: {df.columns}")
    src = lower[time_key]
    if df.schema[src] in (pl.Utf8, pl.String):
        ts = pl.col(src).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False).dt.timestamp("ms").cast(pl.Int64)
    else:
        ts = pl.col(src).cast(pl.Int64)
        ts = pl.when(ts > 100_000_000_000_000).then(ts // 1000).otherwise(ts)
    keep = [ts.alias("ts_ms")]
    for c in df.columns:
        if c == src:
            continue
        if c.lower() in ("symbol", "ignore"):
            continue
        dtype = df.schema[c]
        if dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.UInt32, pl.UInt64) or dtype in (pl.Utf8, pl.String):
            keep.append(pl.col(c).cast(pl.Float64, strict=False).alias(c.lower()))
    out = df.select(keep).drop_nulls(["ts_ms"])
    if "sum_open_interest_value" not in out.columns and "sumopeninterestvalue" in out.columns:
        out = out.rename({"sumopeninterestvalue": "sum_open_interest_value"})
    return out.sort("ts_ms").unique(subset=["ts_ms"], keep="last", maintain_order=True)


def empty_venue_frame(df: pl.DataFrame | None, required: str = "date") -> pl.DataFrame:
    """Cross-exchange bug: an empty venue used to KeyError on 'date'. It must not."""
    if df is None or df.is_empty() or required not in df.columns:
        return pl.DataFrame(schema={required: pl.Utf8, "close": pl.Float64, "venue": pl.Utf8})
    return df


@dataclass
class ArtifactRecord:
    key: str
    status: str
    n_rows: int
    source_url: str
    path: str | None = None
    sha256: str | None = None
    error: str | None = None
    retrieved_at_ms: int | None = None

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "status": self.status,
            "n_rows": self.n_rows,
            "source_url": self.source_url,
            "path": self.path,
            "sha256": self.sha256,
            "error": self.error,
            "retrieved_at_ms": self.retrieved_at_ms,
        }


@dataclass
class FetchReport:
    source: str
    asset: str
    records: list[ArtifactRecord] = field(default_factory=list)

    @property
    def n_ok(self) -> int:
        return sum(1 for r in self.records if r.status == "ok")

    @property
    def n_error(self) -> int:
        return sum(1 for r in self.records if r.status == "error")

    @property
    def n_cached(self) -> int:
        return sum(1 for r in self.records if r.status == "ok" and r.error == "cached")


def _now_ms() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"artifacts": {}}
    return json.loads(path.read_text())


def _save_manifest(path: Path, payload: dict) -> None:
    atomic_write_json(path, payload)


def _fresh(record: dict, dest: Path, force: bool) -> bool:
    """Per-artifact freshness. A directory mtime is not a freshness check."""
    if force:
        return False
    if record.get("status") != "ok":
        return False
    if not dest.exists():
        return False
    digest = record.get("sha256")
    if not digest:
        return False
    return file_sha256(dest) == digest


def _store(df: pl.DataFrame, dest: Path, schema_name: str, params: dict) -> str:
    atomic_write_parquet(
        df, dest, schema_name=schema_name, params=params,
        sort_cols=["ts_us", "agg_trade_id"] if "ts_us" in df.columns else ["ts_ms"],
    )
    return file_sha256(dest)



def _split_frame_by_day(df: pl.DataFrame, time_col: str) -> dict:
    if df.is_empty() or time_col not in df.columns:
        return {}
    tagged = df.with_columns(pl.from_epoch(pl.col(time_col), time_unit="ms").dt.date().alias("_day"))
    out: dict = {}
    for (day,), part in tagged.group_by("_day", maintain_order=True):
        out[day] = part.drop("_day")
    return out


def _store_day(rec_key, dest, df, schema_name, params, url, arts, report, today_ms):
    rec = ArtifactRecord(rec_key, "error", 0, url, retrieved_at_ms=today_ms)
    try:
        if df is None or df.is_empty():
            rec.status = "empty"
            rec.n_rows = 0
        else:
            digest = _store(df, dest, schema_name, params)
            rec.status = "ok"
            rec.n_rows = df.height
            rec.path = str(dest)
            rec.sha256 = digest
    except Exception as exc:  # noqa: BLE001
        rec.status = "error"
        rec.error = f"{type(exc).__name__}: {exc}"
    arts[rec_key] = rec.as_dict()
    report.records.append(rec)
    return rec


def _cached_record(key, dest, url, prev, report, arts):
    rec = ArtifactRecord(
        key, "ok", int(prev.get("n_rows", 0)), url, str(dest),
        prev.get("sha256"), error="cached", retrieved_at_ms=prev.get("retrieved_at_ms"),
    )
    report.records.append(rec)
    arts[key] = {**prev, "error": "cached"}
    return rec


def fetch_trades(root: Path, asset: str, start: date, end: date, *, force: bool = False, timeout: int = 120, progress=None) -> FetchReport:
    """Daily bronze files. Past months come from a single Vision monthly zip when needed."""
    asset = canonical_symbol(asset)
    report = FetchReport("trades", asset)
    man_path = manifest_path(root, "trades", asset)
    manifest = _load_manifest(man_path)
    arts: dict = manifest.setdefault("artifacts", {})
    today = date.today()
    for year, month in iter_months(start, end):
        first, last = clip_month(year, month, start, end)
        days = list(iter_days(first, last))
        use_monthly = not is_current_or_future_month(year, month, today)
        pending = []
        for day in days:
            key = day.isoformat()
            dest = trades_path(root, asset, day)
            prev = arts.get(key, {})
            if _fresh(prev, dest, force):
                _cached_record(key, dest, prev.get("source_url") or trades_url(asset, day), prev, report, arts)
                if progress:
                    progress("cached")
            else:
                pending.append(day)
        if not pending:
            continue
        wrote_from_month = False
        if use_monthly:
            url = trades_month_url(asset, year, month)
            status, body, err = http_get(url, timeout=timeout)
            if status == 200 and body:
                try:
                    frame = parse_aggtrades(body)
                    by_day = _split_frame_by_day(frame, "ts_ms")
                    for day in pending:
                        key = day.isoformat()
                        dest = trades_path(root, asset, day)
                        part = by_day.get(day, pl.DataFrame())
                        rec = _store_day(
                            key, dest, part, "trades_v1",
                            {"asset": asset, "day": key, "source_url": url},
                            url, arts, report, _now_ms(),
                        )
                        if progress:
                            progress("written" if rec.status == "ok" else "failed")
                    wrote_from_month = True
                except Exception as exc:  # noqa: BLE001
                    # Fall back to daily files for this month.
                    wrote_from_month = False
                    fallback_err = f"{type(exc).__name__}: {exc}"
            else:
                fallback_err = err or f"HTTP {status}"
        if not wrote_from_month:
            for day in pending:
                key = day.isoformat()
                dest = trades_path(root, asset, day)
                url = trades_url(asset, day)
                prev = arts.get(key, {})
                status, body, err = http_get(url, timeout=timeout)
                rec = ArtifactRecord(key, "error", 0, url, error=err, retrieved_at_ms=_now_ms())
                if status == 404:
                    rec.status = "missing_not_published"
                    if progress:
                        progress("failed")
                elif status != 200 or err:
                    rec.status = "error"
                    rec.error = err or f"HTTP {status}"
                    if progress:
                        progress("failed")
                else:
                    try:
                        df = parse_aggtrades(body)
                        rec = _store_day(key, dest, df, "trades_v1",
                                         {"asset": asset, "day": key, "source_url": url},
                                         url, arts, report, _now_ms())
                        if progress:
                            progress("written" if rec.status in {"ok", "empty"} else "failed")
                        continue
                    except Exception as exc:  # noqa: BLE001
                        rec.status = "error"
                        rec.error = f"{type(exc).__name__}: {exc}"
                        if progress:
                            progress("failed")
                arts[key] = rec.as_dict()
                report.records.append(rec)
    _save_manifest(man_path, manifest)
    return report


def fetch_klines_1m(root: Path, asset: str, start: date, end: date, *, force: bool = False, timeout: int = 120, progress=None) -> FetchReport:
    """Daily 1m kline files. Past months use the Vision monthly zip when needed."""
    asset = canonical_symbol(asset)
    report = FetchReport("klines_1m", asset)
    man_path = manifest_path(root, "klines_1m", asset)
    manifest = _load_manifest(man_path)
    arts: dict = manifest.setdefault("artifacts", {})
    today = date.today()
    for year, month in iter_months(start, end):
        first, last = clip_month(year, month, start, end)
        days = list(iter_days(first, last))
        use_monthly = not is_current_or_future_month(year, month, today)
        pending = []
        for day in days:
            key = day.isoformat()
            dest = klines_1m_path(root, asset, day)
            prev = arts.get(key, {})
            if _fresh(prev, dest, force):
                _cached_record(key, dest, prev.get("source_url") or klines_1m_url(asset, day), prev, report, arts)
                if progress:
                    progress("cached")
            else:
                pending.append(day)
        if not pending:
            continue
        wrote_from_month = False
        if use_monthly:
            url = klines_1m_month_url(asset, year, month)
            status, body, err = http_get(url, timeout=timeout)
            if status == 200 and body:
                try:
                    frame = parse_klines(body, width_ms=60_000)
                    by_day = _split_frame_by_day(frame, "ts_open_ms")
                    for day in pending:
                        key = day.isoformat()
                        dest = klines_1m_path(root, asset, day)
                        part = by_day.get(day, pl.DataFrame())
                        rec = _store_day(
                            key, dest, part, "kline_1m_v1",
                            {"asset": asset, "day": key, "source_url": url},
                            url, arts, report, _now_ms(),
                        )
                        if progress:
                            progress("written" if rec.status in {"ok", "empty"} else "failed")
                    wrote_from_month = True
                except Exception:
                    wrote_from_month = False
        if not wrote_from_month:
            for day in pending:
                key = day.isoformat()
                dest = klines_1m_path(root, asset, day)
                url = klines_1m_url(asset, day)
                status, body, err = http_get(url, timeout=timeout)
                rec = ArtifactRecord(key, "error", 0, url, error=err, retrieved_at_ms=_now_ms())
                if status == 404:
                    rec.status = "missing_not_published"
                    if progress:
                        progress("failed")
                elif status != 200 or err:
                    rec.status = "error"
                    rec.error = err or f"HTTP {status}"
                    if progress:
                        progress("failed")
                else:
                    try:
                        df = parse_klines(body, width_ms=60_000)
                        rec = _store_day(
                            key, dest, df, "kline_1m_v1",
                            {"asset": asset, "day": key, "source_url": url},
                            url, arts, report, _now_ms(),
                        )
                        if progress:
                            progress("written" if rec.status in {"ok", "empty"} else "failed")
                        continue
                    except Exception as exc:  # noqa: BLE001
                        rec.status = "error"
                        rec.error = f"{type(exc).__name__}: {exc}"
                        if progress:
                            progress("failed")
                arts[key] = rec.as_dict()
                report.records.append(rec)
    _save_manifest(man_path, manifest)
    return report


def fetch_funding(root: Path, asset: str, start: date, end: date, *, force: bool = False, timeout: int = 120) -> FetchReport:
    asset = canonical_symbol(asset)
    report = FetchReport("funding", asset)
    man_path = manifest_path(root, "funding", asset)
    manifest = _load_manifest(man_path)
    arts: dict = manifest.setdefault("artifacts", {})
    for year, month in iter_months(start, end):
        key = f"{year:04d}-{month:02d}"
        dest = funding_path(root, asset, year, month)
        url = funding_url(asset, year, month)
        assert "/daily/fundingRate/" not in url
        prev = arts.get(key, {})
        if _fresh(prev, dest, force):
            report.records.append(ArtifactRecord(key, "ok", int(prev.get("n_rows", 0)), url, str(dest), prev.get("sha256"), retrieved_at_ms=prev.get("retrieved_at_ms")))
            continue
        status, body, err = http_get(url, timeout=timeout)
        rec = ArtifactRecord(key, "error", 0, url, error=err, retrieved_at_ms=_now_ms())
        if status == 404:
            # The current month is unpublished until month-end. That is not a crash.
            rec.status = "missing_not_published"
        elif status != 200 or err:
            rec.status = "error"
            rec.error = err or f"HTTP {status}"
        else:
            try:
                df = parse_funding(body)
                if df.is_empty():
                    rec.status = "empty"
                else:
                    digest = _store(df, dest, "funding_v1", {"asset": asset, "month": key, "source_url": url})
                    rec.status = "ok"
                    rec.n_rows = df.height
                    rec.path = str(dest)
                    rec.sha256 = digest
            except Exception as exc:
                rec.status = "error"
                rec.error = f"{type(exc).__name__}: {exc}"
        arts[key] = rec.as_dict()
        report.records.append(rec)
    _save_manifest(man_path, manifest)
    return report


def fetch_metrics(root: Path, asset: str, start: date, end: date, *, force: bool = False, timeout: int = 120) -> FetchReport:
    asset = canonical_symbol(asset)
    report = FetchReport("metrics", asset)
    man_path = manifest_path(root, "metrics", asset)
    manifest = _load_manifest(man_path)
    arts: dict = manifest.setdefault("artifacts", {})
    for day in iter_days(start, end):
        key = day.isoformat()
        dest = metrics_path(root, asset, day)
        url = metrics_url(asset, day)
        prev = arts.get(key, {})
        if _fresh(prev, dest, force):
            report.records.append(ArtifactRecord(key, "ok", int(prev.get("n_rows", 0)), url, str(dest), prev.get("sha256"), retrieved_at_ms=prev.get("retrieved_at_ms")))
            continue
        status, body, err = http_get(url, timeout=timeout)
        rec = ArtifactRecord(key, "error", 0, url, error=err, retrieved_at_ms=_now_ms())
        if status == 404:
            rec.status = "missing_not_published"
        elif status != 200 or err:
            rec.status = "error"
            rec.error = err or f"HTTP {status}"
        else:
            try:
                df = parse_metrics(body)
                if "long_short_ratio" in df.columns and "count_long_short_ratio" not in df.columns and "sum_toptrader_long_short_ratio" not in df.columns:
                    # Parser must not have created this name. Defensive drop if a source
                    # file literally contains a constant we did not request is still kept
                    # only when the column was in the file. Nothing is added here.
                    pass
                if df.is_empty():
                    rec.status = "empty"
                else:
                    digest = _store(df, dest, "metrics_v1", {"asset": asset, "day": key, "source_url": url})
                    rec.status = "ok"
                    rec.n_rows = df.height
                    rec.path = str(dest)
                    rec.sha256 = digest
            except Exception as exc:
                rec.status = "error"
                rec.error = f"{type(exc).__name__}: {exc}"
        arts[key] = rec.as_dict()
        report.records.append(rec)
    _save_manifest(man_path, manifest)
    return report


def fetch_perp_1d(root: Path, asset: str, start: date, end: date, *, force: bool = False, timeout: int = 120) -> FetchReport:
    """UM daily klines, used only for basis. A 404 does not abort the asset."""
    asset = canonical_symbol(asset)
    report = FetchReport("perp_1d", asset)
    man_path = manifest_path(root, "perp_1d", asset)
    manifest = _load_manifest(man_path)
    arts: dict = manifest.setdefault("artifacts", {})
    for day in iter_days(start, end):
        key = day.isoformat()
        dest = perp_1d_path(root, asset, day)
        url = perp_1d_url(asset, day)
        prev = arts.get(key, {})
        if _fresh(prev, dest, force):
            report.records.append(ArtifactRecord(key, "ok", int(prev.get("n_rows", 0)), url, str(dest), prev.get("sha256"), retrieved_at_ms=prev.get("retrieved_at_ms")))
            continue
        status, body, err = http_get(url, timeout=timeout)
        rec = ArtifactRecord(key, "error", 0, url, error=err, retrieved_at_ms=_now_ms())
        if status == 404:
            rec.status = "missing_not_published"
        elif status != 200 or err:
            rec.status = "error"
            rec.error = err or f"HTTP {status}"
        else:
            try:
                df = parse_klines(body, width_ms=86_400_000)
                if df.is_empty():
                    rec.status = "empty"
                else:
                    digest = _store(df, dest, "kline_1d_um_v1", {"asset": asset, "day": key, "source_url": url})
                    rec.status = "ok"
                    rec.n_rows = df.height
                    rec.path = str(dest)
                    rec.sha256 = digest
            except Exception as exc:
                rec.status = "error"
                rec.error = f"{type(exc).__name__}: {exc}"
        arts[key] = rec.as_dict()
        report.records.append(rec)
    _save_manifest(man_path, manifest)
    return report


class BinanceVisionSource:
    """``MarketSource`` for Binance spot/UM via data.binance.vision. ``calendar`` is 24/7."""

    market = "crypto"

    def __init__(self, instruments: list[str]):
        self._instruments = [canonical_symbol(s) for s in instruments]

    def list_instruments(self) -> list[str]:
        return list(self._instruments)

    def fetch_ticks_or_bars(self, instrument: str, start: str, end: str) -> list[pl.DataFrame]:
        raise NotImplementedError(
            "Use fetch_trades / fetch_klines_1m. This method is the Protocol slot; "
            "the orchestrator calls the per-source functions so each artifact is recorded."
        )

    def calendar(self) -> str:
        return "24/7"

    def default_lags(self) -> dict[str, int]:
        return {"trades": 0, "klines": 60_000, "funding": 0, "metrics_daily": 86_400_000}
