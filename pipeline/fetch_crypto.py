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
import shutil
import tempfile
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
    month_last,
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


HTTP_ATTEMPTS = 4
HTTP_BACKOFF_S = 2.0


def _transient(status: int) -> bool:
    """Network error (0), throttling (429), or a server error: worth another try."""
    return status == 0 or status == 429 or status >= 500


def _sleep(seconds: float) -> None:
    import time
    time.sleep(seconds)


def http_get(url: str, timeout: int = 120) -> tuple[int, bytes, str | None]:
    """Return (status, body, error). HTTP errors do not raise. For small daily files.

    Transient failures (network, 429, 5xx) are retried with exponential backoff.
    A 404 is an answer, not a failure, and is returned at once.
    """
    status, body, err = 0, b"", None
    for attempt in range(HTTP_ATTEMPTS):
        status, body, err = _http_get_once(url, timeout)
        if not _transient(status):
            return status, body, err
        if attempt + 1 < HTTP_ATTEMPTS:
            _sleep(HTTP_BACKOFF_S * (2 ** attempt))
    return status, body, err


def _http_get_once(url: str, timeout: int) -> tuple[int, bytes, str | None]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return int(resp.status), resp.read(), None
    except urllib.error.HTTPError as exc:
        return int(exc.code), b"", f"HTTP {exc.code}"
    except Exception as exc:  # network, timeout, decode
        return 0, b"", f"{type(exc).__name__}: {exc}"


def http_get_to_path(url: str, dest: Path, timeout: int = 120) -> tuple[int, str | None]:
    """Stream a response onto disk. The body is never held as one bytes object.

    Used for Vision monthly zips, which can be a gigabyte compressed. A failure
    leaves no dest file.
    """
    status, err = 0, None
    for attempt in range(HTTP_ATTEMPTS):
        status, err = _http_get_to_path_once(url, dest, timeout)
        if not _transient(status):
            return status, err
        if attempt + 1 < HTTP_ATTEMPTS:
            _sleep(HTTP_BACKOFF_S * (2 ** attempt))
    return status, err


def _http_get_to_path_once(url: str, dest: Path, timeout: int) -> tuple[int, str | None]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            with open(tmp, "wb") as f:
                shutil.copyfileobj(resp, f, length=1024 * 1024)
            tmp.replace(dest)
            return int(resp.status), None
    except urllib.error.HTTPError as exc:
        if tmp.exists():
            tmp.unlink()
        return int(exc.code), f"HTTP {exc.code}"
    except Exception as exc:  # network, timeout, disk
        if tmp.exists():
            tmp.unlink()
        return 0, f"{type(exc).__name__}: {exc}"


CSV_CHUNK_ROWS = 250_000


def _first_zip_member(zf: zipfile.ZipFile) -> str:
    names = [n for n in zf.namelist() if not n.endswith("/")]
    if not names:
        raise ValueError("zip has no files")
    return names[0]


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


def _iter_csv_chunks(fh, columns: list[str] | None, chunk_rows: int = CSV_CHUNK_ROWS):
    """Yield DataFrames of at most ``chunk_rows`` from a binary CSV stream."""
    first = fh.readline()
    while first is not None and first.strip() == b"":
        first = fh.readline()
        if first == b"":
            return
    if not first:
        return
    header = first if _has_header(first) else None
    leftover = [] if header else [first]

    def flush(lines: list[bytes]) -> pl.DataFrame:
        if header is not None:
            raw = header + (b"" if header.endswith(b"\n") else b"\n") + b"".join(lines)
            return _csv(raw, columns)
        return _csv(b"".join(lines), columns)

    batch: list[bytes] = leftover
    for line in fh:
        if not line.strip():
            continue
        batch.append(line)
        if len(batch) >= chunk_rows:
            yield flush(batch)
            batch = []
    if batch:
        yield flush(batch)


def _iter_source_csv(source: bytes | Path, columns: list[str] | None, chunk_rows: int = CSV_CHUNK_ROWS):
    """Iterate CSV tables from a zip/CSV path or an in-memory zip/CSV body."""
    if isinstance(source, (bytes, bytearray)):
        if source[:2] == b"PK":
            with zipfile.ZipFile(io.BytesIO(source)) as zf:
                with zf.open(_first_zip_member(zf)) as fh:
                    yield from _iter_csv_chunks(fh, columns, chunk_rows)
        else:
            yield from _iter_csv_chunks(io.BytesIO(source), columns, chunk_rows)
        return
    path = Path(source)
    with path.open("rb") as probe:
        magic = probe.read(2)
    if magic == b"PK":
        with zipfile.ZipFile(path) as zf:
            with zf.open(_first_zip_member(zf)) as fh:
                yield from _iter_csv_chunks(fh, columns, chunk_rows)
    else:
        with path.open("rb") as fh:
            yield from _iter_csv_chunks(fh, columns, chunk_rows)


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


def _normalize_aggtrades(df: pl.DataFrame) -> pl.DataFrame:
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
    return df.filter((pl.col("price") > 0) & (pl.col("qty") > 0))


def _finalize_aggtrades(df: pl.DataFrame) -> pl.DataFrame:
    if df is None or df.is_empty():
        return df if df is not None else pl.DataFrame()
    return (
        df.sort(["ts_us", "agg_trade_id"])
        .unique(subset=["agg_trade_id"], keep="last", maintain_order=True)
        .sort(["ts_us", "agg_trade_id"])
    )


def iter_aggtrades(source: bytes | Path, chunk_rows: int = CSV_CHUNK_ROWS):
    """Yield normalized trade chunks from a zip/CSV path or body. Does not unique across chunks."""
    for raw in _iter_source_csv(source, _AGG_HEADERLESS, chunk_rows):
        part = _normalize_aggtrades(raw)
        if not part.is_empty():
            yield part


def parse_aggtrades(content: bytes) -> pl.DataFrame:
    """Parse a Vision aggTrades zip or raw CSV.

    Sorted by ``(ts_us, agg_trade_id)`` and deduped on ``agg_trade_id``.
    ``unique()`` without an explicit sort is not used: it shuffled v1 files.
    """
    parts = list(iter_aggtrades(content))
    if not parts:
        return pl.DataFrame()
    return _finalize_aggtrades(pl.concat(parts, how="diagonal_relaxed"))


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


def _normalize_klines(df: pl.DataFrame, width_ms: int) -> pl.DataFrame:
    lower = {c.lower().strip(): c for c in df.columns}
    def col(*names: str) -> str:
        for n in names:
            if n in lower:
                return lower[n]
            if n in df.columns:
                return n
        raise ValueError(f"kline missing {names}; got {df.columns}")
    open_name = col("open_time", "open time")
    # Spot Vision files switched to microseconds on 2025-01-01. Futures stay in ms.
    open_ms = pl.col(open_name).cast(pl.Int64)
    open_ms = pl.when(open_ms > 100_000_000_000_000).then(open_ms // 1000).otherwise(open_ms)
    df = df.select([
        open_ms.alias("ts_open_ms"),
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
    return df.with_columns((pl.col("ts_open_ms") + width_ms).alias("ts_ms"))


def _finalize_klines(df: pl.DataFrame) -> pl.DataFrame:
    if df is None or df.is_empty():
        return df if df is not None else pl.DataFrame()
    return df.sort("ts_open_ms").unique(subset=["ts_open_ms"], keep="last", maintain_order=True)


def iter_klines(source: bytes | Path, *, width_ms: int = 60_000, chunk_rows: int = CSV_CHUNK_ROWS):
    for raw in _iter_source_csv(source, _KLINE_HEADERLESS, chunk_rows):
        part = _normalize_klines(raw, width_ms)
        if not part.is_empty():
            yield part


def parse_klines(content: bytes, *, width_ms: int = 60_000) -> pl.DataFrame:
    parts = list(iter_klines(content, width_ms=width_ms))
    if not parts:
        return pl.DataFrame()
    return _finalize_klines(pl.concat(parts, how="diagonal_relaxed"))


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


ABSENT_AFTER_DAYS = 7


def _fresh(record: dict, dest: Path, force: bool) -> bool:
    """Per-artifact freshness. A directory mtime is not a freshness check.

    Status ``empty`` means a successful fetch found no rows for that day (for
    example a pre-listing day inside a monthly zip). Status ``absent`` means
    Vision answered 404 for a period that ended long enough ago that it will
    never be published. Neither has a parquet, and neither triggers another
    download (``--force`` retries them).
    """
    if force:
        return False
    if record.get("status") in ("empty", "absent"):
        return True
    if record.get("status") != "ok":
        return False
    if not dest.exists():
        return False
    digest = record.get("sha256")
    if not digest:
        return False
    return file_sha256(dest) == digest


def _sidecar_matches(dest: Path) -> str | None:
    """The file's sha256 when its ``.sha256`` sidecar agrees, else ``None``.

    ``atomic_write_parquet`` writes the sidecar after the file is in place, so a
    matching pair is a finished artifact even if the fetch manifest was never
    saved (an interrupted run) or comes from another data root.
    """
    sidecar = dest.with_suffix(dest.suffix + ".sha256")
    if not dest.exists() or not sidecar.exists():
        return None
    try:
        expected = sidecar.read_text().strip()
        actual = file_sha256(dest)
    except OSError:
        return None
    return actual if actual == expected else None


def _check_fresh(arts: dict, key: str, dest: Path, force: bool) -> bool:
    """``_fresh`` on the manifest record, or adopt a finished file the record does not know."""
    if _fresh(arts.get(key, {}), dest, force):
        return True
    if force:
        return False
    digest = _sidecar_matches(dest)
    if digest is None:
        return False
    try:
        import pyarrow.parquet as pq
        n_rows = int(pq.read_metadata(dest).num_rows)
    except Exception:  # noqa: BLE001
        return False
    prev = arts.get(key, {})
    arts[key] = {
        **prev, "key": key, "status": "ok", "n_rows": n_rows, "path": str(dest),
        "sha256": digest, "error": None, "adopted": True,
    }
    return True


def _miss_status(period_end: date, today: date | None = None) -> str:
    """A 404 for a period that ended over a week ago is permanent (``absent``)."""
    today = today or date.today()
    if (today - period_end).days > ABSENT_AFTER_DAYS:
        return "absent"
    return "missing_not_published"


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


def _record_absent(key, url, arts, report, progress=None):
    rec = ArtifactRecord(key, "absent", 0, url, error="HTTP 404 (monthly zip)", retrieved_at_ms=_now_ms())
    arts[key] = rec.as_dict()
    report.records.append(rec)
    if progress:
        progress("cached")
    return rec


def _cached_record(key, dest, url, prev, report, arts):
    status = prev.get("status") or "ok"
    rec = ArtifactRecord(
        key, status, int(prev.get("n_rows", 0)), url,
        str(dest) if status == "ok" else None,
        prev.get("sha256"), error="cached", retrieved_at_ms=prev.get("retrieved_at_ms"),
    )
    report.records.append(rec)
    arts[key] = {**prev, "error": "cached"}
    return rec


def _split_chunks_to_day_parts(chunks, time_col: str, tmpdir: Path) -> dict:
    """Write each chunk's per-day slice to a temp parquet. One chunk in memory at a time."""
    parts: dict = {}
    tmpdir = Path(tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)
    for i, chunk in enumerate(chunks):
        if chunk is None or chunk.is_empty() or time_col not in chunk.columns:
            continue
        tagged = chunk.with_columns(pl.from_epoch(pl.col(time_col), time_unit="ms").dt.date().alias("_day"))
        for (day,), part in tagged.group_by("_day", maintain_order=True):
            path = tmpdir / f"{day.isoformat()}_{i}.parquet"
            part.drop("_day").write_parquet(path)
            parts.setdefault(day, []).append(path)
    return parts


def _concat_day_parts(paths: list, finalize) -> pl.DataFrame:
    if not paths:
        return pl.DataFrame()
    frames = [pl.read_parquet(p) for p in paths]
    return finalize(pl.concat(frames, how="diagonal_relaxed"))


def fetch_trades(root: Path, asset: str, start: date, end: date, *, force: bool = False, timeout: int = 120, progress=None) -> FetchReport:
    """Daily bronze files. Past months come from a single Vision monthly zip when needed."""
    asset = canonical_symbol(asset)
    report = FetchReport("trades", asset)
    man_path = manifest_path(root, "trades", asset)
    manifest = _load_manifest(man_path)
    arts: dict = manifest.setdefault("artifacts", {})
    today = date.today()
    for year, month in iter_months(start, end):
        _save_manifest(man_path, manifest)  # progress survives Ctrl+C
        first, last = clip_month(year, month, start, end)
        days = list(iter_days(first, last))
        use_monthly = not is_current_or_future_month(year, month, today)
        pending = []
        for day in days:
            key = day.isoformat()
            dest = trades_path(root, asset, day)
            if _check_fresh(arts, key, dest, force):
                prev = arts[key]
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
            with tempfile.TemporaryDirectory(prefix="vision_trades_") as td:
                zpath = Path(td) / "month.zip"
                status, err = http_get_to_path(url, zpath, timeout=timeout)
                if status == 200 and zpath.exists() and zpath.stat().st_size:
                    try:
                        day_parts = _split_chunks_to_day_parts(
                            iter_aggtrades(zpath), "ts_ms", Path(td) / "days",
                        )
                        for day in pending:
                            key = day.isoformat()
                            dest = trades_path(root, asset, day)
                            part = _concat_day_parts(day_parts.get(day, []), _finalize_aggtrades)
                            rec = _store_day(
                                key, dest, part, "trades_v1",
                                {"asset": asset, "day": key, "source_url": url},
                                url, arts, report, _now_ms(),
                            )
                            if progress:
                                progress("written" if rec.status in {"ok", "empty"} else "failed")
                        wrote_from_month = True
                    except Exception as exc:  # noqa: BLE001
                        # Fall back to daily files for this month.
                        wrote_from_month = False
                        fallback_err = f"{type(exc).__name__}: {exc}"
                elif status == 404 and _miss_status(month_last(year, month)) == "absent":
                    # Vision ships a monthly zip for every past month the symbol traded
                    # (including a partial listing month). No zip a week after the month
                    # ended: nothing to fetch daily. A recent month falls back to daily files.
                    for day in pending:
                        _record_absent(day.isoformat(), url, arts, report, progress)
                    wrote_from_month = True
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
                    rec.status = _miss_status(day)
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
        _save_manifest(man_path, manifest)  # progress survives Ctrl+C
        first, last = clip_month(year, month, start, end)
        days = list(iter_days(first, last))
        use_monthly = not is_current_or_future_month(year, month, today)
        pending = []
        for day in days:
            key = day.isoformat()
            dest = klines_1m_path(root, asset, day)
            if _check_fresh(arts, key, dest, force):
                prev = arts[key]
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
            with tempfile.TemporaryDirectory(prefix="vision_klines_") as td:
                zpath = Path(td) / "month.zip"
                status, err = http_get_to_path(url, zpath, timeout=timeout)
                if status == 200 and zpath.exists() and zpath.stat().st_size:
                    try:
                        day_parts = _split_chunks_to_day_parts(
                            iter_klines(zpath, width_ms=60_000), "ts_open_ms", Path(td) / "days",
                        )
                        for day in pending:
                            key = day.isoformat()
                            dest = klines_1m_path(root, asset, day)
                            part = _concat_day_parts(day_parts.get(day, []), _finalize_klines)
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
                elif status == 404 and _miss_status(month_last(year, month)) == "absent":
                    for day in pending:
                        _record_absent(day.isoformat(), url, arts, report, progress)
                    wrote_from_month = True
        if not wrote_from_month:
            for day in pending:
                key = day.isoformat()
                dest = klines_1m_path(root, asset, day)
                url = klines_1m_url(asset, day)
                status, body, err = http_get(url, timeout=timeout)
                rec = ArtifactRecord(key, "error", 0, url, error=err, retrieved_at_ms=_now_ms())
                if status == 404:
                    rec.status = _miss_status(day)
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
        _save_manifest(man_path, manifest)  # progress survives Ctrl+C
        key = f"{year:04d}-{month:02d}"
        dest = funding_path(root, asset, year, month)
        url = funding_url(asset, year, month)
        assert "/daily/fundingRate/" not in url
        if _check_fresh(arts, key, dest, force):
            prev = arts[key]
            report.records.append(ArtifactRecord(key, "ok", int(prev.get("n_rows", 0)), url, str(dest), prev.get("sha256"), error="cached", retrieved_at_ms=prev.get("retrieved_at_ms")))
            continue
        status, body, err = http_get(url, timeout=timeout)
        rec = ArtifactRecord(key, "error", 0, url, error=err, retrieved_at_ms=_now_ms())
        if status == 404:
            # The current month is unpublished until month-end. That is not a crash.
            rec.status = _miss_status(month_last(year, month))
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
        if day.day == 1:
            _save_manifest(man_path, manifest)  # progress survives Ctrl+C
        key = day.isoformat()
        dest = metrics_path(root, asset, day)
        url = metrics_url(asset, day)
        if _check_fresh(arts, key, dest, force):
            prev = arts[key]
            report.records.append(ArtifactRecord(key, "ok", int(prev.get("n_rows", 0)), url, str(dest), prev.get("sha256"), error="cached", retrieved_at_ms=prev.get("retrieved_at_ms")))
            continue
        status, body, err = http_get(url, timeout=timeout)
        rec = ArtifactRecord(key, "error", 0, url, error=err, retrieved_at_ms=_now_ms())
        if status == 404:
            rec.status = _miss_status(day)
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
    """UM daily klines, used only for basis. A 404 does not abort the asset.

    Past months come from one Vision monthly zip (one request instead of ~30);
    the current month, or a month whose zip is missing, uses the daily files.
    """
    asset = canonical_symbol(asset)
    report = FetchReport("perp_1d", asset)
    man_path = manifest_path(root, "perp_1d", asset)
    manifest = _load_manifest(man_path)
    arts: dict = manifest.setdefault("artifacts", {})
    today = date.today()
    for year, month in iter_months(start, end):
        _save_manifest(man_path, manifest)  # progress survives Ctrl+C
        first, last = clip_month(year, month, start, end)
        pending = []
        for day in iter_days(first, last):
            key = day.isoformat()
            dest = perp_1d_path(root, asset, day)
            url = perp_1d_url(asset, day)
            if _check_fresh(arts, key, dest, force):
                prev = arts[key]
                report.records.append(ArtifactRecord(key, prev.get("status", "ok"), int(prev.get("n_rows", 0)), url, str(dest), prev.get("sha256"), error="cached", retrieved_at_ms=prev.get("retrieved_at_ms")))
                continue
            pending.append(day)
        if not pending:
            continue
        if not is_current_or_future_month(year, month, today):
            url = perp_1d_month_url(asset, year, month)
            status, body, err = http_get(url, timeout=timeout)
            if status == 200 and body:
                try:
                    by_day = _split_frame_by_day(parse_klines(body, width_ms=86_400_000), "ts_open_ms")
                    for day in pending:
                        key = day.isoformat()
                        _store_day(
                            key, perp_1d_path(root, asset, day), by_day.get(day),
                            "kline_1d_um_v1", {"asset": asset, "day": key, "source_url": url},
                            url, arts, report, _now_ms(),
                        )
                    continue
                except Exception:  # noqa: BLE001  fall back to daily files
                    pass
        for day in pending:
            key = day.isoformat()
            dest = perp_1d_path(root, asset, day)
            url = perp_1d_url(asset, day)
            status, body, err = http_get(url, timeout=timeout)
            rec = ArtifactRecord(key, "error", 0, url, error=err, retrieved_at_ms=_now_ms())
            if status == 404:
                rec.status = _miss_status(day)
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
        from pipeline.config import lag_ms
        return {"trades": 0, "klines": 60_000, "funding": 0, "metrics_daily": lag_ms("metrics_daily")}
