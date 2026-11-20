"""Dukascopy spot-FX ticks. Same ``MarketSource`` slot as Binance Vision.

URL (month is 0-indexed):
``https://datafeed.dukascopy.com/datafeed/<PAIR>/<YYYY>/<MM>/<DD>/<HH>h_ticks.bi5``

Each file is LZMA (alone) of 20-byte big-endian records:
``uint32 ms_offset, uint32 ask, uint32 bid, float32 ask_vol, float32 bid_vol``.
Prices are integers divided by 1e5, or by 1e3 when the pair contains JPY.
The volume floats are not exchange volume. Bars use the tick count instead.

Hours are cached as parquet and skipped on resume. 503s and timeouts back off.
Empty hours (weekends, no prints) are cached as an empty frame so they are not
requested again.
"""

from __future__ import annotations

import lzma
import struct
import time
from datetime import date, datetime, timedelta, timezone

import polars as pl

from pipeline.config import fx_price_divisor, load_fx
from pipeline.fetch_crypto import http_get
from pipeline.io import atomic_write_parquet
from pipeline.paths import fx_hour_path
from pipeline.symbols import canonical_fx_symbol

_REC = struct.Struct(">IIIff")
_TICK_SCHEMA = {
    "ts_ms": pl.Int64,
    "bid": pl.Float64,
    "ask": pl.Float64,
    "mid": pl.Float64,
    "spread": pl.Float64,
}
_FEED = "https://datafeed.dukascopy.com/datafeed"
_last_request = 0.0


def hour_url(pair: str, hour_start: datetime) -> str:
    pair = canonical_fx_symbol(pair)
    hour_start = hour_start.astimezone(timezone.utc)
    return (
        f"{_FEED}/{pair}/{hour_start.year}/{hour_start.month - 1:02d}/"
        f"{hour_start.day:02d}/{hour_start.hour:02d}h_ticks.bi5"
    )


def decode_bi5(blob: bytes, hour_start_ms: int, price_divisor: float) -> pl.DataFrame:
    """Decode one hour. An empty blob is an empty hour, not an error."""
    if not blob:
        return pl.DataFrame(schema=_TICK_SCHEMA)
    try:
        raw = lzma.decompress(blob)
    except lzma.LZMAError:
        raw = lzma.decompress(blob, format=lzma.FORMAT_ALONE)
    if len(raw) % _REC.size != 0:
        raise ValueError(f"bi5 payload length {len(raw)} is not a multiple of 20")
    ts, bid, ask, mid, spread = [], [], [], [], []
    for off in range(0, len(raw), _REC.size):
        ms, ask_i, bid_i, _ask_vol, _bid_vol = _REC.unpack_from(raw, off)
        if ms > 3_600_000:
            raise ValueError(f"tick offset {ms} ms is outside the hour")
        a = ask_i / price_divisor
        b = bid_i / price_divisor
        if a <= 0 or b <= 0 or a < b:
            raise ValueError(f"bad quote bid={b} ask={a}")
        ts.append(hour_start_ms + int(ms))
        bid.append(b)
        ask.append(a)
        mid.append((a + b) / 2.0)
        spread.append(a - b)
    if not ts:
        return pl.DataFrame(schema=_TICK_SCHEMA)
    return pl.DataFrame({
        "ts_ms": ts, "bid": bid, "ask": ask, "mid": mid, "spread": spread,
    }).sort("ts_ms")


def encode_bi5(ticks: pl.DataFrame, hour_start_ms: int, price_divisor: float) -> bytes:
    """Test helper. The inverse of ``decode_bi5`` for a synthetic hour."""
    raw = bytearray()
    for row in ticks.sort("ts_ms").iter_rows(named=True):
        ms = int(row["ts_ms"]) - hour_start_ms
        ask_i = int(round(float(row["ask"]) * price_divisor))
        bid_i = int(round(float(row["bid"]) * price_divisor))
        raw += _REC.pack(ms, ask_i, bid_i, float(row.get("ask_vol", 0.0)), float(row.get("bid_vol", 0.0)))
    return lzma.compress(bytes(raw), format=lzma.FORMAT_ALONE)


def _pause(min_interval_s: float) -> None:
    global _last_request
    wait = min_interval_s - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.monotonic()


def fetch_hour(
    root,
    pair: str,
    hour_start: datetime,
    *,
    force: bool = False,
    timeout: int = 60,
    min_interval_s: float | None = None,
    max_retries: int | None = None,
) -> pl.DataFrame:
    """Fetch one UTC hour, or read the cached parquet. Retries 503 and timeouts."""
    pair = canonical_fx_symbol(pair)
    hour_start = hour_start.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    path = fx_hour_path(root, pair, hour_start.date(), hour_start.hour)
    if path.exists() and not force:
        return pl.read_parquet(path)
    cfg = load_fx()
    interval = cfg["min_interval_s"] if min_interval_s is None else min_interval_s
    retries = int(cfg["max_retries"] if max_retries is None else max_retries)
    divisor = fx_price_divisor(pair)
    hour_ms = int(hour_start.timestamp() * 1000)
    url = hour_url(pair, hour_start)
    last_err = "no attempt"
    for attempt in range(retries):
        _pause(interval)
        try:
            status, body, err = http_get(url, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 — timeouts and resets are the retry case
            status, body, err = 0, b"", f"{type(exc).__name__}: {exc}"
        if status == 200:
            frame = decode_bi5(body, hour_ms, divisor)
            atomic_write_parquet(frame, path, schema_name="fx_ticks_v1", params={"pair": pair, "hour": hour_start.isoformat()})
            return frame
        if status in (404, 204):
            frame = pl.DataFrame(schema=_TICK_SCHEMA)
            atomic_write_parquet(
                frame, path, schema_name="fx_ticks_v1",
                params={"pair": pair, "hour": hour_start.isoformat(), "empty": True},
            )
            return frame
        last_err = err or f"HTTP {status}"
        # 503, 429, connection reset, timeout.
        time.sleep(min(32.0, 2.0 ** attempt))
    raise RuntimeError(f"{pair} {hour_start.isoformat()} failed after {retries} tries: {last_err}")


def iter_hours(start: date, end: date):
    cur = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
    stop = datetime(end.year, end.month, end.day, tzinfo=timezone.utc) + timedelta(days=1)
    while cur < stop:
        yield cur
        cur += timedelta(hours=1)


def fetch_fx_range(root, pair: str, start: date, end: date, **kwargs) -> int:
    """Fetch every hour in ``[start, end]`` inclusive. Returns hours written or read.

    A 503 after the per-hour retries is not cached as empty. The hour is
    retried with a long pause, then the range continues so a later resume
    can fill the hole. Failed hours are reported at the end.
    """
    n = n_cached = n_wrote = 0
    failed: list[str] = []
    force = bool(kwargs.get("force"))
    pair = canonical_fx_symbol(pair)
    for hour in iter_hours(start, end):
        path = fx_hour_path(root, pair, hour.date(), hour.hour)
        existed = path.exists() and not force
        if existed:
            n += 1
            n_cached += 1
            if hour.hour == 23:
                print(f"[fetch] {pair} {hour.date()} cached={n_cached} wrote={n_wrote}")
            continue
        last_err = None
        for attempt in range(3):
            try:
                fetch_hour(root, pair, hour, **kwargs)
                last_err = None
                break
            except RuntimeError as exc:
                last_err = exc
                wait = 45.0 * (attempt + 1)
                print(f"[fetch] {pair} {hour.isoformat()} {exc}; retry in {wait:.0f}s")
                time.sleep(wait)
        n += 1
        if last_err:
            failed.append(f"{hour.isoformat()} {last_err}")
        else:
            n_wrote += 1
        if hour.hour == 23:
            print(f"[fetch] {pair} {hour.date()} cached={n_cached} wrote={n_wrote} failed={len(failed)}")
    print(f"[fetch] {pair} hours={n} cached={n_cached} wrote={n_wrote} failed={len(failed)}")
    if failed:
        raise RuntimeError(f"{pair}: {len(failed)} hours failed: " + "; ".join(failed[:8]))
    return n


def read_fx_ticks(root, pair: str, start: date, end: date) -> pl.DataFrame:
    pair = canonical_fx_symbol(pair)
    frames = []
    for hour in iter_hours(start, end):
        path = fx_hour_path(root, pair, hour.date(), hour.hour)
        if path.exists():
            frames.append(pl.read_parquet(path))
    if not frames:
        return pl.DataFrame(schema=_TICK_SCHEMA)
    return pl.concat([f for f in frames if not f.is_empty()], how="vertical").sort("ts_ms") if any(not f.is_empty() for f in frames) else pl.DataFrame(schema=_TICK_SCHEMA)


class DukascopySource:
    """``MarketSource`` for Dukascopy spot FX. Calendar is the FX week, not 24/7."""

    market = "fx"

    def __init__(self, instruments: list[str]):
        self._instruments = [canonical_fx_symbol(s) for s in instruments]

    def list_instruments(self) -> list[str]:
        return list(self._instruments)

    def fetch_ticks_or_bars(self, instrument: str, start: str, end: str) -> list[pl.DataFrame]:
        raise NotImplementedError(
            "Use fetch_fx_range. The orchestrator records each hour itself."
        )

    def calendar(self) -> str:
        return "fx_week"

    def default_lags(self) -> dict[str, int]:
        return {"ticks": 0}
