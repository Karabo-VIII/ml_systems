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

import numpy as np
import polars as pl

from pipeline.config import fx_price_divisor, load_fx
from pipeline.fetch_crypto import _http_get_once


def http_get(url: str, timeout: int = 60):
    """One attempt. ``fetch_hour`` owns the retry policy for Dukascopy (``fx.max_retries``)."""
    return _http_get_once(url, timeout)
from pipeline.io import atomic_write_parquet
from pipeline.paths import fx_hour_path
from pipeline.symbols import canonical_fx_symbol

_REC = struct.Struct(">IIIff")
_REC_DTYPE = np.dtype([("ms", ">u4"), ("ask", ">u4"), ("bid", ">u4"), ("ask_vol", ">f4"), ("bid_vol", ">f4")])
_TICK_SCHEMA = {
    "ts_ms": pl.Int64,
    "bid": pl.Float64,
    "ask": pl.Float64,
    "mid": pl.Float64,
    "spread": pl.Float64,
}
_FEED = "https://datafeed.dukascopy.com/datafeed"


class _Pacer:
    """Request spacing for Dukascopy, which throttles a fast client with 503s,
    timeouts and resets. The interval doubles on each of those (up to
    ``MAX_INTERVAL_S``) and shrinks by 15% after each answered request, never
    below ``fx.min_interval_s``: the client settles near the rate the server
    tolerates instead of retrying into the throttle."""

    MAX_INTERVAL_S = 60.0

    def __init__(self) -> None:
        self.interval = 0.0
        self.last = 0.0

    def wait(self, min_interval: float) -> None:
        self.interval = max(self.interval, float(min_interval))
        delay = self.interval - (time.monotonic() - self.last)
        if delay > 0:
            time.sleep(delay)
        self.last = time.monotonic()

    def answered(self, min_interval: float) -> None:
        self.interval = max(float(min_interval), self.interval * 0.85)

    def throttled(self) -> None:
        self.interval = min(self.MAX_INTERVAL_S, max(2.0, self.interval * 2.0))


_PACER = _Pacer()


def fx_market_closed(hour_start: datetime) -> bool:
    """True for an hour wholly inside the weekend close under either DST regime.

    The FX week closes Friday 17:00 and reopens Sunday 17:00 America/New_York,
    which is 21:00 or 22:00 UTC. Every hour from Friday 22:00 UTC to Sunday
    20:00 UTC is therefore shut in summer and in winter, so it is recorded as
    an empty, closed hour without a request. The hours at the edges are fetched.
    """
    h = hour_start.astimezone(timezone.utc)
    wd = h.weekday()  # Monday = 0
    if wd == 5:
        return True
    if wd == 4 and h.hour >= 22:
        return True
    if wd == 6 and h.hour < 20:
        return True
    return False


def hour_url(pair: str, hour_start: datetime) -> str:
    pair = canonical_fx_symbol(pair)
    hour_start = hour_start.astimezone(timezone.utc)
    return (
        f"{_FEED}/{pair}/{hour_start.year}/{hour_start.month - 1:02d}/"
        f"{hour_start.day:02d}/{hour_start.hour:02d}h_ticks.bi5"
    )


def decode_bi5(blob: bytes, hour_start_ms: int, price_divisor: float) -> pl.DataFrame:
    """Decode one hour. An empty blob is an empty hour, not an error.

    Vectorised (numpy structured dtype). Ticks keep the file's order, which is
    time order; ticks in the same millisecond stay in that order (stable sort).
    """
    if not blob:
        return pl.DataFrame(schema=_TICK_SCHEMA)
    try:
        raw = lzma.decompress(blob)
    except lzma.LZMAError:
        raw = lzma.decompress(blob, format=lzma.FORMAT_ALONE)
    if len(raw) % _REC.size != 0:
        raise ValueError(f"bi5 payload length {len(raw)} is not a multiple of 20")
    rec = np.frombuffer(raw, dtype=_REC_DTYPE)
    if len(rec) == 0:
        return pl.DataFrame(schema=_TICK_SCHEMA)
    ms = rec["ms"].astype(np.int64)
    if (ms >= 3_600_000).any():
        raise ValueError(f"tick offset {int(ms.max())} ms is outside the hour")
    ask = rec["ask"].astype(np.float64) / price_divisor
    bid = rec["bid"].astype(np.float64) / price_divisor
    if (ask <= 0).any() or (bid <= 0).any():
        bad = int(np.argmax((ask <= 0) | (bid <= 0)))
        raise ValueError(f"bad quote bid={bid[bad]} ask={ask[bad]}")
    # A crossed quote (ask < bid) is kept as shipped; the validator counts them.
    return pl.DataFrame({
        "ts_ms": hour_start_ms + ms, "bid": bid, "ask": ask,
        "mid": (ask + bid) / 2.0, "spread": ask - bid,
    }, schema=_TICK_SCHEMA).sort("ts_ms", maintain_order=True)


def encode_bi5(ticks: pl.DataFrame, hour_start_ms: int, price_divisor: float) -> bytes:
    """Test helper. The inverse of ``decode_bi5`` for a synthetic hour."""
    raw = bytearray()
    for row in ticks.sort("ts_ms", maintain_order=True).iter_rows(named=True):
        ms = int(row["ts_ms"]) - hour_start_ms
        ask_i = int(round(float(row["ask"]) * price_divisor))
        bid_i = int(round(float(row["bid"]) * price_divisor))
        raw += _REC.pack(ms, ask_i, bid_i, float(row.get("ask_vol", 0.0)), float(row.get("bid_vol", 0.0)))
    return lzma.compress(bytes(raw), format=lzma.FORMAT_ALONE)


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
    """Fetch one UTC hour, or read the cached parquet. Retries 503 and timeouts.

    An hour inside the weekend close (:func:`fx_market_closed`) is cached as an
    empty, closed hour without a request (``fx.skip_closed_hours``).
    """
    pair = canonical_fx_symbol(pair)
    hour_start = hour_start.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    path = fx_hour_path(root, pair, hour_start.date(), hour_start.hour)
    if path.exists() and not force:
        return pl.read_parquet(path)
    cfg = load_fx()
    if cfg.get("skip_closed_hours", True) and fx_market_closed(hour_start):
        frame = pl.DataFrame(schema=_TICK_SCHEMA)
        atomic_write_parquet(
            frame, path, schema_name="fx_ticks_v1",
            params={"pair": pair, "hour": hour_start.isoformat(), "empty": True, "closed": True},
        )
        return frame
    interval = cfg["min_interval_s"] if min_interval_s is None else min_interval_s
    retries = int(cfg["max_retries"] if max_retries is None else max_retries)
    divisor = fx_price_divisor(pair)
    hour_ms = int(hour_start.timestamp() * 1000)
    url = hour_url(pair, hour_start)
    last_err = "no attempt"
    for attempt in range(retries):
        _PACER.wait(interval)
        try:
            status, body, err = http_get(url, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 — timeouts and resets are the retry case
            status, body, err = 0, b"", f"{type(exc).__name__}: {exc}"
        if status in (200, 204, 404):
            _PACER.answered(interval)
        if status == 200:
            frame = decode_bi5(body, hour_ms, divisor)
            atomic_write_parquet(frame, path, schema_name="fx_ticks_v1", params={"pair": pair, "hour": hour_start.isoformat()})
            return frame
        if status in (404, 204):
            frame = pl.DataFrame(schema=_TICK_SCHEMA)
            # Cache an empty hour only once it is safely in the past. A recent or
            # future hour may simply not be published yet.
            if _hour_settled(hour_start):
                atomic_write_parquet(
                    frame, path, schema_name="fx_ticks_v1",
                    params={"pair": pair, "hour": hour_start.isoformat(), "empty": True},
                )
            return frame
        last_err = err or f"HTTP {status}"
        # 503, 429, connection reset, timeout: the server's throttle. Slow down.
        _PACER.throttled()
    raise RuntimeError(f"{pair} {hour_start.isoformat()} failed after {retries} tries: {last_err}")


SETTLE_HOURS = 6


def _hour_settled(hour_start: datetime, now: datetime | None = None) -> bool:
    now = now or datetime.now(tz=timezone.utc)
    return hour_start + timedelta(hours=1 + SETTLE_HOURS) <= now


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
            except ValueError as exc:  # a corrupt file: not cached, reported, range continues
                last_err = exc
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
    frames = [f for f in frames if not f.is_empty()]
    if not frames:
        return pl.DataFrame(schema=_TICK_SCHEMA)
    # Stable: ticks sharing a millisecond keep the order Dukascopy shipped them in,
    # which decides a bar's close. An unstable sort may swap them.
    return pl.concat(frames, how="vertical").sort("ts_ms", maintain_order=True)


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
