"""Dukascopy tick ingester — the primary FREE bulk tick archive (FX / metals).

The FX analog of crypto's Binance Vision: free, no-auth, deep tick history,
direct hourly file downloads. One `.bi5` file per (instrument, UTC hour):

    {base_url}/{INSTRUMENT}/{YYYY}/{MM}/{DD}/{HH}h_ticks.bi5

Three correctness gotchas (all handled here):
  1. MONTH IS ZERO-INDEXED in the URL: 00 = January ... 11 = December.
  2. `.bi5` = LZMA-compressed. Decompress, then read 20-byte big-endian
     records: (ms_offset:u32, ask:i32, bid:i32, ask_vol:f32, bid_vol:f32).
     ms_offset is milliseconds from the top of that hour.
  3. Integer prices are scaled per-instrument: divide by `price_divisor`
     (100000 for 5-decimal FX, 1000 for 3-decimal JPY pairs + metals).

Weekend / holiday hours return an empty (0-byte) 200 or a 404 — that is the
24/5 FX calendar, NOT a fetch error, so those hours are silently skipped.

Output (bronze): data/raw/<PAIR>/ticks/<pair>_ticks_<YYYY-MM-DD>.parquet with
columns [timestamp(ms, UTC), bid, ask, bid_volume, ask_volume].

Run standalone:
    python -m pipeline.ingest.dukascopy --pairs EURUSD --start 2026-06-29 --end 2026-07-01
"""
from __future__ import annotations

import argparse
import io
import lzma
import struct
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Bootstrap src/ onto sys.path so `pipeline.*` imports work when run as a script.
_PIPELINE_PARENT = str(Path(__file__).resolve().parents[2])
if _PIPELINE_PARENT not in sys.path:
    sys.path.insert(0, _PIPELINE_PARENT)

from pipeline import layout
from pipeline.parquet_io import atomic_write_parquet, is_fresh
from pipeline.universe_loader import UniverseLoader

__contract__ = {
    "kind": "pipeline_stage",
    "stage": "fetch_dukascopy",
    "inputs": {"args": ["--pairs", "--universe", "--start", "--end", "--workers", "--force"],
               "config_keys": ["source.dukascopy.base_url", "source.dukascopy.month_zero_indexed"]},
    "outputs": {"files": "data/raw/<PAIR>/ticks/<pair>_ticks_<YYYY-MM-DD>.parquet",
                "row_unit": "one parquet per (pair, UTC day); rows = ticks",
                "columns": ["timestamp", "bid", "ask", "bid_volume", "ask_volume"]},
    "invariants": {
        "month_zero_indexed": True,
        "lzma_decompress": True,
        "per_pair_price_divisor": True,
        "weekend_empty_hour_is_not_error": True,
        "atomic_write": True,
    },
    "rationale": "Bronze tick layer; every FX bar + feature depends on this schema.",
}

BASE_URL = "https://datafeed.dukascopy.com/datafeed"
_REC = struct.Struct(">IIIff")   # ms_offset(u32), ask(u32), bid(u32), ask_vol(f32), bid_vol(f32)
_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"),
}


def get_session(retries: int = 3, backoff: float = 0.5) -> requests.Session:
    s = requests.Session()
    retry = Retry(total=retries, connect=retries, read=retries,
                  backoff_factor=backoff, status_forcelist=(500, 502, 503, 504))
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update(_HEADERS)
    return s


def _decompress_bi5(raw: bytes) -> bytes:
    """Decompress a .bi5 payload. Dukascopy uses an LZMA/alone stream; try the
    auto format first, then the legacy alone-format."""
    if not raw:
        return b""
    for fmt in (lzma.FORMAT_AUTO, lzma.FORMAT_ALONE):
        try:
            return lzma.decompress(raw, format=fmt)
        except lzma.LZMAError:
            continue
    raise lzma.LZMAError("bi5 payload not decodable as LZMA (auto/alone)")


def _hour_url(pair_dukas: str, dt: datetime, *, month_zero_indexed: bool = True,
              base_url: str = BASE_URL) -> str:
    month = dt.month - 1 if month_zero_indexed else dt.month
    return (f"{base_url}/{pair_dukas}/{dt.year:04d}/{month:02d}/"
            f"{dt.day:02d}/{dt.hour:02d}h_ticks.bi5")


def parse_hour_bytes(decompressed: bytes, hour_start_ms: int, divisor: float) -> dict:
    """Parse a decompressed hour buffer into column arrays (ms UTC + prices)."""
    n = len(decompressed) // _REC.size
    ts, bid, ask, bidv, askv = [], [], [], [], []
    for ms_off, ask_i, bid_i, ask_v, bid_v in _REC.iter_unpack(
            decompressed[: n * _REC.size]):
        ts.append(hour_start_ms + int(ms_off))
        ask.append(ask_i / divisor)
        bid.append(bid_i / divisor)
        askv.append(float(ask_v))
        bidv.append(float(bid_v))
    return {"timestamp": ts, "bid": bid, "ask": ask,
            "bid_volume": bidv, "ask_volume": askv}


def fetch_hour(session: requests.Session, pair_dukas: str, dt: datetime,
               divisor: float, *, month_zero_indexed: bool = True,
               base_url: str = BASE_URL) -> pl.DataFrame | None:
    """Fetch + parse one hourly .bi5. Returns a DataFrame, or None for a
    closed/empty hour (weekend/holiday). Raises on a genuine HTTP error."""
    url = _hour_url(pair_dukas, dt, month_zero_indexed=month_zero_indexed, base_url=base_url)
    r = session.get(url, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    if not r.content:
        return None
    raw = _decompress_bi5(r.content)
    if not raw:
        return None
    hour_start_ms = int(dt.replace(minute=0, second=0, microsecond=0,
                                   tzinfo=timezone.utc).timestamp() * 1000)
    cols = parse_hour_bytes(raw, hour_start_ms, divisor)
    if not cols["timestamp"]:
        return None
    return pl.DataFrame(cols, schema={
        "timestamp": pl.Int64, "bid": pl.Float64, "ask": pl.Float64,
        "bid_volume": pl.Float64, "ask_volume": pl.Float64,
    })


def fetch_day(session: requests.Session, pair: str, d: date, divisor: float, *,
              month_zero_indexed: bool = True, base_url: str = BASE_URL,
              throttle_ms: int = 120) -> pl.DataFrame | None:
    """Fetch all 24 hourly files for a UTC day and concat. None if the whole
    day is closed (weekend)."""
    n = layout.normalize_pair(pair)
    frames = []
    for hour in range(24):
        dt = datetime(d.year, d.month, d.day, hour, tzinfo=timezone.utc)
        df = fetch_hour(session, n.dukas, dt, divisor,
                        month_zero_indexed=month_zero_indexed, base_url=base_url)
        if df is not None and len(df):
            frames.append(df)
        if throttle_ms:
            time.sleep(throttle_ms / 1000.0)
    if not frames:
        return None
    return pl.concat(frames, how="vertical").sort("timestamp").unique(
        subset=["timestamp", "bid", "ask"], keep="first").sort("timestamp")


def fetch_pair_range(pair: str, start: date, end: date, divisor: float, *,
                     force: bool = False, month_zero_indexed: bool = True,
                     base_url: str = BASE_URL, throttle_ms: int = 120) -> dict:
    """Fetch [start, end) UTC days for one pair, writing one parquet per day.
    Skips already-present days (unless force) and closed weekend days."""
    session = get_session()
    n_days = (end - start).days
    n_written = n_skipped = n_closed = n_ticks = 0
    for i in range(n_days):
        d = start + timedelta(days=i)
        out = layout.tick_day_path(pair, d)
        if not force and is_fresh(out):
            n_skipped += 1
            continue
        # Skip Saturday entirely (weekday 5); Sunday(6)/Friday(4) partial hours
        # are handled by empty-hour skipping inside fetch_day.
        if d.weekday() == 5:
            n_closed += 1
            continue
        try:
            df = fetch_day(session, pair, d, divisor,
                           month_zero_indexed=month_zero_indexed,
                           base_url=base_url, throttle_ms=throttle_ms)
        except (requests.RequestException, lzma.LZMAError) as e:
            print(f"  [dukascopy] {pair} {d} FAIL: {type(e).__name__}: {e}", flush=True)
            continue
        if df is None or len(df) == 0:
            n_closed += 1
            continue
        atomic_write_parquet(df, out, required_cols={"timestamp", "bid", "ask"})
        n_written += 1
        n_ticks += len(df)
    return {"status": "ok", "pair": pair, "written": n_written,
            "skipped": n_skipped, "closed": n_closed, "ticks": n_ticks}


# ─── standalone CLI ─────────────────────────────────────────────────────────

def _worker(pair, start, end, divisor, force, throttle_ms):
    return fetch_pair_range(pair, start, end, divisor, force=force,
                            throttle_ms=throttle_ms)


def main() -> None:
    from pipeline import cli
    from pipeline.dispatch import run_per_task
    ap = argparse.ArgumentParser(description="Dukascopy tick fetcher (bronze layer)")
    cli.add_standard_args(ap, default_workers=4, default_start="2020-01-01")
    ap.add_argument("--throttle-ms", type=int, default=120,
                    help="Polite pause between hourly requests (default 120).")
    args = ap.parse_args()

    pairs = cli.resolve_pairs(args, stage_name="dukascopy")
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    U = UniverseLoader.load()

    tasks = [(p, start, end, U.price_divisor(p), args.force, args.throttle_ms)
             for p in pairs]
    if args.dry_run:
        for t in tasks:
            print(f"  would fetch {t[0]} {t[1]}..{t[2]} divisor={t[3]}")
        return
    run_per_task(tasks, _worker, workers=args.workers,
                 mode="thread" if args.workers > 1 else "serial",
                 stage_name="dukascopy",
                 progress_summary_keys=["written", "skipped", "closed", "ticks"])


if __name__ == "__main__":
    main()
