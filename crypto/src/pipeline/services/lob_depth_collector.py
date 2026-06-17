"""G2: LOB depth-5 WebSocket collector for queue-imbalance signal extraction.

Closes G2 from gap audit 2026-04-25.

What this does:
  Subscribes to Binance Spot WebSocket combined stream `<symbol>@depth5@100ms`
  for a configurable symbol list. Buffers L5 bid/ask snapshots in memory,
  flushes to parquet every N seconds with timestamps. Files rotate daily.

Capacity:
  100ms cadence x 86400s/day = 864K snapshots/day per symbol.
  Each snapshot = 5 bids + 5 asks = ~200 bytes serialized.
  Daily storage per symbol: ~170MB raw, ~30MB compressed parquet.
  10 symbols x 30 days = ~9GB compressed. Manageable.

Usage:
  python -m src.frontier.ingest.lob_depth_collector
      --symbols BTCUSDT,ETHUSDT,SOLUSDT
      --duration-min 60          # collect 60 minutes (default: forever)
      --flush-interval-s 60       # flush every 60 seconds

Output:
  data/lob/<SYMBOL>/<YYYY-MM-DD>/<HH>.parquet   (canonical, per STATE.md)

Schema:
  ts_ms (i64), bid_p1..bid_p5 (f64), bid_q1..bid_q5 (f64),
  ask_p1..ask_p5 (f64), ask_q1..ask_q5 (f64), update_id (i64)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import websockets

ROOT = Path(__file__).resolve().parents[3]
# Canonical streaming-LOB output per STATE.md: data/lob/<SYM>/<DATE>/<HH>.parquet.
# (Was data/processed/panels/daily -- which polluted the daily-panel namespace
# with per-symbol subdirs AND did not match the documented consumer path.)
LOB_DIR = ROOT / "data" / "lob"
LOB_DIR.mkdir(parents=True, exist_ok=True)

WS_BASE = "wss://stream.binance.com:9443"


def parse_depth_msg(msg_data: dict) -> dict:
    """Parse a single @depth5 stream message into row format.

    Binance @depth5 schema:
      {"lastUpdateId": ..., "bids": [["price","qty"], ...], "asks": [["price","qty"], ...]}
    Combined stream wraps it: {"stream": "btcusdt@depth5", "data": {...}}
    """
    if "data" in msg_data:
        d = msg_data["data"]
    else:
        d = msg_data
    bids = d.get("bids", [])[:5]
    asks = d.get("asks", [])[:5]
    while len(bids) < 5:
        bids.append(["0", "0"])
    while len(asks) < 5:
        asks.append(["0", "0"])
    return {
        "ts_ms": int(time.time() * 1000),
        "update_id": int(d.get("lastUpdateId", 0)),
        **{f"bid_p{i+1}": float(bids[i][0]) for i in range(5)},
        **{f"bid_q{i+1}": float(bids[i][1]) for i in range(5)},
        **{f"ask_p{i+1}": float(asks[i][0]) for i in range(5)},
        **{f"ask_q{i+1}": float(asks[i][1]) for i in range(5)},
    }


class LOBCollector:
    def __init__(self, symbols: list[str], flush_interval_s: float = 60.0):
        self.symbols = [s.upper() for s in symbols]
        self.flush_interval_s = flush_interval_s
        self.buffer: dict[str, list[dict]] = {s: [] for s in self.symbols}
        self.last_flush = time.monotonic()
        self.shutdown = False

    def _stream_url(self) -> str:
        streams = "/".join(f"{s.lower()}@depth5@100ms" for s in self.symbols)
        return f"{WS_BASE}/stream?streams={streams}"

    def _output_path(self, symbol: str) -> Path:
        now = datetime.now(timezone.utc)
        out = LOB_DIR / symbol / now.strftime("%Y-%m-%d")
        out.mkdir(parents=True, exist_ok=True)
        return out / f"{now.strftime('%H')}.parquet"

    def flush_buffers(self) -> None:
        # KNOWN PERF DEBT (CDAP A3_CONCAT_IN_LOOP, flagged 2026-05-22):
        # With flush_interval_s=60 and per-hour output path (HH.parquet),
        # each flush within the same hour re-reads + concats + rewrites
        # the entire hour's accumulated data → O(N^2) over the hour.
        # Tolerable for current LOB collector load (~30k rows/hour);
        # canonical fix path = per-flush timestamped files (HH_MM_SS.parquet)
        # + separate hourly consolidation job, deferred to dedicated refactor.
        for symbol, rows in self.buffer.items():
            if not rows:
                continue
            df = pl.DataFrame(rows)
            out = self._output_path(symbol)
            try:
                if out.exists():
                    existing = pl.read_parquet(out)
                    combined = pl.concat([existing, df])
                else:
                    combined = df
                # Atomic: write tmp then os.replace, so a kill mid-write can't
                # corrupt the hour's accumulated LOB file.
                tmp = out.with_suffix(".parquet.tmp")
                combined.write_parquet(tmp)
                os.replace(str(tmp), str(out))
                print(f"[lob] flushed {len(rows):>5} rows -> {out.name} ({symbol})", flush=True)
            except Exception as e:
                print(f"[lob] flush failed for {symbol}: {e}", flush=True)
            self.buffer[symbol] = []
        self.last_flush = time.monotonic()

    async def run(self, duration_min: float | None = None) -> None:
        url = self._stream_url()
        end_at = time.monotonic() + duration_min * 60 if duration_min else None
        print(f"[lob] connecting to {url}", flush=True)
        attempt = 0
        while not self.shutdown:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    attempt = 0
                    print(f"[lob] connected. flushing every {self.flush_interval_s}s", flush=True)
                    while not self.shutdown:
                        if end_at and time.monotonic() >= end_at:
                            print("[lob] duration reached", flush=True)
                            self.shutdown = True
                            break
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=self.flush_interval_s)
                        except asyncio.TimeoutError:
                            self.flush_buffers()
                            continue
                        data = json.loads(msg)
                        stream = data.get("stream", "")
                        symbol = stream.split("@")[0].upper()
                        if symbol not in self.buffer:
                            continue
                        row = parse_depth_msg(data)
                        self.buffer[symbol].append(row)
                        if time.monotonic() - self.last_flush >= self.flush_interval_s:
                            self.flush_buffers()
            except Exception as e:
                attempt += 1
                wait = min(60, 2 ** attempt)
                print(f"[lob] connection lost: {e}; reconnect in {wait}s", flush=True)
                await asyncio.sleep(wait)
        # Final flush
        self.flush_buffers()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    parser.add_argument("--duration-min", type=float, default=None,
                        help="Stop after this many minutes. Default: run until SIGINT.")
    parser.add_argument("--flush-interval-s", type=float, default=60.0)
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    collector = LOBCollector(symbols, flush_interval_s=args.flush_interval_s)

    def _stop(signum, frame):
        print("\n[lob] shutdown signal", flush=True)
        collector.shutdown = True
    signal.signal(signal.SIGINT, _stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _stop)

    try:
        asyncio.run(collector.run(duration_min=args.duration_min))
    except KeyboardInterrupt:
        pass
    print("[lob] done", flush=True)


if __name__ == "__main__":
    main()
