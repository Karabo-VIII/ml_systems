"""Catastrophic-correctness gate for the forex pipeline.

A fast, high-signal check of the invariants that, if broken, silently corrupt
everything downstream. Runs on synthetic data always, and additionally validates
any REAL bars present on disk. Complements run_tests.py (unit) and
scripts/check_contracts.py (static). Exit 0 clean, 1 on any violation.

    python run_invariants_check.py
"""
from __future__ import annotations

import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import polars as pl  # noqa: E402
from pipeline import layout, build_bars, data_quality  # noqa: E402
from pipeline.ingest import dukascopy  # noqa: E402
from pipeline.universe_loader import UniverseLoader  # noqa: E402

_fails: list[str] = []


def check(name: str, cond: bool):
    print(f"  {'OK  ' if cond else 'FAIL'} {name}")
    if not cond:
        _fails.append(name)


def main() -> int:
    print("[invariants] synthetic checks")
    # 1. bi5 month zero-indexed (June -> 05)
    url = dukascopy._hour_url("EURUSD", datetime(2026, 6, 30, 13, tzinfo=timezone.utc))
    check("dukascopy URL month is zero-indexed (June->05)", "/2026/05/30/" in url)

    # 2. bi5 record parse: ask>bid, correct scaling
    buf = struct.pack(">IIIff", 1000, 114000, 113990, 1.0, 1.0)
    cols = dukascopy.parse_hour_bytes(buf, 0, 100000.0)
    check("bi5 parse ask>bid and /divisor", cols["ask"][0] > cols["bid"][0]
          and abs(cols["ask"][0] - 1.14) < 1e-9)

    # 3. universe metadata sanity
    U = UniverseLoader.load()
    check("universe u7=7 u28=28", len(U.list("u7")) == 7 and len(U.list("u28")) == 28)
    check("JPY pip=0.01 divisor=1000", U.pip("USDJPY") == 0.01 and U.price_divisor("USDJPY") == 1000.0)

    # 4. resample OHLC integrity on synthetic ticks
    minute = 1_700_000_000_000 // 60000 * 60000
    ticks = pl.DataFrame({
        "timestamp": [minute + 1000, minute + 30000, minute + 59000],
        "bid": [1.10, 1.101, 1.1005], "ask": [1.1001, 1.1011, 1.1006],
        "bid_volume": [1.0] * 3, "ask_volume": [1.0] * 3,
    }).with_columns([
        ((pl.col("bid") + pl.col("ask")) / 2).alias("mid"),
        (pl.col("ask") - pl.col("bid")).alias("spread"),
        pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("dt"),
    ])
    bars = build_bars._resample(ticks, "1m")
    ok = (len(bars) == 1 and bars["high"][0] >= bars["low"][0]
          and bars["ask_close"][0] > bars["bid_close"][0] and bars["tick_count"][0] == 3)
    check("resample OHLC integrity + ask>bid + tick_count", ok)

    # 5. weekend-gap classification
    fri = int(datetime(2026, 6, 26, 21, tzinfo=timezone.utc).timestamp() * 1000)
    sun = int(datetime(2026, 6, 28, 21, tzinfo=timezone.utc).timestamp() * 1000)
    check("weekend gap detected; intraday gap not",
          data_quality._is_weekend_gap(fri, sun)
          and not data_quality._is_weekend_gap(fri, fri + 2 * 3_600_000))

    # 6. REAL data (if present): no negative spread, OHLC integrity, price in band
    print("[invariants] real-data checks (if bars present)")
    checked_real = 0
    for pair in U.list("u7"):
        for cad in ("1h", "1d"):
            path = layout.bars_latest(pair, cad)
            if path is None or not path.exists():
                continue
            m = data_quality.bar_metrics(pair, cad, U)
            checked_real += 1
            neg = m.get("negative_spread_bars", 0)
            integ = m.get("ohlc_integrity_violations", 0)
            check(f"{pair} {cad}: no neg spread + OHLC ok + not fail "
                  f"(status={m['status']})", neg == 0 and integ == 0 and m["status"] != "fail")
    if checked_real == 0:
        print("  (no real bars on disk yet -- synthetic checks only)")

    print(f"\n[invariants] {'ALL OK' if not _fails else str(len(_fails)) + ' FAILED: ' + ', '.join(_fails)}")
    return 0 if not _fails else 1


if __name__ == "__main__":
    sys.exit(main())
