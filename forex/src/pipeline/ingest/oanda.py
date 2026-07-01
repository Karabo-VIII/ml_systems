"""OANDA v20 candle ingester — the keyed live/recent secondary source.

Free with an OANDA practice (demo) account: create a personal access token and
export it as OANDA_API_TOKEN. Pulls OHLC candles (bid + ask + mid) at S5..D
granularity, paginating past the 5000-candles/request cap. Used for recent /
live data and forward paper-trading, complementing the deep Dukascopy archive.

Output: data/processed/bars/<cadence>/<pair>_<cadence>_<YYYYMMDD>.parquet — the
SAME bar schema build_bars.py produces, so OANDA candles are a drop-in for the
recent tail where a tick backfill is unnecessary.

Requires no smoke-test key to import; raises a clear error if the token is
absent when actually fetching.

    export OANDA_API_TOKEN=...   # from OANDA practice account
    python -m pipeline.ingest.oanda --pairs EURUSD --granularity 1h --start 2026-06-01
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import requests

_PIPELINE_PARENT = str(Path(__file__).resolve().parents[2])
if _PIPELINE_PARENT not in sys.path:
    sys.path.insert(0, _PIPELINE_PARENT)

from pipeline import layout
from pipeline.parquet_io import atomic_write_parquet

__contract__ = {
    "kind": "pipeline_stage",
    "stage": "fetch_oanda",
    "inputs": {"args": ["--pairs", "--granularity", "--start", "--end"],
               "env": ["OANDA_API_TOKEN"],
               "config_keys": ["source.oanda.practice_host"]},
    "outputs": {"files": "data/processed/bars/<cadence>/<pair>_<cadence>_<YYYYMMDD>.parquet",
                "columns": ["timestamp", "open", "high", "low", "close",
                            "bid_close", "ask_close", "spread", "tick_count"]},
    "invariants": {"paginate_past_5000": True, "underscore_instrument": True,
                   "token_from_env_never_committed": True, "atomic_write": True},
    "rationale": "Keyed live/recent secondary; recent tail without a tick backfill.",
}

PRACTICE_HOST = "https://api-fxpractice.oanda.com"
LIVE_HOST = "https://api-fxtrade.oanda.com"
MAX_COUNT = 5000

# OANDA granularity <-> our cadence subfolder name.
GRAN_TO_CADENCE = {"M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m",
                   "H1": "1h", "H4": "4h", "D": "1d"}
CADENCE_TO_GRAN = {v: k for k, v in GRAN_TO_CADENCE.items()}
# Seconds per granularity, for pagination stepping.
GRAN_SECONDS = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800,
                "H1": 3600, "H4": 14400, "D": 86400}


def _token() -> str:
    tok = os.environ.get("OANDA_API_TOKEN")
    if not tok:
        raise RuntimeError(
            "OANDA_API_TOKEN not set. Create a free OANDA practice account, "
            "generate a personal access token (My Account -> Manage API Access), "
            "and `export OANDA_API_TOKEN=...`.")
    return tok


def _rfc3339(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_candles(pair: str, granularity: str, start: datetime, end: datetime, *,
                  host: str = PRACTICE_HOST, session: requests.Session | None = None) -> pl.DataFrame:
    """Fetch bid+ask+mid OHLC candles for [start, end), paginating past 5000."""
    n = layout.normalize_pair(pair)
    sess = session or requests.Session()
    sess.headers.update({"Authorization": f"Bearer {_token()}"})
    url = f"{host}/v3/instruments/{n.oanda}/candles"
    step = GRAN_SECONDS[granularity] * MAX_COUNT
    rows: list[dict] = []
    cur = start
    while cur < end:
        params = {
            "granularity": granularity,
            "from": _rfc3339(cur),
            "to": _rfc3339(min(end, datetime.fromtimestamp(cur.timestamp() + step, tz=timezone.utc))),
            "price": "MBA",           # mid + bid + ask
            "count": MAX_COUNT,
        }
        r = sess.get(url, params=params, timeout=30)
        r.raise_for_status()
        candles = r.json().get("candles", [])
        for c in candles:
            if not c.get("complete", False):
                continue
            ts = int(datetime.strptime(c["time"][:19], "%Y-%m-%dT%H:%M:%S")
                     .replace(tzinfo=timezone.utc).timestamp() * 1000)
            mid, bid, ask = c.get("mid", {}), c.get("bid", {}), c.get("ask", {})
            rows.append({
                "timestamp": ts,
                "open": float(mid["o"]), "high": float(mid["h"]),
                "low": float(mid["l"]), "close": float(mid["c"]),
                "bid_close": float(bid["c"]), "ask_close": float(ask["c"]),
                "spread": float(ask["c"]) - float(bid["c"]),
                "tick_count": int(c.get("volume", 0)),
            })
        cur = datetime.fromtimestamp(cur.timestamp() + step, tz=timezone.utc)
        time.sleep(0.1)
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).unique(subset=["timestamp"], keep="last").sort("timestamp")


def ingest(pair: str, granularity: str, start: datetime, end: datetime, *,
           host: str = PRACTICE_HOST) -> dict:
    df = fetch_candles(pair, granularity, start, end, host=host)
    if df.is_empty():
        return {"status": "empty", "pair": pair, "bars": 0}
    cadence = GRAN_TO_CADENCE[granularity]
    end_date = datetime.fromtimestamp(df["timestamp"].max() / 1000, tz=timezone.utc).date()
    out = layout.bars_path(pair, cadence, latest_date=end_date)
    atomic_write_parquet(df, out, required_cols={"timestamp", "open", "close", "spread"})
    return {"status": "ok", "pair": pair, "cadence": cadence, "bars": len(df),
            "path": str(out)}


def main() -> None:
    from pipeline import cli
    ap = argparse.ArgumentParser(description="OANDA v20 candle fetcher (keyed)")
    cli.add_standard_args(ap, default_workers=2, default_start="2024-01-01")
    ap.add_argument("--granularity", default="1h", choices=list(CADENCE_TO_GRAN),
                    help="Bar cadence to fetch (default 1h).")
    args = ap.parse_args()
    pairs = cli.resolve_pairs(args, stage_name="oanda")
    gran = CADENCE_TO_GRAN[args.granularity]
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    for p in pairs:
        try:
            res = ingest(p, gran, start, end)
            print(f"[oanda] {p}: {res}", flush=True)
        except Exception as e:
            print(f"[oanda] {p} FAIL: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
