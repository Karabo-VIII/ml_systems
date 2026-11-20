"""Alpha turn-011 helper: fetch daily Binance klines for full U50 universe.

Reuses cycle_gate cache dir so btc_dominance / funding-flip / liq-cascade
probes all share the same daily-kline parquet cache.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.strategy.universe import UNIVERSE_50_LIQUID

CACHE_DIR = ROOT / "logs" / "frontier" / "cycle_gate"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
URL = "https://api.binance.com/api/v3/klines"


def fetch(symbol: str, start: str = "2020-01-01") -> pd.DataFrame:
    start_ms = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    all_rows: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        url = (f"{URL}?symbol={symbol}&interval=1d&startTime={cursor}"
               f"&endTime={end_ms}&limit=1000")
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                rows = json.loads(r.read().decode())
        except Exception as e:
            print(f"  [ERR] {symbol} @ {cursor}: {e}")
            break
        if not rows:
            break
        all_rows.extend(rows)
        last_ts = rows[-1][0]
        if last_ts <= cursor:
            break
        cursor = last_ts + 86_400_000
        if len(rows) < 1000:
            break
        time.sleep(0.1)
    if not all_rows:
        return pd.DataFrame(columns=["date", "close"])
    cols = ["open_ts", "open", "high", "low", "close", "volume",
            "close_ts", "qav", "n_trades", "tb_bav", "tb_qav", "ignore"]
    df = pd.DataFrame(all_rows, columns=cols)
    df["date"] = pd.to_datetime(df["open_ts"], unit="ms").dt.tz_localize(None).dt.normalize()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df[["date", "close"]].drop_duplicates("date").reset_index(drop=True)


def main() -> None:
    # Symbols need an odd mapping -- e.g. 1000SATS -> 1000SATSUSDT on Binance
    missing = []
    fetched = []
    for asset in UNIVERSE_50_LIQUID:
        symbol = f"{asset}USDT"
        cache_file = CACHE_DIR / f"{symbol.lower()}_daily_klines.parquet"
        if cache_file.exists():
            continue
        print(f"fetching {symbol}...", end=" ", flush=True)
        df = fetch(symbol)
        if df.empty:
            missing.append(asset)
            print("EMPTY")
            continue
        df.to_parquet(cache_file)
        fetched.append(asset)
        print(f"{len(df)} days")
    print(f"\nfetched: {len(fetched)}, missing: {len(missing)}")
    if missing:
        print(f"missing assets: {missing}")


if __name__ == "__main__":
    main()
