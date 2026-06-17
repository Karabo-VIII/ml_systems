"""Deribit DVol index ingest (free public API, no key).

DVol = Deribit's implied volatility index. Analogous to VIX but for crypto
options. Computed from full BTC/ETH options chain.

Signal intuition:
    DVol spikes during fear/capitulation → often precedes bottoms (contrarian buy)
    DVol collapses to historical lows → complacency → often precedes volatility (avoid)
    DVol term-structure inversion (front-month > deferred) = short-term stress

Endpoint:
    GET /api/v2/public/get_volatility_index_data?currency={BTC|ETH}&resolution=1D
         &start_timestamp=...&end_timestamp=...

Daily resolution. Response: [ts_ms, open, high, low, close] per bar.
Max history: several years (we request from 2021-01-01).

Output:
    data/frontier/dvol/dvol_daily.parquet
        columns: date, asset, dvol_open, dvol_high, dvol_low, dvol_close
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "data" / "processed" / "panels" / "daily"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "dvol_daily.parquet"


def fetch_dvol(currency: str, start_ms: int, end_ms: int) -> list[list]:
    url = ("https://www.deribit.com/api/v2/public/get_volatility_index_data?"
           + urllib.parse.urlencode({
               "currency": currency,
               "start_timestamp": start_ms,
               "end_timestamp": end_ms,
               "resolution": "1D",
           }))
    req = urllib.request.Request(url, headers={"User-Agent": "v4-frontier/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
        return data.get("result", {}).get("data", [])
    except Exception as e:
        print(f"  [{currency}] failed: {e}")
        return []


def main():
    # Fetch from 2021-01-01 to today
    start_ms = int(pd.Timestamp("2021-01-01").timestamp() * 1000)
    end_ms = int(time.time() * 1000)

    frames = []
    # Deribit returns max ~1 year per request; paginate in 12-month chunks
    for currency in ["BTC", "ETH"]:
        rows = []
        cur = start_ms
        while cur < end_ms:
            chunk_end = min(cur + int(1.5 * 365 * 86400 * 1000), end_ms)  # 1.5y chunks
            batch = fetch_dvol(currency, cur, chunk_end)
            if batch:
                rows.extend(batch)
                # advance past last ts
                last_ts = batch[-1][0]
                cur = last_ts + 86400 * 1000
            else:
                break
            time.sleep(0.5)

        if not rows:
            print(f"[{currency}] no data")
            continue

        df = pd.DataFrame(rows, columns=["ts_ms", "dvol_open", "dvol_high", "dvol_low", "dvol_close"])
        df["date"] = pd.to_datetime(df["ts_ms"].apply(lambda _t: _t // 1000 if _t >= 1e15 else _t), unit="ms").dt.normalize()
        df["asset"] = currency
        df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
        print(f"[{currency}] {len(df)} daily rows, range {df['date'].min().date()} -> {df['date'].max().date()}")
        frames.append(df[["date", "asset", "dvol_open", "dvol_high", "dvol_low", "dvol_close"]])

    if frames:
        panel = pd.concat(frames, ignore_index=True)
        panel.to_parquet(OUT_PATH, index=False)
        print(f"\n[dvol] saved: {OUT_PATH} ({len(panel)} rows)")


if __name__ == "__main__":
    main()
