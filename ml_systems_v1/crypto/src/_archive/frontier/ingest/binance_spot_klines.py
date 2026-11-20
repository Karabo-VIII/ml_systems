"""Binance spot daily klines from data.binance.vision (free, no key).

Monthly archive at:
    data/spot/monthly/klines/<SYMBOL>/1d/<SYMBOL>-1d-YYYY-MM.zip

CSV columns:
    open_time, open, high, low, close, volume, close_time, quote_volume,
    num_trades, taker_buy_base, taker_buy_quote, ignore

We just need close prices to compute spot-perp basis.

Output:
    data/frontier/basis/spot_klines_daily.parquet
        columns: date, asset, spot_close
"""
from __future__ import annotations

import concurrent.futures
import csv
import io
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "data" / "processed" / "panels" / "daily"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "spot_klines_daily.parquet"

BASE = "https://data.binance.vision/data/spot/monthly/klines"
UA = "v4-frontier/1.0"


def fetch_month(symbol: str, ym: str, retries: int = 3) -> list[dict]:
    url = f"{BASE}/{symbol}/1d/{symbol}-1d-{ym}.zip"
    import time
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                content = r.read()
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                with z.open(z.namelist()[0]) as f:
                    text = f.read().decode()
            rows = []
            for line in text.splitlines():
                parts = line.split(",")
                if len(parts) < 7:
                    continue
                try:
                    ts = int(parts[0])
                    close = float(parts[4])
                except ValueError:
                    continue  # skip header if present
                # Binance switched ms -> microseconds at some point in 2025.
                # Autodetect: >10^15 implies microseconds; else ms.
                unit = "us" if ts > 1e15 else "ms"
                rows.append({
                    "date": pd.to_datetime(ts, unit=unit).normalize(),
                    "asset": symbol.replace("USDT", ""),
                    "spot_close": close,
                })
            return rows
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return []
            time.sleep(1 + i)
        except Exception:
            time.sleep(1 + i)
    return []


def fetch_asset(symbol: str, start: str = "2020-01", end: str | None = None) -> pd.DataFrame:
    start_d = pd.Timestamp(start + "-01")
    end_d = pd.Timestamp.now() if end is None else pd.Timestamp(end + "-01")
    months = pd.date_range(start_d, end_d, freq="MS").strftime("%Y-%m").tolist()
    all_rows = []
    ok = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch_month, symbol, ym): ym for ym in months}
        for fut in concurrent.futures.as_completed(futs):
            rows = fut.result()
            if rows:
                all_rows.extend(rows)
                ok += 1
    df = pd.DataFrame(all_rows).drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    print(f"[{symbol}] fetched {ok}/{len(months)} months = {len(df)} daily rows")
    return df


def main():
    # Top 10 spot-listed assets (same as our chimera universe)
    assets = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
              "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]

    frames = []
    for a in assets:
        df = fetch_asset(a, start="2020-01")
        if len(df) > 0:
            frames.append(df)

    panel = pd.concat(frames, ignore_index=True).sort_values(["asset", "date"]).reset_index(drop=True)
    panel.to_parquet(OUT_PATH, index=False)
    print(f"\n[spot] saved: {OUT_PATH} ({len(panel)} rows, {panel['asset'].nunique()} assets)")
    print(f"  date range: {panel['date'].min().date()} -> {panel['date'].max().date()}")


if __name__ == "__main__":
    main()
