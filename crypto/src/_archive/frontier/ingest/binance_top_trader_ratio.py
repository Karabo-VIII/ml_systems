"""Binance top-trader long/short ratio ingest (free public API, no key).

Endpoints:
    /futures/data/topLongShortAccountRatio  (ratio of TOP accounts long vs short)
    /futures/data/topLongShortPositionRatio (ratio of TOP positions long vs short)
    /futures/data/globalLongShortAccountRatio (ratio of ALL accounts)

Top = top 20% of accounts by balance (per Binance). This is the closest free
public proxy for "what do smart/large traders do".

Signal intuition (from FinTwit folklore + published work on retail positioning):
    topLongShortPositionRatio
    - Extreme LONG (>2.0): top traders are heavily long → can precede rally
      (if persistent) OR reversion (if crowded-long = unwind risk)
    - Extreme SHORT (<0.5): top traders heavily short → capitulation signal
      or reversion up
    - Changes in ratio (d/dt) often precede price moves by 12-48h

Output:
    data/frontier/leaderboard/top_trader_ratio_daily.parquet
        columns: date, asset, long_acct_ratio, long_pos_ratio, global_acct_ratio
        (all as fractions; ratio column = long / short)

Rate-limited: 1 req/0.5s = safe for free tier (no key = 50 req/5min limit).
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "data" / "processed" / "panels" / "daily"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "top_trader_ratio_daily.parquet"

BASE = "https://fapi.binance.com/futures/data"
UA = "v4_crypto_stystem-frontier/1.0"

ENDPOINTS = {
    "long_acct_ratio": "topLongShortAccountRatio",
    "long_pos_ratio": "topLongShortPositionRatio",
    "global_acct_ratio": "globalLongShortAccountRatio",
}


def _fetch(endpoint: str, symbol: str, period: str, start_ms: int, end_ms: int, limit: int = 500) -> list:
    params = {
        "symbol": symbol,
        "period": period,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }
    url = f"{BASE}/{endpoint}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"  [WARN] {symbol} {endpoint}: {e}")
        return []


def _parse_rows(rows: list, col: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["timestamp", col])
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_numeric(df["timestamp"])
    # longShortRatio is long/short; also compute long_share = long_account / (long+short)
    if "longShortRatio" in df.columns:
        df[col] = pd.to_numeric(df["longShortRatio"])
    if "longAccount" in df.columns:
        df[col + "_long_share"] = pd.to_numeric(df["longAccount"])
    elif "longPosition" in df.columns:
        df[col + "_long_share"] = pd.to_numeric(df["longPosition"])
    return df[["timestamp"] + [c for c in df.columns if c.startswith(col)]]


def fetch_asset(symbol: str, start_date: str = "2022-01-01") -> pd.DataFrame:
    start_ms = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_ms = int(pd.Timestamp.now().timestamp() * 1000)
    asset_dfs = {}
    for col, endpoint in ENDPOINTS.items():
        # Paginate: 500 days per request
        cur = start_ms
        rows = []
        while cur < end_ms:
            chunk = _fetch(endpoint, symbol, "1d", cur, end_ms, limit=500)
            if not chunk:
                break
            rows.extend(chunk)
            last_ts = int(chunk[-1]["timestamp"])
            if last_ts <= cur:
                break
            cur = last_ts + 86400 * 1000
            time.sleep(0.3)
            if len(chunk) < 500:
                break
        asset_dfs[col] = _parse_rows(rows, col).drop_duplicates("timestamp")
    # Merge all three
    merged = None
    for col, df in asset_dfs.items():
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on="timestamp", how="outer")
    if merged is None or len(merged) == 0:
        return pd.DataFrame()
    merged["date"] = pd.to_datetime(merged["timestamp"].apply(lambda _t: _t // 1000 if _t >= 1e15 else _t), unit="ms").dt.normalize()
    merged["asset"] = symbol.replace("USDT", "")
    return merged


def main():
    # Assets: use the 54 assets with funding data (already know these have perp support)
    fp_dir = ROOT / "data" / "raw"
    assets = sorted([
        d.name for d in fp_dir.iterdir()
        if d.is_dir() and (d / "funding").is_dir() and any((d / "funding").iterdir())
    ])
    print(f"[top_trader] {len(assets)} assets to fetch")

    all_rows = []
    n_failed = 0
    n_empty = 0
    for i, a in enumerate(assets, 1):
        print(f"[{i}/{len(assets)}] {a}...", flush=True)
        try:
            df = fetch_asset(a, start_date="2022-01-01")
        except Exception as e:
            print(f"  FAILED: {e}")
            n_failed += 1
            continue
        if len(df) == 0:
            print(f"  empty, skipping")
            n_empty += 1
            continue
        print(f"  {len(df)} days, range {df['date'].min().date()} -> {df['date'].max().date()}")
        all_rows.append(df)
        time.sleep(0.2)

    n_expected = len(assets)
    n_ok = len(all_rows)
    print(f"\n[top_trader] ingest summary: {n_ok}/{n_expected} ok, "
          f"{n_failed} failed, {n_empty} empty")

    if not all_rows:
        # Hard fail: orchestrator must see non-zero exit so downstream
        # (top_trader_signals -> v51 chimera) doesn't silently use stale data.
        print("[top_trader] HARD FAIL: no data fetched", flush=True)
        sys.exit(2)

    out = pd.concat(all_rows, ignore_index=True)
    out = out.sort_values(["asset", "date"]).drop_duplicates(["asset", "date"]).reset_index(drop=True)

    # G-AUDIT-020: atomic-tmp-rename + column-name verify (RED TEAM contract)
    _tmp = OUT_PATH.with_suffix(".parquet.tmp")
    out.to_parquet(_tmp, index=False)
    import pyarrow.parquet as _pq
    _written = set(_pq.read_schema(_tmp).names)
    if "date" not in _written or "asset" not in _written:
        _tmp.unlink(missing_ok=True)
        raise ValueError(
            f"top_trader_ratio: missing date/asset cols (got {sorted(_written)[:5]}...)")
    if OUT_PATH.exists():
        OUT_PATH.unlink()
    _tmp.rename(OUT_PATH)
    print(f"\n[top_trader] saved: {OUT_PATH}")
    print(f"  total rows: {len(out)}, assets: {out['asset'].nunique()}, dates: {out['date'].nunique()}")
    # Partial-coverage warning — orchestrator can use rc=1 to flag for review.
    if n_ok < n_expected * 0.5:
        print(f"[top_trader] WARN: only {n_ok}/{n_expected} assets succeeded "
              f"(<50% coverage); flagging as warn", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
