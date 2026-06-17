"""Cross-exchange daily closes from Coinbase, Bybit, OKX (all free public APIs).

Fetches daily OHLC from 3 exchanges for top assets, then computes cross-exchange
spreads vs Binance spot (from existing spot_klines data).

Signal hypothesis: When spreads widen (e.g. Binance trades +0.5% rich vs
Coinbase), it signals regional demand imbalance — often precedes a move
toward the expensive side. Widening spread = stress / arbitrage opportunity.

Exchanges:
    - Coinbase: /products/{base}-USD/candles (epoch seconds)
    - Bybit: /v5/market/kline (spot, D interval, limit=1000)
    - OKX: /v5/market/candles (1D, limit=300; paginate)

Output:
    data/frontier/spreads/cross_exchange_daily.parquet
        date, asset, binance_close, coinbase_close, bybit_close, okx_close,
        cb_bn_spread_bps, by_bn_spread_bps, ok_bn_spread_bps
"""
from __future__ import annotations
import os

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from pipeline.ingest._manifest import MissingManifest

OUT_DIR = ROOT / "data" / "processed" / "panels" / "daily"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "cross_exchange_daily.parquet"

# Manifest: one entry per (asset, date_str) when ALL 3 venues returned 0 rows.
# Re-check after 30 days (exchange delistings are usually permanent; API
# outages usually resolve within a week, so 30d is safe conservative).
_MANIFEST_ROOT = OUT_DIR / "cross_exchange_manifests"
_mm = MissingManifest(_MANIFEST_ROOT)

UA = "v4-frontier/1.0"


# 2026-05-22 oracle pipeline-progress closure: lazy phase_log helper with
# dual-import fallback (works whether src/ or src/pipeline/ is on sys.path).
def _pl(phase, message, **kw):
    try:
        from progress import phase_log
    except ImportError:
        from pipeline.progress import phase_log
    phase_log("xex", phase, message, **kw)


def fetch_coinbase(asset: str, start: str, end: str) -> pd.DataFrame:
    """Coinbase: max 300 candles per request. Paginate by 300-day chunks."""
    rows = []
    cur = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    while cur < end_dt:
        chunk_end = min(cur + pd.Timedelta(days=299), end_dt)
        url = (f"https://api.exchange.coinbase.com/products/{asset}-USD/candles"
               f"?granularity=86400&start={cur.isoformat()}Z&end={chunk_end.isoformat()}Z")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=20) as r:
                data = json.loads(r.read())
            for row in data:
                # [time, low, high, open, close, volume]
                rows.append({
                    "date": pd.to_datetime(int(row[0]), unit="s").normalize(),
                    "close": float(row[4]),
                })
        except Exception as e:
            print(f"  Coinbase {asset} {cur.date()}: {e}", flush=True)
        cur = chunk_end + pd.Timedelta(days=1)
        time.sleep(0.3)
    return pd.DataFrame(rows).drop_duplicates(subset="date").sort_values("date")


def fetch_bybit(asset: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Bybit paginates via startTime/endTime, limit=1000."""
    rows = []
    cur = start_ms
    while cur < end_ms:
        url = (f"https://api.bybit.com/v5/market/kline?category=spot&symbol={asset}USDT"
               f"&interval=D&start={cur}&end={end_ms}&limit=1000")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=20) as r:
                data = json.loads(r.read())
            lst = data.get("result", {}).get("list", [])
            if not lst:
                break
            for r_row in lst:
                ts = int(r_row[0])
                rows.append({
                    "date": pd.to_datetime(ts, unit="ms").normalize(),
                    "close": float(r_row[4]),
                })
            # Bybit returns newest first; advance past oldest returned
            oldest = int(lst[-1][0])
            if oldest <= cur or len(lst) < 1000:
                break
            cur = oldest + 86400 * 1000
        except Exception as e:
            print(f"  Bybit {asset}: {e}", flush=True)
            break
        time.sleep(0.3)
    return pd.DataFrame(rows).drop_duplicates(subset="date").sort_values("date")


def fetch_okx(asset: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """OKX paginates backward via 'after' (older-than ts), limit=300.

    IMPORTANT: use `bar=1Dutc` (NOT `1D`). Default `1D` closes at 16:00 UTC
    (=00:00 UTC+8 China time), creating ~9% apparent spread vs UTC-aligned
    venues like Binance/Coinbase. `1Dutc` closes at 00:00 UTC.
    """
    rows = []
    after = end_ms
    for _ in range(50):
        url = (f"https://www.okx.com/api/v5/market/history-candles?"
               f"instId={asset}-USDT&bar=1Dutc&after={after}&limit=300")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=20) as r:
                data = json.loads(r.read())
            lst = data.get("data", [])
            if not lst:
                break
            for r_row in lst:
                ts = int(r_row[0])
                if ts < start_ms:
                    continue
                rows.append({
                    "date": pd.to_datetime(ts, unit="ms").normalize(),
                    "close": float(r_row[4]),
                })
            oldest = int(lst[-1][0])
            if oldest <= start_ms or len(lst) < 300:
                break
            after = oldest
        except Exception as e:
            print(f"  OKX {asset}: {e}", flush=True)
            break
        time.sleep(0.3)
    return pd.DataFrame(rows).drop_duplicates(subset="date").sort_values("date")


def main():
    import argparse
    import sys as _sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Delete existing panel and rebuild from venue APIs.")
    ap.add_argument("--recheck-missing", action="store_true",
                    help="Bypass the confirmed_missing manifest and re-attempt "
                         "every previously all-venue-zero date range.")
    args, _ = ap.parse_known_args()

    if args.force and OUT_PATH.exists():
        OUT_PATH.unlink()
        print(f"[spreads] [force] deleted prior {OUT_PATH.name}; rebuilding",
              flush=True)

    assets = ["BTC", "ETH", "SOL", "XRP", "DOGE"]  # top-5 with highest cross-exchange liquidity
    # 2026-05-24: start moved 2024-01-01 -> 2023-01-01 to match
    # Binance Vision earliest-availability anchor (was silently skipping ~1y
    # of cross-exchange spread history on fresh rebuilds).
    start = "2023-01-01"
    end = pd.Timestamp.now().strftime("%Y-%m-%d")
    start_ms = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms = int(pd.Timestamp(end).timestamp() * 1000)

    # For the manifest: use a per-asset "date range" key (start..end)
    # so that if ALL venues fail for an asset we skip on next run.
    # Key format: "<start>:<end>" -- encodes the exact request range.
    range_key = f"{start}:{end}"

    frames = []
    n_fully_failed = 0
    for asset in assets:
        # confirmed_missing: skip asset if the full range was confirmed missing recently.
        if not args.recheck_missing and _mm.is_known_missing(asset, range_key):
            _pl("BUILD", f"{asset}: confirmed-missing (all venues) -- skipping "
                f"(use --recheck-missing to retry)")
            n_fully_failed += 1
            # Still need a frame so the panel doesn't drop the asset entirely;
            # inject an empty frame so downstream merge stays consistent.
            frames.append(pd.DataFrame(columns=["date", "coinbase_close",
                                                  "bybit_close", "okx_close", "asset"]))
            continue
        _pl("BUILD", f"{asset}: fetching 3 exchanges...")
        cb = fetch_coinbase(asset, start, end).rename(columns={"close": "coinbase_close"})
        by = fetch_bybit(asset, start_ms, end_ms).rename(columns={"close": "bybit_close"})
        ok = fetch_okx(asset, start_ms, end_ms).rename(columns={"close": "okx_close"})
        print(f"  {asset}: CB {len(cb)}, Bybit {len(by)}, OKX {len(ok)}")
        if len(cb) == 0 and len(by) == 0 and len(ok) == 0:
            # All three venues silently failed for this asset (B6: surface it).
            _pl("WARN", f"{asset}: all 3 venues returned 0 rows -- marking manifest")
            _mm.mark_missing(asset, range_key)
            n_fully_failed += 1
        else:
            # At least one venue succeeded -- clear any stale manifest entry.
            _mm.unmark_missing(asset, range_key)
        merged = cb.merge(by, on="date", how="outer").merge(ok, on="date", how="outer")
        merged["asset"] = asset
        frames.append(merged)

    panel = pd.concat(frames, ignore_index=True).sort_values(["asset", "date"]).reset_index(drop=True)

    # Merge with Binance spot (from Item A4 spot klines)
    binance = pd.read_parquet(ROOT / "data" / "processed" / "panels" / "daily" / "spot_klines_daily.parquet")
    binance = binance.rename(columns={"spot_close": "binance_close"})[["date", "asset", "binance_close"]]

    full = panel.merge(binance, on=["date", "asset"], how="left")
    full["cb_bn_spread_bps"] = (full["coinbase_close"] - full["binance_close"]) / full["binance_close"] * 10000
    full["by_bn_spread_bps"] = (full["bybit_close"] - full["binance_close"]) / full["binance_close"] * 10000
    full["ok_bn_spread_bps"] = (full["okx_close"] - full["binance_close"]) / full["binance_close"] * 10000

    # G-AUDIT-020: atomic write (was: direct overwrite -> partial files on crash).
    tmp = OUT_PATH.with_suffix(".parquet.tmp")
    full.to_parquet(tmp, index=False)
    if OUT_PATH.exists():
        OUT_PATH.unlink()
    os.replace(str(tmp), str(OUT_PATH))  # atomic overwrite (Windows-safe)
    print(f"\n[spreads] saved: {OUT_PATH} ({len(full)} rows)")
    print("\n[spread summary per asset]:")
    for a in full["asset"].unique():
        sub = full[full["asset"] == a]
        cb = sub["cb_bn_spread_bps"].dropna()
        print(f"  {a}: CB-BN mean {cb.mean():+.1f} bps std {cb.std():.1f} bps (n={len(cb)})")

    if n_fully_failed == len(assets):
        # All assets had every venue fail -> non-zero exit so orchestrator
        # surfaces it instead of accepting an empty/stale panel.
        _sys.exit(2)
    if n_fully_failed > 0:
        _sys.exit(1)


if __name__ == "__main__":
    main()
