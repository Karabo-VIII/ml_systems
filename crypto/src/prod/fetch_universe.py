#!/usr/bin/env python
"""
Fetch Universe Data
=====================

Fetches recent aggTrade data for the screened universe of assets.
Uses Binance public REST API (no auth needed for aggTrades).

This is separate from the historical pipeline (src/pipeline/fetch_all.py)
because:
  1. We only need 30 days of data (not years)
  2. We fetch for 20-50 assets (not just 10)
  3. The data goes to data/prod/ (not data/raw/)
  4. It's designed for daily refresh, not one-time historical load

Usage:
    # Fetch last 30 days for screened universe
    python src/prod/fetch_universe.py

    # Fetch last 7 days for specific assets
    python src/prod/fetch_universe.py --days 7 --assets btcusdt ethusdt solusdt

    # Fetch for top 50 screened assets
    python src/prod/fetch_universe.py --from-screener --days 30
"""
import argparse
import json
import sys
import time
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prod.config import STATE_DIR, ACTIVE_ASSETS

logger = logging.getLogger("prod.fetch")

PROD_DATA_DIR = PROJECT_ROOT / "data" / "prod"


def fetch_aggtrades_window(exchange, binance_symbol: str,
                            start_ms: int, end_ms: int) -> list:
    """Fetch aggTrades for a short time window using Binance REST directly.

    Uses exchange.request() to hit the raw aggTrades endpoint with
    startTime/endTime, which works reliably for windows <= 1 hour.
    Returns list of {timestamp, price, qty, is_buyer_maker} dicts.
    """
    try:
        params = {
            "symbol": binance_symbol,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 1000,
        }
        raw = exchange.exchange.publicGetAggTrades(params)
        return [
            {
                "timestamp": int(t["T"]),
                "price": float(t["p"]),
                "qty": float(t["q"]),
                "is_buyer_maker": t["m"],
            }
            for t in raw
        ]
    except Exception as e:
        logger.debug("Chunk fetch failed for %s: %s", binance_symbol, e)
        return []


def fetch_asset_data(exchange, symbol: str, days: int = 30,
                      output_dir: Path = None):
    """Fetch recent aggTrades for one asset using chunked time windows.

    Strategy: chunk by 1-hour windows for bulk history, then 5-minute
    windows for the last 24 hours (more granular near the present).
    This avoids Binance's rejection of large time ranges on busy pairs.
    """
    if output_dir is None:
        output_dir = PROD_DATA_DIR

    asset_clean = symbol.replace("/", "").lower()
    binance_sym = symbol.replace("/", "")  # BTC/USDT -> BTCUSDT

    out_dir = output_dir / asset_clean
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "recent_trades.parquet"

    # Check if we already have recent data
    if out_file.exists():
        try:
            import polars as pl
            existing = pl.read_parquet(out_file)
            if len(existing) > 0:
                last_ts = existing["timestamp"].max()
                last_date = datetime.fromtimestamp(last_ts / 1000, timezone.utc)
                age_hours = (datetime.now(timezone.utc) - last_date).total_seconds() / 3600
                if age_hours < 6:
                    logger.info("%s: data is %.1fh old, skipping", asset_clean, age_hours)
                    return len(existing)
        except Exception:
            pass

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)

    # Chunking strategy:
    # - Days 1 to (days-1): 1-hour chunks (24 chunks/day)
    # - Last 24 hours: 5-minute chunks (288 chunks)
    one_hour_ms = 3_600_000
    five_min_ms = 300_000
    one_day_ms = 86_400_000

    boundary_ms = end_ms - one_day_ms  # 24h ago

    # Build chunk list
    chunks = []
    # Bulk: 1-hour chunks from start to 24h ago
    t = start_ms
    while t < boundary_ms:
        chunk_end = min(t + one_hour_ms, boundary_ms)
        chunks.append((t, chunk_end))
        t = chunk_end

    # Granular: 5-minute chunks for last 24h
    t = boundary_ms
    while t < end_ms:
        chunk_end = min(t + five_min_ms, end_ms)
        chunks.append((t, chunk_end))
        t = chunk_end

    total_chunks = len(chunks)
    all_trades = []
    empty_chunks = 0

    # Resume: check if partial file exists (from prior interrupted run)
    partial_file = out_dir / "recent_trades_partial.parquet"
    resume_from_ms = start_ms
    if partial_file.exists():
        try:
            import polars as pl
            partial_df = pl.read_parquet(partial_file)
            if len(partial_df) > 0:
                resume_from_ms = int(partial_df["timestamp"].max()) + 1
                all_trades = partial_df.to_dicts()
                # Skip chunks we already have
                chunks = [(cs, ce) for cs, ce in chunks if ce > resume_from_ms]
                total_chunks = len(chunks) + len(all_trades) // 100  # approximate
                logger.info("%s: resuming from %d existing trades, %d chunks remaining",
                           asset_clean, len(all_trades), len(chunks))
        except Exception:
            pass

    import polars as pl

    for ci, (cs, ce) in enumerate(chunks):
        trades = fetch_aggtrades_window(exchange, binance_sym, cs, ce)

        if trades:
            all_trades.extend(trades)
        else:
            empty_chunks += 1

        # Progress every 50 chunks
        if (ci + 1) % 50 == 0 or ci == len(chunks) - 1:
            pct = (ci + 1) / len(chunks) * 100
            logger.info("%s: %d/%d chunks (%.0f%%), %d trades",
                       asset_clean, ci + 1, len(chunks), pct, len(all_trades))

        # Incremental save every 200 chunks (crash-safe resume)
        if (ci + 1) % 200 == 0 and all_trades:
            try:
                df_partial = pl.DataFrame(all_trades).sort("timestamp")
                df_partial = df_partial.unique(subset=["timestamp", "price", "qty"],
                                               maintain_order=True)
                df_partial.write_parquet(partial_file)
            except Exception:
                pass

        # Rate limit: ~10 req/s is safe for public Binance
        if (ci + 1) % 8 == 0:
            time.sleep(0.5)

    if not all_trades:
        logger.warning("%s: no trades fetched (%d empty chunks)", asset_clean, empty_chunks)
        return 0

    # Save final parquet
    try:
        df = pl.DataFrame(all_trades)
        df = df.sort("timestamp")
        df = df.unique(subset=["timestamp", "price", "qty"], maintain_order=True)
        df.write_parquet(out_file)
        # Clean up partial file
        if partial_file.exists():
            partial_file.unlink()
        logger.info("%s: saved %d trades (%d chunks, %d empty)",
                    asset_clean, len(df), len(chunks), empty_chunks)
        return len(df)
    except Exception as e:
        logger.error("%s: failed to save: %s", asset_clean, e)
        return 0


def run_fetch(assets: List[str] = None, days: int = 30,
              from_screener: bool = False):
    """Fetch recent data for all assets in the universe."""
    print("=" * 70)
    print("  UNIVERSE DATA FETCHER")
    print(f"  Days: {days}")
    print("=" * 70)

    # Determine asset list
    if from_screener:
        universe_path = STATE_DIR / "universe.json"
        if universe_path.exists():
            with open(universe_path) as f:
                universe = json.load(f)
            assets = [a["symbol"] for a in universe.get("assets", [])]
            print(f"  Loaded {len(assets)} assets from universe screener")
        else:
            print("  [WARN] No universe.json found. Run universe_screener.py first.")
            print("  Falling back to default 10 assets.")
            assets = [a.upper().replace("USDT", "/USDT") for a in ACTIVE_ASSETS]
    elif assets:
        # Convert to ccxt format
        assets = [a.upper().replace("USDT", "/USDT") if "/" not in a else a
                  for a in assets]
    else:
        assets = [a.upper().replace("USDT", "/USDT") for a in ACTIVE_ASSETS]

    print(f"  Fetching {len(assets)} assets, last {days} days")

    try:
        import ccxt
        exchange = ccxt.binance({"options": {"defaultType": "spot"},
                                 "enableRateLimit": True})
    except ImportError:
        print("  [ERROR] ccxt not installed. Run: pip install ccxt")
        return

    PROD_DATA_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for i, symbol in enumerate(assets):
        asset_clean = symbol.replace("/", "").lower()
        print(f"\n  [{i+1}/{len(assets)}] {symbol}...", end=" ", flush=True)

        n_trades = fetch_asset_data(exchange, symbol, days=days)
        results[asset_clean] = n_trades
        print(f"{n_trades:,} trades")

    # Summary
    total = sum(results.values())
    fetched = sum(1 for v in results.values() if v > 0)
    print(f"\n  Done: {fetched}/{len(assets)} assets, {total:,} total trades")
    print(f"  Data saved to: {PROD_DATA_DIR}")

    # Save manifest
    manifest_path = PROD_DATA_DIR / "fetch_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "days": days,
            "assets": results,
        }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch recent trade data for the trading universe")
    parser.add_argument("--days", type=int, default=30,
                        help="Number of days to fetch (default: 30)")
    parser.add_argument("--assets", nargs="+", default=None,
                        help="Specific assets (default: screened universe)")
    parser.add_argument("--from-screener", action="store_true",
                        help="Use assets from universe_screener.py output")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s")

    run_fetch(
        assets=args.assets,
        days=args.days,
        from_screener=args.from_screener,
    )


if __name__ == "__main__":
    main()
