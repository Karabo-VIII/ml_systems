"""Harvest per-symbol launch dates from Binance Futures exchangeInfo.

Binance Futures publishes `onboardDate` (ms epoch) for every listed symbol via
https://fapi.binance.com/fapi/v1/exchangeInfo. This is the canonical listing
date. We cache it in `config/asset_launch_dates.json` so fetch_all.py can
skip pre-listing dates instead of attempting them and getting 404s for years
of pre-listing history.

Why this matters:
  Without launch dates, fetch_all defaults to BTC's launch (2019-09-08) for
  unknown symbols. For PEPE (May 2023 listing), that means ~3.5 years of
  pre-listing 404 attempts -- 1300+ wasted HTTP roundtrips per data type
  per asset. With launch dates: 0 pre-listing attempts.

Usage:
  python scripts/harvest_launch_dates.py            # write cache
  python scripts/harvest_launch_dates.py --print    # dump cache to stdout
  python scripts/harvest_launch_dates.py --diff     # show what would change

Cache format: JSON dict mapping {SYMBOL: "YYYY-MM-DD"}.
Cache path: config/asset_launch_dates.json (committed to git).
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = PROJECT_ROOT / "config" / "asset_launch_dates.json"
EXCHANGE_INFO_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"
USER_AGENT = "v4_crypto_stystem-launch-harvester/1.0"


def fetch_exchange_info() -> dict:
    """GET /fapi/v1/exchangeInfo. Returns the parsed JSON dict.

    Raises urllib.error.URLError / json.JSONDecodeError on failure.
    """
    req = urllib.request.Request(EXCHANGE_INFO_URL,
                                  headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def parse_launch_dates(exchange_info: dict) -> dict[str, str]:
    """Map symbol -> 'YYYY-MM-DD' launch date from exchangeInfo['symbols'].

    Filters out delisted / non-USDT / inactive symbols. We only care about
    *currently listed* USDT-margined perpetuals -- delisted symbols are
    irrelevant since we won't be fetching new data for them.
    """
    out: dict[str, str] = {}
    for s in exchange_info.get("symbols", []):
        symbol = s.get("symbol", "")
        if not symbol.endswith("USDT"):
            continue
        if s.get("contractType") != "PERPETUAL":
            continue
        if s.get("status") != "TRADING":
            continue
        ts_ms = s.get("onboardDate")
        if ts_ms is None:
            continue
        try:
            d = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).date()
        except (TypeError, ValueError, OSError):
            continue
        out[symbol] = d.strftime("%Y-%m-%d")
    return out


def load_cache() -> dict[str, str]:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text())
    except json.JSONDecodeError:
        return {}


def write_cache(launch_dates: dict[str, str]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(sorted(launch_dates.items()))
    CACHE_PATH.write_text(json.dumps(payload, indent=2) + "\n",
                           encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--print", action="store_true",
                    help="Dump current cache to stdout (no fetch).")
    ap.add_argument("--diff", action="store_true",
                    help="Fetch fresh + show what would change vs cache (no write).")
    args = ap.parse_args()

    if args.print:
        cache = load_cache()
        if not cache:
            print(f"[harvest] no cache at {CACHE_PATH.relative_to(PROJECT_ROOT)}")
            return 1
        print(f"[harvest] {CACHE_PATH.relative_to(PROJECT_ROOT)} ({len(cache)} symbols):")
        for sym, d in sorted(cache.items()):
            print(f"  {sym:<14}  {d}")
        return 0

    print(f"[harvest] fetching {EXCHANGE_INFO_URL}...", flush=True)
    info = fetch_exchange_info()
    fresh = parse_launch_dates(info)
    print(f"[harvest] fetched onboardDate for {len(fresh)} active USDT perpetuals",
          flush=True)

    cache = load_cache()

    if args.diff:
        added = sorted(set(fresh) - set(cache))
        removed = sorted(set(cache) - set(fresh))
        changed = sorted(s for s in fresh if s in cache and fresh[s] != cache[s])
        print(f"[harvest] diff vs cache:")
        print(f"  added:   {len(added)}  {added[:8]}")
        print(f"  removed: {len(removed)} (delisted)  {removed[:8]}")
        print(f"  changed: {len(changed)}  {changed[:8]}")
        return 0

    write_cache(fresh)
    print(f"[harvest] wrote {CACHE_PATH.relative_to(PROJECT_ROOT)} "
          f"({len(fresh)} symbols)", flush=True)
    # Print 5 ASCII-safe sample entries (Windows cp1252 chokes on unicode
    # symbol names — defensive even though USDT perpetuals shouldn't have any).
    def _ascii_safe(s: str) -> bool:
        try:
            s.encode("cp1252"); return True
        except UnicodeEncodeError:
            return False
    sample = [(s, d) for s, d in sorted(fresh.items()) if _ascii_safe(s)]
    print(f"[harvest] sample (ASCII-safe):")
    for sym, d in sample[:3]:
        print(f"  {sym:<14}  {d}")
    print(f"  ...")
    for sym, d in sample[-3:]:
        print(f"  {sym:<14}  {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
