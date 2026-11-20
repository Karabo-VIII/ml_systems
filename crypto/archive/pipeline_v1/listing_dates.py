"""listing_dates.py -- centralized Binance listing-date resolver.

Phase 8 of pipeline overhaul. Single source of truth for "when did Binance
first list asset X?". Replaces per-producer copy-paste of the launch-date
lookup. Every producer iterating dates SHOULD call this module's helpers
to skip pre-listing dates.

WHY CENTRALIZE
==============
Before this module: `fetch_all.py` had `_resolve_launch_date()` with cache
+ 1000-prefix fallback + hardcoded majors. NO OTHER producer consumed it.
Result: bars / chimera / panel features all iterated date ranges including
pre-listing days, generating:
  - "confirmed-missing" markers for dates that NEVER had data
  - wasted fetch attempts (404s on Binance Vision)
  - wasted compute cycles (zero-row dataframes processed through chimera)
  - polluted audit findings (low-population looks like producer bug)

With this module: one call to `get_listing_date(symbol)` from anywhere.
Centralized cache, thread-safe singleton, single-API contract.

DESIGN
======
- Lazy singleton with threading.Lock (R32+++ audit flagged the original
  cache as race-prone with --asset-workers>1)
- Reads config/asset_launch_dates.json (531+ symbols from Binance API)
- Falls back to 1000-prefix variant for low-priced tokens (PEPE, SHIB, etc.)
- Falls back to legacy hardcoded majors dict
- Final fallback: configurable default (2019-09-25 = Binance Futures launch)

API
===
    from pipeline.listing_dates import (
        get_listing_date,             # symbol -> date
        is_pre_listing,               # (symbol, date) -> bool
        filter_pre_listing,           # (symbol, [dates]) -> [dates >= listing]
        resolve_fapi_symbol,          # 1000-prefix lookup for fapi calls
        reset_cache,                  # test-only invalidation
    )

USAGE PATTERNS
==============

1. Skip pre-listing dates in a date-iterating loop:
    for date in iter_dates:
        if is_pre_listing(symbol, date):
            continue
        # ... process date ...

2. Pre-filter a full date range:
    dates = filter_pre_listing(symbol, date_range("2020-01-01", "2026-05-15"))

3. Resolve the fapi (1000-prefix) symbol once per asset:
    fapi_sym = resolve_fapi_symbol("PEPEUSDT")   # -> "1000PEPEUSDT"
"""
from __future__ import annotations

__contract__ = {
    "kind": "framework_helper",
    "stage": "pipeline_io",
    "owner": "pipeline/orchestration",
    "outputs": "listing-date lookup helpers (pure functions + lazy cache)",
    "invariants": [
        "thread-safe singleton: load-once protected by threading.Lock",
        "1000-prefix fallback for low-priced asset listings",
        "ASSET_LAUNCH_DATES hardcoded majors as final on-disk fallback",
        "default = 2019-09-25 (Binance Futures launch day) if nothing matches",
    ],
}

import json
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCH_DATES_CACHE_PATH = PROJECT_ROOT / "config" / "asset_launch_dates.json"

# Final fallback when nothing in cache matches. Binance Futures launched
# 2019-09-25; no asset can have data before this.
DEFAULT_LAUNCH = datetime(2019, 9, 25)

# Legacy hardcoded majors (used when JSON cache is missing or pre-cache run).
# Matches fetch_all.py ASSET_LAUNCH_DATES dict.
ASSET_LAUNCH_DATES: dict[str, datetime] = {
    "BTCUSDT": datetime(2019, 9, 8),
    "ETHUSDT": datetime(2019, 11, 27),
    "SOLUSDT": datetime(2020, 9, 14),
    "XRPUSDT": datetime(2020, 1, 6),
    "BNBUSDT": datetime(2020, 2, 10),
    "DOGEUSDT": datetime(2019, 7, 5),
    "ADAUSDT": datetime(2018, 4, 17),
    "AVAXUSDT": datetime(2020, 9, 22),
    "LINKUSDT": datetime(2019, 1, 16),
    "LTCUSDT": datetime(2017, 12, 13),
}

# Thread-safe lazy singleton state
_CACHE: dict[str, datetime] | None = None
_CACHE_LOADED = False
_CACHE_LOCK = threading.Lock()


# ============================================================================
# Internal cache management
# ============================================================================

def _load_cache() -> dict[str, datetime]:
    """Load config/asset_launch_dates.json into a {symbol: datetime} dict.
    Returns empty dict if file is missing / unparseable.

    Phase 8 thread-safety: load-once under threading.Lock so multiple
    --asset-workers don't race on first access.
    """
    global _CACHE, _CACHE_LOADED
    if _CACHE_LOADED:
        return _CACHE or {}
    with _CACHE_LOCK:
        # Double-check inside the lock
        if _CACHE_LOADED:
            return _CACHE or {}
        out: dict[str, datetime] = {}
        if LAUNCH_DATES_CACHE_PATH.exists():
            try:
                payload = json.loads(LAUNCH_DATES_CACHE_PATH.read_text(
                    encoding="utf-8"))
                for sym, ds in payload.items():
                    try:
                        out[sym] = datetime.strptime(ds, "%Y-%m-%d")
                    except ValueError:
                        continue
            except Exception as e:
                print(f"[listing_dates] WARN: failed to load "
                      f"{LAUNCH_DATES_CACHE_PATH.name}: {e}", flush=True)
        _CACHE = out
        _CACHE_LOADED = True
    return out


def reset_cache() -> None:
    """Test-only: invalidate the singleton so the next call re-loads."""
    global _CACHE, _CACHE_LOADED
    with _CACHE_LOCK:
        _CACHE = None
        _CACHE_LOADED = False


# ============================================================================
# Public API
# ============================================================================

def _normalize_symbol(symbol: str) -> str:
    """Normalize a symbol to the cache's key format (uppercase, USDT suffix)."""
    s = symbol.upper().strip()
    if not s.endswith("USDT"):
        s = s + "USDT"
    return s


def get_listing_date(symbol: str,
                       default: datetime = DEFAULT_LAUNCH) -> datetime:
    """Resolve a symbol's Binance Futures launch date.

    Priority:
        1. config/asset_launch_dates.json (~531 symbols from Binance API)
        2. 1000-prefix variant (PEPEUSDT -> 1000PEPEUSDT)
        3. ASSET_LAUNCH_DATES legacy hardcoded majors
        4. `default` (2019-09-25 = Binance Futures launch)

    Args:
        symbol: asset symbol; case-insensitive; USDT suffix added if missing
        default: returned if no source matches (default = Futures launch day)

    Returns:
        datetime of first day data could exist for this symbol.
        NOTE: always a datetime (not date). Do NOT compare the result directly
        against a datetime.date object (`datetime > date` raises TypeError in
        Python 3); use .date() first, or use is_pre_listing() which normalizes.

    Thread-safe; cache is loaded once and shared across threads.
    """
    sym = _normalize_symbol(symbol)
    cache = _load_cache()
    if sym in cache:
        return cache[sym]
    # 1000-prefix fallback for low-priced tokens
    base = sym.replace("USDT", "")
    prefixed = f"1000{base}USDT"
    if prefixed in cache:
        return cache[prefixed]
    if sym in ASSET_LAUNCH_DATES:
        return ASSET_LAUNCH_DATES[sym]
    return default


def is_pre_listing(symbol: str, d: datetime | date | str) -> bool:
    """Return True if `d` is BEFORE the asset's listing date.

    Use in date-iterating loops to skip dates that NEVER had data:

        for date in iter_dates:
            if is_pre_listing(symbol, date):
                continue
            # ... process ...
    """
    if isinstance(d, str):
        try:
            d = datetime.strptime(d, "%Y-%m-%d")
        except ValueError:
            return False
    elif isinstance(d, date) and not isinstance(d, datetime):
        d = datetime(d.year, d.month, d.day)
    listing = get_listing_date(symbol)
    return d < listing


def filter_pre_listing(symbol: str,
                          dates: Iterable[datetime | date | str]
                          ) -> list[datetime | date | str]:
    """Strip pre-listing dates from a date sequence.

    Preserves input types (e.g. if you pass date strings, you get back
    date strings). Returns a list (eager).
    """
    return [d for d in dates if not is_pre_listing(symbol, d)]


def resolve_fapi_symbol(symbol: str) -> str:
    """Return the FAPI symbol form (1000-prefix if needed for low-priced asset).

    Used by Binance fapi REST calls where SHIB/PEPE/BONK/FLOKI/etc. must
    be queried as 1000SHIBUSDT / 1000PEPEUSDT / etc.

    Priority:
        1. bare symbol in cache -> use bare
        2. 1000-prefix variant in cache -> use prefixed
        3. else -> bare (best-effort)
    """
    sym = _normalize_symbol(symbol)
    cache = _load_cache()
    if sym in cache:
        return sym
    base = sym.replace("USDT", "")
    prefixed = f"1000{base}USDT"
    if prefixed in cache:
        return prefixed
    return sym


def n_cached_symbols() -> int:
    """Return how many symbols are in the cache (diagnostic)."""
    return len(_load_cache())


__all__ = [
    "get_listing_date", "is_pre_listing", "filter_pre_listing",
    "resolve_fapi_symbol", "n_cached_symbols", "reset_cache",
    "DEFAULT_LAUNCH", "ASSET_LAUNCH_DATES", "LAUNCH_DATES_CACHE_PATH",
]
