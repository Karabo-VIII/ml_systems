"""Binance new-listing announcement scraper (P2 skeleton, 2026-04-21).

Scrapes https://www.binance.com/en/support/announcement/c-48 (new listings
category) and extracts structured events:
    (announcement_timestamp_ms, asset_symbol, listing_type, listing_time_ms)

Writes to `data/events/binance_listings.parquet` (append-only).

Run nightly via cron:
    python src/data/binance_listings_scraper.py

--- LIMITATIONS (stub) ---
Binance support pages use JS-rendered content and anti-scraping protection.
For production, use one of:
  1. Playwright / Selenium (handles JS)
  2. Third-party feed: CryptoPanic API (free tier: 50 req/day), Messari
  3. Twitter/X webhook on @binance announcements account + LLM classifier
  4. Binance official announcement API (not publicly documented but discoverable
     via network tab on the announcements page)

The current implementation attempts a best-effort JSON API fetch that works
for the announcements index page. Extraction of listing details (symbol,
listing time) requires per-announcement parsing.

Pair with `engine_listing_arb.py` in strat_profiles.py once feed is populated.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import requests

# G-AUDIT-018: was parents[2] which resolves to src/frontier/ingest/, breaking
# all downstream paths. Canonical is parents[3] = project root.
ROOT = Path(__file__).resolve().parents[3]
EVENTS_DIR = ROOT / "data" / "events"
EVENTS_DIR.mkdir(parents=True, exist_ok=True)
PARQUET_FP = EVENTS_DIR / "binance_listings.parquet"


def _fetch_listing_page(category_id: int = 48, page: int = 1) -> Optional[Dict]:
    """Binance exposes a JSON API for announcements at:
    /bapi/composite/v1/public/cms/article/list/query
    category 48 = New Cryptocurrency Listing."""
    url = "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query"
    params = {"type": 1, "catalogId": category_id, "pageNo": page, "pageSize": 20}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0) listings-scraper"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[scraper] fetch failed: {e}")
        return None


SYMBOL_RE = re.compile(r"\(([A-Z0-9]{2,12})\)")


def _parse_announcement(ann: Dict) -> Optional[Dict]:
    """Extract asset + listing time from announcement dict."""
    title = ann.get("title", "")
    # Binance titles are typically: "Binance Will List <Asset> (SYMBOL)"
    m = SYMBOL_RE.search(title)
    if not m:
        return None
    symbol = m.group(1)
    ts = ann.get("releaseDate") or ann.get("publishDate")
    if not ts:
        return None
    try:
        ts_ms = int(ts)
        if ts_ms < 1e12:
            ts_ms = int(ts_ms * 1000)
    except (TypeError, ValueError):
        return None
    # Listing time is usually 1-2 days after announcement; best effort extract
    return {
        "announcement_ts_ms": ts_ms,
        "symbol": symbol,
        "title": title,
        "type": "LIST",  # Could be LIST / DELIST / FUTURES
        "listing_ts_ms": None,   # Parse from article body if needed
    }


def scrape(max_pages: int = 3) -> List[Dict]:
    events = []
    for page in range(1, max_pages + 1):
        data = _fetch_listing_page(page=page)
        if not data or "data" not in data:
            break
        articles = data["data"].get("articles", [])
        if not articles:
            break
        for ann in articles:
            parsed = _parse_announcement(ann)
            if parsed:
                events.append(parsed)
    return events


def update_store(events: List[Dict]) -> int:
    """Append new events to the parquet store, dedupe on (ts, symbol)."""
    if not events:
        return 0
    new_df = pd.DataFrame(events)
    if PARQUET_FP.exists():
        old_df = pd.read_parquet(PARQUET_FP)
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined = combined.drop_duplicates(subset=["announcement_ts_ms", "symbol"])
    combined = combined.sort_values("announcement_ts_ms")
    combined.to_parquet(PARQUET_FP, index=False)
    return len(new_df) - (len(combined) - (len(old_df) if PARQUET_FP.exists() else 0))


def main():
    print(f"[scraper] Binance new-listing announcement fetch -> {PARQUET_FP}")
    events = scrape(max_pages=3)
    print(f"[scraper] parsed {len(events)} candidate events")
    if not events:
        print("[scraper] no events retrieved -- API may be blocked or schema changed")
        return
    print(f"[scraper] sample:")
    for e in events[:5]:
        ts = datetime.fromtimestamp(e["announcement_ts_ms"] / 1000, tz=timezone.utc)
        print(f"  {ts.isoformat()} {e['symbol']:10s} {e['title'][:80]}")
    added = update_store(events)
    print(f"[scraper] wrote {added} new events")


if __name__ == "__main__":
    main()
