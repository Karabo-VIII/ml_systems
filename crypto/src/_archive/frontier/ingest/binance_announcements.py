"""
Binance announcement scraper + normalizer.

Reusable ingest module for:
  - p11 Announcement-Volatility pillar (event-triggered 1-48h trade)
  - p6 HODLer Airdrop pillar (scrape HODLer announcements)
  - p10 IEO pillar (scrape Launchpad announcements)
  - General monitoring of Binance-side listings/delistings/maintenance/
    regulatory updates

API:
    fetch_announcements(category: str, page_size: int = 50, max_pages: int = 4)
        -> list[Announcement]
    classify(text: str) -> str
        Returns one of 8 categories: listing / delisting / monitoring /
        margin / earn / maintenance / wallet / regulatory
    extract_tokens(text: str) -> list[str]
        Returns list of affected USDT ticker symbols

Sources:
    https://www.binance.com/bapi/composite/v1/public/cms/article/list/query
    (Binance public JSON used by web UI, no auth required)

Cache:
    logs/frontier/announcements/{category}_{start_ts}.parquet
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = ROOT / "logs" / "frontier" / "announcements"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Binance catalog IDs for article-catalog-list endpoint
# (Discovered empirically by sweeping catalog IDs; Binance does not document these publicly.)
# Turn 022 audit: cat 93 was incorrectly labeled "maintenance" - it actually
# holds trading competitions. Real wallet-maintenance events are in cat 157
# ("Latest Binance News"/general). Fixed mapping below.
CATEGORY_CATALOG_MAP = {
    "listing":       48,   # New Crypto Listing (includes futures launches + spot listings)
    "delisting":     161,  # Delistings (spot + futures)
    "trading_comp":  93,   # Trading competitions (was mislabeled 'maintenance')
    "airdrop":       128,  # HODLer Airdrops
    "launchpool":    48,   # Launchpool / IEO (same bucket as listings)
    "general":       157,  # Latest Binance News -- INCLUDES wallet maintenance events
    "maintenance":   157,  # Alias: wallet maintenance lives in general
    "wallet":        49,   # Binance Wallet / Earn products (also apps)
    "earn":          94,   # Earn products (but mostly FAQ content, low event density)
}

URL = "https://www.binance.com/bapi/composite/v1/public/cms/article/catalog/list/query"
DETAIL_URL = "https://www.binance.com/bapi/composite/v1/public/cms/article/detail/query"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# Binance list-endpoint returns only id/code/title; body + publishDate are NULL.
# Primary: fetch per-article detail (authoritative publishDate).
# Fallback: extract date from title embedded "(YYYY-MM-DD)".
TITLE_DATE_PATTERN = re.compile(r"\((\d{4}-\d{2}-\d{2})(?:\s*&\s*\d{4}-\d{2}-\d{2})?\)")
SLUG_PATTERN = re.compile(r"/announcement/([0-9a-f]{20,40})")


def _fetch_with_backoff(url: str, max_retries: int = 5, base_delay: float = 1.0) -> dict | None:
    """GET with exponential backoff on 429 / transient errors."""
    delay = base_delay
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            return None
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            return None
    return None


def fetch_article_detail(slug: str) -> dict | None:
    """Get publishDate + body + catalogName via detail endpoint."""
    if not slug:
        return None
    payload = _fetch_with_backoff(f"{DETAIL_URL}?articleCode={slug}")
    if not payload or not payload.get("success"):
        return None
    return payload.get("data")


@dataclass
class Announcement:
    article_id: str
    title: str
    body: str
    release_ms: int
    category_slug: str
    url: str
    tokens: list[str] = field(default_factory=list)
    classified_as: Optional[str] = None

    def release_date(self) -> str:
        return pd.Timestamp(self.release_ms, unit="ms").strftime("%Y-%m-%d %H:%M:%S")


def _extract_slug(url: str) -> str | None:
    m = SLUG_PATTERN.search(url or "")
    return m.group(1) if m else None


def enrich_with_details(items: list[Announcement], max_details: int | None = None) -> list[Announcement]:
    """For items with release_ms=0 or empty body, fetch per-article detail.
    Applies exponential backoff on 429."""
    n_fetched = 0
    for a in items:
        if max_details is not None and n_fetched >= max_details:
            break
        if a.release_ms > 0 and a.body:
            continue
        slug = _extract_slug(a.url) or a.article_id
        detail = fetch_article_detail(slug)
        if detail:
            a.release_ms = int(detail.get("publishDate") or a.release_ms or 0)
            a.body = detail.get("body") or a.body or ""
            n_fetched += 1
        time.sleep(0.15)
    return items


def fetch_announcements(category: str, page_size: int = 20, max_pages: int = 4,
                        year: Optional[int] = None,
                        enrich: bool = True) -> list[Announcement]:
    """Fetch announcements from Binance public catalog article list."""
    cat_id = CATEGORY_CATALOG_MAP.get(category)
    if cat_id is None:
        raise ValueError(f"unknown category: {category}")

    out: list[Announcement] = []
    for page in range(1, max_pages + 1):
        params = {
            "catalogId": cat_id,
            "pageNo": page,
            "pageSize": page_size,
        }
        req_url = f"{URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(req_url, headers={"User-Agent": UA, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                payload = json.loads(r.read().decode())
        except Exception as e:
            print(f"[WARN] fetch page {page} failed: {e}")
            break

        data = payload.get("data", {}) or {}
        articles = data.get("articles") or []
        if not articles and data.get("catalogs"):
            cat0 = data["catalogs"][0] if data["catalogs"] else {}
            articles = cat0.get("articles", []) or []
        if not articles:
            break

        for a in articles:
            title = a.get("title") or ""
            # Extract date from title (format "(YYYY-MM-DD)")
            # If no date in title, release_ms=0 and caller must handle
            release_ms = 0
            m = TITLE_DATE_PATTERN.search(title)
            if m:
                try:
                    release_ms = int(pd.Timestamp(m.group(1)).timestamp() * 1000)
                except Exception:
                    pass
            # Fallback: API fields (usually null in list endpoint but try)
            if release_ms == 0:
                release_ms = int(a.get("releaseDate") or a.get("publishDate")
                                 or a.get("createTime") or 0)
            out.append(Announcement(
                article_id=str(a.get("id") or a.get("code") or ""),
                title=title,
                body=a.get("body") or a.get("summary") or "",
                release_ms=release_ms,
                category_slug=category,
                url=f"https://www.binance.com/en/support/announcement/{a.get('code', '')}",
            ))
        time.sleep(0.2)
        if len(articles) < page_size:
            break

    # Enrich via detail endpoint (authoritative publishDate + body)
    if enrich and out:
        out = enrich_with_details(out)

    # Optional year filter
    if year is not None and out:
        yr_start = int(pd.Timestamp(f"{year}-01-01").timestamp() * 1000)
        yr_end = int(pd.Timestamp(f"{year + 1}-01-01").timestamp() * 1000)
        out = [a for a in out if yr_start <= a.release_ms < yr_end]

    return out


# Upgraded classifier — order matters (more specific patterns first).
# Bravo turn 017 audit caught 60-100% miss rate on original patterns; merged
# proposed upgrades here.
CATEGORY_PATTERNS = [
    ("futures_launch", re.compile(r"\b(usd.{0,3}-?margined|perpetual contract|futures will launch)\b", re.I)),
    ("trading_comp",   re.compile(r"\b(trading competition|trading tournament|trade to share|trade.{1,30}share.{1,20}reward)\b", re.I)),
    ("delisting",      re.compile(r"\b(will\s+delist|removal\s+of\s+(spot|margin)\s+trading|delisting notice|will be delisted)\b", re.I)),
    ("margin_change",  re.compile(r"\b(margin will add|margin trading pair|risk limit|margin.{0,20}tier)\b", re.I)),
    ("airdrop",        re.compile(r"\b(hodler airdrop|megadrop|airdrop program|airdrop distribution)\b", re.I)),
    ("launchpool",     re.compile(r"\b(launchpool|launchpad|ieo|initial exchange offering)\b", re.I)),
    ("listing",        re.compile(r"\b(will list|seed tag|innovation zone|new crypto listing|spot\s+listing)\b|\b[A-Z]{2,10}usdt\b.*\b(spot|listing)\b", re.I)),
    ("monitoring",     re.compile(r"\b(monitoring tag|risk warning|special treatment)\b", re.I)),
    ("earn",           re.compile(r"\b(simple earn|apr\s+(increase|decrease|change|update)|staking\s+\w+|flexible savings)\b", re.I)),
    ("maintenance",    re.compile(r"\b(wallet\s+maintenance|deposit(s)?\s+and\s+withdrawal(s)?|scheduled maintenance|system upgrade)\b", re.I)),
    ("wallet",         re.compile(r"\b(wallet integration|network upgrade|fork\b|hard fork)\b", re.I)),
    ("regulatory",     re.compile(r"\b(compliance|regulatory|sanction|kyc|fatf)\b", re.I)),
]


def classify(text: str) -> str:
    for name, pat in CATEGORY_PATTERNS:
        if pat.search(text):
            return name
    return "general"


TOKEN_PATTERN = re.compile(r"\b([A-Z]{2,10})USDT\b|\(([A-Z]{2,10})\)")


def extract_tokens(text: str) -> list[str]:
    """Pull likely token tickers from announcement text.

    Matches 'TOKEUSDT' inline references and '(TOKEN)' parenthetical.
    """
    hits = []
    for m in TOKEN_PATTERN.finditer(text):
        sym = m.group(1) or m.group(2)
        if sym and sym.isupper() and 2 <= len(sym) <= 10:
            hits.append(sym)
    # de-dup preserving order
    seen = set()
    out = []
    for s in hits:
        if s not in seen:
            out.append(s); seen.add(s)
    return out


def normalize_batch(items: list[Announcement]) -> pd.DataFrame:
    """Convert list to DataFrame with classified category + extracted tokens."""
    rows = []
    for a in items:
        full_text = f"{a.title} {a.body}"
        cls = classify(full_text)
        tok = extract_tokens(full_text)
        d = asdict(a)
        d["classified_as"] = cls
        d["tokens"] = tok
        rows.append(d)
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df["release_ts"] = pd.to_datetime(df["release_ms"], unit="ms")
    return df


def cache_path(category: str) -> Path:
    return CACHE_DIR / f"{category}_recent.parquet"


def load_cached(category: str) -> pd.DataFrame | None:
    p = cache_path(category)
    if p.exists():
        return pd.read_parquet(p)
    return None


def save_cached(category: str, df: pd.DataFrame) -> None:
    df.to_parquet(cache_path(category))


if __name__ == "__main__":
    # Smoke test -- fetch recent listings + classify
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="listing")
    ap.add_argument("--pages", type=int, default=2)
    args = ap.parse_args()

    print(f"[FETCH] {args.category} announcements (max_pages={args.pages})...")
    items = fetch_announcements(args.category, max_pages=args.pages)
    print(f"  fetched {len(items)} announcements")
    if not items:
        print("  (empty -- endpoint may be blocked or rate-limited)")
    else:
        df = normalize_batch(items)
        save_cached(args.category, df)
        print(f"  cached to {cache_path(args.category)}")
        print()
        # Display-safe (strip non-ASCII for cp1252)
        def safe(s: str) -> str:
            return str(s).encode("ascii", errors="replace").decode("ascii")
        for _, row in df.head(8).iterrows():
            print(f"  {row['release_ts']}  [{row['classified_as']:<12s}]  "
                  f"tokens={row['tokens']}  {safe(row['title'])[:80]}")
