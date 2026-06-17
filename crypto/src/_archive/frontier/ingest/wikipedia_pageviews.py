"""Wikipedia pageviews — free social velocity proxy (no API key).

Wikipedia's pageview API: rate-limit friendly, 100 req/sec allowed.
Endpoint: /api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{ARTICLE}/daily/{START}/{END}
Format: YYYYMMDD00 for timestamps.

Per-asset Wikipedia article names (hand-mapped):
    BTC: Bitcoin
    ETH: Ethereum
    SOL: Solana
    BNB: BNB (cryptocurrency)  [alt: Binance]
    XRP: XRP
    DOGE: Dogecoin
    ADA: Cardano
    AVAX: Avalanche (blockchain platform)
    LINK: Chainlink_(blockchain)
    LTC: Litecoin

Output:
    data/frontier/social/wiki_pageviews_daily.parquet
        columns: date, asset, wiki_views
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
OUT_PATH = OUT_DIR / "wiki_pageviews_daily.parquet"

ASSET_TO_ARTICLE = {
    "BTC": "Bitcoin",
    "ETH": "Ethereum",
    "SOL": "Solana_(blockchain_platform)",
    "BNB": "Binance",                          # BNB_(cryptocurrency) redirects; use Binance
    "XRP": "Ripple_(payment_protocol)",         # XRP/Ripple_(cryptocurrency) redirect; this has real traffic
    "DOGE": "Dogecoin",
    "ADA": "Cardano_(blockchain_platform)",
    "AVAX": "Avalanche_(blockchain_platform)",
    "LINK": "Chainlink_(blockchain)",
    "LTC": "Litecoin",
}

UA = "v4-frontier-research/1.0 (noreply@example.com)"


def fetch_article(article: str, start: str, end: str) -> list[dict]:
    article_enc = urllib.parse.quote(article, safe="")
    url = (f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/"
           f"all-access/user/{article_enc}/daily/{start}/{end}")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read()).get("items", [])
    except Exception as e:
        print(f"  [{article}] failed: {e}")
        return []


def main():
    start_ts = "2024010100"
    end_ts = pd.Timestamp.now().strftime("%Y%m%d00")

    frames = []
    for asset, article in ASSET_TO_ARTICLE.items():
        print(f"[{asset}] fetching '{article}'...", flush=True)
        items = fetch_article(article, start_ts, end_ts)
        if not items:
            print(f"  [{asset}] empty (article may not exist)")
            continue
        df = pd.DataFrame(items)
        df["date"] = pd.to_datetime(df["timestamp"].str[:8], format="%Y%m%d")
        df["asset"] = asset
        df["wiki_views"] = df["views"]
        print(f"  [{asset}] {len(df)} days, mean views {df['wiki_views'].mean():.0f}")
        frames.append(df[["date", "asset", "wiki_views"]])
        time.sleep(0.3)

    if not frames:
        print("[err] no data")
        return

    panel = pd.concat(frames, ignore_index=True).sort_values(["asset", "date"]).reset_index(drop=True)
    panel.to_parquet(OUT_PATH, index=False)
    print(f"\n[wiki] saved: {OUT_PATH} ({len(panel)} rows)")


if __name__ == "__main__":
    main()
