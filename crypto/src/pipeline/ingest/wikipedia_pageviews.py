"""Wikipedia pageviews -- free social-velocity proxy (no API key) -- canonical producer.

Wikimedia REST pageviews API (rate-limit friendly). Per-asset hand-mapped article.

Output (registry source `wiki_pageviews`, feature_registry.yaml):
    data/raw_external/wikipedia/wiki_pageviews_daily.parquet
        columns: date, asset, wiki_views
        coverage: u10 (hand-mapped articles); extend ASSET_TO_ARTICLE for more.

Provenance: restored 2026-05-29 from src/_archive/frontier/ingest/wikipedia_pageviews.py
(orphaned when src/frontier was archived; chimera `soc_*` feature went ~5wk stale).
Now writes the registry path, atomic, with the canonical refresh.py CLI.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

# Path bootstrap: refresh.py runs this as a direct script, so src/ is not on
# sys.path. Mirror the canonical producer pattern (etf_flows.py) before importing
# the pipeline framework. Fixes ModuleNotFoundError on the 2026-05-30 refresh run.
import sys
_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src" / "pipeline"))
sys.path.insert(0, str(_ROOT / "src"))
from pipeline.parquet_io import atomic_write_parquet

__contract__ = {
    "kind": "external_ingest",
    "inputs": ["wikimedia.org pageviews REST API (per-article, daily)"],
    "outputs": ["data/raw_external/wikipedia/wiki_pageviews_daily.parquet"],
    "invariants": [
        "schema: date, asset, wiki_views",
        "coverage: u10 hand-mapped articles (extend ASSET_TO_ARTICLE to widen)",
        "atomic_write_via_parquet_io",
    ],
}

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_PATH = PROJECT_ROOT / "data" / "raw_external" / "wikipedia" / "wiki_pageviews_daily.parquet"
UA = "v4-pipeline/1.0 (research; noreply@example.com)"

ASSET_TO_ARTICLE = {
    "BTC": "Bitcoin",
    "ETH": "Ethereum",
    "SOL": "Solana_(blockchain_platform)",
    "BNB": "Binance",
    "XRP": "Ripple_(payment_protocol)",
    "DOGE": "Dogecoin",
    "ADA": "Cardano_(blockchain_platform)",
    "AVAX": "Avalanche_(blockchain_platform)",
    "LINK": "Chainlink_(blockchain)",
    "LTC": "Litecoin",
}


def fetch_article(article: str, start: str, end: str) -> list[dict]:
    article_enc = urllib.parse.quote(article, safe="")
    url = (f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/"
           f"all-access/user/{article_enc}/daily/{start}/{end}")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read()).get("items", [])
    except Exception as e:
        print(f"  [{article}] failed: {e}", flush=True)
        return []


def build_panel(start: str = "2024-01-01") -> pd.DataFrame:
    start_ts = pd.Timestamp(start).strftime("%Y%m%d00")
    end_ts = pd.Timestamp.now().strftime("%Y%m%d00")
    frames = []
    for asset, article in ASSET_TO_ARTICLE.items():
        items = fetch_article(article, start_ts, end_ts)
        if not items:
            print(f"[wiki] {asset}: empty", flush=True)
            continue
        df = pd.DataFrame(items)
        df["date"] = pd.to_datetime(df["timestamp"].str[:8], format="%Y%m%d")
        df["asset"] = asset
        df["wiki_views"] = df["views"]
        frames.append(df[["date", "asset", "wiki_views"]])
        print(f"[wiki] {asset}: {len(df)} days, mean {df['wiki_views'].mean():.0f}", flush=True)
        time.sleep(0.3)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["asset", "date"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="Fetch Wikipedia pageviews -> raw_external/wikipedia/")
    ap.add_argument("--start", default="2024-01-01", help="History start (YYYY-MM-DD)")
    ap.add_argument("--force", action="store_true", help="No-op (always full refetch)")
    ap.add_argument("--workers", type=int, default=1, help="No-op; accepted for refresh.py")
    ap.add_argument("--dry-run", action="store_true", help="Fetch + report, do not write")
    args = ap.parse_args()
    panel = build_panel(start=args.start)
    if panel.empty:
        print("[wiki] ERROR: no data fetched; output unchanged", flush=True)
        raise SystemExit(2)
    if args.dry_run:
        print(f"[wiki] DRY-RUN: would write {len(panel)} rows to {OUT_PATH}", flush=True)
        return
    atomic_write_parquet(panel, OUT_PATH, required_cols={"date", "asset", "wiki_views"})
    print(f"[wiki] saved: {OUT_PATH} ({len(panel)} rows)", flush=True)


if __name__ == "__main__":
    main()
