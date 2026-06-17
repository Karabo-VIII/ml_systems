"""
Bravo turn 017 -- fill Binance announcement dates via per-article detail fetch.

Paranoid finding (turn 017): Alpha's turn-016 scraper cached 268 announcements
across 5 categories, but:
  - Alpha's "~60% via title regex" claim applies ONLY to listings (24/60 = 40%).
  - ALL OTHER CATEGORIES (delisting / maintenance / earn / airdrop = 208 rows):
    0% date extraction. 100% epoch0.

Without proper dates, Phase 1 event-study cannot run -- can't compute forward
returns without trigger timestamp.

Fix: the Binance detail endpoint
    https://www.binance.com/bapi/composite/v1/public/cms/article/detail/query
      ?articleCode=<slug>
returns `publishDate` as Unix ms. The slug is extractable from the URL field
in Alpha's parquet cache (format: .../announcement/<slug>).

This script:
  1. Read each category parquet
  2. For rows with release_ms == 0, extract slug from url, fetch detail, update
  3. Write back to parquet
  4. Report before/after coverage + audit classification quality
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import io as _io
sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd

ROOT = Path('c:/Users/karab/Documents/coding/v4_crypto_stystem')
CACHE = ROOT / 'logs/frontier/announcements'
DETAIL_URL = 'https://www.binance.com/bapi/composite/v1/public/cms/article/detail/query'
UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'

SLUG_RE = re.compile(r'/announcement/([0-9a-f]{20,40})')


def extract_slug(url: str) -> str | None:
    m = SLUG_RE.search(url or '')
    return m.group(1) if m else None


def fetch_detail(slug: str, timeout: int = 10) -> dict | None:
    u = f'{DETAIL_URL}?articleCode={slug}'
    req = urllib.request.Request(u, headers={'User-Agent': UA, 'Accept': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            p = json.loads(r.read().decode())
    except Exception as e:
        return None
    return p.get('data') if p.get('success') else None


def fill_category(cat: str) -> dict:
    path = CACHE / f'{cat}_recent.parquet'
    if not path.exists():
        return {'cat': cat, 'error': 'no_cache'}
    df = pd.read_parquet(path)
    before_real = int((df['release_ms'] > 0).sum())
    before_epoch0 = int((df['release_ms'] == 0).sum())

    need_fix = df[df['release_ms'] == 0].copy()
    filled = 0
    missed = 0
    for idx, row in need_fix.iterrows():
        slug = extract_slug(row['url'])
        if not slug:
            missed += 1
            continue
        detail = fetch_detail(slug)
        if detail is None:
            missed += 1
            continue
        ms = detail.get('publishDate')
        if not ms or ms <= 0:
            missed += 1
            continue
        df.at[idx, 'release_ms'] = int(ms)
        filled += 1
        time.sleep(0.15)  # polite rate

    # refresh release_ts
    df['release_ts'] = pd.to_datetime(df['release_ms'], unit='ms')
    df.to_parquet(path)
    after_real = int((df['release_ms'] > 0).sum())
    after_epoch0 = int((df['release_ms'] == 0).sum())
    return {
        'cat': cat, 'n': len(df),
        'before_real': before_real, 'before_epoch0': before_epoch0,
        'after_real': after_real, 'after_epoch0': after_epoch0,
        'filled': filled, 'missed': missed,
    }


def audit_classification(cat: str) -> pd.DataFrame | None:
    path = CACHE / f'{cat}_recent.parquet'
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df


def main() -> None:
    cats = ['listing', 'delisting', 'maintenance', 'earn', 'airdrop']
    print('=== Fill announcement dates via detail endpoint ===\n')
    print(f'{"cat":<14s} {"n":>4s} {"before_real":>11s} {"before_eph0":>12s} {"after_real":>10s} {"after_eph0":>10s} {"filled":>6s} {"missed":>6s}')
    print('-'*80)
    results = []
    for cat in cats:
        r = fill_category(cat)
        results.append(r)
        if 'error' in r:
            print(f'{cat:<14s}  ERROR: {r["error"]}')
            continue
        print(f'{r["cat"]:<14s} {r["n"]:>4d} {r["before_real"]:>11d} {r["before_epoch0"]:>12d} {r["after_real"]:>10d} {r["after_epoch0"]:>10d} {r["filled"]:>6d} {r["missed"]:>6d}')
    print()

    # Classification audit
    print('=== Classification quality audit (spot-check 5 per category) ===')
    for cat in cats:
        df = audit_classification(cat)
        if df is None: continue
        print(f'\n--- {cat} ---')
        by_cls = df['classified_as'].value_counts().to_dict()
        print(f'classifier counts: {by_cls}')
        # Show misclassifications (where category != classified)
        mismatches = df[df['classified_as'] != cat]
        if len(mismatches) == 0:
            print('  all match category')
        else:
            print(f'  {len(mismatches)}/{len(df)} mis-matched (classifier != category)')
            for _, row in mismatches.head(3).iterrows():
                title = str(row['title'])[:90].encode('ascii','replace').decode('ascii')
                print(f'    [{row["classified_as"]:<10s}] {title}')

    # Recompute overall coverage
    total_n = sum(r['n'] for r in results if 'error' not in r)
    total_after_real = sum(r['after_real'] for r in results if 'error' not in r)
    total_filled = sum(r['filled'] for r in results if 'error' not in r)
    print(f'\n=== Summary ===')
    print(f'total announcements: {total_n}')
    print(f'date coverage BEFORE (from Alpha turn 016): {sum(r["before_real"] for r in results if "error" not in r)}/{total_n} = {100*sum(r["before_real"] for r in results if "error" not in r)/total_n:.1f}%')
    print(f'date coverage AFTER (this fix):             {total_after_real}/{total_n} = {100*total_after_real/total_n:.1f}%')
    print(f'filled this run: {total_filled}  (remaining missed: {sum(r["missed"] for r in results if "error" not in r)})')


if __name__ == '__main__':
    main()
