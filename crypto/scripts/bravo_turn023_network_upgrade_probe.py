"""
Bravo turn 023 -- p11 Phase 1 probe on NETWORK-UPGRADE events.

Novel category (not Alpha's maintenance-resumed, not delisting-rebound):
Binance periodically announces support for token X's network upgrade /
hard fork / token-swap. Mechanism hypothesis:
  - Positive dev-activity signal (active protocol maintenance)
  - Fork snapshot -> some holders get airdropped / reward
  - Renewed interest in token X, 1-7d mild bullish
  - Expected magnitude +1-5% mean-reversion bullish per p11 scoping

Data: catalog-157 cache (refreshed turn 023 to 100 articles, 100% bodies,
97% tokens). Filter to titles matching network-upgrade patterns.

Design (D1-compliant):
  Entry: announcement + entry_lag_min=60 (retail-latency)
  Horizons: 1h, 6h, 12h, 24h, 72h, 168h
  Cost: 20 bps round-trip
  Universe: tokens extracted per announcement (filtered to valid USDT pairs)
  Chronological splits constrained by current cache: 2025-09-05 -> 2026-04-16
  (only ~7 months; split simply as WINDOW vs shuffle-null as TRAIN-vs-OOS
   stand-in until historical pagination ships)

Fixed harness: src/frontier/utils/event_study.py with fixed shuffle-null
(turn 021).
"""
from __future__ import annotations

import sys, io, json, re
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np

ROOT = Path('c:/Users/karab/Documents/coding/v4_crypto_stystem')
sys.path.insert(0, str(ROOT))

from src.frontier.utils.event_study import run_event_study, pretty_report

# Load catalog-157 cache (maintenance alias -- post-turn-22 fix)
df = pd.read_parquet(ROOT / 'logs/frontier/announcements/maintenance_recent.parquet')
print(f'Raw cat-157 rows: {len(df)}')

# Filter to network-upgrade events
UPGRADE_PAT = re.compile(r'(network upgrade|hard fork|token swap|rebranding|integration of)', re.I)
upgrade_df = df[df['title'].str.contains(UPGRADE_PAT, regex=True)].copy()
print(f'Network-upgrade events: {len(upgrade_df)}')

# Expand to (symbol, event_date) pairs; skip stablecoins (they don't pump on upgrades)
STABLECOIN_EXCLUDE = {'USDT','USDC','DAI','USDS','TUSD','BUSD','FDUSD','USD','PAX','UTC','UST','U'}
pairs = []
for _, row in upgrade_df.iterrows():
    if row['release_ms'] == 0:
        continue
    for tok in row['tokens']:
        tok = str(tok).upper().strip()
        if tok in STABLECOIN_EXCLUDE:
            continue
        if 2 <= len(tok) <= 10 and tok.isalpha():
            pairs.append({
                'symbol': tok,
                'event_date': pd.Timestamp(row['release_ms'], unit='ms'),
                'category': 'network_upgrade',
                'source_title': str(row['title'])[:100],
            })

events = pd.DataFrame(pairs).drop_duplicates(subset=['symbol','event_date']).reset_index(drop=True)
print(f'(symbol, event_date) pairs after stablecoin filter: {len(events)}')
print(f'Unique symbols: {events["symbol"].nunique()}')
print(f'Date span: {events["event_date"].min()} -> {events["event_date"].max()}')
print('Sample:')
for _, row in events.head(10).iterrows():
    s = str(row['source_title']).encode('ascii','replace').decode('ascii')
    print(f'  {row["event_date"]}  {row["symbol"]:<8s}  {s[:80]}')
print()

# Run event study
print('Running event_study with fixed shuffle_null...')
results = run_event_study(
    events_df=events[['symbol','event_date','category']],
    horizons_h=[1, 6, 12, 24, 72, 168],
    interval='1h',
    use_spot=True,
    entry_lag_min=60,
    cost_rt_pct=0.0020,
    splits={
        'ALL':     ('2025-09-01', '2026-04-30'),
        'HALF1':   ('2025-09-01', '2026-01-31'),
        'HALF2':   ('2026-02-01', '2026-04-30'),
    },
    shuffle_null_n=10,
    verbose=False,
)

print()
print(pretty_report(results))

# Compare real vs null
print('\n=== Real vs Null (fixed shuffle) ===')
all_stats = results['per_split_horizon'].get('ALL', {})
null = results.get('null', {})
for h in (1, 6, 12, 24, 72, 168):
    r = all_stats.get(h, {})
    n = null.get(h, {})
    if r.get('status') == 'thin' or n.get('status') == 'thin':
        continue
    real_t = r.get('t_stat', 0)
    null_p5 = n.get('null_t_p5', 0)
    null_p95 = n.get('null_t_p95', 0)
    null_mean = n.get('null_t_mean', 0)
    real_mean = r.get('mean_pct', 0)
    if real_t > null_p95:
        verdict = 'SIGNAL (above null p95)'
    elif real_t < null_p5:
        verdict = 'SIGNAL-NEG (below null p5)'
    else:
        verdict = 'WITHIN-NULL'
    print(f'  h{h:>3d}h: real_t={real_t:+5.2f}  mean={real_mean:+5.2f}%  null_p5={null_p5:+5.2f}  null_mean={null_mean:+5.2f}  null_p95={null_p95:+5.2f}  -> {verdict}')

# Save
OUT = ROOT / 'logs/frontier/p11_event_study/bravo_turn023_network_upgrade.json'
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, 'w') as f:
    summary = {
        'n_events_input': len(events),
        'per_split_horizon': results['per_split_horizon'],
        'null': results['null'],
        'per_asset_summary': {k: {str(h): v for h,v in vv.items()} for k, vv in results['per_asset_horizon'].items()},
    }
    json.dump(summary, f, indent=2, default=str)
print(f'\n[SAVE] {OUT}')
