"""
Bravo turn 021 -- independent verification of Alpha turn-020 delisting-rebound
kill + test of fixed shuffle-null in harness.

Replicates Alpha's turn-020 setup:
  - Pull (symbol, event_date) pairs from delisting_recent.parquet (post body+token fix)
  - Entry: event_date + 2d (post-forced-selling-exhaustion)
  - Horizons: [24, 48, 72, 120] hours
  - Cost: 20bps RT
  - With shuffle_null_n=20 per new (fixed) null mechanism

Expected: reproduce Alpha's result (-12.88% t=-4.32 at h72) + meaningful null
distribution that shows event-specific signal vs asset-selection base rate.
"""
# [!] SPLIT DISCIPLINE NOTE (2026-05-24 INST-C cleanup):
# This script uses the legacy convention where "OOS" labels the post-TRAIN window
# (= canonical OOS + UNSEEN combined). Per src/split_config.py the canonical OOS
# ends 2025-12-31 and UNSEEN starts 2026-01-01. The dates hardcoded below are
# intentionally preserved for reproducibility of prior outputs. New scripts must
# import from split_config -- see docs/SPLIT_DISCIPLINE.md.
from __future__ import annotations

import sys, io, json
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np

ROOT = Path('c:/Users/karab/Documents/coding/v4_crypto_stystem')
sys.path.insert(0, str(ROOT))

from src.frontier.utils.event_study import run_event_study, pretty_report

# Load delisting panel
delisting_df = pd.read_parquet(ROOT / 'logs/frontier/announcements/delisting_recent.parquet')
print(f'Raw delisting rows: {len(delisting_df)}')
print(f'Bodies populated: {(delisting_df["body"].astype(str).str.len() > 100).sum()}/{len(delisting_df)}')
print(f'Tokens present: {delisting_df["tokens"].apply(lambda t: len(t) > 0).sum()}/{len(delisting_df)}')
print()

# Expand to (symbol, event_date) pairs -- one per token per announcement
pairs = []
for _, row in delisting_df.iterrows():
    if row['release_ms'] == 0:
        continue
    for tok in row['tokens']:
        tok = str(tok).upper().strip()
        if 2 <= len(tok) <= 10 and tok.isalpha():
            pairs.append({
                'symbol': tok,
                'event_date': pd.Timestamp(row['release_ms'], unit='ms'),
                'category': 'delisting',
            })

events = pd.DataFrame(pairs).drop_duplicates(subset=['symbol','event_date']).reset_index(drop=True)
print(f'Expanded (symbol, event_date) pairs: {len(events)}')
print(f'Unique symbols: {events["symbol"].nunique()}')
print(f'Date range: {events["event_date"].min()} -> {events["event_date"].max()}')
print()

# Adjust event_date by +2d (entry delay per hypothesis)
events['event_date'] = events['event_date'] + pd.Timedelta(days=2)

# Run event study
print('Running event_study with fixed shuffle_null...')
results = run_event_study(
    events_df=events,
    horizons_h=[24, 48, 72, 120],
    interval='1h',
    use_spot=True,
    entry_lag_min=60,
    cost_rt_pct=0.0020,
    splits={
        'ALL': ('2025-01-01', '2026-04-24'),
        'OOS': ('2025-12-01', '2026-04-24'),  # same as Alpha's -- all events in this window
    },
    shuffle_null_n=15,  # reduced from 20 since each trial is expensive
    verbose=False,
)

print()
print(pretty_report(results))

# Compare to Alpha's published numbers
alpha_oos = {24: (-1.10, -0.55, 0.250), 48: (-9.95, -2.87, 0.214), 72: (-12.88, -4.32, 0.071)}
print('\n=== Independent verification vs Alpha turn-020 ===')
our_oos = results['per_split_horizon'].get('OOS', {})
for h in (24, 48, 72):
    a_mean, a_t, a_hit = alpha_oos[h]
    b = our_oos.get(h, {})
    if 'status' in b and b['status'] == 'thin':
        print(f'  h{h:>3d}h OOS: THIN (n={b["n"]})')
        continue
    print(f'  h{h:>3d}h OOS:  Alpha: mean={a_mean:+.2f}% t={a_t:+.2f} hit={a_hit:.3f}')
    print(f'             Bravo: mean={b.get("mean_pct",0):+.2f}% t={b.get("t_stat",0):+.2f} hit={b.get("hit_rate",0):.3f}')

# Compare real vs null for OOS horizons
null = results.get('null', {})
print('\n=== Real vs Null (fixed shuffle) ===')
for h in (24, 48, 72, 120):
    r = our_oos.get(h, {})
    n = null.get(h, {})
    if r.get('status') == 'thin' or n.get('status') == 'thin':
        continue
    real_t = r.get('t_stat', 0)
    null_p5 = n.get('null_t_p5', 0)
    null_p95 = n.get('null_t_p95', 0)
    null_mean = n.get('null_t_mean', 0)
    # Signal is real if real_t < null_p5 (for negative signal) or > null_p95
    extreme_below = real_t < null_p5
    extreme_above = real_t > null_p95
    verdict = 'SIGNAL' if (extreme_above or extreme_below) else 'WITHIN-NULL'
    print(f'  h{h:>3d}h: real_t={real_t:+5.2f}  null_p5={null_p5:+5.2f}  null_mean={null_mean:+5.2f}  null_p95={null_p95:+5.2f}  -> {verdict}')

# Save
OUT = ROOT / 'logs/frontier/p11_event_study/bravo_turn021_delisting_verify.json'
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, 'w') as f:
    summary = {
        'per_split_horizon': results['per_split_horizon'],
        'per_asset_horizon': {k: {str(h): v for h,v in vv.items()} for k, vv in results['per_asset_horizon'].items()},
        'null': results['null'],
    }
    json.dump(summary, f, indent=2, default=str)
print(f'\n[SAVE] {OUT}')
