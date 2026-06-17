"""
Bravo turn 012 -- paranoid OOS validation of Alpha's E3 funding-flip.

Alpha's 3 candidates (full-window):
  ZEC h10d: n=111, +8.42%, t=2.80, hit 60%
  TRX h10d: n=229, +1.40%, t=2.70, hit 62%
  BNB  h3d: n=160, +1.61%, t=2.09, hit 58%

FDR concern: 3 of 45 tests at nominal alpha=0.05 = 2.25 expected false positives.
Chronological split required before ship.

Splits:
  TRAIN: funding-series-start .. 2023-12-31
  VAL:   2024-01-01 .. 2024-12-31
  OOS:   2025-01-01 .. 2026-04-19
"""
# [!] SPLIT DISCIPLINE NOTE (2026-05-24 INST-C cleanup):
# This script uses the legacy convention where "OOS" labels the post-TRAIN window
# (= canonical OOS + UNSEEN combined). Per src/split_config.py the canonical OOS
# ends 2025-12-31 and UNSEEN starts 2026-01-01. The dates hardcoded below are
# intentionally preserved for reproducibility of prior outputs. New scripts must
# import from split_config -- see docs/SPLIT_DISCIPLINE.md.
from __future__ import annotations

import json, sys, io
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path('c:/Users/karab/Documents/coding/v4_crypto_stystem')
COST_PCT = 0.0010

CANDIDATES = [
    ('ZEC', 10),
    ('TRX', 10),
    ('BNB', 3),
]

SPLITS = {
    'TRAIN': ('2020-01-01', '2023-12-31'),
    'VAL':   ('2024-01-01', '2024-12-31'),
    'OOS':   ('2025-01-01', '2026-04-19'),
}


def load_funding():
    return pd.read_parquet(ROOT / 'data/frontier/funding/funding_panel_daily.parquet')


def load_px(asset):
    cache = ROOT / 'logs/frontier/cycle_gate' / f'{asset.lower()}usdt_daily_klines.parquet'
    if cache.exists():
        return pd.read_parquet(cache)
    return None


def find_fund_col(fund_df, asset):
    # Funding panel columns: look for asset match
    for c in fund_df.columns:
        if asset.upper() in c.upper() and 'funding' in c.lower():
            return c
    # try lowercase
    for c in fund_df.columns:
        if c.lower().startswith(asset.lower()):
            return c
    return None


def prep(asset, fund_df):
    col = find_fund_col(fund_df, asset)
    if col is None:
        # look in columns list
        return None, f'no_fund_col (cols={list(fund_df.columns)[:5]}...)'
    fd = fund_df[['date', col]].copy().dropna().reset_index(drop=True)
    fd['fund_prev'] = fd[col].shift(1)
    fd['flip_neg'] = (fd[col] < 0) & (fd['fund_prev'] >= 0)
    px = load_px(asset)
    if px is None:
        return None, 'no_px'
    fd['date'] = pd.to_datetime(fd['date']).dt.normalize()
    px['date'] = pd.to_datetime(px['date']).dt.normalize()
    df = fd.merge(px[['date','close']], on='date', how='inner').sort_values('date').reset_index(drop=True)
    if len(df) < 100:
        return None, f'thin:{len(df)}'
    for h in (1,3,5,10):
        df[f'fwd_{h}d'] = df['close'].shift(-h) / df['close'] - 1.0
    return df, None


def split_stats(df, split_name, h):
    lo, hi = SPLITS[split_name]
    mask = (df['date'] >= lo) & (df['date'] <= hi) & df['flip_neg']
    sub = df[mask].copy()
    n_trig = len(sub)
    arr = (sub[f'fwd_{h}d'] - 2*COST_PCT).dropna().values
    n = len(arr)
    if n < 3:
        return {'n_trig': n_trig, 'n': n, 'status': 'thin'}
    mean = arr.mean()
    std = arr.std()
    t = mean/(std/np.sqrt(n)) if std > 0 else 0.0
    hit = (arr > 0).mean()
    return {'n_trig': n_trig, 'n': n, 'mean_pct': mean*100, 't_stat': t, 'hit_rate': hit}


def main():
    fund_df = load_funding()
    print(f'Funding panel shape: {fund_df.shape}')
    print(f'First 10 cols: {list(fund_df.columns)[:10]}')
    print()

    results = {}
    for asset, h in CANDIDATES:
        print(f'=== {asset} h{h}d ===')
        df, err = prep(asset, fund_df)
        if err:
            print(f'  ERROR: {err}')
            results[f'{asset}_h{h}'] = {'error': err}
            continue
        per = {}
        for split in ('TRAIN','VAL','OOS'):
            s = split_stats(df, split, h)
            per[split] = s
            if s.get('status') == 'thin':
                print(f'  {split}: n_trig={s["n_trig"]} n={s["n"]} thin')
            else:
                star = '*' if s['t_stat'] > 2 and s['mean_pct'] > 0 and s['hit_rate'] > 0.5 else ' '
                print(f'  {split}: n_trig={s["n_trig"]:3d}  n={s["n"]:3d}  mean={s["mean_pct"]:+6.2f}%  t={s["t_stat"]:+5.2f}  hit={s["hit_rate"]:.2f}{star}')
        passes = sum(1 for spl in ('TRAIN','VAL','OOS') if per[spl].get('t_stat',0)>2 and per[spl].get('mean_pct',0)>0 and per[spl].get('hit_rate',0)>0.5)
        if passes == 3:
            v = 'PASS_3of3'
        elif passes == 2:
            v = 'PARTIAL_2of3'
        elif passes == 1:
            v = 'FRAGILE_1of3'
        else:
            v = 'FAIL_0of3'
        print(f'  VERDICT: {v}')
        results[f'{asset}_h{h}'] = {'per_split': per, 'verdict': v}
        print()

    OUT = ROOT / 'logs/frontier/e3_funding_flip/bravo_turn012_paranoid_oos.json'
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'[SAVE] {OUT}')


if __name__ == '__main__':
    main()
