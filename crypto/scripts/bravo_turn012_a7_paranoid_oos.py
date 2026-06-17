"""
Bravo turn 012 -- paranoid OOS validation of Alpha's A7 liq-cascade 3-7d MR.

Alpha's full-window result (2020-2026 concat): BTC h7d +5.93%/t=4.41/hit 81% (n=21),
ADA h7d +9.24%/t=3.10/hit 72% (n=60), SOL/XRP/AVAX also ship. LTC is negative
control. 10 U10 assets only (liq data coverage).

THIS SCRIPT: chronological split + shuffle-entry-date control.

Splits (after liq data starts 2020-01):
  TRAIN: 2020-01-01 .. 2023-12-31  (~4yr)
  VAL:   2024-01-01 .. 2024-12-31  (1yr)
  OOS:   2025-01-01 .. 2026-04-19  (~16mo)

Per-asset per-split: recompute mean_pct / t_stat / hit_rate / n for h3d, h5d, h7d.

Shuffle-entry control: randomize which days are "triggers" (preserving trigger
count) and re-run 20 times -> null distribution of t_stat. Real t_stat must
exceed ~95th percentile of null.

Verdict per asset:
  PASS: t>2, mean>0, hit>0.5 in IS + VAL + OOS (all 3) AND shuffle control
        says t > 95th percentile of null
  PARTIAL: passes in 2/3 chronological splits
  FAIL: passes in 0-1 splits
"""
# [!] SPLIT DISCIPLINE NOTE (2026-05-24 INST-C cleanup):
# This script uses the legacy convention where "OOS" labels the post-TRAIN window
# (= canonical OOS + UNSEEN combined). Per src/split_config.py the canonical OOS
# ends 2025-12-31 and UNSEEN starts 2026-01-01. The dates hardcoded below are
# intentionally preserved for reproducibility of prior outputs. New scripts must
# import from split_config -- see docs/SPLIT_DISCIPLINE.md.
from __future__ import annotations

import json
import sys
import io
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path('c:/Users/karab/Documents/coding/v4_crypto_stystem')

CASCADE_LOOKBACK = 20
CASCADE_MULT = 3.0
DELAY_DAYS = 2
COST_PCT = 0.0010
N_SHUFFLE = 40  # null sims
RNG = np.random.default_rng(42)

SPLITS = {
    'TRAIN': ('2020-01-01', '2023-12-31'),
    'VAL':   ('2024-01-01', '2024-12-31'),
    'OOS':   ('2025-01-01', '2026-04-19'),
}


def load_liq() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data/frontier/liquidations/liq_daily_approx.parquet")


def load_daily_price(asset: str) -> pd.DataFrame | None:
    cache = ROOT / "logs/frontier/cycle_gate" / f"{asset.lower()}usdt_daily_klines.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    return None


def prep_asset_panel(asset: str, liq: pd.DataFrame) -> pd.DataFrame | None:
    liq_a = liq[liq['asset'] == asset].copy().sort_values('date').reset_index(drop=True)
    if len(liq_a) < 50:
        return None
    px = load_daily_price(asset)
    if px is None:
        return None
    liq_a['liq_ma20'] = liq_a['liq_total_usd'].rolling(CASCADE_LOOKBACK, min_periods=10).mean()
    liq_a['spike'] = liq_a['liq_total_usd'] > (CASCADE_MULT * liq_a['liq_ma20'])
    liq_a['spike'] = liq_a['spike'].fillna(False)
    px['date'] = pd.to_datetime(px['date']).dt.normalize()
    liq_a['date'] = pd.to_datetime(liq_a['date']).dt.normalize()
    df = liq_a.merge(px[['date','close']], on='date', how='inner').sort_values('date').reset_index(drop=True)
    if len(df) < 50:
        return None
    df['spike_lag2'] = df['spike'].shift(DELAY_DAYS).fillna(False).astype(bool)
    for h in (3,5,7):
        df[f'fwd_{h}d'] = df['close'].shift(-h) / df['close'] - 1.0
    return df


def compute_split(df: pd.DataFrame, split_name: str, triggers_col: str = 'spike_lag2') -> dict:
    lo, hi = SPLITS[split_name]
    mask = (df['date'] >= lo) & (df['date'] <= hi)
    sub = df[mask & df[triggers_col]].copy()
    n_trig = len(sub)
    out = {'n_trig': n_trig}
    for h in (3,5,7):
        col = f'fwd_{h}d'
        arr = (sub[col] - 2*COST_PCT).dropna().values
        n = len(arr)
        if n < 3:
            out[f'h{h}d'] = {'n': n, 'status': 'thin'}
            continue
        mean = float(arr.mean())
        std = float(arr.std())
        t = mean / (std/np.sqrt(n)) if std > 0 else 0.0
        hit = float((arr > 0).mean())
        out[f'h{h}d'] = {'n': n, 'mean_pct': mean*100, 't_stat': t, 'hit_rate': hit}
    return out


def shuffle_null(df: pd.DataFrame, split_name: str, h: int, n_shuffle: int = N_SHUFFLE) -> dict:
    """Shuffle entry-dates (preserve count) -- null distribution of t_stat."""
    lo, hi = SPLITS[split_name]
    mask = (df['date'] >= lo) & (df['date'] <= hi)
    sub = df[mask].copy().reset_index(drop=True)
    real_triggers = int(sub['spike_lag2'].sum())
    if real_triggers < 3:
        return {'status':'thin'}
    ts = []
    fwd_col = f'fwd_{h}d'
    for _ in range(n_shuffle):
        rand_idx = RNG.choice(len(sub), size=real_triggers, replace=False)
        arr = (sub[fwd_col].iloc[rand_idx] - 2*COST_PCT).dropna().values
        n = len(arr)
        if n < 3:
            continue
        mean = arr.mean()
        std = arr.std()
        t_null = mean / (std/np.sqrt(n)) if std > 0 else 0.0
        ts.append(t_null)
    if not ts:
        return {'status':'thin'}
    ts_arr = np.array(ts)
    return {
        'null_t_mean': float(ts_arr.mean()),
        'null_t_std': float(ts_arr.std()),
        'null_t_p95': float(np.percentile(ts_arr, 95)),
        'null_t_p99': float(np.percentile(ts_arr, 99)),
        'n_shuffles': len(ts_arr),
    }


def verdict(per_split: dict) -> str:
    passes = 0
    for split_name in ('TRAIN','VAL','OOS'):
        spl = per_split.get(split_name, {})
        # at least one horizon passes t>2, mean>0, hit>0.5
        any_pass = False
        for h in (3,5,7):
            r = spl.get(f'h{h}d', {})
            if r.get('status') == 'thin':
                continue
            if r.get('t_stat',0) > 2 and r.get('mean_pct',0) > 0 and r.get('hit_rate',0) > 0.5:
                any_pass = True
                break
        if any_pass:
            passes += 1
    if passes == 3:
        return 'PASS_3of3'
    elif passes == 2:
        return 'PARTIAL_2of3'
    elif passes == 1:
        return 'FRAGILE_1of3'
    else:
        return 'FAIL_0of3'


def main() -> None:
    liq = load_liq()
    assets = sorted(liq['asset'].unique())
    print(f'Running A7 paranoid OOS on {len(assets)} assets with liq coverage...\n')
    all_results = {}
    for asset in assets:
        df = prep_asset_panel(asset, liq)
        if df is None:
            all_results[asset] = {'error':'prep_failed'}
            print(f'{asset:>5s}: PREP FAILED')
            continue
        per_split = {}
        for split in ('TRAIN','VAL','OOS'):
            per_split[split] = compute_split(df, split)
        v = verdict(per_split)
        all_results[asset] = {'per_split': per_split, 'verdict': v}
        print(f'=== {asset} ===  verdict: {v}')
        for split in ('TRAIN','VAL','OOS'):
            spl = per_split[split]
            nt = spl['n_trig']
            parts = [f'n_trig={nt:3d}']
            for h in (3,5,7):
                r = spl.get(f'h{h}d', {})
                if r.get('status') == 'thin':
                    parts.append(f'h{h}:thin')
                    continue
                m = r.get('mean_pct',0)
                t = r.get('t_stat',0)
                hit = r.get('hit_rate',0)
                star = '*' if t > 2 and m > 0 and hit > 0.5 else ' '
                parts.append(f'h{h}:m={m:+5.2f}% t={t:+4.2f} hit={hit:.2f}{star}')
            print(f'  {split:<6s}: ' + '  '.join(parts))
        print()

    # Shuffle-null for the strongest candidates (PASS_3of3 or PARTIAL_2of3)
    print('\n=== Shuffle-null controls for strong candidates (OOS only) ===')
    for asset, r in all_results.items():
        if 'error' in r:
            continue
        if r['verdict'] not in ('PASS_3of3','PARTIAL_2of3'):
            continue
        df = prep_asset_panel(asset, liq)
        for h in (3,5,7):
            real_r = r['per_split']['OOS'].get(f'h{h}d', {})
            if real_r.get('status') == 'thin':
                continue
            real_t = real_r.get('t_stat', 0)
            if real_t <= 2:
                continue
            null = shuffle_null(df, 'OOS', h)
            if null.get('status') == 'thin':
                continue
            p95 = null['null_t_p95']
            pass_null = real_t > p95
            print(f'  {asset} h{h}d OOS: real_t={real_t:+.2f} vs null_p95={p95:+.2f} -> {"PASS" if pass_null else "FAIL-shuffle"}')

    # Summary
    print('\n=== VERDICT SUMMARY ===')
    summary_rows = []
    for asset, r in all_results.items():
        if 'error' in r:
            summary_rows.append((asset, 'ERROR', r['error']))
        else:
            summary_rows.append((asset, r['verdict'], ''))
    # sort by verdict priority
    priority = {'PASS_3of3':0,'PARTIAL_2of3':1,'FRAGILE_1of3':2,'FAIL_0of3':3,'ERROR':4}
    summary_rows.sort(key=lambda x: (priority.get(x[1],9), x[0]))
    for asset, v, extra in summary_rows:
        print(f'  {asset:<5s}  {v:<14s}  {extra}')

    # Dump full
    OUT = ROOT / 'logs/frontier/a7_liq_cascade/bravo_turn012_paranoid_oos.json'
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\n[SAVE] {OUT}')


if __name__ == '__main__':
    main()
