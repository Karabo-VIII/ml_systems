"""
Bravo turn 015 -- U50 cross-asset breakout-density probe (D6 novel angle).

Mechanism: daily count of U50 assets breaking above N-day high is a regime
meter of synchronized-bull expansion. Hypothesis: when density >= threshold,
the cohort that just broke out continues momentum 3-7 days. When density is
low, either stay flat (no entry) or use short-only (not allowed under SPOT).

This is cross-sectional-synchronization signal, distinct from per-asset
breakouts (which asym_breakout sleeve already captures). Tests whether an
AGGREGATE signal across U50 provides additional regime information the
existing per-asset breakout sleeve doesn't.

Setup (D6-compliant hold 3-7d):
  - Compute daily breakout_density_N: count of U50 assets whose close >
    trailing N-day high (excluding today).
  - Entry: on days where breakout_density_N >= density_threshold
  - Basket: equal-weight the assets that broke out that day
  - Hold: 5 trading days
  - 20 bps round-trip cost
  - Chronological split: TRAIN 2020-23, VAL 2024, OOS 2025-26

Paranoid gate: if regime-concentrated (works 2020-24, dies 2025-26 per D6
pattern), CONCEDE. Only report as candidate if OOS alone clears t>2 at n>=20.
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

sys.path.insert(0, str(ROOT))
from src.strategy.universe import UNIVERSE_50_LIQUID

COST_PCT = 0.0010  # per side, 10 bps -> 20 bps round-trip
HOLD_DAYS = 5
BREAKOUT_WINDOW = 20

SPLITS = {
    'TRAIN': ('2020-01-01', '2023-12-31'),
    'VAL':   ('2024-01-01', '2024-12-31'),
    'OOS':   ('2025-01-01', '2026-04-19'),
}


def load_daily(asset: str) -> pd.DataFrame | None:
    cache = ROOT / 'logs/frontier/cycle_gate' / f'{asset.lower()}usdt_daily_klines.parquet'
    if not cache.exists():
        cache = ROOT / 'logs/frontier/cycle_gate' / f'{asset.lower()}_daily_klines.parquet'
        if not cache.exists():
            return None
    df = pd.read_parquet(cache)
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    return df[['date','close']].sort_values('date').reset_index(drop=True)


def build_panel() -> pd.DataFrame:
    """Return a long panel [date, asset, close] for all U50 assets."""
    rows = []
    for asset in UNIVERSE_50_LIQUID:
        df = load_daily(asset)
        if df is None:
            continue
        df['asset'] = asset
        rows.append(df)
    panel = pd.concat(rows, ignore_index=True)
    return panel


def compute_breakout_flags(panel: pd.DataFrame, N: int = BREAKOUT_WINDOW) -> pd.DataFrame:
    """Add per-asset breakout_today flag: close > N-day trailing high (excluding today)."""
    panel = panel.sort_values(['asset','date']).reset_index(drop=True)
    # rolling max over past N days, excluding today
    panel['rolling_max'] = panel.groupby('asset')['close'].transform(
        lambda s: s.shift(1).rolling(N, min_periods=max(5, N//2)).max()
    )
    panel['breakout_today'] = (panel['close'] > panel['rolling_max']).fillna(False)
    return panel


def compute_density(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-day density across all assets with data that day."""
    grp = panel.groupby('date').agg(
        n_assets=('asset','count'),
        n_breakouts=('breakout_today','sum'),
    ).reset_index()
    grp['density'] = grp['n_breakouts'] / grp['n_assets']
    return grp


def compute_forward_returns(panel: pd.DataFrame, hold: int = HOLD_DAYS) -> pd.DataFrame:
    panel = panel.sort_values(['asset','date']).reset_index(drop=True)
    panel[f'fwd_{hold}d'] = panel.groupby('asset')['close'].transform(lambda s: s.shift(-hold) / s - 1.0)
    return panel


def probe(panel: pd.DataFrame, density: pd.DataFrame, threshold: float, split_name: str) -> dict:
    """For days in split where density >= threshold, collect the forward returns
    of the assets that broke out that day, EW averaged."""
    lo, hi = SPLITS[split_name]
    # days passing threshold
    days = density[(density['density'] >= threshold) & (density['date'] >= lo) & (density['date'] <= hi)]['date'].values
    # pick the broken-out assets on those days
    trig = panel[(panel['date'].isin(days)) & (panel['breakout_today'])].copy()
    if len(trig) == 0:
        return {'split': split_name, 'n_events': 0, 'status': 'no_triggers'}
    # per-day basket return (EW of those-that-broke-out)
    fwd_col = f'fwd_{HOLD_DAYS}d'
    basket = trig.groupby('date')[fwd_col].mean().dropna()
    if len(basket) < 5:
        return {'split': split_name, 'n_events': len(basket), 'status': 'thin'}
    # Apply 2x cost (enter + exit basket)
    net = basket - 2*COST_PCT
    n = len(net)
    mean = float(net.mean())
    std = float(net.std())
    t = mean / (std/np.sqrt(n)) if std > 0 else 0.0
    hit = float((net > 0).mean())
    return {
        'split': split_name,
        'n_events': n,
        'mean_pct': mean*100,
        't_stat': t,
        'hit_rate': hit,
        'std_pct': std*100,
    }


def main():
    print(f'Building U50 panel...')
    panel = build_panel()
    print(f'  {panel["asset"].nunique()} assets, {len(panel)} rows, {panel["date"].min().date()} -> {panel["date"].max().date()}')

    print(f'\nComputing {BREAKOUT_WINDOW}-day breakout flags + forward {HOLD_DAYS}d returns...')
    panel = compute_breakout_flags(panel, N=BREAKOUT_WINDOW)
    panel = compute_forward_returns(panel, hold=HOLD_DAYS)

    density = compute_density(panel)
    print(f'\nDensity distribution (overall):')
    print(f'  mean_density={density["density"].mean():.3f}')
    print(f'  p50={density["density"].median():.3f}')
    print(f'  p75={density["density"].quantile(0.75):.3f}')
    print(f'  p90={density["density"].quantile(0.90):.3f}')
    print(f'  p95={density["density"].quantile(0.95):.3f}')
    print(f'  max={density["density"].max():.3f}')
    print(f'  n_assets_per_day p5={density["n_assets"].quantile(0.05):.0f}, median={density["n_assets"].median():.0f}, max={density["n_assets"].max():.0f}')

    # Sweep thresholds
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30]
    print(f'\n=== Sweep thresholds × splits ===')
    results = {}
    for thr in thresholds:
        print(f'\nDensity >= {thr*100:.0f}%:')
        per_split = {}
        for split in ('TRAIN','VAL','OOS'):
            r = probe(panel, density, thr, split)
            per_split[split] = r
            if r.get('status'):
                print(f'  {split}: {r["status"]} (n_events={r["n_events"]})')
            else:
                star = '*' if r['t_stat'] > 2 and r['mean_pct'] > 0 and r['hit_rate'] > 0.5 else ' '
                print(f'  {split}: n_events={r["n_events"]:3d}  mean={r["mean_pct"]:+5.2f}%  t={r["t_stat"]:+4.2f}  hit={r["hit_rate"]:.2f}{star}')
        # verdict
        passes = sum(1 for s in ('TRAIN','VAL','OOS') if per_split[s].get('t_stat',0) > 2 and per_split[s].get('mean_pct',0) > 0 and per_split[s].get('hit_rate',0) > 0.5)
        if passes == 3:
            v = 'PASS_3of3'
        elif passes == 2:
            v = 'PARTIAL_2of3'
        elif passes == 1:
            v = 'FRAGILE_1of3'
        else:
            v = 'FAIL_0of3'
        print(f'  VERDICT: {v}')
        results[f'thr_{thr}'] = {'per_split': per_split, 'verdict': v}

    OUT = ROOT / 'logs/frontier/breakout_density/bravo_turn015_U50.json'
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\n[SAVE] {OUT}')


if __name__ == '__main__':
    main()
