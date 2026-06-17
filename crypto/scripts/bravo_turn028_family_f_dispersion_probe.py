"""
Bravo turn 028 -- NEW PROBE -- Family F cross-asset dispersion (per user
"new probe" directive at session close).

Mechanism (asymmetric framing, untested per CLAUDE.md Family F gap):
  - When U50 cross-sectional return dispersion (std) is HIGH, idiosyncratic
    moves dominate -> stock-picker alpha is available
  - When dispersion is LOW, beta dominates -> top-K rotation = noise
  - Family F: BUY top-K by recent momentum ONLY in high-dispersion regime;
    stay in cash in low-dispersion regime

Spot-only compliant: only long-side, no shorts. Per session learning, regime-
conditional sleeves on the existing blend show no lift (orthogonality), but
THIS is a regime-conditional NEW SLEEVE (not a meta-multiplier on existing).
That's a fundamentally different shape.

Setup:
  - Universe: U50 (per strat_test_min_universe constitution rule)
  - Daily dispersion: std of daily returns across U50 assets
  - Regime: top-quartile dispersion vs bottom-quartile
  - Strategy: BUY top-5 by 7d momentum on regime days, hold 5 days
  - Cost: 20bps round-trip
  - Splits: TRAIN 2020-23, VAL 2024, OOS 2025-26 (per session paranoid default)

Honest expectation: if 6 ortho evidences + 7 regime-death probes hold, this
is also likely to fail in 2025-26. But it's a genuinely untested asymmetric
mechanism (Family F was identified gap in CLAUDE.md). Fresh-eyes choice.
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

ROOT = Path('c:/Users/karab/Documents/coding/ml_systems')
sys.path.insert(0, str(ROOT))

from src.strategy.universe import UNIVERSE_50_LIQUID

COST_RT = 0.0020
SPLITS = {
    'TRAIN': ('2020-01-01','2023-12-31'),
    'VAL':   ('2024-01-01','2024-12-31'),
    'OOS':   ('2025-01-01','2026-04-19'),
}


def load_panel():
    rows = []
    for asset in UNIVERSE_50_LIQUID:
        cache = ROOT / 'logs/frontier/cycle_gate' / f'{asset.lower()}usdt_daily_klines.parquet'
        if not cache.exists():
            cache = ROOT / 'logs/frontier/cycle_gate' / f'{asset.lower()}_daily_klines.parquet'
            if not cache.exists():
                continue
        df = pd.read_parquet(cache)
        df['date'] = pd.to_datetime(df['date']).dt.normalize()
        df['asset'] = asset
        rows.append(df[['date','close','asset']])
    return pd.concat(rows, ignore_index=True).sort_values(['asset','date']).reset_index(drop=True)


def stats(arr, label=''):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n < 3:
        return {'label': label, 'n': n, 'status': 'thin'}
    eq = (1+arr).cumprod()
    peak = np.maximum.accumulate(eq)
    dd = ((eq-peak)/peak).min()
    days = n  # entries (per-trade not per-day)
    mu = arr.mean() * 365
    sd = arr.std(ddof=1) * np.sqrt(365)
    sharpe = mu/sd if sd > 0 else 0
    cagr = eq[-1]**(365.0/days) - 1
    hit = (arr > 0).mean()
    return {'label':label, 'n':n, 'cagr':cagr, 'sharpe':sharpe, 'dd':dd, 'hit':hit, 'mean_pct':arr.mean()*100, 't_stat': arr.mean()/(arr.std()/np.sqrt(n)) if arr.std()>0 else 0}


def main():
    panel = load_panel()
    print(f'Loaded panel: {panel["asset"].nunique()} assets, {panel["date"].min().date()} -> {panel["date"].max().date()}, {len(panel)} rows')

    # daily returns per asset
    panel = panel.sort_values(['asset','date']).reset_index(drop=True)
    panel['ret_1d'] = panel.groupby('asset')['close'].transform(lambda s: s.pct_change())
    panel['ret_7d'] = panel.groupby('asset')['close'].transform(lambda s: s.pct_change(7))
    panel['ret_5d_fwd'] = panel.groupby('asset')['close'].transform(lambda s: s.shift(-5)/s - 1)

    # daily dispersion: cross-sectional std of 1d returns
    disp = panel.groupby('date').agg(
        n_assets=('asset','count'),
        ret_std=('ret_1d', 'std'),
        ret_mean=('ret_1d','mean'),
    ).reset_index()

    # quartile labels (within full sample)
    disp['disp_quartile'] = pd.qcut(disp['ret_std'].fillna(0), 4, labels=['Q1_low','Q2','Q3','Q4_high'])

    # for each day, pick top-5 by 7d momentum and compute basket 5d-fwd return
    panel = panel.merge(disp[['date','disp_quartile','ret_std']], on='date', how='left')

    # Top-5 basket per day (forward 5d return EW basket)
    def basket_ret(g):
        # rank by 7d momentum desc, take top 5
        ranked = g.dropna(subset=['ret_7d','ret_5d_fwd']).nlargest(5, 'ret_7d')
        if len(ranked) < 3:
            return np.nan
        return ranked['ret_5d_fwd'].mean()

    print('\nComputing top-5 basket per day (this takes a moment)...')
    daily_basket = panel.groupby('date').apply(basket_ret).reset_index(name='basket_5d_fwd')
    daily_basket = daily_basket.merge(disp[['date','disp_quartile','ret_std']], on='date', how='left')
    daily_basket['basket_net'] = daily_basket['basket_5d_fwd'] - 2*COST_RT

    print(f'Daily basket rows: {len(daily_basket)}; non-null: {daily_basket["basket_5d_fwd"].notna().sum()}')

    # by dispersion quartile across full window + per-split
    print('\n=== Family F probe: Top-5 momentum basket conditional on dispersion ===')
    for split_name, (lo, hi) in SPLITS.items():
        sub = daily_basket[(daily_basket['date'] >= lo) & (daily_basket['date'] <= hi)]
        print(f'\n--- {split_name}: {sub["date"].min().date() if len(sub) else "n/a"} -> {sub["date"].max().date() if len(sub) else "n/a"} ---')
        for q in ['Q1_low','Q2','Q3','Q4_high']:
            sub_q = sub[sub['disp_quartile']==q]
            arr = sub_q['basket_net'].dropna().values
            s = stats(arr, q)
            if s.get('status') == 'thin':
                print(f'  {q:<10s}  n={s["n"]:4d}  thin')
            else:
                star = '*' if s['t_stat']>2 and s['mean_pct']>0 and s['hit']>0.5 else ' '
                print(f'  {q:<10s}  n={s["n"]:4d}  mean={s["mean_pct"]:+5.2f}%  t={s["t_stat"]:+5.2f}  hit={s["hit"]:.2f}  CAGR={s["cagr"]*100:+6.1f}%  Sh={s["sharpe"]:+5.2f}{star}')

    # also report flat top-K (regime-agnostic) for comparison
    print('\n=== Baseline: flat top-5 momentum (regime-agnostic) ===')
    for split_name, (lo, hi) in SPLITS.items():
        sub = daily_basket[(daily_basket['date'] >= lo) & (daily_basket['date'] <= hi)]
        arr = sub['basket_net'].dropna().values
        s = stats(arr, split_name)
        if s.get('status')=='thin':
            print(f'  {split_name}: thin')
        else:
            print(f'  {split_name}: n={s["n"]:4d}  mean={s["mean_pct"]:+5.2f}%  t={s["t_stat"]:+5.2f}  hit={s["hit"]:.2f}  CAGR={s["cagr"]*100:+6.1f}%  Sh={s["sharpe"]:+5.2f}')

    # save
    OUT = ROOT / 'logs/frontier/family_f_dispersion/turn028_probe.json'
    OUT.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        'n_panel_rows': int(len(panel)),
        'n_daily_basket': int(daily_basket['basket_5d_fwd'].notna().sum()),
    }
    with open(OUT, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'\n[SAVE] {OUT}')


if __name__ == '__main__':
    main()
