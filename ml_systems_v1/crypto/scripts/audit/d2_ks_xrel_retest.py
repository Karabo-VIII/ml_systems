"""d2_ks_xrel_retest.py -- Re-run winner-vs-non-mover KS discriminator test with
xrel_ features included. Populates ks_winner_v_nonmover in feature_catalog.parquet.

Test: for each candidate feature, does it discriminate (asset, date) WINNER (+5% next-day)
from (asset, date) NON-MOVER (|1d_ret| < 1%) at the same 4h/1h microstructure horizon?

D-2 (yesterday): 0 of 48 norm_*-only features achieved KS > 0.15
D-2 v2 (today, with xrel_): expect xrel_*_xrank to score KS 0.10-0.13 per sister's docs
"""
from __future__ import annotations

import sys, io, json, random
from datetime import datetime, timezone
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import polars as pl, pandas as pd, numpy as np
from scipy.stats import ks_2samp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHIM_DIR = PROJECT_ROOT / 'data' / 'processed' / 'chimera' / 'dollar'
OUT_DIR = PROJECT_ROOT / 'runs' / 'audit' / 'D2_KS_XREL_RETEST'
OUT_DIR.mkdir(parents=True, exist_ok=True)
CATALOG_FP = PROJECT_ROOT / 'data' / 'processed' / 'feature_catalog.parquet'

# Use 1d-derived returns to define winner vs non-mover.
# Then sample T-1 features from the asset's dollar chimera.
# Test features = ALL columns in feature_catalog EXCEPT target_, is_, asset_, prefixes
# that are metadata not predictive features.
EXCLUDE_PREFIXES = ('target_', 'is_', 'asset_', 'sector_', 'tier_')
EXCLUDE_EXACT = {'timestamp','bar_id','open','high','low','close','volume',
                  'volume_usd','buy_vol','sell_vol','date'}

WINDOW_START_MS = int(pd.Timestamp('2025-01-01').timestamp() * 1000)
WINDOW_END_MS = int(pd.Timestamp('2026-05-15').timestamp() * 1000)
MOVE_THRESHOLD = 0.05
NONMOVE_THRESHOLD = 0.01

WINNERS_TARGET = 500
NONMOVERS_TARGET = 500


def get_dollar_for_asset(asset: str) -> pl.DataFrame | None:
    fps = sorted(CHIM_DIR.glob(f'{asset.lower()}usdt_v51_chimera_*.parquet'))
    if not fps: return None
    return pl.read_parquet(fps[-1])


def sample_winner_features(asset: str, df_dollar: pl.DataFrame, candidate_dates: list, feat_cols: list) -> list[dict]:
    """For each winner date, get the LAST DOLLAR BAR on the day BEFORE the move (T-1)."""
    out = []
    for d in candidate_dates:
        d0 = pd.Timestamp(d).normalize()
        ts_t1_lo = int((d0 - pd.Timedelta(days=1)).timestamp() * 1000)
        ts_t1_hi = int(d0.timestamp() * 1000) - 1
        sub = df_dollar.filter((pl.col('timestamp') >= ts_t1_lo) & (pl.col('timestamp') <= ts_t1_hi))
        if sub.height == 0: continue
        last = sub.tail(1).row(0, named=True)
        row = {'asset': asset, 'date': str(d), 'label': 'winner'}
        for f in feat_cols:
            row[f] = last.get(f)
        out.append(row)
    return out


def sample_nonmover_features(asset: str, df_dollar: pl.DataFrame, candidate_dates: list, feat_cols: list) -> list[dict]:
    """For each non-mover date, get the LAST DOLLAR BAR on the day BEFORE."""
    return [{'asset': asset, 'date': str(d), 'label': 'nonmover',
             **{f: (r := df_dollar.filter(
                 (pl.col('timestamp') >= int((pd.Timestamp(d).normalize() - pd.Timedelta(days=1)).timestamp() * 1000))
                 & (pl.col('timestamp') <= int(pd.Timestamp(d).normalize().timestamp() * 1000) - 1)
             ).tail(1)).row(0, named=True).get(f) if r.height > 0 else None for f in feat_cols}}
            for d in candidate_dates
            if df_dollar.filter(
                (pl.col('timestamp') >= int((pd.Timestamp(d).normalize() - pd.Timedelta(days=1)).timestamp() * 1000))
                & (pl.col('timestamp') <= int(pd.Timestamp(d).normalize().timestamp() * 1000) - 1)
            ).height > 0]


def main():
    print('=' * 72)
    print('D-2 v2 KS RE-TEST -- WINNER vs NON-MOVER w/ xrel_ features')
    print('=' * 72)

    # Load feature catalog
    catalog = pd.read_parquet(CATALOG_FP)
    feat_cols = [r['feature_name'] for _, r in catalog.iterrows()
                 if r['feature_name'] not in EXCLUDE_EXACT
                 and not any(r['feature_name'].startswith(p) for p in EXCLUDE_PREFIXES)]
    print(f'Feature catalog: {len(catalog)} total; {len(feat_cols)} testable')

    # Build winner + non-mover (asset, date) samples using 1d chimera for return computation
    print('\nSampling 1d returns to find winners + non-movers...')
    chim_1d = PROJECT_ROOT / 'data' / 'processed' / 'chimera' / '1d'
    winner_pairs = []  # (asset, date)
    nonmover_pairs = []
    fps_1d = sorted(chim_1d.glob('*usdt_v51_chimera_1d_*.parquet'))
    for fp in fps_1d:
        asset = fp.stem.split('usdt_v51_chimera_1d')[0].upper()
        df = pl.read_parquet(fp, columns=['timestamp','close']).with_columns([
            (pl.col('close') / pl.col('close').shift(1) - 1).alias('ret_1d'),
            pl.from_epoch(pl.col('timestamp'), time_unit='ms').dt.date().alias('date'),
        ])
        df = df.filter((pl.col('timestamp') >= WINDOW_START_MS) & (pl.col('timestamp') <= WINDOW_END_MS))
        wins = df.filter(pl.col('ret_1d') >= MOVE_THRESHOLD)
        nms = df.filter((pl.col('ret_1d').abs() < NONMOVE_THRESHOLD) & (pl.col('ret_1d').is_not_null()))
        for d in wins['date'].to_list():
            winner_pairs.append((asset, d))
        for d in nms['date'].to_list():
            nonmover_pairs.append((asset, d))

    print(f'  total winners: {len(winner_pairs):,}')
    print(f'  total non-movers: {len(nonmover_pairs):,}')

    random.seed(42)
    winner_sample = random.sample(winner_pairs, min(WINNERS_TARGET, len(winner_pairs)))
    nonmover_sample = random.sample(nonmover_pairs, min(NONMOVERS_TARGET, len(nonmover_pairs)))
    print(f'  sampled: {len(winner_sample)} winners + {len(nonmover_sample)} non-movers')

    # Extract T-1 features for both
    print('\nExtracting T-1 features...')
    by_asset = {}
    for asset, d in winner_sample + nonmover_sample:
        by_asset.setdefault(asset, []).append((d, 'winner' if (asset, d) in set(winner_sample) else 'nonmover'))
    win_set = set(winner_sample)

    rows = []
    n_proc = 0
    for asset, dates_lbls in by_asset.items():
        df_d = get_dollar_for_asset(asset)
        if df_d is None: continue
        for d, _ in dates_lbls:
            label = 'winner' if (asset, d) in win_set else 'nonmover'
            d0 = pd.Timestamp(d).normalize()
            ts_lo = int((d0 - pd.Timedelta(days=1)).timestamp() * 1000)
            ts_hi = int(d0.timestamp() * 1000) - 1
            sub = df_d.filter((pl.col('timestamp') >= ts_lo) & (pl.col('timestamp') <= ts_hi))
            if sub.height == 0: continue
            last = sub.tail(1).row(0, named=True)
            row = {'asset': asset, 'date': str(d), 'label': label}
            for f in feat_cols:
                row[f] = last.get(f)
            rows.append(row)
        n_proc += 1
        if n_proc % 20 == 0:
            print(f'  scanned {n_proc}/{len(by_asset)} assets | {len(rows)} samples')

    print(f'\n[done] {len(rows)} samples extracted')

    df = pd.DataFrame(rows)
    n_win = (df['label'] == 'winner').sum()
    n_nm  = (df['label'] == 'nonmover').sum()
    print(f'  winners: {n_win}, non-movers: {n_nm}')

    # KS test per feature
    results = []
    win_df = df[df['label'] == 'winner']
    nm_df = df[df['label'] == 'nonmover']
    for f in feat_cols:
        h = pd.to_numeric(win_df[f], errors='coerce').dropna().to_numpy()
        n = pd.to_numeric(nm_df[f], errors='coerce').dropna().to_numpy()
        if len(h) < 50 or len(n) < 50:
            results.append({'feature': f, 'ks': float('nan'), 'p': float('nan'),
                              'n_win': len(h), 'n_nm': len(n)})
            continue
        ks = ks_2samp(h, n)
        results.append({'feature': f, 'ks': float(ks[0]), 'p': float(ks[1]),
                          'n_win': len(h), 'n_nm': len(n)})

    df_res = pd.DataFrame(results).sort_values('ks', ascending=False)
    df_res.to_csv(OUT_DIR / 'ks_results.csv', index=False)
    print(f'\n[wrote] {OUT_DIR / "ks_results.csv"}')

    # Top-25 features by KS
    print(f'\nTOP 25 DISCRIMINATORS (winner vs non-mover):')
    print(f'{"feature":<45} {"KS":>8} {"p":>10} {"n_win":>7} {"n_nm":>7}')
    print('-' * 80)
    for _, r in df_res.head(25).iterrows():
        if pd.isna(r['ks']): continue
        print(f'{r["feature"]:<45} {r["ks"]:>8.4f} {r["p"]:>10.2e} {r["n_win"]:>7d} {r["n_nm"]:>7d}')

    # Count of features above key thresholds
    n_above_015 = (df_res['ks'] > 0.15).sum()
    n_above_010 = (df_res['ks'] > 0.10).sum()
    n_above_005 = (df_res['ks'] > 0.05).sum()
    print(f'\nFeatures with KS > 0.15: {n_above_015} (yesterday: 0)')
    print(f'Features with KS > 0.10: {n_above_010}')
    print(f'Features with KS > 0.05: {n_above_005}')

    # xrel_ specific
    xrel_results = df_res[df_res['feature'].str.startswith('xrel_')].copy()
    print(f'\n[xrel_ family] {len(xrel_results)} features tested')
    print(xrel_results.head(10).to_string(index=False))

    # Populate catalog with KS
    catalog['ks_winner_v_nonmover'] = catalog['feature_name'].map(
        df_res.set_index('feature')['ks'].to_dict()
    )
    catalog.to_parquet(CATALOG_FP, index=False)
    print(f'\n[updated] {CATALOG_FP} with KS column')

    (OUT_DIR / 'summary.json').write_text(json.dumps({
        'generated': datetime.now(timezone.utc).isoformat(),
        'n_winners_sampled': int(n_win),
        'n_nonmovers_sampled': int(n_nm),
        'n_features_tested': len(feat_cols),
        'n_ks_gt_015': int(n_above_015),
        'n_ks_gt_010': int(n_above_010),
        'n_ks_gt_005': int(n_above_005),
        'top_5_features': df_res.head(5)[['feature','ks']].to_dict(orient='records'),
    }, indent=2))


if __name__ == '__main__':
    main()
