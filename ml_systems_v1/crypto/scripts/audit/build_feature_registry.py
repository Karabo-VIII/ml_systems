"""build_feature_registry.py -- create canonical metadata catalog for v51 chimera features.

User mandate 2026-05-18: "I now need to start keeping metadata information, starting
with the features." Motivation: D-1+D-2 finding revealed that norm_* z-scoring destroys
predictive signal -- but nobody KNEW that metadata until the sleeve failed.

This script auto-extracts schema from a chimera v51 dollar parquet, classifies each
feature by prefix + heuristics, and writes BOTH:
  - data/processed/feature_catalog.parquet -- machine-readable
  - config/feature_catalog.yaml -- human-editable + version-controlled metadata catalog

IMPORTANT (2026-05-20 fix): writes to config/feature_catalog.yaml, NOT
config/feature_registry.yaml. The pipeline-spec YAML (`feature_registry.yaml`,
sources+chimera_v51 schema) is consumed by src/pipeline/feature_registry.py and
must not be overwritten by this metadata builder.

Per-feature fields:
  feature_name        str  -- exact column name
  prefix              str  -- e.g., xrel_, norm_, xd_, liq_, hbr_, bs_, mv_, etc.
  base_type           str  -- raw / normalized / xsec_rank / xsec_ratio / binary /
                              categorical / cross_asset / signed_flow / vol_metric /
                              regime_flag / target / metadata
  semantic_class      str  -- microstructure | flow | regime | vol | momentum |
                              listing | basis | reference | derived | target
  description         str  -- auto + manual override
  source_producer     str  -- script that created it (best-effort; auto from prefix)
  lookahead_safe      bool -- TRUE unless flagged otherwise
  is_z_scored         bool -- TRUE for norm_*
  is_cross_asset      bool -- TRUE for xrel_*, xd_*, xrel_*
  preserves_magnitude bool -- FALSE for norm_*; TRUE for raw, xrel_xrank, xrel_ratio
  expected_range      str  -- e.g., "[0,1]" or "[-3,+3]" or "[0, inf)" or "{0,1}"
  ks_winner_v_nonmover float -- populated by D-2 re-test (NaN if not measured)
  notes               str  -- manual annotations
  added_date          str  -- approximate (from prefix family)
"""
from __future__ import annotations

import sys, io, json
from datetime import datetime, timezone
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import polars as pl
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHIM_FP = PROJECT_ROOT / 'data' / 'processed' / 'chimera' / 'dollar'
OUT_PARQUET = PROJECT_ROOT / 'data' / 'processed' / 'feature_catalog.parquet'
OUT_YAML = PROJECT_ROOT / 'config' / 'feature_catalog.yaml'

# Prefix-based heuristic classifiers
PREFIX_RULES = {
    'xrel_':  ('xsec_rank_or_ratio', 'cross_asset',  True,  False, True,  'ADD 2026-05-18: cross-asset relative features (xrank/xpct10/xratio); fixes norm_* z-score wall'),
    'norm_':  ('normalized',          'microstructure', True, True,  False, 'Per-asset rolling z-score; preserves cross-sectional ranking but destroys absolute magnitude'),
    'xd_':    ('cross_asset',         'flow',         True,  False, True,  'Cross-asset BTC-relative or cohort metric (e.g., xd_btc_return)'),
    'liq_':   ('signed_flow',         'flow',         True,  False, True,  'Liquidation panel features (BinanceVision liquidations); raw USD'),
    'hbr_':   ('vol_metric',          'microstructure', True, False, True,  'Hawkes branching panel: self-excitation η, n_trades, intensity'),
    'bs_':    ('derived',             'basis',        True,  False, True,  'Basis-signals (perp vs spot premium)'),
    'mv_':    ('categorical',         'listing',      True,  False, True,  'Multi-venue listing indicators (n venues, is_multi_venue)'),
    'wh_':    ('signed_flow',         'flow',         True,  False, True,  'Whale flow features (wh_whale_net_usd raw signed)'),
    'etf_':   ('signed_flow',         'flow',         True,  False, True,  'ETF flow features (BTC/ETH ETF panel)'),
    'stbl_':  ('signed_flow',         'flow',         True,  False, True,  'Stablecoin supply features'),
    'bd_':    ('vol_metric',          'microstructure', True, False, True,  'Book depth panel (bd_*); 30s snapshot snapshots'),
    'lob_':   ('vol_metric',          'microstructure', True, False, True,  'LOB proxy panel; raw values'),
    'xex_':   ('vol_metric',          'flow',         True,  False, True,  'Cross-exchange spread/anomaly features'),
    'te_':    ('vol_metric',          'flow',         True,  False, True,  'Transfer entropy / cluster co-movement'),
    's3_':    ('vol_metric',          'microstructure', True, False, True,  's3 features'),
    'rv_':    ('vol_metric',          'vol',          True,  False, True,  'Realized vol / bipower variation; RAW (proven KS=0.25 baseline)'),
    'dv_':    ('vol_metric',          'vol',          True,  False, True,  'DVol features (implied/realized vol panel)'),
    'target_': ('target',             'target',       True,  False, False, 'Target columns (forward returns); NEVER use as feature'),
    'is_':    ('binary',              'reference',    True,  False, False, 'Universe membership flags (is_u10/u50/u100)'),
    'asset_': ('categorical',         'reference',    True,  False, False, 'Asset DNA / classification metadata'),
    'regime_': ('categorical',        'regime',       True,  False, True,  'Regime label encoded'),
    'cluster_': ('categorical',       'regime',       True,  False, True,  'L2 cluster_id assignment'),
    'bucket_': ('categorical',        'reference',    True,  False, True,  'DNA bucket (BLUE/STEADY/VOLATILE/DEGEN)'),
    'tier_':  ('categorical',         'reference',    True,  False, True,  'Liquidity tier'),
    'sector_': ('categorical',        'reference',    True,  False, True,  'Asset sector (DeFi/L1/L2/meme/AI/etc)'),
}

# Specific overrides for non-prefixed columns
SPECIFIC_OVERRIDES = {
    'timestamp':  ('raw', 'reference', True, False, False, 'Bar timestamp (13-digit ms)'),
    'bar_id':     ('raw', 'reference', True, False, False, 'Globally unique per-asset bar id'),
    'open':       ('raw', 'reference', True, False, True,  'OHLC open price (raw USD)'),
    'high':       ('raw', 'reference', True, False, True,  'OHLC high price (raw USD)'),
    'low':        ('raw', 'reference', True, False, True,  'OHLC low price (raw USD)'),
    'close':      ('raw', 'reference', True, False, True,  'OHLC close price (raw USD)'),
    'volume':     ('raw', 'flow',       True, False, True,  'Bar volume (raw base units)'),
    'volume_usd': ('raw', 'flow',       True, False, True,  'Bar volume (raw USD)'),
    'buy_vol':    ('raw', 'flow',       True, False, True,  'Buy-side volume (raw)'),
    'sell_vol':   ('raw', 'flow',       True, False, True,  'Sell-side volume (raw)'),
    'hurst_regime': ('vol_metric', 'regime', True, False, True, 'Hurst-regime classification (raw, not z-scored)'),
    'regime_label': ('categorical', 'regime', True, False, True, 'Regime label (categorical) encoded'),
    'date':       ('raw', 'reference', True, False, False, 'Date string YYYY-MM-DD'),
    'tier':       ('categorical', 'reference', True, False, True, 'Liquidity/DNA tier label'),
}


def classify(col: str) -> dict:
    if col in SPECIFIC_OVERRIDES:
        bt, sc, lh, zs, pm, notes = SPECIFIC_OVERRIDES[col]
        return {'prefix': '', 'base_type': bt, 'semantic_class': sc,
                'lookahead_safe': lh, 'is_z_scored': zs, 'preserves_magnitude': pm,
                'notes': notes}
    for pfx, vals in PREFIX_RULES.items():
        if col.startswith(pfx):
            bt, sc, lh, zs, pm, notes = vals
            return {'prefix': pfx, 'base_type': bt, 'semantic_class': sc,
                    'lookahead_safe': lh, 'is_z_scored': zs,
                    'preserves_magnitude': pm, 'notes': notes}
    # Unknown -- conservative
    return {'prefix': 'unknown', 'base_type': 'raw', 'semantic_class': 'derived',
            'lookahead_safe': True, 'is_z_scored': False, 'preserves_magnitude': True,
            'notes': 'Unclassified -- review manually'}


def producer_guess(prefix: str) -> str:
    """Best-effort guess of which producer script creates this feature family."""
    return {
        'xrel_': 'src/pipeline/add_xrel_features.py',
        'norm_': 'src/pipeline/make_dataset_v51.py (feature normalization step)',
        'xd_':   'src/pipeline/make_dataset_legacy.py phase 2 (cross-asset enrichment)',
        'liq_':  'src/pipeline/features/liq_features_long.py',
        'hbr_':  'src/pipeline/features/hawkes_branching_panel.py',
        'bs_':   'src/pipeline/features/basis_signals.py',
        'mv_':   'src/pipeline/features/multi_venue_listing.py',
        'wh_':   'src/pipeline/features/whale_flow_panel.py',
        'etf_':  'src/pipeline/features/etf_flows_panel.py',
        'stbl_': 'src/pipeline/features/stablecoin_supply_panel.py',
        'bd_':   'src/pipeline/features/book_depth_panel.py',
        'lob_':  'src/pipeline/features/lob_proxy_panel.py',
        'xex_':  'src/pipeline/features/cross_exchange.py',
        'te_':   'src/pipeline/features/transfer_entropy_panel.py',
        's3_':   'src/pipeline/features/s3_panel.py',
        'rv_':   'src/pipeline/features/realized_vol_panel.py',
        'dv_':   'src/pipeline/features/dvol_panel.py',
        'target_': 'src/pipeline/make_dataset_v51.py (target construction)',
        '':      'src/pipeline/dollar_bars_v51.py (base OHLCV)',
    }.get(prefix, 'unknown')


def main():
    print('=' * 72)
    print('FEATURE REGISTRY BUILD -- chimera v51 dollar')
    print('=' * 72)

    fp = sorted(CHIM_FP.glob('btcusdt_v51_chimera_*.parquet'))[-1]
    print(f'Source schema: {fp.name}')

    schema = pl.read_parquet_schema(fp)
    cols = list(schema.keys())
    print(f'Total cols: {len(cols)}')

    # Non-null %, using a quick sample (BTC has 2.6M rows; limit to 10K for speed)
    df_sample = pl.read_parquet(fp, n_rows=10000)
    nn_pct = {c: float(df_sample[c].is_not_null().mean()) for c in cols if c in df_sample.columns}

    rows = []
    for c in cols:
        meta = classify(c)
        rows.append({
            'feature_name': c,
            'prefix': meta['prefix'],
            'base_type': meta['base_type'],
            'semantic_class': meta['semantic_class'],
            'description': meta['notes'].split(';')[0][:120],
            'source_producer': producer_guess(meta['prefix']),
            'lookahead_safe': meta['lookahead_safe'],
            'is_z_scored': meta['is_z_scored'],
            'preserves_magnitude': meta['preserves_magnitude'],
            'is_cross_asset': meta['prefix'] in ('xrel_', 'xd_'),
            'expected_range': '[0,1]' if meta['base_type'] == 'xsec_rank_or_ratio' and 'xrank' in c
                              else '{0,1}' if 'pct10' in c or meta['base_type'] == 'binary'
                              else '[-3,+3]' if meta['is_z_scored']
                              else '[0, inf)' if meta['semantic_class'] in ('vol', 'flow') and not meta['is_z_scored']
                              else 'varies',
            'ks_winner_v_nonmover': float('nan'),  # populated by D-2 re-test
            'non_null_pct_sample': round(nn_pct.get(c, float('nan')) * 100, 1),
            'notes': meta['notes'],
            'added_date': '2026-05-18' if meta['prefix'] == 'xrel_' else 'pre-2026-05-18',
        })

    df_cat = pd.DataFrame(rows)
    df_cat.to_parquet(OUT_PARQUET, index=False)
    print(f'[wrote] {OUT_PARQUET} ({len(df_cat)} rows)')

    # YAML registry (human-editable)
    by_prefix = {}
    for r in rows:
        pfx = r['prefix'] or '(base)'
        if pfx not in by_prefix:
            by_prefix[pfx] = {
                'description': r['notes'],
                'source_producer': r['source_producer'],
                'is_z_scored': r['is_z_scored'],
                'preserves_magnitude': r['preserves_magnitude'],
                'lookahead_safe': r['lookahead_safe'],
                'features': [],
            }
        by_prefix[pfx]['features'].append({
            'name': r['feature_name'],
            'base_type': r['base_type'],
            'semantic_class': r['semantic_class'],
            'expected_range': r['expected_range'],
            'non_null_pct': r['non_null_pct_sample'],
        })

    registry = {
        'meta': {
            'generated': datetime.now(timezone.utc).isoformat(),
            'source_parquet': str(fp.relative_to(PROJECT_ROOT)),
            'total_features': len(rows),
            'n_prefixes': len(by_prefix),
            'invariant': 'Single source of truth for chimera v51 feature metadata. Auto-regenerate via scripts/audit/build_feature_registry.py.',
        },
        'prefix_families': by_prefix,
    }
    with open(OUT_YAML, 'w', encoding='utf-8') as f:
        yaml.safe_dump(registry, f, default_flow_style=False, sort_keys=False)
    print(f'[wrote] {OUT_YAML}')

    # Print summary
    print()
    print(f'{"prefix":<10} {"n_features":>10} {"semantic_class":<20} {"preserves_mag":>14} {"z_scored":>10}')
    print('-' * 70)
    summary = df_cat.groupby('prefix').agg({
        'feature_name': 'count',
        'semantic_class': 'first',
        'preserves_magnitude': 'first',
        'is_z_scored': 'first',
    }).reset_index().sort_values('feature_name', ascending=False)
    for _, r in summary.iterrows():
        pfx = r['prefix'] if r['prefix'] else '(base)'
        print(f'{pfx:<10} {r["feature_name"]:>10d} {r["semantic_class"]:<20} {str(r["preserves_magnitude"]):>14} {str(r["is_z_scored"]):>10}')


if __name__ == '__main__':
    main()
