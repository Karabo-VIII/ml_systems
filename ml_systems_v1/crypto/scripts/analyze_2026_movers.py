"""
2026 Crypto Daily Movers Analysis — capture-opportunity audit.

Census + profile + predictability study on 48-asset panel, 2026-01-01 to 2026-04-16.

Outputs:
  - logs/movers_2026_census.csv        (asset x day moves, labeled)
  - logs/movers_2026_asset_profile.csv  (per-asset mover stats)
  - logs/movers_2026_predict_features.csv (feature importance for predicting next-day >2%)
  - Console: distributions, findings, recommendations

"""
from __future__ import annotations
from pathlib import Path
import sys
import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'processed'
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'src' / 'strategy'))

THRESH_LEVELS = [0.02, 0.03, 0.05, 0.10]


def _dna_bucket(asset: str) -> str:
    try:
        from universe import DNA_BUCKET
        return DNA_BUCKET.get(asset, 'UNKNOWN')
    except Exception:
        BLUE = {'BTC', 'ETH'}
        STEADY = {'BNB', 'SOL', 'XRP', 'ADA', 'AVAX', 'LINK', 'LTC', 'DOT', 'ATOM', 'TRX'}
        VOLATILE = {'DOGE', 'MATIC', 'NEAR', 'FTM'}
        if asset in BLUE: return 'BLUE'
        if asset in STEADY: return 'STEADY'
        if asset in VOLATILE: return 'VOLATILE'
        return 'DEGEN'


def build_2026_panel():
    """Load all chimera parquets, filter to 2026, build daily panel."""
    rows = []
    for fp in sorted(DATA.glob('*_chimera.parquet')):
        asset = fp.stem.replace('usdt_v50_chimera', '').upper()
        df = pl.read_parquet(fp).to_pandas()
        if len(df) < 500:
            continue
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
        agg = {'close': 'last', 'open': 'first', 'high': 'max', 'low': 'min',
               'volume': 'sum', 'volume_usd': 'sum', 'tick_count': 'sum',
               'buy_vol': 'sum', 'sell_vol': 'sum'}
        for c in df.columns:
            if c.startswith('norm_') or c.startswith('xd_') or c == 'hurst_regime':
                agg[c] = 'last'
        d = df.groupby('date').agg({k: v for k, v in agg.items() if k in df.columns}).reset_index()
        d['ret_1d'] = d['close'].pct_change()
        d['ret_prev_1d'] = d['ret_1d'].shift(1)
        d['abs_ret_1d'] = d['ret_1d'].abs()
        d['abs_ret_prev_1d'] = d['abs_ret_1d'].shift(1)
        d['hl_range'] = (d['high'] - d['low']) / d['open']
        d['asset'] = asset
        d['bucket'] = _dna_bucket(asset)
        d['date'] = pd.to_datetime(d['date'])
        rows.append(d)
    panel = pd.concat(rows, ignore_index=True)
    panel = panel[(panel['date'] >= '2026-01-01') & (panel['date'] <= '2026-04-16')]
    panel = panel.dropna(subset=['ret_1d'])
    return panel


def _universe_movers_per_day(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-day summary: how many assets move >2%, magnitude of biggest."""
    out = []
    for d, g in panel.groupby('date'):
        n_over2 = (g['abs_ret_1d'] > 0.02).sum()
        n_over3 = (g['abs_ret_1d'] > 0.03).sum()
        n_over5 = (g['abs_ret_1d'] > 0.05).sum()
        n_over10 = (g['abs_ret_1d'] > 0.10).sum()
        max_abs = g['abs_ret_1d'].max()
        max_up = g['ret_1d'].max()
        max_dn = g['ret_1d'].min()
        mean_abs = g['abs_ret_1d'].mean()
        out.append({
            'date': d, 'n_assets': len(g),
            'n_over2pc': n_over2, 'n_over3pc': n_over3,
            'n_over5pc': n_over5, 'n_over10pc': n_over10,
            'max_abs_ret': max_abs, 'max_up_ret': max_up, 'max_dn_ret': max_dn,
            'mean_abs_ret': mean_abs,
        })
    return pd.DataFrame(out).sort_values('date')


def _asset_profile(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-asset: count and magnitude of movers."""
    out = []
    for a, g in panel.groupby('asset'):
        n = len(g)
        if n == 0:
            continue
        r = g['ret_1d'].dropna()
        ar = r.abs()
        row = {
            'asset': a, 'bucket': g['bucket'].iloc[0],
            'n_days': n, 'mean_ret': r.mean(), 'mean_abs_ret': ar.mean(),
            'std_ret': r.std(),
        }
        for t in THRESH_LEVELS:
            row[f'n_over_{int(t*100)}pc'] = (ar > t).sum()
            row[f'frac_over_{int(t*100)}pc'] = (ar > t).mean()
            up_mask = r > t
            dn_mask = r < -t
            row[f'n_up_{int(t*100)}pc'] = up_mask.sum()
            row[f'n_dn_{int(t*100)}pc'] = dn_mask.sum()
            if up_mask.sum() > 0:
                row[f'mean_up_magnitude_{int(t*100)}pc'] = r[up_mask].mean()
            if dn_mask.sum() > 0:
                row[f'mean_dn_magnitude_{int(t*100)}pc'] = r[dn_mask].mean()
        out.append(row)
    return pd.DataFrame(out)


def _persistence_test(panel: pd.DataFrame) -> dict:
    """Does a >2% move yesterday predict a >2% move today? Does direction continue or reverse?"""
    out = {}
    # Condition: abs_ret_prev > 0.02
    cond = panel['abs_ret_prev_1d'] > 0.02
    sub = panel[cond].dropna(subset=['ret_1d', 'ret_prev_1d'])
    if len(sub) > 50:
        out['n_mover_prev'] = len(sub)
        out['p_mover_today_given_mover_prev'] = (sub['abs_ret_1d'] > 0.02).mean()
        # Base rate
        out['p_mover_base'] = (panel['abs_ret_1d'] > 0.02).mean()
        # Directional continuation
        cont = np.sign(sub['ret_1d']) == np.sign(sub['ret_prev_1d'])
        out['p_direction_continues'] = cont.mean()
        # Mean return CONDITIONAL on prev mover direction
        prev_up = sub[sub['ret_prev_1d'] > 0.02]
        prev_dn = sub[sub['ret_prev_1d'] < -0.02]
        if len(prev_up) > 20:
            out['mean_ret_after_up_mover'] = prev_up['ret_1d'].mean()
        if len(prev_dn) > 20:
            out['mean_ret_after_dn_mover'] = prev_dn['ret_1d'].mean()
    return out


def _predictability(panel: pd.DataFrame):
    """Can we predict next-day >2% move from today's features?"""
    feat_cols = [c for c in panel.columns
                 if c.startswith('norm_') or c.startswith('xd_')]
    feat_cols += ['hl_range', 'ret_1d', 'abs_ret_1d', 'hurst_regime']
    feat_cols = [c for c in feat_cols if c in panel.columns]

    # Target: next-day |ret| > 2%
    df = panel.copy().sort_values(['asset', 'date'])
    df['y_mover_next'] = (df.groupby('asset')['abs_ret_1d'].shift(-1) > 0.02).astype(int)
    df['y_up_next'] = (df.groupby('asset')['ret_1d'].shift(-1) > 0.02).astype(int)
    df['y_dn_next'] = (df.groupby('asset')['ret_1d'].shift(-1) < -0.02).astype(int)
    df = df.dropna(subset=['y_mover_next'] + feat_cols)

    if len(df) < 200:
        return None

    X = df[feat_cols].fillna(0).values
    y_abs = df['y_mover_next'].values
    y_up = df['y_up_next'].values
    y_dn = df['y_dn_next'].values

    try:
        from catboost import CatBoostClassifier
    except ImportError:
        print("  CatBoost not available; skipping predictability test")
        return None

    n = len(df)
    split = int(n * 0.7)
    X_train, X_test = X[:split], X[split:]

    results = {}
    for name, y in [('mover_any', y_abs), ('mover_up', y_up), ('mover_dn', y_dn)]:
        y_train, y_test = y[:split], y[split:]
        if y_train.sum() < 10 or y_test.sum() < 5:
            continue
        try:
            clf = CatBoostClassifier(iterations=300, depth=5, learning_rate=0.05,
                                      random_seed=42, verbose=0)
            clf.fit(X_train, y_train)
            pred = clf.predict_proba(X_test)[:, 1]
            # AUC
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_test, pred) if len(np.unique(y_test)) > 1 else np.nan
            # Top-decile hit rate
            thresh = np.quantile(pred, 0.9)
            top = pred >= thresh
            hit_rate_top = y_test[top].mean() if top.sum() else 0
            base_rate = y_test.mean()
            results[name] = {
                'auc': auc, 'base_rate': base_rate,
                'top_decile_hit_rate': hit_rate_top,
                'top_decile_lift': (hit_rate_top / base_rate) if base_rate > 0 else 0,
                'n_train': split, 'n_test': n - split,
                'n_positive_test': int(y_test.sum()),
            }
            # Feature importance
            imp = clf.get_feature_importance()
            top_feats = sorted(zip(feat_cols, imp), key=lambda x: -x[1])[:10]
            results[name]['top_features'] = top_feats
        except Exception as e:
            results[name] = {'error': str(e)}
    return results


def _burst_structure(day_summary: pd.DataFrame) -> dict:
    """Are movers evenly distributed or concentrated on burst days?"""
    out = {}
    # Concentration: share of all >2% movers on top-10% of days
    n_days = len(day_summary)
    top_count = max(1, n_days // 10)
    top_days = day_summary.nlargest(top_count, 'n_over2pc')
    out['n_days'] = n_days
    out['total_over2_events'] = day_summary['n_over2pc'].sum()
    out['share_on_top10pc_days'] = (top_days['n_over2pc'].sum() /
                                     day_summary['n_over2pc'].sum())
    out['mean_over2_per_day'] = day_summary['n_over2pc'].mean()
    out['median_over2_per_day'] = day_summary['n_over2pc'].median()
    out['max_over2_day'] = day_summary['n_over2pc'].max()
    out['days_with_zero_over2'] = (day_summary['n_over2pc'] == 0).sum()
    # Autocorrelation of # movers per day
    s = day_summary['n_over2pc']
    if len(s) > 10:
        ac1 = s.autocorr(lag=1)
        ac3 = s.autocorr(lag=3)
        out['ac_lag1'] = ac1
        out['ac_lag3'] = ac3
    return out


def _cross_sectional_capture_potential(panel: pd.DataFrame) -> dict:
    """If you could always pick the top-mover each day, what would you capture?"""
    out = {}
    tops = []
    for d, g in panel.groupby('date'):
        if len(g) == 0: continue
        best_up = g['ret_1d'].max()
        best_abs = g['abs_ret_1d'].max()
        best_asset = g.loc[g['abs_ret_1d'].idxmax(), 'asset']
        tops.append({'date': d, 'best_up': best_up, 'best_abs': best_abs,
                      'best_asset': best_asset})
    t = pd.DataFrame(tops)
    out['n_days'] = len(t)
    out['mean_best_up_per_day'] = t['best_up'].mean()
    out['median_best_up_per_day'] = t['best_up'].median()
    out['mean_best_abs_per_day'] = t['best_abs'].mean()
    out['n_days_best_over_2pc'] = (t['best_abs'] > 0.02).sum()
    out['n_days_best_over_5pc'] = (t['best_abs'] > 0.05).sum()
    out['n_days_best_over_10pc'] = (t['best_abs'] > 0.10).sum()
    # Theoretical perfect-pick compound
    up_rets = t['best_up'].clip(-0.1, 0.5)  # cap to exclude tail outliers
    if len(up_rets) > 0:
        out['perfect_pick_compound'] = float(np.prod(1 + up_rets) - 1)
    return out


def main():
    print("[Analysis] Building 2026 panel...")
    panel = build_2026_panel()
    print(f"  panel: {len(panel):,} asset-day rows, "
          f"{panel['asset'].nunique()} assets, "
          f"{panel['date'].nunique()} days, "
          f"range {panel['date'].min()} -> {panel['date'].max()}")

    # Day-level summary
    day_summary = _universe_movers_per_day(panel)
    print(f"\n[Day-level] Distribution of per-day mover counts (out of "
          f"{panel['asset'].nunique()} assets):")
    for col in ['n_over2pc', 'n_over3pc', 'n_over5pc', 'n_over10pc']:
        print(f"  {col}: mean {day_summary[col].mean():.2f}, "
              f"median {day_summary[col].median():.0f}, "
              f"max {day_summary[col].max()}")

    # Asset profile
    asset_profile = _asset_profile(panel)
    asset_profile = asset_profile.sort_values('frac_over_2pc', ascending=False)
    print(f"\n[Asset-level] Top 15 by fraction of days with >2% move:")
    for _, r in asset_profile.head(15).iterrows():
        print(f"  {r['asset']:<8} ({r['bucket']:<9}) "
              f"n={r['n_days']:>4} | "
              f">2%: {r['frac_over_2pc']:5.1%} ({r['n_over_2pc']:>3} days) | "
              f">5%: {r['frac_over_5pc']:5.1%} | "
              f">10%: {r['frac_over_10pc']:5.1%} | "
              f"vol {r['std_ret']*100:4.1f}%")

    # Bucket summary
    print(f"\n[Bucket-level] Move frequency by DNA bucket:")
    for b, g in asset_profile.groupby('bucket'):
        total = g['n_days'].sum()
        over2 = g['n_over_2pc'].sum()
        over5 = g['n_over_5pc'].sum()
        print(f"  {b:<9}: n={len(g):>2} assets, "
              f"{over2/total*100:5.1f}% of asset-days >2%, "
              f"{over5/total*100:4.1f}% >5%")

    # Persistence
    persist = _persistence_test(panel)
    print(f"\n[Persistence] >2% mover yesterday -> today:")
    if persist:
        print(f"  P(mover today | mover yesterday) = {persist['p_mover_today_given_mover_prev']:.2%}")
        print(f"  P(mover base rate)                = {persist['p_mover_base']:.2%}")
        lift = persist['p_mover_today_given_mover_prev'] / persist['p_mover_base']
        print(f"  Lift: {lift:.2f}x")
        print(f"  P(direction continues)             = {persist['p_direction_continues']:.2%}")
        if 'mean_ret_after_up_mover' in persist:
            print(f"  Mean return after UP mover         = {persist['mean_ret_after_up_mover']*100:+.2f}%")
        if 'mean_ret_after_dn_mover' in persist:
            print(f"  Mean return after DOWN mover       = {persist['mean_ret_after_dn_mover']*100:+.2f}%")

    # Burst structure
    burst = _burst_structure(day_summary)
    print(f"\n[Burst structure]:")
    print(f"  Total >2% events: {burst['total_over2_events']}")
    print(f"  Share on top-10% of days: {burst['share_on_top10pc_days']:.1%}")
    print(f"  Median movers per day: {burst['median_over2_per_day']:.0f}")
    print(f"  Max on one day: {burst['max_over2_day']}")
    print(f"  Days with ZERO >2% movers: {burst['days_with_zero_over2']}/{burst['n_days']}")
    if 'ac_lag1' in burst:
        print(f"  Autocorr lag-1: {burst['ac_lag1']:+.3f}")
        print(f"  Autocorr lag-3: {burst['ac_lag3']:+.3f}")

    # Cross-sectional capture potential
    xs = _cross_sectional_capture_potential(panel)
    print(f"\n[Cross-sectional ceiling] If you always picked the DAILY TOP UP-MOVER:")
    print(f"  Mean best-up per day: {xs['mean_best_up_per_day']*100:+.2f}%")
    print(f"  Median best-up per day: {xs['median_best_up_per_day']*100:+.2f}%")
    print(f"  Days where best move >5%: {xs['n_days_best_over_5pc']}/{xs['n_days']} "
          f"({xs['n_days_best_over_5pc']/xs['n_days']*100:.1f}%)")
    print(f"  Days where best move >10%: {xs['n_days_best_over_10pc']}/{xs['n_days']} "
          f"({xs['n_days_best_over_10pc']/xs['n_days']*100:.1f}%)")
    print(f"  Perfect-pick compound (capped): {xs['perfect_pick_compound']*100:+.1f}%")

    # Predictability
    print(f"\n[Predictability] CatBoost: predict next-day |ret|>2% from today's features")
    pred = _predictability(panel)
    if pred:
        for name, r in pred.items():
            if 'error' in r:
                print(f"  {name}: ERROR {r['error']}")
                continue
            print(f"  TARGET={name}: AUC={r['auc']:.3f}, base_rate={r['base_rate']:.2%}, "
                  f"top-decile hit={r['top_decile_hit_rate']:.2%} "
                  f"({r['top_decile_lift']:.2f}x lift)")
            print(f"    top features: "
                  f"{', '.join([f'{f}({v:.1f})' for f, v in r['top_features'][:5]])}")

    # Directionality
    up2 = (panel['ret_1d'] > 0.02).sum()
    dn2 = (panel['ret_1d'] < -0.02).sum()
    print(f"\n[Directionality] Up vs Down >2% moves:")
    print(f"  UP >2%:   {up2} ({up2/(up2+dn2)*100:.1f}%)")
    print(f"  DOWN >2%: {dn2} ({dn2/(up2+dn2)*100:.1f}%)")
    up5 = (panel['ret_1d'] > 0.05).sum()
    dn5 = (panel['ret_1d'] < -0.05).sum()
    print(f"  UP >5%:   {up5} ({up5/(up5+dn5)*100:.1f}%)")
    print(f"  DOWN >5%: {dn5} ({dn5/(up5+dn5)*100:.1f}%)")

    # Save artifacts
    day_summary.to_csv(ROOT / 'logs' / 'movers_2026_day_summary.csv', index=False)
    asset_profile.to_csv(ROOT / 'logs' / 'movers_2026_asset_profile.csv', index=False)
    print(f"\n[Files] Saved:")
    print(f"  logs/movers_2026_day_summary.csv")
    print(f"  logs/movers_2026_asset_profile.csv")


if __name__ == "__main__":
    main()
