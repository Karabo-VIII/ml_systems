"""
Priority 1b: Setup classifiers as INPUT FEATURES to the xsec ranker,
not as a gate. Lets the XGBRanker LEARN how to use setup probabilities
conditional on context — instead of hard-thresholding.

Baseline: K=5+5 FULL (regime+stop+meta) = +314%/Sh 3.70/DD -8% combined.
Target: +features adds 3-15% Sharpe uplift IF setup probabilities carry
information orthogonal to existing features.

If this also fails, the setup classifiers are confirmed useless for xsec context.
"""
import glob
import pickle
import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed'
MAKER_RT = 0.08

SETUP_MODELS = {}
for name in ['bounce_ml', 'fade_ml', 'swing_ml']:
    fp = ROOT / 'models' / name / f'{name}_v1.pkl'
    if fp.exists():
        SETUP_MODELS[name] = pickle.load(open(fp, 'rb'))

META_PKL = ROOT / 'models' / 'meta_labeler' / 'v8_catboost.pkl'
meta_obj = pickle.load(open(META_PKL, 'rb')) if META_PKL.exists() else None
meta_model = meta_obj['model'] if isinstance(meta_obj, dict) and 'model' in meta_obj else None
meta_features = meta_obj.get('feature_names', []) if isinstance(meta_obj, dict) else []

all_fps = sorted(glob.glob(str(DATA / '*_chimera.parquet')))
assets = [Path(f).stem.replace('usdt_v50_chimera', '').upper() for f in all_fps]
btc = pl.read_parquet(DATA / 'btcusdt_v50_chimera.parquet').to_pandas()
btc['date'] = pd.to_datetime(btc['timestamp'], unit='ms').dt.date
btc_d = btc.groupby('date').agg({'close': 'last'}).reset_index()
btc_d['btc_30d'] = btc_d['close'].pct_change(30)
btc_d['btc_ret_1d'] = btc_d['close'].pct_change()
btc_d['btc_ret_7d'] = btc_d['close'].pct_change(7)
btc_d['date'] = pd.to_datetime(btc_d['date'])

rows = []
for a in assets:
    fp = DATA / f'{a.lower()}usdt_v50_chimera.parquet'
    df = pl.read_parquet(fp).to_pandas()
    if len(df) < 1000:
        continue
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
    agg = {'close': 'last', 'open': 'first', 'high': 'max', 'low': 'min', 'volume': 'sum'}
    for c in df.columns:
        if c.startswith('norm_') or c.startswith('xd_') or c == 'hurst_regime':
            agg[c] = 'last'
    d = df.groupby('date').agg(agg).reset_index()
    d['ret_d'] = d['close'].pct_change()
    d['ret_1d'] = d['ret_d']
    d['ret_3d'] = d['close'].pct_change(3)
    d['ret_7d'] = d['close'].pct_change(7)
    d['ret_14d'] = d['close'].pct_change(14)
    d['vol_7d'] = d['ret_1d'].rolling(7).std()
    d['vol_30d'] = d['ret_1d'].rolling(30).std()
    d['hl'] = (d['high'] - d['low']) / d['open']
    d['fwd_1d'] = d['close'].shift(-1) / d['close'] - 1
    d['fwd_2d'] = d['close'].shift(-2) / d['close'] - 1
    d['fwd_3d'] = d['close'].shift(-3) / d['close'] - 1
    d['max_fwd_3d'] = np.maximum.reduce([d['fwd_1d'], d['fwd_2d'], d['fwd_3d']])
    d['min_fwd_3d'] = np.minimum.reduce([d['fwd_1d'], d['fwd_2d'], d['fwd_3d']])
    d['asset'] = a
    rows.append(d)

panel = pd.concat(rows, ignore_index=True).dropna(subset=['fwd_3d'])
panel['date'] = pd.to_datetime(panel['date'])
panel = panel.merge(btc_d[['date', 'btc_30d']], on='date', how='left')

# Score setup classifiers on EVERY asset-day
def score_setup_all(panel_df, model_obj):
    feats = model_obj['features']
    X_cols = []
    for f in feats:
        if f in panel_df.columns:
            X_cols.append(panel_df[f].fillna(0).values)
        else:
            X_cols.append(np.zeros(len(panel_df)))
    X = np.vstack(X_cols).T
    try:
        return model_obj['clf'].predict_proba(X)[:, 1]
    except Exception:
        return np.full(len(panel_df), 0.5)

print('  scoring setup classifiers on full panel...')
panel['p_bounce'] = score_setup_all(panel, SETUP_MODELS['bounce_ml']) if 'bounce_ml' in SETUP_MODELS else 0.5
panel['p_fade'] = score_setup_all(panel, SETUP_MODELS['fade_ml']) if 'fade_ml' in SETUP_MODELS else 0.5
panel['p_swing'] = score_setup_all(panel, SETUP_MODELS['swing_ml']) if 'swing_ml' in SETUP_MODELS else 0.5
print(f'  p_bounce mean: {panel["p_bounce"].mean():.3f}, std: {panel["p_bounce"].std():.3f}')
print(f'  p_fade   mean: {panel["p_fade"].mean():.3f}, std: {panel["p_fade"].std():.3f}')
print(f'  p_swing  mean: {panel["p_swing"].mean():.3f}, std: {panel["p_swing"].std():.3f}')

# Derived composite features
panel['p_long_setup'] = np.maximum(panel['p_bounce'], panel['p_swing'])
panel['p_short_setup'] = panel['p_fade']
panel['p_setup_spread'] = panel['p_long_setup'] - panel['p_short_setup']

feat_list_base = [c for c in panel.columns if c.startswith('norm_') or c.startswith('xd_')]
feat_list_base += ['ret_1d', 'ret_3d', 'ret_7d', 'ret_14d', 'vol_7d', 'vol_30d', 'hl', 'hurst_regime']
feat_list_base = [c for c in feat_list_base if c in panel.columns]
feat_list_with_setup = feat_list_base + ['p_bounce', 'p_fade', 'p_swing',
                                          'p_long_setup', 'p_short_setup', 'p_setup_spread']

panel = panel.sort_values(['date', 'asset']).reset_index(drop=True)
panel['rank_target'] = panel.groupby('date')['fwd_3d'].rank(pct=True).apply(lambda x: int(x * 31))


def _score_meta(rows_df):
    if meta_model is None:
        return np.ones(len(rows_df)) * 0.5
    X_cols = []
    for f in meta_features:
        if f in rows_df.columns:
            X_cols.append(rows_df[f].fillna(0).values)
        else:
            X_cols.append(np.zeros(len(rows_df)))
    X = np.vstack(X_cols).T
    try:
        p = meta_model.predict_proba(X)
        return p[:, 1] if p.ndim == 2 and p.shape[1] == 2 else p.flatten()
    except Exception:
        return np.ones(len(rows_df)) * 0.5


import xgboost as xgb


def train_test_window(train_end, test_start, test_end, label, feat_list):
    tr = panel[panel['date'] < train_end].copy()
    te = panel[(panel['date'] >= test_start) & (panel['date'] < test_end)].copy()
    if len(tr) < 5000 or len(te) < 500:
        return None
    tr_g = tr.groupby('date').size().values
    ranker = xgb.XGBRanker(
        objective='rank:ndcg', tree_method='hist', learning_rate=0.05,
        max_depth=6, n_estimators=500, random_state=42, eval_metric='ndcg@5',
    )
    ranker.fit(tr[feat_list].fillna(0).values, tr['rank_target'].values,
               group=tr_g, verbose=False)
    te['pred'] = ranker.predict(te[feat_list].fillna(0).values)
    # Feature importance on the FIRST window only
    if label.startswith('WF1'):
        importance = ranker.feature_importances_
        top = sorted(zip(feat_list, importance), key=lambda x: -x[1])[:10]
        print(f"  [{label}] top-10 features: {[(f, f'{v:.0f}') for f, v in top]}")
    return te


def bt(te, K_long, K_short, stop=0.10,
       regime_gate=True, meta_gate=True, meta_thresh=0.45, bear_thresh=-0.15):
    daily = []
    for d in sorted(te['date'].unique()):
        grp = te[te['date'] == d]
        if len(grp) < K_long + K_short:
            daily.append(0); continue
        btc30 = float(grp['btc_30d'].iloc[0]) if len(grp) else 0.0
        if regime_gate and (pd.isna(btc30) or btc30 < bear_thresh):
            daily.append(0); continue
        long_r = short_r = 0
        if K_long > 0:
            top = grp.sort_values('pred', ascending=False).head(K_long).copy()
            if meta_gate:
                p_win = _score_meta(top)
                top = top.assign(p_win=p_win)
                top = top[top['p_win'] >= meta_thresh]
            if len(top) > 0:
                rs = []
                for _, r in top.iterrows():
                    p = r['fwd_3d']
                    if stop and r['min_fwd_3d'] < -stop:
                        p = -stop
                    rs.append(p)
                long_r = np.mean(rs) * 100 - MAKER_RT
        if K_short > 0:
            bot = grp.sort_values('pred', ascending=True).head(K_short).copy()
            if len(bot) > 0:
                rs = []
                for _, r in bot.iterrows():
                    p = -r['fwd_3d']
                    if stop and r['max_fwd_3d'] > stop:
                        p = -stop
                    rs.append(p)
                short_r = np.mean(rs) * 100 - MAKER_RT
        n_s = (1 if K_long > 0 and long_r != 0 else 0) + (1 if K_short > 0 and short_r != 0 else 0)
        if n_s > 0:
            daily.append((long_r + short_r) / n_s / 3)
        else:
            daily.append(0)
    r = np.array(daily)
    if r.std() == 0:
        return {'total': 0, 'sharpe': 0, 'dd': 0}
    eq = np.cumprod(1 + r / 100)
    cm = np.maximum.accumulate(eq)
    return {'total': (eq[-1] - 1) * 100,
            'sharpe': r.mean() * 252 / (r.std() * np.sqrt(252)),
            'dd': ((eq - cm) / cm).min() * 100}


print('\n=== Priority 1b: Setup probs as INPUT FEATURES to xsec ranker ===')
print(f"{'window':<30} {'K=5+5 FULL (base)':>28} {'K=5+5 FULL + setup feats':>32}")

windows = [
    ('2024-10-01', '2024-10-01', '2025-03-16', 'WF1 Oct24-Mar25 (5mo)'),
    ('2025-03-16', '2025-03-16', '2025-09-01', 'WF2 Mar25-Sep25 (6mo)'),
    ('2025-09-01', '2025-09-01', '2026-04-16', 'WF3 Sep25-Apr26 (7mo)'),
    ('2024-10-01', '2024-10-01', '2026-04-16', 'COMBINED 18mo'),
]

for train_end, ts, te_end, label in windows:
    # Baseline
    te_base = train_test_window(train_end, ts, te_end, label, feat_list_base)
    # With setup features
    te_setup = train_test_window(train_end, ts, te_end, label, feat_list_with_setup)
    if te_base is None or te_setup is None:
        continue
    r_base = bt(te_base, 5, 5, stop=0.10, regime_gate=True, meta_gate=True, meta_thresh=0.45)
    r_setup = bt(te_setup, 5, 5, stop=0.10, regime_gate=True, meta_gate=True, meta_thresh=0.45)
    print(f"{label:<30} {r_base['total']:>+11.0f}% Sh{r_base['sharpe']:>+.2f} DD{r_base['dd']:>+.0f}% "
          f"{r_setup['total']:>+11.0f}% Sh{r_setup['sharpe']:>+.2f} DD{r_setup['dd']:>+.0f}%")
