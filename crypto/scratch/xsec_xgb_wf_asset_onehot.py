"""
Priority 3 — Asset-specific refinement via one-hot asset identity.

Baseline: xsec K=5+5 FULL d-neut (regime+stop+meta) = +314% / Sh 3.70 / DD -8%
on 18mo combined WF.

Hypothesis: adding 48-dim asset one-hot encoding lets XGBRanker learn
per-asset feature interactions (e.g., TAO reacts to funding differently
than BTC). This is the cheapest possible asset-FiLM implementation.

Ship criterion: combined Sharpe > 3.70 * 1.05 = 3.89.
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

META_PKL = ROOT / 'models' / 'meta_labeler' / 'v8_catboost.pkl'
meta_obj = pickle.load(open(META_PKL, 'rb')) if META_PKL.exists() else None
meta_model = meta_obj['model'] if isinstance(meta_obj, dict) and 'model' in meta_obj else None
meta_features = meta_obj.get('feature_names', []) if isinstance(meta_obj, dict) else []


all_fps = sorted(glob.glob(str(DATA / '*_chimera.parquet')))
assets = [Path(f).stem.replace('usdt_v50_chimera', '').upper() for f in all_fps]
btc = pl.read_parquet(DATA / 'btcusdt_v50_chimera.parquet').to_pandas()
btc['date'] = pd.to_datetime(btc['timestamp'], unit='ms').dt.date
btc_d = btc.groupby('date').agg({'close': 'last', 'volume': 'sum'}).reset_index()
btc_d['btc_30d'] = btc_d['close'].pct_change(30)
btc_d['btc_14d'] = btc_d['close'].pct_change(14)
btc_d['btc_7d'] = btc_d['close'].pct_change(7)
btc_d['btc_ret_1d'] = btc_d['close'].pct_change()
btc_d['btc_vol_30d'] = btc_d['btc_ret_1d'].rolling(30).std()
btc_d['date'] = pd.to_datetime(btc_d['date'])

rows = []
for a in assets:
    fp = DATA / f'{a.lower()}usdt_v50_chimera.parquet'
    df = pl.read_parquet(fp).to_pandas()
    if len(df) < 1000: continue
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
    agg = {'close': 'last', 'open': 'first', 'high': 'max', 'low': 'min', 'volume': 'sum'}
    for c in df.columns:
        if c.startswith('norm_') or c.startswith('xd_') or c == 'hurst_regime':
            agg[c] = 'last'
    d = df.groupby('date').agg(agg).reset_index()
    d['ret_1d'] = d['close'].pct_change()
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
panel = panel.merge(btc_d[['date', 'btc_30d', 'btc_14d', 'btc_7d', 'btc_vol_30d']], on='date', how='left')
panel['asset_vs_btc_7d'] = panel['ret_7d'] - panel['btc_7d']
panel['btc_regime'] = np.sign(panel['btc_7d'].fillna(0))

panel['bc_ratio'] = panel.groupby('asset')['volume'].transform(
    lambda v: (v / v.rolling(30).mean()).fillna(1.0))
panel['bc_ratio_trend_3d'] = panel.groupby('asset')['bc_ratio'].transform(
    lambda v: v - v.shift(3))
if 'norm_flow_imbalance' in panel.columns:
    panel['flow_persistence'] = panel.groupby('asset')['norm_flow_imbalance'].transform(
        lambda v: v.rolling(7).mean())
else:
    panel['flow_persistence'] = 0
if 'norm_vpin' in panel.columns:
    panel['vpin_spike'] = panel.groupby('asset')['norm_vpin'].transform(
        lambda v: (v - v.rolling(30).mean()) > 0).astype(float)
else:
    panel['vpin_spike'] = 0

DNA_MAP = {
    'BTC': 'BLUE', 'ETH': 'BLUE',
    'BNB': 'STEADY', 'SOL': 'STEADY', 'XRP': 'STEADY', 'ADA': 'STEADY',
    'AVAX': 'STEADY', 'LINK': 'STEADY', 'LTC': 'STEADY',
    'DOGE': 'VOLATILE', 'TRX': 'STEADY', 'DOT': 'STEADY', 'MATIC': 'VOLATILE',
    'NEAR': 'VOLATILE', 'FTM': 'VOLATILE', 'ATOM': 'STEADY',
}
for b in ['BLUE', 'STEADY', 'VOLATILE', 'DEGEN']:
    panel[f'bucket_{b}'] = panel['asset'].apply(
        lambda a, b=b: 1.0 if DNA_MAP.get(a, 'VOLATILE') == b else 0.0)

# 48-asset one-hot identity
unique_assets = sorted(panel['asset'].unique())
print(f'[asset one-hot] {len(unique_assets)} unique assets')
for a in unique_assets:
    panel[f'is_{a}'] = (panel['asset'] == a).astype(float)
asset_onehot_cols = [f'is_{a}' for a in unique_assets]

feat_list_base = [c for c in panel.columns if c.startswith('norm_') or c.startswith('xd_')]
feat_list_base += ['ret_1d', 'ret_3d', 'ret_7d', 'ret_14d', 'vol_7d', 'vol_30d', 'hl', 'hurst_regime']
feat_list_base = [c for c in feat_list_base if c in panel.columns]
feat_list_onehot = feat_list_base + asset_onehot_cols

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


def train_test_window(train_end, test_start, test_end, feat_list, label=''):
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
    return te


def bt(te, K_long, K_short, stop=0.10, regime_gate=True, meta_gate=True,
       meta_thresh=0.45, bear_thresh=-0.15):
    daily = []
    for d in sorted(te['date'].unique()):
        grp = te[te['date'] == d]
        if len(grp) < K_long + K_short:
            daily.append(0); continue
        btc30 = float(grp['btc_30d'].iloc[0]) if len(grp) else 0.0
        if regime_gate and (pd.isna(btc30) or btc30 < bear_thresh):
            daily.append(0); continue
        long_r = short_r = 0
        n_s = 0
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
                n_s += 1
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
                n_s += 1
        daily.append((long_r + short_r) / max(n_s, 1) / 3 if n_s > 0 else 0)
    r = np.array(daily)
    if r.std() == 0:
        return {'total': 0, 'sharpe': 0, 'dd': 0}
    eq = np.cumprod(1 + r / 100)
    cm = np.maximum.accumulate(eq)
    return {'total': (eq[-1] - 1) * 100,
            'sharpe': r.mean() * 252 / (r.std() * np.sqrt(252)),
            'dd': ((eq - cm) / cm).min() * 100}


print('\n=== Priority 3: Asset one-hot refinement on xsec K=5+5 FULL d-neut ===')
print(f"{'window':<30} {'baseline (42 feats)':>24} {'+48 asset one-hots':>24}")

windows = [
    ('2024-10-01', '2024-10-01', '2025-03-16', 'WF1 Oct24-Mar25'),
    ('2025-03-16', '2025-03-16', '2025-09-01', 'WF2 Mar25-Sep25'),
    ('2025-09-01', '2025-09-01', '2026-04-16', 'WF3 Sep25-Apr26'),
    ('2024-10-01', '2024-10-01', '2026-04-16', 'COMBINED 18mo'),
]

for train_end, ts, te_end, label in windows:
    te_base = train_test_window(train_end, ts, te_end, feat_list_base, label)
    te_oh = train_test_window(train_end, ts, te_end, feat_list_onehot, label)
    if te_base is None or te_oh is None:
        continue
    r_b = bt(te_base, 5, 5, stop=0.10, regime_gate=True, meta_gate=True, meta_thresh=0.45)
    r_oh = bt(te_oh, 5, 5, stop=0.10, regime_gate=True, meta_gate=True, meta_thresh=0.45)
    print(f"{label:<30} {r_b['total']:>+8.0f}% Sh{r_b['sharpe']:>+.2f} DD{r_b['dd']:>+.0f}% "
          f"{r_oh['total']:>+8.0f}% Sh{r_oh['sharpe']:>+.2f} DD{r_oh['dd']:>+.0f}%")
