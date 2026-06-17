"""
Priority 2 — MoE Strategy Gate v2.

v1 failure: single-winner argmax favored K=1 high-vol champion (D), blowing
up drawdown. Sharpe 2.73 vs baseline 4.73 (42% worse) despite higher return.

v2 fix:
  1. Exclude K=1 (D) — keep only 3 dollar-neutral champions (A, B, C).
  2. Use SOFTMAX-blended daily return (not hard argmax).
  3. Alternative: use rolling-Sharpe argmax as label (risk-adjusted winner).

Baseline (A_only, xsec K=5+5 FULL): test set +148.7% Sh 4.73 DD -6.3%.
Ship criterion: Sharpe > 4.73 * 1.05 = 4.97.
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
btc_d['btc_vol_14d'] = btc_d['btc_ret_1d'].rolling(14).std()
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
panel = panel.merge(btc_d[['date', 'btc_30d', 'btc_14d', 'btc_7d',
                            'btc_vol_30d', 'btc_vol_14d']], on='date', how='left')
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

feat_list = [c for c in panel.columns if c.startswith('norm_') or c.startswith('xd_')]
feat_list += ['ret_1d', 'ret_3d', 'ret_7d', 'ret_14d', 'vol_7d', 'vol_30d', 'hl', 'hurst_regime']
feat_list = [c for c in feat_list if c in panel.columns]

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
from catboost import CatBoostClassifier, CatBoostRegressor


def train_ranker(train_end, xsec_type='xgb_ndcg'):
    tr = panel[panel['date'] < train_end].copy()
    tr_g = tr.groupby('date').size().values
    if xsec_type == 'xgb_ndcg':
        r = xgb.XGBRanker(objective='rank:ndcg', tree_method='hist',
                          learning_rate=0.05, max_depth=6, n_estimators=500,
                          random_state=42, eval_metric='ndcg@5')
        r.fit(tr[feat_list].fillna(0).values, tr['rank_target'].values,
              group=tr_g, verbose=False)
    return r


def compute_daily_returns_for_champion(te, K_long, K_short, stop=0.10,
                                         regime_gate=False, meta_gate=False,
                                         meta_thresh=0.45, bear_thresh=-0.15):
    daily = {}
    for d in sorted(te['date'].unique()):
        grp = te[te['date'] == d]
        if len(grp) < K_long + K_short:
            daily[d] = 0.0; continue
        btc30 = float(grp['btc_30d'].iloc[0]) if len(grp) else 0.0
        if regime_gate and (pd.isna(btc30) or btc30 < bear_thresh):
            daily[d] = 0.0; continue
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
        daily[d] = (long_r + short_r) / max(n_s, 1) / 3 if n_s > 0 else 0.0
    return pd.Series(daily).sort_index()


def compute_metrics(r):
    if len(r) == 0 or r.std() == 0:
        return {'total': 0, 'sharpe': 0, 'dd': 0}
    eq = np.cumprod(1 + r / 100)
    cm = np.maximum.accumulate(eq)
    return {'total': (eq[-1] - 1) * 100,
            'sharpe': r.mean() * 252 / (r.std() * np.sqrt(252)),
            'dd': ((eq - cm) / cm).min() * 100}


print('=== Priority 2 v2: MoE gate between 3 d-neut champions (no K=1) ===\n')

train_end = '2024-10-01'
ts, te_end = '2024-10-01', '2026-04-16'
print(f'[COMBINED 18mo] training XGB ranker...')
ranker = train_ranker(train_end, xsec_type='xgb_ndcg')
te = panel[(panel['date'] >= ts) & (panel['date'] < te_end)].copy()
te['pred'] = ranker.predict(te[feat_list].fillna(0).values)

print('  computing per-champion daily returns (3 d-neut champions)...')
daily_A = compute_daily_returns_for_champion(te, 5, 5, stop=0.10,
                                               regime_gate=True, meta_gate=True,
                                               meta_thresh=0.45)
daily_B = compute_daily_returns_for_champion(te, 5, 5, stop=0.10,
                                               regime_gate=False, meta_gate=False)
daily_C = compute_daily_returns_for_champion(te, 3, 3, stop=0.10,
                                               regime_gate=False, meta_gate=False)

champs = {'A_xsec_full': daily_A, 'B_xgb_55': daily_B, 'C_xgb_33': daily_C}
daily_df = pd.DataFrame(champs).fillna(0)

print('  per-champion cumulative over 560 days:')
for name, s in champs.items():
    m = compute_metrics(s.values)
    print(f'    {name}: {m["total"]:+.1f}% Sh{m["sharpe"]:+.2f} DD{m["dd"]:+.1f}%')

# Market-state features
daily_state = pd.DataFrame(index=daily_df.index)
daily_state['btc_30d'] = te.groupby('date')['btc_30d'].first().reindex(daily_df.index)
daily_state['btc_14d'] = te.groupby('date')['btc_14d'].first().reindex(daily_df.index)
daily_state['btc_7d'] = te.groupby('date')['btc_7d'].first().reindex(daily_df.index)
daily_state['btc_vol_30d'] = te.groupby('date')['btc_vol_30d'].first().reindex(daily_df.index)
daily_state['btc_vol_14d'] = te.groupby('date')['btc_vol_14d'].first().reindex(daily_df.index)
daily_state['cross_vol_mean'] = te.groupby('date')['xd_cross_vol_mean'].mean().reindex(daily_df.index)
daily_state['cross_return_mean'] = te.groupby('date')['xd_cross_return_mean'].mean().reindex(daily_df.index)
for c in champs:
    daily_state[f'{c}_lag1'] = champs[c].shift(1).reindex(daily_df.index)
    daily_state[f'{c}_lag3_mean'] = champs[c].rolling(3).mean().shift(1).reindex(daily_df.index)
daily_state = daily_state.fillna(0)

n = len(daily_df)
split = int(n * 0.60)

# === v2 LABELING STRATEGIES ===

# Strategy (i): argmax daily return (same as v1 but with 3 champs not 4)
winner_argmax = daily_df.idxmax(axis=1)
print(f'\n  v2i argmax label distribution:')
print(winner_argmax.value_counts().to_string())

# Strategy (ii): rolling-Sharpe argmax (risk-adjusted winner per rolling 14d)
roll = daily_df.rolling(14).apply(lambda r: r.mean() / (r.std() + 1e-10) if r.std() else 0, raw=True)
winner_rollsharpe = roll.shift(1).idxmax(axis=1)
print(f'\n  v2ii rolling-Sharpe label distribution:')
print(winner_rollsharpe.value_counts().to_string())


# === RUN EACH STRATEGY ===
print(f'\n  === v2 results (test 224 days, split {daily_df.index[split]:%Y-%m-%d} to {daily_df.index[-1]:%Y-%m-%d}) ===')

# Baseline: always A
r_base = daily_df.iloc[split:]['A_xsec_full'].values
m_base = compute_metrics(r_base)
print(f'\n  A_only (baseline, current champion):')
print(f'    total {m_base["total"]:+.1f}%, Sharpe {m_base["sharpe"]:+.2f}, DD {m_base["dd"]:+.1f}%')

# Equal-blend of 3 d-neut
r_eq = daily_df.iloc[split:].mean(axis=1).values
m_eq = compute_metrics(r_eq)
print(f'\n  Equal-blend (A+B+C / 3):')
print(f'    total {m_eq["total"]:+.1f}%, Sharpe {m_eq["sharpe"]:+.2f}, DD {m_eq["dd"]:+.1f}%')

# Train v2i MoE (argmax)
y_train_argmax = winner_argmax.iloc[:split].dropna()
X_train = daily_state.iloc[:split].loc[y_train_argmax.index].values
X_test = daily_state.iloc[split:].values
if len(y_train_argmax.unique()) >= 2 and len(y_train_argmax) > 30:
    gate_i = CatBoostClassifier(iterations=200, depth=4, learning_rate=0.05,
                                 random_seed=42, verbose=0)
    gate_i.fit(X_train, y_train_argmax.values)
    pred_i = gate_i.predict(X_test).flatten()
    moe_i = np.array([daily_df.iloc[split + idx][pc if not isinstance(pc, (np.ndarray, list)) else pc[0]]
                       for idx, pc in enumerate(pred_i)])
    m_i = compute_metrics(moe_i)
    print(f'\n  v2i MoE (argmax daily):')
    print(f'    total {m_i["total"]:+.1f}%, Sharpe {m_i["sharpe"]:+.2f}, DD {m_i["dd"]:+.1f}%')

# Train v2ii MoE (rolling-Sharpe)
y_train_roll = winner_rollsharpe.iloc[:split].dropna()
if len(y_train_roll.unique()) >= 2 and len(y_train_roll) > 30:
    X_train_roll = daily_state.iloc[:split].loc[y_train_roll.index].values
    gate_ii = CatBoostClassifier(iterations=200, depth=4, learning_rate=0.05,
                                  random_seed=42, verbose=0)
    gate_ii.fit(X_train_roll, y_train_roll.values)
    pred_ii = gate_ii.predict(X_test).flatten()
    moe_ii = np.array([daily_df.iloc[split + idx][pc if not isinstance(pc, (np.ndarray, list)) else pc[0]]
                        for idx, pc in enumerate(pred_ii)])
    m_ii = compute_metrics(moe_ii)
    print(f'\n  v2ii MoE (rolling-Sharpe argmax):')
    print(f'    total {m_ii["total"]:+.1f}%, Sharpe {m_ii["sharpe"]:+.2f}, DD {m_ii["dd"]:+.1f}%')

# === SOFTMAX SOFT-BLEND ===
# For each day, predict CLASS PROBABILITIES -> soft weights -> blended return
if len(y_train_argmax) > 30:
    prob_i = gate_i.predict_proba(X_test)
    # Map class names to indices
    class_names = gate_i.classes_
    # Weighted blend
    moe_soft = np.zeros(len(X_test))
    for di, d in enumerate(daily_df.iloc[split:].index):
        for ci, cn in enumerate(class_names):
            moe_soft[di] += prob_i[di][ci] * daily_df.loc[d, cn]
    m_s = compute_metrics(moe_soft)
    print(f'\n  v2iii MoE (soft-blend from argmax probs):')
    print(f'    total {m_s["total"]:+.1f}%, Sharpe {m_s["sharpe"]:+.2f}, DD {m_s["dd"]:+.1f}%')
