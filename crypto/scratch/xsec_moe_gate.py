"""
Priority 2 — MoE Strategy Gate (v1).

Train a gating model that picks between 4 validated xsec-class champions
per-day, based on market-state features (BTC vol regime, dispersion,
funding, breadth).

Champions (all direct-backtest; unaffected by MtM bug):
  A. xsec K=5+5 FULL d-neut (regime+stop+meta)    Sh 3.70 / +314% / DD -8%
  B. XGB K=5+5 d-neut WF                           Sh 3.36 / +227%
  C. XGB K=3+3 d-neut WF                           Sh 2.89 / +269%
  D. CatBoost K=1 stop (no macro) WF               Sh 1.69 / +778% / DD -78%

For each day in test window:
  1. Compute REALIZED daily return of each champion
  2. Training label: argmax(daily_return) across champions
  3. Train CatBoostClassifier on market-state features → label
  4. In backtest: gate picks champion each day based on predicted class
  5. Measure combined portfolio return vs static-blend baseline

Expected per LLMoE literature: +20-30% Sharpe lift over static allocation.
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

# Load meta-labeler (for champion A)
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
    if len(df) < 1000:
        continue
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
    d['vol_14d'] = d['ret_1d'].rolling(14).std()
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
panel = panel.merge(btc_d[['date', 'btc_30d', 'btc_14d', 'btc_7d', 'btc_ret_1d',
                            'btc_vol_30d', 'btc_vol_14d']], on='date', how='left')
panel['asset_vs_btc_7d'] = panel['ret_7d'] - panel['btc_7d']
panel['btc_regime'] = np.sign(panel['btc_7d'].fillna(0))

# Meta-labeler extra features (needed for champion A)
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
    'BNB': 'STEADY', 'SOL': 'STEADY', 'XRP': 'STEADY',
    'ADA': 'STEADY', 'AVAX': 'STEADY', 'LINK': 'STEADY', 'LTC': 'STEADY',
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
from catboost import CatBoostClassifier


def train_ranker(train_end, xsec_type='xgb_ndcg'):
    tr = panel[panel['date'] < train_end].copy()
    tr_g = tr.groupby('date').size().values
    if xsec_type == 'xgb_ndcg':
        r = xgb.XGBRanker(objective='rank:ndcg', tree_method='hist',
                          learning_rate=0.05, max_depth=6, n_estimators=500,
                          random_state=42, eval_metric='ndcg@5')
        r.fit(tr[feat_list].fillna(0).values, tr['rank_target'].values,
              group=tr_g, verbose=False)
    elif xsec_type == 'cat_reg':
        from catboost import CatBoostRegressor
        r = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.03,
                               l2_leaf_reg=3.0, random_seed=42, verbose=0)
        r.fit(tr[feat_list].fillna(0).values, tr['fwd_3d'].values, verbose=0)
    return r


def compute_daily_returns_for_champion(te, K_long, K_short, stop=0.10,
                                         regime_gate=False, meta_gate=False,
                                         meta_thresh=0.45, bear_thresh=-0.15):
    """Compute daily portfolio return for a given champion config."""
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


print('=== Priority 2: MoE Strategy Gate (v1) ===')
print('4 champions. Per-day realized returns + gating classifier.\n')

# WF windows: use WF1+WF2 for MoE training, WF3 for MoE test
wf_splits = [
    ('2024-10-01', '2024-10-01', '2026-04-16', 'COMBINED 18mo'),
]

for train_end, ts, te_end, label in wf_splits:
    print(f'[{label}] training 2 ranker types...')
    ranker_xgb = train_ranker(train_end, xsec_type='xgb_ndcg')
    ranker_cat = train_ranker(train_end, xsec_type='cat_reg')
    te = panel[(panel['date'] >= ts) & (panel['date'] < te_end)].copy()
    te['pred_xgb'] = ranker_xgb.predict(te[feat_list].fillna(0).values)
    te['pred_cat'] = ranker_cat.predict(te[feat_list].fillna(0).values)

    # Compute per-champion daily returns
    print(f'  computing per-champion daily returns...')
    # Champion A: xsec K=5+5 FULL d-neut (regime+stop+meta)
    te['pred'] = te['pred_xgb']
    daily_A = compute_daily_returns_for_champion(
        te, 5, 5, stop=0.10, regime_gate=True, meta_gate=True, meta_thresh=0.45)
    # Champion B: XGB K=5+5 d-neut plain
    daily_B = compute_daily_returns_for_champion(
        te, 5, 5, stop=0.10, regime_gate=False, meta_gate=False)
    # Champion C: XGB K=3+3 d-neut plain
    daily_C = compute_daily_returns_for_champion(
        te, 3, 3, stop=0.10, regime_gate=False, meta_gate=False)
    # Champion D: CatBoost K=1 stop (no macro)
    te['pred'] = te['pred_cat']
    daily_D = compute_daily_returns_for_champion(
        te, 1, 0, stop=0.10, regime_gate=False, meta_gate=False)

    champs = {'A_xsec_full': daily_A, 'B_xgb_55': daily_B,
              'C_xgb_33': daily_C, 'D_cat_1stop': daily_D}
    daily_df = pd.DataFrame(champs).fillna(0)
    print(f'  per-champion cumulative returns over {len(daily_df)} days:')
    for name, s in champs.items():
        eq = np.cumprod(1 + s.values / 100)
        tot = (eq[-1] - 1) * 100
        sh = s.values.mean() * 252 / (s.values.std() * np.sqrt(252)) if s.values.std() > 0 else 0
        print(f'    {name}: total {tot:+.1f}%, Sharpe {sh:+.2f}')

    # Argmax label per day (which champion was best)
    winner = daily_df.idxmax(axis=1)
    winner_counts = winner.value_counts()
    print(f'  winner distribution:\n{winner_counts.to_string()}')

    # Market-state features for MoE classifier (1 row per day)
    # Use BTC regime + cross-asset dispersion proxies + previous-day performance
    daily_state = pd.DataFrame(index=daily_df.index)
    daily_state['btc_30d'] = te.groupby('date')['btc_30d'].first().reindex(daily_df.index)
    daily_state['btc_14d'] = te.groupby('date')['btc_14d'].first().reindex(daily_df.index)
    daily_state['btc_7d'] = te.groupby('date')['btc_7d'].first().reindex(daily_df.index)
    daily_state['btc_vol_30d'] = te.groupby('date')['btc_vol_30d'].first().reindex(daily_df.index)
    daily_state['btc_vol_14d'] = te.groupby('date')['btc_vol_14d'].first().reindex(daily_df.index)
    daily_state['cross_vol_mean'] = te.groupby('date')['xd_cross_vol_mean'].mean().reindex(daily_df.index)
    daily_state['cross_return_mean'] = te.groupby('date')['xd_cross_return_mean'].mean().reindex(daily_df.index)
    # Prior day per-champion return (1-bar lag) as regime indicator
    for c in champs:
        daily_state[f'{c}_lag1'] = champs[c].shift(1).reindex(daily_df.index)
    daily_state = daily_state.fillna(0)

    # Split: first 60% for MoE training, rest for test
    n = len(daily_df)
    split = int(n * 0.60)
    X_train = daily_state.iloc[:split].values
    y_train = winner.iloc[:split]
    X_test = daily_state.iloc[split:].values
    y_test = winner.iloc[split:]

    # Only train if we have at least 30 samples of at least 2 classes
    train_classes = y_train.unique()
    print(f'  MoE train/test split: {split}/{n-split} days, train classes: {len(train_classes)}')

    if len(train_classes) < 2 or split < 30:
        print('  Insufficient diversity for MoE training; skipping')
        continue

    gate = CatBoostClassifier(iterations=200, depth=4, learning_rate=0.05,
                               random_seed=42, verbose=0)
    gate.fit(X_train, y_train.values)
    pred_winner = gate.predict(X_test).flatten()

    # Realize MoE returns: each day in test, use the PREDICTED champion's return
    moe_returns = []
    for i, d in enumerate(daily_df.iloc[split:].index):
        pred_c = pred_winner[i]
        if isinstance(pred_c, (np.ndarray, list)):
            pred_c = pred_c[0] if len(pred_c) else 'A_xsec_full'
        moe_returns.append(daily_df.loc[d, pred_c])
    moe_returns = np.array(moe_returns)

    # Baseline: always A_xsec_full (current champion)
    baseline_A = daily_df.iloc[split:]['A_xsec_full'].values
    # Baseline: equal-weighted blend of all 4
    baseline_eq = daily_df.iloc[split:].mean(axis=1).values

    print(f'\n  === MoE RESULT (test set: {len(moe_returns)} days = {daily_df.index[split]:%Y-%m-%d} to {daily_df.index[-1]:%Y-%m-%d}) ===')
    for name, r in [('A_only (baseline)', baseline_A),
                     ('eq_blend', baseline_eq),
                     ('MoE_gated', moe_returns)]:
        if len(r) == 0 or r.std() == 0:
            print(f'    {name}: insufficient data')
            continue
        eq = np.cumprod(1 + r / 100)
        tot = (eq[-1] - 1) * 100
        sh = r.mean() * 252 / (r.std() * np.sqrt(252))
        cm = np.maximum.accumulate(eq)
        dd = ((eq - cm) / cm).min() * 100
        print(f'    {name}: total {tot:+.1f}%, Sharpe {sh:+.2f}, DD {dd:+.1f}%')
