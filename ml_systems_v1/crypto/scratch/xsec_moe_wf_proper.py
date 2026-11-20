"""
Proper 3-window walk-forward validation of MoE v1 (4 champions, argmax).

Prior test: MoE v1 achieved +382.7% / Sh 2.73 / DD -37.9% on a SINGLE
10-month test subset (2025-09-02 to 2026-04-16). That's WF3-biased.

This script: train the gating classifier on each non-overlap WF window
separately and test on the next window, mimicking the xsec_xgb_walkforward.py
protocol.

If MoE v1 is WF-robust under CAGR/DD criteria, it becomes champion under
the Wealth axis. If it degrades in WF1/WF2 (as argmax labels can overfit),
we concede.
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
panel = panel.merge(btc_d[['date', 'btc_30d', 'btc_14d', 'btc_7d', 'btc_vol_30d', 'btc_vol_14d']], on='date', how='left')
panel['asset_vs_btc_7d'] = panel['ret_7d'] - panel['btc_7d']
panel['btc_regime'] = np.sign(panel['btc_7d'].fillna(0))
panel['bc_ratio'] = panel.groupby('asset')['volume'].transform(lambda v: (v / v.rolling(30).mean()).fillna(1.0))
panel['bc_ratio_trend_3d'] = panel.groupby('asset')['bc_ratio'].transform(lambda v: v - v.shift(3))
if 'norm_flow_imbalance' in panel.columns:
    panel['flow_persistence'] = panel.groupby('asset')['norm_flow_imbalance'].transform(lambda v: v.rolling(7).mean())
else:
    panel['flow_persistence'] = 0
if 'norm_vpin' in panel.columns:
    panel['vpin_spike'] = panel.groupby('asset')['norm_vpin'].transform(lambda v: (v - v.rolling(30).mean()) > 0).astype(float)
else:
    panel['vpin_spike'] = 0

DNA_MAP = {'BTC': 'BLUE', 'ETH': 'BLUE', 'BNB': 'STEADY', 'SOL': 'STEADY',
           'XRP': 'STEADY', 'ADA': 'STEADY', 'AVAX': 'STEADY', 'LINK': 'STEADY',
           'LTC': 'STEADY', 'DOGE': 'VOLATILE', 'TRX': 'STEADY', 'DOT': 'STEADY',
           'MATIC': 'VOLATILE', 'NEAR': 'VOLATILE', 'FTM': 'VOLATILE', 'ATOM': 'STEADY'}
for b in ['BLUE', 'STEADY', 'VOLATILE', 'DEGEN']:
    panel[f'bucket_{b}'] = panel['asset'].apply(lambda a, b=b: 1.0 if DNA_MAP.get(a, 'VOLATILE') == b else 0.0)

feat_list = [c for c in panel.columns if c.startswith('norm_') or c.startswith('xd_')]
feat_list += ['ret_1d', 'ret_3d', 'ret_7d', 'ret_14d', 'vol_7d', 'vol_30d', 'hl', 'hurst_regime']
feat_list = [c for c in feat_list if c in panel.columns]
panel = panel.sort_values(['date', 'asset']).reset_index(drop=True)
panel['rank_target'] = (panel.groupby('date')['fwd_3d'].rank(pct=True) * 31).fillna(0).astype(int)


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


def train_rankers(train_end):
    tr = panel[panel['date'] < train_end].copy()
    tr_g = tr.groupby('date').size().values
    xgb_r = xgb.XGBRanker(objective='rank:ndcg', tree_method='hist', learning_rate=0.05,
                          max_depth=6, n_estimators=500, random_state=42, eval_metric='ndcg@5')
    xgb_r.fit(tr[feat_list].fillna(0).values, tr['rank_target'].values, group=tr_g, verbose=False)
    cat_r = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.03, l2_leaf_reg=3.0,
                               random_seed=42, verbose=0)
    cat_r.fit(tr[feat_list].fillna(0).values, tr['fwd_3d'].values, verbose=0)
    return xgb_r, cat_r


def compute_daily_returns(te, K_long, K_short, stop=0.10,
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


def metrics(r):
    if len(r) < 2 or r.std() == 0:
        return {'total': 0.0, 'sharpe': 0.0, 'dd': 0.0, 'cagr': 0.0,
                'calmar': 0.0, 'sortino': 0.0, 'win_rate': 0.0, 'n_days': len(r)}
    eq = np.cumprod(1 + r / 100)
    tot = (eq[-1] - 1) * 100
    sh = r.mean() * 252 / (r.std() * np.sqrt(252))
    cm = np.maximum.accumulate(eq)
    dd_series = (eq - cm) / cm
    max_dd = dd_series.min() * 100
    n_days = len(r)
    days_per_year = 365
    cagr = ((eq[-1]) ** (days_per_year / n_days) - 1) * 100
    calmar = abs(cagr / max_dd) if max_dd != 0 else 0
    neg_r = r[r < 0]
    sortino = r.mean() * 252 / (neg_r.std() * np.sqrt(252)) if len(neg_r) > 0 and neg_r.std() > 0 else sh * 2
    win_rate = (r > 0).mean() * 100
    return {'total': float(tot), 'sharpe': float(sh), 'dd': float(max_dd),
            'cagr': float(cagr), 'calmar': float(calmar), 'sortino': float(sortino),
            'win_rate': float(win_rate), 'n_days': int(n_days)}


# Sequential WF: train through window_end-1, test on window
windows = [
    ('2024-10-01', '2025-03-16', 'WF1 Oct24-Mar25 (5mo)'),
    ('2025-03-16', '2025-09-01', 'WF2 Mar25-Sep25 (6mo)'),
    ('2025-09-01', '2026-04-16', 'WF3 Sep25-Apr26 (7mo)'),
]

print('=== MoE v1 proper 3-window WF ===')
print('Train gating classifier on data through window_start; test on window_start..window_end\n')

all_daily_moe = []
all_daily_A = []
all_daily_B = []
all_daily_C = []
all_daily_D = []

for train_end, test_end, label in windows:
    print(f'[{label}] training rankers (data < {train_end})...')
    xgb_r, cat_r = train_rankers(train_end)
    te = panel[(panel['date'] >= train_end) & (panel['date'] < test_end)].copy()
    te['pred_xgb'] = xgb_r.predict(te[feat_list].fillna(0).values)
    te['pred_cat'] = cat_r.predict(te[feat_list].fillna(0).values)

    te['pred'] = te['pred_xgb']
    daily_A = compute_daily_returns(te, 5, 5, stop=0.10, regime_gate=True, meta_gate=True, meta_thresh=0.45)
    daily_B = compute_daily_returns(te, 5, 5, stop=0.10, regime_gate=False, meta_gate=False)
    daily_C = compute_daily_returns(te, 3, 3, stop=0.10, regime_gate=False, meta_gate=False)
    te['pred'] = te['pred_cat']
    daily_D = compute_daily_returns(te, 1, 0, stop=0.10, regime_gate=False, meta_gate=False)

    daily_df = pd.DataFrame({'A_xsec_full': daily_A, 'B_xgb_55': daily_B,
                              'C_xgb_33': daily_C, 'D_cat_1stop': daily_D}).fillna(0)

    # Train gate on FIRST HALF of this window (in-window split)
    n = len(daily_df)
    if n < 40:
        continue
    gate_split = int(n * 0.5)
    winner = daily_df.iloc[:gate_split].idxmax(axis=1)
    if winner.nunique() < 2:
        continue
    # State features for gate
    daily_state = pd.DataFrame(index=daily_df.index)
    daily_state['btc_30d'] = te.groupby('date')['btc_30d'].first().reindex(daily_df.index)
    daily_state['btc_14d'] = te.groupby('date')['btc_14d'].first().reindex(daily_df.index)
    daily_state['btc_7d'] = te.groupby('date')['btc_7d'].first().reindex(daily_df.index)
    daily_state['btc_vol_30d'] = te.groupby('date')['btc_vol_30d'].first().reindex(daily_df.index)
    daily_state['btc_vol_14d'] = te.groupby('date')['btc_vol_14d'].first().reindex(daily_df.index)
    daily_state['cross_vol_mean'] = te.groupby('date')['xd_cross_vol_mean'].mean().reindex(daily_df.index)
    daily_state['cross_return_mean'] = te.groupby('date')['xd_cross_return_mean'].mean().reindex(daily_df.index)
    for c in ['A_xsec_full', 'B_xgb_55', 'C_xgb_33', 'D_cat_1stop']:
        daily_state[f'{c}_lag1'] = daily_df[c].shift(1).reindex(daily_df.index)
    daily_state = daily_state.fillna(0)

    X_tr = daily_state.iloc[:gate_split].values
    X_te = daily_state.iloc[gate_split:].values
    gate = CatBoostClassifier(iterations=200, depth=4, learning_rate=0.05, random_seed=42, verbose=0)
    gate.fit(X_tr, winner.values)

    pred_c = gate.predict(X_te).flatten()
    moe_r = np.array([daily_df.iloc[gate_split + i][pc if not isinstance(pc, (np.ndarray, list)) else pc[0]]
                       for i, pc in enumerate(pred_c)])

    r_test_A = daily_df.iloc[gate_split:]['A_xsec_full'].values
    m_A = metrics(r_test_A)
    m_moe = metrics(moe_r)

    print(f'  [{label}] window test ({len(moe_r)} days):')
    print(f'    A_only: tot {m_A["total"]:+.1f}%, Sh {m_A["sharpe"]:+.2f}, DD {m_A["dd"]:+.1f}%, CAGR {m_A["cagr"]:+.1f}%, Calmar {m_A["calmar"]:.2f}')
    print(f'    MoE v1: tot {m_moe["total"]:+.1f}%, Sh {m_moe["sharpe"]:+.2f}, DD {m_moe["dd"]:+.1f}%, CAGR {m_moe["cagr"]:+.1f}%, Calmar {m_moe["calmar"]:.2f}')

    all_daily_moe.append(pd.Series(moe_r, index=daily_df.index[gate_split:]))
    all_daily_A.append(pd.Series(r_test_A, index=daily_df.index[gate_split:]))

# COMBINED metrics
if all_daily_moe:
    combined_moe = pd.concat(all_daily_moe).values
    combined_A = pd.concat(all_daily_A).values
    m_moe_c = metrics(combined_moe)
    m_A_c = metrics(combined_A)
    print(f'\n=== COMBINED 3-window WF ({len(combined_moe)} days) ===')
    print(f'  A_only (baseline):  tot {m_A_c["total"]:+.1f}%, Sh {m_A_c["sharpe"]:+.2f}, DD {m_A_c["dd"]:+.1f}%, CAGR {m_A_c["cagr"]:+.1f}%, Calmar {m_A_c["calmar"]:.2f}, Sortino {m_A_c["sortino"]:.2f}')
    print(f'  MoE v1 gated:       tot {m_moe_c["total"]:+.1f}%, Sh {m_moe_c["sharpe"]:+.2f}, DD {m_moe_c["dd"]:+.1f}%, CAGR {m_moe_c["cagr"]:+.1f}%, Calmar {m_moe_c["calmar"]:.2f}, Sortino {m_moe_c["sortino"]:.2f}')
    print(f'\n  MoE v1 CAGR / A_only CAGR: {m_moe_c["cagr"] / m_A_c["cagr"]:.2f}x')
