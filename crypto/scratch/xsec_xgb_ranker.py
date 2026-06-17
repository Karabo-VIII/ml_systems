"""
XGBoost LambdaMART via rank:ndcg and rank:pairwise objectives.
Compare to CatBoost regressor and CatBoost YetiRank head-to-head.
"""
# [!] SPLIT DISCIPLINE NOTE (2026-05-24 INST-C cleanup):
# This script uses the legacy convention where "OOS"/"test" labels the post-TRAIN
# window (= canonical OOS + UNSEEN combined). Per src/split_config.py the
# canonical OOS ends 2025-12-31 and UNSEEN starts 2026-01-01. The dates hardcoded
# below are intentionally preserved for reproducibility of prior outputs. New
# scripts must import from split_config -- see docs/SPLIT_DISCIPLINE.md.
import polars as pl, pandas as pd, numpy as np
from pathlib import Path
import glob, warnings, pickle
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed'
MAKER_RT = 0.08

all_fps = sorted(glob.glob(str(DATA/'*_chimera.parquet')))
assets = [Path(f).stem.replace('usdt_v50_chimera','').upper() for f in all_fps]

btc = pl.read_parquet(DATA/'btcusdt_v50_chimera.parquet').to_pandas()
btc['date'] = pd.to_datetime(btc['timestamp'], unit='ms').dt.date
btc_d = btc.groupby('date').agg({'close':'last'}).reset_index()
btc_d['btc_30d'] = btc_d['close'].pct_change(30)
btc_d['date'] = pd.to_datetime(btc_d['date'])

rows = []
for a in assets:
    fp = DATA/f'{a.lower()}usdt_v50_chimera.parquet'
    df = pl.read_parquet(fp).to_pandas()
    if len(df) < 1000: continue
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
    agg = {'close':'last','open':'first','high':'max','low':'min','volume':'sum'}
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
    d['hl'] = (d['high']-d['low'])/d['open']
    d['fwd_1d'] = d['close'].shift(-1)/d['close'] - 1
    d['fwd_2d'] = d['close'].shift(-2)/d['close'] - 1
    d['fwd_3d'] = d['close'].shift(-3)/d['close'] - 1
    d['max_fwd_3d'] = np.maximum.reduce([d['fwd_1d'], d['fwd_2d'], d['fwd_3d']])
    d['min_fwd_3d'] = np.minimum.reduce([d['fwd_1d'], d['fwd_2d'], d['fwd_3d']])
    d['asset'] = a
    rows.append(d)
panel = pd.concat(rows, ignore_index=True).dropna(subset=['fwd_3d'])
panel['date'] = pd.to_datetime(panel['date'])
panel = panel.merge(btc_d[['date','btc_30d']], on='date', how='left')
feat_list = [c for c in panel.columns if c.startswith('norm_') or c.startswith('xd_')]
feat_list += ['ret_1d','ret_3d','ret_7d','ret_14d','vol_7d','vol_30d','hl','hurst_regime']
feat_list = [c for c in feat_list if c in panel.columns]

panel = panel.sort_values(['date','asset']).reset_index(drop=True)
# Integer rank target 0..31 for NDCG
panel['rank_target'] = panel.groupby('date')['fwd_3d'].rank(pct=True).apply(lambda x: int(x * 31))

train = panel[panel['date']<'2024-10-01'].copy()
val = panel[(panel['date']>='2024-10-01') & (panel['date']<'2025-03-16')].copy()
test = panel[panel['date']>='2025-03-16'].copy()

train_group = train.groupby('date').size().values
val_group = val.groupby('date').size().values
test_group = test.groupby('date').size().values

X_tr = train[feat_list].fillna(0).values
y_tr = train['rank_target'].values
X_va = val[feat_list].fillna(0).values
y_va = val['rank_target'].values
X_te = test[feat_list].fillna(0).values

import xgboost as xgb

print(f'XGBoost version: {xgb.__version__}')

results = {}
for obj in ['rank:ndcg', 'rank:pairwise', 'rank:map']:
    try:
        print(f'\nTraining {obj}...')
        ranker = xgb.XGBRanker(
            objective=obj,
            tree_method='hist',
            learning_rate=0.05,
            max_depth=6,
            n_estimators=500,
            random_state=42,
            eval_metric='ndcg@5',
            early_stopping_rounds=30,
        )
        ranker.fit(X_tr, y_tr, group=train_group,
                    eval_set=[(X_va, y_va)], eval_group=[val_group],
                    verbose=False)
        test[f'pred_{obj.replace(":","_")}'] = ranker.predict(X_te)
        print(f'  best iter: {ranker.best_iteration}')
        results[obj] = ranker
    except Exception as e:
        print(f'  FAILED: {e}')

def backtest(test_df, K_long, K_short, pred_col, stop_loss=0.10):
    daily_ret = []
    dates = sorted(test_df['date'].unique())
    for d in dates:
        grp = test_df[test_df['date']==d]
        if len(grp) < K_long + K_short:
            daily_ret.append(0.0); continue
        long_ret = 0.0; short_ret = 0.0
        if K_long > 0:
            top = grp.sort_values(pred_col, ascending=False).head(K_long)
            rs = []
            for _, r in top.iterrows():
                p = r['fwd_3d']
                if stop_loss and r['min_fwd_3d'] < -stop_loss: p = -stop_loss
                rs.append(p)
            long_ret = np.mean(rs) * 100 - MAKER_RT
        if K_short > 0:
            bot = grp.sort_values(pred_col, ascending=True).head(K_short)
            rs = []
            for _, r in bot.iterrows():
                p = -r['fwd_3d']
                if stop_loss and r['max_fwd_3d'] > stop_loss: p = -stop_loss
                rs.append(p)
            short_ret = np.mean(rs) * 100 - MAKER_RT
        n_sleeves = (1 if K_long>0 else 0) + (1 if K_short>0 else 0)
        port = (long_ret + short_ret) / max(n_sleeves,1) / 3
        daily_ret.append(port)
    r = np.array(daily_ret)
    if r.std() == 0:
        return {'total':0,'sharpe':0,'dd':0}
    eq = np.cumprod(1 + r/100)
    cm = np.maximum.accumulate(eq)
    dd = ((eq-cm)/cm).min()*100
    sharpe = r.mean()*252 / (r.std()*np.sqrt(252))
    return {'total':(eq[-1]-1)*100, 'sharpe':sharpe, 'dd':dd}

print('\n=== XGBoost LambdaMART objectives — test set 2025-03-16 -> 2026-04-16 ===')
print(f"{'objective':<18} {'config':<16} {'total%':>10} {'Sh':>5} {'DD%':>7}")
for obj in ['rank:ndcg', 'rank:pairwise', 'rank:map']:
    col = f'pred_{obj.replace(":","_")}'
    if col not in test.columns: continue
    for (kl, ks, label) in [(1,0,'K=1 long'), (3,0,'K=3 long'), (5,0,'K=5 long'),
                              (1,1,'K=1+1 d-neut'), (3,3,'K=3+3 d-neut'), (5,5,'K=5+5 d-neut')]:
        r = backtest(test, kl, ks, col)
        print(f"{obj:<18} {label:<16} {r['total']:>+9.1f}% {r['sharpe']:>+4.2f} {r['dd']:>+6.1f}%")

# Save best
if 'rank:ndcg' in results:
    (ROOT/'models/xsec_ranker/xgb_ndcg_v1.pkl').write_bytes(pickle.dumps({'ranker': results['rank:ndcg'], 'features': feat_list}))
    print(f'\nSaved XGBoost rank:ndcg ranker to models/xsec_ranker/xgb_ndcg_v1.pkl')
