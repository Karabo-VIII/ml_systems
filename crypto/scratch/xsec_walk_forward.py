"""
Walk-forward validation of the cross-sectional ranker.
Train 3 different models on 3 different cutoffs, test each on its forward window.
If +668% is regime-specific, we'll see the drop here.
"""
import polars as pl, pandas as pd, numpy as np
from pathlib import Path
import glob, warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed'
MAKER_RT = 0.08

all_fps = sorted(glob.glob(str(DATA/'*_chimera.parquet')))
assets = [Path(f).stem.replace('usdt_v50_chimera','').upper() for f in all_fps]

# Load BTC for macro gate
btc = pl.read_parquet(DATA/'btcusdt_v50_chimera.parquet').to_pandas()
btc['date'] = pd.to_datetime(btc['timestamp'], unit='ms').dt.date
btc_d = btc.groupby('date').agg({'close':'last'}).reset_index()
btc_d['btc_30d'] = btc_d['close'].pct_change(30)
btc_d['date'] = pd.to_datetime(btc_d['date'])

# Panel
rows = []
for a in assets:
    fp = DATA/f'{a.lower()}usdt_v50_chimera.parquet'
    df = pl.read_parquet(fp).to_pandas()
    if len(df) < 1000: continue
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
    agg = {'close':'last','open':'first','high':'max','low':'min','volume':'sum'}
    feat_cols = [c for c in df.columns if c.startswith('norm_') or c.startswith('xd_') or c == 'hurst_regime']
    for fc in feat_cols: agg[fc] = 'last'
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
    d['min_fwd_3d'] = np.minimum.reduce([d['fwd_1d'], d['fwd_2d'], d['fwd_3d']])
    d['asset'] = a
    rows.append(d)
panel = pd.concat(rows, ignore_index=True).dropna(subset=['fwd_3d'])
panel['date'] = pd.to_datetime(panel['date'])
panel = panel.merge(btc_d[['date','btc_30d']], on='date', how='left')
feat_list = [c for c in panel.columns if c.startswith('norm_') or c.startswith('xd_')]
feat_list += ['ret_1d','ret_3d','ret_7d','ret_14d','vol_7d','vol_30d','hl','hurst_regime']
feat_list = [c for c in feat_list if c in panel.columns]

from catboost import CatBoostRegressor

def train_and_test(train_end, test_start, test_end, label):
    tr = panel[panel['date'] < train_end]
    te = panel[(panel['date']>=test_start) & (panel['date']<test_end)]
    if len(tr) < 5000 or len(te) < 500:
        print(f"  {label}: insufficient data tr={len(tr)} te={len(te)}")
        return None
    reg = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.03, l2_leaf_reg=3.0,
                             random_seed=42, verbose=0)
    reg.fit(tr[feat_list].fillna(0).values, tr['fwd_3d'].values, verbose=0)
    te_df = te.copy().reset_index(drop=True)
    te_df['pred'] = reg.predict(te_df[feat_list].fillna(0).values)
    return te_df

def backtest(te_df, K=1, macro_gate=True, stop_loss=0.10):
    daily_ret = []
    dates = sorted(te_df['date'].unique())
    for d in dates:
        grp = te_df[te_df['date']==d]
        if len(grp) < K:
            daily_ret.append(0.0); continue
        btc30 = grp['btc_30d'].iloc[0]
        if macro_gate and (pd.isna(btc30) or btc30 < 0):
            daily_ret.append(0.0); continue
        top = grp.sort_values('pred', ascending=False).head(K).copy()
        rets = []
        for _, r in top.iterrows():
            pnl = r['fwd_3d']
            if stop_loss is not None and r['min_fwd_3d'] < -stop_loss:
                pnl = -stop_loss
            rets.append(pnl)
        port_ret = np.mean(rets) * 100 - MAKER_RT
        daily_ret.append(port_ret / 3)
    r = np.array(daily_ret)
    if r.std() == 0:
        return {'total':0,'sharpe':0,'dd':0,'mean':0,'n':0}
    eq = np.cumprod(1 + r/100)
    cm = np.maximum.accumulate(eq)
    dd = ((eq-cm)/cm).min()*100
    total = (eq[-1]-1)*100
    sharpe = r.mean()*252 / (r.std()*np.sqrt(252))
    return {'total':total, 'sharpe':sharpe, 'dd':dd, 'mean':r.mean()*100, 'n':int((np.abs(r)>1e-6).sum())}

print("=== WALK-FORWARD VALIDATION: is +668%/13mo stable across regimes? ===")
print(f"{'window':<35} {'n_days':>7} {'K=1 stop tot':>14} {'Sh':>5} {'K=3 macro+st tot':>18} {'Sh':>5} {'K=5 macro+st tot':>18} {'Sh':>5}")

# 3 non-overlapping test windows, each with its own training cutoff
windows = [
    ('2024-10-01', '2024-10-01', '2025-03-16', 'WF1: Oct24-Mar25 (5mo)'),
    ('2025-03-16', '2025-03-16', '2025-09-01', 'WF2: Mar25-Sep25 (6mo)'),
    ('2025-09-01', '2025-09-01', '2026-04-16', 'WF3: Sep25-Apr26 (7mo)'),
    ('2024-10-01', '2024-10-01', '2026-04-16', 'COMBINED: Oct24-Apr26 (18mo)'),
]
for train_end, test_start, test_end, label in windows:
    te_df = train_and_test(train_end, test_start, test_end, label)
    if te_df is None: continue
    # K=1 + stop only (no macro)
    r1 = backtest(te_df, K=1, macro_gate=False, stop_loss=0.10)
    # K=3 macro + stop
    r3 = backtest(te_df, K=3, macro_gate=True, stop_loss=0.10)
    # K=5 macro + stop
    r5 = backtest(te_df, K=5, macro_gate=True, stop_loss=0.10)
    ndays = len(te_df['date'].unique())
    print(f"{label:<35} {ndays:>7} {r1['total']:>+12.1f}% {r1['sharpe']:>+4.2f} {r3['total']:>+16.1f}% {r3['sharpe']:>+4.2f} {r5['total']:>+16.1f}% {r5['sharpe']:>+4.2f}")
