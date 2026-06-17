"""
Cross-sectional ranker v2: add risk controls to tame -78% DD.
Controls to test:
  1. Macro gate: only trade when BTC 30d return > 0
  2. Volatility targeting: scale position inverse to forecast vol
  3. Stop-loss per position (-10% hard stop)
  4. Confidence gate (only pick if prediction > threshold)
  5. Multi-asset blending (top-K=2-3 diversifies)
  6. Kelly fraction sizing based on prediction magnitude
"""
import polars as pl, pandas as pd, numpy as np
from pathlib import Path
import glob, warnings, pickle
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed'
MAKER_RT = 0.08

all_fps = sorted(glob.glob(str(DATA/'*_chimera.parquet')))
assets = [Path(f).stem.replace('usdt_v50_chimera','').upper() for f in all_fps]

# Load BTC for macro gate
btc_df = pl.read_parquet(DATA/'btcusdt_v50_chimera.parquet').to_pandas()
btc_df['date'] = pd.to_datetime(btc_df['timestamp'], unit='ms').dt.date
btc_daily = btc_df.groupby('date').agg({'close':'last'}).reset_index()
btc_daily['btc_30d'] = btc_daily['close'].pct_change(30)
btc_daily['date'] = pd.to_datetime(btc_daily['date'])

# Build panel as before
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
    d['asset'] = a
    # For stop-loss, need daily forward series
    d['fwd_1d'] = d['close'].shift(-1) / d['close'] - 1
    d['fwd_2d'] = d['close'].shift(-2) / d['close'] - 1
    d['fwd_3d'] = d['close'].shift(-3) / d['close'] - 1
    # min forward 3d = worst intraday drawdown while holding (conservative estimate)
    d['min_fwd_3d'] = np.minimum.reduce([d['fwd_1d'], d['fwd_2d'], d['fwd_3d']])
    rows.append(d)

panel = pd.concat(rows, ignore_index=True).dropna(subset=['fwd_3d'])
panel['date'] = pd.to_datetime(panel['date'])
panel = panel.merge(btc_daily[['date','btc_30d']], on='date', how='left')

feat_list = [c for c in panel.columns if c.startswith('norm_') or c.startswith('xd_')]
feat_list += ['ret_1d','ret_3d','ret_7d','ret_14d','vol_7d','vol_30d','hl','hurst_regime']
feat_list = [c for c in feat_list if c in panel.columns]

train = panel[panel['date']<'2024-10-01']
val = panel[(panel['date']>='2024-10-01') & (panel['date']<'2025-03-16')]
test = panel[panel['date']>='2025-03-16']
print(f"Train/Val/Test: {len(train):,}/{len(val):,}/{len(test):,}")

from catboost import CatBoostRegressor
reg = CatBoostRegressor(iterations=1000, depth=6, learning_rate=0.03, l2_leaf_reg=3.0,
                         random_seed=42, verbose=0, eval_metric='RMSE', early_stopping_rounds=50)
reg.fit(train[feat_list].fillna(0).values, train['fwd_3d'].values,
        eval_set=(val[feat_list].fillna(0).values, val['fwd_3d'].values),
        use_best_model=True, verbose=0)

test_df = test.copy().reset_index(drop=True)
test_df['pred'] = reg.predict(test_df[feat_list].fillna(0).values)

def backtest(test_df, K=1, macro_gate=False, stop_loss=None, vol_target=False, min_pred=None,
              kelly_sizing=False, label=''):
    """Backtest top-K long with various risk controls."""
    daily_ret = []
    dates = sorted(test_df['date'].unique())
    for d in dates:
        grp = test_df[test_df['date']==d]
        if len(grp) < K:
            daily_ret.append(0.0); continue
        # Macro gate
        btc30 = grp['btc_30d'].iloc[0] if len(grp) else 0
        if macro_gate and (pd.isna(btc30) or btc30 < 0):
            daily_ret.append(0.0); continue
        top = grp.sort_values('pred', ascending=False).head(K).copy()
        # Min prediction gate
        if min_pred is not None and top['pred'].mean() < min_pred:
            daily_ret.append(0.0); continue
        # Realize return for these picks, with optional stop loss
        rets = []
        for _, r in top.iterrows():
            pnl = r['fwd_3d']
            if stop_loss is not None and r['min_fwd_3d'] < -stop_loss:
                # stopped out at stop-loss level
                pnl = -stop_loss
            rets.append(pnl)
        port_ret = np.mean(rets) * 100 - MAKER_RT
        # Volatility targeting: scale by inverse vol
        if vol_target:
            vol_scale = min(1.0, 0.02 / (top['vol_30d'].mean() + 1e-6))  # target 2% daily vol
            port_ret *= vol_scale
        # Kelly sizing (proportional to prediction magnitude, capped)
        if kelly_sizing:
            pred_mag = top['pred'].mean()
            kelly_frac = max(0.25, min(1.5, pred_mag * 10))  # 25%-150% of Kelly
            port_ret *= kelly_frac
        # Divide by 3 for 3-day overlap
        daily_ret.append(port_ret / 3)
    r = np.array(daily_ret)
    ann_vol = r.std() * np.sqrt(252)
    sharpe = r.mean() * 252 / ann_vol if ann_vol > 0 else 0
    eq = np.cumprod(1 + r/100)
    cm = np.maximum.accumulate(eq)
    dd = ((eq - cm)/cm).min() * 100
    total = (eq[-1] - 1) * 100
    n_active = int((np.abs(r) > 1e-6).sum())
    return {'label':label, 'total':total, 'sharpe':sharpe, 'dd':dd, 'mean':r.mean(), 'vol':ann_vol, 'n_active':n_active}

print()
print("=== VARIANTS ===")
print(f"{'label':<50} {'total%':>10} {'sharpe':>7} {'dd%':>7} {'avg/d%':>8} {'n_active':>9}")
configs = [
    ('K=1 baseline', {'K':1}),
    ('K=1 macro gate', {'K':1, 'macro_gate':True}),
    ('K=1 stop -10%', {'K':1, 'stop_loss':0.10}),
    ('K=1 macro + stop -10%', {'K':1, 'macro_gate':True, 'stop_loss':0.10}),
    ('K=1 macro + stop -10% + voltarget', {'K':1, 'macro_gate':True, 'stop_loss':0.10, 'vol_target':True}),
    ('K=1 macro + stop -15%', {'K':1, 'macro_gate':True, 'stop_loss':0.15}),
    ('K=1 min_pred 0.01', {'K':1, 'min_pred':0.01}),
    ('K=1 min_pred 0.02 + macro', {'K':1, 'min_pred':0.02, 'macro_gate':True}),
    ('K=1 macro + stop -10% + kelly', {'K':1, 'macro_gate':True, 'stop_loss':0.10, 'kelly_sizing':True}),
    ('K=2 macro + stop -10%', {'K':2, 'macro_gate':True, 'stop_loss':0.10}),
    ('K=3 macro + stop -10%', {'K':3, 'macro_gate':True, 'stop_loss':0.10}),
    ('K=3 macro + stop -10% + vol', {'K':3, 'macro_gate':True, 'stop_loss':0.10, 'vol_target':True}),
    ('K=5 macro + stop -10%', {'K':5, 'macro_gate':True, 'stop_loss':0.10}),
    ('K=5 macro + stop -10% + kelly', {'K':5, 'macro_gate':True, 'stop_loss':0.10, 'kelly_sizing':True}),
]
for name, kw in configs:
    r = backtest(test_df, **kw, label=name)
    print(f"{name:<50} {r['total']:>+9.1f}% {r['sharpe']:>+6.2f} {r['dd']:>+6.1f}% {r['mean']:>+7.3f}% {r['n_active']:>9}")
