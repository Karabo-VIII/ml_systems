"""
Extend xsec ranker to PERP short-the-bottom-K sleeve.
Test long-K + short-K (dollar-neutral) and long-K only vs short-K only.
"""
import polars as pl, pandas as pd, numpy as np
from pathlib import Path
import glob, warnings, pickle
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed'
MAKER_RT = 0.08

# Reuse trained xsec_ranker model (already saved)
model_path = ROOT / 'models' / 'xsec_ranker' / 'xsec_ranker_v1.pkl'
bundle = pickle.loads(model_path.read_bytes())
reg = bundle['reg']
feat_list = bundle['features']

# Build panel for test period
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
    # For shorts: MAX forward = worst case adverse against a short
    d['max_fwd_3d'] = np.maximum.reduce([d['fwd_1d'], d['fwd_2d'], d['fwd_3d']])
    d['min_fwd_3d'] = np.minimum.reduce([d['fwd_1d'], d['fwd_2d'], d['fwd_3d']])
    d['asset'] = a
    rows.append(d)
panel = pd.concat(rows, ignore_index=True).dropna(subset=['fwd_3d'])
panel['date'] = pd.to_datetime(panel['date'])
panel = panel.merge(btc_d[['date','btc_30d']], on='date', how='left')

test = panel[panel['date']>='2025-03-16'].copy()
test['pred'] = reg.predict(test[feat_list].fillna(0).values)

def run(test_df, K_long=0, K_short=0, macro_gate=False, short_in_bear_only=False, stop_long=0.10, stop_short=0.10):
    """
    K_long: number of long positions from top of ranking
    K_short: number of short positions from bottom of ranking
    macro_gate: if True, only LONG when BTC 30d > 0
    short_in_bear_only: if True, only SHORT when BTC 30d < 0 (capturing bear regime)
    """
    daily_ret = []
    dates = sorted(test_df['date'].unique())
    for d in dates:
        grp = test_df[test_df['date']==d]
        if len(grp) < K_long + K_short:
            daily_ret.append(0.0); continue
        btc30 = grp['btc_30d'].iloc[0]
        btc30 = 0 if pd.isna(btc30) else btc30

        long_ret = 0.0
        short_ret = 0.0

        if K_long > 0:
            if macro_gate and btc30 < 0:
                pass
            else:
                top = grp.sort_values('pred', ascending=False).head(K_long)
                rets = []
                for _, r in top.iterrows():
                    pnl = r['fwd_3d']
                    if stop_long is not None and r['min_fwd_3d'] < -stop_long:
                        pnl = -stop_long
                    rets.append(pnl)
                long_ret = np.mean(rets) * 100 - MAKER_RT

        if K_short > 0:
            if short_in_bear_only and btc30 >= 0:
                pass
            else:
                bot = grp.sort_values('pred', ascending=True).head(K_short)
                rets = []
                for _, r in bot.iterrows():
                    # Short: profit if price drops. pnl = -fwd_3d
                    pnl = -r['fwd_3d']
                    # Stop-out: if price moves UP (max_fwd_3d) > stop_short, stopped
                    if stop_short is not None and r['max_fwd_3d'] > stop_short:
                        pnl = -stop_short
                    rets.append(pnl)
                short_ret = np.mean(rets) * 100 - MAKER_RT

        # Portfolio return: average of long and short sleeves, or just one if K=0
        n_sleeves = (1 if K_long > 0 else 0) + (1 if K_short > 0 else 0)
        if n_sleeves == 0:
            daily_ret.append(0.0); continue
        port_ret = (long_ret + short_ret) / n_sleeves / 3  # /3 for 3-day overlap
        daily_ret.append(port_ret)
    r = np.array(daily_ret)
    if r.std() == 0:
        return {'total':0,'sharpe':0,'dd':0,'mean':0,'vol':0}
    eq = np.cumprod(1 + r/100)
    cm = np.maximum.accumulate(eq)
    dd = ((eq-cm)/cm).min()*100
    sharpe = r.mean()*252 / (r.std()*np.sqrt(252))
    return {'total':(eq[-1]-1)*100, 'sharpe':sharpe, 'dd':dd, 'mean':r.mean()*100, 'vol':r.std()*np.sqrt(252)*100}

print("=== PERP short-the-bottom-K extension of ranker ===")
print(f"{'config':<55} {'total%':>9} {'Sh':>5} {'dd%':>7} {'vol%':>6}")
configs = [
    ('K_long=1 stop (baseline from yesterday)', dict(K_long=1, stop_long=0.10)),
    ('K_short=1 stop (SHORT bottom only)', dict(K_short=1, stop_short=0.10)),
    ('K_short=3 stop (SHORT bottom-3 only)', dict(K_short=3, stop_short=0.10)),
    ('K_short=5 stop (SHORT bottom-5 only)', dict(K_short=5, stop_short=0.10)),
    ('K_long=1 + K_short=1 (dollar-neutral top/bot)', dict(K_long=1, K_short=1, stop_long=0.10, stop_short=0.10)),
    ('K_long=3 + K_short=3 dollar-neutral', dict(K_long=3, K_short=3, stop_long=0.10, stop_short=0.10)),
    ('K_long=5 + K_short=5 dollar-neutral', dict(K_long=5, K_short=5, stop_long=0.10, stop_short=0.10)),
    ('K_long=3 macro + K_short=3 bear-only (regime symm)',
      dict(K_long=3, K_short=3, macro_gate=True, short_in_bear_only=True, stop_long=0.10, stop_short=0.10)),
    ('K_long=1 always + K_short=1 bear-only',
      dict(K_long=1, K_short=1, short_in_bear_only=True, stop_long=0.10, stop_short=0.10)),
]
for label, kw in configs:
    r = run(test, **kw)
    print(f"{label:<55} {r['total']:>+8.1f}% {r['sharpe']:>+4.2f} {r['dd']:>+6.1f}% {r['vol']:>5.1f}%")
