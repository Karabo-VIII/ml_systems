"""XGB rank:ndcg walk-forward at 4h cadence.

Question: does the xsec ranker that ships at daily (n=128 days, Sh 3.70 OOS)
survive at 4h cadence, unlocking 6x more decisions/day?

Mirror of scratch/xsec_xgb_walkforward.py but:
    - 4h bucketing instead of daily resample
    - fwd target = fwd_3 buckets (= 12h forward)
    - vol/return windows rescaled 6x (ret_6bars = prior 24h, ret_42bars = 7d, etc.)
    - stop = 0.10 (unchanged; realized 12h moves similar magnitude to 3d daily)
    - MAKER_RT = 0.08 (same)

Ship criterion: any (K_long, K_short) cell has
    - 3/3 WF windows positive Sharpe AND
    - combined Sharpe >= 2.0 AND
    - combined DD > -30%
at 4h. If yes -> shipped, re-rank with the dual-axis framework. If no -> concede.
"""
import polars as pl, pandas as pd, numpy as np
from pathlib import Path
import glob, warnings, time, sys
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed'
MAKER_RT = 0.08
BUCKET_MS = 4 * 60 * 60 * 1000  # 4h

all_fps = sorted(glob.glob(str(DATA/'*_chimera.parquet')))
assets = [Path(f).stem.replace('usdt_v50_chimera','').upper() for f in all_fps]
print(f"[info] universe size: {len(assets)}")

# Load BTC at 4h for regime feature
btc = pl.read_parquet(DATA/'btcusdt_v50_chimera.parquet', columns=['timestamp','close']).to_pandas()
btc['bucket'] = btc['timestamp'] // BUCKET_MS
btc_b = btc.groupby('bucket').agg({'close':'last','timestamp':'last'}).reset_index()
btc_b = btc_b.rename(columns={'close':'btc_close'})
btc_b['btc_ret_6'] = btc_b['btc_close'].pct_change(6)        # 24h (6 buckets)
btc_b['btc_ret_42'] = btc_b['btc_close'].pct_change(42)     # 7 days
btc_b['btc_30d'] = btc_b['btc_close'].pct_change(180)       # 30 days (180 buckets)
btc_b['date'] = pd.to_datetime(btc_b['timestamp'], unit='ms')

t0 = time.time()
rows = []
for a in assets:
    fp = DATA/f'{a.lower()}usdt_v50_chimera.parquet'
    try:
        df = pl.read_parquet(fp).to_pandas()
    except Exception:
        continue
    if len(df) < 5000: continue
    df['bucket'] = df['timestamp'] // BUCKET_MS
    agg = {'close':'last','open':'first','high':'max','low':'min','volume':'sum','timestamp':'last'}
    norm_cols = [c for c in df.columns if c.startswith('norm_') or c.startswith('xd_')]
    for c in norm_cols:
        agg[c] = 'last'
    if 'hurst_regime' in df.columns:
        agg['hurst_regime'] = 'last'
    d = df.groupby('bucket').agg(agg).reset_index()
    d['ret_1'] = d['close'].pct_change(1)
    d['ret_6'] = d['close'].pct_change(6)        # 24h
    d['ret_18'] = d['close'].pct_change(18)     # 72h (3d)
    d['ret_42'] = d['close'].pct_change(42)     # 7d
    d['ret_84'] = d['close'].pct_change(84)     # 14d
    d['vol_42'] = d['ret_1'].rolling(42).std()
    d['vol_180'] = d['ret_1'].rolling(180).std()
    d['hl'] = (d['high']-d['low'])/d['open']
    d['fwd_1'] = d['close'].shift(-1)/d['close']-1     # 4h forward
    d['fwd_3'] = d['close'].shift(-3)/d['close']-1     # 12h forward
    d['fwd_6'] = d['close'].shift(-6)/d['close']-1     # 24h forward
    d['max_fwd_3'] = np.maximum.reduce([d['close'].shift(-i)/d['close']-1 for i in [1,2,3]])
    d['min_fwd_3'] = np.minimum.reduce([d['close'].shift(-i)/d['close']-1 for i in [1,2,3]])
    d['asset'] = a
    d['date'] = pd.to_datetime(d['timestamp'], unit='ms')
    rows.append(d)
print(f"[info] loaded {len(rows)} assets in {time.time()-t0:.1f}s")

panel = pd.concat(rows, ignore_index=True).dropna(subset=['fwd_3', 'ret_6', 'vol_42'])
panel = panel.merge(btc_b[['bucket','btc_30d','btc_ret_6','btc_ret_42']], on='bucket', how='left')
panel['asset_vs_btc_42'] = panel['ret_42'] - panel['btc_ret_42']
panel['btc_regime'] = np.sign(panel['btc_ret_42'].fillna(0))

feat_list = [c for c in panel.columns if c.startswith('norm_') or c.startswith('xd_')]
feat_list += ['ret_1','ret_6','ret_18','ret_42','ret_84','vol_42','vol_180','hl','hurst_regime',
              'btc_30d','btc_ret_6','btc_ret_42','asset_vs_btc_42','btc_regime']
feat_list = [c for c in feat_list if c in panel.columns]
print(f"[info] feature count: {len(feat_list)}")

# Rank target: per-BUCKET (date) percentile rank of fwd_3 (12h forward)
panel = panel.sort_values(['date','asset']).reset_index(drop=True)
panel['rank_target'] = panel.groupby('date')['fwd_3'].rank(pct=True).apply(lambda x: int(x*31))
panel = panel.dropna(subset=['rank_target'])

print(f"[info] panel shape: {panel.shape}, unique buckets: {panel['date'].nunique()}")

import xgboost as xgb

def train_test(train_end, test_start, test_end):
    tr = panel[panel['date']<train_end]
    te = panel[(panel['date']>=test_start)&(panel['date']<test_end)].copy()
    if len(tr)<20000 or len(te)<2000:
        print(f"  skip: tr={len(tr)} te={len(te)}")
        return None
    tr_g = tr.groupby('date').size().values
    ranker = xgb.XGBRanker(
        objective='rank:ndcg', tree_method='hist', learning_rate=0.05,
        max_depth=6, n_estimators=300, random_state=42, eval_metric='ndcg@5',
    )
    ranker.fit(tr[feat_list].fillna(0).values, tr['rank_target'].values, group=tr_g, verbose=False)
    te = te.assign(pred=ranker.predict(te[feat_list].fillna(0).values))
    return te

def bt(te, K_long, K_short, stop=0.10):
    """Backtest -- returns list of per-bucket bucket-net-returns."""
    bucket_r = []
    for d in sorted(te['date'].unique()):
        grp = te[te['date']==d]
        if len(grp) < K_long+K_short:
            bucket_r.append(0); continue
        long_r=short_r=0
        if K_long>0:
            top = grp.sort_values('pred',ascending=False).head(K_long)
            rs=[]
            for _,r in top.iterrows():
                p=r['fwd_3']
                if stop and r['min_fwd_3']<-stop: p=-stop
                rs.append(p)
            long_r = np.mean(rs)*100 - MAKER_RT
        if K_short>0:
            bot = grp.sort_values('pred',ascending=True).head(K_short)
            rs=[]
            for _,r in bot.iterrows():
                p=-r['fwd_3']
                if stop and r['max_fwd_3']>stop: p=-stop
                rs.append(p)
            short_r = np.mean(rs)*100 - MAKER_RT
        n_s = (1 if K_long>0 else 0)+(1 if K_short>0 else 0)
        bucket_r.append((long_r+short_r)/max(n_s,1)/3)   # /3 because 3-bar hold = serialized overlap reduction
    r = np.array(bucket_r)
    if r.std()==0:
        return {'total':0, 'sharpe':0, 'dd':0, 'n_buckets':len(r)}
    # annualization: 6 buckets/day x 252 = 1512 buckets/yr
    eq = np.cumprod(1+r/100)
    cm = np.maximum.accumulate(eq)
    return {
        'total':(eq[-1]-1)*100,
        'sharpe':r.mean()*1512/(r.std()*np.sqrt(1512)),
        'dd':((eq-cm)/cm).min()*100,
        'n_buckets':len(r),
    }

print('\n=== XGB rank:ndcg @ 4h CADENCE walk-forward ===')
windows = [
    ('2024-10-01', '2024-10-01', '2025-03-16', 'WF1 Oct24-Mar25'),
    ('2025-03-16', '2025-03-16', '2025-09-01', 'WF2 Mar25-Sep25'),
    ('2025-09-01', '2025-09-01', '2026-04-16', 'WF3 Sep25-Apr26'),
    ('2024-10-01', '2024-10-01', '2026-04-16', 'COMBINED 18mo'),
]
print(f"{'window':<22} {'K=1L':>14} {'K=3L':>14} {'K=3+3dn':>14} {'K=5+5dn':>14}")
results = {}
for train_end, ts, te_end, label in windows:
    te = train_test(train_end, ts, te_end)
    if te is None:
        print(f"{label:<22} [insufficient data]")
        continue
    r1 = bt(te, 1, 0)
    r3 = bt(te, 3, 0)
    r33 = bt(te, 3, 3)
    r55 = bt(te, 5, 5)
    results[label] = (r1, r3, r33, r55)
    print(f"{label:<22} {r1['total']:>+7.1f}% Sh{r1['sharpe']:>+4.2f} "
          f"{r3['total']:>+7.1f}% Sh{r3['sharpe']:>+4.2f} "
          f"{r33['total']:>+7.1f}% Sh{r33['sharpe']:>+4.2f} "
          f"{r55['total']:>+7.1f}% Sh{r55['sharpe']:>+4.2f}")

# Ship decision
print()
combined = results.get('COMBINED 18mo')
if combined:
    best_idx = max(range(4), key=lambda i: combined[i]['sharpe'])
    best_names = ['K=1L', 'K=3L', 'K=3+3dn', 'K=5+5dn']
    best = combined[best_idx]
    wf_sharpes = []
    for lbl in ['WF1 Oct24-Mar25', 'WF2 Mar25-Sep25', 'WF3 Sep25-Apr26']:
        if lbl in results:
            wf_sharpes.append(results[lbl][best_idx]['sharpe'])
    pos_windows = sum(1 for s in wf_sharpes if s > 0)
    print(f"Best strategy at 4h: {best_names[best_idx]}")
    print(f"  Combined: total={best['total']:+.1f}% Sharpe={best['sharpe']:+.2f} DD={best['dd']:+.1f}%")
    print(f"  WF windows positive: {pos_windows}/{len(wf_sharpes)}")
    if pos_windows == len(wf_sharpes) and best['sharpe'] >= 2.0 and best['dd'] > -30:
        print(f"[SHIP] xsec XGB ranker survives at 4h; deploy at sub-day cadence for 6x decision density")
    else:
        print(f"[CONCEDE] xsec XGB at 4h fails ship criteria (Sh>=2.0, 3/3 WF pos, DD>-30%)")
