"""XGB binary classification at 4h cadence -- paradigm-distinct from rank:ndcg.

Motivation: prior xsec rank:ndcg at 4h conceded (Sharpe 0.19 vs daily 3.70).
Rank optimizes ordinal position; at 4h, most bars' forward returns fail to
clear the 0.12% taker RT cost, so ranking coin-flip-noise above coin-flip-noise
gives Sharpe 0.19.

Cost-aware binary classification optimizes the DECISION BOUNDARY that matters:
    y_long_tw  = (fwd_3bucket > +COST_THRESH)  # is there a tradable long?
    y_short_tw = (fwd_3bucket < -COST_THRESH)  # is there a tradable short?

At 4h taker RT = 0.12%, COST_THRESH = 0.002 (2x cost). Expected class prevalence
~25-35% per side; XGB binary with scale_pos_weight handles imbalance.

Inference: trade only if p > 0.55. Size by (p - 0.5) * 2. Trade both sides
independently (long + short can co-exist as market-neutral when both fire).

Ship criterion: 3/3 WF windows positive Sharpe, combined >= 1.5, DD > -30%.
Concede otherwise.
"""
import polars as pl, pandas as pd, numpy as np
from pathlib import Path
import glob, warnings, time, sys
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed'
TAKER_RT_PCT = 0.12    # taker RT cost in %
COST_THRESH = 0.002    # 2x taker RT = 0.2% forward move threshold
PROB_THRESH = 0.55     # trade only if p > 0.55
BUCKET_MS = 4 * 60 * 60 * 1000  # 4h

all_fps = sorted(glob.glob(str(DATA/'*_chimera.parquet')))
assets = [Path(f).stem.replace('usdt_v50_chimera','').upper() for f in all_fps]
print(f"[info] universe size: {len(assets)}")

# BTC at 4h for regime feature
btc = pl.read_parquet(DATA/'btcusdt_v50_chimera.parquet', columns=['timestamp','close']).to_pandas()
btc['bucket'] = btc['timestamp'] // BUCKET_MS
btc_b = btc.groupby('bucket').agg({'close':'last','timestamp':'last'}).reset_index()
btc_b = btc_b.rename(columns={'close':'btc_close'})
btc_b['btc_ret_6'] = btc_b['btc_close'].pct_change(6)
btc_b['btc_ret_42'] = btc_b['btc_close'].pct_change(42)
btc_b['btc_30d'] = btc_b['btc_close'].pct_change(180)
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
    d['ret_6'] = d['close'].pct_change(6)
    d['ret_18'] = d['close'].pct_change(18)
    d['ret_42'] = d['close'].pct_change(42)
    d['vol_42'] = d['ret_1'].rolling(42).std()
    d['vol_180'] = d['ret_1'].rolling(180).std()
    d['hl'] = (d['high']-d['low'])/d['open']
    d['fwd_3'] = d['close'].shift(-3)/d['close']-1    # 12h forward
    d['max_fwd_3'] = np.maximum.reduce([d['close'].shift(-i)/d['close']-1 for i in [1,2,3]])
    d['min_fwd_3'] = np.minimum.reduce([d['close'].shift(-i)/d['close']-1 for i in [1,2,3]])
    d['asset'] = a
    d['date'] = pd.to_datetime(d['timestamp'], unit='ms')
    rows.append(d)
print(f"[info] loaded {len(rows)} assets in {time.time()-t0:.1f}s")

panel = pd.concat(rows, ignore_index=True).dropna(subset=['fwd_3','ret_6','vol_42'])
panel = panel.merge(btc_b[['bucket','btc_30d','btc_ret_6','btc_ret_42']], on='bucket', how='left')
panel['asset_vs_btc_42'] = panel['ret_42'] - panel['btc_ret_42']
panel['btc_regime'] = np.sign(panel['btc_ret_42'].fillna(0))

# BINARY LABELS (directional trade-worthiness)
panel['y_long'] = (panel['fwd_3'] > COST_THRESH).astype(int)
panel['y_short'] = (panel['fwd_3'] < -COST_THRESH).astype(int)

# Report class balance
print(f"[info] class balance: y_long={panel['y_long'].mean():.3f}, y_short={panel['y_short'].mean():.3f}")

feat_list = [c for c in panel.columns if c.startswith('norm_') or c.startswith('xd_')]
feat_list += ['ret_1','ret_6','ret_18','ret_42','vol_42','vol_180','hl','hurst_regime',
              'btc_30d','btc_ret_6','btc_ret_42','asset_vs_btc_42','btc_regime']
feat_list = [c for c in feat_list if c in panel.columns]
print(f"[info] feature count: {len(feat_list)}")

panel = panel.sort_values(['date','asset']).reset_index(drop=True)
print(f"[info] panel shape: {panel.shape}, unique buckets: {panel['date'].nunique()}")

import xgboost as xgb

def train_test_cls(train_end, test_start, test_end, side):
    """Train binary classifier for one side. Returns te with p_{side} column."""
    tr = panel[panel['date']<train_end]
    te = panel[(panel['date']>=test_start)&(panel['date']<test_end)].copy()
    if len(tr)<20000 or len(te)<2000:
        return None
    y_col = f'y_{side}'
    pos = tr[y_col].sum()
    neg = (tr[y_col]==0).sum()
    spw = neg / max(pos, 1)
    clf = xgb.XGBClassifier(
        objective='binary:logistic', tree_method='hist', learning_rate=0.03,
        max_depth=5, n_estimators=700, random_state=42, eval_metric='aucpr',
        scale_pos_weight=spw, subsample=0.8, colsample_bytree=0.7,
    )
    clf.fit(tr[feat_list].fillna(0).values, tr[y_col].values, verbose=False)
    te_y = te[y_col].values
    te_p = clf.predict_proba(te[feat_list].fillna(0).values)[:, 1]
    # Quick AUC report on test fold
    try:
        from sklearn.metrics import roc_auc_score, precision_score
        auc = roc_auc_score(te_y, te_p) if len(np.unique(te_y)) > 1 else float('nan')
        pred_at_thresh = (te_p > PROB_THRESH).astype(int)
        prec = precision_score(te_y, pred_at_thresh, zero_division=0) if pred_at_thresh.sum() > 0 else float('nan')
        print(f"    {side}: AUC={auc:.3f}  prec@{PROB_THRESH}={prec:.3f}  n_fire={pred_at_thresh.sum()}/{len(te_p)}")
    except Exception:
        pass
    te = te.assign(**{f'p_{side}': te_p})
    return te


def bt_cls(te_long, te_short, stop=0.10):
    """Backtest: trade long if p_long > PROB_THRESH, short if p_short > PROB_THRESH.
    Size by (p - 0.5) * 2 normalized. Both sides may co-fire same bar on different assets."""
    # Merge both side predictions
    # te_long has p_long, te_short has p_short; use same rows
    te = te_long.copy()
    te['p_short'] = te_short['p_short'].values
    bucket_r = []
    for d in sorted(te['date'].unique()):
        grp = te[te['date']==d]
        # Long fires
        longs = grp[grp['p_long'] > PROB_THRESH].copy()
        # Short fires (exclude any that also fired long to avoid contradictory trades)
        shorts = grp[(grp['p_short'] > PROB_THRESH) & (grp['p_long'] <= PROB_THRESH)].copy()
        trades = []
        for _, r in longs.iterrows():
            w = (r['p_long'] - 0.5) * 2   # in [0, 1]
            p = r['fwd_3']
            if stop and r['min_fwd_3'] < -stop: p = -stop
            trades.append(w * p)
        for _, r in shorts.iterrows():
            w = (r['p_short'] - 0.5) * 2
            p = -r['fwd_3']
            if stop and r['max_fwd_3'] > stop: p = -stop
            trades.append(w * p)
        if not trades:
            bucket_r.append(0)
            continue
        # Normalize total weight to 1; apply maker cost
        total_w = (longs['p_long']-0.5).sum()*2 + (shorts['p_short']-0.5).sum()*2
        mean_r = sum(trades) / max(total_w, 1e-9)
        bucket_r.append(mean_r*100 - TAKER_RT_PCT)
    r = np.array(bucket_r)
    non_zero = r[r != 0]
    n_active = len(non_zero)
    if r.std()==0 or n_active == 0:
        return {'total':0,'sharpe':0,'dd':0,'n_active':0,'n_total':len(r)}
    # annualization 6/day x 252 = 1512 buckets/yr; divide by 3 for hold-overlap
    r = r / 3
    eq = np.cumprod(1+r/100)
    cm = np.maximum.accumulate(eq)
    return {
        'total':(eq[-1]-1)*100,
        'sharpe':r.mean()*1512/(r.std()*np.sqrt(1512)),
        'dd':((eq-cm)/cm).min()*100,
        'n_active':n_active,
        'n_total':len(r),
    }


print('\n=== XGB BINARY CLASSIFICATION @ 4h walk-forward ===')
windows = [
    ('2024-10-01', '2024-10-01', '2025-03-16', 'WF1 Oct24-Mar25'),
    ('2025-03-16', '2025-03-16', '2025-09-01', 'WF2 Mar25-Sep25'),
    ('2025-09-01', '2025-09-01', '2026-04-16', 'WF3 Sep25-Apr26'),
    ('2024-10-01', '2024-10-01', '2026-04-16', 'COMBINED 18mo'),
]
results = {}
for train_end, ts, te_end, label in windows:
    print(f"\n{label}:")
    te_l = train_test_cls(train_end, ts, te_end, 'long')
    te_s = train_test_cls(train_end, ts, te_end, 'short')
    if te_l is None or te_s is None:
        print(f"  [insufficient data]")
        continue
    r = bt_cls(te_l, te_s)
    results[label] = r
    print(f"  Backtest: total={r['total']:+.1f}% Sharpe={r['sharpe']:+.2f} DD={r['dd']:+.1f}% "
          f"active_buckets={r['n_active']}/{r['n_total']}")

print()
combined = results.get('COMBINED 18mo')
if combined:
    wf_sharpes = []
    for lbl in ['WF1 Oct24-Mar25', 'WF2 Mar25-Sep25', 'WF3 Sep25-Apr26']:
        if lbl in results:
            wf_sharpes.append(results[lbl]['sharpe'])
    pos_windows = sum(1 for s in wf_sharpes if s > 0)
    print(f"Combined: total={combined['total']:+.1f}% Sharpe={combined['sharpe']:+.2f} DD={combined['dd']:+.1f}%")
    print(f"WF windows positive: {pos_windows}/{len(wf_sharpes)}")
    if pos_windows == len(wf_sharpes) and combined['sharpe'] >= 1.5 and combined['dd'] > -30:
        print(f"[SHIP] Cost-aware 4h classifier beats rank:ndcg (Sh {combined['sharpe']:.2f} vs 0.19)")
    elif combined['sharpe'] > 0.5:
        print(f"[PARTIAL] Beats rank:ndcg (Sh 0.19) but fails strict ship criteria. Mark for retune.")
    else:
        print(f"[CONCEDE] Cost-aware 4h classifier ALSO fails. Sub-day is intrinsically unprofitable on dollar bars.")
